#!/usr/bin/env python3
"""Read an OFFLINE W&B run's logged history without syncing it.

BSC compute nodes have no outbound internet, so every run here uses
WANDB_MODE=offline and the loss exists only inside `<run>/run-<id>.wandb`.
That is exactly why a diverged Attn-IPA run once went unnoticed: the progress
bar looked fine and nobody could see the loss. This decodes the datastore in
place, so a RUNNING job can be inspected without waiting for it to finish and
without a network round trip.

  python3 scripts/bsc/read_offline_wandb.py <wandb-dir-or-run-dir> [-k loss -k val_loss] [-n 12]

With no -k it lists the available metric keys and shows the last value of each.
"""
from __future__ import annotations
import argparse, glob, json, os, sys


def read_run(run_dir):
    """-> (display_name, [(step, {key: value}), ...])"""
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    wf = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wf:
        return None, []
    ds = datastore.DataStore()
    try:
        ds.open_for_scan(wf[0])
    except Exception as e:
        return f"<unreadable: {e}>", []

    name, hist = None, []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break                      # truncated tail on a live run: normal
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        kind = rec.WhichOneof("record_type")
        if kind == "run" and not name:
            name = rec.run.display_name or rec.run.run_id
        elif kind == "history":
            row, step = {}, None
            for it in rec.history.item:
                try:
                    v = json.loads(it.value_json)
                except Exception:
                    continue
                k = it.key or ".".join(it.nested_key)
                if k == "_step":
                    step = v
                elif isinstance(v, (int, float)):
                    row[k] = v
            if row:
                hist.append((step, row))
    return name, hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("-k", "--key", action="append", default=[])
    ap.add_argument("-n", "--n-rows", type=int, default=10)
    a = ap.parse_args()

    runs = ([a.path] if glob.glob(os.path.join(a.path, "*.wandb"))
            else sorted(glob.glob(os.path.join(a.path, "offline-run-*"))))
    if not runs:
        sys.exit(f"no offline runs under {a.path}")

    for rd in runs:
        name, hist = read_run(rd)
        if not hist:
            continue
        print(f"\n=== {name or '?'}   [{os.path.basename(rd)}]   {len(hist)} history rows ===")
        keys = a.key or sorted({k for _, r in hist for k in r})
        if not a.key:
            print("  metrics:", ", ".join(keys))
        # trend: first vs last, and the last n rows
        for k in keys:
            pts = [(s, r[k]) for s, r in hist if k in r]
            if not pts:
                continue
            first, last = pts[0][1], pts[-1][1]
            arrow = "->" if len(pts) > 1 else "  "
            trend = ""
            if len(pts) > 1 and isinstance(first, (int, float)) and first:
                trend = f"  ({100*(last-first)/abs(first):+.1f}%)"
            print(f"  {k:28} n={len(pts):<6} first={first:<12.5g} {arrow} last={last:<12.5g}{trend}")
        if a.key:
            print(f"\n  last {a.n_rows} rows:")
            sel = [(s, r) for s, r in hist if any(k in r for k in a.key)][-a.n_rows:]
            for s, r in sel:
                vals = "  ".join(f"{k}={r[k]:.5g}" for k in a.key if k in r)
                print(f"    step {s:<8} {vals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
