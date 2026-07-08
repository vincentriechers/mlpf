#!/usr/bin/env python
"""Filter DELPHI pf_tree parquet files for MLPF training.

Drops "zero-input" X_gen targets: gen targets that no ygen_hit entry AND no
ygen_track entry points at (they have no detector input at all, so the network
can never reconstruct them and the object-condensation loss is diluted).
Surviving target indices are remapped so ygen_hit / ygen_track stay consistent
with the filtered X_gen. X_hit and X_track are passed through UNCHANGED.

ygen_hit_calom (unused by the training pipeline, only present in the files) is
remapped with the same index map; entries pointing at dropped targets are set
to -1.

Input and output files have the SAME schema (scalar awkward Record, one row,
100 events per file, SNAPPY compression).

Usage:
    python scripts/preprocess_delphi.py \
        --input-dir  /srv/beegfs/scratch/users/r/riechers/delphi_mlpf/digitized \
        --output-dir /srv/beegfs/scratch/users/r/riechers/delphi_mlpf/digitized_filtered \
        --workers 16

Only needs numpy/awkward/pyarrow (no torch) — runs in the `delphi_converter`
conda env.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import awkward as ak
import numpy as np

# class id -> name, mapping from |pid|; everything not listed is a charged hadron
NEUTRAL_HADRONS = {2112, 130, 310, 3122}
CLASS_NAMES = ["photon", "electron", "muon", "neutral_hadron", "charged_hadron"]


def pid_to_class(abs_pid):
    """Vectorized |pid| -> class index (order of CLASS_NAMES)."""
    cls = np.full(abs_pid.shape, 4, dtype=np.int64)  # default: charged hadron
    cls[abs_pid == 22] = 0
    cls[abs_pid == 11] = 1
    cls[abs_pid == 13] = 2
    cls[np.isin(abs_pid, list(NEUTRAL_HADRONS))] = 3
    return cls


def zero_hit_counts_per_class(pid, yh):
    """(n_zero_hit_per_class, n_per_class) for one event."""
    cls = pid_to_class(np.abs(pid).astype(np.int64))
    has_hit = np.zeros(len(pid), dtype=bool)
    ref = yh[yh >= 0]
    has_hit[ref] = True
    n_per_class = np.bincount(cls, minlength=5)
    n_zero = np.bincount(cls[~has_hit], minlength=5)
    return n_zero, n_per_class


def process_file(io_paths):
    in_path, out_path = io_paths
    arr = ak.from_parquet(in_path)
    n_events = len(arr["X_gen"])

    stats = {
        "events": n_events,
        "targets_before": 0,
        "targets_after": 0,
        "zero_hit_before": np.zeros(5, dtype=np.int64),
        "class_before": np.zeros(5, dtype=np.int64),
        "zero_hit_after": np.zeros(5, dtype=np.int64),
        "class_after": np.zeros(5, dtype=np.int64),
        "calom_dropped_refs": 0,
        "neg1_hit": 0,
    }

    new_xgen, new_yh, new_yt, new_yc = [], [], [], []
    for ev in range(n_events):
        # empty per-event lists convert to shape (0,) — force the 2D/1D shapes
        xg = ak.to_numpy(arr["X_gen"][ev]) if len(arr["X_gen"][ev]) else np.zeros((0, 21))
        yh = ak.to_numpy(arr["ygen_hit"][ev]).astype(np.int64).reshape(-1)
        yt = ak.to_numpy(arr["ygen_track"][ev]).astype(np.int64).reshape(-1)
        yc = ak.to_numpy(arr["ygen_hit_calom"][ev]).astype(np.int64).reshape(-1)
        n_gen = len(xg)

        keep = np.zeros(n_gen, dtype=bool)
        keep[yh[yh >= 0]] = True
        keep[yt[yt >= 0]] = True

        # dropped targets have no hit/track pointing at them BY CONSTRUCTION
        dropped = np.where(~keep)[0]
        assert not np.isin(yh, dropped).any(), f"{in_path} ev{ev}: dropped target referenced by ygen_hit"
        assert not np.isin(yt, dropped).any(), f"{in_path} ev{ev}: dropped target referenced by ygen_track"

        # old index -> new index; -1 stays -1
        remap = np.full(n_gen + 1, -1, dtype=np.int64)  # last slot serves index -1
        remap[:n_gen][keep] = np.arange(keep.sum())
        yh_new = remap[yh]
        yt_new = remap[yt]
        yc_new = remap[yc]
        assert (yh_new >= 0).all() == (yh >= 0).all() and ((yh_new == -1) == (yh == -1)).all(), \
            f"{in_path} ev{ev}: -1 pattern in ygen_hit changed"
        assert ((yt_new == -1) == (yt == -1)).all(), f"{in_path} ev{ev}: -1 pattern in ygen_track changed"
        stats["calom_dropped_refs"] += int(((yc_new == -1) & (yc >= 0)).sum())
        stats["neg1_hit"] += int((yh == -1).sum())

        # bookkeeping
        z_b, c_b = zero_hit_counts_per_class(xg[:, 0], yh)
        z_a, c_a = zero_hit_counts_per_class(xg[keep][:, 0], yh_new)
        stats["targets_before"] += n_gen
        stats["targets_after"] += int(keep.sum())
        stats["zero_hit_before"] += z_b
        stats["class_before"] += c_b
        stats["zero_hit_after"] += z_a
        stats["class_after"] += c_a

        new_xgen.append(xg[keep])
        new_yh.append(yh_new)
        new_yt.append(yt_new)
        new_yc.append(yc_new)

    record = ak.Record(
        {
            "X_track": arr["X_track"],  # untouched
            "X_hit": arr["X_hit"],      # untouched
            "X_gen": ak.Array(new_xgen),
            "ygen_track": ak.Array(new_yt),
            "ygen_hit": ak.Array(new_yh),
            "ygen_hit_calom": ak.Array(new_yc),
        }
    )
    ak.to_parquet(record, out_path, compression="SNAPPY")
    return stats


def verify_file(io_paths):
    """Re-read one input/output pair and check the invariants end-to-end."""
    in_path, out_path = io_paths
    a = ak.from_parquet(in_path)
    b = ak.from_parquet(out_path)
    n_events = len(a["X_gen"])
    assert len(b["X_gen"]) == n_events
    for ev in range(n_events):
        xh_a = ak.to_numpy(a["X_hit"][ev])
        xh_b = ak.to_numpy(b["X_hit"][ev])
        assert xh_a.shape == xh_b.shape and (xh_a == xh_b).all(), f"{out_path} ev{ev}: X_hit changed"
        xt_a = ak.to_numpy(a["X_track"][ev]) if len(a["X_track"][ev]) else np.zeros((0, 27))
        xt_b = ak.to_numpy(b["X_track"][ev]) if len(b["X_track"][ev]) else np.zeros((0, 27))
        assert xt_a.shape == xt_b.shape and (xt_a == xt_b).all(), f"{out_path} ev{ev}: X_track changed"

        yh_a = ak.to_numpy(a["ygen_hit"][ev]).astype(np.int64).reshape(-1)
        yh_b = ak.to_numpy(b["ygen_hit"][ev]).astype(np.int64).reshape(-1)
        assert (yh_a == -1).sum() == (yh_b == -1).sum(), f"{out_path} ev{ev}: -1 count changed"
        # surviving targets keep identical feature rows and labels stay aligned
        xg_a = ak.to_numpy(a["X_gen"][ev]) if len(a["X_gen"][ev]) else np.zeros((0, 21))
        xg_b = ak.to_numpy(b["X_gen"][ev]) if len(b["X_gen"][ev]) else np.zeros((0, 21))
        lab = yh_b >= 0
        assert (np.abs(xg_b[yh_b[lab], 8] - xg_a[yh_a[lab], 8]) < 1e-12).all(), \
            f"{out_path} ev{ev}: hit->target energy mismatch after remap"
        yt_a = ak.to_numpy(a["ygen_track"][ev]).astype(np.int64)
        yt_b = ak.to_numpy(b["ygen_track"][ev]).astype(np.int64)
        labt = yt_b >= 0
        assert (np.abs(xg_b[yt_b[labt], 8] - xg_a[yt_a[labt], 8]) < 1e-12).all(), \
            f"{out_path} ev{ev}: track->target energy mismatch after remap"
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-files", type=int, default=None, help="only process the first N files (for testing)")
    p.add_argument("--verify-sample", type=int, default=20, help="re-read this many random output files and re-check invariants")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "pf_tree_*.parquet")))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"no pf_tree_*.parquet in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    pairs = [(f, os.path.join(args.output_dir, os.path.basename(f))) for f in files]

    print(f"processing {len(pairs)} files with {args.workers} workers")
    with Pool(args.workers) as pool:
        all_stats = pool.map(process_file, pairs, chunksize=8)

    # ---- aggregate ----
    tot = {k: sum(s[k] for s in all_stats) for k in ("events", "targets_before", "targets_after", "calom_dropped_refs", "neg1_hit")}
    zb = np.sum([s["zero_hit_before"] for s in all_stats], axis=0)
    cb = np.sum([s["class_before"] for s in all_stats], axis=0)
    za = np.sum([s["zero_hit_after"] for s in all_stats], axis=0)
    ca = np.sum([s["class_after"] for s in all_stats], axis=0)

    lines = []
    lines.append("==== preprocess_delphi report ====")
    lines.append(f"files: {len(pairs)}   events: {tot['events']}")
    lines.append(f"targets/event: before {tot['targets_before']/tot['events']:.2f}  ->  after {tot['targets_after']/tot['events']:.2f}")
    lines.append(f"targets dropped: {tot['targets_before']-tot['targets_after']} ({100*(1-tot['targets_after']/tot['targets_before']):.1f}%)")
    lines.append(f"ygen_hit -1 entries (unchanged by construction, asserted): {tot['neg1_hit']}")
    lines.append(f"ygen_hit_calom refs to dropped targets set to -1: {tot['calom_dropped_refs']}")
    lines.append("zero-hit target fraction per class (before -> after):")
    for i, name in enumerate(CLASS_NAMES):
        fb = zb[i] / cb[i] if cb[i] else 0.0
        fa = za[i] / ca[i] if ca[i] else 0.0
        lines.append(f"  {name:16s} {100*fb:5.1f}% ({zb[i]}/{cb[i]})  ->  {100*fa:5.1f}% ({za[i]}/{ca[i]})")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.output_dir, "preprocess_report.txt"), "w") as f:
        f.write(report + "\n")

    # ---- verification pass on a random sample of written files ----
    if args.verify_sample:
        rng = np.random.default_rng(0)
        sample = [pairs[i] for i in rng.choice(len(pairs), size=min(args.verify_sample, len(pairs)), replace=False)]
        print(f"verifying {len(sample)} random output files (byte-compare X_hit/X_track, label consistency)...")
        with Pool(min(args.workers, len(sample))) as pool:
            pool.map(verify_file, sample)
        print("verification OK")


if __name__ == "__main__":
    main()
