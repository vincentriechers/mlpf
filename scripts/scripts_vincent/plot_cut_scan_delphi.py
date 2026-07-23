#!/usr/bin/env python3
"""Target-definition cut scan on an existing clustering-eval dataframe.

The production pipeline applies NO minimum-calo-hit-count cut and NO
"charged targets must have a track" cut (the only target requirement is
>10 MeV assigned calo energy OR the daughter-walk keep;
docs/TRACKONLY_INVESTIGATION.md).  This script shows, from an already-run
eval, what each hypothetical cut would do: which fraction of truth targets
survives and what the clustering efficiency is on the survivors.

Rigorous numbers need a retraining with the cut applied to the target set;
this is the cheap plotting-level preview.

Per-target calo-hit counts are not stored in the showers dataframe, so they
are re-derived from the evaluated parquet files and joined per event via a
sorted-energy fingerprint (events whose fingerprints disagree are dropped
and counted).

Usage:
    python training/plot_cut_scan_delphi.py \
        [--df <showers .pt>] [--val-dir <validation_filtered>] [--out <dir>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEF_DF = ("/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/models/"
          "delphi_500k_sB02_trk005/showers_df_evaluation/"
          "eval_clustering_delphi_val200k.pkl0_0_None.pt")
DEF_VAL = "/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/validation_filtered"
DEF_OUT = ("/home/users/r/riechers/delphi_converter/analysis/plots/"
           "clustering_val200k_cuts")

# 4-class grouping as in the standard plotter: CH / NH / photon (e, mu kept
# separately since the cuts affect them differently)
CH_PIDS = {211, 321, 2212, 3112, 3222, 3224, 3312, 3334}
NH_PIDS = {130, 310, 2112, 3122, 3212, 3322, 111, 12, 14, 16}


def category(pid: np.ndarray) -> np.ndarray:
    a = np.abs(pid.astype(int))
    out = np.full(len(a), "other", dtype=object)
    out[np.isin(a, list(CH_PIDS))] = "CH"
    out[np.isin(a, list(NH_PIDS))] = "NH"
    out[a == 22] = "photon"
    out[a == 11] = "electron"
    out[a == 13] = "muon"
    return out


def _per_file(path: str):
    rec = ak.from_parquet(path)
    xg, yh, yt = rec["X_gen"], rec["ygen_hit"], rec["ygen_track"]
    out = []
    for i in range(len(xg)):
        gen = np.asarray(xg[i])
        if gen.size == 0:
            out.append(None)
            continue
        n = len(gen)
        h = np.asarray(yh[i], dtype=np.int64)
        t = np.asarray(yt[i], dtype=np.int64)
        nh = np.bincount(h[(h >= 0) & (h < n)], minlength=n)
        nt = np.bincount(t[(t >= 0) & (t < n)], minlength=n)
        out.append((gen[:, 8], nh, nt > 0, gen[:, 2]))
    return out


def per_target_from_parquets(val_dir: str, workers: int = 8):
    """event-ordered list of (E array, n_hits array, has_trk array, charge)."""
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(str(f) for f in Path(val_dir).glob("pf_tree_*.parquet"))
    events = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_per_file, files, chunksize=4):
            events.extend(out)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", default=DEF_DF)
    ap.add_argument("--val-dir", default=DEF_VAL)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.df)
    truth = df[~df["reco_showers_E"].isna()].copy()
    truth["found"] = ~truth["pred_showers_E"].isna()
    truth["cat"] = category(truth["pid"].values)

    events = per_target_from_parquets(args.val_dir)
    print(f"parquet events: {len(events)}  df events: "
          f"{truth['number_batch'].nunique()}")

    # The eval dataloader ran with several workers, so number_batch order is
    # NOT parquet event order.  Match events content-wise instead: fingerprint
    # = (n_targets, sum E, max E) in integer MeV (float32 df values and
    # float64 parquet values round-trip to the same integers).  Only events
    # whose fingerprint is unique on BOTH sides are used.
    def mev(a):
        return np.round(np.asarray(a, dtype=np.float64) * 1000).astype(np.int64)

    pq_key = {}
    n_amb_pq = 0
    for j, e in enumerate(events):
        if e is None:
            continue
        E = mev(e[0])
        k = (len(E), int(E.sum()), int(E.max()))
        if k in pq_key:
            pq_key[k] = None  # ambiguous
            n_amb_pq += 1
        else:
            pq_key[k] = j

    truth = truth.copy()
    truth["number_batch"] = truth["number_batch"].astype(np.int64)
    truth["E_mev"] = mev(truth["true_showers_E"].values)
    ev_stats = truth.groupby("number_batch")["E_mev"].agg(["size", "sum", "max"])
    n_match, n_miss = 0, 0
    batch_to_pq = {}
    for nb, (sz, sm, mx) in ev_stats.iterrows():
        j = pq_key.get((int(sz), int(sm), int(mx)))
        if j is None:
            n_miss += 1
        else:
            batch_to_pq[int(nb)] = j
            n_match += 1
    print(f"event fingerprint match: {n_match}/{len(ev_stats)} "
          f"(ambiguous pq keys: {n_amb_pq}, unmatched df events: {n_miss})")

    truth = truth[truth["number_batch"].isin(batch_to_pq)].copy()
    truth["pq_event"] = truth["number_batch"].map(batch_to_pq)
    pq = pd.DataFrame({
        "pq_event": np.concatenate(
            [np.full(len(events[j][0]), j, dtype=np.int64)
             for j in sorted(set(batch_to_pq.values()))]),
        "E_mev": np.concatenate(
            [mev(events[j][0]) for j in sorted(set(batch_to_pq.values()))]),
        "n_hits": np.concatenate(
            [events[j][1] for j in sorted(set(batch_to_pq.values()))]),
        "pq_has_trk": np.concatenate(
            [events[j][2] for j in sorted(set(batch_to_pq.values()))]),
    })
    truth = truth.sort_values(["pq_event", "E_mev"])
    truth["dup"] = truth.groupby(["pq_event", "E_mev"]).cumcount()
    pq = pq.sort_values(["pq_event", "E_mev"])
    pq["dup"] = pq.groupby(["pq_event", "E_mev"]).cumcount()
    truth = truth.merge(pq, on=["pq_event", "E_mev", "dup"], how="left")
    matched = truth["n_hits"].notna()
    print(f"matched target rows: {matched.mean():.4f}")
    truth = truth[matched].copy()
    truth["n_hits"] = truth["n_hits"].astype(int)

    has_trk = truth["is_track_in_MC"] == 1
    is_ch = truth["cat"] == "CH"
    cuts = {
        "no cut": np.ones(len(truth), bool),
        "≥1 hit": (truth["n_hits"] >= 1).values,
        "≥2 hits": (truth["n_hits"] >= 2).values,
        "≥3 hits": (truth["n_hits"] >= 3).values,
        "≥5 hits": (truth["n_hits"] >= 5).values,
        "≥10 hits": (truth["n_hits"] >= 10).values,
        "CH must have track": (~is_ch | has_trk).values,
        "≥1 hit & CH-track": ((truth["n_hits"] >= 1) & (~is_ch | has_trk)).values,
        "≥3 hits & CH-track": ((truth["n_hits"] >= 3) & (~is_ch | has_trk)).values,
    }

    cats = ["CH", "photon", "NH", "electron", "muon"]
    rows = []
    for cname, m in cuts.items():
        r = {"cut": cname,
             "surviving_frac_all": m.mean(),
             "eff_all": truth["found"][m].mean()}
        for c in cats:
            mc = (truth["cat"] == c).values
            r[f"surv_{c}"] = m[mc].mean() if mc.any() else np.nan
            sel = m & mc
            r[f"eff_{c}"] = truth["found"][sel].mean() if sel.any() else np.nan
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out / "cut_scan_summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # efficiency vs E for a few key cuts
    bins = np.array([0.1, 0.3, 0.5, 1, 2, 5, 10, 50])
    ctr = 0.5 * (bins[:-1] + bins[1:])
    key_cuts = ["no cut", "CH must have track", "≥1 hit", "≥3 hits",
                "≥5 hits"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, c in zip(axes, ["CH", "photon", "NH"]):
        mc = (truth["cat"] == c).values
        # cuts that select identical subsets of this category would overdraw
        # each other (e.g. every photon has >=1 hit, and CH-track cuts are
        # no-ops for neutrals) — draw each unique selection once, with all
        # equivalent cut names merged into the label
        drawn: dict[bytes, str] = {}
        for cname in key_cuts:
            key = (cuts[cname] & mc).tobytes()
            if key in drawn:
                drawn[key] += f" = {cname}"
            else:
                drawn[key] = cname
        for key, label in drawn.items():
            m = np.frombuffer(key, dtype=bool)
            e = truth["true_showers_E"].values
            eff = [truth["found"][m & (e >= lo) & (e < hi)].mean()
                   if (m & (e >= lo) & (e < hi)).sum() > 50 else np.nan
                   for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(ctr, eff, "o-", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("true E [GeV]")
        ax.set_title(c)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("clustering efficiency")
    fig.suptitle("Efficiency vs E under hypothetical target cuts "
                 "(plot-level; rigorous version needs retraining with the cut)")
    fig.tight_layout()
    fig.savefig(out / "cut_scan_eff_vs_E.pdf")
    fig.savefig(out / "cut_scan_eff_vs_E.png", dpi=140)

    # survival + efficiency bars
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(cuts))
    for c in ["CH", "photon", "NH"]:
        a1.plot(x, table[f"surv_{c}"], "o-", label=c)
        a2.plot(x, table[f"eff_{c}"], "o-", label=c)
    for ax, ttl in ((a1, "surviving target fraction"),
                    (a2, "efficiency on survivors")):
        ax.set_xticks(x)
        ax.set_xticklabels(list(cuts), rotation=35, ha="right", fontsize=8)
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(out / "cut_scan_survival.pdf")
    fig2.savefig(out / "cut_scan_survival.png", dpi=140)
    print(f"wrote plots + CSV to {out}")


if __name__ == "__main__":
    main()
