#!/usr/bin/env python3
"""DELPHI-native plots for a full-pipeline eval dataframe (clustering + EC + PID).

Replaces src.evaluation.full_evaluation for DELPHI: that plotter is built
around the CLD o3_v01-vs-o2_v05 detector comparison (duplicate curves, ratio
panels, Pandora slots, event-energy axis windows tuned to CLD response).
Here there is one model, so instead of model-vs-model the curves distinguish
*target categories* — the same hypothetical cuts as the clustering cut scan:

    no cut / CH must have track / >=1 hit / >=3 hits / >=5 hits

Per-target calo-hit counts and track flags come from the evaluated parquet
files via the sorted-energy event fingerprint (see plot_cut_scan_delphi.py,
whose helpers this script imports).  Axis windows are derived from the data.

Outputs (PDF+PNG) in --out:
    eff_clustering_cuts   clustering efficiency vs true E, per class x cut
    eff_pid_cuts          matched AND correct 4-class PID vs true E
    fake_rate             fakes/event vs calibrated E, per predicted class
    energy_response_cuts  median calibrated/true vs E (band = IQR/2), per cut
    event_energy          per-event sum(pred calibrated)/sum(true), auto range
    full_eval_cuts_summary.csv

Usage:
    python training/plot_full_eval_delphi.py \
        [--df <eval_full .pt>] [--val-dir <validation_filtered>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cut_scan_delphi import category, per_target_from_parquets  # noqa: E402

DEF_DF = ("/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/models/"
          "delphi_props_smoketest/showers_df_evaluation/"
          "eval_full_delphi_firstlook.pkl0_0_None.pt")
DEF_VAL = "/srv/beegfs/scratch/users/r/riechers/delphi_mlpf/validation_filtered"
DEF_OUT = ("/home/users/r/riechers/delphi_converter/analysis/plots/"
           "full_eval_firstlook_delphi")

# 4-class ids used by --PID-4-class (src/utils/pid_conversion.py)
CLASS_ID = {"CH": 1, "photon": 3, "NH": 2}
PANELS = ["CH", "photon", "NH"]
KEY_CUTS = ["no cut", "CH must have track", "≥1 hit", "≥3 hits", "≥5 hits"]
E_BINS = np.array([0.1, 0.3, 0.5, 1, 2, 5, 10, 50])


def join_hit_counts(truth: pd.DataFrame, val_dir: str) -> pd.DataFrame:
    """Attach per-target n_hits from the parquets via the event fingerprint
    (n_targets, sum E, max E) in integer MeV; identical to the cut scan."""
    events = per_target_from_parquets(val_dir)

    def mev(a):
        return np.round(np.asarray(a, dtype=np.float64) * 1000).astype(np.int64)

    pq_key, n_amb = {}, 0
    for j, e in enumerate(events):
        if e is None:
            continue
        E = mev(e[0])
        k = (len(E), int(E.sum()), int(E.max()))
        if k in pq_key:
            pq_key[k] = None
            n_amb += 1
        else:
            pq_key[k] = j

    truth = truth.copy()
    truth["number_batch"] = truth["number_batch"].astype(np.int64)
    truth["E_mev"] = mev(truth["true_showers_E"].values)
    ev = truth.groupby("number_batch")["E_mev"].agg(["size", "sum", "max"])
    batch_to_pq = {}
    for nb, (sz, sm, mx) in ev.iterrows():
        j = pq_key.get((int(sz), int(sm), int(mx)))
        if j is not None:
            batch_to_pq[int(nb)] = j
    print(f"event fingerprint match: {len(batch_to_pq)}/{len(ev)} "
          f"(ambiguous pq keys: {n_amb})")

    truth = truth[truth["number_batch"].isin(batch_to_pq)].copy()
    truth["pq_event"] = truth["number_batch"].map(batch_to_pq)
    used = sorted(set(batch_to_pq.values()))
    pq = pd.DataFrame({
        "pq_event": np.concatenate(
            [np.full(len(events[j][0]), j, dtype=np.int64) for j in used]),
        "E_mev": np.concatenate([mev(events[j][0]) for j in used]),
        "n_hits": np.concatenate([events[j][1] for j in used]),
    })
    for frame in (truth, pq):
        frame.sort_values(["pq_event", "E_mev"], inplace=True)
        frame["dup"] = frame.groupby(["pq_event", "E_mev"]).cumcount()
    truth = truth.merge(pq, on=["pq_event", "E_mev", "dup"], how="left")
    ok = truth["n_hits"].notna()
    print(f"matched target rows: {ok.mean():.4f}")
    truth = truth[ok].copy()
    truth["n_hits"] = truth["n_hits"].astype(int)
    return truth


def build_cuts(truth: pd.DataFrame) -> dict[str, np.ndarray]:
    has_trk = (truth["is_track_in_MC"] >= 1).values
    is_ch = (truth["cat"] == "CH").values
    nh = truth["n_hits"].values
    return {
        "no cut": np.ones(len(truth), bool),
        "CH must have track": ~is_ch | has_trk,
        "≥1 hit": nh >= 1,
        "≥3 hits": nh >= 3,
        "≥5 hits": nh >= 5,
    }


def dedup_cuts(cuts, panel_mask):
    """Cuts selecting identical subsets of this panel are drawn once, with
    the equivalent names merged into the label (as in the cut scan)."""
    drawn: dict[bytes, str] = {}
    for name in KEY_CUTS:
        key = (cuts[name] & panel_mask).tobytes()
        if key in drawn:
            drawn[key] += f" = {name}"
        else:
            drawn[key] = name
    return [(np.frombuffer(k, dtype=bool), lab) for k, lab in drawn.items()]


def binned(e, sel, values, reducer, min_n=50):
    out = []
    for lo, hi in zip(E_BINS[:-1], E_BINS[1:]):
        m = sel & (e >= lo) & (e < hi)
        out.append(reducer(values[m]) if m.sum() > min_n else np.nan)
    return np.array(out)


def cut_curve_figure(truth, cuts, numerator, ylabel, title, path):
    """1x3 panels (CH/photon/NH); one curve per unique cut selection; the
    plotted quantity is mean(numerator) among truth targets passing the cut."""
    ctr = 0.5 * (E_BINS[:-1] + E_BINS[1:])
    e = truth["true_showers_E"].values
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, cat_name in zip(axes, PANELS):
        mc = (truth["cat"] == cat_name).values
        for m, label in dedup_cuts(cuts, mc):
            ax.plot(ctr, binned(e, m, numerator, np.mean), "o-", label=label)
        ax.set_xscale("log")
        ax.set_xlabel("true E [GeV]")
        ax.set_title(cat_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", default=DEF_DF)
    ap.add_argument("--val-dir", default=DEF_VAL)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(args.df)
    truth = df[df["pid"].notna()].copy()
    fakes = df[df["pid"].isna()].copy()
    n_events = df["number_batch"].nunique()
    truth["cat"] = category(truth["pid"].values)
    truth["found"] = truth["pred_showers_E"].notna()
    class_ids = truth["cat"].map(CLASS_ID)
    truth["pid_ok"] = truth["found"] & (truth["pred_pid_matched"] == class_ids)

    truth = join_hit_counts(truth, args.val_dir)
    cuts = build_cuts(truth)

    cut_curve_figure(
        truth, cuts, truth["found"].values.astype(float),
        "clustering efficiency",
        "DELPHI clustering efficiency vs E under target cuts (plot-level)",
        out / "eff_clustering_cuts")
    cut_curve_figure(
        truth, cuts, truth["pid_ok"].values.astype(float),
        "matched × correct PID",
        "DELPHI clustering + 4-class PID efficiency vs E under target cuts",
        out / "eff_pid_cuts")

    # ---- energy response: median calibrated/true per E bin, band = IQR/2 ----
    ctr = 0.5 * (E_BINS[:-1] + E_BINS[1:])
    e = truth["true_showers_E"].values
    ratio = (truth["calibrated_E"] / truth["true_showers_E"]).values
    ratio = np.where(truth["found"].values, ratio, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, cat_name in zip(axes, PANELS):
        mc = (truth["cat"] == cat_name).values & truth["found"].values
        for m, label in dedup_cuts(cuts, mc):
            med = binned(e, m, ratio, np.nanmedian)
            q25 = binned(e, m, ratio, lambda x: np.nanpercentile(x, 25))
            q75 = binned(e, m, ratio, lambda x: np.nanpercentile(x, 75))
            (line,) = ax.plot(ctr, med, "o-", label=label)
            ax.fill_between(ctr, q25, q75, color=line.get_color(), alpha=0.15)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("true E [GeV]")
        ax.set_title(cat_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("calibrated E / true E  (median, IQR band)")
    fig.suptitle("DELPHI energy response vs E under target cuts (matched showers)")
    fig.tight_layout()
    fig.savefig(out / "energy_response_cuts.pdf")
    fig.savefig(out / "energy_response_cuts.png", dpi=140)
    plt.close(fig)

    # ---- fakes: rate per event and calibrated-E spectrum, per pred class ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fe = fakes["calibrated_E"].values
    for cat_name in PANELS:
        m = (fakes["pred_pid_matched"] == CLASS_ID[cat_name]).values
        per_ev = [np.sum(m & (fe >= lo) & (fe < hi)) / n_events
                  for lo, hi in zip(E_BINS[:-1], E_BINS[1:])]
        a1.plot(ctr, per_ev, "o-", label=f"pred {cat_name}")
    a1.plot(ctr, [np.sum((fe >= lo) & (fe < hi)) / n_events
                  for lo, hi in zip(E_BINS[:-1], E_BINS[1:])],
            "k.--", label="all fakes")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("calibrated E [GeV]")
    a1.set_ylabel("fake clusters / event / bin")
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8)
    hi = np.nanpercentile(fe, 99.5) if len(fe) else 1.0
    a2.hist(np.clip(fe, 0, hi), bins=60, histtype="step", color="k")
    a2.set_xlabel("fake calibrated E [GeV]")
    a2.set_ylabel("fakes")
    a2.grid(alpha=0.3)
    fig.suptitle(f"DELPHI fake clusters ({len(fakes)} in {n_events} events "
                 f"= {len(fakes) / n_events:.2f}/event)")
    fig.tight_layout()
    fig.savefig(out / "fake_rate.pdf")
    fig.savefig(out / "fake_rate.png", dpi=140)
    plt.close(fig)

    # ---- event-level energy: sum(pred calibrated incl. fakes)/sum(true) ----
    true_sum = truth.groupby("number_batch")["true_showers_E"].sum()
    pred_truth = truth[truth["found"]].groupby("number_batch")["calibrated_E"].sum()
    pred_fake = fakes.groupby("number_batch")["calibrated_E"].sum()
    pred_sum = pred_truth.add(pred_fake, fill_value=0.0)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    panels = [("inclusive", None)] + [(c, c) for c in PANELS]
    for ax, (label, cat_name) in zip(axes, panels):
        if cat_name is None:
            r = (pred_sum / true_sum).dropna().values
        else:
            ts = truth[truth["cat"] == cat_name].groupby("number_batch")[
                "true_showers_E"].sum()
            pt = truth[truth["found"] & (truth["cat"] == cat_name)].groupby(
                "number_batch")["calibrated_E"].sum()
            pf = fakes[fakes["pred_pid_matched"] == CLASS_ID[cat_name]].groupby(
                "number_batch")["calibrated_E"].sum()
            r = (pt.add(pf, fill_value=0.0) / ts).dropna().values
        # axis window from the data, not from CLD-tuned constants
        lo, hi = np.nanpercentile(r, [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        ax.hist(r, bins=np.linspace(lo - pad, hi + pad, 80), histtype="step")
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.axvline(np.median(r), color="C3", lw=1.0,
                   label=f"median {np.median(r):.3f}")
        ax.set_xlabel("Σ pred calibrated E / Σ true E")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("events")
    fig.suptitle("DELPHI per-event energy sum (matched + fakes, per predicted class)")
    fig.tight_layout()
    fig.savefig(out / "event_energy.pdf")
    fig.savefig(out / "event_energy.png", dpi=140)
    plt.close(fig)

    # ---- summary table ----
    rows = []
    for name, m in cuts.items():
        r = {"cut": name, "surviving_frac": m.mean(),
             "eff_all": truth["found"][m].mean(),
             "pid_eff_all": truth["pid_ok"][m].mean()}
        for cat_name in PANELS:
            mc = (truth["cat"] == cat_name).values
            sel = m & mc
            r[f"surv_{cat_name}"] = m[mc].mean()
            r[f"eff_{cat_name}"] = truth["found"][sel].mean()
            r[f"pid_eff_{cat_name}"] = truth["pid_ok"][sel].mean()
            med = np.nanmedian(np.where(sel & truth["found"].values, ratio, np.nan))
            r[f"response_{cat_name}"] = med
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out / "full_eval_cuts_summary.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote plots + CSV to {out}")


if __name__ == "__main__":
    main()
