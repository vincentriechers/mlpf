"""DELPHI adaptation of the ARC/05 full-evaluation plots.

This reuses src.evaluation.full_evaluation (style, curve computation, the
per-energy PID confusion grid, event energy/mass comparison, resolution) and
adapts the comparison axis to DELPHI's needs: there is one model and no
Pandora baseline, so instead of geometry/method the overlaid curves
distinguish *target categories* — the hypothetical target cuts from the
clustering cut scan:

    no cut / CH must have track / >=1 hit / >=3 hits / >=5 hits

Differences from the CLD driver:
  * single --mlpf input; no ratio panels; no Pandora slots
  * per-target calo-hit counts joined from the evaluated parquets via the
    sorted-energy event fingerprint (--val-dir), as in the converter's
    plot_cut_scan_delphi.py
  * cuts that select an identical subset of a panel's particles are drawn
    once with the equivalent names merged into the label
  * event energy/mass axis windows derived from the data, not CLD constants
  * confusion grid: single dataset, DELPHI energy bins (0.1-1, 1-10, 10-50)

Usage (inside the gatr container, mlpf repo root):
    python -m src.evaluation.full_evaluation_delphi \
        --mlpf <eval_full pickle or glob> \
        --val-dir <validation_filtered> \
        --output-dir <dir>
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import src.evaluation.full_evaluation as fe

DELPHI_LABEL = "HitPF"
KEY_CUTS = ["no cut", "CH must have track", "≥1 hit", "≥3 hits", "≥5 hits"]
CUT_STYLES = {
    "no cut": {"color": "#0F4C5C", "marker": "o"},
    "CH must have track": {"color": "#E36414", "marker": "^"},
    "≥1 hit": {"color": "#4F84C4", "marker": "s"},
    "≥3 hits": {"color": "#C7522A", "marker": "D"},
    "≥5 hits": {"color": "#9C6644", "marker": "v"},
    DELPHI_LABEL: {"color": "#E36414", "marker": "o"},
}

# DELPHI Z->qqbar at 91 GeV populates lower energies than the CLD samples
DELPHI_XLIMS = {"charged_hadrons": (0.1, 50.0), "photons": (0.1, 50.0),
                "neutral_hadrons": (0.5, 50.0)}
DELPHI_PARTICLES = deepcopy(fe.PARTICLES)
for _p in DELPHI_PARTICLES:
    _p["xlim"] = DELPHI_XLIMS.get(_p["key"], _p["xlim"])
DELPHI_RESOLUTION_PARTICLES = deepcopy(fe.RESOLUTION_PARTICLES)
for _p in DELPHI_RESOLUTION_PARTICLES:
    _p["xlim"] = DELPHI_XLIMS.get(_p["key"], _p["xlim"])
DELPHI_CONFUSION_ENERGY_BINS = [(0.1, 1.0), (1.0, 10.0), (10.0, 50.0)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="DELPHI full-evaluation plots with target-cut overlays.")
    parser.add_argument("--mlpf", required=True,
                        help="DELPHI HitPF evaluation pickle or quoted glob.")
    parser.add_argument("--val-dir", required=True,
                        help="Directory with the evaluated pf_tree parquets "
                             "(per-target hit counts for the cut overlays).")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DELPHI style hooks (patched into fe so its plot functions pick them up)
# ---------------------------------------------------------------------------
def build_style_delphi(label, confusion=False):
    base = str(label).split(" = ")[0]
    style = CUT_STYLES.get(base, {"color": "black", "marker": "o"})
    return {"color": style["color"], "marker": style["marker"],
            "linestyle": "-", "markersize": 9, "alpha": 0.14}


def comparison_sort_key_delphi(label):
    base = str(label).split(" = ")[0]
    return (KEY_CUTS.index(base) if base in KEY_CUTS else len(KEY_CUTS), str(label))


def style_metric_axis_delphi(ax, particle, ylabel, metric_key, all_y, logy):
    # fe.style_metric_axis with the legend renamed from Geometry/method
    ax.set_title(particle["label"])
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(*particle["xlim"])
    ax.set_xscale("log")
    fe.apply_log_energy_ticks(ax, particle["xlim"])
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    fe.set_metric_ylim(ax, metric_key, all_y)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=17, title="Target cut", title_fontsize=16)


def apply_fe_patches():
    fe.build_style = build_style_delphi
    fe.comparison_sort_key = comparison_sort_key_delphi


# ---------------------------------------------------------------------------
# per-target hit counts from the parquets (fingerprint join, as in the
# converter's plot_cut_scan_delphi.py)
# ---------------------------------------------------------------------------
def _per_file_targets(path):
    rec = ak.from_parquet(path)
    xg, yh = rec["X_gen"], rec["ygen_hit"]
    out = []
    for i in range(len(xg)):
        gen = np.asarray(xg[i])
        if gen.size == 0:
            out.append(None)
            continue
        n = len(gen)
        h = np.asarray(yh[i], dtype=np.int64)
        out.append((gen[:, 8], np.bincount(h[(h >= 0) & (h < n)], minlength=n)))
    return out


def attach_target_info(frame, val_dir, workers=8):
    """Add an n_hits column to the truth rows via the per-event fingerprint
    (n_targets, sum E, max E) in integer MeV.  Fake rows keep n_hits NaN."""
    files = sorted(str(f) for f in Path(val_dir).glob("pf_tree_*.parquet"))
    events = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for out in ex.map(_per_file_targets, files, chunksize=4):
            events.extend(out)

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

    frame = frame.copy().reset_index(drop=True)
    truth = frame[frame["pid"].notna()].copy()
    truth["orig_idx"] = truth.index
    truth["number_batch"] = truth["number_batch"].astype(np.int64)
    truth["E_mev"] = mev(truth["true_showers_E"].values)
    ev = truth.groupby("number_batch")["E_mev"].agg(["size", "sum", "max"])
    batch_to_pq = {}
    for nb, (sz, sm, mx) in ev.iterrows():
        j = pq_key.get((int(sz), int(sm), int(mx)))
        if j is not None:
            batch_to_pq[int(nb)] = j
    print(f"[{DELPHI_LABEL}] event fingerprint match: "
          f"{len(batch_to_pq)}/{len(ev)} (ambiguous pq keys: {n_amb})")

    truth = truth[truth["number_batch"].isin(batch_to_pq)].copy()
    truth["pq_event"] = truth["number_batch"].map(batch_to_pq)
    used = sorted(set(batch_to_pq.values()))
    pq = pd.DataFrame({
        "pq_event": np.concatenate(
            [np.full(len(events[j][0]), j, dtype=np.int64) for j in used]),
        "E_mev": np.concatenate([mev(events[j][0]) for j in used]),
        "n_hits_pq": np.concatenate([events[j][1] for j in used]),
    })
    for f in (truth, pq):
        f.sort_values(["pq_event", "E_mev"], inplace=True)
        f["dup"] = f.groupby(["pq_event", "E_mev"]).cumcount()
    joined = truth.merge(pq, on=["pq_event", "E_mev", "dup"], how="left")
    print(f"[{DELPHI_LABEL}] matched target rows: "
          f"{joined['n_hits_pq'].notna().mean():.4f}")

    frame["n_hits"] = np.nan
    frame.loc[joined["orig_idx"].values, "n_hits"] = joined["n_hits_pq"].values
    return frame


def build_cut_datasets(frame):
    is_fake = frame["pid"].isna()
    pid_class = frame["pid"].map(fe.pid_conversion_dict)
    is_ch = pid_class == 1
    has_trk = frame["is_track_in_MC"].fillna(0) >= 1
    nh = frame["n_hits"]
    # restrict all datasets to fingerprint-matched truth rows so the cut
    # overlays share denominators; fakes are kept in every dataset
    truth_ok = frame["pid"].notna() & nh.notna()
    masks = {
        "no cut": truth_ok,
        "CH must have track": truth_ok & ((~is_ch) | has_trk),
        "≥1 hit": truth_ok & (nh >= 1),
        "≥3 hits": truth_ok & (nh >= 3),
        "≥5 hits": truth_ok & (nh >= 5),
    }
    return [{"label": name, "frame": frame[mask | is_fake].copy(),
             "is_pandora": False} for name, mask in masks.items()]


# ---------------------------------------------------------------------------
# per-panel dedup of identical cut selections (e.g. every photon has >=1
# hit, and the CH-track cut is a no-op for neutrals)
# ---------------------------------------------------------------------------
def _dedup_series(series):
    drawn, order = {}, []
    for label, x, y, err in series:
        key = np.round(np.nan_to_num(np.asarray(y, float), nan=-999.0), 9).tobytes()
        if key in drawn:
            drawn[key]["label"] += f" = {label}"
        else:
            drawn[key] = {"label": label, "x": x, "y": y, "err": err}
            order.append(key)
    return [drawn[k] for k in order]


def plot_metric_grid_delphi(model_curves, metric_key, error_key, ylabel,
                            output_path, logy=False):
    fig, axes = plt.subplots(1, 3, figsize=fe.THREE_PANEL_FIGSIZE,
                             gridspec_kw=fe.THREE_PANEL_GRID_KW)
    scale = fe.metric_scale(metric_key)
    x_key = fe.metric_x_key(metric_key)
    for ax, particle in zip(axes, DELPHI_PARTICLES):
        key = particle["key"]
        series = []
        for label, curves in fe.ordered_curve_items(model_curves):
            curve = curves[key]
            series.append((label,
                           np.asarray(curve[x_key], dtype=float),
                           np.asarray(curve[metric_key], dtype=float) * scale,
                           np.asarray(curve[error_key], dtype=float) * scale))
        all_y = []
        for entry in _dedup_series(series):
            style = build_style_delphi(entry["label"])
            valid = np.isfinite(entry["x"]) & np.isfinite(entry["y"])
            if not np.any(valid):
                continue
            ax.plot(entry["x"][valid], entry["y"][valid], label=entry["label"],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=style["markersize"])
            fe._draw_error_band(ax, entry["x"][valid], entry["y"][valid],
                                entry["err"][valid] / 2.0, style)
            all_y.extend(entry["y"][valid].tolist())
        style_metric_axis_delphi(ax, particle, ylabel, metric_key, all_y, logy)
    fe.save_fixed_canvas(fig, output_path)


def plot_resolution_comparison_delphi(model_curves, output_path):
    fig = plt.figure(figsize=fe.THREE_PANEL_FIGSIZE)
    axes = fe.make_centered_two_panel_axes(fig)
    for ax, particle in zip(axes, DELPHI_RESOLUTION_PARTICLES):
        key = particle["key"]
        series = []
        for label, curves in fe.ordered_curve_items(model_curves):
            curve = curves[key]
            series.append((label,
                           np.asarray(curve["energy"], dtype=float),
                           np.asarray(curve["resolution"], dtype=float),
                           np.asarray(curve["errors"], dtype=float)))
        all_y = []
        for entry in _dedup_series(series):
            style = build_style_delphi(entry["label"])
            valid = fe.filter_resolution_plot_points(key, entry["x"],
                                                     entry["y"], entry["err"])
            if not np.any(valid):
                continue
            ax.plot(entry["x"][valid], entry["y"][valid], label=entry["label"],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], markersize=style["markersize"])
            fe._draw_error_band(ax, entry["x"][valid], entry["y"][valid],
                                entry["err"][valid] / 2.0, style)
            all_y.extend((entry["y"][valid] + entry["err"][valid] / 2.0).tolist())
        ax.set_title(particle["label"])
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel(r"Energy resolution $\sigma/\mu$")
        ax.set_xlim(*particle["xlim"])
        ax.set_xscale("log")
        fe.apply_log_energy_ticks(ax, particle["xlim"])
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(*fe.get_resolution_ylim(particle["key"], all_y))
        ax.legend(fontsize=15, title="Target cut", title_fontsize=14)
    fe.save_fixed_canvas(fig, output_path)


NHITS_BINS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40,
                       50, 65, 80, 100], dtype=float)


def plot_nhits_diagnostic(frame, output_path, min_n=30):
    """Clustering-vs-regression diagnostic vs the TRUTH calo-hit count, for
    truth particles that left >=1 hit.  Top row per class: histogram of truth
    particles per n_hits bin (left axis, log) with the clustering efficiency
    and the clustering x correct-4-class-PID efficiency overlaid (right axis).  Bottom row: median calibrated_E/true_E with
    central-68% (16th-84th percentile) error bars for the matched showers of
    the same bins.  A neutral class that is
    efficient but badly calibrated points at the energy regression; one that
    is simply not matched at low n_hits points at the clustering."""
    truth = frame[frame["pid"].notna() & (frame["n_hits"] >= 1)].copy()
    truth["cls"] = truth["pid"].map(fe.pid_conversion_dict)
    matched = truth["pred_showers_E"].notna()
    pid_ok = matched & (truth["pred_pid_matched"] == truth["cls"])
    response = truth["calibrated_E"] / truth["true_showers_E"]

    centers = 0.5 * (NHITS_BINS[:-1] + NHITS_BINS[1:])
    hist_color, eff_color, resp_color = "#b8b8b8", "#0F4C5C", "#E36414"
    fig, axes = plt.subplots(
        2, 3, figsize=(23, 12.5),
        gridspec_kw={"left": 0.060, "right": 0.955, "bottom": 0.09,
                     "top": 0.94, "wspace": 0.44, "hspace": 0.28})
    for col, particle in enumerate(DELPHI_PARTICLES):
        sel = truth["cls"] == particle["class_id"]
        # linear axis capped at the last bin edge; the tail beyond it is
        # counted in the last (overflow) bin
        nh = truth.loc[sel, "n_hits"].values.astype(float)
        nh = np.minimum(nh, NHITS_BINS[-1] - 0.5)
        found = matched[sel].values
        found_pid = pid_ok[sel].values
        resp = response[sel].values

        counts, eff, eff_err, eff_pid, eff_pid_err, med, p16, p84 = (
            [] for _ in range(8))
        for lo, hi in zip(NHITS_BINS[:-1], NHITS_BINS[1:]):
            m = (nh >= lo) & (nh < hi)
            counts.append(m.sum())
            if m.sum() >= min_n:
                p = found[m].mean()
                eff.append(p)
                eff_err.append(np.sqrt(p * (1 - p) / m.sum()))
                q = found_pid[m].mean()
                eff_pid.append(q)
                eff_pid_err.append(np.sqrt(q * (1 - q) / m.sum()))
                r = resp[m & found]
                r = r[np.isfinite(r)]
                if len(r) >= min_n:
                    med.append(np.median(r))
                    p16.append(np.percentile(r, 16))
                    p84.append(np.percentile(r, 84))
                else:
                    med.append(np.nan), p16.append(np.nan), p84.append(np.nan)
            else:
                eff.append(np.nan)
                eff_err.append(np.nan)
                eff_pid.append(np.nan)
                eff_pid_err.append(np.nan)
                med.append(np.nan), p16.append(np.nan), p84.append(np.nan)
        counts, eff, eff_err = np.array(counts), np.array(eff), np.array(eff_err)
        eff_pid, eff_pid_err = np.array(eff_pid), np.array(eff_pid_err)
        med, p16, p84 = np.array(med), np.array(p16), np.array(p84)

        # top: population + clustering efficiency
        ax = axes[0, col]
        ax.stairs(counts, NHITS_BINS, fill=True,
                  color=hist_color, alpha=0.55, label="truth particles")
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.5, counts[counts > 0].min() * 0.5)
                    if (counts > 0).any() else 0.5)
        ax.set_ylabel("Truth particles (≥1 hit)")
        ax.set_title(particle["label"])
        ax_eff = ax.twinx()
        ax_eff.plot(centers, eff, "o-", color=eff_color, markersize=8,
                    label="clustering efficiency")
        ax_eff.fill_between(centers, eff - eff_err, eff + eff_err,
                            color=eff_color, alpha=0.15, linewidth=0)
        ax_eff.plot(centers, eff_pid, "^--", color="#C7522A", markersize=8,
                    label="clustering × PID efficiency")
        ax_eff.fill_between(centers, eff_pid - eff_pid_err,
                            eff_pid + eff_pid_err,
                            color="#C7522A", alpha=0.15, linewidth=0)
        ax_eff.set_ylim(0.0, 1.05)
        ax_eff.set_ylabel("Clustering efficiency", color=eff_color)
        ax_eff.tick_params(axis="y", labelcolor=eff_color)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_eff.get_legend_handles_labels()
        ax_eff.legend(h1 + h2, l1 + l2, fontsize=16, loc="lower right")

        # bottom: energy response of the matched showers; bars span the
        # central 68% (16th-84th percentile), the Gaussian-1sigma equivalent
        ax = axes[1, col]
        yerr = np.vstack([med - p16, p84 - med])
        ax.errorbar(centers, med, yerr=np.where(np.isfinite(yerr), yerr, 0),
                    fmt="o-", color=resp_color, markersize=8, capsize=4,
                    linewidth=2.2, elinewidth=1.6,
                    label=r"median $\pm$ q68 (16-84%)")
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_ylabel(r"$E_{\mathrm{calib}} / E_{\mathrm{true}}$ (matched)")
        fin = np.isfinite(p16) & np.isfinite(p84)
        if fin.any():
            lo_y = min(0.75, np.nanmin(p16[fin]) - 0.05)
            hi_y = max(1.25, np.nanmax(p84[fin]) + 0.05)
            ax.set_ylim(lo_y, hi_y)
        ax.legend(fontsize=16, loc="upper right")

        for ax in (axes[0, col], axes[1, col]):
            ax.set_xlim(0, NHITS_BINS[-1])
            ax.set_xlabel("Truth calo hits / particle")
            ax.grid(True, axis="x", alpha=0.25, linestyle="--")
            ax.set_axisbelow(True)
    fig.text(0.955, 0.015, f"last bin includes overflow (>{NHITS_BINS[-1]:.0f} hits)",
             ha="right", va="bottom", fontsize=15, color="0.35")
    fe.save_fixed_canvas(fig, output_path)


def plot_confusion_matrix_delphi(frame, output_path):
    """fe.plot_confusion_matrix_grid for a single model: one value per cell
    (full-cell colour fill, big row-normalised %, small count), no quadrant
    sub-grid or mini-boxes — those exist to overlay the four CLD datasets."""
    class_order = fe.CONFUSION_CLASS_ORDER
    n = len(class_order)
    x_labels = [fe.CONFUSION_CLASS_NAMES[c] for c in class_order] + ["missed"]
    y_labels = [fe.CONFUSION_CLASS_NAMES[c] for c in class_order] + ["fake"]
    fake_row = n
    color = CUT_STYLES[DELPHI_LABEL]["color"]
    cmap = LinearSegmentedColormap.from_list("delphi", ["#ffffff", color])

    fig, axes = plt.subplots(
        len(DELPHI_CONFUSION_ENERGY_BINS), 1,
        figsize=(13.5, 9.8 * len(DELPHI_CONFUSION_ENERGY_BINS)),
        gridspec_kw={"left": 0.050, "right": 0.995, "bottom": 0.03,
                     "top": 0.97, "hspace": 0.24})
    for ax, (elo, ehi) in zip(np.atleast_1d(axes), DELPHI_CONFUSION_ENERGY_BINS):
        matrix = fe.compute_confusion_matrix(frame, False, elo, ehi)
        percent = fe.mixed_percentages(matrix, fake_row, fake_norm="column")
        n_rows = n_cols = n + 1
        ax.set_xlim(-1.65, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_aspect("equal")
        ax.text(-0.83, -0.22, r"$N_{\mathrm{true}}$", ha="center", va="center",
                fontsize=29, fontweight="bold")
        for i in range(n_rows):
            if i < fake_row:
                ax.add_patch(Rectangle((-1.60, i), 1.45, 1, facecolor="#fafafa",
                                       edgecolor="0.45", linewidth=1.2))
                ax.text(-0.875, i + 0.5, f"{int(matrix[i, :].sum())}",
                        ha="center", va="center", fontsize=19,
                        fontweight="bold", color=color)
            for j in range(n_cols):
                p = float(np.clip(percent[i, j], 0.0, 100.0))
                text_color = "black" if p < 62.0 else "white"
                ax.add_patch(Rectangle(
                    (j, i), 1, 1,
                    facecolor=cmap(0.04 + 0.96 * (p / 100.0) ** 1.15),
                    edgecolor="0.20", linewidth=1.4, alpha=0.9))
                ax.text(j + 0.5, i + 0.44, f"{int(np.rint(p))}",
                        ha="center", va="center", fontsize=26,
                        fontweight="bold", color=text_color)
                ax.text(j + 0.5, i + 0.78, f"{matrix[i, j]}",
                        ha="center", va="center", fontsize=12,
                        color=text_color)
        ax.hlines(fake_row, xmin=-1.65, xmax=n_cols, linewidth=1.6,
                  color="black", alpha=0.85)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=30)
        ax.set_yticklabels(y_labels, rotation=0, fontsize=30)
        ax.set_xlabel("Predicted", fontsize=33, fontweight="bold")
        ax.set_ylabel("True", fontsize=33, fontweight="bold")
        ax.set_title(rf"${elo:g}\,\mathrm{{GeV}} < E < {ehi:g}\,\mathrm{{GeV}}$",
                     fontsize=37, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fe.save_fixed_canvas(fig, output_path)


def plot_event_comparison_delphi(frame, output_path, component_label=None,
                                 class_id=None, truth_ids=None, logy=False):
    """fe.plot_event_comparison for a single dataset, with axis windows from
    the data instead of CLD-tuned constants."""
    if class_id is not None or truth_ids is not None:
        sub = fe.select_event_component(frame, class_id=class_id,
                                        truth_ids=truth_ids)
    else:
        sub = frame
    if len(sub) == 0:
        return
    energy_over_true, mass_over_true = fe.compute_event_distributions(sub, False)

    fig, axes = plt.subplots(1, 2, figsize=fe.EVENT_FIGSIZE,
                             gridspec_kw=fe.EVENT_GRID_KW)
    style = build_style_delphi(DELPHI_LABEL)
    for ax, values, xlabel in (
            (axes[0], energy_over_true, r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$"),
            (axes[1], mass_over_true, r"$M_{\mathrm{reco}} / M_{\mathrm{true}}$")):
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        if not len(finite):
            continue
        lo, hi = np.percentile(finite, [0.5, 99.5])
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        bins = np.linspace(lo - pad, hi + pad, 120)
        stats = fe.summarize_distribution(values)
        label = DELPHI_LABEL
        if stats:
            label += f"\nmed={stats['median']:.3f}, q68={stats['q68']:.3f}"
        fe._plot_histogram(ax, values, bins, style, label)
        ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized entries")
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(True, alpha=0.25, linestyle="--")
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=19, title="Model", title_fontsize=17)
    if component_label:
        fig.suptitle(component_label, fontsize=28, y=0.96)
    fe.save_fixed_canvas(fig, output_path)


def plot_visible_mass_delphi(frame, output_path):
    """Absolute visible mass and visible energy per event (truth vs reco),
    same computation as the M_reco/M_true ratio panels
    (calculate_event_mass_resolution with ML-PID mass assignment) but drawn
    in GeV so the Z peak is visible directly."""
    dic = fe.calculate_event_mass_resolution(
        frame, False, perfect_pid=False, mass_zero=False, ML_pid=True,
        fake=False)
    panels = [
        (np.asarray(dic["mass_true"]), np.asarray(dic["mass_pred_p"]),
         r"Visible mass [GeV]", 91.19),
        (np.asarray(dic["E_true"]), np.asarray(dic["E_pred"]),
         r"Visible energy [GeV]", 91.19),
    ]
    fig, axes = plt.subplots(1, 2, figsize=fe.EVENT_FIGSIZE,
                             gridspec_kw=fe.EVENT_GRID_KW)
    for ax, (true_v, pred_v, xlabel, ref) in zip(axes, panels):
        true_v = true_v[np.isfinite(true_v)]
        pred_v = pred_v[np.isfinite(pred_v)]
        lo = min(np.percentile(true_v, 0.5), np.percentile(pred_v, 0.5))
        hi = max(np.percentile(true_v, 99.5), np.percentile(pred_v, 99.5))
        pad = 0.05 * (hi - lo)
        bins = np.linspace(lo - pad, hi + pad, 110)
        for values, label, color in (
                (true_v, "truth", "0.35"),
                (pred_v, DELPHI_LABEL, CUT_STYLES[DELPHI_LABEL]["color"])):
            stats = fe.summarize_distribution(values)
            ax.hist(values, bins=bins, histtype="step",
                    weights=np.ones_like(values) / len(values),
                    color=color, linewidth=2.4, alpha=0.85,
                    label=f"{label}\nmed={stats['median']:.1f}, "
                          f"q68={stats['q68']:.1f} GeV")
        ax.axvline(ref, color="black", linewidth=1.2, linestyle="--",
                   alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized entries")
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=17)
    fig.suptitle(r"$Z\to q\bar{q}$ visible mass / energy", fontsize=28, y=0.96)
    fe.save_fixed_canvas(fig, output_path)


def main():
    args = parse_args()
    fe.configure_plot_style()
    apply_fe_patches()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_dir = os.path.join(args.output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    frame = fe.validate_hitpf_frame(
        fe.prepare_mlpf_frame(fe.load_eval_dataframe(args.mlpf)), DELPHI_LABEL)
    frame = attach_target_info(frame, args.val_dir)
    datasets = build_cut_datasets(frame)

    model_eff_curves = {}
    resolution_curves = {}
    summary_rows = []
    for dataset in datasets:
        label = dataset["label"]
        model_eff_curves[label] = fe.compute_particle_curves(dataset["frame"], False)
        resolution_curves[label] = fe.compute_resolution_curves(dataset["frame"], False)
        summary_rows.extend(fe.summarize_run(label, dataset["frame"], False))
        print(f"[{label}] {len(dataset['frame'])} rows")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "full_evaluation_summary.csv"), index=False)

    plot_metric_grid_delphi(
        model_eff_curves, "eff", "errors", "Clustering efficiency",
        os.path.join(summary_dir, "overview_Efficiency_clustering.pdf"))
    plot_metric_grid_delphi(
        model_eff_curves, "eff_pid", "errors_pid", "Efficiency with PID",
        os.path.join(summary_dir, "overview_Efficiency_pid.pdf"))
    plot_metric_grid_delphi(
        model_eff_curves, "fake_rate", "fake_rate_err", "Fake rate",
        os.path.join(summary_dir, "overview_FakeRate.pdf"), logy=True)
    plot_metric_grid_delphi(
        model_eff_curves, "fake_energy_fraction", "fake_energy_fraction_err",
        r"Fake energy [$\%$]",
        os.path.join(summary_dir, "overview_FakeEnergy.pdf"), logy=True)

    plot_confusion_matrix_delphi(
        frame, os.path.join(summary_dir, "pid_confusion_matrix_per_energy.pdf"))

    plot_nhits_diagnostic(
        frame, os.path.join(summary_dir, "nhits_efficiency_response.pdf"))

    plot_visible_mass_delphi(
        frame, os.path.join(summary_dir, "visible_mass.pdf"))

    plot_event_comparison_delphi(
        frame, os.path.join(summary_dir, "event_energy_mass_comparison.pdf"),
        component_label="Inclusive")
    for component in fe.EVENT_COMPONENTS:
        plot_event_comparison_delphi(
            frame, os.path.join(summary_dir, component["output_name"]),
            component_label=component["label"],
            class_id=component.get("class_id"),
            truth_ids=component.get("truth_ids"),
            logy=component.get("logy", False))

    plot_resolution_comparison_delphi(
        resolution_curves,
        os.path.join(summary_dir, "particle_energy_resolution.pdf"))

    print("Wrote DELPHI full-evaluation plots to", summary_dir)
    print("Summary table:",
          os.path.join(args.output_dir, "full_evaluation_summary.csv"))


if __name__ == "__main__":
    main()
