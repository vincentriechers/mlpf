"""DELPHI full-evaluation plots with the TRAINED MODELS as the comparison axis.

`full_evaluation_delphi.py` overlays hypothetical target cuts for one model; the
comparison the read-out-input study needs is between the models themselves, so this
driver reuses that module's DELPHI-tuned plot functions with the model labels on the
comparison axis.

  * per-particle curves (clustering efficiency, efficiency with PID, fake rate, fake
    energy, energy resolution): every model in ONE figure per metric;
  * single-dataset figures (PID confusion grid, n_hits diagnostic, visible mass,
    event energy and mass): one per model, in <output-dir>/<model>/.

DELANA has no per-particle dataframe in this pipeline, so it appears in none of these.
The event-level DELANA comparison lives in
`delphi_converter/analysis/fair_visible_energy.py` and its figures in
`delphi_converter/analysis/raw_hits/plot_model_vs_delana.py`.

Colours follow the study's house style (`delphi_converter/analysis/chef2026/chef_style.py`):
HitPF teal, Mask3D amethyst, so these figures read as siblings of the event-level ones.

Usage (LCG_109 view or the gatr container, from the mlpf repo root):
    python -m src.evaluation.full_evaluation_delphi_models \
        --model HitPF=<pickle or glob> --model Mask3D=<pickle or glob> \
        --val-dir <directory of evaluated pf_tree parquets> --output-dir <dir>
"""

import argparse
import os

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.evaluation.full_evaluation as fe
import src.evaluation.full_evaluation_delphi as fed

# House identity colours; markers keep the classes apart without relying on colour.
MODEL_STYLES = {
    "HitPF":  {"color": "#0F4C5C", "marker": "o"},
    "Mask3D": {"color": "#8E44AD", "marker": "s"},
    "DELANA": {"color": "#E8503A", "marker": "^"},
}


DELANA_STYLE = {"color": "#E8503A", "marker": "^", "linestyle": "--", "markersize": 9, "alpha": 0.10}
GE, GP = 8, slice(12, 15)          # X_gen energy, momentum
LST, LPDG, LP, LE = 1, 2, slice(3, 6), 6   # X_lu status, PDG, momentum, energy
NEUTRINOS = (12, 14, 16)
PE, PP, PLOCK = 4, slice(1, 4), 11  # X_pa energy, momentum, lock


def delana_event_quantities(val_dir, skip_files=0, n_files=None):
    """Per event, from the SAME parquets the models were evaluated on: DELANA's visible
    energy and mass, and the training targets' visible energy and mass.

    DELANA is a list of four-vectors, so its visible mass is the mass of their sum; the
    models' mass in these figures instead comes from their predicted momenta with a mass
    hypothesis per predicted class. The target values printed by `main` let the caller
    check that the two denominators agree before trusting the overlay.
    """
    files = sorted(glob.glob(os.path.join(val_dir, "pf_tree_*.parquet")))[skip_files:]
    files = files[:n_files] if n_files else files
    dE, dM, tE, tM, gE, gM = [], [], [], [], [], []
    for f in files:
        row = pq.read_table(f, columns=["X_gen", "ygen_hit", "ygen_track", "X_pa",
                                        "X_lu"]).to_pylist()[0]
        for i in range(len(row["X_gen"])):
            xg = np.asarray(row["X_gen"][i], dtype=float).reshape(-1, 21)
            seen = {int(k) for k in np.asarray(row["ygen_hit"][i], dtype=np.int64) if k >= 0}
            seen |= {int(k) for k in np.asarray(row["ygen_track"][i], dtype=np.int64) if k >= 0}
            idx = np.fromiter(seen, dtype=np.int64) if seen else np.zeros(0, dtype=np.int64)
            idx = idx[idx < len(xg)]
            e = float(xg[idx, GE].sum()) if len(idx) else 0.0
            p = xg[idx, GP].sum(axis=0) if len(idx) else np.zeros(3)
            tE.append(e); tM.append(float(np.sqrt(max(e * e - p @ p, 0.0))))
            pa = (np.asarray(row["X_pa"][i], dtype=float).reshape(-1, 18)
                  if row["X_pa"][i] else np.zeros((0, 18)))
            un = pa[pa[:, PLOCK] == 0] if len(pa) else pa
            e = float(un[:, PE].sum()) if len(un) else 0.0
            p = un[:, PP].sum(axis=0) if len(un) else np.zeros(3)
            dE.append(e); dM.append(float(np.sqrt(max(e * e - p @ p, 0.0))))
            lu = (np.asarray(row["X_lu"][i], dtype=float).reshape(-1, 7)
                  if row["X_lu"][i] else np.zeros((0, 7)))
            vis = ((lu[:, LST] == 1) & ~np.isin(np.abs(lu[:, LPDG]), NEUTRINOS)
                   if len(lu) else np.zeros(0, bool))
            e = float(lu[vis, LE].sum()) if len(lu) else 0.0
            p = lu[vis, LP].sum(axis=0) if len(lu) else np.zeros(3)
            gE.append(e); gM.append(float(np.sqrt(max(e * e - p @ p, 0.0))))
    return {k: np.asarray(v) for k, v in
            (("delana_E", dE), ("delana_M", dM), ("target_E", tE), ("target_M", tM),
             ("gen_E", gE), ("gen_M", gM))}


def plot_visible_scale_comparison(model_dics, delana, output_path):
    """Absolute visible energy and visible mass per event, in GeV: the truth, DELANA and
    every model on one pair of axes. The Z peak is directly visible."""
    fig, axes = plt.subplots(1, 2, figsize=fe.EVENT_FIGSIZE, gridspec_kw=fe.EVENT_GRID_KW)
    panels = [("E_true", "E_pred", "gen_E", "delana_E", "Visible energy [GeV]"),
              ("mass_true", "mass_pred_p", "gen_M", "delana_M", "Visible mass [GeV]")]
    for ax, (tk, pk, gk, dk, xlabel) in zip(axes, panels):
        truth = np.asarray(next(iter(model_dics.values()))[tk], dtype=float)
        # the grey curve is the TRAINING TARGETS, which exclude particles that never reach a
        # calorimeter; the generator curve is what a complete reconstruction should reproduce
        series = [("Training targets", truth, {"color": "#808080", "linestyle": "-"})]
        if delana is not None:
            series.append(("DELANA", delana[dk], DELANA_STYLE))
        for name, dic in model_dics.items():
            series.append((name, np.asarray(dic[pk], dtype=float), fed.build_style_delphi(name)))
        pooled = np.concatenate([np.asarray(v, dtype=float) for _, v, _ in series])
        pooled = pooled[np.isfinite(pooled)]
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        bins = np.linspace(max(lo - 0.05 * (hi - lo), 0.0), hi + 0.05 * (hi - lo), 110)
        for name, values, style in series:
            stats = fe.summarize_distribution(values)
            label = name + (f"  {stats['median']:.1f} +- {stats['q68']:.1f}" if stats else "")
            fe._plot_histogram(ax, values, bins, style, label)
        if delana is not None:
            # the generator is a near-delta at the beam energy, so it is a reference band,
            # not a histogram: drawn as one it flattens every reconstruction curve
            g = fe.summarize_distribution(np.asarray(delana[gk], dtype=float))
            ax.axvspan(g["median"] - g["q68"], g["median"] + g["q68"],
                       color="#111111", alpha=0.08, zorder=0)
            ax.axvline(g["median"], color="#111111", lw=2.0, ls="-.", zorder=1)
            ax.plot([], [], color="#111111", lw=2.0, ls="-.",
                    label=f"Generator  {g['median']:.1f} +- {g['q68']:.1f}")
        else:
            ax.axvline(91.19, color="black", linewidth=1.2, linestyle=":", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized entries")
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(fontsize=14, title="median +- q68 [GeV]", title_fontsize=13)
    fe.save_fixed_canvas(fig, output_path)


def plot_event_comparison_models(frames, output_path, component_label=None,
                                 class_id=None, truth_ids=None, logy=False, extra=None):
    """fed.plot_event_comparison_delphi with every model overlaid, as the ARC figure
    overlays its datasets. Binning comes from the pooled distribution so the curves share it."""
    subs = {}
    for name, frame in frames.items():
        sub = (fe.select_event_component(frame, class_id=class_id, truth_ids=truth_ids)
               if (class_id is not None or truth_ids is not None) else frame)
        if len(sub):
            subs[name] = fe.compute_event_distributions(sub, False)
    if not subs:
        return
    fig, axes = plt.subplots(1, 2, figsize=fe.EVENT_FIGSIZE, gridspec_kw=fe.EVENT_GRID_KW)
    panels = ((0, r"$E_{\mathrm{reco}} / E_{\mathrm{true}}$"),
              (1, r"$M_{\mathrm{reco}} / M_{\mathrm{true}}$"))
    for ax, (j, xlabel) in zip(axes, panels):
        pooled = [np.asarray(v[j], dtype=float) for v in subs.values()]
        if extra is not None:
            pooled.append(np.asarray(extra[1][j], dtype=float))
        pooled = np.concatenate(pooled)
        pooled = pooled[np.isfinite(pooled)]
        if not len(pooled):
            continue
        lo, hi = np.percentile(pooled, [0.5, 99.5])
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        bins = np.linspace(lo - pad, hi + pad, 120)
        for name in frames:
            if name not in subs:
                continue
            values = np.asarray(subs[name][j], dtype=float)
            stats = fe.summarize_distribution(values)
            label = name + (f"\nmed={stats['median']:.3f}, q68={stats['q68']:.3f}" if stats else "")
            fe._plot_histogram(ax, values, bins, fed.build_style_delphi(name), label)
        if extra is not None:
            values = np.asarray(extra[1][j], dtype=float)
            stats = fe.summarize_distribution(values)
            label = extra[0] + (f"\nmed={stats['median']:.3f}, q68={stats['q68']:.3f}" if stats else "")
            fe._plot_histogram(ax, values, bins, DELANA_STYLE, label)
        ax.axvline(1.0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized entries")
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(True, alpha=0.25, linestyle="--")
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=17, title=fed.LEGEND_TITLE, title_fontsize=15)
    if component_label:
        fig.suptitle(component_label, fontsize=28, y=0.96)
    fe.save_fixed_canvas(fig, output_path)


def enable_ratio_panels(order):
    """Make the ARC ratio machinery (built for ARC/05 geometries) compare the second
    model with the first. Returns the pair label, or None if there is nothing to pair."""
    if len(order) < 2:
        return None
    num, den = order[1], order[0]
    pair = f"{num} / {den}"
    fe.RATIO_METHOD_STYLES[pair] = {"color": fed.CUT_STYLES[num]["color"],
                                    "marker": fed.CUT_STYLES[num]["marker"],
                                    "linestyle": "-", "markersize": 6, "alpha": 0.15}
    fe.build_ratio_pairs = lambda curves, _p=[(pair, num, den)]: _p
    original = fe.style_ratio_axis
    fe.style_ratio_axis = (lambda ax, particle, ylabel, values, metric_key=None, _o=original, _l=pair:
                           _o(ax, particle, _l, values, metric_key=metric_key))
    # the ratio figures read the module-level particle lists, which still carry the CLD energy ranges
    fe.PARTICLES = fed.DELPHI_PARTICLES
    fe.RESOLUTION_PARTICLES = fed.DELPHI_RESOLUTION_PARTICLES
    return pair


def parse_args():
    p = argparse.ArgumentParser(description="DELPHI full-evaluation plots, models overlaid.")
    p.add_argument("--model", action="append", required=True, metavar="NAME=GLOB",
                   help="repeatable; NAME labels the curve, GLOB is the evaluation pickle")
    p.add_argument("--val-dir", required=True,
                   help="directory with the evaluated pf_tree parquets (per-target hit counts)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--delana", action="store_true",
                   help="add DELANA to the event-level figures, read from the val-dir parquets")
    p.add_argument("--skip-files", type=int, default=0,
                   help="with --delana: the parquet slice the models were evaluated on")
    p.add_argument("--n-files", type=int, default=None)
    p.add_argument("--skip-target-info", action="store_true",
                   help="skip the parquet join; drops only the n_hits diagnostic")
    return p.parse_args()


def main():
    args = parse_args()
    specs = [s.split("=", 1) for s in args.model]
    order = [n for n, _ in specs]

    fe.configure_plot_style()
    fed.apply_fe_patches()
    # the comparison axis is the models, in the order they were given
    fed.CUT_STYLES.update({n: MODEL_STYLES.get(n, {"color": "black", "marker": "o"}) for n in order})
    fe.comparison_sort_key = lambda label: (order.index(str(label)) if str(label) in order
                                            else len(order), str(label))
    fed.LEGEND_TITLE = "Reconstruction"                # not the cut driver's "Target cut"
    fe.LEGEND_TITLE = "Reconstruction"                 # nor the ARC driver's "Geometry / method"
    fe.TWO_PANEL_CENTERED_GRID_KW["wspace"] = 0.42     # the long y label needs more room than the CLD default

    summary_dir = os.path.join(args.output_dir, "summary_plots")
    os.makedirs(summary_dir, exist_ok=True)

    frames, eff_curves, res_curves, summary_rows = {}, {}, {}, []
    for name, pattern in specs:
        fed.DELPHI_LABEL = name                       # the single-dataset figures read this
        frame = fe.validate_hitpf_frame(
            fe.prepare_mlpf_frame(fe.load_eval_dataframe(pattern)), name)
        if not args.skip_target_info:
            frame = fed.attach_target_info(frame, args.val_dir)
        frames[name] = frame
        eff_curves[name] = fe.compute_particle_curves(frame, False)
        res_curves[name] = fe.compute_resolution_curves(frame, False)
        summary_rows.extend(fe.summarize_run(name, frame, False))
        print(f"[{name}] {len(frame)} rows")

    delana = None
    if args.delana:
        delana = delana_event_quantities(args.val_dir, args.skip_files, args.n_files)
        print(f"[DELANA] {len(delana['delana_E'])} events from the parquets")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "full_evaluation_summary.csv"), index=False)

    for metric, err, ylabel, fname, logy in [
            ("eff", "errors", "Clustering efficiency", "overview_Efficiency_clustering.pdf", False),
            ("eff_pid", "errors_pid", "Efficiency with PID", "overview_Efficiency_pid.pdf", False),
            ("fake_rate", "fake_rate_err", "Fake rate", "overview_FakeRate.pdf", True),
            ("fake_energy_fraction", "fake_energy_fraction_err", r"Fake energy [$\%$]",
             "overview_FakeEnergy.pdf", True)]:
        fed.plot_metric_grid_delphi(eff_curves, metric, err, ylabel,
                                    os.path.join(summary_dir, fname), logy=logy)

    fed.plot_resolution_comparison_delphi(
        res_curves, os.path.join(summary_dir, "particle_energy_resolution.pdf"))

    # the ARC set also carries a ratio panel under each overview; here it is model / model
    pair = enable_ratio_panels(order)
    if pair:
        for metric, err, ylabel, fname, logy in [
                ("eff", "errors", "Clustering efficiency",
                 "overview_Efficiency_clustering_with_ratio.pdf", False),
                ("eff_pid", "errors_pid", "Efficiency with PID",
                 "overview_Efficiency_pid_with_ratio.pdf", False),
                ("fake_rate", "fake_rate_err", "Fake rate",
                 "overview_FakeRate_with_ratio.pdf", True),
                ("fake_energy_fraction", "fake_energy_fraction_err", r"Fake energy [$\%$]",
                 "overview_FakeEnergy_with_ratio.pdf", True)]:
            try:
                fe.plot_metric_grid_with_ratio(eff_curves, metric, err, ylabel,
                                               os.path.join(summary_dir, fname), logy=logy)
            except Exception as exc:                                   # noqa: BLE001
                print(f"[warn] ratio figure {fname} failed: {exc}")
        try:
            fe.plot_resolution_comparison_with_ratio(
                res_curves, os.path.join(summary_dir,
                                         "particle_energy_resolution_with_ratio.pdf"))
        except Exception as exc:                                       # noqa: BLE001
            print(f"[warn] resolution ratio figure failed: {exc}")

    # event energy and mass with the models overlaid (the ARC figure; the per-model
    # copies below are what the single-dataset DELPHI driver produces)
    extra = None
    model_dics = {n: fe.calculate_event_mass_resolution(f, False, perfect_pid=False,
                                                        mass_zero=False, ML_pid=True, fake=False)
                  for n, f in frames.items()}
    if delana is not None:
        ok = (delana["target_E"] > 0) & (delana["target_M"] > 0)
        extra = ("DELANA", (np.where(ok, delana["delana_E"] / delana["target_E"], np.nan),
                            np.where(ok, delana["delana_M"] / delana["target_M"], np.nan)))
        # the two sides must share a denominator, or the overlay compares different things
        for name, dic in model_dics.items():
            for key, mine, what in (("E_true", delana["target_E"], "energy"),
                                    ("mass_true", delana["target_M"], "mass")):
                a, b = np.median(np.asarray(dic[key], dtype=float)), np.median(mine[ok])
                flag = "OK" if abs(a - b) < 0.05 * max(abs(b), 1e-9) else "MISMATCH"
                print(f"[DELANA] truth {what:6s} median: {name} {a:.3f} vs parquets {b:.3f}  -> {flag}")
        plot_visible_scale_comparison(
            model_dics, delana, os.path.join(summary_dir, "visible_scale_comparison.pdf"))

    plot_event_comparison_models(
        frames, os.path.join(summary_dir, "event_energy_mass_comparison.pdf"),
        component_label="Inclusive", extra=extra)
    for component in fe.EVENT_COMPONENTS:
        plot_event_comparison_models(
            frames, os.path.join(summary_dir, component["output_name"]),
            component_label=component["label"], class_id=component.get("class_id"),
            truth_ids=component.get("truth_ids"), logy=component.get("logy", False))

    for name, frame in frames.items():
        fed.DELPHI_LABEL = name
        d = os.path.join(summary_dir, name)
        os.makedirs(d, exist_ok=True)
        fed.plot_confusion_matrix_delphi(
            frame, os.path.join(d, "pid_confusion_matrix_per_energy.pdf"))
        if not args.skip_target_info:
            fed.plot_nhits_diagnostic(frame, os.path.join(d, "nhits_efficiency_response.pdf"))
        fed.plot_visible_mass_delphi(frame, os.path.join(d, "visible_mass.pdf"))
        fed.plot_event_comparison_delphi(
            frame, os.path.join(d, "event_energy_mass_comparison.pdf"),
            component_label="Inclusive")
        for component in fe.EVENT_COMPONENTS:
            fed.plot_event_comparison_delphi(
                frame, os.path.join(d, component["output_name"]),
                component_label=component["label"], class_id=component.get("class_id"),
                truth_ids=component.get("truth_ids"), logy=component.get("logy", False))

    print("Wrote DELPHI model-comparison plots to", summary_dir)


if __name__ == "__main__":
    main()
