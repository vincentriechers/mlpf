"""
Post-mortem analysis of model performance.

Investigates what drives M_pred/M_true degradation by decomposing contributions from:
  1. PID mis-classification
  2. Energy correction errors
  3. Clustering errors (fakes / missed particles)

per particle class (electron, charged hadron, neutral hadron, photon, muon).

Usage:
    python -m src.evaluation.postmortem
    # or set DATA_PATH below and run directly
"""

import matplotlib
matplotlib.rc('font', size=9)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
from torch_scatter import scatter_sum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.inference.pandas_helpers import open_mlpf_dataframe, concat_with_batch_fix
from src.utils.inference.event_metrics import (
    calculate_event_mass_resolution,
    particle_masses_4_class,
    safeint,
)
from src.utils.pid_conversion import pid_conversion_dict, our_to_pandora_mapping

# ---------------------------------------------------------------------------
# Paths – edit to point at the evaluation .pt files
# ---------------------------------------------------------------------------
dir_top = "/eos/user/m/mgarciam/datasets_mlpf/models_trained_CLD/16032026/"
PATH_model   = os.path.join(dir_top, "showers_df_evaluation/test_tangent0_0_None.pt")
PATH_pandora = os.path.join(dir_top, "showers_df_evaluation/test_tangent0_0_None_pandora.pt")
OUTPUT_DIR   = os.path.join(dir_top, "0", "postmortem")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data  (same pre-processing as all_plots.py)
# ---------------------------------------------------------------------------
print("Loading data …")
sd_hgb, _    = open_mlpf_dataframe(PATH_model,   False, False)
sd_pandora, _ = open_mlpf_dataframe(PATH_pandora, False, False)

# Re-classify low-energy photons predicted as neutral hadrons → electron
mask = (sd_hgb.pred_pid_matched == 4) * (sd_hgb.calibrated_E < 1)
sd_hgb.loc[mask, "pred_pid_matched"] = 1

# Add pion/electron mass to calibrated_E so that it represents total energy
mask = sd_hgb.pred_pid_matched == 1
sd_hgb.loc[mask, "calibrated_E"] = np.sqrt(
    sd_hgb[mask]["calibrated_E"] ** 2 + (1.3957018e-1 ** 2)
)
mask = sd_hgb.pred_pid_matched == 0
sd_hgb.loc[mask, "calibrated_E"] = np.sqrt(
    sd_hgb[mask]["calibrated_E"] ** 2 + (5.10998902e-4 ** 2)
)

print(f"Model rows : {len(sd_hgb)}")
print(f"Pandora rows: {len(sd_pandora)}")

# ---------------------------------------------------------------------------
# Helper: IQR-based sigma/mu (same as in mass.py)
# ---------------------------------------------------------------------------
def iqr_sigma_over_mu(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    med = np.median(arr)
    p16, p84 = np.percentile(arr, 16), np.percentile(arr, 84)
    return med, (p84 - p16) / (2 * med) if med != 0 else np.nan


# ---------------------------------------------------------------------------
# 1.  OVERALL  M_pred / M_true  and  E_pred / E_true
# ---------------------------------------------------------------------------
print("\n=== OVERALL mass & energy resolution ===")
dic_model   = calculate_event_mass_resolution(sd_hgb,    False, ML_pid=True)
dic_pandora = calculate_event_mass_resolution(sd_pandora, True,  ML_pid=True)

mass_med_m, mass_res_m  = iqr_sigma_over_mu(dic_model["mass_over_true_p"].numpy())
mass_med_p, mass_res_p  = iqr_sigma_over_mu(dic_pandora["mass_over_true_p"].numpy())
E_med_m,    E_res_m     = iqr_sigma_over_mu(dic_model["E_over_true"].numpy())
E_med_p,    E_res_p     = iqr_sigma_over_mu(dic_pandora["E_over_true"].numpy())

print(f"  Model   | M_pred/M_true  median={mass_med_m:.3f}  σ/μ={mass_res_m*100:.1f}%")
print(f"  Pandora | M_pred/M_true  median={mass_med_p:.3f}  σ/μ={mass_res_p*100:.1f}%")
print(f"  Model   | E_pred/E_true  median={E_med_m:.3f}  σ/μ={E_res_m*100:.1f}%")
print(f"  Pandora | E_pred/E_true  median={E_med_p:.3f}  σ/μ={E_res_p*100:.1f}%")


class_names = {0: "electron", 1: "ch. hadron", 2: "neu. hadron", 3: "photon", 4: "muon"}
class_to_pdg = {0: [11], 1: [211, 321, 2212], 2: [130, 2112, 310], 3: [22], 4: [13]}

# ---------------------------------------------------------------------------
# 2.  ORACLE STUDIES  – replace one component with truth at a time
#     Each oracle isolates the impact of that component on M_pred/M_true.
# ---------------------------------------------------------------------------
print("\n=== Oracle studies (ablations) ===")

def mass_res_from_dic(d):
    return iqr_sigma_over_mu(d["mass_over_true_p"].numpy())

# 2a. Perfect PID  (use true PID mass assignment)
dic_perfect_pid = calculate_event_mass_resolution(sd_hgb, False, perfect_pid=True, ML_pid=False)
med_pp, res_pp = mass_res_from_dic(dic_perfect_pid)
print(f"  Oracle perfect PID        | M_pred/M_true  median={med_pp:.3f}  σ/μ={res_pp*100:.1f}%")

# 2b. Perfect energy  (replace calibrated_E with true_showers_E)
df_perfect_E = sd_hgb.copy(deep=True)
df_perfect_E["calibrated_E"] = df_perfect_E["true_showers_E"]
dic_perfect_E = calculate_event_mass_resolution(df_perfect_E, False, ML_pid=True)
med_pe, res_pe = mass_res_from_dic(dic_perfect_E)
print(f"  Oracle perfect energy     | M_pred/M_true  median={med_pe:.3f}  σ/μ={res_pe*100:.1f}%")

# 2c. Perfect direction  (replace pred_pos_matched with true_pos)
df_perfect_dir = sd_hgb.copy(deep=True)
df_perfect_dir["pred_pos_matched"] = df_perfect_dir["true_pos"]
dic_perfect_dir = calculate_event_mass_resolution(df_perfect_dir, False, ML_pid=True)
med_pd, res_pd = mass_res_from_dic(dic_perfect_dir)
print(f"  Oracle perfect direction  | M_pred/M_true  median={med_pd:.3f}  σ/μ={res_pd*100:.1f}%")

# 2d. Perfect PID + energy
df_perfect_pid_E = sd_hgb.copy(deep=True)
df_perfect_pid_E["calibrated_E"] = df_perfect_pid_E["true_showers_E"]
dic_perfect_pid_E = calculate_event_mass_resolution(df_perfect_pid_E, False, perfect_pid=True, ML_pid=False)
med_ppe, res_ppe = mass_res_from_dic(dic_perfect_pid_E)
print(f"  Oracle perfect PID+E      | M_pred/M_true  median={med_ppe:.3f}  σ/μ={res_ppe*100:.1f}%")

# 2e. Perfect energy per particle class  (replace calibrated_E with true_showers_E for one class at a time)
#     Neutral hadrons (cls 2) are smeared with HCAL resolution 0.5/√E instead
#     of getting perfect true energy, since that is the physical limit.
NEUTRAL_HADRON_CLS = 2
res_pe_per_class = {}
for cls_idx, cls_name in class_names.items():
    df_pe_cls = sd_hgb.copy(deep=True)
    mask_cls = df_pe_cls["pid_4_class_true"] == cls_idx
    e_true_cls = df_pe_cls.loc[mask_cls, "true_showers_E"].values
    if cls_idx == NEUTRAL_HADRON_CLS:
        hcal_sigma = 0.5 * np.sqrt(np.clip(e_true_cls, 0, None))
        e_assigned = e_true_cls + np.random.normal(0, hcal_sigma)
        e_assigned  = np.clip(e_assigned, 0, None)
        label_suffix = " (HCAL smear)"
    else:
        e_assigned   = e_true_cls
        label_suffix = ""
    df_pe_cls.loc[mask_cls, "calibrated_E"] = e_assigned
    dic_pe_cls = calculate_event_mass_resolution(df_pe_cls, False, ML_pid=True)
    _, res_cls = mass_res_from_dic(dic_pe_cls)
    res_pe_per_class[cls_idx] = res_cls
    print(f"  Oracle perfect E ({cls_name:12s}{label_suffix}) | σ/μ={res_cls*100:.1f}%  (Δ={(mass_res_m - res_cls)*100:+.1f}%)")

# 2f. Perfect energy for CH with track only  (is_track_in_MC==1)
#     CH without a track are left with their original calibrated_E
df_pe_ch_track = sd_hgb.copy(deep=True)
mask_ch_track = (df_pe_ch_track["pid_4_class_true"] == 1) & (df_pe_ch_track["is_track_in_MC"] == 1)
df_pe_ch_track.loc[mask_ch_track, "calibrated_E"]    = df_pe_ch_track.loc[mask_ch_track, "true_showers_E"]
df_pe_ch_track.loc[mask_ch_track, "pred_pos_matched"] = df_pe_ch_track.loc[mask_ch_track, "true_pos"]
df_pe_ch_track.loc[mask_ch_track, "pred_pid_matched"] = df_pe_ch_track.loc[mask_ch_track, "pid_4_class_true"]
dic_pe_ch_track = calculate_event_mass_resolution(df_pe_ch_track, False, ML_pid=True)
_, res_pe_ch_track = mass_res_from_dic(dic_pe_ch_track)
n_ch_track   = mask_ch_track.sum()
n_ch_notrack = ((sd_hgb["pid_4_class_true"] == 1) & (sd_hgb["is_track_in_MC"] == 0)).sum()
print(f"  Oracle perfect E (ch.h w/ track, N={n_ch_track}) | σ/μ={res_pe_ch_track*100:.1f}%  (Δ={(mass_res_m - res_pe_ch_track)*100:+.1f}%)")
print(f"    (CH without track left as-is, N={n_ch_notrack})")

# 2g. Fix track failures at different energy thresholds
TRACK_FIX_THRESHOLDS = [20, 10, 5, 1]  # GeV
res_track_fix = {}
for thr in TRACK_FIX_THRESHOLDS:
    df_tf = sd_hgb.copy(deep=True)
    mask_tf = (
        (df_tf["pid_4_class_true"] == 1) &
        (df_tf["is_track_in_MC"] == 1) &
        (df_tf["is_track_correct"] == 0) &
        (df_tf["true_showers_E"] > thr)
    )
    n_fixed = mask_tf.sum()
    # Fix energy, direction, and PID — a correct track gives all three
    df_tf.loc[mask_tf, "calibrated_E"]    = df_tf.loc[mask_tf, "true_showers_E"]
    df_tf.loc[mask_tf, "pred_pos_matched"] = df_tf.loc[mask_tf, "true_pos"]
    df_tf.loc[mask_tf, "pred_pid_matched"] = df_tf.loc[mask_tf, "pid_4_class_true"]
    dic_tf = calculate_event_mass_resolution(df_tf, False, ML_pid=True)
    _, res_tf = mass_res_from_dic(dic_tf)
    res_track_fix[thr] = (res_tf, n_fixed)
    print(f"  Oracle fix track failures E>{thr} GeV (N={n_fixed}) | σ/μ={res_tf*100:.1f}%  (Δ={(mass_res_m - res_tf)*100:+.1f}%)")

# 2h. Improved clustering: recover missed particles with E_true > 1 GeV
#     These are rows where true_showers_E is not NaN but pred_showers_E is NaN.
#     We add them back with calibrated_E = true_showers_E and pred_pos_matched = true_pos,
#     and assign pred_pid_matched from the true class.
RECOVERY_THRESHOLD = 1.0  # GeV
missed_mask = (
    ~np.isnan(sd_hgb["true_showers_E"]) &
    np.isnan(sd_hgb["pred_showers_E"]) &
    (sd_hgb["true_showers_E"] > RECOVERY_THRESHOLD)
)
n_recovered = missed_mask.sum()
print(f"  Oracle improved clustering: recovering {n_recovered} missed particles with E_true > {RECOVERY_THRESHOLD} GeV")

# Smear recovered energies with HCAL stochastic resolution: σ/E = 0.5/√E  → σ = 0.5·√E
np.random.seed(42)
e_true_recovered = sd_hgb.loc[missed_mask, "true_showers_E"].values
hcal_sigma       = 0.5 * np.sqrt(e_true_recovered)          # absolute σ [GeV]
e_smeared        = e_true_recovered + np.random.normal(0, hcal_sigma)
e_smeared        = np.clip(e_smeared, 0, None)               # no negative energies

df_improved_clust = sd_hgb.copy(deep=True)
df_improved_clust.loc[missed_mask, "calibrated_E"]    = e_smeared
df_improved_clust.loc[missed_mask, "pred_showers_E"]  = e_smeared
df_improved_clust.loc[missed_mask, "pred_pos_matched"] = df_improved_clust.loc[missed_mask, "true_pos"]
df_improved_clust.loc[missed_mask, "pred_pid_matched"] = df_improved_clust.loc[missed_mask, "pid_4_class_true"]
dic_improved_clust = calculate_event_mass_resolution(df_improved_clust, False, ML_pid=True)
_, res_improved_clust = mass_res_from_dic(dic_improved_clust)
print(f"  Oracle improved clustering (HCAL smear 0.5/√E) | σ/μ={res_improved_clust*100:.1f}%  (Δ={(mass_res_m - res_improved_clust)*100:+.1f}%)")

# 2i. Full reconstruction oracles: fix energy + direction + PID + add back missed
#     for photons, neutral hadrons, and tracked CH.
def full_reco_oracle(df_base, cls_idx, smear_hcal=False, track_only=False):
    """Return (mass σ/μ, energy σ/μ) after perfect reconstruction of a particle class.
    Fixes existing reconstructed particles AND adds back missed ones (reco_showers_E fix)."""
    df = df_base.copy(deep=True)
    if track_only:
        mask_reco = (df["pid_4_class_true"] == cls_idx) & (df["is_track_in_MC"] == 1)
    else:
        mask_reco = df["pid_4_class_true"] == cls_idx

    # Only assign true energy where it is available (guard against NaN true_showers_E)
    mask_valid  = mask_reco & ~np.isnan(df["true_showers_E"])
    mask_missed = mask_valid & np.isnan(df["pred_showers_E"])

    e_true_valid = df.loc[mask_valid, "true_showers_E"].values
    if smear_hcal:
        hcal_sig = 0.5 * np.sqrt(np.clip(e_true_valid, 0, None))
        e_assign = np.clip(e_true_valid + np.random.normal(0, hcal_sig), 0, None)
    else:
        e_assign = e_true_valid.copy()

    df.loc[mask_valid, "calibrated_E"]    = e_assign
    df.loc[mask_valid, "pred_pos_matched"] = df.loc[mask_valid, "true_pos"]
    df.loc[mask_valid, "pred_pid_matched"] = cls_idx

    # Add back missed particles: fix pred_showers_E and reco_showers_E
    # so they are not filtered by df[df.reco_showers_E != 0.0]
    e_true_missed = df.loc[mask_missed, "true_showers_E"].values
    if smear_hcal:
        hcal_sig_m = 0.5 * np.sqrt(np.clip(e_true_missed, 0, None))
        e_missed   = np.clip(e_true_missed + np.random.normal(0, hcal_sig_m), 0, None)
    else:
        e_missed = e_true_missed.copy()
    df.loc[mask_missed, "calibrated_E"]   = e_missed
    df.loc[mask_missed, "pred_showers_E"] = e_missed
    df.loc[mask_missed, "reco_showers_E"] = e_true_missed   # must be non-zero to pass filter

    n_missed = mask_missed.sum()
    dic = calculate_event_mass_resolution(df, False, ML_pid=True)
    _, res_mass   = mass_res_from_dic(dic)
    _, res_energy = iqr_sigma_over_mu(dic["E_over_true"].numpy())
    return res_mass, res_energy, n_missed

# Photons
res_oracle_photon, Eres_oracle_photon, n_missed_photon = full_reco_oracle(sd_hgb, cls_idx=3, smear_hcal=False)
print(f"  Oracle perfect photon reco  (N missed recovered={n_missed_photon}) | mass σ/μ={res_oracle_photon*100:.1f}%  (Δ={(mass_res_m - res_oracle_photon)*100:+.1f}%)  | E σ/μ={Eres_oracle_photon*100:.1f}%  (Δ={(E_res_m - Eres_oracle_photon)*100:+.1f}%)")

# Neutral hadrons (HCAL smear)
res_oracle_nh, Eres_oracle_nh, n_missed_nh = full_reco_oracle(sd_hgb, cls_idx=2, smear_hcal=True)
print(f"  Oracle perfect NH reco      (N missed recovered={n_missed_nh}) | mass σ/μ={res_oracle_nh*100:.1f}%  (Δ={(mass_res_m - res_oracle_nh)*100:+.1f}%)  | E σ/μ={Eres_oracle_nh*100:.1f}%  (Δ={(E_res_m - Eres_oracle_nh)*100:+.1f}%)")

# Tracked CH (is_track_in_MC==1)
res_oracle_track, Eres_oracle_track, n_missed_track = full_reco_oracle(sd_hgb, cls_idx=1, smear_hcal=False, track_only=True)
print(f"  Oracle perfect track reco   (N missed recovered={n_missed_track}) | mass σ/μ={res_oracle_track*100:.1f}%  (Δ={(mass_res_m - res_oracle_track)*100:+.1f}%)  | E σ/μ={Eres_oracle_track*100:.1f}%  (Δ={(E_res_m - Eres_oracle_track)*100:+.1f}%)")

# Summary comparison table
track_fix_labels = [f"  Oracle: fix track failures E>{thr} GeV (N={res_track_fix[thr][1]})"
                    for thr in TRACK_FIX_THRESHOLDS]
track_fix_sigmas = [res_track_fix[thr][0] for thr in TRACK_FIX_THRESHOLDS]

full_reco_labels = [
    "  Oracle: perfect photon reco (E+dir+PID+missed)",
    "  Oracle: perfect NH reco (HCAL smear+dir+PID+missed)",
    "  Oracle: perfect track reco (E+dir+PID+missed, tracked CH)",
]
full_reco_sigmas       = [res_oracle_photon,  res_oracle_nh,  res_oracle_track]
full_reco_sigmas_energy = [Eres_oracle_photon, Eres_oracle_nh, Eres_oracle_track]

ablation_labels = (
    ["Pandora", "Oracle: perfect PID",
     "Oracle: perfect E (all)", "Oracle: perfect dir", "Oracle: perfect PID+E",
     f"Oracle: improved clustering E>{RECOVERY_THRESHOLD} GeV (HCAL smear)"]
    + full_reco_labels
    + track_fix_labels
    + [f"  Oracle: perfect E ({class_names[c]})" + (" (HCAL smear)" if c == NEUTRAL_HADRON_CLS else "")
       for c in sorted(res_pe_per_class)]
    + [f"  Oracle: perfect E (ch.h w/ track only)"]
)
ablation_sigmas = (
    [mass_res_p, res_pp, res_pe, res_pd, res_ppe, res_improved_clust]
    + full_reco_sigmas
    + track_fix_sigmas
    + [res_pe_per_class[c] for c in sorted(res_pe_per_class)]
    + [res_pe_ch_track]
)

# track-fix bars: gradient of reds from light (high thr) to dark (low thr)
track_fix_colors = ["#f4a582", "#d6604d", "#b2182b", "#67001f"]
# full-reco bars: distinct colors for photon / NH / track
full_reco_colors = ["#edc948", "#59a14f", "#4e79a7"]

n_global = 6 + len(full_reco_labels) + len(TRACK_FIX_THRESHOLDS)  # bars before the separator
colors_bar = (
    ["#0F4C5C", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
    + full_reco_colors
    + track_fix_colors
    + ["#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5", "#c49c94"]
    + ["#ffbb78"]
)
hatches = [""] * (len(ablation_labels) - 1) + ["//"]

n_bars = len(ablation_labels)
fig, ax = plt.subplots(figsize=(20, max(8, n_bars * 0.65)))
bars = ax.barh(ablation_labels, [s * 100 for s in ablation_sigmas],
               color=colors_bar, hatch=hatches, edgecolor="white")
# Annotate each bar with its value
for bar, val in zip(bars, ablation_sigmas):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%", va="center", fontsize=9)
ax.set_xlabel("σ/μ of M_pred/M_true  [%]")
ax.set_title("Oracle ablation: impact of each component on mass resolution")
ax.axvline(mass_res_m * 100, color="#E36414", linestyle="--", linewidth=1.2)
ax.axhline(n_global - 0.5, color="gray", linestyle=":", linewidth=0.8)  # separator between global and per-class
ax.grid(axis="x", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "oracle_ablation_summary.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR}/oracle_ablation_summary.pdf")

# --- Energy-resolution ablation (same structure, metric = E_reco/E_true σ/μ) ---
# Reuse same per-class perfect-E oracles but read the energy resolution instead
res_pe_per_class_energy = {}
for cls_idx, cls_name in class_names.items():
    df_pe_cls = sd_hgb.copy(deep=True)
    mask_cls = df_pe_cls["pid_4_class_true"] == cls_idx
    mask_valid_cls = mask_cls & ~np.isnan(df_pe_cls["true_showers_E"])
    e_true_cls = df_pe_cls.loc[mask_valid_cls, "true_showers_E"].values
    if cls_idx == NEUTRAL_HADRON_CLS:
        hcal_sigma = 0.5 * np.sqrt(np.clip(e_true_cls, 0, None))
        e_assigned = np.clip(e_true_cls + np.random.normal(0, hcal_sigma), 0, None)
    else:
        e_assigned = e_true_cls
    df_pe_cls.loc[mask_valid_cls, "calibrated_E"] = e_assigned
    dic_pe_cls_e = calculate_event_mass_resolution(df_pe_cls, False, ML_pid=True)
    _, res_e_cls = iqr_sigma_over_mu(dic_pe_cls_e["E_over_true"].numpy())
    res_pe_per_class_energy[cls_idx] = res_e_cls

# Perfect-E-all oracle energy resolution
dic_perfect_E_e = calculate_event_mass_resolution(df_perfect_E, False, ML_pid=True)
_, res_pe_energy = iqr_sigma_over_mu(dic_perfect_E_e["E_over_true"].numpy())

# Perfect-dir oracle energy resolution
dic_perfect_dir_e = calculate_event_mass_resolution(df_perfect_dir, False, ML_pid=True)
_, res_pd_energy = iqr_sigma_over_mu(dic_perfect_dir_e["E_over_true"].numpy())

# Perfect PID+E oracle energy resolution
dic_perfect_pid_E_e = calculate_event_mass_resolution(df_perfect_pid_E, False, perfect_pid=True, ML_pid=False)
_, res_ppe_energy = iqr_sigma_over_mu(dic_perfect_pid_E_e["E_over_true"].numpy())

# Improved clustering oracle energy resolution
dic_ic_e = calculate_event_mass_resolution(df_improved_clust, False, ML_pid=True)
_, res_ic_energy = iqr_sigma_over_mu(dic_ic_e["E_over_true"].numpy())

# CH w/ track oracle energy resolution
dic_pe_ch_track_e = calculate_event_mass_resolution(df_pe_ch_track, False, ML_pid=True)
_, res_pe_ch_track_energy = iqr_sigma_over_mu(dic_pe_ch_track_e["E_over_true"].numpy())

# Track-fix oracles energy resolution
res_track_fix_energy = {}
for thr in TRACK_FIX_THRESHOLDS:
    df_tf = sd_hgb.copy(deep=True)
    mask_tf = (
        (df_tf["pid_4_class_true"] == 1) &
        (df_tf["is_track_in_MC"] == 1) &
        (df_tf["is_track_correct"] == 0) &
        (df_tf["true_showers_E"] > thr)
    )
    df_tf.loc[mask_tf, "calibrated_E"]    = df_tf.loc[mask_tf, "true_showers_E"]
    df_tf.loc[mask_tf, "pred_pos_matched"] = df_tf.loc[mask_tf, "true_pos"]
    df_tf.loc[mask_tf, "pred_pid_matched"] = df_tf.loc[mask_tf, "pid_4_class_true"]
    dic_tf_e = calculate_event_mass_resolution(df_tf, False, ML_pid=True)
    _, res_tf_e = iqr_sigma_over_mu(dic_tf_e["E_over_true"].numpy())
    res_track_fix_energy[thr] = res_tf_e

ablation_sigmas_energy = (
    [E_res_p,
     iqr_sigma_over_mu(calculate_event_mass_resolution(sd_hgb, False, perfect_pid=True, ML_pid=False)["E_over_true"].numpy())[1],
     res_pe_energy, res_pd_energy, res_ppe_energy, res_ic_energy]
    + full_reco_sigmas_energy
    + [res_track_fix_energy[thr] for thr in TRACK_FIX_THRESHOLDS]
    + [res_pe_per_class_energy[c] for c in sorted(res_pe_per_class_energy)]
    + [res_pe_ch_track_energy]
)

fig2, ax2 = plt.subplots(figsize=(20, max(8, n_bars * 0.65)))
bars2 = ax2.barh(ablation_labels, [s * 100 for s in ablation_sigmas_energy],
                 color=colors_bar, hatch=hatches, edgecolor="white")
for bar, val in zip(bars2, ablation_sigmas_energy):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
             f"{val*100:.1f}%", va="center", fontsize=9)
ax2.set_xlabel("σ/μ of E_reco/E_true  [%]")
ax2.set_title("Oracle ablation: impact of each component on event energy resolution")
ax2.axvline(E_res_m * 100, color="#E36414", linestyle="--", linewidth=1.2)
ax2.axhline(n_global - 0.5, color="gray", linestyle=":", linewidth=0.8)
ax2.grid(axis="x", alpha=0.4)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "oracle_ablation_energy.pdf"), bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {OUTPUT_DIR}/oracle_ablation_energy.pdf")


# ---------------------------------------------------------------------------
# 3.  PER-CLASS BREAKDOWN
#     For each true particle class: energy resolution, PID purity, fake rate
# ---------------------------------------------------------------------------
print("\n=== Per-class breakdown ===")

rows = []
for cls_idx, pdg_list in class_to_pdg.items():
    name = class_names[cls_idx]
    # filter by true PID
    mask_m = sd_hgb["pid_4_class_true"] == cls_idx
    mask_p = sd_pandora["pid_4_class_true"] == cls_idx
    n_true = mask_m.sum()
    if n_true == 0:
        continue

    df_cls  = sd_hgb[mask_m]
    df_cls_p = sd_pandora[mask_p]

    # --- Clustering: fake rate (predicted but no true match) and miss rate ---
    # Fakes: pred_showers_E is not NaN but true_showers_E is NaN
    n_fakes_m = (np.isnan(df_cls["true_showers_E"]) & ~np.isnan(df_cls["pred_showers_E"])).sum()
    # Missed: true_showers_E is not NaN but pred_showers_E is NaN
    n_missed_m = (~np.isnan(df_cls["true_showers_E"]) & np.isnan(df_cls["pred_showers_E"])).sum()
    fake_rate_m   = n_fakes_m  / n_true * 100
    missed_rate_m = n_missed_m / n_true * 100

    # --- PID purity ---
    # Among particles that are reconstructed (matched), how often is PID correct?
    matched_mask = ~np.isnan(df_cls["pred_pid_matched"])
    n_matched = matched_mask.sum()
    if n_matched > 0:
        pid_correct = (df_cls.loc[matched_mask, "pred_pid_matched"] == cls_idx).sum()
        pid_purity_m = pid_correct / n_matched * 100
    else:
        pid_purity_m = np.nan

    # --- Energy resolution (per particle, not per event) ---
    df_matched = df_cls[~np.isnan(df_cls["calibrated_E"]) & ~np.isnan(df_cls["true_showers_E"])]
    if len(df_matched) > 0:
        e_ratio = df_matched["calibrated_E"].values / df_matched["true_showers_E"].values
        e_ratio = e_ratio[np.isfinite(e_ratio) & (df_matched["true_showers_E"].values > 0)]
        e_med, e_res = iqr_sigma_over_mu(e_ratio)
    else:
        e_med, e_res = np.nan, np.nan

    # --- Event-level mass resolution for this class only ---
    try:
        dic_cls = calculate_event_mass_resolution(df_cls, False, ML_pid=True)
        m_med, m_res = mass_res_from_dic(dic_cls)
    except Exception:
        m_med, m_res = np.nan, np.nan

    rows.append({
        "class": name,
        "n_true": int(n_true),
        "fake_rate_%": round(fake_rate_m, 1),
        "missed_rate_%": round(missed_rate_m, 1),
        "pid_purity_%": round(pid_purity_m, 1) if not np.isnan(pid_purity_m) else np.nan,
        "E_median": round(e_med, 3),
        "E_sigma_over_mu_%": round(e_res * 100, 1),
        "M_median": round(m_med, 3) if not np.isnan(m_med) else np.nan,
        "M_sigma_over_mu_%": round(m_res * 100, 1) if not np.isnan(m_res) else np.nan,
    })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))
csv_path = os.path.join(OUTPUT_DIR, "per_class_summary.csv")
df_summary.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ---------------------------------------------------------------------------
# 4.  PID CONFUSION  – energy-weighted mis-classification matrix
# ---------------------------------------------------------------------------
print("\n=== PID confusion (energy-weighted) ===")

matched_hgb = sd_hgb[~np.isnan(sd_hgb["pred_pid_matched"]) & ~np.isnan(sd_hgb["pid_4_class_true"])]
true_cls  = matched_hgb["pid_4_class_true"].astype(int).values
pred_cls  = matched_hgb["pred_pid_matched"].astype(int).values
energies  = matched_hgb["true_showers_E"].fillna(0).values

n_cls = 5
cm_counts  = np.zeros((n_cls, n_cls), dtype=float)
cm_energy  = np.zeros((n_cls, n_cls), dtype=float)

for t, p, e in zip(true_cls, pred_cls, energies):
    if 0 <= t < n_cls and 0 <= p < n_cls:
        cm_counts[t, p] += 1
        cm_energy[t, p] += e

# Normalise by true class (row-normalised)
row_sum_counts = cm_counts.sum(axis=1, keepdims=True)
row_sum_energy = cm_energy.sum(axis=1, keepdims=True)
cm_norm_counts = np.where(row_sum_counts > 0, cm_counts / row_sum_counts, 0)
cm_norm_energy = np.where(row_sum_energy > 0, cm_energy / row_sum_energy, 0)

labels = ["e", "ch.h", "neu.h", "γ", "μ"]
fig, axes = plt.subplots(1, 2, figsize=(24, 10))
for ax, cm, title in zip(axes,
                          [cm_norm_counts, cm_norm_energy],
                          ["PID confusion (count-normalised)", "PID confusion (energy-normalised)"]):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=12)
    fig.colorbar(im, ax=ax)
fig.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "pid_confusion_matrix.pdf")
fig.savefig(cm_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {cm_path}")


# ---------------------------------------------------------------------------
# 5.  ENERGY CORRECTION  vs  TRUE ENERGY  (per class)
# ---------------------------------------------------------------------------
print("\n=== Energy correction E_pred/E_true vs E_true ===")

energy_bins = np.exp(np.linspace(np.log(0.2), np.log(80), 15))
fig, axes = plt.subplots(2, 3, figsize=(30, 18))
axes = axes.flatten()

def _bin_energy_ratio(e_true, e_pred, energy_bins):
    """Return (bin_centers, medians, p16, p84) for e_pred/e_true binned by e_true."""
    bin_meds, bin_p16, bin_p84, bin_centers = [], [], [], []
    ratio = e_pred / e_true
    for i in range(len(energy_bins) - 1):
        in_bin = (e_true >= energy_bins[i]) & (e_true < energy_bins[i+1])
        if in_bin.sum() < 5:
            continue
        r = ratio[in_bin]
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        bin_meds.append(np.median(r))
        bin_p16.append(np.percentile(r, 16))
        bin_p84.append(np.percentile(r, 84))
        bin_centers.append(0.5 * (energy_bins[i] + energy_bins[i+1]))
    return (np.array(bin_centers), np.array(bin_meds),
            np.array(bin_p16),     np.array(bin_p84))

for ax_idx, (cls_idx, name) in enumerate(class_names.items()):
    ax = axes[ax_idx]

    # --- Model ---
    mask_m = (sd_hgb["pid_4_class_true"] == cls_idx) & \
             (~np.isnan(sd_hgb["calibrated_E"])) & \
             (~np.isnan(sd_hgb["true_showers_E"])) & \
             (sd_hgb["true_showers_E"] > 0)
    df_cls = sd_hgb[mask_m]

    # --- Pandora ---
    mask_p = (sd_pandora["pid_4_class_true"] == cls_idx) & \
             (~np.isnan(sd_pandora["pandora_calibrated_pfo"])) & \
             (~np.isnan(sd_pandora["true_showers_E"])) & \
             (sd_pandora["true_showers_E"] > 0)
    df_cls_p = sd_pandora[mask_p]

    if len(df_cls) == 0 and len(df_cls_p) == 0:
        ax.set_visible(False)
        continue

    if len(df_cls) > 0:
        bc_m, bm_m, b16_m, b84_m = _bin_energy_ratio(
            df_cls["true_showers_E"].values,
            df_cls["calibrated_E"].values,
            energy_bins,
        )
        ax.fill_between(bc_m, b16_m, b84_m, alpha=0.25, color="#E36414")
        ax.plot(bc_m, bm_m, "o-", color="#E36414", label=f"HitPF (N={len(df_cls)})", linewidth=1.5, markersize=4)

    if len(df_cls_p) > 0:
        bc_p, bm_p, b16_p, b84_p = _bin_energy_ratio(
            df_cls_p["true_showers_E"].values,
            df_cls_p["pandora_calibrated_pfo"].values,
            energy_bins,
        )
        ax.fill_between(bc_p, b16_p, b84_p, alpha=0.25, color="#0F4C5C")
        ax.plot(bc_p, bm_p, "s-", color="#0F4C5C", label=f"Pandora (N={len(df_cls_p)})", linewidth=1.5, markersize=4)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("E_true [GeV]")
    ax.set_ylabel("E_pred / E_true")
    ax.set_title(name)
    ax.set_ylim([0.3, 1.7])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

axes[-1].set_visible(False)
fig.suptitle("Per-particle energy correction vs true energy", fontsize=13)
fig.tight_layout()
ecor_path = os.path.join(OUTPUT_DIR, "energy_correction_per_class.pdf")
fig.savefig(ecor_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {ecor_path}")


# ---------------------------------------------------------------------------
# 6.  CLUSTERING QUALITY  – fake energy fraction and missed energy fraction
#     per event, shown as distribution
# ---------------------------------------------------------------------------
print("\n=== Clustering quality: fake / missed energy per event ===")

n_events = int(sd_hgb["number_batch"].max()) + 1
E_true_total  = np.zeros(n_events)
E_fake_model  = np.zeros(n_events)
E_missed_model = np.zeros(n_events)

for i, row in sd_hgb.iterrows():
    b = int(row["number_batch"])
    e_true = row["true_showers_E"]
    e_pred = row["pred_showers_E"]
    is_fake   = np.isnan(e_true) and not np.isnan(e_pred)
    is_missed = not np.isnan(e_true) and np.isnan(e_pred)
    if not np.isnan(e_true):
        E_true_total[b] += e_true
    if is_fake and not np.isnan(e_pred):
        E_fake_model[b] += e_pred
    if is_missed:
        E_missed_model[b] += e_true

valid = E_true_total > 0
fake_frac   = E_fake_model[valid]   / E_true_total[valid]
missed_frac = E_missed_model[valid] / E_true_total[valid]

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
axes[0].hist(fake_frac,   bins=50, range=(0, 0.5), color="#E36414", histtype="step", linewidth=2)
axes[0].set_xlabel("Fake energy / True event energy")
axes[0].set_title(f"Fake energy fraction  median={np.median(fake_frac):.3f}")
axes[0].axvline(np.median(fake_frac), color="#E36414", linestyle="--")
axes[0].grid(alpha=0.3)

axes[1].hist(missed_frac, bins=50, range=(0, 0.5), color="#0F4C5C", histtype="step", linewidth=2)
axes[1].set_xlabel("Missed energy / True event energy")
axes[1].set_title(f"Missed energy fraction  median={np.median(missed_frac):.3f}")
axes[1].axvline(np.median(missed_frac), color="#0F4C5C", linestyle="--")
axes[1].grid(alpha=0.3)

fig.suptitle("Clustering quality per event", fontsize=11)
fig.tight_layout()
clust_path = os.path.join(OUTPUT_DIR, "clustering_fake_missed_energy.pdf")
fig.savefig(clust_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {clust_path}")

print(f"\n  Fake energy fraction   median={np.median(fake_frac)*100:.2f}%  mean={np.mean(fake_frac)*100:.2f}%")
print(f"  Missed energy fraction median={np.median(missed_frac)*100:.2f}%  mean={np.mean(missed_frac)*100:.2f}%")


# ---------------------------------------------------------------------------
# 7.  TRACK-TO-MC MATCHING  –  impact of is_track_correct on CH classification
#     and energy scale
# ---------------------------------------------------------------------------
print("\n=== Track-to-MC matching analysis (charged hadrons) ===")

ch_mask = sd_hgb["pid_4_class_true"] == 1          # true charged hadrons
ch_has_track = ch_mask & (sd_hgb["is_track_in_MC"] == 1)
ch_no_track  = ch_mask & (sd_hgb["is_track_in_MC"] == 0)

# Split track-CH by whether the track was matched correctly
ch_track_correct   = ch_has_track & (sd_hgb["is_track_correct"] > 0)
ch_track_wrong     = ch_has_track & (sd_hgb["is_track_correct"] == 0)

n_ch_total         = ch_mask.sum()
n_ch_track_correct = ch_track_correct.sum()
n_ch_track_wrong   = ch_track_wrong.sum()
n_ch_no_track      = ch_no_track.sum()

print(f"  True CH total:              {n_ch_total}")
print(f"  With track, correct match:  {n_ch_track_correct}  ({n_ch_track_correct/n_ch_total*100:.1f}%)")
print(f"  With track, WRONG match:    {n_ch_track_wrong}   ({n_ch_track_wrong/n_ch_total*100:.1f}%)")
print(f"  No track in MC:             {n_ch_no_track}  ({n_ch_no_track/n_ch_total*100:.1f}%)")

# PID outcome per group
def pid_breakdown(mask, label):
    df_g = sd_hgb[mask & ~np.isnan(sd_hgb["pred_pid_matched"])]
    if len(df_g) == 0:
        print(f"  {label}: no matched particles")
        return
    counts = df_g["pred_pid_matched"].value_counts().sort_index()
    total  = len(df_g)
    parts  = ", ".join(f"{class_names.get(int(k), int(k))}={v/total*100:.1f}%" for k, v in counts.items())
    print(f"  {label} pred PID: {parts}")

pid_breakdown(ch_track_correct, "CH track correct")
pid_breakdown(ch_track_wrong,   "CH track WRONG ")
pid_breakdown(ch_no_track,      "CH no track    ")

# ── Diagnostic: why does fixing wrong tracks not help much? ──
df_wrong = sd_hgb[ch_track_wrong].copy()
df_wrong_reco = df_wrong[~np.isnan(df_wrong["calibrated_E"]) & ~np.isnan(df_wrong["true_showers_E"]) & (df_wrong["true_showers_E"] > 0)]
df_wrong_missed = df_wrong[np.isnan(df_wrong["pred_showers_E"]) & ~np.isnan(df_wrong["true_showers_E"])]

e_ratio_wrong = df_wrong_reco["calibrated_E"].values / df_wrong_reco["true_showers_E"].values
e_ratio_wrong = e_ratio_wrong[np.isfinite(e_ratio_wrong)]
med_wrong, res_wrong = iqr_sigma_over_mu(e_ratio_wrong)

print(f"\n  Diagnostic: wrong-track CH")
print(f"    Total wrong-track CH:           {len(df_wrong)}")
print(f"    Reconstructed (not missed):     {len(df_wrong_reco)}  ({len(df_wrong_reco)/max(len(df_wrong),1)*100:.1f}%)")
print(f"    Missed (pred_showers_E is NaN): {len(df_wrong_missed)}  ({len(df_wrong_missed)/max(len(df_wrong),1)*100:.1f}%)")
print(f"    E_pred/E_true for reconstructed wrong-track CH:  median={med_wrong:.3f}  σ/μ={res_wrong*100:.1f}%")
print(f"    True energy of wrong-track CH: mean={df_wrong['true_showers_E'].mean():.2f} GeV  median={df_wrong['true_showers_E'].median():.2f} GeV")
total_E_wrong = df_wrong["true_showers_E"].sum()
total_E_all   = sd_hgb["true_showers_E"].sum()
print(f"    Total true energy of wrong-track CH: {total_E_wrong:.1f} GeV  ({total_E_wrong/total_E_all*100:.2f}% of all true energy)")

# ── 7a. PID outcome breakdown (stacked bar) ──
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

groups      = ["CH: track correct", "CH: track wrong", "CH: no track (MC)"]
group_masks = [ch_track_correct,    ch_track_wrong,     ch_no_track]
pred_classes = [0, 1, 2, 3, 4]
class_colors = {"electron": "#4e79a7", "ch. hadron": "#f28e2b",
                "neu. hadron": "#59a14f", "photon": "#edc948", "muon": "#b07aa1"}

pid_fracs = {g: [] for g in groups}
for grp, msk in zip(groups, group_masks):
    df_g = sd_hgb[msk & ~np.isnan(sd_hgb["pred_pid_matched"])]
    total = len(df_g) if len(df_g) > 0 else 1
    for pc in pred_classes:
        pid_fracs[grp].append((df_g["pred_pid_matched"] == pc).sum() / total * 100)

x     = np.arange(len(groups))
width = 0.55
bottoms = np.zeros(len(groups))
ax = axes[0]
for pc in pred_classes:
    vals = [pid_fracs[g][pc] for g in groups]
    bars = ax.bar(x, vals, width, bottom=bottoms,
                  label=class_names[pc], color=list(class_colors.values())[pc])
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 2:
            ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9)
    bottoms += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(groups, rotation=15, ha="right")
ax.set_ylabel("Fraction of particles [%]")
ax.set_title("Predicted PID for true CH, split by track matching")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 115)
ax.grid(axis="y", alpha=0.3)

# ── 7b. Energy correction E_pred/E_true vs E_true  per track-match group ──
ax = axes[1]
group_styles = [
    ("CH: track correct", ch_track_correct, "#2ca02c", "o-"),
    ("CH: track wrong",   ch_track_wrong,   "#d62728", "s-"),
    ("CH: no track (MC)", ch_no_track,      "#9467bd", "^-"),
]
for label, msk, color, fmt in group_styles:
    df_g = sd_hgb[msk & ~np.isnan(sd_hgb["calibrated_E"]) & ~np.isnan(sd_hgb["true_showers_E"]) & (sd_hgb["true_showers_E"] > 0)]
    if len(df_g) < 10:
        continue
    bc, bm, b16, b84 = _bin_energy_ratio(df_g["true_showers_E"].values, df_g["calibrated_E"].values, energy_bins)
    ax.fill_between(bc, b16, b84, alpha=0.2, color=color)
    ax.plot(bc, bm, fmt, color=color, label=f"{label} (N={len(df_g)})", linewidth=1.5, markersize=4)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("E_true [GeV]")
ax.set_ylabel("E_pred / E_true")
ax.set_title("Energy correction for CH split by track matching")
ax.set_ylim([0.3, 1.7])
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.tight_layout()
track_path = os.path.join(OUTPUT_DIR, "track_matching_analysis.pdf")
fig.savefig(track_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {track_path}")

# ── 7c. Track failure rate vs true energy ──
#     For CH that should have a track (is_track_in_MC==1), what fraction
#     have a wrong match (is_track_correct==0)?  Binned by true energy.
df_ch_should_have_track = sd_hgb[ch_has_track & ~np.isnan(sd_hgb["true_showers_E"]) & (sd_hgb["true_showers_E"] > 0)].copy()

track_fail_rate, track_fail_centers = [], []
for i in range(len(energy_bins) - 1):
    in_bin = (df_ch_should_have_track["true_showers_E"] >= energy_bins[i]) & \
             (df_ch_should_have_track["true_showers_E"] <  energy_bins[i + 1])
    n_bin = in_bin.sum()
    if n_bin < 5:
        continue
    n_wrong = (df_ch_should_have_track.loc[in_bin, "is_track_correct"] == 0).sum()
    # Binomial error
    p = n_wrong / n_bin
    err = np.sqrt(p * (1 - p) / n_bin)
    track_fail_rate.append((p * 100, err * 100))
    track_fail_centers.append(0.5 * (energy_bins[i] + energy_bins[i + 1]))

track_fail_centers = np.array(track_fail_centers)
track_fail_vals    = np.array([r[0] for r in track_fail_rate])
track_fail_errs    = np.array([r[1] for r in track_fail_rate])

# Pandora equivalent
df_pan_ch = sd_pandora[
    (sd_pandora["pid_4_class_true"] == 1) &
    (sd_pandora["is_track_in_MC"] == 1) &
    ~np.isnan(sd_pandora["true_showers_E"]) &
    (sd_pandora["true_showers_E"] > 0)
].copy()

pan_fail_rate, pan_fail_centers = [], []
for i in range(len(energy_bins) - 1):
    in_bin = (df_pan_ch["true_showers_E"] >= energy_bins[i]) & \
             (df_pan_ch["true_showers_E"] <  energy_bins[i + 1])
    n_bin = in_bin.sum()
    if n_bin < 5:
        continue
    n_wrong = (df_pan_ch.loc[in_bin, "is_track_correct"] == 0).sum()
    p = n_wrong / n_bin
    err = np.sqrt(p * (1 - p) / n_bin)
    pan_fail_rate.append((p * 100, err * 100))
    pan_fail_centers.append(0.5 * (energy_bins[i] + energy_bins[i + 1]))

pan_fail_centers = np.array(pan_fail_centers)
pan_fail_vals    = np.array([r[0] for r in pan_fail_rate])
pan_fail_errs    = np.array([r[1] for r in pan_fail_rate])

fig, ax = plt.subplots(figsize=(14, 7))
ax.errorbar(track_fail_centers, track_fail_vals, yerr=track_fail_errs,
            fmt="o-", color="#E36414", linewidth=1.8, markersize=5,
            label=f"HitPF  (total N={len(df_ch_should_have_track)})")
if len(pan_fail_centers) > 0:
    ax.errorbar(pan_fail_centers, pan_fail_vals, yerr=pan_fail_errs,
                fmt="s-", color="#0F4C5C", linewidth=1.8, markersize=5,
                label=f"Pandora (total N={len(df_pan_ch)})")
ax.set_xscale("log")
ax.set_xlabel("E_true [GeV]")
ax.set_ylabel("Track failure rate [%]  (is_track_correct == 0)")
ax.set_title("Fraction of CH (is_track_in_MC=1) with wrong track match vs true energy")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)
fig.tight_layout()
fail_path = os.path.join(OUTPUT_DIR, "track_failure_rate_vs_energy.pdf")
fig.savefig(fail_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {fail_path}")

# ── 7e. Oracle: perfect track matching ──
#     For CH with is_track_in_MC==1 and is_track_correct==0, give true energy
df_perfect_track = sd_hgb.copy(deep=True)
df_perfect_track.loc[ch_track_wrong, "calibrated_E"]    = df_perfect_track.loc[ch_track_wrong, "true_showers_E"]
df_perfect_track.loc[ch_track_wrong, "pred_pos_matched"] = df_perfect_track.loc[ch_track_wrong, "true_pos"]
df_perfect_track.loc[ch_track_wrong, "pred_pid_matched"] = df_perfect_track.loc[ch_track_wrong, "pid_4_class_true"]
dic_perfect_track = calculate_event_mass_resolution(df_perfect_track, False, ML_pid=True)
_, res_perfect_track = mass_res_from_dic(dic_perfect_track)
print(f"\n  Oracle perfect track match | σ/μ={res_perfect_track*100:.1f}%  (Δ={(mass_res_m - res_perfect_track)*100:+.1f}%)")


# ---------------------------------------------------------------------------
# 8.  NEUTRAL ENERGY REGRESSION  –  true neutrals vs CH misidentified as neutral
# ---------------------------------------------------------------------------
print("\n=== Neutral energy regression ===")

# True neutrals (neu. hadron) that were predicted as neutral
true_neu_pred_neu = (sd_hgb["pid_4_class_true"] == 2) & (sd_hgb["pred_pid_matched"] == 2)
# True CH predicted as neutral (track-wrong or no-track → fell through as neutral)
ch_as_neu = (sd_hgb["pid_4_class_true"] == 1) & (sd_hgb["pred_pid_matched"] == 2)
# True photons predicted as neutral (γ→neu.h confusion)
gamma_as_neu = (sd_hgb["pid_4_class_true"] == 3) & (sd_hgb["pred_pid_matched"] == 2)
# All particles reconstructed as neutrals (regardless of truth)
all_pred_neu = sd_hgb["pred_pid_matched"] == 2

print(f"  Particles predicted as neutral hadron:")
print(f"    True neu. hadron → neu.h :  {true_neu_pred_neu.sum()}  ({true_neu_pred_neu.sum()/all_pred_neu.sum()*100:.1f}% of pred neu.h)")
print(f"    True CH          → neu.h :  {ch_as_neu.sum()}  ({ch_as_neu.sum()/all_pred_neu.sum()*100:.1f}% of pred neu.h)")
print(f"    True photon      → neu.h :  {gamma_as_neu.sum()}  ({gamma_as_neu.sum()/all_pred_neu.sum()*100:.1f}% of pred neu.h)")

# ── 8a. E_pred/E_true vs E_true  for each contamination source ──
fig, axes = plt.subplots(1, 2, figsize=(22, 8))
ax = axes[0]
neu_groups = [
    ("True neu.h → pred neu.h", true_neu_pred_neu, "#59a14f", "o-"),
    ("True CH → pred neu.h",    ch_as_neu,          "#d62728", "s-"),
    ("True γ → pred neu.h",     gamma_as_neu,        "#edc948", "^-"),
]
for label, msk, color, fmt in neu_groups:
    df_g = sd_hgb[msk & ~np.isnan(sd_hgb["calibrated_E"]) & ~np.isnan(sd_hgb["true_showers_E"]) & (sd_hgb["true_showers_E"] > 0)]
    if len(df_g) < 5:
        continue
    bc, bm, b16, b84 = _bin_energy_ratio(df_g["true_showers_E"].values, df_g["calibrated_E"].values, energy_bins)
    ax.fill_between(bc, b16, b84, alpha=0.2, color=color)
    ax.plot(bc, bm, fmt, color=color, label=f"{label} (N={len(df_g)})", linewidth=1.5, markersize=4)

# Pandora true neutrals for reference
df_pan_neu = sd_pandora[(sd_pandora["pid_4_class_true"] == 2) & ~np.isnan(sd_pandora["pandora_calibrated_pfo"]) & (sd_pandora["true_showers_E"] > 0)]
if len(df_pan_neu) > 5:
    bc, bm, b16, b84 = _bin_energy_ratio(df_pan_neu["true_showers_E"].values, df_pan_neu["pandora_calibrated_pfo"].values, energy_bins)
    ax.fill_between(bc, b16, b84, alpha=0.15, color="#0F4C5C")
    ax.plot(bc, bm, "D--", color="#0F4C5C", label=f"Pandora true neu.h (N={len(df_pan_neu)})", linewidth=1.5, markersize=4)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xscale("log")
ax.set_xlabel("E_true [GeV]")
ax.set_ylabel("E_pred / E_true")
ax.set_title("Energy correction: neutral-predicted particles by true species")
ax.set_ylim([0.3, 2.0])
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ── 8b. E_pred/E_true distribution (all energies) per group ──
ax = axes[1]
bins_ratio = np.linspace(0, 3, 80)
for label, msk, color, _ in neu_groups:
    df_g = sd_hgb[msk & ~np.isnan(sd_hgb["calibrated_E"]) & ~np.isnan(sd_hgb["true_showers_E"]) & (sd_hgb["true_showers_E"] > 0)]
    if len(df_g) < 5:
        continue
    ratio = df_g["calibrated_E"].values / df_g["true_showers_E"].values
    ratio = ratio[np.isfinite(ratio)]
    med, res = iqr_sigma_over_mu(ratio)
    ax.hist(ratio, bins=bins_ratio, histtype="step", color=color, linewidth=1.8,
            label=f"{label}\nmedian={med:.2f}  σ/μ={res*100:.1f}%", density=True)

if len(df_pan_neu) > 5:
    ratio_p = df_pan_neu["pandora_calibrated_pfo"].values / df_pan_neu["true_showers_E"].values
    ratio_p = ratio_p[np.isfinite(ratio_p)]
    med_p, res_p = iqr_sigma_over_mu(ratio_p)
    ax.hist(ratio_p, bins=bins_ratio, histtype="step", color="#0F4C5C", linewidth=1.8, linestyle="--",
            label=f"Pandora true neu.h\nmedian={med_p:.2f}  σ/μ={res_p*100:.1f}%", density=True)

ax.axvline(1.0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("E_pred / E_true")
ax.set_title("E_pred/E_true distribution for neutral-predicted particles")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([0, 3])

fig.tight_layout()
neu_path = os.path.join(OUTPUT_DIR, "neutral_energy_regression.pdf")
fig.savefig(neu_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {neu_path}")

# ── 8c. ECAL/HCAL split for neutral regression ──
print("\n  ECAL vs HCAL fraction for neutral-predicted particles:")
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

for ax, (e_col, label_col) in zip(axes, [("ECAL_hits", "ECAL fraction"), ("HCAL_hits", "HCAL fraction")]):
    for lbl, msk, color, _ in neu_groups:
        df_g = sd_hgb[msk & ~np.isnan(sd_hgb["calibrated_E"]) & (sd_hgb["true_showers_E"] > 0)]
        if len(df_g) < 5 or e_col not in sd_hgb.columns:
            continue
        total_e = df_g["ECAL_hits"].values + df_g["HCAL_hits"].values
        frac    = df_g[e_col].values / np.where(total_e > 0, total_e, np.nan)
        frac    = frac[np.isfinite(frac)]
        ax.hist(frac, bins=40, range=(0, 1), histtype="step", color=color,
                linewidth=1.8, label=f"{lbl} (N={len(frac)})", density=True)
    ax.set_xlabel(label_col)
    ax.set_ylabel("Density")
    ax.set_title(f"{label_col} for neutral-predicted particles")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.tight_layout()
calorimeter_path = os.path.join(OUTPUT_DIR, "neutral_ecal_hcal_fractions.pdf")
fig.savefig(calorimeter_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {calorimeter_path}")


# ---------------------------------------------------------------------------
# 9.  NEUTRAL HADRON CALIBRATION AT HCAL RESOLUTION LIMIT?
#     Compare current E_pred/E_true σ vs expected 0.5/√E per energy bin.
# ---------------------------------------------------------------------------
print("\n=== Is neutral hadron calibration at the HCAL resolution limit? ===")

df_neu_reco = sd_hgb[
    true_neu_pred_neu &
    ~np.isnan(sd_hgb["calibrated_E"]) &
    ~np.isnan(sd_hgb["true_showers_E"]) &
    (sd_hgb["true_showers_E"] > 0)
].copy()

# Bin σ(E_pred/E_true) vs E_true and compare to HCAL stochastic term
sigma_model, sigma_hcal, bin_centers_neu = [], [], []
for i in range(len(energy_bins) - 1):
    in_bin = (df_neu_reco["true_showers_E"] >= energy_bins[i]) & \
             (df_neu_reco["true_showers_E"] <  energy_bins[i + 1])
    if in_bin.sum() < 10:
        continue
    e_true_bin = df_neu_reco.loc[in_bin, "true_showers_E"].values
    ratio = df_neu_reco.loc[in_bin, "calibrated_E"].values / e_true_bin
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) < 5:
        continue
    p16, p84 = np.percentile(ratio, 16), np.percentile(ratio, 84)
    med = np.median(ratio)
    # σ/E in this bin (IQR-based)
    sigma_over_E = (p84 - p16) / (2 * med) if med > 0 else np.nan
    e_mid = 0.5 * (energy_bins[i] + energy_bins[i + 1])
    sigma_model.append(sigma_over_E)
    sigma_hcal.append(0.5 / np.sqrt(e_mid))   # expected HCAL stochastic
    bin_centers_neu.append(e_mid)
    print(f"    E={e_mid:.1f} GeV  σ/E model={sigma_over_E*100:.1f}%  HCAL limit={0.5/np.sqrt(e_mid)*100:.1f}%"
          f"  ratio={sigma_over_E/(0.5/np.sqrt(e_mid)):.2f}x")

sigma_model    = np.array(sigma_model)
sigma_hcal     = np.array(sigma_hcal)
bin_centers_neu = np.array(bin_centers_neu)

# Pandora neutral hadrons for comparison
df_pan_neu_reco = sd_pandora[
    (sd_pandora["pid_4_class_true"] == 2) &
    ~np.isnan(sd_pandora["pandora_calibrated_pfo"]) &
    (sd_pandora["true_showers_E"] > 0)
].copy()

sigma_pandora, bin_centers_pan = [], []
for i in range(len(energy_bins) - 1):
    in_bin = (df_pan_neu_reco["true_showers_E"] >= energy_bins[i]) & \
             (df_pan_neu_reco["true_showers_E"] <  energy_bins[i + 1])
    if in_bin.sum() < 10:
        continue
    e_true_bin = df_pan_neu_reco.loc[in_bin, "true_showers_E"].values
    ratio = df_pan_neu_reco.loc[in_bin, "pandora_calibrated_pfo"].values / e_true_bin
    ratio = ratio[np.isfinite(ratio)]
    if len(ratio) < 5:
        continue
    p16, p84 = np.percentile(ratio, 16), np.percentile(ratio, 84)
    med = np.median(ratio)
    sigma_pandora.append((p84 - p16) / (2 * med) if med > 0 else np.nan)
    bin_centers_pan.append(0.5 * (energy_bins[i] + energy_bins[i + 1]))

sigma_pandora    = np.array(sigma_pandora)
bin_centers_pan  = np.array(bin_centers_pan)

# Dense curve for HCAL limit
e_curve = np.logspace(np.log10(energy_bins[0]), np.log10(energy_bins[-1]), 200)

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# Left: σ/E vs E_true
ax = axes[0]
ax.plot(bin_centers_neu, sigma_model * 100,  "o-", color="#E36414", linewidth=1.8,
        markersize=5, label="HitPF (true neu.h → pred neu.h)")
if len(sigma_pandora) > 0:
    ax.plot(bin_centers_pan, sigma_pandora * 100, "s-", color="#0F4C5C", linewidth=1.8,
            markersize=5, label="Pandora (true neu.h)")
ax.plot(e_curve, 0.5 / np.sqrt(e_curve) * 100, "k--", linewidth=1.5,
        label="HCAL limit: 0.5/√E")
ax.set_xscale("log")
ax.set_xlabel("E_true [GeV]")
ax.set_ylabel("σ(E_pred/E_true)  [%]")
ax.set_title("Neutral hadron energy resolution vs HCAL limit")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# Right: ratio model_σ / HCAL_limit — how many times worse than limit?
ax = axes[1]
ratio_to_limit = sigma_model / sigma_hcal
ax.plot(bin_centers_neu, ratio_to_limit, "o-", color="#E36414", linewidth=1.8,
        markersize=5, label="HitPF / HCAL limit")
if len(sigma_pandora) > 0:
    # interpolate HCAL limit at pandora bin centers
    hcal_at_pan = 0.5 / np.sqrt(bin_centers_pan)
    ax.plot(bin_centers_pan, sigma_pandora / hcal_at_pan, "s-", color="#0F4C5C",
            linewidth=1.8, markersize=5, label="Pandora / HCAL limit")
ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="At HCAL limit")
ax.set_xscale("log")
ax.set_xlabel("E_true [GeV]")
ax.set_ylabel("σ_model / σ_HCAL")
ax.set_title("How many times worse than HCAL stochastic limit?")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

fig.tight_layout()
hcal_limit_path = os.path.join(OUTPUT_DIR, "neutral_hcal_resolution_limit.pdf")
fig.savefig(hcal_limit_path, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {hcal_limit_path}")


# ---------------------------------------------------------------------------
# 10. ORACLE: ADD BACK MISSED NEUTRAL HADRONS WITH HCAL SMEARING
#     Fix the previous oracle by also setting reco_showers_E so missed particles
#     are not filtered out by the df[df.reco_showers_E != 0.0] cut inside
#     calculate_event_mass_resolution.
# ---------------------------------------------------------------------------
print("\n=== Oracle: add back missed neutral hadrons (HCAL smear) ===")

missed_neu_mask = (
    (sd_hgb["pid_4_class_true"] == 2) &
    ~np.isnan(sd_hgb["true_showers_E"]) &
    np.isnan(sd_hgb["pred_showers_E"]) &
    (sd_hgb["true_showers_E"] > RECOVERY_THRESHOLD)
)
n_missed_neu = missed_neu_mask.sum()
print(f"  Missed neutral hadrons with E_true > {RECOVERY_THRESHOLD} GeV: {n_missed_neu}")

e_true_missed_neu = sd_hgb.loc[missed_neu_mask, "true_showers_E"].values
hcal_sigma_neu    = 0.5 * np.sqrt(np.clip(e_true_missed_neu, 0, None))
e_smeared_neu     = np.clip(e_true_missed_neu + np.random.normal(0, hcal_sigma_neu), 0, None)

df_oracle_neu = sd_hgb.copy(deep=True)
df_oracle_neu.loc[missed_neu_mask, "calibrated_E"]    = e_smeared_neu
df_oracle_neu.loc[missed_neu_mask, "pred_showers_E"]  = e_smeared_neu
# reco_showers_E must be non-zero to pass the filter in calculate_event_mass_resolution
df_oracle_neu.loc[missed_neu_mask, "reco_showers_E"]  = e_true_missed_neu
df_oracle_neu.loc[missed_neu_mask, "pred_pos_matched"] = df_oracle_neu.loc[missed_neu_mask, "true_pos"]
df_oracle_neu.loc[missed_neu_mask, "pred_pid_matched"] = 2  # neutral hadron

dic_oracle_neu = calculate_event_mass_resolution(df_oracle_neu, False, ML_pid=True)
_, res_oracle_neu = mass_res_from_dic(dic_oracle_neu)
print(f"  Oracle add missed neu.h (HCAL smear) | σ/μ={res_oracle_neu*100:.1f}%  (Δ={(mass_res_m - res_oracle_neu)*100:+.1f}%)")


# ---------------------------------------------------------------------------
# 11.  SUMMARY PRINT
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("POST-MORTEM SUMMARY")
print("="*65)
print(f"Baseline model   M σ/μ = {mass_res_m*100:.1f}%  (Pandora: {mass_res_p*100:.1f}%)")
print(f"")
print("  Impact of each component on mass resolution (σ/μ):")
delta_pid  = mass_res_m - res_pp
delta_E    = mass_res_m - res_pe
delta_dir  = mass_res_m - res_pd
delta_pidE = mass_res_m - res_ppe
print(f"  Fixing PID alone  → Δσ/μ = {delta_pid*100:+.1f}%  (new σ/μ={res_pp*100:.1f}%)")
print(f"  Fixing Energy alone (all) → Δσ/μ = {delta_E*100:+.1f}%  (new σ/μ={res_pe*100:.1f}%)")
for c, r in res_pe_per_class.items():
    print(f"    Fixing E ({class_names[c]:12s}) → Δσ/μ = {(mass_res_m-r)*100:+.1f}%  (new σ/μ={r*100:.1f}%)")
print(f"    Fixing E (ch.h w/ track only) → Δσ/μ = {(mass_res_m-res_pe_ch_track)*100:+.1f}%  (new σ/μ={res_pe_ch_track*100:.1f}%)")
print(f"  Fixing track matching (wrong→true E) → Δσ/μ = {(mass_res_m-res_perfect_track)*100:+.1f}%  (new σ/μ={res_perfect_track*100:.1f}%)")
print(f"  Improved clustering E>{RECOVERY_THRESHOLD} GeV (HCAL smear 0.5/√E) → Δσ/μ = {(mass_res_m-res_improved_clust)*100:+.1f}%  (new σ/μ={res_improved_clust*100:.1f}%)")
print(f"  Add missed neu.h    E>{RECOVERY_THRESHOLD} GeV (HCAL smear, reco_E fixed) → Δσ/μ = {(mass_res_m-res_oracle_neu)*100:+.1f}%  (new σ/μ={res_oracle_neu*100:.1f}%)")
for thr in TRACK_FIX_THRESHOLDS:
    r, n = res_track_fix[thr]
    print(f"  Fix track failures E>{thr:2d} GeV (N={n}) → Δσ/μ = {(mass_res_m-r)*100:+.1f}%  (new σ/μ={r*100:.1f}%)")
print(f"  Fixing Dir. alone → Δσ/μ = {delta_dir*100:+.1f}%  (new σ/μ={res_pd*100:.1f}%)")
print(f"  Fixing PID+E      → Δσ/μ = {delta_pidE*100:+.1f}%  (new σ/μ={res_ppe*100:.1f}%)")
print(f"")
print(f"  Clustering (model-level):")
print(f"    Fake energy fraction   median = {np.median(fake_frac)*100:.2f}%")
print(f"    Missed energy fraction median = {np.median(missed_frac)*100:.2f}%")
print(f"")
print("  Per-class PID purity & energy resolution:")
for _, row in df_summary.iterrows():
    print(f"    {row['class']:12s}  PID purity={row['pid_purity_%']}%  "
          f"E σ/μ={row['E_sigma_over_mu_%']}%  "
          f"fake={row['fake_rate_%']}%  missed={row['missed_rate_%']}%")
print("="*65)
print(f"\nAll plots saved to: {OUTPUT_DIR}")
