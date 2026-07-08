#!/usr/bin/env python3
"""
3-D visualisation of UOT track-hit matching results.

For each event (or a selected list) produces a side-by-side figure:
  Left  — UOT assignment:  hits coloured by assigned track (or grey = neutral)
  Right — MC truth:        hits coloured by truth label (neutral / charged)

Hit positions are the raw (x, y, z) calorimeter coordinates in mm.
Tracks are drawn as arrows from the calo reference point along the
calo-face momentum direction (scaled to a fixed length for visibility).

Usage (inside gatr:v9 container):
  python scripts/uot_3d_visualisation.py [--events 0 1 2 ...] [--out-dir ...]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import awkward as ak
from scipy.special import logsumexp as sci_logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.shower_cost import (
    shower_cost as _shower_cost_fn,
    cld_combined_params, cld_ecal_params, cld_hcal_params,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ── constants ─────────────────────────────────────────────────────────────────
M_PION   = 0.13957
EPS_SAFE = 1e-30
FLOAT64  = np.float64

DEFAULT_PARQUET = (
    "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/"
    "Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10601.parquet"
)
DEFAULT_OUT = "uot_matching_results"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data helpers (same as main script)
# ═══════════════════════════════════════════════════════════════════════════════

def safe_eta_phi(px, py, pz, f=1e-6):
    rT  = np.sqrt(px**2 + py**2)
    th  = np.arctan2(rT + f, pz)
    eta = -np.log(np.tan(0.5 * th) + f)
    phi = np.arctan2(py, px)
    return eta.astype(np.float32), phi.astype(np.float32)


def load_event(data, idx):
    return {field: np.array(data[field][idx]) for field in data.fields}


def extract_tracks(ev):
    X = ev.get("X_track", np.empty((0, 25)))
    if len(X) == 0:
        return None
    p_mag = X[:, 5].astype(np.float32)
    eta, phi = safe_eta_phi(X[:, 22], X[:, 23], X[:, 24])
    E_exp = np.sqrt(p_mag**2 + M_PION**2).astype(np.float32)
    # calo reference point (mm) and calo-face momentum direction (unit)
    ref_xyz  = X[:, 12:15].astype(np.float32)       # [T, 3]
    dir_calo = X[:, 22:25].astype(np.float32)        # [T, 3]
    norm = np.linalg.norm(dir_calo, axis=1, keepdims=True) + 1e-8
    dir_unit = dir_calo / norm                        # unit vectors
    return dict(eta=eta, phi=phi, p=p_mag, E_exp=E_exp,
                ref_xyz=ref_xyz, dir_unit=dir_unit,
                gen_link=np.array(ev["ygen_track"], dtype=np.int64).flatten())


def extract_hits(ev):
    X = ev.get("X_hit", np.empty((0, 12)))
    if len(X) == 0:
        return None
    hit_type = (X[:, 10] + 1).astype(np.int32)
    hit_E    = X[:, 5].astype(np.float32)
    hit_xyz  = X[:, 6:9].astype(np.float32)          # [H, 3]
    eta, phi = safe_eta_phi(hit_xyz[:, 0], hit_xyz[:, 1], hit_xyz[:, 2])
    gen_link = np.array(ev["ygen_hit"], dtype=np.int64).flatten()
    cal_mask = (hit_type >= 2) & (hit_type <= 3) & (hit_E > 0)
    return dict(eta=eta[cal_mask], phi=phi[cal_mask],
                xyz=hit_xyz[cal_mask],   # kept for plotting AND cost computation
                E=hit_E[cal_mask],
                type=hit_type[cal_mask],
                gen_link=gen_link[cal_mask])


def get_truth_neutrality(tracks, hits):
    """
    Particle-flow definition: a hit is 'neutral' if its linked MC particle
    does NOT appear in ygen_track (i.e. it belongs to a neutral particle or
    an unlinked hit where ygen_hit == 0).

    Parquet convention: ygen_hit == 0 → dummy X_gen row (no real MC particle).
    Any ygen_hit > 0 that is NOT in the set of track-linked particle indices
    is also neutral (e.g. π⁰, γ, K⁰L, n).
    """
    if tracks is None or len(tracks["gen_link"]) == 0:
        return np.ones(len(hits["gen_link"]), dtype=bool)
    # particles that leave a track → charged
    charged_mc = set(int(x) for x in tracks["gen_link"] if x > 0)
    return np.array([(int(g) not in charged_mc) for g in hits["gen_link"]], dtype=bool)


# ═══════════════════════════════════════════════════════════════════════════════
#  UOT Sinkhorn (same as main script)
# ═══════════════════════════════════════════════════════════════════════════════

def _decompose(tracks, hits):
    """Return (d_3D, d_par, d_perp) arrays [T, H] in mm."""
    delta      = hits["xyz"][None, :, :] - tracks["ref_xyz"][:, None, :]   # [T,H,3]
    d_3D       = np.sqrt(np.sum(delta**2, axis=2))
    d_par      = np.sum(delta * tracks["dir_unit"][:, None, :], axis=2)    # signed
    d_perp_vec = delta - d_par[:, :, None] * tracks["dir_unit"][:, None, :]
    d_perp     = np.sqrt(np.sum(d_perp_vec**2, axis=2))
    return d_3D, d_par, d_perp


def _open_minimum(C, d_3D, cost_fn, n_min_hits, C_inf):
    """Guarantee n_min_hits reachable hits per track by un-blocking nearest."""
    for k in range(C.shape[0]):
        if int((C[k] < C_inf * 0.5).sum()) < n_min_hits:
            nearest = np.argsort(d_3D[k])[:n_min_hits]
            C[k, nearest] = cost_fn(k, nearest)
    return C


def isotropic_cost(tracks, hits,
                   sigma_perp=300.0, max_d3D=6000.0,
                   n_min_hits=20, C_inf=1e6):
    """
    Cost = (d_perp / σ_perp)²   [purely transverse, isotropic along track axis]

    Hard cutoff: d_3D > max_d3D → C_inf
    """
    d_3D, d_par, d_perp = _decompose(tracks, hits)
    C = (d_perp / sigma_perp) ** 2
    C[d_3D > max_d3D] = C_inf
    return _open_minimum(C, d_3D,
                         lambda k, idx: (d_perp[k, idx] / sigma_perp) ** 2,
                         n_min_hits, C_inf)


def anisotropic_cost(tracks, hits,
                     sigma_perp=300.0, sigma_par=4000.0, max_d3D=6000.0,
                     n_min_hits=20, C_inf=1e6):
    """
    Anisotropic ellipsoidal cost aligned with the shower axis:

        C[k,j] = (d_perp / σ_perp)²  +  (d_par / σ_par)²

    σ_par >> σ_perp  →  moving *along* the track direction is cheap,
                         moving *sideways* is expensive.

    Negative d_par (hit behind the calo entry point) is treated the same
    as positive (symmetric along-axis penalty) since some ECAL hits may
    sit slightly before the extrapolated entry point due to geometry.

    Hard cutoff: d_3D > max_d3D → C_inf
    """
    d_3D, d_par, d_perp = _decompose(tracks, hits)
    C = (d_perp / sigma_perp) ** 2 + (d_par / sigma_par) ** 2
    C[d_3D > max_d3D] = C_inf
    return _open_minimum(
        C, d_3D,
        lambda k, idx: ((d_perp[k, idx] / sigma_perp) ** 2
                        + (d_par[k, idx] / sigma_par) ** 2),
        n_min_hits, C_inf,
    )


# keep old name as alias so existing calls don't break
combined_cost = isotropic_cost


def sinkhorn_uot(mu, nu, C, eps=0.02, tau=0.20, n_iter=300, tol=1e-6):
    mu64, nu64, C64 = mu.astype(FLOAT64), nu.astype(FLOAT64), C.astype(FLOAT64)
    log_mu, log_nu  = np.log(mu64 + EPS_SAFE), np.log(nu64 + EPS_SAFE)
    M     = -C64 / eps
    alpha = tau / (tau + eps)
    a = np.zeros(len(mu), dtype=FLOAT64)
    b = np.zeros(len(nu), dtype=FLOAT64)
    for it in range(n_iter):
        a_new = alpha * (log_mu - sci_logsumexp(b[None, :] + M, axis=1))
        b_new = alpha * (log_nu - sci_logsumexp(a_new[:, None] + M, axis=0))
        if it > 0 and max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b))) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return (a[:, None] + b[None, :] + M).astype(np.float32)


def run_uot(tracks, hits, eps=0.02, tau=0.20, n_iter=300,
            sigma_perp=300.0, max_d3D=6000.0, n_min_hits=20,
            cost_mode="anisotropic", sigma_par=4000.0,
            shower_params=None):
    """Return per-hit assignment (0-based track index or -1) and transport fractions.

    cost_mode : "isotropic"   → cost = (d_perp/σ_perp)²
                "anisotropic" → cost = (d_perp/σ_perp)² + (d_par/σ_par)²
                "shower"      → physics-motivated CLD shower cone cost
    """
    n_hits = len(hits["E"])
    if tracks is None or len(tracks["eta"]) == 0:
        return np.full(n_hits, -1, dtype=np.int32), np.zeros(n_hits, np.float32)

    if cost_mode == "shower":
        params = shower_params if shower_params is not None else cld_combined_params()
        C = _shower_cost_fn(tracks, hits, params)
    elif cost_mode == "anisotropic":
        C = anisotropic_cost(tracks, hits, sigma_perp=sigma_perp,
                             sigma_par=sigma_par, max_d3D=max_d3D,
                             n_min_hits=n_min_hits)
    else:
        C = isotropic_cost(tracks, hits, sigma_perp=sigma_perp,
                           max_d3D=max_d3D, n_min_hits=n_min_hits)
    mu   = tracks["E_exp"]
    nu   = hits["E"]
    logT = sinkhorn_uot(mu, nu, C, eps=eps, tau=tau, n_iter=n_iter)
    T    = np.exp(logT)

    frac = (T.sum(0) / (nu + 1e-10)).astype(np.float32)

    # Adaptive energy-balance threshold
    E_neutral_exp = max(0.0, float(nu.sum()) - float(mu.sum()))
    if E_neutral_exp <= 0:
        return np.zeros(n_hits, dtype=np.int32), frac   # all charged

    order   = np.argsort(frac)
    cum_E   = np.cumsum(nu[order])
    idx     = min(np.searchsorted(cum_E, E_neutral_exp), n_hits - 1)
    thr     = float(frac[order[idx]])

    best_track = np.argmax(T, axis=0).astype(np.int32)
    # thr + tiny: out-of-cone hits have frac==0 exactly (C_inf underflows exp)
    assignment = np.where(frac < thr + 1e-9, np.int32(-1), best_track).copy()

    # ── Per-track energy cap: ratio must not exceed 1 ────────────────────────
    # If a track absorbed more energy than expected, strip the hits with the
    # lowest transport fraction (most weakly associated) until within budget.
    for k in range(len(mu)):
        mask_k  = assignment == k
        E_k     = float(nu[mask_k].sum())
        E_exp_k = float(mu[k])
        if E_k <= E_exp_k:
            continue
        # Sort assigned hits by frac ascending (weakest first)
        idxs  = np.where(mask_k)[0]
        order_k = idxs[np.argsort(frac[idxs])]
        for j in order_k:
            if E_k <= E_exp_k:
                break
            E_k -= float(nu[j])
            assignment[j] = -1   # demote to neutral

    return assignment, frac


# ═══════════════════════════════════════════════════════════════════════════════
#  Colour helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Palette for up to ~30 tracks — distinct, perceptually uniform
_TAB20 = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)

NEUTRAL_COLOR  = "#B0B0B0"    # light grey  — unmatched / neutral
TRACK_COLOR    = "#000000"    # black       — track markers
CHARGED_COLOR  = "#E8413A"    # red         — truth charged
TRUE_NEU_COLOR = "#4C9BE8"    # blue        — truth neutral


def assignment_colors(assignment, n_tracks):
    """Map UOT assignment array → RGBA colour per hit."""
    palette = _TAB20[:max(n_tracks, 1)]
    colors  = np.empty((len(assignment), 4))
    for j, a in enumerate(assignment):
        if a < 0:
            colors[j] = mcolors.to_rgba(NEUTRAL_COLOR, alpha=0.35)
        else:
            colors[j] = mcolors.to_rgba(palette[a % len(palette)], alpha=0.70)
    return colors


def truth_colors(truth_neutral):
    """Map truth boolean array → RGBA colour per hit."""
    colors = np.empty((len(truth_neutral), 4))
    for j, neu in enumerate(truth_neutral):
        if neu:
            colors[j] = mcolors.to_rgba(TRUE_NEU_COLOR, alpha=0.35)
        else:
            colors[j] = mcolors.to_rgba(CHARGED_COLOR,  alpha=0.70)
    return colors


# ═══════════════════════════════════════════════════════════════════════════════
#  3-D plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _scatter3d(ax, xyz, colors, sizes, zorder=1):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
               c=colors, s=sizes, linewidths=0, depthshade=True, zorder=zorder)


def _draw_tracks(ax, tracks, palette, arrow_len=400.0):
    """
    Draw each track as a coloured arrow: origin = calo reference point,
    direction = calo-face momentum unit vector, scaled to arrow_len mm.
    Also mark the reference point with a larger marker.
    """
    if tracks is None:
        return
    for k, (ref, d) in enumerate(zip(tracks["ref_xyz"], tracks["dir_unit"])):
        col = palette[k % len(palette)]
        end = ref + d * arrow_len
        ax.plot([ref[0], end[0]], [ref[1], end[1]], [ref[2], end[2]],
                color=col, lw=2.0, zorder=5)
        ax.scatter(*ref, c=[col], s=80, marker="^",
                   edgecolors="k", linewidths=0.5, zorder=6)


def _set_axes(ax, xyz_hits, tracks, title):
    """Equal-aspect 3-D axes with labels."""
    # Combine hit and track reference positions for bounding box
    pts = xyz_hits.copy()
    if tracks is not None:
        pts = np.vstack([pts, tracks["ref_xyz"]])

    lo = pts.min(axis=0) - 100
    hi = pts.max(axis=0) + 100
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo).max()

    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_xlabel("x (mm)", fontsize=7, labelpad=2)
    ax.set_ylabel("y (mm)", fontsize=7, labelpad=2)
    ax.set_zlabel("z (mm)", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9, pad=4)


# ═══════════════════════════════════════════════════════════════════════════════
#  Legend patches
# ═══════════════════════════════════════════════════════════════════════════════

def _uot_legend(ax, n_tracks, palette):
    from matplotlib.patches import Patch
    handles = [Patch(color=NEUTRAL_COLOR, alpha=0.6, label="Neutral (UOT)")]
    for k in range(min(n_tracks, 10)):
        handles.append(Patch(color=palette[k % len(palette)],
                             label=f"Track {k}"))
    if n_tracks > 10:
        handles.append(Patch(color="white", label=f"… +{n_tracks-10} more"))
    ax.legend(handles=handles, loc="upper left", fontsize=6,
              framealpha=0.6, markerscale=0.7)


def _truth_legend(ax):
    from matplotlib.patches import Patch
    handles = [
        Patch(color=TRUE_NEU_COLOR, alpha=0.6, label="Neutral (truth)"),
        Patch(color=CHARGED_COLOR,  alpha=0.8, label="Charged (truth)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.6)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main plot function
# ═══════════════════════════════════════════════════════════════════════════════

def plot_event_3d(event_idx, tracks, hits, assignment, truth_neutral,
                  out_dir, eps, tau, elev=25, azim=45, dpi=150):
    """
    Two-panel 3-D figure: [UOT assignment | MC truth].
    Returns path to saved file.
    """
    xyz  = hits["xyz"]                          # [H, 3]
    sz   = np.clip(hits["E"] * 25, 3, 120)     # marker size ∝ energy

    n_tracks = len(tracks["eta"]) if tracks is not None else 0
    palette  = _TAB20[:max(n_tracks, 1)]

    col_uot   = assignment_colors(assignment, n_tracks)
    col_truth = truth_colors(truth_neutral)

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#F5F5F5")

    # ── Left: UOT assignment ────────────────────────────────────────────────
    ax_uot = fig.add_subplot(121, projection="3d")
    ax_uot.set_facecolor("#F5F5F5")

    # Neutral hits first (behind), then charged (on top)
    neu_mask = assignment < 0
    if neu_mask.any():
        _scatter3d(ax_uot, xyz[neu_mask],  col_uot[neu_mask],  sz[neu_mask],  zorder=1)
    if (~neu_mask).any():
        _scatter3d(ax_uot, xyz[~neu_mask], col_uot[~neu_mask], sz[~neu_mask], zorder=2)

    _draw_tracks(ax_uot, tracks, palette)
    _set_axes(ax_uot, xyz, tracks,
              f"UOT assignment  (ε={eps}, τ={tau})\n"
              f"neutral={int(neu_mask.sum())}  charged={int((~neu_mask).sum())}")
    _uot_legend(ax_uot, n_tracks, palette)
    ax_uot.view_init(elev=elev, azim=azim)

    # ── Right: MC truth ──────────────────────────────────────────────────────
    ax_tr = fig.add_subplot(122, projection="3d")
    ax_tr.set_facecolor("#F5F5F5")

    if truth_neutral.any():
        _scatter3d(ax_tr, xyz[truth_neutral],  col_truth[truth_neutral],
                   sz[truth_neutral],  zorder=1)
    if (~truth_neutral).any():
        _scatter3d(ax_tr, xyz[~truth_neutral], col_truth[~truth_neutral],
                   sz[~truth_neutral], zorder=2)

    _draw_tracks(ax_tr, tracks, palette)
    _set_axes(ax_tr, xyz, tracks,
              f"MC truth\n"
              f"neutral={int(truth_neutral.sum())}  "
              f"charged={int((~truth_neutral).sum())}")
    _truth_legend(ax_tr)
    ax_tr.view_init(elev=elev, azim=azim)

    # ── Global title & metrics ───────────────────────────────────────────────
    pred_neutral = assignment < 0
    TP = int(np.sum( pred_neutral &  truth_neutral))
    FP = int(np.sum( pred_neutral & ~truth_neutral))
    TN = int(np.sum(~pred_neutral & ~truth_neutral))
    FN = int(np.sum(~pred_neutral &  truth_neutral))
    prec = TP / (TP + FP + 1e-10)
    rec  = TP / (TP + FN + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    E_rat = (tracks["E_exp"].sum() / (hits["E"].sum() + 1e-6)
             if tracks is not None else 0)

    fig.suptitle(
        f"Event {event_idx}  |  tracks={n_tracks}  hits={len(xyz)}  "
        f"E_trk/E_hit={E_rat:.2f}\n"
        f"Neutral: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}",
        fontsize=10, y=1.01
    )

    fname = Path(out_dir) / f"3d_event_{event_idx:04d}.png"
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    return fname


def plot_event_3d_multiview(event_idx, tracks, hits, assignment, truth_neutral,
                            out_dir, eps, tau, dpi=150):
    """
    Four-panel figure: [UOT side | UOT top | truth side | truth top].
    Gives a richer spatial understanding.
    """
    xyz  = hits["xyz"]
    sz   = np.clip(hits["E"] * 25, 3, 120)

    n_tracks = len(tracks["eta"]) if tracks is not None else 0
    palette  = _TAB20[:max(n_tracks, 1)]
    col_uot   = assignment_colors(assignment, n_tracks)
    col_truth = truth_colors(truth_neutral)
    neu_mask  = assignment < 0

    views = [(25, 45, "Side view"),
             (85, 0,  "Top view (η-φ plane)")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("#F5F5F5")

    for row, (elev, azim, view_label) in enumerate(views):
        # -- UOT (left column) -----------------------------------------------
        ax = axes[row, 0]
        ax.set_facecolor("#F5F5F5")
        if neu_mask.any():
            _scatter3d(ax, xyz[neu_mask],  col_uot[neu_mask],  sz[neu_mask],  1)
        if (~neu_mask).any():
            _scatter3d(ax, xyz[~neu_mask], col_uot[~neu_mask], sz[~neu_mask], 2)
        _draw_tracks(ax, tracks, palette)
        _set_axes(ax, xyz, tracks,
                  f"UOT — {view_label}\n"
                  f"neutral={int(neu_mask.sum())}  charged={int((~neu_mask).sum())}")
        if row == 0:
            _uot_legend(ax, n_tracks, palette)
        ax.view_init(elev=elev, azim=azim)

        # -- Truth (right column) --------------------------------------------
        ax = axes[row, 1]
        ax.set_facecolor("#F5F5F5")
        if truth_neutral.any():
            _scatter3d(ax, xyz[truth_neutral],  col_truth[truth_neutral],
                       sz[truth_neutral],  1)
        if (~truth_neutral).any():
            _scatter3d(ax, xyz[~truth_neutral], col_truth[~truth_neutral],
                       sz[~truth_neutral], 2)
        _draw_tracks(ax, tracks, palette)
        _set_axes(ax, xyz, tracks,
                  f"Truth — {view_label}\n"
                  f"neutral={int(truth_neutral.sum())}  "
                  f"charged={int((~truth_neutral).sum())}")
        if row == 0:
            _truth_legend(ax)
        ax.view_init(elev=elev, azim=azim)

    # Global title
    pred_neutral = assignment < 0
    TP = int(np.sum( pred_neutral &  truth_neutral))
    FP = int(np.sum( pred_neutral & ~truth_neutral))
    TN = int(np.sum(~pred_neutral & ~truth_neutral))
    FN = int(np.sum(~pred_neutral &  truth_neutral))
    prec = TP / (TP + FP + 1e-10)
    rec  = TP / (TP + FN + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    E_rat = (tracks["E_exp"].sum() / (hits["E"].sum() + 1e-6)
             if tracks is not None else 0)

    fig.suptitle(
        f"Event {event_idx}  |  tracks={n_tracks}  hits={len(xyz)}  "
        f"E_trk/E_hit={E_rat:.2f}  |  "
        f"Neutral: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}\n"
        f"UOT params: ε={eps}  τ={tau}  adaptive energy-balance threshold",
        fontsize=11, y=1.01
    )

    fname = Path(out_dir) / f"3d_multiview_event_{event_idx:04d}.png"
    plt.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close()
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  Interactive HTML plots via Plotly
# ═══════════════════════════════════════════════════════════════════════════════

def _hex_palette(n):
    """Return n distinct hex colour strings from tab20/tab20b/tab20c."""
    colours = []
    for c in _TAB20[:n]:
        r, g, b = (int(x * 255) for x in c[:3])
        colours.append(f"#{r:02x}{g:02x}{b:02x}")
    return colours


# Common particle names keyed by |PDG id|
_PDG_NAMES = {
    11: "e±", 13: "μ±", 15: "τ±",
    12: "νe", 14: "νμ", 16: "ντ",
    22: "γ",
    111: "π⁰", 211: "π±",
    130: "K⁰L", 310: "K⁰S", 321: "K±",
    2112: "n", 2212: "p",
    1000010020: "d",
}

def _pdg_name(pid):
    return _PDG_NAMES.get(abs(int(pid)), f"PDG {int(pid)}")


def plot_event_html(event_idx, tracks, hits, assignment, truth_neutral,
                    X_gen, out_dir, eps, tau):
    """
    Interactive side-by-side Plotly figure saved as a self-contained HTML file.

    Left subplot  — UOT assignment:
        · Each assigned track gets a distinct colour.
        · Neutral hits are light grey, semi-transparent.
        · Tracks shown as coloured diamond markers + directional arrows.

    Right subplot — MC truth, coloured by MC particle:
        · One colour per MC parent particle (gen_link index).
        · Hover shows particle index, PDG id, name, energy, hit energy.
        · Noise/unlinked hits (gen_link = -1) shown in grey.

    The HTML file is fully self-contained and opens in any browser.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    xyz      = hits["xyz"]       # [H, 3]
    hit_E    = hits["E"]         # [H]
    hit_eta  = hits["eta"]
    hit_phi  = hits["phi"]
    gen_link = hits["gen_link"]  # [H] MC particle index, -1 = noise

    n_tracks = len(tracks["eta"]) if tracks is not None else 0
    # Separate palettes: left uses first N colours, right uses a large pool
    palette_uot  = _hex_palette(max(n_tracks, 1))
    n_particles  = len(X_gen) if X_gen is not None else 0
    palette_mc   = _hex_palette(max(n_particles, 1))

    neu_mask = assignment < 0   # bool [H]

    # Metrics
    pred_neutral = neu_mask
    TP = int(np.sum( pred_neutral &  truth_neutral))
    FP = int(np.sum( pred_neutral & ~truth_neutral))
    FN = int(np.sum(~pred_neutral &  truth_neutral))
    prec = TP / (TP + FP + 1e-10)
    rec  = TP / (TP + FN + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    E_rat = (tracks["E_exp"].sum() / (hit_E.sum() + 1e-6) if tracks is not None else 0)

    # Count unique MC particles for subtitle
    unique_parts = sorted(set(int(g) for g in gen_link if g > 0))  # exclude dummy index 0
    n_unique = len(unique_parts)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"UOT assignment  (ε={eps}, τ={tau})<br>"
            f"neutral={int(neu_mask.sum())}  charged={int((~neu_mask).sum())}",
            f"MC truth — coloured by particle<br>"
            f"{n_unique} particles  ({int(truth_neutral.sum())} neutral hits)",
        ],
        horizontal_spacing=0.02,
    )

    # ── Left panel: UOT assignment ────────────────────────────────────────────

    if neu_mask.any():
        idxs = np.where(neu_mask)[0]
        fig.add_trace(go.Scatter3d(
            x=xyz[neu_mask, 0], y=xyz[neu_mask, 1], z=xyz[neu_mask, 2],
            mode="markers",
            marker=dict(size=2, color=NEUTRAL_COLOR, opacity=0.25, line=dict(width=0)),
            name="Neutral (UOT)",
            text=[f"hit {j}<br>E={hit_E[j]:.4f} GeV<br>"
                  f"η={hit_eta[j]:.3f}  φ={hit_phi[j]:.3f}<br>label=neutral"
                  for j in idxs],
            hoverinfo="text",
            legendgroup="neutral_uot",
        ), row=1, col=1)

    for k in range(n_tracks):
        mask_k = assignment == k   # 0-based: argmax returns 0..n_tracks-1
        if not mask_k.any():
            continue
        idxs = np.where(mask_k)[0]
        col  = palette_uot[k % len(palette_uot)]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask_k, 0], y=xyz[mask_k, 1], z=xyz[mask_k, 2],
            mode="markers",
            marker=dict(size=3, color=col, opacity=0.80, line=dict(width=0)),
            name=f"Track {k+1}",
            text=[f"hit {j}<br>E={hit_E[j]:.4f} GeV<br>"
                  f"η={hit_eta[j]:.3f}  φ={hit_phi[j]:.3f}<br>→ track {k+1}"
                  for j in idxs],
            hoverinfo="text",
            legendgroup=f"trk{k}",
        ), row=1, col=1)

    # ── Right panel: one trace per MC particle ────────────────────────────────

    # Noise / unlinked hits (ygen_hit == 0 → dummy X_gen row, no real MC particle)
    noise_mask = gen_link == 0
    if noise_mask.any():
        idxs = np.where(noise_mask)[0]
        fig.add_trace(go.Scatter3d(
            x=xyz[noise_mask, 0], y=xyz[noise_mask, 1], z=xyz[noise_mask, 2],
            mode="markers",
            marker=dict(size=2, color=NEUTRAL_COLOR, opacity=0.20, line=dict(width=0)),
            name="Noise / unlinked",
            text=[f"hit {j}<br>E={hit_E[j]:.4f} GeV<br>gen_link=-1 (noise)"
                  for j in idxs],
            hoverinfo="text",
            legendgroup="noise",
        ), row=1, col=2)

    # One trace per MC particle, sorted by total deposited energy (largest first)
    part_energies = {}
    for p in unique_parts:
        mask_p = gen_link == p
        part_energies[p] = float(hit_E[mask_p].sum())
    sorted_parts = sorted(unique_parts, key=lambda p: -part_energies[p])

    # Map MC particle index → colour used in right panel (built while drawing hits)
    part_color_map: dict = {}

    for rank, p in enumerate(sorted_parts):
        mask_p = gen_link == p
        if not mask_p.any():
            continue
        idxs = np.where(mask_p)[0]
        col  = palette_mc[rank % len(palette_mc)]
        part_color_map[p] = col

        # MC particle info from X_gen
        if X_gen is not None and p < len(X_gen):
            pid    = int(X_gen[p, 0])
            pname  = _pdg_name(pid)
            E_part = float(X_gen[p, 8])   # true energy (col 8)
            p_mag  = float(X_gen[p, 11])  # momentum magnitude (col 11)
            part_label = f"part {p} | {pname} (PDG {pid})"
            part_info  = (f"PDG={pid} ({pname})<br>"
                          f"E_true={E_part:.3f} GeV  p={p_mag:.3f} GeV<br>"
                          f"E_hits={part_energies[p]:.4f} GeV")
        else:
            part_label = f"part {p}"
            part_info  = f"E_hits={part_energies[p]:.4f} GeV"

        hover_p = [
            f"hit {j}<br>E={hit_E[j]:.4f} GeV<br>"
            f"η={hit_eta[j]:.3f}  φ={hit_phi[j]:.3f}<br>"
            f"{part_info}"
            for j in idxs
        ]
        fig.add_trace(go.Scatter3d(
            x=xyz[mask_p, 0], y=xyz[mask_p, 1], z=xyz[mask_p, 2],
            mode="markers",
            marker=dict(size=3, color=col, opacity=0.75, line=dict(width=0)),
            name=part_label,
            text=hover_p, hoverinfo="text",
            legendgroup=f"part{p}",
        ), row=1, col=2)

    # ── Tracks: arrows in both panels ────────────────────────────────────────
    if tracks is not None:
        ARROW_LEN = 400.0   # mm
        for k, (ref, d, E_exp) in enumerate(
                zip(tracks["ref_xyz"], tracks["dir_unit"], tracks["E_exp"])):
            col_left  = palette_uot[k % len(palette_uot)]
            # Right panel: match the colour of hits linked to the same MC particle
            mc_part   = int(tracks["gen_link"][k]) if tracks["gen_link"][k] > 0 else -1
            col_right = part_color_map.get(mc_part, "#808080")  # grey if unlinked

            end  = ref + d * ARROW_LEN
            eta_t, phi_t = float(tracks["eta"][k]), float(tracks["phi"][k])
            hover_trk = (f"Track {k+1}<br>p={tracks['p'][k]:.2f} GeV<br>"
                         f"E_exp={E_exp:.2f} GeV<br>"
                         f"η={eta_t:.3f}  φ={phi_t:.3f}")

            for col_idx in [1, 2]:
                col = col_left if col_idx == 1 else col_right
                # Arrow line
                fig.add_trace(go.Scatter3d(
                    x=[ref[0], end[0]], y=[ref[1], end[1]], z=[ref[2], end[2]],
                    mode="lines",
                    line=dict(color=col, width=4),
                    name=f"Track {k+1} dir" if col_idx == 1 else None,
                    showlegend=False,
                    text=[hover_trk, hover_trk], hoverinfo="text",
                ), row=1, col=col_idx)
                # Cone marker at reference point
                fig.add_trace(go.Scatter3d(
                    x=[ref[0]], y=[ref[1]], z=[ref[2]],
                    mode="markers",
                    marker=dict(size=8, color=col, symbol="diamond",
                                line=dict(color="black", width=1)),
                    name=f"Track {k+1}" if col_idx == 1 else None,
                    showlegend=False,
                    text=[hover_trk], hoverinfo="text",
                ), row=1, col=col_idx)

    # ── Scene axes (shared style) ─────────────────────────────────────────────
    axis_style = dict(
        backgroundcolor="#f0f0f0",
        gridcolor="white", showbackground=True,
        zerolinecolor="white",
        tickfont=dict(size=9),
    )
    scene_kw = dict(
        xaxis=dict(title="x (mm)", **axis_style),
        yaxis=dict(title="y (mm)", **axis_style),
        zaxis=dict(title="z (mm)", **axis_style),
        aspectmode="data",
    )
    fig.update_layout(scene=scene_kw, scene2=scene_kw)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f"Event {event_idx}  |  tracks={n_tracks}  hits={len(xyz)}  "
                  f"E_trk/E_hit={E_rat:.2f}  |  "
                  f"Neutral: prec={prec:.3f}  rec={rec:.3f}  F1={f1:.3f}"),
            font=dict(size=13),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor="#f8f8f8",
        legend=dict(
            itemsizing="constant",
            font=dict(size=10),
            tracegroupgap=2,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1600, height=750,
    )

    fname = Path(out_dir) / f"3d_event_{event_idx:04d}.html"
    fig.write_html(str(fname), include_plotlyjs="cdn", full_html=True)

    # ── Per-track energy table ────────────────────────────────────────────────
    table_rows = ""
    total_E_exp = 0.0
    total_E_hits = 0.0
    total_n_hits = 0
    for k in range(n_tracks):
        mask_k     = assignment == k
        n_k        = int(mask_k.sum())
        E_hits_k   = float(hit_E[mask_k].sum())
        E_exp_k    = float(tracks["E_exp"][k])
        p_k        = float(tracks["p"][k])
        ratio      = E_hits_k / E_exp_k if E_exp_k > 0 else 0.0
        # colour-code ratio: green=good, orange=over, red=very off
        if   ratio < 0.3:  rc = "#c0392b"   # too few hits
        elif ratio < 0.8:  rc = "#e67e22"   # partial
        elif ratio < 1.5:  rc = "#27ae60"   # good
        else:              rc = "#8e44ad"   # overshot
        trk_col = palette_uot[k % len(palette_uot)]
        table_rows += (
            f"<tr>"
            f"<td style='color:{trk_col};font-weight:bold;padding:6px 10px'>"
            f"  &#9632; Track {k+1}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{p_k:.3f}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{E_exp_k:.3f}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{E_hits_k:.3f}</td>"
            f"<td style='padding:6px 10px;text-align:right;color:{rc};font-weight:bold'>{ratio:.2f}</td>"
            f"<td style='padding:6px 10px;text-align:right'>{n_k}</td>"
            f"</tr>\n"
        )
        total_E_exp  += E_exp_k
        total_E_hits += E_hits_k
        total_n_hits += n_k

    # Neutral row
    neu_E = float(hit_E[assignment < 0].sum())
    n_neu = int((assignment < 0).sum())
    table_rows += (
        f"<tr style='color:#888'>"
        f"<td><i>Neutral / unassigned</i></td>"
        f"<td>—</td><td>—</td>"
        f"<td>{neu_E:.3f}</td><td>—</td><td>{n_neu}</td>"
        f"</tr>\n"
    )

    total_ratio = total_E_hits / total_E_exp if total_E_exp > 0 else 0.0
    table_html = f"""
<div style="margin:24px auto;max-width:860px;font-family:'Helvetica Neue',Arial,sans-serif;font-size:13px;">
  <h3 style="margin-bottom:8px">
    Track energy summary &mdash; Event {event_idx}
    &nbsp;<span style="font-weight:normal;font-size:11px;color:#555">
      (E_exp = &radic;(p&sup2;+m&#960;&sup2;), pion hypothesis)
    </span>
  </h3>
  <table style="border-collapse:collapse;width:100%;background:#fff;
                box-shadow:0 1px 4px rgba(0,0,0,.15)">
    <thead>
      <tr style="background:#2c3e50;color:#fff">
        <th style="padding:7px 10px;text-align:left">Track</th>
        <th style="padding:7px 10px;text-align:right">p (GeV)</th>
        <th style="padding:7px 10px;text-align:right">E_exp (GeV)</th>
        <th style="padding:7px 10px;text-align:right">E_hits (GeV)</th>
        <th style="padding:7px 10px;text-align:right">E_hits / E_exp</th>
        <th style="padding:7px 10px;text-align:right">N hits</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
    <tfoot>
      <tr style="background:#ecf0f1;font-weight:bold">
        <td style="padding:7px 10px">Total (charged)</td>
        <td style="padding:7px 10px;text-align:right">—</td>
        <td style="padding:7px 10px;text-align:right">{total_E_exp:.3f}</td>
        <td style="padding:7px 10px;text-align:right">{total_E_hits:.3f}</td>
        <td style="padding:7px 10px;text-align:right">{total_ratio:.2f}</td>
        <td style="padding:7px 10px;text-align:right">{total_n_hits}</td>
      </tr>
      <tr style="background:#ecf0f1">
        <td style="padding:7px 10px">All calo hits</td>
        <td style="padding:7px 10px;text-align:right">—</td>
        <td style="padding:7px 10px;text-align:right">—</td>
        <td style="padding:7px 10px;text-align:right">{float(hit_E.sum()):.3f}</td>
        <td style="padding:7px 10px;text-align:right">—</td>
        <td style="padding:7px 10px;text-align:right">{len(hit_E)}</td>
      </tr>
    </tfoot>
  </table>
  <p style="color:#555;font-size:11px;margin-top:6px">
    Ratio colour: <span style="color:#27ae60">&#9632; 0.8–1.5 good</span> &nbsp;
    <span style="color:#e67e22">&#9632; 0.3–0.8 partial</span> &nbsp;
    <span style="color:#c0392b">&#9632; &lt;0.3 too few hits</span> &nbsp;
    <span style="color:#8e44ad">&#9632; &gt;1.5 overshot</span>
  </p>
</div>
"""

    # Inject table before </body>
    html_text = fname.read_text()
    html_text = html_text.replace("</body>", table_html + "\n</body>")
    fname.write_text(html_text)

    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="3-D visualisation of UOT track-hit matching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--events", nargs="+", type=int,
                        default=None,
                        help="Event indices to plot (default: auto-select 8 interesting ones)")
    parser.add_argument("--eps",  type=float, default=0.02)
    parser.add_argument("--tau",  type=float, default=0.20)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--sigma-perp", type=float, default=300.0,
                        help="Transverse distance scale σ_perp (mm)")
    parser.add_argument("--sigma-par",  type=float, default=4000.0,
                        help="Along-axis scale σ_par (mm) — anisotropic mode only")
    parser.add_argument("--max-d3D",    type=float, default=6000.0,
                        help="Hard 3-D distance cutoff from track entry point (mm)")
    parser.add_argument("--n-min-hits", type=int,   default=20,
                        help="Min hits guaranteed per track (expands cutoff if needed)")
    parser.add_argument("--compare",    action="store_true",
                        help="Run both isotropic and anisotropic modes and compare")
    parser.add_argument("--compare-shower", action="store_true",
                        help="Run shower + anisotropic modes and compare")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--multiview", action="store_true",
                        help="Also produce 4-panel (side+top) PNG figures")
    parser.add_argument("--html", action="store_true",
                        help="Produce interactive HTML files (Plotly)")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip static PNG output (useful with --html)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.parquet} …", flush=True)
    data     = ak.from_parquet(args.parquet)
    n_total  = len(data["X_track"])

    # If no events specified, auto-select a variety: good F1, bad F1, many tracks, few tracks
    if args.events is None:
        # Quickly scan to pick events with different E_trk/E_hit ratios
        ratios = []
        for i in range(min(n_total, 100)):
            ev = {f: np.array(data[f][i]) for f in data.fields}
            Xt = ev["X_track"]
            Xh = ev["X_hit"]
            if len(Xt) == 0:
                ratios.append(99.0); continue
            hE = Xh[:, 5]; ht = (Xh[:, 10] + 1).astype(int)
            cal = (ht >= 2) & (ht <= 3) & (hE > 0)
            pm  = Xt[:, 5]
            Eexp = np.sqrt(pm**2 + M_PION**2)
            r = Eexp.sum() / (hE[cal].sum() + 1e-6)
            ratios.append(float(r))
        ratios = np.array(ratios)
        # Pick: low ratio (good performance), medium, high
        low_r  = np.where(ratios < 0.3)[0]
        mid_r  = np.where((ratios >= 0.4) & (ratios < 0.7))[0]
        high_r = np.where((ratios >= 0.7) & (ratios < 1.0))[0]
        selected = []
        for group in [low_r, mid_r, high_r]:
            if len(group) >= 3:
                selected.extend(group[:3].tolist())
            else:
                selected.extend(group.tolist())
        args.events = selected[:9]
        print(f"  Auto-selected events: {args.events}")

    if args.compare:
        modes = ["anisotropic", "isotropic"]
    elif args.compare_shower:
        modes = ["shower", "anisotropic"]
    else:
        modes = ["shower"]

    print(f"\n{'='*70}")
    print(f"  Plotting {len(args.events)} events in 3-D")
    print(f"  ε={args.eps}  τ={args.tau}  modes={modes}")
    print(f"  σ_perp={args.sigma_perp}mm  σ_par={args.sigma_par}mm"
          f"  max_d3D={args.max_d3D}mm  n_min_hits={args.n_min_hits}")
    print(f"  out → {out_dir}")
    print(f"{'='*70}\n", flush=True)

    for idx in args.events:
        ev    = load_event(data, idx)
        tracks = extract_tracks(ev)
        hits   = extract_hits(ev)
        if hits is None or len(hits["E"]) == 0:
            print(f"  Event {idx}: no hits — skipped"); continue

        X_gen         = ev.get("X_gen")
        truth_neutral = get_truth_neutrality(tracks, hits)
        n_trk = len(tracks["eta"]) if tracks is not None else 0
        E_rat = (tracks["E_exp"].sum() / (hits["E"].sum() + 1e-6)
                 if tracks is not None else 0)

        results = {}
        for mode in modes:
            asgn, frac = run_uot(tracks, hits, eps=args.eps, tau=args.tau,
                                 n_iter=args.n_iter,
                                 sigma_perp=args.sigma_perp,
                                 sigma_par=args.sigma_par,
                                 max_d3D=args.max_d3D,
                                 n_min_hits=args.n_min_hits,
                                 cost_mode=mode)
            results[mode] = (asgn, frac)

            pred_neutral = asgn < 0
            TP = int(np.sum( pred_neutral &  truth_neutral))
            FP = int(np.sum( pred_neutral & ~truth_neutral))
            FN = int(np.sum(~pred_neutral &  truth_neutral))
            prec = TP / (TP + FP + 1e-10)
            rec  = TP / (TP + FN + 1e-10)
            f1   = 2 * prec * rec / (prec + rec + 1e-10)
            print(f"  [{mode[:4]}] Event {idx:3d}  trk={n_trk}  "
                  f"E_trk/E_hit={E_rat:.2f}  "
                  f"F1={f1:.3f}  prec={prec:.3f}  rec={rec:.3f}", flush=True)

        # ── Per-track ratio comparison table ─────────────────────────────────
        if args.compare and tracks is not None:
            nu = hits["E"]
            mu = tracks["E_exp"]
            asgn_aniso, _ = results["anisotropic"]
            asgn_iso,   _ = results["isotropic"]
            print(f"  {'Trk':>4}  {'E_exp':>7}  "
                  f"{'E_hits(aniso)':>14}  {'ratio_a':>8}  "
                  f"{'E_hits(iso)':>12}  {'ratio_i':>8}")
            for k in range(n_trk):
                E_exp_k = float(mu[k])
                E_a = float(nu[asgn_aniso == k].sum())
                E_i = float(nu[asgn_iso   == k].sum())
                ra  = E_a / E_exp_k if E_exp_k > 0 else 0
                ri  = E_i / E_exp_k if E_exp_k > 0 else 0
                flag = " ◄" if abs(ra - ri) > 0.15 else ""
                print(f"  {k+1:>4}  {E_exp_k:>7.2f}  "
                      f"{E_a:>14.2f}  {ra:>8.2f}  "
                      f"{E_i:>12.2f}  {ri:>8.2f}{flag}")
            print()

        # ── Output files ──────────────────────────────────────────────────────
        for mode in modes:
            asgn, frac = results[mode]
            suffix = f"_{mode[:4]}"

            if not args.no_png:
                p1 = plot_event_3d(idx, tracks, hits, asgn, truth_neutral,
                                   out_dir, args.eps, args.tau, dpi=args.dpi)
                # rename to include mode suffix
                p1_new = p1.parent / p1.name.replace(".png", f"{suffix}.png")
                p1.rename(p1_new)
                print(f"    → {p1_new}")

            if args.html:
                h1 = plot_event_html(idx, tracks, hits, asgn, truth_neutral,
                                     X_gen, out_dir, args.eps, args.tau)
                h1_new = h1.parent / h1.name.replace(".html", f"{suffix}.html")
                h1.rename(h1_new)
                print(f"    → {h1_new}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
