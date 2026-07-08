#!/usr/bin/env python3
"""
Unbalanced Optimal Transport (UOT) for particle-flow track-hit separation.

Core idea
─────────
Charged particles leave both a tracker track AND calorimeter deposits.
Neutral particles (γ, K0L, n) leave ONLY calorimeter hits.
Use UOT to "transport" track energies onto calorimeter hits.
Hits that receive little transport weight are neutral-particle candidates.

Transport problem
─────────────────
    min_{T≥0}  <C, T>  +  ε · KL(T | μ⊗ν)
                        +  τ · KL(T1  | μ)   ← track marginal relaxation
                        +  τ · KL(T'1 | ν)   ← hit   marginal relaxation

where
  · C[i,j] = ΔR²(track_i, hit_j)      angular cost in η-φ space
  · μ_i    = expected energy of track i  (pion hyp.: E = √(p²+mπ²))
  · ν_j    = measured energy of hit j
  · ε      — entropic reg. (sharpness, ~ΔR² scale; try 0.02–0.2)
  · τ      — KL penalty on marginals (try 0.2–2.0; smaller→ more neutral)

Log-domain Sinkhorn iterations (Chizat et al. 2018, §4.2)
──────────────────────────────────────────────────────────
  α = τ / (τ + ε)
  Init: a = 0, b = 0  (dual potentials)
  Repeat until convergence:
    a_i ← α · [ log μ_i  − LSE_j(b_j  − C_{ij}/ε) ]
    b_j ← α · [ log ν_j  − LSE_i(a_i  − C_{ij}/ε) ]
  Transport plan:  T_{ij} = exp(a_i + b_j − C_{ij}/ε)

References
──────────
  Chizat, Peyré, Schmitzer, Vialard (2018). "Scaling algorithms for
    unbalanced optimal transport problems." Math. Comp. 87(314), 2563–2609.
  Séjourné, Feydy, Vialard, Trouvé, Peyré (2019). "Sinkhorn divergences
    for unbalanced optimal transport." arXiv:1910.12958.
  Cuturi (2013). "Sinkhorn distances: lightspeed computation of OT
    distances." NeurIPS.

Usage (inside gatr:v9 Singularity container)
─────────────────────────────────────────────
  python scripts/uot_track_hit_matching.py [options]

Key options
  --n-events     N      events to process (default 30)
  --eps          0.05   entropic regularisation
  --tau          0.5    marginal KL penalty
  --neutral-fraction 0.25  transport-fraction threshold → neutral
  --scan                hyper-parameter scan
  --device cpu|cuda     computation device
"""

import sys
import argparse
import time
import functools
from pathlib import Path

import numpy as np
import awkward as ak

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from scipy.special import logsumexp as sci_logsumexp

print(f"PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

M_PION   = 0.13957   # GeV — pion hypothesis for track → expected energy
EPS_SAFE = 1e-30     # floor for log(x + eps)
FLOAT64  = np.float64  # dtype for Sinkhorn (64-bit for log-domain stability)

DEFAULT_PARQUET = (
    "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/"
    "Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10601.parquet"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_parquet(path: str):
    return ak.from_parquet(path)


def event_to_numpy(data, idx: int) -> dict:
    return {field: np.array(data[field][idx]) for field in data.fields}


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def safe_eta_phi(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                 floor: float = 1e-6):
    """
    Pseudorapidity η and azimuthal φ from Cartesian coordinates.
    Works equally for position vectors (mm) and momentum vectors (GeV).
    """
    r_T   = np.sqrt(px**2 + py**2)
    theta = np.arctan2(r_T + floor, pz)          # θ ∈ (0, π)
    eta   = -np.log(np.tan(0.5 * theta) + floor)  # η = −ln tan(θ/2)
    phi   = np.arctan2(py, px)                    # φ ∈ (−π, π]
    return eta.astype(np.float32), phi.astype(np.float32)


def extract_tracks(event: dict):
    """
    X_track column layout (CLD, 25 columns):
      [0]     elem_type = 1
      [5]     |p|  momentum magnitude (GeV)
      [6:9]   (px,py,pz) at primary vertex
      [12:15] (x,y,z)    reference point at calorimeter face (mm)
      [15]    chi²
      [16]    ndf
      [22:25] (px,py,pz) at calorimeter face  ← direction for matching
    ygen_track: MC particle index (−1 = unlinked noise)
    """
    X = event.get("X_track", np.empty((0, 25), dtype=np.float32))
    if len(X) == 0:
        return None

    p_mag  = X[:, 5].astype(np.float32)
    # Direction at calo face gives the best angular prediction
    px_c, py_c, pz_c = X[:, 22], X[:, 23], X[:, 24]

    eta, phi = safe_eta_phi(px_c, py_c, pz_c)
    E_exp    = np.sqrt(p_mag**2 + M_PION**2).astype(np.float32)
    gen_link = np.array(event["ygen_track"], dtype=np.int64).flatten()

    ref_xyz  = X[:, 12:15].astype(np.float32)          # calo face entry point (mm)
    dir_calo = X[:, 22:25].astype(np.float32)           # momentum at calo face
    norm     = np.linalg.norm(dir_calo, axis=1, keepdims=True) + 1e-8
    dir_unit = dir_calo / norm                           # unit direction vectors

    return dict(eta=eta, phi=phi, p=p_mag, E_exp=E_exp, gen_link=gen_link,
                ref_xyz=ref_xyz, dir_unit=dir_unit)


def extract_hits(event: dict, include_muon: bool = False):
    """
    X_hit column layout (CLD, 12 columns):
      [0]     (2.0 for ECAL, 3.0 for HCAL — copy of hit_type?)
      [5]     energy (GeV)
      [6:9]   (x,y,z) hit position (mm)
      [10]    hit sub-type:  1→ECAL, 2→HCAL  (hit_type = col[10]+1)
    ygen_hit: MC particle index (−1 = noise/unlinked)
    """
    X = event.get("X_hit", np.empty((0, 12), dtype=np.float32))
    if len(X) == 0:
        return None

    hit_type = (X[:, 10] + 1).astype(np.int32)   # 2=ECAL, 3=HCAL
    hit_E    = X[:, 5].astype(np.float32)
    hit_x    = X[:, 6].astype(np.float32)
    hit_y    = X[:, 7].astype(np.float32)
    hit_z    = X[:, 8].astype(np.float32)

    eta, phi = safe_eta_phi(hit_x, hit_y, hit_z)
    gen_link = np.array(event["ygen_hit"], dtype=np.int64).flatten()

    if include_muon:
        cal_mask = (hit_type >= 2) & (hit_E > 0)
    else:
        cal_mask = (hit_type >= 2) & (hit_type <= 3) & (hit_E > 0)

    hit_xyz = np.stack([hit_x, hit_y, hit_z], axis=1).astype(np.float32)  # [H, 3]

    return dict(
        eta=eta[cal_mask],
        phi=phi[cal_mask],
        E=hit_E[cal_mask],
        type=hit_type[cal_mask],
        gen_link=gen_link[cal_mask],
        orig_mask=cal_mask,
        xyz=hit_xyz[cal_mask],
    )


def get_truth_neutrality(tracks, hits) -> np.ndarray:
    """
    Boolean array [n_hits]: True ↔ hit originates from a neutral particle.

    A hit is "charged-origin" iff its MC parent particle also left a track.
    We identify charged MC parents as the set of valid ygen_track values.
    """
    if tracks is None or len(tracks["gen_link"]) == 0:
        return np.ones(len(hits["gen_link"]), dtype=bool)

    charged_mc = set(int(x) for x in tracks["gen_link"].tolist() if x >= 0)
    return np.array(
        [(int(gl) not in charged_mc) for gl in hits["gen_link"]],
        dtype=bool,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  COST MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def combined_cost(
    track_eta:      np.ndarray,   # [T]
    track_phi:      np.ndarray,   # [T]
    track_ref_xyz:  np.ndarray,   # [T, 3]  calo entry point (mm)
    track_dir_unit: np.ndarray,   # [T, 3]  unit direction at calo face
    hit_eta:        np.ndarray,   # [H]
    hit_phi:        np.ndarray,   # [H]
    hit_xyz:        np.ndarray,   # [H, 3]  hit position (mm)
    max_dR:         float = 0.4,  # hard angular cutoff — hits outside cone cannot be assigned
    sigma_perp:     float = 400.0,# scale of transverse-distance penalty (mm)
    lambda_perp:    float = 0.1,  # weight of transverse penalty relative to ΔR²
    C_inf:          float = 1e6,  # cost for pairs outside hard cutoffs
) -> np.ndarray:
    """
    Geometric cost matrix [T, H] for UOT track-hit matching.

    Two complementary constraints prevent absorbing neutral hits:

    1. Hard ΔR cone cutoff (max_dR)
       Hits outside the cone are completely blocked (cost = C_inf).
       Showers are collimated; anything beyond ΔR ~ 0.4 is almost certainly
       from a different particle.

    2. 3-D transverse distance penalty (lambda_perp, sigma_perp)
       For each (track k, hit j) pair, compute the distance of the hit from
       the track axis (calo entry point + direction):

           delta    = hit_xyz[j] - track_ref_xyz[k]          [3]
           d_par    = dot(delta, track_dir_unit[k])           scalar
           d_perp   = |delta - d_par * track_dir_unit[k]|    scalar (mm)

       Shower development is along the track direction; transversely displaced
       hits are suppressed by adding lambda_perp * (d_perp / sigma_perp)².

    Combined cost:
        C[k, j] = ΔR²(k,j) + lambda_perp * (d_perp(k,j) / sigma_perp)²
        C[k, j] = C_inf   if ΔR(k,j) > max_dR
    """
    # ── Angular cost ΔR² [T, H] ──────────────────────────────────────────────
    deta = track_eta[:, None] - hit_eta[None, :]
    dphi = track_phi[:, None] - hit_phi[None, :]
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    dR2  = (deta**2 + dphi**2).astype(np.float64)   # [T, H]

    # ── 3-D transverse distance [T, H] ───────────────────────────────────────
    # delta[k, j, :] = hit_xyz[j] - track_ref_xyz[k]
    delta  = hit_xyz[None, :, :] - track_ref_xyz[:, None, :]   # [T, H, 3]
    # projection along track direction
    d_par  = np.sum(delta * track_dir_unit[:, None, :], axis=2)  # [T, H]
    # perpendicular component
    d_perp_vec = delta - d_par[:, :, None] * track_dir_unit[:, None, :]  # [T, H, 3]
    d_perp     = np.sqrt(np.sum(d_perp_vec**2, axis=2))                   # [T, H] mm

    # ── Combined cost ─────────────────────────────────────────────────────────
    C = dR2 + lambda_perp * (d_perp / sigma_perp) ** 2

    # Hard ΔR cutoff: block hits outside the cone
    C[dR2 > max_dR ** 2] = C_inf

    return C.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  UNBALANCED SINKHORN  (numpy + scipy, log-domain)
# ═══════════════════════════════════════════════════════════════════════════════

def sinkhorn_uot(
    mu:     np.ndarray,    # [T] track expected energies (unnormalised)
    nu:     np.ndarray,    # [H] hit energies
    C:      np.ndarray,    # [T, H] ΔR² cost matrix
    eps:    float = 0.05,
    tau:    float = 0.1,
    n_iter: int   = 300,
    tol:    float = 1e-6,  # early stopping on dual-potential change
    device: str   = "cpu", # kept for API compatibility; GPU path uses torch
) -> np.ndarray:
    """
    Log-domain unbalanced Sinkhorn (Chizat et al. 2018, Algorithm 1).

    Solves:
        min_{T≥0}  <C,T>  +  ε·KL(T|μ⊗ν)  +  τ·KL(T1|μ)  +  τ·KL(T'1|ν)

    Iterations:
        α = τ / (τ + ε)                         (damping factor ∈ (0,1))
        a_i ← α · [ log μ_i − LSE_j(b_j − C_{ij}/ε) ]
        b_j ← α · [ log ν_j − LSE_i(a_i − C_{ij}/ε) ]
    Transport plan:  T_{ij} = exp(a_i + b_j − C_{ij}/ε)

    Physics note
    ────────────
    Small τ (0.05–0.2) → highly unbalanced → neutral hits receive little
    transport and their frac_transported ≪ 1.  Large τ → nearly balanced →
    every hit gets some transport regardless of neutrality.
    Recommended: τ ≈ 0.05–0.2,  ε ≈ 0.02–0.1 for ΔR² costs.
    """
    # Use GPU via PyTorch when requested and available
    if device == "cuda" and torch.cuda.is_available():
        return _sinkhorn_uot_torch(mu, nu, C, eps=eps, tau=tau, n_iter=n_iter)

    # ── Fast CPU path with numpy / scipy ────────────────────────────────────
    mu64 = mu.astype(FLOAT64)
    nu64 = nu.astype(FLOAT64)
    C64  = C.astype(FLOAT64)

    log_mu = np.log(mu64 + EPS_SAFE)   # [T]
    log_nu = np.log(nu64 + EPS_SAFE)   # [H]
    M      = -C64 / eps                 # [T, H]  kernel log-matrix

    alpha  = tau / (tau + eps)
    a = np.zeros(len(mu), dtype=FLOAT64)
    b = np.zeros(len(nu), dtype=FLOAT64)

    for it in range(n_iter):
        # LSE over hits (axis=1): shape [T]
        a_new = alpha * (log_mu - sci_logsumexp(b[None, :] + M, axis=1))
        # LSE over tracks (axis=0): shape [H]
        b_new = alpha * (log_nu - sci_logsumexp(a_new[:, None] + M, axis=0))

        # Early stopping: check max absolute change in duals
        if it > 0:
            da = np.max(np.abs(a_new - a))
            db = np.max(np.abs(b_new - b))
            if max(da, db) < tol:
                a, b = a_new, b_new
                break

        a, b = a_new, b_new

    log_T = a[:, None] + b[None, :] + M   # [T, H]
    return log_T.astype(np.float32)


def _sinkhorn_uot_torch(mu, nu, C, eps, tau, n_iter):
    """GPU-accelerated Sinkhorn using PyTorch (fallback for CUDA devices)."""
    dev = torch.device("cuda")
    dt  = torch.float64
    log_mu = torch.log(torch.tensor(mu, dtype=dt, device=dev) + EPS_SAFE)
    log_nu = torch.log(torch.tensor(nu, dtype=dt, device=dev) + EPS_SAFE)
    M      = torch.tensor(-C / eps,   dtype=dt, device=dev)
    alpha  = tau / (tau + eps)
    a = torch.zeros(len(mu), dtype=dt, device=dev)
    b = torch.zeros(len(nu), dtype=dt, device=dev)
    for _ in range(n_iter):
        a = alpha * (log_mu - torch.logsumexp(b.unsqueeze(0) + M, dim=1))
        b = alpha * (log_nu - torch.logsumexp(a.unsqueeze(1) + M, dim=0))
    return (a.unsqueeze(1) + b.unsqueeze(0) + M).cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  PER-EVENT MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _energy_balance_threshold(frac_transported, hit_E, track_E_exp_sum):
    """
    Adaptive threshold based on energy conservation.

    Expected neutral energy = max(0, E_hits − E_tracks).
    Sort hits by frac_transported (ascending = most neutral first),
    classify as neutral until their cumulative energy reaches E_neutral_expected.

    This is event-adaptive and physics-motivated: it ensures the total
    classified neutral energy equals the "unexplained" calorimeter energy.

    Returns
    -------
    threshold : float
        The frac_transported value that separates neutral from charged hits.
    """
    E_hits_total  = float(hit_E.sum())
    E_neutral_exp = max(0.0, E_hits_total - float(track_E_exp_sum))

    if E_neutral_exp <= 0:
        return 0.0  # all hits are charged by energy balance

    order     = np.argsort(frac_transported)   # ascending
    cum_E     = np.cumsum(hit_E[order])
    idx_cross = np.searchsorted(cum_E, E_neutral_exp)

    if idx_cross >= len(frac_transported):
        idx_cross = len(frac_transported) - 1

    # Threshold is the frac_transported value at the crossing point
    return float(frac_transported[order[idx_cross]])


def match_event(
    tracks,
    hits,
    eps:              float = 0.02,
    tau:              float = 0.05,
    n_iter:           int   = 300,
    neutral_fraction: float = 0.25,   # used when adaptive_threshold=False
    adaptive_threshold: bool = True,  # use energy-balance threshold
    max_dR:           float = 0.4,    # hard angular cone cutoff
    sigma_perp:       float = 400.0,  # transverse distance scale (mm)
    lambda_perp:      float = 0.1,    # weight of transverse penalty
    device:           str   = "cpu",
):
    """
    Run UOT on one event and return per-hit assignment.

    Thresholding strategies
    ───────────────────────
    adaptive_threshold=True  (recommended):
        Classify as neutral all hits whose transport fraction is below the
        value that makes the total neutral energy equal to
        max(0, E_hits − E_tracks).  This is event-adaptive and satisfies
        energy conservation.

    adaptive_threshold=False:
        Use a fixed frac_transported < neutral_fraction threshold.

    Returns
    -------
    assignment : int32 [n_hits]
        −1  → neutral candidate
        k≥0 → matched to track k (highest transport weight)
    frac_transported : float32 [n_hits]
        Σ_i T_{ij} / ν_j  — fraction of hit energy accounted for by tracks.
    threshold : float
        Threshold used to classify neutral hits.
    dt : float   wall-clock seconds
    """
    n_hits = len(hits["E"])

    if tracks is None or len(tracks["eta"]) == 0:
        return (
            np.full(n_hits, -1, dtype=np.int32),
            np.zeros(n_hits, dtype=np.float32),
            0.0, 0.0,
        )

    # Cost matrix [T, H] — angular + geometric constraints
    C  = combined_cost(
        tracks["eta"], tracks["phi"], tracks["ref_xyz"], tracks["dir_unit"],
        hits["eta"],   hits["phi"],   hits["xyz"],
        max_dR=max_dR, sigma_perp=sigma_perp, lambda_perp=lambda_perp,
    )
    mu = tracks["E_exp"]   # [T]
    nu = hits["E"]          # [H]

    t0 = time.perf_counter()
    log_T = sinkhorn_uot(mu, nu, C, eps=eps, tau=tau, n_iter=n_iter, device=device)
    dt = time.perf_counter() - t0

    T = np.exp(log_T)                                          # [T, H]

    # Fraction of hit energy transported from all tracks
    frac_transported = (T.sum(axis=0) / (nu + 1e-10)).astype(np.float32)

    # Determine threshold
    if adaptive_threshold:
        threshold = _energy_balance_threshold(
            frac_transported, nu, float(mu.sum())
        )
    else:
        threshold = neutral_fraction

    # thr + tiny: out-of-cone hits have frac == 0 exactly (exp(-C_inf/eps) underflows);
    # without the offset, frac < 0 is never true and they'd all be labelled charged.
    neutral_mask = frac_transported < threshold + 1e-9
    # 1-based track label (0-indexed argmax + 1); neutral → -1
    best_track   = np.argmax(T, axis=0).astype(np.int32) + 1
    assignment   = np.where(neutral_mask, np.int32(-1), best_track)

    return assignment, frac_transported, threshold, dt


def make_full_label_vector(event: dict, hits, assignment_cal: np.ndarray) -> np.ndarray:
    """
    Build the final label vector of size N_hit + N_track.

    Layout mirrors the concatenation used throughout the codebase:
        [ hit_0, hit_1, …, hit_{N_hit-1},  track_0, track_1, …, track_{N_trk-1} ]

    Labels
    ------
    -1      : neutral hit  (low UOT transport)  OR  non-calorimeter hit
     k (≥1) : assigned to track k  (1-based index, same for the track node itself)

    Parameters
    ----------
    event          : raw event dict with 'X_hit', 'X_track'
    hits           : dict returned by extract_hits (contains orig_mask)
    assignment_cal : int32 array [n_cal_hits] — per-filtered-hit labels (already 1-based or -1)
    """
    n_hit = len(event["X_hit"])
    n_trk = len(event["X_track"])

    labels = np.full(n_hit + n_trk, -1, dtype=np.int32)

    # ── Hit labels: map filtered-hit assignment back to full hit array ───────
    cal_mask = hits["orig_mask"]              # bool [n_hit]
    labels[:n_hit][cal_mask] = assignment_cal

    # ── Track labels: each track k gets label k+1 ────────────────────────────
    for k in range(n_trk):
        labels[n_hit + k] = k + 1

    return labels


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(assignment: np.ndarray, truth_neutral: np.ndarray) -> dict:
    pred_neutral = assignment == -1
    TP = int(np.sum( pred_neutral &  truth_neutral))
    FP = int(np.sum( pred_neutral & ~truth_neutral))
    TN = int(np.sum(~pred_neutral & ~truth_neutral))
    FN = int(np.sum(~pred_neutral &  truth_neutral))

    precision = TP / (TP + FP + 1e-10)
    recall    = TP / (TP + FN + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-10)

    return dict(
        precision=float(precision), recall=float(recall),
        f1=float(f1), accuracy=float(accuracy),
        TP=TP, FP=FP, TN=TN, FN=FN,
        n_pred_neutral=TP + FP,
        n_pred_charged=TN + FN,
        n_true_neutral=TP + FN,
        n_true_charged=TN + FP,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

_C = dict(TP="#4C9BE8", FP="#FF7F0E", TN="#D62728", FN="#9467BD")


def plot_event(tracks, hits, assignment, truth_neutral, frac_transported,
               event_idx, out_dir, eps, tau, neutral_fraction):
    fig = plt.figure(figsize=(19, 5.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    pred_neutral = assignment == -1
    pred_charged = ~pred_neutral
    sz_h = np.clip(hits["E"] * 40, 4, 200)

    # ── A: η-φ map ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    kw = dict(alpha=0.55, linewidths=0)
    ax.scatter(hits["eta"][pred_neutral], hits["phi"][pred_neutral],
               s=sz_h[pred_neutral], c="#4C9BE8", label="Neutral (pred)", **kw)
    ax.scatter(hits["eta"][pred_charged], hits["phi"][pred_charged],
               s=sz_h[pred_charged], c="#D62728", label="Charged (pred)", **kw)
    if tracks is not None:
        ax.scatter(tracks["eta"], tracks["phi"],
                   s=np.clip(tracks["E_exp"] * 40, 10, 300), c="black",
                   marker="+", linewidths=1.5, zorder=5, label="Tracks")
    ax.set_xlabel("η"); ax.set_ylabel("φ")
    ax.set_title(f"Event {event_idx} – UOT prediction\n(size ∝ energy)")
    ax.legend(fontsize=8)

    # ── B: truth vs prediction ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1])
    conf_items = [
        ("TP", pred_neutral &  truth_neutral, "o", "Correct neutral"),
        ("FP", pred_neutral & ~truth_neutral, "x", "Wrong neutral"),
        ("TN", pred_charged & ~truth_neutral, "s", "Correct charged"),
        ("FN", pred_charged &  truth_neutral, "^", "Missed neutral"),
    ]
    for key, mask, mk, lbl in conf_items:
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(hits["eta"][mask], hits["phi"][mask],
                   s=np.clip(hits["E"][mask] * 40, 4, 200),
                   c=_C[key], marker=mk, alpha=0.65,
                   linewidths=0.5 if mk != "x" else 1.5,
                   label=f"{lbl} ({n})")
    ax.set_xlabel("η"); ax.set_ylabel("φ")
    ax.set_title("Truth vs Prediction")
    ax.legend(fontsize=7)

    # ── C: transport-fraction histogram ──────────────────────────────────────
    ax = fig.add_subplot(gs[2])
    if frac_transported is not None and len(frac_transported) > 0:
        q99 = max(float(np.percentile(frac_transported, 99)), 1e-4)
        bins = np.linspace(0, q99, 60)
        kw2 = dict(histtype="step", lw=2)
        ax.hist(frac_transported[ truth_neutral], bins=bins,
                color="#4C9BE8", label="True neutral", **kw2)
        ax.hist(frac_transported[~truth_neutral], bins=bins,
                color="#D62728", label="True charged", **kw2)
        ax.axvline(neutral_fraction, color="k", ls="--", lw=1.5,
                   label=f"Threshold = {neutral_fraction:.2f}")
        ax.set_xlabel("Fraction of hit energy transported from tracks")
        ax.set_ylabel("Hits"); ax.set_yscale("log")
        ax.set_title("Transport fraction\n(neutrals should cluster left)")
        ax.legend(fontsize=8)

    plt.suptitle(
        f"UOT track-hit matching  |  ε={eps}  τ={tau}  thr={neutral_fraction}",
        fontsize=11, y=1.01
    )
    fname = Path(out_dir) / f"event_{event_idx:04d}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    return fname


def plot_summary(all_metrics, out_dir, eps, tau, neutral_fraction):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, title) in zip(axes.flat, [
        ("precision", "Neutral precision"),
        ("recall",    "Neutral recall"),
        ("f1",        "Neutral F1"),
        ("accuracy",  "Overall accuracy"),
    ]):
        vals = [m[key] for m in all_metrics]
        ax.hist(vals, bins=20, color="#4C9BE8", edgecolor="white")
        ax.axvline(np.mean(vals), color="tomato", ls="--",
                   label=f"mean={np.mean(vals):.3f}")
        ax.set_title(title); ax.legend(fontsize=9)
    plt.suptitle(
        f"Summary – {len(all_metrics)} events | ε={eps} τ={tau} thr={neutral_fraction}",
        fontsize=12)
    plt.tight_layout()
    fname = Path(out_dir) / "summary_metrics.png"
    plt.savefig(fname, dpi=150); plt.close()
    return fname


def plot_aggregate_transport(all_frac_neutral, all_frac_charged,
                             neutral_fraction, out_dir):
    frac_n = np.concatenate(all_frac_neutral)
    frac_c = np.concatenate(all_frac_charged)
    q99 = max(float(np.percentile(np.concatenate([frac_n, frac_c]), 99)), 1e-4)
    bins = np.linspace(0, q99, 80)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(frac_n, bins=bins, histtype="step", lw=2,
            color="#4C9BE8", label=f"True neutral  (n={len(frac_n):,})")
    ax.hist(frac_c, bins=bins, histtype="step", lw=2,
            color="#D62728", label=f"True charged  (n={len(frac_c):,})")
    ax.axvline(neutral_fraction, color="k", ls="--",
               label=f"Threshold={neutral_fraction:.2f}")
    ax.set_xlabel("Fraction of hit energy transported from tracks")
    ax.set_ylabel("Hits"); ax.set_yscale("log")
    ax.set_title("Aggregate transport fraction (all events)")
    ax.legend()
    fname = Path(out_dir) / "transport_fraction_aggregate.png"
    plt.savefig(fname, dpi=150); plt.close()
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  HYPER-PARAMETER SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def scan_hyperparams(data, n_events, n_iter, device, out_dir):
    """Grid search over (ε, τ) and neutral_fraction."""
    eps_vals = [0.02, 0.05, 0.1,  0.2 ]
    tau_vals = [0.02, 0.05, 0.1,  0.2 ]
    thr_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    # Pre-load events
    events_data = []
    for i in range(min(n_events, len(data["X_track"]))):
        ev  = event_to_numpy(data, i)
        trk = extract_tracks(ev)
        hts = extract_hits(ev)
        if hts is None or len(hts["E"]) == 0:
            continue
        events_data.append((trk, hts, get_truth_neutrality(trk, hts)))

    if not events_data:
        print("  No events for scan."); return

    # ε × τ grid (threshold=0.25)
    f1_grid = np.zeros((len(eps_vals), len(tau_vals)))
    for ei, eps in enumerate(eps_vals):
        for ti, tau in enumerate(tau_vals):
            f1s = []
            for trk, hts, truth in events_data:
                asgn, _, _, _ = match_event(
                    trk, hts, eps=eps, tau=tau,
                    n_iter=n_iter, adaptive_threshold=True, device=device)
                f1s.append(evaluate(asgn, truth)["f1"])
            f1_grid[ei, ti] = float(np.mean(f1s))
            print(f"    ε={eps:.2f}  τ={tau:.2f}  "
                  f"F1={f1_grid[ei,ti]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(f1_grid, origin="lower", aspect="auto",
                   extent=[-0.5, len(tau_vals)-.5, -.5, len(eps_vals)-.5],
                   vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
    ax.set_yticks(range(len(eps_vals))); ax.set_yticklabels(eps_vals)
    ax.set_xlabel("τ"); ax.set_ylabel("ε")
    ax.set_title("Mean neutral F1  (threshold=0.25)")
    plt.colorbar(im, ax=ax, label="F1")
    for ei in range(len(eps_vals)):
        for ti in range(len(tau_vals)):
            ax.text(ti, ei, f"{f1_grid[ei,ti]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.tight_layout()
    fname = Path(out_dir) / "scan_eps_tau.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  ε×τ scan → {fname}")

    # Threshold scan using best (ε, τ)
    best_ei, best_ti = np.unravel_index(f1_grid.argmax(), f1_grid.shape)
    best_eps, best_tau = eps_vals[best_ei], tau_vals[best_ti]
    print(f"  Best (ε, τ) = ({best_eps}, {best_tau})", flush=True)

    f1_thr = []
    for thr in thr_vals:
        f1s = []
        for trk, hts, truth in events_data:
            asgn, _, _, _ = match_event(
                trk, hts, eps=best_eps, tau=best_tau,
                n_iter=n_iter, adaptive_threshold=True, device=device)
            f1s.append(evaluate(asgn, truth)["f1"])
        f1_thr.append(float(np.mean(f1s)))
        print(f"    thr={thr:.2f}  F1={f1_thr[-1]:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thr_vals, f1_thr, "o-", color="#4C9BE8", lw=2)
    ax.set_xlabel("Neutral fraction threshold"); ax.set_ylabel("Mean neutral F1")
    ax.set_title(f"Threshold scan  (ε={best_eps}, τ={best_tau})")
    ax.grid(alpha=0.3)
    fname2 = Path(out_dir) / "scan_threshold.png"
    plt.savefig(fname2, dpi=150); plt.close()
    print(f"  Threshold scan → {fname2}")
    best_thr = thr_vals[int(np.argmax(f1_thr))]
    print(f"  Best threshold = {best_thr}  →  F1={max(f1_thr):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="UOT track-hit matching for particle flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--n-events", type=int, default=30,
                        help="Number of events to process")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Entropic regularisation ε")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="KL marginal penalty τ (smaller→more unbalanced→more neutrals)")
    parser.add_argument("--n-iter", type=int, default=300,
                        help="Sinkhorn iterations")
    parser.add_argument("--neutral-fraction", type=float, default=0.25,
                        help="Transport-fraction threshold (used when --fixed-threshold)")
    parser.add_argument("--fixed-threshold", action="store_true",
                        help="Use fixed neutral-fraction threshold instead of energy-balance adaptive")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="PyTorch device")
    parser.add_argument("--scan", action="store_true",
                        help="Run hyper-parameter scan then exit")
    parser.add_argument("--out-dir", default="uot_matching_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  UOT Track-Hit Matching for Particle Flow")
    thresh_mode = "fixed" if args.fixed_threshold else "adaptive (energy-balance)"
    print(f"  ε={args.eps}  τ={args.tau}  n_iter={args.n_iter}"
          f"  threshold={thresh_mode}  device={args.device}")
    print(f"  output → {out_dir}")
    print(f"{sep}\n", flush=True)

    print(f"Loading {args.parquet} …")
    data = load_parquet(args.parquet)
    n_total  = len(data["X_track"])
    n_events = min(args.n_events, n_total)
    print(f"  {n_total} events total, processing {n_events}\n", flush=True)

    if args.scan:
        print("Running hyper-parameter scan …")
        scan_hyperparams(data, n_events=min(10, n_events),
                         n_iter=args.n_iter, device=args.device,
                         out_dir=out_dir)
        print(); return

    # ── Per-event loop ────────────────────────────────────────────────────────
    all_metrics      = []
    all_frac_neutral = []
    all_frac_charged = []
    n_plot = 5

    for i in range(n_events):
        event  = event_to_numpy(data, i)
        tracks = extract_tracks(event)
        hits   = extract_hits(event)

        if hits is None or len(hits["E"]) == 0:
            print(f"  Event {i:3d}: no calorimeter hits — skipped.")
            continue

        truth_neutral = get_truth_neutrality(tracks, hits)
        n_trk = len(tracks["eta"]) if tracks is not None else 0

        assignment, frac_transported, threshold, dt = match_event(
            tracks, hits,
            eps=args.eps, tau=args.tau, n_iter=args.n_iter,
            neutral_fraction=args.neutral_fraction,
            adaptive_threshold=not args.fixed_threshold,
            device=args.device,
        )

        # ── Full label vector [N_hit + N_track] ──────────────────────────────
        # assignment uses 1-based track labels; neutral/non-calo hits → -1
        full_labels = make_full_label_vector(event, hits, assignment)
        if i == 0:
            n_hit = len(event["X_hit"]); n_trk = len(event["X_track"])
            print(f"\n  full_labels shape : {full_labels.shape}  "
                  f"(first {n_hit} = hits, last {n_trk} = tracks)")
            print(f"  unique labels     : {np.unique(full_labels)}")
            print(f"  n neutral (-1)    : {(full_labels[:n_hit] == -1).sum()}")
            print(f"  n charged (≥1)    : {(full_labels[:n_hit]  > 0).sum()}")
            print(f"  track labels      : {full_labels[n_hit:]}\n")

        metrics = evaluate(assignment, truth_neutral)
        all_metrics.append(metrics)
        all_frac_neutral.append(frac_transported[ truth_neutral])
        all_frac_charged.append(frac_transported[~truth_neutral])

        E_rat = tracks["E_exp"].sum() / (hits["E"].sum() + 1e-6) if tracks is not None else 0
        print(
            f"  Event {i:3d}  trk={n_trk:3d}  hits={len(hits['E']):5d}  "
            f"E_trk/E_hit={E_rat:.2f}  thr={threshold:.3f}  "
            f"true_neutral={metrics['n_true_neutral']:5d}  "
            f"pred_neutral={metrics['n_pred_neutral']:5d}  "
            f"prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
            f"F1={metrics['f1']:.3f}  acc={metrics['accuracy']:.3f}  "
            f"t={dt*1e3:.0f}ms",
            flush=True,
        )

        if i < n_plot:
            fname = plot_event(
                tracks, hits, assignment, truth_neutral, frac_transported,
                i, out_dir, args.eps, args.tau, threshold,
            )
            print(f"             → {fname}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_metrics:
        print("No events processed. Check parquet path."); return

    print(f"\n{sep}")
    print(f"  Summary — {len(all_metrics)} events")
    print(f"{sep}")
    print(f"  {'Metric':<28}  {'mean':>7}  {'std':>7}  {'min':>7}  {'max':>7}")
    print("  " + "─" * 56)
    for key in ["precision", "recall", "f1", "accuracy"]:
        v = np.array([m[key] for m in all_metrics])
        print(f"  {'neutral_' + key:<28}  "
              f"{v.mean():7.4f}  {v.std():7.4f}  "
              f"{v.min():7.4f}  {v.max():7.4f}")

    avg = lambda k: np.mean([m[k] for m in all_metrics])
    print(f"\n  Avg pred neutral: {avg('n_pred_neutral'):6.1f}  "
          f"(true: {avg('n_true_neutral'):6.1f})")
    print(f"  Avg pred charged: {avg('n_pred_charged'):6.1f}  "
          f"(true: {avg('n_true_charged'):6.1f})")

    sf  = plot_summary(all_metrics, out_dir, args.eps, args.tau, args.neutral_fraction)
    print(f"\n  Per-metric histograms → {sf}")

    if all_frac_neutral:
        af = plot_aggregate_transport(
            all_frac_neutral, all_frac_charged,
            args.neutral_fraction, out_dir)
        print(f"  Aggregate transport fraction → {af}")

    print(f"\n  All outputs in: {out_dir}/")
    print("Done.\n", flush=True)


if __name__ == "__main__":
    main()
