"""
Unbalanced Optimal Transport (UOT) track-hit matching.

Given a DGL graph (as built by functions_graph.py), assigns each calorimeter
hit to a charged track or labels it as neutral using a log-domain Sinkhorn
algorithm with geometric constraints.

Main entry point
----------------
    labels = uot_track_hit_labels(g)

Output
------
    int32 tensor of shape [N_nodes]:

        labels[i]  = -1        node i is neutral / non-calo / unassigned
        labels[i]  =  k (≥1)   hit i is assigned to track k  (1-based)
        labels[i]  =  k (≥1)   track node i is labelled k    (1-based)

Graph ndata used
----------------
    hit_type          : float, 1=tracker/track, 2=ECAL, 3=HCAL, 4=muon
    pos_hits_xyz      : [N, 3]  xyz for hits; ref_xyz at calo face for tracks
    pos_pxpypz_at_calo: [N, 3]  zero for hits; momentum direction at calo for tracks
    e_hits            : [N, 1]  energy for hits; zero for tracks
    p_hits            : [N, 1]  zero for hits; |p| for tracks
                        (p_hits > 0 is used to identify track nodes)
"""

import numpy as np
import torch
from src.dataset.shower_cost import (
    shower_cost,
    ShowerCostParams,
    cld_combined_params,
)

try:
    from scipy.special import logsumexp as _sci_logsumexp
    def _logsumexp(a, axis):
        return _sci_logsumexp(a, axis=axis)
except ImportError:
    def _logsumexp(a, axis):
        a_max = np.max(a, axis=axis, keepdims=True)
        out = np.log(np.sum(np.exp(a - a_max), axis=axis))
        return out + np.squeeze(a_max, axis=axis)


# ── Physical constants ────────────────────────────────────────────────────────
_M_PION  = 0.13957   # GeV  (pion hypothesis for track expected energy)
_EPS_LOG = 1e-30     # safe floor for log


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature extraction from DGL graph
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_from_graph(g):
    """
    Extract track and calo-hit features from graph ndata.

    Returns
    -------
    tracks : dict or None
        Keys: E_exp [T], ref_xyz [T,3], dir_unit [T,3]
    hits : dict or None
        Keys: E [H_cal], xyz [H_cal,3]
    track_mask : np.ndarray bool [N_nodes]
        True for track nodes (p_hits > 0).
    cal_mask : np.ndarray bool [N_nodes]
        True for ECAL/HCAL nodes with positive energy.
    """
    hit_type = g.ndata["hit_type"].cpu().numpy().astype(np.int32)   # [N]
    p_hits   = g.ndata["p_hits"].squeeze(1).cpu().numpy()            # [N]
    e_hits   = g.ndata["e_hits"].squeeze(1).cpu().numpy()            # [N]
    pos_xyz  = g.ndata["pos_hits_xyz"].cpu().numpy()                 # [N, 3]
    p_calo   = g.ndata["pos_pxpypz_at_calo"].cpu().numpy()           # [N, 3]

    # ── Tracks: nodes with non-zero momentum ─────────────────────────────────
    track_mask = p_hits > 0                                           # [N] bool

    if track_mask.sum() == 0:
        return None, None, track_mask, np.zeros(len(hit_type), dtype=bool)

    ref_xyz  = pos_xyz[track_mask]                                    # [T, 3]
    p_mag    = p_hits[track_mask]                                     # [T]
    E_exp    = np.sqrt(p_mag**2 + _M_PION**2)                        # [T]
    dir_calo = p_calo[track_mask]                                     # [T, 3]
    norm     = np.linalg.norm(dir_calo, axis=1, keepdims=True) + 1e-8
    dir_unit = dir_calo / norm                                        # [T, 3]

    tracks = dict(E_exp=E_exp, ref_xyz=ref_xyz, dir_unit=dir_unit)

    # ── Calo hits: ECAL (2) or HCAL (3) with positive energy, not tracks ─────
    cal_mask = (~track_mask) & (hit_type >= 2) & (hit_type <= 3) & (e_hits > 0)

    if cal_mask.sum() == 0:
        return tracks, None, track_mask, cal_mask

    hits = dict(E=e_hits[cal_mask], xyz=pos_xyz[cal_mask])

    return tracks, hits, track_mask, cal_mask


# ═══════════════════════════════════════════════════════════════════════════════
#  Cost matrix
# ═══════════════════════════════════════════════════════════════════════════════

def _decompose(tracks, hits):
    """
    Decompose hit-track displacement into parallel and perpendicular components.

    Returns
    -------
    d_3D   : [T, H] total 3-D distance (mm)
    d_par  : [T, H] signed distance along track shower axis
    d_perp : [T, H] transverse distance from shower axis
    """
    delta      = hits["xyz"][None, :, :] - tracks["ref_xyz"][:, None, :]   # [T,H,3]
    d_3D       = np.sqrt(np.sum(delta**2, axis=2))                          # [T,H]
    d_par      = np.sum(delta * tracks["dir_unit"][:, None, :], axis=2)     # [T,H]
    d_perp_vec = delta - d_par[:, :, None] * tracks["dir_unit"][:, None, :]
    d_perp     = np.sqrt(np.sum(d_perp_vec**2, axis=2))                     # [T,H]
    return d_3D, d_par, d_perp


def _combined_cost(tracks, hits,
                   sigma_perp=300.0, max_d3D=6000.0,
                   n_min_hits=20, C_inf=1e6):
    """
    Isotropic cost: C = (d_perp / sigma_perp)²
    """
    d_3D, _d_par, d_perp = _decompose(tracks, hits)

    C = (d_perp / sigma_perp) ** 2
    C[d_3D > max_d3D] = C_inf

    for k in range(C.shape[0]):
        if int((C[k] < C_inf * 0.5).sum()) < n_min_hits:
            nearest = np.argsort(d_3D[k])[:n_min_hits]
            C[k, nearest] = (d_perp[k, nearest] / sigma_perp) ** 2

    return C.astype(np.float32)


def _anisotropic_cost(tracks, hits,
                      sigma_perp=300.0, sigma_par=4000.0,
                      max_d3D=6000.0, n_min_hits=20, C_inf=1e6):
    """
    Anisotropic cost: C = (d_perp / sigma_perp)² + (d_par / sigma_par)²

    Moving along the shower axis is cheap; moving sideways is expensive.
    """
    d_3D, d_par, d_perp = _decompose(tracks, hits)

    C = (d_perp / sigma_perp) ** 2 + (d_par / sigma_par) ** 2
    C[d_3D > max_d3D] = C_inf

    for k in range(C.shape[0]):
        if int((C[k] < C_inf * 0.5).sum()) < n_min_hits:
            nearest = np.argsort(d_3D[k])[:n_min_hits]
            C[k, nearest] = (
                (d_perp[k, nearest] / sigma_perp) ** 2
                + (d_par[k, nearest] / sigma_par) ** 2
            )

    return C.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Sinkhorn UOT (log-domain, numpy + scipy)
# ═══════════════════════════════════════════════════════════════════════════════

def _sinkhorn_uot(mu, nu, C, eps=0.02, tau=0.20, n_iter=300, tol=1e-6):
    """
    Log-domain unbalanced Sinkhorn (Chizat et al. 2018).

    Returns log-transport plan logT of shape [T, H].
    """
    mu64 = mu.astype(np.float64)
    nu64 = nu.astype(np.float64)
    C64  = C.astype(np.float64)

    log_mu = np.log(mu64 + _EPS_LOG)
    log_nu = np.log(nu64 + _EPS_LOG)
    M      = -C64 / eps
    alpha  = tau / (tau + eps)

    a = np.zeros(len(mu64), dtype=np.float64)
    b = np.zeros(len(nu64), dtype=np.float64)

    for it in range(n_iter):
        a_new = alpha * (log_mu - _logsumexp(b[None, :] + M, axis=1))
        b_new = alpha * (log_nu - _logsumexp(a_new[:, None] + M, axis=0))
        if it > 0:
            if max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b))) < tol:
                a, b = a_new, b_new
                break
        a, b = a_new, b_new

    return (a[:, None] + b[None, :] + M).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main matching function
# ═══════════════════════════════════════════════════════════════════════════════

def uot_track_hit_labels(
    g,
    eps=0.02,
    tau=0.20,
    n_iter=300,
    sigma_perp=300.0,
    max_d3D=6000.0,
    n_min_hits=20,
    cost_mode="shower",
    sigma_par=4000.0,
    shower_params=None,
):
    """
    Run UOT track-hit matching on a DGL graph and return a label tensor.

    Parameters
    ----------
    g : dgl.DGLGraph
        Graph built by functions_graph.py.  Must have ndata keys:
        hit_type, pos_hits_xyz, pos_pxpypz_at_calo, e_hits, p_hits.
    eps : float
        Sinkhorn regularisation.  Smaller → sharper per-track assignments.
    tau : float
        KL marginal penalty.  Smaller → more unbalanced (fewer forced hits).
    n_iter : int
        Maximum Sinkhorn iterations.
    sigma_perp : float
        Scale of transverse-distance cost (mm).
    max_d3D : float
        Hard 3-D distance cutoff from the track calo entry point (mm).
    n_min_hits : int
        Minimum hits guaranteed reachable per track (expands cutoff if needed).
    cost_mode : str
        ``"shower"`` (default) — physics-motivated cone cost tuned for CLD:
        opening transverse profile + asymmetric longitudinal Gaussian.
        ``"anisotropic"`` — ellipsoidal: ``C = (d_perp/sigma_perp)² + (d_par/sigma_par)²``.
        ``"isotropic"``   — purely transverse: ``C = (d_perp/sigma_perp)²``.
    sigma_par : float
        Scale of along-axis cost for ``"anisotropic"`` mode (mm).
    shower_params : ShowerCostParams, optional
        Parameters for ``"shower"`` mode.  Uses ``cld_combined_params()``
        if None.

    Returns
    -------
    labels : torch.Tensor, shape [N_nodes], dtype int32
        -1   → neutral / non-calo / unassigned node
         k≥1 → assigned to track k (1-based); track nodes also carry k.
    """
    n_nodes = g.num_nodes()
    labels  = torch.full((n_nodes,), -1, dtype=torch.int32)

    tracks, hits, track_mask, cal_mask = _extract_from_graph(g)

    # Label track nodes 1-based in their order of appearance
    track_indices = np.where(track_mask)[0]
    for k, idx in enumerate(track_indices):
        labels[int(idx)] = k + 1

    if tracks is None or hits is None:
        return labels

    mu = tracks["E_exp"]   # [T]
    nu = hits["E"]         # [H_cal]

    # ── Cost matrix ───────────────────────────────────────────────────────────
    if cost_mode == "shower":
        params = shower_params if shower_params is not None else cld_combined_params()
        C = shower_cost(tracks, hits, params)
    elif cost_mode == "anisotropic":
        C = _anisotropic_cost(
            tracks, hits,
            sigma_perp=sigma_perp, sigma_par=sigma_par,
            max_d3D=max_d3D, n_min_hits=n_min_hits,
        )
    else:
        C = _combined_cost(
            tracks, hits,
            sigma_perp=sigma_perp, max_d3D=max_d3D, n_min_hits=n_min_hits,
        )

    # ── Sinkhorn ─────────────────────────────────────────────────────────────
    logT = _sinkhorn_uot(mu, nu, C, eps=eps, tau=tau, n_iter=n_iter)
    T    = np.exp(logT)                                           # [T, H_cal]

    frac = (T.sum(0) / (nu + 1e-10)).astype(np.float32)          # [H_cal]

    # ── Adaptive energy-balance threshold ─────────────────────────────────────
    E_neutral_exp = max(0.0, float(nu.sum()) - float(mu.sum()))

    if E_neutral_exp > 0:
        order     = np.argsort(frac)
        cum_E     = np.cumsum(nu[order])
        idx_cross = min(np.searchsorted(cum_E, E_neutral_exp), len(frac) - 1)
        thr       = float(frac[order[idx_cross]])
    else:
        thr = -1.0

    best_track  = np.argmax(T, axis=0).astype(np.int32)          # 0-based [H_cal]
    neutral_mask = frac < thr + 1e-9
    assignment  = np.where(neutral_mask, np.int32(-1), best_track + 1)

    # ── Per-track energy cap (ratio ≤ 1) ─────────────────────────────────────
    for k in range(len(mu)):
        mask_k  = assignment == (k + 1)
        E_k     = float(nu[mask_k].sum())
        E_exp_k = float(mu[k])
        if E_k <= E_exp_k:
            continue
        idxs    = np.where(mask_k)[0]
        order_k = idxs[np.argsort(frac[idxs])]
        for j in order_k:
            if E_k <= E_exp_k:
                break
            E_k -= float(nu[j])
            assignment[j] = -1

    # ── Map calo-hit assignments back to graph node indices ───────────────────
    cal_indices = np.where(cal_mask)[0]
    labels[cal_indices] = torch.tensor(assignment, dtype=torch.int32)

    return labels
