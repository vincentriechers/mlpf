"""
Shower-shaped anisotropic cost for unbalanced optimal transport hit–track assignment.

The cost encodes the geometry of an electromagnetic / hadronic shower:
  - a narrow core at the shower entrance that widens with depth (cone opening)
  - an asymmetric longitudinal profile (sharp pre-max rise, long post-max tail)
  - a hard 3-D distance cutoff with a minimum-neighbour guarantee

Parameters are tuned for the CLD detector (CLD_o2_v07) geometry:

    ECAL barrel : Si-W,  40 layers × 4.85 mm ≈ 202 mm ≈ 22 X₀
                  inner_r = 2150 mm, outer_r = 2352 mm
                  X₀(W) = 3.5 mm  →  0.56 X₀/layer
                  Effective Molière radius R_M ≈ 15 mm  (CALICE Si-W measurement)

    HCAL barrel : Steel-scintillator, 44 layers × 26.5 mm ≈ 1166 mm ≈ 5 λ_I
                  inner_r = 2400 mm (gap of 48 mm after ECAL)
                  λ_I(steel) ≈ 166 mm  →  0.114 λ_I/layer

The track reference point (ref_xyz from X_track[:,12:15]) is the extrapolated
track position at the ECAL face, so d_par = 0 is the ECAL entrance.

    Depth markers along d_par from ECAL face:
        EM shower max    ≈  60 mm  (7 X₀ × 8.7 mm/X₀)
        ECAL back face   ≈ 202 mm
        HCAL front face  ≈ 250 mm  (+ 48 mm gap)
        Hadronic max     ≈ 450 mm  (1.0 λ_I into HCAL for ~5 GeV pion)
        HCAL back face   ≈ 1416 mm

Usage
-----
    from src.dataset.shower_cost import shower_cost, ShowerCostParams
    from src.dataset.shower_cost import cld_combined_params   # recommended default

    params = cld_combined_params()
    C = shower_cost(tracks, hits, params)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShowerCostParams:
    """All tuneable parameters of the shower-shaped cost.

    Transverse (perpendicular) profile
    -----------------------------------
    sigma0  : Core width at shower entrance [mm].
              Set to the Molière radius of the first absorber material.
    alpha   : Cone opening rate [dimensionless].
              sigma_perp(d_par) = sigma0 + alpha * max(0, d_par).
              For CLD: at hadronic max (d_par≈450mm) this gives
              sigma0 + 450*alpha ≈ shower lateral spread in HCAL.

    Longitudinal (parallel) profile
    --------------------------------
    d_max      : Position of shower maximum along track axis [mm].
    sigma_rise : Width of the upstream (pre-maximum) Gaussian [mm].
                 Should cover the distance from track face to shower max.
    sigma_tail : Width of the downstream (post-maximum) Gaussian [mm].
                 sigma_tail >> sigma_rise reflects the long shower tail.

    Safety cuts
    -----------
    max_d3D   : Hard 3-D distance cutoff [mm].
    n_min_hits: Minimum candidate hits guaranteed per track.
    C_inf     : Prohibitive cost assigned beyond the cutoff.
    """

    # --- transverse ---
    sigma0: float = 30.0       # mm  (tuned on CLD Z→uds events)
    alpha: float  = 0.10       # dimensionless cone opening

    # --- longitudinal ---
    d_max: float       = 300.0    # mm  (shower max near ECAL/HCAL boundary)
    sigma_rise: float  = 300.0    # mm  (flat upstream: accepts all ECAL hits)
    sigma_tail: float  = 1000.0   # mm  (downstream HCAL tail)

    # --- safety ---
    max_d3D: float   = 6000.0   # mm
    n_min_hits: int  = 20
    C_inf: float     = 1e6


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decompose(tracks: dict, hits: dict):
    """
    Decompose hit–track displacement into parallel and perpendicular components.

    Parameters
    ----------
    tracks : dict
        "ref_xyz"  : ndarray [T, 3]  – calo-face entry point of each track
        "dir_unit" : ndarray [T, 3]  – unit momentum direction at calo face
    hits : dict
        "xyz"      : ndarray [H, 3]  – 3-D position of each hit

    Returns
    -------
    d_3D   : [T, H]  total 3-D distance (mm)
    d_par  : [T, H]  signed depth along shower axis (mm)
    d_perp : [T, H]  transverse distance from shower axis (mm)
    """
    delta = hits["xyz"][None, :, :] - tracks["ref_xyz"][:, None, :]   # [T, H, 3]
    d_3D  = np.sqrt(np.sum(delta ** 2, axis=2))                        # [T, H]
    d_par = np.sum(delta * tracks["dir_unit"][:, None, :], axis=2)    # [T, H]
    d_perp_vec = delta - d_par[:, :, None] * tracks["dir_unit"][:, None, :]
    d_perp = np.sqrt(np.sum(d_perp_vec ** 2, axis=2))                 # [T, H]
    return d_3D, d_par, d_perp


# ─────────────────────────────────────────────────────────────────────────────
#  Raw cost (no cutoffs)
# ─────────────────────────────────────────────────────────────────────────────

def _shower_cost_raw(d_par: np.ndarray,
                     d_perp: np.ndarray,
                     p: ShowerCostParams) -> np.ndarray:
    """
    Shower-shaped cost (no hard cutoff applied).

        C = (d_perp / σ_perp(d_par))²  +  ((d_par - d_max) / σ_long)²

    where
        σ_perp(d_par) = sigma0 + alpha * max(0, d_par)   [opening cone]
        σ_long = sigma_rise  if d_par < d_max             [asymmetric]
               = sigma_tail  if d_par ≥ d_max
    """
    # --- transverse: cone opening with depth ---
    sigma_perp = p.sigma0 + p.alpha * np.maximum(0.0, d_par)    # [T, H]
    cost_perp  = (d_perp / sigma_perp) ** 2

    # --- longitudinal: asymmetric Gaussian around shower max ---
    delta_par  = d_par - p.d_max
    sigma_long = np.where(delta_par < 0, p.sigma_rise, p.sigma_tail)
    cost_long  = (delta_par / sigma_long) ** 2

    return cost_perp + cost_long


# ─────────────────────────────────────────────────────────────────────────────
#  Main cost function
# ─────────────────────────────────────────────────────────────────────────────

def shower_cost(tracks: dict,
                hits: dict,
                params: ShowerCostParams | None = None) -> np.ndarray:
    """
    Shower-shaped anisotropic cost matrix for hit–track assignment.

    Parameters
    ----------
    tracks : dict
        "ref_xyz"  : ndarray [T, 3]
        "dir_unit" : ndarray [T, 3]
    hits : dict
        "xyz"      : ndarray [H, 3]
    params : ShowerCostParams, optional
        Uses ``cld_combined_params()`` if None.

    Returns
    -------
    C : ndarray [T, H], float32
    """
    if params is None:
        params = cld_combined_params()

    d_3D, d_par, d_perp = _decompose(tracks, hits)

    C = _shower_cost_raw(d_par, d_perp, params)
    C[d_3D > params.max_d3D] = params.C_inf

    # guarantee at least n_min_hits reachable hits per track
    for k in range(C.shape[0]):
        if int((C[k] < params.C_inf * 0.5).sum()) < params.n_min_hits:
            nearest = np.argsort(d_3D[k])[: params.n_min_hits]
            C[k, nearest] = _shower_cost_raw(
                d_par[k, nearest], d_perp[k, nearest], params
            )

    return C.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Ellipsoidal cost (original, for comparison)
# ─────────────────────────────────────────────────────────────────────────────

def ellipsoidal_cost(tracks: dict,
                     hits: dict,
                     sigma_perp: float = 300.0,
                     sigma_par: float = 4000.0,
                     max_d3D: float = 6000.0,
                     n_min_hits: int = 20,
                     C_inf: float = 1e6) -> np.ndarray:
    """
    Simple anisotropic ellipsoidal cost (baseline formulation).

        C = (d_perp / sigma_perp)² + (d_par / sigma_par)²
    """
    d_3D, d_par, d_perp = _decompose(tracks, hits)

    C = (d_perp / sigma_perp) ** 2 + (d_par / sigma_par) ** 2
    C[d_3D > max_d3D] = C_inf

    for k in range(C.shape[0]):
        if int((C[k] < C_inf * 0.5).sum()) < n_min_hits:
            nearest = np.argsort(d_3D[k])[: n_min_hits]
            C[k, nearest] = (
                (d_perp[k, nearest] / sigma_perp) ** 2
                + (d_par[k, nearest] / sigma_par) ** 2
            )

    return C.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  CLD parameter presets
# ─────────────────────────────────────────────────────────────────────────────

def cld_combined_params() -> ShowerCostParams:
    """
    CLD combined ECAL+HCAL preset for charged hadrons (recommended default).

    Tuned for charged pions in Z→uds events (typical E ~ 1–20 GeV).
    The cost covers the full shower from ECAL face to deep HCAL.

    Derivation
    ----------
    Parameters obtained by grid search over 1600 combinations on 9 Z→uds events
    from CLD_o2_v05 simulation (pf_tree_10601.parquet), maximising mean F1.
    Mean F1=0.864, prec=0.854, rec=0.889 across 9 events.

    sigma0 = 30 mm
        Wider than RM(Si-W)≈15 mm — reflects the hadronic component and
        track extrapolation smearing at the calo face.

    alpha = 0.10
        Cone opens at 10%/mm:
          - At HCAL face  (d_par≈250 mm): σ_perp ≈ 55 mm
          - At shower max (d_par≈300 mm): σ_perp ≈ 60 mm
          - At HCAL back  (d_par≈1416 mm): σ_perp ≈ 172 mm

    d_max = 300 mm
        Effective shower maximum closer to the ECAL/HCAL boundary than
        the naive hadronic estimate (450 mm).  In Z→uds the typical charged
        particle energy is 1–5 GeV so showers don't penetrate deeply.

    sigma_rise = 300 mm
        Covers the full ECAL (202 mm) + transition region with a broad
        upstream window, accepting ECAL hits with low penalty.

    sigma_tail = 1000 mm
        Shorter tail sufficient for Z→uds energies; prevents absorbing
        distant HCAL hits from neutral showers.

    max_d3D = 6000 mm
        Covers full detector reach.
    """
    return ShowerCostParams(
        sigma0=30.0,
        alpha=0.10,
        d_max=300.0,
        sigma_rise=300.0,
        sigma_tail=1000.0,
        max_d3D=6000.0,
        n_min_hits=20,
    )


def cld_ecal_params() -> ShowerCostParams:
    """
    CLD ECAL-only preset for EM showers (electrons / photon conversions).

    ECAL: Si-W, 40 layers × 4.85 mm = 202 mm ≈ 22 X₀
          X₀(W) = 3.5 mm, R_M(Si-W) ≈ 15 mm (CALICE)

    sigma0 = 15 mm
        R_M(Si-W) ≈ 15 mm from CALICE ECAL measurements.
        Contains >90% of the EM shower transversely at the max.

    alpha = 0.06
        EM showers stay narrow; at ECAL back face (d_par=202 mm):
        σ_perp = 15 + 0.06×202 ≈ 27 mm ≈ 1.8 R_M.

    d_max = 60 mm
        EM shower max at ~7 X₀ × (4.85 mm / 0.56 X₀/layer) = 60 mm
        (averaged over E = 1–20 GeV; t_max = ln(E/Ec) ≈ 5–8 X₀).

    sigma_rise = 30 mm  (~3 X₀ upstream of max)
    sigma_tail = 160 mm  (tail to ECAL back: 202 - 60 = 142 mm, round up)

    max_d3D = 2500 mm
        Restricts to ECAL + small margin; prevents absorbing HCAL hits.
    """
    return ShowerCostParams(
        sigma0=15.0,
        alpha=0.06,
        d_max=60.0,
        sigma_rise=30.0,
        sigma_tail=160.0,
        max_d3D=2500.0,
        n_min_hits=10,
    )


def cld_hcal_params() -> ShowerCostParams:
    """
    CLD HCAL-focused preset for deep hadronic showers.

    HCAL: Steel-scintillator, 44 layers × 26.5 mm = 1166 mm ≈ 5 λ_I
          λ_I(steel) ≈ 166 mm, R_M(Fe) ≈ 17.8 mm (PDG)

    sigma0 = 20 mm
        Start at ECAL face (same origin as combined preset);
        hadronic shower hasn't started yet so EM core width applies.

    alpha = 0.14
        Hadronic showers are wider than EM; at hadronic max (d_par≈450mm):
        σ_perp = 20 + 0.14×450 = 83 mm ≈ 0.5 λ_I.
        At HCAL back (d_par≈1416mm): σ_perp ≈ 218 mm ≈ 1.3 λ_I.

    d_max = 500 mm
        Slightly deeper hadronic max (conservative for higher-energy pions).

    sigma_rise = 250 mm
        Broad rise covering ECAL + transition (250 mm) before hadronic max.

    sigma_tail = 3000 mm
        Very long tail: 95% hadronic containment at ~4 λ_I past max
        = 500 + 4×166 ≈ 1164 mm; sigma = 3000 mm keeps deep hits cheap.

    max_d3D = 6000 mm
    """
    return ShowerCostParams(
        sigma0=20.0,
        alpha=0.14,
        d_max=500.0,
        sigma_rise=250.0,
        sigma_tail=3000.0,
        max_d3D=6000.0,
        n_min_hits=20,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    n_tracks, n_hits = 3, 200
    dirs = np.random.randn(n_tracks, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    tracks = {"ref_xyz": np.zeros((n_tracks, 3)), "dir_unit": dirs}
    hits   = {"xyz": np.random.randn(n_hits, 3) * 1500}

    for name, C in [
        ("ellipsoidal ", ellipsoidal_cost(tracks, hits)),
        ("cld_combined", shower_cost(tracks, hits, cld_combined_params())),
        ("cld_ecal    ", shower_cost(tracks, hits, cld_ecal_params())),
        ("cld_hcal    ", shower_cost(tracks, hits, cld_hcal_params())),
    ]:
        finite = C[C < 1e5]
        print(f"{name}  shape={C.shape}  "
              f"min={finite.min():.3f}  median={np.median(finite):.1f}  "
              f"n_finite={len(finite)}")
