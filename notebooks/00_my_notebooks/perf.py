"""
perf.py

Unified performance computations on MLPF parquet samples.

Sections:
  1) Tracking: efficiency + resolution
  2) Calorimetry / energy deposition:
     - Event-level (visible E_true vs hit-sum E_reco)
     - Particle-level (E_true vs assigned hit energy via ygen_hit)

Keep plotting in plotting.py, keep notebook thin.
"""

from __future__ import annotations

import glob
from typing import Dict, Any, Iterator, Optional, Tuple, List

import numpy as np
import awkward as ak
from scipy.optimize import curve_fit


# ====================================================================
# Common I/O
# ====================================================================
def list_parquet_files(parquet_dir: str, n_files: int) -> List[str]:
    files = sorted(glob.glob(f"{parquet_dir}/*.parquet"))
    return files[:n_files]


def iter_events(parquet_files: List[str], fields: Tuple[str, ...]) -> Iterator[Dict[str, Any]]:
    cols = list(fields)
    for fn in parquet_files:
        data = ak.from_parquet(fn, columns=cols)
        n_ev = len(data[cols[0]])
        for i in range(n_ev):
            yield {k: data[k][i] for k in cols}


def eta_to_theta_deg(eta: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.arctan(np.exp(-eta))
    return np.degrees(theta)


# ====================================================================
# Tracking selections
# ====================================================================
def primary_mask(gen: np.ndarray, idx: Dict[str, int], r_max: float, z_max: float) -> np.ndarray:
    status = gen[:, idx["GEN_STATUS"]]
    vx = gen[:, idx["GEN_VX"]]
    vy = gen[:, idx["GEN_VY"]]
    vz = gen[:, idx["GEN_VZ"]]
    r = np.sqrt(vx**2 + vy**2)
    return (status == 1) & (r < r_max) & (np.abs(vz) < z_max)


def pdg_mask(gen: np.ndarray, idx: Dict[str, int], pdg_filter: Optional[List[int]]) -> np.ndarray:
    if pdg_filter is None:
        return np.ones(len(gen), dtype=bool)
    pdg = gen[:, idx["GEN_PDG"]]
    return np.isin(pdg, pdg_filter)


def reconstructable_mask(
    gen: np.ndarray,
    idx: Dict[str, int],
    pt_min: float,
    eta_max: float,
    cos_theta_max: float,
    primary_r_max: float,
    primary_z_max: float,
    pdg_filter: Optional[List[int]] = None,
) -> np.ndarray:
    status = gen[:, idx["GEN_STATUS"]]
    charge = gen[:, idx["GEN_CHARGE"]]
    pt = gen[:, idx["GEN_PT"]]
    eta = gen[:, idx["GEN_ETA"]]

    th = eta_to_theta_deg(eta)
    cos_th = np.cos(np.radians(th))

    prim = primary_mask(gen, idx, primary_r_max, primary_z_max)
    pdg_ok = pdg_mask(gen, idx, pdg_filter)

    return (
        (status == 1)
        & (np.abs(charge) > 0)
        & (pt > pt_min)
        & (np.abs(eta) < eta_max)
        & (np.abs(cos_th) < cos_theta_max)
        & prim
        & pdg_ok
    )


# ====================================================================
# 1) TRACKING: Efficiency
# ====================================================================
def compute_tracking_efficiency_binned(
    parquet_dir: str,
    n_files: int,
    idx: Dict[str, int],
    pt_bins: Optional[np.ndarray] = None,
    theta_bins: Optional[np.ndarray] = None,
    theta_regions: Optional[np.ndarray] = None,
    pt_min: float = 0.1,
    eta_max: float = 10.0,
    cos_theta_max: float = 0.99,
    primary_r_max: float = 9999.0,
    primary_z_max: float = 9999.0,
    pdg_filter: Optional[List[int]] = None,
) -> Dict[str, Any]:
    files = list_parquet_files(parquet_dir, n_files)
    events = iter_events(files, ("X_gen", "ygen_track"))

    out: Dict[str, Any] = {"eff_global": np.nan, "eff_global_err": np.nan}

    if pt_bins is not None:
        pt_bins = np.asarray(pt_bins, float)
        out["pt_bins"] = pt_bins
        out["pt_num"] = np.zeros(len(pt_bins) - 1, int)
        out["pt_den"] = np.zeros(len(pt_bins) - 1, int)

    if theta_bins is not None:
        theta_bins = np.asarray(theta_bins, float)
        out["theta_bins"] = theta_bins
        out["theta_num"] = np.zeros(len(theta_bins) - 1, int)
        out["theta_den"] = np.zeros(len(theta_bins) - 1, int)

    if (pt_bins is not None) and (theta_bins is not None):
        out["num_p_theta"] = np.zeros((len(pt_bins) - 1, len(theta_bins) - 1), int)
        out["den_p_theta"] = np.zeros((len(pt_bins) - 1, len(theta_bins) - 1), int)

    if (theta_regions is not None) and (pt_bins is not None):
        theta_regions = np.asarray(theta_regions, float)
        out["theta_regions"] = theta_regions
        out["pt_num_theta"] = np.zeros((len(theta_regions), len(pt_bins) - 1), int)
        out["pt_den_theta"] = np.zeros((len(theta_regions), len(pt_bins) - 1), int)

    total_den, total_num = 0, 0

    for ev in events:
        gen = np.asarray(ev["X_gen"])
        ytrk = np.asarray(ev["ygen_track"])
        if gen.size == 0:
            continue

        recoable = reconstructable_mask(
            gen, idx, pt_min, eta_max, cos_theta_max, primary_r_max, primary_z_max, pdg_filter
        )
        if not np.any(recoable):
            continue

        n_gen = len(gen)
        reco_gen_idx = np.where(recoable)[0]

        y_valid = (ytrk >= 0) & (ytrk < n_gen)
        if np.any(y_valid):
            matched_all = np.unique(ytrk[y_valid].astype(int))
            matched = np.intersect1d(matched_all, reco_gen_idx, assume_unique=False)
        else:
            matched = np.array([], dtype=int)

        total_den += len(reco_gen_idx)
        total_num += len(matched)

        pt = gen[:, idx["GEN_PT"]]
        th = eta_to_theta_deg(gen[:, idx["GEN_ETA"]])

        pt_reco = pt[recoable]
        th_reco = th[recoable]
        is_matched = np.isin(reco_gen_idx, matched)

        if "pt_den" in out:
            ib = np.digitize(pt_reco, out["pt_bins"]) - 1
            for i in ib:
                if 0 <= i < len(out["pt_den"]):
                    out["pt_den"][i] += 1
            ibm = np.digitize(pt_reco[is_matched], out["pt_bins"]) - 1
            for i in ibm:
                if 0 <= i < len(out["pt_num"]):
                    out["pt_num"][i] += 1

        if "theta_den" in out:
            it = np.digitize(th_reco, out["theta_bins"]) - 1
            for i in it:
                if 0 <= i < len(out["theta_den"]):
                    out["theta_den"][i] += 1
            itm = np.digitize(th_reco[is_matched], out["theta_bins"]) - 1
            for i in itm:
                if 0 <= i < len(out["theta_num"]):
                    out["theta_num"][i] += 1

        if "den_p_theta" in out:
            ip = np.digitize(pt_reco, out["pt_bins"]) - 1
            it = np.digitize(th_reco, out["theta_bins"]) - 1
            good = (ip >= 0) & (ip < out["den_p_theta"].shape[0]) & (it >= 0) & (it < out["den_p_theta"].shape[1])
            for a, b in zip(ip[good], it[good]):
                out["den_p_theta"][a, b] += 1
            ipm = ip[is_matched]
            itm = it[is_matched]
            goodm = (ipm >= 0) & (ipm < out["num_p_theta"].shape[0]) & (itm >= 0) & (itm < out["num_p_theta"].shape[1])
            for a, b in zip(ipm[goodm], itm[goodm]):
                out["num_p_theta"][a, b] += 1

        if ("pt_den_theta" in out) and ("theta_regions" in out):
            for ir, (tmin, tmax) in enumerate(out["theta_regions"]):
                inreg = (th_reco >= tmin) & (th_reco < tmax)
                if not np.any(inreg):
                    continue
                ib = np.digitize(pt_reco[inreg], out["pt_bins"]) - 1
                for i in ib:
                    if 0 <= i < out["pt_den_theta"].shape[1]:
                        out["pt_den_theta"][ir, i] += 1
                inreg_m = inreg & is_matched
                ibm = np.digitize(pt_reco[inreg_m], out["pt_bins"]) - 1
                for i in ibm:
                    if 0 <= i < out["pt_num_theta"].shape[1]:
                        out["pt_num_theta"][ir, i] += 1

    if total_den > 0:
        eff = total_num / total_den
        err = np.sqrt(eff * (1 - eff) / total_den)
    else:
        eff, err = np.nan, np.nan
    out["eff_global"], out["eff_global_err"] = eff, err

    if "pt_den" in out:
        den = out["pt_den"].astype(float)
        num = out["pt_num"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["pt_eff"] = num / den
        out["pt_err"] = np.where(den > 0, np.sqrt(out["pt_eff"] * (1 - out["pt_eff"]) / den), np.nan)

    if "theta_den" in out:
        den = out["theta_den"].astype(float)
        num = out["theta_num"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["theta_eff"] = num / den
        out["theta_err"] = np.where(den > 0, np.sqrt(out["theta_eff"] * (1 - out["theta_eff"]) / den), np.nan)

    if "den_p_theta" in out:
        den = out["den_p_theta"].astype(float)
        num = out["num_p_theta"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["eff_p_theta"] = num / den
        out["err_p_theta"] = np.where(den > 0, np.sqrt(out["eff_p_theta"] * (1 - out["eff_p_theta"]) / den), np.nan)

    if "pt_den_theta" in out:
        den = out["pt_den_theta"].astype(float)
        num = out["pt_num_theta"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["pt_eff_theta"] = num / den
        out["pt_err_theta"] = np.where(den > 0, np.sqrt(out["pt_eff_theta"] * (1 - out["pt_eff_theta"]) / den), np.nan)

    return out


# ====================================================================
# 2) TRACKING: Resolution
# ====================================================================
def _gauss(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_3sigma_clipped(residuals: np.ndarray, min_entries: int = 50) -> Tuple[float, float]:
    residuals = np.asarray(residuals, float)
    if residuals.size < min_entries:
        return np.nan, np.nan

    q16, q84 = np.percentile(residuals, [16, 84])
    sig0 = 0.5 * (q84 - q16)
    if not np.isfinite(sig0) or sig0 <= 0:
        return np.nan, np.nan

    m = np.abs(residuals - 0.0) < 3.0 * sig0
    res = residuals[m]
    if res.size < min_entries:
        return np.nan, np.nan

    nb = max(10, min(40, res.size // 5))
    counts, edges = np.histogram(res, bins=nb)
    xc = 0.5 * (edges[:-1] + edges[1:])
    ok = counts > 0
    if np.count_nonzero(ok) < 5:
        return np.nan, np.nan

    xfit, yfit = xc[ok], counts[ok]
    A0, mu0, s0 = yfit.max(), 0.0, np.std(res)
    if not np.isfinite(s0) or s0 <= 0:
        return np.nan, np.nan

    try:
        popt, _ = curve_fit(_gauss, xfit, yfit, p0=[A0, mu0, s0], maxfev=5000)
        _, mu, sig = popt
        if not np.isfinite(sig) or sig <= 0:
            return np.nan, np.nan
        return float(sig), float(mu)
    except Exception:
        return np.nan, np.nan


def compute_track_resolution_p_theta(
    parquet_dir: str,
    n_files: int,
    idx: Dict[str, int],
    p_bins: np.ndarray,
    theta_bins: np.ndarray,
    observable: str = "d0",    # "d0" or "pt"
    min_entries: int = 50,
    pt_min: float = 0.1,
    eta_max: float = 10.0,
    cos_theta_max: float = 0.99,
    primary_r_max: float = 9999.0,
    primary_z_max: float = 9999.0,
    pdg_filter: Optional[List[int]] = None,
) -> Dict[str, Any]:
    p_bins = np.asarray(p_bins, float)
    theta_bins = np.asarray(theta_bins, float)
    n_p = len(p_bins) - 1
    n_th = len(theta_bins) - 1

    files = list_parquet_files(parquet_dir, n_files)
    events = iter_events(files, ("X_gen", "X_track", "ygen_track"))

    bins = [[] for _ in range(n_p * n_th)]

    for ev in events:
        gen = np.asarray(ev["X_gen"])
        trk = np.asarray(ev["X_track"])
        y = np.asarray(ev["ygen_track"])

        if gen.size == 0 or trk.size == 0:
            continue

        recoable = reconstructable_mask(
            gen, idx, pt_min, eta_max, cos_theta_max, primary_r_max, primary_z_max, pdg_filter
        )
        n_gen = len(gen)

        yv = (y >= 0) & (y < n_gen)
        if not np.any(yv):
            continue

        igen = y[yv].astype(int)
        trk_sel = trk[yv]
        keep = recoable[igen]
        if not np.any(keep):
            continue

        igen = igen[keep]
        trk_sel = trk_sel[keep]

        p_true = gen[igen, idx["GEN_P"]]
        th_true = eta_to_theta_deg(gen[igen, idx["GEN_ETA"]])

        if observable == "d0":
            residual = trk_sel[:, idx["TRK_D0"]]
            ylabel = r"$\sigma(\Delta d_0)$"
        elif observable == "pt":
            pt_true = gen[igen, idx["GEN_PT"]]
            pt_reco = trk_sel[:, idx["TRK_PT"]]
            m = pt_true > 0
            if not np.any(m):
                continue
            p_true = p_true[m]
            th_true = th_true[m]
            residual = (pt_true[m] - pt_reco[m]) / (pt_true[m] ** 2)
            ylabel = r"$\sigma((p_{T,true}-p_{T,reco})/p_{T,true}^2)$ [GeV$^{-1}$]"
        else:
            raise ValueError("observable must be 'd0' or 'pt'")

        ip = np.digitize(p_true, p_bins) - 1
        it = np.digitize(th_true, theta_bins) - 1
        good = (ip >= 0) & (ip < n_p) & (it >= 0) & (it < n_th)
        if not np.any(good):
            continue

        ip = ip[good]
        it = it[good]
        res = residual[good]
        flat = ip * n_th + it

        for b in np.unique(flat):
            m = flat == b
            bins[b].extend(res[m].tolist())

    sigma = np.full((n_p, n_th), np.nan)
    mu = np.full((n_p, n_th), np.nan)
    nent = np.zeros((n_p, n_th), int)

    for ip in range(n_p):
        for it in range(n_th):
            flat = ip * n_th + it
            arr = np.asarray(bins[flat], float)
            nent[ip, it] = arr.size
            if arr.size < min_entries:
                continue
            sig, mean = fit_gaussian_3sigma_clipped(arr, min_entries=min_entries)
            sigma[ip, it] = sig
            mu[ip, it] = mean

    return {
        "p_bins": p_bins,
        "theta_bins": theta_bins,
        "sigma": sigma,
        "mu": mu,
        "n_entries": nent,
        "observable": observable,
        "ylabel": ylabel,
    }


# ====================================================================
# 3) CALO / ENERGY DEPOSITION: Event-level + Particle-level
# ====================================================================
def _bin_edges_from_data(x: np.ndarray, bin_width: float, xmin: Optional[float] = None) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        raise RuntimeError("Empty array for binning.")
    x_min = float(np.min(x) if xmin is None else xmin)
    x_max = float(np.max(x))
    if x_max <= x_min:
        raise RuntimeError("Invalid range for binning.")
    n_bins = int(np.ceil((x_max - x_min) / bin_width))
    return x_min + np.arange(n_bins + 1) * bin_width


def _sigma_err_from_n(sigma: np.ndarray, n: np.ndarray) -> np.ndarray:
    return sigma / np.sqrt(2.0 * np.maximum(n - 1, 1))


def compute_event_energy_binned(
    parquet_dir: str,
    n_files: int,
    idx: Dict[str, int],
    bin_width: float = 5.0,
    E_true_bins: Optional[np.ndarray] = None,
    exclude_neutrinos: bool = True,
    exclude_muons_in_true: bool = False,
    allowed_subdets: Optional[List[int]] = None,
    E_true_min_for_autobin: Optional[float] = None,
) -> Dict[str, Any]:
    files = list_parquet_files(parquet_dir, n_files)
    events = iter_events(files, ("X_gen", "X_hit"))

    E_true_list, E_reco_list = [], []

    for ev in events:
        gen = np.asarray(ev["X_gen"])
        hits = np.asarray(ev["X_hit"])
        if gen.size == 0 or hits.size == 0:
            continue

        status = gen[:, idx["GEN_STATUS"]]
        pdg = gen[:, idx["GEN_PDG"]]
        vis = (status == 1)

        if exclude_neutrinos:
            vis &= ~np.isin(np.abs(pdg), [12, 14, 16])
        if exclude_muons_in_true:
            vis &= ~(np.abs(pdg) == 13)

        if not np.any(vis):
            continue

        E_true = float(np.sum(gen[vis, idx["GEN_E"]]))
        if E_true <= 0:
            continue

        Ehit = hits[:, idx["HIT_E"]]
        if allowed_subdets is not None:
            subdet = hits[:, idx["HIT_SUBDET"]].astype(int)
            Ehit = Ehit[np.isin(subdet, allowed_subdets)]
        E_reco = float(np.sum(Ehit))

        E_true_list.append(E_true)
        E_reco_list.append(E_reco)

    E_true_arr = np.asarray(E_true_list, float)
    E_reco_arr = np.asarray(E_reco_list, float)
    if E_true_arr.size == 0:
        raise RuntimeError("No valid event energies found.")

    if E_true_bins is None:
        E_true_bins = _bin_edges_from_data(E_true_arr, bin_width, xmin=E_true_min_for_autobin)
    else:
        E_true_bins = np.asarray(E_true_bins, float)

    n_bins = len(E_true_bins) - 1
    centers = 0.5 * (E_true_bins[:-1] + E_true_bins[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = E_reco_arr / E_true_arr
        R = (E_reco_arr - E_true_arr) / E_true_arr

    ib = np.digitize(E_true_arr, E_true_bins) - 1

    mean_ratio = np.full(n_bins, np.nan)
    mean_ratio_err = np.full(n_bins, np.nan)
    sigma = np.full(n_bins, np.nan)
    n_entries = np.zeros(n_bins, int)

    for i in range(n_bins):
        m = ib == i
        n = int(np.sum(m))
        n_entries[i] = n
        if n <= 0:
            continue

        rr = ratio[m]
        rres = R[m]

        mean_ratio[i] = float(np.mean(rr))
        if n > 1:
            mean_ratio_err[i] = float(np.std(rr, ddof=1) / np.sqrt(n))
            sigma[i] = float(np.std(rres, ddof=1))
        else:
            mean_ratio_err[i] = np.nan
            sigma[i] = np.nan

    sigma_err = _sigma_err_from_n(sigma, n_entries)

    return {
        "E_true_bins": E_true_bins,
        "E_true_centers": centers,
        "mean_ratio": mean_ratio,
        "mean_ratio_err": mean_ratio_err,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "n_entries": n_entries,
        "residual_mode": "R = (E_reco - E_true) / E_true",
        "E_true_events": E_true_arr,
        "E_reco_events": E_reco_arr,
    }


def compute_particle_energy_binned(
    parquet_dir: str,
    n_files: int,
    idx: Dict[str, int],
    pdg_filter: Tuple[int, ...],
    bin_width: float = 1.0,
    E_true_bins: Optional[np.ndarray] = None,
    allowed_subdets: Optional[List[int]] = None,
    E_true_min_for_autobin: Optional[float] = None,
    E_true_max_for_autobin: Optional[float] = None,
) -> Dict[str, Any]:
    files = list_parquet_files(parquet_dir, n_files)
    events = iter_events(files, ("X_gen", "X_hit", "ygen_hit"))

    E_true_all, E_reco_all = [], []

    for ev in events:
        gen = np.asarray(ev["X_gen"])
        hits = np.asarray(ev["X_hit"])
        yhit = np.asarray(ev["ygen_hit"])

        n_gen = len(gen)
        if n_gen == 0 or len(hits) == 0:
            continue

        status = gen[:, idx["GEN_STATUS"]]
        pdg = gen[:, idx["GEN_PDG"]]
        sel = (status == 1) & np.isin(pdg, pdg_filter)
        if not np.any(sel):
            continue

        valid = (yhit >= 0) & (yhit < n_gen)
        if allowed_subdets is not None:
            subdet = hits[:, idx["HIT_SUBDET"]].astype(int)
            valid &= np.isin(subdet, allowed_subdets)
        if not np.any(valid):
            continue

        gidx = yhit[valid].astype(int)
        Ehit = hits[valid, idx["HIT_E"]]
        E_reco_per_gen = np.bincount(gidx, weights=Ehit, minlength=n_gen)

        E_true_sel = gen[sel, idx["GEN_E"]]
        E_reco_sel = E_reco_per_gen[sel]

        mE = (E_true_sel > 0)
        if E_true_max_for_autobin is not None:
            mE &= (E_true_sel <= E_true_max_for_autobin)
        if not np.any(mE):
            continue

        E_true_all.extend(E_true_sel[mE].tolist())
        E_reco_all.extend(E_reco_sel[mE].tolist())

    E_true_arr = np.asarray(E_true_all, float)
    E_reco_arr = np.asarray(E_reco_all, float)
    if E_true_arr.size == 0:
        raise RuntimeError("No particle energies found for this selection.")

    if E_true_bins is None:
        E_for_bins = E_true_arr
        if E_true_max_for_autobin is not None:
            E_for_bins = E_for_bins[E_for_bins <= E_true_max_for_autobin]
            if E_for_bins.size == 0:
                E_for_bins = E_true_arr
        if E_true_min_for_autobin is None:
            E_true_min_for_autobin = float(np.min(E_for_bins))
        E_true_bins = _bin_edges_from_data(E_for_bins, bin_width, xmin=E_true_min_for_autobin)
    else:
        E_true_bins = np.asarray(E_true_bins, float)

    n_bins = len(E_true_bins) - 1
    centers = 0.5 * (E_true_bins[:-1] + E_true_bins[1:])

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = E_reco_arr / E_true_arr
        R = (E_reco_arr - E_true_arr) / E_true_arr

    ib = np.digitize(E_true_arr, E_true_bins) - 1

    mean_ratio = np.full(n_bins, np.nan)
    mean_ratio_err = np.full(n_bins, np.nan)
    sigma = np.full(n_bins, np.nan)
    n_entries = np.zeros(n_bins, int)

    for i in range(n_bins):
        m = ib == i
        n = int(np.sum(m))
        n_entries[i] = n
        if n <= 0:
            continue
        rr = ratio[m]
        rres = R[m]
        mean_ratio[i] = float(np.mean(rr))
        if n > 1:
            mean_ratio_err[i] = float(np.std(rr, ddof=1) / np.sqrt(n))
            sigma[i] = float(np.std(rres, ddof=1))
        else:
            mean_ratio_err[i] = np.nan
            sigma[i] = np.nan

    sigma_err = _sigma_err_from_n(sigma, n_entries)

    return {
        "E_true_bins": E_true_bins,
        "E_true_centers": centers,
        "mean_ratio": mean_ratio,
        "mean_ratio_err": mean_ratio_err,
        "sigma": sigma,
        "sigma_err": sigma_err,
        "n_entries": n_entries,
        "residual_mode": "R = (E_reco - E_true) / E_true",
        "E_true_particles": E_true_arr,
        "E_reco_particles": E_reco_arr,
        "pdg_filter": pdg_filter,
    }


def collect_particle_ratio(
    parquet_dir: str,
    n_files: int,
    idx: Dict[str, int],
    pdg_filter: Tuple[int, ...],
    allowed_subdets: Optional[List[int]] = None,
    E_true_min: Optional[float] = None,
    E_true_max: Optional[float] = None,
) -> np.ndarray:
    out = compute_particle_energy_binned(
        parquet_dir=parquet_dir,
        n_files=n_files,
        idx=idx,
        pdg_filter=pdg_filter,
        bin_width=1.0,
        E_true_bins=np.array([0.0, 1.0]),  # dummy
        allowed_subdets=allowed_subdets,
    )
    E_true = out["E_true_particles"]
    E_reco = out["E_reco_particles"]
    m = np.ones_like(E_true, dtype=bool)
    if E_true_min is not None:
        m &= (E_true >= E_true_min)
    if E_true_max is not None:
        m &= (E_true < E_true_max)
    E_true = E_true[m]
    E_reco = E_reco[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (E_reco / E_true).astype(float)
