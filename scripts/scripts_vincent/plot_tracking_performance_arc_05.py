#!/usr/bin/env python3
"""Generate ARC-vs-CLD tracking efficiency and resolution PDF plots."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import awkward as ak
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
from matplotlib.text import Text
from scipy.optimize import curve_fit


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from data_generation.preprocessing.utils import (  # noqa: E402
    hit_feature_order,
    particle_feature_order,
    track_feature_order,
)


DEFAULT_CLD_DIR = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_ss_CLD_o2_v05/05"
DEFAULT_ARC_DIR = "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/Z_ss_CLD_o2_v05/arc"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "tracking_plots_arc_05")
DEFAULT_PLOT_FONT_SIZE = 18
DEFAULT_PLOT_LINEWIDTH = 2.2
DEFAULT_PLOT_MARKERSIZE = 8
DEFAULT_PLOT_AXES_LINEWIDTH = 1.3
DEFAULT_PLOT_TICK_WIDTH = 1.2

GEOMETRY_COLORS = {
    "ARC": "#E36414",
    "CLD": "#0F4C5C",
}
GEOMETRY_DISPLAY_NAMES = {
    "ARC": "o3_v01",
    "CLD": "o2_v05",
}
PION_PDG_FILTER = [211, -211]
PION_LABEL = r"$\pi^\pm$"
CLD_LINESTYLE = "-"
ARC_LINESTYLE = (0, (3.0, 2.0))
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
CURVE_COLORS = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)

DEFAULT_INDICES: Dict[str, int] = {
    "GEN_PDG": 0,
    "GEN_STATUS": 1,
    "GEN_CHARGE": 2,
    "GEN_PT": 3,
    "GEN_ETA": 4,
    "GEN_P": 11,
    "GEN_VX": 15,
    "GEN_VY": 16,
    "GEN_VZ": 17,
    "TRK_PT": 1,
    "TRK_ETA": 2,
    "TRK_D0": 18,
}


def configure_matplotlib() -> None:
    requested = str(os.environ.get("MLPF_PLOT_USETEX", "auto")).strip().lower()
    if requested in {"1", "true", "yes", "on"}:
        use_tex = latex_smoke_test()
    elif requested in {"0", "false", "no", "off"}:
        use_tex = False
    else:
        use_tex = latex_smoke_test()

    matplotlib.rc("font", size=DEFAULT_PLOT_FONT_SIZE)
    plt.rc("text", usetex=use_tex)
    plt.rc("font", family="serif")
    plt.rcParams["font.size"] = DEFAULT_PLOT_FONT_SIZE
    plt.rcParams["axes.labelsize"] = DEFAULT_PLOT_FONT_SIZE
    plt.rcParams["axes.titlesize"] = DEFAULT_PLOT_FONT_SIZE + 1
    plt.rcParams["xtick.labelsize"] = DEFAULT_PLOT_FONT_SIZE - 1
    plt.rcParams["ytick.labelsize"] = DEFAULT_PLOT_FONT_SIZE - 1
    plt.rcParams["legend.fontsize"] = DEFAULT_PLOT_FONT_SIZE - 1
    plt.rcParams["legend.title_fontsize"] = DEFAULT_PLOT_FONT_SIZE - 1
    plt.rcParams["lines.linewidth"] = DEFAULT_PLOT_LINEWIDTH
    plt.rcParams["lines.markersize"] = DEFAULT_PLOT_MARKERSIZE
    plt.rcParams["axes.linewidth"] = DEFAULT_PLOT_AXES_LINEWIDTH
    plt.rcParams["axes.edgecolor"] = "0.35"
    plt.rcParams["xtick.major.width"] = DEFAULT_PLOT_TICK_WIDTH
    plt.rcParams["ytick.major.width"] = DEFAULT_PLOT_TICK_WIDTH
    plt.rcParams["xtick.minor.width"] = DEFAULT_PLOT_TICK_WIDTH
    plt.rcParams["ytick.minor.width"] = DEFAULT_PLOT_TICK_WIDTH
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.framealpha"] = 0.95
    plt.rcParams["legend.edgecolor"] = "0.75"
    plt.rcParams["legend.handlelength"] = 5.0
    plt.rcParams["legend.handletextpad"] = 0.8
    plt.rcParams["grid.linewidth"] = 0.9
    if use_tex:
        plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
    print(f"[plot-style] text.usetex={use_tex} (MLPF_PLOT_USETEX={requested})")


def latex_smoke_test() -> bool:
    if not all(shutil.which(command) for command in ("latex", "dvipng", "gs")):
        return False
    try:
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "smoke.tex")
            with open(tex_path, "w", encoding="utf-8") as handle:
                handle.write(
                    r"\documentclass{article}" "\n"
                    r"\begin{document}" "\n"
                    r"$lp$" "\n"
                    r"\end{document}" "\n"
                )
            subprocess.run(
                [
                    "latex",
                    "-interaction=nonstopmode",
                    "--halt-on-error",
                    f"--output-directory={tmpdir}",
                    tex_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate the old ARC-vs-CLD tracking efficiency/resolution plots "
            "as standalone PDF files."
        )
    )
    parser.add_argument("--cld-dir", default=DEFAULT_CLD_DIR, help="Directory with CLD/05 parquet files.")
    parser.add_argument("--arc-dir", default=DEFAULT_ARC_DIR, help="Directory with ARC parquet files.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF plots are written.",
    )
    parser.add_argument("--n-files", type=int, default=100, help="Maximum number of parquet files per geometry.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events per geometry. If the value is larger than available, all events are used.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny Matplotlib/LaTeX smoke test, save one PDF, and exit.",
    )
    return parser.parse_args()


def run_smoke_test(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tracking_plot_style_smoke.pdf")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    y_cld = np.array([0.82, 0.88, 0.92, 0.95, 0.97, 0.985])
    y_arc = np.array([0.84, 0.90, 0.935, 0.958, 0.978, 0.99])
    ax.plot(x, y_arc, label="o3_v01", color=GEOMETRY_COLORS["ARC"], linestyle=ARC_LINESTYLE, marker="o")
    ax.plot(x, y_cld, label="o2_v05", color=GEOMETRY_COLORS["CLD"], linestyle=CLD_LINESTYLE, marker="s")
    ax.set_xscale("log")
    ax.set_xlabel(r"$p_T$ [GeV]")
    ax.set_ylabel(r"Tracking efficiency for $\pi^\pm$")
    ax.set_title(r"Tracking style smoke test: $20^\circ < \theta < 40^\circ$")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(title="Geometry")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"[smoke-test] wrote {output_path}")
    return output_path


def build_indices(
    particle_features: Optional[Sequence[str]] = None,
    track_features: Optional[Sequence[str]] = None,
    hit_features: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    idx = dict(DEFAULT_INDICES)

    if particle_features is not None:
        idx.update(
            {
                "GEN_PDG": particle_features.index("PDG"),
                "GEN_STATUS": particle_features.index("generatorStatus"),
                "GEN_CHARGE": particle_features.index("charge"),
                "GEN_PT": particle_features.index("pt"),
                "GEN_ETA": particle_features.index("eta"),
                "GEN_P": particle_features.index("p"),
                "GEN_VX": particle_features.index("vertex.x"),
                "GEN_VY": particle_features.index("vertex.y"),
                "GEN_VZ": particle_features.index("vertex.z"),
            }
        )

    if track_features is not None:
        idx.update(
            {
                "TRK_PT": track_features.index("pt"),
                "TRK_ETA": track_features.index("eta"),
                "TRK_D0": track_features.index("D0"),
            }
        )

    if hit_features is not None:
        _ = hit_features

    return idx


def resolve_parquet_files(parquet_dir: str, n_files: Optional[int]) -> list[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in '{parquet_dir}'.")
    if n_files is not None:
        files = files[:n_files]
    return files


def theta_deg_from_eta(eta: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.arctan(np.exp(-np.asarray(eta)))
    return np.degrees(theta)


def ensure_cols(mat: ak.Array, min_cols: int, fill: float = 0.0) -> ak.Array:
    try:
        arr = ak.to_numpy(mat)
        if arr.shape[1] >= min_cols:
            return mat
    except Exception:
        pass
    padded = ak.pad_none(mat, min_cols, axis=1, clip=True)
    return ak.fill_none(padded, fill)


def _required_gen_cols(idx: Dict[str, int], extra: Sequence[str] = ()) -> int:
    keys = [
        "GEN_PDG",
        "GEN_STATUS",
        "GEN_CHARGE",
        "GEN_PT",
        "GEN_ETA",
        "GEN_VX",
        "GEN_VY",
        "GEN_VZ",
        *extra,
    ]
    return max(idx[key] for key in keys) + 1


def _flat_numpy_1d(arr: Any) -> np.ndarray:
    if arr is None:
        return np.asarray([], dtype=int)
    if isinstance(arr, ak.Array):
        return ak.to_numpy(ak.flatten(arr, axis=None))
    return np.asarray(arr)


def reconstructable_mask(
    gen: ak.Array,
    *,
    indices: Dict[str, int],
    pt_min: float,
    eta_max: float,
    cos_theta_max: float,
    primary_r_min: float,
    primary_r_max: float,
    primary_z_max: float,
    pdg_filter: Optional[Iterable[int]],
    status_filter: Iterable[int],
) -> ak.Array:
    charge = gen[:, indices["GEN_CHARGE"]]
    pt = gen[:, indices["GEN_PT"]]
    eta = gen[:, indices["GEN_ETA"]]
    status = gen[:, indices["GEN_STATUS"]]
    vx = gen[:, indices["GEN_VX"]]
    vy = gen[:, indices["GEN_VY"]]
    vz = gen[:, indices["GEN_VZ"]]

    r = np.sqrt(vx**2 + vy**2)
    cos_theta = np.cos(2.0 * np.arctan(np.exp(-eta)))

    mask = (
        (np.abs(charge) > 0)
        & (pt > pt_min)
        & (np.abs(eta) < eta_max)
        & (np.abs(cos_theta) < cos_theta_max)
        & ak.Array(np.isin(ak.to_numpy(status), list(status_filter)))
        & (r >= primary_r_min)
        & (r < primary_r_max)
        & (np.abs(vz) < primary_z_max)
    )

    if pdg_filter is not None:
        pdg = gen[:, indices["GEN_PDG"]]
        mask = mask & ak.Array(np.isin(ak.to_numpy(pdg), list(pdg_filter)))

    return mask


def _matched_gen_indices(ytrk_ev: Any, n_gen: int, reco_gen_indices: np.ndarray) -> np.ndarray:
    y_np = _flat_numpy_1d(ytrk_ev)
    valid = (y_np >= 0) & (y_np < n_gen)
    if not np.any(valid):
        return np.asarray([], dtype=int)
    matched_gen_all = np.unique(y_np[valid])
    return np.intersect1d(matched_gen_all, reco_gen_indices, assume_unique=False)


def _fill_bin_counts(values: np.ndarray, bins: np.ndarray, counts: np.ndarray) -> None:
    indices = np.digitize(values, bins) - 1
    valid = (indices >= 0) & (indices < len(counts))
    if np.any(valid):
        np.add.at(counts, indices[valid], 1)


def compute_tracking_efficiency_simple(
    parquet_dir: str,
    n_files: int,
    pt_min: float,
    eta_max: float,
    primary_r_max: float,
    primary_z_max: float,
    pdg_filter: Optional[Iterable[int]],
    pt_bins: Sequence[float],
    eta_bins: Sequence[float],
    indices: Dict[str, int],
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    files = resolve_parquet_files(parquet_dir, n_files)
    min_cols = _required_gen_cols(indices)

    pt_bins = np.asarray(pt_bins, dtype=float)
    eta_bins = np.asarray(eta_bins, dtype=float)

    result: Dict[str, Any] = {
        "pt_bins": pt_bins,
        "eta_bins": eta_bins,
        "pt_num": np.zeros(len(pt_bins) - 1, dtype=int),
        "pt_den": np.zeros(len(pt_bins) - 1, dtype=int),
        "eta_num": np.zeros(len(eta_bins) - 1, dtype=int),
        "eta_den": np.zeros(len(eta_bins) - 1, dtype=int),
        "n_files": len(files),
        "n_events": 0,
    }

    num_den = 0
    num_num = 0

    for file_name in files:
        data = ak.from_parquet(file_name)
        for gen_ev, ytrk_ev in zip(data["X_gen"], data["ygen_track"]):
            if max_events is not None and result["n_events"] >= max_events:
                break
            result["n_events"] += 1
            gen = ensure_cols(gen_ev, min_cols)
            n_gen = len(gen)
            if n_gen == 0:
                continue

            pt = ak.to_numpy(gen[:, indices["GEN_PT"]])
            eta = ak.to_numpy(gen[:, indices["GEN_ETA"]])
            reco_mask = ak.to_numpy(
                reconstructable_mask(
                    gen,
                    indices=indices,
                    pt_min=pt_min,
                    eta_max=eta_max,
                    cos_theta_max=1.0,
                    primary_r_min=0.0,
                    primary_r_max=primary_r_max,
                    primary_z_max=primary_z_max,
                    pdg_filter=pdg_filter,
                    status_filter=(1,),
                )
            )
            if not np.any(reco_mask):
                continue

            reco_gen_indices = np.flatnonzero(reco_mask)
            matched_gen = _matched_gen_indices(ytrk_ev, n_gen, reco_gen_indices)
            matched_mask = np.isin(reco_gen_indices, matched_gen)

            pt_reco = pt[reco_mask]
            eta_reco = eta[reco_mask]
            _fill_bin_counts(pt_reco, pt_bins, result["pt_den"])
            _fill_bin_counts(pt_reco[matched_mask], pt_bins, result["pt_num"])
            _fill_bin_counts(eta_reco, eta_bins, result["eta_den"])
            _fill_bin_counts(eta_reco[matched_mask], eta_bins, result["eta_num"])

            num_den += reco_gen_indices.size
            num_num += matched_gen.size
        if max_events is not None and result["n_events"] >= max_events:
            break

    with np.errstate(divide="ignore", invalid="ignore"):
        result["pt_eff"] = result["pt_num"] / result["pt_den"]
        result["eta_eff"] = result["eta_num"] / result["eta_den"]

    result["eff_global"] = num_num / num_den if num_den > 0 else np.nan
    result["num_den"] = num_den
    result["num_num"] = num_num
    result["n_particles"] = num_den
    result["n_particles_label"] = "reconstructable gen particles"
    return result


def compute_tracking_efficiency(
    parquet_dir: str,
    n_files: int,
    pt_min: float,
    eta_max: float,
    cos_theta_max: float,
    primary_r_min: float,
    primary_r_max: float,
    primary_z_max: float,
    pdg_filter: Optional[Iterable[int]],
    pt_bins: Sequence[float],
    theta_bins: Sequence[float],
    theta_regions: np.ndarray,
    status_filter: Iterable[int],
    indices: Dict[str, int],
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    files = resolve_parquet_files(parquet_dir, n_files)
    min_cols = _required_gen_cols(indices)

    pt_bins = np.asarray(pt_bins, dtype=float)
    theta_bins = np.asarray(theta_bins, dtype=float)
    theta_regions = np.asarray(theta_regions, dtype=float)

    n_pt = len(pt_bins) - 1
    n_th = len(theta_bins) - 1
    n_regions = len(theta_regions)

    result: Dict[str, Any] = {
        "pt_bins": pt_bins,
        "theta_bins": theta_bins,
        "theta_regions": theta_regions,
        "pt_num": np.zeros(n_pt, dtype=int),
        "pt_den": np.zeros(n_pt, dtype=int),
        "theta_num": np.zeros(n_th, dtype=int),
        "theta_den": np.zeros(n_th, dtype=int),
        "num_p_theta": np.zeros((n_pt, n_th), dtype=int),
        "den_p_theta": np.zeros((n_pt, n_th), dtype=int),
        "pt_num_theta": np.zeros((n_regions, n_pt), dtype=int),
        "pt_den_theta": np.zeros((n_regions, n_pt), dtype=int),
        "n_files": len(files),
        "n_events": 0,
    }

    num_den = 0
    num_num = 0

    for file_name in files:
        data = ak.from_parquet(file_name)
        for gen_ev, ytrk_ev in zip(data["X_gen"], data["ygen_track"]):
            if max_events is not None and result["n_events"] >= max_events:
                break
            result["n_events"] += 1
            gen = ensure_cols(gen_ev, min_cols)
            n_gen = len(gen)
            if n_gen == 0:
                continue

            pt = ak.to_numpy(gen[:, indices["GEN_PT"]])
            eta = ak.to_numpy(gen[:, indices["GEN_ETA"]])
            theta = theta_deg_from_eta(eta)

            reco_mask = ak.to_numpy(
                reconstructable_mask(
                    gen,
                    indices=indices,
                    pt_min=pt_min,
                    eta_max=eta_max,
                    cos_theta_max=cos_theta_max,
                    primary_r_min=primary_r_min,
                    primary_r_max=primary_r_max,
                    primary_z_max=primary_z_max,
                    pdg_filter=pdg_filter,
                    status_filter=status_filter,
                )
            )
            if not np.any(reco_mask):
                continue

            reco_gen_indices = np.flatnonzero(reco_mask)
            matched_gen = _matched_gen_indices(ytrk_ev, n_gen, reco_gen_indices)
            matched_mask = np.isin(reco_gen_indices, matched_gen)

            pt_reco = pt[reco_mask]
            theta_reco = theta[reco_mask]

            _fill_bin_counts(pt_reco, pt_bins, result["pt_den"])
            _fill_bin_counts(pt_reco[matched_mask], pt_bins, result["pt_num"])
            _fill_bin_counts(theta_reco, theta_bins, result["theta_den"])
            _fill_bin_counts(theta_reco[matched_mask], theta_bins, result["theta_num"])

            ip = np.digitize(pt_reco, pt_bins) - 1
            it = np.digitize(theta_reco, theta_bins) - 1
            valid = (ip >= 0) & (ip < n_pt) & (it >= 0) & (it < n_th)
            if np.any(valid):
                flat = ip[valid] * n_th + it[valid]
                np.add.at(result["den_p_theta"].ravel(), flat, 1)

            ip_match = ip[matched_mask]
            it_match = it[matched_mask]
            valid_match = (ip_match >= 0) & (ip_match < n_pt) & (it_match >= 0) & (it_match < n_th)
            if np.any(valid_match):
                flat_match = ip_match[valid_match] * n_th + it_match[valid_match]
                np.add.at(result["num_p_theta"].ravel(), flat_match, 1)

            for region_index, (theta_min, theta_max) in enumerate(theta_regions):
                in_region = (theta_reco >= theta_min) & (theta_reco < theta_max)
                if not np.any(in_region):
                    continue
                pt_region = pt_reco[in_region]
                _fill_bin_counts(pt_region, pt_bins, result["pt_den_theta"][region_index])
                region_gen = reco_gen_indices[in_region]
                region_matched = np.isin(region_gen, matched_gen)
                if np.any(region_matched):
                    _fill_bin_counts(pt_region[region_matched], pt_bins, result["pt_num_theta"][region_index])

            num_den += reco_gen_indices.size
            num_num += matched_gen.size
        if max_events is not None and result["n_events"] >= max_events:
            break

    with np.errstate(divide="ignore", invalid="ignore"):
        result["pt_eff"] = result["pt_num"] / result["pt_den"]
        result["theta_eff"] = result["theta_num"] / result["theta_den"]
        result["eff_p_theta"] = result["num_p_theta"] / result["den_p_theta"]
        result["pt_eff_theta"] = result["pt_num_theta"] / result["pt_den_theta"]

    result["pt_err"] = np.full_like(result["pt_eff"], np.nan, dtype=float)
    valid = result["pt_den"] > 0
    result["pt_err"][valid] = np.sqrt(result["pt_eff"][valid] * (1.0 - result["pt_eff"][valid]) / result["pt_den"][valid])

    result["theta_err"] = np.full_like(result["theta_eff"], np.nan, dtype=float)
    valid = result["theta_den"] > 0
    result["theta_err"][valid] = np.sqrt(
        result["theta_eff"][valid] * (1.0 - result["theta_eff"][valid]) / result["theta_den"][valid]
    )

    result["err_p_theta"] = np.full_like(result["eff_p_theta"], np.nan, dtype=float)
    valid = result["den_p_theta"] > 0
    result["err_p_theta"][valid] = np.sqrt(
        result["eff_p_theta"][valid] * (1.0 - result["eff_p_theta"][valid]) / result["den_p_theta"][valid]
    )

    result["pt_err_theta"] = np.full_like(result["pt_eff_theta"], np.nan, dtype=float)
    valid = result["pt_den_theta"] > 0
    result["pt_err_theta"][valid] = np.sqrt(
        result["pt_eff_theta"][valid] * (1.0 - result["pt_eff_theta"][valid]) / result["pt_den_theta"][valid]
    )

    result["eff_global"] = num_num / num_den if num_den > 0 else np.nan
    result["num_den"] = num_den
    result["num_num"] = num_num
    result["n_particles"] = num_den
    result["n_particles_label"] = "reconstructable gen particles"
    return result


def _gauss(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _fit_gaussian_3sigma_clipped_fast(residuals: np.ndarray, min_entries: int) -> Tuple[float, float]:
    residuals = np.asarray(residuals, dtype=float)
    if residuals.size < min_entries:
        return np.nan, np.nan

    q16, q84 = np.percentile(residuals, [16, 84])
    sigma0 = 0.5 * (q84 - q16)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        return np.nan, np.nan

    clipped = residuals[np.abs(residuals) < 3.0 * sigma0]
    if clipped.size < min_entries:
        return np.nan, np.nan

    n_hist_bins = max(10, min(40, clipped.size // 5))
    counts, edges = np.histogram(clipped, bins=n_hist_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = counts > 0
    if np.count_nonzero(valid) < 5:
        return np.nan, np.nan

    amplitude0 = counts[valid].max()
    sigma_guess = np.std(clipped)
    if not np.isfinite(sigma_guess) or sigma_guess <= 0:
        return np.nan, np.nan

    try:
        fit, _ = curve_fit(
            _gauss,
            centers[valid],
            counts[valid],
            p0=[amplitude0, 0.0, sigma_guess],
            maxfev=5000,
        )
    except Exception:
        return np.nan, np.nan

    _, mean, sigma = fit
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan, np.nan
    return float(sigma), float(mean)


def compute_track_resolution_p_theta_fit(
    parquet_dir: str,
    n_files: int,
    pt_min: float,
    eta_max: float,
    cos_theta_max: float,
    primary_r_max: float,
    primary_z_max: float,
    pdg_filter: Optional[Iterable[int]],
    p_bins: Sequence[float],
    theta_bins: Sequence[float],
    min_entries: int,
    observable: str,
    indices: Dict[str, int],
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    if observable not in {"d0", "pt"}:
        raise ValueError(f"Unsupported observable '{observable}'.")

    files = resolve_parquet_files(parquet_dir, n_files)
    min_cols = _required_gen_cols(indices, extra=("GEN_P",))

    p_bins = np.asarray(p_bins, dtype=float)
    theta_bins = np.asarray(theta_bins, dtype=float)
    n_p = len(p_bins) - 1
    n_th = len(theta_bins) - 1

    residual_bins = [[] for _ in range(n_p * n_th)]
    n_events = 0

    for file_name in files:
        data = ak.from_parquet(file_name)
        for gen_ev, trk_ev, ytrk_ev in zip(data["X_gen"], data["X_track"], data["ygen_track"]):
            if max_events is not None and n_events >= max_events:
                break
            n_events += 1
            gen = ensure_cols(gen_ev, min_cols)
            n_gen = len(gen)
            n_trk = len(trk_ev)
            if n_gen == 0 or n_trk == 0:
                continue

            pt_gen = ak.to_numpy(gen[:, indices["GEN_PT"]])
            eta_gen = ak.to_numpy(gen[:, indices["GEN_ETA"]])
            theta_true_all = theta_deg_from_eta(eta_gen)

            reco_mask = ak.to_numpy(
                reconstructable_mask(
                    gen,
                    indices=indices,
                    pt_min=pt_min,
                    eta_max=eta_max,
                    cos_theta_max=cos_theta_max,
                    primary_r_min=0.0,
                    primary_r_max=primary_r_max,
                    primary_z_max=primary_z_max,
                    pdg_filter=pdg_filter,
                    status_filter=(1,),
                )
            )

            y_np = _flat_numpy_1d(ytrk_ev)
            valid = (y_np >= 0) & (y_np < n_gen)
            if not np.any(valid):
                continue

            gen_indices = y_np[valid]
            trk_selected = trk_ev[valid, :]
            valid = reco_mask[gen_indices]
            if not np.any(valid):
                continue

            gen_indices = gen_indices[valid]
            trk_selected = trk_selected[valid, :]

            p_true = ak.to_numpy(gen[gen_indices, indices["GEN_P"]])
            theta_true = theta_true_all[gen_indices]

            if observable == "d0":
                residual = ak.to_numpy(trk_selected[:, indices["TRK_D0"]])
            else:
                pt_true = pt_gen[gen_indices]
                pt_reco = ak.to_numpy(trk_selected[:, indices["TRK_PT"]])
                pt_valid = pt_true > 0
                if not np.any(pt_valid):
                    continue
                p_true = p_true[pt_valid]
                theta_true = theta_true[pt_valid]
                pt_true = pt_true[pt_valid]
                pt_reco = pt_reco[pt_valid]
                residual = (pt_true - pt_reco) / (pt_true**2)

            ip = np.digitize(p_true, p_bins) - 1
            it = np.digitize(theta_true, theta_bins) - 1
            valid = (ip >= 0) & (ip < n_p) & (it >= 0) & (it < n_th)
            if not np.any(valid):
                continue

            ip = ip[valid]
            it = it[valid]
            residual = residual[valid]
            flat = ip * n_th + it

            for bin_index in np.unique(flat):
                mask = flat == bin_index
                residual_bins[int(bin_index)].extend(np.asarray(residual[mask], dtype=float))
        if max_events is not None and n_events >= max_events:
            break

    sigma = np.full((n_p, n_th), np.nan, dtype=float)
    mean = np.full((n_p, n_th), np.nan, dtype=float)
    n_entries = np.zeros((n_p, n_th), dtype=int)

    for ip in range(n_p):
        for it in range(n_th):
            flat_index = ip * n_th + it
            residual = np.asarray(residual_bins[flat_index], dtype=float)
            n_entries[ip, it] = residual.size
            if residual.size < min_entries:
                continue
            sigma_fit, mean_fit = _fit_gaussian_3sigma_clipped_fast(residual, min_entries=min_entries)
            if np.isnan(sigma_fit):
                sigma_fit = np.std(residual)
                mean_fit = np.mean(residual)
            sigma[ip, it] = sigma_fit
            mean[ip, it] = mean_fit

    result: Dict[str, Any] = {
        "p_bins": p_bins,
        "theta_bins": theta_bins,
        "sigma": sigma,
        "mu": mean,
        "n_entries": n_entries,
        "observable": observable,
        "n_files": len(files),
        "n_events": n_events,
        "n_particles": int(sum(len(residual) for residual in residual_bins)),
        "n_particles_label": "matched reco tracks (used in fit)",
    }
    if observable == "d0":
        result["sigma_d0"] = sigma
    else:
        result["sigma_R"] = sigma
    return result


def _ratio_and_error(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_err: Optional[np.ndarray] = None,
    denominator_err: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    ratio_err = None
    if numerator_err is not None and denominator_err is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_err = ratio * np.sqrt((numerator_err / numerator) ** 2 + (denominator_err / denominator) ** 2)
    return ratio, ratio_err


def _auto_ratio_ylim(
    ratios: Sequence[np.ndarray],
    errors: Sequence[Optional[np.ndarray]],
    *,
    reference: float = 1.0,
    min_span: float = 0.08,
    padding_fraction: float = 0.12,
) -> Tuple[float, float]:
    lows = [reference]
    highs = [reference]

    for ratio, error in zip(ratios, errors):
        ratio = np.asarray(ratio, dtype=float)
        finite_ratio = np.isfinite(ratio)
        if not np.any(finite_ratio):
            continue

        low = ratio[finite_ratio].copy()
        high = ratio[finite_ratio].copy()
        if error is not None:
            error = np.asarray(error, dtype=float)
            finite_error = np.isfinite(error[finite_ratio])
            low[finite_error] -= error[finite_ratio][finite_error]
            high[finite_error] += error[finite_ratio][finite_error]

        lows.extend(low[np.isfinite(low)])
        highs.extend(high[np.isfinite(high)])

    ymin = float(np.nanmin(lows))
    ymax = float(np.nanmax(highs))
    span = max(ymax - ymin, min_span)
    center = 0.5 * (ymin + ymax)
    padding = padding_fraction * span
    return center - 0.5 * span - padding, center + 0.5 * span + padding


def make_ratio_axes(figsize: Tuple[float, float] = (8.0, 7.0)) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax_top, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    return fig, ax_top, ax_ratio


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_color("0.35")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", direction="out", width=1.0)
    ax.tick_params(axis="both", which="minor", direction="out", width=0.8)
    ax.set_axisbelow(True)


def geometry_series_style(geometry: str, color: Any, marker: str) -> Dict[str, Any]:
    return {
        "color": color,
        "marker": marker,
        "linestyle": ARC_LINESTYLE if geometry == "ARC" else CLD_LINESTYLE,
        "linewidth": 2.4 if geometry == "ARC" else 2.1,
        "markersize": 5.5,
        "markerfacecolor": color if geometry == "ARC" else "white",
        "markeredgewidth": 1.1,
        "capsize": 3,
    }


def ratio_series_style(color: Any, marker: str) -> Dict[str, Any]:
    return {
        "color": color,
        "marker": marker,
        "linestyle": "-",
        "linewidth": 1.8,
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "capsize": 3,
    }


def _legend_text_cell(
    text: str,
    *,
    width: float,
    height: float = 16.0,
    fontsize: float = 10.0,
    align: str = "left",
) -> DrawingArea:
    area = DrawingArea(width, height, 0, 0)
    x = 0.0 if align == "left" else width / 2.0
    area.add_artist(
        Text(
            x=x,
            y=height / 2.0,
            text=text,
            ha=align,
            va="center",
            fontsize=fontsize,
        )
    )
    return area


def _legend_line_cell(
    geometry: str,
    color: Any,
    marker: str,
    *,
    width: float,
    height: float = 16.0,
) -> DrawingArea:
    area = DrawingArea(width, height, 0, 0)
    pad = 8.0
    y = height / 2.0
    area.add_artist(
        Line2D(
            [pad, width - pad],
            [y, y],
            color=color,
            marker=marker,
            linestyle=ARC_LINESTYLE if geometry == "ARC" else CLD_LINESTYLE,
            linewidth=2.8 if geometry == "ARC" else 2.4,
            markersize=6.0,
            markerfacecolor=color if geometry == "ARC" else "white",
            markeredgewidth=1.2,
        )
    )
    return area


def add_table_legend(
    ax: plt.Axes,
    bin_labels: Sequence[str],
    bin_styles: Sequence[Tuple[Any, str]],
    *,
    bin_header: str,
    loc: str = "lower left",
    bbox_to_anchor: Tuple[float, float] = (0.012, 0.02),
) -> None:
    text_width = 115.0
    line_width = 96.0
    row_gap = 2.0
    col_gap = 8.0

    rows = [
        HPacker(
            children=[
                _legend_text_cell(bin_header, width=text_width, align="center"),
                _legend_text_cell(GEOMETRY_DISPLAY_NAMES["CLD"], width=line_width, align="center"),
                _legend_text_cell(GEOMETRY_DISPLAY_NAMES["ARC"], width=line_width, align="center"),
            ],
            align="center",
            pad=0.0,
            sep=col_gap,
        )
    ]

    for label, (color, marker) in zip(bin_labels, bin_styles):
        rows.append(
            HPacker(
                children=[
                    _legend_text_cell(label, width=text_width, align="center"),
                    _legend_line_cell("CLD", color, marker, width=line_width),
                    _legend_line_cell("ARC", color, marker, width=line_width),
                ],
                align="center",
                pad=0.0,
                sep=col_gap,
            )
        )

    legend_box = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=rows, align="left", pad=0.0, sep=row_gap),
        frameon=True,
        pad=0.25,
        borderpad=0.7,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
    )
    legend_box.patch.set_alpha(0.95)
    legend_box.patch.set_edgecolor("0.75")
    ax.add_artist(legend_box)


def _save_figure(fig: plt.Figure, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_efficiency_1d_ratio(
    bin_edges: np.ndarray,
    eff_cld: np.ndarray,
    eff_arc: np.ndarray,
    *,
    x_label: str,
    title: str,
    output_path: str,
    xscale: Optional[str] = None,
    ratio_ylim: Optional[Tuple[float, float]] = None,
) -> str:
    fig, ax_top, ax_ratio = make_ratio_axes()

    ax_top.step(
        bin_edges[:-1],
        eff_cld,
        where="post",
        label=GEOMETRY_DISPLAY_NAMES["CLD"],
        color=GEOMETRY_COLORS["CLD"],
        linewidth=2.2,
        linestyle=CLD_LINESTYLE,
    )
    ax_top.step(
        bin_edges[:-1],
        eff_arc,
        where="post",
        label=GEOMETRY_DISPLAY_NAMES["ARC"],
        color=GEOMETRY_COLORS["ARC"],
        linewidth=2.2,
        linestyle=ARC_LINESTYLE,
    )
    ax_top.set_ylabel("Tracking efficiency")
    ax_top.set_ylim(0.0, 1.02)
    ax_top.set_title(title)
    ax_top.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_top.legend(fontsize=12, title="Geometry", title_fontsize=11, loc="best", handlelength=6.5)

    ratio, _ = _ratio_and_error(eff_arc, eff_cld)
    ax_ratio.step(bin_edges[:-1], ratio, where="post", color="0.2", linewidth=2.0)
    ax_ratio.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax_ratio.set_ylabel(f"{GEOMETRY_DISPLAY_NAMES['ARC']} / {GEOMETRY_DISPLAY_NAMES['CLD']}")
    ax_ratio.set_xlabel(x_label)
    ax_ratio.set_ylim(*(ratio_ylim or _auto_ratio_ylim([ratio], [None])))
    ax_ratio.grid(True, axis="y", alpha=0.25, linestyle="--")

    if xscale:
        ax_top.set_xscale(xscale)
        ax_ratio.set_xscale(xscale)

    style_axes(ax_top)
    style_axes(ax_ratio)
    return _save_figure(fig, output_path)


def plot_eff_vs_pt_multi_theta_cld_arc(
    res_cld: Dict[str, Any],
    res_arc: Dict[str, Any],
    *,
    theta_region_indices: Sequence[int],
    particle_label: str,
    output_path: str,
    ymin_ratio: Optional[float],
    ymax_ratio: Optional[float],
) -> str:
    pt_edges = res_cld["pt_bins"]
    pt_centers = 0.5 * (pt_edges[:-1] + pt_edges[1:])
    theta_regions = res_cld["theta_regions"]

    fig, ax_top, ax_ratio = make_ratio_axes(figsize=(9.0, 7.6))
    bin_labels = []
    bin_styles = []
    ratio_values = []
    ratio_errors = []

    for index, region_index in enumerate(theta_region_indices):
        marker = MARKERS[index % len(MARKERS)]
        color = CURVE_COLORS[index % len(CURVE_COLORS)]
        eff_cld = res_cld["pt_eff_theta"][region_index, :]
        err_cld = res_cld["pt_err_theta"][region_index, :]
        eff_arc = res_arc["pt_eff_theta"][region_index, :]
        err_arc = res_arc["pt_err_theta"][region_index, :]
        den_cld = res_cld["pt_den_theta"][region_index, :]
        den_arc = res_arc["pt_den_theta"][region_index, :]

        valid = (den_cld >= 30) & (den_arc >= 30)
        eff_cld = np.where(valid, eff_cld, np.nan)
        err_cld = np.where(valid, err_cld, np.nan)
        eff_arc = np.where(valid, eff_arc, np.nan)
        err_arc = np.where(valid, err_arc, np.nan)

        theta_min, theta_max = theta_regions[region_index]
        bin_label = rf"$[{theta_min:.0f}^\circ,{theta_max:.0f}^\circ)$"

        ax_top.errorbar(pt_centers, eff_cld, yerr=err_cld, **geometry_series_style("CLD", color, marker))
        ax_top.errorbar(pt_centers, eff_arc, yerr=err_arc, **geometry_series_style("ARC", color, marker))
        bin_labels.append(bin_label)
        bin_styles.append((color, marker))

        ratio, ratio_err = _ratio_and_error(eff_arc, eff_cld, err_arc, err_cld)
        ratio_valid = valid & np.isfinite(ratio)
        ratio = np.where(ratio_valid, ratio, np.nan)
        if ratio_err is not None:
            ratio_err = np.where(ratio_valid, ratio_err, np.nan)
        ratio_values.append(ratio)
        ratio_errors.append(ratio_err)
        ax_ratio.errorbar(pt_centers, ratio, yerr=ratio_err, **ratio_series_style(color, marker))

    ax_top.set_ylabel("Tracking efficiency")
    ax_top.set_title(rf"Tracking efficiency vs $p_T$ for {particle_label}")
    ax_top.grid(True, axis="y", alpha=0.25, linestyle="--")
    add_table_legend(ax_top, bin_labels, bin_styles, bin_header=r"$\theta$")

    ax_ratio.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax_ratio.set_xlabel(r"$p_T$ [GeV]")
    ax_ratio.set_ylabel(f"{GEOMETRY_DISPLAY_NAMES['ARC']} / {GEOMETRY_DISPLAY_NAMES['CLD']}")
    ratio_ylim = (
        (ymin_ratio, ymax_ratio)
        if ymin_ratio is not None and ymax_ratio is not None
        else _auto_ratio_ylim(ratio_values, ratio_errors)
    )
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(True, axis="y", alpha=0.25, linestyle="--")

    style_axes(ax_top)
    style_axes(ax_ratio)
    return _save_figure(fig, output_path)


def plot_eff_vs_theta_multi_p_cld_arc(
    res_cld: Dict[str, Any],
    res_arc: Dict[str, Any],
    *,
    p_bin_indices: Sequence[int],
    particle_label: str,
    output_path: str,
    ymin_ratio: Optional[float],
    ymax_ratio: Optional[float],
) -> str:
    theta_edges = res_cld["theta_bins"]
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    p_edges = res_cld["pt_bins"]

    fig, ax_top, ax_ratio = make_ratio_axes(figsize=(9.0, 7.6))
    bin_labels = []
    bin_styles = []
    ratio_values = []
    ratio_errors = []

    for index, p_index in enumerate(p_bin_indices):
        marker = MARKERS[index % len(MARKERS)]
        color = CURVE_COLORS[index % len(CURVE_COLORS)]
        eff_cld = res_cld["eff_p_theta"][p_index, :]
        err_cld = res_cld["err_p_theta"][p_index, :]
        eff_arc = res_arc["eff_p_theta"][p_index, :]
        err_arc = res_arc["err_p_theta"][p_index, :]
        den_cld = res_cld["den_p_theta"][p_index, :]
        den_arc = res_arc["den_p_theta"][p_index, :]

        valid = (den_cld >= 30) & (den_arc >= 30)
        eff_cld = np.where(valid, eff_cld, np.nan)
        err_cld = np.where(valid, err_cld, np.nan)
        eff_arc = np.where(valid, eff_arc, np.nan)
        err_arc = np.where(valid, err_arc, np.nan)

        p_min = p_edges[p_index]
        p_max = p_edges[p_index + 1]
        bin_label = rf"$[{p_min:.2g},{p_max:.2g}]$ GeV"

        ax_top.errorbar(theta_centers, eff_cld, yerr=err_cld, **geometry_series_style("CLD", color, marker))
        ax_top.errorbar(theta_centers, eff_arc, yerr=err_arc, **geometry_series_style("ARC", color, marker))
        bin_labels.append(bin_label)
        bin_styles.append((color, marker))

        ratio, ratio_err = _ratio_and_error(eff_arc, eff_cld, err_arc, err_cld)
        ratio_valid = valid & np.isfinite(ratio)
        ratio = np.where(ratio_valid, ratio, np.nan)
        if ratio_err is not None:
            ratio_err = np.where(ratio_valid, ratio_err, np.nan)
        ratio_values.append(ratio)
        ratio_errors.append(ratio_err)
        ax_ratio.errorbar(theta_centers, ratio, yerr=ratio_err, **ratio_series_style(color, marker))

    ax_top.set_ylabel("Tracking efficiency")
    ax_top.set_title(rf"Tracking efficiency for {particle_label}")
    ax_top.grid(True, axis="y", alpha=0.25, linestyle="--")
    add_table_legend(ax_top, bin_labels, bin_styles, bin_header=r"$p_T$")

    ax_ratio.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax_ratio.set_xlabel(r"$\theta$ [deg]")
    ax_ratio.set_ylabel(f"{GEOMETRY_DISPLAY_NAMES['ARC']} / {GEOMETRY_DISPLAY_NAMES['CLD']}")
    ratio_ylim = (
        (ymin_ratio, ymax_ratio)
        if ymin_ratio is not None and ymax_ratio is not None
        else _auto_ratio_ylim(ratio_values, ratio_errors)
    )
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(True, axis="y", alpha=0.25, linestyle="--")

    style_axes(ax_top)
    style_axes(ax_ratio)
    return _save_figure(fig, output_path)


def plot_track_res_vs_theta_multi_p_cld_arc(
    res_cld: Dict[str, Any],
    res_arc: Dict[str, Any],
    *,
    p_bin_indices: Sequence[int],
    particle_label: str,
    output_path: str,
    ymin_ratio: Optional[float],
    ymax_ratio: Optional[float],
) -> str:
    theta_edges = res_cld["theta_bins"]
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    p_edges = res_cld["p_bins"]

    sigma_cld = res_cld["sigma_R"] if res_cld["observable"] == "pt" else res_cld["sigma_d0"]
    sigma_arc = res_arc["sigma_R"] if res_arc["observable"] == "pt" else res_arc["sigma_d0"]
    y_label = (
        r"$\sigma\!\left(\frac{\Delta p_T}{p_{T,\mathrm{true}}^2}\right)$ [GeV$^{-1}$]"
        if res_cld["observable"] == "pt"
        else r"$\sigma(\Delta d_0)$"
    )

    fig, ax_top, ax_ratio = make_ratio_axes(figsize=(9.0, 7.6))
    bin_labels = []
    bin_styles = []
    ratio_values = []
    ratio_errors = []

    for index, p_index in enumerate(p_bin_indices):
        marker = MARKERS[index % len(MARKERS)]
        color = CURVE_COLORS[index % len(CURVE_COLORS)]
        s_cld = sigma_cld[p_index, :]
        s_arc = sigma_arc[p_index, :]
        c_cld = res_cld["n_entries"][p_index, :]
        c_arc = res_arc["n_entries"][p_index, :]
        valid = (c_cld >= 50) & (c_arc >= 50) & np.isfinite(s_cld) & np.isfinite(s_arc)

        s_cld = np.where(valid, s_cld, np.nan)
        s_arc = np.where(valid, s_arc, np.nan)
        err_cld = np.where(valid, s_cld / np.sqrt(2 * np.maximum(c_cld - 1, 1)), np.nan)
        err_arc = np.where(valid, s_arc / np.sqrt(2 * np.maximum(c_arc - 1, 1)), np.nan)

        p_min = p_edges[p_index]
        p_max = p_edges[p_index + 1]
        bin_label = rf"$[{p_min:.1f},{p_max:.1f}]$ GeV"

        ax_top.errorbar(theta_centers, s_cld, yerr=err_cld, **geometry_series_style("CLD", color, marker))
        ax_top.errorbar(theta_centers, s_arc, yerr=err_arc, **geometry_series_style("ARC", color, marker))
        bin_labels.append(bin_label)
        bin_styles.append((color, marker))

        ratio, ratio_err = _ratio_and_error(s_arc, s_cld, err_arc, err_cld)
        ratio_values.append(ratio)
        ratio_errors.append(ratio_err)
        ax_ratio.errorbar(theta_centers, ratio, yerr=ratio_err, **ratio_series_style(color, marker))

    ax_top.set_ylabel(y_label)
    ax_top.set_title(rf"Tracking resolution for {particle_label}")
    ax_top.set_yscale("log")
    ax_top.grid(True, axis="y", alpha=0.25, linestyle="--")
    add_table_legend(
        ax_top,
        bin_labels,
        bin_styles,
        bin_header=r"$p$",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )

    ax_ratio.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax_ratio.set_xlabel(r"$\theta$ [deg]")
    ax_ratio.set_ylabel(f"{GEOMETRY_DISPLAY_NAMES['ARC']} / {GEOMETRY_DISPLAY_NAMES['CLD']}")
    ratio_ylim = (
        (ymin_ratio, ymax_ratio)
        if ymin_ratio is not None and ymax_ratio is not None
        else _auto_ratio_ylim(ratio_values, ratio_errors)
    )
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(True, axis="y", alpha=0.25, linestyle="--")

    style_axes(ax_top)
    style_axes(ax_ratio)
    return _save_figure(fig, output_path)


def plot_track_res_vs_p_multi_theta_cld_arc(
    res_cld: Dict[str, Any],
    res_arc: Dict[str, Any],
    *,
    theta_bin_indices: Sequence[int],
    particle_label: str,
    output_path: str,
    ymin_ratio: Optional[float],
    ymax_ratio: Optional[float],
) -> str:
    p_edges = res_cld["p_bins"]
    p_centers = 0.5 * (p_edges[:-1] + p_edges[1:])
    theta_edges = res_cld["theta_bins"]

    sigma_cld = res_cld["sigma_R"] if res_cld["observable"] == "pt" else res_cld["sigma_d0"]
    sigma_arc = res_arc["sigma_R"] if res_arc["observable"] == "pt" else res_arc["sigma_d0"]
    y_label = (
        r"$\sigma\!\left(\frac{\Delta p_T}{p_{T,\mathrm{true}}^2}\right)$ [GeV$^{-1}$]"
        if res_cld["observable"] == "pt"
        else r"$\sigma(\Delta d_0)$"
    )

    fig, ax_top, ax_ratio = make_ratio_axes(figsize=(9.0, 7.6))
    bin_labels = []
    bin_styles = []
    ratio_values = []
    ratio_errors = []

    for index, theta_index in enumerate(theta_bin_indices):
        marker = MARKERS[index % len(MARKERS)]
        color = CURVE_COLORS[index % len(CURVE_COLORS)]
        s_cld = sigma_cld[:, theta_index]
        s_arc = sigma_arc[:, theta_index]
        c_cld = res_cld["n_entries"][:, theta_index]
        c_arc = res_arc["n_entries"][:, theta_index]
        valid = (c_cld >= 50) & (c_arc >= 50) & np.isfinite(s_cld) & np.isfinite(s_arc)

        s_cld = np.where(valid, s_cld, np.nan)
        s_arc = np.where(valid, s_arc, np.nan)
        err_cld = np.where(valid, s_cld / np.sqrt(2 * np.maximum(c_cld - 1, 1)), np.nan)
        err_arc = np.where(valid, s_arc / np.sqrt(2 * np.maximum(c_arc - 1, 1)), np.nan)

        theta_min = theta_edges[theta_index]
        theta_max = theta_edges[theta_index + 1]
        bin_label = rf"$[{theta_min:.0f}^\circ,{theta_max:.0f}^\circ]$"

        ax_top.errorbar(p_centers, s_cld, yerr=err_cld, **geometry_series_style("CLD", color, marker))
        ax_top.errorbar(p_centers, s_arc, yerr=err_arc, **geometry_series_style("ARC", color, marker))
        bin_labels.append(bin_label)
        bin_styles.append((color, marker))

        ratio, ratio_err = _ratio_and_error(s_arc, s_cld, err_arc, err_cld)
        ratio_values.append(ratio)
        ratio_errors.append(ratio_err)
        ax_ratio.errorbar(p_centers, ratio, yerr=ratio_err, **ratio_series_style(color, marker))

    ax_top.set_ylabel(y_label)
    ax_top.set_title(rf"Tracking resolution for {particle_label}")
    ax_top.set_yscale("log")
    ax_top.grid(True, axis="y", alpha=0.25, linestyle="--")
    add_table_legend(ax_top, bin_labels, bin_styles, bin_header=r"$\theta$")

    ax_ratio.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax_ratio.set_xlabel(r"$p$ [GeV]")
    ax_ratio.set_ylabel(f"{GEOMETRY_DISPLAY_NAMES['ARC']} / {GEOMETRY_DISPLAY_NAMES['CLD']}")
    ratio_ylim = (
        (ymin_ratio, ymax_ratio)
        if ymin_ratio is not None and ymax_ratio is not None
        else _auto_ratio_ylim(ratio_values, ratio_errors)
    )
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.grid(True, axis="y", alpha=0.25, linestyle="--")

    style_axes(ax_top)
    style_axes(ax_ratio)
    return _save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    if args.smoke_test:
        run_smoke_test(args.output_dir)
        return

    indices = build_indices(particle_feature_order, track_feature_order, hit_feature_order)
    os.makedirs(args.output_dir, exist_ok=True)

    saved_paths: list[str] = []

    theta_regions = np.array(
        [
            [20, 30],
            [30, 40],
            [40, 50],
            [50, 60],
            [60, 70],
            [70, 80],
            [80, 90],
            [90, 100],
            [100, 110],
            [110, 120],
            [120, 130],
            [130, 140],
            [140, 150],
            [150, 160],
        ],
        dtype=float,
    )
    theta_bins = np.concatenate([theta_regions[:1, 0], theta_regions[:, 1]])

    eff_cld = compute_tracking_efficiency(
        args.cld_dir,
        n_files=args.n_files,
        pt_min=0.1,
        eta_max=10.0,
        cos_theta_max=1.0,
        primary_r_min=0.0,
        primary_r_max=9999.0,
        primary_z_max=9999.0,
        pdg_filter=PION_PDG_FILTER,
        pt_bins=[0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0],
        theta_bins=theta_bins,
        theta_regions=theta_regions,
        status_filter=[1],
        indices=indices,
        max_events=args.max_events,
    )
    eff_arc = compute_tracking_efficiency(
        args.arc_dir,
        n_files=args.n_files,
        pt_min=0.1,
        eta_max=10.0,
        cos_theta_max=1.0,
        primary_r_min=0.0,
        primary_r_max=9999.0,
        primary_z_max=9999.0,
        pdg_filter=PION_PDG_FILTER,
        pt_bins=[0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0],
        theta_bins=theta_bins,
        theta_regions=theta_regions,
        status_filter=[1],
        indices=indices,
        max_events=args.max_events,
    )

    saved_paths.append(
        plot_eff_vs_theta_multi_p_cld_arc(
            eff_cld,
            eff_arc,
            p_bin_indices=[1, 2, 4],
            particle_label=PION_LABEL,
            output_path=os.path.join(args.output_dir, "tracking_efficiency_theta.pdf"),
            ymin_ratio=None,
            ymax_ratio=None,
        )
    )

    resolution_cld = compute_track_resolution_p_theta_fit(
        args.cld_dir,
        n_files=args.n_files,
        pt_min=1.0,
        eta_max=10.0,
        cos_theta_max=0.99,
        primary_r_max=50.0,
        primary_z_max=50.0,
        pdg_filter=PION_PDG_FILTER,
        p_bins=np.logspace(0.0, np.log10(40.0), 8),
        theta_bins=np.linspace(20.0, 160.0, 16),
        min_entries=50,
        observable="pt",
        indices=indices,
        max_events=args.max_events,
    )
    resolution_arc = compute_track_resolution_p_theta_fit(
        args.arc_dir,
        n_files=args.n_files,
        pt_min=1.0,
        eta_max=10.0,
        cos_theta_max=0.99,
        primary_r_max=50.0,
        primary_z_max=50.0,
        pdg_filter=PION_PDG_FILTER,
        p_bins=np.logspace(0.0, np.log10(40.0), 8),
        theta_bins=np.linspace(20.0, 160.0, 16),
        min_entries=50,
        observable="pt",
        indices=indices,
        max_events=args.max_events,
    )

    saved_paths.append(
        plot_track_res_vs_theta_multi_p_cld_arc(
            resolution_cld,
            resolution_arc,
            p_bin_indices=[0, 1, 2],
            particle_label=PION_LABEL,
            output_path=os.path.join(args.output_dir, "tracking_resolution_theta.pdf"),
            ymin_ratio=None,
            ymax_ratio=None,
        )
    )
    print("Saved plots:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
