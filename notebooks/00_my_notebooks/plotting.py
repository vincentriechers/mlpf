"""
plotting.py

Minimal plotting helpers for CLD vs ARC comparisons.

Design principles:
  - Keep only generic plotting primitives here.
  - No detector/physics-specific wrappers (those belong in the notebook).
  - Provide "top + ratio" layouts to avoid copy/paste everywhere.
"""

from __future__ import annotations

from typing import Optional, Tuple, Sequence, Callable, List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Helper: ratio + error propagation
# ----------------------------------------------------------------------
def _safe_ratio_and_err(
    y_num: np.ndarray,
    y_den: np.ndarray,
    y_num_err: Optional[np.ndarray],
    y_den_err: Optional[np.ndarray],
):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = y_num / y_den

    rerr = None
    if (y_num_err is not None) and (y_den_err is not None):
        with np.errstate(divide="ignore", invalid="ignore"):
            rerr = r * np.sqrt((y_num_err / y_num) ** 2 + (y_den_err / y_den) ** 2)

    return r, rerr


# ----------------------------------------------------------------------
# 1) Generic "top + ratio" plot for 1D arrays
# ----------------------------------------------------------------------
def plot_compare_with_ratio(
    x: np.ndarray,
    yA: np.ndarray,
    yB: np.ndarray,
    yAerr: Optional[np.ndarray] = None,
    yBerr: Optional[np.ndarray] = None,
    labelA: str = "CLD",
    labelB: str = "ARC",
    xlabel: str = "x",
    ylabel: str = "y",
    title: Optional[str] = None,
    ratio_label: Optional[str] = None,
    ratio_ylim: Tuple[float, float] = (0.9, 1.1),
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    savepath: Optional[str] = None,
    dpi: int = 200,
):
    """
    Two-panel compare plot:
      - top: yA and yB vs x
      - bottom: ratio yB/yA (+ propagated uncertainty if errors are given)

    Returns: (fig, (ax_top, ax_ratio))
    """
    x = np.asarray(x)
    yA = np.asarray(yA)
    yB = np.asarray(yB)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.errorbar(x, yA, yerr=yAerr, fmt="o-", ms=4, label=labelA)
    ax1.errorbar(x, yB, yerr=yBerr, fmt="s--", ms=4, label=labelB)

    ax1.set_ylabel(ylabel)
    if title:
        ax1.set_title(title)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend()

    if xscale:
        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)
    if yscale:
        ax1.set_yscale(yscale)

    ratio, ratio_err = _safe_ratio_and_err(yB, yA, yBerr, yAerr)

    ax2.errorbar(x, ratio, yerr=ratio_err, fmt="o-", ms=4)
    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_ylim(*ratio_ylim)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ratio_label if ratio_label else f"{labelB}/{labelA}")
    ax2.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, (ax1, ax2)


# ----------------------------------------------------------------------
# 2) Multi-curve "top + ratio" plot
# ----------------------------------------------------------------------
def plot_multicurve_with_ratio(
    x: np.ndarray,
    curvesA: Sequence[np.ndarray],
    curvesB: Sequence[np.ndarray],
    labels: Sequence[str],
    curvesAerr: Optional[Sequence[np.ndarray]] = None,
    curvesBerr: Optional[Sequence[np.ndarray]] = None,
    labelA: str = "CLD",
    labelB: str = "ARC",
    xlabel: str = "x",
    ylabel: str = "y",
    title: Optional[str] = None,
    ratio_label: Optional[str] = None,
    ratio_ylim: Tuple[float, float] = (0.9, 1.1),
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    mask_good: Optional[Sequence[np.ndarray]] = None,
    savepath: Optional[str] = None,
    dpi: int = 200,
):
    """
    Multi-curve compare plot (e.g. multiple p-bins or theta-bins):
      - curvesA[k] vs x: solid marker
      - curvesB[k] vs x: dashed marker
      - ratio: curvesB[k]/curvesA[k] in bottom panel

    mask_good (optional):
      list of boolean masks per curve to hide invalid bins.

    Returns: (fig, (ax_top, ax_ratio))
    """
    x = np.asarray(x)

    n = len(labels)
    if len(curvesA) != n or len(curvesB) != n:
        raise ValueError("curvesA/curvesB must match labels length")

    if curvesAerr is None:
        curvesAerr = [None] * n
    if curvesBerr is None:
        curvesBerr = [None] * n
    if mask_good is None:
        mask_good = [np.ones_like(x, dtype=bool) for _ in range(n)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k in range(n):
        c = color_cycle[k % len(color_cycle)]
        m = np.asarray(mask_good[k], dtype=bool)

        yA = np.asarray(curvesA[k])
        yB = np.asarray(curvesB[k])
        eA = curvesAerr[k] if curvesAerr[k] is None else np.asarray(curvesAerr[k])
        eB = curvesBerr[k] if curvesBerr[k] is None else np.asarray(curvesBerr[k])

        yA_plot = np.where(m, yA, np.nan)
        yB_plot = np.where(m, yB, np.nan)
        eA_plot = None if eA is None else np.where(m, eA, np.nan)
        eB_plot = None if eB is None else np.where(m, eB, np.nan)

        ax1.errorbar(x, yA_plot, yerr=eA_plot, fmt="o-", ms=4, color=c,
                     label=f"{labelA}, {labels[k]}")
        ax1.errorbar(x, yB_plot, yerr=eB_plot, fmt="s--", ms=4, color=c,
                     label=f"{labelB}, {labels[k]}")

        r, rerr = _safe_ratio_and_err(yB_plot, yA_plot, eB_plot, eA_plot)
        ax2.errorbar(x, r, yerr=rerr, fmt="o-", ms=4, color=c)

    ax1.set_ylabel(ylabel)
    if title:
        ax1.set_title(title)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(fontsize=7, ncol=2)

    if xscale:
        ax1.set_xscale(xscale)
        ax2.set_xscale(xscale)
    if yscale:
        ax1.set_yscale(yscale)

    ax2.axhline(1.0, color="gray", ls="--", lw=1)
    ax2.set_ylim(*ratio_ylim)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ratio_label if ratio_label else f"{labelB}/{labelA}")
    ax2.grid(True, ls=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, (ax1, ax2)


# ----------------------------------------------------------------------
# 3) Histogram compare (CLD vs ARC)
# ----------------------------------------------------------------------
def plot_step_hist_compare(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 60,
    x_range: Optional[Tuple[float, float]] = None,
    density: bool = True,
    labelA: str = "CLD",
    labelB: str = "ARC",
    xlabel: str = "x",
    title: Optional[str] = None,
    logy: bool = False,
    savepath: Optional[str] = None,
    dpi: int = 200,
):
    """
    Step histogram overlay: a vs b.
    Returns: (fig, ax)
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(a, bins=bins, range=x_range, histtype="step", density=density,
            linewidth=1.8, label=labelA)
    ax.hist(b, bins=bins, range=x_range, histtype="step", density=density,
            linewidth=1.8, linestyle="--", label=labelB)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized entries" if density else "Entries")
    if title:
        ax.set_title(title)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()

    if logy:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()

    if savepath:
        fig.savefig(savepath, dpi=dpi)

    return fig, ax
