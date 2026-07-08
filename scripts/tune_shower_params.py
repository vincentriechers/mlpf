"""
Grid search over ShowerCostParams to maximise mean F1 across a set of events.
Evaluates on the same parquet file used by uot_3d_visualisation.py.

Usage
-----
    python3 scripts/tune_shower_params.py
"""
import sys
import numpy as np
import awkward as ak
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.uot_3d_visualisation import (
    load_event, extract_tracks, extract_hits,
    get_truth_neutrality, run_uot,
)
from src.dataset.shower_cost import ShowerCostParams

PARQUET = (
    "/eos/experiment/fcc/users/m/mgarciam/mlpf/CLD/train/"
    "Z_uds_CLD_o2_v05_eval_v1/05/pf_tree_10601.parquet"
)
# Use the same auto-selected events from previous runs
EVENTS = [2, 10, 14, 0, 1, 3, 4, 7, 13]

# ─── grid ────────────────────────────────────────────────────────────────────
GRID = {
    "sigma0":      [10.0, 15.0, 20.0, 30.0],
    "alpha":       [0.05, 0.08, 0.10, 0.12, 0.15],
    "d_max":       [300.0, 400.0, 450.0, 550.0, 650.0],
    "sigma_rise":  [100.0, 150.0, 200.0, 300.0],
    "sigma_tail":  [1000.0, 1500.0, 2000.0, 3000.0],
}
# Fixed
MAX_D3D   = 6000.0
N_MIN     = 20


def eval_params(events_data, params: ShowerCostParams):
    f1s, precs, recs = [], [], []
    for ev, tracks, hits, truth_neutral in events_data:
        if tracks is None or hits is None:
            continue
        asgn, _ = run_uot(tracks, hits, cost_mode="shower", shower_params=params)
        pred_neutral = asgn < 0
        TP = int(np.sum( pred_neutral &  truth_neutral))
        FP = int(np.sum( pred_neutral & ~truth_neutral))
        FN = int(np.sum(~pred_neutral &  truth_neutral))
        prec = TP / (TP + FP + 1e-10)
        rec  = TP / (TP + FN + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        f1s.append(f1); precs.append(prec); recs.append(rec)
    return np.mean(f1s), np.mean(precs), np.mean(recs)


def main():
    print(f"Loading {PARQUET} …")
    data = ak.from_parquet(PARQUET)

    print("Preparing events …")
    events_data = []
    for idx in EVENTS:
        ev     = load_event(data, idx)
        tracks = extract_tracks(ev)
        hits   = extract_hits(ev)
        if hits is None or len(hits["E"]) == 0:
            continue
        truth_neutral = get_truth_neutrality(tracks, hits)
        events_data.append((idx, tracks, hits, truth_neutral))
    print(f"  {len(events_data)} events loaded\n")

    keys   = list(GRID.keys())
    values = list(GRID.values())
    total  = 1
    for v in values:
        total *= len(v)
    print(f"Searching {total} combinations …\n")

    best_f1    = -1.0
    best_params = None
    best_row   = None
    results    = []

    for combo in product(*values):
        p = ShowerCostParams(
            sigma0=combo[0], alpha=combo[1],
            d_max=combo[2], sigma_rise=combo[3], sigma_tail=combo[4],
            max_d3D=MAX_D3D, n_min_hits=N_MIN,
        )
        f1, prec, rec = eval_params(events_data, p)
        results.append((f1, prec, rec, combo))
        if f1 > best_f1:
            best_f1     = f1
            best_params = p
            best_row    = (f1, prec, rec, combo)

    # ── top-20 sorted by F1 ───────────────────────────────────────────────────
    results.sort(key=lambda x: -x[0])
    print(f"{'Rank':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  "
          + "  ".join(f"{k:>12}" for k in keys))
    print("-" * (4 + 7*3 + 13*len(keys)))
    for rank, (f1, prec, rec, combo) in enumerate(results[:20], 1):
        vals = "  ".join(f"{v:>12.1f}" for v in combo)
        print(f"{rank:>4}  {f1:.4f}  {prec:.4f}  {rec:.4f}  {vals}")

    print(f"\n{'='*70}")
    print("Best ShowerCostParams:")
    print(f"  sigma0     = {best_params.sigma0}")
    print(f"  alpha      = {best_params.alpha}")
    print(f"  d_max      = {best_params.d_max}")
    print(f"  sigma_rise = {best_params.sigma_rise}")
    print(f"  sigma_tail = {best_params.sigma_tail}")
    print(f"  max_d3D    = {best_params.max_d3D}")
    print(f"  n_min_hits = {best_params.n_min_hits}")
    print(f"\n  mean F1={best_row[0]:.4f}  prec={best_row[1]:.4f}  rec={best_row[2]:.4f}")

    # ── per-event breakdown for best params ──────────────────────────────────
    print(f"\nPer-event breakdown (best params):")
    print(f"  {'Evt':>4}  {'trk':>4}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    for idx, tracks, hits, truth_neutral in events_data:
        asgn, _ = run_uot(tracks, hits, cost_mode="shower", shower_params=best_params)
        pred = asgn < 0
        TP = int(np.sum( pred &  truth_neutral))
        FP = int(np.sum( pred & ~truth_neutral))
        FN = int(np.sum(~pred &  truth_neutral))
        prec = TP / (TP + FP + 1e-10)
        rec  = TP / (TP + FN + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        n_trk = len(tracks["eta"]) if tracks else 0
        print(f"  {idx:>4}  {n_trk:>4}  {f1:.4f}  {prec:.4f}  {rec:.4f}")


if __name__ == "__main__":
    main()
