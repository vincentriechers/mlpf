#!/usr/bin/env python
"""End-to-end smoke test for the DELPHI dataset path.

Two stages:
  1. raw checks (any env with awkward/numpy): feature dims (27/14/21),
     events/file, per-event averages against the validated reference numbers
     (raw files: 29.2 tracks, 492 calo hits, 67.5 gen targets, ~8.8 hits per
     with-hit target, median gen E 0.76 GeV).
  2. pipeline checks (training container): load one FILTERED file through the
     actual training input pipeline (SimpleIterDataset -> create_graph with
     --delphi) and assert graph contents, label ranges and mm-scale positions.

Usage:
    python scripts/smoke_test_delphi.py --raw-dir <digitized> [--n-raw-files 20]
    python scripts/smoke_test_delphi.py --filtered-file <digitized_filtered/pf_tree_X.parquet>
(either or both stages, depending on which args are given)
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def close(val, ref, tol=0.10):
    return abs(val - ref) <= tol * abs(ref)


def raw_checks(raw_dir, n_files):
    import awkward as ak

    files = sorted(glob.glob(os.path.join(raw_dir, "pf_tree_*.parquet")))[:n_files]
    assert files, f"no files in {raw_dir}"
    n_ev = n_trk = n_hit = n_gen = n_empty = 0
    hits_per_target_num = hits_per_target_den = 0
    all_E = []
    for fn in files:
        a = ak.from_parquet(fn)
        ne = len(a["X_gen"])
        n_ev += ne
        for ev in range(ne):
            xt = a["X_track"][ev]
            xh = a["X_hit"][ev]
            if len(a["X_gen"][ev]) == 0:
                # rare no-gen-target events; the loader drops them as empty graphs
                n_empty += 1
                continue
            xg = ak.to_numpy(a["X_gen"][ev])
            yh = ak.to_numpy(a["ygen_hit"][ev]).astype(np.int64)
            if len(xt) > 0:
                assert len(xt[0]) == 27, f"X_track dim {len(xt[0])} != 27"
            assert len(xh[0]) == 14, f"X_hit dim {len(xh[0])} != 14"
            assert xg.shape[1] == 21, f"X_gen dim {xg.shape[1]} != 21"
            n_trk += len(xt)
            n_hit += len(xh)
            n_gen += len(xg)
            cnt = np.bincount(yh[yh >= 0], minlength=len(xg))
            hits_per_target_num += cnt[cnt > 0].sum()
            hits_per_target_den += (cnt > 0).sum()
            all_E.append(np.median(xg[:, 8]))  # per-event median, averaged below
    trk_ev, hit_ev, gen_ev = n_trk / n_ev, n_hit / n_ev, n_gen / n_ev
    hpt = hits_per_target_num / hits_per_target_den
    medE = float(np.mean(all_E))  # reference 0.76 = mean over events of per-event median
    print(f"[raw] {len(files)} files, {n_ev} events ({n_empty} with zero gen targets, skipped)")
    print(f"[raw] tracks/ev {trk_ev:.1f} (ref 29.2)   hits/ev {hit_ev:.1f} (ref 492)   targets/ev {gen_ev:.1f} (ref 67.5)")
    print(f"[raw] hits per with-hit target {hpt:.2f} (ref ~8.8)   median gen E {medE:.3f} GeV (ref 0.76)")
    assert close(trk_ev, 29.2), "tracks/ev off by >10%"
    assert close(hit_ev, 492), "hits/ev off by >10%"
    assert close(gen_ev, 67.5), "targets/ev off by >10%"
    assert close(hpt, 8.8), "hits-per-target off by >10%"
    assert close(medE, 0.76), "median gen E off by >10%"
    print("[raw] all checks passed")


def pipeline_checks(filtered_file):
    import torch
    from src.utils.parser_args import parser
    from src.dataset.dataset import SimpleIterDataset

    args = parser.parse_args(["--delphi", "--gpus", ""])
    # mirror what train_lightning1.main() derives from --delphi
    args.ILD = False
    args.pandora = False
    args.local_rank = None
    args.prediction = False

    ds = SimpleIterDataset(
        {"_": [filtered_file]},
        None,
        for_training=True,
        fetch_by_files=True,
        fetch_step=1,
        infinity_mode=False,
        name="delphi_smoke",
        args_parse=args,
    )
    n_ev = n_trk = n_hit = 0
    n_gen_tot = 0
    it = ds.__iter__()  # _SimpleIter implements __next__ only
    while True:
        try:
            g, y = next(it)
        except StopIteration:
            break
        n_ev += 1
        ht = g.ndata["hit_type"]
        n_trk += int((ht == 1).sum())
        n_hit += int((ht != 1).sum())
        n_gen = len(y)
        n_gen_tot += n_gen
        # node features: xyz(3) + one-hot(5) + e(1) + p(1)
        assert g.ndata["h"].shape[1] == 10, f"g.ndata['h'] dim {g.ndata['h'].shape[1]} != 10"
        # labels: 0 (noise) .. n_gen; indices into y
        pn = g.ndata["particle_number"]
        assert pn.min() >= 0 and pn.max() <= n_gen, "particle_number out of range"
        assert torch.isfinite(g.ndata["h"]).all(), "non-finite node features"
        # positions must be mm-scale after the --delphi rescale (DELPHI calo ~2 m)
        calo_r = torch.norm(g.ndata["pos_hits_xyz"][ht != 1], dim=1)
        assert 1000 < calo_r.median() < 10000, f"calo |r| median {calo_r.median():.0f} not mm-scale"
        assert torch.isfinite(y.E).all() and (y.E > 0).all(), "bad target energies"
    print(f"[pipeline] events loaded: {n_ev} (file should have ~100; a few may be dropped as empty)")
    print(f"[pipeline] tracks/ev {n_trk/n_ev:.1f} (ref 29.2)   calo hits/ev {n_hit/n_ev:.1f} (ref 492)")
    print(f"[pipeline] targets/ev {n_gen_tot/n_ev:.1f} (filtered, expect ~56-60)")
    assert n_ev >= 90, "lost more than 10% of events in the pipeline"
    assert close(n_trk / n_ev, 29.2), "pipeline tracks/ev off by >10%"
    assert close(n_hit / n_ev, 492), "pipeline hits/ev off by >10%"
    assert 50 <= n_gen_tot / n_ev <= 66, "pipeline targets/ev outside filtered expectation"
    print("[pipeline] all checks passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--n-raw-files", type=int, default=20)
    p.add_argument("--filtered-file", default=None)
    a = p.parse_args()
    if not a.raw_dir and not a.filtered_file:
        sys.exit("give --raw-dir and/or --filtered-file")
    if a.raw_dir:
        raw_checks(a.raw_dir, a.n_raw_files)
    if a.filtered_file:
        pipeline_checks(a.filtered_file)
