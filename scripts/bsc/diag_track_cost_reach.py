"""How far does the track-consistency matching cost actually reach?

`track_cost_weight` can only help truth particles whose `gt_mask` contains a
tracker hit. If that were a small fraction, the term would be nearly inert and
a null result would mean "the term never fired", not "the assignment was not
the problem". This measures the reach on REAL DELPHI batches before the answer
is interpreted, and it prints the three numbers that bound it:

  * tracker hits that belong to NO truth particle (`particle_number == 0`).
    Roughly half of DELPHI's tracks are unlabelled noise, which is the reason a
    global `track_loss_weight` is a blunt instrument: it up-weights those too.
    The matching-cost term does not — a noise track is in no particle's
    `gt_mask`, so it only ever costs a query that claims it.
  * valid GT particles that own at least one tracker hit — the population the
    term can move at all.
  * the energy those particles carry, which is what the term is worth if it
    works perfectly.

    PYTHONPATH=$PWD python scripts/bsc/diag_track_cost_reach.py --delphi \
        --data-train <dir>/ --data-config config_files/config_hits_track_delphi.yaml \
        --network-config src/models/wrapper/example_mode_mask3d.py \
        --batch-size 64 --num-workers 2 --fetch-by-files --fetch-step 4 \
        --tracks --start-lr 1e-4 --num-epochs 1 --train-batches 1 --gpus ""
"""
import glob
import sys

import torch

from src.utils.parser_args import parser
from src.utils.train_utils import train_load
from src.models.Mask3D.targets import build_targets

N_BATCHES = 8
TRACK_HIT_TYPE = 1


def main():
    args = parser.parse_args()
    args.ILD = False
    args.pandora = False
    args.local_rank = 0
    files = []
    for folder in args.data_train:
        files.extend(sorted(glob.glob(folder + "*.parquet")))
    args.data_train = files
    train_loader, _, _, _ = train_load(args)

    tot = dict(ev=0, trk=0, trk_noise=0, part=0, part_trk=0,
               E=0.0, E_trk=0.0, hits_with=0, hits_without=0, n_with=0, n_without=0)
    it = iter(train_loader)
    for _ in range(N_BATCHES):
        try:
            batch = next(it)
        except StopIteration:
            break
        g, y = batch[0], batch[1]
        tg = build_targets(g, y=y, ILD=False)
        gt_mask = tg["gt_mask"]                       # (B, N_p, N_k) bool
        gt_valid = tg["gt_valid"]                     # (B, N_p)
        sub = tg["hit_subsystem"]                     # (B, N_k)
        kv = tg["key_valid"]                          # (B, N_k)
        E = tg["target_E"]                            # (B, N_p)
        is_track = (sub == TRACK_HIT_TYPE) & kv

        # a particle "owns a track" if its mask contains >=1 tracker hit
        owns = (gt_mask & is_track.unsqueeze(1)).any(-1) & gt_valid
        # tracker hits claimed by no valid particle = unlabelled / noise tracks
        claimed = (gt_mask & gt_valid.unsqueeze(-1)).any(1)          # (B, N_k)
        n_hits = (gt_mask & kv.unsqueeze(1)).sum(-1)                 # (B, N_p)

        tot["ev"] += int(gt_valid.size(0))
        tot["trk"] += int(is_track.sum())
        tot["trk_noise"] += int((is_track & ~claimed).sum())
        tot["part"] += int(gt_valid.sum())
        tot["part_trk"] += int(owns.sum())
        tot["E"] += float(E[gt_valid].sum())
        tot["E_trk"] += float(E[owns].sum())
        tot["hits_with"] += int(n_hits[owns].sum())
        tot["n_with"] += int(owns.sum())
        sel = gt_valid & ~owns
        tot["hits_without"] += int(n_hits[sel].sum())
        tot["n_without"] += int(sel.sum())

    ev = max(tot["ev"], 1)
    print(f"\n=== track-cost reach, {tot['ev']} real DELPHI events ===")
    print(f"  tracker hits / event                 {tot['trk'] / ev:8.2f}")
    print(f"  ... belonging to NO truth particle   {tot['trk_noise'] / ev:8.2f}"
          f"   ({100 * tot['trk_noise'] / max(tot['trk'], 1):.1f} % of tracks)")
    print(f"  valid GT particles / event           {tot['part'] / ev:8.2f}")
    print(f"  ... owning >= 1 tracker hit          {tot['part_trk'] / ev:8.2f}"
          f"   ({100 * tot['part_trk'] / max(tot['part'], 1):.1f} % of particles)")
    print(f"  truth energy / event                 {tot['E'] / ev:8.2f} GeV")
    print(f"  ... in track-owning particles        {tot['E_trk'] / ev:8.2f} GeV"
          f"   ({100 * tot['E_trk'] / max(tot['E'], 1e-9):.1f} % of the energy)")
    print(f"  mean hits per track-owning particle  "
          f"{tot['hits_with'] / max(tot['n_with'], 1):8.2f}")
    print(f"  mean hits per neutral particle       "
          f"{tot['hits_without'] / max(tot['n_without'], 1):8.2f}")
    print("\nThe term can move the second group and only that group. The last two"
          "\nlines are the 1-vs-N ratio the whole argument rests on: one tracker"
          "\nhit against that many calo hits inside a global dice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
