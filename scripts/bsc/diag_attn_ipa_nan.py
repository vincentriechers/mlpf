"""Localise where an Attn-IPA forward/loss first goes non-finite, on ONE real
DELPHI batch. Run under the same env as training:

    PYTHONPATH=$PWD python scripts/bsc/diag_attn_ipa_nan.py --delphi \
        --data-train /gpfs/scratch/ehpc1013/vriecher/data/500k_delana/digitized/ ...

Prints an isfinite audit of, in order: the raw graph fields, build_targets'
outputs, the encoder output, the decoder's per-layer logits, and finally each
loss term. The first FALSE is the culprit.
"""
import glob
import sys

import torch

from src.utils.parser_args import parser
from src.utils.train_utils import train_load
from src.models.Mask3D.targets import build_targets
from src.models.Mask3D.loss import mask3d_loss
from src.models.Mask3D.matcher import Matcher
from src.models.Mask3D.attn_ipa_model import AttnIPAModel


def audit(name, t):
    if t is None:
        print(f"  {name:34} None"); return True
    if not torch.is_tensor(t):
        print(f"  {name:34} (not a tensor: {type(t).__name__})"); return True
    if not t.is_floating_point():
        print(f"  {name:34} {str(tuple(t.shape)):20} dtype={t.dtype} "
              f"min={t.min().item() if t.numel() else '-'} max={t.max().item() if t.numel() else '-'}")
        return True
    ok = bool(torch.isfinite(t).all())
    n_nan = int(torch.isnan(t).sum()); n_inf = int(torch.isinf(t).sum())
    flag = "OK " if ok else "*** NON-FINITE"
    extra = ""
    if t.numel():
        f = t[torch.isfinite(t)]
        if f.numel():
            extra = f" finite[min/max]={f.min().item():.4g}/{f.max().item():.4g}"
    print(f"  {name:34} {str(tuple(t.shape)):20} {flag} nan={n_nan} inf={n_inf}{extra}")
    return ok


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
    batch = next(iter(train_loader))
    g, y = batch[0], batch[1]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = g.to(dev)

    print("\n=== 1. raw graph fields (from the parquet loader) ===")
    for k in ("pos_hits_xyz", "e_hits", "p_hits", "hit_type",
              "chi_squared_tracks", "particle_number", "pos_pxpypz_at_calo"):
        if k in g.ndata:
            audit(f"g.ndata[{k}]", g.ndata[k])
        else:
            print(f"  g.ndata[{k}]  ABSENT")
    print("\n=== 2. Particles_GT (y) ===")
    for k in ("E", "pid", "coord", "angle", "batch_number"):
        audit(f"y.{k}", getattr(y, k, None))

    print("\n=== 3. build_targets ===")
    tg = build_targets(g, y=y, ILD=False)
    for k in ("feats_flat", "key_valid", "gt_mask", "gt_valid",
              "target_E", "target_coord", "target_pid", "hit_e", "hit_subsystem"):
        if k in tg:
            audit(f"targets[{k}]", tg[k])
    gt = tg["gt_mask"]
    print(f"  gt_mask hits per particle: min={int(gt.sum(-1).min())} "
          f"max={int(gt.sum(-1).max())}  particles with ZERO hits="
          f"{int(((gt.sum(-1) == 0) & tg['gt_valid']).sum())}")
    print(f"  gt_valid particles/event: {tg['gt_valid'].sum(-1).tolist()[:8]}")

    print("\n=== 4. model forward ===")
    opts = dict(track_loss_weight=3.0, window_size=None)
    model = AttnIPAModel(args, dev, **opts).to(dev).eval()
    with torch.no_grad():
        per_layer, targets, q_s, _ = model(g, y, 0)
    audit("encoder->decoder queries q_s", q_s)
    for i, L in enumerate(per_layer):
        if i in (0, len(per_layer) - 1):
            audit(f"per_layer[{i}].mask_logits", L["mask_logits"])
            audit(f"per_layer[{i}].cls_logits", L["cls_logits"])

    print("\n=== 5. loss terms ===")
    with torch.no_grad():
        loss, parts = mask3d_loss(
            per_layer, targets, Matcher(parallel_solver=False),
            weights=model.loss_weights, aux_layer_weight=model.aux_layer_weight,
            mask_loss_type=model.mask_loss_type, focal_gamma=model.focal_gamma,
            mask_cost_use_classification=model.mask_cost_use_classification,
            obj_bce_cost_weight=model.obj_bce_cost_weight,
            track_loss_weight=model.track_loss_weight,
            track_hit_type=model.track_hit_type,
            dice_size_weighting=model.dice_size_weighting,
        )
    print(f"  total loss = {float(loss)}")
    for k, v in parts.items():
        try:
            fv = float(v)
            print(f"  {k:34} {fv}   {'*** NON-FINITE' if fv != fv or abs(fv) == float('inf') else ''}")
        except Exception:
            print(f"  {k:34} {v!r}")


if __name__ == "__main__":
    main()
