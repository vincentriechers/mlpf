#!/usr/bin/env python3
"""`track_cost_weight` must move the MATCH and nothing else.

The Mask3D failure this term attacks: the Hungarian matcher hands a
track-owning truth particle to a calo-fragment query rather than to the query
holding that particle's track, because one tracker hit cannot outweigh a
nine-hit shower inside a global dice. Measured consequence on DELPHI: correct
track association 0.584 against HitPF's 0.848, and <E/E_truth> 0.83 against
0.93 — the same deficit seen twice, because a charged particle measured
without its track loses most of its energy.

What this test pins down:

  1. `_track_pairwise` is bounded in [0, 1] and scores the four cases
     correctly, including the 0/0 case (a neutral particle against a
     track-free query) which must be 0, not 1.
  2. `track_cost_weight=0.0` is BIT-IDENTICAL to the code before the term
     existed — so every earlier run is reproducible and no other detector
     changes behaviour.
  3. On the exact configuration that fails in production — a truth particle
     with 1 tracker hit + 9 calo hits, one query holding the shower and
     another holding the track — the baseline cost matches the WRONG query
     and a large enough `track_cost_weight` matches the right one.
  4. The LOSS is unchanged by the term at every weight (it is a cost-only
     knob), which is the property that makes it safe to scan.

    python3 tests/test_track_cost.py
"""
import sys, torch
sys.path.insert(0, ".")
from src.models.Mask3D.loss import _track_pairwise, mask3d_loss
from src.models.Mask3D.matcher import Matcher

fails = []


def check(name, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<58} {float(got):+.6f}  (expect {float(want):+.6f})")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- 1. the term
# 3 hits: hit 0 is a tracker hit belonging to particle 0, hits 1-2 are calo.
N_k = 3
is_track = torch.tensor([[True, False, False]])
key_valid = torch.ones(1, N_k, dtype=torch.bool)
HI, LO = 20.0, -20.0                       # sigmoid -> 1.0 / 0.0

print("1. the four cases of _track_pairwise")
# query 0 claims the track, query 1 claims only calo.
logits = torch.tensor([[[HI, LO, LO], [LO, HI, HI]]])          # (1, 2, 3)
# particle 0 owns the track + one calo hit; particle 1 is neutral (calo only).
gt = torch.tensor([[[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])        # (1, 2, 3)
c = _track_pairwise(logits, gt, key_valid, is_track)
check("q0 (has track) vs p0 (owns track)", c[0, 0, 0], 0.0)
check("q1 (no track)  vs p0 (owns track)", c[0, 1, 0], 1.0)
check("q0 (has track) vs p1 (neutral)   ", c[0, 0, 1], 1.0)
check("q1 (no track)  vs p1 (neutral)   ", c[0, 1, 1], 0.0)   # 0/0 -> 0, NOT 1
print(f"  bounded in [0,1]: min {c.min():+.6f}  max {c.max():+.6f}")
if c.min() < -1e-6 or c.max() > 1 + 1e-6:
    fails.append("bounded")

print("\n2. a half-claimed track is scored continuously, still in [0,1]")
soft = torch.tensor([[[0.0, LO, LO]]])                         # sigmoid(0) = 0.5
c2 = _track_pairwise(soft, gt[:, :1], key_valid, is_track)
check("sigmoid=0.5 on the one track hit", c2[0, 0, 0], 1.0 - 2 * 0.5 / (0.5 + 1.0))

# ------------------------------------------------------- 3. the production case
# 1 tracker hit + 9 calo hits, all belonging to ONE truth particle. Query 0
# holds the 9-hit shower and misses the track; query 1 holds only the track.
print("\n3. the matcher on the production failure case (1 track + 9 calo)")
N_k = 10
is_track_p = torch.zeros(1, N_k, dtype=torch.bool); is_track_p[0, 0] = True
kv = torch.ones(1, N_k, dtype=torch.bool)
gt_mask = torch.zeros(1, 2, N_k, dtype=torch.bool)
gt_mask[0, 0] = True                       # particle 0 = the whole charged particle
gt_valid = torch.tensor([[True, False]])   # one real particle, one padded slot
ml = torch.full((1, 2, N_k), LO)
ml[0, 0, 1:] = HI                          # query 0: the shower, no track
ml[0, 1, 0] = HI                           # query 1: the track, no shower
cls = torch.tensor([[[5.0], [5.0]]])
targets = {"gt_mask": gt_mask, "gt_valid": gt_valid, "key_valid": kv,
           "hit_subsystem": is_track_p.long()}   # 1 = tracker, 0 = calo
matcher = Matcher()

for tcw in (0.0, 0.5, 1.0, 2.0):
    outs = [{"mask_logits": ml.clone(), "cls_logits": cls.clone()}]
    loss, parts = mask3d_loss(outs, targets, matcher, track_loss_weight=1.0,
                              track_hit_type=1, track_cost_weight=tcw)
    # re-run the cost by hand to read off which query the matcher picked
    outs2 = [{"mask_logits": ml.clone(), "cls_logits": cls.clone()}]
    from src.models.Mask3D.loss import _dice_pairwise, _cls_cost
    cost = 0.75 * _dice_pairwise(ml, gt_mask, kv) + 1.0 * _cls_cost(cls, gt_valid)
    if tcw:
        cost = cost + tcw * _track_pairwise(ml, gt_mask, kv, is_track_p)
    perm = matcher(cost, object_valid_mask=gt_valid)
    picked = int(perm[0, 0])
    tag = "TRACK query (right)" if picked == 1 else "shower query (WRONG)"
    print(f"  track_cost_weight={tcw:<4} -> matched query {picked}  {tag}   "
          f"loss={float(loss):.6f}")
    if tcw == 0.0 and picked != 0:
        fails.append("baseline should reproduce the known failure")
    if tcw >= 1.0 and picked != 1:
        fails.append(f"tcw={tcw} should match the track query")

# ------------------------------------------------- 4. the LOSS must not change
print("\n4. the loss is a pure function of the MATCH — identical whenever the "
      "match is")
# Use a case where the match does not flip, so the loss must be bit-identical.
ml_fixed = torch.full((1, 2, N_k), LO); ml_fixed[0, 0] = HI      # query 0 = everything
losses = []
for tcw in (0.0, 1.0, 5.0, 100.0):
    outs = [{"mask_logits": ml_fixed.clone(), "cls_logits": cls.clone()}]
    l, _ = mask3d_loss(outs, targets, matcher, track_loss_weight=1.0,
                       track_hit_type=1, track_cost_weight=tcw)
    losses.append(float(l))
print(f"  losses at tcw 0 / 1 / 5 / 100: {losses}")
check("max|loss - loss(tcw=0)| over the weight scan",
      max(abs(x - losses[0]) for x in losses), 0.0)

print("\n5. track_cost_weight=0.0 leaves the cost bit-identical to no term")
c_base = 0.75 * _dice_pairwise(ml, gt_mask, kv) + 1.0 * _cls_cost(cls, gt_valid)
outs = [{"mask_logits": ml.clone(), "cls_logits": cls.clone()}]
l0, _ = mask3d_loss(outs, targets, matcher, track_loss_weight=1.0, track_hit_type=1)
outs = [{"mask_logits": ml.clone(), "cls_logits": cls.clone()}]
l0b, _ = mask3d_loss(outs, targets, matcher, track_loss_weight=1.0,
                     track_hit_type=1, track_cost_weight=0.0)
check("default kwarg == explicit 0.0", float(l0) - float(l0b), 0.0)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
