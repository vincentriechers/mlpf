#!/usr/bin/env python3
"""`hit_weight` must appear exactly ONCE in every weighted loss term.

Regression test for the bug that made `track_loss_weight` corrupt the Mask3D
objective. `_dice_pairwise` and `_mask_dice_loss` scaled BOTH `p` and `t` by the
weight, so the intersection carried w^2 against a denominator carrying w, and
`1 - 2*inter/(p_sum + t_sum)` went UNBOUNDED BELOW. `_bce_pairwise` and
`_focal_pairwise` had the same shape: `diff` and `t` were both weighted.

Every Mask3D run in the DELPHI study used track_loss_weight=3.0, at which a
PERFECTLY predicted cluster of one tracker hit and nine calo hits scored
dice = -0.5 instead of 0. At 10.0 it scored -4.74 and training diverged
downward (job 45244434, cancelled).

    python3 tests/test_dice_hit_weight.py
"""
import sys, torch
sys.path.insert(0, ".")
from src.models.Mask3D.loss import (_dice_pairwise, _mask_dice_loss,
                                    _bce_pairwise, _focal_pairwise)

N = 10                                   # hit 0 is the tracker hit
WEIGHTS = (1.0, 3.0, 10.0, 50.0)
fails = []


def hw(w):
    h = torch.ones(1, N); h[0, 0] = w; return h


def check(name, got, want, tol=1e-5):
    ok = abs(got - want) < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<52} {got:+.6f}  (expect {want:+.6f})")
    if not ok:
        fails.append(name)


print("1. a PERFECT prediction scores dice = 0 at every weight")
logits, targets = torch.full((1, 1, N), 20.0), torch.ones(1, 1, N)
kv = torch.ones(1, N, dtype=torch.bool)
for w in WEIGHTS:
    check(f"_dice_pairwise, track_loss_weight={w}",
          _dice_pairwise(logits, targets, kv, hit_weight=hw(w)).item(), 0.0)
    check(f"_mask_dice_loss, track_loss_weight={w}",
          _mask_dice_loss(logits, targets.bool(), kv,
                          torch.ones(1, 1, dtype=torch.bool), hit_weight=hw(w)).item(), 0.0)

print("\n2. a fully WRONG prediction scores dice = 1 at every weight")
wrong = torch.full((1, 1, N), -20.0)
for w in WEIGHTS:
    check(f"_dice_pairwise (empty pred), track_loss_weight={w}",
          _dice_pairwise(wrong, targets, kv, hit_weight=hw(w)).item(), 1.0, tol=1e-4)

print("\n3. weight=1 is identical to no weighting, for all four terms")
torch.manual_seed(0)
lg = torch.randn(1, 3, N); tg = (torch.rand(1, 2, N) > 0.5)
for fn, nm in ((_dice_pairwise, "_dice_pairwise"), (_bce_pairwise, "_bce_pairwise"),
               (_focal_pairwise, "_focal_pairwise")):
    a = fn(lg, tg, kv, hit_weight=None)
    b = fn(lg, tg, kv, hit_weight=torch.ones(1, N))
    d = (a - b).abs().max().item()
    print(f"  {'ok  ' if d < 1e-6 else 'FAIL'} {nm:<52} max|diff| = {d:.2e}")
    if d >= 1e-6:
        fails.append(nm + " w=1 identity")

print("\n4. every term stays bounded as the weight grows (no runaway)")
for fn, nm, lo, hi in ((_dice_pairwise, "_dice_pairwise", -1e-4, 1.0001),
                       (_bce_pairwise, "_bce_pairwise", -1e-4, 1e3),
                       (_focal_pairwise, "_focal_pairwise", -1e-4, 1e3)):
    for w in WEIGHTS + (1e3,):
        v = fn(lg, tg, kv, hit_weight=hw(w))
        mn, mx = v.min().item(), v.max().item()
        ok = mn >= lo and mx <= hi
        print(f"  {'ok  ' if ok else 'FAIL'} {nm} w={w:<6g} range [{mn:+.4f}, {mx:+.4f}]")
        if not ok:
            fails.append(f"{nm} unbounded at w={w}")

print("\n" + ("FAILED: " + ", ".join(fails) if fails else "ALL CHECKS PASS"))
sys.exit(1 if fails else 0)
