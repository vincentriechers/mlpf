#!/usr/bin/env python3
"""Track-owned queries (src/layers/track_pin.py) do exactly what they say.

    python3 tests/test_track_pin.py
"""
import sys, torch
sys.path.insert(0, ".")
from src.layers.track_pin import track_pin_matrix, pin_tracks_to_seed_queries

fails = []
def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)

print("1. inference pin: a tracker hit joins the query it seeded, only if that query is valid")
hit_type = torch.tensor([1, 2, 2, 1, 2])            # keys 0 and 3 are tracks
seed_idx = torch.tensor([3, 0, -1])                 # q0 <- key 3, q1 <- key 0, q2 unseeded
valid_q = torch.tensor([True, False, True])
argmax_q = torch.full((5,), 2); assigned = torch.zeros(5, dtype=torch.bool)
n = pin_tracks_to_seed_queries(argmax_q, assigned, valid_q, seed_idx, hit_type)
check("one hit pinned", n == 1)
check("key 3 -> query 0, assigned", argmax_q[3] == 0 and bool(assigned[3]))
check("key 0 untouched (its query is invalid)", argmax_q[0] == 2 and not bool(assigned[0]))
check("calo hits untouched", (argmax_q[[1, 2, 4]] == 2).all() and not assigned[[1, 2, 4]].any())
check("seed_idx None is a no-op", pin_tracks_to_seed_queries(argmax_q, assigned, valid_q, None, hit_type) == 0)

print("2. matching pin matrix: (query, particle of its seed track) and nothing else")
key_valid = torch.ones(1, 5, dtype=torch.bool)
hit_sub = hit_type.view(1, 5)
key_slot = torch.tensor([[0, 0, 1, 1, -1]])         # key 0 track of p0, key 3 track of p1
gt_valid = torch.tensor([[True, True]])
pin = track_pin_matrix(seed_idx.view(1, 3), key_valid, hit_sub, key_slot, gt_valid)
want = torch.zeros(1, 3, 2, dtype=torch.bool); want[0, 0, 1] = True; want[0, 1, 0] = True
check("pin[q0,p1] and pin[q1,p0] only", torch.equal(pin, want))
pin2 = track_pin_matrix(seed_idx.view(1, 3), key_valid, hit_sub, key_slot, torch.tensor([[True, False]]))
check("invalid GT slot is never pinned", not pin2[0, 0, 1] and pin2[0, 1, 0])
seed_calo = torch.tensor([[1, 2, 4]])               # queries seeded by calo hits
check("calo-seeded queries are never pinned", not track_pin_matrix(seed_calo, key_valid, hit_sub, key_slot, gt_valid).any())
check("empty particle axis is safe", track_pin_matrix(seed_idx.view(1, 3), key_valid, hit_sub, key_slot, torch.zeros(1, 0, dtype=torch.bool)).shape == (1, 3, 0))

print("3. end to end: the pin overrides a perfect-but-swapped Hungarian assignment")
from src.models.Mask3D.loss import mask3d_loss
from src.models.Mask3D.matcher import Matcher
# keys: 0 track p0, 1 calo p0, 2 track p1, 3 calo p1.  q0 predicts p0 perfectly, q1 predicts p1.
gt_mask = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]], dtype=torch.bool)
ml = torch.tensor([[[10., 10., -10., -10.], [-10., -10., 10., 10.]]])
cl = torch.full((1, 2, 1), 10.)
targets = {"gt_mask": gt_mask, "gt_valid": torch.ones(1, 2, dtype=torch.bool),
           "key_valid": torch.ones(1, 4, dtype=torch.bool),
           "hit_subsystem": torch.tensor([[1, 2, 1, 2]]), "key_slot": torch.tensor([[0, 0, 1, 1]]),
           "seed_idx": torch.tensor([[2, 0]])}          # q0 seeded by p1's track, q1 by p0's track
per_layer = [{"mask_logits": ml.clone(), "cls_logits": cl.clone()}]
m = Matcher(parallel_solver=False)
l0, _ = mask3d_loss([{"mask_logits": ml.clone(), "cls_logits": cl.clone()}], targets, m)
l_off, _ = mask3d_loss([{"mask_logits": ml.clone(), "cls_logits": cl.clone()}], targets, m, track_pin_cost=0.0)
l_on, _ = mask3d_loss([{"mask_logits": ml.clone(), "cls_logits": cl.clone()}], targets, m, track_pin_cost=1000.0)
check("track_pin_cost=0 reproduces the default loss", abs(float(l0) - float(l_off)) < 1e-6)
check("pinned matching is the swapped (worse-fitting) one: loss rises", float(l_on) > float(l_off) + 0.5)
t2 = dict(targets); t2["seed_idx"] = torch.tensor([[0, 2]])   # seeds agree with the natural match
l_agree, _ = mask3d_loss([{"mask_logits": ml.clone(), "cls_logits": cl.clone()}], targets | {"seed_idx": torch.tensor([[0, 2]])}, m, track_pin_cost=1000.0)
check("pin that agrees with the natural match leaves the loss unchanged", abs(float(l_agree) - float(l_off)) < 1e-6)

print("\nall checks passed" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
