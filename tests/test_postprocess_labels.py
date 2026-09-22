#!/usr/bin/env python3
"""`postprocess_labels` must be inert when disabled and do exactly what it says
on a hand-built event when enabled.

    python3 tests/test_postprocess_labels.py
"""
import os, sys
import torch
sys.path.insert(0, ".")
from src.layers.postprocess_labels import postprocess_labels


class G:  # minimal stand-in for a DGL graph
    def __init__(self, pos, ht, e):
        self.ndata = {"pos_hits_xyz": pos, "hit_type": ht, "e_hits": e}


fails = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)


# Event, mm. Two showers A (x~0) and B (x~1000), a fragment F 20 mm from A,
# a far-away lone calo hit L, and three tracks: t1 unassigned near A, t2
# unassigned near B, t3 unassigned far from anything.
pos = torch.tensor([
    [0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [5., 5., 0.],     # A calo (4)
    [1000., 0., 0.], [1010., 0., 0.], [1000., 10., 0.],           # B calo (3)
    [25., 0., 0.], [30., 0., 0.],                                 # F fragment (2)
    [3000., 3000., 0.],                                           # L lone (1)
    [2., 2., 0.],                                                 # t1 near A
    [1002., 2., 0.],                                              # t2 near B
    [5000., 0., 0.],                                              # t3 far
])
ht = torch.tensor([2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 1, 1, 1])
e = torch.tensor([1., 1., 1., 1., 2., 2., 2., 0.1, 0.1, 0.3, 0., 0., 0.])
lab0 = torch.tensor([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 0, 0, 0])
g = G(pos, ht, e)

print("1. disabled -> identical object returned")
os.environ.pop("CLUSTER_POST", None)
check("returns the same tensor", postprocess_labels(g, lab0) is lab0)

print("2. attach: tracks within R go to the nearest trackless cluster, far one stays 0")
cfg = dict(r_attach=100., attach_any=False, lonely=False, stats=False)
lab = postprocess_labels(g, lab0, cfg)
check("t1 -> A", int(lab[10]) == 1)
check("t2 -> B", int(lab[11]) == 2)
check("t3 stays unassigned", int(lab[12]) == 0)
check("calo labels untouched", torch.equal(lab[:10], lab0[:10]))

print("3. attach, one track per trackless cluster: a second track near A is refused")
pos2 = torch.cat([pos, torch.tensor([[4., 0., 0.]])]); ht2 = torch.cat([ht, torch.tensor([1])])
e2 = torch.cat([e, torch.tensor([0.])]); lab2 = torch.cat([lab0, torch.tensor([0])])
lab = postprocess_labels(G(pos2, ht2, e2), lab2, cfg)
check("exactly one of the two A-tracks attached", int((lab[[10, 13]] == 1).sum()) == 1)
check("the closer one won", int(lab[10]) == 1)
cfg_any = dict(cfg, attach_any=True)
lab = postprocess_labels(G(pos2, ht2, e2), lab2, cfg_any)
check("with attach_any both attach", int((lab[[10, 13]] == 1).sum()) == 2)

print("4. lonely: leftover tracks become their own clusters")
lab = postprocess_labels(g, lab0, dict(cfg, lonely=True))
check("t3 got a new label", int(lab[12]) == 5)

print("ALL OK")
