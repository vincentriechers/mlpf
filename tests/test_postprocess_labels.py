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
    def __init__(self, pos, ht, e, p=None):
        self.ndata = {"pos_hits_xyz": pos, "hit_type": ht, "e_hits": e}
        if p is not None:
            self.ndata["p_hits"] = p


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

print("5. track above the beam energy: leaves its cluster, never attached, never promoted")
# t1 (60 GeV) sits inside A, t2 (5 GeV) inside B, t3 (70 GeV) and t4 (3 GeV) unassigned; t4 near A
pos5 = torch.cat([pos, torch.tensor([[3., 1., 0.]])]); ht5 = torch.cat([ht, torch.tensor([1])])
e5 = torch.cat([e, torch.tensor([0.])]); p5 = torch.tensor([0.] * 10 + [60., 5., 70., 3.])
lab5 = torch.tensor([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 0, 0])
cfg5 = dict(cfg, attach_any=True, lonely=True, track_pmax=45.6)
lab = postprocess_labels(G(pos5, ht5, e5, p5), lab5, cfg5)
check("60 GeV track removed from A", int(lab[10]) == 0)
check("5 GeV track stays in B", int(lab[11]) == 2)
check("70 GeV unassigned track not promoted", int(lab[12]) == 0)
check("3 GeV track near A attached to A", int(lab[13]) == 1)
check("calo labels untouched", torch.equal(lab[:10], lab5[:10]))
lab = postprocess_labels(G(pos5, ht5, e5, p5), lab5, dict(cfg5, track_pmax=float("inf")))
check("switch off: 60 GeV track stays in A", int(lab[10]) == 1)
check("switch off: 70 GeV track promoted", int(lab[12]) > 4)

print("6. one track per cluster: the track nearest the calorimeter centroid stays, the others are promoted")
# A holds t1 (near A) plus t3 (far away) and t2 is alone in B; one-per-cluster keeps t1 in A and frees t3
lab6 = torch.tensor([1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 1, 2, 1])
cfg6 = dict(cfg, lonely=True, one_track=True)
lab = postprocess_labels(g, lab6, cfg6)
check("t1 stays in A", int(lab[10]) == 1)
check("t2 stays in B", int(lab[11]) == 2)
check("t3 released and promoted to its own label", int(lab[12]) > 4)
check("calo labels untouched", torch.equal(lab[:10], lab6[:10]))
lab = postprocess_labels(g, lab6, dict(cfg6, one_track=False))
check("switch off: t3 stays in A", int(lab[12]) == 1)

print("7. few-cell neutral clusters: only an energetic barrel EM cluster without a track and with too few cells is dissolved")
# N1 2 EM cells 3 GeV (dissolved); N2 6 EM cells 3 GeV; N3 2 EM cells 0.6 GeV; C 2 EM cells 3 GeV + track;
# H 2 hadronic cells 3 GeV; F 2 EM cells 3 GeV in the end cap (|z| = 3000 mm)
pos7 = torch.zeros(17, 3); pos7[14:16, 2] = 3000.
ht7 = torch.tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1])
e7 = torch.tensor([1.5, 1.5, .5, .5, .5, .5, .5, .5, .3, .3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.])
lab7 = torch.tensor([1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 4])
g7 = G(pos7, ht7, e7, p=torch.zeros(17))
cfg7 = dict(r_attach=0., attach_any=False, lonely=False, stats=False, neutral_mincells=6, neutral_mincells_e=1.0)
lab = postprocess_labels(g7, lab7, cfg7)
check("N1 dissolved", bool((lab[0:2] == 0).all()))
check("everything else untouched", torch.equal(lab[2:], lab7[2:]))
check("off by default", torch.equal(postprocess_labels(g7, lab7, dict(cfg7, neutral_mincells=0)), lab7))
check("energy threshold respected", torch.equal(postprocess_labels(g7, lab7, dict(cfg7, neutral_mincells_e=5.0)), lab7))

print("ALL OK" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
