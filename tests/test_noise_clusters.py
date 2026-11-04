#!/usr/bin/env python3
"""noise_pseudo_clusters: label-0 calorimeter cells linked within the radius form one cluster each; tracks and
labelled cells are never part of one; the energy floor applies to the cluster sum.

    python3 tests/test_noise_clusters.py
"""
import sys
import torch
sys.path.insert(0, ".")
from src.layers.noise_clusters import noise_pseudo_clusters


class G:  # minimal stand-in for a DGL graph
    def __init__(self, pos, ht, e):
        self.ndata = {"pos_hits_xyz": pos, "hit_type": ht, "e_hits": e}


fails = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)


# cm. N1: three noise cells in a row 10 cm apart; N2: one isolated noise cell; S: a labelled shower next to N1;
# T: a track node with label 0 inside N1.
pos = torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.],        # N1
                    [500., 0., 0.],                                     # N2
                    [5., 5., 0.], [8., 6., 0.],                         # S (labelled)
                    [12., 1., 0.]])                                     # T (track)
ht = torch.tensor([2., 2., 2., 2., 2., 2., 1.])
e = torch.tensor([0.05, 0.4, 0.05, 0.02, 1.0, 0.5, 0.0])
lab = torch.tensor([0, 0, 0, 0, 3, 3, 0])
g = G(pos, ht, e)
m = noise_pseudo_clusters(g, lab, 12.0)
check("two noise clusters", len(m) == 2)
sets = sorted([tuple(torch.where(x)[0].tolist()) for x in m])
check("N1 = cells 0-2, N2 = cell 3", sets == [(0, 1, 2), (3,)])
check("no track or labelled cell in a cluster", not any(x[4:].any() for x in m))
check("link radius respected (8 cm splits N1)", len(noise_pseudo_clusters(g, lab, 8.0)) == 4)
check("energy floor on the cluster sum", len(noise_pseudo_clusters(g, lab, 12.0, emin=0.1)) == 1)
check("no noise cells -> no cluster", noise_pseudo_clusters(g, torch.tensor([1, 1, 1, 1, 3, 3, 0]), 12.0) == [])
print("FAILED: " + ", ".join(fails) if fails else "all ok")
sys.exit(1 if fails else 0)
