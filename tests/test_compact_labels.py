#!/usr/bin/env python3
"""Re-densifying cluster labels must keep 0 = noise: with a noise label present it equals the plain
`torch.unique(..., return_inverse=True)`; with every hit assigned (e.g. a Bhabha event) no cluster may become noise.

    python3 tests/test_compact_labels.py
"""
import sys
import torch
sys.path.insert(0, ".")
from src.layers.inference_oc import compact_labels_preserve_zero

fails = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)


g = torch.Generator().manual_seed(1)
same = True
for _ in range(500):
    lab = torch.randint(0, 40, (300,), generator=g)
    lab[0] = 0
    same &= torch.equal(compact_labels_preserve_zero(lab), torch.unique(lab, return_inverse=True)[1])
check("noise present: same as torch.unique inverse", same)
lab = torch.tensor([3, 3, 7, 7, 7, 12])
out = compact_labels_preserve_zero(lab)
check("all hits assigned: every cluster kept", out.tolist() == [1, 1, 2, 2, 2, 3])
check("empty event", compact_labels_preserve_zero(torch.zeros(0, dtype=torch.long)).numel() == 0)
print("FAILED: " + ", ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
