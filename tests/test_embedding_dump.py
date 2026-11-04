#!/usr/bin/env python3
"""embedding_dump must write nothing when DUMP_EMBEDDINGS is unset, leave the mask head's logits unchanged when set,
and write what it was given.

    python3 tests/test_embedding_dump.py
"""
import glob, os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, ".")
from src.layers import embedding_dump as ed
from src.models.Mask3D.tasks import ObjectHitMaskTask

fails = []


def check(name, ok):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        fails.append(name)


class G:  # minimal stand-in for a DGL graph
    def __init__(self, n):
        self.ndata = {"hit_type": torch.tensor([2.] * (n - 1) + [1.]), "e_hits": torch.rand(n, 1),
                      "p_hits": torch.rand(n, 1), "pos_hits_xyz": torch.rand(n, 3) * 1000,
                      "particle_number": torch.arange(n).float()}


tmp = tempfile.mkdtemp()
os.environ.pop("DUMP_EMBEDDINGS", None)
torch.manual_seed(0)
head = ObjectHitMaskTask(16)
q, k = torch.randn(2, 5, 16), torch.randn(2, 7, 16)
ref = head(q, k).detach()
g = G(7)
ed.add_event(g, torch.zeros(7), coords=torch.rand(7, 3), beta=torch.rand(7))
ed.flush()
check("unset: nothing buffered, no stash", not ed._BUF and not ed._STASH)

os.environ["DUMP_EMBEDDINGS"] = tmp
out = head(q, k).detach()
check("set: mask logits unchanged", torch.equal(out, ref))
x, o = ed._STASH["mask_head"]
check("set: stashed unit-norm projections", x.shape == (2, 7, 16) and torch.allclose(x.norm(dim=-1), torch.ones(2, 7)))
lab = torch.tensor([1, 1, 2, 0, 2, 1, 1])
ed.add_mask3d_event(g, lab, out[1], torch.tensor([3., -3., 3., 3., -3.]), 1, 7, 0.5)
ed.add_event(g, lab, coords=torch.rand(7, 3), beta=torch.rand(7))
ed.flush()
f = sorted(glob.glob(os.path.join(tmp, "emb_*.npz")))
check("one file written", len(f) == 1)
d = np.load(f[0])
check("per-hit arrays", d["e"].shape == (14,) and d["pos"].shape == (14, 3) and d["feat"].shape == (7, 16))
check("labels and event counter", list(d["lab"][:7]) == lab.tolist() and list(np.unique(d["ev"])) == [0, 1])
check("maxp/q = max over all queries", np.allclose(d["maxp"], out[1].sigmoid().max(0).values.numpy())
      and list(d["q"]) == out[1].sigmoid().max(0).indices.tolist())
check("valid queries only", list(d["qidx"]) == [0, 2, 3] and d["qfeat"].shape == (3, 16))
print("FAILED: %s" % fails if fails else "all ok")
sys.exit(1 if fails else 0)
