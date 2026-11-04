"""Per-hit embedding dump at inference, env DUMP_EMBEDDINGS=<dir> (unset: every call is a no-op).

Per hit: ev (event counter of this process), hit_type, e (e_hits), p (p_hits), pos (pos_hits_xyz, mm), pn
(particle_number), lab (final predicted label) and the embedding: HitPF coords (3) + beta (logit); Mask3D feat
(unit-norm mask-head hit projection, float16), maxp (max sigmoid mask logit over the competing queries) and q (that
query). Mask3D also writes the valid queries' unit-norm projections (qfeat, qidx, qev, qcls). One npz per
DUMP_EMBEDDINGS_EVERY events (default 200) and at exit. Reads model tensors only.
"""
import atexit
import os

import numpy as np

_STASH = {}
_BUF = []
_STATE = {"ev": 0, "file": 0, "hooked": False}


def enabled():
    return os.environ.get("DUMP_EMBEDDINGS", "")


def stash(key, value):
    """Keep the latest value of a tensor (or tuple of tensors) computed deep inside the model."""
    if enabled():
        _STASH[key] = tuple(v.detach() for v in value) if isinstance(value, tuple) else value.detach()


def _np(t, dtype):
    return t.detach().float().cpu().numpy().astype(dtype) if t is not None else None


def add_event(g, labels, coords=None, beta=None, feat=None, maxp=None, q=None, queries=None):
    if not enabled():
        return
    nd = g.ndata
    n = int(nd["hit_type"].shape[0])
    rec = {
        "ev": np.full(n, _STATE["ev"], np.int32),
        "hit_type": _np(nd["hit_type"].view(-1), np.int8),
        "e": _np(nd["e_hits"].view(-1), np.float32),
        "p": _np(nd["p_hits"].view(-1), np.float32),
        "pos": _np(nd["pos_hits_xyz"], np.float32),
        "pn": _np(nd["particle_number"].view(-1), np.int32),
        "lab": _np(labels.view(-1), np.int32),
    }
    for k, v, dt in (("coords", coords, np.float32), ("beta", beta, np.float32), ("feat", feat, np.float16),
                     ("maxp", maxp, np.float32), ("q", q, np.int16)):
        if v is not None:
            rec[k] = _np(v.reshape(n, -1) if v.dim() > 1 else v.view(-1), dt)
    if queries is not None:                       # (qfeat (K, D), qidx (K,), qcls (K,))
        qf, qi, qc = queries
        rec["qfeat"], rec["qidx"], rec["qcls"] = _np(qf, np.float16), _np(qi, np.int16), _np(qc, np.float32)
        rec["qev"] = np.full(len(rec["qidx"]), _STATE["ev"], np.int32)
    _BUF.append(rec)
    _STATE["ev"] += 1
    if not _STATE["hooked"]:
        atexit.register(flush)
        _STATE["hooked"] = True
    if len(_BUF) >= int(os.environ.get("DUMP_EMBEDDINGS_EVERY", "200")):
        flush()


def add_mask3d_event(g, labels, ev_mask_logits, ev_cls_logits, i, n_i, cls_threshold):
    """Mask3D: slice the stashed mask-head projections for event i, winning query as in labels_from_masks."""
    if not enabled():
        return
    import torch
    x, o = _STASH["mask_head"]                    # (B, N_max, D), (B, N_q, D)
    probs = ev_mask_logits.float().sigmoid()
    cls = ev_cls_logits.view(-1).float().sigmoid()
    valid = cls >= cls_threshold
    if os.environ.get("M3D_VALID_ARGMAX") and valid.any():
        probs = probs.masked_fill(~valid.view(-1, 1), 0.0)
    maxp, q = probs.max(dim=0)
    qi = torch.nonzero(valid).view(-1)
    add_event(g, labels, feat=x[i, :n_i], maxp=maxp, q=q, queries=(o[i, qi], qi, cls[qi]))


def flush():
    if not _BUF:
        return
    out = {}
    for k in _BUF[0]:
        out[k] = np.concatenate([r[k] for r in _BUF if k in r], axis=0)
    d = enabled()
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "emb_%d_%04d.npz" % (os.getpid(), _STATE["file"])), **out)
    print("[embedding_dump] %d events -> %s" % (len(_BUF), d), flush=True)
    _STATE["file"] += 1
    _BUF.clear()
