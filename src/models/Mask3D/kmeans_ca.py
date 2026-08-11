"""k-Max-DeepLab style cross-attention (k-means M-step).

Port of `src/hepattn/utils/kmeans_ca.py` from
github.com/samvanstroud/hepattn @ 1df05ccb ("Implement k-Max-DeepLab").

Replaces softmax(QK^T) V cross-attention with a HARD k-means update:
  - For each KEY position m, find the query that has the highest assignment
    logit (argmax over N queries). The logits are either Q·K (computed
    here) OR (preferred) the previous-layer's mask logits, passed via
    `logits` — exposing the mask predictions themselves as the
    cluster-assignment matrix.
  - The query at that argmax index accumulates the key's value (sum or mean).

Effectively each query becomes the centroid of the keys assigned to it.
Returns the per-query contribution WITHOUT residual (the caller is
expected to do `q = q + KMeansCrossAttention(...)`).
"""
import torch
from torch import nn


class KMeansCrossAttention(nn.Module):
    def __init__(self, dim: int, update: str = "mean", value_proj: bool = False,
                 respect_attn_mask: bool = False, eps: float = 1e-6):
        super().__init__()
        assert update in {"sum", "mean"}, "update must be 'sum' or 'mean'"
        self.dim = dim
        self.update = update
        self.respect_attn_mask = bool(respect_attn_mask)
        self.eps = eps
        self.v_proj = nn.Linear(dim, dim, bias=False) if value_proj else None

    def forward(self, q, k=None, v=None, attn_mask=None, q_mask=None,
                kv_mask=None, logits=None, **kwargs):
        """
        q          (B, N, D)    queries (used only if `logits` is None).
        k          (B, M, D)    keys (used only if `logits` is None).
        v          (B, M, D)    values (the per-key features aggregated into queries).
        attn_mask  (B, N, M)    bool. True = allowed. Only enforced when
                                `respect_attn_mask=True`; otherwise ignored —
                                the assignment logits carry the gating.
        q_mask     (B, N)       bool. True where queries are valid.
        kv_mask    (B, M)       bool. True where keys/values are valid.
        logits     (B, N, M)    optional float. If given, used as the
                                assignment logits. Typical use: pass the
                                previous-layer's mask logits so the same
                                quantity the Mask2Former gate uses ALSO
                                drives the k-means hard assignment.

        Returns (B, N, D): per-query update (no residual added).
        """
        if v is None:
            raise ValueError("KMeansCrossAttention requires v (values).")
        neg_inf = float("-inf")

        if logits is None:
            if k is None:
                raise ValueError("Provide either `logits` or `k`.")
            logits = q @ k.transpose(-2, -1)                              # (B, N, M)

        if q_mask is not None:
            logits = logits.masked_fill(~q_mask.unsqueeze(-1), neg_inf)
        if self.respect_attn_mask and attn_mask is not None:
            logits = logits.masked_fill(~attn_mask, neg_inf)
        if kv_mask is not None:
            logits = logits.masked_fill(~kv_mask.unsqueeze(-2), neg_inf)

        # For each key m, the best query n (argmax over N=dim -2).
        max_val, idx = logits.max(dim=-2)                                  # (B, M), (B, M)
        valid = torch.isfinite(max_val)                                    # (B, M)

        vv = v if self.v_proj is None else self.v_proj(v)                  # (B, M, D)
        vv = vv * valid.unsqueeze(-1).to(vv.dtype)

        B, M, D = vv.shape
        N = logits.shape[-2]
        out = vv.new_zeros((B, N, D))
        # Scatter each key's value into its argmax query's slot.
        out.scatter_add_(1, idx.unsqueeze(-1).expand(B, M, D), vv)

        if self.update == "mean":
            counts = vv.new_zeros((B, N))
            counts.scatter_add_(1, idx, valid.to(vv.dtype))
            out = out / (counts.clamp_min(1.0).unsqueeze(-1) + self.eps)
        return out
