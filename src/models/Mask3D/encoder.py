"""Pre-norm transformer encoder over a flat hit-token stream.

Operates on `(total_hits, D)`. Output is also `(total_hits, D)`; the
LightningModule scatters it into `(B, N_max, D)` once for the decoder.

Two attention modes:

1. Full-event (window_size=None) — uses xformers `BlockDiagonalMask` so
   each event attends to all its own hits, no padding waste. Same pattern
   as `Gatr_pf_e_noise.py`.

2. Phi-windowed sliding attention (window_size=W) — uses flash-attn's
   varlen kernel with `window_size=(W//2, W//2)`. Hits are sorted by
   `phi = atan2(y, x)` within each event before the encoder, so each token
   ends up attending to its W//2 phi-neighbors on each side (clamped at
   the event boundary). True sliding window, NOT non-overlapping chunks.

   `window_wrap=True`: alternate layers cycle-roll each event by W//2
   positions. Phi is naturally cyclic on [-π, π], and the rolling brings
   what used to be the wrap boundary (smallest phi ↔ largest phi) into
   the middle of the sequence, giving cross-boundary attention every two
   layers without needing per-layer cyclic padding.
"""
import torch
import torch.nn as nn

from src.models.Mask3D.attention import (
    BlockDiagonalSelfAttention,
    FlashVarlenSelfAttention,
    FFN,
    _xformers,
)


def _roll_perm_per_event(seq_lens, shift, device):
    """Permutation that rolls each event's hits left by `shift` positions.

    Operates on the flat (total_hits,) layout where events are concatenated.
    Inverse is just `inv[perm] = arange`. Used between alternating encoder
    layers so the phi-wrap boundary is mid-event in odd layers.
    """
    parts = []
    offset = 0
    for L in seq_lens:
        if L <= 0:
            continue
        s = shift % L
        if s == 0:
            local = torch.arange(L, device=device, dtype=torch.long)
        else:
            local = torch.cat([
                torch.arange(s, L, device=device, dtype=torch.long),
                torch.arange(0, s, device=device, dtype=torch.long),
            ])
        parts.append(local + offset)
        offset += L
    if not parts:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.cat(parts)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block — or HybridNorm block (arXiv:2503.04598).

    HybridNorm (`hybrid_norm=True`):
      - layer 0:  pre-norm before attention (standard) + pre-norm before FFN.
      - layer ≥1: NO pre-norm before attention (`qkv_norm` inside handles
                  conditioning) + post-norm style on FFN
                  (`x = LN(x); x = x + FFN(x)`).

    `window_size`: when set, the layer uses flash-attn varlen sliding
    window with `(W//2, W//2)` instead of the xformers BlockDiagonal path.
    """

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0,
                 depth=0, hybrid_norm=False, window_size=None):
        super().__init__()
        is_hybrid_subsequent = hybrid_norm and depth > 0
        self.use_pre_norm_attn  = not is_hybrid_subsequent
        self.use_post_norm_dense = is_hybrid_subsequent
        qkv_norm = hybrid_norm

        self.norm1 = nn.LayerNorm(dim) if self.use_pre_norm_attn else nn.Identity()
        self.window_size = int(window_size) if window_size else None
        if self.window_size:
            self.attn = FlashVarlenSelfAttention(
                dim, num_heads, window_size=self.window_size,
                dropout=dropout, qkv_norm=qkv_norm,
            )
        else:
            self.attn = BlockDiagonalSelfAttention(
                dim, num_heads, dropout=dropout, qkv_norm=qkv_norm,
            )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, x_flat, attn_ctx):
        """`attn_ctx` is the BlockDiagonalMask (full path) OR the tuple
        `(cu_seqlens, max_seqlen)` (windowed path)."""
        normed = self.norm1(x_flat)
        if self.window_size:
            cu_seqlens, max_seqlen = attn_ctx
            x_flat = x_flat + self.attn(normed, cu_seqlens, max_seqlen)
        else:
            x_flat = x_flat + self.attn(normed, attn_ctx)
        if self.use_post_norm_dense:
            x_flat = self.norm2(x_flat)
            x_flat = x_flat + self.ffn(x_flat)
        else:
            x_flat = x_flat + self.ffn(self.norm2(x_flat))
        return x_flat


class Encoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, ffn_mult=4, dropout=0.0,
                 hybrid_norm=False, window_size=None, window_wrap=False):
        super().__init__()
        self.window_size = int(window_size) if window_size else None
        self.window_wrap = bool(window_wrap)
        self.layers = nn.ModuleList(
            [EncoderLayer(dim, num_heads, ffn_mult, dropout,
                          depth=i, hybrid_norm=hybrid_norm,
                          window_size=self.window_size)
             for i in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x_flat, seq_lens, phi=None):
        if not self.window_size:
            _, BlockDiagonalMask = _xformers()
            block_mask = BlockDiagonalMask.from_seqlens(seq_lens)
            for layer in self.layers:
                x_flat = layer(x_flat, block_mask)
            return self.final_norm(x_flat)

        if phi is None:
            raise ValueError(
                "Encoder.forward(): `phi` is required when `window_size` is set."
            )

        W = self.window_size
        device = x_flat.device

        # ---- Sort by (event, phi) so position-along-axis = phi-rank ----
        seq_lens_t = torch.as_tensor(seq_lens, device=device, dtype=torch.long)
        batch_ids = torch.repeat_interleave(
            torch.arange(len(seq_lens), device=device, dtype=torch.long),
            seq_lens_t,
        )
        sort_keys = batch_ids.to(torch.float64) * 1e6 + phi.to(torch.float64)
        perm = torch.argsort(sort_keys, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=device)
        x_flat = x_flat[perm]

        # ---- Build cu_seqlens for flash-attn varlen ----
        cu_seqlens = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = seq_lens_t.to(torch.int32).cumsum(0)
        max_seqlen = int(seq_lens_t.max().item()) if seq_lens_t.numel() else 0
        attn_ctx = (cu_seqlens, max_seqlen)

        # ---- Per-event cycle-roll (only for odd layers if window_wrap) ----
        if self.window_wrap:
            roll_perm = _roll_perm_per_event(seq_lens, W // 2, device)
            inv_roll = torch.empty_like(roll_perm)
            inv_roll[roll_perm] = torch.arange(roll_perm.size(0), device=device)

        for i, layer in enumerate(self.layers):
            if self.window_wrap and (i % 2 == 1):
                x_flat = x_flat[roll_perm]
                x_flat = layer(x_flat, attn_ctx)
                x_flat = x_flat[inv_roll]
            else:
                x_flat = layer(x_flat, attn_ctx)

        x_flat = self.final_norm(x_flat)
        # Restore original (graph-node) order so downstream uses target's
        # batch_ids / local_idx as-is.
        return x_flat[inv_perm]
