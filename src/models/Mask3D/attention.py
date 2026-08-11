"""Attention building blocks.

Two flavours:

  - `BlockDiagonalSelfAttention`: operates on a *flat* `(total_hits, D)`
    token stream and uses xformers' `memory_efficient_attention` with a
    `BlockDiagonalMask` so events don't cross-attend. This is what the
    encoder uses — it never materializes an N_max² scores matrix and
    matches the pattern in `Gatr_pf_e_noise.py:146` (and hepattn's
    flash-varlen path).

  - `MultiHeadAttention`: operates on padded `(B, N, D)` tensors with an
    optional `(B, N_q, N_k)` boolean mask. Used for the decoder's
    cross-attn (queries × keys) and queries' self-attn — these are
    small (N_q is fixed, e.g. 320) so the padded path is cheap.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def _xformers():
    """Lazy xformers import — keeps the module loadable in envs without xformers."""
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha import BlockDiagonalMask
    return memory_efficient_attention, BlockDiagonalMask


def _flash_varlen():
    """Lazy flash-attn import — only required when window_size is set."""
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func


class FlashVarlenSelfAttention(nn.Module):
    """Sliding-window self-attention over a flat (total_hits, D) token stream.

    Uses flash-attn's varlen kernel with `window_size=(W//2, W//2)`:
    each token attends to itself + W//2 neighbors on each side, clamped to
    the event boundary (no cross-event leakage). Caller is responsible for
    sorting hits into the order it wants the window to follow (we phi-sort
    in `Encoder.forward`).

    `qkv_norm`: per-head LayerNorm on Q and K post-projection (HybridNorm).
    """

    def __init__(self, dim, num_heads, window_size, dropout=0.0, qkv_norm=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_left = window_size // 2
        self.window_right = window_size // 2
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qkv_norm = qkv_norm
        if qkv_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x_flat, cu_seqlens, max_seqlen):
        """`cu_seqlens`: int32 (B+1,) cumulative event lengths, on x_flat.device.
        `max_seqlen`: python int — longest event length in the batch."""
        flash_attn_varlen_func = _flash_varlen()
        N, D = x_flat.shape
        H, dh = self.num_heads, self.head_dim
        Q = self.q_proj(x_flat).view(N, H, dh)
        K = self.k_proj(x_flat).view(N, H, dh)
        V = self.v_proj(x_flat).view(N, H, dh)
        if self.qkv_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # flash-attn requires fp16 or bf16. Lightning's mixed-precision
        # wrapper usually casts inputs already; if a caller passes fp32 we
        # cast here and restore afterwards so the module is self-contained.
        orig_dtype = Q.dtype
        if orig_dtype == torch.float32:
            Q, K, V = Q.half(), K.half(), V.half()

        out = flash_attn_varlen_func(
            Q, K, V,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            window_size=(self.window_left, self.window_right),
        )                                              # (N, H, dh)
        out = out.to(orig_dtype).reshape(N, D)
        return self.out_proj(out)


class BlockDiagonalSelfAttention(nn.Module):
    """Self-attention on a flat (total_hits, D) token stream.

    Each event attends only to its own hits via xformers BlockDiagonalMask
    (events along the seq axis, no cross-event leakage, no padding waste).

    `qkv_norm`: when True, apply LayerNorm to Q and K after their projection
    and before the attention dot-product (a.k.a. "QK-norm", arXiv:2010.04245).
    Used by HybridNorm.
    """

    def __init__(self, dim, num_heads, dropout=0.0, qkv_norm=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qkv_norm = qkv_norm
        if qkv_norm:
            # LN per-head (applied along the head_dim) — matches hepattn.
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x_flat, block_mask):
        # x_flat: (total, D); block_mask: BlockDiagonalMask
        memory_efficient_attention, _ = _xformers()
        N, D = x_flat.shape
        H, dh = self.num_heads, self.head_dim
        Q = self.q_proj(x_flat).view(1, N, H, dh)
        K = self.k_proj(x_flat).view(1, N, H, dh)
        V = self.v_proj(x_flat).view(1, N, H, dh)
        if self.qkv_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)
        out = memory_efficient_attention(
            Q, K, V, attn_bias=block_mask,
            p=self.dropout if self.training else 0.0,
        )
        out = out.view(N, D)
        return self.out_proj(out)


class MultiHeadAttention(nn.Module):
    """Padded multi-head attention with optional key padding + per-pair attn mask.

    `qkv_norm`: when True, apply per-head LayerNorm to Q and K post-projection
    (QK-norm). Used by HybridNorm to normalise attention statistics inside the
    block when the surrounding pre-norm is removed.
    """

    def __init__(self, dim, num_heads, dropout=0.0, qkv_norm=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout
        self.qkv_norm = qkv_norm
        if qkv_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def _split_heads(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None,
                attn_bias=None):
        """`attn_bias`: optional (B, N_q, N_k) float tensor added to the
        scaled QK^T logits before softmax. Composes with the boolean
        `attn_mask` / `key_padding_mask`: disallowed positions stay at
        -inf, allowed positions get `bias` added. Used by the GMM-query
        path to inject a per-(query, point) spatial log-density bias.
        """
        B, Nq, D = q.shape
        Nk = k.size(1)
        Q = self._split_heads(self.q_proj(q))
        K = self._split_heads(self.k_proj(k))
        V = self._split_heads(self.v_proj(v))
        if self.qkv_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        if _HAS_SDPA:
            sdpa_mask = None
            if attn_bias is not None:
                # SDPA's `attn_mask` is interpreted as additive when it's
                # a float tensor — so we fold bool disallow → -inf and the
                # spatial bias into a single float (B, 1, N_q, N_k) tensor.
                bias = attn_bias.view(B, 1, Nq, Nk).to(Q.dtype)
                bool_disallow = None
                if attn_mask is not None:
                    bool_disallow = ~attn_mask.view(B, 1, Nq, Nk)
                if key_padding_mask is not None:
                    kpm = ~key_padding_mask.view(B, 1, 1, Nk).expand(
                        B, 1, Nq, Nk
                    )
                    bool_disallow = (
                        kpm if bool_disallow is None
                        else (bool_disallow | kpm)
                    )
                if bool_disallow is not None:
                    bias = bias.masked_fill(bool_disallow, float("-inf"))
                sdpa_mask = bias
            elif key_padding_mask is not None and attn_mask is None:
                sdpa_mask = key_padding_mask.view(B, 1, 1, Nk)
            elif attn_mask is not None:
                m = attn_mask.view(B, 1, Nq, Nk)
                if key_padding_mask is not None:
                    m = m & key_padding_mask.view(B, 1, 1, Nk)
                sdpa_mask = m
            out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=sdpa_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            out = out.transpose(1, 2).contiguous().view(B, Nq, D)
            return self.out_proj(out)

        # Fallback for torch < 2.0 (materializes scores; only for small N).
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.view(B, 1, 1, Nk).expand(B, 1, Nq, Nk)
        if attn_mask is not None:
            am = attn_mask.view(B, 1, Nq, Nk)
            mask = am if mask is None else (mask & am)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            scores = scores + attn_bias.view(B, 1, Nq, Nk).to(scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = scores.softmax(dim=-1)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)
