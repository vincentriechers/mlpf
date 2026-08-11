"""Flash-attn varlen sliding-window attention dispatch for GATr.

GATr's geometric attention ultimately calls
``gatr.primitives.attention.scaled_dot_product_attention`` (the dispatcher
that routes to xformers ``memory_efficient_attention`` for AttentionBias
masks, and to ``torch.nn.functional.scaled_dot_product_attention`` otherwise).
xformers does NOT have a kernel for *bidirectional sliding-window attention
with per-event blocks*; the closest available bias is
``BlockDiagonalCausalLocalAttentionMask`` (causal only).

Flash-Attention's ``flash_attn_varlen_func`` does have it natively:
``window_size=(W//2, W//2)`` gives bidirectional sliding window, and
``cu_seqlens`` provides per-event separation. This is the recipe hepattn
uses (cf. ``hepattn/models/attention.py::_flash_varlen_attention``).

This module:
  - Defines :class:`FlashVarlenWindowMask`, a tiny mask object carrying
    ``cu_seqlens``, ``max_seqlen``, ``window_size``.
  - Defines :func:`install_window_dispatch`, a one-time runtime patch of
    ``gatr.primitives.attention._sdpa_graph_breaking``: when the attn_mask
    is a :class:`FlashVarlenWindowMask`, the patched function calls
    flash-attn varlen with the right window; otherwise it forwards to the
    original GATr SDPA dispatcher (xformers / torch SDPA).
  - Defines :func:`build_flash_varlen_window_mask` to construct the mask
    from per-event seq_lens.

We do NOT edit the on-disk ``gatr`` package. The patch is applied once at
import time and is a no-op for any caller that doesn't use this mask type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class FlashVarlenWindowMask:
    """Carries the bookkeeping for a flash-attn varlen sliding window call.

    Attributes
    ----------
    cu_seqlens
        ``(num_events + 1,)`` int32 tensor of cumulative event lengths,
        starting at 0. Standard flash-attn varlen format.
    max_seqlen
        Largest per-event seq length in the batch.
    window_size
        Total window size ``W``; tokens see ``W//2`` neighbours on each
        side in the (phi-)sorted order. Bidirectional.
    """

    cu_seqlens: Tensor
    max_seqlen: int
    window_size: int


def build_flash_varlen_window_mask(
    seq_lens: Tensor | list[int],
    window_size: int,
    device: Optional[torch.device] = None,
) -> FlashVarlenWindowMask:
    """Build the cu_seqlens/max_seqlen pair from per-event seq_lens.

    ``seq_lens`` may be a 1-D tensor or a Python list. The returned mask
    holds an ``int32`` tensor on ``device``.
    """
    if not isinstance(seq_lens, Tensor):
        seq_lens = torch.as_tensor(seq_lens, dtype=torch.long, device=device)
    elif device is not None and seq_lens.device != device:
        seq_lens = seq_lens.to(device)
    cu = torch.zeros(seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
    cu[1:] = seq_lens.to(torch.int32).cumsum(0)
    max_seqlen = int(seq_lens.max().item()) if seq_lens.numel() else 0
    return FlashVarlenWindowMask(
        cu_seqlens=cu, max_seqlen=max_seqlen, window_size=int(window_size)
    )


# --------------------------------------------------------------------------
# Flash-attn varlen path
# --------------------------------------------------------------------------

def _flash_varlen_sdpa(q: Tensor, k: Tensor, v: Tensor, mask: FlashVarlenWindowMask) -> Tensor:
    """Bidirectional sliding-window attention via flash-attn varlen.

    Inputs ``q``, ``k``, ``v`` come in with the shape GATr's SDPA dispatcher
    expects: ``[batch, heads, items, head_dim]``. We assume ``batch=1`` —
    GATr's ``Gatr_pf_e_noise``-style usage feeds events packed along the
    item dim with a per-event block mask, so the leading batch is always 1.
    Multi-batch packing would need a different cu_seqlens.
    """
    from flash_attn import flash_attn_varlen_func

    if q.ndim != 4:
        raise RuntimeError(
            f"FlashVarlenWindowMask: expected q with shape [B, H, N, D], got {tuple(q.shape)}"
        )
    B, H, N, D = q.shape
    if B != 1:
        raise RuntimeError(
            "FlashVarlenWindowMask requires batch=1 (events packed along item dim with cu_seqlens)."
        )

    # Reshape [1, H, N, D] -> [N, H, D] for varlen.
    q_f = q.transpose(1, 2).reshape(N, H, D).contiguous()
    k_f = k.transpose(1, 2).reshape(N, H, D).contiguous()
    v_f = v.transpose(1, 2).reshape(N, H, D).contiguous()

    cu = mask.cu_seqlens
    if cu.device != q.device:
        cu = cu.to(q.device)

    W = mask.window_size
    # window_size=(left, right) → bidirectional sliding window of total size W,
    # ±W//2 on each side. Matches hepattn (cf. set_backend in
    # ``hepattn/models/attention.py``).
    out = flash_attn_varlen_func(
        q_f, k_f, v_f,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=mask.max_seqlen, max_seqlen_k=mask.max_seqlen,
        dropout_p=0.0,
        causal=False,
        window_size=(W // 2, W // 2),
    )
    # Back to [1, H, N, D].
    return out.unsqueeze(0).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------
# Runtime patch
# --------------------------------------------------------------------------

_INSTALLED = False


def install_window_dispatch() -> None:
    """Install the flash-varlen window dispatch into GATr's attention path.

    Idempotent: subsequent calls are no-ops. Wraps
    ``gatr.primitives.attention._sdpa_graph_breaking`` so that when the
    ``attn_mask`` is a :class:`FlashVarlenWindowMask` we route to flash-attn
    varlen with sliding window; otherwise we delegate to the original
    function (xformers / torch SDPA, exactly as before).
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import gatr.primitives.attention as _gatr_prim_attn
    from gatr.utils.tensors import expand_pairwise

    original = _gatr_prim_attn._sdpa_graph_breaking

    @torch.compiler.disable
    def _patched(q, k, v, attn_mask):
        if isinstance(attn_mask, FlashVarlenWindowMask):
            # Same expand_pairwise the original applies, then flash-varlen
            # in place of SDPA. We don't propagate the mask further — flash
            # consumes cu_seqlens / window_size directly.
            q, k, v = expand_pairwise(q, k, v, exclude_dims=(-2,))
            v_out = _flash_varlen_sdpa(q, k, v, attn_mask)
            return q, k, v_out
        return original(q, k, v, attn_mask)

    _gatr_prim_attn._sdpa_graph_breaking = _patched
    _INSTALLED = True
