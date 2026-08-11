"""Local fork-utilities for adding flash-attn varlen sliding-window attention to GATr.

Nothing in here modifies the upstream ``gatr`` package on disk; we install the
sliding-window dispatch via a one-time runtime patch of
``gatr.primitives.attention._sdpa_graph_breaking`` (see
``flash_varlen_window.install_window_dispatch``). The patch only changes
behaviour when the attention mask is a :class:`FlashVarlenWindowMask`; all
existing GATr callers that pass ``BlockDiagonalMask`` / ``None`` are
untouched.
"""
from .flash_varlen_window import (
    FlashVarlenWindowMask,
    install_window_dispatch,
    build_flash_varlen_window_mask,
)

__all__ = [
    "FlashVarlenWindowMask",
    "install_window_dispatch",
    "build_flash_varlen_window_mask",
]
