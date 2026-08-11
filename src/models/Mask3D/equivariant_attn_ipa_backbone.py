"""Equivariant attention backbone for the IPA decoder.

Differs from `attn_ipa_backbone.AttnIPABackbone` in exactly one place:
the per-hit input embedding. Where the vanilla backbone has
`InputNet(feats_flat)` that consumes the absolute hit position as a
scalar feature (and applies a Fourier PE on `(x, y, z)`) — both of which
break equivariance under global rotation — this backbone uses
`EquivariantInputNet`, which:

  - drops the absolute `(x, y, z)` from the MLP input,
  - replaces the absolute-position Fourier PE with one on
    SO(2)_z-invariants `(r, |z|)` (or SO(3)-invariant `|x|`),
  - explicitly adds a `log(1+E)` channel so per-hit energy enters with a
    sensible dynamic range.

The rest of the pipe is unchanged:

  * Encoder is the same flat-token-stream transformer
    (BlockDiagonal or phi-windowed FlashVarlen).
  * The backbone returns `(feats, raw_xyz)` — same shape and same frame
    as the vanilla backbone, so the IPA decoder is plug-compatible.

End-to-end equivariance follows from:
  - This backbone's output `feats` is an SO(*)-invariant scalar tensor.
  - `raw_xyz` is the input position, passed through.
  - The IPA decoder takes `(feats, xyz)` and produces per-query `(T, t)`
    which transform equivariantly (T → R T, t → R t) under input rotation
    by R. Mask logits are invariant under R (they depend only on
    invariants of `(T, t, xyz)`).

This is the "lightweight equivariant" path — much cheaper than GATr and
nearly drop-in. For full equivariance ALSO inside the encoder's attention
pattern, drop the phi-windowed mode (since `phi = atan2(y, x)` is SO(2)_z
but not SO(3) invariant); the BlockDiagonal mode is symmetry-agnostic.
"""
import torch
import torch.nn as nn

from src.models.Mask3D.equivariant_input_net import (
    EquivariantInputNet,
    LocalGeometryHead,
)
from src.models.Mask3D.encoder import Encoder


class EquivariantAttnIPABackbone(nn.Module):
    """SO(*)-invariant-feature backbone returning per-hit (feats, raw_xyz)."""

    def __init__(
        self,
        in_dim,                              # kept for arg-compat (unused)
        dim=256,
        num_heads=8,
        num_layers=8,
        ffn_mult=4,
        dropout=0.0,
        hybrid_norm=False,
        window_size=1024,
        window_wrap=True,
        pos_scale=0.001,
        n_oh=None,                           # required: hit_type one-hot width
        symmetry="SO(2)_z",                  # "SO(2)_z" (CLD-natural) or "SO(3)"
        # Local-geometry head: enrich each token's input feature with a
        # small encoding of its k nearest neighbours' displacements,
        # expressed in a per-token cylindrical / PCA local frame
        # (SO(*)-invariant). This is the "look at displacement vectors and
        # such" component — gives the encoder geometric awareness without
        # writing absolute coordinates into any token's feature.
        local_geom=True,
        local_geom_k=64,
        local_geom_heads=4,
        local_geom_head_dim=24,
        local_geom_num_blocks=2,
        local_geom_ffn_mult=2,
    ):
        super().__init__()
        # `in_dim` is kept in the signature so existing call-sites that
        # pass `in_dim=3 + n_oh + 3` still work, but we derive `n_oh`
        # explicitly (or back it out from in_dim if not provided) — the
        # EquivariantInputNet's internal scalar layout is different.
        if n_oh is None:
            # vanilla layout was: pos(3) + n_oh + (E, p, chi2)  → in_dim = n_oh + 6
            n_oh = int(in_dim) - 6
            assert n_oh > 0, (
                f"could not infer n_oh from in_dim={in_dim} "
                "(expected in_dim = n_oh + 6); pass n_oh explicitly"
            )
        self.input_net = EquivariantInputNet(
            n_oh=n_oh, dim=dim, symmetry=symmetry,
        )
        self.local_geom = (
            LocalGeometryHead(
                dim=dim, k=local_geom_k,
                num_heads=local_geom_heads,
                head_dim=local_geom_head_dim,
                num_blocks=local_geom_num_blocks,
                ffn_mult=local_geom_ffn_mult,
                symmetry=symmetry,
            )
            if local_geom else None
        )
        self.encoder = Encoder(
            dim=dim, num_heads=num_heads, num_layers=num_layers,
            ffn_mult=ffn_mult, dropout=dropout,
            hybrid_norm=hybrid_norm,
            window_size=window_size, window_wrap=window_wrap,
        )
        self.window_size = window_size
        self.pos_scale = float(pos_scale)
        self.symmetry = symmetry

    def forward(self, feats_flat, seq_lens):
        # feats_flat layout matches the vanilla backbone:
        # (pos(3), hit_type_oh(n_oh), E, p, chi2). EquivariantInputNet
        # drops cols 0..2 from its scalar path and re-introduces them only
        # as SO(*)-invariants (r, |z|, etc.).
        x_flat = self.input_net(feats_flat)
        # Local-geometry stack of cross-attention blocks. Each block uses
        # the current token feature as query and per-neighbour local-frame
        # displacements as key/value. Internal residuals across blocks
        # propagate k-hop info along the k-NN graph. SO(*)-invariant by
        # construction (every input to every block is a scalar invariant).
        if self.local_geom is not None:
            x_flat = self.local_geom(
                feats_flat[:, :3], x_flat, seq_lens,
            )

        phi_flat = None
        if self.window_size:
            # The phi-windowed encoder uses `atan2(y, x)` to define the
            # 1-D ordering for the sliding window. This is SO(2)_z- and
            # SO(2)_z-reflection-invariant (modulo a global shift in phi
            # which the window-wrap mechanism already absorbs). It is NOT
            # SO(3)-invariant; for full SO(3) equivariance, disable the
            # phi-windowed path and use the BlockDiagonal encoder.
            pos_xy = feats_flat[:, :2]
            phi_flat = torch.atan2(pos_xy[:, 1], pos_xy[:, 0])

        feats = self.encoder(x_flat, seq_lens, phi=phi_flat)
        points = feats_flat[:, :3].contiguous()
        return feats, points
