"""GATr backbone variant that exposes per-hit (scalar feature, 3-D point).

Forks `gatr_backbone.py::GATrBackbone`. The difference: instead of returning a
single flat `(N, dim)` feature tensor, we return two tensors per hit:

  * `feats`  (N, dim)  — projected scalar invariants + auxiliary scalar output,
                         used by the IPA decoder as the "scalar" channel.
  * `points` (N, 3)    — geometric 3-D point recovered from GATr's
                         multivector via `extract_point`. Used by the IPA
                         decoder as the per-hit position `x_j` in the
                         `T_i·t_i − x_j` local-frame distance term.

Everything else (input embedding, block-diagonal varlen attention, batch-norm
on raw xyz) matches the upstream backbone.
"""
import dgl
import torch
import torch.nn as nn

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, embed_scalar, extract_point
from xformers.ops.fmha import BlockDiagonalMask


class GATrIPABackbone(nn.Module):
    def __init__(
        self,
        dim,
        num_blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        n_input_scalars=2,
        use_input_norm=True,
        # When True (default), the returned `points` are GATr's output trivector
        # decoded via `extract_point` — a context-aware position the network
        # learned to put there. When False, we instead return the RAW hit
        # positions (in the same scaled-metres frame as `build_targets`'s
        # `pos_padded`, i.e. `g.ndata["pos_hits_xyz"] * 0.001`). Variant B:
        # keeps GATr's cross-hit attention for the scalar feature path while
        # pinning the geometric coordinate exactly to the detector hit, so
        # the IPA mask-logit `‖T·t − x‖²` and the FAPE centroid loss both
        # measure real Euclidean distance to real hits.
        use_extract_point=True,
        pos_scale=0.001,
    ):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=n_input_scalars,
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.input_norm = nn.BatchNorm1d(3, momentum=0.1)
        # MV (16) + scalar (1) → dim, same as the standard GATrBackbone.
        self.proj = nn.Linear(16 + 1, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.use_extract_point = bool(use_extract_point)
        self.pos_scale = float(pos_scale)

    @staticmethod
    def _block_mask(g):
        seqlens = dgl.unbatch(g) if not hasattr(g, "batch_num_nodes") else None
        if seqlens is not None:
            seqlens = [int(gi.number_of_nodes()) for gi in seqlens]
        else:
            seqlens = g.batch_num_nodes().tolist()
        return BlockDiagonalMask.from_seqlens(seqlens)

    def forward(self, g):
        xyz = g.ndata["pos_hits_xyz"].float()
        hit_type = g.ndata["hit_type"].float().view(-1, 1)
        if self.use_input_norm:
            xyz_norm = self.input_norm(xyz)
        else:
            xyz_norm = xyz
        emb_mv = embed_point(xyz_norm) + embed_scalar(hit_type)  # (N, 16)
        emb_mv = emb_mv.unsqueeze(-2)                            # (N, 1, 16)

        e = g.ndata["e_hits"].float().view(-1, 1)
        p = g.ndata["p_hits"].float().view(-1, 1)
        scalars = torch.cat([e, p], dim=1)                       # (N, n_input_scalars)

        attn_mask = self._block_mask(g)
        embedded_outputs, scalar_outputs = self.gatr(
            emb_mv, scalars=scalars, attention_mask=attn_mask,
        )                                                        # (N, 1, 16), (N, 1)

        # Scalar feature stream — full MV (16-dim) + the extra scalar channel,
        # projected to `dim`. Same as the standard backbone.
        mv_flat = embedded_outputs[:, 0, :]                      # (N, 16)
        feats = self.proj(torch.cat([mv_flat, scalar_outputs.view(-1, 1)], dim=-1))
        feats = self.out_norm(feats)                             # (N, dim)

        # Geometric point stream.
        # use_extract_point=True (default): trivector component of the GATr
        #   output. Context-aware position in the post-BN xyz frame.
        # use_extract_point=False (variant B): raw hit position, scaled by
        #   pos_scale so it matches build_targets'`feats_flat[:, :3]` frame
        #   (typically metres). Pristine geometry; the IPA mask-logit and
        #   the FAPE centroid loss then measure real Euclidean distance.
        if self.use_extract_point:
            points = extract_point(embedded_outputs[:, 0, :])    # (N, 3)
        else:
            points = xyz * self.pos_scale                        # (N, 3) — raw
        return feats, points
