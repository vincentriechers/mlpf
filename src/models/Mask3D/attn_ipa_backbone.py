"""Plain-attention backbone for the IPA decoder (variant C).

Same idea as `GATrIPABackbone` but the geometric-algebra GATr blocks are
replaced by the standard Mask3D `InputNet` + `Encoder` pair. The encoder
operates on a flat token stream with BlockDiagonalMask (and optional
phi-windowed local attention, mirroring `mask3d_model.py`).

Returns the same two-tensor contract the IPA decoder expects:

  * `feats`  (N, dim)  — projected per-hit features, post-encoder.
  * `points` (N, 3)    — RAW hit position in the scaled-metres frame
                         (mm * `pos_scale`, default 0.001 → m). Same frame
                         as `build_targets`'s `feats_flat[:, :3]`, so the
                         IPA decoder's geometric term `T_i·t_i − x_j` and
                         the FAPE centroid loss both measure real Euclidean
                         distance to real detector hits.

Rationale: GATr's `extract_point` outputs a *learned* position, which can
drift away from the real hit geometry. Variant B already kept the encoder
geometric (GATr) but pinned the points to raw xyz; variant C goes further
and replaces the backbone entirely with a vanilla attention encoder, so
ALL geometric reasoning lives downstream in the IPA decoder. Useful as a
control to disentangle "what does GATr's cross-hit MV mixing buy us" from
"what does the IPA decoder buy us".
"""
import torch
import torch.nn as nn

from src.models.Mask3D.input_net import InputNet
from src.models.Mask3D.encoder import Encoder


class AttnIPABackbone(nn.Module):
    """Vanilla-attention backbone returning per-hit (feats, raw_xyz)."""

    def __init__(
        self,
        in_dim,
        dim=256,
        num_heads=8,
        num_layers=8,
        ffn_mult=4,
        dropout=0.0,
        hybrid_norm=False,
        window_size=1024,
        window_wrap=True,
        pos_scale=0.001,
    ):
        super().__init__()
        self.input_net = InputNet(in_dim=in_dim, dim=dim)
        self.encoder = Encoder(
            dim=dim, num_heads=num_heads, num_layers=num_layers,
            ffn_mult=ffn_mult, dropout=dropout,
            hybrid_norm=hybrid_norm,
            window_size=window_size, window_wrap=window_wrap,
        )
        self.window_size = window_size
        self.pos_scale = float(pos_scale)

    def forward(self, feats_flat, seq_lens):
        # feats_flat: (total_hits, in_dim). First 3 columns are the
        # scaled-metres (x, y, z). build_targets already multiplied raw mm
        # coordinates by `pos_scale`, so feats_flat[:, :3] is exactly the
        # frame we want to pass to the IPA decoder as `x_j`.
        x_flat = self.input_net(feats_flat)
        phi_flat = None
        if self.window_size:
            pos_xy = feats_flat[:, :2]
            phi_flat = torch.atan2(pos_xy[:, 1], pos_xy[:, 0])
        feats = self.encoder(x_flat, seq_lens, phi=phi_flat)        # (N, dim)
        points = feats_flat[:, :3].contiguous()                     # (N, 3)
        return feats, points
