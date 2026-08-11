"""GATr-based backbone for the Mask3D decoder.

Drop-in replacement for `InputNet + Encoder` that produces per-hit
embeddings of shape `(total_hits, dim)`. Mirrors how
`src/models/GATr/Gatr_pf_e_noise.py` wires up GATr — same input
embedding (`embed_point + embed_scalar`), same `BlockDiagonalMask` for
varlen attention, same scalar-channel layout (`e_hits, p_hits`).

Differences vs. that model:

  - We don't produce OC-style `(coords, beta)` outputs — we project the
    GATr output (the multivector channel + scalar output) into the same
    `dim`-D space the transformer encoder would produce, so the
    downstream `MaskFormerDecoder` can consume keys exactly as before.
  - No clustering / beta head; everything after the backbone is shared
    with the transformer-backbone variant.

Pros vs. the transformer encoder:
  - E(3)-equivariant: rotations / translations of input → same
    rotated / translated geometric features. Detector physics is
    rotationally symmetric (around beam axis), so this is a real prior.
  - Multivector representation captures higher-order geometric structure
    (e.g. plane bivectors) that scalar transformers can't directly
    encode.

Cons:
  - More parameters and FLOPs per token than a plain transformer.
  - Requires GATr installed (`from gatr import ...`).
"""
import dgl
import torch
import torch.nn as nn

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, embed_scalar
from xformers.ops.fmha import BlockDiagonalMask


class GATrBackbone(nn.Module):
    """Per-hit GATr feature extractor.

    Args:
        dim:               output channel count (matches decoder's `dim`).
        num_blocks:        number of GATr blocks.
        hidden_mv_channels: GATr hidden multivector channels.
        hidden_s_channels:  GATr hidden scalar channels.
        n_input_scalars:    number of per-hit scalar inputs (default 2:
                            `e_hits`, `p_hits`). Mirrors `Gatr_pf_e_noise`.
        use_input_norm:     BatchNorm on raw `pos_hits_xyz` before
                            embedding (matches existing GATr setup).
    """

    def __init__(
        self,
        dim,
        num_blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        n_input_scalars=2,
        use_input_norm=True,
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
            # Same as Gatr_pf_e_noise.ScaledGooeyBatchNorm2_1: BN on raw xyz.
            # Keeps pre-GATr position scale stable across detector regions.
            self.input_norm = nn.BatchNorm1d(3, momentum=0.1)

        # Project GATr output to the decoder's `dim`. The multivector
        # output has 16 algebra components per channel (pga.GA_3 has dim 16);
        # we use the single output channel + the single scalar output → 17.
        self.proj = nn.Linear(16 + 1, dim)
        self.out_norm = nn.LayerNorm(dim)

    @staticmethod
    def _block_mask(g):
        # Per-event hit counts → BlockDiagonalMask. Equivalent to the
        # `obtain_batch_numbers + bincount` pattern in `Gatr_pf_e_noise.py:145`,
        # but reads counts directly from the batched DGL graph (one fewer
        # GPU→CPU sync in the bincount path).
        seqlens = dgl.unbatch(g) if not hasattr(g, "batch_num_nodes") else None
        if seqlens is not None:
            seqlens = [int(gi.number_of_nodes()) for gi in seqlens]
        else:
            seqlens = g.batch_num_nodes().tolist()
        return BlockDiagonalMask.from_seqlens(seqlens)

    def forward(self, g):
        # Position + categorical hit-type → multivector channel.
        xyz = g.ndata["pos_hits_xyz"].float()
        hit_type = g.ndata["hit_type"].float().view(-1, 1)
        if self.use_input_norm:
            xyz = self.input_norm(xyz)
        emb_mv = embed_point(xyz) + embed_scalar(hit_type)  # (N, 16)
        emb_mv = emb_mv.unsqueeze(-2)                       # (N, 1, 16)

        # Scalar-channel inputs (energy, |p|).
        e = g.ndata["e_hits"].float().view(-1, 1)
        p = g.ndata["p_hits"].float().view(-1, 1)
        scalars = torch.cat([e, p], dim=1)                  # (N, n_input_scalars)

        # Block-diagonal varlen attention so events stay separate.
        attn_mask = self._block_mask(g)

        embedded_outputs, scalar_outputs = self.gatr(
            emb_mv, scalars=scalars, attention_mask=attn_mask,
        )                                                   # (N, 1, 16), (N, 1)

        # Concatenate the algebra components + the scalar output → (N, 17),
        # then project to `dim`. We don't apply `extract_point` here because
        # we're not using GATr as a coord predictor; the full algebra output
        # (incl. scalars / vectors / bivectors / pseudoscalars) is richer
        # for a downstream non-equivariant decoder.
        mv_flat = embedded_outputs[:, 0, :]                 # (N, 16)
        feats = torch.cat([mv_flat, scalar_outputs.view(-1, 1)], dim=-1)
        feats = self.proj(feats)
        return self.out_norm(feats)
