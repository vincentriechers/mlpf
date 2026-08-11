"""Bridge Mask3D outputs into the OC-style format expected by EnergyCorrection.

`EnergyCorrection.forward_correction` consumes a per-hit tensor
    x = [cluster_coord(3), beta(1)]    shape (N_hits_total, 4)
which is interpreted by `obtain_clustering_for_matched_showers` as
    coords = x[:, 0:3]   (clustered with DPC/DBSCAN)
    beta   = x[:, 3]     (passed through sigmoid for soft membership)

Mask3D produces per-(query, hit) mask logits + per-query embeddings; we
convert these to OC-shaped output by:
  - assigning each hit to its argmax query
  - using a learned 3-D projection of that query's embedding as the "coord"
    (so all hits in the same cluster share identical coords → DPC trivially
    recovers the cluster)
  - using sigmoid of the assigned mask logit as "beta"

This is a translation layer; the long-term plan is to add direct per-query
regression heads and skip the OC bridge entirely.
"""
import torch
import torch.nn as nn


class ECAdapter(nn.Module):
    def __init__(self, dim, coord_scale=10.0):
        super().__init__()
        self.coord_proj = nn.Linear(dim, 3)
        self.coord_scale = coord_scale

    def forward(self, mask_logits, queries, batch_ids, local_idx, key_valid):
        """
        mask_logits: (B, N_q, N_k)
        queries:     (B, N_q, D)
        batch_ids:   (N_hits_total,)
        local_idx:   (N_hits_total,)   index into the padded N_k axis
        key_valid:   (B, N_k)
        Returns x of shape (N_hits_total, 4) = [coord_xyz, beta_logit].
        """
        # Per-hit assigned query (argmax over queries of the mask logit)
        # mask_logits shape (B, N_q, N_k) → argmax along dim=1 gives (B, N_k)
        assigned = mask_logits.argmax(dim=1)             # (B, N_k)
        # Per-hit assigned mask logit
        gathered = mask_logits.gather(1, assigned.unsqueeze(1)).squeeze(1)  # (B, N_k)
        # Per-hit "cluster coord" = projected query embedding of the assigned query
        coord_per_query = self.coord_proj(queries) * self.coord_scale       # (B, N_q, 3)
        # gather (B, N_k, 3)
        idx = assigned.unsqueeze(-1).expand(-1, -1, 3)
        coord_dense = coord_per_query.gather(1, idx)                        # (B, N_k, 3)

        # Flatten back to per-hit format using (batch_ids, local_idx)
        coord_flat = coord_dense[batch_ids, local_idx]                      # (N_hits, 3)
        beta_flat = gathered[batch_ids, local_idx].view(-1, 1)              # (N_hits, 1)

        # Padded hits should never be queried (they're not in batch_ids/local_idx)
        x = torch.cat([coord_flat, beta_flat], dim=1)                       # (N_hits, 4)
        return x
