"""IPA-style decoder for the GATr-IPA model.

Per query we carry a triple (q_s, T_i, t_i):

* q_s ∈ R^D           scalar query feature.
* T_i ∈ SO(3)          rotation matrix (3x3), parameterised via the 6-D
                       Gram-Schmidt form of Zhou et al. (2019).
* t_i ∈ R^3            translation, in the world frame (the same frame
                       backbone `extract_point` returned x_j in).

Per decoder layer:

1. Mask logit for each (query i, hit j):
       logit_ij = (q_s_i · proj_k(f_j)) / sqrt(d)
                   − γ_h · ‖t_i − x_j‖²
   where γ_h is a learnable per-head scalar (softplus-positive). This is the
   AlphaFold IPA "single point per query" simplification of Algorithm 22 line 7.
2. Cross-attention from q_s to the hit features, gated by the
   Mask2Former binary mask `sigmoid(prev_mask_logits) ≥ threshold`. The
   attention also picks up an IPA "value point" term — Σ_j a_ij ·
   T_iᵀ (x_j − t_i) — concatenated alongside the scalar value and projected
   back into the query's update.
3. FFN on q_s.
4. Frame update head: an MLP on the refreshed q_s predicts (Δ6D, Δt). Zero-
   initialised → starts as identity (no update at step 0).

Each layer emits `mask_logits` and `cls_logits` for the matcher / loss. The
loss expectations match `mask3d_loss`'s `per_layer_outs[*]` schema, so the
existing loss pipeline runs unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Mask3D.attention import MultiHeadAttention, FFN


# ---------------------------------------------------------------------------
# Rotation helpers (Gram-Schmidt 6-D ↔ 3×3) and frame composition.
# ---------------------------------------------------------------------------


_IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. (2019) 6-D → 3×3 rotation.

    Input shape (..., 6), output (..., 3, 3). The first three numbers define
    the first column basis vector; the second three are projected orthogonal
    to it; the third column is the cross product. Differentiable and
    well-defined except on a measure-zero set (a1 parallel to a2).
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_proj, dim=-1, eps=1e-8)
    b3 = torch.linalg.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                            # columns


def compose_frames(T_old, t_old, dT, dt):
    """Compose a frame update.

    T_new = T_old @ dT
    t_new = t_old + T_old @ dt        (dt expressed in the OLD local frame)

    Shapes (B, N_q, 3, 3) and (B, N_q, 3); broadcastable.
    """
    T_new = torch.matmul(T_old, dT)
    t_new = t_old + torch.einsum("bnij,bnj->bni", T_old, dt)
    return T_new, t_new


# ---------------------------------------------------------------------------
# IPA decoder layer.
# ---------------------------------------------------------------------------


class IPADecoderLayer(nn.Module):
    """One IPA-style decoder block.

    forward args:
        q_s        (B, N_q, D)  current query scalar features
        T          (B, N_q, 3, 3) current per-query rotation
        t          (B, N_q, 3) current per-query translation
        keys       (B, N_k, D)  per-hit scalar features (from backbone)
        x          (B, N_k, 3)  per-hit positions (from backbone extract_point)
        key_valid  (B, N_k) bool — True where the slot is a real hit
        attn_mask  optional (B, N_q, N_k) bool from Mask2Former-style gating
    """

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0,
                 # k-Max-DeepLab variant for the q←k cross-attn pass.
                 # "softmax" (default) = standard SDPA with IPA geometric
                 # bias; "kmeans" = hard-assignment M-step using the
                 # previous-layer mask logits as the (B,N_q,N_k) assignment.
                 # IPA point-value & frame-update machinery unchanged.
                 cross_attn_mode="softmax",
                 kmeans_value_proj=False,
                 kmeans_respect_attn_mask=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.cross_attn_mode = str(cross_attn_mode)

        # In kmeans mode the layer's qn is consumed by KMeansCrossAttention
        # only when `logits` isn't supplied (Q·K fallback). We always pass
        # the previous-layer mask logits, so qn is never read → norm_q1's
        # LayerNorm params would receive no gradient and DDP would crash.
        # Use Identity in that case (no parameters).
        self.norm_q1 = (nn.Identity() if self.cross_attn_mode == "kmeans"
                        else nn.LayerNorm(dim))
        self.norm_k1 = nn.LayerNorm(dim)
        if self.cross_attn_mode == "kmeans":
            from src.models.Mask3D.kmeans_ca import KMeansCrossAttention
            self.cross_attn = KMeansCrossAttention(
                dim, update="mean", value_proj=bool(kmeans_value_proj),
                respect_attn_mask=bool(kmeans_respect_attn_mask),
            )
        elif self.cross_attn_mode == "softmax":
            self.cross_attn = MultiHeadAttention(
                dim, num_heads, dropout=dropout, qkv_norm=False,
            )
        else:
            raise ValueError(
                f"cross_attn_mode must be 'softmax' or 'kmeans', "
                f"got {cross_attn_mode!r}"
            )
        # IPA "point-value" projection: from the per-hit position in the
        # query's local frame back into the query's feature update.
        # We pack [scalar_attn_out, mean_point_local, mean_point_norm] →
        # 3-channel input → linear back to D. Mean_point_local is 3D so this
        # is D + 3 + 1 = D + 4 input.
        self.point_value_proj = nn.Linear(dim + 4, dim)

        self.norm_q2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, mult=ffn_mult, dropout=dropout)

        self.norm_q3 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(
            dim, num_heads, dropout=dropout, qkv_norm=False,
        )

        # Learnable per-head distance weight γ_h. Softplus → positive.
        # We average across heads to produce one scalar weight per query/key
        # pair on the mask logit. This is a single scalar in our simplified
        # IPA (one point per query, all heads share the geometric term).
        self.gamma_pre = nn.Parameter(torch.zeros(num_heads))

        # Frame update head: q_s → (Δ6D rotation, Δt). Zero-init so frames
        # start as identity. ΔT is composed onto the right of T.
        self.frame_update = nn.Linear(dim, 6 + 3)
        nn.init.zeros_(self.frame_update.weight)
        nn.init.zeros_(self.frame_update.bias)
        # Bias on Δ6D so it decodes to the identity rotation at zero (and
        # zero weights mean the prediction is identity at step 0).
        with torch.no_grad():
            self.frame_update.bias[:6].copy_(_IDENTITY_6D)
        # (We then *subtract* the identity 6D before applying the GS map
        # at the call site so the residual ΔT starts at the identity rotation.)

    def forward(self, q_s, T, t, keys, x, key_valid, attn_mask=None,
                kmeans_logits=None):
        B, N_q, D = q_s.shape
        N_k = keys.size(1)

        # ----- 1. Distance term shared by mask logit & cross-attn bias -----
        # `t` is the query's reference point in the WORLD frame (see module
        # docstring, `compose_frames`, the seeding path, and `loss.py:838`).
        # World-frame distance is `‖t − x‖²` — applying T to t (the old
        # `T·t − x`) was a long-standing bug that conflicted with that
        # convention and silently mis-steered every cross-attention bias.
        delta = t[:, :, None, :] - x[:, None, :, :]               # (B, N_q, N_k, 3)
        dist2 = (delta * delta).sum(-1)                            # (B, N_q, N_k)
        gamma = F.softplus(self.gamma_pre).mean()                  # scalar
        # Negative — closer points get a higher (less negative) bias.
        attn_bias = -gamma * dist2                                 # (B, N_q, N_k)

        # ----- 2. Cross-attention (q_s ← attend over hits) -----------------
        qn = self.norm_q1(q_s)
        kn = self.norm_k1(keys)
        if self.cross_attn_mode == "kmeans":
            # k-Max-DeepLab style: hard argmax assignment of each key to its
            # best query (using the previous-layer mask logits). The IPA
            # point-value + frame-update machinery below is unchanged.
            cross_out = self.cross_attn(
                qn, k=kn, v=kn,
                kv_mask=key_valid,
                attn_mask=attn_mask,
                logits=kmeans_logits,
            )
        else:
            # Use the geometric distance as an additive attn_bias (closer
            # points naturally win attention even with no mask). Combined
            # with the optional Mask2Former `attn_mask` (binary), this mimics
            # the AlphaFold IPA logit form.
            cross_out = self.cross_attn(
                qn, kn, kn,
                key_padding_mask=key_valid,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        # IPA point-value: each query gets, per (query, hit), the point j in
        # query i's *local* frame = T_iᵀ (x_j − t_i). Aggregate by the attn
        # weights — but `cross_out` is the standard attention output and the
        # attn weights themselves aren't returned by SDPA. Use a soft
        # softmax-of-(attn_bias + mask) approximation: this is cheap and
        # captures the dominant geometric term.
        if attn_mask is not None:
            mask_float = torch.zeros_like(attn_bias).masked_fill(
                ~attn_mask, float("-inf"),
            )
            ipa_logits = (attn_bias + mask_float).masked_fill(
                ~key_valid.unsqueeze(1), float("-inf"),
            )
        else:
            ipa_logits = attn_bias.masked_fill(
                ~key_valid.unsqueeze(1), float("-inf"),
            )
        # If every key is masked for a query, softmax produces NaN. Fall back
        # to uniform attention in that row.
        all_masked = torch.isinf(ipa_logits).all(dim=-1, keepdim=True)
        ipa_logits = torch.where(all_masked, torch.zeros_like(ipa_logits), ipa_logits)
        ipa_w = ipa_logits.softmax(dim=-1)                         # (B, N_q, N_k)
        # Point j in query i's local frame.
        rel_local = torch.einsum(
            "bnij,bnkj->bnki", T.transpose(-1, -2), -delta,
        )                                                          # = T_iᵀ (x_j − t_i)
        ipa_point = torch.einsum("bnk,bnki->bni", ipa_w, rel_local)   # (B, N_q, 3)
        ipa_norm = ipa_point.norm(dim=-1, keepdim=True)              # (B, N_q, 1)

        # Combine scalar attn output with the IPA point output.
        combined = torch.cat([cross_out, ipa_point, ipa_norm], dim=-1)
        q_s = q_s + self.point_value_proj(combined)

        # ----- 3. FFN -----------------------------------------------------
        q_s = q_s + self.ffn(self.norm_q2(q_s))

        # ----- 4. Self-attention among queries ----------------------------
        qn = self.norm_q3(q_s)
        q_s = q_s + self.self_attn(qn, qn, qn)

        # ----- 5. Frame update --------------------------------------------
        upd = self.frame_update(q_s)                                # (B, N_q, 9)
        d6 = upd[..., :6]
        dt = upd[..., 6:9]
        dT = rotation_6d_to_matrix(d6)                              # (B, N_q, 3, 3)
        T_new, t_new = compose_frames(T, t, dT, dt)
        return q_s, T_new, t_new


# ---------------------------------------------------------------------------
# Full IPA decoder.
# ---------------------------------------------------------------------------


class IPADecoder(nn.Module):
    """Stack of `IPADecoderLayer`s with mask + cls heads at every layer.

    `forward` returns a list of `{mask_logits, cls_logits}` dicts (one per
    layer + the layer-0 initial-query heads), matching the per-layer schema
    `mask3d_loss` already consumes.
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_queries,
        num_layers,
        ffn_mult=4,
        dropout=0.0,
        mask_threshold=0.5,
        num_classes=1,
        share_decoder_heads=True,
        # When True, the mask logit's geometric term is the output of a small
        # MLP on the LOCAL-FRAME relative coordinate `T_iᵀ(x_j − t_i)` and
        # its norm — direction-aware (the mask can prefer "+x of my anchor"
        # vs "−x"). When False (default, backward-compatible), the geometric
        # term is just `−γ · ‖t_i − x_j‖²` (a single scalar per pair).
        mask_direction_aware=False,
        mask_pos_mlp_hidden=16,
        # When True, the mask-logit feature-similarity branch also reads the
        # absolute hit position: `kk = mask_proj_k(concat(f_j, x_j))`. Useful
        # when the backbone has no intrinsic geometric processing (variant C
        # = plain attention encoder), so the mask head's feature path is also
        # coordinate-aware in addition to the (already-present) geometric
        # branch via `T_iᵀ(x_j − t_i)`. Default False for backward compat.
        coord_augmented_keys=False,
        # Initial-query strategy (mirrors MaskFormerDecoder).
        #   "static"  — broadcast learned `q_init` (D) and `t_init` (3) across
        #               events; no event conditioning at layer 0. Original
        #               IPADecoder behaviour.
        #   "encoder" — pick the top-`num_queries` encoder tokens per event
        #               by L2 norm (with optional `track_seed_bonus` on
        #               tracker hits). Initial `q_s` = seed_feature +
        #               `q_init` (per-slot residual), initial `t` =
        #               seed_xyz + `t_init` (per-slot residual). Event-
        #               conditioned from layer 0 → fixes the "layer-0 mask
        #               is hopeless because queries are identical across
        #               events" failure mode that pollutes the per-layer
        #               loss mean reported in wandb.
        # Default "encoder" so new runs match the vanilla-Mask3D recipe.
        dynamic_query_source="encoder",
        track_seed_bonus=6e4,
        track_hit_type=1,
        # k-Max-DeepLab style cross-attention. "softmax" = standard SDPA
        # with IPA geometric bias (default). "kmeans" = hard-assignment
        # M-step using the previous-layer mask logits as the assignment
        # matrix; IPA point-value + frame-update machinery unchanged.
        cross_attn_mode="softmax",
        kmeans_value_proj=False,
        kmeans_respect_attn_mask=False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.mask_threshold = mask_threshold

        # Per-query parameters. With dynamic_query_source="encoder", `q_init`
        # and `t_init` are added as RESIDUALS on top of the per-event encoder-
        # picked seed (feature, xyz); with "static" they ARE the broadcast
        # queries (no seeding). Init scale ~0.02 / 0 keeps the seed dominant
        # at step 0 either way.
        self.q_init = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.t_init = nn.Parameter(torch.zeros(num_queries, 3))
        self.dynamic_query_source = str(dynamic_query_source)
        self.track_seed_bonus = float(track_seed_bonus)
        self.track_hit_type = int(track_hit_type)

        self.cross_attn_mode = str(cross_attn_mode)
        self.layers = nn.ModuleList([
            IPADecoderLayer(
                dim, num_heads, ffn_mult=ffn_mult, dropout=dropout,
                cross_attn_mode=self.cross_attn_mode,
                kmeans_value_proj=bool(kmeans_value_proj),
                kmeans_respect_attn_mask=bool(kmeans_respect_attn_mask),
            )
            for _ in range(num_layers)
        ])

        n_slots = 1 if share_decoder_heads else (num_layers + 1)
        # Mask head: per-(query, key) logit. We do the standard MaskFormer
        # dot product on the feature side; the geometric term is added inside
        # the layer via `attn_bias`. For the per-layer mask logit we emit
        # the full IPA-style logit so the matcher / loss see the same form
        # that drives the cross-attention.
        self.coord_augmented_keys = bool(coord_augmented_keys)
        proj_k_in = dim + (3 if self.coord_augmented_keys else 0)
        self.mask_proj_q = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_slots)]
        )
        self.mask_proj_k = nn.ModuleList(
            [nn.Linear(proj_k_in, dim, bias=False) for _ in range(n_slots)]
        )
        # Mask geometric term — two mutually exclusive parameterisations.
        # We only create the one that's actually used so DDP doesn't
        # complain about unused params on the active code path.
        self.mask_direction_aware = bool(mask_direction_aware)
        if self.mask_direction_aware:
            # Small per-slot MLP on the local-frame relative coordinate.
            # Zero-init final layer → at step 0 the geometric bias is 0
            # and the mask logit reduces to the feature dot product.
            self.mask_pos_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(4, mask_pos_mlp_hidden),
                    nn.GELU(),
                    nn.Linear(mask_pos_mlp_hidden, 1),
                )
                for _ in range(n_slots)
            ])
            for mlp in self.mask_pos_mlp:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
        else:
            # Scalar squared-distance weight γ ≥ 0 (per-slot softplus-positive).
            self.mask_gamma_pre = nn.Parameter(torch.zeros(n_slots))
        # Classification head (objectness).
        self.cls_heads = nn.ModuleList(
            [nn.Linear(dim, num_classes) for _ in range(n_slots)]
        )
        self.share_decoder_heads = share_decoder_heads

    def _mask_and_cls(self, q_s, T, t, keys, x, key_valid, slot):
        idx = 0 if self.share_decoder_heads else slot
        # Feature similarity (B, N_q, N_k).
        qk = self.mask_proj_q[idx](q_s)
        if self.coord_augmented_keys:
            # Concatenate absolute hit position to the key features so the
            # mask head's feature dot product is itself coordinate-aware.
            kk = self.mask_proj_k[idx](torch.cat([keys, x], dim=-1))
        else:
            kk = self.mask_proj_k[idx](keys)
        scale = 1.0 / math.sqrt(qk.size(-1))
        feature_sim = torch.einsum("bnd,bkd->bnk", qk, kk) * scale
        # Geometric term.
        if self.mask_direction_aware:
            # Hit position in the query's LOCAL frame: rel = T_iᵀ (x_j − t_i).
            # (B, 1, N_k, 3) − (B, N_q, 1, 3) → (B, N_q, N_k, 3); then rotate
            # into the local frame with T_iᵀ.
            delta_world = x[:, None, :, :] - t[:, :, None, :]               # (B, N_q, N_k, 3)
            rel = torch.einsum(
                "bnij,bnkj->bnki", T.transpose(-1, -2), delta_world,
            )                                                               # (B, N_q, N_k, 3)
            rel_norm = rel.norm(dim=-1, keepdim=True)                       # (B, N_q, N_k, 1)
            geom_bias = self.mask_pos_mlp[idx](
                torch.cat([rel, rel_norm], dim=-1)
            ).squeeze(-1)                                                   # (B, N_q, N_k)
            mask_logits = feature_sim + geom_bias
        else:
            # World-frame distance ‖t − x‖² (was ‖T·t − x‖² — same bug as in
            # IPADecoderLayer.forward).
            delta = t[:, :, None, :] - x[:, None, :, :]
            dist2 = (delta * delta).sum(-1)
            gamma = F.softplus(self.mask_gamma_pre[idx])
            mask_logits = feature_sim - gamma * dist2
        # Mask out padded keys (so the matcher / loss never sees them).
        mask_logits = mask_logits.masked_fill(~key_valid.unsqueeze(1), -1e4)
        cls_logits = self.cls_heads[idx](q_s)                               # (B, N_q, 1)
        return mask_logits, cls_logits

    def _seed_queries_from_encoder(self, keys, x, key_valid, hit_subsystem=None):
        """Pick top-`num_queries` encoder tokens per event by L2 norm.

        Returns:
            q_seed (B, N_q, D) — gathered encoder features (seeds)
            t_seed (B, N_q, 3) — gathered per-hit positions (seeds)
            seed_idx (B, N_q) — diagnostic
        Padded-slot seeds are zeroed out so the per-slot learned residual
        (`q_init`, `t_init`) is the only signal for events smaller than N_q.
        """
        B, N_k, D = keys.shape
        # fp32 scoring so an AMP run with a large `track_seed_bonus` stays
        # numerically stable (the bonus must dwarf any encoder L2 norm).
        scores = keys.float().norm(dim=-1)                                # (B, N_k)
        if hit_subsystem is not None and self.track_seed_bonus != 0.0:
            track_mask = hit_subsystem == self.track_hit_type
            scores = scores + track_mask.float() * self.track_seed_bonus
        scores = scores.masked_fill(~key_valid, float("-inf"))
        # `topk` raises "selected index k out of range" when the PADDED key
        # width is smaller than num_queries, which is decided by the largest
        # event in the batch — so this never fires during training (batch 8+ of
        # ~531-hit DELPHI events gives N_k ~ 700) but does at eval, where the
        # reference configuration is --batch-size 1 and any event with fewer
        # than num_queries=320 hits crashes it. Clamp k and zero-pad the missing
        # slots, which is exactly the semantics this function already documents:
        # a padded slot gets a zero seed, leaving the learned per-slot residual
        # (`q_init` / `t_init`) as the only signal. When N_k >= num_queries this
        # is bit-identical to the previous code.
        k = min(int(self.num_queries), int(N_k))
        seed_idx = scores.topk(k, dim=1).indices                          # (B, k)
        idx_d = seed_idx.unsqueeze(-1).expand(-1, -1, D)
        idx_3 = seed_idx.unsqueeze(-1).expand(-1, -1, 3)
        q_seed = keys.gather(1, idx_d)                                    # (B, k, D)
        t_seed = x.gather(1, idx_3)                                       # (B, k, 3)
        seed_valid = key_valid.gather(1, seed_idx)                        # (B, k)
        q_seed = q_seed * seed_valid.unsqueeze(-1).to(q_seed.dtype)
        t_seed = t_seed * seed_valid.unsqueeze(-1).to(t_seed.dtype)
        if k < self.num_queries:
            pad = self.num_queries - k
            q_seed = F.pad(q_seed, (0, 0, 0, pad))                        # (B, N_q, D)
            t_seed = F.pad(t_seed, (0, 0, 0, pad))                        # (B, N_q, 3)
            seed_idx = F.pad(seed_idx, (0, pad))                          # diagnostic only
        return q_seed, t_seed, seed_idx

    def forward(self, keys, x, key_valid, hit_subsystem=None,
                disable_mask_attention=False):
        B = keys.size(0)
        device = keys.device
        # Initial queries.
        if self.dynamic_query_source == "encoder":
            q_seed, t_seed, _ = self._seed_queries_from_encoder(
                keys, x, key_valid, hit_subsystem=hit_subsystem,
            )
            # Per-slot residual on top of the event-specific seed.
            q_s = q_seed + self.q_init.unsqueeze(0)
            t = t_seed + self.t_init.unsqueeze(0)
        else:
            q_s = self.q_init.unsqueeze(0).expand(B, -1, -1).contiguous()
            t = self.t_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        T = torch.eye(3, device=device).expand(B, self.num_queries, 3, 3).contiguous()

        per_layer = []
        # Layer-0 heads (on the initial queries / identity frame).
        ml, cl = self._mask_and_cls(q_s, T, t, keys, x, key_valid, slot=0)
        per_layer.append({"mask_logits": ml, "cls_logits": cl})

        for li, layer in enumerate(self.layers):
            if disable_mask_attention:
                attn_mask = None
            else:
                # Mask2Former-style: gate next-layer attention on previous mask.
                attn_mask = (ml.sigmoid() >= self.mask_threshold).detach()
                attn_mask = attn_mask & key_valid.unsqueeze(1)
                all_false = ~attn_mask.any(dim=-1, keepdim=True)
                attn_mask = attn_mask | all_false
            # In kmeans mode, pass the just-computed mask logits as the
            # k-means assignment matrix — same quantity that drives `attn_mask`.
            kmeans_logits = ml if self.cross_attn_mode == "kmeans" else None
            q_s, T, t = layer(q_s, T, t, keys, x, key_valid,
                              attn_mask=attn_mask, kmeans_logits=kmeans_logits)
            ml, cl = self._mask_and_cls(
                q_s, T, t, keys, x, key_valid, slot=(li + 1),
            )
            per_layer.append({"mask_logits": ml, "cls_logits": cl})

        return per_layer, q_s, T, t
