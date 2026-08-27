"""MaskFormer-style decoder with iteratively-refined mask attention.

This is the heart of the Mask3D adaptation. Each layer:
  1. Run task heads on the current queries -> mask logits + class logits
  2. attn_mask = (sigmoid(mask_logits) >= 0.5)             (Mask2Former trick)
     unmask_all_false: queries with empty masks are forced to attend
     everywhere, otherwise the softmax over zero allowed keys NaNs out.
  3. Cross-attend queries -> keys, restricted to attn_mask.
  4. FFN on queries.
  5. Self-attn among queries.

Outputs at every layer are kept for deep supervision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Mask3D.attention import MultiHeadAttention, FFN
from src.models.Mask3D.tasks import ObjectHitMaskTask, ObjectClassificationTask
from src.models.Mask3D.gmm_query import (
    GaussianQueryMixture,
    DynamicGaussianQueryMixture,
    QueryAnchorHead,
)


def unmask_all_false(attn_mask):
    """If a query has zero allowed keys, allow all of them (avoids NaN softmax)."""
    if attn_mask is None:
        return None
    all_false = ~attn_mask.any(dim=-1, keepdim=True)     # (B, N_q, 1)
    return attn_mask | all_false


class MaskFormerDecoderLayer(nn.Module):
    """Mask-attention decoder block, optionally HybridNorm-wrapped.

    HybridNorm (`hybrid_norm=True`):
      - layer 0:  pre-norm before each attn block + pre-norm before FFN.
      - layer ≥1: NO pre-norm before attn (qkv-norm inside conditions Q/K)
                  + post-norm style on each FFN.
    `bidirectional_ca` adds a key→query CA + FFN at the end of the block;
    HybridNorm rules apply to that sub-block too.
    """

    def __init__(self, dim, num_heads, ffn_mult=4, dropout=0.0,
                 bidirectional_ca=False, depth=0, hybrid_norm=False,
                 # k-Max-DeepLab variant: "softmax" = standard MaskFormer
                 # cross-attn (default); "kmeans" = hard-assignment k-means
                 # M-step cross-attn (see kmeans_ca.py). Uses the previous
                 # layer's mask logits as the assignment matrix.
                 cross_attn_mode="softmax",
                 # Whether to also enforce the binary attn_mask
                 # (sigmoid(prev_ml)>=0.5) on top of the argmax. Default off:
                 # the mask logits already drive the assignment.
                 kmeans_respect_attn_mask=False,
                 # Learnable linear V-projection on the kmeans value path.
                 # False = aggregate raw key features (hepattn default);
                 # True = apply nn.Linear(dim,dim) to values first.
                 kmeans_value_proj=False):
        super().__init__()
        is_hybrid_subsequent = hybrid_norm and depth > 0
        self.use_pre_norm_attn  = not is_hybrid_subsequent
        self.use_post_norm_dense = is_hybrid_subsequent
        qkv_norm = hybrid_norm

        ln = lambda: nn.LayerNorm(dim) if self.use_pre_norm_attn else nn.Identity()

        self.cross_attn_mode = str(cross_attn_mode)
        # `norm_q1` is only meaningful when the cross-attn consumes q via
        # Q·K (softmax mode). In kmeans mode the assignment uses the
        # passed-in mask logits and the q argument is never read, so its
        # LayerNorm would have no caller → DDP "unused parameters" crash.
        # Make it parameter-free in that case.
        self.norm_q1 = nn.Identity() if self.cross_attn_mode == "kmeans" else ln()
        self.norm_k1 = ln()
        if self.cross_attn_mode == "kmeans":
            from src.models.Mask3D.kmeans_ca import KMeansCrossAttention
            self.cross_attn = KMeansCrossAttention(
                dim, update="mean", value_proj=bool(kmeans_value_proj),
                respect_attn_mask=bool(kmeans_respect_attn_mask),
            )
        elif self.cross_attn_mode == "softmax":
            self.cross_attn = MultiHeadAttention(
                dim, num_heads, dropout=dropout, qkv_norm=qkv_norm
            )
        else:
            raise ValueError(
                f"cross_attn_mode must be 'softmax' or 'kmeans', "
                f"got {cross_attn_mode!r}"
            )
        self.norm_q2 = nn.LayerNorm(dim)        # FFN norm, used either pre or post
        self.ffn = FFN(dim, mult=ffn_mult, dropout=dropout)
        self.norm_q3 = ln()
        self.self_attn = MultiHeadAttention(
            dim, num_heads, dropout=dropout, qkv_norm=qkv_norm
        )

        # hepattn `bidirectional_ca` (default True there): after q ← CA(q, k, k)
        # the keys also cross-attend back to queries, so k is refined each layer.
        self.bidirectional_ca = bidirectional_ca
        if bidirectional_ca:
            self.norm_k2 = ln()
            self.norm_q4 = ln()
            self.kv_cross_attn = MultiHeadAttention(
                dim, num_heads, dropout=dropout, qkv_norm=qkv_norm
            )
            self.norm_k3 = nn.LayerNorm(dim)    # k-side FFN norm
            self.kv_ffn = FFN(dim, mult=ffn_mult, dropout=dropout)

    def forward(self, q, k, key_valid, attn_mask=None, attn_bias=None,
                kmeans_logits=None):
        """`attn_bias`: optional (B, N_q, N_k) additive float bias on the
        q→k cross-attention logits — the GMM log-density path. When
        `bidirectional_ca=True`, the k→q sub-block receives the transposed
        bias.

        `kmeans_logits`: only used when `cross_attn_mode='kmeans'`. The
        (B, N_q, N_k) assignment logits — typically the previous-layer's
        mask logits. Ignored in 'softmax' mode.
        """
        # 1. Cross-attend queries to keys (with mask attention)
        qn = self.norm_q1(q)
        kn = self.norm_k1(k)
        if self.cross_attn_mode == "kmeans":
            # KMeansCrossAttention: each key is hard-assigned to its argmax
            # query (via `kmeans_logits`), and the query receives the mean
            # of its assigned values. No attn_bias path here — the
            # assignment logits are the only gating.
            q = q + self.cross_attn(
                qn, k=kn, v=kn,
                kv_mask=key_valid,
                attn_mask=attn_mask,
                logits=kmeans_logits,
            )
        else:
            q = q + self.cross_attn(
                qn, kn, kn,
                key_padding_mask=key_valid,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        # 2. FFN — pre-norm or HybridNorm post-norm style
        if self.use_post_norm_dense:
            q = self.norm_q2(q)
            q = q + self.ffn(q)
        else:
            q = q + self.ffn(self.norm_q2(q))
        # 3. Self-attn among queries
        qn = self.norm_q3(q)
        q = q + self.self_attn(qn, qn, qn)

        # 4. (optional) keys cross-attend back to queries.
        if self.bidirectional_ca:
            if attn_mask is not None:
                attn_mask_t = attn_mask.transpose(-2, -1)              # (B, N_k, N_q)
                # Same fix as the q→k direction: any key with no incoming
                # attention from any query (a row of all-False in the
                # transposed mask) would softmax over an empty set in the
                # k→q reverse cross-attn → NaN. Open up those rows so
                # they attend everywhere.
                attn_mask_t = unmask_all_false(attn_mask_t)
            else:
                attn_mask_t = None
            kn = self.norm_k2(k)
            qn = self.norm_q4(q)
            # Transpose the GMM bias for the k→q direction: bias[b, p, q]
            # = "query q's spatial density at point p" stays semantically
            # the same after the transpose (point p prefers queries that
            # have high density at its location).
            attn_bias_t = (
                attn_bias.transpose(-2, -1) if attn_bias is not None else None
            )
            # No key_padding_mask: queries are always real (no dynamic queries).
            # The transposed attn_mask already gates by key validity.
            k = k + self.kv_cross_attn(
                kn, qn, qn, attn_mask=attn_mask_t, attn_bias=attn_bias_t,
            )
            if self.use_post_norm_dense:
                k = self.norm_k3(k)
                k = k + self.kv_ffn(k)
            else:
                k = k + self.kv_ffn(self.norm_k3(k))

        return q, k


class MaskFormerDecoder(nn.Module):
    def __init__(self, dim, num_heads, num_queries, num_layers,
                 num_classes=1, ffn_mult=4, dropout=0.0, mask_threshold=0.5,
                 bidirectional_ca=False, num_subsystems=1, subsystem_offset=0,
                 hybrid_norm=False, share_decoder_heads=True,
                 dynamic_query_source="static",
                 track_seed_bonus=6e4, track_hit_type=1,
                 use_gmm_query=False, gmm_K=4, gmm_sigma_init=1.0,
                 gmm_coord_range=(-6.0, 6.0), gmm_min_log_w=-30.0,
                 gmm_sigma_long=None, gmm_sigma_short=None,
                 gmm_init_rotation="random",
                 gmm_dynamic=False, gmm_dynamic_head_hidden_mult=2,
                 gmm_dynamic_per_layer=False,
                 gmm_warmup_hard_mask=False, gmm_warmup_topk=256,
                 # v5: anchor + scale per query. When `use_query_anchor=True`,
                 # all GMM density code is bypassed and an MLP-on-q produces
                 # `(μ, σ)` per query per layer; the cross-attention bias
                 # becomes `mlp([(p−μ)/σ, ‖·‖])`. Backward-compatible default
                 # is False (existing GMM path unchanged).
                 use_query_anchor=False,
                 query_anchor_sigma_init=0.5,
                 query_anchor_per_layer=True,
                 # k-Max-DeepLab cross-attention variant (single switch for
                 # every decoder layer). "softmax" = vanilla MaskFormer
                 # cross-attn (default); "kmeans" = hard-assignment k-means
                 # M-step using the previous-layer's mask logits as the
                 # assignment matrix (see kmeans_ca.py).
                 cross_attn_mode="softmax",
                 kmeans_respect_attn_mask=False,
                 kmeans_value_proj=False):
        super().__init__()
        self.cross_attn_mode = str(cross_attn_mode)
        if self.cross_attn_mode == "kmeans" and bidirectional_ca:
            # The k→q reverse direction is not part of k-Max-DeepLab; the
            # kmeans M-step only updates queries. We could still run
            # softmax k→q here, but for clarity surface this combination
            # to the user explicitly.
            raise ValueError(
                "cross_attn_mode='kmeans' with bidirectional_ca=True is "
                "not implemented (k-Max-DeepLab updates queries only). "
                "Disable bidirectional_ca for kmeans."
            )
        self.num_queries = num_queries
        self.mask_threshold = mask_threshold
        self.num_subsystems = num_subsystems
        self.subsystem_offset = subsystem_offset
        self.share_decoder_heads = share_decoder_heads
        # Tracker hits are physically pre-clustered (≈1 hit per charged
        # particle in the trkr layer), so they are excellent seeds for
        # queries. With `track_seed_bonus` added to their L2 scoring rule,
        # `topk` picks tracker hits before calo hits up to N_tracker_hits per
        # event — guaranteeing N_tracker_hits queries start out spatially
        # attached to a real track. The remaining query slots are filled by
        # high-norm calo encoder tokens as before.
        self.track_seed_bonus = float(track_seed_bonus)
        self.track_hit_type = int(track_hit_type)
        # ----------------------------------------------------------------
        # Gaussian-mixture spatial bias for cross-attention. Each query owns
        # a learnable mixture of K Gaussians in 3-space (means, full
        # covariance via Cholesky, mixing weights). The mixture's
        # log-density at every encoder hit is added — alongside the
        # existing binary mask — to the QK^T/√d logits before softmax.
        # Adapted from `/afs/cern.ch/work/m/mgarciam/private/mask3d`.
        # `use_gmm_query=False` (default) leaves the model identical to
        # the pre-GMM behaviour.
        # ----------------------------------------------------------------
        self.use_gmm_query = bool(use_gmm_query)
        # `gmm_dynamic=False` (default): static per-slot GMM as before.
        # `gmm_dynamic=True`: head-on-q event-conditional GMM.
        #   `gmm_dynamic_per_layer=False` (default & recommended): call the
        #     head ONCE per forward, using the initial seed queries q_0.
        #     Event-conditional but layer-static. No per-layer feedback loop.
        #   `gmm_dynamic_per_layer=True`: call the head at every decoder
        #     layer with the layer's current refined `q`. More expressive
        #     in principle but observed to be unstable in practice (40462110
        #     went NaN at step 1042 because the per-layer feedback amplified
        #     the head's gradient 8× and the mask predictions destabilised).
        # Zero-init head → identical to static at step 0 (backward compatible).
        self.gmm_dynamic = bool(gmm_dynamic) and self.use_gmm_query
        self.gmm_dynamic_per_layer = bool(gmm_dynamic_per_layer)
        # During the warm-up window the iterative `sigmoid(mask_logits)≥0.5`
        # attn-mask is disabled. With `gmm_warmup_hard_mask=False` (default)
        # the cross-attention then sees every key (only the GMM `attn_bias`
        # softly biases the softmax). With `gmm_warmup_hard_mask=True` we
        # additionally restrict attention to the top-`gmm_warmup_topk` keys
        # per query by GMM log-density — the GMM becomes a hard spatial
        # gate, forcing queries to learn their Gaussians' positions via the
        # BCE / coverage supervision rather than leaning on the mask head.
        self.gmm_warmup_hard_mask = bool(gmm_warmup_hard_mask)
        self.gmm_warmup_topk = int(gmm_warmup_topk)
        # ----------------------------------------------------------------
        # v5 anchor path. When `use_query_anchor=True` the existing GMM
        # density / mixture machinery is BYPASSED in `forward`: the per-
        # cross-attention bias comes from a small MLP on `(p−μ_q)/σ_q`
        # instead of `log p_q(x)`. `anchor_head` and `rel_pos_bias_mlp` are
        # only instantiated when needed so backward-compat models load
        # cleanly. `use_query_anchor` is honoured even when `use_gmm_query`
        # is also True — anchor takes precedence.
        # ----------------------------------------------------------------
        self.use_query_anchor = bool(use_query_anchor)
        self.query_anchor_per_layer = bool(query_anchor_per_layer)
        if self.use_query_anchor:
            self.anchor_head = QueryAnchorHead(
                hidden_dim=dim, sigma_init=query_anchor_sigma_init,
            )
            self.rel_pos_bias_mlp = nn.Sequential(
                nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 1),
            )
        if not self.use_gmm_query:
            self.gmm = None
        elif self.gmm_dynamic:
            self.gmm = DynamicGaussianQueryMixture(
                query_dim=dim,
                num_queries=num_queries,
                num_gaussians=gmm_K,
                sigma_init=gmm_sigma_init,
                coord_range=tuple(gmm_coord_range),
                min_log_w=gmm_min_log_w,
                sigma_long=gmm_sigma_long,
                sigma_short=gmm_sigma_short,
                init_rotation=gmm_init_rotation,
                head_hidden_mult=int(gmm_dynamic_head_hidden_mult),
            )
        else:
            self.gmm = GaussianQueryMixture(
                num_queries=num_queries,
                num_gaussians=gmm_K,
                sigma_init=gmm_sigma_init,
                coord_range=tuple(gmm_coord_range),
                min_log_w=gmm_min_log_w,
                sigma_long=gmm_sigma_long,
                sigma_short=gmm_sigma_short,
                init_rotation=gmm_init_rotation,
            )
        # ----------------------------------------------------------------
        # `dynamic_query_source` controls how initial queries are formed:
        #
        #   "static":  N_q learned vectors, randn-init. Mask2Former / DETR
        #              convention. Simple but no spatial prior at step 0 —
        #              all 320 queries are i.i.d. Gaussian. The Hungarian
        #              matcher then assigns them arbitrarily and the model
        #              spends most of its early gradient budget learning to
        #              spatially differentiate queries from scratch. This
        #              is the cold-start failure mode of run vhc0nms5.
        #
        #   "encoder": Pick the top-N_q encoder hit-tokens by L2 norm and
        #              use those as initial queries. Mask3D-paper FPS analog
        #              (hepattn's `dynamic_query_source: hit`): queries are
        #              seeded with real hit features that ALREADY carry
        #              spatial information from the encoder, so layer-0
        #              masks are non-random from the start. The encoder
        #              still receives gradient through the seed gather, so
        #              the SEEDS are learned end-to-end even though the
        #              SCORING rule is fixed. A learned per-slot residual
        #              `query_bias` is added so queries can still
        #              differentiate themselves and learn a reusable prior
        #              across events.
        # ----------------------------------------------------------------
        if dynamic_query_source not in ("static", "encoder"):
            raise ValueError(
                f"dynamic_query_source must be 'static' or 'encoder'; got "
                f"{dynamic_query_source!r}."
            )
        self.dynamic_query_source = dynamic_query_source

        # Static path: full learned queries.
        # Encoder path: a small learned residual added on top of seeds. We
        # initialize this small (0.02·randn) so the encoder seed dominates
        # at step 0 — otherwise we'd have the same cold-start problem as
        # the static path.
        if dynamic_query_source == "static":
            self.initial_queries = nn.Parameter(torch.randn(num_queries, dim))
        else:
            self.query_bias = nn.Parameter(0.02 * torch.randn(num_queries, dim))
            # NOTE: previous revisions had `self.queryness = nn.Linear(dim, 1)`
            # here, intended as a learnable per-token scoring head. Removed:
            # `topk` only consumes `.indices` (non-differentiable), so the
            # scorer's outputs never reached the loss — it could not learn
            # AND its parameters tripped DDP's unused-params check (job
            # 40330284). Scoring now uses `keys.norm(dim=-1)` instead — a
            # fixed rule that picks high-magnitude encoder tokens, which
            # tend to be salient. The encoder still receives gradient
            # through the seed gather, so the seeds themselves remain
            # learned end-to-end. A real "learnable proposals" head would
            # need an auxiliary hit-filter loss (hepattn-style two-stage);
            # see followup options if encoder-seeded queries help.

        self.layers = nn.ModuleList(
            [MaskFormerDecoderLayer(dim, num_heads, ffn_mult, dropout,
                                    bidirectional_ca=bidirectional_ca,
                                    depth=i, hybrid_norm=hybrid_norm,
                                    cross_attn_mode=self.cross_attn_mode,
                                    kmeans_respect_attn_mask=bool(kmeans_respect_attn_mask),
                                    kmeans_value_proj=bool(kmeans_value_proj))
             for i in range(num_layers)]
        )

        # Task heads. With `share_decoder_heads=True` (default, hepattn-style)
        # there's a SINGLE pair of heads called at every decoder layer —
        # their parameters always feed the final-layer loss path, so DDP
        # never sees them as "unused" even when `aux_layer_weight=0`.
        # With False, separate heads per layer (richer deep supervision but
        # the intermediate ones are unused under aux_layer_weight=0, so DDP
        # needs `find_unused_parameters=True`).
        n_head_slots = 1 if share_decoder_heads else (num_layers + 1)
        self.mask_heads = nn.ModuleList(
            [ObjectHitMaskTask(dim, num_subsystems=num_subsystems,
                               subsystem_offset=subsystem_offset)
             for _ in range(n_head_slots)]
        )
        self.cls_heads = nn.ModuleList(
            [ObjectClassificationTask(dim, num_classes=num_classes)
             for _ in range(n_head_slots)]
        )

    def _heads(self, q, k, layer_idx, hit_subsystem=None):
        idx = 0 if self.share_decoder_heads else layer_idx
        return {
            "mask_logits": self.mask_heads[idx](q, k, hit_subsystem=hit_subsystem),
            "cls_logits": self.cls_heads[idx](q),
        }

    def _seed_queries_from_encoder(self, keys, key_valid, hit_subsystem=None):
        """Pick top-N_q encoder tokens by L2 norm, gather them, and add a
        learned per-slot residual.

        keys:           (B, N_k, D), encoder output scattered to padded form.
        key_valid:      (B, N_k) bool — True where a real hit lives.
        hit_subsystem:  optional (B, N_k) int — per-hit subsystem id. When
                        provided and `track_seed_bonus > 0`, tracker hits get
                        a fixed score bonus so they are picked first by topk.
        Returns:        q (B, N_q, D), and `seed_idx` (B, N_q) for diagnostics.
        """
        B, N_k, _ = keys.shape
        # Compute the topk score in fp32: under AMP the encoder runs in
        # low-precision and `track_seed_bonus` is added to keys.norm(); we
        # cast both sides to fp32 so the additive bonus never saturates
        # the surrounding tensor's range. Selection is non-differentiable
        # (topk → integer indices) so the fp32 scope is purely numerical.
        scores = keys.float().norm(dim=-1)                        # (B, N_k)
        # Tracker hits get a large additive bonus so they sit at the top of
        # the topk regardless of encoder L2 norm. Up to N_tracker queries
        # are guaranteed to be track-seeded; the rest fall back to highest-
        # norm non-tracker tokens.
        if hit_subsystem is not None and self.track_seed_bonus != 0.0:
            track_mask = hit_subsystem == self.track_hit_type
            scores = scores + track_mask.float() * self.track_seed_bonus
        # Padded slots can never be queries; force them to -inf so topk
        # picks them only as a last resort (never under N_q ≤ min seq_len).
        scores = scores.masked_fill(~key_valid, float("-inf"))
        # Stable topk: return both indices and values; ascending=False.
        #
        # k is CLAMPED to the padded key width. The note below handles padded
        # SLOTS (N_k >= num_queries, some slots invalid); it does not handle the
        # padded WIDTH itself being smaller than num_queries, which is decided
        # by the largest event in the batch. So this never fires at training
        # batch sizes but does at eval, where the reference configuration is
        # --batch-size 1 and any event with fewer hits than num_queries raises
        # `RuntimeError: selected index k out of range`. Identical bug and fix
        # to ipa_decoder._seed_queries_from_encoder (commit bfe748c); this is
        # the MaskFormerDecoder copy, missed at the time.
        k = min(int(self.num_queries), int(keys.size(1)))
        topk = scores.topk(k, dim=1)
        seed_idx = topk.indices                                   # (B, k)
        idx_exp = seed_idx.unsqueeze(-1).expand(-1, -1, keys.size(-1))
        seed = keys.gather(1, idx_exp)                            # (B, k, D)
        # If an event has fewer valid hits than `num_queries`, the spillover
        # slots picked padded positions (their score was -inf, but topk
        # still has to pick *something* for those slots). Replace those
        # seeds with zeros so the resulting query is just `query_bias` —
        # the slot stays usable, just without an encoder-derived prior.
        # Avoids crashing on the rare small event without sacrificing the
        # encoder-seeded prior on the rest.
        seed_valid = key_valid.gather(1, seed_idx)                # (B, k)
        seed = seed * seed_valid.unsqueeze(-1).to(seed.dtype)
        if k < self.num_queries:
            # Zero-pad the missing slots, which is exactly the semantics above:
            # a slot with a zero seed becomes just `query_bias`.
            pad = self.num_queries - k
            seed = F.pad(seed, (0, 0, 0, pad))                    # (B, N_q, D)
            seed_idx = F.pad(seed_idx, (0, pad))                  # diagnostic only
        q = seed + self.query_bias.unsqueeze(0)                   # (B, N_q, D)
        return q, seed_idx

    def forward(self, keys, key_valid, hit_subsystem=None, pos_padded=None,
                disable_mask_attention=False):
        """`pos_padded`: optional (B, N_k, 3) per-key xyz in the same frame
        the GMM was initialised with (metres, scaled by build_targets'
        `pos_scale`). Required when `use_gmm_query=True`; ignored otherwise.

        `disable_mask_attention=True` skips building the per-layer binary
        `attn_mask` from sigmoid(mask_logits) ≥ threshold — only the GMM
        spatial bias (via `attn_bias`) and key-padding then drive the
        cross-attention. Used during GMM warm-up so the model is forced
        to make the Gaussians do the spatial discrimination instead of
        leaning on the mask-attention path.
        """
        B = keys.size(0)
        if self.dynamic_query_source == "encoder":
            q, seed_idx = self._seed_queries_from_encoder(
                keys, key_valid, hit_subsystem=hit_subsystem,
            )
        else:
            q = self.initial_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
            seed_idx = None

        # v5 anchor path — per-query center for `QueryAnchorHead`. With
        # encoder-seeded queries we use the position of the seeded encoder
        # token (so the anchor starts near that token's xyz, then gets
        # nudged by the MLP delta). With static queries we fall back to
        # zeros (the MLP delta is then the entire μ).
        anchor_center_xyz = None
        if self.use_query_anchor and pos_padded is not None:
            if seed_idx is not None:
                idx_exp = seed_idx.unsqueeze(-1).expand(B, self.num_queries, 3)
                anchor_center_xyz = pos_padded.gather(1, idx_exp)        # (B, N_q, 3)
            else:
                anchor_center_xyz = torch.zeros(
                    B, self.num_queries, 3,
                    device=pos_padded.device, dtype=pos_padded.dtype,
                )

        # GMM spatial bias.
        #
        # We request the UNCLAMPED log-density from the GMM and apply the
        # `min_log_w` clamp only on the attention copy. Why: the clamp is
        # there to keep the attention softmax numerically safe, but
        # `torch.clamp` has zero gradient w.r.t. (μ, Σ) wherever it fires,
        # which is exactly the regime in which the new GMM coverage-loss
        # term needs gradient (hits very far from current μ → very negative
        # log_w → clamp would otherwise kill the supervision signal).
        #
        # Static GMM: parameters are not a function of q → log-density depends
        # only on hit xyz → compute once before the layer loop.
        # Dynamic GMM (default `gmm_dynamic_per_layer=False`): compute the
        #   head ONCE here from the initial seed queries q_0. Event-conditional
        #   but layer-static; no per-layer feedback loop, no 8× gradient
        #   amplification on the head.
        # Dynamic GMM with `gmm_dynamic_per_layer=True`: deferred to inside
        #   the layer loop, recomputed from each layer's refined `q`.
        #   Observed to be unstable (40462110 → NaN at step 1042) — kept as
        #   an opt-in for ablation, not recommended.
        attn_bias = None              # unclamped — returned to the caller (loss)
        attn_bias_attn = None         # clamped (or anchor bias) — used inside attention
        if self.use_query_anchor:
            # v5 anchor path takes precedence over GMM. Compute the initial
            # anchor-based attention bias from the seed queries; if
            # `query_anchor_per_layer=True` (default), the bias is
            # re-predicted inside the layer loop from each layer's refined q.
            if pos_padded is None:
                raise ValueError(
                    "MaskFormerDecoder(use_query_anchor=True) requires "
                    "`pos_padded` (B, N_k, 3) so we can build the relative "
                    "coordinates `(p - μ)/σ`."
                )
            mu, sigma = self.anchor_head(q, anchor_center_xyz)              # (B, N_q, 3), (B, N_q, 1)
            rel = (pos_padded[:, None, :, :] - mu[:, :, None, :]) / sigma.unsqueeze(2)
            rel_norm = rel.norm(dim=-1, keepdim=True)
            pos_bias = self.rel_pos_bias_mlp(
                torch.cat([rel, rel_norm], dim=-1)
            ).squeeze(-1)                                                   # (B, N_q, N_k)
            attn_bias = pos_bias
            attn_bias_attn = pos_bias        # MLP output is bounded — no clamp.
        elif self.use_gmm_query:
            if pos_padded is None:
                raise ValueError(
                    "MaskFormerDecoder(use_gmm_query=True) requires "
                    "`pos_padded` (B, N_k, 3) in MaskFormerDecoder.forward(...). "
                    "Pass the scaled per-hit xyz from build_targets."
                )
            if not self.gmm_dynamic:
                attn_bias = self.gmm(pos_padded, clamp=False)
                attn_bias_attn = attn_bias.clamp(min=self.gmm.min_log_w)
            elif not self.gmm_dynamic_per_layer:
                # Dynamic head, single call on initial queries.
                attn_bias = self.gmm(q, pos_padded, clamp=False)
                attn_bias_attn = attn_bias.clamp(min=self.gmm.min_log_w)
            # else: per-layer recompute inside the loop below.

        per_layer = []

        # Run heads on the initial queries (layer 0); their mask becomes the
        # gate for the first cross-attn.
        outs = self._heads(q, keys, layer_idx=0, hit_subsystem=hit_subsystem)
        per_layer.append(outs)

        for li, layer in enumerate(self.layers):
            if disable_mask_attention:
                # Warm-up branch. Two flavours:
                # * `gmm_warmup_hard_mask=False` (default): no binary mask;
                #   only the GMM `attn_bias` softly modulates the softmax.
                #   Every key remains attendable.
                # * `gmm_warmup_hard_mask=True`: build a hard topK mask from
                #   `attn_bias` (= GMM log-density) so each query can only
                #   attend to its `gmm_warmup_topk` highest-density keys.
                #   Forces the model to USE the GMM during warm-up; BCE
                #   containment loss provides the differentiable supervision
                #   that moves μ / Σ into place (gate itself is non-diff).
                if (self.gmm_warmup_hard_mask and self.use_gmm_query
                        and attn_bias is not None):
                    K = min(self.gmm_warmup_topk, attn_bias.size(-1))
                    # Mask invalid keys to -inf so topk never selects padding.
                    log_w_for_topk = attn_bias.masked_fill(
                        ~key_valid.unsqueeze(1), float("-inf"),
                    )
                    _, top_idx = log_w_for_topk.topk(K, dim=-1)   # (B, N_q, K)
                    gmm_inside = torch.zeros_like(attn_bias, dtype=torch.bool)
                    gmm_inside.scatter_(-1, top_idx, True)
                    gmm_inside = gmm_inside & key_valid.unsqueeze(1)
                    attn_mask = unmask_all_false(gmm_inside)
                else:
                    attn_mask = None
            else:
                attn_mask = (outs["mask_logits"].sigmoid() >= self.mask_threshold).detach()
                attn_mask = attn_mask & key_valid.unsqueeze(1)   # (B, N_q, N_k)
                attn_mask = unmask_all_false(attn_mask)
            # Per-layer dynamic-GMM recompute is opt-in only. Default
            # `gmm_dynamic_per_layer=False` reuses the once-per-forward
            # bias built above; this is the stable path.
            if self.use_gmm_query and self.gmm_dynamic and self.gmm_dynamic_per_layer:
                attn_bias = self.gmm(q, pos_padded, clamp=False)
                attn_bias_attn = attn_bias.clamp(min=self.gmm.min_log_w)
            # v5 per-layer anchor recompute. Default ON — the reference's
            # v5 also recomputes (μ, σ) every layer from current q. Disable
            # via `query_anchor_per_layer=False` for once-per-forward.
            if self.use_query_anchor and self.query_anchor_per_layer:
                mu, sigma = self.anchor_head(q, anchor_center_xyz)
                rel = (pos_padded[:, None, :, :] - mu[:, :, None, :]) / sigma.unsqueeze(2)
                rel_norm = rel.norm(dim=-1, keepdim=True)
                pos_bias = self.rel_pos_bias_mlp(
                    torch.cat([rel, rel_norm], dim=-1)
                ).squeeze(-1)
                attn_bias = pos_bias
                attn_bias_attn = pos_bias
            # `keys` is updated when bidirectional_ca=True; otherwise unchanged.
            # Use the clamped GMM bias for the attention softmax (safe in
            # bf16); the unclamped copy is returned to the loss path.
            # In kmeans mode pass the just-computed mask_logits as the
            # k-means assignment matrix — the same quantity that drives
            # `attn_mask` ALSO drives the hard argmax-assignment.
            kmeans_logits = (
                outs["mask_logits"] if self.cross_attn_mode == "kmeans"
                else None
            )
            q, keys = layer(
                q, keys, key_valid,
                attn_mask=attn_mask,
                attn_bias=attn_bias_attn,
                kmeans_logits=kmeans_logits,
            )
            outs = self._heads(q, keys, layer_idx=li + 1, hit_subsystem=hit_subsystem)
            per_layer.append(outs)

        # Final outputs are the last entry; per_layer has num_layers + 1 entries.
        # `attn_bias` is the per-(query, hit) GMM log-density (None when
        # use_gmm_query=False) — returned so the loss can compute its
        # coverage / NLL term against matched GT masks.
        return per_layer, q, attn_bias
