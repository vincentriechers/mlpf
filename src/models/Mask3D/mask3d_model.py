"""Mask3D-style Lightning module for HEP particle reconstruction.

Mirrors the shape of `src/models/GATr/Gatr_pf_e_noise.py::ExampleWrapper`
so it slots into `src/utils/train_utils.py::model_setup` unchanged. Each
event becomes a sequence of hit tokens; a transformer encoder + a
MaskFormer-style decoder with iterative mask attention predict per-hit
particle masks plus per-query validity. Loss is Hungarian-matched with
deep supervision across decoder layers.

Designed to coexist with the existing Object Condensation / Energy
Correction pipeline: when `--correction` is enabled, an `ECAdapter`
converts the model's outputs into the OC-style `[cluster_coord, beta]`
contract that `EnergyCorrection.forward_correction` already expects.
"""
import os
import time
import torch
import torch.nn as nn
import lightning as L
import wandb

from src.models.Mask3D.input_net import InputNet, PerSubsystemInputNet
# GATr backbone is imported lazily — only when `--network-option
# use_gatr_backbone True` is set, since it pulls in the gatr package.
from src.models.Mask3D.encoder import Encoder
from src.models.Mask3D.decoder import MaskFormerDecoder
from src.models.Mask3D.targets import build_targets
from src.models.Mask3D.loss import mask3d_loss
from src.models.Mask3D.matcher import Matcher
from src.models.Mask3D.viz import plot_pred_vs_true_clusters_fig
from src.models.Mask3D.ec_adapter import ECAdapter
from src.models.energy_correction_NN_v1 import EnergyCorrection
import dgl
from src.layers.inference_oc_mask3d import (
    create_and_store_graph_output_mask3d,
    labels_from_masks,
)
from src.layers.inference_oc import remove_bad_tracks_from_cluster_v1


class CosineAnnealingThenFixedScheduler:
    def __init__(self, optimizer, T_max, fixed_lr):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=fixed_lr)
        self.fixed_lr = 1e-6
        self.T_max = T_max
        self.step_count = 0
        self.optimizer = optimizer

    def step(self):
        if self.step_count < self.T_max:
            self.cosine_scheduler.step()
        else:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.fixed_lr
        self.step_count += 1

    def get_last_lr(self):
        if self.step_count < self.T_max:
            return self.cosine_scheduler.get_last_lr()
        return [self.fixed_lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        return {"step_count": self.step_count,
                "cosine_scheduler_state": self.cosine_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler_state"])


class ExampleWrapper(L.LightningModule):
    """Mask3D-style top-level Lightning module.

    Hyper-parameters (passed via --network-option). Defaults match the
    hepattn CLD `base.yaml` reference config.

        dim:               model dim (default 256)
        n_heads:           attention heads (default 8)
        enc_layers:        encoder layers (default 8)
        dec_layers:        decoder layers (default 8)
        num_queries:       object queries (default 320)
        mask_bce_weight:   default 2.0
        mask_dice_weight:  default 1.0
        obj_bce_weight:    default 1.0
        null_weight:       default 0.25
        aux_layer_weight:  default 1.0
    """

    def __init__(
        self,
        args,
        dev,
        dim=256,
        n_heads=8,
        enc_layers=8,
        dec_layers=8,
        num_queries=320,
        # Defaults mirror hepattn `cld/configs/base.yaml`: BCE+Dice with
        # dice 3× heavier than BCE, low objectness loss weight, no deep
        # supervision. The previous defaults (focal, mask_bce=2, dice=1,
        # obj_bce=1, aux_layer_weight=1) corresponded to the Mask2Former /
        # MaskFormer paper recipe; that one fights the BCE-trap with focal
        # instead of with weight balance.
        mask_bce_weight=0.25,
        mask_dice_weight=0.75,
        obj_bce_weight=0.1,
        null_weight=1.0,
        aux_layer_weight=0.0,
        mask_loss_type="bce",
        focal_gamma=2.0,
        # hepattn's matching cost uses dice ONLY for masks (BCE/focal terms
        # commented out in their YAML). Setting this False matches that.
        mask_cost_use_classification=False,
        # The objectness loss has a *small* weight (0.1) but the matching
        # cost stays at 1.0 — so we keep two separate knobs.
        obj_bce_cost_weight=1.0,
        # Decoder layers cross-attend k → q after q ← k, refining keys each
        # layer. Mirrors hepattn `bidirectional_ca: True`. Off by default
        # because it adds compute and an extra parameter block per layer;
        # flip on with --network-option bidirectional_ca True.
        bidirectional_ca=False,
        # Compute mask losses + mask matching cost per hit-type subsystem
        # and SUM across subsystems present in the batch. Mirrors hepattn's
        # CLD config: 5 separate ObjectHitMaskTask instances each
        # contributing additively to the total. Pair this with
        # `num_subsystems > 1` to also get the architectural separation
        # (separate constituent_net per subsystem).
        per_subsystem_loss=False,
        # Number of detector subsystems for separate mask heads
        # (`ObjectHitMaskTask(num_subsystems=N)`). When >1, each subsystem
        # gets its own constituent (key) projection MLP; the query
        # projection stays shared. For our CLD encoding
        # (hit_type ∈ {0=noise, 1=tracker, 2=ecal, 3=hcal, 4=muon}), use
        # `num_subsystems=4` paired with `subsystem_offset=1` — only the
        # 4 real detectors get heads, noise is excluded.
        num_subsystems=1,
        # Shift applied to hit_type → head index. CLD: pass 1 so hit_type=1
        # maps to head 0, hit_type=4 to head 3, and hit_type=0 (noise)
        # gets clamped (no head of its own). Also tells the per-subsystem
        # loss to skip noise (hit_type < subsystem_offset).
        subsystem_offset=0,
        # HybridNorm (arXiv:2503.04598): layer-0 uses pre-norm, layer ≥ 1
        # uses no pre-norm (qk-norm inside attn) + post-norm style FFN.
        # Mirrors hepattn `hybrid_norm: true` in CLD base.yaml.
        hybrid_norm=False,
        # Per-subsystem input embedding (hepattn-style). When True, replaces
        # the single shared `InputNet` with `PerSubsystemInputNet`: one
        # `InputNet` per detector, each with detector-specific fields
        # (tracker uses χ²+p; ecal/hcal use log_e; muon uses xyz+r/θ/φ
        # only). Requires `num_subsystems=4, subsystem_offset=1` (CLD).
        per_subsystem_input=False,
        # Share a single set of (mask_head, cls_head) across all decoder
        # layers. Default True (matches hepattn). With False, each layer
        # gets its own heads — but combined with `aux_layer_weight=0` the
        # intermediate heads' params never receive gradients and DDP needs
        # `find_unused_parameters=True`. Sharing removes that need.
        share_decoder_heads=True,
        # GATr backbone: when True, replaces the (`InputNet` + `Encoder`)
        # pair with a `GATrBackbone` — geometric-algebra-transformer over
        # per-hit positions + scalar features. E(3)-equivariant; pulls in
        # the `gatr` package. Mutually exclusive with `per_subsystem_input`
        # (the GATr backbone reads from `g.ndata` directly, not from the
        # per-subsystem feats_flat layout).
        use_gatr_backbone=False,
        gatr_blocks=10,
        gatr_hidden_mv=16,
        gatr_hidden_s=64,
        # Phi-windowed encoder attention (hepattn parity, CLD base config).
        # Hits are sorted by phi within each event before the encoder, then
        # attention is local: each event of length L is split into
        # non-overlapping chunks of `window_size`. Only takes effect when an
        # encoder is used (i.e. NOT `use_gatr_backbone=True`).
        # `window_wrap=True` alternates layers with a Swin-style W//2 shift
        # so chunk boundaries don't sit on the same hits twice in a row.
        # Defaults match hepattn (window_size: 1024, window_wrap: true);
        # set window_size=None to disable and recover full-event attention.
        window_size=1024,
        window_wrap=True,
        # Initial-query strategy. "static" = N_q learned vectors (Mask2Former
        # default; cold-start failure mode in run vhc0nms5). "encoder" = top-K
        # encoder tokens by a learnable queryness score, plus a learned slot-
        # specific bias. The encoder path bakes spatial information into
        # queries from step 0 — Mask3D-paper FPS analog. Default to "encoder"
        # because the cold-start hit on this codebase is severe.
        dynamic_query_source="encoder",
        # ---------------- Track-relevance knobs ----------------
        # Tracker hits (hit_type=1) are physically pre-clustered — each one
        # is ≈ one charged particle. With the default global BCE/Dice loss
        # averaged across all keys, tracker hits (~5–50 per event) provide
        # negligible gradient compared to the ~thousands of calo hits, and
        # the model never learns to associate them with their particle. The
        # following three knobs make tracks first-class citizens of the
        # reconstruction:
        #
        #   track_loss_weight     scales tracker hits' contribution in
        #                         BCE+Dice loss AND in the matching cost.
        #                         Set to 1.0 to disable.
        #   track_seed_bonus      additive bonus on the encoder-seed score
        #                         for tracker hits, so they sit at the top
        #                         of the topk and are preferred as initial
        #                         queries. 0.0 disables. Large default
        #                         guarantees track-seeded queries up to
        #                         N_tracker per event.
        #   force_track_assignment  at inference, assign every tracker hit
        #                         to its argmax-query as long as that
        #                         query is valid — never drop a tracker
        #                         hit because mask prob < threshold. OFF
        #                         by default because some tracker hits
        #                         (fakes, secondaries with particle_number
        #                         == 0) legitimately belong to no cluster;
        #                         forcing assignment would invent fake
        #                         clusters for them. Opt in via the flag
        #                         when you know all tracks in your data
        #                         have a matched cluster.
        track_loss_weight=3.0,
        track_seed_bonus=6e4,
        force_track_assignment=False,
        track_hit_type=1,
        track_mask_threshold=0.1,
        # ---------------- Gaussian-mixture query bias ----------------
        # Each query owns K learnable Gaussians in 3-space (means, full
        # covariance via Cholesky, mixing weights). Their log-density at
        # every encoder hit is added to the cross-attention QK^T/√d
        # logits — alongside the existing binary mask attention. The
        # GMM's xyz frame is the same scaled metres frame build_targets
        # uses (mm → m via `pos_scale=0.001`), so `gmm_coord_range` and
        # `gmm_sigma_init` should be in metres.
        #
        #   use_gmm_query     master enable.
        #   gmm_K             Gaussians per query (default 4).
        #   gmm_sigma_init    initial std of each Gaussian (metres).
        #   gmm_coord_range   (lo, hi) range μ is sampled from at init.
        #                     ±6 m comfortably covers the CLD active volume.
        #   gmm_min_log_w     lower clamp on the bias — prevents a starved
        #                     Gaussian from sending its bias to -∞ and
        #                     permanently zeroing the gradient. -30 ≈
        #                     softmax weight 1e-13, effectively dead but
        #                     not collapsed.
        use_gmm_query=False,
        gmm_K=4,
        gmm_sigma_init=1.0,
        gmm_coord_range=(-6.0, 6.0),
        gmm_min_log_w=-30.0,
        # ---------------- GMM init shape + warm-up ----------------
        # When both `gmm_sigma_long` and `gmm_sigma_short` are set, each
        # Gaussian is initialised as Σ = R diag(σ_long², σ_short², σ_short²) Rᵀ
        # with a random rotation R. Defaults None → fall back to the
        # original isotropic σ_init init. Tune for the dataset:
        # CLD/HCAL curling showers want shafts of ~0.5 m × 0.05 m × 0.05 m.
        #
        # `gmm_warmup_steps` > 0 disables the Mask2Former binary attn_mask
        # for the first N training steps so the GMM is the only spatial
        # discriminator in the cross-attention. Forces the means to
        # migrate toward real-hit regions before mask-attention takes
        # over again. 0 disables the warm-up.
        gmm_sigma_long=None,
        gmm_sigma_short=None,
        gmm_init_rotation="random",
        gmm_warmup_steps=0,
        # Event-conditional GMM: a small MLP on each query's current feature
        # vector predicts deltas (Δμ, Δlog_diag, Δoff_diag, Δπ_logit) on top
        # of a per-slot static base prior. Recomputed every decoder layer so
        # the GMM follows the iterative query refinement. Head is zero-init
        # → dynamic = static at step 0 (backward compatible).
        gmm_dynamic=False,
        gmm_dynamic_head_hidden_mult=2,
        gmm_dynamic_per_layer=False,
        # Negative-log-likelihood loss term on the matched (query, GT particle)
        # pairs: for each matched pair, push the query's GMM mixture to
        # maximise the log-density of the GT particle's hits. Direct
        # supervision on μ / Σ — far stronger / cleaner than letting the GMM
        # learn only via the attn_bias path. 0.0 disables the term (default).
        # A small positive value (e.g. 0.1) is usually sufficient.
        gmm_coverage_weight=0.0,
        # BCE containment loss on the matched (query, GT particle) GMM
        # log-densities: treats `log p_q(x_k)` as a binary classifier logit
        # for "is hit k inside cluster matched by query q?" with target =
        # `gt_mask[b, p, k]`. Penalises both false negatives (cluster hits
        # outside the GMM) and false positives (non-cluster hits the GMM
        # puts mass on). Per-(q, p, k) gradient bounded to [-1, 1] so it
        # composes safely with the rest of the loss. 0.0 disables.
        gmm_bce_weight=0.0,
        # Hard topK GMM gate during warm-up. When True, the cross-attention's
        # binary mask is replaced (only while `disable_mask_attention=True`)
        # by the top-`gmm_warmup_topk` keys per query by GMM log-density.
        # Forces queries to USE the GMM positions during warm-up — pair with
        # `gmm_bce_weight > 0` so the supervision moves μ to real hits before
        # the gate locks them in.
        gmm_warmup_hard_mask=False,
        gmm_warmup_topk=256,
        # v5 anchor mode (separate-from-GMM design): each query gets a 3D
        # anchor μ and scale σ from an MLP on its feature vector + an
        # encoder-seed center; cross-attention bias becomes
        # `mlp([(p−μ)/σ, ‖·‖])`. Bypasses the K-mixture GMM. Backward-
        # compatible default is False (existing GMM path unchanged).
        use_query_anchor=False,
        query_anchor_sigma_init=0.5,
        query_anchor_per_layer=True,
        # k-Max-DeepLab variant of MaskFormerDecoder cross-attention.
        # "softmax" (default) keeps the standard MaskFormer behaviour;
        # "kmeans" replaces q←k with hard-assignment k-means M-step
        # (KMeansCrossAttention) using the previous-layer mask logits as
        # the assignment matrix. Only takes effect when use_ipa_decoder=False.
        cross_attn_mode="softmax",
        kmeans_respect_attn_mask=False,
        kmeans_value_proj=False,
        # Per-particle dice reweighting. 'none' (default) keeps the original
        # uniform mean-over-matched-particles dice (small + large clusters
        # contribute equally regardless of hit count). 'linear' / 'sqrt' /
        # 'log' weight each particle's dice score by |m_p|, √|m_p|, or
        # log(|m_p|+1) — addresses Mask2Former's tendency to under-cluster
        # large showers (the dice penalty for dropping hundreds of peripheral
        # hits is otherwise diluted by the average across many small,
        # well-fit clusters).
        dice_size_weighting="none",
        # ---------------- IPA decoder (AlphaFold-style) ----------------
        # When True, replace (InputNet + Encoder) + MaskFormerDecoder with
        # `AttnIPABackbone` (plain-attention backbone returning per-hit
        # feature + raw xyz) + `IPADecoder` (per-query frame (q_s, T, t),
        # geometric mask logit). The rest of the pipeline — Hungarian
        # matcher, mask3d_loss, EC adapter, predict / shower storage — is
        # unchanged because the IPA decoder emits the same per-layer
        # {mask_logits, cls_logits} schema and `queries` (= q_s) tensor.
        # Mirrors the `use_gatr_backbone` swap pattern.
        use_ipa_decoder=False,
        # Equivariant backbone swap (mirrors attn_ipa_model): when True the
        # IPA-decoder path builds EquivariantAttnIPABackbone instead of the
        # vanilla AttnIPABackbone, so equivariant attn_ipa checkpoints
        # (encoder.local_geom.* keys, invariant input net) load at eval.
        equivariant_backbone=False,
        equivariant_symmetry="SO(2)_z",
        equivariant_local_geom=True,
        equivariant_local_geom_k=64,
        equivariant_local_geom_heads=4,
        equivariant_local_geom_head_dim=24,
        equivariant_local_geom_num_blocks=2,
        equivariant_local_geom_ffn_mult=2,
        # IPADecoder knobs (only used when use_ipa_decoder=True).
        ipa_mask_direction_aware=False,
        ipa_coord_augmented_keys=False,
        ipa_mask_threshold=0.5,
        # FAPE-style frame supervision weights (0 disables; loss-only, has
        # no effect at inference / predict).
        frame_translation_weight=0.0,
        frame_rotation_weight=0.0,
        # Per-hit energy weighting of the mask losses (none / log / linear);
        # loss-only, no effect at inference / predict.
        energy_weight_mode="none",
        energy_weight_scale=1.0,
        energy_completeness_weight=0.0,
        recall_weight=0.0,
        recall_tau=5.0,
        recall_E_alpha=0.5,
        **kwargs,
    ):
        super().__init__()
        self.strict_loading = False
        self.args = args
        self.dev = dev
        self.loss_final = 0.0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

        # Input feature width: pos(3) + hit_type_oh(5 or 6) + e(1) + p(1) + chi2(1) = 11 or 12
        n_oh = 6 if getattr(args, "ILD", False) else 5
        in_dim = 3 + n_oh + 1 + 1 + 1   # +chi2
        self.in_dim = in_dim
        self.dim = dim

        self.per_subsystem_input = per_subsystem_input
        self.use_gatr_backbone = use_gatr_backbone
        self.use_ipa_decoder = bool(use_ipa_decoder)
        self.frame_translation_weight = float(frame_translation_weight)
        self.frame_rotation_weight = float(frame_rotation_weight)
        self.energy_weight_mode = str(energy_weight_mode)
        self.energy_weight_scale = float(energy_weight_scale)
        self.energy_completeness_weight = float(energy_completeness_weight)
        self.recall_weight = float(recall_weight)
        self.recall_tau = float(recall_tau)
        self.recall_E_alpha = float(recall_E_alpha)

        if use_gatr_backbone and per_subsystem_input:
            raise ValueError(
                "use_gatr_backbone=True is incompatible with "
                "per_subsystem_input=True. The GATr backbone reads from "
                "`g.ndata` directly and produces unified per-hit features; "
                "there is no per-subsystem split inside it."
            )

        if self.use_ipa_decoder:
            from src.models.Mask3D.attn_ipa_backbone import AttnIPABackbone
            # NOTE: the backbone MUST be stored as `self.encoder` (not
            # `self.input_net`) so the state_dict keys match the training
            # module `attn_ipa_model.AttnIPAModel`, which did
            # `self.encoder = AttnIPABackbone(...)`. The checkpoint keys are
            # `encoder.input_net.*` / `encoder.encoder.*`; binding to
            # `input_net` here would silently skip every backbone weight
            # under `strict_loading=False` → random backbone → NaN outputs.
            if equivariant_backbone:
                from src.models.Mask3D.equivariant_attn_ipa_backbone import (
                    EquivariantAttnIPABackbone,
                )
                self.encoder = EquivariantAttnIPABackbone(
                    in_dim=in_dim, dim=dim, num_heads=n_heads,
                    num_layers=enc_layers, hybrid_norm=hybrid_norm,
                    window_size=window_size, window_wrap=window_wrap,
                    n_oh=n_oh,
                    symmetry=str(equivariant_symmetry),
                    local_geom=bool(equivariant_local_geom),
                    local_geom_k=int(equivariant_local_geom_k),
                    local_geom_heads=int(equivariant_local_geom_heads),
                    local_geom_head_dim=int(equivariant_local_geom_head_dim),
                    local_geom_num_blocks=int(equivariant_local_geom_num_blocks),
                    local_geom_ffn_mult=int(equivariant_local_geom_ffn_mult),
                )
            else:
                self.encoder = AttnIPABackbone(
                    in_dim=in_dim, dim=dim, num_heads=n_heads,
                    num_layers=enc_layers, hybrid_norm=hybrid_norm,
                    window_size=window_size, window_wrap=window_wrap,
                )
            self.input_net = None
        elif use_gatr_backbone:
            # Lazy import — only required when this flag is set.
            from src.models.Mask3D.gatr_backbone import GATrBackbone
            self.input_net = GATrBackbone(
                dim=dim,
                num_blocks=gatr_blocks,
                hidden_mv_channels=gatr_hidden_mv,
                hidden_s_channels=gatr_hidden_s,
            )
            # GATr replaces BOTH the input embedding and the encoder —
            # there's nothing left for a separate transformer encoder to do.
            self.encoder = None
        elif per_subsystem_input:
            if num_subsystems != 4 or subsystem_offset != 1:
                raise ValueError(
                    "per_subsystem_input=True currently requires "
                    "num_subsystems=4 and subsystem_offset=1 (CLD layout: "
                    "tracker / ecal / hcal / muon, with hit_type=0 = noise)."
                )
            self.input_net = PerSubsystemInputNet(
                dim=dim, n_oh=n_oh,
                num_subsystems=num_subsystems,
                subsystem_offset=subsystem_offset,
            )
            self.encoder = Encoder(
                dim=dim, num_heads=n_heads, num_layers=enc_layers,
                hybrid_norm=hybrid_norm,
                window_size=window_size, window_wrap=window_wrap,
            )
        else:
            self.input_net = InputNet(in_dim=in_dim, dim=dim)
            self.encoder = Encoder(
                dim=dim, num_heads=n_heads, num_layers=enc_layers,
                hybrid_norm=hybrid_norm,
                window_size=window_size, window_wrap=window_wrap,
            )
        self.window_size = window_size
        if self.use_ipa_decoder:
            from src.models.Mask3D.ipa_decoder import IPADecoder
            self.decoder = IPADecoder(
                dim=dim, num_heads=n_heads,
                num_queries=num_queries, num_layers=dec_layers,
                num_classes=1,
                mask_threshold=ipa_mask_threshold,
                mask_direction_aware=ipa_mask_direction_aware,
                coord_augmented_keys=ipa_coord_augmented_keys,
                dynamic_query_source=dynamic_query_source,
                track_seed_bonus=track_seed_bonus,
                track_hit_type=track_hit_type,
                cross_attn_mode=cross_attn_mode,
                kmeans_value_proj=kmeans_value_proj,
                kmeans_respect_attn_mask=kmeans_respect_attn_mask,
            )
        else:
            self.decoder = MaskFormerDecoder(
                dim=dim, num_heads=n_heads,
                num_queries=num_queries, num_layers=dec_layers,
                num_classes=1,
                bidirectional_ca=bidirectional_ca,
                num_subsystems=num_subsystems,
                subsystem_offset=subsystem_offset,
                hybrid_norm=hybrid_norm,
                share_decoder_heads=share_decoder_heads,
                dynamic_query_source=dynamic_query_source,
                track_seed_bonus=track_seed_bonus,
                track_hit_type=track_hit_type,
                use_gmm_query=use_gmm_query,
                gmm_K=gmm_K,
                gmm_sigma_init=gmm_sigma_init,
                gmm_coord_range=gmm_coord_range,
                gmm_min_log_w=gmm_min_log_w,
                gmm_sigma_long=gmm_sigma_long,
                gmm_sigma_short=gmm_sigma_short,
                gmm_init_rotation=gmm_init_rotation,
                gmm_dynamic=gmm_dynamic,
                gmm_dynamic_head_hidden_mult=gmm_dynamic_head_hidden_mult,
                gmm_dynamic_per_layer=gmm_dynamic_per_layer,
                gmm_warmup_hard_mask=gmm_warmup_hard_mask,
                gmm_warmup_topk=gmm_warmup_topk,
                use_query_anchor=use_query_anchor,
                query_anchor_sigma_init=query_anchor_sigma_init,
                query_anchor_per_layer=query_anchor_per_layer,
                cross_attn_mode=cross_attn_mode,
                kmeans_respect_attn_mask=kmeans_respect_attn_mask,
                kmeans_value_proj=kmeans_value_proj,
            )
        self.use_gmm_query = bool(use_gmm_query)
        self.gmm_warmup_steps = int(gmm_warmup_steps)
        self.gmm_coverage_weight = float(gmm_coverage_weight)
        self.gmm_bce_weight = float(gmm_bce_weight)
        self.dice_size_weighting = str(dice_size_weighting)

        # Stash for loss + inference paths.
        self.track_loss_weight = float(track_loss_weight)
        self.track_hit_type = int(track_hit_type)
        self.force_track_assignment = bool(force_track_assignment)
        self.track_mask_threshold = float(track_mask_threshold)
        # Eval/--predict hit-assignment thresholds (sweepable via CLI for the
        # clustering-completeness vs Pandora test). Fall back to the static
        # defaults / the ctor track threshold so training is unaffected.
        self.eval_mask_threshold = float(getattr(args, "eval_mask_threshold", 0.5))
        self.eval_track_mask_threshold = float(
            getattr(args, "eval_track_mask_threshold", self.track_mask_threshold)
        )

        self.loss_weights = {
            "mask_bce": mask_bce_weight,
            "mask_dice": mask_dice_weight,
            "obj_bce": obj_bce_weight,
            "null_weight": null_weight,
        }
        self.aux_layer_weight = aux_layer_weight
        self.mask_loss_type = mask_loss_type
        self.focal_gamma = focal_gamma
        self.mask_cost_use_classification = mask_cost_use_classification
        self.obj_bce_cost_weight = obj_bce_cost_weight
        self.per_subsystem_loss = per_subsystem_loss
        self.subsystem_offset = subsystem_offset

        # Hungarian matcher (mirrors hepattn). Sequential by default — for our
        # batch sizes scipy is sub-ms per call and pool overhead dominates;
        # flip parallel_solver=True if you scale up.
        self.matcher = Matcher(parallel_solver=False)

        # Energy-correction bolt-on (only used when --correction is set)
        if getattr(self.args, "correction", False):
            self.ec_adapter = ECAdapter(dim=dim)
            self.energy_correction = EnergyCorrection(self)
            self.ec_model_wrapper_charged = self.energy_correction.model_charged
            self.ec_model_wrapper_neutral = self.energy_correction.model_neutral
            self.pids_neutral = self.energy_correction.pids_neutral
            self.pids_charged = self.energy_correction.pids_charged
        else:
            self.pids_neutral = []
            self.pids_charged = []

        self._fix_clusters_class = []
        if getattr(self.args, "fix_ch", False):
            self._fix_clusters_class.append(1)
        if getattr(self.args, "fix_neutrals", False):
            self._fix_clusters_class.append(2)
        if getattr(self.args, "fix_photons", False):
            self._fix_clusters_class.append(3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _scatter_flat_to_padded(self, x_flat, batch_ids, local_idx, key_valid):
        """Scatter a flat (total_hits, D) tensor into padded (B, N_max, D)."""
        B, N_max = key_valid.shape
        D = x_flat.size(-1)
        out = x_flat.new_zeros(B, N_max, D)
        out[batch_ids, local_idx] = x_flat
        return out

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        targets = build_targets(g, y=y, ILD=getattr(self.args, "ILD", False))

        # ---------------- IPA decoder path ----------------
        # AttnIPABackbone returns per-hit (feats, raw_xyz); IPADecoder carries
        # per-query frames and emits the same per-layer {mask_logits,
        # cls_logits} schema + a `queries` (= q_s) tensor, so everything
        # downstream (correction / predict / EC adapter) is identical.
        if self.use_ipa_decoder:
            # Backbone is stored as `self.encoder` (AttnIPABackbone) so the
            # checkpoint keys line up — see __init__ note.
            feats_flat, points_flat = self.encoder(
                targets["feats_flat"], targets["seq_lens"],
            )
            keys_padded = self._scatter_flat_to_padded(
                feats_flat, targets["batch_ids"], targets["local_idx"],
                targets["key_valid"],
            )
            points_padded = self._scatter_flat_to_padded(
                points_flat, targets["batch_ids"], targets["local_idx"],
                targets["key_valid"],
            )
            per_layer, queries, T, t = self.decoder(
                keys_padded, points_padded, targets["key_valid"],
                hit_subsystem=targets.get("hit_subsystem"),
            )
            # Stash for FAPE-style frame supervision in mask3d_loss.
            self._last_T = T
            self._last_t = t
            self._last_points_padded = points_padded
            attn_bias = None
            return self._post_decoder(
                g, y, targets, per_layer, queries, attn_bias,
                return_train=return_train,
            )

        # Encoder runs flat — no padding waste.
        if self.use_gatr_backbone:
            # GATr produces per-hit `(total_hits, dim)` features directly.
            # No separate encoder run — the GATr blocks ARE the encoder.
            keys_flat = self.input_net(g)
        else:
            if self.per_subsystem_input:
                x_flat = self.input_net(targets["feats_flat"], targets["hit_type_flat"])
            else:
                x_flat = self.input_net(targets["feats_flat"])
            phi_flat = None
            if self.window_size:
                # feats_flat[:, :3] = (x, y, z) in the same scaled units used
                # by the input net; phi is scale-invariant (atan2 of y/x).
                pos_xy = targets["feats_flat"][:, :2]
                phi_flat = torch.atan2(pos_xy[:, 1], pos_xy[:, 0])
            keys_flat = self.encoder(x_flat, targets["seq_lens"], phi=phi_flat)

        # Scatter to padded for the decoder cross-attn (queries are fixed-N,
        # so the padded cost is (B, N_q, N_max) — small).
        keys_padded = self._scatter_flat_to_padded(
            keys_flat, targets["batch_ids"], targets["local_idx"], targets["key_valid"]
        )
        # GMM-query path AND v5 anchor path both need per-key xyz in the
        # padded layout. `feats_flat[:, :3]` is already in the scaled-metres
        # frame that build_targets emits, so we just scatter it the same way.
        pos_padded = None
        if self.use_gmm_query or getattr(self.decoder, "use_query_anchor", False):
            pos_flat = targets["feats_flat"][:, :3]
            pos_padded = self._scatter_flat_to_padded(
                pos_flat, targets["batch_ids"], targets["local_idx"],
                targets["key_valid"],
            )
        # Pass per-hit subsystem to the decoder so per-subsystem mask heads
        # can pick the right constituent projection per key.
        # Hard GMM warm-up: while global_step < gmm_warmup_steps (and we're in
        # training mode), disable the binary mask-attention so only the GMM
        # spatial bias gates the cross-attention. The Gaussian means have to
        # migrate to real-hit regions or the loss can't go down. After the
        # gate flips, mask-attention re-engages and both prior together drive
        # cluster prediction.
        disable_mask_attn = (
            self.training
            and self.use_gmm_query
            and self.gmm_warmup_steps > 0
            and int(getattr(self, "global_step", 0)) < self.gmm_warmup_steps
        )
        per_layer, queries, attn_bias = self.decoder(
            keys_padded, targets["key_valid"],
            hit_subsystem=targets.get("hit_subsystem"),
            pos_padded=pos_padded,
            disable_mask_attention=disable_mask_attn,
        )
        return self._post_decoder(
            g, y, targets, per_layer, queries, attn_bias,
            return_train=return_train,
        )

    def _post_decoder(self, g, y, targets, per_layer, queries, attn_bias,
                      return_train=False):
        """Shared post-decoder path: EC/correction branch + 4-tuple return.

        Both the standard (MaskFormerDecoder) and IPA (IPADecoder) paths
        funnel through here so the energy-correction / predict pipeline is
        identical regardless of which decoder produced `per_layer`/`queries`.
        """
        if getattr(self.args, "correction", False):
            # Stash the raw mask outputs so validation_step can build mask-based
            # labels for downstream evaluation (the EC pipeline still consumes
            # the OC-style x_oc tensor for energy correction).
            self._last_mask_logits = per_layer[-1]["mask_logits"]
            self._last_cls_logits = per_layer[-1]["cls_logits"]
            self._last_key_valid = targets["key_valid"]
            self._last_queries = queries
            x_oc = self.ec_adapter(
                per_layer[-1]["mask_logits"], queries,
                targets["batch_ids"], targets["local_idx"], targets["key_valid"],
            )

            # Optionally bypass the EC's redundant DPC clustering by passing
            # Mask3D's per-(query, hit) argmax labels through. coords/beta on
            # the graph still come from `x_oc` (via obtain_clustering_for_
            # matched_showers) so the EC heads' input features are unchanged.
            precomputed_labels = None
            if getattr(self.args, "mask3d_use_mask_labels", False):
                ml = per_layer[-1]["mask_logits"]               # (B, N_q, N_max)
                cl = per_layer[-1]["cls_logits"]                # (B, N_q, 1) or (B, N_q)
                if cl.dim() == 3:
                    cl = cl.squeeze(-1)
                kv = targets["key_valid"]                       # (B, N_max)
                hs = targets.get("hit_subsystem")               # (B, N_max) or None
                B = ml.size(0)
                g_list = dgl.unbatch(g)
                _truth_tracks = getattr(self.args, "truth_tracking", False)
                per_event = []
                for i in range(B):
                    n_i = int(kv[i].sum().item())
                    ht_i = hs[i, :n_i] if hs is not None else None
                    lab_i = labels_from_masks(
                        ml[i, :, :n_i], cl[i],
                        hit_type=ht_i,
                        mask_threshold=self.eval_mask_threshold,
                        track_mask_threshold=self.eval_track_mask_threshold,
                        track_hit_type=getattr(self, "track_hit_type", 1),
                        force_track_assignment=getattr(
                            self, "force_track_assignment", True,
                        ),
                    )
                    if not _truth_tracks:
                        # Drop tracks whose momentum is inconsistent with the
                        # cluster's calo energy, HERE — before the EC consumes
                        # the labels — so EC corrections and inference both see
                        # the same post-removal labels. Doing this AFTER the EC
                        # (e.g. in utils_training's mask3d branch) desyncs the
                        # per-shower arrays → pandas length error.
                        lab_i, _ = remove_bad_tracks_from_cluster_v1(
                            g_list[i], lab_i
                        )
                        # Re-densify: remove_bad_tracks can empty a cluster and
                        # leave a label gap, which crashes match_showers
                        # downstream (sizes by len(unique) but indexes by
                        # max+1).
                        _, lab_i = torch.unique(lab_i, return_inverse=True)
                    per_event.append(lab_i)
                # Concatenate into a flat (total_hits,) tensor in graph node
                # order — `obtain_clustering_for_matched_showers` slices it
                # per event using `batch_g.batch_num_nodes()`.
                precomputed_labels = (
                    torch.cat(per_event, dim=0) if per_event else None
                )

            result = self.energy_correction.forward_correction(
                g, x_oc, y, return_train,
                precomputed_labels=precomputed_labels,
            )
            return result

        # 4th slot used to be a placeholder; now carries the GMM `attn_bias`
        # (per-(query, hit) log-density, None when use_gmm_query=False) so
        # mask3d_loss can compute its coverage-NLL term against matched GT.
        return per_layer, targets, queries, attn_bias

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]

        if not getattr(self.args, "correction", False):
            per_layer, targets, queries, attn_bias = self(batch_g, y, batch_idx)
            loss, parts = mask3d_loss(
                per_layer, targets, self.matcher,
                weights=self.loss_weights,
                aux_layer_weight=self.aux_layer_weight,
                mask_loss_type=self.mask_loss_type,
                focal_gamma=self.focal_gamma,
                mask_cost_use_classification=self.mask_cost_use_classification,
                obj_bce_cost_weight=self.obj_bce_cost_weight,
                per_subsystem_loss=self.per_subsystem_loss,
                subsystem_offset=self.subsystem_offset,
                track_loss_weight=self.track_loss_weight,
                track_hit_type=self.track_hit_type,
                attn_bias=attn_bias,
                gmm_coverage_weight=self.gmm_coverage_weight,
                gmm_bce_weight=self.gmm_bce_weight,
                dice_size_weighting=self.dice_size_weighting,
                T_final=getattr(self, "_last_T", None),
                t_final=getattr(self, "_last_t", None),
                pos_padded=getattr(self, "_last_points_padded", None),
                frame_translation_weight=self.frame_translation_weight,
                frame_rotation_weight=self.frame_rotation_weight,
                energy_weight_mode=self.energy_weight_mode,
                energy_weight_scale=self.energy_weight_scale,
                energy_completeness_weight=self.energy_completeness_weight,
                recall_weight=self.recall_weight,
                recall_tau=self.recall_tau,
                recall_E_alpha=self.recall_E_alpha,
            )
            if self.trainer.is_global_zero and (batch_idx % 10) == 0:
                _log = {
                    "loss": loss.item(),
                    "loss_mask_bce": float(parts["mask_bce"]),
                    "loss_mask_dice": float(parts["mask_dice"]),
                    "loss_obj_bce": float(parts["obj_bce"]),
                }
                if "gmm_coverage" in parts:
                    _log["loss_gmm_coverage"] = float(parts["gmm_coverage"])
                if "gmm_bce" in parts:
                    _log["loss_gmm_bce"] = float(parts["gmm_bce"])
                if "energy_completeness" in parts:
                    _log["loss_energy_completeness"] = float(parts["energy_completeness"])
                if "recall" in parts:
                    _log["loss_recall"] = float(parts["recall"])
                for _k in ("Erecall_highE", "Erecall_lowE", "dice_highE", "n_highE",
                           "evt_E_frac_soft", "evt_E_frac_hard", "evt_E_frac_hard_min"):
                    if _k in parts:
                        _log[_k] = float(parts[_k])
                wandb.log(_log)
            # Cluster-space analogue: every 500 global steps, log a side-by-side
            # 3-D scatter of one event's hits coloured by predicted cluster ID
            # vs. true particle ID.
            if (
                self.trainer.is_global_zero
                and getattr(self.args, "log_wandb", False)
                and self.global_step > 0
                and (self.global_step % 500) == 0
            ):
                try:
                    fig = plot_pred_vs_true_clusters_fig(
                        batch_g,
                        per_layer[-1]["mask_logits"],
                        per_layer[-1]["cls_logits"],
                        targets["key_valid"],
                        event_idx=0,
                    )
                    wandb.log({"clusters_pred_vs_true": fig,
                               "global_step": self.global_step})
                except Exception as e:
                    print(f"[mask3d_model] cluster-viz failed at step "
                          f"{self.global_step}: {e}")
        else:
            # EC training path (mirrors GATr model)
            #
            # NOTE: this used to pass return_train=True, but that selects
            # forward_correction's 8-tuple branch while get_loss below unpacks
            # ELEVEN values — `ValueError: not enough values to unpack
            # (expected 11, got 8)` on the first batch (job 45146645). The GATr
            # model gets this right by simply not passing the flag
            # (Gatr_pf_e_noise.training_step), so return_train stays False and
            # forward_correction returns the 11-tuple get_loss consumes.
            result = self(batch_g, y, batch_idx)
            self.energy_correction.global_step = self.global_step
            fixed = self.current_epoch != 0
            (
                loss_EC, _, loss_neutral_pid, loss_charged_pid, loss_score, self.stats
            ) = self.energy_correction.get_loss(batch_g, y, result, self.stats, fixed)
            loss = loss_EC + loss_neutral_pid + loss_charged_pid

        self.loss_final += float(loss.item() if hasattr(loss, "item") else loss)
        self.number_b += 1
        return loss

    def validation_step(self, batch, batch_idx):
        self.create_paths()
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]

        if getattr(self.args, "correction", False):
            result = self(batch_g, y, 1, use_gt_clusters=getattr(self.args, "use_gt_clusters", False))
            outputs = self.energy_correction.get_validation_step_outputs(batch_g, y, result)
            e_cor1, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels = outputs
            e_cor = e_cor1
            if not getattr(self.args, "predict", False):
                return
            # Use mask-based labels (same convention as the non-correction path)
            # paired with EC outputs so the shower frames carry corrected
            # energies / positions / PIDs.
            (
                df_batch_pandora,
                df_batch1,
                self.total_number_events,
            ) = create_and_store_graph_output_mask3d(
                batch_g,
                self._last_mask_logits,
                self._last_cls_logits,
                self._last_key_valid,
                y,
                0, batch_idx, 0,
                path_save=self.show_df_eval_path,
                store=True, predict=True, e_corr=e_cor,
                tracks=self.args.tracks,
                total_number_events=self.total_number_events,
                pred_pos=pred_pos, pred_ref_pt=pred_ref_pt, pred_pid=pred_pid,
                use_gt_clusters=getattr(self.args, "use_gt_clusters", False),
                fix_clusters_class=self._fix_clusters_class,
                pids_neutral=self.pids_neutral, pids_charged=self.pids_charged,
                number_of_fakes=num_fakes, extra_features=extra_features,
                fakes_labels=fakes_labels,
                pandora_available=self.args.pandora,
                truth_tracks=getattr(self.args, "truth_tracking", False),
                mask_threshold=self.eval_mask_threshold,
                track_mask_threshold=self.eval_track_mask_threshold,
                force_track_assignment=self.force_track_assignment,
                track_hit_type=self.track_hit_type,
                queries=getattr(self, "_last_queries", None),
                gmm_module=self.decoder.gmm if self.use_gmm_query else None,
            )
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)
            return
        else:
            # Plain Mask3D path: forward + loss for logging, and (when --predict)
            # convert mask outputs into per-hit cluster labels for the eval frames.
            per_layer, targets, queries, attn_bias = self(batch_g, y, 1)
            loss, parts = mask3d_loss(
                per_layer, targets, self.matcher,
                weights=self.loss_weights,
                aux_layer_weight=self.aux_layer_weight,
                mask_loss_type=self.mask_loss_type,
                focal_gamma=self.focal_gamma,
                mask_cost_use_classification=self.mask_cost_use_classification,
                obj_bce_cost_weight=self.obj_bce_cost_weight,
                per_subsystem_loss=self.per_subsystem_loss,
                subsystem_offset=self.subsystem_offset,
                track_loss_weight=self.track_loss_weight,
                track_hit_type=self.track_hit_type,
                attn_bias=attn_bias,
                gmm_coverage_weight=self.gmm_coverage_weight,
                gmm_bce_weight=self.gmm_bce_weight,
                dice_size_weighting=self.dice_size_weighting,
                T_final=getattr(self, "_last_T", None),
                t_final=getattr(self, "_last_t", None),
                pos_padded=getattr(self, "_last_points_padded", None),
                frame_translation_weight=self.frame_translation_weight,
                frame_rotation_weight=self.frame_rotation_weight,
                energy_weight_mode=self.energy_weight_mode,
                energy_weight_scale=self.energy_weight_scale,
                energy_completeness_weight=self.energy_completeness_weight,
                recall_weight=self.recall_weight,
                recall_tau=self.recall_tau,
                recall_E_alpha=self.recall_E_alpha,
            )
            if self.trainer.is_global_zero and (batch_idx % 10) == 0:
                _log = {
                    "val_loss": float(loss.item()),
                    "val_mask_bce": float(parts["mask_bce"]),
                    "val_mask_dice": float(parts["mask_dice"]),
                    "val_obj_bce": float(parts["obj_bce"]),
                }
                if "gmm_coverage" in parts:
                    _log["val_gmm_coverage"] = float(parts["gmm_coverage"])
                if "gmm_bce" in parts:
                    _log["val_gmm_bce"] = float(parts["gmm_bce"])
                if "energy_completeness" in parts:
                    _log["val_energy_completeness"] = float(parts["energy_completeness"])
                if "recall" in parts:
                    _log["val_recall"] = float(parts["recall"])
                for _k in ("Erecall_highE", "Erecall_lowE", "dice_highE", "n_highE",
                           "evt_E_frac_soft", "evt_E_frac_hard", "evt_E_frac_hard_min"):
                    if _k in parts:
                        _log["val_" + _k] = float(parts[_k])
                wandb.log(_log)
            if not getattr(self.args, "predict", False):
                return
            mask_logits = per_layer[-1]["mask_logits"]
            cls_logits = per_layer[-1]["cls_logits"]

            # --- optional matched-query frame dump (IPA alignment check) ---
            _qf_N = int(getattr(self.args, "dump_query_frames", 0) or 0)
            if (
                _qf_N > 0
                and getattr(self, "_last_T", None) is not None
                and getattr(self, "_qf_n", 0) < _qf_N
            ):
                from src.models.Mask3D.loss import _dice_pairwise, _cls_cost
                self._qf_n = getattr(self, "_qf_n", 0)
                T_ = self._last_T
                t_ = self._last_t
                pts = self._last_points_padded
                gm = targets["gt_mask"]; gv = targets["gt_valid"]
                kv = targets["key_valid"]
                Bq, _, Np = mask_logits.size(0), 0, gv.size(1)
                cl = cls_logits.squeeze(-1) if cls_logits.dim() == 3 else cls_logits
                cost = (
                    self.loss_weights["mask_dice"]
                    * _dice_pairwise(mask_logits, gm, kv)
                    + self.obj_bce_cost_weight * _cls_cost(cl[..., None], gv)
                )
                perm = self.matcher(
                    cost, object_valid_mask=gv
                ).to(mask_logits.device)                       # (B, N_q)
                matched_q = perm[:, :Np]                        # (B, N_p)
                qdir = os.path.join(self.args.model_prefix, "query_frames")
                os.makedirs(qdir, exist_ok=True)
                for b in range(Bq):
                    if self._qf_n >= _qf_N:
                        break
                    torch.save(
                        {
                            "points_padded": pts[b].detach().cpu(),
                            "key_valid": kv[b].detach().cpu(),
                            "gt_mask": gm[b].detach().cpu(),
                            "gt_valid": gv[b].detach().cpu(),
                            "T": T_[b].detach().cpu(),
                            "t": t_[b].detach().cpu(),
                            "matched_q": matched_q[b].detach().cpu(),
                            "target_E": targets["target_E"][b].detach().cpu(),
                            "particle_axis": targets["particle_axis"][b].detach().cpu()
                            if targets.get("particle_axis") is not None else None,
                            "target_angle": targets["target_angle"][b].detach().cpu()
                            if targets.get("target_angle") is not None else None,
                            "target_mom": targets["target_mom"][b].detach().cpu()
                            if targets.get("target_mom") is not None else None,
                            "target_pid": targets["target_pid"][b].detach().cpu()
                            if targets.get("target_pid") is not None else None,
                            "particle_ref": targets["particle_ref"][b].detach().cpu()
                            if targets.get("particle_ref") is not None else None,
                            "has_track": targets["has_track"][b].detach().cpu()
                            if targets.get("has_track") is not None else None,
                            "hit_subsystem": targets["hit_subsystem"][b].detach().cpu()
                            if targets.get("hit_subsystem") is not None else None,
                        },
                        os.path.join(qdir, f"ev_{self._qf_n:04d}.pt"),
                    )
                    self._qf_n = getattr(self, "_qf_n", 0) + 1
                print(f"[query_frames] dumped {self._qf_n}/{_qf_N} -> {qdir}")

            (
                df_batch_pandora,
                df_batch1,
                self.total_number_events,
            ) = create_and_store_graph_output_mask3d(
                batch_g, mask_logits, cls_logits, targets["key_valid"], y,
                0, batch_idx, 0,
                path_save=self.show_df_eval_path,
                store=True, predict=True, e_corr=None,
                tracks=self.args.tracks,
                total_number_events=self.total_number_events,
                pred_pos=None, pred_ref_pt=None, pred_pid=None,
                use_gt_clusters=getattr(self.args, "use_gt_clusters", False),
                fix_clusters_class=self._fix_clusters_class,
                pids_neutral=self.pids_neutral, pids_charged=self.pids_charged,
                number_of_fakes=None, extra_features=None,
                fakes_labels=None,
                pandora_available=self.args.pandora,
                truth_tracks=getattr(self.args, "truth_tracking", False),
                mask_threshold=self.eval_mask_threshold,
                track_mask_threshold=self.eval_track_mask_threshold,
                force_track_assignment=self.force_track_assignment,
                track_hit_type=self.track_hit_type,
                queries=queries,
                gmm_module=self.decoder.gmm if self.use_gmm_query else None,
            )
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)
            return

    # ------------------------------------------------------------------
    # Misc plumbing (mirrors GATr model)
    # ------------------------------------------------------------------

    def create_paths(self):
        show_df_eval_path = os.path.join(self.args.model_prefix, "showers_df_evaluation")
        self.show_df_eval_path = show_df_eval_path

    def on_train_epoch_end(self):
        denom = max(self.number_b, 1)
        self.log("train_loss_epoch", self.loss_final / denom)

    def on_train_epoch_start(self):
        self.loss_final = 0.0
        self.number_b = 0
        if self.current_epoch == 0:
            self.stats = {"counts": {}, "counts_pid_neutral": {}, "counts_pid_charged": {}}

    def on_validation_epoch_start(self):
        self.total_number_events = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and getattr(self.args, "predict", False):
            from src.layers.inference_oc import store_at_batch_end
            import pandas as pd
            if self.args.pandora and len(self.df_showers_pandora):
                self.df_showers_pandora = pd.concat(self.df_showers_pandora)
            else:
                self.df_showers_pandora = []
            if len(self.df_showes_db):
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(self.args.model_prefix, "showers_df_evaluation")
                              + "/" + self.args.name_output,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0, predict=True, store=True,
                    pandora_available=self.args.pandora,
                )
        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def configure_optimizers(self):
        """Optimizer + scheduler.

        Defaults match the reconstruct-anything / hepattn CLD recipe:
            * Lion(lr=args.start_lr, weight_decay=1e-5)
            * OneCycleLR(max_lr=args.start_lr, pct_start=0.01,
                         div_factor=4.0, final_div_factor=10.0)

        `--optimizer` (`adam` / `adamw` / `ranger`) and `--lr-scheduler`
        (`reduceplateau` / `cosine`) override these defaults.
        """
        # ----- optimizer -----
        opt_name = (getattr(self.args, "optimizer", None) or "lion").lower()
        lr = self.args.start_lr
        weight_decay = 1e-5

        # Hard-fail on missing optimizers instead of silent AdamW fallback —
        # the previous behavior masked the fact that BSC's env doesn't have
        # `lion_pytorch` / `pytorch_optimizer` installed (we observed this
        # in run vhc0nms5: `--optimizer ranger` was requested, fell back to
        # AdamW silently, but wandb metadata listed ranger). Failing fast
        # surfaces the env mismatch.
        if opt_name == "lion":
            try:
                from lion_pytorch import Lion
            except ImportError as e:
                raise ImportError(
                    "[mask3d_model] --optimizer lion requested but "
                    "`lion_pytorch` is not installed. Either `pip install "
                    "lion-pytorch` or pass --optimizer adamW."
                ) from e
            optimizer = Lion(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name in ("adam", "adamw"):
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay,
            )
        elif opt_name == "ranger":
            try:
                from pytorch_optimizer import Ranger
            except ImportError as e:
                raise ImportError(
                    "[mask3d_model] --optimizer ranger requested but "
                    "`pytorch_optimizer.Ranger` is not installed. Either "
                    "`pip install pytorch-optimizer` or pass --optimizer adamW."
                ) from e
            optimizer = Ranger(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown --optimizer {opt_name!r}. "
                f"Choices: lion, adam, adamw, radam, ranger"
            )

        # ----- scheduler -----
        sched_raw = getattr(self.args, "lr_scheduler", None) or "onecycle"
        sched_norm = sched_raw.lower().replace("_", "").replace("-", "")

        # Legacy CLI choices ("flat+decay", "flat+linear", "flat+cos",
        # "steps") are remapped to OneCycleLR — that's reconstruct-anything's
        # default and the closest analogue. The Mask3D model is set up for
        # one-cycle's start/peak/end shape, not for plateau-then-decay.
        _LEGACY_TO_ONECYCLE = {"flatdecay", "flatlinear", "flatcos", "steps"}

        if sched_norm in _LEGACY_TO_ONECYCLE:
            print(f"[mask3d_model] WARN: --lr-scheduler {sched_raw!r} is "
                  f"not natively supported by the Mask3D model; using "
                  f"OneCycleLR instead. Pass `--lr-scheduler one-cycle` "
                  f"to silence this.")
            sched_name = "onecycle"
        elif sched_norm == "none":
            sched_name = "none"
        else:
            sched_name = sched_norm

        if sched_name in ("onecycle", "onecyclelr"):
            # reconstruct-anything CLD: peak max_lr ramped from initial_lr =
            # max_lr/4 over the first 1% of steps, then cosine-decayed to
            # final_lr = initial_lr/10. We treat `args.start_lr` as max_lr
            # (the peak of the one-cycle), so e.g. --start-lr 4e-5 reproduces
            # their schedule exactly.
            steps_per_epoch = (
                getattr(self.args, "train_batches", None)
                or getattr(self.args, "steps_per_epoch", None)
                or 12000
            )
            total_steps = max(1, int(self.args.num_epochs * steps_per_epoch))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=0.01,
                div_factor=4.0,
                final_div_factor=10.0,
            )
            sched_cfg = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif sched_name in ("reduceplateau", "plateau", "reducelronplateau"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5,
            )
            sched_cfg = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_loss_epoch",
                "frequency": 1,
            }
        elif sched_name == "cosine":
            # Legacy CosineAnnealingThenFixedScheduler — kept as opt-in for
            # backward compatibility with old runs / checkpoints.
            scheduler = CosineAnnealingThenFixedScheduler(
                optimizer, T_max=int(36400 * 2), fixed_lr=1e-5,
            )
            sched_cfg = {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "train_loss_epoch",
                "frequency": 1,
            }
        elif sched_name == "none":
            # No scheduler — return just the optimizer (constant LR).
            print(f"[mask3d_model] optimizer={opt_name}  scheduler=none  start_lr={lr}")
            return optimizer
        else:
            print(f"[mask3d_model] WARN: --lr-scheduler {sched_raw!r} not "
                  f"recognised; falling back to OneCycleLR. Choices: "
                  f"onecycle, reduceplateau, cosine, none.")
            steps_per_epoch = (
                getattr(self.args, "train_batches", None)
                or getattr(self.args, "steps_per_epoch", None)
                or 12000
            )
            total_steps = max(1, int(self.args.num_epochs * steps_per_epoch))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=0.01,
                div_factor=4.0,
                final_div_factor=10.0,
            )
            sched_cfg = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        self.scheduler = sched_cfg["scheduler"]
        print(f"[mask3d_model] optimizer={opt_name}  "
              f"scheduler={sched_name}  start_lr={lr}")
        return {"optimizer": optimizer, "lr_scheduler": sched_cfg}
