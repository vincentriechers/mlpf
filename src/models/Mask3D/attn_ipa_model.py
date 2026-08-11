"""Lightning module for the Attn-IPA experiment (variant C).

Same structure as `gatr_ipa_model.py` but with the GATr backbone swapped
for an `InputNet` + standard transformer `Encoder` pair (`AttnIPABackbone`).
The decoder side is identical — same `IPADecoder` with frame-carrying
queries + the existing FAPE-style frame supervision in `mask3d_loss`.

Why a separate variant: GATr destroys some of the hit-level geometric
structure via its `extract_point` output (the "point" used in the IPA
geometric term becomes a learned position). Variants A/B keep GATr;
variant B already pins the IPA point to raw xyz. Variant C drops GATr
entirely so the encoder is just self-attention over hits — no equivariant
bias — and we rely on:

  1. `InputNet`'s Fourier positional encoding on raw xyz to inject
     position into the feature stream.
  2. The IPA decoder's geometric mask term (`T_iᵀ(x_j − t_i)` or
     `‖T·t − x‖²`) for query-relative spatial reasoning.
  3. Optionally, `ipa_coord_augmented_keys=True` so the mask head's
     feature-similarity branch ALSO reads `x_j` directly (compensates
     for the lack of GATr's intrinsic position channel).
"""
import os

import lightning as L
import torch
import wandb

from src.models.Mask3D.attn_ipa_backbone import AttnIPABackbone
from src.models.Mask3D.equivariant_attn_ipa_backbone import (
    EquivariantAttnIPABackbone,
)
from src.models.Mask3D.ipa_decoder import IPADecoder
from src.models.Mask3D.loss import mask3d_loss
from src.models.Mask3D.matcher import Matcher
from src.models.Mask3D.targets import build_targets


class AttnIPAModel(L.LightningModule):
    """Attention backbone + IPA decoder; reuses Mask3D's matcher / loss / hooks."""

    def __init__(
        self,
        args,
        dev,
        dim=256,
        n_heads=8,
        enc_layers=8,
        dec_layers=8,
        num_queries=320,
        # Encoder knobs
        hybrid_norm=False,
        window_size=1024,
        window_wrap=True,
        # Mask / loss recipe — same defaults as gatr_ipa_model.py
        mask_bce_weight=0.25,
        mask_dice_weight=0.75,
        obj_bce_weight=0.1,
        null_weight=0.25,
        aux_layer_weight=1.0,
        mask_loss_type="bce",
        focal_gamma=2.0,
        mask_cost_use_classification=False,
        obj_bce_cost_weight=1.0,
        track_loss_weight=1.0,
        track_hit_type=1,
        track_mask_threshold=0.1,
        force_track_assignment=False,
        dice_size_weighting="none",
        # IPA decoder knobs.
        ipa_mask_direction_aware=False,
        # Mask2Former binary attn-gate threshold inside the IPA decoder
        # (ipa_decoder.py: attn_mask = sigmoid(mask) >= thr). Lower → the
        # next decoder layer's cross-attention can reach more peripheral
        # hits → better recall on large/extended showers (the diagnosed
        # halo-drop). Default 0.5 (backward-compatible).
        ipa_mask_threshold=0.5,
        # Variant C-specific: concat x_j to the keys before the mask-logit
        # feature-similarity projection. Compensates for the absence of GATr
        # by making the mask head's feature path coordinate-aware in addition
        # to its (already-present) geometric branch.
        ipa_coord_augmented_keys=False,
        # Initial-query strategy ("static" or "encoder"). Default "encoder"
        # mirrors baseline Mask3D so layer-0 mask logits are event-conditioned.
        dynamic_query_source="encoder",
        track_seed_bonus=6e4,
        # k-Max-DeepLab cross-attention variant in the IPA decoder.
        cross_attn_mode="softmax",
        kmeans_value_proj=False,
        kmeans_respect_attn_mask=False,
        # FAPE-style frame supervision for matched charged (E ≥ 1 GeV) pairs.
        frame_translation_weight=0.0,
        frame_rotation_weight=0.0,
        # Per-hit energy weighting of the mask losses (none / log / linear).
        energy_weight_mode="none",
        energy_weight_scale=1.0,
        # Differentiable energy-completeness loss (relative-L1 on the soft
        # mask's collected energy vs true shower E). 0.0 = disabled.
        energy_completeness_weight=0.0,
        # Per-truth recall (max-over-queries coverage) loss.
        recall_weight=0.0,
        recall_tau=5.0,
        recall_E_alpha=0.5,
        # Backbone selection / equivariance.
        # When True, use `EquivariantAttnIPABackbone` (drops absolute
        # position from per-hit input features, replaces it with
        # SO(2)_z-invariants (r, |z|) and a log(1+E) channel). Combined
        # with the equivariant IPA decoder this yields an end-to-end
        # SO(2)_z-equivariant pipeline (full SO(3) if `equivariant_symmetry`
        # is "SO(3)" AND the encoder windowing is disabled).
        equivariant_backbone=False,
        equivariant_symmetry="SO(2)_z",
        equivariant_local_geom=True,
        equivariant_local_geom_k=64,
        equivariant_local_geom_heads=4,
        equivariant_local_geom_head_dim=24,
        equivariant_local_geom_num_blocks=2,
        equivariant_local_geom_ffn_mult=2,
        **kwargs,
    ):
        super().__init__()
        self.strict_loading = False
        self.args = args
        self.dev = dev
        self.loss_final = 0.0
        self.number_b = 0
        self.dim = dim

        # Input feature width: pos(3) + hit_type_oh(5 or 6) + e(1) + p(1) + chi2(1)
        n_oh = 6 if getattr(args, "ILD", False) else 5
        in_dim = 3 + n_oh + 1 + 1 + 1
        self.in_dim = in_dim

        # Backbone: InputNet + Encoder → per-hit (feats, raw_xyz).
        # Toggle to the SO(*)-invariant version via `equivariant_backbone=True`.
        self.equivariant_backbone = bool(equivariant_backbone)
        self.equivariant_symmetry = str(equivariant_symmetry)
        if self.equivariant_backbone:
            self.encoder = EquivariantAttnIPABackbone(
                in_dim=in_dim,
                dim=dim,
                num_heads=n_heads,
                num_layers=enc_layers,
                hybrid_norm=hybrid_norm,
                window_size=window_size,
                window_wrap=window_wrap,
                n_oh=n_oh,
                symmetry=self.equivariant_symmetry,
                local_geom=bool(equivariant_local_geom),
                local_geom_k=int(equivariant_local_geom_k),
                local_geom_heads=int(equivariant_local_geom_heads),
                local_geom_head_dim=int(equivariant_local_geom_head_dim),
                local_geom_num_blocks=int(equivariant_local_geom_num_blocks),
                local_geom_ffn_mult=int(equivariant_local_geom_ffn_mult),
            )
        else:
            self.encoder = AttnIPABackbone(
                in_dim=in_dim,
                dim=dim,
                num_heads=n_heads,
                num_layers=enc_layers,
                hybrid_norm=hybrid_norm,
                window_size=window_size,
                window_wrap=window_wrap,
            )
        # Decoder: IPA-style with per-query frames.
        self.decoder = IPADecoder(
            dim=dim,
            num_heads=n_heads,
            num_queries=num_queries,
            num_layers=dec_layers,
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

        self.matcher = Matcher(parallel_solver=False)
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

        self.track_loss_weight = float(track_loss_weight)
        self.track_hit_type = int(track_hit_type)
        self.track_mask_threshold = float(track_mask_threshold)
        self.force_track_assignment = bool(force_track_assignment)
        self.dice_size_weighting = str(dice_size_weighting)
        self.frame_translation_weight = float(frame_translation_weight)
        self.frame_rotation_weight = float(frame_rotation_weight)
        self.energy_weight_mode = str(energy_weight_mode)
        self.energy_weight_scale = float(energy_weight_scale)
        self.energy_completeness_weight = float(energy_completeness_weight)
        self.recall_weight = float(recall_weight)
        self.recall_tau = float(recall_tau)
        self.recall_E_alpha = float(recall_E_alpha)

    @staticmethod
    def _scatter_flat_to_padded(x_flat, batch_ids, local_idx, key_valid):
        B, N_max = key_valid.shape
        D = x_flat.size(-1)
        out = x_flat.new_zeros(B, N_max, D)
        out[batch_ids, local_idx] = x_flat
        return out

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        targets = build_targets(g, y=y, ILD=getattr(self.args, "ILD", False))

        feats_flat, points_flat = self.encoder(
            targets["feats_flat"], targets["seq_lens"],
        )                                                                          # (N, dim), (N, 3)
        feats_padded = self._scatter_flat_to_padded(
            feats_flat, targets["batch_ids"], targets["local_idx"], targets["key_valid"],
        )                                                                          # (B, N_max, dim)
        points_padded = self._scatter_flat_to_padded(
            points_flat, targets["batch_ids"], targets["local_idx"], targets["key_valid"],
        )                                                                          # (B, N_max, 3)

        per_layer, q_s, T, t = self.decoder(
            feats_padded, points_padded, targets["key_valid"],
            hit_subsystem=targets.get("hit_subsystem"),
        )
        self._last_T = T
        self._last_t = t
        self._last_points_padded = points_padded
        return per_layer, targets, q_s, None

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        per_layer, targets, queries, _ = self(batch_g, y, batch_idx)
        loss, parts = mask3d_loss(
            per_layer, targets, self.matcher,
            weights=self.loss_weights,
            aux_layer_weight=self.aux_layer_weight,
            mask_loss_type=self.mask_loss_type,
            focal_gamma=self.focal_gamma,
            mask_cost_use_classification=self.mask_cost_use_classification,
            obj_bce_cost_weight=self.obj_bce_cost_weight,
            track_loss_weight=self.track_loss_weight,
            track_hit_type=self.track_hit_type,
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
                "loss": float(loss.item()),
                "loss_mask_bce": float(parts["mask_bce"]),
                "loss_mask_dice": float(parts["mask_dice"]),
                "loss_obj_bce": float(parts["obj_bce"]),
            }
            if "frame_translation" in parts:
                _log["loss_frame_translation"] = float(parts["frame_translation"])
            if "frame_rotation" in parts:
                _log["loss_frame_rotation"] = float(parts["frame_rotation"])
            if "energy_completeness" in parts:
                _log["loss_energy_completeness"] = float(parts["energy_completeness"])
            if "recall" in parts:
                _log["loss_recall"] = float(parts["recall"])
            for _k in ("Erecall_highE", "Erecall_lowE", "dice_highE", "n_highE",
                       "evt_E_frac_soft", "evt_E_frac_hard", "evt_E_frac_hard_min"):
                if _k in parts:
                    _log[_k] = float(parts[_k])
            wandb.log(_log)
        self.loss_final += float(loss.item() if hasattr(loss, "item") else loss)
        self.number_b += 1
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        per_layer, targets, queries, _ = self(batch_g, y, 1)
        loss, parts = mask3d_loss(
            per_layer, targets, self.matcher,
            weights=self.loss_weights,
            aux_layer_weight=self.aux_layer_weight,
            mask_loss_type=self.mask_loss_type,
            focal_gamma=self.focal_gamma,
            mask_cost_use_classification=self.mask_cost_use_classification,
            obj_bce_cost_weight=self.obj_bce_cost_weight,
            track_loss_weight=self.track_loss_weight,
            track_hit_type=self.track_hit_type,
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
            if "frame_translation" in parts:
                _log["val_frame_translation"] = float(parts["frame_translation"])
            if "frame_rotation" in parts:
                _log["val_frame_rotation"] = float(parts["frame_rotation"])
            if "energy_completeness" in parts:
                _log["val_energy_completeness"] = float(parts["energy_completeness"])
            if "recall" in parts:
                _log["val_recall"] = float(parts["recall"])
            for _k in ("Erecall_highE", "Erecall_lowE", "dice_highE", "n_highE",
                       "evt_E_frac_soft", "evt_E_frac_hard", "evt_E_frac_hard_min"):
                if _k in parts:
                    _log["val_" + _k] = float(parts[_k])
            wandb.log(_log)
        return loss

    def create_paths(self):
        show_df_eval_path = os.path.join(self.args.model_prefix, "showers_df_evaluation")
        self.show_df_eval_path = show_df_eval_path

    def on_train_epoch_end(self):
        denom = max(self.number_b, 1)
        self.log("train_loss_epoch", self.loss_final / denom)

    def on_train_epoch_start(self):
        self.loss_final = 0.0
        self.number_b = 0

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        opt_name = (getattr(self.args, "optimizer", None) or "adamw").lower()
        lr = self.args.start_lr
        weight_decay = 1e-5
        if opt_name in ("adam", "adamw"):
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"[attn_ipa_model] unsupported --optimizer {opt_name!r}. "
                "Pass --optimizer adamW (the only one wired here)."
            )
        total_steps = max(
            int(self.args.num_epochs) * int(self.args.train_batches), 1,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            total_steps=total_steps,
            pct_start=0.01,
            div_factor=4.0, final_div_factor=10.0,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


ExampleWrapper = AttnIPAModel
