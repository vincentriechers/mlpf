"""Lightning module for the GATr-IPA experiment.

Replaces Mask3D's encoder + MaskFormer decoder with:
  * `GATrIPABackbone` — GATr that returns per-hit (scalar feature, 3-D point).
  * `IPADecoder`      — AlphaFold-IPA-style decoder where queries carry
                        (q_s, T_i, t_i) and the cross-attention bias is the
                        squared distance `‖T_i·t_i − x_j‖²`.

Forward output mirrors `Mask3DModel`'s 4-tuple so the existing training_step /
validation_step / mask3d_loss / matcher all consume the same shapes. The
4th slot (`attn_bias` in Mask3D) is None here — the IPA layer's geometric
term lives inside the cross-attention itself and is not exposed as a
single per-(query, hit) log-density.
"""
import os
import sys
from time import time

import lightning as L
import torch
import torch.nn as nn
import wandb

from src.models.Mask3D.gatr_ipa_backbone import GATrIPABackbone
from src.models.Mask3D.ipa_decoder import IPADecoder
from src.models.Mask3D.loss import mask3d_loss
from src.models.Mask3D.matcher import Matcher
from src.models.Mask3D.targets import build_targets


class GATrIPAModel(L.LightningModule):
    """GATr backbone + IPA decoder; reuses the Mask3D matcher / loss / hooks.

    All hyperparameters except those specific to the new architecture
    (`dim`, `n_heads`, `dec_layers`, `num_queries`, plus the IPA-decoder
    `mask_threshold`) are inherited from `args` (CLI flags), matching
    `mask3d_model.py` so the same wrapper script works with both models.
    """

    def __init__(
        self,
        args,
        dev,
        dim=256,
        n_heads=8,
        dec_layers=8,
        num_queries=320,
        # GATr backbone knobs
        gatr_num_blocks=10,
        gatr_hidden_mv_channels=16,
        gatr_hidden_s_channels=64,
        # Variant B: when False, the backbone returns raw `pos_hits_xyz` (in
        # scaled-metres) as the geometric point passed to the IPA decoder,
        # instead of GATr's `extract_point(out)`. The scalar features still
        # come from the full GATr backbone. Default True keeps backward-
        # compatible behaviour for existing checkpoints.
        gatr_extract_point=True,
        # Mask / loss recipe — same defaults as mask3d_model.py
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
        # When True, the mask logit's geometric term is an MLP on the local-
        # frame relative coordinate `T_iᵀ(x_j − t_i)` (anisotropic / direction-
        # aware). When False (default), the geometric term is the scalar
        # squared distance `−γ·‖T_i·t_i − x_j‖²`.
        ipa_mask_direction_aware=False,
        ipa_mask_threshold=0.5,
        # FAPE-style frame supervision for matched (charged, E≥1 GeV) pairs.
        # Set to 0 (default) to disable. The translation term penalises
        # ‖t_i − centroid_of_GT_hits‖²; the rotation term penalises
        # 1 − ⟨T_i·ẑ, particle_axis⟩, where axis is built from y.angle (θ, φ)
        # by build_targets. Loss is masked to `targets["particle_supervisable"]`.
        frame_translation_weight=0.0,
        frame_rotation_weight=0.0,
        # Initial-query strategy ("static" or "encoder"). Default "encoder"
        # mirrors baseline Mask3D so layer-0 mask logits are event-conditioned.
        dynamic_query_source="encoder",
        track_seed_bonus=6e4,
        # Per-hit energy weighting of the mask losses (none / log / linear).
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
        self.dim = dim

        # Backbone: GATr → per-hit (feats, points).
        self.encoder = GATrIPABackbone(
            dim=dim,
            num_blocks=gatr_num_blocks,
            hidden_mv_channels=gatr_hidden_mv_channels,
            hidden_s_channels=gatr_hidden_s_channels,
            use_extract_point=gatr_extract_point,
        )
        # Decoder: IPA-style with per-query frames.
        self.decoder = IPADecoder(
            dim=dim,
            num_heads=n_heads,
            num_queries=num_queries,
            num_layers=dec_layers,
            mask_threshold=ipa_mask_threshold,
            mask_direction_aware=ipa_mask_direction_aware,
            dynamic_query_source=dynamic_query_source,
            track_seed_bonus=track_seed_bonus,
            track_hit_type=track_hit_type,
        )

        # Hungarian matcher + loss recipe (same as Mask3D).
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @staticmethod
    def _scatter_flat_to_padded(x_flat, batch_ids, local_idx, key_valid):
        B, N_max = key_valid.shape
        D = x_flat.size(-1)
        out = x_flat.new_zeros(B, N_max, D)
        out[batch_ids, local_idx] = x_flat
        return out

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        targets = build_targets(g, y=y, ILD=getattr(self.args, "ILD", False))

        feats_flat, points_flat = self.encoder(g)                                # (N, dim), (N, 3)
        feats_padded = self._scatter_flat_to_padded(
            feats_flat, targets["batch_ids"], targets["local_idx"], targets["key_valid"],
        )                                                                         # (B, N_max, dim)
        points_padded = self._scatter_flat_to_padded(
            points_flat, targets["batch_ids"], targets["local_idx"], targets["key_valid"],
        )                                                                         # (B, N_max, 3)

        per_layer, q_s, T, t = self.decoder(
            feats_padded, points_padded, targets["key_valid"],
            hit_subsystem=targets.get("hit_subsystem"),
        )
        # Stash the final frames + the per-key positions so the training/
        # validation step can pass them to mask3d_loss for FAPE-style frame
        # supervision (the 4-tuple shape stays compatible with Mask3DModel).
        self._last_T = T
        self._last_t = t
        self._last_points_padded = points_padded
        return per_layer, targets, q_s, None

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

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
                f"[gatr_ipa_model] unsupported --optimizer {opt_name!r}. "
                "Pass --optimizer adamW (the only one wired here)."
            )
        # OneCycleLR — same as mask3d_model.py default.
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


# Alias for the network-config wrapper convention.
ExampleWrapper = GATrIPAModel
