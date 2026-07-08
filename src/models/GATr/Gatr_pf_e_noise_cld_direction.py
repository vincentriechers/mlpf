"""GATr variant for CLD-geometry symmetries, with track-direction input.

This is `Gatr_pf_e_noise_cld.py` (the per-event beam-axis ẑ reference NODE that
restricts GATr's E(3)-equivariance to the CLD azimuthal SO(2)_φ symmetry) plus
two extra inputs borrowed from `Gatr_pf_e_noise_direction.py`:

  1. **Track momentum direction at the calorimeter.** Each hit node's input
     multivector gains `embed_translation(pos_pxpypz_at_calo)` (summed into the
     same mv channel as the point + hit-type scalar). `pos_pxpypz_at_calo` is the
     track momentum at the calo entry point — nonzero only for track nodes, zero
     for calo hits — so this hands the network the extrapolated track direction
     as a proper GA translation, which is exactly the geometric cue Pandora uses
     for track–cluster association. On the output side there is NO learned
     clustering head: the cluster coordinate IS the equivariant 3-D output point,
     so the clustering is exactly equivariant under the CLD SO(2)_φ symmetry. Two
     alternatives are deliberately avoided: a learned `Linear(3 -> 3)` does not
     commute with rotations (breaks equivariance), and concatenating the extracted
     translation vector (as `Gatr_pf_e_noise_direction.py` does) mixes two
     different group representations — a translation-covariant point with a
     translation-invariant direction — which cannot be made equivariant at all.

  2. **log E and log p scalars.** The scalar inputs are now
     [e_hits, p_hits, log(e_hits), log(p_hits)] (clamped before the log so the
     zeros — p=0 for calo hits, e=0 for tracks — don't blow up). The log channels
     give the network a roughly-linear handle across the wide dynamic range of
     hit energies / track momenta without changing the raw e, p it already saw.

Everything about the CLD ẑ reference node (splicing, attention mask, dropping the
ref before the heads) is unchanged from `Gatr_pf_e_noise_cld.py`; see that file's
docstring for the symmetry argument and the z-mirror (Z₂) caveat. The rest of the
pipeline (OC loss, clustering, energy-correction head, lightning hooks,
schedulers) is also identical.
"""

from os import path
import sys
import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import (
    embed_point,
    extract_scalar,
    extract_point,
    embed_scalar,
    embed_translation,
    extract_translation,
)
import torch
import torch.nn as nn
import numpy as np
import dgl
from src.layers.object_cond import object_condensation_loss2
from src.layers.utils_training import obtain_batch_numbers
from src.models.energy_correction_NN_v1 import EnergyCorrection
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.nn.tools import log_losses_wandb
import torch.nn.functional as F
from src.layers.CML_loss import supcon_loss_node_equal


# Physical cap on |pos_pxpypz_at_calo| (GeV) applied when --clip-calo-momentum is set.
# A single track in a 91 GeV collision cannot exceed the beam energy (~45.6 GeV = ecm/2);
# detector resolution smears the Z->ll leptons up to ~56 GeV. ILD track reconstruction
# occasionally emits unphysical ~100-240 GeV momenta (seen in Zee/Zuds), which, fed raw
# through embed_translation, dominate the input multivector. Cap at 60 GeV: above the
# physical lepton+resolution edge (~56), below the glitch band (>=100). The clip is
# isotropic (scales the whole vector) so it preserves the track direction, the exact
# zeros for calo hits, and the SO(2)_phi equivariance. Raise proportionally for higher ecm.
_P_CALO_CLIP_GEV = 60.0


class ExampleWrapper(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        config=None,
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3
        # Cluster directly on the first `clust_dim` components of the equivariant output
        # point — x_cluster_coord = points[:, :clust_dim], NO learned head. clust_dim == 3
        # -> the full point (E(3)/SO(2)_φ equivariant). Matches the point-direct
        # checkpoints (14062026 / the 8-GPU resume run): no clustering.weight.
        # Use --clustering_space_dim 3.
        clust_dim = getattr(args, "clustering_space_dim", 3)
        self.output_dim = clust_dim + 1  # clust_dim coords + 1 beta
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.dev = dev
        self.config = config

        # --- CLD-symmetry: per-event beam-axis reference NODE ---------------------------
        # ẑ = (0,0,1), registered as a (non-trainable) buffer. At forward time we splice
        # one ref node per event into the batched sequence (carrying embed_translation(ẑ)
        # in channel 0) and rebuild the attention mask so each ref node only attends
        # within its own event — no cross-event mixing. Discarded before clustering/β.
        # In_mv channel count is therefore the SAME as the base model — ref is a node,
        # not a channel.
        self.register_buffer(
            "cld_z_ref", torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        )  # (1, 3)
        # ---------------------------------------------------------------------------------

        _in_mv_uot = 1 if getattr(args, "uot_labels", False) else 0
        _in_mv = 1 + _in_mv_uot  # hit + (optional) uot ref channel — CLD ref is a NODE
        self.gatr = GATr(
            in_mv_channels=_in_mv,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=4,  # e, p, log(e), log(p)
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        if getattr(args, "uot_labels", False):
            self.ScaledGooeyBatchNorm2_ref = nn.BatchNorm1d(3, momentum=0.1)
        # No learned clustering head: the cluster coordinate is the first clust_dim
        # components of the equivariant output point (point-direct).
        self.beta = nn.Linear(2, 1)
        # Initialize the energy correction module
        if self.args.correction:
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

    def _splice_cld_ref_nodes(self, embedded_inputs, scalars, hit_counts):
        """Splice one ẑ-reference node per event into the batched sequence.

        For each event i with `hit_counts[i]` hits, we insert *one* extra node carrying
        `embed_translation(ẑ)` in mv channel 0 (other channels and scalars are 0). The
        ref node is placed at the END of the event's block, so the per-event block
        layout is  [hit_0, hit_1, ..., hit_{n_i-1}, ref_i].

        Parameters
        ----------
        embedded_inputs : (N_hits, n_in_mv, 16)
        scalars         : (N_hits, n_in_s)
        hit_counts      : (n_graphs,) int — number of hits per event in the batch

        Returns
        -------
        new_embedded_inputs : (N_hits + n_graphs, n_in_mv, 16)
        new_scalars         : (N_hits + n_graphs, n_in_s)
        new_counts          : (n_graphs,) — hit_counts + 1, used to rebuild the mask
        hit_pos             : (N_hits,) long — index of each original hit in the new
                              sequence, for slicing the GATr output back to hits-only
        """
        device = embedded_inputs.device
        hit_counts = hit_counts.to(device=device, dtype=torch.long)
        n_graphs = hit_counts.shape[0]
        N_hits = embedded_inputs.shape[0]
        new_counts = hit_counts + 1
        N_new = int(new_counts.sum().item())

        # For each hit k in event i, its new position is k + i (events before it each
        # contribute one extra ref slot ahead). Ref position for event i is the last
        # slot of its block.
        event_of_hit = torch.repeat_interleave(
            torch.arange(n_graphs, device=device), hit_counts
        )
        hit_pos = torch.arange(N_hits, device=device) + event_of_hit  # (N_hits,)
        new_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), new_counts.cumsum(0)]
        )
        ref_pos = new_offsets[1:] - 1  # (n_graphs,)

        # Allocate and scatter
        new_embedded_inputs = torch.zeros(
            N_new, embedded_inputs.shape[1], embedded_inputs.shape[2],
            dtype=embedded_inputs.dtype, device=device,
        )
        new_embedded_inputs[hit_pos] = embedded_inputs

        # Ref nodes: channel 0 carries embed_translation(ẑ); other mv channels / scalars stay 0.
        ref_z = self.cld_z_ref.expand(n_graphs, -1)        # (n_graphs, 3)
        embedded_z = embed_translation(ref_z)              # (n_graphs, 16)
        new_embedded_inputs[ref_pos, 0, :] = embedded_z

        new_scalars = torch.zeros(
            N_new, scalars.shape[1], dtype=scalars.dtype, device=device
        )
        new_scalars[hit_pos] = scalars

        return new_embedded_inputs, new_scalars, new_counts, hit_pos

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        tic = time.time()
        if not use_gt_clusters:
            inputs = g.ndata["pos_hits_xyz"].float()
            inputs_scalar = g.ndata["hit_type"].float().view(-1, 1)
            inputs = self.ScaledGooeyBatchNorm2_1(inputs)
            # Hit node input mv: point (position) + hit-type scalar + track momentum
            # direction at the calo entry point (zero for calo hits, nonzero for tracks).
            p_calo = g.ndata["pos_pxpypz_at_calo"].float()
            if getattr(self.args, "unit_calo_direction", False):
                # Unit-normalize the track calo-momentum to a PURE DIRECTION; the magnitude
                # is already fed via the [e, p, log e, log p] scalars, so this decouples the
                # geometric track-cluster-association cue from |p|. Guard the zeros: the vast
                # majority of nodes are calo hits (and any track with a failed fit) that carry
                # p_calo=0 -> keep them exactly 0 (no direction); only charged-track nodes with
                # |p|>0 get a unit vector. Isotropic (p/|p| commutes with rotation) so SO(2)_phi
                # equivariance is preserved exactly, and it removes all scale/outlier issues.
                _pn = p_calo.norm(dim=1, keepdim=True)
                p_calo = torch.where(
                    _pn > 1e-6, p_calo / _pn.clamp_min(1e-6), torch.zeros_like(p_calo)
                )
            elif getattr(self.args, "clip_calo_momentum", False):
                # Isotropic norm clip: scale down ONLY tracks whose |p| exceeds the physical
                # cap (removes ~100-240 GeV ILD reco glitches). Preserves direction, the
                # calo-hit zeros (|p|=0 -> factor 1 -> stays 0) and SO(2)_phi equivariance.
                _pn = p_calo.norm(dim=1, keepdim=True)
                p_calo = p_calo * torch.clamp(_P_CALO_CLIP_GEV / _pn.clamp_min(1e-9), max=1.0)
            embedded_hits = (
                embed_point(inputs)
                + embed_scalar(inputs_scalar)
                + embed_translation(p_calo)
            )
            embedded_inputs = embedded_hits.unsqueeze(-2)  # (N_hits, 1, 16)

            if getattr(self.args, "fuot_labels", False) and "uot_ref_xyz" in g.ndata:
                uot_ref = self.ScaledGooeyBatchNorm2_ref(g.ndata["uot_ref_xyz"].float())
                embedded_ref = embed_point(uot_ref).unsqueeze(-2)  # (N_hits, 1, 16)
                embedded_inputs = torch.cat([embedded_inputs, embedded_ref], dim=-2)

            # Scalars: raw e, p plus their logs (clamped — p=0 for calo hits, e=0 for tracks).
            e_hits = g.ndata["e_hits"].float()
            p_hits = g.ndata["p_hits"].float()
            scalars = torch.cat(
                (
                    e_hits,
                    p_hits,
                    torch.log(e_hits.clamp_min(1e-6)),
                    torch.log(p_hits.clamp_min(1e-6)),
                ),
                dim=1,
            )

            # --- CLD: splice one ẑ-reference node per event into the batched sequence ---
            # `new_counts = batch_num_nodes + 1` drives the BlockDiagonal attention mask
            # so each ref node attends only within its own event. Ref nodes' GATr outputs
            # are dropped before the clustering / β heads run; `hit_pos` restores the
            # original per-hit ordering and length expected downstream.
            hit_counts = g.batch_num_nodes()
            embedded_inputs, scalars, new_counts, hit_pos = self._splice_cld_ref_nodes(
                embedded_inputs, scalars, hit_counts
            )
            mask = BlockDiagonalMask.from_seqlens(new_counts.tolist())
            # ----------------------------------------------------------------------------

            # Pass augmented sequence through GATr
            embedded_outputs, scalar_outputs = self.gatr(
                embedded_inputs, scalars=scalars, attention_mask=mask
            )  # (N_hits + n_graphs, 1, 16) and (N_hits + n_graphs, 1)

            # Drop the per-event ref nodes — heads see only hit-position outputs.
            embedded_outputs = embedded_outputs[hit_pos]   # (N_hits, 1, 16)
            scalar_outputs = scalar_outputs[hit_pos]        # (N_hits, 1)

            points = extract_point(embedded_outputs[:, 0, :])
            nodewise_outputs = extract_scalar(embedded_outputs)  # (N_hits, 1, 1)
            x_scalar = torch.cat(
                (nodewise_outputs.view(-1, 1), scalar_outputs.view(-1, 1)), dim=1
            )
            # Cluster directly on the first output_dim-1 components of the equivariant
            # output point (no learned head); clust_dim=3 -> full point.
            x_cluster_coord = points[:, : self.output_dim - 1]
            beta = self.beta(x_scalar)
            g.ndata["final_cluster"] = x_cluster_coord
            g.ndata["beta"] = beta.view(-1)
            x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)

            pred_energy_corr = torch.ones_like(beta.view(-1, 1)).flatten()
            toc = time.time()
        else:
            x = torch.ones_like(g.ndata["h"][:, 0:4])
        if self.args.correction:
            with torch.autocast(device_type="cuda", enabled=False):
                result = self.energy_correction.forward_correction(g, x.float(), y, return_train)
            return result
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0, 0

    def build_attention_mask(self, g):
        return BlockDiagonalMask.from_seqlens(g.batch_num_nodes().tolist())

    def unfreeze_all(self):
        for p in self.energy_correction.model_charged.parameters():
            p.requires_grad = True
        for p in self.energy_correction.model_neutral.gatr_pid.parameters():
            p.requires_grad = True
        for p in self.energy_correction.model_neutral.PID_head.parameters():
            p.requires_grad = True

    def training_step(self, batch, batch_idx):
        y = batch[1]
        batch_g = batch[0]
        use_gt = self.args.use_gt_clusters if self.args.correction else False
        if self.trainer.is_global_zero:
            result = self(batch_g, y, batch_idx, use_gt_clusters=use_gt)
        else:
            result = self(batch_g, y, 1, use_gt_clusters=use_gt)

        model_output = result[0].float()
        e_cor = result[1].float()
        if not self.args.correction:
            CML_loss = False
            if CML_loss:
                node_counts = batch_g.batch_num_nodes().tolist()
                embeddings_split = torch.split(model_output[:, 0:self.output_dim - 1], node_counts)
                group_ids_split = torch.split(batch_g.ndata["particle_number"], node_counts)
                per_event_losses = [
                    supcon_loss_node_equal(emb, gids)
                    for emb, gids in zip(embeddings_split, group_ids_split)
                ]
                loss = torch.stack(per_event_losses).mean()
                losses = {}
            else:
                with torch.autocast(device_type="cuda", enabled=False):
                    (loss, losses,) = object_condensation_loss2(
                        batch_g,
                        model_output,
                        e_cor,
                        y,
                        clust_loss_only=True,
                        add_energy_loss=False,
                        calc_e_frac_loss=False,
                        q_min=self.args.qmin,
                        frac_clustering_loss=self.args.frac_cluster_loss,
                        attr_weight=self.args.L_attractive_weight,
                        repul_weight=self.args.L_repulsive_weight,
                        fill_loss_weight=self.args.fill_loss_weight,
                        use_average_cc_pos=self.args.use_average_cc_pos,
                        loss_type=self.args.losstype,
                        output_dim=self.output_dim,
                    )
        else:
            losses = {}
        if self.args.correction:
            self.energy_correction.global_step = self.global_step
            if self.current_epoch == 0:
                fixed = False
            else:
                fixed = True
            loss_EC, loss_pos, loss_neutral_pid, loss_charged_pid, loss_score, self.stats = self.energy_correction.get_loss(
                batch_g, y, result, self.stats, fixed
            )
            loss = loss_EC + loss_neutral_pid + loss_charged_pid
        else:
            loss_score = 0
        if self.trainer.is_global_zero and not self.args.correction:
            log_losses_wandb(True, batch_idx, 0, losses, loss, loss_score)
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output
        del e_cor
        del losses
        return loss

    def validation_step(self, batch, batch_idx):
        self.create_paths()
        self.validation_step_outputs = []
        y = batch[1]
        batch_g = batch[0]
        shap_vals, ec_x = None, None
        if self.args.correction:
            result = self(batch_g, y, 1, use_gt_clusters=self.args.use_gt_clusters)
            model_output = result[0]
            outputs = self.energy_correction.get_validation_step_outputs(batch_g, y, result)
            e_cor1, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels = outputs
            e_cor = e_cor1
        else:
            model_output, e_cor1, loss_ll, _ = self(batch_g, y, 1)
            e_cor1 = torch.ones_like(model_output[:, 0].view(-1, 1))
            e_cor = e_cor1
            pred_pos = None
            pred_pid = None
            pred_ref_pt = None
            num_fakes = None
            extra_features = None
            fakes_labels = None
        if self.args.explain_ec:
            self.validation_step_outputs.append(
                [model_output, e_cor, batch_g, y, shap_vals, ec_x, num_fakes]
            )
        if self.args.predict:
            if self.args.correction:
                model_output1 = model_output
                e_corr = e_cor
            else:
                model_output1 = torch.cat((model_output, e_cor.view(-1, 1)), dim=1)
                e_corr = None

            (
                df_batch_pandora,
                df_batch1,
                self.total_number_events,
            ) = create_and_store_graph_output(
                batch_g,
                model_output1,
                y,
                0,
                batch_idx,
                0,
                path_save=self.show_df_eval_path,
                store=True,
                predict=True,
                e_corr=e_corr,
                tracks=self.args.tracks,
                shap_vals=shap_vals,
                ec_x=ec_x,
                clust_dim=self.output_dim - 1,
                total_number_events=self.total_number_events,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                use_gt_clusters=self.args.use_gt_clusters,
                fix_clusters_class=self._fix_clusters_class,
                pids_neutral=self.pids_neutral,
                pids_charged=self.pids_charged,
                number_of_fakes=num_fakes,
                extra_features=extra_features,
                fakes_labels=fakes_labels,
                pandora_available=self.args.pandora,
                truth_tracks=self.args.truth_tracking,
            )
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showes_db.append(df_batch1)
        del model_output

    def create_paths(self):
        show_df_eval_path = os.path.join(
            self.args.model_prefix, "showers_df_evaluation"
        )
        self.show_df_eval_path = show_df_eval_path

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        self.make_mom_zero()
        if self.current_epoch == 0:
            self.stats = {}
            self.stats["counts"] = {}
            self.stats["counts_pid_neutral"] = {}
            self.stats["counts_pid_charged"] = {}

    def on_validation_epoch_start(self):
        self.total_number_events = 0
        self.make_mom_zero()
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd
                if self.args.explain_ec:
                    shap_vals = self.validation_step_outputs[0][4]
                    path_shap_vals = os.path.join(
                        self.args.model_prefix, "shap_vals.pkl"
                    )
                    torch.save(shap_vals, path_shap_vals)
                    print("SHAP values saved!")
                if self.args.pandora:
                    self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                else:
                    self.df_showers_pandora = []
                self.df_showes_db = pd.concat(self.df_showes_db)
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ) + "/" + self.args.name_output,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showes_db,
                    step=0,
                    predict=True,
                    store=True,
                    pandora_available=self.args.pandora,
                )
        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)
        scheduler = CosineAnnealingThenFixedScheduler(optimizer, T_max=int(36400 * 2), fixed_lr=1e-5)  # 10000
        self.scheduler = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "monitor": "train_loss_epoch",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        scheduler.step()

    def correction_training_step(self, e_cor, e_true, neutral_idx):
        if self.args.correction:
            loss_EC_neutrals = torch.nn.L1Loss()(
                e_cor[neutral_idx], e_true[neutral_idx]
            )
            wandb.log({"loss_EC_neutrals": loss_EC_neutrals})
            loss = loss + loss_EC_neutrals


def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(0, number_graphs):
        gj = graphs_eval[index]
        num_nodes = gj.number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
    return torch.cat(batch_numbers, dim=0)


class CosineAnnealingThenFixedScheduler:
    def __init__(self, optimizer, T_max, fixed_lr):
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=fixed_lr)
        self.fixed_lr = fixed_lr
        self.T_max = T_max
        self.step_count = 0
        self.optimizer = optimizer

    def step(self):
        if self.step_count < self.T_max:
            self.cosine_scheduler.step()
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.fixed_lr
        self.step_count += 1

    def get_last_lr(self):
        if self.step_count < self.T_max:
            return self.cosine_scheduler.get_last_lr()
        return [self.fixed_lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "cosine_scheduler_state": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler_state"])
