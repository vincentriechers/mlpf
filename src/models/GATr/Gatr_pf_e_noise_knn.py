from os import path
import sys
import time
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
import torch
import torch.nn as nn
from src.logger.plotting_tools import PlotCoordinates
import numpy as np
import dgl
from src.layers.object_cond import object_condensation_loss2
from src.layers.utils_training import obtain_batch_numbers
from src.models.energy_correction_NN_v1 import EnergyCorrection
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.utils.nn.tools import log_losses_wandb
import torch.nn.functional as F
from src.layers.CML_loss import supcon_loss_node_equal
from xformers.ops.fmha import BlockDiagonalMask


class ExampleWrapper(L.LightningModule):
    """GATr model with memory-efficient local attention.

    Hits within each event are sorted by a 3D Morton (Z-order) curve so that
    spatially nearby hits are adjacent in the sequence.  A causal sliding window
    of width k (xformers BlockDiagonalCausalLocalAttentionMask) is then applied,
    giving O(n·k) attention cost instead of O(n²).  Across GATr's multiple
    layers information propagates in both directions along the sorted sequence,
    so the effective receptive field is bidirectional despite the causal window.

    Parameters
    ----------
    k : int
        Local attention window width (number of preceding tokens each token attends to).
        Should be ≥ the typical number of physical neighbours you care about.
    """

    def __init__(
        self,
        args,
        dev,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        k=64,
        config=None,
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3
        self.output_dim = 4
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showes_db = []
        self.args = args
        self.dev = dev
        self.config = config
        self.k = k
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=2,
            out_s_channels=1,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.ScaledGooeyBatchNorm2_1 = nn.BatchNorm1d(self.input_dim, momentum=0.1)
        self.clustering = nn.Linear(3, self.output_dim - 1, bias=False)
        self.beta = nn.Linear(2, 1)
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
        if getattr(self.args, 'fix_ch', False):
            self._fix_clusters_class.append(1)
        if getattr(self.args, 'fix_neutrals', False):
            self._fix_clusters_class.append(2)
        if getattr(self.args, 'fix_photons', False):
            self._fix_clusters_class.append(3)

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        tic = time.time()
        if not use_gt_clusters:
            inputs = g.ndata["pos_hits_xyz"].float()
            if self.trainer.is_global_zero and step_count % 500 == 0:
                g.ndata["original_coords"] = g.ndata["pos_hits_xyz"]
                PlotCoordinates(
                    g,
                    path="input_coords",
                    outdir=self.args.model_prefix,
                    predict=self.args.predict,
                    epoch=str(self.current_epoch) + eval,
                    step_count=step_count,
                )
            inputs_scalar = g.ndata["hit_type"].float().view(-1, 1)
            inputs = self.ScaledGooeyBatchNorm2_1(inputs)
            embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
            embedded_inputs = embedded_inputs.unsqueeze(-2)  # (N, 1, 16)
            scalars = torch.cat(
                (g.ndata["e_hits"].float(), g.ndata["p_hits"].float()), dim=1
            )

            # Memory-efficient local attention via xformers:
            #   - Sort hits within each event by 3D Morton code so that
            #     spatially nearby hits are adjacent in the sequence.
            #   - BlockDiagonalMask isolates events (no cross-event attention).
            #   - make_local_attention(k) creates a causal sliding window of
            #     width k; over multiple GATr layers this gives effective
            #     bidirectional receptive fields without an O(n²) dense matrix.
            batch_numbers = obtain_batch_numbers(g)
            batch_sizes = torch.bincount(batch_numbers.long()).tolist()

            sort_indices = self._morton_sort_indices(inputs, batch_sizes)
            inv_sort = torch.argsort(sort_indices)

            embedded_inputs_sorted = embedded_inputs[sort_indices]
            scalars_sorted = scalars[sort_indices]

            valid_sizes = [int(s) for s in batch_sizes if int(s) > 0]
            mask = BlockDiagonalMask.from_seqlens(valid_sizes).make_local_attention(self.k)

            embedded_outputs_sorted, scalar_outputs_sorted = self.gatr(
                embedded_inputs_sorted, scalars=scalars_sorted, attention_mask=mask
            )

            # Restore original node ordering
            embedded_outputs = embedded_outputs_sorted[inv_sort]
            scalar_outputs = scalar_outputs_sorted[inv_sort]

            points = extract_point(embedded_outputs[:, 0, :])
            nodewise_outputs = extract_scalar(embedded_outputs)  # (N, 1, 1)
            x_point = points
            x_scalar = torch.cat(
                (nodewise_outputs.view(-1, 1), scalar_outputs.view(-1, 1)), dim=1
            )
            x_cluster_coord = self.clustering(x_point)
            beta = self.beta(x_scalar)
            g.ndata["final_cluster"] = x_cluster_coord
            g.ndata["beta"] = beta.view(-1)
            if self.trainer.is_global_zero and step_count % 500 == 0:
                PlotCoordinates(
                    g,
                    path="final_clustering",
                    outdir=self.args.model_prefix,
                    predict=self.args.predict,
                    epoch=str(self.current_epoch) + eval,
                    step_count=step_count,
                )
            x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
            pred_energy_corr = torch.ones_like(beta.view(-1, 1)).flatten()
            toc = time.time()
        else:
            x = torch.ones_like(g.ndata["h"][:, 0:4])
        if self.args.correction:
            result = self.energy_correction.forward_correction(g, x, y, return_train)
            return result
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0, 0

    def _morton_sort_indices(self, positions, batch_sizes):
        """Return indices that sort nodes within each event by their 3D Morton code.

        A Morton (Z-order) curve interleaves the bits of quantised x, y, z
        coordinates, so nearby hits in 3D space map to nearby positions in the
        1D sorted sequence.  This makes a causal sliding-window attention mask
        a good approximation of true 3D-KNN attention without materialising an
        O(n²) dense matrix.

        Parameters
        ----------
        positions : torch.Tensor, shape (N, 3)
            Normalised 3D positions for all nodes (post BatchNorm).
        batch_sizes : list[int]

        Returns
        -------
        sort_indices : torch.LongTensor, shape (N,)
            Permutation that reorders nodes globally so that within each event
            nodes are sorted by their Morton code.
        """
        device = positions.device
        # Quantise to 21-bit integers so that 3 coordinates fit in one int64.
        # Shift from [-~4, ~4] (post-BatchNorm) to [0, 2^21).
        pos_cpu = positions.detach().cpu()
        scale = (2 ** 21 - 1) / (pos_cpu.max(0).values - pos_cpu.min(0).values + 1e-6)
        qpos = ((pos_cpu - pos_cpu.min(0).values) * scale).long().clamp(0, 2 ** 21 - 1)

        def _spread(v):
            # Spread the bits of a 21-bit integer into every 3rd bit of a 63-bit int.
            v = v & 0x1FFFFF
            v = (v | (v << 32)) & 0x1F00000000FFFF
            v = (v | (v << 16)) & 0x1F0000FF0000FF
            v = (v | (v << 8))  & 0x100F00F00F00F00F
            v = (v | (v << 4))  & 0x10C30C30C30C30C3
            v = (v | (v << 2))  & 0x1249249249249249
            return v

        morton = _spread(qpos[:, 0]) | (_spread(qpos[:, 1]) << 1) | (_spread(qpos[:, 2]) << 2)

        sort_indices = []
        offset = 0
        for size in batch_sizes:
            size = int(size)
            if size == 0:
                continue
            local_order = torch.argsort(morton[offset: offset + size])
            sort_indices.append(local_order + offset)
            offset += size
        return torch.cat(sort_indices).to(device)

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

        model_output = result[0]
        e_cor = result[1]
        if not self.args.correction:
            CML_loss = False
            if CML_loss:
                node_counts = batch_g.batch_num_nodes().tolist()
                embeddings_split = torch.split(model_output[:, 0:3], node_counts)
                group_ids_split = torch.split(batch_g.ndata["particle_number"], node_counts)
                per_event_losses = [
                    supcon_loss_node_equal(emb, gids)
                    for emb, gids in zip(embeddings_split, group_ids_split)
                ]
                loss = torch.stack(per_event_losses).mean()
                losses = {}
            else:
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
        cluster_features_path = os.path.join(self.args.model_prefix, "cluster_features")
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
        scheduler = CosineAnnealingThenFixedScheduler(
            optimizer, T_max=int(36400 * 2), fixed_lr=1e-5
        )
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
    batch = torch.cat(batch_numbers, dim=0)
    return batch


class CosineAnnealingThenFixedScheduler:
    def __init__(self, optimizer, T_max, fixed_lr):
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=fixed_lr)
        self.fixed_lr = 1e-6
        self.T_max = T_max
        self.step_count = 0
        self.optimizer = optimizer

    def step(self):
        if self.step_count < self.T_max:
            self.cosine_scheduler.step()
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.fixed_lr
        self.step_count += 1

    def get_last_lr(self):
        if self.step_count < self.T_max:
            return self.cosine_scheduler.get_last_lr()
        else:
            return [self.fixed_lr for _ in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "cosine_scheduler_state": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.cosine_scheduler.load_state_dict(state_dict["cosine_scheduler_state"])


def criterion(ypred, ytrue, step):
    if True or step < 5000:
        return F.l1_loss(ypred, ytrue)
    else:
        losses = F.l1_loss(ypred, ytrue, reduction="none") / ytrue.abs()
        if len(losses.shape) > 0:
            if int(losses.size(0) * 0.05) > 1:
                top_percentile = torch.kthvalue(
                    losses, int(losses.size(0) * 0.95)
                ).values
                mask = losses > top_percentile
                losses[mask] = 0.0
        return losses.mean()
