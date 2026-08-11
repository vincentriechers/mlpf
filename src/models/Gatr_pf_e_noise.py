"""
This file includes code adapted from:

    Geometric Algebra Transformer (GATr)
    https://github.com/Qualcomm-AI-research/geometric-algebra-transformer

The original implementation is by Qualcomm AI Research. It has been modified
and integrated into this project for particle-flow reconstruction at the
CLD detector (FCC-ee). Please refer to the original repository for
authorship, documentation, and license information.
"""
import torch
import torch.nn as nn
import dgl
from src.layers.object_cond import object_condensation_loss2
from src.models.energy_correction_NN import EnergyCorrection
from src.layers.inference_oc import create_and_store_graph_output
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR
from xformers.ops.fmha import BlockDiagonalMask
import os
import wandb
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar, extract_point, embed_scalar
from src.utils.logger_wandb import log_losses_wandb


class ExampleWrapper(L.LightningModule):
    def __init__(
        self,
        args,
        dev,
        blocks=10,
        hidden_mv_channels=16,
        hidden_s_channels=64,
        config=None
    ):
        super().__init__()
        self.strict_loading = False
        self.input_dim = 3
        self.output_dim = 4
        self.loss_final = 0
        self.number_b = 0
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showers_db = []
        self.args = args
        self.dev = dev
        self.config = config
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

    def forward(self, g, y, step_count, eval="", return_train=False, use_gt_clusters=False):
        if not use_gt_clusters:
            inputs = g.ndata["pos_hits_xyz"].float()
            inputs_scalar = g.ndata["hit_type"].float().view(-1, 1)
            inputs = self.ScaledGooeyBatchNorm2_1(inputs)
            embedded_inputs = embed_point(inputs) + embed_scalar(inputs_scalar)
            embedded_inputs = embedded_inputs.unsqueeze(-2)  # (N, 1, 16)
            mask = self.build_attention_mask(g)
            scalars = torch.cat((g.ndata["e_hits"].float(), g.ndata["p_hits"].float()), dim=1)
            embedded_outputs, scalar_outputs = self.gatr(
                embedded_inputs, scalars=scalars, attention_mask=mask
            )
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
            x = torch.cat((x_cluster_coord, beta.view(-1, 1)), dim=1)
        else:
            x = torch.ones_like(g.ndata["h"][:, 0:4])

        if self.args.correction:
            result = self.energy_correction.forward_correction(g, x, y, return_train)
            return result
        else:
            pred_energy_corr = torch.ones_like(beta.view(-1, 1))
            return x, pred_energy_corr, 0, 0

    def build_attention_mask(self, g):
        batch_numbers = obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )

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
        if self.trainer.is_global_zero:
            result = self(batch_g, y, batch_idx)
        else:
            result = self(batch_g, y, 1)

        model_output = result[0]
        e_cor = result[1]
        (loss, losses) = object_condensation_loss2(
            batch_g,
            model_output,
            e_cor,
            y,
            q_min=self.args.qmin,
            use_average_cc_pos=self.args.use_average_cc_pos,
        )
        if self.args.correction:
            self.energy_correction.global_step = self.global_step
            fixed = self.current_epoch > 0
            loss_EC, loss_pos, loss_neutral_pid, loss_charged_pid = self.energy_correction.get_loss(
                batch_g, y, result, self.stats, fixed
            )
            loss = loss_EC + loss_neutral_pid + loss_charged_pid

        if self.trainer.is_global_zero:
            log_losses_wandb(True, batch_idx, 0, losses, loss)
        self.loss_final = loss.item() + self.loss_final
        self.number_b = self.number_b + 1
        del model_output
        del e_cor
        del losses
        return loss

    def validation_step(self, batch, batch_idx):
        self.create_paths()
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
                ec_x=ec_x,
                total_number_events=self.total_number_events,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                use_gt_clusters=self.args.use_gt_clusters,
                number_of_fakes=num_fakes,
                extra_features=extra_features,
                fakes_labels=fakes_labels,
                pandora_available=self.args.pandora,
            )
            self.df_showers_pandora.append(df_batch_pandora)
            self.df_showers_db.append(df_batch1)
        del model_output

    def create_paths(self):
        show_df_eval_path = os.path.join(self.args.model_prefix, "showers_df_evaluation")
        self.show_df_eval_path = show_df_eval_path

    def on_train_epoch_end(self):
        self.log("train_loss_epoch", self.loss_final / self.number_b)

    def on_train_epoch_start(self):
        self.loss_final = 0
        self.number_b = 0
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
        self.df_showers_db = []
        self.validation_step_outputs = []

    def make_mom_zero(self):
        if self.current_epoch > 1 or self.args.predict:
            print("making momentum 0")
            self.ScaledGooeyBatchNorm2_1.momentum = 0

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            if self.args.predict:
                from src.layers.inference_oc import store_at_batch_end
                import pandas as pd
                
                if self.args.pandora:
                    self.df_showers_pandora = pd.concat(self.df_showers_pandora)
                else:
                    self.df_showers_pandora = []
                self.df_showers_db = pd.concat(self.df_showers_db)
                store_at_batch_end(
                    path_save=os.path.join(
                        self.args.model_prefix, "showers_df_evaluation"
                    ) + "/" + self.args.name_output,
                    df_batch_pandora=self.df_showers_pandora,
                    df_batch1=self.df_showers_db,
                    step=0,
                    predict=True,
                    store=True,
                    pandora_available=self.args.pandora
                )

        self.validation_step_outputs = []
        self.df_showers = []
        self.df_showers_pandora = []
        self.df_showers_db = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.start_lr)
        scheduler = CosineAnnealingThenFixedScheduler(optimizer, T_max=int(36400 * 2), fixed_lr=1e-5)
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


def obtain_batch_numbers(g):
    graphs_eval = dgl.unbatch(g)
    number_graphs = len(graphs_eval)
    batch_numbers = []
    for index in range(number_graphs):
        num_nodes = graphs_eval[index].number_of_nodes()
        batch_numbers.append(index * torch.ones(num_nodes))
    return torch.cat(batch_numbers, dim=0)


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
                param_group["lr"] = self.fixed_lr
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
