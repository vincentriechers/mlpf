"""
PID + energy correction module.
The model is called after object condensation clustering to correct
reconstructed energies and predict particle IDs.
"""
import numpy as np
import wandb
import torch
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_add, scatter_mean
from typing import NamedTuple, Any

from src.layers.utils_training import obtain_clustering_for_matched_showers
from src.utils.post_clustering_features import (
    get_post_clustering_features, get_extra_features, calculate_eta, calculate_phi,
)
from src.utils.pid_conversion import pid_conversion_dict
from src.layers.regression.loss_regression import obtain_PID_charged, obtain_PID_neutral
from src.models.energy_correction_charged import ChargedEnergyCorrection
from src.models.energy_correction_neutral import (
    NeutralEnergyCorrection, criterion_E_cor, correct_mask_neutral,
)


class _ClusteringOutput(NamedTuple):
    """Structured return type for clustering_and_global_features."""
    graphs:           Any            # batched DGL graph (feature-augmented)
    batch_idx:        torch.Tensor
    high_level_feats: torch.Tensor   # per-shower aggregate features
    charged_idx:      torch.Tensor
    neutral_idx:      torch.Tensor
    feats_charged:    torch.Tensor   # NaN-zeroed high_level_feats[charged_idx]
    feats_neutral:    torch.Tensor   # NaN-zeroed high_level_feats[neutral_idx]
    pred_energy:      torch.Tensor   # ones placeholder, filled by forward_correction
    pred_pos:         torch.Tensor
    pred_pid:         torch.Tensor
    true:             Any
    true_pid:         torch.Tensor
    true_coords:      torch.Tensor
    sum_e:            torch.Tensor
    e_true_daughters: torch.Tensor
    n_fakes:          int
    extra_features:   torch.Tensor
    fakes_idx:        torch.Tensor


def _zero_nans(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    out[out != out] = 0
    return out


def _decode_pid(pred_pid: torch.Tensor, pids: list, logits: torch.Tensor, idx: torch.Tensor) -> None:
    if pids and len(idx):
        labels = np.array(pids)[np.argmax(logits.cpu().detach(), axis=1)]
        pred_pid[idx.flatten()] = torch.tensor(labels).long().to(idx.device)


class EnergyCorrection:
    def __init__(self, main_model):
        self.args = main_model.args
        self.get_PID_categories()
        self.get_energy_correction()
        self.pid_conversion_dict = pid_conversion_dict
        self.main_model = main_model
        self.global_step = 0

    def get_PID_categories(self):
        self.pids_neutral = [2, 3]
        self.pids_charged = [0, 1, 4]

    def get_energy_correction(self):
        self.model_charged = ChargedEnergyCorrection(args=self.args)
        self.model_neutral = NeutralEnergyCorrection(args=self.args)

    def clustering_and_global_features(self, g, x, y, add_fakes=True) -> _ClusteringOutput:
        (
            graphs_new, true_new, sum_e, true_pid,
            e_true_corr_daughters, true_coords, number_of_fakes, fakes_idx,
        ) = obtain_clustering_for_matched_showers(
            g, x, y, self.main_model.trainer.global_rank,
            use_gt_clusters=self.args.use_gt_clusters,
            add_fakes=add_fakes,
        )

        batch_num_nodes = graphs_new.batch_num_nodes()
        batch_idx = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
        batch_idx = torch.tensor(batch_idx).to(self.main_model.device)

        graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
        graphs_sum_features = scatter_add(graphs_new.ndata["h"], batch_idx, dim=0)
        graphs_sum_features = graphs_sum_features[batch_idx]
        betas = torch.sigmoid(graphs_new.ndata["h"][:, -1])
        graphs_new.ndata["h"] = torch.cat(
            (graphs_new.ndata["h"], graphs_sum_features), dim=1
        )

        high_level = get_post_clustering_features(graphs_new, sum_e)
        extra_features = get_extra_features(graphs_new, betas)

        dev = graphs_new.ndata["h"].device
        n = high_level.shape[0]
        pred_energy = torch.ones(n, device=dev)
        pred_pos    = torch.ones(n, 3, device=dev)
        pred_pid    = torch.ones(n, device=dev).long()

        node_features_avg = scatter_mean(graphs_new.ndata["h"], batch_idx, dim=0)[:, 0:3]
        eta = calculate_eta(node_features_avg[:, 0], node_features_avg[:, 1], node_features_avg[:, 2])
        phi = calculate_phi(node_features_avg[:, 0], node_features_avg[:, 1])
        high_level = torch.cat(
            (high_level, node_features_avg, eta.view(-1, 1), phi.view(-1, 1)), dim=1
        )

        num_tracks  = high_level[:, 7]
        charged_idx = torch.where(num_tracks >= 1)[0]
        neutral_idx = torch.where(num_tracks < 1)[0]
        assert len(charged_idx) + len(neutral_idx) == len(num_tracks)
        assert high_level.shape[0] == graphs_new.batch_num_nodes().shape[0]

        return _ClusteringOutput(
            graphs=graphs_new,
            batch_idx=batch_idx,
            high_level_feats=high_level,
            charged_idx=charged_idx,
            neutral_idx=neutral_idx,
            feats_charged=_zero_nans(high_level[charged_idx]),
            feats_neutral=_zero_nans(high_level[neutral_idx]),
            pred_energy=pred_energy,
            pred_pos=pred_pos,
            pred_pid=pred_pid,
            true=true_new,
            true_pid=true_pid,
            true_coords=true_coords,
            sum_e=sum_e,
            e_true_daughters=e_true_corr_daughters,
            n_fakes=number_of_fakes,
            extra_features=extra_features,
            fakes_idx=fakes_idx,
        )

    def forward_correction(self, g, x, y, return_train):
        cf = self.clustering_and_global_features(g, x, y, add_fakes=self.args.predict)

        charged_energies = self.model_charged.charged_prediction(
            cf.graphs, cf.charged_idx, cf.feats_charged
        )
        neutral_energies, neutral_pxyz_avg = self.model_neutral.neutral_prediction(
            cf.graphs, cf.neutral_idx, cf.feats_neutral
        )

        if len(self.pids_charged):
            charged_energies, charged_positions, charged_PID_pred, charged_ref_pt_pred = charged_energies
        else:
            charged_energies, charged_positions, _ = charged_energies
        if len(self.pids_neutral):
            neutral_energies, neutral_positions, neutral_PID_pred, neutral_ref_pt_pred = neutral_energies
        else:
            neutral_energies, neutral_positions, _ = neutral_energies

        cf.pred_energy[cf.charged_idx.flatten()] = charged_energies
        cf.pred_energy[cf.neutral_idx.flatten()] = neutral_energies

        _decode_pid(cf.pred_pid, self.pids_charged, charged_PID_pred, cf.charged_idx)
        _decode_pid(cf.pred_pid, self.pids_neutral, neutral_PID_pred, cf.neutral_idx)

        cf.pred_energy[cf.pred_energy < 0] = 0.0

        pred_ref_pt = torch.ones_like(cf.pred_pos)
        if len(cf.charged_idx):
            pred_ref_pt[cf.charged_idx.flatten()] = charged_ref_pt_pred.to(pred_ref_pt.device)
            cf.pred_pos[cf.charged_idx.flatten()] = charged_positions.float().to(cf.pred_pos.device)
        if len(cf.neutral_idx):
            pred_ref_pt[cf.neutral_idx.flatten()] = neutral_ref_pt_pred.to(cf.neutral_idx.device)
            cf.pred_pos[cf.neutral_idx.flatten()] = neutral_positions.to(cf.neutral_idx.device).float()

        predictions = {
            "pred_energy_corr": cf.pred_energy,
            "pred_pos":         cf.pred_pos,
            "neutrals_idx":     cf.neutral_idx.flatten(),
            "charged_idx":      cf.charged_idx.flatten(),
            "pred_ref_pt":      pred_ref_pt,
            "extra_features":   cf.extra_features,
            "fakes_labels":     cf.fakes_idx,
        }
        if len(self.pids_charged) or len(self.pids_neutral):
            predictions["pred_PID"]          = cf.pred_pid
            predictions["charged_PID_pred"]  = charged_PID_pred
            predictions["neutral_PID_pred"]  = neutral_PID_pred

        if return_train:
            return x, predictions, cf.true, cf.sum_e, cf.true_pid, cf.true, cf.true_coords, cf.n_fakes
        else:
            return (
                x, predictions, cf.true, cf.sum_e, cf.graphs, cf.batch_idx,
                cf.high_level_feats, cf.true_pid, cf.e_true_daughters,
                cf.true_coords, cf.n_fakes,
            )

    def get_loss(self, batch_g, y, result, stats, fixed):
        (
            model_output, dic_e_cor, e_true, e_sum_hits, new_graphs, batch_id,
            graph_level_features, pid_true_matched, e_true_corr_daughters,
            part_coords_matched, num_fakes,
        ) = result

        e_cor = dic_e_cor["pred_energy_corr"]
        mask_neutral_for_loss = correct_mask_neutral(
            torch.tensor(pid_true_matched), dic_e_cor["neutrals_idx"]
        )

        e_true_neutrals = e_true[mask_neutral_for_loss]
        e_pred_neutrals = e_cor[mask_neutral_for_loss]
        e_reco_neutrals = e_sum_hits[mask_neutral_for_loss]
        in_distribution = (torch.abs(e_true_neutrals - e_reco_neutrals) / e_true_neutrals) < 0.6
        ypred  = e_pred_neutrals[in_distribution]
        ybatch = e_true_neutrals[in_distribution]

        loss_EC_neutrals = criterion_E_cor(ypred.flatten(), ybatch.flatten()) if len(ypred) > 0 else 0
        wandb.log({"loss_EC_neutrals": loss_EC_neutrals})

        loss_neutral_pid = 0
        loss_charged_pid = 0

        if len(self.pids_charged):
            charged_PID_pred, charged_PID_true_onehot, mask_charged = obtain_PID_charged(
                dic_e_cor, pid_true_matched, self.pids_charged, self.args, self.pid_conversion_dict
            )
            loss_charged_pid, acc_charged = pid_loss(
                charged_PID_pred, charged_PID_true_onehot,
                e_true[dic_e_cor["charged_idx"]], mask_charged, fixed, "charged",
            )
            wandb.log({"loss_charged_pid": loss_charged_pid})

        if len(self.pids_neutral):
            neutral_PID_pred, neutral_PID_true_onehot, mask_neutral = obtain_PID_neutral(
                dic_e_cor, pid_true_matched, self.pids_neutral, self.args, self.pid_conversion_dict
            )
            loss_neutral_pid, acc_neutral = pid_loss(
                neutral_PID_pred, neutral_PID_true_onehot,
                e_true, mask_neutral, fixed, "neutral",
            )
            wandb.log({"loss_neutral_pid": loss_neutral_pid})

        return loss_EC_neutrals, 0, loss_neutral_pid, loss_charged_pid

    def get_validation_step_outputs(self, batch_g, y, result):
        (
            model_output, e_cor, e_true, e_sum_hits,
            new_graphs, batch_id, graph_level_features,
            pid_true_matched, e_true_corr_daughters,
            coords_true, num_fakes,
        ) = result

        if len(self.pids_charged):
            charged_idx = e_cor["charged_idx"]
        if len(self.pids_neutral):
            neutral_idx = e_cor["neutrals_idx"]
        pred_pid         = e_cor["pred_PID"]
        charged_PID_pred = e_cor["charged_PID_pred"]
        neutral_PID_pred = e_cor["neutral_PID_pred"]
        pred_pos         = e_cor["pred_pos"]
        pred_ref_pt      = e_cor["pred_ref_pt"]
        extra_features   = e_cor["extra_features"]
        fakes_labels     = e_cor["fakes_labels"]
        e_cor            = e_cor["pred_energy_corr"]

        PID_logits = torch.zeros(len(e_cor), len(self.pids_charged) + len(self.pids_neutral)).float()
        PID_logits[charged_idx.cpu(), 0] = charged_PID_pred.detach().cpu()[:, 0]
        PID_logits[charged_idx.cpu(), 1] = charged_PID_pred.detach().cpu()[:, 1]
        PID_logits[charged_idx.cpu(), 4] = charged_PID_pred.detach().cpu()[:, 2]
        PID_logits[neutral_idx.cpu(), 2] = neutral_PID_pred.detach().cpu()[:, 0]
        PID_logits[neutral_idx.cpu(), 3] = neutral_PID_pred.detach().cpu()[:, 1]

        extra_features = extra_features.detach().cpu()
        extra_features = torch.cat((extra_features, PID_logits), dim=1).numpy()

        return e_cor, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels


def pid_loss(
    pid_pred_all: torch.Tensor,
    pid_true_all: torch.Tensor,
    e_true: torch.Tensor,
    mask: torch.Tensor,
    frozen: bool = False,
    name: str = "",
) -> tuple:
    if not len(pid_pred_all):
        return 0, 0
    mask = mask.bool()
    pid_pred = pid_pred_all[mask]
    pid_true = pid_true_all[mask]
    if not len(pid_pred):
        return 0, 0
    acc  = torch.sum(pid_pred == pid_true) / len(pid_pred)
    loss = CrossEntropyLoss()(pid_pred, pid_true)
    return loss, acc
