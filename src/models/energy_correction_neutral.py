"""
energy_correction_neutral.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from xformers.ops.fmha import BlockDiagonalMask
import dgl

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, embed_scalar
from src.models.E_correction_module import Net
from src.layers.tools_for_regression import ECNetWrapperAvg, AverageHitsP


class NeutralEnergyCorrection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_features_global = 16
        self.in_features_gnn = 16   # GATr multivector output dim per batch
        self.pid_channels = [2, 3]
        self.args = args
        n_layers = 3

        gatr_kwargs = dict(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=4,
            in_s_channels=2,
            out_s_channels=None,
            hidden_s_channels=4,
            num_blocks=3,
            attention=SelfAttentionConfig(),
            mlp=MLPConfig(),
        )
        self.gatr = GATr(**gatr_kwargs)
        self.gatr_pid = GATr(**gatr_kwargs)

        out_features_gnn = self.in_features_gnn
        in_features_global = self.in_features_global
        n_pid_classes = len(self.pid_channels)
        out_f = 1  # Energy prediction (scalar)

        pid_layers = [nn.Linear(out_features_gnn + in_features_global, 64)]
        for _ in range(n_layers - 1):
            pid_layers.append(nn.Linear(64, 64))
            pid_layers.append(nn.ReLU())
        pid_layers.append(nn.Linear(64, n_pid_classes))
        self.PID_head = nn.Sequential(*pid_layers)

        self.model = Net(
            in_features=out_features_gnn + in_features_global,
            out_features=out_f,
            return_raw=True,
        )
        self.ec_model_wrapper_neutral_avg = ECNetWrapperAvg()
        self.AvgHits = AverageHitsP(ecal_only=True)

    def neutral_prediction(self, graphs_new, neutral_idx, features_neutral_no_nan):
        unbatched = dgl.unbatch(graphs_new)
        if len(neutral_idx) > 0:
            neutral_graphs = dgl.batch([unbatched[i] for i in neutral_idx])
            neutral_energies = self.predict(
                features_neutral_no_nan,
                neutral_graphs,
            )
            neutral_pxyz_avg = self.ec_model_wrapper_neutral_avg.predict(
                features_neutral_no_nan,
                neutral_graphs,
            )[1]
        else:
            empty = torch.tensor([]).to(graphs_new.ndata["h"].device)
            neutral_energies = [empty, empty, empty, empty]
            neutral_pxyz_avg = empty
        return neutral_energies, neutral_pxyz_avg

    def predict(self, x_global_features, graphs_new=None):
        """
        Forward pass for neutral energy correction.
        :param x_global_features: Global graph-level features (batch, in_features_global)
        :param graphs_new: Batched DGL graph of hit-level data
        :return: (E_pred, direction, pid_pred, ref_pt_pred)
        """
        if graphs_new is not None:
            batch_num_nodes = graphs_new.batch_num_nodes()
            batch_idx = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
            batch_idx = torch.tensor(batch_idx).to(graphs_new.device)

            hits_points = graphs_new.ndata["h"][:, 0:3]
            hit_type = graphs_new.ndata["h"][:, 4:8].argmax(dim=1)
            p = graphs_new.ndata["h"][:, 9]
            e = graphs_new.ndata["h"][:, 8]

            embedded_inputs = embed_point(hits_points) + embed_scalar(hit_type.view(-1, 1))
            extra_scalars = torch.cat([p.unsqueeze(1), e.unsqueeze(1)], dim=1)
            mask = self.build_attention_mask(graphs_new)
            embedded_inputs = embedded_inputs.unsqueeze(-2)

            embedded_outputs, _ = self.gatr(
                embedded_inputs, scalars=extra_scalars, attention_mask=mask
            )
            embedded_outputs_per_batch = scatter_sum(embedded_outputs[:, 0, :], batch_idx, dim=0)
            model_x = torch.cat([x_global_features, embedded_outputs_per_batch], dim=1)

            embedded_outputs_pid, _ = self.gatr_pid(
                embedded_inputs, scalars=extra_scalars, attention_mask=mask
            )
            embedded_outputs_per_batch_pid = scatter_sum(
                embedded_outputs_pid[:, 0, :], batch_idx, dim=0
            )
            model_x_pid = torch.cat([x_global_features, embedded_outputs_per_batch_pid], dim=1)

        res = self.model(model_x)
        pid_pred = self.PID_head(model_x_pid)
        E_pred = res[:, 0]

        _, p_pred, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
        p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
        return E_pred, p_pred, pid_pred, ref_pt_pred

    @staticmethod
    def obtain_batch_numbers(g):
        graphs_eval = dgl.unbatch(g)
        batch_numbers = []
        for index, gj in enumerate(graphs_eval):
            num_nodes = gj.number_of_nodes()
            batch_numbers.append(index * torch.ones(num_nodes))
        return torch.cat(batch_numbers, dim=0)

    def build_attention_mask(self, g):
        batch_numbers = self.obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )


def correct_mask_neutral(pid_neutral, neural_mask):
    """
    Filter neutral-candidate indices to keep only genuine neutral PIDs.
    """
    pid_neutral = pid_neutral.to(neural_mask.device)
    pid_neutral = torch.abs(pid_neutral)
    keep_list = torch.tensor([22, 130, 2112], device=pid_neutral.device)
    selected_pids = pid_neutral[neural_mask]
    keep_mask = torch.isin(selected_pids, keep_list)
    return neural_mask[keep_mask.to(neural_mask.device)]


def criterion_E_cor(ypred, ytrue):
    if len(ypred) > 0:
        return torch.mean(F.l1_loss(ypred, ytrue, reduction="none"))
    else:
        return 0
