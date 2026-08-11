"""
energy_correction_charged.py
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_sum
from xformers.ops.fmha import BlockDiagonalMask
import dgl

from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, embed_scalar
from src.layers.tools_for_regression import PickPAtDCA


class ChargedEnergyCorrection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_features_global = 16
        self.in_features_gnn = 16   # GATr multivector output dim per batch
        self.pid_channels = [0, 1, 4]
        n_layers = 3
        self.args = args

        self.gatr = GATr(
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

        out_features_gnn = self.in_features_gnn
        in_features_global = self.in_features_global
        n_pid_classes = len(self.pid_channels)

        pid_layers = [nn.Linear(out_features_gnn + in_features_global + 1, 64)]
        for _ in range(n_layers - 1):
            pid_layers.append(nn.Linear(64, 64))
            pid_layers.append(nn.ReLU())
        pid_layers.append(nn.Linear(64, n_pid_classes))
        self.PID_head = nn.Sequential(*pid_layers)

        self.PickPAtDCA = PickPAtDCA()

    def charged_prediction(self, graphs_new, charged_idx, graphs_high_level_features):
        unbatched = dgl.unbatch(graphs_new)
        if len(charged_idx) > 0:
            charged_graphs = dgl.batch([unbatched[i] for i in charged_idx])
            charged_energies = self.predict(
                graphs_high_level_features,
                charged_graphs,

            )
        else:
            empty = torch.tensor([]).to(graphs_new.ndata["h"].device)
            charged_energies = [empty, empty, empty, empty]
        return charged_energies

    def predict(self, x_global_features, graphs_new=None):
        """
        Forward pass for charged energy correction.
        :param x_global_features: Global graph-level features (batch, in_features_global)
        :param graphs_new: Batched DGL graph of hit-level data
        :return: (E, direction, pid_pred, ref_pt_pred)
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

            recovered_E = x_global_features[:, 6] / x_global_features[:, 3]
            x_global_features = torch.cat((x_global_features, recovered_E.view(-1, 1)), dim=1)
            model_x = torch.cat([x_global_features, embedded_outputs_per_batch], dim=1)

        pid_pred = self.PID_head(model_x)
        p_tracks, pos, ref_pt_pred = self.PickPAtDCA.predict(x_global_features, graphs_new)
        E = torch.norm(pos, dim=1)
        pos = (pos / torch.norm(pos, dim=1).unsqueeze(1)).clone()
        return E, pos, pid_pred, ref_pt_pred

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
