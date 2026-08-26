"""
    PID predict energy correction
    The model taken from notebooks/train_energy_correction_head.py
    At first the model is fixed and the weights are loaded from earlier training
"""
import wandb
from xformers.ops.fmha import BlockDiagonalMask
from gatr.interface  import (
    embed_point,
    embed_scalar,
)
from src.layers.utils_training import obtain_clustering_for_matched_showers
from torch_scatter import scatter_add, scatter_mean
from src.utils.post_clustering_features import (
    get_post_clustering_features,
    get_extra_features,
    calculate_eta,
    calculate_phi,
)
from time import time
import numpy as np
from gatr import GATr, SelfAttentionConfig, MLPConfig
import pickle
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.utils.pid_conversion import pid_conversion_dict
# from torch_geometric.nn.models import GAT, GraphSAGE
from torch_scatter import scatter_mean, scatter_sum
from gatr import GATr
import dgl
from src.layers.tools_for_regression import ECNetWrapperAvg,PickPAtDCA, AverageHitsP,NeutralPCA,ThrustAxis
from src.models.GATr.E_correction_module import Net, CPU_Unpickler
from src.layers.regression.loss_regression import loss_position, loss_score_func, obtain_PID_charged, obtain_PID_neutral
class EnergyCorrectionWrapper(torch.nn.Module):
    def __init__(
        self,
        device,
        in_features_global=13,
        in_features_gnn=13,
        out_features_gnn=16,
        ckpt_file=None,
        gnn=True,
        pos_regression=False,
        gatr=False,
        charged=False,
        unit_p=False,
        pid_channels=0,  # PID: list of possible PID values to classify using an additional head. If empty, don't do PID.
        out_f=1,
        ignore_global_features_for_p=True,  # Whether to ignore the high-level features for the momentum regression and just use the GATr outputs
        neutral_avg=False,
        neutral_PCA=False,
        neutral_thrust_axis=False,
        simple_p_GNN=False,
        predict=True,
        args=None
    ):
        super(EnergyCorrectionWrapper, self).__init__()
        if not charged:
            self.ec_model_wrapper_neutral_avg = ECNetWrapperAvg(ILD=args.ILD if args is not None else False)
        self.charged = charged
        self.args = args
        self.predict_arg = predict
        self.simple_p_GNN = simple_p_GNN
        self.neutral_avg = neutral_avg
        self.pos_regression = pos_regression
        self.unit_p = unit_p
        self.neutral_PCA = neutral_PCA
        self.neutral_thrust_axis = neutral_thrust_axis
        self.use_gatr = gatr
        self.separate_pid_gatr = args.separate_PID_GATr
        self.n_layers_pid_head = args.n_layers_PID_head
    

        self.ignore_global_features_for_p = ignore_global_features_for_p
        if self.charged:
            self.ignore_global_features_for_p = False
        if not self.charged:
            self.model = Net(
                in_features=out_features_gnn + in_features_global, out_features=out_f,
                return_raw=True
            )
            self.model.explainer_mode = False
        # use a GAT
        if gnn:
            if self.use_gatr:
                self.gatr = GATr(
                    in_mv_channels=1,
                    out_mv_channels=1,
                    hidden_mv_channels=4,
                    in_s_channels=2,
                    out_s_channels=None,
                    hidden_s_channels=4,
                    num_blocks=3,
                    attention=SelfAttentionConfig(),  # Use default parameters for attention
                    mlp=MLPConfig(),  # Use default parameters for MLP
                )
                self.gnn = "gatr"
                if self.separate_pid_gatr and not self.charged:
                    print("Separate PID GATr")
                    self.gatr_pid = GATr(
                        in_mv_channels=1,
                        out_mv_channels=1,
                        hidden_mv_channels=4,
                        in_s_channels=2,
                        out_s_channels=None,
                        hidden_s_channels=4,
                        num_blocks=3,
                        attention=SelfAttentionConfig(),  # Use default parameters for attention
                        mlp=MLPConfig(),  # Use default parameters for MLP
                    )
        else:
            self.gnn = None
        self.pid_channels = pid_channels
        if pid_channels > 1: # 1 is just the 'other' category
            n_layers = self.n_layers_pid_head
            if n_layers == 1:
                self.PID_head = nn.Linear(out_features_gnn + in_features_global, pid_channels)   # Additional head for PID classification
            else:
                self.PID_head = nn.ModuleList()
                if self.charged:
                    self.PID_head.append(nn.Linear(out_features_gnn + in_features_global+1, 64))
                else:
                    self.PID_head.append(nn.Linear(out_features_gnn + in_features_global, 64))
                for i in range(n_layers - 1):
                    self.PID_head.append(nn.Linear(64, 64))
                    self.PID_head.append(nn.ReLU())
                self.PID_head.append(nn.Linear(64, pid_channels))
                self.PID_head = nn.Sequential(*self.PID_head)
            self.PID_head.to(device)
        self.fake_score_network = False 
        if self.fake_score_network:
            n_layers = 3
            self.fakes_head = nn.ModuleList()
            self.fakes_head.append(nn.Linear(out_features_gnn + in_features_global, 64))
            for i in range(n_layers - 1):
                self.fakes_head.append(nn.Linear(64, 64))
                self.fakes_head.append(nn.ReLU())
            self.fakes_head.append(nn.Linear(64, 1))
            self.fakes_head = nn.Sequential(*self.fakes_head)
            self.fakes_head.to(device)
        if ckpt_file is not None and ckpt_file != "" and not self.charged:
            # self.model.model = pickle.load(open(ckpt_file, 'rb'))
            with open(ckpt_file.strip(), "rb") as f:
                self.model.model = CPU_Unpickler(f).load()
                # if self.use_gatr:
                #     self.gatr = CPU_Unpickler(f).load()
            print("Loaded energy correction model weights from ECNetWrapperGNNGlobalFeaturesSeparate", ckpt_file)

        else:
            print("Not loading energy correction model weights")
        if not self.charged:
            self.model.to(device)
        self.PickPAtDCA = PickPAtDCA()
        self.AvgHits = AverageHitsP(ecal_only=True, ILD=self.args.ILD)
        self.NeutralPCA = NeutralPCA()
        self.ThrustAxis = ThrustAxis()

    def charged_prediction(self, graphs_new, charged_idx, graphs_high_level_features):
        # Prediction for charged particles
        unbatched = dgl.unbatch(graphs_new)
        if len(charged_idx) > 0:
            charged_graphs = dgl.batch([unbatched[i] for i in charged_idx])
            charged_energies = self.predict(
                graphs_high_level_features,
                charged_graphs,
                explain=self.args.explain_ec,
            )
        else:
            empty_tensor = torch.tensor([]).to(graphs_new.ndata["h"].device)
            if not self.args.regress_pos:
                charged_energies = empty_tensor
            else:
                charged_energies = [
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                ]
            if self.pid_channels:
                charged_energies += [empty_tensor, empty_tensor]
        
        return charged_energies

    def neutral_prediction(self, graphs_new, neutral_idx, features_neutral_no_nan):
        unbatched = dgl.unbatch(graphs_new)
        if len(neutral_idx) > 0:
            neutral_graphs = dgl.batch([unbatched[i] for i in neutral_idx])
            neutral_energies = self.predict(
                features_neutral_no_nan,
                neutral_graphs,
                explain=self.args.explain_ec,
            )
            neutral_pxyz_avg = self.ec_model_wrapper_neutral_avg.predict(
                features_neutral_no_nan,
                neutral_graphs,
                explain=self.args.explain_ec,
            )[1]
        else:
            empty_tensor = torch.tensor([]).to(graphs_new.ndata["h"].device)
            if not self.args.regress_pos:
                neutral_energies = empty_tensor
            else:
                neutral_energies = [
                    empty_tensor,
                    empty_tensor,
                    empty_tensor,
                ]
            if self.pid_channels:
                neutral_energies += [empty_tensor, empty_tensor]
            neutral_pxyz_avg = empty_tensor
        return neutral_energies, neutral_pxyz_avg
    def predict(self, x_global_features, graphs_new=None, explain=False):
        """
        Forward, named 'predict' for compatibility reasons
        :param x_global_features: Global features of the graphs - to be concatenated to each node feature
        :param graphs_new:
        :return:
        """
        if graphs_new is not None and self.gnn is not None:
            batch_num_nodes = graphs_new.batch_num_nodes()  # Num. of hits in each graph
            batch_idx = []
            batch_bounds = []
            for i, n in enumerate(batch_num_nodes):
                batch_idx.extend([i] * n)
                batch_bounds.append(n)
            batch_idx = torch.tensor(batch_idx).to(graphs_new.device)
            x = graphs_new.ndata["h"]
            edge_index = torch.stack(graphs_new.edges())
            hits_points = graphs_new.ndata["h"][:, 0:3]
            if self.args.ILD:
                hit_type = graphs_new.ndata["h"][:, 4:9].argmax(dim=1)
                p = graphs_new.ndata["h"][:, 10]
                e = graphs_new.ndata["h"][:, 9]
            else:
                hit_type = graphs_new.ndata["h"][:, 4:8].argmax(dim=1)
                p = graphs_new.ndata["h"][:, 9]
                e = graphs_new.ndata["h"][:, 8]
            embedded_inputs = embed_point(hits_points) + embed_scalar(
                hit_type.view(-1, 1)
            )
            extra_scalars = torch.cat(
                [ p.unsqueeze(1), e.unsqueeze(1)], dim=1
            )
            mask = self.build_attention_mask(graphs_new)
            embedded_inputs = embedded_inputs.unsqueeze(-2)
            embedded_outputs, _ = self.gatr(
                embedded_inputs, scalars=extra_scalars, attention_mask=mask
            )
            embedded_outputs_per_batch = scatter_sum(
                embedded_outputs[:, 0, :], batch_idx, dim=0
            )
            if self.charged:
                recovered_E = x_global_features[:,6]/x_global_features[:,3]
                x_global_features = torch.cat((x_global_features,recovered_E.view(-1,1) ), dim=1)

            model_x = torch.cat(
                [x_global_features, embedded_outputs_per_batch], dim=1
            )
            if self.separate_pid_gatr and not self.charged:
                
                embedded_outputs, _ = self.gatr_pid(
                    embedded_inputs, scalars=extra_scalars, attention_mask=mask
                )
               
                embedded_outputs_per_batch1 = scatter_sum(
                    embedded_outputs[:, 0, :], batch_idx, dim=0
                )
                model_x_pid = torch.cat(
                    [x_global_features, embedded_outputs_per_batch1], dim=1
                )
            else:
                model_x_pid = model_x

        if not self.charged:
            # Predict energy for neutrals using the neural network
            res = self.model(model_x)
        if self.pid_channels > 1:
            # cast at the boundary: under Lightning bf16-mixed autocast the
            # heads emit bf16; everything downstream expects fp32
            pid_pred = self.PID_head(model_x_pid).float()
        else:
            pid_pred = None
        if self.fake_score_network:
            score_pred = self.fakes_head(model_x_pid)
        else:
            score_pred = None
        if self.pos_regression:
            if self.charged:
                p_tracks, pos, ref_pt_pred = self.PickPAtDCA.predict(x_global_features, graphs_new, self.args.ILD)
                
                E = torch.norm(pos, dim=1)
                if self.unit_p:
                    pos = (pos / torch.norm(pos, dim=1).unsqueeze(1)).clone()
                return E, pos, pid_pred, ref_pt_pred, score_pred
            else:
                E_pred = res[:, 0].float()
                if torch.sum(torch.isnan(E_pred))>0:
                    print("FOUND NAANANANNANANNA!!!!!!")
                    print("nans in x_global_features", torch.sum(torch.isnan(x_global_features)))   
                    print(x_global_features) 
                # E_pred = torch.clamp(E_pred, min=0, max=None)
                _, _, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                if self.neutral_avg:
                    _, p_pred, ref_pt_pred = self.AvgHits.predict(x_global_features, graphs_new)
                elif self.neutral_PCA:
                    _, p_pred, ref_pt_pred = self.NeutralPCA.predict(x_global_features, graphs_new)
                elif self.neutral_thrust_axis:
                    _, p_pred, ref_pt_pred = self.ThrustAxis.predict(x_global_features, graphs_new)
                else:
                    p_pred = res[:, 1:4]
                    raise NotImplementedError
                if self.unit_p:
                    p_pred = (p_pred / torch.norm(p_pred, dim=1).unsqueeze(1)).clone()
                return E_pred, p_pred, pid_pred, ref_pt_pred, score_pred
        else:
            # normalize the vectors
            # E = torch.clamp(res[0].flatten(), min=0, max=None)
            # p = res[1]  # / torch.norm(res[1], dim=1).unsqueeze(1)
            # if self.use_gatr and not use_full_mv:
            #     p = p_vectors_per_batch
            # return E, p
            return torch.clamp(res[:, 0].float(), min=0, max=None)
    @staticmethod
    def obtain_batch_numbers(g):
        graphs_eval = dgl.unbatch(g)
        number_graphs = len(graphs_eval)
        batch_numbers = []
        for index in range(0, number_graphs):
            gj = graphs_eval[index]
            num_nodes = gj.number_of_nodes()
            batch_numbers.append(index * torch.ones(num_nodes))
            num_nodes = gj.number_of_nodes()
        batch = torch.cat(batch_numbers, dim=0)
        return batch

    def build_attention_mask(self, g):
        batch_numbers = self.obtain_batch_numbers(g)
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch_numbers.long()).tolist()
        )



class EnergyCorrection():
    def __init__(self, main_model):
        #super(EnergyCorrection, self).__init__()
        self.args = main_model.args
        self.get_PID_categories(main_model)
        self.get_energy_correction(main_model)
        self.pid_conversion_dict = pid_conversion_dict
        self.main_model = main_model
        self.fake_score_network = False
        self.global_step = 0
    def get_PID_categories(self, main_model):
        assert main_model.args.add_track_chis
        if len(main_model.args.classify_pid_charged):
            pids_charged = [int(x) for x in self.args.classify_pid_charged.split(",")]
        else:
            pids_charged = []
        if len(self.args.classify_pid_neutral):
            pids_neutral = [int(x) for x in self.args.classify_pid_neutral.split(",")]
        else:
            pids_neutral = []
        if len(pids_charged):
            print("Also running classification for charged particles", self.pids_charged)
        if len(pids_neutral):
            print("Also running classification for neutral particles", self.pids_neutral)
        pids_charged = [0, 1, 2, 3]  # electron, CH, NH, gamma
        pids_neutral = [0, 1, 2, 3]  # electron, CH, NH, gamma
        if self.args.restrict_PID_charge:
            print("Restricting PID classification to match charge")
            pids_charged = [0, 1]
            pids_neutral = [2, 3]
        if self.args.is_muons:
            pids_charged += [4]
            if not self.args.restrict_PID_charge:
                pids_neutral += [4]
        self.pids_charged = pids_charged
        self.pids_neutral = pids_neutral

    def get_energy_correction(self, main_model):
        # To be called by the model to initialize the energy correction modules
        ckpt_neutral = main_model.args.ckpt_neutral
        ckpt_charged = main_model.args.ckpt_charged
        dev = main_model.dev
        num_global_features = 14
        if main_model.args.is_muons:
            num_global_features += 2 # for the muon calorimeter hits and the number of muon hits
        self.model_charged = EnergyCorrectionWrapper(
            device=dev,
            in_features_global=num_global_features,
            in_features_gnn=20,
            ckpt_file=ckpt_charged,
            gnn=True,
            gatr=True,
            pos_regression=self.args.regress_pos,
            charged=True,
            pid_channels=len(self.pids_charged),
            unit_p=self.args.regress_unit_p,
            out_f=1,
            args=self.args
        )
        self.model_neutral = EnergyCorrectionWrapper(
            device=dev,
            in_features_global=num_global_features,
            in_features_gnn=20,
            ckpt_file=ckpt_neutral,
            gnn=True,
            gatr=True,
            pos_regression=self.args.regress_pos,
            pid_channels=len(self.pids_neutral),
            unit_p=self.args.regress_unit_p,
            out_f=1,  # To change to 1 for new models!!!!
            neutral_avg=True,
            predict=self.args.predict,
            args=self.args
        )

    def clustering_and_global_features(self, g, x, y, add_fakes=True):
        time_matching_start = time()
        # Match graphs
        (
            graphs_new, # Contains both fakes and true showers
            true_new, # FOR THE MATCHED SHOWERS
            sum_e, # FOR THE MATCHED + FAKE SHOWERS
            true_pid, # FOR THE MATCHED SHOWERS
            e_true_corr_daughters, # FOR THE MATCHED SHOWERS
            true_coords, # FOR THE MATCHED SHOWERS
            number_of_fakes,
            fakes_idx
        ) = obtain_clustering_for_matched_showers(
            g,
            x,
            y,
            0,
            use_gt_clusters=self.args.use_gt_clusters,
            add_fakes=add_fakes,
            truth_tracks=self.args.truth_tracking,
            fix_clusters_class=getattr(self.main_model, '_fix_clusters_class', None),
        )
        time_matching_end = time()
        # wandb.log({"time_clustering_matching": time_matching_end - time_matching_start})
        batch_num_nodes = graphs_new.batch_num_nodes()
        batch_idx = []
        for i, n in enumerate(batch_num_nodes):
            batch_idx.extend([i] * n)
        batch_idx = torch.tensor(batch_idx).to(graphs_new.ndata["h"].device)
        graphs_new.ndata["h"][:, 0:3] = graphs_new.ndata["h"][:, 0:3] / 3300
        # TODO: add global features to each node here
        graphs_sum_features = scatter_add(graphs_new.ndata["h"], batch_idx, dim=0)
        # now multiply graphs_sum_features so the shapes match
        graphs_sum_features = graphs_sum_features[batch_idx]
        # append the new features to "h" (graphs_sum_features)
        shape0 = graphs_new.ndata["h"].shape
        betas = torch.sigmoid(graphs_new.ndata["h"][:, -1])
        graphs_new.ndata["h"] = torch.cat(
            (graphs_new.ndata["h"], graphs_sum_features), dim=1
        )
        assert shape0[1] * 2 == graphs_new.ndata["h"].shape[1]
        # print("Also computing graph-level features")
        graphs_high_level_features = get_post_clustering_features(
            graphs_new, sum_e, is_muons=self.main_model.args.is_muons, add_hit_chis=self.args.add_track_chis, ILD=self.args.ILD
        )
        extra_features = get_extra_features(graphs_new, betas)
        pred_energy_corr = torch.ones(graphs_high_level_features.shape[0]).to(
            graphs_new.ndata["h"].device
        )
        if self.args.regress_pos:
            pred_pos = torch.ones((graphs_high_level_features.shape[0], 3)).to(
                graphs_new.ndata["h"].device
            )
            pred_pid = torch.ones((graphs_high_level_features.shape[0])).to(
                graphs_new.ndata["h"].device
            ).long()
        else:
            pred_pos = None
            pred_pid = torch.ones((graphs_high_level_features.shape[0])).to(
                graphs_new.ndata["h"].device
            ).long()
        node_features_avg = scatter_mean(graphs_new.ndata["h"], batch_idx, dim=0)[
            :, 0:3
        ]
        eta, phi = calculate_eta(
            node_features_avg[:, 0],
            node_features_avg[:, 1],
            node_features_avg[:, 2],
        ), calculate_phi(node_features_avg[:, 0], node_features_avg[:, 1])
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, node_features_avg), dim=1
        )
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, eta.view(-1, 1)), dim=1
        )
        graphs_high_level_features = torch.cat(
            (graphs_high_level_features, phi.view(-1, 1)), dim=1
        )
        num_tracks = graphs_high_level_features[:, 7]
        num_hits = graphs_high_level_features[:, 2]
        charged_idx = torch.where((num_tracks >= 1))[0]
        neutral_idx = torch.where((num_tracks < 1))[0]
        # assert their union is the whole set
        assert len(charged_idx) + len(neutral_idx) == len(num_tracks)
        assert (
            graphs_high_level_features.shape[0] == graphs_new.batch_num_nodes().shape[0]
        )
        features_neutral_no_nan = graphs_high_level_features[neutral_idx]
        # features_neutral_no_nan[features_neutral_no_nan != features_neutral_no_nan] = 0  # only catches NaN, not inf
        features_neutral_no_nan[~torch.isfinite(features_neutral_no_nan)] = 0
        features_charged_no_nan = graphs_high_level_features[charged_idx]
        # features_charged_no_nan[features_charged_no_nan != features_charged_no_nan] = 0  # only catches NaN, not inf
        features_charged_no_nan[~torch.isfinite(features_charged_no_nan)] = 0
        # if self.args.ec_model == "gat" or self.args.ec_model == "gat-concat":
        return (
            graphs_new,
            graphs_high_level_features,
            charged_idx,
            neutral_idx,
            features_neutral_no_nan,
            sum_e,
            pred_pos,
            true_new,
            true_pid,
            true_coords,
            batch_idx,
            e_true_corr_daughters,
            pred_energy_corr,
            pred_pid,
            features_charged_no_nan,
            number_of_fakes,
            extra_features,
            fakes_idx
        )

    def forward_correction(self, g, x, y, return_train):
        time_matching_start = time()
        (
            graphs_new,
            graphs_high_level_features,
            charged_idx,
            neutral_idx,
            features_neutral_no_nan,
            sum_e,
            pred_pos,
            true_new,
            true_pid,
            true_coords,
            batch_idx,
            e_true_corr_daughters,
            pred_energy_corr,
            pred_pid,
            features_charged_no_nan,
            number_of_fakes,
            extra_features,
            fakes_idx
        ) = self.clustering_and_global_features(g, x, y, add_fakes=self.args.predict)
        charged_energies = self.model_charged.charged_prediction(
            graphs_new, charged_idx, features_charged_no_nan
        )
        neutral_energies, neutral_pxyz_avg = self.model_neutral.neutral_prediction(
            graphs_new, neutral_idx, features_neutral_no_nan
        )
        if self.args.regress_pos:
            if len(self.pids_charged):
                charged_energies, charged_positions, charged_PID_pred, charged_ref_pt_pred, charged_score_pred= charged_energies # charged_pxyz_pred: we are also storing the xyz of the track, to see the effect of the weirdly fitted tracks on the results
            else:
                charged_energies, charged_positions, _ = charged_energies
            if len(self.pids_neutral):
                neutral_energies, neutral_positions, neutral_PID_pred, neutral_ref_pt_pred, neutral_score_pred= neutral_energies
            else:
                neutral_energies, neutral_positions, _ = neutral_energies
        if self.args.explain_ec:
            assert not self.args.regress_pos, "not implemented"
            (
                charged_energies,
                charged_energies_shap_vals,
                charged_energies_ec_x,
            ) = charged_energies
            (
                neutral_energies,
                neutral_energies_shap_vals,
                neutral_energies_ec_x,
            ) = neutral_energies
            shap_vals = (
                torch.ones(
                    graphs_high_level_features.shape[0],
                    charged_energies_shap_vals[0].shape[1],
                )
                .to(graphs_new.ndata["h"].device)
                .detach()
                .cpu()
                .numpy()
            )
            ec_x = torch.zeros(
                graphs_high_level_features.shape[0],
                charged_energies_ec_x.shape[1],
            )
            shap_vals[charged_idx.detach().cpu().numpy()] = charged_energies_shap_vals[
                0
            ]
            shap_vals[neutral_idx.detach().cpu().numpy()] = neutral_energies_shap_vals[
                0
            ]
            ec_x[charged_idx.detach().cpu().numpy()] = charged_energies_ec_x[0]
            ec_x[neutral_idx.detach().cpu().numpy()] = neutral_energies_ec_x[0]
        # dummy loss to make it work without complaining about not using params in loss
        # the GATr EC models can emit bf16 under autocast while
        # pred_energy_corr is fp32 (crashes index_put on dtype mismatch);
        # cast to the destination dtype
        pred_energy_corr[charged_idx.flatten()] = (
            charged_energies.to(pred_energy_corr.dtype) #/ sum_e.flatten()[charged_idx.flatten()]
        )
        pred_energy_corr[neutral_idx.flatten()] = (
            neutral_energies.to(pred_energy_corr.dtype) #/ sum_e.flatten()[neutral_idx.flatten()]
        )
        if self.fake_score_network:
            score_object = pred_pid.clone()
            if len(charged_idx):
                score_object[charged_idx.flatten()] = charged_score_pred.to(score_object.dtype)
            if len(neutral_idx):
                score_object[neutral_idx.flatten()] = neutral_score_pred.to(score_object.dtype)
        

        if len(self.pids_charged):
            if len(charged_idx):
                charged_PID_pred_for_labels = charged_PID_pred.detach().float().cpu()
                if charged_PID_pred_for_labels.ndim == 1:
                    charged_PID_pred_for_labels = charged_PID_pred_for_labels.reshape(1, -1)
                charged_PID_pred1 = np.array(self.pids_charged)[np.argmax(charged_PID_pred_for_labels, axis=1)]  #0,1,2
            else:
                charged_PID_pred1 = []
            pred_pid[charged_idx.flatten()] = torch.tensor(charged_PID_pred1).long().to(charged_idx.device)

        if len(self.pids_neutral):
            if len(neutral_idx):
                neutral_PID_pred_for_labels = neutral_PID_pred.detach().float().cpu()
                if neutral_PID_pred_for_labels.ndim == 1:
                    neutral_PID_pred_for_labels = neutral_PID_pred_for_labels.reshape(1, -1)
                neutral_PID_pred1 = np.array(self.pids_neutral)[np.argmax(neutral_PID_pred_for_labels, axis=1)] #0,1
            else:
                neutral_PID_pred1 = []
            pred_pid[neutral_idx.flatten()] = torch.tensor(neutral_PID_pred1).long().to(neutral_idx.device)
        

        pred_energy_corr[pred_energy_corr < 0] = 0.0
        if self.args.regress_pos:
            pred_ref_pt = torch.ones_like(pred_pos)
            if len(charged_idx):
                pred_ref_pt[charged_idx.flatten()] = charged_ref_pt_pred.to(pred_ref_pt.device).float()
                pred_pos[charged_idx.flatten()] = charged_positions.float().to(pred_pos.device)
            if len(neutral_idx):
                pred_ref_pt[neutral_idx.flatten()] = neutral_ref_pt_pred.to(neutral_idx.device).float()
                pred_pos[neutral_idx.flatten()] = neutral_positions.to(neutral_idx.device).float()
            pred_energy_corr = {
                "pred_energy_corr": pred_energy_corr,
                "pred_pos": pred_pos,
                "neutrals_idx": neutral_idx.flatten(),
                "charged_idx": charged_idx.flatten(),
                "pred_ref_pt": pred_ref_pt,
                "extra_features": extra_features,
                "fakes_labels": fakes_idx
            }
            if len(self.pids_charged) or len(self.pids_neutral):
                pred_energy_corr["pred_PID"] = pred_pid
                pred_energy_corr["charged_PID_pred"] = charged_PID_pred
                pred_energy_corr["neutral_PID_pred"] = neutral_PID_pred
                if self.fake_score_network:
                    pred_energy_corr["score_object"]=score_object
                    pred_energy_corr["fakes_idx"] = fakes_idx
    
        if return_train:
            return (
                x,
                pred_energy_corr,
                true_new,
                sum_e,
                true_pid,
                true_new,
                true_coords,
                number_of_fakes
            )
        else:
            if self.args.explain_ec:
                return (
                    x,
                    pred_energy_corr,
                    true_new,
                    sum_e,
                    graphs_new,
                    batch_idx,
                    graphs_high_level_features,
                    true_pid,
                    e_true_corr_daughters,
                    shap_vals,
                    ec_x,
                    number_of_fakes
                )
            return (
                x,
                pred_energy_corr,
                true_new,
                sum_e,
                graphs_new,
                batch_idx,
                graphs_high_level_features,
                true_pid,
                e_true_corr_daughters,
                true_coords,
                number_of_fakes
            )
    @staticmethod
    def criterion(ypred, ytrue, step):
        return F.l1_loss(ypred, ytrue)

    def get_loss(self, batch_g, y, result, stats, fixed):
        (
            model_output,
            dic_e_cor,
            e_true,
            e_sum_hits,
            new_graphs,
            batch_id,
            graph_level_features,
            pid_true_matched,
            e_true_corr_daughters,
            part_coords_matched,
            num_fakes
        ) = result
        e_cor = dic_e_cor["pred_energy_corr"]
        mask_neutral_for_loss = correct_mask_neutral(torch.tensor(pid_true_matched), dic_e_cor["neutrals_idx"])

        e_true_neutrals = e_true[mask_neutral_for_loss]
        e_pred_neutrals = e_cor[mask_neutral_for_loss]
        e_reco_neutrals = e_sum_hits[mask_neutral_for_loss]
        in_distribution = (torch.abs(e_true_neutrals-e_reco_neutrals)/e_true_neutrals)<0.6
        ypred = e_pred_neutrals[in_distribution]
        ybatch = e_true_neutrals[in_distribution]
        if len(ypred)>0:
            pid_neutrals = torch.tensor(pid_true_matched)[mask_neutral_for_loss.cpu()].to(ypred.device)
            loss_EC_neutrals, stats = criterion_E_cor(ypred.flatten(), ybatch.flatten(), self.global_step, torch.abs(pid_neutrals[in_distribution]), stats, frozen=fixed)
        else:
            loss_EC_neutrals = 0
        filt_neutrons = (e_true[dic_e_cor["neutrals_idx"]] < 5).cpu() & (torch.tensor(pid_true_matched)[dic_e_cor["neutrals_idx"].cpu()] == 2112)
        # loss_EC_neutrons = torch.nn.L1Loss()(  # returns NaN on empty tensor when filt_neutrons is all False
        #     e_cor[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu(), e_true[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu()
        # )
        if filt_neutrons.any():
            loss_EC_neutrons = torch.nn.L1Loss()(
                e_cor[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu(), e_true[dic_e_cor["neutrals_idx"]][filt_neutrons].detach().cpu()
            )
        else:
            loss_EC_neutrons = torch.tensor(0.0)

        ########### loss PID ###########
        # correct assignation of PIDs without track and go from PID montecarlo number to int
        if len(self.pids_charged):
            charged_PID_pred, charged_PID_true_onehot, mask_charged = obtain_PID_charged(dic_e_cor,pid_true_matched, self.pids_charged, self.args, self.pid_conversion_dict)

        if len(self.pids_neutral):
            neutral_PID_pred, neutral_PID_true_onehot, mask_neutral = obtain_PID_neutral(dic_e_cor,pid_true_matched, self.pids_neutral, self.args, self.pid_conversion_dict)

        if len(self.pids_charged):
            loss_charged_pid,acc_charged, stats= pid_loss_weighted(charged_PID_pred, charged_PID_true_onehot,e_true[dic_e_cor["charged_idx"]], mask_charged, stats, fixed, "charged",
                weighting=_pid_weighting_for(self.args, "charged"),
                beta=getattr(self.args, "pid_class_weighting_beta", 0.999),
                soft_muon_cut=getattr(self.args, "pid_soft_muon_cut", 0.0))

        if len(self.pids_neutral):
            loss_neutral_pid,acc_neutral, stats = pid_loss_weighted(neutral_PID_pred, neutral_PID_true_onehot,e_true, mask_neutral, stats, fixed, "neutral",
                weighting=_pid_weighting_for(self.args, "neutral"),
                beta=getattr(self.args, "pid_class_weighting_beta", 0.999),
                soft_muon_cut=getattr(self.args, "pid_soft_muon_cut", 0.0))

        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            wandb.log({
                "loss_EC_neutrals": loss_EC_neutrals,
                "loss_EC_neutrons": loss_EC_neutrons,
                "loss_charged_pid": loss_charged_pid,
                "loss_neutral_pid": loss_neutral_pid,
            })
        ########### loss score ###########
        if self.fake_score_network:
            loss_score = loss_score_func(dic_e_cor)
        else:
            loss_score = 0
        return loss_EC_neutrals, 0, loss_neutral_pid, loss_charged_pid, loss_score, stats

    def get_validation_step_outputs(self, batch_g, y, result):
        if self.args.explain_ec:
            (
                model_output,
                e_cor,
                e_true,
                e_sum_hits,
                new_graphs,
                batch_id,
                graph_level_features,
                pid_true_matched,
                e_true_corr_daughters,
                shap_vals,
                ec_x,
                num_fakes
            ) = result
        else:
            (
                model_output,
                e_cor,
                e_true,
                e_sum_hits,
                new_graphs,
                batch_id,
                graph_level_features,
                pid_true_matched,
                e_true_corr_daughters,
                coords_true,
                num_fakes,
            ) = result
        if self.args.regress_pos:
            if len(self.pids_charged):
                charged_idx = e_cor["charged_idx"]
            if len(self.pids_neutral):
                neutral_idx = e_cor["neutrals_idx"]
            pred_pid = e_cor["pred_PID"]
            e_cor, pred_pos, pred_ref_pt, extra_features, fakes_labels, charged_PID_pred, neutral_PID_pred = e_cor["pred_energy_corr"], e_cor["pred_pos"], e_cor[
                "pred_ref_pt"], e_cor["extra_features"], e_cor["fakes_labels"], e_cor["charged_PID_pred"], e_cor["neutral_PID_pred"]
            max_len = max(len(self.pids_charged), len(self.pids_neutral))
            charged_PID_pred_cpu = charged_PID_pred.detach().float().cpu()
            neutral_PID_pred_cpu = neutral_PID_pred.detach().float().cpu()
            if charged_PID_pred_cpu.ndim == 1:
                charged_PID_pred_cpu = charged_PID_pred_cpu.reshape(1, -1)
            if neutral_PID_pred_cpu.ndim == 1:
                neutral_PID_pred_cpu = neutral_PID_pred_cpu.reshape(1, -1)
            if self.args.restrict_PID_charge:
                PID_logits = torch.zeros(len(e_cor), len(self.pids_charged)+ len(self.pids_neutral)).float()
                PID_logits = PID_logits.clone()
                charged_idx_cpu = charged_idx.cpu()
                neutral_idx_cpu = neutral_idx.cpu()
                if len(charged_idx_cpu) and charged_PID_pred_cpu.shape[1] >= 3:
                    PID_logits[charged_idx_cpu,0] = charged_PID_pred_cpu[: len(charged_idx_cpu), 0]
                    PID_logits[charged_idx_cpu,1] = charged_PID_pred_cpu[: len(charged_idx_cpu), 1]
                    PID_logits[charged_idx_cpu,4] = charged_PID_pred_cpu[: len(charged_idx_cpu), 2]
                if len(neutral_idx_cpu) and neutral_PID_pred_cpu.shape[1] >= 2:
                    PID_logits[neutral_idx_cpu,2] = neutral_PID_pred_cpu[: len(neutral_idx_cpu), 0]
                    PID_logits[neutral_idx_cpu,3] = neutral_PID_pred_cpu[: len(neutral_idx_cpu), 1]
            else:
                PID_logits = torch.zeros(len(e_cor), max_len).float()
                PID_logits[charged_idx.cpu()] = charged_PID_pred_cpu
                PID_logits[neutral_idx.cpu()] = neutral_PID_pred_cpu

            extra_features = extra_features.detach().float().cpu()
            extra_features = torch.cat((extra_features, PID_logits), dim=1).numpy()

        else:
            pred_pos = None
            pred_ref_pt = None
            e_cor = None
            pred_pid = None
            extra_features = None
            fakes_labels = None
        
        return e_cor, pred_pos, pred_ref_pt, pred_pid, num_fakes, extra_features, fakes_labels





def criterion_E_cor(ypred, ytrue, step, pid_neutrals, stats, frozen=False):
    if len(ypred)>0:
        stats = 0
        return torch.mean(F.l1_loss(ypred, ytrue, reduction='none')), stats
    else:
        return 0, stats


def _pid_weighting_for(args, name):
    """Resolve the class-weighting mode for one PID head.

    The two heads have very different balance, so one global setting is the
    wrong granularity (measured on 2500 DELPHI events, reconstructable truth
    objects only):

        charged [0,1,4]  electron 50.3% obj / 20.5% E, charged hadron 47.3/77.1,
                         muon 2.3/2.4          -> 21x imbalance, but the rare
                         class carries energy in PROPORTION to its count
        neutral [2,3]    neutral hadron 16.8% obj / 33.8% E, photon 83.2/66.2
                         -> only 5x imbalance, but the rare class carries TWICE
                         its share of the energy (E/obj 2.32 vs 0.92)

    So the neutral head has a physics case for upweighting -- neutral hadrons
    are exactly where particle flow is supposed to beat a classical
    reconstruction, and getting them wrong costs visible-energy resolution
    directly. The charged head mostly does not: upweighting muons 21x buys
    confusion-matrix accuracy on 2.4% of the energy.

    Per-head flags fall back to the global one, which falls back to "none".
    """
    per_head = getattr(args, "pid_class_weighting_" + name, None)
    if per_head:
        return per_head
    return getattr(args, "pid_class_weighting", "none") or "none"


def pid_loss_weighted(neutral_PID_pred, neutral_PID_true_onehot, e_true, mask_neutral,
                      stats, frozen=False, name="", weighting="none", beta=0.999,
                      soft_muon_cut=0.0):
    """Cross-entropy on the PID head, optionally class-balanced.

    **Both extras default to OFF, reproducing the previous behaviour exactly**
    (plain unweighted `CrossEntropyLoss`, no candidate filtering). This module is
    shared by DELPHI, CLD and ALLEGRO — `Gatr_pf_e_noise.py`, `mask3d_model.py`
    and seven other GATr variants all import it — so neither may become a default.
    `CrossEntropyLoss(weight=None)` is identical to `CrossEntropyLoss()`.

    `weighting` — how to build the per-class weights from the running class
    counts. Measured mu:e weight ratios on DELPHI's charged head
    (`pids_charged = [0, 1, 4]`, counts e 36 499 / charged-hadron 34 299 /
    muon 1 734):

        none       1.00   uniform (previous behaviour)
        effective  1.21   class-balanced effective number, arXiv:1901.05555
        sqrt_inv   4.59   1/sqrt(N)
        inv       21.05   1/N inverse frequency

    `inv` is aggressive: it buys PID accuracy on muons, which carry ~1.7 % of the
    visible energy, at the cost of the energy resolution that is the actual
    figure of merit. `sqrt_inv` is the middle ground. Note `effective` saturates
    at an effective count of `1/(1-beta)` = 1000 for the default beta, and every
    DELPHI class has N >> 1000, so it is nearly inert unless beta is raised to
    ~1 - 1/N_typical.

    The counts are accumulated per process, so under DDP each rank builds its own
    weights. They converge to the same distribution but are not identical
    step-to-step; that is inherited behaviour, not new.

    `soft_muon_cut` (GeV, 0 = off) — drop charged candidates whose TRUE class is
    muon and whose true energy is below the cut. Below ~1.5 GeV a muon does not
    reach the muon chambers, so the label is not learnable. The original
    expression for this was `torch.argmax(pid_true) == 2` with no `dim`, which
    reduces over the FLATTENED tensor to a single scalar: since pid_true is
    one-hot it silently meant "if the batch's FIRST candidate is a muon, drop
    EVERY candidate under the cut, of any class". Reduced over the class axis
    here, and e_true is flattened so an (N, 1) energy cannot broadcast the mask
    to (N, N).
    """
    if len(neutral_PID_pred):
        mask_neutral = mask_neutral.bool()
        pid_pred = neutral_PID_pred[mask_neutral]
        pid_true = neutral_PID_true_onehot[mask_neutral]

        weights = None
        if weighting and weighting != "none":
            key = "counts_pid_" + name
            # Created in Gatr_pf_e_noise.on_train_epoch_start, but ONLY when
            # current_epoch == 0 — so it is absent when resuming with
            # --resume-ckpt at a later epoch. Create it lazily rather than
            # turning that into a KeyError hours into a run.
            if key not in stats:
                stats[key] = {}
            if not frozen:
                for c in neutral_PID_true_onehot.argmax(dim=1).tolist():
                    stats[key][c] = stats[key].get(c, 0) + 1
            num_classes = neutral_PID_true_onehot.shape[1]
            counts = torch.tensor(
                [stats[key].get(i, 1) for i in range(num_classes)],
                dtype=torch.float, device=neutral_PID_pred.device,
            )
            counts[counts == 0] = 1
            if weighting == "sqrt_inv":
                weights = 1.0 / counts.sqrt()
            elif weighting == "effective":
                b = torch.tensor(float(beta), device=counts.device)
                weights = 1.0 / ((1.0 - torch.pow(b, counts)) / (1.0 - float(beta)))
            else:  # "inv"
                weights = 1.0 / counts
            weights = weights / weights.mean()

        if soft_muon_cut and name == "charged":
            e_true_ = e_true[mask_neutral]
            mask_muons = (torch.argmax(pid_true, dim=1) == 2) & (
                e_true_.view(-1) < float(soft_muon_cut)
            )
            pid_pred = pid_pred[~mask_muons]
            pid_true = pid_true[~mask_muons]

        if len(pid_pred):
            # NOTE: this compares raw logits to a one-hot target, so it is very
            # nearly always 0. Pre-existing; left as-is so logged values do not
            # silently change. It is a logged metric only, not part of the loss.
            acc = torch.sum(pid_pred == pid_true) / len(pid_pred)
            return torch.nn.CrossEntropyLoss(weight=weights)(
                pid_pred,
                pid_true
            ), acc, stats
        else:
            return 0, 0, stats
    else:
        return 0, 0, stats


def correct_mask_neutral(pid_neutral, neural_mask):
    """
    pid_neutral: tensor of PIDs (shape [N])
    neural_mask: tensor of indices of neutral candidates (e.g. LongTensor)

    we remove indices where pid is in remove list
    """
    pid_neutral = pid_neutral.to(neural_mask.device)
    pid_neutral = torch.abs(pid_neutral)
    # PIDs to remove
    #remove_list = torch.tensor([-211, 211, -11, 11, 13, -13, 2212, 321], device=pid_neutral.device)
    keep_list = torch.tensor([22, 130, 2112], device=pid_neutral.device)

    # get PIDs corresponding to the given indices
    selected_pids = pid_neutral[neural_mask]          # <- index access
    # build mask: True = keep, False = remove
    keep_mask = torch.isin(selected_pids, keep_list)

    # filter indices
    corrected_indices = neural_mask[keep_mask.to(neural_mask.device)]

    return corrected_indices

def correct_mask_charged(pid_neutral, neural_mask, E):
    """
    pid_neutral: tensor of PIDs (shape [N])
    neural_mask: tensor of indices of neutral candidates (e.g. LongTensor)

    we remove indices where pid is in remove list
    """
    pid_neutral = pid_neutral.to(neural_mask.device)
    pid_neutral = torch.abs(pid_neutral)
    # PIDs to remove
    #remove_list = torch.tensor([-211, 211, -11, 11, 13, -13, 2212, 321], device=pid_neutral.device)
    keep_list = torch.tensor([22, 130, 2112], device=pid_neutral.device)

    # get PIDs corresponding to the given indices
    selected_pids = pid_neutral[neural_mask]          # <- index access
    # build mask: True = keep, False = remove
    keep_mask = torch.isin(selected_pids, keep_list)

    # filter indices
    corrected_indices = neural_mask[keep_mask.to(neural_mask.device)]

    return corrected_indices
