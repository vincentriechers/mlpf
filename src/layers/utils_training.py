
from lightning.pytorch.callbacks import BaseFinetuning
import os
import torch
import torch.nn as nn
import dgl
from src.layers.inference_oc import (
    get_clustering,
)
from src.layers.inference_oc import hfdb_obtain_labels, clustering_obtain_labels, DPC_custom_CLD, DPC_custom_CLD_240
from src.layers.inference_oc import match_showers
# import torch_cmspepr  # unused here; compiled ext not available in the gatr:v9 container
from src.layers.inference_oc import remove_bad_tracks_from_cluster_v1
from src.layers.inference_oc import _fix_labels_for_classes
from src.layers.inference_oc import compact_labels_preserve_zero
# from src.layers.dpc_track_seeded import DPC_track_seeded  # module absent from maskIPA snapshot; only used in commented-out code
class FreezeClustering(BaseFinetuning):
    def __init__(
        self,
    ):
        super().__init__()
        # self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # print("freezing the following module:", pl_module)
        # freeze any module you want
        # Here, we are freezing `feature_extractor`

        # The clustering stack has DIFFERENT module names per architecture, so
        # freeze whichever exist instead of hardcoding one model's:
        #
        #   GATr / HitPF   ScaledGooeyBatchNorm2_1, gatr, clustering, beta
        #   Mask3D/AttnIPA input_net, encoder, decoder
        #
        # The hardcoded GATr list made every Mask3D stage-2 run die on startup
        # with `AttributeError: 'ExampleWrapper' object has no attribute
        # 'ScaledGooeyBatchNorm2_1'` (job 45144848) — --correction itself is
        # supported by mask3d_model (it builds an ECAdapter), it was only this
        # callback that assumed GATr.
        CLUSTERING_MODULES = (
            "ScaledGooeyBatchNorm2_1", "gatr", "clustering", "beta",   # GATr
            "input_net", "encoder", "decoder",                         # Mask3D
        )
        frozen = []
        for name in CLUSTERING_MODULES:
            mod = getattr(pl_module, name, None)
            if isinstance(mod, torch.nn.Module):
                self.freeze(mod)
                frozen.append(name)
        if not frozen:
            # Never let --freeze-clustering silently do nothing: a stage-2 run
            # that trains the clustering backbone alongside the EC heads is not
            # the recipe, and would look perfectly healthy in the logs.
            raise RuntimeError(
                "--freeze-clustering: no known clustering module found on "
                f"{type(pl_module).__name__}. Tried {CLUSTERING_MODULES}. Add "
                "this architecture's module names to CLUSTERING_MODULES in "
                "src/layers/utils_training.py:FreezeClustering."
            )
        print("CLUSTERING HAS BEEN FROZEN:", ", ".join(frozen))

    def finetune_function(self, pl_module, current_epoch, optimizer):
        print("Not finetunning")
        # # When `current_epoch` is 10, feature_extractor will start training.
        # if current_epoch == self._unfreeze_at_epoch:
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor,
        #         optimizer=optimizer,
        #         train_bn=True,
        #     )


def obtain_batch_numbers(x, g):
    counts = g.batch_num_nodes()
    return torch.repeat_interleave(
        torch.arange(len(counts), device=x.device), counts
    )



def add_lonely_tracks(g, labels, p_max=2.0):
    """Recovery option (env ADD_LONELY_TRACKS=1): promote 'lonely' tracks — track hits
    (hit_type==1) left in label 0 (noise), i.e. not picked up by the clustering step —
    with momentum < p_max GeV to their own new clusters. Because it runs before the
    energy-correction head, these soft charged tracks are then reconstructed as particles
    (energy from the track), so they contribute to the visible mass/energy."""
    ht = g.ndata["hit_type"].view(-1)
    p = g.ndata["p_hits"].view(-1)
    lab = labels.view(-1)
    lonely = (ht == 1) & (lab == 0) & (p < p_max)
    idx = torch.nonzero(lonely).view(-1)
    if idx.numel() == 0:
        return labels
    labels = labels.clone()
    start = int(labels.max().item()) + 1
    labels[idx] = torch.arange(start, start + idx.numel(), device=labels.device, dtype=labels.dtype)
    return labels


def obtain_clustering_for_matched_showers(
    batch_g, model_output, y_all, local_rank, use_gt_clusters=False, add_fakes=True, truth_tracks=False,
    fix_clusters_class=None, clust_dim=3,
    precomputed_labels=None,
):
    """Match predicted showers to GT particles per event.

    `precomputed_labels` (optional): a flat (total_hits,) int64 tensor of
    cluster IDs (0 = noise, 1..K = cluster) in graph node order. When set,
    skip the DPC + bad-track-filter step and use these labels directly —
    used by the Mask3D path so the EC pipeline doesn't redundantly
    recluster the ECAdapter's synthetic coords. coords/beta on the graph
    are still set from `model_output` so downstream feature computation
    (e.g. the EC heads' `graphs_new.ndata["h"]`) is unchanged.
    """
    if use_gt_clusters:
        pass  # GT clusters mode: labels come from particle_number
    graphs_showers_matched = []
    graphs_showers_fakes = []
    true_energy_showers = []
    reco_energy_showers = []
    reco_energy_showers_fakes = []
    energy_true_daughters = []
    y_pids_matched = []
    y_coords_matched = []
    all_cluster_labels = []
    if not  use_gt_clusters:
        batch_g.ndata["coords"] = model_output[:, 0:clust_dim]
        batch_g.ndata["beta"] = model_output[:, clust_dim]
    graphs = dgl.unbatch(batch_g)
    # Per-event hit-offset table for slicing precomputed_labels.
    if precomputed_labels is not None:
        node_counts = batch_g.batch_num_nodes()                # (B,) tensor
        hit_offsets = torch.zeros(len(graphs) + 1, dtype=torch.long,
                                  device=node_counts.device)
        hit_offsets[1:] = node_counts.cumsum(0)
    batch_id = y_all.batch_number
    for i in range(0, len(graphs)):
        mask = batch_id == i
        dic = {}
        dic["graph"] = graphs[i]
        y = y_all.copy()
        # if "unique_list_particles" in y.__dict__:
        #    del y.unique_list_particles
        y.mask(mask.flatten())
        dic["part_true"] = y
        if not  use_gt_clusters:
            betas = torch.sigmoid(dic["graph"].ndata["beta"])
            X = dic["graph"].ndata["coords"]
        clustering_mode = "dbscan"
        if clustering_mode == "clustering_normal":
            labels = clustering_obtain_labels( X,torch.sigmoid(betas.view(-1)), betas.device,  tbeta=0.2, td=0.05)
        elif clustering_mode == "dbscan":
            if use_gt_clusters:
                labels = dic["graph"].ndata["particle_number"].type(torch.int64)
            elif precomputed_labels is not None:
                # Mask3D path: use the model's per-(query, hit) argmax labels
                # directly. They're already dense (0 = noise, 1..K = cluster)
                # via labels_from_masks, so no DPC + bad-track + re-densify
                # cycle is needed. Same node order as the graph.
                #
                # NOTE: remove_bad_tracks_from_cluster_v1 must NOT be applied
                # here. On the Mask3D path the EC pipeline pre-computes
                # corrections_per_shower / pred_pos / pred_pid / etc. indexed
                # by THESE original mask labels (enforced by
                # --mask3d-use-mask-labels). Mutating/re-densifying labels
                # after that desyncs match_showers from the EC outputs and
                # the per-shower arrays end up different lengths
                # (pd.DataFrame -> "All arrays must be of the same length").
                # To clean bad tracks on this path, do it where
                # `labels_from_masks` is generated, BEFORE the EC runs.
                labels = (
                    precomputed_labels[hit_offsets[i]:hit_offsets[i + 1]]
                    .to(model_output.device).long()
                )
            else:
                #labels =DPC_track_seeded(X, dic["graph"], model_output.device)
                labels =DPC_custom_CLD(X, dic["graph"], model_output.device)
                if not truth_tracks:
                    labels, _ = remove_bad_tracks_from_cluster_v1(dic["graph"], labels)
                    labels = compact_labels_preserve_zero(labels)  # remap positives to 1..N, keep 0=noise
                if os.environ.get("ADD_LONELY_TRACKS") == "1":
                    labels = add_lonely_tracks(dic["graph"], labels, float(os.environ.get("LONELY_TRACK_PMAX", "2.0")))
                # labels = clustering_obtain_labels( X,betas.view(-1), betas.device,  tbeta=0.7, td=0.3)
                #if labels.min() == 0 and labels.sum() == 0:
                #    labels += 1  # Quick hack

        all_cluster_labels.append(labels)

        if fix_clusters_class:
            labels = _fix_labels_for_classes(
                labels, dic["graph"], dic["part_true"], fix_clusters_class, model_output.device
            )
            if not truth_tracks:
                labels, _ = remove_bad_tracks_from_cluster_v1(dic["graph"], labels)
                labels = compact_labels_preserve_zero(labels)
        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
        if os.environ.get("DUMP_NH_FATE") == "1":
            # Diagnostic: for each true neutral hadron (K_L/n), trace where its CALO energy
            # goes under the model's clustering -> reconstructed (owns its dominant cluster),
            # absorbed (dominant cluster owned by another particle), or noise (label 0).
            import numpy as _np
            _g = dic["graph"]; _ht = _g.ndata["hit_type"].view(-1).long().cpu().numpy()
            _calo = _ht != 1
            _pn = _g.ndata["particle_number"].view(-1).long().cpu().numpy()[_calo]
            _eh = _g.ndata["e_hits"].view(-1).float().cpu().numpy()[_calo]
            _lb = labels.view(-1).long().cpu().numpy()[_calo]
            _pt = dic["part_true"]; _pids = _pt.pid.view(-1).cpu().numpy(); _Et = _pt.E_corrected.view(-1).cpu().numpy()
            _gs = _pt.gen_status.view(-1).cpu().numpy()
            _rows = []
            # row format: pid, gen_status, true_E, Edep, n_calo_hits, noise_frac, fate(0=recon,1=noise,2=absorbed,3=no-deposit), owner_pid
            for _k in range(_pids.shape[0]):
                if int(abs(_pids[_k])) not in (130, 2112):
                    continue
                _hk = _pn == (_k + 1); _Edep = float(_eh[_hk].sum()); _nh = int(_hk.sum())
                if _Edep <= 0:
                    _rows.append((float(_pids[_k]), float(_gs[_k]), float(_Et[_k]), 0.0, _nh, 0.0, 3, 0)); continue
                _labs = _lb[_hk]; _ek = _eh[_hk]
                _bL = 0; _bE = -1.0
                for _L in _np.unique(_labs):
                    _e = _ek[_labs == _L].sum()
                    if _e > _bE: _bE = _e; _bL = int(_L)
                _noise = float(_ek[_labs == 0].sum()) / _Edep
                if _bL == 0:
                    _rows.append((float(_pids[_k]), float(_gs[_k]), float(_Et[_k]), _Edep, _nh, _noise, 1, 0)); continue
                _inL = _lb == _bL; _pin = _pn[_inL]; _ein = _eh[_inL]
                _oP = 0; _oE = -1.0
                for _p in _np.unique(_pin):
                    _e = _ein[_pin == _p].sum()
                    if _e > _oE: _oE = _e; _oP = int(_p)
                if _oP == (_k + 1):
                    _rows.append((float(_pids[_k]), float(_gs[_k]), float(_Et[_k]), _Edep, _nh, _noise, 0, int(_pids[_k])))
                elif _oP == 0:
                    _rows.append((float(_pids[_k]), float(_gs[_k]), float(_Et[_k]), _Edep, _nh, _noise, 1, 0))
                else:
                    _rows.append((float(_pids[_k]), float(_gs[_k]), float(_Et[_k]), _Edep, _nh, _noise, 2, int(_pids[_oP - 1])))
            with open(os.environ.get("DUMP_NH_FATE_PATH", "/tmp/nh_fate.csv"), "a") as _f:
                for _r in _rows:
                    _f.write(",".join(str(_x) for _x in _r) + "\n")
        shower_p_unique = torch.unique(labels)
        shower_p_unique, row_ind, col_ind, i_m_w, _ = match_showers(
            labels, dic, particle_ids, model_output, local_rank, i, None
        )
        row_ind = torch.Tensor(row_ind).to(model_output.device).long()
        col_ind = torch.Tensor(col_ind).to(model_output.device).long()
        if torch.sum(particle_ids == 0) > 0:
            row_ind_ = row_ind - 1
        else:
            # if there is no zero then index 0 corresponds to particle 1.
            row_ind_ = row_ind
        index_matches = col_ind + 1
        index_matches = index_matches.to(model_output.device).long()
        """
                    ### Plot shapes of some showers, to debug what's wrong with the energies
                    debug_showers = False
                    if debug_showers:
                        energy_true_part = dic["part_true"][:, 3].detach().cpu()
                        from torch_scatter import scatter_sum
                        energy_sum_hits = scatter_sum(dic["graph"].ndata["e_hits"], dic["graph"].ndata["particle_number"].type(torch.int64), dim=0).flatten().detach().cpu()
                        energy_noise = str(round(energy_sum_hits[0].item(), 2))
                        n_hits_noise = torch.sum(dic["graph"].ndata["particle_number"] == 0).detach().cpu().item()
                        #frac_energy_sum = energy_sum_hits / energy_true_part[1:]
                        import matplotlib.pyplot as plt
                        n_particles = len(particle_ids)
                        fig = plt.figure(figsize=(18, 4 * n_particles))
                        for j in range(n_particles):
                            mask = labels == j
                            # make ax projection 3D
                            #ax.scatter(X[mask, 0].detach().cpu(), X[mask, 1].detach().cpu(), c=dic["graph"].ndata["hit_type"][mask].detach().cpu())
                            ax = fig.add_subplot(n_particles, 1, j+1, projection='3d')
                            ax.scatter(X[mask, 0].detach().cpu(), X[mask, 1].detach().cpu(), X[mask, 2].detach().cpu(), c=dic["graph"].ndata["hit_type"][mask].detach().cpu())
                            pnum = (particle_ids[j]-1).type(torch.int64).detach().cpu()
                            part_xyz = dic["part_true"][pnum, [0,1,2]].detach().cpu()
                            ax.scatter(part_xyz[0], part_xyz[1], part_xyz[2], c='r', s=100)
                            ax.set_title(f"gr. {i}, E c.f. = {str(round(energy_true_part[pnum].item() / energy_sum_hits[1:][pnum].item() - 1, 2))}, Etrue = {round(energy_true_part[pnum].item(), 2)}, Esum_hits = {round(energy_sum_hits[1:][pnum].item(), 2)}, Nnoisehits = {n_hits_noise}, Enoise = {energy_noise}, eta={part_eta},phi={part_phi}")
                        # log to wandb
                        wandb.log({"showers": [wandb.Image(fig, caption="showers")]})
        """
        for j, unique_showers_label in enumerate(index_matches):
            if torch.sum(unique_showers_label == index_matches) == 1:
                index_in_matched = torch.argmax(
                    (unique_showers_label == index_matches) * 1
                )
                mask = labels == unique_showers_label
                # non_graph = torch.sum(mask)
                sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
                g = dgl.graph(([], []))
                g.add_nodes(sls_graph.shape[0])
                g =  g.to(sls_graph.device)
                g.ndata["h"] = graphs[i].ndata["h"][mask]
                if "pos_pxpypz" in graphs[i].ndata:
                    g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
                if "pos_pxpypz_at_vertex" in graphs[i].ndata:
                    g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
                        "pos_pxpypz_at_vertex"
                    ][mask]
                g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
                energy_t = dic["part_true"].E.to(model_output.device)
                energy_t_corr_daughters = dic["part_true"].m.to(
                    model_output.device
                )
                true_energy_shower = energy_t[row_ind_[j]]
                y_pids_matched.append(y.pid[row_ind_[j]].item())
                y_coords_matched.append(y.coord[row_ind_[j]].detach().cpu().numpy())
                energy_true_daughters.append(energy_t_corr_daughters[row_ind_[j]])
                reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                graphs_showers_matched.append(g)
                true_energy_showers.append(true_energy_shower.view(-1))
                reco_energy_showers.append(reco_energy_shower.view(-1))
        pred_showers = shower_p_unique
        pred_showers[index_matches] = -1
        pred_showers[
            0
        ] = (
            -1
        )
        mask_fakes = pred_showers != -1
        fakes_idx = torch.where(mask_fakes)[0]
        if add_fakes:
            for j in fakes_idx:
                mask = labels == j
                sls_graph = graphs[i].ndata["pos_hits_xyz"][mask][:, 0:3]
                g = dgl.graph(([], []))
                g.add_nodes(sls_graph.shape[0])
                g =  g.to(sls_graph.device)
                
                #g = dgl.remove_self_loop(g)
                g.ndata["h"] = graphs[i].ndata["h"][mask]
                   
                if "pos_pxpypz" in graphs[i].ndata:
                    g.ndata["pos_pxpypz"] = graphs[i].ndata["pos_pxpypz"][mask]
                if "pos_pxpypz_at_vertex" in graphs[i].ndata:
                    g.ndata["pos_pxpypz_at_vertex"] = graphs[i].ndata[
                        "pos_pxpypz_at_vertex"
                    ][mask]
                g.ndata["chi_squared_tracks"] = graphs[i].ndata["chi_squared_tracks"][mask]
                graphs_showers_fakes.append(g)
                reco_energy_shower = torch.sum(graphs[i].ndata["e_hits"][mask])
                reco_energy_showers_fakes.append(reco_energy_shower.view(-1))
    batch_g.ndata["pred_cluster_labels"] = torch.cat(all_cluster_labels)
    graphs_showers_matched = dgl.batch(graphs_showers_matched + graphs_showers_fakes)
    true_energy_showers = torch.cat(true_energy_showers, dim=0)
    reco_energy_showers = torch.cat(reco_energy_showers + reco_energy_showers_fakes, dim=0)
    e_true_corr_daughters = torch.cat(energy_true_daughters, dim=0)
    number_of_fakes = len(reco_energy_showers_fakes)
    return (
        graphs_showers_matched,
        true_energy_showers,
        reco_energy_showers,
        y_pids_matched,
        e_true_corr_daughters,
        y_coords_matched,
        number_of_fakes,
        fakes_idx
    )
