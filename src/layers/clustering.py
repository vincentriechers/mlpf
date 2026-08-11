"""Clustering algorithms for particle-flow reconstruction.

Adapted from densitypeakclustering (https://github.com/lanbing510/DensityPeakCluster).
"""
import torch
import numpy as np
from torch_scatter import scatter_add
import densitypeakclustering as dc


def local_density_energy(D, d_c, energies, normalize=False):
    D_cuttoff = D < d_c
    rho = np.zeros((D.shape[0],))
    for s in range(len(rho)):
        rho[s] = np.sum(energies[D_cuttoff[s, :]] * np.exp(-(D[s, D_cuttoff[s, :]] / d_c) ** 2))
    if normalize:
        rho = rho / np.max(rho)
    return rho


def DPC_custom_CLD(X, g, device):
    d_c = 0.1
    rho_min = 0.05
    delta_min = 0.4
    D = dc.distance_matrix(X.detach().cpu())
    rho = local_density_energy(D, d_c, g.ndata["e_hits"].view(-1).cpu().numpy())
    delta, nearest = dc.distance_to_larger_density(D, rho)
    centers = dc.cluster_centers(rho, delta, rho_min=rho_min, delta_min=delta_min)
    ids = dc.assign_cluster_id(rho, nearest, centers)
    core_ids = np.full(len(X), -1)
    D[np.isnan(D)] = 0
    for indx, c in enumerate(centers):
        idx = np.where((ids == indx) & (D[:, c] < 0.5))[0]
        core_ids[idx] = indx
    labels = torch.Tensor(core_ids) + 1
    return labels.long().to(device)


def remove_bad_tracks_from_cluster(g, labels_hdb):
    mask_hit_type_t1 = g.ndata["hit_type"] == 2
    mask_hit_type_t2 = g.ndata["hit_type"] == 1
    mask_hit_type_t4 = g.ndata["hit_type"] == 4
    labels_hdb_corrected_tracks = labels_hdb.clone()
    labels_changed_tracks = 0.0 * (labels_hdb.clone())
    for i in range(0, torch.max(labels_hdb) + 1):
        mask_labels_i = labels_hdb == i
        if torch.sum(mask_hit_type_t2[mask_labels_i]) > 0 and i > 0:
            e_cluster = torch.sum(g.ndata["e_hits"][mask_labels_i])
            p_track = g.ndata["p_hits"][mask_labels_i * mask_hit_type_t2]
            number_of_hits_muon = torch.sum(mask_labels_i * mask_hit_type_t4)
            diffs = torch.abs(e_cluster - p_track) / p_track
            diffs = diffs.view(-1)
            sigma_4 = 4 * 0.5 / torch.sqrt(p_track).view(-1)
            bad_diffs = diffs > sigma_4
            bad_tracks = bad_diffs * (number_of_hits_muon < 1)
            cluster_t2_nodes = torch.nonzero(mask_labels_i & mask_hit_type_t2).view(-1)
            bad_tracks_nodes = cluster_t2_nodes[bad_tracks]
            labels_hdb_corrected_tracks[bad_tracks_nodes] = 0
            if torch.sum(bad_tracks_nodes) > 0:
                labels_changed_tracks[mask_labels_i] = 1
    return labels_hdb_corrected_tracks, labels_changed_tracks


def remove_labels_of_double_showers(labels, g):
    is_track_per_shower = scatter_add(1 * (g.ndata["hit_type"] == 1), labels).int()
    e_hits_sum = scatter_add(g.ndata["e_hits"].view(-1), labels.view(-1).long()).int()
    mask_tracks = g.ndata["hit_type"] == 1
    for i, label_i in enumerate(torch.unique(labels)):
        if is_track_per_shower[label_i] == 2:
            if label_i > 0:
                sum_pred_2 = e_hits_sum[label_i]
                mask_labels_i = labels == label_i
                mask_label_i_and_is_track = mask_labels_i * mask_tracks
                tracks_E = g.ndata['h'][:, -1][mask_label_i_and_is_track]
                chi_tracks = g.ndata['chi_squared_tracks'][mask_label_i_and_is_track]
                ind_min_E = torch.argmax(torch.abs(tracks_E - sum_pred_2))
                ind_min_chi = torch.argmax(chi_tracks)
                mask_hit_type_t1 = g.ndata["hit_type"][mask_labels_i] == 2
                mask_hit_type_t2 = g.ndata["hit_type"][mask_labels_i] == 1
                mask_all = mask_hit_type_t1
                index_sorted = torch.argsort(g.ndata["radial_distance"][mask_labels_i][mask_hit_type_t1])
                mask_sorted_ind = index_sorted < 10
                mean_pos_cluster = torch.mean(
                    g.ndata["pos_hits_xyz"][mask_labels_i][mask_all][mask_sorted_ind], dim=0
                )
                pos_track = g.ndata["pos_hits_xyz"][mask_labels_i][mask_hit_type_t2]
                distance_track_cluster = torch.norm(pos_track - mean_pos_cluster, dim=1) / 1000
                ind_max_dtc = torch.argmax(distance_track_cluster)
                if torch.min(distance_track_cluster) < 0.4:
                    ind_min = ind_max_dtc
                elif ind_min_E == ind_min_chi:
                    ind_min = ind_min_E
                elif torch.max(chi_tracks - torch.min(chi_tracks)) < 2:
                    ind_min = ind_min_E
                else:
                    ind_min = ind_min_chi
                ind_change = torch.argwhere(mask_label_i_and_is_track)[ind_min]
                labels[ind_change] = 0
    return labels
