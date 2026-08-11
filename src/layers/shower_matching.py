"""Shower matching utilities for particle-flow reconstruction."""
import torch
import numpy as np
from torch_scatter import scatter_add
from scipy.optimize import linear_sum_assignment


class CachedIndexList:
    def __init__(self, lst):
        self.lst = lst
        self.cache = {}

    def index(self, value):
        if value in self.cache:
            return self.cache[value]
        else:
            idx = self.lst.index(value)
            self.cache[value] = idx
            return idx


def get_labels_pandora(dic, device):
    labels_pandora = dic["graph"].ndata["pandora_pfo"].long()
    labels_pandora = labels_pandora + 1
    map_from = list(np.unique(labels_pandora.detach().cpu()))
    map_from = CachedIndexList(map_from)
    cluster_id = map(lambda x: map_from.index(x), labels_pandora.detach().cpu().numpy())
    labels_pandora = torch.Tensor(list(cluster_id)).long().to(device)
    return labels_pandora


def obtain_intersection_matrix(shower_p_unique, particle_ids, labels, dic, e_hits):
    len_pred_showers = len(shower_p_unique)
    intersection_matrix = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    intersection_matrix_w = torch.zeros((len_pred_showers, len(particle_ids))).to(
        shower_p_unique.device
    )
    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        h_hits = e_hits.clone()
        counts[mask_p] = 1
        h_hits[~mask_p] = 0
        intersection_matrix[:, index] = scatter_add(counts, labels)
        intersection_matrix_w[:, index] = scatter_add(h_hits, labels.to(h_hits.device))
    return intersection_matrix, intersection_matrix_w


def obtain_union_matrix(shower_p_unique, particle_ids, labels, dic):
    len_pred_showers = len(shower_p_unique)
    union_matrix = torch.zeros((len_pred_showers, len(particle_ids)))
    for index, id in enumerate(particle_ids):
        counts = torch.zeros_like(labels)
        mask_p = dic["graph"].ndata["particle_number"] == id
        for index_pred, id_pred in enumerate(shower_p_unique):
            mask_pred_p = labels == id_pred
            mask_union = mask_pred_p + mask_p
            union_matrix[index_pred, index] = torch.sum(mask_union)
    return union_matrix


def obtain_intersection_values(intersection_matrix_w, row_ind, col_ind, dic):
    list_intersection_E = []
    particle_ids = torch.unique(dic["graph"].ndata["particle_number"])
    if torch.sum(particle_ids == 0) > 0:
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, 1:], 1, 0)
        row_ind = row_ind - 1
    else:
        intersection_matrix_wt = torch.transpose(intersection_matrix_w[1:, :], 1, 0)
    for i in range(0, len(col_ind)):
        list_intersection_E.append(
            intersection_matrix_wt[row_ind[i], col_ind[i]].view(-1)
        )
    if len(list_intersection_E) > 0:
        return torch.cat(list_intersection_E, dim=0)
    else:
        return 0


def match_showers(
    labels,
    dic,
    particle_ids,
    model_output,
    local_rank,
    i,
    path_save,
    pandora=False,
    hdbscan=False,
):
    iou_threshold = 0.25
    shower_p_unique = torch.unique(labels)
    if torch.sum(labels == 0) == 0:
        shower_p_unique = torch.cat(
            (
                torch.Tensor([0]).to(shower_p_unique.device).view(-1),
                shower_p_unique.view(-1),
            ),
            dim=0,
        )
    e_hits = dic["graph"].ndata["e_hits"].view(-1)
    i_m, i_m_w = obtain_intersection_matrix(
        shower_p_unique, particle_ids, labels, dic, e_hits
    )
    i_m = i_m.to(model_output.device)
    i_m_w = i_m_w.to(model_output.device)
    u_m = obtain_union_matrix(shower_p_unique, particle_ids, labels, dic)
    u_m = u_m.to(model_output.device)
    iou_matrix = i_m / u_m
    if torch.sum(particle_ids == 0) > 0:
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, 1:], 1, 0).clone().detach().cpu().numpy()
        )
    else:
        iou_matrix_num = (
            torch.transpose(iou_matrix[1:, :], 1, 0).clone().detach().cpu().numpy()
        )
    iou_matrix_num[iou_matrix_num < iou_threshold] = 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix_num)
    mask_matching_matrix = iou_matrix_num[row_ind, col_ind] > 0
    row_ind = row_ind[mask_matching_matrix]
    col_ind = col_ind[mask_matching_matrix]
    if torch.sum(particle_ids == 0) > 0:
        row_ind = row_ind + 1
    return shower_p_unique, row_ind, col_ind, i_m_w, iou_matrix
