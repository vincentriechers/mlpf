"""DataFrame construction and shower-level helpers for particle-flow reconstruction."""
import torch
import pandas as pd
from torch_scatter import scatter_add, scatter_mean, scatter_max

from src.layers.clustering import remove_labels_of_double_showers
from src.layers.shower_matching import obtain_intersection_values


# ---------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------

def nan_like(t):
    return torch.zeros_like(t) * torch.nan


def nan_tensor(*size, device):
    return torch.zeros(*size, device=device) * torch.nan


def _window(tensor, start, count):
    return tensor[start : start + count]


def _compute_pandora_momentum(labels, g):
    """Scatter-mean the pandora momentum/reference-point node features per cluster.

    Returns (pxyz, ref_pt, pandora_pid, calc_pandora_momentum).  All three
    tensor outputs are None when the graph does not carry 'pandora_momentum'.
    """
    calc_pandora_momentum = "pandora_momentum" in g.ndata
    if not calc_pandora_momentum:
        return None, None, None, False
    px = scatter_mean(g.ndata["pandora_momentum"][:, 0], labels)
    py = scatter_mean(g.ndata["pandora_momentum"][:, 1], labels)
    pz = scatter_mean(g.ndata["pandora_momentum"][:, 2], labels)
    ref_pt_px = scatter_mean(g.ndata["pandora_reference_point"][:, 0], labels)
    ref_pt_py = scatter_mean(g.ndata["pandora_reference_point"][:, 1], labels)
    ref_pt_pz = scatter_mean(g.ndata["pandora_reference_point"][:, 2], labels)
    pandora_pid = scatter_mean(g.ndata["pandora_pid"], labels)
    ref_pt = torch.stack((ref_pt_px, ref_pt_py, ref_pt_pz), dim=1)
    pxyz = torch.stack((px, py, pz), dim=1)
    return pxyz, ref_pt, pandora_pid, True


# ---------------------------------------------------------------------------
# Per-shower correction
# ---------------------------------------------------------------------------

def get_correction_per_shower(labels, dic):
    unique_labels = torch.unique(labels)
    list_corr = []
    for ii, pred_label in enumerate(unique_labels):
        if ii == 0:
            if pred_label != 0:
                list_corr.append(dic["graph"].ndata["correction"][0].view(-1) * 0)
        mask = labels == pred_label
        corrections_E_label = dic["graph"].ndata["correction"][mask]
        betas_label_indmax = torch.argmax(dic["graph"].ndata["beta"][mask])
        list_corr.append(corrections_E_label[betas_label_indmax].view(-1))
    corrections = torch.cat(list_corr, dim=0)
    return corrections


# ---------------------------------------------------------------------------
# Track–cluster distance helpers
# ---------------------------------------------------------------------------

def distance_to_true_cluster_of_track(dic, labels):
    g = dic["graph"]
    mask_hit_type_t2 = g.ndata["hit_type"] == 1
    if torch.sum(labels.unique() == 0) == 0:
        distances = torch.zeros(len(labels.unique()) + 1).float().to(labels.device)
        number_of_tracks = torch.zeros(len(labels.unique()) + 1).int()
    else:
        distances = torch.zeros(len(labels.unique())).float().to(labels.device)
        number_of_tracks = torch.zeros(len(labels.unique())).int()
    for i, label in enumerate(labels.unique()):
        mask_labels_i = labels == label
        mask = mask_labels_i * mask_hit_type_t2
        if mask.sum() == 0:
            continue
        pos_track = g.ndata["pos_hits_xyz"][mask][0]
        if pos_track.shape[0] == 0:
            continue
        true_part_idx_track = g.ndata["particle_number"][mask_labels_i * mask_hit_type_t2][0].int()
        mask_labels_i_true = g.ndata["particle_number"] == true_part_idx_track
        mean_pos_cluster_true = torch.mean(
            g.ndata["pos_hits_xyz"][mask_labels_i_true], dim=0
        )
        number_of_tracks[label] = torch.sum(mask_labels_i_true * mask_hit_type_t2)
        distances[label] = torch.norm(mean_pos_cluster_true - pos_track) / 3300
    return distances, number_of_tracks


def distance_to_cluster_track(dic, is_track_in_MC):
    g = dic["graph"]
    mask_hit_type_t1 = g.ndata["hit_type"] == 2
    mask_hit_type_t2 = g.ndata["hit_type"] == 1
    pos_track = g.ndata["pos_hits_xyz"][mask_hit_type_t2]
    particle_track = g.ndata["particle_number"][mask_hit_type_t2]
    if len(particle_track) > 0:
        mean_pos_cluster_all = []
        for i in particle_track:
            if i == 0:
                mean_pos_cluster_all.append(torch.zeros((1, 3)).view(-1, 3).to(particle_track.device))
            else:
                mask_labels_i = g.ndata["particle_number"] == i
                mean_pos_cluster = torch.mean(g.ndata["pos_hits_xyz"][mask_labels_i * mask_hit_type_t1], dim=0)
                mean_pos_cluster_all.append(mean_pos_cluster.view(-1, 3))
        mean_pos_cluster_all = torch.cat(mean_pos_cluster_all, dim=0)
        distance_track_cluster = torch.norm(mean_pos_cluster_all - pos_track, dim=1) / 1000
        if len(particle_track) > len(torch.unique(particle_track)):
            distance_track_cluster_unique = []
            for i in torch.unique(particle_track):
                mask_tracks = particle_track == i
                distance_track_cluster_unique.append(torch.min(distance_track_cluster[mask_tracks]).view(-1))
            distance_track_cluster_unique = torch.cat(distance_track_cluster_unique, dim=0)
            unique_particle_track = torch.unique(particle_track)
        else:
            distance_track_cluster_unique = distance_track_cluster
            unique_particle_track = particle_track
        distance_to_cluster_all = is_track_in_MC.clone().float()
        distance_to_cluster_all[unique_particle_track.long()] = distance_track_cluster_unique
        return distance_to_cluster_all
    else:
        return is_track_in_MC.clone().float()


# ---------------------------------------------------------------------------
# Main DataFrame builder
# ---------------------------------------------------------------------------

def generate_showers_data_frame(
    labels,
    dic,
    shower_p_unique,
    particle_ids,
    row_ind,
    col_ind,
    i_m_w,
    pandora=False,
    e_corr=None,
    number_of_showers_total=None,
    step=0,
    number_in_batch=0,
    ec_x=None,
    pred_pos=None,
    pred_pid=None,
    pred_ref_pt=None,
    number_of_fake_showers_total=None,
    number_of_fakes=None,
    extra_features=None,
    labels_clusters_removed_tracks=None,
):
    e_pred_showers = scatter_add(dic["graph"].ndata["e_hits"].view(-1), labels)
    e_pred_showers_ecal = scatter_add(1 * (dic["graph"].ndata["hit_type"].view(-1) == 2), labels)
    e_pred_showers_hcal = scatter_add(1 * (dic["graph"].ndata["hit_type"].view(-1) == 3), labels)
    if not pandora:
        removed_tracks = scatter_add(1 * labels_clusters_removed_tracks, labels)
    if pandora:
        e_pred_showers_cali = scatter_mean(
            dic["graph"].ndata["pandora_pfo_energy"].view(-1), labels
        )
        e_pred_showers_pfo = scatter_mean(
            dic["graph"].ndata["pandora_pfo_energy"].view(-1), labels
        )
        pxyz_pred_pfo, ref_pt_pred_pfo, pandora_pid, calc_pandora_momentum = \
            _compute_pandora_momentum(labels, dic["graph"])
    else:
        if e_corr is None:
            corrections_per_shower = get_correction_per_shower(labels, dic)
            e_pred_showers_cali = e_pred_showers * corrections_per_shower
        else:
            corrections_per_shower = e_corr.view(-1)
            if number_of_fakes > 0:
                corrections_per_shower_fakes = corrections_per_shower[-number_of_fakes:]
                corrections_per_shower = corrections_per_shower[:-number_of_fakes]

    e_reco_showers = scatter_add(
        dic["graph"].ndata["e_hits"].view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )
    e_label_showers = scatter_max(
        labels.view(-1),
        dic["graph"].ndata["particle_number"].long(),
    )[0]
    is_track_in_MC = scatter_add(
        1 * (dic["graph"].ndata["hit_type"].view(-1) == 1),
        dic["graph"].ndata["particle_number"].long(),
    )
    track_chi = scatter_add(
        1 * (dic["graph"].ndata["chi_squared_tracks"].view(-1) == 1),
        dic["graph"].ndata["particle_number"].long(),
    )
    distance_to_cluster_all = distance_to_cluster_track(dic, is_track_in_MC)
    distances, number_of_tracks = distance_to_true_cluster_of_track(dic, labels)

    row_ind = torch.Tensor(row_ind).to(e_pred_showers.device).long()
    col_ind = torch.Tensor(col_ind).to(e_pred_showers.device).long()

    if torch.sum(particle_ids == 0) > 0:
        row_ind_ = row_ind - 1
    else:
        row_ind_ = row_ind

    pred_showers = shower_p_unique
    energy_t = (
        dic["part_true"].E_corrected.view(-1).to(e_pred_showers.device)
    ).float()
    gen_status = (
        dic["part_true"].gen_status.view(-1).to(e_pred_showers.device)
    ).float()
    vertex = dic["part_true"].vertex.to(e_pred_showers.device)
    pos_t = dic["part_true"].coord.to(e_pred_showers.device)
    pid_t = dic["part_true"].pid.to(e_pred_showers.device)
    if not pandora:
        labels = remove_labels_of_double_showers(labels, dic["graph"])
    is_track_per_shower = scatter_add(1 * (dic["graph"].ndata["hit_type"] == 1), labels).int()
    is_track = torch.zeros(energy_t.shape).to(e_pred_showers.device)

    index_matches = col_ind + 1
    index_matches = index_matches.to(e_pred_showers.device).long()

    dev = e_pred_showers.device
    matched_es = nan_like(energy_t)
    matched_ECAL = nan_like(energy_t)
    matched_HCAL = nan_like(energy_t)
    matched_positions = nan_tensor(energy_t.shape[0], 3, device=dev)
    matched_ref_pt = nan_tensor(energy_t.shape[0], 3, device=dev)
    matched_pid = nan_like(energy_t).long()
    matched_positions_pfo = nan_tensor(energy_t.shape[0], 3, device=dev)
    matched_pandora_pid = nan_tensor(energy_t.shape[0], device=dev)
    matched_ref_pts_pfo = nan_tensor(energy_t.shape[0], 3, device=dev)
    matched_extra_features = torch.zeros((energy_t.shape[0], 7)) * torch.nan

    matched_es[row_ind_] = e_pred_showers[index_matches]
    matched_ECAL[row_ind_] = 1.0 * e_pred_showers_ecal[index_matches]
    matched_HCAL[row_ind_] = 1.0 * e_pred_showers_hcal[index_matches]

    if pandora:
        matched_es_cali = matched_es.clone()
        matched_es_cali[row_ind_] = e_pred_showers_cali[index_matches]
        matched_es_cali_pfo = matched_es.clone()
        matched_es_cali_pfo[row_ind_] = e_pred_showers_pfo[index_matches]
        matched_pandora_pid[row_ind_] = pandora_pid[index_matches]
        if calc_pandora_momentum:
            matched_positions_pfo[row_ind_] = pxyz_pred_pfo[index_matches]
            matched_ref_pts_pfo[row_ind_] = ref_pt_pred_pfo[index_matches]
        is_track[row_ind_] = is_track_per_shower[index_matches].float()
    else:
        if e_corr is None:
            matched_es_cali = matched_es.clone()
            matched_es_cali[row_ind_] = e_pred_showers_cali[index_matches]
            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind_] = corrections_per_shower[index_matches]
            cluster_removed_tracks = matched_es.clone()
        else:
            matched_es_cali = matched_es.clone()
            number_of_showers = e_pred_showers[index_matches].shape[0]
            matched_es_cali[row_ind_] = _window(
                corrections_per_shower, number_of_showers_total, number_of_showers
            )
            cluster_removed_tracks = matched_es.clone()
            cluster_removed_tracks[row_ind_] = 1.0 * removed_tracks[index_matches]

            if pred_pos is not None:
                matched_positions[row_ind_] = _window(pred_pos, number_of_showers_total, number_of_showers)
                matched_ref_pt[row_ind_] = _window(pred_ref_pt, number_of_showers_total, number_of_showers)
                matched_pid[row_ind_] = _window(pred_pid, number_of_showers_total, number_of_showers)
                if not pandora:
                    matched_extra_features[row_ind_] = torch.tensor(
                        _window(extra_features, number_of_showers_total, number_of_showers)
                    )

            calibration_per_shower = matched_es.clone()
            calibration_per_shower[row_ind_] = _window(
                corrections_per_shower, number_of_showers_total, number_of_showers
            )
            number_of_showers_total = number_of_showers_total + number_of_showers
        is_track[row_ind_] = is_track_per_shower[index_matches].float()

    # match the tracks to the particle
    dic["graph"].ndata["particle_number_u"] = dic["graph"].ndata["particle_number"].clone()
    dic["graph"].ndata["particle_number_u"][dic["graph"].ndata["particle_number_u"] == 0] = 100
    tracks_label = scatter_max(
        (dic["graph"].ndata["hit_type"] == 1) * (dic["graph"].ndata["particle_number_u"]), labels
    )[0].int()
    tracks_label = tracks_label - 1
    tracks_label[tracks_label < 0] = 0
    matched_es_tracks = nan_like(energy_t)
    matched_es_tracks_1 = nan_like(energy_t)
    matched_es_tracks[row_ind_] = row_ind_.float()
    matched_es_tracks_1[row_ind_] = tracks_label[index_matches].float()
    matched_es_tracks_1 = 1.0 * (matched_es_tracks == matched_es_tracks_1)
    matched_es_tracks_1 = matched_es_tracks_1 * is_track

    intersection_E = nan_like(energy_t)
    if len(col_ind) > 0:
        ie_e = obtain_intersection_values(i_m_w, row_ind, col_ind, dic)
        intersection_E[row_ind_] = ie_e.to(e_pred_showers.device)
        pred_showers[index_matches] = -1
        pred_showers[0] = -1
        mask = pred_showers != -1
        fakes_in_event = mask.sum()
        fake_showers_e = e_pred_showers[mask]
        fake_showers_e_hcal = e_pred_showers_hcal[mask]
        fake_showers_e_ecal = e_pred_showers_ecal[mask]
        number_of_fake_showers = mask.sum()

        all_labels = labels.unique().to(e_pred_showers.device)
        number_of_fake_showers = mask.sum()
        fakes_labels = torch.where(mask)[0].to(e_pred_showers.device)
        fake_showers_distance_to_cluster = distances[fakes_labels.cpu()]
        fake_showers_num_tracks = number_of_tracks[fakes_labels.cpu()]

        if e_corr is None or pandora:
            fake_showers_e_cali = e_pred_showers_cali[mask]
        else:
            fakes_positions = pred_pos[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total + number_of_fake_showers]
            fake_showers_e_cali = e_corr[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total + number_of_fake_showers]
            fakes_pid_pred = pred_pid[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total + number_of_fake_showers]
            fake_showers_e_reco = e_reco_showers[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total + number_of_fake_showers]
            fakes_positions = fakes_positions.to(e_pred_showers.device)
            fakes_extra_features = extra_features[-number_of_fakes:][number_of_fake_showers_total:number_of_fake_showers_total + number_of_fake_showers]
            fake_showers_e_cali = fake_showers_e_cali.to(e_pred_showers.device)
            fakes_pid_pred = fakes_pid_pred.to(e_pred_showers.device)
            fake_showers_e_reco = fake_showers_e_reco.to(e_pred_showers.device)

        if pandora:
            fake_pandora_pid = (torch.zeros((fake_showers_e.shape[0], 3)) * torch.nan).to(dev)
            fake_pandora_pid = pandora_pid[mask]
            if calc_pandora_momentum:
                fake_positions_pfo = nan_tensor(fake_showers_e.shape[0], 3, device=dev)
                fake_positions_pfo = pxyz_pred_pfo[mask]
                fakes_positions_ref = nan_tensor(fake_showers_e.shape[0], 3, device=dev)
                fakes_positions_ref = ref_pt_pred_pfo[mask]
        if not pandora:
            if e_corr is None:
                fake_showers_e_cali_factor = corrections_per_shower[mask]
            else:
                fake_showers_e_cali_factor = fake_showers_e_cali
        fake_showers_showers_e_truw = nan_tensor(fake_showers_e.shape[0], device=dev)
        fake_showers_vertex = nan_tensor(fake_showers_e.shape[0], 3, device=dev)
        fakes_is_track = (torch.zeros((fake_showers_e.shape[0])) * torch.nan).to(dev)
        fakes_is_track = is_track_per_shower[mask]
        fakes_positions_t = nan_tensor(fake_showers_e.shape[0], 3, device=dev)
        if not pandora:
            number_of_fake_showers_total = number_of_fake_showers_total + number_of_fake_showers

        energy_t = torch.cat((energy_t, fake_showers_showers_e_truw), dim=0)
        gen_status = torch.cat((gen_status, fake_showers_showers_e_truw), dim=0)
        vertex = torch.cat((vertex, fake_showers_vertex), dim=0)
        pid_t = torch.cat((pid_t.view(-1), fake_showers_showers_e_truw), dim=0)
        pos_t = torch.cat((pos_t, fakes_positions_t), dim=0)
        e_reco = torch.cat((e_reco_showers[1:], fake_showers_showers_e_truw), dim=0)
        e_labels = torch.cat((e_label_showers[1:], 0 * fake_showers_showers_e_truw), dim=0)
        is_track_in_MC = torch.cat((is_track_in_MC[1:], fake_showers_num_tracks.to(e_reco.device)), dim=0)
        track_chi = torch.cat((track_chi[1:], fake_showers_num_tracks.to(e_reco.device)), dim=0)
        distance_to_cluster_MC = torch.cat(
            (distance_to_cluster_all[1:], fake_showers_distance_to_cluster.to(e_reco.device)), dim=0
        )
        e_pred = torch.cat((matched_es, fake_showers_e), dim=0)
        e_pred_ECAL = torch.cat((matched_ECAL, fake_showers_e_ecal), dim=0)
        e_pred_HCAL = torch.cat((matched_HCAL, fake_showers_e_hcal), dim=0)
        e_pred_cali = torch.cat((matched_es_cali, fake_showers_e_cali), dim=0)
        if pred_pos is not None:
            e_pred_pos = torch.cat((matched_positions, fakes_positions), dim=0)
            e_pred_pid = torch.cat((matched_pid, fakes_pid_pred), dim=0)
            e_pred_ref_pt = torch.cat((matched_ref_pt, fakes_positions), dim=0)
            extra_features_all = torch.cat(
                (matched_extra_features, torch.tensor(fakes_extra_features)), dim=0
            )
        if pandora:
            e_pred_cali_pfo = torch.cat((matched_es_cali_pfo, fake_showers_e_cali), dim=0)
            positions_pfo = torch.cat((matched_positions_pfo, fake_positions_pfo), dim=0)
            pandora_pid = torch.cat((matched_pandora_pid, fake_pandora_pid), dim=0)
            ref_pts_pfo = torch.cat((matched_ref_pts_pfo, fakes_positions_ref), dim=0)
        else:
            cluster_removed_tracks = torch.cat((cluster_removed_tracks, 0 * fake_showers_e_cali), dim=0)
        if not pandora:
            calibration_factor = torch.cat((calibration_per_shower, fake_showers_e_cali_factor), dim=0)

        e_pred_t = torch.cat(
            (intersection_E, nan_like(fake_showers_e)),
            dim=0,
        )
        is_track = torch.cat((is_track, fakes_is_track.to(is_track.device)), dim=0)
        matched_es_tracks_1 = torch.cat(
            (matched_es_tracks_1, 0 * fakes_is_track.to(is_track.device)), dim=0
        )

        # Build shared base dict, then update with pandora- or non-pandora-specific keys
        d = {
            "true_showers_E": energy_t.detach().cpu(),
            "reco_showers_E": e_reco.detach().cpu(),
            "pred_showers_E": e_pred.detach().cpu(),
            "e_pred_and_truth": e_pred_t.detach().cpu(),
            "pid": pid_t.detach().cpu(),
            "step": torch.ones_like(energy_t.detach().cpu()) * step,
            "number_batch": torch.ones_like(energy_t.detach().cpu()) * number_in_batch,
            "is_track_in_cluster": is_track.detach().cpu(),
            "is_track_correct": matched_es_tracks_1.detach().cpu(),
            "is_track_in_MC": is_track_in_MC.detach().cpu(),
            "track_chi": track_chi.detach().cpu(),
            "distance_to_cluster_MC": distance_to_cluster_MC.detach().cpu(),
            "vertex": vertex.detach().cpu().tolist(),
            "ECAL_hits": e_pred_ECAL.detach().cpu(),
            "HCAL_hits": e_pred_HCAL.detach().cpu(),
            "gen_status": gen_status.detach().cpu(),
            "labels": e_labels.detach().cpu(),
        }
        if pandora:
            d.update({
                "pandora_calibrated_E": e_pred_cali.detach().cpu(),
                "pandora_calibrated_pfo": e_pred_cali_pfo.detach().cpu(),
                "pandora_calibrated_pos": positions_pfo.detach().cpu().tolist(),
                "pandora_ref_pt": ref_pts_pfo.detach().cpu().tolist(),
                "pandora_pid": pandora_pid.detach().cpu(),
            })
        else:
            d.update({
                "calibration_factor": calibration_factor.detach().cpu(),
                "calibrated_E": e_pred_cali.detach().cpu(),
                "cluster_removed_tracks": cluster_removed_tracks.detach().cpu(),
            })
            if pred_pos is not None:
                d["pred_pos_matched"] = e_pred_pos.detach().cpu().tolist()
                d["pred_pid_matched"] = e_pred_pid.detach().cpu().tolist()
                d["pred_ref_pt_matched"] = e_pred_ref_pt.detach().cpu().tolist()
                d["matched_extra_features"] = extra_features_all.detach().cpu().tolist()

        d["true_pos"] = pos_t.detach().cpu().tolist()
        df = pd.DataFrame(data=d)
        if number_of_showers_total is None:
            return df
        else:
            return df, number_of_showers_total, number_of_fake_showers_total
    else:
        return [], 0, 0
