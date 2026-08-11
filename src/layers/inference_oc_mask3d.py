"""Inference / evaluation utilities for the Mask3D model.

The original `src/layers/inference_oc.py::create_and_store_graph_output`
expects an Object-Condensation-style `model_output = [coord(3), beta(1)]`
tensor and runs DPC/DBSCAN on it to recover clusters. Mask3D already
produces per-(query, hit) mask logits — there is nothing to cluster, so
this module:

  * removes the clustering steps (no DPC, no DBSCAN, no `coords`/`beta`),
  * converts (mask_logits, cls_logits) into per-hit cluster labels, then
  * reuses the same `match_showers` + `generate_showers_data_frame`
    machinery from `inference_oc.py` to produce the evaluation data
    frames downstream code expects.
"""
import os

import torch
import dgl
import pandas as pd

from src.layers.inference_oc import (
    match_showers,
    generate_showers_data_frame,
    store_at_batch_end,
    get_labels_pandora,
    _fix_labels_for_classes,
    remove_bad_tracks_from_cluster_v1,
)


def labels_from_masks(
    mask_logits,
    cls_logits,
    mask_threshold=0.5,
    cls_threshold=0.5,
    hit_type=None,
    track_mask_threshold=None,
    track_hit_type=1,
    force_track_assignment=False,
):
    """Convert per-(query, hit) mask logits into per-hit cluster labels.

    Args:
        mask_logits: (N_q, N_k) float — per-query mask logits for one event.
        cls_logits:  (N_q,) or (N_q, 1) float — per-query validity logits.
        mask_threshold: minimum sigmoid(mask) for a hit to be considered owned.
        cls_threshold:  minimum sigmoid(cls) for a query to be considered real.
        hit_type: optional (N_k,) int tensor of per-hit detector ids. When
            paired with `track_mask_threshold`, lets tracker hits (hit_type==1)
            use a different (typically lower) mask threshold than calo hits.
        track_mask_threshold: minimum sigmoid(mask) for tracker hits. If None,
            tracker hits use `mask_threshold` like everything else.
        track_hit_type: integer hit_type id for tracker hits (default 1).
        force_track_assignment: if True, tracker hits are assigned to their
            argmax-query regardless of mask probability (only gated by query
            validity). Each tracker hit IS a particle, so we never want to
            drop one to "unassigned" just because the model's confidence on
            its best match happened to fall below the calo threshold.

    Returns:
        labels: (N_k,) int64. 0 = noise / unassigned, 1..K = cluster id
                (densely re-indexed so labels.unique() == arange(K+1)).
    """
    if mask_logits.numel() == 0:
        return torch.zeros(0, dtype=torch.int64, device=mask_logits.device)

    probs = mask_logits.sigmoid()                              # (N_q, N_k)
    valid_q = cls_logits.view(-1).sigmoid() >= cls_threshold   # (N_q,)
    import os
    if os.environ.get("M3D_VALID_ARGMAX") and valid_q.any():
        # Mask2Former semantics: only valid (objectness>=thr) queries compete
        # in the per-hit argmax. Without this, unmatched "collector" queries
        # (e.g. trained into existence by a max-over-ALL-queries recall loss)
        # can win the argmax everywhere and, being invalid, drop every hit
        # to noise.
        probs = probs.masked_fill(~valid_q.view(-1, 1), 0.0)
    max_prob, argmax_q = probs.max(dim=0)                      # (N_k,)
    if os.environ.get("M3D_DEBUG_LABELS"):
        cs = cls_logits.view(-1).sigmoid()
        print(
            "[labels_from_masks] N_q=%d valid_q=%d cls_sig[min/med/max]=%.3f/%.3f/%.3f "
            "| N_k=%d maxprob[q10/med/q90]=%.3f/%.3f/%.3f frac>=thr(%.2f)=%.3f"
            % (len(cs), int(valid_q.sum()), cs.min(), cs.median(), cs.max(),
               len(max_prob), max_prob.quantile(0.1), max_prob.median(),
               max_prob.quantile(0.9), float(mask_threshold),
               (max_prob >= mask_threshold).float().mean())
        )

    if hit_type is not None and track_mask_threshold is not None:
        # Per-hit threshold: tracker hits (hit_type==1) get the looser cut.
        per_hit_thr = torch.full_like(max_prob, float(mask_threshold))
        per_hit_thr[hit_type.view(-1) == track_hit_type] = float(track_mask_threshold)
        is_assigned = (max_prob >= per_hit_thr) & valid_q[argmax_q]
    else:
        is_assigned = (max_prob >= mask_threshold) & valid_q[argmax_q]

    if force_track_assignment and hit_type is not None:
        is_track = hit_type.view(-1) == track_hit_type
        is_assigned = is_assigned | (is_track & valid_q[argmax_q])
    raw = torch.where(is_assigned, argmax_q + 1, torch.zeros_like(argmax_q))

    # Compress to dense ids: 0 stays at 0; used queries → 1..K.
    unique, inverse = torch.unique(raw, sorted=True, return_inverse=True)
    if (unique == 0).any():
        labels = inverse                                       # 0 → 0, rest sequential
    else:
        labels = inverse + 1                                   # no noise, all assigned
    return labels.to(torch.int64)


def create_and_store_graph_output_mask3d(
    batch_g,
    mask_logits,             # (B, N_q, N_max) padded mask logits, in feed order
    cls_logits,              # (B, N_q, 1) or (B, N_q) per-query validity
    key_valid,               # (B, N_max) bool, True where the slot is a real hit
    y,                       # Particles_GT
    local_rank,
    step,
    epoch,
    path_save,
    store=False,
    predict=False,
    e_corr=None,
    tracks=False,
    store_epoch=False,
    total_number_events=0,
    pred_pos=None,
    pred_ref_pt=None,
    use_gt_clusters=False,
    fix_clusters_class=None,
    pids_neutral=None,
    pids_charged=None,
    pred_pid=None,
    number_of_fakes=None,
    extra_features=None,
    fakes_labels=None,
    pandora_available=False,
    truth_tracks=False,
    mask_threshold=0.5,
    cls_threshold=0.5,
    track_mask_threshold=0.1,
    force_track_assignment=False,
    track_hit_type=1,
    queries=None,
    gmm_module=None,
):
    """Mask3D-flavoured `create_and_store_graph_output`.

    Differences from the OC version:
      - takes `(mask_logits, cls_logits, key_valid)` instead of a per-hit
        `[coord, beta]` tensor,
      - skips DPC/DBSCAN/coord-based clustering — labels come directly
        from the mask head's argmax, gated by per-query validity,
      - no `bad_tracks` removal (Mask3D doesn't produce those by-product
        clusters; queries either fire or they don't).
    """
    graphs = dgl.unbatch(batch_g)
    batch_id = y.batch_number.view(-1)

    # cls_logits accepts both (B, N_q, 1) and (B, N_q)
    if cls_logits.dim() == 3:
        cls_logits_b = cls_logits.squeeze(-1)
    else:
        cls_logits_b = cls_logits

    # Build a snapshot of the (event-independent) GMM parameters once per
    # batch so we can drop a copy into every saved graph. `mu`, `Sigma`, and
    # `pi` are the minimum needed to plot/reuse the mixtures; `min_log_w`
    # and the raw L parametrisation are kept for reproducibility.
    gmm_snapshot = None
    if gmm_module is not None:
        with torch.no_grad():
            L = gmm_module._build_L(dtype=torch.float32)             # (N_q, K, 3, 3)
            Sigma = L @ L.transpose(-1, -2)                          # (N_q, K, 3, 3)
            pi = gmm_module.pi_logits.detach().float().softmax(-1)   # (N_q, K)
            gmm_snapshot = {
                "mu":          gmm_module.mu.detach().cpu().float(),
                "Sigma":       Sigma.cpu(),
                "pi":          pi.cpu(),
                "L":           L.cpu(),
                "log_diag":    gmm_module.log_diag.detach().cpu().float(),
                "off_diag":    gmm_module.off_diag.detach().cpu().float(),
                "pi_logits":   gmm_module.pi_logits.detach().cpu().float(),
                "min_log_w":   float(gmm_module.min_log_w),
                "sigma_init":  float(gmm_module.sigma_init),
                "num_queries": int(gmm_module.num_queries),
                "num_gaussians": int(gmm_module.num_gaussians),
                "coord_units": "metres (pos_scale=0.001 applied to mm input)",
            }

    df_list1 = []
    df_list_pandora = []
    number_of_showers_total1 = 0
    number_of_fake_showers_total1 = 0

    for i in range(len(graphs)):
        mask_i = batch_id == i
        dic = {"graph": graphs[i]}
        y1 = y.copy()
        y1.mask(mask_i)
        dic["part_true"] = y1

        # Trim padded mask logits down to this event's real hits
        n_i = int(key_valid[i].sum().item())
        ev_mask_logits = mask_logits[i, :, :n_i]                # (N_q, n_i)
        ev_cls_logits = cls_logits_b[i]                         # (N_q,)

        if use_gt_clusters:
            labels = dic["graph"].ndata["particle_number"].type(torch.int64)
        else:
            labels = labels_from_masks(
                ev_mask_logits, ev_cls_logits,
                mask_threshold=mask_threshold,
                cls_threshold=cls_threshold,
                hit_type=dic["graph"].ndata["hit_type"],
                track_mask_threshold=track_mask_threshold,
                track_hit_type=track_hit_type,
                force_track_assignment=force_track_assignment,
            )
            if not truth_tracks:
                # MUST mirror the EC-side cleaning in mask3d_model._post_decoder
                # (labels_from_masks → remove_bad_tracks → re-densify). The EC
                # outputs (e_corr / pred_pos / pred_pid / extra_features) were
                # computed against those cleaned labels; if the dataframe side
                # here doesn't apply the *identical* transform the cluster
                # counts diverge and generate_showers_data_frame builds
                # mismatched-length arrays (pd.DataFrame -> "All arrays must
                # be of the same length"). Same per-event graph + deterministic
                # transform → identical labels on both sides.
                import os as _os
                if _os.environ.get("M3D_DEBUG_LABELS"):
                    _u = torch.unique(labels)
                    print("[m3d-df] labels pre-clean : n_clusters=%d has0=%s n_hits0=%d"
                          % (len(_u), bool((_u == 0).any()), int((labels == 0).sum())))
                labels, _ = remove_bad_tracks_from_cluster_v1(
                    dic["graph"], labels
                )
                if _os.environ.get("M3D_DEBUG_LABELS"):
                    _u = torch.unique(labels)
                    print("[m3d-df] labels post-clean: n_clusters=%d has0=%s n_hits0=%d"
                          % (len(_u), bool((_u == 0).any()), int((labels == 0).sum())))
                _, labels = torch.unique(labels, return_inverse=True)

        # Oracle override for selected classes (kept for parity with OC path)
        if fix_clusters_class:
            labels = _fix_labels_for_classes(
                labels, dic["graph"], dic["part_true"],
                fix_clusters_class, ev_mask_logits.device,
            )
            _, labels = torch.unique(labels, return_inverse=True)

        if predict and pandora_available:
            labels_pandora = get_labels_pandora(tracks, dic, ev_mask_logits.device)

        particle_ids = torch.unique(dic["graph"].ndata["particle_number"])

        # match_showers only uses `model_output` for its `.device`; pass the
        # mask logits to keep that contract.
        shower_p_unique, row_ind, col_ind, i_m_w, _ = match_showers(
            labels, dic, particle_ids, ev_mask_logits,
            local_rank, i, path_save, tracks=tracks, hdbscan=True,
        )
        dic["graph"].ndata["labels"] = labels

        # Stash per-event Mask3D outputs (queries, mask/cls logits, GMM
        # params) AND a plain-tensor copy of the hit data needed to plot
        # the event later. Stored under `dic["mask3d"]` so the saved file
        # is plot-ready without a DGL import. Queries / logits trimmed to
        # this event's real-hit count.
        # if predict:
        #     g_i = dic["graph"]
        #     pt = dic.get("part_true")
        #     mask3d_blob = {
        #         # Mask3D model outputs (this event)
        #         "mask_logits": ev_mask_logits.detach().cpu(),         # (N_q, n_hits)
        #         "cls_logits":  ev_cls_logits.detach().cpu(),          # (N_q,)
        #         "mask_threshold":       float(mask_threshold),
        #         "cls_threshold":        float(cls_threshold),
        #         "track_mask_threshold": float(track_mask_threshold),
        #         "track_hit_type":       int(track_hit_type),
        #         # Hit-level tensors — what you actually need to draw the event
        #         "hits": {
        #             "pos_hits_xyz":    g_i.ndata["pos_hits_xyz"].detach().cpu(),     # (n_hits, 3), in mm
        #             "hit_type":        g_i.ndata["hit_type"].detach().cpu().view(-1),
        #             "particle_number": g_i.ndata["particle_number"].detach().cpu().view(-1),
        #             "e_hits":          g_i.ndata["e_hits"].detach().cpu().view(-1),
        #             "labels_pred":     labels.detach().cpu().view(-1),               # mask3d cluster id per hit
        #             "pos_units":       "mm",
        #         },
        #     }
        #     # GT particles (small; useful for overlaying truth points / energies)
        #     if pt is not None:
        #         mask3d_blob["truth"] = {
        #             "coord":     getattr(pt, "coord", None),
        #             "E":         getattr(pt, "E", None),
        #             "E_corrected": getattr(pt, "E_corrected", None),
        #             "pid":       getattr(pt, "pid", None),
        #             "vertex":    getattr(pt, "vertex", None),
        #             "angle":     getattr(pt, "angle", None),
        #         }
        #         mask3d_blob["truth"] = {
        #             k: (v.detach().cpu() if hasattr(v, "detach") else v)
        #             for k, v in mask3d_blob["truth"].items() if v is not None
        #         }
        #     if queries is not None:
        #         mask3d_blob["queries"] = queries[i].detach().cpu()    # (N_q, D)
        #     if gmm_snapshot is not None:
        #         mask3d_blob["gmm"] = gmm_snapshot
        #     dic["mask3d"] = mask3d_blob

        #     graphs_dir = os.path.join(path_save, "graphs")
        #     os.makedirs(graphs_dir, exist_ok=True)
        #     torch.save(
        #         dic,
        #         os.path.join(graphs_dir, f"{local_rank}_{step}_{i}.pt"),
        #     )
        if predict and pandora_available:
            (
                shower_p_unique_pandora,
                row_ind_pandora,
                col_ind_pandora,
                i_m_w_pandora,
                _,
            ) = match_showers(
                labels_pandora, dic, particle_ids, ev_mask_logits,
                local_rank, i, path_save, pandora=True, tracks=tracks,
            )
        if len(shower_p_unique) > 1:
            df_event1, number_of_showers_total1, number_of_fake_showers_total1 = generate_showers_data_frame(
                labels,
                dic,
                shower_p_unique,
                particle_ids,
                row_ind,
                col_ind,
                i_m_w,
                e_corr=e_corr,
                number_of_showers_total=number_of_showers_total1,
                step=step,
                number_in_batch=total_number_events,
                tracks=tracks,
                ec_x=None,
                shap_vals=None,
                pred_pos=pred_pos,
                pred_ref_pt=pred_ref_pt,
                pred_pid=pred_pid,
                save_plots_to_folder=path_save + "/Mask3D_",
                number_of_fakes=number_of_fakes,
                number_of_fake_showers_total=number_of_fake_showers_total1,
                extra_features=extra_features,
                labels_clusters_removed_tracks=None,
            )
            if len(df_event1) > 1:
                df_list1.append(df_event1)

            if predict and pandora_available:
                df_event_pandora = generate_showers_data_frame(
                    labels_pandora,
                    dic,
                    shower_p_unique_pandora,
                    particle_ids,
                    row_ind_pandora,
                    col_ind_pandora,
                    i_m_w_pandora,
                    pandora=True,
                    step=step,
                    number_in_batch=total_number_events,
                    tracks=tracks,
                    save_plots_to_folder=path_save + "/Pandora",
                )
                if df_event_pandora is not None and not isinstance(df_event_pandora, tuple):
                    df_list_pandora.append(df_event_pandora)

            total_number_events += 1

    df_batch1 = pd.concat(df_list1) if df_list1 else pd.DataFrame()
    df_batch_pandora = (
        pd.concat(df_list_pandora) if (predict and pandora_available and df_list_pandora) else []
    )

    if store:
        store_at_batch_end(
            path_save,
            df_batch1,
            df_batch_pandora,
            local_rank,
            step,
            epoch,
            predict=predict,
            store=store_epoch,
            pandora_available=pandora_available,
        )

    if predict:
        return df_batch_pandora, df_batch1, total_number_events
    return df_batch1
