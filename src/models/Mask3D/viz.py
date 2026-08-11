"""Mask3D training-time visualizations.

The OC pipeline plots a learned 3-D *clustering space* (via
`PlotCoordinates`); Mask3D has no such space — predictions are mask
logits per (query, hit). The natural analogue is to overlay predicted
vs. true cluster IDs on the actual detector hit positions, which is what
this module does.

Used from `mask3d_model.training_step` every N optimizer steps.
"""
import dgl
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from src.layers.inference_oc_mask3d import labels_from_masks


@torch.no_grad()
def plot_pred_vs_true_clusters_fig(
    batch_g,
    mask_logits,
    cls_logits,
    key_valid,
    event_idx=0,
    mask_threshold=0.5,
    cls_threshold=0.5,
    marker_size=2,
):
    """Build a side-by-side plotly figure: hit xyz coloured by predicted
    cluster ID (left) vs. true particle ID (right) for one event.

    Args:
        batch_g:       DGL batched graph (uses event `event_idx`).
        mask_logits:   (B, N_q, N_max) padded mask logits in feed order.
        cls_logits:    (B, N_q) or (B, N_q, 1) per-query validity logits.
        key_valid:     (B, N_max) bool, True where a real hit lives.
        event_idx:     which event in the batch to plot.
        mask_threshold/cls_threshold: passed through to `labels_from_masks`.
        marker_size:   plotly marker size; small (2-3) keeps thousands of
                       hits readable.

    Returns:
        plotly.graph_objects.Figure
    """
    graphs = dgl.unbatch(batch_g)
    if event_idx >= len(graphs):
        event_idx = 0
    g_i = graphs[event_idx]

    pos = g_i.ndata["pos_hits_xyz"].detach().float().cpu().numpy()  # (n_i, 3)
    true_labels = g_i.ndata["particle_number"].detach().long().cpu().numpy()

    n_i = int(key_valid[event_idx].sum().item())
    ev_mask_logits = mask_logits[event_idx, :, :n_i]
    if cls_logits.dim() == 3:
        ev_cls_logits = cls_logits[event_idx].squeeze(-1)
    else:
        ev_cls_logits = cls_logits[event_idx]

    pred_labels = labels_from_masks(
        ev_mask_logits, ev_cls_logits,
        mask_threshold=mask_threshold,
        cls_threshold=cls_threshold,
    ).detach().cpu().numpy()

    # Defensive: pos_hits_xyz might include trailing nodes that were not in
    # the encoder's feed order (shouldn't happen, but guard).
    n_plot = min(pos.shape[0], pred_labels.shape[0], true_labels.shape[0])
    pos = pos[:n_plot]
    pred_labels = pred_labels[:n_plot]
    true_labels = true_labels[:n_plot]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(
            f"Predicted clusters ({len(set(pred_labels.tolist()))} ids, "
            f"{int((pred_labels == 0).sum())} noise)",
            f"True particles ({len(set(true_labels.tolist()))} ids)",
        ),
        horizontal_spacing=0.02,
    )
    common_marker = dict(size=marker_size, colorscale="Turbo",
                         showscale=False, line=dict(width=0))
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker={**common_marker, "color": pred_labels},
            name="pred",
            hovertemplate="cid=%{marker.color}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker={**common_marker, "color": true_labels},
            name="true",
            hovertemplate="pid=%{marker.color}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        height=600, width=1300, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


@torch.no_grad()
def plot_query_frames_fig(
    points_padded,   # (B, N_k, 3) hit xyz in the decoder's working frame
    key_valid,       # (B, N_k) bool
    gt_mask,         # (B, N_p, N_k) bool
    gt_valid,        # (B, N_p) bool
    T,               # (B, N_q, 3, 3) per-query rotation (col 2 = T·ẑ axis)
    t,               # (B, N_q, 3)    per-query translation
    matched_q,       # (B, N_p) int — query index matched to each GT slot
    target_E,        # (B, N_p) true shower energy
    particle_axis=None,   # (B, N_p, 3) true unit direction (for reference)
    event_idx=0,
    n_showers=4,
    marker_size=2,
):
    """One event, a few showers: hit cloud + the MATCHED query's frame.

    For each of the `n_showers` most energetic valid showers:
      * its hits (one colour),
      * the matched query translation `t` (diamond),
      * the matched query axis arrow `t → t + L·(T·ẑ)` (solid, same colour)
        — this is the FAPE-supervised direction,
      * the true particle axis through the hit centroid (dashed black) when
        `particle_axis` is given.
    Arrow length `L` = the shower's hit RMS extent, so a well-aligned query
    arrow lies along the shower's principal elongation.
    """
    b = event_idx
    kv = key_valid[b].bool()
    pos = points_padded[b][kv].detach().float().cpu()                # (n_i, 3)
    gm = gt_mask[b][:, kv].bool().detach().cpu()                     # (N_p, n_i)
    gv = gt_valid[b].bool().detach().cpu()
    tE = target_E[b].detach().float().cpu()
    order = torch.argsort(torch.where(gv, tE, tE.new_full(tE.shape, -1)),
                          descending=True)
    sel = [int(p) for p in order if bool(gv[p]) and int(gm[p].sum()) > 0][:n_showers]

    palette = ["#e6194B", "#3cb44b", "#4363d8", "#f58231",
               "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
    fig = go.Figure()
    # faint full event for context
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers",
        marker=dict(size=1, color="lightgray", opacity=0.25),
        name="all hits", hoverinfo="skip",
    ))
    for j, p in enumerate(sel):
        col = palette[j % len(palette)]
        hp = pos[gm[p]]                                              # (m, 3)
        c = hp.mean(0)
        ext = float(hp.std(0).norm().clamp(min=1e-3)) * 2.0          # arrow len
        q = int(matched_q[b, p])
        tq = t[b, q].detach().float().cpu()
        axis = T[b, q, :, 2].detach().float().cpu()                  # T·ẑ
        axis = axis / axis.norm().clamp(min=1e-6)
        tip = tq + ext * axis
        fig.add_trace(go.Scatter3d(
            x=hp[:, 0], y=hp[:, 1], z=hp[:, 2], mode="markers",
            marker=dict(size=marker_size, color=col),
            name=f"shower p{p} E={tE[p]:.1f} (q{q})",
        ))
        fig.add_trace(go.Scatter3d(
            x=[tq[0]], y=[tq[1]], z=[tq[2]], mode="markers",
            marker=dict(size=5, color=col, symbol="diamond"),
            name=f"q{q} t", showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[tq[0], tip[0]], y=[tq[1], tip[1]], z=[tq[2], tip[2]],
            mode="lines", line=dict(color=col, width=6),
            name=f"q{q} axis", showlegend=False,
        ))
        if particle_axis is not None:
            pa = particle_axis[b, p].detach().float().cpu()
            pa = pa / pa.norm().clamp(min=1e-6)
            a0, a1 = c - ext * pa, c + ext * pa
            fig.add_trace(go.Scatter3d(
                x=[a0[0], a1[0]], y=[a0[1], a1[1]], z=[a0[2], a1[2]],
                mode="lines", line=dict(color="black", width=3, dash="dash"),
                name=f"p{p} true axis", showlegend=False,
            ))
    fig.update_layout(
        height=750, width=950,
        title=f"event {b}: matched-query frames vs showers "
              f"(solid=query T·ẑ, dashed=true axis)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
