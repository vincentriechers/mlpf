"""Build per-batch tensors for Mask3D-style training.

Hit-level data stays *flat* — `(total_hits, ...)` — so the encoder can run
on a concatenated token stream with a BlockDiagonalMask (mirrors the
GATr model's pattern in `src/models/GATr/Gatr_pf_e_noise.py:146` and
hepattn's `flash-varlen` path). Only the per-particle GT (which is
needed for the (B, N_q, N_p) Hungarian cost matrix) is padded.

Returned dict:
    feats_flat   (total_hits, F)        raw per-hit features, concatenated
    seq_lens     list[int]              hits per event (sums to total_hits)
    batch_ids    (total_hits,)          event index per hit
    local_idx    (total_hits,)          intra-event hit index (for scattering
                                        encoder output to padded form)
    key_valid    (B, N_h_max)           True where a real hit lives in the
                                        padded layout the decoder uses
    gt_mask      (B, N_p_max, N_h_max)  bool, hit→particle assignment
    gt_valid     (B, N_p_max)           True where a real particle slot lives
    target_E     (B, N_p_max)
    target_coord (B, N_p_max, 3)
    target_pid   (B, N_p_max)
"""
import torch
import torch.nn.functional as F
import dgl


def _batch_ids_from_graph(g):
    graphs = dgl.unbatch(g)
    ids = []
    for i, gi in enumerate(graphs):
        ids.append(torch.full((gi.number_of_nodes(),), i,
                              dtype=torch.long, device=g.device))
    return torch.cat(ids, dim=0)


def build_targets(g, y, ILD=False, pos_scale=0.001):
    """Build the (padded + flat) tensors the Mask3D model consumes.

    `pos_scale` (default `0.001`, i.e. mm → m): every coordinate that
    feeds the model gets multiplied by this. Mirrors hepattn's CLD data
    loader (`convert_mm_to_m`, `cld/data.py:147`) — they note that this
    conversion is *required* to keep the positional encoding from blowing
    up. The scaling applies only to `feats_flat` here; downstream graph
    fields like `g.ndata["pos_hits_xyz"]` are left untouched in mm so the
    rest of the codebase (matching, plotting, EC pipeline) keeps using mm
    as it always has.
    """
    device = g.device
    n_oh = 6 if ILD else 5

    # ---------------- flat hit features ----------------
    pos = g.ndata["pos_hits_xyz"].float() * pos_scale         # mm → m by default
    e = g.ndata["e_hits"].float().view(-1, 1)
    p = g.ndata["p_hits"].float().view(-1, 1)
    hit_type = g.ndata["hit_type"].long().view(-1)
    chi2 = g.ndata.get("chi_squared_tracks", None)
    chi2 = chi2.float().view(-1, 1) if chi2 is not None else torch.zeros_like(e)
    ht_oh = F.one_hot(hit_type.clamp(min=0, max=n_oh - 1), num_classes=n_oh).float()
    feats_flat = torch.cat([pos, ht_oh, e, p, chi2], dim=1)       # (total_hits, F)

    batch_ids = _batch_ids_from_graph(g)
    B = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=B)
    seq_lens = counts.tolist()
    N_h_max = int(counts.max().item())

    # local_idx: position of each hit inside its event (0..n_i-1).
    # Computed as (global_idx - cumulative_offset_of_its_event).
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    if B > 1:
        offsets[1:] = counts.cumsum(0)[:-1]
    local_idx = torch.arange(batch_ids.size(0), device=device) - offsets[batch_ids]

    # key_valid in padded layout (used by the decoder cross-attn + loss)
    key_valid = torch.zeros(B, N_h_max, dtype=torch.bool, device=device)
    key_valid[batch_ids, local_idx] = True

    # Per-hit subsystem (hit_type) in padded layout — used by per_subsystem_loss
    # to split the mask loss / matching cost across detector subsystems.
    hit_subsystem = torch.zeros(B, N_h_max, dtype=torch.long, device=device)
    hit_subsystem[batch_ids, local_idx] = hit_type

    # Per-hit energy (GeV) in padded layout — used by mask3d_loss's optional
    # energy weighting (`energy_weight_mode`). `e` is g.ndata["e_hits"] and
    # is NOT scaled by pos_scale (only positions are), so this is the raw
    # physical hit energy.
    hit_e = torch.zeros(B, N_h_max, device=device)
    hit_e[batch_ids, local_idx] = e.view(-1)

    # ---------------- padded GT mask ----------------
    particle_number = g.ndata["particle_number"].long().view(-1)
    p_counts = []
    for i in range(B):
        m = batch_ids == i
        if m.any():
            p_counts.append(int(particle_number[m].max().item()))
        else:
            p_counts.append(0)
    N_p_max = max(max(p_counts) if p_counts else 0, 1)

    gt_mask = torch.zeros(B, N_p_max, N_h_max, dtype=torch.bool, device=device)
    gt_valid = torch.zeros(B, N_p_max, dtype=torch.bool, device=device)
    for i in range(B):
        m = batch_ids == i
        n_p_i = p_counts[i]
        if n_p_i == 0:
            continue
        gt_valid[i, :n_p_i] = True
        ev_part = particle_number[m]
        ev_idx = local_idx[m]
        real = ev_part > 0
        if real.any():
            p_slot = ev_part[real] - 1
            h_slot = ev_idx[real]
            gt_mask[i, p_slot, h_slot] = True

    # ---------------- per-particle attrs (padded) ----------------
    target_E = torch.zeros(B, N_p_max, device=device)
    target_coord = torch.zeros(B, N_p_max, 3, device=device)
    target_pid = torch.zeros(B, N_p_max, dtype=torch.long, device=device)
    if y is not None and getattr(y, "E", None) is not None:
        y_batch = y.batch_number.view(-1).long().to(device)
        y_E = y.E.view(-1).float().to(device)
        y_pid = y.pid.view(-1).long().to(device) if y.pid is not None else None
        y_coord = y.coord.float().to(device) if y.coord is not None else None
        for i in range(B):
            mask = y_batch == i
            n_part_i = int(mask.sum().item())
            if n_part_i == 0:
                continue
            n_fill = min(n_part_i, N_p_max)
            target_E[i, :n_fill] = y_E[mask][:n_fill]
            if y_coord is not None:
                target_coord[i, :n_fill] = y_coord[mask][:n_fill]
            if y_pid is not None:
                target_pid[i, :n_fill] = y_pid[mask][:n_fill]

    # ---------------- frame-supervision targets (optional) ----------------
    # Per-particle unit flight direction = unit(MC-truth momentum).
    # `y.coord` = features_particles[:,12:15] is the EDM4hep
    # MCParticle.momentum (px,py,pz) — a MOMENTUM 3-vector, NOT a position
    # (despite the `coord` name). So the exact particle direction is just
    # its normalized momentum: no (eta,phi) reconstruction, no IP / vertex /
    # endpoint assumption, no angle-convention risk.
    #
    # `target_coord` above is filled from the same `y.coord`, so it already
    # holds the momentum; we reuse it. The (eta,phi) path below
    # (`y.angle` = features_particles[:,4:6], col0 = PSEUDORAPIDITY not θ;
    # n = [cosφ·sech η, sinφ·sech η, tanh η]) is kept ONLY as a fallback
    # for particles whose momentum is missing/zero.
    target_angle = torch.zeros(B, N_p_max, 2, device=device)
    if y is not None and getattr(y, "angle", None) is not None:
        y_angle_all = y.angle.float().to(device)
        for i in range(B):
            mask = y_batch == i
            n_part_i = int(mask.sum().item())
            if n_part_i == 0:
                continue
            n_fill = min(n_part_i, N_p_max)
            target_angle[i, :n_fill] = y_angle_all[mask][:n_fill]

    eta = target_angle[..., 0]
    phi = target_angle[..., 1]
    sech = 1.0 / torch.cosh(eta)
    axis_from_angle = torch.stack(
        [sech * torch.cos(phi), sech * torch.sin(phi), torch.tanh(eta)],
        dim=-1,
    )                                                                       # (B, N_p_max, 3)
    target_mom = target_coord                                               # (B,N_p,3) = momentum
    mom_norm = target_mom.norm(dim=-1, keepdim=True)
    particle_axis = torch.where(
        mom_norm > 1e-6, target_mom / mom_norm.clamp(min=1e-6), axis_from_angle
    )                                                                       # (B, N_p_max, 3)

    # ---- track-based axis for CHARGED particles (TrackState::AtCalorimeter) ----
    # For a particle WITH a reconstructed track the physically correct
    # supervision is the track momentum AT THE CALORIMETER (the
    # post-B-field-curvature incidence direction), anchored at the track's
    # referencePoint_calo — NOT the production momentum (pre-curvature) and
    # NOT the IP→hit line. Track hits are exactly the hits whose
    # `pos_pxpypz_at_calo` is non-zero (calo hits store zeros); a track
    # hit's position in `pos` IS referencePoint_calo. Neutrals (no track)
    # keep the unit(production momentum) axis above.
    pcalo = (g.ndata["pos_pxpypz_at_calo"]
             if "pos_pxpypz_at_calo" in g.ndata else None)
    BN = B * N_p_max
    slot = particle_number - 1
    base_valid = (particle_number > 0) & (slot >= 0) & (slot < N_p_max)
    flat = batch_ids * N_p_max + slot.clamp(min=0)
    # default (neutral) reference point = centroid of all the particle's hits
    cen_sum = torch.zeros(BN, 3, device=device)
    cen_cnt = torch.zeros(BN, device=device)
    if base_valid.any():
        cen_sum.index_add_(0, flat[base_valid], pos[base_valid].to(cen_sum.dtype))
        cen_cnt.index_add_(0, flat[base_valid],
                           torch.ones(int(base_valid.sum()), device=device))
    particle_ref = (cen_sum / cen_cnt.clamp(min=1).unsqueeze(-1)).view(B, N_p_max, 3)
    has_track = torch.zeros(B, N_p_max, dtype=torch.bool, device=device)
    if pcalo is not None:
        pcalo = pcalo.float()
        track_hit = base_valid & (pcalo.norm(dim=1) > 1e-9)
        dir_sum = torch.zeros(BN, 3, device=device)
        ref_sum = torch.zeros(BN, 3, device=device)
        tcnt = torch.zeros(BN, device=device)
        if track_hit.any():
            dir_sum.index_add_(0, flat[track_hit], pcalo[track_hit])
            ref_sum.index_add_(0, flat[track_hit], pos[track_hit].to(ref_sum.dtype))
            tcnt.index_add_(0, flat[track_hit],
                            torch.ones(int(track_hit.sum()), device=device))
        has_track = (tcnt > 0).view(B, N_p_max)
        td = dir_sum.view(B, N_p_max, 3)
        track_axis = td / td.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        track_ref = (ref_sum / tcnt.clamp(min=1).unsqueeze(-1)).view(B, N_p_max, 3)
        m3 = has_track.unsqueeze(-1)
        particle_axis = torch.where(m3, track_axis, particle_axis)
        particle_ref = torch.where(m3, track_ref, particle_ref)

    # Frame-supervision filter: only particles with a RECONSTRUCTED TRACK
    # and E ≥ 1 GeV. These are exactly the ones for which the rotation
    # target is the post-curvature track momentum at the calo (the most
    # reliable direction). Charged-no-track particles fall back to the
    # production momentum above, but we explicitly exclude them from the
    # frame loss to keep supervision tight.
    is_min_E = target_E >= 1.0                                              # (B, N_p_max)
    particle_supervisable = has_track & is_min_E & gt_valid                 # (B, N_p_max)

    return {
        "feats_flat": feats_flat,
        "seq_lens": seq_lens,
        "batch_ids": batch_ids,
        "local_idx": local_idx,
        "key_valid": key_valid,
        "hit_subsystem": hit_subsystem,
        "hit_e": hit_e,
        # Flat (total_hits,) integer hit_type — used by PerSubsystemInputNet
        # to route each hit to its detector-specific input net.
        "hit_type_flat": hit_type,
        "gt_mask": gt_mask,
        "gt_valid": gt_valid,
        "target_E": target_E,
        "target_coord": target_coord,
        "target_pid": target_pid,
        # Frame-supervision targets (used by FAPE-style frame loss in mask3d_loss).
        "particle_axis": particle_axis,
        "particle_supervisable": particle_supervisable,
        # Diagnostics/validation: raw (eta,phi) and the MC momentum vector
        # (y.coord = EDM4hep MCParticle.momentum) the axis is derived from.
        "target_angle": target_angle,
        "target_mom": target_mom,
        # Per-particle reference point for the axis (track referencePoint_calo
        # for charged-with-track, else hit centroid) + has-track flag.
        "particle_ref": particle_ref,
        "has_track": has_track,
    }
