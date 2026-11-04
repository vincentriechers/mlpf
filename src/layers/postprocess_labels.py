"""Post-hoc cluster-label repair, shared by the Mask3D and HitPF inference paths.

One operation, aimed at a defect that was MEASURED on the DELPHI energy
budget (`delphi_converter/analysis/energy_where.py`, 15 000 events, Mask3D
32-epoch + valid-argmax configuration; numbers are shares of the event's truth
energy):

  TRACK ATTACHMENT. 12 % of the energy is charged particles that have a
     track in the event but not in their cluster (median response 0.43), and a
     further 20 % is charged particles whose track carries NO truth label at
     all — 64.8 % of DELPHI tracker hits are unlabelled. A learned assignment
     can never recover the second group because there is nothing to learn from.
     Geometry can: every track node's position is its extrapolated calorimeter
     impact point (`X_track[:, 12:15]`, `referencePoint_calo`, stored in
     `pos_hits_xyz`), so an unassigned track is attached to the cluster whose
     nearest calo hit lies within `r_attach` of that point. This is what DPC
     does for HitPF implicitly (track nodes live in the same clustering space)
     and what DELANA does explicitly.

The step is OFF unless enabled from the environment, so every existing
evaluation is reproduced bit-for-bit:

    CLUSTER_POST=1                 master switch
    POST_TRACK_ATTACH_R=<mm>       step 1 radius (0 = off)
    POST_TRACK_ATTACH_ANY=1        allow attaching to clusters that already hold
                                   a track (default: only to trackless clusters)
    POST_TRACK_PMAX_ALL=<GeV>      a track at or above this momentum is used nowhere:
                                   it leaves its cluster (which then takes its energy
                                   from the calorimeter), is not attached and is not
                                   promoted. 45.6 = the beam energy; above it a track
                                   is mis-measured. Default: off.
    POST_TRACK_ONE_PER_CLUSTER=1   a cluster keeps one track, the one whose calorimeter entry point is
                                   nearest its energy-weighted calorimeter centroid; every other track
                                   is released and, with POST_TRACK_LONELY, becomes its own charged
                                   particle. The energy step takes ONE track momentum per cluster, so
                                   without this a multi-track cluster drops the others. Default: off.
    POST_TRACK_LONELY=1            tracks still unattached become one-track
                                   clusters (charged particle with no calo
                                   deposit); default off
    POST_NEUTRAL_MINCELLS=<n>      a cluster WITHOUT a track, mostly electromagnetic and in the barrel (HPC),
                                   whose cells sum to more than POST_NEUTRAL_MINCELLS_E GeV (default 1) needs
                                   at least n cells, or it is dissolved (cells unassigned). A photon of that
                                   energy fills six or more cells; in recorded 1994 data single-pad noise
                                   strings make 1-5 GeV clusters of 1-5 cells (12 GeV per event, 0.2 GeV in
                                   simulation; delphi_study/REAL_DATA_ZPEAK.md section 6). Default 0: off.
    POST_RECHECK=1                 re-run the track/calo consistency check
                                   (`remove_bad_tracks_from_cluster_v1`) AFTER
                                   attachment. Attachment runs after the
                                   pipeline's own check, so attached tracks are
                                   otherwise never checked — and on the Mask3D
                                   path that check is worth 0.03 in rel68.
    POST_STATS=1                   per-event counts to stdout

The function is deterministic and per event, which matters: on the Mask3D path
the labels are computed twice, once before the energy-correction stage and once
when the dataframe is built, and the two must agree exactly or the per-shower
arrays come out different lengths. Call it in BOTH places, after
`remove_bad_tracks_from_cluster_v1` and before the re-densify.

Units: `pos_hits_xyz` is in mm (the loader multiplies the cm parquet by 10).
"""
import os

import torch

TRACK_HIT_TYPE = 1
BARREL_HALF_LENGTH_MM = 2700.0     # HPC cells end at |z| = 254 cm, FEMC blocks start at 284 cm


def _cfg():
    if os.environ.get("CLUSTER_POST", "0") != "1":
        return None
    return {
        "r_attach": float(os.environ.get("POST_TRACK_ATTACH_R", "0")),
        "attach_any": os.environ.get("POST_TRACK_ATTACH_ANY", "0") == "1",
        "lonely": os.environ.get("POST_TRACK_LONELY", "0") == "1",
        "track_pmax": float(os.environ.get("POST_TRACK_PMAX_ALL", "inf")),
        "one_track": os.environ.get("POST_TRACK_ONE_PER_CLUSTER", "0") == "1",
        # Momentum cap for lone tracks [GeV]. Default: none (unchanged behaviour). 45.6 = the beam energy
        # at the Z peak, the most any single particle can carry (the rest of the event recoils against it).
        "lonely_pmax": float(os.environ.get("POST_TRACK_LONELY_PMAX", "inf")),
        # Maximum |D0| [cm] for a track to be promoted. A track from the primary vertex has
        # |D0| ~ 0.09 cm; one left unlabelled by the truth (i.e. a decay or interaction
        # product, or a fake) has ~0.9 cm. Promoting the latter risks counting a particle
        # whose parent, or whose siblings, are already reconstructed. Default: no cut.
        "lonely_maxd0": float(os.environ.get("POST_TRACK_LONELY_MAXD0", "inf")),
        "lonely_maxz0": float(os.environ.get("POST_TRACK_LONELY_MAXZ0", "inf")),
        "stats": os.environ.get("POST_STATS", "0") == "1",
        "recheck": os.environ.get("POST_RECHECK", "0") == "1",
        "neutral_mincells": int(os.environ.get("POST_NEUTRAL_MINCELLS", "0")),
        "neutral_mincells_e": float(os.environ.get("POST_NEUTRAL_MINCELLS_E", "1.0")),
    }


def postprocess_labels(g, labels, cfg=None):
    """labels: (N,) int64, 0 = unassigned, 1..K clusters. Returns new labels
    with the same convention (NOT re-densified — the caller does that, exactly
    as it already does for the bad-track step)."""
    cfg = cfg or _cfg()
    if cfg is None:
        return labels
    lab = labels.view(-1).clone().long()
    pos = g.ndata["pos_hits_xyz"][:, :3].float()
    ht = g.ndata["hit_type"].view(-1).long()
    e = g.ndata["e_hits"].view(-1).float()
    is_trk = ht == TRACK_HIT_TYPE
    is_calo = ~is_trk
    n_att = n_lonely = n_over = n_split = 0

    # ---------------- 0. a track above the beam energy is used nowhere ----
    over = torch.zeros_like(is_trk)
    if cfg.get("track_pmax", float("inf")) != float("inf"):
        over = is_trk & (g.ndata["p_hits"].view(-1).float() >= cfg["track_pmax"])
        n_over = int((over & (lab > 0)).sum())
        lab[over] = 0

    # ---------------- 1. attach unassigned tracks to the nearest cluster ----
    if cfg["r_attach"] > 0:
        un_idx = torch.nonzero(is_trk & (lab == 0) & ~over).view(-1)
        cal_idx = torch.nonzero(is_calo & (lab > 0)).view(-1)
        if un_idx.numel() and cal_idx.numel():
            d = torch.cdist(pos[un_idx], pos[cal_idx])           # (n_un, n_cal)
            if not cfg["attach_any"]:
                K = int(lab.max().item())
                has_trk = torch.zeros(K + 1, dtype=torch.bool, device=lab.device)
                has_trk[lab[is_trk & (lab > 0)]] = True
                d = d.masked_fill(has_trk[lab[cal_idx]].view(1, -1), float("inf"))
            dmin, j = d.min(dim=1)
            ok = dmin < cfg["r_attach"]
            if ok.any():
                # one track per cluster when only trackless clusters are
                # allowed: resolve collisions by keeping the closest track
                tgt = lab[cal_idx[j[ok]]]
                src = un_idx[ok]
                order = torch.argsort(dmin[ok])
                taken = set()
                for o in order.tolist():
                    c = int(tgt[o])
                    if not cfg["attach_any"] and c in taken:
                        continue
                    lab[src[o]] = c
                    taken.add(c)
                    n_att += 1
    # ---------------- 1a. one track per cluster ----
    if cfg.get("one_track"):
        trk_idx = torch.nonzero(is_trk & (lab > 0)).view(-1)
        if trk_idx.numel():
            cl = lab[trk_idx]
            uniq, counts = torch.unique(cl, return_counts=True)
            for c in uniq[counts > 1].tolist():
                t = trk_idx[cl == c]
                calo = is_calo & (lab == c)
                keep = t[0]
                if calo.any():
                    w = e[calo].clamp(min=0) + 1e-9
                    cen = (pos[calo] * w.view(-1, 1)).sum(0) / w.sum()
                    keep = t[torch.argmin(torch.norm(pos[t] - cen, dim=1))]
                rel = t[t != keep]
                lab[rel] = 0
                n_split += int(rel.numel())

    # ---------------- 1b. every track still unassigned becomes its own particle ----
    # Independent of the attachment radius (it used to sit inside the block above, so with
    # POST_TRACK_ATTACH_R=0 it silently did nothing). With attachment on, the order is unchanged:
    # attach first, then promote what is left.
    if cfg["lonely"]:
        keep = is_trk & (lab == 0) & ~over
        if cfg.get("lonely_pmax", float("inf")) != float("inf"):
            keep = keep & (g.ndata["p_hits"].view(-1).float() < cfg["lonely_pmax"])
        if cfg.get("lonely_maxd0", float("inf")) != float("inf") and "d0_tracks" in g.ndata:
            keep = keep & (g.ndata["d0_tracks"].view(-1).float() < cfg["lonely_maxd0"])
        if cfg.get("lonely_maxz0", float("inf")) != float("inf") and "z0_tracks" in g.ndata:
            keep = keep & (g.ndata["z0_tracks"].view(-1).float() < cfg["lonely_maxz0"])
        still = torch.nonzero(keep).view(-1)
        if still.numel():
            start = int(lab.max().item()) + 1
            lab[still] = torch.arange(start, start + still.numel(),
                                      device=lab.device, dtype=lab.dtype)
            n_lonely = int(still.numel())

    # ---------------- 2. an energetic barrel EM cluster without a track needs enough cells ----
    n_few = 0
    if cfg.get("neutral_mincells", 0) > 0 and int(lab.max().item()) > 0:
        K = int(lab.max().item()) + 1
        has_trk = torch.zeros(K, dtype=torch.bool, device=lab.device)
        has_trk[lab[is_trk & (lab > 0)]] = True
        cl = lab[is_calo]
        n_em = torch.bincount(cl[ht[is_calo] == 2], minlength=K)
        n_all = torch.bincount(cl, minlength=K)
        w = e[is_calo].clamp(min=0)
        e_sum = torch.bincount(cl, weights=w, minlength=K)
        z_cen = torch.bincount(cl, weights=w * pos[is_calo, 2], minlength=K) / e_sum.clamp(min=1e-9)
        bad = (~has_trk & (n_all > 0) & (2 * n_em >= n_all) & (z_cen.abs() < BARREL_HALF_LENGTH_MM)
               & (e_sum > cfg.get("neutral_mincells_e", 1.0)) & (n_all < cfg["neutral_mincells"]))
        bad[0] = False
        n_few = int(bad.sum())
        lab[bad[lab] & is_calo] = 0

    if cfg.get("recheck") and n_att:
        # lazy import: inference_oc imports modules that import this one
        from src.layers.inference_oc import remove_bad_tracks_from_cluster_v1
        lab, _ = remove_bad_tracks_from_cluster_v1(g, lab)

    if cfg["stats"]:
        print(f"[cluster-post] tracks attached={n_att} lonely={n_lonely} above-pmax removed={n_over} released from multi-track clusters={n_split} few-cell neutral clusters dissolved={n_few} "
              f"clusters={int(len(torch.unique(lab[lab > 0])))}", flush=True)
    return lab
