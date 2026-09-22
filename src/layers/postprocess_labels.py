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
    POST_TRACK_LONELY=1            tracks still unattached become one-track
                                   clusters (charged particle with no calo
                                   deposit); default off
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


def _cfg():
    if os.environ.get("CLUSTER_POST", "0") != "1":
        return None
    return {
        "r_attach": float(os.environ.get("POST_TRACK_ATTACH_R", "0")),
        "attach_any": os.environ.get("POST_TRACK_ATTACH_ANY", "0") == "1",
        "lonely": os.environ.get("POST_TRACK_LONELY", "0") == "1",
        # Momentum cap for lone tracks [GeV]. Default: none (unchanged behaviour). 50 = the LEP-1 beam
        # energy; about 1 % of simulated DELPHI events hold a mis-measured track above it.
        "lonely_pmax": float(os.environ.get("POST_TRACK_LONELY_PMAX", "inf")),
        "stats": os.environ.get("POST_STATS", "0") == "1",
        "recheck": os.environ.get("POST_RECHECK", "0") == "1",
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
    n_att = n_lonely = 0

    # ---------------- 1. attach unassigned tracks to the nearest cluster ----
    if cfg["r_attach"] > 0:
        un_idx = torch.nonzero(is_trk & (lab == 0)).view(-1)
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
    # ---------------- 1b. every track still unassigned becomes its own particle ----
    # Independent of the attachment radius (it used to sit inside the block above, so with
    # POST_TRACK_ATTACH_R=0 it silently did nothing). With attachment on, the order is unchanged:
    # attach first, then promote what is left.
    if cfg["lonely"]:
        keep = is_trk & (lab == 0)
        if cfg.get("lonely_pmax", float("inf")) != float("inf"):
            keep = keep & (g.ndata["p_hits"].view(-1).float() < cfg["lonely_pmax"])
        still = torch.nonzero(keep).view(-1)
        if still.numel():
            start = int(lab.max().item()) + 1
            lab[still] = torch.arange(start, start + still.numel(),
                                      device=lab.device, dtype=lab.dtype)
            n_lonely = int(still.numel())

    if cfg.get("recheck") and n_att:
        # lazy import: inference_oc imports modules that import this one
        from src.layers.inference_oc import remove_bad_tracks_from_cluster_v1
        lab, _ = remove_bad_tracks_from_cluster_v1(g, lab)

    if cfg["stats"]:
        print(f"[cluster-post] tracks attached={n_att} lonely={n_lonely} "
              f"clusters={int(len(torch.unique(lab[lab > 0])))}", flush=True)
    return lab
