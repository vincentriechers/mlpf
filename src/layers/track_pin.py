"""Track-owned queries: pin each tracker hit to the query it seeded.

Decoder queries are seeded from encoder tokens with tracker hits first
(`track_seed_bonus`), so query q is born from the tracker hit `seed_idx[q]`.
With `track_pin` on, that link is used in two places:

  * matching (training): the Hungarian cost of (q, the truth particle of its
    seed track) gets a large bonus, so the query is trained to be that
    particle's cluster. Particles without a track, and tracks whose particle
    is not a target, are matched as before.
  * label derivation (inference): the tracker hit joins its seeded query
    whenever that query is valid, regardless of mask probability, so a track
    can never be dropped from the cluster it seeded. Calo hits follow the
    usual argmax/threshold rule.

Both helpers are plain torch so they can be unit-tested without dgl.
"""
import torch


def track_pin_matrix(seed_idx, key_valid, hit_subsystem, key_slot, gt_valid, track_hit_type=1):
    """(B, N_q, N_p) bool: True where query q was seeded by a tracker hit of GT slot p.

    seed_idx:      (B, N_q) long, key index that seeded each query, -1 if none.
    key_valid:     (B, N_k) bool.
    hit_subsystem: (B, N_k) long, per-key hit_type.
    key_slot:      (B, N_k) long, GT particle slot of each key (particle_number - 1), -1 for noise/pad.
    gt_valid:      (B, N_p) bool.
    """
    B, N_q = seed_idx.shape
    N_p = gt_valid.shape[1]
    pin = torch.zeros(B, N_q, N_p, dtype=torch.bool, device=seed_idx.device)
    if N_p == 0 or N_q == 0:
        return pin
    k = seed_idx.clamp(min=0)
    ok = (seed_idx >= 0) & key_valid.gather(1, k) & (hit_subsystem.gather(1, k) == track_hit_type)
    slot = key_slot.gather(1, k)
    ok = ok & (slot >= 0) & (slot < N_p)
    ok = ok & gt_valid.gather(1, slot.clamp(min=0, max=N_p - 1))
    b, q = torch.nonzero(ok, as_tuple=True)
    pin[b, q, slot[b, q]] = True
    return pin


def pin_tracks_to_seed_queries(argmax_q, is_assigned, valid_q, seed_idx, hit_type, track_hit_type=1):
    """In place: tracker hit seed_idx[q] is assigned to query q if q is valid. Returns the count.

    argmax_q, is_assigned: (N_k,) per-hit assignment being built by labels_from_masks.
    valid_q:               (N_q,) bool.
    seed_idx:              (N_q,) long, -1 where the slot has no seed.
    hit_type:              (N_k,) long.
    """
    if seed_idx is None:
        return 0
    n_k = hit_type.numel()
    q = torch.arange(seed_idx.numel(), device=seed_idx.device)
    k = seed_idx.view(-1).to(argmax_q.device)
    ok = (k >= 0) & (k < n_k) & valid_q.view(-1)[q]
    ok = ok & (hit_type.view(-1)[k.clamp(min=0, max=max(n_k - 1, 0))] == track_hit_type)
    argmax_q[k[ok]] = q[ok].to(argmax_q.dtype)
    is_assigned[k[ok]] = True
    return int(ok.sum())
