"""Hungarian-matched Mask3D loss with deep supervision (hepattn-style).

Mirrors `hepattn.models.maskformer.MaskFormer.loss`:

  1. Per-layer cost matrices `(B, N_q, N_p)` are computed in parallel
     (parallel matching, same as Mask2Former).
  2. All layer costs are stacked into a single `(L*B, N_q, N_p)` tensor and
     the `Matcher` (see `matcher.py`) is called once. One GPU→CPU sync,
     one scipy loop (optionally thread-parallel) — instead of L per-layer
     syncs and L sequential scipy loops.
  3. The matcher returns a *full* permutation of the N_q queries: the first
     `n_real_b` slots are the matched queries (so they align with GT slot
     0..n_real-1), the rest are the unused query indices in natural order.
  4. Predictions are permuted using those indices, after which the loss
     functions use `gt_valid` directly — no `match_valid`/gather/scatter.
"""
import torch
import torch.nn.functional as F


# ---------------- pairwise costs (B, N_q, N_p) ----------------

def _bce_pairwise(logits, targets, key_valid, hit_weight=None):
    """Per-pair BCE between (B, N_q, N_k) logits and (B, N_p, N_k) bool targets,
    averaged over valid keys. Closed-form O(B*N_q*N_p + B*N_q*N_k) — no explicit
    (B, N_q, N_p, N_k) tensor.

    BCE(σ(z), y=1) = softplus(-z); BCE(σ(z), y=0) = softplus(+z).
    For target t∈{0,1}: bce = (softplus(-z) - softplus(+z))·t + softplus(+z).

    `hit_weight` (B, N_k) optionally scales each key's contribution AND
    denominator — used to upweight tracker hits so the dominant calo hits
    don't drown them out of the gradient signal.

    THE WEIGHT MUST APPEAR EXACTLY ONCE. `diff` already carries `valid_f`, so
    `t` is masked with the bare validity flag and NOT weighted again — weighting
    both sides of the einsum puts w^2 in the numerator against w in the
    denominator, which is unbounded and is what broke the dice terms below.
    """
    valid_f = key_valid.float()
    if hit_weight is not None:
        valid_f = valid_f * hit_weight.to(valid_f.dtype)
    pos_q = F.softplus(-logits) * valid_f.unsqueeze(1)
    neg_q = F.softplus(logits) * valid_f.unsqueeze(1)
    diff = pos_q - neg_q
    t = targets.float() * key_valid.unsqueeze(1).float()
    inter = torch.einsum("bnk,bpk->bnp", diff, t)
    sum_neg = neg_q.sum(dim=-1, keepdim=True)
    denom = valid_f.sum(dim=-1).view(-1, 1, 1).clamp(min=1.0)
    return (inter + sum_neg) / denom


def _focal_pairwise(logits, targets, key_valid, gamma=2.0, hit_weight=None):
    """Per-pair focal cost — hepattn `mask_focal_cost`.

        focal(z, t) = (1 - p_t)^γ · BCE(z, t)

    where p_t = σ(z) if t=1 else 1-σ(z). Same closed form as BCE pairwise:

        focal_pos_q(k) = (1 - σ(z_qk))^γ · softplus(-z_qk)     # cost if t=1
        focal_neg_q(k) = (σ(z_qk))^γ     · softplus(+z_qk)     # cost if t=0
        cost(q, p)     = mean_k[ t_pk·focal_pos + (1-t_pk)·focal_neg ]
                       = (⟨focal_pos − focal_neg, t_p⟩ + ⟨focal_neg, 1⟩) / N_valid

    Why focal here: vanilla BCE has a trivial local minimum at "predict 0
    everywhere" because masks are sparse (~1% positive rate). Focal zeroes
    the gradient on easy negatives via (σ)^γ ≈ 0, so the loss is dominated
    by the small set of real positives the model is missing — no trap.
    """
    valid_f = key_valid.float()
    if hit_weight is not None:
        valid_f = valid_f * hit_weight.to(valid_f.dtype)
    p = logits.sigmoid()
    focal_pos = ((1.0 - p) ** gamma) * F.softplus(-logits) * valid_f.unsqueeze(1)
    focal_neg = (p ** gamma) * F.softplus(logits) * valid_f.unsqueeze(1)
    diff = focal_pos - focal_neg
    # weight once: `diff` already carries valid_f (see _bce_pairwise)
    t = targets.float() * key_valid.unsqueeze(1).float()
    inter = torch.einsum("bnk,bpk->bnp", diff, t)
    sum_neg = focal_neg.sum(dim=-1, keepdim=True)
    denom = valid_f.sum(dim=-1).view(-1, 1, 1).clamp(min=1.0)
    return (inter + sum_neg) / denom


def _dice_pairwise(logits, targets, key_valid, eps=1e-6, hit_weight=None):
    """Soft-Dice cost between sigmoid(logits) and targets, padded keys masked.
    Returns (B, N_q, N_p).

    With `hit_weight` (B, N_k), each hit's contribution to intersection and
    each cardinality is scaled — tracker hits weigh more so their dice loss
    actually drives gradient instead of being averaged away by the calo hits.

    THE WEIGHT MUST APPEAR EXACTLY ONCE in each of the three sums. Scaling both
    `p` and `t` put w^2 in the intersection against w in the denominator, so
    `1 - 2*inter/(p_sum + t_sum)` became UNBOUNDED BELOW: a perfectly predicted
    cluster of one tracker hit plus nine calo hits scored -0.5 at
    track_loss_weight=3 and -4.74 at 10 instead of 0, and the optimiser chased
    it. See `tests/test_dice_hit_weight.py`.
    """
    w = key_valid.unsqueeze(1).float()
    if hit_weight is not None:
        w = w * hit_weight.to(w.dtype).unsqueeze(1)
    p = logits.sigmoid() * w                      # w * pred
    t = targets.float()                           # masked below by w in the sums
    inter = torch.einsum("bnk,bpk->bnp", p, t)    # sum_k w*pred*tgt
    p_sum = p.sum(dim=-1, keepdim=True)           # sum_k w*pred
    t_sum = (t * w).sum(dim=-1, keepdim=True).transpose(1, 2)   # sum_k w*tgt
    return 1.0 - (2.0 * inter + eps) / (p_sum + t_sum + eps)


def _cls_cost(cls_logits, gt_valid):
    """Binary objectness cost — hepattn `object_bce_cost`:

        cost[b, q, p] = -prob_q · t_p - (1 - prob_q) · (1 - t_p)

    i.e. the negative of the predicted probability of the correct class
    (DETR's "approximate CE by -probs[target_class]" trick).

    cls_logits: (B, N_q, 1); gt_valid: (B, N_p) → cost (B, N_q, N_p).
    """
    probs = cls_logits[..., 0].sigmoid().unsqueeze(-1)            # (B, N_q, 1)
    targets = gt_valid.float().unsqueeze(1)                       # (B, 1, N_p)
    return -probs * targets - (1.0 - probs) * (1.0 - targets)


def _permute(t, perm):
    """t: (B, N_q, *extra); perm: (B, N_q) -> (B, N_q, *extra) gathered along dim 1."""
    B, N_q = perm.shape
    extra = list(t.shape[2:])
    idx = perm.view(B, N_q, *([1] * len(extra))).expand(B, N_q, *extra)
    return t.gather(1, idx)


# ---------------- post-permutation losses ----------------
# After permuting predictions, query position `i` corresponds to GT slot `i`
# for i < N_p; gt_valid masks out the padded GT slots directly. No need for
# a separate match_valid mask or per-event gather.

def _mask_bce_loss(perm_mask_logits_p, gt_mask, key_valid, gt_valid, hit_weight=None):
    """perm_mask_logits_p: (B, N_p, N_k) — first N_p slots of the permuted
    queries (the matched ones)."""
    pair = F.binary_cross_entropy_with_logits(
        perm_mask_logits_p, gt_mask.float(), reduction="none"
    )
    weight = key_valid.unsqueeze(1).float() * gt_valid.unsqueeze(-1).float()
    if hit_weight is not None:
        weight = weight * hit_weight.to(weight.dtype).unsqueeze(1)
    return (pair * weight).sum() / weight.sum().clamp(min=1.0)


def _mask_focal_loss(perm_mask_logits_p, gt_mask, key_valid, gt_valid, gamma=2.0,
                     hit_weight=None):
    """Mask focal loss — hepattn `mask_focal_loss`.

        focal(z, t) = (1 - p_t)^γ · BCE(z, t)

    Easy negatives have p_t ≈ 1, so (1-p_t)^γ ≈ 0 and they contribute nearly
    no loss/gradient — the model is forced to focus on hard examples (the
    real positives in the mask).
    """
    targets = gt_mask.float()
    ce = F.binary_cross_entropy_with_logits(
        perm_mask_logits_p, targets, reduction="none"
    )
    p = perm_mask_logits_p.sigmoid()
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal = ce * ((1.0 - p_t) ** gamma)
    weight = key_valid.unsqueeze(1).float() * gt_valid.unsqueeze(-1).float()
    if hit_weight is not None:
        weight = weight * hit_weight.to(weight.dtype).unsqueeze(1)
    return (focal * weight).sum() / weight.sum().clamp(min=1.0)


def _mask_dice_loss(perm_mask_logits_p, gt_mask, key_valid, gt_valid, eps=1e-6,
                    hit_weight=None, size_weighting="none", particle_energy=None):
    """Mean per-particle dice across the matched (q, p) pairs.

    `size_weighting` controls the per-particle weight applied to each dice
    score before averaging:
      'none'   — uniform weight (1.0 per valid GT slot). Default.
      'linear' — weight = |m_p| (GT cluster hit count).
      'sqrt'   — weight = sqrt(|m_p|).
      'log'    — weight = log(|m_p| + 1).
      'energy' / 'energy_sqrt' / 'energy_log' — weight by the GT particle's
               TRUE energy (resp. E, sqrt(E), log(E+1)). Unlike the per-hit
               `energy_weight_mode`, this is a PER-SHOWER weight: it pushes
               the model to fully capture energetic showers (incl. their
               diffuse low-E periphery) without down-weighting any individual
               hit — addresses the large-shower under-collection where the
               mask gets the core but drops the halo. Requires
               `particle_energy` (B, N_p), same GT-slot order as gt_mask.
    `linear` emphasises large clusters most strongly — under-clustering them
    is then penalised proportionally to their size. `sqrt` / `log` soften
    the bias if 'linear' over-prioritises huge showers at small-cluster
    expense.
    """
    kv = key_valid.unsqueeze(1).float()
    if hit_weight is not None:
        kv = kv * hit_weight.to(kv.dtype).unsqueeze(1)
    # weight once per sum -- see _dice_pairwise
    pr = perm_mask_logits_p.sigmoid()
    tt = gt_mask.float()
    inter = (kv * pr * tt).sum(dim=-1)
    p_sum = (kv * pr).sum(dim=-1)
    t_sum = (kv * tt).sum(dim=-1)
    dice = 1.0 - (2.0 * inter + eps) / (p_sum + t_sum + eps)

    # Per-particle weighting by GT cluster size.
    raw_size = gt_mask.float().sum(dim=-1)          # (B, N_p) — GT hit count
    if size_weighting == "none":
        size_w = torch.ones_like(raw_size)
    elif size_weighting == "linear":
        size_w = raw_size
    elif size_weighting == "sqrt":
        size_w = raw_size.clamp(min=0).sqrt()
    elif size_weighting == "log":
        size_w = torch.log(raw_size.clamp(min=0) + 1.0)
    elif size_weighting in ("energy", "energy_sqrt", "energy_log"):
        if particle_energy is None:
            raise ValueError(
                f"size_weighting={size_weighting!r} requires particle_energy "
                "(targets['target_E'])"
            )
        pe = particle_energy.to(raw_size.dtype).clamp(min=0.0)
        if size_weighting == "energy":
            size_w = pe
        elif size_weighting == "energy_sqrt":
            size_w = pe.sqrt()
        else:  # energy_log
            size_w = torch.log(pe + 1.0)
    else:
        raise ValueError(
            "size_weighting must be 'none'/'linear'/'sqrt'/'log'/"
            "'energy'/'energy_sqrt'/'energy_log', "
            f"got {size_weighting!r}"
        )
    w = gt_valid.float() * size_w
    return (dice * w).sum() / w.sum().clamp(min=1.0)


def _object_bce_loss(perm_cls_logits, gt_valid, null_weight=0.25):
    """Per-query objectness loss — hepattn `ObjectClassificationTask.loss`
    (binary case) + `object_bce_loss`.

    After permutation:
        positions [0:N_p]  → matched queries; target = gt_valid (1 if real GT).
        positions [N_p:]   → unmatched queries; target = 0 (negatives).
    Padded GT slots (gt_valid=False) are also negatives — the model should
    reject them.

        sample_weight = target + null_weight · (1 - target)
        loss = F.binary_cross_entropy_with_logits(
                   logits, target, weight=sample_weight)   # default mean
    """
    B, N_q, _ = perm_cls_logits.shape
    N_p = gt_valid.size(1)

    target = torch.zeros(B, N_q, device=perm_cls_logits.device)
    target[:, :N_p] = gt_valid.float()

    sample_weight = target + null_weight * (1.0 - target)

    return F.binary_cross_entropy_with_logits(
        perm_cls_logits[..., 0], target, weight=sample_weight
    )


def _recall_loss(final_mask_logits, gt_mask, gt_valid, key_valid,
                 hit_e, target_E, tau=5.0, energy_weight_alpha=0.5, eps=1e-6):
    """Per-truth coverage loss: ensure SOME query has high σ on each GT hit.

    For each truth p and each GT hit k of p, we compute the soft-max over
    queries (log-sum-exp at temperature `tau`; tau large = near-hard-max) of
    the mask logits. Whichever query is currently winning the per-hit
    argmax — Hungarian or not — gets pushed to fire on this hit.

        smooth_max_k  = (1/τ) · logsumexp_q( τ · mask_logits[q, k] )
        best_claim_k  = σ(smooth_max_k)
        L_per_hit_k   = 1 − best_claim_k
        L_per_part p  = Σ_k hit_e[k] · L_hit_k  /  Σ_k hit_e[k]       (k in gt_mask[p])
        L_recall      = Σ_p (target_E_p^α) · L_per_part / Σ_p (target_E_p^α)

    Directly attacks the "at inference, no query crossed 0.5 on the truth's
    hits → unmatched truth" failure that the per-event Σpred/Σreco-variance
    diagnostic surfaced (model std 0.025 vs Pandora 0.010).
    """
    # Soft max over queries → (B, N_k).
    smooth_max_k = (1.0 / tau) * torch.logsumexp(
        tau * final_mask_logits, dim=1,
    )
    best_claim_k = torch.sigmoid(smooth_max_k)                       # (B, N_k)
    per_hit_miss = 1.0 - best_claim_k                                # (B, N_k)

    # Restrict to (p, k) pairs that are real hits AND in p's GT mask, and
    # weight each hit by its energy.
    mask_pk = gt_mask.float() * key_valid.unsqueeze(1).float()       # (B, N_p, N_k)
    hit_w = hit_e.to(mask_pk.dtype).unsqueeze(1) * mask_pk           # (B, N_p, N_k)
    num_p = (hit_w * per_hit_miss.unsqueeze(1)).sum(-1)              # (B, N_p)
    den_p = hit_w.sum(-1).clamp(min=eps)                             # (B, N_p)
    L_per_p = num_p / den_p                                          # (B, N_p)  ∈ [0,1]

    # Particle weight ∝ target_E^α.
    valid = gt_valid.float() * (target_E > 0).float()                # (B, N_p)
    w_p = valid * target_E.clamp(min=0).pow(float(energy_weight_alpha))
    return (w_p * L_per_p).sum() / w_p.sum().clamp(min=eps)


# ---------------- top-level loss with deep supervision ----------------

def mask3d_loss(
    per_layer_outs,
    targets,
    matcher,
    weights=None,
    aux_layer_weight=0.0,
    mask_loss_type="bce",
    focal_gamma=2.0,
    mask_cost_use_classification=False,
    obj_bce_cost_weight=1.0,
    per_subsystem_loss=False,
    subsystem_offset=0,
    track_loss_weight=1.0,
    track_hit_type=1,
    attn_bias=None,
    gmm_coverage_weight=0.0,
    gmm_bce_weight=0.0,
    dice_size_weighting="none",
    # FAPE-style frame supervision for IPA models (no-op otherwise).
    # All four kwargs must be supplied AND at least one weight > 0 for the
    # frame loss to fire. Applied only to matched (query q, GT particle p)
    # pairs in the FINAL layer that pass `targets["particle_supervisable"]`
    # (charged + E ≥ 1 GeV, see `build_targets`).
    T_final=None,          # (B, N_q, 3, 3) rotation matrices from the decoder
    t_final=None,          # (B, N_q, 3)    translations
    pos_padded=None,       # (B, N_k, 3)    per-hit positions, same frame as t
    frame_translation_weight=0.0,
    frame_rotation_weight=0.0,
    # Per-hit ENERGY weighting of the mask losses (BCE + dice + focal +
    # matching cost), folded multiplicatively into the same `hit_weight`
    # path that `track_loss_weight` uses. Makes the mask losses prioritise
    # getting energetic hits right — directly relevant to energy
    # resolution, since a shower's reconstructed energy is dominated by its
    # few high-energy core hits, not its diffuse low-energy tail.
    #   "none"   — disabled (default, fully backward-compatible).
    #   "log"    — factor = 1 + log1p(e_hit). Bounded, never < 1, so it
    #              only UP-weights energetic hits and never starves
    #              low-energy / tracker hits (preserves track weighting).
    #   "linear" — factor = 1 + e_hit / energy_weight_scale. Stronger; tune
    #              `energy_weight_scale` (GeV) so the boost isn't dominated
    #              by a handful of very energetic hits.
    # Requires `targets["hit_e"]` (per-hit energy in padded layout, built
    # by build_targets). If absent, energy weighting is silently skipped.
    energy_weight_mode="none",
    energy_weight_scale=1.0,
    # Differentiable energy-completeness term. For each matched (query, GT
    # particle) in the FINAL layer, push the SOFT mask to collect the
    # particle's full energy:
    #   collected_p = Σ_k σ(mask_qk) · e_k · key_valid
    #   L_ec        = mean_p |collected_p / E_p − 1|     (relative L1)
    # over valid GT slots with E_p > 0. Dice/BCE saturate once the cluster
    # core is covered (flat gradient on the diffuse low-prob halo); this
    # term keeps gradient exactly on the under-collected energy tail — the
    # quantity that makes Σpred/Σreco fall short of Pandora. 0.0 disables
    # it (default, fully backward-compatible). Requires targets["hit_e"]
    # and targets["target_E"] (both built by build_targets).
    energy_completeness_weight=0.0,
    # Per-truth recall loss: pushes the per-hit MAX over queries of σ(mask)
    # to 1 on every GT hit (energy-weighted), independent of the Hungarian
    # match. Directly attacks the "no query fires on this truth → unmatched"
    # failure mode that drives per-event Σpred/Σreco variance. 0.0 disables.
    recall_weight=0.0,
    recall_tau=5.0,             # soft-max temperature; ↑ = nearer hard max.
    recall_E_alpha=0.5,         # particle weight ∝ target_E^α.
):
    """Compute the Mask3D loss across decoder layers.

    per_layer_outs: list of dicts with 'mask_logits' (B, N_q, N_k) and
                    'cls_logits' (B, N_q, 1). Last entry = final layer.
    targets:        from build_targets — must contain gt_mask, gt_valid, key_valid.
    matcher:        a `Matcher` instance (see `src.models.Mask3D.matcher`).
    weights:        optional dict overriding the defaults below.
    aux_layer_weight: weight applied to non-final-layer losses, divided
                    equally among them. **0.0 disables deep supervision** —
                    only the final layer is matched and lossed (matches
                    hepattn `has_intermediate_loss: False`).
    mask_loss_type: "bce" (hepattn-style, paired with strong dice weight)
                    or "focal" (Mask2Former-style, removes BCE-trap).
                    Drives both the loss AND the matching cost.
    focal_gamma:    γ in (1 - p_t)^γ when `mask_loss_type == "focal"`.
    mask_cost_use_classification:
                    If True, the matching cost includes the mask
                    classification term (BCE or focal) weighted by
                    `weights["mask_bce"]`. If False, the matching cost
                    uses dice only — matches hepattn `base.yaml` (mask_bce
                    / mask_focal are commented out under `costs:`).
    obj_bce_cost_weight:
                    Weight on the cls cost in matching, *separate* from
                    the loss weight `weights["obj_bce"]`. hepattn keeps
                    matching at 1.0 while the loss weight is 0.1.
    per_subsystem_loss:
                    If True, the mask loss and the mask matching cost are
                    computed *per hit-type subsystem* and **summed** across
                    subsystems present in the batch. Mirrors hepattn's
                    CLD config: 5 separate `ObjectHitMaskTask` instances,
                    each contributing its own loss/cost additively (so
                    with weights mask_bce=0.25, mask_dice=0.75 the total
                    becomes 5·0.25=1.25 BCE + 5·0.75=3.75 Dice).
                    Requires `targets["hit_subsystem"]` (built by
                    `build_targets`).
    """
    # Defaults mirror hepattn `cld/configs/base.yaml`. The original
    # Mask2Former values (mask_bce=2, mask_dice=1, obj_bce=1) are also valid
    # but require focal loss to escape the BCE trap.
    w = {"mask_bce": 0.25, "mask_dice": 0.75, "obj_bce": 0.1, "null_weight": 1.0}
    if weights:
        w.update(weights)

    gt_mask = targets["gt_mask"]                                  # (B, N_p, N_k)
    gt_valid = targets["gt_valid"]                                # (B, N_p)
    key_valid = targets["key_valid"]                              # (B, N_k)
    B, N_p = gt_valid.shape
    n_layers = len(per_layer_outs)
    N_q = per_layer_outs[0]["mask_logits"].size(1)

    # Per-key loss weight: 1.0 everywhere, `track_loss_weight` on tracker hits.
    # Tracker hits are ~5–50 / event vs thousands of calo hits, so without this
    # the model can score ~99% dice while completely ignoring tracker hits.
    # Boosting their contribution forces the matcher and the loss to actually
    # pay attention to them.
    hit_weight = None
    if track_loss_weight is not None and track_loss_weight != 1.0:
        hit_sub = targets.get("hit_subsystem")
        if hit_sub is not None:
            is_track = (hit_sub == track_hit_type)
            hit_weight = torch.where(
                is_track,
                torch.full_like(hit_sub, 0, dtype=torch.float32) + float(track_loss_weight),
                torch.full_like(hit_sub, 0, dtype=torch.float32) + 1.0,
            )

    # Energy weighting — multiplied into hit_weight so it composes with the
    # track upweighting above (tracker hits, whose e_hit ≈ 0, keep their
    # ~track_loss_weight factor; energetic calo hits get an extra boost).
    if energy_weight_mode and energy_weight_mode != "none":
        hit_e = targets.get("hit_e")
        if hit_e is not None:
            e = hit_e.clamp(min=0).to(torch.float32)
            if energy_weight_mode == "log":
                ew = 1.0 + torch.log1p(e)
            elif energy_weight_mode == "linear":
                ew = 1.0 + e / float(energy_weight_scale)
            else:
                raise ValueError(
                    "energy_weight_mode must be 'none' / 'log' / 'linear', "
                    f"got {energy_weight_mode!r}"
                )
            hit_weight = ew if hit_weight is None else hit_weight * ew

    # Capacity guard: the model can only emit N_q clusters. If a batch has
    # more real particles than queries, we have no choice but to drop the
    # excess for this batch (matched_mask vs. gt_mask would otherwise be
    # shape-mismatched). Warn once per process so this is visible — the
    # right long-term fix is `--network-option num_queries <larger>`.
    if N_p > N_q:
        if not getattr(mask3d_loss, "_warned_truncate", False):
            print(
                f"[mask3d_loss] WARNING: batch has {N_p} particles but "
                f"num_queries={N_q}; truncating GT to {N_q}. Increase "
                f"--network-option num_queries to keep all particles."
            )
            mask3d_loss._warned_truncate = True
        gt_mask = gt_mask[:, :N_q]
        gt_valid = gt_valid[:, :N_q]
        N_p = N_q

    # ---- per-layer costs, stacked ----
    if mask_loss_type == "focal":
        def _mask_pairwise_cost(logits, tgt, kv, hit_weight=None):
            return _focal_pairwise(
                logits, tgt, kv, gamma=focal_gamma, hit_weight=hit_weight,
            )
    elif mask_loss_type == "bce":
        _mask_pairwise_cost = _bce_pairwise
    else:
        raise ValueError(
            f"mask_loss_type must be 'focal' or 'bce', got {mask_loss_type!r}"
        )

    # Deep supervision: when `aux_layer_weight == 0`, only the final layer
    # is matched + lossed — matches hepattn `has_intermediate_loss: False`
    # and saves the whole aux-layer cost-matrix + matching pass.
    deep_supervision = aux_layer_weight != 0.0
    layers_for_loss = (
        list(range(len(per_layer_outs))) if deep_supervision
        else [len(per_layer_outs) - 1]
    )

    # Per-subsystem key masks: a list of (B, N_k) bool tensors, one per
    # subsystem present in the batch. With per_subsystem_loss=False the list
    # has a single entry equal to key_valid (so the existing single-pass
    # math is unchanged).
    if per_subsystem_loss:
        if "hit_subsystem" not in targets:
            raise ValueError(
                "per_subsystem_loss=True requires targets['hit_subsystem'] "
                "(rebuild targets with the updated build_targets)."
            )
        hit_subsystem = targets["hit_subsystem"]                  # (B, N_k)
        # Subsystem ids that actually appear in this batch's real hits.
        present = torch.unique(hit_subsystem[key_valid]).tolist()
        # Skip "below-offset" subsystems (e.g. hit_type=0 noise hits when
        # subsystem_offset=1) — they're not a real detector subsystem and
        # don't have a mask head; including them in the per-subsystem loss
        # would just punish the model for predicting the right thing
        # (low logits everywhere on noise hits).
        present = [s for s in present if s >= subsystem_offset]
        sub_kvs = [
            key_valid & (hit_subsystem == s) for s in present
        ]
        if not sub_kvs:
            # Fallback: no real-detector hits in this batch — use the global
            # mask. Shouldn't happen in practice.
            sub_kvs = [key_valid]
    else:
        sub_kvs = [key_valid]
    n_subs = len(sub_kvs)

    # Defensive logit clamp: bf16/fp16 + bidirectional cross-attn can spike
    # logits to ±inf on rare batches before grad-clip catches the gradient.
    # Clamping at ±30 keeps softplus / sigmoid in a numerically safe regime
    # (sigmoid(30) ≈ 1 - 1e-13, softplus(30) ≈ 30), no expressive power lost
    # since the cross-entropy gradient saturates well before that point.
    _LOGIT_CLAMP = 30.0
    for outs in per_layer_outs:
        outs["mask_logits"] = outs["mask_logits"].clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        outs["cls_logits"] = outs["cls_logits"].clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)

    with torch.no_grad():
        layer_costs = []
        for li in layers_for_loss:
            outs = per_layer_outs[li]
            # Mask-side cost is averaged over per-subsystem dice (and optionally
            # mask classification). With n_subs=1 (the default), this collapses
            # to the original single-pass math.
            mask_cost_acc = None
            for sub_kv in sub_kvs:
                term = w["mask_dice"] * _dice_pairwise(
                    outs["mask_logits"], gt_mask, sub_kv, hit_weight=hit_weight
                )
                if mask_cost_use_classification:
                    term = term + w["mask_bce"] * _mask_pairwise_cost(
                        outs["mask_logits"], gt_mask, sub_kv, hit_weight=hit_weight
                    )
                mask_cost_acc = term if mask_cost_acc is None else mask_cost_acc + term
            # Sum across subsystems (matches hepattn's separate per-subsystem
            # tasks each contributing additively).
            cost = mask_cost_acc
            cost = cost + obj_bce_cost_weight * _cls_cost(
                outs["cls_logits"], gt_valid
            )
            # Padded GT slots are excluded by the matcher via object_valid_mask
            # (truncates rows to the first n_real), so no cost inflation needed
            # — same as hepattn.
            layer_costs.append(cost)

        stacked = torch.stack(layer_costs, dim=0)                 # (L', B, N_q, N_p)
        # Belt-and-braces: even with the logit clamp above, a degenerate
        # batch (e.g. all hits in one event have key_valid=False for some
        # subsystem) can leave a 0/0 entry in dice. Replace any non-finite
        # cost with a finite value so the matcher never sees NaN/Inf. The
        # actual scipy NaN guard lives in matcher.forward; this just keeps
        # the GPU side clean for any downstream debug.
        stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e6, neginf=-1e6)
        Lp, _, N_q, _ = stacked.shape
        stacked = stacked.reshape(Lp * B, N_q, N_p)
        stacked_gt_valid = gt_valid.unsqueeze(0).expand(Lp, -1, -1).reshape(Lp * B, N_p)

        # Single batched Hungarian — one GPU→CPU sync, one scipy loop
        # (optionally thread-parallel; see Matcher).
        pred_perm = matcher(stacked, object_valid_mask=stacked_gt_valid)  # (L'*B, N_q)
        pred_perm = pred_perm.view(Lp, B, N_q)

    # ---- per-layer permutation + losses ----
    total = 0.0
    parts = {"mask_bce": 0.0, "mask_dice": 0.0, "obj_bce": 0.0}
    n_aux = max(n_layers - 1, 1)

    for slot, li in enumerate(layers_for_loss):
        outs = per_layer_outs[li]
        perm = pred_perm[slot]                                    # (B, N_q)
        perm_mask_logits = _permute(outs["mask_logits"], perm)    # (B, N_q, N_k)
        perm_cls_logits = _permute(outs["cls_logits"], perm)      # (B, N_q, 1)

        # First N_p positions correspond to GT slots 0..N_p-1.
        matched_mask = perm_mask_logits[:, :N_p]                  # (B, N_p, N_k)

        # Per-subsystem mask losses, summed across subsystems present in the
        # batch (n_subs=1 → single global loss, identical to the no-split path).
        # Hepattn parallel: each of their 5 ObjectHitMaskTask instances
        # contributes its own loss; we mirror that by summing here.
        l_bce = 0.0
        l_dice = 0.0
        for sub_kv in sub_kvs:
            if mask_loss_type == "focal":
                l_bce_s = _mask_focal_loss(
                    matched_mask, gt_mask, sub_kv, gt_valid,
                    gamma=focal_gamma, hit_weight=hit_weight,
                )
            else:
                l_bce_s = _mask_bce_loss(
                    matched_mask, gt_mask, sub_kv, gt_valid, hit_weight=hit_weight,
                )
            l_dice_s = _mask_dice_loss(
                matched_mask, gt_mask, sub_kv, gt_valid, hit_weight=hit_weight,
                size_weighting=dice_size_weighting,
                particle_energy=targets.get("target_E"),
            )
            l_bce = l_bce + l_bce_s
            l_dice = l_dice + l_dice_s
        l_obj = _object_bce_loss(perm_cls_logits, gt_valid, null_weight=w["null_weight"])

        layer_loss = (
            w["mask_bce"] * l_bce + w["mask_dice"] * l_dice + w["obj_bce"] * l_obj
        )

        is_final = li == n_layers - 1
        wl = 1.0 if is_final else aux_layer_weight / n_aux
        total = total + wl * layer_loss

        parts["mask_bce"] = parts["mask_bce"] + l_bce.detach()
        parts["mask_dice"] = parts["mask_dice"] + l_dice.detach()
        parts["obj_bce"] = parts["obj_bce"] + l_obj.detach()

    n_reported = len(layers_for_loss)
    for k in parts:
        parts[k] = parts[k] / n_reported

    # ---- high-energy clustering observable (diagnostic, no grad) ----
    # Energy-weighted hit RECALL on the FINAL layer for matched GT particles,
    # split at 10 GeV true energy. recall = (true-hit energy captured by the
    # predicted mask) / (total true-hit energy). 1.0 = the cluster captures
    # the full shower; < 1 = under-collection (the left-shifted-mass cause).
    # Watch `Erecall_highE` ↑ across recipes to see which improves big-shower
    # clustering. Set AFTER the parts/n_reported normalisation so these
    # single-shot metrics aren't divided.
    with torch.no_grad():
        he = targets.get("hit_e")            # (B, N_k) per-hit energy (GeV)
        tE = targets.get("target_E")         # (B, N_p) true particle energy
        if he is not None and tE is not None and N_p > 0:
            perm_f = pred_perm[-1]                                  # (B, N_q)
            mlf = _permute(
                per_layer_outs[-1]["mask_logits"], perm_f
            )[:, :N_p]                                              # (B, N_p, N_k)
            kvf = key_valid.unsqueeze(1).float()                    # (B,1,N_k)
            hew = he.unsqueeze(1).to(kvf.dtype) * kvf               # energy weight
            gtf = gt_mask.float()
            predbin = (mlf.sigmoid() >= 0.5).float()                # (B,N_p,N_k)
            capt = (predbin * gtf * hew).sum(-1)                    # captured E
            tot = (gtf * hew).sum(-1).clamp(min=1e-6)               # total true E
            erec = capt / tot                                       # (B,N_p)
            vmask = gt_valid.bool()
            hiE = vmask & (tE > 10.0)
            loE = vmask & (tE <= 10.0) & (tE > 0.0)

            def _wmean(metric, sel):
                s = sel.float().sum().clamp(min=1.0)
                return ((metric * sel.float()).sum() / s).detach()

            # loss-style dice (1 - overlap; lower = better) for the same split
            p_ = mlf.sigmoid() * kvf
            t_ = gtf * kvf
            inter = (p_ * t_).sum(-1)
            d = 1.0 - (2.0 * inter + 1e-6) / (p_.sum(-1) + t_.sum(-1) + 1e-6)

            parts["Erecall_highE"] = _wmean(erec, hiE)   # ↑ better
            parts["Erecall_lowE"] = _wmean(erec, loE)
            parts["dice_highE"] = _wmean(d, hiE)         # ↓ better
            parts["n_highE"] = hiE.float().sum().detach()

            # ---- per-event energy reconstruction fraction (training proxy
            #      for the showers_df Σpred/Σreco diagnostic) -----------------
            # numerator uses the per-hit MAX over queries — the same
            # quantity the recall loss optimises and inference's
            # argmax-per-hit decides on. Two variants:
            #   `_soft` : sum_k (max_q σ_qk) · e_k           (smooth, ∈ [0,1])
            #   `_hard` : sum_k 1[max_q σ_qk ≥ 0.5] · e_k    (matches eval)
            # Denominator: Σ_k e_k for hits belonging to ANY truth particle.
            ml_final = per_layer_outs[-1]["mask_logits"]                # (B,N_q,N_k)
            sigma_max_k = ml_final.sigmoid().max(dim=1).values          # (B,N_k)
            kvf = key_valid.to(sigma_max_k.dtype)                       # (B,N_k)
            hew = he.to(sigma_max_k.dtype) * kvf                        # (B,N_k)
            any_truth = gt_mask.any(dim=1).to(sigma_max_k.dtype)        # (B,N_k)
            Sreco = (hew * any_truth).sum(dim=-1)                       # (B,)
            Spred_soft = (hew * sigma_max_k).sum(dim=-1)                # (B,)
            Spred_hard = (hew * (sigma_max_k >= 0.5).to(sigma_max_k.dtype)).sum(dim=-1)
            valid_ev = Sreco > 0
            if valid_ev.any():
                rs = (Spred_soft[valid_ev] / Sreco[valid_ev].clamp(min=1e-6))
                rh = (Spred_hard[valid_ev] / Sreco[valid_ev].clamp(min=1e-6))
                parts["evt_E_frac_soft"]     = rs.mean().detach()
                parts["evt_E_frac_hard"]     = rh.mean().detach()
                parts["evt_E_frac_hard_min"] = rh.min().detach()    # worst event in batch

    # ---- optional energy-completeness loss ----
    # Per matched (query, GT particle) pair in the FINAL layer, push the
    # SOFT predicted mask to collect the particle's full energy. Unlike
    # dice/BCE (which go flat once the cluster core is covered), this keeps
    # gradient on the under-collected diffuse halo — the part that makes
    # Σpred/Σreco fall short of Pandora's raw clustering completeness.
    if energy_completeness_weight > 0.0 and N_p > 0:
        he = targets.get("hit_e")            # (B, N_k) per-hit energy (GeV)
        tE = targets.get("target_E")         # (B, N_p) true particle energy
        if he is not None and tE is not None:
            perm_ec = pred_perm[-1]                                 # (B, N_q)
            ml_ec = _permute(
                per_layer_outs[-1]["mask_logits"], perm_ec
            )[:, :N_p]                                              # (B, N_p, N_k)
            kv = key_valid.unsqueeze(1).to(ml_ec.dtype)            # (B,1,N_k)
            hew = he.unsqueeze(1).to(ml_ec.dtype) * kv             # (B,1,N_k)
            collected = (ml_ec.sigmoid() * hew).sum(-1)            # (B, N_p)
            tEf = tE.to(collected.dtype)
            sel = gt_valid.float() * (tEf > 0).float()             # (B, N_p)
            rel = (collected - tEf).abs() / tEf.clamp(min=0.1)
            ec_loss = (rel * sel).sum() / sel.sum().clamp(min=1.0)
            total = total + energy_completeness_weight * ec_loss
            parts["energy_completeness"] = ec_loss.detach()

    # ---- optional per-truth recall loss ----
    # For each GT hit, push the MAX over queries of σ(mask) up — so SOME
    # query (Hungarian-matched or not) fires confidently on each truth's
    # hits, matching the per-hit argmax that inference actually does.
    if recall_weight > 0.0 and N_p > 0:
        he = targets.get("hit_e")
        tE = targets.get("target_E")
        if he is not None and tE is not None:
            rec_loss = _recall_loss(
                per_layer_outs[-1]["mask_logits"], gt_mask, gt_valid, key_valid,
                he, tE, tau=recall_tau, energy_weight_alpha=recall_E_alpha,
            )
            total = total + recall_weight * rec_loss
            parts["recall"] = rec_loss.detach()

    # ---- optional GMM coverage / NLL term ----
    # For each matched (query, GT particle) pair in the FINAL layer, push the
    # query's mixture density to maximise log p(x_i) for x_i in the matched
    # GT mask. Direct ML supervision on μ / Σ; no architectural constraint on
    # the mask head. Disabled when weight == 0 or attn_bias not provided.
    if (
        attn_bias is not None
        and gmm_coverage_weight > 0.0
        and N_p > 0
    ):
        # Final layer's permutation. First N_p slots are matched queries
        # aligned to GT slots 0..N_p-1.
        perm_final = pred_perm[-1]                                 # (B, N_q)
        matched_q = perm_final[:, :N_p]                            # (B, N_p)
        idx = matched_q.unsqueeze(-1).expand(B, N_p, attn_bias.size(-1))
        log_w_matched = attn_bias.gather(1, idx)                   # (B, N_p, N_k)
        # Hit must (a) be a real key, (b) belong to a valid GT slot, (c) be
        # part of the GT particle's mask.
        weight = (
            gt_mask.float()
            * key_valid.unsqueeze(1).float()
            * gt_valid.unsqueeze(-1).float()
        )                                                          # (B, N_p, N_k)
        denom = weight.sum().clamp(min=1.0)
        cov_loss = -(log_w_matched * weight).sum() / denom
        total = total + gmm_coverage_weight * cov_loss
        parts["gmm_coverage"] = cov_loss.detach()

    # ---- optional GMM containment-BCE term ----
    # Stronger supervision than `gmm_coverage` alone: treat `log p_q(x_k)` as
    # the logit for "hit k belongs to cluster matched by query q?". Apply
    # `binary_cross_entropy_with_logits` against the GT mask. Pushes density
    # UP on real cluster hits (true positives) AND DOWN on non-cluster hits
    # (false positives) so the Gaussians both move toward and shrink onto
    # the matched cluster's actual shape. Gradient per (q, p, k) is bounded
    # to [-1, 1] (sigmoid(logit) − target), so this is safer than the
    # unclamped NLL.
    if (
        attn_bias is not None
        and gmm_bce_weight > 0.0
        and N_p > 0
    ):
        perm_final = pred_perm[-1]                                 # (B, N_q)
        matched_q = perm_final[:, :N_p]                            # (B, N_p)
        idx = matched_q.unsqueeze(-1).expand(B, N_p, attn_bias.size(-1))
        log_w_matched = attn_bias.gather(1, idx)                   # (B, N_p, N_k)
        target = gt_mask.float()                                   # (B, N_p, N_k)
        # Per-(q, p, k) BCE-with-logits — numerically stable for any log_w.
        bce = F.binary_cross_entropy_with_logits(
            log_w_matched, target, reduction="none",
        )                                                          # (B, N_p, N_k)
        # Mask to real keys × valid GT slots. NO `gt_mask` factor here —
        # we want both positive (in-mask) and negative (out-of-mask) hits
        # contributing to the BCE.
        weight = (
            key_valid.unsqueeze(1).float()
            * gt_valid.unsqueeze(-1).float()
        )                                                          # (B, N_p, N_k)
        denom = weight.sum().clamp(min=1.0)
        bce_loss = (bce * weight).sum() / denom
        total = total + gmm_bce_weight * bce_loss
        parts["gmm_bce"] = bce_loss.detach()

    # ---- FAPE-style frame supervision (IPA models) ----
    # For each matched (query, GT particle) in the FINAL layer where the
    # particle is supervisable (charged, E ≥ 1 GeV):
    #   L_t = ‖t_i − centroid_p‖²                (translation)
    #   L_R = 1 − ⟨T_i · ẑ, axis_p⟩               (axis alignment, ẑ = [0,0,1])
    # The rotation around the particle axis is NOT supervised — there is
    # no physical "x-axis" for a particle, so we leave that DoF free.
    supervisable = targets.get("particle_supervisable")
    particle_axis = targets.get("particle_axis")
    if (
        T_final is not None
        and t_final is not None
        and pos_padded is not None
        and supervisable is not None
        and particle_axis is not None
        and (frame_translation_weight > 0 or frame_rotation_weight > 0)
        and N_p > 0
    ):
        perm_final = pred_perm[-1]                                         # (B, N_q)
        matched_q = perm_final[:, :N_p]                                    # (B, N_p)
        # Gather matched-query frames.
        idx_T = matched_q.unsqueeze(-1).unsqueeze(-1).expand(B, N_p, 3, 3)
        T_m = T_final.gather(1, idx_T)                                      # (B, N_p, 3, 3)
        idx_t = matched_q.unsqueeze(-1).expand(B, N_p, 3)
        t_m = t_final.gather(1, idx_t)                                      # (B, N_p, 3)
        # Per-particle shower anchor. PREFER `targets["particle_ref"]` —
        # for tracked particles this is `referencePoint_calo` (the track's
        # position at the calo entry), which is the physically right anchor
        # for the IPA frame (rotation axis already points along the track
        # direction at the calo, so anchor + axis form a coherent ray
        # starting at the shower entry). Fall back to the hit centroid for
        # backward-compat when particle_ref isn't provided.
        gt_f = gt_mask.float()
        has_hits = gt_f.sum(-1) > 0                                         # (B, N_p)
        ref_target = targets.get("particle_ref")
        if ref_target is None:
            denom = gt_f.sum(-1, keepdim=True).clamp(min=1.0)               # (B, N_p, 1)
            ref_target = (
                gt_f.unsqueeze(-1) * pos_padded.unsqueeze(1)
            ).sum(dim=2) / denom                                            # (B, N_p, 3)
        ref_target = ref_target.to(t_m.dtype)
        # Supervisable particles that also have ≥1 hit in the GT mask.
        w = (supervisable & has_hits).float()                               # (B, N_p)
        denom_w = w.sum().clamp(min=1.0)

        # Translation loss (in metres²).
        L_t = (((t_m - ref_target).pow(2).sum(-1)) * w).sum() / denom_w
        # Rotation loss: 1 − cos(angle). T_m's third column = T_m·ẑ.
        T_z = T_m[..., 2]                                                   # (B, N_p, 3)
        axis = particle_axis.to(T_z.dtype)                                  # (B, N_p, 3)
        cos_align = (T_z * axis).sum(-1)                                    # (B, N_p)
        L_R = ((1.0 - cos_align) * w).sum() / denom_w

        frame_loss = (
            frame_translation_weight * L_t + frame_rotation_weight * L_R
        )
        total = total + frame_loss
        parts["frame_translation"] = L_t.detach()
        parts["frame_rotation"] = L_R.detach()

    return total, parts
