"""Task heads: per-(query, hit) mask logits and per-query validity logits.

Mirrors hepattn's ObjectHitMaskTask + ObjectClassificationTask. Both
operate on padded query (B, N_q, D) and key (B, N_k, D) tensors.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, dim, out_dim=None, hidden=None):
        super().__init__()
        out_dim = out_dim or dim
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ObjectHitMaskTask(nn.Module):
    """Predicts a per-(query, hit) mask logit via a bilinear einsum.

        logit[b, n, m] = scale * <object_net(q[b, n]), constituent_net_s(k[b, m])>

    With `num_subsystems > 1`, the constituent (key) projection is a
    `ModuleList` of `num_subsystems` MLPs — one per detector subsystem —
    and each hit's projection is taken from its subsystem's MLP. The
    object (query) projection is shared across subsystems because queries
    are global.

    This mirrors hepattn's CLD config which has 5 separate
    `ObjectHitMaskTask` instances (vtxd / trkr / ecal / hcal / muon),
    each with its own `constituent_net` against shared queries.
    """

    def __init__(self, dim, num_subsystems=1, subsystem_offset=0,
                 cosine_logits=True, init_logit_scale=10.0,
                 max_logit_scale=30.0):
        super().__init__()
        self.num_subsystems = num_subsystems
        # `subsystem_offset` shifts hit_type → head-index mapping. For CLD
        # data (`hit_type ∈ {0=noise, 1=tracker, 2=ecal, 3=hcal, 4=muon}`)
        # use `num_subsystems=4, subsystem_offset=1` so heads cover only the
        # 4 real detectors and noise hits (hit_type=0) get clamped down to
        # head 0.
        self.subsystem_offset = subsystem_offset
        self.object_net = _MLP(dim)
        if num_subsystems == 1:
            self.constituent_net = _MLP(dim)
        else:
            self.constituent_net = nn.ModuleList(
                [_MLP(dim) for _ in range(num_subsystems)]
            )

        # ----------------------------------------------------------------
        # Logit-scale design: the mask head is `logit = scale * <o, x>`. The
        # original `scale = 1/sqrt(dim)` paired with un-normalized MLP outputs
        # produced dot products O(±100) → logits O(±6) → sigmoid saturation
        # and a vanishing gradient (run vhc0nms5 plateaued at val_dice ≈ 0.45
        # with this exact failure mode: crisp masks, wrong locations).
        #
        # `cosine_logits=True` (new default): L2-normalize both projections
        # and treat `scale` as a temperature in log-space, CLIP-style. With
        # init temperature 10, logits sit in [-10, +10], sigmoid stays in its
        # well-conditioned regime, gradients are healthy.
        # `max_logit_scale` clamps `log_scale` so the optimizer can't drive
        # logits to ±∞ during training and re-introduce saturation.
        # `cosine_logits=False` reproduces the legacy un-normalized path
        # exactly (for ablation / loading old checkpoints).
        # ----------------------------------------------------------------
        self.cosine_logits = cosine_logits
        if cosine_logits:
            # log-space scale ⇒ scale = exp(log_scale). Init to log(10).
            self.log_scale = nn.Parameter(
                torch.tensor(math.log(float(init_logit_scale)))
            )
            self.max_log_scale = math.log(float(max_logit_scale))
        else:
            self.scale = nn.Parameter(torch.tensor(1.0 / dim ** 0.5))

    def _einsum(self, o, x):
        if self.cosine_logits:
            o = F.normalize(o, dim=-1)
            x = F.normalize(x, dim=-1)
            scale = self.log_scale.clamp(max=self.max_log_scale).exp()
            return scale * torch.einsum("bnc,bmc->bnm", o, x)
        return self.scale * torch.einsum("bnc,bmc->bnm", o, x)

    def forward(self, q, k, hit_subsystem=None):
        o = self.object_net(q)                           # (B, N_q, D)

        if self.num_subsystems == 1:
            x = self.constituent_net(k)                  # (B, N_k, D)
            return self._einsum(o, x)

        if hit_subsystem is None:
            raise ValueError(
                "ObjectHitMaskTask(num_subsystems > 1) requires `hit_subsystem` "
                "(per-key int subsystem id, shape (B, N_k))."
            )

        B, N_k, D = k.shape
        x_all = torch.stack(
            [self.constituent_net[s](k) for s in range(self.num_subsystems)],
            dim=2,
        )                                                # (B, N_k, S, D)
        sub_idx = (hit_subsystem - self.subsystem_offset).clamp(
            0, self.num_subsystems - 1
        )
        idx = sub_idx.unsqueeze(-1).unsqueeze(-1).expand(B, N_k, 1, D)
        x = x_all.gather(2, idx).squeeze(2)              # (B, N_k, D)
        return self._einsum(o, x)


class ObjectClassificationTask(nn.Module):
    """Per-query validity classification (binary or multi-class)."""

    def __init__(self, dim, num_classes=1):
        super().__init__()
        self.net = _MLP(dim, out_dim=num_classes)

    def forward(self, q):
        return self.net(q)                               # (B, N_q, C)
