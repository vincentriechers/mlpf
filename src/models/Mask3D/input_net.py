"""Per-hit feature embedding (single unified InputNet).

Operates on flat `(total_hits, F)` tensors — the encoder runs on a flat
token stream with BlockDiagonalMask, so there is no padding to mask out.
"""
import math
import torch
import torch.nn as nn


class FourierPosEnc(nn.Module):
    """Random Fourier features on 3-D positions."""

    def __init__(self, dim, num_freqs=32, scale=1.0):
        super().__init__()
        proj = torch.randn(3, num_freqs) * scale
        self.register_buffer("proj", proj)
        self.proj_out = nn.Linear(2 * num_freqs, dim)

    def forward(self, xyz):  # (N, 3)
        ang = 2 * math.pi * (xyz @ self.proj)            # (N, F)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.proj_out(feats)


class InputNet(nn.Module):
    def __init__(self, in_dim, dim, use_fourier=True, num_freqs=32, fourier_scale=1.0):
        # `fourier_scale=1.0` matches hepattn's `FourierPositionEncoder` default
        # and pairs with the m-scale positions produced by `build_targets`
        # (which divides mm coords by 1000). Effective lowest-frequency
        # wavelength is ≈ 1 m — appropriate for detector geometry. The
        # previous default 0.05 was tuned for raw mm coordinates, which gave
        # wavelengths of ~2 cm and a near-noise PE.
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.use_fourier = use_fourier
        if use_fourier:
            self.pe = FourierPosEnc(dim, num_freqs=num_freqs, scale=fourier_scale)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feats_flat):
        # feats_flat: (total_hits, F). First 3 columns are (x, y, z).
        h = self.norm_in(feats_flat)
        x = self.mlp(h)
        if self.use_fourier:
            x = x + self.pe(feats_flat[..., :3])
        return self.out_norm(x)


# ----------------------------------------------------------------------
# Per-subsystem InputNet (hepattn-style)
# ----------------------------------------------------------------------

# Per-detector field set. We start from the standard `feats_flat` layout
# `(pos(3), ht_oh(n_oh), e, p, chi²)` and DERIVE additional geometric /
# log-energy fields per hit. Each subsystem then takes a detector-specific
# subset:
#
#   tracker (hit_type=1):  x, y, z, r, θ, φ, p, χ²        → 8 fields
#   ecal    (hit_type=2):  x, y, z, r, θ, φ, log(E+1)     → 7 fields
#   hcal    (hit_type=3):  x, y, z, r, θ, φ, log(E+1)     → 7 fields
#   muon    (hit_type=4):  x, y, z, r, θ, φ               → 6 fields
#
# This mirrors hepattn CLD `base.yaml`'s 5 separate InputNets — except
# hepattn has tracker `u/v` measurements (`u.a, u.b, v.a, v.b, du, dv`)
# which our data pipeline doesn't expose, so the tracker net here is
# 8 fields (with `p` and `χ²`) instead of hepattn's 13.
_FIELD_DIMS = {
    1: 8,    # tracker
    2: 7,    # ecal
    3: 7,    # hcal
    4: 6,    # muon
}


def _derive_geom(xyz):
    """xyz: (N, 3) → (r, θ, φ) (N, 3)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r2 = x * x + y * y
    r = torch.sqrt(r2 + 1e-12)
    theta = torch.atan2(r, z)
    phi = torch.atan2(y, x)
    return torch.stack([r, theta, phi], dim=-1)


class PerSubsystemInputNet(nn.Module):
    """Hepattn-style per-detector InputNet.

    `num_subsystems` separate `InputNet`s — one each for tracker, ecal,
    hcal, muon (with `subsystem_offset=1` to skip noise / hit_type=0).
    Each subsystem reads a detector-specific subset of features that we
    derive on the fly from the standard `feats_flat`:

      - geometry: `(r, θ, φ)` from `(x, y, z)`
      - energy:   `log(e_hit + 1)` for calorimeter hits

    Hits of `hit_type < subsystem_offset` (i.e. noise) are routed to the
    first subsystem's net (`tracker`). Their `gt_mask` is all-zero so the
    loss naturally trains them to have low logits — same treatment as in
    `ObjectHitMaskTask`.
    """

    def __init__(
        self,
        dim,
        n_oh,
        num_subsystems=4,
        subsystem_offset=1,
        use_fourier=True,
        num_freqs=32,
        fourier_scale=1.0,                       # see note in InputNet
    ):
        super().__init__()
        if num_subsystems != 4:
            raise NotImplementedError(
                "PerSubsystemInputNet currently hardcodes the CLD layout: "
                "4 subsystems = tracker/ecal/hcal/muon. Add field configs "
                "to _FIELD_DIMS / _build_subsystem_feats for other setups."
            )
        self.dim = dim
        self.num_subsystems = num_subsystems
        self.subsystem_offset = subsystem_offset

        # Indices into the standard feats_flat layout
        # `(pos(3), ht_oh(n_oh), e(1), p(1), chi²(1))`. ht_oh occupies
        # cols [3, 3+n_oh); e/p/χ² immediately follow.
        base = 3 + n_oh
        self._idx_e = base + 0
        self._idx_p = base + 1
        self._idx_chi = base + 2

        # One InputNet per subsystem (in head order: subsystem_offset → 0,
        # +1 → 1, …). Tracker = first net.
        self.input_nets = nn.ModuleList([
            InputNet(
                in_dim=_FIELD_DIMS[subsystem_offset + s],
                dim=dim,
                use_fourier=use_fourier,
                num_freqs=num_freqs,
                fourier_scale=fourier_scale,
            )
            for s in range(num_subsystems)
        ])

    def _build_subsystem_feats(self, sub_feats_flat, ht_value):
        """sub_feats_flat is already masked to a single subsystem's hits.
        Returns the detector-specific feature tensor for that subsystem.
        """
        xyz = sub_feats_flat[:, :3]
        rtp = _derive_geom(xyz)
        e = sub_feats_flat[:, self._idx_e:self._idx_e + 1]
        p = sub_feats_flat[:, self._idx_p:self._idx_p + 1]
        chi = sub_feats_flat[:, self._idx_chi:self._idx_chi + 1]
        log_e = torch.log1p(e.clamp(min=0))

        if ht_value == 1:                                         # tracker
            return torch.cat([xyz, rtp, p, chi], dim=-1)
        if ht_value in (2, 3):                                    # ecal / hcal
            return torch.cat([xyz, rtp, log_e], dim=-1)
        if ht_value == 4:                                         # muon
            return torch.cat([xyz, rtp], dim=-1)
        # Should not reach here — caller filters by hit_type.
        raise ValueError(f"Unknown hit_type for per-subsystem features: {ht_value}")

    def forward(self, feats_flat, hit_type):
        out = feats_flat.new_zeros(feats_flat.size(0), self.dim)

        for s in range(self.num_subsystems):
            ht = s + self.subsystem_offset
            mask = hit_type == ht
            if not mask.any():
                continue
            sub_feats = self._build_subsystem_feats(feats_flat[mask], ht)
            out[mask] = self.input_nets[s](sub_feats)

        # Noise hits (hit_type < subsystem_offset) → first subsystem's net.
        # Use that net's expected field set (tracker fields when offset=1).
        if self.subsystem_offset > 0:
            noise_mask = hit_type < self.subsystem_offset
            if noise_mask.any():
                sub_feats = self._build_subsystem_feats(
                    feats_flat[noise_mask], self.subsystem_offset
                )
                out[noise_mask] = self.input_nets[0](sub_feats)

        return out
