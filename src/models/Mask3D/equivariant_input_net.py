"""SO(3) / SO(2)_z-invariant input embedding for the equivariant Attn-IPA
backbone.

Difference vs `input_net.InputNet`:

* Position `x = (x, y, z)` is NOT consumed as a scalar feature. The vanilla
  `InputNet` did `MLP(LayerNorm(concat(x, hit_type_oh, E, p, chi2)))` + a
  Fourier positional encoding on the absolute `(x, y, z)` — both break
  SO(3) and SO(2)_z invariance because they treat the world-frame
  coordinates as scalar inputs to a learned linear map.

* This module:
    1. Drops `x` from the MLP input.
    2. Builds **cylindrical-invariant scalars** `r = sqrt(x² + y²)` and
       `|z|` from `x`. These are exact invariants of SO(2)_z + the
       z-reflection that CLD's geometry actually exhibits. For full SO(3)
       use `symmetry="SO(3)"` which substitutes a single `|x|` instead.
    3. Adds a `log(1 + E)` channel so the dynamic range of per-hit energy
       doesn't get clobbered by the LayerNorm + linear (raw E goes from
       0.01 GeV to ~50 GeV per hit; `log1p` brings that to a sane scale).
    4. Optional Fourier positional encoding is computed on the invariants
       (`r, |z|` or `|x|`), NOT on `(x, y, z)`.

The downstream contract is unchanged: the backbone still returns
`(feats, raw_xyz)`; the raw positions flow through to the IPA decoder
unchanged so all geometric reasoning continues to live in the decoder's
(T, t) frames. Stripping `x` out of the BACKBONE features is what makes
the encoder's output an SO(3)/SO(2)_z scalar invariant — which combined
with the equivariant IPA decoder yields end-to-end equivariance.
"""
import math
import torch
import torch.nn as nn


class LocalGeomAttentionBlock(nn.Module):
    """One transformer-style block of per-token local cross-attention.

    Query: the token's own invariant feature (from the previous layer).
    Keys / Values: per-neighbour encoding of the precomputed SO(*)-
    invariant `nb_inv` tensor (`Δx` in a local frame + `|Δx|`).

    Structure (pre-norm transformer):
        f' = f + MHA(LN(f), enc(nb_inv))
        f  = f' + FFN(LN(f'))

    Stacking N blocks lets information propagate N hops along the k-NN
    graph: block-`i` attends to neighbours whose features already carry
    info aggregated by block-`(i-1)`. The k-NN structure itself is fixed
    by the positions and computed ONCE upstream (in `LocalGeometryHead`).
    """

    def __init__(self, dim, num_heads=4, head_dim=24, ffn_mult=2,
                 dropout=0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim  = int(head_dim)
        inner = self.num_heads * self.head_dim

        self.norm_q   = nn.LayerNorm(dim)
        # Per-neighbour invariant encoder. Each block re-encodes the same
        # geometric invariants in its own basis so subsequent blocks see
        # geometry through a different lens.
        self.nb_enc = nn.Sequential(
            nn.Linear(4, inner),
            nn.GELU(),
            nn.Linear(inner, inner),
        )
        self.q_proj   = nn.Linear(dim,   inner, bias=False)
        self.k_proj   = nn.Linear(inner, inner, bias=False)
        self.v_proj   = nn.Linear(inner, inner, bias=False)
        self.out_proj = nn.Linear(inner, dim,   bias=True)
        self.scale = self.head_dim ** -0.5

        # Per-token FFN — gives the block a node-level non-linearity in
        # the same spirit as a Mask2Former transformer layer.
        self.norm_f = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, f_flat, nb_inv):
        """
        f_flat : (N, dim)     current per-token invariant feature.
        nb_inv : (N, k, 4)    invariant per-neighbour features (Δx_local,|Δx|).
        Returns: (N, dim)     updated per-token feature.
        """
        N, k, _ = nb_inv.shape
        H, D = self.num_heads, self.head_dim
        # ---- Local cross-attention --------------------------------------
        nb_emb = self.nb_enc(nb_inv)                                # (N, k, inner)
        qn = self.norm_q(f_flat)
        Q = self.q_proj(qn   ).view(N, 1, H, D)
        K = self.k_proj(nb_emb).view(N, k, H, D)
        V = self.v_proj(nb_emb).view(N, k, H, D)
        scores = torch.einsum("nqhd,nkhd->nhqk", Q, K) * self.scale  # (N, H, 1, k)
        attn   = scores.softmax(dim=-1)
        attended = torch.einsum("nhqk,nkhd->nqhd", attn, V).reshape(N, H * D)
        f_flat = f_flat + self.out_proj(attended)
        # ---- FFN --------------------------------------------------------
        f_flat = f_flat + self.ffn(self.norm_f(f_flat))
        return f_flat


class LocalGeometryHead(nn.Module):
    """Per-token local-geometry encoder.

    Computes per-token SO(*)-invariant features of the k-NN
    neighbourhood ONCE (KNN, local frame, displacement projections), then
    applies a stack of `num_blocks` `LocalGeomAttentionBlock`s on top.

    `num_blocks > 1` gives a k-hop receptive field over the k-NN graph
    (info from neighbours-of-neighbours, etc.), in the same way stacked
    GNN layers expand the receptive field. Each block's K/V comes from
    re-encoding the same `nb_inv` tensor in that block's own learned
    basis, so subsequent blocks can specialise (e.g. early blocks
    discover "I'm at a shower edge", later blocks aggregate higher-level
    structure).

    For each hit `j`:
      1. Find `k` nearest neighbours within the same event.
      2. Build a per-token local frame:
         - "SO(2)_z" (default): cylindrical at j → `(r̂_j, ẑ, φ̂_j)`.
         - "SO(3)": PCA frame of local k-NN displacements, sign-pinned by
           orientation toward the local centroid (rotation-stable).
      3. Project each displacement `Δx_{jk} = x_k − x_j` onto the frame
         → 3 invariants + `|Δx_{jk}|` = `nb_inv ∈ R⁴` per neighbour.
      4. Stack `num_blocks` of `LocalGeomAttentionBlock` over (f, nb_inv).

    The KNN is computed per-event with `torch.cdist + topk`. For CLD-scale
    events (~10 k hits) this is one 400 MB transient per event, fine on
    A100; for larger problems swap in `torch_cluster.knn`.
    """

    def __init__(self, dim, k=64, num_heads=4, head_dim=24,
                 num_blocks=2, ffn_mult=2, symmetry="SO(2)_z"):
        super().__init__()
        assert symmetry in ("SO(2)_z", "SO(3)"), symmetry
        self.k = int(k)
        self.symmetry = symmetry
        self.num_blocks = int(num_blocks)
        self.blocks = nn.ModuleList([
            LocalGeomAttentionBlock(
                dim=dim, num_heads=num_heads, head_dim=head_dim,
                ffn_mult=ffn_mult,
            )
            for _ in range(self.num_blocks)
        ])

    # ----- local frame ------------------------------------------------------
    def _cylindrical_frame(self, x):
        """(N, 3) → (N, 3, 3) with columns (r̂, ẑ, φ̂)."""
        r2 = x[:, 0] ** 2 + x[:, 1] ** 2
        r = r2.clamp(min=1e-12).sqrt()
        zero = torch.zeros_like(r)
        one = torch.ones_like(r)
        r_hat   = torch.stack([x[:, 0] / r,  x[:, 1] / r, zero], dim=-1)
        z_hat   = torch.stack([zero,         zero,         one ], dim=-1)
        phi_hat = torch.stack([-x[:, 1] / r, x[:, 0] / r, zero], dim=-1)
        return torch.stack([r_hat, z_hat, phi_hat], dim=-1)            # (N, 3, 3) cols

    def _so3_pca_frame(self, x_flat, knn_idx):
        """Per-token SO(3)-equivariant frame from a local PCA.

        For each token j: take the k-NN displacements (N, k, 3), build the
        (3, 3) covariance, eigendecompose, use eigenvectors as columns of
        R_j. Under a global rotation R, the covariance transforms as
        R Σ Rᵀ, whose eigenvectors are R times the originals → frame
        rotates equivariantly, projections are invariant.

        Sign convention. `torch.linalg.eigh` returns eigenvectors with an
        arbitrary sign that can flip under rotation. We fix the signs by
        orienting each eigenvector to lie in the same hemisphere as the
        local centroid offset `c_j = (1/k) Σ_k (x_k − x_j)` (which itself
        rotates equivariantly under R). `e_c · c_j` is invariant, so its
        sign is rotation-stable. For the rare case `e_c · c_j ≈ 0` we
        fall back to a sum of inner products with the displacements (also
        equivariant).
        """
        nb_pos = x_flat[knn_idx]                                       # (N, k, 3)
        d = nb_pos - x_flat.unsqueeze(1)                                # (N, k, 3)
        cov = torch.einsum("nki,nkj->nij", d, d) / max(self.k, 1)       # (N, 3, 3)
        cov_f = cov.float() + 1e-6 * torch.eye(3, device=cov.device)
        _, evecs = torch.linalg.eigh(cov_f)                             # ascending
        evecs = evecs.flip(-1).to(x_flat.dtype)                         # (N, 3, 3) cols

        # Rotation-stable sign: orient each eigenvector toward the local
        # centroid offset. Fall back to a deterministic second signal if
        # the centroid is (numerically) perpendicular.
        c = d.mean(dim=1)                                               # (N, 3)
        # Σ_k |d_k · e_c| · sign(d_k · e_c) — same as Σ d·e, but explicit:
        proj_c = torch.einsum("nij,nj->ni",
                              evecs.transpose(-1, -2), c)               # (N, 3)
        proj_d = torch.einsum("nij,nkj->nki",
                              evecs.transpose(-1, -2), d).sum(dim=1)    # (N, 3)
        primary = proj_c
        fallback = proj_d
        sign = torch.where(
            primary.abs() > 1e-6,
            primary.sign(),
            fallback.sign().clamp(min=-1.0).where(
                fallback.sign() != 0, torch.ones_like(fallback)
            ),
        )
        sign = sign.unsqueeze(1)                                        # (N, 1, 3)
        return evecs * sign                                              # (N, 3, 3)

    # ----- KNN -------------------------------------------------------------
    @staticmethod
    def _knn_per_event(x_flat, seq_lens, k):
        """Per-event KNN within the flat (total, 3) layout. Returns (N, k)
        absolute indices into `x_flat`. Self is excluded; padded with the
        token's own index when the event has < k neighbours."""
        N = x_flat.shape[0]
        device = x_flat.device
        out = torch.zeros(N, k, dtype=torch.long, device=device)
        offset = 0
        for L in seq_lens:
            L = int(L)
            if L <= 0:
                continue
            x_ev = x_flat[offset:offset + L]
            d2 = torch.cdist(x_ev, x_ev, p=2)                           # (L, L)
            d2.fill_diagonal_(float("inf"))
            k_eff = min(k, max(L - 1, 1))
            idx = d2.topk(k_eff, largest=False).indices                 # (L, k_eff)
            if k_eff < k:
                self_idx = torch.arange(L, device=device).unsqueeze(-1)\
                    .expand(L, k - k_eff)
                idx = torch.cat([idx, self_idx], dim=-1)
            out[offset:offset + L] = idx + offset
            offset += L
        return out

    # ----- forward ---------------------------------------------------------
    def forward(self, x_flat, f_flat, seq_lens):
        """Stack of `num_blocks` local cross-attention layers.

        Args:
            x_flat   : (N, 3)   world-frame hit positions.
            f_flat   : (N, dim) per-token invariant features (from
                                EquivariantInputNet). Used as the query
                                for the FIRST block; subsequent blocks
                                use the previous block's output.
            seq_lens : list[int] per-event hit counts.

        Returns: (N, dim) — final per-token feature (this is the OUTPUT
        of the local stack, NOT a residual — the caller decides whether
        to add it residually).
        """
        # --- precompute invariant per-neighbour features ONCE ----------
        knn_idx = self._knn_per_event(x_flat, seq_lens, self.k)        # (N, k)
        disp    = x_flat[knn_idx] - x_flat.unsqueeze(1)                # (N, k, 3)
        if self.symmetry == "SO(2)_z":
            frame = self._cylindrical_frame(x_flat)                    # (N, 3, 3)
        else:
            frame = self._so3_pca_frame(x_flat, knn_idx)               # (N, 3, 3)
        disp_local = torch.einsum("nij,nkj->nki",
                                  frame.transpose(-1, -2), disp)       # (N, k, 3)
        dist = disp.norm(dim=-1, keepdim=True)                         # (N, k, 1)
        nb_inv = torch.cat([disp_local, dist], dim=-1)                 # (N, k, 4)

        # --- stacked local cross-attention -----------------------------
        f = f_flat
        for block in self.blocks:
            f = block(f, nb_inv)
        return f


class FourierInvariantPosEnc(nn.Module):
    """Random Fourier features on a small set of input INVARIANTS (not on
    `(x, y, z)`). Same shape as `FourierPosEnc` but the input axis is
    `n_inv` rather than 3."""

    def __init__(self, dim, n_inv, num_freqs=32, scale=1.0):
        super().__init__()
        proj = torch.randn(n_inv, num_freqs) * scale
        self.register_buffer("proj", proj)
        self.proj_out = nn.Linear(2 * num_freqs, dim)
        self.n_inv = n_inv

    def forward(self, inv):  # (N, n_inv)
        ang = 2 * math.pi * (inv @ self.proj)            # (N, F)
        feats = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.proj_out(feats)


class EquivariantInputNet(nn.Module):
    """Drop-in equivariant replacement for `input_net.InputNet`.

    Input contract (same `feats_flat` layout as the rest of the stack):
        feats_flat: (total_hits, 3 + n_oh + 3)
            cols 0..2  : world-frame position (x, y, z) in metres
            cols 3..3+n_oh-1 : hit_type one-hot
            cols last 3 : (E, p, chi2)

    Output: (total_hits, dim) per-token embedding that is INVARIANT under
    the requested symmetry (SO(2)_z by default, or SO(3) if requested).

    `symmetry`:
      - "SO(2)_z" (default): use cylindrical invariants `(r, |z|)`.
      - "SO(3)"          : use radial invariant `|x|` only.

    Energy is fed in twice: raw `E` (preserves units the rest of the loss
    cares about) AND `log1p(E)` (so the LayerNorm sees a reasonably-scaled
    quantity). Both are SO(*)-invariant scalars.
    """

    def __init__(
        self,
        n_oh,
        dim,
        symmetry="SO(2)_z",
        use_fourier=True,
        num_freqs=32,
        fourier_scale=1.0,
    ):
        super().__init__()
        assert symmetry in ("SO(2)_z", "SO(3)"), symmetry
        self.symmetry = symmetry
        self.n_oh = int(n_oh)

        # Scalar invariants we hand the MLP, in order:
        #   r or |x|    1
        #   |z|         1 (only for SO(2)_z)
        #   hit_type    n_oh
        #   E           1
        #   log1p(E)    1
        #   p           1
        #   chi2        1
        n_geom_inv = 2 if symmetry == "SO(2)_z" else 1
        in_dim = n_geom_inv + n_oh + 4
        self.in_dim = in_dim

        self.norm_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.use_fourier = use_fourier
        if use_fourier:
            self.pe = FourierInvariantPosEnc(
                dim, n_inv=n_geom_inv, num_freqs=num_freqs,
                scale=fourier_scale,
            )
        self.out_norm = nn.LayerNorm(dim)

    # ----- helpers ----------------------------------------------------------
    def _invariants(self, xyz):
        """xyz: (N, 3) → (N, n_geom_inv) cylindrical or radial invariants."""
        if self.symmetry == "SO(2)_z":
            r = (xyz[..., 0] ** 2 + xyz[..., 1] ** 2).clamp(min=1e-12).sqrt()
            absz = xyz[..., 2].abs()
            return torch.stack([r, absz], dim=-1)
        else:  # SO(3)
            r = (xyz ** 2).sum(-1).clamp(min=1e-12).sqrt()
            return r.unsqueeze(-1)

    # ----- forward ----------------------------------------------------------
    def forward(self, feats_flat):
        """
        feats_flat: (N, 3 + n_oh + 3) — same layout as the vanilla net.
        Returns: (N, dim) per-token features.
        """
        xyz = feats_flat[..., :3]
        ht  = feats_flat[..., 3 : 3 + self.n_oh]
        E   = feats_flat[..., 3 + self.n_oh    : 3 + self.n_oh + 1]
        p   = feats_flat[..., 3 + self.n_oh + 1: 3 + self.n_oh + 2]
        chi = feats_flat[..., 3 + self.n_oh + 2: 3 + self.n_oh + 3]

        inv = self._invariants(xyz)                          # (N, n_geom_inv)
        logE = torch.log1p(E.clamp(min=0.0))

        x_scalar = torch.cat([inv, ht, E, logE, p, chi], dim=-1)
        h = self.norm_in(x_scalar)
        out = self.mlp(h)
        if self.use_fourier:
            out = out + self.pe(inv)
        return self.out_norm(out)
