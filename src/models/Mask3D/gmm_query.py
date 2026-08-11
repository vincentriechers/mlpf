"""Per-query learnable spatial Gaussian Mixture in R^3.

Adapted from `mask3d_gmm/models/gmm_query.py` in the user's playground
(`/afs/cern.ch/work/m/mgarciam/private/mask3d`). Original lineage: each
MaskFormer query owns K Gaussians in 3-space; their mixture log-density
becomes an additive bias on the cross-attention QK^T/sqrt(d) logits,
alongside (not in place of) the existing Mask2Former binary instance mask.

Math
----
Per query q, K Gaussians:
  mu_qk in R^3                       (learnable mean)
  Sigma_qk = L_qk L_qk^T             (Cholesky, SPD by construction)
    parameterized via:
      log_diag (N, K, 3)  -> diag(L) = softplus(log_diag) + eps
      off_diag (N, K, 3)  -> L[1,0], L[2,0], L[2,1]
  pi_qk in R                         (mixing logit, softmax over K)

Log-density at point x in R^3:
  log N(x; mu, Sigma) = -1/2 [ d log(2 pi) + log|Sigma| + (x-mu)^T Sigma^{-1} (x-mu) ]
                     = -1/2 [ d log(2 pi) + 2 sum_i log L_ii + ||L^{-1}(x-mu)||^2 ]
Mixture log-density:
  log w(x) = logsumexp_k [ log pi_k + log N_k(x) ]

HEP-fork specifics
------------------
* Coordinates entering the model are in metres (build_targets scales mm → m
  via `pos_scale=0.001`). CLD detector occupies roughly a ±6 m cube, so we
  default `coord_range=(-6, 6)` (μ init range) and `sigma_init=1.0` m.
  The reference's `(-1, 1)` cube + 0.5 σ is appropriate for a normalised
  S3DIS scene, not a HEP detector.
* `linalg.solve_triangular` is not bf16-stable on every GPU; the bias is
  computed internally in float32 and cast to the caller's dtype only at the
  return point. The tensor is small (B × N_q × N_max), so the fp32 detour
  is essentially free.
* Padded slots from the encoder layout are NOT filtered here — the boolean
  `attn_mask` already forces them to -inf at softmax time, so adding a
  finite bias to them has no effect.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


_LOG_2PI = math.log(2.0 * math.pi)


class GaussianQueryMixture(nn.Module):
    """Spatial mixture of K Gaussians per query, used as an additive
    cross-attention bias.

    Args:
        num_queries: N_q.
        num_gaussians: K Gaussians per query (mixture components).
        sigma_init: initial std of each Gaussian, in the same length units
            as the points fed to `forward(point_xyz)` — metres for our CLD
            data pipeline.
        coord_range: (low, high) range for initial μ sampling, in the same
            length units. Defaults to (-6 m, +6 m), which covers the CLD /
            ILD active volume.
        min_log_w: lower clamp on the emitted mixture log-density. Prevents
            a Gaussian whose mass drifts away from real hits from collapsing
            the cross-attention to all -inf forever (which would zero the
            gradient permanently). Default -30.
    """

    def __init__(
        self,
        num_queries: int,
        num_gaussians: int = 4,
        sigma_init: float = 1.0,
        coord_range: tuple[float, float] = (-6.0, 6.0),
        min_log_w: float = -30.0,
        sigma_long: float | None = None,
        sigma_short: float | None = None,
        init_rotation: str = "random",
    ):
        super().__init__()
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be >= 1")
        self.num_queries = num_queries
        self.num_gaussians = num_gaussians
        self.sigma_init = sigma_init
        self.min_log_w = min_log_w
        # Track the anisotropic-init scales for extra_repr / forensics.
        self.sigma_long = sigma_long
        self.sigma_short = sigma_short
        self.init_rotation = init_rotation

        N, K = num_queries, num_gaussians

        lo, hi = coord_range
        mu_init = torch.empty(N, K, 3).uniform_(lo, hi)
        self.mu = nn.Parameter(mu_init)

        if sigma_long is not None and sigma_short is not None:
            # Anisotropic init: each Gaussian is a "shaft" of half-length σ_long
            # along one axis and σ_short across the other two. With a random
            # rotation per (N_q, K) the shafts span all orientations, so a
            # K-mixture can tile a curving HCAL shower with a few elongated
            # ellipsoids instead of trying to cover it with isotropic blobs.
            #
            # Σ = R · diag(σ_long², σ_short², σ_short²) · Rᵀ
            #   = (R · diag(σ)) (R · diag(σ))ᵀ  → Cholesky(Σ) is lower-tri L.
            # We then decompose L into the (log_diag, off_diag) form the rest
            # of this module expects so `_build_L` rebuilds the same Σ.
            if init_rotation == "random":
                A = torch.randn(N, K, 3, 3)
                Q, _ = torch.linalg.qr(A)
                # Force det(R) = +1 (proper rotation). Flip a column when QR
                # returned a reflection. Covariance is invariant under col-sign
                # flips so this is purely cosmetic, but keeps R in SO(3).
                det = torch.linalg.det(Q)                                       # (N, K)
                Q[..., :, 0] = Q[..., :, 0] * torch.sign(det).unsqueeze(-1)
                R = Q                                                           # (N, K, 3, 3)
            elif init_rotation == "identity":
                R = torch.eye(3).expand(N, K, 3, 3).contiguous()
            else:
                raise ValueError(
                    f"init_rotation must be 'random' or 'identity', got {init_rotation!r}"
                )

            sigmas = torch.tensor(
                [sigma_long, sigma_short, sigma_short], dtype=torch.float32
            )                                                                   # (3,)
            M = R * sigmas                                                      # (N, K, 3, 3)
            Cov = M @ M.transpose(-1, -2)                                       # SPD by construction
            # Tiny jitter for numerical safety with very small σ.
            eye = torch.eye(3).expand(N, K, 3, 3)
            L = torch.linalg.cholesky(Cov + 1e-8 * eye)                         # (N, K, 3, 3) lower-tri

            # `_build_L` reconstructs L as
            #   L[..., 0, 0] = softplus(log_diag[..., 0]) + 1e-4
            #   L[..., 1, 0] = off_diag[..., 0]
            #   L[..., 1, 1] = softplus(log_diag[..., 1]) + 1e-4
            #   L[..., 2, 0] = off_diag[..., 1]
            #   L[..., 2, 1] = off_diag[..., 2]
            #   L[..., 2, 2] = softplus(log_diag[..., 2]) + 1e-4
            # so log_diag = softplus⁻¹(L_diag − 1e-4), and off_diag are the
            # lower-triangle entries verbatim.
            l_diag = torch.stack(
                [L[..., 0, 0], L[..., 1, 1], L[..., 2, 2]], dim=-1
            ).clamp(min=1e-4 + 1e-7)                                            # > 1e-4 so log/expm1 stays finite
            log_diag_init = torch.log(torch.expm1(l_diag - 1e-4))               # (N, K, 3)
            off_diag_init = torch.stack(
                [L[..., 1, 0], L[..., 2, 0], L[..., 2, 1]], dim=-1
            )                                                                   # (N, K, 3)
            self.log_diag = nn.Parameter(log_diag_init)
            self.off_diag = nn.Parameter(off_diag_init)
        else:
            # softplus^{-1}(sigma_init) so that softplus(log_diag) ≈ sigma_init
            # at init. softplus(x) = log(1 + e^x); inverse: log(e^σ − 1).
            inv_sp = math.log(math.expm1(sigma_init))
            self.log_diag = nn.Parameter(torch.full((N, K, 3), inv_sp))
            self.off_diag = nn.Parameter(torch.zeros(N, K, 3))

        # Uniform mixing at init: softmax(zeros) is uniform over K.
        self.pi_logits = nn.Parameter(torch.zeros(N, K))

    # ---- Cholesky factor construction --------------------------------------

    def _build_L(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Lower-triangular Cholesky factor L: (N, K, 3, 3)."""
        diag = F.softplus(self.log_diag.to(dtype)) + 1e-4    # (N, K, 3) > 0
        off = self.off_diag.to(dtype)                        # (N, K, 3)
        N, K = diag.shape[:2]
        L = torch.zeros(N, K, 3, 3, device=diag.device, dtype=dtype)
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 0] = off[..., 0]
        L[..., 1, 1] = diag[..., 1]
        L[..., 2, 0] = off[..., 1]
        L[..., 2, 1] = off[..., 2]
        L[..., 2, 2] = diag[..., 2]
        return L

    # ---- Forward -----------------------------------------------------------

    def forward(self, point_xyz: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        """Compute the log mixture density at every input point, per query.

        Args:
            point_xyz: (B, P, 3) point coordinates in the same frame as the
                Gaussian means (metres for our pipeline). May contain padded
                slots — those are harmless because the boolean attn_mask
                downstream forces them to -inf at softmax time anyway.
            clamp: when True (default, used for the attention-bias path)
                lower-bound the returned log-density at `self.min_log_w`.
                The clamp protects the attention softmax from underflow but
                also zeros the gradient w.r.t. (μ, Σ) wherever the clamp
                fires — which is exactly the regime where the GMM coverage
                loss needs gradient. Pass clamp=False from the loss path.
        """
        in_dtype = point_xyz.dtype
        # Internal math in fp32 — solve_triangular is not always bf16-safe
        # and the tensor is tiny, so the precision lift is cheap.
        pts = point_xyz.to(torch.float32)
        B, P, three = pts.shape
        if three != 3:
            raise ValueError(f"GMM expects last dim 3, got {three}")
        N, K = self.num_queries, self.num_gaussians

        L = self._build_L(dtype=torch.float32)                       # (N, K, 3, 3)
        # L is lower-triangular 3×3; solving L y = (x − μ) once gives the
        # whitened residual. Cheaper than building Σ⁻¹.
        eye = torch.eye(3, device=L.device, dtype=L.dtype).expand(N, K, 3, 3)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)   # (N, K, 3, 3)

        # diff_{n,k,b,p,i} = x_{b,p,i} - mu_{n,k,i}
        diff = pts.view(1, 1, B, P, 3) - self.mu.view(N, K, 1, 1, 3)
        # y_{n,k,b,p,i} = sum_j L_inv_{n,k,i,j} diff_{n,k,b,p,j}
        y = torch.einsum("nkij,nkbpj->nkbpi", L_inv, diff)
        quad = (y * y).sum(-1)                                       # (N, K, B, P)

        # log|Sigma| = 2 sum_i log L_ii
        log_det = 2.0 * torch.log(
            torch.diagonal(L, dim1=-2, dim2=-1)
        ).sum(-1)                                                    # (N, K)

        log_normal = -0.5 * (
            3.0 * _LOG_2PI + log_det.view(N, K, 1, 1) + quad
        )                                                            # (N, K, B, P)

        log_pi = F.log_softmax(self.pi_logits.to(torch.float32), dim=-1)  # (N, K)
        log_w = torch.logsumexp(
            log_pi.view(N, K, 1, 1) + log_normal, dim=1
        )                                                            # (N, B, P)
        log_w = log_w.permute(1, 0, 2).contiguous()                  # (B, N, P)
        if clamp:
            log_w = log_w.clamp(min=self.min_log_w)
        return log_w.to(in_dtype)

    # ---- Auxiliary outputs -------------------------------------------------

    def regularizer(self, vol_w: float = 0.0, mean_w: float = 0.0) -> torch.Tensor:
        """Optional regularizer (off unless caller passes non-zero weights).

        - vol_w  * mean log|Sigma_qk|     keeps Gaussians from exploding.
        - mean_w * ||mu_qk||^2             keeps means inside the volume.
        """
        loss = self.mu.new_zeros(())
        if vol_w == 0.0 and mean_w == 0.0:
            return loss
        L = self._build_L(dtype=torch.float32)
        log_det = 2.0 * torch.log(
            torch.diagonal(L, dim1=-2, dim2=-1)
        ).sum(-1)
        if vol_w != 0.0:
            loss = loss + vol_w * log_det.mean()
        if mean_w != 0.0:
            loss = loss + mean_w * (self.mu**2).mean()
        return loss

    def extra_repr(self) -> str:
        if self.sigma_long is not None and self.sigma_short is not None:
            shape = (
                f"sigma_long={self.sigma_long}, sigma_short={self.sigma_short}, "
                f"init_rotation={self.init_rotation!r}"
            )
        else:
            shape = f"sigma_init={self.sigma_init}"
        return (
            f"num_queries={self.num_queries}, K={self.num_gaussians}, "
            f"{shape}, min_log_w={self.min_log_w}"
        )


class DynamicGaussianQueryMixture(nn.Module):
    """Event-conditional GMM = static per-slot prior + dynamic head on q.

    The static `GaussianQueryMixture` gives each query slot a learnable
    spatial prior — that prior is shared across events and stays constant
    across decoder layers. This wrapper adds a small MLP head that maps
    each query's *current feature embedding* `q[b, qi, :]` to per-Gaussian
    deltas:

        μ        = μ_base[qi]        + Δμ(q[b, qi])
        log_diag = log_diag_base[qi] + Δlog_diag(q[b, qi])
        off_diag = off_diag_base[qi] + Δoff_diag(q[b, qi])
        pi_logit = pi_logit_base[qi] + Δpi_logit(q[b, qi])

    so the GMM becomes a function of (event, query, decoder layer). The
    head is zero-initialised, so at step 0 the dynamic GMM is *identical*
    to its static base — backward-compatible by construction. As training
    progresses the head's weights learn to refine the spatial prior using
    the encoder/decoder features that already condition on the point cloud.

    Forward signature mirrors the static class but takes `q` first:
        log_w = dyn(q, point_xyz, clamp=...)
    `q` is the per-event, per-decoder-layer query feature tensor of shape
    `(B, N_q, D)`; the decoder calls this method once per layer so the
    GMM tracks the iterative query refinement.
    """

    def __init__(
        self,
        query_dim: int,
        num_queries: int,
        num_gaussians: int = 4,
        sigma_init: float = 1.0,
        coord_range: tuple[float, float] = (-6.0, 6.0),
        min_log_w: float = -30.0,
        sigma_long: float | None = None,
        sigma_short: float | None = None,
        init_rotation: str = "random",
        head_hidden_mult: int = 2,
    ):
        super().__init__()
        # Per-slot static base prior (anisotropic shaft init when σ_long/short
        # supplied — same machinery as the standalone static GMM).
        self.base = GaussianQueryMixture(
            num_queries=num_queries,
            num_gaussians=num_gaussians,
            sigma_init=sigma_init,
            coord_range=coord_range,
            min_log_w=min_log_w,
            sigma_long=sigma_long,
            sigma_short=sigma_short,
            init_rotation=init_rotation,
        )
        self.num_queries = num_queries
        self.num_gaussians = num_gaussians
        self.min_log_w = min_log_w
        # Track for repr / forensics
        self.query_dim = query_dim
        self.head_hidden_mult = head_hidden_mult

        # Output: K Gaussians × (3 Δμ + 3 Δlog_diag + 3 Δoff_diag + 1 Δπ_logit) = 10K
        K = num_gaussians
        out_dim = K * 10
        hidden = max(query_dim * head_hidden_mult, out_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # Zero-init the final linear so the head outputs zero deltas at start
        # → dynamic ≡ static at step 0. Lets the model decide for itself how
        # much to deviate from the static prior, and avoids cold-starting in
        # a different regime than the static-GMM ablation.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def mu(self):
        return self.base.mu

    @property
    def log_diag(self):
        return self.base.log_diag

    @property
    def off_diag(self):
        return self.base.off_diag

    @property
    def pi_logits(self):
        return self.base.pi_logits

    @property
    def sigma_init(self):
        return self.base.sigma_init

    @property
    def sigma_long(self):
        return self.base.sigma_long

    @property
    def sigma_short(self):
        return self.base.sigma_short

    @property
    def init_rotation(self):
        return self.base.init_rotation

    def _build_L_dynamic(self, log_diag, off_diag, dtype=torch.float32):
        """Same Cholesky-from-(log_diag, off_diag) reconstruction as the
        static `_build_L`, but on per-(batch, query) tensors of shape
        (..., 3) — final L has shape (..., 3, 3).
        """
        diag = F.softplus(log_diag.to(dtype)) + 1e-4
        off = off_diag.to(dtype)
        L = torch.zeros(
            *diag.shape[:-1], 3, 3,
            device=diag.device, dtype=dtype,
        )
        L[..., 0, 0] = diag[..., 0]
        L[..., 1, 0] = off[..., 0]
        L[..., 1, 1] = diag[..., 1]
        L[..., 2, 0] = off[..., 1]
        L[..., 2, 1] = off[..., 2]
        L[..., 2, 2] = diag[..., 2]
        return L, diag                # diag returned for `log_det = 2·sum(log diag)`

    def forward(
        self,
        q: torch.Tensor,
        point_xyz: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        """Compute the event-conditional GMM log-density.

        Args:
            q: (B, N_q, D) per-event, per-query feature embeddings. The
                decoder passes the *current* refined queries (so the GMM
                tracks layer-by-layer refinement when called per layer).
            point_xyz: (B, N_k, 3) hit positions in the same metres frame
                as the base GMM was initialised in.
            clamp: same semantics as the static class. False for the
                coverage-loss path (preserves gradient at far-from-μ hits).
        Returns:
            log_w: (B, N_q, N_k) per-event GMM log-density.
        """
        B, N_q, D = q.shape
        N_k = point_xyz.size(1)
        K = self.num_gaussians

        # Predict deltas from current q. With zero-init final layer, deltas
        # are exactly zero at step 0 → dynamic output = static base output.
        deltas = self.head(q).view(B, N_q, K, 10)
        d_mu        = deltas[..., 0:3]                                    # (B, N_q, K, 3)
        d_log_diag  = deltas[..., 3:6]
        d_off_diag  = deltas[..., 6:9]
        d_pi_logit  = deltas[..., 9]                                      # (B, N_q, K)

        # Base parameters (broadcast over batch).
        mu_base        = self.base.mu.unsqueeze(0)                        # (1, N_q, K, 3)
        log_diag_base  = self.base.log_diag.unsqueeze(0)
        off_diag_base  = self.base.off_diag.unsqueeze(0)
        pi_logits_base = self.base.pi_logits.unsqueeze(0)                 # (1, N_q, K)

        mu        = mu_base        + d_mu
        log_diag  = log_diag_base  + d_log_diag
        off_diag  = off_diag_base  + d_off_diag
        pi_logits = pi_logits_base + d_pi_logit

        # Cholesky math in fp32 (solve_triangular not always bf16-stable).
        L, diag = self._build_L_dynamic(log_diag, off_diag, dtype=torch.float32)
        eye = torch.eye(3, device=L.device, dtype=L.dtype).expand(B, N_q, K, 3, 3)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False)        # (B, N_q, K, 3, 3)

        pts = point_xyz.to(torch.float32)
        # diff_{b,n,k,p,i} = x_{b,p,i} - mu_{b,n,k,i}
        diff = pts.view(B, 1, 1, N_k, 3) - mu.to(torch.float32).view(B, N_q, K, 1, 3)
        # y_{b,n,k,p,i} = sum_j L_inv_{b,n,k,i,j} diff_{b,n,k,p,j}
        y = torch.einsum("bnkij,bnkpj->bnkpi", L_inv, diff)
        quad = (y * y).sum(-1)                                            # (B, N_q, K, N_k)

        log_det = 2.0 * torch.log(diag).sum(-1)                           # (B, N_q, K)

        log_normal = -0.5 * (
            3.0 * _LOG_2PI + log_det.unsqueeze(-1) + quad
        )                                                                  # (B, N_q, K, N_k)
        log_pi = F.log_softmax(pi_logits.to(torch.float32), dim=-1)       # (B, N_q, K)
        log_w = torch.logsumexp(
            log_pi.unsqueeze(-1) + log_normal, dim=2,
        )                                                                  # (B, N_q, N_k)

        if clamp:
            log_w = log_w.clamp(min=self.min_log_w)
        return log_w.to(q.dtype)

    def regularizer(self, vol_w: float = 0.0, mean_w: float = 0.0) -> torch.Tensor:
        """Same penalties as the static base — applied only to the static
        base parameters. Head-driven deltas aren't regularised here; if
        their growth becomes a problem, add weight decay in the optimizer.
        """
        return self.base.regularizer(vol_w=vol_w, mean_w=mean_w)

    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, num_queries={self.num_queries}, "
            f"K={self.num_gaussians}, head_hidden_mult={self.head_hidden_mult}, "
            f"min_log_w={self.min_log_w}  (base prior below)"
        )


# ---------------------------------------------------------------------------
# v5: single anchor + scale per query (no K-mixture, no density).
# Adapted from /afs/cern.ch/work/m/mgarciam/private/mask3d/Mask3D/models/gmm_query.py.
# ---------------------------------------------------------------------------


class QueryAnchorHead(nn.Module):
    """Predicts a 3D anchor μ_q and an isotropic scale σ_q per query.

    Drop-in replacement for the K-Gaussian mixture used in `use_gmm_query`
    mode. Each query gets a single "where I live" point and a "how big a
    neighbourhood I care about" scalar, then the decoder builds relative
    coordinates `(p - μ_q)/σ_q` and feeds them through a small MLP to
    produce an additive attention bias. No density evaluation, no
    logsumexp, no Cholesky — just an anchor + scale.

    Args:
        hidden_dim: width of the query feature vector.
        sigma_init: initial scale (in the same coord frame as the points
            you pass to the model). Final MLP layer is zero-initialised,
            so at step 0 each query's μ equals its `center_xyz` input and
            σ equals `sigma_init`.
    """

    def __init__(self, hidden_dim: int, sigma_init: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma_init = sigma_init
        # MLP: hidden_dim → hidden_dim → 4   (Δμ_x, Δμ_y, Δμ_z, Δlog_σ)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )
        # Zero-init final layer → identity-prior at step 0 (μ = center,
        # σ = sigma_init). The model starts behaving like there is no
        # head, and the head only "wakes up" as training pushes it off
        # zero. Backward-compat with use_query_anchor=False at init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # softplus⁻¹(sigma_init) so softplus(log_sigma_init) ≈ sigma_init.
        self.register_buffer(
            "_inv_sp_sigma",
            torch.tensor(math.log(math.expm1(sigma_init)), dtype=torch.float32),
        )

    def forward(
        self,
        query_feat: torch.Tensor,                       # (B, N_q, D)
        center_xyz: torch.Tensor,                       # (B, N_q, 3)
    ):
        """Return `(mu, sigma)` of shapes `(B, N_q, 3)` and `(B, N_q, 1)`."""
        delta = self.mlp(query_feat)                    # (B, N_q, 4)
        mu = center_xyz + delta[..., :3]
        log_sigma = self._inv_sp_sigma + delta[..., 3:4]
        sigma = F.softplus(log_sigma) + 1e-4
        return mu, sigma

    def extra_repr(self) -> str:
        return f"hidden_dim={self.hidden_dim}, sigma_init={self.sigma_init}"
