"""Hungarian matcher mirroring `hepattn.models.matcher.Matcher`.

Stateful nn.Module that solves a batched linear-assignment problem in one
call. Designed to be invoked once per training step with a stacked cost
tensor `(L*B, N_q, N_p)` (decoder layers folded into the batch axis), so
the full deep-supervision matching costs a single GPU→CPU sync and a
single (optionally thread-parallel) scipy loop.

API & internals follow hepattn directly:
  - `Matcher.forward(costs, object_valid_mask, query_valid_mask)` →
    int64 `(LB, N_q)` *permutation*: the first `n_real_b` slots are the
    matched queries (aligned with GT slots `0..n_real_b-1`), the rest are
    the unused query indices in natural order.
  - Costs are transposed to `(LB, N_p, N_q)` so each row is a GT slot.
  - Optional thread/process pools speed up the scipy loop when the batch
    × layers axis is large; defaults to sequential because for our typical
    Mask3D batches (~10 events × ~9 layers, n_real ≤ ~30) scipy is already
    sub-millisecond per call and pool overhead dominates.

We omit hepattn's `lap1015` solver (extra binary dep) and the adaptive
solver-selection logic (only scipy is available, so there's nothing to
choose between).
"""
from __future__ import annotations

import atexit
import contextlib
from multiprocessing import get_context, shared_memory
from multiprocessing.pool import ThreadPool
from threading import Lock
from typing import Literal

import numpy as np
import scipy.optimize
import torch
from torch import nn


_POOL_LOCK = Lock()
_THREAD_POOLS: dict[int, ThreadPool] = {}
_PROCESS_POOLS: dict[int, "object"] = {}


def _get_thread_pool(n_jobs: int) -> ThreadPool:
    with _POOL_LOCK:
        pool = _THREAD_POOLS.get(n_jobs)
        if pool is None:
            pool = ThreadPool(processes=n_jobs)
            _THREAD_POOLS[n_jobs] = pool
        return pool


def _get_process_pool(n_jobs: int):
    with _POOL_LOCK:
        pool = _PROCESS_POOLS.get(n_jobs)
        if pool is None:
            ctx = get_context("spawn")
            pool = ctx.Pool(processes=n_jobs)
            _PROCESS_POOLS[n_jobs] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    for pool in list(_THREAD_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
    for pool in list(_PROCESS_POOLS.values()):
        try:
            pool.close()
            pool.join()
        except Exception:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass


def _solve_scipy(cost: np.ndarray) -> np.ndarray:
    _, col_idx = scipy.optimize.linear_sum_assignment(cost)
    return col_idx


def match_individual(cost: np.ndarray, default_idx: np.ndarray) -> np.ndarray:
    """Solve one (n_real, N_q) cost and return a full N_q permutation.

    The first n_real positions are scipy's column assignment; the remaining
    positions are the unused query indices in natural order.
    """
    pred_idx = np.asarray(_solve_scipy(cost), dtype=np.int64)
    remaining = np.ones(default_idx.shape[0], dtype=np.bool_)
    remaining[pred_idx] = False
    return np.concatenate([pred_idx, default_idx[remaining]])


def match_parallel(
    costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8
) -> torch.Tensor:
    """Thread-pool matching over the leading batch axis."""
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs
    default_idx = np.arange(pred_dim, dtype=np.int64)

    if n_jobs <= 1 or batch_size <= 1:
        results = [
            match_individual(costs_t[i][: lengths_np[i]], default_idx)
            for i in range(batch_size)
        ]
        return torch.from_numpy(np.stack(results, axis=0))

    def _run(i: int) -> np.ndarray:
        return match_individual(costs_t[i][: lengths_np[i]], default_idx)

    pool = _get_thread_pool(n_jobs)
    results = pool.map(_run, range(batch_size), chunksize=chunk_size)
    return torch.from_numpy(np.stack(results, axis=0))


def _mp_match_task(args):
    shm_name, shape, dtype_str, i, length, pred_dim = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        costs_t = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        default_idx = np.arange(pred_dim, dtype=np.int64)
        return match_individual(costs_t[i][:length], default_idx)
    finally:
        shm.close()


def match_multiprocess(
    costs_t: np.ndarray, lengths_np: np.ndarray, pred_dim: int, n_jobs: int = 8
) -> torch.Tensor:
    """Multi-process matching with shared memory (bypasses GIL).

    Mostly useful when scipy itself is the bottleneck (large `n_real`).
    For typical Mask3D events the thread backend is faster.
    """
    batch_size = len(costs_t)
    n_jobs = min(n_jobs, batch_size)
    chunk_size = (batch_size + n_jobs - 1) // n_jobs

    shm = shared_memory.SharedMemory(create=True, size=costs_t.nbytes)
    try:
        shm_arr = np.ndarray(costs_t.shape, dtype=costs_t.dtype, buffer=shm.buf)
        shm_arr[...] = costs_t
        tasks = [
            (shm.name, costs_t.shape, costs_t.dtype.str, i, int(lengths_np[i]), pred_dim)
            for i in range(batch_size)
        ]
        pool = _get_process_pool(n_jobs)
        results = pool.map(_mp_match_task, tasks, chunksize=chunk_size)
        return torch.from_numpy(np.stack(results, axis=0))
    finally:
        try:
            shm.close()
        finally:
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()


class Matcher(nn.Module):
    """Stateful Hungarian matcher.

    Args:
        parallel_solver:  if True, run scipy across the (LB) batch axis with
                          a persistent thread or process pool.
        parallel_backend: 'thread' (default — scipy releases the GIL during
                          LAP) or 'process' (shared-memory pool).
        n_jobs:           number of workers if parallel_solver=True.

    `forward` is `@torch.no_grad`; it returns int64 `(LB, N_q)`.
    """

    def __init__(
        self,
        parallel_solver: bool = False,
        parallel_backend: Literal["thread", "process"] = "thread",
        n_jobs: int = 8,
    ):
        super().__init__()
        if parallel_backend not in {"thread", "process"}:
            raise ValueError(
                f"parallel_backend must be 'thread' or 'process', got: {parallel_backend}"
            )
        self.parallel_solver = parallel_solver
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs

    def compute_matching(
        self,
        costs: np.ndarray,
        object_valid_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if object_valid_mask is None:
            object_valid_mask = torch.ones(
                (costs.shape[0], costs.shape[2]), dtype=torch.bool
            )
        object_valid_mask = object_valid_mask.detach().bool()
        lengths_np = (
            object_valid_mask.sum(dim=1).cpu().numpy().astype(np.int32, copy=False)
        )
        pred_dim = costs.shape[1]

        if query_valid_mask is not None:
            query_valid_mask = query_valid_mask.detach().bool()
            invalid_query = (~query_valid_mask).cpu().numpy()  # (LB, N_q)
            costs = np.where(
                invalid_query[:, :, None], np.finfo(np.float32).max / 10, costs
            )

        # rows = GT slots, cols = queries
        costs_t = np.ascontiguousarray(costs.swapaxes(1, 2))  # (LB, N_p, N_q)

        if self.parallel_solver:
            if self.parallel_backend == "thread":
                return match_parallel(costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)
            return match_multiprocess(costs_t, lengths_np, pred_dim, n_jobs=self.n_jobs)

        default_idx = np.arange(pred_dim, dtype=np.int64)
        results = [
            match_individual(costs_t[i][: lengths_np[i]], default_idx)
            for i in range(len(costs_t))
        ]
        return torch.from_numpy(np.stack(results, axis=0))

    @torch.no_grad()
    def forward(
        self,
        costs: torch.Tensor,
        object_valid_mask: torch.Tensor | None = None,
        query_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = costs.device
        costs_np = costs.detach().to(torch.float32).cpu().numpy()
        # NaN/Inf guard: a single bad logit anywhere in the model can produce
        # non-finite cost entries, and scipy.optimize.linear_sum_assignment
        # crashes on those (`matrix contains invalid numeric entries`). The
        # matcher is the canary, but the underlying cause is upstream — by
        # the time we get here we just want training to continue: replace
        # non-finite entries with a large finite cost (worse than any real
        # match) so the assignment is still well-defined. Combined with the
        # `gradient_clip_val=0.1` on the Trainer this is enough to ride
        # through the occasional hot-step without a job crash.
        if not np.isfinite(costs_np).all():
            big = np.finfo(np.float32).max / 1e3
            costs_np = np.nan_to_num(
                costs_np, nan=big, posinf=big, neginf=-big, copy=False
            )
        pred_idxs = self.compute_matching(costs_np, object_valid_mask, query_valid_mask)
        assert torch.all(pred_idxs >= 0), "Matcher error!"
        return pred_idxs.to(device)
