"""Small immutable request objects for one pairwise-fusion solve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import torch

from ...config import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DenseFallbackPolicy,
)
from ...io.data import TumorData
from .types import PairwiseFusionGraph, SolverContext, SolverState, TorchRuntime


ArrayLike: TypeAlias = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class FusionProblem:
    """Observed objective and graph contract for one lambda value.

    With a supplied :class:`SolverContext`, ``graph=None`` means exact reuse
    of that context's already frozen graph; adaptive controls are construction
    inputs only. An explicit graph must match the frozen graph identity.
    """

    data: TumorData
    lambda_value: float
    major_prior: float
    eps: float
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0
    objective_shape: str = "auto"


@dataclass(frozen=True, slots=True)
class SolvePlan:
    """Numerical method and resource policy, excluding attempt-local state."""

    outer_max_iter: int
    inner_max_iter: int
    tol: float
    certification_tol: float | None = None
    use_backward_error_progress: bool = False
    stagnation_audit_patience: int = 4
    device: str | None = DEFAULT_DEVICE
    dtype: str | None = DEFAULT_DTYPE
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    dense_fallback_policy: DenseFallbackPolicy = DEFAULT_DENSE_FALLBACK_POLICY
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    certificate_max_iter: int = DEFAULT_CERTIFICATE_MAX_ITER
    certificate_refinement_rounds: int = DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE
    verbose: bool = False


@dataclass(frozen=True, slots=True)
class SolverInit:
    """Optional reusable context, starts, runtime, and continuation state."""

    phi_start: ArrayLike | None = None
    exact_pilot: ArrayLike | None = None
    pooled_start: ArrayLike | None = None
    scalar_well_starts: tuple[ArrayLike, ...] | None = None
    runtime: TorchRuntime | None = None
    solver_context: SolverContext | None = None
    solver_state: SolverState | None = None
    start_mode: Literal["full", "warm_plus_pilot", "warm_only"] = "full"
    append_default_nonconvex_starts: bool | None = None

    def __post_init__(self) -> None:
        if self.solver_context is None:
            return
        redundant = tuple(
            name
            for name, value in (
                ("exact_pilot", self.exact_pilot),
                ("pooled_start", self.pooled_start),
                ("scalar_well_starts", self.scalar_well_starts),
                ("runtime", self.runtime),
            )
            if value is not None
        )
        if redundant:
            joined = ", ".join(redundant)
            raise ValueError(
                "SolverInit cannot combine solver_context with context-owned "
                f"fields: {joined}."
            )


@dataclass(frozen=True, slots=True)
class SolveBudget:
    """Attempt-local work remainder; not a reusable resource configuration."""

    max_edge_pass_equivalents: int | None = None

    def __post_init__(self) -> None:
        value = self.max_edge_pass_equivalents
        if value is not None and int(value) <= 0:
            raise ValueError("max_edge_pass_equivalents must be positive when set.")


__all__ = ["ArrayLike", "FusionProblem", "SolveBudget", "SolvePlan", "SolverInit"]
