"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING

from .core.fusion.defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    normalize_dense_fallback_policy,
)
from .core.fusion.profiles import (
    DEFAULT_COMPUTATION_PROFILE,
    ComputationProfile,
    get_computation_profile,
)

if TYPE_CHECKING:
    from .core.fusion.types import PairwiseFusionGraph
    from .model_selection.contracts import SelectionContract


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _score_name(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_")
    if value not in {"fixed_partition_bic", "fixed_partition_dirichlet_score"}:
        raise ValueError("Unknown fixed-partition selection score.")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = DEFAULT_DEVICE
    dtype: str = "float32"
    fallback: str = DEFAULT_DENSE_FALLBACK_POLICY
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback", normalize_dense_fallback_policy(self.fallback))


@dataclass(frozen=True, slots=True)
class CertificateConfig:
    max_iter: int
    refinement_rounds: int
    column_tolerance_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE

    def __post_init__(self) -> None:
        if int(self.max_iter) < 1 or int(self.refinement_rounds) < 0:
            raise ValueError("Certificate iteration budgets are invalid.")
        _positive("certificate_column_tol_scale", self.column_tolerance_scale)

    @staticmethod
    def admission_tolerance(solver_tolerance: float) -> float:
        return 5.0 * _positive("tol", solver_tolerance)


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS


@dataclass(frozen=True, slots=True)
class SolverConfig:
    outer_max_iter: int
    inner_max_iter: int
    tolerance: float
    objective_shape: str
    certificate: CertificateConfig
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    # When set, KKT admission and certificate construction use this tolerance
    # while ``tolerance`` only controls how deep the solver iterates. A deep
    # recovery solve can then be admitted against the immutable profile gate
    # instead of an accidentally tighter one. None means both coincide.
    certification_tolerance: float | None = None
    # Internal recovery mode: once the frozen context is float64, iterative
    # progress uses the same componentwise residual as terminal admission.
    use_backward_error_progress: bool = False

    def __post_init__(self) -> None:
        _positive("tol", self.tolerance)
        if self.certification_tolerance is not None:
            _positive("certification_tolerance", self.certification_tolerance)


@dataclass(frozen=True, slots=True)
class RefitConfig:
    tolerance: float
    max_iter: int
    mode: str
    grid_points: int
    local_steps: int

    def __post_init__(self) -> None:
        _positive("selection_refit_tol", self.tolerance)
        if int(self.max_iter) < 1:
            raise ValueError("selection_refit_max_iter must be positive.")


@dataclass(frozen=True, slots=True)
class LambdaSearchConfig:
    exploration_budget: int
    refinement_budget: int
    solver_retry_limit: int


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Resolved selection contract and all numerical selection settings."""

    score: str
    contract: SelectionContract
    graph_pilot_source: str
    partition_tolerance: float
    refit: RefitConfig
    lambda_search: LambdaSearchConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _score_name(self.score))
        _positive("selection_partition_tol", self.partition_tolerance)

    @property
    def contract_id(self) -> str:
        return str(self.contract.contract_id)

    @property
    def dirichlet_alpha(self) -> float:
        return float(self.contract.partition_config.classification_alpha)

    @property
    def dirichlet_code_weight(self) -> float:
        return float(self.contract.partition_config.classification_code_weight)


@dataclass(frozen=True, slots=True)
class GraphConfig:
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Canonical boundary consumed by the solver, selector, and serializers."""

    lambda_value: float
    major_prior: float
    eps: float
    runtime: RuntimeConfig
    solver: SolverConfig
    selection: SelectionConfig
    graph: GraphConfig
    computation_profile: ComputationProfile

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.lambda_value)) or float(self.lambda_value) < 0.0:
            raise ValueError("lambda_value must be finite and nonnegative.")
        if not 0.0 < float(self.major_prior) < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        _positive("eps", self.eps)
        if self.selection.contract.force_float64 and self.runtime.dtype != "float64":
            object.__setattr__(self, "runtime", replace(self.runtime, dtype="float64"))


def resolve_fit_config(
    *,
    lambda_value: float = 0.0,
    computation_profile: str = DEFAULT_COMPUTATION_PROFILE,
    selection_contract: str | None = None,
    selection_score: str = "fixed_partition_dirichlet_score",
    outer_max_iter: int | None = None,
    inner_max_iter: int | None = None,
    tol: float | None = None,
    selection_partition_tol: float | None = None,
    selection_refit_tol: float | None = None,
    selection_refit_max_iter: int | None = None,
    certificate_max_iter: int | None = None,
    certificate_refinement_rounds: int | None = None,
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    major_prior: float = 0.5,
    eps: float = 1e-6,
    graph: PairwiseFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    device: str = DEFAULT_DEVICE,
    dtype: str | None = None,
    objective_shape: str = "auto",
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES,
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY,
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH,
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS,
    verbose: bool = False,
) -> FitConfig:
    """Resolve a profile and selection contract once into concrete settings."""

    from .model_selection.contracts import (
        DEFAULT_SELECTION_CONTRACT,
        get_selection_contract,
    )

    profile = get_computation_profile(computation_profile)
    contract = get_selection_contract(selection_contract or DEFAULT_SELECTION_CONTRACT)
    graph_source = str(contract.graph_pilot_source)
    if graph_source == "profile_default":
        graph_source = "partition_guide" if profile.is_strict else "zero_penalty_pilot"
    runtime = RuntimeConfig(
        device=str(device),
        dtype="float64" if contract.force_float64 else str(dtype or profile.raw_dtype),
        fallback=str(dense_fallback_policy),
        verbose=bool(verbose),
    )
    certificate = CertificateConfig(
        max_iter=int(profile.certificate_max_iter if certificate_max_iter is None else certificate_max_iter),
        refinement_rounds=int(profile.certificate_refinement_rounds if certificate_refinement_rounds is None else certificate_refinement_rounds),
        column_tolerance_scale=float(certificate_column_tol_scale),
    )
    solver = SolverConfig(
        outer_max_iter=int(profile.outer_max_iter if outer_max_iter is None else outer_max_iter),
        inner_max_iter=int(profile.inner_max_iter if inner_max_iter is None else inner_max_iter),
        tolerance=float(profile.solver_tolerance if tol is None else tol),
        objective_shape=str(objective_shape),
        certificate=certificate,
        resources=ResourceConfig(
            workset_max_bytes=int(workset_max_bytes),
            compressed_cache_max_bytes=int(compressed_cache_max_bytes),
            workset_add_batch=int(workset_add_batch),
            workset_max_expansions=int(workset_max_expansions),
        ),
    )
    selection = SelectionConfig(
        score=str(selection_score),
        contract=contract,
        graph_pilot_source=graph_source,
        partition_tolerance=float(profile.partition_tolerance if selection_partition_tol is None else selection_partition_tol),
        refit=RefitConfig(
            tolerance=float(profile.refit_tolerance if selection_refit_tol is None else selection_refit_tol),
            max_iter=int(profile.refit_max_iter if selection_refit_max_iter is None else selection_refit_max_iter),
            mode=str(profile.scalar_mode),
            grid_points=int(profile.scalar_grid_points),
            local_steps=int(profile.scalar_local_steps),
        ),
        lambda_search=LambdaSearchConfig(
            exploration_budget=int(profile.lambda_budget),
            refinement_budget=int(profile.lambda_refinement_budget),
            solver_retry_limit=int(profile.solver_retry_limit),
        ),
    )
    return FitConfig(
        lambda_value=float(lambda_value),
        major_prior=float(major_prior),
        eps=float(eps),
        runtime=runtime,
        solver=solver,
        selection=selection,
        graph=GraphConfig(
            graph=graph,
            adaptive_weight_gamma=float(adaptive_weight_gamma),
            adaptive_weight_floor=float(adaptive_weight_floor),
            adaptive_weight_baseline=float(adaptive_weight_baseline),
        ),
        computation_profile=profile,
    )


__all__ = [
    "CertificateConfig", "FitConfig", "GraphConfig", "LambdaSearchConfig",
    "RefitConfig", "ResourceConfig", "RuntimeConfig", "SelectionConfig",
    "SolverConfig", "resolve_fit_config",
]
