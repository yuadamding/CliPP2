"""Immutable fit configuration and the legacy flat-options adapter.

The nested dataclasses are the canonical representation of one fit's runtime,
solver, certificate, resource, refit, lambda-search, and selection settings.
``FitOptions`` preserves the historical flat constructor used by callers while
providing a lossless :attr:`FitOptions.config` view.

Selection-contract resolution intentionally lives at this top-level boundary,
not in :mod:`CliPP2.core`.  This keeps the numerical core independent of the
model-selection policy package without changing legacy constructor behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING

from .core.fusion.defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_OPTIMIZATION_TOLERANCE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
)
from .core.fusion.profiles import (
    DEFAULT_COMPUTATION_PROFILE,
    get_computation_profile,
)

if TYPE_CHECKING:
    from .core.fusion.types import PairwiseFusionGraph


def _positive_finite(name: str, value: float) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return normalized


def _selection_score(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {
        "fixed_partition_bic",
        "fixed_partition_dirichlet_score",
    }:
        raise ValueError("Unknown fixed-partition selection score.")
    return normalized


def _selection_contract(value: str):
    """Resolve application policy without coupling the numerical core to it."""

    from .model_selection.contracts import get_selection_contract

    return get_selection_contract(value)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution placement and precision for the raw optimizer."""

    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    fallback: str = DEFAULT_DENSE_FALLBACK_POLICY


@dataclass(frozen=True, slots=True)
class CertificateConfig:
    """Terminal certificate effort; admission remains ``5 * solver.tolerance``."""

    max_iter: int = 128
    refinement_rounds: int = 1
    column_tolerance_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE

    def __post_init__(self) -> None:
        _positive_finite("certificate_column_tol_scale", self.column_tolerance_scale)

    @staticmethod
    def admission_tolerance(solver_tolerance: float) -> float:
        """Return the fixed full-KKT gate without introducing a second tolerance."""

        return 5.0 * _positive_finite("tol", solver_tolerance)


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    """Bounded memory and working-set controls."""

    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Raw fixed-objective solver controls."""

    outer_max_iter: int = 6
    inner_max_iter: int = 25
    tolerance: float = DEFAULT_OPTIMIZATION_TOLERANCE
    objective_shape: str = "auto"
    certificate: CertificateConfig = field(default_factory=CertificateConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    def __post_init__(self) -> None:
        _positive_finite("tol", self.tolerance)


@dataclass(frozen=True, slots=True)
class RefitConfig:
    """Fixed-label scalar-refit accuracy and profile-defined algorithm."""

    tolerance: float = 1e-5
    max_iter: int = 64
    mode: str = "grid_local"
    grid_points: int = 64
    local_steps: int = 3

    def __post_init__(self) -> None:
        _positive_finite("selection_refit_tol", self.tolerance)
        if int(self.max_iter) < 1:
            raise ValueError("selection_refit_max_iter must be positive.")


@dataclass(frozen=True, slots=True)
class LambdaSearchConfig:
    """Profile-defined bounded lambda-controller budgets."""

    exploration_budget: int = 8
    refinement_budget: int = 2
    solver_retry_limit: int = 1


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Partition generation, refit, and score contract."""

    score: str = "fixed_partition_dirichlet_score"
    contract_id: str = "hybrid-ward-cem-v1"
    partition_tolerance: float = 2e-4
    dirichlet_alpha: float = 1.0
    dirichlet_code_weight: float = 0.7
    refit: RefitConfig = field(default_factory=RefitConfig)
    lambda_search: LambdaSearchConfig = field(default_factory=LambdaSearchConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _selection_score(self.score))
        contract = _selection_contract(self.contract_id)
        object.__setattr__(self, "contract_id", contract.contract_id)
        object.__setattr__(
            self,
            "dirichlet_alpha",
            float(contract.partition_config.classification_alpha),
        )
        object.__setattr__(
            self,
            "dirichlet_code_weight",
            float(contract.partition_config.classification_code_weight),
        )
        _positive_finite("selection_partition_tol", self.partition_tolerance)
        _positive_finite("selection_dirichlet_alpha", self.dirichlet_alpha)
        weight = float(self.dirichlet_code_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "selection_dirichlet_code_weight must be nonnegative and finite."
            )


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Canonical immutable configuration for one fixed-objective fit."""

    lambda_value: float
    major_prior: float = 0.5
    eps: float = 1e-6
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0
    summary_tol: float | None = 2e-4
    verbose: bool = False
    computation_profile: str = DEFAULT_COMPUTATION_PROFILE

    def __post_init__(self) -> None:
        lambda_value = float(self.lambda_value)
        if not math.isfinite(lambda_value) or lambda_value < 0.0:
            raise ValueError("lambda_value must be finite and nonnegative.")
        if not 0.0 < float(self.major_prior) < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        _positive_finite("eps", self.eps)
        profile = get_computation_profile(self.computation_profile)
        object.__setattr__(self, "computation_profile", profile.name)
        contract = _selection_contract(self.selection.contract_id)
        if contract.force_float64 and self.runtime.dtype != "float64":
            object.__setattr__(self, "runtime", replace(self.runtime, dtype="float64"))

    def to_options(self) -> FitOptions:
        """Return the backward-compatible flat adapter."""

        return FitOptions.from_config(self)


@dataclass
class FitOptions:
    """Backward-compatible flat adapter over :class:`FitConfig`.

    Existing construction, attribute access, and ``dataclasses.replace`` calls
    remain valid.  New code should consume :attr:`config` and pass the nested
    groups down as migration reaches each subsystem.
    """

    lambda_value: float
    outer_max_iter: int = 6
    inner_max_iter: int = 25
    tol: float = DEFAULT_OPTIMIZATION_TOLERANCE
    major_prior: float = 0.5
    eps: float = 1e-6
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    summary_tol: float | None = 2e-4
    selection_score: str = "fixed_partition_dirichlet_score"
    selection_partition_tol: float = 2e-4
    selection_refit_tol: float = 1e-5
    selection_refit_max_iter: int = 64
    selection_contract: str = "hybrid-ward-cem-v1"
    selection_dirichlet_alpha: float = 1.0
    selection_dirichlet_code_weight: float = 0.7
    objective_shape: str = "auto"
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    certificate_max_iter: int = 128
    certificate_refinement_rounds: int = 1
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE
    verbose: bool = False
    computation_profile: str = DEFAULT_COMPUTATION_PROFILE

    def __post_init__(self) -> None:
        # Importing policy at the application configuration boundary avoids the
        # former core.model -> model_selection dependency.
        profile = get_computation_profile(self.computation_profile)
        self.computation_profile = profile.name
        contract = _selection_contract(self.selection_contract)
        self.selection_contract = contract.contract_id
        self.selection_dirichlet_alpha = float(
            contract.partition_config.classification_alpha
        )
        self.selection_dirichlet_code_weight = float(
            contract.partition_config.classification_code_weight
        )
        if contract.force_float64:
            self.dtype = "float64"
        self.selection_score = _selection_score(self.selection_score)
        for name in (
            "tol",
            "eps",
            "selection_partition_tol",
            "selection_refit_tol",
            "certificate_column_tol_scale",
            "selection_dirichlet_alpha",
        ):
            _positive_finite(name, getattr(self, name))
        if not math.isfinite(float(self.lambda_value)) or float(self.lambda_value) < 0.0:
            raise ValueError("lambda_value must be finite and nonnegative.")
        if not 0.0 < float(self.major_prior) < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        if int(self.selection_refit_max_iter) < 1:
            raise ValueError("selection_refit_max_iter must be positive.")
        weight = float(self.selection_dirichlet_code_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "selection_dirichlet_code_weight must be nonnegative and finite."
            )

    @property
    def config(self) -> FitConfig:
        """Losslessly group the legacy fields into immutable typed sections."""

        profile = get_computation_profile(self.computation_profile)
        runtime = RuntimeConfig(
            device=str(self.device),
            dtype=str(self.dtype),
            fallback=str(self.dense_fallback_policy),
        )
        resources = ResourceConfig(
            workset_max_bytes=int(self.workset_max_bytes),
            compressed_cache_max_bytes=int(self.compressed_cache_max_bytes),
            workset_add_batch=int(self.workset_add_batch),
            workset_max_expansions=int(self.workset_max_expansions),
        )
        certificate = CertificateConfig(
            max_iter=int(self.certificate_max_iter),
            refinement_rounds=int(self.certificate_refinement_rounds),
            column_tolerance_scale=float(self.certificate_column_tol_scale),
        )
        solver = SolverConfig(
            outer_max_iter=int(self.outer_max_iter),
            inner_max_iter=int(self.inner_max_iter),
            tolerance=float(self.tol),
            objective_shape=str(self.objective_shape),
            certificate=certificate,
            resources=resources,
        )
        refit = RefitConfig(
            tolerance=float(self.selection_refit_tol),
            max_iter=int(self.selection_refit_max_iter),
            mode=str(profile.scalar_mode),
            grid_points=int(profile.scalar_grid_points),
            local_steps=int(profile.scalar_local_steps),
        )
        lambda_search = LambdaSearchConfig(
            exploration_budget=int(profile.lambda_budget),
            refinement_budget=int(profile.lambda_refinement_budget),
            solver_retry_limit=int(profile.solver_retry_limit),
        )
        selection = SelectionConfig(
            score=str(self.selection_score),
            contract_id=str(self.selection_contract),
            partition_tolerance=float(self.selection_partition_tol),
            dirichlet_alpha=float(self.selection_dirichlet_alpha),
            dirichlet_code_weight=float(self.selection_dirichlet_code_weight),
            refit=refit,
            lambda_search=lambda_search,
        )
        return FitConfig(
            lambda_value=float(self.lambda_value),
            major_prior=float(self.major_prior),
            eps=float(self.eps),
            runtime=runtime,
            solver=solver,
            selection=selection,
            graph=self.graph,
            adaptive_weight_gamma=float(self.adaptive_weight_gamma),
            adaptive_weight_floor=float(self.adaptive_weight_floor),
            adaptive_weight_baseline=float(self.adaptive_weight_baseline),
            summary_tol=self.summary_tol,
            verbose=bool(self.verbose),
            computation_profile=str(self.computation_profile),
        )

    def to_config(self) -> FitConfig:
        """Named equivalent of :attr:`config` for serialization boundaries."""

        return self.config

    @classmethod
    def from_config(cls, config: FitConfig) -> FitOptions:
        """Flatten a canonical config for legacy solver and runner callers."""

        solver = config.solver
        resources = solver.resources
        certificate = solver.certificate
        selection = config.selection
        refit = selection.refit
        return cls(
            lambda_value=float(config.lambda_value),
            outer_max_iter=int(solver.outer_max_iter),
            inner_max_iter=int(solver.inner_max_iter),
            tol=float(solver.tolerance),
            major_prior=float(config.major_prior),
            eps=float(config.eps),
            graph=config.graph,
            adaptive_weight_gamma=float(config.adaptive_weight_gamma),
            adaptive_weight_floor=float(config.adaptive_weight_floor),
            adaptive_weight_baseline=float(config.adaptive_weight_baseline),
            device=str(config.runtime.device),
            dtype=str(config.runtime.dtype),
            summary_tol=config.summary_tol,
            selection_score=str(selection.score),
            selection_partition_tol=float(selection.partition_tolerance),
            selection_refit_tol=float(refit.tolerance),
            selection_refit_max_iter=int(refit.max_iter),
            selection_contract=str(selection.contract_id),
            selection_dirichlet_alpha=float(selection.dirichlet_alpha),
            selection_dirichlet_code_weight=float(selection.dirichlet_code_weight),
            objective_shape=str(solver.objective_shape),
            workset_max_bytes=int(resources.workset_max_bytes),
            compressed_cache_max_bytes=int(resources.compressed_cache_max_bytes),
            dense_fallback_policy=str(config.runtime.fallback),
            workset_add_batch=int(resources.workset_add_batch),
            workset_max_expansions=int(resources.workset_max_expansions),
            certificate_max_iter=int(certificate.max_iter),
            certificate_refinement_rounds=int(certificate.refinement_rounds),
            certificate_column_tol_scale=float(certificate.column_tolerance_scale),
            verbose=bool(config.verbose),
            computation_profile=str(config.computation_profile),
        )


__all__ = [
    "CertificateConfig",
    "FitConfig",
    "FitOptions",
    "LambdaSearchConfig",
    "RefitConfig",
    "ResourceConfig",
    "RuntimeConfig",
    "SelectionConfig",
    "SolverConfig",
]
