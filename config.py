"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Mapping, TypeAlias, cast

if TYPE_CHECKING:
    from .core.fusion.types import PairwiseFusionGraph


DenseFallbackPolicy: TypeAlias = Literal["device_only", "cpu_allowed", "error"]
ProfileName: TypeAlias = Literal["strict", "balanced", "fast"]
FailurePolicy: TypeAlias = Literal["error", "save-diagnostics", "best-effort"]
UnsupportedPolicy: TypeAlias = Literal["error", "mask"]

DEFAULT_DEVICE: Final = "cuda"
DEFAULT_DTYPE: Final = "float32"
DEFAULT_OPTIMIZATION_TOLERANCE: Final = 8e-4
DEFAULT_DENSE_FALLBACK_POLICY: Final[DenseFallbackPolicy] = "device_only"
DENSE_FALLBACK_POLICIES: Final = ("device_only", "cpu_allowed", "error")

DEFAULT_WORKSET_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_COMPRESSED_CACHE_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_WORKSET_ADD_BATCH: Final = 64
DEFAULT_WORKSET_MAX_EXPANSIONS: Final = 16
DEFAULT_CERTIFICATE_MAX_ITER: Final = 512
DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS: Final = 2
DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE: Final = 1.0
DEFAULT_DOSAGE_PRIOR_PENALTY: Final = 3.0
FAILURE_POLICIES: Final[tuple[FailurePolicy, ...]] = (
    "error",
    "save-diagnostics",
    "best-effort",
)
DEFAULT_FAILURE_POLICY: Final[FailurePolicy] = "best-effort"

# Selection constants are estimator definitions rather than tunable policy.
PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT: Final = 1.05
LIKELIHOOD_PARTITION_K_MAX: Final = 50


@dataclass(frozen=True)
class PartitionCandidateConfig:
    """Fixed Ward/CEM proposal-generator contract."""

    k_anchors: tuple[int, ...]
    max_candidates_per_k: int
    cem_max_iter: int
    generation_refit_max_iter: int
    final_phi_ladder_kmax: int


@dataclass(frozen=True)
class SelectionPolicy:
    """Versioned identity of the sole production selection policy."""

    policy_id: Literal["hybrid-ward-cem-bic-v1"]
    graph_pilot_source: Literal["zero_penalty_pilot"]
    partition_config: PartitionCandidateConfig

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


PRODUCTION_SELECTION_POLICY: Final = SelectionPolicy(
    policy_id="hybrid-ward-cem-bic-v1",
    graph_pilot_source="zero_penalty_pilot",
    partition_config=PartitionCandidateConfig(
        k_anchors=(*range(1, 16), 20, 25, 30, 40, 50),
        max_candidates_per_k=5,
        cem_max_iter=8,
        generation_refit_max_iter=32,
        final_phi_ladder_kmax=30,
    ),
)


def normalize_dense_fallback_policy(value: str) -> DenseFallbackPolicy:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = DEFAULT_DENSE_FALLBACK_POLICY
    if normalized not in DENSE_FALLBACK_POLICIES:
        raise ValueError(
            "dense_fallback_policy must be device_only, cpu_allowed, or error."
        )
    return cast(DenseFallbackPolicy, normalized)


def normalize_failure_policy(value: str) -> FailurePolicy:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in FAILURE_POLICIES:
        choices = ", ".join(FAILURE_POLICIES)
        raise ValueError(f"failure_policy must be one of: {choices}")
    return cast(FailurePolicy, normalized)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Resolved non-inferential controls for one tumor run."""

    unsupported_policy: UnsupportedPolicy = "error"
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY
    use_warm_starts: bool = True
    write_outputs: bool = True
    failure_policy: FailurePolicy = DEFAULT_FAILURE_POLICY

    def __post_init__(self) -> None:
        unsupported = self.unsupported_policy
        if not isinstance(unsupported, str) or unsupported not in {"error", "mask"}:
            raise ValueError("unsupported_policy must be error or mask.")
        penalty = self.dosage_prior_penalty
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
            raise ValueError("dosage_prior_penalty must be numeric.")
        if not math.isfinite(float(penalty)) or float(penalty) < 0.0:
            raise ValueError("dosage_prior_penalty must be finite and nonnegative.")
        for name in ("use_warm_starts", "write_outputs"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        object.__setattr__(
            self,
            "unsupported_policy",
            cast(UnsupportedPolicy, unsupported),
        )
        object.__setattr__(self, "dosage_prior_penalty", float(penalty))
        object.__setattr__(
            self,
            "failure_policy",
            normalize_failure_policy(str(self.failure_policy)),
        )


@dataclass(frozen=True, slots=True)
class CheckpointRequest:
    """One normalized request to disable, create, or resume a checkpoint."""

    path: Path | str | None = None
    enabled: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.resume, bool):
            raise ValueError("checkpoint enabled and resume controls must be boolean.")
        path = None if self.path is None else Path(self.path)
        enabled = bool(self.enabled or self.resume or path is not None)
        if self.resume and path is None:
            raise ValueError("A resumed checkpoint requires an explicit path.")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "enabled", enabled)

    def resolve_path(self, *, outdir: Path, tumor_id: str) -> Path | None:
        if self.path is not None:
            return Path(self.path)
        if not self.enabled:
            return None
        return Path(outdir) / ".clipp2-checkpoints" / f"{tumor_id}.checkpoint"


DEFAULT_RUN_CONFIG: Final = RunConfig()
DEFAULT_CHECKPOINT_REQUEST: Final = CheckpointRequest()
_RUN_CONFIG_FIELDS: Final = frozenset(
    {
        "unsupported_policy",
        "dosage_prior_penalty",
        "use_warm_starts",
        "write_outputs",
        "failure_policy",
    }
)


def resolve_run_config_mapping(values: Mapping[str, object]) -> RunConfig:
    """Resolve strict JSON runner fields against the canonical defaults."""

    unknown = set(values) - _RUN_CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "Unknown runner configuration field(s): "
            + ", ".join(sorted(unknown))
        )
    return RunConfig(**dict(values))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ComputationProfile:
    """Named defaults compiled into a resolved :class:`FitConfig`."""

    name: ProfileName
    raw_dtype: Literal["float32", "float64"]
    scalar_mode: Literal["interval_certified", "grid_local"]
    scalar_grid_points: int
    scalar_local_steps: int
    lambda_budget: int
    lambda_refinement_budget: int
    outer_max_iter: int
    inner_max_iter: int
    solver_tolerance: float
    solver_retry_limit: int
    partition_tolerance: float
    refit_tolerance: float
    refit_max_iter: int
    certificate_max_iter: int
    certificate_refinement_rounds: int


STRICT_PROFILE: Final = ComputationProfile(
    name="strict",
    raw_dtype="float64",
    scalar_mode="interval_certified",
    scalar_grid_points=17,
    scalar_local_steps=0,
    lambda_budget=12,
    lambda_refinement_budget=12,
    outer_max_iter=8,
    inner_max_iter=30,
    solver_tolerance=5e-5,
    solver_retry_limit=4,
    partition_tolerance=1e-4,
    refit_tolerance=1e-7,
    refit_max_iter=128,
    certificate_max_iter=512,
    certificate_refinement_rounds=2,
)

BALANCED_PROFILE: Final = ComputationProfile(
    name="balanced",
    raw_dtype="float32",
    scalar_mode="grid_local",
    scalar_grid_points=64,
    scalar_local_steps=3,
    lambda_budget=8,
    lambda_refinement_budget=2,
    outer_max_iter=6,
    inner_max_iter=25,
    solver_tolerance=DEFAULT_OPTIMIZATION_TOLERANCE,
    solver_retry_limit=1,
    partition_tolerance=2e-4,
    refit_tolerance=1e-5,
    refit_max_iter=64,
    certificate_max_iter=128,
    certificate_refinement_rounds=1,
)

FAST_PROFILE: Final = ComputationProfile(
    name="fast",
    raw_dtype="float32",
    scalar_mode="grid_local",
    scalar_grid_points=32,
    scalar_local_steps=1,
    lambda_budget=6,
    lambda_refinement_budget=0,
    outer_max_iter=4,
    inner_max_iter=16,
    solver_tolerance=1e-3,
    solver_retry_limit=0,
    partition_tolerance=1e-3,
    refit_tolerance=1e-4,
    refit_max_iter=32,
    certificate_max_iter=64,
    certificate_refinement_rounds=0,
)

COMPUTATION_PROFILES: Final = {
    profile.name: profile
    for profile in (STRICT_PROFILE, BALANCED_PROFILE, FAST_PROFILE)
}
COMPUTATION_PROFILE_NAMES: Final = tuple(COMPUTATION_PROFILES)
DEFAULT_COMPUTATION_PROFILE: Final[ProfileName] = "balanced"


def get_computation_profile(value: str) -> ComputationProfile:
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return COMPUTATION_PROFILES[cast(ProfileName, normalized)]
    except KeyError as error:
        allowed = ", ".join(COMPUTATION_PROFILE_NAMES)
        raise ValueError(f"computation_profile must be one of: {allowed}.") from error


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    fallback: str = DEFAULT_DENSE_FALLBACK_POLICY
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fallback", normalize_dense_fallback_policy(self.fallback)
        )


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
    # Hardware-independent, cumulative raw-solver budget.  None deliberately
    # preserves an unlimited search until a profile-specific cap is calibrated.
    max_tumor_edge_pass_equivalents: int | None = None
    # Cumulative post-mandatory-guide fixed-partition scalar objective
    # evaluations.  The complete guide defines the graph/objective and is
    # therefore accounted and reported separately rather than truncated.
    # Enforcement is at candidate boundaries because a certified coordinate
    # minimization is atomic and its realized interval count is not known in
    # advance.
    max_partition_refit_objective_evaluations: int | None = None
    # Exact boundary on the number of direct Ward/CEM candidates evaluated.
    max_direct_partition_candidates: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_tumor_edge_pass_equivalents",
            "max_partition_refit_objective_evaluations",
            "max_direct_partition_candidates",
        ):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when set.")


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
    stagnation_audit_patience: int = 4

    def __post_init__(self) -> None:
        _positive("tol", self.tolerance)
        if self.certification_tolerance is not None:
            _positive("certification_tolerance", self.certification_tolerance)
        if int(self.stagnation_audit_patience) < 1:
            raise ValueError("stagnation_audit_patience must be positive.")


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
    no_progress_patience: int = 3

    def __post_init__(self) -> None:
        if int(self.exploration_budget) < 1:
            raise ValueError("exploration_budget must be positive.")
        if int(self.refinement_budget) < 0:
            raise ValueError("refinement_budget must be nonnegative.")
        if int(self.solver_retry_limit) < 0:
            raise ValueError("solver_retry_limit must be nonnegative.")
        if int(self.no_progress_patience) < 1:
            raise ValueError("no_progress_patience must be positive.")


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Numerical settings for the immutable production selection policy."""

    partition_tolerance: float
    refit: RefitConfig
    lambda_search: LambdaSearchConfig

    def __post_init__(self) -> None:
        _positive("selection_partition_tol", self.partition_tolerance)


@dataclass(frozen=True, slots=True)
class GraphConfig:
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Canonical boundary consumed by the solver, selector, and serializers."""

    major_prior: float
    eps: float
    runtime: RuntimeConfig
    solver: SolverConfig
    selection: SelectionConfig
    graph: GraphConfig
    profile_name: ProfileName

    def __post_init__(self) -> None:
        if not 0.0 < float(self.major_prior) < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        _positive("eps", self.eps)
        object.__setattr__(
            self, "profile_name", get_computation_profile(self.profile_name).name
        )


def _resolve_fit_config_values(
    *,
    computation_profile: str = DEFAULT_COMPUTATION_PROFILE,
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
    max_tumor_edge_pass_equivalents: int | None = None,
    max_partition_refit_objective_evaluations: int | None = None,
    max_direct_partition_candidates: int | None = None,
    stagnation_audit_patience: int = 4,
    lambda_no_progress_patience: int = 3,
    verbose: bool = False,
) -> FitConfig:
    """Resolve a profile once into the production fit contract."""

    profile = get_computation_profile(computation_profile)
    runtime = RuntimeConfig(
        device=str(device),
        dtype=str(dtype or profile.raw_dtype),
        fallback=str(dense_fallback_policy),
        verbose=bool(verbose),
    )
    certificate = CertificateConfig(
        max_iter=int(
            profile.certificate_max_iter
            if certificate_max_iter is None
            else certificate_max_iter
        ),
        refinement_rounds=int(
            profile.certificate_refinement_rounds
            if certificate_refinement_rounds is None
            else certificate_refinement_rounds
        ),
        column_tolerance_scale=float(certificate_column_tol_scale),
    )
    solver = SolverConfig(
        outer_max_iter=int(
            profile.outer_max_iter if outer_max_iter is None else outer_max_iter
        ),
        inner_max_iter=int(
            profile.inner_max_iter if inner_max_iter is None else inner_max_iter
        ),
        tolerance=float(profile.solver_tolerance if tol is None else tol),
        objective_shape=str(objective_shape),
        stagnation_audit_patience=int(stagnation_audit_patience),
        certificate=certificate,
        resources=ResourceConfig(
            workset_max_bytes=int(workset_max_bytes),
            compressed_cache_max_bytes=int(compressed_cache_max_bytes),
            workset_add_batch=int(workset_add_batch),
            workset_max_expansions=int(workset_max_expansions),
            max_tumor_edge_pass_equivalents=(
                None
                if max_tumor_edge_pass_equivalents is None
                else int(max_tumor_edge_pass_equivalents)
            ),
            max_partition_refit_objective_evaluations=(
                None
                if max_partition_refit_objective_evaluations is None
                else int(max_partition_refit_objective_evaluations)
            ),
            max_direct_partition_candidates=(
                None
                if max_direct_partition_candidates is None
                else int(max_direct_partition_candidates)
            ),
        ),
    )
    selection = SelectionConfig(
        partition_tolerance=float(
            profile.partition_tolerance
            if selection_partition_tol is None
            else selection_partition_tol
        ),
        refit=RefitConfig(
            tolerance=float(
                profile.refit_tolerance
                if selection_refit_tol is None
                else selection_refit_tol
            ),
            max_iter=int(
                profile.refit_max_iter
                if selection_refit_max_iter is None
                else selection_refit_max_iter
            ),
            mode=str(profile.scalar_mode),
            grid_points=int(profile.scalar_grid_points),
            local_steps=int(profile.scalar_local_steps),
        ),
        lambda_search=LambdaSearchConfig(
            exploration_budget=int(profile.lambda_budget),
            refinement_budget=int(profile.lambda_refinement_budget),
            solver_retry_limit=int(profile.solver_retry_limit),
            no_progress_patience=int(lambda_no_progress_patience),
        ),
    )
    return FitConfig(
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
        profile_name=profile.name,
    )


def resolve_fit_config(**values: object) -> FitConfig:
    """Resolve strict expert overrides into the canonical fit configuration.

    The public boundary is one option mapping (spelled as keyword arguments
    for ergonomic Python use).  Profile expansion and all validation still
    happen exactly once in this module.
    """

    allowed = {
        "adaptive_weight_baseline",
        "adaptive_weight_floor",
        "adaptive_weight_gamma",
        "certificate_column_tol_scale",
        "certificate_max_iter",
        "certificate_refinement_rounds",
        "compressed_cache_max_bytes",
        "computation_profile",
        "dense_fallback_policy",
        "device",
        "dtype",
        "eps",
        "graph",
        "inner_max_iter",
        "lambda_no_progress_patience",
        "major_prior",
        "max_direct_partition_candidates",
        "max_partition_refit_objective_evaluations",
        "max_tumor_edge_pass_equivalents",
        "objective_shape",
        "outer_max_iter",
        "selection_partition_tol",
        "selection_refit_max_iter",
        "selection_refit_tol",
        "stagnation_audit_patience",
        "tol",
        "verbose",
        "workset_add_batch",
        "workset_max_bytes",
        "workset_max_expansions",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            "Unknown fit configuration field(s): " + ", ".join(sorted(unknown))
        )
    return _resolve_fit_config_values(**values)  # type: ignore[arg-type]


__all__ = [
    "BALANCED_PROFILE",
    "COMPUTATION_PROFILE_NAMES",
    "CertificateConfig",
    "CheckpointRequest",
    "ComputationProfile",
    "DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE",
    "DEFAULT_CERTIFICATE_MAX_ITER",
    "DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS",
    "DEFAULT_COMPRESSED_CACHE_MAX_BYTES",
    "DEFAULT_COMPUTATION_PROFILE",
    "DEFAULT_CHECKPOINT_REQUEST",
    "DEFAULT_DENSE_FALLBACK_POLICY",
    "DEFAULT_DEVICE",
    "DEFAULT_DOSAGE_PRIOR_PENALTY",
    "DEFAULT_DTYPE",
    "DEFAULT_FAILURE_POLICY",
    "DEFAULT_OPTIMIZATION_TOLERANCE",
    "DEFAULT_RUN_CONFIG",
    "DEFAULT_WORKSET_ADD_BATCH",
    "DEFAULT_WORKSET_MAX_BYTES",
    "DEFAULT_WORKSET_MAX_EXPANSIONS",
    "DENSE_FALLBACK_POLICIES",
    "DenseFallbackPolicy",
    "FAILURE_POLICIES",
    "FailurePolicy",
    "FAST_PROFILE",
    "FitConfig",
    "GraphConfig",
    "LIKELIHOOD_PARTITION_K_MAX",
    "LambdaSearchConfig",
    "PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT",
    "PRODUCTION_SELECTION_POLICY",
    "PartitionCandidateConfig",
    "ProfileName",
    "RefitConfig",
    "ResourceConfig",
    "RuntimeConfig",
    "RunConfig",
    "STRICT_PROFILE",
    "SelectionConfig",
    "SelectionPolicy",
    "SolverConfig",
    "UnsupportedPolicy",
    "get_computation_profile",
    "normalize_dense_fallback_policy",
    "normalize_failure_policy",
    "resolve_fit_config",
    "resolve_run_config_mapping",
]
