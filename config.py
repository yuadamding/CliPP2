"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import struct
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

DenseFallbackPolicy: TypeAlias = Literal["device_only", "cpu_allowed", "error"]

# Fixed model identifiers shared by input, inference, reporting and simulation.
CLONAL_INTEGER_MODEL_ID = "clipp2_clonal_integer_multiplicity_mixture_v1"
CLONAL_INTEGER_GENERATOR_VERSION = "integer_1_to_major_cap6_v1"
CLONAL_INTEGER_PRIOR_MODE = "uniform_distinct_integer_v1"
MAX_MAJOR_CN = 6
MULTIPLICITY_POLICIES = ("independent_broad", "shared_broad", "shared_balanced_single")

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


def normalize_runtime_dtype(value: str | None) -> str:
    dtype = "auto" if value is None else str(value).strip().lower()
    if dtype == "auto":
        dtype = DEFAULT_DTYPE
    if dtype not in ("float32", "float64"):
        raise ValueError("Runtime dtype must be float32 or float64; float16 is unsupported.")
    return dtype


@lru_cache(maxsize=64)
def validate_likelihood_precision(eps: float, dtype: str = "float64") -> float:
    """Reject unrepresentable clipping endpoints without changing the objective."""
    epsilon = float(eps)
    if not math.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    name = normalize_runtime_dtype(dtype)
    lower, upper = epsilon, 1.0 - epsilon
    if name == "float32":
        lower, upper = struct.unpack("=ff", struct.pack("=ff", lower, upper))
    if not 0.0 < lower < upper < 1.0:
        raise ValueError(
            f"eps={epsilon:g} is incompatible with {name}: clipping endpoints "
            "must remain strictly inside (0, 1). Choose a representable eps "
            "or use float64 before model preparation; epsilon is never adjusted."
        )
    return epsilon


def normalize_dense_fallback_policy(value: str) -> DenseFallbackPolicy:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = DEFAULT_DENSE_FALLBACK_POLICY
    if normalized not in DENSE_FALLBACK_POLICIES:
        raise ValueError(
            "dense_fallback_policy must be device_only, cpu_allowed, or error."
        )
    return cast(DenseFallbackPolicy, normalized)


ProfileName: TypeAlias = Literal["strict", "balanced", "fast"]


@dataclass(frozen=True, slots=True)
class ComputationProfile:
    """Resource policy and statistical contract for one tumor fit.

    ``strict`` is the reference implementation. ``balanced`` and ``fast``
    deliberately trade stronger refit certification for bounded single-tumor
    latency; their output provenance remains explicit.
    """

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

    @property
    def is_strict(self) -> bool:
        return self.name == "strict"

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

# For a single tumor, the complete graph is retained in balanced mode to avoid
# silently changing the estimator.  The dominant latency reductions come from
# approximate scalar refits and a shorter lambda path.
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
    solver_tolerance=8e-4,
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
        raise ValueError(
            f"computation_profile must be one of: {allowed}."
        ) from error



SELECTION_CONTRACT_ID = "hybrid-ward-cem-v1"
SELECTION_SCORE = "fixed_partition_dirichlet_score"
DIRICHLET_ALPHA = 1.0
DIRICHLET_CODE_WEIGHT = 0.7
PARTITION_K_ANCHORS = (*range(1, 16), 20, 25, 30, 40, 50)
PARTITION_MAX_CANDIDATES_PER_K = 5
PARTITION_CEM_MAX_ITER = 8
PARTITION_GENERATION_REFIT_MAX_ITER = 32
FINAL_PHI_LADDER_KMAX = 30
FINAL_PHI_PARENT_COUNT = 1
PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT = 1.05
LIKELIHOOD_PARTITION_K_MAX = 50

if TYPE_CHECKING:
    from .core.fusion.types import PairwiseFusionGraph


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = DEFAULT_DEVICE
    dtype: str = "float32"
    fallback: str = DEFAULT_DENSE_FALLBACK_POLICY
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_runtime_dtype(self.dtype))
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

    partition_tolerance: float
    refit: RefitConfig
    lambda_search: LambdaSearchConfig

    def __post_init__(self) -> None:
        _positive("selection_partition_tol", self.partition_tolerance)

    @property
    def score(self) -> str:
        return SELECTION_SCORE

    @property
    def graph_pilot_source(self) -> str:
        return "zero_penalty_pilot"

    @property
    def contract_id(self) -> str:
        return SELECTION_CONTRACT_ID

    @property
    def dirichlet_alpha(self) -> float:
        return DIRICHLET_ALPHA

    @property
    def dirichlet_code_weight(self) -> float:
        return DIRICHLET_CODE_WEIGHT


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
    eps: float
    runtime: RuntimeConfig
    solver: SolverConfig
    selection: SelectionConfig
    graph: GraphConfig
    computation_profile: ComputationProfile
    multiplicity_policy: str = "independent_broad"

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.lambda_value)) or float(self.lambda_value) < 0.0:
            raise ValueError("lambda_value must be finite and nonnegative.")
        validate_likelihood_precision(self.eps, self.runtime.dtype)
        if self.multiplicity_policy not in MULTIPLICITY_POLICIES:
            raise ValueError(f"multiplicity_policy must be one of {MULTIPLICITY_POLICIES}.")


def resolve_fit_config(
    *,
    lambda_value: float = 0.0,
    multiplicity_policy: str = "independent_broad",
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

    profile = get_computation_profile(computation_profile)
    runtime = RuntimeConfig(
        device=str(device),
        dtype=str(dtype or profile.raw_dtype),
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
        multiplicity_policy=multiplicity_policy,
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
