"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
from numbers import Integral
import struct
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .core.fusion.types import PairwiseFusionGraph

# Fixed model identifiers shared by input, inference, reporting and simulation.
CLONAL_INTEGER_MODEL_ID = "clipp2_clonal_integer_multiplicity_mixture_v1"
CLONAL_INTEGER_GENERATOR_VERSION = "integer_1_to_major_v2"
CLONAL_INTEGER_PRIOR_MODE = "uniform_distinct_integer_v1"
DEFAULT_MAX_MAJOR_CN: Final = 4

DEFAULT_DEVICE: Final = "cuda"
DEFAULT_DTYPE: Final = "float32"
DEFAULT_OPTIMIZATION_TOLERANCE: Final = 8e-4
ALGORITHM_ID: Final = "independent_hybrid_balanced_v1"


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


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def validate_max_major_cn(value: int) -> int:
    """Require an explicit positive integer input-eligibility cutoff."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_major_cn must be a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = DEFAULT_DEVICE
    dtype: str = "float32"
    verbose: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", normalize_runtime_dtype(self.dtype))


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
    # recovery solve can then be admitted against the immutable production gate
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
    # Internal frozen-graph identity; never accepted by the public FitConfig.
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Public execution and CN eligibility options; numerical policy is fixed."""

    device: str = DEFAULT_DEVICE
    verbose: bool = False
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be a boolean.")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))


@dataclass(frozen=True, slots=True)
class _FitOptions:
    """Internal resolved constants, replaceable only for bounded recovery."""

    runtime: RuntimeConfig
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN
    lambda_value: float = 0.0
    eps: float = 1e-6
    solver: SolverConfig = field(default_factory=lambda: SolverConfig(
        outer_max_iter=6, inner_max_iter=25, tolerance=8e-4,
        objective_shape="auto", certificate=CertificateConfig(128, 1),
    ))
    selection: SelectionConfig = field(default_factory=lambda: SelectionConfig(
        partition_tolerance=2e-4,
        refit=RefitConfig(tolerance=1e-5, max_iter=64, grid_points=64, local_steps=3),
        lambda_search=LambdaSearchConfig(8, 2, 1),
    ))
    graph: GraphConfig = field(default_factory=GraphConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))

    @property
    def multiplicity_policy(self) -> str:
        return "independent_broad"


def resolve_fit_config(
    *, device: str = DEFAULT_DEVICE, verbose: bool = False,
    max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, verbose=verbose, max_major_cn=max_major_cn)


def _resolve_fit_options(config: FitConfig) -> _FitOptions:
    if type(config) is not FitConfig:
        raise TypeError("fit_config must be a FitConfig containing device, verbose and max_major_cn.")
    options = _FitOptions(
        runtime=RuntimeConfig(device=config.device, verbose=config.verbose),
        max_major_cn=config.max_major_cn,
    )
    validate_likelihood_precision(options.eps, options.runtime.dtype)
    return options


__all__ = ["FitConfig", "resolve_fit_config"]
