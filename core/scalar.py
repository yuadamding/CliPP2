"""Shared scalar likelihood problem and bounded minimizers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
from time import perf_counter

import numpy as np

from ..config import validate_likelihood_precision
from ..io.data import ImmutableArrayRecord, TumorData, readonly_array, tumor_data_fingerprint
from .objective import ObservedModel, candidate_terms_numpy, compile_observed_model


@dataclass(frozen=True, slots=True)
class ScalarProblem(ImmutableArrayRecord):
    """One cluster-region slice of a canonical observed model."""

    alt: np.ndarray
    nonalt: np.ndarray
    observed: np.ndarray
    slope: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    lower: float
    upper: float
    eps: float

    def __post_init__(self) -> None:
        alt = np.asarray(self.alt, dtype=np.float64).reshape(-1)
        nonalt = np.asarray(self.nonalt, dtype=np.float64).reshape(-1)
        observed = np.asarray(self.observed, dtype=bool).reshape(-1)
        if alt.shape != nonalt.shape or observed.shape != alt.shape:
            raise ValueError("ScalarProblem observation arrays must have one shape.")
        candidate_shape = (alt.size, np.asarray(self.slope).shape[-1])
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in (
            ("slope", np.float64),
            ("log_prior", np.float64),
            ("valid", bool),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.ndim != 2 or value.shape != candidate_shape:
                raise ValueError(f"ScalarProblem.{name} must have shape {candidate_shape}.")
            arrays[name] = value
        lower = float(self.lower)
        upper = float(self.upper)
        eps = validate_likelihood_precision(self.eps)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError("Require a finite scalar interval with lower <= upper.")
        for name, value in {
            "alt": alt,
            "nonalt": nonalt,
            "observed": observed,
            **arrays,
        }.items():
            object.__setattr__(self, name, readonly_array(value))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "eps", eps)


@dataclass(frozen=True, slots=True)
class ApproximateScalarMinimum:
    argmin: float
    attained_value: float
    grid_points_evaluated: int
    final_grid_spacing: float
    best_second_loss_gap: float
    method: str = "vectorized_grid_local_v1"


@dataclass(frozen=True)
class ScalarGlobalMinimumCertificate:
    argmin: float
    attained_value: float
    global_lower_bound: float
    optimality_gap: float
    globally_certified: bool
    method: str
    intervals_evaluated: int


@dataclass(slots=True)
class _ScalarWorkStats:
    """Ephemeral physical work, not the preserved logical refit counters.

    Solves and evaluations count dispatched work, including a failing call.
    Interval evaluations count lower-bound calls (not fixed-coordinate loss
    evaluations); grid points count candidates sent to the grid loss routine.
    The sink contains no model/source references and is not part of fit outputs.
    """

    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    scalar_solves: int = 0
    scalar_failures: int = 0
    interval_evaluations: int = 0
    grid_points_evaluated: int = 0
    scalar_seconds: float = 0.0


def scalar_problem_from_model(
    model: ObservedModel,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    eps: float,
    respect_observed: bool = True,
) -> ScalarProblem:
    if model.coupling == "joint" and model.shape[1] > 1:
        raise ValueError("A regional ScalarProblem is not the joint observed likelihood; use a joint refit.")
    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    region = int(region_index)
    alt = model.alt[rows, region]
    nonalt = model.nonalt[rows, region]
    return ScalarProblem(
        alt=alt,
        nonalt=nonalt,
        observed=(
            (model.observed[rows, region] if respect_observed else True)
            & ((alt + nonalt) > 0.0)
        ),
        slope=model.slope[rows, region],
        log_prior=model.log_prior[rows, region],
        valid=model.valid[rows, region],
        lower=lower,
        upper=upper,
        eps=eps,
    )


def scalar_loss(
    problem: ScalarProblem, beta: float | np.ndarray
) -> float | np.ndarray:
    loss, _ = _scalar_terms(problem, beta, with_gradient=False)
    return loss


def scalar_loss_and_gradient(
    problem: ScalarProblem, beta: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Evaluate the canonical scalar loss and its left derivative."""

    loss, gradient = _scalar_terms(problem, beta, with_gradient=True)
    assert gradient is not None
    return loss, gradient


def _scalar_terms(
    problem: ScalarProblem,
    beta: float | np.ndarray,
    *,
    with_gradient: bool,
) -> tuple[float | np.ndarray, float | np.ndarray | None]:
    values = np.asarray(beta, dtype=np.float64)
    scalar = values.ndim == 0
    flat = values.reshape(-1)
    active = problem.observed
    if not np.any(active):
        loss = np.zeros(flat.size, dtype=np.float64)
        gradient = np.zeros_like(loss) if with_gradient else None
    else:
        candidate = flat[None, :, None]
        candidate_slope = problem.slope[active, None, :]
        mass = candidate_slope * candidate
        probability = np.clip(mass, problem.eps, 1.0 - problem.eps)
        valid = problem.valid[active, None, :]
        slope = np.where(
            (mass > problem.eps) & (mass < 1.0 - problem.eps), candidate_slope, 0.0,
        ) if with_gradient else None
        log_kernel, state_score, _ = candidate_terms_numpy(
            problem.alt[active, None, None], problem.nonalt[active, None, None],
            probability, slope, derivative_order=int(with_gradient),
        )
        joint = log_kernel + problem.log_prior[active, None, :]
        del log_kernel
        joint = np.where(valid, joint, -np.inf)
        log_normalizer = np.logaddexp.reduce(joint, axis=-1)
        loss = -np.sum(log_normalizer, axis=0, dtype=np.float64)
        gradient = None
        if with_gradient:
            posterior = np.where(
                valid,
                np.exp(joint - log_normalizer[..., None]),
                0.0,
            )
            gradient = -np.sum(
                posterior * state_score,
                axis=(0, 2),
                dtype=np.float64,
            )
    loss_out: float | np.ndarray = (
        float(loss[0]) if scalar else loss.reshape(values.shape)
    )
    if gradient is None:
        return loss_out, None
    gradient_out: float | np.ndarray = (
        float(gradient[0]) if scalar else gradient.reshape(values.shape)
    )
    return loss_out, gradient_out


def scalar_breakpoints(
    problem: ScalarProblem, *, observed_only: bool = True
) -> np.ndarray:
    points = [problem.lower, problem.upper]
    rows = np.flatnonzero(problem.observed) if observed_only else range(problem.alt.size)
    for row in rows:
        for candidate in np.flatnonzero(problem.valid[row]):
            slope = float(problem.slope[row, candidate])
            for target in (problem.eps, 1.0 - problem.eps):
                if slope > 0.0:
                    value = target / slope
                    if problem.lower < value < problem.upper:
                        points.append(value)
    return np.unique(
        np.clip(np.asarray(points, dtype=np.float64), problem.lower, problem.upper)
    )


def approximate_scalar_minimum(
    problem: ScalarProblem,
    *,
    grid_points: int,
    local_steps: int,
    hint: float | None = None,
    include_breakpoints: bool = True,
    _work_stats: _ScalarWorkStats | None = None,
) -> ApproximateScalarMinimum:
    """Deterministic bounded grid search with local bracket refinement."""

    if int(grid_points) < 3:
        raise ValueError("grid_points must be at least three.")
    if int(local_steps) < 0:
        raise ValueError("local_steps must be nonnegative.")
    initial = np.linspace(
        problem.lower, problem.upper, num=int(grid_points), dtype=np.float64
    )
    extras = (
        scalar_breakpoints(problem, observed_only=False)
        if include_breakpoints
        else np.empty(0, dtype=np.float64)
    )
    if hint is not None and np.isfinite(float(hint)):
        extras = np.append(extras, np.clip(float(hint), problem.lower, problem.upper))
    grid = np.unique(
        np.clip(
            np.concatenate((initial, extras)), problem.lower, problem.upper
        )
    )
    evaluated: dict[float, float] = {}

    def evaluate(candidates: np.ndarray) -> None:
        unique = np.asarray(
            [float(value) for value in candidates if float(value) not in evaluated],
            dtype=np.float64,
        )
        if unique.size:
            if _work_stats is not None:
                _work_stats.grid_points_evaluated += int(unique.size)
            evaluated.update(
                zip(unique.tolist(), np.asarray(scalar_loss(problem, unique)).tolist())
            )

    evaluate(grid)
    final_spacing = float(problem.upper - problem.lower)
    for _ in range(int(local_steps)):
        ordered = np.asarray(sorted(evaluated), dtype=np.float64)
        losses = np.asarray([evaluated[float(value)] for value in ordered])
        best = int(np.nanargmin(losses))
        left = float(ordered[max(best - 1, 0)])
        right = float(ordered[min(best + 1, ordered.size - 1)])
        if right <= left:
            break
        final_spacing = float((right - left) / 4.0)
        evaluate(np.linspace(left, right, num=5, dtype=np.float64))
    ordered = np.asarray(sorted(evaluated), dtype=np.float64)
    losses = np.asarray([evaluated[float(value)] for value in ordered])
    finite = np.flatnonzero(np.isfinite(losses))
    if not finite.size:
        fallback = hint if hint is not None else problem.lower
        return ApproximateScalarMinimum(
            argmin=float(np.clip(fallback, problem.lower, problem.upper)),
            attained_value=float("inf"),
            grid_points_evaluated=len(evaluated),
            final_grid_spacing=final_spacing,
            best_second_loss_gap=float("inf"),
        )
    ranked = finite[np.argsort(losses[finite], kind="stable")]
    best = int(ranked[0])
    gap = (
        float(losses[int(ranked[1])] - losses[best])
        if ranked.size > 1
        else float("inf")
    )
    return ApproximateScalarMinimum(
        argmin=float(ordered[best]),
        attained_value=float(losses[best]),
        grid_points_evaluated=len(evaluated),
        final_grid_spacing=final_spacing,
        best_second_loss_gap=max(gap, 0.0),
    )


def _active_candidate_arrays(problem: ScalarProblem) -> tuple[np.ndarray, ...]:
    active = problem.observed
    return (
        problem.alt[active],
        problem.nonalt[active],
        problem.slope[active],
        problem.log_prior[active],
        problem.valid[active],
    )


def _interval_lower_bound(problem: ScalarProblem, left: float, right: float) -> float:
    alt, nonalt, candidate_slope, log_prior, valid = _active_candidate_arrays(
        problem
    )
    mass_left = float(left) * candidate_slope
    mass_right = float(right) * candidate_slope
    probability_left = np.clip(mass_left, problem.eps, 1.0 - problem.eps)
    probability_right = np.clip(mass_right, problem.eps, 1.0 - problem.eps)
    probability_min = np.minimum(probability_left, probability_right)
    probability_max = np.maximum(probability_left, probability_right)
    total = alt + nonalt
    with np.errstate(divide="ignore", invalid="ignore"):
        empirical = np.divide(
            alt,
            total,
            out=np.full_like(alt, 0.5, dtype=np.float64),
            where=total > 0.0,
        )
    mode = np.clip(empirical[:, None], probability_min, probability_max)
    component_upper = np.where(
        valid,
        alt[:, None] * np.log(mode)
        + nonalt[:, None] * np.log1p(-mode)
        + log_prior,
        -np.inf,
    )
    maximum = np.max(component_upper, axis=1)
    mixture_upper = maximum + np.log(
        np.sum(np.exp(component_upper - maximum[:, None]), axis=1)
    )
    component_bound = float(-np.sum(mixture_upper))

    # A budget-coarsened initial interval can straddle a derivative kink.
    # The component envelope remains valid there, but a midpoint Taylor bound
    # based on one branch's slopes does not bound the entire interval.
    crosses_kink = np.zeros_like(valid)
    for threshold in (problem.eps, 1.0 - problem.eps):
        crosses_kink |= (mass_left < threshold) & (threshold < mass_right)
    if np.any(valid & crosses_kink):
        return float(np.nextafter(component_bound, -np.inf))

    midpoint = float(left + 0.5 * (right - left))
    half_width = float(0.5 * (right - left))
    raw_mass = midpoint * candidate_slope
    probability = np.clip(raw_mass, problem.eps, 1.0 - problem.eps)
    slope = np.where(
        (raw_mass > problem.eps) & (raw_mass < 1.0 - problem.eps),
        candidate_slope,
        0.0,
    )
    joint = np.where(
        valid,
        alt[:, None] * np.log(probability)
        + nonalt[:, None] * np.log1p(-probability)
        + log_prior,
        -np.inf,
    )
    maximum = np.max(joint, axis=1)
    weights = np.where(valid, np.exp(joint - maximum[:, None]), 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    state_score = slope * (
        alt[:, None] / probability - nonalt[:, None] / (1.0 - probability)
    )
    loss_gradient = -float(np.sum(weights * state_score))
    score_bound = np.where(
        valid,
        slope
        * (
            alt[:, None] / probability_min
            + nonalt[:, None] / (1.0 - probability_max)
        ),
        0.0,
    )
    curvature_bound = np.where(
        valid,
        slope * slope
        * (
            alt[:, None] / np.square(probability_min)
            + nonalt[:, None] / np.square(1.0 - probability_max)
        ),
        0.0,
    )
    hessian_bound = float(
        np.sum(
            np.max(curvature_bound, axis=1) + np.square(np.max(score_bound, axis=1))
        )
    )
    taylor_bound = (
        float(scalar_loss(problem, midpoint))
        - abs(loss_gradient) * half_width
        - 0.5 * hessian_bound * half_width * half_width
    )
    return float(np.nextafter(max(component_bound, taylor_bound), -np.inf))


def certify_scalar_minimum(
    problem: ScalarProblem,
    *,
    tolerance: float,
    max_intervals: int,
    hint: float | None = None,
    _work_stats: _ScalarWorkStats | None = None,
) -> ScalarGlobalMinimumCertificate:
    """Certify the global scalar minimum, or return a valid unresolved bound."""

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Scalar certification tolerance must be positive and finite.")
    if int(max_intervals) < 1:
        raise ValueError("max_intervals must be positive.")
    if not np.any(problem.observed):
        beta = float(0.5 * (problem.lower + problem.upper))
        return ScalarGlobalMinimumCertificate(
            beta, 0.0, 0.0, 0.0, True, "interval_binomial_mixture_bound_v1", 0
        )
    if problem.upper <= problem.lower:
        loss = float(scalar_loss(problem, problem.lower))
        return ScalarGlobalMinimumCertificate(
            problem.lower,
            loss,
            loss,
            0.0,
            bool(np.isfinite(loss)),
            "fixed_scalar_coordinate_v1",
            1,
        )
    points = np.concatenate(
        (
            np.linspace(problem.lower, problem.upper, num=17, dtype=np.float64),
            scalar_breakpoints(problem),
        )
    )
    if hint is not None and np.isfinite(float(hint)):
        points = np.append(points, np.clip(float(hint), problem.lower, problem.upper))
    points = np.unique(np.clip(points, problem.lower, problem.upper))
    if points.size - 1 > int(max_intervals):
        # Keep both endpoints and full interval coverage, never an incomplete
        # prefix of the initial partition. Kink-crossing bounds stay conservative.
        indices = np.linspace(0, points.size - 1, int(max_intervals) + 1, dtype=int)
        points = points[indices]
    best_beta = float(points[0])
    best_value = float("inf")

    def consider(beta: float) -> None:
        nonlocal best_beta, best_value
        value = float(scalar_loss(problem, beta))
        tie = tolerance * 0.25
        if value < best_value - tie or (
            abs(value - best_value) <= tie and beta < best_beta
        ):
            best_beta, best_value = beta, value

    for value in points:
        consider(float(value))
    if hint is not None and np.isfinite(float(hint)):
        consider(float(np.clip(hint, problem.lower, problem.upper)))
    heap: list[tuple[float, float, float, int]] = []
    intervals = 0
    serial = 0

    def interval_bound(left: float, right: float) -> float:
        if _work_stats is not None:
            _work_stats.interval_evaluations += 1
        return _interval_lower_bound(problem, left, right)

    for left, right in zip(points[:-1], points[1:]):
        if right > left:
            bound = interval_bound(float(left), float(right))
            heapq.heappush(heap, (bound, float(left), float(right), serial))
            intervals += 1
            serial += 1
    certified = False
    # A split replaces its parent with *both* children. Never consume the last
    # budget slot on only the left child and silently lose the right interval:
    # that would make the remaining heap's lower bound invalid.
    while heap and intervals + 2 <= int(max_intervals):
        lower_bound = min(float(heap[0][0]), best_value)
        if np.isfinite(best_value) and best_value - lower_bound <= tolerance:
            certified = True
            break
        bound, left, right, _ = heapq.heappop(heap)
        if bound > best_value:
            continue
        midpoint = float(left + 0.5 * (right - left))
        if not left < midpoint < right:
            continue
        consider(midpoint)
        for child_left, child_right in ((left, midpoint), (midpoint, right)):
            child_bound = interval_bound(child_left, child_right)
            intervals += 1
            if child_bound <= best_value:
                heapq.heappush(
                    heap, (child_bound, child_left, child_right, serial)
                )
                serial += 1
    lower_bound = min(float(heap[0][0]), best_value) if heap else best_value
    gap = max(float(best_value - lower_bound), 0.0)
    certified = bool(
        certified
        or (
            np.isfinite(best_value)
            and np.isfinite(lower_bound)
            and gap <= tolerance
        )
    )
    return ScalarGlobalMinimumCertificate(
        argmin=float(np.clip(best_beta, problem.lower, problem.upper)),
        attained_value=best_value,
        global_lower_bound=lower_bound,
        optimality_gap=gap,
        globally_certified=certified,
        method="interval_binomial_mixture_bound_v1",
        intervals_evaluated=intervals,
    )


@dataclass(frozen=True)
class PartitionRefitResult:
    """Observed-likelihood refit of one immutable partition."""

    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    fit_loss: float
    n_clusters: int
    boundary_count: int
    active_degrees_of_freedom: int
    finite_candidate_found: bool
    refit_coordinate_count: int
    refit_finite_coordinate_count: int
    refit_total_grid_points: int
    refit_max_grid_spacing: float
    refit_total_candidate_basins: int
    refit_total_refined_candidates: int
    refit_min_best_second_loss_gap: float
    labels: np.ndarray
    loglik_source: str = "partition_constrained_observed_mle"
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_optimum_certified: bool = False
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0
    refit_mode: str = "interval_certified"
    locally_converged: bool = False


@dataclass(frozen=True)
class _RefitCoordinateResult:
    beta: float
    loss: float
    global_lower_bound: float
    optimality_gap: float
    finite_candidate_found: bool
    globally_certified: bool
    certificate_method: str
    certificate_intervals: int
    grid_points: int = 0
    grid_spacing: float = 0.0
    best_second_loss_gap: float = float("inf")


@dataclass(frozen=True, slots=True)
class _RefitCoordinateKey:
    source_hash: str
    model_hash: str
    members: bytes
    region: int
    lower: float
    upper: float
    eps: float
    mode: str
    tolerance: float
    max_iter: int
    grid_points: int
    local_steps: int
    include_breakpoints: bool


class _RefitCoordinateCache:
    """Pool-local LRU of complete scalar results, never model/source arrays.

    Bound both entry overhead and the total variable-sized membership keys.
    Budget-exhausted results are reusable only under the same exact policy;
    their unresolved certificate and logical work counts remain unchanged.
    """

    def __init__(
        self, *, max_entries: int = 1024, max_membership_bytes: int = 8 * 1024**2,
        work_stats: _ScalarWorkStats | None = None,
    ):
        if max_entries < 0 or max_membership_bytes < 0:
            raise ValueError("Scalar cache limits must be nonnegative.")
        self.max_entries = int(max_entries)
        self.max_membership_bytes = int(max_membership_bytes)
        self._membership_bytes = 0
        self._entries: OrderedDict[_RefitCoordinateKey, _RefitCoordinateResult] = OrderedDict()
        self.work = _ScalarWorkStats() if work_stats is None else work_stats

    def get(self, key: _RefitCoordinateKey) -> _RefitCoordinateResult | None:
        result = self._entries.get(key)
        if result is not None:
            self._entries.move_to_end(key)
            self.work.cache_hits += 1
        else:
            self.work.cache_misses += 1
        return result

    def put(self, key: _RefitCoordinateKey, result: _RefitCoordinateResult) -> None:
        size = len(key.members)
        if self.max_entries == 0 or size > self.max_membership_bytes:
            return
        if key in self._entries:
            self._entries.pop(key)
            self._membership_bytes -= size
        self._entries[key] = result
        self._membership_bytes += size
        while (
            len(self._entries) > self.max_entries
            or self._membership_bytes > self.max_membership_bytes
        ):
            oldest, _ = self._entries.popitem(last=False)
            self._membership_bytes -= len(oldest.members)
            self.work.cache_evictions += 1


def canonical_partition_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.size == 0:
        return values.copy()
    remapped = np.empty_like(values)
    mapping: dict[int, int] = {}
    for index, value in enumerate(values):
        remapped[index] = mapping.setdefault(int(value), len(mapping))
    return remapped


def _fit_coordinate(
    problem: ScalarProblem,
    *,
    mode: str,
    tolerance: float,
    max_iter: int,
    grid_points: int,
    local_steps: int,
    include_breakpoints: bool,
    _work_stats: _ScalarWorkStats | None = None,
) -> _RefitCoordinateResult:
    started = perf_counter() if _work_stats is not None else 0.0
    if _work_stats is not None:
        _work_stats.scalar_solves += 1
    try:
        if mode == "interval_certified":
            result = certify_scalar_minimum(
                problem,
                tolerance=tolerance,
                max_intervals=max(int(max_iter) * 256, 4096),
                _work_stats=_work_stats,
            )
            return _RefitCoordinateResult(
                beta=float(result.argmin),
                loss=float(result.attained_value),
                global_lower_bound=float(result.global_lower_bound),
                optimality_gap=float(result.optimality_gap),
                finite_candidate_found=bool(np.isfinite(result.attained_value)),
                globally_certified=bool(result.globally_certified),
                certificate_method=str(result.method),
                certificate_intervals=int(result.intervals_evaluated),
            )
        result = approximate_scalar_minimum(
            problem,
            grid_points=grid_points,
            local_steps=local_steps,
            include_breakpoints=include_breakpoints,
            _work_stats=_work_stats,
        )
        return _RefitCoordinateResult(
            beta=float(result.argmin),
            loss=float(result.attained_value),
            global_lower_bound=float("-inf"),
            optimality_gap=float("inf"),
            finite_candidate_found=bool(np.isfinite(result.attained_value)),
            globally_certified=False,
            certificate_method=str(result.method),
            certificate_intervals=0,
            grid_points=int(result.grid_points_evaluated),
            grid_spacing=float(result.final_grid_spacing),
            best_second_loss_gap=float(result.best_second_loss_gap),
        )
    except Exception:
        if _work_stats is not None:
            _work_stats.scalar_failures += 1
        raise
    finally:
        if _work_stats is not None:
            _work_stats.scalar_seconds += perf_counter() - started


def partition_constrained_observed_refit(
    data: TumorData,
    labels: np.ndarray,
    *,
    eps: float,
    tol: float,
    max_iter: int,
    scalar_mode: str = "interval_certified",
    scalar_grid_points: int = 64,
    scalar_local_steps: int = 3,
    _model: ObservedModel | None = None,
    _coordinate_cache: _RefitCoordinateCache | None = None,
    _work_stats: _ScalarWorkStats | None = None,
    multiplicity_policy: str | None = None,
) -> PartitionRefitResult:
    """Refit cluster centers without changing partition labels."""

    if _coordinate_cache is not None:
        if _work_stats is not None and _work_stats is not _coordinate_cache.work:
            raise ValueError("Refit and cache must share one physical-work sink.")
        _work_stats = _coordinate_cache.work
    tolerance = float(tol)
    epsilon = float(eps)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Partition refit tolerance must be positive and finite.")
    if int(max_iter) < 1:
        raise ValueError("Partition refit interval budget must be positive.")
    mode = str(scalar_mode).strip().lower().replace("-", "_")
    if mode not in {"interval_certified", "grid_local"}:
        raise ValueError("scalar_mode must be interval_certified or grid_local.")
    if int(scalar_grid_points) < 3:
        raise ValueError("scalar_grid_points must be at least three.")
    if int(scalar_local_steps) < 0:
        raise ValueError("scalar_local_steps must be nonnegative.")
    normalized_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if normalized_labels.size != int(data.num_mutations):
        raise ValueError("labels must contain one entry per tumor mutation.")
    normalized_labels = canonical_partition_labels(normalized_labels)
    n_clusters = (
        int(normalized_labels.max()) + 1 if normalized_labels.size else 0
    )
    n_regions = int(data.num_regions)

    policy = multiplicity_policy or ("independent_broad" if _model is None else _model.support_policy)
    model = compile_observed_model(data, eps=epsilon, multiplicity_policy=policy)
    if _model is not None and _model.fingerprint != model.fingerprint:
        raise ValueError("The supplied scalar model does not match the tumor objective.")
    if model.shape != (int(data.num_mutations), n_regions):
        raise ValueError("The supplied scalar model does not match the tumor shape.")
    if model.coupling == "joint":
        from .joint import joint_partition_refit
        return joint_partition_refit(model, normalized_labels, eps=epsilon,
                                     tol=tolerance, max_iter=int(max_iter))
    upper_matrix = model.upper
    observed = model.observed & ((model.alt + model.nonalt) > 0.0)
    centers = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_lower = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_certified = np.ones((n_clusters, n_regions), dtype=bool)
    certificate_methods: set[str] = set()
    certificate_intervals = 0
    total_grid_points = 0
    max_grid_spacing = 0.0
    best_second_loss_gaps: list[float] = []
    total_loss = 0.0
    finite_coordinates = 0
    boundary_count = 0
    active_df = 0
    coordinate_tolerance = tolerance / max(n_clusters * n_regions, 1)
    boundary_tolerance = max(10.0 * tolerance, 1e-8)

    for cluster in range(n_clusters):
        members = np.flatnonzero(normalized_labels == cluster)
        # flatnonzero is sorted, fixing membership identity and summation order.
        member_key = members.astype(np.int64, copy=False).tobytes() if _coordinate_cache is not None else b""
        for region in range(n_regions):
            lower = epsilon
            upper = float(np.min(upper_matrix[members, region]))
            if not np.isfinite(upper) or upper < lower:
                upper = lower
            key = None if _coordinate_cache is None else _RefitCoordinateKey(
                tumor_data_fingerprint(data), model.fingerprint, member_key, region,
                lower, upper, epsilon, mode, coordinate_tolerance, int(max_iter),
                int(scalar_grid_points), int(scalar_local_steps), True,
            )
            coordinate = None if key is None else _coordinate_cache.get(key)
            if coordinate is None:
                coordinate = _fit_coordinate(
                    scalar_problem_from_model(
                        model,
                        members,
                        region,
                        lower=lower,
                        upper=upper,
                        eps=epsilon,
                    ),
                    mode=mode,
                    tolerance=coordinate_tolerance,
                    max_iter=max_iter,
                    grid_points=scalar_grid_points,
                    local_steps=scalar_local_steps,
                    include_breakpoints=True,
                    _work_stats=_work_stats,
                )
                if key is not None:
                    _coordinate_cache.put(key, coordinate)
            centers[cluster, region] = coordinate.beta
            coordinate_lower[cluster, region] = coordinate.global_lower_bound
            coordinate_certified[cluster, region] = coordinate.globally_certified
            certificate_intervals += coordinate.certificate_intervals
            total_grid_points += int(coordinate.grid_points)
            max_grid_spacing = max(max_grid_spacing, float(coordinate.grid_spacing))
            if np.isfinite(float(coordinate.best_second_loss_gap)):
                best_second_loss_gaps.append(float(coordinate.best_second_loss_gap))
            certificate_methods.add(coordinate.certificate_method)
            total_loss += coordinate.loss
            finite_coordinates += int(coordinate.finite_candidate_found)
            if np.any(observed[members, region]):
                at_boundary = bool(
                    coordinate.beta <= lower + boundary_tolerance
                    or coordinate.beta >= upper - boundary_tolerance
                )
                boundary_count += int(at_boundary)
                active_df += int(not at_boundary)

    selected_lower_bound = float(np.sum(coordinate_lower))
    selected_coordinates_certified = bool(np.all(coordinate_certified))
    phi = (
        centers[normalized_labels]
        if normalized_labels.size
        else np.empty((0, n_regions))
    )
    global_gap = (
        max(float(total_loss - selected_lower_bound), 0.0)
        if mode == "interval_certified"
        else float("inf")
    )
    global_certified = bool(
        mode == "interval_certified"
        and selected_coordinates_certified
        and np.isfinite(total_loss)
        and np.isfinite(selected_lower_bound)
        and global_gap <= tolerance
    )
    method_suffix = (
        "_interval_certified"
        if mode == "interval_certified"
        else "_grid_local_approximate"
    )
    return PartitionRefitResult(
        phi=np.clip(phi, epsilon, upper_matrix).astype(np.float64, copy=False),
        cluster_centers=centers,
        loglik=float(-total_loss),
        fit_loss=float(total_loss),
        n_clusters=n_clusters,
        boundary_count=int(boundary_count),
        active_degrees_of_freedom=int(active_df),
        finite_candidate_found=bool(
            finite_coordinates == n_clusters * n_regions and np.isfinite(total_loss)
        ),
        refit_coordinate_count=n_clusters * n_regions,
        refit_finite_coordinate_count=int(finite_coordinates),
        refit_total_grid_points=int(total_grid_points),
        refit_max_grid_spacing=float(max_grid_spacing),
        refit_total_candidate_basins=0,
        refit_total_refined_candidates=(
            int(certificate_intervals)
            if mode == "interval_certified"
            else int(n_clusters * n_regions * int(scalar_local_steps))
        ),
        refit_min_best_second_loss_gap=(
            float(min(best_second_loss_gaps))
            if best_second_loss_gaps
            else float("inf")
        ),
        labels=normalized_labels.copy(),
        loglik_source=(
            "fixed_partition_observed_refit_path"
            + method_suffix
        ),
        global_lower_bound=selected_lower_bound,
        global_optimality_gap=global_gap,
        global_optimum_certified=global_certified,
        global_certificate_method=(
            "+".join(sorted(certificate_methods))
            if certificate_methods
            else "fixed_or_unobserved_coordinates_v1"
        ),
        global_certificate_intervals=int(certificate_intervals),
        refit_mode=mode,
    )
