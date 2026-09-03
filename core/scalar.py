"""Shared scalar likelihood problem and bounded minimizers."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from ..io.data import TumorData
from .objective import (
    ObservedModel,
    compile_observed_model,
    default_phi_initialization,
    observed_box_fingerprint,
)


@dataclass(frozen=True, slots=True)
class ScalarProblem:
    """One cluster-region slice of a canonical observed model."""

    alt: np.ndarray
    nonalt: np.ndarray
    observed: np.ndarray
    first_scale: np.ndarray
    second_scale: np.ndarray
    switch: np.ndarray
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
        path_shape = (alt.size, np.asarray(self.first_scale).shape[-1])
        arrays: dict[str, np.ndarray] = {}
        for name, dtype in (
            ("first_scale", np.float64),
            ("second_scale", np.float64),
            ("switch", np.float64),
            ("log_prior", np.float64),
            ("valid", bool),
        ):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.ndim != 2 or value.shape != path_shape:
                raise ValueError(f"ScalarProblem.{name} must have shape {path_shape}.")
            arrays[name] = value
        lower = float(self.lower)
        upper = float(self.upper)
        eps = float(self.eps)
        if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
            raise ValueError("Require a finite scalar interval with lower <= upper.")
        if not np.isfinite(eps) or not 0.0 < eps < 0.5:
            raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
        for name, value in {
            "alt": alt,
            "nonalt": nonalt,
            "observed": observed,
            **arrays,
        }.items():
            value = np.array(value, copy=True, order="C")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
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
    objective_evaluations: int = 0


@dataclass(frozen=True)
class ScalarGlobalMinimumCertificate:
    argmin: float
    attained_value: float
    global_lower_bound: float
    optimality_gap: float
    globally_certified: bool
    method: str
    intervals_evaluated: int
    argmin_lower: float | None = None
    argmin_upper: float | None = None
    statistically_identified: bool = True
    objective_evaluations: int = 0

    @property
    def representative(self) -> float:
        """Return the finite display representative of the argmin set."""

        return float(self.argmin)


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
        first_scale=model.first_scale[rows, region],
        second_scale=model.second_scale[rows, region],
        switch=model.switch[rows, region],
        log_prior=model.log_prior[rows, region],
        valid=model.valid[rows, region],
        lower=lower,
        upper=upper,
        eps=eps,
    )


def scalar_loss(problem: ScalarProblem, beta: float | np.ndarray) -> float | np.ndarray:
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
        first = problem.first_scale[active, None, :]
        second = problem.second_scale[active, None, :]
        switch = problem.switch[active, None, :]
        mass = first * np.minimum(candidate, switch)
        mass += second * np.maximum(candidate - switch, 0.0)
        probability = np.clip(mass, problem.eps, 1.0 - problem.eps)
        valid = problem.valid[active, None, :]
        joint = (
            problem.alt[active, None, None] * np.log(probability)
            + problem.nonalt[active, None, None] * np.log1p(-probability)
            + problem.log_prior[active, None, :]
        )
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
            segment_slope = np.where(candidate <= switch, first, second)
            slope = np.where(
                (mass > problem.eps) & (mass < 1.0 - problem.eps),
                segment_slope,
                0.0,
            )
            state_score = slope * (
                problem.alt[active, None, None] / probability
                - problem.nonalt[active, None, None] / (1.0 - probability)
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
    rows = (
        np.flatnonzero(problem.observed) if observed_only else range(problem.alt.size)
    )
    for row in rows:
        for path in np.flatnonzero(problem.valid[row]):
            first = float(problem.first_scale[row, path])
            second = float(problem.second_scale[row, path])
            switch = float(problem.switch[row, path])
            if problem.lower < switch < problem.upper:
                points.append(switch)
            for target in (problem.eps, 1.0 - problem.eps):
                if first > 0.0:
                    value = target / first
                    if problem.lower < value <= min(switch, problem.upper):
                        points.append(value)
                if second > 0.0:
                    value = switch + (target - first * switch) / second
                    if max(switch, problem.lower) <= value < problem.upper:
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
        np.clip(np.concatenate((initial, extras)), problem.lower, problem.upper)
    )
    evaluated: dict[float, float] = {}

    def evaluate(candidates: np.ndarray) -> None:
        unique = np.asarray(
            [float(value) for value in candidates if float(value) not in evaluated],
            dtype=np.float64,
        )
        if unique.size:
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
            objective_evaluations=len(evaluated),
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
        objective_evaluations=len(evaluated),
    )


def _active_path_arrays(problem: ScalarProblem) -> tuple[np.ndarray, ...]:
    active = problem.observed
    return (
        problem.alt[active],
        problem.nonalt[active],
        problem.first_scale[active],
        problem.second_scale[active],
        problem.switch[active],
        problem.log_prior[active],
        problem.valid[active],
    )


def _probabilities(
    problem: ScalarProblem,
    beta: float,
    first: np.ndarray,
    second: np.ndarray,
    switch: np.ndarray,
) -> np.ndarray:
    mass = first * np.minimum(float(beta), switch)
    mass += second * np.maximum(float(beta) - switch, 0.0)
    return np.clip(mass, problem.eps, 1.0 - problem.eps)


def _interval_lower_bound(problem: ScalarProblem, left: float, right: float) -> float:
    alt, nonalt, first, second, switch, log_prior, valid = _active_path_arrays(problem)
    probability_left = _probabilities(problem, left, first, second, switch)
    probability_right = _probabilities(problem, right, first, second, switch)
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
        alt[:, None] * np.log(mode) + nonalt[:, None] * np.log1p(-mode) + log_prior,
        -np.inf,
    )
    maximum = np.max(component_upper, axis=1)
    mixture_upper = maximum + np.log(
        np.sum(np.exp(component_upper - maximum[:, None]), axis=1)
    )
    component_bound = float(-np.sum(mixture_upper))

    midpoint = float(left + 0.5 * (right - left))
    half_width = float(0.5 * (right - left))
    probability = _probabilities(problem, midpoint, first, second, switch)
    raw_mass = first * np.minimum(midpoint, switch) + second * np.maximum(
        midpoint - switch, 0.0
    )
    slope = np.where(
        (raw_mass > problem.eps) & (raw_mass < 1.0 - problem.eps),
        np.where(midpoint <= switch, first, second),
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
        * (alt[:, None] / probability_min + nonalt[:, None] / (1.0 - probability_max)),
        0.0,
    )
    curvature_bound = np.where(
        valid,
        slope
        * slope
        * (
            alt[:, None] / np.square(probability_min)
            + nonalt[:, None] / np.square(1.0 - probability_max)
        ),
        0.0,
    )
    hessian_bound = float(
        np.sum(np.max(curvature_bound, axis=1) + np.square(np.max(score_bound, axis=1)))
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
) -> ScalarGlobalMinimumCertificate:
    """Certify the global scalar minimum, or return a valid unresolved bound."""

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Scalar certification tolerance must be positive and finite.")
    if int(max_intervals) < 1:
        raise ValueError("max_intervals must be positive.")
    if not np.any(problem.observed):
        representative = float(
            np.clip(
                0.5 * (problem.lower + problem.upper)
                if hint is None or not np.isfinite(float(hint))
                else float(hint),
                problem.lower,
                problem.upper,
            )
        )
        return ScalarGlobalMinimumCertificate(
            representative,
            0.0,
            0.0,
            0.0,
            True,
            "flat_unobserved_coordinate_v1",
            0,
            argmin_lower=float(problem.lower),
            argmin_upper=float(problem.upper),
            statistically_identified=False,
            objective_evaluations=0,
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
            argmin_lower=float(problem.lower),
            argmin_upper=float(problem.upper),
            statistically_identified=bool(np.any(problem.observed)),
            objective_evaluations=1,
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
    best_beta = float(points[0])
    best_value = float("inf")
    objective_evaluations = 0

    def consider(beta: float) -> None:
        nonlocal best_beta, best_value, objective_evaluations
        objective_evaluations += 1
        value = float(scalar_loss(problem, beta))
        tie = tolerance * 0.25
        if value < best_value - tie or (
            abs(value - best_value) <= tie and beta < best_beta
        ):
            best_beta, best_value = beta, value

    for value in points:
        consider(float(value))
    heap: list[tuple[float, float, float, int]] = []
    intervals = 0
    serial = 0
    for left, right in zip(points[:-1], points[1:]):
        if right > left:
            bound = _interval_lower_bound(problem, float(left), float(right))
            objective_evaluations += 1
            heapq.heappush(heap, (bound, float(left), float(right), serial))
            intervals += 1
            serial += 1
    certified = False
    while heap and intervals < int(max_intervals):
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
            child_bound = _interval_lower_bound(problem, child_left, child_right)
            objective_evaluations += 1
            intervals += 1
            if child_bound <= best_value:
                heapq.heappush(heap, (child_bound, child_left, child_right, serial))
                serial += 1
            if intervals >= int(max_intervals):
                break
    lower_bound = min(float(heap[0][0]), best_value) if heap else best_value
    gap = max(float(best_value - lower_bound), 0.0)
    certified = bool(
        certified
        or (np.isfinite(best_value) and np.isfinite(lower_bound) and gap <= tolerance)
    )
    return ScalarGlobalMinimumCertificate(
        argmin=float(np.clip(best_beta, problem.lower, problem.upper)),
        attained_value=best_value,
        global_lower_bound=lower_bound,
        optimality_gap=gap,
        globally_certified=certified,
        method="interval_binomial_mixture_bound_v1",
        intervals_evaluated=intervals,
        argmin_lower=float(np.clip(best_beta, problem.lower, problem.upper)),
        argmin_upper=float(np.clip(best_beta, problem.lower, problem.upper)),
        statistically_identified=True,
        objective_evaluations=int(objective_evaluations),
    )


@dataclass(frozen=True)
class PartitionFit:
    """Observed-likelihood refit of one immutable partition."""

    labels: np.ndarray
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    finite_candidate_found: bool
    observed_model_hash: str
    observed_likelihood_hash: str
    reporting_model_hash: str
    observed_box_hash: str
    likelihood_eps_hex: str
    global_optimum_certified: bool = False
    partition_signature: str = ""
    refit_numerically_resolved: bool = False
    fit_loss: float = float("nan")
    n_clusters: int = 0
    boundary_count: int = 0
    active_degrees_of_freedom: int = 0
    refit_coordinate_count: int = 0
    refit_finite_coordinate_count: int = 0
    refit_total_grid_points: int = 0
    refit_max_grid_spacing: float = 0.0
    refit_total_candidate_basins: int = 0
    refit_total_refined_candidates: int = 0
    refit_min_best_second_loss_gap: float = float("inf")
    loglik_source: str = "partition_constrained_observed_mle"
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0
    refit_mode: str = "interval_certified"
    coordinate_argmin_lower: np.ndarray | None = None
    coordinate_argmin_upper: np.ndarray | None = None
    coordinate_statistically_identified: np.ndarray | None = None
    refit_objective_evaluations: int = 0

    def __post_init__(self) -> None:
        for name in (
            "observed_model_hash",
            "observed_likelihood_hash",
            "reporting_model_hash",
            "observed_box_hash",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"PartitionFit.{name} must be a SHA-256 digest.")
        eps_token = str(self.likelihood_eps_hex)
        try:
            epsilon = float.fromhex(eps_token)
        except ValueError as exc:
            raise ValueError(
                "PartitionFit.likelihood_eps_hex must be a hexadecimal float."
            ) from exc
        if (
            not np.isfinite(epsilon)
            or not 0.0 < epsilon < 0.5
            or epsilon.hex() != eps_token
        ):
            raise ValueError(
                "PartitionFit.likelihood_eps_hex must canonically identify a valid eps."
            )
        if (
            int(self.refit_coordinate_count) < 0
            or int(self.refit_objective_evaluations) < 0
        ):
            raise ValueError("Partition-refit work counters must be nonnegative.")
        if self.global_optimum_certified and (
            not np.isfinite(float(self.global_lower_bound))
            or not np.isfinite(float(self.global_optimality_gap))
            or float(self.global_optimality_gap) < 0.0
            or str(self.global_certificate_method) == "none"
        ):
            raise ValueError("A global refit claim requires a finite certificate.")
        for name, dtype in (
            ("labels", np.int64),
            ("phi", np.float64),
            ("cluster_centers", np.float64),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        centers = np.asarray(self.cluster_centers)
        lower = (
            centers
            if self.coordinate_argmin_lower is None
            else self.coordinate_argmin_lower
        )
        upper = (
            centers
            if self.coordinate_argmin_upper is None
            else self.coordinate_argmin_upper
        )
        identified = (
            np.ones(centers.shape, dtype=bool)
            if self.coordinate_statistically_identified is None
            else self.coordinate_statistically_identified
        )
        for name, values, dtype in (
            ("coordinate_argmin_lower", lower, np.float64),
            ("coordinate_argmin_upper", upper, np.float64),
            ("coordinate_statistically_identified", identified, bool),
        ):
            value = np.array(values, dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def validate_observed_model(self, model: ObservedModel, *, eps: float) -> None:
        """Fail closed unless this refit belongs to ``model`` and ``eps``."""

        expected = {
            "observed_model_hash": str(model.fingerprint),
            "observed_likelihood_hash": str(model.likelihood_fingerprint),
            "reporting_model_hash": str(model.reporting_fingerprint),
            "observed_box_hash": observed_box_fingerprint(model),
            "likelihood_eps_hex": float(eps).hex(),
        }
        for name, value in expected.items():
            if str(getattr(self, name)) != value:
                raise ValueError(
                    f"PartitionFit {name} does not match the observed model."
                )


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
    argmin_lower: float
    argmin_upper: float
    statistically_identified: bool
    grid_points: int = 0
    grid_spacing: float = 0.0
    best_second_loss_gap: float = float("inf")
    objective_evaluations: int = 0


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
    hint: float | None = None,
) -> _RefitCoordinateResult:
    if not np.any(problem.observed):
        representative = float(
            np.clip(
                0.5 * (problem.lower + problem.upper)
                if hint is None or not np.isfinite(float(hint))
                else float(hint),
                problem.lower,
                problem.upper,
            )
        )
        return _RefitCoordinateResult(
            beta=representative,
            loss=0.0,
            global_lower_bound=0.0,
            optimality_gap=0.0,
            finite_candidate_found=True,
            globally_certified=True,
            certificate_method="flat_unobserved_coordinate_v1",
            certificate_intervals=0,
            argmin_lower=float(problem.lower),
            argmin_upper=float(problem.upper),
            statistically_identified=False,
            objective_evaluations=0,
        )
    if mode == "interval_certified":
        result = certify_scalar_minimum(
            problem,
            tolerance=tolerance,
            max_intervals=max(int(max_iter) * 256, 4096),
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
            argmin_lower=float(result.argmin_lower),
            argmin_upper=float(result.argmin_upper),
            statistically_identified=bool(result.statistically_identified),
            objective_evaluations=int(result.objective_evaluations),
        )
    result = approximate_scalar_minimum(
        problem,
        grid_points=grid_points,
        local_steps=local_steps,
        include_breakpoints=include_breakpoints,
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
        argmin_lower=float(result.argmin),
        argmin_upper=float(result.argmin),
        statistically_identified=True,
        grid_points=int(result.grid_points_evaluated),
        grid_spacing=float(result.final_grid_spacing),
        best_second_loss_gap=float(result.best_second_loss_gap),
        objective_evaluations=int(result.objective_evaluations),
    )


def partition_constrained_observed_refit(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    scalar_mode: str = "interval_certified",
    scalar_grid_points: int = 64,
    scalar_local_steps: int = 3,
    _model: ObservedModel | None = None,
) -> PartitionFit:
    """Refit cluster centers without changing partition labels."""

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
    n_clusters = int(normalized_labels.max()) + 1 if normalized_labels.size else 0
    n_regions = int(data.num_regions)

    model = (
        compile_observed_model(data, major_prior=major_prior, eps=epsilon)
        if _model is None
        else _model
    )
    phi_initialization = default_phi_initialization(model, eps=epsilon)
    if model.shape != (int(data.num_mutations), n_regions):
        raise ValueError("The supplied scalar model does not match the tumor shape.")
    if not np.all(np.asarray(model.lower, dtype=np.float64) == epsilon):
        raise ValueError("The supplied scalar model does not match the refit eps box.")
    upper_matrix = model.upper
    observed = model.observed & ((model.alt + model.nonalt) > 0.0)
    centers = np.zeros((n_clusters, n_regions), dtype=np.float64)
    argmin_lower = np.zeros((n_clusters, n_regions), dtype=np.float64)
    argmin_upper = np.zeros((n_clusters, n_regions), dtype=np.float64)
    statistically_identified = np.zeros((n_clusters, n_regions), dtype=bool)
    coordinate_lower = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_certified = np.ones((n_clusters, n_regions), dtype=bool)
    certificate_methods: set[str] = set()
    certificate_intervals = 0
    total_grid_points = 0
    total_objective_evaluations = 0
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
        for region in range(n_regions):
            lower = epsilon
            upper = float(np.min(upper_matrix[members, region]))
            if not np.isfinite(upper) or upper < lower:
                upper = lower
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
                include_breakpoints=model.has_internal_switches,
                hint=float(np.mean(phi_initialization[members, region])),
            )
            centers[cluster, region] = coordinate.beta
            argmin_lower[cluster, region] = coordinate.argmin_lower
            argmin_upper[cluster, region] = coordinate.argmin_upper
            statistically_identified[cluster, region] = (
                coordinate.statistically_identified
            )
            coordinate_lower[cluster, region] = coordinate.global_lower_bound
            coordinate_certified[cluster, region] = coordinate.globally_certified
            certificate_intervals += coordinate.certificate_intervals
            total_grid_points += int(coordinate.grid_points)
            total_objective_evaluations += int(coordinate.objective_evaluations)
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
    path_suffix = "_path" if model.requires_generic_path_solver else ""
    return PartitionFit(
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
        observed_model_hash=str(model.fingerprint),
        observed_likelihood_hash=str(model.likelihood_fingerprint),
        reporting_model_hash=str(model.reporting_fingerprint),
        observed_box_hash=observed_box_fingerprint(model),
        likelihood_eps_hex=epsilon.hex(),
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
            float(min(best_second_loss_gaps)) if best_second_loss_gaps else float("inf")
        ),
        labels=normalized_labels.copy(),
        loglik_source=("fixed_partition_observed_refit" + path_suffix + method_suffix),
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
        coordinate_argmin_lower=argmin_lower,
        coordinate_argmin_upper=argmin_upper,
        coordinate_statistically_identified=statistically_identified,
        refit_objective_evaluations=int(total_objective_evaluations),
    )
