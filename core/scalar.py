"""Shared scalar likelihood problem and bounded minimizers."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .objective import ObservedModel


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


@dataclass(frozen=True)
class ScalarGlobalMinimumCertificate:
    argmin: float
    attained_value: float
    global_lower_bound: float
    optimality_gap: float
    globally_certified: bool
    method: str
    intervals_evaluated: int


def scalar_problem_from_model(
    model: ObservedModel,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    eps: float,
) -> ScalarProblem:
    """Select one shared-center coordinate without recompiling its likelihood."""

    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    region = int(region_index)
    alt = model.alt[rows, region]
    nonalt = model.nonalt[rows, region]
    return ScalarProblem(
        alt=alt,
        nonalt=nonalt,
        observed=model.observed[rows, region] & ((alt + nonalt) > 0.0),
        first_scale=model.first_scale[rows, region],
        second_scale=model.second_scale[rows, region],
        switch=model.switch[rows, region],
        log_prior=model.log_prior[rows, region],
        valid=model.valid[rows, region],
        lower=lower,
        upper=upper,
        eps=eps,
    )


def scalar_loss(
    problem: ScalarProblem, beta: float | np.ndarray
) -> float | np.ndarray:
    """Evaluate the canonical observed loss at one or many shared CCFs."""

    values = np.asarray(beta, dtype=np.float64)
    scalar = values.ndim == 0
    flat = values.reshape(-1)
    active = problem.observed
    if not np.any(active):
        loss = np.zeros(flat.size, dtype=np.float64)
    else:
        candidate = flat[None, :, None]
        first = problem.first_scale[active, None, :]
        second = problem.second_scale[active, None, :]
        switch = problem.switch[active, None, :]
        mass = first * np.minimum(candidate, switch)
        mass += second * np.maximum(candidate - switch, 0.0)
        probability = np.clip(mass, problem.eps, 1.0 - problem.eps)
        joint = (
            problem.alt[active, None, None] * np.log(probability)
            + problem.nonalt[active, None, None] * np.log1p(-probability)
            + problem.log_prior[active, None, :]
        )
        joint = np.where(problem.valid[active, None, :], joint, -np.inf)
        loss = -np.sum(
            np.logaddexp.reduce(joint, axis=-1), axis=0, dtype=np.float64
        )
    return float(loss[0]) if scalar else loss.reshape(values.shape)


def scalar_breakpoints(
    problem: ScalarProblem, *, observed_only: bool = True
) -> np.ndarray:
    """Return switches and clipping knots in the problem's bounded interval."""

    points = [problem.lower, problem.upper]
    rows = np.flatnonzero(problem.observed) if observed_only else range(problem.alt.size)
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
    alt, nonalt, first, second, switch, log_prior, valid = _active_path_arrays(
        problem
    )
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
    heap: list[tuple[float, float, float, int]] = []
    intervals = 0
    serial = 0
    for left, right in zip(points[:-1], points[1:]):
        if right > left:
            bound = _interval_lower_bound(problem, float(left), float(right))
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
            intervals += 1
            if child_bound <= best_value:
                heapq.heappush(
                    heap, (child_bound, child_left, child_right, serial)
                )
                serial += 1
            if intervals >= int(max_intervals):
                break
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


__all__ = [
    "ApproximateScalarMinimum",
    "ScalarGlobalMinimumCertificate",
    "ScalarProblem",
    "approximate_scalar_minimum",
    "certify_scalar_minimum",
    "scalar_breakpoints",
    "scalar_loss",
    "scalar_problem_from_model",
]
