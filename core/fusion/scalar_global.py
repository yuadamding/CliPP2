"""Certified global minimization for one-dimensional observed likelihoods.

The raw scalar pilot and every fixed-partition center reduce to the same
problem: minimize a sum of binomial mixture losses over a compact CCF
interval.  This module uses deterministic interval branch-and-bound.  On each
interval it upper-bounds every component log likelihood at its binomial mode;
``logsumexp`` of those component bounds is an upper bound on the mixture log
likelihood and therefore gives a rigorous lower bound on its negative log
likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from ...io.data import TumorData


@dataclass(frozen=True)
class ScalarGlobalMinimumCertificate:
    argmin: float
    attained_value: float
    global_lower_bound: float
    optimality_gap: float
    globally_certified: bool
    method: str
    intervals_evaluated: int


@dataclass(frozen=True)
class _ScalarMixtureProblem:
    alt: np.ndarray
    total: np.ndarray
    first_scale: np.ndarray
    second_scale: np.ndarray
    switch_fraction: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray

    def probabilities(self, beta: float, eps: float) -> np.ndarray:
        value = float(beta)
        mass = self.first_scale * np.minimum(value, self.switch_fraction)
        mass += self.second_scale * np.maximum(
            value - self.switch_fraction, 0.0
        )
        return np.clip(mass, float(eps), 1.0 - float(eps))

    def loss(self, beta: float, eps: float) -> float:
        probability = self.probabilities(beta, eps)
        nonalt = self.total - self.alt
        joint = np.where(
            self.valid,
            self.alt[:, None] * np.log(probability)
            + nonalt[:, None] * np.log1p(-probability)
            + self.log_prior,
            -np.inf,
        )
        maximum = np.max(joint, axis=1)
        value = maximum + np.log(
            np.sum(np.exp(joint - maximum[:, None]), axis=1)
        )
        return float(-np.sum(value, dtype=np.float64))

    def interval_lower_bound(self, left: float, right: float, eps: float) -> float:
        probability_left = self.probabilities(left, eps)
        probability_right = self.probabilities(right, eps)
        probability_min = np.minimum(probability_left, probability_right)
        probability_max = np.maximum(probability_left, probability_right)
        with np.errstate(divide="ignore", invalid="ignore"):
            empirical = np.divide(
                self.alt,
                self.total,
                out=np.full_like(self.alt, 0.5, dtype=np.float64),
                where=self.total > 0.0,
            )
        component_mode = np.clip(
            empirical[:, None], probability_min, probability_max
        )
        nonalt = self.total - self.alt
        component_upper = np.where(
            self.valid,
            self.alt[:, None] * np.log(component_mode)
            + nonalt[:, None] * np.log1p(-component_mode)
            + self.log_prior,
            -np.inf,
        )
        maximum = np.max(component_upper, axis=1)
        mixture_upper = maximum + np.log(
            np.sum(np.exp(component_upper - maximum[:, None]), axis=1)
        )
        component_bound = float(-np.sum(mixture_upper))

        midpoint = float(left + 0.5 * (right - left))
        half_width = float(0.5 * (right - left))
        probability = self.probabilities(midpoint, eps)
        raw_mass = self.first_scale * np.minimum(
            midpoint, self.switch_fraction
        ) + self.second_scale * np.maximum(
            midpoint - self.switch_fraction, 0.0
        )
        slope = np.where(
            (raw_mass > float(eps)) & (raw_mass < 1.0 - float(eps)),
            np.where(
                midpoint <= self.switch_fraction,
                self.first_scale,
                self.second_scale,
            ),
            0.0,
        )
        nonalt = self.total - self.alt
        joint = np.where(
            self.valid,
            self.alt[:, None] * np.log(probability)
            + nonalt[:, None] * np.log1p(-probability)
            + self.log_prior,
            -np.inf,
        )
        maximum = np.max(joint, axis=1)
        weights = np.where(
            self.valid, np.exp(joint - maximum[:, None]), 0.0
        )
        weights /= np.sum(weights, axis=1, keepdims=True)
        state_score = slope * (
            self.alt[:, None] / probability
            - nonalt[:, None] / (1.0 - probability)
        )
        loss_gradient = -float(np.sum(weights * state_score))
        score_abs_bound = slope * (
            self.alt[:, None] / probability_min
            + nonalt[:, None] / (1.0 - probability_max)
        )
        component_hessian_bound = slope * slope * (
            self.alt[:, None] / (probability_min * probability_min)
            + nonalt[:, None]
            / ((1.0 - probability_max) * (1.0 - probability_max))
        )
        score_abs_bound = np.where(self.valid, score_abs_bound, 0.0)
        component_hessian_bound = np.where(
            self.valid, component_hessian_bound, 0.0
        )
        hessian_bound = float(
            np.sum(
                np.max(component_hessian_bound, axis=1)
                + np.max(score_abs_bound, axis=1) ** 2
            )
        )
        taylor_bound = (
            self.loss(midpoint, eps)
            - abs(loss_gradient) * half_width
            - 0.5 * hessian_bound * half_width * half_width
        )
        # Both constructions are lower bounds; their maximum is also a lower
        # bound. Round outward after combining them.
        return float(
            np.nextafter(max(component_bound, taylor_bound), -np.inf)
        )

    def breakpoints(self, lower: float, upper: float, eps: float) -> np.ndarray:
        points: list[float] = [float(lower), float(upper)]
        for row in range(self.valid.shape[0]):
            for path in np.flatnonzero(self.valid[row]):
                first = float(self.first_scale[row, path])
                second = float(self.second_scale[row, path])
                switch = float(self.switch_fraction[row, path])
                if lower < switch < upper:
                    points.append(switch)
                for target in (float(eps), 1.0 - float(eps)):
                    if first > 0.0:
                        value = target / first
                        if lower < value <= min(switch, upper):
                            points.append(value)
                    if second > 0.0:
                        value = switch + (target - first * switch) / second
                        if max(switch, lower) <= value < upper:
                            points.append(value)
        return np.unique(np.clip(np.asarray(points, dtype=np.float64), lower, upper))


def _problem_from_tumor(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    major_prior: float,
) -> _ScalarMixtureProblem:
    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    region = int(region_index)
    alt = np.asarray(data.alt_counts, dtype=np.float64)[rows, region]
    total = np.asarray(data.total_counts, dtype=np.float64)[rows, region]
    path = getattr(data, "path_likelihood", None)
    if path is not None:
        scale = np.asarray(data.scaling, dtype=np.float64)[rows, region, None]
        return _ScalarMixtureProblem(
            alt=alt,
            total=total,
            first_scale=scale * path.first_copy[rows, region, :],
            second_scale=scale * path.second_copy[rows, region, :],
            switch_fraction=path.switch_fraction[rows, region, :],
            log_prior=path.log_prior[rows, region, :],
            valid=path.valid[rows, region, :],
        )

    prior = float(major_prior)
    if not np.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("major_prior must lie strictly in (0, 1).")
    scale = np.asarray(data.scaling, dtype=np.float64)[rows, region]
    minor = scale * np.asarray(data.minor_cn, dtype=np.float64)[rows, region]
    major = scale * np.asarray(data.major_cn, dtype=np.float64)[rows, region]
    fixed = scale * np.asarray(data.fixed_multiplicity, dtype=np.float64)[rows, region]
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)[rows, region]
    first = np.column_stack((np.where(ambiguous, minor, fixed), major))
    valid = np.column_stack((np.ones(rows.size, dtype=bool), ambiguous))
    log_prior = np.column_stack(
        (
            np.where(ambiguous, np.log1p(-prior), 0.0),
            np.full(rows.size, np.log(prior), dtype=np.float64),
        )
    )
    log_prior = np.where(valid, log_prior, -np.inf)
    return _ScalarMixtureProblem(
        alt=alt,
        total=total,
        first_scale=first,
        second_scale=first.copy(),
        switch_fraction=np.zeros_like(first),
        log_prior=log_prior,
        valid=valid,
    )


def _observed_rows(
    data: TumorData, mutation_indices: np.ndarray, region_index: int
) -> np.ndarray:
    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    observed = getattr(data, "count_observed", None)
    if observed is not None:
        rows = rows[np.asarray(observed, dtype=bool)[rows, int(region_index)]]
    return rows[
        np.asarray(data.total_counts, dtype=np.float64)[rows, int(region_index)] > 0.0
    ]


def evaluate_tumor_scalar_loss(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    beta: float,
    *,
    major_prior: float,
    eps: float,
) -> float:
    rows = _observed_rows(data, mutation_indices, region_index)
    if rows.size == 0:
        return 0.0
    return _problem_from_tumor(
        data, rows, region_index, major_prior=major_prior
    ).loss(float(beta), float(eps))


def certify_tumor_scalar_minimum(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tolerance: float,
    max_intervals: int,
    hint: float | None = None,
) -> ScalarGlobalMinimumCertificate:
    """Certify the global scalar optimum, or return a valid unresolved bound."""

    lower = float(lower)
    upper = float(upper)
    tolerance = float(tolerance)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("Require a finite scalar interval with lower <= upper.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Scalar certification tolerance must be positive and finite.")
    if int(max_intervals) < 1:
        raise ValueError("max_intervals must be positive.")
    rows = _observed_rows(data, mutation_indices, region_index)
    if rows.size == 0:
        beta = float(0.5 * (lower + upper))
        return ScalarGlobalMinimumCertificate(
            argmin=beta,
            attained_value=0.0,
            global_lower_bound=0.0,
            optimality_gap=0.0,
            globally_certified=True,
            method="interval_binomial_mixture_bound_v1",
            intervals_evaluated=0,
        )
    problem = _problem_from_tumor(
        data, rows, region_index, major_prior=major_prior
    )
    if upper <= lower:
        loss = problem.loss(lower, eps)
        return ScalarGlobalMinimumCertificate(
            argmin=lower,
            attained_value=loss,
            global_lower_bound=loss,
            optimality_gap=0.0,
            globally_certified=bool(np.isfinite(loss)),
            method="fixed_scalar_coordinate_v1",
            intervals_evaluated=1,
        )

    initial = np.concatenate(
        (
            np.linspace(lower, upper, num=17, dtype=np.float64),
            problem.breakpoints(lower, upper, eps),
        )
    )
    if hint is not None and np.isfinite(float(hint)):
        initial = np.append(initial, np.clip(float(hint), lower, upper))
    points = np.unique(np.clip(initial, lower, upper))

    best_beta = float(points[0])
    best_value = float("inf")

    def consider(beta: float) -> None:
        nonlocal best_beta, best_value
        value = problem.loss(beta, eps)
        tie = tolerance * 0.25
        if value < best_value - tie or (
            abs(value - best_value) <= tie and float(beta) < best_beta
        ):
            best_beta = float(beta)
            best_value = float(value)

    for value in points:
        consider(float(value))

    heap: list[tuple[float, float, float, int]] = []
    serial = 0
    intervals_evaluated = 0
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        bound = problem.interval_lower_bound(float(left), float(right), eps)
        intervals_evaluated += 1
        heapq.heappush(heap, (bound, float(left), float(right), serial))
        serial += 1

    globally_certified = False
    global_lower = float(heap[0][0]) if heap else best_value
    while heap and intervals_evaluated < int(max_intervals):
        global_lower = min(float(heap[0][0]), best_value)
        gap = max(best_value - global_lower, 0.0)
        if np.isfinite(best_value) and gap <= tolerance:
            globally_certified = True
            break
        bound, left, right, _ = heapq.heappop(heap)
        if bound > best_value:
            continue
        midpoint = float(left + 0.5 * (right - left))
        if not left < midpoint < right:
            continue
        consider(midpoint)
        for child_left, child_right in ((left, midpoint), (midpoint, right)):
            child_bound = problem.interval_lower_bound(child_left, child_right, eps)
            intervals_evaluated += 1
            if child_bound <= best_value:
                heapq.heappush(
                    heap,
                    (child_bound, child_left, child_right, serial),
                )
                serial += 1
            if intervals_evaluated >= int(max_intervals):
                break

    global_lower = min(float(heap[0][0]), best_value) if heap else best_value
    gap = max(float(best_value - global_lower), 0.0)
    globally_certified = bool(
        globally_certified
        or (np.isfinite(best_value) and np.isfinite(global_lower) and gap <= tolerance)
    )
    return ScalarGlobalMinimumCertificate(
        argmin=float(np.clip(best_beta, lower, upper)),
        attained_value=float(best_value),
        global_lower_bound=float(global_lower),
        optimality_gap=float(gap),
        globally_certified=globally_certified,
        method="interval_binomial_mixture_bound_v1",
        intervals_evaluated=int(intervals_evaluated),
    )


__all__ = [
    "ScalarGlobalMinimumCertificate",
    "certify_tumor_scalar_minimum",
    "evaluate_tumor_scalar_loss",
]
