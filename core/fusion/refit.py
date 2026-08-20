"""Immutable-partition observed-likelihood refitting under explicit profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData
from ..objective import compile_observed_model
from ..scalar import (
    ScalarProblem,
    approximate_scalar_minimum,
    certify_scalar_minimum,
    scalar_problem_from_model,
)


@dataclass(frozen=True)
class PartitionRefitResult:
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


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    remapped = np.empty_like(labels)
    mapping: dict[int, int] = {}
    for index, value in enumerate(labels):
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
) -> _RefitCoordinateResult:
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
        grid_points=int(result.grid_points_evaluated),
        grid_spacing=float(result.final_grid_spacing),
        best_second_loss_gap=float(result.best_second_loss_gap),
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
) -> PartitionRefitResult:
    """Refit cluster centers without changing partition labels."""

    tol = float(tol)
    eps = float(eps)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition refit tolerance must be positive and finite.")
    if int(max_iter) < 1:
        raise ValueError("Partition refit interval budget must be positive.")
    normalized_scalar_mode = str(scalar_mode).strip().lower().replace("-", "_")
    if normalized_scalar_mode not in {"interval_certified", "grid_local"}:
        raise ValueError("scalar_mode must be interval_certified or grid_local.")
    if int(scalar_grid_points) < 3:
        raise ValueError("scalar_grid_points must be at least three.")
    if int(scalar_local_steps) < 0:
        raise ValueError("scalar_local_steps must be nonnegative.")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != int(data.num_mutations):
        raise ValueError("labels must contain one entry per tumor mutation.")
    labels = _canonical_labels(labels)
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_regions = int(data.num_regions)

    model = compile_observed_model(data, major_prior=major_prior, eps=eps)
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
    coordinate_tolerance = tol / max(n_clusters * n_regions, 1)
    boundary_tolerance = max(10.0 * tol, 1e-8)

    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        for region in range(n_regions):
            lower = eps
            upper = float(np.min(upper_matrix[members, region]))
            if not np.isfinite(upper) or upper < lower:
                upper = lower
            problem = scalar_problem_from_model(
                model,
                members,
                region,
                lower=lower,
                upper=upper,
                eps=eps,
            )
            coordinate = _fit_coordinate(
                problem,
                mode=normalized_scalar_mode,
                tolerance=coordinate_tolerance,
                max_iter=max_iter,
                grid_points=scalar_grid_points,
                local_steps=scalar_local_steps,
                include_breakpoints=data.path_likelihood is not None,
            )
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

    phi = centers[labels] if labels.size else np.empty((0, n_regions))
    global_gap = (
        max(float(total_loss - selected_lower_bound), 0.0)
        if normalized_scalar_mode == "interval_certified"
        else float("inf")
    )
    global_certified = bool(
        normalized_scalar_mode == "interval_certified"
        and selected_coordinates_certified
        and np.isfinite(total_loss)
        and np.isfinite(selected_lower_bound)
        and global_gap <= tol
    )
    method_suffix = (
        "_interval_certified"
        if normalized_scalar_mode == "interval_certified"
        else "_grid_local_approximate"
    )
    path_suffix = "_path" if data.path_likelihood is not None else ""
    loglik_source = "fixed_partition_observed_refit" + path_suffix + method_suffix
    return PartitionRefitResult(
        phi=np.clip(phi, eps, upper_matrix).astype(np.float64, copy=False),
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
            if normalized_scalar_mode == "interval_certified"
            else int(n_clusters * n_regions * int(scalar_local_steps))
        ),
        refit_min_best_second_loss_gap=(
            float(min(best_second_loss_gaps)) if best_second_loss_gaps else float("inf")
        ),
        labels=labels.copy(),
        loglik_source=loglik_source,
        global_lower_bound=selected_lower_bound,
        global_optimality_gap=global_gap,
        global_optimum_certified=global_certified,
        global_certificate_method=(
            "+".join(sorted(certificate_methods))
            if certificate_methods
            else "fixed_or_unobserved_coordinates_v1"
        ),
        global_certificate_intervals=int(certificate_intervals),
        refit_mode=normalized_scalar_mode,
    )


__all__ = ["PartitionRefitResult", "partition_constrained_observed_refit"]
