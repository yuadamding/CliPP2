"""Immutable-partition observed-likelihood refitting with global certificates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData
from .scalar_global import (
    certify_tumor_scalar_minimum,
    evaluate_tumor_scalar_loss,
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
    anchor_mode: str
    anchor_deviance_increase: float
    second_best_anchor_deviance_increase: float
    loglik_source: str = "partition_constrained_observed_mle"
    clonal_cluster: int | None = None
    fixed_anchor_target: np.ndarray | None = None
    fixed_anchor_block_signature: str = "none"
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_optimum_certified: bool = False
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0


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


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    remapped = np.empty_like(labels)
    mapping: dict[int, int] = {}
    for index, value in enumerate(labels):
        remapped[index] = mapping.setdefault(int(value), len(mapping))
    return remapped


def _certified_refit_cluster_region(
    data: TumorData,
    *,
    mutation_indices: np.ndarray,
    region_index: int,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> _RefitCoordinateResult:
    certificate = certify_tumor_scalar_minimum(
        data,
        np.asarray(mutation_indices, dtype=np.int64),
        int(region_index),
        lower=float(lower),
        upper=float(upper),
        major_prior=float(major_prior),
        eps=float(eps),
        tolerance=float(tol),
        max_intervals=max(int(max_iter) * 256, 4096),
    )
    return _RefitCoordinateResult(
        beta=float(certificate.argmin),
        loss=float(certificate.attained_value),
        global_lower_bound=float(certificate.global_lower_bound),
        optimality_gap=float(certificate.optimality_gap),
        finite_candidate_found=bool(np.isfinite(certificate.attained_value)),
        globally_certified=bool(certificate.globally_certified),
        certificate_method=str(certificate.method),
        certificate_intervals=int(certificate.intervals_evaluated),
    )


def _observed_positive_depth_mask(data: TumorData) -> np.ndarray:
    observed = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    if data.count_observed is not None:
        observed &= np.asarray(data.count_observed, dtype=bool)
    return observed


def partition_constrained_observed_refit(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    anchor_mode: str = "clonal_required",
    anchor_cluster: int | None = None,
    fixed_anchor_target: np.ndarray | None = None,
    fixed_anchor_block_signature: str = "none",
    anchor_feasibility_tol: float = 1e-8,
    grid_refinement_factor: int = 1,
) -> PartitionRefitResult:
    """Globally refit cluster centers without changing partition labels.

    ``grid_refinement_factor`` is retained as a no-op compatibility argument;
    the production implementation no longer uses a sampling grid.
    """

    tol = float(tol)
    eps = float(eps)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition refit tolerance must be positive and finite.")
    if int(max_iter) < 1:
        raise ValueError("Partition refit interval budget must be positive.")
    if int(grid_refinement_factor) < 1:
        raise ValueError("grid_refinement_factor must be a positive integer.")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != int(data.num_mutations):
        raise ValueError("labels must contain one entry per tumor mutation.")
    labels = _canonical_labels(labels)
    normalized_anchor_mode = str(anchor_mode).strip().lower()
    if normalized_anchor_mode not in {"none", "clonal_required"}:
        raise ValueError("anchor_mode must be none or clonal_required.")
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_regions = int(data.num_regions)
    if anchor_cluster is not None:
        anchor_cluster = int(anchor_cluster)
        if normalized_anchor_mode != "clonal_required":
            raise ValueError("anchor_cluster requires clonal_required mode.")
        if not 0 <= anchor_cluster < n_clusters:
            raise ValueError("anchor_cluster must identify an occupied block.")
    if fixed_anchor_target is not None:
        fixed_anchor_target = np.asarray(
            fixed_anchor_target, dtype=np.float64
        ).reshape(-1)
        if anchor_cluster is None:
            raise ValueError("fixed_anchor_target requires anchor_cluster.")
        if fixed_anchor_target.shape != (n_regions,) or not np.all(
            np.isfinite(fixed_anchor_target)
        ):
            raise ValueError("fixed_anchor_target must be finite and region-aligned.")
    fixed_anchor_block_signature = str(fixed_anchor_block_signature)
    if anchor_cluster is not None and fixed_anchor_block_signature == "none":
        raise ValueError("An explicit raw clonal cluster requires its signature.")
    anchor_feasibility_tol = float(anchor_feasibility_tol)
    if not np.isfinite(anchor_feasibility_tol) or anchor_feasibility_tol < 0.0:
        raise ValueError("anchor_feasibility_tol must be nonnegative and finite.")

    upper_matrix = np.asarray(data.phi_upper, dtype=np.float64)
    observed = _observed_positive_depth_mask(data)
    centers = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_loss = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_lower = np.zeros((n_clusters, n_regions), dtype=np.float64)
    coordinate_certified = np.ones((n_clusters, n_regions), dtype=bool)
    coordinate_boundary = np.zeros((n_clusters, n_regions), dtype=bool)
    coordinate_active = np.zeros((n_clusters, n_regions), dtype=bool)
    certificate_methods: set[str] = set()
    certificate_intervals = 0
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
            coordinate = _certified_refit_cluster_region(
                data,
                mutation_indices=members,
                region_index=region,
                lower=lower,
                upper=upper,
                major_prior=float(major_prior),
                eps=eps,
                tol=coordinate_tolerance,
                max_iter=int(max_iter),
            )
            centers[cluster, region] = coordinate.beta
            coordinate_loss[cluster, region] = coordinate.loss
            coordinate_lower[cluster, region] = coordinate.global_lower_bound
            coordinate_certified[cluster, region] = coordinate.globally_certified
            certificate_intervals += coordinate.certificate_intervals
            certificate_methods.add(coordinate.certificate_method)
            total_loss += coordinate.loss
            finite_coordinates += int(coordinate.finite_candidate_found)
            if np.any(observed[members, region]):
                at_boundary = bool(
                    coordinate.beta <= lower + boundary_tolerance
                    or coordinate.beta >= upper - boundary_tolerance
                )
                coordinate_boundary[cluster, region] = at_boundary
                coordinate_active[cluster, region] = not at_boundary
                boundary_count += int(at_boundary)
                active_df += int(not at_boundary)

    clonal_cluster: int | None = None
    anchor_deviance_increase = 0.0
    second_best_anchor_deviance_increase = float("inf")
    selected_lower_bound = float(np.sum(coordinate_lower))
    selected_coordinates_certified = bool(np.all(coordinate_certified))
    if n_clusters and normalized_anchor_mode == "clonal_required":
        occupied = list(range(n_clusters))
        anchored_loss = np.zeros((n_clusters, n_regions), dtype=np.float64)
        anchored_center = np.zeros((n_clusters, n_regions), dtype=np.float64)
        anchor_increase = np.full(n_clusters, np.inf, dtype=np.float64)
        anchor_lower = np.full(n_clusters, np.inf, dtype=np.float64)
        for candidate in occupied:
            members = np.flatnonzero(labels == candidate)
            for region in range(n_regions):
                upper = float(np.min(upper_matrix[members, region]))
                if not np.isfinite(upper) or upper < eps:
                    upper = eps
                pinned = float(
                    fixed_anchor_target[region]
                    if fixed_anchor_target is not None
                    and candidate == anchor_cluster
                    else min(1.0, upper)
                )
                if pinned > upper + anchor_feasibility_tol:
                    raise ValueError("Raw anchor block cannot preserve CCF one.")
                anchored_center[candidate, region] = pinned
                anchored_loss[candidate, region] = evaluate_tumor_scalar_loss(
                    data,
                    members,
                    region,
                    pinned,
                    major_prior=float(major_prior),
                    eps=eps,
                )
            anchor_increase[candidate] = float(
                np.sum(anchored_loss[candidate] - coordinate_loss[candidate])
            )
            anchor_lower[candidate] = float(
                np.sum(coordinate_lower)
                - np.sum(coordinate_lower[candidate])
                + np.sum(anchored_loss[candidate])
            )
        anchor_order = sorted(
            occupied, key=lambda value: (float(anchor_increase[value]), value)
        )
        clonal_cluster = int(
            anchor_order[0] if anchor_cluster is None else anchor_cluster
        )
        anchor_deviance_increase = float(anchor_increase[clonal_cluster])
        alternatives = [value for value in anchor_order if value != clonal_cluster]
        if alternatives:
            second_best_anchor_deviance_increase = float(
                anchor_increase[alternatives[0]]
            )
        if anchor_cluster is None:
            selected_lower_bound = float(np.min(anchor_lower[occupied]))
            selected_coordinates_certified = bool(np.all(coordinate_certified))
        else:
            selected_lower_bound = float(anchor_lower[clonal_cluster])
            selected_coordinates_certified = bool(
                np.all(
                    coordinate_certified[
                        np.arange(n_clusters) != int(clonal_cluster)
                    ]
                )
            )
        total_loss += float(anchor_increase[clonal_cluster])
        centers[clonal_cluster] = anchored_center[clonal_cluster]
        boundary_count -= int(np.sum(coordinate_boundary[clonal_cluster]))
        active_df -= int(np.sum(coordinate_active[clonal_cluster]))

    phi = centers[labels] if labels.size else np.empty((0, n_regions))
    global_gap = max(float(total_loss - selected_lower_bound), 0.0)
    global_certified = bool(
        selected_coordinates_certified
        and np.isfinite(total_loss)
        and np.isfinite(selected_lower_bound)
        and global_gap <= tol
    )
    path_suffix = (
        "_path_interval_certified" if data.path_likelihood is not None else ""
    )
    if normalized_anchor_mode == "none":
        loglik_source = "fixed_partition_observed_mle" + path_suffix
    elif anchor_cluster is None:
        loglik_source = "clonal_anchored_partition_observed_mle" + path_suffix
    else:
        loglik_source = (
            "raw_clonal_anchor_preserved_partition_observed_mle" + path_suffix
        )
    return PartitionRefitResult(
        phi=np.clip(phi, eps, upper_matrix).astype(np.float64, copy=False),
        cluster_centers=centers,
        loglik=float(-total_loss),
        fit_loss=float(total_loss),
        n_clusters=n_clusters,
        boundary_count=int(boundary_count),
        active_degrees_of_freedom=int(active_df),
        finite_candidate_found=bool(
            finite_coordinates == n_clusters * n_regions
            and np.isfinite(total_loss)
        ),
        refit_coordinate_count=n_clusters * n_regions,
        refit_finite_coordinate_count=int(finite_coordinates),
        refit_total_grid_points=0,
        refit_max_grid_spacing=0.0,
        refit_total_candidate_basins=0,
        refit_total_refined_candidates=int(certificate_intervals),
        refit_min_best_second_loss_gap=float("inf"),
        labels=labels.copy(),
        anchor_mode=normalized_anchor_mode,
        anchor_deviance_increase=anchor_deviance_increase,
        second_best_anchor_deviance_increase=second_best_anchor_deviance_increase,
        loglik_source=loglik_source,
        clonal_cluster=clonal_cluster,
        fixed_anchor_target=(
            None if clonal_cluster is None else centers[clonal_cluster].copy()
        ),
        fixed_anchor_block_signature=(
            fixed_anchor_block_signature if clonal_cluster is not None else "none"
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
    )


__all__ = ["PartitionRefitResult", "partition_constrained_observed_refit"]
