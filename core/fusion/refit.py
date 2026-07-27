from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...io.data import PathLikelihoodSpec, TumorData
from .starts import (
    _mutation_region_breakpoints,
    _golden_section_minimize,
    _sample_loss_grid_numpy,
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
    finite_candidate_found: (
        bool  # True only if a finite loss was obtained; does NOT certify global MLE
    )
    refit_coordinate_count: int
    refit_finite_coordinate_count: int
    refit_total_grid_points: int
    refit_max_grid_spacing: float
    refit_total_candidate_basins: int
    refit_total_refined_candidates: int
    refit_min_best_second_loss_gap: float
    loglik_source: str = "partition_constrained_observed_mle"

    @property
    def converged(self) -> bool:
        # Backward-compatible alias; use finite_candidate_found for new code.
        return self.finite_candidate_found


@dataclass(frozen=True)
class _RefitCoordinateResult:
    beta: float
    loss: float
    finite_candidate_found: bool
    grid_points: int
    max_grid_spacing: float
    candidate_basins: int
    refined_candidates: int
    best_second_loss_gap: float


def _canonical_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    remapped = np.empty_like(labels)
    label_map: dict[int, int] = {}
    next_label = 0
    for idx, value in enumerate(labels):
        key = int(value)
        label = label_map.get(key)
        if label is None:
            label = next_label
            label_map[key] = label
            next_label += 1
        remapped[idx] = label
    return remapped


def _objective_grid(
    values: np.ndarray,
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous: np.ndarray,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    return _sample_loss_grid_numpy(
        np.asarray(values, dtype=np.float64),
        alt=np.asarray(alt, dtype=np.float64),
        total=np.asarray(total, dtype=np.float64),
        b_minus=np.asarray(b_minus, dtype=np.float64),
        b_plus=np.asarray(b_plus, dtype=np.float64),
        b_fixed=np.asarray(b_fixed, dtype=np.float64),
        ambiguous=np.asarray(ambiguous, dtype=bool),
        major_prior=float(major_prior),
        eps=float(eps),
    )


def _path_objective_grid(
    values: np.ndarray,
    *,
    alt: np.ndarray,
    total: np.ndarray,
    scaling: np.ndarray,
    spec: PathLikelihoodSpec,
    eps: float,
) -> np.ndarray:
    """Vectorized fixed-prior path loss for one cluster-region coordinate."""

    beta = np.asarray(values, dtype=np.float64).reshape(1, -1, 1)
    alt = np.asarray(alt, dtype=np.float64).reshape(-1, 1, 1)
    nonalt = np.asarray(total, dtype=np.float64).reshape(-1, 1, 1) - alt
    scale = np.asarray(scaling, dtype=np.float64).reshape(-1, 1, 1)
    first = np.asarray(spec.first_copy[:, 0, :], dtype=np.float64)[:, None, :]
    second = np.asarray(spec.second_copy[:, 0, :], dtype=np.float64)[:, None, :]
    switch = np.asarray(spec.switch_fraction[:, 0, :], dtype=np.float64)[:, None, :]
    valid = np.asarray(spec.valid[:, 0, :], dtype=bool)[:, None, :]
    log_prior = np.asarray(spec.log_prior[:, 0, :], dtype=np.float64)[:, None, :]
    mass = scale * (
        first * np.minimum(beta, switch) + second * np.maximum(beta - switch, 0.0)
    )
    prob = np.clip(mass, float(eps), 1.0 - float(eps))
    log_joint = np.where(
        valid,
        alt * np.log(prob) + nonalt * np.log1p(-prob) + log_prior,
        -np.inf,
    )
    maximum = np.max(log_joint, axis=-1, keepdims=True)
    log_normalizer = maximum + np.log(
        np.sum(np.exp(log_joint - maximum), axis=-1, keepdims=True)
    )
    return -np.sum(log_normalizer[..., 0], axis=0)


def _path_cluster_region_breakpoints(
    *,
    alt: np.ndarray,
    total: np.ndarray,
    scaling: np.ndarray,
    spec: PathLikelihoodSpec,
    lower: float,
    upper: float,
    eps: float,
) -> np.ndarray:
    """Switches, inverse-VAF points, and clipping boundaries for path refits."""

    points: list[float] = [float(lower), float(upper)]
    first = np.asarray(spec.first_copy[:, 0, :], dtype=np.float64)
    second = np.asarray(spec.second_copy[:, 0, :], dtype=np.float64)
    switch = np.asarray(spec.switch_fraction[:, 0, :], dtype=np.float64)
    valid = np.asarray(spec.valid[:, 0, :], dtype=bool)
    scale = np.asarray(scaling, dtype=np.float64)[:, None]
    smoothed_vaf = (np.asarray(alt, dtype=np.float64)[:, None] + 0.5) / (
        np.asarray(total, dtype=np.float64)[:, None] + 1.0
    )

    def add_mass_inverse(target_mass: np.ndarray) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            left = target_mass / first
            right = switch + (target_mass - first * switch) / second
        left_ok = (
            valid
            & (first > 0.0)
            & np.isfinite(left)
            & (left >= lower)
            & (left <= np.minimum(switch, upper))
        )
        right_ok = (
            valid
            & (second > 0.0)
            & np.isfinite(right)
            & (right >= np.maximum(switch, lower))
            & (right <= upper)
        )
        points.extend(left[left_ok].tolist())
        points.extend(right[right_ok].tolist())

    points.extend(switch[valid & (switch >= lower) & (switch <= upper)].tolist())
    with np.errstate(divide="ignore", invalid="ignore"):
        add_mass_inverse(smoothed_vaf / scale)
        add_mass_inverse(np.full_like(first, float(eps)) / scale)
        add_mass_inverse(np.full_like(first, 1.0 - float(eps)) / scale)
    result = np.unique(np.round(np.asarray(points, dtype=np.float64), 14))
    return result[(result >= lower - 1e-12) & (result <= upper + 1e-12)]


def _cluster_region_candidate_grid(
    *,
    lower: float,
    upper: float,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous: np.ndarray,
    eps: float,
    hint: float | None,
) -> np.ndarray:
    if upper <= lower + 1e-12:
        return np.asarray([float(lower)], dtype=np.float64)

    points: list[float] = [float(lower), float(upper)]
    if hint is not None and np.isfinite(float(hint)):
        points.append(float(np.clip(float(hint), lower, upper)))

    points.extend(
        np.geomspace(
            max(float(lower), float(eps)), float(upper), num=97, dtype=np.float64
        ).tolist()
    )
    points.extend(
        np.linspace(float(lower), float(upper), num=49, dtype=np.float64).tolist()
    )

    for idx in range(int(b_minus.shape[0])):
        points.extend(
            _mutation_region_breakpoints(
                lower=float(lower),
                upper=float(upper),
                b_minus=float(b_minus[idx]),
                b_plus=float(b_plus[idx]),
                b_fixed=float(b_fixed[idx]),
                ambiguous=bool(ambiguous[idx]),
                eps=float(eps),
            ).tolist()
        )

    grid = np.unique(np.round(np.asarray(points, dtype=np.float64), 14))
    grid = grid[(grid >= float(lower) - 1e-12) & (grid <= float(upper) + 1e-12)]
    if grid.size == 0:
        return np.asarray([float(lower)], dtype=np.float64)
    return np.clip(grid, float(lower), float(upper))


def _path_cluster_region_candidate_grid(
    *,
    alt: np.ndarray,
    total: np.ndarray,
    scaling: np.ndarray,
    spec: PathLikelihoodSpec,
    lower: float,
    upper: float,
    eps: float,
    hint: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if upper <= lower + 1e-12:
        point = np.asarray([float(lower)], dtype=np.float64)
        return point, point
    hard_breakpoints = _path_cluster_region_breakpoints(
        alt=alt,
        total=total,
        scaling=scaling,
        spec=spec,
        lower=float(lower),
        upper=float(upper),
        eps=float(eps),
    )
    points: list[float] = hard_breakpoints.tolist()
    if hint is not None and np.isfinite(float(hint)):
        points.append(float(np.clip(float(hint), lower, upper)))
    points.extend(
        np.geomspace(
            max(float(lower), float(eps)),
            float(upper),
            num=129,
            dtype=np.float64,
        ).tolist()
    )
    points.extend(
        np.linspace(float(lower), float(upper), num=65, dtype=np.float64).tolist()
    )
    grid = np.unique(np.round(np.asarray(points, dtype=np.float64), 14))
    grid = grid[(grid >= lower - 1e-12) & (grid <= upper + 1e-12)]
    return np.clip(grid, lower, upper), hard_breakpoints


def _refit_cluster_region(
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous: np.ndarray,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint: float | None,
    path_spec: PathLikelihoodSpec | None = None,
    path_scaling: np.ndarray | None = None,
) -> _RefitCoordinateResult:
    lower = float(lower)
    upper = float(upper)

    def objective(values):
        if path_spec is not None:
            if path_scaling is None:
                raise ValueError("path_scaling is required with path_spec.")
            return _path_objective_grid(
                values,
                alt=alt,
                total=total,
                scaling=path_scaling,
                spec=path_spec,
                eps=eps,
            )
        return _objective_grid(
            values,
            alt=alt,
            total=total,
            b_minus=b_minus,
            b_plus=b_plus,
            b_fixed=b_fixed,
            ambiguous=ambiguous,
            major_prior=major_prior,
            eps=eps,
        )

    if upper <= lower + 1e-12:
        loss = float(objective(np.asarray([lower], dtype=np.float64))[0])
        return _RefitCoordinateResult(
            beta=lower,
            loss=loss,
            finite_candidate_found=bool(np.isfinite(loss)),
            grid_points=1,
            max_grid_spacing=0.0,
            candidate_basins=1 if np.isfinite(loss) else 0,
            refined_candidates=0,
            best_second_loss_gap=float("inf"),
        )

    if path_spec is None:
        grid = _cluster_region_candidate_grid(
            lower=lower,
            upper=upper,
            b_minus=b_minus,
            b_plus=b_plus,
            b_fixed=b_fixed,
            ambiguous=ambiguous,
            eps=eps,
            hint=hint,
        )
        hard_breakpoints = np.asarray([], dtype=np.float64)
    else:
        if path_scaling is None:
            raise ValueError("path_scaling is required with path_spec.")
        grid, hard_breakpoints = _path_cluster_region_candidate_grid(
            alt=alt,
            total=total,
            scaling=path_scaling,
            spec=path_spec,
            lower=lower,
            upper=upper,
            eps=eps,
            hint=hint,
        )
    losses = objective(grid)
    finite = np.isfinite(losses)
    max_grid_spacing = float(np.max(np.diff(grid))) if grid.size > 1 else 0.0
    if not np.any(finite):
        return _RefitCoordinateResult(
            beta=lower,
            loss=float("inf"),
            finite_candidate_found=False,
            grid_points=int(grid.size),
            max_grid_spacing=float(max_grid_spacing),
            candidate_basins=0,
            refined_candidates=0,
            best_second_loss_gap=float("inf"),
        )

    best_loss = float(np.min(losses[finite]))
    best_beta = float(grid[np.where(finite)[0][int(np.argmin(losses[finite]))]])
    candidate_losses: list[float] = []

    local_indices: set[int] = set()
    for idx in np.where(finite)[0].tolist():
        left_loss = (
            float(losses[idx - 1])
            if idx > 0 and np.isfinite(losses[idx - 1])
            else float("inf")
        )
        right_loss = (
            float(losses[idx + 1])
            if idx + 1 < losses.size and np.isfinite(losses[idx + 1])
            else float("inf")
        )
        if (
            float(losses[idx]) <= left_loss + 1e-10
            and float(losses[idx]) <= right_loss + 1e-10
        ):
            local_indices.add(int(idx))
    if not local_indices:
        local_indices.add(int(np.nanargmin(losses)))

    refined_candidates = 0
    selected_indices = sorted(local_indices, key=lambda value: float(losses[value]))[
        :16
    ]
    for idx in selected_indices:
        if np.isfinite(losses[idx]):
            candidate_losses.append(float(losses[idx]))
        center = float(grid[idx])
        is_hard_breakpoint = bool(
            hard_breakpoints.size
            and np.any(np.isclose(center, hard_breakpoints, rtol=0.0, atol=1e-12))
        )
        intervals = (
            [
                (float(grid[max(idx - 1, 0)]), center),
                (center, float(grid[min(idx + 1, grid.size - 1)])),
            ]
            if is_hard_breakpoint
            else [
                (
                    float(grid[max(idx - 1, 0)]),
                    float(grid[min(idx + 1, grid.size - 1)]),
                )
            ]
        )
        for left, right in intervals:
            if right <= left + 1e-12:
                continue
            refined_beta, refined_loss = _golden_section_minimize(
                objective,
                left=left,
                right=right,
                tol=float(tol),
                max_iter=max(int(max_iter), 32),
            )
            refined_candidates += 1
            if np.isfinite(refined_loss):
                candidate_losses.append(float(refined_loss))
            if np.isfinite(refined_loss) and refined_loss < best_loss:
                best_beta = float(refined_beta)
                best_loss = float(refined_loss)

    if hint is not None and np.isfinite(float(hint)):
        hint_beta = float(np.clip(float(hint), lower, upper))
        hint_loss = float(objective(np.asarray([hint_beta], dtype=np.float64))[0])
        if np.isfinite(hint_loss):
            candidate_losses.append(float(hint_loss))
        if np.isfinite(hint_loss) and hint_loss < best_loss:
            best_beta = hint_beta
            best_loss = hint_loss

    finite_candidate_losses = np.asarray(
        [value for value in candidate_losses if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite_candidate_losses.size >= 2:
        unique_losses = np.unique(np.round(finite_candidate_losses, 12))
        if unique_losses.size >= 2:
            best_second_loss_gap = float(unique_losses[1] - unique_losses[0])
        else:
            best_second_loss_gap = 0.0
    else:
        best_second_loss_gap = float("inf")

    return _RefitCoordinateResult(
        beta=float(np.clip(best_beta, lower, upper)),
        loss=float(best_loss),
        finite_candidate_found=bool(np.isfinite(best_loss)),
        grid_points=int(grid.size),
        max_grid_spacing=float(max_grid_spacing),
        candidate_basins=int(len(local_indices)),
        refined_candidates=int(refined_candidates),
        best_second_loss_gap=float(best_second_loss_gap),
    )


def partition_constrained_observed_refit(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint_phi: np.ndarray | None = None,
) -> PartitionRefitResult:
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition refit tolerance must be a positive finite value.")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != int(data.num_mutations):
        raise ValueError(
            "labels must contain one entry per tumor mutation "
            f"({int(data.num_mutations)})."
        )
    labels = _canonical_labels(labels)
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_regions = int(data.num_regions)
    centers = np.zeros((n_clusters, n_regions), dtype=np.float64)
    phi = np.zeros((int(data.num_mutations), n_regions), dtype=np.float64)

    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    b_minus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(
        data.minor_cn, dtype=np.float64
    )
    b_plus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(
        data.major_cn, dtype=np.float64
    )
    b_fixed = np.asarray(data.scaling, dtype=np.float64) * np.asarray(
        data.fixed_multiplicity, dtype=np.float64
    )
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    path_spec = getattr(data, "path_likelihood", None)
    upper_matrix = np.asarray(data.phi_upper, dtype=np.float64)
    hint_matrix = None if hint_phi is None else np.asarray(hint_phi, dtype=np.float64)
    # Only observed mutation_regions contribute to the likelihood, exactly as the torch fit
    # objective masks them (torch_backend.mutation_region_terms_torch). The BIC denominator
    # (effective_bic_mutation_region_count) also excludes them, so the refit loglik numerator
    # must too — otherwise BIC scores a model that was never fit.
    count_observed = getattr(data, "count_observed", None)
    observed_matrix = (
        None if count_observed is None else np.asarray(count_observed, dtype=bool)
    )

    total_loss = 0.0
    boundary_count = 0
    active_df = 0
    converged = True
    boundary_tol = max(float(tol) * 10.0, 1e-8)
    refit_coordinate_count = 0
    refit_finite_coordinate_count = 0
    refit_total_grid_points = 0
    refit_max_grid_spacing = 0.0
    refit_total_candidate_basins = 0
    refit_total_refined_candidates = 0
    refit_min_best_second_loss_gap = float("inf")

    for cluster_idx in range(n_clusters):
        member_mask = labels == int(cluster_idx)
        if not np.any(member_mask):
            continue
        member_rows = np.where(member_mask)[0]
        for region_idx in range(n_regions):
            lower = float(eps)
            # Feasibility box uses every member mutation_region; the likelihood uses only the
            # observed ones, so the collapsed center stays representable for all.
            upper = float(np.min(upper_matrix[member_rows, region_idx]))
            if not np.isfinite(upper) or upper < lower:
                upper = lower
            if observed_matrix is None:
                obs_rows = member_rows
            else:
                obs_rows = member_rows[observed_matrix[member_rows, region_idx]]
            if obs_rows.size == 0:
                # No observed mutation_region constrains this cluster/sample: zero likelihood
                # contribution (as in the fit), so the center is arbitrary in-box.
                # Match the torch refit's midpoint choice for a consistent phi.
                beta = float(0.5 * (lower + upper))
                centers[cluster_idx, region_idx] = beta
                refit_coordinate_count += 1
                refit_finite_coordinate_count += 1
                continue
            hint = None
            if hint_matrix is not None:
                hint = float(np.median(hint_matrix[obs_rows, region_idx]))
            coordinate = _refit_cluster_region(
                alt=alt[obs_rows, region_idx],
                total=total[obs_rows, region_idx],
                b_minus=b_minus[obs_rows, region_idx],
                b_plus=b_plus[obs_rows, region_idx],
                b_fixed=b_fixed[obs_rows, region_idx],
                ambiguous=ambiguous[obs_rows, region_idx],
                lower=lower,
                upper=upper,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=tol,
                max_iter=max(int(max_iter), 32),
                hint=hint,
                path_spec=(
                    None
                    if path_spec is None
                    else replace(
                        path_spec,
                        first_copy=path_spec.first_copy[
                            obs_rows, region_idx : region_idx + 1, :
                        ],
                        second_copy=path_spec.second_copy[
                            obs_rows, region_idx : region_idx + 1, :
                        ],
                        switch_fraction=path_spec.switch_fraction[
                            obs_rows, region_idx : region_idx + 1, :
                        ],
                        log_prior=path_spec.log_prior[
                            obs_rows, region_idx : region_idx + 1, :
                        ],
                        valid=path_spec.valid[obs_rows, region_idx : region_idx + 1, :],
                        legacy_major_indicator=(
                            None
                            if path_spec.legacy_major_indicator is None
                            else path_spec.legacy_major_indicator[
                                obs_rows, region_idx : region_idx + 1, :
                            ]
                        ),
                    )
                ),
                path_scaling=(
                    None
                    if path_spec is None
                    else np.asarray(data.scaling, dtype=np.float64)[
                        obs_rows, region_idx
                    ]
                ),
            )
            beta = float(coordinate.beta)
            loss = float(coordinate.loss)
            centers[cluster_idx, region_idx] = float(beta)
            total_loss += float(loss)
            converged = bool(converged and coordinate.finite_candidate_found)
            refit_coordinate_count += 1
            refit_finite_coordinate_count += int(coordinate.finite_candidate_found)
            refit_total_grid_points += int(coordinate.grid_points)
            refit_max_grid_spacing = max(
                refit_max_grid_spacing, float(coordinate.max_grid_spacing)
            )
            refit_total_candidate_basins += int(coordinate.candidate_basins)
            refit_total_refined_candidates += int(coordinate.refined_candidates)
            refit_min_best_second_loss_gap = min(
                refit_min_best_second_loss_gap,
                float(coordinate.best_second_loss_gap),
            )
            at_boundary = bool(
                beta <= lower + boundary_tol or beta >= upper - boundary_tol
            )
            boundary_count += int(at_boundary)
            active_df += int(not at_boundary)

    if labels.size:
        phi = centers[labels]
    return PartitionRefitResult(
        phi=np.clip(phi, float(eps), upper_matrix).astype(np.float64, copy=False),
        cluster_centers=centers.astype(np.float64, copy=False),
        loglik=float(-total_loss),
        fit_loss=float(total_loss),
        n_clusters=int(n_clusters),
        boundary_count=int(boundary_count),
        active_degrees_of_freedom=int(active_df),
        finite_candidate_found=bool(converged and np.isfinite(total_loss)),
        refit_coordinate_count=int(refit_coordinate_count),
        refit_finite_coordinate_count=int(refit_finite_coordinate_count),
        refit_total_grid_points=int(refit_total_grid_points),
        refit_max_grid_spacing=float(refit_max_grid_spacing),
        refit_total_candidate_basins=int(refit_total_candidate_basins),
        refit_total_refined_candidates=int(refit_total_refined_candidates),
        refit_min_best_second_loss_gap=float(refit_min_best_second_loss_gap),
        loglik_source=(
            "partition_constrained_observed_mle"
            if path_spec is None
            else "partition_constrained_observed_mle_path_multibasin"
        ),
    )


__all__ = [
    "PartitionRefitResult",
    "partition_constrained_observed_refit",
]
