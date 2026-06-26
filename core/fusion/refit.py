from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData
from .starts import _cell_breakpoints, _golden_section_minimize, _sample_loss_grid_numpy


@dataclass(frozen=True)
class PartitionRefitResult:
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    fit_loss: float
    n_clusters: int
    boundary_count: int
    active_degrees_of_freedom: int
    converged: bool
    loglik_source: str = "partition_constrained_observed_mle"


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


def _cluster_sample_candidate_grid(
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

    points.extend(np.geomspace(max(float(lower), float(eps)), float(upper), num=97, dtype=np.float64).tolist())
    points.extend(np.linspace(float(lower), float(upper), num=49, dtype=np.float64).tolist())

    for idx in range(int(b_minus.shape[0])):
        points.extend(
            _cell_breakpoints(
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


def _refit_cluster_sample(
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
) -> tuple[float, float, bool]:
    lower = float(lower)
    upper = float(upper)
    if upper <= lower + 1e-12:
        loss = float(
            _objective_grid(
                np.asarray([lower], dtype=np.float64),
                alt=alt,
                total=total,
                b_minus=b_minus,
                b_plus=b_plus,
                b_fixed=b_fixed,
                ambiguous=ambiguous,
                major_prior=major_prior,
                eps=eps,
            )[0]
        )
        return lower, loss, np.isfinite(loss)

    objective = lambda values: _objective_grid(
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
    grid = _cluster_sample_candidate_grid(
        lower=lower,
        upper=upper,
        b_minus=b_minus,
        b_plus=b_plus,
        b_fixed=b_fixed,
        ambiguous=ambiguous,
        eps=eps,
        hint=hint,
    )
    losses = objective(grid)
    finite = np.isfinite(losses)
    if not np.any(finite):
        return lower, float("inf"), False

    best_loss = float(np.min(losses[finite]))
    best_beta = float(grid[np.where(finite)[0][int(np.argmin(losses[finite]))]])

    local_indices: set[int] = set()
    for idx in np.where(finite)[0].tolist():
        left_loss = float(losses[idx - 1]) if idx > 0 and np.isfinite(losses[idx - 1]) else float("inf")
        right_loss = float(losses[idx + 1]) if idx + 1 < losses.size and np.isfinite(losses[idx + 1]) else float("inf")
        if float(losses[idx]) <= left_loss + 1e-10 and float(losses[idx]) <= right_loss + 1e-10:
            local_indices.add(int(idx))
    if not local_indices:
        local_indices.add(int(np.nanargmin(losses)))

    for idx in sorted(local_indices, key=lambda value: float(losses[value]))[:16]:
        left = float(grid[max(idx - 1, 0)])
        right = float(grid[min(idx + 1, grid.size - 1)])
        if right <= left + 1e-12:
            continue
        refined_beta, refined_loss = _golden_section_minimize(
            objective,
            left=left,
            right=right,
            tol=float(tol),
            max_iter=max(int(max_iter), 32),
        )
        if np.isfinite(refined_loss) and refined_loss < best_loss:
            best_beta = float(refined_beta)
            best_loss = float(refined_loss)

    if hint is not None and np.isfinite(float(hint)):
        hint_beta = float(np.clip(float(hint), lower, upper))
        hint_loss = float(objective(np.asarray([hint_beta], dtype=np.float64))[0])
        if np.isfinite(hint_loss) and hint_loss < best_loss:
            best_beta = hint_beta
            best_loss = hint_loss

    return float(np.clip(best_beta, lower, upper)), float(best_loss), bool(np.isfinite(best_loss))


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
    labels = _canonical_labels(np.asarray(labels, dtype=np.int64))
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    n_samples = int(data.num_samples)
    centers = np.zeros((n_clusters, n_samples), dtype=np.float64)
    phi = np.zeros((int(data.num_mutations), n_samples), dtype=np.float64)

    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    b_minus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64)
    b_plus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64)
    b_fixed = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    upper_matrix = np.asarray(data.phi_upper, dtype=np.float64)
    hint_matrix = None if hint_phi is None else np.asarray(hint_phi, dtype=np.float64)

    total_loss = 0.0
    boundary_count = 0
    active_df = 0
    converged = True
    boundary_tol = max(float(tol) * 10.0, 1e-8)

    for cluster_idx in range(n_clusters):
        member_mask = labels == int(cluster_idx)
        if not np.any(member_mask):
            continue
        for sample_idx in range(n_samples):
            lower = float(eps)
            upper = float(np.min(upper_matrix[member_mask, sample_idx]))
            if not np.isfinite(upper) or upper < lower:
                upper = lower
            hint = None
            if hint_matrix is not None:
                hint = float(np.median(hint_matrix[member_mask, sample_idx]))
            beta, loss, ok = _refit_cluster_sample(
                alt=alt[member_mask, sample_idx],
                total=total[member_mask, sample_idx],
                b_minus=b_minus[member_mask, sample_idx],
                b_plus=b_plus[member_mask, sample_idx],
                b_fixed=b_fixed[member_mask, sample_idx],
                ambiguous=ambiguous[member_mask, sample_idx],
                lower=lower,
                upper=upper,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=tol,
                max_iter=max(int(max_iter), 32),
                hint=hint,
            )
            centers[cluster_idx, sample_idx] = float(beta)
            total_loss += float(loss)
            converged = bool(converged and ok)
            at_boundary = bool(beta <= lower + boundary_tol or beta >= upper - boundary_tol)
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
        converged=bool(converged and np.isfinite(total_loss)),
    )


__all__ = [
    "PartitionRefitResult",
    "partition_constrained_observed_refit",
]
