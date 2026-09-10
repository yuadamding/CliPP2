from __future__ import annotations

import heapq
from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ...io.data import TumorData
from ...config import (
    DIRICHLET_ALPHA,
    DIRICHLET_CODE_WEIGHT,
    FitConfig,
    LIKELIHOOD_PARTITION_K_MAX,
    PARTITION_CEM_MAX_ITER,
    PARTITION_GENERATION_REFIT_MAX_ITER,
    PARTITION_K_ANCHORS,
    PARTITION_MAX_CANDIDATES_PER_K,
)
from ..objective import (
    ObservedModel, TorchObservedModel, compile_observed_model, _observed_reduction_numpy,
    observed_loss_grid_torch,
)
from ..bic import fixed_partition_dirichlet_score
from ..scalar import (
    PartitionRefitResult,
    _RefitCoordinateCache,
    _ScalarWorkStats,
    canonical_partition_labels as _canonical_labels,
    partition_constrained_observed_refit,
)
from .torch_backend import as_runtime_tensor
from .types import PreparedProblem


# Bound each temporary used to initialize the dense Ward cost matrix.  The
# unchunked broadcast has shape (M, M, S), so its memory grows by several
# copies of M^2*S even though the persistent Ward state is only O(M^2).
_WARD_INITIAL_PAIRWISE_WORK_ELEMENTS = 4_000_000
# Refresh gathers need no region axis. Each temporary holds at most this
# many elements, except that one complete active row is the minimum unit.
_WARD_REFRESH_WORK_ELEMENTS = 1_000_000
_WARD_HEAP_ACTIVE_MULTIPLIER = 4
_WARD_HEAP_MIN_ENTRIES = 64


def _ward_refresh_minima(
    cost_matrix: torch.Tensor,
    invalid_rows: np.ndarray,
    active_ids: np.ndarray,
    node_slot: np.ndarray,
    row_best_cost: np.ndarray,
    row_best_column: np.ndarray,
    work_elements: int,
) -> None:
    """Bound each gather by max(work_elements, active columns), preserving ties."""
    active_slots = torch.as_tensor(node_slot[active_ids], device=cost_matrix.device)
    rows_per_chunk = max(1, int(work_elements) // int(active_ids.size))
    for start in range(0, int(invalid_rows.size), rows_per_chunk):
        rows = invalid_rows[start:start + rows_per_chunk]
        slots = torch.as_tensor(node_slot[rows], device=cost_matrix.device)
        # Slot order differs from logical ID order after reuse. The complete
        # active-column axis stays in logical order in every batch, so min's
        # first exact tie remains the smallest logical partner.
        costs, columns = torch.min(
            cost_matrix[slots[:, None], active_slots[None, :]], dim=1
        )
        row_best_cost[rows] = costs.detach().cpu().numpy()
        row_best_column[rows] = active_ids[columns.detach().cpu().numpy()]


def _ward_compact_heap(
    row_heap: list,
    active_cpu: np.ndarray,
    row_best_cost: np.ndarray,
    row_best_column: np.ndarray,
    row_version: np.ndarray,
    finite_large: float,
) -> None:
    """Discard stale entries without changing the logical lexicographic order."""
    row_heap[:] = [
        (float(row_best_cost[row]), int(row), int(row_best_column[row]), int(row_version[row]))
        for row in np.flatnonzero(active_cpu)
        if float(row_best_cost[row]) < finite_large * 0.5
    ]
    heapq.heapify(row_heap)


@dataclass(frozen=True)
class PartitionCandidate:
    labels: np.ndarray
    K: int
    source: str
    phi_start: np.ndarray
    fit_loss: float
    bic: float
    finite_candidate_found: bool = True
    requested_k: int | None = None
    component_death_count: int = 0


@torch.no_grad()
def observed_curvature_at_pilot_torch(
    model: TorchObservedModel,
    pilot_phi: torch.Tensor,
    *,
    eps: float,
    step_fraction: float = 1e-3,
    min_step: float = 1e-4,
    curvature_floor: float = 1e-6,
    curvature_cap_quantile: float = 0.995,
) -> torch.Tensor:
    """Evaluate a controlled runtime view; preparation owns source integrity."""
    if not isinstance(model, TorchObservedModel) or not torch.is_tensor(pilot_phi):
        raise TypeError("Curvature requires a TorchObservedModel and a pilot Tensor.")
    if tuple(pilot_phi.shape) != model.shape:
        raise ValueError("Curvature pilot and model must have the same shape.")
    if pilot_phi.dtype != model.alt.dtype or pilot_phi.device != model.alt.device:
        raise ValueError("Curvature pilot and model must share dtype and device.")
    phi0 = pilot_phi
    upper = model.upper
    lower_value = float(eps)
    lower = torch.full_like(phi0, lower_value)
    x0 = torch.minimum(torch.maximum(phi0, lower), upper)
    width = torch.clamp(upper - lower_value, min=0.0)
    step_base = torch.maximum(torch.maximum(width, torch.abs(x0)), torch.ones_like(x0))
    step = torch.maximum(
        torch.full_like(x0, float(min_step)),
        float(step_fraction) * step_base,
    )
    left = torch.maximum(lower, x0 - step)
    right = torch.minimum(upper, x0 + step)
    h_left = x0 - left
    h_right = right - x0
    valid = (h_left > 1e-12) & (h_right > 1e-12)

    f_left = observed_loss_grid_torch(
        model, left, eps=eps
    )
    f0 = observed_loss_grid_torch(
        model, x0, eps=eps
    )
    f_right = observed_loss_grid_torch(
        model, right, eps=eps
    )
    denom = h_left * h_right * (h_left + h_right)
    curvature = (
        2.0 * (h_left * f_right - (h_left + h_right) * f0 + h_right * f_left) / denom
    )
    floor = torch.full_like(curvature, float(curvature_floor))
    curvature = torch.where(
        valid & torch.isfinite(curvature), torch.maximum(curvature, floor), floor
    )

    finite = curvature[torch.isfinite(curvature)]
    if finite.numel() and 0.0 < float(curvature_cap_quantile) < 1.0:
        cap = torch.quantile(finite, float(curvature_cap_quantile))
        if bool(torch.isfinite(cap).item()) and float(cap.item()) > float(
            curvature_floor
        ):
            curvature = torch.minimum(curvature, cap)
    return torch.maximum(curvature, floor)


@torch.no_grad()
def hessian_weighted_ward_label_sets_torch(
    pilot_phi: torch.Tensor,
    curvature: torch.Tensor,
    *,
    K_grid: Sequence[int],
    initial_pairwise_work_elements: int = _WARD_INITIAL_PAIRWISE_WORK_ELEMENTS,
    refresh_work_elements: int = _WARD_REFRESH_WORK_ELEMENTS,
) -> dict[int, np.ndarray]:
    """Run Ward on prepared tensors; proposal boundaries own normalization."""
    if not torch.is_tensor(pilot_phi) or not torch.is_tensor(curvature):
        raise TypeError("Ward requires pilot and curvature Tensors.")
    if pilot_phi.ndim != 2 or pilot_phi.shape != curvature.shape:
        raise ValueError("Ward pilot and curvature must have the same two-dimensional shape.")
    if pilot_phi.dtype != curvature.dtype or pilot_phi.device != curvature.device:
        raise ValueError("Ward pilot and curvature must share dtype and device.")
    if pilot_phi.dtype not in (torch.float32, torch.float64):
        raise ValueError("Ward requires float32 or float64 tensors.")
    phi0, h = pilot_phi, curvature
    num_mutations = int(phi0.shape[0])
    requested = {int(k) for k in K_grid if 1 <= int(k) <= num_mutations}
    if not requested:
        return {}

    num_regions = int(phi0.shape[1])
    if int(initial_pairwise_work_elements) < 1:
        raise ValueError("initial_pairwise_work_elements must be positive.")
    if int(refresh_work_elements) < 1:
        raise ValueError("refresh_work_elements must be positive.")
    max_nodes = max(2 * num_mutations - 1, 1)
    H = phi0.new_zeros((max_nodes, num_regions))
    mu = torch.zeros_like(H)
    H[:num_mutations] = h
    mu[:num_mutations] = phi0
    mutation_cluster = torch.arange(
        num_mutations, dtype=torch.long, device=phi0.device
    )

    finite_large = torch.finfo(phi0.dtype).max / 16.0
    # Merge IDs remain immutable and increasing, but only M physical rows are
    # needed. Reuse the left child's slot; a row stores costs only to greater
    # logical IDs, irrespective of its physical column order.
    cost_matrix = phi0.new_full((num_mutations, num_mutations), finite_large)
    node_slot = np.full(max_nodes, -1, dtype=np.int64)
    node_slot[:num_mutations] = np.arange(num_mutations)
    # Compute the exact singleton Ward costs in row blocks.  This retains the
    # same cost arithmetic and merge order while avoiding simultaneous
    # (M, M, S) denominator, weight, difference, and product tensors.  It is
    # particularly important for multi-region cohorts with thousands of
    # mutations, where the former initializer could exhaust a 10-GiB GPU
    # before the ADMM fit started.
    pair_region_elements_per_row = max(num_mutations * num_regions, 1)
    initial_row_chunk = max(
        1,
        min(
            num_mutations,
            int(initial_pairwise_work_elements) // pair_region_elements_per_row,
        ),
    )
    all_columns = torch.arange(num_mutations, dtype=torch.long, device=phi0.device)
    H_initial = H[:num_mutations]
    mu_initial = mu[:num_mutations]
    tiny = torch.finfo(phi0.dtype).tiny
    for row_start in range(0, num_mutations, initial_row_chunk):
        row_stop = min(row_start + initial_row_chunk, num_mutations)
        H_left = H_initial[row_start:row_stop].unsqueeze(1)
        denom = H_left + H_initial.unsqueeze(0)
        weight = H_left * H_initial.unsqueeze(0)
        weight.div_(denom.clamp_min(tiny))
        weight.masked_fill_(denom <= 0.0, 0.0)
        diff = mu_initial[row_start:row_stop].unsqueeze(1) - mu_initial.unsqueeze(0)
        diff.square_().mul_(weight)
        initial_cost = 0.5 * torch.sum(diff, dim=2)
        row_ids = torch.arange(
            row_start, row_stop, dtype=torch.long, device=phi0.device
        )
        upper_mask = all_columns.unsqueeze(0) > row_ids.unsqueeze(1)
        cost_matrix[row_start:row_stop, :num_mutations] = torch.where(
            upper_mask,
            initial_cost,
            finite_large,
        )
        # Do not retain one block while allocating the next, or keep the last
        # block alive throughout merging alongside later refresh temporaries.
        del denom, weight, diff, initial_cost, row_ids, upper_mask

    # One exact row minimum, ranked by (cost, logical left ID, logical right
    # ID), reproduces the original full logical-matrix row-major argmin.
    # Share this maintenance across devices and region counts: no full-matrix
    # scan is required on every multi-region CPU merge.
    initial_row_cost, initial_row_column = torch.min(cost_matrix, dim=1)
    row_best_cost = np.full(max_nodes, finite_large, dtype=np.float64)
    row_best_column = np.zeros(max_nodes, dtype=np.int64)
    row_best_cost[:num_mutations] = initial_row_cost.detach().cpu().numpy()
    row_best_column[:num_mutations] = initial_row_column.detach().cpu().numpy()
    row_version = np.zeros(max_nodes, dtype=np.int64)
    row_heap = [
        (float(row_best_cost[row]), row, int(row_best_column[row]), 0)
        for row in range(num_mutations)
        if float(row_best_cost[row]) < finite_large * 0.5
    ]
    heapq.heapify(row_heap)

    def current_labels() -> np.ndarray:
        return _canonical_labels(
            mutation_cluster.detach().cpu().numpy().astype(np.int64, copy=False)
        )

    out: dict[int, np.ndarray] = {}
    active_count = num_mutations
    if active_count in requested:
        out[active_count] = current_labels()

    next_cluster_id = num_mutations
    active_cpu = np.zeros((max_nodes,), dtype=bool)
    active_cpu[:num_mutations] = True
    while active_count > 1 and requested - set(out):
        while row_heap:
            min_cost, left, right, version = heapq.heappop(row_heap)
            if (
                active_cpu[left]
                and active_cpu[right]
                and int(row_version[left]) == int(version)
                and int(row_best_column[left]) == int(right)
            ):
                break
        else:
            min_cost = float("inf")
            left = right = -1
        if not np.isfinite(min_cost) or min_cost >= finite_large * 0.5:
            raise RuntimeError(
                "Hessian-weighted Ward cost matrix exhausted before all clusters were merged."
            )
        new_id = next_cluster_id
        next_cluster_id += 1

        H_new = H[left] + H[right]
        H[new_id] = H_new
        mu[new_id] = torch.where(
            H_new > 0.0,
            (H[left] * mu[left] + H[right] * mu[right])
            / H_new.clamp_min(torch.finfo(phi0.dtype).tiny),
            0.5 * (mu[left] + mu[right]),
        )
        mutation_cluster = torch.where(
            (mutation_cluster == left) | (mutation_cluster == right),
            torch.full_like(mutation_cluster, new_id),
            mutation_cluster,
        )

        active_cpu[left] = False
        active_cpu[right] = False
        active_cpu[new_id] = True
        new_slot, right_slot = int(node_slot[left]), int(node_slot[right])
        node_slot[new_id] = new_slot
        cost_matrix[new_slot, :] = finite_large
        cost_matrix[:, new_slot] = finite_large
        cost_matrix[right_slot, :] = finite_large
        cost_matrix[:, right_slot] = finite_large

        other_ids = np.flatnonzero(active_cpu[:new_id])
        other = torch.as_tensor(other_ids, dtype=torch.long, device=phi0.device)
        if other.numel():
            denom_vec = H[new_id].unsqueeze(0) + H[other]
            weight_vec = torch.where(
                denom_vec > 0.0,
                H[new_id].unsqueeze(0)
                * H[other]
                / denom_vec.clamp_min(torch.finfo(phi0.dtype).tiny),
                torch.zeros_like(denom_vec),
            )
            diff_vec = mu[new_id].unsqueeze(0) - mu[other]
            cost_vec = 0.5 * torch.sum(weight_vec * torch.square(diff_vec), dim=1)
            other_slots = torch.as_tensor(node_slot[other_ids], device=phi0.device)
            cost_matrix[other_slots, new_slot] = cost_vec
            cost_values = cost_vec.detach().cpu().numpy()
            invalid_best = np.isin(
                row_best_column[other_ids],
                np.asarray([left, right], dtype=np.int64),
            )
            invalid_rows = other_ids[invalid_best]
            direct_rows = other_ids[
                (~invalid_best) & (cost_values < row_best_cost[other_ids])
            ]

            if invalid_rows.size:
                _ward_refresh_minima(
                    cost_matrix, invalid_rows, np.append(other_ids, new_id),
                    node_slot, row_best_cost, row_best_column, int(refresh_work_elements),
                )
            if direct_rows.size:
                direct_positions = np.searchsorted(other_ids, direct_rows)
                row_best_cost[direct_rows] = cost_values[direct_positions]
                row_best_column[direct_rows] = new_id

            for row in np.concatenate((invalid_rows, direct_rows)):
                row = int(row)
                row_version[row] += 1
                if float(row_best_cost[row]) < finite_large * 0.5:
                    heapq.heappush(
                        row_heap,
                        (
                            float(row_best_cost[row]),
                            row,
                            int(row_best_column[row]),
                            int(row_version[row]),
                        ),
                    )

        active_count -= 1
        # At most active_count-1 rows were pushed in this merge. Rebuilding
        # here bounds stale Python heap storage by O(M), including the temporary
        # replacement list; it does not alter cost arithmetic or tie order.
        if len(row_heap) > max(
            _WARD_HEAP_MIN_ENTRIES, _WARD_HEAP_ACTIVE_MULTIPLIER * active_count
        ):
            _ward_compact_heap(
                row_heap, active_cpu, row_best_cost, row_best_column,
                row_version, finite_large,
            )
        if active_count in requested:
            out[active_count] = current_labels()
    return out


def _loss_to_centers(
    data: TumorData,
    centers: np.ndarray,
    *,

    eps: float,
    infeasible_penalty: float = 1e100,
    _model: ObservedModel | None = None,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    model = (
        compile_observed_model(data, eps=eps)
        if _model is None
        else _model
    )
    num_mutations = int(data.num_mutations)
    num_clusters = int(centers.shape[0])
    cost = np.zeros((num_mutations, num_clusters), dtype=np.float64)
    infeasible = np.zeros((num_mutations, num_clusters), dtype=bool)

    for cluster_idx in range(num_clusters):
        phi_for_center = np.broadcast_to(centers[cluster_idx], model.shape)
        loss = _observed_reduction_numpy(model, phi_for_center, eps=float(eps), output="loss")
        cost[:, cluster_idx] = np.sum(loss, axis=1)
        infeasible[:, cluster_idx] = np.any(
            phi_for_center > model.upper + max(float(eps), 1e-8), axis=1
        )

    cost[infeasible] = float(infeasible_penalty)
    return cost


def _repair_empty_clusters(labels: np.ndarray, cost: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).copy()
    cost = np.asarray(cost)
    if np.issubdtype(cost.dtype, np.floating):
        infeasible_cutoff = min(1e99, float(np.finfo(cost.dtype).max) / 64.0)
    else:
        infeasible_cutoff = 1e99
    num_clusters = int(cost.shape[1])
    for cluster_idx in range(num_clusters):
        if np.any(labels == cluster_idx):
            continue
        counts = np.bincount(labels, minlength=num_clusters)
        donor_mask = counts[labels] > 1
        if not np.any(donor_mask):
            break
        donor_indices = np.where(donor_mask)[0]
        current_cost = cost[donor_indices, labels[donor_indices]]
        target_cost = cost[donor_indices, cluster_idx]
        finite_target = np.isfinite(target_cost) & (target_cost < infeasible_cutoff)
        if np.any(finite_target):
            gains = current_cost[finite_target] - target_cost[finite_target]
            selected = donor_indices[finite_target][int(np.argmax(gains))]
        else:
            selected = donor_indices[int(np.argmax(current_cost))]
        labels[int(selected)] = int(cluster_idx)
    return labels


def _classification_leave_one_out_log_cluster_weights(
    labels: np.ndarray,
    *,
    num_clusters: int,
) -> np.ndarray:
    """Return each mutation's Dirichlet conditional log cluster weights.

    For mutation ``i`` and block ``k``, the integrated allocation conditional is
    ``(n[k, -i] + alpha) / (n - 1 + K * alpha)``. The mutation being reassigned
    must therefore be removed from its current block count. Using the full count
    would create a self-reinforcing update that is not a conditional move under
    the Dirichlet-integrated exact-partition score.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    num_clusters = int(num_clusters)
    alpha = DIRICHLET_ALPHA
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if labels.size == 0:
        raise ValueError("classification cluster weights require at least one label.")
    if np.any(labels < 0) or np.any(labels >= num_clusters):
        raise ValueError("labels must be in [0, num_clusters).")

    counts = np.bincount(labels, minlength=num_clusters).astype(np.float64, copy=False)
    leave_one_out_counts = np.broadcast_to(
        counts[None, :], (labels.size, num_clusters)
    ).copy()
    leave_one_out_counts[np.arange(labels.size), labels] -= 1.0
    probabilities = (leave_one_out_counts + alpha) / (
        float(labels.size - 1) + alpha * float(num_clusters)
    )
    return np.log(probabilities)


def _classification_assignment_cost(
    count_cost: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Add the weighted negative log allocation term to assignment costs."""
    count_cost = np.asarray(count_cost)
    if count_cost.ndim != 2 or count_cost.shape[1] <= 0:
        raise ValueError("count_cost must be a non-empty mutation-by-cluster matrix.")
    if np.asarray(labels).reshape(-1).size != int(count_cost.shape[0]):
        raise ValueError("labels must contain one entry per mutation cost row.")
    log_weights = _classification_leave_one_out_log_cluster_weights(
        labels,
        num_clusters=int(count_cost.shape[1]),
    )
    return count_cost - DIRICHLET_CODE_WEIGHT * log_weights


def _classification_refit_score(
    data: TumorData,
    labels: np.ndarray,
    refit: PartitionRefitResult,
) -> float:
    return fixed_partition_dirichlet_score(
        loglik=float(refit.loglik),
        num_clusters=int(refit.n_clusters),
        labels=labels,
        partition_signature="",
        data=data,
    ).value


def _classification_score_strictly_improves(
    proposed_score: float,
    current_score: float,
) -> bool:
    """Require a deterministic improvement before accepting a CEM update."""

    proposed = float(proposed_score)
    current = float(current_score)
    if not np.isfinite(proposed):
        return False
    if not np.isfinite(current):
        return True
    tolerance = 64.0 * np.finfo(np.float64).eps * (1.0 + abs(proposed) + abs(current))
    return bool(proposed < current - tolerance)


def _validated_refinement_labels(
    data: TumorData,
    labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != int(data.num_mutations):
        raise ValueError(
            "labels must contain one entry per tumor mutation "
            f"({int(data.num_mutations)})."
        )
    return _canonical_labels(labels)


def refine_partition_likelihood(
    data: TumorData,
    labels: np.ndarray,
    *,

    eps: float,
    tol: float,
    max_iter: int = PARTITION_CEM_MAX_ITER,
    refit_max_iter: int = PARTITION_GENERATION_REFIT_MAX_ITER,
    _refit_labels: Callable[[np.ndarray], PartitionRefitResult] | None = None,
    _model: ObservedModel | None = None,
) -> PartitionRefitResult:
    """Host CEM with fixed allocation scoring and empty-cluster repair."""
    labels = _validated_refinement_labels(data, labels)
    model = (
        compile_observed_model(data, eps=eps)
        if _model is None
        else _model
    )

    def refit_labels(current_labels: np.ndarray) -> PartitionRefitResult:
        if _refit_labels is not None:
            return _refit_labels(current_labels)
        return partition_constrained_observed_refit(
            data,
            current_labels,
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(refit_max_iter), 32),
            _model=model,
        )

    refit = refit_labels(labels)
    score = _classification_refit_score(data, labels, refit)
    for _ in range(max(int(max_iter), 0)):
        labels_key = _label_key(labels)
        count_cost = _loss_to_centers(
            data,
            refit.cluster_centers,
            eps=float(eps),
            _model=model,
        )
        assignment_cost = _classification_assignment_cost(count_cost, labels)
        labels_next = np.argmin(assignment_cost, axis=1).astype(np.int64, copy=False)
        labels_next = _repair_empty_clusters(labels_next, assignment_cost)
        labels_next = _canonical_labels(labels_next)
        if _label_key(labels_next) == labels_key:
            labels = labels_next
            break
        proposed_refit = refit_labels(labels_next)
        proposed_score = _classification_refit_score(data, labels_next, proposed_refit)
        # Simultaneous reassignment is only a proposal: admit it after its
        # fixed-label refit improves the same declared score.
        if not _classification_score_strictly_improves(proposed_score, score):
            break
        score = float(proposed_score)
        refit = proposed_refit
        labels = labels_next
    return refit


def _label_key(labels: np.ndarray) -> bytes:
    labels = _canonical_labels(labels)
    return labels.astype(np.int32, copy=False).tobytes()


def generate_likelihood_partition_starts(
    data: TumorData,
    *,
    eps: float,
    label_sets: dict[int, np.ndarray],
    max_candidates_per_K: int = PARTITION_MAX_CANDIDATES_PER_K,
    cem_max_iter: int = PARTITION_CEM_MAX_ITER,
    refit_max_iter: int = PARTITION_GENERATION_REFIT_MAX_ITER,
    tol: float = 1e-3,
    _work_stats: _ScalarWorkStats | None = None,
) -> list[PartitionCandidate]:
    """Refit plain Ward and host-CEM proposals under one fixed score policy."""
    label_sets = {
        int(k): _canonical_labels(np.asarray(labels, dtype=np.int64))
        for k, labels in label_sets.items()
        if 1 <= int(k) <= int(data.num_mutations)
    }
    candidates: list[PartitionCandidate] = []
    seen: set[bytes] = set()
    source_model = compile_observed_model(data, eps=float(eps))
    # One model, tolerance and scalar backend per call: immutable labels alone
    # identify each local refit, including repeated CEM proposals.
    refit_cache: dict[bytes, PartitionRefitResult] = {}
    coordinate_cache = _RefitCoordinateCache(work_stats=_work_stats)

    def cached_refit(labels: np.ndarray) -> PartitionRefitResult:
        labels_key = _label_key(labels)
        cached = refit_cache.get(labels_key)
        if cached is not None:
            return cached
        result = partition_constrained_observed_refit(
            data, labels, eps=float(eps), tol=float(tol),
            max_iter=max(int(refit_max_iter), 32), _model=source_model,
            _coordinate_cache=coordinate_cache,
        )
        refit_cache[labels_key] = result
        return result

    for requested_k in sorted(label_sets):
        labels0 = _canonical_labels(label_sets[int(requested_k)])
        for source in (f"hessian_ward_K{requested_k}", f"hessian_ward_cem_K{requested_k}"):
            if source.startswith("hessian_ward_cem"):
                refit = refine_partition_likelihood(
                    data, labels0, eps=float(eps), tol=float(tol),
                    max_iter=int(cem_max_iter), refit_max_iter=int(refit_max_iter),
                    _refit_labels=cached_refit, _model=source_model,
                )
            else:
                refit = cached_refit(labels0)
            labels_used = refit.labels
            key = _label_key(labels_used)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(PartitionCandidate(
                labels=_canonical_labels(labels_used), K=int(refit.n_clusters), source=source,
                phi_start=refit.phi, fit_loss=float(refit.fit_loss),
                bic=_classification_refit_score(data, labels_used, refit),
                finite_candidate_found=bool(refit.finite_candidate_found),
                requested_k=int(requested_k),
            ))

    by_k: dict[int, list[PartitionCandidate]] = {}
    for candidate in candidates:
        by_k.setdefault(int(candidate.K), []).append(candidate)
    kept: list[PartitionCandidate] = []
    for values in by_k.values():
        values = sorted(values, key=lambda item: (float(item.bic), float(item.fit_loss), str(item.source)))
        kept.extend(values[:max(int(max_candidates_per_K), 1)])
    return sorted(kept, key=lambda item: (float(item.bic), int(item.K), str(item.source)))


def generate_partition_initializer_pool(
    *,
    context: PreparedProblem,
    pilot_phi: np.ndarray | torch.Tensor,
    fit_options: FitConfig,
    curvature: np.ndarray | torch.Tensor | None = None,
    declared_k_grid: tuple[int, ...] | None = None,
) -> tuple[PartitionCandidate, ...]:
    """Generate the deterministic pilot or final-Phi Ward/host-CEM pool.

    These scored proposals supply the raw guide and the independent direct
    candidate pool; final selection still refits labels under its own gate.
    """
    context.validate(allow_deferred_graph=True)
    if float(fit_options.eps) != context.eps:
        raise ValueError("Partition options must preserve the prepared likelihood epsilon.")
    data, runtime = context.source_data, context.runtime
    pilot_tensor = as_runtime_tensor(pilot_phi, runtime)
    if declared_k_grid is None:
        k_grid = [int(k) for k in PARTITION_K_ANCHORS if 1 <= int(k) <= int(data.num_mutations)]
        k_cap = min(int(LIKELIHOOD_PARTITION_K_MAX), int(data.num_mutations))
        if k_cap > 0 and k_cap not in k_grid:
            k_grid.append(k_cap)
        k_grid = sorted(set(k_grid))
    else:
        k_grid = sorted({int(k) for k in declared_k_grid if 1 <= int(k) <= int(data.num_mutations)})
    if curvature is None:
        curvature = observed_curvature_at_pilot_torch(
            context.model, pilot_tensor, eps=context.eps,
        )
    else:
        curvature = as_runtime_tensor(curvature, runtime)
    label_sets = hessian_weighted_ward_label_sets_torch(
        pilot_tensor, curvature, K_grid=k_grid,
    )
    return tuple(generate_likelihood_partition_starts(
        data, eps=float(fit_options.eps), label_sets=label_sets,
        tol=float(fit_options.solver.tolerance),
    ))
