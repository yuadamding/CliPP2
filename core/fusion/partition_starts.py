from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence

import numpy as np
import torch

from ...io.data import TumorData
from ..objective import ObservedModel, compile_observed_model, observed_terms_numpy
from ..bic import (
    PARTITION_DIRICHLET_SCORE_WEIGHT,
    bic_degrees_of_freedom,
    cluster_sizes_from_labels,
    compute_bic_with_df,
    compute_partition_dirichlet_score,
    effective_bic_mutation_region_count,
)
from .refit import (
    PartitionRefitResult,
    _canonical_labels,
    partition_constrained_observed_refit,
)
from .torch_backend import (
    TorchTumorData,
    as_runtime_tensor,
    copy_torch_tumor_data,
    mutation_region_loss_grid_torch,
    mutation_region_terms_torch,
    resolve_runtime,
    to_torch_tumor_data,
)
from .types import TorchRuntime


# Bound each temporary used to initialize the dense Ward cost matrix.  The
# unchunked broadcast has shape (M, M, S), so its memory grows by several
# copies of M^2*S even though the persistent Ward state is only O(M^2).
_WARD_INITIAL_PAIRWISE_WORK_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class PartitionCandidate:
    labels: np.ndarray
    K: int
    source: str
    phi_start: np.ndarray
    fit_loss: float
    bic: float
    finite_candidate_found: bool = True
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PartitionRefinementResult:
    labels: np.ndarray
    refit: PartitionRefitResult
    iterations: int
    accepted_updates: int
    initial_k: int
    final_k: int
    component_death_count: int
    score_before: float
    score_after: float


def compute_partition_bic(
    *, fit_loss: float, num_clusters: int, data: TumorData
) -> float:
    # fit_loss is the negative log-likelihood (loglik = -fit_loss); delegate to the
    # single BIC definition in core.bic so the formula/observed-mutation_region count never drift.
    return compute_bic_with_df(
        -float(fit_loss),
        bic_degrees_of_freedom(num_clusters, data),
        effective_bic_mutation_region_count(data),
    )


def _as_numpy(array: np.ndarray | object) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def _torch_device_name(device: torch.device) -> str:
    return device.type if device.index is None else f"{device.type}:{device.index}"


def _partition_work_dtype(dtype: torch.dtype) -> torch.dtype:
    # Likelihood/BIC candidate generation uses logs and reductions; keep it above fp16.
    return torch.float32 if dtype == torch.float16 else dtype


def _resolve_partition_runtime(
    *,
    data: TumorData,
    exact_pilot: np.ndarray | torch.Tensor | object | None = None,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> tuple[TorchRuntime, TorchTumorData]:
    if torch_data is not None:
        runtime_device = (
            torch.device(device) if device is not None else torch_data.alt.device
        )
        runtime_dtype = (
            torch_data.alt.dtype
            if dtype is None or isinstance(dtype, torch.dtype)
            else resolve_runtime(
                str(runtime_device),
                dtype=str(dtype),
            ).dtype
        )
        if isinstance(dtype, torch.dtype):
            runtime_dtype = dtype
        runtime_dtype = _partition_work_dtype(runtime_dtype)
        runtime = TorchRuntime(
            device=runtime_device,
            device_name=_torch_device_name(runtime_device),
            dtype=runtime_dtype,
        )
        return runtime, copy_torch_tumor_data(
            torch_data, dtype=runtime.dtype, device=runtime.device
        )

    if torch.is_tensor(exact_pilot):
        runtime_device = (
            torch.device(device) if device is not None else exact_pilot.device
        )
        runtime_dtype = exact_pilot.dtype if dtype is None else dtype
        if not isinstance(runtime_dtype, torch.dtype):
            runtime_dtype = resolve_runtime(
                str(runtime_device), dtype=str(runtime_dtype)
            ).dtype
        runtime_dtype = _partition_work_dtype(runtime_dtype)
        runtime = TorchRuntime(
            device=runtime_device,
            device_name=_torch_device_name(runtime_device),
            dtype=runtime_dtype,
        )
    else:
        requested_device = (
            "cuda" if device is None and torch.cuda.is_available() else device
        )
        runtime = resolve_runtime(
            None if requested_device is None else str(requested_device),
            dtype=None if dtype is None else str(dtype),
        )
        if runtime.dtype == torch.float16:
            runtime = TorchRuntime(
                device=runtime.device,
                device_name=runtime.device_name,
                dtype=torch.float32,
            )
    return runtime, to_torch_tumor_data(data, runtime)


def _as_torch(
    array: np.ndarray | torch.Tensor | object, *, runtime: TorchRuntime
) -> torch.Tensor:
    return as_runtime_tensor(array, runtime)


def _mutation_region_loss_matrix_torch(
    torch_data: TorchTumorData,
    beta: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> torch.Tensor:
    return mutation_region_terms_torch(
        torch_data,
        beta,
        major_prior=float(major_prior),
        eps=float(eps),
    ).loss


@torch.no_grad()
def observed_curvature_at_pilot_torch(
    data: TumorData,
    exact_pilot: np.ndarray | torch.Tensor | object,
    *,
    major_prior: float,
    eps: float,
    step_fraction: float = 1e-3,
    min_step: float = 1e-4,
    curvature_floor: float = 1e-6,
    curvature_cap_quantile: float = 0.995,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=exact_pilot,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    phi0 = _as_torch(exact_pilot, runtime=runtime)
    upper = torch_data.phi_upper
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

    f_left = _mutation_region_loss_matrix_torch(
        torch_data, left, major_prior=major_prior, eps=eps
    )
    f0 = _mutation_region_loss_matrix_torch(
        torch_data, x0, major_prior=major_prior, eps=eps
    )
    f_right = _mutation_region_loss_matrix_torch(
        torch_data, right, major_prior=major_prior, eps=eps
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
    exact_pilot: np.ndarray | torch.Tensor | object,
    curvature: np.ndarray | torch.Tensor,
    *,
    K_grid: Sequence[int],
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    initial_pairwise_work_elements: int = _WARD_INITIAL_PAIRWISE_WORK_ELEMENTS,
) -> dict[int, np.ndarray]:
    if torch.is_tensor(exact_pilot):
        pilot_device = exact_pilot.device
        pilot_dtype = exact_pilot.dtype
    elif torch.is_tensor(curvature):
        pilot_device = curvature.device
        pilot_dtype = curvature.dtype
    else:
        pilot_device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else "cpu"
        )
        pilot_dtype = torch.float64
    runtime_device = torch.device(device) if device is not None else pilot_device
    if dtype is None:
        runtime_dtype = pilot_dtype
    elif isinstance(dtype, torch.dtype):
        runtime_dtype = dtype
    else:
        runtime_dtype = resolve_runtime(str(runtime_device), dtype=str(dtype)).dtype
    runtime_dtype = _partition_work_dtype(runtime_dtype)
    runtime = TorchRuntime(
        device=runtime_device,
        device_name=_torch_device_name(runtime_device),
        dtype=runtime_dtype,
    )
    phi0 = _as_torch(exact_pilot, runtime=runtime)
    h = _as_torch(curvature, runtime=runtime)
    if tuple(phi0.shape) != tuple(h.shape):
        raise ValueError("exact_pilot and curvature must have the same shape.")
    num_mutations = int(phi0.shape[0])
    requested = {int(k) for k in K_grid if 1 <= int(k) <= num_mutations}
    if not requested:
        return {}

    num_regions = int(phi0.shape[1])
    if int(initial_pairwise_work_elements) < 1:
        raise ValueError("initial_pairwise_work_elements must be positive.")
    max_nodes = max(2 * num_mutations - 1, 1)
    H = torch.zeros(
        (max_nodes, num_regions), dtype=runtime.dtype, device=runtime.device
    )
    mu = torch.zeros_like(H)
    H[:num_mutations] = h
    mu[:num_mutations] = phi0
    mutation_cluster = torch.arange(
        num_mutations, dtype=torch.long, device=runtime.device
    )

    finite_large = torch.finfo(runtime.dtype).max / 16.0
    cost_matrix = torch.full(
        (max_nodes, max_nodes), finite_large, dtype=runtime.dtype, device=runtime.device
    )
    # Compute the exact singleton Ward costs in row blocks.  This retains the
    # same dense cost matrix and merge order while avoiding simultaneous
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
    all_columns = torch.arange(num_mutations, dtype=torch.long, device=runtime.device)
    H_initial = H[:num_mutations]
    mu_initial = mu[:num_mutations]
    tiny = torch.finfo(runtime.dtype).tiny
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
            row_start, row_stop, dtype=torch.long, device=runtime.device
        )
        upper_mask = all_columns.unsqueeze(0) > row_ids.unsqueeze(1)
        cost_matrix[row_start:row_stop, :num_mutations] = torch.where(
            upper_mask,
            initial_cost,
            finite_large,
        )

    # Keep one exact minimum per matrix row in a lazy heap where it benchmarks
    # faster: all CUDA inputs and the single-region CPU path used by CliPPSim.
    # Updating only rows whose current partner disappeared, plus rows improved
    # by the new cluster, preserves the same row-major argmin tie order. The
    # dense reduction remains faster for small multi-region CPU tensors.
    use_row_heap = bool(runtime.device.type == "cuda" or num_regions == 1)
    row_heap: list[tuple[float, int, int, int]] = []
    row_best_cost: np.ndarray | None = None
    row_best_column: np.ndarray | None = None
    row_version: np.ndarray | None = None
    if use_row_heap:
        initial_row_cost, initial_row_column = torch.min(cost_matrix, dim=1)
        row_best_cost = (
            initial_row_cost.detach().cpu().numpy().astype(np.float64, copy=True)
        )
        row_best_column = (
            initial_row_column.detach().cpu().numpy().astype(np.int64, copy=True)
        )
        row_version = np.zeros((max_nodes,), dtype=np.int64)
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
        if use_row_heap:
            assert row_best_column is not None and row_version is not None
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
        else:
            flat_index = int(torch.argmin(cost_matrix).item())
            min_cost = float(cost_matrix.reshape(-1)[flat_index].item())
            left = int(flat_index // max_nodes)
            right = int(flat_index % max_nodes)
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
            / H_new.clamp_min(torch.finfo(runtime.dtype).tiny),
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
        cost_matrix[left, :] = finite_large
        cost_matrix[:, left] = finite_large
        cost_matrix[right, :] = finite_large
        cost_matrix[:, right] = finite_large
        cost_matrix[new_id, :] = finite_large
        cost_matrix[:, new_id] = finite_large

        other_ids = np.flatnonzero(active_cpu[:new_id])
        other = torch.as_tensor(other_ids, dtype=torch.long, device=runtime.device)
        if other.numel():
            denom_vec = H[new_id].unsqueeze(0) + H[other]
            weight_vec = torch.where(
                denom_vec > 0.0,
                H[new_id].unsqueeze(0)
                * H[other]
                / denom_vec.clamp_min(torch.finfo(runtime.dtype).tiny),
                torch.zeros_like(denom_vec),
            )
            diff_vec = mu[new_id].unsqueeze(0) - mu[other]
            cost_vec = 0.5 * torch.sum(weight_vec * torch.square(diff_vec), dim=1)
            cost_matrix[other, new_id] = cost_vec
            if use_row_heap:
                assert (
                    row_best_cost is not None
                    and row_best_column is not None
                    and row_version is not None
                )
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
                    invalid_tensor = torch.as_tensor(
                        invalid_rows, dtype=torch.long, device=runtime.device
                    )
                    refreshed_cost, refreshed_column = torch.min(
                        cost_matrix[invalid_tensor], dim=1
                    )
                    row_best_cost[invalid_rows] = refreshed_cost.detach().cpu().numpy()
                    row_best_column[invalid_rows] = (
                        refreshed_column.detach().cpu().numpy()
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
        if active_count in requested:
            out[active_count] = current_labels()
    return out


def _loss_to_centers(
    data: TumorData,
    centers: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    infeasible_penalty: float = 1e100,
    _model: ObservedModel | None = None,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    model = (
        compile_observed_model(data, major_prior=major_prior, eps=eps)
        if _model is None
        else _model
    )
    num_mutations = int(data.num_mutations)
    num_clusters = int(centers.shape[0])
    cost = np.zeros((num_mutations, num_clusters), dtype=np.float64)
    infeasible = np.zeros((num_mutations, num_clusters), dtype=bool)

    for cluster_idx in range(num_clusters):
        phi_for_center = np.broadcast_to(centers[cluster_idx], model.shape)
        terms = observed_terms_numpy(model, phi_for_center, eps=float(eps))
        cost[:, cluster_idx] = np.sum(terms.loss, axis=1)
        infeasible[:, cluster_idx] = np.any(
            phi_for_center > model.upper + max(float(eps), 1e-8), axis=1
        )

    cost[infeasible] = float(infeasible_penalty)
    return cost


@torch.no_grad()
def _loss_to_centers_torch(
    data: TumorData,
    centers: np.ndarray | torch.Tensor,
    *,
    major_prior: float,
    eps: float,
    infeasible_penalty: float = 1e100,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> torch.Tensor:
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=centers,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    centers_t = _as_torch(centers, runtime=runtime)
    if torch_data.path_likelihood is not None:
        cost_columns: list[torch.Tensor] = []
        for cluster_idx in range(int(centers_t.shape[0])):
            phi_for_center = (
                centers_t[cluster_idx].unsqueeze(0).expand(int(data.num_mutations), -1)
            )
            terms = mutation_region_terms_torch(
                torch_data,
                phi_for_center,
                major_prior=float(major_prior),
                eps=float(eps),
            )
            cost_column = torch.sum(terms.loss, dim=1)
            infeasible = torch.any(
                phi_for_center > torch_data.phi_upper + max(float(eps), 1e-8),
                dim=1,
            )
            safe_penalty = min(
                float(infeasible_penalty),
                float(torch.finfo(cost_column.dtype).max) / 16.0,
            )
            cost_columns.append(
                torch.where(
                    infeasible,
                    torch.full_like(cost_column, safe_penalty),
                    cost_column,
                )
            )
        return torch.stack(cost_columns, dim=1)
    beta = centers_t.unsqueeze(0)
    loss = mutation_region_loss_grid_torch(
        beta,
        alt=torch_data.alt.unsqueeze(1),
        total=torch_data.total.unsqueeze(1),
        b_minus=torch_data.b_minus.unsqueeze(1),
        b_plus=torch_data.b_plus.unsqueeze(1),
        b_fixed=torch_data.b_fixed.unsqueeze(1),
        ambiguous=torch_data.ambiguous.unsqueeze(1),
        major_prior=float(major_prior),
        eps=float(eps),
    )
    if torch_data.count_observed is not None:
        loss = torch.where(
            torch_data.count_observed.unsqueeze(1),
            loss,
            torch.zeros_like(loss),
        )
    cost = torch.sum(loss, dim=2)
    infeasible = torch.any(
        beta > torch_data.phi_upper.unsqueeze(1) + max(float(eps), 1e-8), dim=2
    )
    safe_penalty = min(
        float(infeasible_penalty), float(torch.finfo(cost.dtype).max) / 16.0
    )
    return torch.where(
        infeasible,
        torch.full_like(cost, float(safe_penalty)),
        cost,
    )


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
    alpha: float,
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
    alpha = float(alpha)
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("classification_weight_alpha must be positive and finite.")
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
    *,
    alpha: float,
    code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
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
        alpha=float(alpha),
    )
    weight = _validated_classification_code_weight(code_weight)
    return count_cost - weight * log_weights


def _classification_refit_score(
    data: TumorData,
    labels: np.ndarray,
    refit: PartitionRefitResult,
    *,
    alpha: float,
    code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
) -> float:
    return compute_partition_dirichlet_score(
        float(refit.loglik),
        cluster_sizes_from_labels(labels),
        data,
        alpha=float(alpha),
        code_weight=_validated_classification_code_weight(code_weight),
    )


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


def _validate_classification_weight_alpha(alpha: float | None) -> None:
    if alpha is not None and (not np.isfinite(float(alpha)) or float(alpha) <= 0.0):
        raise ValueError("classification_weight_alpha must be positive and finite.")


def _validated_classification_code_weight(code_weight: float) -> float:
    weight = float(code_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("classification_code_weight must be nonnegative and finite.")
    return weight


def refine_partition_likelihood_with_trace(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int = 12,
    refit_max_iter: int = 32,
    hint_phi: np.ndarray | None = None,
    classification_weight_alpha: float | None = None,
    classification_code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
    allow_component_death: bool = False,
    _refit_labels: Callable[[np.ndarray], PartitionRefitResult] | None = None,
    _model: ObservedModel | None = None,
) -> PartitionRefinementResult:
    labels = _validated_refinement_labels(data, labels)
    initial_k = int(np.unique(labels).size)
    _validate_classification_weight_alpha(classification_weight_alpha)
    classification_code_weight = _validated_classification_code_weight(
        classification_code_weight
    )
    model = (
        compile_observed_model(data, major_prior=major_prior, eps=eps)
        if _model is None
        else _model
    )

    def refit_labels(current_labels: np.ndarray) -> PartitionRefitResult:
        if _refit_labels is not None:
            return _refit_labels(current_labels)
        return partition_constrained_observed_refit(
            data,
            current_labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(refit_max_iter), 32),
            _model=model,
        )

    refit = refit_labels(labels)
    refit_key = _label_key(labels)
    best_labels: np.ndarray | None = None
    best_refit: PartitionRefitResult | None = None
    best_score = float("inf")
    if classification_weight_alpha is not None:
        best_score = _classification_refit_score(
            data,
            labels,
            refit,
            alpha=float(classification_weight_alpha),
            code_weight=classification_code_weight,
        )
        best_labels = labels.copy()
        best_refit = refit
        score_before = score_after = float(best_score)
    else:
        score_before = score_after = float(refit.fit_loss)
    iterations = 0
    accepted_updates = 0
    for iteration in range(max(int(max_iter), 0)):
        iterations = int(iteration + 1)
        labels_key = _label_key(labels)
        if refit_key != labels_key:
            refit = refit_labels(labels)
            refit_key = labels_key
        count_cost = _loss_to_centers(
            data,
            refit.cluster_centers,
            major_prior=float(major_prior),
            eps=float(eps),
            _model=model,
        )
        assignment_cost = (
            count_cost
            if classification_weight_alpha is None
            else _classification_assignment_cost(
                count_cost,
                labels,
                alpha=float(classification_weight_alpha),
                code_weight=classification_code_weight,
            )
        )
        labels_next = np.argmin(assignment_cost, axis=1).astype(np.int64, copy=False)
        if not bool(allow_component_death):
            labels_next = _repair_empty_clusters(labels_next, assignment_cost)
        labels_next = _canonical_labels(labels_next)
        if _label_key(labels_next) == labels_key:
            labels = labels_next
            break
        if classification_weight_alpha is not None:
            proposed_refit = refit_labels(labels_next)
            proposed_score = _classification_refit_score(
                data,
                labels_next,
                proposed_refit,
                alpha=float(classification_weight_alpha),
                code_weight=classification_code_weight,
            )
            # A simultaneous reassignment is only a proposal. Accept it only
            # after an exact fixed-label refit proves that the declared score
            # decreased; otherwise retain the current best state and stop.
            if not _classification_score_strictly_improves(
                proposed_score,
                best_score,
            ):
                break
            best_score = float(proposed_score)
            score_after = float(best_score)
            best_labels = labels_next.copy()
            best_refit = proposed_refit
            refit = proposed_refit
            refit_key = _label_key(labels_next)
        labels = labels_next
        accepted_updates += 1
    labels_key = _label_key(labels)
    if refit_key != labels_key:
        refit = refit_labels(labels)
    if (
        classification_weight_alpha is not None
        and best_labels is not None
        and best_refit is not None
    ):
        final_labels = _canonical_labels(best_labels)
        final_refit = best_refit
    else:
        final_labels = _canonical_labels(labels)
        final_refit = refit
        score_after = float(final_refit.fit_loss)
    final_k = int(np.unique(final_labels).size)
    return PartitionRefinementResult(
        labels=final_labels,
        refit=final_refit,
        iterations=int(iterations),
        accepted_updates=int(accepted_updates),
        initial_k=int(initial_k),
        final_k=int(final_k),
        component_death_count=max(int(initial_k - final_k), 0),
        score_before=float(score_before),
        score_after=float(score_after),
    )


@torch.no_grad()
def partition_constrained_observed_refit_torch(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint_phi: np.ndarray | torch.Tensor | None = None,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> PartitionRefitResult:
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition refit tolerance must be a positive finite value.")
    if getattr(data, "path_likelihood", None) is not None:
        return partition_constrained_observed_refit(
            data,
            labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(max_iter), 32),
        )
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=hint_phi,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    labels_np = _validated_refinement_labels(data, labels)
    n_clusters = int(labels_np.max()) + 1 if labels_np.size else 0
    n_regions = int(data.num_regions)
    if n_clusters <= 0:
        empty_centers = np.zeros((0, n_regions), dtype=np.float64)
        empty_phi = np.zeros((int(data.num_mutations), n_regions), dtype=np.float64)
        return PartitionRefitResult(
            phi=empty_phi,
            cluster_centers=empty_centers,
            loglik=0.0,
            fit_loss=0.0,
            n_clusters=0,
            boundary_count=0,
            active_degrees_of_freedom=0,
            finite_candidate_found=True,
            refit_coordinate_count=0,
            refit_finite_coordinate_count=0,
            refit_total_grid_points=0,
            refit_max_grid_spacing=0.0,
            refit_total_candidate_basins=0,
            refit_total_refined_candidates=0,
            refit_min_best_second_loss_gap=float("inf"),
            labels=labels_np.astype(np.int64, copy=True),
            loglik_source="partition_constrained_observed_mle_cuda_unimodal",
        )

    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=runtime.device)
    membership = torch.nn.functional.one_hot(labels_t, num_classes=n_clusters).to(
        dtype=runtime.dtype
    )
    lower = torch.full(
        (n_clusters, n_regions), float(eps), dtype=runtime.dtype, device=runtime.device
    )
    upper = torch.empty_like(lower)
    # Canonical labels are contiguous, so every cluster index is present.
    for cluster_idx in range(n_clusters):
        member_mask = labels_t == int(cluster_idx)
        upper[cluster_idx] = torch.min(torch_data.phi_upper[member_mask], dim=0).values
    upper = torch.where(torch.isfinite(upper) & (upper >= lower), upper, lower)
    initial_width = torch.clamp(upper - lower, min=0.0)

    def objective(beta_ks: torch.Tensor) -> torch.Tensor:
        beta = beta_ks.unsqueeze(0)
        loss = mutation_region_loss_grid_torch(
            beta,
            alt=torch_data.alt.unsqueeze(1),
            total=torch_data.total.unsqueeze(1),
            b_minus=torch_data.b_minus.unsqueeze(1),
            b_plus=torch_data.b_plus.unsqueeze(1),
            b_fixed=torch_data.b_fixed.unsqueeze(1),
            ambiguous=torch_data.ambiguous.unsqueeze(1),
            major_prior=float(major_prior),
            eps=float(eps),
        )
        # Mask unobserved mutation_regions out of the likelihood, matching the fit objective
        # (torch_backend.mutation_region_terms_torch) and the numpy refit; torch.where avoids
        # inf*0 = nan when an infeasible beta makes the loss non-finite.
        if torch_data.count_observed is not None:
            loss = torch.where(
                torch_data.count_observed.unsqueeze(1),
                loss,
                torch.zeros_like(loss),
            )
        return torch.sum(loss * membership.unsqueeze(2), dim=0)

    left = lower.clone()
    right = upper.clone()
    ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    n_iter = max(int(max_iter), 32)
    for _ in range(n_iter):
        if bool(
            torch.all(
                torch.abs(right - left)
                <= tol * (1.0 + torch.abs(left) + torch.abs(right))
            ).item()
        ):
            break
        x1 = right - float(ratio) * (right - left)
        x2 = left + float(ratio) * (right - left)
        f1 = objective(x1)
        f2 = objective(x2)
        keep_left_interval = f1 <= f2
        right = torch.where(keep_left_interval, x2, right)
        left = torch.where(keep_left_interval, left, x1)

    midpoint = 0.5 * (left + right)
    candidates = [midpoint, left, right, lower, upper]
    if hint_phi is not None:
        hint_t = _as_torch(hint_phi, runtime=runtime)
        hint_centers = torch.empty(
            (n_clusters, n_regions), dtype=runtime.dtype, device=runtime.device
        )
        for cluster_idx in range(n_clusters):
            member_mask = labels_t == int(cluster_idx)
            hint_centers[cluster_idx] = torch.median(hint_t[member_mask], dim=0).values
        candidates.append(torch.minimum(torch.maximum(hint_centers, lower), upper))
    candidate_values = torch.stack(candidates, dim=0)
    candidate_losses = torch.stack(
        [objective(candidate) for candidate in candidates], dim=0
    )
    best_idx = torch.argmin(candidate_losses, dim=0, keepdim=True)
    centers = torch.gather(candidate_values, 0, best_idx).squeeze(0)
    best_loss = torch.gather(candidate_losses, 0, best_idx).squeeze(0)
    total_loss = torch.sum(best_loss)

    sorted_losses = torch.sort(candidate_losses, dim=0).values
    if sorted_losses.shape[0] >= 2:
        second_gap = torch.min(sorted_losses[1] - sorted_losses[0])
        best_second_loss_gap = float(second_gap.detach().cpu().item())
    else:
        best_second_loss_gap = float("inf")
    boundary_tol = max(float(tol) * 10.0, 1e-8)
    at_boundary = (centers <= lower + boundary_tol) | (centers >= upper - boundary_tol)
    boundary_count = int(torch.sum(at_boundary).detach().cpu().item())
    active_df = int(centers.numel() - boundary_count)
    phi = centers[labels_t]
    phi = torch.minimum(
        torch.maximum(phi, torch.full_like(phi, float(eps))), torch_data.phi_upper
    )
    finite_candidate_found = bool(torch.isfinite(total_loss).item())
    refit_coordinate_count = int(n_clusters * n_regions)
    finite_coordinate_count = int(
        torch.sum(torch.isfinite(best_loss)).detach().cpu().item()
    )
    return PartitionRefitResult(
        phi=phi.detach().cpu().numpy().astype(np.float64, copy=False),
        cluster_centers=centers.detach().cpu().numpy().astype(np.float64, copy=False),
        loglik=float(-total_loss.detach().cpu().item()),
        fit_loss=float(total_loss.detach().cpu().item()),
        n_clusters=int(n_clusters),
        boundary_count=int(boundary_count),
        active_degrees_of_freedom=int(active_df),
        finite_candidate_found=finite_candidate_found,
        refit_coordinate_count=refit_coordinate_count,
        refit_finite_coordinate_count=finite_coordinate_count,
        refit_total_grid_points=int(
            refit_coordinate_count * (2 * n_iter + len(candidates))
        ),
        refit_max_grid_spacing=float(torch.max(initial_width).detach().cpu().item())
        if initial_width.numel()
        else 0.0,
        refit_total_candidate_basins=refit_coordinate_count,
        refit_total_refined_candidates=refit_coordinate_count,
        refit_min_best_second_loss_gap=float(best_second_loss_gap),
        labels=labels_np.astype(np.int64, copy=True),
        loglik_source="partition_constrained_observed_mle_cuda_unimodal",
    )


@torch.no_grad()
def refine_partition_likelihood_torch_with_trace(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int = 12,
    refit_max_iter: int = 32,
    hint_phi: np.ndarray | torch.Tensor | None = None,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    classification_weight_alpha: float | None = None,
    classification_code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
    allow_component_death: bool = False,
    _refit_labels: Callable[[np.ndarray], PartitionRefitResult] | None = None,
) -> PartitionRefinementResult:
    if getattr(data, "path_likelihood", None) is not None:
        return refine_partition_likelihood_with_trace(
            data,
            labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=int(max_iter),
            refit_max_iter=int(refit_max_iter),
            hint_phi=None if hint_phi is None else _as_numpy(hint_phi),
            classification_weight_alpha=classification_weight_alpha,
            classification_code_weight=classification_code_weight,
            allow_component_death=bool(allow_component_death),
            _refit_labels=_refit_labels,
        )
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=hint_phi,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    labels = _validated_refinement_labels(data, labels)
    initial_k = int(np.unique(labels).size)
    _validate_classification_weight_alpha(classification_weight_alpha)
    classification_code_weight = _validated_classification_code_weight(
        classification_code_weight
    )

    def refit_labels(current_labels: np.ndarray) -> PartitionRefitResult:
        if _refit_labels is not None:
            return _refit_labels(current_labels)
        return partition_constrained_observed_refit_torch(
            data,
            current_labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(refit_max_iter), 32),
            hint_phi=hint_phi,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )

    refit = refit_labels(labels)
    refit_key = _label_key(labels)
    best_labels: np.ndarray | None = None
    best_refit: PartitionRefitResult | None = None
    best_score = float("inf")
    if classification_weight_alpha is not None:
        best_score = _classification_refit_score(
            data,
            labels,
            refit,
            alpha=float(classification_weight_alpha),
            code_weight=classification_code_weight,
        )
        best_labels = labels.copy()
        best_refit = refit
        score_before = score_after = float(best_score)
    else:
        score_before = score_after = float(refit.fit_loss)
    iterations = 0
    accepted_updates = 0
    for iteration in range(max(int(max_iter), 0)):
        iterations = int(iteration + 1)
        labels_key = _label_key(labels)
        if refit_key != labels_key:
            refit = refit_labels(labels)
            refit_key = labels_key
        cost_t = _loss_to_centers_torch(
            data,
            refit.cluster_centers,
            major_prior=float(major_prior),
            eps=float(eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        if classification_weight_alpha is None:
            assignment_cost_t = cost_t
        else:
            log_weights = _classification_leave_one_out_log_cluster_weights(
                labels,
                num_clusters=int(cost_t.shape[1]),
                alpha=float(classification_weight_alpha),
            )
            assignment_cost_t = cost_t - classification_code_weight * torch.as_tensor(
                log_weights,
                dtype=cost_t.dtype,
                device=cost_t.device,
            )
        labels_next = (
            torch.argmin(assignment_cost_t, dim=1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64, copy=False)
        )
        if not bool(allow_component_death):
            labels_next = _repair_empty_clusters(
                labels_next,
                assignment_cost_t.detach().cpu().numpy(),
            )
        labels_next = _canonical_labels(labels_next)
        if _label_key(labels_next) == labels_key:
            labels = labels_next
            break
        if classification_weight_alpha is not None:
            proposed_refit = refit_labels(labels_next)
            proposed_score = _classification_refit_score(
                data,
                labels_next,
                proposed_refit,
                alpha=float(classification_weight_alpha),
                code_weight=classification_code_weight,
            )
            if not _classification_score_strictly_improves(
                proposed_score,
                best_score,
            ):
                break
            best_score = float(proposed_score)
            score_after = float(best_score)
            best_labels = labels_next.copy()
            best_refit = proposed_refit
            refit = proposed_refit
            refit_key = _label_key(labels_next)
        labels = labels_next
        accepted_updates += 1
    labels_key = _label_key(labels)
    if refit_key != labels_key:
        refit = refit_labels(labels)
    if (
        classification_weight_alpha is not None
        and best_labels is not None
        and best_refit is not None
    ):
        final_labels = _canonical_labels(best_labels)
        final_refit = best_refit
    else:
        final_labels = _canonical_labels(labels)
        final_refit = refit
        score_after = float(final_refit.fit_loss)
    final_k = int(np.unique(final_labels).size)
    return PartitionRefinementResult(
        labels=final_labels,
        refit=final_refit,
        iterations=int(iterations),
        accepted_updates=int(accepted_updates),
        initial_k=int(initial_k),
        final_k=int(final_k),
        component_death_count=max(int(initial_k - final_k), 0),
        score_before=float(score_before),
        score_after=float(score_after),
    )


def _label_key(labels: np.ndarray) -> bytes:
    labels = _canonical_labels(labels)
    return labels.astype(np.int32, copy=False).tobytes()


def generate_likelihood_partition_starts(
    data: TumorData,
    *,
    exact_pilot: np.ndarray | object,
    major_prior: float,
    eps: float,
    K_grid: Sequence[int],
    max_candidates_per_K: int = 5,
    cem_max_iter: int = 12,
    refit_max_iter: int = 32,
    tol: float = 1e-3,
    curvature: np.ndarray | torch.Tensor | None = None,
    label_sets: dict[int, np.ndarray] | None = None,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
    use_torch: bool = True,
    classification_weight_alpha: float | None = None,
    classification_code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
    allow_component_death: bool = False,
    include_plain_ward: bool = True,
    include_ward_cem: bool = True,
) -> list[PartitionCandidate]:
    _validate_classification_weight_alpha(classification_weight_alpha)
    classification_code_weight = _validated_classification_code_weight(
        classification_code_weight
    )
    use_torch_runtime = bool(use_torch)
    runtime: TorchRuntime | None = None
    partition_torch_data: TorchTumorData | None = None
    if use_torch_runtime:
        runtime, partition_torch_data = _resolve_partition_runtime(
            data=data,
            exact_pilot=exact_pilot,
            torch_data=torch_data,
            device=device,
            dtype=dtype,
        )
    phi0 = (
        _as_torch(exact_pilot, runtime=runtime).detach().cpu().numpy()
        if use_torch_runtime and runtime is not None
        else _as_numpy(exact_pilot)
    )
    requested_grid = {int(k) for k in K_grid if 1 <= int(k) <= int(data.num_mutations)}
    if label_sets is None:
        raise ValueError(
            "generate_likelihood_partition_starts requires precomputed label_sets; "
            "the partition initializer derives them with "
            "hessian_weighted_ward_label_sets_torch before calling."
        )
    label_sets = {
        int(k): _canonical_labels(np.asarray(labels, dtype=np.int64))
        for k, labels in label_sets.items()
        if int(k) in requested_grid
    }
    candidates: list[PartitionCandidate] = []
    seen: set[bytes] = set()
    source_model = compile_observed_model(
        data, major_prior=float(major_prior), eps=float(eps)
    )
    # This cache never escapes one generation call, so labels fully identify a
    # refit under the shared data, tolerance, hint, runtime, and backend.
    refit_cache: dict[bytes, PartitionRefitResult] = {}

    def cached_refit(labels: np.ndarray) -> PartitionRefitResult:
        labels_key = _label_key(labels)
        cached = refit_cache.get(labels_key)
        if cached is not None:
            return cached
        if (
            use_torch_runtime
            and runtime is not None
            and partition_torch_data is not None
        ):
            result = partition_constrained_observed_refit_torch(
                data,
                labels,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=float(tol),
                max_iter=max(int(refit_max_iter), 32),
                hint_phi=exact_pilot,
                torch_data=partition_torch_data,
                device=runtime.device,
                dtype=runtime.dtype,
            )
        else:
            result = partition_constrained_observed_refit(
                data,
                labels,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=float(tol),
                max_iter=max(int(refit_max_iter), 32),
                _model=source_model,
            )
        refit_cache[labels_key] = result
        return result

    for requested_k in sorted(label_sets):
        labels0 = _canonical_labels(label_sets[int(requested_k)])
        source_labels: list[tuple[str, np.ndarray]] = []
        if include_plain_ward:
            source_labels.append((f"hessian_ward_K{int(requested_k)}", labels0))
        if include_ward_cem:
            source_labels.append(
                (f"hessian_ward_cem_K{int(requested_k)}", labels0)
            )
        for source, labels in source_labels:
            trace: PartitionRefinementResult | None = None
            if source.startswith("hessian_ward_cem"):
                if (
                    use_torch_runtime
                    and runtime is not None
                    and partition_torch_data is not None
                ):
                    trace = refine_partition_likelihood_torch_with_trace(
                        data,
                        labels,
                        major_prior=float(major_prior),
                        eps=float(eps),
                        tol=float(tol),
                        max_iter=int(cem_max_iter),
                        refit_max_iter=int(refit_max_iter),
                        hint_phi=exact_pilot,
                        torch_data=partition_torch_data,
                        device=runtime.device,
                        dtype=runtime.dtype,
                        classification_weight_alpha=classification_weight_alpha,
                        classification_code_weight=classification_code_weight,
                        allow_component_death=bool(allow_component_death),
                        _refit_labels=cached_refit,
                    )
                else:
                    trace = refine_partition_likelihood_with_trace(
                        data,
                        labels,
                        major_prior=float(major_prior),
                        eps=float(eps),
                        tol=float(tol),
                        max_iter=int(cem_max_iter),
                        refit_max_iter=int(refit_max_iter),
                        hint_phi=phi0,
                        classification_weight_alpha=classification_weight_alpha,
                        classification_code_weight=classification_code_weight,
                        allow_component_death=bool(allow_component_death),
                        _refit_labels=cached_refit,
                        _model=source_model,
                    )
                labels_used, refit = trace.labels, trace.refit
            else:
                refit = cached_refit(labels)
                labels_used = labels

            key = _label_key(labels_used)
            if key in seen:
                continue
            seen.add(key)
            candidate_k = int(refit.n_clusters)
            classic_bic = compute_partition_bic(
                fit_loss=float(refit.fit_loss),
                num_clusters=candidate_k,
                data=data,
            )
            bic = (
                classic_bic
                if classification_weight_alpha is None
                else compute_partition_dirichlet_score(
                    -float(refit.fit_loss),
                    cluster_sizes_from_labels(labels_used),
                    data,
                    alpha=float(classification_weight_alpha),
                    code_weight=classification_code_weight,
                )
            )
            candidates.append(
                PartitionCandidate(
                    labels=_canonical_labels(labels_used),
                    K=candidate_k,
                    source=source,
                    phi_start=refit.phi,
                    fit_loss=float(refit.fit_loss),
                    bic=float(bic),
                    finite_candidate_found=bool(refit.finite_candidate_found),
                    diagnostics={
                        "requested_K": float(requested_k),
                        "pre_refinement_signature": _label_key(labels0).hex(),
                        "cem_iterations": float(0 if trace is None else trace.iterations),
                        "cem_accepted_updates": float(
                            0 if trace is None else trace.accepted_updates
                        ),
                        "initial_K": float(
                            int(np.unique(labels0).size)
                            if trace is None
                            else trace.initial_k
                        ),
                        "final_K": float(candidate_k),
                        "component_death_count": float(
                            0 if trace is None else trace.component_death_count
                        ),
                        "refinement_score_before": float(
                            refit.fit_loss if trace is None else trace.score_before
                        ),
                        "refinement_score_after": float(
                            refit.fit_loss if trace is None else trace.score_after
                        ),
                        "deterministic_generation": 1.0,
                    },
                )
            )

    by_k: dict[int, list[PartitionCandidate]] = {}
    for candidate in candidates:
        by_k.setdefault(int(candidate.K), []).append(candidate)
    kept: list[PartitionCandidate] = []
    for candidate_k, values in by_k.items():
        values = sorted(
            values,
            key=lambda item: (float(item.bic), float(item.fit_loss), str(item.source)),
        )
        kept.extend(values[: max(int(max_candidates_per_K), 1)])
    return sorted(
        kept, key=lambda item: (float(item.bic), int(item.K), str(item.source))
    )
