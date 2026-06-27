from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from collections.abc import Sequence

import numpy as np
import torch

from ...io.data import TumorData
from ..bic import bic_degrees_of_freedom, compute_bic_with_df, effective_bic_mutation_region_count
from .refit import PartitionRefitResult, _canonical_labels, partition_constrained_observed_refit
from .starts import _mutation_region_loss_grid_numpy
from .torch_backend import (
    TorchTumorData,
    as_runtime_tensor,
    mutation_region_loss_grid_torch,
    resolve_runtime,
    to_torch_tumor_data,
)
from .types import TorchRuntime


@dataclass(frozen=True)
class PartitionCandidate:
    labels: np.ndarray
    K: int
    source: str
    theta: np.ndarray
    phi_start: np.ndarray
    fit_loss: float
    bic: float
    active_df: int | None = None
    n_eff: int | None = None
    finite_candidate_found: bool = True
    diagnostics: dict[str, float] = field(default_factory=dict)


def compute_partition_bic(*, fit_loss: float, num_clusters: int, data: TumorData) -> float:
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


def _copy_torch_tumor_data(data: TorchTumorData, *, dtype: torch.dtype, device: torch.device) -> TorchTumorData:
    return TorchTumorData(
        alt=data.alt.to(dtype=dtype, device=device),
        total=data.total.to(dtype=dtype, device=device),
        nonalt=data.nonalt.to(dtype=dtype, device=device),
        phi_upper=data.phi_upper.to(dtype=dtype, device=device),
        ambiguous=data.ambiguous.to(device=device),
        b_minus=data.b_minus.to(dtype=dtype, device=device),
        b_plus=data.b_plus.to(dtype=dtype, device=device),
        b_fixed=data.b_fixed.to(dtype=dtype, device=device),
        count_observed=None if data.count_observed is None else data.count_observed.to(device=device),
    )


def _resolve_partition_runtime(
    *,
    data: TumorData,
    exact_pilot: np.ndarray | torch.Tensor | object | None = None,
    torch_data: TorchTumorData | None = None,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> tuple[TorchRuntime, TorchTumorData]:
    if torch_data is not None:
        runtime_device = torch.device(device) if device is not None else torch_data.alt.device
        runtime_dtype = torch_data.alt.dtype if dtype is None or isinstance(dtype, torch.dtype) else resolve_runtime(
            str(runtime_device),
            dtype=str(dtype),
        ).dtype
        if isinstance(dtype, torch.dtype):
            runtime_dtype = dtype
        runtime_dtype = _partition_work_dtype(runtime_dtype)
        runtime = TorchRuntime(
            device=runtime_device,
            device_name=_torch_device_name(runtime_device),
            dtype=runtime_dtype,
        )
        return runtime, _copy_torch_tumor_data(torch_data, dtype=runtime.dtype, device=runtime.device)

    if torch.is_tensor(exact_pilot):
        runtime_device = torch.device(device) if device is not None else exact_pilot.device
        runtime_dtype = exact_pilot.dtype if dtype is None else dtype
        if not isinstance(runtime_dtype, torch.dtype):
            runtime_dtype = resolve_runtime(str(runtime_device), dtype=str(runtime_dtype)).dtype
        runtime_dtype = _partition_work_dtype(runtime_dtype)
        runtime = TorchRuntime(
            device=runtime_device,
            device_name=_torch_device_name(runtime_device),
            dtype=runtime_dtype,
        )
    else:
        requested_device = "cuda" if device is None and torch.cuda.is_available() else device
        runtime = resolve_runtime(None if requested_device is None else str(requested_device), dtype=None if dtype is None else str(dtype))
        if runtime.dtype == torch.float16:
            runtime = TorchRuntime(device=runtime.device, device_name=runtime.device_name, dtype=torch.float32)
    return runtime, to_torch_tumor_data(data, runtime)


def _as_torch(array: np.ndarray | torch.Tensor | object, *, runtime: TorchRuntime) -> torch.Tensor:
    return as_runtime_tensor(array, runtime)


def _data_arrays(data: TumorData) -> dict[str, np.ndarray]:
    return {
        "alt": np.asarray(data.alt_counts, dtype=np.float64),
        "total": np.asarray(data.total_counts, dtype=np.float64),
        "b_minus": np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64),
        "b_plus": np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64),
        "b_fixed": np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64),
        "ambiguous": np.asarray(data.multiplicity_estimation_mask, dtype=bool),
        "upper": np.asarray(data.phi_upper, dtype=np.float64),
    }


def _mutation_region_loss_vector_numpy(
    beta: float,
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
    beta_values = np.full(np.asarray(alt).shape, float(beta), dtype=np.float64)
    out = np.empty_like(beta_values, dtype=np.float64)
    for idx in np.ndindex(beta_values.shape):
        out[idx] = float(
            _mutation_region_loss_grid_numpy(
                np.asarray([beta_values[idx]], dtype=np.float64),
                alt=float(alt[idx]),
                total=float(total[idx]),
                b_minus=float(b_minus[idx]),
                b_plus=float(b_plus[idx]),
                b_fixed=float(b_fixed[idx]),
                ambiguous=bool(ambiguous[idx]),
                major_prior=float(major_prior),
                eps=float(eps),
            )[0]
        )
    return out


def _single_mutation_region_loss(
    beta: float,
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    major_prior: float,
    eps: float,
) -> float:
    return float(
        _mutation_region_loss_grid_numpy(
            np.asarray([float(beta)], dtype=np.float64),
            alt=float(alt),
            total=float(total),
            b_minus=float(b_minus),
            b_plus=float(b_plus),
            b_fixed=float(b_fixed),
            ambiguous=bool(ambiguous),
            major_prior=float(major_prior),
            eps=float(eps),
        )[0]
    )


def _mutation_region_loss_matrix_torch(
    torch_data: TorchTumorData,
    beta: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> torch.Tensor:
    return mutation_region_loss_grid_torch(
        beta,
        alt=torch_data.alt,
        total=torch_data.total,
        b_minus=torch_data.b_minus,
        b_plus=torch_data.b_plus,
        b_fixed=torch_data.b_fixed,
        ambiguous=torch_data.ambiguous,
        major_prior=float(major_prior),
        eps=float(eps),
    )


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

    f_left = _mutation_region_loss_matrix_torch(torch_data, left, major_prior=major_prior, eps=eps)
    f0 = _mutation_region_loss_matrix_torch(torch_data, x0, major_prior=major_prior, eps=eps)
    f_right = _mutation_region_loss_matrix_torch(torch_data, right, major_prior=major_prior, eps=eps)
    denom = h_left * h_right * (h_left + h_right)
    curvature = 2.0 * (h_left * f_right - (h_left + h_right) * f0 + h_right * f_left) / denom
    floor = torch.full_like(curvature, float(curvature_floor))
    curvature = torch.where(valid & torch.isfinite(curvature), torch.maximum(curvature, floor), floor)

    finite = curvature[torch.isfinite(curvature)]
    if finite.numel() and 0.0 < float(curvature_cap_quantile) < 1.0:
        cap = torch.quantile(finite, float(curvature_cap_quantile))
        if bool(torch.isfinite(cap).item()) and float(cap.item()) > float(curvature_floor):
            curvature = torch.minimum(curvature, cap)
    return torch.maximum(curvature, floor)


def observed_curvature_at_pilot(
    data: TumorData,
    exact_pilot: np.ndarray | object,
    *,
    major_prior: float,
    eps: float,
    step_fraction: float = 1e-3,
    min_step: float = 1e-4,
    curvature_floor: float = 1e-6,
    curvature_cap_quantile: float = 0.995,
) -> np.ndarray:
    phi0 = _as_numpy(exact_pilot)
    arrays = _data_arrays(data)
    upper = arrays["upper"]
    lower = float(eps)
    h = np.full(phi0.shape, float(curvature_floor), dtype=np.float64)

    for mutation_idx in range(phi0.shape[0]):
        for region_idx in range(phi0.shape[1]):
            x0 = float(np.clip(phi0[mutation_idx, region_idx], lower, upper[mutation_idx, region_idx]))
            width = max(float(upper[mutation_idx, region_idx]) - lower, 0.0)
            step = max(float(min_step), float(step_fraction) * max(width, abs(x0), 1.0))
            left = max(lower, x0 - step)
            right = min(float(upper[mutation_idx, region_idx]), x0 + step)
            h_left = x0 - left
            h_right = right - x0
            if h_left <= 1e-12 or h_right <= 1e-12:
                continue
            f_left = _single_mutation_region_loss(
                left,
                alt=arrays["alt"][mutation_idx, region_idx],
                total=arrays["total"][mutation_idx, region_idx],
                b_minus=arrays["b_minus"][mutation_idx, region_idx],
                b_plus=arrays["b_plus"][mutation_idx, region_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, region_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, region_idx],
                major_prior=major_prior,
                eps=eps,
            )
            f0 = _single_mutation_region_loss(
                x0,
                alt=arrays["alt"][mutation_idx, region_idx],
                total=arrays["total"][mutation_idx, region_idx],
                b_minus=arrays["b_minus"][mutation_idx, region_idx],
                b_plus=arrays["b_plus"][mutation_idx, region_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, region_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, region_idx],
                major_prior=major_prior,
                eps=eps,
            )
            f_right = _single_mutation_region_loss(
                right,
                alt=arrays["alt"][mutation_idx, region_idx],
                total=arrays["total"][mutation_idx, region_idx],
                b_minus=arrays["b_minus"][mutation_idx, region_idx],
                b_plus=arrays["b_plus"][mutation_idx, region_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, region_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, region_idx],
                major_prior=major_prior,
                eps=eps,
            )
            curvature = (
                2.0
                * (h_left * f_right - (h_left + h_right) * f0 + h_right * f_left)
                / (h_left * h_right * (h_left + h_right))
            )
            if np.isfinite(curvature):
                h[mutation_idx, region_idx] = max(float(curvature), float(curvature_floor))

    finite = h[np.isfinite(h)]
    if finite.size and 0.0 < float(curvature_cap_quantile) < 1.0:
        cap = float(np.quantile(finite, float(curvature_cap_quantile)))
        if np.isfinite(cap) and cap > float(curvature_floor):
            h = np.minimum(h, cap)
    return np.maximum(h, float(curvature_floor))


def _ward_merge_cost(H_a: np.ndarray, mu_a: np.ndarray, H_b: np.ndarray, mu_b: np.ndarray) -> float:
    denom = H_a + H_b
    weight = np.divide(H_a * H_b, denom, out=np.zeros_like(denom), where=denom > 0.0)
    value = 0.5 * float(np.sum(weight * np.square(mu_a - mu_b)))
    return value if np.isfinite(value) else float("inf")


def hessian_weighted_ward_label_sets(
    exact_pilot: np.ndarray | object,
    curvature: np.ndarray,
    *,
    K_grid: Sequence[int],
) -> dict[int, np.ndarray]:
    phi0 = _as_numpy(exact_pilot)
    h = np.asarray(curvature, dtype=np.float64)
    if phi0.shape != h.shape:
        raise ValueError("exact_pilot and curvature must have the same shape.")
    num_mutations = int(phi0.shape[0])
    requested = {int(k) for k in K_grid if 1 <= int(k) <= num_mutations}
    if not requested:
        return {}

    H: dict[int, np.ndarray] = {idx: h[idx].copy() for idx in range(num_mutations)}
    mu: dict[int, np.ndarray] = {idx: phi0[idx].copy() for idx in range(num_mutations)}
    members: dict[int, np.ndarray] = {idx: np.asarray([idx], dtype=np.int64) for idx in range(num_mutations)}
    active: set[int] = set(range(num_mutations))
    version: dict[int, int] = {idx: 0 for idx in range(num_mutations)}
    heap: list[tuple[float, int, int, int, int]] = []

    for left in range(num_mutations - 1):
        for right in range(left + 1, num_mutations):
            cost = _ward_merge_cost(H[left], mu[left], H[right], mu[right])
            heapq.heappush(heap, (float(cost), left, right, 0, 0))

    def current_labels() -> np.ndarray:
        labels = np.full((num_mutations,), -1, dtype=np.int64)
        for label, cluster_id in enumerate(sorted(active)):
            labels[members[cluster_id]] = int(label)
        return labels

    out: dict[int, np.ndarray] = {}
    if num_mutations in requested:
        out[num_mutations] = current_labels()

    next_cluster_id = num_mutations
    while len(active) > 1 and requested - set(out):
        while heap:
            cost, left, right, left_version, right_version = heapq.heappop(heap)
            if (
                left in active
                and right in active
                and version.get(left, -1) == left_version
                and version.get(right, -1) == right_version
            ):
                break
        else:
            raise RuntimeError("Hessian-weighted Ward heap exhausted before all clusters were merged.")

        new_id = next_cluster_id
        next_cluster_id += 1
        H_new = H[left] + H[right]
        mu_new = np.divide(
            H[left] * mu[left] + H[right] * mu[right],
            H_new,
            out=0.5 * (mu[left] + mu[right]),
            where=H_new > 0.0,
        )
        members_new = np.concatenate([members[left], members[right]])

        active.remove(left)
        active.remove(right)
        active.add(new_id)
        H[new_id] = H_new
        mu[new_id] = mu_new
        members[new_id] = members_new
        version[new_id] = 0

        for other in list(active):
            if other == new_id:
                continue
            cost = _ward_merge_cost(H[new_id], mu[new_id], H[other], mu[other])
            a, b = (new_id, other) if new_id < other else (other, new_id)
            heapq.heappush(heap, (float(cost), a, b, version[a], version[b]))

        current_k = len(active)
        if current_k in requested:
            out[current_k] = current_labels()
    return out


@torch.no_grad()
def hessian_weighted_ward_label_sets_torch(
    exact_pilot: np.ndarray | torch.Tensor | object,
    curvature: np.ndarray | torch.Tensor,
    *,
    K_grid: Sequence[int],
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> dict[int, np.ndarray]:
    if torch.is_tensor(exact_pilot):
        pilot_device = exact_pilot.device
        pilot_dtype = exact_pilot.dtype
    elif torch.is_tensor(curvature):
        pilot_device = curvature.device
        pilot_dtype = curvature.dtype
    else:
        pilot_device = torch.device("cuda" if device is None and torch.cuda.is_available() else "cpu")
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
    max_nodes = max(2 * num_mutations - 1, 1)
    H = torch.zeros((max_nodes, num_regions), dtype=runtime.dtype, device=runtime.device)
    mu = torch.zeros_like(H)
    H[:num_mutations] = h
    mu[:num_mutations] = phi0
    active = torch.zeros((max_nodes,), dtype=torch.bool, device=runtime.device)
    active[:num_mutations] = True
    mutation_cluster = torch.arange(num_mutations, dtype=torch.long, device=runtime.device)

    denom = H[:num_mutations, None, :] + H[None, :num_mutations, :]
    weight = torch.where(denom > 0.0, H[:num_mutations, None, :] * H[None, :num_mutations, :] / denom.clamp_min(torch.finfo(runtime.dtype).tiny), torch.zeros_like(denom))
    diff = mu[:num_mutations, None, :] - mu[None, :num_mutations, :]
    initial_cost = 0.5 * torch.sum(weight * torch.square(diff), dim=2)
    finite_large = torch.finfo(runtime.dtype).max / 16.0
    cost_matrix = torch.full((max_nodes, max_nodes), finite_large, dtype=runtime.dtype, device=runtime.device)
    upper_mask = torch.triu(torch.ones((num_mutations, num_mutations), dtype=torch.bool, device=runtime.device), diagonal=1)
    cost_matrix[:num_mutations, :num_mutations] = torch.where(upper_mask, initial_cost, torch.full_like(initial_cost, finite_large))

    def current_labels() -> np.ndarray:
        return _canonical_labels(mutation_cluster.detach().cpu().numpy().astype(np.int64, copy=False))

    out: dict[int, np.ndarray] = {}
    active_count = num_mutations
    if active_count in requested:
        out[active_count] = current_labels()

    next_cluster_id = num_mutations
    while active_count > 1 and requested - set(out):
        flat_index = int(torch.argmin(cost_matrix).item())
        min_cost = float(cost_matrix.reshape(-1)[flat_index].item())
        if not np.isfinite(min_cost) or min_cost >= finite_large * 0.5:
            raise RuntimeError("Hessian-weighted Ward dense CUDA cost matrix exhausted before all clusters were merged.")
        left = int(flat_index // max_nodes)
        right = int(flat_index % max_nodes)
        new_id = next_cluster_id
        next_cluster_id += 1

        H_new = H[left] + H[right]
        H[new_id] = H_new
        mu[new_id] = torch.where(
            H_new > 0.0,
            (H[left] * mu[left] + H[right] * mu[right]) / H_new.clamp_min(torch.finfo(runtime.dtype).tiny),
            0.5 * (mu[left] + mu[right]),
        )
        mutation_cluster = torch.where(
            (mutation_cluster == left) | (mutation_cluster == right),
            torch.full_like(mutation_cluster, new_id),
            mutation_cluster,
        )

        active[left] = False
        active[right] = False
        active[new_id] = True
        cost_matrix[left, :] = finite_large
        cost_matrix[:, left] = finite_large
        cost_matrix[right, :] = finite_large
        cost_matrix[:, right] = finite_large
        cost_matrix[new_id, :] = finite_large
        cost_matrix[:, new_id] = finite_large

        other = torch.nonzero(active, as_tuple=False).flatten()
        other = other[other != new_id]
        if other.numel():
            denom_vec = H[new_id].unsqueeze(0) + H[other]
            weight_vec = torch.where(
                denom_vec > 0.0,
                H[new_id].unsqueeze(0) * H[other] / denom_vec.clamp_min(torch.finfo(runtime.dtype).tiny),
                torch.zeros_like(denom_vec),
            )
            diff_vec = mu[new_id].unsqueeze(0) - mu[other]
            cost_vec = 0.5 * torch.sum(weight_vec * torch.square(diff_vec), dim=1)
            cost_matrix[other, new_id] = cost_vec

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
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    arrays = _data_arrays(data)
    num_mutations = int(data.num_mutations)
    num_clusters = int(centers.shape[0])
    cost = np.zeros((num_mutations, num_clusters), dtype=np.float64)
    infeasible = np.zeros((num_mutations, num_clusters), dtype=bool)

    for cluster_idx in range(num_clusters):
        for region_idx in range(int(data.num_regions)):
            beta = float(centers[cluster_idx, region_idx])
            cost[:, cluster_idx] += _mutation_region_loss_vector_numpy(
                beta,
                alt=arrays["alt"][:, region_idx],
                total=arrays["total"][:, region_idx],
                b_minus=arrays["b_minus"][:, region_idx],
                b_plus=arrays["b_plus"][:, region_idx],
                b_fixed=arrays["b_fixed"][:, region_idx],
                ambiguous=arrays["ambiguous"][:, region_idx],
                major_prior=major_prior,
                eps=eps,
            )
            infeasible[:, cluster_idx] |= beta > arrays["upper"][:, region_idx] + max(float(eps), 1e-8)

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
    cost = torch.sum(loss, dim=2)
    infeasible = torch.any(beta > torch_data.phi_upper.unsqueeze(1) + max(float(eps), 1e-8), dim=2)
    safe_penalty = min(float(infeasible_penalty), float(torch.finfo(cost.dtype).max) / 16.0)
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


def refine_partition_likelihood(
    data: TumorData,
    labels: np.ndarray,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int = 12,
    refit_max_iter: int = 32,
    hint_phi: np.ndarray | None = None,
) -> tuple[np.ndarray, PartitionRefitResult]:
    labels = _canonical_labels(np.asarray(labels, dtype=np.int64))
    refit: PartitionRefitResult | None = None
    for _ in range(max(int(max_iter), 0)):
        refit = partition_constrained_observed_refit(
            data,
            labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(refit_max_iter), 32),
            hint_phi=hint_phi,
        )
        cost = _loss_to_centers(data, refit.cluster_centers, major_prior=float(major_prior), eps=float(eps))
        labels_next = np.argmin(cost, axis=1).astype(np.int64, copy=False)
        labels_next = _repair_empty_clusters(labels_next, cost)
        labels_next = _canonical_labels(labels_next)
        if np.array_equal(labels_next, labels):
            labels = labels_next
            break
        labels = labels_next
    refit = partition_constrained_observed_refit(
        data,
        labels,
        major_prior=float(major_prior),
        eps=float(eps),
        tol=float(tol),
        max_iter=max(int(refit_max_iter), 32),
        hint_phi=hint_phi,
    )
    return _canonical_labels(labels), refit


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
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=hint_phi,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    labels_np = _canonical_labels(np.asarray(labels, dtype=np.int64))
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
            loglik_source="partition_constrained_observed_mle_cuda_unimodal",
        )

    labels_t = torch.as_tensor(labels_np, dtype=torch.long, device=runtime.device)
    membership = torch.nn.functional.one_hot(labels_t, num_classes=n_clusters).to(dtype=runtime.dtype)
    lower = torch.full((n_clusters, n_regions), float(eps), dtype=runtime.dtype, device=runtime.device)
    upper = torch.empty_like(lower)
    for cluster_idx in range(n_clusters):
        member_mask = labels_t == int(cluster_idx)
        if bool(torch.any(member_mask).item()):
            upper[cluster_idx] = torch.min(torch_data.phi_upper[member_mask], dim=0).values
        else:
            upper[cluster_idx] = lower[cluster_idx]
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
        if bool(torch.all(torch.abs(right - left) <= tol * (1.0 + torch.abs(left) + torch.abs(right))).item()):
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
        hint_centers = torch.empty((n_clusters, n_regions), dtype=runtime.dtype, device=runtime.device)
        for cluster_idx in range(n_clusters):
            member_mask = labels_t == int(cluster_idx)
            if bool(torch.any(member_mask).item()):
                hint_centers[cluster_idx] = torch.median(hint_t[member_mask], dim=0).values
            else:
                hint_centers[cluster_idx] = lower[cluster_idx]
        candidates.append(torch.minimum(torch.maximum(hint_centers, lower), upper))
    candidate_values = torch.stack(candidates, dim=0)
    candidate_losses = torch.stack([objective(candidate) for candidate in candidates], dim=0)
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
    phi = torch.minimum(torch.maximum(phi, torch.full_like(phi, float(eps))), torch_data.phi_upper)
    finite_candidate_found = bool(torch.isfinite(total_loss).item())
    refit_coordinate_count = int(n_clusters * n_regions)
    finite_coordinate_count = int(torch.sum(torch.isfinite(best_loss)).detach().cpu().item())
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
        refit_total_grid_points=int(refit_coordinate_count * (2 * n_iter + len(candidates))),
        refit_max_grid_spacing=float(torch.max(initial_width).detach().cpu().item()) if initial_width.numel() else 0.0,
        refit_total_candidate_basins=refit_coordinate_count,
        refit_total_refined_candidates=refit_coordinate_count,
        refit_min_best_second_loss_gap=float(best_second_loss_gap),
        loglik_source="partition_constrained_observed_mle_cuda_unimodal",
    )


@torch.no_grad()
def refine_partition_likelihood_torch(
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
) -> tuple[np.ndarray, PartitionRefitResult]:
    runtime, torch_data = _resolve_partition_runtime(
        data=data,
        exact_pilot=hint_phi,
        torch_data=torch_data,
        device=device,
        dtype=dtype,
    )
    labels = _canonical_labels(np.asarray(labels, dtype=np.int64))
    refit: PartitionRefitResult | None = None
    for _ in range(max(int(max_iter), 0)):
        refit = partition_constrained_observed_refit_torch(
            data,
            labels,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=float(tol),
            max_iter=max(int(refit_max_iter), 32),
            hint_phi=hint_phi,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        cost_t = _loss_to_centers_torch(
            data,
            refit.cluster_centers,
            major_prior=float(major_prior),
            eps=float(eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        labels_next = torch.argmin(cost_t, dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
        labels_next = _repair_empty_clusters(labels_next, cost_t.detach().cpu().numpy())
        labels_next = _canonical_labels(labels_next)
        if np.array_equal(labels_next, labels):
            labels = labels_next
            break
        labels = labels_next
    refit = partition_constrained_observed_refit_torch(
        data,
        labels,
        major_prior=float(major_prior),
        eps=float(eps),
        tol=float(tol),
        max_iter=max(int(refit_max_iter), 32),
        hint_phi=hint_phi,
        torch_data=torch_data,
        device=runtime.device,
        dtype=runtime.dtype,
    )
    return _canonical_labels(labels), refit


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
) -> list[PartitionCandidate]:
    use_torch_runtime = bool(use_torch)
    runtime: TorchRuntime | None = None
    partition_torch_data: TorchTumorData | None = None
    if use_torch_runtime:
        try:
            runtime, partition_torch_data = _resolve_partition_runtime(
                data=data,
                exact_pilot=exact_pilot,
                torch_data=torch_data,
                device=device,
                dtype=dtype,
            )
        except RuntimeError:
            use_torch_runtime = False
    phi0 = _as_torch(exact_pilot, runtime=runtime).detach().cpu().numpy() if use_torch_runtime and runtime is not None else _as_numpy(exact_pilot)
    requested_grid = {int(k) for k in K_grid if 1 <= int(k) <= int(data.num_mutations)}
    if label_sets is None:
        if curvature is None:
            if use_torch_runtime and runtime is not None and partition_torch_data is not None:
                curvature = observed_curvature_at_pilot_torch(
                    data,
                    exact_pilot,
                    major_prior=float(major_prior),
                    eps=float(eps),
                    torch_data=partition_torch_data,
                    device=runtime.device,
                    dtype=runtime.dtype,
                )
            else:
                curvature = observed_curvature_at_pilot(
                    data,
                    phi0,
                    major_prior=float(major_prior),
                    eps=float(eps),
                )
        if use_torch_runtime and runtime is not None:
            label_sets = hessian_weighted_ward_label_sets_torch(
                exact_pilot,
                curvature,
                K_grid=K_grid,
                device=runtime.device,
                dtype=runtime.dtype,
            )
        else:
            label_sets = hessian_weighted_ward_label_sets(phi0, _as_numpy(curvature), K_grid=K_grid)
    else:
        label_sets = {
            int(k): _canonical_labels(np.asarray(labels, dtype=np.int64))
            for k, labels in label_sets.items()
            if int(k) in requested_grid
        }
    candidates: list[PartitionCandidate] = []
    seen: set[bytes] = set()
    n_eff = effective_bic_mutation_region_count(data)

    for requested_k in sorted(label_sets):
        labels0 = _canonical_labels(label_sets[int(requested_k)])
        for source, labels in (
            (f"hessian_ward_K{int(requested_k)}", labels0),
            (f"hessian_ward_cem_K{int(requested_k)}", labels0),
        ):
            if source.startswith("hessian_ward_cem"):
                if use_torch_runtime and runtime is not None and partition_torch_data is not None:
                    labels_used, refit = refine_partition_likelihood_torch(
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
                    )
                else:
                    labels_used, refit = refine_partition_likelihood(
                        data,
                        labels,
                        major_prior=float(major_prior),
                        eps=float(eps),
                        tol=float(tol),
                        max_iter=int(cem_max_iter),
                        refit_max_iter=int(refit_max_iter),
                        hint_phi=phi0,
                    )
            else:
                if use_torch_runtime and runtime is not None and partition_torch_data is not None:
                    refit = partition_constrained_observed_refit_torch(
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
                    refit = partition_constrained_observed_refit(
                        data,
                        labels,
                        major_prior=float(major_prior),
                        eps=float(eps),
                        tol=float(tol),
                        max_iter=max(int(refit_max_iter), 32),
                        hint_phi=phi0,
                    )
                labels_used = labels

            key = _label_key(labels_used)
            if key in seen:
                continue
            seen.add(key)
            candidate_k = int(refit.n_clusters)
            bic = compute_partition_bic(fit_loss=float(refit.fit_loss), num_clusters=candidate_k, data=data)
            candidates.append(
                PartitionCandidate(
                    labels=_canonical_labels(labels_used),
                    K=candidate_k,
                    source=source,
                    theta=refit.cluster_centers,
                    phi_start=refit.phi,
                    fit_loss=float(refit.fit_loss),
                    bic=float(bic),
                    active_df=int(refit.active_degrees_of_freedom),
                    n_eff=int(n_eff),
                    finite_candidate_found=bool(refit.finite_candidate_found),
                    diagnostics={
                        "requested_K": float(requested_k),
                        "refit_boundary_count": float(refit.boundary_count),
                        "refit_coordinate_count": float(refit.refit_coordinate_count),
                        "refit_finite_coordinate_count": float(refit.refit_finite_coordinate_count),
                        "refit_total_grid_points": float(refit.refit_total_grid_points),
                        "refit_max_grid_spacing": float(refit.refit_max_grid_spacing),
                        "refit_total_candidate_basins": float(refit.refit_total_candidate_basins),
                        "refit_total_refined_candidates": float(refit.refit_total_refined_candidates),
                        "refit_min_best_second_loss_gap": float(refit.refit_min_best_second_loss_gap),
                        "partition_generation_cuda": float(
                            use_torch_runtime
                            and runtime is not None
                            and runtime.device.type == "cuda"
                        ),
                    },
                )
            )

    by_k: dict[int, list[PartitionCandidate]] = {}
    for candidate in candidates:
        by_k.setdefault(int(candidate.K), []).append(candidate)
    kept: list[PartitionCandidate] = []
    for candidate_k, values in by_k.items():
        values = sorted(values, key=lambda item: (float(item.bic), float(item.fit_loss), str(item.source)))
        kept.extend(values[: max(int(max_candidates_per_K), 1)])
    return sorted(kept, key=lambda item: (float(item.bic), int(item.K), str(item.source)))
