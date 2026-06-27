from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from collections.abc import Sequence

import numpy as np

from ...io.data import TumorData
from .refit import PartitionRefitResult, _canonical_labels, partition_constrained_observed_refit
from .starts import _cell_loss_grid_numpy


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


def effective_bic_cell_count(data: TumorData) -> int:
    positive_depth = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    count_observed = getattr(data, "count_observed", None)
    if count_observed is not None:
        positive_depth = positive_depth & np.asarray(count_observed, dtype=bool)
    return max(int(np.sum(positive_depth)), 1)


def compute_partition_bic(*, fit_loss: float, num_clusters: int, data: TumorData) -> float:
    n_eff = effective_bic_cell_count(data)
    df = max(int(num_clusters), 1) * int(data.num_samples)
    return float(2.0 * float(fit_loss) + float(df) * np.log(float(n_eff)))


def _as_numpy(array: np.ndarray | object) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


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


def _cell_loss_vector_numpy(
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
            _cell_loss_grid_numpy(
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


def _single_cell_loss(
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
        _cell_loss_grid_numpy(
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
        for sample_idx in range(phi0.shape[1]):
            x0 = float(np.clip(phi0[mutation_idx, sample_idx], lower, upper[mutation_idx, sample_idx]))
            width = max(float(upper[mutation_idx, sample_idx]) - lower, 0.0)
            step = max(float(min_step), float(step_fraction) * max(width, abs(x0), 1.0))
            left = max(lower, x0 - step)
            right = min(float(upper[mutation_idx, sample_idx]), x0 + step)
            h_left = x0 - left
            h_right = right - x0
            if h_left <= 1e-12 or h_right <= 1e-12:
                continue
            f_left = _single_cell_loss(
                left,
                alt=arrays["alt"][mutation_idx, sample_idx],
                total=arrays["total"][mutation_idx, sample_idx],
                b_minus=arrays["b_minus"][mutation_idx, sample_idx],
                b_plus=arrays["b_plus"][mutation_idx, sample_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, sample_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, sample_idx],
                major_prior=major_prior,
                eps=eps,
            )
            f0 = _single_cell_loss(
                x0,
                alt=arrays["alt"][mutation_idx, sample_idx],
                total=arrays["total"][mutation_idx, sample_idx],
                b_minus=arrays["b_minus"][mutation_idx, sample_idx],
                b_plus=arrays["b_plus"][mutation_idx, sample_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, sample_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, sample_idx],
                major_prior=major_prior,
                eps=eps,
            )
            f_right = _single_cell_loss(
                right,
                alt=arrays["alt"][mutation_idx, sample_idx],
                total=arrays["total"][mutation_idx, sample_idx],
                b_minus=arrays["b_minus"][mutation_idx, sample_idx],
                b_plus=arrays["b_plus"][mutation_idx, sample_idx],
                b_fixed=arrays["b_fixed"][mutation_idx, sample_idx],
                ambiguous=arrays["ambiguous"][mutation_idx, sample_idx],
                major_prior=major_prior,
                eps=eps,
            )
            curvature = (
                2.0
                * (h_left * f_right - (h_left + h_right) * f0 + h_right * f_left)
                / (h_left * h_right * (h_left + h_right))
            )
            if np.isfinite(curvature):
                h[mutation_idx, sample_idx] = max(float(curvature), float(curvature_floor))

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
        for sample_idx in range(int(data.num_samples)):
            beta = float(centers[cluster_idx, sample_idx])
            cost[:, cluster_idx] += _cell_loss_vector_numpy(
                beta,
                alt=arrays["alt"][:, sample_idx],
                total=arrays["total"][:, sample_idx],
                b_minus=arrays["b_minus"][:, sample_idx],
                b_plus=arrays["b_plus"][:, sample_idx],
                b_fixed=arrays["b_fixed"][:, sample_idx],
                ambiguous=arrays["ambiguous"][:, sample_idx],
                major_prior=major_prior,
                eps=eps,
            )
            infeasible[:, cluster_idx] |= beta > arrays["upper"][:, sample_idx] + max(float(eps), 1e-8)

    cost[infeasible] = float(infeasible_penalty)
    return cost


def _repair_empty_clusters(labels: np.ndarray, cost: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).copy()
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
        finite_target = np.isfinite(target_cost) & (target_cost < 1e99)
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
    curvature: np.ndarray | None = None,
    label_sets: dict[int, np.ndarray] | None = None,
) -> list[PartitionCandidate]:
    phi0 = _as_numpy(exact_pilot)
    requested_grid = {int(k) for k in K_grid if 1 <= int(k) <= int(data.num_mutations)}
    if label_sets is None:
        if curvature is None:
            curvature = observed_curvature_at_pilot(
                data,
                phi0,
                major_prior=float(major_prior),
                eps=float(eps),
            )
        label_sets = hessian_weighted_ward_label_sets(phi0, curvature, K_grid=K_grid)
    else:
        label_sets = {
            int(k): _canonical_labels(np.asarray(labels, dtype=np.int64))
            for k, labels in label_sets.items()
            if int(k) in requested_grid
        }
    candidates: list[PartitionCandidate] = []
    seen: set[bytes] = set()
    n_eff = effective_bic_cell_count(data)

    for requested_k in sorted(label_sets):
        labels0 = _canonical_labels(label_sets[int(requested_k)])
        for source, labels in (
            (f"hessian_ward_K{int(requested_k)}", labels0),
            (f"hessian_ward_cem_K{int(requested_k)}", labels0),
        ):
            if source.startswith("hessian_ward_cem"):
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


def summarize_best_bic_by_K(candidates: Sequence[PartitionCandidate]) -> list[dict[str, object]]:
    by_k: dict[int, PartitionCandidate] = {}
    counts: dict[int, int] = {}
    for candidate in candidates:
        counts[int(candidate.K)] = counts.get(int(candidate.K), 0) + 1
        current = by_k.get(int(candidate.K))
        if current is None or float(candidate.bic) < float(current.bic):
            by_k[int(candidate.K)] = candidate
    rows: list[dict[str, object]] = []
    for candidate_k in sorted(by_k):
        candidate = by_k[candidate_k]
        rows.append(
            {
                "K": int(candidate_k),
                "candidate_count": int(counts[candidate_k]),
                "best_source": str(candidate.source),
                "best_fit_loss": float(candidate.fit_loss),
                "best_bic": float(candidate.bic),
                "best_active_df": -1 if candidate.active_df is None else int(candidate.active_df),
                "n_eff": -1 if candidate.n_eff is None else int(candidate.n_eff),
            }
        )
    return rows
