from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import PatientData

try:  # pragma: no cover - sklearn is expected in the project env, but keep a safe fallback.
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None


@dataclass(frozen=True)
class PartitionClusterProfile:
    indices: tuple[int, ...]
    centers: np.ndarray
    cost: float


@dataclass(frozen=True)
class PartitionSearchArtifacts:
    phi: np.ndarray
    phi_clustered: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    gamma_major: np.ndarray
    major_probability: np.ndarray
    major_call: np.ndarray
    multiplicity_call: np.ndarray
    multiplicity_estimated_mask: np.ndarray
    loglik: float
    penalized_objective: float
    lambda_value: float
    n_clusters: int
    iterations: int
    converged: bool
    device: str
    z_norm: np.ndarray
    history: list[float]


def _structural_penalty(data: PatientData, lambda_value: float, cluster_count: int) -> float:
    if cluster_count <= 0:
        return 0.0
    ms = max(int(data.num_mutations * data.num_samples), 2)
    m = max(int(data.num_mutations), 2)
    scale = float(data.num_samples) * float(np.log(ms)) + float(np.log(m))
    return float(lambda_value) * float(cluster_count) * scale


def _observed_region_loss_grid(
    beta_values: np.ndarray,
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous_mask: np.ndarray,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    beta_grid = np.asarray(beta_values, dtype=np.float64).reshape(-1, 1)
    alt_vec = np.asarray(alt, dtype=np.float64).reshape(1, -1)
    total_vec = np.asarray(total, dtype=np.float64).reshape(1, -1)
    nonalt = total_vec - alt_vec
    amb = np.asarray(ambiguous_mask, dtype=bool).reshape(1, -1)

    if alt_vec.shape[1] == 0:
        return np.zeros((beta_grid.shape[0],), dtype=np.float64)

    losses = np.zeros((beta_grid.shape[0], alt_vec.shape[1]), dtype=np.float64)
    fixed_mask = ~amb

    if np.any(amb):
        p_minus = np.clip(beta_grid * np.asarray(b_minus, dtype=np.float64).reshape(1, -1), eps, 1.0 - eps)
        p_plus = np.clip(beta_grid * np.asarray(b_plus, dtype=np.float64).reshape(1, -1), eps, 1.0 - eps)
        log_minor = (
            alt_vec * np.log(p_minus)
            + nonalt * np.log1p(-p_minus)
            + float(np.log(max(1.0 - major_prior, eps)))
        )
        log_major = (
            alt_vec * np.log(p_plus)
            + nonalt * np.log1p(-p_plus)
            + float(np.log(max(major_prior, eps)))
        )
        amb_losses = -np.logaddexp(log_minor, log_major)
        losses[:, amb.reshape(-1)] = amb_losses[:, amb.reshape(-1)]

    if np.any(fixed_mask):
        p_fixed = np.clip(beta_grid * np.asarray(b_fixed, dtype=np.float64).reshape(1, -1), eps, 1.0 - eps)
        fixed_losses = -(alt_vec * np.log(p_fixed) + nonalt * np.log1p(-p_fixed))
        losses[:, fixed_mask.reshape(-1)] = fixed_losses[:, fixed_mask.reshape(-1)]

    return np.sum(losses, axis=1)


def _golden_section_refine(
    objective,
    *,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    if right <= left + 1e-12:
        value = float(objective(np.asarray([left], dtype=np.float64))[0])
        return float(left), value

    golden = 0.5 * (np.sqrt(5.0) - 1.0)
    c = right - golden * (right - left)
    d = left + golden * (right - left)
    fc = float(objective(np.asarray([c], dtype=np.float64))[0])
    fd = float(objective(np.asarray([d], dtype=np.float64))[0])

    for _ in range(max(int(max_iter), 8)):
        if abs(right - left) <= tol * (1.0 + abs(left) + abs(right)):
            break
        if fc <= fd:
            right = d
            d = c
            fd = fc
            c = right - golden * (right - left)
            fc = float(objective(np.asarray([c], dtype=np.float64))[0])
        else:
            left = c
            c = d
            fc = fd
            d = left + golden * (right - left)
            fd = float(objective(np.asarray([d], dtype=np.float64))[0])

    if fc <= fd:
        return float(c), float(fc)
    return float(d), float(fd)


def _optimize_region_center(
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous_mask: np.ndarray,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    beta_hint: float | None = None,
) -> tuple[float, float]:
    lower = float(max(lower, eps))
    upper = float(max(upper, lower))
    objective = lambda grid: _observed_region_loss_grid(
        grid,
        alt=alt,
        total=total,
        b_minus=b_minus,
        b_plus=b_plus,
        b_fixed=b_fixed,
        ambiguous_mask=ambiguous_mask,
        major_prior=major_prior,
        eps=eps,
    )
    if upper <= lower + 1e-12:
        value = float(objective(np.asarray([lower], dtype=np.float64))[0])
        return lower, value

    if beta_hint is not None and np.isfinite(beta_hint):
        hint = float(np.clip(beta_hint, lower, upper))
        local_left = max(lower, hint / 3.0)
        local_right = min(upper, hint * 3.0)
    else:
        local_left = lower
        local_right = upper

    if local_right <= local_left + 1e-12:
        local_left, local_right = lower, upper

    grid = np.exp(np.linspace(np.log(local_left), np.log(local_right), num=17))
    grid = np.unique(np.clip(np.concatenate(([lower], grid, [upper])), lower, upper))
    grid_values = objective(grid)
    best_index = int(np.argmin(grid_values))
    best_beta = float(grid[best_index])
    best_value = float(grid_values[best_index])

    left = float(lower if best_index == 0 else grid[best_index - 1])
    right = float(upper if best_index == grid.shape[0] - 1 else grid[best_index + 1])
    refined_beta, refined_value = _golden_section_refine(
        objective,
        left=left,
        right=right,
        tol=tol,
        max_iter=max_iter,
    )
    if refined_value < best_value:
        return refined_beta, refined_value
    return best_beta, best_value


def _cluster_major_posterior(
    beta: float,
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    ambiguous_mask: np.ndarray,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    alt_vec = np.asarray(alt, dtype=np.float64)
    total_vec = np.asarray(total, dtype=np.float64)
    nonalt = total_vec - alt_vec
    gamma_major = np.zeros_like(alt_vec, dtype=np.float64)
    amb = np.asarray(ambiguous_mask, dtype=bool)
    if not np.any(amb):
        return gamma_major

    p_minus = np.clip(np.asarray(b_minus, dtype=np.float64)[amb] * beta, eps, 1.0 - eps)
    p_plus = np.clip(np.asarray(b_plus, dtype=np.float64)[amb] * beta, eps, 1.0 - eps)
    log_minor = (
        alt_vec[amb] * np.log(p_minus)
        + nonalt[amb] * np.log1p(-p_minus)
        + float(np.log(max(1.0 - major_prior, eps)))
    )
    log_major = (
        alt_vec[amb] * np.log(p_plus)
        + nonalt[amb] * np.log1p(-p_plus)
        + float(np.log(max(major_prior, eps)))
    )
    delta = np.clip(log_major - log_minor, -60.0, 60.0)
    gamma_major[amb] = 1.0 / (1.0 + np.exp(-delta))
    return gamma_major


def _profile_cluster(
    data: PatientData,
    indices: np.ndarray,
    *,
    lambda_value: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    beta_hint: np.ndarray | None,
    cache: dict[tuple[int, ...], PartitionClusterProfile],
) -> PartitionClusterProfile:
    ordered = tuple(int(i) for i in np.sort(np.asarray(indices, dtype=np.int64)))
    if not ordered:
        raise ValueError("Cannot profile an empty cluster.")
    cached = cache.get(ordered)
    if cached is not None:
        return cached

    rows = np.asarray(ordered, dtype=np.int64)
    centers = np.zeros((data.num_samples,), dtype=np.float32)
    total_cost = 0.0
    for sample_idx in range(data.num_samples):
        upper = float(np.min(data.phi_upper[rows, sample_idx]))
        beta0 = None if beta_hint is None else float(beta_hint[sample_idx])
        beta, value = _optimize_region_center(
            alt=data.alt_counts[rows, sample_idx],
            total=data.total_counts[rows, sample_idx],
            b_minus=data.scaling[rows, sample_idx] * data.minor_cn[rows, sample_idx],
            b_plus=data.scaling[rows, sample_idx] * data.major_cn[rows, sample_idx],
            b_fixed=data.scaling[rows, sample_idx] * data.fixed_multiplicity[rows, sample_idx],
            ambiguous_mask=data.multiplicity_estimation_mask[rows, sample_idx],
            lower=eps,
            upper=upper,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=beta0,
        )
        centers[sample_idx] = float(beta)
        total_cost += float(value)

    profile = PartitionClusterProfile(indices=ordered, centers=centers, cost=float(total_cost))
    cache[ordered] = profile
    return profile


def _pilot_profiles_exact(data: PatientData, *, major_prior: float, eps: float, tol: float, max_iter: int) -> np.ndarray:
    pilot = np.zeros_like(data.phi_init, dtype=np.float32)
    for mutation_idx in range(data.num_mutations):
        for sample_idx in range(data.num_samples):
            upper = float(data.phi_upper[mutation_idx, sample_idx])
            beta, _ = _optimize_region_center(
                alt=np.asarray([data.alt_counts[mutation_idx, sample_idx]], dtype=np.float64),
                total=np.asarray([data.total_counts[mutation_idx, sample_idx]], dtype=np.float64),
                b_minus=np.asarray([data.scaling[mutation_idx, sample_idx] * data.minor_cn[mutation_idx, sample_idx]], dtype=np.float64),
                b_plus=np.asarray([data.scaling[mutation_idx, sample_idx] * data.major_cn[mutation_idx, sample_idx]], dtype=np.float64),
                b_fixed=np.asarray([data.scaling[mutation_idx, sample_idx] * data.fixed_multiplicity[mutation_idx, sample_idx]], dtype=np.float64),
                ambiguous_mask=np.asarray([data.multiplicity_estimation_mask[mutation_idx, sample_idx]], dtype=bool),
                lower=eps,
                upper=upper,
                major_prior=major_prior,
                eps=eps,
                tol=tol,
                max_iter=max_iter,
                beta_hint=float(data.phi_init[mutation_idx, sample_idx]),
            )
            pilot[mutation_idx, sample_idx] = float(beta)
    return pilot.astype(np.float32, copy=False)


def _warm_start_labels(phi_start: np.ndarray) -> np.ndarray | None:
    rounded = np.round(np.asarray(phi_start, dtype=np.float32), 6)
    _, labels = np.unique(rounded, axis=0, return_inverse=True)
    num_clusters = int(np.unique(labels).shape[0])
    if 1 < num_clusters < rounded.shape[0]:
        return labels.astype(np.int64, copy=False)
    return None


def _initialize_microclusters(pilot: np.ndarray, phi_start: np.ndarray | None) -> np.ndarray:
    if phi_start is not None:
        warm_labels = _warm_start_labels(phi_start)
        if warm_labels is not None:
            return warm_labels

    num_mutations = int(pilot.shape[0])
    if num_mutations <= 1:
        return np.zeros((num_mutations,), dtype=np.int64)

    target_k = min(num_mutations, max(2, int(np.sqrt(max(num_mutations, 1)))))
    if KMeans is None:
        order = np.argsort(np.linalg.norm(pilot, axis=1))
        labels = np.zeros((num_mutations,), dtype=np.int64)
        for cluster_id, indices in enumerate(np.array_split(order, target_k)):
            labels[np.asarray(indices, dtype=np.int64)] = cluster_id
        return labels

    kmeans = KMeans(n_clusters=target_k, n_init=5, random_state=0)
    labels = kmeans.fit_predict(pilot).astype(np.int64, copy=False)
    _, relabeled = np.unique(labels, return_inverse=True)
    return relabeled.astype(np.int64, copy=False)


def _clusters_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    clusters: list[np.ndarray] = []
    for label in np.unique(labels):
        clusters.append(np.flatnonzero(labels == label).astype(np.int64))
    return clusters


def _labels_from_clusters(clusters: list[np.ndarray], num_mutations: int) -> np.ndarray:
    labels = np.full((num_mutations,), -1, dtype=np.int64)
    for cluster_id, cluster in enumerate(clusters):
        labels[np.asarray(cluster, dtype=np.int64)] = cluster_id
    if np.any(labels < 0):
        raise RuntimeError("Partition construction left unassigned mutations.")
    return labels


def _partition_cost(
    clusters: list[np.ndarray],
    *,
    data: PatientData,
    options_major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    lambda_value: float,
    cache: dict[tuple[int, ...], PartitionClusterProfile],
    beta_hints: dict[tuple[int, ...], np.ndarray] | None = None,
) -> tuple[float, list[PartitionClusterProfile]]:
    profiles: list[PartitionClusterProfile] = []
    total = 0.0
    for cluster in clusters:
        key = tuple(int(i) for i in np.sort(cluster.astype(np.int64, copy=False)))
        hint = None if beta_hints is None else beta_hints.get(key)
        profile = _profile_cluster(
            data=data,
            indices=cluster,
            lambda_value=lambda_value,
            major_prior=options_major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=hint,
            cache=cache,
        )
        profiles.append(profile)
        total += float(profile.cost)
    total += _structural_penalty(data, lambda_value, len(clusters))
    return float(total), profiles


def _binary_refine_split(
    cluster_indices: np.ndarray,
    pilot: np.ndarray,
    *,
    data: PatientData,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    lambda_value: float,
    cache: dict[tuple[int, ...], PartitionClusterProfile],
) -> tuple[np.ndarray, np.ndarray] | None:
    if cluster_indices.size < 2:
        return None
    cluster_pilot = pilot[cluster_indices]
    centroid = np.mean(cluster_pilot, axis=0, dtype=np.float64)
    first_seed_pos = int(np.argmax(np.sum((cluster_pilot - centroid[None, :]) ** 2, axis=1)))
    first_seed = cluster_pilot[first_seed_pos]
    second_seed_pos = int(np.argmax(np.sum((cluster_pilot - first_seed[None, :]) ** 2, axis=1)))
    if first_seed_pos == second_seed_pos:
        return None
    second_seed = cluster_pilot[second_seed_pos]

    dist_first = np.sum((cluster_pilot - first_seed[None, :]) ** 2, axis=1)
    dist_second = np.sum((cluster_pilot - second_seed[None, :]) ** 2, axis=1)
    assign_right = dist_second < dist_first
    left = np.sort(cluster_indices[~assign_right].astype(np.int64, copy=False))
    right = np.sort(cluster_indices[assign_right].astype(np.int64, copy=False))
    if left.size == 0 or right.size == 0:
        return None

    for _ in range(max(4, max_iter // 4)):
        improved = False
        left_profile = _profile_cluster(
            data=data,
            indices=left,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=np.mean(pilot[left], axis=0),
            cache=cache,
        )
        right_profile = _profile_cluster(
            data=data,
            indices=right,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=np.mean(pilot[right], axis=0),
            cache=cache,
        )
        current_cost = left_profile.cost + right_profile.cost

        for source_name in ("left", "right"):
            source = left if source_name == "left" else right
            dest = right if source_name == "left" else left
            if source.size <= 1:
                continue
            source_cost = left_profile.cost if source_name == "left" else right_profile.cost
            dest_cost = right_profile.cost if source_name == "left" else left_profile.cost
            best_delta = 0.0
            best_move: tuple[np.ndarray, np.ndarray] | None = None
            for mutation in source:
                new_source = source[source != mutation]
                new_dest = np.sort(np.append(dest, mutation).astype(np.int64))
                source_new_cost = _profile_cluster(
                    data=data,
                    indices=new_source,
                    lambda_value=lambda_value,
                    major_prior=major_prior,
                    eps=eps,
                    tol=tol,
                    max_iter=max_iter,
                    beta_hint=np.mean(pilot[new_source], axis=0),
                    cache=cache,
                ).cost
                dest_new_cost = _profile_cluster(
                    data=data,
                    indices=new_dest,
                    lambda_value=lambda_value,
                    major_prior=major_prior,
                    eps=eps,
                    tol=tol,
                    max_iter=max_iter,
                    beta_hint=np.mean(pilot[new_dest], axis=0),
                    cache=cache,
                ).cost
                delta = (source_new_cost + dest_new_cost) - (source_cost + dest_cost)
                if delta < best_delta - tol:
                    if source_name == "left":
                        best_move = (new_source, new_dest)
                    else:
                        best_move = (new_dest, new_source)
                    best_delta = float(delta)
            if best_move is not None:
                left, right = best_move
                improved = True
                break
        if not improved:
            break

        updated_cost = _profile_cluster(
            data=data,
            indices=left,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=np.mean(pilot[left], axis=0),
            cache=cache,
        ).cost + _profile_cluster(
            data=data,
            indices=right,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            beta_hint=np.mean(pilot[right], axis=0),
            cache=cache,
        ).cost
        if updated_cost >= current_cost - tol:
            break

    return left, right


def fit_profiled_partition_search(
    data: PatientData,
    *,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | None = None,
    verbose: bool = False,
) -> PartitionSearchArtifacts:
    pilot = (
        _pilot_profiles_exact(
            data,
            major_prior=major_prior,
            eps=eps,
            tol=max(tol, 1e-6),
            max_iter=max(inner_max_iter, 16),
        )
        if phi_start is None
        else np.clip(np.asarray(phi_start, dtype=np.float32), eps, data.phi_upper)
    )
    labels = _initialize_microclusters(pilot, phi_start)
    clusters = _clusters_from_labels(labels)
    cache: dict[tuple[int, ...], PartitionClusterProfile] = {}
    history: list[float] = []
    converged = False
    iterations = 0

    for outer_iter in range(max(int(outer_max_iter), 1)):
        iterations = outer_iter + 1
        objective, profiles = _partition_cost(
            clusters,
            data=data,
            options_major_prior=major_prior,
            eps=eps,
            tol=max(tol, 1e-6),
            max_iter=max(inner_max_iter, 16),
            lambda_value=lambda_value,
            cache=cache,
        )
        history.append(float(-objective))
        if verbose:
            print(
                f"[partition-search] iter={iterations:02d} "
                f"K={len(clusters)} objective={objective:.6f}"
            )

        improved = False

        accepted_moves = 0
        while accepted_moves < max(int(outer_max_iter), 8):
            best_delta = 0.0
            best_clusters: list[np.ndarray] | None = None
            current_penalty = _structural_penalty(data, lambda_value, len(clusters))
            for cluster_idx, cluster in enumerate(clusters):
                cluster_profile = _profile_cluster(
                    data=data,
                    indices=cluster,
                    lambda_value=lambda_value,
                    major_prior=major_prior,
                    eps=eps,
                    tol=max(tol, 1e-6),
                    max_iter=max(inner_max_iter, 16),
                    beta_hint=np.mean(pilot[cluster], axis=0),
                    cache=cache,
                )
                for mutation in cluster:
                    remaining = cluster[cluster != mutation]
                    remaining_cost = 0.0 if remaining.size == 0 else _profile_cluster(
                        data=data,
                        indices=remaining,
                        lambda_value=lambda_value,
                        major_prior=major_prior,
                        eps=eps,
                        tol=max(tol, 1e-6),
                        max_iter=max(inner_max_iter, 16),
                        beta_hint=None if remaining.size == 0 else np.mean(pilot[remaining], axis=0),
                        cache=cache,
                    ).cost
                    penalty_after_remove = _structural_penalty(
                        data,
                        lambda_value,
                        len(clusters) - (1 if remaining.size == 0 else 0),
                    )
                    for dest_idx, dest in enumerate(clusters):
                        if dest_idx == cluster_idx:
                            continue
                        augmented = np.sort(np.append(dest, mutation).astype(np.int64))
                        augmented_cost = _profile_cluster(
                            data=data,
                            indices=augmented,
                            lambda_value=lambda_value,
                            major_prior=major_prior,
                            eps=eps,
                            tol=max(tol, 1e-6),
                            max_iter=max(inner_max_iter, 16),
                            beta_hint=np.mean(pilot[augmented], axis=0),
                            cache=cache,
                        ).cost
                        delta = (
                            remaining_cost
                            + augmented_cost
                            - cluster_profile.cost
                            - _profile_cluster(
                                data=data,
                                indices=dest,
                                lambda_value=lambda_value,
                                major_prior=major_prior,
                                eps=eps,
                                tol=max(tol, 1e-6),
                                max_iter=max(inner_max_iter, 16),
                                beta_hint=np.mean(pilot[dest], axis=0),
                                cache=cache,
                            ).cost
                            + penalty_after_remove
                            - current_penalty
                        )
                        if delta < best_delta - tol:
                            updated = [c.copy() for c in clusters]
                            if remaining.size == 0:
                                updated.pop(cluster_idx)
                                insert_idx = dest_idx - (1 if dest_idx > cluster_idx else 0)
                                updated[insert_idx] = augmented
                            else:
                                updated[cluster_idx] = remaining
                                updated[dest_idx] = augmented
                            best_delta = float(delta)
                            best_clusters = [np.sort(c.astype(np.int64, copy=False)) for c in updated]
                    if remaining.size > 0:
                        singleton = np.asarray([int(mutation)], dtype=np.int64)
                        delta_new = (
                            remaining_cost
                            + _profile_cluster(
                                data=data,
                                indices=singleton,
                                lambda_value=lambda_value,
                                major_prior=major_prior,
                                eps=eps,
                                tol=max(tol, 1e-6),
                                max_iter=max(inner_max_iter, 16),
                                beta_hint=pilot[singleton[0]],
                                cache=cache,
                            ).cost
                            - cluster_profile.cost
                            + _structural_penalty(data, lambda_value, len(clusters) + 1)
                            - current_penalty
                        )
                        if delta_new < best_delta - tol:
                            updated = [c.copy() for c in clusters]
                            updated[cluster_idx] = remaining
                            updated.append(singleton)
                            best_delta = float(delta_new)
                            best_clusters = [np.sort(c.astype(np.int64, copy=False)) for c in updated]
            if best_clusters is None:
                break
            clusters = best_clusters
            accepted_moves += 1
            improved = True

        merge_improved = True
        while merge_improved:
            merge_improved = False
            best_delta = 0.0
            best_merge: list[np.ndarray] | None = None
            current_penalty = _structural_penalty(data, lambda_value, len(clusters))
            for left_idx in range(len(clusters)):
                left = clusters[left_idx]
                left_cost = _profile_cluster(
                    data=data,
                    indices=left,
                    lambda_value=lambda_value,
                    major_prior=major_prior,
                    eps=eps,
                    tol=max(tol, 1e-6),
                    max_iter=max(inner_max_iter, 16),
                    beta_hint=np.mean(pilot[left], axis=0),
                    cache=cache,
                ).cost
                for right_idx in range(left_idx + 1, len(clusters)):
                    right = clusters[right_idx]
                    right_cost = _profile_cluster(
                        data=data,
                        indices=right,
                        lambda_value=lambda_value,
                        major_prior=major_prior,
                        eps=eps,
                        tol=max(tol, 1e-6),
                        max_iter=max(inner_max_iter, 16),
                        beta_hint=np.mean(pilot[right], axis=0),
                        cache=cache,
                    ).cost
                    merged = np.sort(np.concatenate([left, right]).astype(np.int64, copy=False))
                    merged_cost = _profile_cluster(
                        data=data,
                        indices=merged,
                        lambda_value=lambda_value,
                        major_prior=major_prior,
                        eps=eps,
                        tol=max(tol, 1e-6),
                        max_iter=max(inner_max_iter, 16),
                        beta_hint=0.5 * (
                            np.mean(pilot[left], axis=0) + np.mean(pilot[right], axis=0)
                        ),
                        cache=cache,
                    ).cost
                    delta = (
                        merged_cost
                        - left_cost
                        - right_cost
                        + _structural_penalty(data, lambda_value, len(clusters) - 1)
                        - current_penalty
                    )
                    if delta < best_delta - tol:
                        updated = [c.copy() for c in clusters]
                        updated[left_idx] = merged
                        updated.pop(right_idx)
                        best_delta = float(delta)
                        best_merge = [np.sort(c.astype(np.int64, copy=False)) for c in updated]
            if best_merge is not None:
                clusters = best_merge
                improved = True
                merge_improved = True

        best_delta = 0.0
        best_split: list[np.ndarray] | None = None
        current_penalty = _structural_penalty(data, lambda_value, len(clusters))
        for cluster_idx, cluster in enumerate(clusters):
            if cluster.size < 4:
                continue
            cluster_cost = _profile_cluster(
                data=data,
                indices=cluster,
                lambda_value=lambda_value,
                major_prior=major_prior,
                eps=eps,
                tol=max(tol, 1e-6),
                max_iter=max(inner_max_iter, 16),
                beta_hint=np.mean(pilot[cluster], axis=0),
                cache=cache,
            ).cost
            split_clusters = _binary_refine_split(
                cluster,
                pilot,
                data=data,
                major_prior=major_prior,
                eps=eps,
                tol=max(tol, 1e-6),
                max_iter=max(inner_max_iter, 16),
                lambda_value=lambda_value,
                cache=cache,
            )
            if split_clusters is None:
                continue
            left, right = split_clusters
            split_cost = _profile_cluster(
                data=data,
                indices=left,
                lambda_value=lambda_value,
                major_prior=major_prior,
                eps=eps,
                tol=max(tol, 1e-6),
                max_iter=max(inner_max_iter, 16),
                beta_hint=np.mean(pilot[left], axis=0),
                cache=cache,
            ).cost + _profile_cluster(
                data=data,
                indices=right,
                lambda_value=lambda_value,
                major_prior=major_prior,
                eps=eps,
                tol=max(tol, 1e-6),
                max_iter=max(inner_max_iter, 16),
                beta_hint=np.mean(pilot[right], axis=0),
                cache=cache,
            ).cost
            delta = (
                split_cost
                - cluster_cost
                + _structural_penalty(data, lambda_value, len(clusters) + 1)
                - current_penalty
            )
            if delta < best_delta - tol:
                updated = [c.copy() for c in clusters]
                updated[cluster_idx] = left
                updated.append(right)
                best_delta = float(delta)
                best_split = [np.sort(c.astype(np.int64, copy=False)) for c in updated]
        if best_split is not None:
            clusters = best_split
            improved = True

        if not improved:
            converged = True
            break

    final_objective, final_profiles = _partition_cost(
        clusters,
        data=data,
        options_major_prior=major_prior,
        eps=eps,
        tol=max(tol, 1e-6),
        max_iter=max(inner_max_iter, 16),
        lambda_value=lambda_value,
        cache=cache,
    )
    cluster_labels = _labels_from_clusters(clusters, data.num_mutations)
    cluster_centers = np.vstack([profile.centers for profile in final_profiles]).astype(np.float32, copy=False)
    phi_clustered = cluster_centers[cluster_labels]
    gamma_major = np.ones_like(phi_clustered, dtype=np.float32)

    for cluster_id, profile in enumerate(final_profiles):
        rows = np.asarray(profile.indices, dtype=np.int64)
        for sample_idx in range(data.num_samples):
            gamma = _cluster_major_posterior(
                float(profile.centers[sample_idx]),
                alt=data.alt_counts[rows, sample_idx],
                total=data.total_counts[rows, sample_idx],
                b_minus=data.scaling[rows, sample_idx] * data.minor_cn[rows, sample_idx],
                b_plus=data.scaling[rows, sample_idx] * data.major_cn[rows, sample_idx],
                ambiguous_mask=data.multiplicity_estimation_mask[rows, sample_idx],
                major_prior=major_prior,
                eps=eps,
            )
            gamma_major[rows, sample_idx] = np.where(
                data.multiplicity_estimation_mask[rows, sample_idx],
                gamma,
                1.0,
            ).astype(np.float32, copy=False)

    major_probability = gamma_major.astype(np.float32, copy=False)
    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        data.multiplicity_estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(np.float32, copy=False)
    loglik = float(-sum(profile.cost for profile in final_profiles))
    penalized_objective = float(-final_objective)
    history.append(penalized_objective)

    return PartitionSearchArtifacts(
        phi=phi_clustered.astype(np.float32, copy=False),
        phi_clustered=phi_clustered.astype(np.float32, copy=False),
        cluster_labels=cluster_labels.astype(np.int64, copy=False),
        cluster_centers=cluster_centers.astype(np.float32, copy=False),
        gamma_major=major_probability.astype(np.float32, copy=False),
        major_probability=major_probability.astype(np.float32, copy=False),
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call.astype(np.float32, copy=False),
        multiplicity_estimated_mask=data.multiplicity_estimation_mask.astype(bool, copy=False),
        loglik=loglik,
        penalized_objective=penalized_objective,
        lambda_value=float(lambda_value),
        n_clusters=int(cluster_centers.shape[0]),
        iterations=int(iterations),
        converged=bool(converged),
        device="cpu",
        z_norm=np.zeros(0, dtype=np.float32),
        history=[float(value) for value in history],
    )


__all__ = [
    "PartitionClusterProfile",
    "PartitionSearchArtifacts",
    "fit_profiled_partition_search",
]
