from __future__ import annotations

import hashlib

import numpy as np

from ..io.data import TumorData
from ..core.fusion.multiplicity import infer_multiplicity_posterior_numpy
from ..core.fusion.partition_starts import PartitionCandidate
from .config import (
    LIKELIHOOD_PARTITION_K_ANCHORS,
    LIKELIHOOD_PARTITION_K_MAX,
)

def _likelihood_partition_k_grid(num_mutations: int) -> list[int]:
    k_max = min(int(LIKELIHOOD_PARTITION_K_MAX), int(num_mutations))
    if k_max <= 0:
        return []
    anchors = [int(value) for value in LIKELIHOOD_PARTITION_K_ANCHORS if 1 <= int(value) <= k_max]
    if k_max not in anchors:
        anchors.append(k_max)
    return sorted(set(anchors))


def _partition_candidate_requested_k(candidate: PartitionCandidate) -> int:
    value = candidate.diagnostics.get("requested_K", candidate.K)
    if not np.isfinite(float(value)):
        return int(candidate.K)
    return int(round(float(value)))


def _best_partition_candidate(candidates: list[PartitionCandidate]) -> PartitionCandidate | None:
    finite_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.finite_candidate_found)
        and np.isfinite(float(candidate.bic))
        and np.isfinite(float(candidate.fit_loss))
    ]
    if not finite_candidates:
        return None
    return min(
        finite_candidates,
        key=lambda candidate: (
            float(candidate.bic),
            float(candidate.fit_loss),
            int(candidate.K),
            str(candidate.source),
        ),
    )


def _likelihood_partition_refinement_k_grid(
    candidates: list[PartitionCandidate],
    sparse_grid: list[int],
    *,
    num_mutations: int,
) -> tuple[list[int], str]:
    if not candidates or not sparse_grid:
        return [], "none"
    best = _best_partition_candidate(candidates)
    if best is None:
        return [], "none"

    k_cap = min(int(LIKELIHOOD_PARTITION_K_MAX), int(num_mutations))
    grid = sorted(set(int(k) for k in sparse_grid if 1 <= int(k) <= k_cap))
    if not grid:
        return [], "none"

    requested_k = int(np.clip(_partition_candidate_requested_k(best), 1, k_cap))
    effective_k = int(np.clip(int(best.K), 1, k_cap))
    focus_k = requested_k if requested_k in grid else effective_k
    if focus_k not in grid:
        focus_k = min(grid, key=lambda value: abs(int(value) - int(effective_k)))
    focus_idx = grid.index(int(focus_k))

    left_anchor = grid[focus_idx - 1] if focus_idx > 0 else int(focus_k)
    right_anchor = grid[focus_idx + 1] if focus_idx + 1 < len(grid) else int(focus_k)
    left_gap = int(focus_k) - int(left_anchor)
    right_gap = int(right_anchor) - int(focus_k)
    hits_k_cap = bool(k_cap == int(LIKELIHOOD_PARTITION_K_MAX) and effective_k >= k_cap)
    in_sparse_interval = bool(left_gap > 1 or right_gap > 1)

    if hits_k_cap and focus_idx > 0:
        left_anchor = grid[focus_idx - 1]
        right_anchor = int(focus_k)
        reason = "k_cap"
    elif in_sparse_interval:
        reason = "coarse_interval"
    else:
        return [], "none"

    refine_grid = [
        int(k)
        for k in range(int(left_anchor) + 1, int(right_anchor))
        if int(k) not in grid and 1 <= int(k) <= k_cap
    ]
    if not refine_grid:
        return [], "none"
    return refine_grid, reason


def _deduplicate_partition_candidates(
    candidates: list[PartitionCandidate],
) -> list[PartitionCandidate]:
    best_by_signature: dict[str, PartitionCandidate] = {}
    for candidate in candidates:
        signature = _partition_signature(candidate.labels)
        current = best_by_signature.get(signature)
        if current is None or (
            float(candidate.bic),
            float(candidate.fit_loss),
            int(candidate.K),
            str(candidate.source),
        ) < (
            float(current.bic),
            float(current.fit_loss),
            int(current.K),
            str(current.source),
        ):
            best_by_signature[signature] = candidate
    return sorted(
        best_by_signature.values(),
        key=lambda candidate: (
            float(candidate.bic),
            int(candidate.K),
            str(candidate.source),
        ),
    )


def _canonical_partition_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
    root_to_label: dict[int, int] = {}
    canonical = np.empty_like(labels)
    next_label = 0
    for idx, value in enumerate(labels):
        key = int(value)
        label = root_to_label.get(key)
        if label is None:
            label = next_label
            root_to_label[key] = label
            next_label += 1
        canonical[idx] = label
    return canonical


def _partition_blocks(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    canonical = _canonical_partition_labels(labels)
    if canonical.size == 0:
        return ()
    blocks = [
        tuple(int(idx) for idx in np.flatnonzero(canonical == int(label)).tolist())
        for label in np.unique(canonical)
    ]
    return tuple(sorted(blocks))


def _partition_signature(labels: np.ndarray) -> str:
    blocks = _partition_blocks(labels)
    if not blocks:
        return "empty"
    hasher = hashlib.blake2b(digest_size=12)
    for block in blocks:
        hasher.update(np.asarray([len(block)], dtype=np.int64).tobytes())
        if block:
            hasher.update(np.asarray(block, dtype=np.int64).tobytes())
    return f"{len(blocks)}:{hasher.hexdigest()}"


def _partition_is_coarsening(fine_labels: np.ndarray, coarse_labels: np.ndarray) -> bool:
    fine = _canonical_partition_labels(fine_labels)
    coarse = _canonical_partition_labels(coarse_labels)
    if fine.shape != coarse.shape:
        return False
    for label in np.unique(fine):
        coarse_values = np.unique(coarse[fine == int(label)])
        if coarse_values.size > 1:
            return False
    return True


def _cluster_sizes_text(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return ""
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return ",".join(str(int(value)) for value in counts.tolist())


def _max_cluster_diameter(diameters: np.ndarray) -> float:
    values = np.asarray(diameters, dtype=np.float64)
    return float(np.max(values)) if values.size else 0.0



def _centers_from_partition_labels(phi: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    labels = _canonical_partition_labels(labels)
    centers = np.zeros((int(num_clusters), phi.shape[1]), dtype=np.float64)
    for label in range(int(num_clusters)):
        mask = labels == int(label)
        if np.any(mask):
            centers[label] = np.mean(phi[mask], axis=0)
    return centers


def _multiplicity_summary_for_phi(
    data: TumorData,
    phi: np.ndarray,
    *,
    major_prior: float = 0.5,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    posterior = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    return (
        posterior.gamma_major,
        posterior.major_call,
        posterior.multiplicity_call,
        posterior.estimation_mask,
    )
