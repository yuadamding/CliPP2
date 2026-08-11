from __future__ import annotations

import hashlib

import numpy as np

from ..core.model import FitResult
from ..core.fusion.solver import cluster_labels_from_edges
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.fusion.refit import _canonical_labels as _canonical_partition_labels
from ..core.fusion.types import (
    CompressedEdgeCertificate,
    PairwiseFusionGraph,
    QuotientWorksetWarmState,
)
from .config import (
    LIKELIHOOD_PARTITION_K_ANCHORS,
    LIKELIHOOD_PARTITION_K_MAX,
)
from .types import FusionPartition


def _likelihood_partition_k_grid(num_mutations: int) -> list[int]:
    k_max = min(int(LIKELIHOOD_PARTITION_K_MAX), int(num_mutations))
    if k_max <= 0:
        return []
    anchors = [
        int(value)
        for value in LIKELIHOOD_PARTITION_K_ANCHORS
        if 1 <= int(value) <= k_max
    ]
    if k_max not in anchors:
        anchors.append(k_max)
    return sorted(set(anchors))


def _partition_candidate_requested_k(candidate: PartitionCandidate) -> int:
    value = candidate.diagnostics.get("requested_K", candidate.K)
    if not np.isfinite(float(value)):
        return int(candidate.K)
    return int(round(float(value)))


def _best_partition_candidate(
    candidates: list[PartitionCandidate],
) -> PartitionCandidate | None:
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
    grid = sorted({int(k) for k in sparse_grid if 1 <= int(k) <= k_cap})
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


def _exact_partition_diameters(phi: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(phi, dtype=np.float64)
    canonical = _canonical_partition_labels(labels)
    n_clusters = int(canonical.max()) + 1 if canonical.size else 0
    diameters = np.zeros(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        rows = values[canonical == cluster]
        if rows.shape[0] <= 1:
            continue
        if rows.shape[1] == 1:
            diameters[cluster] = float(np.max(rows[:, 0]) - np.min(rows[:, 0]))
            continue
        maximum = 0.0
        for start in range(0, rows.shape[0], 512):
            block = rows[start : start + 512]
            distances = np.linalg.norm(block[:, None, :] - rows[None, :, :], axis=-1)
            if distances.size:
                maximum = max(maximum, float(np.max(distances)))
        diameters[cluster] = maximum
    return diameters


def extract_certified_fusion_partition(
    fit: FitResult,
    *,
    graph: PairwiseFusionGraph,
    tolerance: float,
) -> FusionPartition:
    """Extract one raw-fit partition and fail closed on tolerance chaining."""

    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition tolerance must be positive and finite.")
    phi = np.asarray(fit.phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("fit.phi must be a mutation-by-region matrix.")

    state = getattr(fit, "solver_state", None)
    certificate = getattr(state, "certificate", None)
    warm_state = getattr(state, "warm_state", None)
    if bool(getattr(fit, "full_kkt_certified", False)) and isinstance(
        certificate, CompressedEdgeCertificate
    ):
        labels = _canonical_partition_labels(
            certificate.labels.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        source = "solver_quotient"
    elif isinstance(warm_state, QuotientWorksetWarmState):
        labels = _canonical_partition_labels(
            warm_state.labels.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        source = "solver_quotient"
    else:
        labels = _canonical_partition_labels(
            cluster_labels_from_edges(
                phi,
                edge_u=graph.edge_u,
                edge_v=graph.edge_v,
                tol=tol,
            )
        )
        source = "verified_primal_equalities"

    if labels.shape != (phi.shape[0],):
        raise ValueError("Partition labels do not match the raw fit mutation count.")
    diameters = _exact_partition_diameters(phi, labels)
    max_diameter = float(np.max(diameters)) if diameters.size else 0.0
    certified = bool(np.all(np.isfinite(diameters)) and max_diameter <= tol)
    return FusionPartition(
        labels=labels.astype(np.int64, copy=False),
        signature=_partition_signature(labels),
        n_clusters=int(np.unique(labels).size),
        tolerance=tol,
        max_diameter=max_diameter,
        diameter_exact=True,
        certified=certified,
        source=source,
    )


def _cluster_sizes_text(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return ""
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return ",".join(str(int(value)) for value in counts.tolist())
