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
)
from ..io.data import TumorData
from .config import (
    LIKELIHOOD_PARTITION_K_ANCHORS,
    LIKELIHOOD_PARTITION_K_MAX,
)
from .types import FusionPartition, RawClonalBlockCertificate


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


def _partition_signature(
    labels: np.ndarray,
    mutation_ids: tuple[str, ...] | list[str] | None = None,
) -> str:
    blocks = _partition_blocks(labels)
    if not blocks:
        return "empty"
    hasher = hashlib.blake2b(digest_size=12)
    if mutation_ids is None:
        identity_blocks: list[tuple[str, ...] | tuple[int, ...]] = list(blocks)
    else:
        identities = tuple(str(value) for value in mutation_ids)
        if len(identities) != np.asarray(labels).size:
            raise ValueError("mutation_ids must match the partition label count.")
        if len(set(identities)) != len(identities):
            raise ValueError("mutation_ids must be unique for partition identity.")
        identity_blocks = sorted(
            tuple(sorted(identities[index] for index in block)) for block in blocks
        )
    for block in identity_blocks:
        hasher.update(np.asarray([len(block)], dtype=np.int64).tobytes())
        for value in block:
            encoded = str(value).encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
    return f"{len(blocks)}:{hasher.hexdigest()}"


def _raw_clonal_block_signature(member_mutation_ids: tuple[str, ...]) -> str:
    members = tuple(sorted(str(value) for value in member_mutation_ids))
    digest = hashlib.sha256()
    for value in members:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return f"{len(members)}:{digest.hexdigest()[:24]}"


def extract_exact_raw_clonal_block(
    fit: FitResult,
    *,
    data: TumorData,
    witness_index: int,
    target: np.ndarray,
    anchor_tolerance: float,
    minimum_cluster_size: int = 1,
    minimum_observed_support_per_region: int = 1,
) -> RawClonalBlockCertificate:
    """Certify the exact raw CCF-one block containing a witness mutation."""

    phi = np.asarray(fit.phi, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64).reshape(-1)
    tolerance = float(anchor_tolerance)
    witness = int(witness_index)
    if phi.shape != (int(data.num_mutations), int(data.num_regions)):
        raise ValueError("Raw CCF matrix does not match the tumor dimensions.")
    if target_array.shape != (int(data.num_regions),):
        raise ValueError("Raw clonal target must contain one value per region.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Raw clonal-block equality tolerance must be positive.")
    if not 0 <= witness < int(data.num_mutations):
        raise ValueError("Raw clonal witness index is invalid.")
    if int(minimum_cluster_size) < 1:
        raise ValueError("Raw clonal minimum cluster size must be positive.")
    if int(minimum_observed_support_per_region) < 0:
        raise ValueError("Raw clonal minimum observed support must be nonnegative.")

    state = getattr(fit, "solver_state", None)
    certificate = getattr(state, "certificate", None)
    certificate_graph_hash_matches = True
    if isinstance(certificate, CompressedEdgeCertificate):
        expected_graph_hash = str(getattr(fit, "original_graph_hash", ""))
        certificate_graph_hash_matches = bool(
            expected_graph_hash
            and str(certificate.graph_hash) == expected_graph_hash
        )
        certificate_labels = certificate.labels.detach().cpu().numpy()
        if certificate_labels.shape != (int(data.num_mutations),):
            raise ValueError("Compressed certificate labels have the wrong shape.")
    # The biological block is the complete equal-row target set, not a possibly
    # finer solver-compression block.  This also ensures that every raw
    # mutation-specific estimate saturated at the CCF-one boundary remains in
    # the protected clonal block.
    member_mask = np.max(np.abs(phi - target_array[None, :]), axis=1) <= tolerance
    member_mask[witness] = True
    member_indices = np.flatnonzero(member_mask).astype(np.int64)
    member_ids = tuple(str(data.mutation_ids[index]) for index in member_indices)
    member_phi = phi[member_indices]
    centroid = np.mean(member_phi, axis=0)
    common_center = phi[witness].copy()
    member_residual = float(
        np.max(np.abs(member_phi - target_array[None, :]))
    )
    centroid_residual = float(np.max(np.abs(centroid - target_array)))
    observed = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    count_observed = getattr(data, "count_observed", None)
    if count_observed is not None:
        observed &= np.asarray(count_observed, dtype=bool)
    observed_support = np.sum(observed[member_indices], axis=0).astype(np.int64)

    if not certificate_graph_hash_matches:
        failure_reason = "clonal_block_certificate_graph_hash_mismatch"
    elif member_residual > tolerance:
        failure_reason = "clonal_block_member_not_at_target"
    elif centroid_residual > tolerance:
        failure_reason = "clonal_block_centroid_not_at_target"
    elif member_indices.size < int(minimum_cluster_size):
        failure_reason = "clonal_block_below_minimum_size"
    elif np.any(
        observed_support < int(minimum_observed_support_per_region)
    ):
        failure_reason = "clonal_block_insufficient_observed_support"
    else:
        failure_reason = "none"
    return RawClonalBlockCertificate(
        witness_index=witness,
        witness_mutation_id=str(data.mutation_ids[witness]),
        member_indices=member_indices,
        member_mutation_ids=member_ids,
        block_signature=_raw_clonal_block_signature(member_ids),
        target=target_array,
        common_center=common_center,
        centroid=centroid,
        maximum_member_residual=member_residual,
        centroid_residual=centroid_residual,
        cluster_size=int(member_indices.size),
        observed_support_per_region=observed_support,
        equality_tolerance=tolerance,
        certified=failure_reason == "none",
        failure_reason=failure_reason,
    )


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


def _has_cross_close_edge(
    phi: np.ndarray,
    labels: np.ndarray,
    graph: PairwiseFusionGraph,
    *,
    tolerance: float,
    protected_block_mask: np.ndarray | None = None,
) -> bool:
    edge_u = np.asarray(graph.edge_u, dtype=np.int64)
    edge_v = np.asarray(graph.edge_v, dtype=np.int64)
    for start in range(0, edge_u.size, 262_144):
        u = edge_u[start : start + 262_144]
        v = edge_v[start : start + 262_144]
        cross = labels[u] != labels[v]
        if protected_block_mask is not None:
            protected_cross = protected_block_mask[u] != protected_block_mask[v]
            cross &= ~protected_cross
        if not np.any(cross):
            continue
        distances = np.linalg.norm(phi[u[cross]] - phi[v[cross]], axis=1)
        if np.any(distances <= float(tolerance)):
            return True
    return False


def extract_certified_fusion_partition(
    fit: FitResult,
    *,
    graph: PairwiseFusionGraph,
    tolerance: float,
    clonal_block: RawClonalBlockCertificate | None = None,
    mutation_ids: tuple[str, ...] | list[str] | None = None,
) -> FusionPartition:
    """Extract one partition while protecting an exact raw CCF-one block."""

    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition tolerance must be positive and finite.")
    phi = np.asarray(fit.phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("fit.phi must be a mutation-by-region matrix.")

    protected_mask: np.ndarray | None = None
    if clonal_block is None:
        labels = _canonical_partition_labels(
            cluster_labels_from_edges(
                phi,
                edge_u=graph.edge_u,
                edge_v=graph.edge_v,
                tol=tol,
            )
        )
        source = "tolerance_defined_primal"
    else:
        protected_mask = np.zeros(phi.shape[0], dtype=bool)
        protected_mask[np.asarray(clonal_block.member_indices, dtype=np.int64)] = True
        if not np.any(protected_mask):
            raise ValueError("Raw clonal block must contain at least one mutation.")
        labels = np.zeros(phi.shape[0], dtype=np.int64)
        remaining = np.flatnonzero(~protected_mask).astype(np.int64)
        if remaining.size:
            remap = np.full(phi.shape[0], -1, dtype=np.int64)
            remap[remaining] = np.arange(remaining.size, dtype=np.int64)
            edge_u = np.asarray(graph.edge_u, dtype=np.int64)
            edge_v = np.asarray(graph.edge_v, dtype=np.int64)
            keep = (~protected_mask[edge_u]) & (~protected_mask[edge_v])
            remaining_labels = cluster_labels_from_edges(
                phi[remaining],
                edge_u=remap[edge_u[keep]],
                edge_v=remap[edge_v[keep]],
                tol=tol,
            )
            labels[remaining] = np.asarray(remaining_labels, dtype=np.int64) + 1
        labels = _canonical_partition_labels(labels)
        source = "anchor_protected_tolerance_primal"

    if labels.shape != (phi.shape[0],):
        raise ValueError("Partition labels do not match the raw fit mutation count.")
    diameters = _exact_partition_diameters(phi, labels)
    max_diameter = float(np.max(diameters)) if diameters.size else 0.0
    within_ok = bool(np.all(np.isfinite(diameters)) and max_diameter <= tol)
    cross_close = _has_cross_close_edge(
        phi,
        labels,
        graph,
        tolerance=tol,
        protected_block_mask=protected_mask,
    )
    state = getattr(fit, "solver_state", None)
    certificate = getattr(state, "certificate", None)
    certificate_graph_hash_matches = True
    if isinstance(certificate, CompressedEdgeCertificate):
        expected_graph_hash = str(getattr(fit, "original_graph_hash", ""))
        certificate_graph_hash_matches = bool(
            expected_graph_hash
            and str(certificate.graph_hash) == expected_graph_hash
        )
    certified = bool(
        within_ok and not cross_close and certificate_graph_hash_matches
    )
    if not certificate_graph_hash_matches:
        failure_reason = "compressed_certificate_graph_hash_mismatch"
    elif cross_close:
        failure_reason = "cross_block_edge_within_partition_tolerance"
    elif not within_ok:
        failure_reason = "raw_partition_chaining_or_solver_tolerance"
    else:
        failure_reason = "none"
    return FusionPartition(
        labels=labels.astype(np.int64, copy=False),
        signature=_partition_signature(labels, mutation_ids),
        n_clusters=int(np.unique(labels).size),
        tolerance=tol,
        max_diameter=max_diameter,
        diameter_exact=True,
        certified=certified,
        source=source,
        maximal=not cross_close,
        cross_close_edge_found=bool(cross_close),
        certificate_graph_hash_matches=bool(certificate_graph_hash_matches),
        certification_failure_reason=str(failure_reason),
        mutation_ids=() if mutation_ids is None else tuple(mutation_ids),
    )


def _cluster_sizes_text(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return ""
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return ",".join(str(int(value)) for value in counts.tolist())
