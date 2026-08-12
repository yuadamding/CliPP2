from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch

from ..core.model import FitResult

StartArray = np.ndarray | torch.Tensor


def _immutable_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FusionPartition:
    labels: np.ndarray
    signature: str
    n_clusters: int
    tolerance: float
    max_diameter: float
    diameter_exact: bool
    certified: bool
    source: Literal[
        "solver_quotient",
        "verified_primal_equalities",
        "tolerance_defined_primal",
        "anchor_protected_tolerance_primal",
    ]
    maximal: bool = False
    cross_close_edge_found: bool = False
    certificate_graph_hash_matches: bool = True
    certification_failure_reason: str = "none"
    mutation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "labels",
            _immutable_array(self.labels, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "mutation_ids",
            tuple(str(value) for value in self.mutation_ids),
        )


@dataclass(frozen=True)
class RawClonalBlockCertificate:
    """Exact raw CCF-one block witnessed by one constrained mutation.

    The witness identifies a seed-conditioned optimization branch.  The
    biological model object is ``member_indices`` and its common CCF-one
    profile, not the witness itself.
    """

    witness_index: int
    witness_mutation_id: str
    member_indices: np.ndarray
    member_mutation_ids: tuple[str, ...]
    block_signature: str
    target: np.ndarray
    common_center: np.ndarray
    centroid: np.ndarray
    maximum_member_residual: float
    centroid_residual: float
    equality_tolerance: float
    mathematically_certified: bool
    failure_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "member_indices",
            _immutable_array(self.member_indices, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "member_mutation_ids",
            tuple(str(value) for value in self.member_mutation_ids),
        )
        for name in ("target", "common_center", "centroid"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), dtype=np.dtype(np.float64)),
            )
    @property
    def cluster_size(self) -> int:
        return int(self.member_indices.size)

    @property
    def certified(self) -> bool:
        """Compatibility alias for mathematical certification only."""

        return bool(self.mathematically_certified)


@dataclass(frozen=True)
class RawClonalBlockEvidence:
    """Biological support diagnostics, separate from model feasibility."""

    block_signature: str
    cluster_size: int
    observed_support_per_region: np.ndarray
    total_depth_per_region: np.ndarray
    median_depth_per_region: np.ndarray
    minimum_cluster_size: int
    minimum_observed_support_per_region: int
    evidence_gate_passed: bool
    evidence_failure_reason: str

    def __post_init__(self) -> None:
        for name, dtype in (
            ("observed_support_per_region", np.dtype(np.int64)),
            ("total_depth_per_region", np.dtype(np.float64)),
            ("median_depth_per_region", np.dtype(np.float64)),
        ):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), dtype=dtype),
            )


@dataclass(frozen=True)
class PartitionRefitSummary:
    labels: np.ndarray
    partition_signature: str
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    fit_loss: float
    nominal_df: int
    active_df: int
    anchor_mode: Literal["none", "clonal_required"]
    clonal_cluster: int | None
    anchor_deviance_increase: float
    second_best_anchor_deviance_increase: float
    finite_candidate_found: bool
    global_optimum_certified: bool
    loglik_source: str
    refit_numerically_resolved: bool = False
    refit_loglik_refinement_delta: float = float("inf")
    refit_max_center_refinement_delta: float = float("inf")
    refit_coordinate_count: int = 0
    refit_finite_coordinate_count: int = 0
    refit_total_grid_points: int = 0
    refit_max_grid_spacing: float = float("inf")
    refit_total_candidate_basins: int = 0
    refit_total_refined_candidates: int = 0
    refit_min_best_second_loss_gap: float = float("inf")
    fixed_anchor_target: np.ndarray | None = None
    anchor_block_signature: str = "none"
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0

    def __post_init__(self) -> None:
        if self.global_optimum_certified and (
            not np.isfinite(float(self.global_lower_bound))
            or not np.isfinite(float(self.global_optimality_gap))
            or float(self.global_optimality_gap) < 0.0
            or str(self.global_certificate_method) == "none"
        ):
            raise ValueError("A global refit claim requires a finite certificate.")
        object.__setattr__(
            self,
            "labels",
            _immutable_array(self.labels, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "phi",
            _immutable_array(self.phi, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "cluster_centers",
            _immutable_array(self.cluster_centers, dtype=np.dtype(np.float64)),
        )
        if self.fixed_anchor_target is not None:
            object.__setattr__(
                self,
                "fixed_anchor_target",
                _immutable_array(
                    self.fixed_anchor_target, dtype=np.dtype(np.float64)
                ),
            )


@dataclass(frozen=True)
class SelectionScore:
    name: Literal["fixed_partition_bic", "clonal_fixed_partition_bic"]
    value: float
    loglik: float
    penalty: float
    degrees_of_freedom: int
    n_eff: int
    partition_signature: str
    anchor_block_signature: str = "none"
    numerical_uncertainty: float = 0.0

    @property
    def lower_bound(self) -> float:
        return float(self.value - self.numerical_uncertainty)

    @property
    def upper_bound(self) -> float:
        return float(self.value + self.numerical_uncertainty)


@dataclass(frozen=True)
class RawFusionCandidate:
    raw_fit: FitResult
    partition: FusionPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    raw_objective_certified: bool
    eligible_for_selection: bool
    ineligibility_reason: str
    anchor_seed_index: int | None = None
    anchor_seed_mutation_id: str = "none"
    anchor_cluster_label: int | None = None
    anchor_block_signature: str = "none"
    anchor_target: np.ndarray | None = None
    anchor_search_complete: bool = False
    clonal_block: RawClonalBlockCertificate | None = None
    clonal_block_evidence: RawClonalBlockEvidence | None = None

    def __post_init__(self) -> None:
        if self.anchor_target is not None:
            object.__setattr__(
                self,
                "anchor_target",
                _immutable_array(self.anchor_target, dtype=np.dtype(np.float64)),
            )


@dataclass(frozen=True)
class SelectedModel:
    candidate: RawFusionCandidate
    selected_lambda: float
    selected_partition_signature: str
    selected_partition_left_lambda: float | None
    selected_partition_right_lambda: float | None

    def __post_init__(self) -> None:
        candidate = self.candidate
        if self.selected_partition_signature != self.candidate.partition.signature:
            raise ValueError("Selected-model partition signature is inconsistent.")
        if not np.isclose(
            float(self.selected_lambda),
            float(self.candidate.raw_fit.lambda_value),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Selected-model lambda is inconsistent with its raw fit.")
        if not candidate.eligible_for_selection:
            raise ValueError("Selected model must be eligible for selection.")
        if not candidate.raw_objective_certified:
            raise ValueError("Selected model must have a certified raw objective.")
        if not candidate.partition.certified:
            raise ValueError("Selected model must have a certified partition.")
        if not candidate.refit.global_optimum_certified:
            raise ValueError("Selected model must have a globally certified refit.")
        if candidate.score.partition_signature != self.selected_partition_signature:
            raise ValueError("Selected-model score signature is inconsistent.")
        if candidate.score.name == "clonal_fixed_partition_bic":
            clonal_block = candidate.clonal_block
            if clonal_block is None or not clonal_block.certified:
                raise ValueError(
                    "Selected clonal model must have a certified raw CCF-one block."
                )
            if not candidate.anchor_search_complete:
                raise ValueError(
                    "Selected clonal model requires exact witness-search resolution."
                )
            if candidate.anchor_cluster_label is None:
                raise ValueError("Selected clonal model is missing its cluster label.")
            raw_anchor_cluster = int(candidate.anchor_cluster_label)
            if candidate.refit.clonal_cluster != raw_anchor_cluster:
                raise ValueError(
                    "Selected refit must preserve the exact raw clonal cluster."
                )
            if candidate.anchor_block_signature != candidate.score.anchor_block_signature:
                raise ValueError("Selected anchor-block score identity is inconsistent.")
            if candidate.anchor_block_signature != clonal_block.block_signature:
                raise ValueError("Selected clonal-block identity is inconsistent.")
            if candidate.refit.anchor_block_signature != clonal_block.block_signature:
                raise ValueError("Selected refit clonal-block identity is inconsistent.")


@dataclass
class BICSelectionResult:
    selected_model: SelectedModel
    search_df: pd.DataFrame
    bic_df_scale: float
    bic_cluster_penalty: float
    selection_method: str
    profile_name: str
    selection_metric_value: float | None
    selection_lambda_min: float | None
    selection_lambda_max: float | None
    selection_lambda_count: int
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool
    selection_optimum_resolved: bool
    adaptive_search_rounds_completed: int
    adaptive_search_stop_reason: str
    num_candidates: int
    num_converged_candidates: int
    num_candidates_all: int
    num_candidates_certified: int
    selected_kkt_residual: float | None
    best_score_all_evaluated_lambda: float | None
    best_score_all_evaluated_kkt_residual: float | None
    best_score_all_evaluated_selection_eligible: bool
    best_score_certified_lambda: float | None
    best_score_certified_kkt_residual: float | None
    selection_optimizer_limited: bool
    selection_optimizer_limited_reason: str
    selection_used_convergence_fallback: bool
    lambda_search_mode: str
    selected_lambda_representative: float | None
    selected_lambda_left: float | None
    selected_lambda_right: float | None
    selected_lambda_interval_log10_width: float | None
    adaptive_refinement_rounds_completed: int

    @property
    def best_fit(self) -> FitResult:
        return self.selected_model.candidate.raw_fit


@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    pilot_matrix_hash: str
    input_data_hash: str
