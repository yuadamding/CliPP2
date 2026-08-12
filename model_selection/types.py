from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

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
        "legacy_connected_components",
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
class PartitionRefitSummary:
    labels: np.ndarray
    partition_signature: str
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    fit_loss: float
    nominal_df: int
    active_df: int
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
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0
    refit_mode: str = "interval_certified"

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


@dataclass(frozen=True)
class SelectionScore:
    name: Literal[
        "fixed_partition_bic",
        "fixed_partition_dirichlet_score",
    ]
    value: float
    loglik: float
    penalty: float
    degrees_of_freedom: int
    n_eff: int
    partition_signature: str
    numerical_uncertainty: float = 0.0
    assignment_log_evidence: float = 0.0
    assignment_code_weight: float = 0.0
    assignment_penalty: float = 0.0
    assignment_dirichlet_alpha: float = 1.0
    assignment_model_id: str = "none"
    assignment_symmetry_mode: str = "none"
    assignment_arithmetic_uncertainty: float = 0.0
    selection_contract_id: str = "raw-fusion-only-v0.3"

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
    computation_profile: str = "strict"


@dataclass(frozen=True)
class DirectPartition:
    labels: np.ndarray
    signature: str
    n_clusters: int
    source: Literal[
        "pilot_hessian_ward",
        "pilot_hessian_ward_cem",
        "pilot_hessian_ward_cem_component_death",
        "final_phi_hessian_ward",
        "final_phi_hessian_ward_cem",
        "final_phi_hessian_ward_cem_component_death",
        "local_split",
        "local_merge",
    ]
    requested_k: int
    mutation_ids: tuple[str, ...]
    generation_contract_id: str
    parent_raw_candidate_id: int | None = None
    parent_raw_lambda: float | None = None
    parent_raw_phi_hash: str = ""
    pre_refinement_signature: str = ""
    cem_iterations: int = 0
    component_death_count: int = 0
    refinement_score_before: float = float("nan")
    refinement_score_after: float = float("nan")
    deterministic_generation: bool = True

    def __post_init__(self) -> None:
        labels = _immutable_array(self.labels, dtype=np.dtype(np.int64))
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("Direct partition labels must be nonempty and 1-D.")
        unique = np.unique(labels)
        expected = np.arange(unique.size, dtype=np.int64)
        if not np.array_equal(unique, expected):
            raise ValueError("Direct partition labels must be canonical zero-based IDs.")
        if int(self.n_clusters) != int(unique.size):
            raise ValueError("Direct partition cluster count is inconsistent.")
        mutation_ids = tuple(str(value) for value in self.mutation_ids)
        if len(mutation_ids) != labels.size or len(set(mutation_ids)) != labels.size:
            raise ValueError(
                "Direct partition mutation IDs must uniquely identify every label."
            )
        if int(self.requested_k) < 1:
            raise ValueError("Direct partition requested_k must be positive.")
        if not str(self.signature) or not str(self.generation_contract_id):
            raise ValueError("Direct partition identity provenance is required.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "mutation_ids", mutation_ids)


@dataclass(frozen=True)
class DirectPartitionCandidate:
    partition: DirectPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    eligible_for_selection: bool
    ineligibility_reason: str
    computation_profile: str
    candidate_family: str = "direct_partition"


SelectablePartitionCandidate = Union[RawFusionCandidate, DirectPartitionCandidate]


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: int
    candidate: SelectablePartitionCandidate
    row: dict[str, object]

    @property
    def score(self) -> SelectionScore:
        return self.candidate.score

    @property
    def partition_signature(self) -> str:
        return self.candidate.partition.signature

    @property
    def family(self) -> str:
        return (
            "raw_fusion"
            if isinstance(self.candidate, RawFusionCandidate)
            else "direct_partition"
        )


@dataclass(frozen=True)
class SelectedModel:
    raw_reference: RawFusionCandidate
    partition_candidate: SelectablePartitionCandidate
    selected_partition_signature: str
    selected_candidate_family: str
    selected_lambda: float | None
    selected_partition_left_lambda: float | None
    selected_partition_right_lambda: float | None

    def __post_init__(self) -> None:
        raw = self.raw_reference
        candidate = self.partition_candidate
        strict = str(candidate.computation_profile) == "strict"
        if self.selected_partition_signature != candidate.partition.signature:
            raise ValueError("Selected-model partition signature is inconsistent.")
        expected_family = (
            "raw_fusion" if isinstance(candidate, RawFusionCandidate) else "direct_partition"
        )
        if str(self.selected_candidate_family) != expected_family:
            raise ValueError("Selected-model candidate family is inconsistent.")
        if not candidate.eligible_for_selection:
            raise ValueError("Selected model must be eligible for selection.")
        if isinstance(candidate, RawFusionCandidate):
            if self.selected_lambda is None or not np.isclose(
                float(self.selected_lambda),
                float(candidate.raw_fit.lambda_value),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Selected-model lambda is inconsistent with its raw fit."
                )
            if not candidate.raw_objective_certified:
                raise ValueError("Selected raw model must have a certified objective.")
            if not candidate.partition.certified:
                raise ValueError("Selected raw model must have a certified partition.")
        elif self.selected_lambda is not None:
            raise ValueError("Direct partition candidates do not have a selected lambda.")
        if strict and not candidate.refit.global_optimum_certified:
            raise ValueError("Selected model must have a globally certified refit.")
        if candidate.score.partition_signature != self.selected_partition_signature:
            raise ValueError("Selected-model score signature is inconsistent.")
        if (
            not raw.eligible_for_selection
            or not raw.raw_objective_certified
            or not raw.partition.certified
        ):
            raise ValueError("Raw reference must remain a certified raw-fusion model.")

    @property
    def candidate(self) -> SelectablePartitionCandidate:
        """Compatibility alias for the selected partition candidate."""

        return self.partition_candidate

    @property
    def partition(self) -> FusionPartition | DirectPartition:
        return self.partition_candidate.partition

    @property
    def refit(self) -> PartitionRefitSummary:
        return self.partition_candidate.refit

    @property
    def score(self) -> SelectionScore:
        return self.partition_candidate.score

    @property
    def raw_fit(self) -> FitResult:
        return self.raw_reference.raw_fit


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
    ward_candidate_pool_complete: bool = False
    raw_lambda_path_resolved: bool = False
    best_over_evaluated_candidates: bool = True
    global_hybrid_optimum_certified: bool = False

    @property
    def best_fit(self) -> FitResult:
        return self.selected_model.raw_fit


@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    pilot_matrix_hash: str
    input_data_hash: str
