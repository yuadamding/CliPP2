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
    ]
    maximal: bool = False
    cross_close_edge_found: bool = False
    certificate_graph_hash_matches: bool = True
    certification_failure_reason: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "labels",
            _immutable_array(self.labels, dtype=np.dtype(np.int64)),
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

    def __post_init__(self) -> None:
        if self.global_optimum_certified:
            raise ValueError(
                "Partition refits cannot claim global optimality without a "
                "separate certificate object."
            )
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
    name: Literal["fixed_partition_bic", "clonal_fixed_partition_bic"]
    value: float
    loglik: float
    penalty: float
    degrees_of_freedom: int
    n_eff: int
    partition_signature: str


@dataclass(frozen=True)
class RawFusionCandidate:
    raw_fit: FitResult
    partition: FusionPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    raw_objective_certified: bool
    eligible_for_selection: bool
    ineligibility_reason: str


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
        if candidate.score.partition_signature != self.selected_partition_signature:
            raise ValueError("Selected-model score signature is inconsistent.")


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
