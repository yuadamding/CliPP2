from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..core.model import FitResult
StartArray = np.ndarray | torch.Tensor


@dataclass
class SelectionArtifact:
    bic: float | None = None
    classic_bic: float | None = None
    extended_bic: float | None = None
    partition_icl: float | None = None
    partition_log_evidence: float | None = None
    partition_code_deviance: float | None = None
    partition_dirichlet_alpha: float | None = None
    classic_bic_depth_n: float | None = None
    classic_bic_active_df: float | None = None
    classic_bic_active_df_depth_n: float | None = None
    bic_loglik: float | None = None
    bic_loglik_source: str | None = None
    bic_df: float | None = None
    bic_active_df: float | None = None
    bic_n_eff: float | None = None
    bic_depth_n_eff: float | None = None
    bic_partition_tol: float | None = None
    bic_refit_boundary_count: int | None = None
    bic_refit_finite_candidate_found: bool | None = None
    bic_refit_global_optimum_certified: bool | None = None
    bic_refit_coordinate_count: int | None = None
    bic_refit_finite_coordinate_count: int | None = None
    bic_refit_total_grid_points: int | None = None
    bic_refit_max_grid_spacing: float | None = None
    bic_refit_total_candidate_basins: int | None = None
    bic_refit_total_refined_candidates: int | None = None
    bic_refit_min_best_second_loss_gap: float | None = None
    bic_refit_converged: bool | None = None
    bic_refit_phi: np.ndarray | None = None
    bic_refit_cluster_centers: np.ndarray | None = None
    bic_partition_labels: np.ndarray | None = None
    selection_score_name: str | None = None


@dataclass
class BICSelectionResult:
    best_fit: FitResult
    selected_artifact: SelectionArtifact
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
@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    pilot_matrix_hash: str
    input_data_hash: str
