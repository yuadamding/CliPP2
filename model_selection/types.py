from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ..core.model import FitResult
from ..metrics.evaluation import SimulationEvaluation

StartArray = np.ndarray | torch.Tensor


@dataclass
class SimulationDiagnostics:
    selected_evaluation: SimulationEvaluation | None = None
    selected_ari: float | None = None
    best_ari: float | None = None
    ari_optimal_lambda_min: float | None = None
    ari_optimal_lambda_max: float | None = None
    ari_optimal_lambda_count: int = 0
    best_converged_ari: float | None = None
    best_converged_lambda_min: float | None = None
    best_converged_lambda_max: float | None = None
    best_converged_lambda_count: int = 0
    ari_hits_lower_boundary: bool = False
    ari_hits_upper_boundary: bool = False
    ari_boundary_unresolved: bool = False
    ari_optimum_resolved: bool = True
    best_ari_all_evaluated: float | None = None
    best_ari_certified: float | None = None


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
    lambda_bracket_min: float | None
    lambda_bracket_eq: float | None
    lambda_bracket_full: float | None
    adaptive_refinement_rounds_completed: int
    simulation: SimulationDiagnostics = field(default_factory=SimulationDiagnostics)

    @property
    def best_evaluation(self) -> SimulationEvaluation | None:
        return self.simulation.selected_evaluation

    @property
    def selected_ari(self) -> float | None:
        return self.simulation.selected_ari

    @property
    def best_ari(self) -> float | None:
        return self.simulation.best_ari

    @property
    def ari_optimal_lambda_min(self) -> float | None:
        return self.simulation.ari_optimal_lambda_min

    @property
    def ari_optimal_lambda_max(self) -> float | None:
        return self.simulation.ari_optimal_lambda_max

    @property
    def ari_optimal_lambda_count(self) -> int:
        return self.simulation.ari_optimal_lambda_count

    @property
    def best_converged_ari(self) -> float | None:
        return self.simulation.best_converged_ari

    @property
    def best_converged_lambda_min(self) -> float | None:
        return self.simulation.best_converged_lambda_min

    @property
    def best_converged_lambda_max(self) -> float | None:
        return self.simulation.best_converged_lambda_max

    @property
    def best_converged_lambda_count(self) -> int:
        return self.simulation.best_converged_lambda_count

    @property
    def ari_hits_lower_boundary(self) -> bool:
        return self.simulation.ari_hits_lower_boundary

    @property
    def ari_hits_upper_boundary(self) -> bool:
        return self.simulation.ari_hits_upper_boundary

    @property
    def ari_boundary_unresolved(self) -> bool:
        return self.simulation.ari_boundary_unresolved

    @property
    def ari_optimum_resolved(self) -> bool:
        return self.simulation.ari_optimum_resolved

    @property
    def best_ari_all_evaluated(self) -> float | None:
        return self.simulation.best_ari_all_evaluated

    @property
    def best_ari_certified(self) -> float | None:
        return self.simulation.best_ari_certified


ModelSelectionResult = BICSelectionResult


@dataclass(frozen=True)
class _AdaptiveIntervalProposal:
    lambda_value: float
    left_lambda: float
    right_lambda: float
    left_candidate_id: int | None
    right_candidate_id: int | None
    priority_key: tuple[int, float, float, float, float]
    reason: str
    log_width: float
    partition_changed: bool
    nonagglomerative_or_numerically_inconsistent: bool


@dataclass(frozen=True)
class FullFusionKKTResult:
    residual: float
    iterations: int
    converged: bool
    lambda_value: float


@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    pilot_matrix_hash: str
    input_data_hash: str

