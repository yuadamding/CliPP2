from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..io.data import PatientData
from ..metrics.evaluation import SimulationEvaluation, evaluate_fit_against_simulation
from .selection import compute_classic_bic, compute_extended_bic, default_lambda_grid
from .settings import recommend_settings_from_data


@dataclass
class ModelSelectionResult:
    best_fit: FitResult
    best_evaluation: SimulationEvaluation | None
    search_df: pd.DataFrame
    bic_df_scale: float
    bic_cluster_penalty: float
    selection_method: str
    profile_name: str


def _selection_score_value(
    *,
    loglik: float,
    num_clusters: int,
    data: PatientData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
) -> tuple[float, float, float]:
    classic_bic = compute_classic_bic(loglik, num_clusters, data)
    extended_bic = compute_extended_bic(
        loglik,
        num_clusters,
        data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
    )
    normalized = str(selection_score).strip().lower()
    if normalized in {"ebic", "refit_ebic"}:
        return float(extended_bic), float(classic_bic), float(extended_bic)
    if normalized in {"classic_bic", "classic_refit_bic"}:
        return float(classic_bic), float(classic_bic), float(extended_bic)
    raise ValueError(f"Unknown selection_score: {selection_score}")


def _evaluate_candidate(
    *,
    data: PatientData,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    simulation_root: Path | None,
    evaluate_candidate: bool,
    phi_start: np.ndarray | None,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    lambda_value: float,
    selection_score: str,
) -> tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]:
    fit = fit_single_stage_em(
        data=data,
        options=replace(fit_options, lambda_value=float(lambda_value)),
        phi_start=phi_start,
    )
    bic, classic_bic, extended_bic = _selection_score_value(
        loglik=fit.loglik,
        num_clusters=fit.n_clusters,
        data=data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_score=selection_score,
    )
    fit.bic = bic
    fit.classic_bic = classic_bic
    fit.extended_bic = extended_bic
    fit.selection_score_name = selection_score

    evaluation = None
    if evaluate_candidate and simulation_root is not None and (simulation_root / data.patient_id).exists():
        evaluation = evaluate_fit_against_simulation(fit=fit, data=data, simulation_root=simulation_root)

    row: dict[str, float | int | str | bool] = {
        "patient_id": data.patient_id,
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "bic": float(bic),
        "selection_score_name": str(selection_score),
        "classic_bic": float(classic_bic),
        "extended_bic": float(extended_bic),
        "loglik": float(fit.loglik),
        "penalized_objective": float(fit.penalized_objective),
        "n_clusters": int(fit.n_clusters),
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "device": str(fit.device),
        "ARI": np.nan if evaluation is None else float(evaluation.ari),
        "cp_rmse": np.nan if evaluation is None else float(evaluation.cp_rmse),
        "multiplicity_f1": np.nan if evaluation is None else float(evaluation.multiplicity_f1),
        "estimated_clusters": np.nan if evaluation is None else int(evaluation.estimated_clusters),
        "true_clusters": np.nan if evaluation is None else int(evaluation.true_clusters),
        "n_eval_mutations": np.nan if evaluation is None else int(evaluation.n_eval_mutations),
        "n_filtered_mutations": np.nan if evaluation is None else int(evaluation.n_filtered_mutations),
    }
    return fit, evaluation, row


def _grid_search_selection(
    *,
    data: PatientData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    profile_name: str,
    selection_method: str,
    selection_score: str,
) -> ModelSelectionResult:
    if lambda_grid is None:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    lambda_grid = [float(value) for value in np.unique(np.sort(np.asarray(lambda_grid, dtype=float)))]

    phi_start = data.phi_init.copy()
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]] = []
    search_rows: list[dict[str, float | int | str | bool]] = []

    for step, lambda_value in enumerate(lambda_grid):
        fit, evaluation, row = _evaluate_candidate(
            data=data,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            simulation_root=simulation_root,
            evaluate_candidate=evaluate_all_candidates,
            phi_start=phi_start,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=step,
            lambda_value=lambda_value,
            selection_score=selection_score,
        )
        candidate_fits.append((fit, evaluation))
        search_rows.append(row)
        if use_warm_starts:
            phi_start = fit.phi.copy()

    best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    best_fit, best_evaluation = candidate_fits[best_index]
    search_df = pd.DataFrame(search_rows).sort_values(["lambda", "selection_step"]).reset_index(drop=True)
    return ModelSelectionResult(
        best_fit=best_fit,
        best_evaluation=best_evaluation,
        search_df=search_df,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        selection_method=selection_method,
        profile_name=profile_name,
    )


def select_model(
    *,
    data: PatientData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    settings_profile: str,
    selection_score: str,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
) -> ModelSelectionResult:
    normalized_profile = settings_profile.strip().lower()
    if normalized_profile not in {"manual", "auto"}:
        raise ValueError(f"Unknown settings_profile: {settings_profile}")

    effective_lambda_grid_mode = str(lambda_grid_mode)
    effective_bic_df_scale = float(bic_df_scale)
    effective_bic_cluster_penalty = float(bic_cluster_penalty)
    profile_name = "manual"

    if normalized_profile == "auto":
        recommended = recommend_settings_from_data(data, selection_score=selection_score)
        effective_lambda_grid_mode = str(recommended.lambda_grid_mode)
        effective_bic_df_scale = float(recommended.bic_df_scale)
        effective_bic_cluster_penalty = float(recommended.bic_cluster_penalty)
        profile_name = str(recommended.profile_name)

    return _grid_search_selection(
        data=data,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=effective_lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=effective_bic_df_scale,
        bic_cluster_penalty=effective_bic_cluster_penalty,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        profile_name=profile_name,
        selection_method="lambda_path_grid",
        selection_score=selection_score,
    )


__all__ = [
    "ModelSelectionResult",
    "select_model",
]
