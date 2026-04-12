from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from ..core.fusion_solver import load_pairwise_fusion_graph_tsv
from ..io.data import TumorData, load_tumor_tsv
from ..metrics.evaluation import evaluate_fit_against_simulation
from .model_selection import select_model
from .outputs import write_fit_outputs
from .settings import summarize_tumor_regime


def process_one_file_bundle(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    start_time = perf_counter()
    file_path = Path(file_path)
    outdir = Path(outdir)
    data = load_tumor_tsv(file_path)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)
    if graph_file is not None:
        fit_options = replace(
            fit_options,
            graph=load_pairwise_fusion_graph_tsv(
                graph_file,
                num_mutations=data.num_mutations,
                mutation_ids=data.mutation_ids,
            ),
        )

    tumor_regime = summarize_tumor_regime(data)
    simulation_available = simulation_root is not None and (Path(simulation_root) / data.tumor_id).exists()
    if finalize_selected_fit is None:
        finalize_selected_fit = bool(write_outputs or str(selection_score).strip().lower() != "oracle_ari")
    evaluate_all_candidates = simulation_available and (
        write_outputs or str(selection_score).strip().lower() == "oracle_ari"
    )
    selection_result = select_model(
        data=data,
        simulation_root=Path(simulation_root) if simulation_root is not None else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        settings_profile=settings_profile,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        finalize_selected_fit=bool(finalize_selected_fit),
    )
    best_fit = selection_result.best_fit
    best_evaluation = selection_result.best_evaluation
    search_df = selection_result.search_df

    if best_evaluation is None and simulation_available and bool(finalize_selected_fit):
        best_evaluation = evaluate_fit_against_simulation(fit=best_fit, data=data, simulation_root=simulation_root)

    elapsed_seconds = float(perf_counter() - start_time)

    summary = {
        "tumor_id": data.tumor_id,
        "estimator": "objective_faithful_pairwise_fusion",
        "selected_lambda": float(best_fit.lambda_value),
        "loglik": float(best_fit.loglik),
        "summary_loglik": float(best_fit.summary_loglik),
        "bic": float(best_fit.bic if best_fit.bic is not None else np.nan),
        "classic_bic": float(best_fit.classic_bic if best_fit.classic_bic is not None else np.nan),
        "extended_bic": float(best_fit.extended_bic if best_fit.extended_bic is not None else np.nan),
        "n_clusters": int(best_fit.n_clusters),
        "settings_profile": selection_result.profile_name,
        "selection_method": selection_result.selection_method,
        "selection_score_name": str(best_fit.selection_score_name or selection_score),
        "selection_loglik_kind": (
            "summary_clustered"
            if str(best_fit.selection_score_name or selection_score) != "oracle_ari"
            else "oracle_ari_finalized_selected_fit"
            if bool(finalize_selected_fit)
            else "oracle_ari_candidate_fit"
        ),
        "finalize_selected_fit": bool(finalize_selected_fit),
        "selection_used_convergence_fallback": bool(selection_result.selection_used_convergence_fallback),
        "num_candidates": int(selection_result.num_candidates),
        "num_converged_candidates": int(selection_result.num_converged_candidates),
        "selection_metric_value": np.nan
        if selection_result.selection_metric_value is None
        else float(selection_result.selection_metric_value),
        "selection_lambda_min": np.nan
        if selection_result.selection_lambda_min is None
        else float(selection_result.selection_lambda_min),
        "selection_lambda_max": np.nan
        if selection_result.selection_lambda_max is None
        else float(selection_result.selection_lambda_max),
        "selection_lambda_count": int(selection_result.selection_lambda_count),
        "selection_hits_lower_boundary": bool(selection_result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(selection_result.selection_hits_upper_boundary),
        "selection_boundary_unresolved": bool(selection_result.selection_boundary_unresolved),
        "selection_optimum_resolved": bool(selection_result.selection_optimum_resolved),
        "best_ari": np.nan if selection_result.best_ari is None else float(selection_result.best_ari),
        "best_converged_ari": np.nan
        if selection_result.best_converged_ari is None
        else float(selection_result.best_converged_ari),
        "best_converged_lambda_min": np.nan
        if selection_result.best_converged_lambda_min is None
        else float(selection_result.best_converged_lambda_min),
        "best_converged_lambda_max": np.nan
        if selection_result.best_converged_lambda_max is None
        else float(selection_result.best_converged_lambda_max),
        "best_converged_lambda_count": int(selection_result.best_converged_lambda_count),
        "ari_optimal_lambda_min": np.nan
        if selection_result.ari_optimal_lambda_min is None
        else float(selection_result.ari_optimal_lambda_min),
        "ari_optimal_lambda_max": np.nan
        if selection_result.ari_optimal_lambda_max is None
        else float(selection_result.ari_optimal_lambda_max),
        "ari_optimal_lambda_count": int(selection_result.ari_optimal_lambda_count),
        "ari_hits_lower_boundary": bool(selection_result.ari_hits_lower_boundary),
        "ari_hits_upper_boundary": bool(selection_result.ari_hits_upper_boundary),
        "ari_boundary_unresolved": bool(selection_result.ari_boundary_unresolved),
        "ari_optimum_resolved": bool(selection_result.ari_optimum_resolved),
        "oracle_search_rounds_completed": int(selection_result.oracle_search_rounds_completed),
        "oracle_search_stop_reason": str(selection_result.oracle_search_stop_reason),
        "tested_lambda_min": float(search_df["lambda"].min()) if not search_df.empty else np.nan,
        "tested_lambda_max": float(search_df["lambda"].max()) if not search_df.empty else np.nan,
        "tested_lambda_count": int(search_df["lambda"].nunique()) if not search_df.empty else 0,
        "num_regions": int(tumor_regime.num_regions),
        "num_mutations": int(tumor_regime.num_mutations),
        "depth_scale": float(tumor_regime.depth_scale),
        "mean_purity": float(tumor_regime.mean_purity),
        "non_diploid_rate": float(tumor_regime.non_diploid_rate),
        "lambda_grid_mode": str(lambda_grid_mode if lambda_grid is None else "explicit"),
        "bic_df_scale": float(selection_result.bic_df_scale),
        "bic_cluster_penalty": float(selection_result.bic_cluster_penalty),
        "converged": bool(best_fit.converged),
        "device": best_fit.device,
        "graph_name": str(best_fit.graph_name),
        "ARI": (
            float(best_evaluation.ari)
            if best_evaluation is not None
            else float(selection_result.best_ari)
            if selection_result.best_ari is not None
            else np.nan
        ),
        "cp_rmse": np.nan if best_evaluation is None else float(best_evaluation.cp_rmse),
        "multiplicity_f1": np.nan if best_evaluation is None else float(best_evaluation.multiplicity_f1),
        "estimated_clonal_fraction": np.nan
        if best_evaluation is None
        else float(best_evaluation.estimated_clonal_fraction),
        "true_clonal_fraction": np.nan
        if best_evaluation is None
        else float(best_evaluation.true_clonal_fraction),
        "clonal_fraction_error": np.nan
        if best_evaluation is None
        else float(best_evaluation.clonal_fraction_error),
        "n_eval_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_eval_mutations),
        "n_filtered_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_filtered_mutations),
        "elapsed_seconds": elapsed_seconds,
    }

    if write_outputs:
        write_fit_outputs(
            outdir=outdir,
            data=data,
            fit=best_fit,
            search_df=search_df,
            evaluation=best_evaluation,
            run_summary=summary,
        )
    return summary, search_df


def process_one_file(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
) -> dict[str, float | int | str | bool]:
    summary, _ = process_one_file_bundle(
        file_path=file_path,
        outdir=outdir,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        settings_profile=settings_profile,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        graph_file=graph_file,
        finalize_selected_fit=finalize_selected_fit,
    )
    return summary


def run_directory(
    input_dir: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    max_files: int | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.tsv"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]

    if not files:
        raise RuntimeError(f"No TSV files found in {input_dir}")

    summaries = []
    for file_path in files:
        summaries.append(
            process_one_file(
                file_path=file_path,
                outdir=outdir,
                simulation_root=simulation_root,
                lambda_grid=lambda_grid,
                lambda_grid_mode=lambda_grid_mode,
                fit_options=fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                settings_profile=settings_profile,
                selection_score=selection_score,
                use_warm_starts=use_warm_starts,
                write_outputs=write_outputs,
                graph_file=graph_file,
                finalize_selected_fit=finalize_selected_fit,
            )
        )

    summary_df = pd.DataFrame(summaries)
    sort_column = "tumor_id" if "tumor_id" in summary_df.columns else "patient_id"
    summary_df = summary_df.sort_values(sort_column).reset_index(drop=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(outdir) / "single_stage_summary.tsv", sep="\t", index=False)
    return summary_df
