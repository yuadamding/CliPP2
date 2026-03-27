from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from ..io.data import PatientData, load_patient_tsv
from ..metrics.evaluation import evaluate_fit_against_simulation
from .model_selection import select_model
from .outputs import write_fit_outputs
from .settings import summarize_patient_regime


def process_one_file(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    graph_k: int = 8,
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    bo_max_evals: int = 12,
    bo_init_points: int = 5,
    bo_random_seed: int = 0,
) -> dict[str, float | int | str | bool]:
    start_time = perf_counter()
    file_path = Path(file_path)
    outdir = Path(outdir)
    data = load_patient_tsv(file_path)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)

    patient_regime = summarize_patient_regime(data)
    evaluate_all_candidates = write_outputs and simulation_root is not None and (Path(simulation_root) / data.patient_id).exists()
    selection_result = select_model(
        data=data,
        simulation_root=Path(simulation_root) if simulation_root is not None else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        graph_k=graph_k,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        settings_profile=settings_profile,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        bo_max_evals=bo_max_evals,
        bo_init_points=bo_init_points,
        bo_random_seed=bo_random_seed,
    )
    best_fit = selection_result.best_fit
    best_evaluation = selection_result.best_evaluation
    search_df = selection_result.search_df

    if best_evaluation is None and simulation_root is not None and (Path(simulation_root) / data.patient_id).exists():
        best_evaluation = evaluate_fit_against_simulation(fit=best_fit, data=data, simulation_root=simulation_root)

    elapsed_seconds = float(perf_counter() - start_time)

    summary = {
        "patient_id": data.patient_id,
        "selected_lambda": float(best_fit.lambda_value),
        "bic": float(best_fit.bic if best_fit.bic is not None else np.nan),
        "loglik": float(best_fit.loglik),
        "n_clusters": int(best_fit.n_clusters),
        "graph_edges": int(selection_result.graph_edges),
        "settings_profile": selection_result.profile_name,
        "selection_method": selection_result.selection_method,
        "num_samples": int(patient_regime.num_samples),
        "num_mutations": int(patient_regime.num_mutations),
        "depth_scale": float(patient_regime.depth_scale),
        "mean_purity": float(patient_regime.mean_purity),
        "non_diploid_rate": float(patient_regime.non_diploid_rate),
        "graph_k": int(selection_result.graph_k),
        "lambda_grid_mode": str(lambda_grid_mode if lambda_grid is None else "explicit"),
        "bic_df_scale": float(selection_result.bic_df_scale),
        "bic_cluster_penalty": float(selection_result.bic_cluster_penalty),
        "center_merge_tol": float(selection_result.center_merge_tol),
        "converged": bool(best_fit.converged),
        "device": best_fit.device,
        "ARI": np.nan if best_evaluation is None else float(best_evaluation.ari),
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
    return summary


def run_directory(
    input_dir: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    graph_k: int = 8,
    fit_options: FitOptions | None = None,
    max_files: int | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    bo_max_evals: int = 12,
    bo_init_points: int = 5,
    bo_random_seed: int = 0,
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
                graph_k=graph_k,
                fit_options=fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                settings_profile=settings_profile,
                use_warm_starts=use_warm_starts,
                write_outputs=write_outputs,
                bo_max_evals=bo_max_evals,
                bo_init_points=bo_init_points,
                bo_random_seed=bo_random_seed,
            )
        )

    summary_df = pd.DataFrame(summaries).sort_values("patient_id").reset_index(drop=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(outdir) / "single_stage_summary.tsv", sep="\t", index=False)
    return summary_df
