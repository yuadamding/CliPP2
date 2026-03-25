from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.graph import build_knn_graph
from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..io.data import PatientData, load_patient_tsv
from ..metrics.evaluation import SimulationEvaluation, evaluate_fit_against_simulation
from .outputs import write_fit_outputs
from .selection import compute_classic_bic, compute_extended_bic, default_lambda_grid
from .settings import recommend_settings_from_data, summarize_patient_regime


def process_one_file(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    graph_k: int = 8,
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 10.0,
    bic_cluster_penalty: float = 6.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
) -> dict[str, float | int | str | bool]:
    start_time = perf_counter()
    file_path = Path(file_path)
    outdir = Path(outdir)
    data = load_patient_tsv(file_path)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)

    profile_name = "manual"
    patient_regime = summarize_patient_regime(data)
    if settings_profile == "auto":
        recommended = recommend_settings_from_data(data)
        profile_name = recommended.profile_name
        graph_k = recommended.graph_k
        lambda_grid_mode = recommended.lambda_grid_mode
        bic_df_scale = recommended.bic_df_scale
        bic_cluster_penalty = recommended.bic_cluster_penalty
        fit_options = replace(
            fit_options,
            center_merge_tol=recommended.center_merge_tol,
        )
    elif settings_profile != "manual":
        raise ValueError(f"Unknown settings_profile: {settings_profile}")

    graph = build_knn_graph(data.phi_init, k=graph_k)

    if lambda_grid is None:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    lambda_grid = [float(value) for value in np.unique(np.sort(np.asarray(lambda_grid, dtype=float)))]

    search_rows: list[dict[str, float | int | str | bool]] = []
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]] = []
    phi_start = data.phi_init.copy()

    evaluate_all_candidates = write_outputs and simulation_root is not None and (Path(simulation_root) / data.patient_id).exists()

    for lambda_value in lambda_grid:
        options = replace(fit_options, lambda_value=float(lambda_value))
        fit = fit_single_stage_em(data=data, graph=graph, options=options, phi_start=phi_start)
        if use_warm_starts:
            phi_start = fit.phi.copy()
        classic_bic = compute_classic_bic(fit.loglik, fit.n_clusters, data)
        fit.bic = compute_extended_bic(
            fit.loglik,
            fit.n_clusters,
            data,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
        )

        evaluation = None
        if evaluate_all_candidates:
            evaluation = evaluate_fit_against_simulation(fit=fit, data=data, simulation_root=simulation_root)

        candidate_fits.append((fit, evaluation))
        search_rows.append(
            {
                "patient_id": data.patient_id,
                "lambda": fit.lambda_value,
                "bic": fit.bic,
                "classic_bic": classic_bic,
                "loglik": fit.loglik,
                "penalized_objective": fit.penalized_objective,
                "n_clusters": fit.n_clusters,
                "converged": fit.converged,
                "iterations": fit.iterations,
                "device": fit.device,
                "ARI": np.nan if evaluation is None else evaluation.ari,
                "cp_rmse": np.nan if evaluation is None else evaluation.cp_rmse,
                "multiplicity_accuracy": np.nan if evaluation is None else evaluation.multiplicity_accuracy,
                "n_eval_mutations": np.nan if evaluation is None else evaluation.n_eval_mutations,
                "n_filtered_mutations": np.nan if evaluation is None else evaluation.n_filtered_mutations,
            }
        )

    best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    best_fit, best_evaluation = candidate_fits[best_index]
    search_df = pd.DataFrame(search_rows).sort_values("lambda").reset_index(drop=True)

    if best_evaluation is None and simulation_root is not None and (Path(simulation_root) / data.patient_id).exists():
        best_evaluation = evaluate_fit_against_simulation(fit=best_fit, data=data, simulation_root=simulation_root)

    if write_outputs:
        write_fit_outputs(outdir=outdir, data=data, fit=best_fit, search_df=search_df, evaluation=best_evaluation)

    summary = {
        "patient_id": data.patient_id,
        "selected_lambda": float(best_fit.lambda_value),
        "bic": float(best_fit.bic if best_fit.bic is not None else np.nan),
        "loglik": float(best_fit.loglik),
        "n_clusters": int(best_fit.n_clusters),
        "graph_edges": int(graph.num_edges),
        "settings_profile": profile_name,
        "num_samples": int(patient_regime.num_samples),
        "num_mutations": int(patient_regime.num_mutations),
        "depth_scale": float(patient_regime.depth_scale),
        "mean_purity": float(patient_regime.mean_purity),
        "non_diploid_rate": float(patient_regime.non_diploid_rate),
        "graph_k": int(graph_k),
        "lambda_grid_mode": str(lambda_grid_mode),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "center_merge_tol": float(fit_options.center_merge_tol),
        "converged": bool(best_fit.converged),
        "device": best_fit.device,
        "ARI": np.nan if best_evaluation is None else float(best_evaluation.ari),
        "cp_rmse": np.nan if best_evaluation is None else float(best_evaluation.cp_rmse),
        "multiplicity_accuracy": np.nan if best_evaluation is None else float(best_evaluation.multiplicity_accuracy),
        "n_eval_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_eval_mutations),
        "n_filtered_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_filtered_mutations),
        "elapsed_seconds": float(perf_counter() - start_time),
    }
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
    bic_df_scale: float = 10.0,
    bic_cluster_penalty: float = 6.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
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
            )
        )

    summary_df = pd.DataFrame(summaries).sort_values("patient_id").reset_index(drop=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(outdir) / "single_stage_summary.tsv", sep="\t", index=False)
    return summary_df
