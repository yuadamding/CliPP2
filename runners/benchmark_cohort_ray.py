from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import ray

from ..core.model import FitOptions
from .benchmark_common import (
    _fit_options_from_args,
    _parse_lambda_grid,
    materialize_patient_df,
    parse_cohort_patient_id,
    write_benchmark_tables,
    write_patient_checkpoint,
)
from .model_selection import (
    ORACLE_EXPANSION_FACTOR,
    ORACLE_MAX_LAMBDA,
    ORACLE_MAX_SEARCH_ROUNDS,
    ORACLE_MIN_LAMBDA,
    ORACLE_REFINE_POINTS,
    ORACLE_REFINE_ROUNDS,
    ORACLE_ULTRA_DENSE_POINTS,
)
from .pipeline import process_one_file_bundle


def _configure_cpu_runtime() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass


@ray.remote(num_cpus=1)
def _process_one_file_remote(
    *,
    file_path: str,
    outdir: str,
    simulation_root: str,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    graph_file: str | None,
    fit_options_kwargs: dict[str, object],
    bic_df_scale: float,
    bic_cluster_penalty: float,
    settings_profile: str,
    selection_score: str,
    use_warm_starts: bool,
    write_outputs: bool,
) -> dict[str, float | int | str | bool]:
    _configure_cpu_runtime()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        summary, search_df = process_one_file_bundle(
            file_path=Path(file_path),
            outdir=Path(outdir),
            simulation_root=Path(simulation_root),
            lambda_grid=lambda_grid,
            lambda_grid_mode=lambda_grid_mode,
            fit_options=FitOptions(**fit_options_kwargs),
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            settings_profile=settings_profile,
            selection_score=selection_score,
            use_warm_starts=use_warm_starts,
            write_outputs=write_outputs,
            graph_file=None if graph_file is None else Path(graph_file),
        )
        candidate_rows = search_df.to_dict(orient="records")
        return {
            "summary": summary,
            "candidate_rows": candidate_rows,
        }


def _materialize_candidate_df(
    candidate_rows: list[dict[str, float | int | str | bool]],
) -> pd.DataFrame:
    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        return candidate_df
    sort_columns = [
        column
        for column in [
            "N_mean",
            "purity",
            "amp_rate",
            "n_regions",
            "true_K",
            "rep",
            "tumor_id",
            "selection_step",
            "lambda",
        ]
        if column in candidate_df.columns
    ]
    if sort_columns:
        candidate_df = candidate_df.sort_values(sort_columns).reset_index(drop=True)
    else:
        candidate_df = candidate_df.reset_index(drop=True)
    return candidate_df


def _aggregate_tuning_guidance(best_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if best_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    guidance_df = best_df.copy()
    if {"selection_boundary_unresolved", "ari_boundary_unresolved"}.issubset(guidance_df.columns):
        guidance_df = guidance_df.loc[
            ~guidance_df["selection_boundary_unresolved"].astype(bool)
            & ~guidance_df["ari_boundary_unresolved"].astype(bool)
        ].copy()
    if "selection_used_convergence_fallback" in guidance_df.columns:
        guidance_df = guidance_df.loc[~guidance_df["selection_used_convergence_fallback"].astype(bool)].copy()
    if guidance_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    scenario_cols = [
        column
        for column in ["N_mean", "true_K", "purity", "amp_rate", "n_regions", "lambda_mut_setting"]
        if column in guidance_df.columns
    ]
    guidance_by_scenario = (
        guidance_df.groupby(scenario_cols, dropna=False)
        .agg(
            n_tumors=("tumor_id", "size"),
            mean_selected_lambda=("selected_lambda", "mean"),
            median_selected_lambda=("selected_lambda", "median"),
            mean_selection_lambda_min=("selection_lambda_min", "mean"),
            mean_selection_lambda_max=("selection_lambda_max", "mean"),
            mean_selection_lambda_count=("selection_lambda_count", "mean"),
            selection_boundary_unresolved_rate=("selection_boundary_unresolved", "mean"),
            mean_best_ari=("best_ari", "mean"),
            median_best_ari=("best_ari", "median"),
            ari_boundary_unresolved_rate=("ari_boundary_unresolved", "mean"),
            convergence_fallback_rate=("selection_used_convergence_fallback", "mean"),
            mean_oracle_search_rounds_completed=("oracle_search_rounds_completed", "mean"),
            mean_ari=("ARI", "mean"),
            median_ari=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_n_clusters=("n_clusters", "mean"),
        )
        .reset_index()
        .sort_values(scenario_cols)
    )

    guidance_by_n_regions = (
        guidance_df.groupby(["n_regions"], dropna=False)
        .agg(
            n_tumors=("tumor_id", "size"),
            mean_selected_lambda=("selected_lambda", "mean"),
            median_selected_lambda=("selected_lambda", "median"),
            mean_selection_lambda_min=("selection_lambda_min", "mean"),
            mean_selection_lambda_max=("selection_lambda_max", "mean"),
            mean_selection_lambda_count=("selection_lambda_count", "mean"),
            selection_boundary_unresolved_rate=("selection_boundary_unresolved", "mean"),
            mean_best_ari=("best_ari", "mean"),
            median_best_ari=("best_ari", "median"),
            ari_boundary_unresolved_rate=("ari_boundary_unresolved", "mean"),
            convergence_fallback_rate=("selection_used_convergence_fallback", "mean"),
            mean_oracle_search_rounds_completed=("oracle_search_rounds_completed", "mean"),
            mean_ari=("ARI", "mean"),
            median_ari=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_n_clusters=("n_clusters", "mean"),
        )
        .reset_index()
        .sort_values(["n_regions"])
    )
    return guidance_by_scenario, guidance_by_n_regions


def _compute_top_candidates(candidate_df: pd.DataFrame, *, top_k: int = 5) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df

    def _top_group(group: pd.DataFrame) -> pd.DataFrame:
        if "ARI" in group.columns and group["ARI"].notna().any():
            ordered = group.sort_values(
                ["ARI", "cp_rmse", "lambda"],
                ascending=[False, True, True],
                na_position="last",
            )
        else:
            ordered = group.sort_values(["bic", "lambda"], ascending=[True, True], na_position="last")
        return ordered.head(top_k)

    return (
        candidate_df.groupby("tumor_id", group_keys=False)
        .apply(_top_group)
        .reset_index(drop=True)
    )


def _compute_near_best_candidates(candidate_df: pd.DataFrame, *, ari_tol: float = 0.01) -> pd.DataFrame:
    if candidate_df.empty or "ARI" not in candidate_df.columns or candidate_df["ARI"].notna().sum() == 0:
        return pd.DataFrame(columns=candidate_df.columns)

    best_ari_by_tumor = candidate_df.groupby("tumor_id")["ARI"].transform("max")
    mask = candidate_df["ARI"].notna() & (best_ari_by_tumor - candidate_df["ARI"] <= ari_tol + 1e-12)
    return candidate_df.loc[mask].sort_values(["tumor_id", "ARI", "lambda"], ascending=[True, False, True]).reset_index(drop=True)


def _safe_log10(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return np.nan
    return float(np.log10(value))


def _safe_log10_span(lower: float, upper: float) -> float:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= 0.0 or upper < lower:
        return np.nan
    return float(np.log10(upper) - np.log10(lower))


def _enrich_candidate_landscape(candidate_df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    enriched = candidate_df.copy()
    best_cols = [
        column
        for column in [
            "tumor_id",
            "selected_lambda",
            "selection_metric_value",
            "selection_lambda_min",
            "selection_lambda_max",
            "selection_lambda_count",
            "best_ari",
            "ari_optimal_lambda_min",
            "ari_optimal_lambda_max",
            "ari_optimal_lambda_count",
            "selection_boundary_unresolved",
            "ari_boundary_unresolved",
            "selection_used_convergence_fallback",
            "oracle_search_rounds_completed",
            "oracle_search_stop_reason",
            "num_candidates",
            "num_converged_candidates",
            "tested_lambda_min",
            "tested_lambda_max",
            "tested_lambda_count",
        ]
        if column in best_df.columns
    ]
    if best_cols:
        enriched = enriched.merge(best_df[best_cols], on="tumor_id", how="left", suffixes=("", "_tumor"))

    enriched["lambda_log10"] = enriched["lambda"].map(_safe_log10)
    if "selected_lambda" in enriched.columns:
        enriched["selected_lambda_log10"] = enriched["selected_lambda"].map(_safe_log10)
        enriched["abs_log10_distance_to_selected_lambda"] = np.abs(enriched["lambda_log10"] - enriched["selected_lambda_log10"])

    if "ARI" in enriched.columns:
        enriched["best_ari_in_tumor"] = enriched.groupby("tumor_id")["ARI"].transform("max")
        enriched["delta_to_best_ari"] = enriched["best_ari_in_tumor"] - enriched["ARI"]
        enriched["ari_rank_within_tumor"] = (
            enriched.groupby("tumor_id")["ARI"].rank(method="dense", ascending=False, na_option="bottom")
        )
    if "bic" in enriched.columns:
        enriched["best_bic_in_tumor"] = enriched.groupby("tumor_id")["bic"].transform("min")
        enriched["delta_to_best_bic"] = enriched["bic"] - enriched["best_bic_in_tumor"]
        enriched["bic_rank_within_tumor"] = (
            enriched.groupby("tumor_id")["bic"].rank(method="dense", ascending=True, na_option="bottom")
        )
    if "cp_rmse" in enriched.columns:
        enriched["best_cp_rmse_in_tumor"] = enriched.groupby("tumor_id")["cp_rmse"].transform("min")
        enriched["delta_to_best_cp_rmse"] = enriched["cp_rmse"] - enriched["best_cp_rmse_in_tumor"]
        enriched["cp_rmse_rank_within_tumor"] = (
            enriched.groupby("tumor_id")["cp_rmse"].rank(method="dense", ascending=True, na_option="bottom")
        )
    return enriched


def _compute_search_diagnostics(best_df: pd.DataFrame) -> pd.DataFrame:
    if best_df.empty:
        return best_df
    diagnostics = best_df.copy()
    diagnostics["tested_log10_lambda_span"] = [
        _safe_log10_span(lower, upper)
        for lower, upper in zip(
            diagnostics.get("tested_lambda_min", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
            diagnostics.get("tested_lambda_max", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
        )
    ]
    diagnostics["selection_log10_lambda_span"] = [
        _safe_log10_span(lower, upper)
        for lower, upper in zip(
            diagnostics.get("selection_lambda_min", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
            diagnostics.get("selection_lambda_max", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
        )
    ]
    diagnostics["ari_log10_lambda_span"] = [
        _safe_log10_span(lower, upper)
        for lower, upper in zip(
            diagnostics.get("ari_optimal_lambda_min", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
            diagnostics.get("ari_optimal_lambda_max", pd.Series(dtype=float)).to_numpy(dtype=float, copy=False),
        )
    ]
    return diagnostics


def _aggregate_guidance_simple(best_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if best_df.empty:
        return pd.DataFrame()
    guidance_df = best_df.copy()
    if {"selection_boundary_unresolved", "ari_boundary_unresolved"}.issubset(guidance_df.columns):
        guidance_df = guidance_df.loc[
            ~guidance_df["selection_boundary_unresolved"].astype(bool)
            & ~guidance_df["ari_boundary_unresolved"].astype(bool)
        ].copy()
    if "selection_used_convergence_fallback" in guidance_df.columns:
        guidance_df = guidance_df.loc[~guidance_df["selection_used_convergence_fallback"].astype(bool)].copy()
    if guidance_df.empty:
        return pd.DataFrame()
    existing_columns = [column for column in columns if column in guidance_df.columns]
    if not existing_columns:
        return pd.DataFrame()
    aggregated = (
        guidance_df.groupby(existing_columns, dropna=False)
        .agg(
            n_tumors=("tumor_id", "size"),
            mean_selected_lambda=("selected_lambda", "mean"),
            median_selected_lambda=("selected_lambda", "median"),
            mean_selection_lambda_count=("selection_lambda_count", "mean"),
            mean_best_ari=("best_ari", "mean"),
            median_best_ari=("best_ari", "median"),
            mean_ari=("ARI", "mean"),
            median_ari=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_n_clusters=("n_clusters", "mean"),
            mean_oracle_search_rounds_completed=("oracle_search_rounds_completed", "mean"),
        )
        .reset_index()
        .sort_values(existing_columns)
    )
    return aggregated


def _compute_lambda_guidance(candidate_df: pd.DataFrame, *, resolved_tumor_ids: set[str] | None = None) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    filtered_df = candidate_df.copy()
    if resolved_tumor_ids is not None:
        filtered_df = filtered_df.loc[filtered_df["tumor_id"].astype(str).isin(resolved_tumor_ids)].copy()
    if "eligible_for_selection" in filtered_df.columns:
        filtered_df = filtered_df.loc[filtered_df["eligible_for_selection"].astype(bool)].copy()
    if filtered_df.empty:
        return pd.DataFrame(columns=["n_regions", "n_tumors", "lambda_min", "lambda_median", "lambda_max"])
    if "is_ari_optimal" in filtered_df.columns and filtered_df["is_ari_optimal"].any():
        optimal_df = filtered_df.loc[filtered_df["is_ari_optimal"].astype(bool)].copy()
    elif "is_selection_optimal" in filtered_df.columns and filtered_df["is_selection_optimal"].any():
        optimal_df = filtered_df.loc[filtered_df["is_selection_optimal"].astype(bool)].copy()
    else:
        return pd.DataFrame(columns=["n_regions", "n_tumors", "lambda_min", "lambda_median", "lambda_max"])

    group_cols = [column for column in ["n_regions", "N_mean", "purity", "amp_rate", "true_K"] if column in optimal_df.columns]
    if not group_cols:
        group_cols = ["tumor_id"]
    lambda_guidance = (
        optimal_df.groupby(group_cols, dropna=False)
        .agg(
            n_candidates=("lambda", "size"),
            n_tumors=("tumor_id", "nunique"),
            lambda_min=("lambda", "min"),
            lambda_median=("lambda", "median"),
            lambda_max=("lambda", "max"),
            mean_ari=("ARI", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )
    return lambda_guidance


def _write_tuning_memory(
    *,
    outdir: Path,
    patient_rows: list[dict[str, float | int | str | bool]],
    candidate_rows: list[dict[str, float | int | str | bool]],
    search_config: dict[str, object],
) -> None:
    best_df = materialize_patient_df(patient_rows)
    candidate_df = _materialize_candidate_df(candidate_rows)
    resolved_tumor_ids: set[str] | None = None
    if {"selection_boundary_unresolved", "ari_boundary_unresolved"}.issubset(best_df.columns):
        resolved_mask = (
            ~best_df["selection_boundary_unresolved"].astype(bool)
            & ~best_df["ari_boundary_unresolved"].astype(bool)
        )
        if "selection_used_convergence_fallback" in best_df.columns:
            resolved_mask &= ~best_df["selection_used_convergence_fallback"].astype(bool)
        resolved_tumor_ids = set(best_df.loc[resolved_mask, "tumor_id"].astype(str).tolist())

    pd.DataFrame([search_config]).to_csv(outdir / "search_config.tsv", sep="\t", index=False)
    best_df.to_csv(outdir / "per_tumor_best.tsv", sep="\t", index=False)
    search_diagnostics_df = _compute_search_diagnostics(best_df)
    search_diagnostics_df.to_csv(outdir / "search_diagnostics.tsv", sep="\t", index=False)
    enriched_candidate_df = _enrich_candidate_landscape(candidate_df, search_diagnostics_df)
    enriched_candidate_df.to_csv(outdir / "per_candidate.tsv", sep="\t", index=False)
    enriched_candidate_df.to_csv(outdir / "candidate_landscape.tsv", sep="\t", index=False)
    _compute_top_candidates(enriched_candidate_df).to_csv(outdir / "top_candidates.tsv", sep="\t", index=False)
    _compute_near_best_candidates(enriched_candidate_df).to_csv(outdir / "near_best_candidates.tsv", sep="\t", index=False)
    if resolved_tumor_ids is None:
        resolved_best_df = best_df.iloc[0:0].copy()
        unresolved_best_df = best_df.copy()
    else:
        resolved_best_df = best_df.loc[best_df["tumor_id"].astype(str).isin(resolved_tumor_ids)].copy()
        unresolved_best_df = best_df.loc[~best_df["tumor_id"].astype(str).isin(resolved_tumor_ids)].copy()
    resolved_best_df.to_csv(outdir / "resolved_tumor_best.tsv", sep="\t", index=False)
    unresolved_best_df.to_csv(outdir / "unresolved_tumor_best.tsv", sep="\t", index=False)
    if "selection_used_convergence_fallback" in best_df.columns:
        best_df.loc[best_df["selection_used_convergence_fallback"].astype(bool)].to_csv(
            outdir / "convergence_fallback_tumors.tsv",
            sep="\t",
            index=False,
        )

    guidance_by_scenario, guidance_by_n_regions = _aggregate_tuning_guidance(best_df)
    guidance_by_scenario.to_csv(outdir / "guidance_by_scenario.tsv", sep="\t", index=False)
    guidance_by_n_regions.to_csv(outdir / "guidance_by_n_regions.tsv", sep="\t", index=False)
    _aggregate_guidance_simple(best_df, ["true_K"]).to_csv(outdir / "guidance_by_true_k.tsv", sep="\t", index=False)
    _aggregate_guidance_simple(best_df, ["N_mean"]).to_csv(outdir / "guidance_by_depth.tsv", sep="\t", index=False)
    _aggregate_guidance_simple(best_df, ["purity"]).to_csv(outdir / "guidance_by_purity.tsv", sep="\t", index=False)
    _aggregate_guidance_simple(best_df, ["amp_rate"]).to_csv(outdir / "guidance_by_amp_rate.tsv", sep="\t", index=False)
    _compute_lambda_guidance(enriched_candidate_df, resolved_tumor_ids=resolved_tumor_ids).to_csv(outdir / "lambda_guidance.tsv", sep="\t", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "attempted_tumors": int(best_df.shape[0]),
                "evaluated_candidates": int(candidate_df.shape[0]),
                "mean_ARI": float(best_df["ARI"].mean()) if "ARI" in best_df.columns and not best_df.empty else np.nan,
                "median_ARI": float(best_df["ARI"].median()) if "ARI" in best_df.columns and not best_df.empty else np.nan,
                "mean_cp_rmse": float(best_df["cp_rmse"].mean()) if "cp_rmse" in best_df.columns and not best_df.empty else np.nan,
                "mean_selected_lambda": float(best_df["selected_lambda"].mean()) if "selected_lambda" in best_df.columns and not best_df.empty else np.nan,
                "mean_selection_lambda_count": float(best_df["selection_lambda_count"].mean()) if "selection_lambda_count" in best_df.columns and not best_df.empty else np.nan,
                "mean_ari_optimal_lambda_count": float(best_df["ari_optimal_lambda_count"].mean()) if "ari_optimal_lambda_count" in best_df.columns and not best_df.empty else np.nan,
                "selection_boundary_unresolved_tumors": int(best_df["selection_boundary_unresolved"].sum()) if "selection_boundary_unresolved" in best_df.columns and not best_df.empty else 0,
                "ari_boundary_unresolved_tumors": int(best_df["ari_boundary_unresolved"].sum()) if "ari_boundary_unresolved" in best_df.columns and not best_df.empty else 0,
                "selection_convergence_fallback_tumors": int(best_df["selection_used_convergence_fallback"].sum()) if "selection_used_convergence_fallback" in best_df.columns and not best_df.empty else 0,
                "resolved_guidance_tumors": 0 if resolved_tumor_ids is None else int(len(resolved_tumor_ids)),
                "mean_oracle_search_rounds_completed": float(best_df["oracle_search_rounds_completed"].mean()) if "oracle_search_rounds_completed" in best_df.columns and not best_df.empty else np.nan,
                "search_stop_reason_mode": ""
                if "oracle_search_stop_reason" not in best_df.columns or best_df.empty
                else str(best_df["oracle_search_stop_reason"].mode(dropna=True).iloc[0]) if not best_df["oracle_search_stop_reason"].mode(dropna=True).empty else "",
            }
        ]
    )
    summary_df.to_csv(outdir / "summary.tsv", sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m clipp2.runners.benchmark_cohort_ray",
        description="Benchmark CliPP2 on an existing multiregion cohort with Ray parallelism.",
    )
    parser.add_argument("--input-dir", default="/storage/CliPP2/CliPP2Sim1K_TSV", help="Directory with per-tumor TSV files.")
    parser.add_argument("--simulation-root", default="/storage/CliPP2/CliPP2Sim1K", help="Root directory with simulation truth folders.")
    parser.add_argument(
        "--outdir",
        default="/storage/CliPP2/CliPP2Sim1K_benchmark_objective_faithful_oracle_ari_cpu_ray24",
        help="Output directory for benchmark tables.",
    )
    parser.add_argument("--workers", type=int, default=24, help="Number of Ray workers.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of tumors.")
    parser.add_argument("--flush-every", type=int, default=1, help="Write partial summaries every N finished tumors.")
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Optional TSV defining a custom pairwise-fusion graph using either edge_u/edge_v or mutation_u/mutation_v columns, with optional edge_w.",
    )
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero", "ultra_dense_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template when --lambda-grid is not provided.",
    )
    parser.add_argument("--outer-max-iter", type=int, default=4, help="Maximum outer majorization iterations.")
    parser.add_argument("--inner-max-iter", type=int, default=30, help="Maximum inner solver iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Optimization tolerance.")
    parser.add_argument("--bic-df-scale", type=float, default=8.0, help="Extended BIC CP degrees-of-freedom scale.")
    parser.add_argument("--bic-cluster-penalty", type=float, default=4.0, help="Extended BIC cluster-count penalty.")
    parser.add_argument("--settings-profile", choices=["manual", "auto"], default="auto", help="Model-selection strategy.")
    parser.add_argument(
        "--selection-score",
        choices=["ebic", "classic_bic", "refit_ebic", "classic_refit_bic", "oracle_ari"],
        default="oracle_ari",
        help="Candidate scoring objective. BIC-style scores use the post-hoc summary partition/log-likelihood pair; 'oracle_ari' uses simulation truth to select the best lambda and stores the optimal-ARI lambda range.",
    )
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability for major-copy multiplicity.")
    parser.add_argument("--disable-warm-start", action="store_true", help="Disable lambda-path warm starts.")
    parser.add_argument("--write-tumor-outputs", action="store_true", help="Write full per-tumor result files.")
    parser.add_argument("--verbose", action="store_true", help="Print optimizer progress inside each worker.")
    return parser


def run_ray_cohort_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _configure_cpu_runtime()

    input_dir = Path(args.input_dir)
    simulation_root = Path(args.simulation_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.tsv"), key=lambda path: (path.stat().st_size, path.name))
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if not files:
        raise RuntimeError(f"No TSV files found in {input_dir}")

    fit_options = _fit_options_from_args(
        argparse.Namespace(
            outer_max_iter=args.outer_max_iter,
            inner_max_iter=args.inner_max_iter,
            tol=args.tol,
            major_prior=args.major_prior,
            device="cpu",
            verbose=args.verbose,
        )
    )
    fit_options_kwargs = {
        "lambda_value": float(fit_options.lambda_value),
        "outer_max_iter": int(fit_options.outer_max_iter),
        "inner_max_iter": int(fit_options.inner_max_iter),
        "tol": float(fit_options.tol),
        "major_prior": float(fit_options.major_prior),
        "eps": float(fit_options.eps),
        "graph": None,
        "device": "cpu",
        "verbose": bool(fit_options.verbose),
    }
    search_config = {
        "input_dir": str(input_dir),
        "simulation_root": str(simulation_root),
        "outdir": str(outdir),
        "workers": int(args.workers),
        "selection_score": str(args.selection_score),
        "settings_profile": str(args.settings_profile),
        "lambda_grid_mode": str(args.lambda_grid_mode),
        "lambda_grid": "" if args.lambda_grid is None else str(args.lambda_grid),
        "graph_file": "" if args.graph_file is None else str(args.graph_file),
        "outer_max_iter": int(args.outer_max_iter),
        "inner_max_iter": int(args.inner_max_iter),
        "tol": float(args.tol),
        "device": "cpu",
        "major_prior": float(args.major_prior),
        "oracle_refine_rounds": int(ORACLE_REFINE_ROUNDS),
        "oracle_refine_points": int(ORACLE_REFINE_POINTS),
        "oracle_ultra_dense_points": int(ORACLE_ULTRA_DENSE_POINTS),
        "oracle_expansion_factor": float(ORACLE_EXPANSION_FACTOR),
        "oracle_max_search_rounds": int(ORACLE_MAX_SEARCH_ROUNDS),
        "oracle_min_lambda": float(ORACLE_MIN_LAMBDA),
        "oracle_max_lambda": float(ORACLE_MAX_LAMBDA),
        "boundary_hit_means_search_problem": True,
    }

    ray.init(
        num_cpus=max(int(args.workers), 1),
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=True,
    )

    patient_rows: list[dict[str, float | int | str | bool]] = []
    candidate_rows: list[dict[str, float | int | str | bool]] = []
    total_files = len(files)
    start_time = perf_counter()
    in_flight: dict[ray.ObjectRef, Path] = {}
    file_iter = iter(files)
    max_in_flight = max(int(args.workers) * 2, 1)

    def submit_next() -> bool:
        try:
            file_path = next(file_iter)
        except StopIteration:
            return False
        future = _process_one_file_remote.remote(
            file_path=str(file_path),
            outdir=str(outdir),
            simulation_root=str(simulation_root),
            lambda_grid=_parse_lambda_grid(args.lambda_grid),
            lambda_grid_mode=str(args.lambda_grid_mode),
            graph_file=None if args.graph_file is None else str(args.graph_file),
            fit_options_kwargs=fit_options_kwargs,
            bic_df_scale=float(args.bic_df_scale),
            bic_cluster_penalty=float(args.bic_cluster_penalty),
            settings_profile=str(args.settings_profile),
            selection_score=str(args.selection_score),
            use_warm_starts=not args.disable_warm_start,
            write_outputs=bool(args.write_tumor_outputs),
        )
        in_flight[future] = file_path
        return True

    for _ in range(min(max_in_flight, total_files)):
        if not submit_next():
            break

    completed = 0
    while in_flight:
        ready, _ = ray.wait(list(in_flight.keys()), num_returns=1)
        future = ready[0]
        file_path = in_flight.pop(future)
        result = ray.get(future)
        summary = result["summary"]
        local_candidate_rows = result["candidate_rows"]
        meta = parse_cohort_patient_id(file_path.stem)
        patient_rows.append({**meta, **summary})
        candidate_rows.extend({**meta, **row} for row in local_candidate_rows)
        completed += 1

        if completed % max(int(args.flush_every), 1) == 0 or completed == total_files:
            patient_df = write_patient_checkpoint(
                patient_rows,
                outdir=outdir,
                start_time=start_time,
                case_index=completed,
                total_cases=total_files,
                label="ray-cohort-benchmark",
            )
            write_benchmark_tables(patient_df, outdir)
            _write_tuning_memory(
                outdir=outdir,
                patient_rows=patient_rows,
                candidate_rows=candidate_rows,
                search_config=search_config,
            )

        while len(in_flight) < max_in_flight:
            if not submit_next():
                break

    patient_df = materialize_patient_df(patient_rows)
    scenario_df, global_df = write_benchmark_tables(patient_df, outdir)
    _write_tuning_memory(
        outdir=outdir,
        patient_rows=patient_rows,
        candidate_rows=candidate_rows,
        search_config=search_config,
    )
    ray.shutdown()
    return patient_df, scenario_df, global_df


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    patient_df, scenario_df, global_df = run_ray_cohort_benchmark(args)
    print(global_df.to_string(index=False))
    print(scenario_df.head().to_string(index=False))
    print(patient_df.head().to_string(index=False))


__all__ = [
    "build_parser",
    "main",
    "run_ray_cohort_benchmark",
]
