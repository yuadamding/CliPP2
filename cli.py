from __future__ import annotations

import argparse
from pathlib import Path

from . import FitOptions, process_one_file, run_directory, run_simulation_benchmark


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


def _parse_int_grid(value: str | None) -> list[int] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return [int(piece) for piece in cleaned.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-region clipp",
        description=(
            "multi-region clipp: cellular prevalence clustering with exact observed-data "
            "likelihood and objective-faithful pairwise fusion."
        ),
    )
    parser.add_argument("--input-dir", default="CliPP2Sim_TSV", help="Directory with per-tumor TSV files.")
    parser.add_argument("--input-file", default=None, help="Optional single tumor TSV file.")
    parser.add_argument("--outdir", default="multi_region_clipp_results", help="Output directory.")
    parser.add_argument("--simulation-root", default="CliPP2Sim", help="Simulation root for ARI evaluation.")
    parser.add_argument("--lambda-grid", default=None, help="Comma-separated lambda grid or 'auto'.")
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Optional TSV defining a custom pairwise-fusion graph using either edge_u/edge_v or mutation_u/mutation_v columns, with optional edge_w.",
    )
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero", "ultra_dense_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    parser.add_argument(
        "--benchmark-simulation",
        action="store_true",
        help="Run a stratified simulation benchmark instead of processing all input files directly.",
    )
    parser.add_argument(
        "--reps-per-scenario",
        type=int,
        default=1,
        help="Number of representative tumors to benchmark for each simulation scenario.",
    )
    parser.add_argument(
        "--n-mean-values",
        default=None,
        help="Optional comma-separated filter for benchmark depth settings, for example '50,300,1000'.",
    )
    parser.add_argument("--outer-max-iter", type=int, default=8, help="Maximum outer majorization iterations.")
    parser.add_argument("--inner-max-iter", type=int, default=30, help="Maximum inner convex-solver iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Optimization tolerance.")
    parser.add_argument(
        "--bic-df-scale",
        type=float,
        default=8.0,
        help="Scale factor on the CP-profile degrees of freedom in the extended BIC selection score.",
    )
    parser.add_argument(
        "--bic-cluster-penalty",
        type=float,
        default=4.0,
        help="Additional cluster-count complexity penalty in the extended BIC selection score.",
    )
    parser.add_argument(
        "--settings-profile",
        choices=["manual", "auto"],
        default="manual",
        help="Model-selection strategy. 'manual' uses the provided lambda path; 'auto' uses compact pairwise-fusion defaults.",
    )
    parser.add_argument(
        "--selection-score",
        choices=["ebic", "classic_bic", "refit_ebic", "classic_refit_bic", "oracle_ari"],
        default="ebic",
        help="Candidate scoring objective. Legacy refit-* names are accepted as aliases; BIC-style scores are computed on the post-hoc summary partition/log-likelihood pair, while 'oracle_ari' requires simulation truth.",
    )
    parser.add_argument("--disable-warm-start", action="store_true", help="Disable lambda-path warm starts.")
    parser.add_argument(
        "--skip-patient-outputs",
        "--skip-tumor-outputs",
        action="store_true",
        help="Skip per-tumor mutation/cluster/lambda files and only write benchmark summaries.",
    )
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability assigned to major-copy multiplicity.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for the Torch fusion backend.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on the number of files processed.")
    parser.add_argument("--verbose", action="store_true", help="Print optimizer progress.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fit_options = FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        major_prior=args.major_prior,
        device=args.device,
        verbose=args.verbose,
    )
    lambda_grid = _parse_lambda_grid(args.lambda_grid)
    n_mean_values = _parse_int_grid(args.n_mean_values)

    if args.benchmark_simulation:
        patient_df, scenario_df, global_df = run_simulation_benchmark(
            input_dir=Path(args.input_dir),
            simulation_root=Path(args.simulation_root),
            outdir=Path(args.outdir),
            reps_per_scenario=args.reps_per_scenario,
            n_mean_values=n_mean_values,
            lambda_grid=lambda_grid,
            lambda_grid_mode=args.lambda_grid_mode,
            fit_options=fit_options,
            bic_df_scale=args.bic_df_scale,
            bic_cluster_penalty=args.bic_cluster_penalty,
            settings_profile=args.settings_profile,
            selection_score=args.selection_score,
            use_warm_starts=not args.disable_warm_start,
            write_patient_outputs=not args.skip_patient_outputs,
        )
        print(global_df.to_string(index=False))
        print(scenario_df.head().to_string(index=False))
        return

    if args.input_file:
        summary = process_one_file(
            file_path=Path(args.input_file),
            outdir=Path(args.outdir),
            simulation_root=Path(args.simulation_root) if args.simulation_root else None,
            lambda_grid=lambda_grid,
            lambda_grid_mode=args.lambda_grid_mode,
            fit_options=fit_options,
            bic_df_scale=args.bic_df_scale,
            bic_cluster_penalty=args.bic_cluster_penalty,
            settings_profile=args.settings_profile,
            selection_score=args.selection_score,
            use_warm_starts=not args.disable_warm_start,
            write_outputs=not args.skip_patient_outputs,
            graph_file=Path(args.graph_file) if args.graph_file else None,
        )
        print(summary)
        return

    summary_df = run_directory(
        input_dir=Path(args.input_dir),
        outdir=Path(args.outdir),
        simulation_root=Path(args.simulation_root) if args.simulation_root else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=args.lambda_grid_mode,
        fit_options=fit_options,
        max_files=args.max_files,
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        selection_score=args.selection_score,
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_patient_outputs,
        graph_file=Path(args.graph_file) if args.graph_file else None,
    )
    print(summary_df.head().to_string(index=False))


__all__ = ["build_parser", "main"]
