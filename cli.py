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
            "multi-region clipp: unified single-region and multiregion cellular prevalence "
            "clustering with binomial likelihood, major/minor multiplicity candidates, "
            "and graph fused lasso."
        ),
    )
    parser.add_argument("--input-dir", default="CliPP2Sim_PyClone", help="Directory with per-patient TSV files.")
    parser.add_argument("--input-file", default=None, help="Optional single patient TSV file.")
    parser.add_argument("--outdir", default="multi_region_clipp_results", help="Output directory.")
    parser.add_argument("--simulation-root", default="CliPP2Sim", help="Simulation root for ARI evaluation.")
    parser.add_argument("--lambda-grid", default=None, help="Comma-separated lambda grid or 'auto'.")
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    parser.add_argument("--graph-k", type=int, default=8, help="k for the mutation kNN graph.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, or cpu.")
    parser.add_argument(
        "--benchmark-simulation",
        action="store_true",
        help="Run a stratified simulation benchmark instead of processing all input files directly.",
    )
    parser.add_argument(
        "--reps-per-scenario",
        type=int,
        default=1,
        help="Number of representative patients to benchmark for each simulation scenario.",
    )
    parser.add_argument(
        "--n-mean-values",
        default=None,
        help="Optional comma-separated filter for benchmark depth settings, for example '50,300,1000'.",
    )
    parser.add_argument("--em-max-iter", type=int, default=12, help="Maximum EM iterations.")
    parser.add_argument("--admm-max-iter", type=int, default=30, help="Maximum ADMM iterations per M-step.")
    parser.add_argument("--inner-steps", type=int, default=2, help="Quadratic-majorization updates per ADMM iteration.")
    parser.add_argument(
        "--inner-lr",
        type=float,
        default=5e-2,
        help="Deprecated compatibility knob; no longer used by the ADMM solver.",
    )
    parser.add_argument("--cg-max-iter", type=int, default=50, help="Maximum conjugate-gradient iterations for each ADMM w-update.")
    parser.add_argument("--cg-tol", type=float, default=1e-4, help="Relative conjugate-gradient tolerance.")
    parser.add_argument("--curvature-floor", type=float, default=1e-4, help="Minimum diagonal curvature used in the quadratic surrogate.")
    parser.add_argument("--admm-rho", type=float, default=2.0, help="ADMM penalty parameter.")
    parser.add_argument("--em-tol", type=float, default=1e-4, help="Relative EM stopping tolerance.")
    parser.add_argument("--admm-tol", type=float, default=5e-3, help="ADMM stopping tolerance.")
    parser.add_argument("--fused-tol", type=float, default=1e-3, help="Tolerance for calling an edge fused.")
    parser.add_argument(
        "--center-merge-tol",
        type=float,
        default=1e-1,
        help="Global distance threshold for merging nearly identical fused CP profiles.",
    )
    parser.add_argument(
        "--bic-df-scale",
        type=float,
        default=10.0,
        help="Scale factor on the CP-profile degrees of freedom in the extended BIC selection score.",
    )
    parser.add_argument(
        "--bic-cluster-penalty",
        type=float,
        default=6.0,
        help="Additional cluster-count complexity penalty in the extended BIC selection score.",
    )
    parser.add_argument(
        "--settings-profile",
        choices=["manual", "auto"],
        default="auto",
        help="Use fixed manual settings or the rule-based auto profile learned from simulation benchmarks.",
    )
    parser.add_argument(
        "--disable-warm-start",
        action="store_true",
        help="Disable lambda-path warm starts and refit each candidate from the initial CP estimate.",
    )
    parser.add_argument(
        "--skip-patient-outputs",
        action="store_true",
        help="Skip per-patient mutation/cluster/lambda files and only write benchmark summaries.",
    )
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability assigned to major-copy multiplicity.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on the number of files processed.")
    parser.add_argument("--verbose", action="store_true", help="Print EM and ADMM progress.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fit_options = FitOptions(
        lambda_value=0.0,
        em_max_iter=args.em_max_iter,
        em_tol=args.em_tol,
        admm_max_iter=args.admm_max_iter,
        admm_tol=args.admm_tol,
        admm_rho=args.admm_rho,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        cg_max_iter=args.cg_max_iter,
        cg_tol=args.cg_tol,
        curvature_floor=args.curvature_floor,
        major_prior=args.major_prior,
        fused_tol=args.fused_tol,
        center_merge_tol=args.center_merge_tol,
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
            graph_k=args.graph_k,
            fit_options=fit_options,
            bic_df_scale=args.bic_df_scale,
            bic_cluster_penalty=args.bic_cluster_penalty,
            settings_profile=args.settings_profile,
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
            graph_k=args.graph_k,
            fit_options=fit_options,
            bic_df_scale=args.bic_df_scale,
            bic_cluster_penalty=args.bic_cluster_penalty,
            settings_profile=args.settings_profile,
            use_warm_starts=not args.disable_warm_start,
            write_outputs=not args.skip_patient_outputs,
        )
        print(summary)
        return

    summary_df = run_directory(
        input_dir=Path(args.input_dir),
        outdir=Path(args.outdir),
        simulation_root=Path(args.simulation_root) if args.simulation_root else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=args.lambda_grid_mode,
        graph_k=args.graph_k,
        fit_options=fit_options,
        max_files=args.max_files,
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_patient_outputs,
    )
    print(summary_df.head().to_string(index=False))


__all__ = ["build_parser", "main"]
