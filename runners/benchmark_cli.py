from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..sim.generation import parse_float_list, parse_int_list
from .benchmark_common import _aggregate_simple, _fit_options_from_args, _parse_lambda_grid
from .benchmark_cohort import run_single_region_cohort_benchmark
from .benchmark_mass import MassiveMultiregionBenchmarkConfig, run_massive_multiregion_benchmark


def _add_shared_fit_args(
    parser: argparse.ArgumentParser,
    *,
    em_max_iter: int,
    admm_max_iter: int,
) -> None:
    parser.add_argument("--graph-k", type=int, default=8, help="k for the mutation kNN graph.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, or cpu.")
    parser.add_argument("--em-max-iter", type=int, default=em_max_iter, help="Maximum EM iterations.")
    parser.add_argument("--admm-max-iter", type=int, default=admm_max_iter, help="Maximum ADMM iterations per M-step.")
    parser.add_argument("--inner-steps", type=int, default=2, help="Quadratic-majorization updates per ADMM iteration.")
    parser.add_argument("--inner-lr", type=float, default=5e-2, help="Deprecated compatibility knob; no longer used by the ADMM solver.")
    parser.add_argument("--cg-max-iter", type=int, default=30 if em_max_iter >= 8 else 50, help="Maximum conjugate-gradient iterations for each ADMM w-update.")
    parser.add_argument("--cg-tol", type=float, default=1e-4, help="Relative conjugate-gradient tolerance.")
    parser.add_argument("--curvature-floor", type=float, default=1e-4, help="Minimum diagonal curvature used in the quadratic surrogate.")
    parser.add_argument("--admm-rho", type=float, default=2.0, help="ADMM penalty parameter.")
    parser.add_argument("--em-tol", type=float, default=1e-4, help="Relative EM stopping tolerance.")
    parser.add_argument("--admm-tol", type=float, default=5e-3, help="ADMM stopping tolerance.")
    parser.add_argument("--fused-tol", type=float, default=1e-3, help="Tolerance for calling an edge fused.")
    parser.add_argument("--center-merge-tol", type=float, default=8e-2, help="Cluster-center merge threshold.")
    parser.add_argument("--bic-df-scale", type=float, default=8.0, help="Extended BIC CP degrees-of-freedom scale.")
    parser.add_argument("--bic-cluster-penalty", type=float, default=4.0, help="Extended BIC cluster-count penalty.")
    parser.add_argument(
        "--settings-profile",
        choices=["manual", "auto", "bayes", "legacy_auto"],
        default="auto",
        help="Model-selection strategy: manual fixed settings, bayes/auto Bayesian optimization, or legacy_auto for the old rule-based baseline.",
    )
    parser.add_argument(
        "--selection-score",
        choices=["ebic", "refit_ebic", "classic_bic", "classic_refit_bic"],
        default="refit_ebic",
        help="Candidate scoring objective: default refit-EBIC on partition-refit likelihood, current EBIC, naive classic BIC, or classic BIC after partition-wise center refit.",
    )
    parser.add_argument("--bo-max-evals", type=int, default=12, help="Maximum Bayesian optimization evaluations per patient when --settings-profile is bayes/auto.")
    parser.add_argument("--bo-init-points", type=int, default=5, help="Number of initial design points before Gaussian-process-guided proposals.")
    parser.add_argument("--bo-random-seed", type=int, default=0, help="Random seed used by the Bayesian optimizer.")
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability for major-copy multiplicity.")
    parser.add_argument("--disable-warm-start", action="store_true", help="Disable lambda-path warm starts.")
    parser.add_argument("--write-patient-outputs", action="store_true", help="Write full per-patient result files.")
    parser.add_argument("--verbose", action="store_true", help="Print EM and ADMM progress.")


def benchmark_config_from_args(args: argparse.Namespace) -> MassiveMultiregionBenchmarkConfig:
    return MassiveMultiregionBenchmarkConfig(
        outdir=args.outdir,
        temp_sim_root=args.temp_sim_root,
        temp_tsv_root=args.temp_tsv_root,
        purity_list=tuple(parse_float_list(args.purity_list)),
        amp_rate_list=tuple(parse_float_list(args.amp_rate_list)),
        N_list=tuple(parse_int_list(args.N_list)),
        n_samples_list=tuple(parse_int_list(args.n_samples_list)),
        reps=args.reps,
        seed=args.seed,
        K_min=args.K_min,
        K_max=args.K_max,
        lambda_mut=args.lambda_mut,
        lambda_mut_list=tuple(parse_int_list(args.lambda_mut_list)) if args.lambda_mut_list else None,
        alpha_mut=args.alpha_mut,
        alpha_lambda=args.alpha_lambda,
        tau_lineage_min=args.tau_lineage_min,
        tau_lineage_max=args.tau_lineage_max,
        purity_conc=args.purity_conc,
        lineage_zero_prob=args.lineage_zero_prob,
        min_clone_ccf=args.min_clone_ccf,
        min_clone_ccf_l2_norm=args.min_clone_ccf_l2_norm,
        min_mutations_per_clone=args.min_mutations_per_clone,
        min_clone_ccf_distance=args.min_clone_ccf_distance,
        max_rejection_tries=args.max_rejection_tries,
        flush_every=args.flush_every,
        cleanup_temp=not args.keep_temp,
        fit_workers=args.fit_workers,
        worker_threads=args.worker_threads,
        prefetch_workers=args.prefetch_workers,
        prefetch_buffer=args.prefetch_buffer,
    )


def build_single_region_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-region clipp single-region-benchmark",
        description="Benchmark multi-region clipp on a single-region cohort through the shared benchmark framework.",
    )
    parser.add_argument("--input-dir", default="/data/CliPP_Sim/CliPPSim4K_TSV", help="Directory with per-patient TSV files.")
    parser.add_argument("--simulation-root", default="/data/CliPP_Sim/CliPPSim4K", help="Root directory with single-region truth folders.")
    parser.add_argument("--outdir", default="multi_region_clipp_single_region_benchmark", help="Output directory.")
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    _add_shared_fit_args(parser, em_max_iter=8, admm_max_iter=20)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on the number of files processed.")
    parser.add_argument("--flush-every", type=int, default=100, help="Write partial patient summaries every N cases.")
    parser.add_argument("--reps-per-scenario", type=int, default=None, help="Optional balanced subsampling count per single-region scenario.")
    return parser


def build_mass_multiregion_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-region clipp multiregion-benchmark",
        description="Benchmark multi-region clipp on large multiregion simulation cohorts.",
    )
    parser.add_argument("--outdir", default="multi_region_clipp_multiregion_benchmark", help="Directory for benchmark summaries.")
    parser.add_argument("--temp-sim-root", default=None, help="Optional temporary directory for raw simulation folders.")
    parser.add_argument("--temp-tsv-root", default=None, help="Optional temporary directory for merged TSV files.")
    parser.add_argument("--purity-list", default="0.3,0.6,0.9", help="Comma-separated purity values.")
    parser.add_argument("--amp-rate-list", default="0.1,0.2", help="Comma-separated CNA amplification rates.")
    parser.add_argument("--N-list", default="50,75,100,200,300,400,500,1000", help="Comma-separated mean depth values.")
    parser.add_argument("--n-samples-list", default="5,10,15", help="Comma-separated multiregion sample counts.")
    parser.add_argument("--reps", type=int, default=70, help="Replicates per scenario.")
    parser.add_argument("--seed", type=int, default=None, help="Master RNG seed.")
    parser.add_argument("--K-min", type=int, default=5, help="Minimum clone count for harder multiregion cases.")
    parser.add_argument("--K-max", type=int, default=10, help="Maximum clone count for harder multiregion cases.")
    parser.add_argument("--lambda-mut", type=int, default=2000, help="Legacy single Poisson mean for mutation count.")
    parser.add_argument("--lambda-mut-list", default="300,600,1000,2000,4000", help="Comma-separated Poisson means for mutation counts.")
    parser.add_argument("--alpha-mut", type=float, default=10.0, help="Dirichlet concentration for mutation allocation.")
    parser.add_argument("--alpha-lambda", type=float, default=5.0, help="Dirichlet concentration for lineage residual masses.")
    parser.add_argument("--tau-lineage-min", type=float, default=1.0, help="Minimum lineage concentration per sample.")
    parser.add_argument("--tau-lineage-max", type=float, default=50.0, help="Maximum lineage concentration per sample.")
    parser.add_argument("--purity-conc", type=float, default=50.0, help="Beta concentration for sample purities.")
    parser.add_argument("--lineage-zero-prob", type=float, default=0.0, help="Probability of zeroing a lineage in a sample. Default is 0.0 because clone CCF is constrained to stay positive in every region.")
    parser.add_argument("--min-clone-ccf", type=float, default=0.02, help="Minimum allowed clone CCF in every region.")
    parser.add_argument("--min-clone-ccf-l2-norm", type=float, default=0.05, help="Minimum L2 norm of each clone's multiregion CCF vector.")
    parser.add_argument("--min-mutations-per-clone", type=int, default=15, help="Minimum number of mutations assigned to each clone.")
    parser.add_argument("--min-clone-ccf-distance", type=float, default=0.10, help="Minimum L2 distance between any two clones' multiregion CCF profiles.")
    parser.add_argument("--max-rejection-tries", type=int, default=1024, help="Maximum rejection-sampling attempts when enforcing clone constraints.")
    parser.add_argument("--flush-every", type=int, default=50, help="Write partial benchmark summaries every N cases.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary simulation and TSV files.")
    parser.add_argument("--fit-workers", type=int, default=0, help="Number of independent case workers to run in parallel; 0 auto-scales for CPU runs.")
    parser.add_argument("--worker-threads", type=int, default=1, help="BLAS/Torch threads per worker process.")
    parser.add_argument("--prefetch-workers", type=int, default=0, help="CPU worker count for background simulation and TSV conversion; 0 uses all available CPUs.")
    parser.add_argument("--prefetch-buffer", type=int, default=0, help="Maximum number of prepared cases queued ahead of fitting; 0 uses 2x workers.")
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    _add_shared_fit_args(parser, em_max_iter=4, admm_max_iter=8)
    return parser


def main_single_region_benchmark(argv: list[str] | None = None) -> None:
    parser = build_single_region_benchmark_parser()
    args = parser.parse_args(argv)
    patient_df, scenario_df, global_df = run_single_region_cohort_benchmark(
        input_dir=Path(args.input_dir),
        simulation_root=Path(args.simulation_root),
        outdir=Path(args.outdir),
        lambda_grid=_parse_lambda_grid(args.lambda_grid),
        lambda_grid_mode=args.lambda_grid_mode,
        graph_k=args.graph_k,
        fit_options=_fit_options_from_args(args),
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        selection_score=args.selection_score,
        use_warm_starts=not args.disable_warm_start,
        write_patient_outputs=args.write_patient_outputs,
        bo_max_evals=args.bo_max_evals,
        bo_init_points=args.bo_init_points,
        bo_random_seed=args.bo_random_seed,
        max_files=args.max_files,
        flush_every=args.flush_every,
        reps_per_scenario=args.reps_per_scenario,
    )
    print(global_df.to_string(index=False))
    print(_aggregate_simple(patient_df, ["N_mean"]).to_string(index=False))
    print(scenario_df.head().to_string(index=False))


def main_mass_multiregion_benchmark(argv: list[str] | None = None) -> None:
    parser = build_mass_multiregion_benchmark_parser()
    args = parser.parse_args(argv)
    config = benchmark_config_from_args(args)
    patient_df, scenario_df, global_df = run_massive_multiregion_benchmark(
        config=config,
        fit_options=_fit_options_from_args(args),
        lambda_grid=_parse_lambda_grid(args.lambda_grid),
        lambda_grid_mode=args.lambda_grid_mode,
        graph_k=args.graph_k,
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        selection_score=args.selection_score,
        use_warm_starts=not args.disable_warm_start,
        write_patient_outputs=args.write_patient_outputs,
        bo_max_evals=args.bo_max_evals,
        bo_init_points=args.bo_init_points,
        bo_random_seed=args.bo_random_seed,
    )
    print(global_df.to_string(index=False))
    print(_aggregate_simple(patient_df, ["n_samples"]).to_string(index=False))
    print(scenario_df.head().to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: benchmark.py {single-region|mass-multiregion} [args...]")
        return

    mode = argv[0].strip().lower()
    mode_argv = argv[1:]
    if mode == "single-region":
        main_single_region_benchmark(mode_argv)
        return
    if mode == "mass-multiregion":
        main_mass_multiregion_benchmark(mode_argv)
        return
    raise SystemExit(f"Unknown benchmark mode: {mode}")


__all__ = [
    "benchmark_config_from_args",
    "build_mass_multiregion_benchmark_parser",
    "build_single_region_benchmark_parser",
    "main",
    "main_mass_multiregion_benchmark",
    "main_single_region_benchmark",
]
