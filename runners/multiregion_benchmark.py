from __future__ import annotations

import argparse
import itertools as its
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from ..io.conversion import convert_one_patient
from ..sim.generation import parse_float_list, parse_int_list, write_patient_simulation
from .benchmark import _aggregate_patient_results, _parse_patient_id
from .pipeline import process_one_file


@dataclass(frozen=True)
class MassiveMultiregionBenchmarkConfig:
    outdir: str | Path = "multi_region_clipp_multiregion_benchmark"
    temp_sim_root: str | Path | None = None
    temp_tsv_root: str | Path | None = None
    purity_list: tuple[float, ...] = (0.3, 0.6, 0.9)
    amp_rate_list: tuple[float, ...] = (0.1, 0.2)
    N_list: tuple[int, ...] = (50, 75, 100, 200, 300, 400, 500, 1000)
    n_samples_list: tuple[int, ...] = (5, 10, 15)
    reps: int = 70
    seed: int | None = None
    K_min: int = 5
    K_max: int = 10
    lambda_mut: int = 2000
    alpha_mut: float = 10.0
    alpha_split: float = 1.0
    alpha_lambda: float = 5.0
    tau_lineage_min: float = 1.0
    tau_lineage_max: float = 50.0
    purity_conc: float = 50.0
    lineage_zero_prob: float = 0.3
    flush_every: int = 50
    cleanup_temp: bool = True

    @property
    def expected_cases(self) -> int:
        return (
            len(self.N_list)
            * len(self.purity_list)
            * len(self.amp_rate_list)
            * len(self.n_samples_list)
            * int(self.reps)
        )


def _aggregate_simple(patient_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        patient_df.groupby(columns, dropna=False)
        .agg(
            n_patients=("patient_id", "size"),
            mean_true_K=("true_K", "mean"),
            mean_estimated_clusters=("n_clusters", "mean"),
            mean_ARI=("ARI", "mean"),
            median_ARI=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_multiplicity_accuracy=("multiplicity_accuracy", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .reset_index()
        .sort_values(columns)
    )


def run_massive_multiregion_benchmark(
    config: MassiveMultiregionBenchmarkConfig,
    fit_options: FitOptions | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    graph_k: int = 8,
    bic_df_scale: float = 10.0,
    bic_cluster_penalty: float = 6.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    temp_sim_root = Path(config.temp_sim_root) if config.temp_sim_root is not None else outdir / "_tmp_sim"
    temp_tsv_root = Path(config.temp_tsv_root) if config.temp_tsv_root is not None else outdir / "_tmp_tsv"
    temp_sim_root.mkdir(parents=True, exist_ok=True)
    temp_tsv_root.mkdir(parents=True, exist_ok=True)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0, device="cuda")

    rng_master = np.random.default_rng(config.seed)
    patient_rows: list[dict[str, int | float | str | bool]] = []
    total_cases = config.expected_cases
    start_time = perf_counter()
    case_index = 0

    for N_mean, simu_purity, amp_rate, n_samples in its.product(
        config.N_list,
        config.purity_list,
        config.amp_rate_list,
        config.n_samples_list,
    ):
        for sim in range(config.reps):
            child_seed = int(rng_master.integers(0, 2**32 - 1))
            child_rng = np.random.default_rng(child_seed)

            patient_dir = write_patient_simulation(
                rng=child_rng,
                out_dir=temp_sim_root,
                N_mean=N_mean,
                simu_purity=simu_purity,
                amp_rate=amp_rate,
                n_samples=n_samples,
                sim=sim,
                K_min=config.K_min,
                K_max=config.K_max,
                lambda_mut=config.lambda_mut,
                alpha_mut=config.alpha_mut,
                alpha_split=config.alpha_split,
                alpha_lambda=config.alpha_lambda,
                tau_lineage_min=config.tau_lineage_min,
                tau_lineage_max=config.tau_lineage_max,
                purity_conc=config.purity_conc,
                lineage_zero_prob=config.lineage_zero_prob,
            )
            out_tsv = convert_one_patient(patient_dir, temp_tsv_root)
            if out_tsv is None:
                if config.cleanup_temp:
                    shutil.rmtree(patient_dir, ignore_errors=True)
                continue

            summary = process_one_file(
                file_path=out_tsv,
                outdir=outdir,
                simulation_root=temp_sim_root,
                lambda_grid=lambda_grid,
                lambda_grid_mode=lambda_grid_mode,
                graph_k=graph_k,
                fit_options=fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                settings_profile=settings_profile,
                use_warm_starts=use_warm_starts,
                write_outputs=write_patient_outputs,
            )
            patient_rows.append({**_parse_patient_id(out_tsv.stem), **summary})
            case_index += 1

            if config.cleanup_temp:
                out_tsv.unlink(missing_ok=True)
                shutil.rmtree(patient_dir, ignore_errors=True)

            if case_index % max(config.flush_every, 1) == 0 or case_index == total_cases:
                patient_df = pd.DataFrame(patient_rows).sort_values(
                    ["N_mean", "purity", "amp_rate", "n_samples", "rep"]
                ).reset_index(drop=True)
                patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
                elapsed = perf_counter() - start_time
                rate = case_index / max(elapsed, 1e-9)
                remaining = max(total_cases - case_index, 0)
                eta_seconds = remaining / max(rate, 1e-9)
                print(
                    f"[multiregion-benchmark] {case_index}/{total_cases} cases "
                    f"| elapsed={elapsed/60.0:.1f} min | rate={rate:.2f} cases/s "
                    f"| eta={eta_seconds/60.0:.1f} min"
                )

    patient_df = pd.DataFrame(patient_rows).sort_values(["N_mean", "purity", "amp_rate", "n_samples", "rep"]).reset_index(drop=True)
    scenario_df, global_df = _aggregate_patient_results(patient_df)
    by_samples_df = _aggregate_simple(patient_df, ["n_samples"])
    by_depth_df = _aggregate_simple(patient_df, ["N_mean"])
    by_true_k_df = _aggregate_simple(patient_df, ["true_K"])

    patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
    scenario_df.to_csv(outdir / "benchmark_by_scenario.tsv", sep="\t", index=False)
    global_df.to_csv(outdir / "benchmark_global.tsv", sep="\t", index=False)
    by_samples_df.to_csv(outdir / "benchmark_by_n_samples.tsv", sep="\t", index=False)
    by_depth_df.to_csv(outdir / "benchmark_by_depth.tsv", sep="\t", index=False)
    by_true_k_df.to_csv(outdir / "benchmark_by_true_k.tsv", sep="\t", index=False)

    if config.cleanup_temp:
        shutil.rmtree(temp_sim_root, ignore_errors=True)
        shutil.rmtree(temp_tsv_root, ignore_errors=True)

    return patient_df, scenario_df, global_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-region clipp multiregion-benchmark",
        description="Benchmark multi-region clipp on large multiregion simulation cohorts."
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
    parser.add_argument("--lambda-mut", type=int, default=2000, help="Poisson mean for mutation count.")
    parser.add_argument("--alpha-mut", type=float, default=10.0, help="Dirichlet concentration for mutation allocation.")
    parser.add_argument("--alpha-lambda", type=float, default=5.0, help="Dirichlet concentration for lineage residual masses.")
    parser.add_argument("--tau-lineage-min", type=float, default=1.0, help="Minimum lineage concentration per sample.")
    parser.add_argument("--tau-lineage-max", type=float, default=50.0, help="Maximum lineage concentration per sample.")
    parser.add_argument("--purity-conc", type=float, default=50.0, help="Beta concentration for sample purities.")
    parser.add_argument("--lineage-zero-prob", type=float, default=0.3, help="Probability of zeroing a lineage in a sample.")
    parser.add_argument("--flush-every", type=int, default=50, help="Write partial benchmark summaries every N cases.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary simulation and TSV files.")
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero"],
        default="dense_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    parser.add_argument("--graph-k", type=int, default=8, help="k for the mutation kNN graph.")
    parser.add_argument("--device", default="cuda", help="Torch device: auto, cuda, or cpu.")
    parser.add_argument("--em-max-iter", type=int, default=4, help="Maximum EM iterations for the large benchmark.")
    parser.add_argument("--admm-max-iter", type=int, default=8, help="Maximum ADMM iterations per M-step.")
    parser.add_argument("--inner-steps", type=int, default=2, help="Quadratic-majorization updates per ADMM iteration.")
    parser.add_argument("--inner-lr", type=float, default=5e-2, help="Deprecated compatibility knob; no longer used by the ADMM solver.")
    parser.add_argument("--cg-max-iter", type=int, default=50, help="Maximum conjugate-gradient iterations for each ADMM w-update.")
    parser.add_argument("--cg-tol", type=float, default=1e-4, help="Relative conjugate-gradient tolerance.")
    parser.add_argument("--curvature-floor", type=float, default=1e-4, help="Minimum diagonal curvature used in the quadratic surrogate.")
    parser.add_argument("--admm-rho", type=float, default=2.0, help="ADMM penalty parameter.")
    parser.add_argument("--em-tol", type=float, default=1e-4, help="Relative EM stopping tolerance.")
    parser.add_argument("--admm-tol", type=float, default=5e-3, help="ADMM stopping tolerance.")
    parser.add_argument("--fused-tol", type=float, default=1e-3, help="Tolerance for calling an edge fused.")
    parser.add_argument("--center-merge-tol", type=float, default=1e-1, help="Cluster-center merge threshold.")
    parser.add_argument("--bic-df-scale", type=float, default=10.0, help="Extended BIC CP degrees-of-freedom scale.")
    parser.add_argument("--bic-cluster-penalty", type=float, default=6.0, help="Extended BIC cluster-count penalty.")
    parser.add_argument(
        "--settings-profile",
        choices=["manual", "auto"],
        default="auto",
        help="Use fixed manual settings or the rule-based auto profile learned from simulation benchmarks.",
    )
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability for major-copy multiplicity.")
    parser.add_argument("--disable-warm-start", action="store_true", help="Disable lambda-path warm starts.")
    parser.add_argument("--write-patient-outputs", action="store_true", help="Write full per-patient result files.")
    parser.add_argument("--verbose", action="store_true", help="Print EM and ADMM progress.")
    return parser


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


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
        alpha_mut=args.alpha_mut,
        alpha_lambda=args.alpha_lambda,
        tau_lineage_min=args.tau_lineage_min,
        tau_lineage_max=args.tau_lineage_max,
        purity_conc=args.purity_conc,
        lineage_zero_prob=args.lineage_zero_prob,
        flush_every=args.flush_every,
        cleanup_temp=not args.keep_temp,
    )


def fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = benchmark_config_from_args(args)
    patient_df, scenario_df, global_df = run_massive_multiregion_benchmark(
        config=config,
        fit_options=fit_options_from_args(args),
        lambda_grid=_parse_lambda_grid(args.lambda_grid),
        lambda_grid_mode=args.lambda_grid_mode,
        graph_k=args.graph_k,
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        use_warm_starts=not args.disable_warm_start,
        write_patient_outputs=args.write_patient_outputs,
    )
    print(global_df.to_string(index=False))
    print(_aggregate_simple(patient_df, ["n_samples"]).to_string(index=False))
    print(scenario_df.head().to_string(index=False))


__all__ = [
    "MassiveMultiregionBenchmarkConfig",
    "benchmark_config_from_args",
    "build_parser",
    "fit_options_from_args",
    "main",
    "run_massive_multiregion_benchmark",
]
