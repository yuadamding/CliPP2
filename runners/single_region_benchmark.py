from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from .pipeline import process_one_file


SINGLE_REGION_PATTERN = re.compile(
    r"(?P<N_mean>\d+)_(?P<true_K>\d+)_(?P<purity>0(?:\.\d+)?)_(?P<amp_rate>0(?:\.\d+)?)_rep(?P<rep>\d+)"
)


def _parse_single_region_patient_id(patient_id: str) -> dict[str, int | float | str]:
    match = SINGLE_REGION_PATTERN.fullmatch(patient_id)
    if match is None:
        raise ValueError(f"Patient id does not match single-region simulation pattern: {patient_id}")

    return {
        "patient_id": patient_id,
        "N_mean": int(match.group("N_mean")),
        "true_K": int(match.group("true_K")),
        "purity": float(match.group("purity")),
        "amp_rate": float(match.group("amp_rate")),
        "n_samples": 1,
        "rep": int(match.group("rep")),
    }


def _aggregate_single_region(patient_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        patient_df.groupby(group_cols, dropna=False)
        .agg(
            n_patients=("patient_id", "size"),
            n_evaluable_patients=("ARI", lambda s: int(s.notna().sum())),
            mean_selected_lambda=("selected_lambda", "mean"),
            mean_estimated_clusters=("n_clusters", "mean"),
            mean_ARI=("ARI", "mean"),
            median_ARI=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_multiplicity_accuracy=("multiplicity_accuracy", "mean"),
            mean_eval_mutations=("n_eval_mutations", "mean"),
            mean_filtered_mutations=("n_filtered_mutations", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def _select_single_region_files(
    input_dir: Path,
    reps_per_scenario: int | None,
) -> list[Path]:
    files = sorted(input_dir.glob("*.tsv"))
    if reps_per_scenario is None:
        return files

    grouped: dict[tuple[int, int, float, float], list[Path]] = {}
    for file_path in files:
        meta = _parse_single_region_patient_id(file_path.stem)
        key = (
            int(meta["N_mean"]),
            int(meta["true_K"]),
            float(meta["purity"]),
            float(meta["amp_rate"]),
        )
        grouped.setdefault(key, []).append(file_path)

    selected: list[Path] = []
    for key in sorted(grouped):
        scenario_files = grouped[key]
        count = min(max(int(reps_per_scenario), 1), len(scenario_files))
        if count == 1:
            selected.append(scenario_files[0])
            continue
        pick_idx = np.linspace(0, len(scenario_files) - 1, num=count, dtype=int)
        selected.extend(scenario_files[idx] for idx in sorted(set(pick_idx.tolist())))
    return selected


def _global_single_region(patient_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n_patients": int(patient_df.shape[0]),
                "n_evaluable_patients": int(patient_df["ARI"].notna().sum()),
                "mean_ARI": float(patient_df["ARI"].mean()),
                "median_ARI": float(patient_df["ARI"].median()),
                "mean_cp_rmse": float(patient_df["cp_rmse"].mean()),
                "mean_multiplicity_accuracy": float(patient_df["multiplicity_accuracy"].mean()),
                "mean_estimated_clusters": float(patient_df["n_clusters"].mean()),
                "mean_eval_mutations": float(patient_df["n_eval_mutations"].mean()),
                "mean_filtered_mutations": float(patient_df["n_filtered_mutations"].mean()),
                "mean_elapsed_seconds": float(patient_df["elapsed_seconds"].mean()),
            }
        ]
    )


def run_single_region_cohort_benchmark(
    input_dir: str | Path,
    simulation_root: str | Path,
    outdir: str | Path,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "coarse_no_zero",
    graph_k: int = 8,
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 10.0,
    bic_cluster_penalty: float = 6.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = False,
    max_files: int | None = None,
    flush_every: int = 100,
    reps_per_scenario: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    simulation_root = Path(simulation_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = _select_single_region_files(input_dir=input_dir, reps_per_scenario=reps_per_scenario)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    if not files:
        raise RuntimeError(f"No TSV files found in {input_dir}")

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0, device="cuda")

    patient_rows: list[dict[str, int | float | str | bool]] = []
    total_files = len(files)
    start_time = perf_counter()

    for case_index, file_path in enumerate(files, start=1):
        summary = process_one_file(
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
            write_outputs=write_patient_outputs,
        )
        patient_rows.append({**_parse_single_region_patient_id(file_path.stem), **summary})

        if case_index % max(flush_every, 1) == 0 or case_index == total_files:
            patient_df = pd.DataFrame(patient_rows).sort_values(
                ["N_mean", "true_K", "purity", "amp_rate", "rep"]
            ).reset_index(drop=True)
            patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
            elapsed = perf_counter() - start_time
            rate = case_index / max(elapsed, 1e-9)
            remaining = max(total_files - case_index, 0)
            eta_seconds = remaining / max(rate, 1e-9)
            print(
                f"[single-region-benchmark] {case_index}/{total_files} cases "
                f"| elapsed={elapsed/60.0:.1f} min | rate={rate:.2f} cases/s "
                f"| eta={eta_seconds/60.0:.1f} min"
            )

    patient_df = pd.DataFrame(patient_rows).sort_values(["N_mean", "true_K", "purity", "amp_rate", "rep"]).reset_index(drop=True)
    scenario_df = _aggregate_single_region(patient_df, ["N_mean", "true_K", "purity", "amp_rate"])
    depth_df = _aggregate_single_region(patient_df, ["N_mean"])
    true_k_df = _aggregate_single_region(patient_df, ["true_K"])
    purity_df = _aggregate_single_region(patient_df, ["purity"])
    amp_rate_df = _aggregate_single_region(patient_df, ["amp_rate"])
    global_df = _global_single_region(patient_df)

    patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
    scenario_df.to_csv(outdir / "benchmark_by_scenario.tsv", sep="\t", index=False)
    depth_df.to_csv(outdir / "benchmark_by_depth.tsv", sep="\t", index=False)
    true_k_df.to_csv(outdir / "benchmark_by_true_k.tsv", sep="\t", index=False)
    purity_df.to_csv(outdir / "benchmark_by_purity.tsv", sep="\t", index=False)
    amp_rate_df.to_csv(outdir / "benchmark_by_amp_rate.tsv", sep="\t", index=False)
    global_df.to_csv(outdir / "benchmark_global.tsv", sep="\t", index=False)
    return patient_df, scenario_df, global_df


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multi-region clipp single-region-benchmark",
        description="Benchmark multi-region clipp on a single-region simulation cohort.",
    )
    parser.add_argument("--input-dir", default="/data/CliPP_Sim/CliPPSim4K_PyClone", help="Directory with per-patient TSV files.")
    parser.add_argument("--simulation-root", default="/data/CliPP_Sim/CliPPSim4K", help="Root directory with single-region truth folders.")
    parser.add_argument("--outdir", default="multi_region_clipp_single_region_benchmark", help="Output directory.")
    parser.add_argument("--lambda-grid", default=None, help="Optional comma-separated lambda grid.")
    parser.add_argument(
        "--lambda-grid-mode",
        choices=["standard", "dense", "dense_no_zero", "coarse_no_zero"],
        default="coarse_no_zero",
        help="Automatic lambda grid template used when --lambda-grid is not provided.",
    )
    parser.add_argument("--graph-k", type=int, default=8, help="k for the mutation kNN graph.")
    parser.add_argument("--device", default="cuda", help="Torch device: auto, cuda, or cpu.")
    parser.add_argument("--em-max-iter", type=int, default=8, help="Maximum EM iterations.")
    parser.add_argument("--admm-max-iter", type=int, default=20, help="Maximum ADMM iterations per M-step.")
    parser.add_argument("--inner-steps", type=int, default=2, help="Quadratic-majorization updates per ADMM iteration.")
    parser.add_argument("--inner-lr", type=float, default=5e-2, help="Deprecated compatibility knob; no longer used by the ADMM solver.")
    parser.add_argument("--cg-max-iter", type=int, default=30, help="Maximum conjugate-gradient iterations for each ADMM w-update.")
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
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on the number of files processed.")
    parser.add_argument("--flush-every", type=int, default=100, help="Write partial patient summaries every N cases.")
    parser.add_argument("--reps-per-scenario", type=int, default=None, help="Optional balanced subsampling count per single-region scenario.")
    parser.add_argument("--verbose", action="store_true", help="Print EM and ADMM progress.")
    return parser


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
    patient_df, scenario_df, global_df = run_single_region_cohort_benchmark(
        input_dir=Path(args.input_dir),
        simulation_root=Path(args.simulation_root),
        outdir=Path(args.outdir),
        lambda_grid=_parse_lambda_grid(args.lambda_grid),
        lambda_grid_mode=args.lambda_grid_mode,
        graph_k=args.graph_k,
        fit_options=fit_options_from_args(args),
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        settings_profile=args.settings_profile,
        use_warm_starts=not args.disable_warm_start,
        write_patient_outputs=args.write_patient_outputs,
        max_files=args.max_files,
        flush_every=args.flush_every,
        reps_per_scenario=args.reps_per_scenario,
    )
    print(global_df.to_string(index=False))
    print(_aggregate_single_region(patient_df, ["N_mean"]).to_string(index=False))
    print(scenario_df.head().to_string(index=False))


__all__ = [
    "SINGLE_REGION_PATTERN",
    "build_parser",
    "fit_options_from_args",
    "main",
    "run_single_region_cohort_benchmark",
]
