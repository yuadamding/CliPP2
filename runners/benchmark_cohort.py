from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path
from time import perf_counter
import warnings

import pandas as pd

from ..core.model import FitOptions
from .benchmark_common import (
    _parse_patient_id,
    _select_representative_files_with_filter,
    materialize_patient_df,
    parse_cohort_patient_id,
    write_benchmark_tables,
    write_patient_checkpoint,
)
from .pipeline import process_one_file


def run_cohort_benchmark(
    input_dir: str | Path,
    simulation_root: str | Path,
    outdir: str | Path,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = False,
    max_files: int | None = None,
    flush_every: int = 100,
    reps_per_scenario: int | None = None,
    n_mean_values: list[int] | None = None,
    workers: int = 1,
    missing_cna_policy: str = "error",
    patient_id_parser=parse_cohort_patient_id,
    selection_group_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    simulation_root = Path(simulation_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if reps_per_scenario is None:
        files = sorted(input_dir.glob("*.tsv"))
        if n_mean_values is not None:
            n_mean_set = {int(value) for value in n_mean_values}
            files = [file_path for file_path in files if int(patient_id_parser(file_path.stem)["N_mean"]) in n_mean_set]
    else:
        files = _select_representative_files_with_filter(
            input_dir=input_dir,
            reps_per_scenario=reps_per_scenario,
            n_mean_values=n_mean_values,
            patient_id_parser=patient_id_parser,
            selection_group_cols=selection_group_cols,
        )

    if max_files is not None:
        files = files[: max(0, int(max_files))]
    if not files:
        raise RuntimeError(f"No TSV files found in {input_dir}")

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)

    patient_rows: list[dict[str, int | float | str | bool]] = []
    total_files = len(files)
    start_time = perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        worker_count = max(int(workers), 1)
        if worker_count <= 1:
            for case_index, file_path in enumerate(files, start=1):
                summary = process_one_file(
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
                    write_outputs=write_patient_outputs,
                    missing_cna_policy=missing_cna_policy,
                )
                patient_rows.append({**patient_id_parser(file_path.stem), **summary})

                if case_index % max(flush_every, 1) == 0 or case_index == total_files:
                    write_patient_checkpoint(
                        patient_rows,
                        outdir=outdir,
                        start_time=start_time,
                        case_index=case_index,
                        total_cases=total_files,
                        label="cohort-benchmark",
                    )
        else:
            with cf.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_map = {
                    executor.submit(
                        process_one_file,
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
                        write_outputs=write_patient_outputs,
                        missing_cna_policy=missing_cna_policy,
                    ): file_path
                    for file_path in files
                }
                for case_index, future in enumerate(cf.as_completed(future_map), start=1):
                    file_path = future_map[future]
                    summary = future.result()
                    patient_rows.append({**patient_id_parser(file_path.stem), **summary})
                    if case_index % max(flush_every, 1) == 0 or case_index == total_files:
                        write_patient_checkpoint(
                            patient_rows,
                            outdir=outdir,
                            start_time=start_time,
                            case_index=case_index,
                            total_cases=total_files,
                            label="cohort-benchmark",
                        )

    patient_df = materialize_patient_df(patient_rows)
    scenario_df, global_df = write_benchmark_tables(patient_df, outdir)
    return patient_df, scenario_df, global_df


def run_simulation_benchmark(
    input_dir: str | Path,
    simulation_root: str | Path,
    outdir: str | Path,
    reps_per_scenario: int = 1,
    n_mean_values: list[int] | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = True,
    workers: int = 1,
    missing_cna_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_cohort_benchmark(
        input_dir=input_dir,
        simulation_root=simulation_root,
        outdir=outdir,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        settings_profile=settings_profile,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        write_patient_outputs=write_patient_outputs,
        workers=workers,
        missing_cna_policy=missing_cna_policy,
        reps_per_scenario=reps_per_scenario,
        n_mean_values=n_mean_values,
        patient_id_parser=_parse_patient_id,
        selection_group_cols=["N_mean", "true_K", "purity", "amp_rate", "n_samples", "lambda_mut_setting"],
    )


def run_single_region_cohort_benchmark(
    input_dir: str | Path,
    simulation_root: str | Path,
    outdir: str | Path,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = False,
    max_files: int | None = None,
    flush_every: int = 100,
    reps_per_scenario: int | None = None,
    workers: int = 1,
    missing_cna_policy: str = "error",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_cohort_benchmark(
        input_dir=input_dir,
        simulation_root=simulation_root,
        outdir=outdir,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        settings_profile=settings_profile,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        write_patient_outputs=write_patient_outputs,
        max_files=max_files,
        flush_every=flush_every,
        workers=workers,
        missing_cna_policy=missing_cna_policy,
        reps_per_scenario=reps_per_scenario,
        selection_group_cols=["N_mean", "true_K", "purity", "amp_rate", "n_samples", "lambda_mut_setting"],
    )


__all__ = [
    "run_cohort_benchmark",
    "run_simulation_benchmark",
    "run_single_region_cohort_benchmark",
]
