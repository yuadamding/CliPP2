from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass
import itertools as its
import multiprocessing as mp
import os
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from ..io.conversion import convert_one_patient
from ..sim.generation import write_patient_simulation
from .benchmark_common import (
    _parse_patient_id,
    materialize_patient_df,
    write_benchmark_tables,
    write_patient_checkpoint,
)
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
    lambda_mut_list: tuple[int, ...] | None = (300, 600, 1000, 2000, 4000)
    alpha_mut: float = 10.0
    alpha_split: float = 1.0
    alpha_lambda: float = 5.0
    tau_lineage_min: float = 1.0
    tau_lineage_max: float = 50.0
    purity_conc: float = 50.0
    lineage_zero_prob: float = 0.0
    min_clone_ccf: float = 0.02
    min_clone_ccf_l2_norm: float = 0.05
    min_mutations_per_clone: int = 15
    min_clone_ccf_distance: float = 0.10
    max_rejection_tries: int = 1024
    flush_every: int = 50
    cleanup_temp: bool = True
    prefetch_workers: int = 0
    prefetch_buffer: int = 0
    fit_workers: int = 0
    worker_threads: int = 1

    @property
    def expected_cases(self) -> int:
        lambda_count = len(self.lambda_mut_list) if self.lambda_mut_list is not None else 1
        return (
            len(self.N_list)
            * len(self.purity_list)
            * len(self.amp_rate_list)
            * len(self.n_samples_list)
            * lambda_count
            * int(self.reps)
        )


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    N_mean: int
    simu_purity: float
    amp_rate: float
    n_samples: int
    lambda_mut: int
    sim: int
    child_seed: int


def _effective_prefetch_workers(config: MassiveMultiregionBenchmarkConfig) -> int:
    if config.prefetch_workers > 0:
        return int(config.prefetch_workers)
    return max(int(os.cpu_count() or 1), 1)


def _effective_prefetch_buffer(config: MassiveMultiregionBenchmarkConfig) -> int:
    if config.prefetch_buffer > 0:
        return int(config.prefetch_buffer)
    return max(2 * _effective_prefetch_workers(config), 1)


def _effective_fit_workers(
    config: MassiveMultiregionBenchmarkConfig,
    fit_options: FitOptions | None,
) -> int:
    if int(config.fit_workers) > 0:
        return int(config.fit_workers)
    del fit_options
    cpu_count = max(int(os.cpu_count() or 1), 1)
    return max(cpu_count - 2, 1)


def _configure_numeric_runtime(worker_threads: int) -> None:
    threads = max(int(worker_threads), 1)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, min(threads, 2)))
    except Exception:
        pass


def _init_benchmark_worker(worker_threads: int) -> None:
    _configure_numeric_runtime(worker_threads)


def _iter_case_specs(config: MassiveMultiregionBenchmarkConfig) -> list[BenchmarkCaseSpec]:
    rng_master = np.random.default_rng(config.seed)
    specs: list[BenchmarkCaseSpec] = []
    for N_mean, simu_purity, amp_rate, n_samples in its.product(
        config.N_list,
        config.purity_list,
        config.amp_rate_list,
        config.n_samples_list,
    ):
        lambda_values = list(config.lambda_mut_list) if config.lambda_mut_list is not None else [int(config.lambda_mut)]
        for lambda_mut in lambda_values:
            for sim in range(config.reps):
                specs.append(
                    BenchmarkCaseSpec(
                        N_mean=int(N_mean),
                        simu_purity=float(simu_purity),
                        amp_rate=float(amp_rate),
                        n_samples=int(n_samples),
                        lambda_mut=int(lambda_mut),
                        sim=int(sim),
                        child_seed=int(rng_master.integers(0, 2**32 - 1)),
                    )
                )
    return specs


def _prepare_case_artifacts(
    spec: BenchmarkCaseSpec,
    config: MassiveMultiregionBenchmarkConfig,
    temp_sim_root: str | Path,
    temp_tsv_root: str | Path,
) -> tuple[Path | None, Path]:
    child_rng = np.random.default_rng(spec.child_seed)
    patient_dir = write_patient_simulation(
        rng=child_rng,
        out_dir=temp_sim_root,
        N_mean=spec.N_mean,
        simu_purity=spec.simu_purity,
        amp_rate=spec.amp_rate,
        n_samples=spec.n_samples,
        sim=spec.sim,
        K_min=config.K_min,
        K_max=config.K_max,
        lambda_mut=spec.lambda_mut,
        alpha_mut=config.alpha_mut,
        alpha_split=config.alpha_split,
        alpha_lambda=config.alpha_lambda,
        tau_lineage_min=config.tau_lineage_min,
        tau_lineage_max=config.tau_lineage_max,
        purity_conc=config.purity_conc,
        lineage_zero_prob=config.lineage_zero_prob,
        min_clone_ccf=config.min_clone_ccf,
        min_clone_ccf_l2_norm=config.min_clone_ccf_l2_norm,
        min_mutations_per_clone=config.min_mutations_per_clone,
        min_clone_ccf_distance=config.min_clone_ccf_distance,
        max_rejection_tries=config.max_rejection_tries,
    )
    out_tsv = convert_one_patient(patient_dir, Path(temp_tsv_root))
    return out_tsv, patient_dir


def _run_case_worker(
    spec: BenchmarkCaseSpec,
    config: MassiveMultiregionBenchmarkConfig,
    fit_options: FitOptions,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    settings_profile: str,
    selection_score: str,
    use_warm_starts: bool,
    write_patient_outputs: bool,
    temp_sim_root: str | Path,
    temp_tsv_root: str | Path,
    outdir: str | Path,
) -> dict[str, int | float | str | bool] | None:
    _configure_numeric_runtime(config.worker_threads)
    out_tsv = None
    patient_dir = None
    try:
        out_tsv, patient_dir = _prepare_case_artifacts(
            spec=spec,
            config=config,
            temp_sim_root=temp_sim_root,
            temp_tsv_root=temp_tsv_root,
        )
        if out_tsv is None:
            return None

        summary = process_one_file(
            file_path=out_tsv,
            outdir=outdir,
            simulation_root=temp_sim_root,
            lambda_grid=lambda_grid,
            lambda_grid_mode=lambda_grid_mode,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            settings_profile=settings_profile,
            selection_score=selection_score,
            use_warm_starts=use_warm_starts,
            write_outputs=write_patient_outputs,
        )
        return {**_parse_patient_id(out_tsv.stem), **summary}
    finally:
        if config.cleanup_temp:
            if out_tsv is not None:
                Path(out_tsv).unlink(missing_ok=True)
            if patient_dir is not None:
                shutil.rmtree(patient_dir, ignore_errors=True)


def run_massive_multiregion_benchmark(
    config: MassiveMultiregionBenchmarkConfig,
    fit_options: FitOptions | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    bic_df_scale: float = 8.0,
    bic_cluster_penalty: float = 4.0,
    settings_profile: str = "manual",
    selection_score: str = "ebic",
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
        fit_options = FitOptions(lambda_value=0.0)
    _configure_numeric_runtime(config.worker_threads)

    patient_rows: list[dict[str, int | float | str | bool]] = []
    total_cases = config.expected_cases
    start_time = perf_counter()
    case_index = 0
    specs = _iter_case_specs(config)
    fit_worker_count = _effective_fit_workers(config, fit_options)
    print(
        f"[multiregion-benchmark] starting {total_cases} cases "
        f"| fit_workers={fit_worker_count} | worker_threads={int(config.worker_threads)}"
    )

    def _flush_checkpoint() -> None:
        write_patient_checkpoint(
            patient_rows,
            outdir=outdir,
            start_time=start_time,
            case_index=case_index,
            total_cases=total_cases,
            label="multiregion-benchmark",
        )

    def _record_case(out_tsv: Path | None, patient_dir: Path) -> None:
        nonlocal case_index
        if out_tsv is None:
            if config.cleanup_temp:
                shutil.rmtree(patient_dir, ignore_errors=True)
            return

        summary = process_one_file(
            file_path=out_tsv,
            outdir=outdir,
            simulation_root=temp_sim_root,
            lambda_grid=lambda_grid,
            lambda_grid_mode=lambda_grid_mode,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            settings_profile=settings_profile,
            selection_score=selection_score,
            use_warm_starts=use_warm_starts,
            write_outputs=write_patient_outputs,
        )
        patient_rows.append({**_parse_patient_id(out_tsv.stem), **summary})
        case_index += 1

        if config.cleanup_temp:
            out_tsv.unlink(missing_ok=True)
            shutil.rmtree(patient_dir, ignore_errors=True)

        if case_index % max(config.flush_every, 1) == 0 or case_index == total_cases:
            _flush_checkpoint()

    if fit_worker_count > 1:
        inflight_limit = max(int(fit_worker_count) * 8, 1)
        spec_iter = iter(specs)
        with cf.ProcessPoolExecutor(
            max_workers=int(fit_worker_count),
            mp_context=mp.get_context("spawn"),
            initializer=_init_benchmark_worker,
            initargs=(int(config.worker_threads),),
        ) as executor:
            future_map: dict[cf.Future[dict[str, int | float | str | bool] | None], BenchmarkCaseSpec] = {}

            while len(future_map) < inflight_limit:
                try:
                    spec = next(spec_iter)
                except StopIteration:
                    break
                future = executor.submit(
                    _run_case_worker,
                    spec,
                    config,
                    fit_options,
                    lambda_grid,
                    lambda_grid_mode,
                    bic_df_scale,
                    bic_cluster_penalty,
                    settings_profile,
                    selection_score,
                    use_warm_starts,
                    write_patient_outputs,
                    temp_sim_root,
                    temp_tsv_root,
                    outdir,
                )
                future_map[future] = spec

            while future_map:
                done, _ = cf.wait(future_map, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    future_map.pop(future)
                    row = future.result()
                    if row is not None:
                        patient_rows.append(row)
                        case_index += 1
                        if case_index % max(config.flush_every, 1) == 0 or case_index == total_cases:
                            _flush_checkpoint()

                    while len(future_map) < inflight_limit:
                        try:
                            spec = next(spec_iter)
                        except StopIteration:
                            break
                        next_future = executor.submit(
                            _run_case_worker,
                            spec,
                            config,
                            fit_options,
                            lambda_grid,
                            lambda_grid_mode,
                            bic_df_scale,
                            bic_cluster_penalty,
                            settings_profile,
                            selection_score,
                            use_warm_starts,
                            write_patient_outputs,
                            temp_sim_root,
                            temp_tsv_root,
                            outdir,
                        )
                        future_map[next_future] = spec
    else:
        prefetch_workers = _effective_prefetch_workers(config)
        if prefetch_workers <= 1:
            for spec in specs:
                out_tsv, patient_dir = _prepare_case_artifacts(
                    spec=spec,
                    config=config,
                    temp_sim_root=temp_sim_root,
                    temp_tsv_root=temp_tsv_root,
                )
                _record_case(out_tsv=out_tsv, patient_dir=patient_dir)
        else:
            prefetch_buffer = _effective_prefetch_buffer(config)
            spec_iter = iter(specs)
            with cf.ProcessPoolExecutor(max_workers=prefetch_workers) as executor:
                future_map: dict[cf.Future[tuple[Path | None, Path]], BenchmarkCaseSpec] = {}

                while len(future_map) < prefetch_buffer:
                    try:
                        spec = next(spec_iter)
                    except StopIteration:
                        break
                    future = executor.submit(
                        _prepare_case_artifacts,
                        spec,
                        config,
                        temp_sim_root,
                        temp_tsv_root,
                    )
                    future_map[future] = spec

                while future_map:
                    done, _ = cf.wait(future_map, return_when=cf.FIRST_COMPLETED)
                    for future in done:
                        future_map.pop(future)
                        out_tsv, patient_dir = future.result()
                        _record_case(out_tsv=out_tsv, patient_dir=patient_dir)

                        while len(future_map) < prefetch_buffer:
                            try:
                                spec = next(spec_iter)
                            except StopIteration:
                                break
                            next_future = executor.submit(
                                _prepare_case_artifacts,
                                spec,
                                config,
                                temp_sim_root,
                                temp_tsv_root,
                            )
                            future_map[next_future] = spec

    patient_df = materialize_patient_df(patient_rows)
    scenario_df, global_df = write_benchmark_tables(patient_df, outdir)

    if config.cleanup_temp:
        shutil.rmtree(temp_sim_root, ignore_errors=True)
        shutil.rmtree(temp_tsv_root, ignore_errors=True)

    return patient_df, scenario_df, global_df


__all__ = [
    "BenchmarkCaseSpec",
    "MassiveMultiregionBenchmarkConfig",
    "run_massive_multiregion_benchmark",
]
