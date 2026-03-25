from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitOptions
from .pipeline import process_one_file


PATIENT_PATTERN = re.compile(
    r"(?P<N_mean>\d+)_(?P<true_K>\d+)_(?P<purity>0\.\d+)_(?P<amp_rate>0\.\d+)_S(?P<n_samples>\d+)_M(?P<n_mutations>\d+)_rep(?P<rep>\d+)"
)


def _parse_patient_id(patient_id: str) -> dict[str, int | float | str]:
    match = PATIENT_PATTERN.fullmatch(patient_id)
    if match is None:
        raise ValueError(f"Patient id does not match simulation pattern: {patient_id}")

    return {
        "patient_id": patient_id,
        "N_mean": int(match.group("N_mean")),
        "true_K": int(match.group("true_K")),
        "purity": float(match.group("purity")),
        "amp_rate": float(match.group("amp_rate")),
        "n_samples": int(match.group("n_samples")),
        "n_mutations": int(match.group("n_mutations")),
        "rep": int(match.group("rep")),
    }


def _select_representative_files(input_dir: Path, reps_per_scenario: int) -> list[Path]:
    return _select_representative_files_with_filter(
        input_dir=input_dir,
        reps_per_scenario=reps_per_scenario,
        n_mean_values=None,
    )


def _select_representative_files_with_filter(
    input_dir: Path,
    reps_per_scenario: int,
    n_mean_values: list[int] | None,
) -> list[Path]:
    grouped: dict[tuple[int, float, float, int], list[Path]] = {}
    for file_path in sorted(input_dir.glob("*.tsv")):
        meta = _parse_patient_id(file_path.stem)
        if n_mean_values is not None and int(meta["N_mean"]) not in set(n_mean_values):
            continue
        key = (
            int(meta["N_mean"]),
            float(meta["purity"]),
            float(meta["amp_rate"]),
            int(meta["n_samples"]),
        )
        grouped.setdefault(key, []).append(file_path)

    selected: list[Path] = []
    for key in sorted(grouped):
        files = grouped[key]
        count = min(max(reps_per_scenario, 1), len(files))
        pick_idx = np.linspace(0, len(files) - 1, num=count, dtype=int)
        selected.extend(files[idx] for idx in sorted(set(pick_idx.tolist())))
    return selected


def _aggregate_patient_results(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_df = (
        patient_df.groupby(["N_mean", "purity", "amp_rate", "n_samples"], dropna=False)
        .agg(
            n_patients=("patient_id", "size"),
            mean_true_K=("true_K", "mean"),
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
        .sort_values(["N_mean", "purity", "amp_rate", "n_samples"])
    )

    global_df = pd.DataFrame(
        [
            {
                "n_patients": int(patient_df.shape[0]),
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
    return scenario_df, global_df


def run_simulation_benchmark(
    input_dir: str | Path,
    simulation_root: str | Path,
    outdir: str | Path,
    reps_per_scenario: int = 1,
    n_mean_values: list[int] | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "dense_no_zero",
    graph_k: int = 8,
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 10.0,
    bic_cluster_penalty: float = 6.0,
    settings_profile: str = "auto",
    use_warm_starts: bool = True,
    write_patient_outputs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    input_dir = Path(input_dir)
    simulation_root = Path(simulation_root)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0, device="cuda")

    selected_files = _select_representative_files_with_filter(
        input_dir=input_dir,
        reps_per_scenario=reps_per_scenario,
        n_mean_values=n_mean_values,
    )
    patient_rows: list[dict[str, int | float | str | bool]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        for file_path in selected_files:
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
            patient_rows.append({**_parse_patient_id(file_path.stem), **summary})

    patient_df = pd.DataFrame(patient_rows).sort_values(["N_mean", "purity", "amp_rate", "n_samples", "rep"]).reset_index(drop=True)
    scenario_df, global_df = _aggregate_patient_results(patient_df)

    patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
    scenario_df.to_csv(outdir / "benchmark_by_scenario.tsv", sep="\t", index=False)
    global_df.to_csv(outdir / "benchmark_global.tsv", sep="\t", index=False)
    return patient_df, scenario_df, global_df
