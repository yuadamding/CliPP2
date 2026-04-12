from __future__ import annotations

import argparse
import re
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import FitOptions


PATIENT_PATTERN = re.compile(
    r"(?P<N_mean>\d+)_(?P<true_K>\d+)_(?P<purity>0\.\d+)_(?P<amp_rate>0\.\d+)_S(?P<n_samples>\d+)(?:_Lm(?P<lambda_mut_setting>\d+))?_M(?P<n_mutations>\d+)_rep(?P<rep>\d+)"
)
SINGLE_REGION_PATTERN = re.compile(
    r"(?P<N_mean>\d+)_(?P<true_K>\d+)_(?P<purity>0(?:\.\d+)?)_(?P<amp_rate>0(?:\.\d+)?)_rep(?P<rep>\d+)"
)

DEFAULT_PATIENT_SORT_COLUMNS = ["N_mean", "purity", "amp_rate", "n_samples", "true_K", "rep"]
DEFAULT_TUMOR_SORT_COLUMNS = DEFAULT_PATIENT_SORT_COLUMNS


def _safe_nanmean_abs(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.abs(array[finite])))


def _parse_patient_id(patient_id: str) -> dict[str, int | float | str]:
    match = PATIENT_PATTERN.fullmatch(patient_id)
    if match is None:
        raise ValueError(f"Tumor id does not match simulation pattern: {patient_id}")

    return {
        "patient_id": patient_id,
        "tumor_id": patient_id,
        "N_mean": int(match.group("N_mean")),
        "true_K": int(match.group("true_K")),
        "purity": float(match.group("purity")),
        "amp_rate": float(match.group("amp_rate")),
        "n_samples": int(match.group("n_samples")),
        "n_regions": int(match.group("n_samples")),
        "lambda_mut_setting": (
            int(match.group("lambda_mut_setting"))
            if match.group("lambda_mut_setting") is not None
            else np.nan
        ),
        "n_mutations": int(match.group("n_mutations")),
        "rep": int(match.group("rep")),
    }


def parse_single_region_patient_id(patient_id: str) -> dict[str, int | float | str]:
    match = SINGLE_REGION_PATTERN.fullmatch(patient_id)
    if match is None:
        raise ValueError(f"Tumor id does not match single-region simulation pattern: {patient_id}")

    return {
        "patient_id": patient_id,
        "tumor_id": patient_id,
        "N_mean": int(match.group("N_mean")),
        "true_K": int(match.group("true_K")),
        "purity": float(match.group("purity")),
        "amp_rate": float(match.group("amp_rate")),
        "n_samples": 1,
        "n_regions": 1,
        "lambda_mut_setting": np.nan,
        "rep": int(match.group("rep")),
    }


def parse_cohort_patient_id(patient_id: str) -> dict[str, int | float | str]:
    try:
        return _parse_patient_id(patient_id)
    except ValueError:
        return parse_single_region_patient_id(patient_id)


def parse_single_region_tumor_id(tumor_id: str) -> dict[str, int | float | str]:
    return parse_single_region_patient_id(tumor_id)


def parse_cohort_tumor_id(tumor_id: str) -> dict[str, int | float | str]:
    return parse_cohort_patient_id(tumor_id)


def _select_representative_files_with_filter(
    input_dir: Path,
    reps_per_scenario: int,
    n_mean_values: list[int] | None,
    *,
    patient_id_parser=parse_cohort_patient_id,
    selection_group_cols: list[str] | None = None,
) -> list[Path]:
    if selection_group_cols is None:
        selection_group_cols = ["N_mean", "true_K", "purity", "amp_rate", "n_samples", "lambda_mut_setting"]

    grouped: dict[tuple[object, ...], list[Path]] = {}
    n_mean_set = None if n_mean_values is None else {int(value) for value in n_mean_values}

    for file_path in sorted(input_dir.glob("*.tsv")):
        meta = patient_id_parser(file_path.stem)
        if n_mean_set is not None and int(meta["N_mean"]) not in n_mean_set:
            continue
        key = tuple(meta[column] for column in selection_group_cols)
        grouped.setdefault(key, []).append(file_path)

    selected: list[Path] = []
    for key in sorted(grouped):
        files = grouped[key]
        count = min(max(reps_per_scenario, 1), len(files))
        pick_idx = np.linspace(0, len(files) - 1, num=count, dtype=int)
        selected.extend(files[idx] for idx in sorted(set(pick_idx.tolist())))
    return selected


def _add_cluster_count_metrics(patient_df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = patient_df.copy()
    cluster_count_error = enriched_df["n_clusters"] - enriched_df["true_K"]
    enriched_df["cluster_count_error"] = cluster_count_error
    enriched_df["abs_cluster_count_error"] = cluster_count_error.abs()
    enriched_df["cluster_count_exact_match"] = (
        enriched_df["n_clusters"] == enriched_df["true_K"]
    ).astype(float)
    return enriched_df


def _with_tumor_region_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliased = df.copy()
    if "patient_id" in aliased.columns and "tumor_id" not in aliased.columns:
        aliased["tumor_id"] = aliased["patient_id"]
    if "n_samples" in aliased.columns and "n_regions" not in aliased.columns:
        aliased["n_regions"] = aliased["n_samples"]
    if "n_patients" in aliased.columns and "n_tumors" not in aliased.columns:
        aliased["n_tumors"] = aliased["n_patients"]
    if "n_evaluable_patients" in aliased.columns and "n_evaluable_tumors" not in aliased.columns:
        aliased["n_evaluable_tumors"] = aliased["n_evaluable_patients"]
    return aliased


def _display_tumor_region_df(df: pd.DataFrame) -> pd.DataFrame:
    display = _with_tumor_region_aliases(df)
    if "patient_id" in display.columns and "tumor_id" in display.columns:
        display = display.drop(columns=["patient_id"])
    elif "patient_id" in display.columns:
        display = display.rename(columns={"patient_id": "tumor_id"})
    if "n_samples" in display.columns and "n_regions" in display.columns:
        display = display.drop(columns=["n_samples"])
    elif "n_samples" in display.columns:
        display = display.rename(columns={"n_samples": "n_regions"})
    if "n_patients" in display.columns and "n_tumors" in display.columns:
        display = display.drop(columns=["n_patients"])
    elif "n_patients" in display.columns:
        display = display.rename(columns={"n_patients": "n_tumors"})
    if "n_evaluable_patients" in display.columns and "n_evaluable_tumors" in display.columns:
        display = display.drop(columns=["n_evaluable_patients"])
    elif "n_evaluable_patients" in display.columns:
        display = display.rename(columns={"n_evaluable_patients": "n_evaluable_tumors"})
    return display


def _aggregate_patient_results(patient_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_df = _add_cluster_count_metrics(patient_df)
    scenario_cols = [column for column in ["N_mean", "purity", "amp_rate", "n_samples", "lambda_mut_setting"] if column in patient_df.columns]
    scenario_df = (
        patient_df.groupby(scenario_cols, dropna=False)
        .agg(
            n_patients=("patient_id", "size"),
            n_evaluable_patients=("ARI", lambda s: int(s.notna().sum())),
            mean_true_K=("true_K", "mean"),
            mean_selected_lambda=("selected_lambda", "mean"),
            mean_estimated_clusters=("n_clusters", "mean"),
            mean_cluster_count_error=("cluster_count_error", "mean"),
            mean_abs_cluster_count_error=("abs_cluster_count_error", "mean"),
            exact_cluster_count_match_rate=("cluster_count_exact_match", "mean"),
            mean_ARI=("ARI", "mean"),
            median_ARI=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_multiplicity_f1=("multiplicity_f1", "mean"),
            mean_estimated_clonal_fraction=("estimated_clonal_fraction", "mean"),
            mean_true_clonal_fraction=("true_clonal_fraction", "mean"),
            mean_clonal_fraction_error=("clonal_fraction_error", "mean"),
            mean_abs_clonal_fraction_error=(
                "clonal_fraction_error",
                _safe_nanmean_abs,
            ),
            mean_eval_mutations=("n_eval_mutations", "mean"),
            mean_filtered_mutations=("n_filtered_mutations", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .reset_index()
        .sort_values(scenario_cols)
    )

    global_df = pd.DataFrame(
        [
            {
                "n_patients": int(patient_df.shape[0]),
                "n_evaluable_patients": int(patient_df["ARI"].notna().sum()),
                "mean_ARI": float(patient_df["ARI"].mean()),
                "median_ARI": float(patient_df["ARI"].median()),
                "mean_cp_rmse": float(patient_df["cp_rmse"].mean()),
                "mean_multiplicity_f1": float(patient_df["multiplicity_f1"].mean()),
                "mean_estimated_clonal_fraction": float(patient_df["estimated_clonal_fraction"].mean()),
                "mean_true_clonal_fraction": float(patient_df["true_clonal_fraction"].mean()),
                "mean_clonal_fraction_error": float(patient_df["clonal_fraction_error"].mean()),
                "mean_abs_clonal_fraction_error": _safe_nanmean_abs(patient_df["clonal_fraction_error"]),
                "mean_estimated_clusters": float(patient_df["n_clusters"].mean()),
                "mean_cluster_count_error": float(patient_df["cluster_count_error"].mean()),
                "mean_abs_cluster_count_error": float(patient_df["abs_cluster_count_error"].mean()),
                "exact_cluster_count_match_rate": float(patient_df["cluster_count_exact_match"].mean()),
                "mean_eval_mutations": float(patient_df["n_eval_mutations"].mean()),
                "mean_filtered_mutations": float(patient_df["n_filtered_mutations"].mean()),
                "mean_elapsed_seconds": float(patient_df["elapsed_seconds"].mean()),
            }
        ]
    )
    return _with_tumor_region_aliases(scenario_df), _with_tumor_region_aliases(global_df)


def _aggregate_simple(patient_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    patient_df = _add_cluster_count_metrics(patient_df)
    aggregated = (
        patient_df.groupby(columns, dropna=False)
        .agg(
            n_patients=("patient_id", "size"),
            n_evaluable_patients=("ARI", lambda s: int(s.notna().sum())),
            mean_true_K=("true_K", "mean"),
            mean_selected_lambda=("selected_lambda", "mean"),
            mean_estimated_clusters=("n_clusters", "mean"),
            mean_cluster_count_error=("cluster_count_error", "mean"),
            mean_abs_cluster_count_error=("abs_cluster_count_error", "mean"),
            exact_cluster_count_match_rate=("cluster_count_exact_match", "mean"),
            mean_ARI=("ARI", "mean"),
            median_ARI=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_multiplicity_f1=("multiplicity_f1", "mean"),
            mean_estimated_clonal_fraction=("estimated_clonal_fraction", "mean"),
            mean_true_clonal_fraction=("true_clonal_fraction", "mean"),
            mean_clonal_fraction_error=("clonal_fraction_error", "mean"),
            mean_abs_clonal_fraction_error=(
                "clonal_fraction_error",
                _safe_nanmean_abs,
            ),
            mean_eval_mutations=("n_eval_mutations", "mean"),
            mean_filtered_mutations=("n_filtered_mutations", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .reset_index()
        .sort_values(columns)
    )
    return _with_tumor_region_aliases(aggregated)


def materialize_patient_df(
    patient_rows: list[dict[str, int | float | str | bool]],
    *,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    patient_df = pd.DataFrame(patient_rows)
    if patient_df.empty:
        return patient_df
    sort_columns = DEFAULT_PATIENT_SORT_COLUMNS if sort_columns is None else sort_columns
    existing_sort_columns = [column for column in sort_columns if column in patient_df.columns]
    if existing_sort_columns:
        patient_df = patient_df.sort_values(existing_sort_columns).reset_index(drop=True)
    else:
        patient_df = patient_df.reset_index(drop=True)
    return _with_tumor_region_aliases(_add_cluster_count_metrics(patient_df))


def materialize_tumor_df(
    tumor_rows: list[dict[str, int | float | str | bool]],
    *,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    return materialize_patient_df(tumor_rows, sort_columns=sort_columns)


def write_patient_checkpoint(
    patient_rows: list[dict[str, int | float | str | bool]],
    *,
    outdir: Path,
    start_time: float,
    case_index: int,
    total_cases: int,
    label: str,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    patient_df = materialize_patient_df(patient_rows, sort_columns=sort_columns)
    patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
    _display_tumor_region_df(patient_df).to_csv(outdir / "benchmark_tumors.tsv", sep="\t", index=False)

    elapsed = perf_counter() - start_time
    rate = case_index / max(elapsed, 1e-9)
    remaining = max(total_cases - case_index, 0)
    eta_seconds = remaining / max(rate, 1e-9)
    print(
        f"[{label}] {case_index}/{total_cases} cases "
        f"| elapsed={elapsed/60.0:.1f} min | rate={rate:.2f} cases/s "
        f"| eta={eta_seconds/60.0:.1f} min"
    )
    return patient_df


def write_tumor_checkpoint(
    tumor_rows: list[dict[str, int | float | str | bool]],
    *,
    outdir: Path,
    start_time: float,
    case_index: int,
    total_cases: int,
    label: str,
    sort_columns: list[str] | None = None,
) -> pd.DataFrame:
    return write_patient_checkpoint(
        tumor_rows,
        outdir=outdir,
        start_time=start_time,
        case_index=case_index,
        total_cases=total_cases,
        label=label,
        sort_columns=sort_columns,
    )


def write_benchmark_tables(patient_df: pd.DataFrame, outdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_df, global_df = _aggregate_patient_results(patient_df)
    by_samples_df = _aggregate_simple(patient_df, ["n_samples"])
    by_depth_df = _aggregate_simple(patient_df, ["N_mean"])
    by_true_k_df = _aggregate_simple(patient_df, ["true_K"])
    by_purity_df = _aggregate_simple(patient_df, ["purity"])
    by_amp_rate_df = _aggregate_simple(patient_df, ["amp_rate"])

    patient_df.to_csv(outdir / "benchmark_patients.tsv", sep="\t", index=False)
    _display_tumor_region_df(patient_df).to_csv(outdir / "benchmark_tumors.tsv", sep="\t", index=False)
    scenario_df.to_csv(outdir / "benchmark_by_scenario.tsv", sep="\t", index=False)
    by_samples_df.to_csv(outdir / "benchmark_by_n_samples.tsv", sep="\t", index=False)
    _display_tumor_region_df(by_samples_df).to_csv(outdir / "benchmark_by_n_regions.tsv", sep="\t", index=False)
    by_depth_df.to_csv(outdir / "benchmark_by_depth.tsv", sep="\t", index=False)
    by_true_k_df.to_csv(outdir / "benchmark_by_true_k.tsv", sep="\t", index=False)
    by_purity_df.to_csv(outdir / "benchmark_by_purity.tsv", sep="\t", index=False)
    by_amp_rate_df.to_csv(outdir / "benchmark_by_amp_rate.tsv", sep="\t", index=False)
    global_df.to_csv(outdir / "benchmark_global.tsv", sep="\t", index=False)
    return scenario_df, global_df


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


def _fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        major_prior=args.major_prior,
        device=args.device,
        dtype=getattr(args, "dtype", "auto"),
        summary_tol=getattr(args, "summary_tol", None),
        verbose=args.verbose,
    )


__all__ = [
    "DEFAULT_PATIENT_SORT_COLUMNS",
    "DEFAULT_TUMOR_SORT_COLUMNS",
    "PATIENT_PATTERN",
    "SINGLE_REGION_PATTERN",
    "_add_cluster_count_metrics",
    "_aggregate_patient_results",
    "_aggregate_simple",
    "_fit_options_from_args",
    "_parse_lambda_grid",
    "_parse_patient_id",
    "_select_representative_files_with_filter",
    "materialize_patient_df",
    "materialize_tumor_df",
    "parse_cohort_patient_id",
    "parse_cohort_tumor_id",
    "parse_single_region_patient_id",
    "parse_single_region_tumor_id",
    "write_benchmark_tables",
    "write_patient_checkpoint",
    "write_tumor_checkpoint",
]
