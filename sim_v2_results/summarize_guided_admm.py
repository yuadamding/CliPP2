#!/usr/bin/env python3
"""Reproducible audit and aggregation for a simulated CliPP2 benchmark.

The script independently recomputes clustering, CCF, clonal-fraction, and
multiplicity metrics from the per-tumor artifacts.  It also audits the selected
solver provenance.  Its defaults target ``sim_v2_guided_admm_out`` but every
path and the displayed run label are configurable, so it can be reused for a
replacement run without editing this file.

Run from the repository root with the requested environment::

    conda run -n ml1 python sim_v2_results/summarize_guided_admm.py \
        --run-label "diagnostic guided-ADMM"

The default mode is strict: the requested number of unique tumors, a complete
factorial design, matching input/truth/output cohorts, and all six per-tumor
output artifacts are required.  Factor levels are inferred from tumor IDs, so
the same audit supports both the original 120-tumor cohort and Sim1K.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from CliPP2.io.data import TumorData, load_tumor_tsv  # noqa: E402
from CliPP2.metrics.evaluation import (  # noqa: E402
    SimulationTruth,
    _adjusted_rand_index,
    _cluster_level_clonal_fraction,
    load_simulation_truth,
)


ARTIFACT_SUFFIXES = (
    "run_summary.tsv",
    "simulation_eval.tsv",
    "mutation_clusters.tsv",
    "mutation_region_multiplicity.tsv",
    "cluster_centers.tsv",
    "lambda_search.tsv",
)
EXPECTED_SELECTION_PROVENANCE = {
    "selection_method": "online_partition_guided_admm",
    "selection_score_name": "partition_icl",
    "lambda_search_mode": "partition_guided_admm",
    "lambda_source": "online_partition_guide_kkt",
    "initialization_mode": "ward_cem_partition_icl_kkt",
    "initializer_selection_score": "partition_icl",
}
STRICT_SOLVER_CHECKS = (
    "positive_lambda",
    "raw_candidate_source",
    "requested_selection_provenance",
    "requested_initializer_provenance",
    "complete_graph",
    "complete_edge_count",
    "backend_neutral_exact_provenance",
    "kkt_at_or_below_certificate_tolerance",
    "kkt_finite",
    "kkt_at_or_below_selection_gate",
    "raw_kkt_eligible",
    "converged",
    "stationarity_certified",
    "selection_eligible",
    "lambda_online",
    "postselection_truth_only",
)
EXACTNESS_PROVENANCE_VERSION = 1
ACCEPTED_EXACT_CERTIFICATE_STATUSES = frozenset(
    {
        "certified",
        "input_dual_retained",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
        "zero_penalty_no_dual_needed",
    }
)
TUMOR_RE = re.compile(
    r"^(?P<depth>\d+)_(?P<name_k>\d+)_(?P<nominal_purity>[0-9.]+)_"
    r"(?P<amp_rate>[0-9.]+)_S(?P<S>\d+)_Lm(?P<lambda_mut>\d+)_"
    r"M(?P<name_m>\d+)_rep(?P<rep>\d+)$"
)


@dataclass(frozen=True)
class CaseData:
    data: TumorData
    truth: SimulationTruth
    depth: int
    nominal_purity: float
    amp_rate: float
    lambda_mut: int
    replicate: int
    true_k: int
    min_true_linf_separation: float


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def _bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    truthy = {"1", "true", "t", "yes", "y"}
    return values.map(lambda value: str(value).strip().lower() in truthy).astype(bool)


def _numeric(row: pd.Series, key: str, default: float = float("nan")) -> float:
    if key not in row.index:
        return default
    value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
    return default if pd.isna(value) else float(value)


def _boolean(row: pd.Series, key: str, default: bool = False) -> bool:
    if key not in row.index or pd.isna(row[key]):
        return default
    return bool(_bool_series(pd.Series([row[key]])).iloc[0])


def _safe_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _safe_rmse(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else float("nan")


def _min_linf_separation(truth_phi: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    centers = np.vstack([truth_phi[labels == label].mean(axis=0) for label in unique])
    separation = float("inf")
    for left in range(centers.shape[0] - 1):
        distances = np.max(np.abs(centers[left + 1 :] - centers[left]), axis=1)
        separation = min(separation, float(np.min(distances)))
    return separation


def _oracle_input_noise_metrics(
    data: TumorData,
    truth: SimulationTruth,
    *,
    probability_clip: float = 1e-8,
) -> dict[str, float]:
    """Compute truth-conditioned, input-derived binomial CCF information.

    This is an oracle simulation diagnostic, not CliPP2 posterior uncertainty.
    For each true cluster and region, binomial Fisher information is pooled over
    its mutations using the true multiplicity and CCF. Pair separations use the
    independent-estimate variance sum for the two true cluster centers.
    """

    if truth.truth_multiplicity is None:
        return {
            "oracle_fisher_se_mean": float("nan"),
            "oracle_fisher_se_median": float("nan"),
            "oracle_zero_information_cluster_regions": float("nan"),
            "oracle_weakest_pair_mahalanobis": float("nan"),
            "oracle_weakest_pair_max_region_z": float("nan"),
        }
    if not 0.0 < float(probability_clip) < 0.5:
        raise ValueError("probability_clip must be in (0, 0.5)")

    labels = np.asarray(truth.truth_clusters, dtype=np.int64).reshape(-1)
    truth_phi = np.asarray(truth.truth_phi, dtype=np.float64)
    truth_multiplicity = np.asarray(truth.truth_multiplicity, dtype=np.float64)
    if truth_phi.shape != data.total_counts.shape:
        raise ValueError("Oracle Fisher metric truth/input CCF shape mismatch")
    if truth_multiplicity.shape != truth_phi.shape or labels.size != truth_phi.shape[0]:
        raise ValueError("Oracle Fisher metric truth multiplicity/label shape mismatch")

    scaling = np.asarray(data.scaling, dtype=np.float64)
    total_counts = np.asarray(data.total_counts, dtype=np.float64)
    observed_source = getattr(data, "count_observed", None)
    observed = (
        np.ones_like(total_counts, dtype=bool)
        if observed_source is None
        else np.asarray(observed_source, dtype=bool)
    )
    probability_slope = scaling * truth_multiplicity
    probability = np.clip(
        probability_slope * truth_phi,
        float(probability_clip),
        1.0 - float(probability_clip),
    )
    valid = (
        observed
        & np.isfinite(total_counts)
        & (total_counts > 0.0)
        & np.isfinite(probability_slope)
        & (probability_slope > 0.0)
        & np.isfinite(probability)
    )
    information_terms = np.zeros_like(truth_phi, dtype=np.float64)
    information_terms[valid] = (
        total_counts[valid]
        * np.square(probability_slope[valid])
        / (probability[valid] * (1.0 - probability[valid]))
    )

    unique_labels = np.unique(labels)
    cluster_centers = np.vstack(
        [truth_phi[labels == label].mean(axis=0) for label in unique_labels]
    )
    cluster_information = np.vstack(
        [information_terms[labels == label].sum(axis=0) for label in unique_labels]
    )
    cluster_se = np.full_like(cluster_information, np.inf, dtype=np.float64)
    positive_information = np.isfinite(cluster_information) & (
        cluster_information > 0.0
    )
    cluster_se[positive_information] = 1.0 / np.sqrt(
        cluster_information[positive_information]
    )
    zero_information_count = int(np.sum(~positive_information))
    se_mean = float(np.mean(cluster_se))
    se_median = float(np.median(cluster_se))

    if unique_labels.size < 2:
        weakest_mahalanobis = float("nan")
        weakest_max_region_z = float("nan")
    else:
        pair_mahalanobis: list[float] = []
        pair_max_region_z: list[float] = []
        for left in range(unique_labels.size - 1):
            for right in range(left + 1, unique_labels.size):
                difference = np.abs(cluster_centers[left] - cluster_centers[right])
                variance = np.square(cluster_se[left]) + np.square(cluster_se[right])
                z = np.zeros_like(difference, dtype=np.float64)
                finite_variance = np.isfinite(variance) & (variance > 0.0)
                z[finite_variance] = difference[finite_variance] / np.sqrt(
                    variance[finite_variance]
                )
                pair_mahalanobis.append(float(np.sqrt(np.sum(np.square(z)))))
                pair_max_region_z.append(float(np.max(z)) if z.size else 0.0)
        weakest_mahalanobis = float(np.min(pair_mahalanobis))
        weakest_max_region_z = float(np.min(pair_max_region_z))

    return {
        "oracle_fisher_se_mean": se_mean,
        "oracle_fisher_se_median": se_median,
        "oracle_zero_information_cluster_regions": zero_information_count,
        "oracle_weakest_pair_mahalanobis": weakest_mahalanobis,
        "oracle_weakest_pair_max_region_z": weakest_max_region_z,
    }


def _binary_confusion(
    y_true_major: np.ndarray, y_pred_major: np.ndarray
) -> dict[str, float | int]:
    y_true = np.asarray(y_true_major, dtype=bool).reshape(-1)
    y_pred = np.asarray(y_pred_major, dtype=bool).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("Multiplicity truth and prediction shapes differ.")

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    def f1(correct: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * correct + false_positive + false_negative
        return float(2 * correct / denominator) if denominator else 0.0

    major_f1 = f1(tp, fp, fn)
    minor_f1 = f1(tn, fn, fp)
    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total else float("nan")
    return {
        "n": total,
        "tp_major": tp,
        "tn_minor": tn,
        "fp_major": fp,
        "fn_major": fn,
        "major_f1": major_f1,
        "minor_f1": minor_f1,
        "macro_f1": 0.5 * (major_f1 + minor_f1),
        "micro_f1_accuracy": accuracy,
    }


def _sum_confusions(
    rows: pd.DataFrame,
    *,
    prefix: str = "mult_",
) -> dict[str, float | int]:
    tp = int(rows[f"{prefix}tp_major"].sum())
    tn = int(rows[f"{prefix}tn_minor"].sum())
    fp = int(rows[f"{prefix}fp_major"].sum())
    fn = int(rows[f"{prefix}fn_major"].sum())
    return _binary_confusion(
        np.r_[np.ones(tp + fn, dtype=bool), np.zeros(tn + fp, dtype=bool)],
        np.r_[
            np.ones(tp, dtype=bool),
            np.zeros(fn, dtype=bool),
            np.zeros(tn, dtype=bool),
            np.ones(fp, dtype=bool),
        ],
    )


def _parse_tumor_id(tumor_id: str) -> Mapping[str, float | int]:
    match = TUMOR_RE.match(tumor_id)
    if match is None:
        raise ValueError(f"Unexpected simulated tumor ID: {tumor_id!r}")
    fields = match.groupdict()
    return {
        "depth": int(fields["depth"]),
        "name_k": int(fields["name_k"]),
        "nominal_purity": float(fields["nominal_purity"]),
        "amp_rate": float(fields["amp_rate"]),
        "S": int(fields["S"]),
        "lambda_mut": int(fields["lambda_mut"]),
        "name_m": int(fields["name_m"]),
        "replicate": int(fields["rep"]),
    }


def _validate_factorial_design(
    parsed_by_id: Mapping[str, Mapping[str, float | int]],
    *,
    expected_tumors: int,
    expected_per_s: int | None,
) -> None:
    """Require one case for every observed simulation-factor combination."""

    rows = [
        {
            "tumor_id": tumor_id,
            **parsed,
            "purity_replicate": (
                float(parsed["nominal_purity"]),
                int(parsed["replicate"]),
            ),
        }
        for tumor_id, parsed in parsed_by_id.items()
    ]
    design = pd.DataFrame(rows)
    if len(design) != expected_tumors:
        raise ValueError(
            f"Expected {expected_tumors} parsed design rows, found {len(design)}"
        )

    factor_columns = (
        "depth",
        "purity_replicate",
        "amp_rate",
        "S",
        "lambda_mut",
    )
    duplicated = design.duplicated(list(factor_columns), keep=False)
    if duplicated.any():
        example = design.loc[duplicated, ["tumor_id", *factor_columns]].head(2)
        raise ValueError(
            "Duplicate simulation-factor cell(s): "
            + example.to_dict(orient="records").__repr__()
        )

    levels = {
        column: sorted(design[column].unique().tolist()) for column in factor_columns
    }
    expected_cells = set(
        itertools.product(*(levels[column] for column in factor_columns))
    )
    observed_cells = set(
        design.loc[:, factor_columns].itertuples(index=False, name=None)
    )
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells, key=repr)
        extra = sorted(observed_cells - expected_cells, key=repr)
        raise ValueError(
            "Simulation cohort is not a complete factorial design; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(expected_cells) != expected_tumors:
        raise ValueError(
            "Observed factor levels imply "
            f"{len(expected_cells)} cases, not the requested {expected_tumors}"
        )

    counts = design["S"].value_counts().sort_index()
    if counts.nunique() != 1:
        raise ValueError(f"Expected balanced S counts, observed {counts.to_dict()}")
    if expected_per_s is not None and not (counts == expected_per_s).all():
        raise ValueError(
            f"Expected {expected_per_s} tumors per S, observed {counts.to_dict()}"
        )


def _load_summary(
    results_dir: Path,
    *,
    expected_tumors: int,
    allow_incomplete: bool,
) -> pd.DataFrame:
    summary_path = results_dir / "single_stage_summary.tsv"
    paths = sorted(results_dir.glob("*_run_summary.tsv"))
    if not allow_incomplete:
        if len(paths) != int(expected_tumors):
            raise FileNotFoundError(
                f"Expected {expected_tumors} complete per-tumor summaries, "
                f"found {len(paths)} in {results_dir}."
            )
        summary = pd.concat((_read_tsv(path) for path in paths), ignore_index=True)
        # Always refresh the mechanical cohort index from atomic per-tumor
        # bundles. This also incorporates a targeted rerun after a failed
        # directory process; downstream ID/provenance checks validate the
        # reconstructed cohort before any metric is reported.
        summary.to_csv(summary_path, sep="\t", index=False)
    elif summary_path.is_file():
        summary = _read_tsv(summary_path)
    elif paths:
        summary = pd.concat((_read_tsv(path) for path in paths), ignore_index=True)
    else:
        raise FileNotFoundError(f"No summary files found in {results_dir}")
    if "tumor_id" not in summary.columns:
        raise ValueError(f"Missing tumor_id in {summary_path}")
    summary["tumor_id"] = summary["tumor_id"].astype(str)
    if summary["tumor_id"].duplicated().any():
        duplicate = summary.loc[summary["tumor_id"].duplicated(), "tumor_id"].iloc[0]
        raise ValueError(f"Duplicate tumor in cohort summary: {duplicate}")
    return summary


def _load_cases(
    *,
    inputs_dir: Path,
    truth_root: Path,
    summary_ids: set[str],
    expected_tumors: int,
    expected_per_s: int | None,
    allow_incomplete: bool,
) -> dict[str, CaseData]:
    input_paths = sorted(inputs_dir.glob("*.tsv"))
    input_ids = {path.stem for path in input_paths}
    if len(input_ids) != len(input_paths):
        raise ValueError("Duplicate input TSV stems detected.")
    if not allow_incomplete and len(input_paths) != expected_tumors:
        raise ValueError(
            f"Expected {expected_tumors} input TSVs, found {len(input_paths)}"
        )
    if not summary_ids.issubset(input_ids):
        missing = sorted(summary_ids - input_ids)
        raise ValueError(f"Summary tumors missing input TSVs: {missing[:5]}")
    if not allow_incomplete and summary_ids != input_ids:
        missing = sorted(input_ids - summary_ids)
        extra = sorted(summary_ids - input_ids)
        raise ValueError(
            f"Summary/input tumor mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )

    truth_ids = {path.name for path in truth_root.iterdir() if path.is_dir()}
    if not allow_incomplete and truth_ids != input_ids:
        missing = sorted(input_ids - truth_ids)
        extra = sorted(truth_ids - input_ids)
        raise ValueError(
            f"Input/truth tumor mismatch; missing truth={missing[:5]}, "
            f"extra truth={extra[:5]}"
        )

    parsed_by_id = {tumor_id: _parse_tumor_id(tumor_id) for tumor_id in input_ids}
    if not allow_incomplete:
        _validate_factorial_design(
            parsed_by_id,
            expected_tumors=expected_tumors,
            expected_per_s=expected_per_s,
        )

    cases: dict[str, CaseData] = {}
    for path in input_paths:
        tumor_id = path.stem
        if tumor_id not in summary_ids:
            continue
        parsed = parsed_by_id[tumor_id]
        data = load_tumor_tsv(path, validation_mode="strict")
        truth_dir = truth_root / tumor_id
        if not truth_dir.is_dir():
            raise FileNotFoundError(truth_dir)
        truth = load_simulation_truth(data, truth_root)
        true_k = int(np.unique(truth.truth_clusters).size)
        if data.num_regions != parsed["S"]:
            raise ValueError(f"{tumor_id}: S in name does not match input matrix")
        if data.num_mutations != parsed["name_m"]:
            raise ValueError(f"{tumor_id}: M in name does not match input matrix")
        if true_k != parsed["name_k"]:
            raise ValueError(f"{tumor_id}: K in name does not match truth")
        cases[tumor_id] = CaseData(
            data=data,
            truth=truth,
            depth=int(parsed["depth"]),
            nominal_purity=float(parsed["nominal_purity"]),
            amp_rate=float(parsed["amp_rate"]),
            lambda_mut=int(parsed["lambda_mut"]),
            replicate=int(parsed["replicate"]),
            true_k=true_k,
            min_true_linf_separation=_min_linf_separation(
                truth.truth_phi, truth.truth_clusters
            ),
        )
    if not allow_incomplete:
        if len(cases) != expected_tumors:
            raise ValueError(f"Expected {expected_tumors} cases, loaded {len(cases)}")
    return cases


def _aligned_mutation_table(path: Path, data: TumorData) -> pd.DataFrame:
    table = _read_tsv(path)
    if "mutation_id" not in table.columns:
        raise ValueError(f"Missing mutation_id: {path}")
    table["mutation_id"] = table["mutation_id"].astype(str)
    if table["mutation_id"].duplicated().any():
        raise ValueError(f"Duplicate mutations: {path}")
    indexed = table.set_index("mutation_id", drop=False)
    missing = set(data.mutation_ids) - set(indexed.index)
    extra = set(indexed.index) - set(data.mutation_ids)
    if missing or extra:
        raise ValueError(
            f"Mutation mismatch in {path}; missing={list(missing)[:3]}, extra={list(extra)[:3]}"
        )
    return indexed.loc[data.mutation_ids].reset_index(drop=True)


def _phi_matrix(table: pd.DataFrame, data: TumorData, prefix: str) -> np.ndarray:
    columns = [
        f"{prefix}{str(region).replace('sample', 'region')}"
        for region in data.region_ids
    ]
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"Missing CCF columns: {missing[:3]}")
    matrix = table[columns].to_numpy(dtype=float)
    if (
        matrix.shape != (data.num_mutations, data.num_regions)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Invalid CCF artifact matrix.")
    return matrix


def _validate_multiplicity_calls(
    calls: np.ndarray,
    major_cn: np.ndarray,
    minor_cn: np.ndarray,
    *,
    source: str,
) -> None:
    calls = np.asarray(calls, dtype=float)
    major_cn = np.asarray(major_cn, dtype=float)
    minor_cn = np.asarray(minor_cn, dtype=float)
    if calls.shape != major_cn.shape or calls.shape != minor_cn.shape:
        raise ValueError(
            f"{source}: multiplicity/CN shape mismatch "
            f"({calls.shape}, {major_cn.shape}, {minor_cn.shape})"
        )
    valid = np.isfinite(calls) & (
        np.isclose(calls, major_cn, rtol=0.0, atol=1e-8)
        | np.isclose(calls, minor_cn, rtol=0.0, atol=1e-8)
    )
    if not valid.all():
        mutation_index, region_index = np.argwhere(~valid)[0]
        raise ValueError(
            f"{source}: invalid multiplicity call at mutation index "
            f"{mutation_index}, region index {region_index}: "
            f"call={calls[mutation_index, region_index]!r}, "
            f"major_cn={major_cn[mutation_index, region_index]!r}, "
            f"minor_cn={minor_cn[mutation_index, region_index]!r}"
        )


def _aligned_multiplicity(path: Path, case: CaseData) -> np.ndarray:
    table = _read_tsv(path)
    required = {"mutation_id", "region_id", "multiplicity_call"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    table["mutation_id"] = table["mutation_id"].astype(str)
    table["region_id"] = table["region_id"].astype(str)
    if table.duplicated(["mutation_id", "region_id"]).any():
        raise ValueError(f"Duplicate mutation-region cells in {path}")
    expected = case.data.num_mutations * case.data.num_regions
    if len(table) != expected:
        raise ValueError(
            f"{path}: expected {expected} multiplicity rows, found {len(table)}"
        )
    pivot = table.pivot(
        index="mutation_id", columns="region_id", values="multiplicity_call"
    )
    regions = [
        str(region).replace("sample", "region") for region in case.data.region_ids
    ]
    missing_mutations = set(case.data.mutation_ids) - set(pivot.index)
    missing_regions = set(regions) - set(pivot.columns)
    if missing_mutations or missing_regions:
        raise ValueError(f"Multiplicity keys do not align in {path}")
    result = pivot.loc[case.data.mutation_ids, regions].to_numpy(dtype=float)
    _validate_multiplicity_calls(
        result,
        case.data.major_cn,
        case.data.minor_cn,
        source=str(path),
    )
    return result


def _selected_search_row(search: pd.DataFrame, tumor_id: str) -> pd.Series:
    if "is_selected_best_row" in search.columns:
        mask = _bool_series(search["is_selected_best_row"])
        if int(mask.sum()) != 1:
            raise ValueError(
                f"{tumor_id}: expected one selected lambda-search row, found {int(mask.sum())}"
            )
        return search.loc[mask].iloc[0]
    if len(search) != 1:
        raise ValueError(f"{tumor_id}: lambda search has no selected-row marker")
    return search.iloc[0]


def _assert_close(
    name: str, observed: float, expected: float, *, tolerance: float = 2e-7
) -> None:
    if np.isfinite(observed) and np.isfinite(expected):
        if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
            raise ValueError(
                f"{name}: summary={observed:.12g}, recomputed={expected:.12g}"
            )
    elif np.isfinite(observed) != np.isfinite(expected):
        raise ValueError(f"{name}: finite/non-finite mismatch")


def _required_numeric(row: pd.Series, key: str, *, source: str) -> float:
    value = _numeric(row, key)
    if not np.isfinite(value):
        raise ValueError(f"{source}: missing or non-finite numeric field {key!r}")
    return float(value)


def _required_integer(row: pd.Series, key: str, *, source: str) -> int:
    value = _required_numeric(row, key, source=source)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{source}: field {key!r} is not an integer: {value!r}")
    return rounded


def _required_text(row: pd.Series, key: str, *, source: str) -> str:
    if key not in row.index or pd.isna(row[key]):
        raise ValueError(f"{source}: missing text field {key!r}")
    value = str(row[key]).strip()
    if not value:
        raise ValueError(f"{source}: empty text field {key!r}")
    return value


def _optional_exactness_schema(row: pd.Series, *, source: str) -> int | None:
    key = "exactness_provenance_version"
    if key not in row.index or pd.isna(row[key]):
        return None
    value = str(row[key]).strip().lower()
    if value in {"", "nan"}:
        return None
    return _required_integer(row, key, source=source)


def _selected_exactness_contract(
    *,
    summary_row: pd.Series,
    selected_row: pd.Series,
    summary_source: str,
    search_source: str,
    selected_kkt: float,
    solver_tol: float,
    summary_solver: str,
    admm_iterations: int,
) -> dict[str, object]:
    """Return a backend-neutral exactness audit with dense legacy support."""

    summary_schema = _optional_exactness_schema(summary_row, source=summary_source)
    selected_schema = _optional_exactness_schema(selected_row, source=search_source)
    if (summary_schema is None) != (selected_schema is None):
        raise ValueError(
            "Exactness provenance is present in only one selected artifact "
            f"({summary_source}, {search_source})."
        )
    if selected_schema is None:
        legacy_exact = bool(
            summary_solver == "admm_complete_graph" and admm_iterations > 0
        )
        certificate_tolerance = 5.0 * float(solver_tol)
        return {
            "exactness_provenance_mode": "legacy_dense_admm",
            "exactness_provenance_version": 0,
            "explicit_exactness_provenance": False,
            "legacy_dense_exactness_provenance": legacy_exact,
            "objective_faithful": False,
            "full_kkt_certified": legacy_exact,
            "certificate_status_accepted": legacy_exact,
            "certificate_scope_full_original_graph": legacy_exact,
            "certificate_gradient_observed_objective": legacy_exact,
            "full_kkt_tolerance": certificate_tolerance,
            "kkt_at_or_below_certificate_tolerance": bool(
                np.isfinite(selected_kkt) and selected_kkt <= certificate_tolerance
            ),
            "inner_backend": summary_solver,
            "backend_iterations": admm_iterations,
            "backend_neutral_exact_provenance": legacy_exact,
        }

    text_fields = (
        "estimator_role",
        "objective_spec_hash",
        "original_graph_hash",
        "certificate_problem_hash",
        "certificate_scope",
        "certificate_gradient_scope",
        "full_kkt_certificate_status",
        "inner_backend",
    )
    text_values: dict[str, str] = {}
    for key in text_fields:
        summary_value = _required_text(summary_row, key, source=summary_source)
        selected_value = _required_text(selected_row, key, source=search_source)
        if summary_value != selected_value:
            raise ValueError(
                f"Selected exactness field {key!r} mismatch "
                f"({summary_value!r} vs {selected_value!r})"
            )
        text_values[key] = selected_value

    bool_values: dict[str, bool] = {}
    for key in ("objective_faithful", "full_kkt_certified"):
        if key not in summary_row.index or key not in selected_row.index:
            raise ValueError(f"Selected exactness cross-check is missing {key!r}")
        summary_value = _boolean(summary_row, key)
        selected_value = _boolean(selected_row, key)
        if summary_value != selected_value:
            raise ValueError(f"Selected exactness boolean {key!r} mismatch")
        bool_values[key] = selected_value

    summary_tolerance = _required_numeric(
        summary_row, "full_kkt_tolerance", source=summary_source
    )
    selected_tolerance = _required_numeric(
        selected_row, "full_kkt_tolerance", source=search_source
    )
    _assert_close("selected full KKT tolerance", summary_tolerance, selected_tolerance)
    summary_backend_iterations = _required_integer(
        summary_row, "backend_iterations", source=summary_source
    )
    selected_backend_iterations = _required_integer(
        selected_row, "backend_iterations", source=search_source
    )
    if summary_backend_iterations != selected_backend_iterations:
        raise ValueError(
            "Selected backend iteration mismatch "
            f"({summary_backend_iterations} vs {selected_backend_iterations})"
        )
    if selected_backend_iterations < 0:
        raise ValueError("Selected backend iteration count is negative")

    status_accepted = (
        text_values["full_kkt_certificate_status"]
        in ACCEPTED_EXACT_CERTIFICATE_STATUSES
    )
    scope_full = text_values["certificate_scope"] == "full_original_graph"
    gradient_observed = (
        text_values["certificate_gradient_scope"] == "observed_objective"
    )
    schema_v1 = bool(
        summary_schema == EXACTNESS_PROVENANCE_VERSION
        and selected_schema == EXACTNESS_PROVENANCE_VERSION
    )
    certificate_tolerance_ok = bool(
        np.isfinite(selected_tolerance)
        and selected_tolerance > 0.0
        and np.isfinite(selected_kkt)
        and selected_kkt <= selected_tolerance
    )
    exact = bool(
        schema_v1
        and text_values["estimator_role"] == "raw_fused_lambda_path"
        and bool_values["objective_faithful"]
        and bool_values["full_kkt_certified"]
        and scope_full
        and gradient_observed
        and status_accepted
        and certificate_tolerance_ok
    )
    return {
        "exactness_provenance_mode": "schema_v1",
        "exactness_provenance_version": int(selected_schema),
        "explicit_exactness_provenance": schema_v1,
        "legacy_dense_exactness_provenance": False,
        "objective_faithful": bool_values["objective_faithful"],
        "full_kkt_certified": bool_values["full_kkt_certified"],
        "certificate_status_accepted": status_accepted,
        "certificate_scope_full_original_graph": scope_full,
        "certificate_gradient_observed_objective": gradient_observed,
        "full_kkt_tolerance": selected_tolerance,
        "kkt_at_or_below_certificate_tolerance": certificate_tolerance_ok,
        "inner_backend": text_values["inner_backend"],
        "backend_iterations": selected_backend_iterations,
        "backend_neutral_exact_provenance": exact,
    }


def _selected_solver_contract(
    *,
    summary_row: pd.Series,
    selected_row: pd.Series,
    tumor_id: str,
    num_mutations: int,
) -> dict[str, object]:
    """Cross-check the selected search row and audit the requested solver contract."""

    summary_source = f"{tumor_id} run summary"
    search_source = f"{tumor_id} selected lambda-search row"
    if "tumor_id" in selected_row.index and str(selected_row["tumor_id"]) != tumor_id:
        raise ValueError(f"{tumor_id}: selected lambda-search tumor ID mismatch")

    selected_lambda = _required_numeric(
        summary_row, "selected_lambda", source=summary_source
    )
    search_lambda = _required_numeric(selected_row, "lambda", source=search_source)
    _assert_close(f"{tumor_id} selected lambda", selected_lambda, search_lambda)

    summary_k = _required_integer(summary_row, "n_clusters", source=summary_source)
    search_k = _required_integer(selected_row, "n_clusters", source=search_source)
    if summary_k != search_k:
        raise ValueError(
            f"{tumor_id}: selected search K={search_k} does not match summary K={summary_k}"
        )
    if "bic_n_clusters" in selected_row.index:
        bic_k = _required_integer(selected_row, "bic_n_clusters", source=search_source)
        if bic_k != summary_k:
            raise ValueError(
                f"{tumor_id}: selected search BIC K={bic_k} does not match summary K={summary_k}"
            )

    selected_kkt = _required_numeric(
        summary_row, "selected_kkt_residual", source=summary_source
    )
    search_kkt_key = next(
        (
            key
            for key in (
                "raw_kkt_residual",
                "accepted_inner_kkt_residual",
                "inner_kkt_residual",
            )
            if key in selected_row.index and np.isfinite(_numeric(selected_row, key))
        ),
        None,
    )
    if search_kkt_key is None:
        raise ValueError(f"{search_source}: no finite selected-fit KKT residual")
    search_kkt = _required_numeric(selected_row, search_kkt_key, source=search_source)
    _assert_close(f"{tumor_id} selected KKT residual", selected_kkt, search_kkt)

    solver_tol = _required_numeric(summary_row, "tol", source=summary_source)
    if "tol" in selected_row.index:
        _assert_close(
            f"{tumor_id} selected solver tolerance",
            solver_tol,
            _required_numeric(selected_row, "tol", source=search_source),
        )

    summary_pool = _required_text(
        summary_row, "selected_candidate_pool_source", source=summary_source
    )
    search_pool = _required_text(
        selected_row, "candidate_pool_source", source=search_source
    )
    if summary_pool != search_pool:
        raise ValueError(
            f"{tumor_id}: selected candidate source mismatch "
            f"({summary_pool!r} vs {search_pool!r})"
        )

    summary_solver = _required_text(summary_row, "inner_solver", source=summary_source)
    search_solver = _required_text(selected_row, "inner_solver", source=search_source)
    if summary_solver != search_solver:
        raise ValueError(
            f"{tumor_id}: selected inner solver mismatch "
            f"({summary_solver!r} vs {search_solver!r})"
        )

    summary_graph = _required_text(summary_row, "graph_name", source=summary_source)
    search_graph = _required_text(selected_row, "graph_name", source=search_source)
    if summary_graph != search_graph:
        raise ValueError(
            f"{tumor_id}: selected graph mismatch ({summary_graph!r} vs {search_graph!r})"
        )

    summary_admm_iterations = _required_integer(
        summary_row, "admm_iterations", source=summary_source
    )
    search_admm_iterations = _required_integer(
        selected_row, "admm_iterations", source=search_source
    )
    if summary_admm_iterations != search_admm_iterations:
        raise ValueError(
            f"{tumor_id}: selected ADMM iteration mismatch "
            f"({summary_admm_iterations} vs {search_admm_iterations})"
        )

    for key in (
        "converged",
        "stationarity_certified",
        "selection_eligible",
        "raw_kkt_eligible",
        "lambda_path_prespecified",
    ):
        if key not in summary_row.index or key not in selected_row.index:
            raise ValueError(f"{tumor_id}: selected-row cross-check is missing {key!r}")
        summary_value = _boolean(summary_row, key)
        search_value = _boolean(selected_row, key)
        if summary_value != search_value:
            raise ValueError(
                f"{tumor_id}: selected boolean {key!r} mismatch "
                f"({summary_value} vs {search_value})"
            )

    summary_provenance: dict[str, str] = {}
    search_provenance: dict[str, str] = {}
    for key in EXPECTED_SELECTION_PROVENANCE:
        summary_provenance[key] = _required_text(
            summary_row, key, source=summary_source
        )
        search_provenance[key] = _required_text(selected_row, key, source=search_source)
        if summary_provenance[key] != search_provenance[key]:
            raise ValueError(
                f"{tumor_id}: selected provenance {key!r} mismatch "
                f"({summary_provenance[key]!r} vs {search_provenance[key]!r})"
            )

    summary_initializer_source = _required_text(
        summary_row, "initializer_source", source=summary_source
    )
    search_initializer_source = _required_text(
        selected_row, "initializer_source", source=search_source
    )
    if summary_initializer_source != search_initializer_source:
        raise ValueError(
            f"{tumor_id}: selected initializer source mismatch "
            f"({summary_initializer_source!r} vs {search_initializer_source!r})"
        )

    expected_num_edges = int(num_mutations * (num_mutations - 1) // 2)
    selected_num_edges = _required_integer(
        selected_row, "num_edges", source=search_source
    )
    selected_path_prespecified = _boolean(
        selected_row, "lambda_path_prespecified", True
    )
    summary_path_prespecified = _boolean(summary_row, "lambda_path_prespecified", True)
    requested_selection_provenance = all(
        summary_provenance[key] == expected
        for key, expected in EXPECTED_SELECTION_PROVENANCE.items()
        if key not in {"initialization_mode", "initializer_selection_score"}
    )
    requested_initializer_provenance = bool(
        summary_provenance["initialization_mode"]
        == EXPECTED_SELECTION_PROVENANCE["initialization_mode"]
        and summary_provenance["initializer_selection_score"]
        == EXPECTED_SELECTION_PROVENANCE["initializer_selection_score"]
        and summary_initializer_source.startswith("hessian_ward")
    )
    raw_kkt_eligible = _boolean(summary_row, "raw_kkt_eligible")
    converged = _boolean(summary_row, "converged")
    stationarity_certified = _boolean(summary_row, "stationarity_certified")
    selection_eligible = _boolean(summary_row, "selection_eligible")
    kkt_finite = bool(np.isfinite(selected_kkt))
    exactness_contract = _selected_exactness_contract(
        summary_row=summary_row,
        selected_row=selected_row,
        summary_source=summary_source,
        search_source=search_source,
        selected_kkt=selected_kkt,
        solver_tol=solver_tol,
        summary_solver=summary_solver,
        admm_iterations=summary_admm_iterations,
    )
    contract: dict[str, object] = {
        "selected_lambda": selected_lambda,
        "selected_kkt_residual": selected_kkt,
        "solver_tol": solver_tol,
        "positive_lambda": bool(selected_lambda > 0.0),
        "raw_candidate_source": summary_pool == "raw_fused_lambda_path",
        "requested_selection_provenance": requested_selection_provenance,
        "requested_initializer_provenance": requested_initializer_provenance,
        "complete_graph": bool(
            summary_graph.startswith("complete") and search_graph.startswith("complete")
        ),
        "complete_edge_count": selected_num_edges == expected_num_edges,
        "selected_num_edges": selected_num_edges,
        "expected_num_edges": expected_num_edges,
        "admm_solver": summary_solver == "admm_complete_graph",
        "admm_iterations": summary_admm_iterations,
        "admm_used": bool(summary_admm_iterations > 0),
        **exactness_contract,
        "kkt_finite": kkt_finite,
        "kkt_at_or_below_tol": bool(kkt_finite and selected_kkt <= solver_tol),
        "kkt_at_or_below_selection_gate": bool(
            kkt_finite and selected_kkt <= 5.0 * solver_tol
        ),
        "raw_kkt_eligible": raw_kkt_eligible,
        "converged": converged,
        "stationarity_certified": stationarity_certified,
        "selection_eligible": selection_eligible,
        "lambda_path_prespecified": summary_path_prespecified,
        "lambda_online": bool(
            not summary_path_prespecified and not selected_path_prespecified
        ),
        # Missing legacy provenance cannot certify that truth was isolated from
        # candidate selection, so the strict benchmark deliberately fails closed.
        "postselection_truth_only": bool(
            "evaluate_all_candidates" in summary_row.index
            and not _boolean(summary_row, "evaluate_all_candidates", True)
        ),
        "inner_solver": summary_solver,
        "graph_name": summary_graph,
        "selection_method": summary_provenance["selection_method"],
        "selection_score_name": summary_provenance["selection_score_name"],
        "lambda_search_mode": summary_provenance["lambda_search_mode"],
        "lambda_source": summary_provenance["lambda_source"],
        "initialization_mode": summary_provenance["initialization_mode"],
        "initializer_source": summary_initializer_source,
    }
    contract["strict_solver_contract"] = all(
        bool(contract[key]) for key in STRICT_SOLVER_CHECKS
    )
    return contract


def _validated_integer_column(
    table: pd.DataFrame,
    column: str,
    *,
    source: str,
) -> np.ndarray:
    if column not in table.columns:
        raise ValueError(f"{source}: missing integer column {column!r}")
    numeric = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.allclose(
        numeric, np.rint(numeric), rtol=0.0, atol=1e-9
    ):
        raise ValueError(f"{source}: invalid integer values in {column!r}")
    return np.rint(numeric).astype(np.int64)


def _validate_center_consistency(
    *,
    centers: pd.DataFrame,
    mutations: pd.DataFrame,
    data: TumorData,
) -> None:
    """Require cluster-center labels, sizes, and centers to match mutation output."""

    source = f"{data.tumor_id} cluster centers"
    required = {"tumor_id", "cluster_label", "cluster_size"}
    missing = required - set(centers.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if not centers["tumor_id"].astype(str).eq(data.tumor_id).all():
        raise ValueError(f"{source}: tumor ID mismatch")
    if (
        "tumor_id" in mutations.columns
        and not mutations["tumor_id"].astype(str).eq(data.tumor_id).all()
    ):
        raise ValueError(f"{data.tumor_id}: mutation artifact tumor ID mismatch")

    center_labels = _validated_integer_column(centers, "cluster_label", source=source)
    mutation_labels = _validated_integer_column(
        mutations, "cluster_label", source=f"{data.tumor_id} mutation output"
    )
    if len(np.unique(center_labels)) != len(center_labels):
        raise ValueError(f"{source}: duplicate cluster labels")
    expected_labels, expected_sizes = np.unique(mutation_labels, return_counts=True)
    if not np.array_equal(np.sort(center_labels), expected_labels):
        raise ValueError(
            f"{source}: labels do not match mutation clusters "
            f"({sorted(center_labels.tolist())} vs {expected_labels.tolist()})"
        )

    center_sizes = _validated_integer_column(centers, "cluster_size", source=source)
    size_by_label = dict(
        zip(center_labels.tolist(), center_sizes.tolist(), strict=True)
    )
    expected_by_label = dict(
        zip(expected_labels.tolist(), expected_sizes.tolist(), strict=True)
    )
    if size_by_label != expected_by_label:
        raise ValueError(
            f"{source}: cluster sizes do not match mutation membership "
            f"({size_by_label} vs {expected_by_label})"
        )
    if "cluster_size" in mutations.columns:
        mutation_sizes = _validated_integer_column(
            mutations,
            "cluster_size",
            source=f"{data.tumor_id} mutation output",
        )
        expected_row_sizes = np.asarray(
            [expected_by_label[int(label)] for label in mutation_labels], dtype=np.int64
        )
        if not np.array_equal(mutation_sizes, expected_row_sizes):
            raise ValueError(
                f"{data.tumor_id}: mutation cluster_size values are inconsistent"
            )

    indexed_centers = centers.assign(_cluster_label=center_labels).set_index(
        "_cluster_label", drop=True
    )
    for region_id in data.region_ids:
        region_label = str(region_id).replace("sample", "region")
        center_column = f"phi_{region_label}"
        mutation_column = f"summary_phi_{region_label}"
        if (
            center_column not in centers.columns
            or mutation_column not in mutations.columns
        ):
            continue
        center_values = pd.to_numeric(indexed_centers[center_column], errors="coerce")
        if not np.isfinite(center_values.to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: non-finite values in {center_column!r}")
        mutation_values = pd.to_numeric(
            mutations[mutation_column], errors="coerce"
        ).to_numpy(dtype=float)
        if not np.isfinite(mutation_values).all():
            raise ValueError(
                f"{data.tumor_id}: non-finite values in {mutation_column!r}"
            )
        expected_rows = np.asarray(
            [float(center_values.loc[int(label)]) for label in mutation_labels]
        )
        if not np.allclose(mutation_values, expected_rows, rtol=2e-7, atol=2e-7):
            raise ValueError(
                f"{data.tumor_id}: center {center_column!r} does not match "
                f"mutation {mutation_column!r}"
            )


def _assert_evaluation_metric(
    *,
    row: pd.Series,
    key: str,
    expected: float,
    tumor_id: str,
    required: bool,
) -> None:
    if key not in row.index:
        if required:
            raise ValueError(f"{tumor_id}: simulation evaluation is missing {key!r}")
        return
    _assert_close(
        f"{tumor_id} simulation_eval {key}",
        _numeric(row, key),
        float(expected),
    )


def _recompute_case(
    *,
    case: CaseData,
    summary_row: pd.Series,
    results_dir: Path,
    strict_artifacts: bool,
) -> dict[str, object]:
    tumor_id = case.data.tumor_id
    if strict_artifacts:
        for suffix in ARTIFACT_SUFFIXES:
            path = results_dir / f"{tumor_id}_{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)

    run_summary = _read_tsv(results_dir / f"{tumor_id}_run_summary.tsv")
    simulation_eval = _read_tsv(results_dir / f"{tumor_id}_simulation_eval.tsv")
    if len(run_summary) != 1 or len(simulation_eval) != 1:
        raise ValueError(f"{tumor_id}: expected one-row run/evaluation artifacts")
    if str(run_summary.iloc[0]["tumor_id"]) != tumor_id:
        raise ValueError(f"{tumor_id}: run summary tumor ID mismatch")

    mutations = _aligned_mutation_table(
        results_dir / f"{tumor_id}_mutation_clusters.tsv", case.data
    )
    labels = _validated_integer_column(
        mutations,
        "cluster_label",
        source=f"{tumor_id} mutation output",
    )
    raw_phi = _phi_matrix(mutations, case.data, "phi_")
    summary_phi = _phi_matrix(mutations, case.data, "summary_phi_")
    bic_phi = _phi_matrix(mutations, case.data, "bic_refit_phi_")
    ari = float(_adjusted_rand_index(case.truth.truth_clusters, labels))
    raw_squared_error = np.square(raw_phi - case.truth.truth_phi)
    summary_squared_error = np.square(summary_phi - case.truth.truth_phi)
    bic_squared_error = np.square(bic_phi - case.truth.truth_phi)
    ccf_cells = int(case.truth.truth_phi.size)
    raw_ccf_sse = float(np.sum(raw_squared_error))
    summary_ccf_sse = float(np.sum(summary_squared_error))
    bic_refit_ccf_sse = float(np.sum(bic_squared_error))
    raw_rmse = float(np.sqrt(raw_ccf_sse / ccf_cells))
    summary_rmse = float(np.sqrt(summary_ccf_sse / ccf_cells))
    bic_rmse = float(np.sqrt(bic_refit_ccf_sse / ccf_cells))
    estimated_k = int(np.unique(labels).size)
    estimated_clonal_fraction = float(
        _cluster_level_clonal_fraction(summary_phi, labels)
    )
    true_clonal_fraction = float(
        _cluster_level_clonal_fraction(case.truth.truth_phi, case.truth.truth_clusters)
    )
    oracle_noise = _oracle_input_noise_metrics(case.data, case.truth)

    predicted_mult = _aligned_multiplicity(
        results_dir / f"{tumor_id}_mutation_region_multiplicity.tsv", case
    )
    if case.truth.truth_multiplicity is None:
        raise ValueError(f"{tumor_id}: multiplicity truth is missing")
    _validate_multiplicity_calls(
        case.truth.truth_multiplicity,
        case.data.major_cn,
        case.data.minor_cn,
        source=f"{tumor_id} multiplicity truth",
    )
    asym_mask = case.data.major_cn != case.data.minor_cn
    if asym_mask.any():
        confusion = _binary_confusion(
            np.isclose(
                case.truth.truth_multiplicity[asym_mask], case.data.major_cn[asym_mask]
            ),
            np.isclose(predicted_mult[asym_mask], case.data.major_cn[asym_mask]),
        )
    else:
        confusion = {
            "n": 0,
            "tp_major": 0,
            "tn_minor": 0,
            "fp_major": 0,
            "fn_major": 0,
            "major_f1": float("nan"),
            "minor_f1": float("nan"),
            "macro_f1": float("nan"),
            "micro_f1_accuracy": float("nan"),
        }
    estimable_mask = asym_mask & (case.data.minor_cn > 0.0)
    if estimable_mask.any():
        estimable_confusion = _binary_confusion(
            np.isclose(
                case.truth.truth_multiplicity[estimable_mask],
                case.data.major_cn[estimable_mask],
            ),
            np.isclose(
                predicted_mult[estimable_mask],
                case.data.major_cn[estimable_mask],
            ),
        )
    else:
        estimable_confusion = {
            "n": 0,
            "tp_major": 0,
            "tn_minor": 0,
            "fp_major": 0,
            "fn_major": 0,
            "major_f1": float("nan"),
            "minor_f1": float("nan"),
            "macro_f1": float("nan"),
            "micro_f1_accuracy": float("nan"),
        }

    centers = _read_tsv(results_dir / f"{tumor_id}_cluster_centers.tsv")
    if len(centers) != estimated_k:
        raise ValueError(f"{tumor_id}: cluster-center count does not equal estimated K")
    _validate_center_consistency(
        centers=centers,
        mutations=mutations,
        data=case.data,
    )
    search = _read_tsv(results_dir / f"{tumor_id}_lambda_search.tsv")
    if search.empty:
        raise ValueError(f"{tumor_id}: empty lambda-search artifact")
    if (
        "tumor_id" in search.columns
        and not search["tumor_id"].astype(str).eq(tumor_id).all()
    ):
        raise ValueError(f"{tumor_id}: lambda-search tumor ID mismatch")
    selected_search_row = _selected_search_row(search, tumor_id)

    # Independent metric checks against the cohort summary.
    _assert_close(f"{tumor_id} ARI", _numeric(summary_row, "ARI"), ari)
    _assert_close(f"{tumor_id} cp_rmse", _numeric(summary_row, "cp_rmse"), summary_rmse)
    _assert_close(
        f"{tumor_id} raw_cp_rmse", _numeric(summary_row, "raw_cp_rmse"), raw_rmse
    )
    _assert_close(
        f"{tumor_id} summary_cp_rmse",
        _numeric(summary_row, "summary_cp_rmse"),
        summary_rmse,
    )
    _assert_close(
        f"{tumor_id} bic_refit_cp_rmse",
        _numeric(summary_row, "bic_refit_cp_rmse"),
        bic_rmse,
    )
    _assert_close(
        f"{tumor_id} estimated_clonal_fraction",
        _numeric(summary_row, "estimated_clonal_fraction"),
        estimated_clonal_fraction,
    )
    _assert_close(
        f"{tumor_id} true_clonal_fraction",
        _numeric(summary_row, "true_clonal_fraction"),
        true_clonal_fraction,
    )
    if int(_numeric(summary_row, "n_clusters")) != estimated_k:
        raise ValueError(f"{tumor_id}: summary K does not match mutation artifact")
    if confusion["n"]:
        _assert_close(
            f"{tumor_id} multiplicity_asymmetric_f1",
            _numeric(summary_row, "multiplicity_asymmetric_f1"),
            float(confusion["macro_f1"]),
        )
    if estimable_confusion["n"]:
        _assert_close(
            f"{tumor_id} multiplicity_estimable_f1",
            _numeric(summary_row, "multiplicity_estimable_f1"),
            float(estimable_confusion["macro_f1"]),
        )

    evaluation_row = simulation_eval.iloc[0]
    evaluation_expectations = {
        "ARI": ari,
        "cp_rmse": summary_rmse,
        "raw_cp_rmse": raw_rmse,
        "summary_cp_rmse": summary_rmse,
        "bic_refit_cp_rmse": bic_rmse,
        "multiplicity_f1": float(confusion["macro_f1"]),
        "multiplicity_asymmetric_f1": float(confusion["macro_f1"]),
        "multiplicity_estimable_f1": float(estimable_confusion["macro_f1"]),
        "estimated_clonal_fraction": estimated_clonal_fraction,
        "true_clonal_fraction": true_clonal_fraction,
        "clonal_fraction_error": estimated_clonal_fraction - true_clonal_fraction,
        "true_clusters": float(case.true_k),
        "estimated_clusters": float(estimated_k),
        "n_eval_mutations": float(case.data.num_mutations),
        "n_filtered_mutations": 0.0,
    }
    required_evaluation_fields = {
        "ARI",
        "cp_rmse",
        "multiplicity_asymmetric_f1",
        "true_clusters",
        "estimated_clusters",
    }
    for key, expected in evaluation_expectations.items():
        _assert_evaluation_metric(
            row=evaluation_row,
            key=key,
            expected=expected,
            tumor_id=tumor_id,
            required=key in required_evaluation_fields,
        )

    bic_refit_ari = _numeric(summary_row, "bic_refit_ari")
    if "bic_refit_ari" in evaluation_row.index:
        _assert_close(
            f"{tumor_id} simulation_eval bic_refit_ari",
            _numeric(evaluation_row, "bic_refit_ari"),
            bic_refit_ari,
        )

    solver_contract = _selected_solver_contract(
        summary_row=summary_row,
        selected_row=selected_search_row,
        tumor_id=tumor_id,
        num_mutations=case.data.num_mutations,
    )
    if strict_artifacts and not bool(solver_contract["strict_solver_contract"]):
        failures = [
            key for key in STRICT_SOLVER_CHECKS if not bool(solver_contract[key])
        ]
        raise ValueError(
            f"{tumor_id}: requested guided-ADMM solver contract failed: {failures}"
        )

    path_lambdas = pd.to_numeric(
        search.get("lambda", pd.Series(dtype=float)), errors="coerce"
    )
    path_lambdas = path_lambdas[np.isfinite(path_lambdas)]
    source_column = search.get(
        "candidate_pool_source", pd.Series("", index=search.index)
    ).astype(str)
    solver_column = search.get(
        "inner_solver", pd.Series("", index=search.index)
    ).astype(str)

    snr = (
        case.min_true_linf_separation / raw_rmse
        if np.isfinite(case.min_true_linf_separation) and raw_rmse > 0.0
        else float("nan")
    )
    return {
        "tumor_id": tumor_id,
        "S": case.data.num_regions,
        "depth": case.depth,
        "nominal_purity": case.nominal_purity,
        "amp_rate": case.amp_rate,
        "lambda_mut": case.lambda_mut,
        "replicate": case.replicate,
        "purity_replicate": f"{case.nominal_purity:g}_rep{case.replicate}",
        "S_depth": f"S{case.data.num_regions}_depth{case.depth}",
        "num_mutations": case.data.num_mutations,
        "num_observations": int(np.sum(case.data.count_observed)),
        "ccf_cells": ccf_cells,
        "mean_purity": float(np.mean(case.data.purity)),
        "asymmetric_copy_cells": int(confusion["n"]),
        "estimable_copy_cells": int(estimable_confusion["n"]),
        "true_k": case.true_k,
        "estimated_k": estimated_k,
        "k_error": estimated_k - case.true_k,
        "ARI": ari,
        "bic_refit_ari": bic_refit_ari,
        "cp_rmse": summary_rmse,
        "raw_cp_rmse": raw_rmse,
        "summary_cp_rmse": summary_rmse,
        "bic_refit_cp_rmse": bic_rmse,
        "raw_ccf_sse": raw_ccf_sse,
        "summary_ccf_sse": summary_ccf_sse,
        "bic_refit_ccf_sse": bic_refit_ccf_sse,
        "estimated_clonal_fraction": estimated_clonal_fraction,
        "true_clonal_fraction": true_clonal_fraction,
        "clonal_fraction_error": estimated_clonal_fraction - true_clonal_fraction,
        "multiplicity_asymmetric_f1": confusion["macro_f1"],
        "multiplicity_major_f1": confusion["major_f1"],
        "multiplicity_minor_f1": confusion["minor_f1"],
        "multiplicity_accuracy": confusion["micro_f1_accuracy"],
        "multiplicity_estimable_f1": estimable_confusion["macro_f1"],
        "mult_tp_major": confusion["tp_major"],
        "mult_tn_minor": confusion["tn_minor"],
        "mult_fp_major": confusion["fp_major"],
        "mult_fn_major": confusion["fn_major"],
        "estimable_mult_tp_major": estimable_confusion["tp_major"],
        "estimable_mult_tn_minor": estimable_confusion["tn_minor"],
        "estimable_mult_fp_major": estimable_confusion["fp_major"],
        "estimable_mult_fn_major": estimable_confusion["fn_major"],
        "true_min_linf_separation": case.min_true_linf_separation,
        "separation_to_raw_rmse": snr,
        **oracle_noise,
        **solver_contract,
        "converged_outer": _boolean(summary_row, "converged_outer"),
        "global_optimality_certified": _boolean(
            summary_row, "global_optimality_certified"
        ),
        "lambda_proposal_reason": str(summary_row.get("lambda_proposal_reason", "")),
        "adaptive_search_stop_reason": str(
            summary_row.get("adaptive_search_stop_reason", "")
        ),
        "selection_boundary_unresolved": _boolean(
            summary_row, "selection_boundary_unresolved"
        ),
        "selection_optimum_resolved": _boolean(
            summary_row, "selection_optimum_resolved"
        ),
        "path_rows": int(len(search)),
        "path_unique_lambdas": int(path_lambdas.nunique()),
        "path_lambda_min": float(path_lambdas.min())
        if len(path_lambdas)
        else float("nan"),
        "path_lambda_max": float(path_lambdas.max())
        if len(path_lambdas)
        else float("nan"),
        "path_raw_rows": int((source_column == "raw_fused_lambda_path").sum()),
        "path_admm_rows": int((solver_column == "admm_complete_graph").sum()),
        "adaptive_search_rounds": int(
            _numeric(summary_row, "adaptive_search_rounds_completed", 0.0)
        ),
        "selection_elapsed_seconds": _numeric(summary_row, "selection_elapsed_seconds"),
        "elapsed_seconds": _numeric(summary_row, "elapsed_seconds"),
        "internal_elapsed_seconds": _numeric(summary_row, "elapsed_seconds"),
        "input_data_hash": str(summary_row.get("input_data_hash", "")),
    }


def _pooled_rmse(group: pd.DataFrame, sse_column: str) -> float:
    sse = pd.to_numeric(group[sse_column], errors="coerce").to_numpy(dtype=float)
    cells = pd.to_numeric(group["ccf_cells"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(sse) & np.isfinite(cells) & (sse >= 0.0) & (cells > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.sum(sse[valid]) / np.sum(cells[valid])))


def _aggregate_group(group: pd.DataFrame) -> dict[str, float | int]:
    k_error = group["k_error"].to_numpy(dtype=float)
    clonal_error = group["clonal_fraction_error"].to_numpy(dtype=float)
    eligible_mult = group.loc[group["asymmetric_copy_cells"] > 0]
    eligible_estimable_mult = group.loc[group["estimable_copy_cells"] > 0]
    return {
        "tumors": int(len(group)),
        "ARI_mean": float(group["ARI"].mean()),
        "ARI_median": float(group["ARI"].median()),
        "ARI_sd": float(group["ARI"].std(ddof=1)),
        "bic_refit_ari_mean": _safe_mean(group["bic_refit_ari"]),
        # Compatibility aliases retain the prior TSV schema; the explicit
        # tumor-macro names prevent these means being mistaken for pooled RMSE.
        "cp_rmse_mean": float(group["cp_rmse"].mean()),
        "raw_cp_rmse_mean": float(group["raw_cp_rmse"].mean()),
        "summary_cp_rmse_mean": float(group["summary_cp_rmse"].mean()),
        "bic_refit_cp_rmse_mean": float(group["bic_refit_cp_rmse"].mean()),
        "cp_rmse_tumor_macro": float(group["cp_rmse"].mean()),
        "raw_cp_rmse_tumor_macro": float(group["raw_cp_rmse"].mean()),
        "summary_cp_rmse_tumor_macro": float(group["summary_cp_rmse"].mean()),
        "bic_refit_cp_rmse_tumor_macro": float(group["bic_refit_cp_rmse"].mean()),
        "cp_rmse_pooled": _pooled_rmse(group, "summary_ccf_sse"),
        "raw_cp_rmse_pooled": _pooled_rmse(group, "raw_ccf_sse"),
        "summary_cp_rmse_pooled": _pooled_rmse(group, "summary_ccf_sse"),
        "bic_refit_cp_rmse_pooled": _pooled_rmse(group, "bic_refit_ccf_sse"),
        "true_k_mean": float(group["true_k"].mean()),
        "estimated_k_mean": float(group["estimated_k"].mean()),
        "k_bias": float(np.mean(k_error)),
        "k_mae": float(np.mean(np.abs(k_error))),
        "k_rmse": float(np.sqrt(np.mean(np.square(k_error)))),
        "k_exact": int(np.sum(k_error == 0)),
        "k_over": int(np.sum(k_error > 0)),
        "k_under": int(np.sum(k_error < 0)),
        "estimated_clonal_fraction_mean": float(
            group["estimated_clonal_fraction"].mean()
        ),
        "true_clonal_fraction_mean": float(group["true_clonal_fraction"].mean()),
        "clonal_fraction_bias": float(np.mean(clonal_error)),
        "clonal_fraction_mae": float(np.mean(np.abs(clonal_error))),
        "clonal_fraction_rmse": float(np.sqrt(np.mean(np.square(clonal_error)))),
        "multiplicity_eligible_tumors": int(len(eligible_mult)),
        "multiplicity_tumor_macro_f1": _safe_mean(
            eligible_mult["multiplicity_asymmetric_f1"]
        ),
        "multiplicity_estimable_cells": int(group["estimable_copy_cells"].sum()),
        "multiplicity_estimable_eligible_tumors": int(len(eligible_estimable_mult)),
        "multiplicity_estimable_tumor_macro_f1": _safe_mean(
            eligible_estimable_mult["multiplicity_estimable_f1"]
        ),
        "min_true_linf_separation_mean": float(
            group["true_min_linf_separation"].mean()
        ),
        "separation_to_raw_rmse_mean": float(group["separation_to_raw_rmse"].mean()),
        "oracle_fisher_se_mean": _safe_mean(group["oracle_fisher_se_mean"]),
        "oracle_fisher_se_median": float(
            pd.to_numeric(group["oracle_fisher_se_median"], errors="coerce").median()
        ),
        "oracle_zero_information_cluster_regions": int(
            pd.to_numeric(
                group["oracle_zero_information_cluster_regions"], errors="coerce"
            ).sum()
        ),
        "oracle_weakest_pair_mahalanobis_mean": _safe_mean(
            group["oracle_weakest_pair_mahalanobis"]
        ),
        "oracle_weakest_pair_mahalanobis_median": float(
            pd.to_numeric(
                group["oracle_weakest_pair_mahalanobis"], errors="coerce"
            ).median()
        ),
        "oracle_weakest_pair_max_region_z_mean": _safe_mean(
            group["oracle_weakest_pair_max_region_z"]
        ),
        "oracle_weakest_pair_max_region_z_median": float(
            pd.to_numeric(
                group["oracle_weakest_pair_max_region_z"], errors="coerce"
            ).median()
        ),
        "elapsed_seconds_sum": float(group["elapsed_seconds"].sum()),
        "elapsed_seconds_mean": float(group["elapsed_seconds"].mean()),
        "elapsed_seconds_median": float(group["elapsed_seconds"].median()),
        "internal_elapsed_seconds_sum": float(group["internal_elapsed_seconds"].sum()),
        "internal_elapsed_seconds_mean": float(
            group["internal_elapsed_seconds"].mean()
        ),
        "internal_elapsed_seconds_median": float(
            group["internal_elapsed_seconds"].median()
        ),
        "path_unique_lambdas_mean": float(group["path_unique_lambdas"].mean()),
        "selected_lambda_median": float(group["selected_lambda"].median()),
        "selected_kkt_residual_max": float(group["selected_kkt_residual"].max()),
        "strict_solver_contract_count": int(group["strict_solver_contract"].sum()),
    }


def _group_metrics(per_tumor: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = [{"group": "Overall", **_aggregate_group(per_tumor)}]
    for value, group in per_tumor.groupby(column, sort=True):
        rows.append({"group": f"{column}={value}", **_aggregate_group(group)})
    return pd.DataFrame(rows)


def _solver_audit(per_tumor: pd.DataFrame) -> pd.DataFrame:
    checks = (
        "positive_lambda",
        "raw_candidate_source",
        "requested_selection_provenance",
        "requested_initializer_provenance",
        "complete_graph",
        "complete_edge_count",
        "backend_neutral_exact_provenance",
        "explicit_exactness_provenance",
        "legacy_dense_exactness_provenance",
        "objective_faithful",
        "full_kkt_certified",
        "certificate_status_accepted",
        "certificate_scope_full_original_graph",
        "certificate_gradient_observed_objective",
        "kkt_at_or_below_certificate_tolerance",
        "admm_solver",
        "admm_used",
        "kkt_finite",
        "kkt_at_or_below_tol",
        "kkt_at_or_below_selection_gate",
        "raw_kkt_eligible",
        "converged",
        "converged_outer",
        "stationarity_certified",
        "global_optimality_certified",
        "selection_eligible",
        "lambda_online",
        "postselection_truth_only",
        "strict_solver_contract",
    )
    output: list[dict[str, object]] = []
    groups = [("Overall", per_tumor)] + [
        (f"S={value}", group) for value, group in per_tumor.groupby("S", sort=True)
    ]
    for label, group in groups:
        row: dict[str, object] = {"group": label, "tumors": len(group)}
        for check in checks:
            row[f"{check}_count"] = int(group[check].sum())
            row[f"{check}_rate"] = float(group[check].mean())
        row["admm_iterations_mean"] = float(group["admm_iterations"].mean())
        row["admm_iterations_min"] = int(group["admm_iterations"].min())
        row["backend_iterations_mean"] = float(group["backend_iterations"].mean())
        row["backend_iterations_min"] = int(group["backend_iterations"].min())
        row["selected_kkt_median"] = float(group["selected_kkt_residual"].median())
        row["selected_kkt_max"] = float(group["selected_kkt_residual"].max())
        output.append(row)
    return pd.DataFrame(output)


def _correlations(per_tumor: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("ARI", "true_min_linf_separation"),
        ("ARI", "raw_cp_rmse"),
        ("ARI", "separation_to_raw_rmse"),
        ("ARI", "oracle_fisher_se_mean"),
        ("ARI", "oracle_weakest_pair_mahalanobis"),
        ("ARI", "oracle_weakest_pair_max_region_z"),
        ("true_min_linf_separation", "raw_cp_rmse"),
    )
    rows: list[dict[str, object]] = []
    groups = [("Overall", per_tumor)] + [
        (f"S={value}", group) for value, group in per_tumor.groupby("S", sort=True)
    ]
    for group_label, group in groups:
        for left, right in pairs:
            values = group[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
            if (
                len(values) >= 3
                and values[left].nunique() > 1
                and values[right].nunique() > 1
            ):
                pearson = float(values[left].corr(values[right], method="pearson"))
                left_ranks = values[left].rank(method="average")
                right_ranks = values[right].rank(method="average")
                spearman = float(left_ranks.corr(right_ranks, method="pearson"))
                rows.append(
                    {
                        "group": group_label,
                        "left": left,
                        "right": right,
                        "n": len(values),
                        "pearson_r": pearson,
                        "spearman_rho": spearman,
                    }
                )
    return pd.DataFrame(rows)


def _baseline_metrics(
    *,
    directory: Path,
    label: str,
    note: str,
    current_ids: set[str],
) -> pd.DataFrame:
    summary_path = directory / "single_stage_summary.tsv"
    if not summary_path.is_file():
        return pd.DataFrame()
    summary = _read_tsv(summary_path)
    summary["tumor_id"] = summary["tumor_id"].astype(str)
    summary = summary.loc[summary["tumor_id"].isin(current_ids)].copy()
    if summary["tumor_id"].nunique() != len(current_ids):
        return pd.DataFrame()
    parsed = summary["tumor_id"].map(_parse_tumor_id)
    summary["S"] = [int(item["S"]) for item in parsed]
    summary["true_k"] = [int(item["name_k"]) for item in parsed]
    summary["estimated_k"] = pd.to_numeric(summary["n_clusters"], errors="coerce")
    summary["k_error"] = summary["estimated_k"] - summary["true_k"]
    summary["ARI"] = pd.to_numeric(summary.get("ARI"), errors="coerce")
    for column in ("cp_rmse", "raw_cp_rmse", "summary_cp_rmse", "bic_refit_cp_rmse"):
        summary[column] = pd.to_numeric(summary.get(column), errors="coerce")
    summary["clonal_fraction_error"] = pd.to_numeric(
        summary.get("clonal_fraction_error"), errors="coerce"
    )
    summary["elapsed_seconds"] = pd.to_numeric(
        summary.get("elapsed_seconds"), errors="coerce"
    )
    rows: list[dict[str, object]] = []
    groups = [("Overall", summary)] + [
        (f"S={value}", group) for value, group in summary.groupby("S", sort=True)
    ]
    for group_label, group in groups:
        k_error = group["k_error"].to_numpy(dtype=float)
        clonal = group["clonal_fraction_error"].to_numpy(dtype=float)
        rows.append(
            {
                "run": label,
                "status_note": note,
                "group": group_label,
                "tumors": len(group),
                "ARI_mean": float(group["ARI"].mean()),
                "cp_rmse_mean": float(group["cp_rmse"].mean()),
                "raw_cp_rmse_mean": float(group["raw_cp_rmse"].mean()),
                "summary_cp_rmse_mean": float(group["summary_cp_rmse"].mean()),
                "bic_refit_cp_rmse_mean": float(group["bic_refit_cp_rmse"].mean()),
                "true_k_mean": float(group["true_k"].mean()),
                "estimated_k_mean": float(group["estimated_k"].mean()),
                "k_bias": float(np.nanmean(k_error)),
                "k_mae": float(np.nanmean(np.abs(k_error))),
                "k_rmse": float(np.sqrt(np.nanmean(np.square(k_error)))),
                "k_exact": int(np.sum(k_error == 0)),
                "k_over": int(np.sum(k_error > 0)),
                "k_under": int(np.sum(k_error < 0)),
                "clonal_fraction_bias": float(np.nanmean(clonal)),
                "clonal_fraction_mae": float(np.nanmean(np.abs(clonal))),
                "elapsed_seconds_sum": float(group["elapsed_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _current_comparison(per_tumor: pd.DataFrame, label: str, note: str) -> pd.DataFrame:
    by_s = _group_metrics(per_tumor, "S")
    columns = (
        "group",
        "tumors",
        "ARI_mean",
        "cp_rmse_mean",
        "raw_cp_rmse_mean",
        "summary_cp_rmse_mean",
        "bic_refit_cp_rmse_mean",
        "true_k_mean",
        "estimated_k_mean",
        "k_bias",
        "k_mae",
        "k_rmse",
        "k_exact",
        "k_over",
        "k_under",
        "clonal_fraction_bias",
        "clonal_fraction_mae",
        "elapsed_seconds_sum",
    )
    result = by_s.loc[:, columns].copy()
    result.insert(0, "status_note", note)
    result.insert(0, "run", label)
    return result


def _fmt(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(
    frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 6
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join("---:" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(_fmt(row[column], digits) for column, _ in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *rows])


def _factor_counts(frame: pd.DataFrame, column: str) -> str:
    counts = frame[column].value_counts().sort_index()
    return ", ".join(f"{value:g}={int(count)}" for value, count in counts.items())


def _build_report(
    *,
    run_label: str,
    per_tumor: pd.DataFrame,
    by_s: pd.DataFrame,
    by_depth: pd.DataFrame,
    audit: pd.DataFrame,
    confusion: pd.DataFrame,
    correlations: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    overall = by_s.iloc[0]
    overall_audit = audit.iloc[0]
    current_comparison = comparison.loc[
        (comparison["run"] == run_label) & (comparison["group"] == "Overall")
    ].iloc[0]
    sample_values = sorted(int(value) for value in per_tumor["S"].unique())
    historical = comparison.loc[comparison["run"] != run_label]
    historical_lines: list[str] = []
    if not historical.empty:
        historical_lines = [
            "## Historical comparisons",
            "",
            "Historical baselines are shown only when an explicitly supplied baseline contains the complete current cohort. Runtime may not be comparable across hardware or search budgets.",
            "",
            _markdown_table(
                comparison.loc[comparison["group"] == "Overall"],
                [
                    ("run", "Run"),
                    ("ARI_mean", "ARI"),
                    ("cp_rmse_mean", "Tumor-macro CCF RMSE"),
                    ("estimated_k_mean", "Estimated K"),
                    ("k_bias", "K bias"),
                    ("k_mae", "K MAE"),
                    ("k_rmse", "K RMSE"),
                    ("k_exact", "Exact"),
                    ("k_over", "Over"),
                    ("k_under", "Under"),
                ],
            ),
            "",
        ]
    comparison_file_lines = (
        [
            "- `guided_admm_historical_comparison.tsv`: current run and explicitly supplied complete-cohort baselines.",
        ]
        if not historical.empty
        else []
    )
    lines = [
        f"# CliPP2 {len(per_tumor):,}-tumor benchmark: {run_label}",
        "",
        "> Status: diagnostic unless the run label explicitly says final. The solver audit below is about framework compliance; it does not establish that the lambda search explored alternatives to the initializer.",
        "",
        "## Cohort integrity",
        "",
        f"- {len(per_tumor)} unique tumors; S counts: "
        + _factor_counts(per_tumor, "S")
        + ".",
        f"- {int(per_tumor['num_mutations'].sum()):,} mutations and {int(per_tumor['num_observations'].sum()):,} observed mutation-region cells.",
        f"- Depth counts: {_factor_counts(per_tumor, 'depth')}.",
        f"- Mutation-rate setting (`Lm`) counts: {_factor_counts(per_tumor, 'lambda_mut')}.",
        f"- CNA setting: {int((per_tumor['amp_rate'] == 0).sum())} CNA-free and {int((per_tumor['amp_rate'] > 0).sum())} CNA-altered tumors.",
        "- The input IDs form one complete Cartesian design over depth, purity-replicate, CNA rate, sample count, and mutation-rate setting.",
        "- Primary ARI, all CCF RMSE variants, clonal fraction, and multiplicity values were independently reconstructed from per-tumor artifacts and matched the cohort summary and available simulation-evaluation fields.",
        "- BIC-refit ARI is retained from the post-selection run summary because the current output schema does not serialize the separate BIC partition labels; it is additionally cross-checked whenever `simulation_eval.tsv` supplies that field.",
        "",
        "## Performance by number of samples",
        "",
        _markdown_table(
            by_s,
            [
                ("group", "Group"),
                ("ARI_mean", "ARI"),
                ("bic_refit_ari_mean", "BIC-refit ARI"),
                ("cp_rmse_tumor_macro", "Summary RMSE (tumor-macro)"),
                ("cp_rmse_pooled", "Summary RMSE (pooled cells)"),
                ("raw_cp_rmse_tumor_macro", "Raw RMSE (tumor-macro)"),
                ("raw_cp_rmse_pooled", "Raw RMSE (pooled cells)"),
                (
                    "bic_refit_cp_rmse_tumor_macro",
                    "BIC-refit RMSE (tumor-macro)",
                ),
                ("bic_refit_cp_rmse_pooled", "BIC-refit RMSE (pooled cells)"),
                ("true_k_mean", "True K"),
                ("estimated_k_mean", "Estimated K"),
                ("k_bias", "K bias"),
                ("k_mae", "K MAE"),
                ("k_rmse", "K RMSE"),
                ("k_exact", "Exact"),
                ("k_over", "Over"),
                ("k_under", "Under"),
            ],
        ),
        "",
        "The legacy `cp_rmse` is the partition-summary RMSE. Tumor-macro RMSE gives every tumor equal weight; pooled-cell RMSE is reconstructed from summed squared error and mutation-region cell counts. Raw penalized and conditional BIC-refit estimates are reported separately.",
        "",
        "## Clonal fraction and multiplicity",
        "",
        _markdown_table(
            by_s,
            [
                ("group", "Group"),
                ("estimated_clonal_fraction_mean", "Estimated clonal"),
                ("true_clonal_fraction_mean", "True clonal"),
                ("clonal_fraction_bias", "Bias"),
                ("clonal_fraction_mae", "MAE"),
                ("clonal_fraction_rmse", "RMSE"),
                ("multiplicity_eligible_tumors", "F1 tumors"),
                ("multiplicity_tumor_macro_f1", "Multiplicity tumor-macro F1"),
                ("multiplicity_estimable_cells", "Estimable cells"),
                (
                    "multiplicity_estimable_tumor_macro_f1",
                    "Estimable tumor-macro F1",
                ),
            ],
        ),
        "",
        "Multiplicity is scored exactly where `major_cn != minor_cn`; symmetric copy states do not enter the score.",
        "",
        _markdown_table(
            confusion,
            [
                ("group", "Group"),
                ("n", "Cells"),
                ("macro_f1", "Pooled macro F1"),
                ("micro_f1_accuracy", "Micro F1 / accuracy"),
                ("major_f1", "Major F1"),
                ("minor_f1", "Minor F1"),
                ("tp_major", "TP"),
                ("tn_minor", "TN"),
                ("fp_major", "FP"),
                ("fn_major", "FN"),
                ("estimable_n", "Estimable cells"),
                ("estimable_macro_f1", "Estimable pooled macro F1"),
            ],
        ),
        "",
        "The estimable-only subset additionally requires `major_cn > minor_cn > 0`; minor-CN-zero cells are deterministic major-copy calls.",
        "",
        "## Pairwise-fusion exact-solver audit",
        "",
        f"- Positive selected lambda: {int(overall_audit['positive_lambda_count'])}/{len(per_tumor)}.",
        f"- Requested online partition-guided/partition-ICL selection and Ward/CEM initializer provenance: {int(overall_audit['requested_selection_provenance_count'])}/{len(per_tumor)} and {int(overall_audit['requested_initializer_provenance_count'])}/{len(per_tumor)}.",
        f"- Complete pairwise graph, exact M(M-1)/2 edge count, and accepted backend-neutral exactness provenance: {int(overall_audit['complete_graph_count'])}/{len(per_tumor)}, {int(overall_audit['complete_edge_count_count'])}/{len(per_tumor)}, and {int(overall_audit['backend_neutral_exact_provenance_count'])}/{len(per_tumor)}.",
        f"- Schema-v1 / legacy dense exactness provenance: {int(overall_audit['explicit_exactness_provenance_count'])}/{int(overall_audit['legacy_dense_exactness_provenance_count'])}; backend-iteration range {int(per_tumor['backend_iterations'].min())}-{int(per_tumor['backend_iterations'].max())}.",
        f"- Dense-ADMM diagnostic usage (not an eligibility requirement): {int((per_tumor['admm_iterations'] > 0).sum())}/{len(per_tumor)}; iteration range {int(per_tumor['admm_iterations'].min())}-{int(per_tumor['admm_iterations'].max())}.",
        f"- Finite KKT / raw-KKT eligible / stationarity certified: {int(overall_audit['kkt_finite_count'])}/{len(per_tumor)}, {int(overall_audit['raw_kkt_eligible_count'])}/{len(per_tumor)}, {int(overall_audit['stationarity_certified_count'])}/{len(per_tumor)}.",
        f"- Selected KKT at or below the configured solver tolerance: {int(overall_audit['kkt_at_or_below_tol_count'])}/{len(per_tumor)}; median/max residual {_fmt(overall_audit['selected_kkt_median'])}/{_fmt(overall_audit['selected_kkt_max'])}.",
        f"- Selected KKT at or below the explicit 5 x tolerance eligibility gate: {int(overall_audit['kkt_at_or_below_selection_gate_count'])}/{len(per_tumor)}.",
        f"- Selected KKT at or below its recorded full-certificate tolerance: {int(overall_audit['kkt_at_or_below_certificate_tolerance_count'])}/{len(per_tumor)}.",
        f"- Non-prespecified online lambda provenance: {int(overall_audit['lambda_online_count'])}/{len(per_tumor)}.",
        f"- Truth evaluated only after candidate selection: {int(overall_audit['postselection_truth_only_count'])}/{len(per_tumor)}.",
        f"- Full strict solver contract: {int(overall_audit['strict_solver_contract_count'])}/{len(per_tumor)}.",
        f"- Distinct evaluated lambdas per tumor: mean {_fmt(per_tumor['path_unique_lambdas'].mean(), 3)}, range {int(per_tumor['path_unique_lambdas'].min())}-{int(per_tumor['path_unique_lambdas'].max())}.",
        f"- Lambda-selection boundary resolved/unresolved: {int(per_tumor['selection_optimum_resolved'].sum())}/{int(per_tumor['selection_boundary_unresolved'].sum())}; unresolved by S: "
        + ", ".join(
            f"{s}={int(per_tumor.loc[per_tumor['S'] == s, 'selection_boundary_unresolved'].sum())}"
            for s in sample_values
        )
        + ".",
        "",
        "Framework compliance and lambda-search completeness are separate. All selected fits pass the implemented solver gate, but an unresolved boundary means the online search stopped before it certified the whole local score basin.",
        "The output field `global_optimality_certified` is conditional on the configured unimodality assumption for the observed major/minor mixture; stationarity is checked, but global unimodality is not mathematically proved and this field is not part of strict selection eligibility.",
        "",
        "## Distinguishability and noise",
        "",
        _markdown_table(
            by_s,
            [
                ("group", "Group"),
                ("min_true_linf_separation_mean", "Mean min true L-inf separation"),
                ("raw_cp_rmse_mean", "Raw CCF RMSE (tumor-macro)"),
                ("separation_to_raw_rmse_mean", "Separation / raw RMSE"),
                ("oracle_fisher_se_mean", "Oracle Fisher SE"),
                (
                    "oracle_weakest_pair_mahalanobis_mean",
                    "Oracle weakest-pair Mahalanobis",
                ),
                (
                    "oracle_weakest_pair_max_region_z_mean",
                    "Oracle weakest-pair max-region z",
                ),
            ],
        ),
        "",
        f"Oracle cluster-region cells with zero binomial information: {int(overall['oracle_zero_information_cluster_regions'])}.",
        "",
        _markdown_table(
            correlations.loc[correlations["group"] == "Overall"],
            [
                ("left", "Left"),
                ("right", "Right"),
                ("n", "N"),
                ("pearson_r", "Pearson r"),
                ("spearman_rho", "Spearman rho"),
            ],
        ),
        "",
        "The separation/raw-RMSE ratio is descriptive and outcome-dependent, not an uncertainty calibration. The oracle metrics instead pool binomial Fisher information within each true cluster and region using true CCF and multiplicity plus input depth/scaling. They measure input-derived distinguishability for simulation only; they are not CliPP2 posterior uncertainty. The weakest-pair summaries take the minimum across true-cluster pairs of, respectively, the multiregion standardized Mahalanobis distance and the largest single-region z score.",
        "",
        "## Internal runtime and path",
        "",
        f"- Summed internal pipeline runtime: {_fmt(overall['internal_elapsed_seconds_sum'], 3)} seconds ({_fmt(overall['internal_elapsed_seconds_sum'] / 60.0, 3)} minutes).",
        f"- Mean/median/p95/max internal runtime per tumor: {_fmt(per_tumor['internal_elapsed_seconds'].mean(), 3)} / {_fmt(per_tumor['internal_elapsed_seconds'].median(), 3)} / {_fmt(per_tumor['internal_elapsed_seconds'].quantile(0.95), 3)} / {_fmt(per_tumor['internal_elapsed_seconds'].max(), 3)} seconds.",
        "- Internal runtime excludes worker launch, output serialization, retries, and scheduler overhead; use scheduler status/failure logs for end-to-end benchmark time.",
        f"- Median selected lambda: {_fmt(per_tumor['selected_lambda'].median(), 6)}; range {_fmt(per_tumor['selected_lambda'].min(), 6)}-{_fmt(per_tumor['selected_lambda'].max(), 6)}.",
        "",
        *historical_lines,
        "## Files",
        "",
        "- `guided_admm_benchmark_per_tumor.tsv`: independently reconstructed metrics and per-tumor provenance.",
        "- `guided_admm_metrics_by_*.tsv`: performance grouped by S, depth, mutation-rate setting, nominal purity, replicate, joint purity-replicate cell, S-by-depth cell, CNA rate, and true K.",
        "- `guided_admm_solver_audit_by_S.tsv`: strict framework audit.",
        "- `guided_admm_multiplicity_confusion.tsv`: pooled major/minor confusion and F1.",
        "- `guided_admm_distinguishability_correlations.tsv`: Pearson/Spearman analyses overall and within S.",
        *comparison_file_lines,
        "",
        f"Current overall ARI={_fmt(current_comparison['ARI_mean'])}, tumor-macro CCF RMSE={_fmt(current_comparison['cp_rmse_mean'])}, and K bias={_fmt(current_comparison['k_bias'])}.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=HERE / "sim_v2_guided_admm_out"
    )
    parser.add_argument("--inputs-dir", type=Path, default=HERE / "CliPP2Sim_v2_TSV")
    parser.add_argument("--truth-root", type=Path, default=HERE / "CliPP2Sim_v2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--partition-baseline-dir",
        type=Path,
        default=None,
        help="Optional complete-cohort partition baseline",
    )
    parser.add_argument(
        "--original-baseline-dir",
        type=Path,
        default=None,
        help="Optional complete-cohort pre-revision baseline",
    )
    parser.add_argument("--run-label", default="diagnostic guided-ADMM")
    parser.add_argument("--expected-tumors", type=int, default=120)
    parser.add_argument(
        "--expected-per-s",
        type=int,
        default=None,
        help="Optional expected count for every inferred S level",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(
        results_dir,
        expected_tumors=int(args.expected_tumors),
        allow_incomplete=args.allow_incomplete,
    )
    if not args.allow_incomplete and len(summary) != args.expected_tumors:
        raise ValueError(
            f"Expected {args.expected_tumors} summary rows, found {len(summary)}"
        )
    cases = _load_cases(
        inputs_dir=args.inputs_dir.resolve(),
        truth_root=args.truth_root.resolve(),
        summary_ids=set(summary["tumor_id"]),
        expected_tumors=args.expected_tumors,
        expected_per_s=args.expected_per_s,
        allow_incomplete=args.allow_incomplete,
    )
    indexed_summary = summary.set_index("tumor_id", drop=False)
    records = [
        _recompute_case(
            case=cases[tumor_id],
            summary_row=indexed_summary.loc[tumor_id],
            results_dir=results_dir,
            strict_artifacts=not args.allow_incomplete,
        )
        for tumor_id in sorted(cases)
    ]
    per_tumor = (
        pd.DataFrame(records)
        .sort_values(["S", "depth", "lambda_mut", "tumor_id"])
        .reset_index(drop=True)
    )
    grouped_metrics = {
        "S": _group_metrics(per_tumor, "S"),
        "depth": _group_metrics(per_tumor, "depth"),
        "lambda_mut": _group_metrics(per_tumor, "lambda_mut"),
        "nominal_purity": _group_metrics(per_tumor, "nominal_purity"),
        "replicate": _group_metrics(per_tumor, "replicate"),
        "purity_replicate": _group_metrics(per_tumor, "purity_replicate"),
        "S_depth": _group_metrics(per_tumor, "S_depth"),
        "amp_rate": _group_metrics(per_tumor, "amp_rate"),
        "true_k": _group_metrics(per_tumor, "true_k"),
    }
    by_s = grouped_metrics["S"]
    by_depth = grouped_metrics["depth"]
    audit = _solver_audit(per_tumor)
    correlations = _correlations(per_tumor)

    confusion_rows: list[dict[str, object]] = []
    for label, group in [("Overall", per_tumor)] + [
        (f"S={value}", frame) for value, frame in per_tumor.groupby("S", sort=True)
    ]:
        exact = _sum_confusions(group)
        estimable = _sum_confusions(group, prefix="estimable_mult_")
        confusion_rows.append(
            {
                "group": label,
                **exact,
                **{f"estimable_{key}": value for key, value in estimable.items()},
            }
        )
    confusion = pd.DataFrame(confusion_rows)

    comparison_frames = [
        _current_comparison(
            per_tumor,
            args.run_label,
            "current diagnostic; label as final only after algorithmic acceptance",
        )
    ]
    if args.partition_baseline_dir is not None:
        partition = _baseline_metrics(
            directory=args.partition_baseline_dir.resolve(),
            label="obsolete partition-only",
            note="historical quality baseline; selected models were not positive-lambda ADMM fits",
            current_ids=set(per_tumor["tumor_id"]),
        )
        if not partition.empty:
            comparison_frames.append(partition)
    if args.original_baseline_dir is not None:
        original = _baseline_metrics(
            directory=args.original_baseline_dir.resolve(),
            label="original pre-revision",
            note="historical baseline; multiplicity placeholder omitted; runtime not comparable",
            current_ids=set(per_tumor["tumor_id"]),
        )
        if not original.empty:
            comparison_frames.append(original)
    comparison = pd.concat(comparison_frames, ignore_index=True, sort=False)

    outputs = {
        "guided_admm_benchmark_per_tumor.tsv": per_tumor,
        **{
            f"guided_admm_metrics_by_{column}.tsv": frame
            for column, frame in grouped_metrics.items()
        },
        "guided_admm_solver_audit_by_S.tsv": audit,
        "guided_admm_multiplicity_confusion.tsv": confusion,
        "guided_admm_distinguishability_correlations.tsv": correlations,
    }
    comparison_path = output_dir / "guided_admm_historical_comparison.tsv"
    if len(comparison_frames) > 1:
        outputs[comparison_path.name] = comparison
    elif comparison_path.is_file():
        comparison_path.unlink()
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, sep="\t", index=False)

    report = _build_report(
        run_label=args.run_label,
        per_tumor=per_tumor,
        by_s=by_s,
        by_depth=by_depth,
        audit=audit,
        confusion=confusion,
        correlations=correlations,
        comparison=comparison,
    )
    report_path = output_dir / "GUIDED_ADMM_BENCHMARK_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote report and {len(outputs)} TSV files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
