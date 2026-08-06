from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.fusion.path_summary import (
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.model import FitResult
from ..io.data import TumorData


@dataclass
class SimulationEvaluation:
    # Legacy cp_rmse is the partition-summary RMSE; explicit fields below separate sources.
    ari: float
    cp_rmse: float
    multiplicity_f1: float
    estimated_clonal_fraction: float
    true_clonal_fraction: float
    clonal_fraction_error: float
    true_clusters: int
    estimated_clusters: int
    n_eval_mutations: int
    n_filtered_mutations: int
    raw_cp_rmse: float | None = None
    summary_cp_rmse: float | None = None
    # BIC partition-refit metrics (None when refit was not performed)
    bic_refit_ari: float | None = None
    bic_refit_cp_rmse: float | None = None
    # multiplicity_f1 remains the compatibility name for the requested
    # asymmetric-copy metric. The explicit fields make both masks inspectable.
    multiplicity_asymmetric_f1: float | None = None
    multiplicity_estimable_f1: float | None = None
    # Tumor-directory simulations have continuous effective multiplicity.
    # The compatibility ``multiplicity_f1`` above remains the historical
    # asymmetric major/minor score; the explicitly named fields below are the
    # primary path-model multiplicity metrics.
    effective_multiplicity_rmse: float | None = None
    raw_effective_multiplicity_rmse: float | None = None
    summary_effective_multiplicity_rmse: float | None = None
    amplified_mutant_copy_f1: float | None = None
    raw_amplified_mutant_copy_f1: float | None = None
    summary_amplified_mutant_copy_f1: float | None = None
    n_effective_multiplicity_units: int = 0
    # Number of finite units entering amplified-copy classification, followed
    # by the number of positive truth labels among those units.
    n_amplified_mutant_copy_units: int = 0
    n_true_amplified_mutant_copy_units: int = 0


@dataclass(frozen=True)
class SimulationTruth:
    truth_clusters: np.ndarray
    truth_phi: np.ndarray
    truth_multiplicity: np.ndarray | None
    truth_effective_multiplicity: np.ndarray | None = None
    truth_mutant_copy_mass: np.ndarray | None = None


def _comb2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) * 0.5


def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true, dtype=np.int64).reshape(-1)
    labels_pred = np.asarray(labels_pred, dtype=np.int64).reshape(-1)
    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape.")

    n_regions = int(labels_true.size)
    if n_regions < 2:
        return 1.0

    _, true_inverse = np.unique(labels_true, return_inverse=True)
    _, pred_inverse = np.unique(labels_pred, return_inverse=True)
    n_true = int(true_inverse.max()) + 1
    n_pred = int(pred_inverse.max()) + 1

    pair_codes = true_inverse.astype(
        np.int64, copy=False
    ) * n_pred + pred_inverse.astype(np.int64, copy=False)
    pair_counts = np.bincount(pair_codes, minlength=n_true * n_pred)
    sum_comb_contingency = float(np.sum(_comb2(pair_counts[pair_counts > 0])))
    true_counts = np.bincount(true_inverse, minlength=n_true)
    pred_counts = np.bincount(pred_inverse, minlength=n_pred)
    sum_comb_true = float(np.sum(_comb2(true_counts)))
    sum_comb_pred = float(np.sum(_comb2(pred_counts)))
    total_comb = float(n_regions * (n_regions - 1) * 0.5)

    expected_index = (sum_comb_true * sum_comb_pred) / total_comb
    max_index = 0.5 * (sum_comb_true + sum_comb_pred)
    denom = max_index - expected_index
    if abs(denom) <= 1e-12:
        return 1.0
    return float((sum_comb_contingency - expected_index) / denom)


def _macro_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    matches = y_true == y_pred
    true_zero = y_true == 0
    pred_zero = y_pred == 0
    true_one = ~true_zero
    pred_one = ~pred_zero

    tp0 = float(np.sum(matches & true_zero))
    fp0 = float(np.sum(pred_zero & true_one))
    fn0 = float(np.sum(true_zero & pred_one))
    tp1 = float(np.sum(matches & true_one))
    fp1 = float(np.sum(pred_one & true_zero))
    fn1 = float(np.sum(true_one & pred_zero))

    def _f1(tp: float, fp: float, fn: float) -> float:
        precision_denom = tp + fp
        recall_denom = tp + fn
        precision = tp / precision_denom if precision_denom > 0.0 else 0.0
        recall = tp / recall_denom if recall_denom > 0.0 else 0.0
        if precision + recall <= 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    return float(0.5 * (_f1(tp0, fp0, fn0) + _f1(tp1, fp1, fn1)))


def _positive_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return positive-class F1, or NaN when truth contains no positives."""

    truth = np.asarray(y_true, dtype=bool).reshape(-1)
    prediction = np.asarray(y_pred, dtype=bool).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    true_positive = int(np.sum(truth & prediction))
    false_positive = int(np.sum(~truth & prediction))
    false_negative = int(np.sum(truth & ~prediction))
    if true_positive + false_negative == 0:
        return float("nan")
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def _multiplicity_f1_for_mask(
    *,
    truth_multiplicity: np.ndarray,
    predicted_multiplicity: np.ndarray,
    major_cn: np.ndarray,
    mask: np.ndarray,
) -> float:
    subject_mask = np.asarray(mask, dtype=bool)
    if not subject_mask.any():
        return float("nan")
    truth_major = np.isclose(
        truth_multiplicity[subject_mask],
        major_cn[subject_mask],
    )
    pred_major = np.isclose(
        predicted_multiplicity[subject_mask],
        major_cn[subject_mask],
    )
    return _macro_binary_f1(
        truth_major.astype(int).reshape(-1),
        pred_major.astype(int).reshape(-1),
    )


def _region_index_from_label(region_id: str) -> int:
    match = re.fullmatch(r"region([1-9]\d*)", str(region_id))
    if match is None:
        raise ValueError(f"Could not parse region index from '{region_id}'")
    return int(match.group(1)) - 1


def _reindex_by_mutation_id(
    values: np.ndarray,
    truth_ids: list[str] | None,
    data_ids: list[str],
    column_name: str,
    tumor_id: str,
) -> np.ndarray:
    """Reorder truth rows to match data_ids ordering.

    If truth_ids is None (truth file has no mutation_id column) the array is
    returned after a shape check using positional alignment.  When mutation IDs
    are present in the truth file, rows are reordered so alignment is ID-based,
    preventing silent misalignment when truth and data use different orderings.
    """
    if truth_ids is None:
        if values.shape[0] != len(data_ids):
            raise ValueError(
                f"Positional-alignment shape mismatch for '{column_name}' "
                f"in tumor '{tumor_id}': "
                f"{values.shape[0]} truth rows vs {len(data_ids)} mutations."
            )
        return values

    if len(truth_ids) != len(set(truth_ids)):
        raise ValueError(
            f"Duplicate mutation_id values found in truth '{column_name}' "
            f"for tumor '{tumor_id}'."
        )
    if values.shape[0] != len(truth_ids):
        raise ValueError(
            f"ID-alignment shape mismatch for '{column_name}' in tumor '{tumor_id}': "
            f"{values.shape[0]} truth rows vs {len(truth_ids)} mutation IDs."
        )
    if len(data_ids) != len(set(data_ids)):
        raise ValueError(
            f"Duplicate mutation_id values found in loaded data for tumor '{tumor_id}'."
        )
    data_id_set = set(data_ids)
    truth_id_to_index = {tid: i for i, tid in enumerate(truth_ids)}
    missing = [mid for mid in data_ids if mid not in truth_id_to_index]
    extra = [mid for mid in truth_ids if mid not in data_id_set]
    if missing:
        raise ValueError(
            f"Mutations in data not found in truth '{column_name}' "
            f"for tumor '{tumor_id}': {missing[:5]!r}"
        )
    if extra:
        raise ValueError(
            f"Mutations in truth '{column_name}' not found in data "
            f"for tumor '{tumor_id}': {extra[:5]!r}"
        )
    indices = np.array([truth_id_to_index[mid] for mid in data_ids], dtype=np.intp)
    return values[indices]


def _load_single_region_truth(
    tumor_dir: Path,
    data: TumorData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth_df = pd.read_csv(tumor_dir / "truth.txt", sep="\t")
    truth_ids: list[str] | None = (
        truth_df["mutation_id"].astype(str).tolist()
        if "mutation_id" in truth_df.columns
        else None
    )
    raw_clusters = truth_df["cluster_id"].to_numpy(dtype=int)
    truth_clusters = _reindex_by_mutation_id(
        raw_clusters,
        truth_ids,
        data.mutation_ids,
        "truth.txt/cluster_id",
        data.tumor_id,
    )

    truth_cp = pd.read_csv(tumor_dir / "truth_cp.txt", sep="\t")
    cp_ids: list[str] | None = (
        truth_cp["mutation_id"].astype(str).tolist()
        if "mutation_id" in truth_cp.columns
        else None
    )
    raw_phi = truth_cp["ccf"].to_numpy(dtype=np.float32)
    aligned_phi = _reindex_by_mutation_id(
        raw_phi, cp_ids, data.mutation_ids, "truth_cp.txt/ccf", data.tumor_id
    )
    truth_phi = np.zeros((data.num_mutations, 1), dtype=np.float32)
    truth_phi[:, 0] = aligned_phi

    truth_multiplicity = None
    cna_path = tumor_dir / "cna.txt"
    if cna_path.exists():
        cna = pd.read_csv(cna_path, sep="\t")
        if "multiplicity" in cna.columns and cna.shape[0] == data.num_mutations:
            truth_multiplicity = (
                cna["multiplicity"].to_numpy(dtype=np.float32).reshape(-1, 1)
            )

    return truth_clusters, truth_phi, truth_multiplicity


def _truth_sample_index(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return _region_index_from_label(str(value))


def _load_revised_mutation_sample_truth(
    *,
    tumor_dir: Path,
    data: TumorData,
    truth_clusters: np.ndarray,
) -> SimulationTruth:
    """Load continuous clone-mixture truth without reading observed CNA calls."""

    truth_path = tumor_dir / "truth_mutation_sample.tsv"
    frame = pd.read_csv(truth_path, sep="\t")
    required = {
        "mutation_id",
        "sample_id",
        "ccf",
        "mutant_copy_mass",
        "effective_multiplicity",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Revised simulation truth is missing columns {missing}: {truth_path}"
        )
    frame = frame.copy()
    frame["_mutation_id"] = frame["mutation_id"].astype(str)
    frame["_sample_index"] = [
        _truth_sample_index(value) for value in frame["sample_id"]
    ]
    if bool(frame.duplicated(["_mutation_id", "_sample_index"]).any()):
        raise ValueError(
            f"Duplicate mutation_id/sample_id rows in revised truth: {truth_path}"
        )
    expected_samples = {
        _region_index_from_label(region_id) for region_id in data.region_ids
    }
    if set(frame["_sample_index"]) != expected_samples:
        raise ValueError(
            f"Revised truth sample IDs do not match TumorData regions for "
            f"{data.tumor_id!r}."
        )
    expected_pairs = {
        (str(mutation_id), sample_index)
        for mutation_id in data.mutation_ids
        for sample_index in expected_samples
    }
    observed_pairs = set(zip(frame["_mutation_id"], frame["_sample_index"]))
    if observed_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs.difference(observed_pairs))[:5]
        extra_pairs = sorted(observed_pairs.difference(expected_pairs))[:5]
        raise ValueError(
            "Revised truth does not contain the complete mutation-region product; "
            f"missing={missing_pairs}, extra={extra_pairs}."
        )

    lookup = frame.set_index(["_mutation_id", "_sample_index"])
    shape = (data.num_mutations, data.num_regions)
    truth_phi = np.empty(shape, dtype=np.float64)
    truth_mass = np.empty(shape, dtype=np.float64)
    truth_effective = np.empty(shape, dtype=np.float64)
    for mutation_index, mutation_id in enumerate(data.mutation_ids):
        for region_index, region_id in enumerate(data.region_ids):
            sample_index = _region_index_from_label(region_id)
            row = lookup.loc[(str(mutation_id), sample_index)]
            truth_phi[mutation_index, region_index] = float(row["ccf"])
            truth_mass[mutation_index, region_index] = float(row["mutant_copy_mass"])
            truth_effective[mutation_index, region_index] = float(
                row["effective_multiplicity"]
            )

    for name, values in (
        ("ccf", truth_phi),
        ("mutant_copy_mass", truth_mass),
        ("effective_multiplicity", truth_effective),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Revised simulation truth {name} must be finite.")
    if np.any((truth_phi <= 0.0) | (truth_phi > 1.0 + 1e-8)):
        raise ValueError("Revised simulation truth CCF must lie in (0, 1].")
    if np.any(truth_mass <= 0.0) or np.any(truth_effective <= 0.0):
        raise ValueError(
            "Revised simulation mutant-copy mass and effective multiplicity "
            "must be positive."
        )
    if not np.allclose(
        truth_mass,
        truth_phi * truth_effective,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise ValueError(
            "Revised simulation effective multiplicity is inconsistent with "
            "mutant-copy mass / CCF."
        )
    return SimulationTruth(
        truth_clusters=truth_clusters,
        truth_phi=truth_phi,
        truth_multiplicity=None,
        truth_effective_multiplicity=truth_effective,
        truth_mutant_copy_mass=truth_mass,
    )


def _cluster_level_clonal_fraction(phi: np.ndarray, labels: np.ndarray) -> float:
    if phi.size == 0 or labels.size == 0:
        return float("nan")

    _, relabeled = np.unique(labels.astype(np.int64), return_inverse=True)

    num_clusters = int(relabeled.max()) + 1
    centers = np.zeros((num_clusters, phi.shape[1]), dtype=np.float32)
    np.add.at(centers, relabeled, phi.astype(np.float32, copy=False))
    counts = np.bincount(relabeled, minlength=num_clusters).astype(np.float32)
    centers /= np.clip(counts[:, None], 1.0, None)

    clonal_target = np.ones((phi.shape[1],), dtype=np.float32)
    rms_distance = np.sqrt(np.mean((centers - clonal_target[None, :]) ** 2, axis=1))
    clonal_label = int(np.argmin(rms_distance))
    return float(np.mean(relabeled == clonal_label))


def load_simulation_truth(
    data: TumorData,
    simulation_root: str | Path,
) -> SimulationTruth:
    tumor_dir = Path(simulation_root) / data.tumor_id
    if not tumor_dir.exists():
        raise FileNotFoundError(
            f"Simulation directory not found for tumor '{data.tumor_id}': {tumor_dir}"
        )

    truth_df = pd.read_csv(tumor_dir / "truth.txt", sep="\t")
    truth_ids: list[str] | None = (
        truth_df["mutation_id"].astype(str).tolist()
        if "mutation_id" in truth_df.columns
        else None
    )
    raw_clusters = truth_df["cluster_id"].to_numpy(dtype=int)
    truth_clusters = _reindex_by_mutation_id(
        raw_clusters,
        truth_ids,
        data.mutation_ids,
        "truth.txt/cluster_id",
        data.tumor_id,
    )
    if (tumor_dir / "truth_mutation_sample.tsv").is_file():
        return _load_revised_mutation_sample_truth(
            tumor_dir=tumor_dir,
            data=data,
            truth_clusters=truth_clusters,
        )

    if data.num_regions == 1 and (tumor_dir / "truth_cp.txt").exists():
        truth_clusters, truth_phi, truth_multiplicity = _load_single_region_truth(
            tumor_dir=tumor_dir, data=data
        )
        return SimulationTruth(
            truth_clusters=truth_clusters,
            truth_phi=truth_phi,
            truth_multiplicity=truth_multiplicity,
        )

    truth_phi = np.zeros((data.num_mutations, data.num_regions), dtype=np.float32)
    truth_multiplicity = np.zeros(
        (data.num_mutations, data.num_regions), dtype=np.float32
    )

    for column, region_id in enumerate(data.region_ids):
        _region_index_from_label(region_id)
        region_dir = tumor_dir / str(region_id)
        truth_cp = pd.read_csv(region_dir / "truth_cp.txt", sep="\t")
        cna = pd.read_csv(region_dir / "cna.txt", sep="\t")

        cp_ids: list[str] | None = (
            truth_cp["mutation_id"].astype(str).tolist()
            if "mutation_id" in truth_cp.columns
            else None
        )
        aligned_ccf = _reindex_by_mutation_id(
            truth_cp["ccf"].to_numpy(dtype=np.float32),
            cp_ids,
            data.mutation_ids,
            f"{region_id}/truth_cp.txt/ccf",
            data.tumor_id,
        )
        truth_phi[:, column] = aligned_ccf

        cna_ids: list[str] | None = (
            cna["mutation_id"].astype(str).tolist()
            if "mutation_id" in cna.columns
            else None
        )
        aligned_mult = _reindex_by_mutation_id(
            cna["multiplicity"].to_numpy(dtype=np.float32),
            cna_ids,
            data.mutation_ids,
            f"{region_id}/cna.txt/multiplicity",
            data.tumor_id,
        )
        truth_multiplicity[:, column] = aligned_mult

    return SimulationTruth(
        truth_clusters=truth_clusters,
        truth_phi=truth_phi,
        truth_multiplicity=truth_multiplicity,
    )


def _path_effective_predictions(
    *,
    fit: FitResult,
    data: TumorData,
    phi: np.ndarray,
    posterior: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    spec = data.path_likelihood
    if spec is None:
        multiplicity = np.asarray(fit.multiplicity_call, dtype=np.float64)
        return multiplicity, multiplicity > 1.0 + 1e-8
    evaluated_posterior = (
        path_posterior_at_phi_numpy(
            data,
            phi,
            eps=float(getattr(fit, "likelihood_eps", 1e-6)),
        )
        if posterior is None
        else np.asarray(posterior, dtype=np.float64)
    )
    observed = (
        np.ones_like(np.asarray(phi), dtype=bool)
        if data.count_observed is None
        else np.asarray(data.count_observed, dtype=bool)
    )
    summary = summarize_path_posterior_numpy(
        spec,
        phi=np.asarray(phi, dtype=np.float64),
        posterior=evaluated_posterior,
        supported=observed,
    )
    return (
        np.asarray(summary["posterior_effective_multiplicity"], dtype=np.float64),
        np.asarray(summary["amplified_mutant_copy_call"], dtype=np.float64) >= 0.5,
    )


def _revised_multiplicity_metrics(
    *,
    fit: FitResult,
    data: TumorData,
    truth: SimulationTruth,
) -> dict[str, float | int]:
    truth_effective = truth.truth_effective_multiplicity
    truth_mass = truth.truth_mutant_copy_mass
    if truth_effective is None or truth_mass is None:
        return {
            "effective_multiplicity_rmse": float("nan"),
            "raw_effective_multiplicity_rmse": float("nan"),
            "summary_effective_multiplicity_rmse": float("nan"),
            "amplified_mutant_copy_f1": float("nan"),
            "raw_amplified_mutant_copy_f1": float("nan"),
            "summary_amplified_mutant_copy_f1": float("nan"),
            "n_effective_multiplicity_units": 0,
            "n_amplified_mutant_copy_units": 0,
            "n_true_amplified_mutant_copy_units": 0,
        }

    raw_effective, raw_amplified = _path_effective_predictions(
        fit=fit,
        data=data,
        phi=np.asarray(fit.phi, dtype=np.float64),
        posterior=getattr(fit, "path_posterior", None),
    )
    summary_effective, summary_amplified = _path_effective_predictions(
        fit=fit,
        data=data,
        phi=np.asarray(fit.phi_clustered, dtype=np.float64),
        posterior=(
            None
            if data.path_likelihood is not None
            else getattr(fit, "path_posterior", None)
        ),
    )
    observed = (
        np.ones_like(truth_effective, dtype=bool)
        if data.count_observed is None
        else np.asarray(data.count_observed, dtype=bool)
    )
    finite = (
        observed
        & np.isfinite(truth_effective)
        & np.isfinite(raw_effective)
        & np.isfinite(summary_effective)
    )
    if not np.any(finite):
        return {
            "effective_multiplicity_rmse": float("nan"),
            "raw_effective_multiplicity_rmse": float("nan"),
            "summary_effective_multiplicity_rmse": float("nan"),
            "amplified_mutant_copy_f1": float("nan"),
            "raw_amplified_mutant_copy_f1": float("nan"),
            "summary_amplified_mutant_copy_f1": float("nan"),
            "n_effective_multiplicity_units": 0,
            "n_amplified_mutant_copy_units": 0,
            "n_true_amplified_mutant_copy_units": 0,
        }
    raw_rmse = float(
        np.sqrt(np.mean((raw_effective[finite] - truth_effective[finite]) ** 2))
    )
    summary_rmse = float(
        np.sqrt(np.mean((summary_effective[finite] - truth_effective[finite]) ** 2))
    )
    true_amplified = np.asarray(truth_mass) > np.asarray(truth.truth_phi) + 1e-8
    raw_f1 = _positive_binary_f1(
        true_amplified[finite],
        raw_amplified[finite],
    )
    summary_f1 = _positive_binary_f1(
        true_amplified[finite],
        summary_amplified[finite],
    )
    return {
        "effective_multiplicity_rmse": summary_rmse,
        "raw_effective_multiplicity_rmse": raw_rmse,
        "summary_effective_multiplicity_rmse": summary_rmse,
        "amplified_mutant_copy_f1": summary_f1,
        "raw_amplified_mutant_copy_f1": raw_f1,
        "summary_amplified_mutant_copy_f1": summary_f1,
        "n_effective_multiplicity_units": int(np.sum(finite)),
        "n_amplified_mutant_copy_units": int(np.sum(finite)),
        "n_true_amplified_mutant_copy_units": int(np.sum(true_amplified[finite])),
    }


def evaluate_fit_against_simulation(
    fit: FitResult,
    data: TumorData,
    simulation_root: str | Path | None = None,
    simulation_truth: SimulationTruth | None = None,
    bic_refit_phi: np.ndarray | None = None,
    bic_partition_labels: np.ndarray | None = None,
) -> SimulationEvaluation:
    if simulation_truth is None:
        if simulation_root is None:
            raise ValueError(
                "Either simulation_root or simulation_truth must be provided."
            )
        simulation_truth = load_simulation_truth(data, simulation_root)

    truth_clusters = simulation_truth.truth_clusters
    truth_phi = simulation_truth.truth_phi
    truth_multiplicity = simulation_truth.truth_multiplicity

    n_eval_mutations = int(data.num_mutations)
    if n_eval_mutations == 0:
        return SimulationEvaluation(
            ari=float("nan"),
            cp_rmse=float("nan"),
            multiplicity_f1=float("nan"),
            estimated_clonal_fraction=float("nan"),
            true_clonal_fraction=float("nan"),
            clonal_fraction_error=float("nan"),
            true_clusters=int(np.unique(truth_clusters).shape[0]),
            estimated_clusters=int(fit.n_clusters),
            n_eval_mutations=0,
            n_filtered_mutations=int(data.num_mutations),
            raw_cp_rmse=float("nan"),
            summary_cp_rmse=float("nan"),
            multiplicity_asymmetric_f1=float("nan"),
            multiplicity_estimable_f1=float("nan"),
            effective_multiplicity_rmse=float("nan"),
            raw_effective_multiplicity_rmse=float("nan"),
            summary_effective_multiplicity_rmse=float("nan"),
            amplified_mutant_copy_f1=float("nan"),
            raw_amplified_mutant_copy_f1=float("nan"),
            summary_amplified_mutant_copy_f1=float("nan"),
            n_effective_multiplicity_units=0,
            n_amplified_mutant_copy_units=0,
            n_true_amplified_mutant_copy_units=0,
        )

    ari = _adjusted_rand_index(truth_clusters, fit.cluster_labels)
    raw_cp_rmse = float(np.sqrt(np.mean((fit.phi - truth_phi) ** 2)))
    summary_cp_rmse = float(np.sqrt(np.mean((fit.phi_clustered - truth_phi) ** 2)))
    cp_rmse = summary_cp_rmse
    estimated_clonal_fraction = _cluster_level_clonal_fraction(
        fit.phi_clustered,
        fit.cluster_labels,
    )
    true_clonal_fraction = _cluster_level_clonal_fraction(
        truth_phi,
        truth_clusters,
    )
    clonal_fraction_error = float(estimated_clonal_fraction - true_clonal_fraction)
    if truth_multiplicity is None:
        multiplicity_f1 = float("nan")
        multiplicity_asymmetric_f1 = float("nan")
        multiplicity_estimable_f1 = float("nan")
    else:
        # The requested primary metric is defined exactly on asymmetric copy
        # states. Symmetric states have no identifiable major-versus-minor
        # binary label and therefore do not enter this score.
        asymmetric_mask = np.asarray(data.major_cn) != np.asarray(data.minor_cn)
        multiplicity_asymmetric_f1 = _multiplicity_f1_for_mask(
            truth_multiplicity=truth_multiplicity,
            predicted_multiplicity=fit.multiplicity_call,
            major_cn=data.major_cn,
            mask=asymmetric_mask,
        )
        multiplicity_estimable_f1 = _multiplicity_f1_for_mask(
            truth_multiplicity=truth_multiplicity,
            predicted_multiplicity=fit.multiplicity_call,
            major_cn=data.major_cn,
            mask=data.multiplicity_estimation_mask,
        )
        multiplicity_f1 = multiplicity_asymmetric_f1

    revised_metrics = _revised_multiplicity_metrics(
        fit=fit,
        data=data,
        truth=simulation_truth,
    )

    bic_refit_ari: float | None = None
    bic_refit_cp_rmse: float | None = None
    refit_phi = (
        bic_refit_phi
        if bic_refit_phi is not None
        else getattr(fit, "bic_refit_phi", None)
    )
    refit_labels = (
        bic_partition_labels
        if bic_partition_labels is not None
        else getattr(fit, "bic_partition_labels", None)
    )
    if (
        refit_phi is not None
        and refit_labels is not None
        and refit_phi.shape == fit.phi_clustered.shape
        and refit_labels.shape == fit.cluster_labels.shape
    ):
        bic_refit_ari = _adjusted_rand_index(truth_clusters, refit_labels)
        bic_refit_cp_rmse = float(np.sqrt(np.mean((refit_phi - truth_phi) ** 2)))

    return SimulationEvaluation(
        ari=ari,
        cp_rmse=cp_rmse,
        multiplicity_f1=multiplicity_f1,
        estimated_clonal_fraction=estimated_clonal_fraction,
        true_clonal_fraction=true_clonal_fraction,
        clonal_fraction_error=clonal_fraction_error,
        true_clusters=int(np.unique(truth_clusters).shape[0]),
        estimated_clusters=int(fit.n_clusters),
        n_eval_mutations=n_eval_mutations,
        n_filtered_mutations=0,
        raw_cp_rmse=raw_cp_rmse,
        summary_cp_rmse=summary_cp_rmse,
        bic_refit_ari=bic_refit_ari,
        bic_refit_cp_rmse=bic_refit_cp_rmse,
        multiplicity_asymmetric_f1=multiplicity_asymmetric_f1,
        multiplicity_estimable_f1=multiplicity_estimable_f1,
        effective_multiplicity_rmse=float(
            revised_metrics["effective_multiplicity_rmse"]
        ),
        raw_effective_multiplicity_rmse=float(
            revised_metrics["raw_effective_multiplicity_rmse"]
        ),
        summary_effective_multiplicity_rmse=float(
            revised_metrics["summary_effective_multiplicity_rmse"]
        ),
        amplified_mutant_copy_f1=float(revised_metrics["amplified_mutant_copy_f1"]),
        raw_amplified_mutant_copy_f1=float(
            revised_metrics["raw_amplified_mutant_copy_f1"]
        ),
        summary_amplified_mutant_copy_f1=float(
            revised_metrics["summary_amplified_mutant_copy_f1"]
        ),
        n_effective_multiplicity_units=int(
            revised_metrics["n_effective_multiplicity_units"]
        ),
        n_amplified_mutant_copy_units=int(
            revised_metrics["n_amplified_mutant_copy_units"]
        ),
        n_true_amplified_mutant_copy_units=int(
            revised_metrics["n_true_amplified_mutant_copy_units"]
        ),
    )
