from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitResult
from ..io.data import TumorData


@dataclass
class SimulationEvaluation:
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


@dataclass(frozen=True)
class SimulationTruth:
    truth_clusters: np.ndarray
    truth_phi: np.ndarray
    truth_multiplicity: np.ndarray | None


def _comb2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) * 0.5


def _adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    labels_true = np.asarray(labels_true, dtype=np.int64).reshape(-1)
    labels_pred = np.asarray(labels_pred, dtype=np.int64).reshape(-1)
    if labels_true.shape != labels_pred.shape:
        raise ValueError("labels_true and labels_pred must have the same shape.")

    n_samples = int(labels_true.size)
    if n_samples < 2:
        return 1.0

    _, true_inverse = np.unique(labels_true, return_inverse=True)
    _, pred_inverse = np.unique(labels_pred, return_inverse=True)
    n_true = int(true_inverse.max()) + 1 if true_inverse.size else 0
    n_pred = int(pred_inverse.max()) + 1 if pred_inverse.size else 0

    pair_codes = true_inverse.astype(np.int64, copy=False) * int(max(n_pred, 1)) + pred_inverse.astype(np.int64, copy=False)
    pair_counts = np.bincount(pair_codes, minlength=int(max(n_true * n_pred, 1)))
    sum_comb_contingency = float(np.sum(_comb2(pair_counts[pair_counts > 0])))
    true_counts = np.bincount(true_inverse, minlength=n_true)
    pred_counts = np.bincount(pred_inverse, minlength=n_pred)
    sum_comb_true = float(np.sum(_comb2(true_counts)))
    sum_comb_pred = float(np.sum(_comb2(pred_counts)))
    total_comb = float(n_samples * (n_samples - 1) * 0.5)
    if total_comb <= 0.0:
        return 1.0

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


def _region_index_from_label(region_id: str) -> int:
    match = re.search(r"(?:sample|region)(\d+)$", region_id)
    if match is None:
        raise ValueError(f"Could not parse region index from '{region_id}'")
    return int(match.group(1))


def _load_single_region_truth(
    tumor_dir: Path,
    data: TumorData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth_clusters = pd.read_csv(tumor_dir / "truth.txt", sep="\t")["cluster_id"].to_numpy(dtype=int)
    truth_cp = pd.read_csv(tumor_dir / "truth_cp.txt", sep="\t")
    if truth_cp.shape[0] != data.num_mutations:
        raise ValueError(
            f"Single-region truth CP count mismatch for tumor {data.tumor_id}: "
            f"{truth_cp.shape[0]} != {data.num_mutations}"
        )

    truth_phi = np.zeros((data.num_mutations, 1), dtype=np.float32)
    truth_phi[:, 0] = truth_cp["ccf"].to_numpy(dtype=np.float32)

    truth_multiplicity = None
    cna_path = tumor_dir / "cna.txt"
    if cna_path.exists():
        cna = pd.read_csv(cna_path, sep="\t")
        if "multiplicity" in cna.columns and cna.shape[0] == data.num_mutations:
            truth_multiplicity = cna["multiplicity"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return truth_clusters, truth_phi, truth_multiplicity


def _cluster_level_clonal_fraction(phi: np.ndarray, labels: np.ndarray) -> float:
    if phi.size == 0 or labels.size == 0:
        return float("nan")

    _, relabeled = np.unique(labels.astype(np.int64), return_inverse=True)
    if relabeled.size == 0:
        return float("nan")

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
        raise FileNotFoundError(f"Simulation directory not found for tumor '{data.tumor_id}': {tumor_dir}")

    if data.num_samples == 1 and (tumor_dir / "truth_cp.txt").exists():
        truth_clusters, truth_phi, truth_multiplicity = _load_single_region_truth(tumor_dir=tumor_dir, data=data)
        return SimulationTruth(
            truth_clusters=truth_clusters,
            truth_phi=truth_phi,
            truth_multiplicity=truth_multiplicity,
        )

    truth_clusters = pd.read_csv(tumor_dir / "truth.txt", sep="\t")["cluster_id"].to_numpy(dtype=int)
    if truth_clusters.shape[0] != data.num_mutations:
        raise ValueError(
            f"Truth cluster count mismatch for tumor {data.tumor_id}: "
            f"{truth_clusters.shape[0]} != {data.num_mutations}"
        )

    truth_phi = np.zeros((data.num_mutations, data.num_samples), dtype=np.float32)
    truth_multiplicity = np.zeros((data.num_mutations, data.num_samples), dtype=np.float32)

    for column, region_id in enumerate(data.region_ids):
        region_index = _region_index_from_label(region_id)
        region_dir = tumor_dir / f"sample{region_index}"
        truth_cp = pd.read_csv(region_dir / "truth_cp.txt", sep="\t")
        cna = pd.read_csv(region_dir / "cna.txt", sep="\t")

        truth_phi[:, column] = truth_cp["ccf"].to_numpy(dtype=np.float32)
        truth_multiplicity[:, column] = cna["multiplicity"].to_numpy(dtype=np.float32)

    return SimulationTruth(
        truth_clusters=truth_clusters,
        truth_phi=truth_phi,
        truth_multiplicity=truth_multiplicity,
    )


def evaluate_fit_against_simulation(
    fit: FitResult,
    data: TumorData,
    simulation_root: str | Path | None = None,
    simulation_truth: SimulationTruth | None = None,
) -> SimulationEvaluation:
    if simulation_truth is None:
        if simulation_root is None:
            raise ValueError("Either simulation_root or simulation_truth must be provided.")
        simulation_truth = load_simulation_truth(data, simulation_root)

    truth_clusters = simulation_truth.truth_clusters
    truth_phi = simulation_truth.truth_phi
    truth_multiplicity = simulation_truth.truth_multiplicity

    eval_mask = np.ones(data.num_mutations, dtype=bool)
    n_eval_mutations = int(eval_mask.sum())
    n_filtered_mutations = 0
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
        )

    ari = _adjusted_rand_index(truth_clusters[eval_mask], fit.cluster_labels[eval_mask])
    cp_rmse = float(np.sqrt(np.mean((fit.phi_clustered[eval_mask] - truth_phi[eval_mask]) ** 2)))
    estimated_clonal_fraction = _cluster_level_clonal_fraction(
        fit.phi_clustered[eval_mask],
        fit.cluster_labels[eval_mask],
    )
    true_clonal_fraction = _cluster_level_clonal_fraction(
        truth_phi[eval_mask],
        truth_clusters[eval_mask],
    )
    clonal_fraction_error = float(estimated_clonal_fraction - true_clonal_fraction)
    if truth_multiplicity is None:
        multiplicity_f1 = float("nan")
    else:
        cna_subject_mask = data.has_cna & ((data.major_cn != 1.0) | (data.minor_cn != 1.0))
        if not cna_subject_mask.any():
            multiplicity_f1 = float("nan")
        else:
            truth_major = np.isclose(truth_multiplicity[cna_subject_mask], data.major_cn[cna_subject_mask])
            pred_major = np.isclose(fit.multiplicity_call[cna_subject_mask], data.major_cn[cna_subject_mask])
            multiplicity_f1 = _macro_binary_f1(
                truth_major.astype(int).reshape(-1),
                pred_major.astype(int).reshape(-1),
            )

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
        n_filtered_mutations=n_filtered_mutations,
    )


def evaluate_ari_against_simulation(
    fit: FitResult,
    data: TumorData,
    simulation_root: str | Path | None = None,
    simulation_truth: SimulationTruth | None = None,
) -> float:
    if simulation_truth is None:
        if simulation_root is None:
            raise ValueError("Either simulation_root or simulation_truth must be provided.")
        simulation_truth = load_simulation_truth(data, simulation_root)

    return _adjusted_rand_index(
        simulation_truth.truth_clusters.astype(int, copy=False),
        fit.cluster_labels.astype(int, copy=False),
    )
