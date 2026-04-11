from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score

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


def evaluate_fit_against_simulation(
    fit: FitResult,
    data: TumorData,
    simulation_root: str | Path,
) -> SimulationEvaluation:
    tumor_dir = Path(simulation_root) / data.tumor_id
    if not tumor_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found for tumor '{data.tumor_id}': {tumor_dir}")

    if data.num_samples == 1 and (tumor_dir / "truth_cp.txt").exists():
        truth_clusters, truth_phi, truth_multiplicity = _load_single_region_truth(tumor_dir=tumor_dir, data=data)
    else:
        truth_clusters = pd.read_csv(tumor_dir / "truth.txt", sep="\t")["cluster_id"].to_numpy(dtype=int)
        if truth_clusters.shape[0] != data.num_mutations:
            raise ValueError(
                f"Truth cluster count mismatch for tumor {data.tumor_id}: "
                f"{truth_clusters.shape[0]} != {data.num_mutations}"
            )

        truth_phi = np.zeros_like(fit.phi_clustered, dtype=np.float32)
        truth_multiplicity = np.zeros_like(fit.multiplicity_call, dtype=np.float32)

        for column, region_id in enumerate(data.region_ids):
            region_index = _region_index_from_label(region_id)
            region_dir = tumor_dir / f"sample{region_index}"
            truth_cp = pd.read_csv(region_dir / "truth_cp.txt", sep="\t")
            cna = pd.read_csv(region_dir / "cna.txt", sep="\t")

            truth_phi[:, column] = truth_cp["ccf"].to_numpy(dtype=np.float32)
            truth_multiplicity[:, column] = cna["multiplicity"].to_numpy(dtype=np.float32)

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

    ari = float(adjusted_rand_score(truth_clusters[eval_mask], fit.cluster_labels[eval_mask]))
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
            multiplicity_f1 = float(
                f1_score(
                    truth_major.astype(int).reshape(-1),
                    pred_major.astype(int).reshape(-1),
                    average="macro",
                    zero_division=0,
                )
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
