from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..core.model import FitResult
from ..io.data import PatientData


@dataclass
class SimulationEvaluation:
    ari: float
    cp_rmse: float
    multiplicity_accuracy: float
    true_clusters: int
    estimated_clusters: int
    n_eval_mutations: int
    n_filtered_mutations: int


def _sample_index_from_label(sample_id: str) -> int:
    match = re.search(r"sample(\d+)$", sample_id)
    if match is None:
        raise ValueError(f"Could not parse sample index from '{sample_id}'")
    return int(match.group(1))


def _load_single_region_truth(
    patient_dir: Path,
    data: PatientData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    truth_clusters = pd.read_csv(patient_dir / "truth.txt", sep="\t")["cluster_id"].to_numpy(dtype=int)
    truth_cp = pd.read_csv(patient_dir / "truth_cp.txt", sep="\t")
    if truth_cp.shape[0] != data.num_mutations:
        raise ValueError(
            f"Single-region truth CP count mismatch for {data.patient_id}: "
            f"{truth_cp.shape[0]} != {data.num_mutations}"
        )

    truth_phi = np.zeros((data.num_mutations, 1), dtype=np.float32)
    truth_phi[:, 0] = truth_cp["ccf"].to_numpy(dtype=np.float32)

    truth_multiplicity = None
    cna_path = patient_dir / "cna.txt"
    if cna_path.exists():
        cna = pd.read_csv(cna_path, sep="\t")
        if "multiplicity" in cna.columns and cna.shape[0] == data.num_mutations:
            truth_multiplicity = cna["multiplicity"].to_numpy(dtype=np.float32).reshape(-1, 1)

    return truth_clusters, truth_phi, truth_multiplicity


def evaluate_fit_against_simulation(
    fit: FitResult,
    data: PatientData,
    simulation_root: str | Path,
) -> SimulationEvaluation:
    patient_dir = Path(simulation_root) / data.patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found for patient '{data.patient_id}': {patient_dir}")

    if data.num_samples == 1 and (patient_dir / "truth_cp.txt").exists():
        truth_clusters, truth_phi, truth_multiplicity = _load_single_region_truth(patient_dir=patient_dir, data=data)
    else:
        truth_clusters = pd.read_csv(patient_dir / "truth.txt", sep="\t")["cluster_id"].to_numpy(dtype=int)
        if truth_clusters.shape[0] != data.num_mutations:
            raise ValueError(
                f"Truth cluster count mismatch for {data.patient_id}: "
                f"{truth_clusters.shape[0]} != {data.num_mutations}"
            )

        truth_phi = np.zeros_like(fit.phi_clustered, dtype=np.float32)
        truth_multiplicity = np.zeros_like(fit.multiplicity_call, dtype=np.float32)

        for column, sample_id in enumerate(data.sample_ids):
            sample_index = _sample_index_from_label(sample_id)
            sample_dir = patient_dir / f"sample{sample_index}"
            truth_cp = pd.read_csv(sample_dir / "truth_cp.txt", sep="\t")
            cna = pd.read_csv(sample_dir / "cna.txt", sep="\t")

            truth_phi[:, column] = truth_cp["ccf"].to_numpy(dtype=np.float32)
            truth_multiplicity[:, column] = cna["multiplicity"].to_numpy(dtype=np.float32)

    eval_mask = np.ones(data.num_mutations, dtype=bool)
    n_eval_mutations = int(eval_mask.sum())
    n_filtered_mutations = 0
    if n_eval_mutations == 0:
        return SimulationEvaluation(
            ari=float("nan"),
            cp_rmse=float("nan"),
            multiplicity_accuracy=float("nan"),
            true_clusters=int(np.unique(truth_clusters).shape[0]),
            estimated_clusters=int(fit.n_clusters),
            n_eval_mutations=0,
            n_filtered_mutations=int(data.num_mutations),
        )

    ari = float(adjusted_rand_score(truth_clusters[eval_mask], fit.cluster_labels[eval_mask]))
    cp_rmse = float(np.sqrt(np.mean((fit.phi_clustered[eval_mask] - truth_phi[eval_mask]) ** 2)))
    if truth_multiplicity is None:
        multiplicity_accuracy = float("nan")
    else:
        cna_mask = np.all(data.has_cna, axis=1) & np.any(
            (data.major_cn != 1.0) | (data.minor_cn != 1.0), axis=1
        )
        if not cna_mask.any():
            multiplicity_accuracy = float("nan")
        else:
            multiplicity_accuracy = float(
                np.mean(np.isclose(fit.multiplicity_call[cna_mask], truth_multiplicity[cna_mask]))
            )

    return SimulationEvaluation(
        ari=ari,
        cp_rmse=cp_rmse,
        multiplicity_accuracy=multiplicity_accuracy,
        true_clusters=int(np.unique(truth_clusters).shape[0]),
        estimated_clusters=int(fit.n_clusters),
        n_eval_mutations=n_eval_mutations,
        n_filtered_mutations=n_filtered_mutations,
    )
