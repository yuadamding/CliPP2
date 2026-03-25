from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitResult
from ..io.data import PatientData
from ..metrics.evaluation import SimulationEvaluation


def mutation_output_table(data: PatientData, fit: FitResult) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    table = pd.DataFrame(
        {
            "mutation_id": data.mutation_ids,
            "cluster_label": fit.cluster_labels + 1,
            "cluster_size": cluster_sizes[fit.cluster_labels],
        }
    )
    for column, sample_id in enumerate(data.sample_ids):
        table[f"phi_{sample_id}"] = fit.phi_clustered[:, column]
    return table


def cluster_output_table(data: PatientData, fit: FitResult) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    table = pd.DataFrame(
        {
            "cluster_label": np.arange(1, fit.n_clusters + 1, dtype=int),
            "cluster_size": cluster_sizes,
        }
    )
    for column, sample_id in enumerate(data.sample_ids):
        table[f"phi_{sample_id}"] = fit.cluster_centers[:, column]
    return table


def cell_output_table(data: PatientData, fit: FitResult) -> pd.DataFrame:
    mutation_ids = np.repeat(np.asarray(data.mutation_ids, dtype=object), data.num_samples)
    sample_ids = np.tile(np.asarray(data.sample_ids, dtype=object), data.num_mutations)
    cluster_labels = np.repeat(fit.cluster_labels + 1, data.num_samples)
    return pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "sample_id": sample_ids,
            "cluster_label": cluster_labels,
            "phi": fit.phi_clustered.reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
            "major_probability": fit.major_probability.reshape(-1),
            "major_call": fit.major_call.reshape(-1).astype(int),
            "multiplicity_call": fit.multiplicity_call.reshape(-1),
        }
    )


def evaluation_to_frame(evaluation: SimulationEvaluation) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ARI": evaluation.ari,
                "cp_rmse": evaluation.cp_rmse,
                "multiplicity_accuracy": evaluation.multiplicity_accuracy,
                "true_clusters": evaluation.true_clusters,
                "estimated_clusters": evaluation.estimated_clusters,
                "n_eval_mutations": evaluation.n_eval_mutations,
                "n_filtered_mutations": evaluation.n_filtered_mutations,
            }
        ]
    )


def write_fit_outputs(
    outdir: Path,
    data: PatientData,
    fit: FitResult,
    search_df: pd.DataFrame,
    evaluation: SimulationEvaluation | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mutation_output_table(data, fit).to_csv(
        outdir / f"{data.patient_id}_mutation_clusters.tsv",
        sep="\t",
        index=False,
    )
    cluster_output_table(data, fit).to_csv(
        outdir / f"{data.patient_id}_cluster_centers.tsv",
        sep="\t",
        index=False,
    )
    cell_output_table(data, fit).to_csv(
        outdir / f"{data.patient_id}_cell_multiplicity.tsv",
        sep="\t",
        index=False,
    )
    search_df.to_csv(
        outdir / f"{data.patient_id}_lambda_search.tsv",
        sep="\t",
        index=False,
    )
    if evaluation is not None:
        evaluation_to_frame(evaluation).to_csv(
            outdir / f"{data.patient_id}_simulation_eval.tsv",
            sep="\t",
            index=False,
        )


__all__ = [
    "cell_output_table",
    "cluster_output_table",
    "evaluation_to_frame",
    "mutation_output_table",
    "write_fit_outputs",
]
