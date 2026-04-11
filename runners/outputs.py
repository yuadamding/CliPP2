from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitResult
from ..io.data import TumorData
from ..metrics.evaluation import SimulationEvaluation


def _display_region_label(label: str) -> str:
    return str(label).replace("sample", "region")


def mutation_output_table(data: TumorData, fit: FitResult) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": fit.cluster_labels + 1,
            "cluster_size": cluster_sizes[fit.cluster_labels],
        }
    )
    for column, region_id in enumerate(data.region_ids):
        table[f"phi_{_display_region_label(region_id)}"] = fit.phi_clustered[:, column]
    return table


def cluster_output_table(data: TumorData, fit: FitResult) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, fit.n_clusters),
            "cluster_label": np.arange(1, fit.n_clusters + 1, dtype=int),
            "cluster_size": cluster_sizes,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        table[f"phi_{_display_region_label(region_id)}"] = fit.cluster_centers[:, column]
    return table


def cell_output_table(data: TumorData, fit: FitResult) -> pd.DataFrame:
    mutation_ids = np.repeat(np.asarray(data.mutation_ids, dtype=object), data.num_regions)
    region_ids = np.tile(
        np.asarray([_display_region_label(region_id) for region_id in data.region_ids], dtype=object),
        data.num_mutations,
    )
    cluster_labels = np.repeat(fit.cluster_labels + 1, data.num_regions)
    return pd.DataFrame(
        {
            "tumor_id": np.repeat(np.asarray(data.tumor_id, dtype=object), mutation_ids.shape[0]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": cluster_labels,
            "phi": fit.phi_clustered.reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
            "multiplicity_estimated": fit.multiplicity_estimated_mask.reshape(-1).astype(int),
            "gamma_major": fit.gamma_major.reshape(-1),
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
                "multiplicity_f1": evaluation.multiplicity_f1,
                "estimated_clonal_fraction": evaluation.estimated_clonal_fraction,
                "true_clonal_fraction": evaluation.true_clonal_fraction,
                "clonal_fraction_error": evaluation.clonal_fraction_error,
                "true_clusters": evaluation.true_clusters,
                "estimated_clusters": evaluation.estimated_clusters,
                "n_eval_mutations": evaluation.n_eval_mutations,
                "n_filtered_mutations": evaluation.n_filtered_mutations,
            }
        ]
    )


def write_fit_outputs(
    outdir: Path,
    data: TumorData,
    fit: FitResult,
    search_df: pd.DataFrame,
    evaluation: SimulationEvaluation | None,
    run_summary: dict[str, float | int | str | bool] | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mutation_output_table(data, fit).to_csv(
        outdir / f"{data.tumor_id}_mutation_clusters.tsv",
        sep="\t",
        index=False,
    )
    cluster_output_table(data, fit).to_csv(
        outdir / f"{data.tumor_id}_cluster_centers.tsv",
        sep="\t",
        index=False,
    )
    cell_output_table(data, fit).to_csv(
        outdir / f"{data.tumor_id}_cell_multiplicity.tsv",
        sep="\t",
        index=False,
    )
    search_df.to_csv(
        outdir / f"{data.tumor_id}_lambda_search.tsv",
        sep="\t",
        index=False,
    )
    if evaluation is not None:
        evaluation_to_frame(evaluation).to_csv(
            outdir / f"{data.tumor_id}_simulation_eval.tsv",
            sep="\t",
            index=False,
        )
    if run_summary is not None:
        pd.DataFrame([run_summary]).to_csv(
            outdir / f"{data.tumor_id}_run_summary.tsv",
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
