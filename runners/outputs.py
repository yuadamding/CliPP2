from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.fusion.path_summary import (
    DEFAULT_AMPLIFIED_MUTANT_COPY_TOL,
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.model import FitResult
from ..io.data import TumorData
from ..metrics.evaluation import SimulationEvaluation


_AMPLIFIED_MUTANT_COPY_TOL = DEFAULT_AMPLIFIED_MUTANT_COPY_TOL


def _display_region_label(label: str) -> str:
    return str(label)


def _path_supported_mask(
    data: TumorData,
    shape: tuple[int, int],
) -> np.ndarray:
    reasons = getattr(data, "path_unsupported_reason", None)
    if reasons is None:
        return np.ones(shape, dtype=bool)
    reason_array = np.asarray(reasons, dtype=object)
    if tuple(reason_array.shape) != tuple(shape):
        raise ValueError(
            "TumorData.path_unsupported_reason shape does not match "
            f"mutation-region observations: {reason_array.shape} != {shape}."
        )
    supported = np.asarray(pd.isna(reason_array), dtype=bool)
    count_observed = getattr(data, "count_observed", None)
    observed = (
        np.ones(shape, dtype=bool)
        if count_observed is None
        else np.asarray(count_observed, dtype=bool)
    )
    if tuple(observed.shape) != tuple(shape):
        raise ValueError(
            "TumorData.count_observed shape does not match mutation-region "
            f"observations: {observed.shape} != {shape}."
        )
    if np.any((~supported) & observed):
        raise ValueError(
            "Every path_unsupported_reason must correspond to count_observed=False."
        )
    return supported


def _path_summary_arrays(
    data: TumorData,
    fit: FitResult,
    *,
    phi_values: np.ndarray | None = None,
    posterior_values: np.ndarray | None = None,
) -> dict[str, np.ndarray] | None:
    spec = getattr(data, "path_likelihood", None)
    posterior = (
        getattr(fit, "path_posterior", None)
        if posterior_values is None
        else posterior_values
    )
    if spec is None or posterior is None:
        return None
    supported = _path_supported_mask(data, spec.shape[:2])
    phi = (
        np.asarray(fit.phi, dtype=np.float64)
        if phi_values is None
        else np.asarray(phi_values, dtype=np.float64)
    )
    return summarize_path_posterior_numpy(
        spec,
        phi=phi,
        posterior=np.asarray(posterior, dtype=np.float64),
        supported=supported,
    )


def _summary_ccf_path_arrays(
    data: TumorData,
    fit: FitResult,
) -> dict[str, np.ndarray] | None:
    if (
        getattr(data, "path_likelihood", None) is None
        or getattr(fit, "path_posterior", None) is None
    ):
        return None
    summary_phi = np.asarray(fit.phi_clustered, dtype=np.float64)
    summary_posterior = path_posterior_at_phi_numpy(
        data,
        summary_phi,
        eps=float(getattr(fit, "likelihood_eps", 1e-6)),
    )
    return _path_summary_arrays(
        data,
        fit,
        phi_values=summary_phi,
        posterior_values=summary_posterior,
    )


def mutation_output_table(
    data: TumorData,
    fit: FitResult,
    bic_refit_phi: np.ndarray | None = None,
) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    cluster_diameters = np.asarray(fit.cluster_diameters, dtype=np.float64)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": fit.cluster_labels + 1,
            "cluster_size": cluster_sizes[fit.cluster_labels],
            "cluster_diameter": cluster_diameters[fit.cluster_labels],
            "cluster_diameter_exact": np.repeat(
                bool(fit.cluster_diameter_exact), data.num_mutations
            ),
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region_label = _display_region_label(region_id)
        table[f"phi_{region_label}"] = fit.phi[:, column]
        table[f"summary_phi_{region_label}"] = fit.phi_clustered[:, column]
        if bic_refit_phi is not None:
            table[f"bic_refit_phi_{region_label}"] = bic_refit_phi[:, column]
    return table


def cluster_output_table(
    data: TumorData,
    fit: FitResult,
    bic_refit_cluster_centers: np.ndarray | None = None,
) -> pd.DataFrame:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=fit.n_clusters)
    cluster_diameters = np.asarray(fit.cluster_diameters, dtype=np.float64)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, fit.n_clusters),
            "cluster_label": np.arange(1, fit.n_clusters + 1, dtype=int),
            "cluster_size": cluster_sizes,
            "cluster_diameter": cluster_diameters,
            "cluster_diameter_exact": np.repeat(
                bool(fit.cluster_diameter_exact), fit.n_clusters
            ),
        }
    )
    for column, region_id in enumerate(data.region_ids):
        table[f"phi_{_display_region_label(region_id)}"] = fit.cluster_centers[
            :, column
        ]
        if (
            bic_refit_cluster_centers is not None
            and bic_refit_cluster_centers.shape[0] == fit.n_clusters
        ):
            table[f"bic_refit_phi_{_display_region_label(region_id)}"] = (
                bic_refit_cluster_centers[:, column]
            )
    return table


def mutation_region_output_table(
    data: TumorData,
    fit: FitResult,
    bic_refit_phi: np.ndarray | None = None,
) -> pd.DataFrame:
    mutation_ids = np.repeat(
        np.asarray(data.mutation_ids, dtype=object), data.num_regions
    )
    region_ids = np.tile(
        np.asarray(
            [_display_region_label(region_id) for region_id in data.region_ids],
            dtype=object,
        ),
        data.num_mutations,
    )
    cluster_labels = np.repeat(fit.cluster_labels + 1, data.num_regions)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(
                np.asarray(data.tumor_id, dtype=object), mutation_ids.shape[0]
            ),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": cluster_labels,
            "phi": fit.phi.reshape(-1),
            "summary_phi": fit.phi_clustered.reshape(-1),
            "bic_refit_phi": (
                np.full_like(fit.phi, np.nan, dtype=np.float64)
                if bic_refit_phi is None
                else bic_refit_phi
            ).reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
            "multiplicity_estimated": fit.multiplicity_estimated_mask.reshape(
                -1
            ).astype(int),
            "gamma_major": fit.gamma_major.reshape(-1),
            "major_call": fit.major_call.reshape(-1).astype(int),
            "multiplicity_call": fit.multiplicity_call.reshape(-1),
        }
    )
    path_spec = getattr(data, "path_likelihood", None)
    if path_spec is not None:
        # Binary major/minor multiplicity fields have no defined interpretation
        # for categorical occupancy paths.  Retain the legacy columns for schema
        # stability, but never populate them with compatibility placeholders.
        unavailable_integer = pd.array(
            [pd.NA] * len(table),
            dtype="Int64",
        )
        table["multiplicity_estimated"] = unavailable_integer
        table["gamma_major"] = np.nan
        table["major_call"] = unavailable_integer.copy()
        table["multiplicity_call"] = np.nan

    path_summary = _path_summary_arrays(data, fit)
    if path_summary is not None:
        reporting_fingerprint = getattr(data, "path_reporting_fingerprint", None)
        table["likelihood_model_id"] = str(fit.likelihood_model_id)
        if reporting_fingerprint:
            table["path_reporting_fingerprint"] = str(reporting_fingerprint)
        map_index = path_summary["map_index"].reshape(-1)
        table["map_path"] = pd.array(
            [pd.NA if value < 0 else int(value) + 1 for value in map_index],
            dtype="Int64",
        )
        table["pre_switch_path_probability"] = path_summary[
            "single_probability"
        ].reshape(-1)
        table["post_switch_path_probability"] = path_summary[
            "multi_probability"
        ].reshape(-1)
        table["switch_boundary_ambiguity_probability"] = path_summary[
            "boundary_probability"
        ].reshape(-1)
        table["posterior_mutant_copy_mass"] = path_summary[
            "posterior_mutant_copy_mass"
        ].reshape(-1)
        table["posterior_effective_multiplicity"] = path_summary[
            "posterior_effective_multiplicity"
        ].reshape(-1)
        table["map_mutant_copy_mass"] = path_summary["map_mutant_copy_mass"].reshape(-1)
        table["map_effective_multiplicity"] = path_summary[
            "map_effective_multiplicity"
        ].reshape(-1)
        table["amplified_mutant_copy_probability"] = path_summary[
            "amplified_mutant_copy_probability"
        ].reshape(-1)
        amplified_call = path_summary["amplified_mutant_copy_call"].reshape(-1)
        table["amplified_mutant_copy_call"] = pd.array(
            [
                pd.NA if not np.isfinite(value) else int(value)
                for value in amplified_call
            ],
            dtype="Int64",
        )
        table["path_entropy"] = path_summary["path_entropy"].reshape(-1)
        summary_path = _summary_ccf_path_arrays(data, fit)
        if summary_path is not None:
            summary_map_index = summary_path["map_index"].reshape(-1)
            table["summary_map_path"] = pd.array(
                [pd.NA if value < 0 else int(value) + 1 for value in summary_map_index],
                dtype="Int64",
            )
            table["summary_pre_switch_path_probability"] = summary_path[
                "single_probability"
            ].reshape(-1)
            table["summary_post_switch_path_probability"] = summary_path[
                "multi_probability"
            ].reshape(-1)
            table["summary_switch_boundary_ambiguity_probability"] = summary_path[
                "boundary_probability"
            ].reshape(-1)
            table["summary_posterior_mutant_copy_mass"] = summary_path[
                "posterior_mutant_copy_mass"
            ].reshape(-1)
            table["summary_posterior_effective_multiplicity"] = summary_path[
                "posterior_effective_multiplicity"
            ].reshape(-1)
            table["summary_map_mutant_copy_mass"] = summary_path[
                "map_mutant_copy_mass"
            ].reshape(-1)
            table["summary_map_effective_multiplicity"] = summary_path[
                "map_effective_multiplicity"
            ].reshape(-1)
            table["summary_amplified_mutant_copy_probability"] = summary_path[
                "amplified_mutant_copy_probability"
            ].reshape(-1)
            summary_amplified_call = summary_path["amplified_mutant_copy_call"].reshape(
                -1
            )
            table["summary_amplified_mutant_copy_call"] = pd.array(
                [
                    pd.NA if not np.isfinite(value) else int(value)
                    for value in summary_amplified_call
                ],
                dtype="Int64",
            )
            table["summary_path_entropy"] = summary_path["path_entropy"].reshape(-1)
        unsupported = getattr(data, "path_unsupported_reason", None)
        if unsupported is not None:
            reason = np.asarray(unsupported, dtype=object).reshape(-1)
            supported = path_summary["supported"].reshape(-1)
            table["path_supported"] = supported.astype(int)
            table["path_unsupported_reason"] = pd.array(
                np.where(supported, None, reason),
                dtype="string",
            )
    return table


def path_posterior_output_table(data: TumorData, fit: FitResult) -> pd.DataFrame:
    """Return one row per supported, valid mutation-region occupancy path."""

    spec = getattr(data, "path_likelihood", None)
    summary = _path_summary_arrays(data, fit)
    if spec is None or summary is None:
        return pd.DataFrame()
    summary_ccf = _summary_ccf_path_arrays(data, fit)
    if summary_ccf is None:
        raise ValueError("Path data must have a summary-CCF path posterior.")

    supported = _path_supported_mask(data, spec.shape[:2])
    reportable = np.asarray(spec.valid, dtype=bool) & supported[..., None]
    valid_indices = np.argwhere(reportable)
    if valid_indices.size == 0:
        return pd.DataFrame()
    mutation_idx = valid_indices[:, 0]
    region_idx = valid_indices[:, 1]
    path_idx = valid_indices[:, 2]
    posterior = summary["posterior"][mutation_idx, region_idx, path_idx]
    phi = np.asarray(fit.phi, dtype=np.float64)[mutation_idx, region_idx]
    summary_phi = np.asarray(fit.phi_clustered, dtype=np.float64)[
        mutation_idx, region_idx
    ]
    switch = np.asarray(spec.switch_fraction)[mutation_idx, region_idx, path_idx]
    boundary_tol = 1e-8
    path_segment = np.where(
        phi < switch - boundary_tol,
        "first_linear_segment_at_fitted_phi",
        np.where(
            phi > switch + boundary_tol,
            "second_linear_segment_at_fitted_phi",
            "switch_boundary_at_fitted_phi",
        ),
    )
    table = pd.DataFrame(
        {
            "tumor_id": data.tumor_id,
            "mutation_id": np.asarray(data.mutation_ids, dtype=object)[mutation_idx],
            "region_id": np.asarray(
                [_display_region_label(value) for value in data.region_ids],
                dtype=object,
            )[region_idx],
            "likelihood_model_id": str(spec.model_id),
            "likelihood_model_version": str(spec.model_version),
            "candidate_generator_version": str(spec.candidate_generator_version),
            "path_prior_mode": str(spec.prior_mode),
            "path_reporting_fingerprint": str(
                getattr(data, "path_reporting_fingerprint", None) or ""
            ),
            "path_scope": "region_local",
            "path_index": path_idx + 1,
            "path_probability": posterior,
            "map_path": (
                summary["map_index"][mutation_idx, region_idx] == path_idx
            ).astype(int),
            "summary_phi": summary_phi,
            "summary_path_probability": summary_ccf["posterior"][
                mutation_idx, region_idx, path_idx
            ],
            "summary_map_path": (
                summary_ccf["map_index"][mutation_idx, region_idx] == path_idx
            ).astype(int),
            "first_copy": np.asarray(spec.first_copy)[
                mutation_idx, region_idx, path_idx
            ],
            "second_copy": np.asarray(spec.second_copy)[
                mutation_idx, region_idx, path_idx
            ],
            "switch_fraction": switch,
            "path_prior": np.exp(
                np.asarray(spec.log_prior)[mutation_idx, region_idx, path_idx]
            ),
            "mutant_copy_mass": summary["mass"][mutation_idx, region_idx, path_idx],
            "effective_multiplicity": np.divide(
                summary["mass"][mutation_idx, region_idx, path_idx],
                phi,
                out=np.full_like(phi, np.nan, dtype=np.float64),
                where=phi > 0.0,
            ),
            "amplified_mutant_copy": (
                summary["mass"][mutation_idx, region_idx, path_idx]
                > phi + _AMPLIFIED_MUTANT_COPY_TOL
            ).astype(int),
            "path_segment_at_fitted_phi": path_segment,
            "summary_mutant_copy_mass": summary_ccf["mass"][
                mutation_idx, region_idx, path_idx
            ],
            "summary_effective_multiplicity": np.divide(
                summary_ccf["mass"][mutation_idx, region_idx, path_idx],
                summary_phi,
                out=np.full_like(summary_phi, np.nan, dtype=np.float64),
                where=summary_phi > 0.0,
            ),
            "summary_amplified_mutant_copy": (
                summary_ccf["mass"][mutation_idx, region_idx, path_idx]
                > summary_phi + _AMPLIFIED_MUTANT_COPY_TOL
            ).astype(int),
            "summary_path_segment_at_clustered_phi": np.where(
                summary_phi < switch - boundary_tol,
                "first_linear_segment_at_clustered_phi",
                np.where(
                    summary_phi > switch + boundary_tol,
                    "second_linear_segment_at_clustered_phi",
                    "switch_boundary_at_clustered_phi",
                ),
            ),
        }
    )
    annotations = getattr(data, "path_annotations", None)
    if annotations is not None and hasattr(annotations, "columns_for_indices"):
        for name, values in annotations.columns_for_indices(valid_indices).items():
            table[str(name)] = values
    elif annotations is not None:
        aligned_paths = [
            annotations[int(mutation)][int(region)][int(path)]
            for mutation, region, path in valid_indices
        ]

        def alias_values(path, attribute: str) -> list[str]:
            if path is None:
                return []
            return sorted(
                {
                    str(getattr(alias, attribute))
                    for alias in getattr(path, "aliases", ())
                }
            )

        table["biological_duplicate_count"] = [
            0 if path is None else int(path.biological_duplicate_count)
            for path in aligned_paths
        ]
        for column, attribute in (
            ("homolog_mapping_aliases", "mapping_id"),
            ("homolog_aliases", "homolog_id"),
            ("first_state_aliases", "first_state"),
            ("state1_allele_aliases", "state1_allele"),
            ("state2_allele_aliases", "state2_allele"),
            ("copy_relation_aliases", "copy_relation"),
            ("state1_dosage_aliases", "q1"),
            ("state2_dosage_aliases", "q2"),
        ):
            table[column] = [
                ";".join(alias_values(path, attribute)) for path in aligned_paths
            ]
    return table


def evaluation_to_frame(evaluation: SimulationEvaluation) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ARI": evaluation.ari,
                "cp_rmse": evaluation.cp_rmse,
                "raw_cp_rmse": evaluation.raw_cp_rmse,
                "summary_cp_rmse": evaluation.summary_cp_rmse,
                "bic_refit_cp_rmse": evaluation.bic_refit_cp_rmse,
                "multiplicity_f1": evaluation.multiplicity_f1,
                "multiplicity_asymmetric_f1": evaluation.multiplicity_asymmetric_f1,
                "multiplicity_estimable_f1": evaluation.multiplicity_estimable_f1,
                "effective_multiplicity_rmse": evaluation.effective_multiplicity_rmse,
                "raw_effective_multiplicity_rmse": (
                    evaluation.raw_effective_multiplicity_rmse
                ),
                "summary_effective_multiplicity_rmse": (
                    evaluation.summary_effective_multiplicity_rmse
                ),
                "amplified_mutant_copy_f1": evaluation.amplified_mutant_copy_f1,
                "raw_amplified_mutant_copy_f1": (
                    evaluation.raw_amplified_mutant_copy_f1
                ),
                "summary_amplified_mutant_copy_f1": (
                    evaluation.summary_amplified_mutant_copy_f1
                ),
                "n_effective_multiplicity_units": (
                    evaluation.n_effective_multiplicity_units
                ),
                "n_amplified_mutant_copy_units": (
                    evaluation.n_amplified_mutant_copy_units
                ),
                "n_true_amplified_mutant_copy_units": (
                    evaluation.n_true_amplified_mutant_copy_units
                ),
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
    bic_refit_phi: np.ndarray | None = None,
    bic_refit_cluster_centers: np.ndarray | None = None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    mutation_output_table(data, fit, bic_refit_phi=bic_refit_phi).to_csv(
        outdir / f"{data.tumor_id}_mutation_clusters.tsv",
        sep="\t",
        index=False,
    )
    cluster_output_table(
        data, fit, bic_refit_cluster_centers=bic_refit_cluster_centers
    ).to_csv(
        outdir / f"{data.tumor_id}_cluster_centers.tsv",
        sep="\t",
        index=False,
    )
    mutation_region_output_table(data, fit, bic_refit_phi=bic_refit_phi).to_csv(
        outdir / f"{data.tumor_id}_mutation_region_multiplicity.tsv",
        sep="\t",
        index=False,
    )
    path_table = path_posterior_output_table(data, fit)
    if not path_table.empty:
        path_table.to_csv(
            outdir / f"{data.tumor_id}_mutation_region_path_posterior.tsv",
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
    "mutation_region_output_table",
    "cluster_output_table",
    "evaluation_to_frame",
    "mutation_output_table",
    "path_posterior_output_table",
    "write_fit_outputs",
]
