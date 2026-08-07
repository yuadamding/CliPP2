from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.fusion.path_summary import (
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.model import FitResult
from ..io.data import TumorData


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
    # Cluster-level attributes (size, diameter, diameter_exact) live in
    # cluster_centers.tsv, joinable on cluster_label.
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": fit.cluster_labels + 1,
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
        }
    )
    path_spec = getattr(data, "path_likelihood", None)
    if path_spec is None:
        # Binary major/minor multiplicity fields only exist for the legacy
        # two-point likelihood; categorical occupancy paths report the path
        # posterior columns below instead.
        table["multiplicity_estimated"] = fit.multiplicity_estimated_mask.reshape(
            -1
        ).astype(int)
        table["gamma_major"] = fit.gamma_major.reshape(-1)
        table["major_call"] = fit.major_call.reshape(-1).astype(int)
        table["multiplicity_call"] = fit.multiplicity_call.reshape(-1)

    path_summary = _path_summary_arrays(data, fit)
    if path_summary is not None:
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


def write_fit_outputs(
    outdir: Path,
    data: TumorData,
    fit: FitResult,
    bic_refit_phi: np.ndarray | None = None,
    bic_refit_cluster_centers: np.ndarray | None = None,
) -> None:
    """Write the three per-tumor result tables.

    Deliberately nothing else: solver, selection, and benchmark diagnostics
    live in the summary dict the CLI prints to stdout.
    """
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


__all__ = [
    "mutation_region_output_table",
    "cluster_output_table",
    "mutation_output_table",
    "write_fit_outputs",
]
