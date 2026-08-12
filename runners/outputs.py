from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.fusion.multiplicity import infer_multiplicity_posterior_numpy
from ..core.fusion.path_summary import (
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.model import FitResult
from ..io.data import TumorData
from ..model_selection.partitions import _partition_signature
from ..model_selection.types import (
    FusionPartition,
    PartitionRefitSummary,
    RawClonalBlockCertificate,
)


def _display_region_label(label: str) -> str:
    return str(label)


def _validated_profile(
    data: TumorData,
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64)
    expected = (int(data.num_mutations), int(data.num_regions))
    if profile.shape != expected:
        raise ValueError(f"{name} has shape {profile.shape}; expected {expected}.")
    if not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} must contain only finite values.")
    return profile


def _validate_identity(
    raw_fit: FitResult,
    partition: FusionPartition,
    refit: PartitionRefitSummary,
) -> tuple[np.ndarray, np.ndarray]:
    raw_phi = np.asarray(raw_fit.phi, dtype=np.float64)
    labels = np.asarray(partition.labels, dtype=np.int64)
    if labels.shape != (raw_phi.shape[0],):
        raise AssertionError("Selected partition does not match raw fit mutations.")
    if not np.array_equal(labels, np.asarray(refit.labels, dtype=np.int64)):
        raise AssertionError("Selected partition and fixed refit labels differ.")
    if partition.signature != _partition_signature(
        labels,
        partition.mutation_ids if partition.mutation_ids else None,
    ):
        raise AssertionError("Selected partition signature does not match its labels.")
    if partition.signature != refit.partition_signature:
        raise AssertionError("Selected partition and fixed refit signatures differ.")
    if not partition.certified:
        raise AssertionError("Refusing to serialize an uncertified partition.")
    return raw_phi, labels


def _path_supported_mask(data: TumorData, shape: tuple[int, int]) -> np.ndarray:
    reasons = getattr(data, "path_unsupported_reason", None)
    if reasons is None:
        return np.ones(shape, dtype=bool)
    reason_array = np.asarray(reasons, dtype=object)
    if tuple(reason_array.shape) != tuple(shape):
        raise ValueError("path_unsupported_reason shape does not match observations.")
    supported = np.asarray(pd.isna(reason_array), dtype=bool)
    count_observed = getattr(data, "count_observed", None)
    observed = (
        np.ones(shape, dtype=bool)
        if count_observed is None
        else np.asarray(count_observed, dtype=bool)
    )
    if tuple(observed.shape) != tuple(shape):
        raise ValueError("count_observed shape does not match observations.")
    if np.any((~supported) & observed):
        raise ValueError(
            "Every path_unsupported_reason must correspond to count_observed=False."
        )
    return supported


def _path_summary_for_profile(
    data: TumorData,
    phi: np.ndarray,
    *,
    eps: float,
) -> dict[str, np.ndarray] | None:
    spec = getattr(data, "path_likelihood", None)
    if spec is None:
        return None
    posterior = path_posterior_at_phi_numpy(data, phi, eps=float(eps))
    return summarize_path_posterior_numpy(
        spec,
        phi=np.asarray(phi, dtype=np.float64),
        posterior=np.asarray(posterior, dtype=np.float64),
        supported=_path_supported_mask(data, spec.shape[:2]),
    )


def mutation_output_table(
    data: TumorData,
    raw_fit: FitResult,
    partition: FusionPartition,
    refit: PartitionRefitSummary,
    clonal_block: RawClonalBlockCertificate | None = None,
) -> pd.DataFrame:
    raw_phi, labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    raw_anchor_index = getattr(raw_fit, "raw_clonal_anchor_mutation_index", None)
    frozen_indices = tuple(
        int(index)
        for index in getattr(
            raw_fit, "raw_clonal_anchor_frozen_mutation_indices", ()
        )
    )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "selected_cluster_label": labels + 1,
            "raw_clonal_anchor_mutation": np.arange(data.num_mutations)
            == int(raw_anchor_index if raw_anchor_index is not None else -1),
            "raw_clonal_anchor_seed": np.arange(data.num_mutations)
            == int(raw_anchor_index if raw_anchor_index is not None else -1),
            "raw_clonal_witness_mutation": np.arange(data.num_mutations)
            == int(raw_anchor_index if raw_anchor_index is not None else -1),
            "raw_clonal_constraint_frozen_member": np.isin(
                np.arange(data.num_mutations), frozen_indices
            ),
            "is_raw_clonal_cluster_member": (
                np.zeros(data.num_mutations, dtype=bool)
                if clonal_block is None
                else np.isin(
                    np.arange(data.num_mutations), clonal_block.member_indices
                )
            ),
            "raw_clonal_cluster_signature": np.repeat(
                "none" if clonal_block is None else clonal_block.block_signature,
                data.num_mutations,
            ),
            "raw_clonal_cluster_max_member_residual": np.repeat(
                np.nan
                if clonal_block is None
                else float(clonal_block.maximum_member_residual),
                data.num_mutations,
            ),
            "raw_clonal_cluster_centroid_residual": np.repeat(
                np.nan
                if clonal_block is None
                else float(clonal_block.centroid_residual),
                data.num_mutations,
            ),
            "raw_clonal_anchor_constraint_residual": np.repeat(
                float(
                    getattr(raw_fit, "raw_clonal_anchor_constraint_residual", 0.0)
                ),
                data.num_mutations,
            ),
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = _display_region_label(region_id)
        anchor_target = getattr(raw_fit, "raw_clonal_anchor_target", None)
        table[f"raw_clonal_anchor_target_{region}"] = (
            np.nan
            if anchor_target is None
            else float(np.asarray(anchor_target, dtype=np.float64)[column])
        )
        table[f"raw_phi_{region}"] = raw_phi[:, column]
        table[f"fixed_partition_refit_phi_{region}"] = refit_phi[:, column]
        table[f"raw_to_refit_delta_{region}"] = (
            refit_phi[:, column] - raw_phi[:, column]
        )
    return table


def _cluster_raw_diameters(raw_phi: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    result = np.zeros(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        values = raw_phi[labels == cluster]
        if values.shape[0] <= 1:
            continue
        maximum = 0.0
        for start in range(0, values.shape[0], 512):
            block = values[start : start + 512]
            distances = np.linalg.norm(block[:, None, :] - values[None, :, :], axis=-1)
            if distances.size:
                maximum = max(maximum, float(np.max(distances)))
        result[cluster] = maximum
    return result


def cluster_output_table(
    data: TumorData,
    raw_fit: FitResult,
    partition: FusionPartition,
    refit: PartitionRefitSummary,
    clonal_block: RawClonalBlockCertificate | None = None,
) -> pd.DataFrame:
    raw_phi, labels = _validate_identity(raw_fit, partition, refit)
    centers = np.asarray(refit.cluster_centers, dtype=np.float64)
    expected = (int(partition.n_clusters), int(data.num_regions))
    if centers.shape != expected or not np.all(np.isfinite(centers)):
        raise ValueError(f"refit.cluster_centers must have shape {expected}.")
    sizes = np.bincount(labels, minlength=int(partition.n_clusters))
    diameters = _cluster_raw_diameters(raw_phi, labels)
    is_clonal_cluster = np.arange(partition.n_clusters) == int(
        refit.clonal_cluster if refit.clonal_cluster is not None else -1
    )
    if diameters.size and not np.isclose(
        float(np.max(diameters)),
        float(partition.max_diameter),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError(
            "Serialized raw diameter differs from certified partition."
        )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, partition.n_clusters),
            "cluster_label": np.arange(1, partition.n_clusters + 1, dtype=int),
            "cluster_size": sizes,
            "raw_cluster_diameter": diameters,
            "raw_cluster_diameter_exact": np.repeat(
                bool(partition.diameter_exact), partition.n_clusters
            ),
            "clonal_anchor_cluster": is_clonal_cluster,
            "raw_clonal_anchor_cluster": is_clonal_cluster,
            "is_raw_clonal_anchor_cluster": is_clonal_cluster,
            "is_raw_clonal_cluster": is_clonal_cluster,
            "raw_clonal_cluster_size": np.where(
                is_clonal_cluster,
                0 if clonal_block is None else int(clonal_block.cluster_size),
                0,
            ),
            "raw_clonal_cluster_signature": np.where(
                is_clonal_cluster,
                "none" if clonal_block is None else clonal_block.block_signature,
                "none",
            ),
            "raw_clonal_cluster_centroid_residual": np.where(
                is_clonal_cluster,
                np.nan
                if clonal_block is None
                else float(clonal_block.centroid_residual),
                np.nan,
            ),
            "raw_clonal_cluster_max_member_residual": np.where(
                is_clonal_cluster,
                np.nan
                if clonal_block is None
                else float(clonal_block.maximum_member_residual),
                np.nan,
            ),
            "raw_clonal_witness_mutation": np.where(
                is_clonal_cluster,
                "none"
                if clonal_block is None
                else str(clonal_block.witness_mutation_id),
                "none",
            ),
            "raw_clonal_anchor_source": np.repeat(
                str(getattr(raw_fit, "raw_clonal_anchor_source", "none")),
                partition.n_clusters,
            ),
            "anchor_deviance_increase": np.repeat(
                float(refit.anchor_deviance_increase), partition.n_clusters
            ),
            "second_best_anchor_deviance_increase": np.repeat(
                float(refit.second_best_anchor_deviance_increase),
                partition.n_clusters,
            ),
            "partition_signature": np.repeat(
                str(partition.signature), partition.n_clusters
            ),
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = _display_region_label(region_id)
        means = np.asarray(
            [
                np.mean(raw_phi[labels == cluster, column])
                for cluster in range(partition.n_clusters)
            ]
        )
        minima = np.asarray(
            [
                np.min(raw_phi[labels == cluster, column])
                for cluster in range(partition.n_clusters)
            ]
        )
        maxima = np.asarray(
            [
                np.max(raw_phi[labels == cluster, column])
                for cluster in range(partition.n_clusters)
            ]
        )
        table[f"raw_cluster_mean_phi_{region}"] = means
        table[f"raw_cluster_min_phi_{region}"] = minima
        table[f"raw_cluster_max_phi_{region}"] = maxima
        table[f"fixed_partition_refit_phi_{region}"] = centers[:, column]
        anchor_target = getattr(raw_fit, "raw_clonal_anchor_target", None)
        table[f"raw_clonal_anchor_target_{region}"] = (
            np.nan
            if anchor_target is None
            else float(np.asarray(anchor_target, dtype=np.float64)[column])
        )
        table[f"raw_clonal_cluster_centroid_{region}"] = np.where(
            is_clonal_cluster,
            np.nan if clonal_block is None else float(clonal_block.centroid[column]),
            np.nan,
        )
        table[f"raw_clonal_cluster_common_center_{region}"] = np.where(
            is_clonal_cluster,
            np.nan
            if clonal_block is None
            else float(clonal_block.common_center[column]),
            np.nan,
        )
        table[f"raw_clonal_cluster_target_{region}"] = np.where(
            is_clonal_cluster,
            np.nan if clonal_block is None else float(clonal_block.target[column]),
            np.nan,
        )
        table[f"raw_clonal_cluster_observed_support_{region}"] = np.where(
            is_clonal_cluster,
            0
            if clonal_block is None
            else int(clonal_block.observed_support_per_region[column]),
            0,
        )
    return table


def _add_legacy_multiplicity(
    table: pd.DataFrame,
    *,
    data: TumorData,
    phi: np.ndarray,
    prefix: str,
    major_prior: float,
    eps: float,
) -> None:
    posterior = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    table[f"{prefix}_multiplicity_estimated"] = posterior.estimation_mask.reshape(
        -1
    ).astype(int)
    table[f"{prefix}_gamma_major"] = posterior.gamma_major.reshape(-1)
    table[f"{prefix}_major_call"] = posterior.major_call.reshape(-1).astype(int)
    table[f"{prefix}_multiplicity_call"] = posterior.multiplicity_call.reshape(-1)


def _add_path_summary(
    table: pd.DataFrame,
    summary: dict[str, np.ndarray],
    *,
    prefix: str,
) -> None:
    map_index = summary["map_index"].reshape(-1)
    table[f"{prefix}_map_path"] = pd.array(
        [pd.NA if value < 0 else int(value) + 1 for value in map_index],
        dtype="Int64",
    )
    fields = {
        "pre_switch_path_probability": "single_probability",
        "post_switch_path_probability": "multi_probability",
        "switch_boundary_ambiguity_probability": "boundary_probability",
        "posterior_mutant_copy_mass": "posterior_mutant_copy_mass",
        "posterior_effective_multiplicity": "posterior_effective_multiplicity",
        "map_mutant_copy_mass": "map_mutant_copy_mass",
        "map_effective_multiplicity": "map_effective_multiplicity",
        "amplified_mutant_copy_probability": "amplified_mutant_copy_probability",
        "path_entropy": "path_entropy",
    }
    for output_name, source_name in fields.items():
        table[f"{prefix}_{output_name}"] = summary[source_name].reshape(-1)
    amplified = summary["amplified_mutant_copy_call"].reshape(-1)
    table[f"{prefix}_amplified_mutant_copy_call"] = pd.array(
        [pd.NA if not np.isfinite(value) else int(value) for value in amplified],
        dtype="Int64",
    )


def mutation_region_output_table(
    data: TumorData,
    raw_fit: FitResult,
    partition: FusionPartition,
    refit: PartitionRefitSummary,
    *,
    major_prior: float = 0.5,
) -> pd.DataFrame:
    raw_phi, labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    mutation_ids = np.repeat(
        np.asarray(data.mutation_ids, dtype=object), data.num_regions
    )
    region_ids = np.tile(
        np.asarray([_display_region_label(x) for x in data.region_ids], dtype=object),
        data.num_mutations,
    )
    anchor_index = getattr(raw_fit, "raw_clonal_anchor_mutation_index", None)
    anchor_target = getattr(raw_fit, "raw_clonal_anchor_target", None)
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, mutation_ids.shape[0]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "selected_cluster_label": np.repeat(labels + 1, data.num_regions),
            "raw_clonal_anchor_seed": np.repeat(
                np.arange(data.num_mutations)
                == int(anchor_index if anchor_index is not None else -1),
                data.num_regions,
            ),
            "raw_clonal_anchor_target": (
                np.full(raw_phi.size, np.nan, dtype=np.float64)
                if anchor_target is None
                else np.tile(
                    np.asarray(anchor_target, dtype=np.float64), data.num_mutations
                )
            ),
            "raw_clonal_anchor_constraint_residual": np.repeat(
                float(
                    getattr(raw_fit, "raw_clonal_anchor_constraint_residual", 0.0)
                ),
                raw_phi.size,
            ),
            "raw_phi": raw_phi.reshape(-1),
            "fixed_partition_refit_phi": refit_phi.reshape(-1),
            "raw_to_refit_delta": (refit_phi - raw_phi).reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
        }
    )
    eps = float(getattr(raw_fit, "likelihood_eps", 1e-6))
    if getattr(data, "path_likelihood", None) is None:
        _add_legacy_multiplicity(
            table,
            data=data,
            phi=raw_phi,
            prefix="raw",
            major_prior=major_prior,
            eps=eps,
        )
        _add_legacy_multiplicity(
            table,
            data=data,
            phi=refit_phi,
            prefix="refit",
            major_prior=major_prior,
            eps=eps,
        )
    else:
        raw_summary = _path_summary_for_profile(data, raw_phi, eps=eps)
        refit_summary = _path_summary_for_profile(data, refit_phi, eps=eps)
        if raw_summary is None or refit_summary is None:
            raise AssertionError("Path likelihood did not produce both summaries.")
        _add_path_summary(table, raw_summary, prefix="raw")
        _add_path_summary(table, refit_summary, prefix="refit")
        supported = raw_summary["supported"].reshape(-1)
        table["path_supported"] = supported.astype(int)
        reasons = getattr(data, "path_unsupported_reason", None)
        if reasons is not None:
            reason = np.asarray(reasons, dtype=object).reshape(-1)
            table["path_unsupported_reason"] = pd.array(
                np.where(supported, None, reason), dtype="string"
            )
    return table


def write_fit_outputs(
    *,
    outdir: Path,
    data: TumorData,
    raw_fit: FitResult,
    partition: FusionPartition,
    refit: PartitionRefitSummary,
    clonal_block: RawClonalBlockCertificate | None = None,
    major_prior: float = 0.5,
) -> None:
    """Purely serialize one already selected, identity-validated model."""

    _validate_identity(raw_fit, partition, refit)
    outdir.mkdir(parents=True, exist_ok=True)
    mutation_output_table(data, raw_fit, partition, refit, clonal_block).to_csv(
        outdir / f"{data.tumor_id}_mutation_clusters.tsv", sep="\t", index=False
    )
    cluster_output_table(data, raw_fit, partition, refit, clonal_block).to_csv(
        outdir / f"{data.tumor_id}_cluster_centers.tsv", sep="\t", index=False
    )
    mutation_region_output_table(
        data,
        raw_fit,
        partition,
        refit,
        major_prior=float(major_prior),
    ).to_csv(
        outdir / f"{data.tumor_id}_mutation_region_multiplicity.tsv",
        sep="\t",
        index=False,
    )


__all__ = [
    "cluster_output_table",
    "mutation_output_table",
    "mutation_region_output_table",
    "write_fit_outputs",
]
