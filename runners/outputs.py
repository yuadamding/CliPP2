from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.fusion.multiplicity import infer_multiplicity_posterior_numpy
from ..core.fusion.path_summary import (
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.fusion.types import RawFit
from ..io.data import TumorData
from ..model_selection.partitions import _partition_signature
from ..model_selection.types import (
    DirectPartition,
    FusionPartition,
    PartitionRefitSummary,
)

SelectedPartition = FusionPartition | DirectPartition


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
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> np.ndarray:
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
    if isinstance(partition, FusionPartition) and not partition.certified:
        raise AssertionError("Refusing to serialize an uncertified raw partition.")
    if (
        isinstance(partition, DirectPartition)
        and not partition.deterministic_generation
    ):
        raise AssertionError("Refusing to serialize a nondeterministic partition.")
    return labels


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
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, data.num_mutations),
            "mutation_id": data.mutation_ids,
            "cluster_label": labels + 1,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        # The selected fixed-partition refit is the authoritative reported CCF.
        # Keep the compact v0.2.1-style public name requested by downstream
        # consumers; raw-fusion diagnostics remain in the audit tables.
        table[f"phi_{region}"] = refit_phi[:, column]
    return table


def cluster_output_table(
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    centers = np.asarray(refit.cluster_centers, dtype=np.float64)
    expected = (int(partition.n_clusters), int(data.num_regions))
    if centers.shape != expected or not np.all(np.isfinite(centers)):
        raise ValueError(f"refit.cluster_centers must have shape {expected}.")
    sizes = np.bincount(labels, minlength=int(partition.n_clusters))
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, partition.n_clusters),
            "cluster_label": np.arange(1, partition.n_clusters + 1, dtype=int),
            "cluster_size": sizes,
        }
    )
    for column, region_id in enumerate(data.region_ids):
        region = str(region_id)
        table[f"phi_{region}"] = centers[:, column]
    return table


def _add_legacy_multiplicity(
    table: pd.DataFrame,
    *,
    data: TumorData,
    phi: np.ndarray,
    major_prior: float,
    eps: float,
) -> None:
    posterior = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    table["multiplicity_estimated"] = posterior.estimation_mask.reshape(
        -1
    ).astype(int)
    table["gamma_major"] = posterior.gamma_major.reshape(-1)
    table["major_call"] = posterior.major_call.reshape(-1).astype(int)
    table["multiplicity_call"] = posterior.multiplicity_call.reshape(-1)


def _add_path_summary(
    table: pd.DataFrame,
    summary: dict[str, np.ndarray],
) -> None:
    map_index = summary["map_index"].reshape(-1)
    table["map_path"] = pd.array(
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
        table[output_name] = summary[source_name].reshape(-1)
    amplified = summary["amplified_mutant_copy_call"].reshape(-1)
    table["amplified_mutant_copy_call"] = pd.array(
        [pd.NA if not np.isfinite(value) else int(value) for value in amplified],
        dtype="Int64",
    )


def mutation_region_output_table(
    data: TumorData,
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
    *,
    major_prior: float = 0.5,
) -> pd.DataFrame:
    labels = _validate_identity(raw_fit, partition, refit)
    refit_phi = _validated_profile(data, refit.phi, name="refit.phi")
    mutation_ids = np.repeat(
        np.asarray(data.mutation_ids, dtype=object), data.num_regions
    )
    region_ids = np.tile(
        np.asarray([str(x) for x in data.region_ids], dtype=object),
        data.num_mutations,
    )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, mutation_ids.shape[0]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": np.repeat(labels + 1, data.num_regions),
            "phi": refit_phi.reshape(-1),
            "major_cn": data.major_cn.reshape(-1),
            "minor_cn": data.minor_cn.reshape(-1),
        }
    )
    eps = float(getattr(raw_fit, "likelihood_eps", 1e-6))
    if getattr(data, "path_likelihood", None) is None:
        _add_legacy_multiplicity(
            table,
            data=data,
            phi=refit_phi,
            major_prior=major_prior,
            eps=eps,
        )
    else:
        refit_summary = _path_summary_for_profile(data, refit_phi, eps=eps)
        if refit_summary is None:
            raise AssertionError("Path likelihood did not produce a refit summary.")
        _add_path_summary(table, refit_summary)
        supported = refit_summary["supported"].reshape(-1)
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
    raw_fit: RawFit,
    partition: SelectedPartition,
    refit: PartitionRefitSummary,
    major_prior: float = 0.5,
) -> None:
    """Purely serialize one already selected, identity-validated model."""

    tables = {
        "mutation_clusters": mutation_output_table(data, raw_fit, partition, refit),
        "cluster_centers": cluster_output_table(data, raw_fit, partition, refit),
        "mutation_region_multiplicity": mutation_region_output_table(
            data,
            raw_fit,
            partition,
            refit,
            major_prior=float(major_prior),
        ),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix, table in tables.items():
        table.to_csv(outdir / f"{data.tumor_id}_{suffix}.tsv", sep="\t", index=False)


__all__ = [
    "cluster_output_table",
    "mutation_output_table",
    "mutation_region_output_table",
    "write_fit_outputs",
]
