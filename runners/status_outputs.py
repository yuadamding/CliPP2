"""Status-rich serialization for primary, secondary, and diagnostic outcomes.

The three historical public tables remain primary-only.  This module writes
the explicit long-form contract that can retain every input region without
turning a conditional or unidentified coordinate into a certified CliPP2
estimate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.fusion.multiplicity import infer_multiplicity_posterior_numpy
from ..core.fusion.path_summary import (
    path_posterior_at_phi_numpy,
    summarize_path_posterior_numpy,
)
from ..core.fusion.types import RawFit
from ..core.objective import compile_observed_model, observed_terms_numpy
from ..io.data import TumorData
from ..model_selection.partitions import _partition_signature
from ..model_selection.types import (
    SearchCandidate,
    SelectablePartitionCandidate,
)
from .cluster_order import ccf_cluster_order


STATUS_SCHEMA_VERSION = 6


def _mask_hash(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _candidate_parts(
    data: TumorData,
    candidate: SelectablePartitionCandidate | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if candidate is None:
        return None, None, None, None
    partition = candidate.partition
    refit = candidate.refit
    labels = np.asarray(partition.labels, dtype=np.int64)
    if labels.shape != (int(data.num_mutations),):
        raise ValueError("Selected partition does not cover every tumor mutation.")
    if partition.signature != _partition_signature(
        labels,
        partition.mutation_ids if partition.mutation_ids else None,
    ):
        raise ValueError("Selected partition signature is not reconstructible.")
    if partition.signature != refit.partition_signature or not np.array_equal(
        labels,
        np.asarray(refit.labels, dtype=np.int64),
    ):
        raise ValueError("Selected partition and fixed-label refit disagree.")
    centers = np.asarray(refit.cluster_centers, dtype=np.float64)
    expected = (int(partition.n_clusters), int(data.num_regions))
    if centers.shape != expected or not np.all(np.isfinite(centers)):
        raise ValueError(f"Selected centers must have finite shape {expected}.")
    lower = np.asarray(refit.coordinate_argmin_lower, dtype=np.float64)
    upper = np.asarray(refit.coordinate_argmin_upper, dtype=np.float64)
    identified = np.asarray(
        refit.coordinate_statistically_identified,
        dtype=bool,
    )
    if (
        lower.shape != expected
        or upper.shape != expected
        or identified.shape != expected
    ):
        raise ValueError("Selected coordinate status arrays do not match centers.")
    return labels, centers, lower, upper


def _coordinate_identified(
    candidate: SelectablePartitionCandidate | None,
) -> np.ndarray | None:
    if candidate is None:
        return None
    return np.asarray(
        candidate.refit.coordinate_statistically_identified,
        dtype=bool,
    )


def _analysis_tier(
    *, primary: bool, candidate: SelectablePartitionCandidate | None
) -> str:
    if primary:
        return "joint_certified"
    if candidate is not None:
        return "conditional_partition_refit"
    return "unsupported_or_unidentified"


def _dominant_kkt_component(raw_fit: RawFit | None) -> str:
    if raw_fit is None:
        return "not_available"
    components = raw_fit.certificate.components
    values = {
        "stationarity": float(components.stationarity),
        "edge_subgradient": float(components.edge_subgradient),
        "dual_ball": float(components.dual_ball),
        "box": float(components.box),
    }
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    return max(finite, key=finite.get) if finite else "not_available"


def cluster_region_estimates_table(
    *,
    data: TumorData,
    candidate: SelectablePartitionCandidate | None,
    primary_estimator_available: bool,
    diagnostic_raw_fit: RawFit | None,
    eps: float,
    major_prior: float,
) -> pd.DataFrame:
    columns = [
        "tumor_id",
        "cluster_label",
        "ccf_ordered_cluster_label",
        "ccf_distance_to_one",
        "ccf_distance_identified_region_count",
        "region_id",
        "cluster_size",
        "phi_best_available",
        "phi_joint_raw",
        "phi_conditional_refit",
        "phi_single_region",
        "phi_interval_lower",
        "phi_interval_upper",
        "phi_interval_component_count",
        "phi_interval_disconnected",
        "statistically_identified",
        "estimate_tier",
        "estimate_source",
        "observed_mutation_count",
        "supported_mutation_count",
        "included_mutation_count",
        "profile_loglik",
        "profile_optimality_gap",
        "partition_signature",
        "objective_hash",
        "observation_mask_hash",
    ]
    labels, centers, lower, upper = _candidate_parts(data, candidate)
    if candidate is None or labels is None or centers is None:
        return pd.DataFrame(columns=columns)
    identified = _coordinate_identified(candidate)
    assert identified is not None and lower is not None and upper is not None
    ordered_labels, distances, identified_counts = ccf_cluster_order(
        centers,
        statistically_identified=identified,
    )
    available = np.asarray(data.count_available, dtype=bool)
    supported = np.asarray(data.likelihood_supported, dtype=bool)
    included = np.asarray(data.objective_inclusion_mask(), dtype=bool)
    model = compile_observed_model(
        data,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    terms = observed_terms_numpy(model, np.asarray(candidate.refit.phi), eps=float(eps))
    sizes = np.bincount(labels, minlength=centers.shape[0])
    source = str(candidate.partition.source)
    partition_signature = str(candidate.partition.signature)
    raw_phi = (
        None
        if diagnostic_raw_fit is None or not primary_estimator_available
        else np.asarray(diagnostic_raw_fit.phi, dtype=np.float64)
    )
    objective_hash = (
        model.fingerprint
        if diagnostic_raw_fit is None or not primary_estimator_available
        else str(diagnostic_raw_fit.provenance.base_fusion_objective_hash)
    )
    mask_hash = _mask_hash(included)
    rows: list[dict[str, object]] = []
    for cluster in range(centers.shape[0]):
        members = np.flatnonzero(labels == cluster)
        for region, region_id in enumerate(data.region_ids):
            is_identified = bool(identified[cluster, region])
            tier = (
                _analysis_tier(
                    primary=bool(primary_estimator_available),
                    candidate=candidate,
                )
                if is_identified
                else "structural_representative_only"
            )
            rows.append(
                {
                    "tumor_id": data.tumor_id,
                    "cluster_label": cluster + 1,
                    "ccf_ordered_cluster_label": int(ordered_labels[cluster]),
                    "ccf_distance_to_one": float(distances[cluster]),
                    "ccf_distance_identified_region_count": int(
                        identified_counts[cluster]
                    ),
                    "region_id": str(region_id),
                    "cluster_size": int(sizes[cluster]),
                    "phi_best_available": float(centers[cluster, region]),
                    "phi_joint_raw": (
                        np.nan
                        if raw_phi is None
                        else float(np.mean(raw_phi[members, region]))
                    ),
                    "phi_conditional_refit": float(centers[cluster, region]),
                    "phi_single_region": np.nan,
                    "phi_interval_lower": float(lower[cluster, region]),
                    "phi_interval_upper": float(upper[cluster, region]),
                    # Exact profile-set components are not yet represented by
                    # PartitionRefitSummary.  Leave these unknown rather than
                    # fabricating a connected one-component confidence set.
                    "phi_interval_component_count": None,
                    "phi_interval_disconnected": None,
                    "statistically_identified": is_identified,
                    "estimate_tier": tier,
                    "estimate_source": source,
                    "observed_mutation_count": int(np.sum(available[members, region])),
                    "supported_mutation_count": int(np.sum(supported[members, region])),
                    "included_mutation_count": int(np.sum(included[members, region])),
                    "profile_loglik": float(-np.sum(terms.loss[members, region])),
                    # Only a global refit gap is currently retained.  Do not
                    # repeat it as if it were a coordinate profile gap.
                    "profile_optimality_gap": None,
                    "partition_signature": partition_signature,
                    "objective_hash": objective_hash,
                    "observation_mask_hash": mask_hash,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def region_status_table(
    *,
    data: TumorData,
    candidate: SelectablePartitionCandidate | None,
    primary_estimator_available: bool,
    diagnostic_raw_fit: RawFit | None,
    failure_reason: str,
) -> pd.DataFrame:
    available = np.asarray(data.count_available, dtype=bool)
    supported = np.asarray(data.likelihood_supported, dtype=bool)
    included = np.asarray(data.objective_inclusion_mask(), dtype=bool)
    identified = _coordinate_identified(candidate)
    tolerance = (
        np.nan
        if diagnostic_raw_fit is None
        else float(diagnostic_raw_fit.certificate.tolerance)
    )
    source = "none" if candidate is None else str(candidate.partition.source)
    rows: list[dict[str, object]] = []
    for region, region_id in enumerate(data.region_ids):
        identified_fraction = (
            0.0
            if identified is None or identified.shape[0] == 0
            else float(np.mean(identified[:, region]))
        )
        region_tier = _analysis_tier(
            primary=bool(primary_estimator_available), candidate=candidate
        )
        if candidate is not None and identified_fraction == 0.0:
            region_tier = "structural_representative_only"
        rows.append(
            {
                "tumor_id": data.tumor_id,
                "region_id": str(region_id),
                "region_order": region,
                "analysis_tier": region_tier,
                "estimate_source": source,
                "primary_estimator_available": bool(primary_estimator_available),
                "joint_objective_certified": bool(primary_estimator_available),
                "regional_conditional_certified": bool(
                    candidate is not None and candidate.refit.global_optimum_certified
                ),
                "count_available_count": int(np.sum(available[:, region])),
                "count_available_fraction": float(np.mean(available[:, region])),
                "likelihood_supported_count": int(np.sum(supported[:, region])),
                "likelihood_supported_fraction": float(np.mean(supported[:, region])),
                "likelihood_included_count": int(np.sum(included[:, region])),
                "likelihood_included_fraction": float(np.mean(included[:, region])),
                "identified_cluster_fraction": identified_fraction,
                # The current schema-2 certificate stores the authoritative
                # global maximum only.  Do not fabricate a regional residual.
                "stationarity_residual": np.nan,
                "kkt_tolerance": tolerance,
                # A global full-graph residual is not a regional ratio. The
                # authoritative global value remains in analysis_status and
                # raw_attempts until per-region KKT attribution is implemented.
                "kkt_ratio": np.nan,
                "dominant_failure_component": (
                    "none"
                    if primary_estimator_available
                    else _dominant_kkt_component(diagnostic_raw_fit)
                ),
                "rescue_attempted": False,
                "rescue_stage": (
                    "conditional_partition_refit"
                    if candidate is not None and not primary_estimator_available
                    else "none"
                ),
                "failure_reason": failure_reason,
            }
        )
    return pd.DataFrame(rows)


def mutation_region_estimates_table(
    *,
    data: TumorData,
    candidate: SelectablePartitionCandidate | None,
    primary_estimator_available: bool,
    eps: float,
    major_prior: float,
) -> pd.DataFrame:
    shape = (int(data.num_mutations), int(data.num_regions))
    available = np.asarray(data.count_available, dtype=bool)
    supported = np.asarray(data.likelihood_supported, dtype=bool)
    included = np.asarray(data.objective_inclusion_mask(), dtype=bool)
    reasons = np.asarray(data.likelihood_exclusion_reason, dtype=object)
    mutation_ids = np.repeat(np.asarray(data.mutation_ids, dtype=object), shape[1])
    region_ids = np.tile(np.asarray(data.region_ids, dtype=object), shape[0])
    alt = np.asarray(data.alt_counts, dtype=np.float64)
    ref = np.asarray(data.total_counts, dtype=np.float64) - alt

    labels, centers, lower, upper = _candidate_parts(data, candidate)
    if candidate is None or labels is None or centers is None:
        cluster = np.full(shape, -1, dtype=np.int64)
        ordered_cluster = np.full(shape, -1, dtype=np.int64)
        phi = np.full(shape, np.nan, dtype=np.float64)
        interval_lower = np.full(shape, float(eps), dtype=np.float64)
        interval_upper = np.asarray(data.phi_upper, dtype=np.float64)
        identified = np.zeros(shape, dtype=bool)
        tier = np.full(shape, "unsupported_or_unidentified", dtype=object)
        source = "none"
        signature = ""
    else:
        coordinate_identified = _coordinate_identified(candidate)
        assert (
            coordinate_identified is not None
            and lower is not None
            and upper is not None
        )
        cluster = np.broadcast_to(labels[:, None], shape)
        ordered_labels, _distances, _counts = ccf_cluster_order(
            centers,
            statistically_identified=coordinate_identified,
        )
        ordered_cluster = np.broadcast_to(ordered_labels[labels][:, None], shape)
        phi = centers[labels]
        interval_lower = lower[labels]
        interval_upper = upper[labels]
        identified = coordinate_identified[labels]
        base_tier = _analysis_tier(
            primary=bool(primary_estimator_available), candidate=candidate
        )
        tier = np.where(identified, base_tier, "structural_representative_only")
        source = str(candidate.partition.source)
        signature = str(candidate.partition.signature)
    reportable = included & identified
    frame = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, shape[0] * shape[1]),
            "mutation_id": mutation_ids,
            "region_id": region_ids,
            "cluster_label": pd.array(
                [
                    pd.NA if value < 0 else int(value) + 1
                    for value in cluster.reshape(-1)
                ],
                dtype="Int64",
            ),
            "ccf_ordered_cluster_label": pd.array(
                [
                    pd.NA if value < 0 else int(value)
                    for value in ordered_cluster.reshape(-1)
                ],
                dtype="Int64",
            ),
            "phi_best_available": phi.reshape(-1),
            "major_cn": np.asarray(data.major_cn).reshape(-1),
            "minor_cn": np.asarray(data.minor_cn).reshape(-1),
            "alt_count": pd.array(
                np.where(available, alt, None).reshape(-1), dtype="Float64"
            ),
            "ref_count": pd.array(
                np.where(available, ref, None).reshape(-1), dtype="Float64"
            ),
            "count_available": available.reshape(-1).astype(int),
            "likelihood_supported": supported.reshape(-1).astype(int),
            "likelihood_included": included.reshape(-1).astype(int),
            "likelihood_exclusion_reason": pd.array(
                np.where(included, None, reasons).reshape(-1), dtype="string"
            ),
            "phi_estimate_tier": tier.reshape(-1),
            "phi_estimate_source": source,
            "phi_statistically_identified": identified.reshape(-1).astype(int),
            "phi_interval_lower": interval_lower.reshape(-1),
            "phi_interval_upper": interval_upper.reshape(-1),
            "partition_signature": signature,
        }
    )

    if data.path_likelihood is None:
        if candidate is None:
            unavailable_integer = pd.array(
                np.full(shape[0] * shape[1], None, dtype=object),
                dtype="Int64",
            )
            frame["multiplicity_estimated"] = unavailable_integer.copy()
            frame["gamma_major"] = np.nan
            frame["major_call"] = unavailable_integer.copy()
            frame["multiplicity_call"] = np.nan
            return frame
        posterior = infer_multiplicity_posterior_numpy(
            data,
            phi,
            major_prior=float(major_prior),
            eps=float(eps),
        )
        eligible = reportable.reshape(-1)
        frame["multiplicity_estimated"] = pd.array(
            np.where(eligible, posterior.estimation_mask.reshape(-1).astype(int), None),
            dtype="Int64",
        )
        frame["gamma_major"] = np.where(
            eligible, posterior.gamma_major.reshape(-1), np.nan
        )
        frame["major_call"] = pd.array(
            np.where(eligible, posterior.major_call.reshape(-1).astype(int), None),
            dtype="Int64",
        )
        frame["multiplicity_call"] = np.where(
            eligible, posterior.multiplicity_call.reshape(-1), np.nan
        )
        return frame

    spec = data.path_likelihood
    frame["path_supported"] = supported.reshape(-1).astype(int)
    frame["path_unsupported_reason"] = pd.array(
        np.asarray(data.path_unsupported_reason, dtype=object).reshape(-1),
        dtype="string",
    )
    path_fields = {
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
    if candidate is None:
        unavailable_integer = pd.array(
            np.full(shape[0] * shape[1], None, dtype=object),
            dtype="Int64",
        )
        frame["map_path"] = unavailable_integer.copy()
        for output_name in path_fields:
            frame[output_name] = np.nan
        frame["amplified_mutant_copy_call"] = unavailable_integer.copy()
        return frame

    posterior = path_posterior_at_phi_numpy(data, phi, eps=float(eps))
    summary = summarize_path_posterior_numpy(
        spec,
        phi=phi,
        posterior=posterior,
        supported=reportable,
    )
    map_index = summary["map_index"].reshape(-1)
    frame["map_path"] = pd.array(
        [pd.NA if value < 0 else int(value) + 1 for value in map_index],
        dtype="Int64",
    )
    for output_name, source_name in path_fields.items():
        frame[output_name] = summary[source_name].reshape(-1)
    amplified = summary["amplified_mutant_copy_call"].reshape(-1)
    frame["amplified_mutant_copy_call"] = pd.array(
        [pd.NA if not np.isfinite(value) else int(value) for value in amplified],
        dtype="Int64",
    )
    return frame


_RAW_ATTEMPT_COLUMNS = [
    "candidate_id",
    "candidate_selected",
    "search_round",
    "search_phase",
    "source",
    "start_value",
    "breakpoint_escape_changed_count",
    "mathematically_certified",
    "outer_max_iter",
    "inner_max_iter",
    "certificate_max_iter",
    "lambda_value",
    "objective",
    "stationarity",
    "edge_subgradient",
    "dual_ball",
    "box",
    "kkt_residual",
    "kkt_tolerance",
    "certificate_status",
    "certificate_admissible",
    "working_dtype",
    "audit_dtype",
    "precision_polished",
    "precision_polish_delta",
    "promotion_status",
    "mm_consistency_violations",
    "stage_outer_iterations",
    "stage_outer_max_iter",
    "stage_inner_iterations",
    "stage_inner_max_iter",
    "stage_inner_solve_calls",
    "stop_reason",
    "progress_residual_method",
    "solve_tolerance",
    "legacy_stop_kkt_residual",
    "componentwise_stop_kkt_residual",
    "accepted_full_steps",
    "accepted_damped_steps",
    "rejected_outer_steps",
    "work_inner_iterations",
    "work_inner_stationarity_checks",
    "work_inner_full_kkt_audits",
    "work_outer_kkt_audits",
    "work_certificate_iterations",
    "work_certificate_full_graph_passes",
    "work_partition_refit_coordinates",
    "work_partition_refit_objective_evaluations",
    "work_edge_pass_equivalents",
    "work_full_certificate_audit_passes",
    "device",
    "dtype",
    "objective_spec_hash",
    "original_graph_hash",
    "certificate_problem_hash",
    "fallback_reason",
]


def raw_attempts_table(search: tuple[SearchCandidate, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in search:
        for attempt in item.trace.raw_attempts:
            rows.append(
                {
                    "candidate_id": int(item.candidate_id),
                    "candidate_selected": bool(item.selected),
                    "search_round": int(item.trace.search_round),
                    "search_phase": str(item.trace.search_phase),
                    **{
                        column: getattr(attempt, column)
                        for column in _RAW_ATTEMPT_COLUMNS
                        if column
                        not in {
                            "candidate_id",
                            "candidate_selected",
                            "search_round",
                            "search_phase",
                        }
                    },
                }
            )
    return pd.DataFrame(rows, columns=_RAW_ATTEMPT_COLUMNS)


def secondary_cluster_centers_table(
    *,
    data: TumorData,
    candidate: SelectablePartitionCandidate,
) -> pd.DataFrame:
    labels, centers, _lower, _upper = _candidate_parts(data, candidate)
    assert labels is not None and centers is not None
    sizes = np.bincount(labels, minlength=centers.shape[0])
    identified = _coordinate_identified(candidate)
    assert identified is not None
    ordered_labels, distances, identified_counts = ccf_cluster_order(
        centers,
        statistically_identified=identified,
    )
    table = pd.DataFrame(
        {
            "tumor_id": np.repeat(data.tumor_id, centers.shape[0]),
            "cluster_label": np.arange(1, centers.shape[0] + 1),
            "ccf_ordered_cluster_label": ordered_labels,
            "ccf_distance_to_one": distances,
            "ccf_distance_identified_region_count": identified_counts,
            "cluster_size": sizes,
            "estimate_tier": "conditional_partition_refit",
            "primary_estimator_available": False,
        }
    )
    for region, region_id in enumerate(data.region_ids):
        table[f"phi_{region_id}"] = centers[:, region]
    return table


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    if value is pd.NA:
        return None
    return value


def write_status_outputs(
    *,
    outdir: Path,
    data: TumorData,
    summary: dict[str, object],
    region_status: pd.DataFrame,
    raw_attempts: pd.DataFrame,
    cluster_region_estimates: pd.DataFrame,
    mutation_region_estimates: pd.DataFrame,
) -> None:
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    tumor_id = str(data.tumor_id)
    payload = dict(summary)
    payload["status_schema_version"] = STATUS_SCHEMA_VERSION
    payload["regions"] = region_status.replace({np.nan: None}).to_dict("records")
    with (destination / f"{tumor_id}_analysis_status.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    for suffix, table in {
        "region_status": region_status,
        "raw_attempts": raw_attempts,
        "cluster_region_estimates": cluster_region_estimates,
        "mutation_region_estimates": mutation_region_estimates,
    }.items():
        table.to_csv(destination / f"{tumor_id}_{suffix}.tsv", sep="\t", index=False)


__all__ = [
    "STATUS_SCHEMA_VERSION",
    "cluster_region_estimates_table",
    "mutation_region_estimates_table",
    "raw_attempts_table",
    "region_status_table",
    "secondary_cluster_centers_table",
    "write_status_outputs",
]
