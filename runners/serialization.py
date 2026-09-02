"""Versioned serialization boundary for one CliPP2 tumor analysis.

Primary, conditional-secondary, and diagnostic-only outcomes share one status
schema. Only a primary outcome with an independently certified raw reference
may use the three historical compatibility filenames.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
from pathlib import Path

import numpy as np

from .._version import __version__ as _SOFTWARE_VERSION
from ..config import FitConfig
from ..core.fusion.types import RawFit, WorkCounters
from ..io.data import TumorData
from ..model_selection.types import (
    DirectPartition,
    DirectPartitionCandidate,
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    SecondaryFallbackResult,
    SelectablePartitionCandidate,
    SelectionScore,
    TumorSelectionOutcome,
)
from .cluster_order import CCF_CLUSTER_ORDERING_METHOD
from .outputs import write_fit_outputs
from .status_outputs import (
    cluster_region_estimates_table,
    mutation_region_estimates_table,
    raw_attempts_table,
    region_status_table,
    secondary_cluster_centers_table,
    write_status_outputs,
)


SUMMARY_SCHEMA_VERSION = 8
_PRIMARY_SUFFIXES = (
    "mutation_clusters.tsv",
    "cluster_centers.tsv",
    "mutation_region_multiplicity.tsv",
)
_SECONDARY_SUFFIXES = (
    "secondary_cluster_centers.tsv",
    "secondary_mutation_region_estimates.tsv",
)


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisSerialization:
    """One normalized, output-ready view of any typed selection outcome."""

    data: TumorData
    input_file: Path
    fit_config: FitConfig
    selection_result: TumorSelectionOutcome

    @property
    def primary_estimator_available(self) -> bool:
        return bool(
            getattr(
                self.selection_result,
                "primary_estimator_available",
                hasattr(self.selection_result, "selected_model"),
            )
        )

    @property
    def selected_candidate(self) -> SelectablePartitionCandidate | None:
        result = self.selection_result
        if hasattr(result, "selected_model"):
            return result.selected_model.partition_candidate
        if isinstance(result, SecondaryFallbackResult):
            return result.selected_partition
        return None

    @property
    def raw_reference(self) -> RawFusionCandidate | None:
        result = self.selection_result
        if hasattr(result, "selected_model"):
            return result.selected_model.raw_reference
        return None

    @property
    def partition_parent_raw(self) -> RawFusionCandidate | None:
        result = self.selection_result
        if hasattr(result, "selected_model"):
            return result.selected_model.partition_parent_raw
        return None

    @property
    def raw_fit(self) -> RawFit | None:
        reference = self.raw_reference
        return None if reference is None else reference.raw_fit

    @property
    def diagnostic_raw_fit(self) -> RawFit | None:
        raw = self.raw_fit
        if raw is not None:
            return raw
        return getattr(self.selection_result, "best_raw_attempt", None)

    @property
    def partition(self) -> FusionPartition | DirectPartition | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.partition

    @property
    def refit(self) -> PartitionRefitSummary | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.refit

    @property
    def score(self) -> SelectionScore | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.score

    @property
    def failure_reason(self) -> str:
        return str(getattr(self.selection_result, "reason", ""))


def _raw_summary(raw_fit: RawFit | None) -> dict[str, object]:
    if raw_fit is None:
        return {
            "diagnostic_best_raw_lambda": None,
            "diagnostic_best_raw_objective": None,
            "diagnostic_best_raw_kkt_residual": None,
            "diagnostic_best_raw_kkt_tolerance": None,
            "diagnostic_best_raw_certificate_status": None,
        }
    certificate = raw_fit.certificate
    components = getattr(certificate, "components", None)
    residual = getattr(components, "residual", None)
    status = getattr(certificate, "status", None)
    return {
        "diagnostic_best_raw_lambda": float(raw_fit.provenance.lambda_value),
        "diagnostic_best_raw_objective": float(raw_fit.objective.total),
        "diagnostic_best_raw_kkt_residual": (
            None if residual is None else float(residual)
        ),
        "diagnostic_best_raw_kkt_tolerance": float(certificate.tolerance),
        "diagnostic_best_raw_certificate_status": (
            None if status is None else str(status)
        ),
    }


def _search_work_summary(result: TumorSelectionOutcome) -> dict[str, int]:
    """Aggregate each executed raw attempt and uncached partition refit once."""

    authoritative = getattr(result, "search_work", None)
    if isinstance(authoritative, WorkCounters):
        return {
            f"search_work_{item.name}": int(getattr(authoritative, item.name))
            for item in fields(authoritative)
        }

    total = WorkCounters()
    for item in result.search:
        for attempt in item.trace.raw_attempts:
            total = total + WorkCounters(
                inner_iterations=int(attempt.work_inner_iterations),
                inner_stationarity_checks=int(
                    attempt.work_inner_stationarity_checks
                ),
                inner_full_kkt_audits=int(attempt.work_inner_full_kkt_audits),
                outer_kkt_audits=int(attempt.work_outer_kkt_audits),
                certificate_iterations=int(attempt.work_certificate_iterations),
                certificate_full_graph_passes=int(
                    attempt.work_certificate_full_graph_passes
                ),
                partition_refit_coordinates=int(
                    attempt.work_partition_refit_coordinates
                ),
                partition_refit_objective_evaluations=int(
                    attempt.work_partition_refit_objective_evaluations
                ),
                edge_pass_equivalents=int(attempt.work_edge_pass_equivalents),
                edge_region_visits=int(attempt.work_edge_region_visits),
                full_certificate_audit_passes=int(
                    attempt.work_full_certificate_audit_passes
                ),
            )
        if isinstance(item.candidate, DirectPartitionCandidate):
            total = total + item.candidate.work
    return {
        f"search_work_{item.name}": int(getattr(total, item.name))
        for item in fields(total)
    }


def _scalar_refit_budget_summary(
    result: TumorSelectionOutcome,
    search_work: dict[str, int],
) -> dict[str, int | str]:
    """Separate objective-defining guide work from the post-guide cap ledger."""

    mandatory = getattr(result, "mandatory_guide_work", WorkCounters())
    if not isinstance(mandatory, WorkCounters):
        raise TypeError("mandatory_guide_work must be a WorkCounters value.")
    output: dict[str, int | str] = {
        "max_partition_refit_objective_evaluations_scope": (
            "post_mandatory_guide"
        )
    }
    for field_name in (
        "partition_refit_coordinates",
        "partition_refit_objective_evaluations",
    ):
        total = int(search_work[f"search_work_{field_name}"])
        guide = int(getattr(mandatory, field_name))
        if guide > total:
            raise ValueError("Mandatory guide work exceeds total search work.")
        output[f"mandatory_guide_{field_name}"] = guide
        output[f"post_guide_{field_name}"] = int(total - guide)
    return output


def analysis_summary(
    analysis: AnalysisSerialization,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Serialize the analysis summary without changing estimator state."""

    data = analysis.data
    fit_config = analysis.fit_config
    result = analysis.selection_result
    candidate = analysis.selected_candidate
    partition = analysis.partition
    refit = analysis.refit
    score = analysis.score
    raw_reference = analysis.raw_reference
    raw_fit = analysis.raw_fit
    parent_raw = analysis.partition_parent_raw
    primary = bool(analysis.primary_estimator_available)
    optimum_resolved = bool(getattr(result, "selection_optimum_resolved", False))
    selected_lambda = getattr(result, "selected_lambda_representative", None)
    analysis_tier = (
        "joint_certified"
        if primary
        else (
            "conditional_partition_refit"
            if candidate is not None
            else "unsupported_or_unidentified"
        )
    )
    summary: dict[str, object] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "ccf_cluster_ordering_method": CCF_CLUSTER_ORDERING_METHOD,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        "computation_profile": str(fit_config.computation_profile.name),
        "analysis_tier": analysis_tier,
        "primary_estimator_available": primary,
        "failure_reason": analysis.failure_reason,
        "selection_status": (
            ("resolved" if optimum_resolved else "provisional_unresolved")
            if primary
            else analysis_tier
        ),
        "selection_contract_id": str(fit_config.selection.contract.contract_id),
        "selection_optimum_resolved": optimum_resolved,
        "selection_boundary_unresolved": bool(
            getattr(result, "selection_boundary_unresolved", True)
        ),
        "selection_pool_stop_reason": str(
            getattr(result, "selection_pool_stop_reason", "none")
        ),
        "selection_hits_lower_boundary": bool(
            getattr(result, "selection_hits_lower_boundary", False)
        ),
        "selection_hits_upper_boundary": bool(
            getattr(result, "selection_hits_upper_boundary", False)
        ),
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "raw_reference_lambda": (
            None if raw_fit is None else float(raw_fit.provenance.lambda_value)
        ),
        "raw_reference_objective_certified": bool(
            raw_reference is not None and raw_reference.raw_objective_certified
        ),
        "selected_candidate_family": (
            None
            if candidate is None
            else (
                "raw_fusion"
                if isinstance(candidate, RawFusionCandidate)
                else "direct_partition"
            )
        ),
        "selected_partition_source": (
            None if partition is None else str(partition.source)
        ),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else None
        ),
        "selected_partition_parent_phi_hash": (
            str(partition.parent_raw_phi_hash)
            if isinstance(partition, DirectPartition) and partition.parent_raw_phi_hash
            else ""
        ),
        "selected_partition_parent_signature": (
            str(parent_raw.partition.signature) if parent_raw is not None else ""
        ),
        "selected_n_clusters": (
            None if partition is None else int(partition.n_clusters)
        ),
        "selected_partition_signature": (
            "" if partition is None else str(partition.signature)
        ),
        "selected_partition_certified": bool(
            isinstance(partition, FusionPartition) and partition.certified
        ),
        "selected_labels_hash": (
            ""
            if partition is None
            else _array_fingerprint(partition.labels, dtype=np.dtype(np.int64))
        ),
        "raw_reference_phi_hash": (
            ""
            if raw_fit is None
            else _array_fingerprint(raw_fit.phi, dtype=np.dtype(np.float64))
        ),
        "selected_fixed_partition_refit_centers_hash": (
            ""
            if refit is None
            else _array_fingerprint(refit.cluster_centers, dtype=np.dtype(np.float64))
        ),
        "selection_score_name": None if score is None else str(score.name),
        "selection_score": None if score is None else float(score.value),
        "selection_score_numerical_uncertainty": (
            None if score is None else float(score.numerical_uncertainty)
        ),
        "selection_loglik": None if score is None else float(score.loglik),
        "selection_df": None if score is None else int(score.degrees_of_freedom),
        "selection_penalty": None if score is None else float(score.penalty),
        "selection_n_eff": None if score is None else int(score.n_eff),
        "selection_assignment_log_evidence": (
            None if score is None else float(score.assignment_log_evidence)
        ),
        "selection_assignment_code_weight": (
            None if score is None else float(score.assignment_code_weight)
        ),
        "selection_assignment_penalty": (
            None if score is None else float(score.assignment_penalty)
        ),
        "selection_assignment_dirichlet_alpha": (
            None if score is None else float(score.assignment_dirichlet_alpha)
        ),
        "selected_raw_penalized_objective": (
            None if raw_fit is None else float(raw_fit.objective.total)
        ),
        "selected_refit_numerically_resolved": bool(
            refit is not None and refit.refit_numerically_resolved
        ),
        "selected_refit_global_optimum_certified": bool(
            refit is not None and refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": (
            None if refit is None else float(refit.global_optimality_gap)
        ),
        "selected_refit_global_lower_bound": (
            None if refit is None else float(refit.global_lower_bound)
        ),
        "selected_refit_global_certificate_method": (
            None if refit is None else str(refit.global_certificate_method)
        ),
        "selected_raw_solver_primal_tol": float(fit_config.solver.tolerance),
        "recovery_policy": str(fit_config.solver.recovery_policy),
        "stagnation_audit_patience": int(
            fit_config.solver.stagnation_audit_patience
        ),
        "lambda_no_progress_patience": int(
            fit_config.selection.lambda_search.no_progress_patience
        ),
        "max_tumor_edge_pass_equivalents": (
            fit_config.solver.resources.max_tumor_edge_pass_equivalents
        ),
        "max_partition_refit_objective_evaluations": (
            fit_config.solver.resources.max_partition_refit_objective_evaluations
        ),
        "max_direct_partition_candidates": (
            fit_config.solver.resources.max_direct_partition_candidates
        ),
        "selected_full_kkt_tolerance": (
            None if raw_fit is None else float(raw_fit.certificate.tolerance)
        ),
        "selected_full_kkt_residual_method": (
            None if raw_fit is None else str(raw_fit.certificate.residual_method)
        ),
        "selected_working_precision_kkt_residual": (
            None if raw_fit is None else float(raw_fit.certificate.working_residual)
        ),
        "selected_working_dtype": (
            None if raw_fit is None else str(raw_fit.certificate.working_dtype)
        ),
        "selected_certificate_audit_dtype": (
            None if raw_fit is None else str(raw_fit.certificate.audit_dtype)
        ),
        "selected_precision_polish_applied": bool(
            raw_fit is not None and raw_fit.certificate.precision_polished
        ),
        "selected_precision_polish_max_abs_phi_delta": (
            None
            if raw_fit is None
            else float(raw_fit.certificate.precision_polish_delta)
        ),
        "selected_base_fusion_objective_hash": (
            ""
            if raw_fit is None
            else str(raw_fit.provenance.base_fusion_objective_hash)
        ),
        "selected_original_graph_hash": (
            "" if raw_fit is None else str(raw_fit.provenance.original_graph_hash)
        ),
        "selection_method": str(getattr(result, "selection_method", "none")),
        "num_candidates": int(getattr(result, "num_candidates", 0)),
        "num_candidates_certified": int(getattr(result, "num_candidates_certified", 0)),
        "ward_candidate_pool_complete": bool(
            getattr(result, "ward_candidate_pool_complete", False)
        ),
        "raw_lambda_path_complete": bool(
            getattr(result, "raw_lambda_path_resolved", False)
        ),
        "global_hybrid_optimum_certified": bool(
            getattr(result, "global_hybrid_optimum_certified", False)
        ),
        "selected_kkt_residual": getattr(result, "selected_kkt_residual", None),
        "search_stop_reason": str(
            getattr(result, "adaptive_search_stop_reason", "not_recorded")
        ),
        "device": None if raw_fit is None else str(raw_fit.provenance.device),
        "dtype": None if raw_fit is None else str(raw_fit.provenance.dtype),
        "count_available_fraction": float(np.mean(data.count_available)),
        "likelihood_supported_fraction": float(np.mean(data.likelihood_supported)),
        "likelihood_included_fraction": float(np.mean(data.objective_inclusion_mask())),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_seconds_scope": "current_process_segment",
        "cumulative_search_active_seconds": float(
            getattr(result, "cumulative_search_active_seconds", 0.0)
        ),
        "resumed_from_checkpoint": bool(
            getattr(result, "resumed_from_checkpoint", False)
        ),
        "software_version": _SOFTWARE_VERSION,
    }
    summary.update(_raw_summary(analysis.diagnostic_raw_fit))
    search_work_summary = _search_work_summary(result)
    summary.update(search_work_summary)
    summary.update(_scalar_refit_budget_summary(result, search_work_summary))
    return summary


def _remove_exact_outputs(
    outdir: Path,
    tumor_id: str,
    suffixes: tuple[str, ...],
) -> None:
    for suffix in suffixes:
        path = outdir / f"{tumor_id}_{suffix}"
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_analysis_outputs(
    analysis: AnalysisSerialization,
    *,
    outdir: Path,
    summary: dict[str, object],
) -> None:
    """Write primary compatibility files and universal status-rich outputs."""

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    candidate = analysis.selected_candidate
    raw_fit = analysis.raw_fit
    diagnostic_raw = analysis.diagnostic_raw_fit
    primary = bool(analysis.primary_estimator_available)
    eps = float(
        analysis.fit_config.eps
        if diagnostic_raw is None
        else diagnostic_raw.provenance.likelihood_eps
    )
    cluster_region = cluster_region_estimates_table(
        data=analysis.data,
        candidate=candidate,
        primary_estimator_available=primary,
        diagnostic_raw_fit=diagnostic_raw,
        eps=eps,
        major_prior=float(analysis.fit_config.major_prior),
    )
    region_status = region_status_table(
        data=analysis.data,
        candidate=candidate,
        primary_estimator_available=primary,
        diagnostic_raw_fit=diagnostic_raw,
        failure_reason=analysis.failure_reason,
    )
    mutation_region = mutation_region_estimates_table(
        data=analysis.data,
        candidate=candidate,
        primary_estimator_available=primary,
        eps=eps,
        major_prior=float(analysis.fit_config.major_prior),
    )
    attempts = raw_attempts_table(tuple(analysis.selection_result.search))
    write_status_outputs(
        outdir=destination,
        data=analysis.data,
        summary=summary,
        region_status=region_status,
        raw_attempts=attempts,
        cluster_region_estimates=cluster_region,
        mutation_region_estimates=mutation_region,
    )

    tumor_id = str(analysis.data.tumor_id)
    if primary:
        partition = analysis.partition
        refit = analysis.refit
        if raw_fit is None or candidate is None or partition is None or refit is None:
            raise AssertionError(
                "Primary serialization requires a complete selected model."
            )
        _remove_exact_outputs(destination, tumor_id, _SECONDARY_SUFFIXES)
        write_fit_outputs(
            outdir=destination,
            data=analysis.data,
            raw_fit=raw_fit,
            partition=partition,
            refit=refit,
            major_prior=float(analysis.fit_config.major_prior),
        )
        return

    _remove_exact_outputs(destination, tumor_id, _PRIMARY_SUFFIXES)
    if candidate is not None:
        secondary_cluster_centers_table(
            data=analysis.data,
            candidate=candidate,
        ).to_csv(
            destination / f"{tumor_id}_secondary_cluster_centers.tsv",
            sep="\t",
            index=False,
        )
        mutation_region.to_csv(
            destination / f"{tumor_id}_secondary_mutation_region_estimates.tsv",
            sep="\t",
            index=False,
        )
    else:
        _remove_exact_outputs(destination, tumor_id, _SECONDARY_SUFFIXES)


__all__ = [
    "AnalysisSerialization",
    "SUMMARY_SCHEMA_VERSION",
    "analysis_summary",
    "write_analysis_outputs",
]
