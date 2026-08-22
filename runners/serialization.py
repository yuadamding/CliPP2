"""Versioned serialization boundary for one completed tumor analysis.

The numerical and model-selection layers return typed in-memory objects.  This
module is the only runner boundary that translates those objects into the
compatibility summary and the three public TSVs.  Raw-fusion references and
direct-partition parent identity stay inside :class:`AnalysisSerialization`;
the public table writer receives only the selected partition and its immutable
fixed-label refit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from .._version import __version__ as _SOFTWARE_VERSION
from ..config import FitConfig
from ..core.fusion.types import RawFit
from ..io.data import TumorData
from ..model_selection.types import (
    BICSelectionResult,
    DirectPartition,
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    SelectablePartitionCandidate,
    SelectionScore,
)
from .outputs import write_fit_outputs

SUMMARY_SCHEMA_VERSION = 3


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisSerialization:
    """One normalized, output-ready view of a completed selection result.

    Everything downstream consumes this normalized typed boundary.
    """

    data: TumorData
    input_file: Path
    fit_config: FitConfig
    selection_result: BICSelectionResult

    @property
    def selected_candidate(self) -> SelectablePartitionCandidate:
        return self.selection_result.selected_model.partition_candidate

    @property
    def raw_reference(self) -> RawFusionCandidate:
        return self.selection_result.selected_model.raw_reference

    @property
    def partition_parent_raw(self) -> RawFusionCandidate | None:
        return self.selection_result.selected_model.partition_parent_raw

    @property
    def raw_fit(self) -> RawFit:
        return self.raw_reference.raw_fit

    @property
    def partition(self) -> FusionPartition | DirectPartition:
        return self.selected_candidate.partition

    @property
    def refit(self) -> PartitionRefitSummary:
        return self.selected_candidate.refit

    @property
    def score(self) -> SelectionScore:
        return self.selected_candidate.score


def analysis_summary(
    analysis: AnalysisSerialization,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Serialize schema-v3 diagnostics without changing estimator state."""

    data = analysis.data
    fit_config = analysis.fit_config
    profile = fit_config.computation_profile
    result = analysis.selection_result
    selected_model = result.selected_model
    raw_reference = analysis.raw_reference
    raw_fit = analysis.raw_fit
    parent_raw = analysis.partition_parent_raw
    partition = analysis.partition
    refit = analysis.refit
    score = analysis.score
    selected_lambda = result.selected_lambda_representative
    optimum_resolved = bool(result.selection_optimum_resolved)
    selection_contract = fit_config.selection.contract
    selected_partition_certified = bool(
        isinstance(partition, FusionPartition) and partition.certified
    )
    exactness = raw_fit.certificate

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        "computation_profile": str(profile.name),
        "selection_status": (
            "resolved" if optimum_resolved else "provisional_unresolved"
        ),
        "selection_contract_id": str(selection_contract.contract_id),
        "selection_optimum_resolved": optimum_resolved,
        "selection_boundary_unresolved": bool(result.selection_boundary_unresolved),
        "selection_hits_lower_boundary": bool(result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(result.selection_hits_upper_boundary),
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "raw_reference_lambda": float(
            raw_fit.provenance.lambda_value
        ),
        "raw_reference_objective_certified": bool(
            raw_reference.raw_objective_certified
        ),
        "selected_candidate_family": str(selected_model.selected_candidate_family),
        "selected_partition_source": str(partition.source),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else None
        ),
        "selected_partition_parent_phi_hash": (
            str(partition.parent_raw_phi_hash)
            if isinstance(partition, DirectPartition) and parent_raw is not None
            else ""
        ),
        "selected_partition_parent_signature": (
            str(parent_raw.partition.signature) if parent_raw is not None else ""
        ),
        "selected_n_clusters": int(partition.n_clusters),
        "selected_partition_signature": str(partition.signature),
        "selected_partition_certified": selected_partition_certified,
        "selected_labels_hash": _array_fingerprint(
            partition.labels,
            dtype=np.dtype(np.int64),
        ),
        "raw_reference_phi_hash": _array_fingerprint(
            raw_fit.phi,
            dtype=np.dtype(np.float64),
        ),
        "selected_fixed_partition_refit_centers_hash": _array_fingerprint(
            refit.cluster_centers,
            dtype=np.dtype(np.float64),
        ),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "selection_score_numerical_uncertainty": float(score.numerical_uncertainty),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "selection_assignment_log_evidence": float(score.assignment_log_evidence),
        "selection_assignment_code_weight": float(score.assignment_code_weight),
        "selection_assignment_penalty": float(score.assignment_penalty),
        "selection_assignment_dirichlet_alpha": float(
            score.assignment_dirichlet_alpha
        ),
        "selected_raw_penalized_objective": float(raw_fit.objective.total),
        "selected_refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "selected_refit_global_optimum_certified": bool(
            refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": float(refit.global_optimality_gap),
        "selected_refit_global_lower_bound": float(refit.global_lower_bound),
        "selected_refit_global_certificate_method": str(
            refit.global_certificate_method
        ),
        "selected_raw_solver_primal_tol": float(fit_config.solver.tolerance),
        "selected_full_kkt_tolerance": float(raw_fit.certificate.tolerance),
        "selected_full_kkt_residual_method": str(
            exactness.residual_method
        ),
        "selected_working_precision_kkt_residual": float(
            raw_fit.certificate.working_residual
        ),
        "selected_working_dtype": str(
            raw_fit.certificate.working_dtype
        ),
        "selected_certificate_audit_dtype": str(
            raw_fit.certificate.audit_dtype
        ),
        "selected_precision_polish_applied": bool(
            raw_fit.certificate.precision_polished
        ),
        "selected_precision_polish_max_abs_phi_delta": float(
            raw_fit.certificate.precision_polish_delta
        ),
        "selected_base_fusion_objective_hash": str(
            raw_fit.provenance.base_fusion_objective_hash
        ),
        "selected_original_graph_hash": str(raw_fit.provenance.original_graph_hash),
        "selection_method": str(result.selection_method),
        "num_candidates": int(result.num_candidates),
        "num_candidates_certified": int(result.num_candidates_certified),
        "ward_candidate_pool_complete": bool(result.ward_candidate_pool_complete),
        "raw_lambda_path_complete": bool(result.raw_lambda_path_resolved),
        "global_hybrid_optimum_certified": bool(
            result.global_hybrid_optimum_certified
        ),
        "selected_kkt_residual": (
            None
            if result.selected_kkt_residual is None
            else float(result.selected_kkt_residual)
        ),
        "search_stop_reason": str(result.adaptive_search_stop_reason),
        "device": str(raw_fit.provenance.device),
        "dtype": str(raw_fit.provenance.dtype),
        "elapsed_seconds": float(elapsed_seconds),
        "software_version": _SOFTWARE_VERSION,
    }


def write_analysis_outputs(
    analysis: AnalysisSerialization,
    *,
    outdir: Path,
) -> None:
    """Write only the three stable public tables for ``analysis``."""

    write_fit_outputs(
        outdir=Path(outdir),
        data=analysis.data,
        raw_fit=analysis.raw_fit,
        partition=analysis.partition,
        refit=analysis.refit,
        major_prior=float(analysis.fit_config.major_prior),
    )


__all__ = [
    "AnalysisSerialization",
    "SUMMARY_SCHEMA_VERSION",
    "analysis_summary",
    "write_analysis_outputs",
]
