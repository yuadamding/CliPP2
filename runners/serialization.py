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
from typing import cast

import numpy as np

from .._version import __version__ as _SOFTWARE_VERSION
from ..config import FitOptions
from ..core.fusion.profiles import ComputationProfile
from ..core.model import FitResult
from ..io.data import TumorData
from ..model_selection.contracts import get_selection_contract
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

    ``from_selection`` is the single compatibility adapter for older
    lightweight callers whose selected model exposes ``candidate`` instead of
    the current ``partition_candidate``/``raw_reference`` pair.  Everything
    downstream consumes these normalized fields without further fallbacks.
    """

    data: TumorData
    input_file: Path
    fit_options: FitOptions
    computation_profile: ComputationProfile
    selection_result: BICSelectionResult
    selected_candidate: SelectablePartitionCandidate
    raw_reference: RawFusionCandidate
    partition_parent_raw: RawFusionCandidate | None

    @classmethod
    def from_selection(
        cls,
        *,
        data: TumorData,
        input_file: Path,
        fit_options: FitOptions,
        computation_profile: ComputationProfile,
        selection_result: BICSelectionResult,
    ) -> AnalysisSerialization:
        selected_model = selection_result.selected_model
        selected_candidate = getattr(selected_model, "partition_candidate", None)
        if selected_candidate is None:
            selected_candidate = getattr(selected_model, "candidate")
        return cls(
            data=data,
            input_file=Path(input_file),
            fit_options=fit_options,
            computation_profile=computation_profile,
            selection_result=selection_result,
            selected_candidate=cast(SelectablePartitionCandidate, selected_candidate),
            raw_reference=cast(
                RawFusionCandidate,
                getattr(selected_model, "raw_reference", selected_candidate),
            ),
            partition_parent_raw=cast(
                RawFusionCandidate | None,
                getattr(selected_model, "partition_parent_raw", None),
            ),
        )

    @property
    def raw_fit(self) -> FitResult:
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
    fit_options = analysis.fit_options
    profile = analysis.computation_profile
    result = analysis.selection_result
    selected_model = result.selected_model
    raw_reference = analysis.raw_reference
    raw_fit = analysis.raw_fit
    raw_partition = raw_reference.partition
    parent_raw = analysis.partition_parent_raw
    partition = analysis.partition
    refit = analysis.refit
    score = analysis.score
    selected_lambda = result.selected_lambda_representative
    optimum_resolved = bool(result.selection_optimum_resolved)
    selection_contract = get_selection_contract(
        getattr(fit_options, "selection_contract", "hybrid-ward-cem-v1")
    )
    selected_partition_certified = bool(
        isinstance(partition, FusionPartition) and partition.certified
    )
    exactness = getattr(raw_fit, "exactness_provenance", None)

    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        "computation_profile": str(profile.name),
        "target_estimator": "complete_graph_pairwise_fusion",
        "solution_mode": (
            "strict_certified" if profile.is_strict else "approximate_single_tumor_search"
        ),
        "selection_status": (
            "resolved" if optimum_resolved else "provisional_unresolved"
        ),
        "selection_constraint": "none",
        "selection_contract_id": str(selection_contract.contract_id),
        "selection_contract_json": str(selection_contract.to_json()),
        "selection_optimum_resolved": optimum_resolved,
        "selection_boundary_unresolved": bool(result.selection_boundary_unresolved),
        "selection_hits_lower_boundary": bool(result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(result.selection_hits_upper_boundary),
        "objective_equivalent_to_strict_graph": bool(
            profile.objective_equivalent_to_strict
        ),
        "scalar_mode": str(profile.scalar_mode),
        "selected_refit_mode": str(refit.refit_mode),
        "selected_lambda": (
            np.nan if selected_lambda is None else float(selected_lambda)
        ),
        "selected_lambda_applicable": bool(selected_lambda is not None),
        "raw_reference_lambda": float(
            getattr(
                raw_fit,
                "lambda_value",
                np.nan if selected_lambda is None else selected_lambda,
            )
        ),
        "raw_reference_partition_signature": str(raw_partition.signature),
        "raw_reference_objective_certified": bool(
            getattr(raw_reference, "raw_objective_certified", True)
        ),
        "selected_candidate_family": str(
            getattr(selected_model, "selected_candidate_family", "raw_fusion")
        ),
        "selected_partition_source": str(partition.source),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else np.nan
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
        "selected_partition_certification_applicable": bool(
            isinstance(partition, FusionPartition)
        ),
        "selected_partition_maximal": bool(
            partition.maximal if isinstance(partition, FusionPartition) else False
        ),
        "selected_direct_partition_identity_certified": bool(
            isinstance(partition, DirectPartition) and partition.deterministic_generation
        ),
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
        "selection_assignment_model_id": str(score.assignment_model_id),
        "selection_assignment_symmetry_mode": str(score.assignment_symmetry_mode),
        "selection_assignment_arithmetic_uncertainty": float(
            score.assignment_arithmetic_uncertainty
        ),
        "selected_raw_penalized_objective": float(raw_fit.penalized_objective),
        "selected_refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "selected_refit_global_optimum_certified": bool(
            refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": float(refit.global_optimality_gap),
        "selected_refit_global_lower_bound": float(refit.global_lower_bound),
        "selected_refit_global_certificate_method": str(
            refit.global_certificate_method
        ),
        "selected_raw_solver_primal_tol": float(fit_options.tol),
        "selected_full_kkt_tolerance": float(raw_fit.full_kkt_tolerance),
        "selected_full_kkt_residual_method": str(
            getattr(exactness, "residual_method", "unknown")
            if exactness is not None
            else "unknown"
        ),
        "selected_working_precision_kkt_residual": float(
            getattr(raw_fit, "working_precision_kkt_residual", np.nan)
        ),
        "selected_working_dtype": str(
            getattr(raw_fit, "working_dtype", getattr(raw_fit, "dtype", "unknown"))
        ),
        "selected_certificate_audit_dtype": str(
            getattr(raw_fit, "certificate_audit_dtype", "unknown")
        ),
        "selected_precision_polish_applied": bool(
            getattr(raw_fit, "precision_polish_applied", False)
        ),
        "selected_precision_polish_max_abs_phi_delta": float(
            getattr(raw_fit, "precision_polish_max_abs_phi_delta", 0.0)
        ),
        "selected_objective_spec_hash": str(raw_fit.objective_spec_hash),
        "selected_base_fusion_objective_hash": str(
            raw_fit.base_fusion_objective_hash
        ),
        "selected_original_graph_hash": str(raw_fit.original_graph_hash),
        "selection_method": str(result.selection_method),
        "num_candidates": int(result.num_candidates),
        "num_candidates_certified": int(result.num_candidates_certified),
        "ward_candidate_pool_complete": bool(
            getattr(result, "ward_candidate_pool_complete", False)
        ),
        "raw_lambda_path_complete": bool(
            getattr(result, "raw_lambda_path_resolved", False)
        ),
        "global_hybrid_optimum_certified": bool(
            getattr(result, "global_hybrid_optimum_certified", False)
        ),
        "selected_kkt_residual": (
            np.nan
            if result.selected_kkt_residual is None
            else float(result.selected_kkt_residual)
        ),
        "search_stop_reason": str(result.adaptive_search_stop_reason),
        "device": str(raw_fit.device),
        "dtype": str(raw_fit.dtype),
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
        major_prior=float(analysis.fit_options.major_prior),
    )


__all__ = [
    "AnalysisSerialization",
    "SUMMARY_SCHEMA_VERSION",
    "analysis_summary",
    "write_analysis_outputs",
]
