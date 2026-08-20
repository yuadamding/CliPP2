from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd

from ..core.model import (
    FitOptions,
    FitResult,
    fit_fixed_objective,
)
from ..core.fusion.partition_starts import (
    PartitionCandidate,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.profiles import get_computation_profile
from ..core.fusion.solver import (
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
    torch_data_from_context,
)
from ..core.fusion.types import (
    SolverState,
)
from ..io.data import TumorData

from ..model_selection.candidates import (
    PartitionRefitCacheEntry,
    evaluate_direct_partition_candidate,
    evaluate_raw_fusion_candidate,
    validate_candidate_identity,
)
from ..model_selection.contracts import get_selection_contract
from ..model_selection.config import (
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
)
from ..model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
)
from ..model_selection.partition_initializer import (
    PartitionInitializerPool,
    generate_partition_initializer_pool,
)
from ..model_selection.partitions import (
    _best_partition_candidate,
    _cluster_sizes_text,
    _partition_candidate_requested_k,
    _partition_signature,
)
from ..model_selection.proposals import (
    RawStartAttempt as _RawStartAttempt,
    RawStartSpec as _RawStartSpec,
    adaptive_stop_certifies_global_optimum as _adaptive_stop_certifies_global_optimum,
    bootstrap_independent_start_specs as _bootstrap_independent_start_specs,
    build_guided_initialization_with_resource_policy as _build_guided_initialization_with_resource_policy,
    build_partition_guided_graph_with_resource_policy as _build_partition_guided_graph_with_resource_policy,
    candidate_static_metadata as _candidate_static_metadata,
    clone_start as _clone_start,
    direct_partition_source as _direct_partition_source,
    escape_path_breakpoint_retry_state as _escape_path_breakpoint_retry_state,
    offload_solver_state_to_cpu as _offload_solver_state_to_cpu,
    partition_pool_row_metadata as _partition_pool_row_metadata,
    pilot_matrix_hash as _pilot_matrix_hash,
    raw_start_attempt_diagnostic as _raw_start_attempt_diagnostic,
    rescore_partition_candidates as _rescore_partition_candidates,
    select_raw_start_attempt as _select_raw_start_attempt,
    solver_recovery_fit_options as _solver_recovery_fit_options,
)
from ..model_selection.scoring import (
    _annotate_bic_diagnostics,
    _canonical_lambda,
    _prefer_fit_candidate,
    _sorted_unique_lambdas,
    candidate_representative_ids,
    raw_candidate_has_exact_fusion_certificate,
    select_candidate_records,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateRecord,
    RawFusionCandidate,
    SelectedModel,
    StartArray,
)


class NoEligibleModelSelectionCandidatesError(RuntimeError):
    """Model selection failed after every annotated candidate was rejected."""

    tumor_id: str
    search_df: pd.DataFrame

    def __init__(self, tumor_id: str, search_df: pd.DataFrame) -> None:
        self.tumor_id = str(tumor_id)
        self.search_df = search_df.copy()
        super().__init__(
            f"No candidates were eligible for model selection for tumor "
            f"{self.tumor_id}."
        )


class NoCertifiedRawReferenceError(RuntimeError):
    """Hybrid selection found direct candidates but no certified raw reference.

    The full in-memory search table is retained on the exception so a failed
    batch case remains diagnosable without widening the three public output
    tables or weakening the raw-fusion KKT contract.
    """

    tumor_id: str
    search_df: pd.DataFrame
    adaptive_search_stop_reason: str
    diagnostics: dict[str, object]

    def __init__(
        self,
        *,
        tumor_id: str,
        search_df: pd.DataFrame,
        adaptive_search_stop_reason: str,
    ) -> None:
        self.tumor_id = str(tumor_id)
        self.search_df = search_df.copy()
        self.adaptive_search_stop_reason = str(adaptive_search_stop_reason)
        self.diagnostics = _raw_reference_failure_diagnostics(self.search_df)
        reason_counts = self.diagnostics["ineligibility_reason_counts"]
        certificate_counts = self.diagnostics["certificate_status_counts"]
        super().__init__(
            "Hybrid selection requires a certified raw-fusion reference for "
            f"tumor {self.tumor_id}; "
            f"stop={self.adaptive_search_stop_reason}; "
            f"raw_candidates={self.diagnostics['raw_candidate_count']}; "
            f"raw_solver_attempts={self.diagnostics['raw_solver_attempt_count']}; "
            f"raw_certified={self.diagnostics['raw_certified_count']}; "
            f"partition_certified={self.diagnostics['partition_certified_count']}; "
            f"min_kkt_residual={self.diagnostics['min_kkt_residual']}; "
            f"min_kkt_tolerance={self.diagnostics['min_kkt_tolerance']}; "
            "dominant_kkt_component="
            f"{self.diagnostics['best_raw_attempt_dominant_kkt_component']}; "
            f"mm_violating={self.diagnostics['mm_violating_count']}; "
            f"ineligibility_reasons={reason_counts}; "
            f"certificate_statuses={certificate_counts}."
        )


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return stable compact counts for one diagnostic column."""

    if column not in frame.columns:
        return {}
    counts = frame[column].fillna("<missing>").astype(str).value_counts()
    return {
        str(key): int(counts[key])
        for key in sorted(counts.index, key=lambda value: str(value))
    }


def _raw_reference_failure_diagnostics(search_df: pd.DataFrame) -> dict[str, object]:
    """Summarize why no raw candidate can provide hybrid provenance."""

    if "candidate_family" in search_df.columns:
        raw = search_df.loc[
            search_df["candidate_family"].astype(str).eq("raw_fusion")
        ].copy()
    else:
        raw = search_df.iloc[0:0].copy()

    attempt_details: list[dict[str, object]] = []
    if "raw_start_attempt_diagnostics" in raw.columns:
        for row_index, value in raw["raw_start_attempt_diagnostics"].items():
            if isinstance(value, (list, tuple)):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    detail = dict(item)
                    for column in ("search_round", "search_phase", "lambda"):
                        detail.setdefault(column, raw.loc[row_index].get(column))
                    attempt_details.append(detail)
    if not attempt_details:
        for _, row in raw.iterrows():
            attempt_details.append(
                {
                    "source": str(row.get("lambda_start_source", "unknown")),
                    "kkt_residual": row.get(
                        "fixed_objective_kkt_residual", float("nan")
                    ),
                    "kkt_tolerance": row.get("full_kkt_tolerance", float("nan")),
                    "full_kkt_certified": row.get("full_kkt_certified", False),
                    "certificate_status": row.get(
                        "full_kkt_certificate_status", "unknown"
                    ),
                    "mm_consistency_violations": row.get(
                        "mm_consistency_violations", 0
                    ),
                }
            )
    attempt_frame = pd.DataFrame(attempt_details)
    residuals = pd.to_numeric(
        attempt_frame.get(
            "kkt_residual",
            pd.Series(np.nan, index=attempt_frame.index),
        ),
        errors="coerce",
    )
    finite_residuals = residuals[np.isfinite(residuals.to_numpy(dtype=float))]
    min_residual = float("nan")
    min_tolerance = float("nan")
    best_attempt: dict[str, object] = {}
    dominant_component = "unknown"
    if not finite_residuals.empty:
        best_index = finite_residuals.idxmin()
        min_residual = float(finite_residuals.loc[best_index])
        best_row = attempt_frame.loc[best_index]
        if "kkt_tolerance" in attempt_frame.columns:
            tolerance = pd.to_numeric(
                pd.Series([attempt_frame.loc[best_index, "kkt_tolerance"]]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(float(tolerance)):
                min_tolerance = float(tolerance)
        backward_error_available = bool(
            str(best_row.get("certificate_residual_method", ""))
            == "componentwise_box_cone_backward_error_v1"
        )
        component_columns = (
            {
                "stationarity": "backward_error_stationarity_residual",
                "edge_subgradient": "backward_error_edge_subgradient_residual",
                "dual_ball": "backward_error_dual_ball_residual",
                "box": "box_residual",
            }
            if backward_error_available
            else {
                "stationarity": "stationarity_residual",
                "edge_subgradient": "edge_subgradient_residual",
                "dual_ball": "dual_ball_residual",
                "box": "box_residual",
            }
        )
        components = {
            name: float(best_row[column])
            for name, column in component_columns.items()
            if column in best_row.index
            and np.isfinite(float(pd.to_numeric(best_row[column], errors="coerce")))
        }
        if components:
            dominant_component = max(components, key=components.get)
        for key in (
            "search_round",
            "search_phase",
            "lambda",
            "source",
            "n_clusters",
            "outer_iterations",
            "inner_iterations",
            "admm_iterations",
            "certificate_iterations",
            "outer_max_iter",
            "inner_max_iter",
            "certificate_max_iter",
            "stationarity_before_dual_refine",
            "stationarity_after_dual_refine",
        ):
            if key in best_row.index:
                value = best_row[key]
                best_attempt[key] = (
                    value.item() if isinstance(value, np.generic) else value
                )
        best_attempt["kkt_residual"] = min_residual
        best_attempt["kkt_tolerance"] = min_tolerance
        best_attempt["dominant_kkt_component"] = dominant_component

    mm_values = pd.to_numeric(
        attempt_frame.get(
            "mm_consistency_violations",
            pd.Series(0, index=attempt_frame.index),
        ),
        errors="coerce",
    ).fillna(0)
    raw_certified = (
        raw.get(
            "raw_objective_certified",
            pd.Series(False, index=raw.index),
        )
        .fillna(False)
        .astype(bool)
    )
    partition_certified = (
        raw.get(
            "partition_certified",
            pd.Series(False, index=raw.index),
        )
        .fillna(False)
        .astype(bool)
    )
    direct = search_df.loc[
        search_df.get(
            "candidate_family",
            pd.Series("", index=search_df.index),
        )
        .astype(str)
        .eq("direct_partition")
    ]
    direct_eligible = (
        direct.get(
            "eligible_for_selection",
            pd.Series(False, index=direct.index),
        )
        .fillna(False)
        .astype(bool)
    )
    return {
        "raw_candidate_count": int(raw.shape[0]),
        "raw_solver_attempt_count": int(attempt_frame.shape[0]),
        "raw_certified_count": int(raw_certified.sum()),
        "partition_certified_count": int(partition_certified.sum()),
        "direct_candidate_count": int(direct.shape[0]),
        "direct_eligible_count": int(direct_eligible.sum()),
        "min_kkt_residual": min_residual,
        "min_kkt_tolerance": min_tolerance,
        "best_raw_attempt": best_attempt,
        "best_raw_attempt_dominant_kkt_component": dominant_component,
        "mm_violation_min": (int(mm_values.min()) if not mm_values.empty else 0),
        "mm_violating_count": int((mm_values > 0).sum()),
        "ineligibility_reason_counts": _value_counts(raw, "ineligibility_reason"),
        "certificate_status_counts": _value_counts(attempt_frame, "certificate_status"),
        "search_phase_counts": _value_counts(raw, "search_phase"),
        "start_source_counts": _value_counts(raw, "lambda_start_source"),
    }


def _candidate_record_representatives(
    records: list[CandidateRecord],
) -> set[int]:
    """Compatibility wrapper for the typed exact-admission selector."""

    return set(candidate_representative_ids(records))


def candidates_to_dataframe(records: list[CandidateRecord]) -> pd.DataFrame:
    """Materialize the compatibility search table at the reporting boundary."""

    rows: list[dict[str, object]] = []
    for record in records:
        candidate = record.candidate
        raw = candidate if isinstance(candidate, RawFusionCandidate) else None
        partition = candidate.partition
        refit = candidate.refit
        score = candidate.score
        row = dict(record.row)
        # Typed identity and inferential fields are authoritative. Diagnostic
        # payloads may add columns but cannot silently change model selection.
        row.update(
            {
                "_candidate_id": int(record.candidate_id),
                "candidate_id": int(record.candidate_id),
                "candidate_family": str(record.family),
                "candidate_pool_source": (
                    "raw_fused_lambda_path"
                    if raw is not None
                    else str(partition.source)
                ),
                "partition_source": str(partition.source),
                "partition_signature": str(record.partition_signature),
                "n_clusters": int(record.n_clusters),
                "cluster_sizes": _cluster_sizes_text(partition.labels),
                "partition_labels_0based": ",".join(
                    str(int(value))
                    for value in np.asarray(partition.labels, dtype=np.int64)
                ),
                "selection_score_name": str(score.name),
                "selection_score": float(score.value),
                "selection_score_numerical_uncertainty": float(
                    score.numerical_uncertainty
                ),
                "selection_score_lower_bound": float(score.lower_bound),
                "selection_score_upper_bound": float(score.upper_bound),
                "selection_loglik": float(score.loglik),
                "selection_df": int(score.degrees_of_freedom),
                "selection_penalty": float(score.penalty),
                "selection_n_eff": int(score.n_eff),
                "selection_contract_id": str(score.selection_contract_id),
                "bic_loglik_source": str(refit.loglik_source),
                "bic_refit_finite_candidate_found": bool(
                    refit.finite_candidate_found
                ),
                "refit_mode": str(refit.refit_mode),
                "computation_profile": str(candidate.computation_profile),
                "eligible_for_selection": bool(record.eligible_for_selection),
                "ineligibility_reason": str(candidate.ineligibility_reason),
                "lambda": (
                    float(record.lambda_value)
                    if record.lambda_value is not None
                    else float("nan")
                ),
                "lambda_applicable": bool(record.lambda_value is not None),
                "penalized_objective": (
                    float(record.penalized_objective)
                    if record.penalized_objective is not None
                    else float("nan")
                ),
                "mm_consistency_violations": int(
                    record.mm_consistency_violations
                ),
                "partition_certified": bool(
                    raw is not None and partition.certified
                ),
                "partition_certification_applicable": bool(raw is not None),
                "raw_objective_certified": bool(
                    raw is not None and raw.raw_objective_certified
                ),
            }
        )
        row.update(
            {
                f"selection_assignment_{name}": getattr(score, f"assignment_{name}")
                for name in (
                    "log_evidence",
                    "code_weight",
                    "penalty",
                    "dirichlet_alpha",
                    "model_id",
                    "symmetry_mode",
                    "arithmetic_uncertainty",
                )
            }
        )
        row.update(
            {
                f"refit_{name}": getattr(refit, name)
                for name in (
                    "global_optimum_certified",
                    "global_lower_bound",
                    "global_optimality_gap",
                    "global_certificate_method",
                    "global_certificate_intervals",
                    "loglik",
                    "fit_loss",
                    "active_df",
                )
            }
        )
        row.update(
            {
                name: getattr(refit, name)
                for name in (
                    "refit_numerically_resolved",
                    "refit_coordinate_count",
                    "refit_finite_coordinate_count",
                    "refit_total_grid_points",
                    "refit_max_grid_spacing",
                    "refit_total_candidate_basins",
                    "refit_total_refined_candidates",
                    "refit_min_best_second_loss_gap",
                )
            }
        )
        if raw is not None:
            fit = raw.raw_fit
            provenance = getattr(fit, "exactness_provenance", None)
            row.update(
                {
                    "partition_tol": float(partition.tolerance),
                    "partition_maximal": bool(partition.maximal),
                    "partition_cross_close_edge_found": bool(
                        partition.cross_close_edge_found
                    ),
                    "partition_certificate_graph_hash_matches": bool(
                        partition.certificate_graph_hash_matches
                    ),
                    "partition_certification_failure_reason": str(
                        partition.certification_failure_reason
                    ),
                    "partition_max_diameter": float(partition.max_diameter),
                    "partition_diameter_exact": bool(partition.diameter_exact),
                    "raw_kkt_eligible": bool(fit.selection_eligible),
                    "estimator_role": str(fit.estimator_role),
                    "objective_faithful": bool(fit.objective_faithful),
                    "objective_spec_hash": str(fit.objective_spec_hash),
                    "original_graph_hash": str(fit.original_graph_hash),
                    "certificate_problem_hash": str(
                        fit.certificate_problem_hash
                    ),
                    "certificate_scope": str(fit.certificate_scope),
                    "certificate_gradient_scope": str(
                        fit.certificate_gradient_scope
                    ),
                    "full_kkt_certified": bool(fit.full_kkt_certified),
                    "full_kkt_certificate_status": str(
                        fit.full_kkt_certificate_status
                    ),
                    "full_kkt_tolerance": float(fit.full_kkt_tolerance),
                    "fixed_objective_kkt_residual": float(
                        fit.fixed_objective_kkt_residual
                    ),
                    "exactness_provenance_version": int(
                        fit.exactness_provenance_version
                    ),
                    "certificate_residual_method": str(
                        getattr(provenance, "residual_method", "unknown")
                    ),
                    "certificate_audit_dtype": str(
                        fit.certificate_audit_dtype
                    ),
                }
            )
        else:
            row.update(
                {
                    "requested_K": int(partition.requested_k),
                    "parent_raw_candidate_id": (
                        float("nan")
                        if partition.parent_raw_candidate_id is None
                        else int(partition.parent_raw_candidate_id)
                    ),
                    "parent_raw_lambda": (
                        float("nan")
                        if partition.parent_raw_lambda is None
                        else float(partition.parent_raw_lambda)
                    ),
                    "parent_raw_phi_hash": str(partition.parent_raw_phi_hash),
                    "generation_contract_id": str(
                        partition.generation_contract_id
                    ),
                    "cem_iterations": int(partition.cem_iterations),
                    "component_death_count": int(
                        partition.component_death_count
                    ),
                    "refinement_score_before": float(
                        partition.refinement_score_before
                    ),
                    "refinement_score_after": float(
                        partition.refinement_score_after
                    ),
                    "deterministic_partition_generation": bool(
                        partition.deterministic_generation
                    ),
                }
            )
        row.setdefault("selection_step", int(record.candidate_id))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["lambda", "selection_step"], kind="stable")
        .reset_index(drop=True)
    )


def _select_raw_reference(
    records: list[CandidateRecord],
    *,
    tumor_id: str,
    search_df: pd.DataFrame | None = None,
    adaptive_search_stop_reason: str,
) -> CandidateRecord:
    eligible = [
        record
        for record in records
        if isinstance(record.candidate, RawFusionCandidate)
        and raw_candidate_has_exact_fusion_certificate(record.candidate)
    ]
    if not eligible:
        if search_df is None:
            search_df = candidates_to_dataframe(records)
        raise NoCertifiedRawReferenceError(
            tumor_id=tumor_id,
            search_df=search_df,
            adaptive_search_stop_reason=adaptive_search_stop_reason,
        )
    return min(
        eligible,
        key=lambda record: (
            float(record.score.value),
            float(record.score.numerical_uncertainty),
            float(record.candidate.raw_fit.lambda_value),
            float(record.candidate.raw_fit.penalized_objective),
            int(record.candidate_id),
        ),
    )


def _assemble_selection_result(
    *,
    data,
    normalized_score,
    result_entries,
    selection_method,
    adaptive_search_stop_reason,
    selection_start_time,
    strict_positive_exact_fusion: bool = False,
    ward_candidate_pool_complete: bool = False,
) -> BICSelectionResult:
    del normalized_score  # Every typed record already carries its active score.
    try:
        decision = select_candidate_records(
            result_entries,
            strict_positive_exact_fusion=bool(strict_positive_exact_fusion),
        )
    except ValueError as exc:
        search_df = _annotate_bic_diagnostics(
            candidates_to_dataframe(result_entries)
        )
        representative_ids = candidate_representative_ids(
            result_entries,
            strict_positive_exact_fusion=bool(strict_positive_exact_fusion),
        )
        search_df["signature_selection_representative"] = search_df[
            "_candidate_id"
        ].astype(int).isin(representative_ids)
        if strict_positive_exact_fusion:
            search_df["bic_selection_eligible"] = False
        raise NoEligibleModelSelectionCandidatesError(
            tumor_id=data.tumor_id,
            search_df=search_df,
        ) from exc

    selected_record = decision.selected
    selected_candidate = selected_record.candidate
    validate_candidate_identity(selected_candidate)
    if not selected_candidate.eligible_for_selection:
        raise AssertionError("Ineligible partition candidate reached selection.")

    # The raw estimator reference is chosen independently from the selected
    # partition and never inherited by a direct candidate.
    base_search_df = candidates_to_dataframe(result_entries)
    raw_reference_record = _select_raw_reference(
        result_entries,
        tumor_id=str(data.tumor_id),
        search_df=base_search_df,
        adaptive_search_stop_reason=str(adaptive_search_stop_reason),
    )
    raw_reference = raw_reference_record.candidate
    if not isinstance(raw_reference, RawFusionCandidate):  # pragma: no cover
        raise AssertionError("Raw-reference selection returned a direct partition.")
    validate_candidate_identity(raw_reference)
    selected_is_raw = isinstance(selected_candidate, RawFusionCandidate)
    records_by_id = {int(record.candidate_id): record for record in result_entries}
    partition_parent_raw: RawFusionCandidate | None = None
    if not selected_is_raw:
        parent_id = selected_candidate.partition.parent_raw_candidate_id
        if parent_id is not None:
            parent_record = records_by_id.get(int(parent_id))
            if parent_record is None or not isinstance(
                parent_record.candidate, RawFusionCandidate
            ):
                raise AssertionError(
                    "Direct partition refers to a missing/non-raw parent candidate."
                )
            partition_parent_raw = parent_record.candidate
            expected_hash = str(selected_candidate.partition.parent_raw_phi_hash)
            observed_hash = _pilot_matrix_hash(partition_parent_raw.raw_fit.phi)
            if not expected_hash or observed_hash != expected_hash:
                raise AssertionError(
                    "Direct partition parent-Phi provenance is inconsistent."
                )

    final_adaptive_search_stop_reason = str(adaptive_search_stop_reason)
    adaptive_search_global_optimum_certified = _adaptive_stop_certifies_global_optimum(
        final_adaptive_search_stop_reason
    )
    selection_optimum_resolved = bool(
        adaptive_search_global_optimum_certified
        and not decision.selection_boundary_unresolved
    )
    selected_lambda = selected_record.lambda_value
    selected_lambda_log10_width = (
        float(
            np.log10(decision.selected_lambda_right)
            - np.log10(decision.selected_lambda_left)
        )
        if decision.selected_lambda_left is not None
        and decision.selected_lambda_right is not None
        and decision.selected_lambda_left > 0.0
        and decision.selected_lambda_right > 0.0
        else (None if selected_lambda is None else 0.0)
    )
    selected_kkt_value = (
        float(selected_candidate.raw_fit.fixed_objective_kkt_residual)
        if selected_is_raw
        else float("nan")
    )
    selected_kkt_residual = (
        selected_kkt_value if np.isfinite(selected_kkt_value) else None
    )

    # Reporting starts only after the typed selection decision is complete.
    search_df = _annotate_bic_diagnostics(base_search_df)
    ids = search_df["_candidate_id"].astype(int)
    search_df["signature_selection_representative"] = ids.isin(
        decision.representative_ids
    )
    if strict_positive_exact_fusion:
        search_df["bic_selection_eligible"] = ids.isin(decision.eligible_ids)
    search_df["eligible_for_selection"] = ids.isin(decision.eligible_ids)
    search_df["lambda_values_evaluated"] = ",".join(
        f"{float(value):.12g}"
        for value in _sorted_unique_lambdas(
            [
                float(record.lambda_value)
                for record in result_entries
                if record.lambda_value is not None
            ]
        )
    )
    search_df["optimizer_limited_candidate"] = ids.isin(
        decision.optimizer_limited_ids
    )
    search_df["is_selection_optimal"] = ids.isin(decision.optimal_ids)
    search_df["is_selected_best_row"] = ids.eq(int(selected_record.candidate_id))
    search_df["adaptive_search_stop_reason"] = final_adaptive_search_stop_reason
    search_df["adaptive_search_global_optimum_certified"] = bool(
        adaptive_search_global_optimum_certified
    )
    search_df["selection_optimum_resolved"] = bool(selection_optimum_resolved)
    search_df["selected_lambda_representative"] = (
        np.nan if selected_lambda is None else float(selected_lambda)
    )
    search_df["selected_lambda_left"] = (
        np.nan
        if decision.selected_lambda_left is None
        else float(decision.selected_lambda_left)
    )
    search_df["selected_lambda_right"] = (
        np.nan
        if decision.selected_lambda_right is None
        else float(decision.selected_lambda_right)
    )
    search_df["selected_lambda_interval_log10_width"] = (
        np.nan
        if selected_lambda_log10_width is None
        else float(selected_lambda_log10_width)
    )
    search_df["selection_elapsed_seconds"] = float(
        perf_counter() - selection_start_time
    )

    selected_model = SelectedModel(
        raw_reference=raw_reference,
        partition_candidate=selected_candidate,
        selected_partition_signature=str(selected_candidate.partition.signature),
        selected_candidate_family=(
            "raw_fusion" if selected_is_raw else "direct_partition"
        ),
        selected_lambda=(float(selected_lambda) if selected_is_raw else None),
        selected_partition_left_lambda=(
            decision.selected_lambda_left if selected_is_raw else None
        ),
        selected_partition_right_lambda=(
            decision.selected_lambda_right if selected_is_raw else None
        ),
        partition_parent_raw=partition_parent_raw,
    )
    search_df = search_df.drop(columns=["_candidate_id"])
    return BICSelectionResult(
        selected_model=selected_model,
        search_df=search_df,
        selection_method=selection_method,
        selection_hits_lower_boundary=decision.selection_hits_lower_boundary,
        selection_hits_upper_boundary=decision.selection_hits_upper_boundary,
        selection_boundary_unresolved=decision.selection_boundary_unresolved,
        selection_optimum_resolved=bool(selection_optimum_resolved),
        adaptive_search_stop_reason=str(final_adaptive_search_stop_reason),
        num_candidates=int(len(result_entries)),
        selected_lambda_representative=(
            None if selected_lambda is None else float(selected_lambda)
        ),
        num_candidates_certified=int(len(decision.eligible_ids)),
        selected_kkt_residual=selected_kkt_residual,
        ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
        raw_lambda_path_resolved=bool(adaptive_search_global_optimum_certified),
        global_hybrid_optimum_certified=bool(selection_optimum_resolved),
    )


def _partition_guided_admm_selection(
    *,
    data: TumorData,
    fit_options: FitOptions,
    use_warm_starts: bool,
) -> BICSelectionResult:
    """Run the certified raw path and select under one immutable contract.

    Ward/CEM supplies the primal start and initial-lambda scale. The contract
    declares whether the guide or zero-penalty pilot defines the frozen graph,
    and whether retained pilot/final-raw-phi partitions enter the secondary
    selection pool. Direct proposals are evaluated only after the raw lambda
    controller terminates, so they cannot steer or replace the raw optimizer.
    """

    selection_start_time = perf_counter()
    computation_profile = get_computation_profile(fit_options.computation_profile)
    selection_contract = get_selection_contract(fit_options.selection_contract)
    selection_score = str(fit_options.selection_score)
    normalized_score = selection_score
    profile_name = f"{computation_profile.name}_partition_guided_admm_{selection_score}"
    if selection_contract.contract_id != "raw-fusion-only-v0.3":
        profile_name += f"_{selection_contract.contract_id}"
    selection_method = "online_partition_guided_admm"
    if int(data.num_mutations) < 2:
        raise ValueError(
            "partition_guided_admm requires at least two mutations so that a "
            "positive pairwise penalty is solved by ADMM."
        )
    prepare_start_time = perf_counter()
    pilot_context = prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy=str(fit_options.dense_fallback_policy),
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.tol),
        defer_graph=True,
        inner_max_iter=max(int(fit_options.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
        device=fit_options.device,
        dtype=fit_options.dtype,
        objective_shape=str(fit_options.objective_shape),
    )
    pilot_phi: StartArray = pilot_context.exact_pilot
    pilot_runtime = pilot_context.runtime
    pilot_torch_data = torch_data_from_context(pilot_context)
    curvature_start = perf_counter()
    guide_curvature = observed_curvature_at_pilot_torch(
        data,
        pilot_phi,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        torch_data=pilot_torch_data,
        device=pilot_runtime.device,
        dtype=pilot_runtime.dtype,
    )
    guide_curvature_elapsed = float(perf_counter() - curvature_start)
    initializer_pool = generate_partition_initializer_pool(
        data=data,
        pilot_phi=pilot_phi,
        fit_options=fit_options,
        normalized_score=normalized_score,
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
        rescore_candidates=_rescore_partition_candidates,
        curvature=guide_curvature,
        curvature_elapsed_seconds=float(guide_curvature_elapsed),
    )
    guide = _best_partition_candidate(list(initializer_pool.candidates))
    if guide is None:
        raise RuntimeError(
            "No finite active-score partition initializer was available for tumor "
            f"{data.tumor_id}."
        )

    # Keep the partition guide host-backed for exact CPU behavior and fallback.
    # CUDA graph construction uploads this small M x S matrix once; the O(M^2)
    # graph itself stays device-backed and is reused by context preparation.
    guide_phi: StartArray = np.asarray(guide.phi_start)
    partition_guide_signature = _partition_signature(
        np.asarray(guide.labels, dtype=np.int64)
    )
    partition_guide_matrix_hash = _pilot_matrix_hash(guide_phi)
    if fit_options.graph is None:
        graph_pilot_source = str(selection_contract.graph_pilot_source)
        if graph_pilot_source == "partition_guide":
            graph_builder_phi = guide_phi
        elif graph_pilot_source == "zero_penalty_pilot":
            graph_builder_phi = pilot_phi
        else:
            graph_builder_phi = (
                guide_phi if computation_profile.is_strict else pilot_phi
            )
        complete_graph_degree = float(max(int(data.num_mutations) - 1, 1))
        likelihood_noise_degree_exponent = float(
            PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT
        )
        likelihood_noise_divisor = float(
            complete_graph_degree**likelihood_noise_degree_exponent
        )
        selection_graph, prebuilt_tensor_graph, likelihood_noise_tau = (
            _build_partition_guided_graph_with_resource_policy(
                guide_phi=graph_builder_phi,
                guide_curvature=guide_curvature,
                solver_context=pilot_context,
                fit_options=fit_options,
                noise_divisor=likelihood_noise_divisor,
            )
        )
        graph_source = (
            "partition_guide_likelihood_noise_degree_regularized"
            if graph_builder_phi is guide_phi
            else "zero_penalty_likelihood_pilot_degree_regularized"
        )
        graph_pilot_phi: StartArray = graph_builder_phi
    else:
        selection_graph = fit_options.graph
        prebuilt_tensor_graph = None
        likelihood_noise_tau = float("nan")
        likelihood_noise_divisor = float("nan")
        likelihood_noise_degree_exponent = float("nan")
        graph_source = "user_supplied"
        graph_pilot_phi = pilot_phi
    base_solver_context = prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy=str(fit_options.dense_fallback_policy),
        inherited_resource_fallback=pilot_context.resource_fallback,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.tol),
        # The guide initializes adaptive weights, but observed curvature and a
        # mild degree correction set a finite data-derived distance floor. This
        # prevents the fixed 1e-6 floor from making the proposed blocks
        # effectively immutable while retaining the current estimator as the
        # requested initializer.
        graph=selection_graph,
        prebuilt_tensor_graph=prebuilt_tensor_graph,
        inner_max_iter=max(int(fit_options.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
        # Preserve the independent likelihood starts.  The previous flow
        # replaced both with the Ward guide, so nominal "cold" retries were
        # merely duplicates of the same non-convex basin.
        exact_pilot=pilot_context.exact_pilot,
        pooled_start=pilot_context.pooled_start,
        scalar_well_starts=pilot_context.scalar_well_starts,
        device=fit_options.device,
        dtype=fit_options.dtype,
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
        objective_shape=str(fit_options.objective_shape),
    )
    effective_graph = base_solver_context.graph_spec
    effective_tensor_graph = base_solver_context.graph
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm requires the complete pairwise graph so the "
            "inner solver is ADMM."
        )
    effective_fit_options = replace(fit_options, graph=effective_graph)
    raw_guide_labels = np.asarray(guide.labels, dtype=np.int64)
    raw_guide_phi: StartArray = guide_phi
    guided_initialization, base_solver_context, raw_guide_phi = (
        _build_guided_initialization_with_resource_policy(
            data=data,
            guide_phi=raw_guide_phi,
            guide_labels=raw_guide_labels,
            solver_context=base_solver_context,
            fit_options=effective_fit_options,
        )
    )
    guided_initialization = replace(
        guided_initialization,
        solver_state=_offload_solver_state_to_cpu(guided_initialization.solver_state),
    )
    runtime = base_solver_context.runtime
    torch_data = torch_data_from_context(base_solver_context)
    effective_graph = base_solver_context.graph_spec
    effective_tensor_graph = base_solver_context.graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm CPU fallback changed the complete fusion graph."
        )
    prepare_elapsed_seconds = float(perf_counter() - prepare_start_time)

    controller = OnlineLambdaController(
        initial_lambda=float(guided_initialization.lambda_value),
        initial_reason="partition_guide_kkt_balance",
        config=OnlineLambdaConfig(
            guide_n_clusters=int(np.unique(raw_guide_labels).size),
            num_mutations=int(data.num_mutations),
            kkt_tolerance=5.0 * float(effective_fit_options.tol),
            max_unique_lambdas=int(computation_profile.lambda_budget),
            max_refinement_lambdas=int(computation_profile.lambda_refinement_budget),
            max_solver_retries_per_lambda=int(computation_profile.solver_retry_limit),
            partition_event_mode=True,
        ),
    )

    result_entries: list[CandidateRecord] = []
    fit_by_lambda: dict[float, FitResult] = {}
    partition_k_by_lambda: dict[float, int] = {}
    attempts_by_lambda: dict[float, list[FitResult]] = {}
    bic_refit_cache: dict[object, PartitionRefitCacheEntry] = {}
    static_metadata = _candidate_static_metadata(
        data, effective_graph, pilot_phi=graph_pilot_phi
    )
    scalar_likelihood_pilot_hash = _pilot_matrix_hash(pilot_phi)
    next_step = 0
    while True:
        proposal = controller.propose()
        if proposal is None:
            break
        lambda_key = _canonical_lambda(proposal.lambda_value)
        for attempt_key in list(attempts_by_lambda):
            if float(attempt_key) != float(lambda_key):
                del attempts_by_lambda[attempt_key]
        candidate_fit_options = effective_fit_options
        if proposal.phase in {
            "solver_recovery",
            "bootstrap_certification_anchor",
        }:
            candidate_fit_options = _solver_recovery_fit_options(
                data,
                effective_fit_options,
                retry_number=int(proposal.retry_number),
            )
        elif proposal.retry_number > 0:
            effort_factor = int(proposal.retry_number) + 1
            candidate_fit_options = replace(
                effective_fit_options,
                outer_max_iter=max(
                    int(effective_fit_options.outer_max_iter) * effort_factor,
                    int(effective_fit_options.outer_max_iter),
                ),
                inner_max_iter=max(
                    int(effective_fit_options.inner_max_iter) * effort_factor,
                    int(effective_fit_options.inner_max_iter),
                ),
                # Retry effort changes iteration budgets, not the model's
                # numerical admission contract.
                tol=float(effective_fit_options.tol),
            )

        raw_fit_start = perf_counter()
        raw_start_attempt_count = 0
        raw_start_sources: list[str] = []

        def solve_raw_path() -> tuple[FitResult, _RawStartAttempt, dict[str, object]]:
            nonlocal raw_start_attempt_count
            context = base_solver_context
            initialization = guided_initialization
            warm_fit = None
            if proposal.warm_start_lambda is not None:
                warm_fit = fit_by_lambda.get(
                    _canonical_lambda(proposal.warm_start_lambda)
                )
            alternate_fit = None
            if proposal.alternate_start_lambda is not None:
                alternate_fit = fit_by_lambda.get(
                    _canonical_lambda(proposal.alternate_start_lambda)
                )
            same_lambda_attempts = attempts_by_lambda.get(lambda_key, [])
            finite_failed = [
                attempt
                for attempt in same_lambda_attempts
                if attempt.solver_state is not None
                and np.isfinite(float(attempt.fixed_objective_kkt_residual))
            ]
            start_specs: list[_RawStartSpec] = []
            seen_start_states: set[tuple[str, int | str]] = set()

            def append_distinct_start(
                source: str,
                start_value: float,
                state: SolverState | None,
                phi: StartArray | None = None,
            ) -> None:
                # Historical endpoint caches can refer to the exact same state
                # object (for example across a flat partition plateau).  Do
                # not pay for duplicate solves, while retaining states with
                # distinct dual/certificate histories even when their primal
                # matrices happen to match.
                identity: tuple[str, int | str]
                if state is None:
                    if phi is None:
                        raise ValueError("A cold raw start requires an explicit Phi.")
                    identity = ("cold", _pilot_matrix_hash(phi))
                else:
                    identity = ("state", id(state))
                if identity in seen_start_states:
                    return
                seen_start_states.add(identity)
                start_specs.append((str(source), float(start_value), state, phi))

            if proposal.phase == "solver_recovery":
                if finite_failed:
                    best_failed_fit = min(
                        finite_failed,
                        key=lambda attempt: float(attempt.fixed_objective_kkt_residual),
                    )
                    append_distinct_start(
                        "best_same_lambda_kkt_state",
                        float(best_failed_fit.lambda_value),
                        best_failed_fit.solver_state,
                    )
                else:
                    append_distinct_start(
                        "guided_kkt_solver_recovery",
                        float(initialization.lambda_value),
                        initialization.solver_state,
                    )
            elif int(proposal.retry_number) > 0:
                if (
                    use_warm_starts
                    and int(proposal.retry_number) == 1
                    and alternate_fit is not None
                    and alternate_fit.solver_state is not None
                ):
                    append_distinct_start(
                        "alternate_bracket_endpoint",
                        float(proposal.alternate_start_lambda),
                        alternate_fit.solver_state,
                    )
                elif (
                    use_warm_starts
                    and warm_fit is not None
                    and warm_fit.solver_state is not None
                ):
                    append_distinct_start(
                        "same_lambda_retry",
                        float(proposal.warm_start_lambda),
                        warm_fit.solver_state,
                    )
                else:
                    append_distinct_start(
                        "guided_kkt_fallback",
                        float(initialization.lambda_value),
                        initialization.solver_state,
                    )
            else:
                if proposal.phase == "bootstrap_certification_anchor":
                    append_distinct_start(
                        "guided_kkt_bootstrap_anchor",
                        float(initialization.lambda_value),
                        initialization.solver_state,
                    )
                    for (
                        source,
                        start_value,
                        state,
                        phi,
                    ) in _bootstrap_independent_start_specs(
                        initial_lambda=float(initialization.lambda_value),
                        raw_guide_phi=raw_guide_phi,
                        exact_pilot=context.exact_pilot,
                        pooled_start=context.pooled_start,
                        suffix="bootstrap_anchor",
                    ):
                        append_distinct_start(source, start_value, state, phi)
                # Partition-event midpoints compete both bracket endpoints
                # with the fixed guided/cold starts. Applying this bounded bank
                # only at statistical event probes prevents a poor stationary
                # basin from steering the event while preserving the fast
                # one-start continuation path for coarse outward exploration.
                if (
                    proposal.phase != "bootstrap_certification_anchor"
                    and use_warm_starts
                    and warm_fit is not None
                    and warm_fit.solver_state is not None
                ):
                    append_distinct_start(
                        "warm_bracket_left"
                        if proposal.phase == "refine_partition_event"
                        else "warm_endpoint",
                        float(proposal.warm_start_lambda),
                        warm_fit.solver_state,
                    )
                if (
                    proposal.phase == "refine_partition_event"
                    and use_warm_starts
                    and alternate_fit is not None
                    and alternate_fit.solver_state is not None
                ):
                    append_distinct_start(
                        "warm_bracket_right",
                        float(proposal.alternate_start_lambda),
                        alternate_fit.solver_state,
                    )
                if proposal.phase == "refine_partition_event":
                    append_distinct_start(
                        "guided_kkt_multistart",
                        float(initialization.lambda_value),
                        initialization.solver_state,
                    )
                    append_distinct_start(
                        "cold_partition_guide",
                        float(initialization.lambda_value),
                        None,
                        raw_guide_phi,
                    )
                    append_distinct_start(
                        "cold_zero_penalty_pilot",
                        0.0,
                        None,
                        context.exact_pilot,
                    )
                    append_distinct_start(
                        "cold_pooled_likelihood",
                        0.0,
                        None,
                        context.pooled_start,
                    )
                elif not start_specs:
                    append_distinct_start(
                        "guided_kkt_state"
                        if proposal.phase == "initial"
                        else "guided_kkt_fallback",
                        float(initialization.lambda_value),
                        initialization.solver_state,
                    )

                # A K=1 warm endpoint can trap all subsequent lower-lambda
                # continuation probes in the pooled basin.  At that one
                # structural transition, compete the genuinely independent
                # guide/zero-penalty/pooled primals before steering the path.
                warm_key = (
                    None
                    if proposal.warm_start_lambda is None
                    else _canonical_lambda(proposal.warm_start_lambda)
                )
                escaping_k1_basin = bool(
                    int(proposal.retry_number) == 0
                    and proposal.phase != "refine_partition_event"
                    and warm_key is not None
                    and partition_k_by_lambda.get(warm_key) == 1
                    and float(proposal.lambda_value) < float(proposal.warm_start_lambda)
                )
                if escaping_k1_basin:
                    append_distinct_start(
                        "cold_partition_guide_k1_escape",
                        float(initialization.lambda_value),
                        None,
                        raw_guide_phi,
                    )
                    append_distinct_start(
                        "cold_zero_penalty_k1_escape",
                        0.0,
                        None,
                        context.exact_pilot,
                    )
                    append_distinct_start(
                        "cold_pooled_likelihood_k1_escape",
                        0.0,
                        None,
                        context.pooled_start,
                    )

            start_attempts: list[_RawStartAttempt] = []
            for (
                lambda_start_source,
                lambda_start_value,
                original_state,
                explicit_phi_start,
            ) in start_specs:
                solver_state_start, changed_count = _escape_path_breakpoint_retry_state(
                    original_state,
                    start_source=lambda_start_source,
                    start_lambda=lambda_start_value,
                    target_lambda=float(proposal.lambda_value),
                    context=context,
                    tol=float(candidate_fit_options.tol),
                )
                phi_start = _clone_start(
                    solver_state_start.phi
                    if solver_state_start is not None
                    and solver_state_start.phi is not None
                    else (
                        explicit_phi_start
                        if explicit_phi_start is not None
                        else raw_guide_phi
                    )
                )
                seed_fit = fit_fixed_objective(
                    data=data,
                    options=replace(
                        candidate_fit_options,
                        lambda_value=float(proposal.lambda_value),
                    ),
                    phi_start=phi_start,
                    exact_pilot=context.exact_pilot,
                    pooled_start=context.pooled_start,
                    scalar_well_starts=context.scalar_well_starts,
                    start_mode="warm_only",
                    runtime=context.runtime,
                    torch_data=torch_data_from_context(context),
                    solver_context=context,
                    solver_state=solver_state_start,
                    compute_summary=False,
                )
                if str(seed_fit.objective_spec_hash) != str(
                    context.objective_spec_hash
                ):
                    raise AssertionError(
                        "Raw multistart changed the fixed objective identity."
                    )
                if seed_fit.solver_state is not None:
                    seed_fit.solver_state = _offload_solver_state_to_cpu(
                        seed_fit.solver_state
                    )
                attempts_by_lambda.setdefault(lambda_key, []).append(seed_fit)
                mathematically_certified = bool(
                    float(seed_fit.lambda_value) > 0.0
                    and seed_fit.objective_faithful
                    and seed_fit.full_kkt_certified
                    and seed_fit.selection_eligible
                )
                start_attempts.append(
                    _RawStartAttempt(
                        fit=seed_fit,
                        source=str(lambda_start_source),
                        start_value=float(lambda_start_value),
                        breakpoint_escape_changed_count=int(changed_count),
                        mathematically_certified=bool(mathematically_certified),
                    )
                )
                raw_start_attempt_count += 1
                raw_start_sources.append(str(lambda_start_source))

            selected_attempt = _select_raw_start_attempt(start_attempts)
            seed_fit = selected_attempt.fit
            certified_objectives = [
                float(attempt.fit.penalized_objective)
                for attempt in start_attempts
                if attempt.mathematically_certified
                and np.isfinite(float(attempt.fit.penalized_objective))
            ]
            certified_min = (
                min(certified_objectives) if certified_objectives else float("nan")
            )
            certified_max = (
                max(certified_objectives) if certified_objectives else float("nan")
            )
            raw_start_diagnostics: dict[str, object] = {
                "attempt_count": int(len(start_attempts)),
                "certified_count": int(len(certified_objectives)),
                "certified_objective_min": float(certified_min),
                "certified_objective_max": float(certified_max),
                "certified_objective_spread": (
                    float(certified_max - certified_min)
                    if certified_objectives
                    else float("nan")
                ),
                "selected_source": str(selected_attempt.source),
                "attempt_objectives": ",".join(
                    f"{attempt.source}:{float(attempt.fit.penalized_objective):.17g}:"
                    f"{int(attempt.mathematically_certified)}"
                    for attempt in start_attempts
                ),
                "attempt_diagnostics": tuple(
                    _raw_start_attempt_diagnostic(
                        attempt,
                        fit_options=candidate_fit_options,
                    )
                    for attempt in start_attempts
                ),
            }
            # Subsequent bracket proposals must warm-start from the same raw
            # basin that was admitted to partition scoring, never from a lower
            # objective but mathematically uncertified side attempt.
            return seed_fit, selected_attempt, raw_start_diagnostics

        selected_raw_fit, selected_start, raw_start_diagnostics = solve_raw_path()
        raw_fit_elapsed_seconds = float(perf_counter() - raw_fit_start)
        fit, row, artifact = evaluate_raw_fusion_candidate(
            data=data,
            fit_options=effective_fit_options,
            candidate_fit_options=candidate_fit_options,
            phi_start=None,
            exact_pilot=base_solver_context.exact_pilot,
            pooled_start=base_solver_context.pooled_start,
            scalar_well_starts=base_solver_context.scalar_well_starts,
            start_mode="warm_only",
            runtime=runtime,
            torch_data=torch_data,
            solver_context=base_solver_context,
            solver_state=selected_raw_fit.solver_state,
            compute_summary=False,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=next_step,
            lambda_value=float(proposal.lambda_value),
            selection_score=selection_score,
            bic_refit_cache=bic_refit_cache,
            static_metadata=static_metadata,
            precomputed_fit=selected_raw_fit,
        )
        (
            lambda_start_source,
            lambda_start_value,
            path_breakpoint_escape_changed_count,
        ) = (
            str(selected_start.source),
            float(selected_start.start_value),
            int(selected_start.breakpoint_escape_changed_count),
        )
        row["raw_fit_elapsed_seconds"] = float(raw_fit_elapsed_seconds)
        row["raw_mean_start_fit_elapsed_seconds"] = float(
            raw_fit_elapsed_seconds / max(raw_start_attempt_count, 1)
        )
        row["raw_start_attempt_count"] = int(raw_start_attempt_count)
        row["raw_start_attempt_sources"] = ",".join(raw_start_sources)
        row["raw_multistart_applied"] = bool(raw_start_attempt_count > 1)
        row["raw_selected_start_attempt_count"] = int(
            raw_start_diagnostics["attempt_count"]
        )
        row["raw_selected_start_certified_count"] = int(
            raw_start_diagnostics["certified_count"]
        )
        row["raw_selected_start_certified_objective_min"] = float(
            raw_start_diagnostics["certified_objective_min"]
        )
        row["raw_selected_start_certified_objective_max"] = float(
            raw_start_diagnostics["certified_objective_max"]
        )
        row["raw_selected_start_certified_objective_spread"] = float(
            raw_start_diagnostics["certified_objective_spread"]
        )
        row["raw_selected_start_source"] = str(raw_start_diagnostics["selected_source"])
        row["raw_start_objectives"] = str(raw_start_diagnostics["attempt_objectives"])
        row["raw_start_attempt_diagnostics"] = raw_start_diagnostics[
            "attempt_diagnostics"
        ]
        guide_signature = _partition_signature(raw_guide_labels)
        guide_matrix_hash = _pilot_matrix_hash(raw_guide_phi)

        row.update(
            {
                "search_round": int(next_step),
                "search_phase": str(proposal.phase),
                "lambda_source": "online_partition_guide_kkt",
                "lambda_search_mode": "partition_guided_admm",
                "lambda_path_prespecified": False,
                "lambda_proposal_reason": str(proposal.reason),
                "lambda_retry_number": int(proposal.retry_number),
                "lambda_start_source": str(lambda_start_source),
                "lambda_start_value": float(lambda_start_value),
                "path_breakpoint_escape_applied": bool(
                    path_breakpoint_escape_changed_count > 0
                ),
                "path_breakpoint_escape_changed_count": int(
                    path_breakpoint_escape_changed_count
                ),
                "persistent_solver_state_device": "cpu",
                "lambda_warm_start_value": np.nan
                if proposal.warm_start_lambda is None
                else float(proposal.warm_start_lambda),
                "lambda_alternate_start_value": np.nan
                if proposal.alternate_start_lambda is None
                else float(proposal.alternate_start_lambda),
                "lambda_observed_bracket_left": np.nan
                if proposal.bracket_left_lambda is None
                else float(proposal.bracket_left_lambda),
                "lambda_observed_bracket_right": np.nan
                if proposal.bracket_right_lambda is None
                else float(proposal.bracket_right_lambda),
                "candidate_role": "pairwise_fusion_selection",
                "initialization_mode": "ward_cem_active_selection_score_kkt",
                "initializer_selection_score": str(normalized_score),
                "initializer_score_value": float(guide.bic),
                "initializer_K": int(np.unique(raw_guide_labels).size),
                "initializer_requested_K": int(_partition_candidate_requested_k(guide)),
                "initializer_source": str(guide.source),
                "initializer_partition_signature": str(guide_signature),
                "initializer_matrix_hash": str(guide_matrix_hash),
                "partition_guide_K": int(guide.K),
                "partition_guide_signature": str(partition_guide_signature),
                "partition_guide_matrix_hash": str(partition_guide_matrix_hash),
                "fusion_graph_source": str(graph_source),
                "fusion_graph_pilot_matrix_hash": str(
                    static_metadata.pilot_matrix_hash
                ),
                "scalar_likelihood_pilot_matrix_hash": str(
                    scalar_likelihood_pilot_hash
                ),
                "fusion_graph_likelihood_noise_tau": float(likelihood_noise_tau),
                "fusion_graph_likelihood_noise_divisor": float(
                    likelihood_noise_divisor
                ),
                "fusion_graph_likelihood_noise_degree_exponent": float(
                    likelihood_noise_degree_exponent
                ),
                "initializer_pool_size": int(len(initializer_pool.candidates)),
                "initializer_lambda": float(guided_initialization.lambda_value),
                "initializer_kkt_residual": float(
                    guided_initialization.diagnostics.kkt_residual
                ),
                "initializer_max_dual_ball_ratio": float(
                    guided_initialization.diagnostics.max_dual_ball_ratio
                ),
                "initializer_capacity_iterations": int(
                    guided_initialization.diagnostics.capacity_iterations
                ),
                "initializer_capacity_converged": bool(
                    guided_initialization.diagnostics.capacity_converged
                ),
                "initializer_capacity_status": str(
                    guided_initialization.diagnostics.capacity_status
                ),
                "selection_prepare_elapsed_seconds": float(prepare_elapsed_seconds),
                "adaptive_candidate_budget": int(computation_profile.lambda_budget),
                "adaptive_refinement_candidate_budget": int(
                    computation_profile.lambda_refinement_budget
                ),
                "adaptive_max_rounds": int(
                    computation_profile.lambda_budget
                    + computation_profile.lambda_refinement_budget
                ),
                "adaptive_refine_per_round": 1,
                "adaptive_transition_probe_max_candidates": 0,
                "adaptive_initial_lambda_count": 1,
                "likelihood_partition_pool_enabled": True,
                "likelihood_partition_selection_enabled": bool(
                    selection_contract.selectable_partition_pool
                ),
                **_partition_pool_row_metadata(initializer_pool),
            }
        )
        candidate_id = int(len(result_entries))
        row["_candidate_id"] = candidate_id
        row["candidate_id"] = candidate_id
        result_entries.append(
            CandidateRecord(
                candidate_id=candidate_id,
                candidate=artifact,
                row=row,
            )
        )
        incumbent = fit_by_lambda.get(lambda_key)
        if _prefer_fit_candidate(fit, incumbent):
            fit_by_lambda[lambda_key] = fit
            partition_k_by_lambda[lambda_key] = int(row["n_clusters"])

        raw_exact_certified = bool(
            raw_candidate_has_exact_fusion_certificate(artifact)
            and bool(effective_tensor_graph.is_complete)
        )
        selection_score_available = bool(artifact.eligible_for_selection)
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=float(proposal.lambda_value),
                n_clusters=int(row["n_clusters"]),
                partition_signature=str(row["partition_signature"]),
                # The active selection score steers the online-lambda
                # controller (the observation field name is historical).
                partition_icl=(
                    float(artifact.score.value)
                    if selection_score_available
                    else float("inf")
                ),
                kkt_residual=float(row["fixed_objective_kkt_residual"]),
                raw_objective_certified=bool(raw_exact_certified),
                partition_certified=bool(artifact.partition.certified),
                selection_score_available=selection_score_available,
                score_numerical_uncertainty=float(artifact.score.numerical_uncertainty),
                degrees_of_freedom=int(artifact.score.degrees_of_freedom),
            )
        )
        next_step += 1

    ward_candidate_pool_complete = False
    if selection_contract.selectable_partition_pool:
        direct_proposals: list[
            tuple[
                PartitionCandidate,
                str,
                CandidateRecord | None,
                PartitionInitializerPool,
            ]
        ] = [
            (proposal, "pilot", None, initializer_pool)
            for proposal in initializer_pool.candidates
        ]
        config = selection_contract.partition_config
        if config.include_final_phi_ladder and config.final_phi_ladder_kmax > 0:
            raw_parent_records = sorted(
                (
                    record
                    for record in result_entries
                    if isinstance(record.candidate, RawFusionCandidate)
                    and raw_candidate_has_exact_fusion_certificate(
                        record.candidate
                    )
                ),
                key=lambda record: (
                    float(record.score.value),
                    float(record.score.numerical_uncertainty),
                    float(record.candidate.raw_fit.lambda_value),
                    int(record.candidate_id),
                ),
            )[: int(config.final_phi_parent_count)]
            final_k_grid = tuple(
                range(
                    1,
                    min(
                        int(config.final_phi_ladder_kmax),
                        int(data.num_mutations),
                    )
                    + 1,
                )
            )
            for parent_record in raw_parent_records:
                parent = parent_record.candidate
                if not isinstance(parent, RawFusionCandidate):  # pragma: no cover
                    continue
                final_pool = generate_partition_initializer_pool(
                    data=data,
                    pilot_phi=np.asarray(parent.raw_fit.phi, dtype=np.float64),
                    fit_options=effective_fit_options,
                    normalized_score=normalized_score,
                    runtime=runtime,
                    torch_data=torch_data,
                    rescore_candidates=_rescore_partition_candidates,
                    declared_k_grid=final_k_grid,
                    enable_refinement=False,
                )
                direct_proposals.extend(
                    (proposal, "final_phi", parent_record, final_pool)
                    for proposal in final_pool.candidates
                )

        for proposal, stage, parent_record, proposal_pool in direct_proposals:
            parent_candidate = (
                None if parent_record is None else parent_record.candidate
            )
            parent_raw = (
                parent_candidate
                if isinstance(parent_candidate, RawFusionCandidate)
                else None
            )
            source = _direct_partition_source(proposal, stage=stage)
            candidate_id = int(len(result_entries))
            direct_row, direct_candidate = evaluate_direct_partition_candidate(
                data=data,
                proposal=proposal,
                selection_options=effective_fit_options,
                candidate_id=candidate_id,
                source=source,
                parent_raw_candidate_id=(
                    None if parent_record is None else int(parent_record.candidate_id)
                ),
                parent_raw_lambda=(
                    None
                    if parent_raw is None
                    else float(parent_raw.raw_fit.lambda_value)
                ),
                parent_raw_phi_hash=(
                    ""
                    if parent_raw is None
                    else _pilot_matrix_hash(parent_raw.raw_fit.phi)
                ),
                generation_contract_id=selection_contract.contract_id,
                refit_cache=bic_refit_cache,
            )
            direct_row.update(
                {
                    "selection_method": selection_method,
                    "selection_profile": profile_name,
                    "search_round": int(next_step),
                    "search_phase": f"{stage}_direct_partition_pool",
                    "lambda_search_mode": "partition_guided_admm",
                    "candidate_role": "secondary_partition_selection",
                    "likelihood_partition_pool_enabled": True,
                    "likelihood_partition_selection_enabled": True,
                    "ward_candidate_pool_complete": True,
                    **_partition_pool_row_metadata(proposal_pool),
                }
            )
            result_entries.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    candidate=direct_candidate,
                    row=direct_row,
                )
            )
            next_step += 1
        ward_candidate_pool_complete = True

    if not result_entries:
        raise RuntimeError(
            f"No guided ADMM candidates were evaluated for tumor {data.tumor_id}."
        )
    stop_reason = str(controller.stop_reason or "online_lambda_no_terminal_reason")
    return _assemble_selection_result(
        data=data,
        normalized_score=normalized_score,
        result_entries=result_entries,
        selection_method=selection_method,
        adaptive_search_stop_reason=stop_reason,
        selection_start_time=selection_start_time,
        strict_positive_exact_fusion=not bool(
            selection_contract.selectable_partition_pool
        ),
        ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
    )


def select_model(
    *,
    data: TumorData,
    fit_options: FitOptions,
    use_warm_starts: bool,
) -> BICSelectionResult:
    effective_objective_shape = objective_shape_for_data(
        data, str(fit_options.objective_shape)
    )
    if effective_objective_shape != str(fit_options.objective_shape):
        fit_options = replace(
            fit_options,
            objective_shape=effective_objective_shape,
        )

    return _partition_guided_admm_selection(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
    )


__all__ = [
    "BICSelectionResult",
    "NoCertifiedRawReferenceError",
    "NoEligibleModelSelectionCandidatesError",
    "candidates_to_dataframe",
    "select_model",
]
