from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import Counter
import numpy as np
import torch

from ..config import _FitOptions, FINAL_PHI_LADDER_KMAX, FINAL_PHI_PARENT_COUNT
from ..io.validation import validate_public_tumor_data
from ..core.fusion.graph import build_complete_uniform_graph
from ..core.fusion.partition_starts import (
    PartitionCandidate,
    generate_partition_initializer_pool,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.solver import (
    fit_prepared,
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
    promote_solver_context_dtype,
    transfer_scalar_pilot_certificates,
)
from ..core.fusion.types import (
    ConvergenceResult,
    RawFit,
    PreparedProblem,
    SolverState,
)
from ..io.data import TumorData

from ..model_selection.candidates import (
    PartitionRefitCacheEntry,
    evaluate_direct_partition_candidate,
    evaluate_raw_fusion_candidate,
    validate_candidate_identity,
)
from ..config import (
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
)
from ..model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
)
from ..model_selection.partitions import (
    _best_partition_candidate,
)
from ..model_selection.proposals import (
    RawStartAttempt as _RawStartAttempt,
    RawStartSpec as _RawStartSpec,
    adaptive_stop_certifies_global_optimum as _adaptive_stop_certifies_global_optimum,
    bootstrap_independent_start_specs as _bootstrap_independent_start_specs,
    build_guided_initialization_with_resource_policy as _build_guided_initialization_with_resource_policy,
    build_partition_guided_graph_with_resource_policy as _build_partition_guided_graph_with_resource_policy,
    clone_start as _clone_start,
    direct_partition_source as _direct_partition_source,
    escape_emission_breakpoint_retry_state as _escape_emission_breakpoint_retry_state,
    explicit_path_default_start_specs as _explicit_path_default_start_specs,
    offload_raw_fit_to_cpu as _offload_raw_fit_to_cpu,
    pilot_matrix_hash as _pilot_matrix_hash,
    select_raw_start_attempt as _select_raw_start_attempt,
    solver_retry_fit_options,
)
from ..model_selection.scoring import (
    _canonical_lambda,
    _prefer_fit_candidate,
    raw_candidate_has_exact_fusion_certificate,
    select_candidate_records,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateTrace,
    CandidateRecord,
    RawAttemptTrace,
    RawFusionCandidate,
    SearchCandidate,
    SelectedModel,
    StartArray,
)


class NoEligibleModelSelectionCandidatesError(RuntimeError):
    """Model selection failed after every typed candidate was rejected."""

    def __init__(self, tumor_id: str, candidates: tuple[CandidateRecord, ...]) -> None:
        self.tumor_id = str(tumor_id)
        self.candidates = tuple(candidates)
        super().__init__(
            f"No candidates were eligible for model selection for tumor "
            f"{self.tumor_id}."
        )


@dataclass(frozen=True, slots=True)
class RawReferenceFailureDiagnostics:
    raw_candidate_count: int
    raw_solver_attempt_count: int
    raw_certified_count: int
    partition_certified_count: int
    direct_candidate_count: int
    direct_eligible_count: int
    min_kkt_residual: float
    min_kkt_tolerance: float
    best_raw_attempt: RawAttemptTrace | None
    dominant_kkt_component: str
    mm_violation_min: int
    mm_violating_count: int
    ineligibility_reason_counts: tuple[tuple[str, int], ...]
    certificate_status_counts: tuple[tuple[str, int], ...]
    search_phase_counts: tuple[tuple[str, int], ...]
    start_source_counts: tuple[tuple[str, int], ...]
    promotion_status_counts: tuple[tuple[str, int], ...] = ()
    attempt_summaries: tuple[str, ...] = ()


class NoCertifiedRawReferenceError(RuntimeError):
    """Hybrid selection found direct candidates but no certified raw reference."""

    def __init__(
        self,
        *,
        tumor_id: str,
        records: list[CandidateRecord],
        adaptive_search_stop_reason: str,
    ) -> None:
        self.tumor_id = str(tumor_id)
        self.candidates = tuple(records)
        self.adaptive_search_stop_reason = str(adaptive_search_stop_reason)
        self.diagnostics = _raw_reference_failure_diagnostics(records)
        reason_counts = dict(self.diagnostics.ineligibility_reason_counts)
        certificate_counts = dict(self.diagnostics.certificate_status_counts)
        promotion_counts = dict(self.diagnostics.promotion_status_counts)
        best = self.diagnostics.best_raw_attempt
        best_summary = (
            None
            if best is None
            else {
                "phase": best.search_phase,
                "source": best.source,
                "lambda": best.lambda_value,
                "dtype": f"{best.working_dtype}/{best.audit_dtype}",
                "promotion": best.promotion_status,
                "polished": best.precision_polished,
                "stage_outer": (
                    f"{best.convergence.stage_outer_iterations}/"
                    f"{best.convergence.stage_outer_max_iter}"
                ),
                "stage_inner": best.convergence.stage_inner_iterations,
                "stage_inner_max": best.convergence.stage_inner_max_iter,
                "stage_inner_calls": best.convergence.stage_inner_solve_calls,
                "stop": best.convergence.stop_reason,
                "progress": best.convergence.progress_residual_method,
                "solve_tol": best.convergence.solve_tolerance,
                "legacy_kkt": best.convergence.legacy_stop_kkt_residual,
                "componentwise_kkt": best.convergence.componentwise_stop_kkt_residual,
                "steps": (
                    f"{best.convergence.accepted_full_steps}/"
                    f"{best.convergence.accepted_damped_steps}/"
                    f"{best.convergence.rejected_outer_steps}"
                ),
                "fallback": best.fallback_reason,
            }
        )
        super().__init__(
            "Hybrid selection requires a certified raw-fusion reference for "
            f"tumor {self.tumor_id}; "
            f"stop={self.adaptive_search_stop_reason}; "
            f"raw_candidates={self.diagnostics.raw_candidate_count}; "
            f"raw_solver_attempts={self.diagnostics.raw_solver_attempt_count}; "
            f"raw_certified={self.diagnostics.raw_certified_count}; "
            f"partition_certified={self.diagnostics.partition_certified_count}; "
            f"min_kkt_residual={self.diagnostics.min_kkt_residual}; "
            f"min_kkt_tolerance={self.diagnostics.min_kkt_tolerance}; "
            f"dominant_kkt_component={self.diagnostics.dominant_kkt_component}; "
            f"mm_violating={self.diagnostics.mm_violating_count}; "
            f"ineligibility_reasons={reason_counts}; "
            f"certificate_statuses={certificate_counts}; "
            f"promotion_statuses={promotion_counts}; "
            f"best_attempt={best_summary}; "
            f"attempts={self.diagnostics.attempt_summaries}."
        )


def _stable_counts(values) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(Counter(
        "<missing>" if value is None else str(value) for value in values
    ).items()))


def _try_promote_recovery_context(
    context: PreparedProblem,
) -> tuple[PreparedProblem, str]:
    """Promote one frozen recovery context and retain a typed outcome."""

    if context.runtime.dtype == torch.float64:
        return context, "not_needed"
    try:
        promoted = promote_solver_context_dtype(context, dtype=torch.float64)
    except (MemoryError, torch.OutOfMemoryError):
        return context, "failed_memory"
    except RuntimeError:
        return context, "failed_runtime"
    return promoted, "applied"


def _raw_attempt_trace(
    attempt: _RawStartAttempt, *, search_round: int, search_phase: str,
    outer_max_iter: int, inner_max_iter: int, certificate_max_iter: int,
) -> RawAttemptTrace:
    """Project a completed typed solve once, dropping all numerical ownership."""
    fit = attempt.fit
    if not isinstance(fit, RawFit):
        raise TypeError("Raw attempt diagnostics require a current typed RawFit.")
    certificate = fit.certificate
    convergence = fit.convergence
    if not isinstance(convergence, ConvergenceResult):
        raise TypeError("Raw attempt diagnostics require a current typed ConvergenceResult.")
    components = {
        "stationarity": float(certificate.components.stationarity),
        "edge_subgradient": float(certificate.components.edge_subgradient),
        "dual_ball": float(certificate.components.dual_ball),
        "box": float(certificate.components.box),
    }
    finite_components = {
        name: value for name, value in components.items() if np.isfinite(value)
    }
    return RawAttemptTrace(
        search_round=int(search_round),
        search_phase=str(search_phase),
        lambda_value=float(fit.provenance.lambda_value),
        source=str(attempt.source),
        start_value=float(attempt.start_value),
        breakpoint_escape_changed_count=int(attempt.breakpoint_escape_changed_count),
        mathematically_certified=bool(attempt.mathematically_certified),
        objective=float(fit.objective.total),
        kkt_components=certificate.components,
        kkt_tolerance=float(certificate.tolerance),
        dominant_kkt_component=(
            max(finite_components, key=finite_components.get)
            if finite_components
            else "unknown"
        ),
        certificate_status=str(certificate.status),
        certificate_certified=bool(certificate.certified),
        certificate_admissible=bool(certificate.admissible),
        convergence=convergence,
        work=fit.work,
        outer_max_iter=int(outer_max_iter),
        inner_max_iter=int(inner_max_iter),
        certificate_max_iter=int(certificate_max_iter),
        working_dtype=str(certificate.working_dtype),
        audit_dtype=str(certificate.audit_dtype),
        precision_polished=bool(certificate.precision_polished),
        promotion_status=str(attempt.promotion_status),
        fallback_reason=str(certificate.fallback_reason),
    )


def _compact_raw_attempt_summary(item: RawAttemptTrace) -> str:
    """Render one scalar-only attempt token suitable for scheduler stderr."""

    convergence = item.convergence
    steps = (
        f"{convergence.accepted_full_steps}/{convergence.accepted_damped_steps}/"
        f"{convergence.rejected_outer_steps}"
    )
    return (
        f"r{item.search_round}:{item.search_phase}:{item.source}@{item.lambda_value:.6g}"
        f"|kkt={item.kkt_residual:.6g}/{item.kkt_tolerance:.6g}"
        f"|dtype={item.working_dtype}/{item.audit_dtype}"
        f"|prom={item.promotion_status}|polish={int(item.precision_polished)}"
        f"|progress={convergence.progress_residual_method}"
        f"|solve_tol={convergence.solve_tolerance:.6g}"
        f"|outer={convergence.stage_outer_iterations}/{convergence.stage_outer_max_iter}"
        f"|inner={convergence.stage_inner_iterations}/"
        f"{convergence.stage_inner_solve_calls}x{convergence.stage_inner_max_iter}"
        f"|stop={convergence.stop_reason}"
        f"|stop_kkt={convergence.legacy_stop_kkt_residual:.6g}/"
        f"{convergence.componentwise_stop_kkt_residual:.6g}"
        f"|steps={steps}"
        f"|fallback={item.fallback_reason or 'none'}"
    )


def _raw_attempts(record: CandidateRecord) -> tuple[RawAttemptTrace, ...]:
    if record.trace.raw_attempts:
        return record.trace.raw_attempts
    candidate = record.candidate
    if not isinstance(candidate, RawFusionCandidate):
        return ()
    return (
        _raw_attempt_trace(_RawStartAttempt(
            fit=candidate.raw_fit,
            source=str(record.trace.start_source),
            start_value=(
                float("nan")
                if record.trace.start_value is None
                else float(record.trace.start_value)
            ),
            breakpoint_escape_changed_count=int(
                record.trace.breakpoint_escape_changed_count
            ),
            mathematically_certified=bool(candidate.raw_objective_certified),
        ), search_round=record.trace.search_round,
            search_phase=record.trace.search_phase,
            outer_max_iter=0,
            inner_max_iter=0,
            certificate_max_iter=0,
        ),
    )


def _raw_reference_failure_diagnostics(
    records: list[CandidateRecord],
) -> RawReferenceFailureDiagnostics:
    """Summarize raw candidates without flattening solver state."""

    raw = [
        record for record in records if isinstance(record.candidate, RawFusionCandidate)
    ]
    attempts = [
        attempt for record in raw for attempt in _raw_attempts(record)
    ]
    finite_attempts = [
        attempt for attempt in attempts if np.isfinite(attempt.kkt_residual)
    ]
    min_residual = float("nan")
    min_tolerance = float("nan")
    best_attempt: RawAttemptTrace | None = None
    dominant_component = "unknown"
    if finite_attempts:
        best_attempt = min(finite_attempts, key=lambda attempt: attempt.kkt_residual)
        min_residual = float(best_attempt.kkt_residual)
        if np.isfinite(float(best_attempt.kkt_tolerance)):
            min_tolerance = float(best_attempt.kkt_tolerance)
        dominant_component = str(best_attempt.dominant_kkt_component)

    mm_values = [
        attempt.convergence.mm_consistency_violations for attempt in attempts
    ]
    direct = [record for record in records if record.family == "direct_partition"]
    return RawReferenceFailureDiagnostics(
        raw_candidate_count=len(raw),
        raw_solver_attempt_count=len(attempts),
        raw_certified_count=sum(
            bool(record.candidate.raw_objective_certified) for record in raw
        ),
        partition_certified_count=sum(
            bool(record.candidate.partition.certified) for record in raw
        ),
        direct_candidate_count=len(direct),
        direct_eligible_count=sum(record.eligible_for_selection for record in direct),
        min_kkt_residual=min_residual,
        min_kkt_tolerance=min_tolerance,
        best_raw_attempt=best_attempt,
        dominant_kkt_component=dominant_component,
        mm_violation_min=min(mm_values, default=0),
        mm_violating_count=sum(value > 0 for value in mm_values),
        ineligibility_reason_counts=_stable_counts(
            record.candidate.ineligibility_reason for record in raw
        ),
        certificate_status_counts=_stable_counts(
            attempt.certificate_status for attempt in attempts
        ),
        search_phase_counts=_stable_counts(record.trace.search_phase for record in raw),
        start_source_counts=_stable_counts(record.trace.start_source for record in raw),
        promotion_status_counts=_stable_counts(
            attempt.promotion_status for attempt in attempts
        ),
        attempt_summaries=tuple(
            _compact_raw_attempt_summary(attempt) for attempt in attempts
        ),
    )


def _select_raw_reference(
    records: list[CandidateRecord],
    *,
    tumor_id: str,
    adaptive_search_stop_reason: str,
) -> CandidateRecord:
    eligible = [
        record
        for record in records
        if isinstance(record.candidate, RawFusionCandidate)
        and raw_candidate_has_exact_fusion_certificate(record.candidate)
    ]
    if not eligible:
        raise NoCertifiedRawReferenceError(
            tumor_id=tumor_id,
            records=records,
            adaptive_search_stop_reason=adaptive_search_stop_reason,
        )
    return min(
        eligible,
        key=lambda record: (
            float(record.score.value),
            float(record.score.numerical_uncertainty),
            float(record.candidate.raw_fit.provenance.lambda_value),
            float(record.candidate.raw_fit.objective.total),
            int(record.candidate_id),
        ),
    )


def _assemble_selection_result(
    *,
    data,
    result_entries,
    selection_method,
    adaptive_search_stop_reason,
    ward_candidate_pool_complete: bool = False,
) -> BICSelectionResult:
    try:
        decision = select_candidate_records(
            result_entries,
        )
    except ValueError as exc:
        raise NoEligibleModelSelectionCandidatesError(
            tumor_id=data.tumor_id,
            candidates=tuple(result_entries),
        ) from exc

    selected_record = decision.selected
    selected_candidate = selected_record.candidate
    validate_candidate_identity(selected_candidate)
    if not selected_candidate.eligible_for_selection:
        raise AssertionError("Ineligible partition candidate reached selection.")

    # The raw estimator reference is chosen independently from the selected
    # partition and never inherited by a direct candidate.
    raw_reference_record = _select_raw_reference(
        result_entries,
        tumor_id=str(data.tumor_id),
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
    selected_kkt_value = (
        float(selected_candidate.raw_fit.certificate.components.residual)
        if selected_is_raw
        else float("nan")
    )
    selected_kkt_residual = (
        selected_kkt_value if np.isfinite(selected_kkt_value) else None
    )

    selected_model = SelectedModel(
        raw_reference=raw_reference,
        partition_candidate=selected_candidate,
        partition_parent_raw=partition_parent_raw,
    )
    ordered_records = sorted(
        result_entries,
        key=lambda record: (
            record.lambda_value is None,
            float("inf") if record.lambda_value is None else record.lambda_value,
            int(record.candidate_id),
        ),
    )
    search = tuple(
        SearchCandidate(
            record=record,
            selected=int(record.candidate_id) == int(selected_record.candidate_id),
        )
        for record in ordered_records
    )
    return BICSelectionResult(
        selected_model=selected_model,
        search=search,
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
        num_candidates_certified=int(decision.num_eligible),
        selected_kkt_residual=selected_kkt_residual,
        ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
        raw_lambda_path_resolved=bool(adaptive_search_global_optimum_certified),
        global_hybrid_optimum_certified=bool(selection_optimum_resolved),
    )


@dataclass(slots=True)
class _SearchState:
    """Per-tumor search ownership, including its one cached precision twin."""
    result_entries: list[CandidateRecord] = field(default_factory=list)
    fit_by_lambda: dict[float, RawFit] = field(default_factory=dict)
    partition_k_by_lambda: dict[float, int] = field(default_factory=dict)
    recovery_fit_by_lambda: dict[float, RawFit] = field(default_factory=dict)
    bic_refit_cache: dict[object, PartitionRefitCacheEntry] = field(default_factory=dict)
    next_step: int = 0
    recovery_context: PreparedProblem | None = None
    recovery_status: str = "not_requested"


def _solve_raw_proposal(
    search: _SearchState, *, base_solver_context: PreparedProblem,
    guided_initialization, raw_guide_phi: StartArray, proposal,
    candidate_fit_options: _FitOptions, lambda_key: float,
) -> tuple[RawFit, RawAttemptTrace, tuple[RawAttemptTrace, ...]]:
    context = base_solver_context
    # Certification-recovery attempts run at float64: iteration budget
    # alone measurably plateaus above the KKT gate on widened-mixture
    # tumors, while the float64 re-solve of the same frozen objective
    # (identical objective_spec_hash) removes the float32 stationarity
    # floor. Promotion is best-effort; without memory the attempt
    # keeps the working precision and its existing failure mode.
    recovery_promoted = False
    recovery_promotion_status = "not_requested"
    if proposal.phase in {
        "solver_recovery",
        "bootstrap_certification_anchor",
    }:
        if context.runtime.dtype == torch.float64:
            recovery_promotion_status = "not_needed"
        else:
            if search.recovery_context is None:
                (
                    search.recovery_context,
                    search.recovery_status,
                ) = _try_promote_recovery_context(context)
            recovery_promotion_status = str(search.recovery_status)
            if search.recovery_context is not context:
                context = search.recovery_context
                recovery_promoted = True
    initialization = guided_initialization
    warm_fit = None
    if proposal.warm_start_lambda is not None:
        warm_fit = search.fit_by_lambda.get(
            _canonical_lambda(proposal.warm_start_lambda)
        )
    alternate_fit = None
    if proposal.alternate_start_lambda is not None:
        alternate_fit = search.fit_by_lambda.get(
            _canonical_lambda(proposal.alternate_start_lambda)
        )
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
        best_failed_fit = search.recovery_fit_by_lambda.get(lambda_key)
        if best_failed_fit is not None:
            append_distinct_start(
                "best_same_lambda_kkt_state",
                float(best_failed_fit.provenance.lambda_value),
                best_failed_fit.state,
            )
        else:
            append_distinct_start(
                "guided_kkt_solver_recovery",
                float(initialization.lambda_value),
                initialization.solver_state,
            )
    elif int(proposal.retry_number) > 0:
        if (
            int(proposal.retry_number) == 1
            and alternate_fit is not None
            and alternate_fit.state is not None
        ):
            append_distinct_start(
                "alternate_bracket_endpoint",
                float(proposal.alternate_start_lambda),
                alternate_fit.state,
            )
        elif (
            warm_fit is not None
            and warm_fit.state is not None
        ):
            append_distinct_start(
                "same_lambda_retry",
                float(proposal.warm_start_lambda),
                warm_fit.state,
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
            and warm_fit is not None
            and warm_fit.state is not None
        ):
            append_distinct_start(
                "warm_bracket_left"
                if proposal.phase == "refine_partition_event"
                else "warm_endpoint",
                float(proposal.warm_start_lambda),
                warm_fit.state,
            )
        if (
            proposal.phase == "refine_partition_event"
            and alternate_fit is not None
            and alternate_fit.state is not None
        ):
            append_distinct_start(
                "warm_bracket_right",
                float(proposal.alternate_start_lambda),
                alternate_fit.state,
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
            and search.partition_k_by_lambda.get(warm_key) == 1
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

    for source, start_value, state, phi in (
        _explicit_path_default_start_specs(
            scalar_well_starts=context.scalar_well_starts,
            pooled_start=context.pooled_start,
        )
    ):
        append_distinct_start(source, start_value, state, phi)

    start_traces: list[RawAttemptTrace] = []
    selected_attempt: _RawStartAttempt | None = None
    selected_trace: RawAttemptTrace | None = None
    for (
        lambda_start_source,
        lambda_start_value,
        original_state,
        explicit_phi_start,
    ) in start_specs:
        if recovery_promoted:
            # A promoted attempt keeps only the primal start: working-
            # precision dual/certificate state is not carried across
            # the dtype boundary, and the float64 solve refines fresh
            # duals before certification.
            solver_state_start, changed_count = None, 0
            cold_state = original_state
        else:
            solver_state_start, changed_count = _escape_emission_breakpoint_retry_state(
                original_state,
                start_source=lambda_start_source,
                start_lambda=lambda_start_value,
                target_lambda=float(proposal.lambda_value),
                context=context,
                tol=float(candidate_fit_options.solver.tolerance),
            )
            cold_state = solver_state_start
        # A warm attempt already owns its primal. Only cold/promoted
        # attempts need a detached copy; never build and discard a
        # competing clone of the warm state's matrix.
        phi_start = None if solver_state_start is not None else _clone_start(
            cold_state.phi if cold_state is not None and cold_state.phi is not None
            else explicit_phi_start if explicit_phi_start is not None else raw_guide_phi
        )
        seed_fit = fit_prepared(
            context,
            float(proposal.lambda_value),
            candidate_fit_options.solver,
            phi_start=phi_start,
            include_default_starts=False,
            warm_state=solver_state_start,
        )
        if str(seed_fit.provenance.objective_spec_hash) != str(
            context.objective_spec_hash
        ):
            raise AssertionError(
                "Raw multistart changed the fixed objective identity."
            )
        seed_fit = _offload_raw_fit_to_cpu(seed_fit)
        recovery_fit = search.recovery_fit_by_lambda.get(lambda_key)
        residual = float(seed_fit.certificate.components.residual)
        if seed_fit.state is not None and np.isfinite(residual) and (
            recovery_fit is None
            or residual < float(recovery_fit.certificate.components.residual)
        ):
            search.recovery_fit_by_lambda[lambda_key] = seed_fit
        mathematically_certified = bool(
            float(seed_fit.provenance.lambda_value) > 0.0
            and seed_fit.certificate.certified
            and seed_fit.certificate.admissible
        )
        raw_attempt = _RawStartAttempt(
            fit=seed_fit,
            source=str(lambda_start_source),
            start_value=float(lambda_start_value),
            breakpoint_escape_changed_count=int(changed_count),
            mathematically_certified=bool(mathematically_certified),
            promotion_status=str(recovery_promotion_status),
        )
        trace = _raw_attempt_trace(
            raw_attempt, search_round=int(search.next_step), search_phase=str(proposal.phase),
            outer_max_iter=int(candidate_fit_options.solver.outer_max_iter),
            inner_max_iter=int(candidate_fit_options.solver.inner_max_iter),
            certificate_max_iter=int(candidate_fit_options.solver.certificate.max_iter),
        )
        start_traces.append(trace)
        if selected_attempt is None or _select_raw_start_attempt(
            [selected_attempt, raw_attempt]
        ) is raw_attempt:
            selected_attempt, selected_trace = raw_attempt, trace
        del seed_fit, raw_attempt, recovery_fit
    if selected_attempt is None or selected_trace is None:
        raise ValueError("At least one raw start attempt is required.")
    # Subsequent bracket proposals must warm-start from the same raw
    # basin that was admitted to partition scoring, never from a lower
    # objective but mathematically uncertified side attempt.
    return selected_attempt.fit, selected_trace, tuple(start_traces)



def _partition_guided_admm_selection(
    *,
    data: TumorData,
    fit_options: _FitOptions,
) -> BICSelectionResult:
    """Run the certified raw path and select under one immutable contract.

    Ward/CEM supplies the primal start and initial-lambda scale. The independent
    zero-penalty pilot defines the frozen graph. Retained pilot/final-raw-phi
    partitions enter the secondary selection pool. Direct proposals are
    evaluated only after the raw lambda controller terminates, so they cannot
    steer or replace the raw optimizer.
    """

    selection_method = "online_partition_guided_admm"
    if int(data.num_mutations) < 2:
        raise ValueError(
            "partition_guided_admm requires at least two mutations so that a "
            "positive pairwise penalty is solved by ADMM."
        )
    pilot_context = prepare_torch_problem_with_resource_policy(
        data, fit_options, defer_graph=True, graph=None,
    )
    pilot_phi: StartArray = pilot_context.exact_pilot
    pilot_runtime = pilot_context.runtime
    pilot_context.validate(allow_deferred_graph=True)
    guide_curvature = observed_curvature_at_pilot_torch(
        pilot_context.model, pilot_phi, eps=pilot_context.eps,
    )
    initializer_pool = generate_partition_initializer_pool(
        context=pilot_context,
        pilot_phi=pilot_phi,
        fit_options=fit_options,
        curvature=guide_curvature,
    )
    guide = _best_partition_candidate(list(initializer_pool))
    if guide is None:
        raise RuntimeError(
            "No finite active-score partition initializer was available for tumor "
            f"{data.tumor_id}."
        )

    # Keep the partition guide host-backed for exact CPU behavior and fallback.
    # CUDA graph construction uploads this small M x S matrix once; the O(M^2)
    # graph itself stays device-backed and is reused by context preparation.
    guide_phi: StartArray = np.asarray(guide.phi_start)
    graph_builder_phi = pilot_phi
    complete_graph_degree = float(max(int(data.num_mutations) - 1, 1))
    likelihood_noise_degree_exponent = float(
        PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT
    )
    likelihood_noise_divisor = float(
        complete_graph_degree**likelihood_noise_degree_exponent
    )
    selection_graph, prebuilt_tensor_graph, _ = (
        _build_partition_guided_graph_with_resource_policy(
            guide_phi=graph_builder_phi,
            guide_curvature=guide_curvature,
            solver_context=pilot_context,
            fit_options=fit_options,
            noise_divisor=likelihood_noise_divisor,
        )
    )
    base_solver_context = prepare_torch_problem_with_resource_policy(
        data, fit_options,
        # The likelihood pilot initializes adaptive weights; curvature and a
        # mild degree correction set a finite data-derived distance floor. This
        # prevents the fixed 1e-6 floor from making the proposed blocks
        # effectively immutable while retaining the current estimator as the
        # requested initializer.
        graph=selection_graph,
        prebuilt_tensor_graph=prebuilt_tensor_graph,
        # Preserve the independent likelihood starts.  The previous flow
        # replaced both with the Ward guide, so nominal "cold" retries were
        # merely duplicates of the same non-convex basin.
        exact_pilot=pilot_context.exact_pilot,
        pooled_start=pilot_context.pooled_start,
        scalar_well_starts=pilot_context.scalar_well_starts,
        runtime=pilot_runtime,
    )
    base_solver_context = transfer_scalar_pilot_certificates(
        pilot_context, base_solver_context
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
    effective_fit_options = replace(
        fit_options,
        graph=replace(fit_options.graph, graph=effective_graph),
    )
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
    effective_graph = base_solver_context.graph_spec
    effective_tensor_graph = base_solver_context.graph
    effective_fit_options = replace(
        fit_options,
        graph=replace(fit_options.graph, graph=effective_graph),
    )
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm CPU fallback changed the complete fusion graph."
        )
    controller = OnlineLambdaController(
        initial_lambda=float(guided_initialization.lambda_value),
        initial_reason="partition_guide_kkt_balance",
        config=OnlineLambdaConfig(
            guide_n_clusters=int(np.unique(raw_guide_labels).size),
            num_mutations=int(data.num_mutations),
            kkt_tolerance=5.0 * float(effective_fit_options.solver.tolerance),
            max_unique_lambdas=int(
                effective_fit_options.selection.lambda_search.exploration_budget
            ),
            max_refinement_lambdas=int(
                effective_fit_options.selection.lambda_search.refinement_budget
            ),
            max_solver_retries_per_lambda=int(
                effective_fit_options.selection.lambda_search.solver_retry_limit
            ),
            partition_event_mode=True,
        ),
    )

    search = _SearchState()
    while True:
        proposal = controller.propose()
        if proposal is None:
            break
        lambda_key = _canonical_lambda(proposal.lambda_value)
        for attempt_key in list(search.recovery_fit_by_lambda):
            if float(attempt_key) != float(lambda_key):
                del search.recovery_fit_by_lambda[attempt_key]
        candidate_fit_options = solver_retry_fit_options(
            data, effective_fit_options, retry_number=int(proposal.retry_number),
            certification_recovery=proposal.phase in {"solver_recovery", "bootstrap_certification_anchor"},
        )

        selected_raw_fit, selected_start, raw_start_attempts = _solve_raw_proposal(
            search, base_solver_context=base_solver_context,
            guided_initialization=guided_initialization, raw_guide_phi=raw_guide_phi,
            proposal=proposal, candidate_fit_options=candidate_fit_options,
            lambda_key=lambda_key,
        )
        fit, artifact = evaluate_raw_fusion_candidate(
            data=data,
            fit_options=effective_fit_options,
            lambda_value=float(proposal.lambda_value),
            bic_refit_cache=search.bic_refit_cache,
            precomputed_fit=selected_raw_fit,
            source_model=base_solver_context.source_model,
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
        candidate_id = int(len(search.result_entries))
        search.result_entries.append(
            CandidateRecord(
                candidate_id=candidate_id,
                candidate=artifact,
                trace=CandidateTrace(
                    search_round=int(search.next_step),
                    search_phase=str(proposal.phase),
                    start_source=str(lambda_start_source),
                    start_value=float(lambda_start_value),
                    breakpoint_escape_changed_count=int(
                        path_breakpoint_escape_changed_count
                    ),
                    raw_attempts=raw_start_attempts,
                ),
            )
        )
        incumbent = search.fit_by_lambda.get(lambda_key)
        if _prefer_fit_candidate(fit, incumbent):
            search.fit_by_lambda[lambda_key] = (
                replace(fit, state=None)
                if (
                    fit.state is not None
                    and str(fit.provenance.dtype) == "float64"
                    and base_solver_context.runtime.dtype != torch.float64
                )
                else fit
            )
            search.partition_k_by_lambda[lambda_key] = int(artifact.partition.n_clusters)

        raw_exact_certified = bool(
            raw_candidate_has_exact_fusion_certificate(artifact)
            and bool(effective_tensor_graph.is_complete)
        )
        selection_score_available = bool(artifact.eligible_for_selection)
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=float(proposal.lambda_value),
                n_clusters=int(artifact.partition.n_clusters),
                partition_signature=str(artifact.partition.signature),
                # The active selection score steers the online-lambda
                # controller (the observation field name is historical).
                partition_icl=(
                    float(artifact.score.value)
                    if selection_score_available
                    else float("inf")
                ),
                kkt_residual=float(fit.certificate.components.residual),
                raw_objective_certified=bool(raw_exact_certified),
                partition_certified=bool(artifact.partition.certified),
                selection_score_available=selection_score_available,
                score_numerical_uncertainty=float(artifact.score.numerical_uncertainty),
                degrees_of_freedom=int(artifact.score.degrees_of_freedom),
            )
        )
        search.next_step += 1

    # The production candidate pool always includes pilot and final-Phi ladders.
    direct_proposals: list[
        tuple[
            PartitionCandidate,
            str,
            CandidateRecord | None,
        ]
    ] = [
        (proposal, "pilot", None)
        for proposal in initializer_pool
    ]
    raw_parent_records = sorted(
        (
            record
            for record in search.result_entries
            if isinstance(record.candidate, RawFusionCandidate)
            and raw_candidate_has_exact_fusion_certificate(record.candidate)
        ),
        key=lambda record: (
            float(record.score.value),
            float(record.score.numerical_uncertainty),
            float(record.candidate.raw_fit.provenance.lambda_value),
            int(record.candidate_id),
        ),
    )[: FINAL_PHI_PARENT_COUNT]
    final_k_grid = tuple(
        range(
            1,
            min(
                FINAL_PHI_LADDER_KMAX,
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
            context=base_solver_context,
            pilot_phi=np.asarray(parent.raw_fit.phi, dtype=np.float64),
            fit_options=effective_fit_options,
            declared_k_grid=final_k_grid,
        )
        direct_proposals.extend(
            (proposal, "final_phi", parent_record)
            for proposal in final_pool
        )

    for proposal, stage, parent_record in direct_proposals:
        parent_candidate = (
            None if parent_record is None else parent_record.candidate
        )
        parent_raw = (
            parent_candidate
            if isinstance(parent_candidate, RawFusionCandidate)
            else None
        )
        source = _direct_partition_source(proposal, stage=stage)
        candidate_id = int(len(search.result_entries))
        direct_candidate = evaluate_direct_partition_candidate(
            data=data,
            proposal=proposal,
            selection_options=effective_fit_options,
            source=source,
            parent_raw_candidate_id=(
                None if parent_record is None else int(parent_record.candidate_id)
            ),
            parent_raw_lambda=(
                None
                if parent_raw is None
                else float(parent_raw.raw_fit.provenance.lambda_value)
            ),
            parent_raw_phi_hash=(
                ""
                if parent_raw is None
                else _pilot_matrix_hash(parent_raw.raw_fit.phi)
            ),
            refit_cache=search.bic_refit_cache,
            source_model=base_solver_context.source_model,
        )
        search.result_entries.append(
            CandidateRecord(
                candidate_id=candidate_id,
                candidate=direct_candidate,
                trace=CandidateTrace(
                    search_round=int(search.next_step),
                    search_phase=f"{stage}_direct_partition_pool",
                ),
            )
        )
        search.next_step += 1

    if not search.result_entries:
        raise RuntimeError(
            f"No guided ADMM candidates were evaluated for tumor {data.tumor_id}."
        )
    stop_reason = str(controller.stop_reason or "online_lambda_no_terminal_reason")
    return _assemble_selection_result(
        data=data,
        result_entries=search.result_entries,
        selection_method=selection_method,
        adaptive_search_stop_reason=stop_reason,
        ward_candidate_pool_complete=True,
    )


def _select_single_mutation(data: TumorData, fit_config: _FitOptions) -> BICSelectionResult:
    """Fit the separable scalar problem without a guide or lambda ladder.

    All pairwise penalties are identically zero. The existing no-edge solver
    computes scalar mixture starts and audits their coordinatewise refinement;
    it performs no ADMM iterations. Its bounded scalar evidence, full-KKT gate,
    and the profile's distinct fixed-label refit contract remain unchanged.
    """
    graph = fit_config.graph.graph
    if graph is None:
        graph = build_complete_uniform_graph(1)
    options = replace(
        fit_config,
        lambda_value=0.0,
        graph=replace(fit_config.graph, graph=graph),
    )
    context = prepare_torch_problem_with_resource_policy(data, options)
    fit = fit_prepared(context, options.lambda_value, options.solver)
    _, candidate = evaluate_raw_fusion_candidate(
        data=data,
        fit_options=options,
        lambda_value=0.0,
        precomputed_fit=fit,
    )
    result = _assemble_selection_result(
        data=data,
        result_entries=[CandidateRecord(
            candidate_id=0,
            candidate=candidate,
            trace=CandidateTrace(
                search_round=0,
                search_phase="singleton_no_edges",
                start_source="scalar_mixture",
            ),
        )],
        selection_method="singleton_scalar_no_edges",
        adaptive_search_stop_reason="singleton_no_edges",
    )
    # There is no lambda/partition boundary to explore. This topological fact
    # does not upgrade a local KKT point or an approximate scalar refit into a
    # global likelihood optimum.
    globally_certified = bool(
        fit.certificate.global_optimum and candidate.refit.global_optimum_certified
    )
    return replace(
        result,
        selection_hits_lower_boundary=False,
        selection_hits_upper_boundary=False,
        selection_boundary_unresolved=False,
        raw_lambda_path_resolved=True,
        selection_optimum_resolved=globally_certified,
        global_hybrid_optimum_certified=globally_certified,
    )


def select_model(
    *,
    data: TumorData,
    fit_config: _FitOptions,
) -> BICSelectionResult:
    validate_public_tumor_data(data, fit_config)
    effective_objective_shape = objective_shape_for_data(
        data, str(fit_config.solver.objective_shape)
    )
    if effective_objective_shape != str(fit_config.solver.objective_shape):
        fit_config = replace(
            fit_config,
            solver=replace(fit_config.solver, objective_shape=effective_objective_shape),
        )

    if data.num_mutations == 1:
        return _select_single_mutation(data, fit_config)

    return _partition_guided_admm_selection(
        data=data,
        fit_options=fit_config,
    )


__all__ = [
    "BICSelectionResult",
    "NoCertifiedRawReferenceError",
    "NoEligibleModelSelectionCandidatesError",
    "select_model",
]
