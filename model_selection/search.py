from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..config import (
    FitConfig,
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
    PRODUCTION_SELECTION_POLICY,
)
from ..api import fit_fixed_objective
from ..core.fusion.partition_starts import (
    PartitionCandidate,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.interface import SolveBudget, SolverInit
from ..core.fusion.solver import (
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
    promote_solver_context_dtype,
)
from ..core.fusion.types import (
    RawFit,
    SolverContext,
    SolverState,
    WorkCounters,
    WorkLedger,
)
from ..io.data import TumorData
from ..core.scalar import PartitionFit

from ..model_selection.candidates import (
    candidate_from_raw_fit,
    evaluate_direct_partition_candidate,
    validate_candidate_identity,
)
from ..model_selection.checkpoint import (
    SearchCheckpoint,
    build_search_checkpoint_identity,
    load_search_checkpoint,
    save_search_checkpoint,
    solver_state_content_key,
)
from ..model_selection.guided_fusion import GuidedFusionInitialization
from ..model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
    OnlineLambdaProposal,
)
from ..model_selection.partitions import (
    best_partition_candidate,
)
from ..model_selection.proposals import (
    RawStartAttempt,
    RawStartSpec,
    adaptive_stop_certifies_global_optimum,
    bootstrap_independent_start_specs,
    build_guided_initialization_with_resource_policy,
    build_partition_guided_graph_with_resource_policy,
    clone_start,
    direct_partition_source,
    escape_path_breakpoint_retry_state,
    explicit_path_default_start_specs,
    generate_partition_initializer_pool,
    offload_solver_state_to_cpu,
    pilot_matrix_hash,
    rescore_partition_candidates,
    select_raw_start_attempt,
    solver_recovery_fit_options,
)
from ..model_selection.scoring import (
    canonical_lambda,
    prefer_fit_candidate,
    raw_candidate_has_exact_fusion_certificate,
    raw_fit_has_exact_fusion_certificate,
    select_candidate_records,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateTrace,
    CandidateRecord,
    DiagnosticOnlyResult,
    DirectProposal,
    DirectPartitionCandidate,
    PartitionRefitKey,
    RawAttemptSummary,
    RawFusionArtifact,
    RawFusionCandidate,
    SearchReport,
    SecondaryFallbackResult,
    SelectedModel,
    SolveOutcome,
    StartArray,
    TumorSelectionOutcome,
    UnscoredRawFusionCandidate,
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


_RAW_ARTIFACT_TYPES = (RawFusionCandidate, UnscoredRawFusionCandidate)
_SELECTION_METHOD = "online_partition_guided_admm"


@dataclass(slots=True)
class LambdaRun:
    """Persistent result and transient retry attempts for one lambda."""

    lambda_value: float
    attempts: list[SolveOutcome] = field(default_factory=list)
    retained: SolveOutcome | None = None
    partition_k: int | None = None


@dataclass(slots=True)
class DirectPoolState:
    """Checkpointable progress through the deterministic direct pool."""

    next_index: int = 0
    complete: bool = False
    stop_reason: str | None = None
    proposals: list[DirectProposal] | None = None
    proposals_complete: bool = False
    final_parent_next_index: int = 0


@dataclass(slots=True)
class SearchState:
    """Single checkpointable owner of mutable model-selection state."""

    controller: OnlineLambdaController
    lambdas: dict[float, LambdaRun]
    candidates: list[CandidateRecord]
    bic_refit_cache: dict[PartitionRefitKey, PartitionFit]
    work_ledger: WorkLedger
    mandatory_guide_work: WorkCounters
    search_stop_override: str | None
    direct_pool: DirectPoolState
    raw_guide_phi: StartArray
    guided_initialization: GuidedFusionInitialization
    float64_recovery_status: str
    cumulative_search_active_seconds: float
    float64_recovery_context: SolverContext | None = field(
        init=False,
        default=None,
        repr=False,
    )
    checkpoint_generation: int = field(init=False, default=0, repr=False)

    def lambda_run(self, lambda_value: float) -> LambdaRun:
        key = canonical_lambda(lambda_value)
        run = self.lambdas.get(key)
        if run is None:
            run = LambdaRun(lambda_value=float(key))
            self.lambdas[key] = run
        return run

    @property
    def next_candidate_id(self) -> int:
        """Derive the next contiguous identity from the authoritative records."""

        return len(self.candidates)

    @property
    def total_work(self) -> WorkCounters:
        """Return the immutable cumulative-work snapshot."""

        return self.work_ledger.total

    def charge(self, work: WorkCounters) -> None:
        """Accumulate search work through one typed state boundary."""

        self.work_ledger.charge(work)

    def discard_transient_attempts_except(self, lambda_value: float) -> None:
        keep = canonical_lambda(lambda_value)
        for key, run in self.lambdas.items():
            if float(key) != float(keep):
                run.attempts.clear()


@dataclass(frozen=True, slots=True)
class _PartitionGuidedRuntime:
    """Immutable dependencies shared by the four selection phases."""

    data: TumorData
    fit_options: FitConfig
    solver_context: SolverContext
    initializer_pool: tuple[PartitionCandidate, ...]
    checkpoint_path: Path | None
    checkpoint_identity: dict[str, object] | None
    search_segment_start: float
    use_warm_starts: bool
    resume_checkpoint: bool


def _save_search_state(
    *,
    path: Path | None,
    identity: dict[str, object] | None,
    state: SearchState,
    segment_started_at: float,
) -> None:
    if path is None:
        return
    if identity is None:  # pragma: no cover - construction guard
        raise AssertionError("Checkpoint path has no identity guard.")
    payload = replace(
        state,
        cumulative_search_active_seconds=float(
            state.cumulative_search_active_seconds + perf_counter() - segment_started_at
        ),
    )
    state.checkpoint_generation = save_search_checkpoint(
        path,
        identity=identity,
        expected_generation=int(state.checkpoint_generation),
        checkpoint=SearchCheckpoint.capture(payload),
    )


def _post_guide_partition_refit_objective_evaluations(
    total_work: WorkCounters,
    mandatory_guide_work: WorkCounters,
) -> int:
    """Return budgeted scalar work after immutable guide construction.

    The complete mandatory guide can define the adaptive graph and therefore
    the estimator itself.  It is always completed, included in total realized
    work, and excluded only from the optional post-guide resource cap.
    """

    realized = int(total_work.partition_refit_objective_evaluations)
    baseline = int(mandatory_guide_work.partition_refit_objective_evaluations)
    if realized < baseline:
        raise ValueError("Total search work is below mandatory guide work.")
    return int(realized - baseline)


@dataclass(frozen=True, slots=True)
class BestRawAttemptDiagnostics:
    search_round: int
    search_phase: str
    lambda_value: float
    source: str
    kkt_residual: float
    kkt_tolerance: float
    dominant_kkt_component: str
    outer_max_iter: int
    inner_max_iter: int
    certificate_max_iter: int
    working_dtype: str = "not_recorded"
    audit_dtype: str = "not_recorded"
    precision_polished: bool = False
    promotion_status: str = "not_recorded"
    stage_outer_iterations: int = 0
    stage_outer_max_iter: int = 0
    stage_inner_iterations: int = 0
    stage_inner_max_iter: int = 0
    stage_inner_solve_calls: int = 0
    stop_reason: str = "not_recorded"
    progress_residual_method: str = "not_recorded"
    solve_tolerance: float = float("nan")
    legacy_stop_kkt_residual: float = float("inf")
    componentwise_stop_kkt_residual: float = float("inf")
    accepted_full_steps: int = 0
    accepted_damped_steps: int = 0
    rejected_outer_steps: int = 0
    fallback_reason: str = ""


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
    best_raw_attempt: BestRawAttemptDiagnostics | None
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
                    f"{best.stage_outer_iterations}/{best.stage_outer_max_iter}"
                ),
                "stage_inner": best.stage_inner_iterations,
                "stage_inner_max": best.stage_inner_max_iter,
                "stage_inner_calls": best.stage_inner_solve_calls,
                "stop": best.stop_reason,
                "progress": best.progress_residual_method,
                "solve_tol": best.solve_tolerance,
                "legacy_kkt": best.legacy_stop_kkt_residual,
                "componentwise_kkt": best.componentwise_stop_kkt_residual,
                "steps": (
                    f"{best.accepted_full_steps}/"
                    f"{best.accepted_damped_steps}/"
                    f"{best.rejected_outer_steps}"
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
    normalized = ["<missing>" if value is None else str(value) for value in values]
    return tuple((value, normalized.count(value)) for value in sorted(set(normalized)))


def _try_promote_recovery_context(
    context: SolverContext,
) -> tuple[SolverContext, str]:
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


def _raw_attempt_diagnostics(
    attempt: RawAttemptSummary,
    record: CandidateRecord,
) -> BestRawAttemptDiagnostics:
    audit = attempt.fit
    certificate = audit.certificate
    convergence = audit.convergence
    limits = attempt.limits
    components = {
        "stationarity": float(certificate.components.stationarity),
        "edge_subgradient": float(certificate.components.edge_subgradient),
        "dual_ball": float(certificate.components.dual_ball),
        "box": float(certificate.components.box),
    }
    finite_components = {
        name: value for name, value in components.items() if np.isfinite(value)
    }
    return BestRawAttemptDiagnostics(
        search_round=int(record.trace.search_round),
        search_phase=str(record.trace.search_phase),
        lambda_value=float(audit.provenance.lambda_value),
        source=str(attempt.source),
        kkt_residual=float(certificate.components.residual),
        kkt_tolerance=float(certificate.tolerance),
        dominant_kkt_component=(
            max(finite_components, key=finite_components.get)
            if finite_components
            else "unknown"
        ),
        outer_max_iter=int(limits.outer_max_iter),
        inner_max_iter=int(limits.inner_max_iter),
        certificate_max_iter=int(limits.certificate_max_iter),
        working_dtype=str(certificate.working_dtype),
        audit_dtype=str(certificate.audit_dtype),
        precision_polished=bool(certificate.precision_polished),
        promotion_status=str(attempt.promotion_status),
        stage_outer_iterations=int(convergence.stage_outer_iterations),
        stage_outer_max_iter=int(convergence.stage_outer_max_iter),
        stage_inner_iterations=int(convergence.stage_inner_iterations),
        stage_inner_max_iter=int(convergence.stage_inner_max_iter),
        stage_inner_solve_calls=int(convergence.stage_inner_solve_calls),
        stop_reason=str(convergence.stop_reason),
        progress_residual_method=str(convergence.progress_residual_method),
        solve_tolerance=float(convergence.solve_tolerance),
        legacy_stop_kkt_residual=float(convergence.legacy_stop_kkt_residual),
        componentwise_stop_kkt_residual=float(
            convergence.componentwise_stop_kkt_residual
        ),
        accepted_full_steps=int(convergence.accepted_full_steps),
        accepted_damped_steps=int(convergence.accepted_damped_steps),
        rejected_outer_steps=int(convergence.rejected_outer_steps),
        fallback_reason=str(certificate.fallback_reason),
    )


def _compact_raw_attempt_summary(
    attempt: RawAttemptSummary,
    record: CandidateRecord,
) -> str:
    """Render one scalar-only attempt token suitable for scheduler stderr."""

    item = _raw_attempt_diagnostics(attempt, record)
    steps = (
        f"{item.accepted_full_steps}/{item.accepted_damped_steps}/"
        f"{item.rejected_outer_steps}"
    )
    return (
        f"r{item.search_round}:{item.search_phase}:{item.source}@{item.lambda_value:.6g}"
        f"|kkt={item.kkt_residual:.6g}/{item.kkt_tolerance:.6g}"
        f"|dtype={item.working_dtype}/{item.audit_dtype}"
        f"|prom={item.promotion_status}|polish={int(item.precision_polished)}"
        f"|progress={item.progress_residual_method}"
        f"|solve_tol={item.solve_tolerance:.6g}"
        f"|outer={item.stage_outer_iterations}/{item.stage_outer_max_iter}"
        f"|inner={item.stage_inner_iterations}/"
        f"{item.stage_inner_solve_calls}x{item.stage_inner_max_iter}"
        f"|stop={item.stop_reason}"
        f"|stop_kkt={item.legacy_stop_kkt_residual:.6g}/"
        f"{item.componentwise_stop_kkt_residual:.6g}"
        f"|steps={steps}"
        f"|fallback={item.fallback_reason or 'none'}"
    )


def _raw_attempts(record: CandidateRecord) -> tuple[RawAttemptSummary, ...]:
    if record.trace.raw_attempts:
        return record.trace.raw_attempts
    candidate = record.candidate
    if not isinstance(candidate, _RAW_ARTIFACT_TYPES):
        return ()
    return (
        RawAttemptSummary.from_fit(
            candidate.raw_fit,
            source=str(record.trace.start_source),
            start_value=(
                float("nan")
                if record.trace.start_value is None
                else float(record.trace.start_value)
            ),
            breakpoint_escape_changed_count=int(
                record.trace.breakpoint_escape_changed_count
            ),
            mathematically_certified=bool(
                raw_fit_has_exact_fusion_certificate(candidate.raw_fit)
            ),
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
        record
        for record in records
        if isinstance(record.candidate, _RAW_ARTIFACT_TYPES)
    ]
    attempts = [
        (attempt, record) for record in raw for attempt in _raw_attempts(record)
    ]
    finite_attempts = [
        pair
        for pair in attempts
        if np.isfinite(float(pair[0].fit.certificate.components.residual))
    ]
    min_residual = float("nan")
    min_tolerance = float("nan")
    best_attempt: BestRawAttemptDiagnostics | None = None
    dominant_component = "unknown"
    if finite_attempts:
        attempt, record = min(
            finite_attempts,
            key=lambda pair: float(pair[0].fit.certificate.components.residual),
        )
        best_attempt = _raw_attempt_diagnostics(attempt, record)
        min_residual = float(best_attempt.kkt_residual)
        if np.isfinite(float(best_attempt.kkt_tolerance)):
            min_tolerance = float(best_attempt.kkt_tolerance)
        dominant_component = str(best_attempt.dominant_kkt_component)

    mm_values = [
        int(attempt.fit.convergence.mm_consistency_violations)
        for attempt, _ in attempts
    ]
    direct = [record for record in records if record.family == "direct_partition"]
    return RawReferenceFailureDiagnostics(
        raw_candidate_count=len(raw),
        raw_solver_attempt_count=len(attempts),
        raw_certified_count=sum(
            raw_fit_has_exact_fusion_certificate(record.candidate.raw_fit)
            for record in raw
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
            attempt.fit.certificate.status for attempt, _ in attempts
        ),
        search_phase_counts=_stable_counts(record.trace.search_phase for record in raw),
        start_source_counts=_stable_counts(record.trace.start_source for record in raw),
        promotion_status_counts=_stable_counts(
            attempt.promotion_status for attempt, _ in attempts
        ),
        attempt_summaries=tuple(
            _compact_raw_attempt_summary(attempt, record)
            for attempt, record in attempts
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


def select_secondary_fallback(
    records: tuple[CandidateRecord, ...] | list[CandidateRecord],
    *,
    require_global_refit: bool,
) -> CandidateRecord | None:
    """Select one scored direct partition without conferring raw provenance."""

    candidates = [
        record
        for record in records
        if isinstance(record.candidate, DirectPartitionCandidate)
        and record.candidate.eligible_for_selection
        and record.candidate.refit.finite_candidate_found
        and np.isfinite(float(record.score.value))
        and (
            not bool(require_global_refit)
            or record.candidate.refit.global_optimum_certified
        )
    ]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda record: (
            float(record.score.value),
            float(record.score.numerical_uncertainty),
            int(record.n_clusters),
            int(record.candidate_id),
        ),
    )
    validate_candidate_identity(selected.candidate)
    return selected


def _ordered_search(
    records: list[CandidateRecord],
    *,
    selected_candidate_id: int | None,
) -> tuple[CandidateRecord, ...]:
    ordered_records = sorted(
        records,
        key=lambda record: (
            record.lambda_value is None,
            float("inf") if record.lambda_value is None else record.lambda_value,
            int(record.candidate_id),
        ),
    )
    return tuple(ordered_records)


def _best_retained_raw_fit(records: list[CandidateRecord]) -> RawFit | None:
    """Choose one finite raw endpoint for diagnostics, never certification."""

    raw_records = [
        record
        for record in records
        if isinstance(record.candidate, _RAW_ARTIFACT_TYPES)
        and np.isfinite(float(record.candidate.raw_fit.objective.total))
    ]
    if not raw_records:
        return None

    def key(record: CandidateRecord) -> tuple[float, float, float, float, int]:
        fit = record.candidate.raw_fit
        residual = float(fit.certificate.components.residual)
        tolerance = float(fit.certificate.tolerance)
        ratio = (
            residual / tolerance
            if np.isfinite(residual) and np.isfinite(tolerance) and tolerance > 0.0
            else float("inf")
        )
        return (
            ratio,
            residual if np.isfinite(residual) else float("inf"),
            float(fit.objective.total),
            float(fit.provenance.lambda_value),
            int(record.candidate_id),
        )

    return min(raw_records, key=key).candidate.raw_fit


def _compact_raw_candidate(candidate: RawFusionArtifact) -> RawFusionArtifact:
    """Drop transient solver tensors after scalar attempt provenance is saved."""

    raw_fit = candidate.raw_fit
    if raw_fit.state is None and raw_fit.certificate.witness is None:
        return candidate
    return replace(
        candidate,
        raw_fit=replace(
            raw_fit,
            state=None,
            certificate=replace(raw_fit.certificate, witness=None),
        ),
    )


def _detach_solve_outcome(fit: RawFit) -> SolveOutcome:
    """Separate reusable solver state from the tensor-free fitted result."""

    state = fit.state
    if state is not None:
        state = offload_solver_state_to_cpu(state)
    compact_fit = replace(
        fit,
        state=None,
        certificate=replace(fit.certificate, witness=None),
    )
    return SolveOutcome(fit=compact_fit, state=state)


def _assemble_selection_result(
    *,
    data,
    result_entries,
    selection_method,
    adaptive_search_stop_reason,
    strict_positive_exact_fusion: bool = False,
    ward_candidate_pool_complete: bool = False,
    require_global_secondary_refit: bool = False,
    candidate_pool_complete: bool = True,
    raw_lambda_path_resolved: bool | None = None,
    selection_pool_stop_reason: str = "none",
) -> TumorSelectionOutcome:
    final_adaptive_search_stop_reason = str(adaptive_search_stop_reason)
    raw_path_resolved = bool(
        adaptive_stop_certifies_global_optimum(final_adaptive_search_stop_reason)
        if raw_lambda_path_resolved is None
        else raw_lambda_path_resolved
    )
    try:
        decision = select_candidate_records(
            result_entries,
            strict_positive_exact_fusion=bool(strict_positive_exact_fusion),
        )
    except ValueError:
        failure = NoEligibleModelSelectionCandidatesError(
            tumor_id=data.tumor_id,
            candidates=tuple(result_entries),
        )
        return DiagnosticOnlyResult(
            best_raw_attempt=_best_retained_raw_fit(result_entries),
            reason=f"{type(failure).__name__}: {failure}",
            report=SearchReport(
                records=_ordered_search(result_entries, selected_candidate_id=None),
                selected_id=None,
                selection_method=str(selection_method),
                adaptive_search_stop_reason=final_adaptive_search_stop_reason,
                num_candidates_certified=0,
                ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
                raw_lambda_path_resolved=bool(raw_path_resolved),
                selection_pool_stop_reason=str(selection_pool_stop_reason),
            ),
        )

    selected_record = decision.selected
    selected_candidate = selected_record.candidate
    validate_candidate_identity(selected_candidate)
    if not selected_candidate.eligible_for_selection:
        raise AssertionError("Ineligible partition candidate reached selection.")

    # The raw estimator reference is chosen independently from the selected
    # partition and never inherited by a direct candidate.
    try:
        raw_reference_record = _select_raw_reference(
            result_entries,
            tumor_id=str(data.tumor_id),
            adaptive_search_stop_reason=final_adaptive_search_stop_reason,
        )
    except NoCertifiedRawReferenceError as failure:
        secondary_record = select_secondary_fallback(
            result_entries,
            require_global_refit=bool(require_global_secondary_refit),
        )
        if secondary_record is None:
            return DiagnosticOnlyResult(
                best_raw_attempt=_best_retained_raw_fit(result_entries),
                reason=f"{type(failure).__name__}: {failure}",
                report=SearchReport(
                    records=_ordered_search(result_entries, selected_candidate_id=None),
                    selected_id=None,
                    selection_method=str(selection_method),
                    adaptive_search_stop_reason=final_adaptive_search_stop_reason,
                    num_candidates_certified=0,
                    ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
                    selection_pool_stop_reason=str(selection_pool_stop_reason),
                ),
            )
        secondary = secondary_record.candidate
        if not isinstance(secondary, DirectPartitionCandidate):  # pragma: no cover
            raise AssertionError("Secondary selection returned a raw candidate.")
        secondary_eligible_count = sum(
            isinstance(record.candidate, DirectPartitionCandidate)
            and record.candidate.eligible_for_selection
            and record.candidate.refit.finite_candidate_found
            and np.isfinite(float(record.score.value))
            for record in result_entries
        )
        return SecondaryFallbackResult(
            selected_partition=secondary,
            best_raw_attempt=_best_retained_raw_fit(result_entries),
            reason=f"{type(failure).__name__}: {failure}",
            report=SearchReport(
                records=_ordered_search(
                    result_entries,
                    selected_candidate_id=int(secondary_record.candidate_id),
                ),
                selected_id=int(secondary_record.candidate_id),
                selection_method=str(selection_method),
                adaptive_search_stop_reason=final_adaptive_search_stop_reason,
                num_candidates_certified=int(secondary_eligible_count),
                ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
                selection_pool_stop_reason=str(selection_pool_stop_reason),
            ),
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
            observed_hash = pilot_matrix_hash(partition_parent_raw.raw_fit.phi)
            if not expected_hash or observed_hash != expected_hash:
                raise AssertionError(
                    "Direct partition parent-Phi provenance is inconsistent."
                )

    selection_boundary_unresolved = bool(
        decision.selection_boundary_unresolved or not candidate_pool_complete
    )
    selection_optimum_resolved = bool(
        raw_path_resolved and not selection_boundary_unresolved
    )
    selected_model = SelectedModel(
        raw_reference=raw_reference,
        partition_candidate=selected_candidate,
        partition_parent_raw=partition_parent_raw,
    )
    search = _ordered_search(
        result_entries,
        selected_candidate_id=int(selected_record.candidate_id),
    )
    return BICSelectionResult(
        selected_model=selected_model,
        report=SearchReport(
            records=search,
            selected_id=int(selected_record.candidate_id),
            selection_method=selection_method,
            adaptive_search_stop_reason=str(final_adaptive_search_stop_reason),
            num_candidates_certified=int(decision.num_eligible),
            selection_hits_lower_boundary=decision.selection_hits_lower_boundary,
            selection_hits_upper_boundary=decision.selection_hits_upper_boundary,
            selection_boundary_unresolved=selection_boundary_unresolved,
            selection_optimum_resolved=bool(selection_optimum_resolved),
            ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
            raw_lambda_path_resolved=bool(raw_path_resolved),
            global_hybrid_optimum_certified=bool(selection_optimum_resolved),
            selection_pool_stop_reason=str(selection_pool_stop_reason),
        ),
    )


def _restore_search_state(
    *,
    checkpoint_path: Path,
    checkpoint_identity: dict[str, object],
    mandatory_guide_work: WorkCounters,
    solver_context: SolverContext,
) -> SearchState:
    """Restore and validate one checkpoint before any search transaction."""

    loaded_checkpoint, checkpoint_generation = load_search_checkpoint(
        checkpoint_path,
        expected_identity=checkpoint_identity,
        return_generation=True,
    )
    if not isinstance(loaded_checkpoint, SearchCheckpoint):  # pragma: no cover
        raise AssertionError("Checkpoint loader returned the wrong schema type.")
    state = loaded_checkpoint.restore()
    state.checkpoint_generation = int(checkpoint_generation)
    loaded_mandatory_guide_work = state.mandatory_guide_work
    if not isinstance(loaded_mandatory_guide_work, WorkCounters):
        raise ValueError("Checkpoint mandatory guide work is invalid.")
    if loaded_mandatory_guide_work != mandatory_guide_work:
        raise ValueError(
            "Checkpoint mandatory guide work differs from deterministic "
            "guide reconstruction."
        )
    _post_guide_partition_refit_objective_evaluations(
        state.total_work,
        state.mandatory_guide_work,
    )
    loaded_direct_proposals = state.direct_pool.proposals
    if loaded_direct_proposals is not None:
        if not isinstance(loaded_direct_proposals, list):
            raise ValueError("Checkpoint direct proposal state is invalid.")
        if not all(
            isinstance(proposal, DirectProposal)
            for proposal in loaded_direct_proposals
        ):
            raise ValueError("Checkpoint direct proposal entry is invalid.")
    elif state.direct_pool.proposals_complete:
        raise ValueError("Checkpoint marks a missing direct proposal pool as complete.")
    if state.direct_pool.next_index < 0:
        raise ValueError("Checkpoint direct-candidate index is invalid.")
    if state.direct_pool.final_parent_next_index < 0:
        raise ValueError("Checkpoint direct-parent index is invalid.")
    if state.direct_pool.stop_reason is not None and not isinstance(
        state.direct_pool.stop_reason, str
    ):
        raise ValueError("Checkpoint direct-pool stop reason is invalid.")
    if (
        not np.isfinite(state.cumulative_search_active_seconds)
        or state.cumulative_search_active_seconds < 0.0
    ):
        raise ValueError("Checkpoint search elapsed time is invalid.")
    for expected_id, record in enumerate(state.candidates):
        if int(record.candidate_id) != expected_id:
            raise ValueError("Search checkpoint candidate IDs are not contiguous.")
    for run in state.lambdas.values():
        outcomes = (
            *((run.retained,) if run.retained is not None else ()),
            *run.attempts,
        )
        for outcome in outcomes:
            fit = outcome.fit
            if str(fit.provenance.objective_spec_hash) != str(
                solver_context.objective_spec_hash
            ) or str(fit.provenance.original_graph_hash) != str(
                solver_context.graph_hash
            ):
                raise ValueError(
                    "Checkpoint outcome belongs to a different objective or graph."
                )
            if outcome.state is not None and str(
                outcome.state.objective_spec_hash
            ) != str(solver_context.objective_spec_hash):
                raise ValueError(
                    "Checkpoint solver state belongs to a different objective."
                )
    return state


def _prepare_partition_guided_search(
    *,
    data: TumorData,
    fit_options: FitConfig,
    use_warm_starts: bool,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = False,
) -> tuple[_PartitionGuidedRuntime, SearchState]:
    """Prepare the deterministic guide, frozen problem, and resumable state.

    Ward/CEM supplies the primal start and initial-lambda scale. The contract
    declares whether the guide or zero-penalty pilot defines the frozen graph,
    and whether retained pilot/final-raw-phi partitions enter the secondary
    selection pool. Direct proposals are evaluated only after the raw lambda
    controller terminates, so they cannot steer or replace the raw optimizer.
    """

    search_segment_start = perf_counter()
    if int(data.num_mutations) < 2:
        raise ValueError(
            "partition_guided_admm requires at least two mutations so that a "
            "positive pairwise penalty is solved by ADMM."
        )
    pilot_context = prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy=str(fit_options.runtime.fallback),
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.solver.tolerance),
        defer_graph=True,
        inner_max_iter=max(int(fit_options.solver.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.graph.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.graph.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.graph.adaptive_weight_baseline),
        device=fit_options.runtime.device,
        dtype=fit_options.runtime.dtype,
        objective_shape=str(fit_options.solver.objective_shape),
    )
    pilot_phi: StartArray = pilot_context.exact_pilot
    pilot_runtime = pilot_context.runtime
    guide_curvature = observed_curvature_at_pilot_torch(
        data,
        pilot_phi,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        solver_context=pilot_context,
        device=pilot_runtime.device,
        dtype=pilot_runtime.dtype,
    )
    initializer_generation = generate_partition_initializer_pool(
        data=data,
        pilot_phi=pilot_phi,
        fit_options=fit_options,
        runtime=pilot_runtime,
        solver_context=pilot_context,
        rescore_candidates=rescore_partition_candidates,
        curvature=guide_curvature,
    )
    if not initializer_generation.complete:  # pragma: no cover - no guide cap
        raise AssertionError("The mandatory partition-guide pool was truncated.")
    initializer_pool = initializer_generation.candidates
    guide = best_partition_candidate(list(initializer_pool))
    if guide is None:
        raise RuntimeError(
            "No finite active-score partition initializer was available for tumor "
            f"{data.tumor_id}."
        )

    # Keep the partition guide host-backed for exact CPU behavior and fallback.
    # CUDA graph construction uploads this small M x S matrix once; the O(M^2)
    # graph itself stays device-backed and is reused by context preparation.
    guide_phi: StartArray = np.asarray(guide.phi_start)
    if fit_options.graph.graph is None:
        if PRODUCTION_SELECTION_POLICY.graph_pilot_source != "zero_penalty_pilot":
            raise AssertionError(
                "Production graph source must be the zero-penalty pilot."
            )
        graph_builder_phi = pilot_phi
        complete_graph_degree = float(max(int(data.num_mutations) - 1, 1))
        likelihood_noise_degree_exponent = float(
            PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT
        )
        likelihood_noise_divisor = float(
            complete_graph_degree**likelihood_noise_degree_exponent
        )
        selection_graph, prebuilt_tensor_graph, _ = (
            build_partition_guided_graph_with_resource_policy(
                guide_phi=graph_builder_phi,
                guide_curvature=guide_curvature,
                solver_context=pilot_context,
                fit_options=fit_options,
                noise_divisor=likelihood_noise_divisor,
            )
        )
    else:
        selection_graph = fit_options.graph.graph
        prebuilt_tensor_graph = None
    base_solver_context = prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy=str(fit_options.runtime.fallback),
        inherited_resource_fallback=pilot_context.resource_fallback,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.solver.tolerance),
        # The guide initializes adaptive weights, but observed curvature and a
        # mild degree correction set a finite data-derived distance floor. This
        # prevents the fixed 1e-6 floor from making the proposed blocks
        # effectively immutable while retaining the current estimator as the
        # requested initializer.
        graph=selection_graph,
        prebuilt_tensor_graph=prebuilt_tensor_graph,
        inner_max_iter=max(int(fit_options.solver.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.graph.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.graph.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.graph.adaptive_weight_baseline),
        # Keep likelihood-derived starts independent of the Ward guide so
        # nominal cold retries explore distinct non-convex basins.
        exact_pilot=pilot_context.exact_pilot,
        pooled_start=pilot_context.pooled_start,
        scalar_well_starts=pilot_context.scalar_well_starts,
        device=fit_options.runtime.device,
        dtype=fit_options.runtime.dtype,
        runtime=pilot_runtime,
        objective_shape=str(fit_options.solver.objective_shape),
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
        build_guided_initialization_with_resource_policy(
            data=data,
            guide_phi=raw_guide_phi,
            guide_labels=raw_guide_labels,
            solver_context=base_solver_context,
            fit_options=effective_fit_options,
        )
    )
    guided_initialization = replace(
        guided_initialization,
        solver_state=offload_solver_state_to_cpu(guided_initialization.solver_state),
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
            no_progress_patience=int(
                effective_fit_options.selection.lambda_search.no_progress_patience
            ),
        ),
    )

    mandatory_guide_work = initializer_generation.work
    state = SearchState(
        controller=controller,
        lambdas={},
        candidates=[],
        bic_refit_cache={},
        work_ledger=WorkLedger(mandatory_guide_work),
        mandatory_guide_work=mandatory_guide_work,
        search_stop_override=None,
        direct_pool=DirectPoolState(),
        raw_guide_phi=raw_guide_phi,
        guided_initialization=guided_initialization,
        float64_recovery_status="not_requested",
        cumulative_search_active_seconds=0.0,
    )
    checkpoint_identity = (
        None
        if checkpoint_path is None
        else build_search_checkpoint_identity(
            data=data,
            fit_config=effective_fit_options,
            objective_spec_hash=str(base_solver_context.objective_spec_hash),
            original_graph_hash=str(base_solver_context.graph_hash),
            use_warm_starts=bool(use_warm_starts),
            runtime_device_name=str(base_solver_context.runtime.device_name),
            runtime_dtype=str(base_solver_context.runtime.dtype),
        )
    )

    if resume_checkpoint:
        if checkpoint_path is None:
            raise ValueError("resume_checkpoint requires a checkpoint path.")
        if checkpoint_identity is None:  # pragma: no cover - construction guard
            raise AssertionError("Checkpoint path has no identity guard.")
        state = _restore_search_state(
            checkpoint_path=checkpoint_path,
            checkpoint_identity=checkpoint_identity,
            mandatory_guide_work=mandatory_guide_work,
            solver_context=base_solver_context,
        )

    runtime_context = _PartitionGuidedRuntime(
        data=data,
        fit_options=effective_fit_options,
        solver_context=base_solver_context,
        initializer_pool=tuple(initializer_pool),
        checkpoint_path=checkpoint_path,
        checkpoint_identity=checkpoint_identity,
        search_segment_start=search_segment_start,
        use_warm_starts=bool(use_warm_starts),
        resume_checkpoint=bool(resume_checkpoint),
    )
    return runtime_context, state


def _candidate_fit_options_for_proposal(
    context: _PartitionGuidedRuntime,
    proposal: OnlineLambdaProposal,
) -> FitConfig:
    """Resolve retry effort without changing the fixed model contract."""

    effective_fit_options = context.fit_options
    if proposal.phase in {
        "solver_recovery",
        "bootstrap_certification_anchor",
    }:
        return solver_recovery_fit_options(
            context.solver_context.source_model,
            effective_fit_options,
            retry_number=int(proposal.retry_number),
        )
    if proposal.retry_number <= 0:
        return effective_fit_options
    effort_factor = int(proposal.retry_number) + 1
    base_solver = effective_fit_options.solver
    return replace(
        effective_fit_options,
        solver=replace(
            base_solver,
            outer_max_iter=max(
                int(base_solver.outer_max_iter) * effort_factor,
                int(base_solver.outer_max_iter),
            ),
            inner_max_iter=max(
                int(base_solver.inner_max_iter) * effort_factor,
                int(base_solver.inner_max_iter),
            ),
            # Retry effort changes iteration budgets, not the model's
            # numerical admission contract.
            tolerance=float(base_solver.tolerance),
        ),
    )


def _recovery_solver_context(
    base_context: SolverContext,
    state: SearchState,
    proposal: OnlineLambdaProposal,
) -> tuple[SolverContext, bool, str]:
    """Resolve the fixed-objective precision context for one raw proposal."""

    if proposal.phase not in {
        "solver_recovery",
        "bootstrap_certification_anchor",
    }:
        return base_context, False, "not_requested"
    if base_context.runtime.dtype == torch.float64:
        return base_context, False, "not_needed"
    if state.float64_recovery_context is None and str(
        state.float64_recovery_status
    ) not in {"failed_memory", "failed_runtime"}:
        (
            state.float64_recovery_context,
            state.float64_recovery_status,
        ) = _try_promote_recovery_context(base_context)
    promotion_status = str(state.float64_recovery_status)
    if (
        state.float64_recovery_context is not None
        and state.float64_recovery_context is not base_context
    ):
        return state.float64_recovery_context, True, promotion_status
    return base_context, False, promotion_status


def _append_distinct_raw_start(
    start_specs: list[RawStartSpec],
    seen_start_states: set[tuple[object, ...]],
    *,
    source: str,
    start_value: float,
    state: SolverState | None,
    phi: StartArray | None = None,
) -> None:
    """Append one raw start unless the same primal/state is already present."""

    if state is None:
        if phi is None:
            raise ValueError("A cold raw start requires an explicit Phi.")
        identity: tuple[object, ...] = ("cold", pilot_matrix_hash(phi))
    else:
        identity = ("state", solver_state_content_key(state))
    if identity in seen_start_states:
        return
    seen_start_states.add(identity)
    start_specs.append((str(source), float(start_value), state, phi))


def _raw_start_specs(
    solver_context: SolverContext,
    state: SearchState,
    proposal: OnlineLambdaProposal,
    *,
    lambda_key: float,
    use_warm_starts: bool,
) -> list[RawStartSpec]:
    """Build the ordered, distinct start bank for one raw proposal."""

    initialization = state.guided_initialization
    warm_outcome = None
    if proposal.warm_start_lambda is not None:
        warm_run = state.lambdas.get(canonical_lambda(proposal.warm_start_lambda))
        warm_outcome = None if warm_run is None else warm_run.retained
    alternate_outcome = None
    if proposal.alternate_start_lambda is not None:
        alternate_run = state.lambdas.get(
            canonical_lambda(proposal.alternate_start_lambda)
        )
        alternate_outcome = None if alternate_run is None else alternate_run.retained
    same_lambda_attempts = state.lambda_run(lambda_key).attempts
    finite_failed = [
        attempt
        for attempt in same_lambda_attempts
        if attempt.state is not None
        and np.isfinite(float(attempt.fit.certificate.components.residual))
    ]
    start_specs: list[RawStartSpec] = []
    seen_start_states: set[tuple[object, ...]] = set()

    def append_start(
        source: str,
        start_value: float,
        solver_state: SolverState | None,
        phi: StartArray | None = None,
    ) -> None:
        _append_distinct_raw_start(
            start_specs,
            seen_start_states,
            source=source,
            start_value=start_value,
            state=solver_state,
            phi=phi,
        )

    if proposal.phase == "solver_recovery":
        if finite_failed:
            best_failed_fit = min(
                finite_failed,
                key=lambda attempt: float(
                    attempt.fit.certificate.components.residual
                ),
            )
            append_start(
                "best_same_lambda_kkt_state",
                float(best_failed_fit.fit.provenance.lambda_value),
                best_failed_fit.state,
            )
        else:
            append_start(
                "guided_kkt_solver_recovery",
                float(initialization.lambda_value),
                initialization.solver_state,
            )
    elif int(proposal.retry_number) > 0:
        if (
            use_warm_starts
            and int(proposal.retry_number) == 1
            and alternate_outcome is not None
            and alternate_outcome.state is not None
        ):
            append_start(
                "alternate_bracket_endpoint",
                float(proposal.alternate_start_lambda),
                alternate_outcome.state,
            )
        elif (
            use_warm_starts
            and warm_outcome is not None
            and warm_outcome.state is not None
        ):
            append_start(
                "same_lambda_retry",
                float(proposal.warm_start_lambda),
                warm_outcome.state,
            )
        else:
            append_start(
                "guided_kkt_fallback",
                float(initialization.lambda_value),
                initialization.solver_state,
            )
    else:
        if proposal.phase == "bootstrap_certification_anchor":
            append_start(
                "guided_kkt_bootstrap_anchor",
                float(initialization.lambda_value),
                initialization.solver_state,
            )
            for source, start_value, solver_state, phi in (
                bootstrap_independent_start_specs(
                    initial_lambda=float(initialization.lambda_value),
                    raw_guide_phi=state.raw_guide_phi,
                    exact_pilot=solver_context.exact_pilot,
                    pooled_start=solver_context.pooled_start,
                    suffix="bootstrap_anchor",
                )
            ):
                append_start(source, start_value, solver_state, phi)
        # Partition-event midpoints compete both bracket endpoints with the
        # fixed guided/cold starts. The bounded bank only applies at event probes.
        if (
            proposal.phase != "bootstrap_certification_anchor"
            and use_warm_starts
            and warm_outcome is not None
            and warm_outcome.state is not None
        ):
            append_start(
                "warm_bracket_left"
                if proposal.phase == "refine_partition_event"
                else "warm_endpoint",
                float(proposal.warm_start_lambda),
                warm_outcome.state,
            )
        if (
            proposal.phase == "refine_partition_event"
            and use_warm_starts
            and alternate_outcome is not None
            and alternate_outcome.state is not None
        ):
            append_start(
                "warm_bracket_right",
                float(proposal.alternate_start_lambda),
                alternate_outcome.state,
            )
        if proposal.phase == "refine_partition_event":
            append_start(
                "guided_kkt_multistart",
                float(initialization.lambda_value),
                initialization.solver_state,
            )
            append_start(
                "cold_partition_guide",
                float(initialization.lambda_value),
                None,
                state.raw_guide_phi,
            )
            append_start(
                "cold_zero_penalty_pilot",
                0.0,
                None,
                solver_context.exact_pilot,
            )
            append_start(
                "cold_pooled_likelihood",
                0.0,
                None,
                solver_context.pooled_start,
            )
        elif not start_specs:
            append_start(
                "guided_kkt_state"
                if proposal.phase == "initial"
                else "guided_kkt_fallback",
                float(initialization.lambda_value),
                initialization.solver_state,
            )

        # Compete independent primals when lower-lambda continuation escapes K=1.
        warm_key = (
            None
            if proposal.warm_start_lambda is None
            else canonical_lambda(proposal.warm_start_lambda)
        )
        escaping_k1_basin = bool(
            int(proposal.retry_number) == 0
            and proposal.phase != "refine_partition_event"
            and warm_key is not None
            and state.lambdas.get(warm_key) is not None
            and state.lambdas[warm_key].partition_k == 1
            and float(proposal.lambda_value) < float(proposal.warm_start_lambda)
        )
        if escaping_k1_basin:
            append_start(
                "cold_partition_guide_k1_escape",
                float(initialization.lambda_value),
                None,
                state.raw_guide_phi,
            )
            append_start(
                "cold_zero_penalty_k1_escape",
                0.0,
                None,
                solver_context.exact_pilot,
            )
            append_start(
                "cold_pooled_likelihood_k1_escape",
                0.0,
                None,
                solver_context.pooled_start,
            )

    if solver_context.source_model.requires_generic_path_solver:
        for source, start_value, solver_state, phi in explicit_path_default_start_specs(
            scalar_well_starts=solver_context.scalar_well_starts,
            pooled_start=solver_context.pooled_start,
        ):
            append_start(source, start_value, solver_state, phi)
    return start_specs


def _solve_raw_proposal(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    proposal: OnlineLambdaProposal,
    *,
    lambda_key: float,
    candidate_fit_options: FitConfig,
) -> tuple[SolveOutcome, RawStartAttempt, tuple[RawStartAttempt, ...]]:
    """Run the ordered start bank for one fixed-lambda raw transaction."""

    solver_context, recovery_promoted, promotion_status = _recovery_solver_context(
        context.solver_context,
        state,
        proposal,
    )
    start_specs = _raw_start_specs(
        solver_context,
        state,
        proposal,
        lambda_key=lambda_key,
        use_warm_starts=context.use_warm_starts,
    )
    start_attempts: list[RawStartAttempt] = []
    proposal_work = WorkLedger()
    for (
        lambda_start_source,
        lambda_start_value,
        original_state,
        explicit_phi_start,
    ) in start_specs:
        tumor_work_cap = (
            context.fit_options.solver.resources.max_tumor_edge_pass_equivalents
        )
        remaining_before_escape: int | None = None
        if tumor_work_cap is not None:
            remaining_before_escape = int(tumor_work_cap) - int(
                state.total_work.edge_pass_equivalents
                + proposal_work.total.edge_pass_equivalents
            )
            if remaining_before_escape <= 0 and start_attempts:
                break
        if recovery_promoted:
            # Keep only the primal across the precision boundary; the float64
            # solve reconstructs dual/certificate state before certification.
            solver_state_start, changed_count = None, 0
            escape_work = WorkCounters()
            phi_start = clone_start(
                original_state.phi
                if original_state is not None and original_state.phi is not None
                else (
                    explicit_phi_start
                    if explicit_phi_start is not None
                    else state.raw_guide_phi
                )
            )
        else:
            # Preserve the last soft-cap pass for the fit rather than spending
            # it on an escape that cannot be followed within the aggregate cap.
            if remaining_before_escape is not None and remaining_before_escape <= 1:
                solver_state_start, changed_count = original_state, 0
                escape_work = WorkCounters()
            else:
                (
                    solver_state_start,
                    changed_count,
                    escape_work,
                ) = escape_path_breakpoint_retry_state(
                    original_state,
                    start_source=lambda_start_source,
                    start_lambda=lambda_start_value,
                    target_lambda=float(proposal.lambda_value),
                    context=solver_context,
                    tol=float(candidate_fit_options.solver.tolerance),
                )
            phi_start = clone_start(
                solver_state_start.phi
                if solver_state_start is not None and solver_state_start.phi is not None
                else (
                    explicit_phi_start
                    if explicit_phi_start is not None
                    else state.raw_guide_phi
                )
            )
        attempt_budget = SolveBudget()
        if tumor_work_cap is not None:
            remaining_work = int(tumor_work_cap) - int(
                state.total_work.edge_pass_equivalents
                + proposal_work.total.edge_pass_equivalents
                + escape_work.edge_pass_equivalents
            )
            attempt_budget = SolveBudget(
                max_edge_pass_equivalents=max(int(remaining_work), 1)
            )
        seed_fit = fit_fixed_objective(
            data=context.data,
            config=candidate_fit_options,
            lambda_value=float(proposal.lambda_value),
            init=SolverInit(
                phi_start=phi_start,
                solver_context=solver_context,
                solver_state=solver_state_start,
                start_mode="warm_only",
                append_default_nonconvex_starts=False,
            ),
            budget=attempt_budget,
        )
        # Keep pre-solve breakpoint work on its attempt so local and aggregate
        # cap routing both see it.
        seed_fit = replace(seed_fit, work=seed_fit.work + escape_work)
        if str(seed_fit.provenance.objective_spec_hash) != str(
            solver_context.objective_spec_hash
        ):
            raise AssertionError("Raw multistart changed the fixed objective identity.")
        solve_outcome = _detach_solve_outcome(seed_fit)
        proposal_work.charge(seed_fit.work)
        state.lambda_run(lambda_key).attempts.append(solve_outcome)
        # Use the same full schema-2 admission predicate as scoring/controller.
        mathematically_certified = bool(
            raw_fit_has_exact_fusion_certificate(seed_fit)
        )
        start_attempts.append(
            RawStartAttempt(
                outcome=solve_outcome,
                source=str(lambda_start_source),
                start_value=float(lambda_start_value),
                breakpoint_escape_changed_count=int(changed_count),
                mathematically_certified=bool(mathematically_certified),
                promotion_status=str(promotion_status),
            )
        )
        if tumor_work_cap is not None and int(
            state.total_work.edge_pass_equivalents
            + proposal_work.total.edge_pass_equivalents
        ) >= int(tumor_work_cap):
            break
    selected_attempt = select_raw_start_attempt(start_attempts)
    # Warm-start from the same basin admitted to partition scoring.
    return selected_attempt.outcome, selected_attempt, tuple(start_attempts)


def _record_raw_proposal_transaction(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    proposal: OnlineLambdaProposal,
    *,
    lambda_key: float,
    candidate_fit_options: FitConfig,
    selected_outcome: SolveOutcome,
    selected_start: RawStartAttempt,
    raw_start_attempts: tuple[RawStartAttempt, ...],
) -> None:
    """Score, observe, and checkpoint one completed raw-lambda proposal."""

    for raw_attempt in raw_start_attempts:
        state.charge(raw_attempt.fit.work)
    fit = selected_outcome.fit
    artifact = candidate_from_raw_fit(
        data=context.data,
        raw_fit=fit,
        selection_options=context.fit_options,
        refit_cache=state.bic_refit_cache,
        source_model=context.solver_context.source_model,
    )
    state.charge(artifact.work)
    selected_trace_fit = replace(fit, work=fit.work + artifact.work)
    candidate_id = int(len(state.candidates))
    artifact = _compact_raw_candidate(artifact)
    state.candidates.append(
        CandidateRecord(
            candidate_id=candidate_id,
            candidate=artifact,
            trace=CandidateTrace(
                search_round=int(state.next_candidate_id),
                search_phase=str(proposal.phase),
                start_source=str(selected_start.source),
                start_value=float(selected_start.start_value),
                breakpoint_escape_changed_count=int(
                    selected_start.breakpoint_escape_changed_count
                ),
                raw_attempts=tuple(
                    RawAttemptSummary.from_fit(
                        (
                            selected_trace_fit
                            if attempt is selected_start
                            else attempt.fit
                        ),
                        source=str(attempt.source),
                        start_value=float(attempt.start_value),
                        breakpoint_escape_changed_count=int(
                            attempt.breakpoint_escape_changed_count
                        ),
                        mathematically_certified=bool(
                            attempt.mathematically_certified
                        ),
                        outer_max_iter=int(
                            candidate_fit_options.solver.outer_max_iter
                        ),
                        inner_max_iter=int(
                            candidate_fit_options.solver.inner_max_iter
                        ),
                        certificate_max_iter=int(
                            candidate_fit_options.solver.certificate.max_iter
                        ),
                        promotion_status=str(attempt.promotion_status),
                    )
                    for attempt in raw_start_attempts
                ),
            ),
        )
    )
    lambda_run = state.lambda_run(lambda_key)
    incumbent_fit = None if lambda_run.retained is None else lambda_run.retained.fit
    if prefer_fit_candidate(fit, incumbent_fit):
        lambda_run.retained = (
            replace(selected_outcome, state=None)
            if (
                selected_outcome.state is not None
                and str(fit.provenance.dtype) == "float64"
                and context.solver_context.runtime.dtype != torch.float64
            )
            else selected_outcome
        )
        lambda_run.partition_k = int(artifact.partition.n_clusters)

    raw_exact_certified = bool(
        raw_fit_has_exact_fusion_certificate(fit)
        and bool(context.solver_context.graph.is_complete)
    )
    score = artifact.score
    selection_score_available = bool(
        isinstance(artifact, RawFusionCandidate)
        and artifact.eligible_for_selection
        and score is not None
    )
    state.controller.observe(
        OnlineLambdaObservation(
            lambda_value=float(proposal.lambda_value),
            n_clusters=int(artifact.partition.n_clusters),
            partition_signature=str(artifact.partition.signature),
            partition_bic=(
                float(score.value) if selection_score_available else float("inf")
            ),
            kkt_residual=float(fit.certificate.components.residual),
            raw_objective_certified=bool(raw_exact_certified),
            partition_certified=bool(artifact.partition.certified),
            selection_score_available=selection_score_available,
            score_numerical_uncertainty=(
                float(score.numerical_uncertainty)
                if selection_score_available
                else 0.0
            ),
            degrees_of_freedom=(
                int(score.degrees_of_freedom) if selection_score_available else 0
            ),
        )
    )
    resources = context.fit_options.solver.resources
    work_cap = resources.max_tumor_edge_pass_equivalents
    if work_cap is not None and int(state.total_work.edge_pass_equivalents) >= int(
        work_cap
    ):
        state.search_stop_override = "tumor_work_budget_reached"
    refit_work_cap = resources.max_partition_refit_objective_evaluations
    if (
        state.search_stop_override is None
        and refit_work_cap is not None
        and _post_guide_partition_refit_objective_evaluations(
            state.total_work,
            state.mandatory_guide_work,
        )
        >= int(refit_work_cap)
    ):
        state.search_stop_override = (
            "partition_refit_objective_evaluation_budget_reached"
        )
    _save_search_state(
        path=context.checkpoint_path,
        identity=context.checkpoint_identity,
        state=state,
        segment_started_at=context.search_segment_start,
    )


def _run_raw_lambda_path(
    context: _PartitionGuidedRuntime,
    state: SearchState,
) -> str:
    """Evaluate the checkpointed raw-fusion lambda path in controller order."""

    while True:
        if state.search_stop_override is not None:
            break
        proposal = state.controller.propose()
        if proposal is None:
            break
        lambda_key = canonical_lambda(proposal.lambda_value)
        state.discard_transient_attempts_except(lambda_key)
        candidate_fit_options = _candidate_fit_options_for_proposal(
            context,
            proposal,
        )
        selected_outcome, selected_start, raw_start_attempts = _solve_raw_proposal(
            context,
            state,
            proposal,
            lambda_key=lambda_key,
            candidate_fit_options=candidate_fit_options,
        )
        _record_raw_proposal_transaction(
            context,
            state,
            proposal,
            lambda_key=lambda_key,
            candidate_fit_options=candidate_fit_options,
            selected_outcome=selected_outcome,
            selected_start=selected_start,
            raw_start_attempts=raw_start_attempts,
        )

    # A terminal controller reason is established by ``propose()`` after the
    # preceding observation, so persist that final scalar state as well.
    _save_search_state(
        path=context.checkpoint_path,
        identity=context.checkpoint_identity,
        state=state,
        segment_started_at=context.search_segment_start,
    )

    return str(
        state.search_stop_override
        or state.controller.stop_reason
        or "online_lambda_no_terminal_reason"
    )


def _initialize_direct_pool_transaction(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    *,
    final_pool_required: bool,
) -> None:
    """Persist the immutable pilot proposal generation transaction."""

    direct = state.direct_pool
    direct.proposals = [
        DirectProposal(
            candidate=proposal,
            stage="pilot",
            parent_raw_candidate_id=None,
        )
        for proposal in context.initializer_pool
    ]
    direct.proposals_complete = not final_pool_required
    direct.final_parent_next_index = 0
    # Pilot proposals are the first immutable generation transaction.
    # Evaluate them before spending any optional final-Phi generation
    # work, then append one complete parent batch at a time.
    _save_search_state(
        path=context.checkpoint_path,
        identity=context.checkpoint_identity,
        state=state,
        segment_started_at=context.search_segment_start,
    )


def _validate_direct_pool_progress(state: SearchState) -> None:
    """Check that retained direct records are the exact proposal prefix."""

    direct = state.direct_pool
    if direct.proposals is None:  # pragma: no cover - transaction invariant
        raise AssertionError("Direct proposal pool has not been initialized.")
    completed_direct_records = [
        record
        for record in state.candidates
        if isinstance(record.candidate, DirectPartitionCandidate)
    ]
    if len(completed_direct_records) != int(direct.next_index):
        raise ValueError(
            "Checkpoint direct-candidate progress does not match retained records."
        )
    if int(direct.next_index) > len(direct.proposals):
        raise ValueError(
            "Checkpoint direct-candidate index exceeds the retained proposal pool."
        )
    for direct_index, retained_record in enumerate(completed_direct_records):
        direct_proposal = direct.proposals[direct_index]
        proposal = direct_proposal.candidate
        parent_candidate_id = direct_proposal.parent_raw_candidate_id
        source = direct_partition_source(proposal, stage=direct_proposal.stage)
        retained = retained_record.candidate
        if (
            str(retained.partition.source) != str(source)
            or retained.partition.parent_raw_candidate_id != parent_candidate_id
            or not np.array_equal(
                np.asarray(retained.partition.labels, dtype=np.int64),
                np.asarray(proposal.labels, dtype=np.int64),
            )
        ):
            raise ValueError("Retained direct pool does not match checkpoint progress.")


def _direct_pool_budget_stop_reason(
    context: _PartitionGuidedRuntime,
    state: SearchState,
) -> str | None:
    """Return the first exhausted direct-pool resource in policy order."""

    resources = context.fit_options.solver.resources
    candidate_cap = resources.max_direct_partition_candidates
    if candidate_cap is not None and int(state.direct_pool.next_index) >= int(
        candidate_cap
    ):
        return "direct_partition_candidate_budget_reached"
    refit_cap = resources.max_partition_refit_objective_evaluations
    if (
        refit_cap is not None
        and _post_guide_partition_refit_objective_evaluations(
            state.total_work,
            state.mandatory_guide_work,
        )
        >= int(refit_cap)
    ):
        return "partition_refit_objective_evaluation_budget_reached"
    return None


def _evaluate_direct_proposal_transaction(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    *,
    records_by_id: dict[int, CandidateRecord],
) -> None:
    """Evaluate and checkpoint exactly one generated direct proposal."""

    direct = state.direct_pool
    if direct.proposals is None:  # pragma: no cover - transaction invariant
        raise AssertionError("Direct proposal pool has not been initialized.")
    direct_proposal = direct.proposals[int(direct.next_index)]
    proposal = direct_proposal.candidate
    stage = direct_proposal.stage
    parent_candidate_id = direct_proposal.parent_raw_candidate_id
    parent_record = (
        None
        if parent_candidate_id is None
        else records_by_id.get(int(parent_candidate_id))
    )
    if parent_candidate_id is not None and parent_record is None:
        raise ValueError("Checkpoint direct proposal refers to a missing raw parent.")
    parent_candidate = None if parent_record is None else parent_record.candidate
    parent_raw = (
        parent_candidate
        if isinstance(parent_candidate, RawFusionCandidate)
        else None
    )
    source = direct_partition_source(proposal, stage=stage)
    candidate_id = int(len(state.candidates))
    direct_candidate = evaluate_direct_partition_candidate(
        data=context.data,
        proposal=proposal,
        selection_options=context.fit_options,
        source=source,
        parent_raw_candidate_id=parent_candidate_id,
        parent_raw_lambda=(
            None
            if parent_raw is None
            else float(parent_raw.raw_fit.provenance.lambda_value)
        ),
        parent_raw_phi_hash=(
            "" if parent_raw is None else pilot_matrix_hash(parent_raw.raw_fit.phi)
        ),
        refit_cache=state.bic_refit_cache,
        source_model=context.solver_context.source_model,
    )
    state.charge(direct_candidate.work)
    state.candidates.append(
        CandidateRecord(
            candidate_id=candidate_id,
            candidate=direct_candidate,
            trace=CandidateTrace(
                search_round=int(state.next_candidate_id),
                search_phase=f"{stage}_direct_partition_pool",
            ),
        )
    )
    direct.next_index += 1
    _save_search_state(
        path=context.checkpoint_path,
        identity=context.checkpoint_identity,
        state=state,
        segment_started_at=context.search_segment_start,
    )


def _generate_final_phi_pool_transaction(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    *,
    raw_parent_records: list[CandidateRecord],
    final_k_grid: tuple[int, ...],
) -> None:
    """Generate and checkpoint one complete final-Phi parent batch."""

    direct = state.direct_pool
    if direct.proposals is None:  # pragma: no cover - transaction invariant
        raise AssertionError("Direct proposal pool has not been initialized.")
    resources = context.fit_options.solver.resources
    direct_candidate_cap = resources.max_direct_partition_candidates
    direct_refit_cap = resources.max_partition_refit_objective_evaluations
    remaining_direct_capacity = (
        None
        if direct_candidate_cap is None
        else max(int(direct_candidate_cap) - int(direct.next_index), 0)
    )
    remaining_refit_capacity = (
        None
        if direct_refit_cap is None
        else max(
            int(direct_refit_cap)
            - _post_guide_partition_refit_objective_evaluations(
                state.total_work,
                state.mandatory_guide_work,
            ),
            0,
        )
    )
    parent_record = raw_parent_records[int(direct.final_parent_next_index)]
    parent = parent_record.candidate
    if not isinstance(parent, RawFusionCandidate):  # pragma: no cover
        raise AssertionError("Final-Phi parent is not a raw candidate.")
    final_generation = generate_partition_initializer_pool(
        data=context.data,
        pilot_phi=np.asarray(parent.raw_fit.phi, dtype=np.float64),
        fit_options=context.fit_options,
        runtime=context.solver_context.runtime,
        solver_context=context.solver_context,
        rescore_candidates=rescore_partition_candidates,
        declared_k_grid=final_k_grid,
        max_refit_objective_evaluations=remaining_refit_capacity,
        max_candidates=remaining_direct_capacity,
    )
    state.charge(final_generation.work)
    direct.proposals.extend(
        DirectProposal(
            candidate=proposal,
            stage="final_phi",
            parent_raw_candidate_id=int(parent_record.candidate_id),
        )
        for proposal in final_generation.candidates
    )
    if final_generation.complete:
        direct.final_parent_next_index += 1
        direct.proposals_complete = bool(
            int(direct.final_parent_next_index) >= len(raw_parent_records)
        )
    elif final_generation.stop_reason == "candidate_budget_reached":
        direct.stop_reason = "direct_partition_candidate_budget_reached"
    elif final_generation.stop_reason == "refit_objective_evaluation_budget_reached":
        direct.stop_reason = "partition_refit_objective_evaluation_budget_reached"
    else:  # pragma: no cover - typed generation invariant
        raise AssertionError("Unknown partition-generation stop reason.")
    # Generation is an atomic checkpoint transaction. If interrupted before
    # this write, resume deterministically regenerates the same parent pool;
    # after it, only its unevaluated suffix remains.
    _save_search_state(
        path=context.checkpoint_path,
        identity=context.checkpoint_identity,
        state=state,
        segment_started_at=context.search_segment_start,
    )


def _run_direct_candidate_pool(
    context: _PartitionGuidedRuntime,
    state: SearchState,
) -> tuple[bool, DirectPoolState]:
    """Evaluate the deterministic pilot and final-Phi direct proposal pool."""

    data = context.data
    effective_fit_options = context.fit_options
    direct = state.direct_pool
    ward_candidate_pool_complete = bool(direct.complete)
    direct_resources = effective_fit_options.solver.resources
    direct_refit_cap = direct_resources.max_partition_refit_objective_evaluations
    direct_budget_exhausted_before_pool = bool(
        direct_refit_cap is not None
        and _post_guide_partition_refit_objective_evaluations(
            state.total_work,
            state.mandatory_guide_work,
        )
        >= int(direct_refit_cap)
    )
    if direct_budget_exhausted_before_pool:
        direct.stop_reason = "partition_refit_objective_evaluation_budget_reached"
    if not direct.complete and not direct_budget_exhausted_before_pool:
        records_by_id = {
            int(record.candidate_id): record for record in state.candidates
        }
        config = PRODUCTION_SELECTION_POLICY.partition_config
        raw_parent_records = sorted(
            (
                record
                for record in state.candidates
                if isinstance(record.candidate, RawFusionCandidate)
                and raw_candidate_has_exact_fusion_certificate(record.candidate)
            ),
            key=lambda record: (
                float(record.score.value),
                float(record.score.numerical_uncertainty),
                float(record.candidate.raw_fit.provenance.lambda_value),
                int(record.candidate_id),
            ),
        )[:1]
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
        final_pool_required = bool(final_k_grid and raw_parent_records)
        if int(direct.final_parent_next_index) > len(raw_parent_records):
            raise ValueError(
                "Checkpoint direct-parent index exceeds the deterministic parent pool."
            )
        if direct.proposals is None:
            _initialize_direct_pool_transaction(
                context,
                state,
                final_pool_required=final_pool_required,
            )
        _validate_direct_pool_progress(state)

        while not direct.complete:
            # Consume every already-generated proposal before generating an
            # optional final-Phi parent batch. The fixed pilot-then-parent
            # order makes every checkpoint resumable.
            while int(direct.next_index) < len(direct.proposals):
                budget_stop_reason = _direct_pool_budget_stop_reason(context, state)
                if budget_stop_reason is not None:
                    direct.stop_reason = budget_stop_reason
                    break
                _evaluate_direct_proposal_transaction(
                    context,
                    state,
                    records_by_id=records_by_id,
                )

            if int(direct.next_index) < len(direct.proposals):
                break
            if direct.stop_reason is not None:
                break
            if direct.proposals_complete:
                direct.complete = True
                break

            budget_stop_reason = _direct_pool_budget_stop_reason(context, state)
            if budget_stop_reason is not None:
                direct.stop_reason = budget_stop_reason
                break
            if int(direct.final_parent_next_index) >= len(raw_parent_records):
                direct.proposals_complete = True
                _save_search_state(
                    path=context.checkpoint_path,
                    identity=context.checkpoint_identity,
                    state=state,
                    segment_started_at=context.search_segment_start,
                )
                continue
            _generate_final_phi_pool_transaction(
                context,
                state,
                raw_parent_records=raw_parent_records,
                final_k_grid=final_k_grid,
            )

        ward_candidate_pool_complete = bool(direct.complete)
        _save_search_state(
            path=context.checkpoint_path,
            identity=context.checkpoint_identity,
            state=state,
            segment_started_at=context.search_segment_start,
        )

    # Persist a boundary stop even when the budget was already exhausted by
    # raw-candidate refits before pool generation.
    if direct_budget_exhausted_before_pool:
        _save_search_state(
            path=context.checkpoint_path,
            identity=context.checkpoint_identity,
            state=state,
            segment_started_at=context.search_segment_start,
        )

    return bool(ward_candidate_pool_complete), direct


def _finalize_partition_guided_selection(
    context: _PartitionGuidedRuntime,
    state: SearchState,
    *,
    raw_search_stop_reason: str,
    ward_candidate_pool_complete: bool,
    direct: DirectPoolState,
) -> TumorSelectionOutcome:
    """Select from the completed pool and attach aggregate search provenance."""

    data = context.data
    fit_options = context.fit_options
    search_segment_start = context.search_segment_start
    resume_checkpoint = context.resume_checkpoint

    if not state.candidates:
        raise RuntimeError(
            f"No guided ADMM candidates were evaluated for tumor {data.tumor_id}."
        )
    # Preserve the raw controller's terminal provenance.  Direct-pool
    # truncation has its own typed field and must not masquerade as an adaptive
    # lambda-search stop.
    stop_reason = str(raw_search_stop_reason)
    outcome = _assemble_selection_result(
        data=data,
        result_entries=state.candidates,
        selection_method=_SELECTION_METHOD,
        adaptive_search_stop_reason=stop_reason,
        strict_positive_exact_fusion=False,
        ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
        require_global_secondary_refit=fit_options.profile_name == "strict",
        candidate_pool_complete=bool(ward_candidate_pool_complete),
        raw_lambda_path_resolved=bool(
            adaptive_stop_certifies_global_optimum(raw_search_stop_reason)
        ),
        selection_pool_stop_reason=str(direct.stop_reason or "none"),
    )
    return replace(
        outcome,
        report=replace(
            outcome.report,
            search_work=state.total_work,
            mandatory_guide_work=state.mandatory_guide_work,
            cumulative_search_active_seconds=float(
                state.cumulative_search_active_seconds
                + perf_counter()
                - search_segment_start
            ),
            resumed_from_checkpoint=bool(resume_checkpoint),
        ),
    )


def _partition_guided_admm_selection(
    *,
    data: TumorData,
    fit_options: FitConfig,
    use_warm_starts: bool,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = False,
) -> TumorSelectionOutcome:
    """Run the four ordered phases of the immutable production selection."""

    context, state = _prepare_partition_guided_search(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        checkpoint_path=checkpoint_path,
        resume_checkpoint=resume_checkpoint,
    )
    raw_search_stop_reason = _run_raw_lambda_path(context, state)
    ward_candidate_pool_complete, direct = _run_direct_candidate_pool(context, state)
    return _finalize_partition_guided_selection(
        context,
        state,
        raw_search_stop_reason=raw_search_stop_reason,
        ward_candidate_pool_complete=ward_candidate_pool_complete,
        direct=direct,
    )


def select_model(
    *,
    data: TumorData,
    fit_config: FitConfig,
    use_warm_starts: bool,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: bool = False,
) -> TumorSelectionOutcome:
    effective_objective_shape = objective_shape_for_data(
        data, str(fit_config.solver.objective_shape)
    )
    if effective_objective_shape != str(fit_config.solver.objective_shape):
        fit_config = replace(
            fit_config,
            solver=replace(
                fit_config.solver, objective_shape=effective_objective_shape
            ),
        )

    return _partition_guided_admm_selection(
        data=data,
        fit_options=fit_config,
        use_warm_starts=use_warm_starts,
        checkpoint_path=(None if checkpoint_path is None else Path(checkpoint_path)),
        resume_checkpoint=bool(resume_checkpoint),
    )


__all__ = [
    "BICSelectionResult",
    "DiagnosticOnlyResult",
    "NoCertifiedRawReferenceError",
    "NoEligibleModelSelectionCandidatesError",
    "SecondaryFallbackResult",
    "TumorSelectionOutcome",
    "select_secondary_fallback",
    "select_model",
]
