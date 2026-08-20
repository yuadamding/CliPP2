from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from ..config import FitConfig
from ..core.model import fit_fixed_objective
from ..core.fusion.partition_starts import (
    PartitionCandidate,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.solver import (
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
    torch_data_from_context,
)
from ..core.fusion.types import (
    RawFit,
    SolverState,
)
from ..io.data import TumorData

from ..model_selection.candidates import (
    PartitionRefitCacheEntry,
    evaluate_direct_partition_candidate,
    evaluate_raw_fusion_candidate,
    validate_candidate_identity,
)
from ..model_selection.config import (
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
)
from ..model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
)
from ..model_selection.partition_initializer import generate_partition_initializer_pool
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
    escape_path_breakpoint_retry_state as _escape_path_breakpoint_retry_state,
    offload_solver_state_to_cpu as _offload_solver_state_to_cpu,
    pilot_matrix_hash as _pilot_matrix_hash,
    rescore_partition_candidates as _rescore_partition_candidates,
    select_raw_start_attempt as _select_raw_start_attempt,
    solver_recovery_fit_options as _solver_recovery_fit_options,
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
            f"certificate_statuses={certificate_counts}."
        )


def _stable_counts(values) -> tuple[tuple[str, int], ...]:
    normalized = ["<missing>" if value is None else str(value) for value in values]
    return tuple(
        (value, normalized.count(value)) for value in sorted(set(normalized))
    )


def _raw_attempts(record: CandidateRecord) -> tuple[RawAttemptTrace, ...]:
    if record.trace.raw_attempts:
        return record.trace.raw_attempts
    candidate = record.candidate
    if not isinstance(candidate, RawFusionCandidate):
        return ()
    return (
        RawAttemptTrace(
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
        certificate = attempt.fit.certificate
        min_residual = float(certificate.components.residual)
        if np.isfinite(float(certificate.tolerance)):
            min_tolerance = float(certificate.tolerance)
        components = {
            "stationarity": float(certificate.components.stationarity),
            "edge_subgradient": float(certificate.components.edge_subgradient),
            "dual_ball": float(certificate.components.dual_ball),
            "box": float(certificate.components.box),
        }
        finite_components = {
            name: value for name, value in components.items() if np.isfinite(value)
        }
        if finite_components:
            dominant_component = max(finite_components, key=finite_components.get)
        best_attempt = BestRawAttemptDiagnostics(
            search_round=int(record.trace.search_round),
            search_phase=str(record.trace.search_phase),
            lambda_value=float(attempt.fit.provenance.lambda_value),
            source=str(attempt.source),
            kkt_residual=min_residual,
            kkt_tolerance=min_tolerance,
            dominant_kkt_component=dominant_component,
            outer_max_iter=int(attempt.outer_max_iter),
            inner_max_iter=int(attempt.inner_max_iter),
            certificate_max_iter=int(attempt.certificate_max_iter),
        )

    mm_values = [
        int(attempt.fit.convergence.mm_consistency_violations)
        for attempt, _ in attempts
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
            attempt.fit.certificate.status for attempt, _ in attempts
        ),
        search_phase_counts=_stable_counts(record.trace.search_phase for record in raw),
        start_source_counts=_stable_counts(record.trace.start_source for record in raw),
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
    strict_positive_exact_fusion: bool = False,
    ward_candidate_pool_complete: bool = False,
) -> BICSelectionResult:
    try:
        decision = select_candidate_records(
            result_entries,
            strict_positive_exact_fusion=bool(strict_positive_exact_fusion),
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


def _partition_guided_admm_selection(
    *,
    data: TumorData,
    fit_options: FitConfig,
    use_warm_starts: bool,
) -> BICSelectionResult:
    """Run the certified raw path and select under one immutable contract.

    Ward/CEM supplies the primal start and initial-lambda scale. The contract
    declares whether the guide or zero-penalty pilot defines the frozen graph,
    and whether retained pilot/final-raw-phi partitions enter the secondary
    selection pool. Direct proposals are evaluated only after the raw lambda
    controller terminates, so they cannot steer or replace the raw optimizer.
    """

    selection_contract = fit_options.selection.contract
    selection_score = str(fit_options.selection.score)
    normalized_score = selection_score
    selection_method = "online_partition_guided_admm"
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
    pilot_torch_data = torch_data_from_context(pilot_context)
    guide_curvature = observed_curvature_at_pilot_torch(
        data,
        pilot_phi,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        torch_data=pilot_torch_data,
        device=pilot_runtime.device,
        dtype=pilot_runtime.dtype,
    )
    initializer_pool = generate_partition_initializer_pool(
        data=data,
        pilot_phi=pilot_phi,
        fit_options=fit_options,
        normalized_score=normalized_score,
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
        rescore_candidates=_rescore_partition_candidates,
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
    if fit_options.graph.graph is None:
        graph_pilot_source = str(fit_options.selection.graph_pilot_source)
        if graph_pilot_source == "partition_guide":
            graph_builder_phi = guide_phi
        elif graph_pilot_source == "zero_penalty_pilot":
            graph_builder_phi = pilot_phi
        else:  # resolved configurations never retain ``profile_default``
            raise AssertionError("Unresolved graph-pilot source.")
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
        # Preserve the independent likelihood starts.  The previous flow
        # replaced both with the Ward guide, so nominal "cold" retries were
        # merely duplicates of the same non-convex basin.
        exact_pilot=pilot_context.exact_pilot,
        pooled_start=pilot_context.pooled_start,
        scalar_well_starts=pilot_context.scalar_well_starts,
        device=fit_options.runtime.device,
        dtype=fit_options.runtime.dtype,
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
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

    result_entries: list[CandidateRecord] = []
    fit_by_lambda: dict[float, RawFit] = {}
    partition_k_by_lambda: dict[float, int] = {}
    attempts_by_lambda: dict[float, list[RawFit]] = {}
    bic_refit_cache: dict[object, PartitionRefitCacheEntry] = {}
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
            base_solver = effective_fit_options.solver
            candidate_fit_options = replace(
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

        def solve_raw_path() -> tuple[
            RawFit,
            _RawStartAttempt,
            tuple[_RawStartAttempt, ...],
        ]:
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
                if attempt.state is not None
                and np.isfinite(float(attempt.certificate.components.residual))
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
                        key=lambda attempt: float(
                            attempt.certificate.components.residual
                        ),
                    )
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
                    use_warm_starts
                    and int(proposal.retry_number) == 1
                    and alternate_fit is not None
                    and alternate_fit.state is not None
                ):
                    append_distinct_start(
                        "alternate_bracket_endpoint",
                        float(proposal.alternate_start_lambda),
                        alternate_fit.state,
                    )
                elif (
                    use_warm_starts
                    and warm_fit is not None
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
                    and use_warm_starts
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
                    and use_warm_starts
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
                    tol=float(candidate_fit_options.solver.tolerance),
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
                    config=replace(
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
                )
                if str(seed_fit.provenance.objective_spec_hash) != str(
                    context.objective_spec_hash
                ):
                    raise AssertionError(
                        "Raw multistart changed the fixed objective identity."
                    )
                if seed_fit.state is not None:
                    seed_fit = replace(
                        seed_fit,
                        state=_offload_solver_state_to_cpu(seed_fit.state),
                    )
                attempts_by_lambda.setdefault(lambda_key, []).append(seed_fit)
                mathematically_certified = bool(
                    float(seed_fit.provenance.lambda_value) > 0.0
                    and seed_fit.certificate.certified
                    and seed_fit.certificate.admissible
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
            selected_attempt = _select_raw_start_attempt(start_attempts)
            seed_fit = selected_attempt.fit
            # Subsequent bracket proposals must warm-start from the same raw
            # basin that was admitted to partition scoring, never from a lower
            # objective but mathematically uncertified side attempt.
            return seed_fit, selected_attempt, tuple(start_attempts)

        selected_raw_fit, selected_start, raw_start_attempts = solve_raw_path()
        fit, artifact = evaluate_raw_fusion_candidate(
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
            solver_state=selected_raw_fit.state,
            lambda_value=float(proposal.lambda_value),
            selection_score=selection_score,
            bic_refit_cache=bic_refit_cache,
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
        candidate_id = int(len(result_entries))
        result_entries.append(
            CandidateRecord(
                candidate_id=candidate_id,
                candidate=artifact,
                trace=CandidateTrace(
                    search_round=int(next_step),
                    search_phase=str(proposal.phase),
                    start_source=str(lambda_start_source),
                    start_value=float(lambda_start_value),
                    breakpoint_escape_changed_count=int(
                        path_breakpoint_escape_changed_count
                    ),
                    raw_attempts=tuple(
                        RawAttemptTrace(
                            fit=attempt.fit,
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
                        )
                        for attempt in raw_start_attempts
                    ),
                ),
            )
        )
        incumbent = fit_by_lambda.get(lambda_key)
        if _prefer_fit_candidate(fit, incumbent):
            fit_by_lambda[lambda_key] = fit
            partition_k_by_lambda[lambda_key] = int(artifact.partition.n_clusters)

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
        next_step += 1

    ward_candidate_pool_complete = False
    if selection_contract.selectable_partition_pool:
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
        config = selection_contract.partition_config
        if config.include_final_phi_ladder and config.final_phi_ladder_kmax > 0:
            raw_parent_records = sorted(
                (
                    record
                    for record in result_entries
                    if isinstance(record.candidate, RawFusionCandidate)
                    and raw_candidate_has_exact_fusion_certificate(record.candidate)
                ),
                key=lambda record: (
                    float(record.score.value),
                    float(record.score.numerical_uncertainty),
                    float(record.candidate.raw_fit.provenance.lambda_value),
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
            candidate_id = int(len(result_entries))
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
                refit_cache=bic_refit_cache,
            )
            result_entries.append(
                CandidateRecord(
                    candidate_id=candidate_id,
                    candidate=direct_candidate,
                    trace=CandidateTrace(
                        search_round=int(next_step),
                        search_phase=f"{stage}_direct_partition_pool",
                    ),
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
        result_entries=result_entries,
        selection_method=selection_method,
        adaptive_search_stop_reason=stop_reason,
        strict_positive_exact_fusion=not bool(
            selection_contract.selectable_partition_pool
        ),
        ward_candidate_pool_complete=bool(ward_candidate_pool_complete),
    )


def select_model(
    *,
    data: TumorData,
    fit_config: FitConfig,
    use_warm_starts: bool,
) -> BICSelectionResult:
    effective_objective_shape = objective_shape_for_data(
        data, str(fit_config.solver.objective_shape)
    )
    if effective_objective_shape != str(fit_config.solver.objective_shape):
        fit_config = replace(
            fit_config,
            solver=replace(fit_config.solver, objective_shape=effective_objective_shape),
        )

    return _partition_guided_admm_selection(
        data=data,
        fit_options=fit_config,
        use_warm_starts=use_warm_starts,
    )


__all__ = [
    "BICSelectionResult",
    "NoCertifiedRawReferenceError",
    "NoEligibleModelSelectionCandidatesError",
    "select_model",
]
