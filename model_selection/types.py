from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import torch

from ..core.bic import SelectionScore
from ..core.fusion.types import RawFit, SolverState, WorkCounters

StartArray = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    """A durable scalar fit paired with transient continuation state."""

    fit: RawFit
    state: SolverState | None

    def __post_init__(self) -> None:
        if getattr(self.fit, "state", None) is not None:
            raise ValueError("SolveOutcome.fit must not retain SolverState.")
        certificate = getattr(self.fit, "certificate", None)
        if getattr(certificate, "witness", None) is not None:
            raise ValueError("SolveOutcome.fit must not retain a certificate witness.")


def _immutable_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FusionPartition:
    labels: np.ndarray
    signature: str
    certified: bool
    source: Literal[
        "solver_quotient",
        "verified_primal_equalities",
        "tolerance_defined_primal",
        "legacy_connected_components",
    ]
    certification_failure_reason: str = "none"
    mutation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "labels",
            _immutable_array(self.labels, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "mutation_ids",
            tuple(str(value) for value in self.mutation_ids),
        )

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size)


@dataclass(frozen=True)
class PartitionRefitSummary:
    labels: np.ndarray
    partition_signature: str
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    finite_candidate_found: bool
    global_optimum_certified: bool
    refit_numerically_resolved: bool = False
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    refit_mode: str = "interval_certified"
    coordinate_argmin_lower: np.ndarray | None = None
    coordinate_argmin_upper: np.ndarray | None = None
    coordinate_statistically_identified: np.ndarray | None = None
    refit_coordinate_count: int = 0
    refit_objective_evaluations: int = 0

    def __post_init__(self) -> None:
        if int(self.refit_coordinate_count) < 0 or int(
            self.refit_objective_evaluations
        ) < 0:
            raise ValueError("Partition-refit work counters must be nonnegative.")
        object.__setattr__(
            self, "refit_coordinate_count", int(self.refit_coordinate_count)
        )
        object.__setattr__(
            self,
            "refit_objective_evaluations",
            int(self.refit_objective_evaluations),
        )
        if self.global_optimum_certified and (
            not np.isfinite(float(self.global_lower_bound))
            or not np.isfinite(float(self.global_optimality_gap))
            or float(self.global_optimality_gap) < 0.0
            or str(self.global_certificate_method) == "none"
        ):
            raise ValueError("A global refit claim requires a finite certificate.")
        object.__setattr__(
            self,
            "labels",
            _immutable_array(self.labels, dtype=np.dtype(np.int64)),
        )
        object.__setattr__(
            self,
            "phi",
            _immutable_array(self.phi, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "cluster_centers",
            _immutable_array(self.cluster_centers, dtype=np.dtype(np.float64)),
        )
        center_shape = tuple(np.asarray(self.cluster_centers).shape)
        lower = (
            np.asarray(self.cluster_centers, dtype=np.float64)
            if self.coordinate_argmin_lower is None
            else np.asarray(self.coordinate_argmin_lower, dtype=np.float64)
        )
        upper = (
            np.asarray(self.cluster_centers, dtype=np.float64)
            if self.coordinate_argmin_upper is None
            else np.asarray(self.coordinate_argmin_upper, dtype=np.float64)
        )
        identified = (
            np.ones(center_shape, dtype=bool)
            if self.coordinate_statistically_identified is None
            else np.asarray(self.coordinate_statistically_identified, dtype=bool)
        )
        if tuple(lower.shape) != center_shape or tuple(upper.shape) != center_shape:
            raise ValueError("Refit argmin intervals must match cluster_centers.")
        if tuple(identified.shape) != center_shape:
            raise ValueError("Refit identification flags must match cluster_centers.")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(
            lower > upper
        ):
            raise ValueError("Refit argmin intervals must be finite and ordered.")
        object.__setattr__(
            self,
            "coordinate_argmin_lower",
            _immutable_array(lower, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "coordinate_argmin_upper",
            _immutable_array(upper, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "coordinate_statistically_identified",
            _immutable_array(identified, dtype=np.dtype(bool)),
        )


@dataclass(frozen=True)
class RawFusionCandidate:
    """Raw-fusion partition with an authoritative fixed-label refit and score."""

    raw_fit: RawFit
    partition: FusionPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    eligible_for_selection: bool
    ineligibility_reason: str
    work: WorkCounters = WorkCounters()

    @property
    def raw_objective_certified(self) -> bool:
        certificate = self.raw_fit.certificate
        return bool(
            self.raw_fit.provenance.lambda_value > 0.0
            and certificate.certified
            and certificate.admissible
        )


@dataclass(frozen=True)
class UnscoredRawFusionCandidate:
    """Raw optimizer result retained before fixed-partition evaluation.

    A raw fit that fails the exact-fusion admission contract cannot become a
    selectable partition candidate.  Retaining that attempt as a distinct
    type preserves its partition and numerical diagnostics without inventing
    a fixed-label refit or score that downstream selection must ignore.
    """

    raw_fit: RawFit
    partition: FusionPartition
    ineligibility_reason: str
    refit: None = field(default=None, init=False, repr=False)
    score: None = field(default=None, init=False, repr=False)
    eligible_for_selection: Literal[False] = field(default=False, init=False)
    work: WorkCounters = WorkCounters()

    def __post_init__(self) -> None:
        reason = str(self.ineligibility_reason).strip()
        if not reason or reason == "none":
            raise ValueError(
                "An unscored raw-fusion candidate requires an ineligibility reason."
            )
        object.__setattr__(self, "ineligibility_reason", reason)

    @property
    def raw_objective_certified(self) -> bool:
        # Construction and identity validation require failure of the full
        # exact-fusion admission contract, which is stricter than the solver's
        # local ``certified``/``admissible`` booleans alone.
        return False


@dataclass(frozen=True)
class DirectPartition:
    labels: np.ndarray
    signature: str
    source: Literal[
        "pilot_hessian_ward",
        "pilot_hessian_ward_cem",
        "pilot_hessian_ward_cem_component_death",
        "final_phi_hessian_ward",
        "final_phi_hessian_ward_cem",
        "final_phi_hessian_ward_cem_component_death",
    ]
    mutation_ids: tuple[str, ...]
    parent_raw_candidate_id: int | None = None
    parent_raw_lambda: float | None = None
    parent_raw_phi_hash: str = ""

    def __post_init__(self) -> None:
        labels = _immutable_array(self.labels, dtype=np.dtype(np.int64))
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("Direct partition labels must be nonempty and 1-D.")
        unique = np.unique(labels)
        expected = np.arange(unique.size, dtype=np.int64)
        if not np.array_equal(unique, expected):
            raise ValueError(
                "Direct partition labels must be canonical zero-based IDs."
            )
        mutation_ids = tuple(str(value) for value in self.mutation_ids)
        if len(mutation_ids) != labels.size or len(set(mutation_ids)) != labels.size:
            raise ValueError(
                "Direct partition mutation IDs must uniquely identify every label."
            )
        if not str(self.signature):
            raise ValueError("Direct partition identity provenance is required.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "mutation_ids", mutation_ids)

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.labels).size)


@dataclass(frozen=True)
class DirectPartitionCandidate:
    partition: DirectPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    eligible_for_selection: bool
    ineligibility_reason: str
    work: WorkCounters = WorkCounters()


RawFusionArtifact = Union[RawFusionCandidate, UnscoredRawFusionCandidate]
SelectablePartitionCandidate = Union[RawFusionCandidate, DirectPartitionCandidate]
SearchArtifact = Union[SelectablePartitionCandidate, UnscoredRawFusionCandidate]
CandidateFamily = Literal["raw_fusion", "direct_partition"]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Typed search artifact and its compact provenance."""

    candidate_id: int
    candidate: SearchArtifact
    trace: CandidateTrace = field(default_factory=lambda: CandidateTrace())

    def __post_init__(self) -> None:
        if int(self.candidate_id) < 0:
            raise ValueError("candidate_id must be nonnegative.")

    @property
    def score(self) -> SelectionScore | None:
        return self.candidate.score

    @property
    def partition_signature(self) -> str:
        return self.candidate.partition.signature

    @property
    def family(self) -> CandidateFamily:
        return (
            "raw_fusion"
            if isinstance(
                self.candidate,
                (RawFusionCandidate, UnscoredRawFusionCandidate),
            )
            else "direct_partition"
        )

    @property
    def eligible_for_selection(self) -> bool:
        return bool(self.candidate.eligible_for_selection)

    @property
    def lambda_value(self) -> float | None:
        if isinstance(
            self.candidate,
            (RawFusionCandidate, UnscoredRawFusionCandidate),
        ):
            return float(self.candidate.raw_fit.provenance.lambda_value)
        return None

    @property
    def n_clusters(self) -> int:
        return int(self.candidate.partition.n_clusters)

    @property
    def penalized_objective(self) -> float | None:
        if isinstance(
            self.candidate,
            (RawFusionCandidate, UnscoredRawFusionCandidate),
        ):
            return float(self.candidate.raw_fit.objective.total)
        return None

    @property
    def mm_consistency_violations(self) -> int:
        if isinstance(
            self.candidate,
            (RawFusionCandidate, UnscoredRawFusionCandidate),
        ):
            return int(self.candidate.raw_fit.convergence.mm_consistency_violations)
        return 0


@dataclass(frozen=True, slots=True)
class RawAttemptSummary:
    """Tensor-free provenance for one authorized raw optimizer start.

    Retry ownership ends with the adaptive search.  Durable candidate traces
    retain the scalar evidence needed for diagnostics and serialization, but
    never another ``RawFit``, fitted Phi, solver state, or certificate witness.
    """

    source: str
    start_value: float
    breakpoint_escape_changed_count: int
    mathematically_certified: bool
    outer_max_iter: int
    inner_max_iter: int
    certificate_max_iter: int
    objective: float
    lambda_value: float
    stationarity: float
    edge_subgradient: float
    dual_ball: float
    box: float
    kkt_residual: float
    kkt_tolerance: float
    certificate_status: str
    certificate_admissible: bool
    working_dtype: str
    audit_dtype: str
    precision_polished: bool
    precision_polish_delta: float
    mm_consistency_violations: int
    stage_outer_iterations: int
    stage_outer_max_iter: int
    stage_inner_iterations: int
    stage_inner_max_iter: int
    stage_inner_solve_calls: int
    stop_reason: str
    progress_residual_method: str
    solve_tolerance: float
    legacy_stop_kkt_residual: float
    componentwise_stop_kkt_residual: float
    accepted_full_steps: int
    accepted_damped_steps: int
    rejected_outer_steps: int
    work_inner_iterations: int
    work_certificate_iterations: int
    device: str
    dtype: str
    objective_spec_hash: str
    original_graph_hash: str
    certificate_problem_hash: str
    fallback_reason: str
    promotion_status: str = "not_recorded"
    work_inner_stationarity_checks: int = 0
    work_inner_full_kkt_audits: int = 0
    work_outer_kkt_audits: int = 0
    work_certificate_full_graph_passes: int = 0
    work_partition_refit_coordinates: int = 0
    work_partition_refit_objective_evaluations: int = 0
    work_edge_pass_equivalents: int = 0
    work_full_certificate_audit_passes: int = 0

    @classmethod
    def from_fit(
        cls,
        fit: RawFit,
        *,
        source: str,
        start_value: float,
        breakpoint_escape_changed_count: int,
        mathematically_certified: bool,
        outer_max_iter: int,
        inner_max_iter: int,
        certificate_max_iter: int,
        promotion_status: str = "not_recorded",
    ) -> "RawAttemptSummary":
        certificate = fit.certificate
        components = certificate.components
        convergence = fit.convergence
        work = fit.work
        provenance = fit.provenance
        return cls(
            source=str(source),
            start_value=float(start_value),
            breakpoint_escape_changed_count=int(breakpoint_escape_changed_count),
            mathematically_certified=bool(mathematically_certified),
            outer_max_iter=int(outer_max_iter),
            inner_max_iter=int(inner_max_iter),
            certificate_max_iter=int(certificate_max_iter),
            objective=float(fit.objective.total),
            lambda_value=float(provenance.lambda_value),
            stationarity=float(components.stationarity),
            edge_subgradient=float(components.edge_subgradient),
            dual_ball=float(components.dual_ball),
            box=float(components.box),
            kkt_residual=float(components.residual),
            kkt_tolerance=float(certificate.tolerance),
            certificate_status=str(certificate.status),
            certificate_admissible=bool(certificate.admissible),
            working_dtype=str(certificate.working_dtype),
            audit_dtype=str(certificate.audit_dtype),
            precision_polished=bool(certificate.precision_polished),
            precision_polish_delta=float(certificate.precision_polish_delta),
            mm_consistency_violations=int(convergence.mm_consistency_violations),
            stage_outer_iterations=int(
                getattr(
                    convergence,
                    "stage_outer_iterations",
                    getattr(convergence, "iterations", 0),
                )
            ),
            stage_outer_max_iter=int(
                getattr(convergence, "stage_outer_max_iter", outer_max_iter)
            ),
            stage_inner_iterations=int(
                getattr(
                    convergence,
                    "stage_inner_iterations",
                    getattr(work, "inner_iterations", 0),
                )
            ),
            stage_inner_max_iter=int(
                getattr(convergence, "stage_inner_max_iter", inner_max_iter)
            ),
            stage_inner_solve_calls=int(
                getattr(convergence, "stage_inner_solve_calls", 0)
            ),
            stop_reason=str(getattr(convergence, "stop_reason", "not_recorded")),
            progress_residual_method=str(
                getattr(convergence, "progress_residual_method", "not_recorded")
            ),
            solve_tolerance=float(
                getattr(convergence, "solve_tolerance", float("nan"))
            ),
            legacy_stop_kkt_residual=float(
                getattr(convergence, "legacy_stop_kkt_residual", float("inf"))
            ),
            componentwise_stop_kkt_residual=float(
                getattr(
                    convergence,
                    "componentwise_stop_kkt_residual",
                    float("inf"),
                )
            ),
            accepted_full_steps=int(
                getattr(convergence, "accepted_full_steps", 0)
            ),
            accepted_damped_steps=int(
                getattr(convergence, "accepted_damped_steps", 0)
            ),
            rejected_outer_steps=int(
                getattr(convergence, "rejected_outer_steps", 0)
            ),
            work_inner_iterations=int(getattr(work, "inner_iterations", 0)),
            work_inner_stationarity_checks=int(
                getattr(work, "inner_stationarity_checks", 0)
            ),
            work_inner_full_kkt_audits=int(
                getattr(work, "inner_full_kkt_audits", 0)
            ),
            work_outer_kkt_audits=int(getattr(work, "outer_kkt_audits", 0)),
            work_certificate_iterations=int(
                getattr(work, "certificate_iterations", 0)
            ),
            work_certificate_full_graph_passes=int(
                getattr(work, "certificate_full_graph_passes", 0)
            ),
            work_partition_refit_coordinates=int(
                getattr(work, "partition_refit_coordinates", 0)
            ),
            work_partition_refit_objective_evaluations=int(
                getattr(work, "partition_refit_objective_evaluations", 0)
            ),
            work_edge_pass_equivalents=int(
                getattr(work, "edge_pass_equivalents", 0)
            ),
            work_full_certificate_audit_passes=int(
                getattr(work, "full_certificate_audit_passes", 0)
            ),
            device=str(getattr(provenance, "device", "not_recorded")),
            dtype=str(getattr(provenance, "dtype", "not_recorded")),
            objective_spec_hash=str(
                getattr(provenance, "objective_spec_hash", "")
            ),
            original_graph_hash=str(
                getattr(provenance, "original_graph_hash", "")
            ),
            certificate_problem_hash=str(
                getattr(provenance, "certificate_problem_hash", "")
            ),
            fallback_reason=str(certificate.fallback_reason),
            promotion_status=str(promotion_status),
        )


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    """Small immutable trace retained after the adaptive search finishes."""

    search_round: int = -1
    search_phase: str = "unknown"
    start_source: str = "not_applicable"
    start_value: float | None = None
    breakpoint_escape_changed_count: int = 0
    raw_attempts: tuple[RawAttemptSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """One immutable candidate plus selection-decision annotations."""

    record: CandidateRecord
    selected: bool

    @property
    def candidate_id(self) -> int:
        return int(self.record.candidate_id)

    @property
    def candidate(self) -> SearchArtifact:
        return self.record.candidate

    @property
    def trace(self) -> CandidateTrace:
        return self.record.trace


@dataclass(frozen=True, slots=True)
class CandidateSelectionDecision:
    """Complete typed outcome of partition-first candidate selection."""

    selected: CandidateRecord
    num_eligible: int
    selected_lambda_left: float | None
    selected_lambda_right: float | None
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool


@dataclass(frozen=True)
class SelectedModel:
    raw_reference: RawFusionCandidate
    partition_candidate: SelectablePartitionCandidate
    # For a final-raw-Phi direct proposal this is the exact raw candidate whose
    # Phi generated the deterministic Ward/CEM ladder.  It is deliberately
    # separate from ``raw_reference``, which is selected independently as the
    # best certified raw-fusion result for estimator provenance.
    partition_parent_raw: RawFusionCandidate | None = None

    @property
    def selected_partition_signature(self) -> str:
        return str(self.partition_candidate.partition.signature)

    @property
    def selected_candidate_family(self) -> CandidateFamily:
        return (
            "raw_fusion"
            if isinstance(self.partition_candidate, RawFusionCandidate)
            else "direct_partition"
        )

    @property
    def selected_lambda(self) -> float | None:
        if isinstance(self.partition_candidate, RawFusionCandidate):
            return float(self.partition_candidate.raw_fit.provenance.lambda_value)
        return None

    def __post_init__(self) -> None:
        raw = self.raw_reference
        candidate = self.partition_candidate
        if not candidate.eligible_for_selection:
            raise ValueError("Selected model must be eligible for selection.")
        if isinstance(candidate, RawFusionCandidate):
            if self.partition_parent_raw is not None:
                raise ValueError("Raw selected models cannot have a direct parent.")
            if not candidate.raw_objective_certified:
                raise ValueError("Selected raw model must have a certified objective.")
            if not candidate.partition.certified:
                raise ValueError("Selected raw model must have a certified partition.")
        else:
            parent_id = candidate.partition.parent_raw_candidate_id
            if (parent_id is None) != (self.partition_parent_raw is None):
                raise ValueError(
                    "Direct-partition parent provenance is incomplete or spurious."
                )
            if self.partition_parent_raw is not None:
                parent_lambda = candidate.partition.parent_raw_lambda
                if parent_lambda is None or not np.isclose(
                    float(parent_lambda),
                    float(self.partition_parent_raw.raw_fit.provenance.lambda_value),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError("Direct-partition parent lambda is inconsistent.")
        if candidate.score.partition_signature != candidate.partition.signature:
            raise ValueError("Selected-model score signature is inconsistent.")
        if (
            not raw.eligible_for_selection
            or not raw.raw_objective_certified
            or not raw.partition.certified
        ):
            raise ValueError("Raw reference must remain a certified raw-fusion model.")


@dataclass(frozen=True, slots=True)
class BICSelectionResult:
    selected_model: SelectedModel
    search: tuple[SearchCandidate, ...]
    selection_method: str
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool
    selection_optimum_resolved: bool
    adaptive_search_stop_reason: str
    num_candidates: int
    num_candidates_certified: int
    selected_kkt_residual: float | None
    selected_lambda_representative: float | None
    ward_candidate_pool_complete: bool = False
    raw_lambda_path_resolved: bool = False
    global_hybrid_optimum_certified: bool = False
    search_work: WorkCounters = WorkCounters()
    cumulative_search_active_seconds: float = 0.0
    resumed_from_checkpoint: bool = False
    selection_pool_stop_reason: str = "none"

    @property
    def primary_estimator_available(self) -> bool:
        return True


def _validate_outcome_search(
    search: tuple[SearchCandidate, ...],
    *,
    require_selected: bool,
) -> SearchCandidate | None:
    candidate_ids = [int(item.candidate_id) for item in search]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Selection outcome candidate IDs must be unique.")
    selected = [item for item in search if bool(item.selected)]
    expected = 1 if require_selected else 0
    if len(selected) != expected:
        raise ValueError(
            f"Selection outcome must contain exactly {expected} selected candidate(s)."
        )
    return selected[0] if selected else None


def _validate_best_raw_attempt(
    best_raw_attempt: RawFit | None,
    search: tuple[SearchCandidate, ...],
) -> None:
    if best_raw_attempt is None:
        return
    if not any(
        isinstance(
            item.candidate,
            (RawFusionCandidate, UnscoredRawFusionCandidate),
        )
        and item.candidate.raw_fit is best_raw_attempt
        for item in search
    ):
        raise ValueError("best_raw_attempt must come from the retained raw candidates.")


@dataclass(frozen=True, slots=True)
class SecondaryFallbackResult:
    """A scored direct partition without a certified raw-fusion reference.

    This is deliberately not a ``SelectedModel``.  Its fixed-label refit is a
    useful conditional estimate, but it carries no primary-estimator or raw-KKT
    claim and therefore cannot be serialized under the primary compatibility
    filenames.
    """

    selected_partition: DirectPartitionCandidate
    best_raw_attempt: RawFit | None
    reason: str
    search: tuple[SearchCandidate, ...]
    selection_method: str
    adaptive_search_stop_reason: str
    num_candidates: int
    num_candidates_certified: int
    selection_hits_lower_boundary: bool = False
    selection_hits_upper_boundary: bool = False
    selection_boundary_unresolved: bool = True
    ward_candidate_pool_complete: bool = False
    raw_lambda_path_resolved: bool = False
    global_hybrid_optimum_certified: bool = False
    search_work: WorkCounters = WorkCounters()
    cumulative_search_active_seconds: float = 0.0
    resumed_from_checkpoint: bool = False
    selection_pool_stop_reason: str = "none"
    primary_estimator_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        candidate = self.selected_partition
        if not isinstance(candidate, DirectPartitionCandidate):
            raise TypeError("A secondary fallback requires a direct partition.")
        if not candidate.eligible_for_selection:
            raise ValueError("A secondary fallback partition must be eligible.")
        if not candidate.refit.finite_candidate_found or not np.isfinite(
            float(candidate.score.value)
        ):
            raise ValueError("A secondary fallback requires a finite refit and score.")
        if not str(self.reason).strip():
            raise ValueError("A secondary fallback requires an explicit reason.")
        selected = _validate_outcome_search(self.search, require_selected=True)
        if selected is None or selected.candidate is not candidate:
            raise ValueError(
                "The selected search record must own the fallback partition."
            )
        _validate_best_raw_attempt(self.best_raw_attempt, self.search)
        if int(self.num_candidates) != len(self.search):
            raise ValueError("num_candidates must match the retained search.")
        if int(self.num_candidates_certified) < 0:
            raise ValueError("num_candidates_certified must be nonnegative.")

    @property
    def selected_candidate(self) -> DirectPartitionCandidate:
        return self.selected_partition

    @property
    def selected_candidate_id(self) -> int:
        selected = _validate_outcome_search(self.search, require_selected=True)
        if selected is None:  # pragma: no cover - guarded in __post_init__
            raise AssertionError("Secondary fallback lost its selected record.")
        return int(selected.candidate_id)

    @property
    def selected_lambda_representative(self) -> None:
        return None

    @property
    def selected_kkt_residual(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class DiagnosticOnlyResult:
    """A retained search with no defensible selected partition point claim."""

    best_raw_attempt: RawFit | None
    reason: str
    search: tuple[SearchCandidate, ...]
    selection_method: str
    adaptive_search_stop_reason: str
    num_candidates: int
    num_candidates_certified: int = 0
    ward_candidate_pool_complete: bool = False
    raw_lambda_path_resolved: bool = False
    global_hybrid_optimum_certified: bool = False
    search_work: WorkCounters = WorkCounters()
    cumulative_search_active_seconds: float = 0.0
    resumed_from_checkpoint: bool = False
    selection_pool_stop_reason: str = "none"
    primary_estimator_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("A diagnostic-only outcome requires an explicit reason.")
        _validate_outcome_search(self.search, require_selected=False)
        _validate_best_raw_attempt(self.best_raw_attempt, self.search)
        if int(self.num_candidates) != len(self.search):
            raise ValueError("num_candidates must match the retained search.")
        if int(self.num_candidates_certified) != 0:
            raise ValueError("A diagnostic-only outcome cannot claim a selected model.")

    @property
    def selected_candidate(self) -> None:
        return None

    @property
    def selected_lambda_representative(self) -> None:
        return None

    @property
    def selected_kkt_residual(self) -> None:
        return None


TumorSelectionOutcome = Union[
    BICSelectionResult,
    SecondaryFallbackResult,
    DiagnosticOnlyResult,
]
