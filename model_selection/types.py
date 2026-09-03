from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Union

import numpy as np
import torch

from ..core.bic import SelectionScore
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.fusion.types import (
    CertificateResult,
    ConvergenceResult,
    FitProvenance,
    ObjectiveValue,
    RawFit,
    SolverState,
    WorkCounters,
)
from ..core.scalar import PartitionFit

StartArray = np.ndarray | torch.Tensor
PARTITION_REFIT_KEY_SCHEMA = "unanchored_profiled_partition_refit_v5"


@dataclass(frozen=True, slots=True)
class PartitionRefitKey:
    """Complete identity of one cached fixed-partition refit."""

    partition_signature: str
    observed_model_hash: str
    observed_likelihood_hash: str
    reporting_model_hash: str
    observed_box_hash: str
    likelihood_eps_hex: str
    refit_tolerance_hex: str
    refit_max_iter: int
    refit_mode: str
    refit_grid_points: int
    refit_local_steps: int


DirectProposalStage = Literal["pilot", "final_phi"]


@dataclass(frozen=True, slots=True)
class DirectProposal:
    """One deterministic direct-partition proposal and its raw parent."""

    candidate: PartitionCandidate
    stage: DirectProposalStage
    parent_raw_candidate_id: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, PartitionCandidate):
            raise TypeError("Direct proposal candidate must be a PartitionCandidate.")
        if self.stage not in {"pilot", "final_phi"}:
            raise ValueError("Direct proposal stage must be pilot or final_phi.")
        parent_id = self.parent_raw_candidate_id
        if parent_id is not None and (
            isinstance(parent_id, bool) or not isinstance(parent_id, int) or parent_id < 0
        ):
            raise ValueError("Direct proposal parent candidate ID must be nonnegative.")
        if (self.stage == "pilot") != (parent_id is None):
            raise ValueError("Only final-Phi proposals may identify a raw parent.")


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
class RawFusionCandidate:
    """Raw-fusion partition with an authoritative fixed-label refit and score."""

    raw_fit: RawFit
    partition: FusionPartition
    refit: PartitionFit
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
        "final_phi_hessian_ward",
        "final_phi_hessian_ward_cem",
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
    refit: PartitionFit
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
class AttemptLimits:
    """Configured iteration limits for one authorized raw solve."""

    outer_max_iter: int
    inner_max_iter: int
    certificate_max_iter: int


@dataclass(frozen=True, slots=True)
class FitAuditSummary:
    """A raw fit without fitted arrays, continuation state, or dual witness."""

    objective: ObjectiveValue
    certificate: CertificateResult
    convergence: ConvergenceResult
    work: WorkCounters
    provenance: FitProvenance

    def __post_init__(self) -> None:
        if self.certificate.witness is not None:
            raise ValueError("FitAuditSummary cannot retain a certificate witness.")

    @classmethod
    def from_fit(cls, fit: RawFit) -> "FitAuditSummary":
        return cls(
            objective=fit.objective,
            certificate=replace(fit.certificate, witness=None),
            convergence=fit.convergence,
            work=fit.work,
            provenance=fit.provenance,
        )


@dataclass(frozen=True, slots=True)
class RawAttemptSummary:
    """One start plus the tensor-free audit of its raw optimizer result."""

    source: str
    start_value: float
    breakpoint_escape_changed_count: int
    mathematically_certified: bool
    limits: AttemptLimits
    fit: FitAuditSummary
    promotion_status: str = "not_recorded"

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
        return cls(
            source=str(source),
            start_value=float(start_value),
            breakpoint_escape_changed_count=int(breakpoint_escape_changed_count),
            mathematically_certified=bool(mathematically_certified),
            limits=AttemptLimits(
                outer_max_iter=int(outer_max_iter),
                inner_max_iter=int(inner_max_iter),
                certificate_max_iter=int(certificate_max_iter),
            ),
            fit=FitAuditSummary.from_fit(fit),
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
class SearchReport:
    """Common immutable search evidence shared by every outcome tier."""

    records: tuple[CandidateRecord, ...]
    selected_id: int | None
    selection_method: str
    adaptive_search_stop_reason: str
    num_candidates_certified: int
    selection_hits_lower_boundary: bool = False
    selection_hits_upper_boundary: bool = False
    selection_boundary_unresolved: bool = True
    selection_optimum_resolved: bool = False
    ward_candidate_pool_complete: bool = False
    raw_lambda_path_resolved: bool = False
    global_hybrid_optimum_certified: bool = False
    search_work: WorkCounters = WorkCounters()
    mandatory_guide_work: WorkCounters = WorkCounters()
    cumulative_search_active_seconds: float = 0.0
    resumed_from_checkpoint: bool = False
    selection_pool_stop_reason: str = "none"

    def __post_init__(self) -> None:
        ids = tuple(int(record.candidate_id) for record in self.records)
        if len(ids) != len(set(ids)):
            raise ValueError("Selection outcome candidate IDs must be unique.")
        if self.selected_id is not None and int(self.selected_id) not in ids:
            raise ValueError("Selected candidate ID is absent from search records.")

    @property
    def num_candidates(self) -> int:
        return len(self.records)

    @property
    def selected_record(self) -> CandidateRecord | None:
        if self.selected_id is None:
            return None
        return next(
            record
            for record in self.records
            if int(record.candidate_id) == int(self.selected_id)
        )


@dataclass(frozen=True, slots=True)
class BICSelectionResult:
    selected_model: SelectedModel
    report: SearchReport

    def __post_init__(self) -> None:
        selected = self.report.selected_record
        if selected is None or selected.candidate is not self.selected_model.partition_candidate:
            raise ValueError("Search report does not own the selected model.")

    @property
    def primary_estimator_available(self) -> bool:
        return True

    @property
    def selected_lambda_representative(self) -> float | None:
        return self.selected_model.selected_lambda

    @property
    def selected_kkt_residual(self) -> float | None:
        candidate = self.selected_model.partition_candidate
        if not isinstance(candidate, RawFusionCandidate):
            return None
        residual = float(candidate.raw_fit.certificate.components.residual)
        return residual if np.isfinite(residual) else None


def _validate_best_raw_attempt(
    best_raw_attempt: RawFit | None,
    records: tuple[CandidateRecord, ...],
) -> None:
    if best_raw_attempt is None:
        return
    if not any(
        isinstance(
            item.candidate,
            (RawFusionCandidate, UnscoredRawFusionCandidate),
        )
        and item.candidate.raw_fit is best_raw_attempt
        for item in records
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
    report: SearchReport
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
        selected = self.report.selected_record
        if selected is None or selected.candidate is not candidate:
            raise ValueError(
                "The selected search record must own the fallback partition."
            )
        _validate_best_raw_attempt(self.best_raw_attempt, self.report.records)
        if int(self.report.num_candidates_certified) < 0:
            raise ValueError("num_candidates_certified must be nonnegative.")

    @property
    def selected_candidate(self) -> DirectPartitionCandidate:
        return self.selected_partition

    @property
    def selected_candidate_id(self) -> int:
        selected = self.report.selected_record
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
    report: SearchReport
    primary_estimator_available: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not str(self.reason).strip():
            raise ValueError("A diagnostic-only outcome requires an explicit reason.")
        if self.report.selected_id is not None:
            raise ValueError("A diagnostic-only outcome cannot select a candidate.")
        _validate_best_raw_attempt(self.best_raw_attempt, self.report.records)
        if int(self.report.num_candidates_certified) != 0:
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
