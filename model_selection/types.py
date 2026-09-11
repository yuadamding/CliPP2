from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
import torch

from ..core.bic import SelectionScore
from ..core.fusion.types import (
    ConvergenceResult, DenseEdgeCertificate, KKTComponents, RawFit, WorkCounters,
)
from ..io.data import ImmutableArrayRecord, readonly_array

StartArray = np.ndarray | torch.Tensor


def is_zero_edge_singleton(fit: RawFit) -> bool:
    """The only admissible zero-lambda raw model has one row and no edges."""
    witness = fit.certificate.witness
    return bool(
        fit.phi.shape[0] == 1
        and fit.provenance.lambda_value == 0.0
        and isinstance(witness, DenseEdgeCertificate)
        and torch.is_tensor(witness.dual)
        and tuple(witness.dual.shape) == (0, fit.phi.shape[1])
        and witness.graph_hash == fit.provenance.original_graph_hash
    )


@dataclass(frozen=True)
class FusionPartition(ImmutableArrayRecord):
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
            readonly_array(self.labels, dtype=np.int64),
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
class PartitionRefitSummary(ImmutableArrayRecord):
    labels: np.ndarray
    partition_signature: str
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    finite_candidate_found: bool
    global_optimum_certified: bool
    source_data_hash: str
    likelihood_eps: float
    refit_numerically_resolved: bool = False
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    refit_mode: str = "grid_local"
    multiplicity_policy: str = "independent_broad"
    locally_converged: bool | None = None

    def __post_init__(self) -> None:
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
            readonly_array(self.labels, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "phi",
            readonly_array(self.phi, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "cluster_centers",
            readonly_array(self.cluster_centers, dtype=np.float64),
        )


@dataclass(frozen=True)
class RawFusionCandidate:
    raw_fit: RawFit
    partition: FusionPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    eligible_for_selection: bool
    ineligibility_reason: str

    @property
    def raw_objective_certified(self) -> bool:
        certificate = self.raw_fit.certificate
        return bool(
            (
                self.raw_fit.provenance.lambda_value > 0.0
                or is_zero_edge_singleton(self.raw_fit)
            )
            and certificate.certified
            and certificate.admissible
        )


@dataclass(frozen=True)
class DirectPartition(ImmutableArrayRecord):
    labels: np.ndarray
    signature: str
    source: Literal[
        "pilot_hessian_ward",
        "pilot_hessian_ward_cem",
        "pilot_hessian_ward_cem_component_death",
        "final_phi_hessian_ward",
        "final_phi_hessian_ward_cem",
        "final_phi_hessian_ward_cem_component_death",
        "pilot_dosage_rescale",
        "pilot_dosage_split",
        "final_phi_dosage_rescale",
        "final_phi_dosage_split",
    ]
    mutation_ids: tuple[str, ...]
    parent_raw_candidate_id: int | None = None
    parent_raw_lambda: float | None = None
    parent_raw_phi_hash: str = ""

    def __post_init__(self) -> None:
        labels = readonly_array(self.labels, dtype=np.int64)
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


SelectablePartitionCandidate = Union[RawFusionCandidate, DirectPartitionCandidate]
CandidateFamily = Literal["raw_fusion", "direct_partition"]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Typed selection unit and its compact search provenance."""

    candidate_id: int
    candidate: SelectablePartitionCandidate
    trace: CandidateTrace = field(default_factory=lambda: CandidateTrace())

    def __post_init__(self) -> None:
        if int(self.candidate_id) < 0:
            raise ValueError("candidate_id must be nonnegative.")

    @property
    def score(self) -> SelectionScore:
        return self.candidate.score

    @property
    def partition_signature(self) -> str:
        return self.candidate.partition.signature

    @property
    def family(self) -> CandidateFamily:
        return (
            "raw_fusion"
            if isinstance(self.candidate, RawFusionCandidate)
            else "direct_partition"
        )

    @property
    def eligible_for_selection(self) -> bool:
        return bool(self.candidate.eligible_for_selection)

    @property
    def lambda_value(self) -> float | None:
        if isinstance(self.candidate, RawFusionCandidate):
            return float(self.candidate.raw_fit.provenance.lambda_value)
        return None

    @property
    def n_clusters(self) -> int:
        return int(self.candidate.partition.n_clusters)

    @property
    def penalized_objective(self) -> float | None:
        if isinstance(self.candidate, RawFusionCandidate):
            return float(self.candidate.raw_fit.objective.total)
        return None

    @property
    def mm_consistency_violations(self) -> int:
        if isinstance(self.candidate, RawFusionCandidate):
            return int(self.candidate.raw_fit.convergence.mm_consistency_violations)
        return 0


@dataclass(frozen=True, slots=True)
class RawAttemptTrace:
    """Immutable diagnostics only: never retain a raw fit, primal or witness."""

    search_round: int
    search_phase: str
    lambda_value: float
    source: str
    start_value: float
    breakpoint_escape_changed_count: int
    mathematically_certified: bool
    objective: float
    kkt_components: KKTComponents
    kkt_tolerance: float
    dominant_kkt_component: str
    certificate_status: str
    certificate_certified: bool
    certificate_admissible: bool
    convergence: ConvergenceResult
    work: WorkCounters
    outer_max_iter: int
    inner_max_iter: int
    certificate_max_iter: int
    working_dtype: str
    audit_dtype: str
    precision_polished: bool
    promotion_status: str = "not_recorded"
    fallback_reason: str = ""

    @property
    def kkt_residual(self) -> float:
        return self.kkt_components.residual


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    """Small immutable trace retained after the adaptive search finishes."""

    search_round: int = -1
    search_phase: str = "unknown"
    start_source: str = "not_applicable"
    start_value: float | None = None
    breakpoint_escape_changed_count: int = 0
    raw_attempts: tuple[RawAttemptTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """One immutable candidate plus selection-decision annotations."""

    record: CandidateRecord
    selected: bool

    @property
    def candidate_id(self) -> int:
        return int(self.record.candidate_id)

    @property
    def candidate(self) -> SelectablePartitionCandidate:
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
