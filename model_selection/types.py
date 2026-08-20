from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
import torch

from ..core.bic import SelectionScore
from ..core.model import FitResult

StartArray = np.ndarray | torch.Tensor


def _immutable_array(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FusionPartition:
    labels: np.ndarray
    signature: str
    n_clusters: int
    tolerance: float
    max_diameter: float
    diameter_exact: bool
    certified: bool
    source: Literal[
        "solver_quotient",
        "verified_primal_equalities",
        "tolerance_defined_primal",
        "legacy_connected_components",
    ]
    maximal: bool = False
    cross_close_edge_found: bool = False
    certificate_graph_hash_matches: bool = True
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


@dataclass(frozen=True)
class PartitionRefitSummary:
    labels: np.ndarray
    partition_signature: str
    phi: np.ndarray
    cluster_centers: np.ndarray
    loglik: float
    fit_loss: float
    nominal_df: int
    active_df: int
    finite_candidate_found: bool
    global_optimum_certified: bool
    loglik_source: str
    refit_numerically_resolved: bool = False
    refit_loglik_refinement_delta: float = float("inf")
    refit_max_center_refinement_delta: float = float("inf")
    refit_coordinate_count: int = 0
    refit_finite_coordinate_count: int = 0
    refit_total_grid_points: int = 0
    refit_max_grid_spacing: float = float("inf")
    refit_total_candidate_basins: int = 0
    refit_total_refined_candidates: int = 0
    refit_min_best_second_loss_gap: float = float("inf")
    global_lower_bound: float = float("-inf")
    global_optimality_gap: float = float("inf")
    global_certificate_method: str = "none"
    global_certificate_intervals: int = 0
    refit_mode: str = "interval_certified"

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


@dataclass(frozen=True)
class RawFusionCandidate:
    raw_fit: FitResult
    partition: FusionPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    raw_objective_certified: bool
    eligible_for_selection: bool
    ineligibility_reason: str
    computation_profile: str = "strict"


@dataclass(frozen=True)
class DirectPartition:
    labels: np.ndarray
    signature: str
    n_clusters: int
    source: Literal[
        "pilot_hessian_ward",
        "pilot_hessian_ward_cem",
        "pilot_hessian_ward_cem_component_death",
        "final_phi_hessian_ward",
        "final_phi_hessian_ward_cem",
        "final_phi_hessian_ward_cem_component_death",
    ]
    requested_k: int
    mutation_ids: tuple[str, ...]
    generation_contract_id: str
    parent_raw_candidate_id: int | None = None
    parent_raw_lambda: float | None = None
    parent_raw_phi_hash: str = ""
    cem_iterations: int = 0
    component_death_count: int = 0
    refinement_score_before: float = float("nan")
    refinement_score_after: float = float("nan")
    deterministic_generation: bool = True

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
        if int(self.n_clusters) != int(unique.size):
            raise ValueError("Direct partition cluster count is inconsistent.")
        mutation_ids = tuple(str(value) for value in self.mutation_ids)
        if len(mutation_ids) != labels.size or len(set(mutation_ids)) != labels.size:
            raise ValueError(
                "Direct partition mutation IDs must uniquely identify every label."
            )
        if int(self.requested_k) < 1:
            raise ValueError("Direct partition requested_k must be positive.")
        if not str(self.signature) or not str(self.generation_contract_id):
            raise ValueError("Direct partition identity provenance is required.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "mutation_ids", mutation_ids)


@dataclass(frozen=True)
class DirectPartitionCandidate:
    partition: DirectPartition
    refit: PartitionRefitSummary
    score: SelectionScore
    eligible_for_selection: bool
    ineligibility_reason: str
    computation_profile: str


SelectablePartitionCandidate = Union[RawFusionCandidate, DirectPartitionCandidate]
CandidateFamily = Literal["raw_fusion", "direct_partition"]


@dataclass(frozen=True)
class CandidateRecord:
    """Typed selection unit with an inert reporting payload.

    ``row`` is retained for the in-memory search-table compatibility surface.
    Numerical admission and model choice must use the typed properties below,
    never values recovered from that reporting dictionary.
    """

    candidate_id: int
    candidate: SelectablePartitionCandidate
    row: dict[str, object]

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
            return float(self.candidate.raw_fit.lambda_value)
        return None

    @property
    def n_clusters(self) -> int:
        return int(self.candidate.partition.n_clusters)

    @property
    def penalized_objective(self) -> float | None:
        if isinstance(self.candidate, RawFusionCandidate):
            return float(
                getattr(self.candidate.raw_fit, "penalized_objective", float("nan"))
            )
        return None

    @property
    def mm_consistency_violations(self) -> int:
        if isinstance(self.candidate, RawFusionCandidate):
            return int(getattr(self.candidate.raw_fit, "mm_consistency_violations", 0))
        return 0


@dataclass(frozen=True, slots=True)
class CandidateSelectionDecision:
    """Complete typed outcome of partition-first candidate selection."""

    selected: CandidateRecord
    representative_ids: frozenset[int]
    eligible_ids: frozenset[int]
    optimal_ids: frozenset[int]
    optimizer_limited_ids: frozenset[int]
    selected_lambda_left: float | None
    selected_lambda_right: float | None
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool


@dataclass(frozen=True)
class SelectedModel:
    raw_reference: RawFusionCandidate
    partition_candidate: SelectablePartitionCandidate
    selected_partition_signature: str
    selected_candidate_family: str
    selected_lambda: float | None
    selected_partition_left_lambda: float | None
    selected_partition_right_lambda: float | None
    # For a final-raw-Phi direct proposal this is the exact raw candidate whose
    # Phi generated the deterministic Ward/CEM ladder.  It is deliberately
    # separate from ``raw_reference``, which is selected independently as the
    # best certified raw-fusion result for estimator provenance.
    partition_parent_raw: RawFusionCandidate | None = None

    def __post_init__(self) -> None:
        raw = self.raw_reference
        candidate = self.partition_candidate
        strict = str(candidate.computation_profile) == "strict"
        if self.selected_partition_signature != candidate.partition.signature:
            raise ValueError("Selected-model partition signature is inconsistent.")
        expected_family = (
            "raw_fusion"
            if isinstance(candidate, RawFusionCandidate)
            else "direct_partition"
        )
        if str(self.selected_candidate_family) != expected_family:
            raise ValueError("Selected-model candidate family is inconsistent.")
        if not candidate.eligible_for_selection:
            raise ValueError("Selected model must be eligible for selection.")
        if isinstance(candidate, RawFusionCandidate):
            if self.partition_parent_raw is not None:
                raise ValueError("Raw selected models cannot have a direct parent.")
            if self.selected_lambda is None or not np.isclose(
                float(self.selected_lambda),
                float(candidate.raw_fit.lambda_value),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Selected-model lambda is inconsistent with its raw fit."
                )
            if not candidate.raw_objective_certified:
                raise ValueError("Selected raw model must have a certified objective.")
            if not candidate.partition.certified:
                raise ValueError("Selected raw model must have a certified partition.")
        else:
            if self.selected_lambda is not None:
                raise ValueError(
                    "Direct partition candidates do not have a selected lambda."
                )
            parent_id = candidate.partition.parent_raw_candidate_id
            if (parent_id is None) != (self.partition_parent_raw is None):
                raise ValueError(
                    "Direct-partition parent provenance is incomplete or spurious."
                )
            if self.partition_parent_raw is not None:
                parent_lambda = candidate.partition.parent_raw_lambda
                if parent_lambda is None or not np.isclose(
                    float(parent_lambda),
                    float(self.partition_parent_raw.raw_fit.lambda_value),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError("Direct-partition parent lambda is inconsistent.")
        if strict and not candidate.refit.global_optimum_certified:
            raise ValueError("Selected model must have a globally certified refit.")
        if candidate.score.partition_signature != self.selected_partition_signature:
            raise ValueError("Selected-model score signature is inconsistent.")
        if (
            not raw.eligible_for_selection
            or not raw.raw_objective_certified
            or not raw.partition.certified
        ):
            raise ValueError("Raw reference must remain a certified raw-fusion model.")

@dataclass
class BICSelectionResult:
    selected_model: SelectedModel
    search_df: pd.DataFrame
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


@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    pilot_matrix_hash: str
    input_data_hash: str
