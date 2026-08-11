from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult, RawClonalAnchorSpec, fit_fixed_objective
from ..core.fusion.defaults import (
    normalize_dense_fallback_policy,
    normalize_inner_backend,
)
from ..core.fusion.graph import build_likelihood_noise_regularized_adaptive_graph
from ..core.fusion.graph_ops import (
    build_likelihood_noise_regularized_adaptive_tensor_graph,
    tensor_graph_to_pairwise_graph,
)
from ..core.fusion.partition_starts import (
    PartitionCandidate,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.solver import (
    escape_path_breakpoint_solver_state,
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
    torch_data_from_context,
    uses_nonconvex_path_likelihood,
)
from ..core.fusion.torch_backend import dtype_name, mutation_region_terms_torch
from ..core.fusion.types import (
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    DenseWarmState,
    ExactSolverResourceLimit,
    PrimalOnlyWarmState,
    QuotientWorksetWarmState,
    SolverContext,
    SolverState,
)
from ..io.data import TumorData, tumor_objective_fingerprint
from ..core.bic import cluster_sizes_from_labels, compute_partition_icl

from ..model_selection.candidates import (
    _CachedPartitionRefit,
    _evaluate_candidate,
    validate_candidate_identity,
)
from ..model_selection.config import (
    FINAL_PHI_WARD_LADDER_KMAX,
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
    PARTITION_GUIDED_ADMM_MAX_SOLVER_RETRIES_PER_LAMBDA,
    PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS,
)
from ..model_selection.guided_fusion import (
    GuidedFusionInitialization,
    build_guided_fusion_initialization,
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
    _partition_candidate_requested_k,
    _partition_signature,
)
from ..model_selection.scoring import (
    _adaptive_score_column,
    _annotate_bic_diagnostics,
    _bic_selection_eligible_mask,
    _canonical_lambda,
    _exact_fusion_certificate_mask,
    _lambda_applicable_mask,
    _lambda_boundary_flags,
    _lambda_boundary_unresolved,
    _lambda_range_for_optimal_rows,
    _positive_exact_fusion_selection_mask,
    _prefer_fit_candidate,
    _row_bic_selection_eligible,
    _row_lambda_applicable,
    _row_lambda_if_applicable,
    _score_strictly_better,
    _select_best_partition_leftmost,
    _selected_lambda_signature_interval,
    _sorted_unique_lambdas,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateStaticMetadata,
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


@dataclass(frozen=True, slots=True)
class _RawClonalAnchorSearch:
    spec: RawClonalAnchorSpec
    total_eligible_candidates: int
    search_complete: bool
    screening_rule: str
    deviance_by_index: dict[int, float]
    lower_bound_by_index: dict[int, float]


@torch.no_grad()
def _build_raw_clonal_anchor_search(
    data: TumorData,
    context: SolverContext,
    *,
    fit_options: FitOptions,
) -> _RawClonalAnchorSearch:
    """Build the explicit hard-anchor seed set before the lambda path."""

    mode = str(fit_options.raw_clonal_anchor_mode).strip().lower()
    valid_modes = {"none", "specified_seed", "enumerated_seed", "screened_seed"}
    if mode not in valid_modes:
        raise ValueError(
            "raw_clonal_anchor_mode must be none, specified_seed, "
            "enumerated_seed, or screened_seed."
        )
    target = np.full(
        int(data.num_regions),
        float(fit_options.raw_clonal_anchor_target),
        dtype=np.float64,
    )
    feasibility_tolerance = float(fit_options.raw_clonal_anchor_feasibility_tol)
    if not np.all(np.isfinite(target)) or not np.all(target == 1.0):
        raise ValueError("Production raw clonal anchors require target CCF exactly 1.")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance < 0.0:
        raise ValueError("Raw clonal-anchor feasibility tolerance must be nonnegative.")
    upper = context.upper.detach().cpu().numpy()
    eligible = np.flatnonzero(
        np.all(target[None, :] <= upper + feasibility_tolerance, axis=1)
    ).astype(np.int64)
    if mode == "none":
        return _RawClonalAnchorSearch(
            spec=RawClonalAnchorSpec(
                mode="none",
                target=target,
                candidate_mutation_indices=(),
                feasibility_tolerance=feasibility_tolerance,
            ),
            total_eligible_candidates=int(eligible.size),
            search_complete=True,
            screening_rule="none",
            deviance_by_index={},
            lower_bound_by_index={},
        )
    if eligible.size == 0:
        raise RuntimeError("no_feasible_raw_clonal_anchor")
    torch_data = torch_data_from_context(context)
    free_phi = context.exact_pilot
    target_tensor = torch.as_tensor(
        target, dtype=free_phi.dtype, device=free_phi.device
    ).expand_as(free_phi)
    free_terms = mutation_region_terms_torch(
        torch_data,
        free_phi,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
    )
    anchored_terms = mutation_region_terms_torch(
        torch_data,
        target_tensor,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
    )
    free_loss = free_terms.loss.sum(dim=1)
    anchored_loss = anchored_terms.loss.sum(dim=1)
    increases = (anchored_loss - free_loss).detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    eligible = eligible[np.isfinite(increases[eligible])]
    if eligible.size == 0:
        raise RuntimeError("no_finite_feasible_raw_clonal_anchor")
    numerical_tolerance = max(
        1e-10,
        float(fit_options.eps) * (1.0 + abs(float(torch.sum(free_loss).item()))),
    )
    if float(np.min(increases[eligible])) < -numerical_tolerance:
        raise RuntimeError(
            "The zero-penalty pilot is inconsistent with raw anchor screening."
        )
    increases = np.maximum(increases, 0.0)
    ordered = sorted(
        eligible.tolist(),
        key=lambda index: (float(increases[index]), int(index)),
    )
    screening_rule = "none"
    if mode == "specified_seed":
        id_to_index = {str(value): index for index, value in enumerate(data.mutation_ids)}
        requested_ids = tuple(str(value) for value in fit_options.raw_clonal_anchor_mutation_ids)
        if len(requested_ids) != 1 or requested_ids[0] not in id_to_index:
            raise ValueError(
                "specified_seed requires exactly one retained mutation ID."
            )
        specified_index = int(id_to_index[requested_ids[0]])
        if specified_index not in set(eligible.tolist()):
            raise RuntimeError("no_feasible_raw_clonal_anchor")
        evaluated = (specified_index,)
        screening_rule = "user_specified_retained_mutation_id"
        search_complete = True
    elif mode == "enumerated_seed":
        evaluated = tuple(int(index) for index in ordered)
        screening_rule = "complete_feasible_seed_enumeration"
        search_complete = True
    else:
        candidate_max = fit_options.raw_clonal_anchor_candidate_max
        if candidate_max is None or int(candidate_max) < 1:
            raise ValueError("screened_seed requires a positive candidate maximum.")
        evaluated = tuple(int(index) for index in ordered[: int(candidate_max)])
        screening_rule = "zero_penalty_anchor_deviance_then_input_order"
        search_complete = len(evaluated) == len(ordered)
    base_loss = float(torch.sum(free_loss).item())
    return _RawClonalAnchorSearch(
        spec=RawClonalAnchorSpec(
            mode=mode,
            target=target,
            candidate_mutation_indices=evaluated,
            feasibility_tolerance=feasibility_tolerance,
        ),
        total_eligible_candidates=int(eligible.size),
        search_complete=bool(search_complete),
        screening_rule=screening_rule,
        deviance_by_index={
            int(index): float(2.0 * increases[int(index)]) for index in evaluated
        },
        lower_bound_by_index={
            int(index): float(base_loss + increases[int(index)]) for index in evaluated
        },
    )


def _raw_anchor_guide_labels(
    labels: np.ndarray,
    *,
    anchor_mutation_index: int | None,
) -> np.ndarray:
    """Split the fixed mutation from a multi-mutation guide block.

    The Ward/CEM guide remains the graph pilot.  Splitting only its warm-start
    label makes that starting state compatible with the raw fixed coordinate;
    it does not restrict which mutations the fusion estimator can recruit.
    """

    raw = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    if anchor_mutation_index is not None:
        index = int(anchor_mutation_index)
        if int(np.count_nonzero(raw == raw[index])) > 1:
            raw[index] = int(np.max(raw)) + 1
    canonical = np.empty_like(raw)
    mapping: dict[int, int] = {}
    for index, value in enumerate(raw.tolist()):
        canonical[index] = mapping.setdefault(int(value), len(mapping))
    return canonical


def _hash_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())


def _input_data_hash(data: TumorData) -> str:
    """Return the solver's complete, objective-sensitive input identity."""

    return tumor_objective_fingerprint(data)


def _edge_list_hash(edge_u: np.ndarray, edge_v: np.ndarray, edge_w: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    _hash_array(hasher, np.asarray(edge_u, dtype=np.int64))
    _hash_array(hasher, np.asarray(edge_v, dtype=np.int64))
    _hash_array(hasher, np.asarray(edge_w, dtype=np.float64))
    return hasher.hexdigest()


def _pilot_matrix_hash(pilot_phi: StartArray | None) -> str:
    if pilot_phi is None:
        return ""
    if torch.is_tensor(pilot_phi):
        array = pilot_phi.detach().cpu().numpy()
    else:
        array = np.asarray(pilot_phi)
    hasher = hashlib.blake2b(digest_size=16)
    _hash_array(hasher, np.asarray(array, dtype=np.float64))
    return hasher.hexdigest()


def _candidate_static_metadata(
    data: TumorData, graph, pilot_phi: StartArray | None = None
) -> CandidateStaticMetadata:
    edge_count = int(graph.edge_u.size)
    if edge_count:
        edge_weight_min = float(np.min(graph.edge_w))
        edge_weight_max = float(np.max(graph.edge_w))
        edge_weight_mean = float(np.mean(graph.edge_w))
    else:
        edge_weight_min = float("nan")
        edge_weight_max = float("nan")
        edge_weight_mean = float("nan")
    return CandidateStaticMetadata(
        edge_count=edge_count,
        edge_weight_min=edge_weight_min,
        edge_weight_max=edge_weight_max,
        edge_weight_mean=edge_weight_mean,
        edge_list_hash=_edge_list_hash(graph.edge_u, graph.edge_v, graph.edge_w),
        pilot_matrix_hash=_pilot_matrix_hash(pilot_phi),
        input_data_hash=_input_data_hash(data),
    )


def _clone_start(start: StartArray) -> StartArray:
    if torch.is_tensor(start):
        return start.detach().clone()
    return np.asarray(start).copy()


def _offload_solver_state_to_cpu(state: SolverState | None) -> SolverState | None:
    """Move persistent warm-start tensors off the accelerator.

    Online model selection retains several certified and failed candidates so
    later proposals can warm-start from either side of an observed bracket.
    A complete-graph dual has shape E x S and can exceed a GiB for large
    cohorts. Keeping every historical dual on CUDA makes memory scale with the
    number of evaluated lambdas even though only one state is used at a time.
    Host storage preserves the exact float dtype and values; the solver moves
    the selected state back to its runtime device when it is next used.
    """

    if state is None:
        return None

    cpu_tensors: dict[
        tuple[
            torch.device,
            torch.dtype,
            torch.layout,
            int,
            int,
            tuple[int, ...],
            tuple[int, ...],
        ],
        torch.Tensor,
    ] = {}

    def to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        detached = tensor.detach()
        if detached.numel() == 0:
            return detached.to(device="cpu")
        alias_key = (
            detached.device,
            detached.dtype,
            detached.layout,
            int(detached.untyped_storage().data_ptr()),
            int(detached.storage_offset()),
            tuple(int(value) for value in detached.shape),
            tuple(int(value) for value in detached.stride()),
        )
        cached = cpu_tensors.get(alias_key)
        if cached is None:
            cached = detached.to(device="cpu")
            cpu_tensors[alias_key] = cached
        return cached

    def certificate_to_cpu(certificate):
        if isinstance(certificate, DenseEdgeCertificate):
            return replace(certificate, dual=to_cpu(certificate.dual))
        if isinstance(certificate, CompressedEdgeCertificate):
            return replace(
                certificate,
                labels=to_cpu(certificate.labels),
                centers=to_cpu(certificate.centers),
                internal_edge_ids=to_cpu(certificate.internal_edge_ids),
                internal_dual=to_cpu(certificate.internal_dual),
            )
        return certificate

    warm_state = state.warm_state
    if isinstance(warm_state, DenseWarmState):
        warm_state = replace(
            warm_state,
            phi=to_cpu(warm_state.phi),
            dual=to_cpu(warm_state.dual),
        )
    elif isinstance(warm_state, QuotientWorksetWarmState):
        warm_state = replace(
            warm_state,
            phi=to_cpu(warm_state.phi),
            labels=to_cpu(warm_state.labels),
            centers=to_cpu(warm_state.centers),
            quotient_dual=to_cpu(warm_state.quotient_dual),
            internal_edge_ids=to_cpu(warm_state.internal_edge_ids),
            internal_dual=to_cpu(warm_state.internal_dual),
        )
    elif isinstance(warm_state, PrimalOnlyWarmState):
        warm_state = replace(
            warm_state,
            phi=to_cpu(warm_state.phi),
            structure_hint=to_cpu(warm_state.structure_hint),
            certificate_hint=certificate_to_cpu(warm_state.certificate_hint),
        )

    certificate = certificate_to_cpu(state.certificate)

    return SolverState(
        phi=to_cpu(state.phi),
        dual=to_cpu(state.dual),
        previous_lambda=float(state.previous_lambda),
        warm_state=warm_state,
        certificate=certificate,
        quotient_failure=state.quotient_failure,
        objective_spec_hash=str(state.objective_spec_hash),
    )


def _escape_path_breakpoint_retry_state(
    state: SolverState | None,
    *,
    start_source: str,
    start_lambda: float,
    target_lambda: float,
    context: SolverContext,
    tol: float,
) -> tuple[SolverState | None, int]:
    same_lambda_failure = start_source in {
        "same_lambda_retry",
        "best_same_lambda_kkt_state",
    } and _canonical_lambda(start_lambda) == _canonical_lambda(target_lambda)
    if state is None or not same_lambda_failure:
        return state, 0
    return escape_path_breakpoint_solver_state(state, context=context, tol=tol)


def _skip_terminal_solver_recovery(data: TumorData, proposal_phase: str) -> bool:
    return bool(
        str(proposal_phase) == "solver_recovery"
        and uses_nonconvex_path_likelihood(data)
    )


def _build_guided_initialization_with_resource_policy(
    *,
    data: TumorData,
    guide_phi: StartArray,
    guide_labels: np.ndarray | torch.Tensor,
    solver_context: SolverContext,
    fit_options: FitOptions,
) -> tuple[GuidedFusionInitialization, SolverContext, StartArray]:
    """Build guided state with typed allocation failure and optional CPU retry."""

    fallback_policy = normalize_dense_fallback_policy(fit_options.dense_fallback_policy)

    def build(
        *,
        context: SolverContext,
        phi: StartArray,
        labels: np.ndarray | torch.Tensor,
    ) -> GuidedFusionInitialization:
        requested_backend = normalize_inner_backend(fit_options.inner_backend)
        # The dense solver needs the guide's actual edge dual to preserve the
        # historical one-candidate warm-start path.  A compressed guide is the
        # right representation for quotient/workset, but feeding it to dense
        # discards that high-quality dual and can turn one certified fit into a
        # long recovery/search sequence.
        materialize_dense_dual = bool(
            fit_options.materialize_full_dual or requested_backend == "dense"
        )
        return build_guided_fusion_initialization(
            phi,
            labels,
            solver_context=context,
            partition_tolerance=max(float(fit_options.tol), 1e-8),
            kkt_atol=float(fit_options.tol),
            materialize_dense_dual=materialize_dense_dual,
        )

    try:
        return (
            build(context=solver_context, phi=guide_phi, labels=guide_labels),
            solver_context,
            guide_phi,
        )
    except (MemoryError, torch.OutOfMemoryError) as exc:
        cpu_fallback_allowed = bool(
            fallback_policy == "cpu_allowed"
            and solver_context.runtime.device.type != "cpu"
        )
        if not cpu_fallback_allowed:
            if isinstance(exc, ExactSolverResourceLimit):
                raise
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: guided initialization exhausted "
                f"memory on {solver_context.runtime.device_name}."
            ) from exc

        try:
            cpu_guide_phi: StartArray = (
                guide_phi.detach().to(device="cpu")
                if torch.is_tensor(guide_phi)
                else np.asarray(guide_phi)
            )
            cpu_guide_labels = (
                guide_labels.detach().to(device="cpu")
                if torch.is_tensor(guide_labels)
                else np.asarray(guide_labels)
            )
            cpu_context = prepare_torch_problem_with_resource_policy(
                data,
                dense_fallback_policy="device_only",
                inherited_resource_fallback="dense_cpu",
                major_prior=float(solver_context.problem.major_prior),
                eps=float(solver_context.problem.eps),
                tol=float(fit_options.tol),
                graph=solver_context.graph_spec,
                inner_max_iter=max(int(fit_options.inner_max_iter), 16),
                adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
                adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
                adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
                exact_pilot=cpu_guide_phi,
                pooled_start=cpu_guide_phi,
                scalar_well_starts=solver_context.scalar_well_starts,
                device="cpu",
                dtype=dtype_name(solver_context.runtime.dtype),
                objective_shape=str(fit_options.objective_shape),
                clonal_anchor_mutation_index=(
                    solver_context.clonal_anchor_mutation_index
                ),
                clonal_anchor_target=solver_context.clonal_anchor_target,
                clonal_anchor_source=str(solver_context.clonal_anchor_source),
                clonal_anchor_mode=str(solver_context.clonal_anchor_mode),
                clonal_anchor_feasibility_tolerance=float(
                    solver_context.clonal_anchor_feasibility_tolerance
                ),
            )
            guided = build(
                context=cpu_context,
                phi=cpu_guide_phi,
                labels=cpu_guide_labels,
            )
        except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
            if isinstance(cpu_exc, ExactSolverResourceLimit):
                raise cpu_exc from exc
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: guided initialization exhausted "
                "host memory during dense CPU fallback."
            ) from cpu_exc
        return guided, cpu_context, cpu_guide_phi


def _build_partition_guided_graph_with_resource_policy(
    *,
    guide_phi: StartArray,
    guide_curvature: torch.Tensor,
    solver_context: SolverContext,
    fit_options: FitOptions,
    noise_divisor: float,
):
    """Build the adaptive graph on CUDA, with an explicitly authorized host retry."""

    graph_options = {
        "gamma": float(fit_options.adaptive_weight_gamma),
        "minimum_tau": max(
            float(fit_options.adaptive_weight_floor), float(fit_options.eps)
        ),
        "baseline": float(fit_options.adaptive_weight_baseline),
        "noise_divisor": float(noise_divisor),
    }

    def host_array(value):
        return (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )

    def build_host_graph():
        return build_likelihood_noise_regularized_adaptive_graph(
            host_array(guide_phi),
            host_array(guide_curvature),
            lower=host_array(solver_context.lower),
            upper=host_array(solver_context.upper),
            count_observed=(
                None
                if solver_context.problem.count_observed is None
                else host_array(solver_context.problem.count_observed)
            ),
            **graph_options,
        )

    runtime = solver_context.runtime
    if runtime.device.type != "cuda":
        graph, tau = build_host_graph()
        return graph, None, tau

    try:
        tensor_graph, tau = build_likelihood_noise_regularized_adaptive_tensor_graph(
            torch.as_tensor(
                guide_phi,
                dtype=runtime.dtype,
                device=runtime.device,
            ),
            guide_curvature,
            runtime,
            lower=solver_context.lower,
            upper=solver_context.upper,
            count_observed=solver_context.problem.count_observed,
            **graph_options,
        )
        return tensor_graph_to_pairwise_graph(tensor_graph), tensor_graph, tau
    except (MemoryError, torch.OutOfMemoryError) as exc:
        if (
            normalize_dense_fallback_policy(fit_options.dense_fallback_policy)
            != "cpu_allowed"
        ):
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: partition-guided graph construction "
                f"exhausted memory on {runtime.device_name}; host retry is disabled."
            ) from exc
        try:
            graph, tau = build_host_graph()
        except (MemoryError, torch.OutOfMemoryError) as host_exc:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: partition-guided graph construction "
                "exhausted host memory during the authorized CPU retry."
            ) from host_exc
        return graph, None, tau


def _rescore_partition_candidates(
    candidates: list[PartitionCandidate],
    *,
    data: TumorData,
    normalized_score: str,
    bic_df_scale: float,
    bic_cluster_penalty: float,
) -> list[PartitionCandidate]:
    """Put the active selection score in ``PartitionCandidate.bic``.

    Candidate generation historically used that field for per-K ordering,
    refinement focus, and deduplication.  Keeping the field as the active score
    lets those operations follow the requested criterion while the candidate
    output rows continue to report classic BIC explicitly.
    """
    # Ward/CEM is proposal-only. Partition ICL may order initializer starts,
    # but it never enters the production selection pool.
    rescored: list[PartitionCandidate] = []
    for candidate in candidates:
        cluster_sizes = cluster_sizes_from_labels(candidate.labels)
        selected_score = compute_partition_icl(
            -float(candidate.fit_loss),
            cluster_sizes,
            data=data,
        )
        diagnostics = dict(candidate.diagnostics)
        diagnostics["partition_generation_selection_score"] = float(selected_score)
        rescored.append(
            replace(
                candidate,
                bic=float(selected_score),
                diagnostics=diagnostics,
            )
        )
    return rescored


def _partition_pool_row_metadata(
    pool: PartitionInitializerPool,
) -> dict[str, float | int | str]:
    return {
        "partition_generation_elapsed_seconds": float(pool.generation_elapsed_seconds),
        "partition_curvature_elapsed_seconds": float(pool.curvature_elapsed_seconds),
        "partition_ward_elapsed_seconds": float(pool.ward_elapsed_seconds),
        "partition_refine_ward_elapsed_seconds": float(
            pool.refine_ward_elapsed_seconds
        ),
        "partition_initial_generation_elapsed_seconds": float(
            pool.initial_generation_elapsed_seconds
        ),
        "partition_refine_generation_elapsed_seconds": float(
            pool.refine_generation_elapsed_seconds
        ),
        "partition_candidate_count": int(len(pool.candidates)),
        "partition_candidate_refinement_reason": str(pool.refinement_reason),
        "partition_candidate_sparse_k_grid": ",".join(map(str, pool.sparse_k_grid)),
        "partition_candidate_refine_k_grid": ",".join(map(str, pool.refine_k_grid)),
        "partition_candidate_k_grid": ",".join(map(str, pool.combined_k_grid)),
    }


def _assemble_selection_result(
    *,
    search_df,
    data,
    normalized_score,
    result_entries,
    bic_df_scale,
    bic_cluster_penalty,
    selection_method,
    profile_name,
    lambda_search_mode,
    adaptive_search_stop_reason,
    adaptive_search_rounds_completed,
    adaptive_refinement_rounds_completed,
    selection_start_time,
    strict_positive_exact_fusion: bool = False,
) -> BICSelectionResult:
    search_df = _annotate_bic_diagnostics(search_df)
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    candidate_selection_eligible_mask = (
        _positive_exact_fusion_selection_mask(search_df)
        if strict_positive_exact_fusion
        else _bic_selection_eligible_mask(search_df)
    )
    if strict_positive_exact_fusion:
        search_df["bic_selection_eligible"] = candidate_selection_eligible_mask
    num_converged_candidates = int(np.sum(converged_mask))
    num_selection_eligible_candidates = int(np.sum(candidate_selection_eligible_mask))
    if num_selection_eligible_candidates == 0:
        raise NoEligibleModelSelectionCandidatesError(
            tumor_id=data.tumor_id,
            search_df=search_df,
        )
    selection_df = search_df.loc[candidate_selection_eligible_mask].copy()
    selection_used_convergence_fallback = False

    score_column = _adaptive_score_column(normalized_score)
    best_row, selection_metric_value, selection_mask = _select_best_partition_leftmost(
        selection_df,
        score_column=score_column,
    )
    selected_lambda_applicable = _row_lambda_applicable(best_row)
    model_key = (
        "selection_model_signature"
        if "selection_model_signature" in selection_df.columns
        else "partition_signature"
    )
    selected_signature_mask = (
        selection_df[model_key]
        .astype(str)
        .eq(str(best_row[model_key]))
        .to_numpy(dtype=bool)
    )
    selection_lambda_min, selection_lambda_max, selection_lambda_count = (
        _lambda_range_for_optimal_rows(
            selection_df,
            selected_signature_mask,
        )
    )
    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)

    all_scores = (
        search_df[score_column].to_numpy(dtype=float)
        if score_column in search_df.columns
        else np.full(search_df.shape[0], np.nan, dtype=float)
    )
    all_objectives = (
        search_df["penalized_objective"].to_numpy(dtype=float)
        if "penalized_objective" in search_df.columns
        else np.full(search_df.shape[0], np.nan, dtype=float)
    )
    all_mm_violations = (
        search_df["mm_consistency_violations"].to_numpy(dtype=float)
        if "mm_consistency_violations" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=float)
    )
    provisional_mask = (
        np.isfinite(all_scores)
        & np.isfinite(all_objectives)
        & (all_mm_violations <= 0.0)
    )
    provisional_df = search_df.loc[provisional_mask].copy()
    if provisional_df.empty:
        best_score_all_row = None
    else:
        best_score_all_row = provisional_df.sort_values(
            [score_column, "lambda", "selection_step"],
            ascending=[True, True, True],
            na_position="last",
        ).iloc[0]
    certified_score_df = search_df.loc[
        candidate_selection_eligible_mask & provisional_mask
    ].copy()
    if certified_score_df.empty:
        best_score_certified_row = None
    else:
        best_score_certified_row = certified_score_df.sort_values(
            [score_column, "lambda", "selection_step"],
            ascending=[True, True, True],
            na_position="last",
        ).iloc[0]

    num_candidates_all = int(search_df.shape[0])
    num_candidates_certified = int(np.sum(candidate_selection_eligible_mask))

    selected_kkt_residual = (
        float(best_row["fixed_objective_kkt_residual"])
        if "fixed_objective_kkt_residual" in best_row
        and np.isfinite(float(best_row["fixed_objective_kkt_residual"]))
        else None
    )
    selected_provisional_score = float(best_row.get(score_column, np.nan))
    best_score_all_evaluated_lambda = None
    best_score_all_evaluated_kkt_residual = None
    best_score_all_evaluated_selection_eligible = False
    if best_score_all_row is not None:
        best_score_all_evaluated_lambda = _row_lambda_if_applicable(best_score_all_row)
        best_score_all_evaluated_kkt_residual = (
            float(best_score_all_row["fixed_objective_kkt_residual"])
            if np.isfinite(
                float(best_score_all_row.get("fixed_objective_kkt_residual", np.nan))
            )
            else None
        )
        best_score_all_evaluated_selection_eligible = bool(
            _row_bic_selection_eligible(best_score_all_row)
        )
    best_score_certified_lambda = None
    best_score_certified_kkt_residual = None
    if best_score_certified_row is not None:
        best_score_certified_lambda = _row_lambda_if_applicable(
            best_score_certified_row
        )
        best_score_certified_kkt_residual = (
            float(best_score_certified_row["fixed_objective_kkt_residual"])
            if np.isfinite(
                float(
                    best_score_certified_row.get("fixed_objective_kkt_residual", np.nan)
                )
            )
            else None
        )

    selection_optimizer_limited = False
    selection_optimizer_limited_reason = "none"
    optimizer_limited_ids: set[int] = set()
    if best_score_all_row is not None:
        best_score_all_score = float(best_score_all_row.get(score_column, np.nan))
        best_score_all_eligible = bool(_row_bic_selection_eligible(best_score_all_row))
        if (
            _score_strictly_better(
                best_score_all_score,
                selected_provisional_score,
            )
            and not best_score_all_eligible
        ):
            selection_optimizer_limited = True
            selection_optimizer_limited_reason = "best_provisional_score_failed_kkt"

    if "_candidate_id" in search_df.columns and np.isfinite(selected_provisional_score):
        for _, candidate_row in search_df.loc[
            provisional_mask & ~candidate_selection_eligible_mask
        ].iterrows():
            candidate_score = float(candidate_row.get(score_column, np.nan))
            if _score_strictly_better(
                candidate_score,
                selected_provisional_score,
            ):
                optimizer_limited_ids.add(int(candidate_row["_candidate_id"]))

    lambda_applicable_mask = _lambda_applicable_mask(selection_df)
    selection_boundary_lambda_values = selection_lambda_values[lambda_applicable_mask]
    selection_lower_hit, selection_upper_hit = _lambda_boundary_flags(
        selection_boundary_lambda_values,
        best_lambda_min=selection_lambda_min,
        best_lambda_max=selection_lambda_max,
    )
    selection_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_boundary_lambda_values,
        lower_hit=selection_lower_hit,
        upper_hit=selection_upper_hit,
    )
    selection_optimal_ids = set(
        selection_df.loc[selection_mask, "_candidate_id"].astype(int).tolist()
    )
    final_adaptive_search_stop_reason = adaptive_search_stop_reason
    eligible_mask = candidate_selection_eligible_mask
    search_df["eligible_for_selection"] = eligible_mask
    lambda_values_evaluated = ",".join(
        f"{float(value):.12g}"
        for value in _sorted_unique_lambdas(search_df["lambda"].to_numpy(dtype=float))
    )
    search_df["lambda_values_evaluated"] = lambda_values_evaluated
    if "optimizer_limited_candidate" not in search_df.columns:
        search_df["optimizer_limited_candidate"] = False
    if optimizer_limited_ids and "_candidate_id" in search_df.columns:
        search_df["optimizer_limited_candidate"] = (
            search_df["_candidate_id"].astype(int).isin(optimizer_limited_ids)
        )
    search_df["is_selection_optimal"] = (
        search_df["_candidate_id"].astype(int).isin(selection_optimal_ids)
    )
    selected_candidate_id = int(best_row["_candidate_id"])
    search_df["is_selected_best_row"] = (
        search_df["_candidate_id"].astype(int) == selected_candidate_id
    )
    search_df["adaptive_search_stop_reason"] = str(final_adaptive_search_stop_reason)
    selected_lambda_left, selected_lambda_right, selected_lambda_log10_width = (
        _selected_lambda_signature_interval(
            search_df,
            selected_candidate_id=selected_candidate_id,
            normalized_score=normalized_score,
        )
    )
    selected_lambda_representative_value = (
        float(best_row["lambda"]) if selected_lambda_applicable else np.nan
    )
    search_df["selected_lambda_representative"] = selected_lambda_representative_value
    search_df["selected_lambda_left"] = (
        np.nan if selected_lambda_left is None else float(selected_lambda_left)
    )
    search_df["selected_lambda_right"] = (
        np.nan if selected_lambda_right is None else float(selected_lambda_right)
    )
    search_df["selected_lambda_interval_log10_width"] = (
        np.nan
        if selected_lambda_log10_width is None
        else float(selected_lambda_log10_width)
    )
    selection_elapsed_seconds = float(perf_counter() - selection_start_time)
    search_df["selection_elapsed_seconds"] = float(selection_elapsed_seconds)

    best_fit, _, selected_candidate = result_entries[int(best_row["_candidate_id"])]
    if not isinstance(selected_candidate, RawFusionCandidate):
        raise AssertionError("Selected entry is not a raw fusion candidate.")
    validate_candidate_identity(selected_candidate)
    if selected_candidate.raw_fit is not best_fit:
        raise AssertionError("Selected raw fit identity changed during assembly.")
    if not selected_candidate.eligible_for_selection:
        raise AssertionError("Ineligible raw fusion candidate reached selection.")
    selected_model = SelectedModel(
        candidate=selected_candidate,
        selected_lambda=float(best_row["lambda"]),
        selected_partition_signature=str(selected_candidate.partition.signature),
        selected_partition_left_lambda=selected_lambda_left,
        selected_partition_right_lambda=selected_lambda_right,
    )
    search_df = search_df.drop(columns=["_candidate_id"])
    return BICSelectionResult(
        selected_model=selected_model,
        search_df=search_df,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        selection_method=selection_method,
        profile_name=profile_name,
        selection_metric_value=selection_metric_value,
        selection_lambda_min=selection_lambda_min,
        selection_lambda_max=selection_lambda_max,
        selection_lambda_count=selection_lambda_count,
        selection_hits_lower_boundary=selection_lower_hit,
        selection_hits_upper_boundary=selection_upper_hit,
        selection_boundary_unresolved=selection_boundary_unresolved,
        selection_optimum_resolved=not selection_boundary_unresolved,
        adaptive_search_rounds_completed=adaptive_search_rounds_completed,
        adaptive_search_stop_reason=str(final_adaptive_search_stop_reason),
        num_candidates=num_candidates,
        num_converged_candidates=num_converged_candidates,
        selection_used_convergence_fallback=selection_used_convergence_fallback,
        lambda_search_mode=str(lambda_search_mode),
        selected_lambda_representative=None
        if not selected_lambda_applicable
        else float(best_row["lambda"]),
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selected_lambda_interval_log10_width=selected_lambda_log10_width,
        adaptive_refinement_rounds_completed=int(adaptive_refinement_rounds_completed),
        num_candidates_all=num_candidates_all,
        num_candidates_certified=num_candidates_certified,
        selected_kkt_residual=selected_kkt_residual,
        best_score_all_evaluated_lambda=best_score_all_evaluated_lambda,
        best_score_all_evaluated_kkt_residual=best_score_all_evaluated_kkt_residual,
        best_score_all_evaluated_selection_eligible=best_score_all_evaluated_selection_eligible,
        best_score_certified_lambda=best_score_certified_lambda,
        best_score_certified_kkt_residual=best_score_certified_kkt_residual,
        selection_optimizer_limited=selection_optimizer_limited,
        selection_optimizer_limited_reason=selection_optimizer_limited_reason,
    )


def _partition_guided_admm_selection(
    *,
    data: TumorData,
    fit_options: FitOptions,
    use_warm_starts: bool,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> BICSelectionResult:
    """Select a positive pairwise-fusion fit with an online ADMM lambda search.

    The best Ward/CEM partition-ICL guide defines the default adaptive graph,
    supplies the primal start, and sets the initial lambda scale. The graph is
    then frozen for the complete raw-fusion path; Ward/CEM rows themselves are
    never selectable. In clonal-required mode, one mutation is selected once
    from the zero-penalty likelihood pilot and fixed at its feasible clonal CCF
    in every raw solve. Its pairwise weight contrast is bounded by a
    likelihood-curvature noise floor distributed over a mild degree^1.05
    complete-graph correction, avoiding the effectively infinite contrast of
    an exactly fused pilot with a fixed numerical floor. Blockwise KKT capacity
    supplies the first positive lambda and actual-dual state. Every subsequent
    lambda is proposed one at a time from observed certified ADMM fits; no
    lambda grid or multiplier sequence exists in this mode.
    """

    selection_start_time = perf_counter()
    bic_df_scale = 1.0
    bic_cluster_penalty = 0.0
    selection_score = str(fit_options.selection_score)
    normalized_score = selection_score
    profile_name = f"partition_guided_admm_{selection_score}"
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
    raw_anchor_search = _build_raw_clonal_anchor_search(
        data,
        pilot_context,
        fit_options=fit_options,
    )
    anchor_required = (
        str(fit_options.selection_anchor).strip().lower() == "clonal_required"
    )
    if anchor_required != (raw_anchor_search.spec.mode != "none"):
        raise ValueError(
            "Clonal fixed-partition BIC requires a raw clonal-anchor mode; "
            "unanchored BIC requires raw_clonal_anchor_mode='none'."
        )

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
        normalized_score="partition_icl",
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
        rescore_candidates=_rescore_partition_candidates,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        curvature=guide_curvature,
        curvature_elapsed_seconds=float(guide_curvature_elapsed),
    )
    guide = _best_partition_candidate(list(initializer_pool.candidates))
    if guide is None:
        raise RuntimeError(
            f"No finite partition-ICL initializer was available for tumor {data.tumor_id}."
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
        complete_graph_degree = float(max(int(data.num_mutations) - 1, 1))
        likelihood_noise_degree_exponent = float(
            PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT
        )
        likelihood_noise_divisor = float(
            complete_graph_degree**likelihood_noise_degree_exponent
        )
        selection_graph, prebuilt_tensor_graph, likelihood_noise_tau = (
            _build_partition_guided_graph_with_resource_policy(
                guide_phi=guide_phi,
                guide_curvature=guide_curvature,
                solver_context=pilot_context,
                fit_options=fit_options,
                noise_divisor=likelihood_noise_divisor,
            )
        )
        graph_source = "partition_guide_likelihood_noise_degree_regularized"
        graph_pilot_phi: StartArray = guide_phi
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
        exact_pilot=guide_phi,
        pooled_start=guide_phi,
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
    anchor_keys = (
        raw_anchor_search.spec.candidate_mutation_indices
        if raw_anchor_search.spec.mode != "none"
        else (-1,)
    )
    solver_context_by_anchor: dict[int, SolverContext] = {}
    guided_initialization_by_anchor: dict[int, GuidedFusionInitialization] = {}
    guide_phi_by_anchor: dict[int, StartArray] = {}
    guide_labels_by_anchor: dict[int, np.ndarray] = {}
    for anchor_key in anchor_keys:
        anchor_index = None if int(anchor_key) < 0 else int(anchor_key)
        context = (
            base_solver_context
            if anchor_index is None
            else prepare_torch_problem_with_resource_policy(
                data,
                dense_fallback_policy=str(fit_options.dense_fallback_policy),
                inherited_resource_fallback=pilot_context.resource_fallback,
                major_prior=float(fit_options.major_prior),
                eps=float(fit_options.eps),
                tol=float(fit_options.tol),
                graph=effective_graph,
                prebuilt_tensor_graph=effective_tensor_graph,
                inner_max_iter=max(int(fit_options.inner_max_iter), 16),
                adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
                adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
                adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
                exact_pilot=guide_phi,
                pooled_start=guide_phi,
                scalar_well_starts=pilot_context.scalar_well_starts,
                device=fit_options.device,
                dtype=fit_options.dtype,
                runtime=pilot_runtime,
                torch_data=pilot_torch_data,
                objective_shape=str(fit_options.objective_shape),
                clonal_anchor_mutation_index=anchor_index,
                clonal_anchor_target=raw_anchor_search.spec.target,
                clonal_anchor_source=str(raw_anchor_search.screening_rule),
                clonal_anchor_mode=str(raw_anchor_search.spec.mode),
                clonal_anchor_feasibility_tolerance=float(
                    raw_anchor_search.spec.feasibility_tolerance
                ),
            )
        )
        anchor_guide_labels = _raw_anchor_guide_labels(
            np.asarray(guide.labels, dtype=np.int64),
            anchor_mutation_index=context.clonal_anchor_mutation_index,
        )
        anchor_guide_phi: StartArray = context.exact_pilot
        initialization, context, anchor_guide_phi = (
            _build_guided_initialization_with_resource_policy(
                data=data,
                guide_phi=anchor_guide_phi,
                guide_labels=anchor_guide_labels,
                solver_context=context,
                fit_options=effective_fit_options,
            )
        )
        initialization = replace(
            initialization,
            solver_state=_offload_solver_state_to_cpu(initialization.solver_state),
        )
        key = int(anchor_key)
        solver_context_by_anchor[key] = context
        guided_initialization_by_anchor[key] = initialization
        guide_phi_by_anchor[key] = anchor_guide_phi
        guide_labels_by_anchor[key] = anchor_guide_labels
    runtime = next(iter(solver_context_by_anchor.values())).runtime
    if any(
        context.runtime.device != runtime.device
        or context.runtime.dtype != runtime.dtype
        for context in solver_context_by_anchor.values()
    ):
        raise RuntimeError("All raw anchor seeds must use one tensor runtime.")
    torch_data = torch_data_from_context(next(iter(solver_context_by_anchor.values())))
    effective_graph = next(iter(solver_context_by_anchor.values())).graph_spec
    effective_tensor_graph = next(iter(solver_context_by_anchor.values())).graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm CPU fallback changed the complete fusion graph."
        )
    prepare_elapsed_seconds = float(perf_counter() - prepare_start_time)

    controller = OnlineLambdaController(
        initial_lambda=float(
            min(
                initialization.lambda_value
                for initialization in guided_initialization_by_anchor.values()
            )
        ),
        initial_reason="partition_guide_kkt_balance",
        config=OnlineLambdaConfig(
            guide_n_clusters=int(
                max(np.unique(labels).size for labels in guide_labels_by_anchor.values())
            ),
            num_mutations=int(data.num_mutations),
            kkt_tolerance=5.0 * float(effective_fit_options.tol),
            max_unique_lambdas=int(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS),
            max_solver_retries_per_lambda=int(
                PARTITION_GUIDED_ADMM_MAX_SOLVER_RETRIES_PER_LAMBDA
            ),
        ),
    )

    result_entries: list[
        tuple[
            FitResult,
            dict[str, float | int | str | bool],
            RawFusionCandidate,
        ]
    ] = []
    fit_by_lambda: dict[float, FitResult] = {}
    fit_by_anchor_and_lambda: dict[tuple[int, float], FitResult] = {}
    attempts_by_anchor_and_lambda: dict[tuple[int, float], list[FitResult]] = {}
    bic_refit_cache: dict[object, _CachedPartitionRefit] = {}
    static_metadata = _candidate_static_metadata(
        data, effective_graph, pilot_phi=graph_pilot_phi
    )
    scalar_likelihood_pilot_hash = _pilot_matrix_hash(pilot_phi)
    next_step = 0
    terminal_stop_reason: str | None = None

    while True:
        proposal = controller.propose()
        if proposal is None:
            break
        if _skip_terminal_solver_recovery(data, proposal.phase):
            terminal_stop_reason = "online_lambda_uncertified_exact_fusion_result"
            break
        lambda_key = _canonical_lambda(proposal.lambda_value)
        candidate_fit_options = effective_fit_options
        if proposal.phase == "solver_recovery":
            candidate_fit_options = replace(
                effective_fit_options,
                outer_max_iter=max(int(effective_fit_options.outer_max_iter) * 5, 40),
                inner_max_iter=max(int(effective_fit_options.inner_max_iter) * 5, 150),
                # A tighter inner solve can leave a substantially better
                # observed-objective primal even though its final certificate
                # is stricter.  This recovery therefore strengthens rather
                # than relaxes the KKT gate.
                tol=max(
                    0.5 * float(effective_fit_options.tol),
                    float(np.finfo(np.float64).eps),
                ),
                objective_shape=objective_shape_for_data(
                    data, "unimodal_full_step_backtracking"
                ),
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
                tol=max(
                    0.5 * float(effective_fit_options.tol),
                    float(np.finfo(np.float64).eps),
                ),
            )

        raw_anchor_fit_start = perf_counter()
        seed_fits: list[tuple[int, FitResult]] = []
        seed_start_metadata: dict[int, tuple[str, float, int]] = {}
        for anchor_key in anchor_keys:
            key = int(anchor_key)
            context = solver_context_by_anchor[key]
            initialization = guided_initialization_by_anchor[key]
            anchor_guide_phi = guide_phi_by_anchor[key]
            warm_fit = None
            if proposal.warm_start_lambda is not None:
                warm_fit = fit_by_anchor_and_lambda.get(
                    (key, _canonical_lambda(proposal.warm_start_lambda))
                )
            alternate_fit = None
            if proposal.alternate_start_lambda is not None:
                alternate_fit = fit_by_anchor_and_lambda.get(
                    (key, _canonical_lambda(proposal.alternate_start_lambda))
                )
            same_lambda_attempts = attempts_by_anchor_and_lambda.get(
                (key, lambda_key), []
            )
            finite_failed = [
                attempt
                for attempt in same_lambda_attempts
                if attempt.solver_state is not None
                and np.isfinite(float(attempt.fixed_objective_kkt_residual))
            ]
            if proposal.phase == "solver_recovery" and finite_failed:
                best_failed_fit = min(
                    finite_failed,
                    key=lambda attempt: float(attempt.fixed_objective_kkt_residual),
                )
                solver_state_start = best_failed_fit.solver_state
                lambda_start_source = "best_same_lambda_kkt_state"
                lambda_start_value = float(best_failed_fit.lambda_value)
            elif proposal.phase in {"solver_recovery", "initial"}:
                solver_state_start = initialization.solver_state
                lambda_start_source = (
                    "guided_kkt_solver_recovery"
                    if proposal.phase == "solver_recovery"
                    else "guided_kkt_state"
                )
                lambda_start_value = float(initialization.lambda_value)
            elif (
                use_warm_starts
                and int(proposal.retry_number) == 1
                and alternate_fit is not None
                and alternate_fit.solver_state is not None
            ):
                solver_state_start = alternate_fit.solver_state
                lambda_start_source = "alternate_bracket_endpoint"
                lambda_start_value = float(proposal.alternate_start_lambda)
            elif (
                use_warm_starts
                and warm_fit is not None
                and warm_fit.solver_state is not None
            ):
                solver_state_start = warm_fit.solver_state
                lambda_start_source = (
                    "same_lambda_retry"
                    if int(proposal.retry_number) > 0
                    else "warm_endpoint"
                )
                lambda_start_value = float(proposal.warm_start_lambda)
            else:
                solver_state_start = initialization.solver_state
                lambda_start_source = "guided_kkt_fallback"
                lambda_start_value = float(initialization.lambda_value)
            solver_state_start, changed_count = _escape_path_breakpoint_retry_state(
                solver_state_start,
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
                else anchor_guide_phi
            )
            seed_fit = fit_fixed_objective(
                data=data,
                options=replace(
                    candidate_fit_options,
                    lambda_value=float(proposal.lambda_value),
                ),
                phi_start=phi_start,
                exact_pilot=anchor_guide_phi,
                pooled_start=anchor_guide_phi,
                scalar_well_starts=[],
                start_mode="warm_only",
                runtime=context.runtime,
                torch_data=torch_data_from_context(context),
                solver_context=context,
                solver_state=solver_state_start,
                compute_summary=False,
            )
            if seed_fit.solver_state is not None:
                seed_fit.solver_state = _offload_solver_state_to_cpu(
                    seed_fit.solver_state
                )
            initialization_state = initialization.solver_state
            if initialization_state is not None:
                guided_initialization_by_anchor[key] = replace(
                    initialization,
                    solver_state=_offload_solver_state_to_cpu(initialization_state),
                )
            attempts_by_anchor_and_lambda.setdefault((key, lambda_key), []).append(
                seed_fit
            )
            incumbent = fit_by_anchor_and_lambda.get((key, lambda_key))
            if _prefer_fit_candidate(seed_fit, incumbent):
                fit_by_anchor_and_lambda[(key, lambda_key)] = seed_fit
            seed_fits.append((key, seed_fit))
            seed_start_metadata[key] = (
                str(lambda_start_source),
                float(lambda_start_value),
                int(changed_count),
            )
        raw_anchor_fit_elapsed_seconds = float(
            perf_counter() - raw_anchor_fit_start
        )

        individually_certified = [
            (key, seed_fit)
            for key, seed_fit in seed_fits
            if float(seed_fit.lambda_value) > 0.0
            and bool(seed_fit.objective_faithful)
            and bool(seed_fit.full_kkt_certified)
            and bool(seed_fit.selection_eligible)
        ]
        if individually_certified:
            ranked_certified = sorted(
                individually_certified,
                key=lambda item: (
                    float(item[1].penalized_objective),
                    int(item[0]),
                ),
            )
            winning_anchor_key, winning_fit = ranked_certified[0]
            second_objective = (
                float(ranked_certified[1][1].penalized_objective)
                if len(ranked_certified) > 1
                else float("inf")
            )
            objective_gap = second_objective - float(winning_fit.penalized_objective)
            failed_seed_keys = {
                key for key, _ in seed_fits
            }.difference(key for key, _ in individually_certified)
            objective_tolerance = 1e-10 * (
                1.0 + abs(float(winning_fit.penalized_objective))
            )
            anchor_competition_resolved = not any(
                raw_anchor_search.lower_bound_by_index.get(key, float("inf"))
                <= float(winning_fit.penalized_objective) + objective_tolerance
                for key in failed_seed_keys
            )
        else:
            winning_anchor_key, winning_fit = min(
                seed_fits,
                key=lambda item: (
                    float(item[1].fixed_objective_kkt_residual),
                    float(item[1].penalized_objective),
                    int(item[0]),
                ),
            )
            objective_gap = float("inf")
            anchor_competition_resolved = False
        winning_fit.raw_clonal_anchor_search_complete = bool(
            raw_anchor_search.search_complete
        )
        winning_fit.raw_clonal_anchor_total_eligible_candidates = int(
            raw_anchor_search.total_eligible_candidates
        )
        winning_fit.raw_clonal_anchor_candidates_evaluated = int(
            len(seed_fits) if raw_anchor_search.spec.mode != "none" else 0
        )
        winning_fit.raw_clonal_anchor_objective_rank = int(
            1 if raw_anchor_search.spec.mode != "none" else 0
        )
        winning_fit.raw_clonal_anchor_objective_gap_to_second = float(objective_gap)
        winning_fit.raw_clonal_anchor_screening_rule = str(
            raw_anchor_search.screening_rule
        )
        winning_context = solver_context_by_anchor[int(winning_anchor_key)]
        winning_guide_phi = guide_phi_by_anchor[int(winning_anchor_key)]
        fit, row, artifact = _evaluate_candidate(
            data=data,
            fit_options=effective_fit_options,
            candidate_fit_options=candidate_fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            phi_start=None,
            exact_pilot=winning_guide_phi,
            pooled_start=winning_guide_phi,
            scalar_well_starts=[],
            start_mode="warm_only",
            runtime=runtime,
            torch_data=torch_data,
            solver_context=winning_context,
            solver_state=winning_fit.solver_state,
            compute_summary=False,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=next_step,
            lambda_value=float(proposal.lambda_value),
            selection_score=selection_score,
            bic_refit_cache=bic_refit_cache,
            static_metadata=static_metadata,
            precomputed_fit=winning_fit,
            raw_anchor_search_resolved=bool(anchor_competition_resolved),
        )
        lambda_start_source, lambda_start_value, path_breakpoint_escape_changed_count = (
            seed_start_metadata[int(winning_anchor_key)]
        )
        row["raw_fit_elapsed_seconds"] = float(raw_anchor_fit_elapsed_seconds)
        row["raw_anchor_seed_fit_elapsed_seconds"] = float(
            raw_anchor_fit_elapsed_seconds
        )
        row["raw_anchor_mean_seed_fit_elapsed_seconds"] = float(
            raw_anchor_fit_elapsed_seconds / max(len(seed_fits), 1)
        )
        row["raw_anchor_evaluated_seed_indices"] = ",".join(
            str(key) for key, _ in seed_fits if key >= 0
        )
        row["raw_anchor_evaluated_seed_mutation_ids"] = ",".join(
            str(data.mutation_ids[key]) for key, _ in seed_fits if key >= 0
        )
        row["raw_anchor_evaluated_penalized_objectives"] = ",".join(
            f"{key}:{float(seed_fit.penalized_objective):.17g}"
            for key, seed_fit in seed_fits
            if key >= 0
        )
        row["raw_anchor_evaluated_kkt_certified"] = ",".join(
            f"{key}:{int(bool(seed_fit.full_kkt_certified and seed_fit.selection_eligible))}"
            for key, seed_fit in seed_fits
            if key >= 0
        )
        guided_initialization = guided_initialization_by_anchor[
            int(winning_anchor_key)
        ]
        raw_guide_labels = guide_labels_by_anchor[int(winning_anchor_key)]
        guide_phi = guide_phi_by_anchor[int(winning_anchor_key)]
        guide_signature = _partition_signature(raw_guide_labels)
        guide_matrix_hash = _pilot_matrix_hash(guide_phi)

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
                "initialization_mode": "ward_cem_partition_icl_kkt",
                "initializer_selection_score": "partition_icl",
                "initializer_score_value": float(guide.bic),
                "initializer_K": int(np.unique(raw_guide_labels).size),
                "initializer_requested_K": int(_partition_candidate_requested_k(guide)),
                "initializer_source": (
                    f"{guide.source}_raw_anchor_compatible"
                    if raw_anchor_search.spec.mode != "none"
                    else str(guide.source)
                ),
                "initializer_partition_signature": str(guide_signature),
                "initializer_matrix_hash": str(guide_matrix_hash),
                "partition_guide_K": int(guide.K),
                "partition_guide_signature": str(partition_guide_signature),
                "partition_guide_matrix_hash": str(partition_guide_matrix_hash),
                "raw_clonal_anchor_selection_deviance_increase": (
                    np.nan
                    if int(winning_anchor_key) < 0
                    else float(
                        raw_anchor_search.deviance_by_index[int(winning_anchor_key)]
                    )
                ),
                "raw_clonal_anchor_second_ranked_deviance_increase": (
                    np.nan
                    if len(raw_anchor_search.deviance_by_index) < 2
                    else float(
                        sorted(raw_anchor_search.deviance_by_index.values())[1]
                    )
                ),
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
                "adaptive_candidate_budget": int(
                    PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS
                ),
                "adaptive_max_rounds": int(PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS),
                "adaptive_refine_per_round": 1,
                "adaptive_transition_probe_max_candidates": 0,
                "adaptive_initial_anchor_count": 0,
                "likelihood_partition_pool_enabled": True,
                "likelihood_partition_selection_enabled": False,
                **_partition_pool_row_metadata(initializer_pool),
            }
        )
        candidate_id = int(len(result_entries))
        row["_candidate_id"] = candidate_id
        result_entries.append((fit, row, artifact))
        incumbent = fit_by_lambda.get(lambda_key)
        if _prefer_fit_candidate(fit, incumbent):
            fit_by_lambda[lambda_key] = fit

        raw_exact_certified = bool(
            _exact_fusion_certificate_mask(pd.DataFrame([row]))[0]
            and bool(effective_tensor_graph.is_complete)
            and bool(anchor_competition_resolved)
        )
        selection_score_available = bool(artifact.eligible_for_selection)
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=float(proposal.lambda_value),
                n_clusters=int(row["n_clusters"]),
                partition_signature=str(
                    row.get("selection_model_signature", row["partition_signature"])
                ),
                # The active selection score steers the online-lambda
                # controller (the observation field name is historical).
                partition_icl=(
                    float(artifact.score.value)
                    if selection_score_available
                    else float("inf")
                ),
                kkt_residual=float(row["fixed_objective_kkt_residual"]),
                exact_candidate_eligible=bool(raw_exact_certified),
                raw_objective_certified=bool(raw_exact_certified),
                partition_certified=bool(artifact.partition.certified),
                selection_score_available=selection_score_available,
                certificate_status=str(
                    row.get(
                        "full_kkt_certificate_status",
                        fit.outer_kkt_certificate_status,
                    )
                ),
                backend_name=str(row.get("inner_backend", fit.inner_solver)),
                solver_iterations=int(
                    row.get("backend_iterations", fit.inner_iterations)
                ),
                # Compatibility diagnostics for pre-provenance consumers.
                raw_kkt_eligible=bool(row.get("raw_kkt_eligible", False)),
                admm_iterations=int(fit.admm_iterations),
            )
        )
        next_step += 1
    if not result_entries:
        raise RuntimeError(
            f"No guided ADMM candidates were evaluated for tumor {data.tumor_id}."
        )
    search_df = (
        pd.DataFrame([row for _, row, _ in result_entries])
        .sort_values(["lambda", "selection_step"])
        .reset_index(drop=True)
    )
    stop_reason = str(
        terminal_stop_reason
        or controller.stop_reason
        or "online_lambda_no_terminal_reason"
    )
    refinement_rounds = sum(
        1 for proposal in controller.proposal_history if "refine" in str(proposal.phase)
    )
    return _assemble_selection_result(
        search_df=search_df,
        data=data,
        normalized_score=normalized_score,
        result_entries=result_entries,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_method=selection_method,
        profile_name=profile_name,
        lambda_search_mode="partition_guided_admm",
        adaptive_search_stop_reason=stop_reason,
        adaptive_search_rounds_completed=int(len(controller.proposal_history)),
        adaptive_refinement_rounds_completed=int(refinement_rounds),
        selection_start_time=selection_start_time,
        strict_positive_exact_fusion=True,
    )


def select_model(
    *,
    data: TumorData,
    fit_options: FitOptions,
    use_warm_starts: bool,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> BICSelectionResult:
    if int(ward_ladder_kmax) != 0:
        raise ValueError(
            "Final-phi Ward candidates are disabled in objective-faithful "
            "production selection; ward_ladder_kmax must be zero."
        )
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
        ward_ladder_kmax=int(ward_ladder_kmax),
    )


__all__ = [
    "BICSelectionResult",
    "NoEligibleModelSelectionCandidatesError",
    "select_model",
]
