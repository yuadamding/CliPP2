from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult
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
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets_torch,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.refit import PartitionRefitResult
from ..core.fusion.solver import (
    prepare_torch_problem_with_resource_policy,
    torch_data_from_context,
)
from ..core.fusion.torch_backend import dtype_name
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
from ..io.data import TumorData
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    load_simulation_truth,
)
from ..core.bic import (
    LAMBDA_GRID_MODES,
    LambdaBracket,
    cluster_sizes_from_labels,
    is_adaptive_lambda_grid_mode,
    is_partition_guided_lambda_grid_mode,
)

from ..model_selection.adaptive import (
    _adaptive_first_pass_options,
    _adaptive_interval_proposal_records,
    _adaptive_interval_proposals as _adaptive_interval_proposals,
    _adaptive_score_column,
    _adaptive_transition_probe_records,
    _best_candidate_rows_by_lambda as _best_candidate_rows_by_lambda,
    _full_fusion_box_residual_with_dual_balls as _full_fusion_box_residual_with_dual_balls,
    _initial_adaptive_lambda_bracket,
    _lambda_boundary_flags,
    _lambda_boundary_unresolved,
    _score_maximized,
    _score_strictly_better,
    _selected_lambda_signature_interval,
)
from ..model_selection.candidates import (
    _evaluate_candidate,
    _evaluate_partition_candidate,
)
from ..model_selection.config import (
    ADAPTIVE_FIRST_PASS_INNER_MAX_ITER as ADAPTIVE_FIRST_PASS_INNER_MAX_ITER,
    ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER as ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER,
    ADAPTIVE_PATH_MAX_CANDIDATES,
    ADAPTIVE_PATH_MAX_ROUNDS,
    ADAPTIVE_PATH_PARTITION_POOL_MAX_CANDIDATES,
    ADAPTIVE_PATH_PARTITION_POOL_MAX_ROUNDS,
    ADAPTIVE_PATH_PARTITION_POOL_REFINE_PER_ROUND,
    ADAPTIVE_PATH_PARTITION_POOL_TRANSITION_PROBE_MAX_CANDIDATES,
    ADAPTIVE_PATH_REFINE_PER_ROUND,
    ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
    ENABLE_LIKELIHOOD_PARTITION_CANDIDATES,
    LIKELIHOOD_PARTITION_CEM_MAX_ITER,
    LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K,
    LIKELIHOOD_PARTITION_REFIT_MAX_ITER,
    PARTITION_ICL_DIRICHLET_ALPHA,
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
from ..model_selection.partition_initializer import generate_partition_initializer_pool
from ..model_selection.partitions import (
    _best_partition_candidate,
    _deduplicate_partition_candidates,
    _likelihood_partition_k_grid,
    _likelihood_partition_refinement_k_grid,
    _partition_candidate_requested_k,
    _partition_is_coarsening as _partition_is_coarsening,
    _partition_signature,
)
from ..model_selection.scoring import (
    _add_bic_selection_eligible as _add_bic_selection_eligible,
    _annotate_bic_diagnostics,
    _ari_candidate_frame,
    _bic_selection_eligible_mask,
    _canonical_lambda,
    _exact_fusion_certificate_mask,
    _is_bic_selection_eligible as _is_bic_selection_eligible,
    _lambda_applicable_mask,
    _lambda_range_for_optimal_rows,
    _lambda_warm_start_distance,
    _normalize_selection_score_name,
    _optimal_lambda_range,
    _positive_exact_fusion_selection_mask,
    _prefer_fit_candidate,
    _row_bic_selection_eligible,
    _row_lambda_applicable,
    _row_lambda_if_applicable,
    _selection_score_value,
    _sorted_unique_lambdas,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateStaticMetadata,
    ModelSelectionResult,
    SelectionArtifact,
    SimulationDiagnostics,
    StartArray,
)


def _hash_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())


def _input_data_hash(data: TumorData) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for value in data.mutation_ids:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(b"\1")
    for value in data.region_ids:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    for array in (
        data.alt_counts,
        data.total_counts,
        data.purity,
        data.major_cn,
        data.minor_cn,
        data.normal_cn,
        data.has_cna.astype(np.int8, copy=False),
        data.scaling,
        data.phi_upper,
    ):
        _hash_array(hasher, np.asarray(array))
    return hasher.hexdigest()


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
                scalar_well_starts=(),
                device="cpu",
                dtype=dtype_name(solver_context.runtime.dtype),
                objective_shape=str(fit_options.objective_shape),
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


def _fit_phi_start(fit: FitResult) -> StartArray:
    if fit.solver_state is not None and fit.solver_state.phi is not None:
        return fit.solver_state.phi
    return fit.phi


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
    rescored: list[PartitionCandidate] = []
    for candidate in candidates:
        cluster_sizes = cluster_sizes_from_labels(candidate.labels)
        selected_score, _, _, _ = _selection_score_value(
            loglik=-float(candidate.fit_loss),
            num_clusters=int(cluster_sizes.size),
            data=data,
            bic_df_scale=float(bic_df_scale),
            bic_cluster_penalty=float(bic_cluster_penalty),
            selection_score=normalized_score,
            cluster_sizes=cluster_sizes,
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


def _resolve_adaptive_path_config(
    use_partition_pool: bool,
) -> tuple[int, int, int, int]:
    """Adaptive lambda-path budgets as
    (max_candidates, max_rounds, refine_per_round, transition_probe_max_candidates),
    switching between the partition-pool and default budgets on one condition."""
    if use_partition_pool:
        return (
            ADAPTIVE_PATH_PARTITION_POOL_MAX_CANDIDATES,
            ADAPTIVE_PATH_PARTITION_POOL_MAX_ROUNDS,
            ADAPTIVE_PATH_PARTITION_POOL_REFINE_PER_ROUND,
            ADAPTIVE_PATH_PARTITION_POOL_TRANSITION_PROBE_MAX_CANDIDATES,
        )
    return (
        ADAPTIVE_PATH_MAX_CANDIDATES,
        ADAPTIVE_PATH_MAX_ROUNDS,
        ADAPTIVE_PATH_REFINE_PER_ROUND,
        ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
    )


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
    lambda_bracket,
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
        raise RuntimeError(
            f"No candidates were eligible for model selection for tumor {data.tumor_id}."
        )
    selection_df = search_df.loc[candidate_selection_eligible_mask].copy()
    converged_ari_df = _ari_candidate_frame(selection_df.copy())
    selection_used_convergence_fallback = False

    if selection_df.empty:
        raise RuntimeError(
            f"No candidate fits were evaluated for tumor {data.tumor_id}."
        )

    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)
    score_column = _adaptive_score_column(normalized_score)
    _, _, _, selection_metric_value, selection_mask = _optimal_lambda_range(
        selection_df[score_column].to_numpy(dtype=float),
        selection_lambda_values,
        maximize=False,
    )
    tied_df = selection_df.loc[selection_mask].copy()
    tied_df["_lambda_applicable_sort"] = _lambda_applicable_mask(tied_df)
    tied_df = tied_df.sort_values(
        [score_column, "_lambda_applicable_sort", "lambda", "selection_step"],
        ascending=[True, False, True, True],
    )
    best_row = tied_df.iloc[0]
    selected_lambda_applicable = _row_lambda_applicable(best_row)
    selection_lambda_min, selection_lambda_max, selection_lambda_count = (
        _lambda_range_for_optimal_rows(
            selection_df,
            selection_mask,
        )
    )

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
    partition_candidate_mask = (
        search_df["candidate_pool_source"]
        .astype(str)
        .eq("likelihood_partition")
        .to_numpy(dtype=bool)
        if "candidate_pool_source" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=bool)
    )
    provisional_mask = np.isfinite(all_scores) & (
        (np.isfinite(all_objectives) & (all_mm_violations <= 0.0))
        | partition_candidate_mask
    )
    provisional_df = search_df.loc[provisional_mask].copy()
    if provisional_df.empty:
        best_score_all_row = None
    else:
        best_score_all_row = provisional_df.sort_values(
            [score_column, "lambda", "selection_step"],
            ascending=[not _score_maximized(normalized_score), True, True],
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
            ascending=[not _score_maximized(normalized_score), True, True],
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
                normalized_score=normalized_score,
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
                normalized_score=normalized_score,
            ):
                optimizer_limited_ids.add(int(candidate_row["_candidate_id"]))

    best_ari_min, best_ari_max, best_ari_count, best_ari_value, ari_mask = (
        _optimal_lambda_range(
            selection_df["ARI"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=True,
        )
    )
    best_ari_min, best_ari_max, best_ari_count = _lambda_range_for_optimal_rows(
        selection_df, ari_mask
    )
    (
        best_converged_ari_min,
        best_converged_ari_max,
        best_converged_ari_count,
        best_converged_ari_value,
        best_converged_ari_mask,
    ) = _optimal_lambda_range(
        converged_ari_df["ARI"].to_numpy(dtype=float)
        if not converged_ari_df.empty
        else np.asarray([], dtype=float),
        converged_ari_df["lambda"].to_numpy(dtype=float)
        if not converged_ari_df.empty
        else np.asarray([], dtype=float),
        maximize=True,
    )
    best_converged_ari_min, best_converged_ari_max, best_converged_ari_count = (
        _lambda_range_for_optimal_rows(
            converged_ari_df,
            best_converged_ari_mask,
        )
    )
    _, _, _, best_ari_all_evaluated, _ = _optimal_lambda_range(
        search_df["ARI"].to_numpy(dtype=float)
        if "ARI" in search_df.columns
        else np.asarray([], dtype=float),
        search_df["lambda"].to_numpy(dtype=float)
        if "lambda" in search_df.columns
        else np.asarray([], dtype=float),
        maximize=True,
    )
    _, _, _, best_ari_certified, _ = _optimal_lambda_range(
        search_df.loc[candidate_selection_eligible_mask, "ARI"].to_numpy(dtype=float)
        if "ARI" in search_df.columns and np.any(candidate_selection_eligible_mask)
        else np.asarray([], dtype=float),
        search_df.loc[candidate_selection_eligible_mask, "lambda"].to_numpy(dtype=float)
        if "lambda" in search_df.columns and np.any(candidate_selection_eligible_mask)
        else np.asarray([], dtype=float),
        maximize=True,
    )
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
    ari_lower_hit, ari_upper_hit = _lambda_boundary_flags(
        selection_boundary_lambda_values,
        best_lambda_min=best_ari_min,
        best_lambda_max=best_ari_max,
    )
    ari_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_boundary_lambda_values,
        lower_hit=ari_lower_hit,
        upper_hit=ari_upper_hit,
    )
    selection_optimal_ids = set(
        selection_df.loc[selection_mask, "_candidate_id"].astype(int).tolist()
    )
    ari_optimal_ids = set(
        selection_df.loc[ari_mask, "_candidate_id"].astype(int).tolist()
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
    search_df["is_ari_optimal"] = (
        search_df["_candidate_id"].astype(int).isin(ari_optimal_ids)
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

    best_fit, best_evaluation, _, selected_artifact = result_entries[
        int(best_row["_candidate_id"])
    ]
    selected_candidate_ari = (
        float(best_row["ARI"]) if np.isfinite(float(best_row["ARI"])) else None
    )
    selected_ari = (
        float(best_evaluation.ari)
        if best_evaluation is not None
        else selected_candidate_ari
    )
    search_df = search_df.drop(columns=["_candidate_id"])
    simulation_diagnostics = SimulationDiagnostics(
        selected_evaluation=best_evaluation,
        selected_ari=selected_ari,
        best_ari=best_ari_value,
        ari_optimal_lambda_min=best_ari_min,
        ari_optimal_lambda_max=best_ari_max,
        ari_optimal_lambda_count=best_ari_count,
        best_converged_ari=best_converged_ari_value,
        best_converged_lambda_min=best_converged_ari_min,
        best_converged_lambda_max=best_converged_ari_max,
        best_converged_lambda_count=best_converged_ari_count,
        ari_hits_lower_boundary=ari_lower_hit,
        ari_hits_upper_boundary=ari_upper_hit,
        ari_boundary_unresolved=ari_boundary_unresolved,
        ari_optimum_resolved=not ari_boundary_unresolved,
        best_ari_all_evaluated=best_ari_all_evaluated,
        best_ari_certified=best_ari_certified,
    )
    return BICSelectionResult(
        best_fit=best_fit,
        selected_artifact=selected_artifact,
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
        lambda_bracket_min=None
        if lambda_bracket is None
        else float(lambda_bracket.lambda_min),
        lambda_bracket_eq=None
        if lambda_bracket is None
        else float(lambda_bracket.lambda_eq),
        lambda_bracket_full=None
        if lambda_bracket is None
        else float(lambda_bracket.lambda_full),
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
        simulation=simulation_diagnostics,
    )


def _partition_guided_admm_selection(
    *,
    data: TumorData,
    simulation_root: Path | None,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    profile_name: str,
    selection_method: str,
    selection_score: str,
) -> BICSelectionResult:
    """Select a positive pairwise-fusion fit with an online ADMM lambda search.

    Ward/CEM likelihood partitions are proposal-only.  The best partition-ICL
    proposal supplies a primal start and adaptive-weight pilot. Its pairwise
    weight contrast is bounded by a likelihood-curvature noise floor distributed
    over a mild degree^1.05 complete-graph correction, avoiding the effectively
    infinite contrast of an exactly fused pilot with a fixed numerical floor.
    Blockwise KKT capacity supplies the first positive lambda and actual-dual state. Every
    subsequent lambda is proposed one at a time from observed certified ADMM
    fits; no lambda grid or multiplier sequence exists in this mode.
    """

    selection_start_time = perf_counter()
    normalized_score = _normalize_selection_score_name(selection_score)
    if normalized_score != "partition_icl":
        raise ValueError(
            "partition_guided_admm currently requires selection_score='partition_icl'; "
            "use lambda_grid_mode='adaptive_bic' for another score."
        )
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
    solver_context = prepare_torch_problem_with_resource_policy(
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
        scalar_well_starts=(),
        device=fit_options.device,
        dtype=fit_options.dtype,
        runtime=pilot_runtime,
        torch_data=pilot_torch_data,
        objective_shape=str(fit_options.objective_shape),
    )
    effective_graph = solver_context.graph_spec
    effective_tensor_graph = solver_context.graph
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm requires the complete pairwise graph so the "
            "inner solver is ADMM."
        )
    effective_fit_options = replace(fit_options, graph=effective_graph)
    guided_initialization, solver_context, guide_phi = (
        _build_guided_initialization_with_resource_policy(
            data=data,
            guide_phi=guide_phi,
            guide_labels=np.asarray(guide.labels, dtype=np.int64),
            solver_context=solver_context,
            fit_options=effective_fit_options,
        )
    )
    runtime = solver_context.runtime
    torch_data = torch_data_from_context(solver_context)
    effective_graph = solver_context.graph_spec
    effective_tensor_graph = solver_context.graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    if not bool(effective_tensor_graph.is_complete) or int(
        effective_graph.degree_bound
    ) != int(data.num_mutations - 1):
        raise ValueError(
            "partition_guided_admm CPU fallback changed the complete fusion graph."
        )
    guided_initialization = replace(
        guided_initialization,
        solver_state=_offload_solver_state_to_cpu(guided_initialization.solver_state),
    )
    prepare_elapsed_seconds = float(perf_counter() - prepare_start_time)

    simulation_truth: SimulationTruth | None = None
    if (
        evaluate_all_candidates
        and simulation_root is not None
        and (simulation_root / data.tumor_id).exists()
    ):
        simulation_truth = load_simulation_truth(data, simulation_root)

    controller = OnlineLambdaController(
        initial_lambda=float(guided_initialization.lambda_value),
        initial_reason="partition_guide_kkt_balance",
        config=OnlineLambdaConfig(
            guide_n_clusters=int(guide.K),
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
            SimulationEvaluation | None,
            dict[str, float | int | str | bool],
            SelectionArtifact,
        ]
    ] = []
    fit_by_lambda: dict[float, FitResult] = {}
    bic_refit_cache: dict[str, PartitionRefitResult] = {}
    static_metadata = _candidate_static_metadata(
        data, effective_graph, pilot_phi=graph_pilot_phi
    )
    scalar_likelihood_pilot_hash = _pilot_matrix_hash(pilot_phi)
    guide_signature = _partition_signature(np.asarray(guide.labels, dtype=np.int64))
    guide_matrix_hash = _pilot_matrix_hash(guide_phi)
    next_step = 0

    while True:
        proposal = controller.propose()
        if proposal is None:
            break
        lambda_key = _canonical_lambda(proposal.lambda_value)
        warm_fit = None
        if proposal.warm_start_lambda is not None:
            warm_fit = fit_by_lambda.get(_canonical_lambda(proposal.warm_start_lambda))
        alternate_fit = None
        if proposal.alternate_start_lambda is not None:
            alternate_fit = fit_by_lambda.get(
                _canonical_lambda(proposal.alternate_start_lambda)
            )

        if proposal.phase == "solver_recovery":
            finite_failed_entries = [
                (fit, row)
                for fit, _, row, _ in result_entries
                if fit.solver_state is not None
                and _canonical_lambda(fit.lambda_value)
                == _canonical_lambda(proposal.lambda_value)
                and np.isfinite(float(row.get("fixed_objective_kkt_residual", np.nan)))
            ]
            if finite_failed_entries:
                best_failed_fit, best_failed_row = min(
                    finite_failed_entries,
                    key=lambda item: float(item[1]["fixed_objective_kkt_residual"]),
                )
                solver_state_start = best_failed_fit.solver_state
                lambda_start_source = "best_same_lambda_kkt_state"
                lambda_start_value = float(best_failed_fit.lambda_value)
            else:
                solver_state_start = guided_initialization.solver_state
                lambda_start_source = "guided_kkt_solver_recovery"
                lambda_start_value = float(guided_initialization.lambda_value)
        elif proposal.phase == "initial":
            solver_state_start = guided_initialization.solver_state
            lambda_start_source = "guided_kkt_state"
            lambda_start_value = float(guided_initialization.lambda_value)
        elif (
            use_warm_starts
            and int(proposal.retry_number) == 1
            and alternate_fit is not None
            and alternate_fit.solver_state is not None
        ):
            # A failed midpoint can be basin-dependent.  Retry once from the
            # other certified endpoint before spending more work at the failed
            # target state itself.
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
            solver_state_start = guided_initialization.solver_state
            lambda_start_source = "guided_kkt_fallback"
            lambda_start_value = float(guided_initialization.lambda_value)
        if proposal.phase == "solver_recovery":
            phi_start = _clone_start(
                solver_state_start.phi
                if solver_state_start is not None and solver_state_start.phi is not None
                else guide_phi
            )
        else:
            phi_start = _clone_start(
                solver_state_start.phi
                if solver_state_start is not None and solver_state_start.phi is not None
                else guide_phi
            )

        candidate_fit_options = effective_fit_options
        if proposal.phase == "solver_recovery":
            candidate_fit_options = replace(
                effective_fit_options,
                outer_max_iter=max(int(effective_fit_options.outer_max_iter) * 5, 40),
                inner_max_iter=max(int(effective_fit_options.inner_max_iter) * 5, 150),
                objective_shape="unimodal_full_step_backtracking",
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
            )

        fit, evaluation, row, artifact = _evaluate_candidate(
            data=data,
            fit_options=effective_fit_options,
            candidate_fit_options=candidate_fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            simulation_root=simulation_root,
            simulation_truth=simulation_truth,
            evaluate_candidate=evaluate_all_candidates,
            phi_start=phi_start,
            exact_pilot=guide_phi,
            pooled_start=guide_phi,
            scalar_well_starts=[],
            start_mode="warm_only",
            runtime=runtime,
            torch_data=torch_data,
            solver_context=solver_context,
            solver_state=solver_state_start,
            compute_summary=True,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=next_step,
            lambda_value=float(proposal.lambda_value),
            selection_score=selection_score,
            bic_refit_cache=bic_refit_cache,
            static_metadata=static_metadata,
        )
        fit.solver_state = _offload_solver_state_to_cpu(fit.solver_state)

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
                "initializer_K": int(guide.K),
                "initializer_requested_K": int(_partition_candidate_requested_k(guide)),
                "initializer_source": str(guide.source),
                "initializer_partition_signature": str(guide_signature),
                "initializer_matrix_hash": str(guide_matrix_hash),
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
                "partition_generation_elapsed_seconds": float(
                    initializer_pool.generation_elapsed_seconds
                ),
                "partition_curvature_elapsed_seconds": float(
                    initializer_pool.curvature_elapsed_seconds
                ),
                "partition_ward_elapsed_seconds": float(
                    initializer_pool.ward_elapsed_seconds
                ),
                "partition_refine_ward_elapsed_seconds": float(
                    initializer_pool.refine_ward_elapsed_seconds
                ),
                "partition_initial_generation_elapsed_seconds": float(
                    initializer_pool.initial_generation_elapsed_seconds
                ),
                "partition_refine_generation_elapsed_seconds": float(
                    initializer_pool.refine_generation_elapsed_seconds
                ),
                "partition_candidate_count": int(len(initializer_pool.candidates)),
                "partition_candidate_refinement_reason": str(
                    initializer_pool.refinement_reason
                ),
                "partition_candidate_sparse_k_grid": ",".join(
                    str(int(k)) for k in initializer_pool.sparse_k_grid
                ),
                "partition_candidate_refine_k_grid": ",".join(
                    str(int(k)) for k in initializer_pool.refine_k_grid
                ),
                "partition_candidate_k_grid": ",".join(
                    str(int(k)) for k in initializer_pool.combined_k_grid
                ),
            }
        )
        candidate_id = int(len(result_entries))
        row["_candidate_id"] = candidate_id
        result_entries.append((fit, evaluation, row, artifact))
        incumbent = fit_by_lambda.get(lambda_key)
        if _prefer_fit_candidate(fit, incumbent):
            fit_by_lambda[lambda_key] = fit

        exact_raw_eligible = bool(
            _exact_fusion_certificate_mask(pd.DataFrame([row]))[0]
            and bool(effective_tensor_graph.is_complete)
        )
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=float(proposal.lambda_value),
                n_clusters=int(row["n_clusters"]),
                partition_signature=str(row["partition_signature"]),
                partition_icl=float(row["partition_icl"]),
                kkt_residual=float(row["fixed_objective_kkt_residual"]),
                exact_candidate_eligible=bool(exact_raw_eligible),
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
        pd.DataFrame([row for _, _, row, _ in result_entries])
        .sort_values(["lambda", "selection_step"])
        .reset_index(drop=True)
    )
    stop_reason = str(controller.stop_reason or "online_lambda_no_terminal_reason")
    refinement_rounds = sum(
        1 for proposal in controller.proposal_history if "refine" in str(proposal.phase)
    )
    selection_result = _assemble_selection_result(
        search_df=search_df,
        data=data,
        normalized_score=normalized_score,
        result_entries=result_entries,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_method=selection_method,
        profile_name=profile_name,
        lambda_search_mode="partition_guided_admm",
        lambda_bracket=None,
        adaptive_search_stop_reason=stop_reason,
        adaptive_search_rounds_completed=int(len(controller.proposal_history)),
        adaptive_refinement_rounds_completed=int(refinement_rounds),
        selection_start_time=selection_start_time,
        strict_positive_exact_fusion=True,
    )
    return selection_result


def _grid_search_selection(
    *,
    data: TumorData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    profile_name: str,
    selection_method: str,
    selection_score: str,
    finalize_selected_fit: bool,
) -> BICSelectionResult:
    selection_start_time = perf_counter()
    explicit_lambda_grid = lambda_grid is not None
    normalized_lambda_grid_mode = str(lambda_grid_mode).strip().lower()
    if normalized_lambda_grid_mode not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {lambda_grid_mode}")
    if (
        is_partition_guided_lambda_grid_mode(normalized_lambda_grid_mode)
        and lambda_grid is not None
    ):
        raise ValueError(
            "partition_guided_admm does not accept a prespecified lambda grid; "
            "use lambda_grid_mode='adaptive_bic' for legacy grid search."
        )
    partition_guided_mode = bool(
        lambda_grid is None
        and is_partition_guided_lambda_grid_mode(normalized_lambda_grid_mode)
    )
    adaptive_lambda_mode = bool(
        lambda_grid is None
        and is_adaptive_lambda_grid_mode(normalized_lambda_grid_mode)
    )
    normalized_score = _normalize_selection_score_name(selection_score)
    lambda_search_mode = (
        "explicit_grid" if explicit_lambda_grid else normalized_lambda_grid_mode
    )
    if partition_guided_mode:
        return _partition_guided_admm_selection(
            data=data,
            simulation_root=simulation_root,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            use_warm_starts=use_warm_starts,
            evaluate_all_candidates=evaluate_all_candidates,
            profile_name=profile_name,
            selection_method=selection_method,
            selection_score=selection_score,
        )
    lambda_bracket: LambdaBracket | None = None
    if lambda_grid is None and not adaptive_lambda_mode:
        raise ValueError(
            f"lambda_grid_mode={lambda_search_mode!r} requires an explicit lambda grid."
        )
    lambda_grid = [] if lambda_grid is None else _sorted_unique_lambdas(lambda_grid)
    likelihood_partition_pool_enabled = bool(ENABLE_LIKELIHOOD_PARTITION_CANDIDATES)

    prepare_start_time = perf_counter()
    solver_context = prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy=str(fit_options.dense_fallback_policy),
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.tol),
        graph=fit_options.graph,
        inner_max_iter=max(int(fit_options.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
        device=fit_options.device,
        dtype=fit_options.dtype,
        objective_shape=str(fit_options.objective_shape),
    )
    prepare_elapsed_seconds = float(perf_counter() - prepare_start_time)
    runtime = solver_context.runtime
    torch_data = torch_data_from_context(solver_context)
    pilot_phi: StartArray = solver_context.exact_pilot
    pooled_start: StartArray = solver_context.pooled_start
    scalar_well_starts: list[StartArray] = list(solver_context.scalar_well_starts)
    effective_graph = solver_context.graph_spec
    effective_tensor_graph = solver_context.graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    static_metadata = _candidate_static_metadata(
        data, effective_graph, pilot_phi=pilot_phi
    )
    search_fit_options = (
        _adaptive_first_pass_options(effective_fit_options)
        if adaptive_lambda_mode
        else effective_fit_options
    )
    if adaptive_lambda_mode:
        lambda_bracket = _initial_adaptive_lambda_bracket(
            torch_data=torch_data,
            runtime=runtime,
            exact_pilot=pilot_phi,
            pooled_start=pooled_start,
            edge_u=effective_tensor_graph.edge_u,
            edge_v=effective_tensor_graph.edge_v,
            edge_w=effective_tensor_graph.weight,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            tol=float(fit_options.tol),
            degree_bound=int(effective_graph.degree_bound),
            sparse_anchors=bool(likelihood_partition_pool_enabled),
        )
        lambda_grid = list(lambda_bracket.anchors)
    simulation_truth: SimulationTruth | None = None
    if (
        evaluate_all_candidates
        and simulation_root is not None
        and (simulation_root / data.tumor_id).exists()
    ):
        simulation_truth = load_simulation_truth(data, simulation_root)
    result_entries: list[
        tuple[
            FitResult,
            SimulationEvaluation | None,
            dict[str, float | int | str | bool],
            SelectionArtifact,
        ]
    ] = []
    fit_by_lambda: dict[float, FitResult] = {}
    solver_state_by_lambda: dict[float, SolverState] = {}
    partition_labels_by_candidate_id: dict[int, np.ndarray] = {}
    bic_refit_cache: dict[str, PartitionRefitResult] = {}
    next_step = 0
    (
        adaptive_max_candidates,
        adaptive_max_rounds,
        adaptive_refine_per_round,
        adaptive_transition_probe_max_candidates,
    ) = _resolve_adaptive_path_config(
        adaptive_lambda_mode and likelihood_partition_pool_enabled
    )
    adaptive_initial_anchor_count = int(len(lambda_grid))

    def _nearest_phi_start(target_lambda: float) -> StartArray:
        if not fit_by_lambda:
            return _clone_start(pilot_phi)
        nearest_lambda = min(
            fit_by_lambda,
            key=lambda value: _lambda_warm_start_distance(
                source_lambda=float(value),
                target_lambda=float(target_lambda),
            ),
        )
        return _clone_start(_fit_phi_start(fit_by_lambda[nearest_lambda]))

    def _nearest_solver_state(target_lambda: float) -> SolverState | None:
        if not solver_state_by_lambda:
            return None
        nearest_lambda = min(
            solver_state_by_lambda,
            key=lambda value: _lambda_warm_start_distance(
                source_lambda=float(value),
                target_lambda=float(target_lambda),
            ),
        )
        return solver_state_by_lambda.get(nearest_lambda)

    def _evaluate_lambda_sequence(
        lambda_values_to_run: list[float],
        *,
        search_round: int,
        search_phase: str,
        allow_revisit: bool = False,
        candidate_fit_options: FitOptions | None = None,
        start_mode: str = "full",
        compute_summary: bool = True,
        phi_start_by_lambda: dict[float, StartArray] | None = None,
        solver_state_start_by_lambda: dict[float, SolverState] | None = None,
        scalar_well_starts_by_lambda: dict[float, list[StartArray]] | None = None,
        lambda_metadata_by_lambda: dict[float, dict[str, float | int | str | bool]]
        | None = None,
    ) -> None:
        nonlocal next_step
        ordered_lambdas = [
            value
            for value in _sorted_unique_lambdas(lambda_values_to_run)
            if allow_revisit or _canonical_lambda(value) not in fit_by_lambda
        ]
        if not ordered_lambdas:
            return

        previous_phi: StartArray | None = None
        previous_solver_state: SolverState | None = None
        for lambda_value in ordered_lambdas:
            lambda_key = _canonical_lambda(lambda_value)
            solver_state_start: SolverState | None = None
            if phi_start_by_lambda is not None and lambda_key in phi_start_by_lambda:
                phi_start = _clone_start(phi_start_by_lambda[lambda_key])
                if solver_state_start_by_lambda is not None:
                    solver_state_start = solver_state_start_by_lambda.get(lambda_key)
                if solver_state_start is None:
                    solver_state_start = solver_state_by_lambda.get(lambda_key)
            elif use_warm_starts:
                solver_state_start = (
                    previous_solver_state
                    if previous_solver_state is not None
                    else _nearest_solver_state(lambda_value)
                )
                if solver_state_start is not None:
                    phi_start = _clone_start(solver_state_start.phi)
                else:
                    phi_start = (
                        _clone_start(previous_phi)
                        if previous_phi is not None
                        else _nearest_phi_start(lambda_value)
                    )
            else:
                phi_start = _clone_start(pilot_phi)
            candidate_scalar_well_starts = scalar_well_starts
            if (
                scalar_well_starts_by_lambda is not None
                and lambda_key in scalar_well_starts_by_lambda
            ):
                candidate_scalar_well_starts = scalar_well_starts_by_lambda[lambda_key]
            fit, evaluation, row, artifact = _evaluate_candidate(
                data=data,
                fit_options=search_fit_options,
                candidate_fit_options=candidate_fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                simulation_root=simulation_root,
                simulation_truth=simulation_truth,
                evaluate_candidate=evaluate_all_candidates,
                phi_start=phi_start,
                exact_pilot=pilot_phi,
                pooled_start=pooled_start,
                scalar_well_starts=candidate_scalar_well_starts,
                start_mode=start_mode,
                runtime=runtime,
                torch_data=torch_data,
                solver_context=solver_context,
                solver_state=solver_state_start if use_warm_starts else None,
                compute_summary=compute_summary,
                selection_method=selection_method,
                profile_name=profile_name,
                selection_step=next_step,
                lambda_value=lambda_value,
                selection_score=selection_score,
                bic_refit_cache=bic_refit_cache,
                static_metadata=static_metadata,
            )
            row["search_round"] = int(search_round)
            row["search_phase"] = str(search_phase)
            row["lambda_source"] = str(search_phase)
            row["lambda_search_mode"] = str(lambda_search_mode)
            row["selection_prepare_elapsed_seconds"] = float(prepare_elapsed_seconds)
            row["adaptive_candidate_budget"] = int(adaptive_max_candidates)
            row["adaptive_max_rounds"] = int(adaptive_max_rounds)
            row["adaptive_refine_per_round"] = int(adaptive_refine_per_round)
            row["adaptive_transition_probe_max_candidates"] = int(
                adaptive_transition_probe_max_candidates
            )
            row["adaptive_initial_anchor_count"] = int(adaptive_initial_anchor_count)
            row["likelihood_partition_pool_enabled"] = bool(
                likelihood_partition_pool_enabled
            )
            row["lambda_bracket_min"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_min)
            )
            row["lambda_bracket_eq"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_eq)
            )
            row["lambda_bracket_full"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_full)
            )
            if lambda_bracket is None:
                row["lambda_full_residual"] = np.nan
            else:
                for (
                    diagnostic_name,
                    diagnostic_value,
                ) in lambda_bracket.diagnostics.items():
                    if np.isscalar(diagnostic_value):
                        row[str(diagnostic_name)] = float(diagnostic_value)
            if (
                lambda_metadata_by_lambda is not None
                and lambda_key in lambda_metadata_by_lambda
            ):
                row.update(lambda_metadata_by_lambda[lambda_key])
            candidate_id = int(len(result_entries))
            row["_candidate_id"] = candidate_id
            result_entries.append((fit, evaluation, row, artifact))
            if artifact.bic_partition_labels is not None:
                partition_labels_by_candidate_id[candidate_id] = np.asarray(
                    artifact.bic_partition_labels,
                    dtype=np.int64,
                ).copy()
            incumbent = fit_by_lambda.get(lambda_key)
            if _prefer_fit_candidate(fit, incumbent):
                fit_by_lambda[lambda_key] = fit
                if fit.solver_state is not None:
                    solver_state_by_lambda[lambda_key] = fit.solver_state
            next_step += 1
            if use_warm_starts:
                previous_phi = _clone_start(_fit_phi_start(fit))
                previous_solver_state = fit.solver_state

    _evaluate_lambda_sequence(
        lambda_grid,
        search_round=0,
        search_phase="base",
        start_mode="warm_only" if adaptive_lambda_mode and use_warm_starts else "full",
        compute_summary=True,
    )

    if adaptive_lambda_mode and lambda_bracket is not None:
        remaining_transition_budget = max(
            int(adaptive_max_candidates) - len(fit_by_lambda),
            0,
        )
        transition_probe_records = _adaptive_transition_probe_records(
            lambda_bracket,
            list(fit_by_lambda.keys()),
            max_new=min(
                int(adaptive_transition_probe_max_candidates),
                remaining_transition_budget,
            ),
        )
        if transition_probe_records:
            transition_phi_by_lambda: dict[float, StartArray] = {}
            transition_state_by_lambda: dict[float, SolverState] = {}
            transition_starts_by_lambda: dict[float, list[StartArray]] = {}
            transition_metadata_by_lambda: dict[
                float, dict[str, float | int | str | bool]
            ] = {}
            for proposal in transition_probe_records:
                lambda_key = _canonical_lambda(proposal.lambda_value)
                left_fit = fit_by_lambda.get(_canonical_lambda(proposal.left_lambda))
                right_fit = fit_by_lambda.get(_canonical_lambda(proposal.right_lambda))
                transition_starts = [_clone_start(pilot_phi)]
                if right_fit is not None:
                    transition_starts.insert(0, _clone_start(_fit_phi_start(right_fit)))
                if left_fit is not None:
                    transition_phi_by_lambda[lambda_key] = _clone_start(
                        _fit_phi_start(left_fit)
                    )
                    if left_fit.solver_state is not None:
                        transition_state_by_lambda[lambda_key] = left_fit.solver_state
                    transition_starts.insert(0, _clone_start(_fit_phi_start(left_fit)))
                elif right_fit is not None:
                    transition_phi_by_lambda[lambda_key] = _clone_start(
                        _fit_phi_start(right_fit)
                    )
                    if right_fit.solver_state is not None:
                        transition_state_by_lambda[lambda_key] = right_fit.solver_state
                transition_starts_by_lambda[lambda_key] = transition_starts
                transition_metadata_by_lambda[lambda_key] = {
                    "adaptive_interval_left_lambda": float(proposal.left_lambda),
                    "adaptive_interval_right_lambda": float(proposal.right_lambda),
                    "adaptive_interval_log10_width": float(proposal.log_width),
                    "adaptive_interval_priority_class": int(proposal.priority_key[0]),
                    "adaptive_interval_reason": str(proposal.reason),
                    "adaptive_interval_partition_changed": bool(
                        proposal.partition_changed
                    ),
                    "adaptive_interval_nonagglomerative_or_numerically_inconsistent": bool(
                        proposal.nonagglomerative_or_numerically_inconsistent
                    ),
                    "adaptive_transition_probe": True,
                }
            _evaluate_lambda_sequence(
                [float(proposal.lambda_value) for proposal in transition_probe_records],
                search_round=0,
                search_phase="adaptive_transition_probe",
                candidate_fit_options=_adaptive_first_pass_options(
                    effective_fit_options
                ),
                start_mode="full",
                compute_summary=True,
                phi_start_by_lambda=transition_phi_by_lambda,
                solver_state_start_by_lambda=transition_state_by_lambda,
                scalar_well_starts_by_lambda=transition_starts_by_lambda,
                lambda_metadata_by_lambda=transition_metadata_by_lambda,
            )

    adaptive_search_rounds_completed = 0
    adaptive_search_stop_reason = "not_applicable"
    adaptive_refinement_rounds_completed = 0
    if adaptive_lambda_mode:
        for adaptive_round in range(1, adaptive_max_rounds + 1):
            if len(fit_by_lambda) >= adaptive_max_candidates:
                adaptive_search_stop_reason = "adaptive_candidate_budget_reached"
                break
            interim_df = pd.DataFrame([row for _, _, row, _ in result_entries])
            proposal_records = _adaptive_interval_proposal_records(
                interim_df,
                normalized_score=normalized_score,
                tol=float(effective_fit_options.tol),
                max_new=min(
                    adaptive_refine_per_round,
                    max(adaptive_max_candidates - len(fit_by_lambda), 0),
                ),
                partition_labels_by_candidate_id=partition_labels_by_candidate_id,
            )
            proposal_records = [
                proposal
                for proposal in proposal_records
                if _canonical_lambda(proposal.lambda_value) not in fit_by_lambda
            ]
            proposals = [float(proposal.lambda_value) for proposal in proposal_records]
            if not proposals:
                adaptive_search_stop_reason = "adaptive_path_resolved"
                break
            fit_by_candidate_id = {
                int(row["_candidate_id"]): fit
                for fit, _, row, _ in result_entries
                if "_candidate_id" in row
            }
            proposal_phi_by_lambda: dict[float, StartArray] = {}
            proposal_state_by_lambda: dict[float, SolverState] = {}
            proposal_scalar_starts_by_lambda: dict[float, list[StartArray]] = {}
            proposal_metadata_by_lambda: dict[
                float, dict[str, float | int | str | bool]
            ] = {}
            for proposal in proposal_records:
                lambda_key = _canonical_lambda(proposal.lambda_value)
                left_fit = (
                    fit_by_candidate_id.get(int(proposal.left_candidate_id))
                    if proposal.left_candidate_id is not None
                    else None
                )
                right_fit = (
                    fit_by_candidate_id.get(int(proposal.right_candidate_id))
                    if proposal.right_candidate_id is not None
                    else None
                )
                if left_fit is not None:
                    proposal_phi_by_lambda[lambda_key] = _clone_start(
                        _fit_phi_start(left_fit)
                    )
                    if left_fit.solver_state is not None:
                        proposal_state_by_lambda[lambda_key] = left_fit.solver_state
                elif right_fit is not None:
                    proposal_phi_by_lambda[lambda_key] = _clone_start(
                        _fit_phi_start(right_fit)
                    )
                    if right_fit.solver_state is not None:
                        proposal_state_by_lambda[lambda_key] = right_fit.solver_state
                proposal_starts = [_clone_start(start) for start in scalar_well_starts]
                if right_fit is not None:
                    proposal_starts.insert(0, _clone_start(_fit_phi_start(right_fit)))
                proposal_scalar_starts_by_lambda[lambda_key] = proposal_starts
                proposal_metadata_by_lambda[lambda_key] = {
                    "adaptive_interval_left_lambda": float(proposal.left_lambda),
                    "adaptive_interval_right_lambda": float(proposal.right_lambda),
                    "adaptive_interval_log10_width": float(proposal.log_width),
                    "adaptive_interval_priority_class": int(proposal.priority_key[0]),
                    "adaptive_interval_reason": str(proposal.reason),
                    "adaptive_interval_partition_changed": bool(
                        proposal.partition_changed
                    ),
                    "adaptive_interval_nonagglomerative_or_numerically_inconsistent": bool(
                        proposal.nonagglomerative_or_numerically_inconsistent
                    ),
                }
            before = len(fit_by_lambda)
            _evaluate_lambda_sequence(
                proposals,
                search_round=adaptive_round,
                search_phase=f"adaptive_refine_{adaptive_round}",
                start_mode="full",
                compute_summary=True,
                phi_start_by_lambda=proposal_phi_by_lambda,
                solver_state_start_by_lambda=proposal_state_by_lambda,
                scalar_well_starts_by_lambda=proposal_scalar_starts_by_lambda,
                lambda_metadata_by_lambda=proposal_metadata_by_lambda,
            )
            if len(fit_by_lambda) == before:
                adaptive_search_stop_reason = "adaptive_no_new_lambdas"
                break
            adaptive_refinement_rounds_completed = adaptive_round
        else:
            adaptive_search_stop_reason = "adaptive_max_rounds_reached"
    adaptive_search_rounds_completed = int(adaptive_refinement_rounds_completed)
    search_df = (
        pd.DataFrame([row for _, _, row, _ in result_entries])
        .sort_values(["lambda", "selection_step"])
        .reset_index(drop=True)
    )
    partition_generation_elapsed_seconds = 0.0
    partition_curvature_elapsed_seconds = 0.0
    partition_ward_elapsed_seconds = 0.0
    partition_refine_ward_elapsed_seconds = 0.0
    partition_initial_generation_elapsed_seconds = 0.0
    partition_refine_generation_elapsed_seconds = 0.0
    partition_candidate_count = 0
    if likelihood_partition_pool_enabled:
        partition_generation_start_time = perf_counter()
        partition_k_grid = _likelihood_partition_k_grid(int(data.num_mutations))
        partition_curvature_start_time = perf_counter()
        partition_curvature = observed_curvature_at_pilot_torch(
            data,
            pilot_phi,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        partition_curvature_elapsed_seconds = float(
            perf_counter() - partition_curvature_start_time
        )
        partition_ward_start_time = perf_counter()
        partition_label_sets = hessian_weighted_ward_label_sets_torch(
            pilot_phi,
            partition_curvature,
            K_grid=partition_k_grid,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        partition_ward_elapsed_seconds = float(
            perf_counter() - partition_ward_start_time
        )
        partition_initial_start_time = perf_counter()
        partition_candidates = generate_likelihood_partition_starts(
            data,
            exact_pilot=pilot_phi,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            K_grid=partition_k_grid,
            max_candidates_per_K=int(LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K),
            cem_max_iter=int(LIKELIHOOD_PARTITION_CEM_MAX_ITER),
            refit_max_iter=int(LIKELIHOOD_PARTITION_REFIT_MAX_ITER),
            tol=float(effective_fit_options.tol),
            curvature=partition_curvature,
            label_sets=partition_label_sets,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
            use_torch=True,
            classification_weight_alpha=(
                float(PARTITION_ICL_DIRICHLET_ALPHA)
                if normalized_score == "partition_icl"
                else None
            ),
            allow_component_death=bool(normalized_score == "partition_icl"),
        )
        partition_initial_generation_elapsed_seconds = float(
            perf_counter() - partition_initial_start_time
        )
        partition_candidates = _rescore_partition_candidates(
            partition_candidates,
            data=data,
            normalized_score=normalized_score,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
        )
        partition_refine_k_grid, partition_refinement_reason = (
            _likelihood_partition_refinement_k_grid(
                partition_candidates,
                partition_k_grid,
                num_mutations=int(data.num_mutations),
            )
        )
        if partition_refine_k_grid:
            partition_refine_ward_start_time = perf_counter()
            partition_refine_label_sets = hessian_weighted_ward_label_sets_torch(
                pilot_phi,
                partition_curvature,
                K_grid=partition_refine_k_grid,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            partition_refine_ward_elapsed_seconds = float(
                perf_counter() - partition_refine_ward_start_time
            )
            partition_refine_start_time = perf_counter()
            partition_refine_candidates = generate_likelihood_partition_starts(
                data,
                exact_pilot=pilot_phi,
                major_prior=float(effective_fit_options.major_prior),
                eps=float(effective_fit_options.eps),
                K_grid=partition_refine_k_grid,
                max_candidates_per_K=int(LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K),
                cem_max_iter=int(LIKELIHOOD_PARTITION_CEM_MAX_ITER),
                refit_max_iter=int(LIKELIHOOD_PARTITION_REFIT_MAX_ITER),
                tol=float(effective_fit_options.tol),
                curvature=partition_curvature,
                label_sets=partition_refine_label_sets,
                torch_data=torch_data,
                device=runtime.device,
                dtype=runtime.dtype,
                use_torch=True,
                classification_weight_alpha=(
                    float(PARTITION_ICL_DIRICHLET_ALPHA)
                    if normalized_score == "partition_icl"
                    else None
                ),
                allow_component_death=bool(normalized_score == "partition_icl"),
            )
            partition_refine_generation_elapsed_seconds = float(
                perf_counter() - partition_refine_start_time
            )
            partition_refine_candidates = _rescore_partition_candidates(
                partition_refine_candidates,
                data=data,
                normalized_score=normalized_score,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
            )
            partition_candidates = _deduplicate_partition_candidates(
                partition_candidates + partition_refine_candidates
            )
        partition_combined_k_grid = sorted(
            set(partition_k_grid) | set(partition_refine_k_grid)
        )
        partition_refine_k_set = set(partition_refine_k_grid)
        partition_generation_elapsed_seconds = float(
            perf_counter() - partition_generation_start_time
        )
        partition_candidate_count = int(len(partition_candidates))
        for partition_rank, partition_candidate in enumerate(
            partition_candidates, start=1
        ):
            fit, evaluation, row, artifact = _evaluate_partition_candidate(
                data=data,
                fit_options=effective_fit_options,
                candidate=partition_candidate,
                candidate_rank=partition_rank,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                simulation_truth=simulation_truth,
                evaluate_candidate=evaluate_all_candidates,
                selection_method=selection_method,
                profile_name=profile_name,
                selection_step=next_step,
                selection_score=selection_score,
                static_metadata=static_metadata,
                runtime=runtime,
            )
            row["search_round"] = -1
            row["search_phase"] = "likelihood_partition"
            row["lambda_source"] = "likelihood_partition"
            row["lambda_search_mode"] = str(lambda_search_mode)
            row["selection_prepare_elapsed_seconds"] = float(prepare_elapsed_seconds)
            row["adaptive_candidate_budget"] = int(adaptive_max_candidates)
            row["adaptive_max_rounds"] = int(adaptive_max_rounds)
            row["adaptive_refine_per_round"] = int(adaptive_refine_per_round)
            row["adaptive_transition_probe_max_candidates"] = int(
                adaptive_transition_probe_max_candidates
            )
            row["adaptive_initial_anchor_count"] = int(adaptive_initial_anchor_count)
            row["likelihood_partition_pool_enabled"] = bool(
                likelihood_partition_pool_enabled
            )
            row["partition_generation_elapsed_seconds"] = float(
                partition_generation_elapsed_seconds
            )
            row["partition_curvature_elapsed_seconds"] = float(
                partition_curvature_elapsed_seconds
            )
            row["partition_ward_elapsed_seconds"] = float(
                partition_ward_elapsed_seconds
            )
            row["partition_refine_ward_elapsed_seconds"] = float(
                partition_refine_ward_elapsed_seconds
            )
            row["partition_initial_generation_elapsed_seconds"] = float(
                partition_initial_generation_elapsed_seconds
            )
            row["partition_refine_generation_elapsed_seconds"] = float(
                partition_refine_generation_elapsed_seconds
            )
            row["partition_candidate_count"] = int(partition_candidate_count)
            requested_k = _partition_candidate_requested_k(partition_candidate)
            row["partition_candidate_generation_pass"] = (
                "local_refine"
                if int(requested_k) in partition_refine_k_set
                else "sparse_anchor"
            )
            row["partition_candidate_refinement_reason"] = str(
                partition_refinement_reason
            )
            row["partition_candidate_k_grid_size"] = int(len(partition_combined_k_grid))
            row["partition_candidate_sparse_k_grid"] = ",".join(
                str(int(k)) for k in partition_k_grid
            )
            row["partition_candidate_refine_k_grid"] = ",".join(
                str(int(k)) for k in partition_refine_k_grid
            )
            row["partition_candidate_k_grid"] = ",".join(
                str(int(k)) for k in partition_combined_k_grid
            )
            row["lambda_bracket_min"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_min)
            )
            row["lambda_bracket_eq"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_eq)
            )
            row["lambda_bracket_full"] = (
                np.nan if lambda_bracket is None else float(lambda_bracket.lambda_full)
            )
            if lambda_bracket is None:
                row["lambda_full_residual"] = np.nan
            else:
                for (
                    diagnostic_name,
                    diagnostic_value,
                ) in lambda_bracket.diagnostics.items():
                    if np.isscalar(diagnostic_value):
                        row[str(diagnostic_name)] = float(diagnostic_value)
            candidate_id = int(len(result_entries))
            row["_candidate_id"] = candidate_id
            result_entries.append((fit, evaluation, row, artifact))
            if artifact.bic_partition_labels is not None:
                partition_labels_by_candidate_id[candidate_id] = np.asarray(
                    artifact.bic_partition_labels,
                    dtype=np.int64,
                ).copy()
            next_step += 1
        search_df = (
            pd.DataFrame([row for _, _, row, _ in result_entries])
            .sort_values(["lambda", "selection_step"])
            .reset_index(drop=True)
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
        lambda_search_mode=lambda_search_mode,
        lambda_bracket=lambda_bracket,
        adaptive_search_stop_reason=adaptive_search_stop_reason,
        adaptive_search_rounds_completed=adaptive_search_rounds_completed,
        adaptive_refinement_rounds_completed=adaptive_refinement_rounds_completed,
        selection_start_time=selection_start_time,
    )


def select_model(
    *,
    data: TumorData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    finalize_selected_fit: bool = True,
) -> BICSelectionResult:
    normalized_score = _normalize_selection_score_name(selection_score)

    effective_lambda_grid_mode = str(lambda_grid_mode)
    effective_bic_df_scale = float(bic_df_scale)
    effective_bic_cluster_penalty = float(bic_cluster_penalty)
    effective_lambda_grid_mode_normalized = (
        str(effective_lambda_grid_mode).strip().lower()
    )
    if effective_lambda_grid_mode_normalized not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {effective_lambda_grid_mode}")
    if (
        is_partition_guided_lambda_grid_mode(effective_lambda_grid_mode_normalized)
        and lambda_grid is not None
    ):
        raise ValueError(
            "partition_guided_admm does not accept a prespecified lambda grid; "
            "use lambda_grid_mode='adaptive_bic' for legacy grid search."
        )
    guided_default = bool(
        lambda_grid is None
        and is_partition_guided_lambda_grid_mode(effective_lambda_grid_mode_normalized)
    )
    if guided_default:
        if normalized_score != "partition_icl":
            raise ValueError(
                "partition_guided_admm currently requires "
                "selection_score='partition_icl'; use "
                "lambda_grid_mode='adaptive_bic' for another score."
            )
        profile_name = f"partition_guided_admm_{normalized_score}"
        selection_method = "online_partition_guided_admm"
    else:
        profile_name = (
            "adaptive_bic_default"
            if normalized_score == "bic"
            else f"adaptive_{normalized_score}_default"
        )
        selection_method = (
            "lambda_path_grid" if lambda_grid is not None else "adaptive_bic_path"
        )

    return _grid_search_selection(
        data=data,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=effective_lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=effective_bic_df_scale,
        bic_cluster_penalty=effective_bic_cluster_penalty,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        profile_name=profile_name,
        selection_method=selection_method,
        selection_score=selection_score,
        finalize_selected_fit=finalize_selected_fit,
    )


__all__ = [
    "BICSelectionResult",
    "ModelSelectionResult",
    "SelectionArtifact",
    "SimulationDiagnostics",
    "select_model",
]
