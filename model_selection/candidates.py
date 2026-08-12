from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ..core.bic import (
    anchor_prior_adjusted_bic_value,
    compute_bic_with_df,
    effective_bic_depth_count,
    fixed_partition_bic,
)
from ..core.model import (
    FitOptions,
    FitResult,
    effective_raw_clonal_equality_tolerance,
    fit_fixed_objective,
)
from ..core.fusion.refit import (
    PartitionRefitResult,
    partition_constrained_observed_refit,
)
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from .partitions import (
    _cluster_sizes_text,
    _partition_signature,
    evaluate_raw_clonal_block_evidence,
    extract_exact_raw_clonal_block,
    extract_certified_fusion_partition,
)
from .scoring import (
    _effective_bic_partition_tol,
    _normalize_selection_score_name,
    _profile_penalty_from_fit,
)
from .types import (
    CandidateStaticMetadata,
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    RawClonalBlockCertificate,
    RawClonalBlockEvidence,
    SelectionScore,
    StartArray,
)


def validate_candidate_identity(candidate: RawFusionCandidate) -> None:
    """Fail fast when raw partition, refit, and score identities diverge."""

    partition = candidate.partition
    refit = candidate.refit
    score = candidate.score
    if not np.array_equal(partition.labels, refit.labels):
        raise AssertionError("Fixed-partition refit changed raw fusion labels.")
    actual_signature = _partition_signature(
        partition.labels,
        partition.mutation_ids if partition.mutation_ids else None,
    )
    if partition.signature != actual_signature:
        raise AssertionError(
            "Raw fusion partition signature does not match its labels."
        )
    if int(partition.n_clusters) != int(np.unique(partition.labels).size):
        raise AssertionError("Raw fusion partition cluster count is inconsistent.")
    if refit.partition_signature != partition.signature:
        raise AssertionError("Refit partition signature does not match raw partition.")
    if score.partition_signature != partition.signature:
        raise AssertionError("Selection score does not match raw partition.")
    if score.name == "clonal_fixed_partition_bic":
        clonal_block = candidate.clonal_block
        if clonal_block is None or not clonal_block.certified:
            raise AssertionError("Clonal BIC requires a certified raw CCF-one block.")
        evidence = candidate.clonal_block_evidence
        if evidence is not None and evidence.block_signature != clonal_block.block_signature:
            raise AssertionError("Clonal evidence refers to a different raw block.")
        anchor_index = candidate.raw_fit.raw_clonal_anchor_mutation_index
        anchor_target = candidate.raw_fit.raw_clonal_anchor_target
        if anchor_index is None or anchor_target is None:
            raise AssertionError("Clonal BIC requires an explicit raw-fusion anchor.")
        anchor_index = int(anchor_index)
        if not 0 <= anchor_index < int(partition.labels.size):
            raise AssertionError("Raw-fusion anchor mutation index is invalid.")
        if str(candidate.raw_fit.raw_clonal_anchor_source) == "none":
            raise AssertionError("Raw-fusion anchor provenance is missing.")
        witness_hash = str(
            getattr(candidate.raw_fit, "witness_subproblem_hash", "")
        )
        objective_hash = str(getattr(candidate.raw_fit, "objective_spec_hash", ""))
        if witness_hash and witness_hash != objective_hash:
            raise AssertionError("Witness subproblem hash differs from raw objective.")
        raw_anchor_cluster = int(partition.labels[anchor_index])
        block_members = np.flatnonzero(partition.labels == raw_anchor_cluster)
        if not np.array_equal(
            np.sort(block_members), np.sort(clonal_block.member_indices)
        ):
            raise AssertionError(
                "Selection partition expanded or contracted the exact clonal block."
            )
        target_array = np.asarray(clonal_block.target, dtype=np.float64)
        raw_block_phi = np.asarray(candidate.raw_fit.phi, dtype=np.float64)[
            clonal_block.member_indices
        ]
        if not np.allclose(
            raw_block_phi,
            target_array[None, :],
            rtol=0.0,
            atol=float(clonal_block.equality_tolerance),
        ):
            raise AssertionError("A raw clonal-block member is not at CCF one.")
        if not np.allclose(
            np.mean(raw_block_phi, axis=0),
            target_array,
            rtol=0.0,
            atol=float(clonal_block.equality_tolerance),
        ):
            raise AssertionError("Raw clonal-block centroid is not at CCF one.")
        if refit.clonal_cluster != raw_anchor_cluster:
            raise AssertionError(
                "Fixed-partition refit did not preserve the raw anchor cluster."
            )
        if refit.fixed_anchor_target is None or not np.allclose(
            np.asarray(refit.cluster_centers)[raw_anchor_cluster],
            np.asarray(anchor_target, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("Fixed-partition refit changed the raw anchor target.")
        expected_anchor_signature = str(clonal_block.block_signature)
        if score.anchor_block_signature != expected_anchor_signature:
            raise AssertionError("Score anchor block differs from the raw anchor block.")
        if candidate.anchor_seed_index != anchor_index:
            raise AssertionError("Candidate anchor seed differs from its raw fit.")
        if candidate.anchor_cluster_label != raw_anchor_cluster:
            raise AssertionError("Candidate anchor cluster differs from its partition.")
        if candidate.anchor_block_signature != expected_anchor_signature:
            raise AssertionError("Candidate anchor-block signature is inconsistent.")
        if candidate.anchor_target is None or not np.array_equal(
            np.asarray(candidate.anchor_target), np.asarray(anchor_target)
        ):
            raise AssertionError("Candidate anchor target differs from its raw fit.")
        if refit.anchor_block_signature != expected_anchor_signature:
            raise AssertionError("Refit clonal-block signature is inconsistent.")
        raw_anchor_phi = np.asarray(candidate.raw_fit.phi, dtype=np.float64)[
            anchor_index
        ]
        if not np.allclose(
            raw_anchor_phi,
            np.asarray(anchor_target, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError("Raw-fusion anchor is not at its fixed target.")
    score_tolerance = 1e-10 * (1.0 + abs(float(score.value)))
    if not np.isclose(
        float(score.loglik),
        float(refit.loglik),
        rtol=0.0,
        atol=score_tolerance,
    ):
        raise AssertionError("Score likelihood differs from stored refit likelihood.")
    if int(score.degrees_of_freedom) != int(refit.nominal_df):
        raise AssertionError("Score and refit degrees of freedom differ.")
    expected_penalty = float(score.degrees_of_freedom) * np.log(
        max(int(score.n_eff), 1)
    )
    if not np.isclose(
        float(score.penalty), expected_penalty, rtol=0.0, atol=score_tolerance
    ):
        raise AssertionError("Stored BIC penalty is not reconstructible.")
    expected_score = -2.0 * float(refit.loglik) + expected_penalty
    if not np.isclose(
        float(score.value), expected_score, rtol=0.0, atol=score_tolerance
    ):
        raise AssertionError("Stored score is not reconstructible.")
    expected_phi = np.asarray(refit.cluster_centers)[np.asarray(refit.labels)]
    if not np.allclose(
        np.asarray(refit.phi), expected_phi, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("Refit phi does not match centers indexed by labels.")


def _ineligibility_reason(
    *,
    fit: FitResult,
    partition: FusionPartition,
    refit_finite: bool,
    refit_numerically_resolved: bool,
    score_finite: bool,
    raw_clonal_anchor_certified: bool,
    raw_anchor_search_resolved: bool,
) -> str:
    if float(fit.lambda_value) <= 0.0:
        return "nonpositive_lambda"
    if str(fit.estimator_role) != "raw_fused_lambda_path":
        return "not_raw_fused_lambda_path"
    if not bool(fit.objective_faithful):
        return "raw_objective_not_faithful"
    if not bool(fit.full_kkt_certified) or not bool(fit.selection_eligible):
        return "raw_objective_not_kkt_certified"
    if not raw_clonal_anchor_certified:
        return "raw_clonal_cluster_not_certified"
    if not raw_anchor_search_resolved:
        return "raw_clonal_witness_search_unresolved"
    if not partition.certified:
        return str(partition.certification_failure_reason)
    if not refit_finite:
        return "fixed_partition_refit_nonfinite"
    if not refit_numerically_resolved:
        return "fixed_partition_refit_numerically_unresolved"
    if not score_finite:
        return "fixed_partition_score_nonfinite"
    return "none"


@dataclass(frozen=True)
class _CachedPartitionRefit:
    result: PartitionRefitResult
    loglik_refinement_delta: float
    max_center_refinement_delta: float
    numerically_resolved: bool


def _likelihood_model_id(data: TumorData) -> str:
    path = getattr(data, "path_likelihood", None)
    return (
        "legacy_major_minor_mixture_v1"
        if path is None
        else str(path.model_id)
    )


def _anchor_block_signature(labels: np.ndarray, cluster: int | None) -> str:
    if cluster is None:
        return "none"
    mask = np.asarray(labels, dtype=np.int64).reshape(-1) == int(cluster)
    digest = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()[:24]
    return f"{int(np.count_nonzero(mask))}:{digest}"


def _selection_refit_cache_key(
    *,
    data: TumorData,
    partition: FusionPartition,
    selection_options: FitOptions,
    raw_anchor_cluster: int | None = None,
    raw_anchor_target: np.ndarray | None = None,
    raw_anchor_block_signature: str = "none",
) -> tuple[object, ...]:
    return (
        partition.signature,
        str(selection_options.selection_anchor),
        None if raw_anchor_cluster is None else int(raw_anchor_cluster),
        str(raw_anchor_block_signature),
        (
            None
            if raw_anchor_target is None
            else tuple(float(value) for value in np.asarray(raw_anchor_target).reshape(-1))
        ),
        float(selection_options.major_prior),
        float(selection_options.eps),
        float(selection_options.selection_refit_tol),
        int(selection_options.selection_refit_max_iter),
        _likelihood_model_id(data),
        "independent_grid_refinement_v1",
    )


def _fixed_partition_refit(
    *,
    data: TumorData,
    partition: FusionPartition,
    selection_options: FitOptions,
    cache: dict[object, _CachedPartitionRefit] | None,
    raw_anchor_cluster: int | None = None,
    raw_anchor_target: np.ndarray | None = None,
    raw_anchor_block_signature: str = "none",
) -> tuple[_CachedPartitionRefit, bool]:
    refit_spec_key = _selection_refit_cache_key(
        data=data,
        partition=partition,
        selection_options=selection_options,
        raw_anchor_cluster=raw_anchor_cluster,
        raw_anchor_target=raw_anchor_target,
        raw_anchor_block_signature=raw_anchor_block_signature,
    )
    if cache is not None and refit_spec_key in cache:
        return cache[refit_spec_key], True

    kwargs = dict(
        major_prior=float(selection_options.major_prior),
        eps=float(selection_options.eps),
        tol=float(selection_options.selection_refit_tol),
        max_iter=int(selection_options.selection_refit_max_iter),
        anchor_mode=str(selection_options.selection_anchor),
        anchor_cluster=raw_anchor_cluster,
        fixed_anchor_target=raw_anchor_target,
        fixed_anchor_block_signature=raw_anchor_block_signature,
        anchor_feasibility_tol=float(
            selection_options.raw_clonal_anchor_feasibility_tol
        ),
    )
    coarse = partition_constrained_observed_refit(
        data,
        partition.labels,
        grid_refinement_factor=1,
        **kwargs,
    )
    refined = partition_constrained_observed_refit(
        data,
        partition.labels,
        grid_refinement_factor=2,
        **kwargs,
    )
    loglik_delta = abs(float(refined.loglik) - float(coarse.loglik))
    center_delta = float(
        np.max(
            np.abs(
                np.asarray(refined.cluster_centers, dtype=np.float64)
                - np.asarray(coarse.cluster_centers, dtype=np.float64)
            )
        )
    )
    loglik_tolerance = max(
        float(selection_options.selection_refit_tol)
        * (1.0 + abs(float(refined.loglik))),
        1e-10,
    )
    center_tolerance = max(
        10.0 * float(selection_options.selection_refit_tol),
        1e-10,
    )
    numerically_resolved = bool(
        refined.finite_candidate_found
        and int(refined.refit_finite_coordinate_count)
        == int(refined.refit_coordinate_count)
        and np.isfinite(loglik_delta)
        and loglik_delta <= loglik_tolerance
        and np.isfinite(center_delta)
        and center_delta <= center_tolerance
        and float(refined.anchor_deviance_increase) >= -loglik_tolerance
    )
    cached = _CachedPartitionRefit(
        result=refined,
        loglik_refinement_delta=float(loglik_delta),
        max_center_refinement_delta=float(center_delta),
        numerically_resolved=numerically_resolved,
    )
    if cache is not None:
        cache[refit_spec_key] = cached
    return cached, False


def _build_refit_summary(
    refit: PartitionRefitResult,
    *,
    partition_signature: str,
    nominal_df: int,
    resolution: _CachedPartitionRefit,
    anchor_block_signature: str = "none",
) -> PartitionRefitSummary:
    return PartitionRefitSummary(
        labels=np.asarray(refit.labels, dtype=np.int64).copy(),
        partition_signature=str(partition_signature),
        phi=np.asarray(refit.phi, dtype=np.float64).copy(),
        cluster_centers=np.asarray(refit.cluster_centers, dtype=np.float64).copy(),
        loglik=float(refit.loglik),
        fit_loss=float(refit.fit_loss),
        nominal_df=int(nominal_df),
        active_df=int(refit.active_degrees_of_freedom),
        anchor_mode=str(refit.anchor_mode),
        clonal_cluster=(
            None if refit.clonal_cluster is None else int(refit.clonal_cluster)
        ),
        anchor_deviance_increase=float(refit.anchor_deviance_increase),
        second_best_anchor_deviance_increase=float(
            refit.second_best_anchor_deviance_increase
        ),
        finite_candidate_found=bool(refit.finite_candidate_found),
        global_optimum_certified=False,
        loglik_source=str(refit.loglik_source),
        refit_numerically_resolved=bool(resolution.numerically_resolved),
        refit_loglik_refinement_delta=float(
            resolution.loglik_refinement_delta
        ),
        refit_max_center_refinement_delta=float(
            resolution.max_center_refinement_delta
        ),
        refit_coordinate_count=int(refit.refit_coordinate_count),
        refit_finite_coordinate_count=int(refit.refit_finite_coordinate_count),
        refit_total_grid_points=int(refit.refit_total_grid_points),
        refit_max_grid_spacing=float(refit.refit_max_grid_spacing),
        refit_total_candidate_basins=int(refit.refit_total_candidate_basins),
        refit_total_refined_candidates=int(refit.refit_total_refined_candidates),
        refit_min_best_second_loss_gap=float(
            refit.refit_min_best_second_loss_gap
        ),
        fixed_anchor_target=(
            None
            if refit.fixed_anchor_target is None
            else np.asarray(refit.fixed_anchor_target, dtype=np.float64).copy()
        ),
        anchor_block_signature=str(refit.fixed_anchor_block_signature),
    )


def _selection_score_diagnostics(
    *,
    data: TumorData,
    refit: PartitionRefitSummary,
    score: SelectionScore,
) -> dict[str, float]:
    diagnostic_df = int(score.degrees_of_freedom)
    return {
        "classic_bic": compute_bic_with_df(
            refit.loglik,
            diagnostic_df,
            score.n_eff,
        ),
        "classic_bic_active_df": compute_bic_with_df(
            refit.loglik,
            refit.active_df,
            score.n_eff,
        ),
        "classic_bic_depth_n": compute_bic_with_df(
            refit.loglik,
            diagnostic_df,
            effective_bic_depth_count(data),
        ),
    }


def _evaluate_candidate(
    *,
    data: TumorData,
    fit_options: FitOptions,
    candidate_fit_options: FitOptions | None,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    phi_start: StartArray | None,
    exact_pilot: StartArray | None,
    pooled_start: StartArray | None,
    scalar_well_starts: list[StartArray] | None,
    start_mode: str,
    runtime,
    torch_data,
    solver_context,
    solver_state: SolverState | None,
    compute_summary: bool,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    lambda_value: float,
    selection_score: str,
    static_metadata: CandidateStaticMetadata,
    bic_refit_cache: dict[object, _CachedPartitionRefit] | None = None,
    precomputed_fit: FitResult | None = None,
    raw_anchor_search_resolved: bool = True,
) -> tuple[
    FitResult,
    dict[str, float | int | str | bool],
    RawFusionCandidate,
]:
    candidate_start_time = perf_counter()
    canonical_score_name = _normalize_selection_score_name(selection_score)
    raw_fit_options = (
        fit_options if candidate_fit_options is None else candidate_fit_options
    )
    selection_options = fit_options

    raw_fit_start_time = perf_counter()
    fit = precomputed_fit
    if fit is None:
        fit = fit_fixed_objective(
            data=data,
            options=replace(raw_fit_options, lambda_value=float(lambda_value)),
            phi_start=phi_start,
            exact_pilot=exact_pilot,
            pooled_start=pooled_start,
            scalar_well_starts=scalar_well_starts,
            start_mode=start_mode,
            runtime=runtime,
            torch_data=torch_data,
            solver_context=solver_context,
            solver_state=solver_state,
            compute_summary=compute_summary,
        )
    elif not np.isclose(
        float(fit.lambda_value), float(lambda_value), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Precomputed raw fit has the wrong lambda value.")
    raw_fit_elapsed_seconds = float(perf_counter() - raw_fit_start_time)
    graph = selection_options.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    anchor_required = (
        str(selection_options.selection_anchor).strip().lower()
        == "clonal_required"
    )
    raw_anchor_index = fit.raw_clonal_anchor_mutation_index
    raw_anchor_target = fit.raw_clonal_anchor_target
    clonal_block: RawClonalBlockCertificate | None = None
    clonal_block_evidence: RawClonalBlockEvidence | None = None
    if anchor_required and raw_anchor_index is not None and raw_anchor_target is not None:
        raw_clonal_equality_tolerance = effective_raw_clonal_equality_tolerance(
            selection_options
        )
        if raw_clonal_equality_tolerance > float(
            _effective_bic_partition_tol(selection_options)
        ):
            raise ValueError(
                "Raw clonal equality tolerance must not exceed the selection "
                "partition tolerance."
            )
        clonal_block = extract_exact_raw_clonal_block(
            fit,
            data=data,
            witness_index=int(raw_anchor_index),
            target=np.asarray(raw_anchor_target, dtype=np.float64),
            anchor_tolerance=raw_clonal_equality_tolerance,
        )
        clonal_block_evidence = evaluate_raw_clonal_block_evidence(
            clonal_block,
            data=data,
            minimum_cluster_size=int(selection_options.raw_clonal_cluster_min_size),
            minimum_observed_support_per_region=max(
                int(
                    selection_options.raw_clonal_cluster_min_observed_support_per_region
                ),
                int(
                    selection_options.raw_clonal_evidence_min_observed_support_per_region
                ),
            ),
        )

    partition_tolerance = _effective_bic_partition_tol(selection_options)
    partition = extract_certified_fusion_partition(
        fit,
        graph=graph,
        tolerance=partition_tolerance,
        clonal_block=clonal_block,
        mutation_ids=tuple(str(value) for value in data.mutation_ids),
    )

    raw_anchor_cluster: int | None = None
    raw_clonal_anchor_certified = not anchor_required
    if (
        anchor_required
        and clonal_block is not None
        and raw_anchor_index is not None
        and raw_anchor_target is not None
    ):
        raw_anchor_index = int(raw_anchor_index)
        if 0 <= raw_anchor_index < int(data.num_mutations):
            raw_anchor_cluster = int(partition.labels[raw_anchor_index])
            raw_clonal_anchor_certified = bool(
                clonal_block.certified
                and
                str(fit.raw_clonal_anchor_source) != "none"
                and np.array_equal(
                    np.flatnonzero(partition.labels == raw_anchor_cluster),
                    clonal_block.member_indices,
                )
            )

    anchor_block_signature = (
        "none" if clonal_block is None else str(clonal_block.block_signature)
    )

    refit_start_time = perf_counter()
    cached_refit, cache_hit = _fixed_partition_refit(
        data=data,
        partition=partition,
        selection_options=selection_options,
        cache=bic_refit_cache,
        raw_anchor_cluster=raw_anchor_cluster,
        raw_anchor_target=raw_anchor_target,
        raw_anchor_block_signature=anchor_block_signature,
    )
    refit_result = cached_refit.result
    refit_elapsed_seconds = (
        0.0 if cache_hit else float(perf_counter() - refit_start_time)
    )

    score = fixed_partition_bic(
        loglik=float(refit_result.loglik),
        num_clusters=int(partition.n_clusters),
        data=data,
        anchor_mode=str(selection_options.selection_anchor),
        partition_signature=partition.signature,
        anchor_block_signature=anchor_block_signature,
        labels=partition.labels,
        anchor_cluster=raw_anchor_cluster,
    )
    if str(score.name) != canonical_score_name:
        raise AssertionError(
            f"Requested score {canonical_score_name} produced {score.name}."
        )
    refit = _build_refit_summary(
        refit_result,
        partition_signature=partition.signature,
        nominal_df=int(score.degrees_of_freedom),
        resolution=cached_refit,
        anchor_block_signature=anchor_block_signature,
    )
    raw_objective_certified = bool(
        float(fit.lambda_value) > 0.0
        and str(fit.estimator_role) == "raw_fused_lambda_path"
        and bool(fit.objective_faithful)
        and bool(fit.full_kkt_certified)
        and bool(fit.selection_eligible)
    )
    reason = _ineligibility_reason(
        fit=fit,
        partition=partition,
        refit_finite=bool(refit.finite_candidate_found),
        refit_numerically_resolved=bool(refit.refit_numerically_resolved),
        score_finite=bool(np.isfinite(score.value)),
        raw_clonal_anchor_certified=bool(raw_clonal_anchor_certified),
        raw_anchor_search_resolved=bool(raw_anchor_search_resolved),
    )
    candidate = RawFusionCandidate(
        raw_fit=fit,
        partition=partition,
        refit=refit,
        score=score,
        raw_objective_certified=raw_objective_certified,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
        anchor_seed_index=(
            None if raw_anchor_index is None else int(raw_anchor_index)
        ),
        anchor_seed_mutation_id=(
            "none"
            if raw_anchor_index is None
            else str(data.mutation_ids[int(raw_anchor_index)])
        ),
        anchor_cluster_label=raw_anchor_cluster,
        anchor_block_signature=str(anchor_block_signature),
        anchor_target=(
            None
            if raw_anchor_target is None
            else np.asarray(raw_anchor_target, dtype=np.float64).copy()
        ),
        anchor_search_complete=bool(fit.raw_clonal_anchor_search_complete),
        clonal_block=clonal_block,
        clonal_block_evidence=clonal_block_evidence,
    )
    validate_candidate_identity(candidate)

    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)
    score_diagnostics = _selection_score_diagnostics(
        data=data,
        refit=refit,
        score=score,
    )
    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "lambda_applicable": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "estimator_role": str(fit.estimator_role),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "anchor_prior_adjusted_selection_score": float(
            anchor_prior_adjusted_bic_value(
                score,
                num_clusters=int(partition.n_clusters),
            )
        ),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "bic": float(score.value),
        "bic_value": float(score.value),
        "classic_bic": float(score_diagnostics["classic_bic"]),
        "clonal_fixed_partition_bic": (
            float(score.value)
            if score.name == "clonal_fixed_partition_bic"
            else float("nan")
        ),
        "fixed_partition_bic": (
            float(score.value) if score.name == "fixed_partition_bic" else float("nan")
        ),
        "bic_loglik": float(score.loglik),
        "bic_loglik_source": str(refit.loglik_source),
        "bic_df": int(score.degrees_of_freedom),
        "bic_active_df": int(refit.active_df),
        "bic_penalty": float(score.penalty),
        "bic_active_penalty": float(refit.active_df * np.log(max(score.n_eff, 1))),
        "bic_n_eff": int(score.n_eff),
        "classic_bic_depth_n": float(score_diagnostics["classic_bic_depth_n"]),
        "classic_bic_active_df": float(
            score_diagnostics["classic_bic_active_df"]
        ),
        "bic_refit_finite_candidate_found": bool(refit.finite_candidate_found),
        "bic_refit_cache_hit": bool(cache_hit),
        "refit_global_optimum_certified": bool(refit.global_optimum_certified),
        "refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "refit_loglik_refinement_delta": float(
            refit.refit_loglik_refinement_delta
        ),
        "refit_max_center_refinement_delta": float(
            refit.refit_max_center_refinement_delta
        ),
        "refit_coordinate_count": int(refit.refit_coordinate_count),
        "refit_finite_coordinate_count": int(
            refit.refit_finite_coordinate_count
        ),
        "refit_total_grid_points": int(refit.refit_total_grid_points),
        "refit_max_grid_spacing": float(refit.refit_max_grid_spacing),
        "refit_total_candidate_basins": int(
            refit.refit_total_candidate_basins
        ),
        "refit_total_refined_candidates": int(
            refit.refit_total_refined_candidates
        ),
        "refit_min_best_second_loss_gap": float(
            refit.refit_min_best_second_loss_gap
        ),
        "refit_loglik": float(refit.loglik),
        "refit_fit_loss": float(refit.fit_loss),
        "refit_active_df": int(refit.active_df),
        "refit_anchor_mode": str(refit.anchor_mode),
        "refit_clonal_cluster": (
            -1 if refit.clonal_cluster is None else int(refit.clonal_cluster)
        ),
        "anchor_deviance_increase": float(refit.anchor_deviance_increase),
        "second_best_anchor_deviance_increase": float(
            refit.second_best_anchor_deviance_increase
        ),
        "partition_signature": str(partition.signature),
        "anchor_block_signature": str(anchor_block_signature),
        "selection_model_signature": (
            f"{partition.signature}|anchor:{anchor_block_signature}"
        ),
        "solver_branch_signature": str(fit.witness_subproblem_hash),
        "biological_block_signature": str(anchor_block_signature),
        "partition_hash": str(partition.signature),
        "partition_source": str(partition.source),
        "partition_tol": float(partition.tolerance),
        "reporting_partition_tol": float(selection_options.reporting_partition_tol),
        "partition_certified": bool(partition.certified),
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
        "n_clusters": int(partition.n_clusters),
        "bic_n_clusters": int(partition.n_clusters),
        "cluster_sizes": _cluster_sizes_text(partition.labels),
        "eligible_for_selection": bool(candidate.eligible_for_selection),
        "bic_selection_eligible": bool(candidate.eligible_for_selection),
        "selection_eligible": bool(candidate.eligible_for_selection),
        "ineligibility_reason": str(candidate.ineligibility_reason),
        "raw_kkt_eligible": bool(fit.selection_eligible),
        "raw_objective_certified": bool(raw_objective_certified),
        "raw_clonal_anchor_required": bool(anchor_required),
        "raw_clonal_anchor_certified": bool(raw_clonal_anchor_certified),
        "raw_clonal_cluster_certified": bool(
            clonal_block is not None and clonal_block.certified
        ),
        "raw_clonal_model_fitted": bool(
            clonal_block is not None and clonal_block.mathematically_certified
        ),
        "raw_clonal_cluster_failure_reason": (
            "none" if clonal_block is None else str(clonal_block.failure_reason)
        ),
        "raw_clonal_cluster_equality_tol": float(
            effective_raw_clonal_equality_tolerance(selection_options)
        ),
        "raw_clonal_cluster_size": (
            0 if clonal_block is None else int(clonal_block.cluster_size)
        ),
        "raw_clonal_cluster_signature": str(anchor_block_signature),
        "raw_clonal_cluster_max_member_residual": (
            float("nan")
            if clonal_block is None
            else float(clonal_block.maximum_member_residual)
        ),
        "raw_clonal_cluster_centroid_residual": (
            float("nan")
            if clonal_block is None
            else float(clonal_block.centroid_residual)
        ),
        "raw_clonal_cluster_observed_support_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                str(int(value))
                for value in clonal_block_evidence.observed_support_per_region
            )
        ),
        "raw_clonal_cluster_total_depth_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block_evidence.total_depth_per_region
            )
        ),
        "raw_clonal_cluster_median_depth_per_region": (
            "none"
            if clonal_block_evidence is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block_evidence.median_depth_per_region
            )
        ),
        "raw_clonal_cluster_evidence_supported": bool(
            clonal_block_evidence is not None
            and clonal_block_evidence.evidence_gate_passed
        ),
        "clonal_block_biologically_supported": bool(
            clonal_block_evidence is not None
            and clonal_block_evidence.evidence_gate_passed
        ),
        "raw_clonal_cluster_evidence_failure_reason": (
            "none"
            if clonal_block_evidence is None
            else str(clonal_block_evidence.evidence_failure_reason)
        ),
        "raw_clonal_cluster_common_center": (
            "none"
            if clonal_block is None
            else ",".join(
                format(float(value), ".17g")
                for value in clonal_block.common_center
            )
        ),
        "raw_clonal_cluster_centroid": (
            "none"
            if clonal_block is None
            else ",".join(
                format(float(value), ".17g") for value in clonal_block.centroid
            )
        ),
        "raw_clonal_witness_mutation_index": (
            -1 if raw_anchor_index is None else int(raw_anchor_index)
        ),
        "raw_clonal_witness_mutation_id": (
            "none"
            if raw_anchor_index is None
            else str(data.mutation_ids[int(raw_anchor_index)])
        ),
        "raw_clonal_anchor_mutation_index": (
            -1 if raw_anchor_index is None else int(raw_anchor_index)
        ),
        "raw_clonal_anchor_mutation_id": (
            "none"
            if raw_anchor_index is None
            else str(data.mutation_ids[int(raw_anchor_index)])
        ),
        "raw_clonal_anchor_cluster": (
            -1 if raw_anchor_cluster is None else int(raw_anchor_cluster)
        ),
        "raw_clonal_anchor_target": (
            "none"
            if raw_anchor_target is None
            else ",".join(
                format(float(value), ".17g")
                for value in np.asarray(raw_anchor_target).reshape(-1)
            )
        ),
        "raw_clonal_anchor_source": str(fit.raw_clonal_anchor_source),
        "raw_anchor_seed_index": (
            -1 if raw_anchor_index is None else int(raw_anchor_index)
        ),
        "raw_anchor_seed_mutation_id": (
            "none"
            if raw_anchor_index is None
            else str(data.mutation_ids[int(raw_anchor_index)])
        ),
        "raw_anchor_target": (
            "none"
            if raw_anchor_target is None
            else ",".join(
                format(float(value), ".17g")
                for value in np.asarray(raw_anchor_target).reshape(-1)
            )
        ),
        "raw_anchor_constraint_residual": float(
            fit.raw_clonal_anchor_constraint_residual
        ),
        "raw_anchor_search_complete": bool(fit.raw_clonal_anchor_search_complete),
        "raw_anchor_candidates_evaluated": int(
            fit.raw_clonal_anchor_candidates_evaluated
        ),
        "raw_anchor_objective_rank": int(fit.raw_clonal_anchor_objective_rank),
        "raw_anchor_objective_gap_to_second": float(
            fit.raw_clonal_anchor_objective_gap_to_second
        ),
        "raw_clonal_anchor_mode": str(fit.raw_clonal_anchor_mode),
        "raw_clonal_anchor_constraint_residual": float(
            fit.raw_clonal_anchor_constraint_residual
        ),
        "raw_clonal_anchor_frozen_coordinate_count": int(
            fit.raw_clonal_anchor_frozen_coordinate_count
        ),
        "raw_clonal_anchor_search_complete": bool(
            fit.raw_clonal_anchor_search_complete
        ),
        "raw_clonal_anchor_search_resolved": bool(raw_anchor_search_resolved),
        "raw_clonal_witness_coverage_certified": bool(
            fit.raw_clonal_witness_coverage_certified
        ),
        "raw_clonal_branch_stationarity_certified": bool(
            fit.raw_clonal_branch_stationarity_certified
        ),
        "raw_clonal_union_global_optimum_certified": bool(
            fit.raw_clonal_union_global_optimum_certified
        ),
        "raw_clonal_anchor_total_eligible_candidates": int(
            fit.raw_clonal_anchor_total_eligible_candidates
        ),
        "raw_clonal_anchor_candidates_evaluated": int(
            fit.raw_clonal_anchor_candidates_evaluated
        ),
        "raw_clonal_anchor_objective_rank": int(
            fit.raw_clonal_anchor_objective_rank
        ),
        "raw_clonal_anchor_objective_gap_to_second": float(
            fit.raw_clonal_anchor_objective_gap_to_second
        ),
        "raw_clonal_anchor_screening_rule": str(
            fit.raw_clonal_anchor_screening_rule
        ),
        "converged": bool(fit.converged),
        "raw_fit_status": str(fit.failure_reason),
        "loglik": float(fit.loglik),
        "raw_loglik": float(fit.loglik),
        "fit_loss": float(-fit.loglik),
        "penalized_objective": float(fit.penalized_objective),
        "raw_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "raw_penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "fixed_objective_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "raw_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "stationarity_certified": bool(fit.stationarity_certified),
        "global_optimality_certified": bool(fit.global_optimality_certified),
        "global_optimality_basis": str(fit.global_optimality_basis),
        "number_of_starts": int(fit.number_of_starts),
        "number_of_finite_starts": int(fit.number_of_finite_starts),
        "best_start_objective": float(fit.best_start_objective),
        "second_best_start_objective": float(fit.second_best_start_objective),
        "objective_spread_across_starts": float(fit.objective_spread_across_starts),
        "selected_start_objective_rank": int(fit.selected_start_objective_rank),
        "iterations": int(fit.iterations),
        "inner_iterations": int(fit.inner_iterations),
        "admm_iterations": int(fit.admm_iterations),
        "inner_solver": str(fit.inner_solver),
        "inner_backend": str(fit.inner_backend),
        "backend_iterations": int(fit.backend_iterations),
        "quotient_iterations": int(fit.quotient_iterations),
        "workset_iterations": int(fit.workset_iterations),
        "workset_expansions": int(fit.workset_expansions),
        "streamed_edge_passes": int(fit.streamed_edge_passes),
        "dense_iterations": int(fit.dense_iterations),
        "certificate_iterations": int(fit.certificate_iterations),
        "full_certificate_audit_passes": int(fit.full_certificate_audit_passes),
        "fallback_reason": str(fit.fallback_reason),
        "exactness_provenance_version": int(fit.exactness_provenance_version),
        "objective_faithful": bool(fit.objective_faithful),
        "objective_spec_hash": str(fit.objective_spec_hash),
        "base_fusion_objective_hash": str(fit.base_fusion_objective_hash),
        "raw_clonal_union_model_hash": str(fit.raw_clonal_union_model_hash),
        "witness_subproblem_hash": str(fit.witness_subproblem_hash),
        "original_graph_hash": str(fit.original_graph_hash),
        "certificate_problem_hash": str(fit.certificate_problem_hash),
        "certificate_scope": str(fit.certificate_scope),
        "certificate_gradient_scope": str(fit.certificate_gradient_scope),
        "full_kkt_certified": bool(fit.full_kkt_certified),
        "full_kkt_certificate_status": str(fit.full_kkt_certificate_status),
        "full_kkt_tolerance": float(fit.full_kkt_tolerance),
        "raw_solver_primal_tol": float(raw_fit_options.tol),
        "outer_kkt_certificate_status": str(fit.outer_kkt_certificate_status),
        "outer_num_frozen_coordinates": int(fit.outer_num_frozen_coordinates),
        "mm_consistency_violations": int(fit.mm_consistency_violations),
        "failure_reason": str(fit.failure_reason),
        "candidate_elapsed_seconds": float(perf_counter() - candidate_start_time),
        "raw_fit_elapsed_seconds": float(raw_fit_elapsed_seconds),
        "bic_refit_elapsed_seconds": float(refit_elapsed_seconds),
        "primary_phi_source": "raw_pairwise_fusion",
        "refit_phi_source": "fixed_partition_refit",
        "device": str(fit.device),
        "dtype": str(fit.dtype),
        "tol": float(raw_fit_options.tol),
        "outer_max_iter": int(raw_fit_options.outer_max_iter),
        "inner_max_iter": int(raw_fit_options.inner_max_iter),
        "eps": float(raw_fit_options.eps),
        "major_prior": float(raw_fit_options.major_prior),
        "selection_refit_tol": float(selection_options.selection_refit_tol),
        "selection_refit_max_iter": int(
            selection_options.selection_refit_max_iter
        ),
        "graph_name": str(fit.graph_name),
        "num_edges": int(static_metadata.edge_count),
        "edge_weight_min": float(static_metadata.edge_weight_min),
        "edge_weight_max": float(static_metadata.edge_weight_max),
        "edge_weight_mean": float(static_metadata.edge_weight_mean),
        "edge_list_hash": str(static_metadata.edge_list_hash),
        "pilot_matrix_hash": str(static_metadata.pilot_matrix_hash),
        "input_data_hash": str(static_metadata.input_data_hash),
        "fit_compute_summary": bool(compute_summary),
        "fit_start_mode": str(start_mode),
        "solver_state_warm_start": bool(solver_state is not None),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
    }
    return fit, row, candidate


__all__ = [
    "_evaluate_candidate",
    "_fixed_partition_refit",
    "_selection_refit_cache_key",
    "_selection_score_diagnostics",
    "validate_candidate_identity",
]
