from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from ..core.bic import fixed_partition_bic
from ..config import FitConfig
from ..core.objective import (
    ObservedModel,
    compile_observed_model,
    observed_box_fingerprint,
)
from ..core.fusion.types import RawFit, WorkCounters
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.scalar import (
    PartitionFit,
    canonical_partition_labels as _canonical_partition_labels,
    partition_constrained_observed_refit,
)
from ..io.data import TumorData
from .partitions import (
    extract_certified_fusion_partition,
    partition_signature,
)
from .scoring import (
    effective_bic_partition_tol,
    raw_fit_has_exact_fusion_certificate,
)
from .types import (
    DirectPartition,
    DirectPartitionCandidate,
    FusionPartition,
    PartitionRefitKey,
    RawFusionArtifact,
    RawFusionCandidate,
    SearchArtifact,
    SelectionScore,
    UnscoredRawFusionCandidate,
)


def _raw_fit_ineligibility_reason(raw_fit: RawFit) -> str:
    if float(raw_fit.provenance.lambda_value) <= 0.0:
        return "nonpositive_lambda"
    if not raw_fit_has_exact_fusion_certificate(raw_fit):
        return "raw_objective_not_kkt_certified"
    return "none"


def validate_candidate_identity(candidate: SearchArtifact) -> None:
    """Fail fast when partition, refit, score, or estimator identity diverges."""

    partition = candidate.partition
    actual_signature = partition_signature(
        partition.labels,
        partition.mutation_ids if partition.mutation_ids else None,
    )
    if partition.signature != actual_signature:
        raise AssertionError("Partition signature does not match its labels.")
    if int(partition.n_clusters) != int(np.unique(partition.labels).size):
        raise AssertionError("Partition cluster count is inconsistent.")
    if isinstance(candidate, UnscoredRawFusionCandidate):
        expected_reason = _raw_fit_ineligibility_reason(candidate.raw_fit)
        if expected_reason == "none":
            raise AssertionError("An exact-certified raw fit cannot remain unscored.")
        if candidate.ineligibility_reason != expected_reason:
            raise AssertionError(
                "Unscored raw-fusion reason does not match its raw certificate."
            )
        return

    refit = candidate.refit
    score = candidate.score
    if isinstance(candidate, RawFusionCandidate):
        objective = candidate.raw_fit.provenance.objective_key.base
        if refit.observed_likelihood_hash != str(objective.likelihood_hash):
            raise AssertionError(
                "Fixed-partition refit and raw fit use different likelihoods."
            )
        if refit.observed_box_hash != str(objective.box_hash):
            raise AssertionError(
                "Fixed-partition refit and raw fit use different feasible boxes."
            )
        if refit.likelihood_eps_hex != str(objective.eps_hex):
            raise AssertionError(
                "Fixed-partition refit and raw fit use different likelihood eps."
            )
    if not np.array_equal(partition.labels, refit.labels):
        raise AssertionError("Fixed-partition refit changed selected labels.")
    if refit.partition_signature != partition.signature:
        raise AssertionError("Refit partition signature does not match partition.")
    score.validate_against(refit)
    expected_phi = np.asarray(refit.cluster_centers)[np.asarray(refit.labels)]
    if not np.allclose(np.asarray(refit.phi), expected_phi, rtol=0.0, atol=1e-12):
        raise AssertionError("Refit phi does not match centers indexed by labels.")
    if isinstance(candidate, RawFusionCandidate):
        if candidate.eligible_for_selection and (
            not candidate.raw_objective_certified or not candidate.partition.certified
        ):
            raise AssertionError("Selectable raw candidates require both certificates.")
    elif isinstance(candidate, DirectPartitionCandidate):
        if candidate.partition.mutation_ids != tuple(
            str(value) for value in partition.mutation_ids
        ):
            raise AssertionError("Direct partition mutation identity changed.")
    else:  # pragma: no cover - union exhaustiveness guard
        raise TypeError(f"Unsupported candidate type: {type(candidate)!r}")


def _candidate_ineligibility_reason(
    *,
    partition: FusionPartition | DirectPartition,
    refit: PartitionFit,
    score: SelectionScore,
    raw_fit: RawFit | None = None,
    require_global_refit: bool = True,
) -> str:
    if isinstance(partition, FusionPartition):
        if raw_fit is None:
            raise ValueError("Raw-fusion eligibility requires its raw fit.")
        if float(raw_fit.provenance.lambda_value) <= 0.0:
            return "nonpositive_lambda"
        if not raw_fit_has_exact_fusion_certificate(raw_fit):
            return "raw_objective_not_kkt_certified"
        if not partition.certified:
            return str(partition.certification_failure_reason)
    else:
        if raw_fit is not None:
            raise ValueError("Direct-partition eligibility cannot inherit a raw fit.")
        if partition.n_clusters < 1:
            return "empty_partition"
    if not refit.finite_candidate_found:
        return "fixed_partition_refit_nonfinite"
    resolved = (
        refit.refit_numerically_resolved
        if isinstance(partition, FusionPartition)
        else refit.global_optimum_certified
    )
    if require_global_refit and not resolved:
        return "fixed_partition_refit_numerically_unresolved"
    if not np.isfinite(score.value):
        return "fixed_partition_score_nonfinite"
    return "none"


@dataclass(frozen=True)
class PartitionFitLookup:
    fit: PartitionFit
    miss_work: WorkCounters = WorkCounters()


@dataclass(frozen=True, slots=True)
class ScoredPartition:
    """One fixed-label fit paired with its deterministic BIC."""

    fit: PartitionFit
    score: SelectionScore


def _selection_refit_cache_key(
    *,
    model: ObservedModel,
    partition_signature: str,
    selection_options: FitConfig,
) -> PartitionRefitKey:
    refit = selection_options.selection.refit
    return PartitionRefitKey(
        partition_signature=str(partition_signature),
        observed_model_hash=str(model.fingerprint),
        observed_likelihood_hash=str(model.likelihood_fingerprint),
        reporting_model_hash=str(model.reporting_fingerprint),
        observed_box_hash=observed_box_fingerprint(model),
        likelihood_eps_hex=float(selection_options.eps).hex(),
        refit_tolerance_hex=float(refit.tolerance).hex(),
        refit_max_iter=int(refit.max_iter),
        refit_mode=str(refit.mode),
        refit_grid_points=int(refit.grid_points),
        refit_local_steps=int(refit.local_steps),
    )


def _fixed_labels_refit(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    selection_options: FitConfig,
    cache: dict[PartitionRefitKey, PartitionFit] | None,
    source_model: ObservedModel | None = None,
) -> PartitionFitLookup:
    strict_profile = selection_options.profile_name == "strict"
    refit_config = selection_options.selection.refit
    model = (
        compile_observed_model(
            data,
            major_prior=float(selection_options.major_prior),
            eps=float(selection_options.eps),
        )
        if source_model is None
        else source_model
    )
    refit_spec_key = _selection_refit_cache_key(
        model=model,
        partition_signature=partition_signature,
        selection_options=selection_options,
    )
    if cache is not None and refit_spec_key in cache:
        cached = cache[refit_spec_key]
        try:
            cached.validate_observed_model(
                model,
                eps=float(selection_options.eps),
            )
        except ValueError as exc:
            raise AssertionError(
                "Cached fixed-partition refit has the wrong model identity."
            ) from exc
        return PartitionFitLookup(fit=cached)

    kwargs = dict(
        major_prior=float(selection_options.major_prior),
        eps=float(selection_options.eps),
        tol=float(refit_config.tolerance),
        max_iter=int(refit_config.max_iter),
        scalar_mode=str(refit_config.mode),
        scalar_grid_points=int(refit_config.grid_points),
        scalar_local_steps=int(refit_config.local_steps),
    )
    refined = partition_constrained_observed_refit(
        data,
        np.asarray(labels, dtype=np.int64),
        _model=model,
        **kwargs,
    )
    loglik_delta = float(refined.global_optimality_gap)
    loglik_tolerance = max(
        float(refit_config.tolerance) * (1.0 + abs(float(refined.loglik))),
        1e-10,
    )
    numerically_resolved = bool(
        refined.finite_candidate_found
        and int(refined.refit_finite_coordinate_count)
        == int(refined.refit_coordinate_count)
        and (
            (
                refined.global_optimum_certified
                and np.isfinite(loglik_delta)
                and loglik_delta <= loglik_tolerance
            )
            if strict_profile
            else np.isfinite(float(refined.loglik))
        )
    )
    fit = replace(
        refined,
        partition_signature=str(partition_signature),
        refit_numerically_resolved=numerically_resolved,
    )
    try:
        fit.validate_observed_model(model, eps=float(selection_options.eps))
    except ValueError as exc:
        raise AssertionError(
            "Fixed-partition refit has the wrong model identity."
        ) from exc
    if cache is not None:
        cache[refit_spec_key] = fit
    return PartitionFitLookup(
        fit=fit,
        miss_work=WorkCounters(
            partition_refit_coordinates=int(fit.refit_coordinate_count),
            partition_refit_objective_evaluations=int(fit.refit_objective_evaluations),
        ),
    )


def _score_fixed_labels(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    refit_result: PartitionFit,
    selection_options: FitConfig,
) -> SelectionScore:
    strict_profile = selection_options.profile_name == "strict"
    return fixed_partition_bic(
        loglik=float(refit_result.loglik),
        num_clusters=int(np.unique(labels).size),
        data=data,
        partition_signature=str(partition_signature),
        labels=np.asarray(labels, dtype=np.int64),
        loglik_uncertainty=(
            float(refit_result.global_optimality_gap) if strict_profile else 0.0
        ),
    )


def evaluate_partition(
    *,
    data: TumorData,
    partition: FusionPartition | DirectPartition,
    selection_options: FitConfig,
    refit_cache: dict[PartitionRefitKey, PartitionFit] | None,
    source_model: ObservedModel | None = None,
) -> tuple[ScoredPartition, WorkCounters]:
    """Evaluate raw and direct label sets through one refit/score path."""

    cached_refit = _fixed_labels_refit(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        selection_options=selection_options,
        cache=refit_cache,
        source_model=source_model,
    )
    refit_result = cached_refit.fit
    score = _score_fixed_labels(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        refit_result=refit_result,
        selection_options=selection_options,
    )
    return ScoredPartition(fit=refit_result, score=score), cached_refit.miss_work


def candidate_from_raw_fit(
    *,
    data: TumorData,
    raw_fit: RawFit,
    selection_options: FitConfig,
    refit_cache: dict[PartitionRefitKey, PartitionFit] | None = None,
    source_model: ObservedModel | None = None,
) -> RawFusionArtifact:
    """Extract, refit, and score one already-computed raw-fusion result."""

    strict_profile = selection_options.profile_name == "strict"
    graph = selection_options.graph.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    partition_tolerance = effective_bic_partition_tol(selection_options)
    partition = extract_certified_fusion_partition(
        raw_fit,
        graph=graph,
        tolerance=partition_tolerance,
        mutation_ids=tuple(str(value) for value in data.mutation_ids),
    )

    raw_reason = _raw_fit_ineligibility_reason(raw_fit)
    if raw_reason != "none":
        candidate = UnscoredRawFusionCandidate(
            raw_fit=raw_fit,
            partition=partition,
            ineligibility_reason=raw_reason,
        )
        validate_candidate_identity(candidate)
        return candidate

    evaluation, refit_work = evaluate_partition(
        data=data,
        partition=partition,
        selection_options=selection_options,
        refit_cache=refit_cache,
        source_model=source_model,
    )
    refit = evaluation.fit
    score = evaluation.score
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=raw_fit,
        require_global_refit=strict_profile,
    )
    candidate = RawFusionCandidate(
        raw_fit=raw_fit,
        partition=partition,
        refit=refit,
        score=score,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
        work=refit_work,
    )
    validate_candidate_identity(candidate)

    return candidate


def evaluate_direct_partition_candidate(
    *,
    data: TumorData,
    proposal: PartitionCandidate,
    selection_options: FitConfig,
    source: str,
    parent_raw_candidate_id: int | None,
    parent_raw_lambda: float | None,
    refit_cache: dict[PartitionRefitKey, PartitionFit] | None,
    parent_raw_phi_hash: str = "",
    source_model: ObservedModel | None = None,
) -> DirectPartitionCandidate:
    """Evaluate one deterministic non-fusion partition under the common score."""

    labels = _canonical_partition_labels(
        np.asarray(proposal.labels, dtype=np.int64).reshape(-1)
    )
    mutation_ids = tuple(str(value) for value in data.mutation_ids)
    signature = partition_signature(labels, mutation_ids)
    partition = DirectPartition(
        labels=labels,
        signature=signature,
        source=source,
        mutation_ids=mutation_ids,
        parent_raw_candidate_id=parent_raw_candidate_id,
        parent_raw_lambda=parent_raw_lambda,
        parent_raw_phi_hash=str(parent_raw_phi_hash),
    )
    evaluation, refit_work = evaluate_partition(
        data=data,
        partition=partition,
        selection_options=selection_options,
        refit_cache=refit_cache,
        source_model=source_model,
    )
    refit = evaluation.fit
    score = evaluation.score
    strict_profile = selection_options.profile_name == "strict"
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=None,
        require_global_refit=strict_profile,
    )
    candidate = DirectPartitionCandidate(
        partition=partition,
        refit=refit,
        score=score,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
        work=refit_work,
    )
    validate_candidate_identity(candidate)
    return candidate


__all__ = [
    "PartitionFitLookup",
    "ScoredPartition",
    "candidate_from_raw_fit",
    "evaluate_direct_partition_candidate",
    "evaluate_partition",
    "validate_candidate_identity",
]
