from __future__ import annotations

import numpy as np

from ..core.model import FitOptions, FitResult
from .config import SELECTION_SCORE_NAMES
from .types import (
    CandidateRecord,
    CandidateSelectionDecision,
    RawFusionCandidate,
)


_EXACT_OBSERVED_OBJECTIVE_GRADIENT_SCOPES = frozenset(
    {
        "observed_objective",
        "clarke_piecewise_observed_objective_subgradient",
    }
)
_EXACT_CERTIFICATE_SCHEMA_VERSION = 2
_EXACT_CERTIFICATE_RESIDUAL_METHOD = "componentwise_box_cone_backward_error_v1"
_EXACT_CERTIFICATE_STATUSES = frozenset(
    {
        "certified",
        "input_dual_retained",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
        "zero_penalty_no_dual_needed",
    }
)


def _number_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalize_selection_score_name(selection_score: str) -> str:
    normalized = str(selection_score).strip().lower().replace("-", "_")
    if normalized in SELECTION_SCORE_NAMES:
        return normalized
    allowed = ", ".join(SELECTION_SCORE_NAMES)
    raise ValueError(
        f"Unknown selection_score: {selection_score}. Expected one of: {allowed}."
    )


def raw_candidate_has_exact_fusion_certificate(
    candidate: RawFusionCandidate,
) -> bool:
    """Apply the complete schema-2 raw admission contract.

    Keep every raw-candidate consumer on this typed predicate so selection,
    raw-reference provenance, final-Phi parents, and controller certification
    cannot drift apart.
    """

    fit = candidate.raw_fit
    provenance = getattr(fit, "exactness_provenance", None)
    residual_method = str(getattr(provenance, "residual_method", ""))
    residual = _number_or_nan(getattr(fit, "fixed_objective_kkt_residual", np.nan))
    tolerance = _number_or_nan(getattr(fit, "full_kkt_tolerance", np.nan))
    schema_version = _number_or_nan(
        getattr(fit, "exactness_provenance_version", np.nan)
    )
    return bool(
        candidate.eligible_for_selection
        and candidate.raw_objective_certified
        and float(getattr(fit, "lambda_value", 0.0)) > 0.0
        and candidate.partition.certified
        and schema_version == _EXACT_CERTIFICATE_SCHEMA_VERSION
        and residual_method == _EXACT_CERTIFICATE_RESIDUAL_METHOD
        and str(getattr(fit, "certificate_audit_dtype", "")) == "float64"
        and bool(getattr(fit, "selection_eligible", False))
        and str(getattr(fit, "estimator_role", "")) == "raw_fused_lambda_path"
        and bool(getattr(fit, "objective_faithful", False))
        and bool(str(getattr(fit, "objective_spec_hash", "")).strip())
        and bool(str(getattr(fit, "original_graph_hash", "")).strip())
        and bool(str(getattr(fit, "certificate_problem_hash", "")).strip())
        and str(getattr(fit, "certificate_scope", "")) == "full_original_graph"
        and str(getattr(fit, "certificate_gradient_scope", ""))
        in _EXACT_OBSERVED_OBJECTIVE_GRADIENT_SCOPES
        and bool(getattr(fit, "full_kkt_certified", False))
        and str(getattr(fit, "full_kkt_certificate_status", ""))
        in _EXACT_CERTIFICATE_STATUSES
        and np.isfinite(residual)
        and np.isfinite(tolerance)
        and tolerance > 0.0
        and residual <= tolerance
    )


def candidate_is_selection_eligible(
    record: CandidateRecord,
    *,
    strict_positive_exact_fusion: bool,
) -> bool:
    """Return typed candidate admission without consulting reporting fields."""

    candidate = record.candidate
    if isinstance(candidate, RawFusionCandidate):
        return raw_candidate_has_exact_fusion_certificate(candidate)
    return bool(candidate.eligible_for_selection and not strict_positive_exact_fusion)


def _assert_same_signature_consistency(records: list[CandidateRecord]) -> None:
    """Reject search-order-dependent refits or scores for one partition."""

    by_signature: dict[str, list[CandidateRecord]] = {}
    for record in records:
        by_signature.setdefault(record.partition_signature, []).append(record)
    for signature, matches in by_signature.items():
        if len(matches) < 2:
            continue
        reference = matches[0]
        reference_score = reference.score
        reference_refit = reference.candidate.refit
        for record in matches[1:]:
            score = record.score
            refit = record.candidate.refit
            score_consistent = (score.name, score.degrees_of_freedom, score.n_eff) == (
                reference_score.name,
                reference_score.degrees_of_freedom,
                reference_score.n_eff,
            ) and np.allclose(
                [score.value, score.numerical_uncertainty],
                [reference_score.value, reference_score.numerical_uncertainty],
                rtol=0.0,
                atol=1e-10 * (1.0 + abs(float(reference_score.value))),
            )
            refit_consistent = (
                refit.partition_signature == signature
                and np.array_equal(refit.labels, reference_refit.labels)
                and np.allclose(refit.phi, reference_refit.phi, rtol=0.0, atol=1e-12)
                and np.allclose(
                    refit.cluster_centers,
                    reference_refit.cluster_centers,
                    rtol=0.0,
                    atol=1e-12,
                )
                and np.isclose(
                    refit.loglik,
                    reference_refit.loglik,
                    rtol=0.0,
                    atol=1e-10 * (1.0 + abs(float(reference_refit.loglik))),
                )
            )
            if not score_consistent or not refit_consistent:
                raise AssertionError(
                    "One partition signature produced inconsistent fixed-partition "
                    "refits or scores; selection is not search-order invariant."
                )


def candidate_representative_ids(
    records: list[CandidateRecord],
    *,
    strict_positive_exact_fusion: bool = False,
) -> frozenset[int]:
    """Choose one deterministic eligible representative per partition."""

    admitted = [
        record
        for record in records
        if candidate_is_selection_eligible(
            record,
            strict_positive_exact_fusion=strict_positive_exact_fusion,
        )
    ]
    _assert_same_signature_consistency(admitted)
    best: dict[str, CandidateRecord] = {}
    for record in admitted:
        incumbent = best.get(record.partition_signature)
        if incumbent is None or _candidate_representative_key(
            record
        ) < _candidate_representative_key(incumbent):
            best[record.partition_signature] = record
    return frozenset(int(record.candidate_id) for record in best.values())


def _candidate_representative_key(
    record: CandidateRecord,
) -> tuple[float, float, int, int, str, float, int]:
    return (
        float(record.score.value),
        float(record.score.numerical_uncertainty),
        0 if record.candidate.refit.global_optimum_certified else 1,
        0 if record.family == "raw_fusion" else 1,
        str(record.candidate.partition.source),
        float(record.lambda_value) if record.lambda_value is not None else float("inf"),
        int(record.candidate_id),
    )


def select_candidate_records(
    records: list[CandidateRecord],
    *,
    strict_positive_exact_fusion: bool = False,
) -> CandidateSelectionDecision:
    """Select one partition and representative entirely from typed records.

    Selection preserves the historical ordering: deduplicate by immutable
    partition signature, compare score uncertainty intervals, choose the
    deterministic best partition, then take its least-penalized raw
    representative. Reporting rows have no authority in this function.
    """

    representative_ids = candidate_representative_ids(
        records,
        strict_positive_exact_fusion=strict_positive_exact_fusion,
    )
    eligible = [
        record
        for record in records
        if int(record.candidate_id) in representative_ids
        and candidate_is_selection_eligible(
            record,
            strict_positive_exact_fusion=strict_positive_exact_fusion,
        )
    ]
    if not eligible:
        raise ValueError("No typed candidates are eligible for model selection.")
    if any(not np.isfinite(float(record.score.value)) for record in eligible):
        raise ValueError("Every selectable fixed-partition score must be finite.")

    def score_interval(record: CandidateRecord) -> tuple[float, float]:
        uncertainty = max(float(record.score.numerical_uncertainty), 0.0)
        value = float(record.score.value)
        return value - uncertainty, value + uncertainty

    minimum_upper = min(score_interval(record)[1] for record in eligible)
    tied_signatures = {
        record.partition_signature
        for record in eligible
        if score_interval(record)[0] <= minimum_upper
    }

    def signature_key(signature: str) -> tuple[int, int, float, int, str]:
        rows = [
            record for record in eligible if record.partition_signature == signature
        ]
        raw_lambdas = [
            float(record.lambda_value)
            for record in rows
            if record.lambda_value is not None
        ]
        return (
            min(record.n_clusters for record in rows),
            min(int(record.score.degrees_of_freedom) for record in rows),
            min(raw_lambdas) if raw_lambdas else float("inf"),
            0 if any(record.family == "raw_fusion" for record in rows) else 1,
            str(signature),
        )

    selected_signature = min(tied_signatures, key=signature_key)
    optimal = [
        record
        for record in eligible
        if record.partition_signature == selected_signature
    ]

    def final_representative_key(
        record: CandidateRecord,
    ) -> tuple[float, int, float, int]:
        objective = record.penalized_objective
        return (
            float(record.lambda_value)
            if record.lambda_value is not None
            else float("inf"),
            0 if record.family == "raw_fusion" else 1,
            float(objective) if objective is not None else float("inf"),
            int(record.candidate_id),
        )

    selected = min(optimal, key=final_representative_key)
    eligible_ids = frozenset(int(record.candidate_id) for record in eligible)
    optimal_ids = frozenset(int(record.candidate_id) for record in optimal)
    selected_score = float(selected.score.value)
    optimizer_limited_ids = frozenset(
        int(record.candidate_id)
        for record in records
        if int(record.candidate_id) not in eligible_ids
        and _candidate_is_provisionally_comparable(record)
        and _score_strictly_better(float(record.score.value), selected_score)
    )

    # Model choice remains representative-only, while the reported lambda
    # interval spans every admitted raw fit that reconstructs the selected
    # immutable partition.
    selected_lambdas = sorted(
        {
            _canonical_lambda(float(record.lambda_value))
            for record in records
            if record.partition_signature == selected_signature
            and isinstance(record.candidate, RawFusionCandidate)
            and candidate_is_selection_eligible(
                record,
                strict_positive_exact_fusion=strict_positive_exact_fusion,
            )
            and record.lambda_value is not None
            and np.isfinite(float(record.lambda_value))
            and float(record.lambda_value) >= 0.0
        }
    )
    selected_lambda_left = min(selected_lambdas) if selected_lambdas else None
    selected_lambda_right = max(selected_lambdas) if selected_lambdas else None
    representative_lambda = selected.lambda_value
    evaluated_lambdas = [
        float(record.lambda_value)
        for record in eligible
        if record.lambda_value is not None
    ]
    lower_hit, upper_hit = _lambda_boundary_flags(
        evaluated_lambdas,
        best_lambda_min=representative_lambda,
        best_lambda_max=representative_lambda,
    )
    boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=evaluated_lambdas,
        lower_hit=lower_hit,
        upper_hit=upper_hit,
    )
    return CandidateSelectionDecision(
        selected=selected,
        representative_ids=representative_ids,
        eligible_ids=eligible_ids,
        optimal_ids=optimal_ids,
        optimizer_limited_ids=optimizer_limited_ids,
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selection_hits_lower_boundary=bool(lower_hit),
        selection_hits_upper_boundary=bool(upper_hit),
        selection_boundary_unresolved=bool(boundary_unresolved),
    )


def _candidate_is_provisionally_comparable(record: CandidateRecord) -> bool:
    if not np.isfinite(float(record.score.value)):
        return False
    if record.mm_consistency_violations > 0:
        return False
    if record.family == "direct_partition":
        return True
    objective = record.penalized_objective
    return bool(objective is not None and np.isfinite(float(objective)))


def _canonical_lambda(value: float) -> float:
    return float(np.round(float(value), 12))


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]


def _prefer_fit_candidate(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.selection_eligible and not incumbent.selection_eligible:
        return True
    if candidate.selection_eligible != incumbent.selection_eligible:
        return False
    candidate_objective = float(candidate.penalized_objective)
    incumbent_objective = float(incumbent.penalized_objective)
    if (
        np.isfinite(candidate_objective)
        and np.isfinite(incumbent_objective)
        and abs(candidate_objective - incumbent_objective) > 1e-8
    ):
        return bool(candidate_objective < incumbent_objective)
    if np.isfinite(candidate_objective) and not np.isfinite(incumbent_objective):
        return True
    if not np.isfinite(candidate_objective) and np.isfinite(incumbent_objective):
        return False
    candidate_kkt = float(candidate.fixed_objective_kkt_residual)
    incumbent_kkt = float(incumbent.fixed_objective_kkt_residual)
    if (
        np.isfinite(candidate_kkt)
        and np.isfinite(incumbent_kkt)
        and abs(candidate_kkt - incumbent_kkt) > 1e-8
    ):
        return bool(candidate_kkt < incumbent_kkt)
    if np.isfinite(candidate_kkt) and not np.isfinite(incumbent_kkt):
        return True
    return False


def _effective_bic_partition_tol(options: FitOptions) -> float:
    value = options.selection_partition_tol
    return float(max(float(value), 1e-12))


def _profile_penalty_from_fit(fit: FitResult) -> tuple[float, float]:
    penalty = max(float(fit.penalized_objective + fit.loglik), 0.0)
    if float(fit.lambda_value) > 0.0:
        return penalty, float(penalty / float(fit.lambda_value))
    return penalty, float("nan")


def _score_strictly_better(score: float, reference: float) -> bool:
    if not np.isfinite(score) or not np.isfinite(reference):
        return False
    margin = 1e-8 * (1.0 + abs(float(reference)))
    return bool(float(score) < float(reference) - margin)


def _lambda_boundary_flags(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
) -> tuple[bool, bool]:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas or best_lambda_min is None or best_lambda_max is None:
        return False, False
    lower_hit = np.isclose(best_lambda_min, sorted_lambdas[0], rtol=0.0, atol=1e-12)
    upper_hit = np.isclose(best_lambda_max, sorted_lambdas[-1], rtol=0.0, atol=1e-12)
    return bool(lower_hit), bool(upper_hit)


def _lambda_boundary_unresolved(
    *,
    evaluated_lambdas: list[float],
    lower_hit: bool,
    upper_hit: bool,
) -> bool:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return False
    return bool(lower_hit or upper_hit)
