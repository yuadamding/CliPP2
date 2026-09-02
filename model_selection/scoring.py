from __future__ import annotations

import numpy as np

from ..config import FitConfig
from ..core.fusion.types import RawFit
from .config import SELECTION_SCORE_NAMES
from .types import (
    CandidateRecord,
    CandidateSelectionDecision,
    RawFusionCandidate,
    SelectionScore,
    UnscoredRawFusionCandidate,
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


def raw_fit_has_exact_fusion_certificate(fit: RawFit) -> bool:
    """Apply the complete schema-2 admission contract to one raw fit.

    This predicate deliberately excludes partition, refit, and score status.
    The online controller must be able to distinguish an uncertified raw
    objective from a certified raw fit whose downstream partition or score is
    unavailable.
    """

    certificate = fit.certificate
    provenance = fit.provenance
    residual_method = str(certificate.residual_method)
    residual = _number_or_nan(certificate.components.residual)
    tolerance = _number_or_nan(certificate.tolerance)
    schema_version = _number_or_nan(certificate.schema_version)
    return bool(
        float(provenance.lambda_value) > 0.0
        and schema_version == _EXACT_CERTIFICATE_SCHEMA_VERSION
        and residual_method == _EXACT_CERTIFICATE_RESIDUAL_METHOD
        and str(certificate.audit_dtype) == "float64"
        and bool(certificate.admissible)
        and bool(str(provenance.objective_spec_hash).strip())
        and bool(str(provenance.original_graph_hash).strip())
        and bool(str(provenance.certificate_problem_hash).strip())
        and str(certificate.scope) == "full_original_graph"
        and str(certificate.gradient_scope)
        in _EXACT_OBSERVED_OBJECTIVE_GRADIENT_SCOPES
        and bool(certificate.certified)
        and str(certificate.status)
        in _EXACT_CERTIFICATE_STATUSES
        and np.isfinite(residual)
        and np.isfinite(tolerance)
        and tolerance > 0.0
        and residual <= tolerance
    )


def raw_candidate_has_exact_fusion_certificate(
    candidate: RawFusionCandidate,
) -> bool:
    """Apply raw, partition, and score admission to a scored candidate."""

    return bool(
        candidate.eligible_for_selection
        and candidate.partition.certified
        and raw_fit_has_exact_fusion_certificate(candidate.raw_fit)
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
    if isinstance(candidate, UnscoredRawFusionCandidate):
        return False
    return bool(candidate.eligible_for_selection and not strict_positive_exact_fusion)


def _require_record_score(record: CandidateRecord) -> SelectionScore:
    score = record.score
    if score is None:
        raise AssertionError("An unscored raw attempt reached partition selection.")
    return score


def _assert_same_signature_consistency(records: list[CandidateRecord]) -> None:
    """Reject search-order-dependent refits or scores for one partition."""

    by_signature: dict[str, list[CandidateRecord]] = {}
    for record in records:
        by_signature.setdefault(record.partition_signature, []).append(record)
    for signature, matches in by_signature.items():
        if len(matches) < 2:
            continue
        reference = matches[0]
        reference_score = _require_record_score(reference)
        reference_refit = reference.candidate.refit
        for record in matches[1:]:
            score = _require_record_score(record)
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
    score = _require_record_score(record)
    return (
        float(score.value),
        float(score.numerical_uncertainty),
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
    if any(
        not np.isfinite(float(_require_record_score(record).value))
        for record in eligible
    ):
        raise ValueError("Every selectable fixed-partition score must be finite.")

    def score_interval(record: CandidateRecord) -> tuple[float, float]:
        score = _require_record_score(record)
        uncertainty = max(float(score.numerical_uncertainty), 0.0)
        value = float(score.value)
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
            min(
                int(_require_record_score(record).degrees_of_freedom)
                for record in rows
            ),
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
        num_eligible=len(eligible),
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selection_hits_lower_boundary=bool(lower_hit),
        selection_hits_upper_boundary=bool(upper_hit),
        selection_boundary_unresolved=bool(boundary_unresolved),
    )
def _canonical_lambda(value: float) -> float:
    return float(np.round(float(value), 12))


def _prefer_fit_candidate(candidate: RawFit, incumbent: RawFit | None) -> bool:
    if incumbent is None:
        return True
    certified = candidate.certificate.admissible
    if certified != incumbent.certificate.admissible:
        return bool(certified)
    candidate_values = (candidate.objective.total, candidate.certificate.components.residual)
    incumbent_values = (incumbent.objective.total, incumbent.certificate.components.residual)
    for candidate_value, incumbent_value in zip(
        candidate_values, incumbent_values, strict=True
    ):
        candidate_finite, incumbent_finite = np.isfinite(
            (candidate_value, incumbent_value)
        )
        if candidate_finite != incumbent_finite:
            return bool(candidate_finite)
        if candidate_finite and abs(candidate_value - incumbent_value) > 1e-8:
            return bool(candidate_value < incumbent_value)
    return False


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]


def _effective_bic_partition_tol(options: FitConfig) -> float:
    value = options.selection.partition_tolerance
    return float(max(float(value), 1e-12))


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
