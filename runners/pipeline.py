from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Literal

from ..config import FitConfig, resolve_fit_config
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from ..model_selection.candidates import validate_candidate_identity
from ..model_selection.search import (
    NoCertifiedRawReferenceError,
    NoEligibleModelSelectionCandidatesError,
    select_model,
)
from ..model_selection.types import (
    DiagnosticOnlyResult,
    SearchCandidate,
    SecondaryFallbackResult,
    TumorSelectionOutcome,
)
from .serialization import (
    AnalysisSerialization,
    analysis_summary,
    write_analysis_outputs,
)


FailurePolicy = Literal["error", "save-diagnostics", "best-effort"]
FAILURE_POLICIES: tuple[FailurePolicy, ...] = (
    "error",
    "save-diagnostics",
    "best-effort",
)
DEFAULT_FAILURE_POLICY: FailurePolicy = "best-effort"


def normalize_failure_policy(value: str) -> FailurePolicy:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in FAILURE_POLICIES:
        choices = ", ".join(FAILURE_POLICIES)
        raise ValueError(f"failure_policy must be one of: {choices}")
    return normalized  # type: ignore[return-value]


def _outcome_for_failure_policy(
    outcome: TumorSelectionOutcome,
    *,
    failure_policy: str,
    tumor_id: str,
) -> TumorSelectionOutcome:
    policy = normalize_failure_policy(failure_policy)
    if bool(
        getattr(
            outcome,
            "primary_estimator_available",
            hasattr(outcome, "selected_model"),
        )
    ):
        return outcome

    reason = str(getattr(outcome, "reason", "primary estimator unavailable"))
    if policy == "error":
        records = tuple(item.record for item in outcome.search)
        if isinstance(outcome, SecondaryFallbackResult) or reason.startswith(
            "NoCertifiedRawReferenceError:"
        ):
            raise NoCertifiedRawReferenceError(
                tumor_id=str(tumor_id),
                records=list(records),
                adaptive_search_stop_reason=str(
                    outcome.adaptive_search_stop_reason
                ),
            )
        if reason.startswith("NoEligibleModelSelectionCandidatesError:"):
            raise NoEligibleModelSelectionCandidatesError(
                tumor_id=str(tumor_id),
                candidates=records,
            )
        raise RuntimeError(reason)
    if policy == "best-effort" or isinstance(outcome, DiagnosticOnlyResult):
        return outcome
    if not isinstance(outcome, SecondaryFallbackResult):
        raise TypeError(f"Unsupported selection outcome: {type(outcome).__name__}")

    # ``save-diagnostics`` deliberately removes the conditional point claim
    # while preserving every retained candidate and raw-attempt trace.
    search = tuple(replace(item, selected=False) for item in outcome.search)
    return DiagnosticOnlyResult(
        best_raw_attempt=outcome.best_raw_attempt,
        reason=(
            f"{reason}; conditional fallback suppressed by "
            "failure_policy=save-diagnostics"
        ),
        search=search,
        selection_method=outcome.selection_method,
        adaptive_search_stop_reason=outcome.adaptive_search_stop_reason,
        num_candidates=outcome.num_candidates,
        num_candidates_certified=0,
        ward_candidate_pool_complete=outcome.ward_candidate_pool_complete,
        raw_lambda_path_resolved=outcome.raw_lambda_path_resolved,
        global_hybrid_optimum_certified=False,
    )


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    failure_policy: str = DEFAULT_FAILURE_POLICY,
) -> tuple[dict[str, object], tuple[SearchCandidate, ...]]:
    """Fit one canonical tumor TSV file with the default workflow."""

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    normalized_failure_policy = normalize_failure_policy(failure_policy)
    data = load_tumor_txt(
        tumor_file,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
    )

    if fit_config is None:
        fit_config = resolve_fit_config()
    selection_result = _outcome_for_failure_policy(
        select_model(
            data=data,
            fit_config=fit_config,
            use_warm_starts=use_warm_starts,
        ),
        failure_policy=normalized_failure_policy,
        tumor_id=str(data.tumor_id),
    )
    analysis = AnalysisSerialization(
        data=data,
        input_file=Path(tumor_file),
        fit_config=fit_config,
        selection_result=selection_result,
    )
    if analysis.selected_candidate is not None:
        validate_candidate_identity(analysis.selected_candidate)
    if analysis.raw_reference is not None:
        validate_candidate_identity(analysis.raw_reference)
    summary = analysis_summary(
        analysis,
        elapsed_seconds=float(perf_counter() - start_time),
    )

    if write_outputs:
        write_analysis_outputs(
            analysis,
            outdir=outdir,
            summary=summary,
        )
    return summary, selection_result.search


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    failure_policy: str = DEFAULT_FAILURE_POLICY,
) -> dict[str, object]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_config=fit_config,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
        failure_policy=failure_policy,
    )
    return summary


__all__ = [
    "DEFAULT_FAILURE_POLICY",
    "FAILURE_POLICIES",
    "FailurePolicy",
    "normalize_failure_policy",
    "process_tumor",
    "process_tumor_bundle",
]
