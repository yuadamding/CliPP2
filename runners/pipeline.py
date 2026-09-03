from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

from ..config import (
    DEFAULT_CHECKPOINT_REQUEST,
    DEFAULT_RUN_CONFIG,
    CheckpointRequest,
    FailurePolicy,
    FitConfig,
    RunConfig,
    resolve_fit_config,
)
from ..io.tumor_txt import load_tumor_txt
from ..model_selection.search import (
    NoCertifiedRawReferenceError,
    NoEligibleModelSelectionCandidatesError,
    select_model,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateRecord,
    DiagnosticOnlyResult,
    SecondaryFallbackResult,
    TumorSelectionOutcome,
)
from ..reporting import (
    AnalysisView,
    analysis_summary,
    write_analysis_outputs,
)


def _outcome_for_failure_policy(
    outcome: TumorSelectionOutcome,
    *,
    failure_policy: FailurePolicy,
    tumor_id: str,
) -> TumorSelectionOutcome:
    if isinstance(outcome, BICSelectionResult):
        return outcome
    if isinstance(outcome, (SecondaryFallbackResult, DiagnosticOnlyResult)):
        reason = str(outcome.reason)
    else:  # pragma: no cover - closed union boundary
        raise TypeError(f"Unsupported selection outcome: {type(outcome).__name__}")
    if failure_policy == "error":
        records = outcome.report.records
        if isinstance(outcome, SecondaryFallbackResult) or reason.startswith(
            "NoCertifiedRawReferenceError:"
        ):
            raise NoCertifiedRawReferenceError(
                tumor_id=str(tumor_id),
                records=list(records),
                adaptive_search_stop_reason=str(
                    outcome.report.adaptive_search_stop_reason
                ),
            )
        if reason.startswith("NoEligibleModelSelectionCandidatesError:"):
            raise NoEligibleModelSelectionCandidatesError(
                tumor_id=str(tumor_id),
                candidates=records,
            )
        raise RuntimeError(reason)
    if failure_policy == "best-effort" or isinstance(outcome, DiagnosticOnlyResult):
        return outcome
    if not isinstance(outcome, SecondaryFallbackResult):
        raise TypeError(f"Unsupported selection outcome: {type(outcome).__name__}")

    # ``save-diagnostics`` deliberately removes the conditional point claim
    # while preserving every retained candidate and raw-attempt trace.
    return DiagnosticOnlyResult(
        best_raw_attempt=outcome.best_raw_attempt,
        reason=(
            f"{reason}; conditional fallback suppressed by "
            "failure_policy=save-diagnostics"
        ),
        report=replace(
            outcome.report,
            selected_id=None,
            num_candidates_certified=0,
            global_hybrid_optimum_certified=False,
        ),
    )


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    *,
    fit_config: FitConfig | None = None,
    run_config: RunConfig = DEFAULT_RUN_CONFIG,
    checkpoint: CheckpointRequest = DEFAULT_CHECKPOINT_REQUEST,
) -> tuple[dict[str, object], tuple[CandidateRecord, ...]]:
    """Fit one canonical tumor TSV file with the default workflow."""

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    data = load_tumor_txt(
        tumor_file,
        unsupported_policy=run_config.unsupported_policy,
        dosage_prior_penalty=run_config.dosage_prior_penalty,
    )

    if fit_config is None:
        fit_config = resolve_fit_config()
    checkpoint_path = checkpoint.resolve_path(
        outdir=outdir,
        tumor_id=str(data.tumor_id),
    )
    selection_result = _outcome_for_failure_policy(
        select_model(
            data=data,
            fit_config=fit_config,
            use_warm_starts=run_config.use_warm_starts,
            checkpoint_path=checkpoint_path,
            resume_checkpoint=checkpoint.resume,
        ),
        failure_policy=run_config.failure_policy,
        tumor_id=str(data.tumor_id),
    )
    analysis = AnalysisView(
        data=data,
        input_file=Path(tumor_file),
        fit_config=fit_config,
        selection_result=selection_result,
    )
    summary = analysis_summary(
        analysis,
        elapsed_seconds=float(perf_counter() - start_time),
    )

    if run_config.write_outputs:
        write_analysis_outputs(
            analysis,
            outdir=outdir,
            summary=summary,
        )
    return summary, selection_result.report.records


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    *,
    fit_config: FitConfig | None = None,
    run_config: RunConfig = DEFAULT_RUN_CONFIG,
    checkpoint: CheckpointRequest = DEFAULT_CHECKPOINT_REQUEST,
) -> dict[str, object]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_config=fit_config,
        run_config=run_config,
        checkpoint=checkpoint,
    )
    return summary


__all__ = [
    "process_tumor",
    "process_tumor_bundle",
]
