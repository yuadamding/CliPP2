from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from ..core.fusion.profiles import get_computation_profile
from ..core.model import FitOptions
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from .model_selection import select_model
from ..model_selection.candidates import validate_candidate_identity
from .serialization import (
    AnalysisSerialization,
    analysis_summary,
    write_analysis_outputs,
)


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit one canonical tumor TSV file with the default workflow."""

    start_time = perf_counter()
    tumor_file = Path(tumor_file)
    outdir = Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    data = load_tumor_txt(
        tumor_file,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
    )

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)
    computation_profile = get_computation_profile(fit_options.computation_profile)
    selection_result = select_model(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
    )
    analysis = AnalysisSerialization.from_selection(
        data=data,
        input_file=tumor_file,
        fit_options=fit_options,
        computation_profile=computation_profile,
        selection_result=selection_result,
    )
    validate_candidate_identity(analysis.selected_candidate)
    validate_candidate_identity(analysis.raw_reference)
    summary = analysis_summary(
        analysis,
        elapsed_seconds=float(perf_counter() - start_time),
    )

    if write_outputs:
        write_analysis_outputs(
            analysis,
            outdir=outdir,
        )
    return summary, selection_result.search_df


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
) -> dict[str, object]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
    )
    return summary
