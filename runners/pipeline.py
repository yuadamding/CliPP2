from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .._version import __version__ as _SOFTWARE_VERSION
from ..core.model import FitOptions
from ..io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY, load_tumor_txt
from ..model_selection.config import FINAL_PHI_WARD_LADDER_KMAX
from .model_selection import select_model
from ..model_selection.candidates import validate_candidate_identity
from .outputs import write_fit_outputs


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
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
    selection_result = select_model(
        data=data,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        ward_ladder_kmax=int(ward_ladder_kmax),
    )
    selected_model = selection_result.selected_model
    selected_candidate = selected_model.candidate
    validate_candidate_identity(selected_candidate)
    best_fit = selected_candidate.raw_fit
    partition = selected_candidate.partition
    refit = selected_candidate.refit
    score = selected_candidate.score
    search_df = selection_result.search_df
    elapsed_seconds = float(perf_counter() - start_time)
    selected_lambda = selection_result.selected_lambda_representative
    summary: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "input_file": str(tumor_file),
        "selected_lambda": (
            np.nan if selected_lambda is None else float(selected_lambda)
        ),
        "selected_n_clusters": int(partition.n_clusters),
        "selected_partition_signature": str(partition.signature),
        "selection_score_name": str(score.name),
        "selection_metric_value": (
            np.nan
            if selection_result.selection_metric_value is None
            else float(selection_result.selection_metric_value)
        ),
        "selection_method": str(selection_result.selection_method),
        "num_candidates": int(selection_result.num_candidates),
        "num_candidates_certified": int(selection_result.num_candidates_certified),
        "selected_kkt_residual": (
            np.nan
            if selection_result.selected_kkt_residual is None
            else float(selection_result.selected_kkt_residual)
        ),
        "search_stop_reason": str(selection_result.adaptive_search_stop_reason),
        "device": str(best_fit.device),
        "dtype": str(best_fit.dtype),
        "elapsed_seconds": elapsed_seconds,
        "software_version": _SOFTWARE_VERSION,
    }

    if write_outputs:
        validate_candidate_identity(selected_candidate)
        write_fit_outputs(
            outdir=outdir,
            data=data,
            raw_fit=best_fit,
            partition=partition,
            refit=refit,
            major_prior=float(fit_options.major_prior),
        )
    return summary, search_df


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_options: FitOptions | None = None,
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    ward_ladder_kmax: int = FINAL_PHI_WARD_LADDER_KMAX,
) -> dict[str, float | int | str | bool]:
    """Fit one tumor TSV file."""

    summary, _ = process_tumor_bundle(
        tumor_file=tumor_file,
        outdir=outdir,
        fit_options=fit_options,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
        ward_ladder_kmax=int(ward_ladder_kmax),
    )
    return summary
