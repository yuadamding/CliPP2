"""One validated, completion-last tumor fitting workflow."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config, _resolve_fit_options
from .io.tumor_txt import NoEligibleSNVsError, load_tumor_txt
from .model_selection.search import select_model
from .model_selection.types import SearchCandidate
from .reporting import (
    RunPublication, AnalysisSerialization, analysis_summary, write_analysis_outputs,
    _file_hash,
)


def process_tumor_bundle(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
) -> tuple[dict[str, object], tuple[SearchCandidate, ...]]:
    """Fit and publish one tumor; return its summary and candidate evidence."""
    start_time = perf_counter()
    tumor_file, outdir = Path(tumor_file), Path(outdir)
    if not tumor_file.is_file():
        raise FileNotFoundError(f"Tumor input must be a file: {tumor_file}")
    options = _resolve_fit_options(resolve_fit_config() if fit_config is None else fit_config)
    input_sha256 = _file_hash(tumor_file)
    workflow = {"entrypoint": "process_tumor_bundle", "use_warm_starts": True}
    try:
        data = load_tumor_txt(tumor_file, eps=options.eps, max_major_cn=options.max_major_cn)
    except NoEligibleSNVsError as error:
        publication = RunPublication(outdir, error.tumor_id, input_file=tumor_file,
                                     expected_input_sha256=input_sha256,
                                     fit_config=options, workflow=workflow)
        try:
            publication.write_audit(error.cn_filter_report)
        finally:
            publication.fail(error)
        raise
    publication = RunPublication(outdir, data.tumor_id, input_file=tumor_file,
                                 expected_input_sha256=input_sha256,
                                 fit_config=options, workflow=workflow)
    try:
        publication.write_audit(data.cn_filter_report)
        selection_result = select_model(data=data, fit_config=options)
        # This boundary validates selected/refit/raw-parent identities once.
        analysis = AnalysisSerialization(data=data, input_file=tumor_file,
                                         fit_config=options, selection_result=selection_result)
        summary = analysis_summary(analysis, elapsed_seconds=perf_counter() - start_time)
        write_analysis_outputs(analysis, outdir=outdir, publication=publication)
        summary["run_id"] = publication.record["run_id"]
        summary["run_manifest"] = str(publication.path)
    except BaseException as error:
        publication.fail(error)
        raise
    return summary, selection_result.search


def process_tumor(
    tumor_file: str | Path,
    outdir: str | Path,
    fit_config: FitConfig | None = None,
) -> dict[str, object]:
    """Fit and publish one tumor TSV using the fixed production estimator."""
    return process_tumor_bundle(tumor_file, outdir, fit_config)[0]
