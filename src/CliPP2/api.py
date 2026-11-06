"""One free-center, joint-region workflow backed by CliPP2's own kernel."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt


def process_tumor_bundle(tumor_file, outdir, fit_config=None, *, multiplicity_support=None, progress_callback=None):
    """Publish three tables and explicitly retain in-memory research evidence.

    Unlike ``process_tumor``, this diagnostic API returns the complete search
    ledger and candidate bank. Neither API writes these as extra output files.
    An optional progress callback receives detached phase/candidate/batch snapshots;
    Schema 3 also reports audit-pass coverage and verification/output phases.
    """
    return _process_tumor(tumor_file, outdir, fit_config,
                          multiplicity_support=multiplicity_support,
                          progress_callback=progress_callback, retain_evidence=True)


def _process_tumor(tumor_file, outdir, fit_config=None, *, multiplicity_support=None,
                   progress_callback=None, retain_evidence=False):
    """Shared numerical/publication path; research exports are explicitly opt-in."""
    from .fit_execution import fit_prepared
    from .tree.progress import ProgressReporter

    config = resolve_fit_config() if fit_config is None else fit_config
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    started = perf_counter()
    progress = None if progress_callback is None else ProgressReporter(None, progress_callback)
    if progress is not None:
        progress.start_phase('input')
    path = Path(tumor_file)
    if multiplicity_support is None:
        data = load_tumor_txt(path, max_major_cn=config.max_major_cn, initialize=False)
    else:
        from .io.tumor_txt import load_original_catalog, load_multiplicity_bounds
        from .io.data import MultiplicityBounds
        catalog = load_original_catalog(path, max_major_cn=config.max_major_cn)
        data = catalog.fit_data()
        if isinstance(multiplicity_support, (str, Path)):
            multiplicity_support = load_multiplicity_bounds(multiplicity_support, catalog)
        elif type(multiplicity_support) is not MultiplicityBounds:
            raise TypeError('multiplicity_support must be MultiplicityBounds or a support TSV path')
    execution = fit_prepared(data, outdir, config, multiplicity_support=multiplicity_support,
        progress_callback=progress_callback, retain_evidence=retain_evidence,
        started=started, progress=progress)
    return execution.summary, execution.candidates


def process_tumor(tumor_file: str | Path, outdir: str | Path,
                  fit_config: FitConfig | None = None, *, multiplicity_support=None,
                  progress_callback=None) -> dict[str, object]:
    """Publish the README's three tables; return compact operational metadata.

    Verification uses retained objects and compact attempt links, not a JSON
    research ledger. Progress callbacks receive schema-3 detached snapshots.
    """
    summary, _ = _process_tumor(tumor_file, outdir, fit_config,
                               multiplicity_support=multiplicity_support, progress_callback=progress_callback)
    del summary['diagnostics']
    return summary


def assign_mutations(bundle, *, outdir=None, progress_callback=None):
    """Evaluate a validated frozen reference under both multiplicity policies."""
    from .assignment import assign_mutations as execute
    return execute(bundle, outdir=outdir, progress_callback=progress_callback)


def run_workflow(input_path, outdir, workflow_config, *, progress_callback=None,
                 accepted_stage_one=None, accepted_stage_two=None):
    """Execute or replay native stages from complete accepted-fit.v2 receipts.

    Earlier summaries/TSVs are not an import format; inadequate evidence raises
    without silently fitting. Original producer and replay checks stay separate.
    """
    from .workflow import run_workflow as execute
    return execute(input_path, outdir, workflow_config, progress_callback=progress_callback,
                   accepted_stage_one=accepted_stage_one, accepted_stage_two=accepted_stage_two)
