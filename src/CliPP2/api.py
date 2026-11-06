"""One free-center, joint-region workflow backed by CliPP2's own kernel."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt
from .io.region_files import input_fingerprint


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
    from .kernel.model import RegionalModel
    from .tree.selection import fit_regions
    from .io.reporting import ALGORITHM, OUTPUT_SCHEMA_VERSION, check_destination, exclusion_summary, publish
    from ._source import source_fingerprint, git_source_identity
    from .kernel import KERNEL_VERSION, SCORE_DEFINITION
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
    source_root = Path(__file__).resolve().parent
    commit, dirty = git_source_identity(source_root)
    source_hash = source_fingerprint(source_root)
    build_record = source_root / '_build_source.json'
    if commit is None and build_record.is_file():
        built = json.loads(build_record.read_text())
        if built.get('python_source_sha256') != source_hash:
            raise RuntimeError('Installed Python source disagrees with its build record')
        commit, dirty = built.get('commit'), built.get('dirty')
    path = Path(tumor_file)
    input_hash = input_fingerprint(path)
    support_hash = None
    support_path = None
    if multiplicity_support is None:
        data = load_tumor_txt(path, max_major_cn=config.max_major_cn, initialize=False)
    else:
        from .io.tumor_txt import load_original_catalog, load_multiplicity_bounds
        from .io.data import MultiplicityBounds
        catalog = load_original_catalog(path, max_major_cn=config.max_major_cn)
        data = catalog.fit_data()
        if isinstance(multiplicity_support, (str, Path)):
            from ._source import file_sha256
            support_path = Path(multiplicity_support)
            support_hash = file_sha256(support_path)
            multiplicity_support = load_multiplicity_bounds(support_path, catalog)
        elif type(multiplicity_support) is not MultiplicityBounds:
            raise TypeError('multiplicity_support must be MultiplicityBounds or a support TSV path')
    check_destination(outdir, data.tumor_id)
    model = RegionalModel(data, device=config.device, cpu_threads=config.cpu_threads,
                          multiplicity_support=multiplicity_support)
    input_seconds = perf_counter()-started
    try:
        if progress is not None:
            progress.model = model
            progress.emit(complete=True)
        fit_offset = perf_counter()-started
        fit_callback = (None if progress_callback is None else lambda record:
                        progress_callback(dict(record, elapsed_seconds=record['elapsed_seconds']+fit_offset)))
        result = fit_regions(model, config.max_clusters, progress_callback=fit_callback,
                             retain_evidence=retain_evidence, audit_scope=config.audit_scope)
        if input_fingerprint(path) != input_hash:
            raise RuntimeError('Input changed during fitting; no successful publication')
        if support_path is not None and file_sha256(support_path) != support_hash:
            raise RuntimeError('Multiplicity support changed during fitting; no successful publication')
        if source_fingerprint(source_root) != source_hash:
            raise RuntimeError('Python source changed during fitting; no successful publication')
        verification_start = perf_counter()
        publication = publish(model, result, outdir, data.tumor_id, progress=progress)
        summary = {
            'tumor_id': data.tumor_id, 'algorithm': ALGORITHM, 'model_version': model.model_version,
            'output_schema_version': OUTPUT_SCHEMA_VERSION, 'diagnostics_schema_version': 3,
            'kernel_version': KERNEL_VERSION,
            'source_commit': commit, 'source_dirty': dirty,
            'source_sha256': source_hash, 'input_sha256': input_hash,
            'num_mutations': model.n, 'num_regions': model.r,
            'num_clusters': len(result.centers), 'region_ids': model.region_ids,
            'fit_config': asdict(config), 'model_identity': model.identity,
            'multiplicity_support': None if multiplicity_support is None else {
                'policy_id': multiplicity_support.policy_id, 'content_hash': multiplicity_support.content_hash},
            'score': result.score, 'score_definition': SCORE_DEFINITION,
            'clonal_constraint': False, 'cluster_zero': 'largest_final_ccf_l2_norm',
            'exclusions': exclusion_summary(model), 'diagnostics': result.diagnostics,
            'execution': model.execution_diagnostics(),
            'search_status': {'policy': result.diagnostics['search_policy'],
                'audit_scope': config.audit_scope,
                'selected_q': len(result.centers), 'selected_state_id': result.selected.state_index,
                'unresolved_terminal_evaluations': publication['verification']['membership_audit'][
                    'unresolved_evaluations'],
                'selected_count_certified': publication['verification']['membership_audit'][
                    'selected_count_certified'],
                'all_represented_counts_certified': publication['verification']['membership_audit'][
                    'all_represented_counts_certified'],
                'represented_cluster_counts': publication['verification']['membership_audit'][
                    'represented_cluster_counts'],
                'search_work': result.diagnostics['search_work'], 'global_optimality_certified': False},
            'input_seconds': input_seconds,
            'verification_publication_seconds': perf_counter()-verification_start,
            'elapsed_seconds': perf_counter()-started, 'publication': publication,
            'qualification_scope': 'numerical_fit_not_global_optimality_or_biological_accuracy',
        }
        return summary, tuple(result.candidate_bank) if retain_evidence else ()
    finally:
        model.backend.close()


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
                 accepted_stage_one=None):
    """Execute the three-stage workflow with an optional accepted-fit receipt."""
    from .workflow import run_workflow as execute
    return execute(input_path, outdir, workflow_config, progress_callback=progress_callback,
                   accepted_stage_one=accepted_stage_one)
