"""One free-center, joint-region workflow backed by CliPP2's own kernel."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt
from .io.region_files import input_fingerprint


def process_tumor_bundle(tumor_file, outdir, fit_config=None, *, progress_callback=None):
    """Publish three tables and explicitly retain in-memory research evidence.

    Unlike ``process_tumor``, this diagnostic API returns the complete search
    ledger and candidate bank. Neither API writes these as extra output files.
    An optional progress callback receives detached completed-operation records;
    verbose mode otherwise prints them to stderr, never into the output folder.
    """
    from .kernel.model import RegionalModel, MODEL_VERSION
    from .tree.selection import fit_regions
    from .io.reporting import ALGORITHM, OUTPUT_SCHEMA_VERSION, check_destination, exclusion_summary, publish
    from ._source import source_fingerprint, git_source_identity
    from .kernel import KERNEL_VERSION, SCORE_DEFINITION

    config = resolve_fit_config() if fit_config is None else fit_config
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    if progress_callback is None and config.verbose:
        def progress_callback(record):
            print(json.dumps(record, allow_nan=False), file=sys.stderr, flush=True)
    started = perf_counter()
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
    data = load_tumor_txt(path, max_major_cn=config.max_major_cn, initialize=False)
    check_destination(outdir, data.tumor_id)
    model = RegionalModel(data, device=config.device,
                          grid_cache_bytes=config.grid_cache_mb * 1024**2,
                          device_cache_bytes=config.device_cache_mb * 1024**2,
                          workspace_bytes=config.workspace_mb * 1024**2,
                          native_host_memory_bytes=(None if config.native_host_memory_mb is None
                                                    else config.native_host_memory_mb * 1024**2),
                          device_memory_bytes=(None if config.device_memory_mb is None
                                               else config.device_memory_mb * 1024**2),
                          cpu_threads=config.cpu_threads, score_capacity=config.max_clusters)
    try:
        result = (fit_regions(model, config.max_clusters) if progress_callback is None else
                  fit_regions(model, config.max_clusters, progress_callback=progress_callback))
        if input_fingerprint(path) != input_hash:
            raise RuntimeError('Input changed during fitting; no successful publication')
        if source_fingerprint(source_root) != source_hash:
            raise RuntimeError('Python source changed during fitting; no successful publication')
        verification_start = perf_counter()
        publication = publish(model, result, outdir, data.tumor_id)
        summary = {
            'tumor_id': data.tumor_id, 'algorithm': ALGORITHM, 'model_version': MODEL_VERSION,
            'output_schema_version': OUTPUT_SCHEMA_VERSION, 'kernel_version': KERNEL_VERSION,
            'source_commit': commit, 'source_dirty': dirty,
            'source_sha256': source_hash, 'input_sha256': input_hash,
            'num_mutations': model.n, 'num_regions': model.r,
            'num_clusters': len(result.centers), 'region_ids': model.region_ids,
            'fit_config': asdict(config), 'model_identity': model.identity,
            'score': result.score, 'score_definition': SCORE_DEFINITION,
            'clonal_constraint': False, 'cluster_zero': 'largest_final_ccf_l2_norm',
            'exclusions': exclusion_summary(model), 'diagnostics': result.diagnostics,
            'execution': model.execution_diagnostics(),
            'verification_publication_seconds': perf_counter()-verification_start,
            'elapsed_seconds': perf_counter()-started, 'publication': publication,
            'qualification_scope': 'numerical_fit_not_global_optimality_or_biological_accuracy',
        }
        return summary, tuple(result.candidate_bank)
    finally:
        model.backend.close()


def process_tumor(tumor_file: str | Path, outdir: str | Path,
                  fit_config: FitConfig | None = None, *, progress_callback=None) -> dict[str, object]:
    """Publish the README's three tables; return compact operational metadata.

    Full search diagnostics are used for verification but not returned here,
    preventing routine callers from serializing the potentially large ledger.
    """
    summary, _ = (process_tumor_bundle(tumor_file, outdir, fit_config) if progress_callback is None else
                  process_tumor_bundle(tumor_file, outdir, fit_config, progress_callback=progress_callback))
    del summary['diagnostics']
    return summary
