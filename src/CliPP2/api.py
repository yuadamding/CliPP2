"""One free-center, joint-region workflow backed by CliPP2's own kernel."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt


def process_tumor(tumor_file: str | Path, outdir: str | Path,
                  fit_config: FitConfig | None = None, *, multiplicity_support=None,
                  progress_callback=None, run_manifest_id=None, normal_cn_policy=None) -> dict[str, object]:
    """Publish the three fitted tables and return compact operational metadata.

    Verification uses retained objects and compact attempt links. Progress
    callbacks receive schema-3 detached snapshots.
    """
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
        data = load_tumor_txt(path, max_major_cn=config.max_major_cn, normal_cn_policy=normal_cn_policy)
    else:
        from .io.tumor_txt import load_original_catalog, load_multiplicity_bounds
        from .io.data import MultiplicityBounds
        catalog = load_original_catalog(path, max_major_cn=config.max_major_cn)
        data = catalog.fit_data(normal_cn_policy=normal_cn_policy)
        if isinstance(multiplicity_support, (str, Path)):
            multiplicity_support = load_multiplicity_bounds(multiplicity_support, catalog)
        elif type(multiplicity_support) is not MultiplicityBounds:
            raise TypeError('multiplicity_support must be MultiplicityBounds or a support TSV path')
    execution = fit_prepared(data, outdir, config, multiplicity_support=multiplicity_support,
        progress_callback=progress_callback,
        started=started, progress=progress, run_manifest_id=run_manifest_id)
    summary = execution.summary
    del summary['diagnostics']
    return summary
