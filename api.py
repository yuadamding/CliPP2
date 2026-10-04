"""One free-center, joint-region workflow backed by CliPP2's own kernel."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt


def process_tumor_bundle(tumor_file, outdir, fit_config=None):
    """Fit canonical input; verify and publish three minimal tables.

    Conditional centers are free. Candidate-generation convergence is reported,
    not confused with a global-optimality or biological-accuracy proof.
    """
    from .kernel.model import RegionalModel, MODEL_VERSION
    from .kernel.scalar import fit_scalar, supports_scalar
    from .tree.selection import fit_tree
    from .tree.reporting import ALGORITHM, check_destination, exclusion_summary, publish
    from ._source import source_fingerprint, git_source_identity
    from .kernel import KERNEL_VERSION, SCORE_DEFINITION

    config = resolve_fit_config() if fit_config is None else fit_config
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
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
    if not path.is_file():
        raise FileNotFoundError(path)
    input_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    data = load_tumor_txt(path, max_major_cn=config.max_major_cn, initialize=False)
    check_destination(outdir, data.tumor_id)
    model = RegionalModel(data, device=config.device)
    result = (fit_scalar(model, config.max_clusters) if supports_scalar(model)
              else fit_tree(model, config.max_clusters))
    if hashlib.sha256(path.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError('Input changed during fitting; no successful publication')
    if source_fingerprint(source_root) != source_hash:
        raise RuntimeError('Python source changed during fitting; no successful publication')
    verification_start = perf_counter()
    publication = publish(model, result, outdir, data.tumor_id)
    summary = {
        'tumor_id': data.tumor_id, 'algorithm': ALGORITHM, 'model_version': MODEL_VERSION,
        'output_schema_version': 9, 'kernel_version': KERNEL_VERSION,
        'source_commit': commit, 'source_dirty': dirty,
        'source_sha256': source_hash, 'input_sha256': input_hash,
        'num_mutations': model.n, 'num_regions': model.r,
        'num_clusters': len(result.centers), 'region_ids': model.region_ids,
        'fit_config': asdict(config), 'model_identity': model.identity,
        'score': result.score, 'score_definition': SCORE_DEFINITION,
        'clonal_constraint': False, 'cluster_zero': 'largest_final_ccf_l2_norm',
        'exclusions': exclusion_summary(model), 'diagnostics': result.diagnostics,
        'verification_publication_seconds': perf_counter()-verification_start,
        'elapsed_seconds': perf_counter()-started, 'publication': publication,
        'qualification_scope': 'numerical_fit_not_global_optimality_or_biological_accuracy',
    }
    return summary, tuple(result.candidate_bank)


def process_tumor(tumor_file: str | Path, outdir: str | Path,
                  fit_config: FitConfig | None = None) -> dict[str, object]:
    return process_tumor_bundle(tumor_file, outdir, fit_config)[0]
