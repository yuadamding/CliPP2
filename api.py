"""One occupied-clonal-center joint workflow backed by CliPP2's own kernel."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .config import FitConfig, resolve_fit_config
from .io.tumor_txt import load_tumor_txt


def process_tumor_bundle(tumor_file, outdir, fit_config=None, *,
                         checkpoint_path=None, resume=False, checkpoint_stop_after=None):
    """Fit canonical input; verify and publish three minimal tables.

    One occupied conditional center is fixed at one; other centers are free
    except for coordinates constrained by the selected regional-zero policy.
    Candidate-generation convergence is reported,
    not confused with a global-optimality or biological-accuracy proof.
    """
    from .kernel.model import RegionalModel, MODEL_VERSION
    from .kernel.scalar import fit_scalar, supports_scalar
    from .tree.selection import fit_tree
    from .tree.reporting import ALGORITHM, OUTPUT_SCHEMA_VERSION, check_destination, exclusion_summary, publish
    from ._source import source_fingerprint, git_source_identity
    from .kernel import KERNEL_VERSION
    from .kernel.objectives import score_definition_for
    from .kernel.memory import MemoryPolicy, process_memory_snapshot

    config = resolve_fit_config() if fit_config is None else fit_config
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
    if resume and checkpoint_path is None:
        raise ValueError('resume requires checkpoint_path')
    if checkpoint_path is not None and config.search_policy != 'bounded':
        raise ValueError('Checkpointing requires explicit search_policy=bounded')
    if checkpoint_stop_after is not None and checkpoint_path is None:
        raise ValueError('checkpoint_stop_after requires checkpoint_path')
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        if not resume and checkpoint.exists():
            raise FileExistsError('Search checkpoint exists; explicitly resume or use a fresh path')
        if resume and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
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
    memory = MemoryPolicy.resolve(config.ram_budget_gib, config.vram_budget_gib,
                                  device=config.device, qualify_device_weights=config.qualify_device_weights)
    model = RegionalModel(data, device=config.device, memory=memory,
                          regional_zero_policy=config.regional_zero_policy,
                          selection_objective=config.selection_objective)
    model.numerical_diagnostics_dir = config.numerical_diagnostics_dir
    model.numerical_diagnostics_max_bytes = config.numerical_diagnostics_max_bytes
    model.checkpoint_identity = {'source_sha256': source_hash, 'source_commit': commit,
                                 'input_sha256': input_hash}
    search = config.resolved_search(active_hard_zero=bool(model.regional_support.hard_zero.any()))
    continuation = dict(checkpoint_path=checkpoint_path, resume=resume,
                        checkpoint_stop_after=checkpoint_stop_after)
    result = (fit_scalar(model, config.max_clusters, search=search, **continuation) if supports_scalar(model)
              else fit_tree(model, config.max_clusters,
                            search=replace(search, mode=config.search_policy), **continuation))
    # Observation topology is known only after constructing the similarity
    # forest. Preserve the requested policy until then and report the route
    # actually taken, rather than the preliminary capacity-only resolution.
    resolved = result.diagnostics.get('search_policy', {}).get('mode', search.mode)
    search = replace(search, mode=resolved)
    if hashlib.sha256(path.read_bytes()).hexdigest() != input_hash:
        raise RuntimeError('Input changed during fitting; no successful publication')
    if source_fingerprint(source_root) != source_hash:
        raise RuntimeError('Python source changed during fitting; no successful publication')
    verification_start = perf_counter()
    publication = publish(model, result, outdir, data.tumor_id)
    result.diagnostics.update({key: value for key, value in publication['verification'].items()
                               if key in ('regional_zero_policy', 'observed_zero_cells', 'missing_cells',
                                          'forced_zero_cluster_coordinates', 'assigned_zero_violations',
                                          'regional_zero_independently_verified', 'penalty_parameter_count',
                                          'free_center_coordinate_count', 'fixed_zero_coordinate_count')})
    for key, default in (('support_conflicts_rejected', 0), ('support_seed_attempts', 0),
                         ('support_lower_bound', 1), ('support_lower_bound_certified', True)):
        result.diagnostics.setdefault(key, default)
    summary = {
        'tumor_id': data.tumor_id, 'algorithm': ALGORITHM, 'model_version': MODEL_VERSION,
        'output_schema_version': OUTPUT_SCHEMA_VERSION, 'kernel_version': KERNEL_VERSION,
        'source_commit': commit, 'source_dirty': dirty,
        'source_sha256': source_hash, 'input_sha256': input_hash,
        'num_mutations': model.n, 'num_regions': model.r,
        'num_clusters': len(result.centers), 'region_ids': model.region_ids,
        'fit_config': asdict(config), 'model_identity': model.identity,
        'memory_policy': memory.as_dict(), 'memory_caches': model.cache_diagnostics(),
        'memory_usage': process_memory_snapshot(model._torch, model.device),
        'search_policy': asdict(search),
        'requested_search_policy': config.search_policy,
        'regional_zero_policy': config.regional_zero_policy,
        'selection_policy': f'{config.regional_zero_policy}_occupied_clonal_{config.selection_objective}',
        'selection_objective': config.selection_objective,
        'score': result.score, 'score_definition': score_definition_for(config.selection_objective),
        'clonal_constraint': True, 'clonal_centroid_ccf': [1.]*model.r,
        'clonal_cluster_label': 0, 'cluster_zero': 'fixed_occupied_clonal_centroid',
        'exclusions': exclusion_summary(model), 'diagnostics': result.diagnostics,
        'verification_publication_seconds': perf_counter()-verification_start,
        'elapsed_seconds': perf_counter()-started, 'publication': publication,
        'qualification_scope': 'numerical_fit_not_global_optimality_or_biological_accuracy',
    }
    return summary, tuple(result.candidate_bank)


def process_tumor(tumor_file: str | Path, outdir: str | Path,
                  fit_config: FitConfig | None = None, *, checkpoint_path=None,
                  resume=False, checkpoint_stop_after=None) -> dict[str, object]:
    return process_tumor_bundle(tumor_file, outdir, fit_config, checkpoint_path=checkpoint_path,
                                resume=resume, checkpoint_stop_after=checkpoint_stop_after)[0]
