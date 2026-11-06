"""One prepared-data fit, mandatory verification, and publication path."""
from dataclasses import asdict, dataclass
from time import perf_counter

from .config import FitConfig
from .io.data import MultiplicityBounds, readonly_array
from .io.reporting import (ALGORITHM, OUTPUT_SCHEMA_VERSION, check_destination,
                           exclusion_summary, public_label_order, publish)
from .kernel import KERNEL_VERSION, SCORE_DEFINITION
from .kernel.model import RegionalModel
from .tree.progress import ProgressReporter
from .tree.selection import fit_regions


@dataclass(frozen=True)
class FitExecution:
    """Selected native state only; research candidates are explicitly opt-in."""
    fit: object
    mutation_ids: tuple
    region_ids: tuple
    public_order: object
    public_labels: object
    multiplicities: object
    support: MultiplicityBounds
    summary: dict
    producer_receipt: dict
    candidates: tuple = ()


def fit_prepared(data, outdir, config, *, multiplicity_support=None,
                 progress_callback=None, retain_evidence=False, started=None, progress=None):
    """Fit, independently verify, and publish prepared ordinary/workflow data."""
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    started = perf_counter() if started is None else started
    if progress is None and progress_callback is not None:
        progress = ProgressReporter(None, progress_callback)
        progress.start_phase('input')
    check_destination(outdir, data.tumor_id)
    model = RegionalModel(data, device=config.device, cpu_threads=config.cpu_threads,
                          multiplicity_support=multiplicity_support)
    input_seconds = perf_counter()-started
    try:
        if progress is not None:
            progress.model = model
            progress.emit(complete=True)
        fit_offset = perf_counter()-started
        callback = (None if progress_callback is None else lambda record:
                    progress_callback(dict(record, elapsed_seconds=record['elapsed_seconds']+fit_offset)))
        result = fit_regions(model, config.max_clusters, progress_callback=callback,
                             retain_evidence=retain_evidence, audit_scope=config.audit_scope)
        verification_start = perf_counter()
        publication = publish(model, result, outdir, data.tumor_id, progress=progress)
        audit = publication['verification']['membership_audit']
        summary = {
            'tumor_id': data.tumor_id, 'algorithm': ALGORITHM, 'model_version': model.model_version,
            'output_schema_version': OUTPUT_SCHEMA_VERSION, 'diagnostics_schema_version': 3,
            'kernel_version': KERNEL_VERSION,
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
                **{key: audit[key] for key in ('selected_count_certified',
                    'all_represented_counts_certified', 'represented_cluster_counts')},
                'unresolved_terminal_evaluations': audit['unresolved_evaluations'],
                'search_work': result.diagnostics['search_work'], 'global_optimality_certified': False},
            'input_seconds': input_seconds,
            'verification_publication_seconds': perf_counter()-verification_start,
            'elapsed_seconds': perf_counter()-started, 'publication': publication,
            'qualification_scope': 'numerical_fit_not_global_optimality_or_biological_accuracy',
        }
        receipt = dict(schema='clipp2.fit-producer.v1', **{key: summary[key] for key in (
            'algorithm', 'model_version', 'kernel_version', 'fit_config', 'model_identity',
            'multiplicity_support', 'execution', 'publication', 'qualification_scope')})
        fit = result.selected.fit
        order, labels = public_label_order(fit)
        support = MultiplicityBounds(model.mutation_ids, model.region_ids,
            model.multiplicity_min, model.multiplicity_max,
            'original_support' if multiplicity_support is None else multiplicity_support.policy_id)
        return FitExecution(fit, model.mutation_ids, model.region_ids, readonly_array(order),
            readonly_array(labels), readonly_array(model.multiplicity(fit.centers[fit.labels])),
            support, summary, receipt, tuple(result.candidate_bank) if retain_evidence else ())
    finally:
        model.backend.close()
