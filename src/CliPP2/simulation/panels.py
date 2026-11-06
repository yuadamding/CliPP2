"""Prespecified optional sensitivity jobs and fixed-reference resource panels.

Building these jobs does not execute them. Unselected panels remain explicitly
planned, and contributor-source controls are not accepted native fitted states.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import resource
import sys
import tempfile
from time import perf_counter
import traceback

import numpy as np

from CliPP2.assignment import AssignmentBundle, assign_mutations, confidence_passes
from CliPP2.screening import profile_deviance_from_diagnostics
from CliPP2.config import WorkflowConfig
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction

from .acquisition import acquire
from .biology import _plain, generate_history
from .design import seed_for


def sensitivity_jobs(contract, *, extended=False):
    """Matched single perturbations; optional small one-factor acquisition grid."""
    conditions = [('control', {})]
    conditions += [(f'purity_{delta:+.2f}', {'purity_offset': delta}) for delta in (-.10, -.05, .05, .10)]
    conditions += [(f'cn_{value:.2f}', {'cn_error_fraction': value}) for value in (.05, .15)]
    conditions += [(f'mapping_{value:.2f}', {'segment_error_fraction': value}) for value in (.05, .15)]
    conditions += [(f'icc_{value:.2f}', {'overdispersion': value}) for value in (.01, .05)]
    conditions += [(f'odds_{value:.2f}', {'alt_odds_ratio': value}) for value in (.8, 1.25)]
    conditions += [('discovery_alt3', {'discovery_min_alt': 3}),
                   ('combined', {'purity_offset': -.05, 'cn_error_fraction': .05, 'alt_odds_ratio': .8})]
    jobs = []
    for family in ('S02', 'S04', 'S06', 'S12'):
        history_seed = seed_for(contract['master_seed'], 'sensitivity', family, 0)
        acquisition_seed = seed_for(contract['master_seed'], 'sensitivity', family, 0, 4)
        history_id = f'sensitivity-{family}-H001'
        base = dict(phase='sensitivity', family=family, history_id=history_id, history_index=0,
            history_seed=history_seed, acquisition_seed=acquisition_seed,
            dimensions=dict(contract['dimensions']), panel_status='planned_not_executed')
        for name, perturbation in conditions:
            jobs.append(dict(base, cell_id=f'{history_id}-{name}',
                acquisition=dict(id=name, purity=.8, depth=500), perturbation=perturbation,
                pairing='same physical history and seed; input-only errors preserve raw counts'))
        if not extended:
            continue
        for regions in (1, 2, 4, 8):
            for mode, depth in (('per_region', 500), ('fixed_total', 2000/regions)):
                name = f'regions{regions}_{mode}'
                jobs.append(dict(base, cell_id=f'{history_id}-{name}',
                    dimensions=dict(contract['dimensions'], regions=regions),
                    acquisition=dict(id=name, purity=.8, depth=depth), perturbation={},
                    pairing='same physical copy tree and existing-mixture prefix; acquisition seeds paired'))
        for depth in (30, 100, 300, 1000):
            name = f'depth{depth}'
            jobs.append(dict(base, cell_id=f'{history_id}-{name}',
                acquisition=dict(id=name, purity=.8, depth=depth), perturbation={}))
        for purity in (.2, .5, .9):
            name = f'purity{purity:.1f}'
            jobs.append(dict(base, cell_id=f'{history_id}-{name}',
                acquisition=dict(id=name, purity=purity, depth=500), perturbation={}))
    assert len({job['cell_id'] for job in jobs}) == len(jobs)
    return jobs


def contributor_density_diagnostics(history, frame, config, *, densities=(0, 1, 5, 20, 100)):
    """Fixed target set and deterministic nested source subsets, with no fits.

    Selection uses only declared source identities. Qualification evaluates noisy
    calls at true source CCFs. A requested density above the available physical
    source count is labeled limited, never filled with duplicated contributors.
    """
    pool = tuple(history.metadata['declared_source_ids'])
    targets = tuple(history.metadata['declared_remainder_ids'])
    if any(type(value) is not int or value < 0 for value in densities):
        raise ValueError('Contributor densities must be nonnegative integers')
    index = {mutation: i for i, mutation in enumerate(history.mutation_ids)}
    positions = [index[mutation] for mutation in pool]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'input.tsv'
        write_tumor_txt(path, frame, frame.attrs['metadata'])
        catalog = load_original_catalog(path, max_major_cn=config.fit.max_major_cn)
    if not pool:
        return dict(status='not_applicable_no_declared_source_pool', target_ids=targets, records=[])
    data = prepare_prediction(catalog, pool, device=config.fit.device, cpu_threads=config.fit.cpu_threads)
    phi = history.phi[positions]
    with NativeBackend(data, cpu_threads=config.fit.cpu_threads) as backend:
        diagnostic = backend.component_diagnostics(phi)
    if np.isin(diagnostic['status'], (2, 3)).any():
        raise ArithmeticError('Native source diagnostics failed')
    residual = profile_deviance_from_diagnostics(data, phi, diagnostic)
    confidence = confidence_passes(diagnostic['map_log_probability'], config.contributor_probability_min,
        diagnostic['has_second'] & np.isfinite(diagnostic['top_two_gap']))
    qualified = (data.observed & catalog.mapping_genuine[positions] & catalog.eligible[positions]
                 & (catalog.cn_state_count[positions] == 1) & (phi > 0) & (diagnostic['status'] == 0) & confidence
                 & (diagnostic['top_two_gap'] > config.contributor_tie_atol)
                 & np.isfinite(residual) & (residual <= config.contributor_deviance_max))
    source_lookup = {mutation: i for i, mutation in enumerate(pool)}
    source_sets = {segment: tuple(m for m in pool if history.segment_ids[index[m], 0] == segment)
                   for segment in history.segment_keys}
    records = []
    for density in densities:
        selected = {segment: ids[:density] for segment, ids in source_sets.items()}
        audits = []
        totals = dict(evaluable_pairs=0, estimated_exclusions=0, structural_exclusions=0,
                      same_qualified_truth_exclusions=0, same_qualified_truth_unavailable=0,
                      structural_truth_unavailable=0, inherited_support_exclusions=0,
                      additional_cap_exclusions=0, qualified_calls=0, correct_qualified_calls=0)
        for segment in history.segment_keys:
            selected_ids = selected[segment]
            local_targets = [index[m] for m in targets if history.segment_ids[index[m], 0] == segment]
            for r, region in enumerate(history.region_ids):
                ids = [m for m in selected_ids if qualified[source_lookup[m], r]]
                calls = [diagnostic['map_multiplicity'][source_lookup[m], r] for m in ids]
                truth = [history.multiplicity[index[m], r] for m in ids]
                selected_truth = [history.multiplicity[index[m], r] for m in selected_ids]
                truth_known = all(np.isfinite(value) for value in truth)
                structural_known = all(np.isfinite(value) for value in selected_truth)
                estimated_cap = min(calls) if calls else None
                replacement_cap = min(truth) if truth and truth_known else None
                structural_cap = min(selected_truth) if selected_truth and structural_known else None
                totals['qualified_calls'] += len(calls)
                totals['correct_qualified_calls'] += sum(call == truth for call, truth in zip(calls, truth))
                for i in local_targets:
                    if not catalog.eligible[i, r] or history.phi[i, r] <= 0:
                        continue
                    m, upper = history.multiplicity[i, r], catalog.major_cn[i, r]
                    totals['evaluable_pairs'] += 1
                    totals['estimated_exclusions'] += int(m > (upper if estimated_cap is None else estimated_cap))
                    totals['inherited_support_exclusions'] += int(m > upper)
                    totals['additional_cap_exclusions'] += int(m <= upper and estimated_cap is not None and m > estimated_cap)
                    if structural_known:
                        totals['structural_exclusions'] += int(m > (upper if structural_cap is None else structural_cap))
                    else:
                        totals['structural_truth_unavailable'] += 1
                    if not truth_known:
                        totals['same_qualified_truth_unavailable'] += 1
                    else:
                        totals['same_qualified_truth_exclusions'] += int(m > (upper if replacement_cap is None else replacement_cap))
                audits.append(dict(segment_id=segment, region_id=region, requested_count=density,
                    available_count=len(source_sets[segment]), selected_count=len(selected_ids),
                    qualified_count=len(ids), qualified_ids=ids, estimated_cap=estimated_cap,
                    structural_cap=structural_cap, same_qualified_ids_truth_cap=replacement_cap,
                    same_qualified_ids_truth_evaluable=truth_known, original_upper_fallback=not calls))
        records.append(dict(requested_density=density,
            availability='limited_by_source_availability' if any(len(ids) < density for ids in source_sets.values()) else 'complete',
            selected_source_ids=[m for ids in selected.values() for m in ids], **totals, segment_audit=audits))
    return _plain(dict(status='complete', scope='controlled_true_center_call_qualification_not_a_native_fit',
        target_ids=targets, source_pool_ids=pool, records=records,
        notes='The target population is fixed across all densities; missing positive-carrier truth is not substituted.'))


def _scaling_case(path, config_values, mutations, regions, band, seed):
    """One fresh process supplies a cell-specific native+Python peak RSS."""
    path = Path(path)
    started = perf_counter()
    try:
        config = WorkflowConfig.from_dict(config_values)
        history = generate_history('S02', seed, mutations, 50, regions,
                                   major_cn_range=(3, 6) if band == 'moderate' else (7, 11))
        frame = acquire(history, seed+1, .8, 100)
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / 'scaling.tsv'
            write_tumor_txt(input_path, frame, frame.attrs['metadata'])
            catalog = load_original_catalog(input_path)
        centers = {truth_id: history.phi[i] for i, truth_id in enumerate(history.observable_truth_ids)}
        clonal = next(key for key, center in centers.items() if np.all(center == 1))
        order = [clonal, *(key for key in sorted(centers) if key != clonal)]
        labels = tuple(order.index(key) for key in history.observable_truth_ids)
        from .controls import _controlled_reference
        reference = _controlled_reference(tuple(range(len(order))), history.region_ids, np.array([centers[key] for key in order]))
        support = catalog.default_bounds()
        observed = catalog.count_observed & (catalog.total_counts > 0) & catalog.eligible
        bundle = AssignmentBundle(catalog, reference, history.mutation_ids, (), config.assignment,
            history.phi, np.where(observed, history.multiplicity, np.nan), support.lower, support.upper,
            ('controlled_truth_reference',)*mutations, labels, labels, config.fit.device, config.fit.cpu_threads,
            run_manifest_id=os.environ.get("CLIPP2_RUN_MANIFEST_ID"),
            provenance=dict(scope='fixed_reference_assignment_scaling_no_fit'))
        prepared = perf_counter()
        result = assign_mutations(bundle)
        scored = perf_counter()
        assert len(result.assignments) == mutations and len(result.multiplicities) == mutations*regions
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        payload = dict(status='complete', scope='assignment_only_complete_truth_reference', mutations=mutations,
            regions=regions, cn_band=band, candidates=len(order), support_work=int(len(order)*catalog.major_cn[observed].sum()),
            preparation_seconds=prepared-started, assignment_api_seconds=scored-prepared,
            total_seconds=scored-started, peak_rss_bytes=int(rss if sys.platform == 'darwin' else rss*1024),
            output_policy='compact timings/cardinality only; no large assignment TSVs',
            counts=result.summary['status_counts'])
    except Exception as error:
        payload = dict(status='failed', mutations=mutations, regions=regions, cn_band=band,
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    path.write_text(json.dumps(_plain(payload), allow_nan=False)+'\n')


def run_scaling(outdir, config, *, sizes=(1000, 5000, 20000), region_counts=(2, 4, 8),
                cn_bands=('moderate', 'high'), timeout_seconds=600):
    """Explicit optional assignment-only panel; each cell runs in a fresh process."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    context, records = multiprocessing.get_context('spawn'), []
    for mutations in sizes:
        for regions in region_counts:
            for band in cn_bands:
                if band not in ('moderate', 'high'):
                    raise ValueError('Unknown CN scaling band')
                path = outdir / f'scaling_N{mutations}_R{regions}_{band}.json'
                if path.exists():
                    raise FileExistsError(f'Scaling output already exists: {path}')
                process = context.Process(target=_scaling_case,
                    args=(str(path), asdict(config), mutations, regions, band, 20261106))
                process.start()
                process.join(timeout_seconds)
                if process.is_alive():
                    process.terminate()
                    process.join()
                if not path.exists():
                    path.write_text(json.dumps(dict(status='failed', mutations=mutations, regions=regions,
                        cn_band=band, error_type='WorkerFailure', error='timeout or process exit', exitcode=process.exitcode))+'\n')
                records.append(json.loads(path.read_text()))
    report = dict(schema='clipp2.simulation-scaling.v1', status='complete' if all(row['status'] == 'complete' for row in records) else 'partial_failure',
        records=records, fits=0, assignment_passes=sum(row['status'] == 'complete' for row in records),
        memory_measurement='fresh-worker process peak RSS; includes preparation, Python results and native allocations')
    (outdir / 'scaling_summary.json').write_text(json.dumps(report, allow_nan=False)+'\n')
    return report
