"""Qualify physical copy histories before any biological accuracy experiments.

Run from the repository with python -m tools.check_simulation_biology.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.simulation.acquisition import acquire, validate_history_acquisition_link
from CliPP2.simulation.biology import FAMILIES, History, generate_history, history_content_id


def run_cohort_identity_checks(directory, settings_path):
    """Content binding tests without native fitting or changing semantic seeds."""
    from CliPP2.simulation.cohort import cohort_content_id, cohort_history, load_cohort, semantic_seed
    from CliPP2.simulation.study import _cohort, _verify_cohort_snapshot
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    original = json.loads(Path(settings_path).read_text())
    identity = cohort_content_id(original)
    path = directory / "settings.json"
    path.write_text(json.dumps(original, sort_keys=True, indent=1))
    base = load_cohort(path, expected_content_id=identity)
    with patch('CliPP2.simulation.cohort.load_cohort', wraps=load_cohort) as loader:
        worker_snapshot = _verify_cohort_snapshot(path, identity)
        assert all(_cohort(path, identity) is worker_snapshot for _ in range(4))
        assert loader.call_count == 1
    assert base.content_id == identity and worker_snapshot.content_id == identity
    try:
        worker_snapshot.settings['acquisitions']['A1']['mean_depth'] += 1
    except TypeError:
        pass
    else:
        raise AssertionError('Verified cohort settings are mutable')
    changed = deepcopy(original)
    source = next(row for row in changed["histories"]
                  if not np.array_equal(np.asarray(row["tip_fractions"])[:, 0], np.asarray(row["tip_fractions"])[:, 1]))
    source["tip_fractions"] = np.asarray(source["tip_fractions"])[:, [1, 0, 2, 3]].tolist()
    source["true_ccfs"] = {key: np.asarray(values)[[1, 0, 2, 3]].tolist() for key, values in source["true_ccfs"].items()}
    altered_id = cohort_content_id(changed)
    assert altered_id != identity
    path.write_text(json.dumps(changed))
    altered = _cohort(path, altered_id)
    assert altered.content_id == altered_id and altered is not base
    assert _cohort(path, identity) is worker_snapshot
    try:
        _verify_cohort_snapshot(path, identity)
    except ValueError:
        pass
    else:
        raise AssertionError("Fresh execution boundary accepted changed biological truth")
    one = cohort_history(base, source["history_id"])
    two = cohort_history(altered, source["history_id"])
    np.testing.assert_array_equal(two.phi, one.phi[:, [1, 0, 2, 3]])
    assert semantic_seed(base, "identity_test", 1) == semantic_seed(altered, "identity_test", 1)
    for kind in ("event", "acquisition", "mask", "list_order"):
        mutation = deepcopy(original)
        if kind == "event":
            mutation["variants"][0]["event_logs"][0]["identity_test_change"] = True
        elif kind == "acquisition":
            mutation["acquisitions"]["A1"]["mean_depth"] += 1
        elif kind == "mask":
            mutation["variants"][0]["observation_policy"] = "changed_mask"
        else:
            mutation["region_order"].reverse()
        assert cohort_content_id(mutation) != identity
    return dict(status="passed", formatting_and_key_order_stable=True, coherent_fraction_ccf_change_detected=True,
                event_acquisition_mask_and_list_changes_detected=True, same_path_cache_safe=True,
                worker_snapshot_immutable=True, fresh_boundary_detects_changed_content=True,
                semantic_seeds_unchanged=True, native_fit_count=0)


def _rows(history, group):
    index = {mutation: i for i, mutation in enumerate(history.mutation_ids)}
    return [index[mutation] for mutation in history.metadata['mutation_classes'].get(group, ())]


def run_sensitivity_layout_checks(directory):
    """Prepare every small extended layout and stop workers before inference."""
    from collections import defaultdict
    from CliPP2.io.data import content_hash
    from CliPP2.simulation.design import jobs, make_contract, seed_for
    from CliPP2.simulation.panels import sensitivity_jobs
    from CliPP2.simulation.study import _history, _prepare_histories, _run_cell

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = make_contract(mutations=24, segments=3, regions=4)
    original_contract = deepcopy(contract)
    main_before = list(jobs(contract, 'main', 1))
    planned = sensitivity_jobs(contract, extended=True)
    grouped = defaultdict(list)
    for job in planned:
        base = f"sensitivity-{job['family']}-H001"
        name, regions = job['acquisition']['id'], job['dimensions']['regions']
        assert job['cell_id'] == f'{base}-{name}'
        assert job['history_seed'] == seed_for(contract['master_seed'], 'sensitivity', job['family'], 0)
        assert job['acquisition_seed'] == seed_for(contract['master_seed'], 'sensitivity', job['family'], 0, 4)
        expected = base if regions == contract['dimensions']['regions'] else f'{base}-regions{regions}'
        assert job['history_id'] == expected
        if name.startswith('regions'):
            assert job['acquisition']['depth'] == (500 if name.endswith('per_region') else 2000/regions)
        grouped[job['history_id']].append(job)
    assert len(planned) == 120 and len(grouped) == 16
    assert contract == original_contract and list(jobs(contract, 'main', 1)) == main_before
    for job in main_before:
        assert job['history_id'] == f"main-{job['family']}-H001"
        assert job['cell_id'] == f"{job['history_id']}-{job['acquisition']['id']}"
        assert job['history_seed'] == seed_for(contract['master_seed'], 'main', job['family'], 0)
    # The base identity rule also applies when the contract's base is not R=4.
    two_region = sensitivity_jobs(dict(contract, dimensions=dict(contract['dimensions'], regions=2)), extended=True)
    assert all(job['history_id'] == f"sensitivity-{job['family']}-H001" for job in two_region
               if job['dimensions']['regions'] == 2)
    with patch('CliPP2.fit_execution.fit_regions', side_effect=AssertionError('layout check invoked fitting')) as fit, \
            patch('CliPP2.assignment.assign_mutations', side_effect=AssertionError('layout check invoked assignment')) as assign, \
            patch('CliPP2.kernel.backend.NativeBackend', side_effect=AssertionError('layout check opened native backend')) as backend:
        _prepare_histories(directory, planned)
        prepared = {}
        for history_id, members in grouped.items():
            assert len({content_hash({key: job[key] for key in ('family', 'history_seed', 'dimensions')})
                        for job in members}) == 1
            generated = _history(members[0])
            saved = directory / 'histories' / history_id
            manifest = json.loads((saved / 'history_manifest.json').read_text())
            restored = History.from_dict(json.loads((saved / 'history.json').read_text()))
            assert manifest['history_content_id'] == history_content_id(generated) == history_content_id(restored)
            prepared[history_id] = generated
            modes = [job for job in members if job['acquisition']['id'].startswith('regions')]
            assert len(modes) == 2 and len({job['history_id'] for job in modes}) == 1
            assert history_content_id(_history(modes[0])) == history_content_id(_history(modes[1]))
        assert len({history_content_id(history) for history in prepared.values()}) == 16
        for family in ('S02', 'S04', 'S06', 'S12'):
            base = f'sensitivity-{family}-H001'
            large = prepared[base+'-regions8']
            for regions in (1, 2, 4):
                small = prepared[base if regions == 4 else f'{base}-regions{regions}']
                assert small.events == large.events
                np.testing.assert_array_equal(small.clone_fractions, large.clone_fractions[:, :regions])
                np.testing.assert_array_equal(small.phi, large.phi[:, :regions])
        # Exercise live manifest/acquisition/input linkage for every layout; the
        # deliberate sentinel is reached before any paired inference can run.
        with patch('CliPP2.simulation.experiment.run_paired_experiment', side_effect=RuntimeError('layout_preinference_stop')) as boundary:
            for members in grouped.values():
                completion = _run_cell(dict(root=str(directory), job=members[0], selected=dict(contributor=.95, posterior=.95)))
                assert completion['execution_phase'] == 'paired_inference'
                assert completion['error'] == 'layout_preinference_stop'
            assert boundary.call_count == 16
        original_manifests = {key: (directory / 'histories' / key / 'history_manifest.json').read_bytes() for key in grouped}
        _prepare_histories(directory, reversed(planned))
        assert all((directory / 'histories' / key / 'history_manifest.json').read_bytes() == value
                   for key, value in original_manifests.items())
        base = planned[0]
        changes = [dict(family='S01'), dict(history_seed=base['history_seed']+1),
                   dict(dimensions=dict(base['dimensions'], regions=1)),
                   dict(cohort_settings='unused_source.json', cohort_content_id='different_cohort'),
                   dict(source_history_id='different_source')]
        incompatible = [(base, dict(base, **change)) for change in changes]
        cohort = dict(base, cohort_settings='unused_source.json', cohort_content_id='cohort_one', source_history_id='source_one')
        incompatible += [(cohort, dict(cohort, cohort_content_id='cohort_two')),
                         (cohort, dict(cohort, source_history_id='source_two'))]
        for index, pair in enumerate(incompatible):
            destination = directory / f'incompatible-{index}'
            with patch('CliPP2.simulation.study._history', side_effect=AssertionError('collision generated history')), \
                    patch('CliPP2.simulation.study._cohort', side_effect=AssertionError('collision loaded cohort')):
                try:
                    _prepare_histories(destination, pair)
                except ValueError as error:
                    assert 'incompatible biological layout or source identity' in str(error)
                else:
                    raise AssertionError('Incompatible repeated history identity was skipped')
            assert not destination.exists()
        assert fit.call_count == assign.call_count == backend.call_count == 0
    return dict(status='passed', jobs=120, compatible_layout_histories=16,
        generated_prepared_content_matches=True, worker_linkage_reaches_preinference=16,
        shared_depth_mode_history=True, base_ids_cells_seeds_and_depths_unchanged=True,
        physical_events_and_region_prefix_preserved=True, conflicting_ids_rejected_before_generation=True,
        main_roster_unchanged=True, native_fit_count=0, assignment_pass_count=0)


def run_biology_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tested = 0
    for family in FAMILIES:
        for seed in (300, 301, 302):
            history = generate_history(family, seed, 120, 6, 4)
            history.validate()
            acquisition = acquire(history, 900, .8, 100)
            path = directory / f'{family}_{seed}.tsv'
            write_tumor_txt(path, acquisition, acquisition.attrs['metadata'])
            catalog = load_original_catalog(path)
            binding = validate_history_acquisition_link(history, acquisition.attrs['acquisition'], catalog,
                history_manifest=dict(history_content_id=history_content_id(history)))
            assert binding['status'] == 'verified_history_binding', binding
            np.testing.assert_array_equal(catalog.major_cn, history.major_cn)
            np.testing.assert_array_equal(catalog.minor_cn, history.minor_cn)
            np.testing.assert_array_equal(catalog.count_observed, acquisition.attrs['acquisition']['count_observed'])
            np.testing.assert_array_equal(catalog.total_counts, acquisition.attrs['acquisition']['depth'])
            np.testing.assert_array_equal(catalog.alt_counts, acquisition.attrs['acquisition']['alt_counts'])
            assert catalog.mapping_genuine.all() and catalog.normal_cn_supplied.all()
            restored = History.from_dict(json.loads(json.dumps(history.to_dict(), allow_nan=False)))
            np.testing.assert_array_equal(restored.multiplicity, history.multiplicity)
            assert restored.events == history.events
            tested += 1
    null = generate_history('S01', 300, 120, 6)
    assert np.all(null.major_cn == 1) and np.all(null.minor_cn == 1) and np.all(null.multiplicity == 1)
    valid = generate_history('S02', 300, 120, 6)
    source_ids = set(valid.metadata['declared_source_ids'])
    source = [i for i, mutation in enumerate(valid.mutation_ids) if mutation in source_ids]
    remainder = [i for i, mutation in enumerate(valid.mutation_ids) if mutation not in source_ids]
    assert np.all(valid.multiplicity[source] == 2) and np.all(valid.multiplicity[remainder] == 1)
    assert np.all(valid.major_cn > 2), 'positive control must have an active valid cap'
    counter = generate_history('S03', 300, 120, 6)
    assert np.all(counter.phi[_rows(counter, 'E')] == 1) and np.all(counter.phi[_rows(counter, 'R')] == 1)
    assert np.all(counter.multiplicity[_rows(counter, 'E')] == 2) and np.all(counter.multiplicity[_rows(counter, 'R')] == 1)
    shared = set(counter.segment_ids[_rows(counter, 'E'), 0]) & set(counter.segment_ids[_rows(counter, 'R'), 0])
    assert shared, 'clonal cap counterexample must share actual segments'
    for family in ('S04', 'S05', 'S12'):
        exact = generate_history(family, 300, 120, 6)
        a, reference = _rows(exact, 'A'), _rows(exact, 'R')
        np.testing.assert_allclose(exact.phi[a], .5, atol=1e-14)
        assert np.all(exact.multiplicity[a] == 2) and np.all(exact.multiplicity[reference] == 1)
        np.testing.assert_allclose(exact.phi[a]*exact.multiplicity[a], 1., atol=1e-14)
        broken = generate_history(family, 301, 120, 6)
        assert np.any(np.abs(broken.phi[_rows(broken, 'A')]*2-1) > .1)
    distinguishing = generate_history('S12', 302, 120, 6)
    masked = acquire(distinguishing, 23, .8, 100, perturbation={'mask_distinguishing_region': True})
    assert not masked.attrs['acquisition']['count_observed'][:, 3].any()
    np.testing.assert_allclose(distinguishing.phi[_rows(distinguishing, 'A'), :3]*2, 1., atol=1e-14)
    assert np.all(distinguishing.phi[_rows(distinguishing, 'A'), 3] > .6)
    for seed, expected in ((300, 1), (301, 3)):
        wgd = generate_history('S06', seed, 120, 6)
        assert sum(event['event_type'] == 'wgd' for event in wgd.events) == expected
        assert np.all(wgd.major_cn == 2) and np.all(wgd.minor_cn == 2)
        assert {1., 2.} <= set(wgd.multiplicity.flat)
    local = generate_history('S07', 300, 120, 6)
    assert np.all(local.phi[_rows(local, 'A'), 0] == 1)
    assert np.all(local.phi[_rows(local, 'A'), 1:] < 1)
    near = generate_history('S08', 300, 120, 6)
    assert not near.metadata['mutation_classes']['E'] and not near.metadata['mutation_classes']['R']
    assert np.all((near.phi[_rows(near, 'A')] > .96) & (near.phi[_rows(near, 'A')] < 1))
    densities = [len(generate_history('S09', seed, 120, 6).metadata['declared_source_ids']) for seed in (300, 301, 302)]
    assert densities == sorted(set(densities))
    lost = generate_history('S10', 300, 120, 6)
    loss_rows = _rows(lost, 'L')
    assert np.all(lost.phi[loss_rows] == 0) and np.isnan(lost.multiplicity[loss_rows]).all()
    assert np.all(lost.mutant_copy_counts[loss_rows, lost.clone_ids.index('unsampled')] == 1)
    assert {event['event_type'] for event in lost.events} >= {'copy_neutral_loh', 'deletion'}
    assert np.any(lost.major_cn == 11)
    edge = acquire(lost, 30, 1., 100)
    audit = edge.attrs['acquisition']
    assert (~audit['original_eligible']).any() and not audit['count_observed'][0].any()
    assert audit['latent_depth'][1, 0] == 0 and np.all(audit['latent_alt_counts'][loss_rows] == 0)
    assert audit['biological_truth_available'].all() and audit['input_loss'].any()
    incomplete = generate_history('S11', 300, 120, 6)
    assert not np.any(np.all(incomplete.phi == 1, axis=1))
    assert {'remove_ancestor', 'duplicate_ancestor', 'distort_ancestor', 'K1'} <= set(incomplete.metadata['reference_panels'])
    # Adding regions changes neither physical events nor existing regional mixtures.
    for family in FAMILIES:
        large = generate_history(family, 302, 60, 4, 8)
        for regions in (1, 2, 4):
            small = generate_history(family, 302, 60, 4, regions)
            assert small.events == large.events
            np.testing.assert_array_equal(small.clone_fractions, large.clone_fractions[:, :regions])
            np.testing.assert_array_equal(small.phi, large.phi[:, :regions])
    base = acquire(valid, 99, .8, 100)
    for perturbation in ({'purity_offset': -.1}, {'cn_error_fraction': .15}, {'segment_error_fraction': 1.},
                         {'cn_error_fraction': .5, 'segment_error_fraction': 1.}):
        changed = acquire(valid, 99, .8, 100, perturbation=perturbation)
        for key in ('latent_depth', 'latent_alt_counts', 'true_probability'):
            np.testing.assert_array_equal(base.attrs['acquisition'][key], changed.attrs['acquisition'][key])
        write_tumor_txt(directory / 'perturbed.tsv', changed, changed.attrs['metadata'])
        load_original_catalog(directory / 'perturbed.tsv')
    dispersed = acquire(lost, 30, 1., 100, perturbation={'overdispersion': .05, 'alt_odds_ratio': 1.25})
    values = dispersed.attrs['acquisition']
    assert np.all(values['sampled_probability'][values['true_probability'] == 0] == 0)
    assert np.all(values['sampled_probability'][values['true_probability'] == 1] == 1)
    lost.write_truth(directory / 'serialized_truth')
    assert len(pd.read_csv(directory / 'serialized_truth/truth_pairs.tsv', sep='\t')) == 120*4
    assert History.from_dict(json.loads((directory / 'serialized_truth/history.json').read_text())).validate()
    bad = valid.to_dict()
    bad['multiplicity'][0][0] = 1
    try:
        History.from_dict(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted derived multiplicity was accepted')
    if __package__:
        from .check_simulation_truth import run_truth_link_checks
    else:
        from check_simulation_truth import run_truth_link_checks
    truth_binding = run_truth_link_checks(directory / 'truth_binding')
    sensitivity_layouts = run_sensitivity_layout_checks(directory / 'sensitivity_layouts')
    return dict(status='passed', family_histories=tested, physical_event_replay=True,
        integer_single_state_reduction=True, valid_cap_and_clonal_counterexample=True,
        exact_broken_and_masked_aliases=True, mutation_loss_separate_from_input_loss=True,
        region_prefix_pairing=True, input_perturbations_reuse_reads=True,
        canonical_roundtrips=True, truth_binding=truth_binding, sensitivity_layouts=sensitivity_layouts,
        elapsed_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cohort-settings', type=Path, default=Path('CliPP2_phylogeny_cohort_settings.json'))
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_biology_checks(directory)
            result['cohort_identity'] = run_cohort_identity_checks(Path(directory) / 'identity', args.cohort_settings)
    else:
        result = run_biology_checks(args.output)
        result['cohort_identity'] = run_cohort_identity_checks(args.output / 'identity', args.cohort_settings)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
