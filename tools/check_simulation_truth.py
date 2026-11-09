"""Zero-fit checks of retained physical truth, acquisition, and input linkage."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.simulation.acquisition import acquire, validate_history_acquisition_link
from CliPP2.simulation.biology import History, generate_history, history_content_id


def _catalog(directory, name, frame, *, linked=True):
    metadata = dict(frame.attrs['metadata'])
    if not linked:
        metadata.pop('history_content_id', None)
    path = directory / (name + '.tsv')
    write_tumor_txt(path, frame, metadata)
    return load_original_catalog(path)


def _alias_history(history):
    """Swap two physically distinct mutations with equal m*phi in one segment."""
    lookup = {mutation: i for i, mutation in enumerate(history.mutation_ids)}
    a = next(lookup[m] for m in history.metadata['mutation_classes']['A'])
    b = next(lookup[m] for m in history.metadata['mutation_classes']['R']
             if history.segment_ids[lookup[m], 0] == history.segment_ids[a, 0])
    first, second = history.mutation_ids[a], history.mutation_ids[b]
    names = {first: second, second: first}
    value = history.to_dict()
    for name in ('segment_ids', 'phi', 'multiplicity', 'major_cn', 'minor_cn',
                 'mutant_copy_counts', 'mutation_branches', 'observable_truth_ids'):
        value[name][a], value[name][b] = value[name][b], value[name][a]
    for event in value['events']:
        if 'mutation_id' in event:
            event['mutation_id'] = names.get(event['mutation_id'], event['mutation_id'])
    for record in value['copies']:
        record['mutation_ids'] = [names.get(m, m) for m in record['mutation_ids']]
    return History.from_dict(value)


def run_truth_link_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history = generate_history('S07', 300, 60, 6, 4)
    identity = history_content_id(history)
    assert 'history_content_id' not in history.metadata
    assert history_content_id(History.from_dict(history.to_dict())) == identity
    frame = acquire(history, 900, .8, 100)
    catalog = _catalog(directory, 'linked', frame)
    acquisition = frame.attrs['acquisition']
    manifest = dict(history_content_id=identity)
    def state(h=history, a=acquisition, c=catalog, m=manifest):
        return validate_history_acquisition_link(h, a, c, history_manifest=m)['status']
    assert state() == 'verified_history_binding'
    for malformed in (None, [], 'not-an-object', True, 1.5):
        assert state(a=malformed) == 'missing_truth_evidence'
        if malformed is not None:  # None is the helper's explicit omitted-manifest option.
            assert state(m=malformed) == 'missing_truth_evidence'
    for name, value in [('true_purity', None), ('positive_carrier', None), ('true_probability', None)]:
        altered = dict(acquisition)
        altered.pop(name)
        assert state(a=altered) == 'missing_truth_evidence'
    assert state(m=dict(history_content_id='different')) == 'conflicting_truth'
    assert state(m={}) == 'missing_truth_evidence'
    assert state(a=dict(acquisition, history_id='another')) == 'conflicting_truth'
    carrier = acquisition['positive_carrier'].copy()
    carrier[0, 0] = not carrier[0, 0]
    assert state(a=dict(acquisition, positive_carrier=carrier)) == 'conflicting_truth'
    assert state(a=dict(acquisition, true_phi=np.zeros_like(history.phi))) == 'conflicting_truth'
    for index, perturbation in enumerate(({'purity_offset': -.1}, {'cn_error_fraction': .5},
            {'segment_error_fraction': .5}, {'alt_odds_ratio': 1.3}, {'overdispersion': .05},
            {'purity_offset': -.1, 'cn_error_fraction': .5, 'alt_odds_ratio': 1.3, 'overdispersion': .05})):
        technical = acquire(history, 900, .8, 100, perturbation=perturbation)
        supplied = _catalog(directory, f'perturbed-{index}', technical)
        assert state(a=technical.attrs['acquisition'], c=supplied) == 'verified_history_binding'
    legacy = dict(acquisition)
    legacy.pop('history_content_id')
    legacy_catalog = _catalog(directory, 'legacy', frame, linked=False)
    assert state(a=legacy, c=legacy_catalog, m={}) == 'legacy_consistent_unbound'
    # Same identifiers, seed, dimensions and valid events; different regional CCFs.
    order = [1, 0, 2, 3]
    collision = replace(history, clone_fractions=history.clone_fractions[:, order],
        phi=history.phi[:, order], multiplicity=history.multiplicity[:, order],
        major_cn=history.major_cn[:, order], minor_cn=history.minor_cn[:, order])
    assert np.any(history.phi != collision.phi) and collision.history_id == history.history_id
    assert state(h=collision, a=legacy, c=legacy_catalog, m={}) == 'conflicting_truth'
    aliased = generate_history('S04', 300, 120, 6, 4)
    alias = _alias_history(aliased)
    aliased_frame = acquire(aliased, 900, .8, 100)
    alias_frame = acquire(alias, 900, .8, 100)
    np.testing.assert_allclose(aliased_frame.attrs['acquisition']['true_probability'],
                              alias_frame.attrs['acquisition']['true_probability'], atol=1e-12, rtol=0)
    assert history_content_id(alias) != history_content_id(aliased)
    alias_legacy = dict(aliased_frame.attrs['acquisition'])
    alias_legacy.pop('history_content_id')
    alias_catalog = _catalog(directory, 'alias', aliased_frame, linked=False)
    assert state(h=alias, a=alias_legacy, c=alias_catalog, m={}) == 'legacy_consistent_unbound'
    assert state(h=alias, a=aliased_frame.attrs['acquisition'], c=alias_catalog, m={}) == 'conflicting_truth'
    _offline_truth_checks(directory / 'offline', history, frame, collision)
    return dict(status='passed', linked_history_verified=True, contradictory_legacy_truth_rejected=True,
        probability_alias_remains_unbound=True, technical_inputs_distinct_from_biological_truth=True,
        malformed_or_missing_truth_rejected=True, offline_has_zero_inference=True,
        fit_count=0, assignment_pass_count=0)


def _offline_truth_checks(root, history, frame, collision):
    """Exercise real evidence IO, while preventing every inference/generation route."""
    from contextlib import ExitStack
    from CliPP2.io.data import content_hash
    from CliPP2.provenance import prepare_run_manifest
    from CliPP2.simulation.reporting import reevaluate_cell
    root.mkdir()
    job = dict(cell_id='truth-link', phase='main', family=history.family, history_id='retained-history',
        history_seed=history.metadata['seed'], acquisition_seed=900,
        acquisition=dict(id='A4', purity=.8, depth=100), dimensions=dict(mutations=60, segments=6, regions=4))
    attempt = root / 'cells/truth-link/attempt-001'
    attempt.mkdir(parents=True)
    (attempt / 'job.json').write_text(json.dumps(job))
    catalog_frame = frame.copy()
    metadata = dict(frame.attrs['metadata'], tumor_id=job['cell_id'])
    write_tumor_txt(attempt / 'input_catalog.tsv', catalog_frame, metadata)
    catalog = load_original_catalog(attempt / 'input_catalog.tsv')
    acquisition = dict(frame.attrs['acquisition'], id='A4')
    from CliPP2.io.data import json_value
    (attempt / 'acquisition.json').write_text(json.dumps(json_value(acquisition)))
    truth_dir = root / 'histories/retained-history'
    history.write_truth(truth_dir)
    manifest = {key: job[key] for key in ('history_id', 'family', 'history_seed', 'dimensions')}
    manifest['history_content_id'] = history_content_id(history)
    (truth_dir / 'history_manifest.json').write_text(json.dumps(manifest))
    paired = dict(status='complete', original_mutation_ids=history.mutation_ids, region_ids=history.region_ids,
        run_manifest=prepare_run_manifest(catalog), arms={})
    configuration = attempt / 'configuration-1'
    configuration.mkdir()
    (configuration / 'paired_result.json').write_text(json.dumps(paired))
    cell = dict(job=job, attempt=str(attempt.relative_to(root)), status='complete')
    config = dict(reference_match_atol=.025, reference_distance='l2')
    forbidden = ('CliPP2.fit_execution.fit_prepared', 'CliPP2.assignment.assign_mutations',
        'CliPP2.kernel.model.RegionalModel', 'CliPP2.simulation.study._history',
        'CliPP2.simulation.study._acquire', 'CliPP2.simulation.study.assess_development')
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch(name, side_effect=AssertionError('Offline truth check executed ' + name)))
        evaluator = stack.enter_context(patch('CliPP2.simulation.evaluation.evaluate', return_value={}))
        good = reevaluate_cell(root, cell, config)
        assert good['truth_qualification'] == 'verified_history_binding'
        assert evaluator.call_args.kwargs['truth_qualification'] == 'verified_history_binding'
        # Legacy input/manifest links are absent; do not retrofit original proof.
        manifest.pop('history_content_id')
        (truth_dir / 'history_manifest.json').write_text(json.dumps(manifest))
        acquisition.pop('history_content_id')
        (attempt / 'acquisition.json').write_text(json.dumps(json_value(acquisition)))
        metadata.pop('history_content_id')
        write_tumor_txt(attempt / 'input_catalog.tsv', catalog_frame, metadata)
        catalog = load_original_catalog(attempt / 'input_catalog.tsv')
        paired['run_manifest'] = prepare_run_manifest(catalog)
        assert paired['run_manifest']['input_identity']['original_catalog_identity'] == content_hash(catalog.to_dict())
        (configuration / 'paired_result.json').write_text(json.dumps(paired))
        legacy = reevaluate_cell(root, cell, config)
        assert legacy['truth_qualification'] == 'legacy_consistent_unbound'
        evaluator.reset_mock()
        collision.write_truth(truth_dir)
        conflicting = reevaluate_cell(root, cell, config)
        assert conflicting['truth_qualification'] == 'conflicting_truth'
        evaluator.assert_not_called()
        history.write_truth(truth_dir)
        acquisition.pop('true_probability')
        (attempt / 'acquisition.json').write_text(json.dumps(json_value(acquisition)))
        missing = reevaluate_cell(root, cell, config)
        assert missing['truth_qualification'] == 'missing_truth_evidence'
        evaluator.assert_not_called()
        for path in (truth_dir / 'history_manifest.json', attempt / 'acquisition.json',
                     configuration / 'paired_result.json'):
            original = path.read_bytes()
            for malformed in (None, [], 'not-an-object', True, 1.5):
                path.write_text(json.dumps(malformed))
                unavailable = reevaluate_cell(root, cell, config)
                assert unavailable['status'] == 'unavailable_derived_evidence'
                assert unavailable['truth_qualification'] == 'missing_truth_evidence'
                evaluator.assert_not_called()
            path.write_bytes(original)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run_truth_link_checks(parser.parse_args().output), indent=2))
