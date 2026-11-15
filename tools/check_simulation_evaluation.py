#!/usr/bin/env python3
"""Deterministic scientific evaluator contracts; no inferred biological calibration."""
from __future__ import annotations

import argparse
import json
import itertools
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from CliPP2.simulation.evaluation import (aggregate, candidate_correspondence, cap_audit,
    information_diagnostics, mapped_probabilities, partition_scores, probability_scores,
    recovery_reach, reference_center_diagnostics, transition, json_safe, evaluate,
    evaluation_definition, target_reference_availability, _qualified_matching)
from CliPP2.io.data import content_hash
from CliPP2.simulation.evaluation import log_probability_mass


def _emission(alt, depth, major, phi, upper):
    p = np.arange(1, upper+1)*phi/major
    logs = (gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)
            + xlogy(alt, p)+xlog1py(depth-alt, -p))
    return float(np.exp(logsumexp(logs)-np.log(upper))), np.exp(logs), np.exp(logs-logsumexp(logs))


def _current_fixture():
    """Declared current decisions isolate evaluator endpoints from native fitting."""
    ids = ('t_correct', 't_wrong', 't_unmapped', 't_unresolved', 't_clonal', 't_origin2',
           'f_a0', 'f_a1', 'f_a2', 'f_a3', 'f_b_wrong', 'f_b', 'f_mix_a', 'f_mix_b', 'no_info')
    classes = ('a', 'a', 'a', 'a', 'trunk', 'b', 'a', 'a', 'a', 'a', 'b', 'b', 'a', 'b', 'a')
    truth_centers = {'a': [.3, .4], 'b': [.6, .7], 'trunk': [1., 1.]}
    phi = np.array([truth_centers[key] for key in classes])
    observed = np.ones(phi.shape, bool)
    observed[0, 1] = False
    observed[-1] = False
    history = SimpleNamespace(mutation_ids=ids, region_ids=('r1', 'r2'), phi=phi,
        observable_truth_ids=classes, multiplicity=np.ones(phi.shape), major_cn=np.ones(phi.shape),
        minor_cn=np.ones(phi.shape), segment_ids=np.full(phi.shape, 's'),
        metadata=dict(family='S01', history_id='declared-evaluation-fixture'))
    catalog = SimpleNamespace(mutation_ids=ids, region_ids=history.region_ids, eligible=np.ones(phi.shape, bool),
        count_observed=observed, total_counts=np.full(phi.shape, 100), segment_ids=history.segment_ids,
        major_cn=history.major_cn, minor_cn=history.minor_cn, purity=np.ones(phi.shape), max_major_cn=None,
        mapping_genuine=np.ones(phi.shape, bool))
    centers = [[1., 1.], [.3, .4], [.6, .7], [.45, .55]]
    combined = dict(zip(ids, [0]*6+[1]*5+[2, 3, 3, 0]))
    first = dict(mutation_ids=ids[:-1], labels=list(combined.values())[:-1], centers=centers,
                 multiplicities=np.ones((len(ids)-1, 2)).tolist())
    second = dict(mutation_ids=ids[5:-1], labels=[0]+[1]*5+[2, 3, 3], centers=centers,
                  multiplicities=np.ones((len(ids)-6, 2)).tolist())
    records = []
    for i, mutation in enumerate(ids):
        label = (1, 2, 3, 1, 0, 1)[i] if i < 6 else combined[mutation]
        informative = mutation != 'no_info'
        probabilities = np.full(4, .2 if informative else .25)
        if informative:
            probabilities[label] = .4
        records.append(dict(mutation_id=mutation, final_assignment_label=label if informative else None,
            map_cluster_label=label, map_label=label, map_set=[label] if informative else list(range(4)),
            map_tied_labels=[label] if informative else list(range(4)), map_unique=informative,
            evidence_status='observed' if informative else 'prior_only', status='observed' if informative else 'prior_only',
            probabilities=probabilities.tolist(), log_scores=np.log(probabilities).tolist(),
            best_log_probability=float(np.log(.4 if informative else .25)), log_competitor_ratio=float(np.log(1.5)),
            has_finite_alternative=True, posterior_defined=True, assignment_method='marginal', score_defined=True,
            selection_reason='unique_maximum' if informative else 'prior_only',
            target_reason='stage1_reference_recheck' if i < 5 else 'stage2_recheck'))
    arm = dict(status='complete', stage_two=second, combined_labels=combined,
        reference=dict(labels=[0, 1, 2, 3], clonal_label=0, centers=centers,
            source_informative_count=[[0, 1]]*4, source_informative_depth=[[0, 100]]*4,
            source_mean_profile_deviance=[[None, np.inf]]*4, source_maximum_profile_deviance=[[None, np.inf]]*4,
            diagnostic_availability=[['not_applicable', 'available']]*4),
        targets=ids, non_targets=(), assignment=dict(schema='clipp2.assignment.result.v9', protocol_version='2.4',
            assignment_method='marginal', assignments=records, multiplicities=[]))
    from CliPP2.config import STAGE_THREE_REFERENCE_POLICY
    result = dict(schema='clipp2.paired-experiment.v6', protocol_version='2.4', run_manifest_id='evaluation-fixture',
        reference_policy=STAGE_THREE_REFERENCE_POLICY, stage_one=first, g0_ids=ids[:5], remainder_ids=ids[5:-1],
        arms={'B': deepcopy(arm), 'C': arm}, config=dict(fit=dict(max_major_cn=None), stage_three_assignment_method='marginal'))
    return history, catalog, result


def _all_target_checks():
    h, catalog, paired = _current_fixture()
    value = evaluate(h, catalog, paired, truth_qualification='verified_history_binding')
    assert value['schema'] == 'clipp2.simulation.evaluation.v6'
    assert value['evaluation_definition'] == evaluation_definition()
    metrics = {(r['name'], r['view']): r for r in value['metrics'] if r['arm'] == 'C3' and r['stratum'] == 'all'}
    assert metrics['data_supported_map_coverage', '']['denominator'] == 14
    assert metrics['data_supported_map_coverage', '']['value'] == 1
    assert metrics['data_supported_map_coverage', 'unchanged_F']['denominator'] == 0
    assert metrics['data_supported_map_coverage', 'stage2_recheck']['denominator'] == 9
    assert not any('near_clonal' in r['name'] or 'false_merge' in r['name'] for r in value['metrics'])
    rows = [r for r in value['mutations'] if r['arm'] == 'C3']
    assert all(r['role'] == 'target' for r in rows)
    assert all(r['posterior_defined'] and r['map_label'] in r['map_set'] for r in rows)
    from check_support import expect_value_error
    invalid = deepcopy(paired)
    invalid['arms']['C']['non_targets'] = ('t_correct',)
    expect_value_error(lambda: evaluate(h, catalog, invalid))
    from CliPP2.assignment import POSTERIOR_FIELDS
    method_ids = []
    for method in ('marginal', 'profile'):
        current = deepcopy(paired)
        current['config']['stage_three_assignment_method'] = method
        for arm in current['arms'].values():
            arm['assignment']['assignment_method'] = method
            for row in arm['assignment']['assignments']:
                row['assignment_method'] = method
                if method == 'profile':
                    row.update(dict.fromkeys(POSTERIOR_FIELDS))
                    row['posterior_defined'] = False
        assessed = evaluate(h, catalog, current, truth_qualification='verified_history_binding')
        assert assessed['assignment_method'] == method
        method_ids.append(assessed['evaluation_definition_id'])
        if method == 'profile':
            assert assessed['calibration'] == []
            scores = [r for r in assessed['metrics'] if r['name'] in ('binary_brier', 'binary_log_loss', 'multiclass_log_loss')]
            assert scores and all(r['value'] is None and r['status'] == 'undefined_for_profile' for r in scores)
            before = {(r['arm'], r['mutation_id']): r['final_assignment_label'] for r in value['mutations']}
            assert all(r['final_assignment_label'] == before[r['arm'], r['mutation_id']] for r in assessed['mutations'])
            current['arms']['C']['assignment']['assignments'][0]['probabilities'] = [.25]*4
            expect_value_error(lambda: evaluate(h, catalog, current))
    assert method_ids[0] != method_ids[1]
    return dict(F_empty=True, every_fitted_member_reassessed=True, method_specific_posteriors=True)


def _planned_population_checks():
    definition = evaluation_definition()
    roster = [dict(cell_id=f'h{i}-A1', family='S01', history_id=f'h{i}', acquisition=dict(id='A1')) for i in range(2)]

    def cell(job, score, *, failure=False, evaluated=True, truth='verified_history_binding', denominator=10):
        metrics = []
        for arm in ('B3', 'C3'):
            metrics.append(dict(name='binary_map_correct_per_original_evaluable', arm=arm, view='', stratum='all',
                value=score, numerator=None if score is None else score*denominator if denominator else 0,
                denominator=denominator, truth_required=not failure,
                status='known_no_decisions' if failure else 'unknown_evidence' if score is None else 'available'))
            for name, value in (('analysis_available', int(not failure)), ('analysis_failure', int(failure)),
                    ('execution_available', 1), ('execution_failure', int(failure)),
                    ('evaluation_available', int(evaluated)), ('scientifically_applicable', int(not failure))):
                metrics.append(dict(name=name, arm=arm, view='all_planned_cells', stratum='all', value=value,
                                    numerator=value, denominator=1, truth_required=False))
        return dict(schema='clipp2.simulation.evaluation.v6', evaluation_definition=definition,
            evaluation_definition_id=content_hash(definition), truth_qualification=truth,
            cell_id=job['cell_id'], family=job['family'], history_id=job['history_id'], acquisition_id=job['acquisition']['id'],
            original_evaluable_count=denominator, status='execution_failed' if failure else 'complete', metrics=metrics,
            applicability=[dict(arm=arm, status='failed', decision_population=dict(
                scope='whole_original_evaluable', coverage='complete', evaluable_count=denominator,
                accepted_count=0, evidence='failure_before_stage_one')) for arm in ('B3', 'C3')] if failure else [])

    def summary(records, planned=roster, complete=True, families=('S01',)):
        return aggregate(records, planned_roster=planned, roster_complete=complete,
                         families=families, bootstrap_replicates=20, seed=4)

    def endpoint(report, name='binary_map_correct_per_original_evaluable', arm='C3'):
        return next(row for row in report['metrics'] if row['name'] == name and row['arm'] == arm)

    success = cell(roster[0], 1.)
    failed = cell(roster[1], 0., failure=True, evaluated=False, truth='missing_truth_evidence')
    combined = summary([success, failed])
    assert endpoint(combined)['planned_population_estimate'] == .5
    assert endpoint(combined, 'analysis_available')['planned_population_estimate'] == .5
    assert endpoint(combined, 'execution_failure')['planned_population_estimate'] == .5
    assert endpoint(combined, 'evaluation_available')['planned_population_estimate'] == .5
    assert sum(row['weight'] for row in combined['planned_weights']) == 1
    unknown = cell(roster[1], None, evaluated=False)
    missing = summary([success, unknown])
    result = endpoint(missing)
    assert result['planned_population_estimate'] is None and result['equal_family_mean'] is None
    assert result['planned_population_status'] == 'incomplete_evidence' and result['available_case_estimate'] == 1
    assert result['lower_95'] is None and result['upper_95'] is None
    assert result['missingness'][0]['history_id'] == 'h1' and result['missingness'][0]['planned_weight'] == .5
    # A known failure without a real original denominator cannot define a rate.
    no_denominator = cell(roster[1], None, failure=True, evaluated=False, denominator=None)
    assert endpoint(summary([success, no_denominator]))['planned_population_estimate'] is None
    assert endpoint(summary([success]))['planned_population_status'] == 'incomplete_evidence'
    acquisition_roster = [roster[0], dict(roster[0], cell_id='h0-A2', acquisition=dict(id='A2')), roster[1]]
    missing_acquisition = summary([success, failed], planned=acquisition_roster)
    assert endpoint(missing_acquisition)['planned_population_estimate'] is None
    assert [row['weight'] for row in missing_acquisition['planned_weights']] == [.25, .25, .5]
    acquisition_failure = cell(acquisition_roster[1], 0., failure=True, evaluated=False)
    assert endpoint(summary([success, acquisition_failure, failed], planned=acquisition_roster))['planned_population_estimate'] == .25
    other_family = dict(roster[1], family='S02')
    missing_family = summary([success], planned=[roster[0], other_family], families=('S01', 'S02'))
    assert endpoint(missing_family)['planned_population_estimate'] is None
    assert endpoint(missing_family)['available_case_estimate'] == 1
    assert endpoint(missing_family)['unavailable_families'] == ['S02']
    partial = deepcopy(failed)
    for row in partial['metrics']:
        if row['arm'] == 'B3':
            source = next(item for item in success['metrics'] if item['name'] == row['name'] and item['arm'] == 'B3')
            row.update(source)
    partial['truth_qualification'] = 'verified_history_binding'
    partial_summary = summary([success, partial])
    assert endpoint(partial_summary, arm='B3')['planned_population_estimate'] == 1
    assert endpoint(partial_summary, arm='C3')['planned_population_estimate'] == .5
    unavailable = summary([cell(job, None, evaluated=False, truth='missing_truth_evidence') for job in roster])
    assert unavailable['schema'] == 'clipp2.simulation.aggregate.v3'
    assert endpoint(unavailable)['planned_population_estimate'] is None
    assert endpoint(unavailable, 'evaluation_available')['planned_population_estimate'] == 0
    incomplete = summary([success, failed], complete=False)
    assert endpoint(incomplete)['planned_population_status'] == 'incomplete_planned_roster'
    assert endpoint(incomplete)['available_case_estimate'] == .5
    unbound = cell(roster[1], 1., truth='missing_truth_evidence')
    qualified = endpoint(summary([success, unbound]))
    assert qualified['planned_population_estimate'] is None and qualified['available_case_estimate'] == 1
    assert qualified['missingness'][0]['reason'] == 'missing_truth_evidence'
    spoofed = deepcopy(unbound)
    for row in spoofed['metrics']:
        row['truth_required'] = False
    assert endpoint(summary([success, spoofed]))['planned_population_estimate'] is None
    no_count = deepcopy(failed)
    no_count['original_evaluable_count'] = None
    assert endpoint(summary([success, no_count]))['planned_population_estimate'] is None
    unjustified_zero = deepcopy(failed)
    unjustified_zero['status'] = 'complete'
    for state in unjustified_zero['applicability']:
        state['status'] = 'complete'
    assert endpoint(summary([success, unjustified_zero]))['planned_population_estimate'] is None
    old_tag = deepcopy(failed)
    for state in old_tag['applicability']:
        state.pop('decision_population')
    for qualification in ('missing_truth_evidence', 'verified_history_binding'):
        old_tag['truth_qualification'] = qualification
        assert endpoint(summary([success, old_tag]))['planned_population_estimate'] is None
    return dict(success_plus_known_failure_retains_half_weight=True, unknown_evidence_not_zero=True,
        missing_acquisition_history_family_explicit=True, partial_arms_preserved=True,
        unavailable_current_definition_preserved=True, incomplete_roster_not_invented=True,
        verified_primary_excludes_unbound_truth=True)


def run_checks():
    full, masses, probabilities = _emission(2, 3, 4, 1., 4)
    cap, _, _ = _emission(2, 3, 4, 1., 2)
    np.testing.assert_allclose(masses, [.140625, .375, .421875, 0.], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose([full, cap, probabilities[2]], [15/64, 33/128, .45], rtol=1e-14)
    a, ma, _ = _emission(1, 2, 2, 1., 2)
    b, mb, _ = _emission(1, 2, 2, .6, 2)
    np.testing.assert_allclose([a/(a+b), b/(a+b)], [5/14, 9/14], rtol=1e-14)
    assert a < b and ma.max() > mb.max()
    # Different acquisition branches with equal sampled vectors are one observable class.
    assert partition_scores(['same', 'same', 'other'], [2, 2, 7])['ari'] == 1.
    assert partition_scores([], [])['pair_precision'] is None
    assert partition_scores(['a', 'b'], [0, 0])['pair_precision'] == 0.
    assert partition_scores(['a', 'b'], [0, 1])['pair_precision'] is None
    mapping, audit = candidate_correspondence((0, 1, 2, 3), 0,
        {0: ('a',), 1: ('a', 'b', 'c', 'd', 'e'), 2: ('f', 'g'), 3: ('b',)},
        {'a': 'trunk', 'b': 'other', 'c': 'other', 'd': 'other', 'e': 'other', 'f': 'x', 'g': 'y'}, 'trunk')
    assert mapping == {0: 'trunk', 1: 'other', 2: None, 3: 'other'}
    assert audit[2]['source_purity'] == .5
    # Pure source overlap cannot establish correct geometric center recovery.
    geometric = reference_center_diagnostics((1,), [[.9, .9]], {'truth': [.4, .4]})
    assert geometric['recovered_count'] is None and geometric['reference_recall'] is None
    assert geometric['status'] == 'unassessed_no_declared_tolerance'
    np.testing.assert_allclose(geometric['mean_one_to_one_l2_error'], np.sqrt(.5))
    rejected = reference_center_diagnostics((1,), [[.9, .9]], {'truth': [.4, .4]}, match_atol=.05)
    assert rejected['reference_recall'] == 0
    one = reference_center_diagnostics((7,), [[.5, .5]], {'a': [.4, .4], 'b': [.6, .6]}, match_atol=.15)
    assert one['recovered_count'] == 1 and one['reference_recall'] == .5
    # Maximum qualified cardinality is evaluated independently of overlap labels.
    two = reference_center_diagnostics((7, 8), [[.5, .5], [.4, .4]],
                                       {'a': [.4, .4], 'b': [.6, .6]}, match_atol=.15)
    assert two['recovered_count'] == 2 and two['reference_recall'] == 1
    # Invalid forced pairs cannot influence the valid-distance objective.
    repaired = reference_center_diagnostics((0, 1), [[.31], [.28]], {'a': [.30], 'b': [.90]}, match_atol=.025)
    assert repaired['qualified_pairs'][0]['reference_label'] == 0
    np.testing.assert_allclose(repaired['qualified_pairs'][0]['l2_error'], .01)
    l2 = reference_center_diagnostics((0,), [[.1, .1]], {'a': [0., 0.]}, match_atol=.11)
    legacy = reference_center_diagnostics((0,), [[.1, .1]], {'a': [0., 0.]}, match_atol=.11, distance='linf')
    assert l2['recovered_count'] == 0 and legacy['recovered_count'] == 1
    tied = reference_center_diagnostics((8, 2), [[.5], [.5]], {'b': [.5], 'a': [.5]}, match_atol=0)
    assert tied['matching_ambiguous'] and tied['qualified_pairs'][0]['reference_label'] == 2
    # Exact binary-rational objective and lexical tie-break against enumeration.
    rng = np.random.default_rng(251)
    for mode in ('binary_grid', 'decimal', 'floating'):
        for _ in range(40):
            distances = rng.integers(0, 8, size=(3, 3)).astype(float)/(8 if mode == 'binary_grid' else 10)
            if mode == 'floating':
                distances = rng.random((3, 3))
            allowed = rng.random((3, 3)) > .4
            pairs, alternatives = _qualified_matching(distances, allowed)
            candidates = []
            for choices in itertools.product((0, 1, 2, None), repeat=3):
                used = [j for j in choices if j is not None]
                if len(used) != len(set(used)) or any(j is not None and not allowed[i, j] for i, j in enumerate(choices)):
                    continue
                cost = sum((Fraction.from_float(float(distances[i, j])) for i, j in enumerate(choices)
                            if j is not None), Fraction())
                candidates.append(((-len(used), cost), tuple(3 if j is None else j for j in choices)))
            optimum = min(key for key, _ in candidates)
            optimal_choices = sorted(choices for key, choices in candidates if key == optimum)
            actual_choices = tuple(dict(pairs).get(i, 3) for i in range(3))
            assert actual_choices == optimal_choices[0]
            assert bool(alternatives) == (len(optimal_choices) > 1)
    # These physical one-region vectors broke rounded-float subproblem matching.
    decimal = reference_center_diagnostics((0, 1, 2), [[.8], [.9], [.4]],
        dict(enumerate([[.7], [.3], [.1], [.1]])), match_atol=1.)
    assert decimal['recovered_count'] == 3
    exact = reference_center_diagnostics((0, 1), [[.9], [.3]], {0: [0.], 1: [.3]}, match_atol=1.)
    assert [(row['truth_class'], row['reference_label']) for row in exact['qualified_pairs']] == [(0, 0), (1, 1)]
    reference = dict(labels=[0], centers=[[.3, .4]], regional_quality=[[True, False]])
    absent = target_reference_availability([.8, .9], [True, False], reference, match_atol=.1)
    assert absent['geometric_reference_available'] is False and absent['usable_reference_available'] is False
    no_ref = target_reference_availability([.3, .4], [True, True], None, match_atol=.1)
    assert no_ref['geometric_reference_available'] is None and no_ref['usable_availability_status'] == 'unavailable_reference'
    # Full-vector geometry must not collapse to observed coordinates.
    masked_distance = target_reference_availability([.3, .9], [True, False], reference, match_atol=.1)
    assert masked_distance['geometric_reference_available'] is False
    q = mapped_probabilities([.1, .2, .3, .4], (0, 1, 2, 3), mapping, ('trunk', 'other', 'absent'))
    np.testing.assert_allclose([q['other'], q['__unmapped__']], [.6, .3])
    assert np.isposinf(probability_scores(q, None, truth_class='absent')['log_loss'])
    assert log_probability_mass([0., -1000.], [1]) == -1000.
    assert probability_scores(1., False, true_log_probability=-1000.)['log_loss'] == 1000.
    # Exact aliases survive arbitrary depth and vanish only in distinguishing regions.
    phi = np.array([[1., 1.], [.5, .65]])
    multiplicity = np.array([[1., 1.], [2., 2.]])
    args = (phi, multiplicity, np.full((2, 2), 2), np.ones((2, 2)), np.ones((2, 2)),
            np.full((2, 2), 2), np.full((2, 2), 100), np.ones((2, 2), bool), ('clonal', 'sub'))
    distinct = information_diagnostics(*args)
    assert not distinct[0]['exact_alias'] and distinct[0]['expected_kl'] > 0
    masked_args = list(args)
    masked_args[7] = np.array([[1, 0], [1, 0]], bool)
    masked = information_diagnostics(*masked_args)
    assert all(row['exact_alias'] for row in masked)
    # Fixed contributor IDs cannot silently discard a contributor with zero carriers.
    h = SimpleNamespace(mutation_ids=('source', 'target'), region_ids=('r',),
        phi=np.array([[0.], [1.]]), multiplicity=np.array([[np.nan], [2.]]))
    catalog = SimpleNamespace(eligible=np.ones((2, 1), bool), segment_ids=np.array([['s'], ['s']]),
        major_cn=np.full((2, 1), 2), count_observed=np.ones((2, 1), bool), total_counts=np.full((2, 1), 100))
    rows = cap_audit(h, catalog, ('target',), [dict(mutation_id='source', region_id='r', segment_id='s', usable=True)], np.ones((2, 1)))
    assert rows[0]['additional_exclusion'] and not rows[0]['truth_replacement_evaluable']
    assert rows[0]['truth_replacement_upper'] is None
    h.phi[0, 0], h.multiplicity[0, 0] = 1., 1.
    rows = cap_audit(h, catalog, ('target',), [dict(mutation_id='source', region_id='r', segment_id='s', usable=True)], np.ones((2, 1)))
    assert rows[0]['truth_replacement_exclusion'] and not rows[0]['harmful_minimum_call_error']
    assert [transition(x, y) for x, y in ((True, True), (False, True), (False, False),
        (True, False), (True, None), (False, None))] == ['correct_retained', 'error_corrected',
        'error_persisted', 'error_introduced', 'correct_resolution_lost', 'error_withheld']
    states = {'B': dict(available=True, baseline={'t': True, 'f': True}),
        'C': dict(available=True, assignment_available=True, baseline={'t': False, 'f': False},
                  targets={'t'}, non_targets={'f'}, final={'t': True, 'f': False})}
    reach = recovery_reach(states, np.ones(2, bool), {'t': 0, 'f': 1})
    assert reach['H'] == ['f', 't'] and reach['recovery_ceiling'] == {'numerator': 1, 'denominator': 2}
    assert reach['recovered_H']['numerator'] == 1
    # Different mutation/acquisition counts never redefine independent histories.
    cells = []
    for family, histories in (('S01', [(0, [0., 0.]), (1, [1.])]), ('S02', [(0, [1.]), (1, [1.])])):
        for history, values in histories:
            for acquisition, value in enumerate(values):
                cells.append(dict(schema='clipp2.simulation.evaluation.v6', truth_qualification='verified_history_binding',
                    evaluation_definition=evaluation_definition(), evaluation_definition_id=content_hash(evaluation_definition()),
                    family=family, history_id=str(history), acquisition_id=str(acquisition),
                    metrics=[dict(name='score', arm='B', value=value), dict(name='score', arm='C', value=value+.1)]))
    summary = aggregate(cells, bootstrap_replicates=100, seed=1, families=('S01', 'S02'))
    b = next(row for row in summary['metrics'] if row['arm'] == 'B' and row['name'] == 'score')
    paired = next(row for row in summary['metrics'] if row['arm'] == 'C_minus_B' and row['name'] == 'score')
    assert b['equal_family_mean'] == .75
    np.testing.assert_allclose([paired['equal_family_mean'], paired['lower_95'], paired['upper_95']], .1)
    incomplete = aggregate(cells, bootstrap_replicates=0)
    assert all(row['planned_population_estimate'] is None for row in incomplete['metrics'])
    assert incomplete['roster_complete'] is False
    assert summary['schema'] == 'clipp2.simulation.aggregate.v3'
    altered = deepcopy(cells[0])
    altered['evaluation_definition'] = evaluation_definition(dict(reference_match_atol=.1))
    altered['evaluation_definition_id'] = content_hash(altered['evaluation_definition'])
    for invalid in ([cells[0], altered], [dict(cells[0], evaluation_definition_id='tampered')],
                    [cells[0], dict(cells[0], schema='clipp2.simulation.evaluation.v1')]):
        try:
            aggregate(invalid, bootstrap_replicates=0)
        except ValueError:
            pass
        else:
            raise AssertionError('Mixed or invalid evaluation definition was accepted')
    report = dict(schema='clipp2.simulation.evaluation-checks.v3', full_emission=full, capped_emission=cap,
        conditional_map=3, conditional_probability=probabilities[2], marginalization_reverses_profile=True,
        aliases_respect_observed_regions=True, fixed_contributor_identity=True, cap_counterexample=True,
        no_probability_renormalization=True, absent_truth_log_loss='Infinity', recovery_ceiling=True,
        observable_partition=True, history_level_equal_family_aggregation=True, unavailable_families_explicit=True)
    report.update(continuous_reference_geometry=True, binary_recovery_requires_declared_tolerance=True,
                  one_to_one_reference_recall=True, matching_valid_cost_only=True, matching_ambiguity_retained=True,
                  exhaustive_matching_objective_cases=120, binary_objective_unchanged=True,
                  full_vector_l2_and_explicit_legacy_linf=True, incompatible_definition_aggregation_rejected=True)
    report.update(_planned_population_checks(), all_target_MAP=_all_target_checks())
    json.dumps(json_safe(report), allow_nan=False)
    return json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_checks()
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
