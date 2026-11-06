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


def _membership_fixture():
    """Declared frozen decisions isolate endpoint semantics without any inference."""
    ids = ('t_correct', 't_wrong', 't_unmapped', 't_unresolved', 't_clonal', 't_origin2',
           'f_a0', 'f_a1', 'f_a2', 'f_a3', 'f_b_wrong', 'f_b', 'f_mix_a', 'f_mix_b', 'no_info')
    classes = ('a', 'a', 'a', 'a', 'trunk', 'b', 'a', 'a', 'a', 'a', 'b', 'b', 'a', 'b', 'a')
    truth_centers = {'a': [.3, .4], 'b': [.6, .7], 'trunk': [1., 1.]}
    phi = np.array([truth_centers[key] for key in classes])
    shape = phi.shape
    observed = np.ones(shape, bool)
    observed[0, 1] = False
    observed[-1] = False
    history = SimpleNamespace(mutation_ids=ids, region_ids=('r1', 'r2'), phi=phi,
        observable_truth_ids=classes, multiplicity=np.ones(shape), major_cn=np.ones(shape),
        minor_cn=np.ones(shape), segment_ids=np.full(shape, 's'),
        metadata=dict(family='S01', history_id='declared-evaluation-fixture'))
    catalog = SimpleNamespace(mutation_ids=ids, region_ids=history.region_ids, eligible=np.ones(shape, bool),
        count_observed=observed, total_counts=np.full(shape, 100), segment_ids=history.segment_ids,
        major_cn=history.major_cn, minor_cn=history.minor_cn, purity=np.ones(shape), max_major_cn=10,
        mapping_genuine=np.ones(shape, bool))
    centers = [[1., 1.], [.3, .4], [.6, .7], [.45, .55]]
    combined = dict(zip(ids[:-1], [0]*6+[1]*5+[2, 3, 3]))
    first = dict(mutation_ids=ids[:-1], labels=list(combined.values()), centers=centers,
                 multiplicities=np.ones((len(ids)-1, 2)).tolist())
    second = dict(mutation_ids=ids[5:-1], labels=[0]+[1]*5+[2, 3, 3], centers=centers,
                  multiplicities=np.ones((len(ids)-6, 2)).tolist())
    records = [dict(mutation_id=mutation, accepted_label=label,
        status='assigned_relative' if label is not None else 'low_confidence',
        target_reason='stage1_clonal_recheck' if i < 5 else 'stage2_near_clonal_recheck')
        for i, (mutation, label) in enumerate(zip(ids[:6], (1, 2, 3, None, 0, 1)))]
    arm = dict(status='complete', stage_two=second, combined_labels=combined,
        reference=dict(labels=[0, 1, 2, 3], clonal_label=0, centers=centers,
                       regional_quality=[[True, True], [True, False], [True, True], [True, True]]),
        targets=ids[:6], non_targets=ids[6:-1], assignment=dict(assignments=records, multiplicities=[]))
    result = dict(protocol_version='2.2', run_manifest_id='current-evaluation-fixture',
        stage_one=first, g0_ids=ids[:5], remainder_ids=ids[5:-1], arms={'B': arm, 'C': arm},
        config=dict(clonal_distance_max=.05))
    return history, catalog, result


def _new_endpoint_checks():
    h, catalog, paired = _membership_fixture()
    report = evaluate(h, catalog, paired, evaluation_config=dict(reference_match_atol=.02))
    assert report['schema'] == 'clipp2.simulation.evaluation.v2'
    assert report['run_manifest_id'] == paired['run_manifest_id']
    values = {row['name']: row['value'] for row in report['metrics']
              if row['arm'] == 'C3' and row['view'] == '' and row['stratum'] == 'all'}
    # Wrong subclones and unmapped nonclonal candidates preserve binary correctness.
    assert values['correct_resolution'] == values['binary_correct_resolution'] == 5/6
    assert values['selective_error'] == values['binary_selective_error'] == 0
    detailed = report['detailed_membership']['C3']['targets']
    assert (detailed['Mc'], detailed['Mw'], detailed['U'], detailed['unresolved']) == (2, 2, 1, 1)
    assert detailed['mapped_selective_error'] == .5
    assert detailed['detailed_selective_error_lower_bound'] == 2/5
    assert detailed['detailed_selective_error_upper_bound'] == 3/5
    fixed = report['detailed_membership']['C3']['unchanged_F']
    assert (fixed['Mc'], fixed['Mw'], fixed['U']) == (5, 1, 2)
    rows = {row['mutation_id']: row for row in report['mutations'] if row['arm'] == 'C3'}
    assert rows['t_wrong']['binary_clonality_correct'] is True
    assert rows['t_wrong']['detailed_membership_status'] == 'Mw'
    assert rows['t_unmapped']['detailed_membership_correct'] is None
    assert rows['t_correct']['geometric_reference_available'] is True
    assert rows['t_correct']['usable_reference_available'] is True  # Failed region unobserved.
    assert rows['t_correct']['geometric_correct_reference_acceptance'] is True
    assert rows['t_wrong']['geometric_reference_available'] is True
    assert rows['t_wrong']['usable_reference_available'] is False
    assert rows['t_origin2']['target_reason'] == 'stage2_near_clonal_recheck'
    assert rows['f_b_wrong']['role'] == 'unchanged_non_target'
    assert rows['f_b_wrong']['detailed_membership_status'] == 'Mw'
    assert rows['no_info']['usable_reference_available'] is None
    assert rows['no_info']['usable_availability_status'] == 'not_evaluable_no_informative_observation'
    null = evaluate(h, catalog, paired)
    assert all(row['geometric_reference_available'] is None for row in null['mutations'])
    assert all(row['usable_availability_status'] == 'unassessed_no_declared_tolerance' for row in null['mutations'])
    return report


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
    endpoint_report = _new_endpoint_checks()
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
                cells.append(dict(schema='clipp2.simulation.evaluation.v2',
                    evaluation_definition=evaluation_definition(), evaluation_definition_id=content_hash(evaluation_definition()),
                    family=family, history_id=str(history), acquisition_id=str(acquisition),
                    metrics=[dict(name='score', arm='B', value=value), dict(name='score', arm='C', value=value+.1)]))
    summary = aggregate(cells, bootstrap_replicates=100, seed=1, families=('S01', 'S02'))
    b = next(row for row in summary['metrics'] if row['arm'] == 'B')
    paired = next(row for row in summary['metrics'] if row['arm'] == 'C_minus_B')
    assert b['equal_family_mean'] == .75
    np.testing.assert_allclose([paired['equal_family_mean'], paired['lower_95'], paired['upper_95']], .1)
    incomplete = aggregate(cells, bootstrap_replicates=0)
    assert all(row['equal_family_mean'] is None for row in incomplete['metrics'])
    assert summary['schema'] == 'clipp2.simulation.aggregate.v2'
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
    old = dict(cells[0], schema='clipp2.simulation.evaluation.v1', reference_match_atol=.11)
    historical = aggregate([old], bootstrap_replicates=0, families=('S01',))
    assert historical['evaluation_definition']['reference_distance'] == 'linf'
    assert endpoint_report['recovery_reach']['recovery_ceiling'] == {'numerator': 0, 'denominator': 0}
    report = dict(schema='clipp2.simulation.evaluation-checks.v2', full_emission=full, capped_emission=cap,
        conditional_map=3, conditional_probability=probabilities[2], marginalization_reverses_profile=True,
        aliases_respect_observed_regions=True, fixed_contributor_identity=True, cap_counterexample=True,
        no_probability_renormalization=True, absent_truth_log_loss='Infinity', recovery_ceiling=True,
        observable_partition=True, history_level_equal_family_aggregation=True, unavailable_families_explicit=True)
    report.update(continuous_reference_geometry=True, binary_recovery_requires_declared_tolerance=True,
                  one_to_one_reference_recall=True, matching_valid_cost_only=True, matching_ambiguity_retained=True,
                  exhaustive_matching_objective_cases=120, binary_objective_unchanged=True,
                  Mc_Mw_U_and_unchanged_F=True, target_specific_availability_nulls_and_quality=True,
                  full_vector_l2_and_explicit_legacy_linf=True, incompatible_definition_aggregation_rejected=True)
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
