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
    evaluation_definition, legacy_evaluation_definition, target_reference_availability, _qualified_matching)
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


def _routed_map_fixture():
    """Frozen preceding Protocol 2.3 policy; no changed reference construction."""
    h, catalog, paired = _membership_fixture()
    paired.update(protocol_version="2.3", schema="clipp2.paired-experiment.v4",
                  config=dict(stage_two_near_clonal_distance_max=.05))
    for arm in paired["arms"].values():
        arm["combined_labels"]["no_info"] = 0
        arm["targets"] = tuple(arm["targets"]) + ("no_info",)
        # Shared fixture arms are intentionally identical; update each once.
        if arm["assignment"].get("protocol_version") == "2.3":
            arm["targets"] = tuple(dict.fromkeys(arm["targets"]))
            continue
        arm["assignment"].update(protocol_version="2.3", schema="clipp2.assignment.result.v5")
        arm["reference"].update(source_informative_count=[[0, 1]]*4,
            source_informative_depth=[[0, 100]]*4,
            source_mean_profile_deviance=[[None, np.inf]]*4,
            source_maximum_profile_deviance=[[None, np.inf]]*4,
            diagnostic_availability=[["not_applicable", "available"]]*4)
        for row in arm["assignment"]["assignments"]:
            label = row.pop("accepted_label")
            label = 1 if label is None else label
            probabilities = np.full(4, .2)
            probabilities[label] = .4
            row.update(final_assignment_label=label, map_cluster_label=label, map_tied_labels=[label], map_unique=True,
                evidence_status="observed", status="observed", probabilities=probabilities.tolist(),
                log_scores=np.log(probabilities).tolist(), best_log_probability=float(np.log(.4)),
                log_competitor_ratio=float(np.log(1.5)), has_finite_alternative=True,
                diagnostic_status="failed", diagnostic_error="reporting-only fixture failure")
            row["target_reason"] = row["target_reason"].replace("stage1_clonal", "stage1_reference")
        arm["assignment"]["assignments"].append(dict(mutation_id="no_info", final_assignment_label=None,
            map_cluster_label=0, map_tied_labels=[0, 1, 2, 3], map_unique=False, evidence_status="prior_only",
            status="prior_only", probabilities=[.25]*4, log_scores=[-np.log(4)]*4,
            target_reason="stage1_reference_recheck"))
    return h, catalog, paired


def _protocol23_map_checks():
    """Historical routed MAP labels and population denominators stay frozen."""
    h, catalog, paired = _routed_map_fixture()
    report = evaluate(h, catalog, paired, evaluation_config=dict(reference_match_atol=.02),
                      truth_qualification="verified_history_binding")
    assert report["schema"] == "clipp2.simulation.evaluation.v4"
    assert content_hash(report) == "39b02730a48ecd93e1dce30a8ce53ee5f2bb62457ab02fd5078e97ec8546f98e"
    assert report["evaluation_definition"] == legacy_evaluation_definition(4, dict(reference_match_atol=.02))
    def metric(value, name, view=""):
        return next(row for row in value["metrics"] if row["arm"] == "C3" and row["name"] == name
                    and row["view"] == view and row["stratum"] == "all")
    assert metric(report, "data_supported_map_coverage")["value"] == 1
    assert metric(report, "binary_map_error")["denominator"] == 6
    assert metric(report, "binary_map_error")["value"] == 0
    assert metric(report, "detailed_map_error")["value"] == 2/6
    assert metric(report, "prior_only_fraction")["value"] == 1/7
    assert metric(report, "binary_map_correct_per_original_evaluable")["denominator"] == 14
    assert metric(report, "binary_map_error", "unchanged_F")["denominator"] == 8
    assert metric(report, "binary_map_error", "stage2_near_clonal_recheck")["denominator"] == 1
    assert metric(report, "map_ccf_rmse")["value"] > 0
    assert not report["error_coverage_curves"]
    assert not any("selective" in row["name"] or "acceptance" in row["name"] for row in report["metrics"])
    rows = {row["mutation_id"]: row for row in report["mutations"] if row["arm"] == "C3"}
    assert rows["t_unresolved"]["geometric_correct_map"] is True
    assert rows["t_unresolved"]["diagnostic_status"] == "failed"
    assert rows["t_unresolved"]["reference_source_diagnostics"]["source_mean_profile_deviance"] == [None, "Infinity"]
    assert rows["t_unresolved"]["reference_coverage_stratum"] == "zero_in_observed_region"
    assert any(row["view"] == "diagnostic_strata" and row["stratum"] == "reference_coverage_stratum:zero_in_observed_region"
               and row["name"] == "data_supported_map_coverage" and row["value"] == 1 for row in report["metrics"])
    assert rows["no_info"]["map_cluster_label"] == 0 and rows["no_info"]["final_assignment_label"] is None
    assert all("accepted_label" not in row and "reference_quality_pass" not in row for row in rows.values())
    # Displayed probabilities can round to a tie although absolute scores have a
    # unique MAP. Calibration correctness follows the retained MAP, not argmax(p).
    rounded = deepcopy(paired)
    for arm in rounded["arms"].values():
        row = arm["assignment"]["assignments"][1]
        row.update(probabilities=[0., .5, .5, 0.],
                   log_scores=[-np.inf, -1., np.nextafter(-1., 0.), -np.inf])
    rounded_report = evaluate(h, catalog, rounded, truth_qualification="verified_history_binding")
    calibration = next(row for row in rounded_report["calibration"]
                       if row["arm"] == "C3" and row["mutation_id"] == "t_wrong")
    assert calibration["confidence_correct"] is False
    assert not any(":usable:" in row["stratum"] for row in report["metrics"])
    incompatible = deepcopy(paired)
    for arm in incompatible["arms"].values():
        row = arm["assignment"]["assignments"][3]
        row.update(final_assignment_label=None, map_cluster_label=None, evidence_status="model_incompatible",
                   status="model_incompatible", probabilities=None)
    value = evaluate(h, catalog, incompatible, truth_qualification="verified_history_binding")
    assert metric(value, "data_supported_map_coverage")["value"] == 5/6
    assert metric(value, "model_incompatible_fraction")["value"] == 1/7
    failed = deepcopy(paired)
    for arm in failed["arms"].values():
        arm.update(assignment=None, assignment_status="failed", status="failed", failed_stage="assignment")
    value = evaluate(h, catalog, failed, truth_qualification="verified_history_binding")
    assert metric(value, "computation_failure_count")["value"] == 7
    assert metric(value, "binary_map_correct_per_original_evaluable")["value"] == 8/14
    assert metric(value, "data_supported_map_coverage", "whole_original")["value"] == 8/14
    return dict(status="passed", observed_MAP_ignores_diagnostics=True, fixed_error_denominators=True,
                prior_only_incompatible_failure_distinct=True, both_origins_and_unchanged_F=True)


def _all_target_checks():
    """Current schema reassigns every fitted member, without historical routing."""
    h, catalog, paired = _routed_map_fixture()
    paired.update(schema="clipp2.paired-experiment.v5", config=dict(fit=dict(max_major_cn=None)))
    catalog.max_major_cn = None
    for arm in paired["arms"].values():
        if arm["assignment"]["schema"] == "clipp2.assignment.result.v6":
            continue
        prototype = deepcopy(arm["assignment"]["assignments"][0])
        for mutation in arm["non_targets"]:
            label = arm["combined_labels"][mutation]
            probabilities = np.full(4, .2)
            probabilities[label] = .4
            arm["assignment"]["assignments"].append(dict(prototype, mutation_id=mutation,
                final_assignment_label=label, map_cluster_label=label, map_tied_labels=[label],
                probabilities=probabilities.tolist(), log_scores=np.log(probabilities).tolist(),
                target_reason="stage2_recheck"))
        arm.update(targets=tuple(arm["combined_labels"]), non_targets=())
        arm["assignment"]["schema"] = "clipp2.assignment.result.v6"
        for row in arm["assignment"]["assignments"]:
            row.update(map_label=row["map_cluster_label"], map_set=row["map_tied_labels"], posterior_defined=True,
                target_reason=row["target_reason"].replace("stage2_near_clonal_recheck", "stage2_recheck"))
    value = evaluate(h, catalog, paired, truth_qualification="verified_history_binding")
    assert value["schema"] == "clipp2.simulation.evaluation.v5"
    assert value["evaluation_definition"] == evaluation_definition()
    metrics = {(r["name"], r["view"]): r for r in value["metrics"] if r["arm"] == "C3" and r["stratum"] == "all"}
    assert metrics["data_supported_map_coverage", ""]["denominator"] == 14
    assert metrics["data_supported_map_coverage", ""]["value"] == 1
    assert metrics["data_supported_map_coverage", "unchanged_F"]["denominator"] == 0
    assert metrics["data_supported_map_coverage", "stage2_recheck"]["denominator"] == 9
    assert not any("near_clonal" in r["name"] or "false_merge" in r["name"] for r in value["metrics"])
    rows = [r for r in value["mutations"] if r["arm"] == "C3"]
    assert all(r["role"] == "target" for r in rows)
    assert all(r["posterior_defined"] and r["map_label"] in r["map_set"] for r in rows)
    invalid = deepcopy(paired)
    invalid["arms"]["C"]["non_targets"] = ("t_correct",)
    try:
        evaluate(h, catalog, invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Current all-target schema accepted an unchanged population")
    return dict(status="passed", F_empty=True, every_fitted_member_reassessed=True,
                unrestricted_catalog_supported=True, explicit_MAP_fields=True)


def _frozen_historical_checks():
    """Old cached science and decisions survive the new default definition."""
    from contextlib import ExitStack
    import tempfile
    from unittest.mock import patch
    import zipfile
    from CliPP2.simulation.reporting import reevaluate_cell
    assert content_hash(legacy_evaluation_definition(2)) == "728032ef78442bb44387f97d58e409e5415831d2065615d3956757a83e92922e"
    assert content_hash(legacy_evaluation_definition(3)) == "be475aebc3de6b83c9345b11f6f3aac2673efa15f1b4fcb279f69ed4bf0d5440"
    assert content_hash(legacy_evaluation_definition(4)) == "a15483be43fd5fa024331e6711f22b63f6b13e0e0cc6594374e151af1d44eb6a"
    changed = legacy_evaluation_definition(3)
    changed["population_scopes"]["planned_population"].clear()
    assert legacy_evaluation_definition(3)["population_scopes"]["planned_population"]
    with tempfile.TemporaryDirectory(prefix="clipp2-protocol23-history-") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(Path(__file__).parent / "fixtures/f2d1fa1_successful_evidence.zip") as archive:
            archive.extractall(root)
        cell = json.loads((root / "cells/main-S01-H001-A2/completion.json").read_text())
        cached = json.loads((root / cell["evaluations"][0]).read_text())
        before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
        with ExitStack() as stack:
            for target in ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
                           "CliPP2.kernel.model.RegionalModel", "CliPP2.simulation.biology.generate_history",
                           "CliPP2.simulation.acquisition.acquire"):
                stack.enter_context(patch(target, side_effect=AssertionError("Historical report attempted inference")))
            derived = reevaluate_cell(root, cell, dict(reference_match_atol=.05, reference_distance="l2"))
        assert derived["status"] == "derived_from_preserved_evidence", derived.get("error")
        assert derived["schema"] == "clipp2.simulation.evaluation.v3"
        assert derived["evaluation_definition"] == cached["evaluation_definition"]
        assert len(derived["metrics"]) == 853 and derived["metrics"] == cached["metrics"]
        assert derived["mutations"] == cached["mutations"]
        assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    return dict(frozen_v2_v3_identities=True, old_853_metrics_and_decisions_preserved=True,
                no_inference_or_source_changes=True)


def _new_endpoint_checks():
    h, catalog, paired = _membership_fixture()
    report = evaluate(h, catalog, paired, evaluation_config=dict(reference_match_atol=.02),
                      truth_qualification='verified_history_binding')
    assert report['schema'] == 'clipp2.simulation.evaluation.v3'
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
    assert rows['t_unmapped']['baseline_detailed_membership_status'] == 'Mw'
    assert rows['t_unmapped']['detailed_membership_transition'] == 'Mw->U'
    transitions = report['detailed_membership_transitions']['C3']
    origin1 = {(row['baseline_state'], row['final_state']): row['count'] for row in transitions['stage1_clonal_recheck']}
    origin2 = {(row['baseline_state'], row['final_state']): row['count'] for row in transitions['stage2_near_clonal_recheck']}
    assert origin1['Mw', 'U'] == 1 and origin1['Mw', 'Mw'] == 1
    assert origin2['Mw', 'Mw'] == 1
    assert not any(row['name'] == 'multiclass_transition_fraction' for row in report['metrics'])
    assert any(row['name'] == 'legacy_multiclass_transition_fraction_unmapped_as_error' for row in report['metrics'])
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


def _planned_population_checks():
    definition = legacy_evaluation_definition(3)
    roster = [dict(cell_id=f'h{i}-A1', family='S01', history_id=f'h{i}', acquisition=dict(id='A1')) for i in range(2)]

    def cell(job, score, *, failure=False, evaluated=True, truth='verified_history_binding', denominator=10):
        metrics = []
        for arm in ('B3', 'C3'):
            metrics.append(dict(name='binary_correct_per_original_evaluable', arm=arm, view='', stratum='all',
                value=score, numerator=None if score is None else score*denominator if denominator else 0,
                denominator=denominator, truth_required=not failure,
                status='known_no_decisions' if failure else 'unknown_evidence' if score is None else 'available'))
            for name, value in (('analysis_available', int(not failure)), ('analysis_failure', int(failure)),
                    ('execution_available', 1), ('execution_failure', int(failure)),
                    ('evaluation_available', int(evaluated)), ('scientifically_applicable', int(not failure))):
                metrics.append(dict(name=name, arm=arm, view='all_planned_cells', stratum='all', value=value,
                                    numerator=value, denominator=1, truth_required=False))
        return dict(schema='clipp2.simulation.evaluation.v3', evaluation_definition=definition,
            evaluation_definition_id=content_hash(definition), truth_qualification=truth,
            cell_id=job['cell_id'], family=job['family'], history_id=job['history_id'], acquisition_id=job['acquisition']['id'],
            original_evaluable_count=denominator, status='execution_failed' if failure else 'complete', metrics=metrics,
            applicability=[dict(arm=arm, status='failed', decision_population=dict(
                scope='whole_original_evaluable', coverage='complete', evaluable_count=denominator,
                accepted_count=0, evidence='failure_before_stage_one')) for arm in ('B3', 'C3')] if failure else [])

    def summary(records, planned=roster, complete=True, families=('S01',)):
        return aggregate(records, planned_roster=planned, roster_complete=complete,
                         families=families, bootstrap_replicates=20, seed=4)

    def endpoint(report, name='binary_correct_per_original_evaluable', arm='C3'):
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
    legacy = cell(roster[1], 1., truth='legacy_consistent_unbound')
    qualified = endpoint(summary([success, legacy]))
    assert qualified['planned_population_estimate'] is None and qualified['available_case_estimate'] == 1
    assert qualified['legacy_consistent_available_case_estimate'] == 1
    assert qualified['missingness'][0]['reason'] == 'legacy_consistent_unbound'
    spoofed = deepcopy(legacy)
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
        unavailable_current_definition_preserved=True, incomplete_legacy_roster_not_invented=True,
        verified_primary_excludes_unbound_legacy=True)


def _partial_failure_checks():
    """Real retained catalog/manifest; declared memberships; zero inference calls."""
    from contextlib import ExitStack
    import tempfile
    from unittest.mock import patch
    import pandas as pd
    from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
    from CliPP2.provenance import prepare_run_manifest
    from CliPP2.simulation.reporting import placeholder_for_cell

    def metric(record, name='correct_per_original_evaluable', arm='C3'):
        return next(row for row in record['metrics'] if row['name'] == name and row['arm'] == arm)

    with tempfile.TemporaryDirectory(prefix='clipp2-partial-failure-') as temporary, ExitStack() as stack:
        for name in ('CliPP2.fit_execution.fit_prepared', 'CliPP2.kernel.model.RegionalModel',
                     'CliPP2.assignment.assign_mutations'):
            stack.enter_context(patch(name, side_effect=AssertionError('Fallback evaluation invoked inference')))
        root = Path(temporary)
        h, data, paired = _membership_fixture()
        job = dict(cell_id='declared-partial-failure', family='S01', history_id='h0',
                   acquisition=dict(id='A1'), phase='main')
        attempt = root / 'cells' / job['cell_id'] / 'attempt-001'
        configuration = attempt / 'configuration-1'
        configuration.mkdir(parents=True)
        frame = pd.DataFrame([dict(mutation_id=mutation, sample_id=region,
            alt_count=int(round(50*h.phi[i, j])) if data.count_observed[i, j] else '.',
            ref_count=100-int(round(50*h.phi[i, j])) if data.count_observed[i, j] else '.',
            count_observed=int(data.count_observed[i, j]), purity=1., normal_cn=2, normal_cn_supplied=1,
            segment_id='s', mapping_genuine=1, cn_state_id='0', cn_state_fraction=1.,
            allele_a_cn=1, allele_b_cn=1)
            for i, mutation in enumerate(h.mutation_ids) for j, region in enumerate(h.region_ids)])
        write_tumor_txt(attempt / 'input_catalog.tsv', frame,
                        dict(tumor_id=job['cell_id'], fixture='declared_memberships_not_native_inference'))
        catalog = load_original_catalog(attempt / 'input_catalog.tsv')
        order = [h.mutation_ids.index(mutation) for mutation in catalog.mutation_ids]
        truth = SimpleNamespace(**vars(h))
        truth.mutation_ids = catalog.mutation_ids
        for name in ('phi', 'multiplicity', 'major_cn', 'minor_cn', 'segment_ids'):
            setattr(truth, name, getattr(h, name)[order])
        truth.observable_truth_ids = tuple(h.observable_truth_ids[i] for i in order)
        paired['arms'] = {name: deepcopy(value) for name, value in paired['arms'].items()}
        paired.update(status='partial_failure', run_manifest=prepare_run_manifest(catalog),
                      original_mutation_ids=catalog.mutation_ids, region_ids=catalog.region_ids)
        original = deepcopy(paired)
        paired['arms']['C'].update(status='failed', assignment_status='failed', failed_stage='assignment', assignment=None)
        (attempt / 'job.json').write_text(json.dumps(job))
        (attempt / 'acquisition.json').write_text(json.dumps(dict(biological_truth_available=np.ones_like(truth.phi, bool).tolist())))
        cell = dict(job=job, attempt=str(attempt.relative_to(root)), status='execution_failed',
                    original_evaluable_count=14, execution_failures=[dict(arm='C', failed_stage='assignment')])
        paired_path = configuration / 'paired_result.json'

        def fallback(value):
            paired_path.write_text(json.dumps(value))
            return placeholder_for_cell(root, cell, None)

        def planned(value):
            return aggregate([value], planned_roster=[job], roster_complete=True, families=('S01',), bootstrap_replicates=0)

        operational_cases = []

        def compare_operations(value, label):
            complete = evaluate(truth, catalog, value, truth_qualification='verified_history_binding')
            incomplete = fallback(value)
            for arm in ('A', 'B', 'C', 'B3', 'C3'):
                for name in ('analysis_available', 'analysis_failure', 'execution_available',
                             'execution_failure', 'scientifically_applicable'):
                    assert metric(complete, name, arm) == metric(incomplete, name, arm), (label, name, arm)
                assert metric(complete, 'evaluation_available', arm)['value'] == 1
                assert metric(incomplete, 'evaluation_available', arm)['value'] == 0
            operational_cases.append(label)
            return complete, incomplete

        # 1. Full evidence keeps unchanged F's exact contribution after assignment failure.
        full = evaluate(truth, catalog, paired, truth_qualification='verified_history_binding')
        assert metric(full)['value'] == 8/14 and metric(full)['numerator'] == 8
        # 2. Removing truth cannot turn that retained contribution into an exact zero.
        missing = fallback(paired)
        compare_operations(paired, 'assignment_failure')
        for record in (full, missing):
            assert metric(record, 'analysis_available', 'C')['value'] == 1
            assert metric(record, 'analysis_failure', 'C')['value'] == 0
            assert metric(record, 'execution_failure', 'C')['value'] == 0
            assert metric(record, 'scientifically_applicable', 'C3')['value'] == 1
        assert metric(missing)['value'] is None and metric(missing)['truth_required'] is True
        assert metric(planned(missing))['planned_population_estimate'] is None
        assert metric(planned(missing))['planned_population_status'] == 'incomplete_evidence'
        evidence = next(row for row in missing['applicability'] if row['arm'] == 'C3')['decision_population']
        assert evidence['accepted_count'] == evidence['unchanged_count'] == 8 and evidence['accepted_target_count'] == 0
        # 3. Explicitly empty F and no returned target decisions still establish zero.
        empty = deepcopy(paired)
        empty['arms']['C'].update(non_targets=[], targets=list(empty['arms']['C']['combined_labels']),
                                 combined_labels={mutation: 0 for mutation in empty['arms']['C']['combined_labels']})
        zero = fallback(empty)
        assert metric(zero)['value'] == 0 and metric(zero)['status'] == 'known_no_decisions'
        assert metric(planned(zero))['planned_population_estimate'] == 0
        # Returned all-unresolved decisions also prove emptiness without an execution failure.
        withheld = deepcopy(empty)
        withheld['arms']['C'].update(status='complete', assignment_status='complete', failed_stage=None,
            assignment=dict(assignments=[dict(mutation_id=mutation, accepted_label=None)
                                         for mutation in withheld['arms']['C']['targets']]))
        assert metric(fallback(withheld))['value'] == 0
        # One retained accepted target prevents a whole-population zero even with empty F.
        withheld['arms']['C']['assignment']['assignments'][0]['accepted_label'] = 0
        assert metric(fallback(withheld))['value'] is None
        # 4. Missing F evidence or a missing paired result keeps the claim unknown.
        absent = deepcopy(paired)
        absent['arms']['C'].pop('non_targets')
        assert metric(fallback(absent))['value'] is None
        paired_path.unlink()
        missing_payload = placeholder_for_cell(root, cell, None)
        assert metric(missing_payload)['value'] is None
        assert metric(missing_payload, 'analysis_available', 'C3')['value'] == 0
        assert metric(missing_payload, 'execution_failure', 'C3')['value'] == 1
        assert metric(missing_payload, 'execution_failure', 'C')['value'] == 0
        for phase in ('assignment_publication', 'reconciliation'):
            failed_without_payload = dict(cell, execution_failures=[dict(arm='C', failed_stage=phase)])
            record = placeholder_for_cell(root, failed_without_payload, None)
            assert metric(record, 'analysis_available', 'C3')['value'] is None
            assert metric(record, 'execution_failure', 'C3')['value'] == 1
        unexecuted = placeholder_for_cell(root, dict(job=job, status='not_executed'), None)
        for arm in ('A', 'B', 'C', 'B3', 'C3'):
            for name in ('analysis_available', 'execution_available', 'execution_failure'):
                assert metric(unexecuted, name, arm)['value'] == 0
        operational_cases.append('explicit_not_executed_and_assignment_failure_without_payload')
        # Early total failure can establish N later from retained catalog/acquisition.
        early = dict(cell, execution_failures=[dict(arm=None, failed_stage='stage_one')])
        early.pop('original_evaluable_count')
        recovered_denominator = placeholder_for_cell(root, early, None)
        assert recovered_denominator['original_evaluable_count'] == 14
        assert metric(recovered_denominator)['value'] == 0
        assert metric(recovered_denominator, 'execution_failure', 'A')['value'] == 1
        for arm in ('B', 'C', 'B3', 'C3'):
            assert metric(recovered_denominator, 'execution_failure', arm)['value'] == 0
            assert metric(recovered_denominator, 'execution_available', arm)['value'] == 0
        # Present malformed evidence is not equivalent to absent publication.
        paired_path.write_text('null')
        malformed_payload = placeholder_for_cell(root, early, None)
        assert malformed_payload['original_evaluable_count'] is None
        assert metric(malformed_payload)['value'] is None
        # Malformed returned target evidence cannot claim C3 analysis availability.
        for malformed in ({}, dict(assignments=[])):
            invalid = deepcopy(paired)
            invalid['arms']['C']['assignment'] = malformed
            record = fallback(invalid)
            assert metric(record, 'analysis_available', 'C3')['value'] is None
            assert metric(record, 'analysis_available', 'C')['value'] == 1
        publication_failure = deepcopy(paired)
        publication_failure['arms']['C'].update(assignment=original['arms']['C']['assignment'],
                                                failed_stage='assignment_publication', assignment_status='scored')
        for record in compare_operations(publication_failure, 'publication_failure_after_scoring'):
            assert metric(record, 'analysis_available', 'C3')['value'] == 1
            assert metric(record, 'execution_failure', 'C3')['value'] == 1
            assert metric(record, 'execution_failure', 'C')['value'] == 0
            assert metric(record, 'scientifically_applicable', 'C3')['value'] == 1
        # Scientific rejection is inapplicable, without an execution exception.
        rejected = deepcopy(original)
        rejected.update(status='no_eligible_clonal_reference', g0_ids=[])
        for arm in rejected['arms'].values():
            arm.update(status='no_eligible_clonal_reference', combined_labels={}, reference=None,
                       stage_two=None, targets=[], non_targets=[], assignment=None)
        for record in compare_operations(rejected, 'scientific_rejection'):
            for arm in ('B', 'C', 'B3', 'C3'):
                assert metric(record, 'scientifically_applicable', arm)['value'] == 0
                assert metric(record, 'execution_failure', arm)['value'] == 0
        failed_before_stage_one = deepcopy(rejected)
        failed_before_stage_one.update(status='failed', failed_stage='stage_one', stage_one=None)
        for arm in failed_before_stage_one['arms'].values():
            arm['status'] = 'unavailable_upstream'
        for record in compare_operations(failed_before_stage_one, 'retained_upstream_failure'):
            assert metric(record)['value'] == 0
            for arm in ('B', 'C', 'B3', 'C3'):
                assert metric(record, 'execution_failure', arm)['value'] == 0
                assert metric(record, 'execution_available', arm)['value'] == 0
        missing_root_status = deepcopy(original)
        missing_root_status.pop('status')
        missing_root_status['stage_one_status'] = 'accepted'
        for record in compare_operations(missing_root_status, 'accepted_stage_one_unknown_later_operations'):
            assert metric(record, 'analysis_available', 'A')['value'] == 1
            assert metric(record, 'execution_failure', 'A')['value'] is None
        # Retained output does not identify the phase of an otherwise unknown error.
        unknown_phase = deepcopy(publication_failure)
        unknown_phase['arms']['C']['failed_stage'] = 'unrecognized_operation'
        for record in compare_operations(unknown_phase, 'unknown_failure_attribution'):
            for arm in ('C', 'C3'):
                assert metric(record, 'analysis_available', arm)['value'] == 1
                assert metric(record, 'execution_failure', arm)['value'] is None
        # Absent execution facts remain unknown in both derivations.
        missing_state = deepcopy(original)
        missing_state['arms']['C'] = {}
        for record in compare_operations(missing_state, 'missing_execution_facts'):
            for arm in ('C', 'C3'):
                for name in ('analysis_available', 'execution_available', 'execution_failure'):
                    assert metric(record, name, arm)['value'] is None
        # Earlier B/C operations are distinct from the downstream assignment pass.
        for phase in ('stage_two', 'stage_two_accepted_publication', 'classification', 'freeze_reference'):
            stage_two_failure = deepcopy(original)
            stage_two_failure['arms']['C'].update(status='failed', failed_stage=phase,
                combined_labels={}, reference=None, targets=[], non_targets=[], assignment=None)
            for record in compare_operations(stage_two_failure, phase):
                assert metric(record, 'execution_failure', 'C')['value'] == 1
                assert metric(record, 'execution_failure', 'C3')['value'] == 0
                assert metric(record, 'execution_available', 'C3')['value'] == 0
        # 5. Downstream C3 failure does not erase completed B/B3 or C evidence.
        preserved = fallback(paired)
        for arm in ('B', 'B3', 'C'):
            assert metric(preserved, 'analysis_available', arm)['value'] == 1
            assert metric(preserved, 'execution_failure', arm)['value'] == 0
        assert metric(preserved, 'analysis_available', 'C3')['value'] == 0
        assert metric(preserved, 'execution_failure', 'C3')['value'] == 1
        baseline = evaluate(truth, catalog, original, truth_qualification='verified_history_binding')
        for arm in ('B', 'B3', 'C'):
            assert metric(full, arm=arm)['value'] == metric(baseline, arm=arm)['value']
        # 6. Equal-weight successful and established empty populations retain mean .5.
        success = deepcopy(zero)
        other_job = dict(job, cell_id='declared-success', history_id='h1')
        success.update(cell_id=other_job['cell_id'], history_id='h1', truth_qualification='verified_history_binding')
        for row in success['metrics']:
            if row['arm'] == 'C3' and row['name'] in ('correct_per_original_evaluable', 'binary_correct_per_original_evaluable'):
                row.update(value=1., numerator=14, truth_required=True, status='available')
        combined = aggregate([zero, success], planned_roster=[job, other_job], roster_complete=True,
                             families=('S01',), bootstrap_replicates=0)
        assert metric(combined)['planned_population_estimate'] == .5
        return dict(partial_failure_full_evidence_correct=8/14, retained_unchanged_count=8,
            missing_truth_whole_population_unknown=True, empty_F_no_target_decisions_zero=True,
            missing_F_evidence_unknown=True, completed_upstream_availability_retained=True,
            success_plus_total_failure_mean=.5, all_unresolved_empty_population_zero=True,
            retained_accepted_targets_prevent_zero=True, recovered_denominator_preserves_total_failure_zero=True,
            present_null_does_not_recover_denominator=True,
            malformed_assignment_availability_unknown=True, retained_scored_payload_available=True,
            operational_truth_invariance_cases=operational_cases, no_inference_calls=True)


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
                cells.append(dict(schema='clipp2.simulation.evaluation.v3', truth_qualification='verified_history_binding',
                    evaluation_definition=legacy_evaluation_definition(3), evaluation_definition_id=content_hash(legacy_evaluation_definition(3)),
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
    altered['evaluation_definition'] = legacy_evaluation_definition(3, dict(reference_match_atol=.1))
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
    report = dict(schema='clipp2.simulation.evaluation-checks.v3', full_emission=full, capped_emission=cap,
        conditional_map=3, conditional_probability=probabilities[2], marginalization_reverses_profile=True,
        aliases_respect_observed_regions=True, fixed_contributor_identity=True, cap_counterexample=True,
        no_probability_renormalization=True, absent_truth_log_loss='Infinity', recovery_ceiling=True,
        observable_partition=True, history_level_equal_family_aggregation=True, unavailable_families_explicit=True)
    report.update(continuous_reference_geometry=True, binary_recovery_requires_declared_tolerance=True,
                  one_to_one_reference_recall=True, matching_valid_cost_only=True, matching_ambiguity_retained=True,
                  exhaustive_matching_objective_cases=120, binary_objective_unchanged=True,
                  Mc_Mw_U_and_unchanged_F=True, target_specific_availability_nulls_and_quality=True,
                  full_vector_l2_and_explicit_legacy_linf=True, incompatible_definition_aggregation_rejected=True)
    report.update(_planned_population_checks(), detailed_four_state_transitions_by_origin=True,
                  partial_failure_accounting=_partial_failure_checks(), protocol23_MAP=_protocol23_map_checks(),
                  historical_definitions=_frozen_historical_checks(), all_target_MAP=_all_target_checks())
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
