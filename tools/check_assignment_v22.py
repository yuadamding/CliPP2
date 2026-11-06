#!/usr/bin/env python3
"""Focused Protocol 2.2 assignment contracts against the imported installation."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2.assignment import (AssignmentBundle, AssignmentReference, AssignmentResult,
                               assign_mutations, score_summary)
from CliPP2.config import AssignmentThresholds, DescriptiveScreen, PROTOCOL_VERSION
from CliPP2.io.assignment import (load_assignment_bundle, publish_assignment, save_assignment_bundle,
                                  validate_assignment_result)
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt


def _catalog(root, name, alt, depth, major, *, minor=0, observed=True):
    alt, depth, major, minor, observed = np.broadcast_arrays(
        np.atleast_2d(alt), np.atleast_2d(depth), np.atleast_2d(major),
        np.atleast_2d(minor), np.atleast_2d(observed))
    rows = [dict(mutation_id=f"m{i}", sample_id=f"r{j}", alt_count=int(alt[i, j]),
        ref_count=int(depth[i, j]-alt[i, j]), count_observed=int(observed[i, j]), purity=1., normal_cn=2,
        segment_id="s", cn_state_id="0", cn_state_fraction=1., allele_a_cn=int(major[i, j]),
        allele_b_cn=int(minor[i, j])) for i in range(alt.shape[0]) for j in range(alt.shape[1])]
    path = write_tumor_txt(root / (name + ".tsv"), pd.DataFrame(rows))
    return load_original_catalog(path)


def _reference(catalog, centers, quality=None):
    centers = np.asarray(centers, float)
    flags = np.ones(centers.shape, bool) if quality is None else np.asarray(quality, bool)
    return AssignmentReference(tuple(range(len(centers))), catalog.region_ids, centers,
        clonal_label=0 if np.all(centers[0] == 1) else None, regional_quality=flags,
        regional_quality_reasons=tuple(tuple(() if value else ("fixture_source_screen_failed",)
                                              for value in row) for row in flags),
        center_sources=tuple(dict(kind="declared_numerical_fixture", source_stage="fixture", source_label=i)
                             for i in range(len(centers))))


def _bundle(catalog, reference, *, source_stage=None, posterior=.9):
    if source_stage is None:
        source_stage = "stage_one" if reference.clonal_label is not None else "fixture"
    bounds = catalog.default_bounds()
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    thresholds = AssignmentThresholds(posterior, DescriptiveScreen(10., 5., .05), 1e-10, 1e-10, .95)
    return AssignmentBundle(catalog, reference, catalog.mutation_ids, (), thresholds,
        np.ones(informative.shape), np.where(informative, 1., np.nan), bounds.lower, bounds.upper,
        (source_stage,)*len(catalog.mutation_ids), (0,)*len(catalog.mutation_ids),
        (0,)*len(catalog.mutation_ids), cpu_threads=1)


def _reject(operation, exception=ValueError):
    try:
        operation()
    except exception:
        return
    raise AssertionError("Invalid assignment evidence was accepted")


def run_checks(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    assert PROTOCOL_VERSION == "2.2"
    exact = _catalog(root, "exact", 2, 3, 4)
    request = _bundle(exact, _reference(exact, [[1.]]))
    result = assign_mutations(request, outdir=root / "exact_output")
    np.testing.assert_allclose(np.exp(result.assignments[0]["log_scores"]), [15/64], rtol=1e-14)
    assert result.multiplicities[0]["conditional_call"] == 3
    np.testing.assert_allclose(result.multiplicities[0]["call_probability"], .45, rtol=1e-14)
    marginal = _catalog(root, "marginal", 1, 2, 2)
    values = assign_mutations(_bundle(marginal, _reference(marginal, [[1.], [.6]]))).assignments[0]
    np.testing.assert_allclose(values["probabilities"], [5/14, 9/14], rtol=1e-14)

    recover = _catalog(root, "recover", 300, 1000, 4)
    reference = _reference(recover, [[1.], [.4]])
    origin_results = []
    for stage in ("stage_one", "stage_two"):
        request = _bundle(recover, reference, source_stage=stage)
        result = assign_mutations(request, outdir=root / stage)
        origin_results.append(result)
        row = result.assignments[0]
        assert row["accepted_label"] == row["final_assignment_label"] == 1
        assert row["baseline_reference_label"] == 0
        assert row["target_reason"] == ("stage1_clonal_recheck" if stage == "stage_one" else "stage2_near_clonal_recheck")
        np.testing.assert_allclose(row["posterior"], .998343610029, atol=1e-12, rtol=0)
        assert result.multiplicities[0]["assigned_multiplicity"] == 3
    np.testing.assert_array_equal(origin_results[0].assignments[0]["log_scores"], origin_results[1].assignments[0]["log_scores"])
    clonal = _catalog(root, "clonal", 500, 1000, 4)
    result = assign_mutations(_bundle(clonal, _reference(clonal, [[1.], [.4]])))
    np.testing.assert_allclose(result.assignments[0]["posterior"], .999999998633, atol=1e-12, rtol=0)
    assert result.assignments[0]["accepted_label"] == 0 and result.multiplicities[0]["assigned_multiplicity"] == 2
    alias = _catalog(root, "alias", 100, 300, 2, minor=1)
    result = assign_mutations(_bundle(alias, _reference(alias, [[1.], [.5]])))
    assert result.assignments[0]["accepted_label"] is None
    np.testing.assert_allclose(result.assignments[0]["probabilities"], [.5, .5], atol=1e-4)

    failed = _bundle(recover, _reference(recover, [[1.], [.4]], [[True], [False]]))
    unresolved = assign_mutations(failed, outdir=root / "failed_reference")
    row = unresolved.assignments[0]
    assert row["status"] == "unresolved_reference" and row["accepted_label"] is None
    assert row["reference_quality_failed_regions"] == ("r0",)
    np.testing.assert_array_equal(row["probabilities"], origin_results[0].assignments[0]["probabilities"])
    assert unresolved.multiplicities[0]["conditional_call"] == 3
    assert np.isnan(unresolved.multiplicities[0]["assigned_multiplicity"])
    partially_observed = _catalog(root, "masked", [[300, 300]], 1000, 4, observed=[[True, False]])
    masked = _bundle(partially_observed, _reference(partially_observed, [[1., 1.], [.4, .4]],
                                                    [[True, True], [True, False]]))
    assert assign_mutations(masked, outdir=root / "masked_output").assignments[0]["status"] == "assigned_relative"
    _reject(lambda: AssignmentReference((0, 1), recover.region_ids, [[1.], [.4]]), TypeError)
    for field in ("regional_quality", "regional_quality_reasons", "center_sources"):
        _reject(lambda field=field: replace(reference, **{field: None}))

    for width in (1, 3):
        zero = _catalog(root, f"zero{width}", 0, 100, width)
        request = _bundle(zero, _reference(zero, [[0.]]))
        result = assign_mutations(request, outdir=root / f"zero_output{width}")
        row = result.multiplicities[0]
        assert row["multiplicity_status"] == "uninformative_zero_ccf"
        assert np.isnan(row["conditional_call"]) and np.isnan(row["assigned_multiplicity"])
        np.testing.assert_allclose(row["conditional_probabilities"], np.full(width, 1/width))
        assert result.assignments[0]["comparison_scope"] == "single_candidate"
        assert result.assignments[0]["entropy"] == 0 and result.assignments[0]["fit_status"] == "not_tested"
        if width > 1:
            changed = result.to_dict()
            changed["multiplicities"][0]["conditional_map"] = width
            _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(changed)))
    singleton = _catalog(root, "singleton", 50, 100, 1, minor=1)
    result = assign_mutations(_bundle(singleton, _reference(singleton, [[1.]])), outdir=root / "singleton_output")
    assert result.multiplicities[0]["conditional_call"] == result.multiplicities[0]["assigned_multiplicity"] == 1
    assert not result.multiplicities[0]["multiplicity_read_informative"]
    missing = _catalog(root, "missing", 0, 0, 1)
    result = assign_mutations(_bundle(missing, _reference(missing, [[1.]])), outdir=root / "missing_output")
    assert result.assignments[0]["status"] == "no_information" and np.isnan(result.assignments[0]["entropy"])

    boundary = np.log1p(-.9)-np.log(.9)
    assert score_summary([0., boundary], (0, 1), .9, 0.)["posterior_pass"]
    assert not score_summary([0., np.nextafter(boundary, np.inf)], (0, 1), .9, 0.)["posterior_pass"]
    assert not score_summary([0., -1000.], (0, 1), 1., 0.)["posterior_pass"]
    assert score_summary([0., -np.inf], (0, 1), 1., 0.)["posterior_pass"]
    for gap in (50., 800.):
        assert not score_summary([0., -gap], (0, 1), 1., 0.)["posterior_pass"]
        assert score_summary([0., -gap], (0, 1), np.nextafter(1., 0.), 0.)["posterior_pass"]
    assert score_summary([0., 0.], (7, 2), .9, 0.)["best_label"] == 2

    request = _bundle(recover, reference)
    assign_mutations(request, outdir=root / "replay_original")
    save_assignment_bundle(request, root / "bundle.json")
    assign_mutations(load_assignment_bundle(root / "bundle.json"), outdir=root / "replay_restored")
    for suffix in ("mutation_assignments.tsv", "assignment_multiplicity.tsv"):
        name = recover.tumor_id + "_" + suffix
        assert (root / "replay_original" / name).read_bytes() == (root / "replay_restored" / name).read_bytes()
    for cloned in (copy.copy(request), copy.deepcopy(request), pickle.loads(pickle.dumps(request))):
        cloned.validate_integrity()
        assert not cloned.reference.regional_quality.flags.writeable
        assert cloned.content_hash == request.content_hash
    changed_quality = copy.deepcopy(request)
    object.__setattr__(changed_quality.reference, "regional_quality", np.zeros(reference.centers.shape, bool))
    _reject(changed_quality.validate_integrity)
    old = request.to_dict()
    old["schema"] = "clipp2.assignment.bundle.v4"
    _reject(lambda: AssignmentBundle.from_dict(old))
    old = request.to_dict()
    old["protocol_version"] = "2.1"
    _reject(lambda: AssignmentBundle.from_dict(old))
    absent = reference.to_dict()
    del absent["regional_quality"]
    _reject(lambda: AssignmentReference.from_dict(absent))
    valid = assign_mutations(request)
    modified = valid.to_dict()
    modified["assignments"][0]["log_scores"] = [score + 1. for score in modified["assignments"][0]["log_scores"]]
    _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(modified)))
    modified = valid.to_dict()
    modified["multiplicities"][0]["assigned_multiplicity"] = 4
    _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(modified)))
    modified = unresolved.to_dict()
    modified["assignments"][0]["reference_quality_pass"] = True
    _reject(lambda: validate_assignment_result(failed, AssignmentResult.from_dict(modified)))
    _reject(lambda: AssignmentThresholds(.9, DescriptiveScreen(10., 5., .05), 1e-10, 1e-10, .95,
                                        fit_assessment="bootstrap"))
    combined_catalog = _catalog(root, "combined", [[300], [300], [300]], 1000, 4)
    complete = _bundle(combined_catalog, _reference(combined_catalog, [[1.], [.4]]))
    combined_request = replace(complete, target_ids=complete.target_ids[:2], non_target_ids=complete.target_ids[2:],
        native_centers=complete.native_centers[:2], native_calls=complete.native_calls[:2],
        native_lower=complete.native_lower[:2], native_upper=complete.native_upper[:2],
        source_stages=("stage_one", "stage_two"), source_labels=(0, 0), baseline_labels=(0, 0),
        run_manifest_id="external-original")
    combined_result = assign_mutations(combined_request, run_manifest_id="external-current")
    assert combined_result.run_manifest_id == combined_result.summary["evaluation_receipt"]["run_manifest_id"] == "external-current"
    assert combined_result.summary["bundle_run_manifest_id"] == "external-original"
    assert combined_request.validated_payload()["run_manifest_id"] == "external-original"
    with patch.dict(os.environ, {}, clear=True):
        current = assign_mutations(combined_request, outdir=root / "unbound_execution")
        assert current.run_manifest_id is None
        assert current.summary["evaluation_receipt"]["run_manifest_id"] is None
        assert current.summary["bundle_run_manifest_id"] == "external-original"
    with patch.dict(os.environ, {"CLIPP2_RUN_MANIFEST_ID": "environment-current"}):
        current = assign_mutations(combined_request, outdir=root / "environment_execution")
        assert current.run_manifest_id == "environment-current"
        assert AssignmentResult.from_dict(current.to_dict()).run_manifest_id == "environment-current"
        assert assign_mutations(combined_request, run_manifest_id="explicit-current").run_manifest_id == "explicit-current"
    with patch.dict(os.environ, {"CLIPP2_RUN_MANIFEST_ID": " malformed "}):
        _reject(lambda: assign_mutations(combined_request))
        assert assign_mutations(combined_request, run_manifest_id="explicit-current").run_manifest_id == "explicit-current"
    _reject(lambda: assign_mutations(combined_request, run_manifest_id=" malformed "))
    assert combined_request.to_dict()["schema"] == "clipp2.assignment.bundle.v3"
    assert combined_result.to_dict()["schema"] == "clipp2.assignment.result.v4"
    assert combined_result.summary["support_policy"] == "original_full_1_to_major_cn"
    assert all("fit_screen_status" in row and "descriptive_fit_status" not in row for row in combined_result.assignments)
    accounting = dict(status="complete", unchanged_map={complete.target_ids[2]: 1}, population_count=3,
        final_occupancy={"0": 0, "1": 3}, unchanged_non_target_occupancy={"0": 0, "1": 1}, unresolved_targets=0,
        outcomes_by_origin={origin: dict(retained_clonal=0, reassigned=1, unresolved=0)
            for origin in ("stage1_clonal_recheck", "stage2_near_clonal_recheck")}, historical_caps_recomputed=False)
    for changed in (dict(accounting, population_count=4), dict(accounting, population_count=3.),
                    dict(accounting, unchanged_map={}), dict(accounting, final_occupancy={"0": 1, "1": 2})):
        _reject(lambda: publish_assignment(combined_request, combined_result, root / "invalid_accounting",
                                            workflow_accounting=changed))
        assert not (root / "invalid_accounting").exists()
    published = publish_assignment(combined_request, combined_result, root / "combined_output", workflow_accounting=accounting)
    assert published["population_scope"] == "workflow_combined" and published["workflow_accounting"] == accounting
    assert combined_result.summary["population_scope"] == "targets_only"
    for clone in (copy.copy(combined_result), copy.deepcopy(combined_result), pickle.loads(pickle.dumps(combined_result))):
        validate_assignment_result(combined_request, clone)
        assert not clone.assignments[0]["probabilities"].flags.writeable
        assert clone.run_manifest_id == "external-current"
    stale = combined_result.to_dict()
    stale["run_manifest_id"] = "different-current"
    _reject(lambda: validate_assignment_result(combined_request, AssignmentResult.from_dict(stale)))
    stale = combined_result.to_dict()
    stale["schema"] = "clipp2.assignment.result.v3"
    _reject(lambda: AssignmentResult.from_dict(stale))
    for baseline, changed in ((1, False), (0, True)):
        generic = replace(_bundle(recover, reference, source_stage="generic"), baseline_labels=(baseline,))
        generic_result = assign_mutations(generic, outdir=root / f"generic_{baseline}")
        assert generic_result.assignments[0]["reassessment_status"] == "not_applicable"
        assert generic_result.assignments[0]["baseline_changed"] is changed
        assert generic_result.summary["reassessment_counts"]["not_applicable"] == 1
        tampered = generic_result.to_dict()
        tampered["summary"]["reassessment_counts"]["not_applicable"] = 0
        _reject(lambda: validate_assignment_result(generic, AssignmentResult.from_dict(tampered)))
        _reject(lambda: publish_assignment(generic, generic_result, root / "generic_workflow", workflow_accounting={}))
    generic_unresolved = replace(failed, source_stages=("generic",))
    row = assign_mutations(generic_unresolved, outdir=root / "generic_unresolved").assignments[0]
    assert row["reassessment_status"] == "not_applicable" and row["baseline_changed"] is None
    no_clonal = _bundle(recover, _reference(recover, [[.4]]))
    assert assign_mutations(no_clonal, outdir=root / "no_clonal").assignments[0]["reassessment_status"] == "not_applicable"
    for stage in ("stage_one", "stage_two", "stage1", "stage2"):
        _reject(lambda stage=stage: replace(no_clonal, source_stages=(stage,)))
        _reject(lambda stage=stage: replace(_bundle(recover, reference), source_stages=(stage,), baseline_labels=(1,)))
    empty = replace(no_clonal, target_ids=(), non_target_ids=no_clonal.target_ids,
        native_centers=no_clonal.native_centers[:0], native_calls=no_clonal.native_calls[:0],
        native_lower=no_clonal.native_lower[:0], native_upper=no_clonal.native_upper[:0],
        source_stages=(), source_labels=(), baseline_labels=())
    empty_result = assign_mutations(empty, outdir=root / "empty_targets")
    assert sum(empty_result.summary["reassessment_counts"].values()) == 0
    return dict(protocol_version=PROTOCOL_VERSION, absolute_marginal_mass=True,
        both_target_origins_identical=True, recovery_and_clonal_controls=True, alias_unresolved=True,
        quality_preserves_posteriors=True, quality_uses_common_observed_mask=True,
        explicit_conditional_and_assigned_multiplicity=True, stable_competitor_confidence=True,
        immutable_quality_roundtrip=True, byte_identical_bundle_replay=True, independent_publication_checks=True,
        combined_publication_once=True, original_and_current_manifest_links=True, exact_schema_migration=True,
        generic_and_workflow_outcomes=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="clipp2-assignment-v22-") as directory:
        print(json.dumps(run_checks(directory), sort_keys=True))
