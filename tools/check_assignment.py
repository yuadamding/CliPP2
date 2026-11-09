#!/usr/bin/env python3
"""Focused Protocol 2.3 assignment contracts against the imported installation."""
from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal, localcontext
import json
import os
import io
import zipfile
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2.assignment import (AssignmentBundle, AssignmentReference, AssignmentResult,
                               assign_mutations, score_summary)
from CliPP2.config import PROTOCOL_VERSION
from CliPP2.io.assignment import (load_assignment_bundle, publish_assignment, save_assignment_bundle,
                                  validate_assignment_result)
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.screening import binomial_profile_deviance, profile_deviance_from_diagnostics


def _catalog(root, name, alt, depth, major, *, minor=0, observed=True, input_excluded=False):
    alt, depth, major, minor, observed, input_excluded = np.broadcast_arrays(
        np.atleast_2d(alt), np.atleast_2d(depth), np.atleast_2d(major),
        np.atleast_2d(minor), np.atleast_2d(observed), np.atleast_2d(input_excluded))
    rows = [dict(mutation_id=f"m{i}", sample_id=f"r{j}", alt_count=int(alt[i, j]),
        ref_count=int(depth[i, j]-alt[i, j]), count_observed=int(observed[i, j]),
        input_excluded=int(input_excluded[i, j]), purity=1., normal_cn=2,
        segment_id="s", cn_state_id="0", cn_state_fraction=1., allele_a_cn=int(major[i, j]),
        allele_b_cn=int(minor[i, j])) for i in range(alt.shape[0]) for j in range(alt.shape[1])]
    path = write_tumor_txt(root / (name + ".tsv"), pd.DataFrame(rows))
    return load_original_catalog(path)


def _reference(catalog, centers, *, measurements=None):
    centers = np.asarray(centers, float)
    return AssignmentReference(tuple(range(len(centers))), catalog.region_ids, centers,
        clonal_label=0 if np.all(centers[0] == 1) else None,
        center_sources=tuple(dict(center_source="declared_numerical_fixture", source_stage="fixture", source_label=i)
                             for i in range(len(centers))), **(measurements or {}))


def _bundle(catalog, reference, *, source_stage=None):
    if source_stage is None:
        source_stage = "stage_one" if reference.clonal_label is not None else "fixture"
    bounds = catalog.default_bounds()
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    return AssignmentBundle(catalog, reference, catalog.mutation_ids, (),
        np.ones(informative.shape), np.where(informative, 1., np.nan), bounds.lower, bounds.upper,
        (source_stage,)*len(catalog.mutation_ids), (0,)*len(catalog.mutation_ids),
        (0,)*len(catalog.mutation_ids), cpu_threads=1)


def _reject(operation, exception=ValueError):
    try:
        operation()
    except exception:
        return
    raise AssertionError("Invalid assignment evidence was accepted")


def _numerical_screen_checks(root):
    from CliPP2.io.assignment import _independent_deviance

    def oracle(alt, depth, probability):
        if probability == alt/depth:
            return 0.
        if probability in (0., 1.):
            return np.inf
        with localcontext() as context:
            context.prec = 400
            q, p = Decimal.from_float(float(alt/depth)), Decimal.from_float(float(probability))
            terms = (q*(q/p).ln() if q else Decimal(0))
            terms += ((1-q)*((1-q)/(1-p)).ln() if q != 1 else Decimal(0))
            return float(2*Decimal(int(depth))*terms)

    cases = [(50, 100), (1, 3), (1, 10), (9, 10), (1, 10**12), (10**12-1, 10**12), (0, 100), (100, 100)]
    for alt, depth in cases:
        q = alt/depth
        for probability in (q, np.nextafter(q, 0.), np.nextafter(q, 1.), .2, .8,
                            np.nextafter(0., 1.), np.nextafter(1., 0.)):
            expected = oracle(alt, depth, probability)
            actual = float(binomial_profile_deviance(alt, depth, probability))
            independent = _independent_deviance(alt, depth, probability)
            np.testing.assert_allclose(actual, expected, rtol=5e-14, atol=0.)
            np.testing.assert_allclose(independent, expected, rtol=5e-14, atol=0.)
            if np.isfinite(expected) and probability != q:
                assert actual > 0 and independent > 0
    assert binomial_profile_deviance(50, 100, .5) == 0
    adjacent = float(binomial_profile_deviance(50, 100, np.nextafter(.5, 1.)))
    np.testing.assert_allclose(adjacent, 4.930380657631324e-30, rtol=1e-15, atol=0.)
    assert np.isposinf(binomial_profile_deviance(50, 100, np.nan))
    assert np.isnan(binomial_profile_deviance(50, 100, .5, False))
    assert np.isnan(binomial_profile_deviance(0, 0, .5))
    observation = SimpleNamespace(alt=np.array([[50., 1.]]), depth=np.array([[100., 10.]]),
        scaling=np.array([[.25, 1.]]), observed=np.ones((1, 2), bool))
    diagnostic = {"map_multiplicity": np.array([[2., np.nan]])}
    np.testing.assert_array_equal(profile_deviance_from_diagnostics(observation, 1., diagnostic), [[0., np.inf]])
    from CliPP2.screening import summarize_profile_deviance
    measured = summarize_profile_deviance([[0., np.inf], [np.nan, np.nan]],
                                         [[True, True], [False, False]], axis=1)
    np.testing.assert_array_equal(measured["informative_count"], [2, 0])
    assert np.isposinf(measured["mean_profile_deviance"][0])
    assert np.isposinf(measured["maximum_profile_deviance"][0])
    assert np.isnan(measured["mean_profile_deviance"][1])
    assert np.isnan(measured["maximum_profile_deviance"][1])
    catalog = _catalog(root, "deviance_boundary", 50, 100, 1)
    for name, center in (("exact", .5), ("adjacent", np.nextafter(.5, 1.))):
        request = _bundle(catalog, _reference(catalog, [[center]]))
        result = assign_mutations(request, outdir=root / ("deviance_" + name))
        assert result.assignments[0]["status"] == "observed"
        assert result.multiplicities[0]["profile_deviance"] == (0. if name == "exact" else adjacent)
        changed = result.to_dict()
        changed["multiplicities"][0]["profile_deviance"] = adjacent if name == "exact" else 0.
        _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(changed)))
    return dict(high_precision_oracle=True, exact_saturation=True, adjacent_positive=True,
                infinite_and_empty_domains=True, independent_publication=True)


def _loss_only_checks(root):
    """Compare native loss-only candidate scores against the former diagnostic path."""
    from CliPP2.kernel.backend import NativeBackend
    from CliPP2.io.data import content_hash
    node_terms, diagnostics = NativeBackend.node_terms, NativeBackend.component_diagnostics
    cases = (("tie", [[1, 1]], 3, 2, True, [[.6, .6], [.6, .6]]),
             ("missing", [[1, 0]], 3, 2, [[True, False]], [[1., 0.], [.6, 1.]]),
             ("impossible", [[1]], 2, 1, True, [[0.], [1.]]),
             ("high_support", [[4]], 10, 8, True, [[1.], [.5]]),
             ("uninformative", [[1, 1]], 3, 2, False, [[1., 1.], [.5, .5]]))
    for name, alt, depth, major, observed, centers in cases:
        catalog = _catalog(root, "loss_only_"+name, alt, depth, major, observed=observed)
        bundle = _bundle(catalog, _reference(catalog, centers))
        calls = dict(loss=0, diagnostics=0)

        def loss_only(self, coordinates, derivatives=True):
            assert derivatives is False
            calls["loss"] += 1
            return node_terms(self, coordinates, derivatives=False)

        def selected(self, *args, **kwargs):
            calls["diagnostics"] += 1
            return diagnostics(self, *args, **kwargs)

        with patch.object(NativeBackend, "node_terms", loss_only), patch.object(NativeBackend, "component_diagnostics", selected):
            current = assign_mutations(bundle)
        assert calls == dict(loss=len(centers), diagnostics=1)

        def old_candidate(self, coordinates, derivatives=False):
            assert derivatives is False
            result = diagnostics(self, coordinates)
            return np.where(result["status"] == 4, 0., -result["log_emission"])

        with patch.object(NativeBackend, "node_terms", old_candidate):
            previous = assign_mutations(bundle)
        for before, after in zip(previous.assignments, current.assignments):
            np.testing.assert_array_equal(before["log_scores"], after["log_scores"])
            np.testing.assert_array_equal(before["probabilities"], after["probabilities"])
            for field in ("status", "final_assignment_label", "best_label", "map_unique"):
                assert before[field] == after[field]
        assert content_hash(previous.to_dict()["multiplicities"]) == content_hash(current.to_dict()["multiplicities"])
    return dict(cases=len(cases), candidate_scores_exact=True, decisions_exact=True,
                selected_diagnostics_once=True)


def _scalar_cn_checks(root):
    catalog = _catalog(root, "scalar_cn", [[20, 20], [8, 8], [20, 20], [8, 8]], 100, 4, minor=1)
    request = _bundle(catalog, _reference(catalog, [[1., 1.], [.4, .4]]))

    def forbidden(self):
        raise AssertionError("Scalar assignment/publication built a whole-catalog total-CN array")

    with patch.object(type(catalog), "total_cn", property(forbidden)):
        result = assign_mutations(request, outdir=root / "scalar_cn_output")
    assert all(row["original_total_cn"] == 5. for row in result.multiplicities)
    return dict(whole_catalog_sums=0, assignment_and_independent_publication=True)


def _fixed_reference_baseline(root):
    """Reinterpret a separately identified 2.2 bundle without altering historical files."""
    from CliPP2.io.data import content_hash
    with zipfile.ZipFile(Path(__file__).parent / "fixtures/protocol22_baseline.zip") as archive:
        historical = json.loads(archive.read("assignment_bundle.json"))
        assert content_hash(historical["payload"]) == historical["content_hash"]
        payload = copy.deepcopy(historical["payload"])
        assert payload["protocol_version"] == "2.2"
        payload.update(schema="clipp2.assignment.bundle.v4", protocol_version="2.3")
        payload.pop("thresholds")
        reference = payload["reference"]
        reference.pop("regional_quality")
        reference.pop("regional_quality_reasons")
        for name in ("source_informative_count", "source_informative_depth", "source_mean_profile_deviance",
                     "source_maximum_profile_deviance", "diagnostic_availability"):
            reference[name] = None
        request = AssignmentBundle.from_dict(payload)
        previous = pd.read_csv(io.BytesIO(archive.read("assignment/workflow_protocol22_mutation_assignments.tsv")),
                               sep="\t", float_precision="round_trip").set_index("mutation_id")
        previous_pairs = pd.read_csv(io.BytesIO(archive.read("assignment/workflow_protocol22_assignment_multiplicity.tsv")),
                                     sep="\t", float_precision="round_trip").set_index(["mutation_id", "region_id"])
    current = assign_mutations(request, outdir=root / "protocol22_fixed_reference")
    assert request.content_hash != historical["content_hash"]
    for row in current.assignments:
        before = previous.loc[row["mutation_id"]]
        np.testing.assert_array_equal(row["log_scores"], [before[f"log_score_{label}"] for label in request.reference.labels])
        np.testing.assert_array_equal(row["probabilities"], [before[f"probability_{label}"] for label in request.reference.labels])
        np.testing.assert_allclose(row["mean_profile_deviance"], before["mean_profile_deviance"], rtol=1e-12, atol=0.)
        assert row["final_assignment_label"] == row["map_cluster_label"] and row["evidence_status"] == "observed"
    for row in current.multiplicities:
        before = previous_pairs.loc[row["mutation_id"], row["region_id"]]
        for key in ("conditional_map", "call_probability", "profile_deviance"):
            np.testing.assert_allclose(row[key], before[key], rtol=1e-12, atol=0.)
    return dict(targets=len(current.target_ids), scores_exact=True, probabilities_exact=True,
                diagnostic_measurements_preserved=True, separately_identified_derivative=True)


def _accounting_checks(root):
    catalog = _catalog(root, "combined", [[300], [0], [300]], [[1000], [0], [1000]], 4)
    full = _bundle(catalog, _reference(catalog, [[1.], [.4], [.2]]))
    request = replace(full, target_ids=full.target_ids[:2], non_target_ids=full.target_ids[2:],
        native_centers=full.native_centers[:2], native_calls=full.native_calls[:2],
        native_lower=full.native_lower[:2], native_upper=full.native_upper[:2],
        source_stages=("stage_one", "stage_two"), source_labels=(0, 0), baseline_labels=(0, 0))
    result = assign_mutations(request)
    assert [row["evidence_status"] for row in result.assignments] == ["observed", "prior_only"]
    accounting = dict(status="complete", unchanged_map={full.target_ids[2]: 1}, population_count=3,
        final_occupancy={"0": 0, "1": 2, "2": 0}, unchanged_non_target_occupancy={"0": 0, "1": 1, "2": 0},
        unresolved_targets=1, evidence_status_counts=dict(observed=1, prior_only=1, model_incompatible=0, excluded=0),
        outcomes_by_origin={"stage1_reference_recheck": dict(retained_clonal=0, reassigned=1, unresolved=0),
                            "stage2_near_clonal_recheck": dict(retained_clonal=0, reassigned=0, unresolved=1)},
        historical_caps_recomputed=False)
    for wrong in (dict(accounting, population_count=4), dict(accounting, unchanged_map={}),
                  dict(accounting, final_occupancy={"0": 1, "1": 2, "2": 0}),
                  dict(accounting, evidence_status_counts=dict(observed=2, prior_only=0, model_incompatible=0, excluded=0))):
        _reject(lambda: publish_assignment(request, result, root / "invalid_accounting", workflow_accounting=wrong))
        assert not (root / "invalid_accounting").exists()
    published = publish_assignment(request, result, root / "combined_output", workflow_accounting=accounting)
    assert published["workflow_accounting"] == accounting and published["population_scope"] == "workflow_combined"
    assert result.reference.labels == (0, 1, 2) and result.summary["population_scope"] == "targets_only"
    return dict(unchanged_plus_observed_map=True, prior_only_not_counted_as_map=True,
                zero_occupancy_reference_retained=True, tampered_counts_rejected=True)


def run_checks(directory):
    from CliPP2.kernel.backend import NativeBackend
    from CliPP2.io.data import content_hash
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    assert PROTOCOL_VERSION == "2.3"
    numerical = _numerical_screen_checks(root)
    loss_only = _loss_only_checks(root)
    scalar_cn = _scalar_cn_checks(root)
    baseline = _fixed_reference_baseline(root)
    accounting = _accounting_checks(root)
    exact = _catalog(root, "exact", 2, 3, 4)
    request = _bundle(exact, _reference(exact, [[1.]]))
    result = assign_mutations(request, outdir=root / "exact_output")
    np.testing.assert_allclose(np.exp(result.assignments[0]["log_scores"]), [15/64], rtol=1e-14)
    assert result.multiplicities[0]["conditional_call"] == 3
    np.testing.assert_allclose(result.multiplicities[0]["call_probability"], .45, rtol=1e-14)
    marginal = _catalog(root, "marginal", 1, 2, 2)
    weak = assign_mutations(_bundle(marginal, _reference(marginal, [[1.], [.6]])))
    row = weak.assignments[0]
    np.testing.assert_allclose(row["probabilities"], [5/14, 9/14], rtol=1e-14)
    assert row["posterior"] < .9 and row["final_assignment_label"] == 1
    integer_tie = _catalog(root, "integer_tie", 3, 6, 2, minor=1)
    tied_call = assign_mutations(_bundle(integer_tie, _reference(integer_tie, [[1.]])),
                                outdir=root / "integer_tie_output").multiplicities[0]
    assert tied_call["map_tied_multiplicities"] == (1, 2) and not tied_call["map_unique"]
    assert tied_call["assigned_multiplicity"] == tied_call["conditional_call"] == 1

    for scores, labels, call, ties in (([0., 0.], (7, 2), 2, (2, 7)),
            ([0., np.nextafter(0., 1.)], (0, 1), 1, (1,)), ([0., -1000.], (0, 1), 0, (0,))):
        summary = score_summary(scores, labels)
        assert summary["map_cluster_label"] == call and summary["map_tied_labels"] == ties
        assert summary["map_unique"] == (len(ties) == 1)
    assert score_summary([0., -1000.], (0, 1))["has_finite_alternative"]
    assert score_summary([0., -1000.], (0, 1))["log_competitor_ratio"] == -1000
    tied = assign_mutations(_bundle(marginal, _reference(marginal, [[.6], [.6]])), outdir=root / "tied")
    assert tied.assignments[0]["map_tied_labels"] == (0, 1)
    assert tied.assignments[0]["final_assignment_label"] == 0
    tied_request = _bundle(marginal, _reference(marginal, [[.6], [.6]]))
    for values in ([0., 1.], [0, 1.00000000005], [False, 1]):
        changed = tied.to_dict()
        changed["assignments"][0]["map_tied_labels"] = values
        _reject(lambda: validate_assignment_result(tied_request, AssignmentResult.from_dict(changed)))
    for field in ("map_cluster_label", "best_label", "final_assignment_label", "baseline_reference_label"):
        for value in (0., False):
            changed = tied.to_dict()
            changed["assignments"][0][field] = value
            _reject(lambda: AssignmentResult.from_dict(changed))
    for field in ("map_tied_multiplicities", "multiplicity_candidates"):
        changed = result.to_dict()
        changed["multiplicities"][0][field] = [float(value) for value in changed["multiplicities"][0][field]]
        _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(changed)))

    recover = _catalog(root, "recover", [[300], [500]], 1000, 4)
    reference = _reference(recover, [[1.], [.4]])
    request = _bundle(recover, reference)
    result = assign_mutations(request, outdir=root / "origin_one")
    other = assign_mutations(replace(request, source_stages=("stage_two",)*2), outdir=root / "origin_two")
    assert [row["final_assignment_label"] for row in result.assignments] == [1, 0]
    for one, two in zip(result.assignments, other.assignments):
        np.testing.assert_array_equal(one["log_scores"], two["log_scores"])
        np.testing.assert_array_equal(one["probabilities"], two["probabilities"])
    poor_reference = replace(reference, source_informative_count=[[0], [1]], source_informative_depth=[[0], [1]],
        source_mean_profile_deviance=[[np.nan], [1e10]], source_maximum_profile_deviance=[[np.nan], [np.inf]],
        diagnostic_availability=(("missing",), ("available",)))
    poor = assign_mutations(replace(request, reference=poor_reference), outdir=root / "poor_reference")
    _reject(lambda: replace(reference, diagnostic_availability=(("available",), ("available",))))
    for before, after in zip(result.assignments, poor.assignments):
        np.testing.assert_array_equal(before["probabilities"], after["probabilities"])
        assert before["final_assignment_label"] == after["final_assignment_label"]
    large = _catalog(root, "large_deviance", 90, 100, 1)
    large_result = assign_mutations(_bundle(large, _reference(large, [[.01]])), outdir=root / "large_output")
    assert large_result.assignments[0]["posterior"] == 1 and large_result.assignments[0]["final_assignment_label"] == 0
    assert large_result.assignments[0]["mean_profile_deviance"] > 100

    for name, depth, observed, excluded, centers, state in (
            ("prior", 0, True, False, [[1.], [.5]], "prior_only"),
            ("prior_singleton", 0, True, False, [[1.]], "prior_only"),
            ("unobserved", 100, False, False, [[1.], [.5]], "prior_only"),
            ("excluded", 100, True, True, [[1.], [.5]], "excluded"),
            ("impossible", 100, True, False, [[0.]], "model_incompatible")):
        catalog = _catalog(root, name, 0 if depth == 0 else 1, depth, 1, observed=observed, input_excluded=excluded)
        empty_request = _bundle(catalog, _reference(catalog, centers))
        values = assign_mutations(empty_request, outdir=root / (name+"_output")).assignments[0]
        assert values["evidence_status"] == state and values["final_assignment_label"] is None
        if state == "prior_only":
            np.testing.assert_array_equal(values["log_scores"], np.full(len(centers), -np.log(len(centers))))
            np.testing.assert_array_equal(values["probabilities"], np.full(len(centers), 1/len(centers)))
            assert values["map_cluster_label"] == 0 and values["map_unique"] == (len(centers) == 1)
        else:
            assert np.isnan(values["probabilities"]).all() and values["map_cluster_label"] is None
    for width in (1, 3):
        zero = _catalog(root, f"zero{width}", 0, 100, width)
        zero_result = assign_mutations(_bundle(zero, _reference(zero, [[0.]])), outdir=root / f"zero_output{width}")
        pair = zero_result.multiplicities[0]
        assert pair["multiplicity_status"] == "uninformative_zero_ccf"
        assert np.isnan(pair["conditional_call"]) and not pair["multiplicity_read_informative"]
        np.testing.assert_array_equal(pair["conditional_probabilities"], np.full(width, 1/width))
    singleton = _catalog(root, "singleton", 50, 100, 1, minor=1)
    pair = assign_mutations(_bundle(singleton, _reference(singleton, [[1.]]))).multiplicities[0]
    assert pair["multiplicity_status"] == "structural_singleton" and pair["assigned_multiplicity"] == 1
    assert pair["multiplicity_read_informative"]
    incompatible = _catalog(root, "partial_incompatibility", [[1, 10]], 10, 1)
    incompatible_result = assign_mutations(_bundle(incompatible, _reference(incompatible, [[0., 1.]])),
                                          outdir=root / "partial_incompatibility_output")
    assert incompatible_result.assignments[0]["evidence_status"] == "model_incompatible"
    assert incompatible_result.assignments[0]["diagnostic_status"] == "not_applicable"
    for pair in incompatible_result.multiplicities:
        assert np.isnan(pair["phi"]) and np.isnan(pair["profile_deviance"])
        assert pair["diagnostic_status"] == "not_applicable"

    with patch.object(NativeBackend, "component_diagnostics", side_effect=RuntimeError("diagnostic fixture failure")):
        failed = assign_mutations(request, outdir=root / "diagnostic_failure")
    for one, two in zip(result.assignments, failed.assignments):
        np.testing.assert_array_equal(one["log_scores"], two["log_scores"])
        np.testing.assert_array_equal(one["probabilities"], two["probabilities"])
        assert one["final_assignment_label"] == two["final_assignment_label"]
        assert two["diagnostic_status"] == "failed" and two["diagnostic_error"]
        assert np.isnan(two["mean_profile_deviance"])
    with patch.object(NativeBackend, "node_terms", side_effect=RuntimeError("score fixture failure")):
        _reject(lambda: assign_mutations(request), RuntimeError)
    with patch("CliPP2.assignment.binomial_profile_deviance", side_effect=RuntimeError("residual fixture failure")):
        residual_failed = assign_mutations(request, outdir=root / "residual_failure")
    for before, after in zip(result.multiplicities, residual_failed.multiplicities):
        assert before["conditional_map"] == after["conditional_map"]
        np.testing.assert_array_equal(before["conditional_probabilities"], after["conditional_probabilities"])
        assert after["conditional_distribution_defined"] and after["diagnostic_status"] == "failed"
        assert np.isnan(after["profile_deviance"])
    tampered = residual_failed.to_dict()
    tampered["assignments"][0]["final_assignment_label"] = 0
    _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(tampered)))
    tampered = residual_failed.to_dict()
    tampered["multiplicities"][0]["profile_deviance"] = 0.
    _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(tampered)))
    for value in (np.nan, -1.):
        with patch("CliPP2.assignment.binomial_profile_deviance", return_value=value):
            invalid_diagnostic = assign_mutations(request)
        validate_assignment_result(request, invalid_diagnostic)
        for before, after in zip(result.assignments, invalid_diagnostic.assignments):
            np.testing.assert_array_equal(before["probabilities"], after["probabilities"])
            assert before["final_assignment_label"] == after["final_assignment_label"]
            assert after["diagnostic_status"] == "failed"
        for pair in invalid_diagnostic.multiplicities:
            assert pair["conditional_distribution_defined"] and np.isnan(pair["profile_deviance"])
    order = np.array([1, 0])
    reordered = replace(request, target_ids=tuple(request.target_ids[i] for i in order),
        native_centers=request.native_centers[order], native_calls=request.native_calls[order],
        native_lower=request.native_lower[order], native_upper=request.native_upper[order],
        source_stages=tuple(request.source_stages[i] for i in order),
        source_labels=tuple(request.source_labels[i] for i in order), baseline_labels=tuple(request.baseline_labels[i] for i in order))
    permuted = assign_mutations(reordered)
    by_id = {row["mutation_id"]: row for row in result.assignments}
    for row in permuted.assignments:
        assert content_hash(row) == content_hash(by_id[row["mutation_id"]])
    for i, mutation in enumerate(request.target_ids):
        single = replace(request, target_ids=(mutation,), native_centers=request.native_centers[i:i+1],
            native_calls=request.native_calls[i:i+1], native_lower=request.native_lower[i:i+1], native_upper=request.native_upper[i:i+1],
            source_stages=(request.source_stages[i],), source_labels=(request.source_labels[i],), baseline_labels=(request.baseline_labels[i],))
        assert content_hash(assign_mutations(single).assignments[0]) == content_hash(by_id[mutation])

    save_assignment_bundle(request, root / "bundle.json")
    assign_mutations(load_assignment_bundle(root / "bundle.json"), outdir=root / "restored")
    for suffix in ("mutation_assignments.tsv", "assignment_multiplicity.tsv"):
        name = recover.tumor_id + "_" + suffix
        assert (root / "origin_one" / name).read_bytes() == (root / "restored" / name).read_bytes()
    for clone in (copy.copy(request), copy.deepcopy(request), pickle.loads(pickle.dumps(request))):
        clone.validate_integrity()
        assert not clone.reference.source_informative_count.flags.writeable
        assert clone.content_hash == request.content_hash
    mutated = copy.deepcopy(request)
    object.__setattr__(mutated.reference, "source_informative_count", np.zeros(reference.centers.shape))
    _reject(mutated.validate_integrity)
    for field, value in (("schema", "clipp2.assignment.bundle.v3"), ("protocol_version", "2.2")):
        old = request.to_dict()
        old[field] = value
        _reject(lambda: AssignmentBundle.from_dict(old))
    for change in ("scores", "label", "tie", "multiplicity", "deprecated"):
        changed = result.to_dict()
        if change == "scores":
            changed["assignments"][0]["log_scores"] = [score+1 for score in changed["assignments"][0]["log_scores"]]
        elif change == "label":
            changed["assignments"][0]["final_assignment_label"] = 0
        elif change == "tie":
            changed["assignments"][0]["map_tied_labels"] = [0, 1]
        elif change == "multiplicity":
            changed["multiplicities"][0]["assigned_multiplicity"] = 4
        else:
            changed["assignments"][0]["posterior_pass"] = True
        _reject(lambda: validate_assignment_result(request, AssignmentResult.from_dict(changed)))
    for clone in (copy.copy(result), copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        validate_assignment_result(request, clone)
        assert not clone.assignments[0]["probabilities"].flags.writeable
    with patch.dict(os.environ, {"CLIPP2_RUN_MANIFEST_ID": "current"}):
        current = assign_mutations(replace(request, run_manifest_id="original"))
        assert current.run_manifest_id == "current" and current.summary["bundle_run_manifest_id"] == "original"
    _reject(lambda: assign_mutations(request, run_manifest_id=" invalid "))
    _reject(lambda: publish_assignment(request, result, root / "origin_one"), FileExistsError)
    empty = replace(request, target_ids=(), non_target_ids=request.target_ids,
        native_centers=request.native_centers[:0], native_calls=request.native_calls[:0],
        native_lower=request.native_lower[:0], native_upper=request.native_upper[:0],
        source_stages=(), source_labels=(), baseline_labels=())
    empty_result = assign_mutations(empty, outdir=root / "empty_targets")
    assert sum(empty_result.summary["assigned_occupancy"].values()) == 0
    return dict(protocol_version=PROTOCOL_VERSION, fixed_reference_baseline=baseline,
        low_confidence_map=True, exact_and_near_ties=True, finite_competitor_underflow=True,
        diagnostic_measurements_do_not_veto=True, prior_only_distinct_from_incompatible_and_excluded=True,
        conditional_map_without_concentration_gate=True, diagnostic_failure_preserves_scoring=True,
        computational_failure_explicit=True, origin_order_batch_invariance=True,
        immutable_roundtrip_and_tampering=True, byte_identical_replay=True,
        numerical_deviance=numerical, loss_only_scoring=loss_only, scalar_cn=scalar_cn,
        whole_population_accounting=accounting)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="clipp2-assignment-") as directory:
        print(json.dumps(run_checks(directory), sort_keys=True))
