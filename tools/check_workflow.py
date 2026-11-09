#!/usr/bin/env python3
"""Deterministic Protocol 2.3 CPU contracts against one imported installation.

Fixture cutoffs exercise decisions; they are not biological calibration advice.
All inputs and outputs are temporary. No checkout import path is injected.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2 import FitConfig, process_tumor, run_workflow
from CliPP2.assignment import assign_mutations
from CliPP2.cli import main as cli_main
from CliPP2.config import PROTOCOL_VERSION
from CliPP2.io.assignment import (load_accepted_fit, load_assignment_bundle, publish_assignment,
                                 save_assignment_bundle)
from CliPP2.io.data import MultiplicityBounds, OriginalCatalog, content_hash
from CliPP2.io.region_files import read_region_directory
from CliPP2.io.tumor_txt import load_original_catalog, load_tumor_txt, write_tumor_txt
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction

from check_support import (write_catalog, fixture_reference, workflow_config,
                           bundle, expect_value_error, assert_published_files)










def check_purity_consistency(directory):
    """Mean-tolerant source values retain one canonical input across every route."""
    low, high = .799999999925, .800000000075
    frame = pd.DataFrame([dict(mutation_id=f"m{i}", sample_id="r1", alt_count=20,
        ref_count=80, count_observed=1, purity=purity, normal_cn=2, segment_id="s",
        cn_state_id="0", cn_state_fraction=1, allele_a_cn=1, allele_b_cn=1)
        for i, purity in enumerate((low, high))])
    path = write_tumor_txt(directory / "purity.tsv", frame)
    ordinary, catalog = load_tumor_txt(path), load_original_catalog(path)
    np.testing.assert_array_equal(ordinary.purity, np.full((2, 1), low))
    np.testing.assert_array_equal(catalog.purity, ordinary.purity)
    np.testing.assert_array_equal(catalog.fit_data().purity, ordinary.purity)
    assert [float(dict(row)["purity"]) for row in catalog.source_rows] == [low, high]
    restored = OriginalCatalog.from_dict(catalog.to_dict())
    assert restored.content_hash == catalog.content_hash
    roundtrip = write_tumor_txt(directory / "purity_roundtrip.tsv",
        pd.DataFrame([dict(row) for row in catalog.source_rows]), dict(catalog.metadata))
    roundtrip_catalog = load_original_catalog(roundtrip)
    assert roundtrip_catalog.source_rows == catalog.source_rows
    assert dict(roundtrip_catalog.metadata) == dict(catalog.metadata)
    np.testing.assert_array_equal(roundtrip_catalog.purity, catalog.purity)
    subset = catalog.subset(("m1",))
    assert subset.purity[0, 0] == subset.fit_data().purity[0, 0] == low
    assert float(dict(subset.source_rows[0])["purity"]) == high
    assert catalog.fit_data(("m1",)).purity[0, 0] == low
    subset_json = OriginalCatalog.from_dict(subset.to_dict())
    assert subset_json.content_hash == subset.content_hash and subset_json.purity[0, 0] == low
    fresh_subset = write_tumor_txt(directory / "fresh_subset.tsv",
        pd.DataFrame([dict(row) for row in subset.source_rows]), dict(subset.metadata))
    assert load_original_catalog(fresh_subset).purity[0, 0] == high
    # A subset's source mean need not obey the parent's original admission test.
    balanced = pd.concat([frame.assign(mutation_id=f"b{i}", purity=value)
                         .iloc[:1] for i, value in enumerate((low, low, low, high, high, high))])
    parent = load_original_catalog(write_tumor_txt(directory / "balanced.tsv", balanced))
    child = parent.subset(("b0", "b1", "b2", "b3"))
    np.testing.assert_array_equal(child.purity, np.full((4, 1), low))
    # Canonicalization after exclusion also stays identical in the two loaders.
    excluded = frame.copy()
    excluded.loc[0, ["allele_a_cn", "segment_id"]] = [11, "excluded"]
    excluded_path = write_tumor_txt(directory / "purity_excluded.tsv", excluded)
    excluded_catalog = load_original_catalog(excluded_path, max_major_cn=10)
    np.testing.assert_array_equal(excluded_catalog.fit_data().purity, load_tumor_txt(excluded_path, max_major_cn=10).purity)
    assert excluded_catalog.purity[0, 0] == excluded_catalog.purity[1, 0] == high
    assert float(dict(excluded_catalog.source_rows[0])["purity"]) == low
    invalid = frame.assign(purity=[.79, .81])
    invalid_path = directory / "invalid_purity.tsv"
    invalid.to_csv(invalid_path, sep="\t", index=False)
    expect_value_error(lambda: load_tumor_txt(invalid_path))
    expect_value_error(lambda: load_original_catalog(invalid_path))
    expect_value_error(lambda: write_tumor_txt(directory / "invalid_writer.tsv", invalid))
    expect_value_error(lambda: replace(catalog, purity=np.full((2, 1), .7)))
    changed_source = tuple(tuple(dict(row, purity="0.7").items()) for row in catalog.source_rows)
    expect_value_error(lambda: replace(catalog, source_rows=changed_source))
    # Public API routing with no support, an object, and a TSV must admit the same data.
    bounds = catalog.default_bounds()
    support_path = directory / "support.tsv"
    pd.DataFrame([dict(mutation_id=mutation, sample_id="r1", multiplicity_lower=1,
                       multiplicity_upper=1) for mutation in catalog.mutation_ids]).to_csv(
                           support_path, sep="\t", index=False)
    captured = []

    def capture(data, *args, **kwargs):
        captured.append(data)
        if kwargs.get("multiplicity_support") is not None:
            kwargs["multiplicity_support"].for_data(data)
        return SimpleNamespace(summary={"diagnostics": {}, "fixture": "input_route_only"})

    config = FitConfig(device="cpu", max_clusters=1, cpu_threads=1)
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=capture):
        for name, support in (("ordinary", None), ("explicit_object", bounds), ("explicit_file", support_path)):
            process_tumor(path, directory / name, config, multiplicity_support=support)
    assert len(captured) == 3
    for data in captured:
        np.testing.assert_array_equal(data.purity, ordinary.purity)
        assert data.mutation_ids == ordinary.mutation_ids
    return dict(mean_tolerance_preserved=True, canonical_minimum=low, source_values_preserved=True,
                writer_catalog_roundtrip=True, parent_subset_canonical_preserved=True,
                subset_json_preserves_parent_fresh_tsv_recanonicalizes=True,
                exclusions_preserve_common_canonical=True, material_disagreement_rejected=True,
                public_api_support_routes_equivalent=True, native_fit_calls=0)


def check_provenance(directory):
    catalog = write_catalog(directory / "provenance_flags.tsv", [[10]], 20, 1, minor=1)
    base = pd.DataFrame([dict(row) for row in catalog.source_rows])
    accepted_flags = 0
    for field, aliases in (("mapping_genuine", ("mapping_genuine", "_mapping_genuine")),
                           ("normal_cn_supplied", ("normal_cn_supplied", "_normal_cn_supplied")),
                           ("input_excluded", ("input_excluded",))):
        for flag in (0, 1):
            for names in ([(alias,) for alias in aliases] + ([aliases] if len(aliases) > 1 else [])):
                table = base.assign(**{name: flag for name in names})
                path = write_tumor_txt(directory / f"flags_{accepted_flags}.tsv", table)
                retained = load_original_catalog(path)
                assert bool(getattr(retained, field)[0, 0]) == bool(flag)
                assert all(dict(retained.source_rows[0])[name] == str(flag) for name in names)
                changes = {field: ~getattr(retained, field)}
                if field == "input_excluded":
                    changes["eligible"] = ~retained.eligible
                expect_value_error(lambda: replace(retained, **changes))
                accepted_flags += 1
        for name in aliases:
            for invalid in ("false", "True", "2", ".", " 0"):
                table = base.assign(**{name: invalid})
                path = directory / "invalid_provenance.tsv"
                expect_value_error(lambda: write_tumor_txt(path, table))
                table.to_csv(path, sep="\t", index=False)
                expect_value_error(lambda: load_original_catalog(path))
                source = tuple(tuple(dict(row, **{name: invalid}).items()) for row in catalog.source_rows)
                expect_value_error(lambda: replace(catalog, source_rows=source))
        if len(aliases) == 2:
            table = base.assign(**dict(zip(aliases, (0, 1))))
            path = directory / "conflicting_provenance.tsv"
            expect_value_error(lambda: write_tumor_txt(path, table))
            table.to_csv(path, sep="\t", index=False)
            expect_value_error(lambda: load_original_catalog(path))
            source = tuple(tuple(dict(row, **dict(zip(aliases, ("0", "1")))).items())
                           for row in catalog.source_rows)
            expect_value_error(lambda: replace(catalog, source_rows=source))
    source = directory / "uncovered"
    source.mkdir()
    (source / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n1\t100\t10\t10\n")
    (source / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\n1\t200\t300\t1\t1\t2\n")
    (source / "purity.txt").write_text("1\n")
    from CliPP2.io.tumor_txt import audit_input
    expect_value_error(lambda: load_original_catalog(source))
    expect_value_error(lambda: read_region_directory(source))
    expect_value_error(lambda: load_tumor_txt(source))
    audit = audit_input(source)
    assert audit["status"] == "input_missing"
    expect_value_error(lambda: run_workflow(source, directory / "directory_blocked", workflow_config()))
    audit_path = directory / "input_audit.json"
    try:
        cli_main(["audit-input", "--input-dir", str(source), "--output", str(audit_path)])
    except SystemExit as error:
        assert error.code == 1
    else:
        raise AssertionError("Missing CN audit reported success")
    assert json.loads(audit_path.read_text())["status"] == "input_missing"
    # Covered CN with omitted normal CN is usable only under an explicit policy.
    (source / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\n1\t1\t300\t1\t1\t2\n")
    assumed = load_original_catalog(source)
    assert not assumed.normal_cn_supplied.any()
    expect_value_error(lambda: run_workflow(source, directory / "normal_required", workflow_config()))
    accepted = run_workflow(source, directory / "normal_policy", workflow_config(normal_cn_policy="diploid"))
    assert accepted["status"] == "complete"
    metadata, frame = read_region_directory(source)
    write_tumor_txt(directory / "assumed_normal.tsv", frame, metadata)
    assert not load_original_catalog(directory / "assumed_normal.tsv").normal_cn_supplied.any()
    return dict(mapping_round_trip=True, normal_cn_policy_explicit=True,
                normalized_provenance_flag_cases=accepted_flags, malformed_and_conflicting_aliases_rejected=True,
                programmatic_flag_array_mismatch_rejected=True)


def check_full_support_numerics(directory):
    catalog = write_catalog(directory / "exact.tsv", [[2]], 3, 4)
    reference = fixture_reference((0,), catalog.region_ids, [[1.]])
    request = bundle(catalog, reference, calls=[[2]], native_upper=[[2]])
    assert request.bounds.lower[0, 0] == 1 and request.bounds.upper[0, 0] == 4
    result = assign_mutations(request)
    np.testing.assert_allclose(np.exp(result.assignments[0]["log_scores"]), [15/64], rtol=1e-14)
    row = result.multiplicities[0]
    assert row["conditional_call"] == 3
    np.testing.assert_allclose(row["call_probability"], .45, rtol=1e-14)
    capped = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[1]], [[2]], "stage_two_cap")
    observation = prepare_prediction(catalog, catalog.mutation_ids, capped)
    with NativeBackend(observation, cpu_threads=1) as native:
        diagnostic = native.component_diagnostics(np.ones((1, 1)))
    np.testing.assert_allclose(np.exp(diagnostic["log_emission"]), [[33/128]], rtol=1e-14)
    marginal = write_catalog(directory / "marginal.tsv", [[1]], 2, 2)
    reference = fixture_reference((0, 1), marginal.region_ids, [[1.], [.6]])
    probabilities = assign_mutations(bundle(marginal, reference)).assignments[0]["probabilities"]
    np.testing.assert_allclose(probabilities, [5/14, 9/14], rtol=1e-14)
    high = write_catalog(directory / "major_five.tsv", [[10]], 100, 5)
    restored = bundle(high, fixture_reference((0,), high.region_ids, [[1.]]), native_upper=[[2]])
    assert restored.bounds.upper[0, 0] == 5
    return dict(full_mass=15/64, capped_mass=33/128, marginal_probabilities=list(probabilities),
                capped_stage_two_support_restored=True, single_state_major_above_four=True)


def check_assignment_decisions(directory):
    catalog = write_catalog(directory / "target_outlier.tsv", [[20]+[50]*9], 100, 1, minor=1)
    reference = fixture_reference((0,), catalog.region_ids, np.ones((1, 10)))
    row = assign_mutations(bundle(catalog, reference)).assignments[0]
    assert row["evidence_status"] == "observed" and row["posterior"] == 1. and row["final_assignment_label"] == 0
    assert row["maximum_profile_deviance"] > 10. and row["mean_profile_deviance"] < 5.
    assert row["second_best_label"] is None and pd.isna(row["log_score_margin"]) and row["entropy"] == 0.
    exact = write_catalog(directory / "ties.tsv", [[25]], 100, 1, minor=1)
    for labels in ((5, 0), (0, 5)):
        tied = fixture_reference(labels, exact.region_ids, [[.5], [.5]], clonal_label=None)
        decision = assign_mutations(bundle(exact, tied)).assignments[0]
        assert decision["final_assignment_label"] == 0 and tuple(decision["map_tied_labels"]) == (0, 5)
        assert not decision["map_unique"] and decision["evidence_status"] == "observed"
    near = write_catalog(directory / "near_tie.tsv", [[26]], 100, 1, minor=1)
    request = bundle(near, fixture_reference((0, 5), near.region_ids, [[.5], [.50001]], clonal_label=None))
    decision = assign_mutations(request).assignments[0]
    assert decision["final_assignment_label"] == 5 and decision["map_unique"] and decision["posterior"] < .9
    for depth in (100, 10000):
        zero = write_catalog(directory / f"certainty_{depth}.tsv", [[0]], depth, 1, minor=1)
        request = bundle(zero, fixture_reference((0, 1), zero.region_ids, [[1.], [.01]]))
        row = assign_mutations(request).assignments[0]
        assert row["evidence_status"] == "observed" and row["final_assignment_label"] == 1
        assert np.isfinite(row["log_scores"]).all()
    return dict(poor_singleton_fit_measured_not_vetoed=True, exact_and_near_ties=True,
                weak_evidence_map_returned=True, finite_log_evidence_retained=True)


def check_conditional_information(directory):
    for major in (1, 3):
        catalog = write_catalog(directory / f"zero_ccf_{major}.tsv", [[0]], 100, major)
        reference = fixture_reference((0,), catalog.region_ids, [[0.]], clonal_label=None)
        result = assign_mutations(bundle(catalog, reference), outdir=directory / f"zero_ccf_{major}")
        row = result.multiplicities[0]
        np.testing.assert_allclose(row["conditional_probabilities"], np.full(major, 1/major), rtol=1e-14)
        assert row["conditional_distribution_defined"] and row["conditional_map"] == 1
        assert not row["multiplicity_read_informative"] and np.isnan(row["conditional_call"])
        assert row["structural_singleton"] == (major == 1)
        assert row["multiplicity_tie"] == (major > 1)
        assert row["multiplicity_status"] == "uninformative_zero_ccf"
    catalog = write_catalog(directory / "observation_flags.tsv", [[0, 0, 0, 0]],
        [[100, 0, 100, 100]], [[3, 3, 3, 11]], minor=1, observed=[[1, 1, 0, 1]])
    reference = fixture_reference((0,), catalog.region_ids, np.ones((1, 4)))
    result = assign_mutations(bundle(catalog, reference), outdir=directory / "observation_flags")
    assert [row["informative"] for row in result.multiplicities] == [True, False, False, False]
    assert [row["multiplicity_status"] for row in result.multiplicities][1:] == ["missing_observation", "missing_observation", "excluded"]
    for name, depth, major, center, expected in (("missing", 0, 3, 1., "prior_only"),
            ("excluded", 100, 11, 1., "excluded"), ("impossible", 100, 3, 0., "model_incompatible")):
        catalog = write_catalog(directory / f"{name}.tsv", [[0 if not depth else 1]], depth, major)
        reference = fixture_reference((0,), catalog.region_ids, [[center]], clonal_label=None)
        result = assign_mutations(bundle(catalog, reference), outdir=directory / name)
        assert result.assignments[0]["status"] == expected
        if expected == "prior_only":
            np.testing.assert_array_equal(result.assignments[0]["probabilities"], [1.])
            assert result.assignments[0]["final_assignment_label"] is None
        else:
            assert np.isnan(result.assignments[0]["probabilities"]).all()
        assert not result.multiplicities[0]["conditional_distribution_defined"]
    certain = write_catalog(directory / "conditional_certainty.tsv", [[100]], 100, 2)
    request = bundle(certain, fixture_reference((0,), certain.region_ids, [[1.]]))
    row = assign_mutations(request).multiplicities[0]
    assert row["call_probability"] == 1. and row["call_log_probability"] < 0
    assert row["assigned_multiplicity"] == 2 and row["map_unique"]
    return dict(zero_ccf_uniform=True, singleton_structural=True, zero_alt_informative=True,
                missing_excluded_incompatible_distinct=True)


def check_copy_and_publication(directory):
    catalog = write_catalog(directory / "copy.tsv", [[25, 75], [25, 75]], 100, 3, minor=1)
    request = bundle(catalog, fixture_reference((0,), catalog.region_ids, [[1., 1.]]), calls=[[1, 3], [1, 3]])
    for provenance in ({"reason": "copy regression"}, ()):
        original = replace(request, provenance=provenance)
        for clone in (copy.copy(original), copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
            assert clone.content_hash == original.content_hash == content_hash(clone.to_dict())
            for array in (clone.native_centers, clone.native_calls, clone.reference.centers, clone.bounds.upper):
                assert not array.flags.writeable
                expect_value_error(lambda: array.setflags(write=True))
    result = assign_mutations(request, outdir=directory / "valid_publication")
    for value in (result, fixture_reference((0,), catalog.region_ids, [[1., 1.]], lineage=({"origin": "copy"},))):
        for clone in (copy.copy(value), copy.deepcopy(value), pickle.loads(pickle.dumps(value))):
            assert content_hash(clone.to_dict()) == content_hash(value.to_dict())
            arrays = ([row["probabilities"] for row in clone.assignments]
                      if hasattr(clone, "assignments") else [clone.centers])
            assert all(not array.flags.writeable for array in arrays)
    path = directory / "copy_bundle.json"
    save_assignment_bundle(request, path)
    assert load_assignment_bundle(path).content_hash == request.content_hash
    variants = []
    for changes in ({"probabilities": [-.1, 1.1]}, {"probabilities": [.9]},
                    {"reference_labels": (9,)}, {"final_assignment_label": None}, {"evidence_status": "prior_only"}):
        rows = [dict(row) for row in result.assignments]
        rows[0].update(changes)
        variants.append(lambda rows=rows: replace(result, assignments=tuple(rows)))
    rows = [dict(row) for row in result.multiplicities]
    rows[0]["conditional_call"] = 4.
    variants.append(lambda: replace(result, multiplicities=tuple(rows)))
    for index, invalid in enumerate(variants):
        expect_value_error(lambda: publish_assignment(request, invalid(), directory / f"invalid_{index}"))
    stale = copy.deepcopy(request)
    object.__setattr__(stale, "native_calls", np.full((2, 2), 2.))
    expect_value_error(lambda: save_assignment_bundle(stale, directory / "stale.json"))
    expect_value_error(lambda: publish_assignment(stale, result, directory / "stale_output"))
    expect_value_error(lambda: assign_mutations(stale))
    exact = write_catalog(directory / "tamper.tsv", [[2]], 3, 4)
    request = bundle(exact, fixture_reference((0,), exact.region_ids, [[1.]]))
    result = assign_mutations(request)
    for i, changes in enumerate(({"native_call_log_probability": 1000.},
            {"top_two_gap": 0., "multiplicity_tie": True, "multiplicity_status": "ambiguous_tie"},
            {"profile_deviance": 0.})):
        records = [dict(row) for row in result.multiplicities]
        records[0].update(changes)
        assignments = [dict(row) for row in result.assignments]
        if "profile_deviance" in changes:
            assignments[0]["mean_profile_deviance"] = 0.
        invalid = replace(result, multiplicities=tuple(records), assignments=tuple(assignments))
        expect_value_error(lambda: publish_assignment(request, invalid, directory / f"tampered_evidence_{i}"))
    for i, changes in enumerate(({"protocol_version": "2.2"}, {"tumor_id": "other"},
            {"normal_cn_policy": "invented"}, {"reference": {}})):
        invalid = replace(result, summary=dict(result.summary, **changes))
        expect_value_error(lambda: publish_assignment(request, invalid, directory / f"tampered_metadata_{i}"))
    return dict(copy_immutable=True, live_identity_checked=True, invalid_publication_rejected=True)


def check_complete_workflow(directory):
    path = directory / "workflow.tsv"
    write_catalog(path, [[50, 50], [49, 50], [50, 49], [12, 18], [13, 17], [12, 18]], 100, 1, minor=1)
    config = workflow_config()
    output = directory / "workflow"
    plan_config = asdict(config)
    plan_config["fit"].pop("max_clusters")
    plan_config.pop("stage_two_max_clusters")
    plan = dict(schema="clipp2.workflow.plan.v5", protocol_version=PROTOCOL_VERSION,
                input_path=path.name, workflow_config=plan_config)
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan))
    import CliPP2.provenance
    package = Path(CliPP2.provenance.__file__).resolve().parent
    original_rglob = Path.rglob

    def forbid_source_scan(path, pattern):
        if path.resolve() == package or package in path.resolve().parents:
            raise AssertionError("Inference scanned the package source directory")
        return original_rglob(path, pattern)

    with patch.object(Path, "rglob", forbid_source_scan):
        cli_main(["workflow", "--plan", str(plan_path), "--output-dir", str(output)])
    receipt = json.loads((output / "workflow_receipt.json").read_text())
    assert receipt["status"] == "complete" and receipt["protocol_version"] == PROTOCOL_VERSION
    assert_published_files(receipt, output)
    assert receipt["counts"]["N0"] == receipt["counts"]["N2"] == 3
    assert not (output / "donor_contributors.tsv").exists() and not (output / "stage_three_bounds.tsv").exists()
    policies = pd.read_csv(output / "stage2_segment_multiplicity_bounds.tsv", sep="\t")
    assert len(policies) == 2 and (policies["contributor_count"] == 3).all()
    assert not policies["fallback"].any() and "contributor_ids" not in policies
    first, second = (load_accepted_fit(output / f"stage_{stage}_accepted.json") for stage in ("one", "two"))
    assert [first.producer_receipt["fit_config"]["max_clusters"], second.producer_receipt["fit_config"]["max_clusters"]] == [10, 20]
    plan.update(accepted_stage_one="workflow/stage_one_accepted.json", accepted_stage_two="workflow/stage_two_accepted.json")
    plan_path.write_text(json.dumps(plan))
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("accepted result refitted")):
        cli_main(["workflow", "--plan", str(plan_path), "--output-dir", str(directory / "reused")])
    for stage, original in (("one", first), ("two", second)):
        reused = load_accepted_fit(directory / f"reused/stage_{stage}_accepted.json")
        assert reused.producer_receipt == original.producer_receipt and reused.compatibility_receipts
    request = load_assignment_bundle(output / "assignment_bundle.json")
    assert not request.non_target_ids and len(request.target_ids) == 6
    cli_main(["assign", "--bundle", str(output / "assignment_bundle.json"), "--output-dir", str(directory / "replayed_assignment")])
    for name in ("workflow_mutation_assignments.tsv", "workflow_assignment_multiplicity.tsv"):
        assert (output / "assignment" / name).read_bytes() == (directory / "replayed_assignment" / name).read_bytes()
    whole = pd.read_csv(output / "all_original_mutations.tsv", sep="\t")
    assert set(whole["reassessment_status"]) <= {"retained", "reassigned"}
    assert (whole["role"] == "target").all() and whole["posterior_defined"].all()
    catalog = load_original_catalog(path)
    renamed = replace(catalog, tumor_id="another_tumor",
                      metadata=tuple(dict(catalog.metadata, tumor_id="another_tumor").items()))
    expect_value_error(lambda: first.validate(renamed))
    expect_value_error(lambda: second.validate(renamed))
    return dict(status=receipt["status"], counts=receipt["counts"], both_stages_replayed=True,
                cross_tumor_rejected=True, no_automatic_source_scan=True)


def check_group_outlier_and_fit(directory):
    path = directory / "group_outlier.tsv"
    write_catalog(path, [[50]]*19+[[20]], 100, 1, minor=1)
    config = workflow_config(fit=FitConfig(device="cpu", max_clusters=1, cpu_threads=1))
    output = directory / "group_outlier"
    receipt = run_workflow(path, output, config)
    group = pd.read_csv(output / "stage_one_groups.tsv", sep="\t").iloc[0]
    assert receipt["status"] == "complete" and receipt["reference"]["reference_source_role"] == "provisional_reference_source"
    assert json.loads(group["native_mean_profile_deviance"])[0] < 5.
    assert json.loads(group["native_maximum_profile_deviance"])[0] > 10.
    ordinary = directory / "ordinary_group_fit"
    summary = process_tumor(path, ordinary, config.fit)
    assert_published_files(summary["publication"], ordinary)
    for name in summary["publication"]["files"]:
        assert (ordinary / name).read_bytes() == (output / "stage_one" / name).read_bytes()
    return dict(group_outlier_measured_without_veto=True, ordinary_and_workflow_tables_identical=True)


def check_segment_provenance(directory):
    from CliPP2.workflow import _stage_two_caps
    measurements = []
    for size in (100, 200):
        catalog = write_catalog(directory / f"segment_{size}.tsv", np.full((size, 1), 25), 100, 3, minor=1)
        audits = [dict(mutation_id=mutation, region_id="r1", segment_id="shared_segment", cap_source_map_upper=1, contributes_to_cap=True)
                  for mutation in catalog.mutation_ids]
        bounds, policies = _stage_two_caps(catalog, audits)
        assert len(policies) == 1 and policies[0]["contributor_count"] == size
        assert np.all(bounds.upper == 1) and "contributor_ids" not in policies[0]
        measurements.append(dict(mapped_pairs=size, contributors=len(audits), policy_rows=len(policies),
                                 policy_bytes=len(json.dumps(policies))))
    assert measurements[1]["policy_bytes"] == measurements[0]["policy_bytes"]
    return measurements


def check_qualification_metrics(directory):
    from qualify_workflow import partition_matching, reference_errors
    unconstrained = reference_errors([[.2], [.4]], [[.3]])
    assert unconstrained["reference_recall"] is None and unconstrained["recovered_by_truth"] is None
    assert unconstrained["recovery_status"] == "unassessed_no_declared_tolerance"
    assert np.isfinite(unconstrained["nearest_l2"]).all()
    limited = reference_errors([[.2], [.4]], [[.3]], .11)
    assert sum(limited["recovered_by_truth"]) == 1 and limited["reference_recall"] == .5
    _, partition_error = partition_matching([0, 0, 1, 1], [0, 0, 1, 1])
    distorted = reference_errors([[.2], [.4]], [[.7], [.9]], .05)
    assert partition_error == 0 and distorted["reference_recall"] == 0
    absent = reference_errors([[.2], [.4]], [], .05)
    assert absent["reference_recall"] == 0 and absent["nearest_l2"] == [None, None]
    return dict(default_binary_recovery_unassessed=True, continuous_error_always_reported=True,
                one_to_one_recovery=True, perfect_overlap_is_not_center_recovery=True)


def run_checks(directory, device="cpu"):
    if device != "cpu":
        raise ValueError("These contract checks qualify CPU only; CUDA needs a separately executed suite")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from check_inputs import run_input_checks
    from check_native_diagnostics import run_native_diagnostic_checks
    from check_assignment import run_checks as run_assignment_checks
    from check_workflow_edges import run_workflow_edge_checks
    checks = (run_native_diagnostic_checks, run_assignment_checks, run_input_checks, run_workflow_edge_checks, check_purity_consistency, check_provenance, check_full_support_numerics, check_assignment_decisions,
              check_copy_and_publication, check_conditional_information, check_complete_workflow, check_group_outlier_and_fit,
              check_segment_provenance, check_qualification_metrics)
    records = {}
    for check in checks:
        child = directory / check.__name__
        child.mkdir()
        records[check.__name__] = check(child)
    return dict(schema="clipp2.workflow-contract-checks.v4", protocol_version=PROTOCOL_VERSION,
                device="cpu", fixture_config=asdict(workflow_config()), calibrated=False,
                checks=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Check output already exists")
    with tempfile.TemporaryDirectory(prefix="clipp2-contracts-") as temporary:
        report = run_checks(temporary, args.device)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Passed {len(report['checks'])} Protocol {PROTOCOL_VERSION} contract groups")


if __name__ == "__main__":
    main()
