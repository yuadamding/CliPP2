#!/usr/bin/env python3
"""Deterministic CPU contract checks against the imported CliPP2 installation.

Run after installing one wheel or an editable package. Fixture thresholds are
frozen regression settings, not biological calibration recommendations. Inputs
and published results live in a temporary directory unless --output is given.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2 import DescriptiveScreen, FitConfig, WorkflowConfig, process_tumor, run_workflow
from CliPP2.assignment import AssignmentBundle, AssignmentReference, assign_mutations
from CliPP2.cli import main as cli_main
from CliPP2.config import AssignmentThresholds
from CliPP2.io.assignment import load_accepted_fit, load_assignment_bundle, publish_assignment
from CliPP2.io.data import MultiplicityBounds
from CliPP2.io.region_files import read_region_directory
from CliPP2.io.tumor_txt import load_original_catalog, load_tumor_txt, write_tumor_txt


def write_catalog(path, alt, depth, major, *, minor=0, purity=None):
    alt, depth, major, minor = np.broadcast_arrays(
        np.atleast_2d(alt), np.atleast_2d(depth), np.atleast_2d(major), np.atleast_2d(minor))
    purity = np.ones(alt.shape[1]) if purity is None else np.asarray(purity)
    rows = [dict(mutation_id=f"m{i:03}", sample_id=f"r{r+1}",
        alt_count=int(alt[i, r]), ref_count=int(depth[i, r]-alt[i, r]), count_observed=1,
        purity=float(purity[r]), normal_cn=2, segment_id="shared_segment", cn_state_id="0",
        cn_state_fraction=1., allele_a_cn=int(major[i, r]), allele_b_cn=int(minor[i, r]))
        for i in range(alt.shape[0]) for r in range(alt.shape[1])]
    write_tumor_txt(path, pd.DataFrame(rows))
    return load_original_catalog(path)


def thresholds():
    return AssignmentThresholds(.9, DescriptiveScreen(10., 5., .05), 1e-10, 1e-10, .01)


def workflow_config(**changes):
    config = WorkflowConfig(thresholds(), .95, 10., DescriptiveScreen(10., 5., .02), 1e-10,
                            fit=FitConfig(device="cpu", max_clusters=10, cpu_threads=1))
    return replace(config, **changes)


def bundle(catalog, reference, *, bounds=None, target_ids=None, calls=None):
    targets = catalog.mutation_ids if target_ids is None else target_ids
    support = catalog.default_bounds().subset(targets, reference.region_ids)
    shape = len(targets), len(reference.region_ids)
    return AssignmentBundle(catalog, reference, targets, (), support if bounds is None else bounds,
        thresholds(), np.ones(shape), np.ones(shape) if calls is None else calls,
        support.lower, support.upper, ("regression_fixture",)*len(targets), (0,)*len(targets), "cpu", 1)


def expect_value_error(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid evidence was accepted")


def assert_no_file_metadata(value):
    """Numerical/content hashes remain valid; file/source fingerprints do not."""
    if isinstance(value, Mapping):
        assert not any(key.endswith("_sha256") or key in (
            "source_commit", "source_dirty", "python_build_record", "input_identity") for key in value)
        for item in value.values():
            assert_no_file_metadata(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_no_file_metadata(item)


def assert_published_files(receipt, directory):
    assert_no_file_metadata(receipt)
    assert isinstance(receipt["files"], (tuple, list))
    assert all(isinstance(name, str) and (directory / name).is_file() for name in receipt["files"])


def check_provenance(directory):
    source = directory / "uncovered"
    source.mkdir()
    (source / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n1\t100\t10\t10\n")
    (source / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\n1\t200\t300\t1\t1\t2\n")
    (source / "purity.txt").write_text("1\n")
    original = load_original_catalog(source)
    assert not original.mapping_genuine.any()
    metadata, frame = read_region_directory(source)
    converted = directory / "converted.tsv"
    write_tumor_txt(converted, frame, metadata)
    repeated = load_original_catalog(converted)
    np.testing.assert_array_equal(original.mapping_genuine, repeated.mapping_genuine)
    # Ordinary fitting retains its historical adapter policy without raw rows.
    assert load_tumor_txt(source).original_catalog is None
    for name, path in (("directory", source), ("tsv", converted)):
        expect_value_error(lambda: run_workflow(path, directory / (name+"_blocked"), workflow_config()))
    frame["_mapping_genuine"] = "false"
    invalid = directory / "invalid_marker.tsv"
    expect_value_error(lambda: write_tumor_txt(invalid, frame, metadata))
    frame.to_csv(invalid, sep="\t", index=False)
    expect_value_error(lambda: load_original_catalog(invalid))
    return dict(round_trip_preserved=True, uncovered_workflow_rejected=True)


def check_support_and_publication(directory):
    catalog = write_catalog(directory / "alignment.tsv", [[25, 75], [25, 75]], 100, 3, minor=1)
    reference = AssignmentReference((0,), catalog.region_ids, [[1., 1.]])
    bounds = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, [[1, 3], [2, 1]], [[3, 3], [3, 3]], "floor")
    normal = bundle(catalog, reference, bounds=bounds, calls=[[1, 3], [1, 3]])
    permuted = replace(normal, bounds=MultiplicityBounds(bounds.mutation_ids[::-1], bounds.region_ids[::-1],
        bounds.lower[::-1, ::-1], bounds.upper[::-1, ::-1], bounds.policy_id))
    assert normal.content_hash == permuted.content_hash
    first = assign_mutations(normal, outdir=directory / "aligned")
    second = assign_mutations(permuted, outdir=directory / "permuted")
    for filename in ("alignment_mutation_assignments.tsv", "alignment_assignment_multiplicity.tsv"):
        assert (directory / "aligned" / filename).read_bytes() == (directory / "permuted" / filename).read_bytes()
    for name in ("aligned", "permuted"):
        published = directory / name
        summary = json.loads((published / "alignment_assignment_summary.json").read_text())
        assert_published_files(summary, published)
    for row in second.multiplicities:
        for policy in ("restricted", "original"):
            call = row[policy+"_conditional_call"]
            if call is not None and np.isfinite(call):
                assert row[policy+"_lower"] <= call <= row[policy+"_upper"]
    assert second.multiplicities[0]["native_call_excluded"] is False
    bad_rows = [dict(row) for row in first.multiplicities]
    bad_rows[0]["restricted_conditional_call"] = 4.
    expect_value_error(lambda: publish_assignment(normal, replace(first, multiplicities=tuple(bad_rows)),
                                                   directory / "invalid_publication"))
    return dict(canonical_hash_equal=True, permuted_tables_identical=True, invalid_call_rejected=True)


def check_assignment_decisions(directory):
    catalog = write_catalog(directory / "target_outlier.tsv", [[20]+[50]*9], 100, 1, minor=1)
    reference = AssignmentReference((0,), catalog.region_ids, np.ones((1, 10)))
    result = assign_mutations(bundle(catalog, reference))
    row = result.restricted[0]
    assert row["status"] == "unresolved_fit" and row["posterior"] == 1.
    assert row["mean_profile_deviance"] < 5. and row["outlier_fraction"] == .1
    assert row["second_best_label"] is None and pd.isna(row["log_score_margin"]) and row["entropy"] == 0.
    assert row["calibrated_fit_status"] == "not_calibrated"
    exact = write_catalog(directory / "ties.tsv", [[25]], 100, 1, minor=1)
    for labels in ((5, 0), (0, 5)):
        tied = AssignmentReference(labels, exact.region_ids, [[.5], [.5]], clonal_label=None)
        decision = assign_mutations(bundle(exact, tied)).restricted[0]
        assert decision["best_label"] == 0 and decision["status"] == "ambiguous_tie"
    near = write_catalog(directory / "near_tie.tsv", [[26]], 100, 1, minor=1)
    reference = AssignmentReference((0, 5), near.region_ids, [[.5], [.50001]], clonal_label=None)
    request = bundle(near, reference)
    request = replace(request, thresholds=replace(request.thresholds, tie_atol=.001))
    decision = assign_mutations(request).restricted[0]
    assert decision["best_label"] == 5 and decision["status"] == "ambiguous_tie"
    reference = AssignmentReference((5, 0), exact.region_ids, [[0.], [1.]])
    decision = assign_mutations(bundle(exact, reference)).restricted[0]
    assert np.isposinf(decision["log_score_margin"])
    empty = bundle(exact, reference, target_ids=())
    with patch("CliPP2.assignment.NativeBackend", side_effect=AssertionError("empty request scored")):
        empty_result = assign_mutations(empty, outdir=directory / "empty")
    assert not empty_result.restricted and not empty_result.multiplicities
    assignment = pd.read_csv(directory / "empty/ties_mutation_assignments.tsv", sep="\t")
    multiplicity = pd.read_csv(directory / "empty/ties_assignment_multiplicity.tsv", sep="\t")
    assert assignment.empty and "restricted_status" in assignment
    assert multiplicity.empty and "restricted_conditional_call" in multiplicity
    return dict(outlier_fraction=row["outlier_fraction"], singleton_missing_margin=True,
                label_order_independent=True, empty_headers_valid=True)


def check_complete_workflow(directory):
    path = directory / "workflow.tsv"
    write_catalog(path, [[50, 50], [49, 50], [50, 49], [12, 18], [13, 17], [12, 18]], 100, 1, minor=1)
    config = workflow_config()
    output = directory / "workflow"
    plan_config = asdict(config)
    plan_config["fit"].pop("max_clusters")
    plan_config.pop("stage_two_max_clusters")
    plan = dict(schema="clipp2.workflow.plan.v2", input_path=path.name, workflow_config=plan_config)
    plan_path = directory / "plan.json"
    plan_path.write_text(json.dumps(plan))
    cli_main(["workflow", "--plan", str(plan_path), "--output-dir", str(output)])
    receipt = json.loads((output / "workflow_receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert_published_files(receipt, output)
    assert receipt["counts"]["N0"] == receipt["counts"]["N2"] == 3
    for name in ("stage_two_bounds.tsv", "stage_three_bounds.tsv"):
        policies = pd.read_csv(output / name, sep="\t")
        assert len(policies) == 2 and (policies["contributor_count"] == 3).all()
        assert not policies["fallback"].any() and "contributor_ids" not in policies
    assert len(pd.read_csv(output / "mutation_segments.tsv", sep="\t")) == 12
    first = load_accepted_fit(output / "stage_one_accepted.json")
    second = load_accepted_fit(output / "stage_two_accepted.json")
    producer_one, producer_two = first.producer_receipt, second.producer_receipt
    for stage, producer in (("stage_one", producer_one), ("stage_two", producer_two)):
        assert_no_file_metadata(producer)
        assert_published_files(producer["publication"], output / stage)
    assert [producer_one["fit_config"]["max_clusters"], producer_two["fit_config"]["max_clusters"]] == [10, 20]
    plan.update(accepted_stage_one="workflow/stage_one_accepted.json", accepted_stage_two="workflow/stage_two_accepted.json")
    plan_path.write_text(json.dumps(plan))
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("accepted result refitted")):
        cli_main(["workflow", "--plan", str(plan_path), "--output-dir", str(directory / "reused")])
    replay = json.loads((directory / "reused/workflow_receipt.json").read_text())
    assert replay["status"] == "complete"
    assert_published_files(replay, directory / "reused")
    reused_one = load_accepted_fit(directory / "reused/stage_one_accepted.json")
    reused_two = load_accepted_fit(directory / "reused/stage_two_accepted.json")
    assert reused_one.producer_receipt == producer_one and reused_two.producer_receipt == producer_two
    assert reused_one.compatibility_receipts and reused_two.compatibility_receipts
    load_assignment_bundle(output / "assignment_bundle.json")
    cli_main(["assign", "--bundle", str(output / "assignment_bundle.json"),
              "--output-dir", str(directory / "replayed_assignment")])
    for filename in ("workflow_mutation_assignments.tsv", "workflow_assignment_multiplicity.tsv"):
        assert (output / "assignment" / filename).read_bytes() == (directory / "replayed_assignment" / filename).read_bytes()
    whole = pd.read_csv(output / "all_original_mutations.tsv", sep="\t")
    assert "original_input_status" in whole and "original_support_assignment_status" in whole
    assert "original_status" not in whole
    return dict(status=receipt["status"], counts=receipt["counts"], accepted_stages_reused_without_refit=True)


def check_group_outlier_and_fit(directory):
    path = directory / "group_outlier.tsv"
    write_catalog(path, [[50]]*19+[[20]], 100, 1, minor=1)
    config = workflow_config(fit=FitConfig(device="cpu", max_clusters=1, cpu_threads=1))
    output = directory / "group_outlier"
    receipt = run_workflow(path, output, config)
    group = pd.read_csv(output / "stage_one_groups.tsv", sep="\t").iloc[0]
    assert receipt["status"] == "no_eligible_clonal_reference" and group["classification"] == "unresolved"
    assert json.loads(group["native_mean_profile_deviance"])[0] < 5.
    assert json.loads(group["native_outlier_fraction"]) == [.05]
    assert group["original_native_mean_profile_deviance"] == group["native_mean_profile_deviance"]
    whole = pd.read_csv(output / "all_original_mutations.tsv", sep="\t")
    assert whole["original_support_assignment_status"].isna().all()
    ordinary = directory / "ordinary_group_fit"
    summary = process_tumor(path, ordinary, config.fit)
    assert_published_files(summary["publication"], ordinary)
    assert_no_file_metadata(summary)
    assert summary["publication"]["verification"]["membership_audit"]["selected_count_certified"]
    for name in summary["publication"]["files"]:
        assert (ordinary / name).read_bytes() == (output / "stage_one" / name).read_bytes()
    return dict(group_outlier_rejected=True, ordinary_and_workflow_tables_identical=True,
                mandatory_audit_preserved=True, publication_filenames_only=True)


def check_segment_provenance(directory):
    from CliPP2.workflow import _segment_bounds

    measurements = []
    for size in (100, 200):
        catalog = write_catalog(directory / f"segment_{size}.tsv", np.full((size, 1), 25), 100, 3, minor=1)
        audits = [dict(mutation_id=mutation, region_id="r1", segment_id="shared_segment",
                       native_call=1, usable=True) for mutation in catalog.mutation_ids]
        bounds, policies = _segment_bounds(catalog, audits, lower=False, policy="regression_cap")
        assert len(policies) == 1 and policies[0]["contributor_count"] == size
        assert np.all(bounds.upper == 1) and "contributor_ids" not in policies[0]
        measurements.append(dict(mapped_pairs=size, contributors=len(audits), policy_rows=len(policies),
                                 policy_bytes=len(json.dumps(policies))))
    assert measurements[1]["policy_bytes"] == measurements[0]["policy_bytes"]
    return measurements


def run_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checks = (check_provenance, check_support_and_publication, check_assignment_decisions,
              check_complete_workflow, check_group_outlier_and_fit, check_segment_provenance)
    return dict(schema="clipp2.workflow-contract-checks.v1", device="cpu",
                fixture_config=asdict(workflow_config()), calibrated=False,
                checks={check.__name__: check(directory) for check in checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence; all fixture files are temporary.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Choose a new evidence output path")
    with tempfile.TemporaryDirectory(prefix="clipp2-workflow-checks-") as temporary:
        report = run_checks(temporary)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Passed {len(report['checks'])} workflow contract groups on CPU; calibration remains unapproved.")


if __name__ == "__main__":
    main()
