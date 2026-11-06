"""Protocol 2.2 workflow boundary regressions; fixture cutoffs are not calibration."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2 import DescriptiveScreen, FitConfig, run_workflow
from CliPP2.assignment import assign_mutations
from CliPP2.io.assignment import load_accepted_fit, load_assignment_bundle
from CliPP2.io.data import MultiplicityBounds, content_hash
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.workflow import (AcceptedFit, _reconcile_final_memberships, _biological_hash, _fit_accepted,
                            _build_stage_three_inputs, classify_groups, qualify_contributors)

from check_workflow import bundle, expect_value_error, external_build_record, thresholds, workflow_config, write_catalog


def _blocked_count(directory):
    path = directory / "blocked.tsv"
    catalog = write_catalog(path, [[50, 50], [20, 0]], 100, 1, minor=1,
                            observed=[[True, True], [True, False]])
    # A native fit ordinarily rejects this unsupported group. This optimizer
    # stub isolates the coordinator's defensive blocked-stage accounting.
    first = AcceptedFit("stage_one", catalog.mutation_ids, catalog.region_ids,
        np.array([0, 1]), [[1., 1.], [.4, .4]], [.5, .5], np.ones((2, 2)),
        catalog.default_bounds(), _biological_hash(catalog, catalog.mutation_ids),
        "accounting_fixture", 0., 0., 0., 0., {}, biological_identity=catalog.biological_identity,
        tumor_id=catalog.tumor_id)
    with patch("CliPP2.workflow._fit_accepted", return_value=first) as fitter:
        receipt = run_workflow(path, directory / "blocked", workflow_config())
    assert fitter.call_count == 1 and receipt["status"] == "blocked"
    assert receipt["stage_two_candidate_count"] == 1
    assert receipt["counts"]["N1"] == 2 and receipt["counts"]["N0"] == 1
    assert receipt["counts"]["N2"] == receipt["counts"]["Q2"] == 0
    assert receipt["stage_three"] == "not_applicable"
    return dict(candidate_count=1, accepted_stage_two_count=0, optimizer_stub=True)


def _contributor_certainty(directory):
    config = workflow_config(contributor_probability_min=1.,
                             fit=FitConfig(device="cpu", max_clusters=1, cpu_threads=1))
    catalog = write_catalog(directory / "confidence_one.tsv", [[100]], 100, 2)
    fit = _fit_accepted(catalog, catalog.mutation_ids, catalog.default_bounds(),
                        directory / "confidence_fit", "stage_one", config.fit, None)
    row, = qualify_contributors(catalog, fit, config)
    assert row["native_call_probability"] == 1. and np.isfinite(row["native_gap"])
    assert not row["native_confidence_pass"] and row["reason"] == "native_low_confidence"
    relaxed, = qualify_contributors(catalog, fit, replace(config, contributor_probability_min=.95))
    assert relaxed["usable"]
    singleton = write_catalog(directory / "confidence_singleton.tsv", [[50]], 100, 1, minor=1)
    fit = _fit_accepted(singleton, singleton.mutation_ids, singleton.default_bounds(),
                        directory / "singleton_fit", "stage_one", config.fit, None)
    certain, = qualify_contributors(singleton, fit, config)
    assert certain["usable"] and certain["native_confidence_pass"]
    return dict(display_probability=row["native_call_probability"], finite_alternative_rejected=True,
                singleton_certainty_accepted=True)


def _capped_target_and_replay(directory):
    alt = [[100, 25], [25, 100]]*2 + [[25, 25]]*16 + [[12, 12]]*16
    path = directory / "capped_targets.tsv"
    metadata = dict(tumor_id="tumor_A", genome_build="fixture_build",
                    mutation_key_convention="opaque_fixture")
    catalog = write_catalog(path, alt, 100, 4, metadata=metadata)
    config = workflow_config(fit=FitConfig(device="cpu", max_clusters=3, cpu_threads=1),
                             stage_two_max_clusters=3)
    output = directory / "capped_workflow"
    with patch("CliPP2.workflow.qualify_contributors", wraps=qualify_contributors) as qualify:
        receipt = run_workflow(path, output, config, run_build_manifest=external_build_record())
    assert receipt["status"] == "complete" and qualify.call_count == 1
    assert qualify.call_args.args[1].source_id == "stage_one"
    first = load_accepted_fit(output / "stage_one_accepted.json")
    second = load_accepted_fit(output / "stage_two_accepted.json")
    request = load_assignment_bundle(output / "assignment_bundle.json")
    capped_targets = np.array([stage == "stage_two" for stage in request.source_stages])
    assert capped_targets.any() and request.non_target_ids
    assert np.all(request.native_upper[capped_targets] == 1)
    assert np.all(request.bounds.lower == 1) and np.all(request.bounds.upper == 4)
    for source in request.reference.lineage:
        if source["source_stage"] == "stage_two" and source["baseline_reference_label"] != 0:
            position = request.reference.labels.index(source["baseline_reference_label"])
            np.testing.assert_array_equal(request.reference.centers[position], second.centers[source["source_label"]])
    counts = receipt["counts"]
    assert counts["N1"] == counts["N0"] + counts["N2"] == counts["combined_mutations"]
    assert counts["targets"] + counts["non_targets"] == counts["combined_mutations"]
    assert sum(counts[key] for key in ("retained_clonal", "reassigned", "unresolved")) == counts["targets"]
    whole = pd.read_csv(output / "all_original_mutations.tsv", sep="\t")
    assert (whole.loc[whole.role == "non_target", "reassessment_status"] == "unchanged_non_target").all()
    assert not (output / "donor_contributors.tsv").exists()
    assert not (output / "stage_three_bounds.tsv").exists()
    for accepted in (first, second):
        for clone in (copy.copy(accepted), copy.deepcopy(accepted), pickle.loads(pickle.dumps(accepted))):
            assert clone.state_fingerprint == accepted.state_fingerprint
            assert not clone.centers.flags.writeable and not clone.multiplicities.flags.writeable
            expect_value_error(lambda: clone.centers.setflags(write=True))
        _manifest_tampering(catalog, accepted)
        _execution_tampering(catalog, accepted)
        missing_origin = accepted.to_dict()
        missing_origin.pop("tumor_id")
        with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("legacy evidence refitted")):
            expect_value_error(lambda: AcceptedFit.from_dict(missing_origin))
    replay_dir = directory / "same_tumor_replay"
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("replay refitted")):
        replay = run_workflow(path, replay_dir, config, accepted_stage_one=first, accepted_stage_two=second,
                              run_manifest_id="contract-current-replay")
    assert replay["counts"] == counts
    reused = tuple(load_accepted_fit(replay_dir / f"stage_{stage}_accepted.json") for stage in ("one", "two"))
    assert all(value.compatibility_receipts for value in reused)
    assert all(value.compatibility_receipts[-1]["run_manifest_id"] == "contract-current-replay" for value in reused)
    assert all(value.producer_receipt["run_manifest_id"] == "contract-external-build" for value in reused)
    _derived_failures(directory, path, config, first, second)
    _stage_two_preparation_failures(directory, path, config, first, second)
    other_path = directory / "other_tumor.tsv"
    other = write_catalog(other_path, alt, 100, 4, metadata=dict(metadata, tumor_id="tumor_B"))
    np.testing.assert_array_equal(catalog.alt_counts, other.alt_counts)
    with patch("CliPP2.kernel.model.RegionalModel", side_effect=AssertionError("wrong tumor reached numerical replay")):
        for index, pair in enumerate(((first, second), reused)):
            for accepted in pair:
                expect_value_error(lambda: accepted.validate(other))
            expect_value_error(lambda: run_workflow(other_path, directory / f"wrong_tumor_replay_{index}",
                config, accepted_stage_one=pair[0], accepted_stage_two=pair[1]))
    return dict(stage_two_targets=int(capped_targets.sum()), native_upper=1, stage_three_upper=4,
                non_target_centers_preserved=True, counts=counts, accepted_copy_immutable=True,
                repeated_replay_rejects_other_tumor_before_numerics=True,
                originating_manifest_tampering_rejected=True, accepted_stages_survive_derived_failures=True,
                current_and_originating_execution_links_separate=True)


def _manifest_tampering(catalog, accepted):
    """Recomputed content identities cannot supply missing originating evidence."""
    for change in ("empty_build", "version", "backend", "missing_input", "missing_catalog_identity"):
        producer = accepted.to_dict()["producer_receipt"]
        manifest = producer["run_manifest"]
        if change == "empty_build":
            manifest["build"] = {}
        elif change == "version":
            manifest["build"]["artifact"]["implementation_version"] = "altered_source_artifact"
        elif change == "backend":
            kind = producer["execution"]["native"]["kind"]
            manifest["build"]["artifact"][kind+"_build_id"] = "different_backend"
        elif change == "missing_input":
            manifest.pop("input_identity")
        else:
            manifest["input_identity"].pop("original_catalog_identity")
        manifest["manifest_identity"] = content_hash({key: manifest.get(key)
                                                      for key in ("run_manifest_id", "build", "input_identity")})
        changed = replace(accepted, producer_receipt=producer)
        changed = replace(changed, acceptance_evidence=dict(changed.acceptance_evidence,
                                                           accepted_state_hash=changed.state_fingerprint))
        with patch("CliPP2.kernel.model.RegionalModel", side_effect=AssertionError("invalid provenance reached numerics")):
            expect_value_error(lambda: changed.validate(catalog))


def _execution_tampering(catalog, accepted):
    """Recomputed fingerprints cannot hide incomplete producer execution."""
    from CliPP2.provenance import validate_fit_execution_receipt
    original = accepted.to_dict()["producer_receipt"]
    config, execution = original["fit_config"], original["execution"]
    validate_fit_execution_receipt(config, execution)
    # Both emitted continuation routes and explicit absence are valid without an
    # external record; wrong-key and missing evidence must never reach replay.
    for kind, route, key in (("cpu", "native_cpu", "build_id"),
                             ("cuda", "native_owner_cuda_emissions", "native_continuation_build_id")):
        record = dict(execution, device=kind, backend=f"native_{kind}_likelihood_host_forest",
            conditional_refit_backend=f"native_{kind}", weight_execution="cpu" if kind == "cpu" else "cuda_resident",
            native=dict(execution["native"], kind=kind), continuation={"continuation": route, key: "fixture_native"})
        validate_fit_execution_receipt(dict(config, device=kind), record)
        validate_fit_execution_receipt(dict(config, device=kind), dict(record, continuation=None))
        bad = copy.deepcopy(record)
        bad["continuation"].pop(key)
        expect_value_error(lambda: validate_fit_execution_receipt(dict(config, device=kind), bad))
        bad["continuation"][key] = None
        expect_value_error(lambda: validate_fit_execution_receipt(dict(config, device=kind), bad))
    variants = []
    for key in config:
        changed = copy.deepcopy(original)
        changed["fit_config"].pop(key)
        variants.append(changed)
        if key != "cpu_threads":
            changed = copy.deepcopy(original)
            changed["fit_config"][key] = None
            variants.append(changed)
    for key in ("build_id", "numerical_policy", "kind"):
        changed = copy.deepcopy(original)
        changed["execution"]["native"][key] = None
        variants.append(changed)
    for key, value in (("device", "cuda"), ("backend", "wrong_route"),
                       ("weight_execution", None), ("conditional_refit_backend", None),
                       ("continuation", {}), ("continuation", {"continuation": "native_cpu", "build_id": None})):
        changed = copy.deepcopy(original)
        changed["execution"][key] = value
        variants.append(changed)
    changed = copy.deepcopy(original)
    changed["execution"].pop("continuation")
    variants.append(changed)
    changed = copy.deepcopy(original)
    changed["execution"] = {}
    variants.append(changed)
    for producer in variants:
        # No optional external record can mask the structural rejection.
        manifest = producer["run_manifest"]
        manifest["build"] = None
        manifest["manifest_identity"] = content_hash({key: manifest[key]
            for key in ("run_manifest_id", "build", "input_identity")})
        changed = replace(accepted, producer_receipt=producer)
        changed = replace(changed, acceptance_evidence=dict(changed.acceptance_evidence,
                                                           accepted_state_hash=changed.state_fingerprint))
        with patch("CliPP2.kernel.model.RegionalModel", side_effect=AssertionError("invalid execution reached numerics")):
            expect_value_error(lambda: changed.validate(catalog))


def _derived_failures(directory, path, config, first, second):
    """A later diagnostic/publication error must not erase accepted native work."""
    for failed in ("stage_one_diagnostics", "stage_two_diagnostics", "assignment_publication"):
        output = directory / ("failure_"+failed)

        def classify(catalog, fit, settings):
            if failed == fit.source_id+"_diagnostics":
                raise RuntimeError("injected derived diagnostic failure")
            return classify_groups(catalog, fit, settings)

        from CliPP2.io.assignment import publish_assignment

        def publish(*args, **kwargs):
            if failed == "assignment_publication":
                raise RuntimeError("injected derived publication failure")
            return publish_assignment(*args, **kwargs)

        with patch("CliPP2.workflow.classify_groups", side_effect=classify), \
                patch("CliPP2.io.assignment.publish_assignment", side_effect=publish), \
                patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("replay refitted")):
            try:
                run_workflow(path, output, config, accepted_stage_one=first, accepted_stage_two=second)
            except RuntimeError as error:
                assert "injected derived" in str(error)
            else:
                raise AssertionError("Injected derived failure did not occur")
        receipt = json.loads((output / "workflow_receipt.json").read_text())
        assert receipt["status"] == "failed" and receipt["stage_one"] == "accepted"
        assert receipt["failed_phase"] == failed
        assert receipt["counts"]["N1"] == len(first.mutation_ids)
        if failed != "stage_one_diagnostics":
            assert receipt["stage_two"] == "accepted" and receipt["counts"]["N2"] == len(second.mutation_ids)
            assert load_accepted_fit(output / "stage_two_accepted.json").state_fingerprint == second.state_fingerprint


def _stage_two_preparation_failures(directory, path, config, first, second):
    """Stage 2 owns qualification/cap/publication errors before its native fit."""
    from CliPP2.io.assignment import write_supporting_table
    from CliPP2.workflow import _stage_two_caps
    for operation in ("qualification", "caps", "publication"):
        output = directory / ("failure_stage_two_preparation_" + operation)

        def qualify(*args, **kwargs):
            if operation == "qualification":
                raise RuntimeError("injected preparation failure")
            return qualify_contributors(*args, **kwargs)

        def caps(*args, **kwargs):
            if operation == "caps":
                raise RuntimeError("injected preparation failure")
            return _stage_two_caps(*args, **kwargs)

        def publish(destination, *args, **kwargs):
            if operation == "publication" and Path(destination).name == "stage2_segment_multiplicity_bounds.tsv":
                raise RuntimeError("injected preparation failure")
            return write_supporting_table(destination, *args, **kwargs)

        with patch.object(AcceptedFit, "validate", lambda self, *args, **kwargs: self), \
                patch("CliPP2.workflow.qualify_contributors", side_effect=qualify), \
                patch("CliPP2.workflow._stage_two_caps", side_effect=caps), \
                patch("CliPP2.io.assignment.write_supporting_table", side_effect=publish), \
                patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("preparation regression refitted")):
            try:
                run_workflow(path, output, config, accepted_stage_one=first, accepted_stage_two=second)
            except RuntimeError as error:
                assert str(error) == "injected preparation failure"
            else:
                raise AssertionError("Preparation failure was not injected")
        receipt = json.loads((output / "workflow_receipt.json").read_text())
        assert receipt["stage_one"] == "accepted" and receipt["stage_two"] == "failed"
        assert receipt["failed_stage"] == "stage_two" and receipt["failed_phase"] == "stage_two_cap_construction"
        assert receipt["stage_two_candidate_count"] == len(second.mutation_ids)
        assert receipt["counts"]["N1"] == len(first.mutation_ids)
        assert receipt["counts"]["N2"] == receipt["counts"]["Q2"] == 0


def _declared_fit(catalog, stage, ids, labels, centers, support, calls):
    """Controlled estimates isolate coordination; no optimizer discovery claim."""
    labels = np.asarray(labels, dtype=int)
    return AcceptedFit(stage, tuple(ids), catalog.region_ids, labels, centers,
        np.bincount(labels)/len(labels), calls, support, _biological_hash(catalog, ids),
        "declared_coordinator_fixture", 0., 0., 0., 0., {"optimizer_stub": True},
        biological_identity=catalog.biological_identity, tumor_id=catalog.tumor_id)


def _geometric_routing(directory):
    catalog = write_catalog(directory / "geometric.tsv",
        [[250, 250], [300, 300], [250, 0], [300, 300], [237, 250], [237, 250]], 1000, 4,
        observed=[[1, 1], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
    config = workflow_config()
    first = _declared_fit(catalog, "stage_one", catalog.mutation_ids, [0, 1, 1, 1, 1, 1],
        [[1., 1.], [.4, .4]], catalog.default_bounds(),
        np.where(catalog.count_observed, 1., np.nan))
    ids = catalog.mutation_ids[1:]
    caps = MultiplicityBounds(ids, catalog.region_ids, np.ones((5, 2)), np.ones((5, 2)))
    centers = [[1., 1.], [1., 1.], [.4, .4], [.95, 1.], [np.nextafter(.95, 1.), 1.]]
    second = _declared_fit(catalog, "stage_two", ids, np.arange(5), centers, caps,
        np.where(catalog.count_observed[1:], 1., np.nan))
    selected = classify_groups(catalog, first, config)[0]
    groups = classify_groups(catalog, second, config)
    assert selected["classification"] == "clonal_merge_eligible"
    assert groups[0]["near_clonal"] and not groups[0]["native_fit_pass"] and not groups[0]["all_one_fit_pass"]
    assert groups[1]["near_clonal"] and not groups[1]["coverage_pass"]
    assert not groups[2]["native_fit_pass"] and all(groups[2]["original_native_descriptive_pass"])
    assert not groups[3]["near_clonal"] and groups[4]["near_clonal"]
    boundary = groups[3]["distance_to_one"]
    assert not classify_groups(catalog, second, replace(config, clonal_distance_max=boundary))[3]["near_clonal"]
    assert classify_groups(catalog, second,
        replace(config, clonal_distance_max=np.nextafter(boundary, np.inf)))[3]["near_clonal"]
    reference, targets, unchanged, mapping = _build_stage_three_inputs(catalog, first, second, selected, groups, config)
    assert set(targets) == {catalog.mutation_ids[i] for i in (0, 1, 2, 5)} and reference.labels == (0, 1, 2)
    assert set(unchanged) == {catalog.mutation_ids[i] for i in (3, 4)}
    np.testing.assert_array_equal(reference.centers, [[1., 1.], [.4, .4], [.95, 1.]])
    assert not reference.regional_quality[1].any()
    assert all("native_center_descriptive_screen" in reasons for reasons in reference.regional_quality_reasons[1])
    assert [row["baseline_reference_label"] for row in mapping] == [0, 0, 0, 1, 2, 0]
    request = bundle(catalog, reference, target_ids=(ids[0],), native_upper=[[1, 1]])
    gated = assign_mutations(request).assignments[0]
    permissive = replace(reference, regional_quality=np.ones((3, 2), bool),
                         regional_quality_reasons=(((), ()),)*3)
    accepted = assign_mutations(replace(request, reference=permissive)).assignments[0]
    assert gated["status"] == "unresolved_reference" and gated["best_label"] == 1
    assert gated["accepted_label"] is None and accepted["accepted_label"] == 1
    np.testing.assert_array_equal(gated["probabilities"], accepted["probabilities"])
    np.testing.assert_array_equal(gated["log_scores"], accepted["log_scores"])
    return dict(controlled_estimates_not_discovery=True, failed_screens_and_coverage_routed=True,
                strict_boundary=True, frozen_inventory=3, native_capped_reference_quality_gates_acceptance=True)


def _historical_recovery(directory):
    path = directory / "historical_recovery.tsv"
    catalog = write_catalog(path, [[300], [300], [300]], 1000, 4)
    frame = pd.DataFrame([dict(row) for row in catalog.source_rows])
    frame["segment_id"] = ["affected", "affected", "anchor"]
    write_tumor_txt(path, frame, dict(catalog.metadata))
    catalog = load_original_catalog(path)
    # Explicitly permissive group/contributor screens allow the planted native
    # G0 estimate; the target reassessment retains its stricter ordinary screen.
    config = workflow_config(group_screen=DescriptiveScreen(20., 20., .05), contributor_deviance_max=20.)
    fits, assignments = [], []

    def fixture_fit(original, ids, support, outdir, stage, *args, **kwargs):
        if stage == "stage_one":
            value = _declared_fit(original, stage, ids, [0, 1, 1], [[1.], [.4]], support, [[1.], [3.], [3.]])
        else:
            np.testing.assert_array_equal(support.upper, [[1], [4]])
            value = _declared_fit(original, stage, ids, [0, 1], [[1.], [.4]], support, [[1.], [3.]])
        fits.append(value)
        return value

    def record_assignment(request, **kwargs):
        result = assign_mutations(request, **kwargs)
        assignments.append(result)
        return result

    output = directory / "historical_recovery"
    with patch("CliPP2.workflow._fit_accepted", side_effect=fixture_fit) as fitting, \
            patch("CliPP2.workflow.assign_mutations", side_effect=record_assignment) as assigning:
        receipt = run_workflow(path, output, config)
    assert fitting.call_count == 2 and assigning.call_count == 1 and receipt["status"] == "complete"
    result, = assignments
    one, two = result.assignments
    assert one["target_reason"] == "stage1_clonal_recheck" and two["target_reason"] == "stage2_near_clonal_recheck"
    np.testing.assert_array_equal(one["log_scores"], two["log_scores"])
    np.testing.assert_array_equal(one["probabilities"], two["probabilities"])
    assert one["accepted_label"] == two["accepted_label"] == 1
    np.testing.assert_allclose(one["posterior"], .998343610029, rtol=1e-10)
    final = receipt["final_accounting"]
    assert final["final_occupancy"] == {"0": 0, "1": 3} and final["unresolved_targets"] == 0
    assert final["historical_cap_contributors_reassigned"] == 1 and not final["historical_caps_recomputed"]
    request = load_assignment_bundle(output / "assignment_bundle.json")
    assert request.reference.labels == (0, 1) and np.all(request.bounds.upper == 4)
    policies = pd.read_csv(output / "stage2_segment_multiplicity_bounds.tsv", sep="\t")
    assert policies.loc[policies.segment_id == "affected", "multiplicity_endpoint"].item() == 1
    contributors = pd.read_csv(output / "stage2_reference_call_qualification.tsv", sep="\t")
    assert len(contributors) == 1 and contributors.usable.item() and contributors.native_call.item() == 1
    whole = pd.read_csv(output / "all_original_mutations.tsv", sep="\t")
    assert whole.final_assignment_label.tolist() == [1, 1, 1]
    assert whole.baseline_reference_label.tolist() == [0, 0, 1]
    # An unresolved target must never inherit its provisional clonal label.
    bad_reference = replace(request.reference, regional_quality=np.array([[True], [False]]),
                            regional_quality_reasons=(((),), (("declared_quality_failure",),)))
    unresolved = assign_mutations(replace(request, reference=bad_reference))
    rows = pd.read_csv(output / "source_to_reference.tsv", sep="\t").to_dict("records")
    unchanged = {row.mutation_id: int(row.baseline_reference_label)
                 for row in whole.itertuples() if row.role == "non_target"}
    original_rows, accounting = _reconcile_final_memberships(catalog, *fits, rows, unchanged, unresolved,
                                                            contributors=contributors.to_dict("records"))
    assert accounting["final_occupancy"] == {"0": 0, "1": 1} and accounting["unresolved_targets"] == 2
    unresolved_rows = pd.DataFrame(original_rows)
    assert unresolved_rows.loc[unresolved_rows.role == "target", "final_assignment_label"].isna().all()
    expect_value_error(lambda: replace(thresholds(), fit_assessment="bootstrap"))
    return dict(controlled_estimates_not_discovery=True, identical_origin_scores=True,
                zero_clonal_occupancy=True, historical_contributor_moves_without_cap_recomputation=True,
                unresolved_has_no_baseline_fallback=True, bootstrap_rejected_before_execution=True)


def run_workflow_edge_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return dict(blocked_stage_two=_blocked_count(directory),
                contributor_certainty=_contributor_certainty(directory),
                capped_target_and_replay=_capped_target_and_replay(directory),
                geometric_routing=_geometric_routing(directory),
                historical_recovery=_historical_recovery(directory))
