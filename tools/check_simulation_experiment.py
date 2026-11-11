"""Fast actual-fit regression for the paired simulation runner (default 10/20)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2.io.tumor_txt import write_tumor_txt
from CliPP2.assignment import assign_mutations
from CliPP2.simulation.design import workflow_config
from CliPP2.simulation.experiment import run_paired_experiment
from CliPP2.io.assignment import publish_assignment


def _input(path, values, *, tumor="paired_regression"):
    frame = pd.DataFrame([dict(mutation_id=f"m{i:03}", sample_id=f"r{j}",
        alt_count=alt, ref_count=100-alt, purity=1., normal_cn=2,
        segment_id="s0", cn_state_id="0", cn_state_fraction=1.,
        allele_a_cn=1, allele_b_cn=1, count_observed=1, mapping_genuine=1, normal_cn_supplied=1)
        for i, pair in enumerate(values) for j, alt in enumerate(pair)])
    write_tumor_txt(path, frame, metadata=dict(tumor_id=tumor))


def run_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = workflow_config()
    assert config.fit.max_clusters == 10 and config.stage_two_max_clusters == 20
    _input(directory / "input.tsv", [[50, 50]]*4 + [[20, 15]]*4)
    with patch("CliPP2.simulation.experiment.publish_assignment", wraps=publish_assignment) as publisher:
        paired = run_paired_experiment(directory / "input.tsv", directory / "paired", config,
                                       run_manifest_id="simulation_check_origin")
    assert publisher.call_count == 2
    assert all(call.kwargs["workflow_accounting"]["population_count"] == 8 for call in publisher.call_args_list)
    assert paired["status"] == "complete", paired.get("error")
    ledger = paired["cost_ledger"]
    assert (ledger["fit_count"], ledger["assignment_pass_count"]) == (3, 2)
    assert [event["arm"] for event in ledger["events"] if event["operation"] == "fit"] == ["A", "B", "C"]
    assert paired["protocol_version"] == "2.3"
    assert all(fit["producer_receipt"]["run_manifest"] == paired["run_manifest"] for fit in
               [paired["stage_one"], *(arm["stage_two"] for arm in paired["arms"].values())])
    for arm in paired["arms"].values():
        assert arm["status"] == "complete"
        assert set(arm["stage_two"]["mutation_ids"]) == set(paired["remainder_ids"])
        assert arm["stage_two"]["acceptance_evidence"]["membership_audit"]["complete"]
        assert arm["counts"]["Ncombined"] == arm["counts"]["N0"] + arm["counts"]["N2"] == 8
        assert set(arm["targets"]).isdisjoint(arm["non_targets"])
        assert set(arm["targets"] + arm["non_targets"]) == set(arm["combined_labels"])
        assert arm["counts"]["clonal_source_consolidated"] == 1
        assert arm["counts"]["K"] == 1 + arm["counts"]["Q2"] - arm["counts"]["exact_consolidated_centers"]
        assert len(arm["assignment"]["multiplicities"]) == 2*len(arm["targets"])
        assert not arm["non_targets"] and arm["counts"]["F"] == 0
        assert all(row["role"] == "target" for row in arm["original_rows"])
        assert all(row["original_lower"] == row["original_upper"] == 1
                   for row in arm["assignment"]["multiplicities"])
        native = arm["stage_two"]
        for i, mutation in enumerate(native["mutation_ids"]):
            np.testing.assert_array_equal(arm["native_centers"][mutation], native["centers"][native["labels"][i]])
    for left, right in zip(paired["arms"]["B"]["assignment"]["assignments"],
                           paired["arms"]["C"]["assignment"]["assignments"]):
        np.testing.assert_array_equal(left["probabilities"], right["probabilities"])
    checks = ["shared_A_same_remainder_3fits_2passes", "native_audits_and_population_reconciliation",
              "single_copy_B_C_equivalence", "full_support_all_target_reassessment_preserves_native_baselines",
              "one_external_run_manifest_link_shared_by_A_B_C", "publish_once_after_combined_reconciliation"]
    checks.extend(_routing_regressions(directory, paired, config))
    replay = run_paired_experiment(directory / "input.tsv", directory / "replay",
        config, accepted_stage_one=paired["stage_one"],
        accepted_uncapped_stage_two=directory / "paired/arm_outputs/B/accepted_fit.json",
        run_manifest_id="simulation_check_current_replay")
    assert replay["status"] == "complete", replay.get("error")
    assert (replay["cost_ledger"]["fit_count"], replay["cost_ledger"]["reused_fit_count"],
            replay["cost_ledger"]["assignment_pass_count"]) == (1, 2, 2)
    checks.append("validated_A_B_reuse_without_hidden_refit")
    for fit in (replay["stage_one"], replay["arms"]["B"]["stage_two"]):
        assert fit["producer_receipt"]["run_manifest_id"] == "simulation_check_origin"
        assert fit["compatibility_receipts"][-1]["run_manifest_id"] == "simulation_check_current_replay"
    checks.append("original_producer_and_current_execution_links_remain_separate")
    checks.extend(_replay_policy_regressions(directory, paired, config))
    checks.extend(_accepted_stage_failure(directory, paired, config))
    _input(directory / "other_tumor.tsv", [[50, 50]]*4 + [[20, 15]]*4, tumor="other_tumor")
    rejected = run_paired_experiment(directory / "other_tumor.tsv", directory / "rejected", config,
                                     accepted_stage_one=paired["stage_one"])
    assert rejected["status"] == "failed" and rejected["cost_ledger"]["fit_count"] == 0
    assert "biological input identities" in rejected["error"]
    with patch("CliPP2.fit_execution.fit_regions", side_effect=AssertionError("incompatible B started fresh A")) as fitter:
        rejected_b = run_paired_experiment(directory / "other_tumor.tsv", directory / "rejected_stage_two", config,
            accepted_uncapped_stage_two=paired["arms"]["B"]["stage_two"])
    assert rejected_b["status"] == "failed" and rejected_b["failed_stage"] == "stage_two"
    assert rejected_b["cost_ledger"]["fit_count"] == fitter.call_count == 0
    checks.append("cross_tumor_A_or_B_replay_rejected_before_any_refitting")
    for name, values in (("empty_remainder", [[50, 50]]*8), ("no_reference", [[20, 15]]*8)):
        _input(directory / (name + ".tsv"), values)
        result = run_paired_experiment(directory / (name + ".tsv"), directory / name, config)
        assert result["cost_ledger"]["fit_count"] == 1
        if name == "empty_remainder":
            assert result["status"] == "complete" and result["cost_ledger"]["assignment_pass_count"] == 2
            assert all(arm["counts"]["K"] == 1 and arm["counts"]["N2"] == 0 for arm in result["arms"].values())
        else:
            assert result["status"] == "complete" and result["cost_ledger"]["assignment_pass_count"] == 2
            assert result["reference_source"]["distance_to_one"] > .05
            assert result["reference_source"]["reference_source_role"] == "provisional_reference_source"
        assert all(len(arm["original_rows"]) == 8 for arm in result["arms"].values())
        checks.append(name + "_preserves_applicability_and_original_population")
    checks.extend(_control_regressions(directory / "control_regressions"))
    return dict(status="passed", checks=checks, directory=str(directory),
                default_max_clusters=[config.fit.max_clusters, config.stage_two_max_clusters])


def _replay_policy_regressions(directory, paired, config):
    from CliPP2.workflow import AcceptedFit
    from CliPP2.provenance import validate_replay_policy
    first = AcceptedFit.from_dict(paired["stage_one"])
    second = AcceptedFit.from_dict(paired["arms"]["B"]["stage_two"])
    first_identity, second_identity = first.state_fingerprint, second.state_fingerprint
    variants = dict(stage_one_capacity=replace(config, fit=replace(config.fit, max_clusters=1)),
        larger_capacity=replace(config, fit=replace(config.fit, max_clusters=11)),
        stage_two_capacity=replace(config, stage_two_max_clusters=1),
        all_audit=replace(config, fit=replace(config.fit, audit_scope="all")),
        normal_assumption=replace(config, normal_cn_policy="diploid"))
    for name, settings in variants.items():
        with patch("CliPP2.fit_execution.fit_regions", side_effect=AssertionError("paired policy rejection optimized")) as fitter:
            failed = run_paired_experiment(directory / "input.tsv", directory / ("policy_"+name), settings,
                accepted_stage_one=first, accepted_uncapped_stage_two=second)
            assert failed["status"] == "failed" and failed["cost_ledger"]["fit_count"] == 0 and fitter.call_count == 0
            assert failed["failed_stage"] == ("stage_two" if name == "stage_two_capacity" else "stage_one")
    with patch("CliPP2.fit_execution.fit_regions", side_effect=AssertionError("B policy started fresh A optimizer")) as fitter:
        failed = run_paired_experiment(directory / "input.tsv", directory / "policy_before_fresh_A",
            variants["stage_two_capacity"], accepted_uncapped_stage_two=second)
        assert failed["status"] == "failed" and fitter.call_count == 0
    settings = replace(config, fit=replace(config.fit, cpu_threads=2, max_major_cn=11))
    replay = run_paired_experiment(directory / "input.tsv", directory / "policy_thread_override", settings,
        accepted_stage_one=first, accepted_uncapped_stage_two=second)
    assert replay["status"] == "complete"
    assert replay["cost_ledger"]["fit_count"] == 1 and replay["cost_ledger"]["reused_fit_count"] == 2
    assert replay["stage_one"]["producer_receipt"]["fit_config"]["cpu_threads"] == config.fit.cpu_threads
    validate_replay_policy(first, replace(config.fit, device="cuda", cpu_threads=2), normal_cn_policy=config.normal_cn_policy)
    assert (first.state_fingerprint, second.state_fingerprint) == (first_identity, second_identity)
    return ["shared_scientific_replay_policy_rejects_without_optimization",
            "stage_two_policy_preflight_precedes_fresh_stage_one", "replay_execution_overrides_preserve_producer_policy", "unchanged_population_allows_explicit_cutoff_policy_change"]


def _routing_regressions(directory, paired, config):
    """Frozen coordinator inputs preserve near-one and nonclonal alternatives."""
    from CliPP2.io.tumor_txt import load_original_catalog
    from CliPP2.simulation.experiment import _freeze_arm
    from CliPP2.workflow import AcceptedFit, _reconcile_final_memberships
    first = AcceptedFit.from_dict(paired["stage_one"])
    native = AcceptedFit.from_dict(paired["arms"]["C"]["stage_two"])
    second = replace(native, labels=np.array([0, 0, 1, 1]), centers=np.array([[.98, .98], [.4, .3]]),
                     weights=np.array([.5, .5]))
    selected = next(row for row in paired["stage_one_groups"] if row["source_label"] == paired["g0_label"])
    groups = [dict(source_label=0, native_center=second.centers[0].tolist(), informative_counts=[0, 0], informative_depths=[0., 0.],
                   native_mean_profile_deviance=[float("nan")]*2, native_maximum_profile_deviance=[float("nan")]*2,
                   diagnostic_status="complete"),
              dict(source_label=1, native_center=second.centers[1].tolist(), informative_counts=[2, 2], informative_depths=[200., 200.],
                   native_mean_profile_deviance=[100., 0.], native_maximum_profile_deviance=[200., 0.],
                   diagnostic_status="complete")]
    catalog = load_original_catalog(directory / "input.tsv")
    path = directory / "routing_fixture"
    path.mkdir()
    bundle, frozen = _freeze_arm(catalog, first, selected, second, groups, paired["g0_ids"], config, "C", path)
    assert set(bundle.target_ids) == set(first.mutation_ids) and not bundle.non_target_ids
    assert frozen["counts"]["F"] == 0 and frozen["counts"]["K"] == 3
    assert any(np.array_equal(center, [.98, .98]) for center in bundle.reference.centers)
    assert any(np.array_equal(center, [.4, .3]) for center in bundle.reference.centers)
    for mutation, baseline in zip(bundle.target_ids, bundle.baseline_labels):
        if mutation in paired["g0_ids"]:
            assert baseline == 0
            np.testing.assert_array_equal(bundle.reference.centers[baseline], [1., 1.])
        else:
            np.testing.assert_array_equal(bundle.reference.centers[baseline], frozen["native_centers"][mutation])
    scored = assign_mutations(bundle, outdir=directory / "routing_assignment")
    assert all(row["target_reason"] == "stage2_recheck" for row in scored.assignments
               if row["mutation_id"] in native.mutation_ids)
    rows, accounting = _reconcile_final_memberships(catalog, first, second,
        frozen["source_map"], frozen["unchanged_map"], scored, contributors=paired["contributors"])
    assert sum(accounting["final_occupancy"].values()) + accounting["unresolved_targets"] == 8
    assert all(row["role"] == "target" for row in rows)
    return ["all_native_references_and_all_fitted_targets_preserved", "native_baselines_use_exact_reference_lineage",
            "near_one_center_not_geometrically_merged", "all_target_population_reconciles"]


def _accepted_stage_failure(directory, paired, config):
    from CliPP2.workflow import AcceptedFit, classify_groups
    def classify(catalog, fit, settings):
        if fit.source_id == "stage_two":
            raise RuntimeError("derived_stage_two_diagnostics_fixture")
        return classify_groups(catalog, fit, settings)
    # Replayed/native stubs isolate post-acceptance accounting without extra fits.
    with patch("CliPP2.simulation.experiment._fit_accepted",
               return_value=AcceptedFit.from_dict(paired["arms"]["C"]["stage_two"])), \
            patch("CliPP2.simulation.experiment.classify_groups", side_effect=classify):
        failed = run_paired_experiment(directory / "input.tsv", directory / "derived_failure", config,
            accepted_stage_one=paired["stage_one"], accepted_uncapped_stage_two=paired["arms"]["B"]["stage_two"])
    assert failed["stage_one_status"] == "accepted" and failed["status"] == "partial_failure"
    for arm in failed["arms"].values():
        assert arm["stage_two_status"] == "accepted" and arm["counts"]["N2"] == 4
        assert arm["stage_two_candidate_count"] == 4 and arm["failed_stage"] == "classification"
        assert arm["stage_two"]["acceptance_evidence"]["independently_verified"]
    return ["accepted_stage_two_evidence_survives_derived_diagnostic_failure"]


def _control_regressions(directory):
    from CliPP2.simulation.biology import generate_history
    from CliPP2.simulation.controls import (_calibration, _catalog, _snapshot_bundle,
                                           _source_arrays, _truth_reference)
    history = generate_history("S01", 300, n_mutations=12, n_segments=2, n_regions=2)
    catalog = _catalog(history, directory, seed=412)
    reference, members = _truth_reference(history)
    mutation = history.metadata["mutation_classes"]["E"][0]
    sources = _source_arrays(history)
    # A first-read zero-CCF/singleton call stays a valid display baseline even
    # when independent reads have ALT>0. It need not fit the second read sample.
    sources[mutation] = dict(center=[0., 0.], call=[1., 1.], lower=[1, 1], upper=[1, 1],
                             stage="test_first_read_zero_ccf_snapshot", label=0)
    bundle = _snapshot_bundle(history, catalog, reference, (mutation,), workflow_config(),
                              sources=sources, independent=True)
    assert np.all(bundle.native_centers == 0) and np.all(bundle.native_calls == 1)
    scored = assign_mutations(bundle, outdir=directory / "independent_assignment")
    assert len(scored.assignments) == 1
    fake = SimpleNamespace(reference=reference, assignments=[dict(mutation_id=mutation,
        status="observed", evidence_status="observed", final_assignment_label=1,
        probabilities=(0., 1., *([0.]*(len(reference.labels)-2))),
        log_scores=(-1000., 0., *([-2000.]*(len(reference.labels)-2))))])
    row = _calibration(history, catalog, fake, members)["mutations"][0]
    assert np.isclose(row["binary_log_loss"], 1000.) and np.isclose(row["multiclass_log_loss"], 1000.)
    return ["independent_counts_preserve_valid_first_read_baseline",
            "finite_log_loss_survives_probability_display_underflow"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_checks(args.outdir or tempfile.mkdtemp(prefix="clipp2-paired-check-")), indent=2))
