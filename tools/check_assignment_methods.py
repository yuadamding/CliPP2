"""Method-bound Stage 3 regressions; CPU smoke fixtures, optionally CUDA."""
from dataclasses import replace
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2.assignment import (AssignmentBundle, AssignmentResult, POSTERIOR_FIELDS,
    assign_mutations, derive_assignment_bundle, profile_score_summary, _validated_profile_terms)
from CliPP2.io.assignment import (load_assignment_bundle, save_assignment_bundle,
    validate_assignment_result)
from CliPP2.kernel.backend import NativeBackend
from check_support import bundle, fixture_reference, write_catalog


def _fails(operation, error=ValueError):
    try:
        operation()
    except error:
        return
    raise AssertionError("Invalid method evidence was accepted")


def run_checks(root, *, device="cpu"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    catalog = write_catalog(root / "input.tsv", [[5]], 10, 2)
    reference = fixture_reference((0, 1), catalog.region_ids, [[1.], [.6]])
    marginal = replace(bundle(catalog, reference), device=device)
    profile = derive_assignment_bundle(marginal, assignment_method="profile")
    assert marginal.content_hash != profile.content_hash
    assert derive_assignment_bundle(profile) is profile
    assert marginal.assignment_method == "marginal"
    _fails(lambda: replace(marginal, assignment_method="minimum_conditional_entropy"))
    _fails(lambda: assign_mutations(marginal, assignment_method="profile"), TypeError)
    first = assign_mutations(marginal, outdir=root / "marginal")
    second = assign_mutations(profile, outdir=root / "profile")
    assert first.assignments[0]["final_assignment_label"] == 1
    assert second.assignments[0]["final_assignment_label"] == 0
    np.testing.assert_allclose(first.assignments[0]["log_scores"], np.log(np.array([.123046875, .151788735])/2), rtol=1e-14)
    np.testing.assert_allclose(second.assignments[0]["log_scores"], np.log(np.array([.24609375, .2006581248])/2), rtol=1e-14)
    for row in second.assignments:
        assert row["score_defined"] and not row["posterior_defined"]
        assert all(row[field] is None for field in POSTERIOR_FIELDS)
    assert load_assignment_bundle(root / "profile" / "assignment_bundle.json").content_hash == profile.content_hash
    restored = AssignmentResult.from_dict(second.to_dict())
    validate_assignment_result(profile, restored)
    _fails(lambda: validate_assignment_result(marginal, second))
    for field, value in (("probabilities", [.5, .5]), ("posterior", .5), ("posterior_defined", True),
                         ("score_defined", False), ("selection_reason", "lowest_label_in_exact_tie")):
        changed = second.to_dict()
        changed["assignments"][0][field] = value
        _fails(lambda: validate_assignment_result(profile, AssignmentResult.from_dict(changed)))

    # Distinct vectors attain exactly the same binomial maximum at m=1 and m=2.
    tied_catalog = write_catalog(root / "tie.tsv", [[5]], 20, 4)
    tied = replace(bundle(tied_catalog, fixture_reference((0, 1), tied_catalog.region_ids, [[1.], [.5]])),
                   device=device, assignment_method="profile", baseline_labels=(1,))
    selected = assign_mutations(tied, outdir=root / "tie")
    row, pair = selected.assignments[0], selected.multiplicities[0]
    assert row["map_set"] == (0, 1) and not row["map_unique"]
    assert all(row[key] == 1 for key in ("best_label", "map_label", "map_cluster_label", "final_assignment_label"))
    assert row["second_best_label"] == 0 and row["log_score_margin"] == 0
    assert row["selection_reason"] == "baseline_in_exact_tie"
    assert pair["phi"] == .5 and pair["conditional_map"] == pair["assigned_multiplicity"] == 2
    reversed_reference = replace(tied.reference, labels=(1, 0), centers=tied.reference.centers[::-1],
        center_sources=tied.reference.center_sources[::-1])
    reversed_result = assign_mutations(replace(tied, reference=reversed_reference), outdir=root / "tie_reversed")
    assert reversed_result.assignments[0]["final_assignment_label"] == 1
    assert reversed_result.multiplicities[0]["assigned_multiplicity"] == 2
    for scores, labels, baseline, expected in (([0., 0.], (7, 2), 7, 7),
            ([0., 0.], (7, 2), 9, 2), ([0., np.nextafter(0., 1.)], (7, 2), 7, 2)):
        summary = profile_score_summary(scores, labels, baseline)
        assert summary["best_label"] == labels[summary["best_index"]] == expected
        assert summary["second_index"] != summary["best_index"]

    wide_catalog = write_catalog(root / "wide.tsv", [[80]], 100, 10)
    wide = replace(bundle(wide_catalog, fixture_reference((0,), wide_catalog.region_ids, [[1.]]),
                          native_upper=[[2]]), device=device, assignment_method="profile")
    wide_result = assign_mutations(wide, outdir=root / "wide")
    assert wide_result.multiplicities[0]["conditional_map"] == 8
    np.testing.assert_array_equal(wide_result.multiplicities[0]["multiplicity_candidates"], np.arange(1, 11))
    for name, major in (("regional", [[1, 4, 8]]), ("singleton", 1)):
        cat = write_catalog(root / (name+".tsv"), [[7, 13, 25], [15, 20, 40]], 100, major,
                            minor=1, purity=[.6, .8, 1.])
        ref = fixture_reference((0, 1, 2), cat.region_ids, [[1., 1., 1.], [.3, .7, .9], [0., .2, 1.]])
        request = replace(bundle(cat, ref), device=device)
        profiled = assign_mutations(replace(request, assignment_method="profile"), outdir=root / name)
        if name == "singleton":
            for a, b in zip(assign_mutations(request).assignments, profiled.assignments):
                np.testing.assert_allclose(a["log_scores"], b["log_scores"], rtol=1e-13, atol=1e-12)

    for name, alt, depth, observed, centers, state in (
            ("zero", 0, 20, True, [[0.]], "observed"),
            ("prior", 0, 0, True, [[1.], [.5]], "prior_only"),
            ("missing", 1, 20, False, [[1.], [.5]], "prior_only"),
            ("impossible", 1, 20, True, [[0.]], "model_incompatible")):
        cat = write_catalog(root / (name+".tsv"), [[alt]], depth, 2, observed=observed)
        ref = fixture_reference(tuple(range(len(centers))), cat.region_ids, centers,
                                clonal_label=0 if centers[0] == [1.] else None)
        request = replace(bundle(cat, ref), device=device, assignment_method="profile")
        answer = assign_mutations(request, outdir=root / name)
        assert answer.assignments[0]["evidence_status"] == state
        assert answer.assignments[0]["probabilities"] is None
        assert np.isnan(answer.multiplicities[0]["assigned_multiplicity"])
        if name == "zero":
            assert answer.multiplicities[0]["map_tied_multiplicities"] == (1, 2)
        if state != "observed":
            assert answer.assignments[0]["final_assignment_label"] is None

    for status, value in ((2, 0.), (3, 0.), (4, 0.), (9, 0.), (0, np.nan), (0, np.inf), (0, -np.inf), (1, 0.)):
        _fails(lambda: _validated_profile_terms({"status": [[status]], "max_component_log_mass": [[value]]}, [[True]]), ArithmeticError)
    assert _validated_profile_terms({"status": [[4]], "max_component_log_mass": [[np.nan]]}, [[False]])[0, 0] == 0
    assert np.isneginf(_validated_profile_terms({"status": [[1]], "max_component_log_mass": [[-np.inf]]}, [[True]])[0, 0])
    with patch.object(NativeBackend, "component_diagnostics", side_effect=RuntimeError("mandatory scorer failure")):
        _fails(lambda: assign_mutations(profile), RuntimeError)
    original = NativeBackend.component_diagnostics
    def diagnostics(self, centers, native_calls=None):
        if native_calls is not None:
            raise RuntimeError("optional selected diagnostic failure")
        return original(self, centers)
    with patch.object(NativeBackend, "component_diagnostics", diagnostics):
        optional = assign_mutations(profile, outdir=root / "optional_failure")
    assert optional.assignments[0]["diagnostic_status"] == "failed"
    assert optional.multiplicities[0]["assigned_multiplicity"] == second.multiplicities[0]["assigned_multiplicity"]
    with patch("CliPP2.assignment.conditional_multiplicity", side_effect=RuntimeError("essential enumeration failure")):
        _fails(lambda: assign_mutations(profile), ArithmeticError)
    with patch("CliPP2.assignment.binomial_profile_deviance", side_effect=RuntimeError("optional deviance failure")):
        optional = assign_mutations(profile, outdir=root / "deviance_failure")
    assert optional.multiplicities[0]["conditional_distribution_defined"]

    for version in (4, 5):
        old = marginal.to_dict()
        old.update(schema=f"clipp2.assignment.bundle.v{version}", protocol_version="2.3")
        old.pop("assignment_method")
        historical = AssignmentBundle.from_dict(old)
        before = copy.deepcopy(historical.to_dict())
        derived = derive_assignment_bundle(historical, assignment_method="profile")
        assert historical.to_dict() == before and derived.content_hash != historical.content_hash
        assert derived.catalog.content_hash == marginal.catalog.content_hash
        assert derived.reference.to_dict() == marginal.reference.to_dict()
        assert derived.provenance["assignment_derivation"]["source_bundle_hash"] == historical.content_hash
    if device == "cpu":
        from compare_stage3_likelihood import compare
        truth = pd.DataFrame(dict(mutation_id=catalog.mutation_ids, region_id=catalog.region_ids,
            truth_cluster=["clonal"], true_ccf=[1.], true_multiplicity=[1.], cna=[1], observed=[1])).set_index(["mutation_id", "region_id"])
        comparison_dir = root / "comparison"
        comparison_dir.mkdir()
        compared = compare(marginal, truth, comparison_dir)
        assert compared["native_fit_count"] == 0 and compared["assignment_pass_count"] == 2
        assert compared["scoring_changes"] == 1 and compared["tie_policy_changes"] == 0
        assert compared["attribution_max_error"] < 1e-12
        from CliPP2.cli import main
        save_assignment_bundle(marginal, root / "request.json")
        main(["assign", "--bundle", str(root / "request.json"), "--assignment-method", "profile", "--output-dir", str(root / "cli")])
        executed = load_assignment_bundle(root / "cli" / "assignment_bundle.json")
        assert executed.content_hash == profile.content_hash
        historical_path = root / "historical.json"
        historical_path.write_text(json.dumps(dict(payload=historical.to_dict(), content_hash=historical.content_hash)))
        _fails(lambda: main(["assign", "--bundle", str(historical_path), "--output-dir", str(root / "reject")]))
        main(["assign", "--bundle", str(historical_path), "--derive-current", "--assignment-method", "profile",
              "--output-dir", str(root / "derived_cli")])
    return dict(device=device, unique_marginal_profile_reversal=True, exact_tie_baseline_not_first=True,
        original_support_restored=True, mandatory_status_failure=True, optional_failure_preserves_multiplicity=True,
        profile_has_no_cluster_posterior=True, independent_publication=True, historical_derivation=True)
