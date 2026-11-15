"""Read-derived multiplicity floors, native fitting, and original-support reassignment."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2 import FitConfig, run_workflow
from CliPP2.io.assignment import load_accepted_fit, load_assignment_bundle
from CliPP2.io.data import OriginalCatalog
from CliPP2.io.multiplicity import read_floor_bounds
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.workflow import _stage_two_caps

from check_support import expect_value_error, workflow_config, write_catalog


def _catalog(directory, name, rows, *, missing_counts="unobserved"):
    path = write_tumor_txt(directory / (name + ".tsv"), pd.DataFrame(rows))
    return load_original_catalog(path, missing_counts=missing_counts)


def _row(mutation, *, alt=30, depth=100, purity=1., major=4, minor=0,
         normal=2, observed=True, excluded=False):
    return dict(mutation_id=mutation, sample_id="r1", alt_count=alt,
        ref_count=depth-alt, count_observed=int(observed), purity=purity,
        normal_cn=normal, segment_id="segment", cn_state_id="0",
        cn_state_fraction=1., allele_a_cn=major, allele_b_cn=minor,
        input_excluded=int(excluded))


def _estimator_cases(directory):
    catalog = _catalog(directory, "floor_rounding_and_clamp", [
        _row("floor", alt=37, major=6, minor=2),
        _row("high", alt=99, major=6, minor=2),
        _row("zero", alt=0, major=6, minor=2)])
    bounds, audit = read_floor_bounds(catalog)
    np.testing.assert_array_equal(bounds.lower[:, 0], [2, 6, 1])
    np.testing.assert_array_equal(bounds.upper[:, 0], [6, 6, 6])
    assert audit[0]["raw_multiplicity_estimate"] == 2.96
    assert audit[1]["multiplicity_lower"] > 4
    # The estimator always assumes two normal copies, independently of the
    # original likelihood's supplied normal CN.
    for normal in (0, 1, 2):
        catalog = _catalog(directory, f"normal_{normal}", [
            _row("normal", alt=30, purity=.5, minor=1, normal=normal)])
        bounds, audit = read_floor_bounds(catalog)
        assert bounds.lower.item() == 2
        np.testing.assert_allclose(audit[0]["raw_multiplicity_estimate"], 2.1)
    # Canonical input purity is the regional minimum; this boundary detects
    # accidental replacement of the original supplied per-row purity.
    low, high = .799999999925, .800000000075
    catalog = _catalog(directory, "source_purity", [
        _row("a_low", alt=4, depth=9, purity=low, major=3, minor=1),
        _row("b_high", alt=4, depth=9, purity=high, major=3, minor=1)])
    assert catalog.purity[0, 0] == catalog.purity[1, 0] == low
    bounds, audit = read_floor_bounds(catalog)
    np.testing.assert_array_equal(bounds.lower[:, 0], [2, 1])
    assert [row["input_purity"] for row in audit] == [low, high]
    catalog = _catalog(directory, "uninformative", [
        _row("a_observed", alt=75), _row("b_missing", alt=80, observed=False),
        _row("c_zero_depth", alt=0, depth=0), _row("d_excluded", alt=100, excluded=True)])
    bounds, audit = read_floor_bounds(catalog)
    np.testing.assert_array_equal(bounds.lower[:, 0], [3, 1, 1, 1])
    assert len({row["estimate_status"] for row in audit}) == 4
    # Missing-read imputation must not manufacture an original observation.
    imputed = _catalog(directory, "imputed", [
        _row("a_observed", alt=75), _row("b_missing", alt=80, observed=False)],
        missing_counts="zero_alt")
    assert imputed.count_observed.all() and imputed.alt_counts[1, 0] == 0
    assert read_floor_bounds(imputed)[0].lower[1, 0] == 1
    tiny_purity = np.nextafter(0., 1.)
    extreme = _catalog(directory, "tiny_purity", [
        _row("positive", alt=1, purity=tiny_purity, major=6),
        _row("zero", alt=0, purity=tiny_purity, major=6)])
    np.testing.assert_array_equal(read_floor_bounds(extreme)[0].lower[:, 0], [6, 1])
    return dict(floor_not_round=True, zero_alt_one=True, upper_above_four=True,
        fixed_diploid_normal=True, original_supplied_purity=True,
        missing_excluded_zero_depth_unrestricted_lower=True,
        tiny_purity_clamped_before_integer_conversion=True)


def _segment_cap_conflict(directory):
    catalog = write_catalog(directory / "cap_conflict.tsv", [[25], [75]], 100, 4)
    lower, _ = read_floor_bounds(catalog)
    audits = [dict(region_id="r1", segment_id="shared_segment", contributes_to_cap=True,
                   cap_source_map_upper=1)]
    bounds, policies = _stage_two_caps(catalog, audits, lower_bounds=lower)
    np.testing.assert_array_equal(bounds.lower[:, 0], [1, 3])
    np.testing.assert_array_equal(bounds.upper[:, 0], [1, 3])
    assert policies[0]["multiplicity_endpoint"] == 1
    assert policies[0]["lower_bound_override_count"] == 1
    fallback, policies = _stage_two_caps(catalog, [], lower_bounds=lower)
    np.testing.assert_array_equal(fallback.lower, lower.lower)
    np.testing.assert_array_equal(fallback.upper, catalog.major_cn)
    assert policies[0]["fallback"]
    return dict(read_floor_wins_conflicting_segment_cap=True, no_empty_intervals=True,
                original_upper_without_contributors=True)


def _mixed_cn_estimate(directory):
    def estimate(name, states, *, alt=80, major=3):
        row = dict(_row("mixed", alt=alt, major=major, minor=1), segment_cn_states=json.dumps(states))
        catalog = _catalog(directory, name, [row])
        return catalog, *read_floor_bounds(catalog)
    single = [dict(cn_state_id="single", cn_state_fraction=1., major_cn=3, minor_cn=1)]
    mixed = [dict(cn_state_id="gain", cn_state_fraction=.6, major_cn=3, minor_cn=1),
             dict(cn_state_id="base", cn_state_fraction=.4, major_cn=2, minor_cn=1)]
    _, unchanged, single_audit = estimate("single_cn", single)
    catalog, bounds, mixed_audit = estimate("mixed_cn", mixed)
    assert unchanged.lower.item() == 3 and bounds.lower.item() == 2
    assert single_audit[0]["cn_state_count"] == 1 and mixed_audit[0]["cn_state_count"] == 2
    np.testing.assert_allclose(mixed_audit[0]["weighted_total_cn"], 3.6)
    np.testing.assert_allclose(mixed_audit[0]["raw_multiplicity_estimate"], 2.88)
    assert bounds.upper.item() == 3
    # Retained original CN remains the native denominator and upper clamp.
    assert catalog.total_cn.item() == 4 and catalog.major_cn.item() == 3
    assert catalog.prediction_data(catalog.mutation_ids).total_cn.item() == 4
    original, clamped, original_audit = estimate("original_cn_cap", mixed, alt=100, major=2)
    np.testing.assert_allclose(original_audit[0]["weighted_total_cn"], 3.6)
    np.testing.assert_allclose(original_audit[0]["raw_multiplicity_estimate"], 3.6)
    assert clamped.lower.item() == clamped.upper.item() == 2
    assert original.total_cn.item() == original.fit_data().total_cn.item() == 3
    assert original.prediction_data(original.mutation_ids).total_cn.item() == 3
    replayed = OriginalCatalog.from_dict(original.to_dict())
    np.testing.assert_array_equal(read_floor_bounds(replayed)[0].lower, clamped.lower)
    zero_fraction = mixed + [dict(cn_state_id="unused", cn_state_fraction=0., major_cn=1, minor_cn=0)]
    np.testing.assert_array_equal(estimate("zero_fraction_cn", zero_fraction)[1].lower, bounds.lower)
    return dict(single_and_multiple_states_checked=True, weighted_total_cn=3.6,
                fractional_cn_changes_floor=True, original_likelihood_cn_preserved=True,
                explicit_original_major_clamps_mixed_state_estimate=True)


def _native_workflow(directory):
    path = directory / "native.tsv"
    catalog = write_catalog(path, [[75]]*8 + [[5]]*8, 100, 4)
    expected, _ = read_floor_bounds(catalog)
    results = {}
    for method in ("marginal", "profile"):
        config = workflow_config(fit=FitConfig(device="cpu", max_clusters=2,
            cpu_threads=1, fit_multiplicity=method), stage_two_max_clusters=2)
        output = directory / method
        receipt = run_workflow(path, output, config)
        assert receipt["status"] == "complete" and receipt["stage_two"] == "accepted"
        first = load_accepted_fit(output / "stage_one_accepted.json")
        second = load_accepted_fit(output / "stage_two_accepted.json")
        np.testing.assert_array_equal(first.support.lower, expected.lower)
        np.testing.assert_array_equal(first.support.upper, catalog.major_cn)
        for fit in (first, second):
            required = expected.subset(fit.mutation_ids)
            np.testing.assert_array_equal(fit.support.lower, required.lower)
            assert np.all(fit.multiplicities >= required.lower)
            assert np.all(fit.multiplicities <= fit.support.upper)
        request = load_assignment_bundle(output / "assignment_bundle.json")
        assert request.native_lower.max() == 3
        assert np.all(request.bounds.lower == 1) and np.all(request.bounds.upper == 4)
        # Endpoint mismatches are rejected before constructing a native model.
        with patch("CliPP2.kernel.model.RegionalModel", side_effect=AssertionError("mismatched replay reached numerics")):
            expect_value_error(lambda: first.validate(catalog, expected_support=catalog.default_bounds()))
        results[method] = dict(stage_one_floors=True, stage_two_floors=True,
            native_calls_within_bounds=True, stage_three_original_support=True,
            replay_support_mismatch_rejected=True)
    return results


def _mixed_cn_workflow(directory):
    source = directory / "mixed_tumor"
    region = source / "r1"
    region.mkdir(parents=True)
    counts = [80]*8 + [5]*8
    (region / "snv.txt").write_text(
        "chromosome_index\tposition\talt_count\tref_count\n"
        + "".join(f"1\t{101+i}\t{alt}\t{100-alt}\n" for i, alt in enumerate(counts)),
        encoding="utf-8")
    (region / "purity.txt").write_text("1\n", encoding="utf-8")
    (region / "cna.txt").write_text(
        "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\t"
        "normal_cn\tcn_state_id\tcn_state_fraction\n"
        "1\t1\t1000\t3\t1\t4\t2\tgain\t0.6\n"
        "1\t1\t1000\t2\t1\t3\t2\tbase\t0.4\n", encoding="utf-8")
    config = workflow_config(fit=FitConfig(device="cpu", max_clusters=2,
        cpu_threads=1), stage_two_max_clusters=2)
    output = directory / "mixed_workflow"
    receipt = run_workflow(source, output, config)
    assert receipt["status"] == "complete" and receipt["stage_two"] == "accepted"
    catalog = OriginalCatalog.from_dict(json.loads((output / "original_catalog.json").read_text()))
    expected, audit = read_floor_bounds(catalog)
    np.testing.assert_array_equal(expected.lower[:, 0], [2]*8 + [1]*8)
    assert all(row["cn_state_count"] == 2 for row in audit)
    np.testing.assert_allclose([row["weighted_total_cn"] for row in audit], 3.6)
    first = load_accepted_fit(output / "stage_one_accepted.json")
    second = load_accepted_fit(output / "stage_two_accepted.json")
    np.testing.assert_array_equal(first.support.lower, expected.lower)
    request = load_assignment_bundle(output / "assignment_bundle.json")
    np.testing.assert_array_equal(request.bounds.lower, np.ones((16, 1)))
    np.testing.assert_array_equal(request.bounds.upper, np.full((16, 1), 3))
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("mixed CN replay refitted")):
        replay = run_workflow(source, directory / "mixed_replay", config,
            accepted_stage_one=first, accepted_stage_two=second)
    assert replay["status"] == "complete" and replay["counts"] == receipt["counts"]
    return dict(mixed_cna_states_used_by_workflow=True, saved_catalog_reconstructs_floors=True,
                stage_one_lower_two_instead_of_three=True, accepted_fits_replayed_without_refit=True)


def run_multiplicity_floor_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return dict(estimator=_estimator_cases(directory), mixed_cn=_mixed_cn_estimate(directory),
                cap_conflict=_segment_cap_conflict(directory),
                native=_native_workflow(directory), mixed_workflow=_mixed_cn_workflow(directory))
