"""Automatic regional median-depth imputation for mutations absent in some regions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from CliPP2 import FitConfig, run_workflow
from CliPP2.io.assignment import load_accepted_fit, load_assignment_bundle
from CliPP2.io.data import OriginalCatalog
from CliPP2.io.multiplicity import read_floor_bounds
from CliPP2.io.tumor_txt import audit_input, load_original_catalog, load_tumor_txt, write_tumor_txt

from check_support import expect_value_error, workflow_config


def _row(mutation, region="r1", *, alt=10, ref=90, observed=True, excluded=False, major=3):
    return dict(mutation_id=mutation, sample_id=region, alt_count=alt, ref_count=ref,
        count_observed=int(observed), input_excluded=int(excluded), purity=1., normal_cn=2,
        segment_id=f"cn{major}", cn_state_id="0", cn_state_fraction=1.,
        allele_a_cn=major, allele_b_cn=1)


def _canonical_rules(directory):
    rows = [_row("a_observed", alt=30, ref=70), _row("b_observed_zero_alt", alt=0, ref=101),
        _row("c_missing_stale", alt=7, ref=999, observed=False),
        _row("d_zero_depth", alt=0, ref=0),
        _row("e_missing_excluded", alt=20, ref=999, observed=False, excluded=True),
        _row("f_excluded", alt=10, ref=990, excluded=True),
        _row("g_excluded", alt=10, ref=1990, excluded=True),
        _row("h_cutoff", alt=10, ref=4990, major=20)]
    path = write_tumor_txt(directory / "missing_stale.tsv", pd.DataFrame(rows))
    catalog = load_original_catalog(path, max_major_cn=10)
    index = catalog.mutation_ids.index("c_missing_stale")
    assert catalog.alt_counts[index, 0] == 0 and catalog.total_counts[index, 0] == 101
    assert catalog.count_observed[index, 0]
    assert catalog.total_counts[1, 0] == 101 and catalog.alt_counts[1, 0] == 0
    assert catalog.total_counts[3, 0] == 0 and catalog.count_observed[3, 0]
    assert catalog.alt_counts[4, 0] == 20 and catalog.total_counts[4, 0] == 1019
    assert not catalog.count_observed[4, 0] and not catalog.eligible[4, 0]
    raw = dict(catalog.source_rows[index])
    assert raw["alt_count"] == "7" and raw["ref_count"] == "999" and raw["count_observed"] == "0"
    assert catalog.read_count_report["imputed_pairs"] == 1
    assert catalog.read_count_report["regional_medians"] == {"r1": 101}
    # A subset containing no median donors must retain the full-input depth.
    subset = catalog.subset(("c_missing_stale",))
    assert subset.total_counts.item() == 101 and subset.alt_counts.item() == 0
    for frozen in (copy.deepcopy(subset), OriginalCatalog.from_dict(subset.to_dict())):
        assert frozen.content_hash == subset.content_hash and frozen.total_counts.item() == 101
    expected, _ = read_floor_bounds(catalog)
    assert expected.lower[index, 0] == 1
    untouched = load_original_catalog(path, max_major_cn=10, missing_counts="unobserved")
    assert untouched.alt_counts[index, 0] == 7 and untouched.total_counts[index, 0] == 1006
    assert not untouched.count_observed[index, 0] and not untouched.read_count_imputations
    numerical = load_tumor_txt(path, max_major_cn=10)
    position = numerical.mutation_ids.index("c_missing_stale")
    assert numerical.alt_counts[position, 0] == 0 and numerical.total_counts[position, 0] == 101
    assert audit_input(path, max_major_cn=10)["read_counts"] == catalog.read_count_report
    # Raw source export reruns the same default rule, retaining original rows.
    roundtrip = write_tumor_txt(directory / "missing_stale_roundtrip.tsv",
        pd.DataFrame([dict(row) for row in catalog.source_rows]), dict(catalog.metadata))
    restored = load_original_catalog(roundtrip, max_major_cn=10)
    assert restored.source_rows == catalog.source_rows and dict(restored.metadata) == dict(catalog.metadata)
    assert restored.read_count_imputations == catalog.read_count_imputations
    np.testing.assert_array_equal(restored.alt_counts, catalog.alt_counts)
    np.testing.assert_array_equal(restored.total_counts, catalog.total_counts)
    missing = write_tumor_txt(directory / "no_local_donor.tsv", pd.DataFrame([
        _row("m", "r1"), _row("m", "r2", alt=7, ref=999, observed=False)]))
    expect_value_error(lambda: load_original_catalog(missing))
    expect_value_error(lambda: load_tumor_txt(missing))
    failed = audit_input(missing)
    assert failed["status"] == "input_invalid" and "median" in failed["issues"][-1]["message"]
    return dict(stale_missing_counts_replaced=True, integer_median_half_up=True,
        excluded_and_zero_depth_not_donors=True, observed_zero_alt_preserved=True,
        frozen_subset_depth_preserved=True, raw_source_and_roundtrip_preserved=True,
        explicit_optout=True, no_cross_region_depth_fallback=True)


def _regional_union(directory):
    source = directory / "regional_union"
    regional = {"r1": ((100, 10, 40), (200, 20, 31), (500, 0, 0), (700, 0, 99)),
                "r2": ((300, 80, 120), (400, 40, 161), (500, 0, 0), (700, 0, 303))}
    for name, observations in regional.items():
        region = source / name
        region.mkdir(parents=True)
        (region / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n"
            + "".join(f"1\t{position}\t{alt}\t{ref}\n" for position, alt, ref in observations), encoding="utf-8")
        (region / "purity.txt").write_text("1\n", encoding="utf-8")
        header = "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\tnormal_cn"
        rows = ("\tcn_state_id\tcn_state_fraction\n1\t1\t1000\t3\t1\t4\t2\tgain\t0.6\n"
                "1\t1\t1000\t2\t1\t3\t2\tbase\t0.4\n" if name == "r1"
                else "\n1\t1\t1000\t3\t1\t4\t2\n")
        (region / "cna.txt").write_text(header + rows, encoding="utf-8")
    catalog = load_original_catalog(source)
    assert catalog.mutation_ids == ("1:100", "1:200", "1:300", "1:400", "1:500", "1:700")
    np.testing.assert_array_equal(catalog.alt_counts, [[10, 0], [20, 0], [0, 80], [0, 40], [0, 0], [0, 0]])
    np.testing.assert_array_equal(catalog.total_counts, [[50, 201], [51, 201], [51, 200], [51, 201], [0, 0], [99, 303]])
    assert catalog.count_observed.all()
    assert catalog.read_count_report["imputed_pairs"] == 4
    assert catalog.read_count_report["regional_medians"] == {"r1": 51, "r2": 201}
    source_rows = {(row["mutation_id"], row["sample_id"]): row for row in map(dict, catalog.source_rows)}
    assert source_rows["1:100", "r2"]["alt_count"] == "."
    assert source_rows["1:100", "r2"]["ref_count"] == "."
    assert source_rows["1:100", "r2"]["count_observed"] == "0"
    for numerical in (load_tumor_txt(source), catalog.fit_data(), catalog.prediction_data(catalog.mutation_ids)):
        np.testing.assert_array_equal(numerical.alt_counts, catalog.alt_counts)
        np.testing.assert_array_equal(numerical.total_counts, catalog.total_counts)
        assert numerical.count_observed.all()
    assert audit_input(source)["read_counts"] == catalog.read_count_report
    config = workflow_config(fit=FitConfig(device="cpu", cpu_threads=1, max_clusters=2), stage_two_max_clusters=2)
    output = directory / "regional_workflow"
    receipt = run_workflow(source, output, config)
    assert receipt["status"] == "complete"
    saved = OriginalCatalog.from_dict(json.loads((output / "original_catalog.json").read_text()))
    assert saved.content_hash == catalog.content_hash
    first = load_accepted_fit(output / "stage_one_accepted.json")
    second_path = output / "stage_two_accepted.json"
    second = load_accepted_fit(second_path) if second_path.exists() else None
    bundle = load_assignment_bundle(output / "assignment_bundle.json")
    assert bundle.catalog.content_hash == catalog.content_hash
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("missing-read replay refitted")):
        replay = run_workflow(source, directory / "regional_replay", config,
            accepted_stage_one=first, accepted_stage_two=second)
    assert replay["status"] == "complete" and replay["counts"] == receipt["counts"]
    return dict(union_of_region_snvs=True, regional_medians={"r1": 51, "r2": 201},
        mixed_cna_preserved=True, all_input_routes_agree=True, native_workflow_complete=True,
        saved_catalog_and_assignment_counts_preserved=True, accepted_fit_replay_without_refit=True)


def run_missing_read_checks(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return dict(canonical=_canonical_rules(directory), regional=_regional_union(directory))
