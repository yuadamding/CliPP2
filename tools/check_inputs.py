"""Input edge contracts used by the consolidated CPU qualification program."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import pickle
from unittest.mock import patch

import pandas as pd
import numpy as np

from CliPP2.io.assignment import load_accepted_fit, save_accepted_fit
from CliPP2.io.data import MultiplicityBounds, OriginalCatalog, validate_workflow_mapping
from CliPP2.io.region_files import read_region_directory
from CliPP2.io.tumor_txt import load_original_catalog, load_tumor_txt, write_tumor_txt
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction
from CliPP2.workflow import _fit_accepted, classify_groups

from check_workflow import expect_value_error, workflow_config


def _mixed_catalog(directory):
    rows = []
    for i, counts in enumerate(((50, 50), (49, 50), (50, 49), (12, 18), (13, 17), (12, 18))):
        for j, alt in enumerate(counts):
            mixed = (i, j) == (0, 1)
            for state in range(2 if mixed else 1):
                rows.append(dict(mutation_id=f"m{i}", sample_id=f"r{j+1}", alt_count=alt,
                    ref_count=100-alt, count_observed=1, purity=1, normal_cn=2,
                    segment_id=f"s{i}_{j}", cn_state_id=str(state),
                    cn_state_fraction=.5 if mixed else 1,
                    allele_a_cn=11+state if mixed else 1, allele_b_cn=1))
    path = directory / "excluded_mixed.tsv"
    write_tumor_txt(path, pd.DataFrame(rows))
    return path, load_original_catalog(path)


def _normal_provenance(directory, catalog):
    # Conversion must retain the assumption even when raw provenance is omitted.
    source = directory / "normal_flags"
    source.mkdir()
    (source / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n1\t100\t10\t10\n")
    header = "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn"
    (source / "cna.txt").write_text(header + "\n1\t1\t200\t1\t1\t2\n")
    (source / "purity.txt").write_text("0.6\n")
    metadata, frame = read_region_directory(source, preserve_provenance=False)
    path = directory / "compact_normal.tsv"
    write_tumor_txt(path, frame, metadata)
    assumed = load_original_catalog(path)
    assert not assumed.normal_cn_supplied.any()
    expect_value_error(lambda: validate_workflow_mapping(assumed))
    validate_workflow_mapping(assumed, normal_cn_policy="diploid")
    write_tumor_txt(directory / "different_assumption.tsv", frame.assign(normal_cn=1), metadata)
    different = load_original_catalog(directory / "different_assumption.tsv")
    expect_value_error(lambda: validate_workflow_mapping(different, normal_cn_policy="diploid"))
    expect_value_error(lambda: workflow_config(normal_cn_policy="haploid"))
    for marker in ("normal_cn_supplied", "_normal_cn_supplied"):
        for value in ("false", "True", ".", "2", " 0"):
            invalid = frame.assign(**{marker: value})
            expect_value_error(lambda: write_tumor_txt(directory / "bad_flag.tsv", invalid, metadata))
            invalid.to_csv(directory / "bad_flag.tsv", sep="\t", index=False)
            expect_value_error(lambda: load_original_catalog(directory / "bad_flag.tsv"))
    conflict = frame.assign(normal_cn_supplied="1", _normal_cn_supplied="0")
    expect_value_error(lambda: write_tumor_txt(directory / "conflicting_flags.tsv", conflict, metadata))
    # Repeated CN-state rows cannot disagree about whether one normal CN was supplied.
    states = pd.DataFrame([dict(row) for row in catalog.source_rows])
    states["normal_cn_supplied"] = "1"
    states.loc[states["cn_state_id"] == "1", "normal_cn_supplied"] = "0"
    expect_value_error(lambda: write_tumor_txt(directory / "state_flags.tsv", states))
    # A directory flag cannot claim supplied CN when its optional column is absent.
    (source / "cna.txt").write_text(header + "\tnormal_cn_supplied\n1\t1\t200\t1\t1\t2\t1\n")
    expect_value_error(lambda: load_original_catalog(source))
    return dict(compact_conversion_preserves_assumption=True, strict_flags=True,
                explicit_policy_matches_assumption=True)


def run_input_checks(directory):
    """Exercise admitted masked inputs through fitting and accepted-state replay."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, catalog = _mixed_catalog(directory)
    validate_workflow_mapping(catalog)
    original = catalog.default_bounds()
    lower, upper = original.for_data(catalog.prediction_data(catalog.mutation_ids))
    assert catalog.major_cn[0, 1] == 12 and original.upper[0, 1] == 4 and upper[0, 1] == 1
    sanitized = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, lower, upper)
    observation = prepare_prediction(catalog, catalog.mutation_ids, sanitized)
    assert not observation.observed[0, 1] and observation.support[0, 1] == 1
    eligible = load_original_catalog(path, max_major_cn=20)
    eligible.default_bounds().for_data(eligible)
    missing_path = directory / "eligible_mixed_without_reads.tsv"
    write_tumor_txt(missing_path, pd.DataFrame([dict(row) for row in eligible.source_rows]).assign(count_observed=0))
    eligible_missing = load_original_catalog(missing_path, max_major_cn=20)
    for bounds in (sanitized, replace(original, upper=np.where(catalog.cn_state_count > 1, 3, original.upper)),
                   replace(original, lower=np.where(catalog.cn_state_count > 1, 2, original.lower))):
        inert_lower, inert_upper = bounds.for_data(catalog)
        assert inert_lower[0, 1] == inert_upper[0, 1] == 1
        expect_value_error(lambda bounds=bounds: bounds.for_data(eligible))
        expect_value_error(lambda bounds=bounds: bounds.for_data(eligible_missing))

    config = workflow_config()
    accepted = _fit_accepted(catalog, catalog.mutation_ids, None, directory / "fit", "stage_one",
                             replace(config.fit, max_clusters=2), None)
    receipt = directory / "accepted.json"
    save_accepted_fit(accepted, receipt)
    restored = load_accepted_fit(receipt)
    assert restored.support.lower[0, 1] == restored.support.upper[0, 1] == 1
    assert np.isnan(restored.multiplicities[0, 1])
    assert catalog.cn_state_count[0, 1] == 2 and catalog.major_cn[0, 1] == 12
    np.testing.assert_array_equal(restored.support.for_data(catalog)[1], restored.support.upper)
    with NativeBackend(prepare_prediction(catalog, restored.mutation_ids, restored.support), cpu_threads=1) as backend:
        diagnostic = backend.component_diagnostics(restored.centers[restored.labels], restored.multiplicities)
    assert diagnostic["status"][0, 1] == 4
    assert all(np.isnan(values[0, 1]) for name, values in diagnostic.items()
               if name not in ("status", "has_second"))
    assert not diagnostic["has_second"][0, 1]
    assert classify_groups(catalog, restored, config)
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("replay refitted")):
        replayed = restored.validate(catalog, device="cpu", cpu_threads=1)
    assert replayed.compatibility_receipts and replayed.producer_receipt == accepted.producer_receipt
    for key, value in (("genome_build", "hg38"), ("mutation_key_convention", "other_variant_keys"),
                       ("reference_build", "declared_build"), ("region_mapping", "declared_alias")):
        metadata = dict(catalog.metadata, **{key: value})
        other = replace(catalog, metadata=tuple(metadata.items()))
        with patch("CliPP2.kernel.model.RegionalModel", side_effect=AssertionError("identity checked after model")):
            expect_value_error(lambda: restored.validate(other, device="cpu", cpu_threads=1))
    for clone in (copy.copy(catalog), copy.deepcopy(catalog), pickle.loads(pickle.dumps(catalog)),
                  OriginalCatalog.from_dict(catalog.to_dict())):
        assert clone.content_hash == catalog.content_hash
        for array in (clone.normal_cn, clone.normal_cn_supplied, clone.eligible):
            assert not array.flags.writeable
            expect_value_error(lambda: array.setflags(write=True))
    for clone in (copy.deepcopy(sanitized), pickle.loads(pickle.dumps(sanitized))):
        assert clone.content_hash == sanitized.content_hash and not clone.upper.flags.writeable
        expect_value_error(lambda: clone.upper.setflags(write=True))
    return dict(excluded_mixed_fit_screen_replay=True, eligible_mixed_override_rejected=True,
                biological_namespace_checked_before_model=True, immutable_catalog_and_support=True,
                normal_cn_provenance=_normal_provenance(directory, catalog),
                explicit_input_exclusion=check_input_exclusion(directory))


def check_input_exclusion(directory):
    """Explicit exclusions preserve observed diploid counts but add no evidence."""
    from CliPP2.assignment import assign_mutations
    from CliPP2.io.assignment import load_assignment_bundle, save_assignment_bundle
    from CliPP2.kernel.backend import NativeBackend
    from CliPP2.workflow import _biological_hash
    from check_workflow import bundle, fixture_reference

    directory = Path(directory)
    rows = [dict(mutation_id=f"e{i}", sample_id=f"r{j+1}", alt_count=80 if j == 2 else 25,
        ref_count=20 if j == 2 else 75, count_observed=1, purity=1, normal_cn=2,
        segment_id="diploid", cn_state_id="0", cn_state_fraction=1, allele_a_cn=1, allele_b_cn=1,
        input_excluded=int(i == 0 and j == 2)) for i in range(2) for j in range(3)]
    frame = pd.DataFrame(rows)
    path = directory / "explicit_exclusion.tsv"
    write_tumor_txt(path, frame)
    catalog = load_original_catalog(path)
    mask = np.array([[False, False, True], [False, False, False]])
    np.testing.assert_array_equal(catalog.input_excluded, mask)
    np.testing.assert_array_equal(catalog.eligible, ~mask)
    assert catalog.count_observed.all() and np.all(catalog.total_counts == 100)
    assert np.all(catalog.major_cn == 1) and np.all(catalog.minor_cn == 1)
    assert catalog.alt_counts[0, 2] == 80 and catalog.mean_total_cn[0, 2] == 2
    for data in (load_tumor_txt(path), catalog.fit_data(), catalog.prediction_data(catalog.mutation_ids)):
        assert data.input_excluded[0, 2] and not data.count_observed[0, 2]
        assert data.total_counts[0, 2] == 0
        assert any(record.reason == "INPUT_EXCLUDED" for record in data.cn_filter_report.records)
    for restored in (copy.deepcopy(catalog), pickle.loads(pickle.dumps(catalog)),
                     OriginalCatalog.from_dict(catalog.to_dict()), catalog.subset(catalog.mutation_ids)):
        np.testing.assert_array_equal(restored.input_excluded, mask)
        assert not restored.input_excluded.flags.writeable
        assert restored.content_hash == catalog.content_hash
    write_tumor_txt(directory / "excluded_roundtrip.tsv", pd.DataFrame([dict(row) for row in catalog.source_rows]),
                    dict(catalog.metadata))
    restored = load_original_catalog(directory / "excluded_roundtrip.tsv")
    np.testing.assert_array_equal(restored.input_excluded, catalog.input_excluded)
    np.testing.assert_array_equal(restored.alt_counts, catalog.alt_counts)
    assert _biological_hash(restored, restored.mutation_ids) == _biological_hash(catalog, catalog.mutation_ids)
    observation = prepare_prediction(catalog, catalog.mutation_ids)
    with NativeBackend(observation, cpu_threads=1) as backend:
        diagnostic = backend.component_diagnostics(np.ones(mask.shape))
    assert diagnostic["status"][0, 2] == 4 and np.isnan(diagnostic["log_emission"][0, 2])
    reference = fixture_reference((0, 1), catalog.region_ids, [[1., 1., 1.], [.5, .5, .5]])
    request = bundle(catalog, reference)
    assigned = assign_mutations(request, outdir=directory / "explicit_exclusion_assignment")
    row = assigned.multiplicities[2]
    assert row["count_observed"] and row["positive_depth"] and row["original_input_excluded"]
    assert row["excluded"] and not row["informative"] and row["alt_count"] == 80
    assert row["multiplicity_status"] == "excluded" and np.isnan(row["conditional_map"])
    save_assignment_bundle(request, directory / "explicit_exclusion_bundle.json")
    assert load_assignment_bundle(directory / "explicit_exclusion_bundle.json").content_hash == request.content_hash
    changed = frame.copy()
    changed.loc[mask.ravel(), ["alt_count", "ref_count"]] = [0, 100]
    write_tumor_txt(directory / "excluded_counts_changed.tsv", changed)
    changed_catalog = load_original_catalog(directory / "excluded_counts_changed.tsv")
    changed_result = assign_mutations(bundle(changed_catalog, reference))
    np.testing.assert_array_equal(assigned.assignments[0]["log_scores"], changed_result.assignments[0]["log_scores"])
    # An explicit reason stays identity-bearing even if CN independently excludes the same pair.
    cn_frame = frame.assign(allele_a_cn=11, segment_id="high_cn")
    write_tumor_txt(directory / "excluded_reason.tsv", cn_frame)
    with_reason = load_original_catalog(directory / "excluded_reason.tsv")
    write_tumor_txt(directory / "excluded_reason_absent.tsv", cn_frame.assign(input_excluded=0),
                    dict(with_reason.metadata))
    without_reason = load_original_catalog(directory / "excluded_reason_absent.tsv")
    np.testing.assert_array_equal(with_reason.eligible, without_reason.eligible)
    assert _biological_hash(with_reason, with_reason.mutation_ids) != _biological_hash(without_reason, without_reason.mutation_ids)
    for flag in ("false", "True", "2", ".", " 0"):
        expect_value_error(lambda flag=flag: write_tumor_txt(directory / "invalid_excluded.tsv", frame.assign(input_excluded=flag)))
    repeated = pd.concat([frame.assign(cn_state_fraction=.5), frame.assign(cn_state_fraction=.5, cn_state_id="1", input_excluded=0)])
    expect_value_error(lambda: write_tumor_txt(directory / "repeated_excluded.tsv", repeated))
    source = directory / "region_exclusion"
    source.mkdir()
    (source / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\tinput_excluded\n1\t100\t8\t2\t1\n1\t200\t3\t7\t0\n")
    (source / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\tnormal_cn\n1\t1\t300\t1\t1\t2\t2\n")
    (source / "purity.txt").write_text("1\n")
    direct = load_original_catalog(source)
    assert direct.input_excluded[:, 0].tolist() == [True, False] and direct.count_observed.all()
    assert load_tumor_txt(source).mutation_ids == ("1:200",)
    return dict(raw_observed_counts_and_cn_preserved=True, round_trip=True, assignment_zero_evidence=True,
                reason_bound_to_biological_identity=True, strict_flags=True, directory_masks=True)
