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
from CliPP2.io.tumor_txt import SCHEMA_COLUMNS, audit_input, load_original_catalog, load_tumor_txt, write_tumor_txt
from CliPP2.kernel.backend import NativeBackend
from CliPP2.kernel.observations import prepare_prediction
from CliPP2.workflow import _fit_accepted, classify_groups

from check_support import expect_value_error, workflow_config


def _excluded_catalog(directory):
    rows = []
    for i, counts in enumerate(((50, 50), (49, 50), (50, 49), (12, 18), (13, 17), (12, 18))):
        for j, alt in enumerate(counts):
            rows.append(dict(mutation_id=f"m{i}", sample_id=f"r{j+1}", alt_count=alt,
                ref_count=100-alt, count_observed=1, purity=1, normal_cn=2,
                segment_id=f"s{i}_{j}", cn_state_id="0", cn_state_fraction=1,
                allele_a_cn=12 if (i, j) == (0, 1) else 1, allele_b_cn=1))
    path = directory / "excluded_single_state.tsv"
    write_tumor_txt(path, pd.DataFrame(rows))
    return path, load_original_catalog(path, max_major_cn=10)


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
    expect_value_error(lambda: load_tumor_txt(source))
    load_tumor_txt(source, normal_cn_policy="diploid")
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
    # A directory flag cannot claim supplied CN when its optional column is absent.
    (source / "cna.txt").write_text(header + "\tnormal_cn_supplied\n1\t1\t200\t1\t1\t2\t1\n")
    expect_value_error(lambda: load_original_catalog(source))
    # A declared nongenuine mapping cannot be promoted by the directory adapter.
    (source / "cna.txt").write_text(header + "\tnormal_cn\tmapping_genuine\n1\t1\t200\t1\t1\t2\t2\t0\n")
    assert not load_original_catalog(source).mapping_genuine.any()
    expect_value_error(lambda: load_tumor_txt(source))
    return dict(compact_conversion_preserves_assumption=True, strict_flags=True,
                explicit_policy_matches_assumption=True)


def check_normal_cn_fit_metadata(directory):
    """Assumption provenance differs; identical numeric observations do not."""
    from CliPP2.api import process_tumor
    from CliPP2.fit_execution import fit_prepared
    from CliPP2.io.data import json_value
    from check_support import write_catalog

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    supplied_path = directory / "supplied.tsv"
    supplied = write_catalog(supplied_path, [[30, 32], [29, 31], [12, 15], [11, 14]], 100,
        2, minor=1, purity=[.65, .8], metadata={"tumor_id": "normal_policy"})
    assumed_path = directory / "assumed.tsv"
    frame = pd.DataFrame([dict(row) for row in supplied.source_rows])
    write_tumor_txt(assumed_path, frame.assign(normal_cn_supplied=0),
                    dict(supplied.metadata))
    assumed = load_original_catalog(assumed_path)
    assert supplied.normal_cn_supplied.all() and not assumed.normal_cn_supplied.any()
    config = replace(workflow_config().fit, max_clusters=2)
    with patch("CliPP2.fit_execution.fit_prepared", side_effect=AssertionError("Missing policy reached fitting")):
        expect_value_error(lambda: process_tumor(assumed_path, directory / "unauthorized", config))
        expect_value_error(lambda: _fit_accepted(assumed, assumed.mutation_ids, None,
            directory / "unauthorized_workflow", "stage_one", config, None))
    executions, tables = [], []
    for name, path, catalog, policy in (("supplied", supplied_path, supplied, None),
            ("assumed", assumed_path, assumed, "diploid"),
            ("supplied_explicit", supplied_path, supplied, "diploid")):
        def capture(*args, **kwargs):
            execution = fit_prepared(*args, **kwargs)
            executions.append(execution)
            return execution
        ordinary = directory / name
        with patch("CliPP2.fit_execution.fit_prepared", side_effect=capture):
            summary = process_tumor(path, ordinary, config, normal_cn_policy=policy)
            accepted = _fit_accepted(catalog, catalog.mutation_ids, None,
                directory / (name + "_workflow"), "stage_one", config, None, normal_cn_policy=policy)
        assert summary["normal_cn_policy"] == policy
        assert all(execution.producer_receipt["normal_cn_policy"] == policy for execution in executions[-2:])
        assert json_value(accepted.producer_receipt) == json_value(executions[-1].producer_receipt)
        assert accepted.validate(catalog, device="cpu", cpu_threads=1).producer_receipt == accepted.producer_receipt
        tables.append({filename: (ordinary / filename).read_bytes() for filename in summary["publication"]["files"]})
        assert tables[-1] == {filename: (directory / (name + "_workflow") / filename).read_bytes()
                             for filename in summary["publication"]["files"]}
    baseline = executions[0]
    for execution in executions[1:]:
        assert execution.summary["model_identity"] == baseline.summary["model_identity"]
        assert execution.fit.score == baseline.fit.score
        for field in ("centers", "weights"):
            np.testing.assert_array_equal(getattr(execution.fit, field), getattr(baseline.fit, field))
        for field in ("public_labels", "multiplicities"):
            np.testing.assert_array_equal(getattr(execution, field), getattr(baseline, field))
    assert tables[0] == tables[1] == tables[2]
    return dict(supplied_none=True, assumed_diploid=True, supplied_explicit_diploid=True,
                unauthorized_missing_normal_rejected=True, common_and_workflow_receipts_agree=True,
                numerical_identity_and_results_unchanged=True, published_tables_identical=True)


def run_input_checks(directory):
    """Exercise admitted masked inputs through fitting and accepted-state replay."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, catalog = _excluded_catalog(directory)
    validate_workflow_mapping(catalog)
    original = catalog.default_bounds()
    lower, upper = original.for_data(catalog.prediction_data(catalog.mutation_ids))
    assert catalog.major_cn[0, 1] == 12 and original.upper[0, 1] == 12 and upper[0, 1] == 1
    sanitized = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, lower, upper)
    observation = prepare_prediction(catalog, catalog.mutation_ids, sanitized)
    assert not observation.observed[0, 1] and observation.support[0, 1] == 1
    eligible = load_original_catalog(path, max_major_cn=20)
    eligible.default_bounds().for_data(eligible)
    restricted = replace(original, upper=np.minimum(original.upper, 3))
    assert restricted.for_data(eligible)[1][0, 1] == 3
    assert restricted.for_data(catalog)[1][0, 1] == 1
    expect_value_error(lambda: replace(original, upper=original.upper+1).for_data(catalog))

    config = workflow_config()
    config = replace(config, fit=replace(config.fit, max_major_cn=10))
    accepted = _fit_accepted(catalog, catalog.mutation_ids, None, directory / "fit", "stage_one",
                             replace(config.fit, max_clusters=2), None)
    receipt = directory / "accepted.json"
    save_accepted_fit(accepted, receipt)
    restored = load_accepted_fit(receipt)
    assert restored.support.lower[0, 1] == restored.support.upper[0, 1] == 1
    assert np.isnan(restored.multiplicities[0, 1])
    assert catalog.major_cn[0, 1] == 12 and catalog.total_cn[0, 1] == 13
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
    return dict(excluded_single_state_fit_screen_replay=True, single_state_contract=_single_state_contract(directory, catalog),
                unrestricted_cn=check_unrestricted_cn(directory / "unrestricted_cn"),
                biological_namespace_checked_before_model=True, immutable_catalog_and_support=True,
                normal_cn_provenance=_normal_provenance(directory, catalog),
                normal_cn_fit_metadata=check_normal_cn_fit_metadata(directory / "normal_fit_metadata"),
                explicit_input_exclusion=check_input_exclusion(directory))


def check_unrestricted_cn(directory):
    """Population expansion is explicit and cannot reuse a restricted saved input."""
    from CliPP2.config import FitConfig, WorkflowConfig
    from CliPP2.io.data import content_hash
    from CliPP2.kernel.backend import NativeExecutionError
    from scipy.special import logsumexp
    from scipy.stats import binom

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assert FitConfig().max_major_cn is None
    assert "stage_two_near_clonal_distance_max" not in WorkflowConfig.__dataclass_fields__
    for invalid in (0, -1, 1.5, True, "10"):
        expect_value_error(lambda invalid=invalid: FitConfig(max_major_cn=invalid))
    path, restricted = _excluded_catalog(directory)
    expanded = load_original_catalog(path)
    assert expanded.eligible.all() and not restricted.eligible[0, 1]
    assert expanded.max_major_cn is None and expanded.to_dict()["eligibility_policy"] == "unrestricted_valid_cn"
    assert load_tumor_txt(path).major_cn[0, 1] == 12
    expect_value_error(lambda: expanded.compatible_source_identities(10))
    # V4 remains a real identity-preserving reader, never an implicit upgrade.
    old = replace(restricted, _schema_version=4)
    payload = old.to_dict()
    for value in (OriginalCatalog.from_dict(payload), copy.deepcopy(old), pickle.loads(pickle.dumps(old))):
        assert value.to_dict() == payload and value.content_hash == old.content_hash
    assert content_hash(payload) in restricted.compatible_source_identities(10)
    subset = expanded.subset(expanded.mutation_ids[1:])
    old_subset = replace(subset, max_major_cn=10, _schema_version=4)
    assert content_hash(old_subset.to_dict()) in subset.compatible_source_identities(10)
    tampered = old_subset.to_dict()
    tampered["total_counts"][0][0] += 1
    expect_value_error(lambda: OriginalCatalog.from_dict(tampered))
    # Historical source placeholders remain evidence, but new directory joins
    # cannot invent diploid values even for unobserved mutation-region cells.
    source_rows = tuple(tuple(dict(row, mapping_genuine="0").items()) for row in old.source_rows)
    legacy_mapping = replace(old, mapping_genuine=np.zeros(old.major_cn.shape, bool), source_rows=source_rows,
        provenance=tuple(tuple((k, v) for k, v in row if k not in SCHEMA_COLUMNS) for row in source_rows))
    assert OriginalCatalog.from_dict(legacy_mapping.to_dict()).to_dict() == legacy_mapping.to_dict()

    rows = [dict(mutation_id=f"m{i}", sample_id="r", alt_count=7, ref_count=33,
        count_observed=1, purity=.7, normal_cn=i, segment_id=f"s{i}", cn_state_id="0",
        cn_state_fraction=1, allele_a_cn=major, allele_b_cn=1)
        for i, major in enumerate((11, 64, 257))]
    high_path = directory / "high_cn.tsv"
    write_tumor_txt(high_path, pd.DataFrame(rows))
    high = load_original_catalog(high_path)
    assert high.eligible.all()
    for bounds in (high.default_bounds(), MultiplicityBounds(high.mutation_ids, high.region_ids,
                   np.full((3, 1), 2), high.major_cn-1)):
        prepared = prepare_prediction(high, high.mutation_ids, bounds)
        phi = np.full((3, 1), .63)
        with NativeBackend(prepared, cpu_threads=1) as backend:
            diagnostic = backend.component_diagnostics(phi)
            grid = backend.regional_grid(0, np.arange(3), [.63])[:, 0]
        for i in range(3):
            candidates = np.arange(bounds.lower[i, 0], bounds.upper[i, 0]+1)
            probability = (prepared.scaling[i, 0]*candidates)*phi[i, 0]
            expected = logsumexp(binom.logpmf(7, 40, probability))-np.log(len(candidates))
            np.testing.assert_allclose([diagnostic["log_emission"][i, 0], grid[i]], expected, rtol=1e-12, atol=1e-12)
    # This hits checked scratch-size arithmetic, not an actual enormous allocation.
    huge = pd.DataFrame([dict(rows[0], allele_a_cn=2**60)])
    huge_path = directory / "resource_bound.tsv"
    write_tumor_txt(huge_path, huge)
    huge_catalog = load_original_catalog(huge_path)
    prepared = prepare_prediction(huge_catalog, huge_catalog.mutation_ids)
    assert prepared.support[0, 0] == 2**60
    with NativeBackend(prepared, cpu_threads=1) as backend:
        try:
            backend.regional_grid(0, np.array([0]), [.5])
        except NativeExecutionError as error:
            assert "overflow" in str(error).lower()
        else:
            raise AssertionError("Unrepresentable scratch storage was silently accepted")

    ready = audit_input(high_path)
    assert ready["status"] == "ready" and ready["eligible_pairs"] == 3
    missing = pd.DataFrame(rows).assign(purity=".")
    missing.to_csv(directory / "missing.tsv", sep="\t", index=False)
    audit = audit_input(directory / "missing.tsv")
    assert audit["status"] == "input_missing" and len(audit["issues"]) == 3
    assert all("purity" in issue["fields"] for issue in audit["issues"])
    profiles = directory / "profiles"
    profiles.mkdir()
    (profiles / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n1\t100\t3\t7\n")
    audit = audit_input(profiles)
    assert audit["status"] == "input_missing"
    assert {issue["field"] for issue in audit["issues"]} == {"cna.txt", "purity.txt"}
    (profiles / "purity.txt").write_text(".7\n")
    (profiles / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\tnormal_cn\nchr1\t1\t200\t1\t1\t2\t2\n")
    audit = audit_input(profiles)
    assert audit["status"] == "input_missing" and audit["issues"][0]["cause"] == "chromosome_naming"
    assert not any("major_cn" in issue for issue in audit["issues"])
    expect_value_error(lambda: load_original_catalog(profiles))
    expect_value_error(lambda: read_region_directory(profiles))
    missing_reads = directory / "missing_reads_and_cn"
    for sample, position, start, end in (("r1", 100, 1, 150), ("r2", 200, 151, 250)):
        region = missing_reads / sample
        region.mkdir(parents=True)
        (region / "snv.txt").write_text(f"chromosome_index\tposition\talt_count\tref_count\n1\t{position}\t3\t7\n")
        (region / "purity.txt").write_text(".7\n")
        (region / "cna.txt").write_text("chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\tnormal_cn\n"
            f"1\t{start}\t{end}\t1\t1\t2\t2\n")
    audit = audit_input(missing_reads)
    assert audit["status"] == "input_missing" and len(audit["issues"]) == 2
    assert {(row["mutation_id"], row["sample_id"]) for row in audit["issues"]} == {("1:200", "r1"), ("1:100", "r2")}
    expect_value_error(lambda: load_original_catalog(missing_reads))
    return dict(default_unrestricted=True, explicit_cutoff_preserved=True, historical_catalog_identity_preserved=True,
        changed_population_rejected=True, higher_cn_oracles=[11, 64, 257], native_resource_overflow_explicit=True,
        input_audit_missing_profiles_and_mapping=True)


def _single_state_contract(directory, catalog):
    from CliPP2.io.data import TumorData
    from check_support import bundle, fixture_reference
    frame = pd.DataFrame([dict(row) for row in catalog.source_rows])
    for name, invalid in (("two_states", pd.concat([frame, frame.assign(cn_state_id="1")])),
                          ("mixed_fraction", frame.assign(cn_state_fraction=.5)),
                          ("fractional_cn", frame.assign(allele_a_cn=1.5))):
        path = directory / (name + ".tsv")
        invalid.to_csv(path, sep="\t", index=False)
        expect_value_error(lambda path=path: load_original_catalog(path))
        expect_value_error(lambda path=path: load_tumor_txt(path))
        expect_value_error(lambda path=path, invalid=invalid: write_tumor_txt(path, invalid))
    data = catalog.fit_data()
    for field, values in (("major_cn", data.major_cn+.5), ("minor_cn", data.major_cn+1),
                          ("normal_cn", data.normal_cn+.5), ("mapping_genuine", np.zeros_like(data.count_observed)),
                          ("normal_cn_supplied", np.zeros_like(data.count_observed))):
        expect_value_error(lambda field=field, values=values: replace(data, **{field: values}))
    assumed = replace(data, normal_cn_supplied=np.zeros_like(data.count_observed), normal_cn_policy="diploid")
    assert assumed.normal_cn_policy == "diploid"
    for normal in (0, 1, 2):
        direct = TumorData("programmatic", ("m",), ("r",), np.array([[0.]]), np.array([[3.]]),
            np.array([[.8]]), np.array([[8.]]), np.array([[1.]]), np.array([[normal]]))
        bounds = MultiplicityBounds(direct.mutation_ids, direct.region_ids, [[1]], [[8]])
        assert bounds.for_data(direct)[1][0, 0] == 8
        expect_value_error(lambda: replace(bounds, upper=[[9]]).for_data(direct))
    for field, values in (("major_cn", catalog.major_cn+.5), ("minor_cn", catalog.major_cn+1),
                          ("normal_cn", catalog.normal_cn+.5), ("eligible", ~catalog.eligible),
                          ("purity", catalog.purity*.5)):
        expect_value_error(lambda field=field, values=values: replace(catalog, **{field: values}))
    altered = tuple(tuple(dict(row, cn_state_fraction="0.5").items()) for row in catalog.source_rows)
    expect_value_error(lambda: replace(catalog, source_rows=altered))
    expect_value_error(lambda: replace(catalog, source_rows=catalog.source_rows+catalog.source_rows[:1],
                                       provenance=catalog.provenance+catalog.provenance[:1]))
    stale = catalog.to_dict()
    stale["schema"] = "clipp2.original-catalog.v3"
    expect_value_error(lambda: OriginalCatalog.from_dict(stale))
    reference = fixture_reference((0,), catalog.region_ids, [[1.]*len(catalog.region_ids)])
    request = bundle(catalog, reference)
    expect_value_error(lambda: replace(request, native_upper=catalog.major_cn+1))
    # Parsed evidence provenance reaches generic loading, fitting, prediction, and assignment.
    for field in ("mapping_genuine", "normal_cn_supplied"):
        path = directory / (field + ".tsv")
        write_tumor_txt(path, frame.assign(**{field: 0}))
        retained = load_original_catalog(path, max_major_cn=catalog.max_major_cn)
        expect_value_error(lambda: load_tumor_txt(path))
        expect_value_error(lambda: retained.fit_data())
        expect_value_error(lambda: retained.prediction_data(retained.mutation_ids))
        expect_value_error(lambda: replace(request, catalog=retained))
        if field == "normal_cn_supplied":
            assert load_tumor_txt(path, normal_cn_policy="diploid").num_mutations == data.num_mutations
            retained.fit_data(normal_cn_policy="diploid")
            replace(request, catalog=retained, normal_cn_policy="diploid")
    return dict(parsed_and_programmatic_single_state=True, original_bounds_enforced=True,
                normal_cn_0_1_2=True, high_support_preserved=True, explicit_provenance_policy=True,
                old_catalog_schema_rejected=True)


def check_input_exclusion(directory):
    """Explicit exclusions preserve observed diploid counts but add no evidence."""
    from CliPP2.assignment import assign_mutations
    from CliPP2.io.assignment import load_assignment_bundle, save_assignment_bundle
    from CliPP2.kernel.backend import NativeBackend
    from CliPP2.workflow import _biological_hash
    from check_support import bundle, fixture_reference

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
    assert catalog.alt_counts[0, 2] == 80 and catalog.total_cn[0, 2] == 2
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
    with_reason = load_original_catalog(directory / "excluded_reason.tsv", max_major_cn=10)
    write_tumor_txt(directory / "excluded_reason_absent.tsv", cn_frame.assign(input_excluded=0),
                    dict(with_reason.metadata))
    without_reason = load_original_catalog(directory / "excluded_reason_absent.tsv", max_major_cn=10)
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
