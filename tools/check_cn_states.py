"""Single/multiple segment CN-state validation and estimator-input checks."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import tempfile

import pandas as pd

from CliPP2.io.cn_states import (CN_STATE_COLUMNS, SegmentCNStates,
                                parse_segment_cn_states, segment_states_from_catalog)
from CliPP2.io.data import OriginalCatalog
from CliPP2.io.region_files import read_region_directory
from CliPP2.io.tumor_txt import _read_text_table, load_original_catalog, load_tumor_txt, write_tumor_txt

from check_support import expect_value_error, write_catalog


def check_cn_states(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = (("r1", "gain", "a", .6, 3, 1),
               ("r1", "gain", "b", .4, 2, 1),
               ("r2", "gain", "a", 1., 6, 2))
    states = SegmentCNStates(records)
    assert abs(states.weighted_totals["r1", "gain"] - 3.6) < 1e-14
    assert states.weighted_totals["r2", "gain"] == 8.
    assert dict(states.state_counts) == {("r1", "gain"): 2, ("r2", "gain"): 1}
    assert states == SegmentCNStates(tuple(reversed(records)))
    assert hash(states) == hash(SegmentCNStates(tuple(reversed(records))))
    try:
        states.records = ()
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("CN states were mutable")
    catalog = write_catalog(directory / "catalog.tsv", [[20, 30]], 100,
                            [[3, 6]], minor=1, segments="gain")
    invalid_records = ((), (*records, records[0]),
        (("r1", "gain", "a", .9, 3, 1),),
        (("r1", "gain", "a", 0., 3, 1),),
        (("r1", "gain", "a", 1.0000001, 3, 1),),
        (("r1", "gain", "a", -.1, 3, 1),),
        (("r1", "gain", "a", float("nan"), 3, 1),),
        (("r1", "gain", "a", float("inf"), 3, 1),),
        (("r1", "gain", "a", 1., 2.5, 1),),
        (("r1", "gain", "a", 1., -1, 0),),
        (("r1", "gain", "a", 1., 2, 3),),
        (("r1", "gain", "a", 1., 2, -1),),
        (("r1", "gain", "a", 1., 2**63, 1),),
        (("r1", "gain", "a", 1., True, 0),),
        (("r1", "gain", ".", 1., 3, 1),),
        (("r1", "", "a", 1., 3, 1),),
        ((None, "gain", "a", 1., 3, 1),),
        (("r1", "gain", "a", ".", 3, 1),),
        ({"sample_id": "r1"},))
    for invalid in invalid_records:
        expect_value_error(lambda: SegmentCNStates(invalid))
    zero = SegmentCNStates((("r1", "gain", "a", 1., 3, 1),
                            ("r1", "gain", "b", 0., 2, 1)))
    assert zero.weighted_totals["r1", "gain"] == 4.
    assert zero.state_counts["r1", "gain"] == 2
    deletion = SegmentCNStates((("r1", "gain", "retained", .5, 4, 0),
                                ("r1", "gain", "deleted", .5, 0, 0)))
    assert deletion.weighted_totals["r1", "gain"] == 2.
    SegmentCNStates((("r1", "gain", "a", .6, 3, 1),
                     ("r1", "gain", "b", .400000001, 2, 1)))
    expect_value_error(lambda: SegmentCNStates((("r1", "gain", "a", .6, 3, 1),
                                               ("r1", "gain", "b", .4000001, 2, 1))))
    assert segment_states_from_catalog(catalog) is None
    encoded = json.dumps([dict(zip(CN_STATE_COLUMNS[2:], row[2:])) for row in records[:2]])
    assert parse_segment_cn_states(encoded, sample_id="r1", segment_id="gain").records == records[:2]
    frame = pd.DataFrame([dict(row) for row in catalog.source_rows])
    frame["segment_cn_states"] = [encoded if sample == "r1" else "." for sample in frame.sample_id]
    canonical = directory / "states_canonical.tsv"
    write_tumor_txt(canonical, frame, dict(catalog.metadata))
    with_states = load_original_catalog(canonical)
    derived = segment_states_from_catalog(with_states)
    assert derived.weighted_totals["r1", "gain"] == states.weighted_totals["r1", "gain"]
    assert derived.weighted_totals["r2", "gain"] == 7.
    assert derived.state_counts["r2", "gain"] == 1
    from CliPP2.io.multiplicity import read_floor_bounds
    _, audit = read_floor_bounds(with_states)
    assert {row["region_id"]: row["weighted_total_cn_source"] for row in audit} == {
        "r1": "supplied_segment_states", "r2": "single_state_input"}
    assert segment_states_from_catalog(OriginalCatalog.from_dict(with_states.to_dict())) == derived
    assert "segment_cn_states" in _read_text_table(canonical)[1].columns
    load_tumor_txt(canonical)
    compressed = directory / "states_canonical.tsv.gz"
    write_tumor_txt(compressed, frame, dict(catalog.metadata))
    assert segment_states_from_catalog(load_original_catalog(compressed)) == derived
    extra = frame.copy()
    extra["mutation_id"] = "m_extra"
    matching = pd.concat([frame, extra], ignore_index=True)
    write_tumor_txt(directory / "consistent.tsv", matching, dict(catalog.metadata))
    extra.loc[extra.sample_id == "r1", "segment_cn_states"] = "."
    expect_value_error(lambda: write_tumor_txt(directory / "inconsistent.tsv",
        pd.concat([frame, extra], ignore_index=True), dict(catalog.metadata)))
    original_pair = frame.copy()
    original_pair.loc[original_pair.sample_id == "r1", "allele_a_cn"] = "2"
    original_path = directory / "original_pair.tsv"
    write_tumor_txt(original_path, original_pair, dict(catalog.metadata))
    preserved = load_original_catalog(original_path)
    assert preserved.major_cn[0, 0] == 2 and preserved.total_cn[0, 0] == 3
    assert segment_states_from_catalog(OriginalCatalog.from_dict(preserved.to_dict())) == derived
    original_pair["mutation_id"] = "m_extra"
    expect_value_error(lambda: write_tumor_txt(directory / "inconsistent_original_pair.tsv",
        pd.concat([frame, original_pair], ignore_index=True), dict(catalog.metadata)))
    invalid_json = ("", "{}", "[]", "null", "[{}]", "not json",
        '[{"cn_state_id":"a","cn_state_id":"b","cn_state_fraction":1,"major_cn":3,"minor_cn":1}]')
    for value in invalid_json:
        expect_value_error(lambda: parse_segment_cn_states(value, sample_id="r1", segment_id="gain"))
    return dict(single_and_multiple_states=True, weighted_total_cn=True,
                canonical_identity_and_roundtrip=True, explicit_original_pair_preserved=True,
                invalid_states_rejected=True, canonical_state_validation=True,
                legacy_segment_fallback=True, regional_cna=_check_region_inputs(directory))


def _check_region_inputs(directory):
    tumor = directory / "mixed_cna"
    region = tumor / "r1"
    region.mkdir(parents=True)
    (region / "snv.txt").write_text("chromosome_index\tposition\talt_count\tref_count\n"
                                    "1\t100\t80\t20\n1\t150\t60\t40\n", encoding="utf-8")
    (region / "purity.txt").write_text("1\n", encoding="utf-8")
    prefix = "chromosome_index\tstart_position\tend_position\tmajor_cn\tminor_cn\ttotal_cn\tnormal_cn"
    header = prefix + "\tcn_state_id\tcn_state_fraction\n"
    first = "1\t1\t200\t3\t1\t4\t2\ta\t0.6\n"
    second = "1\t1\t200\t2\t1\t3\t2\tb\t0.4\n"
    cna = region / "cna.txt"
    cna.write_text(header + first + second, encoding="utf-8")
    catalog = load_original_catalog(tumor)
    states = segment_states_from_catalog(catalog)
    total = states.weighted_totals["r1", "1:1-200"]
    assert abs(total - 3.6) < 1e-14
    assert states.state_counts["r1", "1:1-200"] == 2
    assert (catalog.major_cn == 3).all() and (catalog.minor_cn == 1).all()
    # At VAF .8 and purity one, mixed CN gives floor(2.88)=2; using the
    # representative total CN instead gives floor(3.2)=3.
    import math
    assert math.floor(.8 * total) == 2 and math.floor(.8 * catalog.total_cn[0, 0]) == 3
    for preserve in (False, True):
        metadata, frame = read_region_directory(tumor, preserve_provenance=preserve)
        assert "segment_cn_states" in frame
        path = directory / f"cna_roundtrip_{preserve}.tsv"
        write_tumor_txt(path, frame, metadata)
        assert segment_states_from_catalog(load_original_catalog(path)) == states
    load_tumor_txt(tumor)
    cna.write_text(header + "1\t1\t200\t3\t1\t4\t2\ta\t1\n", encoding="utf-8")
    single = segment_states_from_catalog(load_original_catalog(tumor))
    assert single.state_counts["r1", "1:1-200"] == 1
    assert single.weighted_totals["r1", "1:1-200"] == 4.
    cna.write_text(header + "1\t1\t200\t4\t0\t4\t2\tretained\t0.5\n"
                   "1\t1\t200\t0\t0\t0\t2\tdeleted\t0.5\n", encoding="utf-8")
    deletion_catalog = load_original_catalog(tumor)
    assert segment_states_from_catalog(deletion_catalog).weighted_totals["r1", "1:1-200"] == 2.
    from CliPP2.io.multiplicity import read_floor_bounds
    assert (read_floor_bounds(deletion_catalog)[0].lower == 1).all()
    cna.write_text(prefix + "\n1\t1\t200\t3\t1\t4\t2\n", encoding="utf-8")
    assert segment_states_from_catalog(load_original_catalog(tumor)) is None
    invalid = (
        header + first + second.replace("0.4", "0.3"),
        header + first + second.replace("\tb\t", "\ta\t"),
        header + first + second.replace("\t1\t200\t", "\t150\t250\t"),
        prefix + "\tcn_state_id\n1\t1\t200\t3\t1\t4\t2\ta\n",
        prefix + "\tcn_state_fraction\n1\t1\t200\t3\t1\t4\t2\t1\n",
        header + first + second.replace("\t3\t2\tb\t", "\t3\t1\tb\t"),
        header.rstrip("\n") + "\tnormal_cn_supplied\n" + first.rstrip("\n") + "\t1\n"
            + second.rstrip("\n") + "\t0\n",
        header + first.replace("0.6", ".") + second,
        prefix + "\n1\t1\t200\t3\t1\t4\t2\n1\t1\t200\t2\t1\t3\t2\n",
    )
    for text in invalid:
        cna.write_text(text, encoding="utf-8")
        expect_value_error(lambda: load_original_catalog(tumor))
    return dict(single_and_multiple_cna_states=True, weighted_total_3_6=True,
                representative_pair_retained=True, canonical_roundtrip=True,
                invalid_segment_states_rejected=True)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="clipp2-cn-states-") as directory:
        print(json.dumps(check_cn_states(directory), sort_keys=True))
