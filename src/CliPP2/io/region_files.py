"""Adapt regional SNV/CNA/purity bundles to the canonical input contract."""
from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path

import pandas as pd

from .tumor_txt import (
    SCHEMA_COLUMNS, TumorTxtError, _finite_float, _identifier,
    _read_text_table, _safe_tumor_id, _validate_metadata, _provenance_flag,
)

_FILES = ("snv.txt", "cna.txt", "purity.txt")
_SNV_COLUMNS = ("chromosome_index", "position", "alt_count", "ref_count")
_CNA_COLUMNS = ("chromosome_index", "start_position", "end_position",
                "major_cn", "minor_cn", "total_cn")


class _MissingProfileValue(TumorTxtError):
    """A required regional value is absent, rather than outside its model domain."""


def _regions(path: Path) -> tuple[Path, ...]:
    """One complete root bundle or immediate regional bundles; never recurse."""
    if not path.is_dir():
        raise FileNotFoundError(f"Tumor input directory does not exist: {path}")
    direct = any((path / name).exists() for name in _FILES)
    children = tuple(sorted(
        (child for child in path.iterdir() if child.is_dir()
         and any((child / name).exists() for name in _FILES)),
        key=lambda child: child.name,
    ))
    if direct and children:
        raise TumorTxtError(f"{path}: ambiguous mixture of root and regional input files.")
    regions = (path,) if direct else children
    if not regions:
        raise TumorTxtError(f"{path}: expected snv.txt, cna.txt and purity.txt, or regional subfolders.")
    for region in regions:
        missing = [name for name in _FILES if not (region / name).is_file()]
        if missing:
            raise TumorTxtError(f"{region}: missing regional input files: {missing}.")
        _identifier(region.name, name="sample_id")
    return regions


def _integer(value: str, *, name: str, minimum: int = 0) -> int:
    if value == ".":
        raise _MissingProfileValue(f"{name} may not be missing.")
    if not value.isascii() or not value.isdecimal() or int(value) < minimum:
        raise TumorTxtError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _read_region(path: Path):
    tokens = (path / "purity.txt").read_text(encoding="utf-8").split()
    if not tokens or tokens == ["."]:
        raise _MissingProfileValue(f"{path}/purity.txt purity may not be missing.")
    if len(tokens) != 1:
        raise TumorTxtError(f"{path}/purity.txt must contain exactly one number.")
    purity = _finite_float(tokens[0], name=f"{path} purity")
    if not 0 < purity <= 1:
        raise TumorTxtError(f"{path} purity must lie in (0, 1].")
    _, table = _read_text_table(path / "snv.txt", columns=_SNV_COLUMNS, canonical=False)
    observations = {}
    observation_rows = {}
    for row in table.to_dict(orient="records"):
        chrom = _identifier(row["chromosome_index"], name=f"{path} chromosome_index")
        position = _integer(row["position"], name=f"{path} position", minimum=1)
        locus = (chrom, position)
        if locus in observations:
            raise TumorTxtError(f"{path}: duplicate SNV locus {locus!r}.")
        observed = _provenance_flag(row, ("count_observed",), "Read observation", default=True)
        counts = tuple("." if row[name] == "." and not observed else _integer(row[name], name=f"{path} {name}")
                       for name in ("alt_count", "ref_count"))
        observations[locus] = (*counts, _provenance_flag(row, ("input_excluded",),
                                                        "Input exclusion", default=False), observed)
        observation_rows[locus] = row
    _, table = _read_text_table(path / "cna.txt", columns=_CNA_COLUMNS,
                                canonical=False, allow_empty=True)
    state_columns = {"cn_state_id", "cn_state_fraction"}
    if state_columns.intersection(table.columns) and not state_columns.issubset(table.columns):
        raise TumorTxtError(f"{path}: CNA states require both cn_state_id and cn_state_fraction.")
    explicit_states = state_columns.issubset(table.columns)
    interval_groups = {}
    interval_rows = {}
    for row in table.to_dict(orient="records"):
        chrom = _identifier(row["chromosome_index"], name=f"{path} chromosome_index")
        start, end = (_integer(row[name], name=f"{path} {name}", minimum=1)
                      for name in ("start_position", "end_position"))
        major, minor, total, normal = (
            _integer(row.get(name, "2"), name=f"{path} {name}")
            for name in ("major_cn", "minor_cn", "total_cn", "normal_cn")
        )
        if start > end or major < minor or total != major + minor:
            raise TumorTxtError(
                f"{path}: invalid CNA interval {chrom}:{start}-{end}; require "
                "start <= end, major_cn >= minor_cn and total_cn = major_cn + minor_cn."
            )
        normal_supplied = _provenance_flag(row, ("normal_cn_supplied", "_normal_cn_supplied"),
                                           "Normal-CN", default="normal_cn" in row)
        if normal_supplied and "normal_cn" not in row:
            raise TumorTxtError(f"{path}: normal CN cannot be marked supplied without normal_cn.")
        genuine = _provenance_flag(row, ("mapping_genuine", "_mapping_genuine"), "Genuine-mapping")
        interval_groups.setdefault((chrom, start, end), []).append(
            (major, minor, normal, normal_supplied, genuine, row))
    intervals = {}
    from .cn_states import SegmentCNStates
    for (chrom, start, end), states in interval_groups.items():
        segment_id = f"{chrom}:{start}-{end}"
        if len(states) > 1 and not explicit_states:
            raise TumorTxtError(f"{path}: duplicate CNA interval {segment_id}; multiple states require "
                                "cn_state_id and cn_state_fraction.")
        if any(state[2:5] != states[0][2:5] for state in states[1:]):
            raise TumorTxtError(f"{path}: normal CN and provenance must agree across states of {segment_id}.")
        encoded = None
        if explicit_states:
            validated = SegmentCNStates(tuple(dict(sample_id=path.name, segment_id=segment_id,
                cn_state_id=state[5]["cn_state_id"], cn_state_fraction=state[5]["cn_state_fraction"],
                major_cn=state[0], minor_cn=state[1]) for state in states))
            encoded = json.dumps([dict(cn_state_id=state[2], cn_state_fraction=state[3],
                                       major_cn=state[4], minor_cn=state[5])
                                  for state in validated.records], separators=(",", ":"))
        # The native likelihood still uses one original integer CN pair. Keep
        # the state with the greatest (major, minor), preserving max major as
        # the biological multiplicity ceiling; all states inform the floor.
        major, minor, normal, normal_supplied, genuine, representative = max(
            states, key=lambda state: (state[0], state[1], str(state[5].get("cn_state_id", ""))))
        intervals.setdefault(chrom, []).append((start, end, major, minor, normal,
                                               normal_supplied, genuine, encoded))
        interval_rows[chrom, start, end] = representative
    indexed = {}
    for chrom, segments in intervals.items():
        segments.sort()
        if any(right[0] <= left[1] for left, right in zip(segments, segments[1:])):
            raise TumorTxtError(f"{path}: overlapping CNA intervals on chromosome {chrom!r}.")
        indexed[chrom] = ([segment[0] for segment in segments], segments)
    return purity, observations, indexed, observation_rows, interval_rows


def read_region_directory(path: str | Path, *, preserve_provenance=True) -> tuple[dict[str, str], pd.DataFrame]:
    """Join inclusive CN intervals, retaining genuine-mapping evidence by default."""
    path = Path(path).resolve()
    regions = {region.name: _read_region(region) for region in _regions(path)}
    loci = sorted({locus for value in regions.values() for locus in value[1]})
    has_exclusions = any(observation[2] for value in regions.values() for observation in value[1].values())
    rows = []
    for sample, value in regions.items():
        purity, observations, intervals, observation_rows, interval_rows = value
        for chrom, position in loci:
            starts, segments = intervals.get(chrom, ([], []))
            index = bisect_right(starts, position) - 1
            genuine = index >= 0 and position <= segments[index][1]
            if not genuine:
                raise TumorTxtError(f"{path}: input_missing CN segment for mutation {chrom}:{position} "
                                    f"in sample {sample!r}; run the input audit to inspect profile and segment mapping.")
            start, end, major, minor, normal, normal_supplied, genuine, cn_states = segments[index]
            alt, ref, excluded, observed = observations.get((chrom, position), (".", ".", False, False))
            row = dict(zip(SCHEMA_COLUMNS, (f"{chrom}:{position}", sample, alt, ref, int(observed), purity,
                         normal, f"{chrom}:{start}-{end}", "0", 1, major, minor)))
            row.update(_mapping_genuine=genuine, _normal_cn_supplied=normal_supplied)
            if cn_states is not None:
                row["segment_cn_states"] = cn_states
            if has_exclusions:
                row["input_excluded"] = int(excluded)
            if preserve_provenance:
                row.update(chromosome_index=chrom, position=position,
                           start_position=start, end_position=end)
                row.update(("snv_"+key, cell) for key, cell in observation_rows.get((chrom, position), {}).items())
                row.update(("cna_"+key, cell) for key, cell in interval_rows.get((chrom, start, end), {}).items())
            rows.append(row)
    metadata = _validate_metadata({"tumor_id": _safe_tumor_id(path.name),
                                  "mutation_key_convention": "chromosome_index:position"}, path=path)
    table = pd.DataFrame(rows).fillna(".")
    return metadata, table if preserve_provenance else table.loc[:, [*SCHEMA_COLUMNS,
        "_mapping_genuine", "_normal_cn_supplied", *(["input_excluded"] if has_exclusions else []),
        *(["segment_cn_states"] if "segment_cn_states" in table else [])]]


def audit_region_directory(path):
    """Inspect requested profiles and segment joins without filling missing CN."""
    path = Path(path).resolve()
    direct = any((path / name).exists() for name in _FILES)
    children = tuple(sorted(child for child in path.iterdir() if child.is_dir()
                            and any((child / name).exists() for name in _FILES)))
    issues = []
    if direct and children:
        return [dict(status="input_invalid", code="ambiguous_region_layout", source=str(path),
                     cause="tumor_region_mapping", message="Root and regional bundles are both present")]
    regions = (path,) if direct or not children else children
    parsed = {}
    for region in regions:
        missing = [name for name in _FILES if not (region / name).is_file()]
        for name in missing:
            issues.append(dict(status="input_missing", code="missing_profile", source=str(region / name),
                sample_id=region.name, field=name, cause="profile_unavailable_or_unmapped",
                message="Recover the source profile or correct the tumor/region path; no values were inferred"))
        if missing:
            continue
        try:
            parsed[region.name] = _read_region(region)
        except (OSError, ValueError) as error:
            missing = isinstance(error, _MissingProfileValue)
            issues.append(dict(status="input_missing" if missing else "input_invalid",
                               code="missing_profile_value" if missing else "invalid_profile", source=str(region),
                               sample_id=region.name, cause="profile_content", message=str(error)))
    loci = sorted({locus for value in parsed.values() for locus in value[1]})
    for sample, (_, _, intervals, _, _) in parsed.items():
        for chrom, position in loci:
            starts, segments = intervals.get(chrom, ([], []))
            index = bisect_right(starts, position) - 1
            if index >= 0 and position <= segments[index][1]:
                continue
            equivalent = [key for key in intervals if key.removeprefix("chr") == chrom.removeprefix("chr")]
            issues.append(dict(status="input_missing", code="missing_cn_segment", sample_id=sample,
                mutation_id=f"{chrom}:{position}", source=str(path), field="tumor_cn",
                cause="chromosome_naming" if equivalent else "segment_join",
                message="No supplied CN interval covers this mutation-region; no diploid CN was inferred"))
    return issues
