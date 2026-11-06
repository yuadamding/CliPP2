"""Adapt regional SNV/CNA/purity bundles to the canonical input contract."""
from __future__ import annotations

from bisect import bisect_right
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
    if not value.isascii() or not value.isdecimal() or int(value) < minimum:
        raise TumorTxtError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _read_region(path: Path, *, preserve_provenance=False):
    tokens = (path / "purity.txt").read_text(encoding="utf-8").split()
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
        counts = tuple(
            _integer(row[name], name=f"{path} {name}") for name in ("alt_count", "ref_count")
        )
        observations[locus] = (*counts, _provenance_flag(row, ("input_excluded",),
                                                        "Input exclusion", default=False))
        observation_rows[locus] = row
    _, table = _read_text_table(path / "cna.txt", columns=_CNA_COLUMNS,
                                canonical=False, allow_empty=True)
    intervals = {}
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
        intervals.setdefault(chrom, []).append((start, end, major, minor, normal, normal_supplied))
        interval_rows[chrom, start, end] = row
    indexed = {}
    for chrom, segments in intervals.items():
        segments.sort()
        if any(right[0] <= left[1] for left, right in zip(segments, segments[1:])):
            raise TumorTxtError(f"{path}: overlapping CNA intervals on chromosome {chrom!r}.")
        indexed[chrom] = ([segment[0] for segment in segments], segments)
    result = purity, observations, indexed
    return (*result, observation_rows, interval_rows) if preserve_provenance else result


def read_region_directory(path: str | Path, *, preserve_provenance=True) -> tuple[dict[str, str], pd.DataFrame]:
    """Join inclusive CN intervals, retaining genuine-mapping evidence by default."""
    path = Path(path).resolve()
    regions = {region.name: _read_region(region, preserve_provenance=preserve_provenance) for region in _regions(path)}
    loci = sorted({locus for value in regions.values() for locus in value[1]})
    has_exclusions = any(observation[2] for value in regions.values() for observation in value[1].values())
    rows = []
    for sample, value in regions.items():
        purity, observations, intervals = value[:3]
        for chrom, position in loci:
            starts, segments = intervals.get(chrom, ([], []))
            index = bisect_right(starts, position) - 1
            genuine = index >= 0 and position <= segments[index][1]
            if not genuine:
                # Uncovered loci are diploid; missing reads remain missing.
                start, end, major, minor, normal, normal_supplied = position, position, 1, 1, 2, False
            else:
                start, end, major, minor, normal, normal_supplied = segments[index]
            observed = (chrom, position) in observations
            alt, ref, excluded = observations.get((chrom, position), (".", ".", False))
            row = dict(zip(SCHEMA_COLUMNS, (f"{chrom}:{position}", sample, alt, ref, int(observed), purity,
                         normal, f"{chrom}:{start}-{end}", "0", 1, major, minor)))
            row.update(_mapping_genuine=genuine, _normal_cn_supplied=normal_supplied)
            if has_exclusions:
                row["input_excluded"] = int(excluded)
            if preserve_provenance:
                row.update(chromosome_index=chrom, position=position,
                           start_position=start, end_position=end)
                row.update(("snv_"+key, cell) for key, cell in value[3].get((chrom, position), {}).items())
                row.update(("cna_"+key, cell) for key, cell in value[4].get((chrom, start, end), {}).items())
            rows.append(row)
    metadata = _validate_metadata({"tumor_id": _safe_tumor_id(path.name),
                                  "mutation_key_convention": "chromosome_index:position"}, path=path)
    table = pd.DataFrame(rows).fillna(".")
    return metadata, table if preserve_provenance else table.loc[:, [*SCHEMA_COLUMNS,
        "_mapping_genuine", "_normal_cn_supplied", *(["input_excluded"] if has_exclusions else [])]]
