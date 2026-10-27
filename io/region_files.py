"""Adapt regional SNV/CNA/purity bundles to the canonical input contract."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path

import pandas as pd

from .._source import file_sha256
from .tumor_txt import (
    SCHEMA_COLUMNS, TumorTxtError, _finite_float, _identifier,
    _read_text_table, _safe_tumor_id, _validate_metadata,
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


def input_fingerprint(path: str | Path) -> str:
    """Bind consumed names and bytes; preserve the canonical file's SHA-256."""
    path = Path(path).resolve()
    if path.is_file():
        return file_sha256(path)
    files = [(str((region / name).relative_to(path)), file_sha256(region / name))
             for region in _regions(path) for name in _FILES]
    # The root name determines tumor ID (and sample ID for a single bundle).
    payload = ["clipp2.regional-files.v1", path.name, sorted(files)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _integer(value: str, *, name: str, minimum: int = 0) -> int:
    if not value.isascii() or not value.isdecimal() or int(value) < minimum:
        raise TumorTxtError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _read_region(path: Path):
    tokens = (path / "purity.txt").read_text(encoding="utf-8").split()
    if len(tokens) != 1:
        raise TumorTxtError(f"{path}/purity.txt must contain exactly one number.")
    purity = _finite_float(tokens[0], name=f"{path} purity")
    if not 0 < purity <= 1:
        raise TumorTxtError(f"{path} purity must lie in (0, 1].")
    _, table = _read_text_table(path / "snv.txt", columns=_SNV_COLUMNS, canonical=False)
    observations = {}
    for row in table.to_dict(orient="records"):
        chrom = _identifier(row["chromosome_index"], name=f"{path} chromosome_index")
        position = _integer(row["position"], name=f"{path} position", minimum=1)
        locus = (chrom, position)
        if locus in observations:
            raise TumorTxtError(f"{path}: duplicate SNV locus {locus!r}.")
        observations[locus] = tuple(
            _integer(row[name], name=f"{path} {name}") for name in ("alt_count", "ref_count")
        )
    _, table = _read_text_table(path / "cna.txt", columns=_CNA_COLUMNS, canonical=False)
    intervals = {}
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
        intervals.setdefault(chrom, []).append((start, end, major, minor, normal))
    indexed = {}
    for chrom, segments in intervals.items():
        segments.sort()
        if any(right[0] <= left[1] for left, right in zip(segments, segments[1:])):
            raise TumorTxtError(f"{path}: overlapping CNA intervals on chromosome {chrom!r}.")
        indexed[chrom] = ([segment[0] for segment in segments], segments)
    return purity, observations, indexed


def read_region_directory(path: str | Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Join inclusive CN intervals to the union of regional SNVs; truth is inert."""
    path = Path(path).resolve()
    regions = {region.name: _read_region(region) for region in _regions(path)}
    loci = sorted({locus for _, observations, _ in regions.values() for locus in observations})
    rows = []
    for sample, (purity, observations, intervals) in regions.items():
        for chrom, position in loci:
            starts, segments = intervals.get(chrom, ([], []))
            index = bisect_right(starts, position) - 1
            if index < 0 or position > segments[index][1]:
                raise TumorTxtError(
                    f"{path}/{sample}: no CNA interval covers {chrom}:{position}; "
                    "coverage is required even for missing regional SNVs."
                )
            start, end, major, minor, normal = segments[index]
            observed = (chrom, position) in observations
            alt, ref = observations.get((chrom, position), (".", "."))
            rows.append((f"{chrom}:{position}", sample, alt, ref, int(observed), purity,
                         normal, f"{chrom}:{start}-{end}", "0", 1, major, minor))
    metadata = _validate_metadata({"tumor_id": _safe_tumor_id(path.name)}, path=path)
    return metadata, pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
