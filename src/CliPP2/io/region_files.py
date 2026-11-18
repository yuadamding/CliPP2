"""Adapt regional SNV/CNA/purity bundles to validated observation rows."""

from bisect import bisect_right
from pathlib import Path

from .cn_states import single_cn_state
from .tumor_txt import (
    TumorTxtError, _check_normal_cn, _finite_float, _identifier, _provenance_flag, _read_text_table, _safe_tumor_id,
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
    if value == ".":
        raise TumorTxtError(f"{name} may not be missing.")
    if not value.isascii() or not value.isdecimal() or int(value) < minimum:
        raise TumorTxtError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _read_region(path):
    tokens = (path / "purity.txt").read_text(encoding="utf-8").split()
    if len(tokens) != 1:
        raise TumorTxtError(f"{path}/purity.txt must contain exactly one number.")
    purity = _finite_float(tokens[0], name=f"{path} purity")
    if not 0 < purity <= 1:
        raise TumorTxtError(f"{path} purity must lie in (0, 1].")

    _, _, rows = _read_text_table(path / "snv.txt", columns=_SNV_COLUMNS, canonical=False)
    observations = {}
    for row in rows:
        locus = (_identifier(row["chromosome_index"], name=f"{path} chromosome_index"),
                 _integer(row["position"], name=f"{path} position", minimum=1))
        if locus in observations:
            raise TumorTxtError(f"{path}: duplicate SNV locus {locus!r}.")
        observed = _provenance_flag(row, ("count_observed",), "Read observation")
        counts = tuple(None if row[name] == "." and not observed else _integer(row[name], name=f"{path} {name}")
                       for name in ("alt_count", "ref_count"))
        observations[locus] = (*counts, observed, _provenance_flag(
            row, ("input_excluded",), "Input exclusion", default=False))

    _, columns, rows = _read_text_table(path / "cna.txt", columns=_CNA_COLUMNS, canonical=False, allow_empty=True)
    state_columns = {"cn_state_id", "cn_state_fraction"}
    explicit = state_columns.issubset(columns)
    if not explicit and state_columns.intersection(columns):
        raise TumorTxtError(f"{path}: CNA states require both cn_state_id and cn_state_fraction.")
    groups = {}
    for row in rows:
        chrom = _identifier(row["chromosome_index"], name=f"{path} chromosome_index")
        start, end = (_integer(row[name], name=f"{path} {name}", minimum=1)
                      for name in ("start_position", "end_position"))
        major, minor, total = (_integer(row[name], name=f"{path} {name}") for name in ("major_cn", "minor_cn", "total_cn"))
        _check_normal_cn(row, path)
        if start > end or major < minor or total != major + minor:
            raise TumorTxtError(f"{path}: invalid CNA interval {chrom}:{start}-{end}; require "
                                "start <= end, major_cn >= minor_cn and total_cn = major_cn + minor_cn.")
        genuine = _provenance_flag(row, ("mapping_genuine", "_mapping_genuine"), "Genuine-mapping")
        groups.setdefault((chrom, start, end), []).append((genuine, major, minor, row))

    intervals = {}
    for (chrom, start, end), group in groups.items():
        segment = f"{chrom}:{start}-{end}"
        genuine, major, minor, _ = group[0]
        if len(group) > 1 and not explicit:
            raise TumorTxtError(f"{path}: duplicate CNA interval {segment}; multiple states require "
                                "cn_state_id and cn_state_fraction.")
        if any(row[0] != genuine for row in group[1:]):
            raise TumorTxtError(f"{path}: mapping flags disagree across states of {segment}.")
        if explicit:
            major, minor = single_cn_state(
                ((row["cn_state_id"], row["cn_state_fraction"], a, b) for _, a, b, row in group),
                sample_id=path.name, segment_id=segment,
            )
        intervals.setdefault(chrom, []).append((start, end, segment, major, minor, genuine))

    indexed = {}
    for chrom, segments in intervals.items():
        segments.sort()
        if any(right[0] <= left[1] for left, right in zip(segments, segments[1:])):
            raise TumorTxtError(f"{path}: overlapping CNA intervals on chromosome {chrom!r}.")
        indexed[chrom] = ([segment[0] for segment in segments], segments)
    return purity, observations, indexed


def read_region_directory(path: Path):
    """Yield typed, one-state observations after joining inclusive CNA intervals."""
    regions = {region.name: _read_region(region) for region in _regions(path)}
    loci = sorted({locus for _, observations, _ in regions.values() for locus in observations})

    def rows():
        for sample, (purity, observations, intervals) in regions.items():
            for chrom, position in loci:
                starts, segments = intervals.get(chrom, ([], []))
                idx = bisect_right(starts, position) - 1
                if idx < 0 or position > segments[idx][1]:
                    raise TumorTxtError(f"{path}: input_missing CN segment for mutation {chrom}:{position} "
                                        f"in sample {sample!r}; supply an inclusive covering CNA interval.")
                _, _, segment, major, minor, genuine = segments[idx]
                alt, ref, observed, excluded = observations.get((chrom, position), (None, None, False, False))
                yield dict(mutation_id=f"{chrom}:{position}", sample_id=sample, alt_count=alt,
                           ref_count=ref, count_observed=observed, input_excluded=excluded, purity=purity,
                           allele_a_cn=major, allele_b_cn=minor, mapping_genuine=genuine, segment_id=segment)

    return _safe_tumor_id(path.name), rows()
