"""Load CliPP2's sole on-disk tumor input format.

A tumor is one directory containing these observed, non-truth inputs:

* ``cn_clone_profiles.tsv``
* ``cn_clone_fractions.tsv``
* ``mutation_segments.tsv``
* ``purity.txt``
* ``regionN/snv.txt`` and ``regionN/cna.txt``
* ``regionN/purity.txt``, with one-based region numbers

External tools, including Battenberg/DPClust workflows, must be converted to
this package-owned format before CliPP2 is run. Hidden mutation histories and
dosages are never valid input. For one or two local allele-specific CN states,
the loader enumerates positive dosage paths on persistent phased homologs A->A
and B->B. Three-or-more-state loci fail explicitly or are count-masked because
they cannot generally be represented by the current single-switch likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from .data import (
    PathLikelihoodSpec,
    TumorData,
    canonicalize_single_switch_path_values,
    _load_observation_tsv,
)


INPUT_SCHEMA_VERSION = "2"
MODEL_ID = "clipp2_tumor_directory_ordered_occupancy_v1"
MODEL_VERSION = "1"
CANDIDATE_GENERATOR_VERSION = "phased_two_state_all_positive_dosages_v1"
PRIOR_MODE = "endpoint_excess_dosage_penalty_fixed_v1"
REPORTING_SEMANTICS = "ordered_local_cn_state_occupancy_v1"
DEFAULT_DOSAGE_PRIOR_PENALTY = 3.0
MORE_THAN_TWO_STATES = "MORE_THAN_TWO_LOCAL_CN_STATES"
NO_POSITIVE_PATH = "NO_POSITIVE_PERSISTENT_HOMOLOG_PATH"
ROOT_TABLE_COLUMNS = {
    "mutation_segments.tsv": (
        "mutation_id",
        "segment_id",
        "chromosome",
        "position",
    ),
    "cn_clone_profiles.tsv": (
        "cn_clone_id",
        "segment_id",
        "chromosome",
        "start",
        "end",
        "allele_a_cn",
        "allele_b_cn",
    ),
    "cn_clone_fractions.tsv": (
        "sample_id",
        "cn_clone_id",
        "tumor_fraction",
    ),
    "purity.txt": ("sample_id", "purity"),
}
REQUIRED_ROOT_FILES = tuple(ROOT_TABLE_COLUMNS)
REGION_TABLE_COLUMNS = {
    "snv.txt": ("chromosome_index", "position", "alt_count", "ref_count"),
    "cna.txt": (
        "chromosome_index",
        "start_position",
        "end_position",
        "major_cn",
        "minor_cn",
    ),
}
REQUIRED_REGION_FILES = ("snv.txt", "cna.txt", "purity.txt")
_FRACTION_TOL = 1e-8


class TumorInputError(ValueError):
    """Raised when a tumor directory violates CliPP2's input contract."""


class UnsupportedTumorInputError(ValueError):
    """Raised when a local state mixture is outside the single-switch model."""

    def __init__(
        self,
        reason: str,
        *,
        region_id: str,
        segment_id: str,
        detail: str,
    ) -> None:
        self.reason = str(reason)
        self.region_id = str(region_id)
        self.segment_id = str(segment_id)
        self.detail = str(detail)
        super().__init__(
            f"{self.reason}: {self.region_id}, segment {self.segment_id}: {self.detail}"
        )


def is_tumor_directory(path: str | Path) -> bool:
    """Return whether ``path`` has the required root-level input files."""

    directory = Path(path)
    return directory.is_dir() and all(
        (directory / name).is_file() for name in REQUIRED_ROOT_FILES
    )


@dataclass(frozen=True, slots=True)
class _LocalCopyNumberState:
    allele_a_cn: int
    allele_b_cn: int
    tumor_fraction: float


@dataclass(frozen=True, slots=True)
class _TumorInputTables:
    tumor_dir: Path
    mutation_ids: tuple[str, ...]
    mutation_segment_ids: tuple[str, ...]
    mutation_chromosomes: tuple[str, ...]
    mutation_positions: tuple[int, ...]
    region_indices: tuple[int, ...]
    local_states: dict[tuple[int, str], tuple[_LocalCopyNumberState, ...]]
    mean_total_cn: dict[tuple[int, str], float]


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    source: Path,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise TumorInputError(f"{source} is missing required columns: {missing}.")


def _canonical_id(value: object, *, name: str) -> str:
    if value is None or bool(pd.isna(value)):
        raise TumorInputError(f"{name} may not be missing.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise TumorInputError(f"{name} must be finite.")
        if numeric.is_integer():
            return str(int(numeric))
    normalized = str(value).strip()
    if not normalized:
        raise TumorInputError(f"{name} may not be empty.")
    return normalized


def _region_index(value: object, *, name: str = "sample_id") -> int:
    normalized = _canonical_id(value, name=name)
    match = re.fullmatch(r"region([1-9]\d*)", normalized)
    if match is None:
        raise TumorInputError(f"{name}={value!r} must be 'regionN' with one-based N.")
    return int(match.group(1)) - 1


def _chromosome(value: object) -> str:
    normalized = _canonical_id(value, name="chromosome")
    if normalized.lower().startswith("chr"):
        normalized = normalized[3:]
    if not normalized:
        raise TumorInputError("chromosome may not be empty.")
    return normalized.upper()


def _nonnegative_integer(value: object, *, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TumorInputError(f"{name} must be a nonnegative integer.") from exc
    if (
        not np.isfinite(numeric)
        or numeric < 0.0
        or not np.isclose(numeric, round(numeric), atol=1e-8, rtol=0.0)
    ):
        raise TumorInputError(f"{name} must be a nonnegative integer.")
    return int(round(numeric))


def _positive_position(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result <= 0:
        raise TumorInputError(f"{name} must be strictly positive.")
    return result


def _read_input_tables(tumor_dir: str | Path) -> _TumorInputTables:
    tumor_dir = Path(tumor_dir)
    if not tumor_dir.is_dir():
        raise FileNotFoundError(f"Tumor input directory does not exist: {tumor_dir}")

    mutation_path = tumor_dir / "mutation_segments.tsv"
    profile_path = tumor_dir / "cn_clone_profiles.tsv"
    fraction_path = tumor_dir / "cn_clone_fractions.tsv"
    for path in (mutation_path, profile_path, fraction_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required tumor input is missing: {path}")

    mutation = pd.read_csv(mutation_path, sep="\t")
    profiles = pd.read_csv(profile_path, sep="\t")
    fractions = pd.read_csv(fraction_path, sep="\t")
    _require_columns(
        mutation,
        ROOT_TABLE_COLUMNS["mutation_segments.tsv"],
        source=mutation_path,
    )
    _require_columns(
        profiles,
        ROOT_TABLE_COLUMNS["cn_clone_profiles.tsv"],
        source=profile_path,
    )
    _require_columns(
        fractions,
        ROOT_TABLE_COLUMNS["cn_clone_fractions.tsv"],
        source=fraction_path,
    )
    if mutation.empty or profiles.empty or fractions.empty:
        raise TumorInputError("Tumor input tables may not be empty.")

    mutation_ids = tuple(
        _canonical_id(value, name="mutation_id") for value in mutation["mutation_id"]
    )
    if len(set(mutation_ids)) != len(mutation_ids):
        raise TumorInputError(f"{mutation_path} contains duplicate mutation_id values.")
    mutation_segment_ids = tuple(
        _canonical_id(value, name="segment_id") for value in mutation["segment_id"]
    )
    mutation_chromosomes = tuple(_chromosome(value) for value in mutation["chromosome"])
    mutation_positions = tuple(
        _positive_position(value, name="mutation position")
        for value in mutation["position"]
    )
    mutation_coordinates = list(zip(mutation_chromosomes, mutation_positions))
    if len(set(mutation_coordinates)) != len(mutation_coordinates):
        raise TumorInputError(
            f"{mutation_path} contains duplicate chromosome/position coordinates."
        )

    profile = profiles.copy()
    profile["_clone"] = [
        _canonical_id(value, name="cn_clone_id") for value in profile["cn_clone_id"]
    ]
    profile["_segment"] = [
        _canonical_id(value, name="segment_id") for value in profile["segment_id"]
    ]
    profile["_chrom"] = [_chromosome(value) for value in profile["chromosome"]]
    profile["_start"] = [
        _positive_position(value, name="segment start") for value in profile["start"]
    ]
    profile["_end"] = [
        _positive_position(value, name="segment end") for value in profile["end"]
    ]
    profile["_a"] = [
        _nonnegative_integer(value, name="allele_a_cn")
        for value in profile["allele_a_cn"]
    ]
    profile["_b"] = [
        _nonnegative_integer(value, name="allele_b_cn")
        for value in profile["allele_b_cn"]
    ]
    if bool((profile["_end"] < profile["_start"]).any()):
        raise TumorInputError("Every copy-number clone segment must have end >= start.")
    if bool(profile.duplicated(["_clone", "_segment"]).any()):
        raise TumorInputError(
            f"{profile_path} contains duplicate cn_clone_id/segment_id rows."
        )

    clone_ids = tuple(dict.fromkeys(profile["_clone"].tolist()))
    segment_ids = tuple(dict.fromkeys(profile["_segment"].tolist()))
    expected_profile_pairs = {
        (clone_id, segment_id) for clone_id in clone_ids for segment_id in segment_ids
    }
    observed_profile_pairs = set(zip(profile["_clone"], profile["_segment"]))
    if observed_profile_pairs != expected_profile_pairs:
        raise TumorInputError(
            f"{profile_path} must contain the complete copy-number clone by segment product."
        )

    segment_coordinates: dict[str, tuple[str, int, int]] = {}
    profile_cn: dict[tuple[str, str], tuple[int, int]] = {}
    for clone_id, segment_id, chromosome, start, end, allele_a, allele_b in profile[
        ["_clone", "_segment", "_chrom", "_start", "_end", "_a", "_b"]
    ].itertuples(index=False, name=None):
        segment_id = str(segment_id)
        coordinates = (str(chromosome), int(start), int(end))
        previous = segment_coordinates.setdefault(segment_id, coordinates)
        if previous != coordinates:
            raise TumorInputError(
                f"Segment {segment_id} has inconsistent coordinates across copy-number clones."
            )
        profile_cn[(str(clone_id), segment_id)] = (
            int(allele_a),
            int(allele_b),
        )

    for mutation_id, segment_id, chromosome, position in zip(
        mutation_ids,
        mutation_segment_ids,
        mutation_chromosomes,
        mutation_positions,
    ):
        if segment_id not in segment_coordinates:
            raise TumorInputError(
                f"Mutation {mutation_id} references unknown segment {segment_id}."
            )
        segment_chromosome, start, end = segment_coordinates[segment_id]
        if chromosome != segment_chromosome or not start <= position <= end:
            raise TumorInputError(
                f"Mutation {mutation_id} coordinate {chromosome}:{position} "
                f"does not lie in segment {segment_id}."
            )

    fraction = fractions.copy()
    fraction["_region"] = [_region_index(value) for value in fraction["sample_id"]]
    fraction["_clone"] = [
        _canonical_id(value, name="cn_clone_id") for value in fraction["cn_clone_id"]
    ]
    fraction["_fraction"] = pd.to_numeric(fraction["tumor_fraction"], errors="coerce")
    if (
        bool(fraction["_fraction"].isna().any())
        or bool((fraction["_fraction"] < 0.0).any())
        or not np.all(np.isfinite(fraction["_fraction"].to_numpy(dtype=float)))
    ):
        raise TumorInputError(
            "copy-number clone tumor fractions must be finite and nonnegative."
        )
    if bool(fraction.duplicated(["_region", "_clone"]).any()):
        raise TumorInputError(
            f"{fraction_path} contains duplicate sample_id/cn_clone_id rows."
        )
    if set(fraction["_clone"]) != set(clone_ids):
        raise TumorInputError(
            f"{fraction_path} and {profile_path} contain different copy-number clone IDs."
        )
    region_indices = tuple(sorted(set(int(value) for value in fraction["_region"])))
    if region_indices != tuple(range(len(region_indices))):
        raise TumorInputError("Tumor region IDs must be contiguous from region1.")
    expected_fraction_pairs = {
        (region_index, clone_id)
        for region_index in region_indices
        for clone_id in clone_ids
    }
    observed_fraction_pairs = set(zip(fraction["_region"], fraction["_clone"]))
    if observed_fraction_pairs != expected_fraction_pairs:
        raise TumorInputError(
            f"{fraction_path} must contain the complete region by copy-number clone product."
        )
    sums = fraction.groupby("_region", sort=True)["_fraction"].sum().to_numpy()
    if not np.allclose(sums, 1.0, atol=_FRACTION_TOL, rtol=0.0):
        raise TumorInputError(
            "copy-number clone fractions must sum to one in every region."
        )

    fraction_lookup = {
        (int(region), str(clone_id)): float(weight)
        for region, clone_id, weight in fraction[
            ["_region", "_clone", "_fraction"]
        ].itertuples(index=False, name=None)
    }
    local_states: dict[tuple[int, str], tuple[_LocalCopyNumberState, ...]] = {}
    mean_total_cn: dict[tuple[int, str], float] = {}
    for region_index in region_indices:
        for segment_id in segment_ids:
            state_fractions: dict[tuple[int, int], float] = {}
            mean_total = 0.0
            for clone_id in clone_ids:
                weight = fraction_lookup[(region_index, clone_id)]
                allele_a, allele_b = profile_cn[(clone_id, segment_id)]
                state_fractions[(allele_a, allele_b)] = (
                    state_fractions.get((allele_a, allele_b), 0.0) + weight
                )
                mean_total += weight * (allele_a + allele_b)
            states = tuple(
                _LocalCopyNumberState(
                    allele_a_cn=int(allele_a),
                    allele_b_cn=int(allele_b),
                    tumor_fraction=float(weight),
                )
                for (allele_a, allele_b), weight in sorted(state_fractions.items())
                if weight > _FRACTION_TOL
            )
            if not states or not np.isclose(
                sum(state.tumor_fraction for state in states),
                1.0,
                atol=_FRACTION_TOL,
                rtol=0.0,
            ):
                raise TumorInputError(
                    f"Local CN-state fractions do not sum to one for "
                    f"region{region_index + 1}, segment {segment_id}."
                )
            local_states[(region_index, segment_id)] = states
            mean_total_cn[(region_index, segment_id)] = float(mean_total)

    return _TumorInputTables(
        tumor_dir=tumor_dir,
        mutation_ids=mutation_ids,
        mutation_segment_ids=mutation_segment_ids,
        mutation_chromosomes=mutation_chromosomes,
        mutation_positions=mutation_positions,
        region_indices=region_indices,
        local_states=local_states,
        mean_total_cn=mean_total_cn,
    )


def _read_purities(tables: _TumorInputTables) -> dict[int, float]:
    purity_path = tables.tumor_dir / "purity.txt"
    if not purity_path.is_file():
        raise FileNotFoundError(
            f"Required tumor purity table is missing: {purity_path}"
        )
    purity = pd.read_csv(purity_path, sep="\t")
    _require_columns(purity, ROOT_TABLE_COLUMNS["purity.txt"], source=purity_path)
    result: dict[int, float] = {}
    for row in purity.itertuples(index=False):
        region_index = _region_index(row.sample_id)
        value = float(row.purity)
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise TumorInputError("Tumor purities must lie in (0, 1].")
        if region_index in result:
            raise TumorInputError(f"Duplicate purity row for region{region_index + 1}.")
        result[region_index] = value
    if set(result) != set(tables.region_indices):
        raise TumorInputError(
            "purity.txt region IDs must exactly match cn_clone_fractions.tsv."
        )
    return result


def _align_region_observations(
    tables: _TumorInputTables,
    region_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    region_id = f"region{region_index + 1}"
    region_dir = tables.tumor_dir / region_id
    snv_path = region_dir / "snv.txt"
    cna_path = region_dir / "cna.txt"
    if not snv_path.is_file() or not cna_path.is_file():
        raise FileNotFoundError(f"{region_id} must contain both snv.txt and cna.txt.")
    snv = pd.read_csv(snv_path, sep="\t")
    cna = pd.read_csv(cna_path, sep="\t")
    _require_columns(
        snv,
        REGION_TABLE_COLUMNS["snv.txt"],
        source=snv_path,
    )
    _require_columns(
        cna,
        REGION_TABLE_COLUMNS["cna.txt"],
        source=cna_path,
    )

    expected = pd.DataFrame(
        {
            "_order": np.arange(len(tables.mutation_ids), dtype=int),
            "_chrom": tables.mutation_chromosomes,
            "_position": tables.mutation_positions,
        }
    )
    snv_keys = snv.copy()
    snv_keys["_chrom"] = [_chromosome(value) for value in snv["chromosome_index"]]
    snv_keys["_position"] = [
        _positive_position(value, name="SNV position") for value in snv["position"]
    ]
    if bool(snv_keys.duplicated(["_chrom", "_position"]).any()):
        raise TumorInputError(f"{snv_path} contains duplicate genomic coordinates.")
    aligned_snv = expected.merge(
        snv_keys,
        on=["_chrom", "_position"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(snv_keys) != len(expected) or bool((aligned_snv["_merge"] != "both").any()):
        raise TumorInputError(
            f"{snv_path} coordinates do not exactly match mutation_segments.tsv."
        )
    aligned_snv = aligned_snv.sort_values("_order").reset_index(drop=True)
    aligned_snv["_alt_count"] = [
        _nonnegative_integer(value, name="alt_count")
        for value in aligned_snv["alt_count"]
    ]
    aligned_snv["_ref_count"] = [
        _nonnegative_integer(value, name="ref_count")
        for value in aligned_snv["ref_count"]
    ]

    cna_keys = cna.copy()
    cna_keys["_chrom"] = [_chromosome(value) for value in cna["chromosome_index"]]
    cna_keys["_start"] = [
        _positive_position(value, name="CNA start position")
        for value in cna["start_position"]
    ]
    cna_keys["_end"] = np.asarray(
        [
            _positive_position(value, name="CNA end position")
            for value in cna["end_position"]
        ],
        dtype=int,
    )
    if np.any(
        cna_keys["_end"].to_numpy(dtype=int) < cna_keys["_start"].to_numpy(dtype=int)
    ):
        raise TumorInputError(f"{cna_path} contains end_position < start_position.")
    for column in ("major_cn", "minor_cn"):
        values = pd.to_numeric(cna_keys[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise TumorInputError(
                f"{cna_path} column {column} must be finite and nonnegative."
            )
        cna_keys[column] = values
    if np.any(
        cna_keys["major_cn"].to_numpy(dtype=float)
        < cna_keys["minor_cn"].to_numpy(dtype=float)
    ):
        raise TumorInputError(f"{cna_path} requires major_cn >= minor_cn.")

    ordered_cna = cna_keys.sort_values(["_chrom", "_start", "_end"])
    for chromosome, intervals in ordered_cna.groupby("_chrom", sort=False):
        starts = intervals["_start"].to_numpy(dtype=int)
        ends = intervals["_end"].to_numpy(dtype=int)
        if starts.size > 1 and np.any(starts[1:] <= ends[:-1]):
            raise TumorInputError(
                f"{cna_path} contains overlapping intervals on chromosome {chromosome}."
            )

    aligned_parts: list[pd.DataFrame] = []
    for chromosome, mutations in expected.groupby("_chrom", sort=False):
        intervals = ordered_cna.loc[ordered_cna["_chrom"] == chromosome]
        if intervals.empty:
            raise TumorInputError(
                f"{cna_path} does not cover mutations on chromosome {chromosome}."
            )
        starts = intervals["_start"].to_numpy(dtype=int)
        ends = intervals["_end"].to_numpy(dtype=int)
        positions = mutations["_position"].to_numpy(dtype=int)
        interval_indices = np.searchsorted(starts, positions, side="right") - 1
        covered = interval_indices >= 0
        if np.any(covered):
            covered[covered] &= positions[covered] <= ends[interval_indices[covered]]
        if not np.all(covered):
            missing_positions = positions[~covered][:5].tolist()
            raise TumorInputError(
                f"{cna_path} does not cover mutation positions "
                f"{chromosome}:{missing_positions}."
            )
        selected = intervals.iloc[interval_indices].copy().reset_index(drop=True)
        selected["_order"] = mutations["_order"].to_numpy(dtype=int)
        selected["_position"] = positions
        aligned_parts.append(selected)
    aligned_cna = (
        pd.concat(aligned_parts, ignore_index=True)
        .sort_values("_order")
        .reset_index(drop=True)
    )
    return aligned_snv, aligned_cna


def _build_observation_frame(
    tumor_dir: str | Path,
) -> pd.DataFrame:
    """Build the complete mutation-by-region table used by the numerical loader."""

    tables = _read_input_tables(tumor_dir)
    purities = _read_purities(tables)
    rows: list[dict[str, object]] = []
    for region_index in tables.region_indices:
        region_id = f"region{region_index + 1}"
        region_dir = tables.tumor_dir / region_id
        region_purity_path = region_dir / "purity.txt"
        if not region_purity_path.is_file():
            raise FileNotFoundError(
                f"Required region purity file is missing: {region_purity_path}"
            )
        region_purity = float(region_purity_path.read_text(encoding="utf-8").strip())
        if not np.isclose(
            region_purity,
            purities[region_index],
            atol=1e-12,
            rtol=0.0,
        ):
            raise TumorInputError(
                f"{region_id}/purity.txt disagrees with root purity.txt."
            )

        snv, cna = _align_region_observations(tables, region_index)
        for mutation_index, mutation_id in enumerate(tables.mutation_ids):
            segment_id = tables.mutation_segment_ids[mutation_index]
            states = tables.local_states[(region_index, segment_id)]
            has_cna = any(
                state.allele_a_cn != 1 or state.allele_b_cn != 1 for state in states
            )
            major_cn = float(cna.iloc[mutation_index]["major_cn"])
            minor_cn = float(cna.iloc[mutation_index]["minor_cn"])
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "sample_id": region_id,
                    "ref_counts": int(snv.iloc[mutation_index]["_ref_count"]),
                    "alt_counts": int(snv.iloc[mutation_index]["_alt_count"]),
                    "normal_cn": 2.0,
                    "major_cn": major_cn,
                    "minor_cn": minor_cn,
                    "has_cna": bool(has_cna),
                    "purity": float(purities[region_index]),
                    "chrom": tables.mutation_chromosomes[mutation_index],
                    "position": tables.mutation_positions[mutation_index],
                }
            )
    frame = pd.DataFrame(rows)
    mutation_order = {value: index for index, value in enumerate(tables.mutation_ids)}
    frame["_mutation_order"] = frame["mutation_id"].map(mutation_order)
    frame["_region_order"] = frame["sample_id"].map(
        {f"region{value + 1}": value for value in tables.region_indices}
    )
    return (
        frame.sort_values(["_mutation_order", "_region_order"])
        .drop(columns=["_mutation_order", "_region_order"])
        .reset_index(drop=True)
    )


def load_tumor_directory(
    tumor_dir: str | Path,
    *,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    eps: float = 1e-6,
) -> TumorData:
    """Validate one tumor directory and construct objective-ready ``TumorData``."""

    tumor_dir = Path(tumor_dir)
    frame = _build_observation_frame(tumor_dir)
    with tempfile.TemporaryDirectory(prefix="clipp2_tumor_input_") as temporary_dir:
        input_path = Path(temporary_dir) / f"{tumor_dir.name}.tsv"
        frame.to_csv(input_path, sep="\t", index=False)
        data = _load_observation_tsv(input_path, eps=eps)
    _attach_path_likelihood(
        data,
        tumor_dir,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
        eps=eps,
    )
    return data


def _enumerate_positive_paths(
    states: tuple[_LocalCopyNumberState, ...],
) -> tuple[tuple[float, float, float], ...]:
    candidates: list[tuple[float, float, float]] = []
    if len(states) == 1:
        state = states[0]
        for copy_number in (state.allele_a_cn, state.allele_b_cn):
            for dosage in range(1, copy_number + 1):
                candidates.append((float(dosage), float(dosage), 1.0))
        if not candidates:
            return ()
        canonical, _ = canonicalize_single_switch_path_values(candidates)
        return tuple(tuple(float(value) for value in row) for row in canonical)
    if len(states) != 2:
        return ()

    state1, state2 = states
    for copy1, copy2 in (
        (state1.allele_a_cn, state2.allele_a_cn),
        (state1.allele_b_cn, state2.allele_b_cn),
    ):
        if copy1 <= 0 or copy2 <= 0:
            continue
        for dosage1 in range(1, copy1 + 1):
            for dosage2 in range(1, copy2 + 1):
                candidates.append(
                    (
                        float(dosage1),
                        float(dosage2),
                        state1.tumor_fraction,
                    )
                )
                candidates.append(
                    (
                        float(dosage2),
                        float(dosage1),
                        state2.tumor_fraction,
                    )
                )
    if not candidates:
        return ()
    canonical, _ = canonicalize_single_switch_path_values(candidates)
    return tuple(tuple(float(value) for value in row) for row in canonical)


def _attach_path_likelihood(
    data: TumorData,
    tumor_dir: str | Path,
    *,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    eps: float = 1e-6,
) -> None:
    """Attach exact one/two-state ordered paths from the tumor input tables."""

    normalized_policy = str(unsupported_policy).strip().lower()
    if normalized_policy not in {"error", "mask"}:
        raise ValueError("unsupported_policy must be 'error' or 'mask'.")
    prior_penalty = float(dosage_prior_penalty)
    if not np.isfinite(prior_penalty) or prior_penalty < 0.0:
        raise ValueError("dosage_prior_penalty must be finite and nonnegative.")
    tables = _read_input_tables(tumor_dir)
    if data.tumor_id != tables.tumor_dir.name:
        raise TumorInputError(
            f"TumorData ID {data.tumor_id!r} does not match tumor directory "
            f"{tables.tumor_dir.name!r}."
        )
    if list(data.mutation_ids) != list(tables.mutation_ids):
        raise TumorInputError(
            "TumorData mutation IDs/order must exactly match mutation_segments.tsv."
        )
    data_to_input_region = {
        data_region_index: _region_index(region_id, name="region_id")
        for data_region_index, region_id in enumerate(data.region_ids)
    }
    if set(data_to_input_region.values()) != set(tables.region_indices):
        raise TumorInputError(
            "TumorData regions must exactly match tumor input region IDs."
        )

    cell_paths: list[list[tuple[tuple[float, float, float], ...]]] = []
    unsupported = np.full(
        (data.num_mutations, data.num_regions),
        None,
        dtype=object,
    )
    mean_total_cn = np.empty((data.num_mutations, data.num_regions), dtype=float)
    has_cna = np.empty((data.num_mutations, data.num_regions), dtype=bool)
    supported = np.ones((data.num_mutations, data.num_regions), dtype=bool)

    for mutation_index, segment_id in enumerate(tables.mutation_segment_ids):
        row_paths: list[tuple[tuple[float, float, float], ...]] = []
        for data_region_index in range(data.num_regions):
            input_region_index = data_to_input_region[data_region_index]
            states = tables.local_states[(input_region_index, segment_id)]
            mean_total_cn[mutation_index, data_region_index] = tables.mean_total_cn[
                (input_region_index, segment_id)
            ]
            has_cna[mutation_index, data_region_index] = any(
                state.allele_a_cn != 1 or state.allele_b_cn != 1 for state in states
            )
            reason: str | None = None
            detail = ""
            if len(states) > 2:
                reason = MORE_THAN_TWO_STATES
                detail = f"observed {len(states)} positive-fraction local CN states"
            else:
                paths = _enumerate_positive_paths(states)
                if not paths:
                    reason = NO_POSITIVE_PATH
                    detail = "no persistent phased homolog has positive copy number"
            if reason is not None:
                if normalized_policy == "error":
                    raise UnsupportedTumorInputError(
                        reason,
                        region_id=f"region{input_region_index + 1}",
                        segment_id=segment_id,
                        detail=detail,
                    )
                supported[mutation_index, data_region_index] = False
                unsupported[mutation_index, data_region_index] = reason
                paths = ((1.0, 1.0, 1.0),)
            row_paths.append(paths)
        cell_paths.append(row_paths)

    max_paths = max(len(paths) for row_paths in cell_paths for paths in row_paths)
    shape = (data.num_mutations, data.num_regions, max_paths)
    first_copy = np.zeros(shape, dtype=float)
    second_copy = np.zeros(shape, dtype=float)
    switch_fraction = np.zeros(shape, dtype=float)
    log_prior = np.full(shape, -np.inf, dtype=float)
    valid = np.zeros(shape, dtype=bool)
    for mutation_index, row_paths in enumerate(cell_paths):
        for region_index, paths in enumerate(row_paths):
            endpoint_mass = np.asarray(
                [
                    first * switch + second * (1.0 - switch)
                    for first, second, switch in paths
                ],
                dtype=float,
            )
            raw_log_prior = -prior_penalty * np.maximum(
                endpoint_mass - 1.0,
                0.0,
            )
            max_log_prior = float(np.max(raw_log_prior))
            normalized_log_prior = raw_log_prior - (
                max_log_prior + np.log(np.sum(np.exp(raw_log_prior - max_log_prior)))
            )
            for path_index, (first, second, switch) in enumerate(paths):
                first_copy[mutation_index, region_index, path_index] = first
                second_copy[mutation_index, region_index, path_index] = second
                switch_fraction[mutation_index, region_index, path_index] = switch
                log_prior[mutation_index, region_index, path_index] = (
                    normalized_log_prior[path_index]
                )
                valid[mutation_index, region_index, path_index] = True

    likelihood = PathLikelihoodSpec(
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        candidate_generator_version=CANDIDATE_GENERATOR_VERSION,
        prior_mode=f"{PRIOR_MODE}:alpha={prior_penalty:.12g}",
        first_copy=first_copy,
        second_copy=second_copy,
        switch_fraction=switch_fraction,
        log_prior=log_prior,
        valid=valid,
    )
    purity = np.asarray(data.purity, dtype=float)
    normal_cn = np.asarray(data.normal_cn, dtype=float)
    denominator = (1.0 - purity) * normal_cn + purity * mean_total_cn
    if np.any(denominator <= 0.0):
        raise TumorInputError("Copy-number mixture produced a nonpositive denominator.")
    scaling = purity / denominator

    old_observed = (
        np.ones_like(supported)
        if data.count_observed is None
        else np.asarray(data.count_observed, dtype=bool)
    )
    data.path_likelihood = likelihood
    data.path_annotations = None
    data.path_reporting_semantics = REPORTING_SEMANTICS
    fingerprint_payload = (
        MODEL_ID,
        MODEL_VERSION,
        CANDIDATE_GENERATOR_VERSION,
        PRIOR_MODE,
        prior_penalty,
        tuple(tables.mutation_segment_ids),
        tuple(sorted(tables.mean_total_cn.items())),
        tuple(unsupported.reshape(-1)),
    )
    reporting_fingerprint = hashlib.sha256(
        repr(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    data.path_reporting_fingerprint = reporting_fingerprint
    data.path_unsupported_reason = unsupported
    data.mean_tumor_total_cn = mean_total_cn
    data.scaling = scaling
    data.has_cna = has_cna
    data.phi_upper = np.ones_like(data.phi_upper, dtype=float)
    data.phi_init = np.clip(np.asarray(data.phi_init, dtype=float), float(eps), 1.0)
    data.count_observed = old_observed & supported
    return None


__all__ = [
    "DEFAULT_DOSAGE_PRIOR_PENALTY",
    "INPUT_SCHEMA_VERSION",
    "MODEL_ID",
    "REQUIRED_ROOT_FILES",
    "REQUIRED_REGION_FILES",
    "ROOT_TABLE_COLUMNS",
    "REGION_TABLE_COLUMNS",
    "TumorInputError",
    "UnsupportedTumorInputError",
    "is_tumor_directory",
    "load_tumor_directory",
]
