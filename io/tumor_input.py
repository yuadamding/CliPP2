"""Load CliPP2's legacy clone-resolved tumor directory format.

A tumor is one directory containing these observed, non-truth inputs:

* ``mutation_segments.tsv``
* ``purity.txt``
* ``regionN/snv.txt`` and ``regionN/cna.txt``
* ``regionN/purity.txt``, with one-based region numbers

External tools, including Battenberg/DPClust workflows, must be converted to
this package-owned format before CliPP2 is run. Hidden mutation histories and
dosages are never valid input. For one or two local unphased CN states, the
loader enumerates the positive unphased dosage envelope; major/minor labels
are not treated as persistent homolog identities. Three-or-more-state loci
fail explicitly or are count-masked because they cannot generally be
represented by the current single-switch likelihood.
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

from .data import TumorData, _load_observation_tsv
from .path_compiler import (
    CompiledPathSet,
    LocalCopyNumberState,
    PATH_CANDIDATE_GENERATOR_VERSION,
    PATH_LIKELIHOOD_MODEL_ID,
    PATH_LIKELIHOOD_MODEL_VERSION,
    build_path_likelihood,
    compile_single_switch_paths,
    dominant_copy_number_state,
    initialize_path_marginal_phi,
    path_prior_mode,
)


INPUT_SCHEMA_VERSION = "3"
MODEL_ID = PATH_LIKELIHOOD_MODEL_ID
MODEL_VERSION = PATH_LIKELIHOOD_MODEL_VERSION
CANDIDATE_GENERATOR_VERSION = PATH_CANDIDATE_GENERATOR_VERSION
PRIOR_MODE = "endpoint_excess_dosage_penalty_biological_alias_mass_v1"
REPORTING_SEMANTICS = "ordered_local_cn_state_occupancy_v1"
DEFAULT_DOSAGE_PRIOR_PENALTY = 3.0
MORE_THAN_TWO_STATES = "MORE_THAN_TWO_LOCAL_CN_STATES"
NO_POSITIVE_PATH = "NO_POSITIVE_UNPHASED_COPY_PATH"
ROOT_TABLE_COLUMNS = {
    "mutation_segments.tsv": (
        "mutation_id",
        "segment_id",
        "chromosome",
        "position",
    ),
    "purity.txt": ("sample_id", "purity"),
}
REQUIRED_ROOT_FILES = tuple(ROOT_TABLE_COLUMNS)
REGION_TABLE_COLUMNS = {
    "snv.txt": ("chromosome_index", "position", "alt_count", "ref_count"),
    "cna.txt": (
        "segment_id",
        "chromosome_index",
        "start_position",
        "end_position",
        "cn_clone_id",
        "tumor_fraction",
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
class _TumorInputTables:
    tumor_dir: Path
    mutation_ids: tuple[str, ...]
    mutation_segment_ids: tuple[str, ...]
    mutation_chromosomes: tuple[str, ...]
    mutation_positions: tuple[int, ...]
    region_indices: tuple[int, ...]
    purities: dict[int, float]
    local_states: dict[tuple[int, int], tuple[LocalCopyNumberState, ...]]
    dominant_cn: dict[tuple[int, int], tuple[int, int]]
    mean_total_cn: dict[tuple[int, int], float]


@dataclass(frozen=True, slots=True)
class _RegionCopyNumberData:
    segment_ids: tuple[str, ...]
    segment_coordinates: dict[str, tuple[str, int, int]]
    clone_ids: tuple[str, ...]
    clone_profile: dict[tuple[str, str], tuple[int, int]]
    local_states: dict[int, tuple[LocalCopyNumberState, ...]]
    dominant_cn: dict[int, tuple[int, int]]
    mean_total_cn: dict[int, float]


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    *,
    source: Path,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise TumorInputError(f"{source} is missing required columns: {missing}.")


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: Iterable[str],
    *,
    source: Path,
) -> None:
    expected_columns = list(expected)
    observed_columns = list(frame.columns)
    if observed_columns != expected_columns:
        raise TumorInputError(
            f"{source} columns must be exactly {expected_columns} in that order; "
            f"observed {observed_columns}."
        )


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


def _read_purities(tumor_dir: Path) -> dict[int, float]:
    purity_path = tumor_dir / "purity.txt"
    if not purity_path.is_file():
        raise FileNotFoundError(
            f"Required tumor purity table is missing: {purity_path}"
        )
    purity = pd.read_csv(purity_path, sep="\t")
    _require_columns(purity, ROOT_TABLE_COLUMNS["purity.txt"], source=purity_path)
    if purity.empty:
        raise TumorInputError(f"{purity_path} may not be empty.")
    result: dict[int, float] = {}
    for row in purity.itertuples(index=False):
        region_index = _region_index(row.sample_id)
        try:
            value = float(row.purity)
        except (TypeError, ValueError) as exc:
            raise TumorInputError("Tumor purities must lie in (0, 1].") from exc
        if not np.isfinite(value) or value <= 0.0 or value > 1.0:
            raise TumorInputError("Tumor purities must lie in (0, 1].")
        if region_index in result:
            raise TumorInputError(f"Duplicate purity row for region{region_index + 1}.")
        result[region_index] = value
    region_indices = tuple(sorted(result))
    if region_indices != tuple(range(len(region_indices))):
        raise TumorInputError("Tumor region IDs must be contiguous from region1.")
    return result


def _region_directory_indices(tumor_dir: Path) -> tuple[int, ...]:
    indices: list[int] = []
    malformed: list[str] = []
    for path in tumor_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("region"):
            continue
        match = re.fullmatch(r"region([1-9]\d*)", path.name)
        if match is None:
            malformed.append(path.name)
        else:
            indices.append(int(match.group(1)) - 1)
    if malformed:
        raise TumorInputError(
            "Region directories must use one-based names regionN; invalid names: "
            f"{sorted(malformed)}."
        )
    return tuple(sorted(indices))


def _read_region_copy_number(
    *,
    tumor_dir: Path,
    region_index: int,
    mutation_ids: tuple[str, ...],
    mutation_segment_ids: tuple[str, ...],
    mutation_chromosomes: tuple[str, ...],
    mutation_positions: tuple[int, ...],
) -> _RegionCopyNumberData:
    region_id = f"region{region_index + 1}"
    cna_path = tumor_dir / region_id / "cna.txt"
    if not cna_path.is_file():
        raise FileNotFoundError(
            f"Required region copy-number table is missing: {cna_path}"
        )
    cna = pd.read_csv(cna_path, sep="\t")
    _require_exact_columns(cna, REGION_TABLE_COLUMNS["cna.txt"], source=cna_path)
    if cna.empty:
        raise TumorInputError(f"{cna_path} may not be empty.")

    frame = cna.copy()
    frame["_segment"] = [
        _canonical_id(value, name="segment_id") for value in frame["segment_id"]
    ]
    frame["_chrom"] = [_chromosome(value) for value in frame["chromosome_index"]]
    frame["_start"] = [
        _positive_position(value, name="CNA start position")
        for value in frame["start_position"]
    ]
    frame["_end"] = [
        _positive_position(value, name="CNA end position")
        for value in frame["end_position"]
    ]
    if bool((frame["_end"] < frame["_start"]).any()):
        raise TumorInputError(f"{cna_path} contains end_position < start_position.")
    frame["_clone"] = [
        _canonical_id(value, name="cn_clone_id") for value in frame["cn_clone_id"]
    ]
    frame["_fraction"] = pd.to_numeric(frame["tumor_fraction"], errors="coerce")
    if (
        bool(frame["_fraction"].isna().any())
        or not np.all(np.isfinite(frame["_fraction"].to_numpy(dtype=float)))
        or bool((frame["_fraction"] < 0.0).any())
    ):
        raise TumorInputError(
            f"{cna_path} tumor_fraction values must be finite and nonnegative."
        )
    frame["_major"] = [
        _nonnegative_integer(value, name="major_cn") for value in frame["major_cn"]
    ]
    frame["_minor"] = [
        _nonnegative_integer(value, name="minor_cn") for value in frame["minor_cn"]
    ]
    if bool((frame["_major"] < frame["_minor"]).any()):
        raise TumorInputError(f"{cna_path} requires major_cn >= minor_cn.")

    pair_columns = ["_segment", "_clone"]
    if bool(frame.duplicated(pair_columns).any()):
        raise TumorInputError(
            f"{cna_path} contains duplicate segment/cn_clone_id rows."
        )
    segment_coordinates: dict[str, tuple[str, int, int]] = {}
    for segment_id, chromosome, start, end in frame[
        ["_segment", "_chrom", "_start", "_end"]
    ].itertuples(index=False, name=None):
        coordinates = (str(chromosome), int(start), int(end))
        previous = segment_coordinates.setdefault(str(segment_id), coordinates)
        if previous != coordinates:
            raise TumorInputError(
                f"Segment {segment_id} has inconsistent coordinates in {cna_path}."
            )
    segments = pd.DataFrame(
        [
            {
                "_segment": segment_id,
                "_chrom": coordinates[0],
                "_start": coordinates[1],
                "_end": coordinates[2],
            }
            for segment_id, coordinates in segment_coordinates.items()
        ]
    ).sort_values(["_chrom", "_start", "_end", "_segment"])
    for chromosome, intervals in segments.groupby("_chrom", sort=False):
        starts = intervals["_start"].to_numpy(dtype=int)
        ends = intervals["_end"].to_numpy(dtype=int)
        if starts.size > 1 and np.any(starts[1:] <= ends[:-1]):
            raise TumorInputError(
                f"{cna_path} contains overlapping intervals on chromosome {chromosome}."
            )

    segment_ids = tuple(sorted(segment_coordinates))
    clone_ids = tuple(sorted(set(frame["_clone"].tolist())))
    expected_pairs = {
        (segment_id, clone_id) for segment_id in segment_ids for clone_id in clone_ids
    }
    observed_pairs = set(frame.loc[:, pair_columns].itertuples(index=False, name=None))
    if observed_pairs != expected_pairs:
        raise TumorInputError(
            f"{cna_path} must contain the complete segment by cn_clone_id product."
        )

    fraction_sums = frame.groupby("_segment", sort=False)["_fraction"].sum()
    if not np.allclose(
        fraction_sums.to_numpy(dtype=float),
        1.0,
        atol=_FRACTION_TOL,
        rtol=0.0,
    ):
        raise TumorInputError(
            f"{cna_path} tumor_fraction values must sum to one for every segment."
        )
    for clone_id, clone_rows in frame.groupby("_clone", sort=False):
        fractions = clone_rows["_fraction"].to_numpy(dtype=float)
        if not np.allclose(
            fractions,
            fractions[0],
            atol=_FRACTION_TOL,
            rtol=0.0,
        ):
            raise TumorInputError(
                f"{cna_path} tumor_fraction for cn_clone_id {clone_id} must be "
                "constant across segments."
            )

    clone_profile: dict[tuple[str, str], tuple[int, int]] = {}
    states_by_segment: dict[str, tuple[LocalCopyNumberState, ...]] = {}
    dominant_by_segment: dict[str, tuple[int, int]] = {}
    mean_by_segment: dict[str, float] = {}
    for segment_id in segment_ids:
        group = frame.loc[frame["_segment"] == segment_id]
        state_fractions: dict[tuple[int, int], float] = {}
        for clone_id, fraction, major_cn, minor_cn in group[
            ["_clone", "_fraction", "_major", "_minor"]
        ].itertuples(index=False, name=None):
            copy_state = (int(major_cn), int(minor_cn))
            weight = float(fraction)
            clone_profile[(segment_id, str(clone_id))] = copy_state
            state_fractions[copy_state] = state_fractions.get(copy_state, 0.0) + weight
        states = tuple(
            LocalCopyNumberState(
                allele_a_cn=major_cn,
                allele_b_cn=minor_cn,
                fraction=float(weight),
            )
            for (major_cn, minor_cn), weight in sorted(state_fractions.items())
            if weight > _FRACTION_TOL
        )
        if not states or not np.isclose(
            sum(state.fraction for state in states),
            1.0,
            atol=_FRACTION_TOL,
            rtol=0.0,
        ):
            raise TumorInputError(
                f"Local CN-state fractions do not sum to one in {cna_path} "
                f"for segment {segment_id}."
            )
        states_by_segment[segment_id] = states
        dominant = dominant_copy_number_state(states)
        dominant_by_segment[segment_id] = (
            dominant.allele_a_cn,
            dominant.allele_b_cn,
        )
        mean_by_segment[segment_id] = float(
            sum(
                state.fraction * (state.allele_a_cn + state.allele_b_cn)
                for state in states
            )
        )

    local_states: dict[int, tuple[LocalCopyNumberState, ...]] = {}
    dominant_cn: dict[int, tuple[int, int]] = {}
    mean_total_cn: dict[int, float] = {}
    for mutation_index, (mutation_id, segment_id, chromosome, position) in enumerate(
        zip(
            mutation_ids,
            mutation_segment_ids,
            mutation_chromosomes,
            mutation_positions,
        )
    ):
        if segment_id not in segment_coordinates:
            raise TumorInputError(
                f"Mutation {mutation_id} references segment {segment_id}, which "
                f"is absent from {cna_path}."
            )
        segment_chromosome, start, end = segment_coordinates[segment_id]
        if chromosome != segment_chromosome or not start <= position <= end:
            raise TumorInputError(
                f"Mutation {mutation_id} coordinate {chromosome}:{position} does not "
                f"lie in segment {segment_id} from {cna_path}."
            )
        local_states[mutation_index] = states_by_segment[segment_id]
        dominant_cn[mutation_index] = dominant_by_segment[segment_id]
        mean_total_cn[mutation_index] = mean_by_segment[segment_id]

    return _RegionCopyNumberData(
        segment_ids=segment_ids,
        segment_coordinates=segment_coordinates,
        clone_ids=clone_ids,
        clone_profile=clone_profile,
        local_states=local_states,
        dominant_cn=dominant_cn,
        mean_total_cn=mean_total_cn,
    )


def _read_input_tables(tumor_dir: str | Path) -> _TumorInputTables:
    tumor_dir = Path(tumor_dir).resolve()
    if not tumor_dir.is_dir():
        raise FileNotFoundError(f"Tumor input directory does not exist: {tumor_dir}")

    mutation_path = tumor_dir / "mutation_segments.tsv"
    if not mutation_path.is_file():
        raise FileNotFoundError(f"Required tumor input is missing: {mutation_path}")
    mutation = pd.read_csv(
        mutation_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    _require_columns(
        mutation,
        ROOT_TABLE_COLUMNS["mutation_segments.tsv"],
        source=mutation_path,
    )
    if mutation.empty:
        raise TumorInputError(f"{mutation_path} may not be empty.")

    normalized_mutations = sorted(
        (
            _canonical_id(row.mutation_id, name="mutation_id"),
            _canonical_id(row.segment_id, name="segment_id"),
            _chromosome(row.chromosome),
            _positive_position(row.position, name="mutation position"),
        )
        for row in mutation.itertuples(index=False)
    )
    mutation_ids = tuple(row[0] for row in normalized_mutations)
    if len(set(mutation_ids)) != len(mutation_ids):
        raise TumorInputError(f"{mutation_path} contains duplicate mutation_id values.")
    mutation_segment_ids = tuple(row[1] for row in normalized_mutations)
    mutation_chromosomes = tuple(row[2] for row in normalized_mutations)
    mutation_positions = tuple(row[3] for row in normalized_mutations)
    mutation_coordinates = list(zip(mutation_chromosomes, mutation_positions))
    if len(set(mutation_coordinates)) != len(mutation_coordinates):
        raise TumorInputError(
            f"{mutation_path} contains duplicate chromosome/position coordinates."
        )

    purities = _read_purities(tumor_dir)
    region_indices = tuple(sorted(purities))
    directory_indices = _region_directory_indices(tumor_dir)
    if directory_indices != region_indices:
        expected = {f"region{index + 1}" for index in region_indices}
        observed = {f"region{index + 1}" for index in directory_indices}
        raise TumorInputError(
            "Root purity sample IDs must exactly match region directories; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}."
        )
    local_states: dict[tuple[int, int], tuple[LocalCopyNumberState, ...]] = {}
    dominant_cn: dict[tuple[int, int], tuple[int, int]] = {}
    mean_total_cn: dict[tuple[int, int], float] = {}
    reference_segment_ids: tuple[str, ...] | None = None
    reference_coordinates: dict[str, tuple[str, int, int]] | None = None
    reference_clones: tuple[str, ...] | None = None
    reference_profile: dict[tuple[str, str], tuple[int, int]] | None = None
    for region_index in region_indices:
        region_data = _read_region_copy_number(
            tumor_dir=tumor_dir,
            region_index=region_index,
            mutation_ids=mutation_ids,
            mutation_segment_ids=mutation_segment_ids,
            mutation_chromosomes=mutation_chromosomes,
            mutation_positions=mutation_positions,
        )
        if reference_segment_ids is None:
            reference_segment_ids = region_data.segment_ids
            reference_coordinates = region_data.segment_coordinates
            reference_clones = region_data.clone_ids
            reference_profile = region_data.clone_profile
        elif (
            region_data.segment_ids != reference_segment_ids
            or region_data.segment_coordinates != reference_coordinates
            or region_data.clone_ids != reference_clones
        ):
            raise TumorInputError(
                "Every region cna.txt must contain the same segment and "
                "cn_clone_id product."
            )
        elif region_data.clone_profile != reference_profile:
            raise TumorInputError(
                "Copy-number states for each segment/cn_clone_id must agree "
                "across regions."
            )
        for mutation_index in range(len(mutation_ids)):
            key = (region_index, mutation_index)
            local_states[key] = region_data.local_states[mutation_index]
            dominant_cn[key] = region_data.dominant_cn[mutation_index]
            mean_total_cn[key] = region_data.mean_total_cn[mutation_index]

    return _TumorInputTables(
        tumor_dir=tumor_dir,
        mutation_ids=mutation_ids,
        mutation_segment_ids=mutation_segment_ids,
        mutation_chromosomes=mutation_chromosomes,
        mutation_positions=mutation_positions,
        region_indices=region_indices,
        purities=purities,
        local_states=local_states,
        dominant_cn=dominant_cn,
        mean_total_cn=mean_total_cn,
    )


def _align_region_observations(
    tables: _TumorInputTables,
    region_index: int,
) -> pd.DataFrame:
    region_id = f"region{region_index + 1}"
    region_dir = tables.tumor_dir / region_id
    snv_path = region_dir / "snv.txt"
    if not snv_path.is_file():
        raise FileNotFoundError(f"Required region SNV table is missing: {snv_path}")
    snv = pd.read_csv(snv_path, sep="\t")
    _require_columns(
        snv,
        REGION_TABLE_COLUMNS["snv.txt"],
        source=snv_path,
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
    return aligned_snv


def _build_observation_frame(
    tumor_dir: str | Path,
) -> tuple[pd.DataFrame, _TumorInputTables]:
    """Build the complete mutation-by-region table used by the numerical loader."""

    tables = _read_input_tables(tumor_dir)
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
            tables.purities[region_index],
            atol=1e-12,
            rtol=0.0,
        ):
            raise TumorInputError(
                f"{region_id}/purity.txt disagrees with root purity.txt."
            )

        snv = _align_region_observations(tables, region_index)
        for mutation_index, mutation_id in enumerate(tables.mutation_ids):
            key = (region_index, mutation_index)
            states = tables.local_states[key]
            has_cna = any(
                state.allele_a_cn != 1 or state.allele_b_cn != 1 for state in states
            )
            major_cn, minor_cn = tables.dominant_cn[key]
            rows.append(
                {
                    "mutation_id": mutation_id,
                    "sample_id": region_id,
                    "ref_counts": int(snv.iloc[mutation_index]["_ref_count"]),
                    "alt_counts": int(snv.iloc[mutation_index]["_alt_count"]),
                    "normal_cn": 2.0,
                    "major_cn": float(major_cn),
                    "minor_cn": float(minor_cn),
                    "has_cna": bool(has_cna),
                    "purity": float(tables.purities[region_index]),
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
    frame = (
        frame.sort_values(["_mutation_order", "_region_order"])
        .drop(columns=["_mutation_order", "_region_order"])
        .reset_index(drop=True)
    )
    return frame, tables


def load_tumor_directory(
    tumor_dir: str | Path,
    *,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    eps: float = 1e-6,
) -> TumorData:
    """Validate one tumor directory and construct objective-ready ``TumorData``."""

    tumor_dir = Path(tumor_dir).resolve()
    frame, tables = _build_observation_frame(tumor_dir)
    with tempfile.TemporaryDirectory(prefix="clipp2_tumor_input_") as temporary_dir:
        input_path = Path(temporary_dir) / f"{tumor_dir.name}.tsv"
        frame.to_csv(input_path, sep="\t", index=False)
        data = _load_observation_tsv(input_path, eps=eps)
    _attach_path_likelihood(
        data,
        tables,
        unsupported_policy=unsupported_policy,
        dosage_prior_penalty=dosage_prior_penalty,
        eps=eps,
    )
    return data


def _attach_path_likelihood(
    data: TumorData,
    tables: _TumorInputTables,
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

    compiled_units: list[list[CompiledPathSet]] = []
    unsupported = np.full(
        (data.num_mutations, data.num_regions),
        None,
        dtype=object,
    )
    mean_total_cn = np.empty((data.num_mutations, data.num_regions), dtype=float)
    has_cna = np.empty((data.num_mutations, data.num_regions), dtype=bool)
    supported = np.ones((data.num_mutations, data.num_regions), dtype=bool)

    for mutation_index, segment_id in enumerate(tables.mutation_segment_ids):
        compiled_row: list[CompiledPathSet] = []
        for data_region_index in range(data.num_regions):
            input_region_index = data_to_input_region[data_region_index]
            key = (input_region_index, mutation_index)
            states = tables.local_states[key]
            mean_total_cn[mutation_index, data_region_index] = tables.mean_total_cn[key]
            has_cna[mutation_index, data_region_index] = any(
                state.allele_a_cn != 1 or state.allele_b_cn != 1 for state in states
            )
            reason: str | None = None
            detail = ""
            if len(states) > 2:
                reason = MORE_THAN_TWO_STATES
                detail = f"observed {len(states)} positive-fraction local CN states"
                compiled = CompiledPathSet((), (), ())
            else:
                compiled = compile_single_switch_paths(
                    states,
                    allele_mode="unphased",
                    dosage_prior_penalty=prior_penalty,
                )
                if not compiled.paths:
                    reason = NO_POSITIVE_PATH
                    detail = "no unphased copy-number path has positive dosage"
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
                compiled = CompiledPathSet(
                    paths=((1.0, 1.0, 1.0),),
                    log_prior=(0.0,),
                    biological_duplicate_count=(1,),
                )
            compiled_row.append(compiled)
        compiled_units.append(compiled_row)

    likelihood, _biological_duplicates = build_path_likelihood(
        compiled_units,
        model_id=MODEL_ID,
        model_version=MODEL_VERSION,
        candidate_generator_version=CANDIDATE_GENERATOR_VERSION,
        prior_mode=path_prior_mode(prior_penalty),
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
    data.count_observed = old_observed & supported
    data.alt_counts = np.where(data.count_observed, data.alt_counts, 0.0)
    data.total_counts = np.where(data.count_observed, data.total_counts, 0.0)
    data.init_major_mask = np.zeros_like(data.count_observed, dtype=bool)
    data.phi_init = initialize_path_marginal_phi(data, eps=eps)
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
