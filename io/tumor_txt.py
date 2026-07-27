"""Read and write CliPP2's single-file long tumor format."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import TumorData
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
from .tumor_input import UnsupportedTumorInputError


TUMOR_TXT_SCHEMA = "clipp2.tumor.long.v1"
TUMOR_TXT_MODEL_ID = PATH_LIKELIHOOD_MODEL_ID
TUMOR_TXT_MODEL_VERSION = PATH_LIKELIHOOD_MODEL_VERSION
TUMOR_TXT_CANDIDATE_GENERATOR_VERSION = PATH_CANDIDATE_GENERATOR_VERSION
TUMOR_TXT_REPORTING_SEMANTICS = "tumor_long_path_mixture_v1"
DEFAULT_DOSAGE_PRIOR_PENALTY = 3.0
MORE_THAN_TWO_STATES = "MORE_THAN_TWO_LOCAL_CN_STATES"
NO_POSITIVE_PATH = "NO_POSITIVE_MUTANT_COPY_PATH"
REQUIRED_METADATA = (
    "schema",
    "tumor_id",
    "genome_build",
    "coordinate_system",
    "missing_value",
)
REQUIRED_COLUMNS = (
    "mutation_id",
    "sample_id",
    "chromosome",
    "position",
    "ref",
    "alt",
    "alt_count",
    "ref_count",
    "count_observed",
    "purity",
    "normal_cn",
    "segment_id",
    "segment_start",
    "segment_end",
    "cn_state_id",
    "cn_state_fraction",
    "allele_a_cn",
    "allele_b_cn",
    "allele_mode",
)
_FRACTION_TOL = 1e-8
_NUMERIC_TOL = 1e-10


class TumorTxtError(ValueError):
    """Raised when a long tumor file violates its public input contract."""


@dataclass(slots=True)
class TumorTxtAnnotations:
    """Reporting-only source metadata retained by :func:`load_tumor_txt`."""

    metadata: dict[str, str]
    optional_columns: tuple[str, ...]
    rows: pd.DataFrame
    _unit_values: dict[str, np.ndarray]
    _biological_duplicates: np.ndarray

    def columns_for_indices(self, indices: np.ndarray) -> dict[str, list[object]]:
        """Return safely aligned reporting columns for valid path indices."""

        mutation = indices[:, 0]
        sample = indices[:, 1]
        path = indices[:, 2]
        result = {
            f"input_{name}": values[mutation, sample].tolist()
            for name, values in self._unit_values.items()
        }
        result["biological_duplicate_count"] = self._biological_duplicates[
            mutation, sample, path
        ].tolist()
        return result


@dataclass(frozen=True, slots=True)
class _ValidatedLongTable:
    metadata: dict[str, str]
    source_rows: pd.DataFrame
    optional_columns: tuple[str, ...]
    mutation_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    rows_by_unit: dict[tuple[str, str], tuple[dict[str, Any], ...]]
    states_by_segment: dict[
        tuple[str, str], tuple[str, tuple[LocalCopyNumberState, ...]]
    ]


def _open_text(path: Path, mode: str):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def _parse_metadata(line: str, *, path: Path, line_number: int) -> tuple[str, str]:
    payload = line[2:]
    if "=" not in payload:
        raise TumorTxtError(
            f"{path}:{line_number}: metadata must use ##key=value syntax."
        )
    key, value = payload.split("=", 1)
    if not key or key != key.strip() or not value:
        raise TumorTxtError(
            f"{path}:{line_number}: metadata keys and values must be nonempty."
        )
    if any(character in key + value for character in "\t\r\n"):
        raise TumorTxtError(
            f"{path}:{line_number}: metadata may not contain tabs or newlines."
        )
    return key, value


def _validate_metadata(metadata: dict[str, str], *, path: Path) -> dict[str, str]:
    missing_metadata = sorted(set(REQUIRED_METADATA).difference(metadata))
    if missing_metadata:
        raise TumorTxtError(f"{path} is missing required metadata: {missing_metadata}.")
    if metadata["schema"] != TUMOR_TXT_SCHEMA:
        raise TumorTxtError(
            f"{path} schema must be {TUMOR_TXT_SCHEMA!r}, not {metadata['schema']!r}."
        )
    if metadata["coordinate_system"] != "1-based-inclusive":
        raise TumorTxtError(f"{path} coordinate_system must be '1-based-inclusive'.")
    if metadata["missing_value"] != ".":
        raise TumorTxtError(f"{path} missing_value must be '.'.")
    validated = dict(metadata)
    validated["tumor_id"] = _safe_tumor_id(validated["tumor_id"])
    _identifier(validated["genome_build"], name="genome_build")
    return validated


def _read_tumor_txt_metadata(path: str | Path) -> dict[str, str]:
    """Read and validate only the metadata prefix of one canonical input."""

    input_path = Path(path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Tumor input file does not exist: {input_path}")
    metadata: dict[str, str] = {}
    header_found = False
    with _open_text(input_path, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##"):
                key, value = _parse_metadata(
                    line,
                    path=input_path,
                    line_number=line_number,
                )
                if key in metadata:
                    raise TumorTxtError(
                        f"{input_path}:{line_number}: duplicate metadata key {key!r}."
                    )
                metadata[key] = value
                continue
            if line.startswith("#"):
                continue
            header_found = True
            break
    if not header_found:
        raise TumorTxtError(f"{input_path} does not contain a table header.")
    return _validate_metadata(metadata, path=input_path)


def _read_text_table(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Tumor input file does not exist: {path}")

    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with _open_text(path, "rt") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if header is None:
                if line.startswith("##"):
                    key, value = _parse_metadata(
                        line,
                        path=path,
                        line_number=line_number,
                    )
                    if key in metadata:
                        raise TumorTxtError(
                            f"{path}:{line_number}: duplicate metadata key {key!r}."
                        )
                    metadata[key] = value
                    continue
                if line.startswith("#"):
                    continue
                header = next(csv.reader([line], delimiter="\t", strict=True))
                invalid_header_name = any(
                    not column or column != column.strip() for column in header
                )
                if not header or len(set(header)) != len(header) or invalid_header_name:
                    raise TumorTxtError(
                        f"{path}:{line_number}: the table header is empty or "
                        "contains blank, whitespace-padded, or duplicate columns."
                    )
                continue
            if line.startswith("##"):
                raise TumorTxtError(
                    f"{path}:{line_number}: metadata must precede the table header."
                )
            if line.startswith("#"):
                continue
            values = next(csv.reader([line], delimiter="\t", strict=True))
            if values == header:
                raise TumorTxtError(
                    f"{path}:{line_number}: a second table header was found."
                )
            if len(values) != len(header):
                raise TumorTxtError(
                    f"{path}:{line_number}: expected {len(header)} tab-delimited "
                    f"fields, observed {len(values)}."
                )
            if any(value == "" for value in values):
                raise TumorTxtError(
                    f"{path}:{line_number}: missing values must be represented by '.'."
                )
            rows.append(values)

    if header is None:
        raise TumorTxtError(f"{path} does not contain a table header.")
    if not rows:
        raise TumorTxtError(f"{path} does not contain any tumor rows.")
    metadata = _validate_metadata(metadata, path=path)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(header))
    if missing_columns:
        raise TumorTxtError(f"{path} is missing required columns: {missing_columns}.")
    return metadata, pd.DataFrame(rows, columns=header, dtype=object)


def _identifier(value: object, *, name: str) -> str:
    text = str(value)
    if (
        not text
        or text == "."
        or text.startswith("#")
        or text != text.strip()
        or any(character in text for character in "\t\r\n")
    ):
        raise TumorTxtError(
            f"{name} must be a nonempty string without surrounding whitespace, "
            "tabs, or newlines."
        )
    return text


def _safe_tumor_id(value: object) -> str:
    tumor_id = _identifier(value, name="tumor_id")
    if tumor_id in {".", ".."} or "/" in tumor_id or "\\" in tumor_id:
        raise TumorTxtError(
            "tumor_id must be a safe single path component without separators."
        )
    return tumor_id


def _finite_float(value: object, *, name: str) -> float:
    if str(value) == ".":
        raise TumorTxtError(f"{name} may not be missing.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TumorTxtError(f"{name} must be numeric.") from exc
    if not np.isfinite(numeric):
        raise TumorTxtError(f"{name} must be finite.")
    return numeric


def _nonnegative_integer(
    value: object,
    *,
    name: str,
    allow_missing: bool = False,
) -> int | None:
    if str(value) == ".":
        if allow_missing:
            return None
        raise TumorTxtError(f"{name} may not be missing.")
    numeric = _finite_float(value, name=name)
    if numeric < 0.0 or not np.isclose(numeric, round(numeric), rtol=0.0, atol=1e-8):
        raise TumorTxtError(f"{name} must be a nonnegative integer.")
    return int(round(numeric))


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    assert result is not None
    if result <= 0:
        raise TumorTxtError(f"{name} must be strictly positive.")
    return result


def _same_float(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=_NUMERIC_TOL))


def _normalize_row(row: Mapping[str, object], *, row_number: int) -> dict[str, Any]:
    context = f"row {row_number}"
    mutation_id = _identifier(row["mutation_id"], name=f"{context} mutation_id")
    sample_id = _identifier(row["sample_id"], name=f"{context} sample_id")
    chromosome = _identifier(row["chromosome"], name=f"{context} chromosome")
    ref = str(row["ref"])
    alt = str(row["alt"])
    if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
        raise TumorTxtError(
            f"{context} ref and alt must each be one uppercase nucleotide in A/C/G/T."
        )
    if ref == alt:
        raise TumorTxtError(f"{context} ref and alt must differ.")
    observed_text = str(row["count_observed"])
    if observed_text not in {"0", "1"}:
        raise TumorTxtError(f"{context} count_observed must be 0 or 1.")
    count_observed = observed_text == "1"
    alt_count = _nonnegative_integer(
        row["alt_count"],
        name=f"{context} alt_count",
        allow_missing=not count_observed,
    )
    ref_count = _nonnegative_integer(
        row["ref_count"],
        name=f"{context} ref_count",
        allow_missing=not count_observed,
    )
    purity = _finite_float(row["purity"], name=f"{context} purity")
    if not 0.0 < purity <= 1.0:
        raise TumorTxtError(f"{context} purity must lie in (0, 1].")
    normal_cn = _finite_float(row["normal_cn"], name=f"{context} normal_cn")
    if normal_cn < 0.0:
        raise TumorTxtError(f"{context} normal_cn must be nonnegative.")
    segment_start = _positive_integer(
        row["segment_start"],
        name=f"{context} segment_start",
    )
    segment_end = _positive_integer(
        row["segment_end"],
        name=f"{context} segment_end",
    )
    if segment_end < segment_start:
        raise TumorTxtError(f"{context} segment_end must be >= segment_start.")
    fraction = _finite_float(
        row["cn_state_fraction"],
        name=f"{context} cn_state_fraction",
    )
    if fraction <= 0.0:
        raise TumorTxtError(f"{context} cn_state_fraction must be positive.")
    allele_a = _nonnegative_integer(
        row["allele_a_cn"],
        name=f"{context} allele_a_cn",
    )
    allele_b = _nonnegative_integer(
        row["allele_b_cn"],
        name=f"{context} allele_b_cn",
    )
    assert allele_a is not None and allele_b is not None
    allele_mode = str(row["allele_mode"])
    if allele_mode not in {"phased", "unphased"}:
        raise TumorTxtError(f"{context} allele_mode must be 'phased' or 'unphased'.")
    if allele_mode == "unphased" and allele_a < allele_b:
        raise TumorTxtError(
            f"{context} unphased states require allele_a_cn >= allele_b_cn."
        )
    position = _positive_integer(row["position"], name=f"{context} position")
    if not segment_start <= position <= segment_end:
        raise TumorTxtError(
            f"{context} mutation position does not lie within its segment bounds."
        )
    return {
        "mutation_id": mutation_id,
        "sample_id": sample_id,
        "chromosome": chromosome,
        "position": position,
        "ref": ref,
        "alt": alt,
        "alt_count": alt_count,
        "ref_count": ref_count,
        "count_observed": count_observed,
        "purity": purity,
        "normal_cn": normal_cn,
        "segment_id": _identifier(
            row["segment_id"],
            name=f"{context} segment_id",
        ),
        "segment_start": segment_start,
        "segment_end": segment_end,
        "cn_state_id": _identifier(
            row["cn_state_id"],
            name=f"{context} cn_state_id",
        ),
        "cn_state_fraction": fraction,
        "allele_a_cn": allele_a,
        "allele_b_cn": allele_b,
        "allele_mode": allele_mode,
    }


def _observation_fields_agree(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    exact = (
        "chromosome",
        "position",
        "ref",
        "alt",
        "alt_count",
        "ref_count",
        "count_observed",
        "segment_id",
        "segment_start",
        "segment_end",
    )
    if any(reference[name] != candidate[name] for name in exact):
        return False
    return _same_float(reference["purity"], candidate["purity"]) and _same_float(
        reference["normal_cn"], candidate["normal_cn"]
    )


def _validate_long_table(
    metadata: dict[str, str],
    table: pd.DataFrame,
) -> _ValidatedLongTable:
    normalized = [
        _normalize_row(row, row_number=index + 1)
        for index, row in enumerate(table.to_dict(orient="records"))
    ]
    rows_by_unit_mutable: dict[tuple[str, str], list[dict[str, Any]]] = {}
    mutation_definitions: dict[str, tuple[str, int, str, str]] = {}
    sample_purities: dict[str, list[float]] = {}
    segment_definitions: dict[tuple[str, str], tuple[str, int, int, str]] = {}
    state_definitions: dict[
        tuple[str, str, str], list[tuple[float, int, int, str]]
    ] = {}
    observed_triples: set[tuple[str, str, str]] = set()

    for row in normalized:
        mutation_id = row["mutation_id"]
        sample_id = row["sample_id"]
        segment_id = row["segment_id"]
        state_id = row["cn_state_id"]
        mutation_definition = (
            row["chromosome"],
            row["position"],
            row["ref"],
            row["alt"],
        )
        previous_mutation = mutation_definitions.setdefault(
            mutation_id,
            mutation_definition,
        )
        if previous_mutation != mutation_definition:
            raise TumorTxtError(
                f"mutation_id {mutation_id!r} maps to inconsistent variant definitions."
            )
        sample_purities.setdefault(sample_id, []).append(row["purity"])
        segment_key = (sample_id, segment_id)
        segment_definition = (
            row["chromosome"],
            row["segment_start"],
            row["segment_end"],
            row["allele_mode"],
        )
        previous_segment = segment_definitions.setdefault(
            segment_key,
            segment_definition,
        )
        if previous_segment != segment_definition:
            raise TumorTxtError(
                f"Segment {segment_key!r} has inconsistent chromosome, bounds, "
                "or allele_mode."
            )
        triple = (mutation_id, sample_id, state_id)
        if triple in observed_triples:
            raise TumorTxtError(
                f"Duplicate mutation/sample/cn_state_id row found for {triple!r}."
            )
        observed_triples.add(triple)
        state_definitions.setdefault(
            (sample_id, segment_id, state_id),
            [],
        ).append(
            (
                row["cn_state_fraction"],
                row["allele_a_cn"],
                row["allele_b_cn"],
                row["allele_mode"],
            )
        )
        rows_by_unit_mutable.setdefault((mutation_id, sample_id), []).append(row)

    canonical_purity: dict[str, float] = {}
    for sample_id, purities in sample_purities.items():
        if not np.allclose(
            purities,
            float(np.mean(purities)),
            rtol=0.0,
            atol=_NUMERIC_TOL,
        ):
            raise TumorTxtError(f"purity must be constant across sample {sample_id!r}.")
        canonical_purity[sample_id] = float(min(purities))
    for row in normalized:
        row["purity"] = canonical_purity[row["sample_id"]]

    state_lookup: dict[tuple[str, str, str], LocalCopyNumberState] = {}
    state_ids_by_segment: dict[tuple[str, str], set[str]] = {}
    for key, definitions in state_definitions.items():
        fractions = np.asarray([definition[0] for definition in definitions])
        allele_definitions = {
            (definition[1], definition[2], definition[3]) for definition in definitions
        }
        if len(allele_definitions) != 1 or not np.allclose(
            fractions,
            float(np.mean(fractions)),
            rtol=0.0,
            atol=_NUMERIC_TOL,
        ):
            raise TumorTxtError(f"CN state {key!r} has an inconsistent definition.")
        allele_a, allele_b, _mode = next(iter(allele_definitions))
        state_lookup[key] = LocalCopyNumberState(
            fraction=float(np.min(fractions)),
            allele_a_cn=int(allele_a),
            allele_b_cn=int(allele_b),
        )
        state_ids_by_segment.setdefault(key[:2], set()).add(key[2])

    states_by_segment: dict[
        tuple[str, str], tuple[str, tuple[LocalCopyNumberState, ...]]
    ] = {}
    for segment_key, state_ids in state_ids_by_segment.items():
        mode = segment_definitions[segment_key][3]
        aggregated: dict[tuple[int, int], float] = {}
        for state_id in sorted(state_ids):
            state = state_lookup[(*segment_key, state_id)]
            copy_key = (state.allele_a_cn, state.allele_b_cn)
            aggregated[copy_key] = aggregated.get(copy_key, 0.0) + state.fraction
        if not np.isclose(
            sum(aggregated.values()),
            1.0,
            rtol=0.0,
            atol=_FRACTION_TOL,
        ):
            raise TumorTxtError(
                f"CN-state fractions must sum to one for segment {segment_key!r}."
            )
        states = tuple(
            LocalCopyNumberState(
                fraction=float(fraction),
                allele_a_cn=allele_a,
                allele_b_cn=allele_b,
            )
            for (allele_a, allele_b), fraction in sorted(aggregated.items())
        )
        states_by_segment[segment_key] = (mode, states)

    mutation_ids = tuple(sorted(mutation_definitions))
    sample_ids = tuple(sorted(sample_purities))
    expected_units = {
        (mutation_id, sample_id)
        for mutation_id in mutation_ids
        for sample_id in sample_ids
    }
    observed_units = set(rows_by_unit_mutable)
    if observed_units != expected_units:
        missing = sorted(expected_units.difference(observed_units))[:5]
        raise TumorTxtError(
            "Every mutation/sample unit must be represented; "
            f"missing examples: {missing}."
        )

    rows_by_unit: dict[tuple[str, str], tuple[dict[str, Any], ...]] = {}
    for unit, unit_rows in rows_by_unit_mutable.items():
        reference = unit_rows[0]
        if any(
            not _observation_fields_agree(reference, candidate)
            for candidate in unit_rows[1:]
        ):
            raise TumorTxtError(
                f"Repeated observation fields disagree within unit {unit!r}."
            )
        canonical_normal_cn = min(row["normal_cn"] for row in unit_rows)
        for row in unit_rows:
            row["normal_cn"] = canonical_normal_cn
        segment_key = (reference["sample_id"], reference["segment_id"])
        expected_state_ids = state_ids_by_segment[segment_key]
        observed_state_ids = {row["cn_state_id"] for row in unit_rows}
        if observed_state_ids != expected_state_ids:
            raise TumorTxtError(
                f"Unit {unit!r} does not contain every state from segment "
                f"{segment_key!r}."
            )
        rows_by_unit[unit] = tuple(
            sorted(unit_rows, key=lambda row: row["cn_state_id"])
        )

    return _ValidatedLongTable(
        metadata=dict(metadata),
        source_rows=table.copy(),
        optional_columns=tuple(
            column for column in table.columns if column not in REQUIRED_COLUMNS
        ),
        mutation_ids=mutation_ids,
        sample_ids=sample_ids,
        rows_by_unit=rows_by_unit,
        states_by_segment=states_by_segment,
    )


def _reporting_fingerprint(
    metadata: Mapping[str, str],
    optional_columns: tuple[str, ...],
    rows: pd.DataFrame,
) -> str:
    payload = {
        "metadata": sorted((str(key), str(value)) for key, value in metadata.items()),
        "optional_columns": list(optional_columns),
        "optional_rows": rows.loc[:, list(optional_columns)].to_dict(orient="records")
        if optional_columns
        else [],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _build_tumor_data(
    validated: _ValidatedLongTable,
    *,
    unsupported_policy: str,
    dosage_prior_penalty: float,
    eps: float,
) -> TumorData:
    mutation_ids = list(validated.mutation_ids)
    sample_ids = list(validated.sample_ids)
    mutation_index = {value: index for index, value in enumerate(mutation_ids)}
    sample_index = {value: index for index, value in enumerate(sample_ids)}
    shape = (len(mutation_ids), len(sample_ids))
    alt_counts = np.zeros(shape, dtype=np.float64)
    total_counts = np.zeros(shape, dtype=np.float64)
    count_observed = np.zeros(shape, dtype=bool)
    purity = np.empty(shape, dtype=np.float64)
    normal_cn = np.empty(shape, dtype=np.float64)
    major_cn = np.empty(shape, dtype=np.float64)
    minor_cn = np.empty(shape, dtype=np.float64)
    has_cna = np.empty(shape, dtype=bool)
    mean_total_cn = np.empty(shape, dtype=np.float64)
    unsupported_reason = np.full(shape, None, dtype=object)
    compiled_units: list[list[CompiledPathSet]] = [
        [CompiledPathSet((), (), ()) for _ in sample_ids] for _ in mutation_ids
    ]

    for unit, rows in validated.rows_by_unit.items():
        mutation_id, sample_id = unit
        i = mutation_index[mutation_id]
        j = sample_index[sample_id]
        row = rows[0]
        count_observed[i, j] = bool(row["count_observed"])
        if row["count_observed"]:
            assert row["alt_count"] is not None and row["ref_count"] is not None
            alt_counts[i, j] = float(row["alt_count"])
            total_counts[i, j] = float(row["alt_count"] + row["ref_count"])
        purity[i, j] = float(row["purity"])
        normal_cn[i, j] = float(row["normal_cn"])
        mode, states = validated.states_by_segment[(sample_id, row["segment_id"])]
        mean_total_cn[i, j] = sum(
            state.fraction * (state.allele_a_cn + state.allele_b_cn) for state in states
        )
        dominant = dominant_copy_number_state(states)
        major_cn[i, j] = float(max(dominant.allele_a_cn, dominant.allele_b_cn))
        minor_cn[i, j] = float(min(dominant.allele_a_cn, dominant.allele_b_cn))
        has_cna[i, j] = any(
            state.allele_a_cn != 1 or state.allele_b_cn != 1 for state in states
        )
        reason: str | None = None
        if len(states) > 2:
            reason = MORE_THAN_TWO_STATES
            detail = f"observed {len(states)} distinct positive local CN states"
            compiled = CompiledPathSet((), (), ())
        else:
            compiled = compile_single_switch_paths(
                states,
                allele_mode=mode,
                dosage_prior_penalty=dosage_prior_penalty,
            )
            if not compiled.paths:
                reason = NO_POSITIVE_PATH
                detail = "no positive mutant-copy dosage path exists"
        if reason is not None:
            if unsupported_policy == "error":
                raise UnsupportedTumorInputError(
                    reason,
                    region_id=sample_id,
                    segment_id=row["segment_id"],
                    detail=detail,
                )
            count_observed[i, j] = False
            unsupported_reason[i, j] = reason
            compiled = CompiledPathSet(
                paths=((1.0, 1.0, 1.0),),
                log_prior=(0.0,),
                biological_duplicate_count=(1,),
            )
        compiled_units[i][j] = compiled

    alt_counts = np.where(count_observed, alt_counts, 0.0)
    total_counts = np.where(count_observed, total_counts, 0.0)
    denominator = (1.0 - purity) * normal_cn + purity * mean_total_cn
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise TumorTxtError(
            "Every mutation/sample unit must have a positive normal-plus-tumor "
            "copy-number denominator."
        )
    scaling = purity / denominator
    path_likelihood, biological_duplicates = build_path_likelihood(
        compiled_units,
        model_id=TUMOR_TXT_MODEL_ID,
        model_version=TUMOR_TXT_MODEL_VERSION,
        candidate_generator_version=TUMOR_TXT_CANDIDATE_GENERATOR_VERSION,
        prior_mode=path_prior_mode(dosage_prior_penalty),
    )

    optional_unit_values: dict[str, np.ndarray] = {}
    for column in validated.optional_columns:
        values = np.full(shape, ".", dtype=object)
        all_unit_constant = True
        for unit, rows in validated.rows_by_unit.items():
            source_indices = [
                validated.source_rows.index[
                    (validated.source_rows["mutation_id"] == unit[0])
                    & (validated.source_rows["sample_id"] == unit[1])
                ].tolist()
            ][0]
            source_values = {
                str(validated.source_rows.at[index, column]) for index in source_indices
            }
            if len(source_values) != 1:
                all_unit_constant = False
                break
            values[mutation_index[unit[0]], sample_index[unit[1]]] = next(
                iter(source_values)
            )
        if all_unit_constant:
            optional_unit_values[column] = values

    annotations = TumorTxtAnnotations(
        metadata=dict(validated.metadata),
        optional_columns=validated.optional_columns,
        rows=validated.source_rows.copy(),
        _unit_values=optional_unit_values,
        _biological_duplicates=biological_duplicates,
    )
    data = TumorData(
        tumor_id=validated.metadata["tumor_id"],
        mutation_ids=mutation_ids,
        region_ids=sample_ids,
        alt_counts=alt_counts,
        total_counts=total_counts,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        has_cna=has_cna,
        scaling=scaling,
        phi_upper=np.ones(shape, dtype=np.float64),
        phi_init=np.full(shape, 0.5, dtype=np.float64),
        init_major_mask=np.zeros(shape, dtype=bool),
        count_observed=count_observed,
        path_likelihood=path_likelihood,
        path_annotations=annotations,
        path_reporting_semantics=TUMOR_TXT_REPORTING_SEMANTICS,
        path_reporting_fingerprint=_reporting_fingerprint(
            validated.metadata,
            validated.optional_columns,
            validated.source_rows,
        ),
        path_unsupported_reason=unsupported_reason,
        mean_tumor_total_cn=mean_total_cn,
    )

    data.phi_init = initialize_path_marginal_phi(data, eps=eps)
    return data


def load_tumor_txt(
    path: str | Path,
    *,
    unsupported_policy: str = "error",
    dosage_prior_penalty: float = DEFAULT_DOSAGE_PRIOR_PENALTY,
    eps: float = 1e-6,
) -> TumorData:
    """Load one ``clipp2.tumor.long.v1`` file directly into ``TumorData``."""

    policy = str(unsupported_policy).strip().lower()
    if policy not in {"error", "mask"}:
        raise ValueError("unsupported_policy must be 'error' or 'mask'.")
    penalty = float(dosage_prior_penalty)
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError("dosage_prior_penalty must be finite and nonnegative.")
    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    input_path = Path(path).resolve()
    metadata, table = _read_text_table(input_path)
    validated = _validate_long_table(metadata, table)
    return _build_tumor_data(
        validated,
        unsupported_policy=policy,
        dosage_prior_penalty=penalty,
        eps=epsilon,
    )


def _alleles_for_mutations(
    mutation: pd.DataFrame,
    mutation_ids: list[str],
    alleles: Mapping[str, tuple[str, str]] | None,
) -> dict[str, tuple[str, str]]:
    if {"ref", "alt"}.issubset(mutation.columns):
        result = {
            str(row.mutation_id): (str(row.ref), str(row.alt))
            for row in mutation.itertuples(index=False)
        }
    elif alleles is not None:
        result = {
            str(mutation_id): (str(values[0]), str(values[1]))
            for mutation_id, values in alleles.items()
        }
    else:
        raise TumorTxtError(
            "The legacy directory does not contain ref/alt alleles. Supply an "
            "alleles mapping keyed by mutation_id for a lossless conversion."
        )
    missing = sorted(set(mutation_ids).difference(result))
    if missing:
        raise TumorTxtError(f"Missing ref/alt alleles for mutation IDs: {missing[:5]}.")
    for mutation_id in mutation_ids:
        ref, alt = result[mutation_id]
        if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
            raise TumorTxtError(
                f"Alleles for mutation {mutation_id!r} must be uppercase SNVs."
            )
        if ref == alt:
            raise TumorTxtError(f"Alleles for mutation {mutation_id!r} must differ.")
    return result


def _format_number(value: float) -> str:
    return f"{float(value):.17g}"


def _canonical_text_value(value: object) -> str:
    if value is None or (
        isinstance(value, (float, np.floating)) and bool(np.isnan(value))
    ):
        return "."
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise TumorTxtError("Long tumor tables may not contain infinite values.")
        return _format_number(float(value))
    text = str(value)
    if "\r" in text or "\n" in text:
        raise TumorTxtError("Long tumor table values may not contain newlines.")
    return text


def write_tumor_txt(
    path: str | Path,
    table: pd.DataFrame,
    metadata: Mapping[str, object],
) -> Path:
    """Validate and write one canonical ``clipp2.tumor.long.v1`` file."""

    normalized_metadata: dict[str, str] = {}
    for raw_key, raw_value in dict(metadata).items():
        key = str(raw_key)
        value = str(raw_value)
        if (
            not key
            or key != key.strip()
            or not value
            or any(character in key + value for character in "\t\r\n")
        ):
            raise TumorTxtError(
                "Metadata keys and values must be nonempty and may not contain "
                "tabs or newlines."
            )
        normalized_metadata[key] = value
    missing_metadata = sorted(set(REQUIRED_METADATA).difference(normalized_metadata))
    if missing_metadata:
        raise TumorTxtError(f"Missing required metadata: {missing_metadata}.")
    if normalized_metadata["schema"] != TUMOR_TXT_SCHEMA:
        raise TumorTxtError(f"schema must be {TUMOR_TXT_SCHEMA!r}.")
    if normalized_metadata["coordinate_system"] != "1-based-inclusive":
        raise TumorTxtError("coordinate_system must be '1-based-inclusive'.")
    if normalized_metadata["missing_value"] != ".":
        raise TumorTxtError("missing_value must be '.'.")
    normalized_metadata["tumor_id"] = _safe_tumor_id(normalized_metadata["tumor_id"])
    _identifier(normalized_metadata["genome_build"], name="genome_build")

    frame = pd.DataFrame(table).copy()
    if not frame.columns.is_unique:
        raise TumorTxtError("Long tumor table columns must be unique.")
    for column in frame.columns:
        if (
            not isinstance(column, str)
            or not column
            or column != column.strip()
            or any(character in column for character in "\t\r\n")
        ):
            raise TumorTxtError(
                "Long tumor table column names must be nonempty strings without "
                "surrounding whitespace, tabs, or newlines."
            )
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise TumorTxtError(
            f"Long tumor table is missing required columns: {missing_columns}."
        )
    optional_columns = [
        column for column in frame.columns if column not in REQUIRED_COLUMNS
    ]
    ordered_columns = [*REQUIRED_COLUMNS, *optional_columns]
    frame = frame.loc[:, ordered_columns].apply(
        lambda column: column.map(_canonical_text_value)
    )
    if frame.empty:
        raise TumorTxtError("Long tumor table may not be empty.")
    if bool((frame == "").to_numpy().any()):
        raise TumorTxtError("Missing values must be represented by '.'.")
    _validate_long_table(normalized_metadata, frame)

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_order = [
        *REQUIRED_METADATA,
        *sorted(set(normalized_metadata).difference(REQUIRED_METADATA)),
    ]
    with _open_text(destination, "wt") as handle:
        for key in metadata_order:
            handle.write(f"##{key}={normalized_metadata[key]}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(ordered_columns)
        writer.writerows(frame.itertuples(index=False, name=None))
    return destination


def convert_tumor_directory_to_txt(
    tumor_dir: str | Path,
    output_path: str | Path,
    *,
    alleles: Mapping[str, tuple[str, str]] | None = None,
    genome_build: str = "unknown",
) -> Path:
    """Export the current legacy directory format to one canonical long file.

    Legacy bundles do not normally contain REF/ALT alleles, so callers must
    supply ``alleles`` unless those columns were added to
    ``mutation_segments.tsv``.
    """

    from .tumor_input import _build_observation_frame, load_tumor_directory

    directory = Path(tumor_dir).resolve()
    raw_observations, _tables = _build_observation_frame(directory)
    observation_lookup = {
        (str(row.mutation_id), str(row.sample_id)): row
        for row in raw_observations.itertuples(index=False)
    }
    data = load_tumor_directory(directory, unsupported_policy="mask")
    mutation_path = directory / "mutation_segments.tsv"
    mutation = pd.read_csv(
        mutation_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    mutation["mutation_id"] = mutation["mutation_id"].astype(str)
    allele_lookup = _alleles_for_mutations(
        mutation,
        data.mutation_ids,
        alleles,
    )
    mutation_lookup = {
        str(row.mutation_id): row for row in mutation.itertuples(index=False)
    }
    rows: list[list[object]] = []
    for sample_index, sample_id in enumerate(data.region_ids):
        cna = pd.read_csv(
            directory / sample_id / "cna.txt",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        for mutation_index, mutation_id in enumerate(data.mutation_ids):
            mutation_row = mutation_lookup[mutation_id]
            segment_id = str(mutation_row.segment_id)
            state_rows = cna.loc[cna["segment_id"].astype(str) == segment_id]
            if state_rows.empty:
                raise TumorTxtError(
                    f"{sample_id}/cna.txt has no rows for segment {segment_id!r}."
                )
            ref, alt = allele_lookup[mutation_id]
            raw_observation = observation_lookup[(mutation_id, sample_id)]
            alt_count = str(int(raw_observation.alt_counts))
            ref_count = str(int(raw_observation.ref_counts))
            aggregated_states: dict[tuple[int, int], float] = {}
            first_state_row: dict[tuple[int, int], object] = {}
            for state in state_rows.itertuples(index=False):
                copy_state = (
                    int(float(state.major_cn)),
                    int(float(state.minor_cn)),
                )
                aggregated_states[copy_state] = aggregated_states.get(
                    copy_state, 0.0
                ) + float(state.tumor_fraction)
                first_state_row.setdefault(copy_state, state)
            retained_states = [
                (copy_state, fraction)
                for copy_state, fraction in sorted(aggregated_states.items())
                if fraction > _FRACTION_TOL
            ]
            for state_index, (copy_state, fraction) in enumerate(
                retained_states,
                start=1,
            ):
                state = first_state_row[copy_state]
                rows.append(
                    [
                        mutation_id,
                        sample_id,
                        str(mutation_row.chromosome),
                        str(mutation_row.position),
                        ref,
                        alt,
                        alt_count,
                        ref_count,
                        1,
                        _format_number(data.purity[mutation_index, sample_index]),
                        _format_number(data.normal_cn[mutation_index, sample_index]),
                        segment_id,
                        str(state.start_position),
                        str(state.end_position),
                        f"state{state_index}",
                        _format_number(fraction),
                        str(copy_state[0]),
                        str(copy_state[1]),
                        "unphased",
                    ]
                )

    return write_tumor_txt(
        output_path,
        pd.DataFrame(rows, columns=REQUIRED_COLUMNS),
        {
            "schema": TUMOR_TXT_SCHEMA,
            "tumor_id": data.tumor_id,
            "genome_build": _identifier(genome_build, name="genome_build"),
            "coordinate_system": "1-based-inclusive",
            "missing_value": ".",
        },
    )


__all__ = [
    "DEFAULT_DOSAGE_PRIOR_PENALTY",
    "REQUIRED_COLUMNS",
    "REQUIRED_METADATA",
    "TUMOR_TXT_SCHEMA",
    "TumorTxtAnnotations",
    "TumorTxtError",
    "convert_tumor_directory_to_txt",
    "load_tumor_txt",
    "write_tumor_txt",
]
