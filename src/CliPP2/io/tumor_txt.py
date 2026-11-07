"""Validate canonical tumor data, including regional SNV/CNA directory inputs."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import CNFilterRecord, CNFilterReport, TumorData, OriginalCatalog, MultiplicityBounds, _indices, PURITY_MEAN_ATOL
from ..config import DEFAULT_MAX_MAJOR_CN, validate_max_major_cn

TUMOR_TXT_SCHEMA = "clipp2.tumor.long.v1"
CN_FILTER_POLICY_ID = "major_limit_mutation_region_v4"
MAJOR_CN_ABOVE_LIMIT = "MAJOR_CN_ABOVE_LIMIT"
INPUT_EXCLUDED = "INPUT_EXCLUDED"
INPUT_MASK_POLICY_ID = "major_limit_and_input_mask_mutation_region_v1"
NO_POSITIVE_PATH = "NO_POSITIVE_MUTANT_COPY_PATH"


# The canonical input schema is documented in README.md.
# Writers emit these first, followed by any inert provenance columns.
SCHEMA_COLUMNS = (
    "mutation_id",
    "sample_id",
    "alt_count",
    "ref_count",
    "count_observed",
    "purity",
    "normal_cn",
    "segment_id",
    "cn_state_id",
    "cn_state_fraction",
    "allele_a_cn",
    "allele_b_cn",
)
_NUMERIC_TOL = PURITY_MEAN_ATOL


class TumorTxtError(ValueError):
    """Raised when a long tumor file violates its public input contract."""


class UnsupportedTumorInputError(ValueError):
    """Raised when retained CN has no supported positive multiplicity."""

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


class NoEligibleSNVsError(ValueError):
    """All input mutations were excluded before likelihood construction."""

    def __init__(self, report: CNFilterReport, *, tumor_id: str) -> None:
        self.cn_filter_report = report
        self.tumor_id = str(tumor_id)
        super().__init__(
            f"No eligible SNVs remain for {self.tumor_id!r}: excluded all "
            f"{report.input_mutation_count} input mutations under {report.policy_id}."
        )


@dataclass(frozen=True, slots=True)
class _ValidatedLongTable:
    metadata: dict[str, str]
    mutation_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    rows_by_unit: dict[tuple[str, str], tuple[dict[str, Any], ...]]


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


def _default_tumor_id(path: Path) -> str:
    """Derive the tumor id from the file name: strip .gz, .tsv/.txt, .clipp2."""
    name = path.name
    for suffix in (".gz", ".tsv", ".txt", ".clipp2"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _validate_metadata(metadata: dict[str, str], *, path: Path) -> dict[str, str]:
    """Apply canonical defaults and validate supplied identity metadata."""
    validated = dict(metadata)
    validated.setdefault("schema", TUMOR_TXT_SCHEMA)
    validated.setdefault("tumor_id", _default_tumor_id(path))
    validated.setdefault("genome_build", "unknown")
    validated.setdefault("coordinate_system", "1-based-inclusive")
    validated.setdefault("mutation_key_convention", "opaque_mutation_id")
    validated.setdefault("missing_value", ".")
    if validated["schema"] != TUMOR_TXT_SCHEMA:
        raise TumorTxtError(
            f"{path} schema must be {TUMOR_TXT_SCHEMA!r}, not {validated['schema']!r}."
        )
    if validated["coordinate_system"] != "1-based-inclusive":
        raise TumorTxtError(f"{path} coordinate_system must be '1-based-inclusive'.")
    if validated["missing_value"] != ".":
        raise TumorTxtError(f"{path} missing_value must be '.'.")
    validated["tumor_id"] = _safe_tumor_id(validated["tumor_id"])
    _identifier(validated["genome_build"], name="genome_build")
    _identifier(validated["mutation_key_convention"], name="mutation_key_convention")
    return validated


def _read_text_table(
    path: Path, *, columns: tuple[str, ...] = SCHEMA_COLUMNS, canonical: bool = True,
    allow_empty: bool = False, preserve_extra: bool = False,
) -> tuple[dict[str, str], pd.DataFrame]:
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
    if not rows and not allow_empty:
        raise TumorTxtError(f"{path} does not contain any tumor rows.")
    if canonical:
        metadata = _validate_metadata(metadata, path=path)
    elif metadata:
        raise TumorTxtError(f"{path}: metadata is supported only in canonical TSV inputs.")
    missing_columns = sorted(set(columns).difference(header))
    if missing_columns:
        raise TumorTxtError(f"{path} is missing required columns: {missing_columns}.")
    # Extra columns are accepted as provenance but never enter normalization,
    # likelihood construction, or the objective identity.
    table = pd.DataFrame(rows, columns=header, dtype=object)
    return metadata, table.loc[:, columns] if canonical and not preserve_extra else table


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


def _same_float(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=_NUMERIC_TOL))


def _normalize_row(row: Mapping[str, object], *, row_number: int) -> dict[str, Any]:
    context = f"row {row_number}"
    mutation_id = _identifier(row["mutation_id"], name=f"{context} mutation_id")
    sample_id = _identifier(row["sample_id"], name=f"{context} sample_id")
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
    if normal_cn < 0.0 or normal_cn != np.rint(normal_cn):
        raise TumorTxtError(f"{context} normal_cn must be a nonnegative integer.")
    fraction = _finite_float(
        row["cn_state_fraction"],
        name=f"{context} cn_state_fraction",
    )
    if fraction != 1.0:
        raise TumorTxtError(f"{context} requires one single-state CN observation with cn_state_fraction=1.")
    allele_a = _nonnegative_integer(
        row["allele_a_cn"],
        name=f"{context} allele_a_cn",
    )
    allele_b = _nonnegative_integer(
        row["allele_b_cn"],
        name=f"{context} allele_b_cn",
    )
    assert allele_a is not None and allele_b is not None
    if allele_a < 1 or allele_a < allele_b:
        raise TumorTxtError(f"{context} requires integer allele_a_cn >= 1 and allele_a_cn >= allele_b_cn.")
    return {
        "mutation_id": mutation_id,
        "sample_id": sample_id,
        "alt_count": alt_count,
        "ref_count": ref_count,
        "count_observed": count_observed,
        "input_excluded": _provenance_flag(row, ("input_excluded",), "Input exclusion", default=False),
        "mapping_genuine": _provenance_flag(row, ("mapping_genuine", "_mapping_genuine"), "Genuine-mapping"),
        "normal_cn_supplied": _provenance_flag(row, ("normal_cn_supplied", "_normal_cn_supplied"), "Normal-CN"),
        "purity": purity,
        "normal_cn": normal_cn,
        "segment_id": _identifier(
            row["segment_id"],
            name=f"{context} segment_id",
        ),
        "cn_state_id": _identifier(
            row["cn_state_id"],
            name=f"{context} cn_state_id",
        ),
        "cn_state_fraction": fraction,
        "allele_a_cn": allele_a,
        "allele_b_cn": allele_b,
    }


def _validate_long_table(metadata, table):
    """One canonical observation and one integer CN pair per region/segment."""
    rows, segments, purities = {}, {}, {}
    for number, source in enumerate(table.to_dict(orient="records"), 1):
        row = _normalize_row(source, row_number=number)
        unit = row["mutation_id"], row["sample_id"]
        if unit in rows:
            raise TumorTxtError(f"One canonical observation is required per mutation/sample pair: {unit!r}.")
        segment = row["sample_id"], row["segment_id"]
        pair = row["allele_a_cn"], row["allele_b_cn"]
        if segments.setdefault(segment, pair) != pair:
            raise TumorTxtError(f"CN pair is inconsistent within region/segment {segment!r}.")
        purities.setdefault(row["sample_id"], []).append(row["purity"])
        rows[unit] = (row,)
    canonical = {}
    for region, values in purities.items():
        if not np.allclose(values, float(np.mean(values)), rtol=0., atol=_NUMERIC_TOL):
            raise TumorTxtError(f"purity must be constant across sample {region!r}.")
        canonical[region] = min(values)
    for (mutation, region), (row,) in rows.items():
        row["_source_purity"] = row["purity"]
        row["purity"] = canonical[region]
    mutations = tuple(sorted({mutation for mutation, _ in rows}))
    regions = tuple(sorted(purities))
    missing = {(mutation, region) for mutation in mutations for region in regions} - set(rows)
    if missing:
        raise TumorTxtError(f"Every mutation/sample pair must be represented; missing examples: {sorted(missing)[:5]}.")
    return _ValidatedLongTable(dict(metadata), mutations, regions, rows)


def _filter_snv_cn(
    validated: _ValidatedLongTable,
    *, max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
) -> tuple[_ValidatedLongTable, CNFilterReport]:
    """Exclude declared pairs or CN above the limit, retaining other regions."""
    max_major_cn = validate_max_major_cn(max_major_cn)
    records: list[CNFilterRecord] = []
    excluded_pairs: set[tuple[str, str]] = set()
    for (mutation_id, sample_id), rows in sorted(validated.rows_by_unit.items()):
        segment_id = rows[0]["segment_id"]
        max_major = rows[0]["allele_a_cn"]
        if rows[0]["input_excluded"] or max_major > max_major_cn:
            excluded_pairs.add((mutation_id, sample_id))
            records.append(CNFilterRecord(
                mutation_id, sample_id, segment_id, INPUT_EXCLUDED if rows[0]["input_excluded"] else MAJOR_CN_ABOVE_LIMIT,
                max_major,
            ))
    retained_ids = tuple(
        mutation_id for mutation_id in validated.mutation_ids
        if any((mutation_id, sample_id) not in excluded_pairs
               for sample_id in validated.sample_ids)
    )
    excluded_ids = set(validated.mutation_ids).difference(retained_ids)
    report = CNFilterReport(
        policy_id=INPUT_MASK_POLICY_ID if any(row.reason == INPUT_EXCLUDED for row in records) else CN_FILTER_POLICY_ID,
        max_major_cn=max_major_cn,
        input_mutation_count=len(validated.mutation_ids),
        retained_mutation_count=len(retained_ids),
        excluded_mutation_ids=tuple(
            mutation_id for mutation_id in validated.mutation_ids
            if mutation_id in excluded_ids
        ),
        records=tuple(records),
    )
    if not retained_ids:
        raise NoEligibleSNVsError(report, tumor_id=validated.metadata["tumor_id"])
    retained_rows = {
        unit: rows for unit, rows in validated.rows_by_unit.items()
        if unit[0] not in excluded_ids
    }
    retained_purity: dict[str, float] = {}
    for unit, rows in retained_rows.items():
        if unit in excluded_pairs:
            continue
        _, sample_id = unit
        value = min(row["_source_purity"] for row in rows)
        retained_purity[sample_id] = min(retained_purity.get(sample_id, value), value)
    retained_rows = {
        unit: tuple(dict(row, purity=retained_purity.get(unit[1], row["purity"])) for row in rows)
        for unit, rows in retained_rows.items()
    }
    return _ValidatedLongTable(validated.metadata, retained_ids, validated.sample_ids, retained_rows), report


def _build_tumor_data(validated, *, report, normal_cn_policy=None):
    """Compile exclusions to inert numerical placeholders; retain all other biology."""
    shape = len(validated.mutation_ids), len(validated.sample_ids)
    arrays = {name: np.zeros(shape) for name in ("alt_counts", "total_counts", "purity")}
    arrays.update(major_cn=np.ones(shape), minor_cn=np.ones(shape), normal_cn=np.full(shape, 2.))
    flags = {name: np.zeros(shape, dtype=bool) for name in
             ("count_observed", "input_excluded", "mapping_genuine", "normal_cn_supplied")}
    excluded = {(row.mutation_id, row.sample_id) for row in report.records}
    for i, mutation in enumerate(validated.mutation_ids):
        for j, region in enumerate(validated.sample_ids):
            row = validated.rows_by_unit[mutation, region][0]
            arrays["purity"][i, j] = row["purity"]
            for name in flags:
                flags[name][i, j] = row[name]
            if (mutation, region) in excluded:
                flags["count_observed"][i, j] = False
                continue
            if row["count_observed"]:
                arrays["alt_counts"][i, j] = row["alt_count"]
                arrays["total_counts"][i, j] = row["alt_count"] + row["ref_count"]
            arrays["major_cn"][i, j], arrays["minor_cn"][i, j] = row["allele_a_cn"], row["allele_b_cn"]
            arrays["normal_cn"][i, j] = row["normal_cn"]
    return TumorData(validated.metadata["tumor_id"], validated.mutation_ids, validated.sample_ids,
                     **arrays, **flags, cn_filter_report=report, normal_cn_policy=normal_cn_policy)


def load_tumor_txt(path: str | Path, *, max_major_cn: int = DEFAULT_MAX_MAJOR_CN,
                   normal_cn_policy=None) -> TumorData:
    """Load single-state scientific inputs, masking excluded pairs before fitting.

    Missing normal CN needs the explicit ``normal_cn_policy='diploid'`` assumption.
    Original raw provenance is retained only by ``load_original_catalog``.
    """
    return _load_input(path, max_major_cn=max_major_cn, numerical=True, normal_cn_policy=normal_cn_policy)


def load_original_catalog(path: str | Path, *, max_major_cn: int = DEFAULT_MAX_MAJOR_CN) -> OriginalCatalog:
    """Preserve every canonical source key and provenance column before exclusions."""
    return _load_input(path, max_major_cn=max_major_cn)


def _load_input(path, *, max_major_cn, numerical=False, normal_cn_policy=None):
    from .region_files import read_region_directory
    max_major_cn = validate_max_major_cn(max_major_cn)
    path = Path(path).resolve()
    metadata, table = (read_region_directory(path, preserve_provenance=not numerical) if path.is_dir()
                       else _read_text_table(path, preserve_extra=True))
    validated = _validate_long_table(metadata, table)
    if numerical:
        filtered, report = _filter_snv_cn(validated, max_major_cn=max_major_cn)
        return _build_tumor_data(filtered, report=report, normal_cn_policy=normal_cn_policy)
    rows = table.to_dict(orient="records")
    return _catalog_from_validated(metadata, table, validated, max_major_cn=max_major_cn,
                                  mapping=_mapping_provenance(rows), normal_supplied=_normal_cn_provenance(rows))


def _mapping_provenance(rows):
    """Honor public and historical provenance columns without truthy-string casts.

    Explicit flags use 0/1 (or in-memory booleans). Absent markers mean the TSV's
    explicit segment mapping is asserted genuine; repeated CN-state rows must agree.
    """
    return _boolean_provenance(rows, ("mapping_genuine", "_mapping_genuine"), "Genuine-mapping")


def _normal_cn_provenance(rows):
    """An explicit canonical normal_cn is supplied unless provenance says assumed."""
    return _boolean_provenance(rows, ("normal_cn_supplied", "_normal_cn_supplied"), "Normal-CN")


def _input_exclusion_provenance(rows):
    """A pair may be explicitly excluded while its counts and CN stay observed."""
    return _boolean_provenance(rows, ("input_excluded",), "Input exclusion", default=False)


def _boolean_provenance(rows, names, description, *, default=True):
    mapping = {}
    for row in rows:
        genuine = _provenance_flag(row, names, description, default=default)
        unit = str(row["mutation_id"]), str(row["sample_id"])
        if mapping.setdefault(unit, genuine) != genuine:
            raise TumorTxtError(f"{description} provenance disagrees within unit {unit!r}.")
    return mapping


def _provenance_flag(row, names, description, *, default=True):
    values = []
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif str(value) in ("0", "1"):
            values.append(str(value) == "1")
        else:
            raise TumorTxtError(f"{name} must be an explicit 0 or 1.")
    if values and any(value != values[0] for value in values):
        raise TumorTxtError(f"{description} provenance columns disagree.")
    return values[0] if values else default


def _catalog_from_validated(metadata, table, validated, *, max_major_cn, mapping, normal_supplied):
    """Freeze the pre-filter source already read by either public loader."""
    shape = (len(validated.mutation_ids), len(validated.sample_ids))
    numeric = {name: np.zeros(shape) for name in ("alt_counts", "total_counts", "major_cn", "minor_cn",
               "normal_cn", "purity")}
    observed, eligible = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    segments = np.empty(shape, dtype=object)
    genuine = np.ones(shape, dtype=bool)
    original_rows = table.to_dict(orient="records")
    # Match the ordinary loader's full-input, post-CN-filter canonical purity.
    canonical_purity = {}
    for (_, region), rows in validated.rows_by_unit.items():
        if rows[0]["allele_a_cn"] <= max_major_cn and not rows[0]["input_excluded"]:
            value = min(row["_source_purity"] for row in rows)
            canonical_purity[region] = min(canonical_purity.get(region, value), value)
    for i, mutation in enumerate(validated.mutation_ids):
        for j, region in enumerate(validated.sample_ids):
            rows = validated.rows_by_unit[mutation, region]
            row = rows[0]
            observed[i, j] = row["count_observed"]
            # Preserve provided counts even when count_observed=0; the mask,
            # rather than destructive normalization, defines numerical use.
            numeric["alt_counts"][i, j] = row["alt_count"] or 0
            numeric["total_counts"][i, j] = (row["alt_count"] or 0) + (row["ref_count"] or 0)
            numeric["major_cn"][i, j] = row["allele_a_cn"]
            numeric["minor_cn"][i, j] = row["allele_b_cn"]
            numeric["normal_cn"][i, j] = row["normal_cn"]
            numeric["purity"][i, j] = canonical_purity.get(region, row["purity"])
            eligible[i, j] = numeric["major_cn"][i, j] <= max_major_cn and not row["input_excluded"]
            segments[i, j] = row["segment_id"]
            genuine[i, j] = mapping[mutation, region]
    source_rows = tuple(tuple((str(key), _canonical_text_value(value)) for key, value in row.items())
                        for row in original_rows)
    provenance = tuple(tuple((key, value) for key, value in row if key not in SCHEMA_COLUMNS)
                       for row in source_rows)
    return OriginalCatalog(metadata["tumor_id"], validated.mutation_ids, validated.sample_ids,
        count_observed=observed, eligible=eligible,
        segment_ids=segments, mapping_genuine=genuine,
        normal_cn_supplied=np.asarray([[normal_supplied[m, r] for r in validated.sample_ids]
                                     for m in validated.mutation_ids], dtype=bool),
        source_rows=source_rows, provenance=provenance, metadata=tuple(metadata.items()),
        input_excluded=np.asarray([[validated.rows_by_unit[m, r][0]["input_excluded"] for r in validated.sample_ids]
                                   for m in validated.mutation_ids], dtype=bool),
        max_major_cn=max_major_cn, **numeric)


def _catalog_data(catalog, mutation_ids, *, prediction, normal_cn_policy=None):
    """Compile selected frozen observations without changing the original population."""
    from .data import validate_workflow_mapping
    mutations, rows = _indices(mutation_ids, catalog.mutation_ids, "mutation")
    validate_workflow_mapping(catalog, mutations, normal_cn_policy=normal_cn_policy)
    input_count = len(mutations)
    eligible_rows = catalog.eligible[rows].any(axis=1)
    excluded = tuple(m for m, keep in zip(mutations, eligible_rows) if not keep)
    records = tuple(CNFilterRecord(mutation, region, str(catalog.segment_ids[i, j]),
        INPUT_EXCLUDED if catalog.input_excluded[i, j] else MAJOR_CN_ABOVE_LIMIT, int(catalog.major_cn[i, j]))
        for i, mutation in zip(rows, mutations) for j, region in enumerate(catalog.region_ids) if not catalog.eligible[i, j])
    if not prediction:
        mutations = tuple(m for m, keep in zip(mutations, eligible_rows) if keep)
        rows = rows[eligible_rows]
    policy = INPUT_MASK_POLICY_ID if any(row.reason == INPUT_EXCLUDED for row in records) else CN_FILTER_POLICY_ID
    report = CNFilterReport(policy, catalog.max_major_cn, input_count, int(eligible_rows.sum()), excluded, records)
    if not prediction and not mutations:
        raise NoEligibleSNVsError(report, tumor_id=catalog.tumor_id)
    use = catalog.eligible[rows]
    observed = use & catalog.count_observed[rows]
    return TumorData(catalog.tumor_id, mutations, catalog.region_ids,
        alt_counts=np.where(observed, catalog.alt_counts[rows], 0.),
        total_counts=np.where(observed, catalog.total_counts[rows], 0.), purity=catalog.purity[rows],
        major_cn=np.where(use, catalog.major_cn[rows], 1.), minor_cn=np.where(use, catalog.minor_cn[rows], 1.),
        normal_cn=np.where(use, catalog.normal_cn[rows], 2.), count_observed=observed,
        cn_filter_report=report, original_catalog=catalog, normal_cn_supplied=catalog.normal_cn_supplied[rows],
        input_excluded=catalog.input_excluded[rows], mapping_genuine=catalog.mapping_genuine[rows],
        normal_cn_policy=normal_cn_policy)


def load_multiplicity_bounds(path: str | Path, catalog_or_data) -> MultiplicityBounds:
    """Read exactly one endpoint row for every supplied mutation–region key."""
    columns = ("mutation_id", "sample_id", "multiplicity_lower", "multiplicity_upper")
    _, table = _read_text_table(Path(path), columns=columns, canonical=False)
    mutations, regions = tuple(catalog_or_data.mutation_ids), tuple(catalog_or_data.region_ids)
    lookup = {}
    for row in table.to_dict(orient="records"):
        key = (_identifier(row["mutation_id"], name="mutation_id"),
               _identifier(row["sample_id"], name="sample_id"))
        if key in lookup:
            raise TumorTxtError(f"Duplicate multiplicity-bound key: {key}")
        endpoints = []
        for name in ("multiplicity_lower", "multiplicity_upper"):
            value = row[name]
            if not value.isascii() or not value.isdecimal() or int(value) < 1 or int(value) >= 2**63:
                raise TumorTxtError(f"{name} must be a positive int64 integer")
            endpoints.append(int(value))
        lookup[key] = endpoints
    expected = {(mutation, region) for mutation in mutations for region in regions}
    if set(lookup) != expected:
        raise TumorTxtError("Multiplicity bounds must cover exactly every supplied original mutation-region key")
    lower = np.array([[lookup[m, r][0] for r in regions] for m in mutations], dtype=np.int64)
    upper = np.array([[lookup[m, r][1] for r in regions] for m in mutations], dtype=np.int64)
    bounds = MultiplicityBounds(mutations, regions, lower, upper)
    bounds.for_data(catalog_or_data)
    return bounds


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
        return f"{float(value):.17g}"
    text = str(value)
    if "\r" in text or "\n" in text:
        raise TumorTxtError("Long tumor table values may not contain newlines.")
    return text


def write_tumor_txt(
    path: str | Path,
    table: pd.DataFrame,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Validate and write one canonical ``clipp2.tumor.long.v1`` file."""

    normalized_metadata: dict[str, str] = {}
    for raw_key, raw_value in dict(metadata or {}).items():
        key = str(raw_key)
        value = str(raw_value)
        if (
            not key
            or key != key.strip()
            or "=" in key
            or not value
            or any(character in key + value for character in "\t\r\n")
        ):
            raise TumorTxtError(
                "Metadata keys and values must be nonempty and may not contain "
                "tabs or newlines; keys may not contain '='."
            )
        if key in normalized_metadata:
            raise TumorTxtError("Metadata keys must be unique after string conversion.")
        normalized_metadata[key] = value
    destination = Path(path).resolve()
    validated_metadata = _validate_metadata(normalized_metadata, path=destination)

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
    missing_columns = sorted(set(SCHEMA_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise TumorTxtError(
            f"Long tumor table is missing required columns: {missing_columns}."
        )
    extra_columns = [column for column in frame.columns if column not in SCHEMA_COLUMNS]
    ordered_columns = [*SCHEMA_COLUMNS, *extra_columns]
    frame = frame.loc[:, ordered_columns].apply(
        lambda column: column.map(_canonical_text_value)
    )
    if frame.empty:
        raise TumorTxtError("Long tumor table may not be empty.")
    if bool((frame == "").to_numpy().any()):
        raise TumorTxtError("Missing values must be represented by '.'.")
    _validate_long_table(validated_metadata, frame)
    _mapping_provenance(frame.to_dict(orient="records"))
    _normal_cn_provenance(frame.to_dict(orient="records"))

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_order = sorted(normalized_metadata)
    with _open_text(destination, "wt") as handle:
        for key in metadata_order:
            handle.write(f"##{key}={normalized_metadata[key]}\n")
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(ordered_columns)
        writer.writerows(frame.itertuples(index=False, name=None))
    return destination


__all__ = [
    "CN_FILTER_POLICY_ID",
    "SCHEMA_COLUMNS",
    "MAJOR_CN_ABOVE_LIMIT",
    "NoEligibleSNVsError",
    "TUMOR_TXT_SCHEMA",
    "TumorTxtError",
    "UnsupportedTumorInputError",
    "load_tumor_txt",
    "load_original_catalog",
    "load_multiplicity_bounds",
    "write_tumor_txt",
]
