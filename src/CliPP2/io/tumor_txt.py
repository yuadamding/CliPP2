"""Load one validated tumor from canonical TSV or regional files."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import gzip
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULT_MISSING_COUNTS
from .data import PURITY_MEAN_ATOL, TumorData
from .read_counts import resolve_read_counts

TUMOR_TXT_SCHEMA = "clipp2.tumor.long.v1"
SCHEMA_COLUMNS = (
    "mutation_id", "sample_id", "alt_count", "ref_count", "count_observed",
    "purity", "segment_id", "cn_state_id", "cn_state_fraction", "allele_a_cn", "allele_b_cn",
)


class TumorTxtError(ValueError):
    """A tumor input violates the documented format."""


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


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


def _read_text_table(path, *, columns=SCHEMA_COLUMNS, canonical=True, allow_empty=False):
    metadata, header, rows = {}, None, []
    with _open_text(path) as handle:
        for number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##"):
                if not canonical or header is not None:
                    raise TumorTxtError(f"{path}:{number}: metadata must precede the canonical header.")
                key, value = _parse_metadata(line, path=path, line_number=number)
                if key in metadata:
                    raise TumorTxtError(f"{path}:{number}: duplicate metadata key {key!r}.")
                metadata[key] = value
                continue
            if line.startswith("#"):
                continue
            values = next(csv.reader([line], delimiter="\t", strict=True))
            if header is None:
                if not values or len(set(values)) != len(values) or any(not v or v != v.strip() for v in values):
                    raise TumorTxtError(f"{path}:{number}: invalid, blank or duplicate header columns.")
                header = values
                missing = sorted(set(columns).difference(header))
                if missing:
                    raise TumorTxtError(f"{path} is missing required columns: {missing}.")
                continue
            if values == header:
                raise TumorTxtError(f"{path}:{number}: a second table header was found.")
            if len(values) != len(header):
                raise TumorTxtError(f"{path}:{number}: expected {len(header)} tab-delimited fields, observed {len(values)}.")
            if "" in values:
                raise TumorTxtError(f"{path}:{number}: missing values must be represented by '.'.")
            rows.append(dict(zip(header, values)))
    if header is None:
        raise TumorTxtError(f"{path} does not contain a table header.")
    if not rows and not allow_empty:
        raise TumorTxtError(f"{path} does not contain any tumor rows.")
    return _validate_metadata(metadata, path=path) if canonical else metadata, header, rows


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


def _check_normal_cn(row, context):
    if "normal_cn" in row and _finite_float(row["normal_cn"], name=f"{context} normal_cn") != 2:
        raise TumorTxtError(f"{context} normal_cn is fixed at 2.")


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
    _check_normal_cn(row, context)
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
    if allele_a < 1 or allele_a >= 2**63 or allele_a < allele_b:
        raise TumorTxtError(f"{context} requires representable integer 1 <= allele_a_cn < 2**63 and allele_a_cn >= allele_b_cn.")
    segment_id = _identifier(row["segment_id"], name=f"{context} segment_id")
    _identifier(row["cn_state_id"], name=f"{context} cn_state_id")
    from .cn_states import parse_segment_cn_states
    state = parse_segment_cn_states(row.get("segment_cn_states"), sample_id=sample_id, segment_id=segment_id)
    if state is not None and state != (allele_a, allele_b):
        raise TumorTxtError(f"{context} segment_cn_states active CN disagrees with the supplied CN pair.")
    return {
        "mutation_id": mutation_id,
        "sample_id": sample_id,
        "alt_count": alt_count,
        "ref_count": ref_count,
        "count_observed": count_observed,
        "input_excluded": _provenance_flag(row, ("input_excluded",), "Input exclusion", default=False),
        "mapping_genuine": _provenance_flag(row, ("mapping_genuine", "_mapping_genuine"), "Genuine-mapping"),
        "purity": purity,
        "segment_id": segment_id,
        "allele_a_cn": allele_a,
        "allele_b_cn": allele_b,
    }


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


def _prepare(tumor_id, parsed, *, missing_counts):
    rows, segments, purities = {}, {}, {}
    for row in parsed:
        mutation, region = row["mutation_id"], row["sample_id"]
        unit = mutation, region
        if unit in rows:
            raise TumorTxtError(f"One canonical observation is required per mutation/sample pair: {unit!r}.")
        segment = region, row["segment_id"]
        pair = row["allele_a_cn"], row["allele_b_cn"]
        if segments.setdefault(segment, pair) != pair:
            raise TumorTxtError(f"CN pair is inconsistent within region/segment {segment!r}.")
        rho = row["purity"]
        purities.setdefault(region, []).append(rho)
        rows[unit] = row

    mutations = tuple(sorted({mutation for mutation, _ in rows}))
    regions = tuple(sorted(purities))
    if not mutations:
        raise TumorTxtError("Input contains no tumor rows.")
    for region, values in purities.items():
        if not np.allclose(values, float(np.mean(values)), rtol=0., atol=PURITY_MEAN_ATOL):
            raise TumorTxtError(f"purity must be constant across sample {region!r}.")
    if len(rows) != len(mutations) * len(regions):
        missing = []
        for mutation in mutations:
            for region in regions:
                if (mutation, region) not in rows:
                    missing.append((mutation, region))
                    if len(missing) == 5:
                        break
            if len(missing) == 5:
                break
        raise TumorTxtError(f"Every mutation/sample pair must be represented; missing examples: {missing}.")

    report = resolve_read_counts(rows, policy=missing_counts)
    grid = [[rows[m, r] for r in regions] for m in mutations]

    def column(name, dtype=float):
        return np.array([[row[name] for row in cells] for cells in grid], dtype=dtype)

    return TumorData(
        tumor_id=tumor_id, mutation_ids=mutations, region_ids=regions,
        alt_counts=column("alt_count"), total_counts=column("depth"),
        purity=np.broadcast_to([min(purities[r]) for r in regions], (len(mutations), len(regions))),
        major_cn=column("allele_a_cn"), minor_cn=column("allele_b_cn"),
        count_observed=column("count_observed", bool), count_imputed=column("count_imputed", bool),
        mapping_genuine=column("mapping_genuine", bool), input_excluded=column("input_excluded", bool),
        read_count_report=report,
    )


def load_tumor_txt(path: str | Path, *, missing_counts=DEFAULT_MISSING_COUNTS) -> TumorData:
    """Validate one full input population and resolve eligible missing reads."""
    from .region_files import read_region_directory

    path = Path(path).resolve()
    if path.is_dir():
        tumor_id, rows = read_region_directory(path)
    else:
        metadata, _, source = _read_text_table(path)
        tumor_id = metadata["tumor_id"]
        rows = (_normalize_row(row, row_number=n) for n, row in enumerate(source, 1))
    return _prepare(tumor_id, rows, missing_counts=missing_counts)
