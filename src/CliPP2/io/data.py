"""Immutable retained biological inputs and their CN exclusion provenance."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import hashlib
import json
from types import MappingProxyType

import numpy as np

from ..config import validate_max_major_cn

# Source purities are admitted within tau of their regional mean, then compiled
# to a canonical minimum. Each source is therefore within 2*tau of that value.
# Subsets keep the parent's canonical value rather than recomputing a minimum.
PURITY_MEAN_ATOL = 1e-10
PURITY_SOURCE_CANONICAL_ATOL = 2 * PURITY_MEAN_ATOL


def readonly_array(value: object, *, dtype=None) -> np.ndarray:
    """Copy into an immutable bytes-backed array, not a reversible write flag."""
    array = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def freeze_record(value):
    """Detach nested scientific records from their caller's mutable containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_record(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(map(freeze_record, value))
    return readonly_array(value) if isinstance(value, np.ndarray) else value


def thaw_record(value, *, arrays_as_lists=False):
    """Detach frozen mappings recursively before constructor reconstruction."""
    if isinstance(value, Mapping):
        return {key: thaw_record(item, arrays_as_lists=arrays_as_lists) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [thaw_record(item, arrays_as_lists=arrays_as_lists) for item in value]
        return tuple(items) if isinstance(value, tuple) and not arrays_as_lists else items
    if isinstance(value, np.ndarray):
        return value.tolist() if arrays_as_lists else value.copy()
    return value.item() if isinstance(value, np.generic) else value


def json_value(value):
    """Encode scientific nonfinite values explicitly, never as bare JSON NaN."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return list(map(json_value, value))
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def content_hash(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def restore_immutable_record(cls, inputs):
    """Re-run validation/freezing rather than restoring NumPy's writable state."""
    return cls(**inputs)


class ImmutableArrayRecord:
    """Copy/pickle immutable dataclasses through their authoritative constructors."""
    __slots__ = ()

    def __reduce__(self):
        return restore_immutable_record, (type(self), {
            item.name: thaw_record(getattr(self, item.name)) for item in fields(self) if item.init
        })


@dataclass(frozen=True, slots=True)
class CNFilterRecord:
    mutation_id: str
    sample_id: str
    segment_id: str
    reason: str
    max_major_cn: int


@dataclass(frozen=True, slots=True)
class CNFilterReport:
    """Pair-level exclusions; excluded_mutation_ids failed in every region."""

    policy_id: str
    max_major_cn: int
    input_mutation_count: int
    retained_mutation_count: int
    excluded_mutation_ids: tuple[str, ...]
    records: tuple[CNFilterRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))


@dataclass(frozen=True)
class TumorData(ImmutableArrayRecord):
    """Validated single-state numerical inputs; excluded entries are inert placeholders."""

    tumor_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    count_observed: np.ndarray | None = None
    cn_filter_report: CNFilterReport | None = None
    original_catalog: OriginalCatalog | None = None
    normal_cn_supplied: np.ndarray | None = None
    input_excluded: np.ndarray | None = None
    mapping_genuine: np.ndarray | None = None
    normal_cn_policy: str | None = None

    def __post_init__(self):
        for name in ("mutation_ids", "region_ids"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain unique nonempty strings")
            object.__setattr__(self, name, values)
        shape = len(self.mutation_ids), len(self.region_ids)
        for name in ("alt_counts", "total_counts", "purity", "major_cn", "minor_cn", "normal_cn"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite aligned numerical arrays")
            if name != "purity" and (np.any(value < 0) or np.any(value != np.rint(value))):
                raise ValueError(f"{name} must contain nonnegative integers")
            object.__setattr__(self, name, readonly_array(value, dtype=float))
        if np.any(self.major_cn < 1) or np.any(self.major_cn >= 2**63) or np.any(self.minor_cn > self.major_cn):
            raise ValueError("Single-state tumor CN requires integer 1 <= major and 0 <= minor <= major")
        if np.any(self.alt_counts > self.total_counts):
            raise ValueError("Alternate counts cannot exceed total counts")
        if np.any(self.purity <= 0) or np.any(self.purity > 1) or not np.all(self.purity == self.purity[:1]):
            raise ValueError("Purity must be canonical within regions and lie in (0,1]")
        for name in ("count_observed", "normal_cn_supplied", "mapping_genuine", "input_excluded"):
            value = getattr(self, name)
            value = np.full(shape, name != "input_excluded", dtype=bool) if value is None else np.asarray(value)
            if value.shape != shape or value.dtype.kind != "b":
                raise ValueError(f"{name} must contain aligned boolean flags")
            object.__setattr__(self, name, readonly_array(value))
        if self.normal_cn_policy not in (None, "diploid"):
            raise ValueError("normal_cn_policy must be None or 'diploid'")
        observed = self.count_observed & ~self.input_excluded & (self.total_counts > 0)
        if np.any(observed & ~self.mapping_genuine):
            raise ValueError("Scientific observations require genuine single-state CN mapping")
        assumed = observed & ~self.normal_cn_supplied
        if np.any(assumed & ((self.normal_cn != 2) | (self.normal_cn_policy != "diploid"))):
            raise ValueError("Scientific observations require supplied normal CN or explicit normal_cn_policy='diploid'")

    @property
    def total_cn(self):
        return self.major_cn + self.minor_cn

    @property
    def num_mutations(self):
        return len(self.mutation_ids)

    @property
    def num_regions(self):
        return len(self.region_ids)


def _indices(requested, available, name):
    requested = tuple(requested)
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate {name} identities")
    lookup = {value: index for index, value in enumerate(available)}
    missing = set(requested).difference(lookup)
    if missing:
        raise ValueError(f"Unknown {name} identities: {sorted(missing)}")
    return requested, np.asarray([lookup[value] for value in requested], dtype=int)


@dataclass(frozen=True, slots=True)
class MultiplicityBounds(ImmutableArrayRecord):
    """Keyed integer support endpoints; these never replace biological CN."""

    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    policy_id: str = "explicit"
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("mutation_ids", "region_ids"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"{name} must contain unique nonempty strings")
            object.__setattr__(self, name, values)
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("Multiplicity bounds require a nonempty policy identity")
        shape = (len(self.mutation_ids), len(self.region_ids))
        for name in ("lower", "upper"):
            values = np.asarray(getattr(self, name))
            if not values.size:
                values = values.reshape(shape)
            if (values.shape != shape or values.dtype.kind not in "iuf"
                    or not np.isfinite(values).all() or np.any(values != np.rint(values))
                    or np.any(values < 1) or np.any(values >= 2**63)):
                raise ValueError("Multiplicity bounds must be aligned positive int64 integers")
            object.__setattr__(self, name, readonly_array(values, dtype=np.int64))
        if np.any(self.lower > self.upper):
            raise ValueError("Multiplicity lower endpoints may not exceed upper endpoints")
        object.__setattr__(self, "_content_hash", self._fingerprint())

    def _fingerprint(self):
        digest = hashlib.sha256(b"clipp2.multiplicity-bounds.v1\0")
        digest.update(json.dumps([self.mutation_ids, self.region_ids, self.policy_id],
                                 ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(self.lower.astype("<i8", copy=False).tobytes())
        digest.update(self.upper.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()

    @property
    def content_hash(self):
        return self._content_hash

    def subset(self, mutation_ids, region_ids=None):
        mutations, rows = _indices(mutation_ids, self.mutation_ids, "mutation")
        regions, cols = _indices(self.region_ids if region_ids is None else region_ids,
                                self.region_ids, "region")
        ix = np.ix_(rows, cols)
        return type(self)(mutations, regions, self.lower[ix], self.upper[ix], self.policy_id)

    def for_data(self, data):
        """Validate original support, then return aligned numerical endpoints.

        Original excluded pairs stay excluded and receive inert [1,1] support.
        Their biological CN and requested endpoints remain in the catalog/record.
        """
        if not isinstance(data, (TumorData, OriginalCatalog)):
            raise TypeError("Multiplicity bounds require validated TumorData or OriginalCatalog")
        aligned = self.subset(data.mutation_ids, data.region_ids)
        catalog = data if isinstance(data, OriginalCatalog) else getattr(data, "original_catalog", None)
        if catalog is None:
            major = np.asarray(data.major_cn)
            eligible = ~np.asarray(data.input_excluded, dtype=bool)
            report = getattr(data, "cn_filter_report", None)
            if report is not None and report.records:
                # Placeholder CN cannot establish original support eligibility.
                raise ValueError("Bounded excluded inputs require their original catalog")
        else:
            _, rows = _indices(data.mutation_ids, catalog.mutation_ids, "mutation")
            _, cols = _indices(data.region_ids, catalog.region_ids, "region")
            ix = np.ix_(rows, cols)
            major, eligible = catalog.major_cn[ix], catalog.eligible[ix]
        if np.any(aligned.upper > major):
            raise ValueError("Multiplicity upper endpoints exceed original major CN")
        return (readonly_array(np.where(eligible, aligned.lower, 1), dtype=np.int64),
                readonly_array(np.where(eligible, aligned.upper, 1), dtype=np.int64))

    def to_dict(self):
        return {"schema": "clipp2.multiplicity-bounds.v1", "mutation_ids": list(self.mutation_ids),
                "region_ids": list(self.region_ids), "lower": self.lower.tolist(),
                "upper": self.upper.tolist(), "policy_id": self.policy_id,
                "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != "clipp2.multiplicity-bounds.v1":
            raise ValueError("Unsupported multiplicity-bounds schema")
        value = cls(*(payload[name] for name in ("mutation_ids", "region_ids", "lower", "upper", "policy_id")))
        if payload.get("content_hash") != value.content_hash:
            raise ValueError("Multiplicity-bounds content hash mismatch")
        return value


_CATALOG_ARRAYS = ("alt_counts", "total_counts", "count_observed", "eligible", "major_cn",
                   "minor_cn", "normal_cn", "purity",
                   "segment_ids", "mapping_genuine", "normal_cn_supplied", "input_excluded")


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True, slots=True)
class OriginalCatalog(ImmutableArrayRecord):
    """All original keys and biological values, before numerical placeholders.

    ``source_rows`` and ``provenance`` are tuples of key/value tuples, retaining
    each canonical observation row and its otherwise inert extra columns.
    """

    tumor_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    count_observed: np.ndarray
    eligible: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    purity: np.ndarray
    segment_ids: np.ndarray
    mapping_genuine: np.ndarray
    normal_cn_supplied: np.ndarray
    source_rows: tuple
    provenance: tuple
    metadata: tuple
    max_major_cn: int = 10
    input_excluded: np.ndarray | None = None
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from .tumor_txt import _identifier, _safe_tumor_id

        object.__setattr__(self, "tumor_id", _safe_tumor_id(self.tumor_id))
        for name in ("mutation_ids", "region_ids"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"Catalog {name} must be unique nonempty strings")
            for value in values:
                _identifier(value, name=name)
            object.__setattr__(self, name, values)
        shape = (len(self.mutation_ids), len(self.region_ids))
        if self.input_excluded is None:
            object.__setattr__(self, "input_excluded", np.zeros(shape, dtype=bool))
        for name in _CATALOG_ARRAYS:
            dtype = (bool if name in ("count_observed", "eligible", "mapping_genuine", "normal_cn_supplied", "input_excluded") else
                     str if name == "segment_ids" else float)
            values = np.asarray(getattr(self, name))
            if not values.size:
                values = values.reshape(shape)
            if values.shape != shape:
                raise ValueError(f"Catalog {name} must have mutation-by-region shape")
            if values.size and dtype is bool and values.dtype.kind != "b":
                raise ValueError(f"Catalog {name} must contain boolean flags")
            object.__setattr__(self, name, readonly_array(values, dtype=dtype))
        for name in ("source_rows", "provenance", "metadata"):
            object.__setattr__(self, name, _tuples(getattr(self, name)))
        for record in (self.metadata, *self.source_rows, *self.provenance):
            if (not isinstance(record, tuple) or any(not isinstance(pair, tuple) or len(pair) != 2
                    or not all(isinstance(value, str) for value in pair) for pair in record)
                    or len({pair[0] for pair in record}) != len(record)):
                raise ValueError("Catalog source/provenance records require immutable string key/value pairs")
        if len(self.source_rows) != len(self.provenance):
            raise ValueError("Catalog provenance must align with every source row")
        if dict(self.metadata).get("tumor_id") != self.tumor_id:
            raise ValueError("Catalog tumor identity disagrees with original metadata")
        source_keys = set()
        for source_row in self.source_rows:
            source = dict(source_row)
            if "mutation_id" not in source or "sample_id" not in source:
                raise ValueError("Catalog source rows require original mutation-region keys")
            source_keys.add((source["mutation_id"], source["sample_id"]))
        if source_keys != {(mutation, region) for mutation in self.mutation_ids for region in self.region_ids}:
            raise ValueError("Catalog source rows do not cover exactly the original keys")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if not np.isfinite(self.purity).all() or np.any(self.purity <= 0) or np.any(self.purity > 1):
            raise ValueError("Catalog purity must lie in (0,1]")
        if len(self.mutation_ids) and not np.all(self.purity == self.purity[:1]):
            raise ValueError("Catalog purity must be canonical within regions")
        for name in ("alt_counts", "total_counts", "major_cn", "minor_cn", "normal_cn"):
            values = getattr(self, name)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Catalog {name} must be finite and nonnegative")
            if np.any(values != np.rint(values)):
                raise ValueError(f"Catalog {name} must be integer-valued")
        if np.any(self.alt_counts > self.total_counts):
            raise ValueError("Catalog alternate counts exceed total counts")
        if np.any(self.major_cn < 1) or np.any(self.major_cn >= 2**63) or np.any(self.minor_cn > self.major_cn):
            raise ValueError("Single-state tumor CN requires integer 1 <= major and 0 <= minor <= major")
        if not np.array_equal(self.eligible, (self.major_cn <= self.max_major_cn) & ~self.input_excluded):
            raise ValueError("Catalog eligibility disagrees with original CN or explicit exclusion")
        from .tumor_txt import _normalize_row
        positions = {(mutation, region): (i, j) for i, mutation in enumerate(self.mutation_ids)
                     for j, region in enumerate(self.region_ids)}
        segments, seen = {}, set()
        for index, source in enumerate(self.source_rows):
            row = _normalize_row(dict(source), row_number=index + 1)
            unit = row["mutation_id"], row["sample_id"]
            if unit in seen:
                raise ValueError("Catalog requires one canonical observation per mutation-region pair")
            seen.add(unit)
            i, j = positions[unit]
            for name in ("mapping_genuine", "normal_cn_supplied", "input_excluded"):
                if getattr(self, name)[i, j] != row[name]:
                    raise ValueError(f"Catalog {name} flags disagree with original provenance")
            pair = row["allele_a_cn"], row["allele_b_cn"]
            segment = (j, row["segment_id"])
            if segments.setdefault(segment, pair) != pair:
                raise ValueError("Catalog region-segment CN pairs must agree across mutations")
            if (pair != (self.major_cn[i, j], self.minor_cn[i, j])
                    or row["normal_cn"] != self.normal_cn[i, j]
                    or abs(row["purity"] - self.purity[i, j]) > PURITY_SOURCE_CANONICAL_ATOL
                    or row["segment_id"] != self.segment_ids[i, j]
                    or row["count_observed"] != self.count_observed[i, j]
                    or (row["alt_count"] or 0) != self.alt_counts[i, j]
                    or (row["alt_count"] or 0)+(row["ref_count"] or 0) != self.total_counts[i, j]):
                raise ValueError("Catalog biological arrays disagree with canonical source observations")
        object.__setattr__(self, "_content_hash", hashlib.sha256(json.dumps(self._payload(), sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def _payload(self):
        payload = {"schema": "clipp2.original-catalog.v4", "tumor_id": self.tumor_id,
                   "mutation_ids": list(self.mutation_ids), "region_ids": list(self.region_ids),
                   "max_major_cn": self.max_major_cn}
        payload.update((name, getattr(self, name).tolist()) for name in _CATALOG_ARRAYS)
        # Omit the redundant all-false mask from the canonical payload.
        if not self.input_excluded.any():
            payload.pop("input_excluded")
        payload.update((name, getattr(self, name)) for name in ("source_rows", "provenance", "metadata"))
        return payload

    @property
    def content_hash(self):
        return self._content_hash

    @property
    def biological_identity(self):
        """Declared biological namespace, independent of file bytes and numerical fit."""
        metadata = dict(self.metadata)
        identity = dict(schema="clipp2.biological-identity.v1", tumor_id=self.tumor_id,
            region_mapping=[dict(region_id=region, source_sample_id=region) for region in self.region_ids],
            mutation_key_convention=metadata.get("mutation_key_convention", "opaque_mutation_id"),
            genome_build=metadata.get("genome_build", "unknown"),
            coordinate_system=metadata.get("coordinate_system", "1-based-inclusive"))
        for name in ("reference_build", "region_mapping"):
            if name in metadata:
                identity["declared_" + name] = metadata[name]
        return identity

    def to_dict(self):
        return dict(self._payload(), content_hash=self.content_hash)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != "clipp2.original-catalog.v4":
            raise ValueError("Expected single-state original-catalog.v4; historical catalog schemas require their pinned revision")
        value = cls(**{item.name: payload[item.name] for item in fields(cls) if item.init and item.name in payload})
        if payload.get("content_hash") != value.content_hash:
            raise ValueError("Original-catalog content hash mismatch")
        return value

    @property
    def total_cn(self):
        return self.major_cn + self.minor_cn

    def fit_data(self, mutation_ids=None, *, normal_cn_policy=None):
        from .tumor_txt import _catalog_data
        return _catalog_data(self, self.mutation_ids if mutation_ids is None else mutation_ids, prediction=False, normal_cn_policy=normal_cn_policy)

    def prediction_data(self, mutation_ids, *, normal_cn_policy=None):
        from .tumor_txt import _catalog_data
        return _catalog_data(self, mutation_ids, prediction=True, normal_cn_policy=normal_cn_policy)

    def subset(self, mutation_ids, region_ids=None):
        """Select keys in requested order without re-canonicalizing purity."""
        mutations, rows = _indices(mutation_ids, self.mutation_ids, "mutation")
        regions, cols = _indices(self.region_ids if region_ids is None else region_ids,
                                self.region_ids, "region")
        ix = np.ix_(rows, cols)
        mutation_set, region_set = set(mutations), set(regions)
        selected = [(row, extra) for row, extra in zip(self.source_rows, self.provenance)
                    if dict(row)["mutation_id"] in mutation_set and dict(row)["sample_id"] in region_set]
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.init}
        values.update((name, getattr(self, name)[ix]) for name in _CATALOG_ARRAYS)
        values.update(mutation_ids=mutations, region_ids=regions,
                      source_rows=tuple(row for row, _ in selected), provenance=tuple(extra for _, extra in selected))
        return type(self)(**values)

    def default_bounds(self):
        upper = self.major_cn
        return MultiplicityBounds(self.mutation_ids, self.region_ids, np.ones_like(upper), upper, "original")


def validate_workflow_mapping(catalog, mutation_ids=None, *, normal_cn_policy=None):
    """Require genuine single-state CN wherever a strict workflow uses evidence.

    Unobserved single-state cells have no likelihood or restricted CCF domain;
    fully excluded/uninformative rows can remain for original-population accounting.
    """
    if not isinstance(catalog, OriginalCatalog):
        raise TypeError("Strict workflow mapping requires an OriginalCatalog")
    if normal_cn_policy not in (None, "diploid"):
        raise ValueError("normal_cn_policy must be None or 'diploid'")
    ids, rows = _indices(catalog.mutation_ids if mutation_ids is None else mutation_ids,
                        catalog.mutation_ids, "mutation")
    eligible = catalog.eligible[rows]
    observed = eligible & catalog.count_observed[rows] & (catalog.total_counts[rows] > 0)
    nongenuine = observed & ~catalog.mapping_genuine[rows]
    if np.any(nongenuine):
        examples = [(ids[i], catalog.region_ids[j]) for i, j in np.argwhere(nongenuine)[:5]]
        raise ValueError("Strict workflow evidence requires genuine, unambiguous single-state CN; "
                         f"invalid mutation-region mappings: {examples}")
    assumed = observed & ~catalog.normal_cn_supplied[rows]
    invalid_normal = assumed if normal_cn_policy is None else assumed & (catalog.normal_cn[rows] != 2)
    if np.any(invalid_normal):
        examples = [(ids[i], catalog.region_ids[j]) for i, j in np.argwhere(invalid_normal)[:5]]
        raise ValueError("Strict workflow evidence requires supplied normal CN or an explicit matching "
                         f"normal_cn_policy='diploid'; invalid mutation-region mappings: {examples}")


__all__ = ["CNFilterRecord", "CNFilterReport", "TumorData", "OriginalCatalog", "MultiplicityBounds",
           "freeze_record", "thaw_record", "json_value", "content_hash", "validate_workflow_mapping"]
