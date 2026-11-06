"""Immutable retained biological inputs and their CN exclusion provenance."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import hashlib
import json
from types import MappingProxyType

import numpy as np

from ..config import MAX_MULTIPLICITY, validate_max_major_cn


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
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        })


@dataclass(frozen=True, slots=True)
class CNFilterRecord:
    mutation_id: str
    sample_id: str
    segment_id: str
    reason: str
    n_distinct_cn_states: int
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
    """CN-compiled inputs, with no CN-population state in the optimizer.

    major_cn/minor_cn are per-allele maxima; mean_total_cn is the actual
    fraction-weighted total. Only clonal entries have a single CN pair.
    Entries identified by cn_filter_report.records are masked placeholders,
    not biological CN observations or multiplicity evaluation targets.
    """

    tumor_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    scaling: np.ndarray
    phi_upper: np.ndarray
    phi_init: np.ndarray
    count_observed: np.ndarray | None = None
    cn_filter_report: CNFilterReport | None = None
    mean_total_cn: np.ndarray | None = None
    cn_state_count: np.ndarray | None = None
    original_catalog: OriginalCatalog | None = None

    def __post_init__(self) -> None:
        for name in ("mutation_ids", "region_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.mean_total_cn is None:
            object.__setattr__(self, "mean_total_cn", np.asarray(self.major_cn) + np.asarray(self.minor_cn))
        if self.cn_state_count is None:
            object.__setattr__(self, "cn_state_count", np.ones_like(self.major_cn, dtype=np.int64))
        for name in (
            "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
            "normal_cn", "scaling", "phi_upper", "phi_init", "count_observed",
            "mean_total_cn", "cn_state_count",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value))

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_regions(self) -> int:
        return int(self.alt_counts.shape[1])


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
        aligned = self.subset(data.mutation_ids, data.region_ids)
        catalog = data if isinstance(data, OriginalCatalog) else getattr(data, "original_catalog", None)
        if catalog is None:
            major, mixed = np.asarray(data.major_cn), np.asarray(data.cn_state_count) > 1
            eligible = np.ones(major.shape, dtype=bool)
            report = getattr(data, "cn_filter_report", None)
            if report is not None and report.records:
                # Placeholder CN cannot establish original support eligibility.
                raise ValueError("Bounded excluded inputs require their original catalog")
        else:
            _, rows = _indices(data.mutation_ids, catalog.mutation_ids, "mutation")
            _, cols = _indices(data.region_ids, catalog.region_ids, "region")
            ix = np.ix_(rows, cols)
            major, mixed, eligible = catalog.major_cn[ix], catalog.cn_state_count[ix] > 1, catalog.eligible[ix]
        if np.any(aligned.upper > major):
            raise ValueError("Multiplicity upper endpoints exceed original major CN")
        default_upper = np.where(mixed, np.minimum(major, MAX_MULTIPLICITY), major)
        if np.any(mixed & ((aligned.lower != 1) | (aligned.upper != default_upper))):
            raise ValueError("Nondefault multiplicity bounds on mixed-CN pairs are unsupported")
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
                   "minor_cn", "normal_cn", "mean_total_cn", "cn_state_count", "purity",
                   "segment_ids", "mapping_genuine")


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True, slots=True)
class OriginalCatalog(ImmutableArrayRecord):
    """All original keys and biological values, before numerical placeholders.

    ``source_rows`` and ``provenance`` are tuples of key/value tuples, retaining
    each source CN-state row and its otherwise inert extra columns. ``cn_states``
    is mutation × region × state, with (fraction, major, minor) state tuples.
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
    mean_total_cn: np.ndarray
    cn_state_count: np.ndarray
    purity: np.ndarray
    segment_ids: np.ndarray
    mapping_genuine: np.ndarray
    cn_states: tuple
    source_rows: tuple
    provenance: tuple
    metadata: tuple
    max_major_cn: int = 10
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        from .tumor_txt import _identifier, _mapping_provenance, _safe_tumor_id

        object.__setattr__(self, "tumor_id", _safe_tumor_id(self.tumor_id))
        for name in ("mutation_ids", "region_ids"):
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values) or any(not isinstance(x, str) or not x for x in values):
                raise ValueError(f"Catalog {name} must be unique nonempty strings")
            for value in values:
                _identifier(value, name=name)
            object.__setattr__(self, name, values)
        shape = (len(self.mutation_ids), len(self.region_ids))
        for name in _CATALOG_ARRAYS:
            dtype = (bool if name in ("count_observed", "eligible", "mapping_genuine") else
                     str if name == "segment_ids" else np.int64 if name == "cn_state_count" else float)
            values = np.asarray(getattr(self, name))
            if not values.size:
                values = values.reshape(shape)
            if values.shape != shape:
                raise ValueError(f"Catalog {name} must have mutation-by-region shape")
            if values.size and dtype is bool and values.dtype.kind != "b":
                raise ValueError(f"Catalog {name} must contain boolean flags")
            if name == "cn_state_count" and (values.dtype.kind not in "iuf"
                    or not np.isfinite(values).all() or np.any(values != np.rint(values))
                    or np.any(values < 1) or np.any(values >= 2**63)):
                raise ValueError("Catalog CN-state counts must be positive int64 integers")
            object.__setattr__(self, name, readonly_array(values, dtype=dtype))
        for name in ("cn_states", "source_rows", "provenance", "metadata"):
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
        source_mapping = _mapping_provenance([dict(row) for row in self.source_rows])
        if any(self.mapping_genuine[i, j] != source_mapping[mutation, region]
               for i, mutation in enumerate(self.mutation_ids) for j, region in enumerate(self.region_ids)):
            raise ValueError("Catalog genuine-mapping flags disagree with original provenance")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if not np.isfinite(self.purity).all() or np.any(self.purity <= 0) or np.any(self.purity > 1):
            raise ValueError("Catalog purity must lie in (0,1]")
        if len(self.mutation_ids) and not np.all(self.purity == self.purity[:1]):
            raise ValueError("Catalog purity must be canonical within regions")
        if len(self.cn_states) != shape[0] or any(len(row) != shape[1] for row in self.cn_states):
            raise ValueError("Catalog CN states must align with every original key")
        for name in ("alt_counts", "total_counts", "major_cn", "minor_cn", "normal_cn", "mean_total_cn"):
            values = getattr(self, name)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Catalog {name} must be finite and nonnegative")
            if name != "mean_total_cn" and np.any(values != np.rint(values)):
                raise ValueError(f"Catalog {name} must be integer-valued")
        if np.any(self.alt_counts > self.total_counts):
            raise ValueError("Catalog alternate counts exceed total counts")
        segment_states = {}
        for i, row in enumerate(self.cn_states):
            for j, states in enumerate(row):
                segment = _identifier(self.segment_ids[i, j], name="segment_id")
                if segment_states.setdefault((j, segment), states) != states:
                    raise ValueError("Catalog region-segment CN states must agree across mutations")
                if not states or any(len(state) != 3 for state in states):
                    raise ValueError("Every original pair requires CN-state definitions")
                fractions = np.asarray([state[0] for state in states], dtype=float)
                major = np.asarray([state[1] for state in states], dtype=float)
                minor = np.asarray([state[2] for state in states], dtype=float)
                if (not np.isfinite(fractions).all() or np.any(fractions <= 0)
                        or not np.isclose(fractions.sum(), 1., atol=1e-8, rtol=0)
                        or not np.isfinite(major).all() or not np.isfinite(minor).all()
                        or np.any(major < minor) or np.any(minor < 0)
                        or np.any(major != np.rint(major)) or np.any(minor != np.rint(minor))):
                    raise ValueError("Invalid original CN-state definition")
                maximum = float(major.max())
                if maximum < 1:
                    raise ValueError("Original (0,0) CN has no positive mutant-copy path")
                total = (float(major[0]+minor[0]) if len(states) == 1 else
                         sum(float(f)*(float(a)+float(b)) for f, a, b in states)/sum(float(f) for f, _, _ in states))
                if (self.major_cn[i, j] != maximum or self.minor_cn[i, j] != minor.max()
                        or self.cn_state_count[i, j] != len(states)
                        or not np.isclose(self.mean_total_cn[i, j], total, atol=1e-12, rtol=1e-12)
                        or self.eligible[i, j] != (maximum <= self.max_major_cn)):
                    raise ValueError("Catalog biology or eligibility disagrees with original CN states")
        object.__setattr__(self, "_content_hash", hashlib.sha256(json.dumps(self._payload(), sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest())

    def _payload(self):
        payload = {"schema": "clipp2.original-catalog.v2", "tumor_id": self.tumor_id,
                   "mutation_ids": list(self.mutation_ids), "region_ids": list(self.region_ids),
                   "max_major_cn": self.max_major_cn}
        payload.update((name, getattr(self, name).tolist()) for name in _CATALOG_ARRAYS)
        payload.update((name, getattr(self, name)) for name in ("cn_states", "source_rows", "provenance", "metadata"))
        return payload

    @property
    def content_hash(self):
        return self._content_hash

    def to_dict(self):
        return dict(self._payload(), content_hash=self.content_hash)

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema") != "clipp2.original-catalog.v2":
            raise ValueError("Unsupported original-catalog schema")
        value = cls(**{item.name: payload[item.name] for item in fields(cls) if item.init})
        if payload.get("content_hash") != value.content_hash:
            raise ValueError("Original-catalog content hash mismatch")
        return value

    def fit_data(self, mutation_ids=None):
        from .tumor_txt import _catalog_data
        return _catalog_data(self, self.mutation_ids if mutation_ids is None else mutation_ids, prediction=False)

    def prediction_data(self, mutation_ids):
        from .tumor_txt import _catalog_data
        return _catalog_data(self, mutation_ids, prediction=True)

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
                      source_rows=tuple(row for row, _ in selected), provenance=tuple(extra for _, extra in selected),
                      cn_states=tuple(tuple(self.cn_states[i][j] for j in cols) for i in rows))
        return type(self)(**values)

    def default_bounds(self):
        upper = np.where(self.cn_state_count > 1, np.minimum(self.major_cn, MAX_MULTIPLICITY), self.major_cn)
        return MultiplicityBounds(self.mutation_ids, self.region_ids, np.ones_like(upper), upper, "original")


def validate_workflow_mapping(catalog, mutation_ids=None):
    """Require genuine single-state CN wherever a strict workflow uses evidence.

    Unobserved single-state cells have no likelihood or restricted CCF domain;
    fully excluded/uninformative rows can remain for original-population accounting.
    Mixed cells in otherwise informative rows impose CCF bounds even without reads.
    """
    if not isinstance(catalog, OriginalCatalog):
        raise TypeError("Strict workflow mapping requires an OriginalCatalog")
    ids, rows = _indices(catalog.mutation_ids if mutation_ids is None else mutation_ids,
                        catalog.mutation_ids, "mutation")
    eligible = catalog.eligible[rows]
    observed = eligible & catalog.count_observed[rows] & (catalog.total_counts[rows] > 0)
    nongenuine = observed & ~catalog.mapping_genuine[rows]
    mixed = eligible & observed.any(axis=1)[:, None] & (catalog.cn_state_count[rows] > 1)
    if np.any(nongenuine | mixed):
        examples = [(ids[i], catalog.region_ids[j]) for i, j in np.argwhere(nongenuine | mixed)[:5]]
        raise ValueError("Strict workflow evidence requires genuine, unambiguous single-state CN; "
                         f"invalid mutation-region mappings: {examples}")


__all__ = ["CNFilterRecord", "CNFilterReport", "TumorData", "OriginalCatalog", "MultiplicityBounds",
           "freeze_record", "json_value", "content_hash", "validate_workflow_mapping"]
