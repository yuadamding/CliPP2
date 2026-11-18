"""Validated provenance for original segment CN-state fractions.

These are fractions of all tumor cells, including cells without a mutation.
The likelihood uses a representative original CN state, not these averages.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math

import numpy as np

from .tumor_txt import TumorTxtError, _finite_float, _identifier


CN_STATE_COLUMNS = ("sample_id", "segment_id", "cn_state_id", "cn_state_fraction",
                    "major_cn", "minor_cn")
CN_STATE_FRACTION_ATOL = 1e-8


def _record(row, number):
    if isinstance(row, Mapping):
        missing = set(CN_STATE_COLUMNS).difference(row)
        if missing:
            raise TumorTxtError(f"CN-state row {number} is missing columns: {sorted(missing)}.")
        values = tuple(row[name] for name in CN_STATE_COLUMNS)
    else:
        if isinstance(row, (str, bytes)):
            raise TumorTxtError(f"CN-state row {number} must contain {len(CN_STATE_COLUMNS)} fields.")
        try:
            values = tuple(row)
        except TypeError as exc:
            raise TumorTxtError(f"CN-state row {number} must contain fields.") from exc
        if len(values) != len(CN_STATE_COLUMNS):
            raise TumorTxtError(f"CN-state row {number} must contain {len(CN_STATE_COLUMNS)} fields.")
    identifiers = []
    for name, value in zip(CN_STATE_COLUMNS[:3], values[:3]):
        if not isinstance(value, str):
            raise TumorTxtError(f"CN-state row {number} {name} must be a nonempty string.")
        identifiers.append(_identifier(value, name=f"CN-state row {number} {name}"))
    numeric = []
    for name, value in zip(CN_STATE_COLUMNS[3:], values[3:]):
        if isinstance(value, (bool, np.bool_)):
            raise TumorTxtError(f"CN-state row {number} {name} must be numeric, not boolean.")
        numeric.append(_finite_float(value, name=f"CN-state row {number} {name}"))
    fraction, major, minor = numeric
    if not 0 <= fraction <= 1:
        raise TumorTxtError(f"CN-state row {number} cn_state_fraction must lie in [0, 1].")
    if (major != math.floor(major) or minor != math.floor(minor)
            or not 0 <= major < 2**63 or not 0 <= minor <= major):
        raise TumorTxtError(f"CN-state row {number} requires integer 0 <= major_cn < 2**63 "
                            "and 0 <= minor_cn <= major_cn.")
    return (*identifiers, fraction, int(major), int(minor))


@dataclass(frozen=True, slots=True)
class SegmentCNStates:
    """Canonical immutable records, keyed by sample, segment, and CN-state ID.

    Fraction sums must equal one within ``CN_STATE_FRACTION_ATOL``. Fractions
    are preserved rather than silently normalized. Row order does not affect
    identity. Zero-fraction states are valid and count as declared states.
    """

    records: tuple

    def __post_init__(self):
        records = tuple(sorted(_record(row, number) for number, row in enumerate(self.records, 1)))
        if not records:
            raise TumorTxtError("At least one segment CN state must be supplied.")
        fractions, keys = {}, set()
        for sample, segment, state, fraction, _, _ in records:
            key = sample, segment, state
            if key in keys:
                raise TumorTxtError(f"Duplicate CN-state key: {key!r}.")
            keys.add(key)
            fractions.setdefault((sample, segment), []).append(fraction)
        for key, values in fractions.items():
            total = math.fsum(values)
            if not math.isclose(total, 1., rel_tol=0., abs_tol=CN_STATE_FRACTION_ATOL):
                raise TumorTxtError(f"CN-state fractions for sample/segment {key!r} must sum to 1; "
                                    f"observed {total:.17g}.")
        object.__setattr__(self, "records", records)

def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TumorTxtError(f"Duplicate field in segment_cn_states JSON: {key!r}.")
        result[key] = value
    return result


def parse_segment_cn_states(value, *, sample_id, segment_id):
    """Validate one canonical JSON cell; an absent cell denotes one-state CN."""
    if value is None or value == ".":
        return None
    if not isinstance(value, str) or not value:
        raise TumorTxtError("segment_cn_states must be a JSON list of CN-state objects or '.'.")
    return _parse_segment_cn_states_json(value, sample_id, segment_id)


@lru_cache(maxsize=4096)
def _parse_segment_cn_states_json(value, sample_id, segment_id):
    # Mutation rows repeat segment descriptions. Immutable values can safely be
    # shared, while a bounded cache avoids retaining an entire cohort's inputs.
    try:
        states = json.loads(value, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise TumorTxtError("segment_cn_states must contain valid JSON.") from exc
    fields = set(CN_STATE_COLUMNS[2:])
    if (not isinstance(states, list) or not states
            or any(not isinstance(state, dict) or set(state) != fields for state in states)):
        raise TumorTxtError("segment_cn_states must be a nonempty JSON list with cn_state_id, "
                            "cn_state_fraction, major_cn and minor_cn for every state.")
    return SegmentCNStates(tuple(dict(sample_id=sample_id, segment_id=segment_id, **state)
                                 for state in states))
