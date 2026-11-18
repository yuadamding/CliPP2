"""Strict validation of optional legacy segment CN-state metadata."""

from functools import lru_cache
import json

import numpy as np

from .tumor_txt import TumorTxtError, _finite_float, _identifier


_FIELDS = ("cn_state_id", "cn_state_fraction", "major_cn", "minor_cn")


def single_cn_state(rows, *, sample_id, segment_id):
    """Return the sole active CN pair; zero-fraction records are inert."""
    context = f"sample/segment {(sample_id, segment_id)!r}"
    seen, active = set(), None
    for state_id, fraction, major, minor in rows:
        if not isinstance(state_id, str):
            raise TumorTxtError(f"{context} CN-state ID must be a string.")
        state_id = _identifier(state_id, name=f"{context} CN-state ID")
        if state_id in seen:
            raise TumorTxtError(f"{context} duplicate CN-state ID {state_id!r}.")
        seen.add(state_id)
        if any(isinstance(value, (bool, np.bool_)) for value in (fraction, major, minor)):
            raise TumorTxtError(f"{context} CN-state values must be numeric, not boolean.")
        fraction, major, minor = (
            _finite_float(value, name=f"{context} {name}")
            for name, value in zip(_FIELDS[1:], (fraction, major, minor))
        )
        if (not 0 <= minor <= major < 2**63
                or not major.is_integer() or not minor.is_integer()):
            raise TumorTxtError(f"{context} requires integer CN with 0 <= minor <= major < 2**63.")
        if fraction not in (0., 1.) or (fraction == 1. and active is not None):
            raise TumorTxtError(f"{context} requires exactly one active CN state of fraction 1; "
                                "mixed CN states are not supported.")
        if fraction:
            active = int(major), int(minor)
    if active is None or active[0] == 0:
        raise TumorTxtError(f"{context} requires one active CN state of fraction 1 and major CN >= 1.")
    return active


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TumorTxtError(f"Duplicate field in segment_cn_states JSON: {key!r}.")
        result[key] = value
    return result


def parse_segment_cn_states(value, *, sample_id, segment_id):
    """Check optional JSON, returning the only positive-fraction state's CN pair."""
    if value is None or value == ".":
        return None
    if not isinstance(value, str) or not value:
        raise TumorTxtError("segment_cn_states must be a JSON list of CN-state objects or '.'.")
    return _parse_segment_cn_states_json(value, sample_id, segment_id)


@lru_cache(maxsize=4096)
def _parse_segment_cn_states_json(value, sample_id, segment_id):
    try:
        states = json.loads(value, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise TumorTxtError("segment_cn_states must contain valid JSON.") from exc
    if (not isinstance(states, list) or not states
            or any(not isinstance(state, dict) or set(state) != set(_FIELDS) for state in states)):
        raise TumorTxtError("segment_cn_states requires cn_state_id, cn_state_fraction, major_cn and minor_cn "
                            "for every state.")
    return single_cn_state((tuple(state[k] for k in _FIELDS) for state in states),
                           sample_id=sample_id, segment_id=segment_id)
