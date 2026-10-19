"""Source-bound, non-executable JSON checkpoints for cumulative bounded search.

The file is an operational artifact, not one of the public scientific TSVs.
No pickle, dynamic imports, or arbitrary class restoration is permitted.
Snapshots are atomic; the prior committed checkpoint remains valid after an
interrupted write. Completed candidates survive beam eviction and continuation.
"""

import base64
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA = "clipp2_bounded_search_checkpoint_v1"


class SearchPaused(RuntimeError):
    """Explicit requested callback budget reached after a durable checkpoint."""


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Checkpoint object arrays are forbidden")
        return {"array": base64.b64encode(value.tobytes()).decode(),
                "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode()}
    if isinstance(value, tuple):
        return {"tuple": [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {"mapping": [[encode(key), encode(item)] for key, item in value.items()]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"array", "dtype", "shape"}:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Checkpoint object arrays are forbidden")
        return np.frombuffer(base64.b64decode(value["array"], validate=True), dtype=dtype).reshape(
            value["shape"]).copy()
    if set(value) == {"bytes"}:
        return base64.b64decode(value["bytes"], validate=True)
    if set(value) == {"tuple"}:
        return tuple(decode(item) for item in value["tuple"])
    if set(value) == {"mapping"}:
        return {decode(key): decode(item) for key, item in value["mapping"]}
    if set(value) == {"float"} and value["float"] in ("nan", "inf", "-inf"):
        return float(value["float"])
    raise ValueError("Invalid checkpoint JSON encoding")


def source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if (path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
                and path.suffix in (".py", ".cpp", ".h", ".hpp")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def identities(model, tree=None, pilot=None):
    result = {"source": source_identity(), "model": model.identity,
              "objective": getattr(model, "selection_objective", "legacy_mixture"),
              "zero_policy": getattr(model, "regional_zero_policy", "likelihood"),
              "mutation_ids": tuple(model.mutation_ids),
              "device": getattr(model, "device", "cpu"),
              "execution_binding": getattr(model, "checkpoint_identity", None)}
    if tree is not None:
        result["tree"] = tree.identity
    if pilot is not None:
        result["pilot"] = hashlib.sha256(np.asarray(pilot, dtype="<f8").tobytes()).hexdigest()
    return result


def write_checkpoint(path, state, *, schema=SCHEMA):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = encode(state)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    document = {"schema": schema, "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "payload": payload}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_checkpoint(path, model, tree=None, pilot=None):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError("Unsupported search checkpoint schema")
    encoded = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    if hashlib.sha256(encoded.encode()).hexdigest() != document.get("sha256"):
        raise ValueError("Search checkpoint checksum mismatch")
    state = decode(document["payload"])
    expected = identities(model, tree, pilot)
    for name, identity in expected.items():
        if state["identities"].get(name) != identity:
            raise ValueError(f"Search checkpoint {name} identity changed; explicit new proposal/refit run required")
    return state


def snapshot_bank(bank):
    from .selection import _TreeRefit
    index = {id(candidate): i for i, candidate in enumerate(bank.candidates)}
    candidates = []
    for candidate in bank.candidates:
        fit = candidate.refit
        values = {field.name: getattr(fit, field.name) for field in fields(fit)
                  if field.name not in ("tree", "partition_cache")}
        candidates.append({"cuts": candidate.cuts, "membership_kind": candidate.membership_kind,
                           "refit_kind": "tree" if isinstance(fit, _TreeRefit) else "dense",
                           "provenance": candidate.provenance, "refit": values})
    maps = {}
    for name in ("by_cuts", "by_membership", "membership_attempts", "tree_attempts", "fitted_states"):
        maps[name] = {key: index[id(value)] if value is not None else None
                      for key, value in getattr(bank, name).items()}
    plain = {name: value for name, value in vars(bank).items()
             if name not in {"model", "tree", "partition_cache", "candidates", "original_selected", *maps}}
    return {"candidates": candidates, "maps": maps, "plain": plain,
            "original_selected_index": (index[id(bank.original_selected)]
                                        if hasattr(bank, "original_selected") else None)}


def restore_bank(model, tree, state):
    from .selection import Candidate, _CandidateBank, _TreeRefit
    from ..kernel.model import RefittedPartition
    from ..io.data import readonly_array
    bank = _CandidateBank(model, tree, compact_general=True)
    for record in state["candidates"]:
        values = dict(record["refit"])
        for name in ("centers", "weights"):
            values[name] = readonly_array(values[name])
        kind = record.get("refit_kind", "tree")  # Original bounded schema is compact.
        if kind == "tree":
            fitted = _TreeRefit(tree=tree, partition_cache=bank.partition_cache, **values)
        elif kind == "dense":
            values["labels"] = readonly_array(values["labels"])
            fitted = RefittedPartition(**values)
        else:
            raise ValueError("Unsupported saved refit representation")
        bank.candidates.append(Candidate(record["cuts"], fitted, record["provenance"],
                                         record["membership_kind"]))
    for name, values in state["maps"].items():
        setattr(bank, name, {key: bank.candidates[index] if index is not None else None
                            for key, index in values.items()})
    for name, value in state["plain"].items():
        setattr(bank, name, value)
    if state.get("original_selected_index") is not None:
        bank.original_selected = bank.candidates[state["original_selected_index"]]
    return bank
