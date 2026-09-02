"""Atomic, identity-guarded checkpoints for the online raw-lambda search."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._version import __version__
from ..config import FitConfig
from ..io.data import TumorData, tumor_data_fingerprint


CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointIdentityMismatchError(ValueError):
    """A checkpoint belongs to a different immutable analysis surface."""


def _float_token(value: float) -> str:
    return float(value).hex()


def _canonical_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        return {"float_hex": _float_token(float(value))}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        return {
            "array_dtype": array.dtype.str,
            "array_shape": list(array.shape),
            "array_sha256": digest,
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_identity_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _canonical_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}:{type(value).__qualname__}",
            "fields": {
                item.name: _canonical_identity_value(getattr(value, item.name))
                for item in fields(value)
                if item.init
            },
        }
    raise TypeError(f"Unsupported checkpoint-identity value: {type(value).__name__}")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _canonical_identity_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_tree_fingerprint() -> str:
    """Hash package Python sources so resumes cannot cross code revisions."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(b"clipp2.source-tree.v1\0")
    for source in sorted(package_root.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        contents = source.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def build_search_checkpoint_identity(
    *,
    data: TumorData,
    fit_config: FitConfig,
    objective_spec_hash: str,
    original_graph_hash: str,
    use_warm_starts: bool,
    runtime_device_name: str,
    runtime_dtype: str,
) -> dict[str, Any]:
    """Return every immutable identity required for a safe search resume."""

    contract = fit_config.selection.contract
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tumor_data_fingerprint": tumor_data_fingerprint(data),
        "objective_spec_hash": str(objective_spec_hash),
        "original_graph_hash": str(original_graph_hash),
        "selection_contract_id": str(contract.contract_id),
        "selection_contract_hash": hashlib.sha256(
            contract.to_json().encode("utf-8")
        ).hexdigest(),
        "computation_profile": str(fit_config.computation_profile.name),
        "resolved_fit_config_hash": _sha256_json(fit_config),
        "use_warm_starts": bool(use_warm_starts),
        "runtime_device_name": str(runtime_device_name),
        "runtime_dtype": str(runtime_dtype),
        "software_version": str(__version__),
        "source_tree_fingerprint": source_tree_fingerprint(),
    }


@lru_cache(maxsize=1)
def _checkpoint_type_registry() -> dict[str, type[Any]]:
    """Build the explicit allowlist used by the non-pickle decoder."""

    from ..core.bic import SelectionScore
    from ..core.fusion.types import (
        CertificateResult,
        CompressedEdgeCertificate,
        ConvergenceResult,
        DenseEdgeCertificate,
        DenseWarmState,
        FitProvenance,
        KKTComponents,
        ObjectiveValue,
        PrimalOnlyWarmState,
        RawFit,
        SolverState,
        WorkCounters,
    )
    from ..core.objective import BaseObjectiveKey, LambdaObjectiveKey
    from ..core.scalar import PartitionRefitResult
    from .candidates import PartitionRefitCacheEntry
    from .guided_fusion import GuidedFusionDiagnostics, GuidedFusionInitialization
    from .types import (
        CandidateRecord,
        CandidateTrace,
        FusionPartition,
        PartitionRefitSummary,
        RawAttemptSummary,
        RawFusionCandidate,
        SolveOutcome,
    )

    allowed = (
        BaseObjectiveKey,
        CandidateRecord,
        CandidateTrace,
        CertificateResult,
        CompressedEdgeCertificate,
        ConvergenceResult,
        DenseEdgeCertificate,
        DenseWarmState,
        FitProvenance,
        FusionPartition,
        GuidedFusionDiagnostics,
        GuidedFusionInitialization,
        KKTComponents,
        LambdaObjectiveKey,
        ObjectiveValue,
        PartitionRefitCacheEntry,
        PartitionRefitResult,
        PartitionRefitSummary,
        PrimalOnlyWarmState,
        RawAttemptSummary,
        RawFit,
        RawFusionCandidate,
        SelectionScore,
        SolveOutcome,
        SolverState,
        WorkCounters,
    )
    # The unscored raw type is introduced by the staged-evaluation change.  A
    # getattr keeps this codec importable while old checkpoints are tested.
    from . import types as selection_types

    unscored = getattr(selection_types, "UnscoredRawFusionCandidate", None)
    if unscored is not None:
        allowed = (*allowed, unscored)
    return {
        f"{item.__module__}:{item.__qualname__}": item
        for item in allowed
    }


def _encode_value(
    value: Any,
    arrays: dict[str, np.ndarray],
    memo: dict[int, str],
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        return {"$float": _float_token(float(value))}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        existing = memo.get(id(value))
        if existing is not None:
            return {"$ref": existing}
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not permitted in a search checkpoint.")
        object_id = f"object_{len(memo):08d}"
        memo[id(value)] = object_id
        key = f"array_{len(arrays):08d}"
        arrays[key] = np.ascontiguousarray(value)
        return {
            "$array": key,
            "$id": object_id,
            "readonly": not bool(value.flags.writeable),
        }
    if torch.is_tensor(value):
        existing = memo.get(id(value))
        if existing is not None:
            return {"$ref": existing}
        object_id = f"object_{len(memo):08d}"
        memo[id(value)] = object_id
        tensor = value.detach().cpu().contiguous()
        key = f"array_{len(arrays):08d}"
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if tensor.dtype == torch.bfloat16:
            arrays[key] = tensor.view(torch.uint8).numpy()
            byte_view = True
        else:
            arrays[key] = tensor.numpy()
            byte_view = False
        return {
            "$tensor": key,
            "$id": object_id,
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "byte_view": byte_view,
        }
    if isinstance(value, tuple):
        return {"$tuple": [_encode_value(item, arrays, memo) for item in value]}
    if isinstance(value, list):
        return {"$list": [_encode_value(item, arrays, memo) for item in value]}
    if isinstance(value, set):
        encoded = [_encode_value(item, arrays, memo) for item in value]
        return {
            "$set": sorted(
                encoded,
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
            )
        }
    if isinstance(value, dict):
        return {
            "$dict": [
                [
                    _encode_value(key, arrays, memo),
                    _encode_value(item, arrays, memo),
                ]
                for key, item in value.items()
            ]
        }
    if is_dataclass(value):
        existing = memo.get(id(value))
        if existing is not None:
            return {"$ref": existing}
        type_name = f"{type(value).__module__}:{type(value).__qualname__}"
        if type_name not in _checkpoint_type_registry():
            raise TypeError(f"Dataclass is not checkpoint-allowlisted: {type_name}")
        object_id = f"object_{len(memo):08d}"
        memo[id(value)] = object_id
        return {
            "$dataclass": type_name,
            "$id": object_id,
            "fields": {
                item.name: _encode_value(getattr(value, item.name), arrays, memo)
                for item in fields(value)
                if item.init
            },
        }
    raise TypeError(f"Unsupported search-checkpoint value: {type(value).__name__}")


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def _decode_value(
    encoded: Any,
    arrays: dict[str, np.ndarray],
    registry: dict[str, type[Any]],
    memo: dict[str, Any],
) -> Any:
    if encoded is None or isinstance(encoded, (bool, int, str)):
        return encoded
    if not isinstance(encoded, dict):
        raise ValueError("Malformed search-checkpoint manifest value.")

    def require_keys(expected: set[str]) -> None:
        if set(encoded) != expected:
            raise ValueError(
                "Malformed search-checkpoint manifest keys: "
                f"expected={sorted(expected)}, observed={sorted(encoded)}"
            )

    if "$ref" in encoded:
        require_keys({"$ref"})
        object_id = str(encoded["$ref"])
        if object_id not in memo:
            raise ValueError("Checkpoint contains a forward or unknown reference.")
        return memo[object_id]
    if "$float" in encoded:
        require_keys({"$float"})
        return float.fromhex(str(encoded["$float"]))
    if "$array" in encoded:
        require_keys({"$array", "$id", "readonly"})
        key = str(encoded["$array"])
        if key not in arrays:
            raise ValueError(f"Checkpoint array is missing: {key}")
        value = np.array(arrays[key], copy=True, order="C")
        if bool(encoded.get("readonly", False)):
            value.setflags(write=False)
        object_id = str(encoded["$id"])
        if object_id in memo:
            raise ValueError("Checkpoint object ID is duplicated.")
        memo[object_id] = value
        return value
    if "$tensor" in encoded:
        require_keys({"$tensor", "$id", "dtype", "shape", "byte_view"})
        key = str(encoded["$tensor"])
        if key not in arrays:
            raise ValueError(f"Checkpoint tensor is missing: {key}")
        dtype_name = str(encoded["dtype"])
        if dtype_name not in _TORCH_DTYPES:
            raise ValueError(f"Checkpoint tensor dtype is unsupported: {dtype_name}")
        tensor = torch.from_numpy(np.array(arrays[key], copy=True, order="C"))
        dtype = _TORCH_DTYPES[dtype_name]
        if bool(encoded.get("byte_view", False)):
            tensor = tensor.view(dtype)
        else:
            tensor = tensor.to(dtype=dtype)
        tensor = tensor.reshape(tuple(int(item) for item in encoded["shape"]))
        object_id = str(encoded["$id"])
        if object_id in memo:
            raise ValueError("Checkpoint object ID is duplicated.")
        memo[object_id] = tensor
        return tensor
    if "$tuple" in encoded:
        require_keys({"$tuple"})
        return tuple(
            _decode_value(item, arrays, registry, memo)
            for item in encoded["$tuple"]
        )
    if "$list" in encoded:
        require_keys({"$list"})
        return [
            _decode_value(item, arrays, registry, memo)
            for item in encoded["$list"]
        ]
    if "$set" in encoded:
        require_keys({"$set"})
        values = [
            _decode_value(item, arrays, registry, memo)
            for item in encoded["$set"]
        ]
        result = set(values)
        if len(result) != len(values):
            raise ValueError("Checkpoint set contains duplicate values.")
        return result
    if "$dict" in encoded:
        require_keys({"$dict"})
        result: dict[Any, Any] = {}
        for pair in encoded["$dict"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Checkpoint mapping entry must be a key/value pair.")
            key = _decode_value(pair[0], arrays, registry, memo)
            if key in result:
                raise ValueError("Checkpoint mapping contains duplicate keys.")
            result[key] = _decode_value(pair[1], arrays, registry, memo)
        return result
    if "$dataclass" in encoded:
        require_keys({"$dataclass", "$id", "fields"})
        type_name = str(encoded["$dataclass"])
        cls = registry.get(type_name)
        if cls is None:
            raise ValueError(f"Checkpoint dataclass is not allowlisted: {type_name}")
        encoded_fields = encoded.get("fields")
        if not isinstance(encoded_fields, dict) or not all(
            isinstance(name, str) for name in encoded_fields
        ):
            raise ValueError("Checkpoint dataclass fields must be a mapping.")
        expected_fields = {item.name for item in fields(cls) if item.init}
        if set(encoded_fields) != expected_fields:
            raise ValueError(
                f"Checkpoint dataclass fields mismatch for {type_name}."
            )
        # JSON canonicalization sorts mapping keys, but references are emitted
        # in dataclass field order.  Decode in that same order so an aliased
        # tensor/state is materialized before a later ``$ref`` reaches it.
        values = {
            item.name: _decode_value(
                encoded_fields[item.name], arrays, registry, memo
            )
            for item in fields(cls)
            if item.init
        }
        value = cls(**values)
        object_id = str(encoded["$id"])
        if object_id in memo:
            raise ValueError("Checkpoint object ID is duplicated.")
        memo[object_id] = value
        return value
    raise ValueError("Malformed search-checkpoint manifest tag.")


def save_search_checkpoint(
    path: str | Path,
    *,
    identity: dict[str, Any],
    workspace: dict[str, Any],
) -> None:
    """Atomically replace one checkpoint after a completed observation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": _canonical_identity_value(identity),
        "workspace": _encode_value(workspace, arrays, {}),
    }
    manifest_bytes = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                __manifest__=np.frombuffer(manifest_bytes, dtype=np.uint8),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_search_checkpoint(
    path: str | Path,
    *,
    expected_identity: dict[str, Any],
) -> dict[str, Any]:
    """Load a trusted-format checkpoint and reject any identity mismatch."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if "__manifest__" not in archive.files:
            raise ValueError("Search checkpoint has no manifest.")
        manifest = json.loads(
            np.asarray(archive["__manifest__"], dtype=np.uint8).tobytes().decode(
                "utf-8"
            )
        )
        if int(manifest.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported search-checkpoint schema version.")
        expected = _canonical_identity_value(expected_identity)
        observed = manifest.get("identity")
        if observed != expected:
            if not isinstance(observed, dict):
                raise CheckpointIdentityMismatchError(
                    "Search checkpoint identity is malformed."
                )
            mismatches = sorted(
                key
                for key in set(expected) | set(observed)
                if observed.get(key) != expected.get(key)
            )
            raise CheckpointIdentityMismatchError(
                "Search checkpoint identity mismatch: " + ", ".join(mismatches)
            )

        referenced_arrays: set[str] = set()

        def collect_array_references(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect_array_references(item)
                return
            if not isinstance(value, dict):
                return
            for tag in ("$array", "$tensor"):
                if tag in value:
                    referenced_arrays.add(str(value[tag]))
            for item in value.values():
                collect_array_references(item)

        collect_array_references(manifest.get("workspace"))
        stored_arrays = set(archive.files) - {"__manifest__"}
        if stored_arrays != referenced_arrays:
            raise ValueError("Checkpoint archive members do not match its manifest.")
        arrays = {
            key: np.array(archive[key], copy=True, order="C")
            for key in archive.files
            if key != "__manifest__"
        }
    decoded = _decode_value(
        manifest.get("workspace"), arrays, _checkpoint_type_registry(), {}
    )
    if not isinstance(decoded, dict):
        raise ValueError("Search checkpoint workspace must be a mapping.")
    return decoded


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointIdentityMismatchError",
    "build_search_checkpoint_identity",
    "load_search_checkpoint",
    "save_search_checkpoint",
    "source_tree_fingerprint",
]
