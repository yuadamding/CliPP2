"""Incremental, identity-guarded checkpoints for the online raw-lambda search.

The on-disk format is a directory containing one atomically replaced JSON
manifest and immutable, content-addressed NumPy objects.  It deliberately does
not use pickle.  Repeated saves therefore write only arrays that have not
already appeared in the checkpoint instead of rewriting the complete solver
history after every lambda observation.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import fields, is_dataclass
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any
import uuid

import numpy as np
import torch

from .._version import __version__
from ..config import FitConfig
from ..io.data import TumorData, tumor_data_fingerprint


CHECKPOINT_SCHEMA_VERSION = 2

_MANIFEST_NAME = "manifest.json"
_ARRAY_DIRECTORY_NAME = "arrays"
_WRITER_LOCK_NAME = ".writer.lock"
_OBJECT_NAME_RE = re.compile(r"^[0-9a-f]{64}\.npy$")
_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "CLIPP2_MAX_COMPLETE_GRAPH_BYTES",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "VECLIB_MAXIMUM_THREADS",
)


class CheckpointIdentityMismatchError(ValueError):
    """A checkpoint belongs to a different immutable analysis surface."""


class CheckpointConcurrentWriterError(RuntimeError):
    """Another writer owns or changed the checkpoint generation."""


class LegacyCheckpointFormatError(ValueError):
    """A monolithic v1 NPZ checkpoint cannot be resumed as schema v2."""


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


def _configuration_text_hash(show: Any) -> str:
    """Hash backend build information without putting host paths in a manifest."""

    stream = io.StringIO()
    try:
        with redirect_stdout(stream):
            returned = show()
    except Exception:  # pragma: no cover - backend-specific diagnostic surface
        return "unavailable"
    contents = stream.getvalue()
    if returned is not None:
        contents += str(returned)
    return hashlib.sha256(contents.encode("utf-8")).hexdigest()


def _cuda_hardware_metadata(runtime_device_name: str) -> dict[str, Any] | None:
    """Describe the selected CUDA device without embedding a machine secret."""

    if not str(runtime_device_name).lower().startswith("cuda"):
        return None
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        device = torch.device(str(runtime_device_name))
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        return {
            "available": True,
            "device_name": str(torch.cuda.get_device_name(index)),
            "compute_capability": [int(capability[0]), int(capability[1])],
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    except Exception as exc:  # pragma: no cover - broken CUDA runtime guard
        return {
            "available": True,
            "metadata_error": type(exc).__name__,
        }


def _numerical_environment_metadata(runtime_device_name: str) -> dict[str, Any]:
    """Return the numerical runtime surface that must match on exact resume."""

    cudnn_version = None
    if torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()
    deterministic_debug_mode = getattr(torch, "get_deterministic_debug_mode", None)
    float32_matmul_precision = getattr(torch, "get_float32_matmul_precision", None)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_implementation": str(sys.implementation.name),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": str(np.__version__),
        "numpy_build_config_sha256": _configuration_text_hash(np.__config__.show),
        "torch_version": str(torch.__version__),
        "torch_build_config_sha256": _configuration_text_hash(torch.__config__.show),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "torch_cudnn_version": (
            None if cudnn_version is None else int(cudnn_version)
        ),
        "cuda_hardware": _cuda_hardware_metadata(runtime_device_name),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_debug_mode": (
            None
            if deterministic_debug_mode is None
            else int(deterministic_debug_mode())
        ),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": (
            None
            if float32_matmul_precision is None
            else str(float32_matmul_precision())
        ),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "environment": {name: os.getenv(name) for name in _ENVIRONMENT_KEYS},
    }


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
        "numerical_environment": _numerical_environment_metadata(
            runtime_device_name
        ),
        "software_version": str(__version__),
        "source_tree_fingerprint": source_tree_fingerprint(),
    }


@lru_cache(maxsize=1)
def _checkpoint_type_registry() -> dict[str, type[Any]]:
    """Build the explicit allowlist used by the non-pickle decoder."""

    from ..core.bic import SelectionScore
    from ..core.fusion.partition_starts import PartitionCandidate
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
        DirectPartition,
        DirectPartitionCandidate,
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
        DirectPartition,
        DirectPartitionCandidate,
        FitProvenance,
        FusionPartition,
        GuidedFusionDiagnostics,
        GuidedFusionInitialization,
        KKTComponents,
        LambdaObjectiveKey,
        ObjectiveValue,
        PartitionRefitCacheEntry,
        PartitionRefitResult,
        PartitionCandidate,
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


def _contiguous_array(value: np.ndarray) -> np.ndarray:
    if value.dtype.hasobject:
        raise TypeError("Object arrays are not permitted in a search checkpoint.")
    if value.flags.c_contiguous:
        return value
    return np.ascontiguousarray(value)


def _array_content_digest(value: np.ndarray) -> str:
    """Hash logical dtype, shape, and C-order bytes without a full byte copy."""

    array = _contiguous_array(value)
    digest = hashlib.sha256()
    digest.update(b"clipp2.checkpoint-array.v2\0")
    dtype = array.dtype.str.encode("ascii")
    shape = json.dumps(
        [int(item) for item in array.shape], separators=(",", ":")
    ).encode("ascii")
    digest.update(len(dtype).to_bytes(8, "little"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(8, "little"))
    digest.update(shape)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _register_array(
    value: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> str:
    array = _contiguous_array(value)
    digest = _array_content_digest(array)
    incumbent = arrays.get(digest)
    if incumbent is not None and (
        incumbent.dtype != array.dtype
        or incumbent.shape != array.shape
    ):
        raise RuntimeError("Checkpoint array SHA-256 collision.")
    arrays.setdefault(digest, array)
    return digest


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
        object_id = f"object_{len(memo):08d}"
        memo[id(value)] = object_id
        key = _register_array(value, arrays)
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
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name not in _TORCH_DTYPES:
            raise TypeError(
                f"Tensor dtype is not supported by search checkpoints: {dtype_name}"
            )
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint8).numpy()
            byte_view = True
        else:
            array = tensor.numpy()
            byte_view = False
        key = _register_array(array, arrays)
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
    arrays: Any,
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
        if not isinstance(encoded["readonly"], bool):
            raise ValueError("Checkpoint array readonly flag must be boolean.")
        stored = arrays[key]
        if encoded["readonly"]:
            # Content objects are mapped read-only, so immutable NumPy payloads
            # can remain zero-copy.  Mutable arrays receive an owned copy below.
            value = stored
            value.setflags(write=False)
        else:
            value = np.array(stored, copy=True, order="C")
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
        if not isinstance(encoded["byte_view"], bool):
            raise ValueError("Checkpoint tensor byte_view flag must be boolean.")
        if encoded["byte_view"] != (dtype_name == "bfloat16"):
            raise ValueError("Checkpoint tensor byte-view metadata is inconsistent.")
        tensor = torch.from_numpy(np.array(arrays[key], copy=True, order="C"))
        dtype = _TORCH_DTYPES[dtype_name]
        if encoded["byte_view"]:
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


def _legacy_checkpoint_error(path: Path) -> LegacyCheckpointFormatError:
    return LegacyCheckpointFormatError(
        f"{path} is a monolithic NPZ checkpoint. Schema-v2 checkpoints are "
        "content-addressed directories; restart with a new checkpoint path."
    )


def _require_private_mode(path: Path, *, directory: bool) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"Checkpoint path must not be a symlink: {path}")
    status = path.stat()
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_kind(status.st_mode):
        raise ValueError(f"Checkpoint path has the wrong file type: {path}")
    if stat.S_IMODE(status.st_mode) & 0o077:
        raise PermissionError(f"Checkpoint path is not private: {path}")
    return status


def _prepare_checkpoint_directory(destination: Path) -> tuple[Path, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError(
            f"Checkpoint directory must not be a symlink: {destination}"
        )
    try:
        destination.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise _legacy_checkpoint_error(destination) from exc
    if not destination.is_dir():
        raise _legacy_checkpoint_error(destination)
    os.chmod(destination, 0o700)
    _require_private_mode(destination, directory=True)

    array_directory = destination / _ARRAY_DIRECTORY_NAME
    if array_directory.is_symlink():
        raise ValueError(
            f"Checkpoint arrays path must be a real directory: {array_directory}"
        )
    try:
        array_directory.mkdir(mode=0o700, exist_ok=True)
    except FileExistsError as exc:
        raise ValueError(
            f"Checkpoint arrays path must be a real directory: {array_directory}"
        ) from exc
    if not array_directory.is_dir():
        raise ValueError(
            f"Checkpoint arrays path must be a real directory: {array_directory}"
        )
    os.chmod(array_directory, 0o700)
    _require_private_mode(array_directory, directory=True)
    return destination / _MANIFEST_NAME, array_directory


def _open_checkpoint_directory(source: Path) -> tuple[Path, Path]:
    if source.is_symlink():
        raise ValueError(f"Checkpoint directory must not be a symlink: {source}")
    if not source.exists():
        raise FileNotFoundError(f"Search checkpoint does not exist: {source}")
    if not source.is_dir():
        raise _legacy_checkpoint_error(source)
    _require_private_mode(source, directory=True)
    manifest_path = source / _MANIFEST_NAME
    array_directory = source / _ARRAY_DIRECTORY_NAME
    if not manifest_path.exists():
        raise ValueError("Search checkpoint has no manifest.json.")
    _require_private_mode(manifest_path, directory=False)
    if not array_directory.exists():
        raise ValueError("Search checkpoint has no arrays directory.")
    _require_private_mode(array_directory, directory=True)
    return manifest_path, array_directory


def _read_manifest_bytes(path: Path) -> bytes:
    _require_private_mode(path, directory=False)
    return path.read_bytes()


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Checkpoint manifest repeats JSON key: {key}")
        result[key] = value
    return result


def _parse_manifest(contents: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(
            contents.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Invalid JSON constant in checkpoint: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Search checkpoint manifest is corrupted.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Search checkpoint manifest must be a mapping.")
    expected_keys = {
        "schema_version",
        "generation",
        "previous_manifest_sha256",
        "identity",
        "objects",
        "workspace",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            "Malformed search-checkpoint manifest keys: "
            f"expected={sorted(expected_keys)}, observed={sorted(manifest)}"
        )
    if type(manifest["schema_version"]) is not int or (
        manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported search-checkpoint schema version.")
    generation = manifest["generation"]
    if type(generation) is not int or generation < 1:
        raise ValueError("Search checkpoint generation must be a positive integer.")
    predecessor = manifest["previous_manifest_sha256"]
    if predecessor is not None and (
        not isinstance(predecessor, str)
        or re.fullmatch(r"[0-9a-f]{64}", predecessor) is None
    ):
        raise ValueError("Search checkpoint predecessor digest is malformed.")
    if generation == 1 and predecessor is not None:
        raise ValueError("First checkpoint generation cannot name a predecessor.")
    if generation > 1 and predecessor is None:
        raise ValueError("Later checkpoint generation must name its predecessor.")
    if not isinstance(manifest["identity"], dict):
        raise ValueError("Search checkpoint identity is malformed.")
    objects = manifest["objects"]
    if not isinstance(objects, dict):
        raise ValueError("Search checkpoint object table must be a mapping.")
    for digest, metadata in objects.items():
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Search checkpoint object digest is malformed.")
        if not isinstance(metadata, dict) or set(metadata) != {
            "dtype",
            "shape",
            "nbytes",
        }:
            raise ValueError("Search checkpoint object metadata is malformed.")
        try:
            dtype = np.dtype(metadata["dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Search checkpoint object dtype is malformed.") from exc
        if dtype.hasobject or str(metadata["dtype"]) != dtype.str:
            raise ValueError("Search checkpoint object dtype is not canonical.")
        shape = metadata["shape"]
        if not isinstance(shape, list) or any(
            type(item) is not int or item < 0 for item in shape
        ):
            raise ValueError("Search checkpoint object shape is malformed.")
        expected_nbytes = int(dtype.itemsize)
        for extent in shape:
            expected_nbytes *= int(extent)
        if type(metadata["nbytes"]) is not int or (
            int(metadata["nbytes"]) != expected_nbytes
        ):
            raise ValueError("Search checkpoint object size is inconsistent.")
    return manifest


def _identity_mismatches(
    expected_identity: dict[str, Any],
    observed_identity: dict[str, Any],
) -> list[str]:
    missing = object()
    return sorted(
        key
        for key in set(expected_identity) | set(observed_identity)
        if observed_identity.get(key, missing) != expected_identity.get(key, missing)
    )


def _require_identity(
    expected_identity: dict[str, Any],
    observed_identity: dict[str, Any],
) -> None:
    if observed_identity == expected_identity:
        return
    mismatches = _identity_mismatches(expected_identity, observed_identity)
    raise CheckpointIdentityMismatchError(
        "Search checkpoint identity mismatch: " + ", ".join(mismatches)
    )


def _collect_array_references(value: Any, result: set[str]) -> None:
    if isinstance(value, list):
        for item in value:
            _collect_array_references(item, result)
        return
    if not isinstance(value, dict):
        return
    for tag in ("$array", "$tensor"):
        if tag in value:
            digest = value[tag]
            if not isinstance(digest, str) or re.fullmatch(
                r"[0-9a-f]{64}", digest
            ) is None:
                raise ValueError("Checkpoint array reference is malformed.")
            result.add(digest)
    for item in value.values():
        _collect_array_references(item, result)


def _object_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "nbytes": int(array.nbytes),
    }


def _require_safe_object_file(path: Path) -> None:
    if path.is_symlink() or not path.exists():
        raise ValueError(f"Checkpoint array is missing or is a symlink: {path.name}")
    _require_private_mode(path, directory=False)


def _load_array_object(
    path: Path,
    *,
    digest: str,
    metadata: dict[str, Any],
) -> np.ndarray:
    _require_safe_object_file(path)
    try:
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError:
            # NumPy cannot memory-map scalar and empty payloads on every
            # supported version.  They remain small and safe to load eagerly.
            value = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"Checkpoint array is corrupted: {digest}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        close = getattr(value, "close", None)
        if close is not None:
            close()
        raise ValueError(f"Checkpoint array is not a plain numeric array: {digest}")
    if (
        value.dtype.str != metadata["dtype"]
        or [int(item) for item in value.shape] != metadata["shape"]
        or int(value.nbytes) != int(metadata["nbytes"])
    ):
        raise ValueError(f"Checkpoint array metadata mismatch: {digest}")
    if _array_content_digest(value) != digest:
        raise ValueError(f"Checkpoint array checksum mismatch: {digest}")
    value.setflags(write=False)
    return value


class _CheckpointObjectStore:
    """Validate object names eagerly and map payloads only when decoded."""

    def __init__(
        self,
        directory: Path,
        metadata: dict[str, dict[str, Any]],
    ) -> None:
        self._directory = directory
        self._metadata = metadata
        self._cache: dict[str, np.ndarray] = {}
        self.accessed: set[str] = set()
        for entry in directory.iterdir():
            if entry.name.startswith(".object-tmp-"):
                # A process death before linking can leave a harmless private
                # temporary. It is never part of a committed generation.
                continue
            if _OBJECT_NAME_RE.fullmatch(entry.name) is None:
                raise ValueError(
                    f"Unexpected checkpoint arrays member: {entry.name}"
                )
            _require_safe_object_file(entry)
        for digest in metadata:
            path = directory / f"{digest}.npy"
            if not path.exists() or path.is_symlink():
                raise ValueError(f"Checkpoint array is missing: {digest}")

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and digest in self._metadata

    def __getitem__(self, digest: str) -> np.ndarray:
        if digest not in self._metadata:
            raise KeyError(digest)
        self.accessed.add(digest)
        if digest not in self._cache:
            self._cache[digest] = _load_array_object(
                self._directory / f"{digest}.npy",
                digest=digest,
                metadata=self._metadata[digest],
            )
        return self._cache[digest]


def _write_array_object(
    directory: Path,
    *,
    digest: str,
    value: np.ndarray,
) -> bool:
    destination = directory / f"{digest}.npy"
    if destination.exists() or destination.is_symlink():
        _require_safe_object_file(destination)
        return False
    temporary = directory / (
        f".object-tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _require_safe_object_file(destination)
        else:
            os.chmod(destination, 0o600)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _checkpoint_writer_lock(directory: Path) -> Any:
    lock_path = directory / _WRITER_LOCK_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise CheckpointConcurrentWriterError(
            f"Checkpoint already has an active or stale writer lock: {lock_path}"
        ) from exc
    status = os.fstat(descriptor)
    try:
        payload = json.dumps(
            {"pid": os.getpid()}, separators=(",", ":")
        ).encode("ascii")
        os.write(descriptor, payload)
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = lock_path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (
            current.st_dev == status.st_dev and current.st_ino == status.st_ino
        ):
            lock_path.unlink()


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def save_search_checkpoint(
    path: str | Path,
    *,
    identity: dict[str, Any],
    workspace: dict[str, Any],
    expected_generation: int | None = None,
) -> int:
    """Commit one incremental checkpoint generation after an observation.

    ``path`` is a directory even when a caller retains the historical ``.npz``
    suffix.  A lock rejects overlapping writers, and a pre-replace generation
    comparison also rejects writers that ignore that lock. A caller must pass
    the generation returned by its previous save or load when updating an
    existing checkpoint, preventing a stale sequential writer from committing.
    """

    destination = Path(path)
    manifest_path, array_directory = _prepare_checkpoint_directory(destination)
    expected_identity = _canonical_identity_value(identity)
    if expected_generation is not None and (
        type(expected_generation) is not int or expected_generation < 0
    ):
        raise ValueError("expected_generation must be a nonnegative integer.")
    with _checkpoint_writer_lock(destination):
        original_manifest_bytes: bytes | None = None
        observed_generation = 0
        if manifest_path.exists() or manifest_path.is_symlink():
            original_manifest_bytes = _read_manifest_bytes(manifest_path)
            original_manifest = _parse_manifest(original_manifest_bytes)
            _require_identity(expected_identity, original_manifest["identity"])
            observed_generation = int(original_manifest["generation"])
            if expected_generation is None:
                raise CheckpointConcurrentWriterError(
                    "Updating an existing checkpoint requires expected_generation."
                )
        if expected_generation is not None and (
            int(expected_generation) != observed_generation
        ):
            raise CheckpointConcurrentWriterError(
                "Checkpoint generation mismatch: "
                f"expected={expected_generation}, observed={observed_generation}."
            )
        generation = observed_generation + 1

        arrays: dict[str, np.ndarray] = {}
        encoded_workspace = _encode_value(workspace, arrays, {})
        objects = {
            digest: _object_metadata(value)
            for digest, value in sorted(arrays.items())
        }
        wrote_object = False
        for digest, value in arrays.items():
            wrote_object |= _write_array_object(
                array_directory,
                digest=digest,
                value=value,
            )
        if wrote_object:
            _fsync_directory(array_directory)

        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "generation": generation,
            "previous_manifest_sha256": (
                None
                if original_manifest_bytes is None
                else hashlib.sha256(original_manifest_bytes).hexdigest()
            ),
            "identity": expected_identity,
            "objects": objects,
            "workspace": encoded_workspace,
        }
        contents = _manifest_bytes(manifest)
        temporary = destination / (
            f".manifest-tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)

            current_manifest_bytes = (
                _read_manifest_bytes(manifest_path)
                if manifest_path.exists()
                else None
            )
            if current_manifest_bytes != original_manifest_bytes:
                raise CheckpointConcurrentWriterError(
                    "Checkpoint generation changed before manifest commit."
                )
            os.replace(temporary, manifest_path)
            _fsync_directory(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return generation


def load_search_checkpoint(
    path: str | Path,
    *,
    expected_identity: dict[str, Any],
    return_generation: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], int]:
    """Load a checkpoint after identity-first, no-pickle validation."""

    source = Path(path)
    manifest_path, array_directory = _open_checkpoint_directory(source)
    manifest = _parse_manifest(_read_manifest_bytes(manifest_path))
    expected = _canonical_identity_value(expected_identity)
    _require_identity(expected, manifest["identity"])

    referenced_arrays: set[str] = set()
    _collect_array_references(manifest["workspace"], referenced_arrays)
    stored_metadata = manifest["objects"]
    if set(stored_metadata) != referenced_arrays:
        raise ValueError(
            "Checkpoint object table does not match its manifest references."
        )
    arrays = _CheckpointObjectStore(array_directory, stored_metadata)
    decoded = _decode_value(
        manifest["workspace"], arrays, _checkpoint_type_registry(), {}
    )
    if arrays.accessed != referenced_arrays:
        raise ValueError("Checkpoint decoder did not consume every referenced array.")
    if not isinstance(decoded, dict):
        raise ValueError("Search checkpoint workspace must be a mapping.")
    if return_generation:
        return decoded, int(manifest["generation"])
    return decoded


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointConcurrentWriterError",
    "CheckpointIdentityMismatchError",
    "LegacyCheckpointFormatError",
    "build_search_checkpoint_identity",
    "load_search_checkpoint",
    "save_search_checkpoint",
    "source_tree_fingerprint",
]
