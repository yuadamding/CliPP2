"""Incremental, identity-guarded checkpoints for the online raw-lambda search.

The on-disk format is a directory containing one atomically replaced JSON
manifest and immutable, content-addressed NumPy objects.  It deliberately does
not use pickle.  Repeated saves therefore write only arrays that have not
already appeared in the checkpoint instead of rewriting the complete solver
history after every lambda observation.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import TYPE_CHECKING, Any
import uuid

import numpy as np
import torch

from .._version import __version__
from ..config import FitConfig, PRODUCTION_SELECTION_POLICY
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
    WorkLedger,
)
from ..core.objective import BaseObjectiveKey, LambdaObjectiveKey
from ..core.scalar import PartitionFit
from ..io.data import TumorData, tumor_data_fingerprint
from .guided_fusion import GuidedFusionDiagnostics, GuidedFusionInitialization
from .online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
    OnlineLambdaProposal,
    OnlineLambdaState,
)
from .types import (
    AttemptLimits,
    CandidateRecord,
    CandidateTrace,
    DirectProposal,
    DirectPartition,
    DirectPartitionCandidate,
    FitAuditSummary,
    FusionPartition,
    PARTITION_REFIT_KEY_SCHEMA,
    PartitionRefitKey,
    RawAttemptSummary,
    RawFusionCandidate,
    SolveOutcome,
    UnscoredRawFusionCandidate,
)

if TYPE_CHECKING:
    from .search import SearchState


CHECKPOINT_SCHEMA_VERSION = 4

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
_INFERENCE_PACKAGE_DIRECTORIES = (
    "core",
    "io",
    "model_selection",
    "reporting",
    "runners",
)


class CheckpointIdentityMismatchError(ValueError):
    """A checkpoint belongs to a different immutable analysis surface."""


class CheckpointConcurrentWriterError(RuntimeError):
    """Another writer owns or changed the checkpoint generation."""


class LegacyCheckpointFormatError(ValueError):
    """A monolithic v1 NPZ checkpoint cannot be resumed as schema v4."""


@dataclass(frozen=True, slots=True)
class LambdaCheckpoint:
    """Durable state for one canonical lambda, without object aliases."""

    lambda_value: float
    attempts: tuple[SolveOutcome, ...]
    retained: SolveOutcome | None
    partition_k: int | None


@dataclass(frozen=True, slots=True)
class DirectPoolCheckpoint:
    """Durable cursor and proposals for the deterministic direct pool."""

    next_index: int
    complete: bool
    stop_reason: str | None
    proposals: tuple[DirectProposal, ...] | None
    proposals_complete: bool
    final_parent_next_index: int


@dataclass(frozen=True, slots=True)
class SearchCheckpoint:
    """The sole v4 payload schema for resumable model selection."""

    controller: OnlineLambdaState
    lambda_runs: tuple[LambdaCheckpoint, ...]
    candidates: tuple[CandidateRecord, ...]
    refit_cache: tuple[tuple[PartitionRefitKey, PartitionFit], ...]
    next_candidate_id: int
    work: WorkCounters
    guide_work: WorkCounters
    search_stop_override: str | None
    direct_pool: DirectPoolCheckpoint
    raw_guide_phi: np.ndarray | torch.Tensor
    guided_initialization: GuidedFusionInitialization
    float64_recovery_status: str
    elapsed_seconds: float

    @classmethod
    def capture(cls, state: SearchState) -> "SearchCheckpoint":
        """Freeze the checkpointable portion of one ``SearchState``."""

        from .search import SearchState

        if not isinstance(state, SearchState):
            raise TypeError("SearchCheckpoint.capture requires SearchState.")
        lambda_runs = tuple(
            LambdaCheckpoint(
                lambda_value=float(run.lambda_value),
                attempts=tuple(run.attempts),
                retained=run.retained,
                partition_k=run.partition_k,
            )
            for run in state.lambdas.values()
        )
        proposals = state.direct_pool.proposals
        direct_pool = DirectPoolCheckpoint(
            next_index=int(state.direct_pool.next_index),
            complete=bool(state.direct_pool.complete),
            stop_reason=state.direct_pool.stop_reason,
            proposals=None if proposals is None else tuple(proposals),
            proposals_complete=bool(state.direct_pool.proposals_complete),
            final_parent_next_index=int(state.direct_pool.final_parent_next_index),
        )
        cache: list[tuple[PartitionRefitKey, PartitionFit]] = []
        for key, value in state.bic_refit_cache.items():
            if not isinstance(key, PartitionRefitKey):
                raise TypeError("Search refit-cache keys must be PartitionRefitKey.")
            cache.append((key, value))
        return cls(
            controller=state.controller.snapshot(),
            lambda_runs=lambda_runs,
            candidates=tuple(state.candidates),
            refit_cache=tuple(cache),
            next_candidate_id=int(state.next_candidate_id),
            work=state.total_work,
            guide_work=state.mandatory_guide_work,
            search_stop_override=state.search_stop_override,
            direct_pool=direct_pool,
            raw_guide_phi=state.raw_guide_phi,
            guided_initialization=state.guided_initialization,
            float64_recovery_status=str(state.float64_recovery_status),
            elapsed_seconds=float(state.cumulative_search_active_seconds),
        )

    def restore(self) -> SearchState:
        """Rebuild mutable runtime search state from the explicit payload."""

        from .search import DirectPoolState, LambdaRun, SearchState

        lambdas = {
            float(item.lambda_value): LambdaRun(
                lambda_value=float(item.lambda_value),
                attempts=list(item.attempts),
                retained=item.retained,
                partition_k=item.partition_k,
            )
            for item in self.lambda_runs
        }
        if len(lambdas) != len(self.lambda_runs):
            raise ValueError("Search checkpoint repeats a lambda run.")
        cache = dict(self.refit_cache)
        if len(cache) != len(self.refit_cache):
            raise ValueError("Search checkpoint repeats a refit-cache key.")
        if int(self.next_candidate_id) != len(self.candidates):
            raise ValueError("Search checkpoint candidate IDs are not continuation-safe.")
        for expected_id, record in enumerate(self.candidates):
            if int(record.candidate_id) != expected_id:
                raise ValueError("Search checkpoint candidate IDs are not contiguous.")
        direct = self.direct_pool
        return SearchState(
            controller=OnlineLambdaController.from_snapshot(self.controller),
            lambdas=lambdas,
            candidates=list(self.candidates),
            bic_refit_cache=cache,
            work_ledger=WorkLedger(self.work),
            mandatory_guide_work=self.guide_work,
            search_stop_override=self.search_stop_override,
            direct_pool=DirectPoolState(
                next_index=int(direct.next_index),
                complete=bool(direct.complete),
                stop_reason=direct.stop_reason,
                proposals=(
                    None if direct.proposals is None else list(direct.proposals)
                ),
                proposals_complete=bool(direct.proposals_complete),
                final_parent_next_index=int(direct.final_parent_next_index),
            ),
            raw_guide_phi=self.raw_guide_phi,
            guided_initialization=self.guided_initialization,
            float64_recovery_status=str(self.float64_recovery_status),
            cumulative_search_active_seconds=float(self.elapsed_seconds),
        )


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
    raise TypeError(f"Unsupported checkpoint-identity value: {type(value).__name__}")


def _fit_config_identity(config: FitConfig) -> dict[str, Any]:
    """Project the resolved fit configuration without dataclass reflection."""

    certificate = config.solver.certificate
    resources = config.solver.resources
    refit = config.selection.refit
    lambda_search = config.selection.lambda_search
    graph = config.graph.graph
    return {
        "major_prior": config.major_prior,
        "eps": config.eps,
        "runtime": {
            "device": config.runtime.device,
            "dtype": config.runtime.dtype,
            "fallback": config.runtime.fallback,
            "verbose": config.runtime.verbose,
        },
        "solver": {
            "outer_max_iter": config.solver.outer_max_iter,
            "inner_max_iter": config.solver.inner_max_iter,
            "tolerance": config.solver.tolerance,
            "objective_shape": config.solver.objective_shape,
            "certification_tolerance": config.solver.certification_tolerance,
            "use_backward_error_progress": config.solver.use_backward_error_progress,
            "stagnation_audit_patience": config.solver.stagnation_audit_patience,
            "certificate": {
                "max_iter": certificate.max_iter,
                "refinement_rounds": certificate.refinement_rounds,
                "column_tolerance_scale": certificate.column_tolerance_scale,
            },
            "resources": {
                "workset_max_bytes": resources.workset_max_bytes,
                "compressed_cache_max_bytes": resources.compressed_cache_max_bytes,
                "workset_add_batch": resources.workset_add_batch,
                "workset_max_expansions": resources.workset_max_expansions,
                "max_tumor_edge_pass_equivalents": (
                    resources.max_tumor_edge_pass_equivalents
                ),
                "max_partition_refit_objective_evaluations": (
                    resources.max_partition_refit_objective_evaluations
                ),
                "max_direct_partition_candidates": (
                    resources.max_direct_partition_candidates
                ),
            },
        },
        "selection": {
            "partition_tolerance": config.selection.partition_tolerance,
            "refit": {
                "tolerance": refit.tolerance,
                "max_iter": refit.max_iter,
                "mode": refit.mode,
                "grid_points": refit.grid_points,
                "local_steps": refit.local_steps,
            },
            "lambda_search": {
                "exploration_budget": lambda_search.exploration_budget,
                "refinement_budget": lambda_search.refinement_budget,
                "solver_retry_limit": lambda_search.solver_retry_limit,
                "no_progress_patience": lambda_search.no_progress_patience,
            },
        },
        "graph": {
            "graph": (
                None
                if graph is None
                else {
                    "edge_u": graph.edge_u,
                    "edge_v": graph.edge_v,
                    "edge_w": graph.edge_w,
                    "name": graph.name,
                    "degree_bound": graph.degree_bound,
                }
            ),
            "adaptive_weight_gamma": config.graph.adaptive_weight_gamma,
            "adaptive_weight_floor": config.graph.adaptive_weight_floor,
            "adaptive_weight_baseline": config.graph.adaptive_weight_baseline,
        },
        "profile_name": config.profile_name,
    }


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


def _inference_source_files(package_root: Path) -> tuple[Path, ...]:
    """Return only Python files shipped as the inference package."""

    files = list(package_root.glob("*.py"))
    for name in _INFERENCE_PACKAGE_DIRECTORIES:
        directory = package_root / name
        if directory.is_dir():
            files.extend(directory.rglob("*.py"))
    return tuple(sorted(set(files)))


def source_tree_fingerprint() -> str:
    """Hash installed inference sources so generated trees cannot alter resume."""

    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update(b"clipp2.source-tree.v2\0")
    for source in _inference_source_files(package_root):
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

    policy = PRODUCTION_SELECTION_POLICY
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tumor_data_fingerprint": tumor_data_fingerprint(data),
        "objective_spec_hash": str(objective_spec_hash),
        "original_graph_hash": str(original_graph_hash),
        "selection_policy_id": str(policy.policy_id),
        "selection_policy_hash": hashlib.sha256(
            policy.to_json().encode("utf-8")
        ).hexdigest(),
        "computation_profile": str(fit_config.profile_name),
        "resolved_fit_config_hash": _sha256_json(_fit_config_identity(fit_config)),
        "use_warm_starts": bool(use_warm_starts),
        "runtime_device_name": str(runtime_device_name),
        "runtime_dtype": str(runtime_dtype),
        "numerical_environment": _numerical_environment_metadata(
            runtime_device_name
        ),
        "software_version": str(__version__),
        "source_tree_fingerprint": source_tree_fingerprint(),
    }


def _contiguous_array(value: np.ndarray) -> np.ndarray:
    if value.dtype.hasobject:
        raise TypeError("Object arrays are not permitted in a search checkpoint.")
    if value.flags.c_contiguous:
        return value
    return np.ascontiguousarray(value)


def _array_content_digest(
    value: np.ndarray,
) -> str:
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
    *,
    digest: str | None = None,
) -> str:
    array = _contiguous_array(value)
    if digest is None:
        digest = _array_content_digest(array)
    incumbent = arrays.get(digest)
    if incumbent is not None and (
        incumbent.dtype != array.dtype
        or incumbent.shape != array.shape
    ):
        raise RuntimeError("Checkpoint array SHA-256 collision.")
    arrays.setdefault(digest, array)
    return digest


def _encode_primitive(
    value: Any,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return {"float": _float_token(float(value))}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        cache_key = ("numpy", id(value))
        digest = digest_cache.get(cache_key)
        if digest is None:
            digest = _array_content_digest(value)
            digest_cache[cache_key] = digest
        _register_array(value, arrays, digest=digest)
        return {"array": digest}
    if torch.is_tensor(value):
        dtype_name = str(value.dtype).removeprefix("torch.")
        if dtype_name not in _TORCH_DTYPES:
            raise TypeError(
                f"Tensor dtype is not supported by search checkpoints: {dtype_name}"
            )
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint8).numpy()
            byte_view = True
        else:
            array = tensor.numpy()
            byte_view = False
        cache_key = ("torch", id(value))
        digest = digest_cache.get(cache_key)
        if digest is None:
            digest = _array_content_digest(array)
            digest_cache[cache_key] = digest
        _register_array(array, arrays, digest=digest)
        return {
            "tensor": digest,
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "byte_view": byte_view,
        }
    if isinstance(value, tuple):
        return {
            "tuple": [
                _encode_primitive(item, arrays, digest_cache)
                for item in value
            ]
        }
    raise TypeError(
        "Checkpoint primitive must be scalar, tuple, NumPy array, or tensor; "
        f"observed {type(value).__name__}."
    )


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


def _require_keys(value: dict[str, Any], expected: set[str], surface: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"Malformed search-checkpoint {surface} keys: "
            f"expected={sorted(expected)}, observed={sorted(value)}"
        )


def _decode_primitive(
    encoded: Any,
    arrays: Any,
) -> Any:
    if encoded is None or isinstance(encoded, (bool, int, str)):
        return encoded
    if not isinstance(encoded, dict):
        raise ValueError("Malformed search-checkpoint manifest value.")
    if "float" in encoded:
        return _float(encoded, "primitive float")
    if "array" in encoded:
        _require_keys(encoded, {"array"}, "array")
        key = encoded["array"]
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Checkpoint array reference is malformed.")
        if key not in arrays:
            raise ValueError(f"Checkpoint array is missing: {key}")
        return arrays[key]
    if "tensor" in encoded:
        _require_keys(
            encoded,
            {"tensor", "dtype", "shape", "byte_view"},
            "tensor",
        )
        key = encoded["tensor"]
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("Checkpoint tensor reference is malformed.")
        if key not in arrays:
            raise ValueError(f"Checkpoint tensor is missing: {key}")
        dtype_name = encoded["dtype"]
        if not isinstance(dtype_name, str):
            raise ValueError("Checkpoint tensor dtype is malformed.")
        if dtype_name not in _TORCH_DTYPES:
            raise ValueError(f"Checkpoint tensor dtype is unsupported: {dtype_name}")
        if not isinstance(encoded["byte_view"], bool):
            raise ValueError("Checkpoint tensor byte_view flag must be boolean.")
        if encoded["byte_view"] != (dtype_name == "bfloat16"):
            raise ValueError("Checkpoint tensor byte-view metadata is inconsistent.")
        shape = encoded["shape"]
        if not isinstance(shape, list) or any(
            type(item) is not int or item < 0 for item in shape
        ):
            raise ValueError("Checkpoint tensor shape is malformed.")
        stored = arrays[key]
        expected_shape = tuple(int(item) for item in shape)
        expected_elements = int(np.prod(expected_shape, dtype=np.int64))
        dtype = _TORCH_DTYPES[dtype_name]
        if encoded["byte_view"]:
            if stored.dtype != np.dtype(np.uint8) or int(stored.size) != 2 * expected_elements:
                raise ValueError("Checkpoint bfloat16 tensor storage is inconsistent.")
            tensor = torch.from_numpy(np.array(stored, copy=True, order="C"))
            tensor = tensor.view(dtype)
        else:
            tensor = torch.from_numpy(np.array(stored, copy=True, order="C"))
            if tensor.dtype != dtype or tuple(stored.shape) != expected_shape:
                raise ValueError("Checkpoint tensor storage metadata is inconsistent.")
        try:
            return tensor.reshape(expected_shape)
        except RuntimeError as exc:
            raise ValueError("Checkpoint tensor shape is inconsistent.") from exc
    if "tuple" in encoded:
        _require_keys(encoded, {"tuple"}, "tuple")
        if not isinstance(encoded["tuple"], list):
            raise ValueError("Checkpoint tuple payload must be a list.")
        return tuple(
            _decode_primitive(item, arrays) for item in encoded["tuple"]
        )
    raise ValueError("Malformed search-checkpoint primitive tag.")


def _values(encoded: Any, size: int, surface: str) -> list[Any]:
    if not isinstance(encoded, list) or len(encoded) != size:
        raise ValueError(
            f"Malformed search-checkpoint {surface}: expected {size} fields."
        )
    return encoded


def _integer(value: Any, surface: str) -> int:
    if type(value) is not int:
        raise ValueError(f"Checkpoint {surface} must be an integer.")
    return value


def _boolean(value: Any, surface: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Checkpoint {surface} must be boolean.")
    return value


def _string(value: Any, surface: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Checkpoint {surface} must be a string.")
    return value


def _float(value: Any, surface: str) -> float:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {surface} must be a hexadecimal float.")
    _require_keys(value, {"float"}, surface)
    token = value["float"]
    if not isinstance(token, str):
        raise ValueError(f"Checkpoint {surface} float token is malformed.")
    try:
        result = float.fromhex(token)
    except ValueError as exc:
        raise ValueError(f"Checkpoint {surface} float token is malformed.") from exc
    if result.hex() != token:
        raise ValueError(f"Checkpoint {surface} float token is not canonical.")
    return result


def _ef(value: float) -> dict[str, str]:
    return {"float": _float_token(value)}


def _array(value: np.ndarray, arrays: Any) -> np.ndarray:
    decoded = _decode_primitive(value, arrays)
    if not isinstance(decoded, np.ndarray):
        raise ValueError("Checkpoint field must reference a NumPy array.")
    return decoded


def _tensor(value: Any, arrays: Any) -> torch.Tensor:
    decoded = _decode_primitive(value, arrays)
    if not torch.is_tensor(decoded):
        raise ValueError("Checkpoint field must reference a tensor.")
    return decoded


@dataclass(frozen=True, slots=True)
class _ValueCodec:
    """One field codec in the explicit, positional schema-v4 layout."""

    encode: Any
    decode: Any


@dataclass(frozen=True, slots=True)
class _PositionalLayout:
    """Declare one known checkpoint value without a runtime type registry.

    Record types are never written to the manifest and cannot be selected by
    input data.  The enclosing search schema chooses each layout explicitly;
    this helper only keeps field order and validation in one place.
    """

    value_type: type
    surface: str
    members: tuple[tuple[str, _ValueCodec], ...]

    def dump(
        self,
        value: Any,
        arrays: dict[str, np.ndarray] | None = None,
        digest_cache: dict[tuple[str, int], str] | None = None,
    ) -> list[Any]:
        if not isinstance(value, self.value_type):
            raise TypeError(
                f"Checkpoint {self.surface} must be {self.value_type.__name__}."
            )
        array_table = {} if arrays is None else arrays
        cache = {} if digest_cache is None else digest_cache
        return [
            codec.encode(
                getattr(value, name),
                array_table,
                cache,
                f"{self.surface} {name}",
            )
            for name, codec in self.members
        ]

    def load(self, encoded: Any, arrays: Any = None) -> Any:
        values = _values(encoded, len(self.members), self.surface)
        return self.value_type(
            **{
                name: codec.decode(item, arrays, f"{self.surface} {name}")
                for (name, codec), item in zip(self.members, values, strict=True)
            }
        )


def _plain_codec(validator: Any) -> _ValueCodec:
    def encode(value: Any, arrays: Any, cache: Any, surface: str) -> Any:
        del arrays, cache
        return validator(value, surface)

    def decode(value: Any, arrays: Any, surface: str) -> Any:
        del arrays
        return validator(value, surface)

    return _ValueCodec(encode=encode, decode=decode)


def _float_codec() -> _ValueCodec:
    def encode(value: Any, arrays: Any, cache: Any, surface: str) -> Any:
        del arrays, cache
        if not isinstance(value, (float, np.floating)):
            raise TypeError(f"Checkpoint {surface} must be a float.")
        return _ef(float(value))

    def decode(value: Any, arrays: Any, surface: str) -> float:
        del arrays
        return _float(value, surface)

    return _ValueCodec(encode=encode, decode=decode)


def _array_codec() -> _ValueCodec:
    def encode(
        value: Any,
        arrays: dict[str, np.ndarray],
        cache: dict[tuple[str, int], str],
        surface: str,
    ) -> Any:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Checkpoint {surface} must be a NumPy array.")
        return _encode_primitive(value, arrays, cache)

    def decode(value: Any, arrays: Any, surface: str) -> np.ndarray:
        del surface
        return _array(value, arrays)

    return _ValueCodec(encode=encode, decode=decode)


def _layout_codec(layout: _PositionalLayout) -> _ValueCodec:
    def encode(
        value: Any,
        arrays: dict[str, np.ndarray],
        cache: dict[tuple[str, int], str],
        surface: str,
    ) -> Any:
        del surface
        return layout.dump(value, arrays, cache)

    def decode(value: Any, arrays: Any, surface: str) -> Any:
        del surface
        return layout.load(value, arrays)

    return _ValueCodec(encode=encode, decode=decode)


def _function_codec(dump: Any, load: Any, *, stored: bool) -> _ValueCodec:
    def encode(value: Any, arrays: Any, cache: Any, surface: str) -> Any:
        del surface
        return dump(value, arrays, cache) if stored else dump(value)

    def decode(value: Any, arrays: Any, surface: str) -> Any:
        del surface
        return load(value, arrays) if stored else load(value)

    return _ValueCodec(encode=encode, decode=decode)


def _tuple_codec(codec: _ValueCodec) -> _ValueCodec:
    def encode(value: Any, arrays: Any, cache: Any, surface: str) -> list[Any]:
        if not isinstance(value, tuple):
            raise TypeError(f"Checkpoint {surface} must be a tuple.")
        return [codec.encode(item, arrays, cache, surface) for item in value]

    def decode(value: Any, arrays: Any, surface: str) -> tuple[Any, ...]:
        if not isinstance(value, list):
            raise ValueError(f"Checkpoint {surface} must be a list.")
        return tuple(codec.decode(item, arrays, surface) for item in value)

    return _ValueCodec(encode=encode, decode=decode)


def _optional_codec(codec: _ValueCodec) -> _ValueCodec:
    def encode(value: Any, arrays: Any, cache: Any, surface: str) -> Any:
        if value is None:
            return None
        return codec.encode(value, arrays, cache, surface)

    def decode(value: Any, arrays: Any, surface: str) -> Any:
        if value is None:
            return None
        return codec.decode(value, arrays, surface)

    return _ValueCodec(encode=encode, decode=decode)


_INTEGER_CODEC = _plain_codec(_integer)
_BOOLEAN_CODEC = _plain_codec(_boolean)
_STRING_CODEC = _plain_codec(_string)
_FLOAT_CODEC = _float_codec()
_ARRAY_CODEC = _array_codec()


_WORK_LAYOUT = _PositionalLayout(
    WorkCounters,
    "work counters",
    tuple(
        (name, _INTEGER_CODEC)
        for name in (
            "inner_iterations",
            "inner_stationarity_checks",
            "inner_full_kkt_audits",
            "outer_kkt_audits",
            "certificate_iterations",
            "certificate_full_graph_passes",
            "partition_refit_coordinates",
            "partition_refit_objective_evaluations",
            "edge_pass_equivalents",
            "edge_region_visits",
        )
    ),
)
_enc_work = _WORK_LAYOUT.dump
_dec_work = _WORK_LAYOUT.load

_BASE_OBJECTIVE_LAYOUT = _PositionalLayout(
    BaseObjectiveKey,
    "base objective",
    tuple(
        (name, _STRING_CODEC)
        for name in ("likelihood_hash", "graph_hash", "box_hash", "eps_hex")
    ),
)
_enc_base_objective = _BASE_OBJECTIVE_LAYOUT.dump
_dec_base_objective = _BASE_OBJECTIVE_LAYOUT.load

_LAMBDA_OBJECTIVE_LAYOUT = _PositionalLayout(
    LambdaObjectiveKey,
    "lambda objective",
    (
        ("base", _layout_codec(_BASE_OBJECTIVE_LAYOUT)),
        ("lambda_hex", _STRING_CODEC),
    ),
)
_enc_lambda_objective = _LAMBDA_OBJECTIVE_LAYOUT.dump
_dec_lambda_objective = _LAMBDA_OBJECTIVE_LAYOUT.load

_OBJECTIVE_LAYOUT = _PositionalLayout(
    ObjectiveValue,
    "objective value",
    (("total", _FLOAT_CODEC),),
)
_enc_objective = _OBJECTIVE_LAYOUT.dump
_dec_objective = _OBJECTIVE_LAYOUT.load

_KKT_LAYOUT = _PositionalLayout(
    KKTComponents,
    "KKT components",
    tuple(
        (name, _FLOAT_CODEC)
        for name in ("stationarity", "edge_subgradient", "dual_ball", "box")
    ),
)
_enc_kkt = _KKT_LAYOUT.dump
_dec_kkt = _KKT_LAYOUT.load


def _enc_edge_certificate(
    value: DenseEdgeCertificate | CompressedEdgeCertificate,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> dict[str, Any]:
    if isinstance(value, DenseEdgeCertificate):
        return {
            "kind": "dense",
            "value": [
                _encode_primitive(value.dual, arrays, digest_cache),
                value.graph_hash,
                value.gradient_scope,
                value.certificate_scope,
            ],
        }
    if isinstance(value, CompressedEdgeCertificate):
        return {
            "kind": "compressed",
            "value": [
                _encode_primitive(value.labels, arrays, digest_cache),
                _encode_primitive(value.centers, arrays, digest_cache),
                _encode_primitive(value.internal_edge_ids, arrays, digest_cache),
                _encode_primitive(value.internal_dual, arrays, digest_cache),
                value.graph_hash,
                value.gradient_scope,
                value.certificate_scope,
            ],
        }
    raise TypeError(f"Unsupported edge certificate: {type(value).__name__}.")


def _dec_edge_certificate(encoded: Any, arrays: Any):
    if not isinstance(encoded, dict):
        raise ValueError("Checkpoint edge certificate must be a mapping.")
    _require_keys(encoded, {"kind", "value"}, "edge certificate")
    kind = _string(encoded["kind"], "edge certificate kind")
    if kind == "dense":
        value = _values(encoded["value"], 4, "dense edge certificate")
        return DenseEdgeCertificate(
            dual=_tensor(value[0], arrays),
            graph_hash=_string(value[1], "certificate graph_hash"),
            gradient_scope=_string(value[2], "certificate gradient_scope"),
            certificate_scope=_string(value[3], "certificate scope"),
        )
    if kind == "compressed":
        value = _values(encoded["value"], 7, "compressed edge certificate")
        return CompressedEdgeCertificate(
            labels=_tensor(value[0], arrays),
            centers=_tensor(value[1], arrays),
            internal_edge_ids=_tensor(value[2], arrays),
            internal_dual=_tensor(value[3], arrays),
            graph_hash=_string(value[4], "certificate graph_hash"),
            gradient_scope=_string(value[5], "certificate gradient_scope"),
            certificate_scope=_string(value[6], "certificate scope"),
        )
    raise ValueError(f"Checkpoint edge certificate kind is unsupported: {kind}")


def _enc_certificate(
    value: CertificateResult,
) -> list[Any]:
    if value.witness is not None:
        raise TypeError("Search checkpoints require a compact certificate witness.")
    return [
        _enc_kkt(value.components),
        value.certified,
        value.admissible,
        value.global_optimum,
        value.status,
        _ef(value.tolerance),
        value.scope,
        value.gradient_scope,
        value.directional_admissible,
        _ef(value.working_residual),
        value.working_dtype,
        value.audit_dtype,
        value.precision_polished,
        _ef(value.precision_polish_delta),
        value.residual_method,
        value.fallback_reason,
    ]


def _dec_certificate(encoded: Any) -> CertificateResult:
    value = _values(encoded, 16, "certificate result")
    return CertificateResult(
        components=_dec_kkt(value[0]),
        certified=_boolean(value[1], "certificate certified"),
        admissible=_boolean(value[2], "certificate admissible"),
        global_optimum=_boolean(value[3], "certificate global_optimum"),
        status=_string(value[4], "certificate status"),
        tolerance=_float(value[5], "certificate tolerance"),
        scope=_string(value[6], "certificate scope"),
        gradient_scope=_string(value[7], "certificate gradient_scope"),
        directional_admissible=_boolean(
            value[8], "certificate directional_admissible"
        ),
        witness=None,
        working_residual=_float(value[9], "certificate working_residual"),
        working_dtype=_string(value[10], "certificate working_dtype"),
        audit_dtype=_string(value[11], "certificate audit_dtype"),
        precision_polished=_boolean(value[12], "certificate precision_polished"),
        precision_polish_delta=_float(
            value[13], "certificate precision_polish_delta"
        ),
        residual_method=_string(value[14], "certificate residual_method"),
        fallback_reason=_string(value[15], "certificate fallback_reason"),
    )


_CONVERGENCE_LAYOUT = _PositionalLayout(
    ConvergenceResult,
    "convergence result",
    (
        ("converged", _BOOLEAN_CODEC),
        ("mm_consistency_violations", _INTEGER_CODEC),
        ("stage_outer_iterations", _INTEGER_CODEC),
        ("stage_outer_max_iter", _INTEGER_CODEC),
        ("stage_inner_iterations", _INTEGER_CODEC),
        ("stage_inner_max_iter", _INTEGER_CODEC),
        ("stage_inner_solve_calls", _INTEGER_CODEC),
        ("stop_reason", _STRING_CODEC),
        ("progress_residual_method", _STRING_CODEC),
        ("solve_tolerance", _FLOAT_CODEC),
        ("legacy_stop_kkt_residual", _FLOAT_CODEC),
        ("componentwise_stop_kkt_residual", _FLOAT_CODEC),
        ("accepted_full_steps", _INTEGER_CODEC),
        ("accepted_damped_steps", _INTEGER_CODEC),
        ("rejected_outer_steps", _INTEGER_CODEC),
    ),
)
_enc_convergence = _CONVERGENCE_LAYOUT.dump
_dec_convergence = _CONVERGENCE_LAYOUT.load

_PROVENANCE_LAYOUT = _PositionalLayout(
    FitProvenance,
    "fit provenance",
    (
        ("objective_key", _layout_codec(_LAMBDA_OBJECTIVE_LAYOUT)),
        ("device", _STRING_CODEC),
        ("dtype", _STRING_CODEC),
        ("inner_solver", _STRING_CODEC),
        ("global_optimality_basis", _STRING_CODEC),
        ("likelihood_eps", _FLOAT_CODEC),
    ),
)
_enc_provenance = _PROVENANCE_LAYOUT.dump
_dec_provenance = _PROVENANCE_LAYOUT.load


def _enc_warm_state(
    value: DenseWarmState | PrimalOnlyWarmState,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> dict[str, Any]:
    if isinstance(value, DenseWarmState):
        return {
            "kind": "dense",
            "value": [
                _encode_primitive(value.phi, arrays, digest_cache),
                None
                if value.dual is None
                else _encode_primitive(value.dual, arrays, digest_cache),
                _ef(value.previous_lambda),
                value.graph_hash,
            ],
        }
    if isinstance(value, PrimalOnlyWarmState):
        return {
            "kind": "primal_only",
            "value": [
                _encode_primitive(value.phi, arrays, digest_cache),
                None
                if value.structure_hint is None
                else _encode_primitive(value.structure_hint, arrays, digest_cache),
                None
                if value.certificate_hint is None
                else _enc_edge_certificate(
                    value.certificate_hint,
                    arrays,
                    digest_cache,
                ),
            ],
        }
    raise TypeError(f"Unsupported warm state: {type(value).__name__}.")


def _dec_warm_state(encoded: Any, arrays: Any):
    if not isinstance(encoded, dict):
        raise ValueError("Checkpoint warm state must be a mapping.")
    _require_keys(encoded, {"kind", "value"}, "warm state")
    kind = _string(encoded["kind"], "warm-state kind")
    if kind == "dense":
        value = _values(encoded["value"], 4, "dense warm state")
        return DenseWarmState(
            phi=_tensor(value[0], arrays),
            dual=None if value[1] is None else _tensor(value[1], arrays),
            previous_lambda=_float(value[2], "warm previous_lambda"),
            graph_hash=_string(value[3], "warm graph_hash"),
        )
    if kind == "primal_only":
        value = _values(encoded["value"], 3, "primal-only warm state")
        return PrimalOnlyWarmState(
            phi=_tensor(value[0], arrays),
            structure_hint=(
                None if value[1] is None else _tensor(value[1], arrays)
            ),
            certificate_hint=(
                None
                if value[2] is None
                else _dec_edge_certificate(value[2], arrays)
            ),
        )
    raise ValueError(f"Checkpoint warm-state kind is unsupported: {kind}")


def _enc_solver_state(
    value: SolverState,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> list[Any]:
    return [
        _encode_primitive(value.phi, arrays, digest_cache),
        None
        if value.dual is None
        else _encode_primitive(value.dual, arrays, digest_cache),
        _ef(value.previous_lambda),
        None
        if value.warm_state is None
        else _enc_warm_state(value.warm_state, arrays, digest_cache),
        None
        if value.certificate is None
        else _enc_edge_certificate(value.certificate, arrays, digest_cache),
        value.objective_spec_hash,
    ]


def _dec_solver_state(encoded: Any, arrays: Any) -> SolverState:
    value = _values(encoded, 6, "solver state")
    return SolverState(
        phi=_tensor(value[0], arrays),
        dual=None if value[1] is None else _tensor(value[1], arrays),
        previous_lambda=_float(value[2], "solver previous_lambda"),
        warm_state=(
            None if value[3] is None else _dec_warm_state(value[3], arrays)
        ),
        certificate=(
            None
            if value[4] is None
            else _dec_edge_certificate(value[4], arrays)
        ),
        objective_spec_hash=_string(value[5], "solver objective_spec_hash"),
    )


def _enc_raw_fit(
    value: RawFit,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> list[Any]:
    if value.state is not None:
        raise TypeError("Search checkpoints require RawFit.state to be detached.")
    return [
        _encode_primitive(value.phi, arrays, digest_cache),
        _enc_objective(value.objective),
        _enc_certificate(value.certificate),
        _enc_convergence(value.convergence),
        _enc_work(value.work),
        _enc_provenance(value.provenance),
    ]


def _dec_raw_fit(encoded: Any, arrays: Any) -> RawFit:
    value = _values(encoded, 6, "raw fit")
    return RawFit(
        phi=_array(value[0], arrays),
        objective=_dec_objective(value[1]),
        certificate=_dec_certificate(value[2]),
        convergence=_dec_convergence(value[3]),
        work=_dec_work(value[4]),
        state=None,
        provenance=_dec_provenance(value[5]),
    )


_RAW_FIT_CODEC = _function_codec(_enc_raw_fit, _dec_raw_fit, stored=True)
_SOLVER_STATE_CODEC = _function_codec(
    _enc_solver_state,
    _dec_solver_state,
    stored=True,
)
_SOLVE_OUTCOME_LAYOUT = _PositionalLayout(
    SolveOutcome,
    "solve outcome",
    (
        ("fit", _RAW_FIT_CODEC),
        ("state", _optional_codec(_SOLVER_STATE_CODEC)),
    ),
)
_enc_solve_outcome = _SOLVE_OUTCOME_LAYOUT.dump
_dec_solve_outcome = _SOLVE_OUTCOME_LAYOUT.load


def _primitive_tuple(encoded: Any, arrays: Any, surface: str) -> tuple[Any, ...]:
    value = _decode_primitive(encoded, arrays)
    if not isinstance(value, tuple):
        raise ValueError(f"Checkpoint {surface} must be a tuple.")
    return value


def _string_tuple(encoded: Any, arrays: Any, surface: str) -> tuple[str, ...]:
    value = _primitive_tuple(encoded, arrays, surface)
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"Checkpoint {surface} must contain only strings.")
    return value


def _string_tuple_codec() -> _ValueCodec:
    def encode(
        value: Any,
        arrays: dict[str, np.ndarray],
        cache: dict[tuple[str, int], str],
        surface: str,
    ) -> Any:
        if not isinstance(value, tuple) or any(
            not isinstance(item, str) for item in value
        ):
            raise TypeError(f"Checkpoint {surface} must be a tuple of strings.")
        return _encode_primitive(value, arrays, cache)

    return _ValueCodec(encode=encode, decode=_string_tuple)


_STRING_TUPLE_CODEC = _string_tuple_codec()
_OPTIONAL_INTEGER_CODEC = _optional_codec(_INTEGER_CODEC)
_OPTIONAL_FLOAT_CODEC = _optional_codec(_FLOAT_CODEC)
_OPTIONAL_ARRAY_CODEC = _optional_codec(_ARRAY_CODEC)

_FUSION_PARTITION_LAYOUT = _PositionalLayout(
    FusionPartition,
    "fusion partition",
    (
        ("labels", _ARRAY_CODEC),
        ("signature", _STRING_CODEC),
        ("certified", _BOOLEAN_CODEC),
        ("source", _STRING_CODEC),
        ("certification_failure_reason", _STRING_CODEC),
        ("mutation_ids", _STRING_TUPLE_CODEC),
    ),
)
_enc_fusion_partition = _FUSION_PARTITION_LAYOUT.dump
_dec_fusion_partition = _FUSION_PARTITION_LAYOUT.load

_DIRECT_PARTITION_LAYOUT = _PositionalLayout(
    DirectPartition,
    "direct partition",
    (
        ("labels", _ARRAY_CODEC),
        ("signature", _STRING_CODEC),
        ("source", _STRING_CODEC),
        ("mutation_ids", _STRING_TUPLE_CODEC),
        ("parent_raw_candidate_id", _OPTIONAL_INTEGER_CODEC),
        ("parent_raw_lambda", _OPTIONAL_FLOAT_CODEC),
        ("parent_raw_phi_hash", _STRING_CODEC),
    ),
)
_enc_direct_partition = _DIRECT_PARTITION_LAYOUT.dump
_dec_direct_partition = _DIRECT_PARTITION_LAYOUT.load


_PARTITION_FIT_LAYOUT = _PositionalLayout(
    PartitionFit,
    "partition fit",
    (
        ("labels", _ARRAY_CODEC),
        ("phi", _ARRAY_CODEC),
        ("cluster_centers", _ARRAY_CODEC),
        ("loglik", _FLOAT_CODEC),
        ("finite_candidate_found", _BOOLEAN_CODEC),
        ("observed_model_hash", _STRING_CODEC),
        ("observed_likelihood_hash", _STRING_CODEC),
        ("reporting_model_hash", _STRING_CODEC),
        ("observed_box_hash", _STRING_CODEC),
        ("likelihood_eps_hex", _STRING_CODEC),
        ("global_optimum_certified", _BOOLEAN_CODEC),
        ("partition_signature", _STRING_CODEC),
        ("refit_numerically_resolved", _BOOLEAN_CODEC),
        ("fit_loss", _FLOAT_CODEC),
        ("n_clusters", _INTEGER_CODEC),
        ("boundary_count", _INTEGER_CODEC),
        ("active_degrees_of_freedom", _INTEGER_CODEC),
        ("refit_coordinate_count", _INTEGER_CODEC),
        ("refit_finite_coordinate_count", _INTEGER_CODEC),
        ("refit_total_grid_points", _INTEGER_CODEC),
        ("refit_max_grid_spacing", _FLOAT_CODEC),
        ("refit_total_candidate_basins", _INTEGER_CODEC),
        ("refit_total_refined_candidates", _INTEGER_CODEC),
        ("refit_min_best_second_loss_gap", _FLOAT_CODEC),
        ("loglik_source", _STRING_CODEC),
        ("global_lower_bound", _FLOAT_CODEC),
        ("global_optimality_gap", _FLOAT_CODEC),
        ("global_certificate_method", _STRING_CODEC),
        ("global_certificate_intervals", _INTEGER_CODEC),
        ("refit_mode", _STRING_CODEC),
        ("coordinate_argmin_lower", _OPTIONAL_ARRAY_CODEC),
        ("coordinate_argmin_upper", _OPTIONAL_ARRAY_CODEC),
        ("coordinate_statistically_identified", _OPTIONAL_ARRAY_CODEC),
        ("refit_objective_evaluations", _INTEGER_CODEC),
    ),
)
_enc_partition_fit = _PARTITION_FIT_LAYOUT.dump
_dec_partition_fit = _PARTITION_FIT_LAYOUT.load


_SCORE_LAYOUT = _PositionalLayout(
    SelectionScore,
    "selection score",
    (
        ("name", _STRING_CODEC),
        ("value", _FLOAT_CODEC),
        ("loglik", _FLOAT_CODEC),
        ("penalty", _FLOAT_CODEC),
        ("degrees_of_freedom", _INTEGER_CODEC),
        ("n_eff", _INTEGER_CODEC),
        ("partition_signature", _STRING_CODEC),
        ("numerical_uncertainty", _FLOAT_CODEC),
    ),
)
_enc_score = _SCORE_LAYOUT.dump
_dec_score = _SCORE_LAYOUT.load

_ATTEMPT_LIMITS_LAYOUT = _PositionalLayout(
    AttemptLimits,
    "attempt limits",
    tuple(
        (name, _INTEGER_CODEC)
        for name in ("outer_max_iter", "inner_max_iter", "certificate_max_iter")
    ),
)
_enc_attempt_limits = _ATTEMPT_LIMITS_LAYOUT.dump
_dec_attempt_limits = _ATTEMPT_LIMITS_LAYOUT.load


_FIT_AUDIT_LAYOUT = _PositionalLayout(
    FitAuditSummary,
    "fit audit",
    (
        ("objective", _layout_codec(_OBJECTIVE_LAYOUT)),
        (
            "certificate",
            _function_codec(_enc_certificate, _dec_certificate, stored=False),
        ),
        ("convergence", _layout_codec(_CONVERGENCE_LAYOUT)),
        ("work", _layout_codec(_WORK_LAYOUT)),
        ("provenance", _layout_codec(_PROVENANCE_LAYOUT)),
    ),
)
_enc_fit_audit = _FIT_AUDIT_LAYOUT.dump
_dec_fit_audit = _FIT_AUDIT_LAYOUT.load

_RAW_ATTEMPT_LAYOUT = _PositionalLayout(
    RawAttemptSummary,
    "raw attempt",
    (
        ("source", _STRING_CODEC),
        ("start_value", _FLOAT_CODEC),
        ("breakpoint_escape_changed_count", _INTEGER_CODEC),
        ("mathematically_certified", _BOOLEAN_CODEC),
        ("limits", _layout_codec(_ATTEMPT_LIMITS_LAYOUT)),
        ("fit", _layout_codec(_FIT_AUDIT_LAYOUT)),
        ("promotion_status", _STRING_CODEC),
    ),
)
_enc_raw_attempt = _RAW_ATTEMPT_LAYOUT.dump
_dec_raw_attempt = _RAW_ATTEMPT_LAYOUT.load


_CANDIDATE_TRACE_LAYOUT = _PositionalLayout(
    CandidateTrace,
    "candidate trace",
    (
        ("search_round", _INTEGER_CODEC),
        ("search_phase", _STRING_CODEC),
        ("start_source", _STRING_CODEC),
        ("start_value", _OPTIONAL_FLOAT_CODEC),
        ("breakpoint_escape_changed_count", _INTEGER_CODEC),
        ("raw_attempts", _tuple_codec(_layout_codec(_RAW_ATTEMPT_LAYOUT))),
    ),
)
_enc_candidate_trace = _CANDIDATE_TRACE_LAYOUT.dump
_dec_candidate_trace = _CANDIDATE_TRACE_LAYOUT.load


def _enc_candidate(
    value: RawFusionCandidate
    | UnscoredRawFusionCandidate
    | DirectPartitionCandidate,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> dict[str, Any]:
    if isinstance(value, RawFusionCandidate):
        return {
            "kind": "raw_scored",
            "value": [
                _enc_raw_fit(value.raw_fit, arrays, digest_cache),
                _enc_fusion_partition(value.partition, arrays, digest_cache),
                _enc_partition_fit(value.refit, arrays, digest_cache),
                _enc_score(value.score),
                value.eligible_for_selection,
                value.ineligibility_reason,
                _enc_work(value.work),
            ],
        }
    if isinstance(value, UnscoredRawFusionCandidate):
        return {
            "kind": "raw_unscored",
            "value": [
                _enc_raw_fit(value.raw_fit, arrays, digest_cache),
                _enc_fusion_partition(value.partition, arrays, digest_cache),
                value.ineligibility_reason,
                _enc_work(value.work),
            ],
        }
    if isinstance(value, DirectPartitionCandidate):
        return {
            "kind": "direct",
            "value": [
                _enc_direct_partition(value.partition, arrays, digest_cache),
                _enc_partition_fit(value.refit, arrays, digest_cache),
                _enc_score(value.score),
                value.eligible_for_selection,
                value.ineligibility_reason,
                _enc_work(value.work),
            ],
        }
    raise TypeError(f"Unsupported search candidate: {type(value).__name__}.")


def _dec_candidate(encoded: Any, arrays: Any):
    if not isinstance(encoded, dict):
        raise ValueError("Checkpoint candidate must be a mapping.")
    _require_keys(encoded, {"kind", "value"}, "candidate")
    kind = _string(encoded["kind"], "candidate kind")
    if kind == "raw_scored":
        value = _values(encoded["value"], 7, "scored raw candidate")
        return RawFusionCandidate(
            raw_fit=_dec_raw_fit(value[0], arrays),
            partition=_dec_fusion_partition(value[1], arrays),
            refit=_dec_partition_fit(value[2], arrays),
            score=_dec_score(value[3]),
            eligible_for_selection=_boolean(value[4], "candidate eligible"),
            ineligibility_reason=_string(value[5], "candidate reason"),
            work=_dec_work(value[6]),
        )
    if kind == "raw_unscored":
        value = _values(encoded["value"], 4, "unscored raw candidate")
        return UnscoredRawFusionCandidate(
            raw_fit=_dec_raw_fit(value[0], arrays),
            partition=_dec_fusion_partition(value[1], arrays),
            ineligibility_reason=_string(value[2], "candidate reason"),
            work=_dec_work(value[3]),
        )
    if kind == "direct":
        value = _values(encoded["value"], 6, "direct candidate")
        return DirectPartitionCandidate(
            partition=_dec_direct_partition(value[0], arrays),
            refit=_dec_partition_fit(value[1], arrays),
            score=_dec_score(value[2]),
            eligible_for_selection=_boolean(value[3], "candidate eligible"),
            ineligibility_reason=_string(value[4], "candidate reason"),
            work=_dec_work(value[5]),
        )
    raise ValueError(f"Checkpoint candidate kind is unsupported: {kind}")


_CANDIDATE_RECORD_LAYOUT = _PositionalLayout(
    CandidateRecord,
    "candidate record",
    (
        ("candidate_id", _INTEGER_CODEC),
        ("candidate", _function_codec(_enc_candidate, _dec_candidate, stored=True)),
        ("trace", _layout_codec(_CANDIDATE_TRACE_LAYOUT)),
    ),
)
_enc_candidate_record = _CANDIDATE_RECORD_LAYOUT.dump
_dec_candidate_record = _CANDIDATE_RECORD_LAYOUT.load


_PARTITION_CANDIDATE_LAYOUT = _PositionalLayout(
    PartitionCandidate,
    "partition candidate",
    (
        ("labels", _ARRAY_CODEC),
        ("K", _INTEGER_CODEC),
        ("source", _STRING_CODEC),
        ("phi_start", _ARRAY_CODEC),
        ("fit_loss", _FLOAT_CODEC),
        ("bic", _FLOAT_CODEC),
        ("finite_candidate_found", _BOOLEAN_CODEC),
        ("requested_k", _OPTIONAL_INTEGER_CODEC),
    ),
)
_enc_partition_candidate = _PARTITION_CANDIDATE_LAYOUT.dump
_dec_partition_candidate = _PARTITION_CANDIDATE_LAYOUT.load


_ONLINE_CONFIG_LAYOUT = _PositionalLayout(
    OnlineLambdaConfig,
    "online lambda config",
    (
        ("guide_n_clusters", _INTEGER_CODEC),
        ("num_mutations", _INTEGER_CODEC),
        ("kkt_tolerance", _FLOAT_CODEC),
        ("lambda_min", _FLOAT_CODEC),
        ("lambda_max", _FLOAT_CODEC),
        ("transition_log10_width_tolerance", _FLOAT_CODEC),
        ("score_relative_tolerance", _FLOAT_CODEC),
        ("max_unique_lambdas", _INTEGER_CODEC),
        ("max_refinement_lambdas", _INTEGER_CODEC),
        ("max_solver_retries_per_lambda", _INTEGER_CODEC),
        ("max_bootstrap_anchor_lambdas", _INTEGER_CODEC),
        ("no_progress_patience", _INTEGER_CODEC),
    ),
)
_enc_online_config = _ONLINE_CONFIG_LAYOUT.dump
_dec_online_config = _ONLINE_CONFIG_LAYOUT.load

_ONLINE_OBSERVATION_LAYOUT = _PositionalLayout(
    OnlineLambdaObservation,
    "online lambda observation",
    (
        ("lambda_value", _FLOAT_CODEC),
        ("n_clusters", _INTEGER_CODEC),
        ("partition_signature", _STRING_CODEC),
        ("partition_bic", _FLOAT_CODEC),
        ("kkt_residual", _FLOAT_CODEC),
        ("raw_objective_certified", _BOOLEAN_CODEC),
        ("partition_certified", _BOOLEAN_CODEC),
        ("selection_score_available", _BOOLEAN_CODEC),
        ("score_numerical_uncertainty", _FLOAT_CODEC),
        ("degrees_of_freedom", _INTEGER_CODEC),
    ),
)
_enc_online_observation = _ONLINE_OBSERVATION_LAYOUT.dump
_dec_online_observation = _ONLINE_OBSERVATION_LAYOUT.load

_ONLINE_PROPOSAL_LAYOUT = _PositionalLayout(
    OnlineLambdaProposal,
    "online lambda proposal",
    (
        ("lambda_value", _FLOAT_CODEC),
        ("phase", _STRING_CODEC),
        ("reason", _STRING_CODEC),
        ("warm_start_lambda", _OPTIONAL_FLOAT_CODEC),
        ("alternate_start_lambda", _OPTIONAL_FLOAT_CODEC),
        ("bracket_left_lambda", _OPTIONAL_FLOAT_CODEC),
        ("bracket_right_lambda", _OPTIONAL_FLOAT_CODEC),
        ("retry_number", _INTEGER_CODEC),
    ),
)
_enc_online_proposal = _ONLINE_PROPOSAL_LAYOUT.dump
_dec_online_proposal = _ONLINE_PROPOSAL_LAYOUT.load


def _enc_online_state(value: OnlineLambdaState) -> list[Any]:
    return [
        value.version,
        _enc_online_config(value.config),
        _ef(value.initial_lambda),
        value.initial_reason,
        [
            [_ef(key), _enc_online_observation(observation)]
            for key, observation in value.certified
        ],
        None
        if value.last_observation is None
        else _enc_online_observation(value.last_observation),
        None if value.retry_key is None else _ef(value.retry_key),
        [
            [_ef(key), _ef(lambda_value), phase, count]
            for key, lambda_value, phase, count in value.attempts
        ],
        [_enc_online_proposal(item) for item in value.proposal_history],
        [_ef(item) for item in value.solver_recovery_keys],
        [_ef(item) for item in value.bootstrap_anchor_keys],
        [_ef(item) for item in value.uncertified_exhausted_keys],
        value.no_progress_streak,
        value.stop_reason,
    ]


def _dec_online_state(encoded: Any) -> OnlineLambdaState:
    value = _values(encoded, 14, "online lambda state")
    certified = value[4]
    attempts = value[7]
    proposals = value[8]
    recovery = value[9]
    bootstrap = value[10]
    exhausted = value[11]
    for name, items in (
        ("certified observations", certified),
        ("attempts", attempts),
        ("proposal history", proposals),
        ("solver recovery keys", recovery),
        ("bootstrap anchor keys", bootstrap),
        ("uncertified exhausted keys", exhausted),
    ):
        if not isinstance(items, list):
            raise ValueError(f"Checkpoint online {name} must be a list.")

    certified_values: list[tuple[float, OnlineLambdaObservation]] = []
    for item in certified:
        pair = _values(item, 2, "certified observation pair")
        certified_values.append(
            (
                _float(pair[0], "certified observation key"),
                _dec_online_observation(pair[1]),
            )
        )
    attempt_values: list[tuple[float, float, str, int]] = []
    for item in attempts:
        attempt = _values(item, 4, "online lambda attempt")
        attempt_values.append(
            (
                _float(attempt[0], "attempt key"),
                _float(attempt[1], "attempt lambda"),
                _string(attempt[2], "attempt phase"),
                _integer(attempt[3], "attempt count"),
            )
        )
    return OnlineLambdaState(
        version=_integer(value[0], "online state version"),
        config=_dec_online_config(value[1]),
        initial_lambda=_float(value[2], "online initial_lambda"),
        initial_reason=_string(value[3], "online initial_reason"),
        certified=tuple(certified_values),
        last_observation=(
            None
            if value[5] is None
            else _dec_online_observation(value[5])
        ),
        retry_key=(
            None if value[6] is None else _float(value[6], "online retry_key")
        ),
        attempts=tuple(attempt_values),
        proposal_history=tuple(_dec_online_proposal(item) for item in proposals),
        solver_recovery_keys=tuple(
            _float(item, "solver recovery key") for item in recovery
        ),
        bootstrap_anchor_keys=tuple(
            _float(item, "bootstrap anchor key") for item in bootstrap
        ),
        uncertified_exhausted_keys=tuple(
            _float(item, "uncertified exhausted key") for item in exhausted
        ),
        no_progress_streak=_integer(value[12], "online no_progress_streak"),
        stop_reason=(
            None if value[13] is None else _string(value[13], "online stop_reason")
        ),
    )


_GUIDED_DIAGNOSTICS_LAYOUT = _PositionalLayout(
    GuidedFusionDiagnostics,
    "guided-fusion diagnostics",
    (
        ("lambda_value", _FLOAT_CODEC),
        ("required_lambda_without_between_edges", _FLOAT_CODEC),
        ("numerical_lambda_floor", _FLOAT_CODEC),
        ("capacity_iterations", _INTEGER_CODEC),
        ("capacity_converged", _BOOLEAN_CODEC),
        ("capacity_status", _STRING_CODEC),
        ("num_mutations", _INTEGER_CODEC),
        ("num_regions", _INTEGER_CODEC),
        ("num_clusters", _INTEGER_CODEC),
        ("within_edge_count", _INTEGER_CODEC),
        ("between_edge_count", _INTEGER_CODEC),
        ("zero_separation_between_edge_count", _INTEGER_CODEC),
        ("gradient_source", _STRING_CODEC),
        ("guide_adjustment_max_abs", _FLOAT_CODEC),
        ("max_within_cluster_deviation", _FLOAT_CODEC),
        ("block_flow_balance_max_abs", _FLOAT_CODEC),
        ("max_dual_ball_ratio", _FLOAT_CODEC),
        ("max_within_dual_ball_ratio", _FLOAT_CODEC),
        ("max_between_dual_ball_ratio", _FLOAT_CODEC),
        ("kkt_residual", _FLOAT_CODEC),
        ("stationarity_residual", _FLOAT_CODEC),
        ("edge_subgradient_residual", _FLOAT_CODEC),
        ("dual_ball_residual", _FLOAT_CODEC),
        ("box_residual", _FLOAT_CODEC),
        ("num_exact_lower_active_coordinates", _INTEGER_CODEC),
        ("num_exact_upper_active_coordinates", _INTEGER_CODEC),
        ("num_exact_frozen_coordinates", _INTEGER_CODEC),
    ),
)
_enc_guided_diagnostics = _GUIDED_DIAGNOSTICS_LAYOUT.dump
_dec_guided_diagnostics = _GUIDED_DIAGNOSTICS_LAYOUT.load


_GUIDED_INITIALIZATION_LAYOUT = _PositionalLayout(
    GuidedFusionInitialization,
    "guided-fusion initialization",
    (
        ("lambda_value", _FLOAT_CODEC),
        ("solver_state", _SOLVER_STATE_CODEC),
        ("diagnostics", _layout_codec(_GUIDED_DIAGNOSTICS_LAYOUT)),
    ),
)
_enc_guided_initialization = _GUIDED_INITIALIZATION_LAYOUT.dump
_dec_guided_initialization = _GUIDED_INITIALIZATION_LAYOUT.load


_SOLVE_OUTCOME_CODEC = _layout_codec(_SOLVE_OUTCOME_LAYOUT)
_LAMBDA_CHECKPOINT_LAYOUT = _PositionalLayout(
    LambdaCheckpoint,
    "lambda checkpoint",
    (
        ("lambda_value", _FLOAT_CODEC),
        ("attempts", _tuple_codec(_SOLVE_OUTCOME_CODEC)),
        ("retained", _optional_codec(_SOLVE_OUTCOME_CODEC)),
        ("partition_k", _OPTIONAL_INTEGER_CODEC),
    ),
)
_enc_lambda_checkpoint = _LAMBDA_CHECKPOINT_LAYOUT.dump
_dec_lambda_checkpoint = _LAMBDA_CHECKPOINT_LAYOUT.load


def _enc_direct_pool(
    value: DirectPoolCheckpoint,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> list[Any]:
    proposals = None
    if value.proposals is not None:
        proposals = [
            [
                _enc_partition_candidate(proposal.candidate, arrays, digest_cache),
                proposal.stage,
                proposal.parent_raw_candidate_id,
            ]
            for proposal in value.proposals
        ]
    return [
        value.next_index,
        value.complete,
        value.stop_reason,
        proposals,
        value.proposals_complete,
        value.final_parent_next_index,
    ]


def _dec_direct_pool(encoded: Any, arrays: Any) -> DirectPoolCheckpoint:
    value = _values(encoded, 6, "direct pool")
    encoded_proposals = value[3]
    proposals = None
    if encoded_proposals is not None:
        if not isinstance(encoded_proposals, list):
            raise ValueError("Checkpoint direct-pool proposals must be a list.")
        decoded_proposals = []
        for encoded_proposal in encoded_proposals:
            proposal = _values(encoded_proposal, 3, "direct-pool proposal")
            decoded_proposals.append(
                DirectProposal(
                    candidate=_dec_partition_candidate(proposal[0], arrays),
                    stage=_string(proposal[1], "direct proposal source"),
                    parent_raw_candidate_id=(
                        None
                        if proposal[2] is None
                        else _integer(proposal[2], "direct proposal parent_id")
                    ),
                )
            )
        proposals = tuple(decoded_proposals)
    return DirectPoolCheckpoint(
        next_index=_integer(value[0], "direct_pool next_index"),
        complete=_boolean(value[1], "direct_pool complete"),
        stop_reason=(
            None if value[2] is None else _string(value[2], "direct_pool stop_reason")
        ),
        proposals=proposals,
        proposals_complete=_boolean(value[4], "direct_pool proposals_complete"),
        final_parent_next_index=_integer(
            value[5], "direct_pool final_parent_next_index"
        ),
    )


def _canonical_hex_string(value: Any, surface: str) -> str:
    token = _string(value, surface)
    try:
        number = float.fromhex(token)
    except ValueError as exc:
        raise ValueError(f"Checkpoint {surface} is not a hexadecimal float.") from exc
    if number.hex() != token:
        raise ValueError(f"Checkpoint {surface} is not canonical.")
    return token


def _digest_string(value: Any, surface: str) -> str:
    digest = _string(value, surface)
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Checkpoint {surface} is not a SHA-256 digest.")
    return digest


def _validate_refit_cache_identity(
    key: PartitionRefitKey,
    fit: PartitionFit,
) -> None:
    expected = (
        fit.partition_signature,
        fit.observed_model_hash,
        fit.observed_likelihood_hash,
        fit.reporting_model_hash,
        fit.observed_box_hash,
        fit.likelihood_eps_hex,
        fit.refit_mode,
    )
    observed = (
        key.partition_signature,
        key.observed_model_hash,
        key.observed_likelihood_hash,
        key.reporting_model_hash,
        key.observed_box_hash,
        key.likelihood_eps_hex,
        key.refit_mode,
    )
    if observed != expected:
        raise ValueError("Checkpoint refit-cache key does not identify its fit.")


def _enc_refit_cache_key(value: PartitionRefitKey) -> list[Any]:
    if not isinstance(value, PartitionRefitKey):
        raise TypeError("Search refit-cache keys must be PartitionRefitKey.")
    result = [
        PARTITION_REFIT_KEY_SCHEMA,
        value.partition_signature,
        value.observed_model_hash,
        value.observed_likelihood_hash,
        value.reporting_model_hash,
        value.observed_box_hash,
        value.likelihood_eps_hex,
        value.refit_tolerance_hex,
        value.refit_max_iter,
        value.refit_mode,
        value.refit_grid_points,
        value.refit_local_steps,
    ]
    if any(not isinstance(result[index], str) for index in range(1, 8)):
        raise TypeError("Search refit-cache identity fields must be strings.")
    if any(type(result[index]) is not int for index in (8, 10, 11)):
        raise TypeError("Search refit-cache numerical fields must be integers.")
    if not isinstance(result[9], str):
        raise TypeError("Search refit-cache mode must be a string.")
    for index in range(2, 6):
        _digest_string(result[index], f"refit-cache key field {index}")
    _canonical_hex_string(result[6], "refit-cache likelihood eps")
    _canonical_hex_string(result[7], "refit-cache tolerance")
    return result


def _dec_refit_cache_key(encoded: Any) -> PartitionRefitKey:
    value = _values(encoded, 12, "refit-cache key")
    if value[0] != PARTITION_REFIT_KEY_SCHEMA:
        raise ValueError("Checkpoint refit-cache key has an unsupported schema.")
    for index in range(1, 8):
        _string(value[index], f"refit-cache key field {index}")
    for index in (8, 10, 11):
        _integer(value[index], f"refit-cache key field {index}")
    _string(value[9], "refit-cache mode")
    for index in range(2, 6):
        _digest_string(value[index], f"refit-cache key field {index}")
    _canonical_hex_string(value[6], "refit-cache likelihood eps")
    _canonical_hex_string(value[7], "refit-cache tolerance")
    return PartitionRefitKey(
        partition_signature=_string(value[1], "refit-cache partition signature"),
        observed_model_hash=_digest_string(value[2], "refit-cache observed model"),
        observed_likelihood_hash=_digest_string(
            value[3], "refit-cache observed likelihood"
        ),
        reporting_model_hash=_digest_string(value[4], "refit-cache reporting model"),
        observed_box_hash=_digest_string(value[5], "refit-cache observed box"),
        likelihood_eps_hex=_canonical_hex_string(
            value[6], "refit-cache likelihood eps"
        ),
        refit_tolerance_hex=_canonical_hex_string(
            value[7], "refit-cache tolerance"
        ),
        refit_max_iter=_integer(value[8], "refit-cache max iter"),
        refit_mode=_string(value[9], "refit-cache mode"),
        refit_grid_points=_integer(value[10], "refit-cache grid points"),
        refit_local_steps=_integer(value[11], "refit-cache local steps"),
    )


def _enc_refit_cache_entry(
    key: PartitionRefitKey,
    fit: PartitionFit,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> list[Any]:
    encoded_key = _enc_refit_cache_key(key)
    _validate_refit_cache_identity(key, fit)
    return [encoded_key, _enc_partition_fit(fit, arrays, digest_cache)]


def _dec_refit_cache_entry(
    encoded: Any,
    arrays: Any,
) -> tuple[PartitionRefitKey, PartitionFit]:
    pair = _values(encoded, 2, "refit-cache entry")
    key = _dec_refit_cache_key(pair[0])
    fit = _dec_partition_fit(pair[1], arrays)
    _validate_refit_cache_identity(key, fit)
    return key, fit


def _encode_search_checkpoint(
    checkpoint: SearchCheckpoint,
    arrays: dict[str, np.ndarray],
    digest_cache: dict[tuple[str, int], str],
) -> dict[str, Any]:
    if (
        isinstance(checkpoint.raw_guide_phi, np.ndarray)
        and checkpoint.raw_guide_phi.dtype.hasobject
    ):
        raise TypeError("Object arrays are not permitted in a search checkpoint.")
    return {
        "controller": _enc_online_state(checkpoint.controller),
        "lambda_runs": [
            _enc_lambda_checkpoint(item, arrays, digest_cache)
            for item in checkpoint.lambda_runs
        ],
        "candidates": [
            _enc_candidate_record(item, arrays, digest_cache)
            for item in checkpoint.candidates
        ],
        "refit_cache": [
            _enc_refit_cache_entry(key, fit, arrays, digest_cache)
            for key, fit in checkpoint.refit_cache
        ],
        "next_candidate_id": checkpoint.next_candidate_id,
        "work": _enc_work(checkpoint.work),
        "guide_work": _enc_work(checkpoint.guide_work),
        "search_stop_override": checkpoint.search_stop_override,
        "direct_pool": _enc_direct_pool(
            checkpoint.direct_pool, arrays, digest_cache
        ),
        "raw_guide_phi": _encode_primitive(
            checkpoint.raw_guide_phi, arrays, digest_cache
        ),
        "guided_initialization": _enc_guided_initialization(
            checkpoint.guided_initialization, arrays, digest_cache
        ),
        "float64_recovery_status": checkpoint.float64_recovery_status,
        "elapsed_seconds": _ef(checkpoint.elapsed_seconds),
    }


def solver_state_content_key(state: SolverState) -> str:
    """Return the checkpoint-stable identity of a continuation state."""

    if not isinstance(state, SolverState):
        raise TypeError("solver state key requires SolverState.")
    encoded = _enc_solver_state(state, {}, {})
    payload = json.dumps(
        encoded,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_search_checkpoint(encoded: Any, arrays: Any) -> SearchCheckpoint:
    if not isinstance(encoded, dict):
        raise ValueError("Checkpoint payload must be a SearchCheckpoint mapping.")
    keys = {
        "controller",
        "lambda_runs",
        "candidates",
        "refit_cache",
        "next_candidate_id",
        "work",
        "guide_work",
        "search_stop_override",
        "direct_pool",
        "raw_guide_phi",
        "guided_initialization",
        "float64_recovery_status",
        "elapsed_seconds",
    }
    _require_keys(encoded, keys, "SearchCheckpoint")
    lambda_runs = encoded["lambda_runs"]
    candidates = encoded["candidates"]
    refit_cache = encoded["refit_cache"]
    for name, value in (
        ("lambda_runs", lambda_runs),
        ("candidates", candidates),
        ("refit_cache", refit_cache),
    ):
        if not isinstance(value, list):
            raise ValueError(f"Checkpoint {name} must be a list.")
    decoded_cache = [
        _dec_refit_cache_entry(item, arrays) for item in refit_cache
    ]
    stop_override = encoded["search_stop_override"]
    return SearchCheckpoint(
        controller=_dec_online_state(encoded["controller"]),
        lambda_runs=tuple(
            _dec_lambda_checkpoint(item, arrays) for item in lambda_runs
        ),
        candidates=tuple(
            _dec_candidate_record(item, arrays) for item in candidates
        ),
        refit_cache=tuple(decoded_cache),
        next_candidate_id=_integer(
            encoded["next_candidate_id"], "next_candidate_id"
        ),
        work=_dec_work(encoded["work"]),
        guide_work=_dec_work(encoded["guide_work"]),
        search_stop_override=(
            None
            if stop_override is None
            else _string(stop_override, "search_stop_override")
        ),
        direct_pool=_dec_direct_pool(encoded["direct_pool"], arrays),
        raw_guide_phi=_decode_guide_array(encoded["raw_guide_phi"], arrays),
        guided_initialization=_dec_guided_initialization(
            encoded["guided_initialization"], arrays
        ),
        float64_recovery_status=_string(
            encoded["float64_recovery_status"], "float64_recovery_status"
        ),
        elapsed_seconds=_float(encoded["elapsed_seconds"], "elapsed_seconds"),
    )


def _decode_guide_array(encoded: Any, arrays: Any) -> np.ndarray | torch.Tensor:
    value = _decode_primitive(encoded, arrays)
    if not isinstance(value, np.ndarray) and not torch.is_tensor(value):
        raise ValueError("Checkpoint raw_guide_phi must be an array or tensor.")
    return value


def _legacy_checkpoint_error(path: Path) -> LegacyCheckpointFormatError:
    return LegacyCheckpointFormatError(
        f"{path} is a monolithic NPZ checkpoint. Earlier schemas are not "
        "compatible with the explicit schema-v4 directory format; restart with "
        "a new checkpoint path."
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
        "checkpoint",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            "Malformed search-checkpoint manifest keys: "
            f"expected={sorted(expected_keys)}, observed={sorted(manifest)}"
        )
    if type(manifest["schema_version"]) is not int or (
        manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION
    ):
        observed = manifest["schema_version"]
        raise ValueError(
            "Unsupported search-checkpoint schema version: "
            f"observed={observed!r}, required={CHECKPOINT_SCHEMA_VERSION}. "
            "Earlier search-state schemas cannot be resumed; restart at a new path."
        )
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
    for tag in ("array", "tensor"):
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
        created = False
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _require_safe_object_file(destination)
        else:
            os.chmod(destination, 0o600)
            created = True
        return created
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
    """Hold a crash-released POSIX writer lock on a persistent private file."""

    lock_path = directory / _WRITER_LOCK_NAME
    if lock_path.is_symlink():
        raise ValueError(
            f"Checkpoint writer lock must be a regular private file: {lock_path}"
        )
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno not in (errno.ELOOP, errno.EISDIR):
            raise
        raise ValueError(
            f"Checkpoint writer lock must be a regular private file: {lock_path}"
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(
                f"Checkpoint writer lock must be a regular file: {lock_path}"
            )
        if stat.S_IMODE(status.st_mode) & 0o077:
            raise PermissionError(
                f"Checkpoint writer lock is not private: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise CheckpointConcurrentWriterError(
                f"Checkpoint already has an active writer lock: {lock_path}"
            ) from exc
        try:
            current = lock_path.lstat()
        except FileNotFoundError as exc:
            raise CheckpointConcurrentWriterError(
                "Checkpoint writer lock disappeared during acquisition."
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != status.st_dev
            or current.st_ino != status.st_ino
        ):
            raise CheckpointConcurrentWriterError(
                "Checkpoint writer lock changed during acquisition."
            )
        payload = json.dumps(
            {"pid": os.getpid()}, separators=(",", ":")
        ).encode("ascii")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, payload)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _save_search_checkpoint_impl(
    path: str | Path,
    *,
    identity: dict[str, Any],
    checkpoint: SearchCheckpoint,
    expected_generation: int | None,
) -> int:
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
        encoded_checkpoint = _encode_search_checkpoint(checkpoint, arrays, {})
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
            "checkpoint": encoded_checkpoint,
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


def save_search_checkpoint(
    path: str | Path,
    *,
    identity: dict[str, Any],
    checkpoint: SearchCheckpoint,
    expected_generation: int | None = None,
) -> int:
    """Commit one explicit search-checkpoint generation after an observation.

    ``path`` is a directory even when a caller retains the historical ``.npz``
    suffix. A POSIX advisory lock rejects overlapping live writers but remains
    harmless after process death. The generation compare-and-swap remains the
    authoritative stale-writer guard. A caller must pass the generation
    returned by its previous save or load when updating an existing checkpoint.
    """

    if not isinstance(checkpoint, SearchCheckpoint):
        raise TypeError("checkpoint must be a SearchCheckpoint.")
    return _save_search_checkpoint_impl(
        path,
        identity=identity,
        checkpoint=checkpoint,
        expected_generation=expected_generation,
    )


def load_search_checkpoint(
    path: str | Path,
    *,
    expected_identity: dict[str, Any],
    return_generation: bool = False,
) -> SearchCheckpoint | tuple[SearchCheckpoint, int]:
    """Load a checkpoint after identity-first, no-pickle validation."""

    source = Path(path)
    manifest_path, array_directory = _open_checkpoint_directory(source)
    manifest = _parse_manifest(_read_manifest_bytes(manifest_path))
    expected = _canonical_identity_value(expected_identity)
    _require_identity(expected, manifest["identity"])

    referenced_arrays: set[str] = set()
    _collect_array_references(manifest["checkpoint"], referenced_arrays)
    stored_metadata = manifest["objects"]
    if set(stored_metadata) != referenced_arrays:
        raise ValueError(
            "Checkpoint object table does not match its manifest references."
        )
    arrays = _CheckpointObjectStore(array_directory, stored_metadata)
    decoded = _decode_search_checkpoint(manifest["checkpoint"], arrays)
    if arrays.accessed != referenced_arrays:
        raise ValueError("Checkpoint decoder did not consume every referenced array.")
    if return_generation:
        return decoded, int(manifest["generation"])
    return decoded


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointConcurrentWriterError",
    "CheckpointIdentityMismatchError",
    "DirectPoolCheckpoint",
    "LambdaCheckpoint",
    "LegacyCheckpointFormatError",
    "SearchCheckpoint",
    "build_search_checkpoint_identity",
    "load_search_checkpoint",
    "save_search_checkpoint",
    "solver_state_content_key",
    "source_tree_fingerprint",
]
