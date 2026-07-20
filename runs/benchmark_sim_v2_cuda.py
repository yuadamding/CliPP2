#!/usr/bin/env python3
"""Benchmark CliPP2 Sim-v2 cases against an isolated CUDA source checkout.

The benchmark times ``process_one_file_bundle`` with CUDA synchronization on
both sides of the measured region. It records peak allocator statistics and
the exact backend provenance for every lambda-search row without including TSV
artifact serialization in the timing.

Example::

    python runs/benchmark_sim_v2_cuda.py \
        --package-parent /data/revised \
        --source-label revised \
        --backend quotient-workset \
        --max-unique-lambdas 1 \
        --strict-backend \
        --output-json runs/revised_quotient.json \
        /data/CliPP2SimV2_TSV/case_a.tsv
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
DEFAULT_WORKSET_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_COMPRESSED_CACHE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_WORKSET_ADD_BATCH = 64
DEFAULT_WORKSET_MAX_EXPANSIONS = 16
DEFAULT_CERTIFICATE_MAX_ITER = 512
DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS = 2
DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE = 1.0

BACKEND_COLUMNS = (
    "inner_iterations",
    "admm_iterations",
    "inner_solver",
    "inner_backend",
    "backend_iterations",
    "quotient_iterations",
    "workset_iterations",
    "workset_expansions",
    "streamed_edge_passes",
    "dense_iterations",
    "fallback_reason",
)
CORRECTNESS_TIMING_COLUMNS = (
    "partition_signature",
    "partition_hash",
    "cluster_sizes",
    "penalized_objective",
    "raw_objective",
    "loglik",
    "raw_loglik",
    "summary_loglik",
    "bic",
    "classic_bic",
    "extended_bic",
    "partition_icl",
    "bic_loglik",
    "bic_n_clusters",
    "n_clusters",
    "fixed_objective_kkt_residual",
    "full_kkt_tolerance",
    "raw_kkt_eligible",
    "bic_selection_eligible",
    "candidate_elapsed_seconds",
    "raw_fit_elapsed_seconds",
    "bic_refit_elapsed_seconds",
)
SEARCH_ID_COLUMNS = (
    "tumor_id",
    "selection_step",
    "search_round",
    "search_phase",
    "lambda",
    "lambda_applicable",
    "candidate_pool_source",
    "candidate_role",
    "estimator_role",
    "device",
    "dtype",
    "is_selected_best_row",
    "full_kkt_certified",
    "full_kkt_certificate_status",
)


class StrictBackendError(RuntimeError):
    """Raised after diagnostics are written for a strict backend violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _seed(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("must lie in [0, 2^32 - 1]")
    return parsed


def _normalize_backend(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"auto", "dense", "quotient_workset"}:
        raise argparse.ArgumentTypeError(
            "must be one of: auto, dense, quotient-workset"
        )
    return normalized


def _normalize_fallback_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"auto", "device_only", "cpu_allowed", "error"}:
        raise argparse.ArgumentTypeError(
            "must be one of: auto, device-only, cpu-allowed, error"
        )
    return normalized


def _normalize_missing_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"error", "all_true"}:
        raise argparse.ArgumentTypeError("must be one of: error, all-true")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-parent",
        type=Path,
        required=True,
        help="Directory containing the CliPP2 package checkout to import.",
    )
    parser.add_argument(
        "--source-label",
        required=True,
        help="Stable label stored with this source variant (for example main or revised).",
    )
    parser.add_argument(
        "--backend",
        "--inner-backend",
        dest="backend",
        type=_normalize_backend,
        default="dense",
        help="Requested inner backend: dense, quotient-workset, or auto.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output JSON atomically.",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help=(
            "Write diagnostics, then fail unless every backend-applicable search "
            "row used the requested backend with an empty fallback reason."
        ),
    )
    parser.add_argument(
        "--max-unique-lambdas",
        type=_positive_int,
        help="Temporarily override the online controller's unique-lambda budget.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=_seed, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable PyTorch deterministic algorithms for the benchmark process.",
    )
    parser.add_argument("--warmup-iterations", type=_nonnegative_int, default=3)
    parser.add_argument("--outer-max-iter", type=_positive_int, default=8)
    parser.add_argument("--inner-max-iter", type=_positive_int, default=30)
    parser.add_argument("--tol", type=_positive_float, default=1e-4)
    parser.add_argument("--summary-tol", type=_positive_float, default=1e-4)
    parser.add_argument("--bic-partition-tol", type=_positive_float, default=1e-4)
    parser.add_argument("--major-prior", type=float, default=0.5)
    parser.add_argument("--bic-df-scale", type=float, default=1.0)
    parser.add_argument("--bic-cluster-penalty", type=float, default=0.0)
    parser.add_argument(
        "--dense-fallback-policy",
        type=_normalize_fallback_policy,
        default="auto",
    )
    parser.add_argument(
        "--workset-max-bytes", type=_positive_int, default=DEFAULT_WORKSET_MAX_BYTES
    )
    parser.add_argument(
        "--compressed-cache-max-bytes",
        type=_positive_int,
        default=DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    )
    parser.add_argument(
        "--workset-add-batch", type=_positive_int, default=DEFAULT_WORKSET_ADD_BATCH
    )
    parser.add_argument(
        "--workset-max-expansions",
        type=_nonnegative_int,
        default=DEFAULT_WORKSET_MAX_EXPANSIONS,
    )
    parser.add_argument(
        "--certificate-max-iter",
        type=_positive_int,
        default=DEFAULT_CERTIFICATE_MAX_ITER,
    )
    parser.add_argument(
        "--certificate-refinement-rounds",
        type=_nonnegative_int,
        default=DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    )
    parser.add_argument(
        "--certificate-column-tol-scale",
        type=_positive_float,
        default=DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    )
    parser.add_argument(
        "--allow-heuristic-structure-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--materialize-full-dual", action="store_true")
    parser.add_argument(
        "--use-warm-starts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--missing-cna-policy",
        type=_normalize_missing_policy,
        default="error",
    )
    parser.add_argument("cases", metavar="CASE", type=Path, nargs="+")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.strict_backend and args.backend == "auto":
        parser.error(
            "--strict-backend requires an explicit dense or quotient-workset backend"
        )
    if not str(args.device).strip().lower().startswith("cuda"):
        parser.error("this harness requires a CUDA --device")
    if not math.isfinite(float(args.major_prior)) or not 0.0 < args.major_prior < 1.0:
        parser.error("--major-prior must be finite and lie strictly between 0 and 1")
    if not math.isfinite(float(args.bic_df_scale)) or args.bic_df_scale <= 0.0:
        parser.error("--bic-df-scale must be finite and positive")
    if not math.isfinite(float(args.bic_cluster_penalty)):
        parser.error("--bic-cluster-penalty must be finite")
    if not str(args.source_label).strip():
        parser.error("--source-label must not be blank")
    args.source_label = str(args.source_label).strip()
    return args


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar structures to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return as_float if math.isfinite(as_float) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, list[Path], Path]:
    package_parent = args.package_parent.expanduser().resolve(strict=True)
    package_root = (package_parent / "CliPP2").resolve(strict=True)
    if not (package_root / "__init__.py").is_file():
        raise ValueError(
            f"--package-parent must contain CliPP2/__init__.py: {package_parent}"
        )
    cases: list[Path] = []
    for case in args.cases:
        resolved = case.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Case path is not a regular file: {resolved}")
        cases.append(resolved)
    output = args.output_json.expanduser().resolve()
    if output in cases:
        raise ValueError("--output-json must not overwrite an input case")
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists; pass --overwrite to replace it: {output}"
        )
    if output.exists() and not output.is_file():
        raise ValueError(f"Output path exists and is not a regular file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return package_parent, cases, output


def _module_path(module: Any) -> Path:
    location = getattr(module, "__file__", None)
    if not location:
        raise RuntimeError(f"Imported module {module!r} has no source path")
    return Path(location).resolve()


def _assert_module_in_root(module: Any, package_root: Path) -> None:
    module_path = _module_path(module)
    if not module_path.is_relative_to(package_root):
        raise RuntimeError(
            f"Mixed CliPP2 import: {module.__name__} resolved to {module_path}, "
            f"outside requested source {package_root}."
        )


def _import_isolated_source(package_parent: Path) -> dict[str, Any]:
    package_root = (package_parent / "CliPP2").resolve(strict=True)
    preloaded = sys.modules.get("CliPP2")
    if preloaded is not None:
        _assert_module_in_root(preloaded, package_root)
    parent_text = str(package_parent)
    sys.path[:] = [parent_text, *[entry for entry in sys.path if entry != parent_text]]
    importlib.invalidate_caches()
    modules = {
        "package": importlib.import_module("CliPP2"),
        "core_model": importlib.import_module("CliPP2.core.model"),
        "pipeline": importlib.import_module("CliPP2.runners.pipeline"),
        "runner_selection": importlib.import_module("CliPP2.runners.model_selection"),
        "selection_config": importlib.import_module("CliPP2.model_selection.config"),
    }
    for module in modules.values():
        _assert_module_in_root(module, package_root)
    return modules


def _git_value(package_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(package_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _python_source_manifest(package_root: Path) -> dict[str, Any]:
    """Hash importable package ``.py`` paths and bytes deterministically."""

    root = package_root.resolve(strict=True)
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*.py"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        package_directory = root
        importable = True
        for directory_name in relative.parts[:-1]:
            package_directory /= directory_name
            if not (package_directory / "__init__.py").is_file():
                importable = False
                break
        if importable:
            files.append((relative.as_posix(), path))
    files.sort(key=lambda item: item[0])

    digest = hashlib.sha256()
    total_file_bytes = 0
    for relative_path, path in files:
        relative_bytes = relative_path.encode("utf-8")
        source_bytes = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, byteorder="big", signed=False))
        digest.update(relative_bytes)
        digest.update(len(source_bytes).to_bytes(8, byteorder="big", signed=False))
        digest.update(source_bytes)
        total_file_bytes += len(source_bytes)
    return {
        "algorithm": "sha256_length_prefixed_posix_relative_path_and_bytes_v1",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_file_bytes": total_file_bytes,
    }


def _source_metadata(
    *, package_parent: Path, source_label: str, package: Any
) -> dict[str, Any]:
    package_root = (package_parent / "CliPP2").resolve(strict=True)
    status = _git_value(
        package_root, "status", "--porcelain=v1", "--untracked-files=all"
    )
    return {
        "label": source_label,
        "package_parent": str(package_parent),
        "package_root": str(package_root),
        "imported_package_file": str(_module_path(package)),
        "package_version": str(getattr(package, "__version__", "unknown")),
        "git_revision": _git_value(package_root, "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "git_status_sha256": None
        if status is None
        else hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "python_source_manifest": _python_source_manifest(package_root),
    }


def _configure_reproducibility(torch: Any, *, seed: int, deterministic: bool) -> None:
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
        numpy.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _determinism_metadata(
    torch: Any, *, seed: int, deterministic_requested: bool
) -> dict[str, Any]:
    warn_only_getter = getattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", None
    )
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    return {
        "requested": bool(deterministic_requested),
        "seed": int(seed),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": None
        if not callable(warn_only_getter)
        else bool(warn_only_getter()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": None
        if matmul_backend is None
        else bool(matmul_backend.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _nvidia_smi_driver_versions() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"versions": [], "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "versions": [],
            "error": completed.stderr.strip() or f"exit status {completed.returncode}",
        }
    versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"versions": versions, "error": None}


def _torch_driver_version(torch: Any) -> int | None:
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except (RuntimeError, TypeError, ValueError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _environment_metadata(torch: Any, device: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    sm_count = getattr(properties, "multi_processor_count", None)
    uuid = getattr(properties, "uuid", None)
    return {
        "captured_at_utc": _utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "package_versions": _package_versions(),
        "torch": {
            "version": str(torch.__version__),
            "cuda_build": None
            if torch.version.cuda is None
            else str(torch.version.cuda),
            "cudnn_version": None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version()),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        },
        "cuda_driver": {
            "torch_driver_version_raw": _torch_driver_version(torch),
            "nvidia_smi": _nvidia_smi_driver_versions(),
        },
        "device": {
            "requested": str(device),
            "logical_index": int(device.index),
            "name": str(torch.cuda.get_device_name(device)),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
            "multi_processor_count": None if sm_count is None else int(sm_count),
            "uuid": None if uuid is None else str(uuid),
        },
        "relevant_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get(
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
            ),
        },
    }


def _resolve_cuda_device(torch: Any, requested: str) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA benchmark requested but torch.cuda.is_available() is false; "
            "CPU fallback is not permitted by this harness."
        )
    device = torch.device(requested)
    if device.type != "cuda":
        raise ValueError(f"Expected a CUDA device, got {requested!r}")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {index} is outside [0, {torch.cuda.device_count()})"
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _cuda_warmup(
    torch: Any, device: Any, *, dtype_name: str, iterations: int
) -> dict[str, Any]:
    if iterations <= 0:
        return {
            "enabled": False,
            "kind": "cuda_context_matmul",
            "iterations": 0,
            "elapsed_seconds": 0.0,
        }
    dtype = getattr(torch, dtype_name)
    started = perf_counter()
    left = torch.linspace(0.0, 1.0, 128 * 128, device=device, dtype=dtype).reshape(
        128, 128
    )
    right = torch.linspace(1.0, 0.0, 128 * 128, device=device, dtype=dtype).reshape(
        128, 128
    )
    product = None
    for _ in range(iterations):
        product = torch.mm(left, right)
    torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    del product, left, right
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "enabled": True,
        "kind": "cuda_context_matmul",
        "matrix_shape": [128, 128],
        "iterations": int(iterations),
        "elapsed_seconds": float(elapsed),
    }


def _memory_snapshot(torch: Any, device: Any, *, include_peaks: bool) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    result = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "device_free_bytes": int(free_bytes),
        "device_total_bytes": int(total_bytes),
    }
    if include_peaks:
        result.update(
            {
                "max_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "max_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
        )
    return result


def _module_label(module: Any) -> str:
    return str(getattr(module, "__name__", type(module).__name__))


@contextmanager
def _temporary_max_unique_lambdas(
    modules: Iterable[Any], override: int | None
) -> Iterable[dict[str, Any]]:
    """Temporarily patch every live controller-budget binding."""

    attribute = "PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS"
    originals: list[tuple[Any, int]] = []
    for module in modules:
        if hasattr(module, attribute):
            originals.append((module, int(getattr(module, attribute))))
    if not originals:
        raise RuntimeError(
            f"The isolated source does not expose the controller budget {attribute}."
        )
    effective = originals[0][1] if override is None else int(override)
    if effective < 1:
        raise ValueError("max_unique_lambdas must be positive")
    if override is not None:
        for module, _ in originals:
            setattr(module, attribute, effective)
    try:
        yield {
            "override_requested": override is not None,
            "effective": effective,
            "original_bindings": {
                _module_label(module): original for module, original in originals
            },
        }
    finally:
        for module, original in originals:
            setattr(module, attribute, original)


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def _backend_family(value: Any) -> str | None:
    if _blank(value):
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized.startswith("not_applicable"):
        return None
    if "quotient" in normalized and "workset" in normalized:
        return "quotient_workset"
    if normalized == "admm_complete_graph" or "dense" in normalized:
        return "dense"
    return "unknown"


def _device_type(value: Any) -> str | None:
    if _blank(value):
        return None
    normalized = str(value).strip().lower()
    return normalized.split(":", 1)[0]


def _actual_backend(record: Mapping[str, Any]) -> Any:
    for column in ("inner_backend", "inner_solver"):
        value = record.get(column)
        normalized = str(value).strip().lower() if not _blank(value) else ""
        if normalized not in {"", "unknown"}:
            return value
    return None


def _backend_applicable(record: Mapping[str, Any]) -> bool:
    source = str(record.get("candidate_pool_source", "")).strip().lower()
    role = str(record.get("estimator_role", "")).strip().lower()
    if source == "raw_fused_lambda_path" or role == "raw_fused_lambda_path":
        return True
    actual = _actual_backend(record)
    return _backend_family(actual) is not None


def _backend_observations(
    records: Iterable[Mapping[str, Any]], *, requested_backend: str
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    requested = _normalize_backend(requested_backend)
    for index, record in enumerate(records):
        observation: dict[str, Any] = {
            "search_row_index": int(index),
            "requested_backend": requested,
        }
        for column in (
            *SEARCH_ID_COLUMNS,
            *CORRECTNESS_TIMING_COLUMNS,
            *BACKEND_COLUMNS,
        ):
            observation[column] = _json_safe(record.get(column))
        actual = _actual_backend(record)
        observation["actual_backend"] = _json_safe(actual)
        observation["actual_backend_family"] = _backend_family(actual)
        observation["backend_applicable"] = _backend_applicable(record)
        observation["missing_backend_columns"] = [
            column for column in BACKEND_COLUMNS if column not in record
        ]
        observations.append(observation)
    return observations


def _audit_backend_observations(
    observations: Iterable[Mapping[str, Any]],
    *,
    requested_backend: str,
    expected_device_type: str = "cuda",
) -> dict[str, Any]:
    requested = _normalize_backend(requested_backend)
    expected_device = _device_type(expected_device_type)
    if expected_device is None:
        raise ValueError("expected_device_type must not be blank")
    applicable = [row for row in observations if bool(row.get("backend_applicable"))]
    violations: list[dict[str, Any]] = []
    if requested == "auto":
        return {
            "requested_backend": requested,
            "expected_device_type": expected_device,
            "applicable_rows": len(applicable),
            "passed": None,
            "violations": [],
        }
    if not applicable:
        violations.append(
            {
                "search_row_index": None,
                "kind": "no_backend_applicable_rows",
                "actual": None,
            }
        )
    for row in applicable:
        row_index = row.get("search_row_index")
        family = row.get("actual_backend_family")
        if family != requested:
            violations.append(
                {
                    "search_row_index": row_index,
                    "kind": "actual_backend_family",
                    "expected": requested,
                    "actual": family,
                    "actual_backend": row.get("actual_backend"),
                }
            )
        recorded_device = row.get("device")
        actual_device = _device_type(recorded_device)
        if actual_device != expected_device:
            violations.append(
                {
                    "search_row_index": row_index,
                    "kind": "actual_device_type",
                    "expected": expected_device,
                    "actual": actual_device,
                    "recorded_device": recorded_device,
                }
            )
        fallback_reason = row.get("fallback_reason")
        if not _blank(fallback_reason):
            violations.append(
                {
                    "search_row_index": row_index,
                    "kind": "fallback_reason",
                    "expected": "",
                    "actual": fallback_reason,
                }
            )
        if row.get("missing_backend_columns"):
            violations.append(
                {
                    "search_row_index": row_index,
                    "kind": "missing_backend_columns",
                    "actual": list(row["missing_backend_columns"]),
                }
            )
    return {
        "requested_backend": requested,
        "expected_device_type": expected_device,
        "applicable_rows": len(applicable),
        "passed": not violations,
        "violations": violations,
    }


def _lambda_budget_audit(
    observations: Iterable[Mapping[str, Any]], *, max_unique_lambdas: int
) -> dict[str, Any]:
    values: set[float] = set()
    for row in observations:
        if not bool(row.get("backend_applicable")):
            continue
        value = row.get("lambda")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric) and numeric >= 0.0:
            values.add(float(round(numeric, 12)))
    ordered = sorted(values)
    return {
        "max_unique_lambdas": int(max_unique_lambdas),
        "observed_unique_lambdas": ordered,
        "observed_unique_lambda_count": len(ordered),
        "passed": len(ordered) <= int(max_unique_lambdas),
    }


def _requested_fit_option_values(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lambda_value": 0.0,
        "outer_max_iter": int(args.outer_max_iter),
        "inner_max_iter": int(args.inner_max_iter),
        "tol": float(args.tol),
        "summary_tol": float(args.summary_tol),
        "bic_partition_tol": float(args.bic_partition_tol),
        "major_prior": float(args.major_prior),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "inner_backend": str(args.backend),
        "workset_max_bytes": int(args.workset_max_bytes),
        "compressed_cache_max_bytes": int(args.compressed_cache_max_bytes),
        "dense_fallback_policy": str(args.dense_fallback_policy),
        "workset_add_batch": int(args.workset_add_batch),
        "workset_max_expansions": int(args.workset_max_expansions),
        "certificate_max_iter": int(args.certificate_max_iter),
        "certificate_refinement_rounds": int(args.certificate_refinement_rounds),
        "certificate_column_tol_scale": float(args.certificate_column_tol_scale),
        "allow_heuristic_structure_splits": bool(args.allow_heuristic_structure_splits),
        "materialize_full_dual": bool(args.materialize_full_dual),
    }


def _fit_options_compatibility(
    core_model: Any, args: argparse.Namespace
) -> dict[str, Any]:
    requested = _requested_fit_option_values(args)
    fields = getattr(core_model.FitOptions, "__dataclass_fields__", {})
    supported = {name: value for name, value in requested.items() if name in fields}
    dropped = sorted(set(requested).difference(supported))
    return {
        "supported_values": supported,
        "unsupported_option_names": dropped,
    }


def _fit_options(core_model: Any, args: argparse.Namespace) -> Any:
    compatibility = _fit_options_compatibility(core_model, args)
    return core_model.FitOptions(**compatibility["supported_values"])


def _run_case(
    *,
    torch: Any,
    device: Any,
    process_one_file_bundle: Any,
    core_model: Any,
    args: argparse.Namespace,
    case_path: Path,
    scratch_dir: Path,
) -> tuple[dict[str, Any], Exception | None]:
    case_result: dict[str, Any] = {
        "case_path": str(case_path),
        "case_name": case_path.stem,
        "input_size_bytes": int(case_path.stat().st_size),
        "input_sha256": _sha256_file(case_path),
        "started_at_utc": _utc_now(),
    }
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = _memory_snapshot(torch, device, include_peaks=True)
    started = perf_counter()
    caught: Exception | None = None
    summary: Mapping[str, Any] | None = None
    search_df: Any = None
    try:
        summary, search_df = process_one_file_bundle(
            file_path=case_path,
            outdir=scratch_dir,
            simulation_root=None,
            lambda_grid=None,
            lambda_grid_mode="partition_guided_admm",
            fit_options=_fit_options(core_model, args),
            bic_df_scale=float(args.bic_df_scale),
            bic_cluster_penalty=float(args.bic_cluster_penalty),
            selection_score="partition_icl",
            use_warm_starts=bool(args.use_warm_starts),
            write_outputs=False,
            finalize_selected_fit=False,
            missing_cna_policy=str(args.missing_cna_policy),
            evaluate_all_candidates=False,
        )
        torch.cuda.synchronize(device)
    except Exception as exc:  # retain a diagnostic JSON before propagating
        caught = exc
        case_result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            torch.cuda.synchronize(device)
        except Exception as sync_exc:  # pragma: no cover - CUDA fault aftermath
            case_result["post_error_synchronize_error"] = (
                f"{type(sync_exc).__name__}: {sync_exc}"
            )
    elapsed = perf_counter() - started
    memory_after = _memory_snapshot(torch, device, include_peaks=True)
    case_result.update(
        {
            "finished_at_utc": _utc_now(),
            "status": "error" if caught is not None else "complete",
            "timing": {
                "wall_seconds": float(elapsed),
                "scope": "process_one_file_bundle(write_outputs=False)",
                "cuda_synchronized_before_and_after": True,
            },
            "cuda_memory": {
                "reset_peak_memory_stats_before_case": True,
                "empty_cache_before_case": True,
                "before": memory_before,
                "after": memory_after,
                "max_memory_allocated_bytes": memory_after[
                    "max_memory_allocated_bytes"
                ],
                "max_memory_reserved_bytes": memory_after["max_memory_reserved_bytes"],
            },
        }
    )
    if caught is not None:
        return case_result, caught
    assert summary is not None and search_df is not None
    records = search_df.to_dict(orient="records")
    observations = _backend_observations(records, requested_backend=args.backend)
    case_result.update(
        {
            "pipeline_summary": _json_safe(dict(summary)),
            "search_row_count": len(records),
            "backend_observations": observations,
            "backend_audit": _audit_backend_observations(
                observations,
                requested_backend=args.backend,
                expected_device_type=_device_type(args.device) or "cuda",
            ),
        }
    )
    return case_result, None


def _aggregate(case_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [case for case in case_results if case.get("status") == "complete"]
    wall_times = [float(case["timing"]["wall_seconds"]) for case in complete]
    allocated = [
        int(case["cuda_memory"]["max_memory_allocated_bytes"]) for case in complete
    ]
    reserved = [
        int(case["cuda_memory"]["max_memory_reserved_bytes"]) for case in complete
    ]
    observations = [
        row for case in complete for row in case.get("backend_observations", [])
    ]
    family_counts: dict[str, int] = {}
    fallback_rows = 0
    for row in observations:
        family = str(row.get("actual_backend_family"))
        family_counts[family] = family_counts.get(family, 0) + 1
        if bool(row.get("backend_applicable")) and not _blank(
            row.get("fallback_reason")
        ):
            fallback_rows += 1
    return {
        "case_count": len(case_results),
        "completed_case_count": len(complete),
        "failed_case_count": len(case_results) - len(complete),
        "wall_seconds_sum": float(sum(wall_times)),
        "wall_seconds_mean": None
        if not wall_times
        else float(sum(wall_times) / len(wall_times)),
        "max_memory_allocated_bytes": None if not allocated else max(allocated),
        "max_memory_reserved_bytes": None if not reserved else max(reserved),
        "search_row_count": len(observations),
        "actual_backend_family_counts": family_counts,
        "fallback_row_count": fallback_rows,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_progress_checkpoint(
    *,
    output_json: Path,
    payload: dict[str, Any],
    total_case_count: int,
    state: str,
    checkpoint_sequence: int,
    strict_backend: bool,
    backend_violations: Sequence[Mapping[str, Any]],
    budget_failures: Sequence[str],
    first_error: BaseException | None,
    active_case_path: str | None = None,
) -> None:
    """Refresh derived progress fields and atomically persist a valid snapshot."""

    if state not in {"running", "complete", "failed", "failed_contract", "interrupted"}:
        raise ValueError(f"Unknown benchmark checkpoint state: {state!r}")
    case_results = payload["cases"]
    attempted_case_count = len(case_results)
    completed_case_count = sum(
        result.get("status") == "complete" for result in case_results
    )
    failed_case_count = sum(result.get("status") == "error" for result in case_results)
    is_final = state != "running"
    current_backend_pass = not backend_violations and first_error is None
    payload["aggregate"] = _aggregate(case_results)
    payload["strict_backend_result"] = {
        "enforced": bool(strict_backend),
        "passed": None
        if not strict_backend or state in {"running", "interrupted"}
        else current_backend_pass,
        "provisional_passed": None if not strict_backend else current_backend_pass,
        "violation_count": len(backend_violations),
        "violations": list(backend_violations),
    }
    payload["lambda_budget_result"] = {
        "passed": not budget_failures,
        "provisional": not is_final,
        "failed_case_paths": list(budget_failures),
    }
    checkpointed_at = _utc_now()
    payload["run_status"] = {
        "state": state,
        "is_final": is_final,
        "total_case_count": int(total_case_count),
        "attempted_case_count": attempted_case_count,
        "completed_case_count": completed_case_count,
        "failed_case_count": failed_case_count,
        "remaining_case_count": max(int(total_case_count) - attempted_case_count, 0),
        "active_case_path": active_case_path,
        "checkpoint_sequence": int(checkpoint_sequence),
        "checkpointed_at_utc": checkpointed_at,
    }
    payload["last_checkpoint_at_utc"] = checkpointed_at
    if is_final:
        payload["finished_at_utc"] = checkpointed_at
    else:
        payload.pop("finished_at_utc", None)
    _atomic_write_json(output_json, payload)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    package_parent, cases, output_json = _resolve_inputs(args)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    modules = _import_isolated_source(package_parent)
    torch = importlib.import_module("torch")
    device = _resolve_cuda_device(torch, str(args.device))
    args.device = str(device)
    _configure_reproducibility(
        torch, seed=int(args.seed), deterministic=bool(args.deterministic)
    )
    environment = _environment_metadata(torch, device)
    environment["determinism"] = _determinism_metadata(
        torch,
        seed=int(args.seed),
        deterministic_requested=bool(args.deterministic),
    )
    warmup = _cuda_warmup(
        torch,
        device,
        dtype_name=str(args.dtype),
        iterations=int(args.warmup_iterations),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "CliPP2 Sim-v2 synchronized CUDA backend benchmark",
        "started_at_utc": _utc_now(),
        "source": _source_metadata(
            package_parent=package_parent,
            source_label=str(args.source_label),
            package=modules["package"],
        ),
        "environment": environment,
        "warmup": warmup,
        "configuration": {
            "requested_backend": str(args.backend),
            "strict_backend": bool(args.strict_backend),
            "device": str(device),
            "dtype": str(args.dtype),
            "seed": int(args.seed),
            "outer_max_iter": int(args.outer_max_iter),
            "inner_max_iter": int(args.inner_max_iter),
            "tol": float(args.tol),
            "summary_tol": float(args.summary_tol),
            "bic_partition_tol": float(args.bic_partition_tol),
            "major_prior": float(args.major_prior),
            "bic_df_scale": float(args.bic_df_scale),
            "bic_cluster_penalty": float(args.bic_cluster_penalty),
            "dense_fallback_policy": str(args.dense_fallback_policy),
            "workset_max_bytes": int(args.workset_max_bytes),
            "compressed_cache_max_bytes": int(args.compressed_cache_max_bytes),
            "workset_add_batch": int(args.workset_add_batch),
            "workset_max_expansions": int(args.workset_max_expansions),
            "certificate_max_iter": int(args.certificate_max_iter),
            "certificate_refinement_rounds": int(args.certificate_refinement_rounds),
            "certificate_column_tol_scale": float(args.certificate_column_tol_scale),
            "allow_heuristic_structure_splits": bool(
                args.allow_heuristic_structure_splits
            ),
            "materialize_full_dual": bool(args.materialize_full_dual),
            "use_warm_starts": bool(args.use_warm_starts),
            "missing_cna_policy": str(args.missing_cna_policy),
            "lambda_grid": None,
            "lambda_grid_mode": "partition_guided_admm",
            "selection_score": "partition_icl",
            "write_outputs": False,
            "evaluate_all_candidates": False,
            "fit_options_compatibility": _fit_options_compatibility(
                modules["core_model"], args
            ),
        },
        "case_paths": [str(case) for case in cases],
        "cases": [],
    }
    first_error: Exception | None = None
    budget_failures: list[str] = []
    backend_violations: list[dict[str, Any]] = []
    checkpoint_sequence = 0
    active_case_path: str | None = None
    try:
        with _temporary_max_unique_lambdas(
            (modules["runner_selection"], modules["selection_config"]),
            args.max_unique_lambdas,
        ) as lambda_budget:
            payload["configuration"]["lambda_budget"] = lambda_budget
            with tempfile.TemporaryDirectory(prefix="clipp2-sim-v2-benchmark-") as temp:
                scratch_dir = Path(temp)
                for case_index, case_path in enumerate(cases):
                    active_case_path = str(case_path)
                    result, error = _run_case(
                        torch=torch,
                        device=device,
                        process_one_file_bundle=modules[
                            "pipeline"
                        ].process_one_file_bundle,
                        core_model=modules["core_model"],
                        args=args,
                        case_path=case_path,
                        scratch_dir=scratch_dir,
                    )
                    active_case_path = None
                    if error is None:
                        budget_audit = _lambda_budget_audit(
                            result["backend_observations"],
                            max_unique_lambdas=int(lambda_budget["effective"]),
                        )
                        result["lambda_budget_audit"] = budget_audit
                        if not budget_audit["passed"]:
                            budget_failures.append(str(case_path))
                        if (
                            args.strict_backend
                            and not result["backend_audit"]["passed"]
                        ):
                            for violation in result["backend_audit"]["violations"]:
                                backend_violations.append(
                                    {"case_path": str(case_path), **violation}
                                )
                    payload["cases"].append(result)
                    if error is not None:
                        first_error = error
                        checkpoint_state = "failed"
                    elif case_index + 1 < len(cases):
                        checkpoint_state = "running"
                    elif budget_failures or backend_violations:
                        checkpoint_state = "failed_contract"
                    else:
                        checkpoint_state = "complete"
                    checkpoint_sequence += 1
                    _write_progress_checkpoint(
                        output_json=output_json,
                        payload=payload,
                        total_case_count=len(cases),
                        state=checkpoint_state,
                        checkpoint_sequence=checkpoint_sequence,
                        strict_backend=bool(args.strict_backend),
                        backend_violations=backend_violations,
                        budget_failures=budget_failures,
                        first_error=first_error,
                    )
                    if error is not None:
                        break
    except KeyboardInterrupt:
        checkpoint_sequence += 1
        payload["interruption"] = {
            "type": "KeyboardInterrupt",
            "at_utc": _utc_now(),
            "active_case_path": active_case_path,
        }
        _write_progress_checkpoint(
            output_json=output_json,
            payload=payload,
            total_case_count=len(cases),
            state="interrupted",
            checkpoint_sequence=checkpoint_sequence,
            strict_backend=bool(args.strict_backend),
            backend_violations=backend_violations,
            budget_failures=budget_failures,
            first_error=first_error,
            active_case_path=active_case_path,
        )
        raise
    if first_error is not None:
        raise RuntimeError(
            f"Benchmark case failed; partial diagnostics were written to {output_json}"
        ) from first_error
    if budget_failures:
        raise RuntimeError(
            "The online controller exceeded the requested unique-lambda budget for: "
            + ", ".join(budget_failures)
        )
    if backend_violations:
        preview = "; ".join(
            f"{Path(item['case_path']).name}:row={item.get('search_row_index')}:"
            f"{item['kind']}"
            for item in backend_violations[:8]
        )
        raise StrictBackendError(
            f"Strict backend validation failed ({len(backend_violations)} violation(s)); "
            f"diagnostics were written to {output_json}. {preview}"
        )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
