#!/usr/bin/env python3
"""Run the CliPP2Sim1K benchmark as isolated, resumable case processes.

The scheduler deliberately keeps orchestration separate from the CliPP2 CLI:
each tumor is fitted in a fresh process, truth is evaluated only after model
selection, and a case becomes complete only after its six output TSVs pass
validation and are promoted as one directory rename.

Example (from the repository root)::

    conda run -n ml1 python -u runs/run_clipp2sim1k_benchmark.py \
        --outdir runs/clipp2sim1k_partition_guided_admm

Use ``--preflight-only`` to create and verify the immutable run provenance and
SQLite state without starting a tumor fit.  ``--max-cases`` limits the
workload for a smoke run without changing the immutable scientific config.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import fcntl
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "CliPP2Sim1K_TSV"
DEFAULT_SIMULATION_ROOT = WORKSPACE_ROOT / "CliPP2Sim1K"
if str(WORKSPACE_ROOT) not in sys.path:
    # The Git root is also the ``CliPP2`` package directory.  Importing the
    # package from a source checkout therefore requires its parent, not the Git
    # root itself, on sys.path.
    sys.path.insert(0, str(WORKSPACE_ROOT))

EXPECTED_CASES = 1_000
MAX_ATTEMPTS = 2
SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 6 * 60 * 60
DEFAULT_TERMINATION_GRACE_SECONDS = 30.0
CONDA_ENVIRONMENT = "ml1"
DEFAULT_WORKSET_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_COMPRESSED_CACHE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_WORKSET_ADD_BATCH = 64
DEFAULT_WORKSET_MAX_EXPANSIONS = 16
DEFAULT_CERTIFICATE_MAX_ITER = 512
DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS = 2
DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE = 1.0
EXACTNESS_PROVENANCE_VERSION = 1
ACCEPTED_EXACT_CERTIFICATE_STATUSES = frozenset(
    {
        "certified",
        "input_dual_retained",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
        "zero_penalty_no_dual_needed",
    }
)

CASE_RE = re.compile(
    r"^(?P<depth>\d+)_(?P<K>\d+)_(?P<purity>\d+(?:\.\d+)?)_"
    r"(?P<cna>\d+(?:\.\d+)?)_S(?P<S>\d+)_Lm(?P<Lm>\d+)_"
    r"M(?P<M>\d+)_rep(?P<rep>\d+)$"
)

EXPECTED_DEPTHS = (50, 100, 300, 500, 1_000)
EXPECTED_REGION_COUNTS = (1, 2, 5, 10, 20)
EXPECTED_TARGET_MUTATION_COUNTS = (300, 600, 1_000, 2_000, 4_000)
EXPECTED_CNA_RATES = (0.0, 0.2)
EXPECTED_PURITY_REPLICATES = ((0.3, 0), (0.6, 0), (0.6, 1), (0.9, 0))

ARTIFACT_SUFFIXES = (
    "mutation_clusters.tsv",
    "cluster_centers.tsv",
    "mutation_region_multiplicity.tsv",
    "lambda_search.tsv",
    "simulation_eval.tsv",
    "run_summary.tsv",
)

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "mutation_clusters.tsv": (
        "tumor_id",
        "mutation_id",
        "cluster_label",
        "cluster_size",
    ),
    "cluster_centers.tsv": ("tumor_id", "cluster_label", "cluster_size"),
    "mutation_region_multiplicity.tsv": (
        "tumor_id",
        "mutation_id",
        "region_id",
        "cluster_label",
        "phi",
        "summary_phi",
        "major_cn",
        "minor_cn",
        "multiplicity_estimated",
        "gamma_major",
        "major_call",
        "multiplicity_call",
    ),
    "lambda_search.tsv": (
        "tumor_id",
        "selection_score_name",
        "selection_step",
        "lambda",
        "lambda_applicable",
        "candidate_pool_source",
        "partition_icl",
        "n_clusters",
        "fixed_objective_kkt_residual",
        "raw_kkt_eligible",
        "bic_selection_eligible",
        "eligible_for_selection",
        "inner_solver",
        "admm_iterations",
        "graph_name",
        "num_edges",
        "tol",
        "input_data_hash",
        "lambda_search_mode",
        "lambda_path_prespecified",
        "evaluation_mode",
        "candidate_evaluation_elapsed_seconds",
        "ARI",
        "cp_rmse",
        "raw_cp_rmse",
        "summary_cp_rmse",
        "bic_refit_cp_rmse",
        "multiplicity_f1",
        "multiplicity_asymmetric_f1",
        "multiplicity_estimable_f1",
        "estimated_clonal_fraction",
        "true_clonal_fraction",
        "clonal_fraction_error",
        "estimated_clusters",
        "true_clusters",
        "n_eval_mutations",
        "n_filtered_mutations",
        "is_selection_optimal",
        "is_selected_best_row",
    ),
    "simulation_eval.tsv": (
        "ARI",
        "cp_rmse",
        "raw_cp_rmse",
        "summary_cp_rmse",
        "bic_refit_cp_rmse",
        "multiplicity_f1",
        "multiplicity_asymmetric_f1",
        "multiplicity_estimable_f1",
        "estimated_clonal_fraction",
        "true_clonal_fraction",
        "clonal_fraction_error",
        "true_clusters",
        "estimated_clusters",
        "n_eval_mutations",
        "n_filtered_mutations",
    ),
    "run_summary.tsv": (
        "tumor_id",
        "device",
        "dtype",
        "lambda_search_mode",
        "selection_score_name",
        "lambda_path_prespecified",
        "evaluate_all_candidates",
        "inner_solver",
        "selected_lambda",
        "selected_candidate_pool_source",
        "selected_kkt_residual",
        "n_clusters",
        "num_regions",
        "num_mutations",
        "input_data_hash",
        "tol",
        "major_prior",
        "bic_df_scale",
        "bic_cluster_penalty",
        "selection_eligible",
        "num_candidates_all",
        "graph_name",
        "ARI",
        "cp_rmse",
        "raw_cp_rmse",
        "summary_cp_rmse",
        "multiplicity_f1",
        "multiplicity_asymmetric_f1",
        "multiplicity_estimable_f1",
        "benchmark_config_sha256",
        "benchmark_source_sha256",
        "benchmark_environment_sha256",
        "benchmark_cohort_sha256",
        "benchmark_input_sha256",
        "benchmark_truth_sha256",
    ),
}

LAMBDA_TRUTH_COLUMNS = (
    "ARI",
    "cp_rmse",
    "raw_cp_rmse",
    "summary_cp_rmse",
    "bic_refit_cp_rmse",
    "multiplicity_f1",
    "multiplicity_asymmetric_f1",
    "multiplicity_estimable_f1",
    "estimated_clonal_fraction",
    "true_clonal_fraction",
    "clonal_fraction_error",
    "estimated_clusters",
    "true_clusters",
    "n_eval_mutations",
    "n_filtered_mutations",
)

STATUS_COLUMNS = (
    "schedule_rank",
    "tumor_id",
    "status",
    "attempts",
    "backend",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "returncode",
    "error_type",
    "error_message",
    "log_path",
    "bundle_path",
    "input_file",
    "input_sha256",
    "truth_sha256",
)

FAILURE_COLUMNS = (
    "tumor_id",
    "attempt",
    "status",
    "backend",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "returncode",
    "timed_out",
    "error_type",
    "error_message",
    "log_path",
    "timeout_seconds",
    "termination_grace_seconds",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_or_verify(path: Path, payload: bytes, *, label: str) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise RuntimeError(
                f"Immutable {label} mismatch at {path}. Refusing to resume; "
                "use a new --outdir for a changed cohort, source tree, or configuration."
            )
        return
    _atomic_write_bytes(path, payload)


def _write_document_and_hash(path: Path, value: object, *, label: str) -> str:
    payload = _canonical_json_bytes(value)
    digest = _sha256_bytes(payload)
    _write_once_or_verify(path, payload, label=label)
    _write_once_or_verify(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
        label=f"{label} hash",
    )
    return digest


def _tree_digest(root: Path) -> tuple[str, int, int]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Truth directory has no files: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"Truth trees may not contain symlinks: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_digest = _sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def _truth_cluster_dimensions(truth_dir: Path) -> tuple[int, int]:
    truth_path = truth_dir / "truth.txt"
    if not truth_path.is_file():
        raise FileNotFoundError(f"Simulation truth is missing {truth_path}")
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "cluster_id" not in reader.fieldnames:
            raise ValueError(f"Simulation truth lacks cluster_id: {truth_path}")
        clusters: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                clusters.append(int(row["cluster_id"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid cluster_id on row {row_number} of {truth_path}: {row.get('cluster_id')!r}"
                ) from exc
    if not clusters:
        raise ValueError(f"Simulation truth contains no mutation rows: {truth_path}")
    return len(clusters), len(set(clusters))


def _factorial_key(
    case: Mapping[str, object],
) -> tuple[int, int, int, float, float, int]:
    return (
        int(case["depth"]),
        int(case["region_count"]),
        int(case["target_mutation_count"]),
        float(case["cna_rate"]),
        float(case["purity"]),
        int(case["replicate"]),
    )


def _validate_factorial_design(cases: Sequence[Mapping[str, object]]) -> None:
    expected = {
        (depth, regions, target_m, cna, purity, replicate)
        for depth in EXPECTED_DEPTHS
        for regions in EXPECTED_REGION_COUNTS
        for target_m in EXPECTED_TARGET_MUTATION_COUNTS
        for cna in EXPECTED_CNA_RATES
        for purity, replicate in EXPECTED_PURITY_REPLICATES
    }
    actual = [_factorial_key(case) for case in cases]
    if len(actual) != len(set(actual)):
        raise ValueError(
            "CliPP2Sim1K factorial design contains duplicate condition cells."
        )
    actual_set = set(actual)
    if actual_set != expected:
        missing = sorted(expected - actual_set)
        extra = sorted(actual_set - expected)
        raise ValueError(
            "CliPP2Sim1K factorial design mismatch: "
            f"missing={missing[:3]} (n={len(missing)}), extra={extra[:3]} (n={len(extra)})."
        )


def build_cohort_manifest(
    input_dir: Path,
    simulation_root: Path,
    *,
    expected_cases: int = EXPECTED_CASES,
) -> dict[str, object]:
    """Hash and cross-check the complete input/truth cohort."""

    input_dir = input_dir.resolve()
    simulation_root = simulation_root.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not simulation_root.is_dir():
        raise FileNotFoundError(
            f"Simulation truth root does not exist: {simulation_root}"
        )

    inputs = sorted(input_dir.glob("*.tsv"), key=lambda path: path.name)
    if len(inputs) != expected_cases:
        raise ValueError(
            f"CliPP2Sim1K preflight requires exactly {expected_cases} input TSVs; "
            f"found {len(inputs)} in {input_dir}."
        )
    stems = [path.stem for path in inputs]
    if len(set(stems)) != len(stems):
        raise ValueError("Input TSV tumor identifiers are not unique.")
    invalid_names = [stem for stem in stems if CASE_RE.fullmatch(stem) is None]
    if invalid_names:
        raise ValueError(
            f"Input TSV names do not match the Sim1K schema: {invalid_names[:3]}"
        )

    truth_dirs = {
        path.name: path for path in simulation_root.iterdir() if path.is_dir()
    }
    missing_truth = sorted(set(stems) - set(truth_dirs))
    extra_truth = sorted(set(truth_dirs) - set(stems))
    if missing_truth or extra_truth:
        raise ValueError(
            "Input/truth cohort mismatch: "
            f"missing_truth={missing_truth[:3]} (n={len(missing_truth)}), "
            f"extra_truth={extra_truth[:3]} (n={len(extra_truth)})."
        )

    from CliPP2.io.data import load_tumor_tsv
    from CliPP2.runners.model_selection import _input_data_hash

    cases: list[dict[str, object]] = []
    for input_path in inputs:
        tumor_id = input_path.stem
        match = CASE_RE.fullmatch(tumor_id)
        assert match is not None
        data = load_tumor_tsv(input_path, missing_cna_policy="error")
        mutation_count = int(match.group("M"))
        region_count = int(match.group("S"))
        true_cluster_count = int(match.group("K"))
        if (
            int(data.num_mutations) != mutation_count
            or int(data.num_regions) != region_count
        ):
            raise ValueError(
                f"Input dimensions disagree with filename for {tumor_id}: "
                f"loaded M={data.num_mutations}, S={data.num_regions}; "
                f"filename M={mutation_count}, S={region_count}."
            )
        truth_mutations, truth_clusters = _truth_cluster_dimensions(
            truth_dirs[tumor_id]
        )
        if truth_mutations != mutation_count or truth_clusters != true_cluster_count:
            raise ValueError(
                f"Truth dimensions disagree with filename for {tumor_id}: "
                f"truth M={truth_mutations}, K={truth_clusters}; "
                f"filename M={mutation_count}, K={true_cluster_count}."
            )
        truth_sha256, truth_file_count, truth_bytes = _tree_digest(truth_dirs[tumor_id])
        input_size = input_path.stat().st_size
        cases.append(
            {
                "tumor_id": tumor_id,
                "input_file": input_path.name,
                "input_bytes": input_size,
                "input_sha256": _sha256_file(input_path),
                "truth_sha256": truth_sha256,
                "truth_file_count": truth_file_count,
                "truth_bytes": truth_bytes,
                "input_data_hash": _input_data_hash(data),
                "mutation_count": mutation_count,
                "region_count": region_count,
                "true_cluster_count": true_cluster_count,
                "depth": int(match.group("depth")),
                "purity": float(match.group("purity")),
                "cna_rate": float(match.group("cna")),
                "target_mutation_count": int(match.group("Lm")),
                "replicate": int(match.group("rep")),
                "workload_proxy_m2s": mutation_count * mutation_count * region_count,
            }
        )

    if expected_cases == EXPECTED_CASES:
        _validate_factorial_design(cases)

    scheduled = sorted(
        cases, key=lambda row: (int(row["workload_proxy_m2s"]), str(row["tumor_id"]))
    )
    ranks = {str(row["tumor_id"]): rank for rank, row in enumerate(scheduled, start=1)}
    for case in cases:
        case["schedule_rank"] = ranks[str(case["tumor_id"])]

    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": "CliPP2Sim1K",
        "expected_cases": expected_cases,
        "input_dir": str(input_dir),
        "simulation_root": str(simulation_root),
        "case_count": len(cases),
        "cases": cases,
    }


def build_source_manifest(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    """Hash the canonical package source plus the benchmark entry point.

    This repository's Git root is the package directory itself.  The previous
    outer-layout lookup (``repo_root / "CliPP2"``) consequently found no solver
    source and allowed code changes during a run to escape the resume guard.
    """

    repo_root = repo_root.resolve()
    excluded_directories = {
        ".git",
        ".github",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "runs",
        "sim_v2_results",
        "tests",
    }
    candidates = list(repo_root.glob("*.py"))
    for child in repo_root.iterdir():
        if (
            child.is_dir()
            and child.name not in excluded_directories
            and not child.name.startswith(".")
        ):
            candidates.extend(child.rglob("*.py"))
    candidates.extend(
        [
            repo_root / "pyproject.toml",
            repo_root / "runs" / Path(__file__).name,
        ]
    )
    unique = sorted({path.resolve() for path in candidates})
    files: list[dict[str, object]] = []
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"Required source file is missing: {path}")
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = str(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "files": files,
    }


def _installed_distributions() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower()] = str(distribution.version)
    return [{"name": name, "version": packages[name]} for name in sorted(packages)]


def collect_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "captured_at": _utc_now(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "packages": _installed_distributions(),
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        result["torch"] = {
            "version": str(torch.__version__),
            "cuda_build": str(torch.version.cuda),
            "cuda_available": cuda_available,
            "cudnn_version": None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - only used on broken environments
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


def stable_environment_fingerprint(
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return deterministic environment identity suitable for resume checks."""

    stable = dict(collect_environment() if environment is None else environment)
    stable.pop("captured_at", None)
    return stable


def _assert_conda_environment(expected: str = CONDA_ENVIRONMENT) -> None:
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    prefix_name = Path(os.environ.get("CONDA_PREFIX", "")).name
    if active != expected and prefix_name != expected:
        raise RuntimeError(
            f"This benchmark must run in conda environment {expected!r}; "
            f"CONDA_DEFAULT_ENV={active!r}, CONDA_PREFIX basename={prefix_name!r}."
        )


def _assert_backend_available(device: str, *, allow_cpu: bool) -> None:
    if device == "cpu":
        if not allow_cpu:
            raise RuntimeError(
                "CPU execution requires the explicit --allow-cpu option; no fallback is automatic."
            )
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError(
            "CUDA was requested but PyTorch could not be imported."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; refusing CPU fallback."
        )


def build_run_config(
    args: argparse.Namespace,
    *,
    cohort_sha256: str,
    source_sha256: str,
    environment_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort_sha256": cohort_sha256,
        "source_sha256": source_sha256,
        "environment_sha256": environment_sha256,
        "conda_environment": CONDA_ENVIRONMENT,
        "device": str(args.device),
        "dtype": "float64",
        "lambda_grid": None,
        "lambda_grid_mode": "partition_guided_admm",
        "selection_score": "partition_icl",
        "evaluate_all_candidates": False,
        "use_warm_starts": True,
        "finalize_selected_fit": True,
        "write_outputs": True,
        "missing_cna_policy": "error",
        "outer_max_iter": int(args.outer_max_iter),
        "inner_max_iter": int(args.inner_max_iter),
        "tol": float(args.tol),
        "summary_tol": float(args.summary_tol),
        "bic_partition_tol": float(args.bic_partition_tol),
        "major_prior": float(args.major_prior),
        "inner_backend": str(args.inner_backend).strip().lower().replace("-", "_"),
        "workset_max_bytes": int(args.workset_max_bytes),
        "compressed_cache_max_bytes": int(args.compressed_cache_max_bytes),
        "dense_fallback_policy": str(args.dense_fallback_policy)
        .strip()
        .lower()
        .replace("-", "_"),
        "workset_add_batch": int(args.workset_add_batch),
        "workset_max_expansions": int(args.workset_max_expansions),
        "certificate_max_iter": int(args.certificate_max_iter),
        "certificate_refinement_rounds": int(args.certificate_refinement_rounds),
        "certificate_column_tol_scale": float(args.certificate_column_tol_scale),
        "allow_heuristic_structure_splits": bool(args.allow_heuristic_structure_splits),
        "materialize_full_dual": bool(args.materialize_full_dual),
        "bic_df_scale": float(args.bic_df_scale),
        "bic_cluster_penalty": float(args.bic_cluster_penalty),
        "max_attempts": MAX_ATTEMPTS,
        "subprocesses": "one_fresh_serial_process_per_case",
        "artifact_promotion": "validated_atomic_directory_rename_plus_flat_copies",
    }


def _fit_options_from_config(config: Mapping[str, object]):
    """Recreate the immutable worker solver options, including exact backends."""

    from CliPP2.core.model import FitOptions

    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=int(config["outer_max_iter"]),
        inner_max_iter=int(config["inner_max_iter"]),
        tol=float(config["tol"]),
        summary_tol=float(config["summary_tol"]),
        bic_partition_tol=float(config["bic_partition_tol"]),
        major_prior=float(config["major_prior"]),
        device=str(config["device"]),
        dtype=str(config["dtype"]),
        inner_backend=str(config.get("inner_backend", "dense")),
        workset_max_bytes=int(
            config.get("workset_max_bytes", DEFAULT_WORKSET_MAX_BYTES)
        ),
        compressed_cache_max_bytes=int(
            config.get("compressed_cache_max_bytes", DEFAULT_COMPRESSED_CACHE_MAX_BYTES)
        ),
        dense_fallback_policy=str(config.get("dense_fallback_policy", "auto")),
        workset_add_batch=int(
            config.get("workset_add_batch", DEFAULT_WORKSET_ADD_BATCH)
        ),
        workset_max_expansions=int(
            config.get("workset_max_expansions", DEFAULT_WORKSET_MAX_EXPANSIONS)
        ),
        certificate_max_iter=int(
            config.get("certificate_max_iter", DEFAULT_CERTIFICATE_MAX_ITER)
        ),
        certificate_refinement_rounds=int(
            config.get(
                "certificate_refinement_rounds",
                DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
            )
        ),
        certificate_column_tol_scale=float(
            config.get(
                "certificate_column_tol_scale",
                DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
            )
        ),
        allow_heuristic_structure_splits=bool(
            config.get("allow_heuristic_structure_splits", True)
        ),
        materialize_full_dual=bool(config.get("materialize_full_dual", False)),
    )


class SchedulerLock(AbstractContextManager["SchedulerLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any | None = None

    def __enter__(self) -> "SchedulerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                f"Another benchmark scheduler holds {self.path}."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\nstarted_at={_utc_now()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


class BenchmarkState:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.connection.close()

    def initialize(self, manifest: Mapping[str, object]) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                tumor_id TEXT PRIMARY KEY,
                schedule_rank INTEGER NOT NULL UNIQUE,
                input_file TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                truth_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                backend TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                elapsed_seconds REAL,
                returncode INTEGER,
                error_type TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL DEFAULT '',
                bundle_path TEXT NOT NULL DEFAULT '',
                artifact_hashes_json TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS attempts (
                tumor_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                backend TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL DEFAULT '',
                elapsed_seconds REAL,
                returncode INTEGER,
                timed_out INTEGER NOT NULL DEFAULT 0,
                error_type TEXT NOT NULL DEFAULT '',
                error_message TEXT NOT NULL DEFAULT '',
                log_path TEXT NOT NULL,
                staging_path TEXT NOT NULL,
                timeout_seconds REAL,
                termination_grace_seconds REAL,
                PRIMARY KEY (tumor_id, attempt),
                FOREIGN KEY (tumor_id) REFERENCES cases(tumor_id)
            );
            """
        )
        attempt_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(attempts)").fetchall()
        }
        for column in ("timeout_seconds", "termination_grace_seconds"):
            if column not in attempt_columns:
                self.connection.execute(
                    f"ALTER TABLE attempts ADD COLUMN {column} REAL"
                )
        cases = list(manifest["cases"])  # type: ignore[arg-type]
        with self.connection:
            for case in cases:
                row = dict(case)
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO cases (
                        tumor_id, schedule_rank, input_file, input_sha256, truth_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["tumor_id"],
                        row["schedule_rank"],
                        row["input_file"],
                        row["input_sha256"],
                        row["truth_sha256"],
                    ),
                )
        database_rows = self.connection.execute(
            "SELECT tumor_id, schedule_rank, input_file, input_sha256, truth_sha256 FROM cases"
        ).fetchall()
        actual = {
            row["tumor_id"]: (
                int(row["schedule_rank"]),
                row["input_file"],
                row["input_sha256"],
                row["truth_sha256"],
            )
            for row in database_rows
        }
        expected = {
            str(dict(case)["tumor_id"]): (
                int(dict(case)["schedule_rank"]),
                dict(case)["input_file"],
                dict(case)["input_sha256"],
                dict(case)["truth_sha256"],
            )
            for case in cases
        }
        if actual != expected:
            raise RuntimeError(
                "SQLite cohort state differs from the immutable cohort manifest."
            )

    def recover_interrupted(self, max_attempts: int = MAX_ATTEMPTS) -> None:
        now = _utc_now()
        with self.connection:
            running = self.connection.execute(
                "SELECT tumor_id, attempts FROM cases WHERE status = 'running'"
            ).fetchall()
            for row in running:
                retry_status = (
                    "retry_pending" if int(row["attempts"]) < max_attempts else "failed"
                )
                self.connection.execute(
                    """
                    UPDATE cases SET status=?, finished_at=?, error_type=?, error_message=?
                    WHERE tumor_id=?
                    """,
                    (
                        retry_status,
                        now,
                        "SchedulerInterrupted",
                        "Scheduler stopped while the isolated case process was running.",
                        row["tumor_id"],
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE attempts SET status='interrupted', finished_at=?, error_type=?, error_message=?
                    WHERE tumor_id=? AND attempt=? AND status='running'
                    """,
                    (
                        now,
                        "SchedulerInterrupted",
                        "Scheduler stopped before recording a subprocess result.",
                        row["tumor_id"],
                        row["attempts"],
                    ),
                )

    def start_attempt(
        self,
        tumor_id: str,
        *,
        backend: str,
        log_path: Path,
        staging_path: Path,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        termination_grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
    ) -> int:
        now = _utc_now()
        with self.connection:
            row = self.connection.execute(
                "SELECT status, attempts FROM cases WHERE tumor_id=?", (tumor_id,)
            ).fetchone()
            if row is None:
                raise KeyError(tumor_id)
            if row["status"] not in {"pending", "retry_pending"}:
                raise RuntimeError(
                    f"Case {tumor_id} cannot start from status {row['status']!r}."
                )
            attempt = int(row["attempts"]) + 1
            if attempt > MAX_ATTEMPTS:
                raise RuntimeError(
                    f"Case {tumor_id} exhausted its {MAX_ATTEMPTS} attempts."
                )
            self.connection.execute(
                """
                UPDATE cases SET status='running', attempts=?, backend=?, started_at=?,
                    finished_at='', elapsed_seconds=NULL, returncode=NULL,
                    error_type='', error_message='', log_path=?
                WHERE tumor_id=?
                """,
                (attempt, backend, now, str(log_path), tumor_id),
            )
            self.connection.execute(
                """
                INSERT INTO attempts (
                    tumor_id, attempt, status, backend, started_at, log_path, staging_path,
                    timeout_seconds, termination_grace_seconds
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    tumor_id,
                    attempt,
                    backend,
                    now,
                    str(log_path),
                    str(staging_path),
                    float(timeout_seconds),
                    float(termination_grace_seconds),
                ),
            )
        return attempt

    def finish_success(
        self,
        tumor_id: str,
        attempt: int,
        *,
        backend: str,
        elapsed_seconds: float,
        returncode: int,
        bundle_path: Path,
        artifact_hashes: Mapping[str, str],
    ) -> None:
        now = _utc_now()
        hashes_json = json.dumps(dict(sorted(artifact_hashes.items())), sort_keys=True)
        with self.connection:
            self.connection.execute(
                """
                UPDATE cases SET status='succeeded', backend=?, finished_at=?, elapsed_seconds=?,
                    returncode=?, error_type='', error_message='', bundle_path=?, artifact_hashes_json=?
                WHERE tumor_id=? AND attempts=?
                """,
                (
                    backend,
                    now,
                    elapsed_seconds,
                    returncode,
                    str(bundle_path),
                    hashes_json,
                    tumor_id,
                    attempt,
                ),
            )
            self.connection.execute(
                """
                UPDATE attempts SET status='succeeded', finished_at=?, elapsed_seconds=?, returncode=?
                WHERE tumor_id=? AND attempt=?
                """,
                (now, elapsed_seconds, returncode, tumor_id, attempt),
            )

    def finish_failure(
        self,
        tumor_id: str,
        attempt: int,
        *,
        elapsed_seconds: float,
        returncode: int | None,
        timed_out: bool,
        error_type: str,
        error_message: str,
    ) -> str:
        now = _utc_now()
        status = "retry_pending" if attempt < MAX_ATTEMPTS else "failed"
        attempt_status = "timed_out" if timed_out else "failed"
        with self.connection:
            self.connection.execute(
                """
                UPDATE cases SET status=?, finished_at=?, elapsed_seconds=?, returncode=?,
                    error_type=?, error_message=? WHERE tumor_id=? AND attempts=?
                """,
                (
                    status,
                    now,
                    elapsed_seconds,
                    returncode,
                    error_type,
                    error_message,
                    tumor_id,
                    attempt,
                ),
            )
            self.connection.execute(
                """
                UPDATE attempts SET status=?, finished_at=?, elapsed_seconds=?, returncode=?,
                    timed_out=?, error_type=?, error_message=?
                WHERE tumor_id=? AND attempt=?
                """,
                (
                    attempt_status,
                    now,
                    elapsed_seconds,
                    returncode,
                    int(timed_out),
                    error_type,
                    error_message,
                    tumor_id,
                    attempt,
                ),
            )
        return status

    def recover_success(
        self,
        tumor_id: str,
        *,
        backend: str,
        bundle_path: Path,
        artifact_hashes: Mapping[str, str],
    ) -> None:
        hashes_json = json.dumps(dict(sorted(artifact_hashes.items())), sort_keys=True)
        with self.connection:
            row = self.connection.execute(
                "SELECT attempts FROM cases WHERE tumor_id=?", (tumor_id,)
            ).fetchone()
            if row is None:
                raise KeyError(tumor_id)
            self.connection.execute(
                """
                UPDATE cases SET status='succeeded', backend=?, finished_at=?, returncode=0,
                    error_type='', error_message='', bundle_path=?, artifact_hashes_json=?
                WHERE tumor_id=?
                """,
                (backend, _utc_now(), str(bundle_path), hashes_json, tumor_id),
            )
            attempt = int(row["attempts"])
            if attempt > 0:
                self.connection.execute(
                    """
                    UPDATE attempts SET status='succeeded', finished_at=?, returncode=0,
                        timed_out=0, error_type='', error_message=''
                    WHERE tumor_id=? AND attempt=?
                    """,
                    (_utc_now(), tumor_id, attempt),
                )

    def case_rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM cases ORDER BY schedule_rank"
        ).fetchall()

    def attempt_rows(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM attempts WHERE status != 'succeeded' ORDER BY started_at, tumor_id, attempt"
        ).fetchall()


def _atomic_write_tsv(
    path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def export_state(state: BenchmarkState, outdir: Path) -> None:
    cases = [dict(row) for row in state.case_rows()]
    attempts = [dict(row) for row in state.attempt_rows()]
    _atomic_write_tsv(outdir / "status.tsv", STATUS_COLUMNS, cases)
    _atomic_write_tsv(outdir / "failures.tsv", FAILURE_COLUMNS, attempts)


def pending_scheduled_cases(
    state: BenchmarkState,
    manifest: Mapping[str, object],
    *,
    max_cases: int | None,
) -> list[dict[str, object]]:
    current_status = {
        str(row["tumor_id"]): str(row["status"]) for row in state.case_rows()
    }
    cases = sorted(
        (dict(case) for case in manifest["cases"]),  # type: ignore[index]
        key=lambda row: int(row["schedule_rank"]),
    )
    pending = [
        case
        for case in cases
        if current_status[str(case["tumor_id"])] not in {"succeeded", "failed"}
    ]
    return pending if max_cases is None else pending[: int(max_cases)]


def expected_artifact_names(tumor_id: str) -> tuple[str, ...]:
    return tuple(f"{tumor_id}_{suffix}" for suffix in ARTIFACT_SUFFIXES)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def _validate_tsv(
    path: Path,
    *,
    tumor_id: str,
    suffix: str,
) -> tuple[list[str], list[list[str]], int]:
    required = REQUIRED_COLUMNS[suffix]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty output artifact: {path}") from exc
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate TSV columns in {path}")
        missing = [column for column in required if column not in header]
        if missing:
            raise ValueError(f"Missing required columns in {path}: {missing}")
        tumor_index = header.index("tumor_id") if "tumor_id" in header else None
        captured: list[list[str]] = []
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"Malformed TSV row {count + 2} in {path}: expected {len(header)} fields, got {len(row)}."
                )
            if tumor_index is not None and row[tumor_index] != tumor_id:
                raise ValueError(
                    f"Tumor identifier mismatch in {path}: expected {tumor_id!r}, got {row[tumor_index]!r}."
                )
            captured.append(row)
            count += 1
    if count == 0:
        raise ValueError(f"Output artifact contains no data rows: {path}")
    return header, captured, count


def _records(parsed: tuple[list[str], list[list[str]], int]) -> list[dict[str, str]]:
    header, rows, _ = parsed
    return [dict(zip(header, row, strict=True)) for row in rows]


def _finite_float(record: Mapping[str, str], column: str, *, context: str) -> float:
    value = record.get(column, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {column}={value!r} in {context}.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {column}={value!r} in {context}.")
    return parsed


def _integer(record: Mapping[str, str], column: str, *, context: str) -> int:
    value = _finite_float(record, column, context=context)
    rounded = int(round(value))
    if not math.isclose(value, float(rounded), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Non-integer {column}={value!r} in {context}.")
    return rounded


def _is_blank_or_nan(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"", "nan"}:
        return True
    try:
        return math.isnan(float(normalized))
    except ValueError:
        return False


def _optional_finite_float(
    record: Mapping[str, str], column: str, *, context: str
) -> float | None:
    value = record.get(column, "")
    if _is_blank_or_nan(value):
        return None
    return _finite_float(record, column, context=context)


def _assert_close(left: float, right: float, *, label: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"Cross-artifact mismatch for {label}: {left!r} != {right!r}.")


def _optional_exactness_schema(
    record: Mapping[str, str], *, context: str
) -> int | None:
    value = record.get("exactness_provenance_version", "")
    if _is_blank_or_nan(value):
        return None
    return _integer(record, "exactness_provenance_version", context=context)


def _validate_selected_exactness_provenance(
    *,
    summary: Mapping[str, str],
    selected: Mapping[str, str],
    selected_kkt: float,
    config: Mapping[str, object],
    tumor_id: str,
) -> str:
    """Validate schema-v1 exactness, or the historical dense-ADMM contract."""

    summary_context = f"run summary for {tumor_id}"
    selected_context = f"selected lambda-search row for {tumor_id}"
    summary_schema = _optional_exactness_schema(summary, context=summary_context)
    selected_schema = _optional_exactness_schema(selected, context=selected_context)
    if (summary_schema is None) != (selected_schema is None):
        raise ValueError(
            f"Exactness provenance is present in only one selected artifact for {tumor_id}."
        )

    if selected_schema is None:
        requested_backend = (
            str(config.get("inner_backend", "dense")).strip().lower().replace("-", "_")
        )
        if requested_backend != "dense":
            raise ValueError(
                f"Selected case {tumor_id} lacks schema-v1 exactness provenance "
                f"for requested backend {requested_backend!r}."
            )
        if selected.get("inner_solver", "") != "admm_complete_graph":
            raise ValueError(
                f"Legacy selected case {tumor_id} was not solved by complete-graph ADMM."
            )
        if _integer(selected, "admm_iterations", context=selected_context) <= 0:
            raise ValueError(f"Selected case {tumor_id} has no ADMM iterations.")
        return "legacy_dense_admm"

    if (
        selected_schema != EXACTNESS_PROVENANCE_VERSION
        or summary_schema != EXACTNESS_PROVENANCE_VERSION
    ):
        raise ValueError(
            f"Unsupported exactness provenance schema for {tumor_id}: "
            f"summary={summary_schema}, selected={selected_schema}."
        )

    expected_text = {
        "estimator_role": "raw_fused_lambda_path",
        "certificate_scope": "full_original_graph",
        "certificate_gradient_scope": "observed_objective",
    }
    cross_checked_text = (
        "estimator_role",
        "objective_spec_hash",
        "original_graph_hash",
        "certificate_problem_hash",
        "certificate_scope",
        "certificate_gradient_scope",
        "full_kkt_certificate_status",
        "inner_backend",
    )
    selected_text: dict[str, str] = {}
    for column in cross_checked_text:
        left = str(summary.get(column, "")).strip()
        right = str(selected.get(column, "")).strip()
        if not left or not right:
            raise ValueError(
                f"Missing schema-v1 exactness field {column!r} for {tumor_id}."
            )
        if left != right:
            raise ValueError(
                f"Exactness field {column!r} disagrees between selected artifacts "
                f"for {tumor_id}: {left!r} != {right!r}."
            )
        selected_text[column] = right
    for column, expected in expected_text.items():
        if selected_text[column] != expected:
            raise ValueError(
                f"Invalid schema-v1 exactness field {column}={selected_text[column]!r} "
                f"for {tumor_id}; expected {expected!r}."
            )

    for column in ("objective_faithful", "full_kkt_certified"):
        if column not in summary or column not in selected:
            raise ValueError(
                f"Missing schema-v1 exactness field {column!r} for {tumor_id}."
            )
        summary_value = _parse_bool(summary[column])
        selected_value = _parse_bool(selected[column])
        if summary_value != selected_value:
            raise ValueError(
                f"Exactness field {column!r} disagrees between selected artifacts "
                f"for {tumor_id}."
            )
        if not selected_value:
            raise ValueError(
                f"Selected case {tumor_id} has {column}=False in schema-v1 provenance."
            )

    status = selected_text["full_kkt_certificate_status"]
    if status not in ACCEPTED_EXACT_CERTIFICATE_STATUSES:
        raise ValueError(
            f"Selected case {tumor_id} has unaccepted full KKT certificate status "
            f"{status!r}."
        )
    selected_tolerance = _finite_float(
        selected, "full_kkt_tolerance", context=selected_context
    )
    summary_tolerance = _finite_float(
        summary, "full_kkt_tolerance", context=summary_context
    )
    _assert_close(
        selected_tolerance,
        summary_tolerance,
        label=f"{tumor_id} full KKT tolerance",
    )
    if selected_tolerance <= 0.0 or selected_kkt > selected_tolerance + 1e-12:
        raise ValueError(
            f"Selected case {tumor_id} does not satisfy its schema-v1 full KKT "
            f"tolerance: {selected_kkt} > {selected_tolerance}."
        )
    summary_backend_iterations = _integer(
        summary, "backend_iterations", context=summary_context
    )
    selected_backend_iterations = _integer(
        selected, "backend_iterations", context=selected_context
    )
    if summary_backend_iterations != selected_backend_iterations:
        raise ValueError(
            f"Backend iteration count disagrees between selected artifacts for {tumor_id}."
        )
    if selected_backend_iterations < 0:
        raise ValueError(f"Negative backend iteration count for {tumor_id}.")
    return "schema_v1_exact"


def _validate_metric_range(
    value: float, *, label: str, lower: float, upper: float
) -> None:
    if value < lower - 1e-12 or value > upper + 1e-12:
        raise ValueError(f"Metric {label}={value!r} is outside [{lower}, {upper}].")


def validate_case_artifacts(
    directory: Path,
    tumor_id: str,
    *,
    expected_device: str,
    input_file: Path,
    expected_case: Mapping[str, object],
    config: Mapping[str, object],
    config_sha256: str,
) -> dict[str, str]:
    from CliPP2.io.data import load_tumor_tsv
    from CliPP2.runners.model_selection import _input_data_hash

    expected = set(expected_artifact_names(tumor_id))
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"Case {tumor_id} must contain exactly six expected TSV artifacts; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )

    parsed: dict[str, tuple[list[str], list[list[str]], int]] = {}
    hashes: dict[str, str] = {}
    for suffix in ARTIFACT_SUFFIXES:
        name = f"{tumor_id}_{suffix}"
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Invalid output artifact: {path}")
        parsed[suffix] = _validate_tsv(path, tumor_id=tumor_id, suffix=suffix)
        hashes[name] = _sha256_file(path)

    data = load_tumor_tsv(input_file, missing_cna_policy="error")
    mutation_count = int(expected_case["mutation_count"])
    region_count = int(expected_case["region_count"])
    true_cluster_count = int(expected_case["true_cluster_count"])
    if data.num_mutations != mutation_count or data.num_regions != region_count:
        raise ValueError(
            f"Validation input dimensions changed for {tumor_id}: "
            f"M={data.num_mutations}, S={data.num_regions}."
        )
    input_data_hash = _input_data_hash(data)
    if input_data_hash != str(expected_case["input_data_hash"]):
        raise ValueError(f"Loaded input-data hash mismatch for {tumor_id}.")

    summary_header, summary_rows, summary_count = parsed["run_summary.tsv"]
    if summary_count != 1:
        raise ValueError(f"Run summary for {tumor_id} must contain exactly one row.")
    summary = dict(zip(summary_header, summary_rows[0], strict=True))
    required_values = {
        "dtype": "float64",
        "lambda_search_mode": "partition_guided_admm",
        "selection_score_name": "partition_icl",
        "selected_candidate_pool_source": "raw_fused_lambda_path",
    }
    for column, expected_value in required_values.items():
        if summary[column] != expected_value:
            raise ValueError(
                f"Run summary invariant failed for {tumor_id}: {column}={summary[column]!r}, "
                f"expected {expected_value!r}."
            )
    allowed_devices = {expected_device}
    if (
        expected_device == "cuda"
        and str(config.get("dense_fallback_policy", "auto"))
        .strip()
        .lower()
        .replace("-", "_")
        == "cpu_allowed"
    ):
        allowed_devices.add("cpu")
    if summary["device"] not in allowed_devices:
        raise ValueError(
            f"Run summary invariant failed for {tumor_id}: "
            f"device={summary['device']!r}, expected one of {sorted(allowed_devices)!r}."
        )
    if _parse_bool(summary["lambda_path_prespecified"]):
        raise ValueError(f"Case {tumor_id} used a prespecified lambda path.")
    if _parse_bool(summary["evaluate_all_candidates"]):
        raise ValueError(f"Case {tumor_id} used truth during candidate evaluation.")

    provenance_values = {
        "benchmark_config_sha256": config_sha256,
        "benchmark_source_sha256": str(config["source_sha256"]),
        "benchmark_environment_sha256": str(config["environment_sha256"]),
        "benchmark_cohort_sha256": str(config["cohort_sha256"]),
        "benchmark_input_sha256": str(expected_case["input_sha256"]),
        "benchmark_truth_sha256": str(expected_case["truth_sha256"]),
    }
    for column, expected_value in provenance_values.items():
        if summary[column] != expected_value:
            raise ValueError(
                f"Run-summary provenance mismatch for {tumor_id}: "
                f"{column}={summary[column]!r}, expected {expected_value!r}."
            )

    summary_context = f"run summary for {tumor_id}"
    if _integer(summary, "num_mutations", context=summary_context) != mutation_count:
        raise ValueError(f"Run-summary mutation count mismatch for {tumor_id}.")
    if _integer(summary, "num_regions", context=summary_context) != region_count:
        raise ValueError(f"Run-summary region count mismatch for {tumor_id}.")
    if summary["input_data_hash"] != input_data_hash:
        raise ValueError(f"Run-summary input-data hash mismatch for {tumor_id}.")
    _assert_close(
        _finite_float(summary, "major_prior", context=summary_context),
        float(config["major_prior"]),
        label=f"{tumor_id} major_prior",
    )
    _assert_close(
        _finite_float(summary, "bic_df_scale", context=summary_context),
        float(config["bic_df_scale"]),
        label=f"{tumor_id} bic_df_scale",
    )
    _assert_close(
        _finite_float(summary, "bic_cluster_penalty", context=summary_context),
        float(config["bic_cluster_penalty"]),
        label=f"{tumor_id} bic_cluster_penalty",
    )
    if not _parse_bool(summary["selection_eligible"]):
        raise ValueError(f"Selected fit for {tumor_id} is not selection eligible.")

    mutation_records = _records(parsed["mutation_clusters.tsv"])
    if len(mutation_records) != mutation_count:
        raise ValueError(
            f"Mutation output for {tumor_id} has {len(mutation_records)} rows; expected {mutation_count}."
        )
    mutation_ids = [record["mutation_id"] for record in mutation_records]
    if (
        mutation_ids != list(data.mutation_ids)
        or len(set(mutation_ids)) != mutation_count
    ):
        raise ValueError(f"Mutation IDs/order do not match the input for {tumor_id}.")
    mutation_label_by_id: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    for record in mutation_records:
        context = f"mutation row {record['mutation_id']} for {tumor_id}"
        label = _integer(record, "cluster_label", context=context)
        if label <= 0:
            raise ValueError(f"Nonpositive cluster label in {context}.")
        mutation_label_by_id[record["mutation_id"]] = label
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    for record in mutation_records:
        label = mutation_label_by_id[record["mutation_id"]]
        reported_size = _integer(
            record, "cluster_size", context=f"mutation output for {tumor_id}"
        )
        if reported_size != cluster_counts[label]:
            raise ValueError(
                f"Mutation-level cluster_size is inconsistent for {tumor_id}, cluster {label}."
            )

    center_records = _records(parsed["cluster_centers.tsv"])
    estimated_clusters = _integer(summary, "n_clusters", context=summary_context)
    expected_labels = list(range(1, estimated_clusters + 1))
    center_labels = [
        _integer(record, "cluster_label", context=f"cluster centers for {tumor_id}")
        for record in center_records
    ]
    if len(center_records) != estimated_clusters or center_labels != expected_labels:
        raise ValueError(
            f"Cluster-center labels for {tumor_id} must be exactly 1..{estimated_clusters}."
        )
    if sorted(cluster_counts) != expected_labels:
        raise ValueError(f"Mutation cluster labels are not contiguous for {tumor_id}.")
    for record, label in zip(center_records, center_labels, strict=True):
        size = _integer(
            record, "cluster_size", context=f"cluster centers for {tumor_id}"
        )
        if size != cluster_counts[label]:
            raise ValueError(
                f"Cluster-center size mismatch for {tumor_id}, cluster {label}."
            )

    multiplicity_records = _records(parsed["mutation_region_multiplicity.tsv"])
    expected_multiplicity_rows = mutation_count * region_count
    if len(multiplicity_records) != expected_multiplicity_rows:
        raise ValueError(
            f"Multiplicity output for {tumor_id} has {len(multiplicity_records)} rows; "
            f"expected {expected_multiplicity_rows}."
        )
    displayed_regions = [
        str(region).replace("sample", "region") for region in data.region_ids
    ]
    mutation_index = {
        mutation_id: index for index, mutation_id in enumerate(data.mutation_ids)
    }
    region_index = {
        region_id: index for index, region_id in enumerate(displayed_regions)
    }
    observed_pairs: set[tuple[str, str]] = set()
    for record in multiplicity_records:
        pair = (record["mutation_id"], record["region_id"])
        if pair in observed_pairs:
            raise ValueError(
                f"Duplicate mutation-region output pair for {tumor_id}: {pair!r}."
            )
        observed_pairs.add(pair)
        if pair[0] not in mutation_index or pair[1] not in region_index:
            raise ValueError(
                f"Unknown mutation-region output pair for {tumor_id}: {pair!r}."
            )
        mutation_i = mutation_index[pair[0]]
        region_i = region_index[pair[1]]
        context = f"multiplicity row {pair!r} for {tumor_id}"
        output_label = _integer(record, "cluster_label", context=context)
        if output_label != mutation_label_by_id[pair[0]]:
            raise ValueError(
                f"Multiplicity cluster label disagrees for {tumor_id}, {pair!r}."
            )
        major = _finite_float(record, "major_cn", context=context)
        minor = _finite_float(record, "minor_cn", context=context)
        _assert_close(
            major,
            float(data.major_cn[mutation_i, region_i]),
            label=f"{context} major_cn",
        )
        _assert_close(
            minor,
            float(data.minor_cn[mutation_i, region_i]),
            label=f"{context} minor_cn",
        )
        phi = _finite_float(record, "phi", context=context)
        summary_phi = _finite_float(record, "summary_phi", context=context)
        if phi < -1e-9 or summary_phi < -1e-9:
            raise ValueError(f"Negative CCF in {context}.")
        estimated = _parse_bool(record["multiplicity_estimated"])
        expected_estimated = bool(
            data.multiplicity_estimation_mask[mutation_i, region_i]
        )
        if estimated != expected_estimated:
            raise ValueError(f"Multiplicity-estimation mask mismatch in {context}.")
        gamma = _finite_float(record, "gamma_major", context=context)
        if gamma < -1e-12 or gamma > 1.0 + 1e-12:
            raise ValueError(f"gamma_major outside [0, 1] in {context}.")
        major_call = _parse_bool(record["major_call"])
        call = _finite_float(record, "multiplicity_call", context=context)
        if expected_estimated:
            if major_call != (gamma >= 0.5):
                raise ValueError(f"major_call disagrees with gamma_major in {context}.")
            expected_call = major if major_call else minor
        else:
            if not major_call or not math.isclose(
                gamma, 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Fixed multiplicity state is not deterministic-major in {context}."
                )
            expected_call = major
        _assert_close(call, expected_call, label=f"{context} multiplicity_call")

    expected_pairs = {
        (mutation_id, region_id)
        for mutation_id in data.mutation_ids
        for region_id in displayed_regions
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"Multiplicity output does not contain the complete input Cartesian product for {tumor_id}."
        )

    lambda_records = _records(parsed["lambda_search.tsv"])
    if len(lambda_records) != _integer(
        summary, "num_candidates_all", context=summary_context
    ):
        raise ValueError(
            f"Lambda-search row count does not match num_candidates_all for {tumor_id}."
        )
    selected_records = [
        record
        for record in lambda_records
        if _parse_bool(record["is_selected_best_row"])
    ]
    if len(selected_records) != 1:
        raise ValueError(
            f"Lambda search for {tumor_id} must identify exactly one selected row."
        )
    for record in lambda_records:
        context = f"lambda-search row for {tumor_id}"
        if record["selection_score_name"] != "partition_icl":
            raise ValueError(
                f"Lambda-search row for {tumor_id} is not scored by partition_icl."
            )
        if record["lambda_search_mode"] != "partition_guided_admm":
            raise ValueError(f"Lambda-search mode mismatch for {tumor_id}.")
        if _parse_bool(record["lambda_path_prespecified"]):
            raise ValueError(
                f"Case {tumor_id} contains a prespecified lambda-search row."
            )
        if record["input_data_hash"] != input_data_hash:
            raise ValueError(f"Lambda-search input-data hash mismatch for {tumor_id}.")
        if record["evaluation_mode"] != "not_evaluated":
            raise ValueError(
                f"Lambda-search row for {tumor_id} was truth evaluated before selection."
            )
        elapsed = _finite_float(
            record, "candidate_evaluation_elapsed_seconds", context=context
        )
        if abs(elapsed) > 1e-12:
            raise ValueError(
                f"Candidate truth-evaluation time is nonzero for {tumor_id}."
            )
        leaked = [
            column
            for column in LAMBDA_TRUTH_COLUMNS
            if not _is_blank_or_nan(record[column])
        ]
        if leaked:
            raise ValueError(
                f"Case {tumor_id} contains candidate truth metrics despite post-selection-only evaluation: {leaked}."
            )

    selected = selected_records[0]
    selected_context = f"selected lambda-search row for {tumor_id}"
    if selected["candidate_pool_source"] != "raw_fused_lambda_path":
        raise ValueError(
            f"Selected case {tumor_id} is not a raw pairwise-fusion candidate."
        )
    if not _parse_bool(selected["lambda_applicable"]):
        raise ValueError(f"Selected lambda is not applicable for {tumor_id}.")
    selected_lambda = _finite_float(selected, "lambda", context=selected_context)
    if selected_lambda <= 0.0:
        raise ValueError(f"Selected lambda must be strictly positive for {tumor_id}.")
    _assert_close(
        selected_lambda,
        _finite_float(summary, "selected_lambda", context=summary_context),
        label=f"{tumor_id} selected lambda",
    )
    if summary["inner_solver"] != selected["inner_solver"]:
        raise ValueError(
            f"Selected inner solver disagrees with the run summary for {tumor_id}."
        )
    expected_edges = mutation_count * (mutation_count - 1) // 2
    if _integer(selected, "num_edges", context=selected_context) != expected_edges:
        raise ValueError(
            f"Selected case {tumor_id} does not use the complete pairwise graph."
        )
    if not selected["graph_name"].startswith("complete_"):
        raise ValueError(f"Selected graph name is not complete for {tumor_id}.")
    if summary["graph_name"] != selected["graph_name"]:
        raise ValueError(
            f"Selected graph name disagrees with run summary for {tumor_id}."
        )
    for column in (
        "raw_kkt_eligible",
        "bic_selection_eligible",
        "eligible_for_selection",
        "is_selection_optimal",
    ):
        if not _parse_bool(selected[column]):
            raise ValueError(f"Selected row for {tumor_id} has {column}=False.")
    selected_kkt = _finite_float(
        selected, "fixed_objective_kkt_residual", context=selected_context
    )
    selected_tol = _finite_float(selected, "tol", context=selected_context)
    if selected_kkt > 5.0 * selected_tol + 1e-12:
        raise ValueError(
            f"Selected KKT residual for {tumor_id} exceeds 5*tol: {selected_kkt} > {5.0 * selected_tol}."
        )
    _validate_selected_exactness_provenance(
        summary=summary,
        selected=selected,
        selected_kkt=selected_kkt,
        config=config,
        tumor_id=tumor_id,
    )
    _assert_close(
        selected_kkt,
        _finite_float(summary, "selected_kkt_residual", context=summary_context),
        label=f"{tumor_id} selected KKT residual",
    )
    if _integer(selected, "n_clusters", context=selected_context) != estimated_clusters:
        raise ValueError(
            f"Selected cluster count disagrees with run summary for {tumor_id}."
        )
    eligible_scores = [
        _finite_float(
            record, "partition_icl", context=f"eligible lambda row for {tumor_id}"
        )
        for record in lambda_records
        if _parse_bool(record["eligible_for_selection"])
    ]
    selected_score = _finite_float(selected, "partition_icl", context=selected_context)
    if not eligible_scores or selected_score > min(eligible_scores) + 1e-9:
        raise ValueError(f"Selected row is not partition-ICL optimal for {tumor_id}.")

    _, _, evaluation_count = parsed["simulation_eval.tsv"]
    if evaluation_count != 1:
        raise ValueError(
            f"Simulation evaluation for {tumor_id} must contain exactly one row."
        )
    evaluation = _records(parsed["simulation_eval.tsv"])[0]
    evaluation_context = f"simulation evaluation for {tumor_id}"
    ari = _finite_float(evaluation, "ARI", context=evaluation_context)
    cp_rmse = _finite_float(evaluation, "cp_rmse", context=evaluation_context)
    raw_cp_rmse = _finite_float(evaluation, "raw_cp_rmse", context=evaluation_context)
    summary_cp_rmse = _finite_float(
        evaluation, "summary_cp_rmse", context=evaluation_context
    )
    _validate_metric_range(ari, label=f"{tumor_id} ARI", lower=-1.0, upper=1.0)
    for label, value in (
        ("cp_rmse", cp_rmse),
        ("raw_cp_rmse", raw_cp_rmse),
        ("summary_cp_rmse", summary_cp_rmse),
    ):
        if value < 0.0:
            raise ValueError(f"Negative {label} for {tumor_id}.")
    _assert_close(cp_rmse, summary_cp_rmse, label=f"{tumor_id} legacy/summary CCF RMSE")
    if (
        _integer(evaluation, "true_clusters", context=evaluation_context)
        != true_cluster_count
    ):
        raise ValueError(f"Truth cluster count mismatch in evaluation for {tumor_id}.")
    if (
        _integer(evaluation, "estimated_clusters", context=evaluation_context)
        != estimated_clusters
    ):
        raise ValueError(
            f"Estimated cluster count mismatch in evaluation for {tumor_id}."
        )
    if (
        _integer(evaluation, "n_eval_mutations", context=evaluation_context)
        != mutation_count
    ):
        raise ValueError(f"Evaluation mutation count mismatch for {tumor_id}.")
    if _integer(evaluation, "n_filtered_mutations", context=evaluation_context) != 0:
        raise ValueError(f"Unexpected filtered mutations in evaluation for {tumor_id}.")

    asymmetric = any(
        not math.isclose(float(data.major_cn[i, j]), float(data.minor_cn[i, j]))
        for i in range(mutation_count)
        for j in range(region_count)
    )
    estimable = bool(data.multiplicity_estimation_mask.any())
    evaluation_metrics: dict[str, float | None] = {}
    for column, expected_present in (
        ("multiplicity_f1", asymmetric),
        ("multiplicity_asymmetric_f1", asymmetric),
        ("multiplicity_estimable_f1", estimable),
    ):
        value = _optional_finite_float(evaluation, column, context=evaluation_context)
        evaluation_metrics[column] = value
        if expected_present != (value is not None):
            raise ValueError(
                f"Multiplicity metric presence is inconsistent for {tumor_id}: {column}."
            )
        if value is not None:
            _validate_metric_range(
                value, label=f"{tumor_id} {column}", lower=0.0, upper=1.0
            )
    if asymmetric:
        assert evaluation_metrics["multiplicity_f1"] is not None
        assert evaluation_metrics["multiplicity_asymmetric_f1"] is not None
        _assert_close(
            evaluation_metrics["multiplicity_f1"],
            evaluation_metrics["multiplicity_asymmetric_f1"],
            label=f"{tumor_id} primary/asymmetric multiplicity F1",
        )

    for column, evaluation_value in (
        ("ARI", ari),
        ("cp_rmse", cp_rmse),
        ("raw_cp_rmse", raw_cp_rmse),
        ("summary_cp_rmse", summary_cp_rmse),
    ):
        _assert_close(
            _finite_float(summary, column, context=summary_context),
            evaluation_value,
            label=f"{tumor_id} summary/evaluation {column}",
        )
    for column, evaluation_value in evaluation_metrics.items():
        summary_value = _optional_finite_float(summary, column, context=summary_context)
        if (summary_value is None) != (evaluation_value is None):
            raise ValueError(
                f"Summary/evaluation presence mismatch for {tumor_id} {column}."
            )
        if summary_value is not None and evaluation_value is not None:
            _assert_close(summary_value, evaluation_value, label=f"{tumor_id} {column}")
    return hashes


def _materialize_flat_bundle(bundle: Path, flat_dir: Path, tumor_id: str) -> None:
    flat_dir.mkdir(parents=True, exist_ok=True)
    for name in expected_artifact_names(tumor_id):
        source = bundle / name
        target = flat_dir / name
        if (
            target.is_file()
            and not target.is_symlink()
            and not os.path.samefile(source, target)
            and target.stat().st_size == source.stat().st_size
            and _sha256_file(target) == _sha256_file(source)
        ):
            continue
        temporary = flat_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    _fsync_directory(flat_dir)


def promote_case_bundle(
    staging_dir: Path,
    outdir: Path,
    tumor_id: str,
    *,
    expected_device: str,
    input_file: Path,
    expected_case: Mapping[str, object],
    config: Mapping[str, object],
    config_sha256: str,
) -> tuple[Path, dict[str, str]]:
    hashes = validate_case_artifacts(
        staging_dir,
        tumor_id,
        expected_device=expected_device,
        input_file=input_file,
        expected_case=expected_case,
        config=config,
        config_sha256=config_sha256,
    )
    for name in expected_artifact_names(tumor_id):
        with (staging_dir / name).open("rb") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(staging_dir)

    bundle_root = outdir / "bundles"
    flat_dir = outdir / "results"
    bundle_root.mkdir(parents=True, exist_ok=True)
    final_bundle = bundle_root / tumor_id
    if final_bundle.exists():
        raise FileExistsError(
            f"Refusing to replace an existing immutable result bundle: {final_bundle}"
        )
    os.replace(staging_dir, final_bundle)
    _fsync_directory(bundle_root)
    _materialize_flat_bundle(final_bundle, flat_dir, tumor_id)
    return final_bundle, hashes


def reconcile_bundles(
    state: BenchmarkState,
    manifest: Mapping[str, object],
    outdir: Path,
    *,
    device: str,
    config: Mapping[str, object],
    config_sha256: str,
) -> None:
    database = {row["tumor_id"]: row for row in state.case_rows()}
    for raw_case in manifest["cases"]:  # type: ignore[index]
        case = dict(raw_case)
        tumor_id = str(case["tumor_id"])
        bundle = outdir / "bundles" / tumor_id
        row = database[tumor_id]
        if bundle.exists():
            hashes = validate_case_artifacts(
                bundle,
                tumor_id,
                expected_device=device,
                input_file=Path(str(manifest["input_dir"])) / str(case["input_file"]),
                expected_case=case,
                config=config,
                config_sha256=config_sha256,
            )
            recorded = str(row["artifact_hashes_json"])
            if recorded and json.loads(recorded) != hashes:
                raise RuntimeError(
                    f"Immutable artifact hash mismatch for completed case {tumor_id}."
                )
            _materialize_flat_bundle(bundle, outdir / "results", tumor_id)
            if row["status"] != "succeeded":
                state.recover_success(
                    tumor_id,
                    backend=device,
                    bundle_path=bundle,
                    artifact_hashes=hashes,
                )
        elif row["status"] == "succeeded":
            raise RuntimeError(
                f"SQLite marks {tumor_id} succeeded but its immutable bundle is missing."
            )


def _terminate_process_group(
    process: subprocess.Popen[bytes], grace_seconds: float
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _set_parent_death_signal(expected_parent_pid: int) -> None:
    """Ask Linux to terminate an isolated worker if its scheduler disappears."""

    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if int(libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)) != 0:  # PR_SET_PDEATHSIG
        os._exit(125)
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def run_isolated_worker(
    command: Sequence[str],
    *,
    log_path: Path,
    timeout_seconds: float,
    termination_grace_seconds: float,
) -> tuple[int, bool, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    parent_pid = os.getpid()
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=partial(_set_parent_death_signal, parent_pid),
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
            return int(returncode), False, monotonic() - started
        except subprocess.TimeoutExpired:
            _terminate_process_group(process, termination_grace_seconds)
            return int(process.returncode), True, monotonic() - started
        except BaseException:
            _terminate_process_group(process, termination_grace_seconds)
            raise


def _worker_command(
    *,
    input_file: Path,
    staging_dir: Path,
    simulation_root: Path,
    config_path: Path,
    config_sha256: str,
    input_sha256: str,
    truth_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--_worker",
        "--input-file",
        str(input_file),
        "--staging-dir",
        str(staging_dir),
        "--simulation-root",
        str(simulation_root),
        "--config-file",
        str(config_path),
        "--config-sha256",
        config_sha256,
        "--input-sha256",
        input_sha256,
        "--truth-sha256",
        truth_sha256,
    ]


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=argparse.SUPPRESS)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--truth-sha256", required=True)
    return parser


def _add_run_summary_provenance(path: Path, values: Mapping[str, str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Run summary is missing a header: {path}")
        rows = list(reader)
        columns = list(reader.fieldnames)
    if len(rows) != 1:
        raise ValueError(
            f"Run summary must contain exactly one row before provenance is added: {path}"
        )
    for column, value in values.items():
        if column in rows[0] and rows[0][column] not in {"", value}:
            raise ValueError(
                f"Run summary already contains conflicting {column}: {path}"
            )
        rows[0][column] = value
        if column not in columns:
            columns.append(column)
    _atomic_write_tsv(path, columns, rows)


def worker_main(argv: Sequence[str]) -> int:
    args = _worker_parser().parse_args(list(argv))
    _assert_conda_environment()
    config_payload = args.config_file.read_bytes()
    if _sha256_bytes(config_payload) != args.config_sha256:
        raise RuntimeError(
            "Worker configuration hash does not match the scheduler command."
        )
    config = json.loads(config_payload)
    source_sha256 = _sha256_bytes(_canonical_json_bytes(build_source_manifest()))
    if source_sha256 != config["source_sha256"]:
        raise RuntimeError(
            "Source tree changed after scheduler preflight; refusing to fit this case."
        )
    environment_sha256 = _sha256_bytes(
        _canonical_json_bytes(stable_environment_fingerprint())
    )
    if environment_sha256 != config["environment_sha256"]:
        raise RuntimeError(
            "Conda environment changed after scheduler preflight; refusing to fit this case."
        )
    tumor_id = args.input_file.stem
    if _sha256_file(args.input_file) != args.input_sha256:
        raise RuntimeError(
            f"Input TSV changed after scheduler preflight: {args.input_file}"
        )
    truth_sha256, _, _ = _tree_digest(args.simulation_root / tumor_id)
    if truth_sha256 != args.truth_sha256:
        raise RuntimeError(
            f"Simulation truth changed after scheduler preflight: {tumor_id}"
        )
    device = str(config["device"])
    _assert_backend_available(device, allow_cpu=device == "cpu")
    if config["lambda_grid"] is not None:
        raise RuntimeError("Sim1K worker refuses a prespecified lambda grid.")
    if config["lambda_grid_mode"] != "partition_guided_admm":
        raise RuntimeError("Sim1K worker requires partition_guided_admm.")
    if config["selection_score"] != "partition_icl":
        raise RuntimeError("Sim1K worker requires partition_icl selection.")
    if bool(config["evaluate_all_candidates"]):
        raise RuntimeError(
            "Sim1K worker requires post-selection-only truth evaluation."
        )
    if str(config["dtype"]) != "float64":
        raise RuntimeError("Sim1K worker requires float64.")
    args.staging_dir.mkdir(parents=True, exist_ok=False)

    from CliPP2.runners.pipeline import process_one_file

    process_one_file(
        file_path=args.input_file,
        outdir=args.staging_dir,
        simulation_root=args.simulation_root,
        lambda_grid=None,
        lambda_grid_mode="partition_guided_admm",
        fit_options=_fit_options_from_config(config),
        bic_df_scale=float(config["bic_df_scale"]),
        bic_cluster_penalty=float(config["bic_cluster_penalty"]),
        selection_score="partition_icl",
        use_warm_starts=True,
        write_outputs=True,
        finalize_selected_fit=True,
        missing_cna_policy="error",
        evaluate_all_candidates=False,
    )
    _add_run_summary_provenance(
        args.staging_dir / f"{tumor_id}_run_summary.tsv",
        {
            "benchmark_config_sha256": str(args.config_sha256),
            "benchmark_source_sha256": str(config["source_sha256"]),
            "benchmark_environment_sha256": str(config["environment_sha256"]),
            "benchmark_cohort_sha256": str(config["cohort_sha256"]),
            "benchmark_input_sha256": str(args.input_sha256),
            "benchmark_truth_sha256": str(args.truth_sha256),
        },
    )
    if _sha256_file(args.input_file) != args.input_sha256:
        raise RuntimeError(f"Input TSV changed while fitting: {args.input_file}")
    truth_sha256, _, _ = _tree_digest(args.simulation_root / tumor_id)
    if truth_sha256 != args.truth_sha256:
        raise RuntimeError(f"Simulation truth changed while fitting: {tumor_id}")
    return 0


def _scheduler_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--simulation-root", type=Path, default=DEFAULT_SIMULATION_ROOT)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit an explicitly requested --device cpu run. CUDA never falls back automatically.",
    )
    parser.add_argument("--outer-max-iter", type=int, default=8)
    parser.add_argument("--inner-max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--summary-tol", type=float, default=1e-4)
    parser.add_argument("--bic-partition-tol", type=float, default=1e-4)
    parser.add_argument("--major-prior", type=float, default=0.5)
    parser.add_argument(
        "--inner-backend",
        choices=("auto", "dense", "quotient-workset"),
        default="dense",
    )
    parser.add_argument(
        "--workset-max-bytes", type=int, default=DEFAULT_WORKSET_MAX_BYTES
    )
    parser.add_argument(
        "--compressed-cache-max-bytes",
        type=int,
        default=DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    )
    parser.add_argument(
        "--dense-fallback-policy",
        choices=("auto", "device-only", "cpu-allowed", "error"),
        default="auto",
    )
    parser.add_argument(
        "--workset-add-batch", type=int, default=DEFAULT_WORKSET_ADD_BATCH
    )
    parser.add_argument(
        "--workset-max-expansions", type=int, default=DEFAULT_WORKSET_MAX_EXPANSIONS
    )
    parser.add_argument(
        "--certificate-max-iter", type=int, default=DEFAULT_CERTIFICATE_MAX_ITER
    )
    parser.add_argument(
        "--certificate-refinement-rounds",
        type=int,
        default=DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    )
    parser.add_argument(
        "--certificate-column-tol-scale",
        type=float,
        default=DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    )
    parser.add_argument(
        "--allow-heuristic-structure-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--materialize-full-dual",
        action="store_true",
        help=(
            "Debugging option: materialize the full E-by-S dual after the "
            "backend memory preflight."
        ),
    )
    parser.add_argument("--bic-df-scale", type=float, default=1.0)
    parser.add_argument("--bic-cluster-penalty", type=float, default=0.0)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=float,
        default=DEFAULT_TERMINATION_GRACE_SECONDS,
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _validate_scheduler_args(args: argparse.Namespace) -> None:
    if args.device == "cpu" and not args.allow_cpu:
        raise ValueError("--device cpu requires --allow-cpu.")
    if args.device == "cuda" and args.allow_cpu:
        raise ValueError("--allow-cpu is only valid together with --device cpu.")
    for name in (
        "outer_max_iter",
        "inner_max_iter",
        "workset_max_bytes",
        "compressed_cache_max_bytes",
        "workset_add_batch",
        "workset_max_expansions",
        "certificate_max_iter",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if int(args.certificate_refinement_rounds) < 0:
        raise ValueError("--certificate-refinement-rounds must be nonnegative.")
    for name in (
        "tol",
        "summary_tol",
        "bic_partition_tol",
        "certificate_column_tol_scale",
        "timeout_seconds",
        "termination_grace_seconds",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if (
        not math.isfinite(float(args.major_prior))
        or not 0.0 <= float(args.major_prior) <= 1.0
    ):
        raise ValueError("--major-prior must be finite and in [0, 1].")
    if not math.isfinite(float(args.bic_df_scale)) or float(args.bic_df_scale) <= 0.0:
        raise ValueError("--bic-df-scale must be finite and positive.")
    if (
        not math.isfinite(float(args.bic_cluster_penalty))
        or float(args.bic_cluster_penalty) < 0.0
    ):
        raise ValueError("--bic-cluster-penalty must be finite and nonnegative.")
    if args.max_cases is not None and not 1 <= int(args.max_cases) <= EXPECTED_CASES:
        raise ValueError(f"--max-cases must be in [1, {EXPECTED_CASES}].")


def scheduler_main(argv: Sequence[str]) -> int:
    args = _scheduler_parser().parse_args(list(argv))
    _validate_scheduler_args(args)
    _assert_conda_environment()
    _assert_backend_available(str(args.device), allow_cpu=bool(args.allow_cpu))
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with SchedulerLock(outdir / ".scheduler.lock"):
        print(
            f"[{_utc_now()}] hashing and validating exactly {EXPECTED_CASES} input/truth cases",
            flush=True,
        )
        manifest = build_cohort_manifest(args.input_dir, args.simulation_root)
        source_manifest = build_source_manifest()
        provenance = outdir / "provenance"
        cohort_sha256 = _write_document_and_hash(
            provenance / "cohort_manifest.json", manifest, label="cohort manifest"
        )
        source_sha256 = _write_document_and_hash(
            provenance / "source_manifest.json",
            source_manifest,
            label="source manifest",
        )
        live_environment = collect_environment()
        environment_sha256 = _write_document_and_hash(
            provenance / "environment_lock.json",
            stable_environment_fingerprint(live_environment),
            label="environment lock",
        )
        config = build_run_config(
            args,
            cohort_sha256=cohort_sha256,
            source_sha256=source_sha256,
            environment_sha256=environment_sha256,
        )
        config_path = provenance / "config.json"
        config_sha256 = _write_document_and_hash(
            config_path, config, label="run configuration"
        )
        environment_path = provenance / "environment.json"
        if environment_path.exists():
            stored_environment = json.loads(
                environment_path.read_text(encoding="utf-8")
            )
            if not isinstance(stored_environment, dict):
                raise RuntimeError(
                    f"Invalid immutable environment capture: {environment_path}"
                )
            expected_hash = _sha256_file(environment_path)
            hash_path = environment_path.with_suffix(
                environment_path.suffix + ".sha256"
            )
            _write_once_or_verify(
                hash_path,
                f"{expected_hash}  {environment_path.name}\n".encode("ascii"),
                label="environment hash",
            )
        else:
            _write_document_and_hash(
                environment_path, live_environment, label="environment capture"
            )

        state = BenchmarkState(outdir / "state.sqlite3")
        try:
            state.initialize(manifest)
            state.recover_interrupted()
            reconcile_bundles(
                state,
                manifest,
                outdir,
                device=str(args.device),
                config=config,
                config_sha256=config_sha256,
            )
            export_state(state, outdir)
            if args.preflight_only:
                print(
                    f"[{_utc_now()}] preflight complete cohort_sha256={cohort_sha256} "
                    f"source_sha256={source_sha256} config_sha256={config_sha256}",
                    flush=True,
                )
                return 0

            cases = pending_scheduled_cases(state, manifest, max_cases=args.max_cases)
            input_dir = Path(str(manifest["input_dir"]))
            simulation_root = Path(str(manifest["simulation_root"]))

            for index, case in enumerate(cases, start=1):
                tumor_id = str(case["tumor_id"])
                while True:
                    row = next(
                        item
                        for item in state.case_rows()
                        if item["tumor_id"] == tumor_id
                    )
                    if row["status"] in {"succeeded", "failed"}:
                        break
                    existing_bundle = outdir / "bundles" / tumor_id
                    if existing_bundle.exists():
                        hashes = validate_case_artifacts(
                            existing_bundle,
                            tumor_id,
                            expected_device=str(args.device),
                            input_file=input_dir / str(case["input_file"]),
                            expected_case=case,
                            config=config,
                            config_sha256=config_sha256,
                        )
                        _materialize_flat_bundle(
                            existing_bundle, outdir / "results", tumor_id
                        )
                        state.recover_success(
                            tumor_id,
                            backend=str(args.device),
                            bundle_path=existing_bundle,
                            artifact_hashes=hashes,
                        )
                        export_state(state, outdir)
                        break
                    attempt_number = int(row["attempts"]) + 1
                    stage = (
                        outdir
                        / ".staging"
                        / f"{tumor_id}.attempt-{attempt_number}.{uuid.uuid4().hex}"
                    )
                    log_path = (
                        outdir / "logs" / f"{tumor_id}.attempt-{attempt_number}.log"
                    )
                    attempt = state.start_attempt(
                        tumor_id,
                        backend=str(args.device),
                        log_path=log_path,
                        staging_path=stage,
                        timeout_seconds=float(args.timeout_seconds),
                        termination_grace_seconds=float(args.termination_grace_seconds),
                    )
                    export_state(state, outdir)
                    command = _worker_command(
                        input_file=input_dir / str(case["input_file"]),
                        staging_dir=stage,
                        simulation_root=simulation_root,
                        config_path=config_path,
                        config_sha256=config_sha256,
                        input_sha256=str(case["input_sha256"]),
                        truth_sha256=str(case["truth_sha256"]),
                    )
                    print(
                        f"[{_utc_now()}] start {index}/{len(cases)} {tumor_id} "
                        f"attempt={attempt}/{MAX_ATTEMPTS} backend={args.device}",
                        flush=True,
                    )
                    returncode: int | None = None
                    timed_out = False
                    elapsed = 0.0
                    try:
                        returncode, timed_out, elapsed = run_isolated_worker(
                            command,
                            log_path=log_path,
                            timeout_seconds=float(args.timeout_seconds),
                            termination_grace_seconds=float(
                                args.termination_grace_seconds
                            ),
                        )
                        if timed_out:
                            raise TimeoutError(
                                f"Case exceeded {float(args.timeout_seconds):g} seconds."
                            )
                        if returncode != 0:
                            raise subprocess.CalledProcessError(returncode, command)
                        bundle, hashes = promote_case_bundle(
                            stage,
                            outdir,
                            tumor_id,
                            expected_device=str(args.device),
                            input_file=input_dir / str(case["input_file"]),
                            expected_case=case,
                            config=config,
                            config_sha256=config_sha256,
                        )
                        state.finish_success(
                            tumor_id,
                            attempt,
                            backend=str(args.device),
                            elapsed_seconds=elapsed,
                            returncode=returncode,
                            bundle_path=bundle,
                            artifact_hashes=hashes,
                        )
                        print(
                            f"[{_utc_now()}] done {index}/{len(cases)} {tumor_id} "
                            f"attempt={attempt} elapsed_seconds={elapsed:.2f}",
                            flush=True,
                        )
                    except BaseException as exc:
                        elapsed = locals().get("elapsed", 0.0)
                        returncode_value = returncode
                        timed_out_value = isinstance(exc, TimeoutError) or timed_out
                        status = state.finish_failure(
                            tumor_id,
                            attempt,
                            elapsed_seconds=float(elapsed),
                            returncode=None
                            if returncode_value is None
                            else int(returncode_value),
                            timed_out=timed_out_value,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        shutil.rmtree(stage, ignore_errors=True)
                        print(
                            f"[{_utc_now()}] {status} {tumor_id} attempt={attempt} "
                            f"error={type(exc).__name__}: {exc} log={log_path}",
                            flush=True,
                        )
                        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                            raise
                    finally:
                        export_state(state, outdir)

            rows = state.case_rows()
            succeeded = sum(row["status"] == "succeeded" for row in rows)
            failed = sum(row["status"] == "failed" for row in rows)
            pending = len(rows) - succeeded - failed
            print(
                f"[{_utc_now()}] scheduler complete succeeded={succeeded} failed={failed} pending={pending} "
                f"flat_results={outdir / 'results'}",
                flush=True,
            )
            return 1 if failed else 0
        finally:
            export_state(state, outdir)
            state.close()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--_worker" in arguments:
        return worker_main(arguments)
    return scheduler_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
