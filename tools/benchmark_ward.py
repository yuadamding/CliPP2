"""Deterministic source-bound Ward microbenchmark; not full-fit qualification.

CPU smoke, from the source root in ml1::

    python tools/benchmark_ward.py --mutations 128 --regions 3 --repeats 5

CUDA must run inside an approved Seadragon LSF GPU allocation, with
CLIPP2_TEST_CUDA=1 and --expected-source-sha256 bound by its immutable source
receipt. This script neither submits jobs nor selects/fetches source revisions.
Use the same arguments in separately checked-out revisions to compare them.
JSON goes to stdout. Timing repeats are uninstrumented; a separate pass records
Python allocation/phase data and optional --profile records PyTorch operations.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import CliPP2  # noqa: E402
from CliPP2._source import source_fingerprint  # noqa: E402
from CliPP2.core.fusion import partition_starts as ward  # noqa: E402


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _git(*args):
    try:
        result = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
    except FileNotFoundError:
        return None  # Immutable source archives need not include Git itself.
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", type=int, default=128)
    parser.add_argument("--regions", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=174)
    parser.add_argument("--fixture", choices=("random", "refresh-stress"), default="random")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args(argv)
    if min(args.mutations, args.regions, args.repeats, args.threads) < 1 or args.warmups < 0:
        parser.error("sizes, repeats and threads must be positive; warmups must be nonnegative")
    if args.device == "cpu" and args.mutations > 2048:
        parser.error("CPU is limited to a <=2048-mutation Ward smoke test, not production timing")
    fingerprint = source_fingerprint(ROOT)
    if args.expected_source_sha256 and args.expected_source_sha256 != fingerprint:
        parser.error("imported source differs from the expected source fingerprint")
    if Path(CliPP2.__file__).resolve().parent != ROOT:
        parser.error("imported CliPP2 is outside this source tree")
    if args.device == "cuda" and not (
        os.environ.get("CLIPP2_TEST_CUDA") == "1"
        and os.environ.get("LSB_JOBID", "").isdigit()
        and args.expected_source_sha256 and torch.cuda.is_available()
    ):
        parser.error("CUDA requires LSF allocation, CLIPP2_TEST_CUDA=1, and bound source SHA-256")
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(args.seed)
    shape, dtype = (args.mutations, args.regions), getattr(torch, args.dtype)
    phi = torch.rand(shape, dtype=dtype, generator=generator)
    curvature = torch.rand(shape, dtype=dtype, generator=generator) * 15
    if args.fixture == "refresh-stress":
        # All row minima initially point to the final zero-curvature node;
        # retiring it forces an almost-full active-row refresh in one merge.
        curvature[-1] = 0
    input_hash = hashlib.sha256(phi.numpy().tobytes() + curvature.numpy().tobytes()).hexdigest()
    phi, curvature = phi.to(args.device), curvature.to(args.device)
    grid = sorted({1, min(args.mutations, 4), min(args.mutations, 16), args.mutations})

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def run():
        return ward.hessian_weighted_ward_label_sets_torch(phi, curvature, K_grid=grid)

    for _ in range(args.warmups):
        run()
    sync()
    baseline_cuda = None
    if args.device == "cuda":
        baseline_cuda = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    seconds, result_hashes = [], []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        result = run()
        sync()
        seconds.append(time.perf_counter() - start)
        result_hashes.append(_digest({k: result[k].tolist() for k in sorted(result)}))
    if len(set(result_hashes)) != 1:
        raise RuntimeError("Repeated Ward cuts differ")
    cuda_peak = None if baseline_cuda is None else torch.cuda.max_memory_allocated() - baseline_cuda

    # Instrument a separate pass, keeping these costs out of timing medians.
    physical = {"refresh_calls": 0, "refresh_gather_peak_elements": 0,
                "heap_peak_entries": 0, "heap_compactions": 0}
    refresh, compact, push = ward._ward_refresh_minima, ward._ward_compact_heap, ward.heapq.heappush

    def measured_refresh(matrix, rows, active, slots, best, columns, budget):
        physical["refresh_calls"] += 1
        gather = min(len(rows), max(1, budget // len(active))) * len(active)
        physical["refresh_gather_peak_elements"] = max(physical["refresh_gather_peak_elements"], gather)
        return refresh(matrix, rows, active, slots, best, columns, budget)

    def measured_compact(*values):
        physical["heap_compactions"] += 1
        return compact(*values)

    def measured_push(heap, item):
        push(heap, item)
        physical["heap_peak_entries"] = max(physical["heap_peak_entries"], len(heap))

    profile = cProfile.Profile()
    tracemalloc.start()
    with patch.object(ward, "_ward_refresh_minima", measured_refresh), \
            patch.object(ward, "_ward_compact_heap", measured_compact), \
            patch.object(ward.heapq, "heappush", measured_push):
        profile.runcall(run)
        sync()
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    phases = []
    for entry in profile.getstats():
        name = entry.code if isinstance(entry.code, str) else entry.code.co_name
        if "ward" in name or "_heapq" in name:
            phases.append({"name": name, "calls": entry.callcount,
                           "inclusive_seconds": entry.totaltime, "self_seconds": entry.inlinetime})
    operations = []
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, profile_memory=True) as trace:
            run()
            sync()
        for event in trace.key_averages():
            operations.append({"name": event.key, "calls": event.count,
                               "self_cpu_us": event.self_cpu_time_total,
                               "self_device_us": getattr(event, "self_device_time_total", 0),
                               "self_cpu_memory_bytes": event.self_cpu_memory_usage,
                               "self_device_memory_bytes": getattr(event, "self_device_memory_usage", 0)})
    element_bytes = phi.element_size()
    settings = {**vars(args), "K_grid": grid,
                "initializer_work_elements": ward._WARD_INITIAL_PAIRWISE_WORK_ELEMENTS,
                "refresh_work_elements": ward._WARD_REFRESH_WORK_ELEMENTS,
                "heap_active_multiplier": ward._WARD_HEAP_ACTIVE_MULTIPLIER,
                "heap_min_entries": ward._WARD_HEAP_MIN_ENTRIES}
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                   "platform": platform.platform(), "processor": platform.processor(),
                   "threads": torch.get_num_threads(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                   "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}
    row_chunk = max(1, min(args.mutations, ward._WARD_INITIAL_PAIRWISE_WORK_ELEMENTS // max(1, phi.numel())))
    record = {
        "schema": "clipp2_ward_microbenchmark_v1", "scope": "Ward only; not end-to-end fit or release qualification",
        "source": {"commit": _git("rev-parse", "HEAD"), "working_tree_status": _git("status", "--porcelain"),
                   "runtime_sha256": fingerprint, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "import_path": CliPP2.__file__},
        "environment": environment, "environment_sha256": _digest(environment),
        "settings": settings, "settings_sha256": _digest(settings), "input_sha256": input_hash,
        "result_sha256": result_hashes[0], "seconds": seconds, "median_seconds": statistics.median(seconds),
        "allocations": {**physical, "persistent_cost_matrix_bytes": args.mutations**2 * element_bytes,
                        "initializer_one_pair_region_tensor_bytes": row_chunk * phi.numel() * element_bytes,
                        "refresh_gather_peak_bytes": physical["refresh_gather_peak_elements"] * element_bytes,
                        "python_tracemalloc_peak_bytes": python_peak, "cuda_peak_allocated_delta_bytes": cuda_peak,
                        "notes": "Terms are not additive peaks: initialization has several live pair-region tensors; "
                        "tracemalloc excludes native tensor storage. Heap entries are counts, not byte estimates. "
                        "CUDA allocator delta excludes input storage and reserved-cache fragmentation. "
                        "Other state and output cuts also use memory; no whole-workflow bound is claimed."},
        "instrumented_cpu_phases": phases, "instrumented_torch_operations": operations,
        "phase_notes": "Inclusive times overlap and include synchronous transfer where present. Heap self times "
                       "are host work; optional operation records separate kernels, copies, and synchronization. "
                       "Instrumented runs do not contribute to reported timing repeats.",
    }
    if source_fingerprint(ROOT) != fingerprint:
        raise RuntimeError("Source changed during benchmark; do not publish this receipt")
    print(json.dumps(record, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
