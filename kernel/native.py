"""Source-bound, in-memory interface to CliPP2's native regional forest kernel."""
import ctypes
import os
import threading
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from runpy import run_path

import numpy as np

NATIVE_ABI = 3
_build_helpers = run_path(str(Path(__file__).with_name('build_identity.py')))
NATIVE_SOURCES = _build_helpers['NATIVE_SOURCES']
_native_identity = _build_helpers['native_identity']

def native_library():
    root = Path(__file__).resolve().parent.parent
    path = next((root/('kernel/_native'+suffix) for suffix in EXTENSION_SUFFIXES
                 if (root/('kernel/_native'+suffix)).is_file()), None)
    if path is None:
        raise RuntimeError('Install CliPP2 (pip install .) to build its native kernel')
    signature = tuple((p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_size)
                      for p in (path, *(root/name for name in NATIVE_SOURCES)))
    return _load_library(root, path, signature)


@lru_cache(maxsize=4)
def _load_library(root, path, signature):
    digest = _native_identity(root)
    library = ctypes.CDLL(str(path))
    library.CliPP2KernelABI.argtypes = []
    library.CliPP2KernelABI.restype = ctypes.c_int
    library.CliPP2KernelBuildId.argtypes = []
    library.CliPP2KernelBuildId.restype = ctypes.c_char_p
    library.CliPP2KernelLastError.argtypes = []
    library.CliPP2KernelLastError.restype = ctypes.c_char_p
    if library.CliPP2KernelABI() != NATIVE_ABI or library.CliPP2KernelBuildId().decode() != digest:
        raise RuntimeError('CliPP2 kernel binary/source or ABI mismatch; rebuild before fitting')
    return library, digest


def forest_quadratic(curvature, rhs, edges, penalty, lower, upper):
    """Solve an N-by-R box QP with one shared forest and region-specific bounds.

    The caller independently audits KKT residuals against the original inputs.
    The native kernel validates the forest and does not reinterpret regions as
    independent cluster assignments.
    """
    h, b, lo, hi = [np.ascontiguousarray(a, dtype=np.float64)
                     for a in (curvature, rhs, lower, upper)]
    if h.ndim != 2 or not all(h.shape) or any(a.shape != h.shape for a in (b, lo, hi)):
        raise ValueError('Forest arrays must have the same nonempty N by R shape')
    n, regions = h.shape
    raw_edges = np.asarray(edges)
    if (raw_edges.ndim != 2 or raw_edges.shape[1] != 2
            or raw_edges.dtype.kind not in 'iu' or np.any(raw_edges < 0)
            or np.any(raw_edges >= n) or max(n, regions, len(raw_edges)) > np.iinfo(np.int32).max):
        raise ValueError('Forest requires integer in-range endpoint pairs')
    edges = np.ascontiguousarray(raw_edges, dtype=np.int32)
    library, _ = native_library()
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    entry = library.CliPP2ForestQP
    entry.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ints, doubles, doubles,
                     ctypes.c_double, doubles, doubles, doubles, ints]
    entry.restype = ctypes.c_int
    result = np.empty_like(h)
    diagnostics = np.empty(2, dtype=np.int32)
    if entry(n, regions, len(edges), edges, h, b, float(penalty), lo, hi, result, diagnostics):
        raise RuntimeError('CliPP2 forest kernel failed: '+library.CliPP2KernelLastError().decode())
    if not np.isfinite(result).all():
        raise RuntimeError('CliPP2 forest kernel returned nonfinite coordinates')
    return result, int(diagnostics[0]), int(diagnostics[1])


class _ContinuationPolicy(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ("levels", "iterations")] + [
        (name, ctypes.c_double) for name in ("stationarity", "constraint", "maximum_penalty")]


class _ContinuationReport(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in (
        "rho", "stationarity", "violation", "block_range", "likelihood", "objective")] + [
        (name, ctypes.c_int64) for name in (
            "iterations", "evaluations", "qp_calls", "qp_unqualified", "warm_starts", "accepted_steps")] + [
        (name, ctypes.c_int) for name in ("levels", "longest", "status")]


def continue_cpu(model, tree, pilot, requested, lower, upper, policy, missing):
    """One source-bound native batch; capacities retain independent state.

    Only explicit OMP_NUM_THREADS authorizes capacity concurrency. The native
    workspace gate can reduce it, and the default is one. NumPy's norm, cut-score
    einsum and scalar sine retain the reference's platform-specific arithmetic.
    """
    from .backend import NativeExecutionError, _Observations, _array

    raw_workers = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        workers = int(raw_workers)
    except ValueError as error:
        raise ValueError("CPU continuation OMP_NUM_THREADS must be a positive integer") from error
    if not 1 <= workers <= np.iinfo(np.int32).max:
        raise ValueError("CPU continuation OMP_NUM_THREADS must be a positive int32")
    if max(model.n, model.r, len(requested), policy.levels, policy.iterations_per_level) > np.iinfo(np.int32).max:
        raise ValueError("CPU continuation dimensions and budgets must fit int32")
    observations = _Observations()
    view = model.backend.lib.CliPP2BackendObservationView
    view.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Observations)]
    view.restype = ctypes.c_int
    model.backend._check(view(model.backend.context, ctypes.byref(observations)))
    initial = _array(np.clip(pilot, lower, upper))
    lower, upper = _array(lower), _array(upper)
    edges = np.ascontiguousarray(tree.edges, dtype=np.int32)
    canonical = np.ascontiguousarray(sorted(range(tree.n), key=lambda i: tree.mutation_ids[i]), dtype=np.int32)
    capacities = np.ascontiguousarray(requested, dtype=np.int32)
    # Scalar evaluation, exactly as in the host controller, not a SIMD sine.
    gaps = np.array([0.] + [4*np.sin(np.pi/(2*size))**2 for size in range(1, model.n+1)])
    native_policy = _ContinuationPolicy(policy.levels, policy.iterations_per_level,
        policy.stationarity_tolerance, policy.constraint_tolerance, policy.maximum_penalty)
    output = np.empty((len(requested), model.n, model.r))
    labels = np.empty((len(requested), model.n), dtype=np.int32)
    cuts = np.empty((len(requested), model.n-1), dtype=np.uint8)
    reports = (_ContinuationReport*len(requested))()
    norm_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_longlong)
    scores_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_longdouble),
        ctypes.c_longlong, ctypes.c_int, ctypes.POINTER(ctypes.c_longdouble))
    reduction_errors = []
    norm_lock = threading.Lock()

    @norm_type
    def norm(values, count):
        try:
            # Do not multiply a user's existing BLAS pool by capacity workers,
            # or change the BLAS thread setting/reduction implementation.
            with norm_lock:
                return float(np.linalg.norm(np.ctypeslib.as_array(values, shape=(count,)))) if count else 0.
        except BaseException as error:
            reduction_errors.append(error)
            return float("nan")

    @scores_type
    def cut_scores(values, count, regions, output):
        try:
            # ctypes' long-double PEP-3118 code is not understood by NumPy;
            # expose the same aligned storage as bytes, then view its dtype.
            def extended(pointer, size):
                raw = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8))
                return np.ctypeslib.as_array(raw,
                    shape=(size*np.dtype(np.longdouble).itemsize,)).view(np.longdouble)

            delta = extended(values, count*regions).reshape(count, regions)
            np.copyto(extended(output, count), np.einsum("ij,ij->i", delta, delta))
            return 0
        except BaseException as error:
            reduction_errors.append(error)
            return 1

    lib, identity = native_library()
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    bytes_ = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    entry = lib.CliPP2ContinueCPU
    entry.argtypes = [ctypes.POINTER(_Observations), ints, ints, doubles, doubles, doubles,
        ctypes.c_int, ints, ctypes.POINTER(_ContinuationPolicy), doubles, norm_type, scores_type,
        ctypes.c_int, ctypes.c_longlong, doubles, ints, bytes_,
        ctypes.POINTER(_ContinuationReport), ctypes.POINTER(ctypes.c_int)]
    entry.restype = ctypes.c_int
    lib.CliPP2ContinuationLastError.argtypes = []
    lib.CliPP2ContinuationLastError.restype = ctypes.c_char_p
    actual = ctypes.c_int()
    code = entry(ctypes.byref(observations), edges, canonical, initial, lower, upper,
        len(requested), capacities, ctypes.byref(native_policy), gaps, norm, cut_scores, workers,
        model.backend.workspace_bytes, output, labels, cuts, reports, ctypes.byref(actual))
    if reduction_errors:
        raise NativeExecutionError("CPU continuation compatibility reduction failed") from reduction_errors[0]
    if code:
        raise NativeExecutionError(lib.CliPP2ContinuationLastError().decode())
    if not np.isfinite(output).all():
        raise NativeExecutionError("CPU continuation returned nonfinite coordinates")
    statuses = ("penalty_stationary_constraint_tolerance", "penalty_numerical_limit_candidate",
                "line_search_stalled_candidate", "finite_budget_candidate")
    execution = {"continuation": "native_cpu", "build_id": identity,
        "workers": actual.value, "capacities": len(requested), "initial_evaluations": 1,
        "node_evaluations": 1+sum(report.evaluations for report in reports),
        "qp_calls": sum(report.qp_calls for report in reports),
        "workspace_limit": model.backend.workspace_bytes,
        "compatibility_reductions": ["numpy_longdouble_einsum", "numpy_blas_norm"],
        "scratch_reserved_bound_bytes": 32*model.n*model.r + actual.value*(
            416*model.n*model.r + 512*model.n + 64*(model.n-1))}
    execution["cell_evaluations"] = execution["node_evaluations"]*model.n*model.r
    result = []
    for index, (capacity, report) in enumerate(zip(requested, reports)):
        if not 0 <= report.status < len(statuses):
            raise NativeExecutionError("CPU continuation returned an unknown status")
        result.append({"labels": labels[index].astype(np.int64), "cuts": np.flatnonzero(cuts[index]),
            "requested_k": int(capacity), "raw_phi": output[index].copy(), "diagnostics": {
                "status": statuses[report.status], "backend": "host_forest", "tree_identity": tree.identity,
                "missing_raw_coordinates_initialized": missing,
                "constrained_optimum_certified": False, "global_optimum_certified": False,
                "rho": report.rho, "iterations": report.iterations, "continuation_levels": report.levels,
                "stationarity_CCF": report.stationarity, "constraint_residual_CCF": report.violation,
                "max_block_range_CCF": report.block_range, "longest_block": report.longest,
                "negative_log_likelihood": report.likelihood, "objective": report.objective,
                "tolerance_met": report.status == 0, "qp_unqualified": report.qp_unqualified,
                "feasible_warm_starts": report.warm_starts, "accepted_steps": report.accepted_steps,
                "penalty_gradient_roundoff_scale": 8*report.rho*np.finfo(float).eps},
            "execution": {**execution, "capacity_node_evaluations": report.evaluations,
                "capacity_qp_calls": report.qp_calls}})
    model._continuation_execution = execution
    return result
