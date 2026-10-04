"""Source-bound, in-memory interface to CliPP2's native forest/chain kernel."""
import ctypes
import hashlib
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

from . import MAX_CLUSTERS


NATIVE_SOURCES = ('kernel/csrc/chain.cpp', 'kernel/csrc/forest.cpp',
                  'kernel/csrc/bindings.cpp', 'kernel/csrc/chain.h')
DIAGNOSTIC_FIELDS = (
    'requested_K', 'actual_blocks', 'rho_internal', 'rho_CP', 'log_rho_CP',
    'objective', 'negative_log_likelihood', 'constraint_residual_CCF',
    'constraint_residual_CP', 'stationarity_CCF', 'iterations',
    'continuation_levels', 'qp_budget_hits', 'ccf_lower_bound', 'ccf_upper_bound',
    'max_block_range_CCF', 'longest_block', 'feasible_warm_starts',
    'penalty_gradient_roundoff_scale',
)
INTEGER_DIAGNOSTICS = frozenset(('requested_K', 'actual_blocks', 'iterations',
    'continuation_levels', 'qp_budget_hits', 'longest_block', 'feasible_warm_starts'))
STATUSES = ('penalty_stationary_constraint_tolerance', 'penalty_numerical_limit_candidate',
            'line_search_stalled_candidate', 'finite_budget_candidate')


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
    digest = hashlib.sha256(b''.join(name.encode()+b'\0'+(root/name).read_bytes()
                                   for name in NATIVE_SOURCES)).hexdigest()
    library = ctypes.CDLL(str(path))
    library.CliPP2KernelABI.argtypes = []
    library.CliPP2KernelABI.restype = ctypes.c_int
    library.CliPP2KernelBuildId.argtypes = []
    library.CliPP2KernelBuildId.restype = ctypes.c_char_p
    library.CliPP2KernelLastError.argtypes = []
    library.CliPP2KernelLastError.restype = ctypes.c_char_p
    if library.CliPP2KernelABI() != 1 or library.CliPP2KernelBuildId().decode() != digest:
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


def _integers(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iuf'
            or not np.isfinite(values).all() or np.any(values != np.rint(values))
            or np.any(values < np.iinfo(np.int32).min)
            or np.any(values > np.iinfo(np.int32).max)):
        raise ValueError(f'{name} must be a nonempty int32-representable vector')
    return np.ascontiguousarray(values, dtype=np.int32)


def solve_chain(arrays, purity, pilot_cp, capacities, *, evaluator=None):
    """Return proposals/diagnostics in memory; optionally evaluate likelihood on CUDA.

    Inputs and outputs follow frozen chain order. Callback failure aborts the
    solve; there is no CPU retry or partially accepted candidate publication.
    """
    if len(arrays) != 4:
        raise ValueError('Expected alt, depth, major and total CN vectors')
    arrays = tuple(_integers(a, name) for a, name in zip(arrays, ('alt', 'depth', 'major', 'total')))
    n = len(arrays[0])
    pilot = np.ascontiguousarray(pilot_cp, dtype=np.float64)
    capacities = _integers(capacities, 'capacities')
    if any(len(a) != n for a in arrays) or pilot.shape != (n,):
        raise ValueError('Chain vectors and pilot must have matching lengths')
    if (n > np.iinfo(np.int32).max or len(capacities) > min(n, MAX_CLUSTERS)
            or np.any(capacities < 1) or np.any(capacities > min(n, MAX_CLUSTERS))
            or len(np.unique(capacities)) != len(capacities)):
        raise ValueError('Expected unique capacities in 1..min(10,N)')
    library, digest = native_library()
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    ptr = ctypes.POINTER(ctypes.c_double)
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr)
    errors = []

    def callback(count, xp, lp, gp, hp):
        try:
            terms = evaluator(np.ctypeslib.as_array(xp, shape=(count,)).copy())
            if len(terms) != 3:
                raise ValueError('Likelihood callback must return loss, gradient and curvature')
            for index, (target, term) in enumerate(zip((lp, gp, hp), terms)):
                value = np.asarray(term, dtype=np.float64)
                if value.shape != (count,) or not np.isfinite(value).all():
                    raise ValueError('Likelihood callback returned invalid terms')
                if index == 2 and np.any(value <= 0):
                    raise ValueError('Likelihood callback requires positive curvature')
                np.ctypeslib.as_array(target, shape=(count,))[:] = value
            return 0
        except BaseException as error:
            errors.append(error)
            return 1

    callback_handle = callback_type(callback) if evaluator is not None else callback_type()
    k = len(capacities)
    labels, cp = np.empty((k, n), dtype=np.int32), np.empty((k, n), dtype=np.float64)
    diagnostics = np.empty((k, len(DIAGNOSTIC_FIELDS)), dtype=np.float64)
    statuses = np.empty(k, dtype=np.int32)
    entry = library.CliPP2SolveChain
    entry.argtypes = [ctypes.c_int, ints, ints, ints, ints, ctypes.c_double, doubles,
                     ints, ctypes.c_int, ints, doubles, doubles, ints, callback_type]
    entry.restype = ctypes.c_int
    status = entry(n, *arrays, float(purity), pilot, capacities, k,
                   labels, cp, diagnostics, statuses, callback_handle)
    if errors:
        raise RuntimeError('CliPP2 likelihood callback failed; no CPU fallback') from errors[0]
    if status:
        raise RuntimeError('CliPP2 native kernel failed: '+library.CliPP2KernelLastError().decode())
    if (not np.isfinite(cp).all() or not np.isfinite(diagnostics).all()
            or np.any(statuses < 0) or np.any(statuses >= len(STATUSES))):
        raise RuntimeError('CliPP2 native kernel returned malformed results')
    proposals = []
    for index, capacity in enumerate(capacities):
        raw = {name: int(value) if name in INTEGER_DIAGNOSTICS else float(value)
               for name, value in zip(DIAGNOSTIC_FIELDS, diagnostics[index])}
        raw.update(status=STATUSES[statuses[index]], constrained_optimum_certified=False,
                   backend='cuda_likelihood_host_chain' if evaluator is not None else 'cpu')
        proposals.append(dict(requested_k=int(capacity), labels=labels[index].copy(),
                              cp=cp[index].copy(), diagnostics=raw))
    return proposals, digest
