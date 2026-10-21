"""Source-bound, in-memory interface to CliPP2's native regional forest kernel."""
import ctypes
import hashlib
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

NATIVE_ABI = 3
NATIVE_SOURCES = ('kernel/csrc/forest.cpp', 'kernel/csrc/bindings.cpp',
                  'kernel/csrc/forest.h')

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
