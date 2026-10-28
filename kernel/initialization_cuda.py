"""Opt-in resident initializer oracle; source/OpenBLAS parity is a separate gate.

Grid policies and optimizer decisions stay in initialization.py. This candidate
uses strict FP64 row dots and the native ordered column-reduction tree, not an
assertion that CUDA reproduces an arbitrary installed BLAS instruction stream.
"""
import ctypes as ct

import numpy as np

from .backend import _D, _I, _array, _ptr


POLICY = "strict_fp64_row_dots_ordered_tree_columns_v1"


def optional_oracle(view, grid, frequencies, block_rows, cache_bytes, *, standard, mask_bounds):
    backend = view.model.backend
    if backend.kind != "cuda" or not getattr(backend, "resident_initialization", False):
        return None
    result = CUDAInitializationOracle(view, grid, frequencies, block_rows, cache_bytes,
                                      standard=standard, mask_bounds=mask_bounds)
    if not result.handle:
        backend.initialization_admission = "direct_reference: resident working budget insufficient"
        return None
    return result


class CUDAInitializationOracle:
    def __init__(self, view, grid, frequencies, block_rows, cache_bytes, *, standard, mask_bounds):
        self.backend = view.model.backend
        self.handle = 0
        self.n, self.q = len(view), len(grid)
        self.standard = bool(standard)
        self.cache_budget = int(cache_bytes)
        frequencies, grid, rows = _array(frequencies), _array(grid), _array(view.rows, np.int64)
        if frequencies.shape != (self.n,):
            raise ValueError("Initializer frequencies must match retained rows")
        self.normalization = float(frequencies.sum())
        signatures = {
            "CreateInitialization": [ct.c_void_p, ct.c_int64, ct.c_int64, _I, ct.c_int64, _D, _D,
                                     ct.c_double, ct.c_int64, ct.c_int, ct.c_int, ct.c_int64, _I],
            "ReleaseInitialization": [ct.c_void_p, ct.c_int64],
            "EvaluateInitialization": [ct.c_void_p, ct.c_int64, ct.c_int64, _D, _I, ct.c_int,
                                       ct.c_int, _D, _D, ct.POINTER(ct.c_int)],
            "PrepareInitializationTransfer": [ct.c_void_p, ct.c_int64, _D, ct.c_int64, ct.c_int64],
            "InitializationTransferDerivative": [ct.c_void_p, ct.c_int64, ct.c_double, _D],
            "InitializationDiagnostics": [ct.c_void_p, ct.c_int64, _I],
        }
        for name, args in signatures.items():
            function = getattr(self.backend.lib, "CliPP2"+name)
            function.argtypes, function.restype = args, ct.c_int
        handle = ct.c_int64()
        self.backend._check(self.backend.lib.CliPP2CreateInitialization(self.backend.context,
            view.region, self.n, _ptr(rows), self.q, _ptr(grid), _ptr(frequencies), self.normalization,
            block_rows, int(mask_bounds), int(standard), self.cache_budget, ct.byref(handle)))
        self.handle = handle.value

    def _require_open(self):
        self.backend._require_open()
        if not self.handle:
            raise RuntimeError("CUDA initialization oracle is closed or was not admitted")

    def expectation(self, weights, *, collect_pilot=False, positions=None, rescale=False):
        self._require_open()
        weights = _array(weights)
        positions = None if positions is None else _array(positions, np.int64)
        if weights.ndim != 1 or (positions is not None and positions.shape != weights.shape):
            raise ValueError("Initializer active positions must match a weight vector")
        result = np.empty(len(weights)+1)
        pilot = np.empty(self.n) if collect_pilot else None
        invalid = ct.c_int()
        self.backend._check(self.backend.lib.CliPP2EvaluateInitialization(self.backend.context,
            self.handle, len(weights), _ptr(weights), None if positions is None else _ptr(positions),
            int(rescale), int(collect_pilot), _ptr(result), None if pilot is None else _ptr(pilot),
            ct.byref(invalid)))
        if invalid.value or not np.isfinite(result).all():
            return None, -np.inf, None
        return result[:-1]/self.normalization, float(result[-1]), pilot

    def prepare_transfer(self, weights, grow, shrink):
        self._require_open()
        weights = _array(weights)
        if weights.shape != (self.q,):
            raise ValueError("Initializer transfer requires all grid weights")
        self.backend._check(self.backend.lib.CliPP2PrepareInitializationTransfer(
            self.backend.context, self.handle, _ptr(weights), grow, shrink))

    def derivative(self, fraction):
        self._require_open()
        result = np.empty(1)
        self.backend._check(self.backend.lib.CliPP2InitializationTransferDerivative(
            self.backend.context, self.handle, fraction, _ptr(result)))
        return float(result[0]) if self.standard else float(result[0]/self.normalization)

    def diagnostics(self):
        self._require_open()
        values = np.empty(8, dtype=np.int64)
        self.backend._check(self.backend.lib.CliPP2InitializationDiagnostics(
            self.backend.context, self.handle, _ptr(values)))
        return {"backend": "cuda_initializer_experimental", "numerical_policy": POLICY,
                "source_blas_parity_qualified": False,
                "device_grid_hits": 0, "device_grid_misses": int(values[0]),
                "device_grid_prefix_bytes": self.n*self.q*8 if values[5] else 0,
                "device_grid_budget_bytes": self.cache_budget,
                "grid_block_evaluations": int(values[0]), "grid_cell_evaluations": int(values[1]),
                "expectations": int(values[2]), "transfer_preparations": int(values[3]),
                "transfer_trials": int(values[4]), "native_working_device_bytes": int(values[6]),
                "native_host_bytes": int(values[7])}

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, 0
            if self.backend.context:
                self.backend._check(self.backend.lib.CliPP2ReleaseInitialization(self.backend.context, handle))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Context teardown remains the non-throwing final owner.
