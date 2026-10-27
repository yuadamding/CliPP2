"""Source-bound emissions/refits and explicitly owned CUDA column/score leases."""
import ctypes as ct
import threading
import weakref
from functools import lru_cache
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

from .build_identity import NATIVE_SOURCES, backend_identity, native_identity as _native_identity
from .execution import resolve_cpu_threads, resolve_memory_budget


_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int64)
_B = ct.POINTER(ct.c_uint8)


class NativeExecutionError(RuntimeError):
    """Backend, resource or arithmetic failure; never a candidate rejection."""


class ScientificCandidateRejection(ValueError):
    """A proposed cluster lacks regional support or a common admissible box."""


class DeviceColumn:
    """One immutable context-bound lease; optional cache eviction is harmless."""

    def __init__(self, backend, handle):
        self._owner = weakref.ref(backend)
        self.handle = int(handle)
        self.n = backend.n

    def backend(self):
        owner = self._owner()
        if owner is None or not self.handle:
            raise NativeExecutionError("Device column lease is closed")
        owner._require_open()
        return owner

    def __len__(self):
        return self.n

    def close(self):
        handle, self.handle = self.handle, 0
        owner = self._owner()
        if handle and owner is not None and owner.context:
            handles = _array([handle], np.int64)
            owner._check(owner.lib.CliPP2ReleaseJointColumns(owner.context, 1, _ptr(handles)))

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DeviceMixtureProblem:
    """Resident numerical oracle for the unchanged host simplex controller."""

    def __init__(self, backend, columns):
        self._owner = weakref.ref(backend)
        self.handle = 0
        handles = backend._column_handles(columns)
        result = ct.c_int64()
        backend._check(backend.lib.CliPP2CreateMixtureProblem(
            backend.context, len(handles), _ptr(handles), ct.byref(result)))
        self.handle = result.value
        self.shape = backend.n, len(handles)

    def backend(self):
        owner = self._owner()
        if owner is None or not self.handle:
            raise NativeExecutionError("Device mixture problem is closed")
        owner._require_open()
        return owner

    def evaluate_weights(self, weights):
        owner = self.backend()
        weights = self._weights(weights)
        objective, likelihood = ct.c_double(), ct.c_double()
        gradient = np.empty(self.shape[1])
        owner._check(owner.lib.CliPP2EvaluateMixture(
            owner.context, self.handle, _ptr(weights), ct.byref(objective),
            _ptr(gradient), ct.byref(likelihood)))
        return objective.value, gradient, likelihood.value

    def _weights(self, weights):
        weights = _array(weights)
        if (weights.shape != (self.shape[1],) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not weights.max() > 0):
            raise ValueError("Mixture weights must be nonnegative finite and nonzero")
        return weights

    def prepare_transfer(self, weights, grow, shrink):
        owner = self.backend()
        weights = self._weights(weights)
        if not (0 <= grow < self.shape[1] and 0 <= shrink < self.shape[1]
                and grow != shrink and weights[shrink] > 0):
            raise ValueError("Invalid pairwise mixture transfer")
        owner._check(owner.lib.CliPP2PrepareMixtureTransfer(
            owner.context, self.handle, _ptr(weights), grow, shrink))

    def transfer_derivative(self, fraction):
        owner = self.backend()
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Mixture transfer fraction must be in [0,1]")
        result = ct.c_double()
        owner._check(owner.lib.CliPP2MixtureTransferDerivative(
            owner.context, self.handle, fraction, ct.byref(result)))
        return result.value

    def close(self):
        handle, self.handle = self.handle, 0
        owner = self._owner()
        if handle and owner is not None and owner.context:
            owner._check(owner.lib.CliPP2ReleaseMixtureProblem(owner.context, handle))

    def __enter__(self):
        self.backend()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _Observations(ct.Structure):
    _fields_ = [("n", ct.c_int64), ("r", ct.c_int64)] + [
        (name, _D) for name in ("alt", "ref", "scaling", "constants", "lower", "upper")
    ] + [("support", _I), ("flags", _B)]


class _Diagnostics(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in (
        "prepared_bytes", "workspace_bytes", "calls", "cell_evaluations", "cache_bytes",
        "staging_bytes", "workspace_limit", "host_to_device_bytes", "device_to_host_bytes",
        "kernel_launches", "synchronizations")]


class _RefitBatch(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("tasks", "members", "grid_points")] + [
        (name, _I) for name in ("regions", "member_offsets", "member_rows", "grid_offsets")
    ] + [(name, _D) for name in ("grid", "lower", "upper", "xatol", "incumbent")]


class _RefitResult(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in (
        "base_center", "base_log_likelihood", "center", "log_likelihood")
    ] + [("evaluations", ct.c_int64), ("status", ct.c_int32)]


class _ConditionalBatch(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("tasks", "members")] + [
        (name, _I) for name in ("regions", "member_offsets", "member_rows")
    ] + [("centers", _D)]


def _ptr(array):
    return array.ctypes.data_as({np.dtype("float64"): _D, np.dtype("int64"): _I,
                                np.dtype("uint8"): _B}[array.dtype])


def _array(value, dtype=np.float64):
    return np.ascontiguousarray(value, dtype=dtype)


class NativeBackend:
    """Owns prepared SoA fields; never silently substitutes a CPU backend."""

    def __init__(self, model, workspace_bytes, *, native_host_memory_bytes=None,
                 device_memory_bytes=None, cpu_threads=None, device_cache_bytes=256*1024**2):
        self.context = None
        kind = "cuda" if model.device.startswith("cuda") else "cpu"
        paths = sorted(Path(__file__).parent.glob(f"_native_{kind}*.so"))
        if len(paths) != 1:
            raise NativeExecutionError(f"Build CliPP2's native {kind} library before fitting")
        lib = self.lib = ct.CDLL(str(paths[0]))
        signatures = {
            "ABI": ([], ct.c_int), "BuildId": ([], ct.c_char_p),
            "NumericalPolicy": ([], ct.c_char_p), "LastError": ([], ct.c_char_p),
            "Create": ([ct.POINTER(_Observations), ct.c_int64, ct.c_int, ct.POINTER(ct.c_void_p)], ct.c_int),
            "Destroy": ([ct.c_void_p], None),
            "GetDiagnostics": ([ct.c_void_p, ct.POINTER(_Diagnostics)], ct.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(lib, "CliPP2Backend" + name)
            function.argtypes, function.restype = args, result
        for name, args in {
            "Nodes": [ct.c_void_p, ct.c_int64, _D, ct.c_int, _D, _D, _D, _B],
            "Cells": [ct.c_void_p, ct.c_int64, _I, _D, ct.c_int, _D, _D, _D, _B],
            "RegionalGrid": [ct.c_void_p, ct.c_int64, ct.c_int64, _I, ct.c_int64, _D, _D, _B],
            "JointColumns": [ct.c_void_p, ct.c_int64, _D, _D, _B],
        }.items():
            function = getattr(lib, "CliPP2Evaluate" + name)
            function.argtypes, function.restype = args, ct.c_int
        lib.CliPP2RefitCentersBatch.argtypes = [ct.c_void_p, ct.POINTER(_RefitBatch), ct.POINTER(_RefitResult)]
        lib.CliPP2RefitCentersBatch.restype = ct.c_int
        lib.CliPP2EvaluateConditionalValues.argtypes = [
            ct.c_void_p, ct.POINTER(_ConditionalBatch), _D, _B]
        lib.CliPP2EvaluateConditionalValues.restype = ct.c_int
        lib.CliPP2MultiplicityMAP.argtypes = [ct.c_void_p, _D, _D, _B]
        lib.CliPP2MultiplicityMAP.restype = ct.c_int
        if kind == "cpu" and hasattr(lib, "CliPP2ProbabilityReuseDiagnostics"):
            lib.CliPP2ProbabilityReuseDiagnostics.argtypes = [ct.c_void_p, _I]
            lib.CliPP2ProbabilityReuseDiagnostics.restype = ct.c_int
        self.identity = lib.CliPP2BackendBuildId().decode()
        self.policy = lib.CliPP2BackendNumericalPolicy().decode()
        if lib.CliPP2BackendABI() != 2 or self.identity != backend_identity(kind):
            raise NativeExecutionError(f"Stale or incompatible native {kind} library; rebuild CliPP2")
        if type(workspace_bytes) is not int or not 0 <= workspace_bytes <= np.iinfo(np.int64).max:
            raise ValueError("workspace_bytes must be a nonnegative int64")
        self.kind, self.n, self.r = kind, model.n, model.r
        self.workspace_bytes = workspace_bytes
        self.cpu_threads = resolve_cpu_threads(cpu_threads)
        self.native_host_memory_bytes = resolve_memory_budget(native_host_memory_bytes)
        self.device_memory_bytes = device_memory_bytes
        required = model.n * model.r * 57
        if kind == "cpu" and required > self.native_host_memory_bytes:
            raise NativeExecutionError(f"Prepared observations require {required} native host bytes; "
                                       f"budget is {self.native_host_memory_bytes}")
        if kind == "cuda":
            lib.CliPP2DeviceMemoryInfo.argtypes = [ct.c_int, _I, _I]
            lib.CliPP2DeviceMemoryInfo.restype = ct.c_int
            free, total = ct.c_int64(), ct.c_int64()
            device_index = int(model.device.split(":", 1)[1]) if ":" in model.device else 0
            self._check(lib.CliPP2DeviceMemoryInfo(device_index, ct.byref(free), ct.byref(total)))
            self.device_memory_bytes = resolve_memory_budget(device_memory_bytes, free=free.value, device=True)
            if required > self.device_memory_bytes:
                raise NativeExecutionError(f"Prepared observations require {required} device bytes; "
                                           f"budget is {self.device_memory_bytes}")
            workspace_bytes = min(workspace_bytes, self.device_memory_bytes - required)
            self.workspace_bytes = workspace_bytes
        fields = [_array(value.T) for value in (
            model.alt, model.depth-model.alt, model.scaling, model.constants, model.lower, model.upper)]
        fields += [_array(model.support.T, np.int64),
                   _array((model.observed.astype(np.uint8) + 2*model.mixed).T, np.uint8)]
        observations = _Observations(model.n, model.r, *map(_ptr, fields))
        self.context = ct.c_void_p()
        device_index = (int(model.device.split(":", 1)[1]) if ":" in model.device else 0) if kind == "cuda" else -1
        self._check(lib.CliPP2BackendCreate(ct.byref(observations), workspace_bytes,
                                          device_index, ct.byref(self.context)))
        if kind == "cpu":
            lib.CliPP2ConfigureCPUExecution.argtypes = [ct.c_void_p, ct.c_int64, ct.c_int]
            self._check(lib.CliPP2ConfigureCPUExecution(
                self.context, self.native_host_memory_bytes, self.cpu_threads))
            configured = _Diagnostics()
            self._check(lib.CliPP2BackendGetDiagnostics(self.context, ct.byref(configured)))
            self.workspace_bytes = configured.workspace_limit
            lib.CliPP2ProbabilityCacheDiagnostics.argtypes = [ct.c_void_p, _I]
            lib.CliPP2ProbabilityCacheDiagnostics.restype = ct.c_int
            lib.CliPP2ReleaseProbabilityCache.argtypes = [ct.c_void_p]
            lib.CliPP2ReleaseProbabilityCache.restype = ct.c_int
        else:
            # Host metadata and device scratch are independently bounded pools.
            # A small tiled device workspace must not become an implicit host
            # memory cap for resident immutable column handles.
            self.host_staging_bytes_limit = self.native_host_memory_bytes
            lib.CliPP2ConfigureCUDAExecution.argtypes = [ct.c_void_p, ct.c_int64]
            lib.CliPP2ConfigureCUDAExecution.restype = ct.c_int
            self._check(lib.CliPP2ConfigureCUDAExecution(self.context, self.host_staging_bytes_limit))
            if type(device_cache_bytes) is not int or not 0 <= device_cache_bytes <= np.iinfo(np.int64).max:
                raise ValueError("device_cache_bytes must be a nonnegative int64")
            signatures = {
                "ConfigureCUDAReuse": [ct.c_void_p, ct.c_int64, ct.c_int64],
                "CreateJointColumns": [ct.c_void_p, ct.c_int64, _D, _I],
                "ImportJointColumns": [ct.c_void_p, ct.c_int64, _D, _I],
                "ReleaseJointColumns": [ct.c_void_p, ct.c_int64, _I],
                "ExportJointColumns": [ct.c_void_p, ct.c_int64, _I, _D],
                "GatherJointColumns": [ct.c_void_p, ct.c_int64, _I, _I, _D],
                "CreateMixtureProblem": [ct.c_void_p, ct.c_int64, _I, _I],
                "ReleaseMixtureProblem": [ct.c_void_p, ct.c_int64],
                "EvaluateMixture": [ct.c_void_p, ct.c_int64, _D, _D, _D, _D],
                "PrepareMixtureTransfer": [ct.c_void_p, ct.c_int64, _D, ct.c_int64, ct.c_int64],
                "MixtureTransferDerivative": [ct.c_void_p, ct.c_int64, ct.c_double, _D],
                "CUDAScoringDiagnostics": [ct.c_void_p, _I],
                "CUDAReuseDiagnostics": [ct.c_void_p, _I],
                "ReleaseCUDAOptionalCache": [ct.c_void_p],
            }
            for name, args in signatures.items():
                function = getattr(lib, "CliPP2" + name)
                function.argtypes, function.restype = args, ct.c_int
            resident_limit = max(0, self.device_memory_bytes-required-self.workspace_bytes)
            self._check(lib.CliPP2ConfigureCUDAReuse(
                self.context, resident_limit, min(device_cache_bytes, resident_limit)))

    @property
    def native_host_bytes(self):
        """Retained native host storage; caller arrays and transient scratch are separate."""
        self._require_open()
        if self.kind == "cuda":
            reuse = np.empty(11, dtype=np.int64)
            self._check(self.lib.CliPP2CUDAReuseDiagnostics(self.context, _ptr(reuse)))
            return int(reuse[9]) + int(reuse[10])
        result = _Diagnostics()
        self._check(self.lib.CliPP2BackendGetDiagnostics(self.context, ct.byref(result)))
        return result.prepared_bytes + result.cache_bytes

    def release_optional_cache(self):
        """Give mandatory continuation state priority over rebuildable read plans."""
        self._require_open()
        if self.kind == "cpu":
            self._check(self.lib.CliPP2ReleaseProbabilityCache(self.context))
        else:
            self._check(self.lib.CliPP2ReleaseCUDAOptionalCache(self.context))
        return self.native_host_bytes

    def _require_open(self):
        if not getattr(self, "context", None):
            raise NativeExecutionError("Native backend context is closed")

    def close(self):
        """Release the native context exactly once, including partial initialization."""
        context = getattr(self, "context", None)
        if context:
            self.context = None
            self.lib.CliPP2BackendDestroy(context)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        try:
            if self.context and self.identity != backend_identity(self.kind):
                raise NativeExecutionError("Native backend source changed during execution")
        finally:
            self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Destructors cannot report errors; explicit close remains available.

    def _check(self, code, status=None):
        if code:
            if code == 2:
                raise ValueError(self.lib.CliPP2BackendLastError().decode())
            raise NativeExecutionError(self.lib.CliPP2BackendLastError().decode())
        if status is not None:
            if np.any(status > 2):
                raise NativeExecutionError("Native likelihood produced invalid arithmetic or an unknown status")
            if np.any(status == 2):
                raise ValueError("Native likelihood received invalid CCFs")

    def diagnostics(self):
        self._require_open()
        result = _Diagnostics()
        self._check(self.lib.CliPP2BackendGetDiagnostics(self.context, ct.byref(result)))
        diagnostics = {"kind": self.kind, "build_id": self.identity, "numerical_policy": self.policy,
                       **{name: getattr(result, name) for name, _ in result._fields_}}
        if self.kind == "cuda":
            diagnostics["scoring"] = self.scoring_diagnostics()
            reuse = np.empty(11, dtype=np.int64)
            self._check(self.lib.CliPP2CUDAReuseDiagnostics(self.context, _ptr(reuse)))
            diagnostics["reuse"] = dict(zip(("resident_limit", "resident_bytes", "resident_peak_bytes",
                "optional_cache_limit", "optional_cache_bytes", "grid_hits", "grid_misses",
                "grid_reused_cells", "refit_physical_emission_cells", "grid_cache_host_bytes",
                "scoring_host_bytes"), map(int, reuse)))
        if self.kind == "cpu" and hasattr(self.lib, "CliPP2ProbabilityReuseDiagnostics"):
            counters = np.empty(5, dtype=np.int64)
            self._check(self.lib.CliPP2ProbabilityReuseDiagnostics(self.context, _ptr(counters)))
            diagnostics["probability_reuse"] = dict(zip(("plans", "basis_components", "shape_evaluations",
                "ordered_row_references", "peak_optional_plan_reservation"), map(int, counters)))
            cache = np.empty(4, dtype=np.int64)
            self._check(self.lib.CliPP2ProbabilityCacheDiagnostics(self.context, _ptr(cache)))
            diagnostics["probability_cache"] = dict(zip(("hits", "misses", "bytes", "model_identity_bytes"), map(int, cache)))
        diagnostics.update(native_host_memory_bytes=self.native_host_memory_bytes,
                           device_memory_bytes=self.device_memory_bytes,
                           cpu_threads=self.cpu_threads, native_host_bytes=self.native_host_bytes)
        if self.kind == "cuda":
            diagnostics["host_staging_bytes_limit"] = self.host_staging_bytes_limit
        return diagnostics

    def _tile_count(self, count, per_cell, fixed=16):
        """Bound native temporary storage; requested host results are separate."""
        if self.kind == "cpu" or not count:
            return count
        size = min(count, (self.workspace_bytes - fixed) // per_cell)
        if size < 1:
            raise NativeExecutionError("CUDA workspace cannot hold one observation and its result")
        return size

    def _rows(self, region, rows):
        raw = np.asarray(rows)
        if (isinstance(region, (bool, np.bool_)) or not isinstance(region, (int, np.integer))
                or not 0 <= region < self.r or raw.ndim != 1 or raw.dtype.kind not in "iu"
                or np.any(raw < 0) or np.any(raw >= self.n)):
            raise ValueError("A regional operation needs integer in-range region and row indices")
        return _array(raw, np.int64)

    def node_terms(self, x, derivatives=True):
        self._require_open()
        x = _array(np.asarray(x).T)
        if x.shape != (self.r, self.n):
            raise ValueError("CCFs must match N by R")
        result = [np.empty_like(x) for _ in range(3 if derivatives else 1)]
        count = x.size
        direct_bytes = count * (33 if derivatives else 17) + 8
        if self.kind == "cpu" or direct_bytes <= self.workspace_bytes:
            status = np.empty(x.shape, dtype=np.uint8)
            pointers = [_ptr(value) for value in result] + [None] * (3-len(result))
            self._check(self.lib.CliPP2EvaluateNodes(self.context, 1, _ptr(x), int(derivatives),
                                                    *pointers, _ptr(status)), status)
        else:
            # Indexed tiles keep the same region-major observation mapping;
            # their extra int64 index vector is included in the budget.
            step = self._tile_count(count, 41 if derivatives else 25)
            for first in range(0, count, step):
                last = min(count, first+step)
                indices = np.arange(first, last, dtype=np.int64)
                status = np.empty(last-first, dtype=np.uint8)
                pointers = [_ptr(value.ravel()[first:last]) for value in result] + [None] * (3-len(result))
                self._check(self.lib.CliPP2EvaluateCells(self.context, len(indices), _ptr(indices),
                    _ptr(x.ravel()[first:last]), int(derivatives), *pointers, _ptr(status)), status)
        # Host continuation reduces these N-by-R arrays in the established
        # row-major order. A transposed view would silently change that order.
        return tuple(np.ascontiguousarray(value.T) for value in result) if derivatives else np.ascontiguousarray(result[0].T)

    def cells(self, region, rows, values):
        self._require_open()
        rows = self._rows(region, rows)
        cells = _array(region*self.n + rows, np.int64)
        x = _array(np.broadcast_to(values, cells.shape))
        result = np.empty_like(x)
        step = self._tile_count(len(x), 25)
        for first in range(0, len(x), step or 1):
            last = min(len(x), first+step)
            status = np.empty(last-first, dtype=np.uint8)
            self._check(self.lib.CliPP2EvaluateCells(self.context, last-first, _ptr(cells[first:last]),
                _ptr(x[first:last]), 0, _ptr(result[first:last]), None, None, _ptr(status)), status)
        return -result

    def regional_grid(self, region, rows, grid):
        self._require_open()
        rows, grid = self._rows(region, rows), _array(grid)
        if grid.ndim != 1:
            raise ValueError("Regional grid must be one-dimensional")
        result = np.empty((len(grid), len(rows)))
        if not result.size:
            return result.T
        if self.kind == "cpu":
            row_step, grid_step = len(rows), len(grid)
        else:
            row_step = self._tile_count(len(rows), 17, fixed=40)
            grid_step = min(len(grid), (self.workspace_bytes - 8*row_step - 32) // (9*row_step+8))
            if grid_step < 1:
                raise NativeExecutionError("CUDA workspace cannot hold one regional grid point")
        for row_first in range(0, len(rows), row_step):
            selected_rows = rows[row_first:row_first+row_step]
            for grid_first in range(0, len(grid), grid_step):
                selected_grid = grid[grid_first:grid_first+grid_step]
                values = np.empty((len(selected_grid), len(selected_rows)))
                status = np.empty(values.shape, dtype=np.uint8)
                self._check(self.lib.CliPP2EvaluateRegionalGrid(self.context, region,
                    len(selected_rows), _ptr(selected_rows), len(selected_grid), _ptr(selected_grid),
                    _ptr(values), _ptr(status)), status)
                result[grid_first:grid_first+len(selected_grid), row_first:row_first+len(selected_rows)] = values
        return result.T

    def joint_columns(self, centers):
        self._require_open()
        centers = _array(centers)
        if centers.ndim != 2 or centers.shape[1] != self.r:
            raise ValueError("Centers must match clusters by regions")
        result = np.empty((len(centers), self.n))
        status = np.empty(result.shape, dtype=np.uint8)
        self._check(self.lib.CliPP2EvaluateJointColumns(self.context, len(centers), _ptr(centers),
                                                      _ptr(result), _ptr(status)), status)
        return result.T

    def resident_joint_columns(self, centers):
        """Evaluate/cache exact centers without exporting their N-sized columns."""
        self._require_open()
        centers = _array(centers)
        if (self.kind != "cuda" or centers.ndim != 2 or not len(centers)
                or centers.shape[1] != self.r or not np.isfinite(centers).all()):
            raise ValueError("Resident centers require a CUDA backend and finite q by R CCFs")
        handles = np.empty(len(centers), dtype=np.int64)
        self._check(self.lib.CliPP2CreateJointColumns(
            self.context, len(centers), _ptr(centers), _ptr(handles)))
        return tuple(DeviceColumn(self, handle) for handle in handles)

    def scoring_diagnostics(self):
        self._require_open()
        if self.kind != "cuda":
            raise ValueError("Resident scoring diagnostics require CUDA")
        counters = np.empty(9, dtype=np.int64)
        self._check(self.lib.CliPP2CUDAScoringDiagnostics(self.context, _ptr(counters)))
        return dict(zip(("column_evaluations", "column_cache_hits", "column_exports",
                         "column_gathers", "weight_evaluations", "transfer_preparations",
                         "transfer_trials", "live_column_handles", "live_mixture_problems"),
                        map(int, counters)))

    def import_joint_columns(self, log_columns):
        """Explicit host-to-device boundary for independent numerical checks."""
        self._require_open()
        columns = np.asarray(log_columns, dtype=float)
        if (self.kind != "cuda" or columns.ndim != 2 or columns.shape[0] != self.n
                or not columns.shape[1] or np.isnan(columns).any() or np.isposinf(columns).any()):
            raise ValueError("Expected nonempty N-by-q finite or -inf CUDA log columns")
        columns = _array(columns.T)
        handles = np.empty(len(columns), dtype=np.int64)
        self._check(self.lib.CliPP2ImportJointColumns(
            self.context, len(columns), _ptr(columns), _ptr(handles)))
        return tuple(DeviceColumn(self, handle) for handle in handles)

    def _column_handles(self, columns):
        self._require_open()
        if self.kind != "cuda" or not len(columns):
            raise ValueError("Expected nonempty resident CUDA columns")
        if any(not isinstance(column, DeviceColumn) or column.backend() is not self
               for column in columns):
            raise ValueError("Device columns belong to another backend context")
        return _array([column.handle for column in columns], np.int64)

    def export_joint_columns(self, columns):
        handles = self._column_handles(columns)
        result = np.empty((len(handles), self.n))
        self._check(self.lib.CliPP2ExportJointColumns(
            self.context, len(handles), _ptr(handles), _ptr(result)))
        return result.T

    def gather_joint_columns(self, columns, labels):
        handles = self._column_handles(columns)
        labels = np.asarray(labels)
        if (labels.shape != (self.n,) or labels.dtype.kind not in "iu"
                or np.any(labels < 0) or np.any(labels >= len(handles))):
            raise ValueError("Assigned column labels must be N valid integer column indices")
        labels, result = _array(labels, np.int64), np.empty(self.n)
        self._check(self.lib.CliPP2GatherJointColumns(
            self.context, len(handles), _ptr(handles), _ptr(labels), _ptr(result)))
        return result

    def mixture_problem(self, columns):
        return DeviceMixtureProblem(self, columns)

    def multiplicity(self, x):
        self._require_open()
        x = _array(np.asarray(x).T)
        if x.shape != (self.r, self.n):
            raise ValueError("CCFs must match N by R")
        result = np.empty_like(x)
        status = np.empty(x.shape, dtype=np.uint8)
        self._check(self.lib.CliPP2MultiplicityMAP(self.context, _ptr(x), _ptr(result), _ptr(status)), status)
        if np.any(status == 1):
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.ascontiguousarray(result.T)

    def refit(self, prepared):
        """Execute model-prepared scientific inputs, preserving their exact order."""
        self._require_open()
        names = ("regions", "member_offsets", "member_rows", "grid_offsets",
                 "grid", "lower", "upper", "xatol", "incumbent")
        raw = [np.asarray(getattr(prepared, name)) for name in names]
        if (any(value.ndim != 1 for value in raw)
                or any(value.dtype.kind not in "iu" or (
                    value.dtype.kind == "u" and np.any(value > np.iinfo(np.int64).max))
                       for value in raw[:4])):
            raise ValueError("Prepared refit indices require one-dimensional int64-compatible integers")
        arrays = [_array(value, np.int64 if index < 4 else np.float64)
                  for index, value in enumerate(raw)]
        regions, member_offsets, members, grid_offsets, grids, lowers, uppers, tolerances, incumbents = arrays
        count = len(regions)
        if (any(len(value) != count for value in arrays[5:])
                or any(len(offsets) != count+1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                       or offsets[-1] != len(values)
                       for offsets, values in ((member_offsets, members), (grid_offsets, grids)))):
            raise ValueError("Prepared refit arrays require consistent one-dimensional CSR fields")
        if not count:
            return []
        if self.kind == "cuda":
            # The native planner owns task slices and parallel-grid waves.
            # Python's former serial-grid estimate cannot bound that scratch.
            batch = _RefitBatch(count, len(members), len(grids), *map(_ptr, arrays))
            fitted = (_RefitResult * count)()
            self._check(self.lib.CliPP2RefitCentersBatch(self.context, ct.byref(batch), fitted))
            return list(fitted)
        sizes = zip(map(int, np.diff(member_offsets)), map(int, np.diff(grid_offsets)))
        costs = [8*(rows+grid+1) for rows, grid in sizes]
        results = []
        for selected in self._task_slices(costs, 8):
            count = selected.stop - selected.start
            offsets = slice(selected.start, selected.stop+1)
            first_member, last_member = member_offsets[[selected.start, selected.stop]]
            first_grid, last_grid = grid_offsets[[selected.start, selected.stop]]
            fields = [_array(regions[selected], np.int64),
                      _array(member_offsets[offsets]-first_member, np.int64),
                      _array(members[first_member:last_member], np.int64),
                      _array(grid_offsets[offsets]-first_grid, np.int64),
                      _array(grids[first_grid:last_grid]), _array(lowers[selected]), _array(uppers[selected]),
                      _array(tolerances[selected]), _array(incumbents[selected])]
            batch = _RefitBatch(count, len(fields[2]), len(fields[4]), *map(_ptr, fields))
            fitted = (_RefitResult * count)()
            self._check(self.lib.CliPP2RefitCentersBatch(self.context, ct.byref(batch), fitted))
            results.extend(fitted)
        return results

    def _task_slices(self, costs, fixed):
        """Tile whole independent tasks; never alter their reductions or grids."""
        task_limit = self.host_staging_bytes_limit // 8 if self.kind == "cuda" else len(costs)
        if costs and task_limit < 1:
            raise NativeExecutionError("CUDA conditional execution requires 8 native host staging bytes per task; "
                                       f"budget is {self.host_staging_bytes_limit}")
        first, used = 0, fixed
        for index, cost in enumerate(costs):
            if cost + fixed > self.workspace_bytes:
                raise NativeExecutionError(
                    f"Native conditional task {index} requires {cost + fixed} workspace bytes; "
                    f"budget is {self.workspace_bytes}. Increase workspace_bytes.")
            if used + cost > self.workspace_bytes or index-first >= task_limit:
                yield slice(first, index)
                first, used = index, fixed
            used += cost
        if first < len(costs):
            yield slice(first, len(costs))

    def conditional_values(self, tasks, centers):
        """Evaluate supplied incumbents natively; transfer only task totals."""
        self._require_open()
        if not tasks:
            return np.empty(0)
        centers = _array(centers)
        if centers.shape != (len(tasks),):
            raise ValueError("Conditional values need one center per task")
        costs = ([8*(len(rows)+1) for rows, _ in tasks] if self.kind == "cpu" else
                 [41+24*len(rows) for rows, _ in tasks])
        values = np.empty(len(tasks))
        for selected in self._task_slices(costs, 8 if self.kind == "cpu" else 128):
            chosen, count = tasks[selected], selected.stop-selected.start
            regions = _array([region for _, region in chosen], np.int64)
            offsets = _array(np.r_[0, np.cumsum([len(rows) for rows, _ in chosen])], np.int64)
            members = _array(np.concatenate([rows for rows, _ in chosen]), np.int64)
            batch = _ConditionalBatch(count, len(members),
                                       *map(_ptr, (regions, offsets, members, centers[selected])))
            status = np.empty(count, dtype=np.uint8)
            self._check(self.lib.CliPP2EvaluateConditionalValues(
                self.context, ct.byref(batch), _ptr(values[selected]), _ptr(status)), status)
        return values


NATIVE_ABI = 4

def native_library():
    root = Path(__file__).resolve().parent.parent
    path = next((root/('kernel/_native'+suffix) for suffix in EXTENSION_SUFFIXES
                 if (root/('kernel/_native'+suffix)).is_file()), None)
    if path is None:
        raise NativeExecutionError('Install CliPP2 (pip install .) to build its native kernel')
    signature = tuple((info.st_mtime_ns, info.st_ctime_ns, info.st_size)
                      for p in (path, *(root/name for name in NATIVE_SOURCES)) for info in (p.stat(),))
    return _load_library(root, path, signature)


@lru_cache(maxsize=4)
def _load_library(root, path, signature):
    digest = _native_identity(root)
    library = ct.CDLL(str(path))
    library.CliPP2KernelABI.argtypes = []
    library.CliPP2KernelABI.restype = ct.c_int
    library.CliPP2KernelBuildId.argtypes = []
    library.CliPP2KernelBuildId.restype = ct.c_char_p
    library.CliPP2KernelLastError.argtypes = []
    library.CliPP2KernelLastError.restype = ct.c_char_p
    if library.CliPP2KernelABI() != NATIVE_ABI or library.CliPP2KernelBuildId().decode() != digest:
        raise NativeExecutionError('CliPP2 kernel binary/source or ABI mismatch; rebuild before fitting')
    return library, digest


def _forest_memory_query(library):
    query = library.CliPP2ForestRequiredBytes
    query.argtypes, query.restype = [ct.c_int, ct.c_int], ct.c_longlong
    return query


def forest_required_bytes(n, edges):
    return _forest_memory_query(native_library()[0])(n, edges)


class PreparedForest:
    """Source-bound host-controller buffers, borrowed until the next solve.

    The numerical C entry still admits and releases its native allocations on
    every call. Only the caller-owned output and verified binding are retained.
    Source identity is rechecked before leaving the continuation batch.
    """
    def __init__(self, n, regions, *, native_host_memory_bytes=None):
        self.shape = n, regions
        self.library, self.identity = native_library()
        self.native_host_memory_bytes = resolve_memory_budget(native_host_memory_bytes)
        self.required = _forest_memory_query(self.library)
        self.result = self.diagnostics = None
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise NativeExecutionError('Prepared forest context is closed')
        return self

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.result = self.diagnostics = None
        if native_library()[1] != self.identity:
            raise NativeExecutionError('Native forest source changed during continuation')

    def __exit__(self, *_):
        self.close()

    def buffers(self):
        if self.closed:
            raise NativeExecutionError('Prepared forest context is closed')
        if self.result is None:
            from .execution import preflight_host_export
            preflight_host_export(self.shape[0]*self.shape[1]*8 + 8)
            self.result = np.empty(self.shape)
            self.diagnostics = np.empty(2, dtype=np.int32)
        return self.result, self.diagnostics


def forest_quadratic(curvature, rhs, edges, penalty, lower, upper, *, native_host_memory_bytes=None,
                     prepared=None):
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
    if prepared is not None and (type(prepared) is not PreparedForest or prepared.shape != h.shape):
        raise ValueError('Prepared forest must match the exact N by R shape')
    if prepared is not None and prepared.closed:
        raise NativeExecutionError('Prepared forest context is closed')
    library = native_library()[0] if prepared is None else prepared.library
    from .execution import preflight_host_export, resolve_memory_budget
    budget = (prepared.native_host_memory_bytes if prepared is not None and native_host_memory_bytes is None
              else resolve_memory_budget(native_host_memory_bytes))
    needed = _forest_memory_query(library)(n, len(edges))
    if needed < 0 or needed > budget:
        raise NativeExecutionError(f"Native forest requires {needed} tracked host bytes; available {budget} bytes")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    entry = library.CliPP2ForestQP
    entry.argtypes = [ct.c_int, ct.c_int, ct.c_int, ints, doubles, doubles,
                     ct.c_double, doubles, doubles, doubles, ints, ct.c_longlong]
    entry.restype = ct.c_int
    if prepared is None:
        preflight_host_export(h.nbytes + 8)
        result, diagnostics = np.empty_like(h), np.empty(2, dtype=np.int32)
    else:
        result, diagnostics = prepared.buffers()
    if entry(n, regions, len(edges), edges, h, b, float(penalty), lo, hi, result, diagnostics, budget):
        raise NativeExecutionError('CliPP2 forest kernel failed: '+library.CliPP2KernelLastError().decode())
    if not np.isfinite(result).all():
        raise NativeExecutionError('CliPP2 forest kernel returned nonfinite coordinates')
    return result, int(diagnostics[0]), int(diagnostics[1])


class _ContinuationPolicy(ct.Structure):
    _fields_ = [(name, ct.c_int) for name in ("levels", "iterations")] + [
        (name, ct.c_double) for name in ("stationarity", "constraint", "maximum_penalty")]


class _ContinuationReport(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in (
        "rho", "stationarity", "violation", "block_range", "likelihood", "objective")] + [
        (name, ct.c_int64) for name in (
            "iterations", "evaluations", "qp_calls", "qp_unqualified", "warm_starts", "accepted_steps")] + [
        (name, ct.c_int) for name in ("levels", "longest", "status")]


def continue_cpu(model, tree, pilot, requested, lower, upper, policy, missing, *, keep_raw_phi=True):
    """One source-bound native batch; capacities retain independent state.

    A finite host admission cap owns mandatory state, separately from temporary
    tile scratch. NumPy norm, cut-score einsum and scalar sine retain the
    reference's platform-specific arithmetic. Exported arrays are caller-owned.
    """
    from .execution import preflight_host_export

    model.backend._require_open()
    requested_workers = model.backend.cpu_threads
    workers = min(requested_workers, max(model.n, model.r, len(requested)))
    if max(model.n, model.r, len(requested), policy.levels, policy.iterations_per_level) > np.iinfo(np.int32).max:
        raise ValueError("CPU continuation dimensions and budgets must fit int32")
    lib, identity = native_library()
    memory = lib.CliPP2ContinuationMemory
    memory.argtypes = [ct.c_int]*4 + [ct.POINTER(ct.c_longlong)]*3
    memory.restype = ct.c_int
    shared, state, per_thread = (ct.c_longlong() for _ in range(3))
    if memory(model.n, model.r, workers, len(requested), ct.byref(shared),
              ct.byref(state), ct.byref(per_thread)):
        raise NativeExecutionError("Continuation memory dimensions overflow")
    available = model.backend.native_host_memory_bytes - model.backend.native_host_bytes
    minimum = shared.value + state.value + per_thread.value
    requested_bytes = shared.value + state.value*(workers if len(requested) >= workers else 1) + per_thread.value*workers
    if requested_bytes > available:
        # Optional plans must not crowd out mandatory state or its worker team.
        model.backend.release_optional_cache()
        available = model.backend.native_host_memory_bytes - model.backend.native_host_bytes
    if minimum > available:
        raise NativeExecutionError(f"CPU continuation requires {minimum} tracked native host bytes; available {available} bytes")
    export_bytes = len(requested)*(model.n*model.r*8*int(keep_raw_phi) + model.n*12
                                  + (model.n-1)*9 + ct.sizeof(_ContinuationReport))
    preflight_host_export(export_bytes)
    observations = _Observations()
    view = model.backend.lib.CliPP2BackendObservationView
    view.argtypes = [ct.c_void_p, ct.POINTER(_Observations)]
    view.restype = ct.c_int
    model.backend._check(view(model.backend.context, ct.byref(observations)))
    initial = _array(np.clip(pilot, lower, upper))
    lower, upper = _array(lower), _array(upper)
    edges = np.ascontiguousarray(tree.edges, dtype=np.int32)
    canonical = np.ascontiguousarray(sorted(range(tree.n), key=lambda i: tree.mutation_ids[i]), dtype=np.int32)
    capacities = np.ascontiguousarray(requested, dtype=np.int32)
    # Scalar evaluation, exactly as in the host controller, not a SIMD sine.
    gaps = np.array([0.] + [4*np.sin(np.pi/(2*size))**2 for size in range(1, model.n+1)])
    native_policy = _ContinuationPolicy(policy.levels, policy.iterations_per_level,
        policy.stationarity_tolerance, policy.constraint_tolerance, policy.maximum_penalty)
    output = np.empty((len(requested), model.n, model.r)) if keep_raw_phi else None
    labels = np.empty((len(requested), model.n), dtype=np.int32)
    cuts = np.empty((len(requested), model.n-1), dtype=np.uint8)
    reports = (_ContinuationReport*len(requested))()
    norm_type = ct.CFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double), ct.c_longlong)
    scores_type = ct.CFUNCTYPE(ct.c_int, ct.POINTER(ct.c_longdouble),
        ct.c_longlong, ct.c_int, ct.POINTER(ct.c_longdouble))
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
            # ct' long-double PEP-3118 code is not understood by NumPy;
            # expose the same aligned storage as bytes, then view its dtype.
            def extended(pointer, size):
                raw = ct.cast(pointer, ct.POINTER(ct.c_uint8))
                return np.ctypeslib.as_array(raw,
                    shape=(size*np.dtype(np.longdouble).itemsize,)).view(np.longdouble)

            delta = extended(values, count*regions).reshape(count, regions)
            np.copyto(extended(output, count), np.einsum("ij,ij->i", delta, delta))
            return 0
        except BaseException as error:
            reduction_errors.append(error)
            return 1

    doubles = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ints = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    bytes_ = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    entry = lib.CliPP2ContinueCPU
    entry.argtypes = [ct.POINTER(_Observations), ints, ints, doubles, doubles, doubles,
        ct.c_int, ints, ct.POINTER(_ContinuationPolicy), doubles, norm_type, scores_type,
        ct.c_int, ct.c_longlong, ct.c_longlong, ct.c_int, ct.c_void_p, ints, bytes_,
        ct.POINTER(_ContinuationReport), ct.POINTER(ct.c_int)]
    entry.restype = ct.c_int
    lib.CliPP2ContinuationLastError.argtypes = []
    lib.CliPP2ContinuationLastError.restype = ct.c_char_p
    actual = ct.c_int()
    code = entry(ct.byref(observations), edges, canonical, initial, lower, upper,
        len(requested), capacities, ct.byref(native_policy), gaps, norm, cut_scores, workers,
        model.backend.workspace_bytes, available, int(keep_raw_phi),
        None if output is None else output.ctypes.data, labels, cuts, reports, ct.byref(actual))
    if reduction_errors:
        raise NativeExecutionError("CPU continuation compatibility reduction failed") from reduction_errors[0]
    if code:
        raise NativeExecutionError(lib.CliPP2ContinuationLastError().decode())
    if output is not None and not np.isfinite(output).all():
        raise NativeExecutionError("CPU continuation returned nonfinite coordinates")
    statuses = ("penalty_stationary_constraint_tolerance", "penalty_numerical_limit_candidate",
                "line_search_stalled_candidate", "finite_budget_candidate")
    execution = {"continuation": "native_cpu", "build_id": identity,
        "workers": actual.value, "capacities": len(requested), "initial_evaluations": 1,
        "node_evaluations": 1+sum(report.evaluations for report in reports),
        "qp_calls": sum(report.qp_calls for report in reports),
        "workspace_limit": model.backend.workspace_bytes,
        "native_host_memory_limit": model.backend.native_host_memory_bytes,
        "native_host_available_for_continuation": available,
        "caller_export_bytes": export_bytes, "raw_coordinates_exported": keep_raw_phi,
        "cpu_thread_budget": requested_workers,
        "parallel_axis": "capacities" if len(requested) >= actual.value else "cells_and_regions",
        "compatibility_reductions": ["numpy_longdouble_einsum", "numpy_blas_norm"],
        "tracked_native_reserved_bound_bytes": shared.value + per_thread.value*actual.value + state.value*(
            actual.value if len(requested) >= actual.value else 1)}
    execution["cell_evaluations"] = execution["node_evaluations"]*model.n*model.r
    result = []
    for index, (capacity, report) in enumerate(zip(requested, reports)):
        if not 0 <= report.status < len(statuses):
            raise NativeExecutionError("CPU continuation returned an unknown status")
        result.append({"labels": labels[index].astype(np.int64), "cuts": np.flatnonzero(cuts[index]),
            "requested_k": int(capacity), **({"raw_phi": output[index]} if keep_raw_phi else {}), "diagnostics": {
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
