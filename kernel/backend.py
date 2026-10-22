"""Source-bound CPU/CUDA emission and batched-refit ABI (host boundary only)."""
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .build_identity import backend_identity
from .execution import resolve_cpu_threads, resolve_memory_budget


_D = ct.POINTER(ct.c_double)
_I = ct.POINTER(ct.c_int64)
_B = ct.POINTER(ct.c_uint8)


class NativeExecutionError(RuntimeError):
    """Backend, resource or arithmetic failure; never a candidate rejection."""


class ScientificCandidateRejection(ValueError):
    """A proposed cluster lacks regional support or a common admissible box."""


@dataclass(frozen=True, slots=True, eq=False)
class CoordinateTask:
    """Exact observed sequence with the enclosing group's all-member box."""

    region: int
    rows: np.ndarray
    lower: float
    upper: float
    xatol: float
    key: tuple


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


class _ResidentRequest(ct.Structure):
    _fields_ = [("plan", ct.c_uint64), ("incumbents", _D)]


@dataclass(frozen=True, slots=True)
class ResidentHandle:
    """Backend-scoped generation handle; no borrowed caller array survives preparation."""

    owner: object
    value: int
    kind: str
    count: int


def _ptr(array):
    return array.ctypes.data_as({np.dtype("float64"): _D, np.dtype("int64"): _I,
                                np.dtype("uint8"): _B}[array.dtype])


def _array(value, dtype=np.float64):
    return np.ascontiguousarray(value, dtype=dtype)


class NativeBackend:
    """Owns prepared SoA fields; never silently substitutes a CPU backend."""

    def __init__(self, model, workspace_bytes, *, native_host_memory_bytes=None,
                 device_memory_bytes=None, cpu_threads=None):
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
        if lib.CliPP2BackendABI() != 1 or self.identity != backend_identity(kind):
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
            self.host_staging_bytes_limit = min(self.native_host_memory_bytes, self.workspace_bytes)
            lib.CliPP2ConfigureCUDAExecution.argtypes = [ct.c_void_p, ct.c_int64]
            lib.CliPP2ConfigureCUDAExecution.restype = ct.c_int
            self._check(lib.CliPP2ConfigureCUDAExecution(self.context, self.host_staging_bytes_limit))
        self._initialize_resident()

    def _initialize_resident(self):
        signatures = {
            "ResidentABI": ([], ct.c_int), "ResidentLastError": ([], ct.c_char_p),
            "ResidentCreate": ([ct.c_void_p, ct.c_int64, ct.c_int64, ct.c_int64, ct.POINTER(ct.c_void_p)], ct.c_int),
            "ResidentDestroy": ([ct.c_void_p], None),
            "ResidentBytes": ([ct.c_void_p, _I], ct.c_int),
            "PrepareCoordinates": ([ct.c_void_p, ct.POINTER(_RefitBatch), ct.POINTER(ct.c_uint64)], ct.c_int),
            "SubmitRefits": ([ct.c_void_p, ct.POINTER(_ResidentRequest), ct.c_int64, ct.POINTER(ct.c_uint64)], ct.c_int),
            "CollectRefits": ([ct.c_void_p, ct.c_uint64, ct.c_int64, ct.POINTER(_RefitResult)], ct.c_int),
            "ReleaseHandle": ([ct.c_void_p, ct.c_uint64], ct.c_int),
            "CreateJointColumns": ([ct.c_void_p, ct.c_int64, _D, ct.POINTER(ct.c_uint64)], ct.c_int),
            "CollectJointColumns": ([ct.c_void_p, ct.c_uint64, ct.c_int64, _D, _B], ct.c_int),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.lib, "CliPP2" + name)
            function.argtypes, function.restype = args, result
        if self.lib.CliPP2ResidentABI() != 1:
            raise NativeExecutionError("Incompatible resident ABI")
        # Reserve the existing backend's maximum scratch/cache allowance before
        # admitting retained objects. Host result arrays remain caller-owned.
        reserved = (self.n*self.r*57 + self.workspace_bytes*3//2 if self.kind == "cpu"
                    else self.host_staging_bytes_limit)
        self.resident = ct.c_void_p()
        self._resident_check(self.lib.CliPP2ResidentCreate(
            self.context, self.n, self.r, max(0, self.native_host_memory_bytes-reserved), ct.byref(self.resident)))

    def _resident_check(self, code):
        if code:
            raise NativeExecutionError(self.lib.CliPP2ResidentLastError().decode())

    @property
    def native_host_bytes(self):
        result = _Diagnostics()
        self._check(self.lib.CliPP2BackendGetDiagnostics(self.context, ct.byref(result)))
        retained = ct.c_int64()
        if getattr(self, "resident", None):
            self._resident_check(self.lib.CliPP2ResidentBytes(self.resident, ct.byref(retained)))
        return retained.value + (result.prepared_bytes + result.cache_bytes if self.kind == "cpu" else 0)

    def release_optional_cache(self):
        """Give mandatory continuation state priority over rebuildable read plans."""
        if self.kind == "cpu":
            self._check(self.lib.CliPP2ReleaseProbabilityCache(self.context))
        return self.native_host_bytes

    def __del__(self):
        resident = getattr(self, "resident", None)
        if resident:
            self.lib.CliPP2ResidentDestroy(resident)
            self.resident = None
        context = getattr(self, "context", None)
        if context:
            self.lib.CliPP2BackendDestroy(context)
            self.context = None

    def _check(self, code, status=None):
        if code:
            raise NativeExecutionError(self.lib.CliPP2BackendLastError().decode())
        if status is not None:
            if np.any(status > 2):
                raise NativeExecutionError("Native likelihood produced invalid arithmetic or an unknown status")
            if np.any(status == 2):
                raise ValueError("Native likelihood received invalid CCFs")

    def diagnostics(self):
        result = _Diagnostics()
        self._check(self.lib.CliPP2BackendGetDiagnostics(self.context, ct.byref(result)))
        diagnostics = {"kind": self.kind, "build_id": self.identity, "numerical_policy": self.policy,
                       **{name: getattr(result, name) for name, _ in result._fields_}}
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
                           cpu_threads=self.cpu_threads, native_host_bytes=self.native_host_bytes,
                           resident_api=1, resident_scheduler="synchronous_host_results")
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
        centers = _array(centers)
        if centers.ndim != 2 or centers.shape[1] != self.r:
            raise ValueError("Centers must match clusters by regions")
        result = np.empty((len(centers), self.n))
        status = np.empty(result.shape, dtype=np.uint8)
        self._check(self.lib.CliPP2EvaluateJointColumns(self.context, len(centers), _ptr(centers),
                                                      _ptr(result), _ptr(status)), status)
        return result.T

    def multiplicity(self, x):
        x = _array(np.asarray(x).T)
        if x.shape != (self.r, self.n):
            raise ValueError("CCFs must match N by R")
        result = np.empty_like(x)
        status = np.empty(x.shape, dtype=np.uint8)
        self._check(self.lib.CliPP2MultiplicityMAP(self.context, _ptr(x), _ptr(result), _ptr(status)), status)
        if np.any(status == 1):
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.ascontiguousarray(result.T)

    def _handle(self, handle, kind):
        if not isinstance(handle, ResidentHandle) or handle.owner is not self or handle.kind != kind:
            raise ValueError("Resident handle has the wrong owner or type")
        return handle.value

    def release(self, handle):
        """Release caller ownership. Jobs keep their plans alive until released."""
        if not isinstance(handle, ResidentHandle) or handle.owner is not self:
            raise ValueError("Resident handle belongs to another backend")
        self._resident_check(self.lib.CliPP2ReleaseHandle(self.resident, handle.value))

    def prepare_coordinates(self, model, tasks):
        """Copy ordered reads, all-member bounds and exact grids into a native plan."""
        tasks = tuple(task if isinstance(task, CoordinateTask) else model._coordinate_task(*task)
                      for task in tasks)
        if not tasks:
            raise ValueError("Resident preparation needs at least one coordinate")
        grids = [model._coordinate_grid(task) for task in tasks]
        fields = [_array([task.region for task in tasks], np.int64),
                  _array(np.r_[0, np.cumsum([len(task.rows) for task in tasks])], np.int64),
                  _array(np.concatenate([task.rows for task in tasks]), np.int64),
                  _array(np.r_[0, np.cumsum([len(grid) for grid in grids])], np.int64),
                  _array(np.concatenate(grids)), _array([task.lower for task in tasks]),
                  _array([task.upper for task in tasks]), _array([task.xatol for task in tasks])]
        batch = _RefitBatch(len(tasks), len(fields[2]), len(fields[4]), *map(_ptr, fields), None)
        handle = ct.c_uint64()
        self._resident_check(self.lib.CliPP2PrepareCoordinates(self.resident, ct.byref(batch), ct.byref(handle)))
        return ResidentHandle(self, handle.value, "plan", len(tasks))

    def submit_refits(self, plan, incumbents=None):
        """Execute in order; v1 completes native/CUDA work before returning."""
        value = self._handle(plan, "plan")
        values = np.full(plan.count, np.nan) if incumbents is None else _array(incumbents)
        if values.shape != (plan.count,):
            raise ValueError("Refitting needs one incumbent per coordinate task")
        request = _ResidentRequest(value, _ptr(values))
        job = ct.c_uint64()
        self._resident_check(self.lib.CliPP2SubmitRefits(self.resident, ct.byref(request), 1, ct.byref(job)))
        return ResidentHandle(self, job.value, "job", plan.count)

    def collect_refits(self, job):
        value = self._handle(job, "job")
        output = (_RefitResult * job.count)()
        self._resident_check(self.lib.CliPP2CollectRefits(self.resident, value, job.count, output))
        return list(output)

    def create_joint_columns(self, centers):
        """Retain checked host columns. Device-resident scoring is a later revision."""
        centers = _array(centers)
        if centers.ndim != 2 or centers.shape[1] != self.r or not len(centers):
            raise ValueError("Centers must match nonempty clusters by regions")
        handle = ct.c_uint64()
        self._resident_check(self.lib.CliPP2CreateJointColumns(
            self.resident, len(centers), _ptr(centers), ct.byref(handle)))
        return ResidentHandle(self, handle.value, "columns", len(centers))

    def collect_columns(self, columns):
        value = self._handle(columns, "columns")
        output = np.empty((columns.count, self.n))
        status = np.empty(output.shape, dtype=np.uint8)
        self._resident_check(self.lib.CliPP2CollectJointColumns(
            self.resident, value, output.size, _ptr(output), _ptr(status)))
        self._check(0, status)
        return output.T

    def refit(self, model, tasks, incumbents=None):
        """Batch exact prepared tasks, retaining base fits separately from incumbents."""
        if not tasks:
            return []
        tasks = [task if isinstance(task, CoordinateTask) else model._coordinate_task(*task)
                 for task in tasks]
        incumbents = (np.full(len(tasks), np.nan) if incumbents is None else _array(incumbents))
        if incumbents.shape != (len(tasks),):
            raise ValueError("Refitting needs one incumbent per coordinate task")
        regions, members, grids, lowers, uppers, tolerances = [], [], [], [], [], []
        for task in tasks:
            regions.append(task.region)
            members.append(task.rows)
            grids.append(model._coordinate_grid(task))
            lowers.append(task.lower)
            uppers.append(task.upper)
            tolerances.append(task.xatol)
        costs = ([8*(len(rows)+len(grid)+1) for rows, grid in zip(members, grids)]
                 if self.kind == "cpu" else
                 [112+24*len(rows)+16*len(grid) for rows, grid in zip(members, grids)])
        results = []
        for selected in self._task_slices(costs, 8 if self.kind == "cpu" else 128):
            count = selected.stop - selected.start
            fields = [_array(regions[selected], np.int64),
                      _array(np.r_[0, np.cumsum([len(x) for x in members[selected]])], np.int64),
                      _array(np.concatenate(members[selected]), np.int64),
                      _array(np.r_[0, np.cumsum([len(x) for x in grids[selected]])], np.int64),
                      _array(np.concatenate(grids[selected])), _array(lowers[selected]), _array(uppers[selected]),
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
