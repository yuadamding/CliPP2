"""CliPP2 emissions, conditional centers and joint reads/validity mixture score.

CCF, not cellular prevalence, is the fitting/geometry scale for every region.
One shared emission primitive serves arrays, pilot grids and coordinate views.
Mixed CN explicitly retains CliPP2's capped support, weighted denominator,
clipped likelihood and original interior box. Assignment validity is observed
jointly with reads; it is not conditioned away by renormalizing mixture weights.
CUDA is an explicit float64 likelihood/host-forest path, never a CPU fallback;
emissions and conditional centers share a native float64 numerical policy.
Forest continuation and candidate selection remain on the host. The reference
weight controller uses resident CUDA evaluation; an explicit experimental mode
also keeps its optimizer state on the device. Neither route falls back to CPU.
"""
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import MappingProxyType
import hashlib
import math
import time

import numpy as np
from scipy.special import gammaln

from . import SCORE_DEFINITION
from .backend import DeviceColumn, NativeBackend, NativeExecutionError, ScientificCandidateRejection
from .initialization import pooled_ccf_initialization
from .likelihood import MIXED_EPS, RegionalView
from .scoring import WEIGHT_CONTROLLER_ID, WeightConvergenceError, fit_mixture_weights
from .topology import Membership
from ..io.data import readonly_array
from ..config import DEFAULT_MAX_CLUSTERS


# Mixed-CN emissions/boxes stay fixed; their assignment mask now enters scoring.
MODEL_VERSION = "clipp2_regional_full_support__general_normal_v1__mixed_cn_bulk_assignment_event_v2"
# Conservative retained Python-object/dictionary reservations, separate from
# array/key payload bytes. Shared immutable owners may be charged more than once.
_CACHE_ENTRY_BYTES = 1280
_MODE_ENTRY_BYTES = 768


@dataclass(frozen=True, slots=True, eq=False)
class CoordinateTask:
    """Exact observed sequence with the enclosing group's all-member box."""

    region: int
    rows: np.ndarray
    lower: float
    upper: float
    xatol: float
    key: tuple


@dataclass(frozen=True, slots=True, eq=False)
class PreparedRefitBatch:
    """Model-owned ordered reads, grids, boxes and incumbent coordinates."""

    regions: np.ndarray
    member_offsets: np.ndarray
    member_rows: np.ndarray
    grid_offsets: np.ndarray
    grid: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    xatol: np.ndarray
    incumbent: np.ndarray


@dataclass(frozen=True, eq=False)
class PreparedGroup:
    """Immutable ordered membership and read/box metadata for every region."""

    model_identity: str
    members: np.ndarray
    tasks: tuple

    @property
    def observed_members(self):
        return tuple(task.rows for task in self.tasks)

    @property
    def support_counts(self):
        return tuple(len(task.rows) for task in self.tasks)


@dataclass(frozen=True)
class JointColumnSet:
    """Call-owned immutable column handles; cache eviction cannot invalidate them.

    CUDA columns own device leases. Materialization is an explicit boundary for
    host proposal/verification consumers, never required by CUDA mixture fitting.
    """

    columns: tuple
    center_keys: tuple
    model_identity: str
    numerical_identity: tuple

    def materialize(self):
        if isinstance(self.columns[0], DeviceColumn):
            return self.columns[0].backend().export_joint_columns(self.columns)
        matrix = np.empty((len(self.columns[0]), len(self.columns)))
        for index, column in enumerate(self.columns):
            matrix[:, index] = column
        return matrix

    def assigned_values(self, labels):
        if isinstance(self.columns[0], DeviceColumn):
            return self.columns[0].backend().gather_joint_columns(self.columns, labels)
        return self.materialize()[np.arange(len(labels)), labels]

    def subset_values(self, rows, indices):
        """Gather only selected mutation rows and center columns, preserving order."""
        rows, indices = np.asarray(rows), np.asarray(indices)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows >= len(self.columns[0]))
                or np.any(rows < 0) or indices.ndim != 1 or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= len(self.columns))):
            raise ValueError("Column subsets require valid integer row and center indices")
        selected = tuple(self.columns[int(index)] for index in indices)
        if not selected:
            return np.empty((len(rows), 0))
        if isinstance(selected[0], DeviceColumn):
            return selected[0].backend().gather_column_rows(selected, rows)
        return np.column_stack([np.asarray(column)[rows] for column in selected])

    def positive_transfers(self, labels):
        """Exact positive finite assignment gains in original row-major order."""
        if isinstance(self.columns[0], DeviceColumn):
            return self.columns[0].backend().positive_transfers(self.columns, labels)
        labels = np.asarray(labels)
        matrix = self.materialize()
        if (labels.shape != (len(matrix),) or labels.dtype.kind not in "iu"
                or np.any(labels < 0) or np.any(labels >= matrix.shape[1])):
            raise ValueError("Assignment labels must match the column set")
        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise ValueError("Reassignment likelihood columns must be finite or -inf")
        assigned = matrix[np.arange(len(matrix)), labels]
        if not np.isfinite(assigned).all():
            raise ValueError("A reassignment parent has inadmissible assigned centers")
        gains = matrix-assigned[:, None]
        rows, destinations = np.nonzero(np.isfinite(gains) & (gains > 0))
        return rows, destinations, gains[rows, destinations]


def _replay_weight_progress(events, names, counts, notify):
    for completed, kind in enumerate(events, 1):
        event = {"operation": "weight_fit",
                 "boundary": "resident_complete" if kind == 3 else "request_complete",
                 "request": names[kind], "occupied_q": len(counts)}
        if kind != 3:
            event["completed_requests"] = completed
        notify(event)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(map(_freeze, value))
    return readonly_array(value) if isinstance(value, np.ndarray) else value


@dataclass(frozen=True, eq=False)
class FitResult:
    """One immutable numerical result; search provenance belongs to Candidate."""

    membership: Membership
    centers: np.ndarray
    weights: np.ndarray
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    score: float
    eligible: bool
    reason: str | None = None
    diagnostics: Mapping = field(default_factory=dict)
    coordinate_fits: np.ndarray | None = None

    def __post_init__(self):
        for name in ("centers", "weights", "coordinate_fits"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))

    @property
    def labels(self):
        return self.membership.labels

    @property
    def penalty(self):
        return self.complexity_penalty

class RegionalModel:
    """Immutable, canonically ordered informative mutation vectors."""

    _resident_weights = False

    def __init__(self, data, device="cpu", *, cache_bytes=32 * 1024 * 1024,
                 grid_cache_bytes=None, weight_cache_bytes=None,
                 workspace_bytes=64 * 1024**2, native_host_memory_bytes=None,
                 device_memory_bytes=None, cpu_threads=None, device_cache_bytes=256*1024**2,
                 score_capacity=DEFAULT_MAX_CLUSTERS, probability_reuse=False,
                 resident_initialization=False, resident_weights=False):
        self.data = data
        ids, regions = tuple(data.mutation_ids), tuple(data.region_ids)
        if any(not isinstance(x, str) for x in ids + regions):
            raise ValueError("Mutation and region identities must be strings")
        if len(set(ids)) != len(ids) or len(set(regions)) != len(regions):
            raise ValueError("Mutation and region identities must be unique")
        self.region_indices = readonly_array(sorted(range(len(regions)), key=regions.__getitem__), dtype=int)
        counts = np.asarray(data.total_counts)
        observed = np.ones_like(counts, dtype=bool) if data.count_observed is None else np.asarray(data.count_observed, dtype=bool)
        observed = observed & np.isfinite(counts) & (counts > 0)
        active = np.flatnonzero(observed.any(axis=1))
        if not len(active):
            raise ValueError("No informative positive-depth mutation remains")
        self.active_indices = readonly_array(sorted(active, key=ids.__getitem__), dtype=int)
        self.mutation_ids = tuple(ids[i] for i in self.active_indices)
        self.region_ids = tuple(regions[i] for i in self.region_indices)
        self.exclusions = tuple({"mutation_id": ids[i], "reason": "no_informative_observations"}
                                for i in np.flatnonzero(~observed.any(axis=1)))
        ix = np.ix_(self.active_indices, self.region_indices)
        self.observed = readonly_array(observed[ix], dtype=bool)
        self.n, self.r = self.observed.shape
        if not self.observed.any(axis=0).all():
            absent = [self.region_ids[j] for j in range(self.r) if not self.observed[:, j].any()]
            raise ValueError(f"Regions have no informative observations: {absent}")
        for name, source in (("alt", data.alt_counts), ("depth", data.total_counts),
                             ("major", data.major_cn), ("normal", data.normal_cn),
                             ("total", data.mean_total_cn), ("purity", data.purity)):
            setattr(self, name, readonly_array(np.asarray(source)[ix], dtype=float))
        self.mixed = readonly_array(np.asarray(data.cn_state_count)[ix] > 1)
        for values in (self.major, self.purity, self.total, self.normal):
            if not np.isfinite(values).all():
                raise ValueError("CN and purity must be finite in every mutation-region")
        if np.any(self.major < 1) or np.any(self.major != np.rint(self.major)):
            raise ValueError("Major CN must be a positive integer")
        if np.any(self.normal < 0) or np.any(self.normal != np.rint(self.normal)):
            raise ValueError("Normal CN must be a nonnegative integer")
        if np.any(self.total[~self.mixed] < self.major[~self.mixed]) or np.any(
                self.total[~self.mixed] != np.rint(self.total[~self.mixed])):
            raise ValueError("Single-state total CN must be an integer at least major CN")
        if np.any(self.purity <= 0) or np.any(self.purity > 1):
            raise ValueError("Purity must be in (0,1]")
        if not np.all(self.purity == self.purity[:1, :]):
            raise ValueError("Purity must be constant within each biological region")
        self.region_purities = readonly_array(self.purity[0])
        if (np.any(~np.isfinite(self.alt[self.observed]))
                or np.any(self.alt[self.observed] < 0)
                or np.any(self.alt[self.observed] > self.depth[self.observed])
                or np.any(self.alt[self.observed] != np.rint(self.alt[self.observed]))
                or np.any(self.depth[self.observed] != np.rint(self.depth[self.observed]))):
            raise ValueError("Informative read counts must be finite nonnegative integers")
        self.alt = readonly_array(np.where(self.observed, self.alt, 0.))
        self.depth = readonly_array(np.where(self.observed, self.depth, 0.))
        denominator = (1 - self.purity) * self.normal + self.purity * self.total
        if np.any(denominator <= 0):
            raise ValueError("Normal-plus-tumor denominator must be positive")
        self.support = readonly_array(np.where(self.mixed, np.minimum(self.major, 4), self.major), dtype=int)
        self.lower = readonly_array(np.where(self.mixed, MIXED_EPS, 0.0))
        self.upper = readonly_array(np.where(self.mixed, np.asarray(data.phi_upper)[ix], 1.0))
        if np.any(self.upper < self.lower) or np.any(self.upper > 1) or not np.isfinite(self.upper).all():
            raise ValueError("Invalid original mixed-CN bounds")
        self.scaling = readonly_array(self.purity / denominator)
        self.constants = readonly_array(
            gammaln(self.depth + 1) - gammaln(self.alt + 1)
            - gammaln(self.depth - self.alt + 1) - np.log(self.support))
        digest = hashlib.sha256(MODEL_VERSION.encode())
        for values in (self.alt, self.depth, self.observed):
            digest.update(values.tobytes())
        # Preserve the established scientific identity without retaining padded
        # N-by-R-by-M arrays. Execution layout and workspace limits are inert.
        multiplicities = np.arange(1, int(self.support.max())+1)
        for row in self.scaling:
            digest.update((row[:, None] * multiplicities).tobytes())
        for values in (self.support, self.mixed, self.lower, self.upper):
            digest.update(values.tobytes())
        digest.update(repr((self.mutation_ids, self.region_ids)).encode())
        self.identity = digest.hexdigest()
        self.device = str(device)
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("Regional likelihood device must be cpu or cuda")
        self.grid_cache_bytes = grid_cache_bytes
        if grid_cache_bytes is not None and (type(grid_cache_bytes) is not int or grid_cache_bytes < 0):
            raise ValueError("grid_cache_bytes must be a nonnegative integer")
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError("workspace_bytes must be a nonnegative integer")
        self._cache_bytes = int(cache_bytes)
        if self._cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        if weight_cache_bytes is None:
            weight_cache_bytes = min(self._cache_bytes, 4 * 1024**2)
        if type(weight_cache_bytes) is not int or weight_cache_bytes < 0:
            raise ValueError("weight_cache_bytes must be a nonnegative integer")
        self._weight_cache_bytes = weight_cache_bytes
        if any(type(flag) is not bool for flag in
               (probability_reuse, resident_initialization, resident_weights)):
            raise TypeError("Experimental CUDA execution flags must be booleans")
        if resident_weights and self.device == "cpu":
            raise ValueError("Resident weight fitting requires explicit CUDA execution")
        self._resident_weights = resident_weights
        execution_flags = {}
        if probability_reuse:
            execution_flags["probability_reuse"] = True
        if resident_initialization:
            execution_flags["resident_initialization"] = True
        self.backend = NativeBackend(self, workspace_bytes,
                                     native_host_memory_bytes=native_host_memory_bytes,
                                     device_memory_bytes=device_memory_bytes,
                                     cpu_threads=cpu_threads, device_cache_bytes=device_cache_bytes,
                                     score_capacity=score_capacity, **execution_flags)
        self.execution_backend = f"native_{self.backend.kind}_likelihood_host_forest"
        self._refits, self._columns = OrderedDict(), OrderedDict()
        self._refit_bytes = self._column_bytes = 0
        self._weight_results, self._weight_result_bytes = OrderedDict(), 0
        # Group metadata, exact component modes and coordinate grids share one
        # bounded host preparation cache. Active caller-owned tasks are scratch.
        self._preparations, self._preparation_bytes = OrderedDict(), 0
        self.telemetry = {"refit_cache_hits": 0, "scalar_refits": 0,
                          "column_cache_hits": 0, "likelihood_columns": 0,
                          "likelihood_column_requests": 0,
                          "column_host_cache_hits": 0, "column_batch_reuses": 0,
                          "column_device_cache_hits": 0, "column_device_retained_hits": 0,
                          "joint_matrix_calls": 0,
                          "group_cache_hits": 0, "coordinate_grid_cache_hits": 0,
                          "component_mode_cache_hits": 0, "coordinate_grids": 0,
                          "incumbent_equal_base_reuses": 0, "fused_incumbents": 0,
                          "unchanged_coordinate_reuses": 0,
                          "scalar_refit_seconds": 0.0, "weight_seconds": 0.0,
                          "weight_cache_hits": 0, "weight_cache_misses": 0,
                          "weight_cache_evictions": 0, "weight_solves": 0}

    def __len__(self):
        return self.n

    def execution_diagnostics(self):
        """Execution-only accounting; never included in model or candidate identity."""
        return {"backend": self.execution_backend, "grid_cache_bytes": self.grid_cache_bytes,
                "weight_execution": "resident_experimental" if self._resident_weights else "host_reference",
                "conditional_refit_backend": f"native_{self.backend.kind}",
                "native": self.backend.diagnostics(),
                "continuation": getattr(self, "_continuation_execution", None),
                "host_cache_bytes": {"preparation": self._preparation_bytes,
                                     "refit": self._refit_bytes, "column": self._column_bytes,
                                     "limit_per_category": self._cache_bytes,
                                     "weight": self._weight_result_bytes,
                                     "weight_limit": self._weight_cache_bytes,
                                     "total": (self._preparation_bytes + self._refit_bytes
                                               + self._column_bytes + self._weight_result_bytes),
                                     "total_limit": 3*self._cache_bytes + self._weight_cache_bytes},
                "device": None}

    def loss(self, x):
        return self.backend.node_terms(x, False)

    def node_terms(self, x):
        """Loss, exact gradient, positive complete-data curvature on CCF scale."""
        return self.backend.node_terms(x)

    def multiplicity(self, x):
        return self.backend.multiplicity(x)

    def view(self, region, rows=None):
        return RegionalView(self, region, np.arange(self.n) if rows is None else rows)

    def _regional_likelihood(self, values, rows, region):
        """Grid view of the same emissions used by the joint model."""
        return self.view(region, rows).log_likelihood_columns(np.atleast_1d(values)).T

    def pilots(self):
        pilots, diagnostics = np.full((self.n, self.r), np.nan), {}
        for region in range(self.r):
            rows = np.flatnonzero(self.observed[:, region])
            pilots[rows, region], diag = pooled_ccf_initialization(
                self.view(region, rows), cache_bytes=self.grid_cache_bytes)
            diagnostics[self.region_ids[region]] = diag
        return pilots, diagnostics

    def observation_keys(self):
        """ID-free scientific row signatures for deterministic proposal ties."""
        result = []
        support = np.arange(1, int(self.support.max())+1)
        for i in range(self.n):
            arrays = (self.observed[i], self.alt[i], self.depth[i],
                      self.scaling[i, :, None]*support, self.support[i],
                      self.mixed[i], self.lower[i], self.upper[i])
            result.append(int.from_bytes(hashlib.sha256(b''.join(a.tobytes() for a in arrays)).digest(), 'big'))
        return result

    def regional_profile_columns(self, region, grid):
        """Host proposal-grid emissions with original boxes, including missing CN.

        This coarse profile is only a proposal heuristic, never a fitted score.
        Missing reads contribute zero, while their original bounds still apply.
        """
        grid = np.asarray(grid, dtype=float)
        columns = np.zeros((self.n, len(grid)))
        rows = np.flatnonzero(self.observed[:, region])
        for first in range(0, len(rows), 256):
            active = rows[first:first+256]
            columns[active] = self._regional_likelihood(grid, active, region).T
        columns[(grid[None, :] < self.lower[:, region, None])
                | (grid[None, :] > self.upper[:, region, None])] = -np.inf
        return columns

    def _fit_scalar(self, members, region):
        return self._fit_coordinates([(members, region)])[0][0]

    def _prepared(self, key, make, counter):
        """Bound all retained owners, conservatively including shared payloads."""
        cached = self._preparations.get(key)
        if cached is not None:
            self._preparations.move_to_end(key)
            self.telemetry[counter] += 1
            return cached[0]
        value, size = make()
        if size <= self._cache_bytes:
            while self._preparation_bytes + size > self._cache_bytes:
                _, (_, removed) = self._preparations.popitem(last=False)
                self._preparation_bytes -= removed
            self._preparations[key] = (value, size)
            self._preparation_bytes += size
        return value

    def prepare_group(self, members):
        """Intern structural metadata; eviction changes cost, never eligibility."""
        members = self.backend._rows(0, members)
        if not len(members):
            raise ValueError("A coordinate task requires nonempty membership")
        group_key = ("group", members.tobytes())

        def prepare_group():
            tasks = []
            size = _CACHE_ENTRY_BYTES + 2 * len(group_key[1])
            for coordinate in range(self.r):
                rows = readonly_array(members[self.observed[members, coordinate]], dtype=np.int64)
                lower = float(self.lower[members, coordinate].max())
                upper = float(self.upper[members, coordinate].min())
                standard = (not self.mixed[rows, coordinate].any()
                            and np.all(self.normal[rows, coordinate] == 2))
                xatol = (1e-12 / float(self.region_purities[coordinate])
                          if standard and lower == 0 and upper == 1 else 1e-12)
                key = (self.identity, self.backend.policy, self.backend.identity, coordinate,
                       rows.tobytes(), np.asarray([lower, upper, xatol]).tobytes())
                tasks.append(CoordinateTask(coordinate, rows, lower, upper, xatol, key))
                size += _CACHE_ENTRY_BYTES + rows.nbytes + len(key[4]) + len(key[5])
            return PreparedGroup(self.identity, readonly_array(members), tuple(tasks)), size

        return self._prepared(group_key, prepare_group, "group_cache_hits")

    def _coordinate_task(self, members, region):
        group = members if isinstance(members, PreparedGroup) else self.prepare_group(members)
        if group.model_identity != self.identity:
            raise ValueError("Prepared group belongs to another regional model")
        task = group.tasks[region]
        if task.key[:3] != (self.identity, self.backend.policy, self.backend.identity):
            raise ValueError("Prepared group belongs to another numerical backend")
        if not len(task.rows):
            raise ScientificCandidateRejection(f"unsupported_cluster_region:{self.region_ids[region]}")
        if task.lower > task.upper:
            raise ScientificCandidateRejection(f"empty_cluster_region_box:{self.region_ids[region]}")
        return task

    def _coordinate_grid(self, task):
        def prepare_grid():
            modes = []
            for row in task.rows:
                def prepare_modes():
                    support = np.arange(1, int(self.support[row, task.region])+1)
                    values = readonly_array((self.alt[row, task.region] / self.depth[row, task.region])
                                            / (self.scaling[row, task.region] * support))
                    return values, _MODE_ENTRY_BYTES + values.nbytes
                modes.append(self._prepared(("modes", task.region, int(row)), prepare_modes,
                                            "component_mode_cache_hits"))
            grid = readonly_array(np.unique(np.r_[np.linspace(task.lower, task.upper, 513),
                np.clip(np.concatenate(modes), task.lower, task.upper)]))
            self.telemetry["coordinate_grids"] += 1
            return grid, _CACHE_ENTRY_BYTES + len(task.key[4]) + grid.nbytes
        return self._prepared(("grid", task.key), prepare_grid, "coordinate_grid_cache_hits")

    def prepare_refit_batch(self, tasks, incumbents=None):
        """Prepare scientific inputs once; the backend only tiles their execution."""
        tasks = tuple(task if isinstance(task, CoordinateTask) else self._coordinate_task(*task)
                      for task in tasks)
        initial = np.full(len(tasks), np.nan) if incumbents is None else np.asarray(incumbents, dtype=float)
        if initial.shape != (len(tasks),):
            raise ValueError("Refitting needs one incumbent per coordinate task")
        grids = [self._coordinate_grid(task) for task in tasks]
        integers = ([task.region for task in tasks],
                    np.r_[0, np.cumsum([len(task.rows) for task in tasks])],
                    np.concatenate([task.rows for task in tasks]) if tasks else [],
                    np.r_[0, np.cumsum([len(grid) for grid in grids])])
        floating = (np.concatenate(grids) if grids else [], [task.lower for task in tasks],
                    [task.upper for task in tasks], [task.xatol for task in tasks], initial)
        return PreparedRefitBatch(*(readonly_array(value, dtype=np.int64) for value in integers),
                                  *(readonly_array(value, dtype=float) for value in floating))

    def _fit_coordinates(self, tasks, incumbents=None, prepared_results=None, *, return_details=False):
        tasks = [self._coordinate_task(*task) for task in tasks]
        initial = np.full(len(tasks), np.nan) if incumbents is None else np.asarray(incumbents, dtype=float)
        if initial.shape != (len(tasks),):
            raise ValueError("Refitting needs one incumbent per coordinate task")
        results, missing = [None]*len(tasks), []
        retained_flags = np.zeros(len(tasks), dtype=bool)
        hits, retained_initial = [], 0
        for index, task in enumerate(tasks):
            prior = None if prepared_results is None else prepared_results[index]
            if (prior is not None
                    and np.float64(initial[index]).tobytes() == np.float64(prior[0]).tobytes()):
                results[index] = tuple(prior[:2])
                retained_flags[index] = bool(prior[2])
                retained_initial += int(retained_flags[index])
                self.telemetry["unchanged_coordinate_reuses"] += 1
                continue
            key = task.key
            if key in self._refits:
                self.telemetry["refit_cache_hits"] += 1
                self._refits.move_to_end(key)
                results[index] = self._refits[key][0]
                if np.isfinite(initial[index]) and task.lower <= initial[index] <= task.upper:
                    if initial[index] == results[index][0]:
                        self.telemetry["incumbent_equal_base_reuses"] += 1
                    else:
                        hits.append(index)
            else:
                missing.append((index, task))
        if missing:
            start = time.perf_counter()
            pending = [task for _, task in missing]
            starts = initial[[index for index, _ in missing]]
            fitted = self.backend.refit(self.prepare_refit_batch(
                pending, None if incumbents is None else starts))
            self.telemetry["scalar_refit_seconds"] += time.perf_counter()-start
            if len(fitted) != len(missing):
                raise NativeExecutionError("Native center refit returned an incomplete task batch")
            # Input support/boxes were checked before native execution. A native
            # failure is not scientific evidence against this partition. In
            # particular, interval status 1 conflates failed arithmetic and
            # unsuccessful search, so cannot safely justify candidate pruning.
            for (_, task), value in zip(missing, fitted):
                if value.status:
                    raise NativeExecutionError(
                        f"Marginal-likelihood center refit failed: native status {value.status}; "
                        f"region {self.region_ids[task.region]}, members {len(task.rows)}")
                if (not np.isfinite([value.base_center, value.base_log_likelihood,
                                     value.center, value.log_likelihood]).all()
                        or not task.lower <= value.base_center <= task.upper
                        or not task.lower <= value.center <= task.upper
                        or value.log_likelihood < value.base_log_likelihood):
                    raise NativeExecutionError("Native center refit returned invalid successful output")
            for (index, task), value in zip(missing, fitted):
                base = (value.base_center, value.base_log_likelihood)
                results[index] = (value.center, value.log_likelihood)
                retained_flags[index] = value.log_likelihood > value.base_log_likelihood
                retained_initial += int(retained_flags[index])
                self.telemetry["fused_incumbents"] += int(
                    np.isfinite(initial[index]) and task.lower <= initial[index] <= task.upper)
                self.telemetry["scalar_refits"] += 1
                self._cache_refit(task.key, base)
        if hits:
            values = self.backend.conditional_values([(tasks[i].rows, tasks[i].region) for i in hits],
                                                     initial[hits])
            for index, value in zip(hits, values):
                if value > results[index][1]:
                    results[index] = (float(initial[index]), float(value))
                    retained_flags[index] = True
                    retained_initial += 1
        return ((results, retained_initial, retained_flags) if return_details
                else (results, retained_initial))

    def _cache_refit(self, key, fitted):
        # Exact observed membership and bound bytes belong to the cache budget.
        size = _CACHE_ENTRY_BYTES + len(key[4]) + len(key[5])
        if size <= self._cache_bytes:
            old = self._refits.pop(key, None)
            if old is not None:
                self._refit_bytes -= old[1]
            while self._refit_bytes + size > self._cache_bytes:
                _, (_, removed) = self._refits.popitem(last=False)
                self._refit_bytes -= removed
            self._refits[key] = (fitted, size)
            self._refit_bytes += size

    def _validate_centers(self, centers):
        """Validate only q-by-R metadata; do not construct an N-by-q mask."""
        centers = np.asarray(centers, dtype=float)
        if (centers.ndim != 2 or not len(centers) or centers.shape[1] != self.r
                or not np.isfinite(centers).all()):
            raise ValueError("Centers must be finite nonempty q by R CCFs")
        return centers

    def assignment_admissibility(self, centers):
        """N-by-q closed-box assignments, including regions without reads."""
        centers = self._validate_centers(centers)
        return ((centers[None, :, :] >= self.lower[:, None, :])
                & (centers[None, :, :] <= self.upper[:, None, :])).all(axis=2)

    def log_likelihood_columns(self, centers):
        return self.joint_column_set(centers).materialize()

    def joint_column_set(self, centers, *, retained_columns=None):
        """Log P(reads, valid assignment | cluster); forbidden pairs are -inf.

        No division by admissible prior mass: the fixed-center weight problem
        remains the same concave mixture optimization used for all-valid inputs.
        Matrix-call telemetry includes proposal scans and conditional refits;
        likelihood_columns counts unique centers actually evaluated, including
        native cache misses on CUDA. For successful calls,
        likelihood_column_requests = likelihood_columns + column_host_cache_hits
        + column_device_cache_hits + column_device_retained_hits + column_batch_reuses.
        Host hits reuse pre-call columns, including any
        call-retained copies; batch reuse serves another request from a column
        evaluated in this call, even if the bounded host cache evicts it.
        The legacy column_cache_hits also includes hits on newly
        cached columns, so it overlaps these disjoint counters; do not add it.
        """
        centers = self._validate_centers(centers)
        self.telemetry["joint_matrix_calls"] += 1
        self.telemetry["likelihood_column_requests"] += len(centers)
        # Native CUDA owns its optional shared cache separately from active
        # leases. Python retains only call-owned handles, never pins an LRU's
        # supposedly evictable entries indefinitely.
        fresh = {}
        evaluated = set()
        retained = {center.tobytes(): self._columns[self.identity, center.tobytes()]
                    for center in centers if (self.identity, center.tobytes()) in self._columns}
        if retained_columns is not None:
            if (retained_columns.model_identity != self.identity
                    or retained_columns.numerical_identity != (self.backend.policy, self.backend.identity)):
                raise ValueError("Retained columns belong to another regional model")
            if self.backend.kind == "cuda":
                self.backend._column_handles(retained_columns.columns)
            retained.update(zip(retained_columns.center_keys, retained_columns.columns))
        missing = {center.tobytes(): center for center in centers
                   if center.tobytes() not in retained}
        if missing:
            values = np.array(list(missing.values()))
            if self.backend.kind == "cuda":
                before = self.backend.scoring_diagnostics()
                fresh = dict(zip(missing, self.backend.resident_joint_columns(values)))
                after = self.backend.scoring_diagnostics()
                evaluated_count = after["column_evaluations"] - before["column_evaluations"]
                self.telemetry["column_device_cache_hits"] += after["column_cache_hits"] - before["column_cache_hits"]
            else:
                matrix = self.backend.joint_columns(values)
                fresh = dict(zip(missing, matrix.T))
                evaluated_count = len(missing)
            self.telemetry["likelihood_columns"] += evaluated_count
        columns = []
        for center in centers:
            key = (self.identity, center.tobytes())
            column = self._columns.get(key)
            if column is None:
                column = retained.get(key[1])
            if column is not None:
                self.telemetry["column_cache_hits"] += 1
                counter = ("column_batch_reuses" if key[1] in evaluated else
                           "column_device_retained_hits" if self.backend.kind == "cuda" else
                           "column_host_cache_hits")
                self.telemetry[counter] += 1
                if key in self._columns:
                    self._columns.move_to_end(key)
            else:
                column = (fresh[key[1]] if self.backend.kind == "cuda"
                          else readonly_array(fresh[key[1]]))
                if key[1] in evaluated:
                    self.telemetry["column_batch_reuses"] += 1
                evaluated.add(key[1])
                size = (column.nbytes + len(key[1]) + _CACHE_ENTRY_BYTES
                        if self.backend.kind == "cpu" else self._cache_bytes + 1)
                if size <= self._cache_bytes:
                    while self._column_bytes + size > self._cache_bytes:
                        removed_key, removed = self._columns.popitem(last=False)
                        self._column_bytes -= removed.nbytes + len(removed_key[1]) + _CACHE_ENTRY_BYTES
                    self._columns[key] = column
                    self._column_bytes += size
            columns.append(column)
        return JointColumnSet(tuple(columns), tuple(center.tobytes() for center in centers), self.identity,
                              (self.backend.policy, self.backend.identity))

    def clear_weight_cache(self):
        """Release only completed weight fits; keep lower caches and counters."""
        self._weight_results.clear()
        self._weight_result_bytes = 0

    def _weight_result(self, columns, initial_counts, matrix=None, progress_callback=None):
        """Reuse an exact ordered problem, never a partition or device handle.

        ``matrix`` is the caller's already materialized CPU column set. Counts
        are keyed before the solver's unchanged normalization. Different starts
        can select different optima for identical columns, including zero support.
        """
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        counts = np.asarray(initial_counts)
        key = self._weight_key(columns, counts)
        cached = self._cached_weight_result(key, columns, progress_callback)
        if cached is not None:
            return cached
        observer = {} if progress_callback is None else {"progress_callback": progress_callback}
        if self.backend.kind == "cuda":
            with self.backend.mixture_problem(columns.columns) as problem:
                self.telemetry["weight_solves"] += 1
                if self._resident_weights:
                    from .resident_weights import fit_resident_weights
                    outcome, = fit_resident_weights(self.backend, [problem], [counts], telemetry=self.telemetry)
                else:
                    result = fit_mixture_weights(problem, counts, telemetry=self.telemetry, **observer)
            if self._resident_weights:
                # Coarse observation only, after the numerical lease closes;
                # never invent the reference controller's inner-loop events.
                if progress_callback is not None:
                    progress_callback(dict(operation="weight_fit", boundary="resident_complete",
                                           request="resident_fit", occupied_q=len(counts)))
                if outcome.operation == "error":
                    raise outcome.error
                result = outcome.result
        else:
            if matrix is None:
                matrix = columns.materialize()
            self.telemetry["weight_solves"] += 1
            result = fit_mixture_weights(matrix, counts, telemetry=self.telemetry, **observer)
        return self._cache_weight_result(key, columns, counts, result)

    def _weight_key(self, columns, counts):
        if (columns.model_identity != self.identity or columns.numerical_identity !=
                (self.backend.policy, self.backend.identity)):
            raise ValueError("Weight columns belong to another regional model or numerical backend")
        controller = ((WEIGHT_CONTROLLER_ID, "resident_portable_blas_v1") if self._resident_weights
                      else WEIGHT_CONTROLLER_ID)
        return (columns.model_identity, self.backend.kind, columns.numerical_identity,
                controller, columns.center_keys,
                counts.dtype.str, counts.shape, counts.tobytes())

    def _solve_weight_problems(self, problems, counts, *, max_active, state_budget_bytes,
                               telemetry=None, progress_callback=None):
        """One bounded submission route; resident mode has coarse events only."""
        from .weight_batch import controller_reservation, fit_weight_batch
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if not self._resident_weights:
            return fit_weight_batch(self.backend, problems, counts, max_active=max_active,
                state_budget_bytes=state_budget_bytes, telemetry=telemetry,
                progress_callback=progress_callback)
        from .resident_weights import fit_resident_weights
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        if type(state_budget_bytes) is not int or state_budget_bytes < 0:
            raise ValueError("state_budget_bytes must be a nonnegative integer")
        problems, counts = tuple(problems), tuple(counts)
        if len(problems) != len(counts):
            raise ValueError("Provide one initial-weight vector per problem")
        charges = [controller_reservation(problem.shape[1]) for problem in problems]
        if any(charge > state_budget_bytes for charge in charges):
            raise MemoryError("One resident weight problem exceeds the live-state budget")
        results, first = [], 0
        while first < len(problems):
            last, reserved = first, 0
            while (last < len(problems) and last-first < max_active
                   and reserved+charges[last] <= state_budget_bytes):
                admission = self.backend.resident_weight_admission(
                    [problem.shape[1] for problem in problems[first:last+1]])
                if any(admission[kind+"_required"] > admission[kind+"_available"]
                       for kind in ("device", "host")):
                    break
                reserved += charges[last]
                last += 1
            if last == first:
                # Mandatory single work may reclaim optional caches; active
                # mixture/column leases remain owned and cannot be evicted.
                self.backend.release_optional_cache()
                admission = self.backend.resident_weight_admission([problems[first].shape[1]])
                if any(admission[kind+"_required"] > admission[kind+"_available"]
                       for kind in ("device", "host")):
                    raise NativeExecutionError("One resident weight solve exceeds the remaining native memory budget")
                last = first+1
            outcomes = fit_resident_weights(self.backend, problems[first:last], counts[first:last],
                                            telemetry=telemetry)
            results.extend(outcomes)
            if progress_callback is not None:
                for index in range(first, last):
                    progress_callback(index, dict(operation="weight_fit", boundary="resident_complete",
                        request="resident_fit", occupied_q=problems[index].shape[1]))
            first = last
        return tuple(results)

    def _weight_admission(self, q, pending_sizes=()):
        """Reserve the next mixture and all concurrently live optimizer state."""
        admission = self.backend.mixture_admission(q)
        if self._resident_weights:
            resident = self.backend.resident_weight_admission((*pending_sizes, q))
            for kind in ("device", "host"):
                admission[kind+"_required"] += resident[kind+"_required"]
                admission[kind+"_available"] = min(admission[kind+"_available"],
                                                    resident[kind+"_available"])
        return admission

    def _cached_weight_result(self, key, columns, progress_callback=None):
        cached = self._weight_results.get(key)
        if cached is not None:
            if progress_callback is not None:
                progress_callback({"operation": "weight_fit", "boundary": "cache_hit",
                                   "occupied_q": len(columns.columns)})
            self._weight_results.move_to_end(key)
            self.telemetry["weight_cache_hits"] += 1
            return cached[0]
        self.telemetry["weight_cache_misses"] += 1
        return None

    def _cache_weight_result(self, key, columns, counts, result):
        # Only successful returns reach admission, after CUDA resources close.
        # The immutable payload contains no membership/conditional diagnostics.
        result = _freeze(result)
        # Charge Python owners per center as well as payload bytes: K is unbounded.
        size = (2*_CACHE_ENTRY_BYTES + sum(len(center)+64 for center in columns.center_keys)
                + counts.nbytes + result["cluster_weights"].nbytes)
        if size <= self._weight_cache_bytes:
            while self._weight_result_bytes + size > self._weight_cache_bytes:
                _, (_, removed) = self._weight_results.popitem(last=False)
                self._weight_result_bytes -= removed
                self.telemetry["weight_cache_evictions"] += 1
            self._weight_results[key] = (result, size)
            self._weight_result_bytes += size
        return result

    def refit(self, labels, initial_centers=None, *, progress_callback=None):
        labels = np.asarray(labels)
        if labels.shape != (self.n,) or not np.isfinite(labels).all() or np.any(labels != np.rint(labels)):
            raise ValueError("Labels must contain one finite integer per mutation")
        _, inverse = np.unique(labels, return_inverse=True)
        groups = tuple(self.prepare_group(np.flatnonzero(inverse == cluster))
                       for cluster in range(int(inverse.max()) + 1))
        return self._refit_groups(inverse, groups, initial_centers,
                                  membership=Membership(explicit_labels=inverse),
                                  progress_callback=progress_callback)

    def refit_prepared(self, partition, initial_centers=None, *, progress_callback=None):
        """Internal path: prepared memberships avoid all-group reconstruction."""
        if partition.model_identity != self.identity:
            raise ValueError("Prepared partition belongs to another regional model")
        return self._refit_groups(partition.labels, partition.groups, initial_centers,
                                  partition.coordinate_fits, partition.columns, partition.membership,
                                  progress_callback=progress_callback)

    def refit_batch_reservation(self, q, *, progress=False):
        """Conservative additional host state, excluding caller-owned outputs."""
        from .weight_batch import controller_reservation
        # Compact one-byte request kinds cover the fixed 2000 SLSQP iterations
        # and 10000 pair-repair/100-step Brent budgets. Capacity is checked while
        # recording; callback dictionaries are reconstructed only at publication.
        events = 2*1024**2+2*q if progress else 0
        # Each observed task retains both its int64 rows and independent key
        # bytes (16*N*R), even with the preparation cache disabled. Membership,
        # group rows, validation copies and general-partition keys need their
        # own allowance. Charge shared objects again rather than assuming reuse.
        prepared = (64+16*self.r)*self.n+(2*_CACHE_ENTRY_BYTES+256)*q*self.r
        return controller_reservation(q)+prepared+4096+events

    def refit_prepared_batch(self, requests, *, max_active=8, state_budget_bytes=8*1024**2,
                             on_result=None):
        """Optional independent refits, with original-order result publication.

        Requests are ``(partition, initial_centers, progress_callback)`` triples.
        Only the weight solves overlap. Conditional preparation and every
        completed-cache admission/result callback retain logical request order.
        Pending partial batches flush before another lease exceeds admission;
        one mandatory request retains native optional-cache eviction behavior.
        Default search dispatch does not enable this method.
        """
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active must be a positive integer")
        if type(state_budget_bytes) is not int or state_budget_bytes < 0:
            raise ValueError("state_budget_bytes must be a nonnegative integer")
        if on_result is not None and not callable(on_result):
            raise TypeError("on_result must be callable or None")
        requests = list(requests)
        results = [None]*len(requests)

        def publish(index, result):
            results[index] = result
            if on_result is not None:
                on_result(index, result)

        if self.backend.kind != "cuda" or max_active == 1:
            for index, (partition, centers, callback) in enumerate(requests):
                publish(index, self.refit_prepared(partition, centers, progress_callback=callback))
            return results
        pending, open_steps = [], []
        leases = ExitStack()
        reserved = 0

        def flush():
            nonlocal leases, reserved
            if not pending:
                return
            batch = pending[:]
            names = ("evaluate", "prepare_transfer", "transfer_derivative", "resident_fit")

            def record(index, event):
                item = batch[index]
                if item["request"][3] is not None:
                    if len(item["events"]) >= 1200000+len(item["request"][1]):
                        raise MemoryError("Ordered weight progress exceeds its reserved capacity")
                    item["events"].append(names.index(event["request"]))

            solve_started = time.perf_counter()
            try:
                outcomes = self._solve_weight_problems([item["problem"] for item in batch],
                    [item["request"][1] for item in batch], max_active=max_active,
                    state_budget_bytes=state_budget_bytes, telemetry=self.telemetry,
                    progress_callback=record if any(item["request"][3] is not None for item in batch) else None)
            finally:
                self.telemetry["weight_seconds"] += time.perf_counter()-solve_started
                # Completed cache entries must never retain live device problems.
                leases.close()
                leases = ExitStack()
            for item, outcome in zip(batch, outcomes):
                columns, counts, _, notify = item["request"]
                mixture = self._cached_weight_result(item["key"], columns, notify)
                if mixture is None:
                    if notify is not None:
                        _replay_weight_progress(item["events"], names, counts, notify)
                    if outcome.operation == "error":
                        fitted = self._finish_refit_steps(item["steps"], error=outcome.error)
                    else:
                        mixture = self._cache_weight_result(item["key"], columns, counts, outcome.result)
                        fitted = self._finish_refit_steps(item["steps"], mixture)
                else:
                    fitted = self._finish_refit_steps(item["steps"], mixture)
                item["steps"].close()
                open_steps.remove(item["steps"])
                publish(item["index"], fitted)
            pending.clear()
            reserved = 0

        try:
            for index, (partition, centers, callback) in enumerate(requests):
                if partition.model_identity != self.identity:
                    raise ValueError("Prepared partition belongs to another regional model")
                q = int(partition.labels.max())+1
                charge = self.refit_batch_reservation(q, progress=callback is not None)
                admission = self._weight_admission(q, [item["problem"].shape[1] for item in pending])
                # Assume every candidate column is new; exact reuse can only
                # reduce the physical reservation. Parent handles remain live.
                new_columns = 8*self.n*q+8*q*(self.r+1)+4
                if pending and (len(pending) >= max_active or reserved+charge > state_budget_bytes
                        or admission["device_required"]+new_columns > admission["device_available"]
                        or admission["host_required"]+64*q > admission["host_available"]):
                    flush()
                if charge > state_budget_bytes:
                    raise MemoryError("One prepared refit exceeds the live-state budget")
                steps = self._refit_steps(partition.labels, partition.groups, centers,
                    partition.coordinate_fits, partition.columns, partition.membership,
                    progress_callback=callback, record_weight_time=False)
                open_steps.append(steps)
                try:
                    request = next(steps)
                except StopIteration as complete:
                    flush()
                    open_steps.remove(steps)
                    publish(index, complete.value)
                    continue
                columns, counts, matrix, notify = request
                key = self._weight_key(columns, counts)
                if key in self._weight_results or any(item["key"] == key for item in pending):
                    flush()
                if key in self._weight_results:
                    mixture = self._weight_result(columns, counts, matrix, progress_callback=notify)
                    fitted = self._finish_refit_steps(steps, mixture)
                    steps.close()
                    open_steps.remove(steps)
                    publish(index, fitted)
                    request = columns = counts = matrix = notify = None
                    continue
                admission = self._weight_admission(q, [item["problem"].shape[1] for item in pending])
                if pending and (admission["device_required"] > admission["device_available"]
                                or admission["host_required"] > admission["host_available"]):
                    flush()
                problem = leases.enter_context(self.backend.mixture_problem(columns.columns))
                self.telemetry["weight_solves"] += 1
                pending.append({"index": index, "steps": steps, "request": request,
                                "key": key, "problem": problem, "events": bytearray()})
                # Only pending owns these leases now. Keeping loop locals alive
                # across flush() would retain the last completed column set
                # while the next conditional preparation allocates new columns.
                request = columns = counts = matrix = notify = None
                reserved += charge
                self.telemetry["refit_batch_live_peak"] = max(
                    self.telemetry.get("refit_batch_live_peak", 0), len(pending))
                self.telemetry["refit_batch_state_bytes_peak"] = max(
                    self.telemetry.get("refit_batch_state_bytes_peak", 0), reserved)
            flush()
            return results
        finally:
            leases.close()
            for steps in open_steps:
                steps.close()

    def _refit_groups(self, inverse, groups, initial_centers, prior=(), prior_columns=None,
                      membership=None, *, progress_callback=None):
        steps = self._refit_steps(inverse, groups, initial_centers, prior, prior_columns,
                                  membership, progress_callback=progress_callback)
        try:
            try:
                columns, counts, matrix, notify = next(steps)
            except StopIteration as complete:
                return complete.value
            try:
                observer = {} if notify is None else {"progress_callback": notify}
                result = self._weight_result(columns, counts, matrix, **observer)
            except WeightConvergenceError as error:
                return self._finish_refit_steps(steps, error=error)
            return self._finish_refit_steps(steps, result)
        finally:
            steps.close()

    @staticmethod
    def _finish_refit_steps(steps, result=None, *, error=None):
        try:
            steps.send(result) if error is None else steps.throw(error)
        except StopIteration as complete:
            return complete.value
        raise RuntimeError("Conditional refit yielded more than one weight problem")

    def _refit_steps(self, inverse, groups, initial_centers, prior=(), prior_columns=None,
                     membership=None, *, progress_callback=None, record_weight_time=True):
        """One shared refit path, suspended only at its independent weight solve."""
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        callback_error = None

        def notify(operation):
            nonlocal callback_error
            try:
                progress_callback(operation)
            except BaseException as error:
                # A user callback can raise the same class as a scientific
                # rejection; preserve its identity instead of rejecting a fit.
                callback_error = error
                raise

        q = int(inverse.max())+1
        centers = np.full((q, self.r), np.nan)
        if initial_centers is not None:
            initial_centers = np.asarray(initial_centers, dtype=float)
            if initial_centers.shape != centers.shape:
                raise ValueError("Initial centers must follow sorted occupied label order")
        penalty = float((q*self.r+q-1)*math.log(self.n))
        try:
            tasks = [(group, region) for group in groups for region in range(self.r)]
            fitted, retained_initial, retained_flags = self._fit_coordinates(
                tasks, None if initial_centers is None else initial_centers.ravel(),
                None if not prior else [value for group in prior for value in
                                        ([None] * self.r if group is None else group)],
                return_details=True)
            centers[:] = np.array([center for center, _ in fitted]).reshape(q, self.r)
        except ScientificCandidateRejection as error:
            return FitResult(membership, centers, np.full(q, np.nan), -np.inf,
                                     -np.inf, penalty, np.inf, False, str(error),
                                     diagnostics={"rejection_kind": "scientific"})
        columns = self.joint_column_set(centers, retained_columns=prior_columns)
        matrix = columns.materialize() if self.backend.kind == "cpu" else None
        conditional = math.fsum(matrix[np.arange(self.n), inverse] if matrix is not None
                                else columns.assigned_values(inverse))
        start = time.perf_counter()
        try:
            mixture = yield columns, np.bincount(inverse), matrix, (None if progress_callback is None else notify)
        except WeightConvergenceError as error:
            if error is callback_error:
                raise
            return FitResult(membership, centers, np.full(q, np.nan), conditional,
                                     -np.inf, penalty, np.inf, False, f"weight_fit:{error}",
                                     diagnostics={"rejection_kind": "weight_convergence"})
        finally:
            if record_weight_time:
                self.telemetry["weight_seconds"] += time.perf_counter()-start
        ll = mixture["log_likelihood"]
        eligible = bool(np.all(mixture["cluster_weights"] > 0))
        return FitResult(membership, centers, mixture["cluster_weights"], conditional,
                                 ll, penalty, -2*ll+penalty, eligible,
                                 None if eligible else "zero_mixture_weight", diagnostics={
                                     "weight_optimality_gap": mixture["weight_optimality_gap"],
                                     "weight_active_score_gap": mixture["weight_active_score_gap"],
                                     "retained_initial_centers": retained_initial,
                                     "scalar_global_optimality_certified": False,
                                     "num_parameters": q*self.r+q-1,
                                     "score_definition": SCORE_DEFINITION},
                                 coordinate_fits=np.column_stack((fitted, retained_flags)).reshape(q, self.r, 3))
