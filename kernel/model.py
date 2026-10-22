"""CliPP2 emissions, conditional centers and joint reads/validity mixture score.

CCF, not cellular prevalence, is the fitting/geometry scale for every region.
One shared emission primitive serves arrays, pilot grids and coordinate views.
Mixed CN explicitly retains CliPP2's capped support, weighted denominator,
clipped likelihood and original interior box. Assignment validity is observed
jointly with reads; it is not conditioned away by renormalizing mixture weights.
CUDA is an explicit float64 likelihood/host-forest path, never a CPU fallback;
emissions and conditional centers share a native float64 numerical policy.
Forest continuation, candidate selection and mixture-weight fitting stay on host.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
from scipy.special import gammaln

from . import SCORE_DEFINITION
from .backend import CoordinateTask, NativeBackend, NativeExecutionError, ScientificCandidateRejection
from .initialization import pooled_ccf_initialization
from .likelihood import MIXED_EPS, RegionalView, emission_terms
from .scoring import WeightConvergenceError, fit_mixture_weights
from ..io.data import readonly_array


# Mixed-CN emissions/boxes stay fixed; their assignment mask now enters scoring.
MODEL_VERSION = "clipp2_regional_full_support__general_normal_v1__mixed_cn_bulk_assignment_event_v2"
# Conservative retained Python-object/dictionary reservations, separate from
# array/key payload bytes. Shared immutable owners may be charged more than once.
_CACHE_ENTRY_BYTES = 1280
_MODE_ENTRY_BYTES = 768


@dataclass(frozen=True)
class RefittedPartition:
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    score: float
    eligible: bool
    reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("labels", "centers", "weights"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))

    @property
    def penalty(self):
        return self.complexity_penalty


class RegionalModel:
    """Immutable, canonically ordered informative mutation vectors."""

    def __init__(self, data, device="cpu", *, cache_bytes=32 * 1024 * 1024,
                 grid_cache_bytes=None, device_cache_bytes=256 * 1024**2,
                 workspace_bytes=64 * 1024**2):
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
        self.workspace = None
        self.grid_cache_bytes = grid_cache_bytes
        if grid_cache_bytes is not None and (type(grid_cache_bytes) is not int or grid_cache_bytes < 0):
            raise ValueError("grid_cache_bytes must be a nonnegative integer")
        if type(device_cache_bytes) is not int or device_cache_bytes < 0:
            raise ValueError("device_cache_bytes must be a nonnegative integer")
        if type(workspace_bytes) is not int or workspace_bytes < 0:
            raise ValueError("workspace_bytes must be a nonnegative integer")
        self.backend = NativeBackend(self, workspace_bytes)
        self.execution_backend = f"native_{self.backend.kind}_likelihood_host_forest"
        self._cache_bytes = int(cache_bytes)
        if self._cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self._refits, self._columns = OrderedDict(), OrderedDict()
        self._refit_bytes = self._column_bytes = 0
        # Group metadata, exact component modes and coordinate grids share one
        # bounded host preparation cache. Active caller-owned tasks are scratch.
        self._preparations, self._preparation_bytes = OrderedDict(), 0
        self.telemetry = {"refit_cache_hits": 0, "scalar_refits": 0,
                          "column_cache_hits": 0, "likelihood_columns": 0,
                          "likelihood_column_requests": 0,
                          "column_host_cache_hits": 0, "column_batch_reuses": 0,
                          "joint_matrix_calls": 0,
                          "group_cache_hits": 0, "coordinate_grid_cache_hits": 0,
                          "component_mode_cache_hits": 0, "coordinate_grids": 0,
                          "incumbent_equal_base_reuses": 0, "fused_incumbents": 0,
                          "scalar_refit_seconds": 0.0, "weight_seconds": 0.0}

    def __len__(self):
        return self.n

    @property
    def slope(self):
        """On-demand padded reference view; never retained by production fits."""
        return self.scaling[..., None] * np.arange(1, int(self.support.max())+1)

    @property
    def valid(self):
        return np.arange(1, int(self.support.max())+1) <= self.support[..., None]

    def execution_diagnostics(self):
        """Execution-only accounting; never included in model or candidate identity."""
        return {"backend": self.execution_backend, "grid_cache_bytes": self.grid_cache_bytes,
                "conditional_refit_backend": f"native_{self.backend.kind}",
                "native": self.backend.diagnostics(),
                "continuation": getattr(self, "_continuation_execution", None),
                "host_cache_bytes": {"preparation": self._preparation_bytes,
                                     "refit": self._refit_bytes, "column": self._column_bytes,
                                     "limit_per_category": self._cache_bytes},
                "device": dict(self.workspace.telemetry) if self.workspace else None}

    def _numpy_kernel(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.r) or np.any(~np.isfinite(x[self.observed])):
            raise ValueError("CCFs must match N by R and be finite where observed")
        if np.any(x[self.observed] < 0) or np.any(x[self.observed] > 1):
            raise ValueError("CCFs must be in [0,1]")
        return emission_terms(np.where(self.observed, x, 0.), **{
            name: getattr(self, name) for name in
            ("slope", "valid", "mixed", "alt", "depth", "constants")})

    def _gpu_terms(self, x, derivatives=False):
        result = self.workspace.node_terms_device(x, derivatives=derivatives)
        if derivatives:
            return tuple(self.workspace.to_host(value) for value in result)
        return self.workspace.to_host(result)

    def loss(self, x):
        if self.workspace:
            return self._gpu_terms(x)
        return self.backend.node_terms(x, False)

    def node_terms(self, x):
        """Loss, exact gradient, positive complete-data curvature on CCF scale."""
        if self.workspace:
            return self._gpu_terms(x, True)
        return self.backend.node_terms(x)

    def posterior(self, x):
        _, _, kernel, ll = self._numpy_kernel(x)
        if not np.isfinite(ll[self.observed]).all():
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.where(self.observed[..., None], np.exp(kernel-ll[..., None]), np.nan)

    def multiplicity(self, x):
        return self.backend.multiplicity(x)

    def view(self, region, rows=None):
        return RegionalView(self, region, np.arange(self.n) if rows is None else rows)

    def _regional_likelihood(self, values, rows, region):
        """Grid view of the same emissions used by the joint model."""
        if self.workspace:
            return self.workspace.to_host(self.workspace.regional_grid_device(
                region, rows, np.atleast_1d(values))).T
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

    def _coordinate_task(self, members, region):
        members = self.backend._rows(region, members)
        if not len(members):
            raise ValueError("A coordinate task requires nonempty membership")
        group_key = ("group", members.tobytes())

        def prepare_group():
            tasks = []
            size = _CACHE_ENTRY_BYTES + len(group_key[1])
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
            return tuple(tasks), size

        task = self._prepared(group_key, prepare_group, "group_cache_hits")[region]
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

    def _fit_coordinates(self, tasks, incumbents=None):
        tasks = [self._coordinate_task(*task) for task in tasks]
        initial = np.full(len(tasks), np.nan) if incumbents is None else np.asarray(incumbents, dtype=float)
        if initial.shape != (len(tasks),):
            raise ValueError("Refitting needs one incumbent per coordinate task")
        results, missing = [None]*len(tasks), []
        hits, retained_initial = [], 0
        for index, task in enumerate(tasks):
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
            fitted = (self.backend.refit(self, pending) if incumbents is None
                      else self.backend.refit(self, pending, starts))
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
                retained_initial += int(value.log_likelihood > value.base_log_likelihood)
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
                    retained_initial += 1
        return results, retained_initial

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

    def assignment_admissibility(self, centers):
        """N-by-q closed-box assignments, including regions without reads."""
        centers = np.asarray(centers, dtype=float)
        if (centers.ndim != 2 or not len(centers) or centers.shape[1] != self.r
                or not np.isfinite(centers).all()):
            raise ValueError("Centers must be finite nonempty q by R CCFs")
        return ((centers[None, :, :] >= self.lower[:, None, :])
                & (centers[None, :, :] <= self.upper[:, None, :])).all(axis=2)

    def log_likelihood_columns(self, centers):
        """Log P(reads, valid assignment | cluster); forbidden pairs are -inf.

        No division by admissible prior mass: the fixed-center weight problem
        remains the same concave mixture optimization used for all-valid inputs.
        Matrix-call telemetry includes proposal scans and conditional refits;
        likelihood_columns counts unique missing centers evaluated in a batch
        on either CPU or CUDA. For successful calls,
        likelihood_column_requests = likelihood_columns + column_host_cache_hits
        + column_batch_reuses. Host hits reuse pre-call columns, including any
        call-retained copies; batch reuse serves another request from a column
        evaluated in this call, even if the bounded host cache evicts it.
        The legacy column_cache_hits also includes hits on newly
        cached columns, so it overlaps these disjoint counters; do not add it.
        """
        centers = np.asarray(centers, dtype=float)
        self.assignment_admissibility(centers)
        self.telemetry["joint_matrix_calls"] += 1
        self.telemetry["likelihood_column_requests"] += len(centers)
        # Compute uncached centers together. Public return values remain NumPy;
        # device consumers can retain workspace.joint_columns_device directly.
        fresh = {}
        evaluated = set()
        retained = {center.tobytes(): self._columns[self.identity, center.tobytes()]
                    for center in centers if (self.identity, center.tobytes()) in self._columns}
        missing = {center.tobytes(): center for center in centers
                   if center.tobytes() not in retained}
        if missing:
            values = np.array(list(missing.values()))
            matrix = (self.workspace.to_host(self.workspace.joint_columns_device(values))
                      if self.workspace else self.backend.joint_columns(values))
            fresh = dict(zip(missing, matrix.T))
            self.telemetry["likelihood_columns"] += len(missing)
        columns = []
        for center in centers:
            key = (self.identity, center.tobytes())
            column = self._columns.get(key)
            if column is None:
                column = retained.get(key[1])
            if column is not None:
                self.telemetry["column_cache_hits"] += 1
                counter = "column_batch_reuses" if key[1] in evaluated else "column_host_cache_hits"
                self.telemetry[counter] += 1
                if key in self._columns:
                    self._columns.move_to_end(key)
            else:
                column = readonly_array(fresh[key[1]])
                if key[1] in evaluated:
                    self.telemetry["column_batch_reuses"] += 1
                evaluated.add(key[1])
                size = column.nbytes + len(key[1]) + _CACHE_ENTRY_BYTES
                if size <= self._cache_bytes:
                    while self._column_bytes + size > self._cache_bytes:
                        removed_key, removed = self._columns.popitem(last=False)
                        self._column_bytes -= removed.nbytes + len(removed_key[1]) + _CACHE_ENTRY_BYTES
                    self._columns[key] = column
                    self._column_bytes += size
            columns.append(column)
        return np.column_stack(columns)

    def refit(self, labels, initial_centers=None):
        labels = np.asarray(labels)
        if labels.shape != (self.n,) or not np.isfinite(labels).all() or np.any(labels != np.rint(labels)):
            raise ValueError("Labels must contain one finite integer per mutation")
        _, inverse = np.unique(labels, return_inverse=True)
        q = int(inverse.max())+1
        centers = np.full((q, self.r), np.nan)
        if initial_centers is not None:
            initial_centers = np.asarray(initial_centers, dtype=float)
            if initial_centers.shape != centers.shape:
                raise ValueError("Initial centers must follow sorted occupied label order")
        penalty = float((q*self.r+q-1)*math.log(self.n))
        retained_initial = 0
        try:
            memberships = [np.flatnonzero(inverse == cluster) for cluster in range(q)]
            tasks = [(members, region) for members in memberships for region in range(self.r)]
            fitted, retained_initial = self._fit_coordinates(
                tasks, None if initial_centers is None else initial_centers.ravel())
            centers[:] = np.array([center for center, _ in fitted]).reshape(q, self.r)
        except ScientificCandidateRejection as error:
            return RefittedPartition(inverse, centers, np.full(q, np.nan), -np.inf,
                                     -np.inf, penalty, np.inf, False, str(error),
                                     diagnostics={"rejection_kind": "scientific"})
        matrix = self.log_likelihood_columns(centers)
        conditional = math.fsum(matrix[np.arange(self.n), inverse])
        start = time.perf_counter()
        try:
            mixture = fit_mixture_weights(matrix, np.bincount(inverse),
                                          telemetry=self.telemetry)
        except WeightConvergenceError as error:
            return RefittedPartition(inverse, centers, np.full(q, np.nan), conditional,
                                     -np.inf, penalty, np.inf, False, f"weight_fit:{error}",
                                     diagnostics={"rejection_kind": "weight_convergence"})
        finally:
            self.telemetry["weight_seconds"] += time.perf_counter()-start
        ll = mixture["log_likelihood"]
        eligible = bool(np.all(mixture["cluster_weights"] > 0))
        return RefittedPartition(inverse, centers, mixture["cluster_weights"], conditional,
                                 ll, penalty, -2*ll+penalty, eligible,
                                 None if eligible else "zero_mixture_weight", diagnostics={
                                     "weight_optimality_gap": mixture["weight_optimality_gap"],
                                     "weight_active_score_gap": mixture["weight_active_score_gap"],
                                     "retained_initial_centers": retained_initial,
                                     "scalar_global_optimality_certified": False,
                                     "num_parameters": q*self.r+q-1,
                                     "score_definition": SCORE_DEFINITION})
