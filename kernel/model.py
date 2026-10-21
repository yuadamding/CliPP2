"""CliPP2 emissions, conditional centers and joint reads/validity mixture score.

CCF, not cellular prevalence, is the fitting/geometry scale for every region.
One shared emission primitive serves arrays, pilot grids and coordinate views.
Mixed CN explicitly retains CliPP2's capped support, weighted denominator,
clipped likelihood and original interior box. Assignment validity is observed
jointly with reads; it is not conditioned away by renormalizing mixture weights.
CUDA is an explicit float64 likelihood/host-forest path, never a CPU fallback;
initialization emissions are batched on device. Conditional-center decisions
retain the reference NumPy/SciPy arithmetic on every device: roundoff-equivalent
conditional optima can have different joint-mixture scores.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
from scipy.special import gammaln

from . import SCORE_DEFINITION
from .initialization import pooled_ccf_initialization
from .likelihood import MIXED_EPS, RegionalView, emission_terms
from .refitting import refit_center
from .scoring import fit_mixture_weights
from ..io.data import readonly_array


# Mixed-CN emissions/boxes stay fixed; their assignment mask now enters scoring.
MODEL_VERSION = "clipp2_regional_full_support__general_normal_v1__mixed_cn_bulk_assignment_event_v2"


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
        m = np.arange(1, int(self.support.max()) + 1)
        self.valid = readonly_array(m <= self.support[..., None])
        self.slope = readonly_array(self.scaling[..., None] * m)
        self.constants = readonly_array(
            gammaln(self.depth + 1) - gammaln(self.alt + 1)
            - gammaln(self.depth - self.alt + 1) - np.log(self.support))
        digest = hashlib.sha256(MODEL_VERSION.encode())
        for values in (self.alt, self.depth, self.observed, self.slope,
                       self.support, self.mixed, self.lower, self.upper):
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
        if self.device.startswith("cuda"):
            from .device import DeviceWorkspace
            self.workspace = DeviceWorkspace(self, device_cache_bytes=device_cache_bytes,
                                             workspace_bytes=workspace_bytes)
        self.execution_backend = "cuda_likelihood_host_forest" if self.workspace else "cpu"
        self._cache_bytes = int(cache_bytes)
        if self._cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self._refits, self._columns = OrderedDict(), OrderedDict()
        self._refit_bytes = self._column_bytes = 0
        self.telemetry = {"refit_cache_hits": 0, "scalar_refits": 0,
                          "column_cache_hits": 0, "likelihood_columns": 0,
                          "joint_matrix_calls": 0,
                          "scalar_refit_seconds": 0.0, "weight_seconds": 0.0}

    def __len__(self):
        return self.n

    def execution_diagnostics(self):
        """Execution-only accounting; never included in model or candidate identity."""
        return {"backend": self.execution_backend, "grid_cache_bytes": self.grid_cache_bytes,
                "conditional_refit_backend": "cpu_reference",
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
        return np.where(self.observed, -self._numpy_kernel(x)[-1], 0.)

    def node_terms(self, x):
        """Loss, exact gradient, positive complete-data curvature on CCF scale."""
        if self.workspace:
            return self._gpu_terms(x, True)
        p, mass, kernel, ll = self._numpy_kernel(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            post = np.where(np.isfinite(kernel), np.exp(kernel-ll[..., None]), 0.)
            slope = np.where(self.mixed[..., None] & ((mass <= MIXED_EPS) | (mass >= 1-MIXED_EPS)), 0., self.slope)
            alt, ref = self.alt[..., None], (self.depth-self.alt)[..., None]
            g = slope * (np.divide(-alt, p, out=np.zeros_like(p), where=alt != 0)
                         + np.divide(ref, 1-p, out=np.zeros_like(p), where=ref != 0))
            h = slope**2 * (np.divide(alt, p*p, out=np.zeros_like(p), where=alt != 0)
                            + np.divide(ref, (1-p)**2, out=np.zeros_like(p), where=ref != 0))
            g = np.where(post > 0, post*g, 0.).sum(-1)
            h = np.where(post > 0, post*h, 0.).sum(-1)
            boundary = self.valid & ~self.mixed[..., None] & (p == 1) & (ref == 1)
            g += np.where(boundary, self.slope*np.exp(self.constants[..., None]-ll[..., None]), 0.).sum(-1)
            h += np.where(self.valid & ~self.mixed[..., None] & (p == 1) & (ref == 2),
                          2*self.slope**2*np.exp(self.constants[..., None]-ll[..., None]), 0.).sum(-1)
        h = np.where(boundary.any(-1), np.inf, np.maximum(h, 1e-8))
        g, h = np.where(np.isfinite(ll), g, 0.), np.where(np.isfinite(ll), h, 1.)
        return tuple(np.where(self.observed, v, 0.) for v in (-ll, g, h))

    def posterior(self, x):
        _, _, kernel, ll = self._numpy_kernel(x)
        if not np.isfinite(ll[self.observed]).all():
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.where(self.observed[..., None], np.exp(kernel-ll[..., None]), np.nan)

    def multiplicity(self, x):
        post = self.posterior(x)
        return np.where(self.observed, np.nan_to_num(post).argmax(-1)+1., np.nan)

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
        arrays = (self.observed, self.alt, self.depth, self.slope, self.support,
                  self.mixed, self.lower, self.upper)
        return [int.from_bytes(hashlib.sha256(b''.join(a[i].tobytes() for a in arrays)).digest(),
                               'big') for i in range(self.n)]

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
        key = (self.identity, region, tuple(map(int, members)))
        if key in self._refits:
            self.telemetry["refit_cache_hits"] += 1
            self._refits.move_to_end(key)
            return self._refits[key][0]
        start = time.perf_counter()
        rows = members[self.observed[members, region]]
        lower, upper = self.lower[members, region].max(), self.upper[members, region].min()
        if not len(rows):
            raise ValueError(f"unsupported_cluster_region:{self.region_ids[region]}")
        if lower > upper:
            raise ValueError(f"empty_cluster_region_box:{self.region_ids[region]}")
        fitted = refit_center(self.view(region, rows), (lower, upper))
        if not math.isfinite(fitted[1]):
            raise RuntimeError("No finite conditional scalar likelihood")
        self.telemetry["scalar_refits"] += 1
        self.telemetry["scalar_refit_seconds"] += time.perf_counter()-start
        # Membership tuples are part of the bounded cache budget, not only values.
        size = 256 + 36*len(members)
        if size <= self._cache_bytes:
            while self._refit_bytes + size > self._cache_bytes:
                _, (_, removed) = self._refits.popitem(last=False)
                self._refit_bytes -= removed
            self._refits[key] = (fitted, size)
            self._refit_bytes += size
        return fitted

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
        emission-column computations and cache hits are counted separately.
        """
        centers = np.asarray(centers, dtype=float)
        allowed = self.assignment_admissibility(centers)
        self.telemetry["joint_matrix_calls"] += 1
        # Compute uncached centers together. Public return values remain NumPy;
        # device consumers can retain workspace.joint_columns_device directly.
        fresh, retained = {}, {}
        if self.workspace:
            retained = {center.tobytes(): self._columns[self.identity, center.tobytes()]
                        for center in centers if (self.identity, center.tobytes()) in self._columns}
            missing = {center.tobytes(): center for center in centers
                       if center.tobytes() not in retained}
            if missing:
                matrix = self.workspace.to_host(self.workspace.joint_columns_device(
                    np.array(list(missing.values()))))
                fresh = dict(zip(missing, matrix.T))
        columns = []
        for k, center in enumerate(centers):
            key = (self.identity, center.tobytes())
            column = self._columns.get(key)
            if column is None:
                column = retained.get(key[1])
            if column is not None:
                self.telemetry["column_cache_hits"] += 1
                if key in self._columns:
                    self._columns.move_to_end(key)
            else:
                if self.workspace:
                    column = readonly_array(fresh[key[1]])
                else:
                    emission = -self.loss(np.broadcast_to(center, (self.n, self.r))).sum(axis=1)
                    column = readonly_array(np.where(allowed[:, k], emission, -np.inf))
                self.telemetry["likelihood_columns"] += 1
                size = column.nbytes + len(key[1]) + 128
                if size <= self._cache_bytes:
                    while self._column_bytes + size > self._cache_bytes:
                        removed_key, removed = self._columns.popitem(last=False)
                        self._column_bytes -= removed.nbytes + len(removed_key[1]) + 128
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
            for cluster in range(q):
                members = memberships[cluster]
                for region in range(self.r):
                    # Grid, Brent and incumbent tie decisions use one arithmetic.
                    center, ll = self._fit_scalar(members, region)
                    if initial_centers is not None:
                        old = initial_centers[cluster, region]
                        if (np.isfinite(old) and self.lower[members, region].max() <= old
                                <= self.upper[members, region].min()):
                            rows = members[self.observed[members, region]]
                            old_ll = float(self.view(region, rows).log_likelihood_columns([old]).T.sum())
                            if old_ll > ll:
                                center, ll = float(old), old_ll
                                retained_initial += 1
                    centers[cluster, region] = center
        except (ValueError, RuntimeError) as error:
            return RefittedPartition(inverse, centers, np.full(q, np.nan), -np.inf,
                                     -np.inf, penalty, np.inf, False, str(error))
        matrix = self.log_likelihood_columns(centers)
        conditional = math.fsum(matrix[np.arange(self.n), inverse])
        start = time.perf_counter()
        try:
            mixture = fit_mixture_weights(matrix, np.bincount(inverse),
                                          telemetry=self.telemetry)
        except (ValueError, RuntimeError) as error:
            return RefittedPartition(inverse, centers, np.full(q, np.nan), conditional,
                                     -np.inf, penalty, np.inf, False, f"weight_fit:{error}")
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
