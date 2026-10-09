"""CliPP2 emissions, conditional centers and joint reads/validity mixture score.

CCF, not cellular prevalence, is the public fitting/geometry scale. Single-state
diploid-normal observations use the shared scalar likelihood primitives.
Mixed CN explicitly retains CliPP2's capped support, weighted denominator,
clipped likelihood and original interior box. Assignment validity is observed
jointly with reads; it is not conditioned away by renormalizing mixture weights.
CUDA is an explicit float64 likelihood/host-forest path, never a CPU fallback;
pooled initialization, scalar refits and weight optimization run on the host.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

from . import SCORE_DEFINITION
from .initialization import pooled_cp_initialization
from .likelihood import MultiplicityModel
from .refitting import refit_center
from .scoring import fit_cluster_weights
from ..io.data import readonly_array


# Mixed-CN emissions/boxes stay fixed; their assignment mask now enters scoring.
MODEL_VERSION = "clipp2_regional_full_support__general_normal_v1__mixed_cn_bulk_assignment_event_v2"
MIXED_EPS = 1e-6
ADAPTER_BLOCK_BYTES = 8 * 1024 * 1024
ADAPTER_MAX_ITERATIONS = 500
ADAPTER_GRID_TOLERANCE = 1e-8


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

    def __init__(self, data, device="cpu", *, cache_bytes=32 * 1024 * 1024):
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
        self._torch = None
        if self.device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
            self._torch = torch
            self._gpu = {name: torch.tensor(np.array(getattr(self, name)), device=self.device)
                         for name in ("slope", "valid", "observed", "mixed", "alt", "depth", "constants")}
        self.execution_backend = "cuda_likelihood_host_forest" if self._torch else "cpu"
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

    def _numpy_kernel(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.r) or np.any(~np.isfinite(x[self.observed])):
            raise ValueError("CCFs must match N by R and be finite where observed")
        if np.any(x[self.observed] < 0) or np.any(x[self.observed] > 1):
            raise ValueError("CCFs must be in [0,1]")
        mass = np.where(self.observed, x, 0.0)[..., None] * self.slope
        p = np.where(self.mixed[..., None], np.clip(mass, MIXED_EPS, 1-MIXED_EPS), np.minimum(1., mass))
        if np.any(p[self.valid & self.observed[..., None]] < 0):
            raise ValueError("CCFs cannot imply negative allele probabilities")
        alt = np.where(self.observed, self.alt, 0.)[..., None]
        ref = np.where(self.observed, self.depth - self.alt, 0.)[..., None]
        kernel = xlogy(alt, p) + xlog1py(ref, -p) + self.constants[..., None]
        kernel = np.where(self.valid, kernel, -np.inf)
        return p, mass, kernel, logsumexp(kernel, axis=-1)

    def _gpu_terms(self, x, derivatives=False):
        t, a = self._torch, self._gpu
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.r) or np.any(~np.isfinite(x[self.observed])):
            raise ValueError("CCFs must match N by R and be finite where observed")
        if np.any(x[self.observed] < 0) or np.any(x[self.observed] > 1):
            raise ValueError("CCFs must be in [0,1]")
        xx = t.tensor(np.where(self.observed, x, 0.), device=self.device)
        mass = xx[..., None] * a["slope"]
        p = t.where(a["mixed"][..., None], mass.clamp(MIXED_EPS, 1-MIXED_EPS), mass.clamp(max=1.))
        alt = t.where(a["observed"], a["alt"], 0.)[..., None]
        ref = t.where(a["observed"], a["depth"]-a["alt"], 0.)[..., None]
        kernel = t.special.xlogy(alt, p) + t.special.xlog1py(ref, -p) + a["constants"][..., None]
        kernel = t.where(a["valid"], kernel, -t.inf)
        ll = t.logsumexp(kernel, -1)
        loss = t.where(a["observed"], -ll, 0.)
        if not derivatives:
            return loss.cpu().numpy()
        post = t.where(t.isfinite(kernel), t.exp(kernel-ll[..., None]), 0.)
        derivative_slope = t.where(a["mixed"][..., None] & ((mass <= MIXED_EPS) | (mass >= 1-MIXED_EPS)), 0., a["slope"])
        safe_p = t.where(alt > 0, p, 1.)
        safe_q = t.where(ref > 0, 1-p, 1.)
        g = derivative_slope * (-alt/safe_p + ref/safe_q)
        h = derivative_slope.square() * (alt/safe_p.square() + ref/safe_q.square())
        g = t.where(post > 0, post*g, 0.).sum(-1)
        h = t.where(post > 0, post*h, 0.).sum(-1)
        boundary = a["valid"] & ~a["mixed"][..., None] & (p == 1) & (ref == 1)
        correction = t.where(boundary, a["slope"] * t.exp(a["constants"][..., None] - ll[..., None]), 0.).sum(-1)
        g += correction
        h += t.where(a["valid"] & ~a["mixed"][..., None] & (p == 1) & (ref == 2),
                     2*a["slope"].square()*t.exp(a["constants"][..., None]-ll[..., None]), 0.).sum(-1)
        h = t.where(boundary.any(-1), t.inf, h.clamp(min=1e-8))
        g = t.where(t.isfinite(ll), g, 0.)
        h = t.where(t.isfinite(ll), h, 1.)
        return tuple(t.where(a["observed"], v, 0.).cpu().numpy() for v in (loss, g, h))

    def loss(self, x):
        if self._torch:
            return self._gpu_terms(x)
        return np.where(self.observed, -self._numpy_kernel(x)[-1], 0.)

    def node_terms(self, x):
        """Loss, exact gradient, positive complete-data curvature on CCF scale."""
        if self._torch:
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

    def _scalar_model(self, rows, region):
        if self.mixed[rows, region].any() or np.any(self.normal[rows, region] != 2):
            return None
        return MultiplicityModel(self.alt[rows, region], self.depth[rows, region],
                                 self.major[rows, region], self.total[rows, region],
                                 self.region_purities[region])

    def _regional_likelihood(self, values, rows, region):
        """Host scalar/grid evaluation, including legacy mixed-CN clipping."""
        values = np.atleast_1d(values)
        p = values[:, None, None] * self.slope[rows, region][None, :, :]
        p = np.where(self.mixed[rows, region][None, :, None],
                     np.clip(p, MIXED_EPS, 1-MIXED_EPS), np.minimum(p, 1.))
        kernel = (xlogy(self.alt[rows, region][None, :, None], p)
                  + xlog1py((self.depth[rows, region]-self.alt[rows, region])[None, :, None], -p)
                  + self.constants[rows, region][None, :, None])
        return logsumexp(np.where(self.valid[rows, region][None, :, :], kernel, -np.inf), -1)

    def pilots(self):
        pilots, diagnostics = np.full((self.n, self.r), np.nan), {}
        for region in range(self.r):
            rows = np.flatnonzero(self.observed[:, region])
            scalar = self._scalar_model(rows, region)
            if scalar is not None:
                cp, diag = pooled_cp_initialization(scalar)
                pilots[rows, region] = cp / scalar.purity
            else:
                pilots[rows, region], diag = _pooled_adapter(self, rows, region)
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
        scalar = self._scalar_model(rows, region)
        if scalar is not None and lower == 0 and upper == 1:
            cp, ll = refit_center(scalar)
            fitted = (cp/scalar.purity, ll)
        else:
            modes = (self.alt[rows, region]/self.depth[rows, region])[:, None] / self.slope[rows, region]
            grid = np.unique(np.r_[np.linspace(lower, upper, 513), np.clip(modes[self.valid[rows, region]], lower, upper)])
            values = np.concatenate([self._regional_likelihood(grid[i:i+32], rows, region).sum(axis=1)
                                     for i in range(0, len(grid), 32)])
            best_index = int(np.argmax(values))
            best = (float(values[best_index]), float(grid[best_index]))
            maxima = np.flatnonzero((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:])
                                    & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:])))+1
            intervals = set() if len(grid) == 1 else {(grid[0], grid[1]), (grid[-2], grid[-1])}
            intervals.update((grid[j-1], grid[j+1]) for j in maxima)
            for lo, hi in sorted(intervals):
                result = minimize_scalar(lambda x: -self._regional_likelihood([x], rows, region).sum(),
                                         bounds=(lo, hi), method="bounded",
                                         options={"xatol": 1e-12, "maxiter": 500})
                if not result.success or not np.isfinite(result.fun):
                    raise RuntimeError("Regional scalar refit failed")
                candidate = (-float(result.fun), float(result.x))
                if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                    best = candidate
            fitted = (best[1], best[0])
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
        columns = []
        for k, center in enumerate(centers):
            key = (self.identity, center.tobytes())
            column = self._columns.get(key)
            if column is not None:
                self.telemetry["column_cache_hits"] += 1
                self._columns.move_to_end(key)
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
            for cluster in range(q):
                members = np.flatnonzero(inverse == cluster)
                for region in range(self.r):
                    center, ll = self._fit_scalar(members, region)
                    if initial_centers is not None:
                        old = initial_centers[cluster, region]
                        if (np.isfinite(old) and self.lower[members, region].max() <= old
                                <= self.upper[members, region].min()):
                            rows = members[self.observed[members, region]]
                            old_ll = float(self._regional_likelihood([old], rows, region).sum())
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
            mixture = fit_cluster_weights(_WeightModel(self.n), np.zeros(q),
                                          np.bincount(inverse), log_kernel=matrix,
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


class _WeightModel(SimpleNamespace):
    """Only cardinality/domain are used when inherited scorer gets log_kernel."""
    def __init__(self, n):
        super().__init__(n=n, purity=1.)

    def __len__(self):
        return self.n


class _GridOracle:
    """Exact regional grid likelihood in row blocks; never retain N by grid.

    The working block is at most ADAPTER_BLOCK_BYTES (or one grid row if that
    row itself exceeds the budget). Thus storage remains O(N + grid), even
    when distinct mixed-CN upper bounds make the grid grow with N.
    """
    def __init__(self, model, rows, region, grid, *, block_bytes=None):
        self.model, self.rows, self.region, self.grid = model, rows, region, grid
        budget = ADAPTER_BLOCK_BYTES if block_bytes is None else int(block_bytes)
        if budget < 8:
            raise ValueError("Grid block budget must hold at least one float64")
        self.block_rows = min(256, max(1, budget//(8*len(grid))),
                              max(1, 1_000_000//(32*model.valid.shape[-1])))
        self.offset = np.empty(len(rows))
        self.evaluations = 0
        self.peak_block_bytes = 0
        for first in range(0, len(rows), self.block_rows):
            stop = min(len(rows), first+self.block_rows)
            values = self.log_block(first, stop)
            self.offset[first:stop] = values.max(axis=1)
        if not np.isfinite(self.offset).all():
            raise RuntimeError("No finite supported grid likelihood for regional initialization")

    def log_block(self, first, stop, positions=None):
        grid = self.grid if positions is None else self.grid[positions]
        rows = self.rows[first:stop]
        values = np.empty((len(rows), len(grid)))
        self.peak_block_bytes = max(self.peak_block_bytes, values.nbytes)
        for begin in range(0, len(grid), 32):
            values[:, begin:begin+32] = self.model._regional_likelihood(
                grid[begin:begin+32], rows, self.region).T
        values[(grid[None, :] < self.model.lower[rows, self.region, None])
               | (grid[None, :] > self.model.upper[rows, self.region, None])] = -np.inf
        return values

    def expectation(self, weights, *, collect_pilot=False):
        self.evaluations += 1
        score, mass = np.zeros(len(weights)), np.empty(len(self.rows))
        pilot = np.empty(len(self.rows)) if collect_pilot else None
        log_likelihood = 0.
        for first in range(0, len(self.rows), self.block_rows):
            stop = min(len(self.rows), first+self.block_rows)
            likelihood = self.log_block(first, stop)
            likelihood -= self.offset[first:stop, None]
            np.exp(likelihood, out=likelihood)
            normalizer = likelihood @ weights
            if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
                return None, -np.inf, None, None
            mass[first:stop] = normalizer
            with np.errstate(over="ignore", invalid="ignore"):
                score += (1/normalizer) @ likelihood
            log_likelihood += float((np.log(normalizer)+self.offset[first:stop]).sum())
            if collect_pilot:
                pilot[first:stop] = likelihood @ (weights*self.grid)/normalizer
        if not np.isfinite(score).all():
            return None, -np.inf, None, None
        return score/len(self.rows), log_likelihood, mass, pilot

    def scaled_columns(self, positions):
        values = np.empty((len(self.rows), len(positions)))
        for first in range(0, len(self.rows), self.block_rows):
            stop = min(len(self.rows), first+self.block_rows)
            values[first:stop] = np.exp(self.log_block(first, stop, positions)
                                        - self.offset[first:stop, None])
        return values


def _pooled_adapter(model, rows, region, *, block_bytes=None):
    """Stream the explicit mixed/general-normal grid; exact concave weight gap.

    Safeguarded SQUAREM and exact grow/shrink exchange mirror the inherited
    pooled initializer without a quadratic-space SLSQP over the whole grid.
    Only this separately versioned adapter uses this route; matched standard
    observations continue to call the unmodified inherited initializer.
    Its masked grid scores reads jointly with assignment validity, without
    normalizing over admissible grid points.
    """
    grid = np.unique(np.r_[np.linspace(0., 1., 257), model.lower[rows, region],
                           model.upper[rows, region]])
    oracle = _GridOracle(model, rows, region, grid, block_bytes=block_bytes)
    weights = np.full(len(grid), 1/len(grid))
    score, likelihood, mass, _ = oracle.expectation(weights)
    initial_likelihood = likelihood
    accelerated = exchanges = active_refits = 0
    step_limit = 1.

    def normalize(values):
        values = np.maximum(values, 0.)
        return values/values.sum()

    for iteration in range(ADAPTER_MAX_ITERATIONS+1):
        if score is None:
            raise RuntimeError("Nonfinite streamed pooled-grid likelihood or gradient")
        gap = max(0., float(score.max()-1))
        if gap <= ADAPTER_GRID_TOLERANCE:
            break
        if iteration == ADAPTER_MAX_ITERATIONS:
            raise RuntimeError(f"Streamed pooled-grid weights did not converge: mean gap {gap:.6g}")
        previous_likelihood = likelihood
        first = normalize(weights*score)
        first_score, _, _, _ = oracle.expectation(first)
        if first_score is None:
            raise RuntimeError("Nonfinite streamed pooled-grid EM step")
        second = normalize(first*first_score)
        second_score, second_likelihood, second_mass, _ = oracle.expectation(second)
        if second_score is None:
            raise RuntimeError("Nonfinite streamed pooled-grid second EM step")
        residual, curvature = first-weights, second-2*first+weights
        step = max(1., min(float(np.sqrt((residual@residual)/max(
            curvature@curvature, np.finfo(float).tiny))), step_limit))
        candidate = normalize(weights+2*step*residual+step*step*curvature)
        candidate_score, candidate_likelihood, candidate_mass, _ = oracle.expectation(candidate)
        while candidate_likelihood < second_likelihood and step > 1.01:
            step = (step+1)/2
            candidate = normalize(weights+2*step*residual+step*step*curvature)
            candidate_score, candidate_likelihood, candidate_mass, _ = oracle.expectation(candidate)
        if candidate_likelihood >= second_likelihood:
            weights, score, likelihood, mass = candidate, candidate_score, candidate_likelihood, candidate_mass
            accelerated += int(step > 1)
        else:
            weights, score, likelihood, mass = second, second_score, second_likelihood, second_mass
        if step >= .99*step_limit:
            step_limit = min(step_limit*4, 1e100)
        grow = int(np.argmax(score))
        positive = np.flatnonzero(weights > 0)
        shrink = int(positive[np.argmin(score[positive])])
        if grow != shrink:
            columns = oracle.scaled_columns([grow, shrink])
            direction = weights[shrink]*(columns[:, 0]-columns[:, 1])

            def derivative(fraction):
                return float(np.mean(direction/np.maximum(mass+fraction*direction, np.finfo(float).tiny)))

            if derivative(0.) > 0:
                fraction = 1. if derivative(1.) >= 0 else brentq(derivative, 0., 1., xtol=1e-14)
                candidate = weights.copy()
                transfer = fraction*weights[shrink]
                candidate[grow] += transfer
                candidate[shrink] = 0. if fraction == 1. else weights[shrink]-transfer
                candidate = normalize(candidate)
                candidate_score, candidate_likelihood, candidate_mass, _ = oracle.expectation(candidate)
                if candidate_likelihood >= likelihood:
                    weights, score, likelihood, mass = candidate, candidate_score, candidate_likelihood, candidate_mass
                    exchanges += 1
        # Inherited acceleration: at most 64 currently positive coordinates,
        # never a quadratic-space SLSQP over the full O(N)-sized grid. Global
        # scores/exchanges still allow every zero-weight grid point to reenter.
        active = np.flatnonzero(weights > 0)
        if (iteration+1) % 10 == 0 and len(active) <= 64:
            previous, objective_value, objective_gradient = None, None, None

            def active_objective(candidate):
                nonlocal previous, objective_value, objective_gradient
                if previous is None or not np.array_equal(previous, candidate):
                    full = np.zeros(len(grid))
                    full[active] = candidate
                    gradient, value, _, _ = oracle.expectation(full)
                    previous = candidate.copy()
                    objective_value = -value/len(rows)
                    objective_gradient = -gradient[active] if gradient is not None else np.zeros(len(active))
                return objective_value, objective_gradient

            optimized = minimize(active_objective, weights[active], jac=True, method="SLSQP",
                                 bounds=[(0., 1.)]*len(active),
                                 constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                              "jac": lambda w: np.ones(len(w))},
                                 options={"ftol": 1e-13, "maxiter": 200})
            active_refits += 1
            if np.isfinite(optimized.x).all() and np.maximum(optimized.x, 0.).sum() > 0:
                candidate = np.zeros(len(grid))
                candidate[active] = normalize(optimized.x)
                candidate_score, candidate_likelihood, candidate_mass, _ = oracle.expectation(candidate)
                if candidate_likelihood >= likelihood:
                    weights, score, likelihood, mass = candidate, candidate_score, candidate_likelihood, candidate_mass
        if likelihood < previous_likelihood-1e-10*max(1., abs(previous_likelihood)):
            raise RuntimeError("Streamed pooled-grid likelihood decreased")
    score, likelihood, _, pilot = oracle.expectation(weights, collect_pilot=True)
    if score is None:
        raise RuntimeError("Nonfinite final streamed pooled-grid evaluation")
    gap = max(0., float(score.max()-1))
    if gap > ADAPTER_GRID_TOLERANCE:
        raise RuntimeError("Streamed pooled-grid final duality gap changed")
    return pilot, {
        "method": "general_normal_mixed_cn_bulk_v1_streamed_pooled_ccf_grid_v2",
        "inherited_single_state_initialization": False,
        "grid_size": len(grid), "iterations": iteration,
        "max_iterations": ADAPTER_MAX_ITERATIONS,
        "grid_optimality_tolerance": ADAPTER_GRID_TOLERANCE,
        "grid_score_excess": gap, "weight_optimality_gap": len(rows)*gap,
        "grid_weight_optimum_certified": True,
        "initial_log_likelihood": initial_likelihood, "log_likelihood": likelihood,
        "likelihood_evaluations": oracle.evaluations, "accelerated_steps": accelerated,
        "exchange_steps": exchanges, "block_rows": oracle.block_rows,
        "active_support_refits": active_refits, "max_active_refit_size": 64,
        "peak_grid_block_bytes": oracle.peak_block_bytes,
        "dense_grid_matrix_retained": False,
    }
