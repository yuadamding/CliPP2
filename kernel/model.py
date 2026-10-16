"""CliPP2 emissions, conditional centers and joint reads/validity mixture score.

CCF, not cellular prevalence, is the public fitting/geometry scale. Input CN
states collapse to maximum major and maximum total CN before inference.
Diploid-normal observations use the shared scalar likelihood primitives.
Assignment validity is observed
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
from .grid_oracle import GridOracle as _GridOracle
from .likelihood import MultiplicityModel
from .refitting import refit_center
from .scoring import fit_cluster_weights
from .memory import ByteLRU
from .refit_store import ExactRefitStore
from .regional_support import RegionalSupport
from .partitions import PartitionState, _PartitionAuthority
from ..io.data import readonly_array


MODEL_VERSION = "clipp2_regional_full_support__max_major_max_total_cn_v1"
ADAPTER_BLOCK_BYTES = 8 * 1024 * 1024
ADAPTER_MAX_ITERATIONS = 500
ADAPTER_GRID_TOLERANCE = 1e-8


@dataclass(frozen=True)
class RefittedPartition:
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    clonal_cluster: int
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

    def __init__(self, data, device="cpu", *, cache_bytes=32 * 1024 * 1024, memory=None,
                 regional_zero_policy="strict"):
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
                             ("total", data.total_cn), ("purity", data.purity)):
            setattr(self, name, readonly_array(np.asarray(source)[ix], dtype=float))
        for values in (self.major, self.purity, self.total, self.normal):
            if not np.isfinite(values).all():
                raise ValueError("CN and purity must be finite in every mutation-region")
        if np.any(self.major < 1) or np.any(self.major != np.rint(self.major)):
            raise ValueError("Major CN must be a positive integer")
        if np.any(self.normal < 0) or np.any(self.normal != np.rint(self.normal)):
            raise ValueError("Normal CN must be a nonnegative integer")
        if np.any(self.total < self.major) or np.any(self.total != np.rint(self.total)):
            raise ValueError("Total CN must be an integer at least major CN")
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
        self._regional_support = RegionalSupport.from_counts(
            self.alt, self.depth, self.observed, policy=regional_zero_policy)
        denominator = (1 - self.purity) * self.normal + self.purity * self.total
        if np.any(denominator <= 0):
            raise ValueError("Normal-plus-tumor denominator must be positive")
        self.support = readonly_array(self.major, dtype=int)
        self.lower = readonly_array(np.zeros_like(self.major))
        self.upper = readonly_array(np.ones_like(self.major))
        self.scaling = readonly_array(self.purity / denominator)
        m = np.arange(1, int(self.support.max()) + 1)
        self.valid = readonly_array(m <= self.support[..., None])
        self.slope = readonly_array(self.scaling[..., None] * m)
        self.constants = readonly_array(
            gammaln(self.depth + 1) - gammaln(self.alt + 1)
            - gammaln(self.depth - self.alt + 1) - np.log(self.support))
        digest = hashlib.sha256(MODEL_VERSION.encode())
        for values in (self.alt, self.depth, self.observed, self.slope,
                       self.support, self.lower, self.upper):
            digest.update(values.tobytes())
        digest.update(repr((self.mutation_ids, self.region_ids)).encode())
        # Preserve the pinned likelihood-mode identity; strict caches occupy a
        # separate immutable constraint namespace even when no zero is active.
        if regional_zero_policy == "strict":
            digest.update(self.regional_support.identity.encode())
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
                         for name in ("slope", "valid", "observed", "alt", "depth", "constants")}
        self.execution_backend = "cuda_likelihood_host_forest" if self._torch else "cpu"
        self.memory = memory
        self._cache_bytes = int(cache_bytes)
        if self._cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self._refits, self._columns = OrderedDict(), OrderedDict()
        self._refit_bytes = self._column_bytes = 0
        self._refit_limit = memory.refit_bytes if memory else self._cache_bytes
        self._column_limit = memory.column_bytes if memory else self._cache_bytes
        # Modal proposal vectors share the existing column allowance; they are
        # not another persistent allocation charged to temporary workspace.
        profile_bytes = min(8*1024*1024, self._column_limit//8)
        self._proposal_profiles = ByteLRU(profile_bytes)
        self._column_limit -= profile_bytes
        self._weights = ByteLRU(memory.weight_bytes if memory else 0)
        self._device_columns = ByteLRU(memory.device_column_bytes if memory else 0)
        self.telemetry = {"refit_cache_hits": 0, "scalar_refits": 0,
                          "column_cache_hits": 0, "likelihood_columns": 0,
                          "joint_matrix_calls": 0,
                          "scalar_refit_seconds": 0.0, "weight_seconds": 0.0,
                          "free_block_refits": 0, "free_block_preparations": 0,
                          "mixture_weight_fits": 0,
                          "anchored_variant_evaluations": 0, "clonal_column_hits": 0,
                          "weight_cache_hits": 0, "refit_cache_evictions": 0,
                          "column_cache_evictions": 0, "likelihood_batches": 0,
                          "proposal_profile_columns": 0,
                          "host_to_device_bytes": 0, "device_to_host_bytes": 0}
        self._refit_store = None
        self._prepare_clonal_column()
        self._partition_authority = _PartitionAuthority(self)

    @property
    def regional_support(self):
        return self._regional_support

    @property
    def regional_zero_policy(self):
        return self.regional_support.policy

    def cache_diagnostics(self):
        result = dict(self.telemetry)
        result.update(refit_cache_bytes=self._refit_bytes, column_cache_bytes=self._column_bytes,
                      completed_weights=self._weights.stats(), device_columns=self._device_columns.stats(),
                      proposal_profiles=self._proposal_profiles.stats())
        if self._refit_store is not None:
            result['refit_grid'] = {**self._refit_store.diagnostics,
                                   'bytes': self._refit_store.bytes}
        return result

    def refit_store(self):
        if self.memory is None or not self.memory.refit_grid_bytes:
            return None
        if self._refit_store is None:
            self._refit_store = ExactRefitStore(self, self.memory.refit_grid_bytes)
        return self._refit_store

    def _prepare_clonal_column(self):
        """One immutable exact emission column, outside the evictable LRU."""
        ones = np.ones((1, self.r))
        allowed = self.assignment_admissibility(ones)[:, 0]
        emission = -self.loss(np.ones((self.n, self.r))).sum(axis=1)
        self.clonal_log_column = readonly_array(np.where(allowed, emission, -np.inf))
        self.clonal_eligible = readonly_array(
            allowed & np.isfinite(emission) & ~self.regional_support.hard_zero.any(axis=1))

    def check_clonal_feasibility(self, max_clusters):
        if not self.clonal_eligible.any():
            raise ValueError("clonal_infeasible_no_eligible_mutation")
        if not self.observed[self.clonal_eligible].any(axis=0).all():
            raise ValueError("clonal_infeasible_missing_regional_support")
        if max_clusters == 1 and not self.clonal_eligible.all():
            raise ValueError("clonal_infeasible_single_cluster_capacity")

    def __len__(self):
        return self.n

    def _numpy_kernel(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n, self.r) or np.any(~np.isfinite(x[self.observed])):
            raise ValueError("CCFs must match N by R and be finite where observed")
        if np.any(x[self.observed] < 0) or np.any(x[self.observed] > 1):
            raise ValueError("CCFs must be in [0,1]")
        mass = np.where(self.observed, x, 0.0)[..., None] * self.slope
        p = np.minimum(1., mass)
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
        p = mass.clamp(max=1.)
        alt = t.where(a["observed"], a["alt"], 0.)[..., None]
        ref = t.where(a["observed"], a["depth"]-a["alt"], 0.)[..., None]
        kernel = t.special.xlogy(alt, p) + t.special.xlog1py(ref, -p) + a["constants"][..., None]
        kernel = t.where(a["valid"], kernel, -t.inf)
        ll = t.logsumexp(kernel, -1)
        loss = t.where(a["observed"], -ll, 0.)
        if not derivatives:
            return loss.cpu().numpy()
        post = t.where(t.isfinite(kernel), t.exp(kernel-ll[..., None]), 0.)
        derivative_slope = a["slope"]
        safe_p = t.where(alt > 0, p, 1.)
        safe_q = t.where(ref > 0, 1-p, 1.)
        g = derivative_slope * (-alt/safe_p + ref/safe_q)
        h = derivative_slope.square() * (alt/safe_p.square() + ref/safe_q.square())
        g = t.where(post > 0, post*g, 0.).sum(-1)
        h = t.where(post > 0, post*h, 0.).sum(-1)
        boundary = a["valid"] & (p == 1) & (ref == 1)
        correction = t.where(boundary, a["slope"] * t.exp(a["constants"][..., None] - ll[..., None]), 0.).sum(-1)
        g += correction
        h += t.where(a["valid"] & (p == 1) & (ref == 2),
                     2*a["slope"].square()*t.exp(a["constants"][..., None]-ll[..., None]), 0.).sum(-1)
        h = t.where(boundary.any(-1), t.inf, h.clamp(min=1e-8))
        g = t.where(t.isfinite(ll), g, 0.)
        h = t.where(t.isfinite(ll), h, 1.)
        return tuple(t.where(a["observed"], v, 0.).cpu().numpy() for v in (loss, g, h))

    def loss(self, x):
        if self._torch:
            return self._gpu_terms(x)
        return np.where(self.observed, -self._numpy_kernel(x)[-1], 0.)

    def node_terms_batch(self, points):
        """Independent N-by-R evaluations; only the leading task axis is new."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 3 or points.shape[1:] != (self.n, self.r):
            raise ValueError('Batched coordinates must be B by N by R')
        if not np.isfinite(points[:, self.observed]).all() or np.any(
                (points[:, self.observed] < 0) | (points[:, self.observed] > 1)):
            raise ValueError('Observed CCFs must be finite in [0,1]')
        self.telemetry['continuation_likelihood_batches'] = self.telemetry.get('continuation_likelihood_batches', 0)+1
        if not self._torch:
            terms = [self.node_terms(x) for x in points]
            return tuple(np.stack([value[j] for value in terms]) for j in range(3))
        t = self._torch
        a = {key: value[None, ...] for key, value in self._gpu.items()}
        with t.no_grad():
            xx = t.tensor(np.where(self.observed[None, ...], points, 0.), device=self.device)
            p = (xx[..., None]*a['slope']).clamp(max=1.)
            alt, ref = a['alt'][..., None], (a['depth']-a['alt'])[..., None]
            kernel = (t.special.xlogy(alt, p)+t.special.xlog1py(ref, -p)+a['constants'][..., None])
            kernel = t.where(a['valid'], kernel, -t.inf)
            ll = t.logsumexp(kernel, -1)
            post = t.where(t.isfinite(kernel), t.exp(kernel-ll[..., None]), 0.)
            safe_p, safe_q = t.where(alt > 0, p, 1.), t.where(ref > 0, 1-p, 1.)
            g = a['slope']*(-alt/safe_p+ref/safe_q)
            h = a['slope'].square()*(alt/safe_p.square()+ref/safe_q.square())
            g = t.where(post > 0, post*g, 0.).sum(-1)
            h = t.where(post > 0, post*h, 0.).sum(-1)
            boundary = a['valid'] & (p == 1) & (ref == 1)
            g += t.where(boundary, a['slope']*t.exp(a['constants'][..., None]-ll[..., None]), 0.).sum(-1)
            h += t.where(a['valid'] & (p == 1) & (ref == 2),
                         2*a['slope'].square()*t.exp(a['constants'][..., None]-ll[..., None]), 0.).sum(-1)
            h = t.where(boundary.any(-1), t.inf, h.clamp(min=1e-8))
            g, h = t.where(t.isfinite(ll), g, 0.), t.where(t.isfinite(ll), h, 1.)
            result = tuple(t.where(a['observed'], v, 0.).cpu().numpy() for v in (-ll, g, h))
        self.telemetry['host_to_device_bytes'] += points.nbytes
        self.telemetry['device_to_host_bytes'] += sum(v.nbytes for v in result)
        return result

    def node_terms(self, x):
        """Loss, exact gradient, positive complete-data curvature on CCF scale."""
        if self._torch:
            return self._gpu_terms(x, True)
        p, mass, kernel, ll = self._numpy_kernel(x)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            post = np.where(np.isfinite(kernel), np.exp(kernel-ll[..., None]), 0.)
            slope = self.slope
            alt, ref = self.alt[..., None], (self.depth-self.alt)[..., None]
            g = slope * (np.divide(-alt, p, out=np.zeros_like(p), where=alt != 0)
                         + np.divide(ref, 1-p, out=np.zeros_like(p), where=ref != 0))
            h = slope**2 * (np.divide(alt, p*p, out=np.zeros_like(p), where=alt != 0)
                            + np.divide(ref, (1-p)**2, out=np.zeros_like(p), where=ref != 0))
            g = np.where(post > 0, post*g, 0.).sum(-1)
            h = np.where(post > 0, post*h, 0.).sum(-1)
            boundary = self.valid & (p == 1) & (ref == 1)
            g += np.where(boundary, self.slope*np.exp(self.constants[..., None]-ll[..., None]), 0.).sum(-1)
            h += np.where(self.valid & (p == 1) & (ref == 2),
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
        if np.any(self.normal[rows, region] != 2):
            return None
        return MultiplicityModel(self.alt[rows, region], self.depth[rows, region],
                                 self.major[rows, region], self.total[rows, region],
                                 self.region_purities[region])

    def _regional_likelihood(self, values, rows, region):
        """Host scalar/grid evaluation for effective integer CN."""
        values = np.atleast_1d(values)
        p = values[:, None, None] * self.slope[rows, region][None, :, :]
        p = np.minimum(p, 1.)
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
                cp, diag = pooled_cp_initialization(scalar, **(
                    {'matrix_cache_bytes': self.memory.initialization_bytes} if self.memory else {}))
                pilots[rows, region] = cp / scalar.purity
            else:
                pilots[rows, region], diag = _pooled_adapter(self, rows, region)
            diagnostics[self.region_ids[region]] = diag
        return pilots, diagnostics

    def observation_keys(self):
        """ID-free scientific row signatures for deterministic proposal ties."""
        arrays = (self.observed, self.alt, self.depth, self.slope, self.support,
                  self.lower, self.upper)
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
        # Exact compact membership bytes replace Python int tuples. The per-fit
        # model and refitter identities are explicit, not global reusable state.
        members = np.asarray(members, dtype=np.int64)
        key = (self.identity, members.tobytes(), refit_center)
        entry = self._refits.get(key)
        if entry is not None and region in entry[0]:
            self.telemetry["refit_cache_hits"] += 1
            self._refits.move_to_end(key)
            return entry[0][region]
        start = time.perf_counter()
        rows = members[self.observed[members, region]]
        lower, upper = self.lower[members, region].max(), self.upper[members, region].min()
        if not len(rows):
            raise ValueError(f"unsupported_cluster_region:{self.region_ids[region]}")
        if lower > upper:
            raise ValueError(f"empty_cluster_region_box:{self.region_ids[region]}")
        hard_zero = self.regional_support.hard_zero[rows, region].any()
        if hard_zero and self.regional_support.positive[rows, region].any():
            raise ValueError(f"regional_zero_positive_conflict:{self.region_ids[region]}")
        if hard_zero:
            if not lower <= 0. <= upper:
                raise ValueError(f"empty_cluster_region_box:{self.region_ids[region]}")
            fitted = (0., float(self._regional_likelihood([0.], rows, region).sum()))
            scalar, store = None, None
        else:
            scalar, store = self._scalar_model(rows, region), self.refit_store()
        if not hard_zero and scalar is not None and lower == 0 and upper == 1:
            cp, ll = refit_center(scalar, **({
                'grid_evaluator': lambda grid: store.grid_log_likelihood(rows, region, grid, scalar=scalar)
            } if store is not None else {}))
            fitted = (cp/scalar.purity, ll)
        elif not hard_zero:
            modes = (self.alt[rows, region]/self.depth[rows, region])[:, None] / self.slope[rows, region]
            grid = np.unique(np.r_[np.linspace(lower, upper, 513), np.clip(modes[self.valid[rows, region]], lower, upper)])
            values = (store.grid_log_likelihood(rows, region, grid) if store is not None else
                      np.concatenate([self._regional_likelihood(grid[i:i+32], rows, region).sum(axis=1)
                                      for i in range(0, len(grid), 32)]))
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
        # One exact membership identity shared across regional refits. Initial
        # center fallbacks stay outside this baseline cache.
        values = dict(entry[0]) if entry is not None else {}
        values[region] = fitted
        size = 768 + members.nbytes + 160*len(values)
        if size <= self._refit_limit:
            old = self._refits.pop(key, None)
            if old is not None:
                self._refit_bytes -= old[1]
            while self._refit_bytes + size > self._refit_limit:
                _, (_, removed) = self._refits.popitem(last=False)
                self._refit_bytes -= removed
                self.telemetry['refit_cache_evictions'] += 1
            self._refits[key] = (values, size)
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
        if self.memory is not None:
            return self._resident_columns(centers, allowed)
        columns = [None]*len(centers)
        for k, center in enumerate(centers):
            if np.array_equal(center, np.ones(self.r)):
                self.telemetry["clonal_column_hits"] += 1
                columns[k] = self.clonal_log_column
                continue
            key = (self.identity, center.tobytes())
            column = self._columns.get(key)
            if column is not None:
                self.telemetry["column_cache_hits"] += 1
                self._columns.move_to_end(key)
                columns[k] = column
            else:
                emission = -self.loss(np.broadcast_to(center, (self.n, self.r))).sum(axis=1)
                self.telemetry["likelihood_columns"] += 1
                self._store_joint_column(key, (k,), emission, allowed, columns)
        return np.column_stack(columns)

    def proposal_log_likelihood_columns(self, centers):
        """Hard-assignment ranking only; raw cached mixture emissions stay intact."""
        raw = self.log_likelihood_columns(centers)
        return np.where(self.regional_support.assignment_mask(centers), raw, -np.inf)

    def _resident_columns(self, centers, allowed):
        """Batch independent emissions only; keep host R-sum and search order."""
        if np.any(centers < 0) or np.any(centers > 1):
            raise ValueError('CCFs must be in [0,1]')
        columns, missing = [None]*len(centers), {}
        for k, center in enumerate(centers):
            if np.array_equal(center, np.ones(self.r)):
                columns[k] = self.clonal_log_column
                self.telemetry['clonal_column_hits'] += 1
                continue
            key = (self.identity, center.tobytes())
            if key in self._columns:
                self._columns.move_to_end(key)
                columns[k] = self._columns[key]
                self.telemetry['column_cache_hits'] += 1
            else:
                missing.setdefault(key, []).append(k)
        pending = []
        for key, positions in missing.items():
            cached = self._device_columns.get(key) if self._torch else None
            if cached is None:
                pending.append((key, positions))
            else:
                values = cached.cpu().numpy()
                self.telemetry['device_to_host_bytes'] += values.nbytes
                self._store_joint_column(key, positions, -values.sum(axis=1), allowed, columns)
        # Only the independent center axis is batched. Mutation/region and
        # multiplicity reductions retain the original contiguous layout.
        workspace = (self.memory.device_workspace_bytes if self._torch
                     else self.memory.workspace_bytes)
        per_center = max(1, self.n*self.r*(128*self.valid.shape[-1]+64))
        width = max(1, min(64, workspace//per_center))
        for first in range(0, len(pending), width):
            batch = pending[first:first+width]
            values = centers[[positions[0] for _, positions in batch]]
            if self._torch and not self.memory.vram_bytes:
                losses = np.stack([self._gpu_terms(np.broadcast_to(c, (self.n, self.r))) for c in values])
            elif self._torch:
                losses = self._batched_device_losses(values, batch)
            else:
                losses = self._batched_host_losses(values)
            self.telemetry['likelihood_batches'] += 1
            self.telemetry['likelihood_columns'] += len(batch)
            for (key, positions), loss in zip(batch, losses):
                self._store_joint_column(key, positions, -loss.sum(axis=1), allowed, columns)
        return np.column_stack(columns)

    def _store_joint_column(self, key, positions, emission, allowed, columns):
        column = readonly_array(np.where(allowed[:, positions[0]], emission, -np.inf))
        size = column.nbytes+len(key[1])+512
        if size <= self._column_limit:
            while self._column_bytes+size > self._column_limit:
                old_key, old = self._columns.popitem(last=False)
                self._column_bytes -= old.nbytes+len(old_key[1])+512
                self.telemetry['column_cache_evictions'] += 1
            self._columns[key] = column
            self._column_bytes += size
        for k in positions:
            columns[k] = column

    def _batched_host_losses(self, centers):
        x = np.where(self.observed[None, ...], centers[:, None, :], 0.)
        p = np.minimum(1., x[..., None]*self.slope[None, ...])
        kernel = (xlogy(self.alt[None, ..., None], p)
                  + xlog1py((self.depth-self.alt)[None, ..., None], -p)
                  + self.constants[None, ..., None])
        ll = logsumexp(np.where(self.valid[None, ...], kernel, -np.inf), axis=-1)
        return np.where(self.observed[None, ...], -ll, 0.)

    def _batched_device_losses(self, centers, keys):
        t, a = self._torch, self._gpu
        with t.no_grad():
            center = t.tensor(centers, device=self.device)
            self.telemetry['host_to_device_bytes'] += centers.nbytes
            x = t.where(a['observed'][None, ...], center[:, None, :], 0.)
            p = (x[..., None]*a['slope'][None, ...]).clamp(max=1.)
            kernel = (t.special.xlogy(a['alt'][None, ..., None], p)
                      + t.special.xlog1py((a['depth']-a['alt'])[None, ..., None], -p)
                      + a['constants'][None, ..., None])
            ll = t.logsumexp(t.where(a['valid'][None, ...], kernel, -t.inf), -1)
            losses = t.where(a['observed'][None, ...], -ll, 0.)
            for (key, _), loss in zip(keys, losses):
                size = loss.numel()*loss.element_size()+len(key[1])+256
                if size <= self._device_columns.budget_bytes:
                    self._device_columns.put(key, loss.clone(), size)
            host = losses.cpu().numpy()
            self.telemetry['device_to_host_bytes'] += host.nbytes
            return host

    def _refit_partition_state(self, labels, state, *, audit=False):
        if (not audit and state is not None and labels is state.labels
                and self._partition_authority.trusts(state, self)):
            self.telemetry['partition_state_trusted_hits'] = self.telemetry.get(
                'partition_state_trusted_hits', 0)+1
            return np.arange(len(state.members)), state.labels, state
        labels = np.asarray(labels)
        if (labels.shape != (self.n,) or labels.dtype.kind not in "iuf"
                or not np.isfinite(labels).all() or np.any(labels != np.rint(labels))):
            raise ValueError("Labels must contain one finite integer per mutation")
        occupied, inverse = np.unique(labels, return_inverse=True)
        if state is not None and not np.array_equal(state.labels, inverse):
            raise ValueError('Partition cache labels disagree with the requested refit')
        checked = PartitionState.from_labels(inverse, self)
        if state is not None and (
                state.support_identity != checked.support_identity
                or len(state.members) != len(checked.members)
                or any(not np.array_equal(getattr(state, name), getattr(checked, name))
                       for name in ("counts", "observed_counts", "positive_counts", "hard_zero_counts",
                                    "forced_zero", "support_conflict", "lower", "upper", "clonal_eligible"))
                or any(not np.array_equal(rows, expected)
                       for rows, expected in zip(state.members, checked.members))):
            raise ValueError('Partition cache support disagrees with the requested refit')
        self.telemetry['partition_state_full_checks'] = self.telemetry.get(
            'partition_state_full_checks', 0)+1
        return occupied, inverse, checked

    def _refit_variants(self, labels, designations, initial_centers, *, partition_state=None,
                        audit_partition_state=False):
        occupied, inverse, summary = self._refit_partition_state(
            labels, partition_state, audit=audit_partition_state)
        q = len(summary.members)
        centers = np.full((q, self.r), np.nan)
        if initial_centers is not None:
            initial_centers = np.asarray(initial_centers, dtype=float)
            if initial_centers.shape != centers.shape:
                raise ValueError("Initial centers must follow sorted occupied label order")
        if designations is None:
            designations = [int(k) for i, k in enumerate(occupied) if summary.clonal_eligible[i]
                            and (summary.observed_counts[i] > 0).all()]
        elif any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                 or k not in occupied for k in designations):
            raise ValueError("clonal_cluster must designate an occupied label")
        designations = [int(np.flatnonzero(occupied == k)[0]) for k in designations]
        penalty = float((q-1)*(self.r+1)*math.log(self.n))
        # Keep the original likelihood route unchanged. Under active strict
        # constraints all direct entry points reject structural conflicts before
        # preparing a free center, resident grid, emission column or weight fit.
        structural = (self.regional_support.check_partition(summary)
                      if self.regional_support.hard_zero.any() else None)
        if structural:
            for clonal in designations:
                self.telemetry["anchored_variant_evaluations"] += 1
                fixed = centers.copy()
                fixed[clonal] = 1.
                yield self._failed_refit(inverse, fixed, clonal, penalty, structural)
            return
        invalid_designations = [k for k in designations if summary.forced_zero[k].any()]
        for clonal in invalid_designations:
            self.telemetry["anchored_variant_evaluations"] += 1
            fixed = centers.copy()
            fixed[clonal] = 1.
            yield self._failed_refit(inverse, fixed, clonal, penalty, "clonal_block_inadmissible")
        designations = [k for k in designations if k not in invalid_designations]
        if not designations:
            return
        errors, retained = {}, np.zeros(q, dtype=int)
        # Prepare independent free fits once, including errors. Fixing a block
        # must not depend on its free optimizer succeeding.
        for cluster in range(q):
            if not any(k != cluster for k in designations):
                continue
            self.telemetry["free_block_preparations"] += 1
            previous_refits = self.telemetry["scalar_refits"]
            try:
                members = summary.members[cluster]
                for region in range(self.r):
                    center, ll = self._fit_scalar(members, region)
                    if initial_centers is not None and not summary.forced_zero[cluster, region]:
                        old = initial_centers[cluster, region]
                        if (np.isfinite(old) and self.lower[members, region].max() <= old
                                <= self.upper[members, region].min()):
                            rows = members[self.observed[members, region]]
                            old_ll = float(self._regional_likelihood([old], rows, region).sum())
                            if old_ll > ll:
                                center, ll = float(old), old_ll
                                retained[cluster] += 1
                    centers[cluster, region] = center
            except (ValueError, RuntimeError) as error:
                errors[cluster] = str(error)
            finally:
                self.telemetry["free_block_refits"] += self.telemetry["scalar_refits"] > previous_refits
        # The family retains q free columns plus one immutable clonal column,
        # not q independent N-by-q matrices. Numerical errors in a free block
        # must still be repairable by designating that block clonal.
        family, bounds = None, {}
        if designations and self.memory is not None and 32*self.n*(q+1) <= self.memory.workspace_bytes:
            from .weight_batch import AnchoredFamily
            finite = np.isfinite(centers).all(axis=1)
            free = np.full((self.n, q), -np.inf)
            if finite.any():
                free[:, finite] = self.log_likelihood_columns(centers[finite])
            family = AnchoredFamily(free, self.clonal_log_column, designations)
            bounds = family.fixed_center_bounds(penalty)
            self.telemetry['clonal_families'] = self.telemetry.get('clonal_families', 0)+1
            self.telemetry['clonal_family_emission_columns'] = self.telemetry.get('clonal_family_emission_columns', 0)+q+1
        qualify = (family is not None and self._torch and self.memory.vram_bytes > 0
                   and self.memory.qualify_device_weights)
        completed, reference_results = [], {}
        for clonal in designations:
            self.telemetry["anchored_variant_evaluations"] += 1
            fixed = centers.copy()
            fixed[clonal] = 1.
            error = next((reason for k, reason in errors.items() if k != clonal), None)
            if not summary.clonal_eligible[clonal]:
                error = "clonal_block_inadmissible"
            elif not (summary.observed_counts[clonal] > 0).all():
                error = "unsupported_cluster_region:clonal"
            if error:
                fitted = self._failed_refit(inverse, fixed, clonal, penalty, error)
                if qualify:
                    completed.append(fitted)
                else:
                    yield fitted
                continue
            previous_hits = self._weights.hits
            fitted = self._score_refit(inverse, fixed, clonal, penalty,
                int(retained.sum()-retained[clonal]), support_summary=summary, **(
                    {'matrix': family.matrix(clonal)} if family is not None else {}))
            if family is not None:
                fitted.diagnostics['fixed_center_bound'] = bounds[clonal]
            if qualify:
                completed.append(fitted)
                if self._weights.hits == previous_hits and fitted.reason in (None, 'zero_mixture_weight'):
                    reference_results[clonal] = dict(cluster_weights=fitted.weights,
                        log_likelihood=fitted.mixture_log_likelihood,
                        weight_optimality_gap=fitted.diagnostics['weight_optimality_gap'],
                        weight_active_score_gap=fitted.diagnostics['weight_active_score_gap'])
            else:
                yield fitted
        if qualify:
            from .weight_batch import qualify_family_weights
            diagnostics = qualify_family_weights(family, np.bincount(inverse), reference_results,
                device=self.device, vram_budget_bytes=self.memory.device_workspace_bytes,
                torch_module=self._torch)
            for fitted in completed:
                fitted.diagnostics['device_weight_qualification'] = diagnostics.get(
                    fitted.clonal_cluster, {'applied': False, 'reason': 'cached_or_failed_reference'})
                yield fitted

    def refit(self, labels, *, clonal_cluster, initial_centers=None):
        return next(self._refit_variants(labels, [clonal_cluster], initial_centers))

    def refit_variants(self, labels, *, initial_centers=None):
        """All admissible fixed-block choices; free-coordinate work is shared."""
        return self._refit_variants(labels, None, initial_centers)

    @staticmethod
    def _failed_refit(labels, centers, clonal, penalty, reason, conditional=-np.inf):
        return RefittedPartition(labels=labels, centers=centers,
            weights=np.full(len(centers), np.nan), clonal_cluster=clonal,
            conditional_log_likelihood=conditional, mixture_log_likelihood=-np.inf,
            complexity_penalty=penalty, score=np.inf, eligible=False, reason=reason)

    def _score_refit(self, inverse, centers, clonal, penalty, retained_initial, *, matrix=None,
                     support_summary=None):
        q = len(centers)
        if not self.regional_support.assigned_mask(inverse, centers).all():
            return self._failed_refit(inverse, centers, clonal, penalty,
                                      "regional_zero_assignment_violation")
        if matrix is None:
            matrix = self.log_likelihood_columns(centers)
        conditional = math.fsum(matrix[np.arange(self.n), inverse])
        if not np.isfinite(conditional):
            return self._failed_refit(inverse, centers, clonal, penalty, "nonfinite_assigned_likelihood")
        start = time.perf_counter()
        counts = np.bincount(inverse)
        key = (self.identity, fit_cluster_weights, centers.dtype.str,
               centers.shape, centers.tobytes(), counts.dtype.str, counts.tobytes())
        try:
            mixture = self._weights.get(key)
            if mixture is None:
                self.telemetry["mixture_weight_fits"] += 1
                mixture = fit_cluster_weights(_WeightModel(self.n), np.zeros(q),
                                              counts, log_kernel=matrix,
                                              telemetry=self.telemetry)
                mixture = {**mixture, 'cluster_weights': readonly_array(mixture['cluster_weights'])}
                self._weights.put(key, mixture, 1024+centers.nbytes+counts.nbytes+mixture['cluster_weights'].nbytes)
            else:
                self.telemetry['weight_cache_hits'] += 1
        except (ValueError, RuntimeError) as error:
            return self._failed_refit(inverse, centers, clonal, penalty, f"weight_fit:{error}", conditional)
        finally:
            self.telemetry["weight_seconds"] += time.perf_counter()-start
        ll = mixture["log_likelihood"]
        eligible = bool(np.all(mixture["cluster_weights"] > 0))
        diagnostics = {
            "weight_optimality_gap": mixture["weight_optimality_gap"],
            "weight_active_score_gap": mixture["weight_active_score_gap"],
            "retained_initial_centers": retained_initial,
            "scalar_global_optimality_certified": False,
            "num_parameters": (q-1)*(self.r+1), "score_definition": SCORE_DEFINITION}
        if self.regional_zero_policy == "strict":
            summary = (self.regional_support.summarize_partition(inverse)
                       if support_summary is None else support_summary)
            zero_coordinates = int(summary.forced_zero.sum())
            diagnostics.update(
                regional_zero_policy="strict", regional_support_identity=self.regional_support.identity,
                regional_zero_assignment_verified=True, assigned_zero_violations=0,
                penalty_parameter_count=(q-1)*(self.r+1),
                free_center_coordinate_count=(q-1)*self.r-zero_coordinates,
                fixed_zero_coordinate_count=zero_coordinates,
                forced_zero_cluster_coordinates=zero_coordinates,
                penalty_dimension_convention="nominal_full_dimension")
        return RefittedPartition(labels=inverse, centers=centers, weights=mixture["cluster_weights"],
                                 clonal_cluster=clonal, conditional_log_likelihood=conditional,
                                 mixture_log_likelihood=ll, complexity_penalty=penalty,
                                 score=-2*ll+penalty, eligible=eligible,
                                 reason=None if eligible else "zero_mixture_weight", diagnostics=diagnostics)


class _WeightModel(SimpleNamespace):
    """Only cardinality/domain are used when inherited scorer gets log_kernel."""
    def __init__(self, n):
        super().__init__(n=n, purity=1.)

    def __len__(self):
        return self.n


def _pooled_adapter(model, rows, region, *, block_bytes=None, matrix_cache_bytes=None):
    """Budgeted general-normal-CN grid; exact concave weight gap.

    Safeguarded SQUAREM and exact grow/shrink exchange mirror the inherited
    pooled initializer without a quadratic-space SLSQP over the whole grid.
    Only this separately versioned adapter uses this route; matched standard
    observations continue to call the unmodified inherited initializer.
    Its masked grid scores reads jointly with assignment validity, without
    normalizing over admissible grid points.
    """
    grid = np.unique(np.r_[np.linspace(0., 1., 257), model.lower[rows, region],
                           model.upper[rows, region]])
    if matrix_cache_bytes is None:
        memory = getattr(model, "memory", None)
        matrix_cache_bytes = memory.initialization_bytes if memory is not None else 0
    oracle = _GridOracle(model, rows, region, grid,
                         block_bytes=ADAPTER_BLOCK_BYTES if block_bytes is None else block_bytes,
                         cache_bytes=matrix_cache_bytes)
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
        "method": "general_normal_integer_cn_streamed_pooled_ccf_grid_v1",
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
        "dense_grid_matrix_retained": oracle.cached,
        "matrix_cache_budget_bytes": oracle.cache_bytes,
        "matrix_cache_resident_bytes": oracle.resident_bytes,
        "matrix_cache_hits": oracle.cache_hits,
        "likelihood_blocks_computed": oracle.likelihood_blocks_computed,
    }
