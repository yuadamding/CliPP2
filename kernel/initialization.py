"""Pooled CCF pilots through the common regional-view emission interface.

Independent mutation maxima cannot resolve multiplicity aliases: a mutation
with multiplicity one in a major-CN-four region can be placed near one quarter
of its true CCF. Pooling the same marginal likelihood across a region provides
an unsupervised initialization. This grid mixture is neither the fitted model
nor a new penalty, and its weights never enter the native objective or BIC.
"""
# Adapted from CliPP1.5; see ../NOTICE.

import operator
import sys
import hashlib

import numpy as np
from scipy.optimize import brentq, minimize

GRID_SIZE = 257
MAX_ADDITIONAL_GRID_POINTS = 2048
MAX_ITERATIONS = 500
MEAN_LOGLIK_TOLERANCE = 1e-8
GRID_OPTIMALITY_TOLERANCE = 1e-8
MAX_ACTIVE_REFIT_SIZE = 64
ACTIVE_REFIT_INTERVAL = 10
MAX_ACTIVE_REFIT_ITERATIONS = 200
MAX_MATRIX_BYTES = 64 * 1024 * 1024
ADAPTER_BLOCK_BYTES = 8 * 1024 * 1024
ADAPTER_MAX_ITERATIONS = 500
ADAPTER_GRID_TOLERANCE = 1e-8


class _GridCache:
    """A byte-bounded, immutable prefix of scaled grid blocks; no LRU churn.

    Account for owned arrays, offsets and tuple metadata, not just matrix
    payload. The last record may hold a row prefix; callers reconstruct the
    original computation block before reductions. Working arrays are separate.
    """

    def __init__(self, cache_bytes):
        self.budget = MAX_MATRIX_BYTES if cache_bytes is None else operator.index(cache_bytes)
        if self.budget < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self._blocks = ()
        self._payload_bytes = 0
        self.retained_bytes = self.hits = self.misses = 0
        self._sealed = self.budget == 0
        self.partial_rows = self.cached_rows = 0

    def get(self, index):
        if index < len(self._blocks):
            self.hits += 1
            return self._blocks[index]
        self.misses += 1
        return None

    def put(self, index, likelihood, offset, *, to_host=None):
        if self._sealed or index != len(self._blocks):
            return
        # Retention granularity must not change the optimizer's reduction blocks.
        headers = (sys.getsizeof((None, None)) + sys.getsizeof(np.empty((0, 0)))
                   + sys.getsizeof(np.empty(0)))
        available = (self.budget - self._payload_bytes - headers
                     - sys.getsizeof(self._blocks + (None,)))
        row_bytes = (likelihood.shape[1] * likelihood.itemsize + offset.itemsize
                     if to_host is None else
                     likelihood.shape[1] * likelihood.element_size() + offset.element_size())
        rows = min(len(offset), max(0, available // row_bytes))
        if rows < len(offset):
            self._sealed = True
            self.partial_rows = rows
        if not rows:
            return
        if to_host is not None:
            # Do not transfer the uncached tail merely to reject it on the host.
            likelihood, offset = to_host((likelihood[:rows], offset[:rows]))
        # Copy a partial prefix: a view would keep the entire parent allocation.
        offset = np.array(offset[:rows], copy=True)
        if rows != len(likelihood) or not likelihood.flags.owndata:
            likelihood = likelihood[:rows].copy()
        record = (likelihood, offset)
        payload = sys.getsizeof(record) + sys.getsizeof(likelihood) + sys.getsizeof(offset)
        updated = self._blocks + (record,)
        size = sys.getsizeof(updated) + self._payload_bytes + payload
        assert size <= self.budget
        likelihood.setflags(write=False)
        offset.setflags(write=False)
        self._blocks = updated
        self._payload_bytes += payload
        self.retained_bytes = size
        self.cached_rows += rows

    def diagnostics(self, total_blocks):
        return {"budget_bytes": self.budget, "retained_bytes": self.retained_bytes,
                "cached_blocks": len(self._blocks) - bool(self.partial_rows),
                "cached_rows": self.cached_rows, "partial_rows": self.partial_rows,
                "total_blocks": total_blocks,
                "hits": self.hits, "misses": self.misses}


class _GridExecution:
    """Optional device reductions with one workspace-owned grid cache.

    Device retention is a deterministic block prefix inside the workspace's
    shared budget. No tensor reference is pinned here outside that owner.
    Evicted entries reload from the host cache or recompute from the model.
    SciPy control and optimization policy remain on the host.
    """

    def __init__(self, view, grid, block_rows):
        self.workspace = getattr(view.model, "workspace", None)
        self.grid = grid
        self.device_bytes = 0
        self.device_blocks = 0
        self.device_sealed = False
        self.device_hits = self.device_misses = 0
        if self.workspace is not None:
            # Grid emission blocks also cache compact observation tiles. Leave
            # room for this view's rows so streamed tail blocks do not evict the
            # admitted grid prefix on every sweep. Other phases still share the
            # workspace LRU; any external eviction has an exact host fallback.
            observation_bytes = len(view) * len(self.workspace._FIELDS) * 8
            self.device_budget = max(0, self.workspace.device_cache_bytes - observation_bytes)
            digest = hashlib.sha256(view.model.identity.encode())
            digest.update(view.rows.tobytes())
            digest.update(grid.tobytes())
            self.key = ("pooled-grid", view.region, block_rows, digest.hexdigest())

    def block(self, index, cache, rows, calculate):
        ws = self.workspace
        if ws is None:
            entry = cache.get(index)
            if entry is None:
                entry = calculate(0)
                cache.put(index, *entry)
            elif len(entry[1]) < rows:
                tail = calculate(len(entry[1]))
                entry = tuple(np.concatenate(pair) for pair in zip(entry, tail))
            return entry
        key = (*self.key, index)
        packed = ws.cache_get(key)
        if packed is not None:
            self.device_hits += 1
            if index == self.device_blocks:
                self.device_blocks += 1
                self.device_bytes += packed.untyped_storage().nbytes()
            return packed[:, :-1], packed[:, -1]
        self.device_misses += 1
        entry = cache.get(index)
        if entry is None:
            entry = calculate(0)
            if not cache._sealed:
                cache.put(index, *entry, to_host=ws.to_host)
        likelihood, offset = (ws.upload(part) for part in entry)
        if len(offset) < rows:
            tail, tail_offset = calculate(len(offset))
            likelihood = ws.torch.cat((likelihood, tail), dim=0)
            offset = ws.torch.cat((offset, tail_offset), dim=0)
        size = (likelihood.numel() + offset.numel()) * likelihood.element_size()
        if (index < self.device_blocks or not self.device_sealed
                and index == self.device_blocks
                and self.device_bytes + size <= self.device_budget):
            packed = ws.torch.cat((likelihood, offset[:, None]), dim=1)
            ws.cache_put(key, packed)
            if index == self.device_blocks:
                self.device_bytes += size
                self.device_blocks += 1
            return packed[:, :-1], packed[:, -1]
        self.device_sealed = True
        return likelihood, offset

    def calculate(self, view, grid, *, mask_bounds=False):
        ws = self.workspace
        if ws is None:
            values = view.log_likelihood_columns(grid)
            if mask_bounds:
                lower, upper = view.bounds()
                values[(grid[None, :] < lower[:, None]) | (grid[None, :] > upper[:, None])] = -np.inf
            offset = values.max(axis=1)
            if not np.isfinite(offset).all():
                raise RuntimeError("No finite grid likelihood for CCF initialization")
            return np.exp(values - offset[:, None]), offset
        values = ws.regional_grid_device(view.region, view.rows, grid)
        if mask_bounds:
            lower, upper = (ws.upload(part) for part in view.bounds())
            points = ws.upload(grid)
            values = values.masked_fill((points[None, :] < lower[:, None])
                                        | (points[None, :] > upper[:, None]), -ws.torch.inf)
        offset = values.max(dim=1).values
        if not ws._valid_scalar(ws.torch.isfinite(offset).all()):
            raise RuntimeError("No finite grid likelihood for CCF initialization")
        return (values - offset[:, None]).exp(), offset

    def expectation(self, blocks, weights, count, *, frequencies=None, positions=None,
                    collect_pilot=False, collect_mass=True):
        ws, grid = self.workspace, self.grid
        t = ws.torch
        weights = ws.upload(weights)
        score = t.zeros_like(weights)
        log_likelihood = t.zeros((), dtype=weights.dtype, device=weights.device)
        mass = np.empty(count) if collect_mass else None
        pilot = np.empty(count) if collect_pilot else None
        index = ws.upload(positions, dtype=t.long) if positions is not None else None
        moment = weights * ws.upload(grid) if collect_pilot else None
        for first, stop, likelihood, offset in blocks:
            if index is not None:
                likelihood = likelihood[:, index]
            normalizer = likelihood @ weights
            if ws._valid_scalar(((normalizer <= 0) | ~t.isfinite(normalizer)).any()):
                return None, -np.inf, None, None
            frequency = (t.ones_like(normalizer) if frequencies is None
                         else ws.upload(frequencies[first:stop]))
            score += (frequency / normalizer) @ likelihood
            log_likelihood += (frequency * (normalizer.log() + offset)).sum()
            if collect_mass:
                mass[first:stop] = ws.to_host(normalizer)
            if collect_pilot:
                pilot[first:stop] = ws.to_host(likelihood @ moment / normalizer)
        if not ws._valid_scalar(t.isfinite(score).all()):
            return None, -np.inf, None, None
        return ws.to_host(score), float(ws.to_host(log_likelihood)), mass, pilot

    def diagnostics(self):
        return {"backend": "cpu" if self.workspace is None else self.workspace.device.type,
                "device_grid_hits": self.device_hits, "device_grid_misses": self.device_misses,
                "device_grid_prefix_bytes": self.device_bytes,
                "device_grid_budget_bytes": 0 if self.workspace is None else self.device_budget}


def _initialization_grid(view):
    """Standard normal-CN-two grid policy, expressed entirely as CCFs.

    A component needs an extra proposal when its binomial-mode CCF standard
    error is smaller than one quarter of the base grid spacing. This numerical
    resolution criterion uses observed counts/CN only, never truth or K. A
    half-count floor handles variance estimates at binomial endpoints. Modes
    at CCF=0 or one are already included. If the proposal budget is exceeded,
    take evenly spaced ranks through sorted observations and multiplicities;
    report this deterministic compression rather than claim full resolution.
    """
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    threshold = 1 / (4 * (GRID_SIZE - 1))
    r, n, major = (view.array(name) for name in ("alt", "depth", "major"))
    vaf = r / n
    scaling = view.array("scaling")
    mode_numerator = vaf / scaling
    se_numerator = np.maximum(np.sqrt(vaf * (1 - vaf) / n), 0.5 / n) / scaling
    # CCF mode and its standard error both divide their numerator by m.
    # Compute eligible m ranges analytically; do not materialize N*major modes.
    bound = np.maximum(mode_numerator, se_numerator / threshold)
    first = np.minimum(major + 1, np.floor(bound) + 1).astype(np.int64)
    counts = np.maximum(0, major.astype(np.int64) - first + 1)
    counts[r == 0] = 0
    cumulative = np.cumsum(counts, dtype=np.int64)
    count = int(cumulative[-1])
    if count:
        ranks = np.linspace(0, count - 1, min(count, MAX_ADDITIONAL_GRID_POINTS), dtype=np.int64)
        row = np.searchsorted(cumulative, ranks, side="right")
        previous = np.r_[0, cumulative[:-1]]
        multiplicity = first[row] + ranks - previous[row]
        proposals = mode_numerator[row] / multiplicity
        grid = np.unique(np.r_[grid, proposals])
    return grid, {
        "narrow_component_modes": count,
        "mode_grid_compressed": count > MAX_ADDITIONAL_GRID_POINTS,
        "added_grid_points": len(grid) - GRID_SIZE,
    }


def pooled_ccf_initialization(view, *, cache_bytes=None):
    """Use the source-scoped grid/convergence policy on shared CCF emissions.

    Standard single-state normal-CN-two observations retain the bounded narrow-
    mode grid and reported iteration-budget behavior. Mixed/general-normal
    observations retain the bound-enriched grid and mandatory finite-grid gap.
    These numerical policies are explicit, not separate statistical models.
    """
    if not view.array("observed").all():
        raise ValueError("Pilot views must contain informative observations only")
    return (_pooled_standard(view, cache_bytes=cache_bytes) if view.standard
            else _pooled_adapter(view, cache_bytes=cache_bytes))


def _pooled_standard(view, *, cache_bytes=None):
    """Return posterior-mean CCFs and finite-grid optimization diagnostics.

    The base grid, resolution rule and optimization budget are data-independent.
    Narrow likelihood modes add bounded numerical proposals. Identical count/CN
    observations are collapsed in sorted order, preserving their frequencies;
    this also makes results exactly equivariant to input-row permutations.
    Large inputs stream likelihood blocks rather than allocate an unbounded
    N-by-grid matrix. Full 1..major support remains in the regional model.
    """
    observations = np.column_stack([view.array(name) for name in ("alt", "depth", "major", "total")])
    unique, indices, inverse, frequencies = np.unique(
        observations, axis=0, return_index=True, return_inverse=True, return_counts=True)
    unique_count = len(unique)
    grid, grid_diagnostics = _initialization_grid(view.subset(indices))
    grid_size = len(grid)
    frequencies = frequencies.astype(float)
    n = float(len(view))
    # Limit both the likelihood matrix and padded multiplicity temporaries.
    matrix_rows = max(1, MAX_MATRIX_BYTES // (8 * grid_size))
    support_rows = max(1, 1_000_000 // int(unique[:, 2].max()))
    block_rows = min(matrix_rows, support_rows)
    cache = _GridCache(cache_bytes)
    execution = _GridExecution(view.subset(indices), grid, block_rows)
    block_count = (unique_count + block_rows - 1) // block_rows
    grid_block_evaluations = 0
    grid_cell_evaluations = 0

    def calculate(first, stop, positions=None):
        nonlocal grid_block_evaluations, grid_cell_evaluations
        grid_block_evaluations += 1
        grid_cell_evaluations += (stop-first) * (len(grid) if positions is None else len(positions))
        block = view.subset(indices[first:stop])
        return execution.calculate(block, grid if positions is None else grid[positions])

    def blocks(positions=None):
        for index, first in enumerate(range(0, unique_count, block_rows)):
            stop = min(unique_count, first + block_rows)
            # The original large-grid streaming path rescales active-support
            # trials locally. Preserve that rule (including underflow behavior)
            # regardless of the new retention budget; these trials are not
            # interchangeable with slices of a full-grid-scaled cache block.
            entry = (calculate(first, stop, positions) if positions is not None else
                     execution.block(index, cache, stop-first,
                                     lambda skip: calculate(first+skip, stop)))
            likelihood, offset = entry
            yield first, stop, likelihood, offset

    evaluations = 0

    def expectation(weights, collect_pilot=False, positions=None):
        nonlocal evaluations
        evaluations += 1
        streamed_active = positions is not None and unique_count > matrix_rows
        if execution.workspace is not None:
            score, likelihood, _, pilot = execution.expectation(
                blocks(positions if streamed_active else None), weights, unique_count,
                frequencies=frequencies, positions=None if streamed_active else positions,
                collect_pilot=collect_pilot, collect_mass=False)
            return None if score is None else score / n, likelihood, pilot
        score = np.zeros(len(weights))
        log_likelihood = 0.0
        pilot = np.empty(unique_count) if collect_pilot else None
        for first, stop, likelihood, offset in blocks(positions if streamed_active else None):
            if positions is not None and not streamed_active:
                likelihood = likelihood[:, positions]
            normalizer = likelihood @ weights
            if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
                return None, -np.inf, None
            frequency = frequencies[first:stop]
            with np.errstate(over="ignore", invalid="ignore"):
                score += (frequency / normalizer) @ likelihood
            if not np.isfinite(score).all():
                return None, -np.inf, None
            log_likelihood += float(frequency @ (np.log(normalizer) + offset))
            if collect_pilot:
                pilot[first:stop] = likelihood @ (weights * grid) / normalizer
        return score / n, log_likelihood, pilot

    def normalize(candidate):
        candidate = np.maximum(candidate, 0.0)
        return candidate / candidate.sum()

    def exchange(weights, score):
        """Exact concave line search; zero-weight grid points can re-enter."""
        grow = int(np.argmax(score))
        positive = np.flatnonzero(weights > 0)
        shrink = int(positive[np.argmin(score[positive])])
        if grow == shrink:
            return weights
        mass, direction = np.empty(unique_count), np.empty(unique_count)
        ws = execution.workspace
        device_weights = ws.upload(weights) if ws is not None else None
        for first, stop, likelihood, _ in blocks():
            if ws is None:
                mass[first:stop] = likelihood @ weights
                direction[first:stop] = weights[shrink] * (likelihood[:, grow] - likelihood[:, shrink])
            else:
                mass[first:stop] = ws.to_host(likelihood @ device_weights)
                direction[first:stop] = ws.to_host(weights[shrink] * (likelihood[:, grow] - likelihood[:, shrink]))

        def derivative(fraction):
            denominator = np.maximum(mass + fraction * direction, np.finfo(float).tiny)
            return float((frequencies / n * direction / denominator).sum())

        if derivative(0.0) <= 0.0:
            return weights
        fraction = 1.0 if derivative(1.0) >= 0.0 else brentq(derivative, 0.0, 1.0, xtol=1e-14)
        candidate = weights.copy()
        transfer = fraction * weights[shrink]
        candidate[grow] += transfer
        candidate[shrink] = 0.0 if fraction == 1.0 else weights[shrink] - transfer
        return normalize(candidate)

    def correct_active_support(weights):
        # Solve only a small *existing* support. This is an acceleration, not
        # pruning: global grid scores and exchange steps still admit every CCF.
        active = np.flatnonzero(weights > 0)
        previous, value, gradient = None, None, None

        def objective(candidate):
            nonlocal previous, value, gradient
            if previous is None or not np.array_equal(candidate, previous):
                score, likelihood, _ = expectation(candidate, positions=active)
                previous = candidate.copy()
                value = -likelihood / n
                gradient = -score if score is not None else np.zeros(len(active))
            return value, gradient

        fit = minimize(
            objective,
            weights[active],
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(active),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(len(w))},
            options={"ftol": 1e-13, "maxiter": MAX_ACTIVE_REFIT_ITERATIONS},
        )
        candidate = weights.copy()
        if np.isfinite(fit.x).all() and np.maximum(fit.x, 0).sum() > 0:
            candidate[active] = normalize(fit.x)
        return candidate

    weights = np.full(grid_size, 1.0 / grid_size)
    score, log_likelihood, _ = expectation(weights)
    if score is None:
        raise RuntimeError("Nonfinite pooled CCF initialization likelihood")
    initial_log_likelihood = log_likelihood
    improvement = 0.0
    accelerated_steps = exchange_steps = active_refits = iteration = 0
    step_limit = 1.0
    for iteration in range(1, MAX_ITERATIONS + 1):
        if max(0.0, float(score.max() - 1.0)) <= GRID_OPTIMALITY_TOLERANCE:
            iteration -= 1
            break
        previous_log_likelihood = log_likelihood
        # Safeguarded SQUAREM extrapolates two EM updates. Negative entries
        # are clipped and renormalized; accepted iterates improve likelihood.
        # Exact zeros are allowed; the global-score exchange below restores
        # components that EM alone cannot reintroduce.
        first = normalize(weights * score)
        first_score, _, _ = expectation(first)
        second = normalize(first * first_score)
        second_score, second_likelihood, _ = expectation(second)
        residual = first - weights
        curvature = second - 2.0 * first + weights
        step = max(
            1.0,
            min(
                float(np.sqrt((residual @ residual) / max(curvature @ curvature, np.finfo(float).tiny))),
                step_limit,
            ),
        )
        candidate = normalize(weights + 2.0 * step * residual + step * step * curvature)
        candidate_score, candidate_likelihood, _ = expectation(candidate)
        while candidate_likelihood < second_likelihood and step > 1.01:
            step = (step + 1.0) / 2.0
            candidate = normalize(weights + 2.0 * step * residual + step * step * curvature)
            candidate_score, candidate_likelihood, _ = expectation(candidate)
        if candidate_likelihood >= second_likelihood:
            weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
            accelerated_steps += int(step > 1.0)
        else:
            weights, score, log_likelihood = second, second_score, second_likelihood
        if step >= 0.99 * step_limit:
            step_limit *= 4.0

        candidate = exchange(weights, score)
        candidate_score, candidate_likelihood, _ = expectation(candidate)
        if candidate_likelihood >= log_likelihood:
            exchange_steps += int(not np.array_equal(candidate, weights))
            weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
        if iteration % ACTIVE_REFIT_INTERVAL == 0 and np.count_nonzero(weights) <= MAX_ACTIVE_REFIT_SIZE:
            candidate = correct_active_support(weights)
            candidate_score, candidate_likelihood, _ = expectation(candidate)
            active_refits += 1
            if candidate_likelihood >= log_likelihood:
                weights, score, log_likelihood = candidate, candidate_score, candidate_likelihood
        improvement = (log_likelihood - previous_log_likelihood) / n
        if improvement < -1e-10 * max(1.0, abs(previous_log_likelihood) / n):
            raise RuntimeError("Pooled CCF initialization likelihood decreased")
    score, log_likelihood, pilot = expectation(weights, collect_pilot=True)
    # Concavity bounds the missing mean log likelihood by max(score)-1:
    # the weighted mean score equals one for every feasible mixture.
    gap = float(max(0.0, score.max() - 1.0))
    converged = gap <= GRID_OPTIMALITY_TOLERANCE
    pilot = np.clip(pilot[inverse], 0.0, 1.0)
    diagnostics = {
        "method": "regional_standard_ccf_grid_accelerated_simplex_v3",
        "role": "initialization_only_frozen_before_all_K",
        "grid_size": grid_size,
        "base_grid_size": GRID_SIZE,
        "max_additional_grid_points": MAX_ADDITIONAL_GRID_POINTS,
        **grid_diagnostics,
        "ccf_bounds": [0.0, 1.0],
        "num_mutations": len(view),
        "num_unique_observations": unique_count,
        "iterations": iteration,
        "max_iterations": MAX_ITERATIONS,
        "mean_loglik_tolerance": MEAN_LOGLIK_TOLERANCE,
        "mean_loglik_improvement": float(improvement),
        "converged_mean_loglik": bool(improvement <= MEAN_LOGLIK_TOLERANCE),
        "converged_grid_weights": converged,
        "termination_reason": "grid_duality_gap" if converged else "iteration_budget",
        "grid_optimality_tolerance": GRID_OPTIMALITY_TOLERANCE,
        "grid_score_excess": gap,
        "grid_weight_optimality_gap": n * gap,
        "grid_weight_optimum_certified": converged,
        "likelihood_evaluations": evaluations,
        "accelerated_steps": accelerated_steps,
        "exchange_steps": exchange_steps,
        "active_support_refits": active_refits,
        "max_active_refit_size": MAX_ACTIVE_REFIT_SIZE,
        "active_refit_interval": ACTIVE_REFIT_INTERVAL,
        "max_active_refit_iterations": MAX_ACTIVE_REFIT_ITERATIONS,
        "initial_log_likelihood": float(initial_log_likelihood),
        "log_likelihood": float(log_likelihood),
        "likelihood_matrix_cached": len(cache._blocks) == block_count and not cache.partial_rows,
        "max_matrix_bytes": MAX_MATRIX_BYTES,
        "constrained_optimum_certified": False,
        "grid_cache": {**cache.diagnostics(block_count),
                       "grid_block_evaluations": grid_block_evaluations,
                       "grid_cell_evaluations": grid_cell_evaluations,
                       **execution.diagnostics()},
    }
    return pilot, diagnostics


class _GridOracle:
    """Exact regional grid likelihood with byte-bounded immutable block reuse.

    The working block is at most ADAPTER_BLOCK_BYTES (or one grid row if that
    row itself exceeds the budget). Cached blocks have a separate explicit
    byte limit; zero restores streaming even for bound-enriched mixed CN.
    """
    def __init__(self, view, grid, *, block_bytes=None, cache_bytes=None):
        self.view, self.grid = view, grid
        budget = ADAPTER_BLOCK_BYTES if block_bytes is None else int(block_bytes)
        if budget < 8:
            raise ValueError("Grid block budget must hold at least one float64")
        self.block_rows = min(256, max(1, budget//(8*len(grid))),
                              max(1, 1_000_000//(32*view.array("valid").shape[-1])))
        self.offset = np.empty(len(view))
        self.cache = _GridCache(cache_bytes)
        self.execution = _GridExecution(view, grid, self.block_rows)
        self.block_count = (len(view) + self.block_rows - 1) // self.block_rows
        self.evaluations = 0
        self.grid_block_evaluations = 0
        self.grid_cell_evaluations = 0
        self.peak_block_bytes = 0
        for first in range(0, len(view), self.block_rows):
            stop = min(len(view), first+self.block_rows)
            if self.execution.workspace is not None:
                _, offset = self.execution.block(first // self.block_rows, self.cache, stop-first,
                                                 lambda skip: self.device_block(first+skip, stop))
                self.offset[first:stop] = self.execution.workspace.to_host(offset)
                continue
            values = self.log_block(first, stop)
            self.offset[first:stop] = values.max(axis=1)
            values -= self.offset[first:stop, None]
            np.exp(values, out=values)
            self.cache.put(first // self.block_rows, values, self.offset[first:stop])
        if not np.isfinite(self.offset).all():
            raise RuntimeError("No finite supported grid likelihood for regional initialization")

    def log_block(self, first, stop, positions=None):
        self.grid_block_evaluations += 1
        grid = self.grid if positions is None else self.grid[positions]
        self.grid_cell_evaluations += (stop-first) * len(grid)
        view = self.view.subset(np.arange(first, stop))
        values = np.empty((len(view), len(grid)))
        self.peak_block_bytes = max(self.peak_block_bytes, values.nbytes)
        for begin in range(0, len(grid), 32):
            values[:, begin:begin+32] = view.log_likelihood_columns(grid[begin:begin+32])
        lower, upper = view.bounds()
        values[(grid[None, :] < lower[:, None])
               | (grid[None, :] > upper[:, None])] = -np.inf
        return values

    def device_block(self, first, stop):
        self.grid_block_evaluations += 1
        self.grid_cell_evaluations += (stop-first) * len(self.grid)
        self.peak_block_bytes = max(self.peak_block_bytes, (stop-first)*len(self.grid)*8)
        return self.execution.calculate(self.view.subset(np.arange(first, stop)),
                                        self.grid, mask_bounds=True)

    def scaled_block(self, first, stop, positions=None):
        if self.execution.workspace is not None:
            likelihood, _ = self.execution.block(first // self.block_rows, self.cache, stop-first,
                                                lambda skip: self.device_block(first+skip, stop))
            if positions is not None:
                index = self.execution.workspace.upload(positions, dtype=self.execution.workspace.torch.long)
                likelihood = likelihood[:, index]
            return likelihood
        entry = self.cache.get(first // self.block_rows)
        if entry is not None and len(entry[1]) == stop-first:
            return entry[0] if positions is None else entry[0][:, positions]
        skip = 0 if entry is None else len(entry[1])
        likelihood = self.log_block(first+skip, stop, positions)
        likelihood -= self.offset[first+skip:stop, None]
        np.exp(likelihood, out=likelihood)
        if entry is not None:
            prefix = entry[0] if positions is None else entry[0][:, positions]
            likelihood = np.concatenate((prefix, likelihood))
        return likelihood

    def expectation(self, weights, *, collect_pilot=False):
        self.evaluations += 1
        if self.execution.workspace is not None:
            def blocks():
                for first in range(0, len(self.view), self.block_rows):
                    stop = min(len(self.view), first+self.block_rows)
                    likelihood, offset = self.execution.block(first // self.block_rows, self.cache, stop-first,
                                                              lambda skip: self.device_block(first+skip, stop))
                    yield first, stop, likelihood, offset
            score, likelihood, mass, pilot = self.execution.expectation(
                blocks(), weights, len(self.view), collect_pilot=collect_pilot)
            return None if score is None else score / len(self.view), likelihood, mass, pilot
        score, mass = np.zeros(len(weights)), np.empty(len(self.view))
        pilot = np.empty(len(self.view)) if collect_pilot else None
        log_likelihood = 0.
        for first in range(0, len(self.view), self.block_rows):
            stop = min(len(self.view), first+self.block_rows)
            likelihood = self.scaled_block(first, stop)
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
        return score/len(self.view), log_likelihood, mass, pilot

    def scaled_columns(self, positions):
        values = np.empty((len(self.view), len(positions)))
        for first in range(0, len(self.view), self.block_rows):
            stop = min(len(self.view), first+self.block_rows)
            block = self.scaled_block(first, stop, positions)
            if self.execution.workspace is not None:
                block = self.execution.workspace.to_host(block)
            values[first:stop] = block
        return values


def _pooled_adapter(view, *, block_bytes=None, cache_bytes=None):
    """Stream the explicit mixed/general-normal grid; exact concave weight gap.

    Safeguarded SQUAREM and exact grow/shrink exchange mirror the inherited
    pooled initializer without a quadratic-space SLSQP over the whole grid.
    Only this explicit numerical policy uses this route; standard observations
    retain their narrow-mode grid and declared iteration-budget behavior.
    Its masked grid scores reads jointly with assignment validity, without
    normalizing over admissible grid points.
    """
    lower, upper = view.bounds()
    grid = np.unique(np.r_[np.linspace(0., 1., 257), lower, upper])
    oracle = _GridOracle(view, grid, block_bytes=block_bytes, cache_bytes=cache_bytes)
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
                    objective_value = -value/len(view)
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
        "grid_score_excess": gap, "weight_optimality_gap": len(view)*gap,
        "grid_weight_optimum_certified": True,
        "initial_log_likelihood": initial_likelihood, "log_likelihood": likelihood,
        "likelihood_evaluations": oracle.evaluations, "accelerated_steps": accelerated,
        "exchange_steps": exchanges, "block_rows": oracle.block_rows,
        "active_support_refits": active_refits, "max_active_refit_size": 64,
        "peak_grid_block_bytes": oracle.peak_block_bytes,
        "dense_grid_matrix_retained": (len(oracle.cache._blocks) == oracle.block_count
                                       and not oracle.cache.partial_rows),
        "grid_cache": {**oracle.cache.diagnostics(oracle.block_count),
                       "grid_block_evaluations": oracle.grid_block_evaluations,
                       "grid_cell_evaluations": oracle.grid_cell_evaluations,
                       **oracle.execution.diagnostics()},
    }
