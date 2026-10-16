"""Exact initialization emissions with independently budgeted resident storage."""
import numpy as np

from ..io.data import readonly_array


class GridOracle:
    """Regional grid oracle retaining the original numerical row/column blocks.

    Capacity controls retention only, never the grid, block shape, offsets or
    reduction order. Zero (the legacy default) streams; insufficient allowance
    also streams without partially retaining a matrix. The resident store is
    immutable and its accounting includes conservative container overhead.
    """

    def __init__(self, model, rows, region, grid, *, block_bytes=8 * 1024 * 1024,
                 cache_bytes=0):
        self.model, self.rows, self.region, self.grid = model, rows, region, grid
        budget = 8 * 1024 * 1024 if block_bytes is None else int(block_bytes)
        self.cache_bytes = int(cache_bytes)
        if budget < 8:
            raise ValueError("Grid block budget must hold at least one float64")
        if self.cache_bytes < 0:
            raise ValueError("Grid cache budget must be nonnegative")
        self.block_rows = min(256, max(1, budget // (8 * len(grid))),
                              max(1, 1_000_000 // (32 * model.valid.shape[-1])))
        self.offset = np.empty(len(rows))
        self.evaluations = self.peak_block_bytes = 0
        self.likelihood_blocks_computed = self.cache_hits = 0
        block_count = (len(rows) + self.block_rows - 1) // self.block_rows
        required = 8 * len(rows) * (len(grid) + 1) + 1024 * block_count + 256
        self.resident_bytes = required if required <= self.cache_bytes else 0
        self.cached = bool(self.resident_bytes)
        self._cache = [] if self.cached else None
        for first, stop in self._ranges():
            values = self.log_block(first, stop)
            self.offset[first:stop] = values.max(axis=1)
            if not np.isfinite(self.offset[first:stop]).all():
                raise RuntimeError("No finite supported grid likelihood for regional initialization")
            if self.cached:
                values -= self.offset[first:stop, None]
                np.exp(values, out=values)
                self._cache.append(readonly_array(values))
        self.offset = readonly_array(self.offset)

    def _ranges(self):
        for first in range(0, len(self.rows), self.block_rows):
            yield first, min(len(self.rows), first + self.block_rows)

    def log_block(self, first, stop, positions=None):
        grid = self.grid if positions is None else self.grid[positions]
        rows = self.rows[first:stop]
        values = np.empty((len(rows), len(grid)))
        self.peak_block_bytes = max(self.peak_block_bytes, values.nbytes)
        self.likelihood_blocks_computed += 1
        for begin in range(0, len(grid), 32):
            values[:, begin:begin + 32] = self.model._regional_likelihood(
                grid[begin:begin + 32], rows, self.region).T
        values[(grid[None, :] < self.model.lower[rows, self.region, None])
               | (grid[None, :] > self.model.upper[rows, self.region, None])] = -np.inf
        return values

    def _scaled_block(self, first, stop, positions=None):
        if self.cached:
            self.cache_hits += 1
            values = self._cache[first // self.block_rows]
            return values if positions is None else values[:, positions]
        values = self.log_block(first, stop, positions)
        values -= self.offset[first:stop, None]
        np.exp(values, out=values)
        return values

    def expectation(self, weights, *, collect_pilot=False):
        self.evaluations += 1
        score, mass = np.zeros(len(weights)), np.empty(len(self.rows))
        pilot = np.empty(len(self.rows)) if collect_pilot else None
        log_likelihood = 0.
        for first, stop in self._ranges():
            likelihood = self._scaled_block(first, stop)
            normalizer = likelihood @ weights
            if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
                return None, -np.inf, None, None
            mass[first:stop] = normalizer
            with np.errstate(over="ignore", invalid="ignore"):
                score += (1 / normalizer) @ likelihood
            log_likelihood += float((np.log(normalizer) + self.offset[first:stop]).sum())
            if collect_pilot:
                pilot[first:stop] = likelihood @ (weights * self.grid) / normalizer
        if not np.isfinite(score).all():
            return None, -np.inf, None, None
        return score / len(self.rows), log_likelihood, mass, pilot

    def scaled_columns(self, positions):
        values = np.empty((len(self.rows), len(positions)))
        for first, stop in self._ranges():
            values[first:stop] = self._scaled_block(first, stop, positions)
        return values
