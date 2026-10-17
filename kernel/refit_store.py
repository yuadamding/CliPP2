"""Bounded exact refit-grid emissions, without changing a block's search grid.

CP and CCF are distinct coordinates. Scalar columns also retain the original
padded multiplicity width, so cache capacity cannot change logsumexp arithmetic.
Off-grid continuous optimization never uses this store.
"""
import numpy as np

from .likelihood import MultiplicityModel
from ..io.data import readonly_array


class ExactRefitStore:
    """Retain demanded regular-grid columns; mode evaluations remain block-local.

Each column holds every mutation in canonical order. Missing observations are
zero, not zero-read evidence. The allowance includes conservative key/array/LRU
overhead in addition to FP64 payloads; temporary baseline-sized reduction arrays
are separate. A zero allowance invokes the original grid evaluation directly.
"""

    def __init__(self, model, budget_bytes):
        self.model = model
        self.budget_bytes = int(budget_bytes)
        if self.budget_bytes < 0:
            raise ValueError("Refit-grid budget must be nonnegative")
        self.bytes = 0
        self._peak_bytes = 0
        self._regular = {}
        self.stats = {"hits": 0, "columns": 0, "evictions": 0,
                      "requests": 0, "uncached_columns": 0}

    def _get(self, key):
        if key in self._regular:
            self.stats["hits"] += 1
            return self._regular[key]
        return None

    @property
    def column_bytes(self):
        # Includes array header, tuple/integers, dictionary node and alignment.
        return 8*self.model.n + 512

    @property
    def diagnostics(self):
        return dict(self.stats, budget_bytes=self.budget_bytes, live_bytes=self.bytes,
                    peak_bytes=self._peak_bytes, regular_columns=len(self._regular),
                    mode_columns=0)

    def _can_store(self):
        # Stable regular admission avoids repeatedly generating full-region
        # columns when only a fraction of one grid can fit.
        available = self.budget_bytes-len(self._regular)*self.column_bytes
        return self.column_bytes <= available

    def _put(self, key, values):
        if key in self._regular or not self._can_store():
            return
        column = readonly_array(values, dtype=float)
        self._regular[key] = column
        self.bytes += self.column_bytes
        self._peak_bytes = max(self._peak_bytes, self.bytes)

    def _full_columns(self, grid, region, width):
        model = self.model
        allowed = model.observed[:, region]
        if width:
            allowed = allowed & (model.normal[:, region] == 2) & (model.major[:, region] <= width)
        rows = np.flatnonzero(allowed)
        result = np.zeros((len(grid), model.n))
        # Bound evaluation workspaces independently of the resident allowance.
        batch = max(1, 1_000_000//max(1, len(grid)*int(model.support.max())))
        for first in range(0, len(rows), batch):
            active = rows[first:first+batch]
            if width:
                scalar = MultiplicityModel(model.alt[active, region], model.depth[active, region],
                    model.major[active, region], model.total[active, region], model.region_purities[region])
                # A row tile can lack the maximum-support observation. Padding
                # must still match the requesting block, not that tile's width.
                multiplicities = np.arange(1, width+1)
                scalar.valid = multiplicities[None, :] <= scalar.major[:, None]
                scalar.scale = multiplicities[None, :] / (
                    2*(1-scalar.purity) + scalar.purity*scalar.total[:, None])
                result[:, active] = scalar.grid_columns(grid)
            else:
                result[:, active] = model._regional_likelihood(grid, active, region)
        self.stats["columns"] += len(grid)
        return result

    def grid_log_likelihood(self, rows, region, grid, *, scalar=None):
        """Return sums in the original member order and original grid batches.

Pass the exact block ``MultiplicityModel`` for a CP grid; omit it for a CCF
grid. Callers retain their own 513 points plus component modes. No global union,
interpolation, prefix differences or regrouped component sums are introduced.
        """
        rows, grid = np.asarray(rows, dtype=int), np.asarray(grid, dtype=float)
        if not self.budget_bytes:
            self.stats["requests"] += len(grid)
            self.stats["uncached_columns"] += len(grid)
            if scalar is not None:
                return scalar.grid_log_likelihood(grid)
            return np.concatenate([self.model._regional_likelihood(grid[i:i+32], rows, region).sum(axis=1)
                                   for i in range(0, len(grid), 32)])
        width = 0 if scalar is None else int(scalar.major.max())
        batch = 32 if scalar is None else max(1, min(32, 1_000_000//max(1, scalar.valid.size)))
        upper = 1. if scalar is None else scalar.purity
        regular = np.linspace(0., upper, 513)
        result = np.empty(len(grid))
        for first in range(0, len(grid), batch):
            points = grid[first:first+batch]
            block = np.empty((len(points), len(rows)))
            cold, direct = [], []
            planned_regular = len(self._regular)
            for j, value in enumerate(points):
                key = (int(region), width, int(value.view(np.uint64)))
                cached = self._get(key)
                self.stats["requests"] += 1
                if cached is not None:
                    block[j] = cached[rows]
                    continue
                index = np.searchsorted(regular, value)
                is_regular = index < len(regular) and regular[index] == value
                available = self.budget_bytes-planned_regular*self.column_bytes
                # A first-use mode of a small block must not trigger a full-N
                # calculation. Mode columns stay exact block-local evaluations
                # until a separately budgeted reuse policy is qualified.
                if is_regular and self.column_bytes <= available:
                    cold.append((j, key))
                    planned_regular += 1
                else:
                    direct.append(j)
            if cold:
                # A small block may request columns for a much larger region.
                # Do not let its 32-point batch allocate 32 full-region vectors
                # when even one column is already several MiB.
                tile = max(1, min(len(cold), (8*1024**2)//max(1, 8*self.model.n)))
                for begin in range(0, len(cold), tile):
                    requested = cold[begin:begin+tile]
                    columns = self._full_columns(points[[j for j, _ in requested]], region, width)
                    for column, (j, key) in zip(columns, requested):
                        block[j] = column[rows]
                        self._put(key, column)
            if direct:
                values = points[direct]
                block[direct] = (scalar.grid_columns(values) if scalar is not None
                                 else self.model._regional_likelihood(values, rows, region))
                self.stats["uncached_columns"] += len(direct)
            result[first:first+batch] = block.sum(axis=1)
        return result
