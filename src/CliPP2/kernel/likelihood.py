"""Lightweight regional views over the native emission backend."""
import numpy as np

from ..io.data import readonly_array

MIXED_EPS = 1e-6


class RegionalView:
    """One model coordinate and selected mutation rows, always on CCF scale."""

    def __init__(self, model, region_index, mutation_indices):
        rows = np.asarray(mutation_indices)
        if (not isinstance(region_index, (int, np.integer))
                or not 0 <= region_index < model.r
                or rows.ndim != 1 or not len(rows)
                or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= model.n)):
            raise ValueError("A regional view needs a valid region and nonempty row indices")
        self.model, self.region = model, int(region_index)
        self.rows = readonly_array(rows, dtype=int)

    def __len__(self):
        return len(self.rows)

    def array(self, name):
        return getattr(self.model, name)[self.rows, self.region]

    @property
    def standard(self):
        """Numerical pilot policy, not a different emission model."""
        return not self.array("mixed").any() and np.all(self.array("normal") == 2)

    def subset(self, indices):
        return RegionalView(self.model, self.region, self.rows[indices])

    def bounds(self):
        return self.array("lower"), self.array("upper")

    def log_likelihood_columns(self, grid):
        """N-by-grid read emissions; assignment bounds are applied by callers."""
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("Regional proposal grid must be one-dimensional")
        # Preserve the row-major BLAS reduction layout, including recomputed
        # suffixes of partially retained initialization blocks.
        return np.ascontiguousarray(self.model.backend.regional_grid(self.region, self.rows, grid))
