"""Shared CCF emission primitive and lightweight regional coordinate views.

A view owns no second likelihood or CP parameterization. The same emission
primitive evaluates full N-by-R states, coordinate refits and pilot grids.
"""

import numpy as np
from scipy.special import logsumexp, xlog1py, xlogy

from ..io.data import readonly_array


MIXED_EPS = 1e-6


def emission_terms(ccf, *, slope, valid, mixed, alt, depth, constants):
    """Broadcast CCFs against observation arrays; the final axis is support.

    Inputs have already been validated by the regional model or its view.
    Single-state support is never truncated; mixed-state clipping is unchanged.
    """
    mass = np.asarray(ccf, dtype=float)[..., None] * slope
    p = np.where(mixed[..., None], np.clip(mass, MIXED_EPS, 1-MIXED_EPS),
                 np.minimum(1., mass))
    kernel = (xlogy(alt[..., None], p)
              + xlog1py((depth-alt)[..., None], -p) + constants[..., None])
    kernel = np.where(valid, kernel, -np.inf)
    return p, mass, kernel, logsumexp(kernel, axis=-1)


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
        """Numerical pilot/refit policy, not a different emission model."""
        return not self.array("mixed").any() and np.all(self.array("normal") == 2)

    @property
    def purity(self):
        return float(self.model.region_purities[self.region])

    def subset(self, indices):
        return RegionalView(self.model, self.region, self.rows[indices])

    def bounds(self):
        return self.array("lower"), self.array("upper")

    def _terms(self, ccf):
        ccf = np.asarray(ccf, dtype=float)
        ccf = np.broadcast_to(ccf, np.broadcast_shapes(ccf.shape, (len(self),)))
        observed = np.broadcast_to(self.array("observed"), ccf.shape)
        if (np.any(~np.isfinite(ccf[observed]))
                or np.any(ccf[observed] < 0) or np.any(ccf[observed] > 1)):
            raise ValueError("Regional CCFs must be finite and in [0,1] where observed")
        return emission_terms(np.where(observed, ccf, 0.), **{name: self.array(name) for name in
                              ("slope", "valid", "mixed", "alt", "depth", "constants")})

    def log_likelihood(self, ccf):
        ccf = np.broadcast_to(np.asarray(ccf, dtype=float), (len(self),))
        return np.where(self.array("observed"), self._terms(ccf)[-1], 0.)

    def log_likelihood_columns(self, grid):
        """N-by-grid read emissions; assignment bounds are applied by callers."""
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("Regional proposal grid must be one-dimensional")
        values = np.empty((len(self), len(grid)))
        batch = max(1, min(32, 1_000_000 // max(1, self.array("valid").size)))
        for first in range(0, len(grid), batch):
            values[:, first:first+batch] = self._terms(grid[first:first+batch, None])[-1].T
        return np.where(self.array("observed")[:, None], values, 0.)

    def grid_log_likelihood(self, grid):
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("Regional proposal grid must be one-dimensional")
        result = np.empty(len(grid))
        batch = max(1, min(32, 1_000_000 // max(1, self.array("valid").size)))
        for first in range(0, len(grid), batch):
            ll = self._terms(grid[first:first+batch, None])[-1]
            result[first:first+batch] = np.where(self.array("observed"), ll, 0.).sum(axis=1)
        return result

    def component_modes(self):
        observed = self.array("observed")
        vaf = np.divide(self.array("alt"), self.array("depth"),
                        out=np.zeros(len(self)), where=observed)
        return (vaf[:, None] / self.array("slope"))[
            self.array("valid") & observed[:, None]]

    def posterior(self, ccf):
        ccf = np.broadcast_to(np.asarray(ccf, dtype=float), (len(self),))
        _, _, kernel, ll = self._terms(ccf)
        observed = self.array("observed")
        if not np.isfinite(ll[observed]).all():
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.where(observed[:, None], np.exp(kernel-ll[:, None]), np.nan)
