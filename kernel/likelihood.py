"""CliPP2 scalar likelihood primitive, using cellular prevalence internally."""
# Adapted from CliPP1.5; see ../NOTICE.

import math
import numpy as np
from scipy.special import gammaln, logsumexp, xlog1py, xlogy


class MultiplicityModel:
    """Single-sample binomial mixture with fixed, uniform multiplicity prior."""

    def __init__(self, r, n, major, total, purity):
        arrays = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (r, n, major, total)]
        if (
            any(x.ndim != 1 for x in arrays)
            or not arrays[0].size
            or any(x.shape != arrays[0].shape for x in arrays)
            or any(not np.all(np.isfinite(x)) or np.any(x != np.rint(x)) for x in arrays)
        ):
            raise ValueError("Counts and copy numbers must be matching finite integer vectors")
        self.r, self.n, self.major, self.total = arrays
        self.purity = float(purity)
        if (
            not math.isfinite(self.purity)
            or not 0 < self.purity <= 1
            or np.any(self.r < 0)
            or np.any(self.n <= 0)
            or np.any(self.r > self.n)
            or np.any(self.major < 1)
            or np.any(self.total < self.major)
        ):
            raise ValueError("Invalid counts, major/total copy numbers, or purity")
        self.major = self.major.astype(int)
        self.m = np.arange(1, int(self.major.max()) + 1)
        self.valid = self.m[None, :] <= self.major[:, None]
        self.scale = self.m[None, :] / (2 * (1 - self.purity) + self.purity * self.total[:, None])
        self.log_constant = (
            gammaln(self.n + 1) - gammaln(self.r + 1) - gammaln(self.n - self.r + 1) - np.log(self.major)
        )

    def __len__(self):
        return len(self.r)

    def subset(self, indices):
        return MultiplicityModel(
            self.r[indices], self.n[indices], self.major[indices], self.total[indices], self.purity
        )

    def _log_kernel(self, cp):
        cp = np.broadcast_to(np.asarray(cp, dtype=float), (len(self),))
        if np.any(~np.isfinite(cp)) or np.any(cp < 0) or np.any(cp > self.purity):
            raise ValueError("Cellular prevalence must be finite and in [0, purity]")
        # Padded states are masked; clip their p before evaluating logarithms.
        p = np.minimum(1.0, cp[:, None] * self.scale)
        kernel = (
            xlogy(self.r[:, None], p) + xlog1py((self.n - self.r)[:, None], -p) + self.log_constant[:, None]
        )
        return np.where(self.valid, kernel, -np.inf)

    def log_likelihood(self, cp):
        """Per-mutation marginal log likelihood, including binomial constants."""
        return logsumexp(self._log_kernel(cp), axis=1)

    def posterior(self, cp):
        """Conditional probabilities for m=1,...,max(major), with padded zeros."""
        kernel = self._log_kernel(cp)
        normalizer = logsumexp(kernel, axis=1)
        if not np.all(np.isfinite(normalizer)):
            raise ValueError("Multiplicity posterior is undefined at zero likelihood")
        return np.exp(kernel - normalizer[:, None])

    def grid_log_likelihood(self, grid):
        """Evaluate center proposals in bounded-memory batches."""
        result = np.empty(len(grid))
        batch = max(1, min(32, 1_000_000 // max(1, self.valid.size)))
        for start in range(0, len(grid), batch):
            cp = grid[start : start + batch]
            p = np.minimum(1.0, cp[:, None, None] * self.scale[None, :, :])
            kernel = (
                xlogy(self.r[None, :, None], p)
                + xlog1py((self.n - self.r)[None, :, None], -p)
                + self.log_constant[None, :, None]
            )
            kernel = np.where(self.valid[None, :, :], kernel, -np.inf)
            result[start : start + batch] = logsumexp(kernel, axis=2).sum(axis=1)
        return result
