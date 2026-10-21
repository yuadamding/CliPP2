"""Bounded float64 tensor emissions shared by regional numerical work.

The cache budget covers retained tensors; the workspace budget covers conservative
emission-tile intermediates, not caller-owned inputs or returned arrays. Both
mutation/grid axes and full multiplicity support are tiled. CPU tensors are an
explicit test emulator, never an implicit fallback from requested CUDA.
"""

from collections import OrderedDict

import numpy as np

from .likelihood import MIXED_EPS


class DeviceWorkspace:
    """One immutable model's observations, bounded caches and tensor operations."""

    _FIELDS = ("alt", "depth", "scaling", "support", "mixed", "observed",
               "constants", "lower", "upper")

    def __init__(self, model, *, device=None, device_cache_bytes=256 * 1024**2,
                 workspace_bytes=64 * 1024**2):
        import torch

        self.torch = torch
        self.model = model
        self.device = torch.device(model.device if device is None else device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Device workspace requires cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.device_cache_bytes = int(device_cache_bytes)
        self.workspace_bytes = int(workspace_bytes)
        if self.device_cache_bytes < 0 or self.workspace_bytes < 384:
            raise ValueError("Device cache must be nonnegative and workspace at least 384 bytes")
        self._cache = OrderedDict()
        self._cache_bytes = 0
        # Compact observation storage avoids a permanent N x R x M allocation.
        self._observations = np.stack([getattr(model, name) for name in self._FIELDS], axis=-1)
        self._observations.flags.writeable = False
        self._max_support = int(model.support.max())
        self.telemetry = {
            "host_to_device_calls": 0, "host_to_device_bytes": 0,
            "device_to_host_calls": 0, "device_to_host_bytes": 0,
            "tensor_upload_calls": 0, "tensor_upload_bytes": 0,
            "cache_hits": 0, "cache_misses": 0, "cache_evictions": 0,
            "cache_bytes": 0, "peak_cache_bytes": 0,
            "emission_tiles": 0, "max_tile_workspace_estimate_bytes": 0,
            "scalar_validation_calls": 0,
            "node_calls": 0, "joint_calls": 0, "regional_grid_calls": 0,
            "task_grid_calls": 0, "task_value_calls": 0,
        }

    def upload(self, value, dtype=None):
        """Convert explicitly, recording actual host-to-CUDA traffic only."""
        t = self.torch
        dtype = t.float64 if dtype is None else dtype
        if isinstance(value, t.Tensor):
            if value.device == self.device and value.dtype == dtype:
                return value
            result = value.to(device=self.device, dtype=dtype)
            from_host = value.device.type == "cpu"
        else:
            # Copy immutable NumPy views to avoid undefined tensor write aliases.
            result = t.tensor(np.asarray(value), device=self.device, dtype=dtype)
            from_host = True
        size = result.numel() * result.element_size()
        self.telemetry["tensor_upload_calls"] += 1
        self.telemetry["tensor_upload_bytes"] += size
        if from_host and self.device.type == "cuda":
            self.telemetry["host_to_device_calls"] += 1
            self.telemetry["host_to_device_bytes"] += size
        return result

    def to_host(self, value):
        """Publish only requested values; device math itself never returns NumPy."""
        if isinstance(value, (tuple, list)):
            return type(value)(self.to_host(part) for part in value)
        if value.device.type == "cuda":
            self.telemetry["device_to_host_calls"] += 1
            self.telemetry["device_to_host_bytes"] += value.numel() * value.element_size()
        return value.detach().cpu().numpy()

    def _valid_scalar(self, value):
        """Explicitly count the validation scalar's otherwise implicit transfer."""
        self.telemetry["scalar_validation_calls"] += 1
        return bool(self.to_host(value))

    def cache_get(self, key):
        result = self._cache.get(key)
        self.telemetry["cache_hits" if result is not None else "cache_misses"] += 1
        if result is not None:
            self._cache.move_to_end(key)
        return result

    def cache_put(self, key, value):
        """Retain a caller-owned tensor within the shared LRU byte ceiling."""
        if not isinstance(value, self.torch.Tensor) or value.device != self.device:
            raise ValueError("Only tensors on this workspace's device may be cached")
        # Count storage, not a view's visible elements: views can retain much more.
        size = value.untyped_storage().nbytes()
        old = self._cache.pop(key, None)
        if old is not None:
            self._cache_bytes -= old.untyped_storage().nbytes()
        if size <= self.device_cache_bytes:
            while self._cache_bytes + size > self.device_cache_bytes:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= removed.untyped_storage().nbytes()
                self.telemetry["cache_evictions"] += 1
            self._cache[key] = value
            self._cache_bytes += size
        self.telemetry["cache_bytes"] = self._cache_bytes
        self.telemetry["peak_cache_bytes"] = max(self.telemetry["peak_cache_bytes"], self._cache_bytes)
        return value

    def _validate_rows(self, region, rows):
        rows = np.asarray(rows)
        if (not isinstance(region, (int, np.integer)) or not 0 <= region < self.model.r
                or rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
                or np.any(rows < 0) or np.any(rows >= self.model.n)):
            raise ValueError("Invalid region or mutation row indices")
        return rows

    def _observed_rows(self, region, rows):
        rows = self._validate_rows(region, rows)
        key = ("observations", int(region), rows.tobytes())
        result = self.cache_get(key)
        if result is None:
            result = self.cache_put(key, self.upload(self._observations[rows, region]))
        return dict(zip(self._FIELDS, result.unbind(-1)))

    def _flat_observations(self, selected):
        key = ("flat_observations", selected.start, selected.stop)
        result = self.cache_get(key)
        if result is None:
            result = self.cache_put(key, self.upload(
                self._observations.reshape(-1, len(self._FIELDS))[selected]))
        return dict(zip(self._FIELDS, result.unbind(-1)))

    def _tiles(self, observations, grids, derivatives=False):
        # Includes observations, probabilities, posterior/derivative temporaries,
        # and partial reductions, conservatively rounded upward per support cell.
        factor = 384 if derivatives else 224
        support = min(self._max_support, 64, max(1, self.workspace_bytes // factor))
        pairs = max(1, self.workspace_bytes // (factor * support))
        width = min(grids, pairs)
        rows = min(observations, max(1, pairs // max(1, width)))
        if not rows or not width:
            return
        for first in range(0, observations, rows):
            for start in range(0, grids, width):
                yield slice(first, min(first + rows, observations)), slice(start, min(start + width, grids)), support, factor

    def _emit(self, ccf, a, support_chunk, factor, derivatives=False):
        """One broadcast G x P tile, reducing full support in bounded chunks."""
        t = self.torch
        observed = a["observed"].bool()
        x = t.where(observed[None, :], ccf, 0.)
        ll = t.full_like(x, -t.inf)
        gradient = t.zeros_like(x) if derivatives else None
        curvature = t.zeros_like(x) if derivatives else None
        boundary_g = t.zeros_like(x) if derivatives else None
        boundary_h = t.zeros_like(x) if derivatives else None
        alt, ref = a["alt"][None, :, None], (a["depth"] - a["alt"])[None, :, None]
        mixed = a["mixed"].bool()[None, :, None]
        for first in range(1, self._max_support + 1, support_chunk):
            stop = min(first + support_chunk, self._max_support + 1)
            m = t.arange(first, stop, device=self.device, dtype=t.float64)
            slope = a["scaling"][:, None] * m
            valid = m <= a["support"][:, None]
            mass = x[..., None] * slope
            p = t.where(mixed, mass.clamp(MIXED_EPS, 1 - MIXED_EPS), mass.clamp(max=1.))
            kernel = (t.special.xlogy(alt, p) + t.special.xlog1py(ref, -p)
                      + a["constants"][None, :, None])
            kernel = t.where(valid, kernel, -t.inf)
            partial = t.logsumexp(kernel, -1)
            combined = t.logaddexp(ll, partial)
            if derivatives:
                post = t.where(t.isfinite(kernel), t.exp(kernel - partial[..., None]), 0.)
                effective = t.where(mixed & ((mass <= MIXED_EPS) | (mass >= 1 - MIXED_EPS)), 0., slope)
                safe_p = t.where(alt > 0, p, 1.)
                safe_q = t.where(ref > 0, 1 - p, 1.)
                g = effective * (-alt / safe_p + ref / safe_q)
                h = effective.square() * (alt / safe_p.square() + ref / safe_q.square())
                g = t.where(post > 0, post * g, 0.).sum(-1)
                h = t.where(post > 0, post * h, 0.).sum(-1)
                old_weight = t.where(t.isfinite(ll), t.exp(ll - combined), 0.)
                new_weight = t.where(t.isfinite(partial), t.exp(partial - combined), 0.)
                gradient = gradient * old_weight + g * new_weight
                curvature = curvature * old_weight + h * new_weight
                boundary = valid & ~mixed & (p == 1)
                boundary_g += t.where(boundary & (ref == 1), slope, 0.).sum(-1)
                boundary_h += t.where(boundary & (ref == 2), 2 * slope.square(), 0.).sum(-1)
            ll = combined
            self.telemetry["emission_tiles"] += 1
            estimated = ccf.numel() * len(m) * factor
            self.telemetry["max_tile_workspace_estimate_bytes"] = max(
                self.telemetry["max_tile_workspace_estimate_bytes"], estimated)
        loss = t.where(observed[None, :], -ll, 0.)
        if not derivatives:
            return loss
        correction = t.exp(a["constants"][None, :] - ll)
        gradient += t.where(boundary_g > 0, boundary_g * correction, 0.)
        curvature += t.where(boundary_h > 0, boundary_h * correction, 0.)
        curvature = t.where(boundary_g > 0, t.inf, curvature.clamp(min=1e-8))
        gradient = t.where(t.isfinite(ll), gradient, 0.)
        curvature = t.where(t.isfinite(ll), curvature, 1.)
        return tuple(t.where(observed[None, :], value, 0.) for value in (loss, gradient, curvature))

    def _validate_ccf(self, ccf, observed=None):
        t = self.torch
        valid = t.isfinite(ccf) & (ccf >= 0) & (ccf <= 1)
        if observed is not None:
            valid |= ~observed
        # Explicit validation is a boundary synchronization, not a numerical
        # fallback; it preserves the existing finite/domain error contract.
        if not self._valid_scalar(t.all(valid)):
            raise ValueError("CCFs must be finite and in [0,1] where observed")

    def _ccf_valid(self, ccf, observed):
        t = self.torch
        return t.all((t.isfinite(ccf) & (ccf >= 0) & (ccf <= 1)) | ~observed)

    def node_terms_device(self, ccf, derivatives=True):
        t = self.torch
        ccf = self.upload(ccf)
        if ccf.shape != (self.model.n, self.model.r):
            raise ValueError("CCFs must match N by R")
        self.telemetry["node_calls"] += 1
        outputs = [t.empty_like(ccf, memory_format=t.contiguous_format)
                   for _ in range(3 if derivatives else 1)]
        valid = t.ones((), device=self.device, dtype=t.bool)
        # A region is an independent emission coordinate, not a launch boundary.
        # Flattening N x R keeps one ordinary node call in one kernel batch when
        # it fits, and tiles only when the explicit byte budget requires it.
        flat_ccf = ccf.reshape(-1)
        flat_outputs = [value.reshape(-1) for value in outputs]
        for selected, _, support, factor in self._tiles(self.model.n * self.model.r, 1, derivatives):
            a = self._flat_observations(selected)
            x = flat_ccf[selected][None, :]
            valid &= self._ccf_valid(x, a["observed"].bool()[None, :])
            values = self._emit(x, a, support, factor, derivatives)
            if not derivatives:
                values = (values,)
            for output, value in zip(flat_outputs, values):
                output[selected] = value[0]
        if not self._valid_scalar(valid):
            raise ValueError("CCFs must be finite and in [0,1] where observed")
        return tuple(outputs) if derivatives else outputs[0]

    def joint_columns_device(self, centers):
        t = self.torch
        centers = self.upload(centers)
        if centers.ndim != 2 or not len(centers) or centers.shape[1] != self.model.r:
            raise ValueError("Centers must be nonempty q by R CCFs")
        self._validate_ccf(centers)
        self.telemetry["joint_calls"] += 1
        result = t.zeros((self.model.n, len(centers)), device=self.device, dtype=t.float64)
        for selected, columns, support, factor in self._tiles(self.model.n, len(centers)):
            rows = np.arange(selected.start, selected.stop)
            values = t.zeros((columns.stop - columns.start, len(rows)), device=self.device, dtype=t.float64)
            allowed = t.ones_like(values, dtype=t.bool)
            for region in range(self.model.r):
                a = self._observed_rows(region, rows)
                x = centers[columns, region, None].expand(-1, len(rows))
                values -= self._emit(x, a, support, factor)
                allowed &= (x >= a["lower"]) & (x <= a["upper"])
            result[selected, columns] = t.where(allowed, values, -t.inf).T
        return result

    def regional_grid_device(self, region, rows, grid):
        t = self.torch
        grid = self.upload(grid)
        rows = self._validate_rows(region, rows)
        if grid.ndim != 1:
            raise ValueError("Regional proposal grid must be one-dimensional")
        # Validate even empty grids/rows before allocating a returned matrix.
        if len(rows) and self.model.observed[rows, region].any():
            self._validate_ccf(grid)
        self.telemetry["regional_grid_calls"] += 1
        result = t.empty((len(rows), len(grid)), device=self.device, dtype=t.float64)
        for selected, columns, support, factor in self._tiles(len(rows), len(grid)):
            a = self._observed_rows(region, rows[selected])
            x = grid[columns, None].expand(-1, selected.stop - selected.start)
            result[selected, columns] = -self._emit(x, a, support, factor).T
        return result
