"""Fresh A initialization followed by B's joint marginal-mixture fitting."""
from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from .io.data import readonly_array
from .kernel.mixture import search_marginal_mixture
from .kernel.model import RegionalModel
from .tree.selection import fit_regions


def require_supported_production(config):
    if config.device != "cpu":
        raise NotImplementedError("A-initialized B fitting currently supports CPU only; CUDA is not qualified and has no CPU fallback")
    if config.fit_multiplicity != "marginal":
        raise NotImplementedError("A-initialized B fitting requires fit_multiplicity='marginal'; Stage 3 profile assignment is separate")


@dataclass(frozen=True)
class PublishedMixtureFit:
    """Publication view retaining B's soft fit without a hard-label refit."""
    mixture: object
    conditional_log_likelihood: float
    complexity_penalty: float

    def __getattr__(self, name):
        return getattr(self.mixture, name)

    @property
    def mixture_log_likelihood(self):
        return self.mixture.log_likelihood


def fit_observations(model, max_clusters, *, progress=None):
    """Fit A afresh on this stage's exact population/support, then optimize B."""
    if progress is not None:
        progress.start_phase("A_initializer")
    callback = None if progress is None else lambda record: progress.emit(operation=record)
    # Separate owners release A's tree/refit caches before B optimization.
    with RegionalModel(model.data, device=model.device, cpu_threads=model.backend.cpu_threads,
            multiplicity_support=model.multiplicity_support, fit_multiplicity=model.fit_multiplicity,
            selection_criterion=model.method_identity["selection_criterion_id"],
            max_multiplicity_difference=model.max_multiplicity_difference) as initial_model:
        if initial_model.identity != model.identity:
            raise ValueError("A initializer and B must have identical observations and multiplicity support")
        initial = fit_regions(initial_model, max_clusters=max_clusters, progress_callback=callback)
        selected = initial.selected.fit
        seed = SimpleNamespace(centers=readonly_array(selected.centers), weights=readonly_array(selected.weights))
    del initial, selected, initial_model
    if progress is not None:
        progress.emit(complete=True)
        progress.start_phase("marginal_mixture_search")
    search = search_marginal_mixture(model, [seed], max_clusters=max_clusters, progress_callback=callback)
    execution = model.execution_diagnostics()
    if execution["continuation"] is not None or execution["host_cache_entries"]["refit"]:
        raise ValueError("B optimization must not perform forest continuation or hard partition refits")
    best = search.best
    columns = model.selection_column_set(best.centers).materialize()
    conditional = math.fsum(columns[np.arange(model.n), best.labels])
    fit = PublishedMixtureFit(best, conditional,
        (best.positive_components * (model.r + 1) - 1) * math.log(model.n))
    if progress is not None:
        progress.emit(complete=True)
    return fit
