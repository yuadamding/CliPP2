"""Tree-free production fitting and its explicit soft-component contract.

MAP labels are a reporting projection only: they neither define the fitted
objective nor cause a conditional center refit or removal of a positive slot.
"""
from dataclasses import dataclass
import math

import numpy as np

from .config import mixture_fitting_identity
from .kernel import ALGORITHM
from .kernel.mixture import search_marginal_mixture


def require_supported_production(config):
    """Reject unqualified optimizers before allocating a backend or output."""
    if config.device != "cpu":
        raise NotImplementedError("Tree-free production marginal-mixture fitting currently supports CPU only; CUDA is not qualified and has no CPU fallback")
    if config.fit_multiplicity != "marginal":
        raise NotImplementedError("Tree-free production fitting requires fit_multiplicity='marginal'; legacy profile fits may be replayed but are not silently refitted")


@dataclass(frozen=True)
class PublishedMixtureFit:
    """Publication view, retaining the actual soft fit without hard refitting."""
    mixture: object
    conditional_log_likelihood: float
    complexity_penalty: float
    method_identity: dict

    def __getattr__(self, name):
        return getattr(self.mixture, name)

    @property
    def fitting_log_likelihood(self):
        return self.mixture.log_likelihood

    @property
    def mixture_log_likelihood(self):
        return self.mixture.log_likelihood

    @property
    def fit_objective_id(self):
        return self.method_identity["fit_objective_id"]

    @property
    def selection_criterion_id(self):
        return self.method_identity["selection_criterion_id"]


@dataclass(frozen=True)
class MixtureSelection:
    fit: PublishedMixtureFit
    state_index: int
    membership_kind: str = "soft_mixture"


@dataclass(frozen=True)
class ProductionMixtureResult:
    search: object
    selected: MixtureSelection
    algorithm: str = ALGORITHM

    @property
    def centers(self):
        return self.selected.fit.centers

    @property
    def score(self):
        return self.selected.fit.score

    @property
    def diagnostics(self):
        return self.search.diagnostics


def fit_observations(model, max_clusters, *, progress=None):
    """Start directly from prepared observations, not an accepted tree fit."""
    if progress is not None:
        progress.start_phase("marginal_mixture_search")
    callback = None if progress is None else lambda record: progress.emit(operation=record)
    search = search_marginal_mixture(model, max_clusters=max_clusters, progress_callback=callback)
    best = search.best
    # One descriptive hard-label likelihood is retained for old table/replay
    # consumers. It is not the fitting objective and is never optimized here.
    columns = model.selection_column_set(best.centers).materialize()
    conditional = math.fsum(columns[np.arange(model.n), best.labels])
    identity = mixture_fitting_identity(model.method_identity["selection_criterion_id"], model.max_multiplicity_difference)
    fit = PublishedMixtureFit(best, conditional,
        (best.positive_components * (model.r + 1) - 1) * math.log(model.n), identity)
    index = next(i for i, candidate in enumerate(search.candidates) if candidate is best)
    if progress is not None:
        progress.emit(complete=True)
    return ProductionMixtureResult(search, MixtureSelection(fit, index))


def mixture_execution_diagnostics(model):
    """Keep native evidence while naming the optimizer that actually ran."""
    record = model.execution_diagnostics()
    record.update(algorithm=ALGORITHM, backend="native_cpu_marginal_mixture",
                  conditional_refit_backend=None, continuation=None,
                  center_execution="cpu_soft_joint_em", weight_execution="cpu",
                  method_identity=mixture_fitting_identity(model.method_identity["selection_criterion_id"],
                      model.max_multiplicity_difference), hard_refits=0)
    return record
