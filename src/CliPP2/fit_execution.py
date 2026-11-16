"""Fit and publish one native workflow stage."""
from dataclasses import dataclass

from .config import FitConfig
from .io.data import MultiplicityBounds, readonly_array
from .io.reporting import check_destination, public_label_order, publish
from .kernel.model import RegionalModel
from .production import fit_observations, require_supported_production
from .tree.progress import ProgressReporter


@dataclass(frozen=True)
class FitExecution:
    fit: object
    mutation_ids: tuple
    region_ids: tuple
    public_order: object
    public_labels: object
    multiplicities: object
    support: MultiplicityBounds


def fit_prepared(data, outdir, config, *, multiplicity_support=None, progress_callback=None):
    if type(config) is not FitConfig:
        raise TypeError('fit_config must be a FitConfig')
    require_supported_production(config)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    check_destination(outdir, data.tumor_id)
    with RegionalModel(data, device=config.device, cpu_threads=config.cpu_threads,
            multiplicity_support=multiplicity_support, fit_multiplicity=config.fit_multiplicity,
            selection_criterion=config.selection_criterion,
            max_multiplicity_difference=config.max_multiplicity_difference) as model:
        progress = None if progress_callback is None else ProgressReporter(model, progress_callback)
        fit = fit_observations(model, config.max_clusters, progress=progress)
        publish(model, fit, outdir, data.tumor_id, progress=progress)
        order, labels = public_label_order(fit)
        support = MultiplicityBounds(model.mutation_ids, model.region_ids,
            model.multiplicity_min, model.multiplicity_max,
            'original_support' if multiplicity_support is None else multiplicity_support.policy_id)
        return FitExecution(fit, model.mutation_ids, model.region_ids, readonly_array(order),
            readonly_array(labels), readonly_array(model.multiplicity(fit.centers[fit.labels])), support)
