"""Explicit CPU research entry point; never substitutes a workflow AcceptedFit."""
from __future__ import annotations

import json
from pathlib import Path

from .io.data import OriginalCatalog, content_hash, json_value
from .io.assignment import load_accepted_fit
from .io.publication import publish_json
from .kernel.model import RegionalModel
from .kernel.mixture import search_marginal_mixture
from ._version import __version__


def run_mixture_experiment(catalog_file, accepted_fit_file, output, *,
                           max_clusters=None, proposal_budget=32, cpu_threads=1,
                           max_iterations=2000, gradient_tolerance=1e-5):
    """Reoptimize saved marginal centers/support under a distinct soft estimator.

    The saved input, population, support and originating fit are independently
    checked first. The output is a research JSON, not a workflow fit receipt:
    positive-weight mixture components need not have occupied MAP assignments.
    """
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    catalog = OriginalCatalog.from_dict(json.loads(Path(catalog_file).read_text()))
    accepted = load_accepted_fit(accepted_fit_file)
    original_config = accepted.validate_input_identity(catalog)
    if original_config.fit_multiplicity != "marginal":
        raise ValueError("Mixture experiments require a saved marginal fit; profile fits are a different estimator")
    accepted = accepted.validate(catalog, device="cpu", cpu_threads=cpu_threads,
                                 run_manifest_id="experimental-joint-marginal-mixture")
    capacity = original_config.max_clusters if max_clusters is None else max_clusters
    model = RegionalModel(catalog.fit_data(accepted.mutation_ids,
        normal_cn_policy=accepted.producer_receipt["normal_cn_policy"]), device="cpu",
        cpu_threads=cpu_threads, multiplicity_support=accepted.support,
        fit_multiplicity="marginal", selection_criterion=original_config.selection_criterion,
        max_multiplicity_difference=original_config.max_multiplicity_difference)
    try:
        search = search_marginal_mixture(model, [accepted], max_clusters=capacity,
            proposal_budget=proposal_budget, max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance)
        fit = search.best
        payload = json_value(dict(schema="clipp2.experimental-mixture.v1",
            estimator="experimental_joint_marginal_mixture", experimental=True,
            implementation_version=__version__, multiplicity_model=dict(model.method_identity),
            workflow_compatible=False, device="cpu", tumor_id=catalog.tumor_id,
            mutation_ids=model.mutation_ids, region_ids=model.region_ids,
            model_identity=model.identity, original_catalog_identity=content_hash(catalog.to_dict()),
            support=accepted.support.to_dict(), originating_accepted_state=accepted.state_fingerprint,
            originating_producer=accepted.producer_receipt,
            compatibility_check=accepted.compatibility_receipts[-1],
            settings=dict(max_clusters=capacity, proposal_budget=proposal_budget,
                          max_iterations=max_iterations, gradient_tolerance=gradient_tolerance),
            centers=fit.centers, weights=fit.weights, responsibilities=fit.responsibilities,
            map_labels=fit.labels, positive_components=fit.positive_components,
            occupied_map_clusters=fit.occupied_map_clusters, log_likelihood=fit.log_likelihood,
            score=fit.score, score_definition="-2*observed_mixture_LL+(K_positive*R+K_positive-1)*log(N)",
            converged=fit.converged, fit_diagnostics=fit.diagnostics,
            search_diagnostics=search.diagnostics,
            candidates=[dict(score=c.score, log_likelihood=c.log_likelihood,
                             positive_components=c.positive_components,
                             occupied_map_clusters=c.occupied_map_clusters) for c in search.candidates],
            execution=model.execution_diagnostics(),
            qualification_scope="experimental_numerical_fit_not_biological_or_global_optimality"))
        publish_json(target, json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return payload
    finally:
        model.close()
