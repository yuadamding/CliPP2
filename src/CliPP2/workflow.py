"""Protocol: fit one mixture and publish verified biological assignments."""

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from threadpoolctl import threadpool_limits

from ._version import __version__
from .config import MODEL_ID, PROTOCOL_VERSION, WorkflowConfig
from .fit_execution import publish_selected
from .io.tumor_txt import load_original_catalog
from .kernel.clonal_mixture import ClonalModel
from .kernel.clonal_search import search
from .provenance import prepare_run_manifest


def run_workflow(
    input_path, outdir, workflow_config, *, progress_callback=None, run_manifest_id=None
):
    """Publish only the Stage 1 tables and return the fit summary in memory."""
    if type(workflow_config) is not WorkflowConfig:
        raise TypeError("workflow_config must be a WorkflowConfig.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    config = workflow_config
    output = Path(outdir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")

    def callback(event):
        progress_callback(dict(event, stage="stage_one"))

    catalog = load_original_catalog(
        input_path,
        max_major_cn=config.fit.max_major_cn,
        missing_counts=config.missing_counts,
    )
    manifest = prepare_run_manifest(catalog, run_manifest_id=run_manifest_id)
    read_counts = catalog.read_count_report
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    unsupported = [
        r for r, ok in zip(catalog.region_ids, informative.any(axis=0)) if not ok
    ]
    if unsupported:
        raise ValueError(f"Requested regions without informative observations: {unsupported}.")
    population = tuple(
        m for m, keep in zip(catalog.mutation_ids, informative.any(axis=1)) if keep
    )
    data = catalog.fit_data(population, normal_cn_policy=config.normal_cn_policy)
    limits = (
        nullcontext()
        if config.fit.cpu_threads is None
        else threadpool_limits(limits=config.fit.cpu_threads, user_api="blas")
    )
    with limits:
        model = ClonalModel(data)
        if set(model.mutation_ids) != set(population) or set(
            model.region_ids
        ) != set(catalog.region_ids):
            raise ValueError("Observation preparation changed the frozen fitting population.")
        selected = search(
            model, max_clusters=config.fit.max_clusters,
            proposal_budget=config.fit.proposal_budget,
            progress_callback=callback if progress_callback is not None else None,
        )
        assignment, record, _ = publish_selected(model, selected.best, catalog, output)
    return dict(
        schema="clipp2.workflow.v9",
        protocol_version=PROTOCOL_VERSION,
        implementation_version=__version__,
        model=MODEL_ID,
        config=asdict(config),
        run_manifest=manifest,
        read_counts=read_counts,
        status="completed",
        tumor_id=catalog.tumor_id,
        original_mutations=len(catalog.mutation_ids),
        fitted_mutations=model.n,
        unassigned_mutations=len(catalog.mutation_ids) - model.n,
        biological_clusters=selected.best.biological_count,
        occupied_clusters=record["occupied_count"],
        assigned_clonal=int((assignment.labels == 0).sum()),
        stage_one_bic=record["bic"],
        final_state_identity=record["state_identity"],
        stage_one_mandatory_coverage_complete=not selected.diagnostics["incomplete_counts"],
    )
