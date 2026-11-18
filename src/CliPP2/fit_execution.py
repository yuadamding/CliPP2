"""Independently validate a selected Protocol 3.0 fit before publication."""

import numpy as np

from .config import MODEL_ID, PROTOCOL_VERSION
from .io.clonal_reporting import OUTPUT_SCHEMA, evaluate_fit, publish_stage
from .io.data import content_hash


def stage_record(model, fit, result, catalog, stage):
    certificate = {
        name: fit.diagnostics[name]
        for name in (
            "requested_template",
            "qualified_template",
            "iterations",
            "projected_gradient_residual",
            "gradient_tolerance",
            "weight_optimality_gap",
            "weight_active_score_gap",
        )
    }
    record = dict(
        schema="clipp2.stage-fit.v2",
        protocol_version=PROTOCOL_VERSION,
        model=MODEL_ID,
        output_schema=OUTPUT_SCHEMA,
        stage=stage,
        tumor_id=catalog.tumor_id,
        catalog_content_hash=catalog.content_hash,
        mutation_ids=model.mutation_ids,
        region_ids=model.region_ids,
        biological_count=fit.biological_count,
        occupied_count=len(set(result.labels)),
        component_labels=result.component_labels,
        labels=result.labels,
        centers=result.centers,
        weights=result.weights,
        clonal_prior_beta=result.clonal_prior_beta,
        internal_to_biological=result.internal_to_biological,
        internal_template=fit.template,
        internal_weights=fit.weights,
        free_centers=fit.centers,
        log_likelihood=float(fit.log_likelihood),
        parameter_count=result.parameter_count,
        bic=float(fit.bic),
        multiplicities=result.multiplicities,
        stationarity=certificate,
    )
    return record | {
        "state_identity": content_hash(
            {k: v for k, v in record.items() if k != "stage"}
        )
    }


def publish_selected(model, fit, catalog, outdir, stage, *, selected_from=None):
    result = evaluate_fit(model, fit, catalog=catalog)
    record = stage_record(model, fit, result, catalog, stage)
    if selected_from is not None:
        prior_assignment, prior_record = selected_from
        if record["state_identity"] != prior_record[
            "state_identity"
        ] or not np.array_equal(
            result.responsibilities, prior_assignment.responsibilities
        ):
            raise ValueError(
                "Final biological assignments differ from the accepted Stage 2 model."
            )
    tables = publish_stage(
        model, fit, result, outdir, include_probabilities=selected_from is not None
    )
    return result, record, tables
