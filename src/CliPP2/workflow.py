"""Protocol 3.0: discover, refine the same fit, publish biological posteriors."""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ._version import __version__
from .config import MODEL_ID, PROTOCOL_VERSION, WorkflowConfig
from .fit_execution import publish_selected
from .io.data import json_value
from .io.publication import publish_json, publish_tsv
from .io.tumor_txt import load_original_catalog
from .kernel.clonal_mixture import ClonalModel
from .production import discover, refine
from .provenance import prepare_run_manifest


def _write(path, object_):
    return publish_json(
        path,
        json.dumps(json_value(object_), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def _accounting(catalog, model, first, final):
    ids = {m: i for i, m in enumerate(model.mutation_ids)}
    rows = []
    for j, mutation in enumerate(catalog.mutation_ids):
        i = ids.get(mutation)
        if i is None:
            reason = (
                "no_informative_reads"
                if catalog.eligible[j].any()
                else "all_pairs_ineligible"
            )
            rows.append(
                (catalog.tumor_id, mutation, None, None, None, "unassigned", reason)
            )
        else:
            label = int(final.labels[i])
            rows.append(
                (
                    catalog.tumor_id,
                    mutation,
                    int(first.labels[i]),
                    label,
                    label,
                    "assigned",
                    "clonal" if label == 0 else "subclonal",
                )
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "tumor_id",
            "mutation_id",
            "stage_one_label",
            "stage_two_label",
            "stage_three_label",
            "status",
            "reason",
        ],
    )
    for name in ("stage_one_label", "stage_two_label", "stage_three_label"):
        frame[name] = frame[name].astype("Int64")
    if (
        len(rows) != len(catalog.mutation_ids)
        or int(frame.status.eq("assigned").sum()) != model.n
    ):
        raise ValueError(
            "Original population accounting does not match the fitted population."
        )
    return frame


def run_workflow(
    input_path, outdir, workflow_config, *, progress_callback=None, run_manifest_id=None
):
    """Use one immutable observation model; never fit in Stage 3."""
    if type(workflow_config) is not WorkflowConfig:
        raise TypeError("workflow_config must be a WorkflowConfig.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    config = workflow_config
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        schema="clipp2.workflow.v8",
        protocol_version=PROTOCOL_VERSION,
        implementation_version=__version__,
        model=MODEL_ID,
        config=asdict(config),
    )
    stage = "input"

    def callback(name):
        if progress_callback is None:
            return None
        return lambda event: progress_callback(dict(event, stage=name))

    try:
        catalog = load_original_catalog(
            input_path,
            max_major_cn=config.fit.max_major_cn,
            missing_counts=config.missing_counts,
        )
        receipt["run_manifest"] = prepare_run_manifest(
            catalog, run_manifest_id=run_manifest_id
        )
        receipt["read_counts"] = catalog.read_count_report
        _write(output / "original_catalog.json", catalog.to_dict())
        informative = (
            catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
        )
        unsupported = [
            r for r, ok in zip(catalog.region_ids, informative.any(axis=0)) if not ok
        ]
        if unsupported:
            raise ValueError(
                f"Requested regions without informative observations: {unsupported}."
            )
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
                raise ValueError(
                    "Observation preparation changed the frozen fitting population."
                )
            stage = "stage_one"
            first = discover(model, config.fit, progress_callback=callback(stage))
            first_assignment, first_record, _ = publish_selected(
                model, first.best, catalog, output / stage, stage
            )
            _write(output / (stage + "_accepted.json"), first_record)
            _write(output / (stage + "_search.json"), first.diagnostics)
            receipt["stage_one_bic"] = float(first.best.bic)
            stage = "stage_two"
            second = refine(model, first, config, progress_callback=callback(stage))
            second_assignment, second_record, _ = publish_selected(
                model, second.best, catalog, output / stage, stage
            )
            _write(output / (stage + "_accepted.json"), second_record)
            _write(output / (stage + "_search.json"), second.diagnostics)
            receipt["stage_two_bic"] = float(second.best.bic)
            stage = "stage_three"
            final, final_record, _ = publish_selected(
                model,
                second.best,
                catalog,
                output / stage,
                stage,
                selected_from=(second_assignment, second_record),
            )
            _write(
                output / (stage + "_accepted.json"),
                final_record
                | {
                    "refit": False,
                    "assignment_policy": "fitted_biological_posterior",
                    "clonal_population_active": 0 in final.component_labels,
                    "selected_stage_two_state_identity": second_record[
                        "state_identity"
                    ],
                    "multiplicity_posterior_mean": final.multiplicity_posterior_mean,
                    "multiplicity_call_probability": final.multiplicity_call_probability,
                    "joint_multiplicity_call_probability": final.joint_call_probability,
                    "conditional_clonal_single_copy_term_probability": final.eta,
                    "conditional_clonal_all_one_probability": final.all_one_probability,
                },
            )
        accounting = _accounting(catalog, model, first_assignment, final)
        publish_tsv(output / "all_original_mutations.tsv", accounting)
        receipt.update(
            status="completed",
            tumor_id=catalog.tumor_id,
            original_mutations=len(accounting),
            fitted_mutations=model.n,
            unassigned_mutations=len(accounting) - model.n,
            biological_clusters=second.best.biological_count,
            occupied_clusters=len(set(final.labels)),
            assigned_clonal=int(np.sum(final.labels == 0)),
            stage_three_refit=False,
            final_state_identity=final_record["state_identity"],
            stage_one_mandatory_coverage_complete=not first.diagnostics[
                "incomplete_counts"
            ],
            stage_two_mandatory_coverage_complete=not second.diagnostics[
                "incomplete_counts"
            ],
        )
        _write(output / "result.json", receipt)
        return receipt
    except Exception as error:
        _write(
            output / "failure.json",
            receipt
            | {
                "status": "failed",
                "failed_stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
