"""Fixed-center, fixed-partition and complete-workflow multiplicity comparisons.

Truth is used only by evaluation and by explicitly declared fixed-state probes.
The two full workflows receive the same reads and settings, differing only in
the regional multiplicity restriction. Reports measure small qualification
fixtures; they are not a claim of cohort-wide scientific superiority.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from CliPP2.config import FitConfig, WorkflowConfig
from CliPP2.io.data import json_value
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.kernel.model import RegionalModel
from CliPP2.simulation.evaluation import partition_scores


MODELS = {"independent": None, "range_one": 1}


def _measure(operation):
    started = time.perf_counter()
    own_trace = not tracemalloc.is_tracing()
    if own_trace:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        value = operation()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if own_trace:
            tracemalloc.stop()
    return value, dict(elapsed_seconds=time.perf_counter()-started,
                       python_peak_increment_bytes=max(0, peak-baseline),
                       memory_scope="Python allocations; native ownership reported separately")


def _truth_metrics(mutations, regions, labels, phi, calls, observed, truth):
    if truth is None:
        return dict(truth_available=False)
    table = pd.read_csv(truth, sep="\t", dtype={"mutation_id": str, "region_id": str, "truth_cluster": str})
    if table.duplicated(["mutation_id", "region_id"]).any():
        raise ValueError("Truth must have one row per mutation-region")
    grouped = table.groupby("mutation_id", sort=False)
    if (grouped.truth_cluster.nunique() != 1).any():
        raise ValueError("A mutation must have one true cluster across regions")
    truth_labels = grouped.truth_cluster.first().to_dict()
    present = [i for i, mutation in enumerate(mutations) if mutation in truth_labels]
    resolved = [i for i in present if labels[i] >= 0]
    true = [truth_labels[mutations[i]] for i in resolved]
    predicted = [int(labels[i]) for i in resolved]
    unique_truth = len({truth_labels[mutations[i]] for i in present})
    unique_fit = len({int(label) for label in labels if label >= 0})
    lookup = table.set_index(["mutation_id", "region_id"])
    ccf_errors, multiplicity_errors = [], []
    multiplicity_correct = multiplicity_total = 0
    for i, mutation in enumerate(mutations):
        for r, region in enumerate(regions):
            if (mutation, region) not in lookup.index or not observed[i, r]:
                continue
            row = lookup.loc[mutation, region]
            if not bool(row.get("observed", 1)):
                continue
            if np.isfinite(row.true_ccf) and np.isfinite(phi[i, r]):
                ccf_errors.append(abs(float(phi[i, r])-float(row.true_ccf)))
            m = row.get("true_multiplicity", np.nan)
            if row.true_ccf > 0 and np.isfinite(m) and m > 0:
                multiplicity_total += 1
                if np.isfinite(calls[i, r]):
                    error = abs(float(calls[i, r])-float(m))
                    multiplicity_errors.append(error)
                    multiplicity_correct += int(error == 0)
    return dict(truth_available=True, true_cluster_count=unique_truth,
        occupied_cluster_count=unique_fit, cluster_count_error=unique_fit-unique_truth,
        membership_ARI=partition_scores(true, predicted)["ari"] if resolved else None,
        membership_evaluated_mutations=len(resolved), membership_unresolved_mutations=len(present)-len(resolved),
        ccf_mean_absolute_error=float(np.mean(ccf_errors)) if ccf_errors else None,
        ccf_evaluated_pairs=len(ccf_errors),
        multiplicity_accuracy=multiplicity_correct/multiplicity_total if multiplicity_total else None,
        multiplicity_mean_absolute_error=float(np.mean(multiplicity_errors)) if multiplicity_errors else None,
        multiplicity_truth_pairs=multiplicity_total, multiplicity_defined_pairs=len(multiplicity_errors))


def _write(output, result):
    with (Path(output) / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(json_value(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def compare_data(data, output, config, *, scope, requests, support=None, truth=None):
    """Use identical centers or memberships and hard feasible supports in both models."""
    if scope not in ("fixed-centers", "fixed-partitions") or not requests:
        raise ValueError("Supply fixed-centers or fixed-partitions and nonempty requests")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope=scope, config=asdict(config), models={},
                  inference_truth_access=False, same_regional_support=True)
    for name, difference in MODELS.items():
        def execute():
            with RegionalModel(data, config.device, cpu_threads=config.cpu_threads,
                    multiplicity_support=support, fit_multiplicity=config.fit_multiplicity,
                    max_multiplicity_difference=difference) as model:
                records = []
                for request in requests:
                    if scope == "fixed-centers":
                        labels = np.asarray(request["labels"], dtype=int)
                        centers = np.asarray(request["centers"], dtype=float)
                        score = model.score_frozen_centers(centers, labels)
                        columns = model.selection_column_set(centers).materialize()
                        assigned = np.argmax(columns, axis=1)
                        assigned[~np.isfinite(columns).any(axis=1)] = -1
                        calls = model.multiplicity(centers[np.maximum(assigned, 0)])
                        phi = centers[np.maximum(assigned, 0)].copy()
                        phi[assigned < 0] = calls[assigned < 0] = np.nan
                        record = dict(request=request, selection=score,
                            equal_prior_assignment_labels=assigned, multiplicities=calls,
                            assignment_definition="fixed-center marginal MAP with equal reference priors")
                    else:
                        labels = np.asarray(request["labels"] if isinstance(request, dict) else request, dtype=int)
                        start = request.get("centers") if isinstance(request, dict) else None
                        fit = model.refit(labels, start)
                        assigned, centers = fit.labels, fit.centers
                        phi = centers[assigned]
                        calls = model.multiplicity(phi)
                        record = dict(labels=assigned, centers=centers, multiplicities=calls,
                            eligible=fit.eligible, fitting_log_likelihood=fit.fitting_log_likelihood,
                            marginal_mixture_log_likelihood=fit.mixture_log_likelihood,
                            BIC_penalty=fit.complexity_penalty, BIC_score=fit.score)
                    record["truth"] = _truth_metrics(model.mutation_ids, model.region_ids,
                        assigned, phi, calls, model.observed, truth)
                    records.append(record)
                return dict(method_identity=dict(model.method_identity), records=records,
                    mutation_ids=model.mutation_ids, region_ids=model.region_ids,
                    lower=model.multiplicity_min, upper=model.multiplicity_max,
                    native_host_bytes=model.backend.native_host_bytes,
                    execution=model.execution_diagnostics())
        value, timing = _measure(execute)
        result["models"][name] = dict(value, **timing)
    return _write(output, result)


def _support_report(directory, max_multiplicity_difference):
    result = {}
    for stage, filename in (("stage_one", "multiplicity_lower_bounds.tsv"),
                            ("stage_two", "stage2_effective_multiplicity_bounds.tsv")):
        path = directory / filename
        if not path.exists():
            continue
        rows = pd.read_csv(path, sep="\t")
        informative = rows["informative"].astype(str).str.lower().isin(("true", "1"))
        adjusted = rows["adjustment_reason"] != "unchanged"
        count = int(informative.sum())
        result[stage] = dict(informative_pairs=count,
            adjusted_pairs=int((informative & adjusted).sum()),
            adjustment_frequency=float((informative & adjusted).sum()/count) if count else None,
            reasons=rows.loc[informative & adjusted, "adjustment_reason"].value_counts().to_dict())
    path = directory / "stage2_segment_multiplicity_bounds.tsv"
    if path.exists():
        result["caps"] = pd.read_csv(path, sep="\t").to_dict(orient="records")
        for cap in result["caps"]:
            cap["max_multiplicity_difference"] = max_multiplicity_difference
    return result


def compare_workflows(input_path, output, config, *, truth=None):
    """Rerun discovery, cap construction and reassignment separately for both policies."""
    from CliPP2.io.assignment import load_assignment_bundle
    from CliPP2.workflow import run_workflow
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope="workflow", config=asdict(config), models={}, inference_truth_access=False)
    for name, difference in MODELS.items():
        directory = output / name
        settings = replace(config, fit=replace(config.fit, max_multiplicity_difference=difference))
        receipt, timing = _measure(lambda: run_workflow(input_path, directory, settings))
        record = dict(receipt=receipt, support=_support_report(directory, difference), **timing)
        if receipt["status"] == "complete":
            bundle = load_assignment_bundle(directory / "assignment_bundle.json")
            table = pd.read_csv(directory / "all_original_mutations.tsv", sep="\t").set_index("mutation_id")
            prefix = bundle.catalog.tumor_id + "_"
            regional = pd.read_csv(directory / "assignment" / (prefix + "assignment_multiplicity.tsv"), sep="\t")
            regional = regional.set_index(["mutation_id", "region_id"])
            labels = np.array([table.loc[mutation, "final_assignment_label"] for mutation in bundle.target_ids])
            labels = np.where(np.isfinite(labels), labels, -1).astype(int)
            phi, calls = [np.full((len(bundle.target_ids), len(bundle.reference.region_ids)), np.nan) for _ in range(2)]
            for i, mutation in enumerate(bundle.target_ids):
                for r, region in enumerate(bundle.reference.region_ids):
                    row = regional.loc[mutation, region]
                    phi[i, r], calls[i, r] = row.phi, row.assigned_multiplicity
            data = bundle.catalog.prediction_data(bundle.target_ids)
            observed = data.count_observed & ~data.input_excluded & (data.total_counts > 0)
            record["truth"] = _truth_metrics(bundle.target_ids, bundle.reference.region_ids,
                                             labels, phi, calls, observed, truth)
            record["final_labels"] = labels
            record["final_multiplicities"] = calls
        result["models"][name] = record
    return _write(output, result)


def run_qualification(output, *, device="cpu"):
    """Small paired scientific probes, including intentional model misspecification."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = {
        "equal": ([3, 3], [2, 2], [[1., .9], [.4, .3]]),
        "difference_one": ([3, 3], [1, 2], [[1., .9], [.4, .3]]),
        "difference_greater_one": ([3, 3], [1, 3], [[1., .9], [.4, .3]]),
        "zero_ccf_discordant_cn": ([1, 5], [1, 4], [[1., 0.], [.3, .6]]),
        "conflicting_floors_caps": ([1, 6], [1, 5], [[1., .95], [.5, .55]]),
        "already_allowed": ([2, 2], [1, 2], [[1., .9], [.4, .3]]),
    }
    summary = dict(schema="clipp2.joint-multiplicity.qualification.v1", scenarios={},
                   scope="small deterministic fixtures; not a full biological cohort")
    config = FitConfig(device=device, max_clusters=2, cpu_threads=1)
    rng = np.random.default_rng(4118)
    for name, (major, multiplicity, centers) in scenarios.items():
        directory = output / name
        directory.mkdir()
        centers, major, multiplicity = np.asarray(centers), np.asarray(major), np.asarray(multiplicity)
        labels = np.repeat(np.arange(2), 4)
        depth = 200
        alt = rng.binomial(depth, centers[labels] * multiplicity / major)
        rows, truth = [], []
        for i, label in enumerate(labels):
            for r in range(2):
                mutation, region = f"m{i:03}", f"r{r+1}"
                rows.append(dict(mutation_id=mutation, sample_id=region,
                    alt_count=int(alt[i, r]), ref_count=int(depth-alt[i, r]), count_observed=1,
                    purity=1., normal_cn=2, segment_id="shared", cn_state_id="0", cn_state_fraction=1.,
                    allele_a_cn=int(major[r]), allele_b_cn=0))
                truth.append(dict(mutation_id=mutation, region_id=region, truth_cluster=int(label),
                    true_ccf=centers[label, r], true_multiplicity=int(multiplicity[r]), observed=1))
        path = write_tumor_txt(directory / "input.tsv", pd.DataFrame(rows), {"tumor_id": name})
        truth_path = directory / "truth.tsv"
        pd.DataFrame(truth).to_csv(truth_path, sep="\t", index=False)
        catalog = load_original_catalog(path)
        state = dict(labels=labels.tolist(), centers=centers.tolist())
        fixed = compare_data(catalog.fit_data(), directory / "fixed_centers", config,
            scope="fixed-centers", requests=[state], truth=truth_path)
        partitioned = compare_data(catalog.fit_data(), directory / "fixed_partitions", config,
            scope="fixed-partitions", requests=[state], truth=truth_path)
        workflows = compare_workflows(path, directory / "workflows", WorkflowConfig(fit=config,
            stage_two_max_clusters=2), truth=truth_path)
        # Explicit cap stress complements whatever donors the small workflow selects.
        from CliPP2.io.multiplicity import read_floor_bounds, reconcile_workflow_bounds
        floors, _ = read_floor_bounds(catalog)
        _, adjustments = reconcile_workflow_bounds(catalog, floors,
            np.ones(catalog.major_cn.shape, dtype=int), stage="stage_two_stress")
        summary["scenarios"][name] = dict(
            true_regional_difference=int(multiplicity.max()-multiplicity.min()),
            declared_model_misspecification=bool(multiplicity.max()-multiplicity.min() > 1),
            fixed_centers={key: value["records"][0]["truth"] for key, value in fixed["models"].items()},
            fixed_partitions={key: value["records"][0]["truth"] for key, value in partitioned["models"].items()},
            workflows={key: dict(truth=value.get("truth"), support=value["support"],
                elapsed_seconds=value["elapsed_seconds"], python_peak_increment_bytes=value["python_peak_increment_bytes"])
                for key, value in workflows["models"].items()},
            raw_cap_one_stress_adjustments=adjustments)
    return _write(output, summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("fixed-centers", "fixed-partitions", "workflow", "qualification"), required=True)
    parser.add_argument("--requests", help="JSON states {labels,centers} or label lists for fixed partitions")
    parser.add_argument("--truth", help="TSV: mutation_id,region_id,truth_cluster,true_ccf,true_multiplicity,observed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fit-multiplicity", choices=("marginal", "profile"), default="marginal")
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--stage-two-max-clusters", type=int, default=20)
    args = parser.parse_args()
    config = FitConfig(device=args.device, cpu_threads=1, max_clusters=args.max_clusters,
                       fit_multiplicity=args.fit_multiplicity)
    if args.scope == "qualification":
        result = run_qualification(args.output_dir, device=args.device)
    elif args.input is None:
        parser.error("--input is required outside qualification")
    elif args.scope == "workflow":
        result = compare_workflows(args.input, args.output_dir, WorkflowConfig(fit=config,
            stage_two_max_clusters=args.stage_two_max_clusters), truth=args.truth)
    else:
        if args.requests is None:
            parser.error("--requests is required for fixed comparisons")
        result = compare_data(load_original_catalog(args.input).fit_data(), args.output_dir, config,
            scope=args.scope, requests=json.loads(Path(args.requests).read_text()), truth=args.truth)
    print(json.dumps(dict(scope=result.get("scope", "qualification"), output=str(Path(args.output_dir).resolve()))))
