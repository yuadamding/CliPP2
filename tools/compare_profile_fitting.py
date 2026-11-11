"""Source-bound research comparisons of profile fitting / marginal selection.

Truth, when supplied, is consumed only after fitting. No method-default change,
cohort launch, retries, threshold relaxation, or inference truth access occurs.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from CliPP2.config import FitConfig, WorkflowConfig
from CliPP2.io.data import json_value, content_hash
from CliPP2.io.reporting import independent_log_columns, _verify_fitted_numerics, publish
from CliPP2.io.tumor_txt import load_original_catalog
from CliPP2.kernel.model import RegionalModel
from CliPP2.tree.selection import fit_regions


def source_record():
    from CliPP2.kernel.build_identity import backend_identity, native_identity
    import CliPP2
    root = Path(CliPP2.__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".cu", ".cuh")}
    return dict(package=str(root), inventory=files, interpreter=__import__("sys").executable,
                native_build_id=native_identity(), cpu_build_id=backend_identity("cpu"),
                cuda_build_id=backend_identity("cuda"))


def biological_metrics(model, labels, centers, truth):
    if truth is None:
        return dict(center_error=None, membership_accuracy=None, false_merges=None, false_splits=None)
    from CliPP2.simulation.evaluation import partition_scores
    table = pd.read_csv(truth, sep="\t", dtype={"mutation_id": str, "region_id": str, "truth_cluster": str})
    if table.duplicated(["mutation_id", "region_id"]).any():
        raise ValueError("Truth must be unique by mutation/region")
    grouped = table.groupby("mutation_id", sort=False)
    if (grouped.truth_cluster.nunique() != 1).any():
        raise ValueError("Truth cluster must be shared across regions")
    mapping = grouped.truth_cluster.first().to_dict()
    rows = [i for i, m in enumerate(model.mutation_ids) if m in mapping]
    true = [mapping[model.mutation_ids[i]] for i in rows]
    pred = np.asarray(labels)[rows]
    _, truth_labels = np.unique(true, return_inverse=True)
    q = int(pred.max()+1) if len(pred) else 0
    counts = np.zeros((len(set(true)), q), dtype=np.int64)
    if len(pred):
        np.add.at(counts, (truth_labels, pred), 1)
    def pairs(a):
        return int(np.sum(a*(a-1)//2))
    tp = pairs(counts)
    errors = []
    lookup = table.set_index(["mutation_id", "region_id"])
    for i in rows:
        for r, region in enumerate(model.region_ids):
            key = model.mutation_ids[i], region
            if key not in lookup.index or not model.observed[i, r]:
                continue
            row = lookup.loc[key]
            if not bool(row.get("observed", 1)) or not np.isfinite(row.true_ccf):
                continue
            errors.append(abs(float(centers[labels[i], r])-float(row.true_ccf)))
    return dict(center_error=float(np.mean(errors)) if errors else None,
                center_error_pairs=len(errors), membership_accuracy=partition_scores(true, pred)["ARI"] if rows else None,
                membership_accuracy_definition="ARI", false_merges=pairs(counts.sum(axis=0))-tp,
                false_splits=pairs(counts.sum(axis=1))-tp, evaluated_mutations=len(rows))


def describe(model, fitted, truth=None):
    if np.isfinite(fitted.mixture_log_likelihood):
        _verify_fitted_numerics(model, fitted, eligible=fitted.eligible)
    profile = independent_log_columns(model, fitted.centers, aggregation="profile")
    return dict(occupied_K=len(fitted.centers), method_identity=dict(model.method_identity),
        centers=fitted.centers, labels=fitted.labels, weights=fitted.weights,
        eligible=fitted.eligible, reason=fitted.reason,
        fitting_log_likelihood=fitted.fitting_log_likelihood,
        profile_fitting_log_likelihood=float(profile[np.arange(model.n), fitted.labels].sum()),
        marginal_conditional_log_likelihood=fitted.conditional_log_likelihood,
        marginal_mixture_log_likelihood=fitted.mixture_log_likelihood,
        BIC_penalty=fitted.complexity_penalty, BIC_score=fitted.score,
        **biological_metrics(model, fitted.labels, fitted.centers, truth))


def compare_data(data, output, config, *, scope="search", partitions=None, states=None,
                 support=None, truth=None):
    """A/B/C scopes share the exact retained population, supports and pilot/tree."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(scope=scope, config=asdict(config), source=source_record(),
                  source_selection=None, cap_changes=None, searches={}, comparisons=[])
    with RegionalModel(data, config.device, cpu_threads=config.cpu_threads, multiplicity_support=support) as marginal, \
            RegionalModel(data, config.device, cpu_threads=config.cpu_threads, multiplicity_support=support,
                          fit_multiplicity="profile") as profile:
        models = dict(marginal=marginal, profile=profile)
        result.update(mutation_ids=marginal.mutation_ids, region_ids=marginal.region_ids,
                      actual_lower=marginal.multiplicity_min, actual_upper=marginal.multiplicity_max,
                      numerical_input_hash=content_hash({key: getattr(marginal, key) for key in
                        ("mutation_ids", "region_ids", "alt", "depth", "observed", "scaling", "lower", "upper",
                         "multiplicity_min", "multiplicity_max")}))
        if scope == "fixed-centers":
            if not states:
                raise ValueError("Fixed-center scope requires explicit states")
            for state in states:
                labels, centers = np.asarray(state["labels"], dtype=int), np.asarray(state["centers"], dtype=float)
                scores = {method: model.score_frozen_centers(centers, labels) for method, model in models.items()}
                for key in ("centers", "weights", "score", "conditional_log_likelihood", "mixture_log_likelihood"):
                    np.testing.assert_array_equal(scores["marginal"][key], scores["profile"][key])
                result["comparisons"].append(dict(state=state, frozen_selection=scores["profile"],
                    fitting_ll={method: float(model.fit_column_set(centers).assigned_values(labels).sum())
                                for method, model in models.items()}))
        else:
            pool = {} if partitions is None else {tuple(map(int, labels)): [] for labels in partitions}
            if scope == "search":
                np.testing.assert_array_equal(marginal.pilots()[0], profile.pilots()[0])
                trees = []
                for method, model in models.items():
                    start = time.perf_counter()
                    fitted = fit_regions(model, config.max_clusters, audit_scope=config.audit_scope)
                    elapsed = time.perf_counter()-start
                    publication = publish(model, fitted, output/method, data.tumor_id)
                    trees.append(fitted.tree.identity)
                    result["searches"][method] = dict(describe(model, fitted.selected.fit, truth),
                        elapsed_seconds=elapsed, publication=publication, tree_identity=fitted.tree.identity,
                        search_policy=fitted.diagnostics["search_policy"])
                    for candidate in fitted.candidate_bank:
                        pool.setdefault(tuple(map(int, candidate.fit.labels)), []).append(method+":eligible")
                    for outcome in fitted.attempt_outcomes:
                        if outcome.outcome == "completed_ineligible" and outcome.fit is not None:
                            pool.setdefault(tuple(map(int, outcome.fit.labels)), []).append(method+":completed_ineligible")
                assert len(set(trees)) == 1, "The complete searches used different initial trees"
            elif scope != "fixed-partitions":
                raise ValueError("Unknown comparison scope")
            if not pool:
                raise ValueError("Fixed partitions require a nonempty pool")
            for labels, sources in pool.items():
                record = dict(labels=labels, discovered_by=sorted(set(sources)), methods={})
                for method, model in models.items():
                    fit = model.refit(np.array(labels, dtype=int))
                    record["methods"][method] = (describe(model, fit, truth) if np.isfinite(fit.centers).all()
                        else dict(eligible=False, reason=fit.reason))
                result["comparisons"].append(record)
        result["execution"] = {method: model.execution_diagnostics() for method, model in models.items()}
    if source_record() != result["source"]:
        raise RuntimeError("Comparison source changed during execution")
    with (output/"comparison.json").open("x") as stream:
        json.dump(json_value(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def compare_workflow(input_path, output, config, *, truth=None):
    """Freeze marginal Stage 1 first, isolate Stage 2, then compare full workflows."""
    from CliPP2.io.assignment import load_accepted_fit
    from CliPP2.workflow import (run_workflow, classify_groups, select_reference_source,
                                collect_cap_contributors, _stage_two_caps)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_binding = source_record()
    config = replace(config, fit=replace(config.fit, fit_multiplicity="marginal"))
    catalog = load_original_catalog(input_path, max_major_cn=config.fit.max_major_cn,
                                    missing_counts=config.missing_counts)
    first = run_workflow(input_path, output/"marginal", config)
    if first["status"] != "complete":
        raise RuntimeError("Baseline workflow did not complete; no hybrid attempt launched")
    frozen = load_accepted_fit(output/"marginal/stage_one_accepted.json")
    source = select_reference_source(classify_groups(catalog, frozen, config))
    g0 = tuple(m for m, k in zip(frozen.mutation_ids, frozen.labels) if k == source["source_label"])
    remainder = tuple(m for m in frozen.mutation_ids if m not in set(g0))
    bounds, _ = _stage_two_caps(catalog, collect_cap_contributors(catalog, frozen, config, mutation_ids=g0))
    isolated = None
    if remainder:
        isolated = compare_data(catalog.fit_data(remainder, normal_cn_policy=config.normal_cn_policy),
            output/"fixed-stage-one", replace(config.fit, max_clusters=config.stage_two_max_clusters),
            support=bounds.subset(remainder), truth=truth)
    second = run_workflow(input_path, output/"profile", replace(config,
                           fit=replace(config.fit, fit_multiplicity="profile")))
    if second["status"] != "complete":
        raise RuntimeError("Hybrid workflow did not complete; preserve its failure receipt")
    revised = load_accepted_fit(output/"profile/stage_one_accepted.json")
    selected = select_reference_source(classify_groups(catalog, revised, config))
    members = tuple(m for m, k in zip(revised.mutation_ids, revised.labels) if k == selected["source_label"])
    revised_bounds, _ = _stage_two_caps(catalog, collect_cap_contributors(catalog, revised, config, mutation_ids=members))
    native_results = {}
    from CliPP2.kernel.model import FitResult
    from CliPP2.kernel.topology import Membership
    for mode in ("marginal", "profile"):
        native_results[mode] = {}
        for stage in ("stage_one", "stage_two"):
            path = output/mode/(stage+"_accepted.json")
            if not path.exists():
                continue
            accepted = load_accepted_fit(path)
            with RegionalModel(catalog.fit_data(accepted.mutation_ids, normal_cn_policy=config.normal_cn_policy), config.fit.device,
                    cpu_threads=config.fit.cpu_threads, multiplicity_support=accepted.support,
                    fit_multiplicity=mode) as model:
                fit = FitResult(Membership(explicit_labels=accepted.labels), accepted.centers, accepted.weights,
                    accepted.conditional_log_likelihood, accepted.mixture_log_likelihood, accepted.complexity_penalty,
                    accepted.score, True, fitting_log_likelihood=accepted.fitting_log_likelihood,
                    fit_objective_id=accepted.method_identity["fit_objective_id"])
                native_results[mode][stage] = describe(model, fit, truth)
    if source_record() != source_binding:
        raise RuntimeError("Workflow comparison source changed during execution")
    result = dict(scope="workflow", source=source_binding, fixed_stage_one_fingerprint=frozen.state_fingerprint,
        frozen_stage_two_remainder=remainder, frozen_stage_two_support=bounds.subset(remainder).to_dict() if remainder else None,
        isolated_stage_two=isolated,
        source_selection=dict(marginal=first["reference"], profile=second["reference"]),
        cap_changes=int(np.count_nonzero(bounds.upper != revised_bounds.upper)),
        workflows=dict(marginal=first, profile=second), native_results=native_results)
    with (output/"comparison.json").open("x") as stream:
        json.dump(json_value(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("fixed-centers", "fixed-partitions", "search", "workflow"), required=True)
    parser.add_argument("--requests", help="JSON list of partitions or {labels,centers} states in canonical mutation order")
    parser.add_argument("--truth", help="Optional keyed TSV: mutation_id, region_id, truth_cluster, true_ccf, observed")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-clusters", type=int, default=10)
    parser.add_argument("--stage-two-max-clusters", type=int, default=20)
    parser.add_argument("--max-major-cn", type=int)
    parser.add_argument("--missing-counts", choices=("zero_alt", "unobserved"), default="zero_alt")
    args = parser.parse_args()
    config = FitConfig(device=args.device, max_clusters=args.max_clusters, max_major_cn=args.max_major_cn, cpu_threads=1)
    if args.scope == "workflow":
        compare_workflow(args.input, args.output_dir, WorkflowConfig(fit=config,
            stage_two_max_clusters=args.stage_two_max_clusters, missing_counts=args.missing_counts), truth=args.truth)
    else:
        catalog = load_original_catalog(args.input, max_major_cn=config.max_major_cn, missing_counts=args.missing_counts)
        requests = None if args.requests is None else json.loads(Path(args.requests).read_text())
        compare_data(catalog.fit_data(), args.output_dir, config, scope=args.scope,
                     partitions=requests if args.scope == "fixed-partitions" else None,
                     states=requests if args.scope == "fixed-centers" else None, truth=args.truth)
