"""Versioned frozen requests, accepted-fit receipts, and no-clobber outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from ..assignment import (AssignmentBundle, AssignmentResult, POLICY_FIELDS,
                          REGIONAL_BASE_FIELDS, REGIONAL_POLICY_FIELDS, REASSESSMENT_OUTCOMES, EVIDENCE_STATUSES,
                          ASSIGNMENT_POLICIES)
from .data import json_value as _json_value, content_hash
from .reporting import MANIFEST_SCHEMA
from .publication import publish_json
from ..config import PROTOCOL_VERSION
from ..screening import summarize_profile_deviance


def reserve_output(outdir):
    path = Path(outdir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _publish_json(path, value):
    return publish_json(path, json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_assignment_bundle(bundle, path):
    if type(bundle) is not AssignmentBundle:
        raise TypeError("A validated AssignmentBundle is required")
    payload = bundle.validated_payload()
    return _publish_json(path, dict(payload=payload, content_hash=bundle.content_hash))


def load_assignment_bundle(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"payload", "content_hash"} or content_hash(value["payload"]) != value["content_hash"]:
        raise ValueError("Frozen assignment bundle hash mismatch")
    bundle = AssignmentBundle.from_dict(value["payload"])
    if bundle.content_hash != value["content_hash"]:
        raise ValueError("Frozen assignment bundle did not round-trip its content")
    return bundle


def save_original_catalog(catalog, path):
    return _publish_json(path, catalog.to_dict())


def save_accepted_fit(fit, path):
    from ..workflow import AcceptedFit
    if type(fit) is not AcceptedFit:
        raise TypeError("An AcceptedFit is required")
    payload = fit.to_dict()
    return _publish_json(path, dict(payload=payload, content_hash=content_hash(payload)))


def load_accepted_fit(path):
    from ..workflow import AcceptedFit
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"payload", "content_hash"} or content_hash(value["payload"]) != value["content_hash"]:
        raise ValueError("An intact full accepted-fit receipt is required; native TSVs alone are insufficient")
    return AcceptedFit.from_dict(value["payload"])


def write_supporting_table(path, records, *, columns=None):
    path = Path(path)
    records = list(records)
    if not records and columns is None:
        raise ValueError("Empty supporting tables require an explicit column schema")
    frame = pd.DataFrame([{key: json.dumps(_json_value(value), separators=(",", ":"), sort_keys=True)
                           if isinstance(value, (tuple, list, Mapping, np.ndarray)) else value
                           for key, value in row.items()} for row in records], columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clipp2-table-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        frame.to_csv(staged, sep="\t", index=False, float_format="%.17g", na_rep="NA")
        parsed = pd.read_csv(staged, sep="\t", dtype=str, keep_default_na=False)
        if len(parsed) != len(frame) or list(parsed.columns) != list(frame.columns):
            raise ValueError("Supporting TSV did not round-trip its schema/cardinality")
        os.link(staged, path)
    return path


def _matches(actual, expected, *, atol=1e-12):
    if expected is None:
        return actual is None
    if isinstance(expected, Mapping):
        return _json_value(actual) == _json_value(expected)
    if isinstance(expected, (bool, np.bool_)):
        return isinstance(actual, (bool, np.bool_)) and actual == expected
    if isinstance(expected, str):
        return actual == expected
    try:
        left, right = np.asarray(actual, float), np.asarray(expected, float)
        return left.shape == right.shape and np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True)
    except (TypeError, ValueError):
        return actual == expected


def _require_fields(record, expected, context, *, atol=1e-12):
    for name, value in expected.items():
        if name not in record or not _matches(record[name], value, atol=atol):
            raise ValueError(f"{context}: inconsistent {name}")


def _require_integer_sequence(actual, expected, context):
    try:
        values = tuple(actual)
    except TypeError as error:
        raise ValueError(f"{context}: an exact integer sequence is required") from error
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values)
            or values != tuple(expected)):
        raise ValueError(f"{context}: an exact integer sequence is required")


def _independent_deviance(alt, depth, probability):
    """Verify deviance using log1p remainders, independently of screen code.

    For d=p-q, KL=q*(d/q-log1p(d/q)) +
    (1-q)*(-d/(1-q)-log1p(-d/(1-q))). The cancelled linear terms
    never enter the floating calculation. A power series evaluates each small
    remainder; this is distinct from the production symmetric-ratio series.
    """
    q, p = float(alt/depth), float(probability)
    if not np.isfinite(p):
        return np.inf
    if p == q:
        return 0.
    if p <= 0 or p >= 1:
        return np.inf
    if q == 0:
        return float(-2*depth*np.log1p(-p))
    if q == 1:
        return float(-2*depth*np.log(p))
    def remainder(z):
        power, total = z*z, z*z/2
        for exponent in range(3, 101):
            power *= -z
            updated = total + power/exponent
            if updated == total:
                break
            total = updated
        return total
    difference = p-q
    if abs(difference) >= .1*min(q, 1-q):
        return float(2*depth*(q*(np.log(q)-np.log(p))
                             + (1-q)*(np.log1p(-q)-np.log1p(-p))))
    return float(2*depth*(q*remainder(difference/q) + (1-q)*remainder(-difference/(1-q))))


def _conditional_evidence(catalog, source, region, phi, low, high, call, native_call):
    """Independently rebuild the selected pair's binomial terms for publication."""
    if not np.isfinite(call):
        raise ValueError("A defined conditional distribution requires a finite MAP")
    candidates = np.arange(low, high + 1)
    purity = catalog.purity[source, region]
    total_cn = catalog.major_cn[source, region] + catalog.minor_cn[source, region]
    denominator = (1-purity)*catalog.normal_cn[source, region] + purity*total_cn
    p = np.minimum((purity/denominator*candidates)*phi, 1.)
    alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
    logs = xlogy(alt, p) + xlog1py(depth-alt, -p)
    chosen = int(call)-low
    if phi == 0 and chosen != 0:
        raise ValueError("The prior-only exact multiplicity tie must display the smallest candidate")
    maximum = float(np.max(logs))
    if chosen != int(np.argmax(logs)):
        raise ValueError("Published conditional MAP does not maximize the component likelihood")
    others = np.delete(logs, chosen)
    log_probability = float(-np.log1p(np.exp(others-logs[chosen]).sum()))
    probabilities = np.exp(logs-logs[chosen]+log_probability)
    gap = float(maximum-np.partition(logs, -2)[-2]) if len(logs) > 1 else np.nan
    native_lp = (float(logs[int(native_call)-low]-logs[chosen]+log_probability)
                 if np.isfinite(native_call) and low <= native_call <= high else -np.inf)
    deviance = _independent_deviance(alt, depth, p[chosen])
    ties = tuple(int(value) for value in candidates[logs == maximum])
    return candidates, probabilities, log_probability, ties, gap, native_lp, deviance


def _independent_scores(catalog, source, reference, informative, *, assignment_method="marginal"):
    """Verify original-CN marginal scores (with priors), or unnormalized profile scores."""
    result = np.full(len(reference.labels), -np.inf)
    if not catalog.eligible[source].any():
        return result
    result[:] = -np.log(len(reference.labels))
    for region in np.flatnonzero(informative):
        width = int(catalog.major_cn[source, region])
        candidates = np.arange(1, width + 1)
        purity = catalog.purity[source, region]
        total_cn = catalog.major_cn[source, region] + catalog.minor_cn[source, region]
        denominator = ((1-purity)*catalog.normal_cn[source, region]
                       + purity*total_cn)
        probabilities = np.minimum((purity/denominator*candidates)[None, :]
                                   * reference.centers[:, region, None], 1.)
        alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
        logs = (gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)
                + xlogy(alt, probabilities) + xlog1py(depth-alt, -probabilities))
        result += np.max(logs, axis=1) if assignment_method == "profile" else logsumexp(logs, axis=1)-np.log(width)
    return result


def validate_assignment_result(bundle, result):
    """Independently verify method-bound scores, exact maxima and selected multiplicities."""
    from ..assignment import assignment_score_summary, assignment_decision, assignment_outcome, target_reason

    if type(bundle) is not AssignmentBundle or type(result) is not AssignmentResult:
        raise ValueError("Assignment output requires validated request/result records")
    bundle.validated_payload()
    if (result.bundle_hash != bundle.content_hash or result.target_ids != bundle.target_ids
            or result.reference.to_dict() != bundle.reference.to_dict()
            or result.summary.get("bundle_hash") != bundle.content_hash
            or result.assignment_method != bundle.assignment_method):
        raise ValueError("Assignment output must match its live frozen request")
    catalog, labels = bundle.catalog, bundle.reference.labels
    index = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    regional = {(row["mutation_id"], row["region_id"]): row for row in result.multiplicities}
    expected_keys = {(mutation, region) for mutation in bundle.target_ids for region in bundle.reference.region_ids}
    if set(regional) != expected_keys or len(result.multiplicities) != len(expected_keys):
        raise ValueError("Assignment output lost or duplicated target-region keys")
    if len(result.assignments) != len(bundle.target_ids):
        raise ValueError("Assignment output lost target rows")
    obsolete = {"accepted_label", "posterior_pass", "reference_quality_pass", "fit_screen_status",
                "fit_status", "mean_gate_pass", "outlier_gate_pass", "cluster_tie"}
    for i, (mutation, record) in enumerate(zip(bundle.target_ids, result.assignments)):
        if obsolete & set(record):
            raise ValueError("Protocol 2.4 assignment records cannot carry statistical acceptance fields")
        original = index[mutation]
        informative = catalog.eligible[original] & catalog.count_observed[original] & (catalog.total_counts[original] > 0)
        count = int(informative.sum())
        _require_integer_sequence(record.get("reference_labels", ()), labels, "Assignment reference label order")
        scores = np.asarray(record["log_scores"], float)
        if scores.shape != (len(labels),) or np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError("Assignment scores have an invalid shape or value")
        for name in ("best_label", "second_best_label", "map_label", "map_cluster_label", "baseline_label",
                     "baseline_reference_label", "final_assignment_label", "source_label"):
            value = record.get(name)
            if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))):
                raise ValueError("Published assignment labels must be integers or missing")
        _require_fields(record, {"log_scores": _independent_scores(catalog, original, bundle.reference, informative,
            assignment_method=bundle.assignment_method)},
            "Independent assignment emissions", atol=max(1e-12, 32*np.finfo(float).eps*catalog.total_counts[original, informative].sum()))
        expected = assignment_score_summary(scores, labels, bundle.assignment_method, bundle.baseline_labels[i])
        _require_integer_sequence(record.get("map_set", ()), expected["map_set"], "Authoritative exact MAP set")
        _require_integer_sequence(record.get("map_tied_labels", ()), expected["map_tied_labels"], "Exact MAP tie set")
        _require_fields(record, {key: value for key, value in expected.items()
                                if key not in ("best_index", "second_index")}, "Assignment probabilities and exact MAP")
        status = assignment_decision(expected, bool(catalog.eligible[original].any()), count)
        selected = expected["best_index"]
        final = labels[selected] if status == "observed" else None
        _require_fields(record, dict(mutation_id=mutation, assignment_method=bundle.assignment_method, status=status, evidence_status=status,
            informative_regions=count, baseline_label=bundle.baseline_labels[i],
            baseline_reference_label=bundle.baseline_labels[i], final_assignment_label=final,
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            target_reason=target_reason(bundle.source_stages[i]),
            **assignment_outcome(bundle.source_stages[i], bundle.baseline_labels[i], final,
                                 bundle.reference.clonal_label)), "Assignment evidence state")
        residuals, errors = [], []
        selected_joint_score = -np.log(len(labels))
        for j, region in enumerate(bundle.reference.region_ids):
            row = regional[mutation, region]
            low, high = int(bundle.bounds.lower[i, j]), int(bundle.bounds.upper[i, j])
            present = bool(informative[j] and selected is not None)
            phi = float(bundle.reference.centers[selected, j]) if present else np.nan
            error = row.get("diagnostic_error")
            failed = row.get("diagnostic_status") == "failed"
            if failed:
                if not present or not isinstance(error, str) or not error.strip():
                    raise ValueError("Failed diagnostics need informative data and an explicit error")
                errors.append(f"{region}: {error}")
            elif error is not None:
                raise ValueError("Available/missing diagnostics cannot carry an unexplained error")
            pair_defined = present and (bundle.assignment_method == "profile"
                                       or not failed or bool(row["conditional_distribution_defined"]))
            singleton = low == high
            actual = dict(assignment_method=bundle.assignment_method, original_lower=low, original_upper=high,
                original_major_cn=catalog.major_cn[original, j], original_minor_cn=catalog.minor_cn[original, j],
                original_normal_cn=catalog.normal_cn[original, j],
                original_total_cn=catalog.major_cn[original, j] + catalog.minor_cn[original, j],
                alt_count=catalog.alt_counts[original, j], total_depth=catalog.total_counts[original, j],
                ref_count=catalog.total_counts[original, j]-catalog.alt_counts[original, j], purity=catalog.purity[original, j],
                normal_cn_supplied=bool(catalog.normal_cn_supplied[original, j]),
                mapping_genuine=bool(catalog.mapping_genuine[original, j]), segment_id=str(catalog.segment_ids[original, j]),
                count_observed=bool(catalog.count_observed[original, j]), positive_depth=bool(catalog.total_counts[original, j] > 0),
                original_eligible=bool(catalog.eligible[original, j]), excluded=not bool(catalog.eligible[original, j]),
                original_input_excluded=bool(catalog.input_excluded[original, j]),
                informative=bool(informative[j]), native_phi=bundle.native_centers[i, j], native_call=bundle.native_calls[i, j],
                native_lower=bundle.native_lower[i, j], native_upper=bundle.native_upper[i, j],
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
                best_label=record["best_label"], membership_status=status, phi=phi,
                center_source=(bundle.reference.center_sources[selected]
                               if selected is not None else None),
                diagnostic_status="failed" if failed else "available" if present else "not_applicable",
                conditional_distribution_defined=pair_defined, structural_singleton=singleton,
                multiplicity_read_informative=bool(pair_defined and phi > 0))
            _require_fields(row, actual, "Regional evidence")
            for name in ("conditional_map", "conditional_call", "assigned_multiplicity"):
                call = float(row[name])
                if not np.isnan(call) and (not np.isfinite(call) or call != np.rint(call) or not low <= call <= high):
                    raise ValueError("Finite conditional calls must belong to the published original support")
            if pair_defined:
                candidates, conditional, log_probability, ties, gap, native_lp, deviance = _conditional_evidence(
                    catalog, original, j, phi, low, high, row["conditional_map"], bundle.native_calls[i, j])
                if bundle.assignment_method == "profile":
                    purity = catalog.purity[original, j]
                    denominator = ((1-purity)*catalog.normal_cn[original, j]
                        + purity*(catalog.major_cn[original, j]+catalog.minor_cn[original, j]))
                    p = min((purity/denominator*row["conditional_map"])*phi, 1.)
                    a, n = catalog.alt_counts[original, j], catalog.total_counts[original, j]
                    selected_joint_score += gammaln(n+1)-gammaln(a+1)-gammaln(n-a+1)+xlogy(a,p)+xlog1py(n-a,-p)
                _require_integer_sequence(row["multiplicity_candidates"], candidates, "Conditional candidates")
                _require_integer_sequence(row["map_tied_multiplicities"], ties, "Exact multiplicity tie set")
                _require_fields(row, dict(top_two_gap=gap, native_call_log_probability=native_lp), "Independent conditional evidence",
                    atol=max(1e-12, 16*np.finfo(float).eps*catalog.total_counts[original, j]))
                _require_fields(row, dict(profile_deviance=np.nan if failed else deviance), "Independent profile deviance", atol=0.)
                conditional_values = np.asarray(row["conditional_probabilities"], float)
                if (conditional_values.shape != conditional.shape or not np.isfinite(conditional_values).all()
                        or np.any(conditional_values < 0) or not np.isclose(conditional_values.sum(), 1., rtol=0, atol=1e-12)):
                    raise ValueError("Conditional probabilities must be finite, nonnegative, and normalized")
                expected_status = "uninformative_zero_ccf" if phi == 0 else "structural_singleton" if singleton else "conditional"
                _require_fields(row, dict(multiplicity_candidates=candidates, conditional_probabilities=conditional,
                    call_probability=float(np.exp(log_probability)), call_log_probability=log_probability,
                    conditional_call=row["conditional_map"] if phi > 0 else np.nan,
                    assigned_multiplicity=row["conditional_map"] if phi > 0 and final is not None else np.nan,
                    has_second_component=not singleton, multiplicity_tie=len(ties) > 1,
                    map_tied_multiplicities=ties, map_unique=len(ties) == 1,
                    multiplicity_status=expected_status), "Conditional distribution")
            else:
                if len(row["multiplicity_candidates"]) or len(row["conditional_probabilities"]):
                    raise ValueError("Undefined conditional distributions must have no candidate probabilities")
                expected_status = ("excluded" if not catalog.eligible[original, j] else "missing_observation"
                    if not informative[j] else "diagnostic_failed" if failed else "model_incompatible")
                _require_fields(row, dict(conditional_map=np.nan, conditional_call=np.nan, call_probability=np.nan,
                    assigned_multiplicity=np.nan, call_log_probability=np.nan, top_two_gap=np.nan,
                    has_second_component=None, multiplicity_tie=None, map_tied_multiplicities=(), map_unique=None,
                    multiplicity_status=expected_status, native_call_log_probability=np.nan,
                    profile_deviance=np.nan if failed or not present else np.inf), "Undefined multiplicity")
            residuals.append(float(row["profile_deviance"]))
        if bundle.assignment_method == "profile" and selected is not None:
            _require_fields({"winning_score": scores[selected]}, {"winning_score": selected_joint_score},
                "Selected multiplicities must reproduce the winning profile score",
                atol=max(1e-12, 32*np.finfo(float).eps*catalog.total_counts[original, informative].sum()))
        diagnostic_mask = informative & (selected is not None)
        measured = (summarize_profile_deviance(residuals, diagnostic_mask, axis=0) if not errors else
                    dict(mean_profile_deviance=np.nan, maximum_profile_deviance=np.nan))
        _require_fields(record, dict(mean_profile_deviance=measured["mean_profile_deviance"],
            maximum_profile_deviance=measured["maximum_profile_deviance"],
            diagnostic_status="failed" if errors else "available" if diagnostic_mask.any() else "not_applicable",
            diagnostic_error="; ".join(errors) if errors else None), "Diagnostic summary", atol=0.)
    _require_fields(result.summary, dict(schema=result.schema, protocol_version=PROTOCOL_VERSION,
        assignment_method=bundle.assignment_method, tie_policy=ASSIGNMENT_POLICIES[bundle.assignment_method],
        interpretation=("marginal_cluster_MAP" if bundle.assignment_method == "marginal"
            else "profile_joint_cluster_multiplicity_MAP_no_cluster_posterior"),
        target_count=len(bundle.target_ids), unchanged_non_target_count=len(bundle.non_target_ids),
        region_count=len(bundle.reference.region_ids), candidate_count=len(labels), support_policy="original_full_1_to_major_cn",
        bounds_hash=bundle.bounds.content_hash, tumor_id=catalog.tumor_id,
        population_scope="targets_only", comparison_scope="single_candidate" if len(labels) == 1 else "multiple_candidates",
        normal_cn_policy=bundle.normal_cn_policy, run_manifest_id=result.run_manifest_id,
        bundle_run_manifest_id=bundle.run_manifest_id, reference=bundle.reference.to_dict()), "Assignment summary")
    _require_fields(result.summary.get("evaluation_receipt", {}), dict(device=bundle.device,
        cpu_threads=bundle.cpu_threads, assignment_policy=ASSIGNMENT_POLICIES[bundle.assignment_method],
        run_manifest_id=result.run_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id), "Evaluation receipt")
    aggregates = dict(assigned_occupancy={str(label): sum(row["final_assignment_label"] == label for row in result.assignments)
                                         for label in labels},
        status_counts={status: sum(row["status"] == status for row in result.assignments)
                       for status in EVIDENCE_STATUSES},
        reassessment_counts={status: sum(row["reassessment_status"] == status for row in result.assignments)
                             for status in REASSESSMENT_OUTCOMES})
    for name, expected in aggregates.items():
        if dict(result.summary.get(name, {})) != expected:
            raise ValueError("Assignment aggregate counts do not reconcile its derived records")


def _validated_workflow_accounting(bundle, result, accounting):
    """Reconcile the unchanged map with accepted and unresolved target records."""
    from ..assignment import target_reason
    if any(target_reason(stage) == "fixed_reference_recheck" for stage in bundle.source_stages):
        raise ValueError("Two-origin workflow accounting cannot include generic assignment targets")
    if not isinstance(accounting, Mapping) or not isinstance(accounting.get("unchanged_map"), Mapping):
        raise ValueError("Workflow accounting requires an explicit unchanged mutation-to-reference map")
    labels, unchanged = bundle.reference.labels, dict(accounting["unchanged_map"])
    if (accounting.get("reassessment_policy") == "all_combined_fitted_mutations"
            and (unchanged or bundle.non_target_ids)):
        raise ValueError("All-population reassessment cannot retain unchanged non-target members")
    if set(unchanged) != set(bundle.non_target_ids):
        raise ValueError("Workflow unchanged membership differs from the frozen non-target population")
    if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or label not in labels
           for label in unchanged.values()):
        raise ValueError("Unchanged workflow members must have valid integer reference labels")
    fixed = {str(label): sum(value == label for value in unchanged.values()) for label in labels}
    final = {str(label): fixed[str(label)] + sum(row["final_assignment_label"] == label for row in result.assignments)
             for label in labels}
    expected = dict(population_count=len(bundle.target_ids)+len(unchanged), final_occupancy=final,
        unchanged_non_target_occupancy=fixed,
        unresolved_targets=sum(row["final_assignment_label"] is None for row in result.assignments))
    def check_counts(actual, wanted):
        if isinstance(wanted, dict):
            return (isinstance(actual, Mapping) and set(actual) == set(wanted)
                    and all(check_counts(actual[key], value) for key, value in wanted.items()))
        return (not isinstance(actual, (bool, np.bool_)) and isinstance(actual, (int, np.integer))
                and actual >= 0 and actual == wanted)
    for key, wanted in expected.items():
        if not check_counts(accounting.get(key), wanted):
            raise ValueError(f"Workflow accounting does not reconcile {key}")
    if "evidence_status_counts" in accounting:
        counts = {status: sum(row["evidence_status"] == status for row in result.assignments)
                  for status in EVIDENCE_STATUSES}
        if not check_counts(accounting["evidence_status_counts"], counts):
            raise ValueError("Workflow evidence statuses do not reconcile target populations")
    if sum(final.values()) + expected["unresolved_targets"] != expected["population_count"]:
        raise ValueError("Workflow final occupancy and unresolved targets do not cover T/F")
    if "status" in accounting and accounting["status"] != "complete":
        raise ValueError("Only complete reconciled workflow accounting can be published")
    if "historical_caps_recomputed" in accounting and accounting["historical_caps_recomputed"] is not False:
        raise ValueError("Reassessment must preserve historical caps")
    if "outcomes_by_origin" in accounting:
        origins = {origin: {outcome: sum(row["target_reason"] == origin and row["reassessment_status"] == outcome
            for row in result.assignments) for outcome in ("retained", "reassigned", "unresolved")}
            for origin in ("stage1_reference_recheck", "stage2_recheck")}
        if not check_counts(accounting["outcomes_by_origin"], origins):
            raise ValueError("Workflow target-origin outcomes do not reconcile target decisions")
    return _json_value(accounting)


def publish_assignment(bundle, result, outdir, *, workflow_accounting=None):
    validate_assignment_result(bundle, result)
    accounting = (None if workflow_accounting is None else
                  _validated_workflow_accounting(bundle, result, workflow_accounting))
    rows = []
    policy_fields = POLICY_FIELDS
    for i, record in enumerate(result.assignments):
        row = dict(tumor_id=bundle.catalog.tumor_id, mutation_id=bundle.target_ids[i],
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            native_phi=json.dumps(_json_value(bundle.native_centers[i])),
            native_calls=json.dumps(_json_value(bundle.native_calls[i])))
        row.update({key: record[key] for key in policy_fields})
        for j, label in enumerate(bundle.reference.labels):
            row["probability_" + str(label)] = (np.nan if record["probabilities"] is None
                                                else float(record["probabilities"][j]))
            row["log_score_" + str(label)] = float(record["log_scores"][j])
        rows.append(row)
    multiplicities = [dict(tumor_id=bundle.catalog.tumor_id, **row) for row in result.multiplicities]
    target_columns = ["tumor_id", "mutation_id", "source_stage", "source_label", "native_phi", "native_calls", *policy_fields]
    for label in bundle.reference.labels:
        target_columns += ["probability_" + str(label), "log_score_" + str(label)]
    regional_columns = ["tumor_id", *REGIONAL_BASE_FIELDS, *REGIONAL_POLICY_FIELDS]
    output = reserve_output(outdir)
    prefix = bundle.catalog.tumor_id + "_"
    paths = [write_supporting_table(output / (prefix + "mutation_assignments.tsv"), rows, columns=target_columns),
             write_supporting_table(output / (prefix + "assignment_multiplicity.tsv"), multiplicities, columns=regional_columns)]
    if isinstance(bundle.provenance, Mapping) and "assignment_derivation" in bundle.provenance:
        paths.append(save_assignment_bundle(bundle, output / "assignment_bundle.json"))
    summary = dict(result.summary, manifest_schema=MANIFEST_SCHEMA, files=[path.name for path in paths])
    if accounting is not None:
        summary.update(population_scope="workflow_combined", workflow_accounting=accounting)
    summary["source_strata"] = {stage: {
        "targets": sum(s == stage for s in bundle.source_stages),
        **{status: sum(s == stage and row["reassessment_status"] == status
                       for s, row in zip(bundle.source_stages, result.assignments))
           for status in REASSESSMENT_OUTCOMES}}
        for stage in sorted(set(bundle.source_stages))}
    _publish_json(output / (prefix + "assignment_summary.json"), summary)
    return summary


def publish_workflow_receipt(outdir, receipt):
    output = Path(outdir)
    files = [path.relative_to(output).as_posix()
             for path in sorted(output.rglob("*")) if path.is_file() and not path.name.startswith(".")
             and path.name != "workflow_receipt.json"]
    return _publish_json(output / "workflow_receipt.json", dict(receipt, manifest_schema=MANIFEST_SCHEMA, files=files))
