"""Truth-blind paired execution for biological plan 1.2 and Protocol 2.3.

Only the observed input and an explicit workflow configuration enter inference.
There is one shared Stage 1, at most two Stage 2 fits, and at most two full-support
assignment passes. Numerical receipts are retained; file checksums are not used.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import json
import os
from time import perf_counter

import numpy as np

from CliPP2._version import __version__
from CliPP2.assignment import AssignmentBundle, assign_mutations
from CliPP2.config import PROTOCOL_VERSION, WorkflowConfig, STAGE_THREE_REFERENCE_POLICY
from CliPP2.io.assignment import (load_accepted_fit, reserve_output, save_accepted_fit,
                                 save_assignment_bundle, save_original_catalog, publish_assignment,
                                 write_supporting_table)
from CliPP2.io.data import json_value, validate_workflow_mapping
from CliPP2.io.publication import publish_json
from CliPP2.io.tumor_txt import audit_input, load_original_catalog
from CliPP2.provenance import prepare_run_manifest, validate_replay_policy
from CliPP2.workflow import (AcceptedFit, CONTRIBUTOR_COLUMNS, GROUP_COLUMNS,
                             SEGMENT_POLICY_COLUMNS, _fit_accepted, _stage_two_caps,
                             _build_stage_three_inputs, _reconcile_final_memberships,
                             classify_groups, collect_cap_contributors, select_reference_source)


def _save_json(path, payload):
    """Publish a new, compact scientific payload without overwriting prior work."""
    return publish_json(path, json.dumps(json_value(payload), separators=(",", ":"), allow_nan=False) + "\n")


def _error(error, stage):
    return dict(status="failed", evidence_status="computation_failed", failed_stage=stage,
                error_type=type(error).__name__, error=str(error))


def _freeze_arm(catalog, first, selected_group, second, groups, g0, config, arm_name, arm_path, *, run_manifest=None):
    """Freeze one clonal reference and Stage 2 alternatives; reassess all members."""
    reference, targets, unchanged, lineage = _build_stage_three_inputs(
        catalog, first, second, selected_group, groups, config)
    g0_set = set(g0)
    centers = reference.centers
    non_targets = tuple(sorted(unchanged))
    sources = {m: (first, i) for i, m in enumerate(first.mutation_ids) if m in g0_set}
    if second is not None:
        sources.update((m, (second, i)) for i, m in enumerate(second.mutation_ids))
    q2, n2 = (0, 0) if second is None else (len(second.centers), len(second.mutation_ids))
    label_map = {(row["source_stage"], row["source_label"]): row["baseline_reference_label"] for row in lineage}
    labels = {m: label_map[fit.source_id, int(fit.labels[i])] for m, (fit, i) in sources.items()}
    if unchanged or set(targets) != set(labels):
        raise AssertionError("Paired Stage 3 requires all fitted mutations and no unchanged subset")
    consolidated_count = 1 + q2 - len(centers)
    native_centers = {m: fit.centers[fit.labels[i]].tolist() for m, (fit, i) in sources.items()}
    selected = [sources[m] for m in targets]
    bundle = AssignmentBundle(catalog, reference, targets, non_targets,
        np.asarray([fit.centers[fit.labels[i]] for fit, i in selected]),
        np.asarray([fit.multiplicities[i] for fit, i in selected]),
        np.asarray([fit.support.lower[i] for fit, i in selected]),
        np.asarray([fit.support.upper[i] for fit, i in selected]),
        tuple(fit.source_id for fit, _ in selected),
        tuple(int(fit.labels[i]) for fit, i in selected), tuple(labels[m] for m in targets),
        device=config.fit.device, cpu_threads=config.fit.cpu_threads,
        normal_cn_policy=config.normal_cn_policy,
        run_manifest_id=None if run_manifest is None else run_manifest["run_manifest_id"],
        provenance=dict(study_plan_version="1.2", protocol_version=PROTOCOL_VERSION, arm=arm_name,
                        reference_policy=STAGE_THREE_REFERENCE_POLICY,
                        native_stage_two_centers_preserved=True))
    positions = {m: i for i, m in enumerate(catalog.mutation_ids)}
    indices = [positions[m] for m in targets]
    eligible = catalog.eligible[indices]
    if (np.any(bundle.bounds.lower[eligible] != 1)
            or not np.array_equal(bundle.bounds.upper[eligible], catalog.major_cn[indices][eligible])):
        raise AssertionError("Paired Stage 3 must restore full original support")
    frozen = dict(combined_labels=labels, native_centers=native_centers,
        reference=reference.to_dict(), targets=targets, non_targets=non_targets,
        source_map=lineage, unchanged_map=unchanged,
        source_members={str(label): [m for m, value in labels.items() if value == label]
                        for label in reference.labels},
        counts=dict(N1=len(first.mutation_ids), N0=len(g0), N2=n2, Q2=q2,
                    clonal_source_consolidated=1, exact_consolidated_centers=consolidated_count,
                    Ncombined=len(labels), combined_mutations=len(labels),
                    K=len(centers), T=len(targets), F=len(non_targets),
                    targets=len(targets), non_targets=len(non_targets)))
    # The baseline and exact scoring request exist on disk before any assignment.
    _save_json(arm_path / "combined_baseline.json", frozen)
    save_assignment_bundle(bundle, arm_path / "assignment_bundle.json")
    write_supporting_table(arm_path / "source_to_reference.tsv", lineage)
    write_supporting_table(arm_path / "assignment_reference_centers.tsv", [dict(
        reference_label=label, prior=1 / len(centers),
        **{"phi_" + region: float(center[j]) for j, region in enumerate(catalog.region_ids)})
        for label, center in enumerate(centers)])
    return bundle, frozen


def run_paired_experiment(input_path, output_dir, config, *, progress_callback=None,
                          accepted_stage_one=None, accepted_uncapped_stage_two=None, run_manifest_id=None):
    """Execute A/B/C/B3/C3 without accessing simulator or evaluator truth.

    Reuse accepts a complete AcceptedFit, its serialized payload, or its published
    receipt path. Every reuse validates biological identity, scientific fit settings,
    actual support, audited evidence and native numerics; it never triggers a fit.
    Failures remain explicit records rather than disappearing from study cells.
    The default WorkflowConfig limits are ten Stage 1 and twenty Stage 2 groups.
    """
    if type(config) is not WorkflowConfig:
        raise TypeError("config must be a WorkflowConfig")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    output = reserve_output(output_dir)
    arm_root = output / "arm_outputs"
    arm_root.mkdir()
    start = perf_counter()
    ledger = dict(fit_count=0, assignment_pass_count=0, reused_fit_count=0, events=[])
    result = dict(schema="clipp2.paired-experiment.v5", study_plan_version="1.2",
        reference_policy=STAGE_THREE_REFERENCE_POLICY,
        protocol_version=PROTOCOL_VERSION, implementation_version=__version__, status="running",
        config=asdict(config), stage_one=None, stage_one_status="not_started", stage_one_groups=[], g0_label=None,
        g0_ids=[], remainder_ids=[], contributors=[], cap_bounds=None, cap_records=[],
        stage_one_candidate_count=0, stage_two_candidate_count=0,
        original_mutation_ids=[], region_ids=[], original_rows=[], arms={}, cost_ledger=ledger,
        inference_truth_access=False, provenance_policy="biological_and_numerical_identity_no_file_checksums")
    for name in ("B", "C"):
        result["arms"][name] = dict(status="not_applicable", stage_two_status="not_applicable",
            assignment_status="not_applicable", stage_two=None, group_records=[],
            combined_labels={}, native_centers={}, reference=None, source_members={}, targets=[],
            non_targets=[], assignment=None, original_rows=[], stage_two_candidate_count=0,
            counts=dict(N1=0, N0=0, N2=0, Q2=0, exact_consolidated_centers=0, Ncombined=0,
                        combined_mutations=0, K=0, T=0, F=0, targets=0, non_targets=0))

    def journal(event):
        with (output / "execution.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(json_value(event), allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def operation(kind, arm, callback):
        event = dict(operation=kind, arm=arm, status="running")
        ledger["events"].append(event)
        event["operation_index"] = len(ledger["events"])
        journal(event)
        counter = {"fit": "fit_count", "assignment": "assignment_pass_count", "reuse": "reused_fit_count"}.get(kind)
        if counter:
            ledger[counter] += 1
        began = perf_counter()
        try:
            value = callback()
            event["status"] = "complete"
            return value
        except Exception as error:
            event.update(_error(error, kind))
            raise
        finally:
            event["elapsed_seconds"] = perf_counter() - began
            journal(event)

    def fit_or_reuse(value, ids, bounds, arm, stage, settings):
        nonlocal first, current
        path = arm_root / arm
        path.mkdir()
        if value is None:
            accepted = operation("fit", arm, lambda: _fit_accepted(catalog, ids, bounds,
                path / "native", stage, settings, progress_callback, normal_cn_policy=config.normal_cn_policy,
                run_manifest=run_manifest, run_manifest_id=resolved_manifest_id))
        else:
            def replay():
                return value.validate(catalog, expected_ids=ids, expected_support=bounds,
                                    device=settings.device, cpu_threads=settings.cpu_threads, run_manifest_id=resolved_manifest_id)
            accepted = operation("reuse", arm, replay)
        if set(accepted.mutation_ids) != set(ids):
            raise AssertionError("Accepted paired population differs from the frozen candidate population")
        if arm == "A":
            first = accepted
            result.update(stage_one=accepted.to_dict(), stage_one_status="accepted")
            for state in result["arms"].values():
                state["counts"]["N1"] = len(accepted.mutation_ids)
        else:
            state = result["arms"][arm]
            state.update(stage_two=accepted.to_dict(), stage_two_status="accepted")
            state["counts"].update(N2=len(accepted.mutation_ids), Q2=len(accepted.centers))
        current = stage + "_accepted_publication"
        save_accepted_fit(accepted, path / "accepted_fit.json")
        return accepted

    catalog, first = None, None
    current = "input"
    try:
        catalog = load_original_catalog(input_path, max_major_cn=config.fit.max_major_cn, missing_counts=config.missing_counts)
        validate_workflow_mapping(catalog, normal_cn_policy=config.normal_cn_policy)
        run_manifest = prepare_run_manifest(catalog, run_manifest_id=run_manifest_id)
        resolved_manifest_id = run_manifest["run_manifest_id"]
        result["run_manifest_id"] = resolved_manifest_id
        result["run_manifest"] = run_manifest
        result["source_cohort_protocol_version"] = dict(catalog.metadata).get("source_cohort_protocol_version")
        _save_json(output / "run_manifest.json", run_manifest)
        save_original_catalog(catalog, output / "original_catalog.json")
        informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
        population = tuple(m for m, keep in zip(catalog.mutation_ids, informative.any(axis=1)) if keep)
        full = catalog.default_bounds()
        result.update(original_mutation_ids=catalog.mutation_ids, region_ids=catalog.region_ids,
                      biological_identity=catalog.biological_identity, stage_one_candidate_count=len(population))
        settings = replace(config.fit, max_clusters=config.stage_two_max_clusters)
        for stage, value, requested in (("stage_one", accepted_stage_one, config.fit),
                                        ("stage_two", accepted_uncapped_stage_two, settings)):
            if value is not None:
                current = stage + "_replay_policy"
                accepted = (value if isinstance(value, AcceptedFit) else AcceptedFit.from_dict(value)
                            if isinstance(value, Mapping) else load_accepted_fit(value))
                if accepted.source_id != stage:
                    raise ValueError("Accepted receipt source stage does not match requested paired replay stage")
                original = validate_replay_policy(accepted, requested, normal_cn_policy=config.normal_cn_policy)
                accepted.validate_input_identity(catalog, original)
                if stage == "stage_one":
                    accepted_stage_one = accepted
                else:
                    accepted_uncapped_stage_two = accepted
        current = "stage_one"
        first = fit_or_reuse(accepted_stage_one, population, full, "A", "stage_one", config.fit)
        current = "stage_one_measurement"
        groups = operation("classification", "A", lambda: classify_groups(catalog, first, config))
        result["stage_one_groups"] = groups
        write_supporting_table(arm_root / "A" / "group_diagnostics.tsv", groups, columns=GROUP_COLUMNS)
        selected = select_reference_source(groups)
        result["reference_source"] = {key: selected[key] for key in (
            "source_label", "native_center", "distance_to_one", "reference_source_role", "selection_rule")}
        g0 = tuple(m for m, label in zip(first.mutation_ids, first.labels) if label == selected["source_label"])
        g0_set = set(g0)
        remainder = tuple(m for m in population if m not in g0_set)
        result.update(g0_label=selected["source_label"], g0_ids=g0, remainder_ids=remainder,
                      stage_two_candidate_count=len(remainder))
        for arm in result["arms"].values():
            arm["counts"]["N0"] = len(g0)
            arm["stage_two_candidate_count"] = len(remainder)
        current = "contributors"
        contributors = operation("contributor_collection", "A", lambda:
            collect_cap_contributors(catalog, first, config, mutation_ids=g0))
        result["contributors"] = contributors
        caps, cap_rows = _stage_two_caps(catalog, contributors)
        result.update(cap_bounds=caps.to_dict(), cap_records=cap_rows)
        write_supporting_table(output / "stage2_cap_contributors.tsv", contributors,
                               columns=CONTRIBUTOR_COLUMNS)
        write_supporting_table(output / "stage2_segment_multiplicity_bounds.tsv", cap_rows,
                               columns=SEGMENT_POLICY_COLUMNS)
        positions = {m: i for i, m in enumerate(catalog.mutation_ids)}
        unsupported = ([] if not remainder else [region for region, keep in zip(catalog.region_ids,
            informative[[positions[m] for m in remainder]].any(axis=0)) if not keep])
        for name, bounds, reuse in (("B", full, accepted_uncapped_stage_two), ("C", caps, None)):
            arm = result["arms"][name]
            current = "stage_two"
            try:
                if unsupported:
                    arm.update(status="blocked", stage_two_status="unsupported_requested_regions",
                               unsupported_regions=unsupported)
                    continue
                if remainder:
                    second = fit_or_reuse(reuse, remainder, bounds.subset(remainder), name, "stage_two", settings)
                    current = "classification"
                    arm["group_records"] = operation("classification", name, lambda: classify_groups(catalog, second, config))
                else:
                    if reuse is not None:
                        raise ValueError("Stage 2 replay is inapplicable to an empty remainder")
                    second = None
                    (arm_root / name).mkdir()
                    arm["stage_two_status"] = "skipped_no_eligible_informative_remainder"
                write_supporting_table(arm_root / name / "group_diagnostics.tsv", arm["group_records"], columns=GROUP_COLUMNS)
                current = "freeze_reference"
                bundle, frozen = _freeze_arm(catalog, first, selected, second, arm["group_records"], g0,
                                             config, name, arm_root / name, run_manifest=run_manifest)
                arm.update(frozen)
                current = "assignment"
                assignment = operation("assignment", name + "3", lambda: assign_mutations(bundle,
                    progress_callback=progress_callback, run_manifest_id=resolved_manifest_id))
                if assignment.target_ids != bundle.target_ids:
                    raise AssertionError("Assignment changed the frozen paired targets")
                current = "reconciliation"
                rows, accounting = _reconcile_final_memberships(catalog, first, second,
                    arm["source_map"], arm["unchanged_map"], assignment, contributors=contributors)
                arm.update(assignment=assignment.to_dict(), assignment_status="scored",
                           original_rows=rows, workflow_accounting=accounting)
                write_supporting_table(arm_root / name / "stage3_vs_two_stage.tsv", rows)
                current = "assignment_publication"
                operation("publication", name + "3", lambda: publish_assignment(bundle, assignment,
                    arm_root / (name + "3"), workflow_accounting=accounting))
                arm.update(assignment_status="complete", status="complete")
                outcomes = {status: sum(row["reassessment_status"] == status for row in assignment.assignments)
                            for status in ("retained", "reassigned", "unresolved")}
                if sum(outcomes.values()) != len(bundle.target_ids):
                    raise AssertionError("Assignment outcomes do not reconcile all paired targets")
                arm["counts"].update(outcomes, unchanged_non_target=len(bundle.non_target_ids))
            except Exception as error:
                arm.update(_error(error, current))
                if current == "assignment":
                    arm["assignment_status"] = "failed"
                elif current == "stage_two" and arm["stage_two_status"] != "accepted":
                    arm["stage_two_status"] = "failed"
        statuses = [arm["status"] for arm in result["arms"].values()]
        result["status"] = ("complete" if all(s == "complete" for s in statuses) else
                            "blocked" if all(s == "blocked" for s in statuses) else "partial_failure")
    except Exception as error:
        stage = current.removesuffix("_replay_policy")
        result.update(_error(error, stage))
        if stage == "input":
            audit = audit_input(input_path, max_major_cn=config.fit.max_major_cn, normal_cn_policy=config.normal_cn_policy)
            result["input_audit"] = audit
            if audit["status"] != "ready":
                result["evidence_status"] = audit["status"]
            _save_json(output / "input_audit.json", audit)
        if stage != current:
            result["failed_phase"] = current
        if stage == "stage_one" and result["stage_one_status"] != "accepted":
            result["stage_one_status"] = "failed"
        for arm in result["arms"].values():
            if arm["status"] == "not_applicable":
                arm.update(status="unavailable_upstream", unavailable_stage=stage)
    if catalog is not None:
        for name, arm in result["arms"].items():
            if not arm["original_rows"]:
                second = AcceptedFit.from_dict(arm["stage_two"]) if arm["stage_two"] else None
                rows, accounting = _reconcile_final_memberships(catalog, first, second,
                    arm.get("source_map", []), arm.get("unchanged_map", {}), None,
                    contributors=result["contributors"])
                arm.update(original_rows=rows, workflow_accounting=accounting)
            arm["derived_phase"] = arm.get("failed_stage", "complete" if arm["status"] == "complete" else arm["status"])
            arm["historical_caps_recomputed"] = False
            path = arm_root / name
            path.mkdir(exist_ok=True)
            if not (path / "stage3_vs_two_stage.tsv").exists():
                write_supporting_table(path / "stage3_vs_two_stage.tsv", arm["original_rows"])
        result["original_rows"] = [dict(mutation_id=m,
            original_input_status=result["arms"]["B"]["original_rows"][i]["original_input_status"])
            for i, m in enumerate(catalog.mutation_ids)]
    ledger["elapsed_seconds"] = perf_counter() - start
    if ledger["fit_count"] > 3 or ledger["assignment_pass_count"] > 2:
        raise AssertionError("Paired experiment exceeded its fit/assignment budget")
    result["manifest_schema"] = "clipp2.file-list.v1"
    result["files"] = sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())
    result["files"].append("paired_result.json")
    result = json_value(result)
    _save_json(output / "paired_result.json", result)
    return result
