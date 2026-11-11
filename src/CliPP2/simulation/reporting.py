"""Offline evaluation of preserved study evidence, with no inference execution."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from CliPP2._version import __version__
from CliPP2.config import STAGE_THREE_REFERENCE_POLICY
from CliPP2.io.data import OriginalCatalog, content_hash
from CliPP2.io.tumor_txt import load_original_catalog
from CliPP2.provenance import validate_run_manifest


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Preserved evidence path is outside its study")
    return path


def validate_cached_evaluation(record, job, path):
    """Reject decoded but structurally invalid caches with their retained context."""
    from .evaluation import record_evaluation_definition
    prefix = f"Invalid cached evaluation for cell {job['cell_id']} at {path}: "
    try:
        if not isinstance(record, Mapping):
            raise ValueError("expected a JSON object")
        for field in ("cell_id", "history_id", "family"):
            if record.get(field) != job[field]:
                raise ValueError(f"{field} is missing or differs from the frozen cell")
        if record.get("acquisition_id") != job["acquisition"]["id"]:
            raise ValueError("acquisition_id is missing or differs from the frozen cell")
        metrics = record.get("metrics")
        if not isinstance(metrics, list) or any(not isinstance(row, Mapping) for row in metrics):
            raise ValueError("metrics must be a list of metric objects")
        for row in metrics:
            if any(not isinstance(row.get(field), str) or not row[field] for field in ("name", "arm")):
                raise ValueError("each metric requires a name and arm")
            if "value" not in row or not (row["value"] is None or isinstance(row["value"], (int, float))
                    or row["value"] in ("Infinity", "-Infinity")):
                raise ValueError("each metric requires a numeric or missing value")
        record_evaluation_definition(record)
    except (ValueError, TypeError) as error:
        raise ValueError(prefix + str(error)) from error
    return record


def preserved_jobs(root):
    """Read submitted jobs without reopening or regenerating an old cohort."""
    result = {}
    for path in sorted((Path(root) / "job_manifests").glob("batch-*.json")):
        for job in _read(path):
            previous = result.setdefault(job["cell_id"], job)
            if previous != job:
                raise ValueError("Preserved job manifests disagree on a cell's identity")
    return list(result.values())


def roster_identity(record):
    """Keep historical hashes; new rosters omit only the local cohort snapshot path."""
    payload = {key: record.get(key) for key in ("schema", "families", "jobs")}
    policy = record.get("identity_policy")
    if policy is not None:
        if policy != "semantic_jobs_without_cohort_path":
            raise ValueError("Unsupported roster identity policy")
        payload["identity_policy"] = policy
        payload["jobs"] = [{key: value for key, value in job.items() if key != "cohort_settings"}
                           for job in payload["jobs"]]
    return content_hash(payload)


def planned_roster(root, contract, allocation, cells):
    """Use a frozen roster; submitted legacy jobs are only a known subset."""
    path = Path(root) / "main_roster.json"
    if path.exists():
        record = _read(path)
        jobs = record.get("jobs", [])
        if (record.get("schema") != "clipp2.simulation-roster.v1"
                or record.get("roster_id") != roster_identity(record)
                or record.get("families") != contract["families"]
                or len(jobs) != allocation.get("planned_main_datasets")
                or len({job["cell_id"] for job in jobs}) != len(jobs)
                or any(job["phase"] != "main" for job in jobs)):
            raise ValueError("Frozen main roster differs from its identity or allocation")
        lookup = {job["cell_id"]: job for job in jobs}
        for cell in cells:
            if cell["job"]["phase"] == "main" and lookup.get(cell["job"]["cell_id"]) != cell["job"]:
                raise ValueError("Completed main cell differs from the frozen roster")
        return dict(jobs=jobs, complete=True, status="frozen_roster", roster_id=record["roster_id"])
    known = {job["cell_id"]: job for job in preserved_jobs(root) if job["phase"] == "main"}
    for cell in cells:
        job = cell["job"]
        if job["phase"] == "main" and known.setdefault(job["cell_id"], job) != job:
            raise ValueError("Completion differs from retained submitted job")
    return dict(jobs=list(known.values()), complete=False, status="unavailable_complete_frozen_roster", roster_id=None)


class UnsupportedArtifactFormat(ValueError):
    """Present historical evidence requires its producing implementation."""

    def __init__(self, schema, reason, *, revision=None):
        self.schema, self.revision = schema, revision or "the original producing revision"
        super().__init__(f"Unsupported retained artifact {schema!r}: {reason}; use {self.revision} in its pinned environment")


def _configuration_payload(attempt, name, index=1):
    """Read selected configuration evidence without extracting or rewriting it."""
    directory = attempt / f"configuration-{index}"
    payloads = []
    if (directory / name).is_file():
        payloads.append(_read(directory / name))
    archive_path = directory.with_suffix(".zip")
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if member.filename == name]
            if len(members) > 1:
                raise ValueError(f"Duplicate archived {name} evidence")
            if members:
                payloads.append(json.loads(archive.read(members[0])))
    if not payloads:
        raise FileNotFoundError(f"No preserved {name} for the selected attempt")
    if any(content_hash(value) != content_hash(payloads[0]) for value in payloads[1:]):
        raise ValueError(f"Directory and archived {name} evidence disagree")
    return payloads[0]


def _configuration(attempt, index=1):
    return _configuration_payload(attempt, "paired_result.json", index)


def _catalog_payload_hash(payload):
    # Catalog v3/v4 identity retains Unicode; the enclosing manifest uses the
    # separately defined io.data.content_hash encoding on the complete record.
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _catalog_evidence(attempt, paired):
    """Validate the original representation before read-only working conversion."""
    if "run_manifest" not in paired:
        raise FileNotFoundError("Missing original run manifest")
    manifest = paired["run_manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("Original run manifest must be a JSON object")
    if not isinstance(manifest.get("schema"), str):
        raise ValueError("Original run manifest must declare its schema")
    if manifest["schema"] != "clipp2.run-manifest.v2":
        raise UnsupportedArtifactFormat(manifest["schema"], "run manifest format is not supported")
    if "input_identity" not in manifest:
        raise FileNotFoundError("Missing original run manifest input binding")
    if not isinstance(manifest["input_identity"], Mapping):
        raise ValueError("Original run manifest input binding must be a JSON object")
    identity = manifest["input_identity"]
    validate_run_manifest(manifest, tumor_id=identity.get("tumor_id"),
                          biological_identity=identity.get("biological_identity"))
    try:
        published_manifest = _configuration_payload(attempt, "run_manifest.json")
    except FileNotFoundError:
        pass
    else:
        if content_hash(published_manifest) != content_hash(manifest):
            raise ValueError("Paired and separately published original run manifests disagree")
    try:
        retained = _configuration_payload(attempt, "original_catalog.json")
    except FileNotFoundError:
        # Minimal historical evidence is usable only if its complete recorded
        # identity matches a reconstructed representation. Never apply today's
        # missing-read default to an older masked population by assumption.
        config = paired.get("config", {})
        cutoffs = [config.get("fit", {}).get("max_major_cn")]
        if paired.get("schema") not in ("clipp2.paired-experiment.v5", "clipp2.paired-experiment.v6"):
            cutoffs.append(config.get("fit", {}).get("max_major_cn", 10))
        policies = (config["missing_counts"],) if "missing_counts" in config else ("zero_alt", "unobserved")
        retained = None
        for cutoff in dict.fromkeys(cutoffs):
            raw = load_original_catalog(attempt / "input_catalog.tsv", max_major_cn=cutoff, missing_counts="unobserved")
            for policy in policies:
                try:
                    catalog = raw if policy == "unobserved" else load_original_catalog(
                        attempt / "input_catalog.tsv", max_major_cn=cutoff, missing_counts=policy)
                except ValueError:
                    if len(policies) == 1:
                        raise
                    continue
                versions = (6,) if catalog.read_count_imputations else ((6, 5) if cutoff is None else (6, 5, 4))
                for version in versions:
                    candidate = replace(catalog, _schema_version=version).to_dict()
                    if identity.get("original_catalog_identity") == content_hash(candidate):
                        retained = candidate
                        break
                if retained is not None:
                    break
            if retained is not None:
                break
        if retained is None:
            raise FileNotFoundError("Original catalog representation is required to verify the manifest; "
                                    "restore original_catalog.json or use the pinned producing revision") from None
    if not isinstance(retained, Mapping):
        raise TypeError("Preserved original catalog must be a JSON object")
    if identity.get("original_catalog_identity") != content_hash(retained):
        raise ValueError("Original catalog payload differs from its retained manifest identity")
    schema = retained.get("schema")
    if schema not in tuple(f"clipp2.original-catalog.v{v}" for v in (3, 4, 5, 6)):
        raise UnsupportedArtifactFormat(schema, "catalog serialization is not supported")
    payload = {key: value for key, value in retained.items() if key != "content_hash"}
    if retained.get("content_hash") != _catalog_payload_hash(payload):
        raise ValueError("Original catalog payload content hash is invalid")
    working = dict(payload)
    if schema.endswith(".v3"):
        counts = np.asarray(payload["cn_state_count"])
        major, minor = np.asarray(payload["major_cn"]), np.asarray(payload["minor_cn"])
        if (counts.shape != major.shape or counts.dtype.kind not in "iuf"
                or not np.isfinite(counts).all() or np.any(counts < 1) or np.any(counts != np.rint(counts))):
            raise ValueError("Historical catalog CN-state counts are invalid")
        if np.any(counts != 1):
            raise UnsupportedArtifactFormat(schema, "only verified single-state catalogs can be converted", revision="f2d1fa1")
        source_rows = [dict(row) for row in payload["source_rows"]]
        source_keys = [(row["mutation_id"], row["sample_id"]) for row in source_rows]
        if len(source_keys) != len(set(source_keys)):
            raise UnsupportedArtifactFormat(schema, "multiple original CN rows per observation require the historical loader",
                                            revision="f2d1fa1")
        fractions = np.asarray([float(row["cn_state_fraction"]) for row in source_rows])
        if not np.allclose(fractions, 1., atol=1e-8, rtol=0):
            raise ValueError("Historical single-state source fractions are invalid")
        if np.any(fractions != 1.):
            raise UnsupportedArtifactFormat(schema, "non-unit historical source fractions require the historical loader",
                                            revision="f2d1fa1")
        states = np.asarray(payload["cn_states"], dtype=float)
        if (states.shape != (*major.shape, 1, 3)
                or not np.allclose(states[..., 0, 0], 1., atol=1e-8, rtol=0)
                or not np.array_equal(states[..., 0, 1], major)
                or not np.array_equal(states[..., 0, 2], minor)
                or not np.array_equal(np.asarray(payload["mean_total_cn"]), major+minor)):
            raise ValueError("Historical single-state catalog CN definitions disagree with its biological arrays")
        for key in ("cn_state_count", "cn_states", "mean_total_cn"):
            del working[key]
        working["schema"] = "clipp2.original-catalog.v4"
    working["content_hash"] = _catalog_payload_hash(working)
    catalog = OriginalCatalog.from_dict(working)
    parsed = load_original_catalog(attempt / "input_catalog.tsv", max_major_cn=catalog.max_major_cn,
                                   missing_counts=catalog.missing_counts)
    parsed = replace(parsed, _schema_version=catalog._schema_version)
    if content_hash(parsed.to_dict()) != content_hash(catalog.to_dict()):
        raise ValueError("Retained catalog science, keys, masks or provenance differ from the preserved input")
    validate_run_manifest(manifest, tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity)
    compatibility = dict(source_catalog_schema=schema, working_catalog_schema=catalog.to_dict()["schema"],
        source_catalog_identity=identity["original_catalog_identity"],
        working_catalog_identity=content_hash(catalog.to_dict()),
        conversion="verified_single_state_v3_to_v4" if schema.endswith(".v3") else "none",
        original_manifest_preserved=True)
    return catalog, compatibility


def _load_evidence(root, cell):
    from .biology import History
    from .acquisition import validate_history_acquisition_link
    job = cell["job"]
    attempt = _within(root, cell["attempt"])
    if attempt.parent != root.resolve() / "cells" / job["cell_id"]:
        raise ValueError("Selected attempt belongs to a different cell")
    if _read(attempt / "job.json") != job:
        raise ValueError("Attempt job differs from the selected completion")
    directory = _within(root, "histories/" + job["history_id"])
    history_manifest = _read(directory / "history_manifest.json")
    if not isinstance(history_manifest, Mapping):
        raise TypeError("Preserved history manifest must be a JSON object")
    required = ("history_id", "family", "history_seed", "dimensions")
    if any(history_manifest.get(key) != job[key] for key in required):
        raise ValueError("Preserved physical history manifest differs from the selected job")
    if job.get("cohort_content_id") is not None and history_manifest.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved physical history belongs to a different cohort content")
    history_payload = _read(directory / "history.json")
    if isinstance(history_payload, Mapping) and history_payload.get("schema") != "clipp2.simulation.history.v1":
        raise UnsupportedArtifactFormat(history_payload.get("schema"), "physical history format is not supported")
    history = History.from_dict(history_payload)
    if job.get("cohort_content_id") is not None and history.metadata.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved truth belongs to a different cohort content")
    if history.family != job["family"] or history.metadata.get("seed") != job["history_seed"]:
        raise ValueError("Preserved truth does not match the job's family and physical history seed")
    if job.get("cohort_settings") and history.history_id != job.get("source_history_id", job["history_id"]):
        raise ValueError("Preserved authoritative history has a different identity")
    dims = job["dimensions"]
    if (len(history.mutation_ids), len(history.region_ids), len(history.segment_keys)) != (
            dims["mutations"], dims["regions"], dims["segments"]):
        raise ValueError("Preserved history dimensions differ from the job")
    paired = _configuration(attempt)
    if not isinstance(paired, Mapping):
        raise TypeError("Preserved paired inference must be a JSON object")
    if paired.get("schema", "clipp2.paired-experiment.v3") not in ("clipp2.paired-experiment.v3", "clipp2.paired-experiment.v4", "clipp2.paired-experiment.v5", "clipp2.paired-experiment.v6"):
        raise UnsupportedArtifactFormat(paired.get("schema"), "paired inference format is not supported")
    catalog, compatibility = _catalog_evidence(attempt, paired)
    if job.get("cohort_content_id") is not None and dict(catalog.metadata).get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved input catalog belongs to a different cohort content")
    if catalog.tumor_id != job["cell_id"]:
        raise ValueError("Preserved input catalog belongs to another cell")
    acquisition = _read(attempt / "acquisition.json")
    if not isinstance(acquisition, Mapping):
        raise TypeError("Preserved acquisition must be a JSON object")
    if acquisition.get("id", acquisition.get("acquisition_id")) != job["acquisition"]["id"]:
        raise ValueError("Preserved acquisition differs from the selected job")
    if "history_id" in acquisition and acquisition["history_id"] != history.history_id:
        raise ValueError("Preserved acquisition belongs to another physical history")
    if "seed" in acquisition and acquisition["seed"] != job["acquisition_seed"]:
        raise ValueError("Preserved acquisition seed differs from the selected job")
    if job.get("cohort_content_id") is not None and acquisition.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved acquisition belongs to another cohort content")
    for key, attribute in (("alt_counts", "alt_counts"), ("total_counts", "total_counts"),
                           ("count_observed", "count_observed"), ("supplied_major_cn", "major_cn"),
                           ("supplied_minor_cn", "minor_cn"), ("supplied_purity", "purity"),
                           ("supplied_segment_ids", "segment_ids"), ("input_excluded", "input_excluded")):
        if key in acquisition and not np.array_equal(np.asarray(acquisition[key]), getattr(catalog, attribute)):
            raise ValueError("Preserved acquisition and input catalog disagree on " + key)
    if tuple(paired.get("original_mutation_ids", ())) != catalog.mutation_ids or tuple(paired.get("region_ids", ())) != catalog.region_ids:
        raise ValueError("Preserved paired inference has different original axes")
    truth_link = validate_history_acquisition_link(history, acquisition, catalog, history_manifest=history_manifest)
    truth_link["catalog_compatibility"] = compatibility
    return history, catalog, paired, acquisition, truth_link


def unavailable_evaluation(job, status, *, evaluation_config, error=None,
                           execution_state=None, evidence_state="missing_truth_evidence",
                           arm_applicability=None, original_evaluable_count=None,
                           evaluation_schema="clipp2.simulation.evaluation.v6", assignment_method="marginal",
                           reference_policy=STAGE_THREE_REFERENCE_POLICY):
    """Unavailable derived evidence receives no invented scientific decisions."""
    from .evaluation import evaluation_definition, legacy_evaluation_definition, normalize_evaluation_config, metric_truth_independent
    if evaluation_schema not in ("clipp2.simulation.evaluation.v3", "clipp2.simulation.evaluation.v4", "clipp2.simulation.evaluation.v5", "clipp2.simulation.evaluation.v6"):
        raise ValueError("Unavailable evidence requires a supported evaluation interpretation")
    definition = (evaluation_definition(evaluation_config, reference_policy=reference_policy,
                    assignment_method=assignment_method) if evaluation_schema.endswith(".v6")
                  else legacy_evaluation_definition(evaluation_schema, evaluation_config, reference_policy=reference_policy))
    arms = ({row["arm"]: row for row in arm_applicability} if isinstance(arm_applicability, (list, tuple))
            else dict(arm_applicability or {}))
    if original_evaluable_count is not None and (type(original_evaluable_count) is not int or original_evaluable_count < 0):
        raise ValueError("An established original evaluable count must be a nonnegative integer")
    metrics, applicability = [], []
    for arm in ("A", "B", "C", "B3", "C3"):
        state = arms.get(arm, {})
        arm_status = state.get("status", None if execution_state == "execution_failed" else execution_state)
        failed = state.get("execution_failure", None if arm_status is None else arm_status in ("failed", "execution_failed"))
        executed = state.get("execution_available", None if arm_status is None else arm_status != "not_executed")
        scientifically_applicable = state.get("scientifically_applicable")
        available = state.get("available")
        if "available" not in state and (failed or arm_status == "not_executed"):
            available = False
        for name, value in (("execution_available", executed), ("execution_failure", failed),
                            ("analysis_available", available), ("analysis_failure", failed),
                            ("evaluation_available", False), ("scientifically_applicable", scientifically_applicable)):
            metrics.append(dict(name=name, arm=arm, view="all_planned_cells", stratum="all",
                value=None if value is None else float(value), numerator=None if value is None else int(value),
                denominator=1, status="available" if value is not None else "unknown_execution_evidence",
                population_scope="planned_population", truth_required=False))
        claim = dict(name="correct_per_original_evaluable", arm=arm, status="known_no_decisions",
                     value=0., numerator=0, denominator=original_evaluable_count, truth_required=False)
        zero = metric_truth_independent(dict(original_evaluable_count=original_evaluable_count,
            applicability=[dict(state, arm=arm)], execution_state=execution_state), claim)
        if arm != "A":
            for name in ("correct_per_original_evaluable", "binary_correct_per_original_evaluable"):
                metrics.append(dict(name=name, arm=arm, view="", stratum="all", value=0. if zero else None,
                    numerator=0 if zero else None, denominator=original_evaluable_count,
                    status="known_no_decisions" if zero else "unknown_scientific_evidence",
                    population_scope="planned_population", truth_required=not zero))
        applicability.append(dict(arm=arm, available=available, status=arm_status or status, planned=True,
                                  execution_available=executed, execution_failure=failed,
                                  scientifically_applicable=scientifically_applicable,
                                  scientific_rejection=bool(state.get("scientific_rejection")),
                                  upstream_failure=bool(state.get("upstream_failure")),
                                  decision_population=state.get("decision_population")))
    if evaluation_schema.endswith((".v4", ".v5", ".v6")):
        metrics = [row for row in metrics if row["name"] != "correct_per_original_evaluable"]
        for row in metrics:
            if row["name"] == "binary_correct_per_original_evaluable":
                row["name"] = "binary_map_correct_per_original_evaluable"
    return dict(schema=evaluation_schema, evaluation_definition=definition,
        evaluation_config=normalize_evaluation_config(evaluation_config), truth_qualification=evidence_state,
        evaluation_definition_id=content_hash(definition), cell_id=job["cell_id"], history_id=job["history_id"],
        family=job["family"], acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"],
        phase=job["phase"], job=job, status=status, execution_state=execution_state,
        original_evaluable_count=original_evaluable_count, error=error, metrics=metrics, mutations=[],
        calibration=[], cap_audit=[], stage_three_containment=[], information=[],
        applicability=applicability)


def _decision_population(accepted, count, evaluable, evidence, *, unchanged=None, accepted_targets=None):
    """Describe accepted decisions on the endpoint population, never only targets."""
    if count is None or (evaluable is None and accepted):
        return None
    use = set(accepted) if evaluable is None else set(accepted) & evaluable
    result = dict(scope="whole_original_evaluable", coverage="complete", evaluable_count=count,
                  accepted_count=len(use), evidence=evidence)
    if unchanged is not None:
        result["unchanged_count"] = len(set(unchanged) if evaluable is None else set(unchanged) & evaluable)
    if accepted_targets is not None:
        result["accepted_target_count"] = len(set(accepted_targets) if evaluable is None else set(accepted_targets) & evaluable)
    return result


def _retained_memberships(arm, original_ids):
    """Recover complete frozen membership axes and any returned target decisions."""
    combined = arm.get("combined_labels")
    if not isinstance(combined, Mapping):
        return None, None, None
    if (not set(combined) <= original_ids or
            any(type(label) is not int or label < 0 for label in combined.values())):
        raise ValueError("Retained combined memberships differ from the original input")
    non_targets, targets = arm.get("non_targets"), arm.get("targets")
    if not isinstance(non_targets, (list, tuple)) or not isinstance(targets, (list, tuple)):
        return combined, None, None
    fixed, selected = set(non_targets), set(targets)
    if (len(fixed) != len(non_targets) or len(selected) != len(targets) or fixed & selected
            or fixed | selected != set(combined)):
        return combined, None, None
    unchanged = arm.get("unchanged_map")
    if unchanged is not None and (not isinstance(unchanged, Mapping)
            or unchanged != {mutation: combined[mutation] for mutation in fixed}):
        return combined, None, None
    assignment = arm.get("assignment")
    if assignment is None:
        no_targets_returned = arm.get("assignment_status") == "failed" or arm.get("failed_stage") == "assignment"
        return combined, fixed, set() if no_targets_returned else None
    if not isinstance(assignment, Mapping) or not isinstance(assignment.get("assignments"), (list, tuple)):
        return combined, fixed, None
    rows = assignment["assignments"]
    label_key = ("final_assignment_label" if assignment.get("protocol_version") in ("2.3", "2.4")
                 or assignment.get("schema") in ("clipp2.assignment.result.v5", "clipp2.assignment.result.v6") else "accepted_label")
    if (any(not isinstance(row, Mapping) or "mutation_id" not in row or label_key not in row for row in rows)
            or len(rows) != len(selected) or {row["mutation_id"] for row in rows} != selected
            or any(row[label_key] is not None and (type(row[label_key]) is not int
                or row[label_key] < 0) for row in rows)):
        return combined, fixed, None
    return combined, fixed, {row["mutation_id"] for row in rows if row[label_key] is not None}


def placeholder_for_cell(root, cell, evaluation_config, *, evidence_state="missing_truth_evidence", error=None,
                         evaluation_schema=None, reference_policy=None):
    """Recover scoped decision evidence without inventing missing correctness."""
    from .evaluation import _operational_states, HISTORICAL_V5_REFERENCE_POLICY
    root, job = Path(root), cell["job"]
    if reference_policy is None:
        reference_policy = cell.get("reference_policy", HISTORICAL_V5_REFERENCE_POLICY)
    count, evaluable, region_count = cell.get("original_evaluable_count"), None, None
    arms, predecision_sources, execution = {}, {}, {"arms": {}}
    if cell.get("status") == "not_executed":
        execution["status"] = "not_executed"
    # An assignment exception only establishes missing target output. Unchanged
    # memberships may already exist; their whole-population contribution survives.
    for failure in cell.get("execution_failures", ()):
        arm, stage = failure.get("arm"), failure.get("failed_stage")
        if arm is None and (stage == "stage_one" or
                stage == "worker" and cell.get("execution_phase") == "before_inference"):
            affected, source = ("A", "B", "C", "B3", "C3"), "failure_before_stage_one"
            execution.update(status="failed", failed_stage=stage)
        elif arm in ("B", "C") and stage == "stage_two":
            affected, source = (arm, arm+"3"), "failure_before_combined_memberships"
        elif arm in ("B", "C") and stage in ("assignment", "reconciliation", "assignment_publication"):
            affected, source = (arm+"3",), None
        else:
            affected, source = (), None
        if arm in ("B", "C"):
            execution["arms"][arm] = dict(status="failed", failed_stage=stage)
        for name in affected:
            if source:
                predecision_sources[name] = source
            arms[name] = dict(status="failed", available=False if source or stage == "assignment" else None,
                decision_population=_decision_population(set(), count, None, source) if source else None)
    for name, state in _operational_states(execution, arms).items():
        arms[name] = dict(arms.get(name, {}), **state)
    try:
        attempt = _within(root, cell["attempt"])
        if _read(attempt / "job.json") != job:
            raise ValueError("Selected attempt job differs")
        try:
            paired = _configuration(attempt)
        except FileNotFoundError:
            paired = None
            # An early failure may precede paired publication. Retained input and
            # acquisition can still establish its denominator, but not memberships.
            catalog = load_original_catalog(attempt / "input_catalog.tsv", missing_counts="unobserved")
        else:
            if not isinstance(paired, Mapping):
                raise TypeError("Preserved paired inference must be a JSON object")
            catalog, _ = _catalog_evidence(attempt, paired)
        if catalog.tumor_id != job["cell_id"]:
            raise ValueError("Original population belongs to another cell")
        region_count = len(catalog.region_ids)
        acquisition = _read(attempt / "acquisition.json")
        if not isinstance(acquisition, Mapping):
            raise TypeError("Preserved acquisition must be a JSON object")
        known = np.asarray(acquisition.get("biological_truth_available"), bool)
        if known.shape == catalog.alt_counts.shape:
            eligible = known.all(axis=1) & (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)).any(axis=1)
            evaluable = {mutation for mutation, keep in zip(catalog.mutation_ids, eligible) if keep}
            count = len(evaluable)
        for name, source in predecision_sources.items():
            arms[name]["decision_population"] = _decision_population(set(), count, evaluable, source)
        if paired is None:
            raise FileNotFoundError("No preserved paired result for membership recovery")
        root_failed = paired.get("status") == "failed" and paired.get("stage_one") is None
        rejection = paired.get("status") in ("no_eligible_clonal_reference", "blocked")
        retained = {"A": dict(available=bool(paired.get("stage_one")) if "stage_one" in paired else None)}
        original_ids = set(catalog.mutation_ids)
        if not isinstance(paired.get("arms", {}), Mapping):
            raise TypeError("Preserved paired arms must be a JSON object")
        for name in ("B", "C"):
            arm = paired.get("arms", {}).get(name, {})
            if not isinstance(arm, Mapping):
                continue
            state = "failed" if root_failed else arm.get("status", paired.get("status"))
            rejected = rejection or state in ("blocked", "no_eligible_clonal_reference")
            if arm.get("combined_labels") or arm.get("assignment"):
                # Contradictory retained decisions override an earlier failure-only claim.
                for label in (name, name+"3"):
                    if label in arms:
                        arms[label]["decision_population"] = None
            combined, fixed, accepted_targets = _retained_memberships(arm, original_ids)
            combined_known = combined is not None
            before_combined = root_failed or rejected or arm.get("failed_stage") == "stage_two"
            combined_complete = combined_known and (bool(combined) or before_combined)
            combined_population = (_decision_population(set(combined), count, evaluable, "retained_combined_memberships")
                                   if combined_complete else None)
            if not combined_known and before_combined:
                combined_population = _decision_population(set(), count, evaluable,
                    "failure_before_stage_one" if root_failed else "scientific_rejection_before_combined_memberships"
                    if rejected else "failure_before_combined_memberships")
            reference = arm.get("reference")
            retained[name] = dict(available=True if combined and reference else False if combined_known and not combined else None,
                                 reference_available=bool(reference) if "reference" in arm else None)
            arms[name] = dict(decision_population=combined_population)
            stage_three_population = None
            if fixed is not None and accepted_targets is not None:
                stage_three_population = _decision_population(fixed | accepted_targets, count, evaluable,
                    "retained_stage_three_memberships", unchanged=fixed, accepted_targets=accepted_targets)
            elif before_combined and (not combined_known or not combined):
                stage_three_population = combined_population
            returned = arm.get("assignment") is not None
            assignment_available = (True if returned and fixed is not None and accepted_targets is not None
                                    else None if returned or "assignment" not in arm else False)
            retained[name+"3"] = dict(available=assignment_available)
            arms[name+"3"] = dict(decision_population=stage_three_population)
        for name, state in _operational_states(paired, retained).items():
            arms[name] = dict(arms.get(name, {}), **state)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        pass  # Missing or inconsistent evidence cannot prove an empty population.
    value = unavailable_evaluation(job, "unavailable_derived_evidence", evaluation_config=evaluation_config,
        reference_policy=reference_policy,
        assignment_method=cell.get("assignment_method", "marginal"),
        execution_state=cell.get("status"), evidence_state=evidence_state, arm_applicability=arms,
        original_evaluable_count=count, error=error, evaluation_schema=evaluation_schema or cell.get("evaluation_schema") or
        ("clipp2.simulation.evaluation.v6" if cell.get("protocol_version") == "2.4" else "clipp2.simulation.evaluation.v4" if cell.get("protocol_version") == "2.3" else "clipp2.simulation.evaluation.v3"))
    if region_count is not None:
        value["region_count"] = region_count
    return value


def reevaluate_cell(root, cell, evaluation_config, *, evaluation_schema=None, reference_policy=None):
    """Derive a new record solely from the selected attempt's saved evidence."""
    from .evaluation import evaluate
    root = Path(root).resolve()
    job = cell["job"]
    truth_link = None
    try:
        history, catalog, paired, acquisition, truth_link = _load_evidence(root, cell)
        qualification = truth_link["status"]
        if qualification in ("conflicting_truth", "missing_truth_evidence"):
            value = placeholder_for_cell(root, cell, evaluation_config, evidence_state=qualification,
                error=dict(type="TruthEvidenceError", message=truth_link["reason"]), evaluation_schema=evaluation_schema,
                reference_policy=reference_policy)
        else:
            value = evaluate(history, catalog, paired, acquisition=acquisition, evaluation_config=evaluation_config,
                             truth_qualification=qualification)
            value.update(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
                acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"], phase=job["phase"], job=job,
                status="derived_from_preserved_evidence")
    except (OSError, ValueError, KeyError, TypeError, AssertionError, zipfile.BadZipFile) as error:
        qualification = ("unsupported_artifact_format" if isinstance(error, UnsupportedArtifactFormat) else
                         "conflicting_truth" if isinstance(error, (ValueError, AssertionError)) else "missing_truth_evidence")
        value = placeholder_for_cell(root, cell, evaluation_config, evidence_state=qualification,
                                     error=dict(type=type(error).__name__, message=str(error)), evaluation_schema=evaluation_schema,
                                     reference_policy=reference_policy)
        if isinstance(error, UnsupportedArtifactFormat):
            value["artifact_compatibility"] = dict(status=qualification, source_schema=error.schema,
                                                  required_pinned_revision=error.revision)
    value["truth_qualification"] = qualification
    if truth_link is not None:
        value["truth_link"] = truth_link
        value["catalog_compatibility"] = truth_link["catalog_compatibility"]
    value["derivation"] = dict(implementation_version=__version__, source_study=str(root),
        selected_attempt=cell.get("attempt"), source_implementation_version=cell.get("implementation_version"),
        source_protocol_version=cell.get("protocol_version"), inference_decisions_preserved=True,
        fit_count=0, assignment_pass_count=0, threshold_selection_count=0)
    if truth_link is not None:
        value["derivation"].update({key: truth_link["catalog_compatibility"][key]
            for key in ("source_catalog_schema", "working_catalog_schema")})
    return value
