"""Versioned frozen requests, accepted-fit receipts, and no-clobber outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from .._source import file_sha256
from ..assignment import AssignmentBundle, AssignmentResult, _json_value, content_hash


def reserve_output(outdir):
    path = Path(outdir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _publish_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clipp2-json-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        # Parse before publication, then link without overwriting any previous attempt.
        json.loads(staged.read_text(encoding="utf-8"))
        os.link(staged, path)
    return path


def save_assignment_bundle(bundle, path):
    if type(bundle) is not AssignmentBundle:
        raise TypeError("A validated AssignmentBundle is required")
    return _publish_json(path, dict(payload=bundle.to_dict(), content_hash=bundle.content_hash))


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
        columns = (["mutation_id", "region_id", "segment_id", "usable", "reason"]
                   if "contributors" in path.name else ["source_stage", "source_label", "classification"])
    frame = pd.DataFrame([{key: json.dumps(_json_value(value), separators=(",", ":"))
                           if isinstance(value, (tuple, list, dict)) else value
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


def publish_assignment(bundle, result, outdir):
    if type(bundle) is not AssignmentBundle or type(result) is not AssignmentResult or result.bundle_hash != bundle.content_hash:
        raise ValueError("Assignment output must match its frozen request")
    if (result.target_ids != bundle.target_ids or result.reference.to_dict() != bundle.reference.to_dict()
            or result.summary.get("bundle_hash") != bundle.content_hash):
        raise ValueError("Assignment result populations/reference differ from the frozen request")
    output = reserve_output(outdir)
    rows = []
    for i, mutation in enumerate(bundle.target_ids):
        row = dict(tumor_id=bundle.catalog.tumor_id, mutation_id=mutation,
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            native_phi=json.dumps(_json_value(bundle.native_centers[i])),
            native_calls=json.dumps(_json_value(bundle.native_calls[i])))
        for name, records in (("restricted", result.restricted), ("original", result.original)):
            record = records[i]
            row.update({name + "_" + key: record[key] for key in ("status", "best_label", "accepted_label",
                       "posterior", "log_score_margin", "mean_profile_deviance", "informative_regions")})
            for j, label in enumerate(bundle.reference.labels):
                row[name + "_probability_" + str(label)] = float(record["probabilities"][j])
                row[name + "_log_score_" + str(label)] = float(record["log_scores"][j])
        row.update({key: value for key, value in result.comparison[i].items() if key != "mutation_id"})
        rows.append(row)
    multiplicities = [dict(tumor_id=bundle.catalog.tumor_id, **row) for row in result.multiplicities]
    keys = {(row["mutation_id"], row["region_id"]) for row in multiplicities}
    expected = {(mutation, region) for mutation in bundle.target_ids for region in bundle.reference.region_ids}
    if len(rows) != len(bundle.target_ids) or keys != expected or len(multiplicities) != len(expected):
        raise ValueError("Assignment output lost or duplicated original target keys")
    prefix = bundle.catalog.tumor_id + "_"
    paths = [write_supporting_table(output / (prefix + "mutation_assignments.tsv"), rows),
             write_supporting_table(output / (prefix + "assignment_multiplicity.tsv"), multiplicities)]
    summary = dict(result.summary)
    summary["files"] = {path.name: file_sha256(path) for path in paths}
    summary["source_strata"] = {stage: {
        "targets": sum(s == stage for s in bundle.source_stages),
        "support_sensitive_decisions": sum(s == stage and bool(row["support_sensitive_decision"])
                                           for s, row in zip(bundle.source_stages, result.comparison)),
        "support_stable_assignments": sum(s == stage and row["support_stable_label"] is not None
                                         for s, row in zip(bundle.source_stages, result.comparison))}
        for stage in sorted(set(bundle.source_stages))}
    _publish_json(output / (prefix + "assignment_summary.json"), summary)
    return summary


def publish_workflow_receipt(outdir, receipt):
    output = Path(outdir)
    files = {path.relative_to(output).as_posix(): file_sha256(path)
             for path in sorted(output.rglob("*")) if path.is_file() and not path.name.startswith(".")
             and path.name != "workflow_receipt.json"}
    return _publish_json(output / "workflow_receipt.json", dict(receipt, files=files))
