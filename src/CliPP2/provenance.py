"""External execution links and biological input binding; no source-tree scans."""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path

from ._version import __version__
from .config import PROTOCOL_VERSION
from .io.data import content_hash, json_value


def validate_run_manifest_id(value):
    if value is not None and (not isinstance(value, str) or not value or value != value.strip()
                              or any(char in value for char in "\t\r\n")):
        raise ValueError("run_manifest_id must be an externally supplied nonempty unpadded string or None")
    return value


def resolve_current_run_manifest_id(value=None):
    """Resolve this execution's link without borrowing historical provenance."""
    return validate_run_manifest_id(value if value is not None else os.environ.get("CLIPP2_RUN_MANIFEST_ID"))


def validate_fit_execution_receipt(config_record, execution, *, build=None):
    """Validate original producer structure before any replay backend is opened.

    Optional external build evidence strengthens these checks; its absence never
    permits missing native execution evidence or defaulted saved settings.
    """
    from .config import FitConfig, fitting_identity
    if not isinstance(config_record, Mapping) or set(config_record) != set(FitConfig.__dataclass_fields__):
        raise ValueError("Original fit configuration must explicitly contain every FitConfig field")
    try:
        config = FitConfig(**config_record)
    except (TypeError, ValueError) as error:
        raise ValueError("Original fit configuration is invalid") from error
    if not isinstance(execution, Mapping) or "continuation" not in execution:
        raise ValueError("Original execution must explicitly record its continuation")
    native = execution.get("native")
    if (not isinstance(native, Mapping) or native.get("kind") not in ("cpu", "cuda")
            or any(not isinstance(native.get(key), str) or not native[key].strip()
                   for key in ("build_id", "numerical_policy"))):
        raise ValueError("Original execution lacks native kind, build identity, or numerical policy")
    kind = native["kind"]
    routes = dict(device=kind, backend=f"native_{kind}_likelihood_host_forest",
                  conditional_refit_backend=f"native_{kind}",
                  weight_execution="cuda_resident" if kind == "cuda" else "cpu")
    if config.device != kind or any(execution.get(key) != value for key, value in routes.items()):
        raise ValueError("Original execution routes disagree with its configured device")
    expected = fitting_identity(config.fit_multiplicity, config.selection_criterion, config.max_multiplicity_difference)
    selection = execution.get("selection_native")
    if (execution.get("method_identity") != expected or native.get("aggregation") != config.fit_multiplicity
            or native.get("max_multiplicity_difference", "missing") != config.max_multiplicity_difference
            or not isinstance(selection, Mapping) or selection.get("aggregation") != "marginal"
            or selection.get("max_multiplicity_difference", "missing") != config.max_multiplicity_difference
            or any(selection.get(key) != native[key] for key in ("kind", "build_id", "numerical_policy"))):
        raise ValueError("Original execution fitting/selection identities disagree")
    continuation = execution["continuation"]
    continuation_id = None
    if continuation is not None:
        if not isinstance(continuation, Mapping):
            raise ValueError("Original continuation must be an explicit record or None")
        route = continuation.get("continuation")
        expected_route, identity_key = (("native_cpu", "build_id") if kind == "cpu" else
                                       ("native_owner_cuda_emissions", "native_continuation_build_id"))
        if route != expected_route:
            raise ValueError("Original continuation route disagrees with its configured device")
        continuation_id = continuation.get(identity_key)
        if not isinstance(continuation_id, str) or not continuation_id.strip():
            raise ValueError("Original continuation lacks its native build identity")
    if build is not None:
        _validate_build_record(build)
        artifact = build["artifact"]
        if artifact[kind + "_build_id"] != native["build_id"]:
            raise ValueError("External build record disagrees with the native backend identity")
        if continuation_id is not None and artifact["native_build_id"] != continuation_id:
            raise ValueError("External build record disagrees with the native continuation identity")
    return config


def validate_replay_policy(accepted, requested_fit_config, *, normal_cn_policy):
    """Require the original scientific search; allow current execution controls."""
    from .config import FitConfig
    if type(requested_fit_config) is not FitConfig:
        raise TypeError("Replay requires a validated requested FitConfig")
    producer = accepted.producer_receipt
    if not isinstance(producer, Mapping):
        raise ValueError("Replay requires the original fit producer policy")
    original = validate_fit_execution_receipt(producer.get("fit_config"), producer.get("execution"))
    if ("normal_cn_policy" not in producer or producer["normal_cn_policy"] not in (None, "diploid")
            or normal_cn_policy not in (None, "diploid") or producer["normal_cn_policy"] != normal_cn_policy):
        raise ValueError("Replay normal-CN assumption differs from the original fitting policy")
    if original.max_clusters != requested_fit_config.max_clusters:
        raise ValueError("Replay search capacity differs from the original fitting policy")
    if (original.fit_multiplicity != requested_fit_config.fit_multiplicity or
            original.selection_criterion != requested_fit_config.selection_criterion or
            original.max_multiplicity_difference != requested_fit_config.max_multiplicity_difference):
        raise ValueError("Replay fitting objective, selection criterion or multiplicity-difference policy differs from the original policy")
    # A different declared CN cutoff can leave the actual observations unchanged.
    # AcceptedFit.validate must still verify the exact requested population,
    # numerical inputs, masks and effective support before any reuse is allowed.
    if len(accepted.centers) > original.max_clusters:
        raise ValueError("Accepted occupied count exceeds the original search capacity")
    return original


def _validate_build_record(record):
    if (not isinstance(record, Mapping) or record.get("schema") != "clipp2.run-build.v1"
            or not isinstance(record.get("run_build_id"), str) or not record["run_build_id"]
            or any(not isinstance(record.get(key), Mapping) or not record[key]
                   for key in ("source", "artifact", "environment"))):
        raise ValueError("Execution provenance requires an external clipp2.run-build.v1 record")
    validate_run_manifest_id(record["run_build_id"])
    artifact = record["artifact"]
    if any(not isinstance(artifact.get(key), str) or not artifact[key]
           for key in ("protocol_version", "implementation_version", "native_build_id", "cpu_build_id", "cuda_build_id")):
        raise ValueError("Run/build record lacks original protocol, implementation, or native build identity")


def prepare_run_manifest(catalog, *, supplied=None, run_manifest_id=None):
    """Retain an external record/link and bind this input without gathering code.

    This function writes no files. None records explicitly unbound execution, rather than inventing a
    source/build identity. Launchers may supply CLIPP2_RUN_BUILD_MANIFEST and
    CLIPP2_RUN_MANIFEST_ID once for a cohort. Native library compatibility remains
    the responsibility of the ordinary backend checks.
    """
    if supplied is None:
        supplied = os.environ.get("CLIPP2_RUN_BUILD_MANIFEST")
    if isinstance(supplied, (str, Path)):
        supplied = json.loads(Path(supplied).read_text())
    identifier = resolve_current_run_manifest_id(run_manifest_id)
    if supplied is not None:
        supplied = json_value(supplied)
        _validate_build_record(supplied)
        if identifier is None:
            identifier = supplied["run_build_id"]
        if identifier != supplied["run_build_id"]:
            raise ValueError("run_manifest_id differs from the supplied external build record")
        artifact = supplied["artifact"]
        if artifact["protocol_version"] != PROTOCOL_VERSION or artifact["implementation_version"] != __version__:
            raise ValueError("External build record describes a different executing protocol or implementation")
    payload = dict(run_manifest_id=identifier, build=supplied, input_identity=dict(tumor_id=catalog.tumor_id,
        biological_identity=catalog.biological_identity, original_catalog_identity=content_hash(catalog.to_dict())))
    return dict(schema="clipp2.run-manifest.v2", **payload, manifest_identity=content_hash(payload))


def validate_run_manifest(manifest, *, tumor_id, biological_identity):
    if not isinstance(manifest, Mapping):
        raise ValueError("Missing original run manifest input binding")
    payload = {key: manifest.get(key) for key in ("run_manifest_id", "build", "input_identity")}
    if (manifest.get("schema") != "clipp2.run-manifest.v2" or manifest.get("manifest_identity") != content_hash(payload)
            or manifest.get("input_identity", {}).get("tumor_id") != tumor_id
            or content_hash(manifest.get("input_identity", {}).get("biological_identity")) != content_hash(biological_identity)):
        raise ValueError("Run manifest identity or originating biological identity differs")
    validate_run_manifest_id(manifest.get("run_manifest_id"))
    if manifest.get("build") is not None:
        _validate_build_record(manifest["build"])
        if manifest["run_manifest_id"] != manifest["build"]["run_build_id"]:
            raise ValueError("Original run manifest link differs from its retained build record")
