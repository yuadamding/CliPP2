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


def prepare_run_manifest(catalog, output=None, supplied=None, *, run_manifest_id=None):
    """Retain an external record/link and bind this input without gathering code.

    ``output`` is retained as a compatibility argument; this function writes no
    files. None records explicitly unbound execution, rather than inventing a
    source/build identity. Launchers may supply CLIPP2_RUN_BUILD_MANIFEST and
    CLIPP2_RUN_MANIFEST_ID once for a cohort. Native library compatibility remains
    the responsibility of the ordinary backend checks.
    """
    if supplied is None:
        supplied = os.environ.get("CLIPP2_RUN_BUILD_MANIFEST")
    if isinstance(supplied, (str, Path)):
        supplied = json.loads(Path(supplied).read_text())
    identifier = run_manifest_id if run_manifest_id is not None else os.environ.get("CLIPP2_RUN_MANIFEST_ID")
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
    identifier = validate_run_manifest_id(identifier)
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
