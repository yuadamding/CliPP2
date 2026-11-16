"""Small external execution links and biological input identities."""
import os

from .io.data import content_hash


def validate_run_manifest_id(value):
    if value is not None and (not isinstance(value, str) or not value or value != value.strip()
                              or any(char in value for char in "\t\r\n")):
        raise ValueError("run_manifest_id must be an externally supplied nonempty unpadded string or None")
    return value


def resolve_current_run_manifest_id(value=None):
    return validate_run_manifest_id(value if value is not None else os.environ.get("CLIPP2_RUN_MANIFEST_ID"))


def prepare_run_manifest(catalog, *, run_manifest_id=None):
    payload = dict(run_manifest_id=resolve_current_run_manifest_id(run_manifest_id), build=None,
        input_identity=dict(tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity,
                            original_catalog_identity=content_hash(catalog.to_dict())))
    return dict(schema="clipp2.run-manifest.v2", **payload, manifest_identity=content_hash(payload))
