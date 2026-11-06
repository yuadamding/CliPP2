"""Atomic publication of already encoded JSON without replacing prior evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def publish_json(path, encoded_json: str) -> Path:
    """Publish one new file; callers own scientific serialization policy."""
    if not isinstance(encoded_json, str):
        raise TypeError("JSON publication requires an encoded string")
    json.loads(encoded_json)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clipp2-json-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_text(encoded_json, encoding="utf-8")
        os.link(staged, path)
    return path
