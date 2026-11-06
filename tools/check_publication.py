"""Deterministic checks of atomic, non-overwriting JSON publication."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from CliPP2.io.publication import publish_json


def run_checks(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "result.json"
    values = ('{"writer":1}\n', '{"writer":2}\n')

    def publish(value):
        try:
            publish_json(path, value)
        except FileExistsError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(publish, values))
    assert sum(outcomes) == 1
    assert path.read_text() == values[outcomes.index(True)]
    retained = path.read_bytes()
    assert not publish('{"replacement":true}')
    assert path.read_bytes() == retained
    invalid = root / "invalid" / "result.json"
    try:
        publish_json(invalid, '{"truncated":')
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Malformed JSON was published")
    assert not invalid.exists()
    assert not list(root.glob(".clipp2-json-*"))
    return dict(status="passed", concurrent_writer_exclusion=True,
                original_bytes_retained=True, invalid_json_unpublished=True,
                temporary_files_removed=True)
