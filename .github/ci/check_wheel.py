"""Verify the compact inference-wheel boundary."""

from __future__ import annotations

from pathlib import Path
import sys
from zipfile import ZipFile


EXPECTED_VERSION = "0.4.0"
FORBIDDEN_PARTS = {
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "simulation",
    "tests",
    "tools",
}


def main(directory: str) -> None:
    wheels = tuple(Path(directory).glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel, found {[path.name for path in wheels]}")
    wheel = wheels[0]
    normalized = wheel.name.lower().replace("-", "_")
    if f"_{EXPECTED_VERSION}_" not in normalized:
        raise SystemExit(f"wheel does not identify CliPP2 {EXPECTED_VERSION}: {wheel.name}")
    with ZipFile(wheel) as archive:
        names = tuple(archive.namelist())
    forbidden = sorted(
        name
        for name in names
        if FORBIDDEN_PARTS.intersection(Path(name).parts)
        or name.endswith((".pyc", ".pyo"))
    )
    if forbidden:
        raise SystemExit(f"non-inference files entered the wheel: {forbidden}")
    expected_example = "CliPP2/examples/exampleTumor1.tsv"
    examples = sorted(name for name in names if name.startswith("CliPP2/examples/"))
    if expected_example not in examples:
        raise SystemExit(f"minimal canonical example is missing: {examples}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_wheel.py WHEEL_DIRECTORY")
    main(sys.argv[1])
