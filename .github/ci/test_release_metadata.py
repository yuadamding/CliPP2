"""Release metadata guards for the v0.4 compact product surface."""

from __future__ import annotations

from pathlib import Path

from CliPP2 import __version__
from CliPP2.model_selection.checkpoint import _inference_source_files


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_release_version_is_0_4_0() -> None:
    assert __version__ == "0.4.0"


def test_inference_distribution_excludes_optional_products() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    core_dependencies = pyproject.split("[project]", 1)[1].split(
        "[project.optional-dependencies]", 1
    )[0]
    package_manifest = pyproject.split("[tool.setuptools]", 1)[1].split(
        "[tool.setuptools.dynamic]", 1
    )[0]
    assert '"CliPP2.reporting"' in package_manifest
    assert '"CliPP2.tools' not in package_manifest
    assert '"CliPP2.simulation' not in package_manifest
    assert "pandas" not in core_dependencies.lower()


def test_packaged_example_is_a_small_schema_smoke_case() -> None:
    example = REPOSITORY_ROOT / "examples" / "exampleTumor1.tsv"
    assert example.stat().st_size < 8 * 1024
    assert len(example.read_text(encoding="utf-8").splitlines()) <= 20


def test_checkpoint_source_identity_excludes_generated_and_optional_trees() -> None:
    relative = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _inference_source_files(REPOSITORY_ROOT)
    }
    assert "model_selection/checkpoint.py" in relative
    assert not [
        name
        for name in relative
        if name.startswith(("build/", ".github/", "tools/", "simulation/"))
    ]
