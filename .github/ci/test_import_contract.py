"""Release checks for the lightweight installed-package boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_IMPORTS = (
    "pandas",
    "torch",
    "tools.simulation",
    "CliPP2.core.fusion.solver",
    "CliPP2.model_selection.search",
)


@pytest.mark.parametrize(
    "statement",
    (
        "import CliPP2; assert CliPP2.__version__",
        "from CliPP2 import fit_fixed_objective; assert callable(fit_fixed_objective)",
        "import CliPP2.cli",
    ),
    ids=("package", "api", "cli"),
)
def test_lightweight_imports_do_not_load_numerical_or_optional_stacks(
    statement: str,
) -> None:
    forbidden = repr(FORBIDDEN_IMPORTS)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"{statement}; import sys; "
            f"forbidden={forbidden}; "
            "assert not (set(forbidden) & set(sys.modules)), "
            "set(forbidden) & set(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_source_simulator_remains_available_but_outside_inference_package() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "tools.simulation", "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cli_exposes_only_the_compact_run_surface() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "CliPP2", "fit", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for option in (
        "--input-file",
        "--outdir",
        "--profile",
        "--device",
        "--failure-policy",
        "--checkpoint",
        "--resume",
        "--config",
    ):
        assert option in result.stdout
    for obsolete in (
        "--dtype",
        "--selection-score",
        "--selection-contract",
        "--checkpoint-file",
        "--resume-checkpoint",
        "--max-direct-partition-candidates",
    ):
        assert obsolete not in result.stdout
