from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_is_default_fit_only() -> None:
    top = subprocess.run(
        [sys.executable, "-m", "CliPP2.cli", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    fit = subprocess.run(
        [sys.executable, "-m", "CliPP2.cli", "fit", "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert top.returncode == 0
    assert fit.returncode == 0
    assert "{fit}" in top.stdout
    assert "--lambda-grid-mode" in fit.stdout
    assert "partition_guided_admm" in fit.stdout
    assert "--settings-profile" not in fit.stdout
    assert "--selection-score" in fit.stdout
    assert "partition_icl" in fit.stdout
    assert "extended_bic" in fit.stdout
    assert "--benchmark-simulation" not in fit.stdout
    assert "--simulation-root" in fit.stdout
    assert "fixed_grid" not in fit.stdout
    assert "dense_no_zero" not in fit.stdout


def test_fit_cli_defaults_to_assignment_aware_selection() -> None:
    from CliPP2.cli import build_parser

    args = build_parser().parse_args(["fit"])

    assert args.selection_score == "partition_icl"
    assert args.lambda_grid_mode == "partition_guided_admm"
    assert args.simulation_root is None
    assert args.bic_df_scale == 1.0
    assert args.bic_cluster_penalty == 0.0
    assert args.materialize_full_dual is False


def test_top_level_package_surface_is_compact() -> None:
    import CliPP2
    import CliPP2.runners as runners

    assert set(CliPP2.__all__) == {
        "FitResult",
        "PairwiseFusionGraph",
        "Problem",
        "SolverOptions",
        "TumorData",
        "__version__",
        "fit",
        "load_tumor_tsv",
    }
    for name in [
        "FitOptions",
        "process_one_file",
        "run_directory",
        "run_simulation_grid",
        "run_simulation_benchmark",
        "run_massive_multiregion_benchmark",
        "select_model",
        "evaluate_fit_against_simulation",
        "generate_and_convert_simulation",
        "plot_benchmark_outcomes",
    ]:
        assert name not in CliPP2.__all__
        assert not hasattr(CliPP2, name)
    assert runners.__all__ == []


def test_installed_console_scripts_are_compact() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    scripts = pyproject["project"]["scripts"]

    assert set(scripts) == {"clipp2"}
    assert scripts["clipp2"] == "CliPP2.cli:main"
