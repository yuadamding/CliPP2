from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


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
    normalized_help = " ".join(top.stdout.split())
    normalized_fit_help = " ".join(fit.stdout.split())
    assert "Production defaults use CUDA" in normalized_help
    assert "device-only" in fit.stdout
    assert "Dense already materializes" in normalized_fit_help


def test_fit_cli_defaults_to_assignment_aware_selection() -> None:
    from CliPP2.cli import _fit_options_from_args, parse_args

    args = parse_args(["fit", "--input-file", "tumor.tsv"])
    options = _fit_options_from_args(args)

    assert args.selection_score == "partition_icl"
    assert args.lambda_grid_mode == "partition_guided_admm"
    assert args.simulation_root is None
    assert args.outdir == "clipp2_results"
    assert args.bic_df_scale == 1.0
    assert args.bic_cluster_penalty == 0.0
    assert args.materialize_full_dual is False
    assert (options.device, options.dtype) == ("cuda", "float64")
    assert options.inner_backend == "dense"
    assert options.dense_fallback_policy == "device_only"


def test_fit_cli_requires_exactly_one_input_selector() -> None:
    from CliPP2.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["fit"])
    with pytest.raises(SystemExit):
        parse_args(["fit", "--input-file", "tumor.tsv", "--input-dir", "tumors"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--selection-score", "bic"],
        ["--selection-score", "extended_bic"],
        ["--lambda-grid", "0.1,1.0"],
    ],
)
def test_partition_guided_cli_rejects_incompatible_options(
    arguments: list[str],
) -> None:
    from CliPP2.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["fit", "--input-file", "tumor.tsv", *arguments])


def test_auto_device_request_is_passed_to_the_runtime() -> None:
    from CliPP2.cli import _fit_options_from_args, parse_args

    args = parse_args(["fit", "--input-file", "tumor.tsv", "--device", "auto"])

    assert _fit_options_from_args(args).device == "auto"


def test_auto_inner_backend_is_a_dense_compatibility_alias() -> None:
    from CliPP2.cli import _fit_options_from_args, parse_args

    args = parse_args(["fit", "--input-file", "tumor.tsv", "--inner-backend", "auto"])

    assert _fit_options_from_args(args).inner_backend == "dense"


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
