"""Static release guards for the compact production architecture."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    REPOSITORY_ROOT / "core",
    REPOSITORY_ROOT / "io",
    REPOSITORY_ROOT / "model_selection",
    REPOSITORY_ROOT / "reporting",
    REPOSITORY_ROOT / "runners",
)


def _production_python_files() -> tuple[Path, ...]:
    files = [
        REPOSITORY_ROOT / "__init__.py",
        REPOSITORY_ROOT / "api.py",
        REPOSITORY_ROOT / "cli.py",
        REPOSITORY_ROOT / "config.py",
    ]
    for root in PRODUCTION_ROOTS:
        files.extend(root.rglob("*.py"))
    return tuple(sorted(path for path in files if path.is_file()))


def test_retired_parallel_authorities_stay_deleted() -> None:
    retired = (
        "core/model.py",
        "core/fusion/defaults.py",
        "core/fusion/multiplicity.py",
        "core/fusion/path_summary.py",
        "core/fusion/profiles.py",
        "model_selection/config.py",
        "model_selection/contracts.py",
        "model_selection/partition_initializer.py",
        "runners/cluster_order.py",
        "runners/outputs.py",
        "runners/serialization.py",
        "runners/status_outputs.py",
    )
    assert not [name for name in retired if (REPOSITORY_ROOT / name).exists()]


def test_production_has_one_score_policy_and_no_optional_table_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in _production_python_files()
    )
    forbidden = (
        "fixed_partition_dirichlet_score",
        "classification_code_weight",
        "partition_icl",
        "raw-fusion-only-v0.3",
        "legacy-0.1-selection-compat",
        "_SEARCH_VALUE_TYPES",
        "_SEARCH_TAG_BY_VALUE",
        '"$dataclass"',
        '"$ref"',
        ".legacy_major",
        "controller_snapshot",
        "enable_refinement",
        "evaluate_raw_fusion_candidate",
        "full_certificate_audit_passes",
        "import pandas",
        "next_step",
        "resolve_fit_config_mapping",
        "from pandas",
        "tools.simulation",
    )
    assert not [token for token in forbidden if token in source]


def test_tumor_data_remains_an_input_only_schema() -> None:
    from CliPP2.io.data import TumorData

    assert tuple(item.name for item in fields(TumorData)) == (
        "tumor_id",
        "mutation_ids",
        "region_ids",
        "alt_counts",
        "total_counts",
        "purity",
        "major_cn",
        "minor_cn",
        "normal_cn",
        "tumor_total_cn",
        "count_available",
        "likelihood_supported",
        "policy_included",
        "emission_paths",
        "exclusion_code",
    )

    numerical_source = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            REPOSITORY_ROOT / "core",
            REPOSITORY_ROOT / "model_selection",
            REPOSITORY_ROOT / "reporting",
            REPOSITORY_ROOT / "runners",
        )
        for path in root.rglob("*.py")
    )
    assert "count_observed" not in numerical_source
    assert "PathLikelihoodSpec" not in numerical_source
    assert "path_unsupported_reason" not in numerical_source
    for retired in (
        "fixed_multiplicity",
        "multiplicity_estimation_mask",
        "multiplicity_low",
    ):
        assert not hasattr(TumorData, retired)


def test_production_partition_policy_has_no_fixed_feature_toggles() -> None:
    from CliPP2.config import PartitionCandidateConfig

    assert tuple(item.name for item in fields(PartitionCandidateConfig)) == (
        "k_anchors",
        "max_candidates_per_k",
        "cem_max_iter",
        "generation_refit_max_iter",
        "final_phi_ladder_kmax",
    )


def test_production_modules_do_not_import_private_cross_module_symbols() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            for imported in node.names:
                name = imported.name
                if name.startswith("_") and not (
                    name.startswith("__") and name.endswith("__")
                ):
                    violations.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}:{name}"
                    )
    assert violations == []


def test_external_fit_entry_points_have_compact_boundaries() -> None:
    violations: list[str] = []
    entry_points = (
        REPOSITORY_ROOT / "api.py",
        REPOSITORY_ROOT / "cli.py",
        REPOSITORY_ROOT / "config.py",
        REPOSITORY_ROOT / "runners" / "pipeline.py",
    )
    for path in entry_points:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            parameters = (
                len(node.args.posonlyargs)
                + len(node.args.args)
                + len(node.args.kwonlyargs)
            )
            if parameters > 8:
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}:"
                    f"{node.name}({parameters})"
                )
    assert violations == []
