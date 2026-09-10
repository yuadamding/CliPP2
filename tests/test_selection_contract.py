"""One production policy with the original hybrid and score constants."""

from dataclasses import FrozenInstanceError, replace

import pytest

from CliPP2.cli import _fit_config_from_args, parse_args
from CliPP2.config import (
    DIRICHLET_ALPHA,
    DIRICHLET_CODE_WEIGHT,
    FINAL_PHI_LADDER_KMAX,
    FINAL_PHI_PARENT_COUNT,
    PARTITION_K_ANCHORS,
    PARTITION_CEM_MAX_ITER,
    PARTITION_GENERATION_REFIT_MAX_ITER,
    PARTITION_MAX_CANDIDATES_PER_K,
    PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT,
    resolve_fit_config,
)
from CliPP2.core.fusion import partition_starts
from test_curvature_preparation import _context
from test_integer_likelihood import integer_data


@pytest.mark.parametrize("profile,expected", [
    ("strict", ("float64", "interval_certified", 17, 0, 12, 12, 8, 30, 5e-5, 4, 1e-4, 1e-7, 128, 512, 2)),
    ("balanced", ("float32", "grid_local", 64, 3, 8, 2, 6, 25, 8e-4, 1, 2e-4, 1e-5, 64, 128, 1)),
    ("fast", ("float32", "grid_local", 32, 1, 6, 0, 4, 16, 1e-3, 0, 1e-3, 1e-4, 32, 64, 0)),
])
def test_profiles_preserve_reference_numerical_settings(profile, expected):
    config = resolve_fit_config(computation_profile=profile)
    refit, solver, search = config.selection.refit, config.solver, config.selection.lambda_search
    observed = (
        config.runtime.dtype, refit.mode, refit.grid_points, refit.local_steps,
        search.exploration_budget, search.refinement_budget,
        solver.outer_max_iter, solver.inner_max_iter, solver.tolerance,
        search.solver_retry_limit, config.selection.partition_tolerance,
        refit.tolerance, refit.max_iter,
        solver.certificate.max_iter, solver.certificate.refinement_rounds,
    )
    assert observed == expected
    assert config.selection.contract_id == "hybrid-ward-cem-v1"
    assert config.selection.score == "fixed_partition_dirichlet_score"
    assert config.selection.graph_pilot_source == "zero_penalty_pilot"
    assert config.selection.dirichlet_alpha == DIRICHLET_ALPHA == 1.0
    assert config.selection.dirichlet_code_weight == DIRICHLET_CODE_WEIGHT == 0.7


@pytest.mark.parametrize("keyword,value", [
    ("selection_contract", "raw-fusion-only-v0.3"),
    ("selection_contract", "legacy-0.1-selection-compat"),
    ("selection_contract", "hybrid-ward-cem-v1"),
    ("selection_score", "fixed_partition_bic"),
    ("selection_score", "fixed_partition_dirichlet_score"),
])
def test_removed_programmatic_selection_switches_fail_explicitly(keyword, value):
    with pytest.raises(TypeError, match=keyword):
        resolve_fit_config(**{keyword: value})


@pytest.mark.parametrize("option", [
    ["--selection-contract", "raw-fusion-only-v0.3"],
    ["--selection-contract", "legacy-0.1-selection-compat"],
    ["--selection-score", "fixed-partition-bic"],
])
def test_removed_cli_selection_switches_are_not_silent_noops(option):
    with pytest.raises(SystemExit) as failure:
        parse_args(["fit", "--input-file", "unused.tsv", *option])
    assert failure.value.code == 2


def test_fit_cli_has_fixed_production_policy_and_no_simulator_command():
    args = parse_args(["fit", "--input-file", "unused.tsv", "--device", "cpu"])
    config = _fit_config_from_args(args)
    assert config.selection.score == "fixed_partition_dirichlet_score"
    assert config.runtime.device == "cpu"
    with pytest.raises(SystemExit):
        parse_args(["simulate"])
    with pytest.raises(TypeError):
        replace(config.selection, score="fixed_partition_bic")
    with pytest.raises((FrozenInstanceError, TypeError)):
        config.selection.score = "fixed_partition_bic"


def test_hybrid_initializer_keeps_ward_cem_and_reference_settings(monkeypatch):
    observed = {}

    def labels(*args, **kwargs):
        observed["k_grid"] = kwargs["K_grid"]
        return ("ward_labels",)

    def generate(data, **kwargs):
        observed.update(kwargs)
        return ["candidate"]

    monkeypatch.setattr(partition_starts, "hessian_weighted_ward_label_sets_torch", labels)
    monkeypatch.setattr(partition_starts, "generate_likelihood_partition_starts", generate)
    context = _context(integer_data(((1,),) * 31))
    result = partition_starts.generate_partition_initializer_pool(
        context=context, pilot_phi=context.exact_pilot,
        fit_options=resolve_fit_config(device="cpu"),
        curvature=context.exact_pilot,
    )
    assert result == ("candidate",)
    assert observed["k_grid"] == [*range(1, 16), 20, 25, 30, 31]
    assert observed["label_sets"] == ("ward_labels",)
    assert not {"classification_weight_alpha", "classification_code_weight", "use_torch",
                "include_plain_ward", "include_ward_cem", "allow_component_death"} & observed.keys()
    assert PARTITION_CEM_MAX_ITER == 8
    assert PARTITION_GENERATION_REFIT_MAX_ITER == 32
    assert PARTITION_MAX_CANDIDATES_PER_K == 5
    assert PARTITION_K_ANCHORS == (*range(1, 16), 20, 25, 30, 40, 50)
    assert FINAL_PHI_LADDER_KMAX == 30
    assert FINAL_PHI_PARENT_COUNT == 1
    assert PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT == 1.05
