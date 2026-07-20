from __future__ import annotations

import numpy as np
import pytest

from CliPP2.model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
    objective_balance_lambda,
)


_EXACT_ELIGIBILITY_FROM_LEGACY_FLAG = object()


def _config(**overrides) -> OnlineLambdaConfig:
    values = {
        "guide_n_clusters": 5,
        "num_mutations": 100,
        "kkt_tolerance": 1e-3,
        "lambda_min": 1e-4,
        "lambda_max": 1e6,
        "transition_log10_width_tolerance": 0.05,
        "max_unique_lambdas": 40,
        "max_solver_retries_per_lambda": 2,
    }
    values.update(overrides)
    return OnlineLambdaConfig(**values)


def _observation(
    proposal,
    *,
    k: int,
    score: float,
    signature: str,
    residual: float = 1e-5,
    eligible: bool = True,
    admm_iterations: int = 3,
    exact_candidate_eligible: bool | None | object = (
        _EXACT_ELIGIBILITY_FROM_LEGACY_FLAG
    ),
    backend_name: str = "admm_complete_graph",
    solver_iterations: int = 3,
) -> OnlineLambdaObservation:
    explicit_eligibility = (
        bool(eligible)
        if exact_candidate_eligible is _EXACT_ELIGIBILITY_FROM_LEGACY_FLAG
        else exact_candidate_eligible
    )
    return OnlineLambdaObservation(
        lambda_value=float(proposal.lambda_value),
        n_clusters=int(k),
        partition_signature=str(signature),
        partition_icl=float(score),
        kkt_residual=float(residual),
        raw_kkt_eligible=bool(eligible),
        admm_iterations=int(admm_iterations),
        exact_candidate_eligible=explicit_eligibility,
        certificate_status="certified" if bool(eligible) else "not_certified",
        backend_name=str(backend_name),
        solver_iterations=int(solver_iterations),
    )


def test_objective_balance_lambda_uses_crossing_without_arbitrary_fallback() -> None:
    value = objective_balance_lambda(
        pilot_loss=10.0,
        pilot_penalty_unit=8.0,
        guide_loss=16.0,
        guide_penalty_unit=5.0,
        lambda_min=1e-6,
        lambda_max=1e6,
    )
    assert value == pytest.approx(2.0)

    dominating_guide = objective_balance_lambda(
        pilot_loss=10.0,
        pilot_penalty_unit=8.0,
        guide_loss=9.0,
        guide_penalty_unit=5.0,
        lambda_min=1e-6,
        lambda_max=1e6,
    )
    assert dominating_guide == pytest.approx(1e-6)

    with pytest.raises(ValueError, match="smaller unit pairwise penalty"):
        objective_balance_lambda(
            pilot_loss=10.0,
            pilot_penalty_unit=5.0,
            guide_loss=16.0,
            guide_penalty_unit=5.0,
            lambda_min=1e-6,
            lambda_max=1e6,
        )


def test_initial_proposal_reason_can_record_guide_kkt_balance() -> None:
    default_controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    default_proposal = default_controller.propose()
    assert default_proposal is not None
    assert default_proposal.reason == "pilot_guide_objective_crossing"

    guided_controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(),
        initial_reason="partition_guide_kkt_balance",
    )
    guided_proposal = guided_controller.propose()
    assert guided_proposal is not None
    assert guided_proposal.reason == "partition_guide_kkt_balance"

    with pytest.raises(ValueError, match="initial_reason"):
        OnlineLambdaController(
            initial_lambda=10.0, config=_config(), initial_reason="  "
        )


def test_explicit_exact_certificate_is_independent_of_backend_iterations() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    initial = controller.propose()
    assert initial is not None

    controller.observe(
        _observation(
            initial,
            k=5,
            score=100.0,
            signature="quotient",
            exact_candidate_eligible=True,
            backend_name="quotient_workset_complete_graph",
            solver_iterations=0,
            admm_iterations=0,
        )
    )

    assert len(controller.observations) == 1
    assert controller.observations[0].backend_name == "quotient_workset_complete_graph"
    next_proposal = controller.propose()
    assert next_proposal is not None
    assert next_proposal.phase == "expand_upper"


def test_schema_absent_observation_keeps_legacy_dense_admm_fallback() -> None:
    accepted = OnlineLambdaController(initial_lambda=10.0, config=_config())
    accepted_initial = accepted.propose()
    assert accepted_initial is not None
    accepted.observe(
        _observation(
            accepted_initial,
            k=20,
            score=100.0,
            signature="legacy-dense",
            exact_candidate_eligible=None,
            eligible=True,
            admm_iterations=3,
        )
    )
    assert len(accepted.observations) == 1

    rejected = OnlineLambdaController(initial_lambda=10.0, config=_config())
    rejected_initial = rejected.propose()
    assert rejected_initial is not None
    rejected.observe(
        _observation(
            rejected_initial,
            k=20,
            score=100.0,
            signature="legacy-non-admm",
            exact_candidate_eligible=None,
            eligible=True,
            admm_iterations=0,
        )
    )
    assert not rejected.observations
    retry = rejected.propose()
    assert retry is not None and retry.phase == "retry_same_lambda"


def test_uncertified_exact_result_retries_then_uses_same_lambda_solver_recovery() -> (
    None
):
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    initial = controller.propose()
    assert initial is not None and initial.phase == "initial"
    controller.observe(
        _observation(initial, k=20, score=100.0, signature="a", residual=1e-1)
    )

    retry_one = controller.propose()
    assert retry_one is not None
    assert retry_one.phase == "retry_same_lambda"
    assert retry_one.lambda_value == initial.lambda_value
    assert retry_one.retry_number == 1
    controller.observe(
        _observation(retry_one, k=20, score=100.0, signature="a", eligible=False)
    )

    retry_two = controller.propose()
    assert retry_two is not None and retry_two.retry_number == 2
    controller.observe(
        _observation(
            retry_two,
            k=20,
            score=100.0,
            signature="a",
            exact_candidate_eligible=False,
            admm_iterations=0,
        )
    )

    recovery = controller.propose()
    assert recovery is not None
    assert recovery.phase == "solver_recovery"
    assert recovery.reason == "monotone_full_step_solver_recovery"
    assert recovery.lambda_value == initial.lambda_value
    assert recovery.retry_number == 3
    controller.observe(
        _observation(recovery, k=20, score=100.0, signature="a", eligible=False)
    )
    assert controller.propose() is None
    assert controller.stop_reason == "online_lambda_uncertified_exact_fusion_result"
    assert not controller.observations


def test_certified_same_lambda_solver_recovery_resumes_online_search() -> None:
    controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(max_solver_retries_per_lambda=0),
    )
    initial = controller.propose()
    controller.observe(
        _observation(initial, k=5, score=100.0, signature="guide", eligible=False)
    )

    recovery = controller.propose()
    assert recovery is not None and recovery.phase == "solver_recovery"
    assert recovery.lambda_value == initial.lambda_value
    controller.observe(_observation(recovery, k=5, score=100.0, signature="guide"))

    next_proposal = controller.propose()
    assert next_proposal is not None
    assert next_proposal.phase == "expand_upper"
    assert next_proposal.lambda_value != recovery.lambda_value
    assert len(controller.observations) == 1


def test_each_failed_online_lambda_gets_at_most_one_solver_recovery() -> None:
    controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(max_solver_retries_per_lambda=0),
    )
    initial = controller.propose()
    controller.observe(_observation(initial, k=20, score=100.0, signature="initial"))
    expansion = controller.propose()
    controller.observe(
        _observation(expansion, k=10, score=90.0, signature="failed", eligible=False)
    )

    recovery = controller.propose()
    assert recovery is not None
    assert recovery.phase == "solver_recovery"
    assert recovery.lambda_value == expansion.lambda_value
    controller.observe(
        _observation(recovery, k=10, score=90.0, signature="failed", eligible=False)
    )

    assert controller.propose() is None
    assert controller.stop_reason == "online_lambda_uncertified_exact_fusion_result"
    recoveries = [
        item for item in controller.proposal_history if item.phase == "solver_recovery"
    ]
    assert len(recoveries) == 1


def test_controller_accepts_the_solvers_five_tol_kkt_boundary() -> None:
    solver_tol = 1e-4
    controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(kkt_tolerance=5.0 * solver_tol),
    )
    initial = controller.propose()
    controller.observe(
        _observation(
            initial,
            k=20,
            score=100.0,
            signature="a",
            residual=5.0 * solver_tol,
        )
    )
    next_proposal = controller.propose()
    assert next_proposal is not None
    assert next_proposal.phase == "expand_upper"
    assert len(controller.observations) == 1


def test_first_outward_lambda_depends_on_observed_cluster_discrepancy() -> None:
    far = OnlineLambdaController(initial_lambda=10.0, config=_config())
    far_initial = far.propose()
    far.observe(_observation(far_initial, k=20, score=100.0, signature="k20"))
    far_next = far.propose()
    assert far_next is not None and far_next.phase == "expand_upper"
    assert far_next.lambda_value == pytest.approx(40.0)

    near = OnlineLambdaController(initial_lambda=10.0, config=_config())
    near_initial = near.propose()
    near.observe(_observation(near_initial, k=10, score=100.0, signature="k10"))
    near_next = near.propose()
    assert near_next is not None and near_next.phase == "expand_upper"
    assert near_next.lambda_value == pytest.approx(20.0)
    assert far_next.lambda_value != near_next.lambda_value


def test_plateau_expansion_uses_previous_step_plus_observed_k_gap() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    first = controller.propose()
    controller.observe(_observation(first, k=20, score=100.0, signature="same"))
    second = controller.propose()
    assert second is not None and second.lambda_value == pytest.approx(40.0)
    controller.observe(_observation(second, k=20, score=100.0, signature="same"))
    third = controller.propose()
    assert third is not None and third.phase == "expand_upper"
    assert third.lambda_value == pytest.approx(640.0)


def test_guide_k_plateau_doubles_observed_log_step() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    first = controller.propose()
    controller.observe(_observation(first, k=5, score=100.0, signature="same"))
    second = controller.propose()
    assert second is not None
    controller.observe(_observation(second, k=5, score=100.0, signature="same"))

    third = controller.propose()
    assert third is not None and third.phase in {"expand_lower", "expand_upper"}
    if third.lambda_value > second.lambda_value:
        expected = second.lambda_value * (second.lambda_value / first.lambda_value) ** 2
    else:
        expected = first.lambda_value * (first.lambda_value / second.lambda_value) ** 2
    assert third.lambda_value == pytest.approx(expected)


def test_observed_secant_slope_targets_guide_cluster_count() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    first = controller.propose()
    controller.observe(_observation(first, k=20, score=120.0, signature="k20"))
    second = controller.propose()
    controller.observe(_observation(second, k=10, score=110.0, signature="k10"))
    third = controller.propose()
    assert second is not None and second.lambda_value == pytest.approx(40.0)
    assert third is not None and third.phase == "expand_upper"
    assert third.lambda_value == pytest.approx(160.0)


def test_skipped_guide_k_transition_is_refined_at_geometric_midpoint() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    first = controller.propose()
    controller.observe(_observation(first, k=20, score=120.0, signature="k20"))
    second = controller.propose()
    assert second is not None and second.lambda_value == pytest.approx(40.0)
    controller.observe(_observation(second, k=2, score=110.0, signature="k2"))

    midpoint = controller.propose()
    assert midpoint is not None
    assert midpoint.phase == "refine_target_transition"
    assert midpoint.lambda_value == pytest.approx(np.sqrt(10.0 * 40.0))
    assert midpoint.bracket_left_lambda == pytest.approx(10.0)
    assert midpoint.bracket_right_lambda == pytest.approx(40.0)


def test_score_basin_requires_two_guards_then_stops_when_boundaries_resolved() -> None:
    controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(transition_log10_width_tolerance=0.3),
    )
    center = controller.propose()
    controller.observe(_observation(center, k=5, score=100.0, signature="best"))

    upper = controller.propose()
    assert upper is not None and upper.phase == "expand_upper"
    assert upper.lambda_value == pytest.approx(12.5)
    controller.observe(_observation(upper, k=4, score=110.0, signature="upper"))

    lower = controller.propose()
    assert lower is not None and lower.phase == "expand_lower"
    assert lower.lambda_value == pytest.approx(20.0 / 3.0)
    controller.observe(_observation(lower, k=6, score=120.0, signature="lower"))

    assert controller.propose() is None
    assert controller.stop_reason == "online_lambda_score_basin_resolved"
    assert controller.best_observation is not None
    assert controller.best_observation.lambda_value == pytest.approx(10.0)


def test_wide_score_basin_boundary_is_geometrically_refined() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    center = controller.propose()
    controller.observe(_observation(center, k=5, score=100.0, signature="best"))
    upper = controller.propose()
    controller.observe(_observation(upper, k=4, score=110.0, signature="upper"))
    lower = controller.propose()
    controller.observe(_observation(lower, k=6, score=120.0, signature="lower"))

    refine = controller.propose()
    assert refine is not None and refine.phase == "refine_score_basin"
    assert refine.lambda_value == pytest.approx(
        np.sqrt(lower.lambda_value * center.lambda_value)
    )
    assert refine.bracket_left_lambda == pytest.approx(lower.lambda_value)
    assert refine.bracket_right_lambda == pytest.approx(center.lambda_value)


def test_certified_cluster_count_increase_is_not_silently_accepted() -> None:
    controller = OnlineLambdaController(initial_lambda=10.0, config=_config())
    first = controller.propose()
    controller.observe(_observation(first, k=20, score=100.0, signature="a"))
    second = controller.propose()
    controller.observe(_observation(second, k=25, score=90.0, signature="b"))

    proposal = controller.propose()
    assert proposal is not None
    assert proposal.phase == "refine_inconsistency"
    assert proposal.lambda_value == pytest.approx(
        np.sqrt(first.lambda_value * second.lambda_value)
    )


def test_candidate_budget_is_a_stop_rule_not_a_hidden_grid() -> None:
    controller = OnlineLambdaController(
        initial_lambda=10.0,
        config=_config(max_unique_lambdas=1),
    )
    first = controller.propose()
    controller.observe(_observation(first, k=20, score=100.0, signature="a"))
    assert controller.propose() is None
    assert controller.stop_reason == "online_lambda_candidate_budget_reached"
