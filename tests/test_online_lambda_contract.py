from __future__ import annotations

import unittest

from CliPP2.model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
)


class OnlineLambdaContractTests(unittest.TestCase):
    @staticmethod
    def _controller() -> OnlineLambdaController:
        return OnlineLambdaController(
            initial_lambda=10.0,
            config=OnlineLambdaConfig(
                guide_n_clusters=2,
                num_mutations=4,
                kkt_tolerance=1e-3,
                max_unique_lambdas=8,
                max_solver_retries_per_lambda=1,
            ),
        )

    def test_partition_failure_does_not_retry_certified_raw_fit(self) -> None:
        controller = self._controller()
        initial = controller.propose()
        self.assertIsNotNone(initial)
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=initial.lambda_value,
                n_clusters=2,
                partition_signature="uncertified-partition",
                partition_icl=float("inf"),
                kkt_residual=1e-5,
                raw_objective_certified=True,
                partition_certified=False,
                selection_score_available=False,
            )
        )
        next_proposal = controller.propose()
        self.assertIsNotNone(next_proposal)
        self.assertNotIn(next_proposal.phase, {"retry_same_lambda", "solver_recovery"})
        self.assertNotEqual(next_proposal.lambda_value, initial.lambda_value)

    def test_raw_kkt_failure_still_retries_same_lambda(self) -> None:
        controller = self._controller()
        initial = controller.propose()
        self.assertIsNotNone(initial)
        controller.observe(
            OnlineLambdaObservation(
                lambda_value=initial.lambda_value,
                n_clusters=2,
                partition_signature="raw-failed",
                partition_icl=float("inf"),
                kkt_residual=1e-1,
                raw_objective_certified=False,
                partition_certified=False,
                selection_score_available=False,
            )
        )
        retry = controller.propose()
        self.assertIsNotNone(retry)
        self.assertEqual(retry.phase, "retry_same_lambda")
        self.assertEqual(retry.lambda_value, initial.lambda_value)

    def test_branch_switch_is_not_a_monotonicity_failure(self) -> None:
        controller = self._controller()
        left = OnlineLambdaObservation(
            lambda_value=1.0,
            n_clusters=2,
            partition_signature="left",
            partition_icl=10.0,
            kkt_residual=1e-5,
            raw_objective_certified=True,
            branch_signature="clonal-block-a",
        )
        right = OnlineLambdaObservation(
            lambda_value=2.0,
            n_clusters=3,
            partition_signature="right",
            partition_icl=9.0,
            kkt_residual=1e-5,
            raw_objective_certified=True,
            branch_signature="clonal-block-b",
        )
        self.assertIsNone(
            controller._unresolved_monotonicity_interval([left, right])
        )
        same_branch_right = OnlineLambdaObservation(
            **{
                **right.__dict__,
                "branch_signature": "clonal-block-a",
            }
        )
        self.assertEqual(
            controller._unresolved_monotonicity_interval(
                [left, same_branch_right]
            ),
            (left, same_branch_right),
        )


if __name__ == "__main__":
    unittest.main()
