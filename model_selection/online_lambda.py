"""Sequential lambda selection for partition-guided convex clustering.

This module deliberately contains no lambda grid and no list of lambda
multipliers.  The caller supplies one data-derived positive lambda (guided
mode uses the initializer's blockwise KKT capacity). Thereafter exactly one
lambda is proposed at a time from certified ADMM observations:

* move toward the guide cluster count using the observed count discrepancy or
  a secant estimate on ``log(K)`` versus ``log(lambda)``;
* bracket a skipped cluster-count transition and bisect it geometrically;
* bracket the best observed ICL basin on both sides and geometrically resolve
  its two partition boundaries.

The controller is intentionally solver-agnostic.  The caller owns ADMM warm
starts and passes the resulting diagnostics back via
``OnlineLambdaObservation``.  An uncertified ADMM result is first retried at
the same lambda.  If any online-proposed lambda exhausts those ordinary
retries, the controller permits one monotone full-step solver recovery at that
same value.  This changes solver effort, not the lambda path; a recovered point
is observed normally before the next lambda can be proposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np


def objective_balance_lambda(
    *,
    pilot_loss: float,
    pilot_penalty_unit: float,
    guide_loss: float,
    guide_penalty_unit: float,
    lambda_min: float,
    lambda_max: float,
) -> float:
    """Return the positive pilot/guide objective-crossing lambda.

    For ``Q_lambda(phi) = loss(phi) + lambda * penalty_unit(phi)``, the
    crossing solves ``Q_lambda(pilot) == Q_lambda(guide)``.  A guide that has
    no larger loss already dominates at lambda zero, so the numerical lower
    search bound is the correct first positive value.  A guide that does not
    reduce the pairwise penalty cannot define a regularization scale and is
    rejected instead of falling back to an arbitrary constant.
    """

    values = (
        float(pilot_loss),
        float(pilot_penalty_unit),
        float(guide_loss),
        float(guide_penalty_unit),
        float(lambda_min),
        float(lambda_max),
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("Objective values and lambda bounds must be finite.")
    if lambda_min <= 0.0 or lambda_max <= lambda_min:
        raise ValueError("Require 0 < lambda_min < lambda_max.")

    penalty_reduction = float(pilot_penalty_unit - guide_penalty_unit)
    penalty_scale = 1.0 + abs(float(pilot_penalty_unit)) + abs(float(guide_penalty_unit))
    numerical_tol = np.finfo(np.float64).eps * penalty_scale
    if penalty_reduction <= numerical_tol:
        raise ValueError(
            "The partition guide must have a smaller unit pairwise penalty than the pilot."
        )

    excess_guide_loss = float(guide_loss - pilot_loss)
    if excess_guide_loss <= 0.0:
        return float(lambda_min)
    crossing = excess_guide_loss / penalty_reduction
    if not isfinite(crossing) or crossing <= 0.0:
        raise ValueError("The pilot/guide objective crossing is not a finite positive value.")
    return float(min(max(crossing, float(lambda_min)), float(lambda_max)))


@dataclass(frozen=True)
class OnlineLambdaConfig:
    """Numerical and resource limits, none of which prescribe a lambda path."""

    guide_n_clusters: int
    num_mutations: int
    kkt_tolerance: float
    lambda_min: float = 1e-6
    lambda_max: float = 1e6
    transition_log10_width_tolerance: float = 0.05
    score_relative_tolerance: float = 1e-8
    max_unique_lambdas: int = 40
    max_solver_retries_per_lambda: int = 2

    def __post_init__(self) -> None:
        if int(self.num_mutations) < 1:
            raise ValueError("num_mutations must be positive.")
        if not 1 <= int(self.guide_n_clusters) <= int(self.num_mutations):
            raise ValueError("guide_n_clusters must lie in [1, num_mutations].")
        if not isfinite(float(self.kkt_tolerance)) or float(self.kkt_tolerance) <= 0.0:
            raise ValueError("kkt_tolerance must be finite and positive.")
        if (
            not isfinite(float(self.lambda_min))
            or not isfinite(float(self.lambda_max))
            or float(self.lambda_min) <= 0.0
            or float(self.lambda_max) <= float(self.lambda_min)
        ):
            raise ValueError("Require 0 < lambda_min < lambda_max.")
        if (
            not isfinite(float(self.transition_log10_width_tolerance))
            or float(self.transition_log10_width_tolerance) <= 0.0
        ):
            raise ValueError("transition_log10_width_tolerance must be positive.")
        if not isfinite(float(self.score_relative_tolerance)) or float(self.score_relative_tolerance) < 0.0:
            raise ValueError("score_relative_tolerance must be finite and nonnegative.")
        if int(self.max_unique_lambdas) < 1:
            raise ValueError("max_unique_lambdas must be positive.")
        if int(self.max_solver_retries_per_lambda) < 0:
            raise ValueError("max_solver_retries_per_lambda must be nonnegative.")


@dataclass(frozen=True)
class OnlineLambdaObservation:
    """One raw pairwise-fusion fit and its conditional partition score."""

    lambda_value: float
    n_clusters: int
    partition_signature: str
    partition_icl: float
    kkt_residual: float
    raw_kkt_eligible: bool
    admm_iterations: int


@dataclass(frozen=True)
class OnlineLambdaProposal:
    """The next single lambda and the observations that justify it."""

    lambda_value: float
    phase: str
    reason: str
    warm_start_lambda: float | None
    alternate_start_lambda: float | None = None
    bracket_left_lambda: float | None = None
    bracket_right_lambda: float | None = None
    retry_number: int = 0


def _lambda_key(value: float) -> float:
    return float(np.round(float(value), 12))


def _log10_width(left: float, right: float) -> float:
    if not (0.0 < float(left) < float(right)):
        return 0.0
    return float((log(float(right)) - log(float(left))) / log(10.0))


def _geometric_midpoint(left: float, right: float) -> float | None:
    if not (0.0 < float(left) < float(right)):
        return None
    midpoint = exp(0.5 * (log(float(left)) + log(float(right))))
    if not (float(left) < midpoint < float(right)):
        return None
    return float(midpoint)


class OnlineLambdaController:
    """State machine for a truly online, guide-directed lambda search.

    Call ``propose()`` once, run ADMM at the returned lambda, then call
    ``observe()``.  Repeat until ``propose()`` returns ``None``.  Calling
    ``propose()`` twice without an intervening observation is an error.
    ``initial_reason`` is provenance only: callers using a guide-derived KKT
    scale can record ``"partition_guide_kkt_balance"`` without changing any
    proposal rule.

    The search is conditional on the usual one-dimensional path assumption:
    increasing lambda should not increase the number of fused groups, and ICL
    has a locally bracketable basin near the likelihood-partition guide.  A
    certified violation of cluster-count monotonicity is geometrically refined
    and reported rather than silently used for selection.
    """

    def __init__(
        self,
        *,
        initial_lambda: float,
        config: OnlineLambdaConfig,
        initial_reason: str = "pilot_guide_objective_crossing",
    ) -> None:
        initial = float(initial_lambda)
        if not isfinite(initial) or initial <= 0.0:
            raise ValueError("initial_lambda must be finite and positive.")
        normalized_initial_reason = str(initial_reason).strip()
        if not normalized_initial_reason:
            raise ValueError("initial_reason must be a non-empty string.")
        self.config = config
        self.initial_lambda = float(
            min(max(initial, float(config.lambda_min)), float(config.lambda_max))
        )
        self.initial_reason = normalized_initial_reason
        self._pending: OnlineLambdaProposal | None = None
        self._certified: dict[float, OnlineLambdaObservation] = {}
        self._last_observation: OnlineLambdaObservation | None = None
        self._retry_key: float | None = None
        self._attempt_count: dict[float, int] = {}
        self._attempted_lambda: dict[float, float] = {}
        self._proposal_history: list[OnlineLambdaProposal] = []
        self._solver_recovery_keys: set[float] = set()
        self._stop_reason: str | None = None

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def stopped(self) -> bool:
        return self._stop_reason is not None

    @property
    def observations(self) -> tuple[OnlineLambdaObservation, ...]:
        return tuple(sorted(self._certified.values(), key=lambda item: item.lambda_value))

    @property
    def proposal_history(self) -> tuple[OnlineLambdaProposal, ...]:
        return tuple(self._proposal_history)

    @property
    def best_observation(self) -> OnlineLambdaObservation | None:
        finite = [item for item in self._certified.values() if isfinite(float(item.partition_icl))]
        if not finite:
            return None
        best_score = min(float(item.partition_icl) for item in finite)
        score_tol = float(self.config.score_relative_tolerance) * (1.0 + abs(best_score))
        tied = [item for item in finite if float(item.partition_icl) <= best_score + score_tol]
        return min(
            tied,
            key=lambda item: (
                abs(log(float(item.lambda_value)) - log(self.initial_lambda)),
                abs(int(item.n_clusters) - int(self.config.guide_n_clusters)),
                float(item.lambda_value),
            ),
        )

    def _is_admm_certified(self, observation: OnlineLambdaObservation) -> bool:
        return bool(
            observation.raw_kkt_eligible
            and int(observation.admm_iterations) > 0
            and isfinite(float(observation.kkt_residual))
            and float(observation.kkt_residual) <= float(self.config.kkt_tolerance)
        )

    def _record_proposal(self, proposal: OnlineLambdaProposal) -> OnlineLambdaProposal:
        key = _lambda_key(proposal.lambda_value)
        if key not in self._attempted_lambda:
            self._attempted_lambda[key] = float(proposal.lambda_value)
        self._pending = proposal
        self._proposal_history.append(proposal)
        return proposal

    def observe(self, observation: OnlineLambdaObservation) -> None:
        """Consume the ADMM result for the outstanding proposal."""

        if self._pending is None:
            raise RuntimeError("observe() requires an outstanding lambda proposal.")
        if _lambda_key(observation.lambda_value) != _lambda_key(self._pending.lambda_value):
            raise ValueError("The observation lambda does not match the outstanding proposal.")
        if not 1 <= int(observation.n_clusters) <= int(self.config.num_mutations):
            raise ValueError("Observed n_clusters must lie in [1, num_mutations].")
        key = _lambda_key(observation.lambda_value)
        self._attempt_count[key] = int(self._attempt_count.get(key, 0) + 1)
        self._last_observation = observation
        self._pending = None
        if self._is_admm_certified(observation):
            incumbent = self._certified.get(key)
            if incumbent is None or float(observation.kkt_residual) < float(incumbent.kkt_residual):
                self._certified[key] = observation
            self._retry_key = None
        else:
            self._retry_key = key

    def propose(self) -> OnlineLambdaProposal | None:
        """Return exactly one next lambda, or ``None`` after a terminal state."""

        if self._pending is not None:
            raise RuntimeError("The outstanding proposal must be observed before proposing again.")
        if self.stopped:
            return None

        retry = self._retry_proposal()
        if retry is not None:
            return self._record_proposal(retry)
        if self.stopped:
            return None

        if not self._certified:
            return self._record_proposal(
                OnlineLambdaProposal(
                    lambda_value=self.initial_lambda,
                    phase="initial",
                    reason=self.initial_reason,
                    warm_start_lambda=None,
                )
            )
        if len(self._attempted_lambda) >= int(self.config.max_unique_lambdas):
            self._stop_reason = "online_lambda_candidate_budget_reached"
            return None

        proposal = self._choose_from_certified_path()
        if proposal is None:
            return None
        key = _lambda_key(proposal.lambda_value)
        if key in self._attempted_lambda:
            self._stop_reason = "online_lambda_no_distinct_float_available"
            return None
        return self._record_proposal(proposal)

    def _retry_proposal(self) -> OnlineLambdaProposal | None:
        if self._retry_key is None:
            return None
        attempts = int(self._attempt_count.get(self._retry_key, 0))
        if attempts > int(self.config.max_solver_retries_per_lambda):
            if self._retry_key not in self._solver_recovery_keys:
                failed = self._last_observation
                if failed is None:
                    self._stop_reason = "online_lambda_missing_failed_observation"
                    return None
                self._solver_recovery_keys.add(self._retry_key)
                return OnlineLambdaProposal(
                    lambda_value=float(failed.lambda_value),
                    phase="solver_recovery",
                    reason="monotone_full_step_admm_recovery",
                    warm_start_lambda=None,
                    retry_number=attempts,
                )
            self._stop_reason = "online_lambda_uncertified_admm_result"
            return None
        failed = self._last_observation
        if failed is None:
            self._stop_reason = "online_lambda_missing_failed_observation"
            return None
        previous = self._proposal_history[-1] if self._proposal_history else None
        return OnlineLambdaProposal(
            lambda_value=float(failed.lambda_value),
            phase="retry_same_lambda",
            reason="admm_kkt_not_certified",
            warm_start_lambda=float(failed.lambda_value),
            alternate_start_lambda=None if previous is None else previous.alternate_start_lambda,
            bracket_left_lambda=None if previous is None else previous.bracket_left_lambda,
            bracket_right_lambda=None if previous is None else previous.bracket_right_lambda,
            retry_number=attempts,
        )

    def _choose_from_certified_path(self) -> OnlineLambdaProposal | None:
        points = list(self.observations)

        inconsistency = self._unresolved_monotonicity_interval(points)
        if inconsistency is not None:
            left, right = inconsistency
            if self._interval_resolved(left, right):
                self._stop_reason = "online_lambda_nonmonotone_fusion_path"
                return None
            return self._midpoint_proposal(
                left,
                right,
                phase="refine_inconsistency",
                reason="cluster_count_increased_with_lambda",
            )

        guide_k = int(self.config.guide_n_clusters)
        if not any(int(item.n_clusters) == guide_k for item in points):
            crossing = self._guide_k_crossing(points)
            if crossing is not None:
                left, right = crossing
                if not self._interval_resolved(left, right):
                    return self._midpoint_proposal(
                        left,
                        right,
                        phase="refine_target_transition",
                        reason="guide_cluster_count_bracketed_but_skipped",
                    )
            elif all(int(item.n_clusters) > guide_k for item in points):
                return self._outward_proposal(points, direction=1, reason="observed_k_above_guide_k")
            elif all(int(item.n_clusters) < guide_k for item in points):
                return self._outward_proposal(points, direction=-1, reason="observed_k_below_guide_k")

        return self._score_basin_proposal(points)

    def _unresolved_monotonicity_interval(
        self,
        points: list[OnlineLambdaObservation],
    ) -> tuple[OnlineLambdaObservation, OnlineLambdaObservation] | None:
        violations = [
            (left, right)
            for left, right in zip(points[:-1], points[1:])
            if int(right.n_clusters) > int(left.n_clusters)
        ]
        if not violations:
            return None
        return min(
            violations,
            key=lambda pair: (
                _log10_width(pair[0].lambda_value, pair[1].lambda_value),
                pair[0].lambda_value,
            ),
        )

    def _guide_k_crossing(
        self,
        points: list[OnlineLambdaObservation],
    ) -> tuple[OnlineLambdaObservation, OnlineLambdaObservation] | None:
        guide_k = int(self.config.guide_n_clusters)
        crossings = [
            (left, right)
            for left, right in zip(points[:-1], points[1:])
            if int(left.n_clusters) > guide_k > int(right.n_clusters)
        ]
        if not crossings:
            return None
        return min(
            crossings,
            key=lambda pair: (
                _log10_width(pair[0].lambda_value, pair[1].lambda_value),
                pair[0].lambda_value,
            ),
        )

    def _interval_resolved(
        self,
        left: OnlineLambdaObservation,
        right: OnlineLambdaObservation,
    ) -> bool:
        return bool(
            _log10_width(left.lambda_value, right.lambda_value)
            <= float(self.config.transition_log10_width_tolerance)
        )

    def _midpoint_proposal(
        self,
        left: OnlineLambdaObservation,
        right: OnlineLambdaObservation,
        *,
        phase: str,
        reason: str,
    ) -> OnlineLambdaProposal | None:
        midpoint = _geometric_midpoint(left.lambda_value, right.lambda_value)
        if midpoint is None:
            self._stop_reason = "online_lambda_no_distinct_float_available"
            return None
        return OnlineLambdaProposal(
            lambda_value=float(midpoint),
            phase=str(phase),
            reason=str(reason),
            warm_start_lambda=float(left.lambda_value),
            alternate_start_lambda=float(right.lambda_value),
            bracket_left_lambda=float(left.lambda_value),
            bracket_right_lambda=float(right.lambda_value),
        )

    def _outward_proposal(
        self,
        points: list[OnlineLambdaObservation],
        *,
        direction: int,
        reason: str,
    ) -> OnlineLambdaProposal | None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1.")
        frontier = points[-1] if direction > 0 else points[0]
        neighbor = None
        if len(points) > 1:
            neighbor = points[-2] if direction > 0 else points[1]
        if direction > 0 and int(frontier.n_clusters) == 1:
            self._stop_reason = "online_lambda_upper_structural_boundary_reached"
            return None
        if direction < 0 and int(frontier.n_clusters) == int(self.config.num_mutations):
            self._stop_reason = "online_lambda_lower_structural_boundary_reached"
            return None

        candidate = self._next_outward_lambda(frontier, neighbor, direction=direction)
        if candidate is None:
            self._stop_reason = (
                "online_lambda_upper_search_bound_reached"
                if direction > 0
                else "online_lambda_lower_search_bound_reached"
            )
            return None
        return OnlineLambdaProposal(
            lambda_value=float(candidate),
            phase="expand_upper" if direction > 0 else "expand_lower",
            reason=str(reason),
            warm_start_lambda=float(frontier.lambda_value),
        )

    def _next_outward_lambda(
        self,
        frontier: OnlineLambdaObservation,
        neighbor: OnlineLambdaObservation | None,
        *,
        direction: int,
    ) -> float | None:
        x_frontier = log(float(frontier.lambda_value))
        k_frontier = int(frontier.n_clusters)
        guide_k = int(self.config.guide_n_clusters)
        if direction > 0:
            adjacent_k = max(k_frontier - 1, 1)
        else:
            adjacent_k = min(k_frontier + 1, int(self.config.num_mutations))
        discrete_resolution = abs(log(float(adjacent_k)) - log(float(k_frontier)))
        target_gap = abs(log(float(k_frontier)) - log(float(guide_k)))
        observed_gap = max(discrete_resolution, target_gap, np.finfo(np.float64).eps)

        proposed_x: float | None = None
        previous_log_step = 0.0
        if neighbor is not None:
            x_neighbor = log(float(neighbor.lambda_value))
            previous_log_step = abs(x_frontier - x_neighbor)
            delta_x = x_frontier - x_neighbor
            delta_log_k = log(float(k_frontier)) - log(float(neighbor.n_clusters))
            if abs(delta_x) > np.finfo(np.float64).eps and abs(delta_log_k) > np.finfo(np.float64).eps:
                slope = delta_log_k / delta_x
                if slope < 0.0:
                    secant_x = x_frontier + (
                        log(float(guide_k)) - log(float(k_frontier))
                    ) / slope
                    if direction * (secant_x - x_frontier) > 0.0:
                        proposed_x = x_frontier + direction * max(
                            abs(secant_x - x_frontier),
                            discrete_resolution,
                        )
        if proposed_x is None:
            if neighbor is None:
                log_step = observed_gap
            elif (
                frontier.partition_signature == neighbor.partition_signature
                and previous_log_step > 0.0
            ):
                # Repeatedly observing the identical partition supplies direct
                # evidence that the last log-step was too short to reach a
                # transition. Expand that *observed* plateau geometrically in
                # log-lambda space instead of crawling by one discrete K gap.
                # This is state-dependent and does not prescribe a lambda list.
                log_step = max(2.0 * previous_log_step, observed_gap)
            else:
                log_step = previous_log_step + observed_gap
            proposed_x = x_frontier + direction * log_step

        lower_x = log(float(self.config.lambda_min))
        upper_x = log(float(self.config.lambda_max))
        bounded_x = min(max(proposed_x, lower_x), upper_x)
        candidate = exp(bounded_x)
        if direction > 0 and candidate <= float(frontier.lambda_value) * (1.0 + 8.0 * np.finfo(float).eps):
            return None
        if direction < 0 and candidate >= float(frontier.lambda_value) * (1.0 - 8.0 * np.finfo(float).eps):
            return None
        return float(candidate)

    def _score_basin_proposal(
        self,
        points: list[OnlineLambdaObservation],
    ) -> OnlineLambdaProposal | None:
        best = self.best_observation
        if best is None:
            self._stop_reason = "online_lambda_no_finite_partition_icl"
            return None
        best_index = next(
            idx
            for idx, item in enumerate(points)
            if _lambda_key(item.lambda_value) == _lambda_key(best.lambda_value)
        )
        run_left = best_index
        while run_left > 0 and points[run_left - 1].partition_signature == best.partition_signature:
            run_left -= 1
        run_right = best_index
        while (
            run_right + 1 < len(points)
            and points[run_right + 1].partition_signature == best.partition_signature
        ):
            run_right += 1

        left_guard = points[run_left - 1] if run_left > 0 else None
        right_guard = points[run_right + 1] if run_right + 1 < len(points) else None
        left_terminal = left_guard is None and int(points[run_left].n_clusters) == int(
            self.config.num_mutations
        )
        right_terminal = right_guard is None and int(points[run_right].n_clusters) == 1

        missing_left = left_guard is None and not left_terminal
        missing_right = right_guard is None and not right_terminal
        if missing_left or missing_right:
            if missing_left and missing_right:
                lower_span = max(log(self.initial_lambda) - log(points[0].lambda_value), 0.0)
                upper_span = max(log(points[-1].lambda_value) - log(self.initial_lambda), 0.0)
                if lower_span < upper_span:
                    direction = -1
                elif upper_span < lower_span:
                    direction = 1
                else:
                    guide_k = int(self.config.guide_n_clusters)
                    if int(best.n_clusters) > guide_k:
                        direction = 1
                    elif int(best.n_clusters) < guide_k:
                        direction = -1
                    else:
                        direction = 1
            else:
                direction = -1 if missing_left else 1
            return self._outward_proposal(
                points,
                direction=direction,
                reason="bracket_best_partition_icl_basin",
            )

        boundary_intervals: list[
            tuple[OnlineLambdaObservation, OnlineLambdaObservation, OnlineLambdaObservation]
        ] = []
        if left_guard is not None:
            boundary_intervals.append((left_guard, points[run_left], left_guard))
        if right_guard is not None:
            boundary_intervals.append((points[run_right], right_guard, right_guard))
        unresolved = [
            item for item in boundary_intervals if not self._interval_resolved(item[0], item[1])
        ]
        if unresolved:
            left, right, guard = max(
                unresolved,
                key=lambda item: (
                    _log10_width(item[0].lambda_value, item[1].lambda_value),
                    -float(guard_score(item[2])),
                ),
            )
            return self._midpoint_proposal(
                left,
                right,
                phase="refine_score_basin",
                reason="resolve_best_partition_signature_boundary",
            )

        if any(not isfinite(float(guard.partition_icl)) for _, _, guard in boundary_intervals):
            self._stop_reason = "online_lambda_nonfinite_icl_guard"
            return None
        self._stop_reason = "online_lambda_score_basin_resolved"
        return None


def guard_score(observation: OnlineLambdaObservation) -> float:
    """Sort non-finite guard scores after finite scores."""

    value = float(observation.partition_icl)
    return value if isfinite(value) else float("inf")


__all__ = [
    "OnlineLambdaConfig",
    "OnlineLambdaController",
    "OnlineLambdaObservation",
    "OnlineLambdaProposal",
    "objective_balance_lambda",
]
