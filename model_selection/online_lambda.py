"""Sequential lambda selection for partition-guided convex clustering.

This module deliberately contains no lambda grid and no list of lambda
multipliers.  The caller supplies one data-derived positive lambda (guided
mode uses the initializer's blockwise KKT capacity). Thereafter exactly one
lambda is proposed at a time from certified exact-fusion observations:

* move toward the guide cluster count using the observed count discrepancy or
  a secant estimate on ``log(K)`` versus ``log(lambda)``;
* bracket a skipped cluster-count transition and bisect it geometrically;
* bracket the best observed BIC basin on both sides and geometrically resolve
  its two partition boundaries.

The controller is intentionally solver-agnostic. The caller owns solver warm
starts and passes the resulting diagnostics back via
``OnlineLambdaObservation``. An uncertified exact-fusion result is first retried at
the same lambda.  If any online-proposed lambda exhausts those ordinary
retries, the controller permits one higher-effort fixed-objective recovery at
that same value.  If the initial lambda still cannot establish any certified
raw reference, a bounded geometric sequence of distinct-lambda certification
anchors is permitted.  The runner evaluates those anchors at recovery effort
from independent cold starts; the KKT gate and fixed objective remain
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np


ONLINE_LAMBDA_STATE_VERSION = 1


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
    # Exploration and boundary/event refinement have separate budgets.  A
    # refinement must not silently consume the last opportunity to explore a
    # new part of the lambda path (or vice versa).
    max_unique_lambdas: int = 40
    max_refinement_lambdas: int = 40
    max_solver_retries_per_lambda: int = 2
    # If the guide-derived first lambda remains uncertified after ordinary
    # retry and recovery, permit a small number of independent certification
    # anchors before failing the entire raw path.  These probes are separate
    # from the statistical exploration budget because no path exists yet.
    max_bootstrap_anchor_lambdas: int = 3
    # Applies only to certified statistical refinement proposals.  Initial
    # path exploration remains exhaustive within its independent budget.
    no_progress_patience: int = 3

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
        if (
            not isfinite(float(self.score_relative_tolerance))
            or float(self.score_relative_tolerance) < 0.0
        ):
            raise ValueError("score_relative_tolerance must be finite and nonnegative.")
        if int(self.max_unique_lambdas) < 1:
            raise ValueError("max_unique_lambdas must be positive.")
        if int(self.max_refinement_lambdas) < 0:
            raise ValueError("max_refinement_lambdas must be nonnegative.")
        if int(self.max_solver_retries_per_lambda) < 0:
            raise ValueError("max_solver_retries_per_lambda must be nonnegative.")
        if int(self.max_bootstrap_anchor_lambdas) < 0:
            raise ValueError("max_bootstrap_anchor_lambdas must be nonnegative.")
        if int(self.no_progress_patience) < 1:
            raise ValueError("no_progress_patience must be positive.")


@dataclass(frozen=True)
class OnlineLambdaObservation:
    """One raw pairwise-fusion fit and its conditional partition score."""

    lambda_value: float
    n_clusters: int
    partition_signature: str
    partition_bic: float
    kkt_residual: float
    raw_objective_certified: bool
    partition_certified: bool
    selection_score_available: bool
    score_numerical_uncertainty: float = 0.0
    degrees_of_freedom: int = 0


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


@dataclass(frozen=True, slots=True)
class OnlineLambdaState:
    """Immutable checkpoint snapshot for the partition-event controller."""

    version: int
    config: OnlineLambdaConfig
    initial_lambda: float
    initial_reason: str
    certified: tuple[tuple[float, OnlineLambdaObservation], ...]
    last_observation: OnlineLambdaObservation | None
    retry_key: float | None
    attempts: tuple[tuple[float, float, str, int], ...]
    proposal_history: tuple[OnlineLambdaProposal, ...]
    solver_recovery_keys: tuple[float, ...]
    bootstrap_anchor_keys: tuple[float, ...]
    uncertified_exhausted_keys: tuple[float, ...]
    no_progress_streak: int
    stop_reason: str | None


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


def _is_refinement_phase(phase: str) -> bool:
    """Return whether a proposal spends the dedicated refinement budget."""

    return str(phase).startswith("refine_")


class OnlineLambdaController:
    """State machine for a truly online, guide-directed lambda search.

    Call ``propose()`` once, run the fusion solver at the returned lambda, then call
    ``observe()``.  Repeat until ``propose()`` returns ``None``.  Calling
    ``propose()`` twice without an intervening observation is an error.
    ``initial_reason`` is provenance only: callers using a guide-derived KKT
    scale can record ``"partition_guide_kkt_balance"`` without changing any
    proposal rule.

    The production search treats every change in partition identity as a model
    event and resolves those events without assuming monotone cluster counts.
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
        self._attempted_phase: dict[float, str] = {}
        self._proposal_history: list[OnlineLambdaProposal] = []
        self._solver_recovery_keys: set[float] = set()
        self._bootstrap_anchor_keys: set[float] = set()
        self._uncertified_exhausted_keys: set[float] = set()
        self._no_progress_streak = 0
        self._stop_reason: str | None = None

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    @property
    def stopped(self) -> bool:
        return self._stop_reason is not None

    @property
    def observations(self) -> tuple[OnlineLambdaObservation, ...]:
        return tuple(
            sorted(self._certified.values(), key=lambda item: item.lambda_value)
        )

    def snapshot(self) -> OnlineLambdaState:
        """Freeze the controller between complete proposal transactions."""

        if self._pending is not None:
            raise RuntimeError(
                "OnlineLambdaController can be checkpointed only after the "
                "outstanding proposal has been observed."
            )
        return OnlineLambdaState(
            version=ONLINE_LAMBDA_STATE_VERSION,
            config=self.config,
            initial_lambda=float(self.initial_lambda),
            initial_reason=str(self.initial_reason),
            certified=tuple(sorted(self._certified.items())),
            last_observation=self._last_observation,
            retry_key=self._retry_key,
            attempts=tuple(
                (
                    float(key),
                    float(value),
                    str(self._attempted_phase[key]),
                    int(self._attempt_count[key]),
                )
                for key, value in self._attempted_lambda.items()
            ),
            proposal_history=tuple(self._proposal_history),
            solver_recovery_keys=tuple(sorted(self._solver_recovery_keys)),
            bootstrap_anchor_keys=tuple(sorted(self._bootstrap_anchor_keys)),
            uncertified_exhausted_keys=tuple(
                sorted(self._uncertified_exhausted_keys)
            ),
            no_progress_streak=int(self._no_progress_streak),
            stop_reason=self._stop_reason,
        )

    @classmethod
    def from_snapshot(cls, state: OnlineLambdaState) -> "OnlineLambdaController":
        """Restore one typed snapshot, validating cross-field invariants."""

        if not isinstance(state, OnlineLambdaState):
            raise TypeError("controller snapshot must be OnlineLambdaState.")
        if int(state.version) != ONLINE_LAMBDA_STATE_VERSION:
            raise ValueError("Unsupported online-lambda state version.")
        controller = cls(
            initial_lambda=float(state.initial_lambda),
            config=state.config,
            initial_reason=str(state.initial_reason),
        )
        if controller.initial_lambda != float(state.initial_lambda):
            raise ValueError("Stored initial lambda lies outside its configured bounds.")
        controller._certified = dict(state.certified)
        controller._last_observation = state.last_observation
        controller._retry_key = state.retry_key
        controller._attempted_lambda = {
            float(key): float(value) for key, value, _, _ in state.attempts
        }
        controller._attempted_phase = {
            float(key): str(phase) for key, _, phase, _ in state.attempts
        }
        controller._attempt_count = {
            float(key): int(count) for key, _, _, count in state.attempts
        }
        if len(controller._attempted_lambda) != len(state.attempts):
            raise ValueError("Controller snapshot contains duplicate attempt keys.")
        controller._proposal_history = list(state.proposal_history)
        controller._solver_recovery_keys = set(state.solver_recovery_keys)
        controller._bootstrap_anchor_keys = set(state.bootstrap_anchor_keys)
        controller._uncertified_exhausted_keys = set(
            state.uncertified_exhausted_keys
        )
        if any(
            len(values) != len(set(values))
            for values in (
                state.solver_recovery_keys,
                state.bootstrap_anchor_keys,
                state.uncertified_exhausted_keys,
            )
        ):
            raise ValueError("Controller snapshot contains duplicate recovery keys.")
        controller._no_progress_streak = int(state.no_progress_streak)
        controller._stop_reason = state.stop_reason
        controller._validate_restored_state()
        return controller

    def _validate_restored_state(self) -> None:
        """Validate relationships that individual field decoders cannot see."""

        attempted_keys = set(self._attempted_lambda)
        if attempted_keys != set(self._attempted_phase) or attempted_keys != set(
            self._attempt_count
        ):
            raise ValueError("Controller attempted-lambda maps have different keys.")
        for key, value in self._attempted_lambda.items():
            if (
                not isfinite(float(value))
                or not float(self.config.lambda_min)
                <= float(value)
                <= float(self.config.lambda_max)
                or _lambda_key(value) != key
            ):
                raise ValueError("Controller state contains a noncanonical lambda key.")

        history_counts: dict[float, int] = {}
        history_first_phase: dict[float, str] = {}
        for proposal in self._proposal_history:
            key = _lambda_key(proposal.lambda_value)
            if key not in attempted_keys:
                raise ValueError("Proposal history contains an unrecorded lambda.")
            history_counts[key] = int(history_counts.get(key, 0) + 1)
            history_first_phase.setdefault(key, str(proposal.phase))
        if history_counts != self._attempt_count:
            raise ValueError("Proposal history does not reproduce attempt counts.")
        if history_first_phase != self._attempted_phase:
            raise ValueError("Proposal history does not reproduce first-attempt phases.")

        for key, observation in self._certified.items():
            if key not in attempted_keys or _lambda_key(observation.lambda_value) != key:
                raise ValueError("Certified observation has inconsistent lambda identity.")
            if not 1 <= int(observation.n_clusters) <= int(
                self.config.num_mutations
            ):
                raise ValueError("Certified observation has an invalid cluster count.")
            if not self._is_exact_fusion_certified(observation):
                raise ValueError("Certified observation fails the controller KKT rule.")

        tracked_key_sets = (
            self._solver_recovery_keys,
            self._bootstrap_anchor_keys,
            self._uncertified_exhausted_keys,
        )
        if any(not keys.issubset(attempted_keys) for keys in tracked_key_sets):
            raise ValueError("Controller recovery state refers to an unattempted lambda.")
        if len(self._bootstrap_anchor_keys) > int(
            self.config.max_bootstrap_anchor_lambdas
        ):
            raise ValueError("Controller state exceeds its bootstrap-anchor budget.")
        if (
            self._stop_reason == "online_lambda_no_meaningful_progress"
            and int(self._no_progress_streak)
            < int(self.config.no_progress_patience)
        ):
            raise ValueError(
                "No-progress stop reason lacks the required certified streak."
            )

        if not self._proposal_history:
            if (
                self._last_observation is not None
                or attempted_keys
                or self._certified
                or any(tracked_key_sets)
                or self._retry_key is not None
                or int(self._no_progress_streak) != 0
                or self._stop_reason is not None
            ):
                raise ValueError("Empty proposal history has nonempty controller state.")
            return
        if self._last_observation is None:
            raise ValueError("Proposal history lacks its last observation.")
        last_key = _lambda_key(self._last_observation.lambda_value)
        if last_key != _lambda_key(self._proposal_history[-1].lambda_value):
            raise ValueError("Last observation does not match the last proposal.")
        if self._retry_key is not None:
            if self._retry_key != last_key or self._is_exact_fusion_certified(
                self._last_observation
            ):
                raise ValueError("Retry key does not identify the failed observation.")
        elif (
            self._stop_reason is None
            and not self._is_exact_fusion_certified(self._last_observation)
        ):
            raise ValueError("Uncertified active state lacks its retry key.")

    @property
    def best_observation(self) -> OnlineLambdaObservation | None:
        finite = [
            item
            for item in self._certified.values()
            if self._selection_score_is_available(item)
        ]
        if not finite:
            return None
        minimum_upper = min(
            float(item.partition_bic)
            + max(float(item.score_numerical_uncertainty), 0.0)
            for item in finite
        )
        tied = [
            item
            for item in finite
            if float(item.partition_bic)
            - max(float(item.score_numerical_uncertainty), 0.0)
            <= minimum_upper
        ]
        return min(
            tied,
            key=lambda item: (
                int(item.n_clusters),
                int(item.degrees_of_freedom),
                float(item.lambda_value),
                str(item.partition_signature),
            ),
        )

    def _is_exact_fusion_certified(self, observation: OnlineLambdaObservation) -> bool:
        return bool(
            observation.raw_objective_certified
            and isfinite(float(observation.kkt_residual))
            and float(observation.kkt_residual) <= float(self.config.kkt_tolerance)
        )

    @staticmethod
    def _selection_score_is_available(
        observation: OnlineLambdaObservation,
    ) -> bool:
        return bool(
            observation.selection_score_available
            and isfinite(float(observation.partition_bic))
        )

    def _record_proposal(self, proposal: OnlineLambdaProposal) -> OnlineLambdaProposal:
        key = _lambda_key(proposal.lambda_value)
        if key not in self._attempted_lambda:
            self._attempted_lambda[key] = float(proposal.lambda_value)
            self._attempted_phase[key] = str(proposal.phase)
        self._pending = proposal
        self._proposal_history.append(proposal)
        return proposal

    def _progress_metrics(
        self,
    ) -> tuple[int, int, float, float, tuple[float, ...]]:
        """Return deterministic search progress from certified observations.

        The tuple records unique scored partitions, the controller's own event
        signatures, best score/KKT values, and the descending vector of every
        unresolved event width.  Using the full width vector recognizes a
        bisection even when another equally wide event remains.
        """

        certified = sorted(
            self._certified.values(), key=lambda item: float(item.lambda_value)
        )
        scored = [
            item for item in certified if self._selection_score_is_available(item)
        ]
        unique_scored = len({str(item.partition_signature) for item in scored})
        unique_events = len(
            {self._event_signature(item) for item in certified}
        )
        best_score_upper = min(
            (
                float(item.partition_bic)
                + max(float(item.score_numerical_uncertainty), 0.0)
                for item in scored
            ),
            default=float("inf"),
        )
        best_kkt = min(
            (float(item.kkt_residual) for item in certified),
            default=float("inf"),
        )
        event_widths = tuple(
            sorted(
                (
                    _log10_width(
                        float(left.lambda_value), float(right.lambda_value)
                    )
                    for left, right in zip(
                        certified[:-1], certified[1:], strict=True
                    )
                    if self._event_signature(left) != self._event_signature(right)
                    and _log10_width(
                        float(left.lambda_value), float(right.lambda_value)
                    )
                    > float(self.config.transition_log10_width_tolerance)
                ),
                reverse=True,
            )
        )
        return (
            unique_scored,
            unique_events,
            best_score_upper,
            best_kkt,
            event_widths,
        )

    def _progress_improved(
        self,
        before: tuple[int, int, float, float, tuple[float, ...]],
        after: tuple[int, int, float, float, tuple[float, ...]],
    ) -> bool:
        if int(after[0]) > int(before[0]):
            return True
        if int(after[1]) > int(before[1]):
            return True
        before_score, after_score = float(before[2]), float(after[2])
        if isfinite(after_score) and (
            not isfinite(before_score)
            or after_score
            < before_score
            - float(self.config.score_relative_tolerance)
            * (1.0 + abs(before_score))
        ):
            return True
        before_kkt, after_kkt = float(before[3]), float(after[3])
        if isfinite(after_kkt) and (
            not isfinite(before_kkt) or after_kkt < 0.95 * before_kkt
        ):
            return True
        before_widths, after_widths = before[4], after[4]
        if not before_widths:
            return False
        return bool(after_widths < before_widths)

    def observe(self, observation: OnlineLambdaObservation) -> None:
        """Consume the exact-fusion result for the outstanding proposal."""

        if self._pending is None:
            raise RuntimeError("observe() requires an outstanding lambda proposal.")
        if _lambda_key(observation.lambda_value) != _lambda_key(
            self._pending.lambda_value
        ):
            raise ValueError(
                "The observation lambda does not match the outstanding proposal."
            )
        if not 1 <= int(observation.n_clusters) <= int(self.config.num_mutations):
            raise ValueError("Observed n_clusters must lie in [1, num_mutations].")
        pending_phase = str(self._pending.phase)
        progress_before = self._progress_metrics()
        key = _lambda_key(observation.lambda_value)
        self._attempt_count[key] = int(self._attempt_count.get(key, 0) + 1)
        self._last_observation = observation
        self._pending = None
        if self._is_exact_fusion_certified(observation):
            incumbent = self._certified.get(key)
            observation_scored = self._selection_score_is_available(observation)
            incumbent_scored = bool(
                incumbent is not None and self._selection_score_is_available(incumbent)
            )
            if (
                incumbent is None
                or (observation_scored and not incumbent_scored)
                or (
                    observation_scored == incumbent_scored
                    and float(observation.kkt_residual) < float(incumbent.kkt_residual)
                )
            ):
                self._certified[key] = observation
            self._retry_key = None
            progress_after = self._progress_metrics()
            if self._progress_improved(progress_before, progress_after):
                self._no_progress_streak = 0
            elif _is_refinement_phase(pending_phase):
                self._no_progress_streak += 1
                if int(self._no_progress_streak) >= int(
                    self.config.no_progress_patience
                ):
                    self._stop_reason = "online_lambda_no_meaningful_progress"
        else:
            self._retry_key = key

    def propose(self) -> OnlineLambdaProposal | None:
        """Return exactly one next lambda, or ``None`` after a terminal state."""

        if self._pending is not None:
            raise RuntimeError(
                "The outstanding proposal must be observed before proposing again."
            )
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
        proposal = self._choose_from_certified_path()
        if proposal is None:
            return None
        key = _lambda_key(proposal.lambda_value)
        if key in self._attempted_lambda:
            self._stop_reason = "online_lambda_no_distinct_float_available"
            return None
        statistical_phases = tuple(
            phase
            for phase in self._attempted_phase.values()
            if str(phase) != "bootstrap_certification_anchor"
        )
        refinement_count = sum(
            _is_refinement_phase(phase) for phase in statistical_phases
        )
        exploration_count = len(statistical_phases) - int(refinement_count)
        if _is_refinement_phase(proposal.phase) and exploration_count < int(
            self.config.max_unique_lambdas
        ):
            # Establish broad path coverage before spending the independent
            # event-resolution budget.  Refining the first local transition
            # encountered can otherwise consume the small balanced budget and
            # hide a remote, better partition behind two worse guards.
            fallback = self._budget_fallback_exploration()
            if fallback is not None:
                proposal = fallback
        if _is_refinement_phase(proposal.phase):
            if refinement_count >= int(self.config.max_refinement_lambdas):
                # A locally preferred refinement must not strand unused
                # exploration capacity.  Continue widening the observed path
                # when possible; only stop once that independent budget is
                # also unavailable.
                if exploration_count < int(self.config.max_unique_lambdas):
                    fallback = self._budget_fallback_exploration()
                    if fallback is not None:
                        proposal = fallback
                    else:
                        self._stop_reason = "online_lambda_refinement_budget_reached"
                        return None
                else:
                    self._stop_reason = "online_lambda_refinement_budget_reached"
                    return None
        elif exploration_count >= int(self.config.max_unique_lambdas):
            self._stop_reason = "online_lambda_candidate_budget_reached"
            return None
        key = _lambda_key(proposal.lambda_value)
        if key in self._attempted_lambda:
            self._stop_reason = "online_lambda_no_distinct_float_available"
            return None
        return self._record_proposal(proposal)

    def _budget_fallback_exploration(self) -> OnlineLambdaProposal | None:
        """Use spare exploration capacity after local refinement is exhausted."""

        points = list(self.observations)
        if not points:
            return None
        lower_span = max(log(self.initial_lambda) - log(points[0].lambda_value), 0.0)
        upper_span = max(log(points[-1].lambda_value) - log(self.initial_lambda), 0.0)
        directions = (-1, 1) if lower_span <= upper_span else (1, -1)
        original_stop_reason = self._stop_reason
        for direction in directions:
            self._stop_reason = None
            proposal = self._event_outward_proposal(
                points,
                direction=direction,
                reason="refinement_budget_exhausted_continue_union_exploration",
            )
            if proposal is not None:
                return proposal
        self._stop_reason = original_stop_reason
        return None

    def _retry_proposal(self) -> OnlineLambdaProposal | None:
        if self._retry_key is None:
            return None
        attempts = int(self._attempt_count.get(self._retry_key, 0))
        if self._retry_key in self._bootstrap_anchor_keys:
            # Each bootstrap anchor is already evaluated at recovery effort
            # with the runner's independent cold-start bank.  Move outward to
            # the next geometric anchor instead of retrying its correlated
            # terminal state.
            bootstrap = self._bootstrap_anchor_proposal(self._last_observation)
            if bootstrap is not None:
                self._uncertified_exhausted_keys.add(self._retry_key)
                self._retry_key = None
                self._bootstrap_anchor_keys.add(_lambda_key(bootstrap.lambda_value))
                return bootstrap
            self._stop_reason = "online_lambda_bootstrap_anchor_uncertified"
            return None
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
                    reason="higher_effort_fixed_objective_solver_recovery",
                    warm_start_lambda=None,
                    retry_number=attempts,
                )
            if self._certified:
                # One locally uncertifiable lambda must not erase already
                # certified path evidence or prevent refinement of other
                # observed partition events.  Record it as an impassable
                # numerical boundary and continue elsewhere; the final search
                # remains explicitly unresolved.
                self._uncertified_exhausted_keys.add(self._retry_key)
                self._retry_key = None
                return None
            bootstrap = self._bootstrap_anchor_proposal(self._last_observation)
            if bootstrap is not None:
                self._uncertified_exhausted_keys.add(self._retry_key)
                self._retry_key = None
                self._bootstrap_anchor_keys.add(_lambda_key(bootstrap.lambda_value))
                return bootstrap
            self._stop_reason = "online_lambda_uncertified_exact_fusion_result"
            return None
        failed = self._last_observation
        if failed is None:
            self._stop_reason = "online_lambda_missing_failed_observation"
            return None
        previous = self._proposal_history[-1] if self._proposal_history else None
        return OnlineLambdaProposal(
            lambda_value=float(failed.lambda_value),
            phase="retry_same_lambda",
            reason="exact_fusion_kkt_not_certified",
            warm_start_lambda=float(failed.lambda_value),
            alternate_start_lambda=None
            if previous is None
            else previous.alternate_start_lambda,
            bracket_left_lambda=None
            if previous is None
            else previous.bracket_left_lambda,
            bracket_right_lambda=None
            if previous is None
            else previous.bracket_right_lambda,
            retry_number=attempts,
        )

    def _bootstrap_anchor_proposal(
        self,
        failed: OnlineLambdaObservation | None,
    ) -> OnlineLambdaProposal | None:
        """Return the next geometric lambda for raw-provenance bootstrap.

        A higher lambda is tried first because its stronger fusion penalty is a
        numerically simpler certification target. Distinct probes then
        alternate around the initial value at offsets ``+1, -1, +2, -2, ...``
        in log-lambda space. This keeps the nearest lower anchor from being
        skipped and treats clipped boundary duplicates deterministically. The
        runner evaluates each proposal at recovery effort with independent
        guide, zero-penalty, and pooled starts.
        """

        if failed is None or len(self._bootstrap_anchor_keys) >= int(
            self.config.max_bootstrap_anchor_lambdas
        ):
            return None
        for distance in range(
            1,
            int(self.config.max_bootstrap_anchor_lambdas) + 1,
        ):
            for direction in (1.0, -1.0):
                candidate = float(
                    min(
                        max(
                            float(self.initial_lambda)
                            * exp(direction * float(distance)),
                            float(self.config.lambda_min),
                        ),
                        float(self.config.lambda_max),
                    )
                )
                key = _lambda_key(candidate)
                if (
                    key == _lambda_key(float(failed.lambda_value))
                    or key in self._attempted_lambda
                ):
                    continue
                return OnlineLambdaProposal(
                    lambda_value=candidate,
                    phase="bootstrap_certification_anchor",
                    reason="initial_lambda_uncertified_probe_distinct_anchor",
                    warm_start_lambda=None,
                    retry_number=0,
                )
        return None

    def _choose_from_certified_path(self) -> OnlineLambdaProposal | None:
        points = list(self.observations)
        return self._choose_from_partition_events(points)

    @staticmethod
    def _event_signature(observation: OnlineLambdaObservation) -> tuple[object, ...]:
        return (
            str(observation.partition_signature),
            int(observation.n_clusters),
            bool(observation.partition_certified),
            bool(OnlineLambdaController._selection_score_is_available(observation)),
        )

    def _choose_from_partition_events(
        self,
        points: list[OnlineLambdaObservation],
    ) -> OnlineLambdaProposal | None:
        """Refine partition events without assuming a monotone guide-K path."""

        best = self.best_observation
        event_intervals: list[
            tuple[OnlineLambdaObservation, OnlineLambdaObservation]
        ] = []
        for left, right in zip(points[:-1], points[1:]):
            if self._event_signature(left) == self._event_signature(right):
                continue
            # Endpoint scores cannot rule out an unobserved partition inside
            # the interval.  Retain every unresolved model event; otherwise a
            # remote optimum can be hidden behind two inferior endpoint
            # partitions.  The resource budget controls how many are refined.
            if not self._interval_resolved(left, right):
                event_intervals.append((left, right))
        if event_intervals:
            left, right = max(
                event_intervals,
                key=lambda pair: (
                    _log10_width(pair[0].lambda_value, pair[1].lambda_value),
                    abs(
                        log(float(pair[1].n_clusters)) - log(float(pair[0].n_clusters))
                    ),
                    -float(pair[0].lambda_value),
                ),
            )
            return self._midpoint_proposal(
                left,
                right,
                phase="refine_partition_event",
                reason="partition_event",
            )

        if best is None:
            direction = -1 if len(points) % 2 == 0 else 1
            return self._event_outward_proposal(
                points,
                direction=direction,
                reason="seek_first_available_partition_score",
            )

        best_index = points.index(best)
        best_event = self._event_signature(best)
        run_left = best_index
        while (
            run_left > 0 and self._event_signature(points[run_left - 1]) == best_event
        ):
            run_left -= 1
        run_right = best_index
        while (
            run_right + 1 < len(points)
            and self._event_signature(points[run_right + 1]) == best_event
        ):
            run_right += 1
        missing_left = run_left == 0
        missing_right = run_right + 1 == len(points)
        if missing_left or missing_right:
            if missing_left and missing_right:
                lower_span = max(
                    log(self.initial_lambda) - log(points[0].lambda_value), 0.0
                )
                upper_span = max(
                    log(points[-1].lambda_value) - log(self.initial_lambda), 0.0
                )
                direction = -1 if lower_span <= upper_span else 1
            else:
                direction = -1 if missing_left else 1
            return self._event_outward_proposal(
                points,
                direction=direction,
                reason="bracket_best_partition_event",
            )

        self._stop_reason = "online_lambda_partition_event_basin_resolved"
        return None

    def _event_outward_proposal(
        self,
        points: list[OnlineLambdaObservation],
        *,
        direction: int,
        reason: str,
        allow_opposite: bool = True,
    ) -> OnlineLambdaProposal | None:
        frontier = points[-1] if direction > 0 else points[0]
        neighbor = (
            points[-2]
            if direction > 0 and len(points) > 1
            else points[1]
            if direction < 0 and len(points) > 1
            else None
        )
        previous_step = (
            1.0
            if neighbor is None
            else abs(log(frontier.lambda_value) - log(neighbor.lambda_value))
        )
        step = max(previous_step, 1.0)
        candidate = exp(log(frontier.lambda_value) + direction * step)
        candidate = min(
            max(candidate, float(self.config.lambda_min)),
            float(self.config.lambda_max),
        )
        if _lambda_key(candidate) in self._uncertified_exhausted_keys:
            if allow_opposite:
                return self._event_outward_proposal(
                    points,
                    direction=-int(direction),
                    reason=f"{reason}_opposite_of_uncertified_boundary",
                    allow_opposite=False,
                )
            self._stop_reason = "online_lambda_uncertified_boundaries_exhausted"
            return None
        if _lambda_key(candidate) == _lambda_key(frontier.lambda_value):
            self._stop_reason = (
                "online_lambda_upper_search_bound_reached"
                if direction > 0
                else "online_lambda_lower_search_bound_reached"
            )
            return None
        return OnlineLambdaProposal(
            lambda_value=float(candidate),
            phase="expand_union_upper" if direction > 0 else "expand_union_lower",
            reason=str(reason),
            warm_start_lambda=float(frontier.lambda_value),
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

__all__ = [
    "ONLINE_LAMBDA_STATE_VERSION",
    "OnlineLambdaConfig",
    "OnlineLambdaController",
    "OnlineLambdaObservation",
    "OnlineLambdaProposal",
    "OnlineLambdaState",
]
