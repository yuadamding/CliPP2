"""Sequential lambda selection for partition-guided convex clustering.

This module deliberately contains no lambda grid and no list of lambda
multipliers.  The caller supplies one data-derived positive lambda (guided
mode uses the initializer's blockwise KKT capacity). Thereafter exactly one
lambda is proposed at a time from certified exact-fusion observations:

* move toward the guide cluster count using the observed count discrepancy or
  a secant estimate on ``log(K)`` versus ``log(lambda)``;
* bracket a skipped cluster-count transition and bisect it geometrically;
* bracket the best observed ICL basin on both sides and geometrically resolve
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

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np


ONLINE_LAMBDA_STATE_SCHEMA = "clipp2.online-lambda-controller.v1"
ONLINE_LAMBDA_STATE_SCHEMA_VERSION = 1


def _encode_float(value: float) -> str:
    """Encode one IEEE-754 value exactly without non-standard JSON numbers."""

    return float(value).hex()


def _decode_float(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a hexadecimal float string.")
    try:
        return float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid hexadecimal float string.") from exc


def _encode_optional_float(value: float | None) -> str | None:
    return None if value is None else _encode_float(value)


def _decode_optional_float(value: object, *, field: str) -> float | None:
    return None if value is None else _decode_float(value, field=field)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field} must be a string-keyed mapping.")
    return value


def _sequence(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    return value


def _integer(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer.")
    result = int(value)
    if minimum is not None and result < int(minimum):
        raise ValueError(f"{field} must be at least {minimum}.")
    return result


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean.")
    return bool(value)


def _text(value: object, *, field: str, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        requirement = "a nonempty string" if nonempty else "a string"
        raise ValueError(f"{field} must be {requirement}.")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field=field, nonempty=True)


def _require_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    field: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{field} has incompatible keys; missing={missing}, "
            f"unexpected={unexpected}."
        )


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
    partition_event_mode: bool = False

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


@dataclass(frozen=True)
class OnlineLambdaObservation:
    """One raw pairwise-fusion fit and its conditional partition score."""

    lambda_value: float
    n_clusters: int
    partition_signature: str
    partition_icl: float
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


_CONFIG_STATE_KEYS = {
    "guide_n_clusters",
    "num_mutations",
    "kkt_tolerance",
    "lambda_min",
    "lambda_max",
    "transition_log10_width_tolerance",
    "score_relative_tolerance",
    "max_unique_lambdas",
    "max_refinement_lambdas",
    "max_solver_retries_per_lambda",
    "max_bootstrap_anchor_lambdas",
    "partition_event_mode",
}
_OBSERVATION_STATE_KEYS = {
    "lambda_value",
    "n_clusters",
    "partition_signature",
    "partition_icl",
    "kkt_residual",
    "raw_objective_certified",
    "partition_certified",
    "selection_score_available",
    "score_numerical_uncertainty",
    "degrees_of_freedom",
}
_PROPOSAL_STATE_KEYS = {
    "lambda_value",
    "phase",
    "reason",
    "warm_start_lambda",
    "alternate_start_lambda",
    "bracket_left_lambda",
    "bracket_right_lambda",
    "retry_number",
}
_CONTROLLER_STATE_KEYS = {
    "schema",
    "schema_version",
    "pending",
    "config",
    "initial_lambda",
    "initial_reason",
    "certified",
    "last_observation",
    "retry_key",
    "attempt_count",
    "attempted_lambdas",
    "proposal_history",
    "solver_recovery_keys",
    "bootstrap_anchor_keys",
    "uncertified_exhausted_keys",
    "stop_reason",
}


def _config_to_state(config: OnlineLambdaConfig) -> dict[str, object]:
    return {
        "guide_n_clusters": int(config.guide_n_clusters),
        "num_mutations": int(config.num_mutations),
        "kkt_tolerance": _encode_float(config.kkt_tolerance),
        "lambda_min": _encode_float(config.lambda_min),
        "lambda_max": _encode_float(config.lambda_max),
        "transition_log10_width_tolerance": _encode_float(
            config.transition_log10_width_tolerance
        ),
        "score_relative_tolerance": _encode_float(config.score_relative_tolerance),
        "max_unique_lambdas": int(config.max_unique_lambdas),
        "max_refinement_lambdas": int(config.max_refinement_lambdas),
        "max_solver_retries_per_lambda": int(
            config.max_solver_retries_per_lambda
        ),
        "max_bootstrap_anchor_lambdas": int(config.max_bootstrap_anchor_lambdas),
        "partition_event_mode": bool(config.partition_event_mode),
    }


def _config_from_state(value: object) -> OnlineLambdaConfig:
    state = _mapping(value, field="config")
    _require_keys(state, expected=_CONFIG_STATE_KEYS, field="config")
    return OnlineLambdaConfig(
        guide_n_clusters=_integer(
            state["guide_n_clusters"], field="config.guide_n_clusters", minimum=1
        ),
        num_mutations=_integer(
            state["num_mutations"], field="config.num_mutations", minimum=1
        ),
        kkt_tolerance=_decode_float(
            state["kkt_tolerance"], field="config.kkt_tolerance"
        ),
        lambda_min=_decode_float(state["lambda_min"], field="config.lambda_min"),
        lambda_max=_decode_float(state["lambda_max"], field="config.lambda_max"),
        transition_log10_width_tolerance=_decode_float(
            state["transition_log10_width_tolerance"],
            field="config.transition_log10_width_tolerance",
        ),
        score_relative_tolerance=_decode_float(
            state["score_relative_tolerance"],
            field="config.score_relative_tolerance",
        ),
        max_unique_lambdas=_integer(
            state["max_unique_lambdas"],
            field="config.max_unique_lambdas",
            minimum=1,
        ),
        max_refinement_lambdas=_integer(
            state["max_refinement_lambdas"],
            field="config.max_refinement_lambdas",
            minimum=0,
        ),
        max_solver_retries_per_lambda=_integer(
            state["max_solver_retries_per_lambda"],
            field="config.max_solver_retries_per_lambda",
            minimum=0,
        ),
        max_bootstrap_anchor_lambdas=_integer(
            state["max_bootstrap_anchor_lambdas"],
            field="config.max_bootstrap_anchor_lambdas",
            minimum=0,
        ),
        partition_event_mode=_boolean(
            state["partition_event_mode"], field="config.partition_event_mode"
        ),
    )


def _observation_to_state(
    observation: OnlineLambdaObservation,
) -> dict[str, object]:
    return {
        "lambda_value": _encode_float(observation.lambda_value),
        "n_clusters": int(observation.n_clusters),
        "partition_signature": str(observation.partition_signature),
        "partition_icl": _encode_float(observation.partition_icl),
        "kkt_residual": _encode_float(observation.kkt_residual),
        "raw_objective_certified": bool(observation.raw_objective_certified),
        "partition_certified": bool(observation.partition_certified),
        "selection_score_available": bool(observation.selection_score_available),
        "score_numerical_uncertainty": _encode_float(
            observation.score_numerical_uncertainty
        ),
        "degrees_of_freedom": int(observation.degrees_of_freedom),
    }


def _observation_from_state(value: object, *, field: str) -> OnlineLambdaObservation:
    state = _mapping(value, field=field)
    _require_keys(state, expected=_OBSERVATION_STATE_KEYS, field=field)
    return OnlineLambdaObservation(
        lambda_value=_decode_float(
            state["lambda_value"], field=f"{field}.lambda_value"
        ),
        n_clusters=_integer(
            state["n_clusters"], field=f"{field}.n_clusters", minimum=1
        ),
        partition_signature=_text(
            state["partition_signature"], field=f"{field}.partition_signature"
        ),
        partition_icl=_decode_float(
            state["partition_icl"], field=f"{field}.partition_icl"
        ),
        kkt_residual=_decode_float(
            state["kkt_residual"], field=f"{field}.kkt_residual"
        ),
        raw_objective_certified=_boolean(
            state["raw_objective_certified"],
            field=f"{field}.raw_objective_certified",
        ),
        partition_certified=_boolean(
            state["partition_certified"], field=f"{field}.partition_certified"
        ),
        selection_score_available=_boolean(
            state["selection_score_available"],
            field=f"{field}.selection_score_available",
        ),
        score_numerical_uncertainty=_decode_float(
            state["score_numerical_uncertainty"],
            field=f"{field}.score_numerical_uncertainty",
        ),
        degrees_of_freedom=_integer(
            state["degrees_of_freedom"],
            field=f"{field}.degrees_of_freedom",
            minimum=0,
        ),
    )


def _proposal_to_state(proposal: OnlineLambdaProposal) -> dict[str, object]:
    return {
        "lambda_value": _encode_float(proposal.lambda_value),
        "phase": str(proposal.phase),
        "reason": str(proposal.reason),
        "warm_start_lambda": _encode_optional_float(proposal.warm_start_lambda),
        "alternate_start_lambda": _encode_optional_float(
            proposal.alternate_start_lambda
        ),
        "bracket_left_lambda": _encode_optional_float(
            proposal.bracket_left_lambda
        ),
        "bracket_right_lambda": _encode_optional_float(
            proposal.bracket_right_lambda
        ),
        "retry_number": int(proposal.retry_number),
    }


def _proposal_from_state(value: object, *, field: str) -> OnlineLambdaProposal:
    state = _mapping(value, field=field)
    _require_keys(state, expected=_PROPOSAL_STATE_KEYS, field=field)
    return OnlineLambdaProposal(
        lambda_value=_decode_float(
            state["lambda_value"], field=f"{field}.lambda_value"
        ),
        phase=_text(state["phase"], field=f"{field}.phase", nonempty=True),
        reason=_text(state["reason"], field=f"{field}.reason", nonempty=True),
        warm_start_lambda=_decode_optional_float(
            state["warm_start_lambda"], field=f"{field}.warm_start_lambda"
        ),
        alternate_start_lambda=_decode_optional_float(
            state["alternate_start_lambda"],
            field=f"{field}.alternate_start_lambda",
        ),
        bracket_left_lambda=_decode_optional_float(
            state["bracket_left_lambda"], field=f"{field}.bracket_left_lambda"
        ),
        bracket_right_lambda=_decode_optional_float(
            state["bracket_right_lambda"], field=f"{field}.bracket_right_lambda"
        ),
        retry_number=_integer(
            state["retry_number"], field=f"{field}.retry_number", minimum=0
        ),
    )


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
        self._attempted_phase: dict[float, str] = {}
        self._proposal_history: list[OnlineLambdaProposal] = []
        self._solver_recovery_keys: set[float] = set()
        self._bootstrap_anchor_keys: set[float] = set()
        self._uncertified_exhausted_keys: set[float] = set()
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

    def state_dict(self) -> dict[str, object]:
        """Return a versioned, lossless, standard-JSON-safe controller state.

        Checkpoints are permitted only between complete proposal/observation
        transactions.  Persisting an outstanding proposal would leave the
        caller unable to prove whether its solver work had already run.
        """

        if self._pending is not None:
            raise RuntimeError(
                "OnlineLambdaController can be checkpointed only after the "
                "outstanding proposal has been observed."
            )
        certified = [
            {
                "key": _encode_float(key),
                "observation": _observation_to_state(observation),
            }
            for key, observation in sorted(self._certified.items())
        ]
        attempt_count = [
            {"key": _encode_float(key), "count": int(count)}
            for key, count in sorted(self._attempt_count.items())
        ]
        attempted_lambdas = [
            {
                "key": _encode_float(key),
                "lambda_value": _encode_float(value),
                "phase": str(self._attempted_phase[key]),
            }
            for key, value in sorted(self._attempted_lambda.items())
        ]
        return {
            "schema": ONLINE_LAMBDA_STATE_SCHEMA,
            "schema_version": ONLINE_LAMBDA_STATE_SCHEMA_VERSION,
            "pending": None,
            "config": _config_to_state(self.config),
            "initial_lambda": _encode_float(self.initial_lambda),
            "initial_reason": str(self.initial_reason),
            "certified": certified,
            "last_observation": (
                None
                if self._last_observation is None
                else _observation_to_state(self._last_observation)
            ),
            "retry_key": _encode_optional_float(self._retry_key),
            "attempt_count": attempt_count,
            "attempted_lambdas": attempted_lambdas,
            "proposal_history": [
                _proposal_to_state(proposal) for proposal in self._proposal_history
            ],
            "solver_recovery_keys": [
                _encode_float(key) for key in sorted(self._solver_recovery_keys)
            ],
            "bootstrap_anchor_keys": [
                _encode_float(key) for key in sorted(self._bootstrap_anchor_keys)
            ],
            "uncertified_exhausted_keys": [
                _encode_float(key)
                for key in sorted(self._uncertified_exhausted_keys)
            ],
            "stop_reason": self._stop_reason,
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping[str, object],
    ) -> "OnlineLambdaController":
        """Restore a state emitted by :meth:`state_dict`, failing closed."""

        state = _mapping(value, field="controller_state")
        _require_keys(
            state,
            expected=_CONTROLLER_STATE_KEYS,
            field="controller_state",
        )
        if state["schema"] != ONLINE_LAMBDA_STATE_SCHEMA:
            raise ValueError("Unsupported online-lambda controller state schema.")
        version = _integer(
            state["schema_version"],
            field="controller_state.schema_version",
            minimum=1,
        )
        if version != ONLINE_LAMBDA_STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported online-lambda controller state schema version."
            )
        if state["pending"] is not None:
            raise ValueError(
                "A controller checkpoint with an outstanding proposal is unsafe."
            )

        config = _config_from_state(state["config"])
        initial_lambda = _decode_float(
            state["initial_lambda"], field="controller_state.initial_lambda"
        )
        controller = cls(
            initial_lambda=initial_lambda,
            config=config,
            initial_reason=_text(
                state["initial_reason"],
                field="controller_state.initial_reason",
                nonempty=True,
            ),
        )
        if controller.initial_lambda != initial_lambda:
            raise ValueError("Stored initial lambda lies outside its configured bounds.")

        def keyed_records(raw: object, *, field: str) -> list[Mapping[str, object]]:
            records = _sequence(raw, field=field)
            return [
                _mapping(record, field=f"{field}[{index}]")
                for index, record in enumerate(records)
            ]

        certified: dict[float, OnlineLambdaObservation] = {}
        for index, record in enumerate(
            keyed_records(state["certified"], field="controller_state.certified")
        ):
            _require_keys(
                record,
                expected={"key", "observation"},
                field=f"controller_state.certified[{index}]",
            )
            key = _decode_float(
                record["key"], field=f"controller_state.certified[{index}].key"
            )
            if key in certified:
                raise ValueError("Controller state contains duplicate certified keys.")
            certified[key] = _observation_from_state(
                record["observation"],
                field=f"controller_state.certified[{index}].observation",
            )

        attempt_count: dict[float, int] = {}
        for index, record in enumerate(
            keyed_records(
                state["attempt_count"], field="controller_state.attempt_count"
            )
        ):
            _require_keys(
                record,
                expected={"key", "count"},
                field=f"controller_state.attempt_count[{index}]",
            )
            key = _decode_float(
                record["key"], field=f"controller_state.attempt_count[{index}].key"
            )
            if key in attempt_count:
                raise ValueError("Controller state contains duplicate attempt keys.")
            attempt_count[key] = _integer(
                record["count"],
                field=f"controller_state.attempt_count[{index}].count",
                minimum=1,
            )

        attempted_lambda: dict[float, float] = {}
        attempted_phase: dict[float, str] = {}
        for index, record in enumerate(
            keyed_records(
                state["attempted_lambdas"],
                field="controller_state.attempted_lambdas",
            )
        ):
            _require_keys(
                record,
                expected={"key", "lambda_value", "phase"},
                field=f"controller_state.attempted_lambdas[{index}]",
            )
            key = _decode_float(
                record["key"],
                field=f"controller_state.attempted_lambdas[{index}].key",
            )
            if key in attempted_lambda:
                raise ValueError("Controller state contains duplicate attempted keys.")
            attempted_lambda[key] = _decode_float(
                record["lambda_value"],
                field=(
                    f"controller_state.attempted_lambdas[{index}].lambda_value"
                ),
            )
            attempted_phase[key] = _text(
                record["phase"],
                field=f"controller_state.attempted_lambdas[{index}].phase",
                nonempty=True,
            )

        proposal_history = [
            _proposal_from_state(
                proposal,
                field=f"controller_state.proposal_history[{index}]",
            )
            for index, proposal in enumerate(
                _sequence(
                    state["proposal_history"],
                    field="controller_state.proposal_history",
                )
            )
        ]

        def key_set(raw: object, *, field: str) -> set[float]:
            result: set[float] = set()
            for index, encoded in enumerate(_sequence(raw, field=field)):
                key = _decode_float(encoded, field=f"{field}[{index}]")
                if key in result:
                    raise ValueError(f"{field} contains duplicate keys.")
                result.add(key)
            return result

        controller._certified = certified
        controller._last_observation = (
            None
            if state["last_observation"] is None
            else _observation_from_state(
                state["last_observation"],
                field="controller_state.last_observation",
            )
        )
        controller._retry_key = _decode_optional_float(
            state["retry_key"], field="controller_state.retry_key"
        )
        controller._attempt_count = attempt_count
        controller._attempted_lambda = attempted_lambda
        controller._attempted_phase = attempted_phase
        controller._proposal_history = proposal_history
        controller._solver_recovery_keys = key_set(
            state["solver_recovery_keys"],
            field="controller_state.solver_recovery_keys",
        )
        controller._bootstrap_anchor_keys = key_set(
            state["bootstrap_anchor_keys"],
            field="controller_state.bootstrap_anchor_keys",
        )
        controller._uncertified_exhausted_keys = key_set(
            state["uncertified_exhausted_keys"],
            field="controller_state.uncertified_exhausted_keys",
        )
        controller._stop_reason = _optional_text(
            state["stop_reason"], field="controller_state.stop_reason"
        )
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

        if not self._proposal_history:
            if (
                self._last_observation is not None
                or attempted_keys
                or self._certified
                or any(tracked_key_sets)
                or self._retry_key is not None
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
        if bool(self.config.partition_event_mode):
            minimum_upper = min(
                float(item.partition_icl)
                + max(float(item.score_numerical_uncertainty), 0.0)
                for item in finite
            )
            tied = [
                item
                for item in finite
                if float(item.partition_icl)
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
        best_score = min(float(item.partition_icl) for item in finite)
        score_tol = float(self.config.score_relative_tolerance) * (
            1.0 + abs(best_score)
        )
        tied = [
            item
            for item in finite
            if float(item.partition_icl) <= best_score + score_tol
        ]
        return min(
            tied,
            key=lambda item: (
                abs(log(float(item.lambda_value)) - log(self.initial_lambda)),
                abs(int(item.n_clusters) - int(self.config.guide_n_clusters)),
                float(item.lambda_value),
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
            and isfinite(float(observation.partition_icl))
        )

    def _record_proposal(self, proposal: OnlineLambdaProposal) -> OnlineLambdaProposal:
        key = _lambda_key(proposal.lambda_value)
        if key not in self._attempted_lambda:
            self._attempted_lambda[key] = float(proposal.lambda_value)
            self._attempted_phase[key] = str(proposal.phase)
        self._pending = proposal
        self._proposal_history.append(proposal)
        return proposal

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
            if bool(self.config.partition_event_mode):
                proposal = self._event_outward_proposal(
                    points,
                    direction=direction,
                    reason="refinement_budget_exhausted_continue_union_exploration",
                )
            else:
                proposal = self._outward_proposal(
                    points,
                    direction=direction,
                    reason="refinement_budget_exhausted_continue_path_exploration",
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

        if bool(self.config.partition_event_mode):
            return self._choose_from_partition_events(points)

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
                return self._outward_proposal(
                    points, direction=1, reason="observed_k_above_guide_k"
                )
            elif all(int(item.n_clusters) < guide_k for item in points):
                return self._outward_proposal(
                    points, direction=-1, reason="observed_k_below_guide_k"
                )

        return self._score_basin_proposal(points)

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
        allow_opposite: bool = True,
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
        if _lambda_key(candidate) in self._uncertified_exhausted_keys:
            if allow_opposite:
                return self._outward_proposal(
                    points,
                    direction=-int(direction),
                    reason=f"{reason}_opposite_of_uncertified_boundary",
                    allow_opposite=False,
                )
            self._stop_reason = "online_lambda_uncertified_boundaries_exhausted"
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
            if (
                abs(delta_x) > np.finfo(np.float64).eps
                and abs(delta_log_k) > np.finfo(np.float64).eps
            ):
                slope = delta_log_k / delta_x
                if slope < 0.0:
                    secant_x = (
                        x_frontier
                        + (log(float(guide_k)) - log(float(k_frontier))) / slope
                    )
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
        if direction > 0 and candidate <= float(frontier.lambda_value) * (
            1.0 + 8.0 * np.finfo(float).eps
        ):
            return None
        if direction < 0 and candidate >= float(frontier.lambda_value) * (
            1.0 - 8.0 * np.finfo(float).eps
        ):
            return None
        return float(candidate)

    def _score_basin_proposal(
        self,
        points: list[OnlineLambdaObservation],
    ) -> OnlineLambdaProposal | None:
        best = self.best_observation
        if best is None:
            return self._unresolved_partition_proposal(points)
        best_index = next(
            idx
            for idx, item in enumerate(points)
            if _lambda_key(item.lambda_value) == _lambda_key(best.lambda_value)
        )
        run_left = best_index
        while (
            run_left > 0
            and points[run_left - 1].partition_signature == best.partition_signature
        ):
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
                lower_span = max(
                    log(self.initial_lambda) - log(points[0].lambda_value), 0.0
                )
                upper_span = max(
                    log(points[-1].lambda_value) - log(self.initial_lambda), 0.0
                )
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
            tuple[
                OnlineLambdaObservation,
                OnlineLambdaObservation,
                OnlineLambdaObservation,
            ]
        ] = []
        if left_guard is not None:
            boundary_intervals.append((left_guard, points[run_left], left_guard))
        if right_guard is not None:
            boundary_intervals.append((points[run_right], right_guard, right_guard))
        unresolved = [
            item
            for item in boundary_intervals
            if not self._interval_resolved(item[0], item[1])
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

        if any(
            not isfinite(float(guard.partition_icl))
            for _, _, guard in boundary_intervals
        ):
            self._stop_reason = "online_lambda_raw_certified_partition_unresolved"
            return None
        self._stop_reason = "online_lambda_score_basin_resolved"
        return None

    def _unresolved_partition_proposal(
        self,
        points: list[OnlineLambdaObservation],
    ) -> OnlineLambdaProposal | None:
        """Search adjacent lambdas without retrying an already certified raw fit."""

        if not points:
            self._stop_reason = "online_lambda_raw_certified_partition_unresolved"
            return None
        left = points[0]
        right = points[-1]
        can_expand_lower = bool(
            float(left.lambda_value) > float(self.config.lambda_min)
            and int(left.n_clusters) < int(self.config.num_mutations)
        )
        can_expand_upper = bool(
            float(right.lambda_value) < float(self.config.lambda_max)
            and int(right.n_clusters) > 1
        )
        if not can_expand_lower and not can_expand_upper:
            self._stop_reason = "online_lambda_raw_certified_partition_unresolved"
            return None
        if can_expand_lower and can_expand_upper:
            lower_span = max(log(self.initial_lambda) - log(left.lambda_value), 0.0)
            upper_span = max(log(right.lambda_value) - log(self.initial_lambda), 0.0)
            direction = -1 if lower_span < upper_span else 1
        else:
            direction = -1 if can_expand_lower else 1
        proposal = self._outward_proposal(
            points,
            direction=direction,
            reason="raw_certified_partition_unresolved_search_neighbor",
        )
        if proposal is None and self._stop_reason is not None:
            self._stop_reason = "online_lambda_raw_certified_partition_unresolved"
        return proposal


def guard_score(observation: OnlineLambdaObservation) -> float:
    """Sort non-finite guard scores after finite scores."""

    value = float(observation.partition_icl)
    return value if isfinite(value) else float("inf")


__all__ = [
    "ONLINE_LAMBDA_STATE_SCHEMA",
    "ONLINE_LAMBDA_STATE_SCHEMA_VERSION",
    "OnlineLambdaConfig",
    "OnlineLambdaController",
    "OnlineLambdaObservation",
    "OnlineLambdaProposal",
]
