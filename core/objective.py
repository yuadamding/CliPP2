"""Canonical source representation of CliPP2's observed-count likelihood.

Both the historical major/minor model and an explicit occupancy-path model are
represented as mixtures of clipped piecewise-affine binomial emissions.
Runtime tensors are always rebuilt from immutable float64 source arrays; a
lower-precision runtime is never the source of a higher-precision view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ..io.data import TumorData
    from .fusion.types import TorchRuntime


_MODEL_FINGERPRINT_SCHEMA = "clipp2.observed-model.v1"
_LIKELIHOOD_FINGERPRINT_SCHEMA = "clipp2.observed-likelihood.v1"
_REPORTING_MODEL_FINGERPRINT_SCHEMA = "clipp2.reporting-model.v2"
_BOX_FINGERPRINT_SCHEMA = "clipp2.objective-box.v1"
_BASE_OBJECTIVE_KEY_SCHEMA = "clipp2.base-objective-key.v1"
_LAMBDA_OBJECTIVE_KEY_SCHEMA = "clipp2.lambda-objective-key.v1"


def _readonly_array(value: object, *, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _hash_text(digest: object, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_array(digest: object, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    _hash_text(digest, name)
    _hash_text(digest, str(array.dtype))
    digest.update(len(array.shape).to_bytes(8, "little"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "little", signed=True))
    digest.update(array.tobytes())


def _model_fingerprint(model: "ObservedModel") -> str:
    """Hash the canonical numerical model, excluding reporting metadata."""

    digest = hashlib.sha256()
    _hash_text(digest, _MODEL_FINGERPRINT_SCHEMA)
    for name in (
        "alt",
        "nonalt",
        "observed",
        "lower",
        "upper",
        "first_scale",
        "second_scale",
        "switch",
        "log_prior",
        "valid",
    ):
        _hash_array(digest, name, getattr(model, name))
    return digest.hexdigest()


def _likelihood_fingerprint(model: "ObservedModel") -> str:
    """Hash only the observed likelihood, excluding its feasible box."""

    digest = hashlib.sha256()
    _hash_text(digest, _LIKELIHOOD_FINGERPRINT_SCHEMA)
    for name in (
        "alt",
        "nonalt",
        "observed",
        "first_scale",
        "second_scale",
        "switch",
        "log_prior",
        "valid",
    ):
        _hash_array(digest, name, getattr(model, name))
    return digest.hexdigest()


def _reporting_model_fingerprint(model: "ObservedModel") -> str:
    """Hash every model value that can change posterior reporting semantics."""

    digest = hashlib.sha256()
    _hash_text(digest, _REPORTING_MODEL_FINGERPRINT_SCHEMA)
    _hash_text(digest, model.fingerprint)
    _hash_array(digest, "first_copy", model.first_copy)
    _hash_array(digest, "second_copy", model.second_copy)
    if model.major_indicator is None:
        _hash_text(digest, "major_indicator:none")
    else:
        _hash_text(digest, "major_indicator:present")
        _hash_array(digest, "major_indicator", model.major_indicator)
    return digest.hexdigest()


def _box_fingerprint(lower: np.ndarray, upper: np.ndarray) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, _BOX_FINGERPRINT_SCHEMA)
    _hash_array(digest, "lower", np.asarray(lower, dtype=np.float64))
    _hash_array(digest, "upper", np.asarray(upper, dtype=np.float64))
    return digest.hexdigest()


def observed_box_fingerprint(model: "ObservedModel") -> str:
    """Hash the canonical feasible box of one compiled observed model."""

    return _box_fingerprint(model.lower, model.upper)


def _key_fingerprint(schema: str, *values: str) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, schema)
    for value in values:
        _hash_text(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BaseObjectiveKey:
    """Source identity of the likelihood, graph, box, and clipping rule."""

    likelihood_hash: str
    graph_hash: str
    box_hash: str
    eps_hex: str

    @property
    def fingerprint(self) -> str:
        return _key_fingerprint(
            _BASE_OBJECTIVE_KEY_SCHEMA,
            self.likelihood_hash,
            self.graph_hash,
            self.box_hash,
            self.eps_hex,
        )


@dataclass(frozen=True, slots=True)
class LambdaObjectiveKey:
    """A base objective bound to one nonnegative fusion penalty."""

    base: BaseObjectiveKey
    lambda_hex: str

    @property
    def fingerprint(self) -> str:
        return _key_fingerprint(
            _LAMBDA_OBJECTIVE_KEY_SCHEMA,
            self.base.fingerprint,
            self.lambda_hex,
        )


@dataclass(frozen=True, slots=True)
class ObservedModel:
    """Immutable float64 source model for observed mutation counts.

    The path arrays have shape ``(mutation, region, path)``.  For CCF ``phi``,
    a path's scaled mutant-copy mass is

    ``first_scale * min(phi, switch) + second_scale * max(phi-switch, 0)``.

    ``major_indicator`` is reporting metadata and is deliberately excluded from
    the numerical fingerprint.  ``model_id`` likewise names the source family
    without changing the represented likelihood.
    """

    alt: np.ndarray
    nonalt: np.ndarray
    observed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    # Exact unscaled coefficients are reporting metadata. Numerical kernels
    # consume the scaled arrays below, whose v1 fingerprints stay unchanged.
    first_copy: np.ndarray
    second_copy: np.ndarray
    first_scale: np.ndarray
    second_scale: np.ndarray
    switch: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    major_indicator: np.ndarray | None
    model_id: str
    fingerprint: str = field(init=False)
    likelihood_fingerprint: str = field(init=False)
    reporting_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        observation_arrays = {
            name: np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            for name in ("alt", "nonalt", "lower", "upper")
        }
        shape = observation_arrays["alt"].shape
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError(
                "ObservedModel observations must have nonempty shape (M, S)."
            )
        for name, value in observation_arrays.items():
            if value.shape != shape:
                raise ValueError(f"ObservedModel.{name} must have shape {shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"ObservedModel.{name} must contain only finite values."
                )
        if np.any(observation_arrays["alt"] < 0.0) or np.any(
            observation_arrays["nonalt"] < 0.0
        ):
            raise ValueError("ObservedModel counts must be nonnegative.")
        if np.any(observation_arrays["lower"] < 0.0) or np.any(
            observation_arrays["upper"] > 1.0
        ):
            raise ValueError("ObservedModel CCF bounds must lie in [0, 1].")
        if np.any(observation_arrays["lower"] > observation_arrays["upper"]):
            raise ValueError("ObservedModel lower bounds cannot exceed upper bounds.")

        observed = np.array(self.observed, dtype=bool, copy=True, order="C")
        if observed.shape != shape:
            raise ValueError(f"ObservedModel.observed must have shape {shape}.")

        path_arrays = {
            name: np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            for name in (
                "first_copy",
                "second_copy",
                "first_scale",
                "second_scale",
                "switch",
                "log_prior",
            )
        }
        path_shape = path_arrays["first_scale"].shape
        if len(path_shape) != 3 or path_shape[:2] != shape or not path_shape[2]:
            raise ValueError(
                "ObservedModel path arrays must have nonempty shape (M, S, K)."
            )
        for name, value in path_arrays.items():
            if value.shape != path_shape:
                raise ValueError(f"ObservedModel.{name} must have shape {path_shape}.")
        valid = np.array(self.valid, dtype=bool, copy=True, order="C")
        if valid.shape != path_shape:
            raise ValueError(f"ObservedModel.valid must have shape {path_shape}.")
        if not np.all(np.any(valid, axis=-1)):
            raise ValueError("Every mutation-region entry must have a valid path.")
        for name in ("first_copy", "second_copy", "first_scale", "second_scale"):
            values = path_arrays[name][valid]
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"Valid {name} values must be finite and nonnegative.")
        switches = path_arrays["switch"][valid]
        if np.any(~np.isfinite(switches)) or np.any(
            (switches < 0.0) | (switches > 1.0)
        ):
            raise ValueError("Valid switch values must be finite and lie in [0, 1].")
        if np.any(~np.isfinite(path_arrays["log_prior"][valid])):
            raise ValueError("Valid log_prior values must be finite.")

        for name in (
            "first_copy",
            "second_copy",
            "first_scale",
            "second_scale",
            "switch",
        ):
            path_arrays[name] = np.where(valid, path_arrays[name], 0.0)
        path_arrays["log_prior"] = np.where(valid, path_arrays["log_prior"], -np.inf)
        maximum = np.max(path_arrays["log_prior"], axis=-1, keepdims=True)
        normalizer = np.squeeze(
            maximum
            + np.log(
                np.sum(
                    np.where(
                        valid,
                        np.exp(path_arrays["log_prior"] - maximum),
                        0.0,
                    ),
                    axis=-1,
                    keepdims=True,
                )
            ),
            axis=-1,
        )
        if not np.allclose(normalizer, 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("ObservedModel.log_prior must normalize over valid paths.")

        major_indicator = self.major_indicator
        if major_indicator is not None:
            major_indicator = np.array(
                major_indicator, dtype=bool, copy=True, order="C"
            )
            if major_indicator.shape != path_shape:
                raise ValueError(
                    f"ObservedModel.major_indicator must have shape {path_shape}."
                )
            major_indicator &= valid

        model_id = str(self.model_id).strip()
        if not model_id:
            raise ValueError("ObservedModel.model_id must be nonempty.")
        for name, value in observation_arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype=np.float64))
        object.__setattr__(self, "observed", _readonly_array(observed, dtype=bool))
        for name, value in path_arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype=np.float64))
        object.__setattr__(self, "valid", _readonly_array(valid, dtype=bool))
        object.__setattr__(
            self,
            "major_indicator",
            (
                None
                if major_indicator is None
                else _readonly_array(major_indicator, dtype=bool)
            ),
        )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "fingerprint", _model_fingerprint(self))
        object.__setattr__(
            self,
            "likelihood_fingerprint",
            _likelihood_fingerprint(self),
        )
        object.__setattr__(
            self,
            "reporting_fingerprint",
            _reporting_model_fingerprint(self),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.alt.shape)

    @property
    def path_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.first_scale.shape)

    @property
    def has_fixed_linear_emission(self) -> bool:
        """Whether every unit has one shared fixed linear emission.

        Normalized aliases with an identical slope are numerically one
        emission.  Any competing slope or genuine internal switch requires
        the generic potentially multimodal solver path.
        """

        first_valid_index = np.argmax(self.valid, axis=-1, keepdims=True)
        reference = np.take_along_axis(
            self.first_scale,
            first_valid_index,
            axis=-1,
        )
        linear = (~self.valid) | (self.first_scale == self.second_scale)
        shared = (~self.valid) | (self.first_scale == reference)
        return bool(np.all(linear & shared))

    @property
    def has_internal_switches(self) -> bool:
        """Whether a valid path changes slope inside its feasible CCF box."""

        lower = self.lower[..., None]
        upper = self.upper[..., None]
        return bool(
            np.any(
                self.valid
                & (self.switch > lower)
                & (self.switch < upper)
                & (self.first_scale != self.second_scale)
            )
        )

    @property
    def is_potentially_multimodal(self) -> bool:
        """Whether the compiled likelihood needs the generic solver route."""

        return not self.has_fixed_linear_emission

    @property
    def uses_binary_linear_mixture_fast_path(self) -> bool:
        """Whether numeric emission arrays have the two-path linear shape.

        Numerical routing deliberately ignores ``major_indicator`` and every
        other reporting-only field. Thus models with equal numerical hashes
        cannot select different solver routes.
        """

        return self.binary_linear_mixture_prior is not None

    @property
    def binary_linear_mixture_prior(self) -> float | None:
        """Return the shared second-path prior admitted by the fast formulas.

        The specialized scalar-root implementation assumes two ordered linear
        paths and one shared prior. Every condition below is encoded in arrays
        covered by the numerical likelihood fingerprint.
        """

        if self.path_shape[-1] != 2 or not np.all(self.valid[..., 0]):
            return None
        if not np.all((~self.valid) | (self.first_scale == self.second_scale)):
            return None
        ambiguous = self.valid[..., 1]
        if not np.any(ambiguous):
            return 0.5
        if np.any(
            self.first_scale[..., 0][ambiguous]
            > self.first_scale[..., 1][ambiguous]
        ):
            return None
        prior = np.exp(self.log_prior[..., 1][ambiguous])
        if (
            np.any(~np.isfinite(prior))
            or np.any(prior <= 0.0)
            or np.any(prior >= 1.0)
            or not np.all(prior == prior[0])
        ):
            return None
        return float(prior[0])

    @property
    def requires_generic_path_solver(self) -> bool:
        """Whether structural emission shape requires generic path handling."""

        return bool(
            not self.has_fixed_linear_emission
            and not self.uses_binary_linear_mixture_fast_path
        )


@dataclass(frozen=True, slots=True)
class TorchObservedModel:
    """Runtime view rebuilt directly from an :class:`ObservedModel`."""

    alt: torch.Tensor
    nonalt: torch.Tensor
    observed: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    first_scale: torch.Tensor
    second_scale: torch.Tensor
    switch: torch.Tensor
    log_prior: torch.Tensor
    valid: torch.Tensor
    model_id: str
    source_fingerprint: str

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.alt.shape)

    @property
    def path_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.first_scale.shape)

    @property
    def total(self) -> torch.Tensor:
        return self.alt + self.nonalt


@dataclass(frozen=True, slots=True)
class ObservedTerms:
    loss: np.ndarray
    gradient: np.ndarray
    hessian_upper: np.ndarray
    posterior: np.ndarray


@dataclass(frozen=True, slots=True)
class TorchObservedTerms:
    """Numerical likelihood terms retained on the Torch runtime device."""

    loss: torch.Tensor
    gradient: torch.Tensor
    hessian_upper: torch.Tensor
    posterior: torch.Tensor


@dataclass(frozen=True, slots=True)
class _NumpyPathKernel:
    mass: np.ndarray
    probability: np.ndarray
    slope: np.ndarray


@dataclass(frozen=True, slots=True)
class _TorchPathKernel:
    mass: torch.Tensor
    probability: torch.Tensor
    slope: torch.Tensor


def emission_probability_scale(data: "TumorData") -> np.ndarray:
    """Compile the sample-specific probability scale from input CN facts."""

    purity = np.asarray(data.purity, dtype=np.float64)
    denominator = (1.0 - purity) * np.asarray(
        data.normal_cn, dtype=np.float64
    ) + purity * np.asarray(data.tumor_total_cn, dtype=np.float64)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("Every tumor coordinate needs a positive CN denominator.")
    return purity / denominator


def piecewise_affine_probability_upper(
    *,
    first_scale: np.ndarray,
    second_scale: np.ndarray,
    switch: np.ndarray,
    valid: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Return the largest CCF whose every valid path probability is in-box."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    first = np.asarray(first_scale, dtype=np.float64)
    second = np.asarray(second_scale, dtype=np.float64)
    breakpoint = np.asarray(switch, dtype=np.float64)
    is_valid = np.asarray(valid, dtype=bool)
    if (
        first.ndim != 3
        or second.shape != first.shape
        or breakpoint.shape != first.shape
        or is_valid.shape != first.shape
    ):
        raise ValueError(
            "Scaled piecewise-affine paths must share shape (mutation, region, path)."
        )

    target = 1.0 - epsilon
    mass_at_switch = first * breakpoint
    mass_at_one = mass_at_switch + second * (1.0 - breakpoint)
    exceeds = is_valid & (mass_at_one > target)
    first_segment = exceeds & (target <= mass_at_switch)
    second_segment = exceeds & ~first_segment

    crossing = np.ones(first.shape, dtype=np.float64)
    crossing[first_segment] = target / first[first_segment]
    crossing[second_segment] = breakpoint[second_segment] + (
        target - mass_at_switch[second_segment]
    ) / second[second_segment]

    # Round toward the feasible side only when the closed-form crossing
    # evaluates above the target. Canonical linear paths retain their exact
    # historical arithmetic and therefore their model hashes.
    at_crossing = first * np.minimum(crossing, breakpoint) + second * np.maximum(
        crossing - breakpoint, 0.0
    )
    round_down = exceeds & (at_crossing > target)
    crossing[round_down] = np.nextafter(crossing[round_down], 0.0)

    lower_mass = first * np.minimum(epsilon, breakpoint) + second * np.maximum(
        epsilon - breakpoint, 0.0
    )
    if np.any(is_valid & (lower_mass > target)):
        raise ValueError(
            "No feasible CCF probability box exists at the requested epsilon."
        )
    return np.clip(
        np.min(np.where(is_valid, crossing, 1.0), axis=-1),
        epsilon,
        1.0,
    )


@dataclass(frozen=True, slots=True)
class _CompiledEmissionPaths:
    first_copy: np.ndarray
    second_copy: np.ndarray
    first_scale: np.ndarray
    second_scale: np.ndarray
    switch: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    major_indicator: np.ndarray | None
    model_id: str


def _compile_emission_paths(
    data: "TumorData", major_prior: float
) -> _CompiledEmissionPaths:
    spec = data.emission_paths
    shape = tuple(int(value) for value in np.asarray(data.alt_counts).shape)
    spec.validate_observation_shape(shape)
    scale = emission_probability_scale(data)[..., None]
    return _CompiledEmissionPaths(
        first_copy=np.asarray(spec.first_copy, dtype=np.float64),
        second_copy=np.asarray(spec.second_copy, dtype=np.float64),
        first_scale=scale * np.asarray(spec.first_copy, dtype=np.float64),
        second_scale=scale * np.asarray(spec.second_copy, dtype=np.float64),
        switch=np.asarray(spec.switch_fraction, dtype=np.float64),
        log_prior=np.asarray(spec.log_prior_for(major_prior), dtype=np.float64),
        valid=np.asarray(spec.valid, dtype=bool),
        major_indicator=(
            None
            if spec.major_indicator is None
            else np.asarray(spec.major_indicator, dtype=bool)
        ),
        model_id=spec.model_id,
    )


def compile_observed_model(
    data: "TumorData",
    *,
    major_prior: float,
    eps: float,
) -> ObservedModel:
    """Compile normalized emission paths into one immutable float64 model."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    if alt.shape != total.shape:
        raise ValueError("TumorData alt_counts and total_counts must have one shape.")
    # Count availability is a reporting/data fact.  Only the explicit
    # inclusion mask defines which coordinates enter this objective.
    observed = np.asarray(data.objective_inclusion_mask(), dtype=bool)
    compiled = _compile_emission_paths(data, major_prior)
    paths = data.emission_paths
    if paths.constrain_probability_box:
        upper = piecewise_affine_probability_upper(
            first_scale=compiled.first_scale,
            second_scale=compiled.second_scale,
            switch=compiled.switch,
            valid=compiled.valid,
            eps=epsilon,
        )
    else:
        upper = np.ones(alt.shape, dtype=np.float64)
    return ObservedModel(
        alt=alt,
        nonalt=total - alt,
        observed=observed,
        lower=np.full(alt.shape, epsilon, dtype=np.float64),
        upper=upper,
        first_copy=compiled.first_copy,
        second_copy=compiled.second_copy,
        first_scale=compiled.first_scale,
        second_scale=compiled.second_scale,
        switch=compiled.switch,
        log_prior=compiled.log_prior,
        valid=compiled.valid,
        major_indicator=compiled.major_indicator,
        model_id=compiled.model_id,
    )


def default_phi_initialization(model: ObservedModel, *, eps: float) -> np.ndarray:
    """Derive a deterministic solver hint from one compiled model.

    This preserves the established smoothed-count major/low hint for the
    shared binary linear mixture. General path models start at the center of
    their compiled feasible box; scalar well construction then performs the
    exact likelihood minimization.
    """

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must lie strictly in (0, 0.5).")
    lower = np.asarray(model.lower, dtype=np.float64)
    upper = np.asarray(model.upper, dtype=np.float64)
    if not model.uses_binary_linear_mixture_fast_path:
        return np.clip(np.full(model.shape, 0.5, dtype=np.float64), lower, upper)

    total = np.asarray(model.alt + model.nonalt, dtype=np.float64)
    smoothed_vaf = (np.asarray(model.alt, dtype=np.float64) + 0.5) / (total + 1.0)
    low_scale = np.asarray(model.first_scale[..., 0], dtype=np.float64)
    major_scale = np.asarray(model.first_scale[..., 1], dtype=np.float64)
    low = np.clip(
        np.divide(
            smoothed_vaf,
            np.clip(low_scale, epsilon, None),
            out=np.zeros_like(smoothed_vaf),
            where=low_scale > 0.0,
        ),
        lower,
        upper,
    )
    major = np.clip(
        np.divide(
            smoothed_vaf,
            np.clip(major_scale, epsilon, None),
            out=np.zeros_like(smoothed_vaf),
            where=major_scale > 0.0,
        ),
        lower,
        upper,
    )
    low_probability = np.clip(low_scale * low, epsilon, 1.0 - epsilon)
    major_probability = np.clip(major_scale * major, epsilon, 1.0 - epsilon)
    nonalt = total - model.alt
    low_loglik = model.alt * np.log(low_probability) + nonalt * np.log1p(
        -low_probability
    )
    major_loglik = model.alt * np.log(major_probability) + nonalt * np.log1p(
        -major_probability
    )
    result = np.where(major_loglik >= low_loglik, major, low)
    return np.clip(result, lower, upper).astype(np.float64, copy=False)


def model_to_torch(
    model: ObservedModel,
    runtime: "TorchRuntime",
) -> TorchObservedModel:
    """Build a runtime view from immutable source arrays, never another view."""

    dtype = runtime.dtype
    device = runtime.device
    if dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError("Observed-model runtime dtype must be floating point.")

    def numeric(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)

    def boolean(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.array(value, copy=True), dtype=torch.bool, device=device
        )

    return TorchObservedModel(
        alt=numeric(model.alt),
        nonalt=numeric(model.nonalt),
        observed=boolean(model.observed),
        lower=numeric(model.lower),
        upper=numeric(model.upper),
        first_scale=numeric(model.first_scale),
        second_scale=numeric(model.second_scale),
        switch=numeric(model.switch),
        log_prior=numeric(model.log_prior),
        valid=boolean(model.valid),
        model_id=model.model_id,
        source_fingerprint=model.fingerprint,
    )


def make_base_objective_key(
    model: ObservedModel,
    *,
    graph_hash: str,
    eps: float,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> BaseObjectiveKey:
    """Construct a dtype-invariant base-objective identity from host sources."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    graph_fingerprint = str(graph_hash).strip()
    if not graph_fingerprint:
        raise ValueError("graph_hash must be nonempty.")
    box_lower = model.lower if lower is None else np.asarray(lower, dtype=np.float64)
    box_upper = model.upper if upper is None else np.asarray(upper, dtype=np.float64)
    if box_lower.shape != model.shape or box_upper.shape != model.shape:
        raise ValueError(f"Objective bounds must have shape {model.shape}.")
    if (
        np.any(~np.isfinite(box_lower))
        or np.any(~np.isfinite(box_upper))
        or np.any(box_lower > box_upper)
    ):
        raise ValueError("Objective bounds must be finite with lower <= upper.")
    return BaseObjectiveKey(
        likelihood_hash=model.likelihood_fingerprint,
        graph_hash=graph_fingerprint,
        box_hash=_box_fingerprint(box_lower, box_upper),
        eps_hex=epsilon.hex(),
    )


def make_lambda_objective_key(
    base: BaseObjectiveKey,
    *,
    lambda_value: float,
) -> LambdaObjectiveKey:
    """Bind one base objective to an exact hexadecimal lambda identity."""

    value = float(lambda_value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("lambda_value must be finite and nonnegative.")
    return LambdaObjectiveKey(base=base, lambda_hex=value.hex())


def _validated_epsilon(eps: float) -> float:
    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    return epsilon


def _path_kernel_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
) -> _NumpyPathKernel:
    epsilon = _validated_epsilon(eps)
    phi_array = np.asarray(phi, dtype=np.float64)
    if phi_array.shape != model.shape or not np.all(np.isfinite(phi_array)):
        raise ValueError(f"phi must be a finite array with shape {model.shape}.")
    expanded_phi = phi_array[..., None]
    mass = model.first_scale * np.minimum(expanded_phi, model.switch)
    mass += model.second_scale * np.maximum(expanded_phi - model.switch, 0.0)
    probability = np.clip(mass, epsilon, 1.0 - epsilon)
    segment_slope = np.where(
        expanded_phi <= model.switch,
        model.first_scale,
        model.second_scale,
    )
    slope = np.where(
        (mass > epsilon) & (mass < 1.0 - epsilon),
        segment_slope,
        0.0,
    )
    return _NumpyPathKernel(mass=mass, probability=probability, slope=slope)


def _path_kernel_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> _TorchPathKernel:
    """Evaluate every canonical path over optional trailing grid dimensions.

    ``phi`` has shape ``(M, S, *grid)``. Runtime model arrays are reshaped,
    not copied, so the kernel stays resident on its original device while a
    caller evaluates any number of candidate CCFs per mutation-region.
    """

    epsilon = _validated_epsilon(eps)
    if phi.ndim < 2 or tuple(phi.shape[:2]) != model.shape:
        raise ValueError(
            "phi must start with the observed-model shape "
            f"{model.shape}, not {tuple(phi.shape)}."
        )
    if phi.dtype != model.alt.dtype or phi.device != model.alt.device:
        raise ValueError("phi must use the observed model's runtime dtype and device.")
    grid_ndim = phi.ndim - 2
    path_shape = (*model.shape, *((1,) * grid_ndim), model.path_shape[-1])

    def path_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(path_shape)

    expanded_phi = phi.unsqueeze(-1)
    first = path_view(model.first_scale)
    second = path_view(model.second_scale)
    switch = path_view(model.switch)
    mass = first * torch.minimum(expanded_phi, switch)
    mass = mass + second * torch.clamp(expanded_phi - switch, min=0.0)
    probability = torch.clamp(mass, min=epsilon, max=1.0 - epsilon)
    segment_slope = torch.where(expanded_phi <= switch, first, second)
    slope = torch.where(
        (mass > epsilon) & (mass < 1.0 - epsilon),
        segment_slope,
        torch.zeros_like(segment_slope),
    )
    return _TorchPathKernel(mass=mass, probability=probability, slope=slope)


def observed_terms_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
) -> ObservedTerms:
    """Evaluate loss, left-gradient, curvature majorant, and path posterior."""

    kernel = _path_kernel_numpy(model, phi, eps=eps)
    probability = kernel.probability
    slope = kernel.slope
    joint = np.where(
        model.valid,
        model.alt[..., None] * np.log(probability)
        + model.nonalt[..., None] * np.log1p(-probability)
        + model.log_prior,
        -np.inf,
    )
    maximum = np.max(joint, axis=-1, keepdims=True)
    unnormalized = np.where(model.valid, np.exp(joint - maximum), 0.0)
    denominator = np.sum(unnormalized, axis=-1, keepdims=True)
    posterior = unnormalized / denominator
    log_normalizer = np.squeeze(maximum + np.log(denominator), axis=-1)
    state_gradient = slope * (
        model.alt[..., None] / probability
        - model.nonalt[..., None] / (1.0 - probability)
    )
    state_curvature = np.square(slope) * (
        model.alt[..., None] / np.square(probability)
        + model.nonalt[..., None] / np.square(1.0 - probability)
    )
    loss = -log_normalizer
    gradient = -np.sum(posterior * state_gradient, axis=-1)
    hessian_upper = np.sum(posterior * state_curvature, axis=-1)
    prior = np.where(model.valid, np.exp(model.log_prior), 0.0)
    posterior = np.where(model.observed[..., None], posterior, prior)
    loss = np.where(model.observed, loss, 0.0)
    gradient = np.where(model.observed, gradient, 0.0)
    hessian_upper = np.where(model.observed, np.maximum(hessian_upper, 1e-8), 0.0)
    return ObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
    )


def observed_terms_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> TorchObservedTerms:
    """Torch counterpart of :func:`observed_terms_numpy`."""

    if tuple(phi.shape) != tuple(model.alt.shape):
        raise ValueError(f"phi must have shape {tuple(model.alt.shape)}.")
    kernel = _path_kernel_torch(model, phi, eps=eps)
    probability = kernel.probability
    slope = kernel.slope
    joint = (
        model.alt.unsqueeze(-1) * torch.log(probability)
        + model.nonalt.unsqueeze(-1) * torch.log1p(-probability)
        + model.log_prior
    ).masked_fill(~model.valid, -torch.inf)
    log_normalizer = torch.logsumexp(joint, dim=-1)
    posterior = torch.softmax(joint, dim=-1)
    state_gradient = slope * (
        model.alt.unsqueeze(-1) / probability
        - model.nonalt.unsqueeze(-1) / (1.0 - probability)
    )
    state_curvature = torch.square(slope) * (
        model.alt.unsqueeze(-1) / torch.square(probability)
        + model.nonalt.unsqueeze(-1) / torch.square(1.0 - probability)
    )
    loss = -log_normalizer
    gradient = -torch.sum(posterior * state_gradient, dim=-1)
    hessian_upper = torch.sum(posterior * state_curvature, dim=-1)
    prior = torch.exp(model.log_prior).masked_fill(~model.valid, 0.0)
    posterior = torch.where(model.observed.unsqueeze(-1), posterior, prior)
    loss = torch.where(model.observed, loss, torch.zeros_like(loss))
    gradient = torch.where(model.observed, gradient, torch.zeros_like(gradient))
    hessian_upper = torch.where(
        model.observed,
        torch.clamp(hessian_upper, min=1e-8),
        torch.zeros_like(hessian_upper),
    )
    return TorchObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
    )


def observed_loss_grid_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
    respect_observed: bool = True,
) -> torch.Tensor:
    """Evaluate the canonical observed loss over trailing candidate grids.

    ``phi`` must have shape ``(M, S, *grid)`` and the result has the same
    shape. This is the sole batched likelihood used by pilot and partition
    start generation.
    """

    kernel = _path_kernel_torch(model, phi, eps=eps)
    grid_ndim = phi.ndim - 2
    observation_shape = (*model.shape, *((1,) * grid_ndim))
    path_shape = (*observation_shape, model.path_shape[-1])

    def observation_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(observation_shape)

    def path_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(path_shape)

    joint = (
        observation_view(model.alt).unsqueeze(-1) * torch.log(kernel.probability)
        + observation_view(model.nonalt).unsqueeze(-1)
        * torch.log1p(-kernel.probability)
        + path_view(model.log_prior)
    ).masked_fill(~path_view(model.valid), -torch.inf)
    loss = -torch.logsumexp(joint, dim=-1)
    if not bool(respect_observed):
        return loss
    return torch.where(
        observation_view(model.observed),
        loss,
        torch.zeros_like(loss),
    )


def observed_em_terms_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    responsibilities: torch.Tensor,
    eps: float,
) -> TorchObservedTerms:
    """Evaluate one categorical EM surrogate for any observed model."""

    if tuple(phi.shape) != model.shape:
        raise ValueError(f"phi must have shape {model.shape}.")
    if tuple(responsibilities.shape) != model.path_shape:
        raise ValueError(
            f"responsibilities must have shape {model.path_shape}, "
            f"not {tuple(responsibilities.shape)}."
        )
    if responsibilities.dtype != model.alt.dtype or (
        responsibilities.device != model.alt.device
    ):
        raise ValueError(
            "responsibilities must use the observed model's dtype and device."
        )
    if not bool(torch.all(torch.isfinite(responsibilities)).item()) or bool(
        torch.any(responsibilities < 0.0).item()
    ):
        raise ValueError("responsibilities must be finite and nonnegative.")

    weights = responsibilities.masked_fill(~model.valid, 0.0)
    normalizer = torch.sum(weights, dim=-1, keepdim=True)
    if bool(torch.any(normalizer <= 0.0).item()):
        raise ValueError("responsibilities must assign mass to a valid path.")
    weights = weights / normalizer
    kernel = _path_kernel_torch(model, phi, eps=eps)
    log_kernel = model.alt.unsqueeze(-1) * torch.log(
        kernel.probability
    ) + model.nonalt.unsqueeze(-1) * torch.log1p(-kernel.probability)
    complete_loss = torch.where(
        model.valid,
        -(log_kernel + model.log_prior),
        torch.zeros_like(log_kernel),
    )
    entropy = torch.where(
        weights > 0.0,
        weights * torch.log(torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)),
        torch.zeros_like(weights),
    )
    loss = torch.sum(weights * complete_loss + entropy, dim=-1)
    state_gradient = kernel.slope * (
        model.alt.unsqueeze(-1) / kernel.probability
        - model.nonalt.unsqueeze(-1) / (1.0 - kernel.probability)
    )
    state_curvature = torch.square(kernel.slope) * (
        model.alt.unsqueeze(-1) / torch.square(kernel.probability)
        + model.nonalt.unsqueeze(-1) / torch.square(1.0 - kernel.probability)
    )
    gradient = -torch.sum(weights * state_gradient, dim=-1)
    hessian_upper = torch.sum(weights * state_curvature, dim=-1)
    prior = torch.exp(model.log_prior).masked_fill(~model.valid, 0.0)
    posterior = torch.where(model.observed.unsqueeze(-1), weights, prior)
    loss = torch.where(model.observed, loss, torch.zeros_like(loss))
    gradient = torch.where(model.observed, gradient, torch.zeros_like(gradient))
    hessian_upper = torch.where(
        model.observed,
        torch.clamp(hessian_upper, min=1e-8),
        torch.zeros_like(hessian_upper),
    )
    return TorchObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
    )


def observed_probability_and_slope_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return canonical path probabilities and left-segment slopes."""

    if tuple(phi.shape) != model.shape:
        raise ValueError(f"phi must have shape {model.shape}.")
    kernel = _path_kernel_torch(model, phi, eps=eps)
    return kernel.probability, kernel.slope


def observed_internal_breakpoints_torch(
    model: TorchObservedModel,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return path switches and clipping points with aligned validity."""

    epsilon = _validated_epsilon(eps)
    points = [model.switch]
    masks = [model.valid]
    for target in (epsilon, 1.0 - epsilon):
        left = torch.where(
            model.first_scale > 0.0,
            model.first_scale.new_full((), target) / model.first_scale,
            torch.full_like(model.first_scale, float("nan")),
        )
        left_valid = (
            model.valid & torch.isfinite(left) & (left >= 0.0) & (left <= model.switch)
        )
        right = torch.where(
            model.second_scale > 0.0,
            model.switch
            + (
                model.second_scale.new_full((), target)
                - model.first_scale * model.switch
            )
            / model.second_scale,
            torch.full_like(model.second_scale, float("nan")),
        )
        right_valid = (
            model.valid
            & torch.isfinite(right)
            & (right >= model.switch)
            & (right <= 1.0)
        )
        points.extend((left, right))
        masks.extend((left_valid, right_valid))
    return torch.cat(points, dim=-1), torch.cat(masks, dim=-1)


def observed_one_sided_gradients_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact left/right loss gradients at canonical breakpoints."""

    epsilon = _validated_epsilon(eps)
    if tuple(phi.shape) != model.shape:
        raise ValueError(f"phi must have shape {model.shape}.")
    kernel = _path_kernel_torch(model, phi, eps=epsilon)
    expanded_phi = phi.unsqueeze(-1)
    left_slope = torch.where(
        expanded_phi <= model.switch, model.first_scale, model.second_scale
    )
    right_slope = torch.where(
        expanded_phi < model.switch, model.first_scale, model.second_scale
    )
    outside = (kernel.mass < epsilon) | (kernel.mass > 1.0 - epsilon)
    left_slope = torch.where(outside, torch.zeros_like(left_slope), left_slope)
    right_slope = torch.where(outside, torch.zeros_like(right_slope), right_slope)
    left_slope = torch.where(
        kernel.mass <= epsilon, torch.zeros_like(left_slope), left_slope
    )
    right_slope = torch.where(
        kernel.mass >= 1.0 - epsilon, torch.zeros_like(right_slope), right_slope
    )
    joint = (
        model.alt.unsqueeze(-1) * torch.log(kernel.probability)
        + model.nonalt.unsqueeze(-1) * torch.log1p(-kernel.probability)
        + model.log_prior
    ).masked_fill(~model.valid, -torch.inf)
    posterior = torch.softmax(joint, dim=-1)
    state_factor = model.alt.unsqueeze(
        -1
    ) / kernel.probability - model.nonalt.unsqueeze(-1) / (1.0 - kernel.probability)
    gradient_left = -torch.sum(posterior * left_slope * state_factor, dim=-1)
    gradient_right = -torch.sum(posterior * right_slope * state_factor, dim=-1)
    gradient_left = torch.where(
        model.observed, gradient_left, torch.zeros_like(gradient_left)
    )
    gradient_right = torch.where(
        model.observed, gradient_right, torch.zeros_like(gradient_right)
    )
    points, valid = observed_internal_breakpoints_torch(model, eps=epsilon)
    at_breakpoint = torch.any(valid & (points == expanded_phi), dim=-1)
    return gradient_left, gradient_right, at_breakpoint


__all__ = [
    "BaseObjectiveKey",
    "LambdaObjectiveKey",
    "ObservedModel",
    "ObservedTerms",
    "TorchObservedModel",
    "TorchObservedTerms",
    "compile_observed_model",
    "default_phi_initialization",
    "emission_probability_scale",
    "make_base_objective_key",
    "make_lambda_objective_key",
    "model_to_torch",
    "observed_em_terms_torch",
    "observed_internal_breakpoints_torch",
    "observed_loss_grid_torch",
    "observed_box_fingerprint",
    "observed_one_sided_gradients_torch",
    "piecewise_affine_probability_upper",
    "observed_probability_and_slope_torch",
    "observed_terms_numpy",
    "observed_terms_torch",
]
