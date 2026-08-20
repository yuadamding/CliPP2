"""Canonical source representation of CliPP2's observed-count likelihood.

This module is intentionally not wired into the production solver yet.  It
provides one representation in which both the historical major/minor model
and an explicit occupancy-path model are mixtures of clipped piecewise-affine
binomial emissions.  Runtime tensors are always rebuilt from the immutable
float64 source arrays; a lower-precision runtime is never the source of a
higher-precision view.
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


@dataclass(frozen=True, slots=True)
class ObservedModel:
    """Immutable float64 source model for observed mutation counts.

    The path arrays have shape ``(mutation, region, path)``.  For CCF ``phi``,
    a path's scaled mutant-copy mass is

    ``first_scale * min(phi, switch) + second_scale * max(phi-switch, 0)``.

    ``legacy_major`` is reporting metadata and is deliberately excluded from
    the numerical fingerprint.  ``model_id`` likewise names the source family
    without changing the represented likelihood.
    """

    alt: np.ndarray
    nonalt: np.ndarray
    observed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    first_scale: np.ndarray
    second_scale: np.ndarray
    switch: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    legacy_major: np.ndarray | None
    model_id: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        observation_arrays = {
            name: np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            for name in ("alt", "nonalt", "lower", "upper")
        }
        shape = observation_arrays["alt"].shape
        if len(shape) != 2 or not shape[0] or not shape[1]:
            raise ValueError("ObservedModel observations must have nonempty shape (M, S).")
        for name, value in observation_arrays.items():
            if value.shape != shape:
                raise ValueError(f"ObservedModel.{name} must have shape {shape}.")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"ObservedModel.{name} must contain only finite values.")
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
            for name in ("first_scale", "second_scale", "switch", "log_prior")
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
        for name in ("first_scale", "second_scale"):
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

        for name in ("first_scale", "second_scale", "switch"):
            path_arrays[name] = np.where(valid, path_arrays[name], 0.0)
        path_arrays["log_prior"] = np.where(
            valid, path_arrays["log_prior"], -np.inf
        )
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

        legacy_major = self.legacy_major
        if legacy_major is not None:
            legacy_major = np.array(
                legacy_major, dtype=bool, copy=True, order="C"
            )
            if legacy_major.shape != path_shape:
                raise ValueError(
                    f"ObservedModel.legacy_major must have shape {path_shape}."
                )
            legacy_major &= valid

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
            "legacy_major",
            None
            if legacy_major is None
            else _readonly_array(legacy_major, dtype=bool),
        )
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "fingerprint", _model_fingerprint(self))

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.alt.shape)

    @property
    def path_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.first_scale.shape)


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
    legacy_major: torch.Tensor | None
    model_id: str
    source_fingerprint: str


@dataclass(frozen=True, slots=True)
class ObservedTerms:
    loss: np.ndarray
    gradient: np.ndarray
    hessian_upper: np.ndarray
    posterior: np.ndarray
    legacy_major_probability: np.ndarray


@dataclass(frozen=True, slots=True)
class TorchObservedTerms:
    loss: torch.Tensor
    gradient: torch.Tensor
    hessian_upper: torch.Tensor
    posterior: torch.Tensor
    legacy_major_probability: torch.Tensor


def _compile_legacy_as_paths(data: "TumorData", major_prior: float) -> dict[str, object]:
    prior = float(major_prior)
    if not np.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("major_prior must lie strictly in (0, 1).")
    scaling = np.asarray(data.scaling, dtype=np.float64)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    fixed = scaling * np.asarray(data.fixed_multiplicity, dtype=np.float64)
    minor = scaling * np.asarray(data.minor_cn, dtype=np.float64)
    major = scaling * np.asarray(data.major_cn, dtype=np.float64)
    first_scale = np.stack((np.where(ambiguous, minor, fixed), major), axis=-1)
    valid = np.stack((np.ones_like(ambiguous), ambiguous), axis=-1)
    log_prior = np.stack(
        (
            np.where(ambiguous, np.log1p(-prior), 0.0),
            np.full_like(fixed, np.log(prior)),
        ),
        axis=-1,
    )
    legacy_major = np.stack((~ambiguous, np.ones_like(ambiguous)), axis=-1)
    return {
        "first_scale": first_scale,
        "second_scale": first_scale.copy(),
        "switch": np.zeros_like(first_scale),
        "log_prior": np.where(valid, log_prior, -np.inf),
        "valid": valid,
        "legacy_major": legacy_major & valid,
        "model_id": "legacy_major_minor_as_paths_v1",
    }


def _compile_explicit_paths(data: "TumorData") -> dict[str, object]:
    spec = data.path_likelihood
    if spec is None:
        raise ValueError("TumorData does not contain an explicit path likelihood.")
    shape = tuple(int(value) for value in np.asarray(data.alt_counts).shape)
    spec.validate_observation_shape(shape)
    if not np.allclose(
        np.asarray(data.phi_upper, dtype=np.float64),
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("An explicit path likelihood requires full CCF support [0, 1].")
    scale = np.asarray(data.scaling, dtype=np.float64)[..., None]
    return {
        "first_scale": scale * np.asarray(spec.first_copy, dtype=np.float64),
        "second_scale": scale * np.asarray(spec.second_copy, dtype=np.float64),
        "switch": np.asarray(spec.switch_fraction, dtype=np.float64),
        "log_prior": np.asarray(spec.log_prior, dtype=np.float64),
        "valid": np.asarray(spec.valid, dtype=bool),
        "legacy_major": (
            None
            if spec.legacy_major_indicator is None
            else np.asarray(spec.legacy_major_indicator, dtype=bool)
        ),
        "model_id": spec.model_id,
    }


def compile_observed_model(
    data: "TumorData",
    *,
    major_prior: float,
    eps: float,
) -> ObservedModel:
    """Compile either supported likelihood family into one float64 model."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    if alt.shape != total.shape:
        raise ValueError("TumorData alt_counts and total_counts must have one shape.")
    observed_value = getattr(data, "count_observed", None)
    observed = (
        np.ones(alt.shape, dtype=bool)
        if observed_value is None
        else np.asarray(observed_value, dtype=bool)
    )
    compiled = (
        _compile_legacy_as_paths(data, major_prior)
        if getattr(data, "path_likelihood", None) is None
        else _compile_explicit_paths(data)
    )
    return ObservedModel(
        alt=alt,
        nonalt=total - alt,
        observed=observed,
        lower=np.full(alt.shape, epsilon, dtype=np.float64),
        upper=np.asarray(data.phi_upper, dtype=np.float64),
        **compiled,
    )


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
        legacy_major=(
            None if model.legacy_major is None else boolean(model.legacy_major)
        ),
        model_id=model.model_id,
        source_fingerprint=model.fingerprint,
    )


def observed_terms_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
) -> ObservedTerms:
    """Evaluate loss, left-gradient, curvature majorant, and path posterior."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
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
    hessian_upper = np.where(
        model.observed, np.maximum(hessian_upper, 1e-8), 0.0
    )
    legacy_major_probability = (
        np.ones(model.shape, dtype=np.float64)
        if model.legacy_major is None
        else np.sum(posterior * model.legacy_major, axis=-1)
    )
    return ObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
        legacy_major_probability=legacy_major_probability,
    )


def observed_terms_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> TorchObservedTerms:
    """Torch counterpart of :func:`observed_terms_numpy`."""

    epsilon = float(eps)
    if not np.isfinite(epsilon) or not 0.0 < epsilon < 0.5:
        raise ValueError("eps must be finite and lie strictly in (0, 0.5).")
    if tuple(phi.shape) != tuple(model.alt.shape):
        raise ValueError(f"phi must have shape {tuple(model.alt.shape)}.")
    expanded_phi = phi.unsqueeze(-1)
    mass = model.first_scale * torch.minimum(expanded_phi, model.switch)
    mass = mass + model.second_scale * torch.clamp(
        expanded_phi - model.switch, min=0.0
    )
    probability = torch.clamp(mass, min=epsilon, max=1.0 - epsilon)
    segment_slope = torch.where(
        expanded_phi <= model.switch,
        model.first_scale,
        model.second_scale,
    )
    slope = torch.where(
        (mass > epsilon) & (mass < 1.0 - epsilon),
        segment_slope,
        torch.zeros_like(segment_slope),
    )
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
    legacy_major_probability = (
        torch.ones_like(loss)
        if model.legacy_major is None
        else torch.sum(posterior * model.legacy_major.to(posterior.dtype), dim=-1)
    )
    return TorchObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
        legacy_major_probability=legacy_major_probability,
    )


__all__ = [
    "ObservedModel",
    "ObservedTerms",
    "TorchObservedModel",
    "TorchObservedTerms",
    "compile_observed_model",
    "model_to_torch",
    "observed_terms_numpy",
    "observed_terms_torch",
]
