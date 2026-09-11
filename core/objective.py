"""Canonical source representation of CliPP2's observed-count likelihood.

The supported model is a uniform mixture over clonal integer multiplicities,
represented by clipped linear binomial emissions.
Runtime tensors are always rebuilt from immutable float64 source arrays; a
lower-precision runtime is never the source of a higher-precision view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch

from ..io.data import ImmutableArrayRecord, readonly_array as _readonly_array
from ..config import (
    CLONAL_INTEGER_GENERATOR_VERSION,
    CLONAL_INTEGER_MODEL_ID,
    CLONAL_INTEGER_PRIOR_MODE,
    validate_likelihood_precision,
)

if TYPE_CHECKING:
    from ..io.data import TumorData
    from .fusion.types import TorchRuntime


_MODEL_FINGERPRINT_SCHEMA = "clipp2.observed-model.linear.v3"
_LIKELIHOOD_FINGERPRINT_SCHEMA = "clipp2.observed-likelihood.linear.v3"
_BOX_FINGERPRINT_SCHEMA = "clipp2.objective-box.v1"
_BASE_OBJECTIVE_KEY_SCHEMA = "clipp2.base-objective-key.v1"
_LAMBDA_OBJECTIVE_KEY_SCHEMA = "clipp2.lambda-objective-key.v1"


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
    _hash_text(digest, model.coupling)
    _hash_text(digest, model.support_policy)
    for name in (
        "alt",
        "nonalt",
        "observed",
        "lower",
        "upper",
        "slope",
        "log_prior",
        "valid",
    ):
        _hash_array(digest, name, getattr(model, name))
    return digest.hexdigest()


def _likelihood_fingerprint(model: "ObservedModel") -> str:
    """Hash only the observed likelihood, excluding its feasible box."""

    digest = hashlib.sha256()
    _hash_text(digest, _LIKELIHOOD_FINGERPRINT_SCHEMA)
    _hash_text(digest, model.coupling)
    _hash_text(digest, model.support_policy)
    for name in (
        "alt",
        "nonalt",
        "observed",
        "slope",
        "log_prior",
        "valid",
    ):
        _hash_array(digest, name, getattr(model, name))
    return digest.hexdigest()


def _box_fingerprint(lower: np.ndarray, upper: np.ndarray) -> str:
    digest = hashlib.sha256()
    _hash_text(digest, _BOX_FINGERPRINT_SCHEMA)
    _hash_array(digest, "lower", np.asarray(lower, dtype=np.float64))
    _hash_array(digest, "upper", np.asarray(upper, dtype=np.float64))
    return digest.hexdigest()


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
class ObservedModel(ImmutableArrayRecord):
    """Immutable float64 source model for observed mutation counts.

    Candidate arrays have shape ``(mutation, region, candidate)``. A
    candidate's scaled mutant-copy mass is ``slope * phi``. Runtime views
    are reconstructed from these immutable float64 sources.
    """

    alt: np.ndarray
    nonalt: np.ndarray
    observed: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    slope: np.ndarray
    log_prior: np.ndarray
    valid: np.ndarray
    model_id: str
    coupling: str = field(default="independent", init=False)
    support_policy: str = field(default="independent_broad", init=False)
    fingerprint: str = field(init=False)
    likelihood_fingerprint: str = field(init=False)
    _convexity: dict[float, bool] = field(default_factory=dict, init=False, repr=False, compare=False)

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

        candidate_arrays = {
            name: np.array(getattr(self, name), dtype=np.float64, copy=True, order="C")
            for name in ("slope", "log_prior")
        }
        candidate_shape = candidate_arrays["slope"].shape
        if len(candidate_shape) != 3 or candidate_shape[:2] != shape or not candidate_shape[2]:
            raise ValueError(
                "ObservedModel candidate arrays must have nonempty shape (M, S, K)."
            )
        for name, value in candidate_arrays.items():
            if value.shape != candidate_shape:
                raise ValueError(f"ObservedModel.{name} must have shape {candidate_shape}.")
        valid = np.array(self.valid, dtype=bool, copy=True, order="C")
        if valid.shape != candidate_shape:
            raise ValueError(f"ObservedModel.valid must have shape {candidate_shape}.")
        if not np.all(np.any(valid, axis=-1)):
            raise ValueError("Every mutation-region entry must have a valid candidate.")
        slopes = candidate_arrays["slope"][valid]
        if np.any(~np.isfinite(slopes)) or np.any(slopes < 0.0):
            raise ValueError("Valid slope values must be finite and nonnegative.")
        if np.any(~np.isfinite(candidate_arrays["log_prior"][valid])):
            raise ValueError("Valid log_prior values must be finite.")

        candidate_arrays["slope"] = np.where(valid, candidate_arrays["slope"], 0.0)
        candidate_arrays["log_prior"] = np.where(
            valid, candidate_arrays["log_prior"], -np.inf
        )
        maximum = np.max(candidate_arrays["log_prior"], axis=-1, keepdims=True)
        normalizer = np.squeeze(
            maximum
            + np.log(
                np.sum(
                    np.where(
                        valid,
                        np.exp(candidate_arrays["log_prior"] - maximum),
                        0.0,
                    ),
                    axis=-1,
                    keepdims=True,
                )
            ),
            axis=-1,
        )
        if not np.allclose(normalizer, 0.0, rtol=0.0, atol=1e-10):
            raise ValueError("ObservedModel.log_prior must normalize over valid candidates.")

        model_id = str(self.model_id).strip()
        if not model_id:
            raise ValueError("ObservedModel.model_id must be nonempty.")
        for name, value in observation_arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype=np.float64))
        object.__setattr__(self, "observed", _readonly_array(observed, dtype=bool))
        for name, value in candidate_arrays.items():
            object.__setattr__(self, name, _readonly_array(value, dtype=np.float64))
        object.__setattr__(self, "valid", _readonly_array(valid, dtype=bool))
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "fingerprint", _model_fingerprint(self))
        object.__setattr__(
            self,
            "likelihood_fingerprint",
            _likelihood_fingerprint(self),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.alt.shape)

    @property
    def candidate_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.slope.shape)

    @property
    def candidate_generator_version(self) -> str:
        return CLONAL_INTEGER_GENERATOR_VERSION

    @property
    def prior_mode(self) -> str:
        return CLONAL_INTEGER_PRIOR_MODE


def has_proven_convex_observed_loss(model: ObservedModel | None, *, eps: float) -> bool:
    """Conservative convexity/endpoint-gradient qualification on the actual box.

    A singleton binomial emission is convex between clipping transitions, but
    its derivative can jump downward when a flat clipped region ends or
    begins. At the lower threshold its right derivative must be nonnegative;
    at the upper threshold its left derivative must be nonpositive. Include
    transitions touching the feasible endpoints: the kernel's zero derivative
    on the flat side cannot otherwise support a global KKT claim there.

    Unobserved, zero-depth and fixed-coordinate losses are constant. Other
    multi-candidate mixtures have no convexity proof here. False means this
    sufficient test did not qualify the loss, not necessarily nonconvexity.
    """
    if not isinstance(model, ObservedModel):
        return False
    try:
        epsilon = _validated_epsilon(eps)
    except ValueError:
        return False
    if epsilon not in model._convexity:
        model._convexity[epsilon] = _qualify_clipped_convexity(model, epsilon)
    return model._convexity[epsilon]


def _qualify_clipped_convexity(model: ObservedModel, epsilon: float) -> bool:
    active = model.observed & ((model.alt > 0.0) | (model.nonalt > 0.0))
    active &= model.lower < model.upper
    if np.any(active & (np.sum(model.valid, axis=-1) != 1)):
        return False
    slope = np.max(np.where(model.valid, model.slope, 0.0), axis=-1)
    mass_lower = slope * model.lower
    mass_upper = slope * model.upper
    upper_clip = 1.0 - epsilon
    # Entirely clipped intervals, including their flat-side endpoint, are
    # constant. Zero slope likewise cannot introduce a clipping transition.
    moving = active & (slope > 0.0) & (mass_upper > epsilon) & (mass_lower < upper_clip)
    lower_transition = moving & (mass_lower <= np.nextafter(epsilon, np.inf))
    upper_transition = moving & (mass_upper >= np.nextafter(upper_clip, -np.inf))

    # Cross products avoid dividing by a tiny clipping probability or summing
    # very large counts. Outward-rounded comparisons decline near-zero signs;
    # the exact zero-alt/all-alt sufficient cases remain available.
    lower_jump_nonnegative = (model.alt == 0.0) | (
        np.nextafter(model.nonalt * epsilon, -np.inf)
        >= np.nextafter(model.alt * upper_clip, np.inf)
    )
    upper_jump_nonnegative = (model.nonalt == 0.0) | (
        np.nextafter(model.alt * (1.0 - upper_clip), -np.inf)
        >= np.nextafter(model.nonalt * upper_clip, np.inf)
    )
    return not bool(np.any(
        (lower_transition & ~lower_jump_nonnegative)
        | (upper_transition & ~upper_jump_nonnegative)
    ))


@dataclass(frozen=True, slots=True)
class TorchObservedModel:
    """Runtime view rebuilt directly from an :class:`ObservedModel`."""

    alt: torch.Tensor
    nonalt: torch.Tensor
    observed: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    slope: torch.Tensor
    log_prior: torch.Tensor
    valid: torch.Tensor
    model_id: str
    source_fingerprint: str
    coupling: str = field(default="independent", init=False)
    support_policy: str = field(default="independent_broad", init=False)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.alt.shape)

    @property
    def candidate_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.slope.shape)

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
    loss: torch.Tensor
    gradient: torch.Tensor
    hessian_upper: torch.Tensor
    posterior: torch.Tensor


@dataclass(frozen=True, slots=True)
class _NumpyEmissionKernel:
    mass: np.ndarray
    probability: np.ndarray
    slope: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _TorchEmissionKernel:
    mass: torch.Tensor
    probability: torch.Tensor
    slope: torch.Tensor | None


def _compile_integer_candidates(major_cn: np.ndarray, scaling: np.ndarray) -> dict[str, object]:
    """Compile complete ordered 1..major_cn support with fixed uniform priors."""
    major = np.asarray(major_cn, dtype=np.float64)
    scale = np.asarray(scaling, dtype=np.float64)
    if major.ndim != 2 or 0 in major.shape or scale.shape != major.shape:
        raise ValueError("major_cn and scaling must have nonempty shape (M, S).")
    if not np.all(np.isfinite(major)) or not np.allclose(major, np.rint(major), rtol=0.0, atol=1e-8):
        raise ValueError("Retained major_cn must contain finite integers.")
    major = np.rint(major)
    if np.any(major < 1):
        raise ValueError("Retained major_cn must be positive; apply CN filtering first.")
    candidates = np.arange(1, int(major.max()) + 1, dtype=np.float64)
    valid = candidates <= major[..., None]
    return {
        "slope": scale[..., None] * np.where(valid, candidates, 0.0),
        "log_prior": np.where(valid, -np.log(major)[..., None], -np.inf),
        "valid": valid,
        "model_id": CLONAL_INTEGER_MODEL_ID,
    }


def compile_integer_observations(
    *, alt_counts: np.ndarray, total_counts: np.ndarray,
    count_observed: np.ndarray | None, phi_upper: np.ndarray,
    major_cn: np.ndarray, scaling: np.ndarray, eps: float,
) -> ObservedModel:
    """Compile canonical arrays before the final immutable input is constructed."""
    epsilon = _validated_epsilon(eps)
    alt = np.asarray(alt_counts, dtype=np.float64)
    total = np.asarray(total_counts, dtype=np.float64)
    if alt.shape != total.shape:
        raise ValueError("TumorData alt_counts and total_counts must have one shape.")
    candidates = _compile_integer_candidates(major_cn, scaling)
    return ObservedModel(
        alt=alt, nonalt=total - alt,
        observed=np.ones(alt.shape, dtype=bool) if count_observed is None else count_observed,
        lower=np.full(alt.shape, epsilon, dtype=np.float64), upper=phi_upper,
        **candidates,
    )


def compile_observed_model(
    data: "TumorData",
    *,
    eps: float,
) -> ObservedModel:
    """Compile the supported uniform integer likelihood into float64 sources."""

    epsilon = _validated_epsilon(eps)
    key = (epsilon, "independent_broad")
    cached = data._compiled_models.get(key)
    if cached is not None:
        return cached
    model = compile_integer_observations(
        alt_counts=data.alt_counts, total_counts=data.total_counts,
        count_observed=data.count_observed, phi_upper=data.phi_upper,
        major_cn=data.major_cn, scaling=data.scaling, eps=epsilon,
    )
    return data._compiled_models.setdefault(key, model)


def model_to_torch(
    model: ObservedModel,
    runtime: "TorchRuntime",
    *,
    eps: float,
) -> TorchObservedModel:
    """Build a runtime view from immutable source arrays, never another view."""

    dtype = runtime.dtype
    device = runtime.device
    epsilon = _validated_epsilon(eps, dtype)
    _validate_candidate_range(model, epsilon, dtype)

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
        slope=numeric(model.slope),
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

    epsilon = _validated_epsilon(eps)
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


def _validated_epsilon(eps: float, dtype: torch.dtype = torch.float64) -> float:
    name = {torch.float32: "float32", torch.float64: "float64"}.get(dtype)
    if name is None:
        raise ValueError("Observed-model runtime dtype must be float32 or float64.")
    return validate_likelihood_precision(eps, name)


def _validate_candidate_range(model: ObservedModel, eps: float, dtype: torch.dtype) -> None:
    """Conservatively preflight intermediate arithmetic from float64 sources.

    Interior endpoints alone do not prevent count/probability-squared overflow.
    Bound the *unweighted* candidate intermediates, including padded entries;
    posterior or observed masks applied later cannot repair NaNs.
    """
    cast_endpoint = np.float32 if dtype == torch.float32 else np.float64
    lower = float(cast_endpoint(eps))
    complement = 1.0 - float(cast_endpoint(1.0 - eps))
    count = max(float(model.alt.max()), float(model.nonalt.max()))
    slope = max(float(model.slope.max()), 1.0)
    # Bounds cover count conversion, log-kernels, scores, summed curvature
    # intermediates, and slope squares; log space avoids overflow in preflight.
    log_slope = np.log(slope)
    log_limit = np.log(torch.finfo(dtype).max)
    log_bound = 2.0 * log_slope
    if count > 0.0:
        log_bound = max(
            log_bound,
            np.log(count) + np.log(2.0) + 2.0 * log_slope
            - 2.0 * np.log(min(lower, complement)),
        )
    if log_bound >= log_limit - np.log(2.0):
        raise ValueError(
            f"Candidate arithmetic may overflow in {dtype} for these counts, "
            "slopes and eps. Use float64 or a representable numerical input "
            "before preparation; the clipping rule is never adjusted."
        )


def _emission_kernel_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
    derivatives: bool = True,
) -> _NumpyEmissionKernel:
    epsilon = _validated_epsilon(eps)
    phi_array = np.asarray(phi, dtype=np.float64)
    if phi_array.shape != model.shape or not np.all(np.isfinite(phi_array)):
        raise ValueError(f"phi must be a finite array with shape {model.shape}.")
    mass = model.slope * phi_array[..., None]
    probability = np.clip(mass, epsilon, 1.0 - epsilon)
    slope = None
    if derivatives:
        slope = np.where(
            (mass > epsilon) & (mass < 1.0 - epsilon),
            model.slope,
            0.0,
        )
    return _NumpyEmissionKernel(mass=mass, probability=probability, slope=slope)


def candidate_terms_numpy(alt, nonalt, probability, slope, *, derivative_order=2):
    """Binomial candidate log-kernel and optional score/curvature terms.

    Inputs broadcast over candidates and grids; probability and slope already
    obey the clipping rule. Derivative order controls numerical work only,
    never marginalization or EM responsibilities.
    """
    log_kernel = alt * np.log(probability) + nonalt * np.log1p(-probability)
    score = curvature = None
    if derivative_order >= 1:
        score = slope * (alt / probability - nonalt / (1.0 - probability))
    if derivative_order >= 2:
        curvature = np.square(slope) * (
            alt / np.square(probability) + nonalt / np.square(1.0 - probability)
        )
    return log_kernel, score, curvature


def _candidate_terms_torch(alt, nonalt, probability, slope, *, derivative_order=2):
    """Torch candidate arithmetic; observed and fixed-weight reductions stay separate."""
    log_kernel = alt * torch.log(probability) + nonalt * torch.log1p(-probability)
    score = curvature = None
    if derivative_order >= 1:
        score = slope * (alt / probability - nonalt / (1.0 - probability))
    if derivative_order >= 2:
        curvature = torch.square(slope) * (
            alt / torch.square(probability) + nonalt / torch.square(1.0 - probability)
        )
    return log_kernel, score, curvature


def _emission_kernel_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
    derivatives: bool = True,
) -> _TorchEmissionKernel:
    """Evaluate every canonical candidate over optional trailing grid dimensions.

    ``phi`` has shape ``(M, S, *grid)``. Runtime model arrays are reshaped,
    not copied, so the kernel stays resident on its original device while a
    caller evaluates any number of candidate CCFs per mutation-region.
    """

    epsilon = _validated_epsilon(eps, model.alt.dtype)
    if phi.ndim < 2 or tuple(phi.shape[:2]) != model.shape:
        raise ValueError(
            "phi must start with the observed-model shape "
            f"{model.shape}, not {tuple(phi.shape)}."
        )
    if phi.dtype != model.alt.dtype or phi.device != model.alt.device:
        raise ValueError("phi must use the observed model's runtime dtype and device.")
    grid_ndim = phi.ndim - 2
    candidate_shape = (*model.shape, *((1,) * grid_ndim), model.candidate_shape[-1])

    def candidate_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(candidate_shape)

    expanded_phi = phi.unsqueeze(-1)
    candidate_slope = candidate_view(model.slope)
    mass = candidate_slope * expanded_phi
    probability = torch.clamp(mass, min=epsilon, max=1.0 - epsilon)
    slope = None
    if derivatives:
        slope = torch.where(
            (mass > epsilon) & (mass < 1.0 - epsilon),
            candidate_slope,
            torch.zeros_like(candidate_slope),
        )
    return _TorchEmissionKernel(mass=mass, probability=probability, slope=slope)


def _observed_reduction_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
    output: Literal["loss", "posterior", "terms"],
) -> np.ndarray | ObservedTerms:
    """One marginalized reduction, with only the requested numerical work."""

    derivatives = output == "terms"
    kernel = _emission_kernel_numpy(model, phi, eps=eps, derivatives=derivatives)
    log_kernel, state_gradient, state_curvature = candidate_terms_numpy(
        model.alt[..., None], model.nonalt[..., None], kernel.probability, kernel.slope,
        derivative_order=2 if derivatives else 0,
    )
    joint = np.where(model.valid, log_kernel + model.log_prior, -np.inf)
    del log_kernel
    maximum = np.max(joint, axis=-1, keepdims=True)
    unnormalized = np.exp(joint - maximum)
    denominator = np.sum(unnormalized, axis=-1, keepdims=True)
    if output != "posterior":
        log_normalizer = np.squeeze(maximum + np.log(denominator), axis=-1)
        loss = -log_normalizer
    if output == "loss":
        return np.where(model.observed, loss, 0.0)
    posterior = unnormalized / denominator
    if derivatives:
        gradient = -np.sum(posterior * state_gradient, axis=-1)
        hessian_upper = np.sum(posterior * state_curvature, axis=-1)
    prior = np.where(model.valid, np.exp(model.log_prior), 0.0)
    posterior = ((np.where(model.observed[..., None], posterior, prior)))
    if output == "posterior":
        return posterior
    loss = np.where(model.observed, loss, 0.0)
    gradient = np.where(model.observed, gradient, 0.0)
    hessian_upper = np.where(
        model.observed, np.maximum(hessian_upper, 1e-8), 0.0
    )
    return ObservedTerms(
        loss=loss,
        gradient=gradient,
        hessian_upper=hessian_upper,
        posterior=posterior,
    )


def observed_terms_numpy(
    model: ObservedModel,
    phi: np.ndarray,
    *,
    eps: float,
) -> ObservedTerms:
    """Evaluate loss, left-gradient, curvature majorant, and candidate posterior."""

    return cast(ObservedTerms, _observed_reduction_numpy(model, phi, eps=eps, output="terms"))


def _state_logits_torch(model, log_kernel):
    """Per-region categorical log probabilities over integer multiplicities."""
    extra = (1,) * (log_kernel.ndim - 3)
    shape = (*model.shape, *extra, model.candidate_shape[-1])
    prior, valid = model.log_prior.reshape(shape), model.valid.reshape(shape)
    return (log_kernel + prior).masked_fill(~valid, -torch.inf)


def observed_terms_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> TorchObservedTerms:
    """Torch counterpart of :func:`observed_terms_numpy`."""

    if tuple(phi.shape) != tuple(model.alt.shape):
        raise ValueError(f"phi must have shape {tuple(model.alt.shape)}.")
    kernel = _emission_kernel_torch(model, phi, eps=eps)
    log_kernel, state_gradient, state_curvature = _candidate_terms_torch(
        model.alt.unsqueeze(-1), model.nonalt.unsqueeze(-1), kernel.probability, kernel.slope,
    )
    joint = _state_logits_torch(model, log_kernel)
    del log_kernel
    log_normalizer = torch.logsumexp(joint, dim=-1)
    posterior = torch.softmax(joint, dim=-1)
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

    kernel = _emission_kernel_torch(model, phi, eps=eps, derivatives=False)
    grid_ndim = phi.ndim - 2
    observation_shape = (*model.shape, *((1,) * grid_ndim))

    def observation_view(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(observation_shape)

    log_kernel, _, _ = _candidate_terms_torch(
        observation_view(model.alt).unsqueeze(-1), observation_view(model.nonalt).unsqueeze(-1),
        kernel.probability, kernel.slope, derivative_order=0,
    )
    joint = _state_logits_torch(model, log_kernel)
    del log_kernel
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
    if tuple(responsibilities.shape) != model.candidate_shape:
        raise ValueError(
            f"responsibilities must have shape {model.candidate_shape}, "
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
        raise ValueError("responsibilities must assign mass to a valid candidate.")
    weights = weights / normalizer
    kernel = _emission_kernel_torch(model, phi, eps=eps)
    log_kernel, state_gradient, state_curvature = _candidate_terms_torch(
        model.alt.unsqueeze(-1), model.nonalt.unsqueeze(-1), kernel.probability, kernel.slope,
    )
    complete_loss = torch.where(
        model.valid,
        (-(log_kernel + model.log_prior)),
        torch.zeros_like(log_kernel),
    )
    del log_kernel
    entropy = torch.where(
        weights > 0.0,
        weights * torch.log(torch.clamp(weights, min=torch.finfo(weights.dtype).tiny)),
        torch.zeros_like(weights),
    )
    loss = torch.sum(weights * complete_loss + entropy, dim=-1)
    gradient = -torch.sum(weights * state_gradient, dim=-1)
    hessian_upper = torch.sum(weights * state_curvature, dim=-1)
    prior = torch.exp(model.log_prior).masked_fill(~model.valid, 0.0)
    posterior = ((torch.where(model.observed.unsqueeze(-1), weights, prior)))
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


def observed_internal_breakpoints_torch(
    model: TorchObservedModel,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return linear-emission clipping points with aligned validity."""

    epsilon = _validated_epsilon(eps, model.alt.dtype)
    points = []
    masks = []
    for target in (epsilon, 1.0 - epsilon):
        point = torch.where(
            model.slope > 0.0,
            model.slope.new_full((), target) / model.slope,
            torch.full_like(model.slope, float("nan")),
        )
        valid = (
            model.valid
            & torch.isfinite(point)
            & (point >= 0.0)
            & (point <= 1.0)
        )
        points.append(point)
        masks.append(valid)
    return torch.cat(points, dim=-1), torch.cat(masks, dim=-1)


def observed_one_sided_gradients_torch(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact left/right loss gradients at canonical breakpoints."""

    epsilon = _validated_epsilon(eps, model.alt.dtype)
    if tuple(phi.shape) != model.shape:
        raise ValueError(f"phi must have shape {model.shape}.")
    kernel = _emission_kernel_torch(model, phi, eps=epsilon)
    expanded_phi = phi.unsqueeze(-1)
    left_slope = model.slope
    right_slope = model.slope
    outside = (kernel.mass < epsilon) | (kernel.mass > 1.0 - epsilon)
    left_slope = torch.where(outside, torch.zeros_like(left_slope), left_slope)
    right_slope = torch.where(outside, torch.zeros_like(right_slope), right_slope)
    left_slope = torch.where(
        kernel.mass <= epsilon, torch.zeros_like(left_slope), left_slope
    )
    right_slope = torch.where(
        kernel.mass >= 1.0 - epsilon, torch.zeros_like(right_slope), right_slope
    )
    log_kernel = (
        model.alt.unsqueeze(-1) * torch.log(kernel.probability)
        + model.nonalt.unsqueeze(-1) * torch.log1p(-kernel.probability)
    )
    joint = _state_logits_torch(model, log_kernel)
    posterior = torch.softmax(joint, dim=-1)
    state_factor = model.alt.unsqueeze(-1) / kernel.probability - model.nonalt.unsqueeze(
        -1
    ) / (1.0 - kernel.probability)
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


@dataclass(frozen=True)
class IntegerMultiplicityPosterior:
    """Categorical multiplicity posterior conditional on the supplied CCF.

    ``multiplicity_call`` is the smallest MAP candidate, including prior-only
    ties. Reporting must hide uninformative calls with multiple candidates;
    a singleton remains structurally fixed at one.
    """

    posterior: np.ndarray
    multiplicity_call: np.ndarray
    map_probability: np.ndarray
    candidate_count: np.ndarray
    informative: np.ndarray


def infer_integer_multiplicity_posterior_numpy(
    data: TumorData,
    phi: np.ndarray,
    *,
    eps: float,
) -> IntegerMultiplicityPosterior:
    """Evaluate the fitted integer mixture without replacing marginalization."""

    phi_array = np.asarray(phi, dtype=np.float64)
    expected_shape = np.asarray(data.alt_counts).shape
    if phi_array.shape != expected_shape:
        raise ValueError(
            "phi must have the same mutation-region shape as the tumor data; "
            f"got {phi_array.shape}, expected {expected_shape}."
        )
    if not np.all(np.isfinite(phi_array)):
        raise ValueError("phi must contain only finite values.")
    model = compile_observed_model(data, eps=eps)
    if np.any((phi_array < model.lower) | (phi_array > model.upper)):
        raise ValueError("phi must lie inside the compiled CCF bounds.")

    posterior = np.asarray(
        _observed_reduction_numpy(model, phi_array, eps=eps, output="posterior"),
        dtype=np.float64,
    )
    # The compiler validates the complete, increasingly ordered integer range.
    # np.argmax therefore selects the lowest candidate on an exact tie.
    map_index = np.argmax(np.where(model.valid, posterior, -np.inf), axis=-1)
    probability = np.take_along_axis(
        posterior, map_index[..., None], axis=-1
    )[..., 0]
    return IntegerMultiplicityPosterior(
        posterior=posterior,
        multiplicity_call=map_index.astype(np.int64) + 1,
        map_probability=probability,
        candidate_count=np.sum(model.valid, axis=-1).astype(np.int64),
        informative=((model.observed & ((model.alt + model.nonalt) > 0.0))),
    )


__all__ = [
    "IntegerMultiplicityPosterior",
    "infer_integer_multiplicity_posterior_numpy",
    "BaseObjectiveKey",
    "LambdaObjectiveKey",
    "ObservedModel",
    "ObservedTerms",
    "TorchObservedModel",
    "TorchObservedTerms",
    "compile_observed_model",
    "has_proven_convex_observed_loss",
    "make_base_objective_key",
    "make_lambda_objective_key",
    "model_to_torch",
    "observed_em_terms_torch",
    "observed_internal_breakpoints_torch",
    "observed_loss_grid_torch",
    "observed_one_sided_gradients_torch",
    "observed_terms_numpy",
    "observed_terms_torch",
]
