from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...io.data import TumorData, tumor_data_fingerprint
from .graph_ops import (
    PDHG_PRECONDITIONER_ETA,
    graph_adjoint_edges,
    graph_forward_edges,
    project_dual_ball,
)
from .types import ObjectiveTerms, TorchRuntime


@dataclass(frozen=True)
class TorchTumorData:
    alt: torch.Tensor
    total: torch.Tensor
    nonalt: torch.Tensor
    phi_upper: torch.Tensor
    ambiguous: torch.Tensor
    b_minus: torch.Tensor
    b_plus: torch.Tensor
    b_fixed: torch.Tensor
    count_observed: torch.Tensor | None = None  # bool (M, S); None means all observed
    data_fingerprint: str = ""


@dataclass(frozen=True)
class TorchMutationRegionTerms:
    loss: torch.Tensor
    grad: torch.Tensor
    hess_upper: torch.Tensor
    gamma_major: torch.Tensor


DEFAULT_INNER_KKT_CHECK_EVERY = 8
DEFAULT_BOX_PHI_ATOL = 1e-8
DEFAULT_BOX_MAX_ITER = 32
# Maximum size of one edge-by-region work tensor.  Complete-graph ADMM keeps
# its two mathematical edge states, but all additional edge work is streamed
# in chunks bounded by this value.
DEFAULT_EDGE_WORK_BYTES = 64 * 1024 * 1024

_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
}


def dtype_name(dtype: torch.dtype) -> str:
    """Inverse of resolve_runtime's dtype parsing (e.g. torch.float64 -> 'float64').

    Replaces the fragile ``str(dtype).replace('torch.', '')`` round-trip and raises
    on an unsupported dtype instead of silently emitting a bad string.
    """
    try:
        return _DTYPE_TO_NAME[dtype]
    except KeyError:
        raise ValueError(f"Unsupported runtime dtype: {dtype!r}")


def as_runtime_tensor(start, runtime: "TorchRuntime") -> torch.Tensor:
    """Move/convert an array-like onto the runtime device & dtype.

    The single conversion idiom shared by the solver, adaptive, and partition
    layers (previously duplicated as _tensor_from_start / _runtime_start_tensor /
    _as_torch). An existing tensor is cast in place; a numpy/array-like is wrapped
    with ``torch.as_tensor`` (never ``from_numpy`` / ``torch.tensor(tensor)``).
    """
    if torch.is_tensor(start):
        return start.to(dtype=runtime.dtype, device=runtime.device)
    return torch.as_tensor(
        np.asarray(start), dtype=runtime.dtype, device=runtime.device
    )


def validate_lambda_value(lambda_value: float) -> float:
    value = float(lambda_value)
    if not np.isfinite(value):
        raise ValueError("lambda_value must be finite.")
    if value < 0.0:
        raise ValueError("lambda_value must be nonnegative.")
    return value


def _edge_chunk_size(
    *,
    num_edges: int,
    num_regions: int,
    dtype: torch.dtype,
    work_bytes: int | None = None,
) -> int:
    """Number of edges whose single ``(edge, region)`` tensor fits the budget."""
    budget = DEFAULT_EDGE_WORK_BYTES if work_bytes is None else int(work_bytes)
    if budget <= 0:
        raise ValueError("edge work budget must be positive.")
    edges = max(int(num_edges), 0)
    if edges == 0:
        return 1
    regions = max(int(num_regions), 1)
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(1, min(edges, budget // max(regions * element_size, 1)))


def _edge_tensor_nbytes(*, num_edges: int, num_regions: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return max(int(num_edges), 0) * max(int(num_regions), 0) * int(element_size)


def _edge_slices(num_edges: int, chunk_size: int):
    for start in range(0, int(num_edges), max(int(chunk_size), 1)):
        yield slice(start, min(start + int(chunk_size), int(num_edges)))


def _box_qp_sweeps_for_atol(
    phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    *,
    max_iter: int = DEFAULT_BOX_MAX_ITER,
) -> int:
    atol = float(phi_atol)
    if not np.isfinite(atol) or atol <= 0.0:
        atol = DEFAULT_BOX_PHI_ATOL
    requested = (
        int(np.ceil(np.log2(1.0 / min(max(atol, np.finfo(float).tiny), 1.0)))) + 1
    )
    return max(16, min(max(int(max_iter), 16), requested))


def _inner_kkt_audit_due(
    *,
    iteration: int,
    max_iter: int,
    kkt_check_every: int,
    cheap_converged: bool,
) -> bool:
    if bool(cheap_converged):
        return True
    if int(iteration) >= int(max_iter):
        return True
    check_every = max(int(kkt_check_every), 1)
    return int(iteration) % check_every == 0


def _binary_entropy_offset_torch(weight: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(weight, min=0.0, max=1.0)
    positive = clipped > 0.0
    one_minus = 1.0 - clipped
    positive_complement = one_minus > 0.0
    term = torch.zeros_like(clipped)
    term = torch.where(positive, term + clipped * torch.log(clipped), term)
    term = torch.where(
        positive_complement, term + one_minus * torch.log(one_minus), term
    )
    return term


def resolve_runtime(device: str | None, *, dtype: str | None = None) -> TorchRuntime:
    """Resolve a device/dtype string pair into a TorchRuntime (the single place
    that maps strings to torch.device/torch.dtype).

    Determinism note: CUDA fits are not bit-reproducible run-to-run (e.g. the
    float index_add_ in graph_ops.graph_adjoint_edges is nondeterministic on GPU),
    and CPU vs GPU labels can differ near lambda-path decision boundaries. For
    reproducible GPU runs, enable torch.use_deterministic_algorithms(True) and set
    CUBLAS_WORKSPACE_CONFIG before fitting.
    """
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested Torch device {device!r}, but CUDA is not available. "
            "Use device='cpu' or device='auto' to permit CPU execution."
        )
    runtime_device = torch.device(requested)
    requested_dtype = "auto" if dtype is None else str(dtype).strip().lower()
    if requested_dtype == "auto":
        runtime_dtype = torch.float64
    elif requested_dtype == "float16":
        runtime_dtype = torch.float16
    elif requested_dtype == "float32":
        runtime_dtype = torch.float32
    elif requested_dtype == "float64":
        runtime_dtype = torch.float64
    else:
        raise ValueError(f"Unknown runtime dtype: {dtype}")
    if runtime_dtype == torch.float16 and runtime_device.type != "cuda":
        raise RuntimeError("Float16 runtime dtype is only supported on CUDA.")
    device_name = (
        runtime_device.type
        if runtime_device.index is None
        else f"{runtime_device.type}:{runtime_device.index}"
    )
    return TorchRuntime(
        device=runtime_device, device_name=device_name, dtype=runtime_dtype
    )


def to_torch_tumor_data(data: TumorData, runtime: TorchRuntime) -> TorchTumorData:
    dtype = runtime.dtype
    device = runtime.device
    scaling = np.asarray(data.scaling, dtype=np.float64)
    count_obs_np = getattr(data, "count_observed", None)
    count_obs_tensor = (
        torch.as_tensor(count_obs_np, dtype=torch.bool, device=device)
        if count_obs_np is not None
        else None
    )
    return TorchTumorData(
        alt=torch.as_tensor(data.alt_counts, dtype=dtype, device=device),
        total=torch.as_tensor(data.total_counts, dtype=dtype, device=device),
        nonalt=torch.as_tensor(
            data.total_counts - data.alt_counts, dtype=dtype, device=device
        ),
        phi_upper=torch.as_tensor(data.phi_upper, dtype=dtype, device=device),
        ambiguous=torch.as_tensor(
            data.multiplicity_estimation_mask, dtype=torch.bool, device=device
        ),
        b_minus=torch.as_tensor(
            scaling * np.asarray(data.minor_cn, dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
        b_plus=torch.as_tensor(
            scaling * np.asarray(data.major_cn, dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
        b_fixed=torch.as_tensor(
            scaling * np.asarray(data.fixed_multiplicity, dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
        count_observed=count_obs_tensor,
        data_fingerprint=tumor_data_fingerprint(data),
    )


def validate_torch_tumor_data(
    tensor_data: TorchTumorData,
    *,
    data: TumorData,
    runtime: TorchRuntime,
    expected_fingerprint: str | None = None,
) -> None:
    """Reject stale or runtime-incompatible tensors before solver reuse."""

    fingerprint = expected_fingerprint or tumor_data_fingerprint(data)
    if tensor_data.data_fingerprint != fingerprint:
        raise ValueError("TorchTumorData fingerprint does not match TumorData.")

    expected_shape = (int(data.num_mutations), int(data.num_regions))

    def validate_field(name: str, *, dtype: torch.dtype) -> None:
        value = getattr(tensor_data, name)
        if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
            raise ValueError(f"TorchTumorData.{name} must have shape {expected_shape}.")
        if value.dtype != dtype:
            raise ValueError(f"TorchTumorData.{name} must use runtime dtype {dtype}.")
        if value.device.type != runtime.device.type or (
            runtime.device.index is not None
            and value.device.index != runtime.device.index
        ):
            raise ValueError(
                f"TorchTumorData.{name} must be on runtime device "
                f"{runtime.device_name}."
            )

    for name in (
        "alt",
        "total",
        "nonalt",
        "phi_upper",
        "b_minus",
        "b_plus",
        "b_fixed",
    ):
        validate_field(name, dtype=runtime.dtype)
    validate_field("ambiguous", dtype=torch.bool)

    expected_observation_mask = data.count_observed is not None
    if (tensor_data.count_observed is not None) != expected_observation_mask:
        raise ValueError(
            "TorchTumorData.count_observed presence does not match TumorData."
        )
    if tensor_data.count_observed is not None:
        validate_field("count_observed", dtype=torch.bool)


def _as_loss_shape(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = mask
    while result.ndim < target.ndim:
        result = result.unsqueeze(-1)
    return result


def clip_probability_and_slope(
    beta: torch.Tensor, scale: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    linear = scale * beta
    prob = torch.clamp(linear, min=float(eps), max=float(1.0 - eps))
    interior = (linear > float(eps)) & (linear < float(1.0 - eps))
    slope = torch.where(interior, scale, torch.zeros_like(scale))
    return prob, slope


def _major_prior_log_tensors(
    major_prior: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    prior_tensor = torch.as_tensor(prior, dtype=dtype, device=device)
    return torch.log1p(-prior_tensor), torch.log(prior_tensor)


def state_log_kernel_grad_and_curvature(
    *,
    alt: torch.Tensor,
    nonalt: torch.Tensor,
    prob: torch.Tensor,
    slope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = slope * (alt / prob - nonalt / (1.0 - prob))
    curvature = torch.square(slope) * (
        alt / torch.square(prob) + nonalt / torch.square(1.0 - prob)
    )
    return grad, curvature


def mutation_region_loss_grid_torch(
    beta_grid: torch.Tensor,
    *,
    alt: torch.Tensor,
    total: torch.Tensor,
    b_minus: torch.Tensor,
    b_plus: torch.Tensor,
    b_fixed: torch.Tensor,
    ambiguous: torch.Tensor,
    major_prior: float,
    eps: float,
) -> torch.Tensor:
    beta = beta_grid
    nonalt = total - alt
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=beta.dtype,
        device=beta.device,
    )
    prob_fixed = torch.clamp(beta * b_fixed, min=float(eps), max=float(1.0 - eps))
    fixed_loss = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))

    prob_minus = torch.clamp(beta * b_minus, min=float(eps), max=float(1.0 - eps))
    prob_plus = torch.clamp(beta * b_plus, min=float(eps), max=float(1.0 - eps))
    log_minor = (
        alt * torch.log(prob_minus)
        + nonalt * torch.log1p(-prob_minus)
        + log_prior_minor
    )
    log_major = (
        alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    )
    amb_loss = -torch.logaddexp(log_minor, log_major)
    return torch.where(_as_loss_shape(ambiguous, beta), amb_loss, fixed_loss)


def mutation_region_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> TorchMutationRegionTerms:
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=phi.dtype,
        device=phi.device,
    )

    prob_fixed, slope_fixed = clip_probability_and_slope(phi, data.b_fixed, eps)
    loss_fixed = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))
    grad_fixed_state, curvature_fixed = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_fixed,
        slope=slope_fixed,
    )

    prob_minus, slope_minus = clip_probability_and_slope(phi, data.b_minus, eps)
    prob_plus, slope_plus = clip_probability_and_slope(phi, data.b_plus, eps)
    log_minor = (
        alt * torch.log(prob_minus)
        + nonalt * torch.log1p(-prob_minus)
        + log_prior_minor
    )
    log_major = (
        alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    )
    norm = torch.logaddexp(log_minor, log_major)
    gamma = torch.sigmoid(log_major - log_minor)
    grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_minus,
        slope=slope_minus,
    )
    grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_plus,
        slope=slope_plus,
    )
    amb_loss = -norm
    amb_grad = -((1.0 - gamma) * grad_minus + gamma * grad_plus)
    amb_curvature = (1.0 - gamma) * curvature_minus + gamma * curvature_plus

    loss = torch.where(amb, amb_loss, loss_fixed)
    grad = torch.where(amb, amb_grad, -grad_fixed_state)
    hess_upper = torch.where(amb, amb_curvature, curvature_fixed)
    gamma_major = torch.where(amb, gamma, torch.ones_like(gamma))

    if data.count_observed is not None:
        loss = torch.where(data.count_observed, loss, torch.zeros_like(loss))
        grad = torch.where(data.count_observed, grad, torch.zeros_like(grad))
        # A masked ambiguous entry contributes no count evidence, so its
        # posterior remains the configured prior.  Keeping a posterior derived
        # from the stored-but-masked counts would make raw solver summaries
        # disagree with the observed-data likelihood and the NumPy posterior
        # used by partition candidates.
        prior_major = torch.exp(log_prior_major)
        gamma_major = torch.where(
            data.count_observed,
            gamma_major,
            torch.where(amb, prior_major, torch.ones_like(gamma_major)),
        )

    hess_upper = torch.clamp(hess_upper, min=1e-8)
    if data.count_observed is not None:
        hess_upper = torch.where(
            data.count_observed, hess_upper, torch.zeros_like(hess_upper)
        )
    return TorchMutationRegionTerms(
        loss=loss,
        grad=grad,
        hess_upper=hess_upper,
        gamma_major=gamma_major,
    )


def em_surrogate_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    omega_major: torch.Tensor,
    major_prior: float,
    eps: float,
) -> TorchMutationRegionTerms:
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=phi.dtype,
        device=phi.device,
    )

    prob_fixed, slope_fixed = clip_probability_and_slope(phi, data.b_fixed, eps)
    loss_fixed = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))
    grad_fixed_state, curvature_fixed = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_fixed,
        slope=slope_fixed,
    )

    omega = torch.clamp(omega_major, min=0.0, max=1.0)
    prob_minus, slope_minus = clip_probability_and_slope(phi, data.b_minus, eps)
    prob_plus, slope_plus = clip_probability_and_slope(phi, data.b_plus, eps)
    log_minor = (
        alt * torch.log(prob_minus)
        + nonalt * torch.log1p(-prob_minus)
        + log_prior_minor
    )
    log_major = (
        alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    )
    grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_minus,
        slope=slope_minus,
    )
    grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_plus,
        slope=slope_plus,
    )

    loss_major = -log_major
    loss_minor = -log_minor
    entropy_offset = _binary_entropy_offset_torch(omega)
    amb_loss = (1.0 - omega) * loss_minor + omega * loss_major + entropy_offset
    amb_grad = -((1.0 - omega) * grad_minus + omega * grad_plus)
    amb_curvature = (1.0 - omega) * curvature_minus + omega * curvature_plus

    loss = torch.where(amb, amb_loss, loss_fixed)
    grad = torch.where(amb, amb_grad, -grad_fixed_state)
    hess_upper = torch.where(amb, amb_curvature, curvature_fixed)
    gamma_major = torch.where(amb, omega, torch.ones_like(omega))

    if data.count_observed is not None:
        loss = torch.where(data.count_observed, loss, torch.zeros_like(loss))
        grad = torch.where(data.count_observed, grad, torch.zeros_like(grad))

    hess_upper = torch.clamp(hess_upper, min=1e-8)
    if data.count_observed is not None:
        hess_upper = torch.where(
            data.count_observed, hess_upper, torch.zeros_like(hess_upper)
        )
    return TorchMutationRegionTerms(
        loss=loss,
        grad=grad,
        hess_upper=hess_upper,
        gamma_major=gamma_major,
    )


def pairwise_penalty_torch(
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    lambda_value = validate_lambda_value(lambda_value)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        return torch.zeros((), dtype=phi.dtype, device=phi.device)
    num_edges = int(edge_u.numel())
    num_regions = int(phi.shape[1])
    if (
        _edge_tensor_nbytes(
            num_edges=num_edges,
            num_regions=num_regions,
            dtype=phi.dtype,
        )
        <= DEFAULT_EDGE_WORK_BYTES
    ):
        diffs = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
        weighted_norm = torch.sum(edge_w * torch.linalg.norm(diffs, dim=1))
    else:
        chunk_size = _edge_chunk_size(
            num_edges=num_edges,
            num_regions=num_regions,
            dtype=phi.dtype,
        )
        weighted_norm = torch.zeros((), dtype=phi.dtype, device=phi.device)
        for edge_slice in _edge_slices(num_edges, chunk_size):
            diffs = graph_forward_edges(
                phi,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
            )
            weighted_norm = weighted_norm + torch.sum(
                edge_w[edge_slice] * torch.linalg.vector_norm(diffs, dim=1)
            )
    return (
        torch.as_tensor(float(lambda_value), dtype=phi.dtype, device=phi.device)
        * weighted_norm
    )


def objective_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    major_prior: float,
    eps: float,
) -> ObjectiveTerms:
    terms = mutation_region_terms_torch(data, phi, major_prior=major_prior, eps=eps)
    fit_loss = torch.sum(terms.loss)
    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    return ObjectiveTerms(
        fit=fit_loss,
        penalty=penalty,
        total=fit_loss + penalty,
        gamma_major=terms.gamma_major,
    )


def objective_value_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    major_prior: float,
    eps: float,
) -> tuple[float, float, float, torch.Tensor]:
    terms = objective_terms_torch(
        data,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        major_prior=major_prior,
        eps=eps,
    )
    return (
        float(terms.fit.item()),
        float(terms.penalty.item()),
        float(terms.total.item()),
        terms.gamma_major,
    )


def project_stationarity_cone_torch(
    total_grad: torch.Tensor,
    *,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Project a total gradient onto the box KKT stationarity cone.

    Boundary membership is intentionally exact.  A coordinate merely near a
    bound is still interior and therefore has the singleton stationarity cone
    ``{0}``.  Frozen coordinates have the full real line as their cone.
    """

    frozen = lower == upper
    lower_only = (phi == lower) & ~frozen
    upper_only = (phi == upper) & ~frozen

    projected = torch.zeros_like(total_grad)
    projected = torch.where(
        lower_only,
        torch.clamp(total_grad, min=0.0),
        projected,
    )
    projected = torch.where(
        upper_only,
        torch.clamp(total_grad, max=0.0),
        projected,
    )
    return torch.where(frozen, total_grad, projected)


def stationarity_residual_torch(
    *,
    total_grad: torch.Tensor,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    atol: float,
) -> torch.Tensor:
    del atol
    projected = torch.minimum(torch.maximum(phi - total_grad, lower), upper)
    return phi - projected


def graph_fusion_kkt_residual_from_grad_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    dual_scale: float = 1.0,
    edge_work_bytes: int | None = None,
) -> dict[str, float]:
    lambda_value = validate_lambda_value(lambda_value)
    dual_scale_value = float(dual_scale)
    if not np.isfinite(dual_scale_value) or dual_scale_value < 0.0:
        raise ValueError("dual_scale must be finite and nonnegative.")
    num_edges = int(edge_u.numel())
    num_regions = int(phi.shape[1])
    valid_dual = bool(
        dual_kkt is not None and tuple(dual_kkt.shape) == (num_edges, num_regions)
    )
    dual = None
    if valid_dual:
        dual = dual_kkt.to(dtype=phi.dtype, device=phi.device)
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=phi.dtype,
        work_bytes=edge_work_bytes,
    )

    adj = torch.zeros_like(phi)
    if num_edges > 0 and lambda_value > 0.0 and dual is not None:
        for edge_slice in _edge_slices(num_edges, chunk_size):
            dual_chunk = dual[edge_slice]
            if dual_scale_value != 1.0:
                dual_chunk = dual_scale_value * dual_chunk
            adj.index_add_(0, edge_u[edge_slice], dual_chunk)
            adj.index_add_(0, edge_v[edge_slice], dual_chunk, alpha=-1.0)
    total_grad = grad_smooth + adj
    stat = stationarity_residual_torch(
        total_grad=total_grad, phi=phi, lower=lower, upper=upper, atol=atol
    )
    smooth_gradient_norm = float(torch.linalg.norm(grad_smooth).item())
    fusion_adjustment_norm = float(torch.linalg.norm(adj).item())
    projected_stationarity_norm = float(torch.linalg.norm(stat).item())
    stationarity_normalizer = float(1.0 + smooth_gradient_norm + fusion_adjustment_norm)
    stationarity_residual = float(
        projected_stationarity_norm / max(stationarity_normalizer, 1e-300)
    )
    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    frozen = upper <= lower + float(atol)
    interior = ~(lower_active | upper_active | frozen)
    diagnostic_upper_active = upper_active & ~frozen
    diagnostic_lower_active = lower_active & ~upper_active & ~frozen
    box_violation = torch.maximum(
        torch.clamp(lower - phi, min=0.0), torch.clamp(phi - upper, min=0.0)
    )
    box_primal_violation = (
        float(torch.max(box_violation).item()) if box_violation.numel() else 0.0
    )
    box_scale = 1.0 + max(
        float(torch.max(torch.abs(lower)).item()) if lower.numel() else 0.0,
        float(torch.max(torch.abs(upper)).item()) if upper.numel() else 0.0,
    )
    box_residual = box_primal_violation / max(box_scale, 1e-300)

    if num_edges == 0 or lambda_value <= 0.0:
        edge_subgradient_residual = 0.0
        dual_ball_residual = 0.0
    else:
        # Proximal fixed-point residual: R_e = d_e - prox_{r_e*|.|_2}(d_e + y_e)
        # Stable form avoids cancellation and gives exact zeros for exact KKT:
        #   |v| >= r: R_e = -y_e + r_e * v / |v|    (d-v+r*v/|v| = -y+r*v/|v|)
        #   |v| <  r: R_e = d_e                       (prox maps v to 0)
        # This is zero iff y_e ∈ r_e * ∂|d_e|_2, for any d_e including near-zero.
        max_edge_residual = 0.0
        max_ball_residual = 0.0
        max_radius = 0.0
        for edge_slice in _edge_slices(num_edges, chunk_size):
            diff = graph_forward_edges(
                phi,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
            )
            radius = float(lambda_value) * edge_w[edge_slice]
            if dual is None:
                dual_chunk = None
                prox_input = diff
            else:
                dual_chunk = dual[edge_slice]
                if dual_scale_value != 1.0:
                    dual_chunk = dual_scale_value * dual_chunk
                prox_input = diff + dual_chunk
            prox_input_norm = torch.linalg.vector_norm(prox_input, dim=1)
            big = prox_input_norm >= radius
            safe_norm = prox_input_norm.clamp_min(1e-300)
            if dual_chunk is None:
                active_residual = radius[:, None] * prox_input / safe_norm[:, None]
                ball_residual = torch.zeros_like(radius)
            else:
                active_residual = (
                    -dual_chunk + radius[:, None] * prox_input / safe_norm[:, None]
                )
                ball_residual = torch.clamp(
                    torch.linalg.vector_norm(dual_chunk, dim=1) - radius,
                    min=0.0,
                )
            edge_residual = torch.where(
                big,
                torch.linalg.vector_norm(active_residual, dim=1),
                torch.linalg.vector_norm(diff, dim=1),
            )
            max_edge_residual = max(
                max_edge_residual,
                float(torch.max(edge_residual).item())
                if edge_residual.numel()
                else 0.0,
            )
            max_ball_residual = max(
                max_ball_residual,
                float(torch.max(ball_residual).item())
                if ball_residual.numel()
                else 0.0,
            )
            max_radius = max(
                max_radius,
                float(torch.max(radius).item()) if radius.numel() else 0.0,
            )

        denom = 1.0 + max_radius
        edge_subgradient_residual = float(max_edge_residual / denom)
        dual_ball_residual = float(max_ball_residual / denom)

    return {
        "stationarity_residual": stationarity_residual,
        "projected_stationarity_residual": stationarity_residual,
        "projected_stationarity_norm": projected_stationarity_norm,
        "stationarity_normalizer": stationarity_normalizer,
        "smooth_gradient_norm": smooth_gradient_norm,
        "fusion_adjustment_norm": fusion_adjustment_norm,
        "edge_subgradient_residual": edge_subgradient_residual,
        "dual_ball_residual": dual_ball_residual,
        "box_primal_violation": box_primal_violation,
        "num_interior_coordinates": int(torch.sum(interior).item()),
        "num_lower_active_coordinates": int(torch.sum(diagnostic_lower_active).item()),
        "num_upper_active_coordinates": int(torch.sum(diagnostic_upper_active).item()),
        "num_frozen_coordinates": int(torch.sum(frozen).item()),
        "box_residual": float(box_residual),
        "kkt_residual": max(
            stationarity_residual,
            edge_subgradient_residual,
            dual_ball_residual,
            float(box_residual),
        ),
    }


def _refine_graph_fusion_dual_certificate_streaming_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int,
    before_diag: dict[str, float],
    edge_work_bytes: int | None,
) -> dict[str, object]:
    """Memory-bounded counterpart of the final dual-certificate refinement."""
    num_edges = int(edge_u.numel())
    num_regions = int(phi.shape[1])
    num_nodes = int(phi.shape[0])
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=phi.dtype,
        work_bytes=edge_work_bytes,
    )
    incoming_valid = bool(
        dual_kkt is not None and tuple(dual_kkt.shape) == (num_edges, num_regions)
    )
    incoming = (
        dual_kkt.to(dtype=phi.dtype, device=phi.device) if incoming_valid else None
    )

    nonzero_edges = 0
    for edge_slice in _edge_slices(num_edges, chunk_size):
        diff = graph_forward_edges(
            phi,
            edge_u=edge_u[edge_slice],
            edge_v=edge_v[edge_slice],
        )
        nonzero_edges += int(
            torch.sum(torch.linalg.vector_norm(diff, dim=1) > float(atol)).item()
        )
    fused_edges = num_edges - nonzero_edges

    dual = torch.zeros(
        (num_edges, num_regions),
        dtype=phi.dtype,
        device=phi.device,
    )
    for edge_slice in _edge_slices(num_edges, chunk_size):
        diff = graph_forward_edges(
            phi,
            edge_u=edge_u[edge_slice],
            edge_v=edge_v[edge_slice],
        )
        diff_norm = torch.linalg.vector_norm(diff, dim=1)
        active = diff_norm > float(atol)
        radius = float(lambda_value) * edge_w[edge_slice]
        analytic_chunk = (
            radius[:, None] * diff / diff_norm[:, None].clamp_min(float(atol))
        )
        if incoming is None:
            dual[edge_slice].copy_(
                torch.where(
                    active[:, None],
                    analytic_chunk,
                    torch.zeros_like(analytic_chunk),
                )
            )
        else:
            projected_incoming = project_dual_ball(
                incoming[edge_slice],
                radius,
            )
            dual[edge_slice].copy_(
                torch.where(
                    active[:, None],
                    analytic_chunk,
                    projected_incoming,
                )
            )

    analytic_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    best_diag = analytic_diag
    best_residual = float(analytic_diag["kkt_residual"])
    best_source = "analytic"
    best_dual: torch.Tensor = dual.clone()
    if incoming is not None:
        incoming_residual = float(before_diag["kkt_residual"])
        if np.isfinite(incoming_residual) and incoming_residual <= best_residual:
            best_diag = before_diag
            best_residual = incoming_residual
            best_source = "incoming"
            best_dual = incoming

    if fused_edges > 0:
        degree = torch.bincount(edge_u, minlength=num_nodes) + torch.bincount(
            edge_v,
            minlength=num_nodes,
        )
        step = 0.25 / max(float(torch.max(degree).item()), 1.0)
        for _ in range(max(int(max_iter), 1)):
            adj = torch.zeros_like(phi)
            for edge_slice in _edge_slices(num_edges, chunk_size):
                dual_chunk = dual[edge_slice]
                adj.index_add_(0, edge_u[edge_slice], dual_chunk)
                adj.index_add_(
                    0,
                    edge_v[edge_slice],
                    dual_chunk,
                    alpha=-1.0,
                )
            stat = stationarity_residual_torch(
                total_grad=grad_smooth + adj,
                phi=phi,
                lower=lower,
                upper=upper,
                atol=atol,
            )
            for edge_slice in _edge_slices(num_edges, chunk_size):
                diff = graph_forward_edges(
                    phi,
                    edge_u=edge_u[edge_slice],
                    edge_v=edge_v[edge_slice],
                )
                fused = torch.linalg.vector_norm(diff, dim=1) <= float(atol)
                if not bool(torch.any(fused).item()):
                    continue
                stat_diff = graph_forward_edges(
                    stat,
                    edge_u=edge_u[edge_slice],
                    edge_v=edge_v[edge_slice],
                )
                radius = float(lambda_value) * edge_w[edge_slice]
                projected = project_dual_ball(
                    dual[edge_slice] - float(step) * stat_diff,
                    radius,
                )
                dual[edge_slice].copy_(
                    torch.where(
                        fused[:, None],
                        projected,
                        dual[edge_slice],
                    )
                )
            diag = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                grad_smooth=grad_smooth,
                dual_kkt=dual,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                atol=atol,
                edge_work_bytes=edge_work_bytes,
            )
            residual = float(diag["kkt_residual"])
            if residual < best_residual:
                best_residual = residual
                best_diag = diag
                if best_source == "incoming":
                    best_dual = dual.clone()
                else:
                    best_dual.copy_(dual)
                best_source = "refined"

    if best_source == "incoming":
        status = "input_dual_retained"
    elif fused_edges > 0:
        status = "refined_fused_edge_dual"
    else:
        status = "analytic_nonfused_dual"
    return {
        "dual": best_dual,
        "diag": best_diag,
        "status": status,
        "dual_refined": bool(best_source != "incoming"),
        "fused_edges": int(fused_edges),
        "nonzero_edges": int(nonzero_edges),
        "stationarity_before": float(before_diag["stationarity_residual"]),
        "stationarity_after": float(best_diag["stationarity_residual"]),
    }


def refine_graph_fusion_dual_certificate_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int = 96,
    edge_work_bytes: int | None = None,
) -> dict[str, object]:
    lambda_value = validate_lambda_value(lambda_value)
    before_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual_kkt,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        dual = torch.zeros(
            (int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device
        )
        after_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            dual_kkt=dual,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=atol,
            edge_work_bytes=edge_work_bytes,
        )
        return {
            "dual": dual,
            "diag": after_diag,
            "status": "zero_penalty_no_dual_needed",
            "dual_refined": False,
            "fused_edges": 0,
            "nonzero_edges": 0,
            "stationarity_before": float(before_diag["stationarity_residual"]),
            "stationarity_after": float(after_diag["stationarity_residual"]),
        }

    budget = (
        DEFAULT_EDGE_WORK_BYTES if edge_work_bytes is None else int(edge_work_bytes)
    )
    if (
        _edge_tensor_nbytes(
            num_edges=int(edge_u.numel()),
            num_regions=int(phi.shape[1]),
            dtype=phi.dtype,
        )
        > budget
    ):
        return _refine_graph_fusion_dual_certificate_streaming_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            dual_kkt=dual_kkt,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=atol,
            max_iter=max_iter,
            before_diag=before_diag,
            edge_work_bytes=budget,
        )

    diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
    diff_norm = torch.linalg.norm(diff, dim=1)
    radius = float(lambda_value) * edge_w
    active = diff_norm > float(atol)
    fused = ~active
    dual = torch.zeros(
        (int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device
    )
    if torch.any(active):
        dual[active] = (
            radius[active, None]
            * diff[active]
            / diff_norm[active, None].clamp_min(float(atol))
        )
    if (
        torch.any(fused)
        and dual_kkt is not None
        and tuple(dual_kkt.shape) == tuple(dual.shape)
    ):
        dual[fused] = dual_kkt.to(dtype=phi.dtype, device=phi.device)[fused]
        fused_radius = radius[fused]
        dual[fused] = project_dual_ball(dual[fused], fused_radius)

    analytic_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
    )
    best_dual = dual.clone()
    best_diag = analytic_diag
    best_residual = float(analytic_diag["kkt_residual"])
    best_source = "analytic"

    # Reconstructing the analytic active-edge subgradient can improve edge
    # feasibility while worsening stationarity at a finite-accuracy primal
    # iterate.  A certificate-refinement routine must be monotone in the full
    # KKT residual, so retain the incoming actual ADMM multiplier whenever it
    # is the stronger certificate.
    if dual_kkt is not None and tuple(dual_kkt.shape) == tuple(dual.shape):
        incoming_residual = float(before_diag["kkt_residual"])
        if np.isfinite(incoming_residual) and incoming_residual <= best_residual:
            best_dual = dual_kkt.to(dtype=phi.dtype, device=phi.device).clone()
            best_diag = before_diag
            best_residual = incoming_residual
            best_source = "incoming"
    if torch.any(fused):
        degree = torch.bincount(
            torch.cat([edge_u, edge_v]),
            minlength=int(phi.shape[0]),
        ).max()
        step = 0.25 / max(float(degree.item()), 1.0)
        for _ in range(max(int(max_iter), 1)):
            adj = graph_adjoint_edges(
                dual,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=int(phi.shape[0]),
            )
            total_grad = grad_smooth + adj
            stat = stationarity_residual_torch(
                total_grad=total_grad,
                phi=phi,
                lower=lower,
                upper=upper,
                atol=atol,
            )
            dual[fused] = (
                dual[fused]
                - float(step)
                * (graph_forward_edges(stat, edge_u=edge_u, edge_v=edge_v)[fused])
            )
            fused_radius = radius[fused]
            dual[fused] = project_dual_ball(dual[fused], fused_radius)
            diag = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                grad_smooth=grad_smooth,
                dual_kkt=dual,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                atol=atol,
            )
            residual = float(diag["kkt_residual"])
            if residual < best_residual:
                best_residual = residual
                best_diag = diag
                best_dual = dual.clone()
                best_source = "refined"

    if best_source == "incoming":
        status = "input_dual_retained"
    elif torch.any(fused):
        status = "refined_fused_edge_dual"
    else:
        status = "analytic_nonfused_dual"

    return {
        "dual": best_dual,
        "diag": best_diag,
        "status": status,
        "dual_refined": bool(best_source != "incoming"),
        "fused_edges": int(torch.sum(fused).item()),
        "nonzero_edges": int(torch.sum(active).item()),
        "stationarity_before": float(before_diag["stationarity_residual"]),
        "stationarity_after": float(best_diag["stationarity_residual"]),
    }


def inner_kkt_residual_torch(
    *,
    phi: torch.Tensor,
    dual: torch.Tensor,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    atol: float,
    dual_scale: float = 1.0,
    edge_work_bytes: int | None = None,
    diagnostics_out: dict[str, float | int] | None = None,
) -> float:
    lambda_value = validate_lambda_value(lambda_value)
    diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=h * (phi - U),
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
        dual_scale=dual_scale,
        edge_work_bytes=edge_work_bytes,
    )
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(diag)
    return float(diag["kkt_residual"])


def solve_majorized_subproblem_pdhg_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    degree_bound: int,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
    tau_node: torch.Tensor | None = None,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(
        torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower),
        upper,
    )
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(
            total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol
        )
        residual = float(
            (torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item()
        )
        empty_dual = torch.zeros(
            (0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device
        )
        # This branch is a closed-form box projection; no PDHG iteration ran.
        return projected, empty_dual, empty_dual, 0, residual <= tol, residual

    if dual_start is not None and tuple(dual_start.shape) == (
        int(edge_u.numel()),
        int(phi.shape[1]),
    ):
        dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
    else:
        dual = torch.zeros(
            (int(edge_u.numel()), int(phi.shape[1])),
            dtype=runtime.dtype,
            device=runtime.device,
        )
    bar = phi.clone()
    del degree_bound
    if tau_node is None:
        node_degree = torch.bincount(
            torch.cat([edge_u, edge_v]),
            minlength=int(num_mutations),
        ).to(dtype=runtime.dtype, device=runtime.device)
        tau_node_t = (PDHG_PRECONDITIONER_ETA / node_degree.clamp_min(1.0))[:, None]
    else:
        tau_node_t = tau_node.to(dtype=runtime.dtype, device=runtime.device)
        if tau_node_t.ndim == 1:
            tau_node_t = tau_node_t[:, None]
        expected_shape = (int(num_mutations), 1)
        if tuple(tau_node_t.shape) != expected_shape:
            raise ValueError(f"tau_node must have shape {expected_shape}.")
    sigma_edge = PDHG_PRECONDITIONER_ETA / 2.0
    radius = float(lambda_value) * edge_w

    converged = False
    iterations = 0
    last_residual = np.inf
    actual_max_iter = max(int(max_iter), 10)
    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        edge_diff = graph_forward_edges(bar, edge_u=edge_u, edge_v=edge_v)
        dual_trial = dual + sigma_edge * edge_diff
        dual_new = project_dual_ball(dual_trial, radius)

        adj = graph_adjoint_edges(
            dual_new,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
        primal_base = phi - tau_node_t * adj
        phi_new = (primal_base + tau_node_t * h * U) / (1.0 + tau_node_t * h)
        phi_new = torch.minimum(torch.maximum(phi_new, lower), upper)
        bar = phi_new + (phi_new - phi)

        audit_due = _inner_kkt_audit_due(
            iteration=iterations,
            max_iter=actual_max_iter,
            kkt_check_every=kkt_check_every,
            cheap_converged=False,
        )
        if audit_due:
            primal_delta = float(
                (
                    torch.linalg.norm(phi_new - phi) / (1.0 + torch.linalg.norm(phi))
                ).item()
            )
            dual_delta = float(
                (
                    torch.linalg.norm(dual_new - dual) / (1.0 + torch.linalg.norm(dual))
                ).item()
            )
        phi = phi_new
        dual = dual_new

        if audit_due:
            cheap_converged = bool(primal_delta <= tol and dual_delta <= tol)
            last_residual = inner_kkt_residual_torch(
                phi=phi,
                dual=dual,
                U=U,
                h=h,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                atol=tol,
            )
            if cheap_converged and last_residual <= 5.0 * tol:
                converged = True
                break

    return phi, dual, dual, iterations, converged, float(last_residual)


def _complete_graph_isotropic_box_qp_torch(
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho: float,
    q: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    num_mutations = int(U.shape[0])
    rho_t = torch.as_tensor(float(rho), dtype=U.dtype, device=U.device)
    rhs = h * U + rho_t * q
    denom = h + rho_t * float(num_mutations)

    lo = torch.sum(lower, dim=0)
    hi = torch.sum(upper, dim=0)
    mid = 0.5 * (lo + hi)
    for _ in range(max(int(max_iter), 16)):
        mid = 0.5 * (lo + hi)
        x_mid = torch.minimum(
            torch.maximum((rhs + rho_t * mid.unsqueeze(0)) / denom, lower),
            upper,
        )
        residual = torch.sum(x_mid, dim=0) - mid
        move_right = residual > 0.0
        lo = torch.where(move_right, mid, lo)
        hi = torch.where(move_right, hi, mid)

    mid = 0.5 * (lo + hi)
    return torch.minimum(
        torch.maximum((rhs + rho_t * mid.unsqueeze(0)) / denom, lower),
        upper,
    )


def _solve_majorized_subproblem_alm_dense_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
    dual_start_is_actual: bool = False,
    spectral_rho: bool = False,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
    box_phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    box_max_iter: int = DEFAULT_BOX_MAX_ITER,
    diagnostics_out: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(
        torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower),
        upper,
    )
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(
            total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol
        )
        residual = float(
            (torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item()
        )
        empty_dual = torch.zeros(
            (0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device
        )
        # This branch is a closed-form box projection; no ADMM iteration ran.
        return projected, empty_dual, empty_dual, 0, residual <= tol, residual

    median_h = torch.median(h)
    if spectral_rho:
        # For the complete graph, the nonzero spectrum of D.T @ D is M.
        # Balancing H against rho * D.T @ D therefore uses median(H) / M,
        # rather than treating the incidence operator as norm one.  Wide
        # numerical bounds guard only overflow/underflow; they do not define a
        # lambda path or alter the penalized objective.
        rho = float(
            torch.clamp(
                median_h / max(float(num_mutations), 1.0),
                min=1e-8,
                max=1e8,
            ).item()
        )
    else:
        rho = float(torch.clamp(median_h, min=1e-3, max=1e3).item())
    radius = float(lambda_value) * edge_w
    if dual_start is not None and tuple(dual_start.shape) == (
        int(edge_u.numel()),
        int(phi.shape[1]),
    ):
        initial_dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
        if bool(dual_start_is_actual):
            initial_dual = project_dual_ball(initial_dual, radius)
            scaled_dual = initial_dual / rho
        else:
            scaled_dual = initial_dual
    else:
        scaled_dual = torch.zeros(
            (int(edge_u.numel()), int(phi.shape[1])),
            dtype=runtime.dtype,
            device=runtime.device,
        )

    shrink_radius = radius / rho

    converged = False
    iterations = 0
    last_residual = np.inf
    actual_dual = rho * scaled_dual
    actual_max_iter = max(int(max_iter), 10)
    box_iter = _box_qp_sweeps_for_atol(box_phi_atol, max_iter=box_max_iter)
    z_previous = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)

    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        edge_diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
        z_argument = edge_diff + scaled_dual
        z_norm = torch.linalg.norm(z_argument, dim=1, keepdim=True)
        shrink = torch.clamp(
            1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12), min=0.0
        )
        z_new = shrink * z_argument

        rhs_edge = z_new - scaled_dual
        q = graph_adjoint_edges(
            rhs_edge,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
        phi_new = _complete_graph_isotropic_box_qp_torch(
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            rho=rho,
            q=q,
            max_iter=box_iter,
        )

        edge_diff_new = graph_forward_edges(phi_new, edge_u=edge_u, edge_v=edge_v)
        primal_residual = edge_diff_new - z_new
        scaled_dual_new = scaled_dual + primal_residual

        actual_dual = rho * scaled_dual_new

        if spectral_rho and iterations % 10 == 0:
            primal_norm = float(torch.linalg.norm(primal_residual).item())
            z_delta = z_new - z_previous
            dual_residual_node = graph_adjoint_edges(
                float(rho) * z_delta,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=int(phi.shape[0]),
            )
            dual_norm = float(torch.linalg.norm(dual_residual_node).item())
            # Compare scale-free residuals.  Multiplying the whole objective by
            # c also multiplies rho and the conventional dual residual by c,
            # while leaving the primal residual unchanged.  Removing that rho
            # factor keeps the adaptive-rho decisions (and ADMM path)
            # equivariant to objective units.
            dual_balance_norm = dual_norm / max(abs(float(rho)), 1e-300)
            next_rho = float(rho)
            if np.isfinite(primal_norm) and np.isfinite(dual_balance_norm):
                if primal_norm > 10.0 * max(dual_balance_norm, 1e-300):
                    next_rho = min(2.0 * float(rho), 1e8)
                elif dual_balance_norm > 10.0 * max(primal_norm, 1e-300):
                    next_rho = max(0.5 * float(rho), 1e-8)
            if next_rho != float(rho):
                # Preserve y = rho*u exactly when changing the scaled-dual
                # parameterization, then update the group-shrinkage radius.
                scaled_dual_new = scaled_dual_new * (float(rho) / next_rho)
                rho = float(next_rho)
                shrink_radius = radius / float(rho)
                actual_dual = float(rho) * scaled_dual_new

        audit_due = _inner_kkt_audit_due(
            iteration=iterations,
            max_iter=actual_max_iter,
            kkt_check_every=kkt_check_every,
            cheap_converged=False,
        )

        phi = phi_new
        scaled_dual = scaled_dual_new
        z_previous = z_new

        if audit_due:
            last_residual = inner_kkt_residual_torch(
                phi=phi,
                dual=actual_dual,
                U=U,
                h=h,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                atol=tol,
                diagnostics_out=diagnostics_out,
            )
            # The audited residual is the full box-QP KKT certificate
            # (stationarity, edge subgradient, dual ball, and box feasibility).
            # Once it is below tolerance, separate iterate-delta heuristics are
            # not an additional optimality requirement and must not force ADMM
            # to exhaust its budget.
            # Allow only the numerical floor contributed by the independently
            # budgeted box solve; this is intentionally much tighter than the
            # downstream 5*tol candidate-certification gate.
            kkt_stop_tol = float(tol) + 0.25 * min(float(box_phi_atol), float(tol))
            if last_residual <= kkt_stop_tol:
                converged = True
                break

    return phi, scaled_dual, actual_dual, iterations, converged, float(last_residual)


def _solve_majorized_subproblem_alm_streaming_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
    dual_start_is_actual: bool = False,
    spectral_rho: bool = False,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
    box_phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    box_max_iter: int = DEFAULT_BOX_MAX_ITER,
    edge_work_bytes: int | None = None,
    diagnostics_out: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    """Exact complete-graph ADMM with bounded edgewise working storage."""
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(
        torch.maximum(
            phi_start.to(dtype=runtime.dtype, device=runtime.device),
            lower,
        ),
        upper,
    )
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(
            total_grad=total_grad,
            phi=projected,
            lower=lower,
            upper=upper,
            atol=tol,
        )
        residual = float(
            (torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item()
        )
        empty_dual = torch.zeros(
            (0, phi.shape[1]),
            dtype=runtime.dtype,
            device=runtime.device,
        )
        return projected, empty_dual, empty_dual, 0, residual <= tol, residual

    num_edges = int(edge_u.numel())
    num_regions = int(phi.shape[1])
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=runtime.dtype,
        work_bytes=edge_work_bytes,
    )
    median_h = torch.median(h)
    if spectral_rho:
        rho = float(
            torch.clamp(
                median_h / max(float(num_mutations), 1.0),
                min=1e-8,
                max=1e8,
            ).item()
        )
    else:
        rho = float(torch.clamp(median_h, min=1e-3, max=1e3).item())

    expected_dual_shape = (num_edges, num_regions)
    scaled_dual = torch.empty(
        expected_dual_shape,
        dtype=runtime.dtype,
        device=runtime.device,
    )
    if dual_start is None or tuple(dual_start.shape) != expected_dual_shape:
        scaled_dual.zero_()
    else:
        for edge_slice in _edge_slices(num_edges, chunk_size):
            initial_chunk = dual_start[edge_slice].to(
                dtype=runtime.dtype,
                device=runtime.device,
            )
            if dual_start_is_actual:
                radius = float(lambda_value) * edge_w[edge_slice]
                norm = torch.linalg.vector_norm(
                    initial_chunk,
                    dim=1,
                    keepdim=True,
                )
                projection_scale = torch.maximum(
                    torch.ones_like(norm),
                    norm / radius[:, None].clamp_min(torch.finfo(runtime.dtype).tiny),
                )
                scaled_dual[edge_slice].copy_(
                    initial_chunk / projection_scale / float(rho)
                )
            else:
                scaled_dual[edge_slice].copy_(initial_chunk)

    # z_state stores the previous/current ADMM split variable.  Together with
    # scaled_dual it is the only second persistent edge-by-region state.
    z_state = torch.empty_like(scaled_dual)
    for edge_slice in _edge_slices(num_edges, chunk_size):
        z_state[edge_slice].copy_(
            graph_forward_edges(
                phi,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
            )
        )

    converged = False
    iterations = 0
    last_residual = np.inf
    actual_max_iter = max(int(max_iter), 10)
    box_iter = _box_qp_sweeps_for_atol(
        box_phi_atol,
        max_iter=box_max_iter,
    )

    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        rho_update_due = bool(spectral_rho and iterations % 10 == 0)
        q = torch.zeros_like(phi)
        dual_residual_node = torch.zeros_like(phi) if rho_update_due else None

        # z update and D.T(z-u), streamed over complete-graph edges.
        for edge_slice in _edge_slices(num_edges, chunk_size):
            z_new = graph_forward_edges(
                phi,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
            )
            z_new.add_(scaled_dual[edge_slice])
            z_norm = torch.linalg.vector_norm(z_new, dim=1, keepdim=True)
            shrink_radius = float(lambda_value) * edge_w[edge_slice] / float(rho)
            shrink = torch.clamp(
                1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12),
                min=0.0,
            )
            z_new.mul_(shrink)

            if dual_residual_node is not None:
                z_delta = z_new - z_state[edge_slice]
                z_delta.mul_(float(rho))
                dual_residual_node.index_add_(
                    0,
                    edge_u[edge_slice],
                    z_delta,
                )
                dual_residual_node.index_add_(
                    0,
                    edge_v[edge_slice],
                    z_delta,
                    alpha=-1.0,
                )

            z_state[edge_slice].copy_(z_new)
            z_new.sub_(scaled_dual[edge_slice])
            q.index_add_(0, edge_u[edge_slice], z_new)
            q.index_add_(0, edge_v[edge_slice], z_new, alpha=-1.0)

        phi_new = _complete_graph_isotropic_box_qp_torch(
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            rho=rho,
            q=q,
            max_iter=box_iter,
        )

        primal_sum_squares = torch.zeros(
            (),
            dtype=runtime.dtype,
            device=runtime.device,
        )
        for edge_slice in _edge_slices(num_edges, chunk_size):
            primal_residual = graph_forward_edges(
                phi_new,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
            )
            primal_residual.sub_(z_state[edge_slice])
            primal_sum_squares.add_(
                torch.dot(
                    primal_residual.reshape(-1),
                    primal_residual.reshape(-1),
                )
            )
            scaled_dual[edge_slice].add_(primal_residual)

        if dual_residual_node is not None:
            primal_norm = float(torch.sqrt(primal_sum_squares).item())
            dual_norm = float(torch.linalg.vector_norm(dual_residual_node).item())
            dual_balance_norm = dual_norm / max(abs(float(rho)), 1e-300)
            next_rho = float(rho)
            if np.isfinite(primal_norm) and np.isfinite(dual_balance_norm):
                if primal_norm > 10.0 * max(dual_balance_norm, 1e-300):
                    next_rho = min(2.0 * float(rho), 1e8)
                elif dual_balance_norm > 10.0 * max(primal_norm, 1e-300):
                    next_rho = max(0.5 * float(rho), 1e-8)
            if next_rho != float(rho):
                scaled_dual.mul_(float(rho) / next_rho)
                rho = float(next_rho)

        phi = phi_new
        audit_due = _inner_kkt_audit_due(
            iteration=iterations,
            max_iter=actual_max_iter,
            kkt_check_every=kkt_check_every,
            cheap_converged=False,
        )
        if audit_due:
            last_residual = inner_kkt_residual_torch(
                phi=phi,
                dual=scaled_dual,
                U=U,
                h=h,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                atol=tol,
                dual_scale=rho,
                edge_work_bytes=edge_work_bytes,
                diagnostics_out=diagnostics_out,
            )
            kkt_stop_tol = float(tol) + 0.25 * min(
                float(box_phi_atol),
                float(tol),
            )
            if last_residual <= kkt_stop_tol:
                converged = True
                break

    # The caller needs both scaled u (for the low-level API) and actual y for
    # cross-subproblem warm starts.  Release z first so its allocation can be
    # reused when materializing y.
    del z_state
    actual_dual = float(rho) * scaled_dual
    return (
        phi,
        scaled_dual,
        actual_dual,
        iterations,
        converged,
        float(last_residual),
    )


def solve_majorized_subproblem_alm_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
    dual_start_is_actual: bool = False,
    spectral_rho: bool = False,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
    box_phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    box_max_iter: int = DEFAULT_BOX_MAX_ITER,
    edge_work_bytes: int | None = None,
    diagnostics_out: dict[str, float | int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    """Solve the complete-graph group-fusion subproblem by scaled-dual ADMM.

    Small problems retain the historical dense implementation exactly.  Once
    one edge-by-region tensor exceeds the work budget, the mathematically
    identical streamed implementation bounds all additional edgewise storage.
    """
    budget = (
        DEFAULT_EDGE_WORK_BYTES if edge_work_bytes is None else int(edge_work_bytes)
    )
    # Validate even for an empty graph so bad explicit configuration is never
    # silently accepted by the closed-form branch.
    _edge_chunk_size(
        num_edges=int(edge_u.numel()),
        num_regions=int(phi_start.shape[1]),
        dtype=runtime.dtype,
        work_bytes=budget,
    )
    use_streaming = bool(
        _edge_tensor_nbytes(
            num_edges=int(edge_u.numel()),
            num_regions=int(phi_start.shape[1]),
            dtype=runtime.dtype,
        )
        > budget
    )
    common = dict(
        runtime=runtime,
        num_mutations=num_mutations,
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        lambda_value=lambda_value,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        tol=tol,
        max_iter=max_iter,
        phi_start=phi_start,
        dual_start=dual_start,
        dual_start_is_actual=dual_start_is_actual,
        spectral_rho=spectral_rho,
        kkt_check_every=kkt_check_every,
        box_phi_atol=box_phi_atol,
        box_max_iter=box_max_iter,
        diagnostics_out=diagnostics_out,
    )
    if not use_streaming:
        return _solve_majorized_subproblem_alm_dense_torch(**common)
    return _solve_majorized_subproblem_alm_streaming_torch(
        **common,
        edge_work_bytes=budget,
    )
