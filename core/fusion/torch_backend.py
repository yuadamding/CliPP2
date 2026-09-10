from __future__ import annotations

import warnings

import numpy as np
import torch

from ...config import normalize_runtime_dtype
from .graph_ops import (
    DETERMINISTIC_COMPLETE_ADJOINT_MAX_BYTES,
    PDHG_PRECONDITIONER_ETA,
    graph_adjoint_edges,
    graph_forward_edges,
    project_dual_ball,
)
from .types import KKTDiagnostics, TorchRuntime


DEFAULT_INNER_KKT_CHECK_EVERY = 8
DEFAULT_BOX_PHI_ATOL = 1e-8
DEFAULT_BOX_MAX_ITER = 32
# Maximum size of one edge-by-region work tensor.  Complete-graph ADMM keeps
# its two mathematical edge states, but all additional edge work is streamed
# in chunks bounded by this value.
DEFAULT_EDGE_WORK_BYTES = 64 * 1024 * 1024
_CERTIFICATE_KKT_ATOL_SCALE = 5.0
_CERTIFICATE_PLATEAU_PATIENCE = 8
_CERTIFICATE_MOVING_PLATEAU_PATIENCE = 16
_CERTIFICATE_PLATEAU_ATOL_SCALE = 0.01
_CERTIFICATE_PLATEAU_EPS_SCALE = 32.0

_DTYPE_TO_NAME = {
    torch.float32: "float32",
    torch.float64: "float64",
}


class CudaUnavailableError(RuntimeError):
    """A valid CUDA request cannot run because CUDA is unavailable."""


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
        np.array(start, copy=True), dtype=runtime.dtype, device=runtime.device
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


def graph_adjoint_edges_in_dtype(
    dual: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
    device: torch.device,
    scale: float = 1.0,
    edge_work_bytes: int | None = None,
) -> torch.Tensor:
    """Apply the graph adjoint with bounded cross-precision work storage."""

    dual_scale = float(scale)
    if not np.isfinite(dual_scale):
        raise ValueError("dual adjoint scale must be finite.")
    source = dual.to(device=device)
    if source.dtype == dtype and dual_scale == 1.0:
        return graph_adjoint_edges(
            source,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(num_nodes),
        )
    num_edges = int(edge_u.numel())
    num_regions = int(source.shape[1])
    complete_edge_count = int(num_nodes) * max(int(num_nodes) - 1, 0) // 2
    dense_workspace_bytes = (
        int(num_nodes)
        * int(num_nodes)
        * max(num_regions, 1)
        * int(torch.empty((), dtype=dtype).element_size())
    )
    if (
        device.type == "cuda"
        and num_edges == complete_edge_count
        and dense_workspace_bytes <= DETERMINISTIC_COMPLETE_ADJOINT_MAX_BYTES
    ):
        promoted = source.to(dtype=dtype)
        if dual_scale != 1.0:
            promoted = dual_scale * promoted
        return graph_adjoint_edges(
            promoted,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(num_nodes),
        )
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=dtype,
        work_bytes=edge_work_bytes,
    )
    adj = torch.zeros((int(num_nodes), num_regions), dtype=dtype, device=device)
    for edge_slice in _edge_slices(num_edges, chunk_size):
        chunk = source[edge_slice].to(dtype=dtype)
        if dual_scale != 1.0:
            chunk = dual_scale * chunk
        adj.index_add_(0, edge_u[edge_slice], chunk)
        adj.index_add_(0, edge_v[edge_slice], chunk, alpha=-1.0)
    return adj


def _graph_edge_activity_counts_torch(
    *,
    phi: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    atol: float,
    edge_work_bytes: int | None,
) -> tuple[int, int]:
    num_edges = int(edge_u.numel())
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
        work_bytes=edge_work_bytes,
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
    return num_edges - nonzero_edges, nonzero_edges


def _update_certificate_refinement_plateau(
    *,
    anchor_residual: float,
    best_residual: float,
    mapping_delta: float,
    stalled_iterations: int,
    atol: float,
    dtype: torch.dtype,
) -> tuple[float, int, bool]:
    """Stop only after both residual progress and dual motion have stalled."""
    scale = max(1.0, abs(float(anchor_residual)))
    progress_floor = max(
        _CERTIFICATE_PLATEAU_ATOL_SCALE * float(atol),
        _CERTIFICATE_PLATEAU_EPS_SCALE * float(torch.finfo(dtype).eps) * scale,
    )
    if np.isfinite(best_residual) and (
        not np.isfinite(anchor_residual)
        or float(anchor_residual) - float(best_residual) > progress_floor
    ):
        return float(best_residual), 0, False
    stalled = int(stalled_iterations) + 1
    mapping_stalled = bool(
        np.isfinite(mapping_delta) and float(mapping_delta) <= progress_floor
    )
    patience = (
        _CERTIFICATE_PLATEAU_PATIENCE
        if mapping_stalled
        else _CERTIFICATE_MOVING_PLATEAU_PATIENCE
    )
    return float(anchor_residual), stalled, stalled >= patience


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


def resolve_runtime(device: str | None, *, dtype: str | None = None) -> TorchRuntime:
    """Resolve a device/dtype string pair into a TorchRuntime (the single place
    that maps strings to torch.device/torch.dtype).

    Determinism note: CUDA fits are not bit-reproducible run-to-run (e.g. the
    float index_add_ in graph_ops.graph_adjoint_edges is nondeterministic on GPU),
    and CPU vs GPU labels can differ near lambda-path decision boundaries. For
    reproducible GPU runs, enable torch.use_deterministic_algorithms(True) and set
    CUBLAS_WORKSPACE_CONFIG before fitting.
    """
    requested_dtype = normalize_runtime_dtype(dtype)
    runtime_dtype = {"float32": torch.float32, "float64": torch.float64}[requested_dtype]
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        runtime_device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Unknown runtime device: {device!r}") from exc
    if runtime_device.type not in {"cpu", "cuda"}:
        raise ValueError("Runtime device must be cpu, cuda, or auto.")
    if runtime_device.type == "cuda":
        if not torch.cuda.is_available():
            raise CudaUnavailableError(
                f"Requested Torch device {device!r}, but CUDA is not available. "
                "Use device='cpu' or device='auto' to permit CPU execution."
            )
        device_index = (
            int(torch.cuda.current_device())
            if runtime_device.index is None
            else int(runtime_device.index)
        )
        if not 0 <= device_index < int(torch.cuda.device_count()):
            raise ValueError(f"CUDA device index is unavailable: {device_index}")
        runtime_device = torch.device("cuda", device_index)
    else:
        runtime_device = torch.device("cpu")
    device_name = str(runtime_device)
    return TorchRuntime(
        device=runtime_device, device_name=device_name, dtype=runtime_dtype
    )


def downward_kink_mask_torch(
    gradient_left: torch.Tensor,
    gradient_right: torch.Tensor,
    at_breakpoint: torch.Tensor,
    *,
    tol: float,
) -> torch.Tensor:
    """Return breakpoints whose derivative jump admits one-sided descent."""

    return at_breakpoint & (gradient_left > gradient_right + max(float(tol), 1e-12))


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


def backward_error_stationarity_residual_torch(
    *,
    grad_smooth: torch.Tensor,
    adj: torch.Tensor,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Return a dimension-stable componentwise box-KKT backward error."""

    total_grad = grad_smooth + adj
    cone_projection = project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    violation = total_grad - cone_projection
    scale = torch.maximum(
        torch.ones_like(violation),
        torch.abs(grad_smooth) + torch.abs(adj),
    )
    if violation.numel() == 0:
        return torch.zeros((), dtype=phi.dtype, device=phi.device)
    return torch.max(torch.abs(violation) / scale)


def edge_kkt_maxima_from_diff_torch(
    *,
    diff: torch.Tensor,
    dual: torch.Tensor | None,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legacy and componentwise-scaled edge KKT maxima."""
    if diff.shape[0] == 0:
        zero = torch.zeros((), dtype=diff.dtype, device=diff.device)
        return zero, zero, zero, zero, zero
    prox_input = diff if dual is None else diff + dual
    prox_input_norm = torch.linalg.vector_norm(prox_input, dim=1)
    big = prox_input_norm >= radius
    # A literal floor can underflow in the working dtype. Only the exact-zero
    # denominator needs replacing; its numerator is zero as well.
    safe_norm = torch.where(
        prox_input_norm > 0.0, prox_input_norm, torch.ones_like(prox_input_norm)
    )
    if dual is None:
        active_residual = radius[:, None] * prox_input / safe_norm[:, None]
        ball_residual = torch.zeros_like(radius)
    else:
        active_residual = -dual + radius[:, None] * prox_input / safe_norm[:, None]
        ball_residual = torch.clamp(
            torch.linalg.vector_norm(dual, dim=1) - radius,
            min=0.0,
        )
    edge_residual = torch.where(
        big,
        torch.linalg.vector_norm(active_residual, dim=1),
        torch.linalg.vector_norm(diff, dim=1),
    )
    scale = torch.maximum(torch.ones_like(radius), radius)
    return (
        torch.max(edge_residual),
        torch.max(ball_residual),
        torch.max(radius),
        torch.max(edge_residual / scale),
        torch.max(ball_residual / scale),
    )


def _residual_maximum_torch(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Keep NaNs visible and reject negative residuals without synchronizing."""
    return torch.where(
        (first < 0.0) | (second < 0.0), float("inf"), torch.maximum(first, second)
    )


def graph_fusion_kkt_diagnostics_from_components_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    adj: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    atol: float,
    max_edge_residual: float | torch.Tensor,
    max_ball_residual: float | torch.Tensor,
    max_radius: float | torch.Tensor,
    max_scaled_edge_residual: float | torch.Tensor | None = None,
    max_scaled_ball_residual: float | torch.Tensor | None = None,
) -> KKTDiagnostics:
    """Assemble exact KKT diagnostics from an adjoint and edgewise maxima."""
    total_grad = grad_smooth + adj
    stat = stationarity_residual_torch(
        total_grad=total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
        atol=atol,
    )
    smooth_gradient_norm = float(torch.linalg.norm(grad_smooth).item())
    fusion_adjustment_norm = float(torch.linalg.norm(adj).item())
    projected_stationarity_norm = float(torch.linalg.norm(stat).item())
    stationarity_normalizer = float(1.0 + smooth_gradient_norm + fusion_adjustment_norm)
    stationarity_residual = float(
        projected_stationarity_norm / max(stationarity_normalizer, 1e-300)
    )
    backward_error_stationarity_residual = float(
        backward_error_stationarity_residual_torch(
            grad_smooth=grad_smooth,
            adj=adj,
            phi=phi,
            lower=lower,
            upper=upper,
        ).item()
    )
    box_violation = torch.maximum(
        torch.clamp(lower - phi, min=0.0),
        torch.clamp(phi - upper, min=0.0),
    )
    box_primal_violation = (
        float(torch.max(box_violation).item()) if box_violation.numel() else 0.0
    )
    box_scale = 1.0 + max(
        float(torch.max(torch.abs(lower)).item()) if lower.numel() else 0.0,
        float(torch.max(torch.abs(upper)).item()) if upper.numel() else 0.0,
    )
    box_residual = box_primal_violation / max(box_scale, 1e-300)

    edge_max = (
        float(max_edge_residual.item())
        if torch.is_tensor(max_edge_residual)
        else float(max_edge_residual)
    )
    ball_max = (
        float(max_ball_residual.item())
        if torch.is_tensor(max_ball_residual)
        else float(max_ball_residual)
    )
    radius_max = (
        float(max_radius.item()) if torch.is_tensor(max_radius) else float(max_radius)
    )
    valid_radius = np.isfinite(radius_max) and radius_max >= 0.0
    edge_denom = 1.0 + radius_max if valid_radius else float("nan")
    edge_subgradient_residual = edge_max / edge_denom
    dual_ball_residual = ball_max / edge_denom
    backward_error_edge_subgradient_residual = (
        edge_subgradient_residual
        if max_scaled_edge_residual is None
        else float(
            max_scaled_edge_residual.item()
            if torch.is_tensor(max_scaled_edge_residual)
            else max_scaled_edge_residual
        )
    )
    backward_error_dual_ball_residual = (
        dual_ball_residual
        if max_scaled_ball_residual is None
        else float(
            max_scaled_ball_residual.item()
            if torch.is_tensor(max_scaled_ball_residual)
            else max_scaled_ball_residual
        )
    )
    if not valid_radius:
        backward_error_edge_subgradient_residual = float("inf")
        backward_error_dual_ball_residual = float("inf")
    return KKTDiagnostics(
        stationarity_residual=stationarity_residual,
        edge_subgradient_residual=edge_subgradient_residual,
        dual_ball_residual=dual_ball_residual,
        box_residual=float(box_residual),
        backward_error_stationarity_residual=(
            backward_error_stationarity_residual
        ),
        backward_error_edge_subgradient_residual=(
            backward_error_edge_subgradient_residual
        ),
        backward_error_dual_ball_residual=backward_error_dual_ball_residual,
    )


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
) -> KKTDiagnostics:
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
        # Keep a cross-precision witness in its source dtype.  Terminal
        # float64 audits cast one bounded edge chunk at a time instead of
        # materializing another complete E x S dual.
        dual = dual_kkt.to(device=phi.device)
    chunk_size = _edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=phi.dtype,
        work_bytes=edge_work_bytes,
    )

    adj = torch.zeros_like(phi)
    if num_edges > 0 and lambda_value > 0.0 and dual is not None:
        # Same-precision CUDA follows the solver's preferred reduction;
        # cross-precision audits cast only bounded edge chunks.
        adj = graph_adjoint_edges_in_dtype(
            dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
            dtype=phi.dtype,
            device=phi.device,
            scale=dual_scale_value,
            edge_work_bytes=edge_work_bytes,
        )
    zero = torch.zeros((), dtype=phi.dtype, device=phi.device)
    max_edge_residual = zero
    max_ball_residual = zero
    max_radius = zero
    max_scaled_edge_residual = zero
    max_scaled_ball_residual = zero
    if num_edges == 0 or lambda_value <= 0.0:
        return graph_fusion_kkt_diagnostics_from_components_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            adj=adj,
            lower=lower,
            upper=upper,
            atol=atol,
            max_edge_residual=max_edge_residual,
            max_ball_residual=max_ball_residual,
            max_radius=max_radius,
            max_scaled_edge_residual=max_scaled_edge_residual,
            max_scaled_ball_residual=max_scaled_ball_residual,
        )

    # Proximal fixed-point residual: R_e = d_e - prox_{r_e*|.|_2}(d_e + y_e).
    # It is zero exactly when y_e lies in r_e * subgradient(|d_e|_2), including
    # at fused edges.  Aggregate maxima on device and synchronize once per
    # diagnostic rather than once per edge chunk.
    for edge_slice in _edge_slices(num_edges, chunk_size):
        diff = graph_forward_edges(
            phi,
            edge_u=edge_u[edge_slice],
            edge_v=edge_v[edge_slice],
        )
        radius = float(lambda_value) * edge_w[edge_slice].to(dtype=phi.dtype)
        dual_chunk = (
            None if dual is None else dual[edge_slice].to(dtype=phi.dtype)
        )
        if dual_chunk is not None and dual_scale_value != 1.0:
            dual_chunk = dual_scale_value * dual_chunk
        (
            edge_max,
            ball_max,
            radius_max,
            scaled_edge_max,
            scaled_ball_max,
        ) = edge_kkt_maxima_from_diff_torch(
            diff=diff,
            dual=dual_chunk,
            radius=radius,
        )
        max_edge_residual = _residual_maximum_torch(max_edge_residual, edge_max)
        max_ball_residual = _residual_maximum_torch(max_ball_residual, ball_max)
        max_radius = _residual_maximum_torch(max_radius, radius_max)
        max_scaled_edge_residual = _residual_maximum_torch(
            max_scaled_edge_residual, scaled_edge_max
        )
        max_scaled_ball_residual = _residual_maximum_torch(
            max_scaled_ball_residual, scaled_ball_max
        )

    return graph_fusion_kkt_diagnostics_from_components_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        adj=adj,
        lower=lower,
        upper=upper,
        atol=atol,
        max_edge_residual=max_edge_residual,
        max_ball_residual=max_ball_residual,
        max_radius=max_radius,
        max_scaled_edge_residual=max_scaled_edge_residual,
        max_scaled_ball_residual=max_scaled_ball_residual,
    )


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
    """Refine one fixed-primal witness against the raw-admission residual.

    Legacy globally normalized diagnostics remain available in each audit,
    but only componentwise backward error governs witness retention, target
    stopping, and plateau detection.
    """
    lambda_value = validate_lambda_value(lambda_value)
    num_edges, num_regions = int(edge_u.numel()), int(phi.shape[1])
    before_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi, grad_smooth=grad_smooth, dual_kkt=dual_kkt,
        lower=lower, upper=upper, edge_u=edge_u, edge_v=edge_v,
        edge_w=edge_w, lambda_value=lambda_value, atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    if num_edges == 0 or lambda_value <= 0.0:
        dual = torch.zeros((num_edges, num_regions), dtype=phi.dtype, device=phi.device)
        after_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi, grad_smooth=grad_smooth, dual_kkt=dual,
            lower=lower, upper=upper, edge_u=edge_u, edge_v=edge_v,
            edge_w=edge_w, lambda_value=lambda_value, atol=atol,
            edge_work_bytes=edge_work_bytes,
        )
        return {
            "dual": dual, "diag": after_diag,
            "status": "zero_penalty_no_dual_needed", "dual_refined": False,
            "fused_edges": 0, "nonzero_edges": 0,
            "stationarity_before": before_diag.stationarity_residual,
            "stationarity_after": after_diag.stationarity_residual,
            "refinement_iterations": 0,
        }

    chunk_size = _edge_chunk_size(
        num_edges=num_edges, num_regions=num_regions, dtype=phi.dtype,
        work_bytes=edge_work_bytes,
    )
    incoming_valid = bool(
        dual_kkt is not None and tuple(dual_kkt.shape) == (num_edges, num_regions)
    )
    incoming = (
        dual_kkt.to(dtype=phi.dtype, device=phi.device) if incoming_valid else None
    )
    incoming_residual = before_diag.backward_error_kkt_residual
    kkt_target = _CERTIFICATE_KKT_ATOL_SCALE * float(atol)
    if incoming_valid and np.isfinite(incoming_residual) and incoming_residual <= kkt_target:
        fused_edges, nonzero_edges = _graph_edge_activity_counts_torch(
            phi=phi, edge_u=edge_u, edge_v=edge_v, atol=atol,
            edge_work_bytes=edge_work_bytes,
        )
        return {
            "dual": incoming, "diag": before_diag,
            "status": "input_dual_retained", "dual_refined": False,
            "fused_edges": fused_edges, "nonzero_edges": nonzero_edges,
            "stationarity_before": before_diag.stationarity_residual,
            "stationarity_after": before_diag.stationarity_residual,
            "refinement_iterations": 0,
        }

    dual = torch.zeros((num_edges, num_regions), dtype=phi.dtype, device=phi.device)
    fused_edges = 0
    # Keep small-problem masks/radii once; larger problems reconstruct only
    # the current bounded chunk instead of retaining all edge work.
    single_chunk = chunk_size == num_edges
    for edge_slice in _edge_slices(num_edges, chunk_size):
        diff = graph_forward_edges(
            phi, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
        )
        diff_norm = torch.linalg.vector_norm(diff, dim=1)
        radius = float(lambda_value) * edge_w[edge_slice]
        active = diff_norm > float(atol)
        fused = ~active
        fused_edges += int(torch.sum(fused).item())
        dual_chunk = dual[edge_slice]
        if bool(torch.any(active).item()):
            dual_chunk[active] = (
                radius[active, None] * diff[active]
                / diff_norm[active, None].clamp_min(float(atol))
            )
        if incoming is not None and bool(torch.any(fused).item()):
            dual_chunk[fused] = project_dual_ball(incoming[edge_slice][fused], radius[fused])
    del diff, diff_norm, active, dual_chunk

    analytic_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi, grad_smooth=grad_smooth, dual_kkt=dual,
        lower=lower, upper=upper, edge_u=edge_u, edge_v=edge_v,
        edge_w=edge_w, lambda_value=lambda_value, atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    best_dual = dual.clone()
    best_diag = analytic_diag
    best_residual = analytic_diag.backward_error_kkt_residual
    best_source = "analytic"
    # Reconstructed nonfused edges can improve feasibility while worsening
    # stationarity. Preserve an equally good or better incoming witness.
    if incoming is not None and np.isfinite(incoming_residual) and incoming_residual <= best_residual:
        best_dual = incoming.clone() if single_chunk else incoming
        best_diag = before_diag
        best_residual = incoming_residual
        best_source = "incoming"

    refinement_iterations = 0
    if fused_edges > 0 and best_residual > kkt_target:
        num_nodes = int(phi.shape[0])
        degree = torch.bincount(torch.cat([edge_u, edge_v]), minlength=num_nodes).max()
        step = 0.25 / max(float(degree.item()), 1.0)
        plateau_anchor, stalled_iterations = best_residual, 0
        for _ in range(max(int(max_iter), 1)):
            refinement_iterations += 1
            if single_chunk:
                adj = graph_adjoint_edges(
                    dual, edge_u=edge_u, edge_v=edge_v, num_nodes=num_nodes,
                )
            else:
                # Preserve the old streamed update's scatter reduction order
                # on every device; full audits retain their own adjoint path.
                adj = torch.zeros_like(phi)
                for edge_slice in _edge_slices(num_edges, chunk_size):
                    adj.index_add_(0, edge_u[edge_slice], dual[edge_slice])
                    adj.index_add_(0, edge_v[edge_slice], dual[edge_slice], alpha=-1.0)
            # Freeze the complete current adjoint/residual before ANY update.
            # Recomputing this between chunks would change the algorithm.
            stat = stationarity_residual_torch(
                total_grad=grad_smooth + adj, phi=phi, lower=lower, upper=upper, atol=atol,
            )
            mapping_delta = 0.0
            for edge_slice in _edge_slices(num_edges, chunk_size):
                if not single_chunk:
                    diff = graph_forward_edges(
                        phi, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
                    )
                    fused = torch.linalg.vector_norm(diff, dim=1) <= float(atol)
                    radius = float(lambda_value) * edge_w[edge_slice]
                if not bool(torch.any(fused).item()):
                    continue
                fused_before = dual[edge_slice][fused]
                stat_diff = graph_forward_edges(
                    stat, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
                )
                fused_after = project_dual_ball(
                    fused_before - float(step) * stat_diff[fused], radius[fused],
                )
                mapping_delta = max(
                    mapping_delta, float(torch.max(torch.abs(fused_after - fused_before)).item()),
                )
                dual[edge_slice][fused] = fused_after
            diag = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi, grad_smooth=grad_smooth, dual_kkt=dual,
                lower=lower, upper=upper, edge_u=edge_u, edge_v=edge_v,
                edge_w=edge_w, lambda_value=lambda_value, atol=atol,
                edge_work_bytes=edge_work_bytes,
            )
            residual = diag.backward_error_kkt_residual
            if residual < best_residual:
                best_residual, best_diag = residual, diag
                if best_source == "incoming":
                    best_dual = dual.clone()
                else:
                    best_dual.copy_(dual)
                best_source = "refined"
            if residual <= kkt_target:
                break
            plateau_anchor, stalled_iterations, plateaued = _update_certificate_refinement_plateau(
                anchor_residual=plateau_anchor, best_residual=best_residual,
                mapping_delta=mapping_delta, stalled_iterations=stalled_iterations,
                atol=atol, dtype=phi.dtype,
            )
            if plateaued:
                break

    status = (
        "input_dual_retained" if best_source == "incoming" else
        "refined_fused_edge_dual" if fused_edges else "analytic_nonfused_dual"
    )
    return {
        "dual": best_dual, "diag": best_diag, "status": status,
        "dual_refined": best_source != "incoming", "fused_edges": fused_edges,
        "nonzero_edges": num_edges - fused_edges,
        "stationarity_before": before_diag.stationarity_residual,
        "stationarity_after": best_diag.stationarity_residual,
        "refinement_iterations": refinement_iterations,
    }


def _complete_graph_admm_stationarity_components_torch(
    *,
    phi: torch.Tensor,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    adj: torch.Tensor,
    atol: float,
) -> tuple[torch.Tensor, float, float]:
    grad_smooth = h * (phi - U)
    stat = stationarity_residual_torch(
        total_grad=grad_smooth + adj,
        phi=phi,
        lower=lower,
        upper=upper,
        atol=atol,
    )
    residual = torch.linalg.vector_norm(stat) / (
        1.0 + torch.linalg.vector_norm(grad_smooth) + torch.linalg.vector_norm(adj)
    )
    backward_error = backward_error_stationarity_residual_torch(
        grad_smooth=grad_smooth,
        adj=adj,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    return grad_smooth, float(residual.item()), float(backward_error.item())


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
    use_backward_error_stopping: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, KKTDiagnostics | None]:
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
        return projected, empty_dual, 0, residual <= tol, None

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
    last_diagnostics = None
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

        audit_due = (
            iterations >= actual_max_iter
            or iterations % max(int(kkt_check_every), 1) == 0
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
            last_diagnostics = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                dual_kkt=dual,
                grad_smooth=h * (phi - U),
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                atol=tol,
            )
            last_residual = float(
                last_diagnostics.backward_error_kkt_residual
                if use_backward_error_stopping
                else last_diagnostics.kkt_residual
            )
            if cheap_converged and last_residual <= 5.0 * tol:
                converged = True
                break

    return phi, dual, iterations, converged, last_diagnostics


def _complete_graph_isotropic_box_qp_torch(
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho: float | torch.Tensor,
    q: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    rho_t = (
        rho.to(dtype=U.dtype, device=U.device)
        if torch.is_tensor(rho)
        else torch.as_tensor(float(rho), dtype=U.dtype, device=U.device)
    )
    if (
        U.device.type == "cuda"
        and U.dtype == torch.float64
        and U.ndim == 2
        and int(U.shape[1]) > 1
    ):
        return _complete_graph_isotropic_box_qp_cuda(
            U,
            h,
            lower,
            upper,
            rho_t,
            q,
            max(int(max_iter), 16),
        )
    if U.device.type == "cpu" and U.ndim == 2 and int(U.shape[1]) == 1:
        exact = _complete_graph_scalar_box_qp_cpu(
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            rho_t=rho_t,
            q=q,
        )
        if exact is not None:
            return exact
    return _complete_graph_isotropic_box_qp_bisection(
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        rho_t=rho_t,
        q=q,
        max_iter=max_iter,
    )


def _complete_graph_scalar_box_qp_cpu(
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho_t: torch.Tensor,
    q: torch.Tensor,
) -> torch.Tensor | None:
    """Solve the one-region complete-graph box QP by its exact breakpoints."""

    num_mutations = int(U.shape[0])
    rhs = h * U + rho_t * q
    denom = h + rho_t * float(num_mutations)
    intercept = (rhs / denom)[:, 0]
    slope = (rho_t / denom)[:, 0]
    lower_scalar = lower[:, 0]
    upper_scalar = upper[:, 0]
    lower_break = (lower_scalar - intercept) / slope
    upper_break = (upper_scalar - intercept) / slope
    breakpoints = torch.cat((lower_break, upper_break))
    delta_intercept = torch.cat(
        (intercept - lower_scalar, upper_scalar - intercept)
    )
    delta_slope = torch.cat((slope, -slope))
    order = torch.argsort(breakpoints, stable=True)
    breakpoints = breakpoints[order]
    piece_intercept = torch.sum(lower_scalar) + torch.cumsum(
        delta_intercept[order], dim=0
    )
    piece_slope = -torch.ones((), dtype=U.dtype) + torch.cumsum(
        delta_slope[order], dim=0
    )
    roots = -piece_intercept / piece_slope
    next_breakpoints = torch.cat(
        (
            breakpoints[1:],
            torch.full((1,), torch.inf, dtype=U.dtype),
        )
    )
    valid = torch.isfinite(roots) & (roots >= breakpoints) & (
        roots <= next_breakpoints
    )
    root_indices = torch.nonzero(valid, as_tuple=False)
    if root_indices.numel() == 0:
        return None
    root = roots[root_indices[0, 0]]
    return torch.minimum(
        torch.maximum((rhs + rho_t * root) / denom, lower),
        upper,
    )


def _complete_graph_isotropic_box_qp_bisection(
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho_t: torch.Tensor,
    q: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    num_mutations = int(U.shape[0])
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


def _complete_graph_isotropic_box_qp_compiled_impl(
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho_t: torch.Tensor,
    q: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    return _complete_graph_isotropic_box_qp_bisection(
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        rho_t=rho_t,
        q=q,
        max_iter=max_iter,
    )


# Multi-region path fits invoke this bounded QP many thousands of times.  A
# shape-specific Inductor kernel amortizes compilation there, while the common
# one-region workflow stays eager to avoid paying its cold-start cost.
_compiled_complete_graph_isotropic_box_qp_cuda = torch.compile(
    _complete_graph_isotropic_box_qp_compiled_impl,
    fullgraph=True,
    dynamic=False,
)
_cuda_box_qp_compile_failed = False


def _complete_graph_isotropic_box_qp_cuda(*args) -> torch.Tensor:
    global _cuda_box_qp_compile_failed
    if _cuda_box_qp_compile_failed:
        return _complete_graph_isotropic_box_qp_compiled_impl(*args)
    try:
        return _compiled_complete_graph_isotropic_box_qp_cuda(*args)
    except Exception as exc:
        exception_module = type(exc).__module__
        if not exception_module.startswith(("torch._dynamo", "torch._inductor")):
            raise
        _cuda_box_qp_compile_failed = True
        warnings.warn(
            "CUDA box-QP compilation failed; falling back to eager Torch kernels: "
            f"{exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _complete_graph_isotropic_box_qp_compiled_impl(*args)


def _closed_form_box_fusion_result(
    *,
    runtime: TorchRuntime,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tol: float,
    diagnostics_out: dict[str, int] | None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, KKTDiagnostics]:
    projected = torch.minimum(torch.maximum(U, lower), upper)
    diag = graph_fusion_kkt_diagnostics_from_components_torch(
        phi=projected,
        grad_smooth=h * (projected - U),
        adj=torch.zeros_like(projected),
        lower=lower,
        upper=upper,
        atol=tol,
        max_edge_residual=0.0,
        max_ball_residual=0.0,
        max_radius=0.0,
    )
    residual = float(diag.kkt_residual)
    empty_dual = torch.zeros(
        (0, projected.shape[1]), dtype=runtime.dtype, device=runtime.device
    )
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(inner_kkt_audits=0, inner_stationarity_checks=0)
    return projected, empty_dual, 0, residual <= tol, diag


def _initial_complete_graph_rho(
    h: torch.Tensor, *, num_mutations: int, spectral_rho: bool
) -> float:
    median_h = torch.median(h)
    if spectral_rho:
        # The nonzero spectrum of the complete-graph D.T @ D is M.
        return float(
            torch.clamp(
                median_h / max(float(num_mutations), 1.0), min=1e-8, max=1e8
            ).item()
        )
    return float(torch.clamp(median_h, min=1e-3, max=1e3).item())


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
    use_backward_error_stopping: bool = False,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
    box_phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    box_max_iter: int = DEFAULT_BOX_MAX_ITER,
    edge_work_bytes: int | None = None,
    diagnostics_out: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, KKTDiagnostics]:
    """One complete-graph ALM driver with the original full/chunked arithmetic."""
    budget = DEFAULT_EDGE_WORK_BYTES if edge_work_bytes is None else int(edge_work_bytes)
    num_edges, num_regions = int(edge_u.numel()), int(phi_start.shape[1])
    # Validate even empty/zero-penalty problems; preserve the historical byte
    # boundary when a budget cannot hold even one edge's work.
    chunk_size = _edge_chunk_size(
        num_edges=num_edges, num_regions=num_regions, dtype=runtime.dtype, work_bytes=budget,
    )
    streamed = _edge_tensor_nbytes(
        num_edges=num_edges, num_regions=num_regions, dtype=runtime.dtype,
    ) > budget
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(
        torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower), upper,
    )
    if lambda_value <= 0.0 or num_edges == 0:
        return _closed_form_box_fusion_result(
            runtime=runtime, U=U, h=h, lower=lower, upper=upper, tol=tol,
            diagnostics_out=diagnostics_out,
        )
    rho = _initial_complete_graph_rho(
        h, num_mutations=num_mutations, spectral_rho=spectral_rho,
    )
    rho_t = torch.as_tensor(rho, dtype=runtime.dtype, device=runtime.device)
    valid_start = dual_start is not None and tuple(dual_start.shape) == (num_edges, num_regions)
    if streamed:
        scaled_dual = torch.empty((num_edges, num_regions), dtype=runtime.dtype, device=runtime.device)
        if not valid_start:
            scaled_dual.zero_()
        else:
            for edge_slice in _edge_slices(num_edges, chunk_size):
                initial_chunk = dual_start[edge_slice].to(dtype=runtime.dtype, device=runtime.device)
                if dual_start_is_actual:
                    radius = float(lambda_value) * edge_w[edge_slice]
                    norm = torch.linalg.vector_norm(initial_chunk, dim=1, keepdim=True)
                    projection_scale = torch.maximum(
                        torch.ones_like(norm),
                        norm / radius[:, None].clamp_min(torch.finfo(runtime.dtype).tiny),
                    )
                    scaled_dual[edge_slice].copy_(initial_chunk / projection_scale / float(rho))
                else:
                    scaled_dual[edge_slice].copy_(initial_chunk)
        # Only the scaled dual and previous/current split remain edge-sized.
        z_state = torch.empty_like(scaled_dual)
        for edge_slice in _edge_slices(num_edges, chunk_size):
            z_state[edge_slice].copy_(graph_forward_edges(
                phi, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
            ))
    else:
        radius = float(lambda_value) * edge_w
        if valid_start:
            initial_dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
            if dual_start_is_actual:
                initial_dual = project_dual_ball(initial_dual, radius)
                scaled_dual = initial_dual / rho
            else:
                scaled_dual = initial_dual
        else:
            scaled_dual = torch.zeros((num_edges, num_regions), dtype=runtime.dtype, device=runtime.device)
        shrink_radius = radius / rho
        edge_diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
        z_previous = edge_diff if spectral_rho else None

    converged = False
    iterations = 0
    last_diagnostics = None
    actual_max_iter = max(int(max_iter), 10)
    box_iter = _box_qp_sweeps_for_atol(box_phi_atol, max_iter=box_max_iter)
    kkt_stop_tol = float(tol) + 0.25 * min(float(box_phi_atol), float(tol))
    kkt_audits = stationarity_checks = 0
    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        rho_used = float(rho)
        rho_update_due = bool(spectral_rho and iterations % 10 == 0)
        final_iteration = iterations >= actual_max_iter
        check_due = bool(final_iteration or iterations % max(int(kkt_check_every), 1) == 0)

        # Phase 1: finish all split updates and D.T(z-u) before the node solve.
        if streamed:
            q = torch.zeros_like(phi)
            dual_residual_node = torch.zeros_like(phi) if rho_update_due else None
            audit_adjoint = torch.zeros_like(phi) if check_due else None
            for edge_slice in _edge_slices(num_edges, chunk_size):
                z_new = graph_forward_edges(
                    phi, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
                )
                z_new.add_(scaled_dual[edge_slice])
                z_norm = torch.linalg.vector_norm(z_new, dim=1, keepdim=True)
                shrink_radius = float(lambda_value) * edge_w[edge_slice] / float(rho)
                shrink = torch.clamp(
                    1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12), min=0.0,
                )
                z_new.mul_(shrink)
                if dual_residual_node is not None:
                    z_delta = z_new - z_state[edge_slice]
                    z_delta.mul_(rho_used)
                    dual_residual_node.index_add_(0, edge_u[edge_slice], z_delta)
                    dual_residual_node.index_add_(0, edge_v[edge_slice], z_delta, alpha=-1.0)
                    del z_delta
                z_state[edge_slice].copy_(z_new)
                z_new.sub_(scaled_dual[edge_slice])
                q.index_add_(0, edge_u[edge_slice], z_new)
                q.index_add_(0, edge_v[edge_slice], z_new, alpha=-1.0)
                del z_new, z_norm, shrink, shrink_radius
        else:
            z_argument = edge_diff + scaled_dual
            z_norm = torch.linalg.norm(z_argument, dim=1, keepdim=True)
            shrink = torch.clamp(
                1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12), min=0.0,
            )
            z_new = shrink * z_argument
            rhs_edge = z_new - scaled_dual
            q = graph_adjoint_edges(
                rhs_edge, edge_u=edge_u, edge_v=edge_v,
                num_nodes=int(phi.shape[0]), prefer_cpu_bincount=True,
            )
            del z_argument, z_norm, shrink, rhs_edge

        phi_new = _complete_graph_isotropic_box_qp_torch(
            U=U, h=h, lower=lower, upper=upper, rho=rho_t, q=q, max_iter=box_iter,
        )

        # Phase 2: finish every dual update/global residual before changing rho.
        if streamed:
            primal_sum_squares = (
                torch.zeros((), dtype=runtime.dtype, device=runtime.device) if rho_update_due else None
            )
            for edge_slice in _edge_slices(num_edges, chunk_size):
                primal_residual = graph_forward_edges(
                    phi_new, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
                )
                primal_residual.sub_(z_state[edge_slice])
                if primal_sum_squares is not None:
                    primal_sum_squares.add_(torch.dot(
                        primal_residual.reshape(-1), primal_residual.reshape(-1),
                    ))
                scaled_dual[edge_slice].add_(primal_residual)
                if audit_adjoint is not None:
                    actual_dual_chunk = rho_used * scaled_dual[edge_slice]
                    audit_adjoint.index_add_(0, edge_u[edge_slice], actual_dual_chunk)
                    audit_adjoint.index_add_(0, edge_v[edge_slice], actual_dual_chunk, alpha=-1.0)
                    del actual_dual_chunk
                del primal_residual
            if rho_update_due:
                primal_norm = float(torch.sqrt(primal_sum_squares).item())
                dual_norm = float(torch.linalg.vector_norm(dual_residual_node).item())
                del primal_sum_squares, dual_residual_node
        else:
            edge_diff_new = graph_forward_edges(phi_new, edge_u=edge_u, edge_v=edge_v)
            primal_residual = edge_diff_new - z_new
            scaled_dual = scaled_dual + primal_residual
            if rho_update_due:
                primal_norm = float(torch.linalg.norm(primal_residual).item())
                z_delta = z_new - z_previous
                dual_residual_node = graph_adjoint_edges(
                    float(rho) * z_delta, edge_u=edge_u, edge_v=edge_v,
                    num_nodes=int(phi.shape[0]), prefer_cpu_bincount=True,
                )
                dual_norm = float(torch.linalg.norm(dual_residual_node).item())
                del z_delta, dual_residual_node

        if rho_update_due:
            # Balance scale-free residuals; rho rescaling preserves actual y.
            dual_balance_norm = dual_norm / max(abs(float(rho)), 1e-300)
            next_rho = float(rho)
            if np.isfinite(primal_norm) and np.isfinite(dual_balance_norm):
                if primal_norm > 10.0 * max(dual_balance_norm, 1e-300):
                    next_rho = min(2.0 * float(rho), 1e8)
                elif dual_balance_norm > 10.0 * max(primal_norm, 1e-300):
                    next_rho = max(0.5 * float(rho), 1e-8)
            if next_rho != float(rho):
                if streamed:
                    scaled_dual.mul_(float(rho) / next_rho)
                else:
                    scaled_dual = scaled_dual * (float(rho) / next_rho)
                rho = float(next_rho)
                rho_t.fill_(rho)
                if not streamed:
                    shrink_radius = radius / float(rho)
        phi = phi_new
        if not streamed:
            edge_diff = edge_diff_new
            if spectral_rho:
                z_previous = z_new

        if check_due:
            stationarity_checks += 1
            if not streamed:
                actual_dual = float(rho) * scaled_dual
                audit_adjoint = graph_adjoint_edges(
                    actual_dual, edge_u=edge_u, edge_v=edge_v,
                    num_nodes=int(phi.shape[0]), prefer_cpu_bincount=True,
                )
            # Streamed audit_adjoint deliberately precedes rho rescaling, as in
            # the original streamed route. Dense audits use the rescaled dual.
            grad_smooth, stationarity_residual, backward_error_stationarity_residual = (
                _complete_graph_admm_stationarity_components_torch(
                    phi=phi, U=U, h=h, lower=lower, upper=upper, adj=audit_adjoint, atol=tol,
                )
            )
            progress_stationarity_residual = (
                backward_error_stationarity_residual if use_backward_error_stopping
                else stationarity_residual
            )
            audit_due = bool(final_iteration or progress_stationarity_residual <= kkt_stop_tol)
        else:
            audit_due = False
        if audit_due:
            kkt_audits += 1
            if streamed:
                edge_max = torch.zeros((), dtype=runtime.dtype, device=runtime.device)
                ball_max = torch.zeros_like(edge_max)
                radius_max = torch.zeros_like(edge_max)
                scaled_edge_max = torch.zeros_like(edge_max)
                scaled_ball_max = torch.zeros_like(edge_max)
                for edge_slice in _edge_slices(num_edges, chunk_size):
                    diff = graph_forward_edges(
                        phi, edge_u=edge_u[edge_slice], edge_v=edge_v[edge_slice],
                    )
                    chunk_edge, chunk_ball, chunk_radius, chunk_scaled_edge, chunk_scaled_ball = (
                        edge_kkt_maxima_from_diff_torch(
                            diff=diff, dual=float(rho) * scaled_dual[edge_slice],
                            radius=float(lambda_value) * edge_w[edge_slice],
                        )
                    )
                    edge_max = _residual_maximum_torch(edge_max, chunk_edge)
                    ball_max = _residual_maximum_torch(ball_max, chunk_ball)
                    radius_max = _residual_maximum_torch(radius_max, chunk_radius)
                    scaled_edge_max = _residual_maximum_torch(scaled_edge_max, chunk_scaled_edge)
                    scaled_ball_max = _residual_maximum_torch(scaled_ball_max, chunk_scaled_ball)
                    del diff
            else:
                edge_max, ball_max, radius_max, scaled_edge_max, scaled_ball_max = (
                    edge_kkt_maxima_from_diff_torch(diff=edge_diff, dual=actual_dual, radius=radius)
                )
            last_diagnostics = graph_fusion_kkt_diagnostics_from_components_torch(
                phi=phi, grad_smooth=grad_smooth, adj=audit_adjoint, lower=lower, upper=upper,
                max_edge_residual=edge_max, max_ball_residual=ball_max, max_radius=radius_max,
                max_scaled_edge_residual=scaled_edge_max, max_scaled_ball_residual=scaled_ball_max,
                atol=tol,
            )
            residual = (
                last_diagnostics.backward_error_kkt_residual if use_backward_error_stopping
                else last_diagnostics.kkt_residual
            )
            if residual <= kkt_stop_tol:
                converged = True
                break
        del q, phi_new
        if streamed:
            if audit_adjoint is not None:
                del audit_adjoint
        else:
            del primal_residual, z_new

    if streamed:
        # Release split storage before materializing the sole outgoing dual.
        del z_state
        actual_dual = float(rho) * scaled_dual
    if diagnostics_out is not None:
        diagnostics_out["inner_kkt_audits"] = int(kkt_audits)
        diagnostics_out["inner_stationarity_checks"] = int(stationarity_checks)
    return phi, actual_dual, iterations, converged, last_diagnostics
