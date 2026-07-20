from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import torch

from ...io.data import TumorData
from .certificates import (
    audit_graph_fusion_certificate,
    refine_graph_fusion_certificate,
)
from .graph import resolve_pairwise_fusion_graph
from .graph_ops import (
    build_complete_adaptive_tensor_graph,
    dense_complete_solver_memory_preflight,
    project_dual_ball,
    tensor_graph_to_pairwise_graph,
    tensorize_graph,
)
from .quotient_workset import (
    compressed_certificate_for_primal,
    solve_majorized_subproblem_quotient_workset_torch,
)
from .starts import (
    compute_pooled_observed_data_start_torch,
    compute_scalar_mutation_region_wells_torch,
    compute_scalar_well_start_bank_torch,
)
from .torch_backend import (
    TorchTumorData,
    as_runtime_tensor,
    mutation_region_terms_torch,
    dtype_name,
    em_surrogate_terms_torch,
    graph_fusion_kkt_residual_from_grad_torch,
    pairwise_penalty_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
    solve_majorized_subproblem_pdhg_torch,
    to_torch_tumor_data,
    validate_lambda_value,
)
from .types import (
    BackendWorkCounters,
    CertificateOptions,
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    DenseWarmState,
    ExactSolverResourceLimit,
    ExactFusionProvenance,
    FusionFitArtifacts,
    InnerDiagnostics,
    InnerSolveResult,
    KKTDiagnostics,
    OuterDiagnostics,
    PairwiseFusionGraph,
    PrimalOnlyWarmState,
    QuotientWorksetWarmState,
    SolverContext,
    SolverState,
    TensorFusionGraph,
    TensorProblem,
    TorchFitResult,
    WorksetMemoryOptions,
)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, value: int) -> int:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return int(parent)

    def union(self, left: int, right: int) -> None:
        root_left = self.find(int(left))
        root_right = self.find(int(right))
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def _cluster_labels(
    phi: np.ndarray,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    tol: float,
) -> np.ndarray:
    num_mutations = int(phi.shape[0])
    if num_mutations == 0:
        return np.zeros((0,), dtype=np.int64)

    if edge_u.size == 0:
        return np.arange(num_mutations, dtype=np.int64)

    fused = np.linalg.norm(phi[edge_u] - phi[edge_v], axis=1) <= float(tol)
    if not np.any(fused):
        return np.arange(num_mutations, dtype=np.int64)

    uf = _UnionFind(num_mutations)
    for left, right in zip(edge_u[fused], edge_v[fused]):
        uf.union(int(left), int(right))

    labels = np.empty(num_mutations, dtype=np.int64)
    root_to_label: dict[int, int] = {}
    next_label = 0
    for idx in range(num_mutations):
        root = uf.find(idx)
        label = root_to_label.get(root)
        if label is None:
            label = next_label
            root_to_label[root] = label
            next_label += 1
        labels[idx] = int(label)
    return labels


def cluster_labels_from_edges(
    phi: np.ndarray,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    tol: float,
) -> np.ndarray:
    return _cluster_labels(
        np.asarray(phi),
        edge_u=np.asarray(edge_u, dtype=np.int64),
        edge_v=np.asarray(edge_v, dtype=np.int64),
        tol=float(tol),
    )


def cluster_diameters_from_edges(
    phi: np.ndarray,
    labels: np.ndarray,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> tuple[np.ndarray, bool]:
    phi = np.asarray(phi)
    labels = np.asarray(labels, dtype=np.int64)
    edge_u = np.asarray(edge_u, dtype=np.int64)
    edge_v = np.asarray(edge_v, dtype=np.int64)
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    diameters = np.zeros(n_clusters, dtype=np.float64)
    if labels.size == 0:
        return diameters, False
    n_rows = int(labels.shape[0])
    expected_complete_edges = n_rows * (n_rows - 1) // 2
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    exact = bool(edge_u.size == expected_complete_edges or np.all(cluster_sizes <= 1))
    if edge_u.size == 0 or edge_v.size == 0:
        return diameters, exact
    same_cluster = labels[edge_u] == labels[edge_v]
    if not np.any(same_cluster):
        return diameters, exact
    same_u = edge_u[same_cluster]
    same_v = edge_v[same_cluster]
    distances = np.linalg.norm(phi[same_u] - phi[same_v], axis=1)
    np.maximum.at(diameters, labels[same_u], distances.astype(np.float64, copy=False))
    return diameters, exact


def _cluster_summary_from_labels(
    phi: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    centers = np.zeros((n_clusters, phi.shape[1]), dtype=phi.dtype)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    np.add.at(centers, labels, phi)
    centers /= np.clip(counts[:, None], 1.0, None)
    phi_clustered = centers[labels]
    return centers.astype(phi.dtype, copy=False), phi_clustered.astype(
        phi.dtype, copy=False
    )


def _deduplicate_starts(
    starts: list[np.ndarray | torch.Tensor],
    *,
    runtime,
    atol: float = 1e-8,
) -> list[np.ndarray | torch.Tensor]:
    unique: list[np.ndarray | torch.Tensor] = []
    unique_tensors: list[torch.Tensor] = []
    for start in starts:
        start_tensor = _tensor_from_start(start, runtime).detach()
        duplicate = any(
            torch.allclose(start_tensor, retained, rtol=0.0, atol=float(atol))
            for retained in unique_tensors
        )
        if duplicate:
            continue
        unique_tensors.append(start_tensor)
        unique.append(start)
    return unique


def _inner_model_value_torch(
    phi: torch.Tensor,
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    quad = 0.5 * torch.sum(h * torch.square(phi - U))
    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    return quad + penalty


def _objective_value_from_mutation_region_terms_torch(
    mutation_region_terms,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> tuple[float, float, float, torch.Tensor]:
    fit_loss_tensor = torch.sum(mutation_region_terms.loss)
    penalty_tensor = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    objective_tensor = fit_loss_tensor + penalty_tensor
    fit_loss, penalty, objective = (
        float(value)
        for value in torch.stack(
            [
                fit_loss_tensor.detach(),
                penalty_tensor.detach(),
                objective_tensor.detach(),
            ]
        ).cpu()
    )
    return fit_loss, penalty, objective, mutation_region_terms.gamma_major


def _objective_value_once_torch(
    torch_data: TorchTumorData,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    major_prior: float,
    eps: float,
) -> tuple[float, float, float, torch.Tensor]:
    terms = mutation_region_terms_torch(
        torch_data,
        phi,
        major_prior=major_prior,
        eps=eps,
    )
    return _objective_value_from_mutation_region_terms_torch(
        terms,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )


def _update_minimum(current: float, candidate: float) -> float:
    if not np.isfinite(candidate):
        return float(current)
    if not np.isfinite(current):
        return float(candidate)
    return float(min(current, candidate))


_MISSING_SURROGATE_CURVATURE = 1e-6
_OUTER_KKT_CHECK_EVERY = 4
_FULL_STEP_MAX_CURVATURE_ATTEMPTS = 24
_UNIMODAL_GLOBAL_OPTIMALITY_BASIS = "assumed_unimodal_objective_plus_kkt"


def _safe_surrogate_curvature_and_gradient(
    surrogate_terms,
    count_observed: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_base = torch.clamp(surrogate_terms.hess_upper, min=_MISSING_SURROGATE_CURVATURE)
    surrogate_grad = surrogate_terms.grad
    if count_observed is None:
        return h_base, surrogate_grad

    observed = count_observed
    h_base = torch.where(
        observed,
        h_base,
        torch.full_like(h_base, _MISSING_SURROGATE_CURVATURE),
    )
    surrogate_grad = torch.where(
        observed, surrogate_grad, torch.zeros_like(surrogate_grad)
    )
    return h_base, surrogate_grad


def _safe_majorized_center(
    phi: torch.Tensor,
    *,
    surrogate_grad: torch.Tensor,
    h: torch.Tensor,
    count_observed: torch.Tensor | None,
) -> torch.Tensor:
    U_raw = phi - surrogate_grad / h
    if count_observed is None:
        return U_raw
    return torch.where(count_observed, U_raw, phi)


def _validate_solver_tolerance(tol: float) -> float:
    value = float(tol)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Solver tolerance must be a positive finite value.")
    return value


def _normalize_inner_backend(inner_backend: str) -> str:
    normalized = str(inner_backend).strip().lower().replace("-", "_")
    if normalized not in {"auto", "dense", "quotient_workset"}:
        raise ValueError("inner_backend must be one of: auto, dense, quotient_workset.")
    return normalized


def _combine_fallback_reasons(*reasons: str) -> str:
    unique: list[str] = []
    for reason in reasons:
        normalized = str(reason).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return ";".join(unique)


def _normalize_objective_shape(objective_shape: str) -> str:
    normalized = str(objective_shape).strip().lower()
    if normalized not in {
        "unimodal",
        "unimodal_full_step_backtracking",
        "generic_nonconvex",
    }:
        raise ValueError(
            "objective_shape must be 'unimodal', "
            "'unimodal_full_step_backtracking', or 'generic_nonconvex'."
        )
    return normalized


def _data_fingerprint(data: TumorData) -> str:
    digest = hashlib.sha256()

    def update_text(value: str) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def update_text_sequence(values: list[str]) -> None:
        digest.update(len(values).to_bytes(8, "little"))
        for value in values:
            update_text(value)

    def update_array(name: str, values: np.ndarray) -> None:
        update_text(name)
        array = np.ascontiguousarray(np.asarray(values))
        update_text(str(array.dtype))
        digest.update(len(array.shape).to_bytes(8, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        digest.update(array.tobytes())

    update_text(data.tumor_id)
    update_text_sequence(list(data.mutation_ids))
    update_text_sequence(list(data.region_ids))
    for name in (
        "alt_counts",
        "total_counts",
        "purity",
        "major_cn",
        "minor_cn",
        "normal_cn",
        "has_cna",
        "scaling",
        "phi_upper",
        "phi_init",
        "init_major_mask",
    ):
        update_array(name, getattr(data, name))
    count_observed = getattr(data, "count_observed", None)
    if count_observed is None:
        count_observed_array = np.ones_like(np.asarray(data.alt_counts), dtype=bool)
    else:
        count_observed_array = np.asarray(count_observed, dtype=bool)
    update_array("count_observed", count_observed_array)
    return digest.hexdigest()


def _graph_fingerprint(graph: PairwiseFusionGraph) -> str:
    """Return an order-sensitive identity for the original weighted graph."""

    digest = hashlib.sha256()
    for name, values in (
        ("edge_u", graph.edge_u),
        ("edge_v", graph.edge_v),
        ("edge_w", graph.edge_w),
    ):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "little"))
        digest.update(encoded_name)
        array = np.ascontiguousarray(np.asarray(values))
        encoded_dtype = str(array.dtype).encode("utf-8")
        digest.update(len(encoded_dtype).to_bytes(8, "little"))
        digest.update(encoded_dtype)
        digest.update(len(array.shape).to_bytes(8, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little", signed=True))
        digest.update(array.tobytes())
    digest.update(str(graph.name).encode("utf-8"))
    digest.update(int(graph.degree_bound).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _objective_spec_fingerprint(
    *,
    data_fingerprint: str,
    graph_hash: str,
    major_prior: float,
    eps: float,
) -> str:
    digest = hashlib.sha256()
    for value in (
        "clipp2_observed_objective_v1",
        data_fingerprint,
        graph_hash,
        float(major_prior).hex(),
        float(eps).hex(),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _certificate_problem_fingerprint(
    *, objective_spec_hash: str, lambda_value: float
) -> str:
    digest = hashlib.sha256()
    for value in (objective_spec_hash, float(lambda_value).hex()):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_from_start(
    start: np.ndarray | torch.Tensor,
    runtime,
) -> torch.Tensor:
    return as_runtime_tensor(start, runtime)


def _validate_prebuilt_tensor_graph(
    graph: PairwiseFusionGraph,
    tensor_graph: TensorFusionGraph,
    *,
    runtime,
    num_nodes: int,
) -> None:
    """Validate cheap invariants for an already paired host/device graph."""

    if int(tensor_graph.num_nodes) != int(num_nodes):
        raise ValueError("prebuilt_tensor_graph has the wrong number of nodes.")
    if str(tensor_graph.name) != str(graph.name):
        raise ValueError("prebuilt_tensor_graph name does not match graph.")
    if tensor_graph.edge_index.ndim != 2 or int(tensor_graph.edge_index.shape[0]) != 2:
        raise ValueError("prebuilt_tensor_graph edge_index must have shape (2, E).")
    edge_count = int(tensor_graph.edge_index.shape[1])
    if edge_count != int(np.asarray(graph.edge_u).size):
        raise ValueError("prebuilt_tensor_graph edge count does not match graph.")
    if tensor_graph.weight.ndim != 1 or int(tensor_graph.weight.numel()) != edge_count:
        raise ValueError("prebuilt_tensor_graph weights must have shape (E,).")
    if tensor_graph.edge_index.dtype != torch.long:
        raise ValueError("prebuilt_tensor_graph edge indices must use torch.long.")
    if tensor_graph.weight.dtype != runtime.dtype:
        raise ValueError(
            "prebuilt_tensor_graph weights do not match the runtime dtype."
        )
    if tuple(tensor_graph.degree.shape) != (int(num_nodes),):
        raise ValueError("prebuilt_tensor_graph degree has the wrong shape.")
    if tuple(tensor_graph.pdhg_tau_node.shape) != (int(num_nodes), 1):
        raise ValueError(
            "prebuilt_tensor_graph PDHG preconditioner has the wrong shape."
        )
    for value in (
        tensor_graph.edge_index,
        tensor_graph.weight,
        tensor_graph.degree,
        tensor_graph.pdhg_tau_node,
    ):
        if value.device.type != runtime.device.type or (
            runtime.device.index is not None
            and value.device.index != runtime.device.index
        ):
            raise ValueError("prebuilt_tensor_graph is not on the runtime device.")
    # The host graph is the authority for objective/certificate hashes.  Exact
    # equality prevents an unrelated device graph with the same name and edge
    # count from being run under false provenance.  This is a D2H validation,
    # not an H2D re-upload; guided construction already materializes this host
    # spec for stable outputs.
    tensor_edge_u = (
        tensor_graph.edge_u.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    tensor_edge_v = (
        tensor_graph.edge_v.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    tensor_weight = (
        tensor_graph.weight.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    if not np.array_equal(tensor_edge_u, np.asarray(graph.edge_u, dtype=np.int64)):
        raise ValueError("prebuilt_tensor_graph edge_u does not match graph.")
    if not np.array_equal(tensor_edge_v, np.asarray(graph.edge_v, dtype=np.int64)):
        raise ValueError("prebuilt_tensor_graph edge_v does not match graph.")
    if not np.array_equal(tensor_weight, np.asarray(graph.edge_w, dtype=np.float64)):
        raise ValueError("prebuilt_tensor_graph weights do not match graph.")


def _project_state_dual(
    state: SolverState | None,
    *,
    runtime,
    edge_w: torch.Tensor,
    lambda_value: float,
    num_edges: int,
    num_regions: int,
) -> torch.Tensor | None:
    if state is None or state.dual is None:
        return None
    if tuple(state.dual.shape) != (int(num_edges), int(num_regions)):
        return None
    dual = state.dual.to(dtype=runtime.dtype, device=runtime.device)
    if int(num_edges) == 0:
        return torch.zeros(
            (0, int(num_regions)), dtype=runtime.dtype, device=runtime.device
        )
    radius = float(lambda_value) * edge_w.to(dtype=runtime.dtype, device=runtime.device)
    return project_dual_ball(dual, radius)


def _invalidate_damped_trial_state(
    *,
    phi: torch.Tensor,
    trial_warm_state: DenseWarmState | QuotientWorksetWarmState | PrimalOnlyWarmState,
) -> tuple[None, None, None, PrimalOnlyWarmState, bool]:
    """Create the only state that may be promoted for a damped MM endpoint."""

    structure_hint = (
        trial_warm_state.labels
        if isinstance(trial_warm_state, QuotientWorksetWarmState)
        else None
    )
    return (
        None,
        None,
        None,
        PrimalOnlyWarmState(
            phi=phi,
            structure_hint=structure_hint,
            structure_hint_is_heuristic=True,
        ),
        False,
    )


def _tensor_problem_from_torch_data(
    torch_data: TorchTumorData,
    *,
    major_prior: float,
    eps: float,
) -> TensorProblem:
    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    prior_tensor = torch.as_tensor(
        prior, dtype=torch_data.alt.dtype, device=torch_data.alt.device
    )
    return TensorProblem(
        alt=torch_data.alt,
        total=torch_data.total,
        nonalt=torch_data.nonalt,
        phi_upper=torch_data.phi_upper,
        ambiguous=torch_data.ambiguous,
        b_minus=torch_data.b_minus,
        b_plus=torch_data.b_plus,
        b_fixed=torch_data.b_fixed,
        eps=float(eps),
        major_prior=prior,
        log_prior_minor=torch.log1p(-prior_tensor),
        log_prior_major=torch.log(prior_tensor),
        count_observed=torch_data.count_observed,
    )


def torch_data_from_context(context: SolverContext) -> TorchTumorData:
    problem = context.problem
    return TorchTumorData(
        alt=problem.alt,
        total=problem.total,
        nonalt=problem.nonalt,
        phi_upper=problem.phi_upper,
        ambiguous=problem.ambiguous,
        b_minus=problem.b_minus,
        b_plus=problem.b_plus,
        b_fixed=problem.b_fixed,
        count_observed=problem.count_observed,
    )


def prepare_torch_problem(
    data: TumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    inner_max_iter: int,
    graph: PairwiseFusionGraph | None = None,
    prebuilt_tensor_graph: TensorFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor]
    | tuple[np.ndarray | torch.Tensor, ...]
    | None = None,
    device: str | None = "cuda",
    dtype: str | None = "float64",
    runtime=None,
    torch_data: TorchTumorData | None = None,
    objective_shape: str = "unimodal",
    defer_graph: bool = False,
) -> SolverContext:
    tol = _validate_solver_tolerance(tol)
    objective_shape = _normalize_objective_shape(objective_shape)
    use_unimodal_objective = objective_shape.startswith("unimodal")
    data_fingerprint = _data_fingerprint(data)
    effective_runtime = (
        resolve_runtime(device, dtype=dtype) if runtime is None else runtime
    )
    effective_torch_data = (
        to_torch_tumor_data(data, effective_runtime)
        if torch_data is None
        else torch_data
    )

    if exact_pilot is None:
        exact_pilot_tensor, secondary_wells, valid_secondary = (
            compute_scalar_mutation_region_wells_torch(
                effective_torch_data,
                phi_init=data.phi_init,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=tol,
                max_iter=max(int(inner_max_iter), 16),
            )
        )
    else:
        exact_pilot_tensor = _tensor_from_start(exact_pilot, effective_runtime)
        if scalar_well_starts is None and not use_unimodal_objective:
            _, secondary_wells, valid_secondary = (
                compute_scalar_mutation_region_wells_torch(
                    effective_torch_data,
                    phi_init=data.phi_init,
                    major_prior=float(major_prior),
                    eps=float(eps),
                    tol=tol,
                    max_iter=max(int(inner_max_iter), 16),
                )
            )
        else:
            secondary_wells = None
            valid_secondary = None

    if defer_graph:
        if graph is not None or prebuilt_tensor_graph is not None:
            raise ValueError(
                "defer_graph=True does not accept a resolved or prebuilt graph."
            )
        # Guided selection needs the likelihood pilot, bounds, and Torch data
        # before its observed-curvature graph is known.  Do not build and copy
        # an O(M^2) adaptive graph that would be discarded immediately.
        effective_graph = PairwiseFusionGraph(
            edge_u=np.zeros((0,), dtype=np.int32),
            edge_v=np.zeros((0,), dtype=np.int32),
            edge_w=np.zeros((0,), dtype=np.float64),
            name="deferred_likelihood_pilot",
            degree_bound=1,
        )
        tensor_graph = tensorize_graph(
            effective_graph,
            effective_runtime,
            num_nodes=data.num_mutations,
        )
    elif prebuilt_tensor_graph is not None:
        if graph is None:
            raise ValueError("prebuilt_tensor_graph requires its host graph spec.")
        effective_graph = resolve_pairwise_fusion_graph(
            data.num_mutations,
            graph=graph,
            pilot_phi=None,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        _validate_prebuilt_tensor_graph(
            effective_graph,
            prebuilt_tensor_graph,
            runtime=effective_runtime,
            num_nodes=data.num_mutations,
        )
        tensor_graph = prebuilt_tensor_graph
    elif graph is None:
        tensor_graph = build_complete_adaptive_tensor_graph(
            exact_pilot_tensor,
            effective_runtime,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        effective_graph = tensor_graph_to_pairwise_graph(tensor_graph)
    else:
        effective_graph = resolve_pairwise_fusion_graph(
            data.num_mutations,
            graph=graph,
            pilot_phi=None,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        tensor_graph = tensorize_graph(
            effective_graph, effective_runtime, num_nodes=data.num_mutations
        )

    if use_unimodal_objective and pooled_start is None:
        pooled_start_tensor = exact_pilot_tensor
    elif pooled_start is None:
        pooled_start_tensor = compute_pooled_observed_data_start_torch(
            effective_torch_data,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=tol,
            max_iter=max(int(inner_max_iter), 16),
            beta_hints=exact_pilot_tensor,
        )
    else:
        pooled_start_tensor = _tensor_from_start(pooled_start, effective_runtime)

    if use_unimodal_objective and scalar_well_starts is None:
        scalar_well_starts_seq = ()
    elif scalar_well_starts is None:
        scalar_well_starts_seq = compute_scalar_well_start_bank_torch(
            effective_torch_data,
            eps=float(eps),
            exact_pilot=exact_pilot_tensor,
            secondary_wells=secondary_wells,
            valid_secondary=valid_secondary,
        )
    else:
        scalar_well_starts_seq = list(scalar_well_starts)

    lower = torch.full_like(effective_torch_data.phi_upper, float(eps))
    upper = torch.minimum(
        effective_torch_data.phi_upper, torch.ones_like(effective_torch_data.phi_upper)
    )
    problem = _tensor_problem_from_torch_data(
        effective_torch_data,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    graph_hash = _graph_fingerprint(effective_graph)
    objective_spec_hash = _objective_spec_fingerprint(
        data_fingerprint=data_fingerprint,
        graph_hash=graph_hash,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    return SolverContext(
        problem=problem,
        graph=tensor_graph,
        graph_spec=effective_graph,
        exact_pilot=exact_pilot_tensor,
        pooled_start=pooled_start_tensor,
        scalar_well_starts=tuple(
            _tensor_from_start(start, effective_runtime)
            for start in scalar_well_starts_seq
        ),
        lower=lower,
        upper=upper,
        runtime=effective_runtime,
        data_fingerprint=data_fingerprint,
        graph_hash=graph_hash,
        objective_spec_hash=objective_spec_hash,
    )


def prepare_torch_problem_with_resource_policy(
    data: TumorData,
    *,
    dense_fallback_policy: str,
    inherited_resource_fallback: str | None = None,
    **prepare_kwargs,
) -> SolverContext:
    """Prepare a context with typed allocation failure and optional CPU retry.

    Model-selection code prepares graphs before it enters the fit API, so its
    allocations need the same exact fallback contract as a direct fit.
    """

    normalized_policy = str(dense_fallback_policy).strip().lower().replace("-", "_")
    if normalized_policy not in {"auto", "device_only", "cpu_allowed", "error"}:
        raise ValueError(
            "dense_fallback_policy must be auto, device_only, cpu_allowed, or error."
        )
    kwargs = dict(prepare_kwargs)
    supplied_prebuilt_tensor_graph = kwargs.pop("prebuilt_tensor_graph", None)
    supplied_runtime = kwargs.pop("runtime", None)
    supplied_torch_data = kwargs.pop("torch_data", None)
    requested_device = kwargs.pop("device", "cuda")
    requested_dtype = kwargs.pop("dtype", "float64")
    resolved_by_cpu_fallback = False
    try:
        requested_runtime = (
            resolve_runtime(requested_device, dtype=requested_dtype)
            if supplied_runtime is None
            else supplied_runtime
        )
    except RuntimeError:
        if normalized_policy != "cpu_allowed":
            raise
        try:
            requested_runtime = resolve_runtime("cpu", dtype=requested_dtype)
        except RuntimeError as cpu_runtime_error:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: the requested runtime is unavailable "
                "and dense CPU fallback does not support the requested dtype."
            ) from cpu_runtime_error
        resolved_by_cpu_fallback = True

    def prepare_on_runtime(*, retain_torch_data: bool) -> SolverContext:
        reusable_tensor_graph = supplied_prebuilt_tensor_graph
        if reusable_tensor_graph is not None and (
            reusable_tensor_graph.weight.device.type != requested_runtime.device.type
            or (
                requested_runtime.device.index is not None
                and reusable_tensor_graph.weight.device.index
                != requested_runtime.device.index
            )
            or reusable_tensor_graph.weight.dtype != requested_runtime.dtype
        ):
            # A resource-policy runtime change invalidates only the device copy;
            # the paired host graph remains available for exact tensorization.
            reusable_tensor_graph = None
        context = prepare_torch_problem(
            data,
            device=requested_runtime.device_name,
            dtype=dtype_name(requested_runtime.dtype),
            runtime=requested_runtime,
            torch_data=supplied_torch_data if retain_torch_data else None,
            prebuilt_tensor_graph=reusable_tensor_graph,
            **kwargs,
        )
        fallback = (
            "dense_cpu" if resolved_by_cpu_fallback else inherited_resource_fallback
        )
        return replace(context, resource_fallback=fallback)

    if resolved_by_cpu_fallback:
        cpu_fits, cpu_bytes, cpu_limit = dense_complete_solver_memory_preflight(
            num_nodes=int(data.num_mutations),
            num_regions=int(data.num_regions),
            runtime=requested_runtime,
        )
        if not cpu_fits:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: dense CPU fallback needs "
                f"approximately {cpu_bytes} bytes (available host limit: "
                f"{cpu_limit})."
            )
    try:
        return prepare_on_runtime(retain_torch_data=not resolved_by_cpu_fallback)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        cpu_fallback_allowed = bool(
            normalized_policy == "cpu_allowed"
            and requested_runtime.device.type != "cpu"
        )
        if not cpu_fallback_allowed:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: exact problem or graph construction "
                f"exhausted memory on {requested_runtime.device_name}."
            ) from exc
        try:
            requested_runtime = resolve_runtime(
                "cpu", dtype=dtype_name(requested_runtime.dtype)
            )
        except RuntimeError as cpu_runtime_error:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: dense CPU fallback does not support "
                f"dtype {dtype_name(requested_runtime.dtype)}."
            ) from cpu_runtime_error
        cpu_fits, cpu_bytes, cpu_limit = dense_complete_solver_memory_preflight(
            num_nodes=int(data.num_mutations),
            num_regions=int(data.num_regions),
            runtime=requested_runtime,
        )
        if not cpu_fits:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: dense CPU fallback needs "
                f"approximately {cpu_bytes} bytes (available host limit: "
                f"{cpu_limit})."
            ) from exc
        resolved_by_cpu_fallback = True
        try:
            return prepare_on_runtime(retain_torch_data=False)
        except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: exact problem or graph construction "
                "exhausted host memory during dense CPU fallback."
            ) from cpu_exc


def _initial_outer_diag() -> dict[str, float | int]:
    """Default outer-KKT diagnostics (all residuals +inf, all counts 0) used until
    the first audit fills them in."""
    return {
        "stationarity_residual": np.inf,
        "projected_stationarity_residual": np.inf,
        "projected_stationarity_norm": np.inf,
        "stationarity_normalizer": np.inf,
        "smooth_gradient_norm": np.inf,
        "fusion_adjustment_norm": np.inf,
        "edge_subgradient_residual": np.inf,
        "dual_ball_residual": np.inf,
        "box_primal_violation": np.inf,
        "num_interior_coordinates": 0,
        "num_lower_active_coordinates": 0,
        "num_upper_active_coordinates": 0,
        "num_frozen_coordinates": 0,
        "box_residual": np.inf,
        "kkt_residual": np.inf,
    }


def _multiplicity_calls(
    data: TumorData,
    gamma_np: np.ndarray,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve per-mutation_region major-copy probability, the boolean major call, and the
    chosen multiplicity, holding non-ambiguous mutation_regions at their fixed value."""
    major_probability = np.where(
        data.multiplicity_estimation_mask,
        gamma_np,
        1.0,
    ).astype(dtype, copy=False)
    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        data.multiplicity_estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(dtype, copy=False)
    return major_probability, major_call, multiplicity_call


def _classify_failure_reason(
    *,
    converged: bool,
    selection_eligible: bool,
    accepted_outer_steps: int,
    accepted_damped_steps: int,
    accepted_full_steps: int,
    mm_consistency_violations: int,
    objective: float,
    fit_loss: float,
    kkt_residual: float,
    tol: float,
    failed_nonfinite_checks: int,
    attempted_outer_steps: int,
    failed_inner_model_checks: int,
    failed_majorization_checks: int,
    failed_em_envelope_checks: int,
    failed_descent_checks: int,
    converged_outer: bool,
    converged_inner: bool,
) -> str:
    """Map the terminal solver state to a human-readable failure/convergence label."""
    if converged:
        return "converged"
    if selection_eligible and accepted_outer_steps == 0:
        return "start_already_stationary"
    if selection_eligible and accepted_damped_steps > 0 and accepted_full_steps == 0:
        return "fixed_objective_kkt_certified_after_damped_steps"
    if selection_eligible:
        return "fixed_objective_kkt_certified"
    if mm_consistency_violations > 0:
        return "mm_consistency_violation"
    if accepted_outer_steps == 0:
        final_penalty = float(objective - fit_loss)
        if final_penalty <= 1e-8 and kkt_residual > 5.0 * tol:
            return "pooled_start_not_stationary_no_descent_step_found"
        if (
            failed_nonfinite_checks > 0
            and failed_nonfinite_checks >= attempted_outer_steps
        ):
            return "all_trials_nonfinite"
        if failed_inner_model_checks > 0 and failed_inner_model_checks >= max(
            attempted_outer_steps - failed_nonfinite_checks, 1
        ):
            return "all_trials_failed_inner_model_decrease"
        if (
            failed_majorization_checks > 0
            and failed_descent_checks == 0
            and failed_em_envelope_checks == 0
        ):
            return "all_trials_failed_majorization"
        if failed_em_envelope_checks > 0 and failed_descent_checks == 0:
            return "all_trials_failed_em_envelope"
        if failed_descent_checks > 0:
            return "all_trials_failed_exact_descent"
        return "no_accepted_outer_step"
    if accepted_damped_steps > 0 and accepted_full_steps == 0:
        return "only_damped_steps_accepted"
    if not converged_outer:
        return "outer_stationarity_residual_above_tolerance"
    if not converged_inner:
        return "inner_kkt_residual_above_tolerance"
    return "outer_stopping_criteria_not_met"


def _solve_inner_subproblem(
    *,
    use_alm: bool,
    runtime,
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
    inner_max_iter: int,
    phi: torch.Tensor,
    dual,
    dual_start_is_actual: bool,
    spectral_rho: bool,
    pdhg_tau_node,
    backend_name: str,
    graph_hash: str,
    backend_mode: str,
    tensor_graph: TensorFusionGraph,
    warm_state,
    certificate_options: CertificateOptions,
    partition_tolerance: float,
    dense_fallback_policy: str,
) -> InnerSolveResult:
    """Dispatch the majorized inner subproblem to the ALM (complete-graph) or PDHG
    solver and wrap its legacy tuple in a representation-aware result."""
    if use_alm and backend_mode == "dense":
        dense_fits, dense_bytes, dense_limit = dense_complete_solver_memory_preflight(
            num_nodes=num_mutations,
            num_regions=int(U.shape[1]),
            runtime=runtime,
        )
        if not dense_fits:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: dense complete-graph solve needs "
                f"approximately {dense_bytes} bytes (available policy limit: "
                f"{dense_limit})."
            )
    if use_alm and backend_mode in {"auto", "quotient_workset"}:
        quotient_result, quotient_attempt_work = (
            solve_majorized_subproblem_quotient_workset_torch(
                runtime=runtime,
                U=U,
                h=h,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                graph=tensor_graph,
                graph_hash=graph_hash,
                tol=tol,
                max_iter=max(inner_max_iter, 10),
                phi_start=phi,
                warm_state=(
                    warm_state
                    if isinstance(
                        warm_state, (QuotientWorksetWarmState, PrimalOnlyWarmState)
                    )
                    else None
                ),
                certificate_options=certificate_options,
                partition_tolerance=partition_tolerance,
            )
        )
        if quotient_result is not None:
            return quotient_result
        if dense_fallback_policy == "error":
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: quotient/workset could not certify "
                "this inner problem and dense fallback is disabled by policy."
            )
        dense_fits, dense_bytes, dense_limit = dense_complete_solver_memory_preflight(
            num_nodes=num_mutations,
            num_regions=int(U.shape[1]),
            runtime=runtime,
        )
        if not dense_fits:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: quotient/workset was not certified "
                f"and dense fallback needs approximately {dense_bytes} bytes "
                f"(available policy limit: {dense_limit})."
            )

    surrogate_diag_values: dict[str, float | int] = {}
    if use_alm:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            inner_iterations,
            inner_ok,
            inner_residual,
        ) = solve_majorized_subproblem_alm_torch(
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
            max_iter=max(inner_max_iter, 10),
            phi_start=phi,
            dual_start=dual,
            dual_start_is_actual=dual_start_is_actual,
            spectral_rho=bool(spectral_rho),
            diagnostics_out=surrogate_diag_values,
        )
    else:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            inner_iterations,
            inner_ok,
            inner_residual,
        ) = solve_majorized_subproblem_pdhg_torch(
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
            degree_bound=degree_bound,
            tol=tol,
            max_iter=max(inner_max_iter, 10),
            phi_start=phi,
            dual_start=dual,
            tau_node=pdhg_tau_node,
        )
    if use_alm:
        # The outer MM loop carries the rho-invariant actual multiplier y.
        # Drop the low-level scaled-u return here so a second complete
        # edge-by-region tensor does not remain live through outer scoring and
        # certificate refinement.
        dual_trial = dual_kkt_trial
    if surrogate_diag_values:
        surrogate_diag = surrogate_diag_values
    else:
        surrogate_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi_trial,
            grad_smooth=h * (phi_trial - U),
            dual_kkt=dual_kkt_trial,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=tol,
        )
    certificate = (
        DenseEdgeCertificate(
            dual=dual_kkt_trial,
            graph_hash=str(graph_hash),
            gradient_scope="mm_surrogate",
        )
        if torch.is_tensor(dual_kkt_trial)
        else None
    )
    quotient_attempt_work = (
        quotient_attempt_work
        if use_alm and backend_mode in {"auto", "quotient_workset"}
        else BackendWorkCounters()
    )
    dense_iterations = int(inner_iterations) if use_alm else 0
    total_inner_iterations = (
        int(inner_iterations)
        + int(quotient_attempt_work.quotient_iterations)
        + int(quotient_attempt_work.workset_iterations)
    )
    return InnerSolveResult(
        phi=phi_trial,
        backend_name=str(backend_name),
        warm_state=DenseWarmState(
            phi=phi_trial,
            dual=dual_trial if torch.is_tensor(dual_trial) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        ),
        surrogate_certificate=certificate,
        surrogate_kkt=KKTDiagnostics.from_mapping(surrogate_diag),
        converged=bool(inner_ok),
        inner_iterations=total_inner_iterations,
        backend_iterations=total_inner_iterations,
        work_counters=BackendWorkCounters(
            quotient_iterations=int(quotient_attempt_work.quotient_iterations),
            workset_iterations=int(quotient_attempt_work.workset_iterations),
            workset_expansions=int(quotient_attempt_work.workset_expansions),
            streamed_edge_passes=int(quotient_attempt_work.streamed_edge_passes),
            dense_iterations=dense_iterations,
        ),
        fallback_reason=(
            "dense_current_device_after_quotient_attempt"
            if (
                use_alm
                and backend_mode in {"auto", "quotient_workset"}
                and lambda_value > 0.0
            )
            else ""
        ),
    )


def _fit_from_start(
    data: TumorData,
    *,
    torch_data,
    runtime,
    graph: PairwiseFusionGraph,
    tensor_graph: TensorFusionGraph,
    graph_hash: str,
    objective_spec_hash: str,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | torch.Tensor,
    solver_state: SolverState | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    summary_tol: float | None,
    compute_summary: bool,
    objective_shape: str,
    inner_backend: str,
    workset_max_bytes: int,
    compressed_cache_max_bytes: int,
    dense_fallback_policy: str,
    workset_add_batch: int,
    workset_max_expansions: int,
    certificate_max_iter: int,
    certificate_refinement_rounds: int,
    certificate_column_tol_scale: float,
    allow_heuristic_structure_splits: bool,
    verbose: bool,
) -> FusionFitArtifacts:
    tol = _validate_solver_tolerance(tol)
    objective_shape = _normalize_objective_shape(objective_shape)
    requested_inner_backend = _normalize_inner_backend(inner_backend)
    normalized_fallback_policy = (
        str(dense_fallback_policy).strip().lower().replace("-", "_")
    )
    if normalized_fallback_policy not in {
        "auto",
        "device_only",
        "cpu_allowed",
        "error",
    }:
        raise ValueError(
            "dense_fallback_policy must be auto, device_only, cpu_allowed, or error."
        )
    certificate_options = CertificateOptions(
        max_iter=max(int(certificate_max_iter), 1),
        refinement_rounds=max(int(certificate_refinement_rounds), 0),
        max_expansions=max(int(workset_max_expansions), 1),
        add_batch=max(int(workset_add_batch), 1),
        mapping_tolerance=max(0.1 * float(tol), float(torch.finfo(runtime.dtype).eps)),
        column_tolerance=max(
            float(certificate_column_tol_scale) * float(tol),
            float(torch.finfo(runtime.dtype).eps),
        ),
        memory=WorksetMemoryOptions(
            max_workset_bytes=int(workset_max_bytes),
            max_compressed_cache_bytes=int(compressed_cache_max_bytes),
            dense_fallback_policy=normalized_fallback_policy,
            allow_heuristic_split_before_dense_fallback=bool(
                allow_heuristic_structure_splits
            ),
        ),
    )
    use_unimodal_objective = objective_shape.startswith("unimodal")
    require_full_step_backtracking = (
        objective_shape == "unimodal_full_step_backtracking"
    )
    edge_u_np = graph.edge_u
    edge_v_np = graph.edge_v
    use_alm = bool(
        tensor_graph.is_complete
        and int(graph.degree_bound) == max(int(data.num_mutations) - 1, 1)
    )
    if requested_inner_backend == "quotient_workset" and not use_alm:
        raise ValueError(
            "inner_backend='quotient_workset' requires the complete original graph."
        )
    edge_u, edge_v, edge_w = (
        tensor_graph.edge_u,
        tensor_graph.edge_v,
        tensor_graph.weight,
    )
    if lambda_value <= 0.0 or int(edge_u.numel()) == 0:
        dense_inner_solver = "closed_form_projection"
    elif use_alm:
        # The complete-graph ALM backend is the scaled-dual ADMM algorithm:
        # group shrinkage, constrained phi update, then dual ascent.
        dense_inner_solver = "admm_complete_graph"
    else:
        dense_inner_solver = "pdhg"
    inner_solver = dense_inner_solver
    use_compressed_certificates = bool(
        requested_inner_backend in {"auto", "quotient_workset"}
        and tensor_graph.is_complete
        and lambda_value > 0.0
    )

    if (
        solver_state is not None
        and solver_state.phi is not None
        and tuple(solver_state.phi.shape) == tuple(torch_data.phi_upper.shape)
    ):
        phi = solver_state.phi.to(dtype=runtime.dtype, device=runtime.device)
    else:
        phi = _tensor_from_start(phi_start, runtime)
    phi = torch.minimum(torch.maximum(phi, lower), upper)

    state_dual = (
        None
        if use_compressed_certificates
        else _project_state_dual(
            solver_state,
            runtime=runtime,
            edge_w=edge_w,
            lambda_value=lambda_value,
            num_edges=int(edge_u.numel()),
            num_regions=int(phi.shape[1]),
        )
    )
    dual = state_dual
    dual_kkt = state_dual
    warm_state = (
        solver_state.warm_state
        if solver_state is not None and solver_state.warm_state is not None
        else DenseWarmState(
            phi=phi,
            dual=state_dual,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    )
    certificate = (
        solver_state.certificate
        if (
            solver_state is not None
            and solver_state.certificate is not None
            and getattr(solver_state.certificate, "graph_hash", None) == graph_hash
        )
        else (
            DenseEdgeCertificate(
                dual=state_dual,
                graph_hash=graph_hash,
                gradient_scope="observed_objective",
            )
            if torch.is_tensor(state_dual)
            else None
        )
    )
    if use_compressed_certificates and not isinstance(
        certificate, CompressedEdgeCertificate
    ):
        certificate = compressed_certificate_for_primal(
            phi,
            graph_hash=graph_hash,
            gradient_scope="observed_objective",
        )
    dual_start_is_actual = bool(use_alm and state_dual is not None)
    history: list[float] = []
    converged = False
    converged_inner = False
    converged_outer = False
    iterations = 0
    inner_iterations = 0
    quotient_iterations = 0
    workset_iterations = 0
    workset_expansions = 0
    streamed_edge_passes = 0
    dense_iterations = 0
    fallback_reason = ""
    current_inner_converged = False
    current_inner_kkt_residual = np.nan
    final_relative_objective_change = np.inf
    final_step_residual = np.inf
    final_inner_kkt_residual = np.nan
    final_outer_diag = _initial_outer_diag()
    outer_kkt_certificate_status = "not_audited"
    outer_kkt_dual_refined = False
    outer_kkt_fused_edges = 0
    outer_kkt_nonzero_edges = 0
    outer_stationarity_residual_before_dual_refine = np.inf
    outer_stationarity_residual_after_dual_refine = np.inf
    accepted_outer_steps = 0
    accepted_full_steps = 0
    accepted_damped_steps = 0
    attempted_outer_steps = 0
    failed_majorization_checks = 0
    failed_inner_model_checks = 0
    failed_em_envelope_checks = 0
    failed_descent_checks = 0
    failed_nonfinite_checks = 0
    mm_consistency_violations = 0
    last_attempted_inner_kkt_residual = np.nan
    best_attempted_inner_kkt_residual = np.nan
    last_attempted_objective_gap = np.nan
    best_attempted_objective_gap = np.nan
    last_attempted_surrogate_gap = np.nan
    best_attempted_surrogate_gap = np.nan
    last_attempted_inner_model_gap = np.nan
    best_attempted_inner_model_gap = np.nan
    last_attempted_em_envelope_gap = np.nan
    best_attempted_em_envelope_gap = np.nan
    accepted_step_type = "none"
    last_reject_reason = "not_attempted"
    full_step_curvature_multiplier = torch.ones_like(phi)

    current_mutation_region_terms = mutation_region_terms_torch(
        torch_data, phi, major_prior=major_prior, eps=eps
    )
    fit_loss, penalty, objective, gamma_major = (
        _objective_value_from_mutation_region_terms_torch(
            current_mutation_region_terms,
            phi,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
        )
    )
    history.append(float(objective))

    for outer_iter in range(max(int(outer_max_iter), 1)):
        iterations = outer_iter + 1
        previous_phi = phi.clone()
        previous_objective = objective
        if use_unimodal_objective:
            surrogate_terms = current_mutation_region_terms
            surrogate_fit_loss = float(fit_loss)
        else:
            surrogate_terms = em_surrogate_terms_torch(
                torch_data,
                phi,
                omega_major=gamma_major,
                major_prior=major_prior,
                eps=eps,
            )
            surrogate_fit_loss = float(torch.sum(surrogate_terms.loss).item())
        h_base, surrogate_grad = _safe_surrogate_curvature_and_gradient(
            surrogate_terms,
            torch_data.count_observed,
        )
        if require_full_step_backtracking:
            forcing_certificate = certificate
            if use_compressed_certificates and forcing_certificate is None:
                forcing_certificate = compressed_certificate_for_primal(
                    phi,
                    graph_hash=graph_hash,
                    gradient_scope="observed_objective",
                )
            forcing_diag = audit_graph_fusion_certificate(
                certificate=forcing_certificate,
                phi=phi,
                grad_smooth=current_mutation_region_terms.grad,
                graph=tensor_graph,
                graph_hash=graph_hash,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                atol=tol,
            ).as_dict()
            inner_progress_tolerance = max(
                5.0 * tol,
                min(
                    float(np.sqrt(tol)),
                    0.9 * float(forcing_diag["kkt_residual"]),
                ),
            )
        else:
            inner_progress_tolerance = 5.0 * tol
        scale = 1.0
        curvature_multiplier = full_step_curvature_multiplier
        accepted = False
        candidate_phi = phi
        candidate_dual = dual
        candidate_dual_kkt = dual_kkt
        candidate_certificate = certificate
        candidate_warm_state = warm_state
        candidate_backend_name = inner_solver
        candidate_dual_start_is_actual = dual_start_is_actual
        candidate_objective = objective
        candidate_fit_loss = fit_loss
        candidate_gamma = gamma_major
        candidate_mutation_region_terms = current_mutation_region_terms
        candidate_inner_residual = np.nan
        candidate_step_type = "none"
        inner_converged = False

        curvature_attempts = (
            _FULL_STEP_MAX_CURVATURE_ATTEMPTS
            if require_full_step_backtracking
            else (1 if use_unimodal_objective else 10)
        )
        for _curvature_attempt in range(curvature_attempts):
            h = (
                h_base * curvature_multiplier
                if require_full_step_backtracking
                else h_base * scale
            )
            U = _safe_majorized_center(
                phi,
                surrogate_grad=surrogate_grad,
                h=h,
                count_observed=torch_data.count_observed,
            )
            if use_unimodal_objective and not require_full_step_backtracking:
                q_current = None
            else:
                q_current = _inner_model_value_torch(
                    phi,
                    U=U,
                    h=h,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
            recovery_inner_model_tol = (
                max(
                    64.0 * float(torch.finfo(phi.dtype).eps),
                    float(tol) ** 2,
                )
                * (1.0 + abs(float(q_current.item())))
                if require_full_step_backtracking
                else 0.0
            )
            inner_phi_start = phi
            inner_dual_start = dual
            inner_dual_start_is_actual = dual_start_is_actual
            inner_warm_start = warm_state
            attempted_inner_iterations = 0
            inner_batch_limit = 8 if require_full_step_backtracking else 1
            for _inner_batch in range(inner_batch_limit):
                inner_result = _solve_inner_subproblem(
                    use_alm=use_alm,
                    runtime=runtime,
                    num_mutations=data.num_mutations,
                    U=U,
                    h=h,
                    lower=lower,
                    upper=upper,
                    lambda_value=lambda_value,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    degree_bound=int(graph.degree_bound),
                    tol=tol,
                    inner_max_iter=inner_max_iter,
                    phi=inner_phi_start,
                    dual=inner_dual_start,
                    dual_start_is_actual=inner_dual_start_is_actual,
                    spectral_rho=bool(require_full_step_backtracking),
                    pdhg_tau_node=tensor_graph.pdhg_tau_node,
                    backend_name=dense_inner_solver,
                    graph_hash=graph_hash,
                    backend_mode=requested_inner_backend,
                    tensor_graph=tensor_graph,
                    warm_state=inner_warm_start,
                    certificate_options=certificate_options,
                    partition_tolerance=max(float(tol), 1e-12),
                    dense_fallback_policy=normalized_fallback_policy,
                )
                quotient_iterations += int(
                    inner_result.work_counters.quotient_iterations
                )
                workset_iterations += int(inner_result.work_counters.workset_iterations)
                workset_expansions += int(inner_result.work_counters.workset_expansions)
                streamed_edge_passes += int(
                    inner_result.work_counters.streamed_edge_passes
                )
                dense_iterations += int(inner_result.work_counters.dense_iterations)
                fallback_reason = _combine_fallback_reasons(
                    fallback_reason,
                    inner_result.fallback_reason,
                )
                phi_trial = inner_result.phi
                dense_warm_state = inner_result.warm_state
                dual_trial = getattr(dense_warm_state, "dual", None)
                surrogate_certificate = inner_result.surrogate_certificate
                dual_kkt_trial = getattr(surrogate_certificate, "dual", None)
                batch_inner_iterations = int(inner_result.inner_iterations)
                inner_ok = bool(inner_result.converged)
                inner_residual = float(inner_result.surrogate_kkt.kkt_residual)
                attempted_inner_iterations += int(batch_inner_iterations)
                batch_inner_certified = bool(inner_ok)
                if require_full_step_backtracking:
                    batch_inner_certified = bool(
                        np.isfinite(float(inner_residual))
                        and float(inner_residual) <= inner_progress_tolerance
                    )
                    batch_q_trial = _inner_model_value_torch(
                        phi_trial,
                        U=U,
                        h=h,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
                    )
                    batch_inner_model_gap = float((batch_q_trial - q_current).item())
                    batch_inner_certified = bool(
                        batch_inner_certified
                        and np.isfinite(batch_inner_model_gap)
                        and batch_inner_model_gap <= recovery_inner_model_tol
                    )
                if batch_inner_certified:
                    inner_ok = True
                    break
                inner_phi_start = phi_trial
                inner_dual_start = dual_kkt_trial if use_alm else dual_trial
                inner_dual_start_is_actual = bool(use_alm)
                inner_warm_start = inner_result.warm_state
            inner_iterations += int(attempted_inner_iterations)
            attempted_outer_steps += 1
            last_attempted_inner_kkt_residual = float(inner_residual)
            best_attempted_inner_kkt_residual = _update_minimum(
                float(best_attempted_inner_kkt_residual),
                float(inner_residual),
            )
            delta = phi_trial - phi
            trial_mutation_region_terms = mutation_region_terms_torch(
                torch_data, phi_trial, major_prior=major_prior, eps=eps
            )
            trial_fit_loss, _, trial_objective, trial_gamma = (
                _objective_value_from_mutation_region_terms_torch(
                    trial_mutation_region_terms,
                    phi_trial,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
            )
            objective_gap = float(trial_objective - previous_objective)
            audit_quadratic_majorizer = bool(
                require_full_step_backtracking or not use_unimodal_objective
            )
            if not audit_quadratic_majorizer:
                inner_model_gap = 0.0
                surrogate_gap = 0.0
                em_envelope_gap = 0.0
            else:
                quadratic_gap = float(
                    torch.sum(
                        surrogate_terms.grad * delta + 0.5 * h * torch.square(delta)
                    ).item()
                )
                majorizer_rhs = surrogate_fit_loss + quadratic_gap
                q_trial = _inner_model_value_torch(
                    phi_trial,
                    U=U,
                    h=h,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
                inner_model_gap = float((q_trial - q_current).item())
                if use_unimodal_objective:
                    # Recovery uses an exact smooth-loss majorization check.
                    # This makes the accepted full ADMM endpoint a valid
                    # proximal-MM update with a matching actual dual.
                    surrogate_gap = float(trial_fit_loss - majorizer_rhs)
                    em_envelope_gap = 0.0
                else:
                    trial_surrogate_terms = em_surrogate_terms_torch(
                        torch_data,
                        phi_trial,
                        omega_major=gamma_major,
                        major_prior=major_prior,
                        eps=eps,
                    )
                    trial_surrogate_loss = float(
                        torch.sum(trial_surrogate_terms.loss).item()
                    )
                    surrogate_gap = float(trial_surrogate_loss - majorizer_rhs)
                    em_envelope_gap = float(
                        (trial_fit_loss - fit_loss)
                        - (trial_surrogate_loss - surrogate_fit_loss)
                    )
            last_attempted_objective_gap = objective_gap
            best_attempted_objective_gap = _update_minimum(
                float(best_attempted_objective_gap), objective_gap
            )
            last_attempted_surrogate_gap = surrogate_gap
            best_attempted_surrogate_gap = _update_minimum(
                float(best_attempted_surrogate_gap), surrogate_gap
            )
            last_attempted_inner_model_gap = inner_model_gap
            best_attempted_inner_model_gap = _update_minimum(
                float(best_attempted_inner_model_gap), inner_model_gap
            )
            last_attempted_em_envelope_gap = em_envelope_gap
            best_attempted_em_envelope_gap = _update_minimum(
                float(best_attempted_em_envelope_gap), em_envelope_gap
            )
            finite_attempt = all(
                np.isfinite(value)
                for value in [
                    inner_model_gap,
                    surrogate_gap,
                    em_envelope_gap,
                    objective_gap,
                    trial_fit_loss,
                    trial_objective,
                ]
            )
            if require_full_step_backtracking:
                numerical_factor = 64.0 * float(torch.finfo(phi.dtype).eps)
                inner_model_tol = max(numerical_factor, float(tol) ** 2) * (
                    1.0 + abs(float(q_current.item()))
                )
                majorization_tol = max(numerical_factor, float(tol) ** 2) * (
                    1.0 + abs(surrogate_fit_loss)
                )
                objective_tol = numerical_factor * (1.0 + abs(previous_objective))
            else:
                inner_model_tol = (
                    1e-8 * (1.0 + abs(float(q_current.item())))
                    if audit_quadratic_majorizer
                    else 0.0
                )
                majorization_tol = 1e-8 * (1.0 + abs(surrogate_fit_loss))
                objective_tol = 1e-8 * (1.0 + abs(previous_objective))
            envelope_tol = 1e-8 * (1.0 + abs(fit_loss))
            if not finite_attempt:
                failed_nonfinite_checks += 1
                last_reject_reason = "rejected_nonfinite_objective"
                scale *= 2.0
                if require_full_step_backtracking:
                    curvature_multiplier = torch.clamp(
                        2.0 * curvature_multiplier,
                        max=1e12,
                    )
                    full_step_curvature_multiplier = curvature_multiplier
                continue
            if audit_quadratic_majorizer and inner_model_gap > inner_model_tol:
                failed_inner_model_checks += 1
                last_reject_reason = "rejected_inner_model_not_decreased"
                if require_full_step_backtracking:
                    break
                scale *= 2.0
                continue
            if (
                audit_quadratic_majorizer
                and not require_full_step_backtracking
                and surrogate_gap > majorization_tol
            ):
                failed_majorization_checks += 1
                last_reject_reason = "rejected_majorization_failed"
                scale *= 2.0
                continue
            if not use_unimodal_objective and em_envelope_gap > envelope_tol:
                failed_em_envelope_checks += 1
                last_reject_reason = "rejected_em_envelope_failed"
                scale *= 2.0
                continue
            if require_full_step_backtracking and not (
                np.isfinite(float(inner_residual))
                and float(inner_residual) <= inner_progress_tolerance
            ):
                failed_inner_model_checks += 1
                last_reject_reason = "rejected_uncertified_inner_admm_step"
                break
            recovery_armijo_rhs = (
                1e-4 * min(float(inner_model_gap), 0.0) + objective_tol
                if require_full_step_backtracking
                else objective_tol
            )
            if objective_gap <= recovery_armijo_rhs:
                accepted = True
                accepted_outer_steps += 1
                accepted_full_steps += 1
                candidate_phi = phi_trial
                # The complete-graph ADMM backend also returns the actual KKT
                # multiplier y=rho*u. Carry y, not the rho-dependent scaled u,
                # across outer MM subproblems because curvature changes rho.
                candidate_dual = dual_kkt_trial if use_alm else dual_trial
                candidate_dual_kkt = dual_kkt_trial
                candidate_certificate = surrogate_certificate
                candidate_warm_state = inner_result.warm_state
                candidate_backend_name = inner_result.backend_name
                candidate_dual_start_is_actual = bool(use_alm)
                candidate_objective = trial_objective
                candidate_fit_loss = trial_fit_loss
                candidate_gamma = trial_gamma
                candidate_mutation_region_terms = trial_mutation_region_terms
                candidate_inner_residual = float(inner_residual)
                candidate_step_type = "full_inner_step"
                accepted_step_type = candidate_step_type
                inner_converged = bool(
                    (
                        np.isfinite(float(inner_residual))
                        and float(inner_residual) <= 5.0 * tol
                    )
                    if require_full_step_backtracking
                    else (
                        inner_ok
                        and np.isfinite(float(inner_residual))
                        and float(inner_residual) <= 5.0 * tol
                    )
                )
                if require_full_step_backtracking:
                    # Retain coordinate-wise curvature evidence while trying a
                    # less conservative metric at the next accepted iterate.
                    full_step_curvature_multiplier = torch.clamp(
                        0.5 * curvature_multiplier,
                        min=1.0,
                        max=1e12,
                    )
                break
            if require_full_step_backtracking:
                # A damped primal point does not share the full subproblem's
                # dual certificate. Enlarge the persistent majorizing
                # curvature and accept only a full proximal-MM/ADMM endpoint.
                # If the resource limit is exhausted, leave this outer iterate
                # unchanged and uncertified rather than interpolating phi.
                failed_descent_checks += 1
                last_reject_reason = "rejected_full_step_for_curvature_backtracking"
                delta_square = torch.square(delta)
                resolution = torch.finfo(phi.dtype).eps * (1.0 + torch.square(phi))
                secant_remainder = (
                    trial_mutation_region_terms.loss
                    - current_mutation_region_terms.loss
                    - surrogate_grad * delta
                )
                required_h = torch.where(
                    delta_square > resolution,
                    2.0
                    * torch.clamp(secant_remainder, min=0.0)
                    / torch.clamp(
                        delta_square,
                        min=torch.finfo(phi.dtype).tiny,
                    ),
                    h,
                )
                target_h = torch.maximum(h, 1.25 * required_h)
                proposed_multiplier = torch.clamp(
                    target_h
                    / torch.clamp(
                        h_base,
                        min=_MISSING_SURROGATE_CURVATURE,
                    ),
                    min=1.0,
                    max=1e12,
                )
                changed = bool(
                    torch.any(
                        proposed_multiplier
                        > curvature_multiplier
                        * (1.0 + 64.0 * torch.finfo(phi.dtype).eps)
                    ).item()
                )
                if changed:
                    curvature_multiplier = proposed_multiplier
                else:
                    curvature_multiplier = torch.clamp(
                        2.0 * curvature_multiplier,
                        max=1e12,
                    )
                full_step_curvature_multiplier = curvature_multiplier
                continue
            if (
                not use_unimodal_objective
                and inner_ok
                and inner_model_gap <= inner_model_tol
                and surrogate_gap <= majorization_tol
                and em_envelope_gap <= envelope_tol
                and objective_gap > max(1e-5, objective_tol)
            ):
                mm_consistency_violations += 1
            theta = 0.5
            damped_accepted = False
            for _line_search_iter in range(12):
                phi_theta = phi + theta * delta
                theta_mutation_region_terms = mutation_region_terms_torch(
                    torch_data, phi_theta, major_prior=major_prior, eps=eps
                )
                theta_fit_loss, _, theta_objective, theta_gamma = (
                    _objective_value_from_mutation_region_terms_torch(
                        theta_mutation_region_terms,
                        phi_theta,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
                    )
                )
                if (
                    np.isfinite(theta_objective)
                    and theta_objective <= previous_objective + objective_tol
                ):
                    accepted = True
                    damped_accepted = True
                    accepted_outer_steps += 1
                    accepted_damped_steps += 1
                    candidate_phi = phi_theta
                    (
                        candidate_dual,
                        candidate_dual_kkt,
                        candidate_certificate,
                        candidate_warm_state,
                        candidate_dual_start_is_actual,
                    ) = _invalidate_damped_trial_state(
                        phi=phi_theta,
                        trial_warm_state=inner_result.warm_state,
                    )
                    candidate_backend_name = inner_result.backend_name
                    candidate_objective = theta_objective
                    candidate_fit_loss = theta_fit_loss
                    candidate_gamma = theta_gamma
                    candidate_mutation_region_terms = theta_mutation_region_terms
                    candidate_inner_residual = np.nan
                    candidate_step_type = "damped_mm_direction"
                    accepted_step_type = candidate_step_type
                    inner_converged = False
                    break
                theta *= 0.5
            if damped_accepted:
                break
            failed_descent_checks += 1
            last_reject_reason = "rejected_exact_descent_failed"
            scale *= 2.0

        if not accepted:
            candidate_phi = phi
            candidate_dual = dual
            candidate_dual_kkt = dual_kkt
            candidate_certificate = certificate
            candidate_warm_state = warm_state
            candidate_backend_name = inner_solver
            candidate_dual_start_is_actual = dual_start_is_actual
            candidate_objective = objective
            candidate_fit_loss = fit_loss
            candidate_gamma = gamma_major
            candidate_mutation_region_terms = current_mutation_region_terms
            candidate_step_type = "none"

        phi = candidate_phi
        dual = candidate_dual
        dual_kkt = candidate_dual_kkt
        certificate = candidate_certificate
        warm_state = candidate_warm_state
        inner_solver = candidate_backend_name
        dual_start_is_actual = candidate_dual_start_is_actual
        objective = candidate_objective
        fit_loss = candidate_fit_loss
        gamma_major = candidate_gamma
        current_mutation_region_terms = candidate_mutation_region_terms
        penalty = objective - fit_loss
        history.append(float(objective))

        if verbose:
            print(
                f"[pairwise-fusion:{runtime.device_name}] iter={iterations:02d} objective={objective:.6f} "
                f"fit={fit_loss:.6f} penalty={penalty:.6f}"
            )

        rel_change = abs(previous_objective - objective) / (
            1.0 + abs(previous_objective)
        )
        step_residual = float(
            (
                torch.linalg.norm(phi - previous_phi)
                / (1.0 + torch.linalg.norm(previous_phi))
            ).item()
        )
        cheap_outer_converged = bool(
            rel_change <= 10.0 * tol and step_residual <= max(1e-8, np.sqrt(tol))
        )
        do_outer_kkt_audit = bool(
            cheap_outer_converged
            or iterations >= max(int(outer_max_iter), 1)
            or iterations % _OUTER_KKT_CHECK_EVERY == 0
            or not np.isfinite(objective)
        )
        outer_diag = final_outer_diag
        outer_converged = False
        if do_outer_kkt_audit:
            outer_terms = current_mutation_region_terms
            if inner_solver == "quotient_workset_complete_graph" or isinstance(
                certificate, CompressedEdgeCertificate
            ):
                observed_start = certificate
                if not isinstance(observed_start, CompressedEdgeCertificate):
                    observed_start = compressed_certificate_for_primal(
                        phi,
                        graph_hash=graph_hash,
                        gradient_scope="observed_objective",
                    )
                observed_refinement = refine_graph_fusion_certificate(
                    certificate=observed_start,
                    phi=phi,
                    grad_smooth=outer_terms.grad,
                    gradient_scope="observed_objective",
                    graph=tensor_graph,
                    graph_hash=graph_hash,
                    lower=lower,
                    upper=upper,
                    lambda_value=lambda_value,
                    atol=tol,
                    options=certificate_options,
                )
                workset_iterations += int(
                    observed_refinement.work_counters.workset_iterations
                )
                workset_expansions += int(
                    observed_refinement.work_counters.workset_expansions
                )
                streamed_edge_passes += int(
                    observed_refinement.work_counters.streamed_edge_passes
                )
                certificate = observed_refinement.certificate
                outer_diag = observed_refinement.diagnostics.as_dict()
            else:
                outer_diag = audit_graph_fusion_certificate(
                    certificate=certificate,
                    phi=phi,
                    grad_smooth=outer_terms.grad,
                    graph=tensor_graph,
                    graph_hash=graph_hash,
                    lower=lower,
                    upper=upper,
                    lambda_value=lambda_value,
                    atol=tol,
                ).as_dict()
            outer_converged = bool(outer_diag["kkt_residual"] <= 5.0 * tol)
        final_relative_objective_change = float(rel_change)
        final_step_residual = float(step_residual)
        if accepted:
            current_inner_converged = bool(inner_converged)
            current_inner_kkt_residual = float(candidate_inner_residual)
        final_inner_kkt_residual = float(current_inner_kkt_residual)
        if do_outer_kkt_audit:
            final_outer_diag = outer_diag
        converged_inner = bool(current_inner_converged)
        converged_outer = bool(outer_converged)
        if (
            rel_change <= tol
            and step_residual <= np.sqrt(tol)
            and current_inner_converged
            and outer_converged
        ):
            converged = True
            break

    final_terms = current_mutation_region_terms
    if inner_solver == "quotient_workset_complete_graph" and not isinstance(
        certificate, CompressedEdgeCertificate
    ):
        certificate = compressed_certificate_for_primal(
            phi,
            graph_hash=graph_hash,
            gradient_scope="observed_objective",
        )
    final_certificate_refinement = refine_graph_fusion_certificate(
        certificate=certificate,
        phi=phi,
        grad_smooth=final_terms.grad,
        gradient_scope="observed_objective",
        graph=tensor_graph,
        graph_hash=graph_hash,
        lower=lower,
        upper=upper,
        lambda_value=lambda_value,
        atol=tol,
        max_iter=96,
        options=(
            certificate_options
            if isinstance(certificate, CompressedEdgeCertificate)
            else None
        ),
    )
    workset_iterations += int(
        final_certificate_refinement.work_counters.workset_iterations
    )
    workset_expansions += int(
        final_certificate_refinement.work_counters.workset_expansions
    )
    streamed_edge_passes += int(
        final_certificate_refinement.work_counters.streamed_edge_passes
    )
    certificate = final_certificate_refinement.certificate
    final_outer_diag = final_certificate_refinement.diagnostics.as_dict()
    final_dual = getattr(certificate, "dual", None)
    outer_kkt_certificate_status = str(final_certificate_refinement.status)
    outer_kkt_dual_refined = bool(final_certificate_refinement.dual_refined)
    outer_kkt_fused_edges = int(final_certificate_refinement.fused_edges)
    outer_kkt_nonzero_edges = int(final_certificate_refinement.nonzero_edges)
    outer_stationarity_residual_before_dual_refine = float(
        final_certificate_refinement.stationarity_before
    )
    outer_stationarity_residual_after_dual_refine = float(
        final_certificate_refinement.stationarity_after
    )
    converged_outer = bool(float(final_outer_diag["kkt_residual"]) <= 5.0 * tol)
    valid_dual_certificate = outer_kkt_certificate_status in {
        "zero_penalty_no_dual_needed",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
        "input_dual_retained",
        "certified",
    }
    selection_eligible = bool(
        np.isfinite(float(objective))
        and converged_outer
        and valid_dual_certificate
        and mm_consistency_violations == 0
    )
    full_kkt_certified = bool(
        np.isfinite(float(final_outer_diag["kkt_residual"]))
        and converged_outer
        and valid_dual_certificate
    )
    exactness_provenance = ExactFusionProvenance(
        schema_version=1,
        estimator_role="raw_fused_lambda_path",
        objective_faithful=True,
        objective_spec_hash=str(objective_spec_hash),
        original_graph_hash=str(graph_hash),
        certificate_problem_hash=_certificate_problem_fingerprint(
            objective_spec_hash=objective_spec_hash,
            lambda_value=lambda_value,
        ),
        certificate_scope="full_original_graph",
        gradient_scope="observed_objective",
        full_kkt_certified=full_kkt_certified,
        status=str(outer_kkt_certificate_status),
        residual=float(final_outer_diag["kkt_residual"]),
        tolerance=5.0 * float(tol),
        backend_name=str(inner_solver),
        backend_iterations=int(inner_iterations),
        quotient_iterations=int(quotient_iterations),
        workset_iterations=int(workset_iterations),
        workset_expansions=int(workset_expansions),
        streamed_edge_passes=int(streamed_edge_passes),
        dense_iterations=int(dense_iterations),
        fallback_reason=str(fallback_reason),
    )
    stationarity_certified = bool(selection_eligible)
    global_optimality_certified = bool(selection_eligible and use_unimodal_objective)
    global_optimality_basis = (
        _UNIMODAL_GLOBAL_OPTIMALITY_BASIS
        if global_optimality_certified
        else "not_certified"
    )
    if use_unimodal_objective and global_optimality_certified:
        converged = True

    failure_reason = _classify_failure_reason(
        converged=converged,
        selection_eligible=selection_eligible,
        accepted_outer_steps=accepted_outer_steps,
        accepted_damped_steps=accepted_damped_steps,
        accepted_full_steps=accepted_full_steps,
        mm_consistency_violations=mm_consistency_violations,
        objective=float(objective),
        fit_loss=float(fit_loss),
        kkt_residual=float(final_outer_diag["kkt_residual"]),
        tol=tol,
        failed_nonfinite_checks=failed_nonfinite_checks,
        attempted_outer_steps=attempted_outer_steps,
        failed_inner_model_checks=failed_inner_model_checks,
        failed_majorization_checks=failed_majorization_checks,
        failed_em_envelope_checks=failed_em_envelope_checks,
        failed_descent_checks=failed_descent_checks,
        converged_outer=converged_outer,
        converged_inner=converged_inner,
    )

    phi_np = phi.detach().cpu().numpy()
    gamma_np = gamma_major.detach().cpu().numpy()
    effective_summary_tol = (
        max(10.0 * float(tol), 1e-4)
        if summary_tol is None
        else max(float(summary_tol), 1e-12)
    )
    cluster_labels = _cluster_labels(
        phi_np,
        edge_u=edge_u_np,
        edge_v=edge_v_np,
        tol=effective_summary_tol,
    )
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    cluster_diameters, cluster_diameter_exact = cluster_diameters_from_edges(
        phi_np,
        cluster_labels,
        edge_u=edge_u_np,
        edge_v=edge_v_np,
    )
    max_cluster_diameter = (
        float(np.max(cluster_diameters)) if cluster_diameters.size else 0.0
    )
    if compute_summary:
        cluster_centers, phi_clustered = _cluster_summary_from_labels(
            phi_np, cluster_labels
        )
        phi_clustered_torch = torch.as_tensor(
            phi_clustered, dtype=runtime.dtype, device=runtime.device
        )
        summary_fit_loss, _, _, _ = _objective_value_once_torch(
            torch_data,
            phi_clustered_torch,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=0.0,
            major_prior=major_prior,
            eps=eps,
        )
        summary_loglik = float(-summary_fit_loss)
    else:
        cluster_centers = np.zeros((n_clusters, phi_np.shape[1]), dtype=phi_np.dtype)
        phi_clustered = phi_np.astype(phi_np.dtype, copy=False)
        summary_loglik = float("nan")

    major_probability, major_call, multiplicity_call = _multiplicity_calls(
        data, gamma_np, phi_np.dtype
    )
    if isinstance(certificate, CompressedEdgeCertificate):
        quotient_dual = (
            warm_state.quotient_dual
            if isinstance(warm_state, QuotientWorksetWarmState)
            else None
        )
        terminal_warm_state = QuotientWorksetWarmState(
            phi=phi.detach(),
            labels=certificate.labels.detach(),
            centers=certificate.centers.detach(),
            quotient_dual=(
                quotient_dual.detach() if torch.is_tensor(quotient_dual) else None
            ),
            internal_edge_ids=certificate.internal_edge_ids.detach(),
            internal_dual=certificate.internal_dual.detach(),
            graph_hash=str(graph_hash),
            previous_lambda=float(lambda_value),
        )
    else:
        terminal_warm_state = DenseWarmState(
            phi=phi.detach(),
            dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    solver_state_out = SolverState(
        phi=phi.detach(),
        dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
        # No terminal ADMM split variable is exported. ``dual`` is the actual
        # KKT multiplier used for warm starts; duplicating it in ``split``
        # wastes a complete edge-by-region tensor and mislabels its units.
        split=None,
        curvature=None,
        previous_lambda=float(lambda_value),
        warm_state=terminal_warm_state,
        certificate=certificate,
    )
    torch_result = TorchFitResult(
        phi_raw=phi.detach(),
        gamma_major=final_terms.gamma_major.detach(),
        dual=solver_state_out.dual,
        fit_loss=torch.as_tensor(
            float(fit_loss), dtype=runtime.dtype, device=runtime.device
        ),
        fusion_penalty=torch.as_tensor(
            float(objective - fit_loss), dtype=runtime.dtype, device=runtime.device
        ),
        objective=torch.as_tensor(
            float(objective), dtype=runtime.dtype, device=runtime.device
        ),
        inner=InnerDiagnostics(
            iterations=int(inner_iterations),
            kkt_residual=float(final_inner_kkt_residual),
            primal_delta=float(final_step_residual),
            dual_delta=float("nan"),
            converged=bool(converged_inner),
        ),
        outer=OuterDiagnostics(
            iterations=int(iterations),
            objective_history=tuple(float(value) for value in history),
            stationarity_residual=float(final_outer_diag["stationarity_residual"]),
            majorization_failures=int(failed_majorization_checks),
            accepted_full_steps=int(accepted_full_steps),
            accepted_damped_steps=int(accepted_damped_steps),
            converged=bool(converged_outer),
        ),
        graph_name=str(graph.name),
        admm_iterations=(
            int(dense_iterations) if inner_solver == "admm_complete_graph" else 0
        ),
        inner_solver=str(inner_solver),
        certificate=certificate,
        exactness_provenance=exactness_provenance,
    )

    return FusionFitArtifacts(
        phi=phi_np.astype(phi_np.dtype, copy=False),
        phi_clustered=phi_clustered.astype(phi_np.dtype, copy=False),
        cluster_labels=cluster_labels.astype(np.int64, copy=False),
        cluster_centers=cluster_centers.astype(phi_np.dtype, copy=False),
        cluster_diameters=cluster_diameters.astype(np.float64, copy=False),
        max_cluster_diameter=float(max_cluster_diameter),
        cluster_diameter_exact=bool(cluster_diameter_exact),
        gamma_major=major_probability.astype(phi_np.dtype, copy=False),
        major_probability=major_probability.astype(phi_np.dtype, copy=False),
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call.astype(phi_np.dtype, copy=False),
        multiplicity_estimated_mask=data.multiplicity_estimation_mask.astype(
            bool, copy=False
        ),
        loglik=float(-fit_loss),
        summary_loglik=summary_loglik,
        penalized_objective=float(objective),
        lambda_value=float(lambda_value),
        n_clusters=n_clusters,
        iterations=int(iterations),
        converged=bool(converged),
        device=runtime.device_name,
        dtype=dtype_name(runtime.dtype),
        graph_name=str(graph.name),
        summary_tol=float(effective_summary_tol),
        history=[float(value) for value in history],
        inner_kkt_residual=float(final_inner_kkt_residual),
        accepted_inner_kkt_residual=float(final_inner_kkt_residual),
        last_attempted_inner_kkt_residual=float(last_attempted_inner_kkt_residual),
        best_attempted_inner_kkt_residual=float(best_attempted_inner_kkt_residual),
        last_attempted_objective_gap=float(last_attempted_objective_gap),
        best_attempted_objective_gap=float(best_attempted_objective_gap),
        last_attempted_surrogate_gap=float(last_attempted_surrogate_gap),
        best_attempted_surrogate_gap=float(best_attempted_surrogate_gap),
        last_attempted_inner_model_gap=float(last_attempted_inner_model_gap),
        best_attempted_inner_model_gap=float(best_attempted_inner_model_gap),
        last_attempted_em_envelope_gap=float(last_attempted_em_envelope_gap),
        best_attempted_em_envelope_gap=float(best_attempted_em_envelope_gap),
        outer_stationarity_residual=float(final_outer_diag["stationarity_residual"]),
        outer_projected_stationarity_residual=float(
            final_outer_diag["projected_stationarity_residual"]
        ),
        outer_projected_stationarity_norm=float(
            final_outer_diag["projected_stationarity_norm"]
        ),
        outer_stationarity_normalizer=float(
            final_outer_diag["stationarity_normalizer"]
        ),
        outer_smooth_gradient_norm=float(final_outer_diag["smooth_gradient_norm"]),
        outer_fusion_adjustment_norm=float(final_outer_diag["fusion_adjustment_norm"]),
        outer_edge_subgradient_residual=float(
            final_outer_diag["edge_subgradient_residual"]
        ),
        outer_dual_ball_residual=float(final_outer_diag["dual_ball_residual"]),
        outer_box_primal_violation=float(final_outer_diag["box_primal_violation"]),
        outer_num_interior_coordinates=int(
            final_outer_diag["num_interior_coordinates"]
        ),
        outer_num_lower_active_coordinates=int(
            final_outer_diag["num_lower_active_coordinates"]
        ),
        outer_num_upper_active_coordinates=int(
            final_outer_diag["num_upper_active_coordinates"]
        ),
        outer_num_frozen_coordinates=int(final_outer_diag["num_frozen_coordinates"]),
        outer_box_residual=float(final_outer_diag["box_residual"]),
        fixed_objective_kkt_residual=float(final_outer_diag["kkt_residual"]),
        outer_kkt_certificate_status=str(outer_kkt_certificate_status),
        outer_kkt_dual_refined=bool(outer_kkt_dual_refined),
        outer_kkt_fused_edges=int(outer_kkt_fused_edges),
        outer_kkt_nonzero_edges=int(outer_kkt_nonzero_edges),
        outer_stationarity_residual_before_dual_refine=float(
            outer_stationarity_residual_before_dual_refine
        ),
        outer_stationarity_residual_after_dual_refine=float(
            outer_stationarity_residual_after_dual_refine
        ),
        converged_inner=bool(converged_inner),
        converged_outer=bool(converged_outer),
        final_relative_objective_change=float(final_relative_objective_change),
        final_step_residual=float(final_step_residual),
        accepted_outer_steps=int(accepted_outer_steps),
        accepted_full_steps=int(accepted_full_steps),
        accepted_damped_steps=int(accepted_damped_steps),
        attempted_outer_steps=int(attempted_outer_steps),
        failed_majorization_checks=int(failed_majorization_checks),
        failed_inner_model_checks=int(failed_inner_model_checks),
        failed_em_envelope_checks=int(failed_em_envelope_checks),
        failed_descent_checks=int(failed_descent_checks),
        failed_nonfinite_checks=int(failed_nonfinite_checks),
        mm_consistency_violations=int(mm_consistency_violations),
        accepted_step_type=str(accepted_step_type),
        last_reject_reason=str(last_reject_reason),
        failure_reason=str(failure_reason),
        selection_eligible=bool(selection_eligible),
        stationarity_certified=bool(stationarity_certified),
        global_optimality_certified=bool(global_optimality_certified),
        global_optimality_basis=str(global_optimality_basis),
        number_of_starts=1,
        number_of_finite_starts=int(np.isfinite(float(objective))),
        best_start_objective=float(objective),
        second_best_start_objective=float("nan"),
        objective_spread_across_starts=0.0,
        selected_start_objective_rank=1,
        solver_state=solver_state_out,
        torch_result=torch_result,
        inner_iterations=int(inner_iterations),
        admm_iterations=(
            int(dense_iterations) if inner_solver == "admm_complete_graph" else 0
        ),
        inner_solver=str(inner_solver),
        certificate=certificate,
        exactness_provenance=exactness_provenance,
    )


def fit_torch(
    data: TumorData,
    *,
    context: SolverContext,
    lambda_value: float,
    state: SolverState | None = None,
    outer_max_iter: int = 8,
    inner_max_iter: int = 30,
    tol: float = 1e-4,
    summary_tol: float | None = None,
    start_mode: str = "warm_only",
    verbose: bool = False,
) -> tuple[TorchFitResult, SolverState]:
    start = state.phi if state is not None else context.exact_pilot
    artifacts = fit_observed_data_pairwise_fusion(
        data,
        lambda_value=float(lambda_value),
        major_prior=float(context.problem.major_prior),
        eps=float(context.problem.eps),
        outer_max_iter=int(outer_max_iter),
        inner_max_iter=int(inner_max_iter),
        tol=float(tol),
        phi_start=start,
        start_mode=start_mode,
        device=context.runtime.device_name,
        dtype=dtype_name(context.runtime.dtype),
        summary_tol=summary_tol,
        solver_context=context,
        solver_state=state,
        compute_summary=False,
        verbose=bool(verbose),
    )
    if artifacts.torch_result is None or artifacts.solver_state is None:
        raise RuntimeError(
            "Torch fit did not produce a tensor result and solver state."
        )
    return artifacts.torch_result, artifacts.solver_state


def fit_observed_data_pairwise_fusion(
    data: TumorData,
    *,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | torch.Tensor | None = None,
    graph: PairwiseFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor]
    | tuple[np.ndarray | torch.Tensor, ...]
    | None = None,
    start_mode: str = "full",
    device: str | None = "cuda",
    dtype: str | None = "float64",
    summary_tol: float | None = None,
    runtime=None,
    torch_data=None,
    solver_context: SolverContext | None = None,
    solver_state: SolverState | None = None,
    compute_summary: bool = True,
    objective_shape: str = "unimodal",
    inner_backend: str = "dense",
    workset_max_bytes: int = 256 * 1024 * 1024,
    compressed_cache_max_bytes: int = 256 * 1024 * 1024,
    dense_fallback_policy: str = "auto",
    workset_add_batch: int = 64,
    workset_max_expansions: int = 16,
    certificate_max_iter: int = 512,
    certificate_refinement_rounds: int = 2,
    certificate_column_tol_scale: float = 1.0,
    allow_heuristic_structure_splits: bool = True,
    verbose: bool = False,
) -> FusionFitArtifacts:
    tol = _validate_solver_tolerance(tol)
    lambda_value = validate_lambda_value(lambda_value)
    objective_shape = _normalize_objective_shape(objective_shape)
    normalized_fallback_policy = (
        str(dense_fallback_policy).strip().lower().replace("-", "_")
    )
    if normalized_fallback_policy not in {
        "auto",
        "device_only",
        "cpu_allowed",
        "error",
    }:
        raise ValueError(
            "dense_fallback_policy must be auto, device_only, cpu_allowed, or error."
        )
    expected_data_fingerprint = _data_fingerprint(data)
    context_prepared_by_cpu_fallback = bool(
        solver_context is not None
        and getattr(solver_context, "resource_fallback", None) == "dense_cpu"
    )
    if solver_context is None:
        try:
            requested_runtime = (
                resolve_runtime(device, dtype=dtype) if runtime is None else runtime
            )
        except RuntimeError:
            if normalized_fallback_policy != "cpu_allowed":
                raise
            try:
                requested_runtime = resolve_runtime("cpu", dtype=dtype)
            except RuntimeError as cpu_runtime_error:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: the requested runtime is "
                    "unavailable and dense CPU fallback does not support the "
                    f"requested dtype {dtype!r}."
                ) from cpu_runtime_error
            context_prepared_by_cpu_fallback = True

        def prepare_context(*, use_supplied_torch_data: bool) -> SolverContext:
            return prepare_torch_problem(
                data,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=tol,
                inner_max_iter=int(inner_max_iter),
                graph=graph,
                adaptive_weight_gamma=float(adaptive_weight_gamma),
                adaptive_weight_floor=float(adaptive_weight_floor),
                adaptive_weight_baseline=float(adaptive_weight_baseline),
                exact_pilot=exact_pilot,
                pooled_start=pooled_start,
                scalar_well_starts=scalar_well_starts,
                device=requested_runtime.device_name,
                dtype=dtype_name(requested_runtime.dtype),
                runtime=requested_runtime,
                torch_data=torch_data if use_supplied_torch_data else None,
                objective_shape=objective_shape,
            )

        if context_prepared_by_cpu_fallback:
            cpu_fits, cpu_bytes, cpu_limit = dense_complete_solver_memory_preflight(
                num_nodes=int(data.num_mutations),
                num_regions=int(data.num_regions),
                runtime=requested_runtime,
            )
            if not cpu_fits:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback needs "
                    f"approximately {cpu_bytes} bytes (available host limit: "
                    f"{cpu_limit})."
                )

        try:
            solver_context = prepare_context(
                use_supplied_torch_data=not context_prepared_by_cpu_fallback
            )
        except (MemoryError, torch.OutOfMemoryError) as exc:
            cpu_fallback_allowed = bool(
                normalized_fallback_policy == "cpu_allowed"
                and requested_runtime.device.type != "cpu"
            )
            if not cpu_fallback_allowed:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: exact problem or graph "
                    f"construction exhausted memory on {requested_runtime.device_name}."
                ) from exc
            try:
                requested_runtime = resolve_runtime(
                    "cpu", dtype=dtype_name(requested_runtime.dtype)
                )
            except RuntimeError as cpu_runtime_error:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback does not "
                    f"support dtype {dtype_name(requested_runtime.dtype)}."
                ) from cpu_runtime_error
            cpu_fits, cpu_bytes, cpu_limit = dense_complete_solver_memory_preflight(
                num_nodes=int(data.num_mutations),
                num_regions=int(data.num_regions),
                runtime=requested_runtime,
            )
            if not cpu_fits:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback needs "
                    f"approximately {cpu_bytes} bytes (available host limit: "
                    f"{cpu_limit})."
                ) from exc
            context_prepared_by_cpu_fallback = True
            try:
                solver_context = prepare_context(use_supplied_torch_data=False)
            except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: exact problem or graph "
                    "construction exhausted host memory during dense CPU fallback."
                ) from cpu_exc
    else:
        if (
            getattr(solver_context, "data_fingerprint", None)
            != expected_data_fingerprint
        ):
            raise ValueError(
                "SolverContext data fingerprint does not match the requested TumorData."
            )
        if (
            abs(float(solver_context.problem.major_prior) - float(major_prior)) > 0.0
            or abs(float(solver_context.problem.eps) - float(eps)) > 0.0
        ):
            raise ValueError(
                "SolverContext major_prior/eps do not match the requested fit options."
            )

    effective_runtime = solver_context.runtime
    effective_graph = solver_context.graph_spec
    effective_exact_pilot = (
        solver_context.exact_pilot if exact_pilot is None else exact_pilot
    )
    effective_pooled_start = (
        solver_context.pooled_start if pooled_start is None else pooled_start
    )
    effective_scalar_well_starts = (
        solver_context.scalar_well_starts
        if scalar_well_starts is None
        else tuple(scalar_well_starts)
    )

    normalized_start_mode = str(start_mode).strip().lower()
    if normalized_start_mode not in {"full", "warm_plus_pilot", "warm_only"}:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    if objective_shape.startswith("unimodal"):
        if phi_start is not None:
            start_bank = [phi_start]
        else:
            start_bank = [effective_exact_pilot]
    else:
        start_bank: list[np.ndarray | torch.Tensor] = []
        if phi_start is not None:
            start_bank.append(phi_start)
        if normalized_start_mode == "full":
            start_bank.extend(effective_scalar_well_starts)
            start_bank.append(effective_pooled_start)
        elif normalized_start_mode == "warm_plus_pilot":
            if phi_start is None:
                start_bank.extend(effective_scalar_well_starts)
                start_bank.append(effective_pooled_start)
            else:
                start_bank.extend(effective_scalar_well_starts)
        elif phi_start is None:
            start_bank.append(effective_exact_pilot)
    start_bank = _deduplicate_starts(start_bank, runtime=effective_runtime)

    def run_start(
        *,
        context: SolverContext,
        start: np.ndarray | torch.Tensor,
        state: SolverState | None,
        backend: str,
        fallback_policy: str,
    ) -> FusionFitArtifacts:
        return _fit_from_start(
            data,
            torch_data=torch_data_from_context(context),
            runtime=context.runtime,
            graph=context.graph_spec,
            tensor_graph=context.graph,
            graph_hash=str(context.graph_hash),
            objective_spec_hash=str(context.objective_spec_hash),
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            outer_max_iter=outer_max_iter,
            inner_max_iter=inner_max_iter,
            tol=tol,
            phi_start=start,
            solver_state=state,
            lower=context.lower,
            upper=context.upper,
            summary_tol=summary_tol,
            compute_summary=compute_summary,
            objective_shape=objective_shape,
            inner_backend=backend,
            workset_max_bytes=workset_max_bytes,
            compressed_cache_max_bytes=compressed_cache_max_bytes,
            dense_fallback_policy=fallback_policy,
            workset_add_batch=workset_add_batch,
            workset_max_expansions=workset_max_expansions,
            certificate_max_iter=certificate_max_iter,
            certificate_refinement_rounds=certificate_refinement_rounds,
            certificate_column_tol_scale=certificate_column_tol_scale,
            allow_heuristic_structure_splits=allow_heuristic_structure_splits,
            verbose=verbose,
        )

    def mark_fallback(
        artifacts: FusionFitArtifacts,
        *,
        reason: str,
        backend_name: str | None = None,
    ) -> FusionFitArtifacts:
        fallback_backend = (
            str(backend_name) if backend_name is not None else artifacts.inner_solver
        )
        fallback_provenance = (
            replace(
                artifacts.exactness_provenance,
                backend_name=fallback_backend,
                fallback_reason=_combine_fallback_reasons(
                    artifacts.exactness_provenance.fallback_reason,
                    reason,
                ),
            )
            if artifacts.exactness_provenance is not None
            else None
        )
        fallback_torch_result = (
            replace(
                artifacts.torch_result,
                inner_solver=fallback_backend,
                exactness_provenance=fallback_provenance,
            )
            if artifacts.torch_result is not None
            else None
        )
        return replace(
            artifacts,
            inner_solver=fallback_backend,
            exactness_provenance=fallback_provenance,
            torch_result=fallback_torch_result,
        )

    def merge_attempted_work(
        artifacts: FusionFitArtifacts,
        attempted: FusionFitArtifacts | None,
    ) -> FusionFitArtifacts:
        """Preserve diagnostic work from a completed attempt before retrying."""

        if attempted is None:
            return artifacts
        current_provenance = artifacts.exactness_provenance
        attempted_provenance = attempted.exactness_provenance
        if current_provenance is None or attempted_provenance is None:
            return artifacts
        merged_provenance = replace(
            current_provenance,
            backend_iterations=(
                int(current_provenance.backend_iterations)
                + int(attempted_provenance.backend_iterations)
            ),
            quotient_iterations=(
                int(current_provenance.quotient_iterations)
                + int(attempted_provenance.quotient_iterations)
            ),
            workset_iterations=(
                int(current_provenance.workset_iterations)
                + int(attempted_provenance.workset_iterations)
            ),
            workset_expansions=(
                int(current_provenance.workset_expansions)
                + int(attempted_provenance.workset_expansions)
            ),
            streamed_edge_passes=(
                int(current_provenance.streamed_edge_passes)
                + int(attempted_provenance.streamed_edge_passes)
            ),
            dense_iterations=(
                int(current_provenance.dense_iterations)
                + int(attempted_provenance.dense_iterations)
            ),
            fallback_reason=_combine_fallback_reasons(
                attempted_provenance.fallback_reason,
                current_provenance.fallback_reason,
            ),
        )
        merged_inner_iterations = int(artifacts.inner_iterations) + int(
            attempted.inner_iterations
        )
        merged_torch_result = artifacts.torch_result
        if merged_torch_result is not None:
            merged_torch_result = replace(
                merged_torch_result,
                inner=replace(
                    merged_torch_result.inner,
                    iterations=(
                        int(merged_torch_result.inner.iterations)
                        + int(attempted.inner_iterations)
                    ),
                ),
                exactness_provenance=merged_provenance,
            )
        return replace(
            artifacts,
            inner_iterations=merged_inner_iterations,
            exactness_provenance=merged_provenance,
            torch_result=merged_torch_result,
        )

    cpu_fallback_context: SolverContext | None = None
    best_artifacts: FusionFitArtifacts | None = None
    start_artifacts: list[FusionFitArtifacts] = []
    for start in start_bank:
        state_for_start = (
            solver_state
            if (solver_state is not None and start is start_bank[0])
            else None
        )
        cpu_seed = state_for_start.phi if state_for_start is not None else start
        attempted_artifacts: FusionFitArtifacts | None = None
        try:
            artifacts = run_start(
                context=solver_context,
                start=start,
                state=state_for_start,
                backend="dense" if context_prepared_by_cpu_fallback else inner_backend,
                fallback_policy=(
                    "device_only"
                    if context_prepared_by_cpu_fallback
                    else dense_fallback_policy
                ),
            )
            provenance = artifacts.exactness_provenance
            compressed_terminal_not_certified = bool(
                isinstance(artifacts.certificate, CompressedEdgeCertificate)
                and (
                    provenance is None
                    or not bool(provenance.full_kkt_certified)
                    or str(provenance.status)
                    not in {
                        "certified",
                        "input_dual_retained",
                        "analytic_nonfused_dual",
                        "refined_fused_edge_dual",
                        "zero_penalty_no_dual_needed",
                    }
                )
            )
            if compressed_terminal_not_certified:
                attempted_artifacts = artifacts
                cpu_seed = (
                    artifacts.torch_result.phi_raw
                    if artifacts.torch_result is not None
                    else artifacts.phi
                )
                if normalized_fallback_policy == "error":
                    raise ExactSolverResourceLimit(
                        "exact_solver_resource_limit: quotient/workset did not "
                        "produce an accepted terminal observed-objective "
                        "certificate and dense fallback is disabled by policy."
                    )
                artifacts = run_start(
                    context=solver_context,
                    start=cpu_seed,
                    state=None,
                    backend="dense",
                    fallback_policy="device_only",
                )
                artifacts = merge_attempted_work(artifacts, attempted_artifacts)
                artifacts = mark_fallback(
                    artifacts,
                    reason="dense_current_device_after_compressed_not_certified",
                )
            if context_prepared_by_cpu_fallback:
                artifacts = mark_fallback(
                    artifacts,
                    reason="dense_cpu_after_context_resource_limit",
                    backend_name="admm_complete_graph_cpu_fallback",
                )
        except (MemoryError, torch.OutOfMemoryError) as exc:
            resource_exc = (
                exc
                if isinstance(exc, ExactSolverResourceLimit)
                else ExactSolverResourceLimit(
                    "exact_solver_resource_limit: exact solver allocation "
                    f"exhausted memory on {effective_runtime.device_name}."
                )
            )
            cpu_fallback_allowed = bool(
                normalized_fallback_policy == "cpu_allowed"
                and effective_runtime.device.type != "cpu"
            )
            if not cpu_fallback_allowed:
                if resource_exc is exc:
                    raise
                raise resource_exc from exc
            try:
                cpu_runtime = resolve_runtime(
                    "cpu", dtype=dtype_name(effective_runtime.dtype)
                )
            except RuntimeError as cpu_runtime_error:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback does not "
                    f"support dtype {dtype_name(effective_runtime.dtype)}."
                ) from cpu_runtime_error
            cpu_fits, cpu_bytes, cpu_limit = dense_complete_solver_memory_preflight(
                num_nodes=int(data.num_mutations),
                num_regions=int(data.num_regions),
                runtime=cpu_runtime,
            )
            if not cpu_fits:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback needs "
                    f"approximately {cpu_bytes} bytes (available host limit: "
                    f"{cpu_limit})."
                ) from resource_exc
            if torch.is_tensor(cpu_seed):
                cpu_start = cpu_seed.detach().to(device="cpu")
            else:
                cpu_start = np.asarray(cpu_seed)
            if cpu_fallback_context is None:
                try:
                    cpu_fallback_context = prepare_torch_problem(
                        data,
                        major_prior=float(major_prior),
                        eps=float(eps),
                        tol=float(tol),
                        inner_max_iter=int(inner_max_iter),
                        graph=effective_graph,
                        exact_pilot=cpu_start,
                        pooled_start=cpu_start,
                        scalar_well_starts=(),
                        device="cpu",
                        dtype=dtype_name(effective_runtime.dtype),
                        objective_shape=objective_shape,
                    )
                except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
                    raise ExactSolverResourceLimit(
                        "exact_solver_resource_limit: exact problem or graph "
                        "construction exhausted host memory during dense CPU "
                        "fallback."
                    ) from cpu_exc
            try:
                artifacts = run_start(
                    context=cpu_fallback_context,
                    start=cpu_start,
                    state=None,
                    backend="dense",
                    fallback_policy="device_only",
                )
            except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
                if isinstance(cpu_exc, ExactSolverResourceLimit):
                    raise
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback exhausted "
                    "host memory during the exact solve."
                ) from cpu_exc
            artifacts = merge_attempted_work(artifacts, attempted_artifacts)
            artifacts = mark_fallback(
                artifacts,
                reason="dense_cpu_after_solver_resource_limit",
                backend_name="admm_complete_graph_cpu_fallback",
            )
        start_artifacts.append(artifacts)
        if best_artifacts is None:
            best_artifacts = artifacts
            continue
        if artifacts.converged and not best_artifacts.converged:
            best_artifacts = artifacts
            continue
        if (
            artifacts.converged == best_artifacts.converged
            and artifacts.penalized_objective
            < best_artifacts.penalized_objective - 1e-8
        ):
            best_artifacts = artifacts

    if best_artifacts is None:
        raise RuntimeError("No valid start produced a fusion fit.")
    objectives = np.asarray(
        [float(item.penalized_objective) for item in start_artifacts], dtype=np.float64
    )
    finite_objectives = objectives[np.isfinite(objectives)]
    if finite_objectives.size:
        sorted_objectives = np.sort(finite_objectives)
        best_start_objective = float(sorted_objectives[0])
        second_best_start_objective = (
            float(sorted_objectives[1]) if sorted_objectives.size >= 2 else float("nan")
        )
        objective_spread = float(sorted_objectives[-1] - sorted_objectives[0])
        selected_objective = float(best_artifacts.penalized_objective)
        selected_rank = int(1 + np.sum(finite_objectives < selected_objective - 1e-8))
    else:
        best_start_objective = float("nan")
        second_best_start_objective = float("nan")
        objective_spread = float("nan")
        selected_rank = 0
    return replace(
        best_artifacts,
        number_of_starts=int(len(start_artifacts)),
        number_of_finite_starts=int(finite_objectives.size),
        best_start_objective=float(best_start_objective),
        second_best_start_objective=float(second_best_start_objective),
        objective_spread_across_starts=float(objective_spread),
        selected_start_objective_rank=int(selected_rank),
    )
