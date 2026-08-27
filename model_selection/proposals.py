from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..core.bic import compute_classic_bic, compute_partition_dirichlet_score
from ..core.fusion.defaults import normalize_dense_fallback_policy
from ..core.fusion.graph import build_likelihood_noise_regularized_adaptive_graph
from ..core.fusion.graph_ops import (
    build_likelihood_noise_regularized_adaptive_tensor_graph,
    tensor_graph_to_pairwise_graph,
)
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.fusion.solver import (
    escape_path_breakpoint_solver_state,
    objective_shape_for_data,
    prepare_torch_problem_with_resource_policy,
)
from ..core.fusion.torch_backend import dtype_name
from ..core.fusion.types import (
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    DenseWarmState,
    ExactSolverResourceLimit,
    PrimalOnlyWarmState,
    SolverContext,
    SolverState,
)
from ..config import FitConfig
from ..core.fusion.types import RawFit
from ..io.data import TumorData
from .guided_fusion import GuidedFusionInitialization, build_guided_fusion_initialization
from .scoring import _canonical_lambda, _prefer_fit_candidate
from .types import StartArray


@dataclass(frozen=True, slots=True)
class RawStartAttempt:
    """One fixed-objective solve from an authorized optimizer state."""

    fit: RawFit
    source: str
    start_value: float
    breakpoint_escape_changed_count: int
    mathematically_certified: bool
    promotion_status: str = "not_recorded"


RawStartSpec = tuple[str, float, SolverState | None, StartArray | None]


def bootstrap_independent_start_specs(
    *,
    initial_lambda: float,
    raw_guide_phi: StartArray,
    exact_pilot: StartArray,
    pooled_start: StartArray,
    suffix: str,
) -> tuple[RawStartSpec, ...]:
    """Return the fixed independent cold-start bank used before path bootstrap."""

    normalized_suffix = str(suffix).strip()
    if not normalized_suffix:
        raise ValueError("Bootstrap start suffix must be non-empty.")
    return (
        (
            f"cold_partition_guide_{normalized_suffix}",
            float(initial_lambda),
            None,
            raw_guide_phi,
        ),
        (f"cold_zero_penalty_{normalized_suffix}", 0.0, None, exact_pilot),
        (f"cold_pooled_likelihood_{normalized_suffix}", 0.0, None, pooled_start),
    )


def explicit_path_default_start_specs(
    *,
    scalar_well_starts: tuple[StartArray, ...] | list[StartArray],
    pooled_start: StartArray,
) -> tuple[RawStartSpec, ...]:
    """Expose the low-level explicit-path defaults as one flat start bank.

    Model selection combines these specifications with its warm and guided
    starts, then performs one stable deduplication pass. This avoids nesting
    the complete default bank inside every externally named start while
    preserving every distinct cold basin and its provenance.
    """

    specs = [
        (f"cold_scalar_well_{index}", 0.0, None, start)
        for index, start in enumerate(scalar_well_starts)
    ]
    specs.append(("cold_pooled_likelihood_default", 0.0, None, pooled_start))
    return tuple(specs)


def select_raw_start_attempt(
    attempts: list[RawStartAttempt],
) -> RawStartAttempt:
    """Choose the lowest-objective certified basin before any partition score.

    The observed-data fusion objective is nonconvex.  KKT certification proves
    stationarity of a start's terminal basin, not that a worse stationary basin
    may steer lambda search merely because it finished first.  When at least
    one authorized start is certified, raw penalized objective is therefore the
    sole primary ordering; partition summaries are deliberately absent here.
    """

    if not attempts:
        raise ValueError("At least one raw start attempt is required.")
    certified = [item for item in attempts if item.mathematically_certified]
    if certified:
        return min(
            certified,
            key=lambda item: (
                (
                    float(item.fit.objective.total)
                    if np.isfinite(float(item.fit.objective.total))
                    else float("inf")
                ),
                (
                    float(item.fit.certificate.components.residual)
                    if np.isfinite(float(item.fit.certificate.components.residual))
                    else float("inf")
                ),
                str(item.source),
            ),
        )
    # Preserve the historical retry routing when no start is certified: retain
    # the best solver candidate, then let the online controller request more
    # effort at this same lambda.
    incumbent = attempts[0]
    for item in attempts[1:]:
        if _prefer_fit_candidate(item.fit, incumbent.fit):
            incumbent = item
    return incumbent


def _hash_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())


def pilot_matrix_hash(pilot_phi: StartArray | None) -> str:
    if pilot_phi is None:
        return ""
    if torch.is_tensor(pilot_phi):
        array = pilot_phi.detach().cpu().numpy()
    else:
        array = np.asarray(pilot_phi)
    hasher = hashlib.blake2b(digest_size=16)
    _hash_array(hasher, np.asarray(array, dtype=np.float64))
    return hasher.hexdigest()


def clone_start(start: StartArray) -> StartArray:
    if torch.is_tensor(start):
        return start.detach().clone()
    return np.asarray(start).copy()


def offload_solver_state_to_cpu(state: SolverState | None) -> SolverState | None:
    """Move persistent warm-start tensors off the accelerator.

    Online model selection retains several certified and failed candidates so
    later proposals can warm-start from either side of an observed bracket.
    A complete-graph dual has shape E x S and can exceed a GiB for large
    cohorts. Keeping every historical dual on CUDA makes memory scale with the
    number of evaluated lambdas even though only one state is used at a time.
    Host storage preserves the exact float dtype and values; the solver moves
    the selected state back to its runtime device when it is next used.
    """

    if state is None:
        return None

    cpu_tensors: dict[
        tuple[
            torch.device,
            torch.dtype,
            torch.layout,
            int,
            int,
            tuple[int, ...],
            tuple[int, ...],
        ],
        torch.Tensor,
    ] = {}

    def to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        detached = tensor.detach()
        if detached.numel() == 0:
            return detached.to(device="cpu")
        alias_key = (
            detached.device,
            detached.dtype,
            detached.layout,
            int(detached.untyped_storage().data_ptr()),
            int(detached.storage_offset()),
            tuple(int(value) for value in detached.shape),
            tuple(int(value) for value in detached.stride()),
        )
        cached = cpu_tensors.get(alias_key)
        if cached is None:
            cached = detached.to(device="cpu")
            cpu_tensors[alias_key] = cached
        return cached

    def certificate_to_cpu(certificate):
        if isinstance(certificate, DenseEdgeCertificate):
            return replace(certificate, dual=to_cpu(certificate.dual))
        if isinstance(certificate, CompressedEdgeCertificate):
            return replace(
                certificate,
                labels=to_cpu(certificate.labels),
                centers=to_cpu(certificate.centers),
                internal_edge_ids=to_cpu(certificate.internal_edge_ids),
                internal_dual=to_cpu(certificate.internal_dual),
            )
        return certificate

    warm_state = state.warm_state
    if isinstance(warm_state, DenseWarmState):
        warm_state = replace(
            warm_state,
            phi=to_cpu(warm_state.phi),
            dual=to_cpu(warm_state.dual),
        )
    elif isinstance(warm_state, PrimalOnlyWarmState):
        warm_state = replace(
            warm_state,
            phi=to_cpu(warm_state.phi),
            structure_hint=to_cpu(warm_state.structure_hint),
            certificate_hint=certificate_to_cpu(warm_state.certificate_hint),
        )

    certificate = certificate_to_cpu(state.certificate)

    return SolverState(
        phi=to_cpu(state.phi),
        dual=to_cpu(state.dual),
        previous_lambda=float(state.previous_lambda),
        warm_state=warm_state,
        certificate=certificate,
        objective_spec_hash=str(state.objective_spec_hash),
    )


def escape_path_breakpoint_retry_state(
    state: SolverState | None,
    *,
    start_source: str,
    start_lambda: float,
    target_lambda: float,
    context: SolverContext,
    tol: float,
) -> tuple[SolverState | None, int]:
    same_lambda_failure = start_source in {
        "same_lambda_retry",
        "best_same_lambda_kkt_state",
    } and _canonical_lambda(start_lambda) == _canonical_lambda(target_lambda)
    if state is None or not same_lambda_failure:
        return state, 0
    return escape_path_breakpoint_solver_state(state, context=context, tol=tol)


def solver_recovery_fit_options(
    data: TumorData,
    fit_options: FitConfig,
    *,
    retry_number: int | None = None,
) -> FitConfig:
    """Increase solver effort without changing the fixed objective family.

    Occupancy-path likelihoods are generically nonconvex.  Their recovery run
    must therefore remain on ``generic_nonconvex`` rather than claiming the
    unimodal full-step contract.  ``objective_shape_for_data`` enforces that
    invariant while the larger iteration budgets give a failed breakpoint/KKT
    state one final correctness-preserving attempt at the unchanged tolerance.
    """

    del retry_number
    effort_factor = 6
    solver = fit_options.solver
    certificate = solver.certificate
    return replace(
        fit_options,
        # The large multi-region failure audit showed that 18/75 continuation
        # reduced the residual materially but stopped above the unchanged KKT
        # gate.  Give the one terminal same-lambda continuation 36/150 or six
        # profile-sized budgets.  Solves still stop early after certification.
        solver=replace(
            solver,
            outer_max_iter=max(int(solver.outer_max_iter) * 24, 144),
            inner_max_iter=max(int(solver.inner_max_iter) * effort_factor, 150),
            certificate=replace(
                certificate,
                max_iter=max(int(certificate.max_iter) * 2, 256),
                refinement_rounds=max(int(certificate.refinement_rounds) + 1, 2),
            ),
            # The controller's full-KKT admission threshold stays the
            # immutable profile value (it is derived from the effective
            # options before any recovery).  The recovery SOLVE, however,
            # stops much deeper: under the widened multiplicity mixture the
            # inner convergence measure plateaus far above true stationarity
            # at the profile tolerance (measured residual 0.07-0.3 against a
            # 0.004 gate), while the same solve driven to 5e-5 certifies.
            # Per-attempt certification at the tighter tolerance is a
            # strictly harder admission, never a weaker one.
            tolerance=min(float(solver.tolerance), 5e-5),
            # Admission stays at the immutable contract gate even though the
            # recovery solve iterates far deeper.
            certification_tolerance=(
                float(solver.tolerance)
                if solver.certification_tolerance is None
                else float(solver.certification_tolerance)
            ),
            use_backward_error_progress=True,
            objective_shape=objective_shape_for_data(
                data, "unimodal_full_step_backtracking"
            ),
        ),
    )


def build_guided_initialization_with_resource_policy(
    *,
    data: TumorData,
    guide_phi: StartArray,
    guide_labels: np.ndarray | torch.Tensor,
    solver_context: SolverContext,
    fit_options: FitConfig,
) -> tuple[GuidedFusionInitialization, SolverContext, StartArray]:
    """Build guided state with typed allocation failure and optional CPU retry."""

    fallback_policy = normalize_dense_fallback_policy(fit_options.runtime.fallback)

    def build(
        *,
        context: SolverContext,
        phi: StartArray,
        labels: np.ndarray | torch.Tensor,
    ) -> GuidedFusionInitialization:
        # The dense solver needs the guide's actual edge dual to preserve the
        # historical one-candidate warm-start path.  A compressed guide is the
        # right representation for quotient/workset, but feeding it to dense
        # discards that high-quality dual and can turn one certified fit into a
        # long recovery/search sequence.
        return build_guided_fusion_initialization(
            phi,
            labels,
            solver_context=context,
            partition_tolerance=max(float(fit_options.solver.tolerance), 1e-8),
            kkt_atol=float(fit_options.solver.tolerance),
        )

    try:
        return (
            build(context=solver_context, phi=guide_phi, labels=guide_labels),
            solver_context,
            guide_phi,
        )
    except (MemoryError, torch.OutOfMemoryError) as exc:
        cpu_fallback_allowed = bool(
            fallback_policy == "cpu_allowed"
            and solver_context.runtime.device.type != "cpu"
        )
        if not cpu_fallback_allowed:
            if isinstance(exc, ExactSolverResourceLimit):
                raise
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: guided initialization exhausted "
                f"memory on {solver_context.runtime.device_name}."
            ) from exc

        try:

            def cpu_start(value: StartArray) -> StartArray:
                return (
                    value.detach().to(device="cpu")
                    if torch.is_tensor(value)
                    else np.asarray(value)
                )

            cpu_guide_phi: StartArray = (
                guide_phi.detach().to(device="cpu")
                if torch.is_tensor(guide_phi)
                else np.asarray(guide_phi)
            )
            cpu_guide_labels = (
                guide_labels.detach().to(device="cpu")
                if torch.is_tensor(guide_labels)
                else np.asarray(guide_labels)
            )
            cpu_context = prepare_torch_problem_with_resource_policy(
                data,
                dense_fallback_policy="device_only",
                inherited_resource_fallback="dense_cpu",
                major_prior=float(solver_context.problem.major_prior),
                eps=float(solver_context.problem.eps),
                tol=float(fit_options.solver.tolerance),
                graph=solver_context.graph_spec,
                inner_max_iter=max(int(fit_options.solver.inner_max_iter), 16),
                adaptive_weight_gamma=float(fit_options.graph.adaptive_weight_gamma),
                adaptive_weight_floor=float(fit_options.graph.adaptive_weight_floor),
                adaptive_weight_baseline=float(fit_options.graph.adaptive_weight_baseline),
                exact_pilot=cpu_start(solver_context.exact_pilot),
                pooled_start=cpu_start(solver_context.pooled_start),
                scalar_well_starts=solver_context.scalar_well_starts,
                device="cpu",
                dtype=dtype_name(solver_context.runtime.dtype),
                objective_shape=str(fit_options.solver.objective_shape),
            )
            guided = build(
                context=cpu_context,
                phi=cpu_guide_phi,
                labels=cpu_guide_labels,
            )
        except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
            if isinstance(cpu_exc, ExactSolverResourceLimit):
                raise cpu_exc from exc
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: guided initialization exhausted "
                "host memory during dense CPU fallback."
            ) from cpu_exc
        return guided, cpu_context, cpu_guide_phi


def build_partition_guided_graph_with_resource_policy(
    *,
    guide_phi: StartArray,
    guide_curvature: torch.Tensor,
    solver_context: SolverContext,
    fit_options: FitConfig,
    noise_divisor: float,
):
    """Build the adaptive graph on CUDA, with an explicitly authorized host retry."""

    graph_options = {
        "gamma": float(fit_options.graph.adaptive_weight_gamma),
        "minimum_tau": max(
            float(fit_options.graph.adaptive_weight_floor), float(fit_options.eps)
        ),
        "baseline": float(fit_options.graph.adaptive_weight_baseline),
        "noise_divisor": float(noise_divisor),
    }

    def host_array(value):
        return (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )

    def build_host_graph():
        return build_likelihood_noise_regularized_adaptive_graph(
            host_array(guide_phi),
            host_array(guide_curvature),
            lower=host_array(solver_context.lower),
            upper=host_array(solver_context.upper),
            count_observed=(
                None
                if solver_context.problem.count_observed is None
                else host_array(solver_context.problem.count_observed)
            ),
            **graph_options,
        )

    runtime = solver_context.runtime
    if runtime.device.type != "cuda":
        graph, tau = build_host_graph()
        return graph, None, tau

    try:
        tensor_graph, tau = build_likelihood_noise_regularized_adaptive_tensor_graph(
            torch.as_tensor(
                guide_phi,
                dtype=runtime.dtype,
                device=runtime.device,
            ),
            guide_curvature,
            runtime,
            lower=solver_context.lower,
            upper=solver_context.upper,
            count_observed=solver_context.problem.count_observed,
            **graph_options,
        )
        return tensor_graph_to_pairwise_graph(tensor_graph), tensor_graph, tau
    except (MemoryError, torch.OutOfMemoryError) as exc:
        if (
            normalize_dense_fallback_policy(fit_options.runtime.fallback)
            != "cpu_allowed"
        ):
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: partition-guided graph construction "
                f"exhausted memory on {runtime.device_name}; host retry is disabled."
            ) from exc
        try:
            graph, tau = build_host_graph()
        except (MemoryError, torch.OutOfMemoryError) as host_exc:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: partition-guided graph construction "
                "exhausted host memory during the authorized CPU retry."
            ) from host_exc
        return graph, None, tau


def rescore_partition_candidates(
    candidates: list[PartitionCandidate],
    *,
    data: TumorData,
    normalized_score: str,
    classification_alpha: float,
    classification_code_weight: float,
) -> list[PartitionCandidate]:
    """Put the active selection score in ``PartitionCandidate.bic``.

    Candidate generation historically used that field for per-K ordering,
    refinement focus, and deduplication.  Keeping the field as the active score
    lets those operations follow the requested criterion while the candidate
    output rows continue to report classic BIC explicitly.
    """
    # This score orders deterministic Ward/CEM proposals and chooses the raw
    # guide. Under a selectable hybrid contract, the authoritative evaluator
    # later refits every retained label set and recomputes the common score;
    # generation-time values never become selection authority directly.
    rescored: list[PartitionCandidate] = []
    for candidate in candidates:
        if normalized_score == "fixed_partition_bic":
            selected_score = compute_classic_bic(
                -float(candidate.fit_loss),
                int(candidate.K),
                data,
            )
        elif normalized_score == "fixed_partition_dirichlet_score":
            selected_score = compute_partition_dirichlet_score(
                -float(candidate.fit_loss),
                np.bincount(
                    np.asarray(candidate.labels, dtype=np.int64),
                    minlength=int(candidate.K),
                ),
                data=data,
                alpha=float(classification_alpha),
                code_weight=float(classification_code_weight),
            )
        else:
            raise ValueError(
                f"Unsupported partition-generation score {normalized_score!r}."
            )
        rescored.append(
            replace(
                candidate,
                bic=float(selected_score),
            )
        )
    return rescored


def adaptive_stop_certifies_global_optimum(stop_reason: str) -> bool:
    """Fail closed until a controller stop carries an explicit global proof.

    Local score/event basins, resource budgets, path nonmonotonicity, and
    uncertified partition summaries are all useful stopping diagnostics, but
    none proves that no lower-score partition exists elsewhere on the path.
    """

    return str(stop_reason) == "online_lambda_global_optimum_certified"


def direct_partition_source(
    proposal: PartitionCandidate,
    *,
    stage: str,
) -> str:
    cem = str(proposal.source).startswith("hessian_ward_cem")
    death = int(proposal.component_death_count) > 0
    prefix = "pilot" if stage == "pilot" else "final_phi"
    suffix = "hessian_ward_cem" if cem else "hessian_ward"
    if cem and death:
        suffix += "_component_death"
    return f"{prefix}_{suffix}"




__all__ = [
    "RawStartAttempt",
    "adaptive_stop_certifies_global_optimum",
    "bootstrap_independent_start_specs",
    "explicit_path_default_start_specs",
    "build_guided_initialization_with_resource_policy",
    "build_partition_guided_graph_with_resource_policy",
    "clone_start",
    "direct_partition_source",
    "escape_path_breakpoint_retry_state",
    "offload_solver_state_to_cpu",
    "pilot_matrix_hash",
    "rescore_partition_candidates",
    "select_raw_start_attempt",
    "solver_recovery_fit_options",
]
