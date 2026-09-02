from __future__ import annotations

from dataclasses import replace

from ..core.fusion.partition_starts import (
    PartitionCandidatePool,
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets_torch,
    observed_curvature_at_pilot_torch,
)
from ..config import FitConfig
from ..io.data import TumorData
from .config import (
    LIKELIHOOD_PARTITION_K_MAX,
)
from .partitions import (
    _deduplicate_partition_candidates,
    _likelihood_partition_refinement_k_grid,
)


def generate_partition_initializer_pool(
    *,
    data: TumorData,
    pilot_phi,
    fit_options: FitConfig,
    normalized_score: str,
    runtime,
    torch_data,
    rescore_candidates,
    curvature=None,
    declared_k_grid: tuple[int, ...] | None = None,
    enable_refinement: bool | None = None,
    max_refit_objective_evaluations: int | None = None,
    max_candidates: int | None = None,
) -> PartitionCandidatePool:
    """Generate the deterministic Ward/CEM pool used to choose one guide.

    The score is supplied explicitly so guide generation and retention use the
    same criterion as final partition selection.
    The chosen guide always supplies the initial solver state and lambda scale.
    It defines the frozen adaptive edge weights in strict mode; approximate
    profiles instead weight the graph from the zero-penalty likelihood pilot.
    These are deterministic generation artifacts. Raw-only uses them as guides;
    selectable contracts pass retained labels through the separate direct-
    partition refit/score gate after the raw path terminates.
    """

    contract = fit_options.selection.contract
    config = contract.partition_config
    if declared_k_grid is None:
        sparse_k_grid = [
            int(value)
            for value in config.k_anchors
            if 1 <= int(value) <= int(data.num_mutations)
        ]
        k_cap = min(int(LIKELIHOOD_PARTITION_K_MAX), int(data.num_mutations))
        if k_cap > 0 and k_cap not in sparse_k_grid:
            sparse_k_grid.append(k_cap)
        sparse_k_grid = sorted(set(sparse_k_grid))
    else:
        sparse_k_grid = sorted(
            {
                int(value)
                for value in declared_k_grid
                if 1 <= int(value) <= int(data.num_mutations)
            }
        )
    refinement_enabled = (
        bool(config.adaptive_k_refinement)
        if enable_refinement is None
        else bool(enable_refinement)
    )

    if curvature is None:
        curvature = observed_curvature_at_pilot_torch(
            data,
            pilot_phi,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )

    def generate(
        k_grid: list[int],
        *,
        refit_evaluation_capacity: int | None,
        candidate_capacity: int | None,
    ) -> PartitionCandidatePool:
        label_sets = hessian_weighted_ward_label_sets_torch(
            pilot_phi,
            curvature,
            K_grid=k_grid,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        candidates = generate_likelihood_partition_starts(
            data,
            exact_pilot=pilot_phi,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            K_grid=k_grid,
            max_candidates_per_K=int(config.max_candidates_per_k),
            cem_max_iter=int(config.cem_max_iter),
            refit_max_iter=int(config.generation_refit_max_iter),
            tol=float(fit_options.solver.tolerance),
            curvature=curvature,
            label_sets=label_sets,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
            # The unanchored NumPy refit remains the scoring authority for
            # categorical occupancy paths.  Entirely one-state inputs retain
            # the historical Torch major/minor refit, which has the same
            # objective and avoids the serial path-refit overhead.
            use_torch=getattr(data, "path_likelihood", None) is None,
            classification_weight_alpha=(
                float(config.classification_alpha)
                if normalized_score == "fixed_partition_dirichlet_score"
                else None
            ),
            classification_code_weight=float(config.classification_code_weight),
            # With fixed K, likelihood-only CEM is exactly aligned with BIC
            # because its complexity penalty is constant.  Component death
            # would change K and would require a full BIC-aware move rule, so
            # keep it disabled for BIC generation rather than silently using
            # the Dirichlet move semantics.
            allow_component_death=bool(
                config.allow_component_death
                and normalized_score == "fixed_partition_dirichlet_score"
            ),
            include_plain_ward=bool(config.include_plain_ward),
            include_ward_cem=bool(config.include_ward_cem),
            max_refit_objective_evaluations=refit_evaluation_capacity,
            max_candidates=candidate_capacity,
        )
        return replace(
            candidates,
            candidates=tuple(
                rescore_candidates(
                    list(candidates.candidates),
                    data=data,
                    normalized_score=normalized_score,
                    classification_alpha=float(config.classification_alpha),
                    classification_code_weight=float(
                        config.classification_code_weight
                    ),
                )
            ),
        )

    initial = generate(
        sparse_k_grid,
        refit_evaluation_capacity=max_refit_objective_evaluations,
        candidate_capacity=max_candidates,
    )
    candidates = list(initial.candidates)

    if refinement_enabled and initial.complete:
        refine_k_grid, _ = _likelihood_partition_refinement_k_grid(
            candidates,
            sparse_k_grid,
            num_mutations=int(data.num_mutations),
        )
    else:
        refine_k_grid = []
    if refine_k_grid:
        remaining_refit_capacity = (
            None
            if max_refit_objective_evaluations is None
            else max(
                int(max_refit_objective_evaluations)
                - int(initial.work.partition_refit_objective_evaluations),
                0,
            )
        )
        remaining_candidate_capacity = (
            None
            if max_candidates is None
            else max(int(max_candidates) - len(candidates), 0)
        )
        refined = generate(
            refine_k_grid,
            refit_evaluation_capacity=remaining_refit_capacity,
            candidate_capacity=remaining_candidate_capacity,
        )
        candidates = _deduplicate_partition_candidates(
            candidates + list(refined.candidates)
        )
        return PartitionCandidatePool(
            candidates=tuple(candidates),
            work=initial.work + refined.work,
            complete=bool(refined.complete),
            stop_reason=refined.stop_reason,
        )

    return PartitionCandidatePool(
        candidates=tuple(candidates),
        work=initial.work,
        complete=bool(initial.complete),
        stop_reason=initial.stop_reason,
    )


__all__ = ["generate_partition_initializer_pool"]
