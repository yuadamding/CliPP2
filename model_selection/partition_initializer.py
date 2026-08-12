from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..core.fusion.partition_starts import (
    PartitionCandidate,
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets_torch,
    observed_curvature_at_pilot_torch,
)
from ..core.model import FitOptions
from ..io.data import TumorData
from .config import (
    LIKELIHOOD_PARTITION_K_MAX,
)
from .contracts import get_selection_contract
from .partitions import (
    _deduplicate_partition_candidates,
    _likelihood_partition_refinement_k_grid,
)


@dataclass(frozen=True)
class PartitionInitializerPool:
    candidates: tuple[PartitionCandidate, ...]
    sparse_k_grid: tuple[int, ...]
    refine_k_grid: tuple[int, ...]
    refinement_reason: str
    generation_elapsed_seconds: float
    curvature_elapsed_seconds: float
    ward_elapsed_seconds: float
    refine_ward_elapsed_seconds: float
    initial_generation_elapsed_seconds: float
    refine_generation_elapsed_seconds: float

    @property
    def combined_k_grid(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.sparse_k_grid) | set(self.refine_k_grid)))


def generate_partition_initializer_pool(
    *,
    data: TumorData,
    pilot_phi,
    fit_options: FitOptions,
    normalized_score: str,
    runtime,
    torch_data,
    rescore_candidates,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    curvature=None,
    curvature_elapsed_seconds: float | None = None,
    declared_k_grid: tuple[int, ...] | None = None,
    enable_refinement: bool | None = None,
) -> PartitionInitializerPool:
    """Generate the deterministic Ward/CEM pool used to choose one guide.

    The score is supplied explicitly so guided fusion can choose its guide by
    exact-partition Dirichlet score without coupling that choice to the final
    raw-fusion score.
    The chosen guide always supplies the initial solver state and lambda scale.
    It defines the frozen adaptive edge weights in strict mode; approximate
    profiles instead weight the graph from the zero-penalty likelihood pilot.
    These are deterministic generation artifacts. Raw-only uses them as guides;
    selectable contracts pass retained labels through the separate direct-
    partition refit/score gate after the raw path terminates.
    """

    generation_start = perf_counter()
    contract = get_selection_contract(fit_options.selection_contract)
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
        curvature_start = perf_counter()
        curvature = observed_curvature_at_pilot_torch(
            data,
            pilot_phi,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        curvature_elapsed = float(perf_counter() - curvature_start)
    else:
        curvature_elapsed = float(curvature_elapsed_seconds or 0.0)

    def generate(k_grid: list[int]) -> tuple[list[PartitionCandidate], float, float]:
        ward_start = perf_counter()
        label_sets = hessian_weighted_ward_label_sets_torch(
            pilot_phi,
            curvature,
            K_grid=k_grid,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        ward_elapsed = float(perf_counter() - ward_start)

        generation_start = perf_counter()
        candidates = generate_likelihood_partition_starts(
            data,
            exact_pilot=pilot_phi,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            K_grid=k_grid,
            max_candidates_per_K=int(config.max_candidates_per_k),
            cem_max_iter=int(config.cem_max_iter),
            refit_max_iter=int(config.generation_refit_max_iter),
            tol=float(fit_options.tol),
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
                if normalized_score == "partition_icl"
                else None
            ),
            classification_code_weight=float(config.classification_code_weight),
            allow_component_death=bool(config.allow_component_death),
            include_plain_ward=bool(config.include_plain_ward),
            include_ward_cem=bool(config.include_ward_cem),
        )
        generation_elapsed = float(perf_counter() - generation_start)
        return (
            rescore_candidates(
                candidates,
                data=data,
                normalized_score=normalized_score,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                classification_alpha=float(config.classification_alpha),
                classification_code_weight=float(config.classification_code_weight),
            ),
            ward_elapsed,
            generation_elapsed,
        )

    candidates, ward_elapsed, initial_generation_elapsed = generate(sparse_k_grid)

    if refinement_enabled:
        refine_k_grid, refinement_reason = _likelihood_partition_refinement_k_grid(
            candidates,
            sparse_k_grid,
            num_mutations=int(data.num_mutations),
        )
    else:
        refine_k_grid, refinement_reason = [], "contract_fixed_k_grid"
    refine_ward_elapsed = 0.0
    refine_generation_elapsed = 0.0
    if refine_k_grid:
        (
            refine_candidates,
            refine_ward_elapsed,
            refine_generation_elapsed,
        ) = generate(refine_k_grid)
        candidates = _deduplicate_partition_candidates(candidates + refine_candidates)

    return PartitionInitializerPool(
        candidates=tuple(candidates),
        sparse_k_grid=tuple(int(k) for k in sparse_k_grid),
        refine_k_grid=tuple(int(k) for k in refine_k_grid),
        refinement_reason=str(refinement_reason),
        generation_elapsed_seconds=float(perf_counter() - generation_start),
        curvature_elapsed_seconds=float(curvature_elapsed),
        ward_elapsed_seconds=float(ward_elapsed),
        refine_ward_elapsed_seconds=float(refine_ward_elapsed),
        initial_generation_elapsed_seconds=float(initial_generation_elapsed),
        refine_generation_elapsed_seconds=float(refine_generation_elapsed),
    )


__all__ = ["PartitionInitializerPool", "generate_partition_initializer_pool"]
