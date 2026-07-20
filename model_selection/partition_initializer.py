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
    LIKELIHOOD_PARTITION_CEM_MAX_ITER,
    LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K,
    LIKELIHOOD_PARTITION_REFIT_MAX_ITER,
    PARTITION_ICL_DIRICHLET_ALPHA,
)
from .partitions import (
    _deduplicate_partition_candidates,
    _likelihood_partition_k_grid,
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
) -> PartitionInitializerPool:
    """Generate the Ward/CEM partition pool used as either proposals or a guide.

    The score is supplied explicitly so guided fusion can always choose its
    initializer by partition ICL without coupling that choice to the final
    fusion-candidate reporting criterion.
    """

    generation_start = perf_counter()
    sparse_k_grid = _likelihood_partition_k_grid(int(data.num_mutations))

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
            max_candidates_per_K=int(LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K),
            cem_max_iter=int(LIKELIHOOD_PARTITION_CEM_MAX_ITER),
            refit_max_iter=int(LIKELIHOOD_PARTITION_REFIT_MAX_ITER),
            tol=float(fit_options.tol),
            curvature=curvature,
            label_sets=label_sets,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
            use_torch=True,
            classification_weight_alpha=(
                float(PARTITION_ICL_DIRICHLET_ALPHA)
                if normalized_score == "partition_icl"
                else None
            ),
            allow_component_death=bool(normalized_score == "partition_icl"),
        )
        generation_elapsed = float(perf_counter() - generation_start)
        return (
            rescore_candidates(
                candidates,
                data=data,
                normalized_score=normalized_score,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
            ),
            ward_elapsed,
            generation_elapsed,
        )

    candidates, ward_elapsed, initial_generation_elapsed = generate(sparse_k_grid)

    refine_k_grid, refinement_reason = _likelihood_partition_refinement_k_grid(
        candidates,
        sparse_k_grid,
        num_mutations=int(data.num_mutations),
    )
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
