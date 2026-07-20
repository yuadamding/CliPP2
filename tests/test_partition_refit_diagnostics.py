from __future__ import annotations

import numpy as np

from CliPP2.core.fusion.refit import partition_constrained_observed_refit
from CliPP2.io.data import TumorData


def _toy_data() -> TumorData:
    alt_counts = np.asarray([[2.0], [7.0]], dtype=np.float64)
    total_counts = np.asarray([[10.0], [12.0]], dtype=np.float64)
    ones = np.ones_like(alt_counts, dtype=np.float64)
    return TumorData(
        tumor_id="toy",
        mutation_ids=["m0", "m1"],
        region_ids=["r0"],
        alt_counts=alt_counts,
        total_counts=total_counts,
        purity=ones,
        major_cn=ones,
        minor_cn=ones,
        normal_cn=np.full_like(alt_counts, 2.0, dtype=np.float64),
        has_cna=np.zeros_like(alt_counts, dtype=bool),
        scaling=np.full_like(alt_counts, 0.5, dtype=np.float64),
        phi_upper=ones,
        phi_init=np.full_like(alt_counts, 0.5, dtype=np.float64),
        init_major_mask=np.ones_like(alt_counts, dtype=bool),
        count_observed=np.ones_like(alt_counts, dtype=bool),
    )


def test_partition_refit_reports_search_diagnostics() -> None:
    result = partition_constrained_observed_refit(
        _toy_data(),
        np.asarray([0, 0], dtype=np.int64),
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        max_iter=64,
        hint_phi=np.asarray([[0.4], [0.6]], dtype=np.float64),
    )

    assert result.finite_candidate_found
    assert result.refit_coordinate_count == 1
    assert result.refit_finite_coordinate_count == 1
    assert result.refit_total_grid_points > 1
    assert result.refit_max_grid_spacing > 0.0
    assert result.refit_total_candidate_basins >= 1
    assert result.refit_total_refined_candidates >= 1
    assert np.isfinite(result.refit_min_best_second_loss_gap)
