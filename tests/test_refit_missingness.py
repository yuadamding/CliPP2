"""Partition-refit missingness regression tests.

The fit objective masks out unobserved mutation_regions (torch_backend.mutation_region_terms_torch), and
the BIC denominator (effective_bic_mutation_region_count) excludes them. The BIC *numerator*
is produced by the partition refit, so the refit loglik must exclude unobserved
mutation_regions too — otherwise BIC scores a model that was never fit. These tests pin that
invariance for both refit backends (numpy and torch) and assert they agree.

Fixture uses major_cn == minor_cn == 1 so multiplicity is non-ambiguous and the
per-sample objective is unimodal, letting the two backends agree tightly.
"""

from __future__ import annotations

import numpy as np
import pytest

from CliPP2.core.bic import compute_classic_bic, effective_bic_mutation_region_count
from CliPP2.core.fusion.partition_starts import (
    partition_constrained_observed_refit_torch,
)
from CliPP2.core.fusion.refit import partition_constrained_observed_refit
from CliPP2.io.data import TumorData


def _tumor(
    alt: list[float], total: list[float], count_observed: list[bool]
) -> TumorData:
    m = len(alt)
    alt_np = np.array([[float(a)] for a in alt], dtype=np.float64)
    total_np = np.array([[float(t)] for t in total], dtype=np.float64)
    purity = np.full((m, 1), 0.8, dtype=np.float64)
    major = np.ones((m, 1), dtype=np.float64)
    minor = np.ones((m, 1), dtype=np.float64)
    normal = np.full((m, 1), 2.0, dtype=np.float64)
    scaling = purity / (purity * (major + minor) + (1.0 - purity) * normal)
    return TumorData(
        tumor_id="refit-missing",
        mutation_ids=[f"m{i}" for i in range(m)],
        region_ids=["s1"],
        alt_counts=alt_np,
        total_counts=total_np,
        purity=purity,
        major_cn=major,
        minor_cn=minor,
        normal_cn=normal,
        has_cna=np.zeros((m, 1), dtype=bool),
        scaling=scaling,
        phi_upper=np.ones((m, 1), dtype=np.float64),
        phi_init=np.full((m, 1), 0.5, dtype=np.float64),
        init_major_mask=np.ones((m, 1), dtype=bool),
        count_observed=np.array([[bool(o)] for o in count_observed], dtype=bool),
    )


_KW = dict(major_prior=0.5, eps=1e-6, tol=1e-7, max_iter=128)

# Third mutation_region has positive depth but is unobserved — the exact bug scenario.
_ALT = [5.0, 6.0, 9.0]
_TOTAL = [10.0, 12.0, 14.0]


def _masked() -> TumorData:
    return _tumor(_ALT, _TOTAL, [True, True, False])


def _dropped() -> TumorData:
    return _tumor(_ALT[:2], _TOTAL[:2], [True, True])


def test_numpy_refit_loglik_ignores_unobserved_positive_depth_mutation_region() -> None:
    r_masked = partition_constrained_observed_refit(
        _masked(), np.array([0, 0, 0]), **_KW
    )
    r_dropped = partition_constrained_observed_refit(
        _dropped(), np.array([0, 0]), **_KW
    )
    assert r_masked.loglik == pytest.approx(r_dropped.loglik, abs=1e-6)


def test_torch_refit_loglik_ignores_unobserved_positive_depth_mutation_region() -> None:
    r_masked = partition_constrained_observed_refit_torch(
        _masked(), np.array([0, 0, 0]), device="cpu", **_KW
    )
    r_dropped = partition_constrained_observed_refit_torch(
        _dropped(), np.array([0, 0]), device="cpu", **_KW
    )
    assert r_masked.loglik == pytest.approx(r_dropped.loglik, abs=1e-6)


def test_masking_strictly_increases_loglik_vs_treating_all_observed() -> None:
    # If the unobserved mutation_region were (wrongly) counted, its negative-log-likelihood
    # term would lower the loglik. Masking must remove that term.
    r_masked = partition_constrained_observed_refit(
        _masked(), np.array([0, 0, 0]), **_KW
    )
    r_all = partition_constrained_observed_refit(
        _tumor(_ALT, _TOTAL, [True, True, True]), np.array([0, 0, 0]), **_KW
    )
    assert r_masked.loglik > r_all.loglik + 1e-6


def test_numpy_and_torch_refit_agree_under_missingness() -> None:
    r_np = partition_constrained_observed_refit(_masked(), np.array([0, 0, 0]), **_KW)
    r_t = partition_constrained_observed_refit_torch(
        _masked(), np.array([0, 0, 0]), device="cpu", **_KW
    )
    assert r_np.loglik == pytest.approx(r_t.loglik, rel=1e-4, abs=1e-5)


def test_bic_numerator_and_denominator_agree_under_missingness() -> None:
    masked, dropped = _masked(), _dropped()
    r_masked = partition_constrained_observed_refit(masked, np.array([0, 0, 0]), **_KW)
    r_dropped = partition_constrained_observed_refit(dropped, np.array([0, 0]), **_KW)
    # Denominator excludes the unobserved mutation_region ...
    assert effective_bic_mutation_region_count(
        masked
    ) == effective_bic_mutation_region_count(dropped)
    # ... and the numerator now does too, so BIC matches the mutation_region-removed model.
    bic_masked = compute_classic_bic(r_masked.loglik, r_masked.n_clusters, masked)
    bic_dropped = compute_classic_bic(r_dropped.loglik, r_dropped.n_clusters, dropped)
    assert bic_masked == pytest.approx(bic_dropped, abs=1e-6)
