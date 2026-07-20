from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.multiplicity import infer_multiplicity_posterior_numpy
from CliPP2.core.fusion.torch_backend import (
    mutation_region_terms_torch,
    resolve_runtime,
    to_torch_tumor_data,
)
from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.io.data import TumorData
from CliPP2.metrics.evaluation import SimulationTruth, evaluate_fit_against_simulation
from CliPP2.model_selection.partitions import _multiplicity_summary_for_phi
from CliPP2.runners.outputs import evaluation_to_frame


def _posterior_tumor_data() -> TumorData:
    alt = np.asarray([[8.0, 3.0, 1.0], [4.0, 6.0, 2.0]])
    total = np.asarray([[20.0, 18.0, 12.0], [20.0, 20.0, 10.0]])
    purity = np.full_like(alt, 0.7)
    major_cn = np.asarray([[2.0, 3.0, 1.0], [2.0, 3.0, 1.0]])
    minor_cn = np.asarray([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    normal_cn = np.full_like(alt, 2.0)
    scaling = purity / (purity * (major_cn + minor_cn) + (1.0 - purity) * normal_cn)
    return TumorData(
        tumor_id="posterior_tumor",
        mutation_ids=["m0", "m1"],
        region_ids=["r0", "r1", "r2"],
        alt_counts=alt,
        total_counts=total,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        has_cna=np.ones_like(alt, dtype=bool),
        scaling=scaling,
        phi_upper=np.ones_like(alt),
        phi_init=np.full_like(alt, 0.5),
        init_major_mask=np.ones_like(alt, dtype=bool),
        count_observed=np.asarray(
            [[True, True, True], [False, True, True]],
            dtype=bool,
        ),
    )


def test_numpy_multiplicity_posterior_matches_torch_on_all_entries() -> None:
    data = _posterior_tumor_data()
    phi = np.asarray([[0.7, 0.45, 0.2], [0.55, 0.6, 0.4]])
    prior = 0.3
    eps = 1e-6

    posterior = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=prior,
        eps=eps,
    )
    runtime = resolve_runtime("cpu", dtype="float64")
    torch_terms = mutation_region_terms_torch(
        to_torch_tumor_data(data, runtime),
        torch.as_tensor(phi, dtype=torch.float64),
        major_prior=prior,
        eps=eps,
    )
    np.testing.assert_allclose(
        posterior.gamma_major,
        torch_terms.gamma_major.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    observed_estimable = data.multiplicity_estimation_mask & data.count_observed
    assert not np.allclose(posterior.gamma_major[observed_estimable], 0.5)


def test_raw_fit_keeps_masked_ambiguous_multiplicity_at_prior() -> None:
    data = _posterior_tumor_data()
    prior = 0.3

    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.0,
            major_prior=prior,
            outer_max_iter=4,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )

    # This entry is ambiguous but its counts are masked.  The raw solver must
    # not use those stored counts to make a multiplicity call.
    assert data.multiplicity_estimation_mask[1, 0]
    assert not data.count_observed[1, 0]
    assert fit.gamma_major[1, 0] == pytest.approx(prior)
    assert not fit.major_call[1, 0]
    assert fit.multiplicity_call[1, 0] == data.minor_cn[1, 0]


def test_numpy_multiplicity_posterior_respects_missing_and_fixed_states() -> None:
    data = _posterior_tumor_data()
    phi = np.full(data.alt_counts.shape, 0.5)
    prior = 0.3

    posterior = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=prior,
        eps=1e-6,
    )

    # This entry is eligible for inference, but its counts are masked.
    assert data.multiplicity_estimation_mask[1, 0]
    assert not data.count_observed[1, 0]
    assert posterior.gamma_major[1, 0] == pytest.approx(prior)
    assert not posterior.major_call[1, 0]
    assert posterior.multiplicity_call[1, 0] == data.minor_cn[1, 0]

    fixed_mask = ~data.multiplicity_estimation_mask
    np.testing.assert_array_equal(posterior.gamma_major[fixed_mask], 1.0)
    np.testing.assert_array_equal(posterior.major_call[fixed_mask], True)
    np.testing.assert_array_equal(
        posterior.multiplicity_call[fixed_mask],
        data.major_cn[fixed_mask],
    )


def test_partition_multiplicity_summary_uses_configured_posterior() -> None:
    data = _posterior_tumor_data()
    phi = np.asarray([[0.7, 0.45, 0.2], [0.55, 0.6, 0.4]])

    summary = _multiplicity_summary_for_phi(
        data,
        phi,
        major_prior=0.8,
        eps=1e-5,
    )
    expected = infer_multiplicity_posterior_numpy(
        data,
        phi,
        major_prior=0.8,
        eps=1e-5,
    )

    np.testing.assert_allclose(summary[0], expected.gamma_major)
    np.testing.assert_array_equal(summary[1], expected.major_call)
    np.testing.assert_array_equal(summary[2], expected.multiplicity_call)
    np.testing.assert_array_equal(summary[3], expected.estimation_mask)


def _evaluation_tumor_data() -> TumorData:
    major_cn = np.asarray([[2.0], [2.0], [3.0], [2.0], [1.0]])
    minor_cn = np.asarray([[1.0], [1.0], [0.0], [2.0], [1.0]])
    shape = major_cn.shape
    purity = np.full(shape, 0.8)
    normal_cn = np.full(shape, 2.0)
    scaling = purity / (purity * (major_cn + minor_cn) + (1.0 - purity) * normal_cn)
    return TumorData(
        tumor_id="evaluation_tumor",
        mutation_ids=[f"m{i}" for i in range(shape[0])],
        region_ids=["r0"],
        alt_counts=np.full(shape, 5.0),
        total_counts=np.full(shape, 20.0),
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        # m2 deliberately has an asymmetric copy state with has_cna=False so
        # the primary metric test enforces the exact major_cn != minor_cn mask.
        has_cna=np.asarray([[True], [True], [False], [True], [True]]),
        scaling=scaling,
        phi_upper=np.ones(shape),
        phi_init=np.full(shape, 0.5),
        init_major_mask=np.ones(shape, dtype=bool),
    )


def test_evaluation_separates_asymmetric_and_estimable_multiplicity_f1() -> None:
    data = _evaluation_tumor_data()
    phi = np.full(data.alt_counts.shape, 0.5)
    fit = SimpleNamespace(
        phi=phi,
        phi_clustered=phi,
        cluster_labels=np.asarray([0, 0, 1, 1, 2]),
        n_clusters=3,
        # Estimable entry m1 is intentionally wrong. The symmetric m3 is also
        # wrong, but must not enter either major-versus-minor score.
        multiplicity_call=np.asarray([[2.0], [2.0], [3.0], [1.0], [1.0]]),
    )
    truth = SimulationTruth(
        truth_clusters=np.asarray([0, 0, 1, 1, 2]),
        truth_phi=phi,
        truth_multiplicity=np.asarray([[2.0], [1.0], [3.0], [2.0], [1.0]]),
    )

    evaluation = evaluate_fit_against_simulation(
        fit=fit,
        data=data,
        simulation_truth=truth,
    )

    # Asymmetric states are m0/m1/m2: truth [major, minor, major], while all
    # three predictions are major. Class-macro F1 = (0.8 + 0.0) / 2.
    assert evaluation.multiplicity_f1 == pytest.approx(0.4)
    assert evaluation.multiplicity_asymmetric_f1 == pytest.approx(0.4)
    # The strict estimable mask is m0/m1, yielding class-macro F1 = 1/3.
    assert evaluation.multiplicity_estimable_f1 == pytest.approx(1.0 / 3.0)

    frame = evaluation_to_frame(evaluation)
    assert frame.loc[0, "multiplicity_f1"] == pytest.approx(0.4)
    assert frame.loc[0, "multiplicity_asymmetric_f1"] == pytest.approx(0.4)
    assert frame.loc[0, "multiplicity_estimable_f1"] == pytest.approx(1.0 / 3.0)
