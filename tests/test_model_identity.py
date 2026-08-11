from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from CliPP2.core.bic import effective_bic_mutation_region_count, fixed_partition_bic
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.fusion.refit import partition_constrained_observed_refit
from CliPP2.io.data import PathLikelihoodSpec, TumorData
from CliPP2.model_selection.partitions import (
    _partition_signature,
    extract_certified_fusion_partition,
)
from CliPP2.model_selection.scoring import (
    _positive_exact_fusion_selection_mask,
    _select_best_partition_leftmost,
)
from CliPP2.model_selection.types import FusionPartition, PartitionRefitSummary
from CliPP2.runners.outputs import write_fit_outputs


def _toy_data() -> TumorData:
    alt = np.asarray([[2.0], [3.0], [8.0], [9.0]], dtype=np.float64)
    total = np.full_like(alt, 10.0)
    ones = np.ones_like(alt)
    return TumorData(
        tumor_id="identity-test",
        mutation_ids=["m0", "m1", "m2", "m3"],
        region_ids=["region1"],
        alt_counts=alt,
        total_counts=total,
        purity=ones,
        major_cn=np.full_like(alt, 2.0),
        minor_cn=ones,
        normal_cn=np.full_like(alt, 2.0),
        has_cna=np.ones_like(alt, dtype=bool),
        scaling=np.full_like(alt, 0.5),
        phi_upper=ones,
        phi_init=np.asarray([[0.2], [0.3], [0.8], [0.9]]),
        init_major_mask=np.ones_like(alt, dtype=bool),
        count_observed=np.ones_like(alt, dtype=bool),
    )


class ModelIdentityTests(unittest.TestCase):
    def test_chain_partition_fails_closed(self) -> None:
        phi = np.asarray([[0.20000], [0.20009], [0.20018]])
        fit = SimpleNamespace(phi=phi, solver_state=None)
        partition = extract_certified_fusion_partition(
            fit,
            graph=build_complete_uniform_graph(3),
            tolerance=1e-4,
        )
        self.assertEqual(partition.n_clusters, 1)
        self.assertAlmostEqual(partition.max_diameter, 1.8e-4, places=12)
        self.assertFalse(partition.certified)

    def test_refit_never_changes_labels(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        refit = partition_constrained_observed_refit(
            data,
            labels,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-6,
            max_iter=32,
            anchor_mode="clonal_required",
        )
        np.testing.assert_array_equal(refit.labels, labels)
        self.assertIsNotNone(refit.clonal_cluster)
        self.assertLessEqual(
            refit.anchor_deviance_increase,
            refit.second_best_anchor_deviance_increase,
        )

    def test_exact_clonal_anchor_uses_minimum_deviance(self) -> None:
        data = _toy_data()
        data.phi_upper = np.asarray([[0.25], [0.25], [1.0], [1.0]])
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        free = partition_constrained_observed_refit(
            data,
            labels,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-7,
            max_iter=64,
            anchor_mode="none",
        )
        self.assertGreater(free.cluster_centers[1, 0], free.cluster_centers[0, 0])
        anchored = partition_constrained_observed_refit(
            data,
            labels,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-7,
            max_iter=64,
            anchor_mode="clonal_required",
        )
        self.assertEqual(anchored.clonal_cluster, 0)
        self.assertAlmostEqual(anchored.cluster_centers[0, 0], 0.25)
        self.assertLess(
            anchored.anchor_deviance_increase,
            anchored.second_best_anchor_deviance_increase,
        )

    def test_selected_score_reconstructs_exactly(self) -> None:
        data = _toy_data()
        score = fixed_partition_bic(
            loglik=-12.5,
            num_clusters=2,
            data=data,
            anchor_mode="clonal_required",
            partition_signature="2:test",
        )
        expected_df = (2 - 1) * data.num_regions
        expected = 25.0 + expected_df * np.log(
            effective_bic_mutation_region_count(data)
        )
        self.assertEqual(score.degrees_of_freedom, expected_df)
        self.assertAlmostEqual(score.value, expected, places=12)
        self.assertAlmostEqual(score.value, -2.0 * score.loglik + score.penalty)

    def test_only_certified_raw_fusion_candidate_can_win(self) -> None:
        base = {
            "bic_selection_eligible": True,
            "candidate_pool_source": "raw_fused_lambda_path",
            "lambda": 1.0,
            "partition_certified": True,
            "raw_kkt_eligible": True,
            "exactness_provenance_version": 1,
            "estimator_role": "raw_fused_lambda_path",
            "objective_faithful": True,
            "objective_spec_hash": "objective",
            "original_graph_hash": "graph",
            "certificate_problem_hash": "problem",
            "certificate_scope": "full_original_graph",
            "certificate_gradient_scope": "observed_objective",
            "full_kkt_certified": True,
            "full_kkt_certificate_status": "certified",
            "fixed_objective_kkt_residual": 1e-7,
            "full_kkt_tolerance": 1e-6,
        }
        rows = [base, {**base, "candidate_pool_source": "likelihood_partition"}]
        mask = _positive_exact_fusion_selection_mask(pd.DataFrame(rows))
        np.testing.assert_array_equal(mask, [True, False])

    def test_selected_partition_plateau_uses_leftmost_lambda(self) -> None:
        frame = pd.DataFrame(
            {
                "partition_signature": ["A", "A", "B"],
                "selection_score": [10.0, 10.0, 11.0],
                "lambda": [2.0, 1.0, 0.5],
                "penalized_objective": [5.0, 6.0, 1.0],
                "selection_step": [0, 1, 2],
            }
        )
        selected, score, optimal = _select_best_partition_leftmost(
            frame,
            score_column="selection_score",
        )
        self.assertEqual(selected["partition_signature"], "A")
        self.assertEqual(float(selected["lambda"]), 1.0)
        self.assertEqual(score, 10.0)
        np.testing.assert_array_equal(optimal, [True, True, False])

    def test_write_outputs_is_pure_and_explicit(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        raw_phi = np.asarray([[0.2], [0.20001], [0.8], [0.80001]])
        raw_fit = SimpleNamespace(phi=raw_phi, likelihood_eps=1e-6)
        partition = FusionPartition(
            labels=labels,
            signature=_partition_signature(labels),
            n_clusters=2,
            tolerance=1e-4,
            max_diameter=1e-5,
            diameter_exact=True,
            certified=True,
            source="verified_primal_equalities",
        )
        refit_phi = np.asarray([[0.25], [0.25], [1.0], [1.0]])
        refit = PartitionRefitSummary(
            labels=labels.copy(),
            partition_signature=partition.signature,
            phi=refit_phi,
            cluster_centers=np.asarray([[0.25], [1.0]]),
            loglik=-10.0,
            fit_loss=10.0,
            nominal_df=1,
            active_df=1,
            anchor_mode="clonal_required",
            clonal_cluster=1,
            anchor_deviance_increase=0.1,
            second_best_anchor_deviance_increase=2.0,
            finite_candidate_found=True,
            global_optimum_certified=True,
            loglik_source="test",
        )
        before = copy.deepcopy((raw_phi, labels, refit_phi))
        with tempfile.TemporaryDirectory() as directory:
            write_fit_outputs(
                outdir=Path(directory),
                data=data,
                raw_fit=raw_fit,
                partition=partition,
                refit=refit,
            )
            mutation = pd.read_csv(
                Path(directory) / "identity-test_mutation_clusters.tsv", sep="\t"
            )
            clusters = pd.read_csv(
                Path(directory) / "identity-test_cluster_centers.tsv", sep="\t"
            )
        np.testing.assert_array_equal(raw_phi, before[0])
        np.testing.assert_array_equal(labels, before[1])
        np.testing.assert_array_equal(refit_phi, before[2])
        self.assertIn("raw_phi_region1", mutation)
        self.assertIn("fixed_partition_refit_phi_region1", mutation)
        self.assertNotIn("phi_region1", mutation)
        self.assertEqual(clusters["partition_signature"].nunique(), 1)
        self.assertEqual(clusters["partition_signature"].iloc[0], partition.signature)

    def test_path_outputs_are_reported_at_both_profiles(self) -> None:
        data = _toy_data()
        shape = (data.num_mutations, data.num_regions, 1)
        data.path_likelihood = PathLikelihoodSpec(
            model_id="test-path",
            first_copy=np.ones(shape),
            second_copy=np.ones(shape),
            switch_fraction=np.full(shape, 0.5),
            log_prior=np.zeros(shape),
            valid=np.ones(shape, dtype=bool),
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        signature = _partition_signature(labels)
        raw_phi = np.asarray([[0.2], [0.2], [0.8], [0.8]])
        raw_fit = SimpleNamespace(phi=raw_phi, likelihood_eps=1e-6)
        partition = FusionPartition(
            labels=labels,
            signature=signature,
            n_clusters=2,
            tolerance=1e-4,
            max_diameter=0.0,
            diameter_exact=True,
            certified=True,
            source="verified_primal_equalities",
        )
        refit = PartitionRefitSummary(
            labels=labels,
            partition_signature=signature,
            phi=np.asarray([[0.25], [0.25], [1.0], [1.0]]),
            cluster_centers=np.asarray([[0.25], [1.0]]),
            loglik=-10.0,
            fit_loss=10.0,
            nominal_df=1,
            active_df=1,
            anchor_mode="clonal_required",
            clonal_cluster=1,
            anchor_deviance_increase=0.1,
            second_best_anchor_deviance_increase=2.0,
            finite_candidate_found=True,
            global_optimum_certified=False,
            loglik_source="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            write_fit_outputs(
                outdir=Path(directory),
                data=data,
                raw_fit=raw_fit,
                partition=partition,
                refit=refit,
            )
            region = pd.read_csv(
                Path(directory) / "identity-test_mutation_region_multiplicity.tsv",
                sep="\t",
            )
        self.assertIn("raw_map_path", region)
        self.assertIn("refit_map_path", region)
        self.assertIn("raw_path_entropy", region)
        self.assertIn("refit_path_entropy", region)


if __name__ == "__main__":
    unittest.main()
