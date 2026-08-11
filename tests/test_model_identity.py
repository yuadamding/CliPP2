from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from CliPP2.core.bic import effective_bic_mutation_region_count, fixed_partition_bic
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.fusion.refit import partition_constrained_observed_refit
from CliPP2.core.fusion.solver import prepare_torch_problem
from CliPP2.core.fusion.types import (
    CompressedEdgeCertificate,
    InfeasibleRawClonalAnchor,
    QuotientWorksetWarmState,
    SolverState,
)
from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.io.data import PathLikelihoodSpec, TumorData
from CliPP2.model_selection.partitions import (
    _partition_signature,
    extract_certified_fusion_partition,
)
from CliPP2.model_selection.candidates import (
    _anchor_block_signature,
    _build_refit_summary,
    _fixed_partition_refit,
    _selection_refit_cache_key,
    _selection_score_diagnostics,
    validate_candidate_identity,
)
from CliPP2.model_selection.scoring import (
    _positive_exact_fusion_selection_mask,
    _select_best_partition_leftmost,
)
from CliPP2.model_selection.types import (
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    SelectedModel,
)
from CliPP2.runners.outputs import write_fit_outputs
from CliPP2.runners.model_selection import _build_raw_clonal_anchor_search


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

    def test_tolerance_partition_merges_finer_quotient_blocks(self) -> None:
        phi = np.asarray([[0.2], [0.20005]])
        warm_state = QuotientWorksetWarmState(
            phi=torch.as_tensor(phi),
            labels=torch.as_tensor([0, 1]),
            centers=torch.as_tensor(phi),
            quotient_dual=None,
            internal_edge_ids=torch.empty(0, dtype=torch.int64),
            internal_dual=torch.empty((0, 1), dtype=torch.float64),
            graph_hash="graph",
            previous_lambda=1.0,
        )
        fit = SimpleNamespace(
            phi=phi,
            solver_state=SimpleNamespace(warm_state=warm_state, certificate=None),
        )
        partition = extract_certified_fusion_partition(
            fit,
            graph=build_complete_uniform_graph(2),
            tolerance=1e-4,
        )
        self.assertEqual(partition.n_clusters, 1)
        self.assertTrue(partition.certified)
        self.assertTrue(partition.maximal)
        self.assertEqual(partition.source, "tolerance_defined_primal")

    def test_compressed_certificate_must_match_original_graph(self) -> None:
        phi = np.asarray([[0.2], [0.8]])
        certificate = CompressedEdgeCertificate(
            labels=torch.as_tensor([0, 1]),
            centers=torch.as_tensor(phi),
            internal_edge_ids=torch.empty(0, dtype=torch.int64),
            internal_dual=torch.empty((0, 1), dtype=torch.float64),
            graph_hash="wrong-graph",
            gradient_scope="observed_objective",
        )
        fit = SimpleNamespace(
            phi=phi,
            original_graph_hash="right-graph",
            solver_state=SimpleNamespace(certificate=certificate),
        )
        partition = extract_certified_fusion_partition(
            fit,
            graph=build_complete_uniform_graph(2),
            tolerance=1e-4,
        )
        self.assertFalse(partition.certified)
        self.assertFalse(partition.certificate_graph_hash_matches)
        self.assertEqual(
            partition.certification_failure_reason,
            "compressed_certificate_graph_hash_mismatch",
        )

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

    def test_explicit_raw_anchor_cluster_is_preserved_by_refit(self) -> None:
        data = _toy_data()
        data.phi_upper = np.asarray([[0.25], [0.25], [1.0], [1.0]])
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        refit = partition_constrained_observed_refit(
            data,
            labels,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-7,
            max_iter=64,
            anchor_mode="clonal_required",
            anchor_cluster=1,
        )
        self.assertEqual(refit.clonal_cluster, 1)
        self.assertAlmostEqual(refit.cluster_centers[1, 0], 1.0)
        self.assertEqual(
            refit.loglik_source,
            "raw_clonal_anchor_preserved_partition_observed_mle",
        )

    def test_raw_anchor_is_a_frozen_hashed_objective_coordinate(self) -> None:
        data = _toy_data()
        kwargs = dict(
            major_prior=0.5,
            eps=1e-6,
            tol=1e-6,
            inner_max_iter=32,
            defer_graph=True,
            device="cpu",
            dtype="float64",
        )
        pilot = prepare_torch_problem(data, **kwargs)
        options = FitOptions(
            lambda_value=0.0,
            raw_clonal_anchor_mode="screened_seed",
            raw_clonal_anchor_candidate_max=2,
        )
        search = _build_raw_clonal_anchor_search(
            data,
            pilot,
            fit_options=options,
        )
        anchor_index = int(search.spec.candidate_mutation_indices[0])
        anchored = prepare_torch_problem(
            data,
            **kwargs,
            clonal_anchor_mutation_index=anchor_index,
            clonal_anchor_target=search.spec.target,
            clonal_anchor_source=search.screening_rule,
            clonal_anchor_mode=search.spec.mode,
            clonal_anchor_feasibility_tolerance=search.spec.feasibility_tolerance,
        )
        index = anchor_index
        self.assertNotEqual(pilot.objective_spec_hash, anchored.objective_spec_hash)
        self.assertTrue(torch.equal(anchored.lower[index], anchored.upper[index]))
        self.assertTrue(
            torch.equal(anchored.exact_pilot[index], anchored.upper[index])
        )
        np.testing.assert_array_equal(
            anchored.clonal_anchor_target.detach().cpu().numpy(),
            search.spec.target,
        )
        second_index = int(search.spec.candidate_mutation_indices[1])
        second = prepare_torch_problem(
            data,
            **kwargs,
            clonal_anchor_mutation_index=second_index,
            clonal_anchor_target=search.spec.target,
            clonal_anchor_source=search.screening_rule,
            clonal_anchor_mode=search.spec.mode,
            clonal_anchor_feasibility_tolerance=search.spec.feasibility_tolerance,
        )
        self.assertNotEqual(anchored.objective_spec_hash, second.objective_spec_hash)
        self.assertFalse(search.search_complete)
        self.assertEqual(search.total_eligible_candidates, data.num_mutations)
        self.assertEqual(len(search.spec.candidate_mutation_indices), 2)

    def test_infeasible_raw_anchor_fails_and_warm_states_are_seed_specific(self) -> None:
        data = _toy_data()
        data.phi_upper[0, 0] = 0.5
        common = dict(
            major_prior=0.5,
            eps=1e-6,
            tol=1e-6,
            inner_max_iter=32,
            graph=build_complete_uniform_graph(data.num_mutations),
            device="cpu",
            dtype="float64",
            clonal_anchor_target=np.ones(data.num_regions),
            clonal_anchor_source="unit_test",
            clonal_anchor_mode="specified_seed",
            clonal_anchor_feasibility_tolerance=1e-8,
        )
        with self.assertRaises(InfeasibleRawClonalAnchor):
            prepare_torch_problem(
                data,
                **common,
                clonal_anchor_mutation_index=0,
            )
        first = prepare_torch_problem(
            data,
            **common,
            clonal_anchor_mutation_index=1,
        )
        second = prepare_torch_problem(
            data,
            **common,
            clonal_anchor_mutation_index=2,
        )
        state = SolverState(
            phi=first.exact_pilot,
            dual=None,
            previous_lambda=1.0,
            objective_spec_hash=first.objective_spec_hash,
        )
        with self.assertRaisesRegex(ValueError, "different raw objective"):
            fit_fixed_objective(
                data,
                FitOptions(
                    lambda_value=1.0,
                    graph=common["graph"],
                    device="cpu",
                    dtype="float64",
                ),
                phi_start=second.exact_pilot,
                exact_pilot=second.exact_pilot,
                pooled_start=second.exact_pilot,
                scalar_well_starts=[],
                start_mode="warm_only",
                runtime=second.runtime,
                torch_data=None,
                solver_context=second,
                solver_state=state,
                compute_summary=False,
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

    def test_missing_nonanchor_coordinates_do_not_count_toward_bic_df(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        data.count_observed[labels == 1, 0] = False
        score = fixed_partition_bic(
            loglik=-12.5,
            num_clusters=2,
            data=data,
            anchor_mode="clonal_required",
            partition_signature="2:test",
            anchor_block_signature="2:anchor",
            labels=labels,
            anchor_cluster=0,
        )
        self.assertEqual(score.degrees_of_freedom, 0)
        self.assertEqual(score.penalty, 0.0)

    def test_selection_refit_is_independent_of_raw_retry_effort(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        partition = FusionPartition(
            labels=labels,
            signature=_partition_signature(labels),
            n_clusters=2,
            tolerance=1e-4,
            max_diameter=0.0,
            diameter_exact=True,
            certified=True,
            source="tolerance_defined_primal",
            maximal=True,
        )
        base = FitOptions(lambda_value=0.0)
        recovery = replace(
            base,
            tol=0.5 * base.tol,
            inner_max_iter=5 * base.inner_max_iter,
            outer_max_iter=5 * base.outer_max_iter,
        )
        self.assertEqual(
            _selection_refit_cache_key(
                data=data,
                partition=partition,
                selection_options=base,
                raw_anchor_cluster=1,
            ),
            _selection_refit_cache_key(
                data=data,
                partition=partition,
                selection_options=recovery,
                raw_anchor_cluster=1,
            ),
        )
        cache = {}
        first, first_hit = _fixed_partition_refit(
            data=data,
            partition=partition,
            selection_options=base,
            cache=cache,
            raw_anchor_cluster=1,
        )
        second, second_hit = _fixed_partition_refit(
            data=data,
            partition=partition,
            selection_options=recovery,
            cache=cache,
            raw_anchor_cluster=1,
        )
        self.assertFalse(first_hit)
        self.assertTrue(second_hit)
        np.testing.assert_array_equal(
            first.result.cluster_centers,
            second.result.cluster_centers,
        )
        self.assertEqual(first.result.loglik, second.result.loglik)
        first_score = fixed_partition_bic(
            loglik=first.result.loglik,
            num_clusters=partition.n_clusters,
            data=data,
            anchor_mode=base.selection_anchor,
            partition_signature=partition.signature,
        )
        second_score = fixed_partition_bic(
            loglik=second.result.loglik,
            num_clusters=partition.n_clusters,
            data=data,
            anchor_mode=base.selection_anchor,
            partition_signature=partition.signature,
        )
        self.assertEqual(first_score.value, second_score.value)

    def test_refit_certificate_is_honest_and_unanchored_bic_is_consistent(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        partition = FusionPartition(
            labels=labels,
            signature=_partition_signature(labels),
            n_clusters=2,
            tolerance=1e-4,
            max_diameter=0.0,
            diameter_exact=True,
            certified=True,
            source="tolerance_defined_primal",
            maximal=True,
        )
        options = FitOptions(
            lambda_value=0.0,
            selection_score="fixed_partition_bic",
            selection_anchor="none",
        )
        resolution, _ = _fixed_partition_refit(
            data=data,
            partition=partition,
            selection_options=options,
            cache={},
        )
        score = fixed_partition_bic(
            loglik=resolution.result.loglik,
            num_clusters=partition.n_clusters,
            data=data,
            anchor_mode="none",
            partition_signature=partition.signature,
        )
        refit = _build_refit_summary(
            resolution.result,
            partition_signature=partition.signature,
            nominal_df=score.degrees_of_freedom,
            resolution=resolution,
        )
        diagnostics = _selection_score_diagnostics(
            data=data,
            refit=refit,
            score=score,
        )
        self.assertFalse(refit.global_optimum_certified)
        self.assertTrue(refit.refit_numerically_resolved)
        self.assertEqual(score.degrees_of_freedom, 2 * data.num_regions)
        self.assertEqual(refit.nominal_df, score.degrees_of_freedom)
        self.assertAlmostEqual(diagnostics["classic_bic"], score.value)

    def test_selected_model_enforces_full_identity_contract(self) -> None:
        data = _toy_data()
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        partition = FusionPartition(
            labels=labels,
            signature=_partition_signature(labels),
            n_clusters=2,
            tolerance=1e-4,
            max_diameter=0.0,
            diameter_exact=True,
            certified=True,
            source="tolerance_defined_primal",
            maximal=True,
        )
        options = FitOptions(lambda_value=0.0)
        resolution, _ = _fixed_partition_refit(
            data=data,
            partition=partition,
            selection_options=options,
            cache={},
        )
        score = fixed_partition_bic(
            loglik=resolution.result.loglik,
            num_clusters=partition.n_clusters,
            data=data,
            anchor_mode=options.selection_anchor,
            partition_signature=partition.signature,
            anchor_block_signature=_anchor_block_signature(
                labels, int(resolution.result.clonal_cluster)
            ),
        )
        refit = _build_refit_summary(
            resolution.result,
            partition_signature=partition.signature,
            nominal_df=score.degrees_of_freedom,
            resolution=resolution,
        )
        anchor_cluster = int(resolution.result.clonal_cluster)
        anchor_index = int(np.flatnonzero(labels == anchor_cluster)[0])
        raw_phi = np.asarray(resolution.result.phi, dtype=np.float64).copy()
        raw_fit = SimpleNamespace(
            lambda_value=1.0,
            phi=raw_phi,
            raw_clonal_anchor_mutation_index=anchor_index,
            raw_clonal_anchor_target=raw_phi[anchor_index].copy(),
            raw_clonal_anchor_source="unit_test_raw_anchor",
        )
        candidate = RawFusionCandidate(
            raw_fit=raw_fit,
            partition=partition,
            refit=refit,
            score=score,
            raw_objective_certified=True,
            eligible_for_selection=True,
            ineligibility_reason="none",
            anchor_seed_index=anchor_index,
            anchor_seed_mutation_id=data.mutation_ids[anchor_index],
            anchor_cluster_label=anchor_cluster,
            anchor_block_signature=score.anchor_block_signature,
            anchor_target=raw_phi[anchor_index].copy(),
            anchor_search_complete=True,
        )
        validate_candidate_identity(candidate)
        SelectedModel(
            candidate=candidate,
            selected_lambda=1.0,
            selected_partition_signature=partition.signature,
            selected_partition_left_lambda=1.0,
            selected_partition_right_lambda=1.0,
        )
        with self.assertRaisesRegex(ValueError, "eligible"):
            SelectedModel(
                candidate=replace(candidate, eligible_for_selection=False),
                selected_lambda=1.0,
                selected_partition_signature=partition.signature,
                selected_partition_left_lambda=1.0,
                selected_partition_right_lambda=1.0,
            )

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

    def test_different_anchor_blocks_are_different_selection_models(self) -> None:
        frame = pd.DataFrame(
            {
                "partition_signature": ["same", "same"],
                "selection_model_signature": ["same|anchor:a", "same|anchor:b"],
                "selection_score": [12.0, 10.0],
                "lambda": [1.0, 2.0],
                "penalized_objective": [5.0, 6.0],
                "selection_step": [0, 1],
            }
        )
        selected, score, optimal = _select_best_partition_leftmost(
            frame,
            score_column="selection_score",
        )
        self.assertEqual(selected["selection_model_signature"], "same|anchor:b")
        self.assertEqual(score, 10.0)
        np.testing.assert_array_equal(optimal, [False, True])

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
            global_optimum_certified=False,
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
