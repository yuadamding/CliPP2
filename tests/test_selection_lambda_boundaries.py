"""Partition deduplication must not discard admitted lambda-range evidence."""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest
import torch

from CliPP2.core.bic import SelectionScore
from CliPP2.core.fusion.types import DenseEdgeCertificate, KKTComponents
from CliPP2.core.objective import make_lambda_objective_key
from CliPP2.model_selection.partitions import _partition_signature
from CliPP2.model_selection.scoring import select_candidate_records
from CliPP2.model_selection.types import (
    CandidateRecord, DirectPartition, DirectPartitionCandidate, RawFusionCandidate,
)
from test_integer_reporting import _data, _selection


def _raw(candidate_id, labels, lambda_value, score_value):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    data = _data(major=(2,) * n, minor=(1,) * n, alt=(25,) * n, total=(100,) * n)
    fit, partition, refit = _selection(data)
    signature = _partition_signature(labels, data.mutation_ids)
    centers = np.arange(len(np.unique(labels)))[:, None] * .1 + .3
    partition = replace(partition, labels=labels, signature=signature)
    refit = replace(
        refit, labels=labels, partition_signature=signature,
        phi=centers[labels], cluster_centers=centers,
    )
    fit = replace(
        fit, provenance=replace(fit.provenance, objective_key=make_lambda_objective_key(
            fit.provenance.objective_key.base, lambda_value=lambda_value,
        )), certificate=replace(
            fit.certificate, components=KKTComponents(0, 0, 0, 0),
            certified=True, admissible=True, status="certified",
            residual_method="componentwise_box_cone_backward_error_v1",
        ),
    )
    score = SelectionScore(
        "fixed_partition_dirichlet_score", score_value, refit.loglik, 0.0,
        partition.n_clusters, n, signature,
    )
    return CandidateRecord(candidate_id, RawFusionCandidate(
        fit, partition, refit, score, True, "none",
    ))


def _direct(record, candidate_id, *, parent_lambda=1000.0):
    raw = record.candidate
    partition = DirectPartition(
        raw.partition.labels, raw.partition.signature, "final_phi_hessian_ward",
        raw.partition.mutation_ids, parent_raw_candidate_id=record.candidate_id,
        parent_raw_lambda=parent_lambda, parent_raw_phi_hash="fixture-parent",
    )
    return CandidateRecord(candidate_id, DirectPartitionCandidate(
        partition, raw.refit, raw.score, True, "none",
    ))


def _flags(decision):
    return (
        decision.selection_hits_lower_boundary,
        decision.selection_hits_upper_boundary,
        decision.selection_boundary_unresolved,
    )


def test_duplicate_partition_at_later_lambda_preserves_explored_range():
    # Zero-lambda multi-mutation fits are not eligible; use 0.1 for A's first fit.
    left = _raw(0, [0, 0, 1, 1], .1, 20.0)
    middle = _raw(1, [0, 1, 2, 3], 1.0, 10.0)
    right = _raw(2, [0, 0, 1, 1], 10.0, 20.0)
    before = select_candidate_records([left, middle])
    assert _flags(before) == (False, True, True)
    for records in permutations((left, middle, right)):
        decision = select_candidate_records(list(records))
        assert decision.selected is middle
        assert decision.selected.candidate.refit is before.selected.candidate.refit
        assert decision.selected.score is before.selected.score
        assert decision.num_eligible == 2
        assert decision.selected_lambda_left == decision.selected_lambda_right == 1.0
        assert _flags(decision) == (False, False, False)
        redundant = replace(right, candidate_id=3)
        assert select_candidate_records([*records, redundant]) == decision


def test_boundary_flags_refer_to_representative_not_partition_support_interval():
    left = _raw(0, [0, 0, 1, 1], .1, 10.0)
    right = _raw(1, [0, 0, 1, 1], 10.0, 10.0)
    decision = select_candidate_records([right, left])
    assert decision.selected is left
    assert (decision.selected_lambda_left, decision.selected_lambda_right) == (.1, 10.0)
    assert _flags(decision) == (True, False, True)


def test_raw_range_survives_a_direct_representative_of_the_same_partition():
    left = _raw(0, [0, 0, 1, 1], .1, 20.0)
    middle = _raw(1, [0, 1, 2, 3], 1.0, 10.0)
    right = _raw(2, [0, 0, 1, 1], 10.0, 20.0)
    direct = _direct(left, 3)
    direct = replace(direct, candidate=replace(
        direct.candidate, refit=replace(
            direct.candidate.refit, global_optimum_certified=True,
            global_lower_bound=-direct.candidate.refit.loglik,
            global_optimality_gap=0.0, global_certificate_method="fixture",
        ),
    ))
    # The resolved direct refit wins A's representative tie; its raw records
    # must nevertheless retain the explored endpoints 0.1 and 10.
    assert select_candidate_records([left, right, direct]).selected is direct
    decision = select_candidate_records([left, middle, right, direct])
    assert decision.selected is middle
    assert decision.num_eligible == 2
    assert _flags(decision) == (False, False, False)


@pytest.mark.parametrize("failure", ["ineligible", "audit", "partition", "zero_lambda"])
def test_unadmitted_raw_records_do_not_extend_lambda_range(failure):
    left = _raw(0, [0, 0, 1, 1], .1, 20.0)
    selected = _raw(1, [0, 1, 2, 3], 1.0, 10.0)
    outside = _raw(2, [0, 0, 1, 1], 100.0, 20.0)
    candidate = outside.candidate
    if failure == "ineligible":
        candidate = replace(candidate, eligible_for_selection=False)
    elif failure == "audit":
        candidate = replace(candidate, raw_fit=replace(
            candidate.raw_fit, certificate=replace(
                candidate.raw_fit.certificate, components=KKTComponents(1, 0, 0, 0),
            ),
        ))
    elif failure == "partition":
        candidate = replace(candidate, partition=replace(candidate.partition, certified=False))
    else:
        candidate = replace(candidate, raw_fit=replace(
            candidate.raw_fit, provenance=replace(
                candidate.raw_fit.provenance,
                objective_key=make_lambda_objective_key(
                    candidate.raw_fit.provenance.objective_key.base, lambda_value=0.0,
                ),
            ),
        ))
    expected = select_candidate_records([left, selected])
    assert select_candidate_records([left, selected, replace(outside, candidate=candidate)]) == expected


def test_direct_parent_lambda_is_not_raw_exploration_evidence():
    raw = _raw(0, [0, 0, 1, 1], 1.0, 10.0)
    direct = _direct(_raw(1, [0, 1, 2, 3], 5.0, 20.0), 2)
    decision = select_candidate_records([raw, direct])
    assert decision.selected is raw
    assert _flags(decision) == (True, True, True)
    direct = replace(direct, candidate=replace(
        direct.candidate, score=replace(direct.score, value=5.0),
    ))
    selected_direct = select_candidate_records([raw, direct])
    assert selected_direct.selected is direct
    assert selected_direct.selected_lambda_left is selected_direct.selected_lambda_right is None
    assert _flags(selected_direct) == (False, False, False)
    assert select_candidate_records([direct]) == replace(selected_direct, num_eligible=1)


def test_certified_zero_edge_singleton_keeps_zero_lambda_boundary():
    record = _raw(0, [0], 0.0, 10.0)
    fit = record.candidate.raw_fit
    fit = replace(fit, certificate=replace(
        fit.certificate, witness=DenseEdgeCertificate(
            torch.empty((0, 1)), fit.provenance.original_graph_hash, "observed_objective",
        ),
    ))
    record = replace(record, candidate=replace(record.candidate, raw_fit=fit))
    decision = select_candidate_records([record])
    assert decision.selected is record
    assert decision.selected_lambda_left == decision.selected_lambda_right == 0.0
    assert _flags(decision) == (True, True, True)
