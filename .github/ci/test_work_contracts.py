"""Focused resource-accounting and bounded-search release contracts."""

from __future__ import annotations

import torch

from CliPP2.core.fusion import certificates as fusion_certificates
from CliPP2.core.fusion import torch_backend as fusion_backend
from CliPP2.model_selection.online_lambda import (
    OnlineLambdaConfig,
    OnlineLambdaController,
    OnlineLambdaObservation,
    OnlineLambdaProposal,
)
from CliPP2.model_selection.proposals import adaptive_stop_certifies_global_optimum


def test_certificate_work_is_charged_once(monkeypatch) -> None:
    phi = torch.full((4, 2), 0.5, dtype=torch.float64)
    edge_index = torch.triu_indices(4, 4, offset=1)
    edge_u, edge_v = edge_index[0], edge_index[1]
    forward_visits = 0
    original_forward = fusion_backend.graph_forward_edges

    def counted_forward(*args, **kwargs):
        nonlocal forward_visits
        forward_visits += int(kwargs["edge_u"].numel()) * int(args[0].shape[1])
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(fusion_backend, "graph_forward_edges", counted_forward)
    certificate = fusion_certificates.refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=torch.zeros_like(phi),
        dual_kkt=torch.zeros((6, 2), dtype=phi.dtype),
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=torch.ones(6, dtype=phi.dtype),
        lambda_value=0.1,
        atol=1e-8,
        max_iter=1,
        edge_work_bytes=1,
    )

    full_pass_visits = int(edge_u.numel()) * int(phi.shape[1])
    assert certificate.status == "input_dual_retained"
    assert forward_visits == full_pass_visits
    audit = certificate.audit
    assert audit.work.edge_region_visits == (
        audit.work.edge_pass_equivalents * full_pass_visits
    )


def test_lambda_no_progress_state_round_trips_and_stays_unresolved() -> None:
    config = OnlineLambdaConfig(
        guide_n_clusters=2,
        num_mutations=4,
        kkt_tolerance=1e-4,
        max_unique_lambdas=4,
        max_refinement_lambdas=4,
        no_progress_patience=1,
    )
    controller = OnlineLambdaController(initial_lambda=10.0, config=config)
    initial = controller.propose()
    assert initial is not None
    controller.observe(
        OnlineLambdaObservation(
            lambda_value=initial.lambda_value,
            n_clusters=2,
            partition_signature="same",
            partition_bic=100.0,
            kkt_residual=1e-8,
            raw_objective_certified=True,
            partition_certified=True,
            selection_score_available=True,
        )
    )
    restored = OnlineLambdaController.from_snapshot(controller.snapshot())
    refinement = OnlineLambdaProposal(
        lambda_value=12.0,
        phase="refine_ci_interval",
        reason="deterministic_no_progress_fixture",
        warm_start_lambda=10.0,
    )
    for item in (controller, restored):
        item._record_proposal(refinement)
        item.observe(
            OnlineLambdaObservation(
                lambda_value=refinement.lambda_value,
                n_clusters=2,
                partition_signature="same",
                partition_bic=100.0,
                kkt_residual=1e-8,
                raw_objective_certified=True,
                partition_certified=True,
                selection_score_available=True,
            )
        )
        assert item.propose() is None
        assert item.stop_reason == "online_lambda_no_meaningful_progress"
        assert not adaptive_stop_certifies_global_optimum(item.stop_reason)
    assert restored.snapshot() == controller.snapshot()
