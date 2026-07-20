from __future__ import annotations

import numpy as np

from CliPP2.core.fusion.solver import cluster_diameters_from_edges


def test_complete_graph_cluster_diameter_exposes_transitive_spread() -> None:
    phi = np.asarray([[0.0], [0.009], [0.018]], dtype=np.float64)
    labels = np.asarray([0, 0, 0], dtype=np.int64)
    edge_u = np.asarray([0, 0, 1], dtype=np.int64)
    edge_v = np.asarray([1, 2, 2], dtype=np.int64)

    diameters, exact = cluster_diameters_from_edges(
        phi,
        labels,
        edge_u=edge_u,
        edge_v=edge_v,
    )

    assert exact
    assert diameters.tolist() == [0.018]


def test_sparse_graph_cluster_diameter_is_marked_inexact() -> None:
    phi = np.asarray([[0.0], [0.009], [0.018]], dtype=np.float64)
    labels = np.asarray([0, 0, 0], dtype=np.int64)
    edge_u = np.asarray([0, 1], dtype=np.int64)
    edge_v = np.asarray([1, 2], dtype=np.int64)

    diameters, exact = cluster_diameters_from_edges(
        phi,
        labels,
        edge_u=edge_u,
        edge_v=edge_v,
    )

    assert not exact
    assert diameters.tolist() == [0.009]
