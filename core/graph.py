from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GraphData:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])


def _resolve_graph_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_knn_graph(
    phi_matrix: np.ndarray,
    k: int = 10,
    min_weight: float = 1e-4,
    device: str = "cpu",
) -> GraphData:
    phi_matrix = np.asarray(phi_matrix, dtype=np.float32)
    num_mutations = phi_matrix.shape[0]
    if num_mutations <= 1:
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return GraphData(src=empty_int, dst=empty_int, weight=empty_float)

    k = max(1, min(int(k), num_mutations - 1))
    torch_device = _resolve_graph_device(device)
    phi_t = torch.as_tensor(phi_matrix, dtype=torch.float32, device=torch_device)
    distances = torch.cdist(phi_t, phi_t, p=2)
    distances.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = torch.topk(distances, k=k, largest=False, dim=1)

    local_scale = knn_dist[:, -1].clone()
    positive = local_scale > 0
    scale_fallback = torch.median(local_scale[positive]) if torch.any(positive) else torch.tensor(1.0, device=torch_device)
    local_scale = torch.where(local_scale > 0, local_scale, scale_fallback)
    local_scale = torch.clamp(local_scale, min=1e-6)

    src = torch.arange(num_mutations, device=torch_device).unsqueeze(1).expand(num_mutations, k).reshape(-1)
    dst = knn_idx.reshape(-1)
    dist = knn_dist.reshape(-1)
    a = torch.minimum(src, dst)
    b = torch.maximum(src, dst)
    keep = a != b
    if not torch.any(keep):
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return GraphData(src=empty_int, dst=empty_int, weight=empty_float)

    src = a[keep]
    dst = b[keep]
    dist = dist[keep]
    denom = local_scale[src] * local_scale[dst]
    weight = torch.exp(-(dist.square()) / torch.clamp(denom, min=1e-6))
    keep = weight >= float(min_weight)
    if not torch.any(keep):
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return GraphData(src=empty_int, dst=empty_int, weight=empty_float)

    src_np = src[keep].detach().cpu().numpy().astype(np.int64)
    dst_np = dst[keep].detach().cpu().numpy().astype(np.int64)
    weight_np = weight[keep].detach().cpu().numpy().astype(np.float32)

    order = np.lexsort((dst_np, src_np))
    src_np = src_np[order]
    dst_np = dst_np[order]
    weight_np = weight_np[order]

    src_out: list[int] = []
    dst_out: list[int] = []
    weight_out: list[float] = []
    current_src = int(src_np[0])
    current_dst = int(dst_np[0])
    current_weight = float(weight_np[0])
    for src_val, dst_val, weight_val in zip(src_np[1:], dst_np[1:], weight_np[1:]):
        src_int = int(src_val)
        dst_int = int(dst_val)
        weight_float = float(weight_val)
        if src_int == current_src and dst_int == current_dst:
            if weight_float > current_weight:
                current_weight = weight_float
            continue
        src_out.append(current_src)
        dst_out.append(current_dst)
        weight_out.append(current_weight)
        current_src = src_int
        current_dst = dst_int
        current_weight = weight_float

    src_out.append(current_src)
    dst_out.append(current_dst)
    weight_out.append(current_weight)

    return GraphData(
        src=np.asarray(src_out, dtype=np.int64),
        dst=np.asarray(dst_out, dtype=np.int64),
        weight=np.asarray(weight_out, dtype=np.float32),
    )
