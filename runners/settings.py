from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import TumorData


@dataclass(frozen=True)
class TumorRegime:
    num_regions: int
    num_mutations: int
    depth_scale: float
    mean_purity: float
    non_diploid_rate: float

    @property
    def num_samples(self) -> int:
        return self.num_regions


def summarize_tumor_regime(data: TumorData) -> TumorRegime:
    non_diploid_mask = data.has_cna & ((data.major_cn != 1.0) | (data.minor_cn != 1.0))
    return TumorRegime(
        num_regions=int(data.num_regions),
        num_mutations=int(data.num_mutations),
        depth_scale=float(data.depth_scale),
        mean_purity=float(np.mean(data.purity)),
        non_diploid_rate=float(np.mean(non_diploid_mask)),
    )


__all__ = [
    "TumorRegime",
    "summarize_tumor_regime",
]
