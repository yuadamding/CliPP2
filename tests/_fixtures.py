"""Synthetic inputs only: no workspace paths, historical checkouts or receipts."""
from dataclasses import replace
import os

import numpy as np
import pandas as pd
import pytest
import torch

from CliPP2.config import CertificateConfig, SolverConfig
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt


@pytest.fixture(params=['cpu', 'cuda'])
def execution_device(request):
    if request.param == 'cuda':
        if os.environ.get('CLIPP2_CUDA_TESTS') != '1':
            pytest.skip('CUDA qualification not requested; CPU skips are not GPU evidence.')
        if not torch.cuda.is_available():
            pytest.fail('CLIPP2_CUDA_TESTS=1 requires an allocated, usable CUDA device.')
    return request.param


def tumor(path, alt=((6., 6.), (18., 18.))):
    counts = np.asarray(alt, dtype=float)
    rows = [dict(mutation_id=f'm{i}', sample_id=f'R{r}', alt_count=value,
                 ref_count=60-value, count_observed=1, purity=.8, normal_cn=2,
                 segment_id=f's{i}', cn_state_id='clonal', cn_state_fraction=1,
                 allele_a_cn=1, allele_b_cn=1)
            for i, row in enumerate(counts) for r, value in enumerate(row)]
    write_tumor_txt(path, pd.DataFrame(rows))
    return load_tumor_txt(path)


def context(data, *, device='cpu', dtype='float64', initialize=True):
    starts = {} if initialize else dict(exact_pilot=data.phi_init, pooled_start=data.phi_init,
                                        scalar_well_starts=())
    return solver.prepare_torch_problem(
        data, eps=1e-6, tol=8e-4, inner_max_iter=64,
        graph=build_complete_uniform_graph(data.num_mutations),
        device=device, dtype=dtype, **starts,
    )


def options():
    return SolverConfig(8, 64, 8e-4, 'auto', CertificateConfig(128, 1))


def two_blocks(tmp_path):
    data = tumor(tmp_path/'blocks.tsv')
    return replace(data, alt_counts=np.array([[24., 18.], [6., 9.]]))
