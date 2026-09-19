"""Execute the actual package through selection, refitting and publication."""
from dataclasses import replace
import hashlib

import numpy as np
import pandas as pd
import pytest

from CliPP2.api import process_tumor
from CliPP2.config import FitConfig, _resolve_fit_options
from CliPP2.model_selection.search import select_model
from CliPP2.reporting import AnalysisSerialization, OUTPUT_SUFFIXES

from ._fixtures import tumor


def test_api_output_reconstructs_exact_occupied_clonality(tmp_path, execution_device):
    path, output = tmp_path/'tiny.tsv', tmp_path/'outputs'
    data = tumor(path, ((24., 24.), (6., 9.), (7., 8.)))
    summary = process_tumor(path, output, FitConfig(device=execution_device))
    publication = summary['publication']
    assert publication['status'] == 'complete'
    assert {p.name for p in output.iterdir()} == {'tiny_'+suffix for suffix in OUTPUT_SUFFIXES}
    assert len(OUTPUT_SUFFIXES) == 4
    for name, record in publication['files'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == record['sha256']
    raw = publication['analysis']['raw_reference']
    assert raw['admissible'] and raw['conditional_kkt_certified']
    assert raw['audit_dtype'] == 'float64' and raw['kkt_tolerance'] == .004
    assert 0 <= raw['kkt_residual'] <= .004
    assert raw['witness_search_complete'] == (not raw['witness_branches_unresolved'])
    wide = pd.read_csv(output/'tiny_mutation_clusters.tsv', sep='\t', float_precision='round_trip')
    centers = pd.read_csv(output/'tiny_cluster_centers.tsv', sep='\t', float_precision='round_trip')
    long = pd.read_csv(output/'tiny_mutation_region_multiplicity.tsv', sep='\t', float_precision='round_trip')
    columns = ['phi_'+r for r in data.region_ids]
    wide = wide.set_index('mutation_id').loc[list(data.mutation_ids)]
    centers = centers.set_index('cluster_label').sort_index()
    labels = wide.cluster_label.to_numpy()
    np.testing.assert_array_equal(wide[columns], centers[columns].to_numpy()[labels])
    np.testing.assert_array_equal(centers.cluster_size, np.bincount(labels, minlength=len(centers)))
    exact = np.all(centers[columns].to_numpy() == 1., axis=1)
    np.testing.assert_array_equal(centers.is_clonal, exact)
    assert exact.any() and centers.loc[exact, 'cluster_size'].ge(1).all()
    for r in data.region_ids:
        rows = long.loc[long.region_id.eq(r)].set_index('mutation_id').loc[list(data.mutation_ids)]
        np.testing.assert_array_equal(rows.phi, wide['phi_'+r])
        np.testing.assert_array_equal(rows.cluster_label, labels)
    with pytest.raises(FileExistsError):
        process_tumor(path, output, FitConfig(device=execution_device))


def test_publication_rejects_forged_raw_certificate(tmp_path):
    path = tmp_path/'certificate.tsv'
    data = tumor(path, ((24., 24.), (24., 24.)))
    config = _resolve_fit_options(FitConfig(device='cpu'))
    selection = select_model(data=data, fit_config=config)
    raw = selection.selected_model.raw_reference
    forged = replace(raw, raw_fit=replace(raw.raw_fit, certificate=replace(
        raw.raw_fit.certificate, conditional_kkt_certified=False)))
    with pytest.raises(ValueError):
        invalid = replace(selection, selected_model=replace(selection.selected_model, raw_reference=forged))
        AnalysisSerialization(data, path, config, invalid)
