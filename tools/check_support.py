"""Shared deterministic fixtures and assertions; no runnable checks or runner imports."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from CliPP2 import FitConfig, WorkflowConfig
from CliPP2.assignment import AssignmentBundle, AssignmentReference
from CliPP2.config import PROTOCOL_VERSION
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt


def write_catalog(path, alt, depth, major, *, minor=0, purity=None, observed=True, metadata=None):
    alt, depth, major, minor, observed = np.broadcast_arrays(
        np.atleast_2d(alt), np.atleast_2d(depth), np.atleast_2d(major),
        np.atleast_2d(minor), np.atleast_2d(observed))
    purity = np.ones(alt.shape[1]) if purity is None else np.asarray(purity)
    rows = [dict(mutation_id=f"m{i:03}", sample_id=f"r{r+1}",
        alt_count=int(alt[i, r]), ref_count=int(depth[i, r]-alt[i, r]), count_observed=int(observed[i, r]),
        purity=float(purity[r]), normal_cn=2, segment_id="shared_segment", cn_state_id="0",
        cn_state_fraction=1., allele_a_cn=int(major[i, r]), allele_b_cn=int(minor[i, r]))
        for i in range(alt.shape[0]) for r in range(alt.shape[1])]
    write_tumor_txt(path, pd.DataFrame(rows), metadata)
    # These controlled fixtures declare a numerical observation mask, not the
    # production default absence assumption. Preserve explicit missing fixtures.
    return load_original_catalog(path, missing_counts="zero_alt" if observed.all() else "unobserved")


def fixture_reference(labels, region_ids, centers, *args, **kwargs):
    """Declare controlled coordinates without inventing diagnostic measurements."""
    kwargs.setdefault("center_sources", tuple(dict(center_source="declared_conditional_fixture") for _ in labels))
    return AssignmentReference(labels, region_ids, centers, *args, **kwargs)


def workflow_config(**changes):
    config = WorkflowConfig(fit=FitConfig(device="cpu", max_clusters=10, cpu_threads=1))
    return replace(config, **changes)


def external_build_record(identifier="contract-external-build"):
    """Explicit test-harness evidence; inference must not gather this itself."""
    from CliPP2._version import __version__
    from CliPP2.kernel.build_identity import backend_identity, native_identity
    return dict(schema="clipp2.run-build.v1", run_build_id=identifier,
        source=dict(kind="declared_contract_test_installation"), environment=dict(purpose="contract_test"),
        artifact=dict(implementation_version=__version__, protocol_version=PROTOCOL_VERSION,
                      native_build_id=native_identity(), cpu_build_id=backend_identity("cpu"),
                      cuda_build_id=backend_identity("cuda")))


def bundle(catalog, reference, *, target_ids=None, calls=None, native_upper=None, native_centers=None):
    targets = catalog.mutation_ids if target_ids is None else tuple(target_ids)
    support = catalog.default_bounds().subset(targets, reference.region_ids)
    shape = len(targets), len(reference.region_ids)
    data = catalog.prediction_data(targets)
    observed = data.count_observed & (data.total_counts > 0)
    baseline = 0 if 0 in reference.labels else reference.labels[0]
    return AssignmentBundle(catalog, reference, targets, (),
        np.ones(shape) if native_centers is None else native_centers,
        np.where(observed, 1., np.nan) if calls is None else calls,
        support.lower, support.upper if native_upper is None else native_upper,
        ("regression_fixture",)*len(targets), (0,)*len(targets), (baseline,)*len(targets), "cpu", 1)


def expect_value_error(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid evidence was accepted")


def assert_published_files(receipt, directory):
    assert receipt["manifest_schema"] == "clipp2.file-list.v1"
    assert isinstance(receipt["files"], (tuple, list))
    assert all(isinstance(name, str) and (directory / name).is_file() for name in receipt["files"])
