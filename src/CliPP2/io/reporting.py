"""Compact native-stage tables and no-clobber publication."""
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

OUTPUT_SUFFIXES = ('mutation_clusters.tsv', 'cluster_centers.tsv',
                   'mutation_region_multiplicity.tsv')
MANIFEST_SCHEMA = 'clipp2.file-list.v1'

def public_label_order(fit):
    """The single public-label convention used by native-stage tables."""
    # Cluster zero is largest norm, never an enforced clonal block.
    order = np.argsort(-np.linalg.norm(fit.centers, axis=1), kind='stable')
    mapping = np.empty(len(order), dtype=int)
    mapping[order] = np.arange(len(order))
    return order, mapping[fit.labels]


def result_tables(model, fit, tumor_id):
    labels, centers = fit.labels, fit.centers
    order, public = public_label_order(fit)
    phi = centers[labels]
    wide = pd.DataFrame(dict(tumor_id=tumor_id, mutation_id=model.mutation_ids, cluster_label=public))
    cluster = pd.DataFrame(dict(tumor_id=tumor_id, cluster_label=np.arange(len(order)),
                                cluster_size=np.bincount(public, minlength=len(order))))
    calls = model.multiplicity(phi)
    calls = np.where(~model.observed & (model.support == 1), model.multiplicity_min, calls)
    calls = np.where(phi == 0, np.nan, calls)
    report = model.data.cn_filter_report
    if report is not None and report.records:
        mutations = {name: i for i, name in enumerate(model.mutation_ids)}
        regions = {name: j for j, name in enumerate(model.region_ids)}
        for record in report.records:
            i, j = mutations.get(record.mutation_id), regions.get(record.sample_id)
            if i is not None and j is not None:
                phi[i, j] = calls[i, j] = np.nan
    for j, region in enumerate(model.region_ids):
        wide['phi_'+region] = phi[:, j]
        cluster['phi_'+region] = centers[order, j]
    long = pd.DataFrame(dict(tumor_id=tumor_id,
        mutation_id=np.repeat(model.mutation_ids, model.r), region_id=np.tile(model.region_ids, model.n),
        phi=phi.ravel(), multiplicity_call=calls.ravel()))
    return dict(zip(OUTPUT_SUFFIXES, (wide, cluster, long)))


def check_destination(outdir, tumor_id):
    if not tumor_id or any(c in tumor_id for c in '/\\\t\r\n'):
        raise ValueError('Tumor ID must be a nonempty single path component')
    outdir = Path(outdir)
    if outdir.exists() and (not outdir.is_dir() or any(
            p.name.startswith(tumor_id+'_') for p in outdir.iterdir())):
        raise FileExistsError('Tumor output namespace already exists; choose a new directory')


def publish(model, fit, outdir, tumor_id, *, progress=None):
    if progress is not None:
        progress.start_phase('output')
    check_destination(outdir, tumor_id)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables, files = result_tables(model, fit, tumor_id), []
    # Per-file no-clobber publication, not a collective transaction.
    with tempfile.TemporaryDirectory(prefix='.clipp2-publication-', dir=outdir) as temporary:
        staged = Path(temporary)
        for suffix, table in tables.items():
            name = tumor_id+'_'+suffix
            table.to_csv(staged/name, sep='\t', index=False, float_format='%.17g', na_rep='NA')
            files.append(name)
        for name in files:
            os.link(staged/name, outdir/name)
    if progress is not None:
        progress.emit(complete=True)
