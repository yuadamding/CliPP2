"""Biological MAP assignments and three-table publication."""
from contextlib import suppress
from dataclasses import dataclass, fields
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .data import readonly_array


OUTPUT_SCHEMA = 18
OUTPUT_SUFFIXES = ("mutation_clusters.tsv", "cluster_centers.tsv",
                   "mutation_region_multiplicity.tsv")


@dataclass(frozen=True)
class BiologicalAssignment:
    """Biological labels and centers with unchanged preprocessing estimates."""

    mutation_ids: tuple
    region_ids: tuple
    component_labels: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    ccf: np.ndarray
    multiplicities: np.ndarray
    log_likelihood: float
    bic: float
    parameter_count: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, readonly_array(value))


def assign_biological(model, fit):
    """Order biological components and retain the frozen multiplicities."""
    template, centers, weights, posterior = fit.template, fit.centers, fit.weights, fit.responsibilities
    q, s = len(template), len(centers)
    if (template not in ((), ("C",))
            or q+s == 0 or centers.shape != (s, model.r) or weights.shape != (q+s,)
            or not np.isfinite(centers).all() or np.any((centers < 0.) | (centers > 1.))
            or np.any(np.all(centers == 1., axis=1)) or len(np.unique(centers, axis=0)) != s
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12)
            or not math.isfinite(fit.log_likelihood) or not math.isfinite(fit.bic)):
        raise ValueError("Selected fit is not an active canonical component model")
    if (posterior.shape != (model.n, q+s) or not np.isfinite(posterior).all() or np.any(posterior < 0.)
            or not np.allclose(posterior.sum(axis=1), 1., rtol=0., atol=1e-10)):
        raise ValueError("Selected fit has invalid responsibilities")

    order = np.r_[np.arange(q), q + np.argsort(-np.linalg.norm(centers, axis=1), kind="stable")]
    component_labels = np.arange(0 if q else 1, s+1)
    biological_centers = np.vstack((np.ones((q, model.r)), centers))[order]
    maximum = np.argmax(posterior[:, order], axis=1)
    labels = component_labels[maximum]  # First maximum: smallest public label.
    if q and not np.isfinite(model.log_clonal[labels == 0]).all():
        raise ValueError("Clonal MAP label has no finite clonal evidence")
    ccf = biological_centers[maximum].copy()
    ccf[model.excluded] = np.nan
    return BiologicalAssignment(model.mutation_ids, model.region_ids, component_labels, labels,
        biological_centers, weights[order], ccf, model.fixed_multiplicity,
        fit.log_likelihood, fit.bic, fit.degrees_of_freedom)


def write_results(tumor_id, assignment, output_dir):
    """Stage and verify three TSVs, rolling back our own links on failure."""
    if (not isinstance(tumor_id, str) or not tumor_id or tumor_id in (".", "..")
            or tumor_id.startswith("#") or tumor_id.strip() != tumor_id
            or any(c in tumor_id for c in "/\\\t\r\n\0")):
        raise ValueError("Tumor ID must be a nonempty safe path component")
    labels, cluster_labels = assignment.labels, assignment.component_labels
    ids, regions = assignment.mutation_ids, assignment.region_ids
    wide = pd.DataFrame(dict(tumor_id=tumor_id, mutation_id=ids, cluster_label=labels))
    cluster = pd.DataFrame(dict(tumor_id=tumor_id, cluster_label=cluster_labels,
        cluster_size=np.bincount(labels, minlength=int(cluster_labels[-1])+1)[cluster_labels]))
    for j, region in enumerate(regions):
        wide["phi_"+region] = assignment.ccf[:, j]
        cluster["phi_"+region] = assignment.centers[:, j]
    long = pd.DataFrame(dict(tumor_id=tumor_id, mutation_id=np.repeat(ids, len(regions)),
        region_id=np.tile(regions, len(ids)), phi=assignment.ccf.ravel(),
        multiplicity_call=assignment.multiplicities.ravel()))
    outdir = Path(output_dir)
    paths = {suffix: outdir/(tumor_id+"_"+suffix) for suffix in OUTPUT_SUFFIXES}
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clipp2-publication-", dir=outdir.parent) as tmp:
        staged = Path(tmp)
        for path, table in zip(paths.values(), (wide, cluster, long)):
            target = staged / path.name
            table.to_csv(target, sep="\t", index=False, float_format="%.17g", na_rep="NA")
            keys = {key: str for key in ("tumor_id", "mutation_id", "region_id") if key in table}
            readback = pd.read_csv(target, sep="\t", dtype=keys, keep_default_na=False,
                na_values={key: ["NA"] for key in table if key not in keys}, float_precision="round_trip")
            pd.testing.assert_frame_equal(table, readback, check_dtype=False, check_exact=True)
        outdir.mkdir(exist_ok=False)
        owner = outdir.lstat()
        try:
            for path in paths.values():
                os.link(staged / path.name, path)
        except BaseException:
            # Never remove an existing destination or any unrelated file.
            with suppress(FileNotFoundError):
                if os.path.samestat(owner, outdir.lstat()):
                    for path in paths.values():
                        with suppress(FileNotFoundError):
                            if os.path.samestat((staged / path.name).lstat(), path.lstat()):
                                path.unlink()
                    outdir.rmdir()
            raise
    return paths
