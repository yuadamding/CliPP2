"""Biological MAP assignments and three-table publication."""
from contextlib import suppress
from dataclasses import dataclass, fields
from itertools import product
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from ..kernel.clonal_mixture import binomial_log_mass
from .data import readonly_array


OUTPUT_SCHEMA = 18
OUTPUT_SUFFIXES = ("mutation_clusters.tsv", "cluster_centers.tsv",
                   "mutation_region_multiplicity.tsv")
_ENUMERATE_MAX = 512


def _uniform_map(masses, upper, count):
    if count <= _ENUMERATE_MAX:
        # Keep exact fsum/lexicographic ordering for the small support.
        candidates = (vector for h in range(1, min(upper)+1)
                      for vector in product(*(range(h, min(h+1, u)+1) for u in upper))
                      if min(vector) == h)
        return min(((math.fsum(m[mi-1] for m, mi in zip(masses, v)), v)
                    for v in candidates), key=lambda pair: (-pair[0], pair[1]))

    best, best_mass = None, -np.inf
    for h in range(1, min(upper)+1):
        low = np.array([m[h-1] for m in masses])
        high = np.array([m[h] if len(m) > h else -np.inf for m in masses])
        use_high = high > low  # Prefer low on an exact tie.
        if use_high.all():
            # At least one coordinate must attain the minimum h.
            use_high[np.argmax(low-high)] = False
        log_mass = math.fsum(np.where(use_high, high, low))
        if not math.isfinite(log_mass):
            continue
        vector = tuple(int(h+v) for v in use_high)
        if log_mass > best_mass or (log_mass == best_mass and (best is None or vector < best)):
            best_mass, best = log_mass, vector
    return best_mass, best


def _enriched_map(masses, weight_u, weight_v):
    upper = tuple(map(len, masses))
    minimum = min(upper)
    count = (minimum-1)*((1 << len(upper))-1) + (1 << sum(u > minimum for u in upper))
    best_mass, best = _uniform_map(masses, upper, count)
    log_u = math.log(weight_u)-math.log(count)
    log_v = math.log(weight_v) if weight_v else -np.inf
    ones_score = float(np.logaddexp(log_u, log_v))+math.fsum(m[0] for m in masses)
    return best if best is not None and log_u+best_mass > ones_score else (1,)*len(masses)


@dataclass(frozen=True)
class BiologicalAssignment:
    """Biological labels, centers, and measured multiplicity calls."""

    mutation_ids: tuple
    region_ids: tuple
    component_labels: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    internal_to_biological: np.ndarray
    clonal_prior_beta: float | None
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
    """Collapse the fitted U and V responsibilities before biological MAP."""
    template, centers, weights, internal = fit.template, fit.centers, fit.weights, fit.responsibilities
    q, s = len(template), len(centers)
    if (template not in ((), ("U",), ("V",), ("U", "V"))
            or (model.structural_uv_equal and "U" in template)
            or q+s == 0 or centers.shape != (s, model.r) or weights.shape != (q+s,)
            or not np.isfinite(centers).all() or np.any((centers < 0.) | (centers > 1.))
            or np.any(np.all(centers == 1., axis=1)) or len(np.unique(centers, axis=0)) != s
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12)
            or not math.isfinite(fit.log_likelihood) or not math.isfinite(fit.bic)):
        raise ValueError("Selected fit is not an active canonical component model")
    if (internal.shape != (model.n, q+s) or not np.isfinite(internal).all() or np.any(internal < 0.)
            or not np.allclose(internal.sum(axis=1), 1., rtol=0., atol=1e-10)):
        raise ValueError("Selected fit has invalid responsibilities")

    order = np.argsort(-np.linalg.norm(centers, axis=1), kind="stable")
    free_labels = np.empty(s, dtype=int)
    free_labels[order] = np.arange(1, s+1)
    internal_map = np.r_[np.zeros(q, dtype=int), free_labels]
    component_labels = np.arange(0 if q else 1, s+1)
    biological_centers, biological_weights = centers[order], weights[q:][order]
    posterior = internal[:, q:][:, order]
    wc = float(math.fsum(weights[:q]))
    wu = float(weights[template.index("U")]) if "U" in template else 0.
    wv = float(weights[template.index("V")]) if "V" in template else 0.
    if q:
        posterior = np.column_stack((internal[:, :q].sum(axis=1), posterior))
        biological_centers = np.vstack((np.ones(model.r), biological_centers))
        biological_weights = np.r_[wc, biological_weights]
    maximum = np.argmax(posterior, axis=1)
    labels = component_labels[maximum]  # First maximum: smallest public label.
    ccf = biological_centers[maximum].copy()
    valid_calls = model.measured & (ccf > 0.)
    calls = np.where(valid_calls, 1., np.nan)
    for i in np.flatnonzero(labels == 0):
        if not ((wu and np.isfinite(model.log_u[i])) or (wv and np.isfinite(model.log_v[i]))):
            raise ValueError("Clonal MAP label has no finite clonal evidence")
        if wu:
            used = np.flatnonzero(model.observed[i])
            maximum = int(model.major[i, used].min())+1
            masses = tuple(binomial_log_mass(model.constants[i, j], model.alt[i, j], model.ref[i, j],
                           model.scaling[i, j]*np.arange(1, min(maximum, model.major[i, j])+1, dtype=float))
                           for j in used)
            for j, value in zip(used, _enriched_map(masses, wu, wv)):
                if valid_calls[i, j]:
                    calls[i, j] = value
    ccf[model.excluded] = np.nan
    return BiologicalAssignment(model.mutation_ids, model.region_ids, component_labels, labels,
        biological_centers, biological_weights, internal_map, None if not q or model.structural_uv_equal else wv/wc,
        ccf, calls, fit.log_likelihood, fit.bic, fit.degrees_of_freedom)


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
