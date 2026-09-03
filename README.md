# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs with
observed-data pairwise fusion, and reports mutant-copy/path posteriors for a
single- or multi-region tumor.

All supported copy-number inputs compile to one immutable piecewise-affine
emission model. The production selection policy is fixed:
`hybrid-ward-cem-bic-v1`. Certified raw-fusion partitions and deterministic
Ward/CEM partitions receive the same immutable-label refit and are compared by
fixed-partition BIC. There is no runtime score or policy switch.

## Install

```bash
pip install .
```

Pandas is not required for fitting or output. Development dependencies are
available with `pip install '.[test]'`; simulation dependencies are available
with `pip install '.[simulation]'`.

## Input

The public input is one tab-delimited long table per tumor. The 12
model-defining columns and a small runnable input are in
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv); see
[`examples/README.md`](examples/README.md) for the column contract.

## Fit

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

The `balanced` profile uses CUDA when available. Use `--device cpu` only for
small smoke tests. The normal command-line surface is intentionally limited to:

```text
--input-file  --outdir  --profile  --device  --failure-policy
--checkpoint  --resume  --config
```

Expert numerical controls belong in a versioned JSON file:

```json
{
  "schema_version": 1,
  "fit": {
    "computation_profile": "balanced",
    "device": "cuda",
    "dtype": "float32",
    "max_tumor_edge_pass_equivalents": 200000
  },
  "run": {
    "failure_policy": "best-effort",
    "unsupported_policy": "error"
  }
}
```

Run it with `--config fit.json`. Explicit `--profile`, `--device`, and
`--failure-policy` arguments override those JSON values. Unknown sections,
fields, duplicate keys, and non-finite numbers fail closed.

The default `best-effort` failure policy saves the highest valid typed outcome
without weakening the raw KKT gate. Complete-graph work remains bounded by the
resolved profile or expert configuration. A resource stop is unresolved, not
convergence.

## Checkpoint and resume

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --checkpoint exampleTumor1.checkpoint

clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --resume exampleTumor1.checkpoint
```

Checkpoints use an explicit search schema, content-addressed NumPy arrays,
file locking, generation compare-and-swap, and atomic manifest replacement.
Resume verifies input, objective, graph, configuration, numerical environment,
software, and source-tree identity. Pickle is never used.

## Outputs

Every primary, conditional, or diagnostic analysis writes exactly four files:

| File | One row per | Purpose |
| --- | --- | --- |
| `{tumor_id}_analysis.json` | analysis | tier, identities, selection, work, masks, and failure provenance |
| `{tumor_id}_clusters.tsv` | selected cluster x region | CCF estimates, intervals, ordering, and support |
| `{tumor_id}_mutations.tsv` | mutation x region | observations, CN, CCF tier, intervals, and posterior summaries |
| `{tumor_id}_attempts.tsv` | raw solver attempt | objectives, KKT diagnostics, limits, work, dtypes, and identities |

`clusters.tsv` is header-only when no partition point claim exists;
`mutations.tsv` always retains every input mutation-region coordinate.

`cluster_label` is the immutable selected partition. The derived
`ccf_ordered_cluster_label` is presentation-only: cluster 0 is closest to CCF
1 across statistically identified regions, and the rest follow by increasing
distance. This ordering never changes selection, scores, labels, or refitted
CCFs and is not a clonal-identifiability claim.

## Scientific boundary

CliPP2 never forces a clonal anchor or CCF-one mutation. The graph, weights,
row-group fusion norm, observed-data likelihood, box, and lambda define the raw
estimator. Fixed-partition refits are secondary, and positive-lambda raw
admission always requires the full-original-graph float64 terminal KKT audit at
the unchanged `5 * tol` threshold.

See [`CHANGELOG.md`](CHANGELOG.md) for the v0.4 breaking surface.
