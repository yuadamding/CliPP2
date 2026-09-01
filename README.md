# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See [`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for profile,
solver, resource, selection-score, partition-tolerance, unsupported-input, and
failure-policy controls. The default `--failure-policy best-effort` saves the
highest valid typed outcome without relaxing any certificate gate.

## Outputs

Saved analyses use the tumor id as a prefix (the input file stem unless a
`##tumor_id` metadata line overrides it). Every saved primary, conditional, or
diagnostic outcome writes these status-rich files:

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_analysis_status.json` | analysis | tier, certification, selection, mask, and failure provenance |
| `{tumor_id}_region_status.tsv` | input region | estimate tier, data-mask counts, identification, and diagnostics |
| `{tumor_id}_raw_attempts.tsv` | raw solver start | objective, KKT components, budgets, dtypes, and hashes |
| `{tumor_id}_cluster_region_estimates.tsv` | selected cluster × region | best-available CCF and argmin-identification fields |
| `{tumor_id}_mutation_region_estimates.tsv` | mutation × region | retained counts, support/inclusion masks, CCF tier, and eligible posterior summaries |

Only a primary result (`primary_estimator_available=true`) writes the historical
compatibility files `mutation_clusters.tsv`, `cluster_centers.tsv`, and
`mutation_region_multiplicity.tsv`. A conditional fallback instead writes
`secondary_cluster_centers.tsv` and
`secondary_mutation_region_estimates.tsv`; a diagnostic-only result writes no
point-estimate compatibility file.

Every available point estimate also reports `ccf_ordered_cluster_label`, a
zero-based presentation label: cluster 0 has the smallest root-mean-square
distance from CCF 1 across its statistically identified regions, and the
remaining clusters follow in increasing distance. Exact ties use the immutable
`cluster_label`; clusters with no identified region sort last. This derived
ordering does not change `cluster_label`, selection, scores, or CCFs, and
cluster 0 is not a clonal-identifiability claim. Cluster-level tables also
report `ccf_distance_to_one` and the identified-region count used for it.
