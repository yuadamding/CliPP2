# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv) and
[`examples/README.md`](examples/README.md).

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for solver,
resource, selection-score, anchor, and partition-tolerance controls.

## Outputs

A fit writes three tables into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | `selected_cluster_label`, `raw_clonal_witness_mutation`, `raw_clonal_constraint_frozen_member`, `is_raw_clonal_cluster_member`, `raw_phi_<region>`, `fixed_partition_refit_phi_<region>`, and their delta |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw mean/min/max and diameter, exact clonal-block centroid/target/residual/support, fixed-partition refit center, and partition signature |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | raw and refit CCFs plus separately prefixed multiplicity or occupancy-path summaries at both profiles |

