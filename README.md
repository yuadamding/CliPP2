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
solver, resource, selection-score, and partition-tolerance controls.

## Outputs

A fit writes three tables into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | selected cluster and final fixed-partition CCF per region |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw-partition diameter, and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | final CCF, copy number, and multiplicity or occupancy-path summary |
