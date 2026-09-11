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
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

CliPP2 excludes a mutation from **all regions** if any region has subclonal
copy number (more than one distinct CN state after identical states are combined)
or major CN greater than six; different clonal CN states between regions are
allowed. For each retained mutation–region pair, fitting marginalizes integer
multiplicity candidates from **1 to major CN** with uniform priors under a
binomial likelihood adjusted for purity, normal/tumor copy number, and CCF.
The reported multiplicity is the highest-posterior candidate conditional on the
final fixed-partition CCF refit, with exact ties choosing the smaller integer—not
a rounded VAF-based estimate. Missing or zero-depth observations are marked
uninformative and their call is left missing, except when the only possible
multiplicity is structurally fixed at one.

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
