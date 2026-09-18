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
or major CN greater than `--max-major-cn` (**default: 4**); different clonal CN
states between regions are allowed. The limit is inclusive: major CN 4 is
retained by default, while 5 and above are excluded. Use `--max-major-cn 6` for
the previous cutoff. For each retained mutation–region pair, fitting marginalizes integer
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


Use `--device cpu` for explicit CPU execution. CUDA is the default; unavailable
CUDA or insufficient memory fails without silently switching devices. Set the
positive-integer CN eligibility cutoff with `--max-major-cn` (default 4).
`--verbose` and standard help and version options are also available.

## Outputs

A successful fit writes four TSV files into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | selected cluster and final fixed-partition CCF per region |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | final CCF, CN, multiplicity MAP call, and posterior probabilities |
| `{tumor_id}_excluded_mutations.tsv` | triggering mutation–region–reason | original-CN exclusion audit; header-only when none are excluded |

