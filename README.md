# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers integer mutant-copy multiplicity from single- or multi-region tumor
sequencing data using observed-data pairwise fusion. Tumor purity and
copy-number states and fractions are supplied inputs, not estimated by CliPP2.

## Install

From the repository root:

```bash
pip install .
```

## Input

Use one tab-delimited file per tumor, containing all regions. Follow
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv) for the input format.
Every mutation–sample pair must be represented, including pairs with missing
read counts. For a segment with multiple CN states, include every state and
keep the repeated observation fields consistent.

## Fit

CUDA is the default:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

Request CPU execution explicitly:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_cpu_results \
  --device cpu
```

The fitting options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--input-file` | Required | Tumor input TSV |
| `--outdir` | `clipp2_results` | Output directory |
| `--device` | `cuda` | Execution device: `cuda` or `cpu` |
| `--max-major-cn` | `4` | Positive-integer CN eligibility cutoff |
| `--verbose` | Disabled | Verbose execution output |

Use `clipp2 fit --help` for command help and `clipp2 --version` for the package
version. 

### CUDA execution

Execution is hybrid: CUDA handles the fusion solver and other tensor-based
stages, while scalar refits and search decisions remain on the CPU. Working
iterations normally use float32, with float64 node-level box-QP arithmetic and
frozen-source float64 terminal auditing.

Unavailable CUDA or insufficient memory does not silently switch the fit to
CPU. A supported compilation failure falls back to eager CUDA kernels, not
CPU execution. Complete-graph edge storage grows quadratically with retained
mutation count and linearly with region count; chunking bounds temporary work,
not the total edge state.

## Outputs

A successful fit writes exactly **three TSV files** into `--outdir`. Filenames
use the input file stem as the tumor ID unless overridden by `##tumor_id`
metadata.

| File | One row per | Columns |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | Retained mutation | `tumor_id`, `mutation_id`, `cluster_label`, `phi_{region}` for each region |
| `{tumor_id}_cluster_centers.tsv` | Selected cluster | `tumor_id`, `cluster_label`, `cluster_size`, `phi_{region}` for each region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | Retained mutation × region | `tumor_id`, `mutation_id`, `region_id`, `phi`, `major_cn`, `minor_cn`, `multiplicity_call` |

### Interpretation

All reported CCFs come from the **selected fixed-partition refit**, not directly
from the raw penalized fusion iterate. Raw-fit certification and refit
qualification are recorded separately in the run summary.

The designated constrained clonal block is always **cluster 0**. Remaining
clusters are ordered by decreasing L2 norm of their final CCF profiles, with
exact ties preserving internal-label order. Another all-one center remains a
separate block; output relabeling does not merge clusters or change CCFs.

`multiplicity_call` is the highest-posterior candidate conditional on the final
refitted CCF, with exact ties choosing the smaller integer. Missing or
zero-depth observations have a blank call when multiple candidates are
possible; a structurally fixed single-copy candidate remains one. Posterior
probabilities are not included in these compact tables.

When mixed-CN observations are present, the mutation–region table additionally
contains `mean_total_cn` for all rows. Mixed-CN rows have blank `major_cn` and
`minor_cn` because no single CN pair describes them. Retain the original input
for state-specific CN information. Cluster membership can be joined from the
mutation table using `tumor_id` and `mutation_id`.
