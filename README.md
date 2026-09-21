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

### CN eligibility

CliPP2 excludes a mutation from **all regions** if **any original CN state in
any region** has major CN greater than `--max-major-cn` (**default: 4**).
The cutoff is inclusive: major CN 4 passes the default filter, whereas 5 does
not. All states are checked, including low-fraction states and regions with
missing read counts; average CN does not determine eligibility.

**Subclonal CN alone is not an exclusion criterion.** Different CN states
between regions are also allowed. If the filter removes every mutation,
loading raises `NoEligibleSNVsError` and no result tables are written.

### Multiplicity and the count model

For each retained mutation–region pair, the likelihood marginalizes integer
multiplicity candidates from **1 to min(4, maximum major CN)** with uniform
priors. For mixed CN, the maximum is taken across that region's CN states.
Increasing `--max-major-cn` changes eligibility only; multiplicity candidates
remain capped at four.

Copy number and purity determine the fixed scaling

```text
scaling = purity / ((1 - purity) * normal_cn + purity * mean_total_cn)
```

Here, `mean_total_cn` is the CN-fraction-weighted total tumor copy number.
Each candidate's binomial probability is the clipped value of
`scaling * multiplicity * CCF`, with probability-safe CCF bounds.

For mixed CN, this is a **bulk-CN approximation**, not an explicit model of
which CN populations carry the mutation. It does not infer CN-population
occupancy, mutation timing, or a lineage history. The resulting CCF upper bound
can be below one even when a mutation passes the CN filter.

## Occupied clonal cluster

CliPP2 requires **at least one retained mutation with CCF exactly 1 in every
region**. Clonal membership and size are inferred. There is no additional
attraction-to-one penalty, minimum clonal size beyond one, or separation
constraint on the other centers.

Eligibility is evaluated using the original float64 bounds. If no retained
mutation can reach one in every region, fitting raises
`ClonalConstraintInfeasibleError`; it does not expand bounds or substitute a
near-one value. A mutation without informative counts can satisfy this domain
constraint, so clonal occupancy alone is not count-based evidence of clonality.

Raw fitting searches eligible witness boxes on the frozen graph. A witness-box
KKT certificate is conditional on that box, not a global guarantee over all
witnesses or partitions. The run summary distinguishes numerical certification
from witness-search completion.

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
version. Solver tolerances, selection score, and partition policy are fixed
internally rather than exposed as CLI controls.

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
