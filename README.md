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

CliPP2 excludes a mutation from **all regions** if **any CN state in any region**
has major CN greater than `--max-major-cn` (**default: 4**). Every state is
checked, including low-fraction subclonal states and regions with missing read
counts; neither average CN nor the dominant state determines eligibility.
The limit is inclusive: major CN 4 passes, while 5 and above exclude the whole
mutation. Use `--max-major-cn 6` for the previous cutoff. Subclonal CN alone
is no longer a filtering reason.

For each retained mutation–region pair, preprocessing compiles integer
multiplicity candidates from **1 to min(4, major CN)** with uniform priors.
For mixed CN, major CN here is the **maximum across that region's states**
(the union of their integer candidate ranges). Raising `--max-major-cn`
changes eligibility only; multiplicity candidates remain capped at four.

CN is compiled once into candidate support and the fixed read-count scaling
`purity / ((1 - purity) * normal_cn + purity * mean_total_cn)`, where
`mean_total_cn` is the CN-fraction-weighted total tumor CN. Fitting then uses
the existing clipped linear binomial likelihood and probability-safe CCF
bounds, with no CN-population occupancy, timing, or lineage constraints.
This is a **bulk-CN approximation** for subclonal mixtures, not a model of
which CN populations carry the mutation. Its existing probability-safe bound
can be below CCF one when the largest candidate exceeds bulk copy availability.
The fusion solver, graph, KKT gate, Ward/CEM, and partition selection are unchanged.

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

For tumors with mixed CN, the mutation–region table additionally reports
`mean_total_cn` and `cn_state_count`. Mixed entries leave `major_cn` and
`minor_cn` blank because no single clonal pair describes them; their candidate
range remains explicit in `multiplicity_candidates`. Use the canonical input
for state-specific CN evaluation. Clonal-only tables retain their schema.
