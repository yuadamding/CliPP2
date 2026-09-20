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
The likelihood, multiplicity priors, graph weights, fusion penalty, and KKT
gate are unchanged by the clonal-existence constraint below.

The reported multiplicity is the highest-posterior candidate conditional on the
final fixed-partition CCF refit, with exact ties choosing the smaller integer—not
a rounded VAF-based estimate. Missing or zero-depth observations are marked
uninformative and their call is left missing, except when the only possible
multiplicity is structurally fixed at one.

## Occupied clonal cluster

CliPP2 requires **at least one retained mutation with CCF exactly 1 in every
region**. Its clonal membership and size are inferred, not supplied. This is
the only additional prior: the existing objective is minimized on the union
of its original feasible boxes with one eligible mutation fixed at the
all-one vector. There is no attraction-to-one penalty, minimum clonal size
beyond one, separation gap, special multiplicity rule, or restriction on the
other centers; a center such as `(1, 0.6, 0.8)` remains permitted.

Witness eligibility uses the original float64 bounds. If no retained mutation
can reach 1 in every region, fitting raises `ClonalConstraintInfeasibleError`;
it does not expand bounds, remove candidates, or substitute a near-one value.
Missing observations do not change this domain test, and satisfying the prior
with an uninformative mutation is not evidence for biological clonality.

The raw solver searches eligible temporary witnesses using the unchanged raw
objective, sharing the frozen graph and processing branches sequentially.
Only valid lower bounds may prune a branch. Worst-case work can approach one
raw fit per eligible witness per lambda; this is not a constant-cost change.
When a retained solution already satisfies another witness, a fresh float64
KKT audit may cover that box without another optimization, but only with a
convexity or singleton-likelihood supporting-tangent proof. Unsupported
mixtures and rejected audits still receive the ordinary solve. The
`witness_branches_reused` diagnostic records these audited branches separately;
sharing a CCF-one row alone never supplies a certificate.
Reuse keeps the existing numerical KKT tolerance; it is not a new zero-gap
proof or a guarantee of bitwise agreement with independently iterated fits.
An individual witness's KKT certificate is conditional on its fixed box, not
a global certificate over all witnesses or clusterings. Search diagnostics
distinguish attempted, pruned and unresolved branches.

The API/CLI summary reports clonal occupancy separately from conditional KKT
certification and witness-search completion. It also reports how many regions
have positive-depth observed counts for the retained witness; zero means the
existence requirement was satisfied without count evidence from that mutation.
Detailed branch coverage and elapsed time remain in the returned publication
record, not an additional output file. If a branch raises before returning its
work counters, `witness_search_work_complete` is false: counted work is then
partial, while the recorded witness-search elapsed time still covers the search.
By contrast, a branch that returns a nonadmissible certificate has completed
its recorded computation: it makes the search unresolved, not the work count
incomplete. This distinction does not relax raw-candidate admission.

Partition refits choose the feasible occupied block with the smallest increase
in count loss when fixed at the all-one center. Other centers retain their
ordinary fixed-label estimates, and CEM may change clonal membership. The
existing nominal `K * regions` complexity and Dirichlet allocation score are
unchanged: for fixed labels, only fitted count loss changes. Grid/local refits
and bounded model selection retain their existing limited optimality claims.
For a one-cluster partition the constraint fixes its entire center, so no free
scalar-center optimization is needed.
For multiple clusters, a recognized numerical failure in a free scalar fit
makes that block unavailable as a free center. Profiling can recover it only
by fixing that same block at one, with valid free fits for every other block.
The returned qualification record retains the failed coordinates and errors;
input, model-consistency, programming and resource errors still propagate.

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
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, `is_clonal`, and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | final CCF, CN, multiplicity MAP call, and posterior probabilities |
| `{tumor_id}_excluded_mutations.tsv` | triggering mutation–region–reason | original-CN exclusion audit; header-only when none are excluded |

For tumors with mixed CN, the mutation–region table additionally reports
`mean_total_cn` and `cn_state_count`. Mixed entries leave `major_cn` and
`minor_cn` blank because no single clonal pair describes them; their candidate
range remains explicit in `multiplicity_candidates`. Use the canonical input
for state-specific CN evaluation. Clonal-CN-only mutation–region tables retain
their schema.

`is_clonal` marks occupied centers exactly equal to one in every region.
Public labels remain ordered by decreasing L2 norm of final CCF, starting at
zero; no output-time CCF reassignment or rounding creates the clonal center.

## Regression tests

Regression and CUDA qualification suites are maintained outside this compact
repository. Install their dependencies with `pip install -e '.[test]'` and run
the separately maintained suite against the intended source revision.

Run CUDA qualification on an allocated LSF GPU with `CLIPP2_CUDA_TESTS=1`.
Explicit CUDA qualification fails if CUDA is unavailable; ordinary CPU runs
report CUDA skips, which are not GPU qualification.
