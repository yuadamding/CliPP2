# CliPP2

CliPP2 fits one joint mutation partition across tumor regions, estimates regional
cancer-cell fractions (CCFs), and reports mutation–region multiplicity.
The default is a **free-center frozen-tree estimator** implemented by
CliPP2's own multi-region [kernel](kernel/). CliPP2 and CliPP1.5 are independent
repositories; the latter is a development comparison reference, not a runtime
dependency. Algorithm ancestry is recorded in [NOTICE](NOTICE).
It does **not** require a clonal cluster or attract centers toward one.

This migration changes the estimator, not merely its implementation speed.
The tree restricts selectable clusters to connected components; it is a
mutation-similarity tree, **not a tumor phylogeny**. Local reference checks
are not a claim of cohort accuracy, GPU speedup, or global optimality.

## Install and fit

Use the `ml1` environment. Installation requires a C++17 compiler to build
CliPP2's source-bound native kernel:

```bash
conda activate ml1
pip install .
clipp2 fit --input-file examples/exampleTumor1.tsv --outdir example_results
```

CUDA is the default. Use `--device cpu` explicitly for CPU execution.
The CUDA implementation is honestly hybrid: float64 likelihood/gradient/
curvature work runs on CUDA; native tree/chain quadratic solves, continuation,
pooled initialization, conditional refits and weight optimization use the host. There is no
silent CPU fallback when CUDA is unavailable.

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-file` | Required | Canonical tumor TSV, optionally gzip |
| `--outdir` | `clipp2_results` | New output namespace |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--max-major-cn` | `4` | Whole-mutation eligibility cutoff |
| `--max-clusters` | `10` | Maximum tree-component capacity, 1–10 |
| `--verbose` | Disabled | Retained CLI compatibility option |

## Input, filtering and multiplicity

Follow [exampleTumor1.tsv](examples/exampleTumor1.tsv). The twelve model columns
are `mutation_id`, `sample_id`, `alt_count`, `ref_count`, `count_observed`,
`purity`, `normal_cn`, `segment_id`, `cn_state_id`, `cn_state_fraction`,
`allele_a_cn`, and `allele_b_cn`. Extra columns remain inert. Represent every
mutation–region pair, including missing counts. Multiple CN states repeat one
read observation, not independent observations. Purity is fixed within a region.

If **any original CN state in any region** has major CN above
`--max-major-cn`, that mutation is excluded from **all regions**. Subclonal
CN itself is retained. Mutations with no informative positive-depth observation
are separately excluded; a region with no informative observations is an error,
not silently removed. Exclusions are recorded in the returned/printed summary.

`load_tumor_txt` performs input validation and preprocessing only
(`initialize=False`; explicit `initialize=True` is rejected). Numerical
initialization belongs to the fitting layer, not the input loader.

For a single CN state, multiplicity is uniformly marginalized over
**1…major CN**. Raising the eligibility cutoff therefore permits support above
four. CCF geometry is on the CCF scale; inherited cellular prevalence is
converted by regional purity. Normal CN other than two uses an explicit
generalized-denominator adapter.

Mixed CN uses the separately versioned `mixed_cn_bulk_assignment_event_v2`
model: uniform
support 1…min(4, maximum major CN), the original fraction-weighted bulk
denominator, clipping and feasible box. This is not an evolutionary
mixed-CN model and is not claimed equivalent to the single-state inheritance.
Multiplicity is independent between mutation–region observations, marginalized
during fitting, and called only conditional on the final refitted CCF.

## Inference and interpretation

1. Regional pooled likelihood initialization creates CCF pilot vectors.
2. A single region uses a specialized frozen chain. Other inputs use a
   deterministic exact minimum spanning tree with streamed distance evaluation.
   Missing-overlap pairs have no finite edge; disconnected overlap is rejected.
3. Vector edge-jump continuation proposes at-most-K connected components.
   Forest QPs and actual-likelihood line searches use one shared regional cut set.
4. Every block–region center is refitted freely. Unsupported centers make a
   candidate ineligible. Connected coarsenings and adjacent-pair boundary repairs
   preserve earlier eligible candidates. Supplemental support-aware proposals
   include zero-jump edges: at most 64 mutations, every observation-supported
   one-edge split is considered when two clusters are requested or repair an
   unsupported higher-capacity seed. Larger trees consider at most 32
   likelihood-ranked splits. The coarse regional profile-grid gain incorporates
   counts and depth, retains missing-region CN bounds, and breaks numerical
   proxy ties by ID-free observation-multiset signatures before edge order.
   Up to 16 cut exchanges from eight unsupported native seeds provide additional
   support repairs. All proposals use the unchanged exact refit and score;
   the four strongest supplemental fits share a budget of 16 boundary-refinement
   proposal evaluations, including rejected proposals. Native refinement is
   unchanged. These are work-count limits, not elapsed-time deadlines.
5. Fitted mixture weights score the **product of observed regional likelihoods
   inside each admissible cluster mixture**, with penalty
   `[q*R + (q-1)] * log(N)`, where N counts retained mutation vectors.
   The winner is the minimum over the entire eligible scored candidate bank.

The centers are conditional partition refits, **not** a joint mixture-center
MLE. Finite continuation, residual convergence, numerical scalar search and
global optimality are distinct claims. The independent publication audit checks
bounds, connected memberships, joint reads/validity likelihood, fitted-weight optimality,
score and complete-bank reconciliation.

### Mixed-CN assignment consistency

For mutation $i$ and center $k$, let $A_{ik}=1$ exactly when the complete CCF
vector lies inside that mutation's original closed regional bounds, and zero
otherwise. This includes CN bounds in regions with missing counts; those regions
contribute no read likelihood. Let $f_{ik}$ be the product of the observed
regional multiplicity-marginalized read likelihoods. Selection uses

$$
\ell_{\mathrm{joint}}(w) = \sum_i \log\!\left(\sum_k w_k A_{ik}f_{ik}\right),
\qquad
\mathrm{score} = -2\ell_{\mathrm{joint}} + [qR+q-1]\log N.
$$

This is a **joint reads-plus-assignment-validity criterion**, not read likelihood
conditioned on validity. With $Z_i=\sum_k w_k A_{ik}$,
$\ell_{\mathrm{joint}}=\ell_{\mathrm{conditional}}+\sum_i\log Z_i$:
the admissibility factor is deliberately retained, with no division by $Z_i$.
Forbidden assignments therefore have zero responsibility in weight fitting,
scoring and boundary refinement. The independent publication audit reconstructs
the same rule from the original input. When all assignments are admissible,
the numerical likelihood and weight optimization are unchanged.

This limited consistency repair does **not** change mixed-CN emissions, support,
clipping, original bounds or the conditional center refit. In particular, the
original mixed-CN box can still exclude biologically possible CCF-one histories.
It adds neither a weight floor nor a clonal, occupancy or evolutionary prior.
The penalty is BIC-style; this joint criterion is not claimed to be a calibrated
biological posterior or a newly validated model-selection rule.

The API/CLI verification diagnostics distinguish four different quantities:

| Field | Meaning |
| --- | --- |
| `hard_partition_log_likelihood` | Read log likelihood at the hard assignments; `conditional_log_likelihood` remains its backward-compatible alias |
| `mixture_log_likelihood` | Joint reads-plus-validity mixture log likelihood used in the score |
| `assignment_validity_log_probability` | $\sum_i\log Z_i$ |
| `validity_conditioned_mixture_log_likelihood` | Joint mixture log likelihood minus the validity term |

The last quantity is evaluated at the **same joint-fitted weights**, not
separately optimized and not used for selection. It is the
$\ell_{\mathrm{conditional}}$ in the decomposition above, not the historical
hard-partition `conditional_log_likelihood` field. These diagnostic additions
do not change the three public TSVs.

Exact single-region compatibility requires matched coordinate tie keys. The
canonical format has no genomic-coordinate fields, so its documented default
uses mutation-ID text to break equal-pilot ties. The internal
`fit_scalar(..., coordinate_keys=(chromosome_text, position_text))` adapter
retains CliPP1.5's exact coordinate-string order for matched comparisons.
No coordinates are guessed from mutation IDs. This compatibility is tested
against the separate reference repository; production loads only CliPP2 code.

The kernel's primary state is an **N mutations × R regions** CCF matrix.
An edge jump is an R-vector, and its L2 norm determines one shared cut set;
regional fits are never clustered independently and reconciled afterward.
Native forest QPs accept all R columns with region-specific bounds and one
shared graph. Missingness, pooled initialization, conditional refits, and the
product-inside-mixture score are explicit parts of this joint-region model.
The single-region native chain is a specialization, not the multi-region model.

Tree storage is linear in mutations × regions, but exact tree construction
still takes O(N²R) distance work once. Host active-set iterations and candidate
refits remain potential bottlenecks. Offline `topology_diagnostics`,
`offline_truth_replay` and tiny exhaustive references quantify tree
fragmentation and search loss; truth never enters production fitting.
The added one-edge coverage does not exhaust higher-order partitions or remove
the topology restriction. In particular, collectively supported mutations can
remain disconnected within the frozen tree when their measurements do not
overlap. Selected-fit verification reconciles the scored bank; it does not
prove search completeness or accurate clone recovery.

Large-tree proposal ranking uses a fixed 34-point regional grid, streamed in
column blocks. The original mixed lower endpoint is included even for narrow
boxes. This is an approximate profile improvement, **not** a bound on the final
mixture score: modes between grid points and omitted edges can still matter.
Scientific-content tie breaking avoids using mutation IDs to choose among
different observations on a fixed tree; it does not make tree construction or
genuinely indistinguishable ties invariant to all renamings. The exchange route
remains support-ranked. Diagnostics report the supported-edge population, exact
evaluated one-edge fraction, ranking method and refinement work counts; timings
include supplemental search separately. Inclusive phase times overlap refit
time and must not be added as independent costs.

## Outputs

Success publishes exactly **three TSVs**, with no JSON manifest or diagnostic
TSV added to the output namespace:

| File | One row per | Columns |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | Retained mutation | `tumor_id`, `mutation_id`, `cluster_label`, `phi_{region}` |
| `{tumor_id}_cluster_centers.tsv` | Selected cluster | `tumor_id`, `cluster_label`, `cluster_size`, `phi_{region}` |
| `{tumor_id}_mutation_region_multiplicity.tsv` | Retained mutation × region | `tumor_id`, `mutation_id`, `region_id`, `phi`, `major_cn`, `minor_cn`, `multiplicity_call` |

Cluster **0 has the largest final CCF-vector L2 norm**; remaining labels follow
decreasing norm, with stable ties. Zero does not imply clonal. Reported CCFs are
the selected final refits, not raw continuation iterates.

Multiplicity is the highest-posterior integer (smaller integer on exact ties).
Missing/zero-depth observations have `NA` calls unless support is structurally
one. If mixed CN occurs, the long table additionally contains `mean_total_cn`;
mixed-state rows have `NA` major/minor CN. Join labels by tumor and mutation ID.

Publication is no-clobber and independently verified, but the three links are
not a collective filesystem transaction. Preserve partial files after a
publication failure and retry only into a fresh namespace. Operational
source/input/config/timing evidence is returned by the API and printed by the
CLI; wrappers should retain it outside the three-table public output.

## Development

The current external suite is `../tests/CliPP2_tree`; older clonal
complete-graph tests target the historical estimator. CliPP1.5 is a read-only
external comparison reference, never imported by the installed package.
Kernel source/build identity is checked before native execution.
The simulation generator remains in [simulation/](simulation/).
The superseded inference implementation and `_clipp15` compatibility package
have been removed. Offline tree diagnostics remain available.
