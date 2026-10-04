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

Mixed CN remains a separately versioned `mixed_cn_bulk_v1` model: uniform
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
   preserve earlier eligible candidates.
5. Fitted mixture weights score the **product of regional likelihoods inside
   each cluster mixture**, with penalty
   `[q*R + (q-1)] * log(N)`, where N counts retained mutation vectors.
   The winner is the minimum over the entire eligible scored candidate bank.

The centers are conditional partition refits, **not** a joint mixture-center
MLE. Finite continuation, residual convergence, numerical scalar search and
global optimality are distinct claims. The independent publication audit checks
bounds, connected memberships, joint likelihood, fitted-weight optimality,
score and complete-bank reconciliation.

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
