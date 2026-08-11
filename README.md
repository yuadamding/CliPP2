# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Production model-selection contract

For every positive fusion penalty λ, CliPP2:

1. solves the fixed raw pairwise-fusion objective, including the configured
   clonal constraint;
2. extracts and diameter-verifies one fusion partition from that raw solution;
3. refits unpenalized cluster centers without changing any label; and
4. scores that immutable partition with fixed-partition BIC.

Only objective-faithful, full-KKT-certified raw fusion candidates with a
certified partition can be selected. The default score is
`clonal_fixed_partition_bic`, with `(K - 1) × S` nominal degrees of freedom and
one raw-fusion mutation fixed at CCF 1 in every region. The graph and weights
are frozen before anchor search. At each λ, CliPP2 solves one seed-conditioned
raw objective per evaluated anchor mutation and chooses the certified fit with
the smallest exact penalized objective. The fixed-partition refit preserves
that raw-selected block at the same CCF-1 target; it never reselects a clonal
block. Mutations whose CCF support does not include one are ineligible. The
unanchored
`fixed_partition_bic` sensitivity mode uses `K × S` degrees of freedom.
With missing mutation-region counts, only observed non-anchor cluster-region
centers contribute to the identifiable BIC dimension.

Raw anchor modes are explicit:

- `specified-seed` solves one user-named retained mutation and is exact for
  that specified biological model.
- `enumerated-seed` solves every feasible retained mutation and is the exact
  unknown-anchor estimator; it can be expensive.
- `screened-seed` ranks mutations by zero-penalty anchor deviance and evaluates
  a bounded set (eight by default). Its output always records the eligible and
  evaluated counts and sets `raw_anchor_search_complete=false` unless the bound
  happened to include every feasible mutation.
- `none` is required by the unanchored sensitivity score.

Anchor seed, target, constraint residual, search completeness, candidate count,
and objective gap to the second seed are recorded in the search/run provenance.
Changing the seed changes the objective hash, and warm states cannot cross seed
objectives.

In the default adaptive-graph workflow, a deterministic Ward/CEM guide is an
objective-defining preprocessing result: it defines the adaptive graph weights,
supplies the initial raw-fusion state, and sets the initial positive-lambda
scale. The resulting complete graph is frozen and hashed before the lambda
path. Ward/CEM partitions are never selectable candidates; every selectable
result is still a raw fusion solution on that one frozen graph. With a
user-supplied graph, Ward/CEM supplies initialization and the initial lambda
scale but does not define the graph.

The fixed-partition refit uses its own immutable numerical specification
(`--selection-refit-tol` and `--selection-refit-max-iter`), independent of raw
solver retries or recovery. CliPP2 repeats the refit on a nested denser grid and
requires the two results to agree before the partition is score-eligible. This
is a numerical-resolution check, not a claim of a globally certified mixture
MLE. Output writing is serialization only and cannot change the selected
estimator, partition, refit, or score.

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
| `{tumor_id}_mutation_clusters.tsv` | mutation | `selected_cluster_label`, `raw_clonal_anchor_seed`, anchor target/residual, `raw_phi_<region>`, `fixed_partition_refit_phi_<region>`, and their delta |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw mean/min/max and diameter, fixed-partition refit center, clonal-anchor diagnostics, and partition signature |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | raw and refit CCFs plus separately prefixed multiplicity or occupancy-path summaries at both profiles |

`raw_phi_*` is the primary pairwise-fusion estimator. It is the returned
solution of the λ-penalized objective. In the default clonal mode its named
anchor mutation is already pinned at the feasible clonal CCF in this raw
solution.

`fixed_partition_refit_phi_*` is a secondary unpenalized center refit using
exactly the selected raw fusion partition. It debiases center summaries but
never changes membership or replaces the raw estimator.

The production selection tolerance deterministically derives the selected
partition from the final raw CCFs on the frozen graph. Certification requires
both within-block compactness and no cross-block graph edge at or below that
tolerance.
`--reporting-partition-tol` is recorded for reporting compatibility but never
changes selected K, labels, membership, or BIC.
