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
one exact raw fusion block constrained to common CCF 1 in every region.
By default, every mutation whose single-copy, mutation-specific unpenalized
CCF exceeds one in every observed region is a mandatory member of this block.
This conservative rule includes any positive true-multiplicity CCF overflow
without using unknown truth multiplicity. A named mutation is only a
computational witness; the biological model object is the complete exact
CCF-one block. Every member and the block centroid must pass the strict clonal
equality tolerance.

The graph and weights are frozen before clonal-constraint construction. When
the unpenalized-overflow set is nonempty, it defines the mandatory CCF-one
block and one stable member suffices as computational witness. If that set is
empty, the default adaptive-exact search starts from eight lower-bound-ranked
witnesses and evaluates every additional witness that cannot be safely
pruned. An unresolved or uncertified witness that could beat the incumbent
makes the candidate ineligible. All distinct raw-objective-tied minimizers
reach fixed-partition scoring.

The exact CCF-one block is protected from the looser general partition
tolerance: near-one mutations cannot be absorbed into it merely because they
are within `--selection-partition-tol`. The fixed-partition refit preserves
that exact block at CCF 1 and never reselects or expands it. Mutations whose
CCF support does not strictly include one are ineligible. The unanchored
`fixed_partition_bic` sensitivity mode uses `K × S` degrees of freedom.
With missing mutation-region counts, only observed non-anchor cluster-region
centers contribute to the identifiable BIC dimension.

Raw clonal-witness modes are explicit. With the default
`--raw-clonal-include-unpenalized-overflow`, the modes control witness
provenance or fallback search without allowing an overflow mutation to leave
the clonal block:

- `specified-witness` solves one user-named retained mutation and is exact for
  that specified biological model.
- `enumerated-witness` solves every feasible retained mutation.
- `adaptive-exact` is the default exact existential-cluster estimator and uses
  valid likelihood lower bounds to prune witnesses that cannot win.
- `screened-witness` evaluates only the requested initial screen. It becomes
  selection-eligible only if lower bounds prove every omitted witness cannot
  beat the incumbent; otherwise it fails closed.
- `none` is required by the unanchored sensitivity score.

The output makes the raw clonal cluster primary and records its size, centroid,
target, maximum member residual, observed support per region, and stable block
signature. The witness mutation remains computational provenance. Base-fusion,
existential-union-model, and witness-subproblem hashes are separate; search
mode and screening source are provenance rather than objective ingredients.
Warm states cannot cross witness subproblems.

Singleton clonal blocks remain valid when no unpenalized-overflow mutation is
available (`--raw-clonal-cluster-min-size=1`), but the block must contain at
least one observed positive-depth mutation in every region. Both requirements
are explicit CLI controls and are reported in the block certificate. The
overflow inclusion rule has a Boolean sensitivity switch, but is enabled in
the production default.

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
| `{tumor_id}_mutation_clusters.tsv` | mutation | `selected_cluster_label`, `raw_clonal_witness_mutation`, `raw_clonal_constraint_frozen_member`, `is_raw_clonal_cluster_member`, `raw_phi_<region>`, `fixed_partition_refit_phi_<region>`, and their delta |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw mean/min/max and diameter, exact clonal-block centroid/target/residual/support, fixed-partition refit center, and partition signature |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | raw and refit CCFs plus separately prefixed multiplicity or occupancy-path summaries at both profiles |

`raw_phi_*` is the primary pairwise-fusion estimator. It is the returned
solution of the λ-penalized objective. In the default clonal mode, every member
of the certified raw clonal block is at CCF 1; its named witness mutation is
only the branch used to realize the existential constraint.

`fixed_partition_refit_phi_*` is a secondary unpenalized center refit using
exactly the selected raw fusion partition. It debiases center summaries but
never changes membership or replaces the raw estimator.

The production selection tolerance deterministically derives non-clonal blocks
from the final raw CCFs on the frozen graph. The separately certified exact
CCF-one block is inserted as a protected block. Certification requires
within-block compactness and no unprotected cross-block graph edge at or below
the general tolerance.
`--reporting-partition-tol` is recorded for reporting compatibility but never
changes selected K, labels, membership, or BIC.
