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
one numerically certified raw fusion block constrained to common CCF 1 in every
region. This changes the raw feasible set and defines a clonal-anchored
estimator distinct from the unanchored fixed objective.

The constraint is existential: at least one feasible mutation row must equal
the all-ones CCF target. CliPP2 decomposes this union into witness-conditioned
subproblems, freezes only the current witness row, and minimizes over the
certified subproblems. No mutation is forced clonal from VAF, read depth,
single-copy CCF inversion, inferred multiplicity, or another count-derived
heuristic. Such quantities cannot establish CCF one under multiplicity
ambiguity or sampling noise.

A named mutation is only computational provenance. The biological model
object is the complete set of solved raw CCF rows within the strict clonal
equality tolerance of the all-ones target. Every member and the block centroid
must pass that tolerance. The effective equality tolerance is the maximum of
its configured floor, machine precision, raw primal tolerance, and certificate
column tolerance, and it cannot exceed the selection-partition tolerance. The
graph and weights are frozen before the
clonal-constraint search. The default `adaptive-bound-complete` search starts
from eight witnesses ranked by observed-data anchor deviance and evaluates
every additional witness whose valid likelihood lower bound can still beat the
incumbent. An unresolved or uncertified competitive witness makes the lambda
candidate ineligible. All distinct raw-objective-tied minimizers reach
fixed-partition scoring.

The exact CCF-one block is protected from the looser general partition
tolerance: near-one mutations cannot be absorbed into it merely because they
are within `--selection-partition-tol`. The fixed-partition refit preserves
that exact block at CCF 1 and never reselects or expands it. Mutations whose
CCF support does not strictly include one are ineligible. The unanchored
`fixed_partition_bic` sensitivity mode uses `K × S` degrees of freedom.
With missing mutation-region counts, only observed non-anchor cluster-region
centers contribute to the identifiable BIC dimension.

Raw clonal-witness modes are explicit:

- `specified-witness` solves one user-named retained mutation and is exact for
  that specified biological model.
- `enumerated-witness` solves every feasible retained mutation.
- `adaptive-bound-complete` is the default existential-cluster estimator. It
  uses valid likelihood lower bounds to certify witness coverage without
  claiming global optimality of each nonconvex branch.
- `screened-witness` evaluates only the requested initial screen. It becomes
  selection-eligible only if lower bounds prove every omitted witness cannot
  beat the incumbent; otherwise it fails closed.
- `none` is required by the unanchored sensitivity score.

The output makes the raw clonal cluster primary and records its size, centroid,
target, maximum member residual, and stable block signature. A separate
biological-evidence object reports observed support, total depth, median depth,
and its QC status; evidence does not alter the raw feasible set, mathematical
certificate, candidate eligibility, or BIC. Singleton blocks are valid for the
exact existential model. The witness mutation remains computational
provenance. Base-fusion, existential-union-model, and witness-subproblem hashes
are separate; search mode and screening source are provenance rather than
objective ingredients. Warm states cannot cross witness subproblems.

The exactness fields are also separate: witness coverage records that every
competitive witness was solved or lower-bound-pruned; branch stationarity
records full raw KKT certification; union global optimality is normally false
for the generic nonconvex observed-data objective. The ordinary conditional
BIC remains the selection criterion. `anchor_prior_adjusted_selection_score`
reports the prespecified uniform-block sensitivity `BIC + 2 log(K)` but does
not steer selection.

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
