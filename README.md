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
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv) and
the external [input-format contract](../docs/input-format.md).

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

The default `balanced` profile runs the raw CUDA solver in `float32`, retains
the complete fusion graph, and uses a bounded lambda path and fixed-partition
refits for practical single-tumor latency. Its output records whether the
bounded search resolved the requested optimum; unresolved runs are explicitly
marked `provisional_unresolved` rather than presented as resolved selections.

Model selection uses the unanchored fixed-partition Dirichlet score. The immutable
raw-fusion partition is refitted without changing labels, its fixed-partition
BIC is calculated, and 0.7 times the deviance of that exact allocation under a
symmetric Dirichlet(1) prior is added. All `K` blocks are
exchangeable, and all `K x S` cluster-center coordinates are estimated freely:
CliPP2 no longer pins or distinguishes a CCF-one cluster. This is an
exact-partition prior score, not the posterior-entropy ICL used for soft mixture
assignments. For fixed `n` and `K`, its allocation term gives more prior mass to
imbalanced than balanced exact partitions.

Use `--profile strict` for stronger per-candidate KKT checks and
interval-certified fixed-partition refits with a `float64` raw solver, or
`--profile fast` for a smaller approximate search budget. Strict still uses a
bounded lambda path; it is not a proof of global path optimality. Explicit
`--dtype` and solver flags override the selected profile. Historical clonal
anchor flags are rejected instead of being silently reinterpreted.

Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for profile,
solver, resource, selection-score, and partition-tolerance controls.

For the full statistical and operational contract, read the
[maintainer guide](../docs/maintainer-guide.md). Benchmark definitions—including
the required CNA-only multiplicity macro-F1—are in the
[evaluation protocol](../docs/evaluation.md). The dated
[correctness audit](../docs/CliPP2_model_selection_correctness_audit_2026-08-12.md)
preserves historical evidence and superseded designs.

## Outputs

A fit writes five tables into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | `selected_cluster_label`, `raw_phi_<region>`, `fixed_partition_refit_phi_<region>`, and their delta |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw mean/min/max and diameter, fixed-partition refit center, and partition signature |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | raw and refit CCFs plus separately prefixed multiplicity or occupancy-path summaries at both profiles |
| `{tumor_id}_run_summary.tsv` | run | selected score decomposition, graph/objective identity, search resolution, and stop reason |
| `{tumor_id}_lambda_search.tsv` | evaluated raw candidate | lambda, raw objective/certificates, immutable partition, refit diagnostics, score decomposition, and selection eligibility |

All output tables include `selection_status`, `selection_optimum_resolved`, and
the controller stop reason. The run summary also records
`selection_constraint=none`.

Multiplicity performance must not be summarized over all mutations. Evaluate
exact multiplicity calls only on mutation-region rows where
`major_cn != minor_cn`; report pooled multiclass macro-F1 as the primary
statistic. See [the evaluation protocol](../docs/evaluation.md) and the reusable
[`evaluate_cna_multiplicity.py`](../scripts/evaluate_cna_multiplicity.py)
evaluator.
