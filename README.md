# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See [`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for profile,
solver, resource, selection-score, partition-tolerance, unsupported-input, and
failure-policy controls. The default `--failure-policy best-effort` saves the
highest valid typed outcome without relaxing any certificate gate.

Long raw-lambda searches can use bounded staged recovery, deterministic work
accounting, and exact resume:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --max-tumor-edge-pass-equivalents 200000 \
  --checkpoint-every-lambda

# Resume later with the same input, resolved options, runtime, and source tree.
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --max-tumor-edge-pass-equivalents 200000 \
  --resume-checkpoint \
  exampleTumor1_results/.clipp2-checkpoints/exampleTumor1.npz
```

`--recovery-policy staged` is the default. It detects recovery-only plateaus
and probes terminal certificate refinement before spending the remaining
certificate budget; `legacy` is retained as a compatibility spelling for the
unstaged v0.3.5 ablation. It disables stagnation-based early termination but
retains the v0.3.5 recovery budgets, certificate implementation, and work
accounting; it does not reproduce commit `23fc222`, which must be run directly
for historical performance attribution. Neither policy changes the
complete-graph objective or the fixed `5 * tol` admission gate. The tumor work
cap is shared across starts, retries, fallbacks, and polishing. Exact realized
graph work is reported as integer edge-region visits; the integer edge-pass
budget charges every full traversal once and conservatively rounds each partial
workset traversal up to one pass. Work stops at safe boundaries with at most 10
edge-pass equivalents reserved for the mandatory terminal audit. The independent
`--max-partition-refit-objective-evaluations` and
`--max-direct-partition-candidates` controls bound post-guide scalar work and
the direct pool at candidate boundaries; a partial pool is explicitly
unresolved. The complete mandatory guide is never truncated because it can
define the graph and objective. Total, mandatory-guide, and post-guide scalar
work are reported separately in analysis-summary schema 8. Certified
refinement also stops after `--lambda-no-progress-patience` consecutive
proposals add no partition, score, KKT, or event-width information. None of
these resource stops is convergence. Checkpoints use an incremental,
content-addressed directory (the default path retains its historical `.npz`
suffix), save after both lambda observations and direct candidates, and only
atomically replace `manifest.json` after completed work.
Resume fails closed unless tumor, objective, graph, contract, full
configuration, warm-start policy, numerical environment and hardware,
software, and source-tree identities match exactly. Use `--checkpoint-file`
with `--checkpoint-every-lambda` to override the hidden default path. Legacy
monolithic NPZ checkpoint files require a fresh run at a new path.

## Outputs

Saved analyses use the tumor id as a prefix (the input file stem unless a
`##tumor_id` metadata line overrides it). Every saved primary, conditional, or
diagnostic outcome writes these status-rich files:

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_analysis_status.json` | analysis | tier, certification, selection, mask, and failure provenance |
| `{tumor_id}_region_status.tsv` | input region | estimate tier, data-mask counts, identification, and diagnostics |
| `{tumor_id}_raw_attempts.tsv` | raw solver start | objective, KKT components, budgets, realized work, dtypes, and hashes |
| `{tumor_id}_cluster_region_estimates.tsv` | selected cluster × region | best-available CCF and argmin-identification fields |
| `{tumor_id}_mutation_region_estimates.tsv` | mutation × region | retained counts, support/inclusion masks, CCF tier, and eligible posterior summaries |

Only a primary result (`primary_estimator_available=true`) writes the historical
compatibility files `mutation_clusters.tsv`, `cluster_centers.tsv`, and
`mutation_region_multiplicity.tsv`. A conditional fallback instead writes
`secondary_cluster_centers.tsv` and
`secondary_mutation_region_estimates.tsv`; a diagnostic-only result writes no
point-estimate compatibility file.

Every available point estimate also reports `ccf_ordered_cluster_label`, a
zero-based presentation label: cluster 0 has the smallest root-mean-square
distance from CCF 1 across its statistically identified regions, and the
remaining clusters follow in increasing distance. Exact ties use the immutable
`cluster_label`; clusters with no identified region sort last. This derived
ordering does not change `cluster_label`, selection, scores, or CCFs, and
cluster 0 is not a clonal-identifiability claim. Cluster-level tables also
report `ccf_distance_to_one` and the identified-region count used for it.
