# Changelog

## 0.4.0

This is a breaking internal-schema and product-surface release. Reproduce v0.3
behavior from its exact tag or commit; v0.4 does not emulate historical paths
inside the production solver.

Changed:

- fixed-partition BIC is the only production score;
- `hybrid-ward-cem-bic-v1` is the only production selection policy;
- all supported copy-number inputs compile to one immutable observed emission
  model and numerical dispatch follows hashed model structure;
- immutable `TumorData` now retains only normalized biological/input facts,
  three base inclusion masks, one `EmissionPaths` value, and compact
  `ExclusionCode` provenance; bounds, scaling, and start hints are derived;
- the raw solver accepts typed problem, plan, initialization, and attempt-budget
  objects through the lazy top-level API, with fixed-objective lambda supplied
  explicitly rather than stored in selection-wide `FitConfig`;
- `SolverContext` is the sole runtime authority; duplicate tensor-data/problem
  wrappers and optional source-model fallbacks were removed, and every context
  now fails closed unless its source likelihood, objective box, host graph, and
  bound runtime graph have one consistent identity;
- constrained nonlinear emission paths now derive their CCF upper bound from
  the exact piecewise-affine probability crossing; model compilation and the
  v0.3 input adapter share that calculation;
- dense and streamed certificate refinement now share one typed certificate
  controller and return `KKTAudit` evidence without diagnostic side channels;
- non-inferential runner controls and checkpoint intent use the frozen
  `RunConfig` and `CheckpointRequest` boundaries rather than parallel CLI and
  pipeline arguments;
- search, work, partition refits, outcomes, and raw-attempt diagnostics use
  composed typed state instead of parallel compatibility records;
- the live lambda controller, next candidate identity, and cumulative work are
  derived or updated through `SearchState` rather than shadow state;
- checkpoint schema v4 is rooted at an explicit search checkpoint, stores
  content-addressed arrays without pickle, and rejects older schemas;
- automatically named checkpoints now use a `.checkpoint` directory suffix
  rather than the obsolete `.npz` file suffix;
- checkpoint source identity hashes only shipped inference modules, so wheel
  builds, CI caches, and source-only simulation files cannot invalidate resume;
- output schema v2 writes exactly `{tumor_id}_analysis.json`,
  `{tumor_id}_clusters.tsv`, `{tumor_id}_mutations.tsv`, and
  `{tumor_id}_attempts.tsv`, with distinct model, likelihood,
  objective-specification, and graph identities;
- the normal CLI exposes eight fit options; expert settings moved to strict
  versioned JSON configuration;
- fitting and output no longer require pandas;
- reporting-model fingerprint schema v2 names generic `major_indicator`
  metadata; numerical model, likelihood, and posterior values are unchanged;
- the minimal canonical example replaces the former full dataset; and
- simulation moved under source-only `tools/simulation` and is excluded from
  the inference wheel.

Unchanged scientific boundaries:

- no clonal anchor or forced CCF-one mutation;
- the observed likelihood, box, complete graph, weights, row-group norm, and
  lambda define the raw estimator;
- partition refits remain secondary to the raw estimator;
- the full-original-graph float64 terminal KKT audit and fixed `5 * tol`
  admission threshold remain mandatory; and
- resource exhaustion remains unresolved rather than convergence.
