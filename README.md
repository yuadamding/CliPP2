# CliPP2 0.5.0

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters mutations, and
infers integer mutant-copy multiplicity from single- or multi-region sequencing.
The production model uses marginalized binomial likelihood with observed-data
pairwise fusion. No clonal anchor or forced CCF-one block is imposed.

## Install and fit

```bash
pip install .
clipp2 fit --input-file examples/exampleTumor1.tsv --outdir exampleTumor1_results
```

The default is CUDA with the `balanced` computation profile. Use `--device cpu`
for a small smoke test; production GPU and cohort qualification are separate.
Working precision supports **float32 and float64 only**; float16 is retired.
Epsilon must retain strictly interior, ordered clipping endpoints in the chosen
dtype. Incompatible requests (for example, float32 with `eps=1e-12`) fail before
pilot/graph preparation, without adjusting epsilon or silently promoting the
objective. Extreme counts/slopes that risk candidate-arithmetic overflow also
fail preflight. Default float32/`eps=1e-6` remains supported.
`clipp2 fit --help` lists numerical profiles, solver/resource budgets, and refit
controls. Use a fresh output directory for each attempt.
The installed wheel includes `CliPP2/examples/minimal.tsv`; the larger example,
simulator, and tests stay in the source repository.

## Input and model

The canonical input has the 12 columns in
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv); extra columns are
inert provenance. The loader validates the complete file and aggregates
identical local CN states before filtering.

An entire mutation is excluded across **all samples** if any sample has multiple
distinct positive-fraction CN states or major CN greater than six. Different
clonal CN states between samples are allowed. `(6,6)` is retained; retained
`(0,0)` is unsupported. Exclusions never change the original input. Zero surviving
mutations produce `NoEligibleSNVsError` and a failed-run audit. One survivor is
fitted as the edge-free scalar problem, without unnecessary lambda search.

Each retained mutation–region unit has exactly `1, …, major_cn` multiplicity
candidates with uniform prior probability. Its candidate VAF is

```text
purity * multiplicity * CCF / ((1-purity)*normal_cn + purity*(major_cn+minor_cn))
```

Fitting and fixed-label refitting marginalize candidates with log-sum-exp; they
do not select a hard multiplicity during optimization. Reported multiplicity is
the smallest exact-tie posterior mode **conditional on the final refitted CCF**.
Missing or zero-depth observations have no informative multi-candidate call;
a singleton candidate is structurally fixed at one.

The sole selection policy is `hybrid-ward-cem-v1`, retaining certified raw-fusion
and Ward/CEM proposals and the existing Dirichlet-augmented fixed-partition score
(`alpha=1`, assignment weight `0.7`). The fusion objective, fixed-label output
meaning, and KKT gate are unchanged. Balanced admission remains `5*tol = 0.004`;
approximate refits and bounded unresolved searches are not global certificates.
Singleton multiplicity is not a convexity proof: probability clipping can
introduce nonconvex transitions. A raw global-optimum flag requires a separate
conservative proof over the actual source emissions, counts, and feasible box,
in addition to KKT admission. Failure of that sufficient proof means unproven,
not necessarily nonconvex.
Both pilot and final-CCF proposal pools use Torch curvature/Ward calculations
followed by the same host-side CEM and fixed-label refit. Plain Ward and Ward+CEM
candidates, empty-cluster repair, and fixed score parameters are retained;
the alternate Torch CEM/refit implementation and its switches are retired.

Release 0.5.0 retires binary/occupancy likelihood families and their options:
`--major-prior`, `--dosage-prior-penalty`, `--unsupported-policy`,
`--selection-contract`, and `--selection-score`. The CN cutoff is fixed, not a
runtime tuning option. Public fits require immutable validated `TumorData` from
`load_tumor_txt`; reconstruction must preserve biological/scaling consistency.
The public API is in `CliPP2.api`, reporting in `CliPP2.reporting`.
Candidates now compile directly from validated CN into the observed model;
`IntegerMultiplicitySpec` and `TumorData.path_likelihood` are removed. The loader
freezes one final input object, reuses its initialization model, and caches the
retained-input identity once. Reconstructing an input computes a fresh identity.

Use `fit_fixed_objective(data, config)` for one lambda. For deliberate reuse:

```python
from CliPP2 import load_tumor_txt, resolve_fit_config, prepare_problem, fit_prepared

options = resolve_fit_config(device="cpu", dtype="float64", lambda_value=1.0)
data = load_tumor_txt("tumor.tsv", eps=options.eps)
problem = prepare_problem(data, options)
fit = fit_prepared(problem, options.lambda_value, options.solver)
```

A prepared fit cannot accept competing data, graph, epsilon, or runtime options.
Changing them requires new preparation. Ordinary in-place changes to its private
Torch views fail before optimization. Float64 continuation rebuilds from frozen
float64 sources, not rounded working tensors. Loaded scaling, bounds, and the
deterministic initialization remain immutable and coherence-checked to preserve
pilot conventions; the compiled likelihood is cached per dataset/epsilon.
The prepared problem owns the source/runtime model pair directly; epsilon is
derived from its immutable objective key rather than stored again.

`fit_prepared(..., warm_state=previous_fit.state, include_default_starts=False)`
registers a warm-only continuation. If `phi_start` is also supplied, it is a
separate cold attempt, not an override of the warm primal. Disabling defaults
without either start is a configuration error. A singleton start on a clipping
plateau with feasible downhill likelihood also retains the existing scalar
pilot, even with defaults disabled; that comparison is not a global proof.
Returned `RawFit.phi`, partition labels, and fixed-refit arrays use immutable
buffers, preserving raw dtype. Copies and pickle roundtrips rerun their normal
constructors to restore validation and immutability; cached identities and
qualification are rebuilt. `SolverState` remains mutable numerical work state.
Discarded multistart fits are released after comparison. Search attempt traces
retain immutable diagnostics, never discarded fits or certificate tensors;
actual controller continuation and bracket states remain available.
Persistent fits keep both continuation state and independent certificate
witnesses on CPU, preserving tensor values, dtype, aliases, and solve-device
provenance. Terminal certification retains only its current evidence. One
refinement driver preserves the original full/chunked adjoint routes and uses
a frozen complete residual for each simultaneous update. Host CEM costs and
multiplicity posteriors skip derivatives without changing marginalization.
ALM likewise has one driver with the original dense/streamed arithmetic; its
outer caller receives only the actual edge multiplier. Curvature reuses the
validated prepared model, and Ward receives normalized tensors at its boundary.
Prepared problems own validation; CEM returns its existing fixed-label refit.
Terminal dense and compressed dual refinement use componentwise backward error
for retention, stopping and plateau decisions, matching final raw admission;
the legacy normalized residual remains diagnostic only. Full-step recovery
requires both loss majorization and objective descent within its numerical
allowances, increasing curvature on rejection without interpolating the primal.
Legacy and backward-error residual totals are derived
separately and fail closed on nonfinite or negative components; zero-radius
edges remain in the graph and use an exact-zero-safe residual calculation.

Ward reuses an `M x M` cost matrix with stable logical merge IDs and a shared
row-minimum heap; exact ties do not depend on physical slot order. CUDA still
uses host heap maintenance and transfers, so this is not fully device-resident.
Proposal refits remain interval-certified scalar searches even in balanced
mode; the final-refit policy is separate. A pool-local, bounded cluster-region
cache reuses only identical source/membership/bounds/numerical-policy problems,
preserving unresolved certificates and logical work diagnostics.

Production proposal curvature still uses the historical working-dtype loss
stencil. Its known float32 cancellation error is exposed by a separate strict
expected-failure accuracy test, not hidden by relaxing the graph goldens.
Source-float64 curvature is more accurate on the inspected smooth fixture,
but changing this graph/proposal metric needs separate numerical qualification;
clipping transitions also require an explicit nonsmooth convention.

## Outputs and integrity

The tumor ID is the input stem unless `##tumor_id` overrides it. A successful
run publishes four TSVs plus one manifest:

| Tumor-prefixed file | Contents |
| --- | --- |
| `mutation_clusters.tsv` | Immutable selected cluster labels and final refitted CCFs |
| `cluster_centers.tsv` | Selected cluster sizes and refitted regional CCFs |
| `mutation_region_multiplicity.tsv` | Retained mutation × region CCF, CN, integer MAP call and conditional probabilities |
| `excluded_mutations.tsv` | Original mutation × sample × exclusion reason |
| `run_manifest.json` | Publication status, numerical qualification, run/source identity, input/configuration/output hashes |

Integer columns include `multiplicity_candidates`, `multiplicity_candidate_count`,
nullable `multiplicity_call`, `multiplicity_call_probability`,
`multiplicity_informative`, and `multiplicity_p1` through `multiplicity_p6`.
Invalid candidates have zero probability. All CCF and multiplicity fields refer
to the same fixed-partition refit. Exclusion reason counts can overlap.

Existing tumor-prefixed files reject a new attempt **before** writing an audit.
There is no implicit overwrite or replacement mode. Failed runs preserve their
exclusion audit and non-complete manifest. Fit tables are staged and published
without clobbering; only after every file hash verifies is the manifest marked
`complete`. Consumers must check that status and the hashes, not merely the
presence of TSVs. Wheels retain hash-validated build provenance; unavailable
Git revision information is recorded as unavailable, never inferred.
Build and runtime use one dependency-free source inventory and hashing helper,
with case-sensitive, UTF-8 POSIX-relative filenames on every platform.

The manifest's `analysis` section separately records raw-reference KKT admission
and global status, selected-partition identity, fixed-label refit qualification,
and bounded-search status, including objective/graph/CCF hashes. A direct
Ward/CEM selection has no `selected_raw_fit` and does not inherit raw KKT
certification. Standalone fit writing records search status as `not_provided`;
unavailable facts are null. `analysis` is null before a qualified fit is ready;
it can remain available after a later publication failure. Publication
`status="complete"` never means global or resolved inference. Older manifests
without `analysis` provide no persisted numerical qualification.

All tables now consume one validated `AnalysisSerialization`. Standalone
`write_fit_outputs` uses that same boundary: ordered mutation/region identities,
counts, CN, purity, observation masks, epsilon, and bounds must match the fitted
source. Raw fits and fixed refits carry their own required source identity.
Old identity-free result objects are rejected, not silently repaired. Final-Phi
direct partitions also require their exact parent provenance; standalone writing
must receive that raw parent, while a full selection keeps parent and reference
separate. Table builders are private; use the two supported output writers.

Stdout summary schema **5** and the manifest share one immutable qualification
record. `raw_reference_*` fields describe the reference; `selected_raw_*` fields
describe an actual raw selection and are null for a direct selection. Ambiguous
`selected_full_kkt_*`, `selected_working_dtype`, and similar raw-reference aliases
are removed. `configured_raw_solver_primal_tol` is configuration metadata;
`raw_reference_solve_tolerance` records the actual solve's tolerance.

## Matched simulation

The simulator stays in the repository but is excluded from the inference wheel.
From an installed source checkout, run:

```bash
python -m tools.simulation --out-dir simulations --tumor-id simulatedTumor1 \
  --mutation-count 300 --clone-count 3 --region-count 2 --seed 1
```

`clipp2 simulate` is retired. The simulator generates clonal trunk gains inherited
unchanged by all descendants, with major CN at most six. Repeated gains and
physical-copy SNV acquisition determine integer truth; descendant SNVs have
multiplicity one. This evolutionary distribution is **not** the inference
model's uniform candidate prior.

Every requested mutation must survive whole-mutation filtering. Validation
rejects exclusions, incomplete truth, or inconsistent CCF/multiplicity/VAF
identities. The bundle contains canonical input, tree/CN/CCF truth,
`truth_mutation_sample.tsv:multiplicity`, and a schema-6 generator manifest.
`effective_multiplicity` is a cross-check of the integer truth.

This matched benchmark does not simulate subclonal CN, deletion/LOH events, or
different clonal CN between regions; these require separate eligibility tests.
Do not combine generator-v6 and older two-state-v5 results.

For multiplicity evaluation, the primary informative-ambiguity population is
`multiplicity_informative & (multiplicity_candidate_count > 1)`. Report pooled
exact-class macro-F1, micro/weighted/per-class F1, and eligible row count, with
balanced-gain, imbalanced-gain, and LOH strata. Preserve the separate historical
CNA-only `major_cn != minor_cn` F1; do not substitute all-mutation accuracy.

## Development and qualification

Use conda environment `ml1` locally:

```bash
conda run -n ml1 python -m pip install -e '.[test]'
conda run -n ml1 python -m pytest -q
conda run -n ml1 python -m ruff check .
```

The checked-in regressions cover filtering, singleton fits, marginalized
likelihood/derivatives, clipping certificates, graph/data identity, output
publication, simulation truth, and an isolated installed-wheel CPU fit. Tests
and tools are excluded from the wheel. GitHub Actions runs these same CPU gates.

Compare each revision with its explicit frozen numerical reference. Candidate
support, scalar evidence, pilot/final proposal pools, graph identities, refits,
scores and their uncertainties, and integer posteriors require paired validation.
Portable numerical references use explicitly bounded dtype-aware comparisons;
same-environment parity, labels and objective/graph identities remain separate
exact checks. CI records dependency and build versions for attribution.
CUDA and representative cohort/release-panel qualification remain necessary;
passing CPU tests alone is not evidence of improved benchmark accuracy.

Detailed dated evidence, parity captures, source measurements, and remaining
release gates live in [QUALIFICATION.md](QUALIFICATION.md).
