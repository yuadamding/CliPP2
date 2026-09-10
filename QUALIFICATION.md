# Numerical qualification

Dated evidence for the changes below, not a claim that a moving branch or a
new device/cohort is qualified. See [README.md](README.md) for current usage.

## Saturated certificate progress and bounded auxiliary work — 2026-09-10

Reference: `54c894c06d538b64e5f6264a7ab7d17f3163df5f`; version remains `0.5.0`.
Tested working-tree inference-source fingerprint:
`19a373cf5205ee796b74749fad5ef9511fcdae0e14b85bcd7a55da57dcd8be46`.
This section supersedes the earlier claim that backward error alone suffices
for plateau tracking. The historical observations below remain unchanged.

### Reproduced false plateau and correction

The review's four-node diploid binomial fixture has alternate counts
`(24,25,24,25)`, reference counts `(75,72,75,72)`, purity one, frozen phi
`(0.5,0.5,0.5,0.5)`, and gradients `(2,-2,2,-2)`. All six complete-graph edges
have weight `1/3`, lambda is 12, and the incoming witness has `y01=4`, others
zero. Its componentwise backward error is exactly one, but an explicit feasible
witness `y01=y23=-2` independently audits to zero at the SAME primal point.

The parent stops after 16 iterations and returns the original witness. Disabling
only its plateau abort, as a diagnostic control, reaches the unchanged gate
after 42 iterations. Backward error saturates while the unscaled box-cone
violation decreases: normalized saturation is not evidence of genuine stagnation.

Fixed-primal refinement now orders witnesses lexicographically by
`(componentwise backward error, unscaled box-cone violation norm)`. The secondary
quantity breaks exact primary ties; it cannot prefer a worse primary residual.
Material improvement in either merit resets plateau patience, but dual motion
alone cannot prevent a genuine stalled plateau. The original 8/16 patience
rules, projected-dual updates, fixed iteration budget, exact box semantics,
and `5*tol = 0.004` admission gate remain. Nonfinite values fail closed.
The secondary norm reuses the same audit adjoint without a second graph
reduction; it does not modify public diagnostics or substitute the legacy
globally normalized residual for certification.

Revised float64 CPU evidence (96 is the maximum budget, not extra patience):

| Budget | Executed iterations | Backward error | Cone violation norm | Gate passed |
| --- | --- | --- | --- | --- |
| 8 | 8 | 1.0 | 6.666669 | no |
| 16 | 16 | 1.0 | 4.714231 | no |
| 24 | 24 | 1.0 | 2.828433 | no |
| 96 | 42 | 0.003262204325 | 0.018393783 | yes |

The incoming violation is `sqrt(80) = 8.944272`. Budget-limited runs now retain
the improved witness WITHOUT claiming certification. Dense and 1/2-edge
streamed routes pass the feasible-witness regression in float32 and float64;
they preserve primal, counts, bounds, graph, weights, lambda and input dual.
Other tests retain first-iteration arithmetic goldens and check conflicting
primary/secondary/legacy rankings, nonfinite values, exact lower/upper/fixed
coordinates, nearest-representable interior coordinates, and genuinely stalled
cases. One mixed fixture deliberately changes from premature stop at 16 to its
existing 32-iteration budget, while remaining correctly uncertified.

### Positive-lambda recovery is tested, not merely inferred from a singleton

The new four-node binomial recovery regression executes the real complete-graph
ADMM at lambda 2 and records actual endpoints, summed likelihood majorization,
nonzero graph penalty, and independent raw audits. In float64 its first endpoint
lowers the penalized objective by `4.401882`, but misses loss majorization by
`+5.033594` and is rejected. The next full endpoint has majorization gap
`-1.850445`, decreases objective by `6.888710`, and is accepted undamped.
Both working dtypes pass; one-attempt exhaustion retains the original primal.

The accepted inner surrogate residual is `2.59694e-05`, whereas its final
raw-objective audit remains `1.0`. The test requires this fit to remain
uncertified, verifies the nonlinear gradient at the returned point, and
independently repeats the float64 terminal audit on the frozen source/graph.
Passing majorization or an inner surrogate certificate does not imply raw KKT
admission. No production recovery or curvature rule was changed in this pass.

### Memory and physical work: bounded terms, not whole-fit estimates

Ward refresh gathers contain at most `max(1,000,000, active_columns)` elements
per batch. Logical column order remains intact, so batching cannot change exact
ties. Deterministic heap compaction discards stale records when the heap exceeds
`max(64, 4*active_count)` after a merge. A merge can transiently add at most M
records; the replacement list also has at most M records. Initialization block
temporaries are released before the next block and before merging. Independent
full logical-matrix oracles verify every requested cut under aggressive batching
and compaction for float32/float64 random, duplicate, tied and zero-weight data.

The checked-in [Ward benchmark](tools/benchmark_ward.py) binds runtime source,
harness, environment, deterministic inputs, settings and repeated result hashes.
It separates uninstrumented timings from a Python-allocation/phase pass; optional
Torch profiling exposes kernel, copy and synchronization operations. CUDA use
requires explicit LSF opt-in and a bound source fingerprint.

Final-source `ml1` CPU refresh-stress fixture, M=2048, S=3, float32, one thread,
five warmed repeats: Ward-only median **0.564955 seconds**. This is a reproducible
observation, NOT a measured parent-to-patch or end-to-end speedup. Its largest
refresh gather is **3,995,744 bytes**, versus the former first-refresh allocation
of `2046*2047*4 = 16,752,648` bytes on the same fixture. Ten compactions leave a
maximum observed heap of 5,934 entries. The persistent cost matrix itself is
16,777,216 bytes; the largest single initialization pair-region tensor is
15,998,976 bytes, with several such temporaries potentially live together.
Python tracemalloc peaks at 1,370,602 bytes and excludes native tensor storage.
These are distinct, nonadditive allocation terms—not a complete peak-memory
bound. The source-bound receipt is
`../validation-saturated-plateau-20260910-n5701j/ward-stress.json`.

`_ScalarWorkStats` records cache hits/misses/evictions, dispatched scalar solves,
failed solves, actual interval-bound/grid-point evaluations, and scalar seconds.
Use `_work_stats=stats` on proposal generation; a coordinate cache shares its
single sink. Physical counts include attempted calls that raise; shortcuts and
hits do not invent solves/evaluations/time. Existing result counters remain
logical and are not reinterpreted. No source tensors or output fields are added.
The checked-in Ward/CEM fixture preserves complete returned proposals while
reducing actual solves **16 -> 14** and interval evaluations **1328 -> 1266**, with
**two coordinate hits**. This is exact avoided work, not a robust time estimate.
Counter collection has small overhead and does not make scalar refits device-side.

### Current checks and remaining release gates

Full final `ml1` CPU regression: **1,215 passed, 16 explicit CUDA skips, one
strict expected failure in 39.15 seconds**. Installed-wheel CPU fitting is
included. Ruff, compilation and `git diff --check` pass. Inference source remains
34 modules, increasing from 705,852 to 712,719 bytes. No likelihood, graph recipe,
score, public output schema, admission threshold or production curvature changed.

Fresh source-bound paired CNA smoke fits compare the revised tree with an
archive of actual parent `54c894c` (archive SHA-256
`eb9d2ab47804ac54a36f0cd4fe7e349acfd0145b7f85e0f5f0d35707ac6efe99`).
All four eight-mutation cases per source succeed: gain/LOH in float32/float64.
All **843 captured leaves** and **16 paired TSV files** are exactly equal,
including labels, refitted CCFs, scores, source/graph/objective identities,
qualification, posteriors, schema and CNA-only multiplicity F1. Each case has
eight eligible `major_cn != minor_cn` rows. Gain macro/weighted-F1 is `0.733333`,
micro-F1 `0.75`, per-class F1 `{1: 0.666667, 2: 0.8}`; LOH macro/weighted-F1
is `0.873016`, micro-F1 `0.875`, per-class F1 `{1: 0.857143, 2: 0.888889}`.
LOH float64 selects a direct partition; the other three select raw partitions.
Those families and their independent raw references are preserved. These are
tiny preservation fixtures, not representative accuracy estimates.

Evidence, copied harness, frozen archive and start/end source receipts are in
`../validation-54c894c-20260910-dkdDb3/`. The `comparison.json` SHA-256 is
`efb1f2b63098495b3e490d32e79ca07945ae6f6cb0cd07759911d9db3dcf11e7`.
Both source captures bind `ml1`; the revised fingerprint matches the one above.

Hosted [CPU regression and wheel run 34505021464](https://github.com/yuadamding/CliPP2/actions/runs/34505021464)
for committed **parent `54c894c`** is independently verified successful:
**1,094 passed, 12 skipped, one expected failure in 91.69 seconds**.
Its [CodeQL run](https://github.com/yuadamding/CliPP2/actions/runs/34505020224)
also succeeded. These post-commit results do not qualify this uncommitted patch.

Outstanding, explicitly not completed in this pass:

- CUDA and representative-cohort qualification. The `clipp2-run`/Seadragon
  workflow requires immutable committed source and the project LSF runbook;
  `/data/CliPP2/docs/seadragon-lsf-gpu.md` is absent. No remote jobs were launched.
  Run the opt-in tests on approved LSF GPUs after restoring that authority;
  separate frozen-objective CPU64/CUDA64/CUDA32 parity from full-workflow
  graph/proposal stability and measure actual full-fit time/peak memory.
- A complete Ward-specific admission preflight. Bounded refresh/heap work and
  measured allocation terms do not define a complete concurrent-memory model
  or authoritative Ward budget. Existing graph/ADMM preflights must not be
  presented as whole-workflow bounds. A rejecting Ward gate needs an explicit
  resource budget and validated live-allocation model, not an invented threshold.
- Production curvature accuracy. The strict expected failure is retained;
  changing the metric, including boundary/kink policy, remains a separately
  qualified numerical-policy change affecting proposals and graph construction.

## Certificate authority, recovery and phase costs — 2026-09-10

Reference: `eda308d77b26bf0b4188186fb6da1ffe7438a685`; version remains `0.5.0`.
Tested working-tree inference-source fingerprint:
`726ca995c679f098318436e4dba5b285cf37ea871a4670edcf29f1bc7048e252`.
Fresh reference archive SHA-256:
`3d617774cf93cf0f18279582c037bad117261fe5fa8fdd6ebea91653a7e0b3a5`.

### Correctness changes

The supported four-mutation binomial example reproduces the review: at fixed
phi `(0.5,0.5,0.5,0.5)`, gradients `(1000,-1000,2,-2)` and lambda 3000, the
incoming witness has legacy residual `0.000249911143` but componentwise backward
error `1.0`. The former routine returned without refining. Dense refinement now
uses backward error for initial admission, incoming/analytic/best-witness
ranking, target tests and plateau tracking. Compressed inherited-state and
final full-audit gates use the same authority; legacy diagnostics remain.

After 18 fixed-primal refinement iterations, backward error is
`0.0032621643204`, passing the unchanged `0.004` gate. An explicitly constructed
exact witness independently audits to zero. Primal, source counts, graph,
weights, lambda and input dual are unchanged. This corrects false-negative
certificate construction, not a demonstrated false-positive terminal audit.
A mixed-case golden deliberately changes its plateau decision from 22 to 16
iterations under the corrected authority; unchanged initial arithmetic and
opposite-ranking/target/plateau cases have separate assertions.

Recovery takes the stricter of the review's two proposed fixes: a full endpoint
must satisfy checked loss majorization AND Armijo objective decrease, within
the existing dtype-aware allowances. For the reviewed singleton likelihood,
the trial `0.4 -> 0.08` decreases loss by `0.3145609871` but has majorization
gap `4.6854390129`. It now triggers the existing curvature-enlargement/full-solve
path. Budget exhaustion retains the original primal without damping. This is
an optimizer acceptance correction, not a behavior-preserving refactor or a
change to the statistical objective.

Selection boundaries now use every eligible, admitted raw lambda before
partition deduplication. A hit refers to the selected representative lambda,
not the partition's separate support interval. A at 0.1, B at 1 and A again
at 10 selects the same B/refit/score without falsely calling 1 the upper
boundary. Rejected raw candidates and direct proposals' parent lambdas cannot
extend the range. Tests cover reordering, redundant duplicates, a direct
representative of a raw-supported partition, and the zero-edge singleton.

### Measured behavior-preserving reductions

Ward reuses `M x M` physical cost slots while retaining increasing logical
merge IDs and exact heap ties. Shared row-minimum maintenance removes the
unconditional full-matrix scan on every multi-region CPU merge. It remains
quadratic storage; row refreshes, pathological ties, host heap growth and
other solver allocations are not eliminated. CUDA still uses host transfers
and synchronization; no fully device-resident implementation is claimed.

In `ml1` (Python 3.13.2, Torch 2.9.1+cu128, NumPy 2.2.6), 848 cuts across 40
random/tied/duplicate/zero-curvature fixtures match both the frozen parent and
an independent full logical-matrix oracle exactly, in float32/float64. A
read-only second review checked 24 additional 67-node/four-region near-tied,
heterogeneous-curvature histories; every cut also matched.

Three-repeat warmed, single-thread **Ward-only CPU** medians:

| M | Regions | Dtype | Parent seconds | Revised seconds | Ratio |
| --- | --- | --- | --- | --- | --- |
| 128 | 3 | float32 | 0.01603 | 0.01515 | 1.06x |
| 512 | 1 | float32 | 0.06678 | 0.06502 | 1.03x |
| 512 | 3 | float32 | 0.52754 | 0.07533 | 7.00x |
| 1024 | 3 | float32 | 4.03124 | 0.17513 | 23.0x |
| 1024 | 6 | float32 | 4.02334 | 0.19319 | 20.8x |
| 1024 | 6 | float64 | 4.24689 | 0.21627 | 19.6x |

The M=1024 float32 persistent cost matrix decreases
`16,760,836 -> 4,194,304` bytes (about 75%). This is NOT whole-process peak RAM,
GPU memory, complete workflow admission sizing, or end-to-end acceleration.
Source-bound reproduction and timing samples are retained outside Git in
`../validation-ward-20260910-LmXDWE/benchmark_ward.py` and `observed.json`.

Proposal generation owns a cluster-region LRU cache capped at 1,024 entries
and 8 MiB of aggregate membership-key payload, without model/source-array
retention. Keys bind source/model identity, sorted exact membership, region,
bounds, epsilon, mode, effective coordinate tolerance, iteration/grid/local-step
controls and breakpoint policy. Tolerance depends on K times region count:
unchanged membership at another K cannot reuse a different-tolerance solve.
Complete results retain exhausted/resolved status, lower bounds, uncertainty
and logical work diagnostics. The full-partition cache and interval-certified
proposal-refit policy remain, including in balanced mode.

Two mixed-CN partitions sharing clusters need **16 -> 12** actual scalar calls;
all refit fields are bitwise identical. A tiny real Ward/CEM pool needs
**16 -> 14** calls with identical complete proposals. These establish avoided
work, not robust end-to-end acceleration. Tests cover source/policy rejection,
LRU/byte eviction, missing/zero-depth rows, exhausted results, source release
and canonical labels. No cache survives its proposal-pool call.

### Curvature: confirmed limitation, not silently changed

Production retains the historical float32 stencil. Its old goldens and exact
downstream proposal/graph tests have NOT been loosened. Independent mixture-
Hessian and float64 autodifferentiation checks agree. At the SAME promoted
float32 pilot, the reviewed coordinate has:

| Calculation | Curvature | Relative error against analytic |
| --- | --- | --- |
| Production float32 loss stencil | 44.7052078247 | 15.8559% |
| Source-float64 loss stencil | 53.1311662305 | 0.0034224% |
| Analytic / independently autodifferentiated float64 | 53.1293479161 | reference |

With the quantile cap disabled, source-float64 stencil error is below `1e-4`
relative on every smooth coordinate in this fixture. A separate strict expected
failure records that production float32 does not meet that accuracy target;
it is not counted as a passing accuracy test. Other tests distinguish clipped
plateaus, an exact kink (no ordinary Hessian), and a stencil crossing a kink.
Float64 fixes loss-cancellation error, not the nonsmooth-stencil interpretation.

Promoting this metric or replacing it by an analytic Hessian remains a separate
behavior-changing numerical revision: Ward merges, proposals, noise scale and
graph weights can change. This pass evaluates the alternative but does not
switch the production estimator or claim that curvature accuracy is fixed.

### Integration, CI and outstanding qualification

The full final `ml1` CPU suite passes **1,094 tests**, with **12 explicit CUDA
skips** and **one strict expected failure for the documented curvature error**.
This includes isolated installed-wheel CPU fitting. Ruff and diff checks pass.

Fresh source-bound captures run four 8-mutation CNA-positive CPU hybrid fits
(gain and LOH, each in float32/float64) against the actual parent archive. All
four succeed with a certified positive-lambda raw reference and float64 audit.
Complete captures are byte-identical: labels, refitted CCFs, scores,
graph/objective identity, numerical qualification, posteriors, all four TSVs
and exact `major_cn != minor_cn` macro/micro/weighted/per-class F1:
`52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4`.
Each fit has eight eligible rows. Gain macro-F1 is `0.733333`, micro `0.75`;
LOH macro-F1 is `0.873016`, micro `0.875`; weighted equals macro on these balanced
truth classes. These tiny fixtures establish preservation, not cohort accuracy.
Baseline/revised receipts and the harness are in
`/storage/CliPP2/validation-eda308d-20260910-HLL87s`; both captures assert source identity at start
and finish. No production timing claim is drawn from these executions.

Hosted [CPU regression and wheel run 34498265860](https://github.com/yuadamding/CliPP2/actions/runs/34498265860/job/102942190044)
for **parent `eda308d`** is independently verified successful:
**1,014 passed, 11 skipped in 86.61 seconds**. This supersedes the pending-CI
statement below; it does not qualify this new patch on the hosted runner.

CUDA and representative cohorts remain unqualified. The `clipp2-run`/Seadragon
skills require an immutable committed source and the project LSF runbook;
`/data/CliPP2/docs/seadragon-lsf-gpu.md` is absent, and this patch is not committed.
No remote jobs, commits or pushes were made. CUDA-only tests require explicit
`CLIPP2_TEST_CUDA=1` inside an approved Seadragon LSF allocation.
Follow-up must separate fixed-objective CPU64/CUDA64/CUDA32 parity from whole-
workflow graph/proposal stability. Measure GPU compute, transfers/host heap,
scalar counts/time, certificate effort, retries, fallbacks, end-to-end time and
peak memory on frozen representative inputs. Include low purity, heterogeneous
depth, missing observations, balanced/imbalanced gains, LOH and region-varying
clonal CN; unsupported subclonal CN is an exclusion/audit test, not an
estimation-accuracy target.

Inference source remains 34 modules (`702,406 -> 705,852` bytes). The small
net increase buys bounded exact reuse and correctness; no likelihood, fusion
objective, graph recipe, score, output schema or balanced admission gate changes.

## Portable curvature tests and final interface cleanup — 2026-09-10

Reference: `932ea66aa221faebffbed785fcd3bafae6581c25`; version remains `0.5.0`.
Final inference-source fingerprint:
`ab0b1e861448a51c6b830a7cadd79c34d93303a181f8251cb39a64dbb448adeb`.
Frozen reference archive SHA-256:
`40c4d362d9295e732dfee82645155aa748edb2a04450f00018272d2c532de5bb`.

### CI failure and its numerical cause

The prior local result of **963 passed / 11 skipped** remains valid historical
evidence, but did not qualify another dependency environment. The pinned
[GitHub Actions job 102927160696](https://github.com/yuadamding/CliPP2/actions/runs/34493845322/job/102927160696)
passed lint and failed regression with **1 failed / 962 passed / 11 skipped**.
Its exact-equality curvature assertion stopped the combined test before its
downstream labels, proposals and graph comparisons could execute.

The original frozen test was reproduced with its CI dependency versions in a
separate local environment. `ml1` was not modified. Both actual implementations
were exercised with recomputed pilots: the supplied-model path and the former
reconstruction convention using an independently constructed runtime model.
Hooks captured the actual helper's left/center/right likelihood evaluations,
pre-cap curvature, quantile cap and post-cap result; this was not a replacement
implementation of the curvature arithmetic.

| Environment | Python | PyTorch / NumPy | Float32 quantile cap |
| --- | --- | --- | --- |
| Historical/local `ml1`, CPU execution | 3.13.2 | 2.9.1+cu128 / 2.2.6 | `238.96022033691406` |
| Original hosted CI, Ubuntu 24.04 | 3.13.15 | 2.14.0+cpu / 2.5.3 | `238.96023559570312` |
| Isolated local CI-dependency reproduction | 3.13.2 | 2.14.0+cpu / 2.5.3 | `238.96023559570312` |

The first divergence is **`torch.quantile(..., 0.995)`**, not model reuse:
pilots, all three likelihood arrays and pre-cap curvature are bitwise equal
across the two executed local environments. The cap differs by exactly one
float32 ULP (`1.52587890625e-5`, relative difference about `6.39e-8`). Float64
curvature is unchanged. Reuse/reconstruction match bitwise within **each**
environment at every captured stage. Explicit downstream tests also establish
unchanged Ward labels, proposal order/families, refits, scores, graph weights,
scale and fingerprints for this fixture in both environments. These results
localize a build-dependent quantile rounding difference; they do not establish
universal cross-platform bitwise reproducibility or a cohort accuracy claim.

The combined regression is split into independent construction-count,
same-environment intermediate parity, pilot/source identity, portable curvature,
Ward behavior, proposal/refit/score, and graph-identity tests. Only the portable
curvature comparison permits **one ULP in its working dtype**, justified by
the observed matrix. The stored goldens are unchanged. Labels, families,
scores, CCFs and graph/source identities retain separate exact assertions, so
a portable numerical mismatch no longer suppresses their execution. CI now
records dependency versions, Python/platform information and Torch build
configuration; it does not pin an older Torch merely to hide the discrepancy.

### Bounded compaction and acceptance

CEM returns its accepted `PartitionRefitResult` directly; the duplicate
`PartitionRefinementResult`, copied labels and refit-derived K bookkeeping are
removed. Exhaustive small repair cases establish that the retained empty-cluster
repair policy preserves occupied K. Published `component_death_count` remains
zero without independently tracking it. Leave-one-out costs, strict score
improvement, canonicalization, cached refits, and both Ward/CEM families remain.

`PreparedProblem.validate()` owns the previous validator body, with unchanged
condition order and error messages. Fitting still rejects deferred graphs;
pre-graph proposals explicitly opt into them while retaining every source,
graph, objective and tensor-version check. The proposal module no longer
imports a private optimizer validator. The normalized validation body matches
the reference exactly after receiver renaming.

Ward consumes matching float32/float64 tensors and allocates from them. Pilot
and optional NumPy curvature normalization happens once at the proposal
boundary. The extra runtime resolution inside each Ward call is removed.
Cost arithmetic, chunking, heap/dense dispatch, tie-breaking and merge order
are unchanged, including zero-curvature and near-tied fixtures.

Both full CPU test environments pass **1,014 tests with 11 CUDA-only skips**;
Ruff and `git diff --check` pass. The isolated dependency environment was
unchanged before/after the suite. Its local Python patch version and host are
not the hosted runner's, so this is **not a new green GitHub Actions run**.
Fresh source-bound `ml1` captures match the pinned reference byte-for-byte:

| Capture | Coverage | Matching SHA-256 |
| --- | --- | --- |
| Ward | 36 cases: both dtypes, one/three-region routes, full/chunked initialization, ties and every requested merge | `0b2fa4643b50281580063e1d32030fa570445cd01c1dffff66c39b23860a7e50` |
| CEM repair | 14,463 assignment/repair cases, including tied/infeasible costs and input preservation | `1a92fff7e77749126a65f2dd5eb0c3e1a29b9d728ac0088a1187684911133754` |
| CEM refits/proposals | 36 complete refits and six ordered proposal pools | `1500e6f28c3218fb1e6d8ca691225ccc02447182b057b6f91ff768df16ca1838` |
| Raw/warm/hybrid | 17,994 numerical leaves, including identities, certificates, refits, scores and uncertainty | `ca13c62e829becec31422936a8e0e2f1f0d0f5f4c57cd16f582e00c4c3eac226` |
| CNA-positive hybrid | 566 numerical leaves, public outputs and CNA-only macro/micro/weighted/per-class F1 | `52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4` |

The CNA population remains exact `major_cn != minor_cn`, eight eligible rows
per tiny fit. All four CNA fits preserve certified positive-lambda references,
float64 audits and identical four-TSV contents. Some deliberately bounded raw
fixtures remain uncertified identically; parity is not cohort success.

The final reachability/ownership audit found concrete consumers for public
exports, resource exceptions/preflights, compressed certificates, CUDA
compiled/eager fallback and precision recovery. None is removed merely because
it is exceptional. All production ALM warm inputs are actual multipliers or
absent; direct low-level tests still exercise the optional scaled-input
convention, which is retained in this bounded P1–P3 pass. No second optimizer
rewrite or arbitrary module-count reduction is warranted by this audit.

Inference source remains **34 modules**, decreasing **703,778 → 702,406 bytes**
(1,372 bytes; 0.19%). No likelihood, graph, fusion norm, scalar reduction,
score, candidate family, output schema, or balanced `0.004` gate changes.
Source contraction and removed redundant work are not a measured GPU speedup.
Hosted CI on the eventual committed patch, CUDA and representative release
panels remain pending. No commit, push or remote run was made during this pass.

## Residual correctness and ALM consolidation — 2026-09-10

Reference: `aa799344d2addb043b85034928cdb3c50e2d9b4b`; version remains `0.5.0`.
The final tested inference-source fingerprint is
`23e2150c18694bf1dbbacab9dcdbe6d695ada86c15f5c017ee4dbd11dc6e40a2`.
The frozen reference archive SHA-256 is
`65bc46668dfda954b7ac1b0fa5a0220ccab82e8abade3b4aa87607dac4c8e1cc`.

The reported residual bug reproduces in the actual reference package. With
`phi=U=(0.2,0.2,0.8)`, complete-graph weights `(0,1,1)`, zero dual and lambda
one, float32 produces a NaN edge component but aggregate residual zero.
Float64 reports legacy residual `0.3` and componentwise backward error `0.6`.
The feasible point `(0.21,0.21,0.79)` lowers the quadratic-plus-fusion objective
from `1.2` to `1.16015`. This establishes an incorrect working diagnostic and
potential stopping decision, **not** a demonstrated false public terminal
certificate: the separate authoritative float64 audit remains in place.

The edge residual now replaces only an exactly zero denominator with one;
nonzero denominators and zero-weight edges are unchanged. Both dtypes report
the expected violation. One scalar residual maximum rejects every nonfinite
or negative component with infinity. On-device maxima preserve NaNs and reject
negative values without a host synchronization. Dense, streamed ALM, compressed
certificate and omitted-edge scans cannot erase invalid components by taking
a maximum with zero. Invalid radius normalizers also fail closed instead of
converting a finite violation into zero by division by infinity.

`KKTDiagnostics` derives its legacy and backward-error totals from their
separate component sets; `KKTComponents` uses the same fail-closed rule. Their
individual diagnostic values remain available. `FitProvenance.likelihood_eps`
derives exactly from the typed objective key. Removed duplicate constructor
fields are not reintroduced by a compatibility wrapper. External diagnostic
properties and published output schema remain unchanged; malformed epsilon
keys can now fail earlier at hexadecimal parsing.

ALM and PDHG return five values: primal, actual edge multiplier, iterations,
convergence and typed diagnostics. Only PDHG's existing closed-form branch can
omit diagnostics. Scaled ADMM duals remain inside ALM; both actual/scaled
warm-start initialization modes remain supported and tested. The outer result
and warm state share the actual multiplier without a second outgoing dual.

One ALM driver replaces its dense/streamed control loops. Each iteration
completes shrinkage and the full adjoint before solving nodes, then completes
every dual update before deciding rho. The original full-array reductions and
streamed accumulation orders remain distinct, including their historical
pre/post-rescaling audit timing. The box-QP solver, compiled/eager dispatch,
spectral-rho schedule, actual-multiplier invariance, stopping convention,
tolerances, audit cadence and work counts are preserved. Even a one-edge
problem retains the streamed route when its byte budget is below one edge.

Curvature consumes one supplied `TorchObservedModel` and same-shape/dtype/device
pilot tensor. Proposal boundaries validate the prepared source and runtime
tensor mutation stamps, including pre-graph contexts; ordinary fits still
reject a deferred graph. The redundant runtime resolver is deleted. On the
pinned five-mutation/two-region preparation, initial curvature, initial pool
and final pool sequence, runtime constructions decrease **3 → 1**. Actual
immutable source construction remains one on both sides; extra boundary
validation uses its existing source cache, not another compilation. Curvature,
scalar pilots, Ward ordering, labels, refits, scores and adaptive graph weights
match their pinned values in both dtypes.

Validation in `ml1`: **963 tests passed, 11 explicit CUDA-only tests skipped**;
Ruff and `git diff --check` pass. This includes the isolated installed-wheel
CPU fit, an independent exhaustive active-set QP/ADMM oracle, actual/scaled
warm continuation, exact rho-rescaling checks, zero-radius violations,
NaN/infinity/negative injection into every residual position and chunk, and
edited prepared-tensor rejection. Fresh paired CPU captures bind the final
source at startup and completion:

| Capture | Coverage | Matching SHA-256 |
| --- | --- | --- |
| Inner solvers | 280 route-specific ALM/PDHG cases, both dtypes, warm modes, active/frozen bounds, zero lambda/no edges, both stopping conventions; every iterate/audit/rho trajectory, including 18 rho-changing cases | `cdd15ee6a2cb0dc58862084cba52376b19aefa0d21c992323e61ea97ab1b6f21` |
| Raw/warm and hybrid | 17,994 numeric leaves; ten raw/warm cases, two tiny hybrid fits, flattened diagnostic totals, derived epsilon, identities, refits, scores and uncertainty | `ca13c62e829becec31422936a8e0e2f1f0d0f5f4c57cd16f582e00c4c3eac226` |
| CNA-positive hybrid | 566 numeric leaves; four fits, public CCFs, labels, posteriors, exact `major_cn != minor_cn` F1 population and all four TSV contents/schemas | `52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4` |

Valid-case captures are byte-identical to the reference. Deliberately invalid
residuals and the zero-radius bug reproduction must change; they are covered
by correctness tests rather than forced into a parity claim. The CNA fixtures
have eight eligible rows each and preserve macro/micro/weighted/per-class F1;
this is smoke-test parity, not evidence of improved cohort accuracy.

Inference source remains **34 modules**, decreasing **713,768 → 703,778 bytes**
(9,990 fewer bytes; 1.40%). No likelihood, objective, graph, candidate family,
score, output schema, or balanced `0.004` admission gate is changed. These
measurements establish source contraction and eliminated runtime model builds,
not a production runtime or peak-memory improvement.

CUDA and representative release/cohort qualification remain pending. No remote
jobs were launched. The eleven CUDA tests require `CLIPP2_TEST_CUDA=1` inside
an approved, commit-pinned Seadragon LSF allocation; they now also cover the
ALM compiled/full/streamed routes and cross-precision zero-radius audit. The
`clipp2-run` skill requires an immutable committed source and the external LSF
project runbook, which is still absent. This working patch is not a committed
run source; local CPU evidence does not replace those release gates.

## Whole-fit ownership and certificate consolidation — 2026-09-09

Reference: `72f76a8c2ab3066ab1df01b5288d303e73e2f8d0`; version remains `0.5.0`.
The final tested inference-source fingerprint is
`2c0af4df27314b582dd71601324d8c3ec64d7dfc7d17b5173af2293800593ee1`.
The frozen reference archive SHA-256 is
`85dde34f48086c56a63920a9e22c3ef243b9694eecda6e42d79f9cdd0856e70c`.

The reported ownership issue is real: replacing only `RawFit.state` leaves
its separate `certificate.witness` on the original device. The replacement
`offload_raw_fit_to_cpu` moves both, including dense/compressed witnesses,
warm-state certificate hints, and witness-only fits with no state. One tensor
memo reuses transfers for identical views across all those owners. Values,
dtype, graph/scope evidence, diagnostics, and recorded solve device are
preserved. Guided initialization moves its two final state tensors to CPU at
construction, so no second generic state-only offloader remains.

Terminal certification adds work counters immediately instead of keeping
every refinement result. It also releases the temporary edge-dual alias after
computing the next generalized-gradient adjustment. The four-pass limit,
optional fifth pass, gradient reconciliation, and final audit are unchanged.
A pinned, instrumented terminal-loop test observes previous live witness
counts **[0,1,2,3,4] → [0,1,1,1,1]** while retaining all five passes and the
same summed work count of 15. The float32 variant additionally checks that
only the final witness survives when the float64 audit begins. These tests
mock certificate results inside actual solver orchestration; they are not
end-to-end GPU memory measurements.

One dual-refinement driver replaces the vectorized and streamed drivers.
Each iteration freezes the complete adjoint/residual before updating any
chunk. Incoming/analytic comparison, strict best-witness updates, ties,
projection, plateau rules, status strings, and work accounting are preserved.
The one-chunk route still uses the existing full graph-adjoint implementation;
the streamed update still uses its original bounded scatter accumulation.
No new forced-streaming CUDA reduction route is introduced. Full-graph audit
routes, including their existing deterministic complete-graph path, remain
unchanged.

Numerical routines now exchange `KKTDiagnostics` directly. Obsolete mapping
conversions and unused residual-copy/cache blocks are deleted; legacy progress
residuals remain distinct from terminal componentwise backward errors.
`RawAttemptTrace` composes the existing frozen, tensor-free `ConvergenceResult`
and derives its aggregate KKT residual. Established failure-message fields and
tokens remain exact, without reintroducing a witness-owning certificate record.

The NumPy evaluator has one private emission/reduction path for loss-only,
posterior-only, and full-derivative work. CEM keeps its existing center loop;
reporting keeps the same marginalized conditional posterior and MAP tie rule.
Internal emission/candidate names replace retired path terminology without
compatibility aliases. Legitimate lambda-path names and public refit
provenance strings are deliberately unchanged.

Validation in `ml1`: **637 tests passed, 6 CUDA-only tests skipped**; Ruff and
`git diff --check` pass. The suite includes isolated wheel installation and
CPU fits, independent simultaneous-update calculations, nonfinite/fail-closed
diagnostics, both witness representations, alias preservation, stateless fits,
and recovery/bracket continuation tests. Pinned fixed-environment CPU captures
are byte-identical, with final-source fingerprints checked before and after
each capture:

| Capture | Coverage | Matching SHA-256 |
| --- | --- | --- |
| Certificate refinement | 36 cases: both dtypes/chunk modes, incoming/analytic/refined witnesses, complete residual sequences and statuses | `02b4f85153360dd5cc96d2cb295961b3f6187e0065759737739480d25eab8a57` |
| Inner solves | 24 ALM/PDHG cases: dense/streamed, zero/positive lambda, legacy/componentwise stopping, values and work counts | `cd28ff64a9ed5bfb6ec5703241435f565fd0d4ca4e5f9d9c80de18a254ab572e` |
| Host reductions | 2,424 numerical leaves: full terms, center costs, CEM decisions, conditional probabilities and MAP calls | `bf4094d42e22e239a859ab31c1a7d8406b4ffc832d318b002f713dfc656b7038` |
| Raw/warm and hybrid | 17,900 numerical leaves: objective/graph identities, basins, certificates, scores and uncertainty | `4041b281a0e16ddbb3405e6a11d367d5080ebdf21b40b68527d39fe56ab96bbd` |
| Proposal pools | 3,432 numerical leaves: ambiguous multi-region pilot/final-Phi pools, labels, refits and ordering | `a2447cfe2b7d2d43318f490cf32d9166e84e67bafc539ac83e2e25b937fffe72` |
| CNA-positive hybrid | 566 numerical leaves: public refits, posteriors, CNA-only F1, qualification and four TSVs | `52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4` |

For a 256-mutation, 3-region, 6-candidate host fixture, instrumented derivative
arrays decrease **3 arrays / 110,592 bytes → zero** per loss/posterior call.
Three center evaluations avoid 331,776 aggregate derivative-array bytes. Full
derivative evaluation is unchanged. This measures selected array allocations,
not total peak memory, RSS, latency, or production VRAM. Small-fixture F1
parity likewise is not a cohort-accuracy claim; the CNA checks retain exact
`major_cn != minor_cn` eligibility and public refitted CCFs.

Inference source remains **34 modules**, decreasing **725,861 → 713,768 bytes**
(12,093 fewer bytes; 1.67%). Tests and qualification evidence remain in the
repository. No likelihood, graph, fusion norm, score, candidate family, global
optimality safeguard, output schema, or balanced `0.004` admission gate changed.

CUDA and representative release/cohort qualification are still pending. No
remote jobs were launched. Six tests are explicitly gated by
`CLIPP2_TEST_CUDA=1` for execution only inside an approved, commit-pinned
Seadragon LSF allocation: persistent dense/compressed history, witness-only
fits, and one-/multi-chunk adjoint routing. The `clipp2-run` workflow requires
an immutable committed source and the external LSF project runbook, which is
currently absent. CPU results do not qualify CUDA behavior or GPU memory use.

## Deletion-first compaction — 2026-09-09

Reference: `bb726ad3fb3737cca48e06cce66943d5fbba52b4`; version remains `0.5.0`.
The tested inference-source fingerprint is
`efe7f40e2912f516f12f5418a064557796a5925766325d85f11eab6ee755cd76`.
This pass completes the precision/memory contraction below without changing
the integer likelihood, graph, fusion penalty, hybrid candidate families,
Dirichlet score (`alpha=1`, weight `0.7`), KKT gate, or output meanings.

Production proposals now have one host CEM/refit implementation; Torch still
computes curvature and Ward labels. The alternate Torch refit/CEM branch,
fixed-policy switches, redundant rescoring, and duplicate best-refit tracking
are deleted. Leave-one-out allocation costs, empty-cluster repair, strict score
improvement, refit caching, both candidate families, ordering, and deduplication
are preserved. Shared constants now live in `config.py`, conditional integer
posteriors in `core/objective.py`, and initializer orchestration in
`core/fusion/partition_starts.py`. Their three former modules have no
compatibility shells. Lazy public imports retain their names and identities.

Search traces contain immutable diagnostics, not fits, primals, solver states,
or certificate witnesses. Start comparison retains its stable ordering and
tie rules. Same-lambda recovery retains precisely the minimum finite-KKT
state previously selected from the complete attempt list; actual selected
history and bracket states remain available. Failure diagnostics use typed
fields; the duplicate diagnostic class and obsolete-object fallbacks are gone.
The prepared problem derives epsilon from its immutable objective key.

Validation in `ml1`: **545 tests passed**, including the independently installed
wheel and its minimal runnable example; Ruff and `git diff --check` pass.
The wheel contains only that minimal example, while the larger example,
simulator, and regression evidence remain in the repository. Tests cover
retired interfaces, host-CEM statistical invariants, exact pinned failure
diagnostic fields/tokens, streaming ties and nonfinite comparisons, and
state release across actual search recovery/bracket orchestration.

Fresh fixed-environment CPU captures match the pinned reference byte-for-byte
in both float32 and float64:

| Capture | Coverage | Matching SHA-256 |
| --- | --- | --- |
| Proposal pools | Ambiguous CN2/3/6, 8 mutations, 2 regions; pilot and final-Phi Ward/CEM labels, refits, ordering, scores and selection | `a2447cfe2b7d2d43318f490cf32d9166e84e67bafc539ac83e2e25b937fffe72` |
| Raw/warm and hybrid | Ten raw/warm cases and two hybrid fits; 17,900 numerical leaves | `4041b281a0e16ddbb3405e6a11d367d5080ebdf21b40b68527d39fe56ab96bbd` |
| CNA-positive hybrid | Four fits; public labels/refit CCFs, score uncertainty, posteriors, CNA-only F1, qualification and four TSVs | `52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4` |

Proposal captures contain nine pilot candidates in each dtype, nine final-Phi
candidates in float32, and ten in float64; both Ward and distinct Ward+CEM
partitions are exercised. CNA-positive fits each contain eight positive-depth
`major_cn != minor_cn` rows. These small fixtures establish parity, not cohort
accuracy or a new F1 performance estimate. Graph/objective identities, selected
basins, certificates, and reporting remain equal in the paired captures.

A larger state-lifetime probe uses real `fit_prepared` orchestration with a
mocked numerical solve: 256 mutations, 3 regions, 32,640 complete-graph edges,
and 783,360 bytes per unique float64 dual buffer shared by state and witness.
At 4/12/32 starts, the reference retains 4/12/32 such buffers at peak; the new
low-level fit and isolated diagnostic-projection probes each peak at two
(incumbent plus current), with one winner remaining afterward. At 32 starts
this is **25,067,520 → 1,566,720 bytes**, a 93.75% reduction in that measured
storage component. Attempt order, tied winner, and identities match. Separate
search tests preserve incumbent, recovery, and historical selected states.
These are weak-reference live-buffer measurements, not RSS, end-to-end solver
memory, production GPU VRAM, or runtime speedup measurements.

The canonical inference inventory decreases **37 → 34 modules** and
**767,700 → 725,861 source bytes** (41,839 fewer bytes; 5.45%). AST counts are
364 → 343 top-level functions, 128 → 120 methods, and 29 → 25 nested functions.
The README now focuses on usage and supported behavior; dated qualification
history is preserved in this document rather than deleted.

CUDA and representative cohort/release-panel qualification remain pending.
No scheduler jobs were launched: the external LSF project runbook is absent,
and the worktree is not yet an immutable committed run source under the
`clipp2-run` contract. No objective or certificate threshold was relaxed to
obtain the CPU parity results.

## Precision and memory contraction (first pass) — 2026-09-09

The reference is `bb726ad3fb3737cca48e06cce66943d5fbba52b4`. This pass narrows
precision support, retains only the best completed multistart fit/context,
omits clipped-slope tensors and masks from likelihood-only grids, and removes
the intermediate `TorchTumorData` wrapper. `PreparedProblem` owns immutable
source and working models, epsilon, and objective identity directly. Promotion
and terminal auditing still rebuild from float64 sources; ordinary runtime
mutations still fail identity checks. No compatibility wrapper remains.

Scoring has one BIC primitive, one Dirichlet mass/uncertainty primitive, and one
final `fixed_partition_dirichlet_score` constructor. Plain-BIC/duplicate numeric
score interfaces are retired, not the BIC contribution. Configuration owns the
unchanged `alpha=1` and assignment weight `0.7`; the score retains nominal
`K*S` degrees of freedom and the positive-depth observed-row count convention.
Score arithmetic uncertainty
and ranking boundaries are preserved, as are the graph, likelihood, scalar
refit modes, raw admission gate, output schemas, and reporting validation.

Validation in `ml1` includes the isolated installed wheel, precision endpoint
and padded-candidate regressions, exact score/uncertainty goldens, and weakref
state-lifetime checks. With 4 or 32 starts, only the incumbent completed-start
dual remains before each next solve. Composite compressed/dense/CPU-fallback
tests also check discarded traceback storage and exact winner-context polishing.
These are CPU lifetime checks, not measured production GPU VRAM.
The full suite passes **508 tests**; Ruff and `git diff --check` also pass.

Pinned float32/float64 CPU comparisons cover ten raw/warm cases and six hybrid
fits: 8–24 mutations, 1/3 regions, CN1–6, balanced CN6, LOH, purity variation,
missing observations, and zero depth. Inputs, priors, pilots, graph/objective
identities, raw results/certificates, selected labels, fixed-refit CCFs, score
uncertainties, posteriors, qualification, and four TSVs match exactly. Four
CNA-positive hybrid fits each have eight `major_cn != minor_cn` positive-depth
rows; their public-refit CNA-only multiplicity F1 also matches. Capture hashes:

- Broader raw/warm and two hybrid checks:
  `4041b281a0e16ddbb3405e6a11d367d5080ebdf21b40b68527d39fe56ab96bbd`.
- Four CNA-positive hybrid checks:
  `52e3bf8dfcf8e1431d45dac757e890d5484ef6ddf7ed332d34d95503082dc7e4`.

For a `128 × 3 × 257 × 6` likelihood grid, CPU dispatch-visible peak live
intermediate storage decreases **14,213,376 → 11,844,864 bytes** in float32 and
**28,424,448 → 23,687,424 bytes** in float64 (about 16.7%); loss arrays match
byte-for-byte. This excludes input storage and opaque operator workspaces and
is not an end-to-end runtime or CUDA memory measurement.

Inference source remains **37 modules**, decreasing **767,700 → 752,294 bytes**
(15,406 fewer bytes, 2.01%). AST definition counts decrease from 364 to 350
top-level functions, 128 to 119 methods, and 29 to 28 nested functions. The
reachability audit removes export-only legacy helpers and test-only forwarding
wrappers after migrating their tests, not public import hooks, documented
writers, or reachable sparse/dense/resource fallbacks. Source accounting
excludes setup, tests, tools, and documentation, as in the reference measure.

These remain bounded synthetic CPU parity checks, not cohort accuracy or CUDA
qualification. Some raw smoke fits remain identically uncertified under their
small budgets; all six hybrid checks have admitted positive-lambda references.
LSF qualification is pending: the external project runbook is currently absent
and surviving historical runners expect obsolete v0.4 interfaces. Do not use
them unchanged or treat these local results as a production release gate.
Version stays **0.5.0**; no output or numerical identity schema is changed.

## Integrity qualification (`bb726ad`) — 2026-09-09

Following the review of `5ca5a1914cffc929949c4d1e22b44432a99c3b9f`, this pass
closes writable-result, mismatched-reporting-input, and selected/raw-reference
summary gaps. It also removes duplicated prepared bounds/hashes and solver
argument forwarding, consolidates preparation/retry configuration, freezes data
once, removes the stored multiplicity specification, and shares NumPy/Torch
candidate arithmetic across scalar, observed, grid, and EM evaluation. The
statistical reductions remain separate. Likelihood-only grids do not compute
posterior derivatives, and warm continuations no longer allocate discarded
copies of their primals. No graph backend or hybrid candidate family is removed.

A pinned comparison captured five tiny CPU-float64 fixtures (CN1, CN2, CN6,
equal CN6, and an independently generated two-region count fixture). The
captured numerical records match **byte for byte**: retained numerical inputs,
candidate support/priors, initialization, pilot, graph weights/hashes, objective
keys, loss/gradient/curvature, raw and warm CCFs/objectives, certificates/stop
reasons, and three hybrid selections' labels/refit CCFs/scores/posteriors/four
TSVs. The two-region fixture's CNA-only multiplicity macro/micro-F1 also matches
(10 eligible rows; a parity check, not a cohort accuracy estimate). Capture
SHA-256: `ce8e7fc88e46e00cbacd9161ad3ecf0eb3133f96029e09741092c84891e5748d`.
These checks do not establish bitwise equivalence on other inputs or devices.

In `ml1`, **437 tests passed** in 26.37 seconds, including the isolated installed
wheel and new result/reporting/copy-integrity regressions. Ruff and
`git diff --check` passed. Validation command:

```bash
env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  conda run -n ml1 python -m pytest -q -p no:cacheprovider
```

Inference source remains 37 Python files and decreases from **772,924 to
767,700 bytes** (5,224 fewer bytes, about 0.7%), excluding setup, tests, tools,
and documentation. The principal simplification is fewer independently stored
quantities and forwarding paths, not fewer module files; new integrity checks
and regression tests are retained.

Retained-input fingerprint schema changes from v2 to **v3** because the stored
candidate specification is gone; numerical model, likelihood, box, graph, and
objective-key schemas remain unchanged. Do not reuse cross-revision input
caches or identity-free saved results. Package version remains **0.5.0**, with
no new release tag. CUDA and representative cohort qualification are pending.

## Correction qualification (`5ca5a19`) — 2026-09-09

The correction-only patch following `1a0893d` addresses clipped-likelihood
global certification, explicit warm/start pairing, portable source hashing,
and manifest numerical qualification. The supplied two-mutation plateau
counterexample reproduced at lambdas 0, 0.1, and 100: objective **276.310391**
was falsely marked global despite a feasible objective of **65.016595**.
The corrected path compares the scalar pilot and reaches the better point,
without claiming whole-box global optimality. Tests cover CPU float32 and
float64, both solver routes, clipping endpoints, warm-only calls, and separate
warm/explicit starts. The objective, clipping, box, graph, and gate are unchanged.

In `ml1`, **350 tests passed** in 26.86 seconds, including the existing reference
pipeline checks, independent numerical formulas, portable-path hash checks,
and isolated installed-wheel CPU fit. Ruff and `git diff --check` passed.
Native Windows execution and production CUDA were not tested in this patch.

At this reference commit the prepared-state, data-ownership, and arithmetic
contractions remained separate work. Its corrections alone made no
benchmark-accuracy claim.

## Release 0.5.0 qualification (`1a0893d`) — 2026-09-09

In `ml1` (Python 3.13.2, NumPy 2.2.6, SciPy 1.18.0, Torch 2.9.1), all
**277 checked-in tests passed**, including the isolated wheel fit. Ruff and
`git diff --check` passed. The untouched reference passed its 152 regressions.

Paired CPU-float64 checks against
`cc5a3d1ac28097c2b3005c1c2615930f0ab424de` covered four tiny CN1/CN2/CN6
fixtures. Selected labels, refitted CCFs, Dirichlet scores, multiplicity
posteriors, four TSV schemas, and fixed-lambda objectives matched exactly.
Maximum raw-CCF difference was `3.8e-15`. Kernel, scalar-bound, initialization,
and pipeline reference values are preserved in the tests.

This is numerical equivalence, **not bitwise search equivalence**: generalized
initialization changed pilots by up to `4.2e-13` and adaptive weights by up to
`5.35e-12` in the paired probes. An adaptive graph hash and one recovery trace
changed. New fingerprint schemas invalidate old caches; within-run graph and
objective identities remain strict. CUDA/cohort qualification is pending.

Inference Python source decreased from 45 modules / 860,280 bytes to 36 modules /
759,123 bytes (about 12% fewer bytes). Moving the simulator also removes its
eight modules from the wheel; restoring tests intentionally increases the
maintained repository's test coverage rather than minimizing its file count.
