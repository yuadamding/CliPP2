# Workflow Protocol 2.3

Protocol 2.3 reports statistical estimates whenever they are computable. Confidence,
fit discrepancy, coverage and exact ties are measurements, not assignment vetoes.
The existing binomial likelihood, integer multiplicity model, free-center fitting,
candidate search, BIC and numerical verification remain unchanged. Package and
scientific protocol versions are separate identities.

## Scientific specification

1. **Stage 1:** fit eligible informative mutations with original multiplicity
   support `1..original_major_cn`, with a default maximum of 10 groups. Select the
   occupied group nearest the all-one vector by full-vector Euclidean distance.
   Break exact distance ties by larger minimum regional informative depth, then
   lower public source label. This is a **provisional reference source**; record
   its actual native center and distance. No proximity, coverage or residual gate
   qualifies the source as biologically clonal.
2. **Stage 2:** remove this source population and fit the remainder with a default
   maximum of 20 groups. Each region–segment upper multiplicity cap is the minimum
   available informative native conditional MAP call from source members.
   Contributors need genuine mapping, observed informative counts, positive native
   CCF and a defined integer MAP inside the source support. Confidence, exact ties
   and residual size do not exclude available calls. Exact multiplicity ties use
   the smallest maximizing integer, with the tie set retained. Compatible zero-CCF
   observations are uninformative for multiplicity and do not supply a cap. With no
   contributor, retain the original major-CN upper bound. The policy identity is
   `minimum_available_native_map`; original biological CN denominators never change.
3. **Stage 3:** route every provisional Stage 1 source member and all members of
   Stage 2 groups satisfying strict full-vector `||phi - 1||_2 < 0.05` to one
   reassessment. This threshold is only a Stage 2 routing parameter. Freeze one
   imposed all-one center (`center_source=fixed_all_ones`) and the native centers
   of unmerged Stage 2 groups. Restore full original multiplicity support and use
   equal cluster priors and common observation masks. Unmerged Stage 2 members
   retain their native membership. Do not add an extra native-source reference.

Unrestricted Stage 1 selection can choose a genuinely subclonal group; excluding
that group can remove a useful subclonal reference. Taking a minimum over all
available informative MAP calls can produce smaller Stage 2 caps. Both effects
are scientific outcomes to measure, not reasons to add hidden acceptance rules.

## Posterior and diagnostics

For each informative target–region pair, the candidate likelihood is the uniform
average of `Binomial(ALT; depth, a*m*phi)` over `m=1..original_major_cn`, with
`a=purity/((1-purity)*normal_cn+purity*(major_cn+minor_cn))`. Combine informative
regions and equal reference priors. Historical calls, target origin, Stage 2 caps
and reference diagnostics do not alter this score or remove a candidate.

| Evidence status | Probabilities | Data-supported final label |
|---|---|---|
| `observed` | Normalized posterior when at least one score is finite | Deterministic MAP representative |
| `prior_only` | Equal prior vector from scores `-log(K)` | Missing |
| `model_incompatible` | Missing: all informative candidate likelihoods are zero | Missing |
| `excluded` | Missing | Missing |
| Computational failure | No claim of successful scoring | Missing |

Exact maximizing log-score equality defines `map_tied_labels`; the lowest public
label is the representative. Near ties use the actual computed maximum. Preserve
finite competing log scores even when displayed probabilities underflow. A
singleton prior-only reference remains unsupported by read evidence, despite its
unique prior maximum.

Report informative counts and depths, mean and maximum profile deviance, native
and all-one group diagnostics, posterior vectors, entropy and margins. Empty
domains have count zero and missing mean/maximum; valid infinite discrepancy stays
infinite. A reporting-only diagnostic failure has its own status and error and
does not erase an independently computed posterior. Numerical or identity failures
remain explicit failures. Conditional MAP multiplicities and probabilities are
reported for informative positive-CCF observations without concentration gates;
missing, excluded and uninformative zero-CCF cases remain distinct.

## Inputs, replay and output identities

The single integer tumor-CN state domain, genuine segment mapping, explicit normal
CN or `normal_cn_policy="diploid"`, original observation masks and explicit input
exclusions remain. The existing major-CN eligibility ceiling remains for this
revision; removing it is a separate population change. Nonempty native fits still
require informative support in every requested region. Statistical coverage
threshold removal does not identify an unsupported coordinate.

New workflow plans/receipts use `plan.v4`/`workflow.v4`; assignment bundles/results
use `bundle.v4`/`result.v5`. Final labels and evidence status are authoritative;
baseline labels and provisional-source roles are separate. Whole-population
counts combine unchanged native members with observed-target MAP labels, while
prior-only, incompatible, excluded and failed cases are counted separately.

Compatible Protocol 2.2 native fits retain their original protocol, producer
receipt, accepted-state hash and numerical evidence. Append current compatibility
evidence without rewriting history. Stage 1 replay requires matching population,
configuration, model and support. Stage 2 also requires the new remainder and
effective caps to match; incompatible replay is refused, never silently reused.
Old assignment results retain their decisions. New MAP interpretations require a
new, separately identified assignment result. No native ABI change follows solely
from these Python policy changes.

## Simulation and evaluation

Simulation contract v6 uses one fixed workflow configuration. Development is a
descriptive assessment, not a threshold search or qualification gate. Poor finite
performance does not prevent main execution. Real computational failures remain
visible in planned-population accounting. Preserve all 60 subcases, 120 development
histories, 600 main histories and 2,640 acquired datasets, including supplied seeds
and acquisition pairing. Source cohort protocol identity remains distinct from
the executing protocol.

Evaluation v4 reports data-supported MAP coverage, binary and detailed subclone
error, CCF error at the selected reference, prior-only and incompatible fractions,
computation failures, unchanged-member performance and whole-original performance,
separately for each target origin. Reference existence, routing and correct
selection remain distinct. Diagnostic quality does not change truth correctness.
Historical v2/v3 evaluation definitions and inference decisions remain frozen.

The Protocol 2.2 document is historical. Native membership audits and independent
numerical verification remain mandatory; neither establishes global optimality or
biological accuracy. CUDA qualification and native cleanup remain deferred.
