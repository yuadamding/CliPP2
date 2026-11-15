# Workflow Protocol 2.4

Stages 1–2 use read-derived multiplicity lower bounds and the Stage 2 segment
caps described below. Reference construction uses one fixed all-one clonal reference,
all distinct Stage 2 centers, and every fitted mutation reassessed under its
original multiplicity support. Native estimates, caps and producer identities
remain provenance. No confidence or distance threshold vetoes an estimate.

## Missing regional SNVs

By default, regional `snv.txt` inputs are joined over the union of all detected
SNVs. For an SNV absent in a region, set ALT to 0 and REF to that region's
median total depth (ALT+REF) among eligible, originally observed positive-depth
pairs. Canonical rows marked `count_observed=0` use the same rule, even if they
contain supplied counts. Half-read median ties round up to preserve integer
counts. Missing and excluded pairs do not contribute to the median; measured
zero-ALT and zero-depth pairs retain their original counts. A region without
any eligible observed positive-depth pair fails validation rather than borrowing
depth from another region.

This is automatic under the default `missing_counts="zero_alt"`. All stages
use the same frozen imputed counts; Stage 2 subsets never recalculate a median.
Raw source rows and the imputation ledger remain in the original catalog, and
the input audit reports regional medians and imputed-pair counts. Saved catalogs
retain their validated regional-median ledgers.
The explicit `missing_counts="unobserved"` option keeps missing pairs uninformative.

## Multiplicity support in Stages 1 and 2

For each mutation's segment in each region, use all supplied CN states to form

$$
\bar C=\sum_s f_s(A_s+B_s),\qquad \sum_s f_s=1.
$$

The fractions $f_s$ describe **all tumor cells**, including cells that do not
carry the mutation. They are supplied input; the workflow does not infer them
from a clustering fit. For a single state, its fraction is 1. Regional
`cna.txt` accepts repeated rows for an exact chromosome/start/end interval with
`cn_state_id` and `cn_state_fraction`; the optional canonical
`segment_cn_states` JSON list preserves those states on conversion. State IDs
must be unique, fractions must be finite and nonnegative and sum to 1, and
normal CN and mapping provenance must agree across the interval's states.
Distinct intervals must not overlap. See the [input example](../examples/README.md).

Regional CNA conversion uses the largest supplied `(major_cn, minor_cn)` pair,
compared in that order, as representative input CN for the existing single-state
likelihood. Canonical TSV inputs retain their explicit `allele_a_cn`/`allele_b_cn`
pair even when `segment_cn_states` contains larger states. The original input
major CN $M$ is the multiplicity ceiling. Weighted CN is used only
for the lower-bound estimate; this extension does not implement a mixed-state
likelihood or replace the likelihood denominator.

With supplied ALT $a$, total depth $d$, and purity $p$, define

$$
m_{\mathrm{raw}}=\frac{a/d}{p}\left[p\bar C+2(1-p)\right],\qquad
L=\max\left(1,\min\left[M,\lfloor m_{\mathrm{raw}}\rfloor\right]\right).
$$

This estimator assumes normal-cell total CN 2 even when the likelihood uses
a different supplied normal CN. It rounds down, has no fixed cap of 4, and
returns $L=1$ for zero ALT, missing observations, or zero depth. It does not
divide by mutation CCF: under the nominal read model, the raw estimate is
carrier CCF times carrier-weighted multiplicity. These floors are a discovery
heuristic and may underestimate subclonal multiplicity.

Stage 1 fits candidates $L,\ldots,M$. At the selected source's native CCF,
cap contributors use this actual native support and retain the complete exact
MAP set. The minimum upper endpoint across available contributors remains the
raw region–segment cap $U$. Stage 2 fits each remainder mutation over
$L,\ldots,\min(M,\max(L,U))$; without a contributor it uses $L,\ldots,M$.
If $U<L$, the sole Stage 2 candidate is $L$.
Thus a segment cap cannot remove every candidate at or above a mutation's
floor. Original read counts, purities, and likelihood CN denominators remain
fixed between the stages.

`multiplicity_lower_bounds.tsv` publishes per-pair estimates and source CN.
`mutation_segments.tsv` retains original endpoints and Stage 1 endpoints.
`stage2_segment_multiplicity_bounds.tsv` retains raw cap endpoints, lower-bound
policy, conflict policy, and override counts across eligible mapped pairs.
Accepted fits bind the actual endpoints; current fits can be reused only
when their numerical support matches the newly requested support.
Stage 3 always restores the original interval $1,\ldots,M$.

## Assignment methods

`WorkflowConfig.stage_three_assignment_method` is `"marginal"` (default) or
`"profile"`. For original major/minor CN $A_{ir},B_{ir}$, purity $\rho_{ir}$,
normal CN $C^N_{ir}$ and candidate CCF $\phi_{kr}$, let

$$
p_{irk}(m)=\frac{\rho_{ir}m\phi_{kr}}
 {(1-\rho_{ir})C^N_{ir}+\rho_{ir}(A_{ir}+B_{ir})},\qquad
\ell_{irk}(m)=\log\operatorname{Bin}(y_{ir};n_{ir},p_{irk}(m)).
$$

Supports are always $m=1,\ldots,A_{ir}$; Stage 2 caps do not apply here.
Only eligible, observed, positive-depth regions contribute. For $K$ frozen
candidates, the authoritative scores are

$$
S^{\mathrm{marginal}}_{ik}=-\log K+
\sum_r\left[\operatorname{logsumexp}_m\ell_{irk}(m)-\log A_{ir}\right],
\qquad
S^{\mathrm{profile}}_{ik}=-\log K+\sum_r\max_m\ell_{irk}(m).
$$

The native maximum already restores the uniform multiplicity factor: profile
scoring must not subtract $\log A_{ir}$ again. Every candidate is scored before
selection. Invalid native statuses fail scoring; they never become missing
evidence, impossible likelihoods or a marginal fallback. No ABI change is needed.

Marginal ties retain the lowest public label. Profile ties retain the baseline
if it belongs to the **exact computed maximizing set**, otherwise its lowest
label. No tolerance defines ties. All label aliases and selected-center indices
refer to that representative; the runner-up excludes it. `map_unique` concerns
clusters, not multiplicities. Baseline retention is not evidence of uniqueness.

At the selected center, the smallest maximizing integer represents the full
multiplicity tie set. Conditional multiplicity probabilities remain meaningful;
they are not cluster probabilities. Compatible zero-CCF observations retain
their multiplicity ambiguity and have no assigned multiplicity. Essential
enumeration failures prevent publication. Optional deviance failures leave
valid assignments and multiplicities available with an explicit diagnostic error.

## Identity, outputs and replay

`AssignmentBundle.assignment_method` is immutable and hash-bound. Execution
accepts only that method, not a separate override. A method change explicitly
derives a request, preserving observations, targets, baselines, reference and
original support, while recording its source hash and both methods.

```sh
clipp2 workflow --plan examples/workflow_protocol24.plan.json --output-dir results/workflow24
clipp2 assign --bundle results/workflow24/assignment_bundle.json --assignment-method profile --output-dir results/profile
```

Omitting `--assignment-method` uses the frozen method.
Derived requests are published as `assignment_bundle.json` inside the same
no-clobber output reservation. The summary binds the executed request and tie rule.

Both methods publish `log_scores`, `score_defined`, `assignment_method`,
`selection_reason`, exact maximizing sets, final labels and selected multiplicities.
Profile has `posterior_defined=False`; `probabilities`, `posterior`,
`best_log_probability`, `second_best_probability`, `entropy` and
`log_competitor_ratio` are null (NA in TSV). Never softmax profile scores to fill
these fields. An observed profile assignment can have a valid final label without
a posterior. Prior-only targets have equal scores but no data-supported final
label; all-impossible targets have no maximizing set or manufactured winner.

| Contract | Version |
| --- | --- |
| Protocol | 2.4 |
| Assignment bundle | v6 |
| Assignment result | v9 |
| Workflow plan / receipt | v6 |
| Paired experiment / evaluation / evaluation definition | v6 |
| Simulation study contract | v8 |

Only these current contracts are supported. Accepted current native fits can be
reused when their observations, support, fitting policy and numerical evidence
match. A Stage 3 method change reruns assignment/evaluation with those compatible
fits. Resume and aggregation distinguish method and tie-policy identities.

## Validation and comparison

Publication independently enumerates original-support binomial terms and checks
absolute scores, exact maximizing sets, representative routing, multiplicities,
posterior nullability, evidence statuses and reconciled occupancy. It also
reconstructs the winning profile score from selected maximizing multiplicities.

Profile evaluation retains accuracy, CCF error, multiplicity, occupancy, tie and
transition metrics. Cluster-posterior Brier/log-loss/calibration are unavailable.
They require marginal mode, `posterior_defined` and a valid probability vector.

`tools/compare_stage3_likelihood.py` compares frozen requests with **zero fits and
two assignment evaluations per bundle**. Lowest-label and baseline-retaining
profile choices share one score matrix. For attribution it independently verifies

$$
S^{\mathrm{marginal}}_{ik}=S^{\mathrm{profile}}_{ik}
+\sum_r\log\sum_m\exp\{\ell_{irk}(m)-\max_u\ell_{irk}(u)\}
-\sum_r\log A_{ir}.
$$

CPU tests and CUDA qualification are separate evidence. The existing general
diagnostic kernel is reused; this change alone makes no speed or accuracy claim.
