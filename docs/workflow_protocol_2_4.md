# Workflow Protocol 2.4

Stages 1–2 and reference construction are unchanged from
[Protocol 2.3](workflow_protocol_2_3.md): one fixed all-one clonal reference,
all distinct Stage 2 centers, and every fitted mutation reassessed under its
original multiplicity support. Native estimates, caps and producer identities
remain provenance. No confidence or distance threshold vetoes an estimate.

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
# Explicitly migrate a historical request without rewriting it:
clipp2 assign --bundle historical/assignment_bundle.json --derive-current --assignment-method profile --output-dir results/derived-profile
```

Omitting `--assignment-method` uses the frozen method. The CLI offers no entropy
option; historical entropy experiments retain their frozen runtime and artifacts.
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

Result v7/v8 remain reserved for frozen entropy/profile comparisons. Historical
payloads retain their bytes, hashes and interpretations. Accepted native fits
from Protocols 2.2/2.3 remain explicitly reusable; a method change reruns only
assignment/evaluation when compatible fits are supplied. Resume/aggregation
must distinguish method and tie-policy identities.

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
