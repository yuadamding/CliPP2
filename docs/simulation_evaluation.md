# Simulation evaluation definitions

Implementation 2.0.0 uses scientific Protocol 2.4 and evaluation/definition v6.
The evaluator consumes retained final labels; it does not fit, reassign, apply
confidence or residual cutoffs, or select a development threshold. New contracts
declare `evaluation.record_schema` explicitly. Package version is not a substitute
for that scientific identity.

## Method-bound MAP endpoints

An informative target with a defined score has
`final_assignment_label = map_label`. Marginal ties use the lowest public label;
profile ties retain the baseline if it is a maximizer, otherwise the lowest label.
Both retain the complete exact maximizing set. Diagnostics do not veto this
result. Profile scores are not cluster posteriors: probability fields are null,
`posterior_defined=False`, and posterior Brier/log-loss/calibration are unavailable.
Conditional multiplicity probabilities remain valid at the selected center.
An eligible target with no informative reads is `prior_only`: equal scores and a
deterministic representative are available, but no data-supported final label. `model_incompatible`, `excluded`, and computational failure are
separate states, with no fabricated final label.

Metrics have explicit populations:

| Metric | Numerator or measurement | Denominator |
|---|---|---|
| `data_supported_map_coverage` | Data-supported final memberships | Original informative members in the selected view |
| `binary_map_error` | Wrong clonal/nonclonal final memberships | All original informative members in the view |
| `detailed_map_error` | Wrong mapped subclone memberships (`Mw`) | All original informative members in the view |
| `detailed_map_correct` / `detailed_map_unmapped` | `Mc` / `U` | All original informative members in the view |
| `map_ccf_rmse` | Full requested-region CCF error at the final reference | Returned labels with known truth |
| `prior_only_fraction` / `model_incompatible_fraction` / `excluded_fraction` | Explicit evidence-state counts | All members in the view, including no-read members |
| `computation_failure_count` / `diagnostic_failure_count` | Recorded failures | Counts, not error rates |
| `binary_map_correct_per_original_evaluable` | Correct observed MAP targets (all fitted members) | Original informative population |

Current views include targets (`view=""`), `whole_original`,
`stage1_reference_recheck` and `stage2_recheck`. Every fitted mutation is a target;
`unchanged_F` is explicitly empty.
Missing labels do not shrink the error denominators. Coverage and evidence-state
counts expose the absence of an estimate, rather than counting it as correct.
Binary correctness does not imply the correct subclone: a wrong nonclonal
subclone can still have the right clonal/nonclonal status. Unmapped candidate
correspondence remains `U`, separately reported rather than assumed incorrect.

`geometric_correct_map` and `geometric_correct_map_resolution` use the declared
center-distance definition without reference-quality predicates. Continuous
center errors and suitable-reference availability remain separate from routing
and selected-center correctness. Missing or failed diagnostics never become zero
residuals or passing flags. Calibration requires marginal assignment, `posterior_defined`, and observed targets with valid
probabilities; prior-only rows do not become read-supported calibration evidence.
Raw source coverage/depth/residuals and target mean/maximum residuals remain in
mutation records. Descriptive strata distinguish zero versus positive source
coverage and finite/infinite/unavailable target residuals without applying a
cutoff. No selective-error threshold curves are constructed.

The provisional Stage 1 source can be subclonal. Evaluation retains its native
centers, source purity, the learned reference geometry, cap exclusion and
whole-original performance. Upper endpoints of available native MAP sets can lower the cap;
truth replacement audits use the exact contributing IDs without requalification.

## Planned populations and missing evidence

The frozen roster supplies family, physical-history, and acquisition identities.
Each planned cell keeps weight `1 / (families × histories_in_family ×
acquisitions_in_history)`. Missing acquisitions or histories cannot disappear from
this denominator. A total planned-cell count alone cannot reconstruct the weights.

For binary correct decisions over the original evaluable mutation population,
`planned_population_estimate` uses every frozen weight. A known inference failure
or scientific rejection earns zero only when retained evidence establishes no
decisions and a positive original evaluable denominator. One equally weighted
success scoring one and one such failure therefore average to `.5`.

Missing or conflicting evaluation evidence is unknown, not a zero or an inferred
execution failure. An unknown planned cell makes the planned estimate null with
`incomplete_evidence`; an unestablished roster gives `incomplete_planned_roster`.
A separate `available_case_estimate` explicitly averages the qualified surviving
cases. Its interval is never shown as an interval for the planned cohort.
Conditional endpoints report their available-case population.

Whole-original Stage 3 correctness is the number of correct data-supported
final memberships divided by the original evaluable population. Every fitted
mutation is reassessed; the unchanged population is empty. Completed B/C results
remain available independently of a later B3/C3 assignment failure.

A truth-independent zero requires explicit `decision_population` evidence:
whole-original-evaluable scope, complete coverage, a positive established
denominator, and zero final decisions. An assignment exception alone does not
establish this claim. Complete all-unresolved target output can prove zero;
malformed or incomplete output cannot establish completed Stage 3 availability.

Records retain missingness by family, history, acquisition and arm, together with
planned, executed, scientifically applicable, evaluated and unavailable-derived
counts. Partial arm failures preserve the other arm's results. Entirely unavailable
current studies still use their frozen current evaluation definition.

Primary truth-dependent estimates require `verified_history_binding`.
Conflicting, unbound or missing truth remains unavailable. Operational execution/evaluation counts do not require biological
truth. An arbitrary `truth_required: false` cannot bypass scientific qualification.

## Operational accounting

Full evaluation and fallback evaluation use the same retained execution facts.
Removing biological truth can make correctness unknown; it does not change the
established operational state of A, B/C, or B3/C3. These fields are separate:

- `scientifically_applicable`: the analysis's scientific prerequisites are
  established. An explicit scientific rejection sets this to zero.
- `analysis_available`: a usable numerical result is retained. B/C availability
  concerns the combined memberships and reference; B3/C3 availability requires
  a complete returned target-decision axis.
- `execution_failure`: an execution operation belonging to that analysis failed.
  `analysis_failure` is an alias of this indicator, not the complement of
  availability.
- `execution_available`: retained evidence establishes that the analysis was
  reached or its scientific applicability was assessed. An explicitly unexecuted
  analysis, including one prevented by an upstream exception, has value zero.
- `evaluation_available`: the scientific evaluation was derived. This can differ
  between full evaluation and a placeholder with unavailable truth.

| Retained C/C3 evidence | C available | C failure | C3 available | C3 failure | C3 applicable |
|---|---:|---:|---:|---:|---:|
| C complete; assignment fails | 1 | 0 | 0 | 1 | 1 |
| C complete; assignment scored; publication fails | 1 | 0 | 1 | 1 | 1 |
| Explicit rejection before Stage 3, without an exception | As established | 0 | 0 | 0 | 0 |

Stage 2 fitting, accepted-fit publication, classification and reference freezing
belong to B/C. Assignment, reconciliation and assignment publication belong to
B3/C3. Missing execution facts or an unrecognized failure phase leave the affected
quantities null; a retained output alone does not identify an unknown exception.
An upstream exception does not become a downstream execution failure. Its explicit
pre-decision evidence can still establish a whole-original empty decision population
under the scoped-zero rules above.

## Detailed membership

Frozen source-member purity still maps candidates to observable truth classes
at the existing `.8` threshold, with the designated clonal reference mapped to
clonal truth. This mapping is distinct from geometric reference recovery.
Accepted targets divide into correct mapped `Mc`, wrong mapped `Mw`, and
unmapped or ambiguous `U`; unresolved targets are separate. Evaluation records
retain these counts for targets, both target origins, unchanged `F`, and the
whole original population.

Partition agreement, pair precision/recall, center errors, multiplicity errors,
and coverage remain available.

Baseline and final membership use the same four-state classifier. Per-mutation
records contain `baseline_detailed_membership_status`,
`detailed_membership_status`, and `detailed_membership_transition`. A 4×4 cross-tab
is retained separately for each target origin. `Mw->U` is unmapped acceptance,
not a persisted mapped error; `U->Mc` is newly mapped-correct without proof that
the baseline was wrong. Binary transitions separately describe clonal/nonclonal changes.

## Reference geometry

New declared assessments use full requested-region, unnormalized L2 distance
`||truth-center - candidate-center||2 <= tolerance`. The current production workflow has no geometric routing cutoff. This optional
reporting tolerance changes neither reference inventory nor assignment.
A new evaluation configuration is:

```json
{
  "schema": "clipp2.simulation.evaluation-config.v1",
  "reference_match_atol": 0.05,
  "reference_distance": "l2"
}
```

`reference_match_atol: null` means unassessed. Explicit
`reference_distance: "linf"` identifies maximum-coordinate geometry when requested.

One-to-one reference matching first maximizes the number of allowed pairs, then
minimizes distance summed over those pairs. Invalid edges never contribute cost.
Stored float distances are compared as exact binary rationals, without a tie
tolerance. Equal-cost optima use a deterministic lexicographic truth/reference order;
alternative optimal choices are retained as ambiguity evidence. Continuous L2
and maximum-coordinate errors remain available without a tolerance. Controlled reference diagnostics use the same declared matcher.

No declared tolerance gives null recovery with an unassessed status. Missing
references, observations or diagnostics retain explicit unavailable states.
Geometric availability never changes the production reference inventory or
vetoes a defined assignment.

## Offline reports from current artifacts

```sh
clipp2-simulate report STUDY_DIRECTORY \
  --evaluation-config evaluation.json \
  --report-dir NEW_REPORT_DIRECTORY
```

Offline reporting reads retained truth, input counts, acquisition evidence,
selected attempts and paired results, including archived results. It performs no
fitting, reassignment, truth generation or read generation. A report never changes
saved inference decisions or overwrites the source study.

Only current artifacts are supported: paired experiment v6, original catalog v6,
assignment bundle v6, assignment result v9 and evaluation/definition v6. Input
catalogs must agree with their manifests, source observations, masks, CN,
metadata and imputation ledgers. Missing or conflicting evidence is reported as
unavailable. Directory and archived copies must agree when both remain, and
each required filename can occur only once within an archive.

Without a new evaluation configuration, reports use cached evaluations under
the study's frozen definition. Every cached record must match that definition
and its cell identity. Structural or identity errors name the cell and file;
definition conflicts require explicit reevaluation into a new directory.
Missing or truncated files remain unavailable evidence.

Evaluation, definition and aggregate records bind the endpoint, population,
missing-evidence treatment, truth qualification, assignment and fitting methods,
distance metric, tolerance and matching policy. Aggregation rejects incompatible
definitions, including different methods, distances or tolerances.

## Consistent operational accounting

The aggregate owns one per-arm accounting table for execution, scientific
applicability, and evaluation availability. Each indicator separately counts
true, false, and unknown; those counts sum to the retained roster size. The HTML
population table and population figure consume that same table. Unknown
execution remains distinct from an explicitly unexecuted analysis.

Per-arm unavailable evaluation means `evaluation_available == 0`. The count of
whole records labeled `unavailable_derived_evidence` is a separate study-level
quantity. Execution still means an analysis was reached or its applicability
was assessed; native fit and assignment costs come from the operation ledger.
