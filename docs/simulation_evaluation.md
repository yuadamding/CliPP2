# Simulation evaluation definitions

Implementation 1.1.0 keeps scientific Protocol 2.2 and the frozen development
objective. Evaluation v3 preserves planned populations, requires qualified truth
for primary science, and uses consistent detailed membership transitions.
It does not refit, reassign, select thresholds, or change historical caps.
New contracts declare `evaluation.record_schema` explicitly, so the package
version does not determine the evaluation schema. Historical contracts without
that field retain their original version-based interpretation.

The existing `correct_resolution` and `selective_error` fields remain binary
clonality endpoints, with explicit `binary_*` aliases. An accepted wrong subclone
can have the correct clonal/nonclonal status. The development constraint of
empirical error at most `.05` is neither a population error guarantee nor a
bound on detailed subclone errors.

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
Conditional endpoints, including mapped selective error and geometric acceptance
within the selected target population, report their available-case population.

Whole-original Stage 3 correctness includes both reassessed targets `T` and
unchanged members `F`: `(correct_F + correct_accepted_T) / original_evaluable`.
Failure before target decisions establishes `correct_accepted_T = 0`; it does
not erase retained F memberships. With valid truth their exact contribution
remains. With missing truth their correctness is unknown, so the whole-population
score stays null. Completed B/C combined results remain available independently
of a later B3/C3 assignment failure.

A truth-independent zero now carries explicit `decision_population` evidence:
whole-original-evaluable scope, complete coverage, the established denominator,
and zero accepted decisions. Retained Stage 3 evidence separately accounts for
unchanged and accepted-target memberships. Bare historical `known_no_decisions`
tags, missing F evidence, and assignment exceptions alone cannot establish this
claim. Re-evaluation must reconstruct it from retained memberships or an explicit
failure before any endpoint decisions. Complete all-unresolved target output with
an explicitly empty F also proves zero; malformed or incomplete returned output
does not establish completed Stage 3 analysis availability.

Records retain missingness by family, history, acquisition and arm, together with
planned, executed, scientifically applicable, evaluated and unavailable-derived
counts. Partial arm failures preserve the other arm's results. Entirely unavailable
current studies still use their frozen current evaluation definition.

Primary truth-dependent estimates require `verified_history_binding`. Reconciled
but unbound historical truth is `legacy_consistent_unbound`, reported only as
`legacy_consistent_available_case_estimate`. Conflicting or missing truth remains
unavailable. Operational execution/evaluation counts do not require biological
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

For evaluable population size `P` and accepted count `A`:

- Mapped correct and wrong resolutions are `Mc/P` and `Mw/P`.
- Mapped selective error is `Mw/(Mc+Mw)`.
- Unmapped acceptance is `U/A`, also reported as `U/P`.
- Detailed selective-error bounds are `Mw/A` and `(Mw+U)/A`. They express
  uncertainty from unresolved candidate mapping, not confidence intervals.

Partition agreement, pair precision/recall, center errors, multiplicity errors,
and coverage remain available. The unchanged-F view makes errors outside the
reassessment population visible.

Baseline and final membership use the same four-state classifier. Per-mutation
records contain `baseline_detailed_membership_status`,
`detailed_membership_status`, and `detailed_membership_transition`. A 4×4 cross-tab
is retained separately for each target origin. `Mw->U` is unmapped acceptance,
not a persisted mapped error; `U->Mc` is newly mapped-correct without proof that
the baseline was wrong. Binary transitions are unchanged. The old diagnostic is
explicitly named `legacy_multiclass_transition_fraction_unmapped_as_error`.

## Geometry and availability

New declared assessments use full requested-region, unnormalized L2 distance
`||truth-center - candidate-center||2 <= tolerance`. Production near-clonal routing
retains its separate strict L2 rule. No reporting tolerance is implied by routing.
A new evaluation configuration is:

```json
{
  "schema": "clipp2.simulation.evaluation-config.v1",
  "reference_match_atol": 0.05,
  "reference_distance": "l2"
}
```

`reference_match_atol: null` means unassessed. Explicit `reference_distance:
"linf"` identifies maximum-coordinate geometry when requested; cached evaluation
v1 records retain their original L∞ and legacy matching definition.

One-to-one reference matching first maximizes the number of allowed pairs, then
minimizes distance summed over those pairs. Invalid edges never contribute cost.
Stored float distances are compared as exact binary rationals, without a tie
tolerance. Equal-cost optima use a deterministic lexicographic truth/reference order;
alternative optimal choices are retained as ambiguity evidence. Continuous L2
and maximum-coordinate errors remain available without a tolerance. The workflow
qualification tool uses the same declared matcher.

Each target and unchanged-F row reports whether any geometrically matching
reference exists, and whether at least one such reference passes quality in all
of that mutation's original informative regions. Distance always uses the full
region vector; only the quality requirement follows the observation mask.
Quality failure in an unobserved region does not make a candidate unusable.

No declared tolerance gives null availability with an unassessed status. No
frozen reference gives null with `unavailable_reference`. No informative
observations gives null usable availability with a not-evaluable status. A valid
reference without a matching center gives false; matching centers that fail
required quality give geometric true and usable false. Missing historical quality
evidence gives null usable availability rather than invented passing flags.

`geometric_correct_reference_resolution` counts accepted matching centers that
pass required quality over the evaluable population. Outcomes are also stratified
by availability and target origin. The binary recovery ceiling remains
`|H intersect T_C|/|H|`; geometric availability never multiplies that binary ceiling.

## Explicit offline reports

```sh
clipp2-simulate report OLD_STUDY \
  --evaluation-config evaluation-v3.json \
  --report-dir NEW_REPORT_DIRECTORY
```

This reads retained truth, input counts, acquisition evidence, selected attempts,
and paired results, including archived results. Missing or ambiguous evidence is
reported as unavailable. Truth and counts are not regenerated from a changed
source path. Frozen development selection and original outputs stay intact.
Verified single-state `original-catalog.v3` evidence from `f2d1fa1` can be
reevaluated read-only. The retained catalog payload is checked against its original
manifest and its original content-hash encoding before conversion. Its current
working representation must then agree with the preserved input's scientific
arrays, mutation/region keys, masks, source rows, metadata and provenance.
Derived records state both `source_catalog_schema` and `working_catalog_schema`;
neither original catalogs nor manifests are rewritten. Directory and archived
configuration copies must agree when both remain. Each required filename may
occur only once within an archive; even identical duplicate members are rejected.

Present but unsupported formats receive `unsupported_artifact_format` and pinned
producer guidance, not `conflicting_truth`. This includes older mixed-state
catalogs, multiple historical CN rows for one observation, and historical
near-one source fractions that the current exact single-state contract cannot
represent unchanged. Such evidence is excluded from primary truth-dependent
estimates. Altered counts, CN, keys, masks, provenance or identity bindings remain
conflicts. Absent manifest or input-binding keys are missing evidence; present
malformed bindings remain conflicts. Current-format evidence
without a redundant catalog copy is usable only if the preserved input exactly
matches the original manifest identity.

The compact `f2d1fa1_successful_evidence.zip` regression fixture preserves an
actual 12-mutation, two-region historical run with three native fits and two
assignment passes. Each arm has seven unchanged members, five accepted targets,
and nonzero calibration losses. Its archived origin receipt and reproducer record
the producing revision, environment, configuration and expected results. Cached
reporting and current reevaluation compare all 853 metrics under the same declared
evaluation definition, without inference or source changes. The separate failure
fixture retains coverage of pre-fit failure accounting.

Without a new evaluation configuration, cached-report behavior is retained.
Cached reports resolve the complete frozen definition before reading cells.
Every cached record must match that definition, even when all records are present
and agree with each other. A conflicting cache requires explicit reevaluation.
Syntactically valid caches must be objects with the expected cell identity,
versioned definition, and a list of metric objects. Structural or identity errors
raise an integrity error naming the cell and file; scientific-definition conflicts
remain fatal. Missing or truncated files retain explicit unavailable placeholders.
Missing records remain unknown under that definition, including historical v1
L∞ geometry, its top-level matching tolerance, and its original matching policy.
Even an empty report with no recoverable complete roster retains the historical
definition and unavailable planned estimates. An explicit new evaluation
configuration requests v3 derivation, including when usable records are absent.

Cached operational values remain historical. To apply corrected stage-specific
accounting, derive a report into a new directory from retained execution evidence;
the source evaluations and reports are preserved.

Evaluation, definition and aggregate v3 records contain the endpoint, population
scope, missing-evidence treatment, truth-qualification policy, metric, tolerance,
matching policy, and semantic definition identity. Distance configuration remains
v1; old v1/v2 evaluation files keep their original definitions. Aggregation
rejects incompatible definitions rather than silently mixing L2, L∞, different
tolerances, or legacy and revised endpoint interpretations.

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
