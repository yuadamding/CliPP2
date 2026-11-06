# Simulation evaluation definitions

Implementation 1.2.2 keeps scientific Protocol 2.2 and the frozen development
objective. Evaluation v2 adds separate detailed-membership and geometric endpoints.
It does not refit, reassign, select thresholds, or change historical caps.

The existing `correct_resolution` and `selective_error` fields remain binary
clonality endpoints, with explicit `binary_*` aliases. An accepted wrong subclone
can have the correct clonal/nonclonal status. The development constraint of
empirical error at most `.05` is neither a population error guarantee nor a
bound on detailed subclone errors.

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
  --evaluation-config evaluation-v2.json \
  --report-dir NEW_REPORT_DIRECTORY
```

This reads retained truth, input counts, acquisition evidence, selected attempts,
and paired results, including archived results. Missing or ambiguous evidence is
reported as unavailable. Truth and counts are not regenerated from a changed
source path. Frozen development selection and original outputs stay intact.
Without a new evaluation configuration, cached-report behavior is retained.

Evaluation and aggregate v2 records contain the complete endpoint definition,
metric, tolerance, matching policy, and a semantic definition identity. Aggregation
rejects incompatible definitions rather than silently mixing L2, L∞, different
tolerances, or legacy and revised endpoint interpretations.
