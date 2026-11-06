# Workflow Protocol 2.2

This note describes the implemented workflow and versioned outputs. Numerical
correctness and biological calibration remain separate claims.

## Stages and decisions

1. **Stage 1:** fit eligible informative mutations with full original single-state
   multiplicity support `1..original_major_cn`. The default cluster limit is 10.
2. **Stage 2:** remove the selected clonal reference members, then fit the remainder
   using segment upper caps from qualified reference contributors. Each cap is the
   minimum qualifying call; absent contributors retain the original upper bound.
   The default cluster limit is 20. Biological CN and denominators stay unchanged.
3. **Stage 3:** route every historical Stage 1 clonal member and every member of
   a geometrically near-clonal Stage 2 group to one reassessment. Freeze one
   all-one center and the native centers of unmerged Stage 2 groups, using equal
   cluster priors and fresh full original support
   `1..original_major_cn`. Stage 2 caps do not restrict this calculation. Unmerged
   Stage 2 members and their native centers remain unchanged; there are no donor
   floors or restricted/original paired assignments.

Let `T` be these targets and `F` the unmerged Stage 2 members. They partition the
accepted baseline. With `Q2` Stage 2 groups and `M` near-clonal groups, the frozen
reference has `K = 1 + Q2 - M` candidates with prior `1/K`. For each informative
target-region pair its emission is
`sum(Binomial(ALT; depth, a*m*phi), m=1..major_cn) / major_cn`, where
`a = purity / ((1-purity)*normal_cn + purity*(major_cn+minor_cn))`.
Every candidate uses the same target observation mask. Source calls, caps and
target origin do not enter this score. A useful subclonal reference must have
been discovered before reassessment can select it.

Clonal proximity uses the complete regional vector and strict Euclidean distance
`||phi - 1||_2 < 0.05`. Strict Stage 1 qualification additionally requires regional
coverage, native-center screens and all-one screens. Stage 2 routing uses geometry
alone: failed coverage or capped/all-one screens never veto reassessment.
An entirely unsupported requested region still blocks a nonempty native fit.
For unmerged references, regional quality requires source coverage and the native
center screen under its accepted Stage 2 support. A confident winner failing
quality in any informative target region yields `unresolved_reference`, preserving
all scores and probabilities. There is no fallback to a runner-up. Target screens
aggregate informative regions instead.
Probability, discrepancy, tie, and multiplicity-concentration cutoffs are explicit.
A confidence threshold of one rejects any finite competing likelihood, even when
the displayed probability rounds to one. Native fits retain mandatory membership
audits and independent numerical verification; these do not certify global optimality.

Version 1.2.2 computes profile deviance directly at the native conditional MAP
probability with a stable binomial deviance formula. Exactly saturated components
have zero deviance; adjacent representable probabilities retain positive loss.
No small-value threshold is applied. This can change contributor or screen
decisions at zero cutoffs; offline evaluation of old results preserves their
original inference decisions. Diagnostic reuse compares effective numerical
support after excluded pairs have been projected to inert endpoints.

The three reassessment outcomes (`retained_clonal`, `reassigned`, `unresolved`)
apply to workflow Stage 1/2 targets with a designated clonal baseline. Generic
fixed-reference requests use `not_applicable`; their `baseline_changed` remains
available for accepted assignments. Errors during cap qualification, construction,
or publication belong to Stage 2 and preserve Stage 1 acceptance.

Strict workflow evidence requires genuine, unambiguous single-state CN. Missing
or excluded observations contribute no read evidence. Normal CN must be supplied,
or the plan must explicitly select `"normal_cn_policy": "diploid"` for assumed
normal CN equal to two. The catalog records whether normal CN was supplied.

## Runnable example

The [six-mutation, two-region table](../examples/workflow_protocol22.tsv) and
[plan](../examples/workflow_protocol22.plan.json) exercise orchestration with
single-state diploid CN and supplied normal CN. Their explicit thresholds are
fixture settings, not biological calibration. Amplified-CN recovery controls
remain separate in the numerical checks.

```sh
clipp2 workflow --plan examples/workflow_protocol22.plan.json \
  --output-dir results/workflow22
clipp2 assign --bundle results/workflow22/assignment_bundle.json \
  --output-dir results/workflow22-replayed-assignment
```

Paths in a plan are relative to that plan. Choose new output directories. The
complete configuration is in the example plan; no unspecified confidence or
residual threshold is inferred from the data.

## Identity and outputs

The following schema names require explicit Protocol 2.2 evidence where
applicable; schema names alone do not establish compatibility:

| Record | Schema |
|---|---|
| Accepted biological input | `clipp2.accepted-biological-input.v2` |
| Accepted state / fit | `clipp2.accepted-state.v3` / `clipp2.accepted-fit.v3` |
| Producer / replay compatibility | `clipp2.fit-producer.v2` / `clipp2.fit-compatibility.v2` |
| Assignment bundle / result | `clipp2.assignment.bundle.v3` / `clipp2.assignment.result.v4` |
| Workflow plan / receipt | `clipp2.workflow.plan.v3` / `clipp2.workflow.v3` |

Original catalogs retain their v3 schema. Paired or missing-origin records are
rejected, not relabeled or implicitly refitted. Rebuild derived references and
bundles from valid reusable source fits, or rerun original inputs. Frozen 2.1
installations remain necessary for interpreting their earlier outputs.

`AcceptedFit.tumor_id` is required and keyword-only. Replay binds the original
tumor, full region mapping, mutation-key convention, declared reference build,
and numerical evidence. It retains the originating producer and appends current
compatibility evidence. A missing original identity is never filled from the
destination. Copies and serialization reconstruct immutable arrays and identities.

External execution provenance is optional and explicit: set plan/API
`run_manifest_id` or `CLIPP2_RUN_MANIFEST_ID`. A supplied `run_build_manifest`
retains the external build record. Originating and current replay/assignment
links remain distinct; absent external provenance is explicitly unbound.
Current links resolve from an explicit argument, then the current environment,
then null; assignment never adopts a bundle's historical link. Producer replay
requires every original fit configuration field and structurally valid native
and continuation routes even when external build provenance is absent.
Inference does not scan or snapshot source directories or gather an environment
manifest. File-byte checksums are absent; biological/content identities and
native build compatibility checks remain.

Publication manifests declare `clipp2.file-list.v1`; `files` is a list of
filenames, not a digest map. Ordinary native table output uses schema version 14.

`source_to_reference.tsv` describes construction only. Every original mutation
has separate `baseline_reference_label` and authoritative `final_assignment_label`
fields. Unresolved targets have no final label. A zero-occupancy reference remains
in the frozen inventory. Workflow `final_accounting` reconciles unchanged members
plus accepted targets for each label, together with unresolved targets, and
reports outcomes separately for both target origins. Historical cap contributors
that move receive `historical_cap_contributor_reassigned`; caps stay unchanged.
Standalone assignment publishes target-only counts. The workflow scores, reconciles
T/F, writes whole-original membership, then publishes combined accounting once.

Stage 2 evidence is retained in `stage2_reference_call_qualification.tsv` and
`stage2_segment_multiplicity_bounds.tsv`. `stage_one_accepted.json` and
`stage_two_accepted.json` preserve native states. `assignment_bundle.json` freezes
original support and reference quality. `all_original_mutations.tsv` records
`membership_origin` and authoritative labels. The `assignment/` directory has
one target-probability table, one regional multiplicity table and one summary.
The valid-empty Stage 2 branch uses `T=G0`, empty `F`, and one clonal candidate.
Every reference explicitly supplies its regional quality matrix, regional
quality reasons, and center-source annotations; omitted quality is not approval.

One assignment is compared with the combined baseline: `retained_clonal`,
`reassigned`, or `unresolved`; unmerged members are `unchanged_non_target`.
Descriptive screen results are separate from `fit_status="not_tested"` when no
calibrated test was run. Blocked Stage 2 reports candidate count separately from
accepted `N2=0`. Later diagnostic/publication failures preserve already accepted
Stage 1/2 statuses and counts, with `current_phase`/`failed_phase` identifying the
derived operation. Unsupported `fit_assessment` requests (including bootstrap)
fail before execution.

Decision order is `filtered`, `no_information`, `incompatible`, `ambiguous_tie`,
`low_confidence`, `unresolved_reference`, `unresolved_fit`, `assigned_relative`.
For a singleton reference, informative compatible targets have probability one
and entropy zero, still subject to quality and fit acceptance;
`comparison_scope="single_candidate"` makes that scope explicit. Conditional
multiplicity evidence remains diagnostic until membership is accepted.
`assigned_multiplicity` is missing for unresolved membership and zero CCF. At
zero CCF and zero ALT, the uniform conditional distribution remains reported as
`uninformative_zero_ccf`. The explicit multiplicity-concentration threshold is
separate from membership and cap-contributor confidence.

## Run and check

Use one installed wheel or editable package, then run from the repository root.
Choose new output paths; existing result namespaces are not overwritten.

```sh
python tools/check_workflow.py --device cpu --output results/contracts.json
python tools/qualify_workflow.py --device cpu --output results/qualification.json --replicates 3 \
  --posterior-min .9 --mean-deviance-max 5 --pair-deviance-max 10 \
  --outlier-fraction-max .05 --tie-atol 1e-10 \
  --multiplicity-tie-atol 1e-10 --multiplicity-probability-min .95
```

The CPU checks cover numerical, input, replay, publication, and workflow contracts.
Qualification executes real `run_workflow()` discovery/routing/reconciliation and
reports correct accepted, wrong accepted, and unresolved targets by origin,
unchanged `F`, whole-population membership errors, and routing coverage.
Continuous reference-center L2 and maximum-coordinate errors are always reported.

`--reference-match-atol` defaults to `None`: binary reference recovery/availability
is then null and explicitly unassessed. A supplied tolerance uses unnormalized full-vector L2
error and one-to-one matching, so one candidate cannot recover several
true groups. Member overlap measures partition agreement, not center recovery.
`--compare-stage-two-support` adds a separate offline capped/full-support comparison;
it never adds a production fit. Small synthetic qualification does not establish
population-level biological improvement. CUDA requires its own built artifact and
execution; these commands exercise CPU.
