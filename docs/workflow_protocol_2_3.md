# Workflow Protocol 2.3

Historical contract. Use its frozen runtime for the comparison options below;
the current implementation follows [Protocol 2.4](workflow_protocol_2_4.md).

Protocol 2.3 reports mathematically defined estimates without confidence,
residual, coverage or distance acceptance thresholds. Numerical verification,
model-domain validation, explicit input exclusions and biological identity checks
remain. The likelihood, BIC, native candidate search and computational controls
are unchanged. Package version and scientific protocol version are separate.

## Population and discovery

Valid integer major CN has no automatic upper eligibility cutoff. The default
`max_major_cn=null` means unrestricted valid CN; an explicitly requested positive
integer retains a separately identified restricted population for historical
comparisons. Native integer, allocation and resource limits remain explicit
computational limits, never silent support truncation. Missing required CN or
purity is an input problem, not a low-confidence estimate. `clipp2 audit-input`
reports missing profiles, regional/segment mapping problems and invalid values
without filling them or running inference.

### Missing-read policy

The default `missing_counts="zero_alt"` resolves eligible `count_observed=0`
pairs (including SNVs absent from a regional list) to ALT=0. Use a positive
supplied total depth (ALT+REF, or REF when ALT is absent); otherwise use that
region's median positive depth among eligible, originally observed pairs, with
half-read medians rounded up. ALT alone cannot supply total depth. If neither
depth source exists, validation fails. CN-filtered/input-excluded pairs and
explicitly observed zero-depth pairs are not imputed or used as median donors.

This is an explicit absence assumption, not proof that unreported SNVs were
measured at zero ALT. Original raw rows and the per-pair depth/source ledger are
retained in catalog v6; all three stages use the same effective counts. Stage 2
subsets reuse these frozen depths, never a newly computed subset median. The
workflow receipt and input audit summarize imputation. Set the workflow config's
`missing_counts="unobserved"` (or standalone `--missing-counts unobserved`) to
retain the prior no-evidence interpretation. Frozen v4/v5 catalogs are read
unchanged, without new imputation.

1. **Stage 1:** fit eligible informative mutations under original support
   `1..original_major_cn`, allowing up to 10 groups by default. Select an occupied
   source by smaller full-vector L2 distance to all ones, then larger minimum
   regional informative depth, then lower public source label. This is a
   **provisional computational source**, not a declaration of biological clonality.
   Its actual native center remains provenance, not a second clonal candidate.
2. **Stage 2:** fit the remaining population, allowing up to 20 groups by default.
   At each mapped source observation, calculate component likelihoods under
   original support and the native source CCF. Preserve the full exact maximizing
   multiplicity set and historical native point call separately. Each defined
   observed contributor supplies the **largest exact maximizing integer**. A
   region–segment cap is the minimum of these endpoints, with original-support
   fallback when no endpoint is available. The policy is
    `minimum_native_map_set_upper`. Unobserved pairs do not supply endpoints;
   all-impossible component likelihoods are undefined, never uniform. A flat
   likelihood supplies the original upper endpoint and cannot restrict support
   alone. In particular, a zero-CCF/zero-ALT observation has no multiplicity read
   information; its flat set is reported as such, without using its arbitrary
   display point call to restrict support. Original CN denominators never change.
3. **Frozen references:** use one canonical all-one clonal hypothesis and every
   native Stage 2 center. Map the selected Stage 1 source to clonal label 0;
   do not add its fitted center as a competing reference. Preserve its native
   center, fitted-state identity and diagnostics as provenance. Consolidate
   exactly equal Stage 2 vectors, including exact all-one vectors into label 0.
   Assign equal priors **after** consolidation. Distinct Stage 2 centers remain
   unchanged; no distance threshold merges genuine nearby subclones.
4. **Stage 3:** reassess every mutation in the combined fitted population once:
   `T = G0 union Stage2`, `F = empty`. Restore full original multiplicity support
   and use common observation masks. There is no geometric merging or selective
   reassessment rule. Stage 1 source members have baseline label 0; Stage 2
   members map to their native center after exact consolidation. Native baseline
   CCFs and multiplicity calls remain unchanged in provenance. MAP assignment
   may move source members away from label 0; occupancy is not forced.

Input and CN-population exclusions take precedence over Stage 2 cap-source
support checks and contributor classification. An excluded pair may have inert
numerical support `[1,1]` even when its retained catalog bounds are wider; this
is not a restriction of biological evidence. It remains marked `excluded`, is
not scored for a cap, and supplies no endpoint. Eligible pairs must still match
their full original support, including eligible pairs with missing reads.
Original CN and exclusion provenance remain unchanged. Read masks are resolved
once by the declared missing-read policy before Stage 1 and remain fixed thereafter.

A nonempty fitting population with an entirely unsupported requested region
remains explicitly unidentifiable. Removing an acceptance rule does not identify
an absent coordinate. An empty Stage 2 remainder is valid: reassess the source
against the single all-one hypothesis.

The cap is a discovery heuristic, not a biological upper-bound certificate. One
uncertain low MAP endpoint can reduce it and exclude another mutation's preferred
multiplicity. Restoring support in Stage 3 does not refit discovery centers.
Replacing a selected source's fitted center with the all-one reference is an
explicit clonal-source modeling assumption. If that source is genuinely
subclonal, its native explanation is no longer a Stage 3 candidate; provenance
retains that fact. The rule does not guarantee biological recovery.

## Likelihood, estimates and information states

For candidate `k`, mutation `i` and informative region `r`, use

```
a = purity / ((1-purity)*normal_cn + purity*(major_cn+minor_cn))
E[i,k,r] = mean(Binomial(ALT; depth, a*m*phi[k,r]), m=1..major_cn)
log_score[i,k] = -log(K) + sum(log(E[i,k,r]), informative r)
```

Normalize in log space. Exact computed log-score equality defines `map_set`;
`map_label` is its smallest public label. Near ties retain their actual maximum
and score margin. `posterior_defined` states whether the probability vector
exists. `map_cluster_label` and `map_tied_labels` remain exact compatibility
aliases. Reference diagnostics do not alter scores, masks, priors or labels.

| Evidence status | Probabilities | Data-supported final label |
|---|---|---|
| `observed` | Normalized posterior when any candidate score is finite | MAP representative |
| `prior_only` | Equal priors from the empty likelihood product | Missing |
| `model_incompatible` | Undefined: all candidates have zero likelihood | Missing |
| `computation_failed` | No successful posterior claimed; failure evidence retained | Missing |
| `input_missing` | Required biological input absent; input audit retained | Missing |
| `excluded` | Explicit exclusion and its provenance retained | Missing |

Invalid model inputs are reported separately as input errors. Core scoring
failures do not publish fabricated assignment results. Valid low probabilities,
poor residuals and finite likelihoods whose displayed probabilities underflow
remain valid measurements. A singleton prior-only vector has no read evidence.
An observed VAF above a candidate's maximum attainable mean is not, by itself,
a zero-likelihood or invalid-input condition.

Report the complete posterior, maximizing set, entropy, margins, and conditional
multiplicity candidates/probabilities/ties. Counts, depths and mean/maximum
profile deviance describe source information. Empty diagnostic domains have
count zero and missing residual summaries; valid infinite discrepancy remains
infinite. Diagnostic failures remain separate from independently valid scoring.
A count-aware floating-point accumulation bound validates mean/max consistency
without modifying raw measurements.

Exact duplicate native centers have disjoint source memberships. Their regional
counts/depths are summed, means are count-weighted and maxima are retained; all
individual source diagnostics remain in provenance. A failed informative source
diagnostic makes that aggregate unavailable, not the estimate. The selected
Stage 1 source contributes diagnostics evaluated at the fixed all-one vector,
not at its replaced native center; both measurements remain in provenance.

Whole-original output includes final MAP labels for fitted targets, explicit
exclusions, and prior-only outcomes for otherwise eligible unobserved mutations.
Those outside the fitted population do not enter fitted-target occupancy counts.
Historical cap contribution remains attached to the discovery fit even if the
mutation later moves away from its native baseline reference.

## Contracts and historical replay

The reference-preserving revision uses workflow plan/receipt **v5**, assignment
bundle/result **v5/v6**, original catalog **v6**, paired experiment **v5**,
simulation contract **v7**, and evaluation/definition **v5**. These versions
separate it from the earlier Protocol 2.3 routing proposal and Protocol 2.2.
The explicit policy `single_fixed_clonal_stage_one_source_v1` distinguishes the
single-clonal-reference revision from the preceding source-preserving inventory;
historical records and fixed-reference replay retain their bound inventories.
Historical records retain their original interpretations; the protocol string
alone never selects an evaluation definition.

Compatible native fits keep their original protocol, producer, accepted-state
identity and verification evidence. Current compatibility is appended. Stage 1
requires unchanged population, biological observations, masks, supports and
native fitting settings. Stage 2 also requires unchanged remainder and effective
caps. A CN-policy label change alone may be compatible only if the complete
actual eligibility mask and input content are identical. Expanded populations
require new fits. Historical catalog v4/v5 readers preserve the original payload.
Changing missing-read policy or effective counts also requires new fits.

Historical assignment artifacts are read-only. To score an explicitly identified
current-policy derivative of a frozen bundle, use `clipp2 assign --derive-current`.
This preserves its supplied reference inventory and target scope; it is a
reporting-only comparison, not a reconstruction of the revised workflow.
Independent publication checks still verify scores, normalization, identities,
supports, immutable content and final population accounting.

### Optional entropy comparison

`clipp2 assign --bundle BUNDLE.json --assignment-rule minimum_conditional_entropy
--output-dir NEW_OUTPUT` reuses the fixed references and original support without
refitting Stage 1/2. The default workflow and assignment rule remain likelihood MAP.
For each candidate, normalize its binomial masses over multiplicities, then sum
`-sum(p * log(p))` across informative regions (nats). Choose the smallest entropy
among candidates with finite joint likelihood. Exact entropy ties use higher
likelihood, then lower public label; there is no fit-quality threshold. Impossible
distributions are ineligible, not zero-entropy; missing regions contribute nothing,
and mutations without informative regions receive no final label. A zero-CCF,
zero-ALT region has uniform multiplicity probabilities and entropy `log(major_cn)`.

This comparison emits result **v7** from the unchanged v5 bundle; normal MAP results
remain v6. `map_label`, `best_label`, `posterior`, probability vectors and `entropy`
retain their **membership-MAP** meanings. The new conditional-multiplicity entropy
vector, selected entropy/probability/log score and `final_assignment_label` describe
the alternative decision. Regional `selected_label`, CCF, multiplicity and residuals
use that selected reference, which need not be the likelihood MAP. Low conditional
entropy measures certainty about multiplicity **given a reference**, not reference
correctness: it can favor confident but extremely poor-fitting explanations.
Independent publication checks verify both the original probabilities and the
alternative selection; comparison outputs must not overwrite the original workflow.

### Optional profile-likelihood comparison

`clipp2 assign --bundle BUNDLE.json --assignment-rule profile_likelihood
--output-dir NEW_OUTPUT` instead maximizes the product of regional maximum
binomial likelihoods. For each informative region and fixed reference, take
`max(Binomial(ALT; depth, p(m, phi)), m=1..original_major_cn)`, then sum its
logarithm across regions. This profiles over **all** original multiplicity
candidates, not only 1 and 2. It does not normalize over multiplicities, average
their likelihoods, or minimize entropy. No uniform-prior factors enter the
reported profile score; equal cluster priors would not change its ordering.
Exact profile-score ties use lower public label. Impossible candidates cannot
win. Unobserved/excluded pairs contribute nothing; with no informative regions,
profile scores and the final assignment are missing. No centers are refitted.

The unchanged v5 bundle produces result **v8** with `profile_log_scores`,
`profile_tied_labels` and `selected_profile_log_score`. Existing `map_label`,
`best_label`, `posterior` and probability/log-score fields still describe the
original **marginal** likelihood. `selected_probability` and `selected_log_score`
give that original marginal evidence for the profile-selected reference;
profile scores are **not posterior probabilities**. `final_assignment_label`
and regional `selected_label`, CCFs, multiplicities and residuals use the profile
winner. Conditional multiplicity distributions remain normalized for reporting.
Independent publication checks verify both scoring definitions. The default
workflow, v6 MAP results and v7 entropy comparison remain unchanged.

## Simulation and attributable comparison

The supplied roster remains 60 subcases, 120 development histories, 600 main
histories and 2,640 acquisitions with unchanged seeds and pairing. One declared
configuration runs throughout. Development is descriptive; neither poor finite
performance nor an acceptance fraction blocks main execution. Computational
failures stay in planned-population accounting. Historical evaluation definitions
v2/v3/v4 stay frozen; current v5 reports all-target MAP and probability outcomes,
reference availability, residuals, origin strata and computational outcomes.

Separate comparisons identify changes in (1) reporting on frozen historical
references, (2) source/cap/reference construction and all-target reassessment on
the same restricted population, and (3) high-CN population expansion. Include a
bounded prespecified full-support Stage 2 comparator to measure cap-induced
reference changes. No biological-performance cutoff controls publication.

```sh
clipp2 workflow --plan examples/workflow_protocol23.plan.json --output-dir results/workflow23
clipp2 audit-input --input-file examples/workflow_protocol22.tsv --output results/input-audit.json
python tools/check_research.py --groups workflow biology evaluation study publication
python tools/qualify_workflow.py --device cpu --replicates 3 --output results/qualification.json
```

CUDA qualification/native cleanup remain deferred. CPU correctness checks do not
establish GPU arithmetic, global optimality, calibration or biological superiority.

The bounded comparison harness accepts an explicitly pinned earlier installation
and source snapshot; it does not discover or modify historical runs:

```sh
python tools/compare_protocol23.py --outdir /new/comparison \
  --baseline-python /historical/venv/bin/python --baseline-source /historical/src \
  --baseline-revision ORIGINAL_COMMIT --current-revision CURRENT_COMMIT
```

It fixes three low-CN cases, two full-support Stage 2 comparisons and one separate
high-CN population-expansion case before execution. Reports retain full-precision
reference inventories, cap endpoints, target populations and membership outcomes.
