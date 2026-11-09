# Workflow Protocol 2.3

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

1. **Stage 1:** fit eligible informative mutations under original support
   `1..original_major_cn`, allowing up to 10 groups by default. Select an occupied
   source by smaller full-vector L2 distance to all ones, then larger minimum
   regional informative depth, then lower public source label. This is a
   **provisional computational source**, not a declaration of biological clonality.
   Its actual native center remains an explicit reference candidate.
2. **Stage 2:** fit the remaining population, allowing up to 20 groups by default.
   At each mapped source observation, calculate component likelihoods under
   original support and the native source CCF. Preserve the full exact maximizing
   multiplicity set and historical native point call separately. Each defined
   observed contributor supplies the **largest exact maximizing integer**. A
   region–segment cap is the minimum of these endpoints, with original-support
   fallback when no endpoint is available. The policy is
   `minimum_native_map_set_upper`. Missing observations do not supply endpoints;
   all-impossible component likelihoods are undefined, never uniform. A flat
   likelihood supplies the original upper endpoint and cannot restrict support
   alone. In particular, a zero-CCF/zero-ALT observation has no multiplicity read
   information; its flat set is reported as such, without using its arbitrary
   display point call to restrict support. Original CN denominators never change.
3. **Frozen references:** take the canonical all-one hypothesis, the provisional
   Stage 1 source's native center, and every native Stage 2 center. Consolidate
   only exactly equal coordinate vectors. Preserve every source's lineage,
   original center, fitted-state identity and diagnostic measurements. Assign
   equal priors **after** consolidation. Near-identical centers remain distinct.
4. **Stage 3:** reassess every mutation in the combined fitted population once:
   `T = G0 union Stage2`, `F = empty`. Restore full original multiplicity support
   and use common observation masks. There is no geometric merging or selective
   reassessment rule. Baseline labels identify each mutation's native source
   center after exact consolidation, not an assumed all-one membership.

A nonempty fitting population with an entirely unsupported requested region
remains explicitly unidentifiable. Removing an acceptance rule does not identify
an absent coordinate. An empty Stage 2 remainder is valid: reassess the source
against its preserved native center and the canonical all-one hypothesis.

The cap is a discovery heuristic, not a biological upper-bound certificate. One
uncertain low MAP endpoint can reduce it and exclude another mutation's preferred
multiplicity. Restoring support in Stage 3 does not refit discovery centers.
Preserving a source at 0.4 alongside 1 and a Stage 2 center at 0.2 prevents that
native explanation from disappearing; it does not guarantee correct recovery.

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
diagnostic makes that aggregate unavailable, not the estimate. The canonical
all-one hypothesis without an exactly matching native source reports discrepancy
against provisional-source reads with that provenance; its center is imposed.

Whole-original output includes final MAP labels for fitted targets, explicit
exclusions, and prior-only outcomes for otherwise eligible unobserved mutations.
Those outside the fitted population do not enter fitted-target occupancy counts.
Historical cap contribution remains attached to the discovery fit even if the
mutation later moves away from its native baseline reference.

## Contracts and historical replay

The reference-preserving revision uses workflow plan/receipt **v5**, assignment
bundle/result **v5/v6**, original catalog **v5**, paired experiment **v5**,
simulation contract **v7**, and evaluation/definition **v5**. These versions
separate it from the earlier Protocol 2.3 routing proposal and Protocol 2.2.
Historical records retain their original interpretations; the protocol string
alone never selects an evaluation definition.

Compatible native fits keep their original protocol, producer, accepted-state
identity and verification evidence. Current compatibility is appended. Stage 1
requires unchanged population, biological observations, masks, supports and
native fitting settings. Stage 2 also requires unchanged remainder and effective
caps. A CN-policy label change alone may be compatible only if the complete
actual eligibility mask and input content are identical. Expanded populations
require new fits. Historical catalog v4 readers preserve the original payload.

Historical assignment artifacts are read-only. To score an explicitly identified
current-policy derivative of a frozen bundle, use `clipp2 assign --derive-current`.
This preserves its supplied reference inventory and target scope; it is a
reporting-only comparison, not a reconstruction of the revised workflow.
Independent publication checks still verify scores, normalization, identities,
supports, immutable content and final population accounting.

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
