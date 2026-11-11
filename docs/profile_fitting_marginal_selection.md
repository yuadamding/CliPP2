# Profile fitting with marginal-mixture selection

This opt-in Stage 1–2 estimator fits each candidate partition using
integer-multiplicity **profile likelihood**, then selects candidates using
**marginal-mixture BIC at their frozen fitted centers**. Marginal fitting remains
the compatibility default. Stage 3's independent assignment method is unchanged.

```bash
clipp2 fit --input-file tumor.tsv --output-dir result --device cpu \
  --fit-multiplicity profile
```

Workflow plans set `workflow_config.fit.fit_multiplicity` to `"profile"`; Stage 2 inherits
this setting when replacing the maximum cluster count. The sole implemented
`selection_criterion` is `"marginal_mixture_bic_at_fitted_centers_v1"`.

## Numerical contract

For mutation $i$, region $r$, center $c$, and allowed integer multiplicity $m$,

$$
b_{ir}(m;c)=\log\operatorname{Bin}(Y_{ir};N_{ir},a_{ir}mc),\qquad
a_{ir}=\frac{\rho_r}{(1-\rho_r)C^N_{ir}+\rho_r(A_{ir}+B_{ir})}.
$$

The existing probability-domain handling, observation masks and actual support
intervals are retained. Stage 2's support restriction does **not** alter its
original CN denominator. For a fixed partition $z$, fitting maximizes

$$
\ell_{\mathrm{fit}}(z,\Phi)=\sum_i\sum_{r\in\mathcal O_i}
\max_{m\in\mathcal M_{ir}}b_{ir}(m;\phi_{z_i r}).
$$

At the returned, immutable centers $\widehat\Phi$, selection forms

$$
A_{ik}=\sum_{r\in\mathcal O_i}\left[
\operatorname{LSE}_{m\in\mathcal M_{ir}}b_{ir}(m;\widehat\phi_{kr})
-\log|\mathcal M_{ir}|\right].
$$

Only cluster mixture weights are then optimized:

$$
\widehat\pi\in\arg\max_{\pi\in\Delta_K}
\sum_i\log\left(\sum_k\pi_k e^{A_{ik}}\right),\qquad
S=-2\ell_{\mathrm{mix}}+(KR+K-1)\log n.
$$

Here $K$ is occupied cluster count and $n$ is the current stage's fitted
mutation count. Positive-weight eligibility and zero-weight repair are unchanged.
This is a hybrid BIC-form criterion, **not** joint mixture-center MLE or a global
clustering optimum. An improved profile fit can have worse marginal BIC.

| Calculation | Evaluator |
| --- | --- |
| Initial pooled pilot/tree | Existing marginal initializer |
| Continuation loss, active-branch derivatives, line search | Configured fitting objective |
| Conditional center refits; boundary/membership/coarse-split proposal gains | Configured fitting objective |
| Weight optimization, frontier/per-K/incumbent/final ranking and audits | Marginal mixture at frozen centers |
| `log_emission`, conditional multiplicity probabilities | Explicit marginal/component diagnostics |
| `max_component_log_mass`, multiplicity MAP sets | Raw component maximum |

## Scalar profile refit

For consecutive multiplicities $m,m+1$ and $0<Y<N$, their switch is

$$
t=-\operatorname{expm1}\left[-\frac{Y}{N-Y}\log(1+1/m)\right],\qquad
c_{\mathrm{switch}}=\frac{t}{a(1+mt)}.
$$

Observation switches are cached, unioned per cluster–region, numerically
bracketed, and deduplicated **only by exact value**. Adjacent floating-point
neighbors are retained. Every interval and endpoint is checked; between switches
the active binomial branches are fixed and concave. A bracketed derivative root
is bisected to the coordinate's declared tolerance. No branch is dropped to meet
a work quota. Failure to complete a root/interval is an explicit numerical error.

Zero ALT prefers the smallest multiplicity away from zero; all ALT prefers the
largest. At zero CCF with zero ALT all multiplicities tie; positive ALT there
has zero likelihood. Exactly equal computed scalar objectives choose the smaller
CCF. This deterministic representative does not imply unique biological CCF.
The refit is independent of its initial center; it is not alternating local MLE.

Continuation uses one active branch's derivatives, never posterior averages.
At exact ties, right/left negative-log slopes are respectively the minimum and
maximum active branch gradients. Their feasible projected directional residual
includes the forest penalty and retains the existing tolerance/box scaling.
Backtracking evaluates the actual profile objective. Continuation remains a
finite-penalty partition proposal; its diagnostics do not certify a global fit.

## Ownership, identities and replay

`fit_backend` and `selection_backend` are immutable-role native contexts; marginal
mode shares one context. Typed joint-column handles identify their aggregation.
Only marginal `selection_columns` can enter prepared parents/weight optimization.
Coordinate, column, weight and audit identities bind the method and native build.
Both contexts close on success, exceptions and interrupted execution.

The four identities are `fit_objective_id`, `selection_criterion_id`,
`initialization_policy_id`, and `scalar_refit_policy_id`. Profile mode uses
`integer_multiplicity_profile_v1`, `marginal_mixture_bic_at_fitted_centers_v1`,
`existing_marginal_pooled_pilot_v1`, and `profile_branch_interval_search_v1`.
Accepted-fit/state v4 and producer v3 bind these identities and
`fitting_log_likelihood`. Historical v3 accepted states remain marginal and retain
their original serialized identity. Replay never relabels a marginal fit as
profile-fitted. Standalone public TSV schemas remain unchanged.

Existing `conditional_log_likelihood` means assigned **marginal** likelihood;
`mixture_log_likelihood`, `complexity_penalty` and `score` also keep their meanings.
Independent publication verification reconstructs fitting, marginal conditional,
marginal mixture and BIC terms separately. Scoring never invokes center refitting.

Native emission ABI 8 and continuation ABI 9 carry the aggregation explicitly,
including CPU observation exports and CUDA continuation request/reply slopes.
CPU/CUDA share the branch-search policy. CUDA executes its emissions, objectives,
roots and columns on-device; the existing forest/controller remains on the host.
Device branch derivatives reuse the ordered block reducer instead of duplicating
the recursive scalar reduction in every lane.
Stale/absent native libraries fail explicitly, with no silent objective/device
fallback. CUDA execution requires its own allocated-device qualification; a CPU
test or a successful CUDA compilation is not GPU numerical evidence.

## Qualification and research comparisons

```bash
python tools/check_profile_likelihood.py --outdir ../profile-checks --device cpu
python tools/compare_profile_fitting.py --input tumor.tsv --output-dir ../paired \
  --scope search --device cpu --max-clusters 10
```

The small qualification includes the two-mutation global/local-mode example,
normalization, singleton support, endpoints, masks, derivatives, exhaustive
small-support branch oracles, cache isolation, frozen scoring, native/host/request
continuation, full searches, workflow inheritance and replay rejection.

Comparison scopes:

- `fixed-centers`: JSON `--requests` list of `{labels, centers}`; score both mode
  identities at exactly the same centers.
- `fixed-partitions`: JSON list of canonical label arrays; fit each under both
  objectives, always using marginal selection.
- `search`: complete matched searches followed by fixed-partition refits of their
  union, including numerically completed zero-weight rejected candidates.
- `workflow`: freeze one marginal Stage 1 source, remainder and cap set for an
  isolated Stage 2 comparison; separately run the full profile workflow so Stage 1
  may change source selection and caps.

Reports separate occupied K, both fitting diagnostics, marginal conditional and
mixture likelihoods, penalty/BIC, source/cap changes and optional truth-based CCF
error, ARI and false-merge/split pair counts. Truth is read only after fitting.
Biological differences are descriptive; numerical and semantic correctness are
the implementation gates. Comparison artifacts are external research outputs,
not additional standalone fit TSVs.
New simulation reports record the fitting identity and refuse to aggregate
different fitting objectives. Historical report hashes remain unchanged.
