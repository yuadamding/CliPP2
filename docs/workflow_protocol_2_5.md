# Workflow Protocol 2.5

Protocol 2.5 restricts each mutation's regional multiplicity range to at most
one throughout inference. The restriction is established before pilot CCFs and
mixture initialization, applies to every fitted candidate, and is retained during
fixed-reference Stage 3 assignment. Different mutations in a CCF cluster may
have different multiplicities.

`workflow_config.fit.max_multiplicity_difference=1` is the default; JSON `null`
selects the independent regional model.
Other limits are rejected. Stage 2 inherits the setting from Stage 1.

The only command is `clipp2 workflow --plan PLAN.json --output-dir OUTPUT`.
Its v7 plan contains `schema`, `protocol_version`, `input_path`, `workflow_config`
and optional `run_manifest_id`. Both fitting stages run afresh; saved-fit and
build-manifest inputs are rejected. Stage 1/2 records preserve provenance only.

## Fixed observation scope

For mutation $i$, use the resolved mask

$$
J_i=\{r:\mathrm{count\_observed}_{ir},\ \neg\mathrm{input\_excluded}_{ir},\ N_{ir}>0\}.
$$

With effective regional intervals $[L_{ir},U_{ir}]$, the admissible set is

$$
\mathcal C_i=\{\mathbf m:m_r\in[L_{ir},U_{ir}]\cap\mathbb Z,
\quad\max_{r\in J_i}m_r-\min_{r\in J_i}m_r\le1\}.
$$

Only coordinates in $J_i$ enter the latent vector. The empty set of observations
has one empty configuration and log emission zero. With one informative region,
the constraint is vacuous. If all supports are `{1,2}`, every vector remains
admissible: the rule does not separately favor equality.

The mask does not change when a proposed CCF becomes zero. At zero CCF, positive
ALT has zero likelihood and zero ALT has component likelihood one. The coordinate
still participates in the latent constraint, but its public biological multiplicity
call is NA and it cannot donate a cap. This is stronger than a rule applied only
to mutation-bearing regions; genuinely discordant copy numbers in absent regions
can therefore affect inference.

Missing regional SNVs are automatically resolved to ALT=0 and REF=the region's
median eligible, originally observed positive depth, with half-read ties rounded
up. Those imputed observations participate in $J_i$. All stages use the same
frozen ledger. Raw input rows remain preserved. `missing_counts="unobserved"`
keeps explicitly missing pairs outside the mask. Measured zero-depth pairs and
input exclusions remain outside it. A region with no usable median donor fails
input validation.

## Raw requests and effective support

The [input contract](../README.md) defines original major CN $M$, supplied purity
$p$, and cell-fraction-weighted total CN $\bar C$. Raw read floors remain

$$
L_r=\max\left(1,\min\left[M_r,
\left\lfloor\frac{a_r/d_r}{p_r}(p_r\bar C_r+2(1-p_r))\right\rfloor\right]\right).
$$

These estimates use original reads and purity, always assume normal CN two,
have no fixed upper cap of four, and become one for zero ALT or missing/zero-depth
pairs. They can underestimate subclonal multiplicity because they do not divide
by CCF. Weighted CN affects these floors; the likelihood retains the supplied
representative major/minor CN and supplied normal CN.

The priority for workflow-generated supports is original major CN, joint range
feasibility, read floors, then donor caps. Let $C_r$ be the raw upper cap, with
$C_r=M_r$ in Stage 1. Over $J_i$ compute

$$
B=\min_r M_r,\qquad L_r^*=\min(L_r,B+1),\qquad
q=\max(1,\max_r L_r^*-1),\qquad U_r^*=\max(C_r,L_r^*,q).
$$

This retains the largest feasible floors and raises caps only as needed. Original
major CN is never exceeded. Stage 2 starts again from the original raw floors
and its raw donor caps, not from Stage 1's relaxed requests. Unobserved coordinates
retain the independent floor/cap treatment. An empty joint support raises an
error rather than silently changing the multiplicity model.

In independent mode, $L^*=L$ and $U^*=\max(C,L)$, including the singleton result
when a cap is below its floor. Both modes exclude zero-CCF donor coordinates.

Supporting outputs retain raw floors, raw caps, effective endpoints and the
reason for each adjustment. Stage 3 uses original $[1,M_r]$ intervals plus the
same difference policy, without empirical floors or donor caps.

## Joint likelihood and numerical implementation

For raw binomial log mass $b_{ir}(m;\phi_r)$, marginal and profile emissions are

$$
A_i(\boldsymbol\phi)=\operatorname{LSE}_{\mathbf m\in\mathcal C_i}
\sum_{r\in J_i}b_{ir}(m_r;\phi_r)-\log|\mathcal C_i|,
\qquad
P_i(\boldsymbol\phi)=\max_{\mathbf m\in\mathcal C_i}\sum_{r\in J_i}b_{ir}(m_r;\phi_r).
$$

The prior is uniform over vectors. Regional prior factors are removed before
joint evaluation, and exactly one joint normalizer is subtracted. The native
canonical-minimum DP visits anchors $h$; coordinates are $h$ or $h+1$ and at least
one must equal $h$. Its two states distinguish whether $h$ has occurred, so each
vector is counted exactly once. Structural support is nonempty exactly when
$\max_r L_r\le\min_r U_r+1$ for nonempty $J_i$.

Regional pooled CCF distributions are retained. Component-wise integration over
those pools is combined through the joint DP to obtain posterior CCF means.
A structurally feasible row with zero pooled joint evidence uses
the labeled interior start `(pooled_CCF+0.5)/2`, preserving NA on unobserved
coordinates; that fallback is not reported as a posterior mean.

Each fitting stage first runs **A**, the recurring
conditional tree/general-partition search, on that stage's observations and
multiplicity support. Its fitted centers and weights initialize **B**, the soft
marginal-mixture search below. This initializer is computed afresh, not borrowed
from a saved fit or another stage. A's conditional fit is an initialization step,
not the final production estimator.

Centers and weights jointly optimize the observed marginal-mixture likelihood.
Soft cluster responsibilities multiply conditional joint-multiplicity
posteriors in the weighted binomial center M-step. The fitting-specific native
E-step returns emissions and packed posterior probabilities without publication
MAP/tie diagnostics. Fixed support normalizers and packed read/slope terms are
reused. Identical slopes are aggregated exactly in the concave coordinate solve;
this does not truncate multiplicity support or merge nearby centers.

Intermediate iterations use EM weight updates, with periodic and final full
weight optimization. Warm-start weights already passing the KKT gate need no
new optimizer run. Endpoint derivatives retain finite probability derivatives
from zero-emission rows through cavity likelihoods. Actual-likelihood rescue
uses objective and gradients normalized by `1+n`; center qualification requires
the feasible projected-gradient residual at most `1e-5`. Final weights must pass
their original stationarity gates. Iteration exhaustion is not convergence.
Monotonicity is checked within each trajectory, not against a different start's
initial likelihood. MAP labels are reported only after fitting and never cause
a hard-partition center refit.

Stage 1–2 execution supports **CPU/marginal only**. CUDA and profile
fit requests fail explicitly, without CPU fallback or A-only substitution.
Stage 3 still supports its separately configured marginal or profile assignment.
With `max_multiplicity_difference=null`, emissions and pooled pilots factor by
region and marginal scores subtract regional interval-width normalizers; the
new production objective remains a soft mixture. Profile assignment uses raw
binomial maxima, including binomial coefficients, without prior normalization.

## Model selection and Stage 3

Stages 1 and 2 explore pilot-based births, transfer-derived starts, all-pair
merges and atomic merge-two/split-third starts. Proposals are constructed lazily
and consumed round-robin, with a recurring candidate frontier. B uses 32 proposal
slots beyond its fresh A initializer. This is a bounded
heuristic search, not exhaustive exploration or a global component-selection
algorithm; an unvisited or unconverged start has not been ruled out.

Each proposal tries its supplied centers and an interior blend, skipping exact
duplicate starts. Every distinct converged result is retained for the outer
BIC comparison, including starts with different effective component counts.
Selecting only the highest likelihood across those counts is not valid.
Exactly identical center vectors are consolidated by summing weights, and
component order is canonicalized before scoring and search deduplication.
Nearby but unequal centers are not silently consolidated. Exact-zero slots are
removed from final states; a distinct positive-weight component is retained
even if it has no MAP-assigned mutations.

For positive component count $K_+$, the fitted objective and selection score are

$$
\ell_{mix}(\Phi,\boldsymbol\pi)=\sum_i\log\sum_k\pi_k e^{A_i(\boldsymbol\phi_k)},
\qquad \mathrm{BIC}=-2\ell_{mix}+(K_+R+K_+-1)\log n.
$$

The fixed difference limit adds no continuous parameter. This is joint
mixture-center optimization, but neither numerical stationarity nor bounded
search proves a global maximum, global BIC minimum, biological accuracy, or
BIC consistency with data-derived support. Optimizer convergence, weight
stationarity and retained-bank BIC checks remain. There is no separate independent
Stage 1/2 numerical-verification or replay pass. Hard-group occupancy and
group-region coverage are not soft-mixture acceptance gates.

The fitting identity is `regional_A_initialized_marginal_mixture_v1`, with
`fresh_A_conditional_tree_search_v1` initialization and the
`observed_joint_marginal_mixture_v1` objective.

Stage 1 selects the occupied MAP group nearest all ones in its **B result** as
its provisional source. A positive component without MAP members remains in the
native fit but cannot supply donor reads. Donor multiplicities use
the projections of the complete joint MAP set, taking each regional projection's
upper endpoint and then the minimum across eligible donors in that region-segment.
Mapping, observation and exclusion conditions remain in force, and CCF must be
positive. The representative joint MAP vector is selected deterministically in
regional lexicographic order.

The Stage 2 population, source references and caps are recomputed from this
Stage 1 B result. Stage 2 then runs its own fresh A initializer and B refinement
under the Stage 2 support; it does not reuse Stage 1's A fit or saved centers.

After Stage 2, every native center with $\|\boldsymbol\phi_k-\mathbf1_R\|_2<0.05$
is merged into the fixed all-one clonal reference, label 0. The threshold is
strict and uses the full vector without division by the number of regions;
centers at distance 0.05 or greater remain unchanged. Other exact duplicate
centers are consolidated as before. This is postprocessing, not a new fit:
native fitted centers, memberships, scores and multiplicities remain intact.
`source_to_reference.tsv` and `baseline_reference_label` record the combined
Stage 2 labels; `reference_vectors.tsv` gives their centers. Diagnostics for
merged sources are evaluated at all ones, with native diagnostics retained.
`merged_groups` counts non-exact near-clonal Stage 2 groups; exact duplicates
are counted separately. The reference-policy identity binds this rule.
Distinct positive Stage 2 components without MAP members also remain reference
candidates; their read-support diagnostics are explicitly empty, not evidence
for deleting their fitted mixture contribution.

Stage 3 freezes this consolidated set and reassesses all fitted mutations; the
merge does not lock their final assignments. Its scores are $A_i-\log K$ or
$P_i-\log K$. Marginal ties
use the lowest label. Profile ties retain the baseline when it is in the exact
maximizing set, otherwise the lowest label. Profile scores are not posterior
probabilities. Prior-only and all-impossible rows retain their explicit statuses.

Selected multiplicities come from a joint MAP vector, not separate regional
re-optimization. Public zero-CCF calls become NA after latent probabilities and
residuals have been computed. Regional maxima remain explicitly regional
diagnostics. Joint selected-component deviances and the nonnegative coupling
profile loss distinguish the cost of coupling from independent regional fit.

## Publication

Current identities bind the difference setting, support normalization, seed
policy, fitting algorithm and both fitting/selection objectives. Saved records
are outputs, not inputs for another fit or resumed workflow.

| Contract | Version |
| --- | --- |
| Protocol / package | 2.5 / 2.2.0 |
| Native mixture fit tables | v17 |
| Workflow plan / receipt | v7 |
| Assignment bundle / result | v7 / v10 |

Each native stage retains its three fit tables. Schema 17 centers include all
positive components, with `cluster_size=0` permitted for MAP-empty components;
summaries distinguish positive components from occupied MAP groups. Public
label 0 remains the largest final-center L2 norm, not an enforced clonal center.

Final native multiplicity reporting runs only for the selected state. Stage 3
retains its fixed-reference scoring, assignment verification and output schema.
The retained `conditional_log_likelihood` is descriptive for soft fits;
`fitting_log_likelihood` is their observed mixture likelihood.
