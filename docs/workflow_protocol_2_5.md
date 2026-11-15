# Workflow Protocol 2.5

Protocol 2.5 restricts each mutation's regional multiplicity range to at most
one throughout inference. The restriction is established before pilot CCFs and
tree construction, applies to every fitted candidate, and is retained during
fixed-reference Stage 3 assignment. Different mutations in a CCF cluster may
have different multiplicities.

`FitConfig.max_multiplicity_difference=1` is the default. `None` selects the
independent regional model; the CLI spelling is `--max-multiplicity-difference none`.
Other limits are rejected. Stage 2 inherits the setting from Stage 1.

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
retain the independent floor/cap treatment. Explicit support TSVs remain hard;
an empty joint support raises an error instead of relaxing the supplied intervals.

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
those pools is combined through the joint DP to obtain posterior CCF means before
`build_tree`. A structurally feasible row with zero pooled joint evidence uses
the labeled interior start `(pooled_CCF+0.5)/2`, preserving NA on unobserved
coordinates; that fallback is not reported as a posterior mean.

Continuation uses one joint loss per mutation and regional gradients/positive
working curvatures. Endpoint derivatives retain contributions from zero-mass
components through cavity likelihoods. Line searches evaluate the actual joint
objective; a diagonal working scale is not an exact Hessian.

Group refits operate on full regional center vectors. Marginal fitting uses
joint posterior EM, interior boundary rescue and feasible stationarity checks.
The accepted stationarity residual is the largest feasible gradient or
directional-slope magnitude divided by one plus the group mutation count, with
tolerance `1e-5` (the inner iteration targets `1e-7`).
Profile fitting uses conditional coordinate envelopes with offsets from the
other regions. Each conditional coordinate search can be global, but the full
multiregion fit is local. Whole-group caches reuse unchanged memberships and
full starting vectors; independent scalar caches are used only with `None`.

Set `fit_multiplicity="profile"` in the workflow plan, or pass
`--fit-multiplicity profile` to `clipp2 fit`, to select profile center fitting.
Stage 3 retains its separately configured assignment method.

With `max_multiplicity_difference=None`, emissions and pooled pilots factor by
region, marginal scores subtract each regional interval-width normalizer, and
center refits are scalar. Independent profile refits search adjacent-branch
breakpoints and bounded derivative roots to find a scalar global maximum.
That guarantee does not extend to joint group refits. Profile scores use raw
binomial maxima, including binomial coefficients, without prior normalization.

## Model selection and Stage 3

Stages 1 and 2 select the lowest BIC in the retained candidate bank after bounded
tree exploration and reassignment proposals. They do not run exhaustive
single-mutation membership refinement or publish local-optimum certificates.
Selected-fit numerical verification and candidate-bank reconciliation remain
mandatory. Stage 3 independently verifies its fixed-reference assignments.

At fitted centers, mixture weights maximize the constrained marginal likelihood:

$$
\ell_{mix}=\max_{\boldsymbol\pi}\sum_i\log\sum_k\pi_k e^{A_i(\boldsymbol\phi_k)},
\qquad \mathrm{BIC}=-2\ell_{mix}+(KR+K-1)\log n.
$$

The fixed difference limit adds no continuous parameter. Profile fitting retains
marginal-mixture selection at its fitted centers. This remains a BIC-form hybrid
criterion with data-derived supports and conditional center fitting, not a claim
of joint mixture-center optimization or new BIC consistency.

Stage 1 selects a provisional source nearest all ones. Donor multiplicities use
the projections of the complete joint MAP set, taking each regional projection's
upper endpoint and then the minimum across eligible donors in that region-segment.
Mapping, observation and exclusion conditions remain in force, and CCF must be
positive. The representative joint MAP vector is selected deterministically in
regional lexicographic order.

Stage 3 freezes one all-one reference plus distinct Stage 2 centers and reassesses
all fitted mutations. Its scores are $A_i-\log K$ or $P_i-\log K$. Marginal ties
use the lowest label. Profile ties retain the baseline when it is in the exact
maximizing set, otherwise the lowest label. Profile scores are not posterior
probabilities. Prior-only and all-impossible rows retain their explicit statuses.

Selected multiplicities come from a joint MAP vector, not separate regional
re-optimization. Public zero-CCF calls become NA after latent probabilities and
residuals have been computed. Regional maxima remain explicitly regional
diagnostics. Joint selected-component deviances and the nonnegative coupling
profile loss distinguish the cost of coupling from independent regional fit.

## Publication and replay

Current identities bind the difference setting, support normalization, pilot
policy, group-refit policy and both fitting/selection objectives. New receipts
require the setting explicitly; older receipts cannot inherit the new default.

| Contract | Version |
| --- | --- |
| Protocol / package | 2.5 / 2.2.0 |
| Fit output / numerical verification | v16 / v6 |
| Workflow plan / receipt | v7 |
| Accepted fit / state / producer | v5 / v5 / v4 |
| Assignment bundle / result | v7 / v10 |

Publication independently reconstructs joint scores, normalization, multiplicity
calls, ties, conditional probabilities, and BIC. Accepted-fit replay verifies
the saved inputs, support, configuration, and numerical state without refitting.
