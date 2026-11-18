# CliPP2 Protocol 3.0

## Observations and emissions

For each informative mutation–region pair, let `Y` be ALT count, `N` depth,
`rho` purity, `CN_N` normal CN, and `A,B` the original representative major
and minor CN. Set `a = rho / ((1-rho)*CN_N + rho*(A+B))`.
The regional distribution is `Binomial(N, min(1, a*m*phi))`. Missing pairs
can have explicitly imputed counts (default) or remain uninformative; eligibility
and the observation mask are fixed before either fit. Missing/zero-depth
coordinates do not contribute to the joint multiplicity constraint unless
resolved by the declared missing-count policy. Requested regions are never
silently dropped. The input is the same across both stages.

For mutation `i`, let `G_i(m; phi)` be the product of regional binomial masses
on informative pairs. Define `M_i` as the integer vectors satisfying
`1 <= m_r <= A_ir` and `max(m)-min(m) <= 1` on those regions. There is no
read-derived lower bound above one. Define:

```text
U_i = mean over m in M_i of G_i(m; 1)    # fixed clonal, uniform joint prior
V_i = G_i(1; 1)                         # fixed clonal, only single-copy
S_ik = G_i(1; phi_k)                    # free CCF, only single-copy
g_i = w_U*U_i + w_V*V_i + sum_k w_k*S_ik
log_likelihood = sum_i log(g_i)
```

Only the free centers are optimized, on the closed `[0,1]^R` box. A center
such as `(1,0.4,0)` is globally subclonal and remains single-copy in all
observed, present regions. The weights are nonnegative and sum to one.

There is **one** biological clonal class, of weight `pi_C = w_U+w_V`. If
identifiable and active, `clonal_prior_beta = w_V/pi_C`. The probability of
single-copy under the prior is `beta + (1-beta)/|M_i|`. The posterior analogue
uses `eta_i = w_V*V_i / (w_U*U_i+w_V*V_i)` instead of `beta` and the
uniform-term posterior instead of its prior. Joint MAP uses the enriched
prior, not the MAP of `U` alone or independent regional MAPs.

An exact all-one free center becomes `V` without changing its emission. Near
ones are not promoted unless a separately fitted candidate wins the score.
Exact equal free centers merge; `U,V` are distinct unless the model structure
makes their distributions identical (all informative major CNs = 1). In
that case the prior enrichment is unidentifiable and `beta` is null.
Coincidental equality of two columns on observed counts is not enough to
declare structural identity.

## Discovery and refinement

Stage 1 prepares single-copy pilots `clip(Y/(N*a),0,1)` on observed coordinates.
The two seed families use farthest-point geometry and recursive largest-gap
or median splits. Unsupported seed coordinates use pooled regional values,
not additional read evidence. Every permitted count receives applicable
templates `()`, `(U)`, `(V)`, `(U,V)` and distinct deterministic starts, outside
the optional proposal budget. Exact duplicates and failures are recorded.

The optimizer uses concave free-coordinate updates, periodic full weight
optimization including permitted zero-weight terms, and constrained ascent
with endpoint-aware raw derivatives. Each accepted fit must pass numerical
qualification. Split, merge, transfer, split–merge, template, promotion and
likelihood-ranked birth proposals extend the seed bank. The search is bounded;
stationarity does not certify global optimality.

For `S` distinct active free centers and `c` active distinct fixed terms,
the biological count is `S + 1(c>0)` and the model parameter count is
`d = S*R + S+c-1`. Select `BIC = -2*log_likelihood + d*log(n)`. A clonal
class with both terms uses one biological slot, but its second weight adds a
parameter. Positive components with no MAP members are not discarded merely
for being MAP-empty. A no-clonal template can use the whole capacity for free
components.

Stage 2 uses the same data and model, keeping the accepted Stage 1 solution in
the candidate bank, warming from it, adding count coverage, and searching
further proposals. Selected Stage 2 BIC cannot exceed accepted Stage 1 BIC.
This is a model-score guarantee, not a biological-accuracy guarantee.

## Assignment and public schema

Stage 3 independently verifies the selected fit and evaluates no new fitted
parameters. The biological clonal posterior is `tau_U + tau_V`; take MAP only
after this sum. All other posteriors use the fitted mixture weights as priors.
No equal-prior or profile assignment is an authoritative CLI mode.

Public tables have the following fields (with `phi_<region>` per region):

| File suffix | Fields |
| --- | --- |
| `mutation_clusters.tsv` | `tumor_id, mutation_id, cluster_label, phi_<region>...` |
| `cluster_centers.tsv` | `tumor_id, cluster_label, cluster_size, phi_<region>...` |
| `mutation_region_multiplicity.tsv` | `tumor_id, mutation_id, region_id, phi, multiplicity_call` |
| `cluster_probabilities.tsv` (Stage 3) | `tumor_id, mutation_id, cluster_label, posterior_probability` |

Label 0 is used only for an active clonal class. The free classes are numbered
from 1, even if the clonal class is absent. Regional multiplicity is NA if the
original observation is missing (including imputed pairs), excluded,
zero-depth, or assigned zero CCF. Present, observed free-class coordinates
use exactly one. The accepted Stage 3 JSON records enriched dosage diagnostics;
the posterior TSV contains every fitted SNV and active biological class.

The root contains stage accepted states and search records, the original
catalog (including missing-read provenance), `all_original_mutations.tsv`,
and a completion or failure receipt. Each stage has its own table directory.
Use a fresh output directory; publication never overwrites existing files.

## Plans and limitations

Use `schema="clipp2.workflow.plan.v8"`, `protocol_version="3.0"`, an
`input_path`, optional `workflow_config`, and optional `run_manifest_id`.
`workflow_config.fit` accepts `device="cpu"`, `cpu_threads`, `max_major_cn`
(default null), `max_clusters` (10), and `proposal_budget` (32). Refinement
uses `stage_two_max_clusters` (20) and `stage_two_proposal_budget` (32).
The same config can specify `normal_cn_policy` and `missing_counts`.
Plan v7 and its assignment, profile fitting, and range-override options are
incompatible. Saved fits are provenance, not public replay inputs.

Subclonal mutations with true dosage above one violate this model, as do
clonal dosage vectors outside the retained range or with true fractional
dosage not representable by one integer component. Mixed-state CN retains
the original representative denominator; this protocol is not a weighted-CN
correction. Classify benchmark cases by those assumptions, and compare BIC
only within the same likelihood, observation population, and parameter count.
