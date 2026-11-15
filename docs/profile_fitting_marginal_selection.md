# Profile fitting with marginal-mixture selection

Stage 1–2 center fitting supports `fit_multiplicity="marginal"` (default) and
`"profile"`. Both select candidate partitions by marginal-mixture BIC at the
fitted centers. Stage 3 has its own independently selected assignment method.

```sh
clipp2 fit --input-file tumor.tsv --output-dir result --fit-multiplicity profile
```

## Default joint support

`max_multiplicity_difference=1` restricts each mutation's regional multiplicities
to vectors whose maximum minus minimum is at most one over the fixed informative
observation mask. Multiplicities remain latent as CCFs change. The complete
[support, initializer, workflow and replay contract](workflow_protocol_2_5.md)
applies to both fitting methods.

For raw binomial component log likelihood $b_{ir}$ and joint support $\mathcal C_i$,

$$
A_i(\boldsymbol\phi)=\operatorname{LSE}_{\mathbf m\in\mathcal C_i}
\sum_r b_{ir}(m_r;\phi_r)-\log|\mathcal C_i|,
\qquad
P_i(\boldsymbol\phi)=\max_{\mathbf m\in\mathcal C_i}\sum_r b_{ir}(m_r;\phi_r).
$$

Marginal fitting maximizes the sum of $A_i$ for the assigned mutations; profile
fitting maximizes the sum of $P_i$. Joint marginal refits use EM and an exact
feasible-gradient check with boundary rescue. Profile refits optimize coordinates
conditionally on the other coordinates using joint multiplicity offsets and an
ordered upper envelope of weighted binomial branches. Complete center vectors
and group results are cached. Full multiregion refits are local, not globally
certified; conditional profile coordinate searches can be globally solved.

At the resulting frozen centers, only cluster weights are optimized:

$$
\widehat\pi\in\arg\max_{\pi\in\Delta_K}\sum_i\log\sum_k\pi_k e^{A_i(\widehat\phi_k)},
\qquad S=-2\ell_{mix}+(KR+K-1)\log n.
$$

The difference rule adds no continuous parameter. A better profile fitting score
can have a worse marginal selection score. Selection is a BIC-form hybrid,
not a joint mixture-center MLE or a guarantee of global clustering optimality.

## Independent comparison

Use `--max-multiplicity-difference none` or Python `None` to recover independent
regional multiplicity emissions, pooled pilots and scalar center refits. JSON
uses `null`. Marginal emissions average over each consecutive integer interval;
profile emissions use its raw binomial maximum. Independent profile scalar
refits use all adjacent-branch breakpoints and bounded derivative-root searches.
This independent scalar global guarantee does not apply to joint group refits.

The original CN/purity denominator remains unchanged in both settings. Raw
binomial masses include binomial coefficients. Joint marginal scores subtract
one joint-vector normalizer; independent scores subtract regional interval-width
normalizers. Profile scores subtract neither.

## Identities and verification

Fit configuration and method identity explicitly retain the difference setting,
normalization, initialization and refit policy. Accepted-fit/state v5 and producer
v4 bind the fitting score separately from marginal conditional/mixture scores.
Changing the setting requires new native fits; a missing policy cannot silently
inherit the new default. Original producer evidence and current execution links
remain separate during replay.

Publication independently reconstructs fitting and selection columns, weights,
penalty and BIC from source observations and effective support. Native CPU, CUDA
ordinary/resident columns and continuation must use the same policy. CUDA must
be tested on a compatible NVIDIA device; CPU success is not CUDA qualification.

```sh
python tools/check_profile_likelihood.py --outdir results/independent-profile-checks
python tools/check_joint_multiplicity.py --outdir results/joint-numerics
python tools/compare_multiplicity_models.py --help
```

Fixed-center comparisons isolate emission effects. Fixed-partition refits measure
changes to CCF and multiplicity estimates. Full-workflow comparisons include the
joint pilots, tree, discovery, donor caps and Stage 3. Preserve equal, difference-one,
difference-greater-than-one, zero-CCF and conflicting-support scenarios explicitly.
