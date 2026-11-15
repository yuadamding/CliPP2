# Joint multiplicity qualification

CPU numerical checks and six small paired scientific fixtures passed on
2026-11-14, using Python 3.12.6 on macOS arm64, native float64 kernels and one CPU
thread. This establishes the tested numerical and integration behavior. It does
not establish improved biological accuracy or global optimization. CUDA execution
and CPU/CUDA numerical agreement were **not tested** on this machine.

## Numerical checks

`tools/check_joint_multiplicity.py` compares the reference dynamic program with
exhaustive enumeration in 300 randomized cases under both settings. The largest
reported absolute numerical error was `1.7763568394002505e-15`. It checks counts,
normalization, posterior summaries, MAP calls and ties, the fixed observation set
at zero CCF, and the empty prediction product. Another 300 randomized support
cases check feasibility, original major-CN preservation and maximal retained
floors. Explicit incompatible support remains an error. Native CPU marginal and
profile scores and derivatives passed reference checks with coupling enabled and
disabled. A further 100 randomized five-region native diagnostic cases verify
scores, posterior marginals, MAP vectors and projected ties, including impossible
rows and zero-CCF coordinates.

`tools/check_joint_model.py` passed these integration checks:

- Joint posterior pilots agree with independent raw-component integration and
  reach `build_tree` before the tree is frozen. The fixture's largest change from
  regional pilots was `0.03489254349112249`.
- Sparse pooled evidence uses the declared finite interior fallback.
- Marginal EM escapes the endpoint fixture and fits CCF `0.5999999994141967`;
  its endpoint negative-log-likelihood derivative is `4`.
- A supplied stationary CCF immediately below one remains a successful fit when
  rounding makes the endpoint root's objective one unit in the last place worse.
- Complete-group refits preserve or improve supplied incumbents. Their reported
  projected residuals agree with the separate public derivative interface and
  meet the `1e-5` stopping tolerance after division by one plus group size, for
  marginal and profile fitting.
- Conditional profile refits match an independent all-pair branch-crossing
  oracle in 32 randomized cases, including zero and all-ALT observations. Across
  200 crossings, the largest fitted-score error was `4.263256414560601e-14`.
- A single mutation move reuses unaffected complete groups and refits changed
  groups. Frozen-center marginal selection does not refit centers.
- Joint marginal and profile searches pass independent publication verification.
  Native CPU, Python-controller and reverse-request continuation agree.
- One informative region and supports already contained in `{1,2}` reproduce
  independent likelihoods and pilots.

`tools/check_profile_likelihood.py` also passed with its scalar-regression models
explicitly configured with `max_multiplicity_difference=None`.

## Scientific fixtures

`tools/compare_multiplicity_models.py` ran fixed-center comparisons, fixed-partition
refits and complete three-stage workflows for both policies. Each fixture has
eight SNVs, two regions, two true clusters, depth 200, purity one and deterministic
binomial reads from seed 4118. Full workflows receive reads without truth. The
fixed-state probes explicitly use the same truth centers or memberships under
both policies. These are small qualification examples, not a simulation cohort.

The following entries are **independent / range-one**. Fixed-center assignments
were identical and correct in all six fixtures. The likelihood column is the
change in fixed-center optimized marginal-mixture log likelihood, range-one minus
independent, using identical supports and centers.

| Fixture | Fixed-center log-likelihood change | Fixed-partition CCF MAE |
|---|---:|---:|
| Equal multiplicities | +2.0068 | 0.2063 / 0.1843 |
| Difference exactly one | +1.5673 | 0.2825 / 0.2343 |
| Difference greater than one | -176.1840 | 0.2276 / 0.1870 |
| Zero CCF with discordant CN | -105.8788 | 0.0303 / 0.1041 |
| Conflicting floors and caps | -516.3507 | 0.0569 / 0.1275 |
| Every vector already allowed | approximately 0 | 0.1811 / 0.0131 |

Even when the likelihoods are identical, fixed-partition results can differ:
independent scalar searches and joint local EM explore different starts and
solutions. In the last row, the joint fit has a lower fitting log likelihood
(`-58.4284` versus `-58.1135`) despite its smaller CCF error. Neither likelihood
improvement nor numerical stationarity implies better truth recovery.

Complete-workflow results use final assignments, which can combine reference
clusters retained from both fitting stages:

| Fixture | Cluster-count error | Membership ARI | CCF MAE | Multiplicity accuracy |
|---|---:|---:|---:|---:|
| Equal multiplicities | 0 / -1 | 0.4948 / 0 | 0.2250 / 0.3500 | 0.4375 / 0.5000 |
| Difference exactly one | +1 / +1 | 0.6957 / 0.6957 | 0.1131 / 0.1131 | 0.7500 / 0.7500 |
| Difference greater than one | -1 / -1 | 0 / 0 | 0.3500 / 0.3500 | 0.5000 / 0.5000 |
| Zero CCF with discordant CN | -1 / -1 | 1 / 1* | 0 / 0* | 0.3333 / 0.3333* |
| Conflicting floors and caps | 0 / 0 | 1 / 1 | 0.0297 / 0.1275 | 0.9375 / 0.5000 |
| Every vector already allowed | 0 / 0 | 1 / 1 | 0.0827 / 0.0827 | 0.5000 / 0.5000 |

\* In the zero-CCF fixture, four of eight mutations remain unresolved under both
policies. ARI covers only the four resolved mutations; CCF error covers eight of
16 pairs. Multiplicity accuracy counts four correct calls among 12 truth-positive
pairs, with only four defined calls. The reports retain these coverage counts.

In the conflicting-floor fixture, range-one relaxes four of 16 requested Stage 1
floors and four of 16 Stage 2 supports; independent mode adjusts none. Both modes
adjust four Stage 2 supports in the equal and greater-than-one fixtures. Other
fixtures have no support adjustments. Raw region-2 donor caps change from 4 to 2
in the zero-CCF fixture and from 6 to 2 in the conflicting-floor fixture. Every
scenario also records reconciliation under an explicit raw-cap-one stress test.

Observed workflow runtimes were 0.51–0.97 seconds for independent and 0.60–1.61
seconds for range-one. Incremental Python allocation peaks were 0.70–1.11 MB and
0.67–0.74 MB, respectively, measured by `tracemalloc`. These are single small runs
with allocation tracing enabled, not performance benchmarks. They exclude total
process RSS and GPU memory; fixed-state reports separately record owned native
host bytes. Larger true multiplicity differences deliberately violate the joint
model, and the examples show mixed or worse recovery as well as unchanged cases.

## Reproduction

From a checkout with its native extension installed, run:

```bash
python tools/check_joint_multiplicity.py --outdir /tmp/joint-numerical-checks
python tools/check_joint_model.py --outdir /tmp/joint-model-checks
python tools/check_profile_likelihood.py --outdir /tmp/independent-profile-checks
python tools/compare_multiplicity_models.py --scope qualification --output-dir /tmp/joint-scientific-qualification
```

The comparison output directory must be new. Each scenario retains generated
inputs, truth, fixed-center and fixed-partition reports, both full workflow
outputs, and a summary `comparison.json`. The run summarized here was written to
`/tmp/clipp2-joint-qualification-final2`; transient result directories are not
repository fixtures. On a CUDA host, run the existing qualification entry point,
which includes joint checks and CPU/CUDA comparisons:

```bash
python tools/check_research.py --groups cuda --outdir /tmp/joint-cuda-qualification
```

GPU numerical qualification remains unexecuted until that run succeeds on a
CUDA host.
