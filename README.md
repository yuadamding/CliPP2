# CliPP2

CliPP2 implements a three-stage research method for joint mutation clustering
across tumor regions. It estimates cancer cell fractions (CCFs) and integer
multiplicities using **one representative allele-specific tumor copy-number
state per mutation–region observation** in the likelihood. Supplied CN-state
fractions additionally determine the read-based multiplicity lower bounds in
Stages 1 and 2; they do not introduce a mixed-state likelihood.

Protocol **2.5** chooses a provisional Stage 1 source, then fits the remainder
using segment caps from that source's multiplicity estimates. By default, each
mutation's multiplicities differ by at most one across informative regions.
After Stage 2, clusters with CCF-vector L2 distance **<0.05** from `(1,1,…)`
merge into clonal cluster **0**, fixed at `(1,1,…)`. Other centers stay unchanged.
Stage 3 restores original support and reassesses every fitted mutation against
this consolidated reference set. Native fit records remain unchanged as provenance.

Cluster limits default to **10 for Stage 1 and 20 for Stage 2**. Valid integer CN
has no automatic eligibility ceiling. Explicit input exclusions, model-domain
checks, optimizer convergence and candidate-bank checks remain. Each fitting
stage first runs **A**, the recurring conditional tree/general-partition search,
on that stage's observations and support, then uses its fitted state to initialize
**B**, joint marginal-mixture optimization of centers and weights. Stage 1's B
result determines the provisional source and caps; Stage 2 runs a fresh A→B fit
on the recomputed population and support. Every distinct converged B start enters
the BIC comparison. Exactly identical centers are consolidated; distinct
positive-weight components remain even when no mutation has that component as
its MAP label. Labels do not trigger a hard center refit.
The bounded birth/transfer/merge/split–merge search is not a local- or
global-optimality certificate.
Package **2.2.0** implements scientific protocol **2.5**. The
[protocol specification](docs/workflow_protocol_2_5.md) defines the likelihood,
BIC, multiplicity constraints, and output contracts.

## Install and run

Requires Python 3.10+ and a C++17 compiler. Run the complete workflow from a JSON
plan; see the [runnable example](examples/README.md):

```sh
python -m pip install .
clipp2 workflow --plan PLAN.json --output-dir OUTPUT
```

This is the only command. Each run starts Stage 1 and Stage 2 afresh; saved fits
cannot be supplied for replay. The plan uses schema `clipp2.workflow.plan.v7`,
protocol `2.5`, `input_path`, `workflow_config`, and optional `run_manifest_id`.
Input paths are relative to the plan. Saved-fit and build-manifest fields are
rejected. Use a new output directory for each run.

Outputs include native Stage 1/2 provenance records, clonal-reference merging,
segment caps, reference vectors, method-specific scores, MAP tie sets and raw diagnostics.
Marginal assignment also reports cluster posterior probabilities; profile does not.
`all_original_mutations.tsv` retains every mutation and its final assignment or
unassigned status. Eligible targets without informative reads have a `prior_only` vector
and no data-supported final label; all-impossible candidates instead yield
`model_incompatible` with no posterior.

Stage 3 defaults to **marginal** likelihood. Set
`workflow_config.stage_three_assignment_method` to `"profile"` to maximize the
joint binomial likelihood over admissible multiplicity vectors. The method is
bound in the saved Stage 3 bundle. Profile ties retain the baseline when it is a
maximizer; otherwise the lowest label wins. Profile scores are not cluster
posteriors.

Stage 1–2 fits require **CPU** and **`fit_multiplicity="marginal"`**.
Requests for CUDA or profile fitting fail
explicitly; there is no silent CPU fallback or A-only substitution. CPU execution
requires neither CUDA nor PyTorch. Stage 3's separate **profile assignment**
remains supported. Better likelihood or BIC does not establish better biological
accuracy or CUDA qualification. See the
[numerical contract](docs/workflow_protocol_2_5.md#joint-likelihood-and-numerical-implementation).

`workflow_config.fit.max_multiplicity_difference` defaults to `1`; set it to
`null` for the independent regional model. Only these settings are implemented.
A region participates according to its resolved positive-depth observation
mask, including imputed observations, even
when its fitted CCF is zero. The corresponding public multiplicity is then NA
and it cannot donate a Stage 2 cap. This fixed latent support is stronger than
a restriction limited to mutation-bearing regions.

## Inputs and outputs

Set the plan's `input_path` to a canonical TSV (gzip supported) or a tumor
directory with `snv.txt`, `cna.txt`, and `purity.txt` in each region folder.
See the [runnable examples](examples/README.md).

| Canonical TSV fields | Meaning |
| --- | --- |
| `mutation_id`, `sample_id` | Mutation and region identifiers |
| `alt_count`, `ref_count`, `count_observed` | Counts and explicit observation status |
| `purity`, `normal_cn` | Purity in `(0, 1]` and nonnegative integer normal CN |
| `segment_id`, `cn_state_id`, `cn_state_fraction` | Genuine segment mapping and one state with fraction 1 |
| `allele_a_cn`, `allele_b_cn` | Integer major/minor CN, with major ≥ minor |
| `segment_cn_states` (optional) | JSON list preserving all segment CN states and their tumor-cell fractions |

Regional `cna.txt` may give multiple states for an exact chromosome/start/end
interval by adding `cn_state_id` and `cn_state_fraction`. Fractions describe
**all tumor cells** and must sum to 1 within the interval. The state with the
largest `(major_cn, minor_cn)` pair, compared in that order, supplies the
representative CN for the existing likelihood and the maximum major-CN clamp.
The full list is retained in the canonical `segment_cn_states` field. Canonical
TSV inputs preserve their explicit `allele_a_cn`/`allele_b_cn` for the likelihood
and major-CN clamp, independently of the states in that list. See the
[regional input example](examples/README.md) for the schema.

Within each interval, state IDs must be unique, fractions finite and nonnegative,
and CN integers satisfying `0 <= minor_cn <= major_cn` and
`total_cn = major_cn + minor_cn`. States must agree on normal CN and provenance
flags; distinct intervals cannot overlap. A mixed segment may include `(0,0)`,
but its representative major CN must be at least 1. Single-state CNA tables
without the two state columns use fraction 1.

For ALT reads `a`, total depth `d`, supplied purity `p`, and original input major
CN `M`, the requested Stage 1–2 lower bound is

```text
Cbar = sum(cn_state_fraction * (major_cn + minor_cn))
lower = max(1, min(M, floor((a / d) / p * (p * Cbar + 2 * (1 - p)))))
```

With one supplied state, `Cbar` is its major plus minor CN. The estimator uses
normal-cell total CN **2**, rounds down, and has no fixed cap of 4. Zero ALT,
missing reads, and zero depth receive lower bound 1. It does not divide by
mutation CCF, so it can underestimate multiplicity for subclonal mutations.
`multiplicity_lower_bounds.tsv` records these raw estimates. Over each mutation's
informative regions, let `B=min(M)`. Joint mode retains `L*=min(lower,B+1)`,
sets `q=max(1,max(L*)-1)`, and uses `U*=max(raw_cap,L*,q)`; Stage 1's raw cap is
`M`. This prioritizes original CN ceilings, the joint range rule, read floors,
then donor caps. Independent mode retains the original floor/cap rule.
Stage 3 restores `1..M` together with the same joint range rule.

CN and normal-CN provenance must be explicit. Regional files may supply
`normal_cn` in their CNA table; an omitted normal CN requires an explicit
`"normal_cn_policy": "diploid"` in `workflow_config`. Non-diploid normal CN
remains supported.

Missing pairs (`count_observed=0`, or an SNV absent from a region's list) default
to **ALT=0 and REF=the region's median total read count** among eligible,
originally observed positive-depth pairs (half-read ties round up). This happens
automatically in the workflow, including when an explicitly missing row
contains counts. Medians are computed separately for each region, before
imputation or fitting, and reused by all stages. If a region has no usable
observed depth, input validation fails. This is an **absence
assumption**, not measured evidence: raw rows and imputation records are retained
in workflow catalogs. Explicit exclusions and observed zero-depth pairs stay
unchanged. To keep missing pairs uninformative, set
`"missing_counts": "unobserved"` in `workflow_config`.

Each native fitting stage publishes mutation clusters, cluster centers, and
regional multiplicity calls. Output schema **17** retains every
positive mixture component in the centers table, including `cluster_size=0`
for components without MAP members. Component count and occupied MAP-group
count are distinct; BIC uses the former. Fitting uses the original CN denominator
even when multiplicity support is restricted. Stage 3 restores that original
support for every fitted mutation. An explicit
`workflow_config.fit.max_major_cn` selects a separately identified restricted
population; the default retains all valid integer CN. Input validation rejects
missing CN/purity and mapping problems without filling absent CN/purity or
modifying source files. Stage 1/2 records are provenance, not replay inputs.

## Contact

Submit an issue or email yding4@mdanderson.org.
