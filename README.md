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
checks and numerical verification remain. Stage 1–2 preserves the tree-search
winner, then repeatedly explores general-partition births, transfers of both
signs, merges and atomic split–merge moves under the same refitter and BIC.
The bounded search reports budget or proposal exhaustion; neither certifies
local or global optimality. It does not run exhaustive membership audits.
Package **2.2.0** implements scientific protocol **2.5**. The
[protocol specification](docs/workflow_protocol_2_5.md) defines the likelihood,
BIC, multiplicity constraints, and saved-state contracts.

## Install and run

Fitting requires Python 3.10+ and a C++17 compiler. CPU is the default backend:

```sh
python -m pip install .
clipp2 fit --input-dir examples/tumor --max-clusters 3 --output-dir results/example-fit
```

The [examples](examples/README.md) also show the plan for a complete three-stage
workflow. Its outputs include native Stage 1/2 fits, provisional-source provenance, clonal-reference merging, segment caps,
reference vectors, method-specific scores, MAP tie sets and raw diagnostics.
Marginal assignment also reports cluster posterior probabilities; profile does not.
`all_original_mutations.tsv` retains every mutation and its final assignment or
unassigned status. Eligible targets without informative reads have a `prior_only` vector
and no data-supported final label; all-impossible candidates instead yield
`model_incompatible` with no posterior. Use a new output directory for each run.

| Operation | Command |
| --- | --- |
| Three-stage method | `clipp2 workflow --plan PLAN.json --output-dir OUTPUT` |
| Standalone native fit | `clipp2 fit --input-file INPUT.tsv --output-dir OUTPUT` |
| Input audit | `clipp2 audit-input --input-file INPUT.tsv --output AUDIT.json` |
| Frozen-reference reassignment | `clipp2 assign --bundle BUNDLE.json --output-dir OUTPUT` |

An opt-in **CPU-only research estimator**, separate from the workflow, jointly
fits mixture centers and weights to observed marginal likelihood:

```sh
clipp2 experiment-mixture --catalog RUN/original_catalog.json \
  --accepted-fit RUN/stage_two_accepted.json --output experiment.json
```

It preserves the saved population and support, uses soft responsibilities, and
reports positive-weight components separately from occupied MAP labels. Its BIC
counts positive components. The experimental JSON cannot replace an accepted
workflow fit; no hard refit is performed after MAP labeling. Better likelihood
or BIC does not establish better biological accuracy or CUDA qualification.

Stage 3 defaults to **marginal** likelihood. Set
`workflow_config.stage_three_assignment_method` to `"profile"` to maximize the
joint binomial likelihood over admissible multiplicity vectors. For a frozen request,
use `clipp2 assign --assignment-method profile`. A method change creates a new request identity and saves the
executed bundle with its outputs. Profile ties retain the baseline when it is a
maximizer; otherwise the lowest label wins. Profile scores are not cluster
posteriors.

CUDA requires an explicit build and device selection:

```sh
CLIPP2_BUILD_CUDA=1 python -m pip install .
clipp2 workflow --help
```

Set the workflow plan's fit device to `cuda`, or use `clipp2 fit --device cuda`.
CUDA builds require `nvcc`; execution requires a compatible NVIDIA GPU and driver.
CPU execution requires neither CUDA nor PyTorch.

Stage 1–2 center fitting defaults to **marginal** likelihood. Opt into
`--fit-multiplicity profile` (or `workflow_config.fit.fit_multiplicity` in a plan)
to maximize integer-multiplicity likelihood while retaining marginal-mixture
BIC selection at the frozen fitted centers. This is independent of Stage 3's
assignment method. See the [numerical contract](docs/workflow_protocol_2_5.md#joint-likelihood-and-numerical-implementation).

`FitConfig.max_multiplicity_difference` defaults to `1`; use `None` in Python, `null` in JSON,
or `--max-multiplicity-difference none` for the independent regional model.
Only `1` and `None` are implemented. A region participates according to its
resolved positive-depth observation mask, including imputed observations, even
when its fitted CCF is zero. The corresponding public multiplicity is then NA
and it cannot donate a Stage 2 cap. This fixed latent support is stronger than
a restriction limited to mutation-bearing regions.

## Inputs and outputs

Use a canonical TSV (`--input-file`, gzip supported) or a tumor directory
(`--input-dir`) with `snv.txt`, `cna.txt`, and `purity.txt` in each region folder.
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
Explicit support TSVs are hard constraints and incompatible joint supports fail
validation. Stage 3 restores `1..M` together with the same joint range rule.

CN and normal-CN provenance must be explicit. Regional files may supply
`normal_cn` in their CNA table; an omitted normal CN requires an explicit
`--normal-cn-policy diploid` for fitting, or `normal_cn_policy` in the workflow
plan. Non-diploid normal CN remains supported.

Missing pairs (`count_observed=0`, or an SNV absent from a region's list) default
to **ALT=0 and REF=the region's median total read count** among eligible,
originally observed positive-depth pairs (half-read ties round up). This happens
automatically in fits and workflows, including when an explicitly missing row
contains counts. Medians are computed separately for each region, before
imputation or fitting, and reused by all stages. If a region has no usable
observed depth, input validation fails. This is an **absence
assumption**, not measured evidence: raw rows and imputation records are retained
in workflow catalogs. Explicit exclusions and observed zero-depth pairs stay
unchanged. To keep missing pairs uninformative, use `--missing-counts unobserved`
for a fit or `"missing_counts": "unobserved"` in the workflow config.

Standalone fitting publishes mutation clusters, cluster centers, and regional
multiplicity calls. Fitting uses the original CN denominator even when
multiplicity support is restricted. Stage 3 restores that original support for every fitted mutation. An explicit
`--max-major-cn` selects a separately identified restricted population; the
default retains all valid integer CN. Input audits report missing CN/purity and
mapping problems without filling absent CN/purity, and report read imputations
without modifying source files.

## Contact

Submit an issue or email yding4@mdanderson.org.
