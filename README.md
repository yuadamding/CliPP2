# CliPP2

CliPP2 implements a three-stage research method for joint mutation clustering
across tumor regions. It estimates cancer cell fractions (CCFs) and integer
multiplicities using **one representative allele-specific tumor copy-number
state per mutation–region observation** in the likelihood. Supplied CN-state
fractions additionally determine the read-based multiplicity lower bounds in
Stages 1 and 2; they do not introduce a mixed-state likelihood.

Protocol **2.4** chooses a provisional Stage 1 source, then fits the remainder
with caps from the minimum upper endpoint of native multiplicity MAP sets.
Both fitting stages restrict candidates below by the floor-based estimate
described below. A conflicting Stage 2 cap is raised to that mutation's lower
bound, making the lower bound its sole candidate.
Stage 3 restores original support and reassesses every fitted mutation against
one fixed all-one clonal reference and every Stage 2 center. The selected Stage 1
source maps to that clonal reference; its native estimate remains provenance,
not a competing cluster. Exact Stage 2 duplicates are consolidated before equal priors.
Confidence, ties, coverage, residuals and distance do not veto defined estimates.

Cluster limits default to **10 for Stage 1 and 20 for Stage 2**. Valid integer CN
has no automatic eligibility ceiling. Explicit input exclusions, model-domain
checks and numerical verification remain. Package **2.0.0** and scientific
protocol **2.4** have separate identities; versioned contracts distinguish the
current single-clonal-reference policy from earlier reference inventories.

## Install and run

Fitting requires Python 3.10+ and a C++17 compiler. CPU is the default backend:

```sh
python -m pip install .
clipp2 workflow --plan examples/workflow_protocol24.plan.json --output-dir results/workflow24
```

The example plan declares one workflow configuration without acceptance thresholds.
Outputs include native Stage 1/2 fits, provisional-source provenance, exact reference consolidation, segment caps,
reference vectors, method-specific scores, MAP tie sets and raw diagnostics.
Marginal assignment also reports cluster posterior probabilities; profile does not.
`all_original_mutations.tsv` retains final membership and unchanged or unassigned
mutations. Eligible targets without informative reads have a `prior_only` vector
and no data-supported final label; all-impossible candidates instead yield
`model_incompatible` with no posterior. Use a new output directory for each run.

| Operation | Command |
| --- | --- |
| Three-stage method | `clipp2 workflow --plan PLAN.json --output-dir OUTPUT` |
| Standalone native fit | `clipp2 fit --input-file INPUT.tsv --output-dir OUTPUT` |
| Input audit | `clipp2 audit-input --input-file INPUT.tsv --output AUDIT.json` |
| Frozen-reference reassignment | `clipp2 assign --bundle BUNDLE.json --output-dir OUTPUT` |
| Biological study | `clipp2-simulate run --cohort-settings SETTINGS.json --outdir OUTPUT` |

Stage 3 defaults to **marginal** likelihood. Set
`workflow_config.stage_three_assignment_method` to `"profile"` to maximize each
region’s binomial likelihood over original multiplicities. For a frozen request,
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
assignment method. See the [numerical contract and comparisons](docs/profile_fitting_marginal_selection.md).

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
[regional input example](examples/README.md) for the schema and validation.

For ALT reads `a`, total depth `d`, supplied purity `p`, and original input major
CN `M`, the Stage 1–2 lower bound is

```text
Cbar = sum(cn_state_fraction * (major_cn + minor_cn))
lower = max(1, min(M, floor((a / d) / p * (p * Cbar + 2 * (1 - p)))))
```

With one supplied state, `Cbar` is its major plus minor CN. The estimator uses
normal-cell total CN **2**, rounds down, and has no fixed cap of 4. Zero ALT,
missing reads, and zero depth receive lower bound 1. It does not divide by
mutation CCF, so it can underestimate multiplicity for subclonal mutations.
`multiplicity_lower_bounds.tsv` records the inputs and estimates; segment-cap
audits count cases where the lower bound overrides a cap. Stage 3 restores
`1..M`, including candidates below the read-based estimate.

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

## Simulation and checks

The exact supplied cohort requires **Python 3.11+ and NumPy 2.3.5**:

```sh
python -m pip install '.[simulation]'
clipp2-simulate run --cohort-settings CliPP2_phylogeny_cohort_settings.json.gz --workers 10 --outdir results/cohort
python tools/check_research.py --groups workflow biology evaluation study publication
```

The full design has 60 cases, 720 physical histories, and 2,640 datasets.
Development and main replicate counts are declared before execution; worker
count does not change the experiment. One fixed configuration runs throughout;
development is descriptive and poor finite performance does not block the main
study. Numerical verification remains mandatory. The deterministic checks above
are separate from biological performance assessment and full-cohort execution.

- [Workflow Protocol 2.4](docs/workflow_protocol_2_4.md)
- [Simulation design and execution](docs/simulation.md)
- [Evaluation definitions](docs/simulation_evaluation.md)

## Contact

Submit an issue or email yding4@mdanderson.org.
