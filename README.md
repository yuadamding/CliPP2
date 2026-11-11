# CliPP2

CliPP2 implements a three-stage research method for joint mutation clustering
across tumor regions. It estimates cancer cell fractions (CCFs) and integer
multiplicities under **one allele-specific tumor copy-number state per
mutation–region observation**. Mixed tumor copy-number states are unsupported.

Protocol **2.3** chooses a provisional Stage 1 source, then fits the remainder
with caps from the minimum upper endpoint of native multiplicity MAP sets.
Stage 3 restores original support and reassesses every fitted mutation against
one fixed all-one clonal reference and every Stage 2 center. The selected Stage 1
source maps to that clonal reference; its native estimate remains provenance,
not a competing cluster. Exact Stage 2 duplicates are consolidated before equal priors.
Confidence, ties, coverage, residuals and distance do not veto defined estimates.

Cluster limits default to **10 for Stage 1 and 20 for Stage 2**. Valid integer CN
has no automatic eligibility ceiling. Explicit input exclusions, model-domain
checks and numerical verification remain. Package **2.0.0** and scientific
protocol **2.3** have separate identities; versioned contracts distinguish the
current single-clonal-reference policy from earlier reference inventories.

## Install and run

Fitting requires Python 3.10+ and a C++17 compiler. CPU is the default backend:

```sh
python -m pip install .
clipp2 workflow --plan examples/workflow_protocol23.plan.json --output-dir results/workflow23
```

The example plan declares one workflow configuration without acceptance thresholds.
Outputs include native Stage 1/2 fits, provisional-source provenance, exact reference consolidation, segment caps,
reference vectors, complete probability vectors, MAP tie sets and raw diagnostics.
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

CUDA requires an explicit build and device selection:

```sh
CLIPP2_BUILD_CUDA=1 python -m pip install .
clipp2 workflow --help
```

Set the workflow plan's fit device to `cuda`, or use `clipp2 fit --device cuda`.
CUDA builds require `nvcc`; execution requires a compatible NVIDIA GPU and driver.
CPU execution requires neither CUDA nor PyTorch.

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

CN and normal-CN provenance must be explicit. Regional files may supply
`normal_cn` in their CNA table; an omitted normal CN requires an explicit
`--normal-cn-policy diploid` for fitting, or `normal_cn_policy` in the workflow
plan. Non-diploid normal CN remains supported.

Missing pairs (`count_observed=0`, or an SNV absent from a region's list) default
to **ALT=0**. REF uses the supplied positive read depth, otherwise that region's
median positive depth among eligible observed pairs (half-read ties round up).
Without either depth source, input validation fails. This is an **absence
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

- [Workflow Protocol 2.3](docs/workflow_protocol_2_3.md)
- [Historical Protocol 2.2](docs/workflow_protocol_2_2.md)
- [Simulation design and execution](docs/simulation.md)
- [Evaluation definitions](docs/simulation_evaluation.md)

## Contact

Submit an issue or email yding4@mdanderson.org.
