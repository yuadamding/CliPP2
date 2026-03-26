# CliPP2

CliPP2 is a research codebase for **single-region and multi-region subclonal reconstruction** from bulk sequencing data. The current implementation fits mutation cellular prevalence profiles using a **binomial likelihood**, supports **major/minor copy-number multiplicity candidates**, constructs a **mutation k-nearest-neighbor graph**, and performs **graph-fused clustering** through a unified EM/ADMM fitting workflow.

This README is written against the **current default-branch layout** of the repository, which is package-based and organized into `core/`, `io/`, `metrics/`, `runners/`, and `sim/`, with `cli.py` and `__main__.py` as the main command-line entry points.

---

## Repository overview

At the top level, the latest repository exposes the following structure:

```text
CliPP2/
├── core/
│   ├── __init__.py
│   ├── graph.py
│   └── model.py
├── io/
│   ├── __init__.py
│   ├── conversion.py
│   └── data.py
├── metrics/
├── runners/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── model_selection.py
│   ├── outputs.py
│   ├── pipeline.py
│   ├── selection.py
│   └── settings.py
├── sim/
│   ├── __init__.py
│   ├── generation.py
│   └── workflows.py
├── __init__.py
├── __main__.py
├── _version.py
└── cli.py
```

The package exports the main solver and workflow interfaces from four subsystems:

- `core`: graph construction and EM/ADMM fitting
- `io`: per-patient TSV loading and simulation-to-TSV conversion
- `metrics`: evaluation utilities
- `runners`: end-to-end fitting, benchmarking, model selection, and settings recommendation
- `sim`: simulation generation workflows

---

## Main capabilities

The current codebase supports the following tasks:

1. **Load a single per-patient TSV file** containing mutation-by-sample observations.
2. **Process an input directory of patient TSV files** and write per-patient outputs plus a cohort summary.
3. **Benchmark on simulation cohorts**, including stratified simulation benchmarking.
4. **Convert the older simulation-folder layout** into the TSV format expected by the current pipeline.
5. **Select regularization settings** using explicit lambda grids or automatic strategies.

The command-line interface exposes options for:

- a single patient file (`--input-file`),
- a directory of patient files (`--input-dir`),
- simulation benchmarking (`--benchmark-simulation`),
- lambda-grid control (`--lambda-grid`, `--lambda-grid-mode`),
- graph construction (`--graph-k`),
- solver controls for EM, ADMM, and conjugate gradient,
- warm starts,
- Bayesian or automatic settings selection,
- and output control.

---

## Input format

### Current primary input unit: one TSV per patient

The current pipeline expects **one TSV file per patient**. The loader requires these columns:

- `mutation_id`
- `sample_id`
- `ref_counts`
- `alt_counts`
- `major_cn`
- `minor_cn`

It also requires a purity column. The accepted names are:

- `purity`, or
- `tumour_content`

If `normal_cn` is absent, the loader fills it with `2.0`. If `has_cna` is absent, the code will also accept `cna_observed` when present. Internally, the loader constructs mutation-by-sample matrices for alternate counts, total counts, purity, major copy number, minor copy number, normal copy number, scaling, initial cellular prevalence, and multiplicity masks.

### Minimal example schema

```text
mutation_id    sample_id    ref_counts    alt_counts    normal_cn    major_cn    minor_cn    has_cna    purity
mut1           sampleA      45            12            2            2           1           1          0.72
mut1           sampleB      38            10            2            2           1           1          0.68
mut2           sampleA      60            4             2            1           1           0          0.72
mut2           sampleB      55            3             2            1           1           0          0.68
```

---

## Converting older simulation folders to TSV

The `io/conversion.py` workflow exists for the older directory layout where each patient contains multiple sample subdirectories, each with:

- `snv.txt`
- `cna.txt`
- `purity.txt`

The converter:

1. builds a union mutation catalog across samples,
2. maps CNA segments to mutation positions,
3. merges SNV counts, CNA states, and purity,
4. fills missing read counts with Jeffreys-style pseudo-counts,
5. fills missing CNA states with diploid defaults,
6. and writes a single per-patient TSV file.

### Example conversion command

```bash
python -m CliPP2.io.conversion \
  --input-root CliPP2Sim \
  --output-root CliPP2Sim_TSV
```

---

## Running the package

### Recommended invocation

Because the repository is currently organized as a Python package with `__main__.py`, the cleanest interface is to run it as a module:

```bash
python -m CliPP2 --input-dir CliPP2Sim_TSV --outdir results
```

This command processes all `*.tsv` files in the input directory and writes a cohort summary plus per-patient outputs.

### Run one patient file

```bash
python -m CliPP2 \
  --input-file CliPP2Sim_TSV/patient_001.tsv \
  --outdir results
```

### Run a simulation benchmark

```bash
python -m CliPP2 \
  --input-dir CliPP2Sim_TSV \
  --simulation-root CliPP2Sim \
  --benchmark-simulation \
  --outdir benchmark_results
```

### Example with explicit model-selection controls

```bash
python -m CliPP2 \
  --input-dir CliPP2Sim_TSV \
  --outdir results \
  --lambda-grid 0.0,0.01,0.05,0.1,0.2 \
  --graph-k 8 \
  --settings-profile auto \
  --device auto
```

---

## Important command-line options

### Inputs and outputs

- `--input-dir`: directory containing per-patient TSV files
- `--input-file`: optional single-patient TSV file
- `--outdir`: output directory
- `--simulation-root`: original simulation folder root used for evaluation

### Model-selection controls

- `--lambda-grid`: explicit comma-separated lambda values
- `--lambda-grid-mode`: automatic lambda-grid template; one of:
  - `standard`
  - `dense`
  - `dense_no_zero`
  - `coarse_no_zero`
- `--settings-profile`: one of:
  - `manual`
  - `auto`
  - `bayes`
  - `legacy_auto`

### Graph and solver controls

- `--graph-k`: number of neighbors in the mutation kNN graph
- `--device`: `auto`, `cuda`, or `cpu`
- `--em-max-iter`
- `--admm-max-iter`
- `--inner-steps`
- `--cg-max-iter`
- `--cg-tol`
- `--admm-rho`
- `--em-tol`
- `--admm-tol`
- `--fused-tol`
- `--center-merge-tol`
- `--major-prior`

### Benchmark and search options

- `--benchmark-simulation`
- `--reps-per-scenario`
- `--n-mean-values`
- `--bo-max-evals`
- `--bo-init-points`
- `--bo-random-seed`
- `--disable-warm-start`
- `--skip-patient-outputs`
- `--max-files`
- `--verbose`

---

## What the pipeline writes

For each processed patient, `runners/outputs.py` writes four main TSV outputs:

1. **`{patient_id}_mutation_clusters.tsv`**
   - one row per mutation
   - includes cluster labels, cluster sizes, and per-sample clustered CP values

2. **`{patient_id}_cluster_centers.tsv`**
   - one row per cluster
   - includes cluster sizes and per-sample cluster-center CP values

3. **`{patient_id}_cell_multiplicity.tsv`**
   - one row per mutation-sample pair
   - includes fitted `phi`, copy numbers, major/minor multiplicity calls, and related posterior summaries

4. **`{patient_id}_lambda_search.tsv`**
   - the lambda-path search record used for model selection

If simulation truth is available, the pipeline also writes:

5. **`{patient_id}_simulation_eval.tsv`**
   - ARI
   - CP RMSE
   - multiplicity accuracy
   - true and estimated cluster counts
   - evaluation sample size summaries

At the cohort level, `run_directory()` writes:

- **`single_stage_summary.tsv`**

This summary includes selected lambda, BIC, log-likelihood, number of clusters, graph edge count, patient regime summaries, solver convergence, and simulation metrics when available.

---

## Internal package map

### `core/`

- `graph.py`: mutation graph construction through a kNN graph on CP profiles
- `model.py`: fit options, fit results, and the main single-stage EM fitting routine

### `io/`

- `data.py`: patient TSV schema loading and initialization logic
- `conversion.py`: conversion from simulation folder layout to TSV

### `runners/`

- `pipeline.py`: one-file and directory-level processing
- `outputs.py`: result-table generation
- `model_selection.py`: candidate fitting and best-model selection
- `selection.py`: BIC utilities and default lambda grids
- `settings.py`: regime summaries and recommended settings
- `benchmark.py`: simulation and cohort benchmarking workflows

### `sim/`

- `generation.py`: simulation-grid configuration and generation
- `workflows.py`: packaged simulation workflows and conversion helpers

---

## Installation notes

This repository does not currently expose standard packaging metadata in the visible top-level tree, so the simplest usage pattern is a **source checkout** in an environment that already has the required dependencies.

Recommended core dependencies:

```bash
pip install numpy pandas torch
```

You may also need scientific Python utilities used by the fitting and benchmarking stack, depending on which workflows you run.

---

## Practical usage notes

- This is a **research codebase**, not a polished end-user package.
- The package now uses a **per-patient TSV input abstraction**, which is a major shift from the older region-folder entry point.
- The default CLI is broad and exposes many tuning knobs; for routine use, start with the automatic settings profile and the default lambda-grid mode.
- If you are running simulation studies, keep both the converted TSV directory and the original simulation root, because evaluation hooks use the original simulation structure when available.

---

## Reproducibility checklist

For any reported result, record at least:

- repository URL and commit hash
- input TSV files used
- original simulation root, if applicable
- lambda grid or lambda-grid mode
- graph k
- settings profile
- EM / ADMM / CG iteration limits and tolerances
- device (`cpu` or `cuda`)
- random seed for Bayesian optimization, if used

---

## Suggested citation language

If you use this repository in research, cite the corresponding paper, preprint, or repository release, and report the exact commit hash used for the analysis.
