# Multi_Region_CliPP

`Multi_Region_CliPP` is a research codebase for multi-region subclonal reconstruction from mutation read counts, copy-number states, and tumor purity. The post-2.0.1 codebase is organized as a Python package with a package CLI (`__main__.py` -> `cli.py`) and a modular layout built around `core/`, `io/`, `metrics/`, `runners/`, and `sim/`.

At a high level, the workflow is:

1. Load one per-patient TSV containing mutation-by-sample observations.
2. Construct an initialization and a mutation graph from the data.
3. Fit a single-stage EM + ADMM model with graph-fused penalization.
4. Search over a lambda grid and select the best solution by extended BIC.
5. Write mutation-level, cluster-level, cell-level, and optional simulation-evaluation outputs.

---

## Repository layout

```text
Multi_Region_CliPP/
├── core/
│   ├── graph.py
│   └── model.py
├── io/
│   ├── conversion.py
│   └── data.py
├── metrics/
│   └── evaluation.py
├── runners/
│   ├── benchmark.py
│   ├── multiregion_benchmark.py
│   ├── outputs.py
│   ├── pipeline.py
│   ├── selection.py
│   ├── settings.py
│   └── single_region_benchmark.py
├── sim/
│   ├── generation.py
│   └── workflows.py
├── __init__.py
├── __main__.py
└── cli.py
```

### What each module does

- `core/`
  - `graph.py`: mutation graph construction.
  - `model.py`: fitting logic, including `FitOptions`, `FitResult`, and the main EM/ADMM solver.

- `io/`
  - `data.py`: loads a per-patient TSV into a `PatientData` object and computes derived quantities such as `scaling`, `phi_upper`, and `phi_init`.
  - `conversion.py`: converts an older simulation-folder layout into the per-patient TSV layout expected by the current package.

- `metrics/`
  - `evaluation.py`: compares fitted results against simulation truth using metrics such as ARI, cellular-prevalence RMSE, and multiplicity accuracy.

- `runners/`
  - `pipeline.py`: main end-to-end fitting workflow for one file or a directory of files.
  - `selection.py`: lambda-grid construction and BIC scoring.
  - `settings.py`: simple regime-based auto-tuning of graph and selection settings.
  - `outputs.py`: writes result tables.
  - `benchmark.py`, `multiregion_benchmark.py`, `single_region_benchmark.py`: benchmarking workflows.

- `sim/`
  - synthetic-data generation and simulation workflows.

---

## Installation

This repository currently behaves like a source checkout rather than a packaged release on PyPI. The simplest setup is:

```bash
git clone https://github.com/yuadamding/Multi_Region_CliPP.git
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas torch scikit-learn
```

Optional extras for notebooks or plotting:

```bash
pip install jupyter matplotlib
```

### Important execution note

The repository root itself is the Python package. That means the intended CLI entry point is:

```bash
python -m Multi_Region_CliPP --help
```

Run that command **from the parent directory of the repository**, not by calling `python cli.py` inside the repository root.

Example:

```bash
# Suppose you cloned into /path/to/Multi_Region_CliPP
cd /path/to
python -m Multi_Region_CliPP --help
```

### Hardware note

The code supports `--device auto`, `--device cuda`, and `--device cpu`. If CUDA is available, `auto` will use the GPU; otherwise it falls back to CPU.

---

## Current input format

The current package expects **one TSV per patient**. Each row represents one mutation observed in one sample.

### Required columns

- `mutation_id`
- `sample_id`
- `ref_counts`
- `alt_counts`
- `major_cn`
- `minor_cn`

### Required purity column

You must provide **one** of the following:

- `purity`
- `tumour_content`

### Optional columns

- `normal_cn`  
  If omitted, it defaults to `2`.

- `has_cna` or `cna_observed`  
  Either can be used to indicate whether CNA information is available for that mutation-sample entry.

### Minimal example

```tsv
mutation_id	sample_id	ref_counts	alt_counts	major_cn	minor_cn	normal_cn	purity	has_cna
1:10583	patientA_sample1	92	11	1	1	2	0.73	1
1:10583	patientA_sample2	87	15	1	1	2	0.69	1
1:10611	patientA_sample1	51	8	2	1	2	0.73	1
1:10611	patientA_sample2	60	4	2	1	2	0.69	1
```

### Internal loading behavior

When a TSV is loaded, the code:

- builds mutation and sample index sets,
- forms mutation-by-sample matrices for counts and copy number,
- computes a scaling term from purity and copy number,
- derives an upper bound `phi_upper`,
- and initializes `phi_init` by comparing major-copy and minor-copy likelihoods.

---

## Converting legacy simulation folders to the new TSV format

The repository still contains a converter for the older folder layout where each patient has per-sample subdirectories with:

- `snv.txt`
- `cna.txt`
- `purity.txt`

The converter:

1. builds a mutation catalog across samples,
2. maps CNA segments to mutation coordinates,
3. writes one merged patient TSV,
4. fills missing counts with Jeffreys-style pseudo-counts,
5. fills missing CNA with diploid defaults (`major_cn = 1`, `minor_cn = 1`).

Run it as a module:

```bash
cd /path/to
python -m Multi_Region_CliPP.io.conversion \
  --input-root /path/to/CliPP2Sim \
  --output-root /path/to/CliPP2Sim_PyClone
```

Each converted patient is written as:

```text
<output-root>/<patient_id>.tsv
```

---

## CLI usage

The main CLI lives in `cli.py` and is exposed through:

```bash
python -m Multi_Region_CliPP
```

### 1. Run a directory of patient TSV files

```bash
cd /path/to
python -m Multi_Region_CliPP \
  --input-dir /path/to/patient_tsvs \
  --outdir /path/to/results
```

This will scan all `*.tsv` files in `--input-dir`, fit each patient, and write a cohort summary to:

```text
<outdir>/single_stage_summary.tsv
```

### 2. Run one patient file

```bash
cd /path/to
python -m Multi_Region_CliPP \
  --input-file /path/to/patient.tsv \
  --outdir /path/to/results
```

### 3. Run a simulation benchmark

```bash
cd /path/to
python -m Multi_Region_CliPP \
  --benchmark-simulation \
  --input-dir /path/to/CliPP2Sim_PyClone \
  --simulation-root /path/to/CliPP2Sim \
  --outdir /path/to/benchmark_results
```

This benchmark runner selects representative files per scenario, fits them, evaluates against simulation truth, and writes benchmark summary tables.

---

## Important CLI options

### Input/output

- `--input-dir`  
  Directory containing patient TSV files. Default: `CliPP2Sim_PyClone`

- `--input-file`  
  Path to a single patient TSV.

- `--outdir`  
  Output directory. Default: `multi_region_clipp_results`

- `--simulation-root`  
  Root directory containing simulation truth. Default: `CliPP2Sim`

### Lambda search

- `--lambda-grid`  
  Comma-separated custom lambda grid, for example:
  ```bash
  --lambda-grid 0.5,1,2,4,8
  ```

- `--lambda-grid-mode`  
  One of:
  - `standard`
  - `dense`
  - `dense_no_zero`
  - `coarse_no_zero`

  Default: `dense_no_zero`

### Graph and device

- `--graph-k`  
  Number of neighbors used to build the mutation graph. Default: `8`

- `--device`  
  One of:
  - `auto`
  - `cuda`
  - `cpu`

  Default: `auto`

### Optimization controls

- `--em-max-iter` (default `8`)
- `--admm-max-iter` (default `20`)
- `--cg-max-iter` (default `30`)
- `--em-tol` (default `1e-4`)
- `--admm-tol` (default `5e-3`)
- `--cg-tol` (default `1e-4`)
- `--admm-rho` (default `2.0`)
- `--major-prior` (default `0.5`)
- `--fused-tol` (default `1e-3`)
- `--center-merge-tol` (default `1e-1`)

### Selection and heuristics

- `--settings-profile`  
  `auto` or `manual`. Default: `auto`

- `--disable-warm-start`  
  Disables lambda-path warm starts.

- `--skip-patient-outputs`  
  Only produce high-level summaries.

- `--bic-df-scale`  
  Extended-BIC weight on cellular-prevalence degrees of freedom. Default: `10.0`

- `--bic-cluster-penalty`  
  Extended-BIC weight on cluster-count complexity. Default: `6.0`

### Benchmarking

- `--benchmark-simulation`
- `--reps-per-scenario`
- `--n-mean-values`
- `--max-files`

---

## Auto settings

When `--settings-profile auto` is used, the code summarizes each patient into a simple regime and chooses a recommended configuration based on sample count, depth, purity, and non-diploid rate.

Named auto profiles include:

- `strong_low_depth`
- `moderate_sparse_graph`
- `fast_low_purity`
- `balanced_high_dimension`
- `strong_default`

These profiles modify at least:

- `graph_k`
- `lambda_grid_mode`
- `bic_df_scale`
- `bic_cluster_penalty`
- `center_merge_tol`

If you need full manual control, use:

```bash
--settings-profile manual
```

and specify your own graph and lambda settings.

---

## What the fitting pipeline actually does

For one patient, `runners/pipeline.py` performs the following steps:

1. Load the patient TSV with `load_patient_tsv`.
2. Build a mutation graph from `data.phi_init`.
3. Construct a lambda grid if you did not provide one.
4. Fit one model per lambda using `fit_single_stage_em`.
5. Score each fit with:
   - classic BIC
   - extended BIC
6. Select the best fit by minimum extended BIC.
7. Optionally evaluate against simulation truth.
8. Write patient-level output files.

---

## Output files

For each patient, the pipeline can write:

### 1. Mutation clusters

```text
<outdir>/<patient_id>_mutation_clusters.tsv
```

Columns include:

- `mutation_id`
- `cluster_label`
- `cluster_size`
- one `phi_<sample_id>` column per sample

This is the easiest file to use if you want one cluster assignment per mutation.

### 2. Cluster centers

```text
<outdir>/<patient_id>_cluster_centers.tsv
```

Columns include:

- `cluster_label`
- `cluster_size`
- one `phi_<sample_id>` column per sample

This file gives the cluster-level cellular-prevalence profiles.

### 3. Cell / mutation-sample multiplicity output

```text
<outdir>/<patient_id>_cell_multiplicity.tsv
```

Columns include:

- `mutation_id`
- `sample_id`
- `cluster_label`
- `phi`
- `major_cn`
- `minor_cn`
- `major_probability`
- `major_call`
- `multiplicity_call`

This is the most detailed output if you want mutation-sample-level multiplicity calls.

### 4. Lambda-search trace

```text
<outdir>/<patient_id>_lambda_search.tsv
```

Columns include fields such as:

- `lambda`
- `bic`
- `classic_bic`
- `loglik`
- `penalized_objective`
- `n_clusters`
- `converged`
- `iterations`
- `device`

If simulation truth is available, the search table also includes evaluation metrics such as:

- `ARI`
- `cp_rmse`
- `multiplicity_accuracy`

### 5. Optional simulation evaluation

```text
<outdir>/<patient_id>_simulation_eval.tsv
```

Written only when simulation truth exists for that patient.

### 6. Cohort summary

When you run a directory, the code also writes:

```text
<outdir>/single_stage_summary.tsv
```

This includes, for each patient:

- selected lambda,
- BIC,
- number of clusters,
- graph size,
- settings profile,
- sample/mutation counts,
- purity/depth summaries,
- and optional evaluation metrics.

### 7. Benchmark summaries

When you run `--benchmark-simulation`, the code writes:

```text
<outdir>/benchmark_patients.tsv
<outdir>/benchmark_by_scenario.tsv
<outdir>/benchmark_global.tsv
```

---

## Programmatic use

The repository also exposes a small public API from `__init__.py`.

### Minimal example

```python
from Multi_Region_CliPP import (
    FitOptions,
    build_knn_graph,
    fit_single_stage_em,
    load_patient_tsv,
)

data = load_patient_tsv("/path/to/patient.tsv")
graph = build_knn_graph(data.phi_init, k=8, device="auto")

options = FitOptions(
    lambda_value=10.0,
    device="auto",
    em_max_iter=8,
    admm_max_iter=20,
    cg_max_iter=30,
)

fit = fit_single_stage_em(
    data=data,
    graph=graph,
    options=options,
    phi_start=data.phi_init.copy(),
)

print("clusters:", fit.n_clusters)
print("loglik:", fit.loglik)
print("converged:", fit.converged)
```

### Useful public symbols

From the package root you can import:

- `FitOptions`, `FitResult`
- `GraphData`, `build_knn_graph`, `fit_single_stage_em`
- `PatientData`, `load_patient_tsv`
- `ConversionConfig`, `convert_simulation_root`
- `evaluate_fit_against_simulation`
- `process_one_file`, `run_directory`, `run_simulation_benchmark`
- simulation helpers such as `generate_and_convert_simulation` and `run_simulation_grid`

---

## Selection details

The code provides both:

- **classic BIC**
- **extended BIC**

The default lambda grid depends on data scale and number of samples. The built-in modes are:

- `standard`
- `dense`
- `dense_no_zero`
- `coarse_no_zero`

The extended BIC augments the usual likelihood-based penalty with explicit penalties for:

- cellular-prevalence degrees of freedom,
- and cluster-count complexity.

If you are comparing tuning runs across datasets, save the full `*_lambda_search.tsv` output rather than only the selected fit.

---

## Benchmark expectations

The simulation benchmark expects patient IDs to follow a structured naming convention, for example something like:

```text
Nmean_trueK_purity_amprate_SnSamples_MnMutations_repR
```

More concretely, the parser expects IDs of the form:

```text
<N_mean>_<true_K>_<purity>_<amp_rate>_S<n_samples>_M<n_mutations>_rep<rep>
```

Use the benchmark runner only when your simulation files follow that naming convention and the corresponding truth directories are present under `--simulation-root`.

---

## Common pitfalls

### 1. Running the wrong entry point

Do **not** treat the current repository like the old flat-script layout. Prefer:

```bash
python -m Multi_Region_CliPP
```

### 2. Using the old folder-style input directly

The current main pipeline expects per-patient TSV files, not raw `snv.txt` / `cna.txt` / `purity.txt` folders. Convert the old layout first if needed.

### 3. Missing purity column

You must provide either:

- `purity`
- or `tumour_content`

### 4. Confusing mutation-level and mutation-sample-level outputs

- `*_mutation_clusters.tsv` is mutation-level.
- `*_cell_multiplicity.tsv` is mutation-sample-level.

### 5. Assuming auto settings are fixed across all data regimes

In `auto` mode, graph size, lambda-grid mode, and BIC penalties are adjusted based on the data regime. Use `manual` mode if you need reproducible hand-tuned settings across experiments.

---

## Recommended reading order for developers

If you want to understand the post-2.0.1 codebase quickly, start here:

1. `cli.py`
2. `runners/pipeline.py`
3. `io/data.py`
4. `core/graph.py`
5. `core/model.py`
6. `runners/outputs.py`
7. `runners/selection.py`
8. `metrics/evaluation.py`
9. `io/conversion.py`

---

## Status

This is best treated as a research repository under active development rather than a polished end-user package. The code already exposes a coherent package CLI and public API, but you should still expect to inspect scripts and module internals when adapting it to new datasets or benchmarking settings.
