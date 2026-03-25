# Multi_Region_CliPP (CliPP2)

`Multi_Region_CliPP` is a research repository for **multi-region subclonal reconstruction** from sequencing-derived mutation counts, copy-number information, and tumor purity estimates. The codebase centers on a **SCAD-penalized ADMM** formulation, with a GPU-oriented implementation that uses conjugate-gradient updates, sparse/vectorized operations, and newer acceleration ideas such as mixed precision and adaptive weighting.

This repository is best understood as a **research codebase** rather than a packaged software product: it contains the main solver, a coreset-based variant, model-selection utilities, simulation scripts, post-processing experiments, and exploratory notebooks in one place.

---

## What this repository contains

At a high level, the repository supports the following workflow:

1. **Read multi-region input data** for the same tumor sample.
2. **Construct mutation-by-region tensors** from SNV counts, CNA values, and purity.
3. **Estimate subclonal structure** using a SCAD-penalized ADMM solver on the logistic scale.
4. **Post-process and summarize** mutation clusters and estimated cellular prevalence values.
5. **Benchmark or extend** the method using simulation, coreset approximation, model selection, and learned post-processing modules.

---

## Main ideas behind the method

The solver is designed to cluster SNVs that share similar frequency patterns across regions. In broad terms, the implementation:

- works in a **multi-sample / multi-region** setting,
- uses a **pairwise-difference penalty** to encourage mutations from the same clone to merge,
- operates through an **ADMM** optimization scheme,
- uses **SCAD** rather than a simple convex fusion penalty,
- includes a **GPU-oriented implementation** to avoid large dense CPU-side linear algebra,
- and includes an **adaptive weighting** strategy in the core solver to handle heterogeneous noise and improve separation of nearby subclones.

For theory and design notes, see [CliPP2.pdf](./CliPP2.pdf) and [gpu_implementation.pdf](./gpu_implementation.pdf).

---

## Repository layout

The current repository tree is organized around a small core library plus a number of research scripts:

```text
Multi_Region_CliPP/
├── clipp2/
│   ├── core.py
│   ├── core_coreset.py
│   └── preprocess.py
├── input_files/
├── simulation/
├── CliPP2.pdf
├── CliPP2.py
├── CliPP2_coreset.py
├── README.md
├── clipp2_model_selection.py
├── clipp2_model_selection_2.py
├── coreset_test.ipynb
├── coreset_time_comparison.png
├── gpu_implementation.pdf
├── make_data_clipp2.py
├── runner.py
├── simulation_data_generation_clipp2.py
├── simulation_data_generation_clipp2_tree.py
├── simulation_py_clone_vi.ipynb
├── simulation_py_clone_vi_res.ipynb
├── test_clipp2_postprocessing.py
├── test_clipp2_postprocessing_plot.py
└── train_clipp2_postprocessing.py
```

### Suggested reading order

For a new user, the most useful entry points are:

- [`CliPP2.py`](./CliPP2.py): top-level end-to-end pipeline script.
- [`clipp2/core.py`](./clipp2/core.py): main optimization logic.
- [`clipp2/preprocess.py`](./clipp2/preprocess.py): input parsing and tensor construction.
- [`CliPP2_coreset.py`](./CliPP2_coreset.py): coreset-based approximation path.
- [`clipp2_model_selection.py`](./clipp2_model_selection.py) and [`clipp2_model_selection_2.py`](./clipp2_model_selection_2.py): tuning / model-selection utilities.
- [`simulation_data_generation_clipp2.py`](./simulation_data_generation_clipp2.py) and [`simulation_data_generation_clipp2_tree.py`](./simulation_data_generation_clipp2_tree.py): simulation data generation.
- [`train_clipp2_postprocessing.py`](./train_clipp2_postprocessing.py), [`test_clipp2_postprocessing.py`](./test_clipp2_postprocessing.py), and [`test_clipp2_postprocessing_plot.py`](./test_clipp2_postprocessing_plot.py): experimental post-processing pipeline.

---

## Expected input layout

The repository is organized around a directory of region-specific inputs. A typical layout is:

```text
input_files/
├── region1/
│   ├── snv.txt
│   ├── cna.txt
│   └── purity.txt
├── region2/
│   ├── snv.txt
│   ├── cna.txt
│   └── purity.txt
└── region3/
    ├── snv.txt
    ├── cna.txt
    └── purity.txt
```

Conceptually:

- `snv.txt` stores mutation-level read count information,
- `cna.txt` stores copy-number related information,
- `purity.txt` stores region-level tumor purity.

The preprocessing code combines these region-specific files, aligns loci across regions, and converts them into the tensor objects required by the ADMM solver.

---

## Installation

This repository is currently used as a **source-code checkout**, not as a published Python package.

### 1. Clone the repository

```bash
git clone https://github.com/yuadamding/Multi_Region_CliPP.git
cd Multi_Region_CliPP
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Minimal dependencies for the main solver path are:

```bash
pip install numpy pandas scipy torch
```

Optional packages for notebooks and plotting:

```bash
pip install jupyter matplotlib
```

### Hardware note

A CUDA-capable GPU is strongly recommended for the main high-performance pipeline. Some scripts in the repository are clearly written with GPU execution in mind.

---

## Running the standard pipeline

A typical top-level run is organized around [`CliPP2.py`](./CliPP2.py). A representative command is:

```bash
python CliPP2.py \
  --input_dir input_files \
  --output_dir output \
  --Lambda 0.1 \
  --subsample_rate 1.0 \
  --dtype float64
```

At a high level, this script:

1. loads all regions from the input directory,
2. constructs the read-count / copy-number tensors,
3. computes an empirical initialization,
4. runs the CLiPP2 solver,
5. and writes a tabular result file.

### Output

The standard pipeline writes a result table named:

```text
output/<input_dir>/clipp2_result.tsv
```

The result table is intended to contain one row per mutation-region pair, with fields such as:

- `chromosome_index`
- `position`
- `region`
- `label`
- `phi`
- `phi_hat`
- `dropped`

Here:

- `label` is the inferred cluster assignment,
- `phi` is the fitted cellular prevalence estimate,
- `phi_hat` is the raw empirical estimate before final clustering/post-processing,
- `dropped` indicates whether a mutation was excluded by subsampling.

---

## Additional workflows in this repository

### 1. Coreset-based acceleration

The repository contains a separate coreset path:

- [`CliPP2_coreset.py`](./CliPP2_coreset.py)
- [`clipp2/core_coreset.py`](./clipp2/core_coreset.py)
- [`coreset_test.ipynb`](./coreset_test.ipynb)
- [`coreset_time_comparison.png`](./coreset_time_comparison.png)

This part of the codebase is useful when studying whether a reduced representative subset can speed up large reconstruction problems while retaining acceptable clustering quality.

### 2. Model selection / tuning

The repository includes dedicated scripts for model-selection experiments:

- [`clipp2_model_selection.py`](./clipp2_model_selection.py)
- [`clipp2_model_selection_2.py`](./clipp2_model_selection_2.py)

These are the first places to inspect if you want to tune regularization strength or compare solutions across multiple penalty levels.

### 3. Simulation

Simulation support is included through:

- [`simulation_data_generation_clipp2.py`](./simulation_data_generation_clipp2.py)
- [`simulation_data_generation_clipp2_tree.py`](./simulation_data_generation_clipp2_tree.py)
- the [`simulation/`](./simulation) directory

These scripts are useful for controlled benchmarking, recovery studies, and method development.

### 4. External-method comparison and notebooks

The notebooks

- [`simulation_py_clone_vi.ipynb`](./simulation_py_clone_vi.ipynb)
- [`simulation_py_clone_vi_res.ipynb`](./simulation_py_clone_vi_res.ipynb)

suggest that the repository also contains comparison workflows against alternative subclonal inference tools.

### 5. Learned post-processing

The repository also includes an experimental learned post-processing path:

- [`train_clipp2_postprocessing.py`](./train_clipp2_postprocessing.py)
- [`test_clipp2_postprocessing.py`](./test_clipp2_postprocessing.py)
- [`test_clipp2_postprocessing_plot.py`](./test_clipp2_postprocessing_plot.py)

These scripts appear intended for training, evaluating, and visualizing a post-processing component on top of the core reconstruction output.

---

## Practical notes for users

### This is a research repository

The repository contains multiple overlapping workflows and experimental branches of the method. That is useful for development, but it also means:

- not every top-level script is part of one unified command-line interface,
- there are both standard and experimental solver variants,
- and reproducible usage is best done by pinning a specific commit and recording the exact script, arguments, hardware, and dependency versions used.

### Recommended reproducibility record

For any analysis, record at least:

- repository commit hash,
- the script you ran,
- the value of `Lambda` or the lambda sequence explored,
- whether coreset approximation was used,
- random seed,
- PyTorch version,
- GPU model / CUDA version,
- and the exact input cohort or simulation setting.

---

## Documentation

For methodological details, see:

- [CliPP2.pdf](./CliPP2.pdf)
- [gpu_implementation.pdf](./gpu_implementation.pdf)

These files are the best starting point for understanding the mathematical formulation and the GPU implementation choices beyond the script-level code.

---

## Release status

A tagged release is available in the repository (`v1.0.0`).

---

## Contact

**Yu Ding, Ph.D.**  
Wenyi Wang Lab, MD Anderson Cancer Center  
Email: `yding4@mdanderson.org`  
Email: `yu.adam.ding@gmail.com`

---

## Citation

If this repository supports your research, please cite the corresponding paper, preprint, or repository release used in your analysis, and include the exact commit or tag for reproducibility.
