# CliPP2

CliPP2 estimates cellular prevalence, clusters SNVs, and infers mutant-copy
multiplicity from single- or multi-region tumor sequencing data with
objective-faithful pairwise fusion.

## Install

```bash
pip install .
```

## Sole public input

CliPP2 accepts one tool-neutral tumor-directory format. All tabular files are
tab-delimited with a header, and genomic coordinates are 1-based and
inclusive. This is input schema version `2`; additional columns are ignored:

```text
tumor_name/
├── mutation_segments.tsv
├── cn_clone_profiles.tsv
├── cn_clone_fractions.tsv
├── purity.txt
├── region1/
│   ├── snv.txt
│   ├── cna.txt
│   └── purity.txt
├── region2/
│   ├── snv.txt
│   ├── cna.txt
│   └── purity.txt
└── ...
```

The required columns are:

| File | Required columns |
| --- | --- |
| `mutation_segments.tsv` | `mutation_id`, `segment_id`, `chromosome`, `position` |
| `cn_clone_profiles.tsv` | `cn_clone_id`, `segment_id`, `chromosome`, `start`, `end`, `allele_a_cn`, `allele_b_cn` |
| `cn_clone_fractions.tsv` | `sample_id`, `cn_clone_id`, `tumor_fraction` |
| root `purity.txt` | `sample_id`, `purity` |
| `regionN/snv.txt` | `chromosome_index`, `position`, `alt_count`, `ref_count` |
| `regionN/cna.txt` | `chromosome_index`, `start_position`, `end_position`, `major_cn`, `minor_cn` |

Each `regionN/purity.txt` is a single numeric value without a header. Root
`sample_id` values must be the one-based labels `region1` through `regionS`,
matching the directory names exactly.

The loader rejects an incomplete or inconsistent bundle:

- Mutation IDs are unique and every mutation lies within its declared segment.
- Clone profiles form the complete clone-by-segment product, segment
  coordinates agree across clones, and allele copy numbers are nonnegative
  integers.
- Clone fractions form the complete region-by-clone product, are finite and
  nonnegative, and sum to one per region within `1e-8`.
- Root purity has exactly one value in `(0, 1]` per region; the region-local
  scalar must match it.
- Every region SNV table has exactly one unique coordinate row for every
  mutation. CNA intervals are nonoverlapping and cover every mutation. Read
  counts are nonnegative integers; copy numbers are finite, nonnegative, and
  satisfy `major_cn >= minor_cn`.
- Every positive-fraction region/segment mixture used by a mutation has at
  most two distinct allele-specific copy-number states and at least one
  positive persistent-homolog dosage path.

Hidden truth, mutation histories, true cluster labels, and true dosages are
never consumed during inference.

## Fit

Fit one tumor directory:

```bash
clipp2 fit --tumor-dir inputs/tumor0 --outdir clipp2_results
```

Fit immediate child tumor directories in a cohort:

```bash
clipp2 fit \
  --cohort-dir inputs \
  --max-tumors 100 \
  --workers 4 \
  --outdir clipp2_results
```

Exactly one of `--tumor-dir` and `--cohort-dir` is required, and
`--max-tumors` is cohort-only. Unsupported three-or-more-state cells fail by
default; `--unsupported-policy mask` explicitly count-masks them and records a
reason. `--dosage-prior-penalty` controls the fixed endpoint excess-dosage
prior penalty and defaults to `3`.

The module entry point is equivalent:

```bash
python -m CliPP2 fit --tumor-dir inputs/tumor0
```

For a complete inspectable input and CPU command, see
[`examples/exampleTumor1`](examples/exampleTumor1) and
[`examples/README.md`](examples/README.md).

Production defaults use CUDA float64 tensors, dense device-only fusion, online
partition-guided ADMM lambda selection, and assignment-aware partition ICL.
Ward/CEM supplies only the initializer, blockwise KKT capacity determines the
first positive lambda, and subsequent candidates are proposed from fit
results. No prespecified lambda path is used; the selected estimator must be a
certified complete-graph ADMM fit. Complete graphs retain quadratic compute
cost even when edge tensors are streamed in bounded chunks. Run
`clipp2 fit --help` for solver and custom-graph controls.

## Preprocessing boundary

Battenberg and DPClust belong upstream of CliPP2. Native outputs from those
packages are not accepted. Preprocessing must convert observed data into the
single directory contract above; CliPP2 exposes no Battenberg- or
DPClust-specific inference path.

The loader compiles the directory into `PathLikelihoodSpec`, the internal
boundary between preprocessing and inference. It contains fixed normalized
path priors and numeric `(mutation, region, path)` arrays for:

```text
M(phi) = first_copy * min(phi, switch_fraction)
       + second_copy * max(phi - switch_fraction, 0)
```

Fitting, refitting, scoring, and certification consume this numeric contract
without branching on an upstream package name.

## Simulation benchmarking

The packaged generator writes one exact-size tumor with the same public input
contract:

```bash
clipp2 simulate \
  --out-dir simulations \
  --tumor-id exampleTumor1 \
  --mutation-count 300 \
  --region-count 2 \
  --seed 1
```

The equivalent Python API is deliberately small:

```python
from CliPP2.simulation import TumorSimulationConfig, simulate_tumor

tumor_dir = simulate_tumor(
    TumorSimulationConfig(out_dir="simulations", tumor_id="exampleTumor1")
)
```

`CliPP2.simulation` is a package parallel to `CliPP2.io` and `CliPP2.core`.
Configuration, tree generation, joint SNV/CNA evolution, output construction,
orchestration, and CLI parsing live in separate modules with one-way
dependencies. `python -m CliPP2.simulation --help` exposes the standalone
generator command.

This creates `simulations/exampleTumor1`. Its ten observed input files have
exactly the columns listed above. Simulator-only tables are prefixed with
`truth`, and `scenario_manifest.json` records separate hashes for observed
input and hidden truth. The generated directory is immediately loadable:

```bash
clipp2 fit \
  --tumor-dir simulations/exampleTumor1 \
  --outdir exampleTumor1_results
```

The single-tumor defaults use 300 SNVs, two regions, three evolutionary
clones, CNA event rate `1.5`, at most two local allele-specific copy-number
states, and at least 60% of SNVs on two-state loci. Clone-specific true
mutant-copy dosage and region-specific effective multiplicity are written only
as benchmark truth. Larger factorial cohorts remain available through
`CliPP2.simulation.run_simulation_grid`.

Evaluation reports clustering ARI, effective-multiplicity RMSE, and
amplified-mutant-copy F1. Truth is loaded only after model selection.
