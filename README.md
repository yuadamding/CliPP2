# CliPP2

CliPP2 estimates cellular prevalence, clusters SNVs, and infers mutant-copy
multiplicity from single- or multi-region tumor sequencing data with
objective-faithful pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one UTF-8, tab-delimited file per tumor:

```text
exampleTumor1.clipp2.txt
```

It uses schema `clipp2.tumor.long.v1`, one ordinary header, and one row per
mutation × sample × local copy-number state:

```text
##schema=clipp2.tumor.long.v1
##tumor_id=exampleTumor1
##genome_build=GRCh38
##coordinate_system=1-based-inclusive
##missing_value=.
mutation_id	sample_id	chromosome	position	ref	alt	alt_count	ref_count	count_observed	purity	normal_cn	segment_id	segment_start	segment_end	cn_state_id	cn_state_fraction	allele_a_cn	allele_b_cn	allele_mode
```

The 19 columns above are required. Extra columns are allowed as reporting
metadata and never alter the objective. IDs are strings, including leading
zeros; sample names are unrestricted identifiers. Rows may be reordered, and
`.clipp2.txt.gz` is supported.

The main rules are:

- `ref` and `alt` are distinct uppercase SNV alleles in `A/C/G/T`.
- Observed counts are nonnegative integers with `count_observed=1`. Missing or
  quality-masked counts use `count_observed=0` and may use `.` for both counts.
- Purity is constant per sample and lies in `(0, 1]`. `normal_cn` is explicit
  and may differ from 2.
- Segment IDs are sample-specific; each 1-based mutation position must fall
  within its inclusive segment bounds.
- Local state fractions are positive, conditional on tumor cells, and sum to
  one per sample-segment.
- `allele_mode=phased` means A/B are persistent homolog labels.
  `allele_mode=unphased` means ordinary major/minor calls and requires
  `allele_a_cn >= allele_b_cn`.
- The current single-switch compiler supports one or two distinct positive
  local CN states. Larger mixtures are rejected or explicitly count-masked
  with `--unsupported-policy mask`.

Repeated mutation, observation, segment, and state fields are checked for
consistency before any inference runs. Hidden truth, mutation histories,
clusters, and dosages are never valid inference inputs.

## Fit

Validate a tumor:

```bash
clipp2 validate --input-file inputs/exampleTumor1.clipp2.txt
```

Fit one tumor:

```bash
clipp2 fit \
  --input-file inputs/exampleTumor1.clipp2.txt \
  --outdir clipp2_results
```

Fit every `.clipp2.txt` or `.clipp2.txt.gz` file in a cohort directory:

```bash
clipp2 fit \
  --input-dir inputs \
  --max-tumors 100 \
  --workers 4 \
  --outdir clipp2_results
```

`--dosage-prior-penalty` controls the fixed endpoint excess-dosage prior and
defaults to `3`.

The module entry point is equivalent:

```bash
python -m CliPP2 fit --input-file inputs/exampleTumor1.clipp2.txt
```

For a complete inspectable input and CPU command, see
[`examples/exampleTumor1.clipp2.txt`](examples/exampleTumor1.clipp2.txt) and
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
packages are not accepted directly. Preprocessing joins observed
mutation-sample counts to sample-segment CN states and writes the package-owned
one-file contract above. CliPP2 contains no caller-specific inference branch
and does not require global CN-clone reconstruction when segment-local state
fractions are available.

The loader compiles the file into `PathLikelihoodSpec`, the internal boundary
between preprocessing and inference. It contains fixed normalized path priors
and numeric `(mutation, sample, path)` arrays for:

```text
M(phi) = first_copy * min(phi, switch_fraction)
       + second_copy * max(phi - switch_fraction, 0)
```

Fitting, refitting, scoring, and certification consume this numeric contract
without branching on an upstream package name.

The former clone-resolved tumor directory remains readable only for migration:

```bash
clipp2 fit --tumor-dir legacy_bundle --outdir clipp2_results
clipp2 convert \
  --tumor-dir legacy_bundle \
  --allele-map mutation_alleles.tsv \
  --output tumor.clipp2.txt
```

The allele map has `mutation_id`, `ref`, and `alt` columns. It is required when
the legacy bundle does not contain alleles; CliPP2 does not invent biological
REF/ALT labels.

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

This creates a benchmark bundle at `simulations/exampleTumor1`. Its sole
observed input is `exampleTumor1.clipp2.txt`; simulator-only tables are
prefixed with `truth`, and `scenario_manifest.json` records separate hashes
for the canonical input and hidden truth. Fit the generated input with:

```bash
clipp2 fit \
  --input-file simulations/exampleTumor1/exampleTumor1.clipp2.txt \
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
