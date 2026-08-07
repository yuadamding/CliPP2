# CliPP2

CliPP2 estimates mutation prevalence, clusters SNVs, and infers mutant-copy
multiplicity from single- or multi-region tumor sequencing data with pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one plain tab-delimited file per tumor, see example, [`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).


## Fit


Validate it:

```bash
clipp2 validate --input-file examples/exampleTumor1.tsv
```

Fit it. On a machine with CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```

On a CPU-only machine, `--device cpu` is mandatory (the default device
`cuda`).



```bash
python -m CliPP2 fit --input-file examples/exampleTumor1.tsv --device cpu
```

The example itself is documented in [`examples/README.md`](examples/README.md).

Run
`clipp2 fit --help` for solver and custom-graph controls.

## Outputs

A fit writes four scientific tables into `--outdir`, prefixed with the tumor id
(the input file name stem, unless a `##tumor_id` metadata line overrides it):

| File | One row per | Contents |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | `cluster_label` plus three prevalence estimates per region: `phi_*` (raw fused fit), `summary_phi_*` (cluster-collapsed), `bic_refit_phi_*` (partition-constrained refit) |
| `{tumor_id}_cluster_centers.tsv` | cluster | `cluster_size`, `cluster_diameter`, `cluster_diameter_exact`, and per-region centers; join to mutations on `cluster_label` |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | the same three phi estimates, local `major_cn`/`minor_cn`, and the mutant-copy summaries: MAP path, path probabilities, posterior/MAP mutant-copy mass and effective multiplicity, amplification call, path entropy (plus `summary_*` twins at the clustered phi) |
| `{tumor_id}_mutation_region_path_posterior.tsv` | supported path | per-path posterior, dosage endpoints (`first_copy`, `second_copy`, `switch_fraction`), prior, mass and multiplicity — written only when a path likelihood exists |
| `{tumor_id}_simulation_eval.tsv` | run | benchmark metrics; only with `--simulation-root` |

No other file is written. Solver and selection diagnostics — the selected
lambda, log-likelihoods, BIC/ICL, convergence and certificate fields,
input/objective hashes, `software_version` — exist only in the run summary that
`clipp2 fit` prints to stdout; redirect it to keep a record. Timings never
enter the tables, so refitting the same input on the same device and dtype
reproduces the output directory byte for byte. `--skip-outputs` suppresses the
tables and prints the summary only.
