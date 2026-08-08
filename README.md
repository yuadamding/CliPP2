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

Model selection enforces a strict clonal restriction: every scored candidate
must contain one cluster pinned at the clonal center, φ = 1 in every region
(clipped to the feasibility box). The pinned centers are constants, so a
K-cluster model is charged (K − 1) × S degrees of freedom. A one-cluster model
is therefore admissible only when the whole tumor is consistent with φ = 1,
which is what makes weakly separated subclones detectable. After selection,
one hard E-step against the anchored refit centers reassigns boundary
mutations (K held fixed) before the tables are written.

Run
`clipp2 fit --help` for solver and custom-graph controls.

## Outputs

A fit writes three tables into `--outdir`, prefixed with the tumor id
(the input file name stem, unless a `##tumor_id` metadata line overrides it):

| File | One row per | Contents |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | `cluster_label` plus three prevalence estimates per region: `phi_*` (raw fused fit), `summary_phi_*` (cluster-collapsed), `bic_refit_phi_*` (clonal-anchored partition refit; the clonal cluster is pinned at φ = 1 per region) |
| `{tumor_id}_cluster_centers.tsv` | cluster | `cluster_size`, `cluster_diameter`, `cluster_diameter_exact`, and per-region centers; join to mutations on `cluster_label` |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | the same three phi estimates, local `major_cn`/`minor_cn`, and the mutant-copy summaries: MAP path, path probabilities, posterior/MAP mutant-copy mass and effective multiplicity, amplification call, path entropy (plus `summary_*` twins at the clustered phi) |


