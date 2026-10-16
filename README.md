# CliPP2

CliPP2 clusters mutations jointly across one or more tumor regions and estimates
regional cancer cell fractions (CCFs) and mutation multiplicities.

## Install

Requires Python 3.10+ and a C++17 compiler. Run from the repository root.

```bash
python -m pip install .
```

GPU execution also requires an NVIDIA GPU, a compatible driver and CUDA-enabled
PyTorch.

## Run

```bash
# CPU
clipp2 fit --input-file tumor.tsv --device cpu --outdir ../clipp2-cpu

# GPU
clipp2 fit --input-file tumor.tsv --device cuda --outdir ../clipp2-gpu
```

Use a new output directory for each run. `--max-clusters` defaults to **10**;
use a smaller value to limit the search.

## Input

Provide one tab-separated file per tumor, optionally gzip-compressed, with these
columns. See [example input](examples/exampleTumor1.tsv).

| Columns | Contents |
| --- | --- |
| `mutation_id`, `sample_id` | Mutation and region identifiers |
| `alt_count`, `ref_count`, `count_observed` | Read counts and observation flag: 1 for observed, 0 for missing |
| `purity`, `normal_cn` | Tumor purity in `(0, 1]` and normal copy number |
| `segment_id`, `cn_state_id`, `cn_state_fraction` | Segment, CN state and its fraction |
| `allele_a_cn`, `allele_b_cn` | Major and minor copy numbers, with `allele_a_cn >= allele_b_cn` |

Include every mutation–region pair, including missing observations. For multiple
CN states, repeat the same read counts across state rows; state fractions must
sum to 1. Purity must be constant within each region.

`--max-major-cn` defaults to **4**. A mutation is excluded from all regions if
any of its CN states exceeds this major-copy-number limit.

## Output

The output directory contains three tables:

| File | Contents |
| --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | Mutation cluster assignments and regional CCFs |
| `{tumor_id}_cluster_centers.tsv` | Cluster sizes and regional CCFs |
| `{tumor_id}_mutation_region_multiplicity.tsv` | CCF and multiplicity estimates for each mutation–region pair |

Cluster 0 has the largest CCF-vector L2 norm and is not necessarily clonal.

## Contact

For questions or concerns, please submit an issue or email yding4@mdanderson.org.
