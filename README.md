# CliPP2

CliPP2 clusters mutations jointly across one or more tumor regions and estimates
regional cancer cell fractions (CCFs) and mutation multiplicities.

## Install

Requires Python 3.10+ and a C++17 compiler. From the repository root:

```bash
python -m pip install .                       # CPU
CLIPP2_BUILD_CUDA=1 python -m pip install .     # CPU + CUDA
```

CUDA builds require `nvcc`; GPU execution requires a compatible NVIDIA GPU and
driver. CPU builds require neither CUDA nor PyTorch.

## Run

```bash
clipp2 fit --input-dir tumor/ --device cpu --output-dir results/
```

Use `--device cuda` for GPU execution and a new output directory for each run.
`--max-clusters` defaults to **20**; larger values can greatly increase search cost.
Use `--verbose` for phase starts and throttled completed-operation snapshots on
stderr, not a heartbeat or an optimizer-request trace. No extra files are written.

`--execution-policy auto` keeps the reference path unless a source-, runtime- and
hardware-matched GPU qualification is installed. `reference` forces that path;
`candidate` explicitly tests the unpromoted CUDA batching/resident policy.

## Input

### Directory

One directory represents one tumor; its region subfolders are fitted jointly:

```text
tumor/
├── region1/
│   ├── snv.txt
│   ├── cna.txt
│   └── purity.txt
└── region2/          # same three files
```

Folder names define tumor/region IDs. See the [runnable example](examples/README.md).
Each region requires:

| File | Required contents |
| --- | --- |
| `snv.txt` | `chromosome_index`, `position`, `ref_count`, `alt_count` |
| `cna.txt` | `chromosome_index`, `start_position`, `end_position`, `major_cn`, `minor_cn`, `total_cn` |
| `purity.txt` | One number in `(0, 1]`, without a header |

Loci not covered by a region's CNA segments default to `major_cn = minor_cn = 1`
(total and normal CN 2). A header-only `cna.txt` applies this default to all loci;
the file is still required. Missing read observations remain missing.

### TSV alternative

Use `--input-file tumor.tsv` instead of `--input-dir` for a single table combining
all regions; gzip is supported. See [example](examples/exampleTumor1.tsv).

| Columns | Contents |
| --- | --- |
| `mutation_id`, `sample_id` | Mutation and region identifiers |
| `alt_count`, `ref_count`, `count_observed` | Read counts and observation flag: 1 for observed, 0 for missing |
| `purity`, `normal_cn` | Tumor purity in `(0, 1]` and nonnegative integer normal CN |
| `segment_id`, `cn_state_id`, `cn_state_fraction` | Segment, CN state and its fraction |
| `allele_a_cn`, `allele_b_cn` | Major and minor copy numbers, with `allele_a_cn >= allele_b_cn` |

Include every mutation–region pair; use `count_observed=0` and `.` for missing
counts. For multiple CN states, repeat the same read counts across state rows;
state fractions must sum to 1. Purity must be constant within each region.

For both input formats, `--max-major-cn` defaults to **10**. A mutation is excluded
only from regions where any of its CN states exceeds this limit; valid observations
in other regions remain. Mutations with no usable observations are dropped.
Regions with no usable observations cause an error rather than being removed.

## Output

The output directory contains three tables:

| File | Contents |
| --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | Mutation cluster assignments and regional CCFs |
| `{tumor_id}_cluster_centers.tsv` | Cluster sizes and regional CCFs |
| `{tumor_id}_mutation_region_multiplicity.tsv` | CCF and multiplicity estimates for each mutation–region pair |

Cluster 0 has the largest CCF-vector L2 norm and is not necessarily clonal.
CN-filtered mutation-region pairs retain rows with `NA` CCF and multiplicity
estimates; their regional CCF entries in the mutation-clusters table are also `NA`.

## Contact

Submit an issue or email yding4@mdanderson.org.

The synthetic-data generator is maintained separately in the `CliPP2-simulation` companion.
