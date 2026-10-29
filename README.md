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

## Output

The output directory contains three tables:

| File | Contents |
| --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | Mutation cluster assignments and regional CCFs |
| `{tumor_id}_cluster_centers.tsv` | Cluster sizes and regional CCFs |
| `{tumor_id}_mutation_region_multiplicity.tsv` | CCF and multiplicity estimates for each mutation–region pair |

## Contact

Submit an issue or email yding4@mdanderson.org.
