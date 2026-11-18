# CliPP2

CliPP2 clusters mutations jointly across one or more tumor regions and estimates
regional cancer cell fractions (CCFs) and mutation multiplicities.

## Install

Requires Python 3.10+. From the repository root:

```bash
python -m pip install .
```

This CPU implementation needs no C++ compiler, CUDA, or PyTorch.

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

Use `--input-file tumor.tsv` instead of `--input-dir` for one table combining
all regions; gzip is supported.

| Columns | Contents |
| --- | --- |
| `mutation_id`, `sample_id` | Mutation and region identifiers |
| `alt_count`, `ref_count`, `count_observed` | Read counts and observation flag: 1 for observed, 0 for missing |
| `purity`, `normal_cn` | Tumor purity in `(0, 1]` and nonnegative integer normal CN |
| `allele_a_cn`, `allele_b_cn` | Major and minor copy numbers, with `allele_a_cn >= allele_b_cn` |

Use one representative row per mutation–region pair with CN state fraction 1.
Optional `segment_cn_states` preserves the original multiple states.

## Output

`--output-dir` contains exactly these three TSVs, with no subdirectories:

| File | Columns |
| --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | `tumor_id`, `mutation_id`, `cluster_label`, `phi_<region>` for each region |
| `{tumor_id}_cluster_centers.tsv` | `tumor_id`, `cluster_label`, `cluster_size`, `phi_<region>` for each region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | `tumor_id`, `mutation_id`, `region_id`, `phi`, `multiplicity_call` |

Label 0 denotes an active clonal class. Multiplicity is NA for missing,
excluded, zero-depth, or zero-CCF pairs.

## Contact

Submit an issue or email yding4@mdanderson.org.
