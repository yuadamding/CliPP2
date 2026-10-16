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

Use a new output directory for each run. `--max-clusters` defaults to **10**
and accepts capacities through **50**. The default `--search-policy auto`
keeps the existing enumerated reference search through 10; above 10 it uses
a deterministic, **nonexhaustive** split–merge search. The likelihood, occupied
clonal constraint, conditional refits and mixture score are unchanged, but
different search coverage can change the selected result. Explicit
`--search-policy exhaustive` is restricted to capacities at most 10.

Bounded search keeps up to `--search-width 16` anchored states per occupied
cluster count, with `--search-proposals 4` proposals per state and
`--search-sweeps 2` alternating downward/upward sweeps. Native proposals at
every requested capacity are retained. Diagnostics report actual evaluated
partitions, clonal hypotheses and unexplored moves; they do not certify global
optimality. Increasing search width is not guaranteed to reproduce or improve
a narrower run's candidate bank.

`--ram-budget-gib 1` controls additional per-fit acceleration workspace for
exact likelihood tables, decoded partitions, refits and completed weight fits.
`--vram-budget-gib 0` leaves additional CUDA batching/storage disabled; a positive
allowance opts into it. Cache limits and estimated batching allowances are
reduced to host/container or device headroom, with temporary workspace reserved;
they are **not** hard combined transient or total process memory limits.
Cache capacity never changes refit grids or replaces continuous
optimization with interpolation. Allocated-GPU qualification of these new CUDA
paths remains required before production use.
`--qualify-device-weights` explicitly enables the batched device weight solver
in **qualification-only** mode (extra work; off by default):
CPU reference weights and exact-zero eligibility remain authoritative. Its
agreement diagnostics are not a measured production speedup. Larger scalar
searches use the shared bounded bank; exact-equal adjacent centers can merge via
budgeted proposals rather than the small-K scalar path's implicit merge.

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

Cluster 0 is the designated occupied clonal cluster, with exact CCF 1 in every
region. Other clusters are ordered by decreasing final-CCF-vector L2 norm.

## Contact

For questions or concerns, please submit an issue or email yding4@mdanderson.org.
