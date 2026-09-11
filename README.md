# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

CliPP2 excludes a mutation from **all regions** if any region has subclonal
copy number (more than one distinct CN state after identical states are combined)
or major CN greater than `--max-major-cn` (**default: 4**); different clonal CN
states between regions are allowed. The limit is inclusive: major CN 4 is
retained by default, while 5 and above are excluded. Use `--max-major-cn 6` for
the previous cutoff. For each retained mutation–region pair, fitting marginalizes integer
multiplicity candidates from **1 to major CN** with uniform priors under a
binomial likelihood adjusted for purity, normal/tumor copy number, and CCF.
The reported multiplicity is the highest-posterior candidate conditional on the
final fixed-partition CCF refit, with exact ties choosing the smaller integer—not
a rounded VAF-based estimate. Missing or zero-depth observations are marked
uninformative and their call is left missing, except when the only possible
multiplicity is structurally fixed at one.

Here, **balanced CN means only `major_cn = minor_cn = 1`**. Equal-copy states
such as `2/2` or `3/3` are not balanced CN; their multiplicity candidates remain
`1..2` or `1..3`, respectively. The historical computational profile name
`balanced` is unrelated to this biological definition.

CNA-only multiplicity evaluation includes mutation–region rows satisfying
`(major_cn != 1) | (minor_cn != 1)`: exclude only `1/1`, not higher-copy equal-CN
states. Report pooled exact-class macro-F1 with eligible row counts, plus
micro-, weighted-, and per-class F1. This evaluation population is separate
from the simulator's multiplicity-sampling rule.

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```


Use `--device cpu` for explicit CPU execution. CUDA is the default; unavailable
CUDA or insufficient memory fails without silently switching devices. Set the
positive-integer CN eligibility cutoff with `--max-major-cn` (default 4).
`--verbose` and standard help and version options are also available.

## Fixed inference workflow

There is one production algorithm: the former **balanced + independent_broad**
default. Multiplicity is marginalized independently in each mutation–region
row; balanced CN (`1/1`) has only the multiplicity-one candidate. Multiplicity
is not forced to agree across regions. A
deterministic scalar pilot defines the complete adaptive fusion graph.
Ward/CEM initialization retains its bounded scalar-certification stage, then
online lambda search evaluates raw-fusion candidates with deterministic warm
and cold starts. Certified raw partitions and direct Ward/CEM proposals compete
under the same immutable-label grid/local refit and fixed-partition Dirichlet
score. No clonal anchor or CCF-one cluster is imposed.

Working arithmetic is float32. Terminal raw audits use float64 and the unchanged
full-KKT cutoff **0.004**; bounded same-objective precision recovery remains.
Dense and chunked complete-graph ALM arithmetic are retained. Fixed-label refits
are approximate CPU float64 fits, not globally certified likelihood optima;
the bounded lambda search does not establish a global hybrid optimum. A direct
Ward/CEM selection does not inherit its raw reference's KKT certificate.

Resolved numerical constants and the fixed algorithm identifier are recorded
in the manifest. They are not user-selectable modes.

### Python API and interface change

```python
from CliPP2 import FitConfig, process_tumor

summary = process_tumor("tumor.tsv", "results", FitConfig(device="cuda", max_major_cn=4))
```

This revision intentionally removes profile, multiplicity-policy, precision,
graph, warm-start, solver/refit/resource tuning, fallback and output-skipping
options from the public CLI/API. Removed arguments raise errors; they are not
silently mapped to defaults. `FitConfig` accepts `device`, `verbose`, and
`max_major_cn`. `load_tumor_txt(..., max_major_cn=4)` uses the same default;
preloaded data must use the same cutoff as the fit configuration.
The separate public raw-fit/preparation and standalone publication entry points
are removed. `computation_profile=balanced` is a fixed historical provenance
label, not a selectable profile. Reproduce
older modes using their exact historical commit.

## Outputs

A successful fit writes five artifacts into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | selected cluster and final fixed-partition CCF per region |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | final CCF, CN, multiplicity MAP call, and posterior probabilities |
| `{tumor_id}_excluded_mutations.tsv` | triggering mutation–region–reason | original-CN exclusion audit; header-only when none are excluded |
| `{tumor_id}_run_manifest.json` | run | source/input/config hashes, numerical qualification, and output hashes |

CCFs use `phi_<region>` in the wide tables and `phi` in the long table.
Publication never changes selected labels or refits CCFs. Outputs are never
overwritten: use a new directory for retries. Exclusion evidence is written
before fitting; the manifest becomes complete only after validated publication.
Numerical failure preserves a failed manifest and exclusion audit, not a
mislabelled successful clustering result.

The manifest records `config.max_major_cn`. Exclusions use reason
`MAJOR_CN_ABOVE_LIMIT`, with the observed `max_major_cn` and configured
`major_cn_limit` in the audit. Summary schema 6 replaces the old hard-coded
`excluded_major_cn_gt6_mutation_count` key with
`excluded_major_cn_above_limit_mutation_count` and records `max_major_cn`.
Posterior columns run from `multiplicity_p1` through
`multiplicity_p<max_major_cn>`; unsupported candidates have zero probability.
Changing the cutoff changes input eligibility, not the KKT gate or solver settings.

## Simulation and checks

The source-only [tree simulation generator](simulation/README.md) remains in
this repository and is excluded from the inference wheel. Its truth-generating
assumptions are unchanged by this inference cleanup.

Tests live outside this Git repository at `../tests/CliPP2_single_mode/` in the
development workspace. Using conda environment `ml1`, run
`python -m pytest -q ../tests/CliPP2_single_mode` and
`ruff check . ../tests/CliPP2_single_mode`. The compact external suite covers the
public contract, likelihood, publication, source identity, ALM arithmetic and
certificate retention. CUDA
checks require `CLIPP2_TEST_CUDA=1` inside a commit-pinned Seadragon LSF GPU job;
skipped CUDA tests do not constitute GPU validation.
