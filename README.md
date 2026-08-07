# CliPP2

CliPP2 estimates mutation prevalence, clusters SNVs, and infers mutant-copy
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
- Local state fractions are positive, conditional on the tumor population, and
  sum to one per sample-segment.
- `allele_mode=phased` means A/B are persistent homolog labels.
  `allele_mode=unphased` means ordinary major/minor calls and requires
  `allele_a_cn >= allele_b_cn`.

Repeated mutation, observation, segment, and state fields are checked for
consistency before any inference runs. Hidden truth, mutation histories,
clusters, and dosages are never valid inference inputs.

### The table must be complete

Two completeness rules determine the exact row count, and both are hard errors
rather than warnings:

- **Every mutation × sample unit must be present.** The loader builds the full
  cross product of all observed `mutation_id` values with all observed
  `sample_id` values and rejects any gap. A mutation reported in only one of two
  samples is an error, not missing data. To express "not observed here", emit the
  row with `count_observed=0` and `.` for both counts.
- **Every unit must repeat every CN state of its segment.** A two-state segment
  therefore requires exactly two rows for every mutation on it, including the
  state the mutation is not on. Omitting one is an error.

So the row count is exactly the sum over mutations and samples of the number of
local CN states on that mutation's segment in that sample.

### Local copy-number states

`cn_state_id` enumerates the distinct local copy-number states of a
sample-segment, and `cn_state_fraction` gives each state's fraction of the tumor
population. The single-switch compiler supports **one or two** distinct positive
states per sample-segment. Three or more are outside the model and are rejected,
or count-masked with reason `MORE_THAN_TWO_LOCAL_CN_STATES` under
`--unsupported-policy mask`.

**One state is fully supported and is the ordinary case for a clonal segment.**
Because fractions must sum to one per sample-segment, a single state necessarily
has `cn_state_fraction=1.0`. There is then no subclonal copy-number switch to
place, so the compiler emits *constant-dosage* paths: `first_copy == second_copy`
and `switch_fraction == 1.0`, leaving mutant-copy multiplicity as the only
remaining uncertainty. A single state contributes one candidate path per integer
dosage from 1 up to `max(allele_a_cn, allele_b_cn)`:

| single state | compiled paths | candidate multiplicities |
| --- | --- | --- |
| `allele_a_cn=1, allele_b_cn=1` | 1 | 1 |
| `allele_a_cn=2, allele_b_cn=1` | 2 | 1, 2 |
| `allele_a_cn=3, allele_b_cn=0` | 3 | 1, 2, 3 |
| `allele_a_cn=0, allele_b_cn=0` | 0 → unsupported | none exist |

A fully deleted segment (`allele_a_cn=allele_b_cn=0`) admits no positive
mutant-copy dosage, so it is rejected — or count-masked with reason
`NO_POSITIVE_MUTANT_COPY_PATH` under `--unsupported-policy mask`. Two-state
segments are what activate the switch term in `M(phi)` below, where dosage
changes at `switch_fraction`.

Mixing is allowed and expected: a tumor may have some one-state and some
two-state segments. The shipped
[`examples/exampleTumor1.clipp2.txt`](examples/exampleTumor1.clipp2.txt) has 6
one-state and 14 two-state sample-segments.

Note that `--unsupported-policy` has opposite defaults on the two subcommands:
`fit` defaults to `error`, so an unsupported locus stops the run; `validate`
defaults to `mask`, so it reports unsupported loci instead of failing on the
first one.

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
  --outdir clipp2_results \
  --device cpu
```

`--workers` above is paired with `--device cpu` deliberately: process
parallelism and the default CUDA device do not mix. See
[Device and precision](#device-and-precision).

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

## Device and precision

### CUDA is the default, so CPU-only machines must opt in

`--device` accepts `auto`, `cpu` or `cuda` and **defaults to `cuda`**. Every
`clipp2 fit` command shown above therefore requires a working CUDA build. There
is no silent fallback: on a machine without CUDA, `--device cuda` fails
immediately with

```text
CudaUnavailableError: Requested Torch device 'cuda', but CUDA is not available.
Use device='cpu' or device='auto' to permit CPU execution.
```

| `--device` | Behaviour |
| --- | --- |
| `cuda` *(default)* | Requires CUDA; raises `CudaUnavailableError` if absent. Uses the current default GPU. |
| `auto` | Uses CUDA when available, CPU otherwise. Prefer this in portable scripts and CI. |
| `cpu` | Forces CPU regardless of what hardware is present. |

The CLI accepts only those three values — `--device cuda:0` is rejected. To pin
a specific GPU, set `CUDA_VISIBLE_DEVICES` in the environment. The Python API is
less restricted: `FitOptions(device="cuda:1")` resolves an explicit index and
validates it against the visible device count.

So the CPU form of the quickstart is:

```bash
clipp2 fit \
  --input-file inputs/exampleTumor1.clipp2.txt \
  --outdir clipp2_results \
  --device cpu
```

### Precision

`--dtype` accepts `auto`, `float16`, `float32` or `float64` and **defaults to
`float64`**; `auto` also resolves to `float64`, on both devices. `float16` is
CUDA-only and raises `RuntimeError: Float16 runtime dtype is only supported on
CUDA` on CPU.

Keep `float64` unless you have a specific reason not to. The binomial
log-likelihood is stiff near the prevalence floor — on the shipped 300-mutation
example the gradient norm reaches roughly `7e6` at a prevalence of `1e-4` — and
the KKT residuals that gate model selection are compared against tolerances
around `5e-5`, so reduced precision erodes exactly the quantities the estimator
certifies against.

### Running on CPU: limit the thread count

This matters more than any other CPU setting. The tensors are small — `(M × S)`
per node and `(E × S)` per edge, where a complete graph has `E = M(M−1)/2` — so
Torch's intra-op thread pool spends more time synchronising than computing, and
it is enabled by default. **Nothing in CliPP2 sets the thread count for you.**

Measured on one simulated 60-mutation, two-region tumor, same inputs and
byte-identical outputs:

| CPU threading | CPU time |
| --- | --- |
| Torch default | 18 min |
| `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` | 1 min 36 s |

The effect is large enough to change what is feasible: the shipped 300-mutation
example did not finish within 900 s under default threading, but completes in
about 29 minutes single-threaded. Compare CPU time rather than wall clock, since
the oversubscribed run also competes with itself.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 clipp2 fit \
  --input-file inputs/exampleTumor1.clipp2.txt \
  --outdir clipp2_results \
  --device cpu
```

Library users can equivalently call `torch.set_num_threads(1)` before fitting.
Do **not** apply this to CUDA runs — there the work is on the device and the
CPU thread pool is not the bottleneck.

### Cohort runs and `--workers`

`--workers` defaults to `1` and only affects directory inputs (`--input-dir`,
`--cohort-dir`); it is ignored for `--input-file`, which always fits in-process.
Workers are separate `spawn` processes.

Combining `--workers > 1` with CUDA is not blocked, only warned about:

```text
RuntimeWarning: workers=4 with device='cuda': multiple processes may each
initialize the same CUDA device, causing OOM or contention.
Use workers=1 for CUDA mode, or set device='cpu'.
```

The warning also fires for `--device auto` on a machine that has a GPU, because
`auto` resolves to CUDA there. Use `--workers 1` for GPU runs and reserve
process parallelism for `--device cpu`, where it composes well with the
single-thread setting above (one thread per worker).

### Device memory

`--dense-fallback-policy` controls what happens when the exact solver cannot fit
its dense edge tensors in device memory. It defaults to `device-only`.

| Policy | Behaviour on device-memory exhaustion |
| --- | --- |
| `device-only` *(default)* | Never migrates solver work to the host; surfaces an explicit resource-limit error instead. |
| `cpu-allowed` | Retries the affected solve on CPU. |
| `error` | Disables dense fallback outright. |

`--workset-max-bytes` and `--compressed-cache-max-bytes` are byte budgets for
the working-set and compressed-certificate machinery, which the dense default
backend does not use; they matter only with `--inner-backend
quotient-workset`.

### Reproducibility across devices and runs

For a fixed device and dtype, CPU fits are bit-reproducible: two runs of the
same input produce byte-identical cluster, prevalence, multiplicity and
path-posterior tables. Only the `*_elapsed_seconds` columns differ, and because
those are embedded in the result tables the files themselves are not byte-equal
even when every scientific value is.

CUDA is weaker, by design of the underlying kernels:

- **CUDA fits are not bit-reproducible run to run.** The float `index_add_` used
  to accumulate the edge adjoint is nondeterministic on GPU.
- **CPU and GPU can assign different labels near lambda-path decision
  boundaries**, where two candidate partitions are nearly tied.

For reproducible GPU runs, enable `torch.use_deterministic_algorithms(True)` and
set `CUBLAS_WORKSPACE_CONFIG` before fitting. If you need an auditable result,
record `--device` and `--dtype` alongside the `input_data_hash`,
`objective_spec_hash` and `original_graph_hash` that every run reports.

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
as benchmark truth. Generate a cohort by calling `simulate_tumor` once per
tumor with a distinct `--seed` / `tumor_id`; the generator is deliberately
one-tumor-at-a-time and owns no factorial grid driver.

Evaluation reports clustering ARI, effective-multiplicity RMSE, and
amplified-mutant-copy F1. Truth is loaded only after model selection.
