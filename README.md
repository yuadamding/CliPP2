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

The header above shows the full canonical column set, but only the **12
columns the objective is computed from are required**:

```text
mutation_id  sample_id  alt_count  ref_count  count_observed  purity
normal_cn  segment_id  cn_state_id  cn_state_fraction  allele_a_cn  allele_b_cn
```

The other 7 — `chromosome`, `position`, `ref`, `alt`, `segment_start`,
`segment_end`, `allele_mode` — are identity and coordinate metadata that never
enter the likelihood. Each may be omitted from the file entirely, or carry the
missing marker `.` in any row. A minimal 12-column file produces byte-identical
results to the full 19-column file. When the optional columns are present and
non-missing they are still validated: provide `ref` and `alt` together or not at
all, provide both segment bounds together, and the position-within-bounds check
runs only when position and bounds are all present. A missing `allele_mode`
defaults to `unphased`; declare `phased` explicitly if your `allele_a_cn` /
`allele_b_cn` are persistent homolog labels with `a < b` anywhere.

Extra columns beyond the 19 are also allowed as reporting metadata and never
alter the objective. IDs are strings, including leading zeros; sample names are
unrestricted identifiers. Rows may be reordered, and `.clipp2.txt.gz` is
supported. `.` is the only missing marker: NA-style spellings (`NA`, `NaN`,
`null`, an empty field) are rejected everywhere — but note that `NA` *is*
accepted as an ordinary identifier string, so a preprocessing bug that writes
`NA` into an ID column creates a real mutation or sample named `NA` rather than
an error.

The main rules are:

- `ref` and `alt`, when provided, are distinct uppercase SNV alleles in
  `A/C/G/T`.
- Observed counts are nonnegative integers with `count_observed=1`. Missing or
  quality-masked counts use `count_observed=0` and may use `.` for both counts.
- Purity is constant per sample and lies in `(0, 1]`. `normal_cn` is explicit
  and may differ from 2 — it enters the prevalence scaling directly, which is
  why it is required rather than defaulted.
- Segment IDs are sample-specific; when position and segment bounds are
  provided, each 1-based mutation position must fall within its inclusive
  segment bounds.
- Local state fractions are positive, conditional on the tumor population, and
  sum to one per sample-segment.
- `allele_mode=phased` means A/B are persistent homolog labels.
  `allele_mode=unphased` (the default when the column is absent or `.`) means
  ordinary major/minor calls and requires `allele_a_cn >= allele_b_cn`.

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

Every command below runs against the shipped example,
[`examples/exampleTumor1.clipp2.txt`](examples/exampleTumor1.clipp2.txt) —
300 SNVs across two regions. Paths are relative to the package directory; a
pip-installed copy keeps the file under `site-packages/CliPP2/examples/`.
Substitute your own `.clipp2.txt` once these work.

Validate it:

```bash
clipp2 validate --input-file examples/exampleTumor1.clipp2.txt
```

Fit it. On a machine with CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.clipp2.txt \
  --outdir exampleTumor1_results
```

On a CPU-only machine, `--device cpu` is mandatory (the default device is
`cuda`) and limiting the Torch thread count is strongly recommended — on this
300-SNV input it is the difference between finishing in about half an hour and
not finishing at all:

```bash
OMP_NUM_THREADS=1 clipp2 fit \
  --input-file examples/exampleTumor1.clipp2.txt \
  --outdir exampleTumor1_results \
  --device cpu
```

See [Device and precision](#device-and-precision) for why, and for the
`--workers` cohort mode that fits every `.clipp2.txt` or `.clipp2.txt.gz` in a
directory via `--input-dir`. `--dosage-prior-penalty` controls the fixed
endpoint excess-dosage prior and defaults to `3`.

The module entry point is equivalent:

```bash
python -m CliPP2 fit --input-file examples/exampleTumor1.clipp2.txt --device cpu
```

The example itself is documented in [`examples/README.md`](examples/README.md).

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

`--device` accepts `auto`, `cpu` or `cuda` and **defaults to `cuda`**. It exists
only on `fit`; `validate`, `convert` and `simulate` never touch a device. Every
`clipp2 fit` command shown above therefore requires a working CUDA build. Under
the default fallback policy there is no silent downgrade: on a machine without
CUDA, `--device cuda` fails immediately with exit status 1 and

```text
CudaUnavailableError: Requested Torch device 'cuda', but CUDA is not available.
Use device='cpu' or device='auto' to permit CPU execution.
```

One flag changes that. `--dense-fallback-policy cpu-allowed` also rescues an
*unavailable device*, not just an out-of-memory one: it catches
`CudaUnavailableError`, re-resolves the runtime to CPU and completes the fit. The
only trace is in the run summary, where `fallback_reason` becomes
`dense_cpu_after_context_resource_limit` and `inner_solver` becomes
`admm_complete_graph_cpu_fallback`. If you use that policy, check those two
columns before reporting a run as GPU-accelerated.

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
  --input-file examples/exampleTumor1.clipp2.txt \
  --outdir exampleTumor1_results \
  --device cpu
```

### Precision

`--dtype` accepts `auto`, `float16`, `float32` or `float64` and **defaults to
`float64`**; `auto` also resolves to `float64` on both devices, so it never
selects reduced precision for a GPU. `float16` is CUDA-only and raises
`RuntimeError: Float16 runtime dtype is only supported on CUDA` elsewhere. The
device is validated before the dtype, so on a CPU-only host
`--dtype float16` at the default device reports the CUDA error, not the float16
one.

**Keep `float64`.** `float32` does run to completion, but the case against it is
structural rather than a matter of accumulated round-off:

- Many solver and certificate thresholds are computed as multiples of
  `torch.finfo(dtype).eps`, which is `2.2e-16` for float64 and `1.2e-7` for
  float32. Dropping precision therefore *loosens the convergence and stationarity
  criteria by orders of magnitude* rather than merely adding noise.
- The certification target does not compensate: `full_kkt_tolerance` is
  `5 × --tol` regardless of dtype.
- The likelihood is stiff where it matters. On the shipped 300-mutation example
  the gradient norm reaches roughly `7e6` at a prevalence of `1e-4`, and low
  prevalence is exactly the regime this tool exists to resolve.
- Reduced precision is not a speed win here. On a 4-mutation input a float32 fit
  took substantially *longer* than float64 because the lambda search explored more
  candidates. That specific ratio is input-dependent and should not be
  generalised, but there is no measured case of float32 being faster.
- Nothing warns you. The output TSVs are cast to float64 for reporting either
  way, and the partition-refit `bic_loglik` is always computed in NumPy float64,
  so a float32 fit *looks* like a float64 fit in its own results. Only the
  `dtype` column records the resolved precision.

`--dtype` is never cross-validated against `--tol`, so a loose dtype with a tight
tolerance is accepted silently.

### Running on CPU: limit the thread count

This matters more than any other CPU setting. The tensors are small — `(M × S)`
per node and `(E × S)` per edge, where the default complete graph has
`E = M(M−1)/2` — so no tensor in the fit reaches Torch's parallelisation grain
size, and the intra-op thread pool spends its time synchronising rather than
computing. It is on by default, and **nothing in CliPP2 sets the thread count for
you.**

Sweep on an 8-mutation, two-region fit, sequential runs on one host:

| Intra-op threads | CPU time | Wall time |
| --- | --- | --- |
| 1 | 16.5 s | 16.5 s |
| 2 | 26.9 s | 16.6 s |
| 4 | 44.4 s | 17.3 s |
| 8 | 74.9 s | 16.8 s |
| Torch default (23 here) | 423 s | 35.5 s |

So threading buys nothing at all in wall time up to 8 threads while burning up to
4.5× the CPU, and at Torch's own default it is **2.2× worse in wall time and
~26× worse in CPU time**. The results are unaffected: across 1, 2, 4, 8 and
default threads every non-timing column of `run_summary.tsv` and
`lambda_search.tsv` matched, and the four per-mutation tables were byte-identical.

Treat CPU time as the number to compare — the multi-threaded figure is mostly
spin time and is itself unstable (two runs of the same fit measured 423 s and
297 s of CPU, against a single-threaded run stable near 16 s).

```bash
OMP_NUM_THREADS=1 clipp2 fit \
  --input-file examples/exampleTumor1.clipp2.txt \
  --outdir exampleTumor1_results \
  --device cpu
```

`OMP_NUM_THREADS=1` alone is sufficient. Note the precedence: Torch reads
`OMP_NUM_THREADS` but `MKL_NUM_THREADS` overrides it when both are set, so a
stray `MKL_NUM_THREADS` from a cluster module file will win. Library users can
call `torch.set_num_threads(1)` before fitting instead.

The effect compounds in cohort mode: `--workers N` spawns N processes that each
independently default to the full intra-op thread count, so `--workers 8` without
a thread limit oversubscribes by roughly 8 × 23 threads. Set the environment
variable once — it is inherited by the workers — and treat one thread per worker
as the baseline.

Do **not** apply this to CUDA runs: the solver tensors live on the device, and the
CPU pool governs only host kernels. The one exception is
`--dense-fallback-policy cpu-allowed`, where a fallback run becomes a CPU run and
the setting starts to matter again.

### Cohort runs and `--workers`

`--input-dir` fits every `.clipp2.txt` or `.clipp2.txt.gz` in a directory. The
shipped `examples/` directory contains exactly one, so this is a runnable
one-tumor cohort:

```bash
OMP_NUM_THREADS=1 clipp2 fit \
  --input-dir examples \
  --outdir exampleTumor1_results \
  --device cpu
```

For a real cohort, add `--workers N` (one process per concurrent tumor) and
`--max-tumors` to cap the run. `--workers` defaults to `1` and only affects
directory inputs (`--input-dir`, `--cohort-dir`); it is accepted but silently
ignored for `--input-file` and `--tumor-dir`, which always fit in-process.
Values below 1 are clamped to 1 rather than rejected. Workers are separate
`spawn` processes, never forked.

Combining `--workers > 1` with CUDA is not blocked, only warned about:

```text
RuntimeWarning: workers=4 with device='cuda': multiple processes may each
initialize the same CUDA device, causing OOM or contention.
Use workers=1 for CUDA mode, or set device='cpu'.
```

The concrete hazard is that the dense-solver memory preflight sizes itself
against 80% of the *currently free* device memory, and every worker computes that
independently and concurrently — so N workers can each conclude the problem fits
and then collectively exhaust the GPU. Nothing pins workers to different devices,
so on a multi-GPU host you must partition them yourself with
`CUDA_VISIBLE_DEVICES`.

The warning also fires for `--device auto` on a machine that has a GPU, since
`auto` resolves to CUDA there. It is keyed on the *requested* device string, so
`--device cuda --workers 2` warns even on a host with no GPU at all. Use
`--workers 1` for GPU runs and reserve process parallelism for `--device cpu`.

### How failures are handled in a cohort

Worth knowing before launching a long run, because the two cases differ sharply:

- **Memory failures are per-tumor and recoverable.**
  `ExactSolverResourceLimit`, `MemoryError` and `torch.OutOfMemoryError` are
  caught per tumor, recorded as a summary row with `selection_eligible=False`,
  and the cohort continues.
- **Any other exception aborts the whole cohort.** The summary table is written
  only after the loop completes, so tumors that already succeeded lose their
  cohort-level row. A tumor that is merely hard to fit can do this: exhausting
  the lambda search without a certified candidate raises
  `NoEligibleModelSelectionCandidatesError`, which is not in the recoverable set.
  Until that is addressed, prefer batching a large cohort into several smaller
  `--input-dir` invocations.

Single-tumor mode has no resource-limit handler at all: a memory failure there
terminates the command with a traceback and a non-zero exit status rather than a
summary row.

### Device memory

`--dense-fallback-policy` controls what happens when the exact solver cannot fit
its dense edge tensors in device memory. It defaults to `device-only`.

| Policy | Behaviour on device-memory exhaustion |
| --- | --- |
| `device-only` *(default)* | Never migrates solver work to the host; raises `ExactSolverResourceLimit` instead. |
| `cpu-allowed` | Re-resolves the runtime to CPU and redoes the affected work there. Also rescues an unavailable CUDA device, as noted above. Cannot rescue `--dtype float16`, which has no CPU implementation. |
| `error` | Intended to disable dense fallback outright. |

With the default dense backend, `error` and `device-only` are indistinguishable:
every out-of-memory handler tests only for `cpu_allowed`, and the two branches
that behave differently under `error` are reachable only through
`--inner-backend quotient-workset`. Do not expect `error` to add a guard to a
default-backend run.

For scripting, note the exception hierarchy is not uniform:
`ExactSolverResourceLimit` subclasses **`MemoryError`**, while
`CudaUnavailableError` subclasses **`RuntimeError`**. Catch both explicitly rather
than relying on a common base.

`CLIPP2_MAX_COMPLETE_GRAPH_BYTES` overrides the preflight's memory budget with a
fixed per-process byte cap, taking precedence over both the CUDA and the host
query. It must be a positive number when set. This is the lever to use when the
automatic 80%-of-free heuristic misjudges a shared GPU.

`--workset-max-bytes` and `--compressed-cache-max-bytes` are byte budgets for the
working-set and compressed-certificate machinery, which the dense default backend
does not use; they matter only with `--inner-backend quotient-workset`.

### Reproducibility across devices and runs

For a fixed device and dtype, CPU fits are bit-reproducible: repeated runs of the
same input produce byte-identical cluster, prevalence, multiplicity and
path-posterior tables, and this holds across thread counts too. Only the
`*_elapsed_seconds` columns differ — and because those are embedded in the result
tables, the files themselves are never byte-equal even when every scientific value
is. Compare columns, not checksums.

Across devices the guarantee is weaker than a pure dispatch would give you,
because the device selects the implementation and not merely where it runs:

- **The fusion graph is built by different code on CPU and CUDA.** The non-CUDA
  path uses a NumPy builder in float64; the CUDA path uses a Torch builder at the
  run's dtype. They implement the same formula but not the same arithmetic, and in
  float64 their edge weights agree only to about `6e-16` relative. Consequently
  `original_graph_hash` and `edge_list_hash` **differ between a CPU run and a CUDA
  run of the same input**: those hashes identify a device-and-dtype-specific
  problem, so do not use them to assert two runs solved the same problem across
  devices. Whether that last-bit difference can flip a final label is unresolved.
- **CUDA fits are not bit-reproducible run to run.** The float `index_add_` used
  to accumulate the edge adjoint has duplicate indices and is documented by
  PyTorch as nondeterministic on CUDA. It sits on the hot path of the default
  dense solver.
- **CPU and GPU can assign different labels near lambda-path decision
  boundaries**, where two candidate partitions are nearly tied.

One divergence that does *not* apply to the documented input format: a CUDA-only
partition-refit implementation exists and would feed `bic_loglik` and therefore the
selected cluster count, but it is reachable only for legacy inputs without a
compiled path likelihood. Files using the `clipp2.tumor.long.v1` contract bypass it
on every device.

For reproducible GPU runs, enable `torch.use_deterministic_algorithms(True)` and
set `CUBLAS_WORKSPACE_CONFIG` before fitting; CliPP2 does not enable either itself.
For an auditable result, record `--device` and `--dtype` alongside the
`input_data_hash`, `objective_spec_hash` and `original_graph_hash` that every run
reports, and compare runs only within a fixed device and dtype.

GPU-specific statements in this section are derived from the implementation rather
than measured, since the reference environment used to write them had no usable
CUDA device.

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
