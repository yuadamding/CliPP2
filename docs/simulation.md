# Exact phylogeny cohort simulation

The `CliPP2.simulation` module implements the supplied Version 1.2 cohort and
Protocol 2.2 paired experiment. The authoritative biological input still declares
Protocol 2.1 and is preserved unchanged. New contracts record
`source_cohort_protocol_version: "2.1"` and execution `protocol_version: "2.2"`.
The authoritative input is
`CliPP2_phylogeny_cohort_settings.json` in the repository root. Its 60 variants
span all 12 families: 600 main histories and 120 development histories, each with
1,500 mutations, 50 segments, and regions `R1`–`R4`. Main histories have four
acquisitions; development histories have A1 and A4: 2,640 datasets in total.

Qualification is separate from a completed native study. A successful replay or
bounded pilot does not establish that the full cohort has finished. Use the
execution records and reports to distinguish completed, failed, and unexecuted
cells; this document does not report study results.

## Environment and commands

Use Python **3.11 or newer** and NumPy **2.3.5** for the exact cohort. The adapter
checks the NumPy version before generating coordinates or reads. From the
repository root, install into a dedicated environment:

```sh
python3.11 -m venv .venv-simulation
source .venv-simulation/bin/activate
python -m pip install '.[simulation]'
python -m CliPP2.simulation --help
```

`clipp2-simulate` is the equivalent installed entry point. Use separate, new
output directories for qualification, materialization, and study execution.

```sh
python -m CliPP2.simulation qualify \
  --cohort-settings CliPP2_phylogeny_cohort_settings.json \
  --outdir results/cohort-qualification --workers 3 --native-timeout 180

python -m CliPP2.simulation run \
  --cohort-settings CliPP2_phylogeny_cohort_settings.json \
  --profile full --workers 10 --outdir results/cohort-full-protocol2.2
```

`qualify` replays all 720 histories, checks canonical input roundtrips, and runs
bounded native pilots. `--skip-native` limits it to the non-fitting checks.
Separate controlled cohort diagnostics include 11 fixed-reference assignment
passes; these are not optimizer fits or independent main histories.

The full profile retains the entire supplied cohort and can take many hours or
days. It does not impose the default balanced profile's timing-based allocation
reduction or a one-hour run deadline. Ten workers use one native CPU thread
each. `--budget-minutes` and `--disk-budget-gib` inform balanced allocation;
they do not make the full profile a smaller study. Optional panels remain
disabled as specified in the source JSON.

For a long run, keep the installed environment unchanged. A non-editable wheel
in a dedicated environment isolates worker imports from later repository edits.
The launcher can retain an external source snapshot/build record. The study
retains its supplied link and frozen cohort settings without scanning source
directories or invoking Git. Set `CLIPP2_RUN_MANIFEST_ID` (or `run --run-manifest-id`)
and optionally `CLIPP2_RUN_BUILD_MANIFEST`; `--source-record` accepts an externally
prepared report record. Missing external evidence is explicitly unbound.

```sh
python -m CliPP2.simulation status results/cohort-full
python -m CliPP2.simulation run --outdir results/cohort-full --resume
python -m CliPP2.simulation run --outdir results/cohort-full --resume --retry-failed
python -m CliPP2.simulation report results/cohort-full
```

`status` reads progress and in-flight operation journals without changing the
study. Resume uses the existing contract, allocation, and selected thresholds;
new command-line settings do not replace them. Completed cells are reused.
`--retry-failed` reruns execution failures as new attempts, retaining old
attempts, completion records, and costs. It does not reopen a frozen development
selection or reinterpret scientific rejection as a worker failure. Package
version 1.2.2 uses simulation contract v4. It binds the entire parsed cohort
settings to a `cohort_content_id` in contracts, jobs, physical histories,
acquisitions, and qualification receipts. Formatting and object-key order do not
affect this identity; values and list order do. Resume verifies the frozen
settings before reusing any work. Semantic random seeds remain unchanged.

Existing 1.2.1 studies retain their original frozen installation and outputs.
They are not migrated or given retrospective cohort identity. A new execution
under 1.2.2 requires a new directory. Qualification has separate numbered
mechanism/cohort attempts; validated published summaries can complete an
interrupted wrapper without rerunning controls. Interrupted costs without a
summary remain explicitly unknown. Reports use new directories.

To apply new evaluation definitions to saved inference, create an explicit JSON
configuration and a new report destination:

```json
{"schema":"clipp2.simulation.evaluation-config.v1","reference_distance":"l2","reference_match_atol":0.025}
```

```sh
clipp2-simulate report results/old-study \
  --evaluation-config evaluation.json --report-dir results/new-evaluation-report
```

The tolerance above is an example reporting choice. Offline evaluation reads
preserved truth, input catalogs, acquisition records, selected attempts, and
paired results, including archives. Missing or conflicting evidence produces an
unavailable derived record. It performs no fitting, assignment, generation, or
threshold selection; frozen development choices and old inference decisions
remain intact. Without `--evaluation-config`, reports use cached evaluations.

Generate every truth history and prescribed input catalog without fitting:

```sh
python -m CliPP2.simulation materialize \
  --cohort-settings CliPP2_phylogeny_cohort_settings.json \
  --outdir results/cohort-materialized
```

## Biological and numerical authority

Physical SNV, copy-gain, and WGD events are replayed through the supplied tree.
Final copy states must match `validated_copy_truth`; sampled tip fractions are
used as supplied. The adapter verifies constant integer multiplicity among
carriers, regional single-state CN, and the physical reduction to the read model.
Normal CN is explicitly supplied, and original CN determines the denominator.

Source history, variant, mutation, class, and batch identities are retained.
Numerical arrays use canonical mutation-ID order; metadata retains the original
prescribed ranks and coordinates. Rank-based S10 masks use those original ranks,
not canonical array positions. Missing counts, observed zero ALT, zero depth,
and explicit input exclusion remain distinct; excluded pairs retain raw counts.

Valid supplied CCF values retain their exact floating-point values. Five source
history/class/region endpoints equal `1.0000000000000002`: T/R3 in S07.03 main
replicates 1, 2, 6, 10 and development replicate 2. These alone are normalized to
one after proving that every positive-fraction genotype carries the mutation.
History metadata records the original value and hexadecimal bits, affected IDs,
and reason. This corrects endpoint roundoff without changing the source JSON.

Fresh PCG64 generators use the prescribed semantic seed for each coordinate,
depth, and ALT draw. SHA256 here maps compact JSON semantic keys to seed128, as
required by the supplied specification; it is not a file checksum. Paired masks
reuse identical latent reads. The native optimizer is deterministic and exposes
no public random fit seed; source fit-seed templates do not alter its execution.

## Experiment, development, and accounting

Each core dataset has up to three fits: A fits the full population, B fits the
remainder with original support, and C fits it with contributor-derived caps.
B3 and C3 each perform one frozen-reference assignment with fresh full original
support. The one-configuration core maxima are **7,920 fits and 5,280 assignment
passes**. Blocked or inapplicable branches may execute fewer operations.

Stage 1 still requires strict clonal geometry, coverage and descriptive screens.
In each Stage 2 arm, geometry alone routes every near-clonal group to reassessment,
even when capped or full-support companion diagnostics fail. Other groups retain
their native accepted centers. Reference quality records regional source coverage
and native descriptive screening under that arm's accepted support. A target whose
winning reference fails quality in an informative region is `unresolved_reference`;
its full scores remain available, with no runner-up substitution or renormalization.

Development evaluates contributor thresholds `.90, .95, .99` and posterior
thresholds `.80, .90, .95, .99`. Posterior alternatives reuse computed probabilities;
they do not trigger new native scoring and retain the same reference-quality gate.
Validated A/B fits are reused across
contributor settings: up to five fits per development dataset with successful
reuse, or nine when reusable fits are unavailable. Development-grid overhead,
qualification, and retry attempts have separate costs from the core maxima.

Clonal proximity is fixed at strict unnormalized `||phi - 1||₂ < .05`; cluster
limits are 10 in Stage 1 and 20 in Stage 2. Descriptive research screens use
pair deviance 9, mean deviance 4, and outlier fraction `.25`. They are not
calibrated statistical tests or clinically validated cutoffs.

`source_to_reference.tsv` records the construction mapping. Authoritative final
rows distinguish `baseline_reference_label` from `final_assignment_label`:
accepted targets receive the accepted label, unresolved targets have no final
label, and non-targets retain theirs. All reference labels remain, including zero
occupancy. Target origins and historical cap contributors that later reassign are
reported separately; historical caps are never recomputed from reassessment.
Both arms use the production transition and final-membership helpers. Scoring is
followed by reconciliation and one publication with combined target/non-target
accounting. Accepted native counts remain recorded if a later derived phase fails.

Reference-center assessment is separate from source-member overlap used for
calibration labels. Continuous Euclidean and maximum regional errors are retained.
Binary reference recovery is unassessed by default. The qualification-only
`qualify --reference-match-atol VALUE` declares an unnormalized full-vector L2
error bound for new evaluations; recall uses one-to-one matching, so one candidate
cannot recover several true groups. Cached historical L∞ results keep their
original definition. See [evaluation definitions](simulation_evaluation.md) for
detailed membership, target-specific availability, and offline re-evaluation. This reporting tolerance never changes fitting, routing or assignment.

Threshold selection uses a fixed original evaluable population in S01, S02,
and S07, with complete-truth separation under original support greater than two
natural-log units. Missing fits and decisions retain their denominator and earn
zero correct-decision credit. Families receive equal weight; all 12 families
remain in the reporting population. A qualifying setting needs positive target
acceptance and selective error at most `.05`. If none qualifies after the allowed
development extension, execution stops before main fitting; continuing requires
a separately declared research revision, not an automatic relaxation.

## Retained outputs and Python API

The output root retains `study_contract.json`, `cohort_settings.json`, allocation,
development selection, the frozen study plan, truth histories, and job manifests.
Each cell retains input counts, acquisition metadata, evaluations, cost ledgers,
and operation/failure records. Completed configuration directories containing
verified native fits, frozen references, and assignment outputs are compressed
into ZIP files and checked before their uncompressed copies are removed. Raw
inputs, truth, metrics, and journals remain available. Reports include HTML,
figures, `aggregate.json`, `tumor_metrics.tsv`, and `execution_ledger.tsv`.
The dataset's A/B/C fits retain one external `run_manifest_id` and bound original
biological/numerical input identity. Optional supplied build metadata is retained
through `CLIPP2_RUN_BUILD_MANIFEST`; source directories are not inspected. Replay
preserves the original producer link and adds the current execution link separately.
No file-byte checksum is used. Declared truth/reference
controls explicitly identify their fixed-coordinate quality intervention; they do
not claim acceptance as learned native reference groups.

```python
from CliPP2.simulation.cohort import load_cohort, cohort_history, acquire_cohort
from CliPP2.io.tumor_txt import write_tumor_txt

cohort = load_cohort("CliPP2_phylogeny_cohort_settings.json")
history = cohort_history(cohort, "development.S03.02.01")
frame = acquire_cohort(history, cohort, "A1", read_replicate_id=1)
history.write_truth("results/example-truth")
write_tumor_txt("results/example-input.tsv", frame, frame.attrs["metadata"])
```

`History.to_dict()` / `History.from_dict()` preserve replayable truth. The
generic generators in `biology.py`, `acquisition.py`, and `panels.py` support
separate fixtures and optional experiments; they do not replace the authoritative
cohort or supply additional independent core histories.
