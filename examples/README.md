# Examples

Run the six-mutation, two-region Protocol 2.5 workflow from the repository root:

```sh
clipp2 workflow --plan examples/workflow_protocol25.plan.json --output-dir results/workflow25
```

The [plan](workflow_protocol25.plan.json) declares one configuration for the
[single-state input](workflow_protocol25.tsv). It has no probability or residual
acceptance thresholds. Stage 1 allows 10 clusters and Stage 2 allows 20; CPU is
the default backend. Stage 1 chooses a provisional source nearest all ones;
Stage 3 uses the fixed all-one reference and distinct Stage 2 centers, returning MAP estimates
for every informative compatible fitted mutation. There is no distance cutoff or
automatic major-CN ceiling.
Set `stage_three_assignment_method` to `"profile"` for joint maximum-likelihood
assignment without cluster posterior probabilities; `"marginal"` remains the default.
The default `max_multiplicity_difference: 1` restricts each mutation's multiplicity
range across informative regions. Set this fit field to `null` for the independent
comparison; the CLI equivalent is `--max-multiplicity-difference none`.

Replay its frozen Stage 3 request into a new directory:

```sh
clipp2 assign --bundle results/workflow25/assignment_bundle.json --output-dir results/replayed-assignment
```

For a standalone fit, use the same TSV or the regional directory fixture:

```sh
clipp2 fit --input-file examples/workflow_protocol25.tsv --max-clusters 3 --output-dir results/example-fit
clipp2 fit --input-dir examples/tumor --max-clusters 3 --output-dir results/regional-fit
```

[`tumor/`](tumor/) has `snv.txt`, `cna.txt`, and `purity.txt` under each region.
Its CNA tables explicitly provide normal CN. Pass `examples/tumor/region1` for
a single-region fit.

SNVs detected in any region are automatically included in every region. Where
an SNV is absent from `snv.txt`, ALT becomes 0 and REF becomes that region's
median observed total depth. For example, observed depths 80, 100, and 140 give
every missing SNV in that region ALT=0, REF=100. No extra flag is required.
Each region supplies its own median; imputed pairs never donate depth, and all
three workflow stages reuse the same counts.

For a segment with multiple CN states, supply one `cna.txt` row per state at
the **same** chromosome and inclusive start/end coordinates, adding
`cn_state_id` and `cn_state_fraction`:

```text
chromosome_index	start_position	end_position	major_cn	minor_cn	total_cn	normal_cn	cn_state_id	cn_state_fraction
1	1	1000000	3	1	4	2	state_a	0.6
1	1	1000000	2	1	3	2	state_b	0.4
```

Fractions are among **all tumor cells**, not just mutation carriers. State IDs
must be unique within an interval; fractions must be finite, nonnegative, and
sum to 1. CN must use integers with 0 ≤ minor ≤ major, and
`total_cn` must equal their sum. States in an interval must agree on normal CN
and provenance flags. A mixed segment can include a `(0,0)` deletion state;
its representative major CN must be at least 1. Distinct intervals cannot overlap. Single-state CNA
tables without the two added columns retain fraction 1.

The example gives weighted total CN `4 * 0.6 + 3 * 0.4 = 3.6`. For the
existing likelihood, the representative state is the largest `(major_cn,
minor_cn)` pair in that order, here `(3, 1)`. Its major CN 3 is the input
multiplicity ceiling. The full states are retained in the optional canonical
TSV field `segment_cn_states`, a JSON list; the ordinary canonical CN columns
describe the representative state. This provides weighted CN for the lower
bound without implementing a mixed-state likelihood.

For the two-state example, the `segment_cn_states` cell contains:

```json
[{"cn_state_id":"state_a","cn_state_fraction":0.6,"major_cn":3,"minor_cn":1},{"cn_state_id":"state_b","cn_state_fraction":0.4,"major_cn":2,"minor_cn":1}]
```

Stages 1 and 2 use the lower bound
`max(1, min(M, floor((ALT / depth) / purity * (purity * weighted_total_cn + 2 * (1 - purity)))))`.
Normal-cell CN is 2 for this calculation, and there is no fixed cap of 4.
Zero ALT, missing observations, and zero depth yield lower bound 1. Stage 2
raises a conflicting segment cap to this lower bound, capped at `M`;
Stage 3 restores candidates `1..M`. These bounds do not divide by mutation
CCF, so subclonal multiplicity can be underestimated. Per-pair estimates are
published in `multiplicity_lower_bounds.tsv`.

See the [workflow definition](../docs/workflow_protocol_2_5.md) and
[simulation guide](../docs/simulation.md). Each command requires a new output
directory; CUDA use requires an explicit CUDA build and device setting.
