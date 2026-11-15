# Examples

Run the six-mutation, two-region workflow from the repository root:

```sh
clipp2 workflow --plan examples/workflow_protocol25.plan.json --output-dir results/workflow25
```

The [plan](workflow_protocol25.plan.json) uses Protocol 2.5, CPU execution,
marginal fitting and assignment, and cluster limits of 10 and 20 for Stages 1
and 2. Its [input](workflow_protocol25.tsv) supplies normal CN explicitly.
Set `fit.max_multiplicity_difference` to `null` for independent regional
multiplicities, or `stage_three_assignment_method` to `"profile"` for profile
assignment. See the [workflow definition](../docs/workflow_protocol_2_5.md).

Replay the frozen Stage 3 request, or run a standalone fit:

```sh
clipp2 assign --bundle results/workflow25/assignment_bundle.json --output-dir results/replayed-assignment
clipp2 fit --input-file examples/workflow_protocol25.tsv --max-clusters 3 --output-dir results/example-fit
clipp2 fit --input-dir examples/tumor --max-clusters 3 --output-dir results/regional-fit
```

Each command uses a new output directory. The [`tumor/`](tumor/) fixture has
`snv.txt`, `cna.txt`, and `purity.txt` in each region, with explicit normal CN.
Pass `examples/tumor/region1` for a single-region fit.

Missing regional SNVs are filled automatically with ALT=0 and REF equal to the
region's median observed positive depth. Raw observations remain in the input
catalog. `missing_counts="unobserved"` opts out of this imputation.

For mixed CN, repeat the same segment coordinates with one row per state:

```text
chromosome_index	start_position	end_position	major_cn	minor_cn	total_cn	normal_cn	cn_state_id	cn_state_fraction
1	1	1000000	3	1	4	2	state_a	0.6
1	1	1000000	2	1	3	2	state_b	0.4
```

Here the weighted total CN is 3.6 and the representative state is `(3,1)`.
The fractions describe all tumor cells and must sum to one. The reader retains
both states for read-floor estimation; the representative CN supplies the
likelihood denominator and major-CN ceiling. Canonical TSVs retain these states
in `segment_cn_states` while preserving their explicit major/minor CN.

The [input contract](../README.md#inputs-and-outputs) defines CN validation,
read floors and their reconciliation with the joint range rule. The
[simulation guide](../docs/simulation.md) describes cohort execution.
