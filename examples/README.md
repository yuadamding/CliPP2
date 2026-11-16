# Examples

Run the six-mutation, two-region example from the repository root. Save this as
`plan.json`; input paths are relative to the plan file:

```json
{
  "schema": "clipp2.workflow.plan.v7",
  "protocol_version": "2.5",
  "input_path": "examples/tumor",
  "workflow_config": {}
}
```

Defaults use CPU execution, marginal fitting and assignment, and cluster limits
of 10 and 20 for Stages 1 and 2. The input supplies normal CN explicitly.
Set `fit.max_multiplicity_difference` to `null` for independent regional
multiplicities, or `stage_three_assignment_method` to `"profile"` for profile
assignment. See the [workflow definition](../docs/workflow_protocol_2_5.md).

Run all three stages in a new output directory:

```sh
clipp2 workflow --plan plan.json --output-dir results/workflow25
```

Both fitting stages use a fresh A initializer followed by B mixture fitting;
Stage 2 uses caps recomputed from Stage 1's B result. Saved fits cannot be replayed.
The [`tumor/`](tumor/) fixture has
`snv.txt`, `cna.txt`, and `purity.txt` in each region, with explicit normal CN.
Set `input_path` to `examples/tumor/region1` for a single-region workflow, or to
a canonical TSV for the other supported input format. The only optional
top-level plan field is `run_manifest_id`; saved-fit and build-manifest fields
are rejected.

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
read floors and their reconciliation with the joint range rule.
