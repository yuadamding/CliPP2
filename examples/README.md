# Examples

From the repository root:

```sh
clipp2 workflow --plan examples/plan.json --output-dir results/example
```

The plan uses `clipp2.workflow.plan.v8` and protocol `3.0`. Paths are relative
to the plan. The six-SNV [tumor](tumor/) has two region directories; use
`"input_path": "tumor/region1"` for a single-region example.

Each region has `snv.txt` (locus and read counts), `cna.txt` (nonoverlapping
inclusive intervals and major/minor/total CN), and `purity.txt` (one fraction).
The fixture supplies normal CN. For mixed CN, repeat the interval with
distinct `cn_state_id` values and `cn_state_fraction` summing to 1:

```text
chromosome_index	start_position	end_position	major_cn	minor_cn	total_cn	normal_cn	cn_state_id	cn_state_fraction
1	1	1000000	3	1	4	2	state_a	0.6
1	1	1000000	2	1	3	2	state_b	0.4
```

The likelihood uses `(3,1)`, not the weighted total 3.6. Neither state
fractions nor read-based multiplicity estimates restrict the clonal support.

See the [protocol](../docs/workflow_protocol_3_0.md) for plan options and outputs.
