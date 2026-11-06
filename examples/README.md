# Examples

Run the six-mutation, two-region Protocol 2.2 workflow from the repository root:

```sh
clipp2 workflow --plan examples/workflow_protocol22.plan.json --output-dir results/workflow22
```

The [plan](workflow_protocol22.plan.json) supplies explicit decision thresholds
for the [single-state input](workflow_protocol22.tsv). These are fixture settings,
not calibrated biological recommendations. Stage 1 allows 10 clusters and
Stage 2 allows 20. The default backend is CPU.

Replay its frozen Stage 3 request into a new directory:

```sh
clipp2 assign --bundle results/workflow22/assignment_bundle.json --output-dir results/replayed-assignment
```

For a standalone fit, use the same TSV or the regional directory fixture:

```sh
clipp2 fit --input-file examples/workflow_protocol22.tsv --max-clusters 3 --output-dir results/example-fit
clipp2 fit --input-dir examples/tumor --max-clusters 3 --output-dir results/regional-fit
```

[`tumor/`](tumor/) has `snv.txt`, `cna.txt`, and `purity.txt` under each region.
Its CNA tables explicitly provide normal CN. Pass `examples/tumor/region1` for
a single-region fit. Every tumor CN observation has one integer state; mixed
states are rejected by fitting and workflow entry points.

See the [workflow definition](../docs/workflow_protocol_2_2.md) and
[simulation guide](../docs/simulation.md). Each command requires a new output
directory; CUDA use requires an explicit CUDA build and device setting.
