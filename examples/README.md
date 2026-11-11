# Examples

Run the six-mutation, two-region Protocol 2.4 workflow from the repository root:

```sh
clipp2 workflow --plan examples/workflow_protocol24.plan.json --output-dir results/workflow23
```

The [plan](workflow_protocol24.plan.json) declares one configuration for the
[single-state input](workflow_protocol22.tsv). It has no probability or residual
acceptance thresholds. Stage 1 allows 10 clusters and Stage 2 allows 20; CPU is
the default backend. Stage 1 chooses a provisional source nearest all ones;
Stage 3 preserves native alternatives and returns deterministic MAP estimates
for every informative compatible fitted mutation. There is no distance cutoff or
automatic major-CN ceiling.
Protocol 2.2/2.3 plans are retained only for their historical installations.
Set `stage_three_assignment_method` to `"profile"` for regional maximum-likelihood
assignment without cluster posterior probabilities; `"marginal"` remains the default.

Replay its frozen Stage 3 request into a new directory:

```sh
clipp2 assign --bundle results/workflow24/assignment_bundle.json --output-dir results/replayed-assignment
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

See the [workflow definition](../docs/workflow_protocol_2_4.md) and
[simulation guide](../docs/simulation.md). Each command requires a new output
directory; CUDA use requires an explicit CUDA build and device setting.
