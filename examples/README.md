# Example

[`exampleTumor1.tsv`](exampleTumor1.tsv): 300 mutations, two regions; all pass
the default CN filter. Mixed states collapse to maximum major and total CN per
mutation–region pair; minor CN is their difference. Fractions do not affect fitting.

Run from the repository root (CUDA by default):

```bash
clipp2 fit --input-file examples/exampleTumor1.tsv --outdir exampleTumor1_results
```

[Input/filter rules](../README.md#input) · [Simulation generator](../simulation/README.md)
