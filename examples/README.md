# Examples

[`tumor/`](tumor/) contains six mutations across two regions:

```text
tumor/
├── region1/    # snv.txt, cna.txt, purity.txt
└── region2/    # snv.txt, cna.txt, purity.txt
```

The regions have independent CN profiles and purities of 0.6 and 0.8.
Locus `1:300` is absent from region2's SNV table but covered by its CNA table;
it is treated as missing, not zero reads. Normal CN is 2. These are illustrative
inputs, not a biological accuracy benchmark.

From the repository root:

```bash
clipp2 fit --input-dir examples/tumor --device cpu --max-clusters 3 --output-dir results/
```

Use `--device cuda` for GPU execution and a new output directory for each run.
For a single-region fit, pass `examples/tumor/region1` instead.

[`exampleTumor1.tsv`](exampleTumor1.tsv) preserves the separate canonical example
(300 mutations, two regions, mixed CN). Use `--input-file examples/exampleTumor1.tsv`
instead of `--input-dir`; it is not a conversion of `tumor/`.
See the [input specification](../README.md#input).
