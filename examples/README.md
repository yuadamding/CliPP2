# exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a compact, complete canonical
12-column input with six mutations across `region1` and `region2`. It covers a
diploid segment, one-state CNA ambiguity, two-state subclonal gain and loss,
amplification, and one explicitly unavailable mutation-region observation.

Every mutation must have one unit for every sample. Repeated rows within a unit
enumerate that sample segment's complete local copy-number state set.

Fit it on CPU:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --device cpu
```

The optional simulator is source tooling rather than part of the inference
wheel. Install the simulation extra and run it from a source checkout:

```bash
python -m pip install -e '.[simulation]'
python -m tools.simulation --out-dir simulated --tumor-id demo
```
