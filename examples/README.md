# Examples

## exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a complete canonical
12-column input with 300 SNVs, samples `region1` and `region2`, and 10 sample-specific CN
intervals. It has 600 mutation-sample units and 998 data rows because 398 units
(66.3%) contain two local CN states. Every unit has at least one non-diploid
local state.

Under the current whole-mutation CN filter, **199 mutations are excluded**
because of subclonal CN; **101 mutations remain** for fitting across both
samples. The example intentionally preserves its original mixed-CN input to
demonstrate eligibility filtering. From a source checkout, use
`python -m tools.simulation` for a fully retained clonal-CN benchmark.

Every mutation must have one unit for every sample. Repeated rows within a unit
enumerate that sample segment's complete local copy-number state set.

Fit it on CUDA (the default):

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```
