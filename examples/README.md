# exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a complete canonical
input with 300 SNVs, samples `region1` and `region2`, and 10 sample-specific CN
intervals. It has 600 mutation-sample units and 998 data rows because 398 units
(66.3%) contain two local CN states. Every unit has at least one non-diploid
local state.

The authoritative schema, metadata defaults, missing-count rules, and local
copy-number-state invariants are in the
[input-format contract](../../docs/input-format.md). In particular, every mutation
must have one unit for every sample, and repeated rows within a unit enumerate
the complete state set for that sample's segment.

Fit it on CPU:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --device cpu
```
