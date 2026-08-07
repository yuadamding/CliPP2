# exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a complete canonical
input with 300 SNVs, samples `region1` and `region2`, and 10 sample-specific CN
intervals. It has 600 mutation-sample units and 998 data rows because 398 units
(66.3%) contain two local CN states. Every unit has at least one non-diploid
local state.

The file carries 12 required columns.

Validate and inspect it:

```bash
clipp2 validate --input-file examples/exampleTumor1.tsv
```

