# exampleTumor1

[`exampleTumor1.clipp2.txt`](exampleTumor1.clipp2.txt) is a complete canonical
input with 300 SNVs, samples `region1` and `region2`, and 10 sample-specific CN
intervals. It has 600 mutation-sample cells and 998 data rows because 398 cells
(66.3%) contain two local CN states. Every cell has at least one non-diploid
local state.

Validate and inspect it:

```bash
clipp2 validate --input-file examples/exampleTumor1.clipp2.txt
```

```python
from CliPP2 import load_tumor_txt

tumor = load_tumor_txt("examples/exampleTumor1.clipp2.txt")
print(tumor.tumor_id, tumor.num_mutations, tumor.region_ids)
```

Run it on CPU:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.clipp2.txt \
  --outdir exampleTumor1_results \
  --device cpu
```

Generate a fresh evolutionary benchmark with hidden multiplicity truth:

```bash
clipp2 simulate \
  --out-dir generated_examples \
  --tumor-id exampleTumor1 \
  --mutation-count 300 \
  --region-count 2 \
  --seed 1
```

The generated benchmark bundle contains the canonical input, truth-prefixed
tables, sample truth subdirectories, and a manifest that hashes input and truth
separately.
