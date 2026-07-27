# exampleTumor1

`exampleTumor1/` is a complete 300-SNV, two-region CliPP2 input. It contains
10 genomic CN intervals. Sixty percent of its SNVs have two local
allele-specific CN states, 70% lie in a segment with a CNA state, and each
region has non-diploid dominant calls in 50% of the intervals.

Validate and inspect it without fitting:

```bash
python - <<'PY'
from CliPP2 import load_tumor_directory

tumor = load_tumor_directory("examples/exampleTumor1")
print(tumor.tumor_id, tumor.num_mutations, tumor.num_regions)
PY
```

Run it on CPU:

```bash
clipp2 fit \
  --tumor-dir examples/exampleTumor1 \
  --outdir exampleTumor1_results \
  --device cpu
```

Generate a fresh 300-SNV, two-region evolutionary benchmark with hidden
multiplicity truth:

```bash
clipp2 simulate \
  --out-dir generated_examples \
  --tumor-id exampleTumor1 \
  --mutation-count 300 \
  --region-count 2 \
  --seed 1
```

The committed example is a compact observed-input fixture. The generated
benchmark additionally contains only truth-prefixed simulator tables and a
manifest that separates input hashes from truth hashes.
