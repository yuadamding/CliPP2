# exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a complete canonical
input with 300 SNVs, samples `region1` and `region2`, and 10 sample-specific CN
intervals. It has 600 mutation-sample units and 998 data rows because 398 units
(66.3%) contain two local CN states. Every unit has at least one non-diploid
local state.

The file carries only the 12 required columns — the ones the objective is
computed from — demonstrating the compact form of the schema. The seven optional
identity/coordinate columns (`chromosome`, `position`, `ref`, `alt`,
`segment_start`, `segment_end`, `allele_mode`) are omitted, so `allele_mode`
takes its default, `unphased`. Writers such as `clipp2 simulate` emit the full
19-column form; both load to bit-identical data.

Validate and inspect it:

```bash
clipp2 validate --input-file examples/exampleTumor1.tsv
```

```python
from CliPP2 import load_tumor_txt

tumor = load_tumor_txt("examples/exampleTumor1.tsv")
print(tumor.tumor_id, tumor.num_mutations, tumor.region_ids)
```

Run it on CPU. `--device cpu` is required because `--device` defaults to `cuda`,
and limiting the Torch thread count is strongly recommended — on this 300-SNV
input it is the difference between finishing in about half an hour and not
finishing at all:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results \
  --device cpu
```

See [Device and precision](../README.md#device-and-precision) for why, and for
the GPU equivalents. Paths above are relative to the package directory; a
pip-installed copy keeps these files under
`site-packages/CliPP2/examples/`.

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
tables, one `regionN/` truth subdirectory per region, and a manifest that hashes
input and truth separately.
