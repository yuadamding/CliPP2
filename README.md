# CliPP2

CliPP2 clusters cellular-prevalence profiles from single- or multi-region tumor
TSV data using objective-faithful pairwise fusion.

## Install

```bash
pip install .
```

## Fit

```bash
clipp2 fit --input-file tumor.tsv --outdir clipp2_results
```

For a directory of tumor TSV files, use `--input-dir` instead. Exactly one input
selector is required.

The production profile is explicit:

- CUDA execution with float64 tensors
- dense exact fusion with device-only fallback
- online partition-guided ADMM lambda selection
- assignment-aware partition ICL scoring

CUDA is therefore required by default. Use `--device auto` to permit CPU
selection when CUDA is unavailable, or `--device cpu` to request CPU execution.
CPU fallback during an accelerator fit remains disabled unless
`--dense-fallback-policy cpu-allowed` is supplied.

Run `clipp2 fit --help` for custom graphs, legacy adaptive-BIC selection, and
advanced solver controls. The module entry point is equivalent:

```bash
python -m CliPP2 fit --input-file tumor.tsv
```
