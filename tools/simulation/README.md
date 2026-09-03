# Optional simulation tool

This source-only tool generates canonical CliPP2 inputs but is intentionally
excluded from the inference wheel. From a repository checkout, install the
optional dependency and run:

```bash
python -m pip install -e '.[simulation]'
python -m tools.simulation --out-dir simulated --tumor-id demo
```

The generated TSV remains consumable by `clipp2 fit`. Simulation depends on
pandas; inference does not.
