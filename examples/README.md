# Examples

## exampleTumor1

[`exampleTumor1.tsv`](exampleTumor1.tsv) is a complete canonical
12-column input with 300 SNVs, samples `region1` and `region2`, and 10 CN
intervals per sample (20 sample-specific intervals). It has 600 mutation-sample
units and 998 data rows because 398 units
(66.3%) contain two local CN states. Every unit has at least one non-diploid
local state.

Under the default whole-mutation CN filter (`--max-major-cn 4`), **all 300
mutations are retained and none are excluded**: the largest major CN is two.
The 199 mutations with mixed CN in at least one sample remain eligible;
subclonal CN alone is not an exclusion reason. A mutation is excluded from
every sample only if any original CN state in any sample exceeds the major-CN
cutoff. The example intentionally preserves its mixed-CN input. From a source
checkout, use
`python -m CliPP2.simulation` from the repository's parent directory for a
fully retained clonal-CN benchmark with sampled integer multiplicity. See
[the simulation guide](../simulation/README.md).

Every mutation must have one unit for every sample. Repeated rows within a unit
enumerate that sample segment's complete local copy-number state set.

Fit it on CUDA (the default):

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```
