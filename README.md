# CliPP2

CliPP2 estimates mutation clusters and regional cancer cell fractions (CCFs)
jointly across tumor regions. **Protocol 3.0** fits one mixture: an optional
clonal class fixed at `(1,1,…)` and free-CCF classes with **multiplicity 1**
in every informative region. Clonal multiplicity may be higher, subject to
the original major-CN bounds and a cross-region range of at most one.

## Install and run

Requires Python 3.10+ and a CPU. No CUDA, PyTorch or compiler is required.

```sh
python -m pip install .
clipp2 workflow --plan examples/plan.json --output-dir results/example
```

Use a new output directory. Stage 1 discovers the mixture from deterministic
single-copy starts; Stage 2 refines the **same mutations and likelihood** and
retains the Stage 1 fit if no candidate improves its BIC. Stage 3 publishes
fitted biological probabilities and multiplicities without refitting.
The default biological cluster capacities are **10 → 20**. There are no read
floors, donor caps, or automatic near-clonal merges.

See the [example](examples/README.md) for the v8 JSON plan and the
[scientific protocol](docs/workflow_protocol_3_0.md) for formulas, search
limits, and output details. Version 2.5 plans and fits are not replayable.

## Input

Provide either a tumor directory with one `snv.txt`, `cna.txt`, and
`purity.txt` per region, or a canonical tumor TSV (gzip supported) with
`mutation_id`, `sample_id`, `alt_count`, `ref_count`, `count_observed`,
`purity`, `normal_cn`, `segment_id`, `cn_state_id`, `cn_state_fraction`,
`allele_a_cn`, and `allele_b_cn`. The canonical row uses one representative
state with fraction 1; optional `segment_cn_states` preserves other states.

The likelihood uses the original representative **integer** CN, not weighted
total CN. For multiple regional CNA states, the pair with the largest
`(major_cn, minor_cn)` supplies representative CN; state fractions are
retained as provenance. Supply normal CN or explicitly set
`"normal_cn_policy": "diploid"`. A major-CN cutoff is optional and masks
only that mutation–region pair. Missing reads default to ALT 0 and the region's
median observed positive depth; set `"missing_counts": "unobserved"` to omit
them from the likelihood.

## Output

Each stage writes `{tumor_id}_mutation_clusters.tsv`,
`{tumor_id}_cluster_centers.tsv`, and
`{tumor_id}_mutation_region_multiplicity.tsv`. Stage 3 also writes
`{tumor_id}_cluster_probabilities.tsv`. Label **0** is reserved for an active
biological clonal class. Multiplicity is NA for missing, excluded, zero-depth,
or zero-CCF pairs. The workflow root records fitted states, search coverage,
the original catalog, and accounting for all original mutations.

## Contact

Submit an issue or email yding4@mdanderson.org.
