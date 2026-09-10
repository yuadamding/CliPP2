# Tree-based simulation

This source-tree generator preserves a clone tree and its multi-region CCFs,
but samples mutation multiplicity independently of CNA timing. It does not
claim a physical-copy amplification history for the sampled multiplicity.

## Copy number and multiplicity

- Start with diploid segments and generate gain-only CNA events on the trunk.
  All descendant clones and all regions inherit the same segment CN profile.
  Every mutation-region has one CN state with fraction exactly `1.0`.
- Major CN is at most six. There are no subclonal CN mixtures, descendant
  CNA events, losses, copy-neutral LOH, or whole-genome duplication.
- For each mutation with `major_cn != minor_cn`, independently draw an integer
  uniformly from `1, ..., min(major_cn, 6)`, with both endpoints included.
  Equal-CN mutations have multiplicity one.
- Draw once per mutation, not once per region. All carrier clones and regions
  use that same multiplicity. This applies to clonal and subclonal mutations;
  removing subclonal **CN** does not remove subclonal mutation clusters.
- Cluster `0` is nonempty and has CCF exactly `1.0` in every region.

For purity `rho`, mutation CCF `phi`, and sampled multiplicity `m`:

```text
p_alt = rho * phi * m / (rho * (major_cn + minor_cn) + (1 - rho) * 2)
depth ~ Poisson(mean_depth)
alt_count ~ Binomial(depth, p_alt)
ref_count = depth - alt_count
```

Separate named RNG streams control topology, clone fractions, mutation counts,
segments, CNA events, multiplicity, purity, depth, and read counts. Repeating a
seed and configuration reproduces the bundle; changing depth does not change
the latent mutation truth.

## Run

From the workspace containing the `CliPP2` repository, using conda `ml1`:

```bash
conda run -n ml1 python -m CliPP2.simulation \
  --out-dir simulations --tumor-id example --seed 1 \
  --mutation-count 300 --clone-count 3 --region-count 2
```

This is a source-only utility, not part of the installed inference wheel.
Existing tumor directories are never overwritten. The former two-state-CN
controls have been removed; they are not accepted compatibility switches.

## Truth and provenance

- `<tumor_id>.clipp2.txt`: canonical observed input; validation requires every
  mutation to survive CliPP2's whole-mutation CN filter.
- `truth.txt`: mutation-to-cluster labels.
- `truth_clone_sample.txt`: clone CCF by sample; sample `0` means `region1`.
- `regionN/truth_cp.txt`: per-mutation CCF in that region.
- `truth_mutation_sample.tsv`: integer `multiplicity`, CCF, mutant-copy mass,
  effective multiplicity, total CN, and expected VAF. Use `multiplicity` as the
  exact-class truth target; effective multiplicity agrees up to round-off.
- `truth_mutation_clone_dosage.tsv`: sampled dosage for each carrier clone;
  noncarriers have `carrier=0` and missing dosage, not multiplicity zero.
- `truth_mutation_history.tsv`: origin clone, segment, sampled multiplicity,
  and its sampling rule. Physical-copy indices and mutation timing are removed.
- `truth_cna_events.tsv`: trunk gains that generated the CN profile, not an
  explanation of sampled mutation dosage. Other tree, CN, and position truth
  tables retain the framework's layout.
- `scenario_manifest.json`: generator version, RNG streams, intended/realized
  conditions, source hash, and input/truth file hashes.

The generator is `tree_clonal_cn_uniform_multiplicity_v7`, output schema `7.0`.
These versions distinguish the changed scientific design and truth fields;
they do not change the CliPP2 inference version. Existing cohorts are unchanged.
For CNA-only multiplicity performance, use exact `major_cn != minor_cn` rows
and pooled exact-class macro-F1, with eligible counts and per-class F1.
