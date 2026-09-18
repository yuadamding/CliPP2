# Tree-based simulation

This source-tree generator shares a mutation clone tree across regions, but
generates CN profiles independently for each region. Multiplicity is sampled
from local CN independently of CNA timing. This is a conditional observation
model, not a jointly reconstructed CNA/physical-copy evolutionary history.

## Copy number and multiplicity

- Start each region with diploid segments and independently generate gain-only
  CNA events using separate regional RNG streams. All clones **within a region**
  have the same local profile; profiles are not copied between regions.
  Every mutation-region has one CN state with fraction exactly `1.0`.
  Independent draws can coincide, particularly with zero CNA rate or saturated CN;
  the generator does not force profiles to differ.
- Major CN is at most six. There are no subclonal CN mixtures, descendant
  CNA events, losses, copy-neutral LOH, or whole-genome duplication.
- For each mutation-region with local `major_cn != minor_cn`, independently draw
  an integer uniformly from `1, ..., min(major_cn, 4)`, endpoints included.
  Equal-CN mutations have multiplicity one under this simulator's sampling rule.
  **Balanced CN means only `major_cn = minor_cn = 1`**; higher-copy equal-CN
  states such as `2/2` are not balanced. The equal-CN sampling rule is separate
  from that definition and from inference's `1..major_cn` candidate support.
- All carrier clones **within that region** use its sampled multiplicity.
  Regions do not share multiplicity draws. This applies to clonal and subclonal mutations;
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
the latent mutation truth. CNA and multiplicity streams have named, independently
spawned regional children recorded in the manifest. The event-rate parameter
applies separately in each region, not as a tumor-wide budget divided by regions.

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

The generator's six-copy CN range is independent of the inference default
`--max-major-cn 4`. Bundle validation explicitly uses a limit of six so it can
check all generated truth. To retain the complete generated cohort during
fitting, pass `--max-major-cn 6`; fitting with the default may exclude generated
mutations having major CN 5 or 6 in **any** region. The CN ceiling remains six;
the independent multiplicity sampling ceiling is four.

## Truth and provenance

- `<tumor_id>.clipp2.txt`: canonical observed input; validation requires every
  mutation to survive CliPP2's whole-mutation CN filter with `max_major_cn=6`.
- `truth.txt`: mutation-to-cluster labels.
- `truth_clone_sample.txt`: clone CCF by sample; sample `0` means `region1`.
- `regionN/truth_cp.txt`: per-mutation CCF in that region.
- `truth_mutation_sample.tsv`: integer `multiplicity`, CCF, mutant-copy mass,
  effective multiplicity, total CN, and expected VAF. Use `multiplicity` as the
  exact-class truth target; effective multiplicity agrees up to round-off.
- `truth_mutation_carriers.tsv`: shared mutation-clone membership. Combine it
  with regional `truth_mutation_sample.tsv` for carrier dosage; noncarriers have
  no multiplicity target. This factorization replaces the old shared-dosage table
  without expanding to a mutation-clone-region table.
- `truth_mutation_history.tsv`: shared origin clone and segment; no misleading
  single multiplicity column. Physical-copy indices and mutation timing are absent.
- `truth_cn_sample.tsv`: one CN profile row per segment and numeric sample ID,
  replacing `truth_cn_clone_profile.tsv`. `regionN/truth_cn_states.tsv` records
  that region's single local state per segment.
- `truth_cna_events.tsv`: local gains, with `sample_id`; event IDs are local to
  each sample. Clone 0 marks a local clonal state, not a common ancestral CNA
  across regions. Events explain local CN, not sampled mutation dosage.
- `scenario_manifest.json`: generator version, RNG streams, intended/realized
  conditions, source hash, and input/truth file hashes.

The generator is `tree_regional_clonal_cn_uniform_multiplicity_v8`, schema `8.0`.
These versions distinguish the changed scientific design and truth fields;
they do not change the CliPP2 inference version. Existing cohorts are unchanged.
For CNA-only multiplicity performance, use `(major_cn != 1) | (minor_cn != 1)`
rows: exclude only `1/1`, including higher-copy equal-CN states. Report pooled
exact-class macro-F1 with eligible counts and micro-, weighted-, and per-class
F1. This evaluation filter does not change the multiplicity-sampling rule above.
