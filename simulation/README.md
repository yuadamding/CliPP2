# Tree-based simulation

Source-only generator: one mutation clone tree, independent regional CN and
dosage. This is a conditional observation model, **not** a reconstructed
CNA/physical-copy evolutionary history or part of the installed inference wheel.

## Copy number and multiplicity

- Each region starts diploid and independently draws gain-only CNA events.
  All clones within it share one CN state per segment (fraction exactly `1.0`);
  independent regional draws may coincide, including at zero event rate or
  saturated CN. The event rate applies per region, not across the tumor.
- Major CN is at most six: no subclonal CN mixtures, descendant CNA events,
  losses, copy-neutral LOH, or whole-genome duplication.
- Each unequal-CN mutation-region independently draws integer multiplicity
  uniformly from `1..min(major_cn, 4)`, inclusive. Equal-CN dosage is one.
  All carrier clones within that region share the draw, independently of CNA
  timing; regions do not share it. Subclonal mutation clusters remain supported.
- **Only `1/1` is balanced CN**, not higher-copy equal-CN states such as `2/2`.
  The equal-CN sampling rule is distinct from inference's `1..major_cn` support.
- Cluster `0` is nonempty and has CCF exactly `1.0` in every region.

For purity `rho`, mutation CCF `phi`, and sampled multiplicity `m`:

```text
p_alt = rho * phi * m / (rho * (major_cn + minor_cn) + (1 - rho) * 2)
depth ~ Poisson(mean_depth)
alt_count ~ Binomial(depth, p_alt)
ref_count = depth - alt_count
```

Named RNG streams separate topology, clone fractions, mutation counts, segments,
CNA events, multiplicity, purity, depth, and reads; CNA/dosage streams have
independent regional children recorded in the manifest. Seed/configuration
replay is deterministic; changing depth preserves latent truth.

## Run

From the workspace containing the `CliPP2` repository, using conda `ml1`:

```bash
conda run -n ml1 python -m CliPP2.simulation \
  --out-dir simulations --tumor-id example --seed 1 \
  --mutation-count 300 --clone-count 3 --region-count 2
```

Existing tumor directories are never overwritten, including concurrent creation.
Two-state-CN controls are unsupported. Bundle validation uses `max_major_cn=6`;
both this limit and the inference default `10` retain all generated CN states.
The simulator's CN ceiling is six and multiplicity ceiling is four.

## Truth and provenance

- `<tumor_id>.clipp2.txt`: canonical input; every mutation must survive the CN filter.
- `truth.txt`: mutation-to-cluster labels.
- `truth_clone_sample.txt`: clone CCF by sample; sample `0` means `region1`.
- `regionN/truth_cp.txt`: per-mutation CCF in that region.
- `truth_mutation_sample.tsv`: integer `multiplicity`, CCF, mutant-copy mass,
  effective multiplicity, total CN, and expected VAF. Use `multiplicity` as the
  exact-class truth target; effective multiplicity agrees up to round-off.
- `truth_mutation_carriers.tsv`: shared mutation-clone membership; combine with
  regional dosage truth above. Noncarriers have no multiplicity target. This
  replaces the old shared-dosage table without mutation-clone-region expansion.
- `truth_mutation_history.tsv`: shared origin clone and segment; no misleading
  single multiplicity column. Physical-copy indices and mutation timing are absent.
- `truth_cn_sample.tsv`: one CN profile row per segment and numeric sample ID,
  replacing `truth_cn_clone_profile.tsv`. `regionN/truth_cn_states.tsv` records
  that region's single local state per segment.
- `truth_cna_events.tsv`: sample-local gain IDs. Clone 0 means a local clonal
  state, not common ancestral CNA across regions. Events explain CN, not dosage.
- `scenario_manifest.json`: generator version, RNG streams, intended/realized
  conditions, source hash, and input/truth file hashes.

The generator is `tree_regional_clonal_cn_uniform_multiplicity_v8`, schema `8.0`.
These identify the scientific design/truth fields, not the inference version;
existing cohorts are unchanged. Evaluate CNA-only multiplicity on
`(major_cn != 1) | (minor_cn != 1)`: exclude only `1/1`, retaining higher-copy
equal-CN states. Report pooled exact-class macro-F1, eligible counts, and micro-,
weighted-, and per-class F1. This filter does not change dosage sampling.
