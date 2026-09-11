# CliPP2

CliPP2 estimates mutation cancer-cell fractions (CCFs), clusters SNVs, and
infers mutant-copy multiplicity from single- or multi-region tumor sequencing
data with observed-data pairwise fusion.

## Install

```bash
pip install .
```

## Input

The public input is one tab-delimited file per tumor. See
[`examples/exampleTumor1.tsv`](examples/exampleTumor1.tsv).

CliPP2 excludes a mutation from **all regions** if any region has subclonal
copy number (more than one distinct CN state after identical states are combined)
or major CN greater than six; different clonal CN states between regions are
allowed. For each retained mutation–region pair, fitting marginalizes integer
multiplicity candidates from **1 to major CN** with uniform priors under a
binomial likelihood adjusted for purity, normal/tumor copy number, and CCF.
The reported multiplicity is the highest-posterior candidate conditional on the
final fixed-partition CCF refit, with exact ties choosing the smaller integer—not
a rounded VAF-based estimate. Missing or zero-depth observations are marked
uninformative and their call is left missing, except when the only possible
multiplicity is structurally fixed at one.

For region-invariant allele-specific CN, `--multiplicity-policy shared_broad`
explicitly assumes one multiplicity per mutation across regions. Regional
binomial evidence is combined **before** multiplicity is marginalized; the
same joint likelihood drives initialization, fusion, joint center refits,
partition scoring, and reported posteriors. Use
`--multiplicity-policy shared_balanced_single` for the current tree simulator:
it additionally restricts balanced-CN mutations (`major_cn == minor_cn`) to
one copy. This is a simulator-specific assumption, not a general biological
rule. Equal CN calls alone do not prove shared multiplicity. Both shared
policies reject different allele-specific CN across regions; compatible CN
history tuples require a separately specified input model. The default remains
`independent_broad`, which permits different regional multiplicities.

Shared refits use bounded, data-derived multistarts; neither they nor their
coherent rescale/split proposals imply a global optimum. The graph construction
rule, selection penalties, and raw KKT admission gate are unchanged. Use the
`balanced` profile for this experimental joint-model path: `strict` still
requires globally certified fixed-label refits and therefore fails closed
when the joint multistart refit cannot provide that certificate. Under a shared
policy, positive-depth evidence in another region can inform a missing region's
multiplicity, but does not identify its CCF. Coupling and support policy are
recorded in the objective identity and run provenance; results from different
policies must not be pooled as one estimator.

## Fit

Fit on CUDA:

```bash
clipp2 fit \
  --input-file examples/exampleTumor1.tsv \
  --outdir exampleTumor1_results
```


Use `--device cpu` on a CPU-only machine. Run `clipp2 fit --help` for profile,
solver, resource, selection-score, and partition-tolerance controls.

## Outputs

A fit writes three tables into `--outdir`, prefixed with the tumor id (the input
file stem unless a `##tumor_id` metadata line overrides it):

| File | One row per | Main fields |
| --- | --- | --- |
| `{tumor_id}_mutation_clusters.tsv` | mutation | selected cluster and final fixed-partition CCF per region |
| `{tumor_id}_cluster_centers.tsv` | selected cluster | size, raw-partition diameter, and final CCF per region |
| `{tumor_id}_mutation_region_multiplicity.tsv` | mutation × region | final CCF, copy number, and multiplicity or occupancy-path summary |
