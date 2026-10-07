# ARCTERX signed longitudinal analysis

This directory documents the cluster-relative and optional Poje-comparison
products created by `drifterlab-epsilon`. The workflow reads the common-grid
reconstruction directly; it does not use encounter-selected pairs and never
edits the trajectory product.

The initial implementation is deliberately staged:

- Stage 2 calculates and saves numerical snapshot and initial-evolution products.
- Stage 3 adds figures, synchronized UTC block-bootstrap intervals, saved
  conditional histograms, and leave-one-platform-out influence checks.
- Stage 4 will run configured reconstruction and sensitivity comparisons.

See [methodology](methodology.md) for the complete definitions, formulas,
validity rules, coverage denominators, support flags, and file organization.
For a single explanation connecting the calculation, figures, Poje et al., and
the differences in the ARCTERX implementation, read the
[interpretation and Poje comparison](interpretation_and_poje_comparison.md).
The first linear Array 4 numerical run is recorded in the
[Stage 2 pilot report](stage2_array4_pilot.md).
The [figure guide](figure_guide.md) explains every Stage 3 panel, comparison
figure, uncertainty label, and visual support convention.
The first uncertainty-aware Array 4 result is recorded in the
[Stage 3 pilot report](stage3_array4_pilot.md).

Each array now exposes four inspection tables (`clusters.csv`,
`epsilon_by_scale.csv`, `statistics.csv`, and `coverage.csv`) plus normalized
pair and platform Parquet data. Stage 3 adds detailed histogram, bootstrap
sensitivity, and leave-one-platform-out Parquet data without adding more routine
inspection CSVs. Snapshot and nested-window observations are not duplicated in
the raw pair table.

Run a configured analysis with:

```powershell
drifterlab-epsilon configs/arcterx/07_microSVP_epsilon.local.yml
```

The local Poje-comparison configuration uses one common array anchor, all
simultaneous within-array pairs, a 15-minute sampling cadence, and a 12-day
window:

```powershell
drifterlab-epsilon configs/arcterx/08_microSVP_epsilon_poje.local.yml
```

After that immutable 12-day product exists, reproduce the trajectory-only Poje
figures with the separate derived workflow:

```powershell
drifterlab-poje configs/arcterx/10_microSVP_poje.local.yml
```

Its definitions and outputs are documented in the
[Poje analysis guide](../poje_analysis/README.md).

Render the first provenance-linked trajectory diagnostic from the completed
baseline run with:

```powershell
drifterlab-epsilon-diagnostics configs/arcterx/09_microSVP_epsilon_diagnostics.local.yml
```

This creates a separate diagnostic bundle and does not rerun or modify the
source epsilon product.

The tracked `configs/arcterx/epsilon.yml` is a path-neutral template. Local
configurations and scientific outputs are intentionally ignored by Git. Set
`stage3.enabled: false` for a numerical-only Stage 2 product.
