# Supplied automatic-QC comparison

The supplied-QC importer converts the campaign MATLAB structures `drifter` and
`drifter_interp` into the same grouped trajectory-product contract consumed by
`drifterlab-analyze`. It does not change the manual-QC builder or unified v2
analysis code.

## Primary self-standing build

Run from the repository root:

```powershell
drifterlab-import-supplied-qc configs/arcterx/automatic_qc/00_microsvp_import.local.yml
```

`reference_alignment.mode: automatic_native` derives each activation start from
the first valid native automatic-QC position. Arrays and clusters are then
derived from those starts and positions using the configured production rules.
The resulting `platform_audit.csv` records source warnings, starts, coverage,
assignments, timestamp snapping, and invalid-coordinate counts.

The supplied 30- and 60-minute spline representations are never refitted. Their
nominal five-minute timestamps are snapped only within the configured tolerance.
If a platform has a real phase offset, `linear_resample` evaluates the already
supplied trajectory on the common grid without bridging invalid source samples.
Invalid coordinate pairs therefore remain gaps rather than being interpolated
across.

The importer publishes these representations as independent product groups so
the unchanged v2 analysis can select one at a time:

- `qc_5min`, `qc_15min`, and `qc_30min` from native automatic QC;
- `supplied_spline30_5min`, `supplied_spline30_15min`, and
  `supplied_spline30_30min`;
- `supplied_spline60_5min`, `supplied_spline60_15min`, and
  `supplied_spline60_30min`.

When an existing store predates this nine-product matrix, rebuild both stores
before running the analyses:

```powershell
drifterlab-import-supplied-qc configs/arcterx/automatic_qc/00_microsvp_import.local.yml --overwrite
drifterlab-import-supplied-qc configs/arcterx/automatic_qc/00b_microsvp_import_manual_reference.local.yml --overwrite
```

## Analysis

The analysis matrix contains all nine products. Products at 5, 15, and 30
minutes use centered velocity spans of 10, 30, and 60 minutes respectively.
Run the three cadence-matched manual analyses first:

```powershell
drifterlab-analyze configs/arcterx/v2/02a_microsvp_analysis_qc_5min.local.yml
drifterlab-analyze configs/arcterx/v2/02b_microsvp_analysis_qc_15min.local.yml
drifterlab-analyze configs/arcterx/v2/02c_microsvp_analysis_qc_30min.local.yml
```

Then run the nine standalone automatic-reference configurations (`02_ar_*`):

```powershell
Get-ChildItem configs/arcterx/automatic_qc/02_ar_*_analysis.local.yml |
  Sort-Object Name |
  ForEach-Object { drifterlab-analyze $_.FullName }
```

Each product is analyzed on its own finite support. Missing automatic positions,
different pair availability, and reduced scale-bin support are outcomes of the
automatic pipeline; no common-support mask is applied.

## Structural and scientific comparison

After the manual and automatic analyses exist, run all nine standalone
comparisons (`03_ar_*`):

```powershell
Get-ChildItem configs/arcterx/automatic_qc/03_ar_*_comparison.local.yml |
  Sort-Object Name |
  ForEach-Object { drifterlab-compare-qc $_.FullName }
```

The comparison publishes `platform_comparison.csv`, `summary.json`, and, when
analysis roots are configured, a long-form `epsilon_by_scale_long.csv`. It
reports activation, array, cluster, coverage, and source-gap differences while
preserving each product's native support.

## Optional manual-reference sensitivity

If the native audit shows materially different starts or assignments, build a
second automatic store:

```powershell
drifterlab-import-supplied-qc configs/arcterx/automatic_qc/00b_microsvp_import_manual_reference.local.yml
```

This second store copies the reviewed starts, assignment metadata, time grids,
and outer review intervals from the manual Zarr. It still uses the automatic
positions and their own missing-data masks. Keep both stores: the native build
assesses the complete automatic pipeline, while the reference-aligned build
isolates the supplied position-QC and trajectory representations.

Run all nine manual-reference analyses (`02_mr_*`) and all nine
manual-reference comparisons (`03_mr_*`) with:

```powershell
Get-ChildItem configs/arcterx/automatic_qc/02_mr_*_analysis.local.yml |
  Sort-Object Name |
  ForEach-Object { drifterlab-analyze $_.FullName }

Get-ChildItem configs/arcterx/automatic_qc/03_mr_*_comparison.local.yml |
  Sort-Object Name |
  ForEach-Object { drifterlab-compare-qc $_.FullName }
```

The copied deployment start is a reference boundary, not an instruction to
invent observations. Analysis activation may therefore still be later when the
automatic product lacks enough finite members at the manual activation time;
that remaining difference is reported as an automatic-QC support outcome.

All workflow commands refuse to replace existing output by default. Use
`--overwrite` only when an atomic replacement is intended.

