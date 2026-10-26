# Campaign Array 3: 30-day manual QC and analysis

`01_microsvp_manual_qc.local.yml` selects the 65 campaign Array 3 platforms
explicitly. It extends each accepted trajectory to 30 days from its first
accepted valid position, bounded by physical drogue loss or record end.

Paths below are relative to
`C:/Users/Jacopo/Documents/DATI/ARCTERX/ARCTERX/Data/manual_QC/MicroSVP/`:

| Product | Path |
| --- | --- |
| Review state and finalized audit Parquets | `01_manual_position/array_03_30D/` |
| Grouped trajectory store | `02_trajectory_products/array_03_30D.zarr` |
| Trajectory figures and statistics | `03_trajectories/array_03_30D/` |
| Epsilon analysis | `04_epsilon/array_03_30D/` |
| FSLE analysis | `05_fsle/array_03_30D/` |
| Spatial epsilon analysis | `06_spatial_eps/array_03_30D/` |
| Endpoint affine deformation | `07_affine_deformation/array_03_30D/` |

## Saved decisions and review

The separate state is initialized with a SQLite backup of the completed
campaign state at `01_manual_position/manual_position_qc.sqlite`. The snapshot
retains position/SST decisions, undo history, and review records. Although the
snapshot contains the campaign state, the explicit platform selection limits
the GUI, finalization readiness, and grouped build to these 65 platforms.
Do not replace the new state after extended review has begun.

Changing the accepted span marks previous chunk reviews dirty when each
platform is loaded. Saved position/SST rejections remain applied; mark the
chunks reviewed again, inspect the added days, and finalize each platform.
The old campaign state, audit Parquets, and 15-day store remain available.

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.manual_position_qc configs\arcterx\v2\microSVP\array_03\01_microsvp_manual_qc.local.yml
```

The GUI builds the new Zarr automatically after all selected platforms have
current finalizations. Its provisional-build button can publish the reviewed
subset earlier. Analysis configs currently allow provisional input; set
`input.allow_provisional: false` to require complete QC.

## Array numbering and analysis

The grouped builder recomputes array IDs from deployment starts. This cohort
forms **local array 1**, so all analysis configs select `arrays: [1]` and
`clusters: all`. The `array_03` config directory, dataset labels, and
`array_03_30D` output names identify the original campaign cohort. Generated
analysis subfolders use the local `array_001` identity.

Clustering uses the normal inference rules. The campaign's single-cluster
overrides for original Arrays 1 and 2 must not be carried into this subset.

The `02*` configs plot trajectories and statistics; `03*` run epsilon;
`04*` run spatial epsilon; `05*` run FSLE; `06*` run endpoint affine deformation.
Existing analysis windows are retained
and epsilon configs add `initial_0_720h` for the complete 30-day period. Daily
spatial maps therefore cover up to 30 daily windows; trajectory plots and FSLE
already use the full available time range.

The affine configs are `06a_microsvp_affine_qc_5min.local.yml` (baseline),
`06b_microsvp_affine_qc_15min.local.yml`, and
`06c_microsvp_affine_qc_30min.local.yml`. Each reads the matching reviewed linear
position product, selects local array 1 and all original clusters, and inherits
that product's interpolation-gap limit. They use daily UTC maps, 6-hour endpoint
fits and the full eligible cluster lifetime. No velocity settings are required.
`plotting.fixed_extent: false` fits each daily frame to its initial cluster
centroids; set it to `true` to keep one shared latitude/longitude view across
dates. Both panels use the same bounds, and rate colour limits remain fixed.
Prepared SST defaults to
`C:/Users/Jacopo/Documents/DATI/VIIRS_NOAA/L3S_daily_20250201_20250331.nc`.
Set `analyses.affine_deformation.sst.file` to
`C:/Users/Jacopo/Documents/DATI/VIIRS_NOAA/L3U_hourly_20250201_20250331.nc`
to use the hourly product instead, or set `sst.enabled: false` for drifter-only
maps. Both files share the same core variables and CF time bounds. Daily
composites enter unchanged; hourly bins receive equal hour weights in each
daily map. Quality selection is already performed during preparation, so the
analysis has no NOAA variable mappings or rejected-flag mask. The fixed colour
limits of 24?26.5 ?C may need adjustment over the full period; they do not filter
SST or change the affine fit. Use separate output roots for different inputs,
windows or QC settings.

With `plotting.fixed_extent: true`, set optional `longitude_bounds: [125, 135]`
and `latitude_bounds: [15, 30]` for an exact geographic view. Either can be
`null` for automatic bounds. These limits affect the map and SST subset only.

For MUR L4, first run the standalone [MUR preparation command](../../../../../docs/mur_sst_preparation.md),
then use `06a_microsvp_affine_qc_5min_mur.local.yml` (5-minute positions) or
`06c_microsvp_affine_qc_30min_mur.local.yml` (30-minute positions). Both point to
`C:/Users/Jacopo/Documents/DATI/MUR_L4/MUR_daily_20250201_20250331.nc` and uses
the separate `mur_T24h_Tprime6h` output root. The VIIRS configs remain available
for comparison. The MUR background is an interpolated foundation-SST analysis,
not an observed daily composite.

Run the baseline from the repository root:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.analyze configs\arcterx\v2\microSVP\array_03\06a_microsvp_affine_qc_5min.local.yml
```

Results are published beneath
`07_affine_deformation/array_03_30D/qc_5min/prepared_sst_T24h_Tprime6h/06_affine_deformation/array_001/`.
See [the affine methodology](../../../../../docs/affine_deformation_methodology.md)
for interpretation, lifecycle, QC flags and replotting saved results.

For example, after building the store:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.analyze configs\arcterx\v2\microSVP\array_03\02b_microsvp_trajectories_qc_15min.local.yml
```

`00_microSVP_drogue_detection.local.yml` remains a shared campaign-wide
preprocessing config. Manual QC continues to use the existing reviewed `myQC`
drogue products; this extension does not require rerunning drogue detection.

The duration in filenames is explicit, not expanded from YAML automatically.
For a future duration, update `maximum_accepted_span_days`, the state/audit
directory, Zarr path, all analysis input/output paths, labels, and time windows
together. Seed a new state from the latest review to retain decisions.
