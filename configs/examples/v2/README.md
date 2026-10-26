# Commented v2 configuration examples

These self-contained YAML files follow the workflow in
`configs/arcterx/v2/microSVP/array_03/`. Every active option has a comment;
comments also show supported alternatives and settings that must change together.
The examples use portable paths and generic selections rather than campaign IDs
or personal absolute paths.

| Order | Example | Purpose | Command |
| --- | --- | --- | --- |
| 00 | [00_drogue_detection.yml](00_drogue_detection.yml) | TTFF/strain detection and persistent drogue review | `drifterlab-drogue` |
| 01 | [01_manual_qc.yml](01_manual_qc.yml) | Manual position/SST review, audit Parquets, grouped v2 Zarr | `drifterlab-manual-position-qc` |
| 02 | [02_trajectories.yml](02_trajectories.yml) | Trajectory maps, optional movies, cluster statistics | `drifterlab-analyze` |
| 03 | [03_epsilon.yml](03_epsilon.yml) | Epsilon, bootstrap intervals, signed-q distributions, diagnostics | `drifterlab-analyze` |
| 04 | [04_spatial_epsilon.yml](04_spatial_epsilon.yml) | Daily geographic epsilon and signed-q scatter maps | `drifterlab-analyze` |
| 05 | [05_fsle.yml](05_fsle.yml) | Direct FSLE, chance pairs, prescribed-power fits and compensated spectra | `drifterlab-analyze` |
| 06 | [06_affine_deformation.yml](06_affine_deformation.yml) | Original-cluster endpoint affine fits and daily principal-axis maps, without SST | `drifterlab-analyze` |
| 06 | [06_affine_deformation_l3s.yml](06_affine_deformation_l3s.yml) | Same estimator with a prepared daily or hourly SST file | `drifterlab-analyze` |

## Prepare and run

Install the project with the plotting and manual-QC extras (see the repository
[README](../../../README.md)). Copy the examples to `*.local.yml` siblings before
editing them; local configurations and `data/` outputs are ignored by Git.
Keep the copies in this directory to preserve relative path resolution.

```powershell
Get-ChildItem configs/examples/v2/*.yml | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $_.DirectoryName ($_.BaseName + '.local.yml'))
}
```

All relative paths resolve **beside the YAML**, regardless of the shell's current
working directory. Here, `../../../data` points to the repository's `data/` folder.
Set `input.directory` in 00 and 01 to your raw MAT directory, and set
`final_zarr.grid.anchor_time` in 01 to a UTC grid origin at or before your first
accepted position. Output paths already connect the examples. If you change
the Zarr path in 01, update `input.trajectory_zarr` in 02-06 together.

Run these commands from the repository root after editing your local copies:

```powershell
drifterlab-drogue configs/examples/v2/00_drogue_detection.local.yml --automatic
drifterlab-drogue configs/examples/v2/00_drogue_detection.local.yml --semiautomatic
drifterlab-manual-position-qc configs/examples/v2/01_manual_qc.local.yml
# Review/finalize all selected platforms; the GUI then builds the grouped Zarr.
drifterlab-analyze configs/examples/v2/02_trajectories.local.yml
drifterlab-analyze configs/examples/v2/03_epsilon.local.yml
drifterlab-analyze configs/examples/v2/04_spatial_epsilon.local.yml
drifterlab-analyze configs/examples/v2/05_fsle.local.yml
drifterlab-analyze configs/examples/v2/06_affine_deformation.local.yml
# Or use 06_affine_deformation_l3s.local.yml after setting the prepared SST file.
```

Steps 02-06 each read the store from 01 independently. Their numbers organize the
examples; running epsilon first is unnecessary for the spatial or FSLE example.
Each analysis has its own output root. Existing analysis output requires an
explicit `--overwrite` when you intend to replace it.

## Choices to keep consistent

- **Schemas:** manual QC uses configuration `schema_version: 2`; the unified
  analyzer uses `schema_version: 1` while reading v2 grouped products. Drogue
  detection has no schema-version option.
- **Final versus provisional:** examples require completed QC with
  `allow_provisional: false`. To analyze the GUI's provisional build, explicitly
  set it to `true` in the relevant analysis copy. Local array/cluster IDs can
  change when the store's reviewed platform population changes.
- **Product cadence and velocity:** analyses select `qc_30min` with a 60-minute
  centered span (+/-30 minutes) and a 121-minute velocity source-gap ceiling,
  following the v2 30-minute examples. For `qc_5min` or `qc_15min`, change the
  product, gap ceiling, output path/label, and epsilon sample cadence together.
  A centered half-span must be an integer multiple of product cadence.
  `centered_adjacent` uses immediate neighbours; `stored_forward` uses the saved
  earlier-endpoint velocity. Both need `total_span_minutes: null` (or omitted),
  and stored-forward mode requires `require_full_stencil_support: true`.
- **Fixed builder products:** 01 documents the exact four products required by
  the current builder. Their order, source, cadence and gap limits are validated.
- **Epsilon populations:** `within_cluster` uses cluster activation;
  `all_array` requires the documented array-anchor mapping,
  `minimum_active_members: all`, and `sample_phase_reference: array_anchor`.
- **Epsilon review switches:** `uncertainty.enabled`, `q_distribution.enabled`,
  and `figures.enabled` must match. They are all on in 03 and off in 04.
  Bootstrap block lengths must divide each configured window's duration.
- **Spatial maps:** hours use the array's first finite-position UTC timestamp.
  Spatial pair selection is configured separately from core epsilon pair scope.
  `resample_stride` thins scatter display only. Choose manual cell sizes or an
  automatic `cell_size_factor`; these are mutually exclusive. Null separation
  edges need a manual cell size. Support enforcement is enabled in this example.
- **FSLE:** distances and fit ranges use kilometres; its plotted rate uses
  inverse days. Epsilon separation bins use metres. The example uses a 10 km
  chance-pair ceiling; the campaign's 1000 km setting can be substituted for a
  broader search. Prescribed exponents are candidate laws, and fit amplitudes
  do not automatically establish a regime or produce a physical epsilon.
- **Presentation:** movies are off in 02; enable them with an available ffmpeg
  encoder. Trajectory colors must use a field constant per platform. Geographic
  rendering uses the plotting extra and may need Cartopy's map data cache.

The examples are configuration references and require your source data and
reviewed products for an actual scientific run. See
[manual QC](../../../docs/manual_position_qc.md),
[unified analysis](../../../docs/unified_analysis.md), and
[direct v2 FSLE](../../../docs/fsle_v2.md) for methodology and output details.

The position-only affine examples use reviewed `qc_5min` positions, inherit the
product's source-gap limit, and need no kinematics settings. Both use daily UTC
maps and 6-hour endpoint fits. See [affine methodology](../../../docs/affine_deformation_methodology.md)
for the lifecycle, provisional quality flags, prepared-file interface and optional NetCDF4 reader, and
replotting saved results without recomputing fits.
