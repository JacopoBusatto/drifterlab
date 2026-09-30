# Trajectory reconstruction and candidate Zarr

## Command and configuration

```powershell
Copy-Item configs/arcterx/trajectory_reconstruction.yml configs/arcterx/trajectory_reconstruction.local.yml
drifterlab-reconstruct-trajectories configs/arcterx/trajectory_reconstruction.local.yml
```

The command reads only finalized schema-4 position-QC Parquets. It never opens a
MAT file, the old master Zarr, or a supplied `drifter_interp` track.

The tracked configuration anchors the phase grid at
`2025-01-12T00:00:00Z`, uses a five-minute step, and requests 15, 30, and
60-minute spline ensembles. Each period must be an integer multiple of the grid
step. Long-gap thresholds are independently configurable and default to their
period. An explicit end must lie on the anchored grid and cannot precede any
eligible fix.

The configured start is both a validation boundary and the phase-zero anchor. Any
eligible position before it aborts the build with all affected platforms and their
earliest timestamps. The published coordinate omits only leading instants that
would be NaN for every platform; both configured and published starts are recorded.
Without an explicit end, the grid ends at the last anchored instant no later than
the latest eligible fix.

The command is idempotent. If the bundle already exists, it validates schema
`1.3`, the reconstruction-only configuration, deployment-array assignments, the
build-report hash, and every QC input hash. A matching bundle is reused; plotting-only YAML changes do not
invalidate it. A mismatch fails with an instruction to use `--overwrite`.
Overwrite builds and validates a complete temporary replacement before swapping
the existing bundle. Plotting runs only after a validated build or reuse.

## Input validation

Each file must contain one platform matching its filename and footer, schema `4.0`,
`source_lon/source_lat`, complete final QC, and a resolved `lost` or `not_lost`
drogue state. The input set must share its QC algorithm, configuration, and
resolution policy. Duplicate accepted timestamps, fewer than two accepted fixes,
invalid accepted coordinates, or a lifespan containing no grid instant are errors.

Accepted fixes satisfy all three conditions:

```text
final_position_status == "valid"
final_position_valid == true
drogue_eligible == true
```

For a lost drogue, accepted times are checked against the single cutoff already in
the Parquet. No loss date or margin is recalculated. Invalid coordinates on rows
already rejected by QC remain in the Parquet audit and do not enter reconstruction.

The native comparison track uses source-valid rows in retained temporal,
drogue-eligible, and deployment-eligible scope, without applying individual final
position decisions. Its source rows are restricted to the inclusive timestamp
span of the first and last accepted QC fixes. Duplicate usable native timestamps
are rejected as ambiguous. Source-invalid and temporally excluded rows never enter
the native track.

Input files are hashed before both passes and again before publication. A changed,
added, or removed Parquet aborts the build and removes only the workflow's exact
temporary bundle.

## Reconstruction policy

Longitude is unwrapped across the dateline. The linear track fills every grid
instant between the first and last accepted fixes, including arbitrarily long
internal gaps. It never extrapolates. `source_gap_minutes` is the time separation
between bracketing accepted fixes and is zero when a grid instant is an exact fix.

For a period `P`, every global phase from `0` through `P/dt - 1` subsamples that
complete linear track. A source-fix gap strictly greater than the configured
threshold divides the spline portions; the linear track is retained across the
gap. Each short-gap portion adds its exact accepted endpoint fixes to every phase,
fits SciPy's natural cubic interpolating spline, evaluates on the five-minute grid,
and equally averages all phase tracks.

Every phase needs at least four unique knots. If any phase lacks support, the whole
portion is linear. The same reported fallback applies if a phase or its average is
non-finite, leaves latitude bounds, or exceeds the coordinate-wise envelope of the
portion's accepted fixes by more than `1e-10` degrees. These portion-wide fallbacks
avoid pointwise method seams and meet the long-gap linear track at the exact
bounding fixes without a position jump. Final longitude is wrapped to
`[-180, 180)`.

No median filter is applied: final position-QC decisions are authoritative.

Before writing the Zarr, platforms are sorted by exact first retained-QC time,
with `platform_id` breaking timestamp ties. Array 1 starts with the earliest
platform; a new array starts only when the adjacent start-time gap is strictly
greater than `arrays.maximum_adjacent_start_gap_hours` (24 hours by default).
Identifiers are positive, contiguous, chronological integers. The Zarr variable
is named `array_id`; figures display it as “Array X”. These temporal deployment
arrays replace the former downstream cohort/cluster workflow and do not use
spatial proximity or later trajectory behavior.

`longitude_native` and `latitude_native` linearly resample the native comparison
fixes on the same shared grid. Their finite mask must equal the QC-derived linear
track's mask for each platform; they cannot extend the grid or an individual
trajectory's lifespan. They are directly comparable grid estimates, not
observations made at every grid timestamp. `native_source_gap_minutes` records the
separation of native fixes bracketing each value, independently of
`source_gap_minutes` for accepted QC fixes.

The ARCTERX Data Guide and Data Processing MicroSVP description motivate the
30/60-minute phase-subsampling and averaging strategy; 15 minutes is this project's
extension. Those documents do not establish boundary conditions, missing-fix
treatment, or fallback rules. The choices above are therefore explicit project
policy, and this product does not claim numerical reproduction of the supplied
campaign tracks.

## Output schema and report

The atomic output bundle contains `trajectories.zarr` and `build_report.csv`. The
Zarr uses `(platform, time)` arrays and stores platform ID, integer `array_id`, shared UTC time,
native, linear, and configured spline longitude/latitude, plus native and
accepted-QC source-gap minutes. Positions are float64 and chunks default to one
platform by 2,016 times. Metadata records the complete effective configuration,
input hashes, units, interpolation/fallback policy, schemas, software version, and
candidate status. Root attributes also record the assignment rule, threshold,
assignment hash, and per-array counts/start ranges.

`start_time`, `start_lon`, and `start_lat` are one-dimensional platform variables.
They preserve the actual first retained QC fix and therefore need not coincide
with the first populated common-grid cell. They are not deployment estimates.

The CSV has one row per platform with its `array_id`, gaps to the preceding and
following chronological starts, eligible/fill bounds, native and accepted-fix
counts, gap quantiles and threshold exceedances, exact/interpolated grid counts, boundary
warnings, and per-period counts for long-gap, insufficient-support, overshoot, and
total linear fallback. Review this report before promoting the candidate product
to a frozen scientific input.

## Optional plotting in the same YAML

Install the optional map dependency once with
`python -m pip install -e ".[plotting]"`. Plotting uses the same command and never
writes to a Zarr store:

```yaml
plotting:
  enabled: true
  output_directory: ../../data/reconstruction/arcterx_trajectory_figures
  dataset_label: MicroSVP
  method: spline_30          # native, linear, spline_15, spline_30, or spline_60
  time: {start: null, end: null}
  platforms: []              # all, or {dataset, platform_id} entries
  check_platforms:
    - {dataset: MicroSVP, platform_id: "300534061905670"}
  map:
    projection: PlateCarree
    central_longitude: 0
    extent: null             # or [west, east, south, north]
    figsize: [12, 8]
    dpi: 150
    land: true
    coastlines: true
    gridlines: true
    label_starts: false
  additional_stores: []
```

`trajectory_overview.png` draws the selected method one platform row at a time,
breaking lines at missing values and longitude wraps. Star markers denote the
exact first retained QC position. `starting_positions.png` maps those exact
coordinates colored by deployment array without implying verified deployment
locations. `starting_positions__array_XX.png` provides one readable map per array,
colored by hours after that array's first start. Optional labels use platform IDs.
Each `check_platforms` entry
adds `reconstruction_check__<dataset>__<platform>.png`, overlaying the native,
linear, and every spline pair present in that store.

Additional read-only stores may use a different time coordinate. Each entry needs
a unique label, direct Zarr path, and method. Method pairs use
`longitude_<method>`/`latitude_<method>`; unsuffixed `longitude`/`latitude` are
also called `native` for compatibility with external stores. Platform selections always use `(dataset, platform_id)`, so
overlapping IDs remain unambiguous. Additional stores must expose the same
`platform_id`, `time`, and exact-start variables; no Parcels trajectory-table
adapter is used.
