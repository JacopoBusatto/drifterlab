# Array trajectory postprocessing

The postprocessing workflow reads an existing consolidated `trajectories.zarr`.
It never rebuilds or modifies the reconstruction product. Its first analysis
block creates one complete-trajectory PNG and, by default, one MP4 movie for
every experimental `array_id`.

Run it with a YAML configuration:

```powershell
drifterlab-postprocess configs/arcterx/postprocessing.local.yml
```

Use `--overwrite` only when intentionally replacing products already declared
by the same configuration. Individual files are written through temporary files
and atomically moved into place; the manifest is written last.

## Configuration

The required shared sections select the trajectory store, one reconstructed
coordinate representation, and an output directory:

```yaml
input:
  trajectories: path/to/trajectories.zarr
  dataset_label: MicroSVP
  coordinate_method: spline_30
output:
  directory: path/to/postprocessing
```

Paths are resolved relative to the YAML file. The trajectory block is enabled
independently rather than through an ordered list of analysis names:

```yaml
trajectory_plotting:
  enabled: true
```

With no further trajectory options, the workflow colors platforms by
`cluster_id`, derives one fixed full-window extent per array, writes hourly movie
frames at 15 frames per second, and retains 24 hours of recent tails. The tracked
ARCTERX example documents every available option.

`color_by` may name any Zarr variable with dimensions `(platform,)`, including
`cluster_id`, `member_id`, or `platform_id`. Identifier variables should normally
use `color_mode: categorical`. Because `member_id` restarts inside every candidate
cluster, its colors repeat across clusters. `platform_id` is the unique-platform
choice. Numeric properties may use `color_mode: numeric` and optional `vmin` and
`vmax` bounds.

## Time reference

Verified nominal deployment times can be declared independently for each array:

```yaml
arrays:
  array_001:
    nominal_deployment_time: "2025-01-12T22:30:00Z"
```

These timestamps must include an explicit UTC offset. If the value is null or
the array is omitted, the movie uses the earliest exact first retained-QC fix in
that array. The annotation then says `Elapsed since first retained fix`; it does
not misidentify that observation as a nominal deployment.

## Products and provenance

Outputs are grouped by array and analysis block:

```text
postprocessing/
  postprocessing_manifest.json
  array_001/
    trajectory_plotting/
      trajectories.png
      trajectories.mp4
```

The PNG and movie share their platform colors and map extent. Movie frames show
instantaneous finite positions, recent tails, UTC and elapsed time, active count,
geographic labels, and a WGS84 geodesic scale bar. The manifest records the
effective resolved configuration, input reconstruction metadata, time-reference
source, and every product path.

MP4 output requires FFmpeg to be available on `PATH`. The workflow streams frames
directly to FFmpeg and does not accumulate temporary frame images.
