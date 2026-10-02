# Array trajectory postprocessing

For a consolidated equation-by-equation description of every implemented
within-cluster metric, output column, and figure, see
[within-array cluster statistics methodology](cluster_statistics_methodology.md).

The postprocessing workflow reads an existing consolidated `trajectories.zarr`.
It never rebuilds or modifies the reconstruction product. Independent analysis
blocks create trajectory maps and exploratory within-cluster statistics for
every experimental `array_id` present in the store. The `arrays` configuration
provides optional metadata; it is not an array-selection filter.

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

The cluster-statistics block is independent of trajectory plotting. It can run
with `trajectory_plotting.enabled: false`, and its time window is never inherited
from trajectory plotting:

```yaml
cluster_statistics:
  enabled: true
  time: {start: null, end: null}
  stop_on_member_loss: false
  percentiles: [0, 25, 50, 75, 100]
  velocity:
    difference_interval_minutes: 30
    histogram_bins: 50
    speed_range_m_s: null       # or [minimum, maximum], in m/s
  plotting:
    dpi: 150
    relative_dispersion_yscale: log  # linear or log
  array_overrides: {}
```

Only `time.start` and `time.end` may be overridden by array in this phase. All
percentiles, velocity settings, membership policy, and metric definitions remain
global within a run. Percentiles may be fractional. Deterministic column suffixes
use `p` as the decimal separator: `0 -> p000`, `25 -> p025`, and
`12.5 -> p012p5`.

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
    cluster_statistics/
      velocity_distributions.png
      absolute_displacement.png
      pair_separation.png
      relative_dispersion.png
      cluster_extent_and_shape.png
      cluster_orientation.png
      cluster_timeseries.csv
      cluster_summary.csv
```

The PNG and movie share their platform colors and map extent. Movie frames show
instantaneous finite positions, recent tails, UTC and elapsed time, active count,
geographic labels, and a WGS84 geodesic scale bar. The manifest records the
effective resolved configuration, input reconstruction metadata, time-reference
source, and every product path.

MP4 output requires FFmpeg to be available on `PATH`. The workflow streams frames
directly to FFmpeg and does not accumulate temporary frame images.

Cluster-statistics figures and CSV files are individually written through a
temporary file and atomically moved into place. The workflow checks every
declared product before starting, honors `--overwrite`, and writes the manifest
last. Raw platform velocity samples are retained only in memory while making
distributions and summaries. They remain reproducible from the immutable
trajectory store, coordinate method, and effective configuration in the
manifest. A future raw-sample product should use optional Parquet rather than
CSV.

## Cluster membership and admitted intervals

Platforms are selected by `array_id` and grouped only by the stored `cluster_id`.
Cluster identifiers are preserved verbatim. `cluster_size` is treated as the
assigned membership size and must be a finite positive integer, constant within
the array/cluster group, and equal to the number of platforms bearing that
assignment. A mismatch is an input-integrity error; postprocessing never repairs
or rewrites assignments.

The outputs distinguish:

- `assigned_cluster_size`: the fixed stored assignment;
- `active_member_count`: assigned platforms with finite selected coordinates at
  a timestamp;
- `valid_pair_count`: simultaneous unordered pairs, equal to
  `N_active * (N_active - 1) / 2`.

With `stop_on_member_loss: false`, every timestamp in the configured window is
admitted and metrics are recalculated from active members. Missing metrics remain
`NaN`, never zero.

With `stop_on_member_loss: true`, analysis begins at the first timestamp when all
assigned members are simultaneously valid. It ends immediately before the first
subsequent loss and never restarts. A cluster that is never complete is marked
unavailable and emits a warning. `cluster_timeseries.csv` nevertheless contains
one row for every configured timestamp. `analysis_included` identifies the
admitted interval; counts remain auditable outside it, while every scientific
metric is `NaN`. The summary records admitted start and end, truncation, and a
termination reason.

## Fixed metric coordinates and velocity

All vector and Euclidean calculations use one WGS84 azimuthal-equidistant
projection per array. Its fixed origin is the wrap-safe spherical mean of one
first-valid selected-coordinate position per platform over the full input
trajectory. Every platform therefore has equal weight, and the origin is
independent of the statistics window. The manifest records the origin, CRS
definition, and derivation method. Longitude wrapping is handled before
projection; no Euclidean calculation is performed directly on degrees.

`difference_interval_minutes` is the total velocity-difference span. For an
interior timestamp, the centered estimate is

```text
[X(t + Δ/2) - X(t - Δ/2)] / Δ.
```

Near an admitted boundary, a forward or backward estimate may use the same total
span `Δ`. Exact timestamps, the reported platform position, and both endpoints
must be finite and inside the admitted interval. There is no one-sided fallback
for a missing interior coordinate or a time gap that does not exactly equal the
requested interval. Centered and boundary estimates therefore have the same span
but different temporal alignment. Internal velocity is platform velocity minus
the simultaneous valid cluster-mean velocity.

`speed_range_m_s` affects histogram display only. CSV statistics always use all
finite admitted speed samples. When the range is null, one set of bins is derived
from all absolute and internal speed samples in the current array so cluster
panels remain comparable.

## Extent, shape, and displacement

At each admitted timestamp, the centroid is the mean active-member position in
the fixed metric frame. Member absolute displacement is measured from each
platform's own first valid admitted position. Centroid displacement is measured
from the first valid admitted centroid. With dynamic membership, a centroid can
move because membership changed as well as because the cluster translated; the
active count must be inspected alongside it.

The population covariance uses normalization by `N`, not `N - 1`. Its ordered
eigenvalues define:

```text
eigenvalue_ratio = lambda_minor / lambda_major
major_scale       = sqrt(lambda_major)
minor_scale       = sqrt(lambda_minor)
aspect_ratio      = sqrt(lambda_minor / lambda_major)
```

The radius of gyration is the root-mean-square member distance from the centroid,
and the implementation verifies
`Rg² = lambda_major + lambda_minor`. It describes overall linear extent; the two
scales and aspect ratio describe elongation. Convex-hull area describes the
occupied two-dimensional footprint. Consequently, a collinear cluster can have
large radius and major scale while its minor scale, aspect ratio, and hull area
are zero or nearly zero. Zero area does not imply zero extent.

The time series also reports `convex_hull_area_ratio = A(t) / A(t₀)`, where
`t₀` is the first admitted timestamp with a finite hull for that cluster. The
reference time and area are recorded in the summary. A zero or unavailable
reference area makes the ratio `NaN` for the complete cluster.

Major-axis orientation is axial, counterclockwise from east in `[0, 180)`.
Orientation is `NaN` for zero-size or isotropic covariance, using a relative
eigenvalue-gap tolerance of `1e-8`. The orientation figure uses points so it
cannot draw a misleading line across the 0/180-degree wrap. Negligible negative
eigenvalues may be clipped at a scale-aware tolerance of 256 times machine
epsilon times the larger of `1 m²` and the covariance scale; materially negative
values fail calculation.
Projected configurations whose minor-to-major singular-value ratio is at most
`1e-8` are treated as numerically collinear for hull area.

Validity depends on simultaneous active membership:

- one member: centroid, velocity, and absolute displacement may be valid;
  covariance geometry, shape, pairs, and hull area are unavailable;
- two members: covariance, its eigenvalues, and pair separation are valid, but
  hull area is unavailable;
- three or more members: covariance geometry is valid and hull area is defined;
  an exactly collinear hull has area zero;
- any unavailable numerical quantity is `NaN`, not zero.

## Pair separation and relative dispersion

Pairs are unique, unordered, and strictly within one stored cluster and array.
For each pair, `t_ij0` is its first simultaneous valid admitted time. In the fixed
metric frame,

```text
R_ij(t) = X_j(t) - X_i(t)
r_ij(t) = |R_ij(t)|
q_ij(t) = |R_ij(t) - R_ij(t_ij0)|²
D²(t)   = mean_ij q_ij(t).
```

Pair separation, mean squared pair separation, and vector relative dispersion
are distinct quantities. In particular, `D²` is not
`[r_ij(t) - r_ij(t_ij0)]²`; that scalar expression loses rotations of the
relative-position vector.

When 0, 25, 50, 75, and 100 percentiles are requested, figures show the median,
the 25–75% inner envelope, the min–max outer envelope, and a separately labelled
mean where relevant. Minima and maxima are sensitive to outliers and erroneous
positions and are descriptive bounds, not uncertainty intervals.

These products support visual, descriptive comparison of clusters belonging to
the same array only. They do not establish statistical equivalence and contain no
hypothesis tests, similarity score, automatic cluster grouping, separation-binned
structure functions, FSLE, or front, eddy, or convergence detection.
