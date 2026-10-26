# Unified analysis of grouped trajectory products

`drifterlab-analyze` is the analysis entry point for the grouped v2 Zarr built
by the manual-QC application. It selects one trajectory product directly from
`/products/<name>` and publishes trajectory plots, trajectory statistics,
epsilon results, epsilon diagnostics, and FSLE spectra in one numbered output tree. The
source Zarr is opened read-only and is not rewritten.

The ARCTERX working configuration is
`configs/arcterx/v2/02a_microsvp_analysis_qc_5min.local.yml`. Local configurations and
scientific outputs are ignored by Git.

Run it from the repository root:

```powershell
drifterlab-analyze configs/arcterx/v2/02a_microsvp_analysis_qc_5min.local.yml
```

If the configured output already exists, the command stops. Use `--overwrite`
only when intentionally replacing it. Replacement is atomic: the old output is
kept until the complete new output has been built and validated.

## Selecting the trajectory source

The input selects one group by name:

```yaml
input:
  trajectory_zarr: C:/path/to/v2_trajectory_products
  trajectory_product: qc_5min
  dataset_label: ARCTERX MicroSVP
  allow_provisional: true
```

`qc_5min` is the baseline reviewed trajectory. Any later product such as
`qc_60min` or `qc_120min` can be selected without changing analysis code,
provided that the builder writes the same validated v2 group contract. Product
existence, platform alignment, cadence, coordinates, velocities, and metadata
are validated before analysis starts.

`allow_provisional: true` is an explicit opt-in for a partial-QC store. Set it
to `false` for the final run after every drifter has been reviewed and the Zarr
has status `final_complete_qc`.

Arrays and clusters can be narrowed for a pilot run:

```yaml
selection:
  arrays: [3]
  clusters: [array_003__cluster_001]
```

Use `all` for the complete selected product.

## Velocity definition

One kinematic definition is shared by trajectory statistics and epsilon:

```yaml
kinematics:
  method: centered_fixed_span
  total_span_minutes: 30
  maximum_source_gap_minutes: 30
  require_full_stencil_support: true
  speed_mask_m_s: null
```

Three modes are supported:

- `centered_fixed_span` uses positions equally spaced before and after the
  reported timestamp. The configured span is the total interval, so the
  baseline 30-minute velocity uses positions at -15 and +15 minutes.
- `centered_adjacent` uses the immediate product-grid neighbour on either side.
  On `qc_5min`, its effective total span is 10 minutes. It needs no explicit
  `total_span_minutes`.
- `stored_forward` uses the velocity already stored in the selected product.
  It connects one product sample to the next and is assigned to the earlier
  endpoint. It also needs no explicit span.

Changing `trajectory_product` changes the position cadence offered to the
analysis. It does not silently redefine `centered_fixed_span`; an incompatible
span is rejected. `centered_adjacent` and `stored_forward` naturally follow the
selected product cadence.

## Analysis structure

The epsilon configuration is organized by scientific purpose rather than by
development stage:

- the core epsilon settings define populations, activation, windows,
  separation bins, and support;
- `uncertainty` defines synchronized UTC block-bootstrap intervals;
- `q_distribution` defines the signed-q distribution products;
- `figures` defines the main epsilon figures.
- `spatial_maps` optionally locates signed pairwise q at each separation midpoint.

The diagnostics module retains only the interpretable trajectory-window,
velocity-availability, and pair-relative-motion figures.
`time_scale_contributions` and `influence_audit` are not part of the unified v2
workflow. The older standalone commands remain available temporarily for
reproducibility of legacy runs.

## Daily geographic epsilon maps

The three `configs/arcterx/v2/04*_microsvp_spatial_eps_*.local.yml` files select
daily maps of spatially conditioned epsilon. Under `analyses.epsilon.spatial_maps`,
the new mode is selected by a `daily_maps` block:

```yaml
spatial_maps:
  enabled: true
  scatter_maps: true
  couples:
    between: cluster
    max_initial_distance_m: null
  windows: all
  daily_maps:
    window_hours: 24
    separation_edges_m: [250, 500, 1000, 2000]
    cell_size_factor: 1
    enforce_support: false
  color:
    maximum_absolute_m2_s3: 0.000005
    linear_threshold_m2_s3: 0.00000001
  dpi: 160
```

The earliest start and latest end of the selected spatial windows define the
overall period. It is split into consecutive, nonoverlapping windows of
`window_hours`; a shorter final window is retained. A 0-360 h period gives exactly
15 daily figures. The UTC reference remains the first finite position entry in
the array's selected trajectory product, before narrowing the cluster selection.
Each figure shows every selected cluster in that array and prints the exact UTC
interval. Gaps between originally selected windows are included in this overall
daily period. Arrays are processed separately.

With `separation_edges_m: null`, omitted edges, or an empty list, each figure has
one comprehensive panel containing all valid pair separations. With positive,
strictly increasing edges such as `[250, 1000, 4000]`, each daily figure contains
four panels: `(0, 250)`, `[250, 1000)`, `[1000, 4000)`, and `[4000, infinity)` metres.
Zero separation is always excluded. Unlike the legacy scatter mode, daily maps
do not trim observations to epsilon's separation-bin range. The separate
`minimum_reliable_separation_m` still applies. `couples` selects pairs exactly as
described in the scatter section below.

In automatic mode, the physical square cell width is
`cell_size_factor * lower_separation_edge_m`. The first band starts at zero and
instead uses its upper edge, the first positive configured boundary. Thus edges
`[250, 500, 1000, 2000]` with factor one give widths of
`[250, 250, 500, 1000, 2000]` metres. This includes the open upper band, whose
reference is its **lower** edge (2000 m), regardless of larger observed
separations. Sizes stay fixed across days and do not depend on data maxima.
This is a configurable display/averaging convention, not a claim that sampling
resolves flow features at that resolution. A pair spanning 180 km can be assigned
by midpoint to a 2 km cell; that does not establish 2 km physical localization.

Manual mode replaces `cell_size_factor`:

```yaml
daily_maps:
  window_hours: 24
  separation_edges_m: [250, 1000, 4000]
  manual_cell_size_m: [500, 1000, 4000, 8000]
```

A scalar manual size applies to every band; a list must contain
`len(separation_edges_m) + 1` positive sizes, including the outer bands. Without
edges, there is one band and an explicit manual size is required because no
bin-derived reference length exists. Manual sizes are mutually exclusive with
non-null `cell_size_factor`. The previous `minimum_cell_size_m` floor is no
longer part of the sizing rule; a non-null legacy value is rejected.

Pair midpoints assign observations to half-open square cells in the existing
array metric projection. The projected grid origin is always `(0, 0)`; boundaries
do not move between days. For each cell, day, and separation band, the estimator is

```text
epsilon_eff = (-5/4) * sum(delta_u_l ** 3) / sum(separation)
```

This preserves the existing ratio-of-sums estimator rather than averaging
individual q values. All valid scheduled observations enter the estimate;
`resample_stride` has no effect on daily maps. The velocity definition and its
validity checks are shared with the epsilon analysis.

`enforce_support` controls whether the existing `analyses.epsilon.support`
requirements mask individual occupied cells. It defaults to `true`, while all
three `04` working configs explicitly set it to `false` for exploratory maps.
With screening enabled, cells below any requirement stay blank and are saved
with `epsilon_eff_m2_s3: NaN`. With screening disabled, every occupied cell with
a finite ratio-of-sums estimate is displayed. `support_ok`, `support_status`,
and all observation/pair/platform/timestamp counts retain the actual screening
results in both modes; `display_ok` records which estimates may be plotted.
Empty cells stay blank in either mode. Zero estimates use the neutral color.
The flag does not bypass position/velocity validity, source support, zero
separation, or nonfinite-value checks. The epsilon analysis's own screening is
unchanged. With one observation, the cell estimate equals that observation's q.
There is no interpolation, smoothing, or contour generation.

All panels and days use one fixed geographic region covering the selected
platform positions over the requested period, with a margin. The extent is
independent of cell size: an open-ended bin's large cells are clipped to the view
and cannot stretch the axes and hide finer panels. Longitude/latitude labels,
coastlines, and the same signed `coolwarm` color limits are used throughout.
The metric square cells are transformed into geographic polygons for display.

These estimates are conditional on the sampled region, day, and separation band;
they are not direct measurements of local viscous dissipation or uniform
area-weighted means. Pair and timestamp counts do not establish independent
realizations, homogeneity, stationarity, or unbiased drifter sampling. To inspect
cluster dominance, each cell's unique cluster count, dominant cluster fraction,
and per-cluster support contributions are saved. Within-cluster observations
contribute one support weight to their cluster; cross-cluster observations split
one support weight equally between their endpoint clusters. The epsilon
calculation counts every pair observation once.

Daily products under `03_epsilon/array_NNN` are:

- `figures/spatial_epsilon/day_NNN.png`: one geographic figure per time window;
- `data/spatial_pair_observations.parquet`: full scheduled observations;
- `data/spatial_epsilon_cells.parquet`: occupied cells, support, sums, and estimates;
- `data/spatial_cluster_contributions.parquet`: cluster support per occupied cell;
- `data/spatial_metadata.json`: resolved bands, sizes, extent, estimator, and windows;
- `tables/spatial_daily_figures.csv`: per-day and per-band counts and color saturation;
- `tables/spatial_pair_selection.csv`: candidate pair selection and initial distances.

Absent rows in the sparse cell table represent empty cells. Daily maps are added
alongside the cluster and array scatter figures described below. `scatter_maps`
defaults to `true`; set it to `false` to request grids only. The `04` configs
explicitly keep both. With daily maps enabled, the shared observation table and
scatter figures also retain observations outside epsilon's separation-bin range.
The scatter display stride never changes the gridded estimates. Omitting
`daily_maps` retains the existing scatter-only mode. All published products are hashed in the unified
manifest. Plotting requires the `drifterlab[plotting]` dependencies and Cartopy
coastline data, as do the existing trajectory maps.

Empty panels may still contain many raw observations: the support requirements
apply per occupied cell. In particular, twelve unique timestamps at a 30-minute
cadence require observations at twelve different sample times in that cell; many
pairs at one time do not satisfy that requirement. The figure footer states all
thresholds. The daily figure table records counts of cells failing each criterion
(these failure counts can overlap), and the maximum unique timestamp count in any
cell. The cell table retains the detailed rejection reasons even when screening
is disabled. The daily figure table distinguishes `supported_cell_count` from
`displayed_cell_count`, and figure footers state whether screening is enforced.

The final open bin includes arbitrarily large observed separations, but its
automatic cell width now uses the lower bin edge, exactly like other positive
lower-edge bins. An observed maximum of 180 km therefore does not enlarge a
`>=2000 m` bin's 2 km cells. Keep the separation scale and cell averaging scale
distinct when interpreting these exploratory fields.

## Spatial pairwise-q scatter maps

Add the following block under `analyses.epsilon`:

```yaml
spatial_maps:
  enabled: true
  couples:
    between: cluster                 # cluster | array
    max_initial_distance_m: null
  windows: all                       # or IDs: [initial_0_6h, initial_0_12h]
  resample_stride: 1                 # every Nth scheduled timestamp
  color:
    maximum_absolute_m2_s3: 0.005
    linear_threshold_m2_s3: 1.0e-8
  dpi: 160
```

Both cluster maps and a combined array map are produced. With `between: cluster`,
only within-cluster pairs appear. With `between: array`, the array map also includes
cross-cluster pairs among the selected platforms. Cluster maps always show their
internal pairs. This selection applies only to the spatial diagnostic; epsilon's
existing `pair_scope` still controls its estimates.

`max_initial_distance_m` filters only cross-cluster pairs in array mode; it is
ignored for within-cluster pairs. The distance is measured in the array projection
at each pair's first simultaneous finite position, before window and velocity
filtering. A selected pair remains selected as it separates. `null` imposes no
initial-distance limit. Pairs without any simultaneous finite position are recorded
as unavailable in the selection table.

Every spatial window uses one UTC origin: the first finite position timestamp in
the array's selected trajectory product, before narrowing the cluster selection.
Window boundaries are `[origin + start_hour, origin + end_hour)`. This applies to
both the cluster and array views, so matching windows show the same UTC interval.
Reusing an epsilon window reuses its hour bounds, **not** its cluster activation
reference. Each figure prints its actual UTC bounds. Custom spatial windows can
also be supplied independently of the epsilon windows:

```yaml
windows:
  - id: hours_6_18
    start_hour: 6
    end_hour: 18
```

The spatial sampling schedule uses epsilon's `sample_cadence_minutes`, anchored
at the array origin. `resample_stride` then keeps every Nth scheduled timestamp,
with all valid pairs at that time. It does not average q or restart the phase after
a gap or at each window. The saved spatial observation table retains every valid
scheduled observation in the union of the requested windows, before this display
stride. The epsilon calculations are unaffected.

Maps use fixed-size, opaque points, equal spatial aspect, and the array's local
metric projection in kilometres. Midpoints are calculated in projected coordinates,
then inverse-transformed for saved longitude and latitude. Each view has fixed
spatial bounds across its windows. Only observations within epsilon's separation
bin range and above its minimum reliable separation are shown; the same velocity
definition, source-gap checks, and speed mask apply.

One `coolwarm` colorbar uses a symmetric logarithmic normalization, bounded by
`±maximum_absolute_m2_s3` with a linear interval around zero controlled by
`linear_threshold_m2_s3`. Both settings remain fixed across all views and windows.
If omitted, they inherit the corresponding q-distribution settings. Values beyond
the color limits remain plotted with saturated colors; extensions and counts make
this visible. Empty windows and singleton clusters receive labelled empty maps.
These are pairwise q observations spanning a separation, not point measurements
of local viscous dissipation. Segment, ellipse, and separation-size overlays are
reserved for a later extension.

Spatial products are published under each `03_epsilon/array_NNN`:

- `figures/spatial_q/array/` and `figures/spatial_q/cluster_*/`: maps per window;
- `data/spatial_pair_observations.parquet`: valid observations before display stride;
- `data/spatial_metadata.json`: resolved settings, UTC origin, and projection;
- `tables/spatial_pair_selection.csv`: initial distances and pair selection reasons;
- `tables/spatial_figures.csv`: UTC windows, observation counts, and saturation counts.

All products are included in the unified manifest. The input trajectory store is
opened read-only.

## Direct FSLE spectra

`analyses.fsle` reads the selected trajectory product directly, independently of
epsilon and velocity settings. It produces separate spectra for every cluster,
pooled within-cluster array pairs, and threshold-selected cross-cluster chance
pairs. All standard figures share axes. A list of prescribed powers and scale
ranges controls amplitude fits and compensated plots. No pair Zarr is created.

See [v2 FSLE methods and passage policies](fsle_v2.md) and the runnable template
[`configs/arcterx/v2/fsle.yml`](../configs/arcterx/v2/fsle.yml). The default
`strict_observed` passage policy and alternative `bridge_missing` remain explicit
while their scientific interpretation is reviewed.

## Endpoint affine deformation

`analyses.affine_deformation.enabled: true` independently adds original-member
endpoint fits and daily principal-axis maps under `06_affine_deformation`.
It uses positions only, supports allowed off-grid endpoint interpolation, and
does not inherit statistics' permanent stop at a temporary observation gap.
Optional prepared daily or hourly SST backgrounds never affect the numerical results.
See [methodology, QC, time references and runnable examples](affine_deformation_methodology.md).

## Output

Enabled modules are published under:

```text
v2_analysis/
|-- 01_trajectory_plots/
|-- 02_trajectory_statistics/
|-- 03_epsilon/
|-- 04_epsilon_diagnostics/
|-- 05_fsle/
|-- 06_affine_deformation/
|-- config_resolved.yml
`-- manifest.json
```

Each module is organized by array. The root manifest records the exact Zarr and
product, product status, selections, velocity definition, source provenance,
module counts, and SHA-256/size record for every published product. Intermediate
legacy layouts and their manifests are not published.

