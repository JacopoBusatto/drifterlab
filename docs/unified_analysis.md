# Unified analysis of grouped trajectory products

`drifterlab-analyze` is the analysis entry point for the grouped v2 Zarr built
by the manual-QC application. It selects one trajectory product directly from
`/products/<name>` and publishes trajectory plots, trajectory statistics,
epsilon results, and epsilon diagnostics in one numbered output tree. The
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

## Spatial pairwise-q maps

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

## Output

Enabled modules are published under:

```text
v2_analysis/
|-- 01_trajectory_plots/
|-- 02_trajectory_statistics/
|-- 03_epsilon/
|-- 04_epsilon_diagnostics/
|-- config_resolved.yml
`-- manifest.json
```

Each module is organized by array. The root manifest records the exact Zarr and
product, product status, selections, velocity definition, source provenance,
module counts, and SHA-256/size record for every published product. Intermediate
legacy layouts and their manifests are not published.

