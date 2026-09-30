# Candidate encounter pairs

This stage discovers pairwise encounters in the read-only reconstructed trajectory
Zarr. It does not depend on deployment cohorts or candidate clusters. A platform
may therefore appear in zero, one, or many pairs.

## Selection

For every two platforms with at least one common valid grid timestamp, the workflow
computes WGS84 geodesic separation using the configured `selection_method`. The
selected encounter is the first stored timestamp whose separation is less than or
equal to `maximum_distance_m`; it is not the closest point over the trajectory.

When `maximum_seconds_from_each_observed_start` is finite, the encounter must obey
both conditions independently:

```text
encounter_time - observed_start_time_1 <= configured seconds
encounter_time - observed_start_time_2 <= configured seconds
```

The observed starts are the exact first retained QC fixes recorded by
reconstruction, not verified deployment coordinates. Setting the option to `null`
searches the pair's full common valid lifetime and therefore enables chance
encounters. It does not disable the distance threshold or temporal-overlap
requirement.

Distances are evaluated at the existing reconstruction cadence without additional
interpolation. There is currently no minimum encounter duration: one qualifying
grid point is sufficient.

## Output

The command atomically publishes:

- `pairs.zarr`, the authoritative grouped-trajectory product;
- `pair_catalog.csv`, one compact diagnostic row per selected pair.

The Zarr uses the `kinematicParcels` grouped-entity layout with dimensions
`(trajectory, obs)`. Each `trajectory` is one pair, deterministic member order is
by platform identifier, and `obs=0` is the selected encounter. Values then continue
through the last common valid timestamp. Shorter rows are padded with NaN/NaT.

The compatibility fields `lon`, `lat`, `z`, `center_lon`, `center_lat`, `lon_1`,
`lat_1`, `lon_2`, and `lat_2` use the configured selection method. `lon`/`lat` are
aliases of the group center, longitude centers use a circular mean, and latitude
centers use an arithmetic mean. The store also retains every complete input
representation explicitly, for example:

```text
lon_linear_1, lat_linear_1, lon_linear_2, lat_linear_2
center_lon_linear, center_lat_linear
lon_spline_30_1, ..., center_lat_spline_30
```

Root attributes identify `canonical_coordinate_method` and
`available_coordinate_methods`. Changing the selection method can change pair
membership and encounter times, so it requires rebuilding the product.

Trajectory-level diagnostics include both observed starts, overlap bounds,
encounter time, encounter distance, delay from each start, post-encounter duration,
and observation count. Method-specific distances are reported at the canonical
encounter time; they do not independently reselect the pair.

## Command

Copy `configs/arcterx/pairs.yml` to an ignored local configuration, set the input
and output paths, and choose a finite `maximum_distance_m`:

```powershell
drifterlab-pairs configs/arcterx/pairs.local.yml
```

Use `--overwrite` only to atomically replace an existing bundle. Input coordinate
values and valid lifetimes are never modified.
