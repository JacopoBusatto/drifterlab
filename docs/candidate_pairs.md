# Candidate encounter pairs

This stage discovers pairwise encounters in the read-only reconstructed trajectory
Zarr. Deployment-array and candidate-cluster assignments can optionally restrict
pair eligibility and are preserved for both members in the output. A platform may
therefore appear in zero, one, or many pairs.

Two explicit Boolean filters can restrict candidate membership before temporal or
distance calculations:

```yaml
filters:
  same_array: false
  same_cluster: false
```

Both keys are required so the grouping choice is explicit in every build.

`false` means that the corresponding relation is not required; it does not require
the members to be different. `same_array: true` requires equal `array_id` values.
`same_cluster: true` requires both equal `array_id` and equal `cluster_id`, so it
already implies the same-array condition. Enabling both is valid but redundant.
Algorithmic singletons cannot form a same-cluster pair with another platform.

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
by platform identifier, and `obs=0` is the pair's first common valid timestamp.
Values continue through the last common valid timestamp, so the complete common
window is retained even when the selected encounter occurs later. The scalar
`encounter_observation` locates the encounter inside that window. Shorter rows are
padded with NaN/NaT.

Each partner retains its exact observed start time and position plus its
`array_id`, `cluster_id`, cluster-local `member_id`, `cluster_size`, and
`cluster_assignment_status`. These are published with `_1` and `_2` suffixes in
both `pairs.zarr` and `pair_catalog.csv`. The IDs describe candidate deployment
grouping; the configured Boolean filters determine whether those IDs restrict
pair selection.

The pair-level Boolean variables `same_array` and `same_cluster` are also retained
in both outputs, even when their corresponding filters are disabled.

The compatibility fields `lon`, `lat`, `z`, `center_lon`, `center_lat`, `lon_1`,
`lat_1`, `lon_2`, and `lat_2` use the configured selection method. `lon`/`lat` are
aliases of the group center, longitude centers use a circular mean, and latitude
centers use an arithmetic mean. The store also retains every complete input
representation explicitly, for example:

```text
lon_linear_1, lat_linear_1, lon_linear_2, lat_linear_2
center_lon_linear, center_lat_linear
lon_native_1, lat_native_1, lon_native_2, lat_native_2
center_lon_native, center_lat_native
lon_spline_30_1, ..., center_lat_spline_30
```

Root attributes identify `canonical_coordinate_method` and
`available_coordinate_methods`. Changing the selection method can change pair
membership and encounter times, so it requires rebuilding the product.
The `native` method is the source-valid pre-point-QC trajectory resampled on the
same per-platform grid lifespan as the accepted-QC methods.

Trajectory-level diagnostics include both observed starts and coordinates,
deployment-group metadata, overlap bounds, encounter time and observation,
encounter distance, delay from each start, full-common-window duration/count, and
post-encounter duration/count. Method-specific distances are reported at the
canonical encounter time; they do not independently reselect the pair.

Product diagnostics distinguish four inventory stages:

- `possible_pair_count`: all unique platform combinations;
- `eligible_pair_count`: combinations remaining after the array/cluster filters;
- `overlapping_pair_count`: eligible combinations with a common valid timestamp;
- `selected_pair_count`: overlapping eligible pairs satisfying the deployment-time
  and encounter-distance criteria.

## Command

Copy `configs/arcterx/pairs.yml` to an ignored local configuration, set the input
and output paths, and choose a finite `maximum_distance_m`:

```powershell
drifterlab-pairs configs/arcterx/pairs.local.yml
```

Use `--overwrite` only to atomically replace an existing bundle. Input coordinate
values and valid lifetimes are never modified.

The completed pair Zarr can be passed directly to `drifterlab-fsle`. That stage
filters the retained `same_array` and `same_cluster` metadata without rebuilding
pair membership; see [overshoot-aware FSLE](fsle.md).
