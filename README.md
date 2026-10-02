# drifterlab

`drifterlab` provides an explicit raw → drogue QC → position QC → trajectory
reconstruction workflow for observational drifters. Production reconstruction is
independent of the campaign-supplied QC MATLAB files and their `drifter_interp`
tracks: it reads only finalized per-platform position-QC Parquets.

Python 3.10 or newer is required. Install the package and test dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,plotting]"
```

## Production commands

For the complete operational sequence, review checkpoints, command-line flags,
and important YAML choices, see [running the complete workflow](docs/running_the_workflow.md).
The implemented within-array metrics and their equations are documented in
[cluster-statistics methodology](docs/cluster_statistics_methodology.md).

Run drogue detection/review first, followed by native-position QC and then
reconstruction:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic

drifterlab-position-qc configs/arcterx/position_qc.local.yml --automatic
# Or finish the conservative queue with --semiautomatic / --manual.

drifterlab-reconstruct-trajectories configs/arcterx/trajectory_reconstruction.local.yml
# Add --overwrite only when intentionally replacing a stale reconstruction.

drifterlab-pairs configs/arcterx/pairs.local.yml
```

Copy the tracked configurations before changing local data paths. Relative paths
resolve beside each YAML file. Campaign data, local configurations, Parquets, and
Zarr products are ignored by Git.

The raw-MAT drogue and position readers remain the upstream source adapters. See
[drogue-loss detection](docs/drogue_loss_review.md) and
[native-position QC](docs/native_position_qc.md) for those stages.

## Position-QC input contract

Position QC publishes one schema-`4.0` `<platform_code>.parquet` per source
trajectory, cropped to the inclusive first-to-last QC-valid time span.
Each saved row is an immutable source observation and includes
`source_lon` and `source_lat` alongside the detailed QC evidence, drogue cutoff,
review snapshots, and final decision. Review decisions remain in the separate
atomic `position_review.csv` file. Rejected observations inside the span remain
available for before/after comparison. A normal position-QC rerun rebuilds older
products and reapplies saved decisions.

Reconstruction accepts only complete schema-4 files with consistent identities,
configuration, algorithm, and resolution policy. It rejects unresolved position
or drogue state, missing accepted coordinates, duplicate accepted timestamps,
changed files, and any eligible position before the configured grid anchor. It
uses only final-valid, drogue-eligible fixes and trusts the cutoff already recorded
by QC; it never estimates drogue loss.

## Candidate trajectory product

The reconstruction command atomically publishes a bundle containing:

- `trajectories.zarr`, the lean common-grid position product;
- `build_report.csv`, the per-platform gap and fallback report.

The Zarr has dimensions `(platform, time)` and only these scientific arrays:

- `platform_id` and the shared UTC `time` coordinate;
- `start_time`, `start_lon`, and `start_lat` for the exact first retained QC fix;
- deterministic candidate `cluster_id`, one-based cluster-local `member_id`,
  `cluster_size`, and `cluster_assignment_status` on `platform`;
- integer `array_id` on `platform`, displayed as “Array 1”, “Array 2”, etc.;
- `longitude_native`, `latitude_native`, linearly resampled from source-valid
  positions before individual point QC but on the accepted-QC grid lifespan;
- `longitude_linear`, `latitude_linear`;
- `longitude_spline_15/30/60`, `latitude_spline_15/30/60`;
- `native_source_gap_minutes` and `source_gap_minutes`.

It does not duplicate native QC rows, sensor variables, QC flags, or velocities.
All methods remain at the configured five-minute output cadence. Longitude is
unwrapped before fitting and wrapped to `[-180, 180)` on output. Positions outside
a platform's first-to-last accepted-fix span are NaN, and no method extrapolates.

Read the product with xarray:

```python
import xarray as xr

with xr.open_zarr(
    "data/reconstruction/arcterx_trajectories_candidate/trajectories.zarr",
    consolidated=True,
    chunks=None,
) as ds:
    print(ds[["longitude_native", "longitude_linear", "longitude_spline_30"]])
```

The product metadata deliberately records
`product_status="candidate_pending_gap_review"`. Inspect `build_report.csv` before
promoting a product to a frozen scientific input. Full method, schema, validation,
and fallback details are in [trajectory reconstruction](docs/trajectory_reconstruction.md).

The same YAML optionally enables batch plotting. A normal rerun validates and
reuses an unchanged Zarr, so plotting selections can be changed without rebuilding
trajectories. It writes an overview, a combined exact-start map, one exact-start
map per deployment array, and optional per-platform native-versus-linear-versus-
spline checks. Arrays are assigned from chronological gaps between exact first
retained QC timestamps. Initial deployment clusters are then inferred within each
array from observed-start time and WGS84 position proximity. Time-aligned distance
at the first common reconstructed grid time is retained as a diagnostic. Both
assignments leave trajectory values and valid spans unchanged.
Per-array start maps use marker shape for cluster number and color for elapsed
start time.

## Candidate encounter pairs

The pair workflow tests every temporally overlapping platform pair on one selected
coordinate representation. It selects the first WGS84 distance-threshold
crossing, optionally limited to a configured number of seconds from both observed
starts. Setting that time limit to `null` enables chance encounters over the full
common lifetime. Optional same-array and same-cluster filters are applied before
the temporal and distance checks. A trajectory may belong to several pairs.

The authoritative `pairs.zarr` uses the `kinematicParcels` grouped-trajectory
layout and retains the complete common valid window, with the selected encounter
marked inside it. Both partners' observed starts and candidate array/cluster/member
metadata are retained. Its canonical fields use the selection representation,
while all available native, linear, and spline coordinates remain explicitly
available for later analysis. See
[candidate encounter pairs](docs/candidate_pairs.md).

## Campaign-supplied QC diagnostics

The delivered ARCTERX QC MATLAB adapter and the investigative tools under
[`scripts/diagnostics/`](scripts/diagnostics/README.md) remain available for
read-only historical comparisons. They are not production inputs and no installed
MAT-to-Zarr command remains. Existing legacy master stores can still be inspected;
tests create a synthetic legacy store without restoring the removed producer.

The inspected delivered-MAT layout is documented in
[ARCTERX supplied-QC schema](docs/arcterx_schema.md). This provenance does not imply
that the new spline products numerically reproduce the supplied tracks.

## Development

Run the focused production tests with:

```powershell
python -m pytest tests/test_position_workflow.py tests/test_native_position_qc_v2.py tests/test_reconstruction.py tests/test_reconstruction_workflow.py tests/test_pairs.py tests/test_pair_workflow.py
```

Run the complete synthetic suite with `python -m pytest`.
