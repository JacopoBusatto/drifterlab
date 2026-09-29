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

Run drogue detection/review first, followed by native-position QC and then
reconstruction:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic

drifterlab-position-qc configs/arcterx/position_qc.local.yml --automatic
# Or finish the conservative queue with --semiautomatic / --manual.

drifterlab-reconstruct-trajectories configs/arcterx/trajectory_reconstruction.local.yml
# Add --overwrite only when intentionally replacing a stale reconstruction.
```

Copy the tracked configurations before changing local data paths. Relative paths
resolve beside each YAML file. Campaign data, local configurations, Parquets, and
Zarr products are ignored by Git.

The raw-MAT drogue and position readers remain the upstream source adapters. See
[drogue-loss detection](docs/drogue_loss_review.md) and
[native-position QC](docs/native_position_qc.md) for those stages.

## Position-QC input contract

Position QC publishes one schema-`3.0` `<platform_code>.parquet` per source
trajectory. Each row is an immutable source observation and now includes
`source_lon` and `source_lat` alongside the detailed QC evidence, drogue cutoff,
review snapshots, and final decision. Review decisions remain in the separate
atomic `position_review.csv` file. A normal position-QC rerun rebuilds schema-2
files and reapplies those decisions.

Reconstruction accepts only complete schema-3 files with consistent identities,
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
- `longitude_linear`, `latitude_linear`;
- `longitude_spline_15/30/60`, `latitude_spline_15/30/60`;
- `source_gap_minutes`.

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
    print(ds[["longitude_linear", "longitude_spline_30"]])
```

The product metadata deliberately records
`product_status="candidate_pending_gap_review"`. Inspect `build_report.csv` before
promoting a product to a frozen scientific input. Full method, schema, validation,
and fallback details are in [trajectory reconstruction](docs/trajectory_reconstruction.md).

The same YAML optionally enables batch plotting. A normal rerun validates and
reuses an unchanged Zarr, so plotting selections can be changed without rebuilding
trajectories. It writes an overview, an exact-start map, and optional per-platform
linear-versus-spline checks to the configured plotting directory.

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
python -m pytest tests/test_position_workflow.py tests/test_native_position_qc_v2.py tests/test_reconstruction.py tests/test_reconstruction_workflow.py
```

Run the complete synthetic suite with `python -m pytest`.
