# drifterlab

`drifterlab` preserves, inventories, and audits observational drifter trajectories.
Its modules follow an explicit raw → QC → reconstruction → trajectory → grouping
workflow. See [the processing workflow](docs/workflow.md) for stage ownership and
a concise distinction between what is implemented now and what is planned next.
The standalone raw-signal drogue workflow is documented in
[drogue-loss detection and review](docs/drogue_loss_review.md).

ARCTERX QC MicroSVP data are the first supported experiment. The current master
product retains native QC positions and both supplied reconstructions, so later
analyses can choose and compare representations without repeating ingestion.

The package installs independently. `kinematicParcels` was consulted for trajectory
layout conventions and is not a dependency. Generic source readers, QC functions,
and trajectory-product code live under `drifterlab.io`, `drifterlab.qc`, and
`drifterlab.trajectories`; campaign assumptions stay under
`drifterlab.experiments`.

## Install and run

Python 3.10 or newer is required. In PowerShell, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
Copy-Item configs/arcterx/microsvp_preprocessing.yml configs/arcterx/microsvp_preprocessing.local.yml
```

Edit `input.directory` in the copied configuration to point at the campaign's
QC MicroSVP directory, then run:

```powershell
.\.venv\Scripts\drifterlab-preprocess-arcterx-microsvp.exe configs/arcterx/microsvp_preprocessing.local.yml
```

On an activated environment, the equivalent portable command is:

```sh
drifterlab-preprocess-arcterx-microsvp configs/arcterx/microsvp_preprocessing.local.yml
# Equivalent module entry point:
python -m drifterlab.cli.preprocess_arcterx_microsvp configs/arcterx/microsvp_preprocessing.local.yml
```

Relative paths resolve against the YAML file's directory, independent of the
working directory. The tracked example therefore uses `../../data/` for outputs.
Local configuration/review files, source MAT files, and generated data are ignored
by Git. Do not commit campaign input files or generated products.

## ARCTERX inputs and scope

The supplied directory structure is:

```text
ARCTERX/Data/
  Raw/
    MicroSVP/
    SVP/
  Quality Controlled/
    MicroSVP/
    SVP/
```

The actual directory is **`Quality Controlled`**, including the space. Only QC
MicroSVP MATLAB files are supported in this release. Raw data have not undergone
the supplied QC. Raw-data QC and QC SVP ingestion are future work.

The supplied native QC already addresses land contamination, SST outliers, and
GPS spikes. Ingestion performs a non-destructive audit: it retains suspicious
values, flags them, and retains the full track after the drogue cutoff.

All source timestamps are interpreted as UTC by the ARCTERX adapter. Generic
MATLAB datenum conversion does not assume a timezone. The current files are
MATLAB Level 5 files readable by SciPy. MATLAB v7.3/HDF5 input is not supported;
it produces an explicit error rather than an inferred schema.

See [the inspected source schema](docs/arcterx_schema.md) for field mappings,
shape exceptions, and scientific provenance.

## Outputs and time axes

The default outputs are:

- `data/microsvp_inventory.parquet`: one row per input, including schema,
  source hash, observation counts, time bounds, sampling statistics, missing
  positions, drogue decisions, reconstruction availability, and audit warnings.
- `data/ARCTERX_MicroSVP_QC.zarr`: the complete master product, with source,
  processing, configuration, and review provenance.

The master has separate native and reconstructed axes:

| Axis | Main variables |
|---|---|
| `trajectory` | `platform_code`, `source_id`, `instrument_type`, source and effective drogue metadata, observation counts, warnings, source hashes |
| `trajectory × obs_qc` | `time_qc`, `lon_qc`, `lat_qc`, `sst_qc`, `slp_qc`, `battery_qc`, `drogue_strain_qc`, `speed_qc_source` |
| `trajectory × obs_interp` | `time_interp`, `lon_interp_30m`, `lat_interp_30m`, `lon_interp_60m`, `lat_interp_60m`, `sst_interp`, both supplied reconstructed speeds |

`obs_qc` and `obs_interp` are row indices, not shared clock times. All inspected
files supply one common reconstructed time vector, so both reconstructions share
`obs_interp`. The generic record/writer model also supports independent axes,
such as `obs_interp_30m` and `obs_interp_60m`, when a future adapter supplies
different time vectors. Ingestion never constructs a union time grid or resamples.

Each source axis is sorted stably and independently. Duplicate timestamps and
invalid times remain present. The master includes `time_matlab_*`,
`source_obs_index_*` (zero-based original indices), `observation_present_*`, and
`n_obs_*`. Padding is `NaT` for time, `NaN` for float data, false for masks, and
`-1` for source indices. Use observation counts or presence masks to distinguish
padding from source rows whose observations are missing.

All native observation variables are retained, including entirely missing SLP.
Numeric `-999` values become `NaN` or `NaT`; meaningful zero and false values stay
intact. Supplied numeric observations use float64 without lossy quantization.
Fractional MATLAB time precision is preserved; times are not snapped to a
five-minute or integer-second grid.

Six supplied battery vectors have no matching timestamp vector. Their root
`battery_qc` rows are missing, and the complete normalized original vectors live
in the same store at `source_unaligned/battery`, in original source order. That
group provides `platform_code`, root `trajectory` indices, `source_obs`, and
`n_source_values`; it intentionally has no inferred clock time.

Read the product with xarray (Dask is optional):

```python
import xarray as xr

with xr.open_zarr("data/ARCTERX_MicroSVP_QC.zarr", chunks=None) as ds:
    track = ds.isel(trajectory=0)
    native = track[["time_qc", "lon_qc", "lat_qc", "analysis_valid_qc"]]
    print(native)

with xr.open_zarr(
    "data/ARCTERX_MicroSVP_QC.zarr",
    group="source_unaligned/battery",
    chunks=None,
) as battery_source:
    print(battery_source.n_source_values.values)
```

## Representations and validity

Native QC positions are the supplied QC fixes. The 30-minute and 60-minute
products come from subsampling, fitting splines, averaging different start
phases, and evaluating back at nominal five-minute resolution. They are not
simply observations saved at different output frequencies. Neither product is
selected as scientifically preferable; interpolation artifacts can affect
velocity increments and structure functions. The single supplied reconstructed
SST remains `sst_interp` rather than being assigned to either position method.

For each representation (`qc`, `interp_30m`, `interp_60m`):

- `position_valid_*` requires finite longitude/latitude within `[-180,180]` and
  `[-90,90]`. Invalid values remain stored; longitudes are not wrapped.
- `drogue_valid_qc` and `drogue_valid_interp` apply the effective drogue decision
  to the corresponding time axis. Unknown decisions and invalid times are false.
- `analysis_valid_*` combines position validity, drogue validity, and valid time.

The source `drogue` variable is **non-dimensional strain**, stored as
`drogue_strain_qc`. It never determines drogue status during ingestion.
`z=0` describes surface position; `nominal_drogue_depth_m=2.5` is separate metadata.

Source loss estimates are approximate, with uncertainty described in the campaign
guide as roughly 1-2 days. Source `drogue_off_time` and `drogue_off_matlab` remain
separate from the effective reviewed decision. The default analysis cutoff is
`effective_drogue_off_time - 24 hours`; only times **strictly before** that cutoff
are drogue-valid. No post-cutoff data are deleted.

For native data, audit flags identify missing/invalid positions, invalid and
duplicate timestamps, irregular sampling, large gaps, and residual speeds above
3 m/s. Interval tolerance is one second around 300 seconds; a large gap exceeds
1,800 seconds plus that tolerance. Speed is calculated using haversine distance
between adjacent chronological valid native rows, assigned to the later row.
Missing positions and nonpositive time intervals are not bridged.

`audit_speed_qc` and `residual_jump_flag_qc` are distinct from the supplied
`speed_qc_source`. **Residual jump flags do not change validity masks.** A user
requiring their exclusion must make that explicit in downstream analysis.
Reconstructed coordinate range flags and masks remain representation-specific.

## Standalone drogue-loss detection and review

The generic drogue workflow reads source files through a configured adapter.
The current `microsvp_mat` reader maps `ObsTimestamp`, `GpsTTFF`, and optional
`Drogue` strain into the common raw-signal schema. A raw per-value-bin method detects
the last persistent cessation of TTFF activity. An independent robust
two-regime L1 detector fits a single downward change to non-overlapping strain
block medians. A small decision layer accepts agreeing clear dates, permits one
clear component when the other has no change or insufficient data, and leaves
conflicts or unresolved evidence undated for review. The physical loss date and
the separately margined analysis cutoff remain distinct.
It never reads or displays the supplied `drogue_off` date and does not modify the
trajectory Zarr. See [the commands, outputs, controls, and initial configurable
thresholds](docs/drogue_loss_review.md).

Use one generic command and choose exactly one mode:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --manual
```

Automatic mode has no GUI, semiautomatic mode reviews only problematic cases,
and manual mode considers every platform. All modes share the configured
automatic Parquet and review CSV. Manual and semiautomatic runs reuse existing
automatic results and resume at the first unfinished review. Add `--overwrite`
to recompute the automatic results without deleting human decisions. Review
decisions save immediately. A different source schema requires only another
reader returning the generic drogue signals; the science and reviewer remain
unchanged.

## Existing trajectory-ingestion drogue decisions

The behavior below belongs to the existing supplied-QC ingestion product. It is
separate from the standalone raw-signal detector above and is retained for
backward compatibility with that product.

Without a review entry, a source loss estimate is used if present. A missing or
unrepresentable source loss date yields `drogue_decision="unknown"`, with false
drogue/analysis masks. No loss date is inferred.

Copy the header-only `configs/arcterx/drogue_reviews.csv` to a local review file,
then point `processing.drogue_review_table` at it. For example, these illustrative
rows show the format (replace IDs with actual input platforms):

```csv
platform_code,decision,drogue_off_time_utc,note
300000000000001,retained,,Reviewed as retained for the supplied track
300000000000002,lost,2025-02-10T12:00:00Z,Reviewed manual loss estimate
```

Treat platform IDs as text when editing a spreadsheet. `retained` applies to all
valid timestamps and requires an empty loss date; `lost` requires a valid explicit
UTC timestamp (`Z` or `+00:00`). Every ID must exist in the current input set.
Duplicate IDs, conflicting fields, malformed dates, and other decisions are errors.

Reviews override effective decisions even if the source already has a loss date.
The source fields are preserved alongside `effective_drogue_off_time`,
`analysis_drogue_end_time`, `drogue_decision_provenance`, and `drogue_review_note`.
The output records the review file hash and effective configuration. To apply
new reviews, rerun ingestion with new output paths; existing outputs are never
overwritten.

## Failure handling and reproducibility

The inventory pass examines every matched file. Unsupported structures, non-battery
shape mismatches, and duplicate platform IDs are recorded as errors and prevent
master publication. Compatible extra scalar/observation fields are retained with
`source_drifter_*` or `source_drifter_interp_*` names and warnings.

After a successful inventory, the writer streams records into a temporary store,
checks that source hashes are unchanged, consolidates metadata, and verifies
readback before publishing. A failed write removes only its own temporary store;
the inventory remains for diagnosis. Rerunning after a failure requires fresh
output paths because the inventory already exists.

The format is Zarr 2 with default chunks of one trajectory by 4,096 observations.
The programmatic interfaces are:

```python
from drifterlab.experiments.arcterx import read_microsvp, preprocess

record = read_microsvp("path/to/ARCTERYX_microsvp_ID.mat")
# record contains normalized, independently sorted source data; no analysis policy.
summary = preprocess("path/to/config.yml")
print(summary.as_dict())
```

Run the synthetic test suite with `python -m pytest`. Tests create their own tiny
MATLAB fixtures and require no campaign data. The initial local dataset validation
is recorded in [the schema document](docs/arcterx_schema.md).

## Optional diagnostics

Downstream broad timing populations use the existing `first_valid_position_time_qc`
field in the master/inventory. All 150 sustained-start estimates matched it, so
sustained-start inference is not part of production preprocessing. Position,
drogue and analysis validity, gap information, and residual-jump flags remain
the reusable QC quantities described above.

One-off timing and sustained-start scripts live in
[`scripts/diagnostics/`](scripts/diagnostics/README.md), with existing outputs in
the Git-ignored `data/diagnostics/`. They remain runnable for troubleshooting and
are outside the installed package and normal preprocessing CLI. See the linked
instructions for optional commands and plotting dependencies.

That directory also contains the raw-only drogue-signal diagnostic for inspecting
population TTFF, strain, and hull-temperature behavior. Its per-drifter figures
overlay the TTFF, strain, and final automatic change dates; it uses no reference
loss date.

## Manual position review

Review residual native-QC position excursions interactively, preserving the
supplied master and coordinates:

```powershell
python -m drifterlab.cli.review_arcterx_positions suggest
python -m drifterlab.cli.review_arcterx_positions review
# After saving and closing the reviewer:
python -m drifterlab.cli.review_arcterx_positions apply
```

The reviewer uses the optional `diagnostics` extra. Decisions are saved separately;
`apply` writes reviewed QC Parquet and recalculates the next queue using surviving
valid points. `suggest` generates separate bidirectional recovery proposals for
blocks of up to three points. In the reviewer, **A** explicitly accepts eligible
high-confidence suggestions for the current region; individual decisions remain
authoritative. See [controls, output files and the review loop](docs/manual_position_review.md).

## Future workflow

```text
ARCTERX master Zarr
    -> infer and review deployment clusters
    -> freeze deployment_groups.parquet
    -> choose qc / interp_30m / interp_60m
    -> synchronize group members
    -> groups and physical pair products
    -> relative dispersion / FSLE / structure functions
```

There is no known supplied deployment-cluster table. Platform IDs, full time axes,
early positions, all representations, and masks are retained for later inference
and review. Physical groups may use all N choose 2 pairs. A future centroid is a
virtual diagnostic; no central physical drifter is fabricated.

The current/planned boundary and module ownership for every stage are summarized
in [the processing workflow](docs/workflow.md). The next drogue task is a separate
downstream validation of finalized reviewed estimates against the supplied
ARCTERX dates; those dates are not inputs to detection or review.

Add future campaigns as new adapters/configurations. Keep their schema and time
conventions in those adapters, and extend generic utilities only where the
functionality is shared.
