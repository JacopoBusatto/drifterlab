# Independent manual trajectory QC

This desktop application performs exhaustive manual QC directly on the raw
trajectory files. It does not use anomaly detection, speed thresholds, or the
existing position-QC workflow to make automatic decisions. Configurable speed
warnings are display-only. Position and SST decisions are independent:
rejecting a position never rejects its SST measurement, and rejecting SST never
removes the position.

## Installation and launch

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[manual-qc]"
drifterlab-manual-position-qc configs\arcterx\02_bis_microSVP_manual_position_qc.local.yml
```

### Platform selection

The optional `selection.platforms` field restricts the manual-QC inventory to an
explicit platform cohort while retaining the complete raw directory and drogue
products:

```yaml
selection:
  platforms:
    - "300534061905700"
    - "300534061905760"
```

If `selection` is omitted, or `platforms` is set to `all`, every platform in the
resolved drogue product is included. Platform IDs must be quoted, nonempty,
unique strings. An ID absent from the drogue product is rejected before the GUI
opens.

Selection affects the GUI inventory, finalization readiness, and optional grouped
Zarr input cohort. The grouped builder recomputes array numbers from the selected
deployment starts. Consequently, a cohort taken from campaign Array 4 may become
`array_001` in a subset-only grouped product. Disable `final_zarr` when the
campaign-wide array numbering must remain authoritative.

The reviewer uses the resolved drogue product only to establish the endpoint.
With `drogue.stop_at: physical_loss`, the loss-time observation is included and
later observations are outside review scope. `analysis_cutoff` uses the
margin-adjusted cutoff instead. An uncertain drogue decision blocks finalization.

Set `review.maximum_accepted_span_days: 15` to review an accepted trajectory
spanning at most fifteen days. Raw observations from switch-on remain visible
so laboratory positions can be rejected. The first manually accepted point with
a valid timestamp and coordinates becomes the deployment start. Rejecting that
point moves the deployment start forward and extends the review tail until the
first accepted point at or beyond fifteen days, or until the drogue endpoint or
record end. Restoring a leading point recalculates and may contract the scope.
Set the option to `null` or omit it to review through the complete drogue-bounded
record.

## Interface and decisions

The map contains the whole in-scope trajectory in gray, the accepted trajectory
in blue, and the active chunk in cyan. Rejected positions remain visible as red
crosses, and the cursor is gold. `Fit chunk`, `Fit trajectory`, `+`, `-`, wheel
zoom, and mouse dragging control the map view.

Speed, vector acceleration, SST, `GpsPosQual`, and `GpsNumSat` share one time
axis but have independent Y axes. Kinematics use adjacent observations and
their actual elapsed times, so irregular sampling is supported. A position edit
immediately recalculates the accepted bridge. SST has independent decisions; an
accepted SST attached to a rejected position is shown with a distinct hollow
marker.

`review.warnings.high_speed_m_s` marks accepted adjacent speeds strictly above
the configured value with magenta triangles. The marker is attached to the
later endpoint of the edge, matching the GUI's speed assignment.
`review.warnings.zero_speed_tolerance_m_s: 0` marks exactly zero speeds with
brown squares; a positive tolerance can also expose near-stationary edges. Set
either option to `null` to disable that warning. These derived markers update
after every position edit and never reject, restore, or finalize observations.

`GpsPosQual` and `GpsNumSat` are compact reference panels, not separate QC
targets. Clicking or rectangle-selecting either panel selects the corresponding
position observations, so `R`/`K` reject or restore positions. Rejected
positions remain marked with red crosses in both panels. The current values are
also included in the selected-point readout. `GpsPosQual` is shown under its
source name and is not labelled as HDOP because its vendor meaning has not been
confirmed.

Click or rectangle-select in SST to activate the SST target. Click the map, or
rectangle-select the map, speed, or acceleration, to activate position. The
target label and Reject/Restore buttons always show what will change. Resampled
markers are display-only.

The wheel zooms time; `Shift`+wheel zooms only that plot's Y axis. `Fit visible
Y` independently fits all five time-series panels without changing time.
`Auto-fit Y after time zoom` does that automatically.

| Control | Effect |
| --- | --- |
| Left / Right | Previous/next raw position, or finite SST observation when SST is active; crossing a boundary activates that chunk |
| R / Delete | Reject the cursor or selection in the active target |
| K | Restore rejected values in the active target |
| P | Previous chunk without changing its review state |
| U | First unreviewed or changed-after-review chunk |
| N | Mark this chunk reviewed and move forward |
| Ctrl+Z | Undo the latest position or SST decision on this platform |
| Ctrl+S | Save the cursor; decisions and progress are already autosaved |
| Escape | Clear the multi-point selection |

Chunk symbols are `○` for unreviewed, `✓` for reviewed, and `!` after an edit
affects a reviewed chunk. This ledger is separate from point decisions because a
clean trajectory may have no rejections: chunk records prove that every interval
was inspected. `U` safely returns to the earliest unfinished location.

`Finalize platform` validates complete chunk coverage, position validity and
unique timing, and unique accepted SST timestamps. It atomically publishes that
platform's audit Parquet. A later edit makes it stale until affected chunks are
reviewed and the platform is finalized again.

## Preview resampling

`Resample now` is a GUI preview from the first through last accepted position.
Position and SST use their own accepted raw observations. SST is linear and is
not extrapolated. Edits mark the preview stale; they never trigger a hidden
resample. The preview is independent of the authoritative grouped products,
whose globally aligned anchor is under `final_zarr.grid.anchor_time`.

## Persistence and products

SQLite transactionally stores independent position/SST rejections, reviewed
chunks, undo history, preview revision, finalization revision, and cursor. The
source SHA-256 is a content fingerprint that prevents decisions being applied to
a replaced file; it is not a platform identifier.

Each finalized Parquet retains every source observation, immutable source index,
raw position, SST, `GpsPosQual`, and `GpsNumSat`, review scope, independent
position and SST statuses, decision times, chunk coverage, drogue dates,
configuration, and provenance. It also records the raw first observation,
reviewed deployment start, requested and achieved accepted spans, review end,
and whether that end was selected by the maximum span, record end, physical
drogue loss, or analysis cutoff. The GPS fields are exact source-aligned
reference values; they do not introduce another decision domain.

When `final_zarr.enabled` is true, every current finalized Parquet with matching
source, drogue, state, and file hashes is eligible. Before review is complete,
`Build provisional Zarr` freezes that exact subset and reports how many drifters
remain. Once the last platform is finalized,
`build_when_all_platforms_reviewed: true` automatically starts the same builder
with status `final_complete_qc`. Publication replaces the previous store only
after full readback validation; failure leaves it untouched.

The grouped v2 store contains:

- `/products/raw_5min`, resampled from all valid native positions inside the
  accepted trajectory lifespan;
- `/products/qc_5min`, `/products/qc_15min`, and `/products/qc_30min`, resampled
  from manually accepted positions;
- configured source-gap masking and no endpoint extrapolation;
- forward velocity assigned to the earlier endpoint in every product;
- root platform metadata, input hashes, build status, exact start coordinates,
  and complete array/initial-cluster assignments;
- `build_manifest.json`, `build_report.csv`, and
  `cluster_pair_diagnostics.csv`.

Arrays use adjacent exact accepted-start times. Initial clustering uses the
configured distance reference and retains both observed-start and first-common-
grid diagnostics. Assignments in a partial store are explicitly provisional and
are recomputed in the complete build. Platform IDs, rather than provisional
array or cluster numbers, are the durable identifiers.

The v2 grouped store intentionally does not contain splines or resampled SST.
The finalized per-platform Parquets remain the authoritative audit products for
SST, GPS quality, satellite count, and the independent QC statuses.
