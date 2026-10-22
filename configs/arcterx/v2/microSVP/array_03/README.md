# Campaign Array 3: 30-day manual QC and analysis

`01_microsvp_manual_qc.local.yml` selects the 65 campaign Array 3 platforms
explicitly. It extends each accepted trajectory to 30 days from its first
accepted valid position, bounded by physical drogue loss or record end.

Paths below are relative to
`C:/Users/Jacopo/Documents/DATI/ARCTERX/ARCTERX/Data/manual_QC/MicroSVP/`:

| Product | Path |
| --- | --- |
| Review state and finalized audit Parquets | `01_manual_position/array_03_30D/` |
| Grouped trajectory store | `02_trajectory_products/array_03_30D.zarr` |
| Trajectory figures and statistics | `03_trajectories/array_03_30D/` |
| Epsilon analysis | `04_epsilon/array_03_30D/` |
| FSLE analysis | `05_fsle/array_03_30D/` |
| Spatial epsilon analysis | `06_spatial_eps/array_03_30D/` |

## Saved decisions and review

The separate state is initialized with a SQLite backup of the completed
campaign state at `01_manual_position/manual_position_qc.sqlite`. The snapshot
retains position/SST decisions, undo history, and review records. Although the
snapshot contains the campaign state, the explicit platform selection limits
the GUI, finalization readiness, and grouped build to these 65 platforms.
Do not replace the new state after extended review has begun.

Changing the accepted span marks previous chunk reviews dirty when each
platform is loaded. Saved position/SST rejections remain applied; mark the
chunks reviewed again, inspect the added days, and finalize each platform.
The old campaign state, audit Parquets, and 15-day store remain available.

Launch from the repository root:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.manual_position_qc configs\arcterx\v2\microSVP\array_03\01_microsvp_manual_qc.local.yml
```

The GUI builds the new Zarr automatically after all selected platforms have
current finalizations. Its provisional-build button can publish the reviewed
subset earlier. Analysis configs currently allow provisional input; set
`input.allow_provisional: false` to require complete QC.

## Array numbering and analysis

The grouped builder recomputes array IDs from deployment starts. This cohort
forms **local array 1**, so all analysis configs select `arrays: [1]` and
`clusters: all`. The `array_03` config directory, dataset labels, and
`array_03_30D` output names identify the original campaign cohort. Generated
analysis subfolders use the local `array_001` identity.

Clustering uses the normal inference rules. The campaign's single-cluster
overrides for original Arrays 1 and 2 must not be carried into this subset.

The `02*` configs plot trajectories and statistics; `03*` run epsilon;
`04*` run spatial epsilon; `05*` run FSLE. Existing analysis windows are retained
and epsilon configs add `initial_0_720h` for the complete 30-day period. Daily
spatial maps therefore cover up to 30 daily windows; trajectory plots and FSLE
already use the full available time range.

For example, after building the store:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.analyze configs\arcterx\v2\microSVP\array_03\02b_microsvp_trajectories_qc_15min.local.yml
```

`00_microSVP_drogue_detection.local.yml` remains a shared campaign-wide
preprocessing config. Manual QC continues to use the existing reviewed `myQC`
drogue products; this extension does not require rerunning drogue detection.

The duration in filenames is explicit, not expanded from YAML automatically.
For a future duration, update `maximum_accepted_span_days`, the state/audit
directory, Zarr path, all analysis input/output paths, labels, and time windows
together. Seed a new state from the latest review to retain decisions.
