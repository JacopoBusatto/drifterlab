# Processing workflow

See [running the complete workflow](running_the_workflow.md) for executable
commands, review checkpoints, rerun behavior, and configuration options for both
MicroSVP and SVP.

```text
RAW MAT
  ↓
DROGUE DETECTION + REVIEW
  ↓
NATIVE-POSITION QC + REVIEW
  ↓
PER-PLATFORM QC PARQUETS
  ↓
COMMON-GRID RECONSTRUCTION
  ↓
LEAN CANDIDATE TRAJECTORY ZARR + GAP REPORT
  â†“ (optional, read-only)
TRAJECTORY OVERVIEW / ARRAY EXACT-START / RECONSTRUCTION-CHECK FIGURES
  ↓
CANDIDATE ENCOUNTER PAIRS FROM OVERLAPPING TRAJECTORIES
  ↓
SCIENCE
```

| Stage | Input | Output | Owner | Status |
|---|---|---|---|---|
| Raw | Campaign source MAT files | Immutable normalized signals/positions | `drifterlab.io` | Implemented |
| Drogue QC | Raw TTFF/strain | Automatic Parquet plus explicit review CSV | `drifterlab.qc`, `drifterlab.workflows`, `drifterlab.review` | Implemented |
| Position QC | Raw positions and resolved drogue decision | Schema-4 per-platform QC Parquets plus review CSV | same generic packages | Implemented |
| Reconstruction | Finalized position-QC Parquets only | Shared-grid native, linear, spline tracks, numeric deployment arrays, and candidate initial clusters | `drifterlab.reconstruction` and reconstruction workflow | Implemented |
| Trajectory product | Reconstructed rows | Lean candidate Zarr and per-platform build report | reconstruction workflow | Implemented |
| Plotting | Validated trajectory Zarr | Overview, combined/per-array exact-start, and optional reconstruction-check PNGs | reconstruction command | Implemented |
| Candidate pairs | Reconstructed trajectories | Grouped pair Zarr plus diagnostic catalog | `drifterlab.pairs` and pair workflow | Implemented |
| FSLE spectra | Grouped pair Zarr | First-passage Parquet, spectra CSV, per-array figures, manifest | `drifterlab.fsle` and FSLE workflow | Implemented |
| Pair science | Grouped pairs | Pair analysis products | future packages | Planned |

The position Parquets are the detailed QC audit record. The trajectory Zarr does
not copy their flags or sensor data. Reconstruction trusts `drogue_eligible` and
the recorded analysis cutoff, including its upstream margin; it never reads the
drogue signals or estimates loss again.

The former production path from delivered campaign-QC MAT files to a padded master
Zarr has been removed. Its supplied-QC reader and independent diagnostic scripts
remain for historical investigation only. In particular, the delivered
`drifter_interp` tracks are not inputs to the production reconstruction stage.

Reconstruction assigns every platform one numeric `array_id` from chronological
gaps between exact first retained-QC timestamps. These are observed starts, not
verified deployment coordinates. Assignment never changes trajectory values or
valid time spans. Inside each array it also assigns deterministic candidate
`cluster_id`/`member_id` metadata from exact start-time differences and WGS84
distance between first retained-QC coordinates. First-common-grid distance is
retained as a time-aligned diagnostic. Diameter and member constraints prevent
chaining; configured array overrides can select a complete array as one group.
This second assignment likewise never changes coverage.

Candidate-pair building is independent of array assignments. It
selects the first configured distance-threshold crossing during each pair's common
lifetime, optionally restricted to a window measured from both observed starts,
then retains all reconstructed coordinate representations in a grouped-trajectory
Zarr. See [candidate encounter pairs](candidate_pairs.md).

See [trajectory reconstruction](trajectory_reconstruction.md) for the numerical
policy and [native-position QC](native_position_qc.md) for the upstream schema.
