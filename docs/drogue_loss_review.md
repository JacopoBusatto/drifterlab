# Generic drogue-loss detection and review

The drogue workflow converts a source file into generic raw signals, runs the
fixed TTFF and strain detectors, combines their evidence, and optionally records
a human decision. Detection, review, and downstream resolution do not depend on
a campaign or instrument:

```text
source-format reader
    -> RawDrogueSignals
    -> TTFF and strain detectors
    -> automatic decision
    -> optional human review
    -> downstream resolver
```

No supplied or reference `drogue_off` date enters this workflow.

## Reader and configuration

`microsvp_mat` is the currently validated reader. It maps one MATLAB
`dataset.drifter_<ID>` structure containing `PlatformId`, `ObsTimestamp`,
`GpsTTFF`, and optional `Drogue` into:

```text
platform_code
time
ttff
strain (optional)
source_path
source_sha256
```

Another dataset with the same source layout can reuse this reader unchanged. A
different layout needs only another small adapter returning the same schema;
the detector, pipeline, reviewer, and resolver remain unchanged.

The workflow YAML has this shape:

```yaml
input:
  reader: microsvp_mat
  directory: ../../data/source/ARCTERX/Data/Raw/MicroSVP
  pattern: "*.mat"

output:
  automatic: ../../data/qc/drogue/auto_drogue_loss.parquet
  review: ../../data/review/drogue/drogue_review.csv

processing:
  missing_value: -999
  analysis_cutoff_margin_hours: 24

detection:
  ttff:
    binning: {...}
    comparison: {...}
  strain: {...}
  combination:
    agreement_tolerance_hours: 48
```

`input.reader`, rather than a campaign name, selects source parsing.

## Commands and shared products

Choose exactly one mode:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --manual
```

All modes use the two configured products. There are no mode-specific files.

- `--automatic` creates the automatic Parquet without a GUI and requires
  `--overwrite` to replace an existing table.
- `--semiautomatic` reuses the automatic table when present and reviews only
  unresolved, conflicting, or stale cases.
- `--manual` reuses the automatic table and reviews every undecided or stale
  platform.
- Adding `--overwrite` to either review mode recomputes the automatic table but
  never deletes human review rows. Changed snapshots return to review as stale.

## Fixed scientific methods

TTFF retains raw integer event counts in fixed time/value bins. Each value bin
finds its last adequately covered persistent cessation, rejecting later
reactivation. Agreement across configured value bins supplies the TTFF date.
Counts are never normalized across bins.

Strain uses non-overlapping UTC-aligned block medians and a robust two-regime L1
fit. Candidate splits must meet the configured duration, downward-drop,
relative-drop, fit-improvement, and ambiguity rules. Missing blocks are not
filled.

The combination is fixed:

- clear TTFF and strain within tolerance: `clear_agreement`, using the earlier
  physical date without averaging;
- two clear dates outside tolerance: `signal_conflict`, without a date;
- one independently clear detector and non-contradictory evidence from the
  other: `clear_ttff_only` or `clear_strain_only`;
- otherwise: `unresolved`.

The physical date and analysis cutoff remain distinct:

```text
analysis_cutoff_time = physical_drogue_loss_time - margin
```

## Reviewer and persistence

The reviewer contains two panels: positive raw TTFF on a logarithmic axis, and
raw strain with block medians and fitted levels. Both show component, automatic,
selected physical-loss, and cutoff dates.

`Accept auto`, `Set manual`, `Not lost`, and `Uncertain` save atomically and
advance. Date nudges are drafts until `Set manual`. `Next` skips without making
a decision. After reaching the end, the reviewer returns to skipped or stale
platforms and closes only when none remain. Existing-decision margin changes
save immediately.

The review CSV stores one latest decision per platform using the single current
13-column schema. Automatic snapshots identify stale decisions, raw hashes
prevent decisions from silently following changed source data, and concurrent
file changes prevent unsafe saves.

## Downstream resolver

`drifterlab.qc.resolve_drogue_decision(automatic_row, review_row)` exposes the
final status, physical loss time, cutoff margin, cutoff time, and decision
source. A human row takes precedence. Without one, a dated automatic result is
used; conflicts and unresolved evidence remain uncertain. Position QC needs no
knowledge of readers, TTFF, strain, the GUI, or campaign identity.
