# Drogue-loss detection and review

The drogue workflow estimates physical loss times independently from raw
`GpsTTFF` and `Drogue` strain, combines clear evidence, and optionally records
human decisions. It never reads or plots supplied ARCTERX `drogue_off` dates.
Hull temperature is not part of detection or interactive review.

## One command, three modes

Run from the repository root with exactly one mode:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --manual
```

- `--automatic` creates its automatic Parquet and does not open a GUI.
- `--semiautomatic` reuses its existing automatic Parquet when available, then
  reviews unresolved, conflicting, or stale platforms.
- `--manual` reuses its existing automatic Parquet when available and resumes
  at the first undecided or stale platform across the complete population.

Add `--overwrite` to regenerate only the selected mode's automatic Parquet.
Automatic mode refuses to replace an existing result without this flag. A
completed reviewer mode reports that no work remains and does not open a GUI.

`--mode automatic|semiautomatic|manual` is equivalent. The configuration's
`experiment` field dispatches to the experiment adapter. Only `arcterx` is
currently implemented; other values fail explicitly.

The old `drifterlab-detect-arcterx-drogue` and
`drifterlab-review-arcterx-drogue` entry points remain deprecated compatibility
commands. They are not the normal workflow.

## Current scientific methods

TTFF uses raw integer event counts in fixed value and time bins. At each time-bin
boundary it tests adequate pre-activity, a quiet 48-hour forward window,
120-hour persistence, and later reactivation. Each value bin contributes at
most one last stable cessation date. Agreement across at least two value bins
within the configured tolerance produces a clear TTFF date. Counts are never
normalized across value bins; all valid TTFF observations separately establish
coverage.

Strain uses one authoritative robust method. Finite observations are aggregated
into non-overlapping UTC-aligned block medians. Every downward split with enough
valid blocks on both sides is fitted with two medians and an L1 objective. The
best split is classified using absolute drop, relative drop, fit improvement,
and comparable well-separated alternatives. Missing blocks are not filled.

Both components expose the common statuses:

```text
clear  weak  ambiguous  no_change  insufficient_data
```

Detailed TTFF reasons remain available in the automatic evidence.

## Combination

- Two clear dates within 48 hours: `clear_agreement`, using the earlier date
  without averaging; source `ttff+strain`.
- Two clear dates outside 48 hours: `signal_conflict`, with no automatic date.
- One clear component plus `no_change` or `insufficient_data` from the other:
  `clear_ttff_only` or `clear_strain_only`.
- Other weak, ambiguous, or unresolved combinations: `unresolved`, with no
  automatic date.

The tolerance remains configurable. Neither detector tunes or moves the other.

## Reviewer

The compact reviewer has two scientific panels:

1. positive raw TTFF on a logarithmic scale;
2. raw strain, robust block medians, and fitted pre/post levels.

Both panels show component dates, the combined or selected physical loss date,
and the analysis cutoff. Controls are `Accept auto`, `Set manual`, `Not lost`,
`Uncertain`, one/six-hour nudges, navigation, quit, and a per-platform
cutoff margin.

The four decision controls save atomically and move to the next unfinished
platform. Date nudges remain drafts until `Set manual` is pressed. `Next` skips
without recording a decision. At the end of the table the reviewer returns to
any skipped or stale platform and closes only after a fresh check finds no work.
Changing the margin of an existing decision saves immediately.

The reviewer reuses its mode's automatic Parquet. It does not rerun
TTFF or strain detection, does not compute detailed TTFF-bin plots, and does not
extract hull temperature. Raw TTFF/strain and display block medians are cached
per platform during the session. Date or margin edits redraw only vertical
overlays and text.

Review statuses are:

```text
accepted_auto  manual_date  not_lost  uncertain
```

Changing the margin never changes a component, automatic, or reviewed physical
date. `not_lost` and `uncertain` both have no cutoff but retain distinct meaning.

## Products

The configured output names are bases. The unified command inserts a mode label
before each extension:

```text
auto_drogue_loss_auto.parquet
auto_drogue_loss_semi.parquet     drogue_review_semi.csv
auto_drogue_loss_manual.parquet   drogue_review_manual.csv
```

The modes are independent: switching mode neither reads nor overwrites another
mode's files. Existing unsuffixed products are not migrated automatically. The
deprecated experiment-specific commands retain their unsuffixed paths for
compatibility.

The automatic Parquet contains one row per platform with component status/date
and concise evidence, the combined status/source/date, default cutoff margin and
cutoff, source provenance, and exact effective configuration. It does not store
per-bin time series or strain objective curves.

The review CSV contains only actual human rows. It stores TTFF/strain/automatic
snapshots needed to detect stale automatic results, the human status and
physical date, selected margin and cutoff, reason, timestamp, and source hash.
Regenerating automatic results never overwrites the corresponding review CSV.
Changed automatic snapshots are marked stale and returned to the review queue;
changed raw-source hashes remain hard errors.

## Downstream resolver

`drifterlab.qc.resolve_drogue_decision(automatic_row, review_row)` computes the
effective state without writing another table:

- a human row takes precedence;
- `accepted_auto` and `manual_date` resolve to `lost` with the reviewed physical
  date and reviewed margin;
- without a human row, a dated automatic result resolves to `lost` using the
  configured default margin;
- `not_lost` remains `not_lost`;
- human `uncertain`, conflicts, and unresolved automatic cases remain
  `uncertain` and never become silently drogued-through-end decisions.

The cutoff is always `physical_loss_time - margin`.

## Configuration shape

The production YAML contains only:

```text
experiment
input
output
processing.missing_value
processing.analysis_cutoff_margin_hours
detection.ttff.binning
detection.ttff.comparison
detection.strain
detection.combination.agreement_tolerance_hours
```

There is no strain method switch, temperature detector configuration, or review
zoom configuration. Only settings used by the current detectors are accepted.
