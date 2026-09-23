# Sustained native-QC trajectory start

This optional historical diagnostic is retained for troubleshooting. Downstream
broad timing populations use the production `first_valid_position_time_qc` field:
all 150 inferred starts matched it exactly, so sustained-start inference is not
part of the production preprocessing CLI or package API.

`inferred_start_time` is the beginning of the first sustained valid native QC
segment. It is a derived diagnostic, not source metadata or a verified deployment
time. The diagnostic reads the master Zarr and the existing timing table; it
preserves the master, inventory, source data, and previous timing outputs.

## Run

From the repository root, after running the preprocessing and raw/QC timing
diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts/diagnostics/sustained_start.py
```

Plots use the optional `diagnostics` extra (`pip install -e ".[diagnostics]"`).
Use `--no-figures` for tables and reports alone. Defaults are:

```text
--master data/ARCTERX_MicroSVP_QC.zarr
--timing-table data/diagnostics/microsvp_timing_diagnostic.parquet
--output-directory data/diagnostics
--continuity-gap-minutes 30
--minimum-duration-minutes 60
--minimum-observations 13
```

Paths are relative to the working directory. With an explicit duration but no
observation-count option, the count is `ceil(duration_minutes / 5) + 1`.
Rerunning replaces this diagnostic's outputs. Master/timing-table IDs, QC source
hashes, first QC times, first valid QC times, and native position-validity masks
are checked before writing results. Output inside the master store is rejected.

## Rule and empirical choice

Use finite, in-range native QC positions with valid timestamps. Stable-sort these
fixes, retaining source indices, and split whenever the gap between consecutive
valid fixes exceeds **30 minutes plus one second**. Missing-position runs can be
bridged within this limit; this is a continuity diagnostic, not the adjacent-row
displacement-speed audit. Drogue validity, analysis masks, reconstructions, speed
flags, and spatial deployment assumptions have no role.

Select the earliest segment with at least **60 minutes of elapsed duration** and
**13 observations at distinct times**, allowing one second at the duration
boundary. This corresponds to 13 fixes including both endpoints of an hour at
five-minute sampling. Duplicates remain in the segment observation count, but
cannot inflate distinct-time support. The start is the segment's first fix, not
the time at which it accumulates an hour of observations. A missing qualifying
segment gives `NaT`, a failure status, and a manual-review flag; there is no
fallback. Saved times are never rounded or snapped to a grid.

The inspected 150 tracks contain 3,753,372 valid timed positions and 3,753,222
consecutive gaps. Median/p90/p95 are approximately 5 minutes, p99 is 6 minutes,
p99.9 is 10 minutes, and the maximum is 181 hours 40 minutes. With one-second
comparison tolerance, 2,821 gaps exceed 30 minutes, producing 2,971 segments.
The 30-minute choice accommodates ordinary missed fixes while separating long
interruptions; it is a conservative documented choice, not a uniquely identified
physical boundary. Strict and tolerance-adjusted counts for 10/15/20/30/60 minutes,
6 hours, and 24 hours are recorded separately.

Descriptive status prioritizes the number of skipped valid segments: none gives
`same_or_nearly_same`, one gives `earlier_isolated_fragment_removed`, and more
than one gives `multiple_early_segments`. No qualifying segment gives
`no_sustained_segment_found`. A separate boolean compares starts within five
minutes plus the one-second tolerance. "Removed" means skipped for inference;
observations remain stored in their original products.

## Results on the delivered files

All **150** default inferred starts equal first valid QC times **exactly**.
No early valid segments are skipped and no trajectory lacks a sustained start.
The shortest first valid segment lasts approximately 100 minutes with 21 fixes.

The earlier January/early-February anomalies in the previous figure were raw
observations or QC timestamps whose positions were invalid. They already fail
the valid-QC filter. The sustained rule confirms the later starts; it does not
produce an additional correction to first-valid QC times. The report cross-checks
these platforms explicitly and the new figures preserve all four timing markers.

The inferred-start populations are derived from consecutive occupied UTC dates,
using the same descriptive rule as the prior diagnostic:

| UTC dates | Drifters |
|---|---:|
| January 12–13, 2025 | 15 |
| February 1, 2025 | 6 |
| February 6, 2025 | 65 |
| February 11, 2025 | 64 |

These counts are unchanged from the first-valid-QC populations. They are not
deployment IDs, and the rule does not establish actual release times.

## Sensitivity and review

At 30-minute continuity, the 30/60/120-minute sustained criteria require 7/13/25
distinct observations. The 30- and 60-minute results are identical for all 150.
The 120-minute rule changes one platform, **300534067830580**, from February 6
05:20 to 08:10 UTC, a delay of 2 hours 50 minutes. Its initial segment has 21
regular fixes over 100 minutes, followed by a 70-minute interruption. Keeping
the 60-minute default avoids rejecting this supported short initial segment.
It is flagged for review of criterion sensitivity, without invalidating its
default estimate. All population counts remain stable.

An additional small continuity check uses 10/15/20/30/60 minutes at the selected
duration/count criterion. The 15–60-minute choices give identical starts.
Ten minutes shifts **300534067045080** by 16 minutes after splitting off its
initial two fixes; this stricter choice does not change any population count.
The gap distribution supports allowing this sampling interruption. No platform
requires a spatial rule to resolve a surviving isolated earlier-date start.

## Outputs

All outputs are under `data/diagnostics/` and ignored by Git:

- `microsvp_sustained_start_diagnostic.csv` and `.parquet`: every original timing
  column, inferred start, duration/delta in hours, segment counts, selected segment
  observation support and source index, settings, status, population and review flags.
- `microsvp_valid_qc_segments.csv` and `.parquet`: every segment's platform ID,
  zero-based segment index, start/end, duration in hours, observation/distinct-time
  counts, median/internal maximum gap in seconds, and endpoint source indices.
  Singletons have zero duration and missing internal-gap statistics.
- `microsvp_sustained_start_sensitivity.csv`: one row per platform and tested
  duration criterion, including inferred times, settings, and support.
- `microsvp_valid_qc_gap_distribution.json`: empirical quantiles and threshold counts.
- `microsvp_sustained_start_summary.json`: counts, populations, individual changes,
  settings, software version, code/input hashes, processing time, and conventions.
- `microsvp_sustained_start_report.md`: readable findings and anomaly cross-check.
- `01_all_drifters_timing_with_inferred_start.png` / `.pdf` and
  `02_feb06_cohort_timing_with_inferred_start.png` / `.pdf`: four-marker figures.

CSV uses full-precision UTC ISO timestamps; import `platform_code` as text in
spreadsheet tools. Parquet preserves string IDs, nullable integer support fields,
booleans, and nanosecond UTC times directly. The local validation also records
whole-master and prior-output hash comparisons in
`microsvp_sustained_start_integrity.json`.
