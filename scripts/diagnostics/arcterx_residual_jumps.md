# Existing residual-jump diagnostic

This optional script explains the native QC flags already stored in the master.
It does not change the 3 m/s threshold, positions, QC definitions, validity masks,
inventory, or master Zarr. It uses no interpolated trajectory or drogue filter.

## Run

From the repository root in the installed environment:

```powershell
python scripts/diagnostics/residual_jumps.py
```

Defaults are `--master data/ARCTERX_MicroSVP_QC.zarr` and
`--output-directory data/diagnostics/residual_jumps`. Add `--qc-directory` with
the original QC MicroSVP directory to verify every flagged endpoint against its
source MATLAB row and source-file hash. `--no-figures` omits plotting; figures
otherwise use the existing optional `diagnostics` dependency extra.

`python -m scripts.diagnostics.residual_jumps` is equivalent. All paths are
relative to the working directory. Reruns replace this diagnostic's outputs.

## Reproduction and traceability

The script calls the unchanged production `consecutive_speed` and
`interval_seconds` functions. Every native row's recomputed speed (including
NaNs) and flag must equal its stored value exactly; a discrepancy stops the run.
The master must specify the existing strict `> 3 m/s` threshold. There is no
threshold override or alternative flagging algorithm.

The production speed is spherical haversine distance divided by actual positive
elapsed time between **immediately adjacent** chronological QC rows whose
positions and times are valid. It is assigned to the later row. Missing rows are
never compressed out, and five minutes is never substituted for the time delta.
There is no maximum-gap exclusion in that implementation, though no observed
flag crosses a materially greater-than-ten-minute interval.

`row_index_1/2` refer to zero-based native QC rows, including invalid positions;
they are not indices in a valid-only subset. `source_obs_index_1/2` map the same
endpoints to the original delivered MATLAB vectors, which run in reverse time
order for this campaign. `distance_m` is the production haversine displacement,
recovered as exact audit speed times actual elapsed seconds. All times are UTC
and retain nanosecond precision. The script does not round input timestamps.

Missing-position counts refer to nonfinite longitude or latitude. The additional
invalid-position count also includes finite coordinates outside the allowed
ranges. Nearby invalid positions outside a flagged pair are recorded separately
from positions between its endpoints.

## Delivered-data findings

All **1,307 flags in 140 drifters** reproduce exactly across 3,755,180 native rows.
All flagged pairs are adjacent in both the master and original QC vectors; none
spans missing or invalid positions. Of 2,801 adjacent pairs touching an invalid
position or time, none receives a finite audit speed. There are 141 flagged pairs
with an invalid position immediately before or after the pair; that nearby QC
masking does not imply bridging.

| Interval test | Strict timestamps | One-second reporting tolerance |
|---|---:|---:|
| <=6 minutes | 1,209 | 1,215 |
| <=10 minutes | 1,293 | 1,307 |
| >10 minutes | 14 | 0 |
| >30 minutes | 0 | 0 |

The strict >10-minute cases exceed ten minutes by at most 4.471 microseconds,
consistent with floating-point MATLAB datenums. Tolerance affects these summary
bins only, never the speed calculation or flag threshold. Of the 1,307 intervals,
316 are approximately one minute and 554 are nominally five minutes. Short
intervals amplify speed for a given displacement, but they are supplied source
timestamps rather than an ingestion artifact.

Median/p95/maximum flagged speeds are **3.929 / 9.167 / 72.642 m/s**.
Median/p95/maximum distances are **973 / 2,037 / 11,018 m**. The top ten drifters
account for 27.93% of flags. The highest counts are 66 for `300534067833570`,
60 for `300534063014450`, and 45 for `300534067045000`; the full table includes
rates per 10,000 eligible pairs to account for different trajectory lengths.

The local run verified **2,460 unique flagged endpoints in 140 source files**:
source hashes, coordinates, timestamps, and supplied speed values match exactly.
This verifies the delivered QC vectors; it does not establish whether upstream
QC omitted records from the raw inputs.

## Supplied speed

The campaign `Arcterx Data Guide.docx` describes speed QC; `Data Processing
MicroSVP.pdf` discusses filtering, subsampling, and spline reconstruction.
Neither establishes the exact differentiation/time-denominator/endpoint
definition for native `drifter.speed`. The master records this uncertainty.

Both supplied endpoint speeds are retained as source context under names ending
in `m_s_definition_unverified`. They are not treated as equivalent to the audit,
used for error scores, or plotted as a reference velocity. Their smaller values
alone cannot establish a bug in either calculation.

## Inspection cases and artifacts

The two figures are a speed-versus-interval scatter and one four-case figure
combining native trajectories with audit-speed time series. The case windows
preserve masked fixes and break lines there; local east/north coordinates are
used only for display. Cases are selected deterministically by maximum speed,
maximum distance, maximum speed at nominal cadence, and the highest-count
platform's maximum speed, removing duplicate selections.

Priority examples, with UTC times rounded to seconds for display:

- `300534063210260`, March 4 20:24–20:25: 4.36 km in one minute (72.64 m/s).
- `300534067044090`, February 12 21:57–22:00: 11.02 km in three minutes
  (61.21 m/s), beside another large flagged leg and nearby masked fixes.
- `300534067834700`, February 13 21:55–22:05: consecutive five-minute legs
  of approximately 6.4 km (21.30 and 21.42 m/s).
- `300534067833570`, March 18 09:54–09:55: a 608 m one-minute excursion;
  this platform has the highest total flag count.

The worst-20 table also identifies `300534063218230` (29.97 m/s) and
`300534063016620` (26.55 m/s, with several large neighboring excursions), among
other inspection candidates. Multiple flagged legs can be one incident; these
diagnostics do not decide which endpoint is wrong.

Outputs under `data/diagnostics/residual_jumps/`:

- `flagged_jumps.csv` / `.parquet`: all endpoint, interval, distance, speed,
  validity, row/source-index, adjacent-mask, and source-provenance fields.
- `flags_by_platform.csv`: all 150 platforms, counts, rates, and audit checks.
- `worst_20_jumps.csv`: largest individual audit speeds.
- `selected_case_observations.csv`: unfiltered native context for the four cases.
- `summary.json` and `report.md`: requested counts, quantiles, findings, cases,
  source checks, software/code hashes, processing date, and conventions.
- `01_speed_vs_interval.png` / `.pdf` and `02_selected_cases.png` / `.pdf`.

These are actual adjacent-position differences present in the delivered QC
coordinates, not an artifact of bridging missing observations. Their physical
interpretation remains open; no additional QC has been applied.
