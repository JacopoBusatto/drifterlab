# ARCTERX raw/QC timing diagnostic

Optional troubleshooting only. The production preprocessing CLI and package API
do not run or expose this script. Outputs remain in `data/diagnostics/`.

The February 6 first-valid-QC-time cohort is not explained by a common three-day
cut at ingestion or upstream QC: **61 of its 65 primary raw observation series
already start on February 6**. The other four contain isolated earlier records.
This establishes the delivered data pattern, not the actual release dates or the
reason for the difference from the cruise-report date supplied in the request.

## Reproduce

Activate the repository environment, then run from the repository root:

```powershell
python -m pip install -e ".[diagnostics]"
python scripts/diagnostics/timing_diagnostic.py `
  --config configs/arcterx/microsvp_preprocessing.local.yml `
  --raw-directory "C:/path/to/ARCTERX/Data/Raw/MicroSVP" `
  --output-directory data/diagnostics
```

The existing YAML supplies the QC input directory, pattern, inventory path, and
missing-value sentinel. The diagnostic does not invoke preprocessing or write the
master. `--no-figures` runs the data comparison without requiring Matplotlib.
Repeated runs replace the diagnostic artifacts in the chosen output directory.

Raw files contain `dataset.drifter_<ID>`, `meta_<ID>`, and `diagnostic_<ID>`.
Primary observations use calendar strings in `ObsTimestamp` and coordinates in
`GpsLongitude`/`GpsLatitude`. The `PlatformId` scalar is cross-checked against other
embedded identifiers. QC files use the existing ARCTERX reader. Filenames and
alphabetical order do not determine matches; duplicates and ambiguities are
reported and not paired.

## Outputs

- `microsvp_timing_diagnostic.parquet` and `.csv`: one row per platform, sorted by
  first valid QC time; source paths/hashes, UTC times, first coordinates, row
  counts, early-row comparisons, case labels, and inventory checks.
- `microsvp_source_matching.json`: unique pairs, duplicate IDs, ambiguous matches,
  unmatched files, and identification errors.
- `microsvp_timing_summary.json` and `microsvp_timing_report.md`: cohort counts,
  delay statistics, outliers, definitions, and factual conclusions.
- `01_all_drifters_timing` and `02_feb06_cohort_timing`, each as a 300-dpi PNG and
  vector PDF. Timeline connectors describe time differences, not continuity of
  observations. The detailed figure marks off-panel January timestamps explicitly.

All generated outputs are under the Git-ignored `data/` directory.

## Definitions

First timestamp is the minimum parseable source time regardless of position.
First valid position additionally requires finite coordinates within longitude
[-180,180] and latitude [-90,90]. This matches the existing inventory definition;
it does not establish deployment, apply drogue masks, or assess raw GPS quality.

`delta_raw_to_qc_first_time_hours` is QC first timestamp minus raw first timestamp.
`delta_raw_to_qc_first_valid_position_hours` compares first valid positions in
the two products. `delta_raw_first_time_to_qc_first_valid_position_hours` also
records the distinction explicitly. The earliest raw positions happen to be
geometrically valid in the inspected files, but the implementation does not assume
that these definitions are interchangeable.

UTC comes from the confirmed ARCTERX campaign convention. Raw calendar strings
are parsed independently of MATLAB datenum conversion. A one-second tolerance
is used only for matching and reporting clock equality, accounting for the few
microseconds of floating-point MATLAB encoding. Parquet and CSV retain full
timestamp precision. Report/figure labels round timestamps to seconds for display.

Earlier raw rows are compared against native QC timestamps to distinguish omitted
rows from retained timestamps with masked positions. Separate telemetry timestamps
from `diagnostic_<ID>` are summarized explicitly; they are not treated as primary
observation timestamps. No source rows are edited or removed by this diagnostic.

## Findings from the 150 delivered pairs

All 150 raw/QC pairs matched uniquely, with no duplicate, unmatched, conflicting,
or ambiguous platform IDs. Recomputed QC first timestamps and first valid position
times match the inventory exactly for every drifter; all QC source hashes match.

The observed first-valid-QC-time cohorts are 15 on January 12–13, 6 on February 1,
65 on February 6, and 64 on February 11. Cohorts combine consecutive observed UTC
calendar dates; they are descriptive and do not assign deployment IDs.

For February 6, the requested cases are **A=61, B=1, C=0, D=3**, using UTC dates
to operationalize “near February 3/6.” The notable earlier-date exceptions are:

| Platform | Earlier raw observation | Treatment visible in QC |
|---|---|---|
| 300534067046080 | Feb 3 01:45; next raw record Feb 6 16:45 | Earlier timestamp omitted; first valid QC Feb 6 16:55 |
| 300534067042080 | Jan 16 04:20 | Earlier timestamp omitted; first valid QC Feb 6 03:00 |
| 300534067043080 | Jan 28 02:00 | Timestamp retained, position masked; first valid QC Feb 6 04:23 |
| 300534067046050 | Feb 5 09:20 | Timestamp retained, position masked; first valid QC Feb 6 03:37 |

Each has just one primary raw observation before February 6. Across all 65 files,
there is only **one** primary raw observation on February 3. The February 6 first
valid QC times span 01:59–17:00 UTC, wider than the approximate 02:00–06:00 interval
in the request. Additional same-day omissions and late starts are listed in the
generated report.

For the February 11 control, 61 raw observation series start on February 11;
three contain isolated earlier records (January 12, February 2, February 7) whose
timestamps are retained in QC with positions masked. First valid QC positions
for 63/64 fall within the supplied 11:00–16:00 UTC report window. The exception,
300534067834590, has 79 early raw rows omitted and first valid QC at 21:00 UTC.

In both February cohorts, the median raw-to-QC first-timestamp delay is zero and
the median raw-to-first-valid-QC-position delay is four minutes. Long extrema are
dominated by isolated earlier records and are reported explicitly in the output.
In each cohort, 63 files retain timestamps before their first valid QC positions.

No preprocessing bug was found. The preprocessing logic, master Zarr, inventory,
and original verification report were left unchanged. This diagnostic does not
infer release times, deployment clusters, pairs, or explanations for the source
clock/report discrepancy.
