# Standalone drogue-loss detection and review

This workflow estimates physical drogue-loss times from raw `GpsTTFF` and
optional `Drogue` strain. Hull temperature is manual-review context only. It
never reads or plots supplied ARCTERX `drogue_off` values or another reference
loss date.

```text
raw TTFF and strain
    -> TTFF per-bin activity cessation + strain rolling-median step
    -> strain-primary combination rule
    -> manual review
    -> reviewed_drogue_loss_time
    -> analysis_cutoff_time = reviewed time - configured margin
```

## Run

Set the raw and output paths in `configs/arcterx/drogue_detection.local.yml`.
After a detector-method change, regenerate the automatic table explicitly:

```powershell
drifterlab-detect-arcterx-drogue configs/arcterx/drogue_detection.local.yml --overwrite
```

Generate one diagnostic figure:

```powershell
python scripts/diagnostics/drogue_signals.py configs/arcterx/drogue_detection.local.yml --output data/diagnostics/drogue_signals --platform 300534061906090 --no-population-figures
```

Launch or resume review:

```powershell
drifterlab-review-arcterx-drogue configs/arcterx/drogue_detection.local.yml
```

The reviewer retains Accept auto, Set manual, No loss, Uncertain, time
adjustments, navigation, save, and resume behavior. The automatic Parquet can be
replaced explicitly, but the review CSV is never overwritten by detection. If a
saved decision refers to a superseded automatic time, review startup stops with
an incompatibility error. Start a new review iteration or migrate it explicitly;
do not silently reuse the old decision.

## TTFF per-bin cessation method

Fixed temporal intervals retain the raw integer event count for every configured
TTFF value bin. Counts are never normalized across value bins. `N_valid` and
timestamp coverage are calculated separately from every finite, non-sentinel
TTFF observation. Therefore values below an excluded first value edge still
establish telemetry availability even though they do not enter a configured
event bin. The upper value bin remains open (`>= last_edge`).

At each temporal-bin boundary `t`, a value bin first has to establish active
baseline behavior in `[t-168h, t)`: at least 12 events, at least three occupied
temporal bins, and at least 75% timestamp coverage. Its baseline rate is raw
events per observed coverage hour. The forward `[t, t+48h)` window is quiet when
it contains at most one event or its event rate is at most 10% of the baseline
rate. The detector then requires every complete 48-hour forward window through
`t+120h` to be adequately covered and quiet. A record that does not physically
reach that horizon yields `insufficient_followup`.

After the persistence horizon, every complete, adequately covered 48-hour
window through the observed record is checked for reactivation. Inadequately
covered late windows provide no evidence either way. A candidate is rejected if
an observed window exceeds both the one-event allowance and the 10% rate limit.
Candidates are searched newest to oldest, so an early lull followed by activity
is rejected and a later final cessation supplies the reported bin boundary.

Each eligible value bin contributes at most one stable-drop date. Dates form
agreement groups only when their complete span is within 48 hours. The largest
group wins, with a later median breaking size ties. Two or more agreeing bins
produce `clear` unless another disjoint group also contains at least two bins.
A single stable bin is `weak`; conflicting stable dates are `ambiguous`.
Singleton outliers remain in the detailed in-memory evidence but do not
invalidate a clear group. Other TTFF statuses are `insufficient_followup`,
`insufficient_data`, `none`, and `unavailable`.

## Strain rolling-median step method

Strain event selection is separate from the TTFF count method. Raw strain is
smoothed with a centered, time-based rolling median. At each observed candidate
time, the detector compares the median of that smoothed signal in configured
pre- and post-windows:

```text
drop_absolute   = median_before - median_after
drop_relative   = drop_absolute / max(abs(median_before), relative_floor)
normalized_drop = drop_absolute / max(MAD_before, MAD_after, variability_floor)
```

A candidate must pass the configured absolute, relative, and normalized
thresholds. The later persistence window must remain within
`persistence_tolerance` of the new post-change level and must retain the required
drop from the pre-change level. Thus an isolated spike is suppressed by the
rolling median, while a temporary dip that returns to the earlier level is
rejected. Separate qualifying episodes produce `ambiguous`; a qualifying event
without a complete persistence interval produces `insufficient_followup`.

The pre/post q25-q75 values are retained for visual context only. They do not
form a competing event selector.

## Signal combination

- A unique clear persistent strain step supplies the physical event
  time.
- A clear TTFF cessation raises confidence when suppression remains adequately
  observed through the strain change. No date averaging is used.
- Clear strain without clear TTFF remains a medium-confidence strain candidate.
- Missing or unclear strain with clear sustained TTFF produces a provisional
  medium-confidence TTFF candidate.
- Ambiguous components or non-overlapping clear changes produce no automatic
  date and require review.
- Neither clear produces no automatic date.

## Exploratory defaults

TTFF value-bin definitions remain configurable and unchanged. The production
cessation defaults are:

| TTFF setting | Default |
| --- | ---: |
| scale / first edge / factor / last edge | log / 10 / 1.5 / 1000 |
| include lower-than-first bin | true |
| temporal-bin cadence | 12 h |
| forward count window | 48 h |
| pre-activity / persistence | 168 / 120 h |
| minimum pre events / occupied temporal bins | 12 / 3 |
| minimum coverage | 0.75 |
| quiet rate fraction / event allowance | 0.10 / 1 |
| reactivation window / agreement tolerance | 48 / 48 h |
| minimum agreeing value bins | 2 |

The provisional strain-step defaults are:

| Strain setting | Default |
| --- | ---: |
| rolling window | 12 h |
| pre / post / persistence | 48 / 48 / 48 h |
| clear absolute / relative / normalized drop | 2.0 / 0.10 / 1.5 |
| weak absolute / relative / normalized drop | 1.0 / 0.05 / 0.75 |
| relative / variability floor | 1.0 / 1.0 |
| persistence tolerance | 1.0 |

Temperature background/variability windows are 24/6 hours. The analysis cutoff
margin is 24 hours. These parameters are provisional smoke-test defaults, not
calibrated scientific thresholds. TTFF bin sensitivity and strain window/drop
thresholds must be assessed on representative tracks without reference dates
before downstream validation.

## Outputs

- `auto_drogue_loss.parquet`: one row per raw file with independent component
  dates/statuses, concise TTFF eligible/stable/agreeing-bin counts, consensus
  span, agreeing-bin pre/post event totals and persistence coverage, strain
  absolute/relative/normalized drops and robust levels, combined date/confidence,
  source hash, and exact configuration JSON. Detector changes require explicit
  regeneration because this automatic schema is versioned by its content.
- `drogue_review.csv`: explicit review decision, automatic time visible during
  review, reviewed physical time, and separate analysis cutoff.
- Existing per-drifter diagnostic PNGs: raw TTFF, integer value-bin count
  heatmap, per-bin forward counts/drop markers/statuses, `N_valid`, availability
  shading, rolling strain with descriptive quantiles, strain-step metrics,
  temperature context, and automatic date lines.

No trajectory Zarr is modified and no observation is deleted.
