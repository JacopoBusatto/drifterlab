# Standalone drogue-loss detection and review

This workflow estimates physical drogue-loss times from raw `GpsTTFF` and
optional `Drogue` strain. Hull temperature is manual-review context only. It
never reads or plots supplied ARCTERX `drogue_off` values or another reference
loss date.

```text
raw TTFF and strain
    -> TTFF distribution change + strain rolling-median step
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

## TTFF distribution method

Each configured TTFF time bin records raw value-bin counts, valid sample count,
normalized fractions, actual observation coverage, and boundary-bin fractions.
With `include_lower_than_first_bin: true` (the backward-compatible default), log
TTFF bins include `[0, first_edge)` and the histogram retains its lower boundary
bin. With it set to `false`, values below `first_edge` are excluded from the
histogram, `n_valid`, normalization denominator, and coverage calculation. All
configurations retain an open `>= last_edge` bin. Missing, non-finite, and
configured sentinel values do not enter `n_valid`.

For every candidate-grid time, normalized distributions are built from all
valid observations in `[t-pre, t)` and `[t, t+post)`. Within finite value bins,
probability is represented uniformly. Open-bin mass is clipped to the first or
last finite edge for distance integration: it still affects occupancy and
direction but is never assigned an arbitrary midpoint or unbounded leverage.
TTFF uses the `log1p(value)` coordinate.

With `F_before` and `F_after` on that common coordinate:

```text
D_down = integral max(F_after - F_before, 0) dx
D_up   = integral max(F_before - F_after, 0) dx
W1     = D_down + D_up
downward_fraction = D_down / W1
```

`D_down` is the change magnitude used for candidate ranking. Acceptance requires
minimum `D_down`, predominantly downward direction, valid sample count, and
actual temporal coverage. `W1`, `D_down`, and `D_up` are a decomposition, not
independent scores.

Persistence compares both the full follow-up interval and its terminal time bin
with the pre-change distribution. Movement back toward the post-change high
regime is bounded by `max_reactivation_fraction`. Candidate selection occurs at
the strongest redistribution peak in each episode; a stronger transient peak
cannot be replaced by a weaker edge candidate. Comparable separated persistent
peaks produce `ambiguous`. A substantial end-of-record change without complete
follow-up produces `insufficient_followup`.

## Strain rolling-median step method

Strain event selection is separate from the TTFF histogram method. Raw strain is
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

## TTFF upper-cloud context

From the same bounded histogram, the detector computes

```text
A_tail = integral P(TTFF > x) d log1p(x)
```

between the configured physical bounds, initially 50-1000. Before/after areas,
their decrease, and the selected before/after survival curves are retained for
inspection. Tail depletion supports interpretation; it is not stacked into a
second confidence score. Values above 1000 remain visible in the open occupancy
bin but have no leverage beyond the cap.

## Signal combination

- A unique clear persistent strain step supplies the physical event
  time.
- A clear TTFF change raises confidence when its confirmed lower-state interval
  overlaps and persists through the strain change. No date averaging or fixed
  +/-24-hour coincidence is used.
- Clear strain without clear TTFF remains a medium-confidence strain candidate.
- Missing or unclear strain with clear sustained TTFF produces a provisional
  medium-confidence TTFF candidate.
- Ambiguous components or non-overlapping clear changes produce no automatic
  date and require review.
- Neither clear produces no automatic date.

## Exploratory defaults

TTFF distribution settings are unchanged:

| TTFF setting | Default |
| --- | ---: |
| scale / first edge / factor / last edge | log / 10 / 1.5 / 1000 |
| include lower-than-first bin | true |
| diagnostic time bin | 12 h |
| pre / post / persistence | 48 / 48 / 72 h |
| minimum valid samples / coverage | 24 / 0.75 |
| minimum `D_down` / downward fraction | 0.35 / 0.75 |
| maximum reactivation fraction | 0.20 |
| peak separation / comparable strength | 48 h / 0.80 |
| tail interval | 50-1000 |

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
  dates/statuses, TTFF directional distances and tail evidence, strain
  absolute/relative/normalized drops and robust levels, coverage, combined
  date/confidence, source hash, and exact configuration JSON.
- `drogue_review.csv`: explicit review decision, automatic time visible during
  review, reviewed physical time, and separate analysis cutoff.
- Existing per-drifter diagnostic PNGs: raw TTFF, normalized occupancy heatmap,
  directional TTFF metrics, selected TTFF survival curves, rolling strain with
  descriptive quantiles, strain-step metrics, temperature context, and
  automatic date lines.

No trajectory Zarr is modified and no observation is deleted.
