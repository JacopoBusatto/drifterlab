# Robust two-regime strain detector

This is the authoritative strain detector used by production drogue detection.
The standalone diagnostic runs the same implementation plus aggregation-time
sensitivity checks; it does not alter the automatic Parquet or reviewer. The
detector never reads supplied `drogue_off` values and does not use TTFF dates.

## Algorithm

Raw finite, non-sentinel strain observations are grouped into non-overlapping,
UTC-aligned intervals. Each interval contributes one median only when it has the
configured minimum raw sample count; missing intervals are retained for display
but are not filled or smoothed. The representative block timestamp is its center.

For valid block medians `s_i`, the one-regime fit is their median `m0` with L1
cost:

```text
J0 = sum |s_i - m0|
```

Every split with the configured equivalent number of valid block-hours on both
sides is evaluated. Each side is represented by its median and the two-regime
cost is:

```text
J1(k) = sum_before |s_i - m_before| + sum_after |s_i - m_after|
```

Only splits with `m_before > m_after` can be selected. The reported candidate
time is the left edge of the first valid post-change block. The fit improvement
is `(J0 - J1_best) / J0`; it is defined as zero when `J0` is zero.

A well-separated downward candidate is comparable when its cost is no more than
`comparable_cost_fraction` above the minimum. A useful best fit with such an
alternative is `ambiguous`. Otherwise a fit passing both drop thresholds and the
fit-improvement threshold is `clear`; passing the improvement threshold and one
drop threshold is `weak`. Rejected changes are `no_change`, while a
record without enough valid blocks on both sides is `insufficient_data`.

The 3-, 6-, and 12-hour fits are reported independently. They are not combined
into a score or used to move the primary 6-hour date.

## Exploratory defaults

```yaml
strain_two_regime:
  aggregation_hours: 6
  min_samples_per_block: 3
  minimum_side_duration_hours: 72
  minimum_drop_absolute: 2.0
  minimum_drop_relative: 0.08
  minimum_fit_improvement: 0.10
  relative_floor: 1.0
  ambiguity:
    comparable_cost_fraction: 0.02
    minimum_separation_hours: 48
sensitivity_aggregation_hours: [3, 6, 12]
```

These values are exploratory and have not been calibrated against a reference
loss date.

## Reproduce

From the repository root:

```powershell
python scripts/diagnostics/strain_two_regime.py `
  configs/arcterx/drogue_detection.local.yml `
  --settings configs/arcterx/strain_two_regime.yml `
  --output data/diagnostics/strain_two_regime `
  --platform 300534067833760 `
  --platform 300534061906090
```

Each figure shows raw strain, block medians, fitted levels, the proposed or
rejected split, optional sensitivity dates, and the full split-cost curve over
the actual strain observation interval.
