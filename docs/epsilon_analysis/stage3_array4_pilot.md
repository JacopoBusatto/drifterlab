# Stage 3 Array 4 review pilot

## Run identity and scope

The first Stage 3 pilot is `epsilon-v1-b75795ab319a6e3e`. It uses configurable
Array 4, within-cluster pairs, `minimum_active_members: 2`, linear positions, the
30-minute centered two-endpoint derivative, full seven-position validity, a
30-minute maximum source-support gap, no speed mask, and the previously frozen
62.5 m–128 km separation edges.

The input remains `candidate_pending_gap_review`. The null minimum reliable
separation, 62.5 m first bin edge, reconstruction choice, q histogram edges, and
bootstrap block durations remain provisional scientific choices.

The run contains 13 activated clusters, 46 snapshot pair observations, and
16,997 temporal pair observations. Its activation anchors and numerical point
estimates match the documented Stage 2 baseline.

## Validation

- The full repository suite passes: 334 tests.
- Focused epsilon and workflow tests pass: 13 tests.
- Tests cover deterministic synchronized block resampling, scheduled empty
  blocks, histogram normalization and missing columns, complete removal of every
  pair containing a left-out platform, and the existing Stage 2 scientific
  identities.
- The maximum epsilon weighted-identity error remains
  $2.71\times10^{-20}\ \mathrm{m^2\,s^{-3}}$.
- All 278 nonempty conditional-histogram columns sum to one to within
  $1.11\times10^{-16}$. No valid q observation falls outside the configured
  $[-0.005,0.005]\ \mathrm{m^2\,s^{-3}}$ histogram range.
- Forty-two PNGs were generated: 39 cluster products and three array comparison
  figures.

The actual pilot figures were visually inspected, including the influential
Cluster 006 six-hour result, a single-pair Cluster 008 snapshot, the six-hour
cluster comparison, and the temporal-window small multiples. Signs, units, zero
lines, snapshot labels, confidence-interval labels, legends, missing-bin color,
and aligned separation limits are present. The heatmap's saturated central cells
represent probability mass at the shared configured color cap, not clipped raw
observations.

## Bootstrap intervals

The primary bootstrap uses 2,000 synchronized 30-minute blocks and a fixed seed.
The 60-minute calculation is retained as a block-duration sensitivity. Blocks
are aligned to each cluster window start and are resampled only after the fixed
window selection. All pair observations in a block remain together; activation
and cluster age are not recomputed.

Of the 186 temporal bins that pass the descriptive Stage 2 support criteria:

| Window | Supported bins | 30-minute CI available | 30-minute CI suppressed for blocks |
|---|---:|---:|---:|
| $[0,6)$ h | 80 | 72 | 8 |
| $[0,12)$ h | 106 | 101 | 5 |

Across both windows, 173 primary intervals are available and 13 are suppressed
because fewer than six blocks contribute. A further 386 temporal rows have no
interval because their descriptive separation-bin support already fails.
Snapshots are labelled `not_applicable_snapshot`.

The 60-minute sensitivity yields 139 intervals and suppresses 47 otherwise
supported bins for inadequate contributing blocks. For the 139 bins where both
durations yield an interval, the median 60-to-30-minute interval-width ratio is
1.012, with an interquartile range of 0.794–1.135. Seven bins change whether the
interval includes zero. This sensitivity means the block duration should remain
provisional rather than treating one interval boundary as definitive.

Of the 173 primary intervals, 138 include zero. This is reported as an interval
property, not as a cascade-regime hypothesis test.

## Platform influence

The detailed leave-one-platform-out product contains 1,122 recalculated
estimates. Among supported temporal bins, 57 six-hour and 63 twelve-hour bins
show at least one sign reversal after a platform is removed. Many removals also
leave inadequate support; the corresponding `support_ok_after_removal` flag must
be inspected before interpreting the recalculated magnitude.

The largest absolute changes occur in the already influential Cluster 006 and
Cluster 008 small-separation bins. For example, Cluster 006 at 125–176.8 m in
the six-hour window has
$\widehat\varepsilon=-1.700834\times10^{-4}\ \mathrm{m^2\,s^{-3}}$; its largest
platform-removal change is approximately
$4.18\times10^{-3}\ \mathrm{m^2\,s^{-3}}$, but the remaining sample fails the
support criteria. The influence result is therefore a warning about population
dependence, not a preferred replacement estimate.

No observation is removed or winsorized by Stage 3.

## Output and figure interpretation

The run is located at:

```text
C:\Users\Jacopo\Documents\DATI\ARCTERX\ARCTERX\Data\myQC\MicroSVP\epsilon_analysis\linear\centered_difference_30min_on_5min_grid\epsilon-v1-b75795ab319a6e3e
```

Routine inspection still starts with `clusters.csv`, `epsilon_by_scale.csv`, and
`coverage.csv`. Primary confidence intervals and summary influence flags are in
`epsilon_by_scale.csv`. Detailed conditional histograms, both bootstrap block
durations, and every platform removal are stored under `array_004/data/`.

The [figure guide](figure_guide.md) explains every cluster panel and comparison
figure. The most direct pilot examples are:

- `array_004/figures/array_004__cluster_006/initial_0_6h.png` for a highly
  influential temporal result;
- `array_004/figures/array_004__cluster_008/snapshot.png` for a descriptive
  single-pair snapshot;
- `array_004/figures/comparisons/epsilon_by_window_small_multiples.png` for
  within-cluster window comparisons;
- `array_004/figures/comparisons/epsilon_by_cluster__initial_0_6h.png` for scale
  overlap across clusters without pooling them.

## Stage status

There are no known departures from the agreed Stage 3 numerical specification.
The figures and intervals reinforce the Stage 2 caution: rare increments and
individual platforms materially influence many estimates, and the signs vary by
cluster, scale, and window. Stage 4 reconstruction and mask comparisons should
therefore remain separate sensitivity runs using the fixed baseline activation
anchors.
