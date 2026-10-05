# Overshoot-aware finite-size Lyapunov exponent

The FSLE workflow reads an existing authoritative `pairs.zarr`. It never searches
for pairs again and never changes the pair or reconstruction products. Run it with:

```powershell
drifterlab-fsle configs/arcterx/06_microSVP_fsle.local.yml
```

Use `--overwrite` only to atomically replace an existing FSLE output bundle.

## Input ensemble

The input may contain same-array, between-array, same-cluster, and between-cluster
pairs. The implemented primary analysis selects only records for which both
`same_array` and `same_cluster` are true. Cross-array and between-cluster pairs are
counted in the manifest but do not enter either primary spectrum.

This distinction does not undo the selection that created `pairs.zarr`. For
example, a product built with `maximum_distance_m: 25000` contains pairs that came
within 25 km, not every temporally overlapping pair. The FSLE manifest records the
input pair counts and encounter threshold so this conditioning remains visible.

The configured coordinate representation is read explicitly from the retained
member fields, such as `lon_linear_1`, `lat_linear_1`, `lon_linear_2`, and
`lat_linear_2`. Pair separation is recalculated as a WGS84 geodesic distance.

## Thresholds

Separation thresholds are constructed around an exact anchor:

```text
delta_n = anchor_scale_km * rho^n
```

for every integer `n` whose threshold lies within the configured minimum and
maximum. With an anchor of 1 km, 1 km is therefore an exact threshold rather than
the closest member of a grid that begins elsewhere. Consecutive thresholds define
the shell `[delta_n, delta_(n+1))`.

## Discrete first passages

Each pair begins at its stored `encounter_observation`; earlier observations in
the common window are excluded. For each shell:

1. shell entry is the first stored sample with
   `delta_n <= separation < delta_(n+1)`;
2. shell exit is the first later stored sample strictly above `delta_(n+1)`;
3. the passage time uses the actual stored timestamps;
4. the entry and exit separations retain their observed overshoot.

An upper-threshold observation before shell entry cannot be used as the exit. A
pair that skips an entire shell between stored samples has no observed entry for
that shell and is reported as not observed rather than assigned an interpolated
passage.

Analysis requires the configured time increment. It stops at the first missing
coordinate, missing time, or cadence mismatch and never restarts. A clock already
running at that boundary is right-censored. A clock that reaches the end of the
pair's common window without an exit is also right-censored.

## Overshoot estimator

The sole FSLE estimator is:

```text
lambda(delta_n) =
    mean[log(r_exit / r_entry)] / mean[T(delta_n)]
```

in inverse days. It uses only reached first passages in the numerator and
denominator. Censored and not-observed pairs remain in the passage table and in
the per-scale accounting; they are not silently treated as successful events.

The spectrum reports the total, at-risk, reached, censored, and not-observed pair
counts plus the number of contributing platforms. `minimum_reached_pairs_per_scale`
controls figure inclusion only. It never removes numerical rows.

## Standard-error bars

Let

```text
g_i = log(r_exit_i / r_entry_i)
tau_i = passage time of reached pair i
M = number of reached pairs
```

The nominal standard error saved as `fsle_standard_error_day_inverse` is

```text
sqrt(
    (mean(g_i^2 / tau_i) / mean(tau_i) - lambda(delta_n)^2) / M
)
```

This is the duration-weighted error calculation in `FSLE_error.pdf`, generalized
to retain the variable observed growth caused by discrete threshold overshoot.
When every `g_i = log(rho)`, it reduces exactly to the fixed-shell formula in the
PDF. At least two reached passages are required; otherwise the value is `NaN`.
The figures show the measured FSLE values as unconnected scatter markers and
`lambda +/- one standard error` when `plotting.standard_error_bars` is true. The
dashed reference-slope guide remains a line because it is a theoretical scaling
guide rather than a sequence of observations.

Pairs are not statistically independent when they share platforms. Consequently,
these are nominal sampling-error bars for comparison with the established FSLE
method, not confidence intervals. They can underestimate uncertainty in pooled
and individual-cluster spectra. The reported passage-time standard deviation
remains a separate descriptive statistic.

## Pooled and individual spectra

Two figures are written for every represented array:

- `fsle_same_cluster_pooled.png` combines all same-cluster pair passages in the
  array into one pair-weighted spectrum;
- `fsle_individual_clusters.png` keeps a separate line for every represented
  non-singleton cluster.

If clusters A, B, and C are present, the pooled calculation combines A-A, B-B,
and C-C pair events before calculating each FSLE value. It is not an arithmetic
average of three cluster FSLE curves. Because a cluster of size `N` can contribute
`N(N-1)/2` pairs, large clusters naturally receive more weight. The long-form
spectrum table distinguishes `same_cluster_pooled` and `individual_cluster` rows.

## Reference slope

The configured reference line is anchored to the pooled FSLE at the exact anchor
scale. For the default Richardson-like guide:

```text
lambda_reference(delta) =
    lambda_pooled(1 km) * (delta / 1 km)^(-2/3)
```

The same array-level reference is drawn in the individual-cluster figure, so
cluster amplitude differences are not removed by separately normalizing every
curve. If the pooled anchor scale does not meet the configured reached-pair
minimum, the reference line is omitted and the manifest records that the anchor
was unavailable. This is a visual slope guide, not a fitted law or an independent
prediction of its amplitude.

## Products and provenance

The output bundle contains:

- `fsle_spectra.csv`: long-form pooled and individual-cluster spectra, including
  the overshoot-aware nominal standard error;
- `fsle_first_passages.parquet`: every analyzed pair-shell reached, censored, or
  not-observed record;
- `array_NNN/fsle_same_cluster_pooled.png`;
- `array_NNN/fsle_individual_clusters.png`;
- `fsle_manifest.json`.

The manifest records the exact thresholds, estimator and start policy, input pair
schema and selection metadata, coordinate method, cluster pair completeness,
configuration, product hashes, and the input consolidated-metadata hash. The
workflow checks that the input hash is unchanged before publication.
