# Direct FSLE analysis of v2 trajectory products

Run `drifterlab-analyze` using a local copy of
[`configs/arcterx/v2/fsle.yml`](../configs/arcterx/v2/fsle.yml). Select one
grouped trajectory product, arrays, and clusters. FSLE can run alone or alongside
the other unified analysis modules. It uses positions and timestamps, independent
of velocity definitions, velocity masks, and epsilon settings. The existing
`drifterlab-fsle` command and its legacy pair-Zarr interpretation are unchanged.

The input store is opened read-only. Distances are WGS84 geodesic separations.
Each pair's distances are calculated in memory and discarded after extracting
compact first-passage records. No pair Zarr or duplicated trajectories are written.

## Populations and time interval

Both array populations are always calculated, separately for each array:

- `within_cluster_pooled`: every unique within-cluster pair with valid overlap,
  starting at its first common valid position. No encounter-distance filter applies.
- `cross_cluster`: unique cross-cluster pairs whose minimum separation is at most
  `chance_pair_threshold_km` (default 10 km). Each starts at its earliest global
  minimum in the selected interval. Cross-array pairs are never formed.

There is also an `individual_cluster` spectrum for every selected cluster,
including an explicit empty spectrum for singletons. Cluster selection restricts
both populations to selected platforms. Pooling combines pair passages before
calculating FSLE; it does not average cluster spectra. Large clusters therefore
contribute more pairs.

`time.start` is inclusive and `time.end` exclusive. Null bounds use the full
selected product. The same interval restricts pair discovery, closest-approach
search, and passage extraction. Closest approach is conditional on the available
record and uses future information; its interval, distance, timestamp, and start
policy are retained in the catalogue and metadata.

## Shells, overshoot, and provisional passage policies

Thresholds are `anchor_scale_km * rho^n`, restricted to the configured scale
range. Consecutive thresholds define half-open shells `[lower, upper)`.

An observed entry requires a sample inside the shell immediately preceded on the
product grid by a valid sample below its lower boundary. An exact lower-boundary
sample at the pair's analysis start also establishes entry. A pair initially
partway through a shell has no witnessed entry into that shell. The first later
sample at or above the upper boundary is its exit. A temporary contraction does
not reset the running clock. Each pair-shell is attempted only once.

Entry and exit distances retain observed overshoot. If an exit sample is also a
witnessed entry into another shell, the same position and timestamp serve both
passages. No threshold crossing times are interpolated. A shell skipped without
an observed entry contributes nothing. Reaching the upper boundary exactly counts
as an exit (the legacy command requires a sample strictly above it).

The policy remains explicit while its scientific interpretation is reviewed:

- `strict_observed` (default) rejects an active passage at the first missing
  pair position or cadence discontinuity. Later shells may acquire fresh observed
  entries after the interruption; analysis does not stop permanently at that gap.
- `bridge_missing` permits an already-started passage to close across missing
  samples, using its full elapsed time and observed growth. It never creates a
  new shell entry across missing samples or fills intermediate skipped shells.

For thresholds `1, 2, 4, 8, 16, 32, 64, 128` km and distances
`0.9, 1.1, 1.2, 1.4, 2.6, NaN, NaN, NaN, 19.5, 25, 31.5, 32.1, 45, 64.5`:

| Shell lower threshold | Strict | Bridge |
| --- | --- | --- |
| 1 km | 1.1 to 2.6 | 1.1 to 2.6 |
| 2 km | Interrupted | 2.6 to 19.5 |
| 4, 8, 16 km | No observed entry | No observed entry |
| 32 km | 32.1 to 64.5 | 32.1 to 64.5 |

`maximum_source_gap_minutes: null` adds no source-gap filter. A finite value
requires both members' finite positions to have finite source-gap metadata at or
below that limit. Rejected source support interrupts a passage under either
policy. For bridge mode, the same finite limit also caps elapsed intervals between
usable observations. Actual missing positions remain governed by the passage
policy. Existing missing positions in the input are never reconstructed.

## Estimator and support

Only completed passages enter

```text
lambda(delta) = mean(log(r_exit / r_entry)) / mean(passage_time_days)
```

The plotted scale is the shell's nominal lower boundary. Incomplete passages
receive no FSLE value, but stay in the passage table as censored records. Every
spectrum retains total, entered, completed, censored, and unobserved pair counts,
the censored fraction, and contributing platform counts. Completed-only estimates
can favor faster passages when censoring is substantial.

`minimum_reached_pairs_per_scale` (default 3) controls plotting and fit eligibility;
numerical rows are retained regardless of support. The optional error bars use
the existing overshoot-aware nominal standard error. Shared-platform pairs are
correlated; these are not confidence intervals. An insufficiently supported or
empty population receives a labelled empty figure with the same axes.

## Prescribed-power fits and compensated plots

Each entry of `fits` specifies an ID, numeric `power`, `scale_min_km`, and
`scale_max_km`. YAML does not evaluate fractions; write `-0.6666666666666666` for
`-2/3`. There is no automatic search for scale ranges or free exponents.

For every spectrum and fit specification, only supported shells whose entire
`[lower, upper)` interval lies within the configured bounds are used. At least
two shells are needed. Fit only the amplitude of `lambda = A * delta^power`, with
equal shell weight in natural-log space:

```text
log(A_SI) = mean(log(lambda_seconds_inverse) - power * log(delta_metres))
```

The fit table saves the exact shells used, requested and actual range, SI
coefficient and units, log residual RMS, and a status for unavailable fits.
Residuals for competing powers should be compared over identical eligible shells.
No independence-based fit confidence interval is calculated.

For `power = -2/3`, `A^3` is also saved in m² s⁻³ as a flux-magnitude proxy, not
an epsilon estimate. Conversion requires an explicit physical proportionality
constant and evidence that Richardson scaling applies. Other powers receive no
epsilon-related quantity. A Richardson slope by itself establishes neither flow
dimensionality nor cascade direction.

Each spectrum has its own standard figure plus one compensated figure per unique
configured exponent: `lambda * delta^(-power)`. Compensation uses metres and
seconds so a fitted horizontal level equals `A_SI`. Multiple fit ranges sharing
an exponent appear on that exponent's one compensated plot.

All standard figures share x and y limits across every selected array and cluster.
Compensated figures share x limits and one common y range per exponent; different
exponents have different dimensions. Automatic limits include supported points,
error bars, and fitted curves; explicit standard limits may override them. There
are separate cluster, pooled within-cluster, and cross-cluster figures, with no
cluster-line overlay. Actual limits and figure paths are saved in the metadata.

## Output and provenance

```text
05_fsle/
|-- fsle_metadata.json
`-- array_NNN/
    |-- data/first_passages.parquet
    |-- tables/pair_selection.csv
    |-- tables/fsle_spectra.csv
    |-- tables/power_law_fits.csv
    `-- figures/
        |-- within_cluster_pooled.png
        |-- cross_cluster.png
        |-- ...__compensated_<power>.png
        `-- clusters/<cluster>[_...].png
```

The pair catalogue retains selected and excluded candidates and their reasons.
The passage table records policy, shell boundaries, identities, entry/exit/censor
times, observed distances, growth, duration, missing-sample counts, and rejection
reasons. The module metadata records resolved settings, exact thresholds, selected
product, cadence, interval, population counts, and figure limits. The unified
manifest hashes every published artifact and checks source metadata before atomic
publication. An unsuccessful overwrite preserves the previous output.
