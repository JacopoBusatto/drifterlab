# Methodology and numerical product definitions

## 1. Scope

The baseline ARCTERX analysis calculates signed two-point velocity statistics
separately inside each accepted deployment cluster. The optional Poje-comparison
mode forms the complete simultaneous pair matrix within each selected array,
including pairs connecting different accepted clusters. Arrays always remain
independent and no cross-array pair is formed.

The pair scopes are `within_cluster` and `all_array`. The latter requires the
explicit array-anchor method
`first_time_all_selected_platforms_reference_valid` and
`minimum_active_members: all`. Its consolidated population identifier is
`array_NNN__all_array`; original endpoint cluster IDs and the within/between
cluster relation remain in the pair and coverage records.

The workflow reads `trajectories.zarr` rather than `pairs.zarr`. Consequently,
pair membership is not conditioned on an encounter distance, initial separation,
or increasing separation.

## 2. Input and coordinate representation

The input must be a consolidated reconstruction schema-1.4 Zarr with dimensions
`(platform, time)` and these fields:

- `platform_id`, `array_id`, `cluster_id`, `cluster_size`, and
  `cluster_assignment_status`;
- `start_time` and the common UTC `time` grid;
- `longitude_<method>` and `latitude_<method>`;
- `source_gap_minutes` from accepted position-QC fixes.

Candidate inputs are rejected unless `allow_candidate_input: true` is explicit.
The manifest retains the candidate/frozen status, reconstruction schema and
algorithm versions, consolidated metadata hash, build-report hash, and cluster
assignment hash.

For each array, the first finite position of every selected platform is used to
calculate an equal-weight spherical-mean origin. All positions are transformed
to one fixed WGS84 azimuthal-equidistant coordinate reference system centered at
that origin. The same projection is used for the reference activation method and
every compared reconstruction. Positions are represented as `(x,y)` in metres.

The azimuthal-equidistant projection and origin are recorded in the manifest.
Pair separations used by the estimator are Euclidean distances in this declared
metric representation. Geographic endpoint and midpoint coordinates are retained
for traceability.

## 3. Velocity calculation and validity

### 3.1 Baseline derivative

On the five-minute common grid, the baseline velocity is the centered,
two-endpoint finite difference with a total span of 30 minutes:

$$
\mathbf u(t)=
\frac{\mathbf x(t+15\,\mathrm{min})-
      \mathbf x(t-15\,\mathrm{min})}
     {1800\,\mathrm{s}}.
$$

Only the two endpoints enter the derivative numerator. This is not a
seven-point higher-order derivative.

The inclusive stencil contains seven five-minute positions:

$$
t-15,\ t-10,\ t-5,\ t,\ t+5,\ t+10,\ t+15\ \mathrm{min}.
$$

Every one of those seven positions must have finite projected `x` and `y` and a
finite `source_gap_minutes` no larger than the configured support threshold. A
single unsupported intervening point invalidates the velocity even if the two
derivative endpoints are finite. No one-sided estimate is substituted at a
trajectory boundary or gap.

This policy is controlled independently for the reference and analysis methods
by `require_full_stencil_support`. With `true`, the seven-position rule above is
enforced. With `false`, reconstructed intervening positions and their source-gap
values are trusted: only the two derivative endpoints must be finite, and
`maximum_source_gap_minutes` is recorded but not applied. Centered derivatives
still require both endpoints, so the flag never permits extrapolation or a
one-sided estimate. For a sensitivity that must retain baseline activation,
leave the reference flag `true` and change only the analysis flag to `false`.

The stored spline products contain sampled positions rather than spline
coefficients. Velocities calculated from `spline_15`, `spline_30`, or
`spline_60` are therefore finite differences of sampled spline positions, not
analytic spline derivatives.

### 3.2 Analysis sampling cadence

`sample_cadence_minutes` is independent of the centered derivative span. The
workflow first calculates vector velocity on the native common grid and applies
the complete validity stencil. It then retains timestamps separated by the
configured cadence and aligned to the fixed population anchor. For example, the
Poje-comparison case evaluates the 30-minute centered derivative every 15
minutes. It does not average scalar speed or interpolate a second time. The
configured cadence must be an integer multiple of the native grid interval.

### 3.3 Optional speed mask

For an explicitly enabled threshold $s_{\max}$, a platform-time velocity is
masked when

$$
\sqrt{u^2+v^2}>s_{\max}.
$$

The proposed $1\,\mathrm{m\,s^{-1}}$ case is a sensitivity, not upstream QC.
Source trajectories are not changed, and large pair contributions are not
trimmed merely because their magnitudes are large.

## 4. Reference activation anchors

Let $M_c(t)$ be the number of assigned members of cluster $c$ with valid
reference positions and velocities at time $t$. For an integer activation
threshold $k\ge2$,

$$
t_{0,c}=\min\{t:M_c(t)\ge k\}.
$$

For `minimum_active_members: all`, $k$ is the assigned cluster size. A singleton
cannot activate for internal-pair analysis. The initial default is $k=2$.

The reference anchor uses the explicitly configured baseline reconstruction,
derivative span, and source-support threshold, with no optional speed mask. It is
not recomputed for reconstruction, speed-mask, or source-gap sensitivities.
Those analyses are evaluated at the saved reference time. If a sensitivity has
fewer than two usable members at the anchor, its snapshot is unavailable; if it
loses only some reference members, the snapshot is marked partially supported.
It is never moved to a later time.

Changing `minimum_active_members` defines a different activation experiment and
therefore intentionally produces a different anchor identity.

For `all_array`, the same validity calculation defines one array anchor: the
first common-grid timestamp at which every selected platform is reference-valid.
The resulting clock is common to within- and between-cluster pairs. A pair is
included only at simultaneous sampled UTC times when both endpoints are
analysis-valid.

The cluster clock is

$$
\tau_c=t-t_{0,c}.
$$

There is no pair-specific restart. A late pair contributes only when both members
become usable, while its coverage denominator still begins at the requested
cluster-window boundary. Wall-clock age continues through gaps.

## 5. Numerical products

### 5.1 Cluster inventory and activation anchors

The per-array `clusters.csv` retains:

- assigned members and possible internal pairs;
- assignment status and platform IDs;
- earliest and latest first-recorded position times;
- reference activation UTC and delay from the earliest recorded position;
- reference and analysis active members at the fixed anchor;
- snapshot availability/partial-support status.

### 5.2 Initial snapshot

At the single UTC instant $t_{0,c}$, every valid simultaneous internal pair is
calculated using the selected analysis reconstruction and validity settings.
Individual platform first fixes from different times are never combined.

Snapshot and temporal summaries remain separate through `analysis_type`, while
their raw pair-time rows are normalized into one table and marked with
`is_snapshot`. The snapshot contains no time series, so a temporal block
bootstrap is inapplicable. Its binned estimates remain descriptive. A bin with one pair is labelled
`descriptive_snapshot_single_pair`; other sparse bins are labelled as low support.
No confidence interval is reported in Stage 2.

### 5.3 Initial-evolution windows

Each configured interval is left-closed and right-open:

$$
W_c(a,b)=\{t:t_{0,c}+a\le t<t_{0,c}+b\}.
$$

The baseline windows are $[0,6)$ and $[0,12)$ hours. Arbitrary cluster-relative
intervals such as $[6,12)$ can be configured without changing the data model.
Clusters retain their own UTC anchors, so these products compare cluster age and
are not simultaneous environmental snapshots.

In `all_array` mode the configured intervals instead use the common array anchor.
The Poje-comparison configuration uses $[0,288)$ hours, or 12 days, sampled every
15 minutes. The retained `cluster_age_hours` compatibility field is therefore
elapsed array-anchor age for the synthetic `array_NNN__all_array` population.

Evolution pair observations are stored once through the union span of the
configured windows. Nested windows reuse those rows rather than duplicating raw
observations.

### 5.4 Active-member history and exclusions

`data/platform_diagnostics.parquet` records each platform's reference and
analysis validity, invalidity reason, velocities, and cluster active-member
counts at every retained grid time. A decline below the activation threshold is
reported but does not reset $t_{0,c}$.

Every scheduled pair-time opportunity receives one exclusive primary status:

1. valid;
2. nonfinite position in the centered stencil;
3. missing source support in the stencil;
4. source gap exceeding the configured threshold;
5. optional speed mask exceeded;
6. nonfinite derived velocity;
7. unavailable centered stencil;
8. zero or invalid separation.

Counts and fractions are saved as `coverage_level = exclusion` rows in
`coverage.csv`.

## 6. Pair kinematics

Platform IDs are sorted before a stable unordered pair ID is hashed. Both member
IDs remain in every row.

For simultaneous positions and velocities,

$$
\mathbf r_{ij}=\mathbf x_j-\mathbf x_i,
\qquad
\delta_{ij}=|\mathbf r_{ij}|,
$$

$$
\Delta\mathbf u_{ij}=\mathbf u_j-\mathbf u_i,
\qquad
\widehat{\mathbf r}_{ij}=\frac{\mathbf r_{ij}}{\delta_{ij}}.
$$

The longitudinal increment is

$$
\Delta u_L=\Delta\mathbf u\mathbin{\cdot}\widehat{\mathbf r}.
$$

The counter-clockwise transverse unit vector and increment are

$$
\widehat{\mathbf t}=(-\widehat r_y,\widehat r_x),
\qquad
\Delta u_T=\Delta\mathbf u\mathbin{\cdot}\widehat{\mathbf t}.
$$

Reversing the member labels reverses both the separation and increment vectors,
leaving $\Delta u_L$, $\Delta u_T$, and $q$ invariant. Zero or nonfinite
separation is invalid.

The observation table retains endpoint positions and velocities, speeds,
source-gap values, metric and geographic midpoints, separation, both increments,
the signed cube, the mixed-third-order contribution $\Delta u_L(\Delta u_T)^2$,
and cluster age.

## 7. Separation conditioning

Samples are assigned by instantaneous separation, so a pair may enter different
bins or revisit a bin. Initial separation and whole-window mean separation do not
control membership.

Configured bin forms are:

- explicit physical edges;
- geometric edges with a ratio and exact anchor;
- linear edges with an exactly dividing physical width.

Bins are left-closed and right-open, except that the final physical edge is
included in the final bin. Scatter plots do not require bins. Axis scaling is a
separate display choice and never changes physical membership.

The Stage 2 pilot uses provisional geometric edges from 62.5 m through 128 km,
with ratio $\sqrt{2}$ and an exact 1 km anchor. These are computational bins, not
a validated MicroSVP resolution limit. Underflow, overflow, and invalid counts
are saved explicitly. No estimate is interpolated across an empty bin.

`minimum_reliable_separation_m` remains null until positional accuracy is known.
If later configured, raw rows below it should remain traceable but must be
excluded from scientific summaries and coverage under the documented validity
policy.

## 8. Structure functions and descriptive moments

For a selected separation bin and snapshot/window, raw longitudinal structure
functions are

$$
S_L^{(n)}=\left\langle(\Delta u_L)^n\right\rangle,
$$

with analogous transverse moments. Stage 2 retains $n=2,3,4$. Raw normalized
ratios are

$$
R_{L,3}=\frac{S_L^{(3)}}{[S_L^{(2)}]^{3/2}},
\qquad
R_{L,4}=\frac{S_L^{(4)}}{[S_L^{(2)}]^2}.
$$

These are not silently called central skewness or flatness. Descriptive central
moments use

$$
\mu_n=\left\langle(\Delta u_L-\langle\Delta u_L\rangle)^n\right\rangle,
$$

$$
\mathrm{skewness}=\frac{\mu_3}{\mu_2^{3/2}},
\qquad
\mathrm{flatness}=\frac{\mu_4}{\mu_2^2}.
$$

Flatness is not excess kurtosis; a Gaussian has flatness 3. Raw and central
quantities are saved with distinct column names. The same convention is applied
to transverse increments and unconditioned descriptive summaries.

## 9. Signed q and exact epsilon estimator

The declared coefficient is the three-dimensional four-fifths-law convention.
For each valid pair-time sample,

$$
q_{ij}(t)=-\frac54
\frac{[\Delta u_{L,ij}(t)]^3}{\delta_{ij}(t)}.
$$

For separation bin $b$, cluster $c$, and product $W$,

$$
S_{LLL}(b,c,W)=
\left\langle(\Delta u_L)^3\right\rangle_{b,c,W},
$$

$$
\widehat\varepsilon_{\mathrm{eff}}(b,c,W)
=-\frac54\frac{S_{LLL}(b,c,W)}{\langle\delta\rangle_{b,c,W}}.
$$

Each pair-time observation has equal weight in the underlying third moment. The
horizontal coordinate is the observed mean separation, while physical bin edges
and nominal centers remain available.

For finite-width bins, the estimator is exactly

$$
\widehat\varepsilon_{\mathrm{eff}}
=\frac{\sum\delta q}{\sum\delta},
$$

not the arithmetic mean $\langle q\rangle$. Both values are saved separately,
and their weighted-identity absolute error is recorded and tested.

Units are metres, metres per second, and square metres per cubic second
(`m2 s-3`) for $q$ and $\widehat\varepsilon_{\mathrm{eff}}$.
Positive epsilon is a diagnostic consistent with downscale transfer and negative
epsilon with upscale transfer, subject to the theoretical and sampling
assumptions. It is not a direct local viscous-dissipation measurement. An
individual instantaneous sign is only a pair contribution.

## 10. Rossby-number diagnostic

The Poje-style longitudinal diagnostic in a bin is

$$
Ro_L(b)=\frac{\sqrt{S_L^{(2)}(b)}}{|f_b|\langle\delta\rangle_b},
$$

where

$$
f_b=2\Omega\sin\left(\langle\phi_{\mathrm{midpoint}}\rangle_b\right),
\qquad
\Omega=7.292115\times10^{-5}\ \mathrm{s^{-1}}.
$$

This ratio-of-bin-statistics matches the paper's scale-dependent definition more
closely than averaging pointwise $\Delta u_L/(f\delta)$, which would overweight
small separations.

## 11. Coverage

For a cluster with $N$ assigned members and a requested window containing $K$
scheduled grid timestamps, the full-window denominator is

$$
N_{\mathrm{expected}}={N\choose2}K.
$$

It includes every assigned unordered pair, even when a member becomes usable
late. Aggregate full-window coverage is

$$
C_{\mathrm{cluster}}=
\frac{N_{\mathrm{valid\ pair-times}}}{N_{\mathrm{expected}}}.
$$

For pair $p$,

$$
C_{p,\mathrm{full}}=\frac{K_{p,\mathrm{valid}}}{K}.
$$

The pair's availability interval is the inclusive scheduled span from its first
to last valid observation inside the requested window. Conditional availability
coverage is separately named:

$$
C_{p,\mathrm{availability}}=
\frac{K_{p,\mathrm{valid}}}{K_{p,\mathrm{first:last}}}.
$$

Counts before first validity, internal gaps, and counts after last validity are
saved separately. Equal requested duration therefore does not imply equal sample
counts and no observation is discarded merely to equalize clusters.

## 12. Per-bin support and influence

Temporal support is evaluated independently in each separation bin, not inferred
from whole-window coverage. Each bin saves:

- pair-time observations;
- unique unordered pairs and contributing platforms;
- distinct UTC timestamps and their fraction of scheduled window times;
- occupied one-hour UTC blocks;
- UTC span and minimum/maximum cluster age.

The initial configurable descriptive threshold is 12 observations, 3 unique
pairs, 3 platforms, and 12 distinct UTC timestamps. Point estimates remain in
low-support rows, but `support_ok` is false and the missing support dimensions are
named. Empty bins remain explicit missing rows. Stage 2 does not calculate a
confidence interval.

Third moments can be dominated by rare observations. Every bin therefore saves:

- the largest and top-five fractions of total absolute cubic contribution;
- an effective absolute-cube observation count;
- the cancellation ratio $|\sum(\Delta u_L)^3|/\sum|(\Delta u_L)^3|$;
- the pair/time of the largest absolute contribution;
- epsilon after removing that one observation and the resulting change.

The pair and UTC of the leading contribution remain in the scale-conditioned
row, so the corresponding raw observation can be selected on demand from the
array-level pair table. These are diagnostics, not trimming rules. Stage 3 will
add leave-one-platform-out checks as a distinct population-influence analysis.

### 12.1 Stage 3 uncertainty and figures

For every cluster-relative temporal window, Stage 3 partitions the already
selected observations into equal UTC blocks aligned to the window start. It
resamples scheduled block identifiers with replacement while keeping all pair
observations in a selected block together. Empty scheduled blocks remain in the
resampling population. Activation, cluster age, window membership, and pair
eligibility are never recomputed on a synthetic timeline.

For each replicate and separation bin the exact estimator is recalculated from
the resampled sums,

$$
\widehat\varepsilon^*_{\mathrm{eff}}
=-\frac54\frac{\sum w_k(\Delta u_{L,k})^3}{\sum w_k\delta_k},
$$

where $w_k$ is the multiplicity of the UTC block containing observation $k$.
Percentile intervals are reported only when descriptive support, contributing
block count, and successful-replicate fraction pass their configured thresholds.
The default 30-minute duration is provisional and is compared with 60 minutes.
Intervals are conditional on the observed drifter population. Snapshots receive
no temporal interval.

Leave-one-platform-out calculations remove every pair containing one platform
and recompute epsilon in the unchanged separation bin. They are influence
diagnostics, not confidence intervals or trimming rules. Detailed results are
stored separately while maximum changes and sign-reversal flags are exposed in
`epsilon_by_scale.csv`.

The conditional signed-q heatmap uses common physical symmetric-log q edges.
Each nonempty separation column is normalized to probability mass one. Because
the q bins have unequal widths, color is probability mass per bin rather than a
probability density. Missing columns remain distinct from measured zero counts.
See [the figure guide](figure_guide.md) for every panel and comparison product.

## 13. Output organization

One scientific identity selects one analysis reconstruction and derivative
configuration:

```text
<output_root>/
  <position_method>/
    centered_difference_<span>min_on_5min_grid/
      epsilon-v1-<hash>/
        config_resolved.yaml
        manifest.json
        progress.md
        array_<id>/
          clusters.csv
          epsilon_by_scale.csv
          statistics.csv
          coverage.csv
          data/
            pair_observations.parquet
            platform_diagnostics.parquet
            q_histograms.parquet
            bootstrap_sensitivity.parquet
            leave_one_platform_out.parquet
          figures/
            README.md
            comparisons/
            <cluster_id>/
```

`clusters.csv` combines inventory, membership, reference activation, analysis
availability, and snapshot counts; its common identifiers use
`analysis_type = inventory` and `window_id = all`. `epsilon_by_scale.csv`
combines snapshot and window structure functions, signed epsilon, Rossby number,
bin support, and influence metrics. Its common identity is `array_id`, `cluster_id`,
`analysis_type`, `window_id`, and `separation_bin_id`. Snapshot and window rows
retain separate UTC fields rather than treating a snapshot as a zero-duration
temporal window.

`statistics.csv` contains labelled pair- and platform-population distributions;
`population_level` and `metric` distinguish the source and quantity.
`coverage.csv` combines cluster, pair, exclusion, and separation-range records,
distinguished by `coverage_level`. Pair rows retain `pair_id`, while exclusion
rows retain an exclusive `exclusion_reason`.

`pair_observations.parquet` contains at most one row for each array, cluster,
pair, and UTC. A row at reference activation has `is_snapshot: true`; it is not
duplicated merely because it also belongs to a configured window. Nested-window
membership is derived from `cluster_age_hours`. The platform table begins at the
cluster's earliest recorded position and extends through the end of its longest
requested window relative to activation. For a cluster that cannot activate, the
same duration begins at its earliest recorded position. It preserves positions,
source gaps, reference and analysis velocities, both validity states and reasons,
active-member counts, and snapshot flags. This includes the pre-activation grid
needed to inspect an activated cluster's anchor without storing irrelevant
campaign-wide times. Information unavailable from valid pair rows therefore
remains auditable. The `figures` directory is empty when Stage 3 is disabled.
When enabled, the additional Parquet datasets retain histogram, block-duration,
and platform-influence details without adding more routine inspection CSVs.

The deterministic run hash includes the resolved scientific configuration,
reference activation rules, selections, windows, bins, masks, schema/algorithm
versions, and source metadata/build-report hashes. The output root and processing
timestamp do not change scientific identity. Publication is atomic and refuses a
silent overwrite. Output schema 2.0 introduced this consolidated layout without
changing the estimator algorithm.

## 14. Relationship to Poje et al. (2017)

Stage 2 implements or prepares these paper-related diagnostics:

| Diagnostic | Treatment |
|---|---|
| Velocity and speed distributions | Saved per cluster/window |
| Separation and longitudinal/transverse increment distributions | Saved descriptively and as pair observations |
| Second-order longitudinal/transverse structure functions | Saved per separation bin |
| Signed third-order longitudinal function | Saved per separation bin |
| Fourth-order moments and normalized ratios | Saved with raw/central labels |
| Four-fifths epsilon diagnostic | Saved with exact finite-bin identity |
| Scale-dependent Rossby number | Saved using local midpoint latitude |
| Mixed longitudinal-transverse third order | Contribution retained; aggregate analysis deferred |
| Relative dispersion and FSLE | Existing workflows remain separate; not rebuilt here |
| Temporal Lagrangian structure and separation memory | Requires a separate time-lag definition; deferred |
| Rotational/divergent compressibility decomposition | Requires transverse scale derivatives/integrals and a defensible lower boundary; deferred |

The ARCTERX cluster-age analysis is not a reproduction of the GLAD experiment.
Poje et al. used different instruments, processing, durations, and sampling
scales. Homogeneity, isotropy, stationarity, incompressibility, surface
convergence, and nonuniform Lagrangian sampling remain limitations.
