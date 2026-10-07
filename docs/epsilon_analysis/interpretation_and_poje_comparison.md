# How to understand and explain the ARCTERX epsilon analysis

This document is the single narrative guide to the ARCTERX signed longitudinal
analysis. It explains what enters the calculation, how to read the figures, what
Poje et al. (2017) calculated, and where the ARCTERX implementation deliberately
differs. Exact implementation details remain in the [methodology](methodology.md),
and every visual encoding is catalogued in the [figure guide](figure_guide.md).

## 1. The question being asked

Two drifters separated by a distance $\delta$ generally have different
velocities. The component of that velocity difference along the line joining the
drifters is $\Delta u_L$. Its signed third moment is useful because, under the
assumptions of homogeneous, isotropic, stationary three-dimensional turbulence,
Kolmogorov's four-fifths law relates it to the downscale energy flux:

$$
S_{LLL}(r)=\left\langle(\Delta u_L)^3\right\rangle_r
=-\frac45\varepsilon r.
$$

A negative third-order moment therefore produces a positive $\varepsilon$ under
this convention. Positive $\varepsilon$ is consistent with downscale transfer;
negative $\varepsilon$ is consistent with upscale transfer. In a real surface
ocean dataset, this is a signed **effective-transfer diagnostic**, not a direct
local measurement of viscous dissipation and not proof of a cascade by itself.

## 2. From trajectories to one plotted estimate

### 2.1 Reconstructed positions and velocity

Accepted native fixes are placed on the common five-minute grid. Exact accepted
fixes have `source_gap_minutes = 0`; interpolated positions carry the separation
in minutes between their two supporting accepted fixes.

For every platform, the baseline velocity is

$$
\mathbf u(t)=
\frac{\mathbf x(t+15\,\mathrm{min})-
      \mathbf x(t-15\,\mathrm{min})}{1800\,\mathrm{s}}.
$$

Only the two endpoints enter the numerator. With
`require_full_stencil_support: true`, all seven five-minute positions from
$t-15$ through $t+15$ minutes must be finite and must satisfy the configured
source-gap threshold. This is a two-endpoint centered derivative with a
seven-position validity rule, not a seven-point high-order derivative.

### 2.2 Cluster activation and time products

Each cluster has its own activation time $t_{0,c}$: the first common-grid time at
which the configured minimum number of members has valid reference positions and
velocities. The baseline uses `minimum_active_members: 2`.

Two kinds of products are kept separate:

- **Snapshot:** simultaneous pairs exactly at $t_{0,c}$; this is cluster time
  zero and has no temporal confidence interval.
- **Window:** all valid simultaneous pairs in a cluster-age interval such as
  $[0,6)$ or $[0,12)$ hours after $t_{0,c}$.

Different clusters activate at different UTC times. Their windows represent the
same requested cluster age, not simultaneous environmental sampling.

### 2.3 Pair geometry and increments

For platforms $i$ and $j$ at the same UTC time,

$$
\mathbf r_{ij}=\mathbf x_j-\mathbf x_i,
\qquad
\delta_{ij}=|\mathbf r_{ij}|,
$$

$$
\Delta\mathbf u_{ij}=\mathbf u_j-\mathbf u_i,
\qquad
\Delta u_L=\Delta\mathbf u_{ij}\cdot
\frac{\mathbf r_{ij}}{\delta_{ij}}.
$$

The transverse increment is retained using the counter-clockwise unit vector
$(-\widehat r_y,\widehat r_x)$. Reversing the pair labels reverses both vectors,
so $\Delta u_L$, $\Delta u_T$, and the derived q are unchanged.

Each pair-time row receives the signed contribution

$$
q_{ij}(t)=-\frac54
\frac{[\Delta u_{L,ij}(t)]^3}{\delta_{ij}(t)}.
$$

The cube is deliberately signed. Large positive and negative contributions may
cancel, and rare increments can dominate the result.

### 2.4 Separation bins and epsilon

Observations are grouped using their instantaneous separation. A pair may move
between bins or revisit a bin. The baseline uses fixed geometric edges with
ratio $\sqrt2$ from 62.5 m to 128 km; those are provisional computational edges,
not a validated MicroSVP resolution limit.

For one cluster, product, and separation bin,

$$
S_{LLL}=\left\langle(\Delta u_L)^3\right\rangle,
$$

$$
\widehat\varepsilon_{\mathrm{eff}}
=-\frac54\frac{S_{LLL}}{\langle\delta\rangle}.
$$

Every valid pair-time row has equal weight in $S_{LLL}$. The horizontal plotting
coordinate is the observed mean separation $\langle\delta\rangle$, while the
physical bin edges and nominal center are also saved.

For a finite-width bin this estimator obeys the exact identity

$$
\widehat\varepsilon_{\mathrm{eff}}
=\frac{\sum\delta q}{\sum\delta}.
$$

It is therefore a separation-weighted mean of the individual q values, not the
arithmetic mean $\langle q\rangle$. The arithmetic mean is saved separately as a
distribution summary.

## 3. How to read the four-panel figure

### Panel A — Pair contributions

Every point is one valid pair at one UTC time: instantaneous q versus
instantaneous separation. It shows the raw signed distribution and reveals
tails, asymmetry, and isolated influential observations. One point's sign does
not identify a cascade. Display subsampling, when required, never changes the
calculation.

### Panel B — Scale-conditioned epsilon

Every marker is the estimate obtained after pooling the observations in one
separation bin. Filled markers pass the descriptive support criteria; open grey
markers are retained but have low or snapshot-only support. Missing bins are not
connected or interpolated.

For temporal windows, error bars are percentile intervals from synchronized UTC
block resampling. All pair observations in a time block are resampled together.
The interval is suppressed if too few blocks contribute. It is conditional on
the observed drifter population and is not the spread of q. Snapshot markers
never receive temporal intervals.

The most important visual checks are:

1. Is the estimate consistently on one side of zero across adjacent supported
   scales?
2. Do its confidence intervals cross zero?
3. Is the curve stable between the 30- and 60-minute block calculations?
4. Is it dominated by one observation or platform?
5. Do the clusters being compared actually overlap in separation?

### Panel C — Conditional q heatmap

Each vertical column is the q histogram within one separation bin, normalized to
probability mass one. It answers: “Given this separation range, how are the
signed pair contributions distributed?” Color is probability mass per q bin,
not raw count or density. The q bins have unequal symmetric-log widths. Grey
columns are missing data, whereas a colored zero is a measured zero count.

The heatmap can reveal cancellation: a bin may contain large contributions on
both sides of zero but have a small mean third moment. Such a pattern does not
make the small resulting epsilon intrinsically robust.

### Panel D — Sampling support

This panel gives pair-time observations, distinct UTC times, distinct pair IDs,
and distinct platform IDs in every separation bin. These are not expected to be
equal. A pair is an edge connecting two platform endpoints: one pair therefore
contains two platforms, and two disjoint pairs contain four. Platform count can
thus exceed pair count when only a sparse subset of all possible cluster pairs
occupies that separation bin.

The activation threshold applies only when selecting the fixed cluster time
origin. For example, four simultaneously usable members have six possible
pairs, but those pairs can lie in several different separation bins. The
threshold does not require six pairs in every bin and does not prevent a fifth
assigned member from contributing later. Support must be judged at the scale of
the estimate. High coverage over the complete window cannot repair a bin
occupied by only one pair or a short interval.

### A short verbal explanation

> Panel A shows all instantaneous signed pair contributions. Panel C reorganizes
> the same information as a separation-conditioned distribution. Panel B first
> averages the signed longitudinal cubes within each separation bin and then
> applies the four-fifths coefficient; its error bars resample complete time
> blocks. Panel D shows whether each scale has enough independent temporal and
> platform support to take that estimate seriously.

## 4. What Poje et al. (2017) did

Poje et al. analyzed 86 GLAD surface drifters from a near-simultaneous release.
The GPS fixes were nominally available every five minutes with reported 5–10 m
position accuracy. Their processing used acceleration QC and non-causal spline
interpolation to produce trajectory and velocity information at uniform
15-minute intervals. They considered all pair separations for up to 12 days.

At each time they calculated the complete pair-separation matrix and the
longitudinal and transverse velocity increments. They obtained statistics by
time-averaging the increment distributions conditioned on binned instantaneous
separation. Their longitudinal structure functions were

$$
S_L^{(n)}(r)=\langle(\Delta u_L)^n\rangle_r.
$$

Their primary energy-flux estimate used the three-dimensional four-fifths law:

$$
\varepsilon_{4/5}(r)
=-\frac54\frac{S_L^{(3)}(r)}{r}.
$$

Figure 7b showed this as the sign-reversed third-order longitudinal structure
function scaled by separation. They found a roughly scale-independent positive
value of order $10^{-7}\ \mathrm{m^2\,s^{-3}}$ over approximately 100 m–10 km,
with a sign change at larger scales. They compared the magnitude with nearby
microstructure measurements, while explicitly noting that surface drifters
sample a two-dimensional, divergent horizontal velocity field.

They also evaluated the mixed longitudinal-transverse relation

$$
\langle(\Delta u_L)^3\rangle
+\langle\Delta u_L(\Delta u_T)^2\rangle
=-2\varepsilon r,
$$

which supplies a second epsilon estimate. Their paper additionally compared
second-order scaling with K41 predictions, calculated a scale-dependent Rossby
number, analyzed dispersion, and compared the observations with synthetic
drifters advected by geostrophic AVISO velocities.

Reference: A. C. Poje, T. M. Özgökmen, D. J. Bogucki, and A. D. Kirwan Jr.,
“Evidence of a forward energy cascade and Kolmogorov self-similarity in
submesoscale ocean surface drifter observations,” *Physics of Fluids* 29,
020701 (2017), [doi:10.1063/1.4974331](https://doi.org/10.1063/1.4974331).

## 5. Poje and ARCTERX: exact comparison

| Choice | Poje et al. | Current ARCTERX analysis |
|---|---|---|
| Dataset | 86 GLAD drifters from one near-simultaneous S1 deployment | MicroSVP drifters separated by accepted deployment array and cluster |
| Native sampling and reconstruction | Five-minute GPS fixes; acceleration QC and non-causal spline processing to uniform 15-minute trajectory/velocity data | Accepted QC fixes reconstructed on a common five-minute grid; configurable linear or stored spline position products |
| Velocity | Velocity information from their spline-processing workflow | Explicit centered difference $[x(t+15)-x(t-15)]/1800$ s |
| Pair population | Complete pair matrix for the selected release | Baseline: pairs within each cluster. Poje test: complete simultaneous pair matrix within each array, including cross-cluster pairs; never cross-array |
| Time selection | Time averages over an evolving dataset extending to 12 days | Baseline: cluster-relative short windows. Poje test: a common array-relative $[0,288)$ hour window sampled every 15 minutes |
| Clock | Common release analysis | Baseline: separate cluster anchors. Poje test: first timestamp when every selected platform is reference-valid |
| Conditioning | Instantaneous binned separation | Instantaneous binned separation |
| Longitudinal increment | Projection of simultaneous velocity difference along pair separation | Same definition and label-invariant geometry |
| Primary third-order law | $-(5/4)S_{LLL}(r)/r$ | $-(5/4)S_{LLL}/\langle\delta\rangle$ using observed mean separation in each finite bin |
| Individual q | Not their primary estimator | Saved for distributions and influence; epsilon equals $\sum\delta q/\sum\delta$, not $\langle q\rangle$ |
| Mixed third order | Aggregated and plotted as an alternate epsilon estimate | Per-observation $\Delta u_L(\Delta u_T)^2$ is saved, but its aggregate epsilon curve is not yet implemented |
| Uncertainty | Bootstrap intervals reported for dispersion; Figure 7 emphasizes scale-conditioned estimates | Dependence-aware synchronized UTC blocks; the Poje test uses a 24-hour primary block with 12- and 48-hour sensitivities |
| Influence | Sampling limitations discussed | Largest cubic contribution and leave-one-platform-out estimates saved per bin |
| Rossby number | Used fixed $f\approx7\times10^{-5}\ \mathrm{s^{-1}}$ | Uses local $f=2\Omega\sin(\langle\phi\rangle)$ in every bin |
| Interpretation | Evidence assessed jointly across structure functions, dispersion, Rossby number, AVISO comparison, and microstructure | Current result is an exploratory signed transfer diagnostic; reconstruction and mask comparisons remain Stage 4 |

The ARCTERX estimate is thus Poje-style but not a reproduction of the GLAD
experiment. Different deployment geometry, duration, reconstruction, velocity
method, clustering, spatial range, and environmental setting prevent direct
numerical equivalence.

## 6. What is already comparable and what remains

Already implemented:

- longitudinal and transverse velocity increments;
- second-, third-, and fourth-order raw structure functions;
- the signed four-fifths epsilon diagnostic;
- raw versus central skewness/flatness distinctions;
- scale-dependent Rossby number;
- conditional signed-q distributions;
- temporal support, block-bootstrap uncertainty, and platform influence.

Useful additions for a closer Poje comparison would be:

1. Aggregate and plot the mixed longitudinal-transverse epsilon estimator.
2. Add an explicit second-order scaling figure and fitted scale-range slope.
3. Run the approved linear-versus-spline comparison at fixed activation anchors.
4. Inspect the separately labelled 12-day all-array result and compare it with
   the preserved within-cluster products.

These additions should not replace the current cluster-relative products or
silently pool asynchronous cluster snapshots.

## 7. Where to find the evidence behind a figure

- `epsilon_by_scale.csv`: point estimates, support, primary confidence intervals,
  exact identity error, Rossby number, and summary influence flags.
- `data/pair_observations.parquet`: every plotted pair-time contribution,
  including positions, increments, q, mixed third-order contribution, UTC, and
  cluster age.
- `data/q_histograms.parquet`: numerical heatmap counts and conditional masses.
- `data/bootstrap_sensitivity.parquet`: both 30- and 60-minute interval results.
- `data/leave_one_platform_out.parquet`: every platform-removal recalculation.
- `coverage.csv`: gaps, exclusions, scheduled denominators, and incomplete
  windows.
- `clusters.csv`: membership, activation timestamp, and snapshot availability.

The first Stage 3 Array 4 result and its limitations are summarized in the
[Stage 3 pilot report](stage3_array4_pilot.md).
