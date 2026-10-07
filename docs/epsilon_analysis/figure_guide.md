# Epsilon-analysis figure guide

Stage 3 writes review figures under each array's `figures/` directory. Figures
visualize the saved numerical products; they do not apply an extra scientific
filter. Complete observations, histogram counts, bootstrap results, and
platform-influence estimates remain available under `data/`.

## Derived trajectory-window diagnostic

The separate `drifterlab-epsilon-diagnostics` renderer reads a completed epsilon
run and does not recalculate its anchors, velocities, masks, pairs, or epsilon.
For every activated cluster, `trajectory_window/<cluster_id>.png` shows the
largest configured elapsed window in the same projected metric coordinates used
for pair separation.

- A coloured solid line is the trajectory where the selected analysis velocity
  is valid.
- The faint continuation is the reconstructed position path.
- An `x` is a finite reconstructed position whose analysis velocity is invalid.
- An open circle is the platform's first analysis-valid point in the window.
- A square marks an exactly available configured intermediate time, such as 6 h.
- A diamond is the final retained analysis-valid point before the half-open
  window boundary.
- The dashed black line is the mean position of analysis-valid members at each
  UTC timestamp.

Axes have equal metric scaling, so spreading, convergence, shear, loops, and
anisotropy are not visually distorted. A platform unavailable at activation can
begin later; its open circle must not be interpreted as a moved activation
anchor. `diagnostic_figures.csv` records figures that were produced and
inventory rows skipped because no reference activation existed.

## Derived velocity-and-availability diagnostic

For every activated cluster, `velocity_availability/<cluster_id>.png` uses the
same fixed activation and largest configured elapsed window as the trajectory
diagnostic. Its panels contain:

- **A:** analysis-valid eastward velocity $u$ for each platform;
- **B:** analysis-valid northward velocity $v$;
- **C:** scalar speed $|\mathbf u|$;
- **D:** differential speed relative to the instantaneous mean of currently
  analysis-valid members,
  $|\mathbf u_i-\overline{\mathbf u}|$;
- **E:** counts of finite reconstructed positions, reference-valid velocities,
  analysis-valid velocities, the activation threshold, and possible
  simultaneous analysis-valid pairs.

The black curves in panels A--C are valid-member means. Invalid velocity samples
are gaps, not interpolated connections. Panel D removes common translation but
is not a longitudinal pair increment; $\Delta u_L$ and $\Delta u_T$ belong to
the next pair-relative diagnostic. The possible-pair curve is $n(n-1)/2$ for
the analysis-valid member count and does not apply separation-bin membership or
minimum-separation filtering.

## Derived pair-relative-motion diagnostic

For every activated cluster with saved pair observations,
`pair_relative_motion/<cluster_id>.png` uses the same fixed activation and
largest configured elapsed window. It reads the saved pair geometry and
velocity increments; it does not reconstruct pairs or recalculate epsilon. Its
panels contain:

- **A:** separation $\delta$ for every unordered platform pair, on a logarithmic
  vertical axis;
- **B:** saved longitudinal relative velocity $\Delta u_L$;
- **C:** a numerical centered separation rate using the source analysis velocity
  span;
- **D:** saved transverse relative velocity $\Delta u_T$;
- **E:** counts of valid, separating, and approaching pairs at each scheduled
  time;
- **F:** positive, negative, and net sums of $(\Delta u_L)^3$ over all available
  separations at each time.

For the current 30-minute source velocity span, panel C uses

$$
\left.\frac{d\delta}{dt}\right|_t
\approx
\frac{\delta(t+15\,\mathrm{min})-\delta(t-15\,\mathrm{min})}
{1800\,\mathrm{s}}.
$$

This is a diagnostic comparison, not the definition used for panel B. The saved
longitudinal increment is calculated by projecting the simultaneous vector
velocity difference onto the pair-separation direction:

$$
\Delta u_L=(\mathbf u_2-\mathbf u_1)\mathbin{\boldsymbol\cdot}
\frac{\mathbf x_2-\mathbf x_1}{\delta}.
$$

Positive $\Delta u_L$ means that the pair is separating; negative $\Delta u_L$
means that it is approaching. Because the adopted four-fifths estimator has a
minus sign, positive cubic increments drive the epsilon estimate negative:

$$
\widehat\varepsilon_{\mathrm{eff}}
=-\frac54\frac{\sum (\Delta u_L)^3}{\sum\delta}.
$$

Panel F intentionally pools separations only to locate times at which one sign
dominates. It is not a scale-conditioned epsilon estimate and must not replace
`epsilon_by_scale.csv`. Gaps mean that no valid simultaneous saved pair
observation exists at that scheduled time. A mismatch between panels B and C
can reflect the finite-difference span, changing pair orientation, curved
motion, or numerical inconsistency and should be inspected rather than silently
corrected.

## Derived time--scale contribution diagnostic

`time_scale_contributions/<cluster_id>.png` decomposes the largest configured
temporal-window estimate without changing its separation bins or weighting. For
time $t$ and separation bin $b$, it defines

$$
c_{t,b}
=-\frac54
\frac{\sum_{i\in(t,b)}(\Delta u_{L,i})^3}
{\sum_{i\in b}\delta_i}.
$$

The denominator is the separation sum over the complete cluster window in bin
$b$, not only the observations at time $t$. Consequently, the cells are
additive:

$$
\sum_t c_{t,b}=\widehat\varepsilon_{\mathrm{eff},b}.
$$

The renderer verifies this identity against the saved `epsilon_by_scale.csv`
value for every finite bin and stops if the maximum absolute error exceeds
$10^{-12}\ \mathrm{m^2\,s^{-3}}$. This decomposition is therefore not a new
estimator.

The panels contain:

- **A:** additive epsilon contribution $c_{t,b}$; blue cells drive the final
  estimate negative and red cells drive it positive;
- **B:** valid pair-observation count in each time--scale cell;
- **C:** the largest single observation's fraction of the cell's total absolute
  cubic magnitude;
- **D:** the time sum of panel A overlaid on the saved epsilon-by-scale result.

Grey cells contain no valid pair observation. Panel C approaches one when a
single observation controls the cell and is not a statistical confidence
measure. The plotted separation range spans the occupied configured bins plus
one neighbouring bin where available.

The combined derived table `data/time_scale_contributions.parquet` contains one
row per occupied cluster, UTC time, and separation bin. In addition to the
additive contribution and support counts, it records positive and negative
cubic sums, cancellation, maximum influence fraction, the most influential
pair, its two endpoint platform IDs, its separation, and its longitudinal
increment. The table permits any colored cell to be traced to the immutable
source `pair_observations.parquet`.

## Derived negative-bin influence audit

`influence_audit/<cluster_id>.png` is produced when a cluster has at least one
negative estimate whose configured separation-bin upper edge is no greater than
`figures.influence_audit.small_scale_maximum_m`. This is a diagnostic selection;
it does not exclude the bin or redefine epsilon.

The panels contain:

- **A:** saved epsilon, epsilon after removing the observation with the largest
  absolute cubic contribution, and the range obtained by removing each platform
  in turn;
- **B:** the largest observation's fraction of total absolute cubic magnitude
  and the maximum relative change under platform removal;
- **C:** original observation, pair, platform, and UTC support;
- **D:** non-exclusive support and sensitivity flags.

The readable root table `influence_audit.csv` contains one row per audited bin.
It retains the baseline support and influence values, leave-one-observation
result, leave-one-platform range, sign-reversing platform IDs, continuous
influence metrics, Boolean flags, and a primary audit classification.

The default descriptive thresholds are:

$$
\frac{\max_i |(\Delta u_{L,i})^3|}
{\sum_i |(\Delta u_{L,i})^3|}\geq0.5
$$

for a concentrated observation contribution, and a maximum relative epsilon
change of at least 0.5 for platform sensitivity. These thresholds label results
for inspection only. Sign reversals are recorded independently of the
thresholds. The primary classification uses this order:

1. low original support;
2. observation and/or platform sign sensitivity;
3. concentrated observation and/or platform influence;
4. persistent negative under the evaluable removal tests;
5. incomplete influence test.

`persistent negative` means only that the sign remained negative in these
specific removal calculations. It is not evidence that the estimate is free of
sampling, deployment, or model-assumption effects. No observation or platform
is removed from the source analysis by this audit.

## Cluster diagnostic figures

Each activated cluster has one `snapshot.png` and one PNG for each configured
cluster-relative temporal window. All four panels use the same configured
separation limits. Separation is logarithmic; signed quantities use a symmetric
log scale so neither sign is discarded.

### A — Pair contributions

The scatter shows instantaneous

$$
q=-\frac54\frac{(\Delta u_L)^3}{\delta}
$$

against instantaneous separation. The horizontal zero line separates positive
and negative pair contributions. A point is one valid pair at one UTC time. Its
sign is not, by itself, evidence for a cascade direction.

If the available count exceeds `stage3.figures.scatter_maximum_points`, the
display uses a deterministic subsample. The title reports displayed and
available counts. All observations still enter the statistics, bootstrap, and
histogram. Signed-q values beyond the configured histogram range are counted and
reported in the figure title rather than silently clipped from the analysis.

### B — Four-fifths signed estimate

The ordinate is

$$
\widehat\varepsilon_{\mathrm{eff}}
=-\frac54\frac{\langle(\Delta u_L)^3\rangle}{\langle\delta\rangle}
=\frac{\sum\delta q}{\sum\delta}.
$$

The horizontal coordinate is the mean observed separation in the physical bin.
Filled red markers pass the descriptive support thresholds. Open grey markers
retain descriptive or low-support estimates. Lines are interrupted across
missing bins; they are not interpolated.

For temporal windows, vertical bars are percentile confidence intervals from
the primary synchronized UTC block bootstrap. A bin receives an interval only
when its descriptive support, contributing-block count, and successful-replicate
fraction all pass. These intervals measure uncertainty of the estimator
conditional on the observed drifter population. They are not the spread of
individual q values and do not account for every sampling bias.

Snapshots never receive temporal confidence intervals. Their figure subtitle
explicitly identifies them as descriptive.

### C — Conditional signed-q distribution

The heatmap is a saved two-dimensional histogram. Columns are separation bins
and rows are fixed physical q bins. Every nonempty separation column is
normalized separately:

$$
P(q_k\mid\delta_b)=\frac{N(q_k,\delta_b)}{N(\delta_b)}.
$$

Color therefore means **conditional probability mass per q bin**, not raw count
and not probability density per unit q. The q bins have unequal widths because
they use a symmetric-log construction. Grey columns are missing; a colored zero
is a measured zero count. Comparable figures share the configured color limit.

### D — Sampling support

The support panel shows four different counts in every separation bin:

- **Pair-time observations:** valid table rows in that bin. The same pair can
  contribute repeatedly at different UTC times.
- **Distinct UTC timestamps:** times at which at least one pair occupied the bin.
- **Distinct pair IDs:** unordered platform pairs that occupied the bin at least
  once during the snapshot/window.
- **Distinct platform IDs:** the union of the two endpoints of those pair IDs.

The pair and platform curves count different objects. Treat pair IDs as graph
edges and platforms as graph vertices. One pair `A-B` gives one pair but two
platforms; two disjoint pairs `A-B` and `C-D` give two pairs but four platforms.
Consequently, platform count can exceed pair count in a sparsely occupied
separation bin. For $E$ observed pairs, the platform count can be as large as
$2E$, subject to the cluster size.

`minimum_active_members` controls only the cluster activation timestamp. It is
neither a per-bin minimum nor a cap. With activation set to four, four usable
members can form up to six pairs at $t_0$, but those six pairs generally occupy
different separation bins. A fifth assigned member may also contribute later.
After activation, the clock is not reset when membership changes.

Support is evaluated within each bin. Good overall window coverage does not make
a sparsely occupied scale reliable. The vertical axis is symmetric-log so zero
counts remain visible; major and minor horizontal grid lines help read counts
across decades.

## Snapshot figures

`snapshot.png` uses simultaneous positions and velocities at the fixed reference
activation timestamp. It never combines individual drifter start times. A
single-pair bin remains visible and is labelled in `epsilon_by_scale.csv`. No
temporal bootstrap is applicable to one timestamp.

## Window figures

Files such as `initial_0_6h.png` use the left-closed, right-open cluster-age
interval. Wall-clock age continues across gaps. Bootstrap selection never
recalculates activation or age.

Primary 30-minute blocks are aligned to the cluster-window start. All pair
observations in one UTC block are resampled together. Scheduled blocks without
an observation in a particular separation bin remain in the resampling
population. The 60-minute block-duration sensitivity is saved in
`data/bootstrap_sensitivity.parquet`; only the primary duration is drawn.

## Comparison figures

`comparisons/epsilon_by_cluster__<window>.png` overlays supported cluster curves
for one window. Clusters are not pooled. Compare them only where supported scale
ranges overlap, remembering that cluster windows occur at different UTC times.

`comparisons/epsilon_by_window_small_multiples.png` gives one panel per cluster
and overlays its configured temporal windows. Missing or unsupported bins remain
absent.

## Influence information

Figures do not remove influential observations. `epsilon_by_scale.csv` retains
the largest-cubic-contribution diagnostics and adds leave-one-platform-out
summaries. Complete recalculations are in
`data/leave_one_platform_out.parquet`. A sign reversal after removing a platform
is an influence warning, not an automatic rejection.

## Units and interpretation

- Separation: metres.
- Velocity increments: metres per second.
- q and $\widehat\varepsilon_{\mathrm{eff}}$: square metres per cubic second.
- Positive epsilon is consistent with downscale transfer and negative epsilon
  with upscale transfer under the declared 3D four-fifths-law convention,
  subject to the theoretical and sampling assumptions.

These are signed effective-transfer diagnostics, not direct local viscous
dissipation measurements. A plateau, sign change, narrow interval, or multimodal
heatmap is not by itself proof of a turbulent cascade regime.
