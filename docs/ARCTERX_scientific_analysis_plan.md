# ARCTERX drifter analysis: scientific and implementation plan

## 1. Scientific objective

The analysis will determine how each ARCTERX drifter array evolves and whether
fronts, filaments, convergence zones, strain fields, or vortices produce distinct
changes in the motion and geometry of the drifter population.

The work will proceed from descriptive observations to increasingly specific
dynamical diagnostics:

1. visualize each array through time;
2. establish baseline trajectory, geometry, and velocity statistics;
3. identify intervals when structures affect all or part of an array;
4. calculate Poje-style velocity statistics and structure functions within those
   physically defined regimes;
5. calculate relative-dispersion and FSLE diagnostics for appropriate initially
   close pairs.

## 2. Fundamental sampling rules

### 2.1 Arrays are independent sampling populations

Drifters belonging to different experimental arrays will not be combined in a
pair or pooled into one population statistic. In particular:

- no structure-function pair will connect two arrays;
- no relative-dispersion or FSLE pair will connect two arrays;
- velocity and geometry statistics will be computed separately for each array;
- each movie will represent one array.

The independently calculated results may later be displayed side by side, but
each array will retain its own release history and physical context.

### 2.2 Deployment clusters and dynamical structures are different concepts

A **deployment cluster** is a static label describing the initial experimental
grouping. A **dynamical structure** is a front, filament, convergence zone, strain
field, or vortex encountered later. Its influence can evolve in time, combine
members of several deployment clusters, or split one deployment cluster into
different behaviors.

Consequently:

- `array_id` and `cluster_id` are stored as platform metadata;
- front or vortex encounters are identified during analysis;
- initial `cluster_id` is never treated as proof that its members remain a
  coherent dynamical group.

## 3. Analysis-ready trajectory product

The common-grid trajectory Zarr will be the central input for the array-level
analyses. In addition to positions and times, it will contain two platform-level
variables:

```text
array_id(platform)
cluster_id(platform)
```

`cluster_id` is an integer assigned progressively within each array. The complete
identifier of a deployment group is therefore `(array_id, cluster_id)`.

The Zarr attributes or build manifest will record:

- the cohort/array assignment configuration;
- temporal and spatial clustering thresholds;
- the clustering algorithm and version;
- the reviewed assignment-file hash, when applicable;
- the date and software version used to produce the assignments.

### 3.1 Initial cluster assignment

Clustering will be performed independently inside each array. Two candidate
members must satisfy a maximum start-time difference and a maximum spatial
distance.

Because this stage infers deployment groups, assignment uses the two observed
starts even when their timestamps differ:

$
d_{ij}^{\mathrm{start}}=d\!\left(
\mathbf{x}_i(t_i^{\mathrm{start}}),
\mathbf{x}_j(t_j^{\mathrm{start}})\right).
$

$
t_{ij}^{*}=\max(t_i^{\mathrm{start}},t_j^{\mathrm{start}}),
$

$
d_{ij}^{\mathrm{common}}=d\!\left(\mathbf{x}_i(t_{ij}^{*}),
                  \mathbf{x}_j(t_{ij}^{*})\right).
$

The maximum start-time difference limits how asynchronous the observed starts may
be. The first-common-grid distance is retained as a time-aligned diagnostic but
does not control the ARCTERX assignment. Both quantities remain explicitly
identified because observed starts are not verified deployment coordinates.
Additional constraints include:

- a maximum cluster diameter to prevent nearest-neighbor chaining;
- an optional maximum number of members;
- deterministic ordering and numbering of the resulting clusters;
- array-specific overrides when the experimental design requires different
  thresholds.

A platform that cannot be linked to another member under these rules receives a
unique one-member `member_id`. This denotes an **algorithmic singleton**. It does
not establish that the platform was intentionally deployed alone.

For an evenly spaced line intentionally treated as one deployment group, the
configuration can assign the complete line one cluster or use thresholds suitable
for that array. No separate deployment-geometry variable is required.

### 3.2 Meaning of a missing deployment fix

Some platforms begin reporting after their physical deployment. Their first
record may therefore be displaced from the deployment point. The assignment must
still treat that retained coordinate as an observed start rather than a verified
deployment fix. The first-common-grid diagnostic helps review such cases but does
not silently replace the configured assignment definition. A platform that does
not link under the configured rules remains a singleton.

Singleton platforms still contribute to:

- array movies;
- array-level velocity statistics;
- all-pairs, within-array structure functions;
- analyses beginning at their first reliable position.

They do not provide reliable evidence for analyses requiring the true deployment
position or initial deployment-group geometry.

## 4. Stage 1: trajectory movies

Create one movie for each experimental array. The primary movie will include:

- instantaneous drifter positions;
- recent trajectory tails;
- persistent colors selected from platform metadata in the postprocessing YAML
  (for example `cluster_id`, `member_id`, or `platform_id`), with deployment
  clusters as the primary default;
- the array centroid;
- optional deployment-cluster centroids and hulls or covariance ellipses;
- UTC time and elapsed time from the nominal array deployment;
- a scale bar, geographic coordinates, and consistent map limits;
- a clear indication of inactive or not-yet-reporting platforms.

When suitable observations are available, produce versions with independent
physical context such as SST, surface temperature, salinity, ocean color, ADT,
ship surveys, or mapped surface currents. Trajectories alone can reveal
convergence or deformation; hydrographic or tracer gradients provide stronger
evidence that the responsible structure is a front.

A synchronized diagnostic panel may show through time:

- active-platform count;
- array size;
- aspect ratio and orientation;
- convex-hull area;
- median and interquartile pair distance;
- mean array speed and internal velocity variance.

The movies are exploratory products used to recognize candidate events and guide
the quantitative analysis. Event boundaries will be based on measurable criteria
and recorded explicitly.

## 5. Stage 2: baseline statistics

Baseline statistics will be computed first for each complete array and then for
each deployment cluster with enough valid members.

### 5.1 Inventory and sampling

For each array and cluster, report:

- number of assigned platforms;
- first and last valid times;
- active-platform count through time;
- missing-data fraction;
- common lifetime;
- reconstructed coordinate method and cadence;
- distributions of initial and simultaneous pair separations.

### 5.2 Array and cluster translation

In local Cartesian coordinates, calculate the centroid

$
\overline{\mathbf{x}}(t)=\frac{1}{N(t)}
\sum_{i=1}^{N(t)}\mathbf{x}_i(t)
$

and its velocity. This separates translation of the population from internal
deformation.

### 5.3 Size and pair-distance statistics

Calculate the radius of gyration

$
R_g^2(t)=\frac{1}{N(t)}\sum_i
\left|\mathbf{x}_i(t)-\overline{\mathbf{x}}(t)\right|^2,
$

convex-hull area, nearest-neighbor distances, and the minimum, median, mean, and
upper quantiles of within-array pair separation.

### 5.4 Shape, elongation, and orientation

Construct the position covariance tensor

$
\mathbf{C}(t)=\frac{1}{N(t)}\sum_i
(\mathbf{x}_i-\overline{\mathbf{x}})
(\mathbf{x}_i-\overline{\mathbf{x}})^{\mathsf T}.
$

From its eigenvalues and eigenvectors, calculate:

- major- and minor-axis scales;
- major-axis orientation;
- aspect ratio

$
\alpha(t)=\frac{\lambda_{\min}}{\lambda_{\max}}.
$

Values near one indicate a relatively isotropic configuration; values near zero
indicate strong elongation. Line deployments are expected to begin with small
aspect ratio and require special care in two-dimensional gradient estimates.

## 6. Stage 2B: velocity statistics

Positions will be converted to eastward and northward velocities using a declared
coordinate representation and derivative method. Results will be checked for
sensitivity to the accepted linear and spline reconstructions.

For every platform,

$
\mathbf{u}_i(t)=(u_i(t),v_i(t)), \qquad
s_i(t)=\sqrt{u_i^2(t)+v_i^2(t)}.
$

### 6.1 Population distributions

For each array and selected time interval, calculate:

- empirical PDFs and CDFs of `u`, `v`, and speed;
- mean, median, standard deviation, interquartile range, and selected percentiles;
- velocity direction distributions;
- the same statistics for each sufficiently populated deployment cluster;
- time series of population medians and percentile envelopes.

Both pooled-observation distributions and time-resolved distributions will be
reported. Time-resolved percentiles avoid allowing long-lived platforms or densely
sampled intervals to dominate the interpretation.

### 6.2 Translation and internal velocity

Define the array mean velocity

$
\overline{\mathbf{u}}_A(t)=\frac{1}{N(t)}\sum_i\mathbf{u}_i(t)
$

and the velocity anomaly

$
\mathbf{u}'_i(t)=\mathbf{u}_i(t)-\overline{\mathbf{u}}_A(t).
$

The mean describes bulk translation; the anomalies describe shear, differential
advection, spreading, and deformation. Calculate the distribution of
`|u'_i|`, the velocity covariance, and the horizontal fluctuation-energy proxy

$
K'(t)=\frac{1}{2}\left\langle u_i'^2+v_i'^2\right\rangle.
$

Cluster-mean velocities and differences among cluster means will identify parts
of an array moving differently. Broadening or bimodality in velocity or velocity
anomaly distributions will be investigated as possible evidence that an array
spans more than one flow regime.

## 7. Stage 3: detection and characterization of dynamical structures

Candidate events will first be identified from the movies and baseline time
series. Quantitative evidence will then be obtained from evolving geometry and
local velocity gradients.

### 7.1 Local affine velocity-gradient fit

For a suitable local subset of drifters, fit

$
u_i \simeq u_c+u_x(x_i-x_c)+u_y(y_i-y_c),
$

$
v_i \simeq v_c+v_x(x_i-x_c)+v_y(y_i-y_c).
$

Derive

$
\delta=u_x+v_y
$

for horizontal divergence,

$
\zeta=v_x-u_y
$

for vertical vorticity, and

$
\sigma_n=u_x-v_y, \qquad
\sigma_s=v_x+u_y, \qquad
\sigma=\sqrt{\sigma_n^2+\sigma_s^2}
$

for normal, shear, and total strain.

The calculations will be normalized by the local Coriolis frequency where useful.

### 7.2 Local subsets and geometric quality

Velocity gradients may be calculated for:

- a valid compact deployment cluster;
- local nearest-neighbor groups within one array;
- all admissible combinations of a chosen number of nearby drifters;
- scale-restricted groups within a maximum radius.

Each estimate must report:

- number of members;
- RMS pair-distance scale;
- aspect ratio;
- condition number or equivalent geometric diagnostic;
- residual of the affine velocity fit.

Strongly elongated groups cannot robustly determine the full two-dimensional
velocity-gradient tensor. Following the literature, `alpha < 0.1` is an initial
candidate rejection threshold to test rather than an automatic final choice.

### 7.3 Event signatures

Interpret combinations of diagnostics rather than a single metric:

| Observed behavior | Candidate interpretation |
|---|---|
| Negative divergence, shrinking area, increasing elongation | Convergence toward a front or filament |
| Strong strain and rapid elongation | Frontal deformation or filamentation |
| Coherent rotation and large absolute vorticity | Eddy or vortex encounter |
| Rapid change in major-axis orientation or folding | Curved front, shear layer, or eddy interaction |
| Spatially separated velocity populations | Array spanning different structures |

The term *front encounter* will be used confidently when supported by an
independent temperature, salinity, density, or tracer gradient. With trajectory
evidence alone, the result will be described in kinematic terms such as
convergence line, strain-dominated feature, or coherent rotation.

### 7.4 Event catalog

Create a reviewed catalog with one row per event containing:

- `array_id`;
- event identifier;
- start and end time;
- affected platforms or selection rule;
- candidate structure type;
- supporting geometry and velocity diagnostics;
- supporting independent observations;
- confidence and notes.

This catalog defines the time windows and platform subsets used in conditioned
turbulence analyses. Event windows should be fixed before inspecting third-order
structure-function results.

## 8. Stage 4: Poje-style analysis

Poje-style two-point velocity statistics will be computed from the common-grid
`trajectories.zarr`, separately for each array. At every time, form every valid
unordered pair inside the selected array or predeclared event subset.

For pair `i,j`, calculate instantaneous separation and velocity increment:

$
\mathbf{r}_{ij}=\mathbf{x}_j-\mathbf{x}_i, \qquad
r_{ij}=|\mathbf{r}_{ij}|,
$

$
\delta\mathbf{u}_{ij}=\mathbf{u}_j-\mathbf{u}_i.
$

Resolve the increment into longitudinal and transverse components:

$
\delta u_L=\delta\mathbf{u}\cdot\widehat{\mathbf{r}},
\qquad
\delta u_T=\delta\mathbf{u}\cdot\widehat{\mathbf{t}}.
$

The analysis will include:

1. PDFs of `delta u_L` and `delta u_T` in separation bins;
2. skewness and flatness of the increment distributions;
3. second-order longitudinal and transverse structure functions;
4. signed third-order longitudinal and mixed structure functions;
5. higher-order moments supported by the sample size;
6. scale-dependent Rossby number;
7. comparison of longitudinal and transverse second-order functions;
8. rotational/divergent decomposition where sampling permits;
9. temporal Lagrangian structure functions and separation-memory diagnostics.

The principal scaling tests are

$
S_{LL}^{(2)}(r)=\left\langle(\delta u_L)^2\mid r\right\rangle
\propto r^{2/3}
$

for Kolmogorov-like second-order scaling, and

$
S_{LLL}(r)=\left\langle(\delta u_L)^3\mid r\right\rangle
\simeq-\frac{4}{5}\varepsilon r
$

for the signed third-order diagnostic. The assumptions behind the coefficient and
the interpretation of an exact dissipation rate are not fully satisfied by a
potentially divergent two-dimensional surface sample. The sign, linearity, scale
range, and robustness of the statistic will therefore be reported explicitly.

Calculations will be made for:

- each complete array over defensible intervals;
- pre-event, event, and post-event windows;
- affected and unaffected subsets when both have adequate sampling;
- relevant sensitivity cases.

The selected encounter-pair Zarr will not be used as the sole source for these
structure functions because it omits valid simultaneous spatial pairs and favors
pairs that met the encounter criterion.

## 9. Stage 5: relative dispersion and FSLE

Release-conditioned relative dispersion and FSLE require pairs with meaningful
initial conditions. These analyses will use eligible pairs from the same array,
with explicit initial-separation and encounter definitions.

### 9.1 Relative dispersion

Calculate

$
D^2(t)=\left\langle
[r_{ij}(t)-r_{ij}(0)]^2
\right\rangle
$

for documented initial-separation classes. Examine ballistic and Richardson-like
intervals, compensated curves, pair-retention counts, and sensitivity to the
definition of `t = 0`.

### 9.2 Finite-size Lyapunov exponent

For thresholds

$
\delta_n=\delta_0\rho^n,
$

measure the first-passage time from `delta_n` to `rho delta_n` and calculate

$
\lambda(\delta_n)=
\frac{\ln\rho}{\left\langle T(\delta_n)\right\rangle}.
$

Test for:

- `lambda(delta) proportional to delta^(-2/3)` for Richardson-like local
  dispersion;
- an approximately constant FSLE for exponential separation in a smooth strain
  field;
- breaks associated with different scale regimes or physical events.

Retain the first-passage-time distributions and report pairs that never reach the
next threshold. These censored pairs cannot be silently discarded.

## 10. Sampling, uncertainty, and sensitivity

Pair observations are dependent because pairs share platforms and adjacent times
share flow history. Raw pair-time counts will not be treated as independent sample
sizes.

Every diagnostic will report, where relevant:

- number of active platforms;
- number of deployment clusters represented;
- number of unique unordered pairs;
- number of time samples;
- contribution by platform and deployment cluster;
- uncertainty from resampling appropriate independent units or temporal blocks.

Sensitivity tests will include:

- linear versus accepted spline reconstruction;
- velocity derivative method and cadence;
- separation-bin boundaries;
- event-window boundaries;
- cluster spatial and temporal thresholds;
- local-gradient neighborhood size;
- aspect-ratio and fit-residual thresholds;
- inclusion or exclusion of algorithmic singletons.

Power-law intervals will be selected using sampling support, compensated plots,
and local slopes. A fitted slope alone will not define an inertial range.

## 11. Implementation order

1. Finalize and validate `array_id(platform)`.
2. Finalize the temporal-spatial deployment-clustering algorithm.
3. Attach deterministic `cluster_id(platform)` metadata to the analysis-ready
   trajectory Zarr and record provenance.
4. Implement the per-array movie workflow.
5. Implement array and deployment-cluster inventory and geometry statistics.
6. Implement velocity estimation and baseline velocity statistics.
7. Inspect movies and diagnostics and construct the reviewed event catalog.
8. Implement local affine velocity-gradient and shape diagnostics.
9. Implement Poje-style all-pairs, within-array structure functions.
10. Implement relative-dispersion and FSLE analyses for eligible pairs.
11. Add conditioned analyses for reviewed physical events.
12. Run uncertainty and sensitivity analyses and produce publication figures.

## 12. Decisions required before coding individual stages

### Metadata and clustering

- maximum start-time difference;
- maximum common-time spatial distance;
- maximum cluster diameter;
- maximum number of members, if imposed;
- array-specific rules for line deployments;
- whether cluster IDs are zero-based or one-based in user-facing outputs.

### Movies

- tail duration and frame cadence;
- platform-level variable and categorical or numeric color treatment;
- map extent policy;
- physical background products available for each array;
- geometry and velocity time series shown beside the map.

### Velocity calculation

- primary coordinate reconstruction;
- primary derivative method and smoothing policy;
- treatment of endpoints and gaps;
- standard time windows for pooled distributions.

### Event detection

- quantitative thresholds used only as flags versus hard classifications;
- minimum duration;
- minimum number of affected platforms;
- review format and confidence scale.

### Poje and FSLE analyses

- separation-bin construction;
- minimum independent sampling support per bin;
- candidate scale ranges;
- FSLE amplification factor `rho` and thresholds;
- bootstrap or block-resampling units.

## 13. Initial reference methods

- Poje, A. C., Özgökmen, T. M., Bogucki, D. J., & Kirwan, A. D. (2017),
  *Evidence of a forward energy cascade and Kolmogorov self-similarity in
  submesoscale ocean surface drifter observations*.
  https://doi.org/10.1063/1.4974331
- Berta, M., Griffa, A., Özgökmen, T. M., & Poje, A. C. (2016),
  *Submesoscale evolution of surface drifter triads in the Gulf of Mexico*.
  https://doi.org/10.1002/2016GL070357
- Tarry, D. R., et al. (2021), *Frontal Convergence and Vertical Velocity
  Measured by Drifters in the Alboran Sea*.
  https://doi.org/10.1029/2020JC016614
- Tarry, D. R., et al. (2022), *Drifter Observations Reveal Intense Vertical
  Velocity in a Surface Ocean Front*.
  https://doi.org/10.1029/2022GL098969
