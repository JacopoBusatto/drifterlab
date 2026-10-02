# Within-array cluster statistics: methodology and interpretation

This document describes the cluster analyses currently implemented by
`drifterlab-postprocess`. It is an equation-by-equation reference for the two
numerical products and six figures written under each array's
`cluster_statistics` directory. It describes implemented calculations, not the
later structure-function, FSLE, or feature-detection stages in the broader
ARCTERX scientific plan.

## 1. Scope and input

The analysis reads one immutable `trajectories.zarr` and processes every
`array_id` present in that store. Arrays are independent sampling populations;
no statistic combines platforms from different arrays. Within an array,
platforms are grouped by the stored `cluster_id`.

The `arrays` YAML section supplies optional metadata and per-array time-window
overrides. It is not an inclusion filter. The selected position representation
(`native`, `linear`, or one of the spline products) is global within a run.

Stored cluster membership is validated before calculation. `cluster_size` must
be finite, positive, integer-valued, constant within each array/cluster group,
and equal to the number of platforms carrying that assignment. An inconsistency
fails the run; postprocessing never silently repairs cluster metadata.

## 2. Metric coordinate system

Longitude and latitude are never used directly for Euclidean calculations. Each
array receives one fixed WGS84 azimuthal-equidistant projection. Its origin is
the wrap-safe spherical mean of one first-valid selected-coordinate position per
platform over the complete input trajectory. Consequently, every platform has
equal weight and changing the analysis time window does not move the projection.

For platform `i`, the projected position is

$$
\mathbf{X}_i(t) = [x_i(t), y_i(t)].
$$

The manifest records the projection origin, CRS definition, and origin method.

## 3. Time window, membership, and validity

Every cluster receives one row for every timestamp in its configured window.
The columns common to all rows are:

- `assigned_cluster_size`: fixed stored membership;
- `active_member_count`: members with finite selected coordinates at that time;
- `valid_pair_count`: simultaneous unordered pairs,
  `N_active(N_active - 1)/2`;
- `analysis_included`: whether scientific metrics are admitted at that time;
- `valid_velocity_sample_count`: finite platform velocity vectors at that time.

Two membership policies are available.

### Dynamic active membership

With `stop_on_member_loss: false`, every timestamp in the configured window is
admitted. Each metric uses the platforms or pairs that are valid at that time.
Changes can therefore reflect both physical evolution and changing membership.

### Stop on member loss

With `stop_on_member_loss: true`, analysis starts at the first timestamp when all
assigned members are simultaneously valid. It ends immediately before the first
subsequent loss and never restarts. Counts remain populated outside this interval,
but all scientific metrics are `NaN` and `analysis_included` is false. The summary
records the admitted start, admitted end, truncation flag, and termination reason.

Unavailable scientific quantities are always represented by `NaN`, not by zero.
Zero is retained only when it is a valid numerical result, such as the hull area
of an exactly collinear configuration.

## 4. Velocity

`difference_interval_minutes` is the total differencing span `Δ`, not a half-span.
At an interior timestamp,

$$
\mathbf{u}_i(t) =
\frac{\mathbf{X}_i(t+\Delta/2)-\mathbf{X}_i(t-\Delta/2)}{\Delta}.
$$

At an admitted boundary, a forward or backward estimate may use the same total
span:

$$
\frac{\mathbf{X}_i(t+\Delta)-\mathbf{X}_i(t)}{\Delta},
\qquad
\frac{\mathbf{X}_i(t)-\mathbf{X}_i(t-\Delta)}{\Delta}.
$$

Actual timestamps and actual elapsed seconds are used. The elapsed interval must
match `Δ` exactly. The reported platform position and the complete coordinate
stencil must be finite, and both endpoints must remain inside the admitted
interval. A stencil never crosses an internal missing coordinate, configured
boundary, or member-loss truncation boundary.

For finite simultaneous velocities, the cluster-mean vector is

$$
\overline{\mathbf{u}}(t)=\frac{1}{N_u(t)}\sum_i\mathbf{u}_i(t).
$$

The reported quantities are:

- `mean_velocity_u_m_s`, `mean_velocity_v_m_s`: components of
  `mean(u_i)`;
- platform speed `s_i = |u_i|` and its mean and configured percentiles;
- internal velocity `u'_i = u_i - mean(u_i)`;
- internal speed `|u'_i|` and its mean and configured percentiles.

Platform-level speed samples exist only in memory while figures and summaries
are constructed. They can be reproduced from the Zarr, coordinate method, and
effective configuration recorded in the manifest.

## 5. Translation and absolute displacement

The active-member centroid is

$$
\overline{\mathbf{X}}(t)=\frac{1}{N(t)}\sum_i\mathbf{X}_i(t).
$$

`centroid_x_m` and `centroid_y_m` are its projected components.
`centroid_displacement_m` is the distance from the first finite admitted
centroid.

Each platform has its own reference position at its first valid admitted time
`t_i0`. Its absolute displacement is

$$
d_i(t)=|\mathbf{X}_i(t)-\mathbf{X}_i(t_{i0})|.
$$

The time series reports the member mean and configured percentiles. These are
absolute translations from member-specific origins; they are not pair
dispersion.

## 6. Covariance geometry, extent, and shape

For `N >= 2` active members, the population covariance uses normalization by
`N`, not `N-1`:

$$
\mathbf{C}(t)=\frac{1}{N(t)}\sum_i
[\mathbf{X}_i(t)-\overline{\mathbf{X}}(t)]
[\mathbf{X}_i(t)-\overline{\mathbf{X}}(t)]^\mathsf{T}.
$$

Its ordered eigenvalues are
`lambda_major >= lambda_minor >= 0`. They define

$$
L_\mathrm{major}=\sqrt{\lambda_\mathrm{major}}, \qquad
L_\mathrm{minor}=\sqrt{\lambda_\mathrm{minor}},
$$

$$
\text{eigenvalue ratio}=
\frac{\lambda_\mathrm{minor}}{\lambda_\mathrm{major}}, \qquad
\text{aspect ratio}=
\sqrt{\frac{\lambda_\mathrm{minor}}{\lambda_\mathrm{major}}}.
$$

The two ratios are distinct. Both range from zero to one; values near zero are
elongated and values near one are isotropic.

The radius of gyration is

$$
R_g(t)=\sqrt{\frac{1}{N(t)}\sum_i
|\mathbf{X}_i(t)-\overline{\mathbf{X}}(t)|^2},
$$

and the implementation checks the identity

$$
R_g^2=\lambda_\mathrm{major}+\lambda_\mathrm{minor}.
$$

The major-axis orientation is the major eigenvector angle counterclockwise from
east, reduced to `[0, 180)` because an axis has no arrow direction. It is `NaN`
when the major eigenvalue is zero or the eigenvalues are equal within the
documented relative tolerance of `1e-8`.

Tiny negative eigenvalues caused by floating-point roundoff are clipped at a
scale-aware tolerance. Materially negative eigenvalues fail the calculation.

## 7. Convex-hull area and normalized area

For at least three active members, `convex_hull_area_m2` is the area of their
two-dimensional convex hull in the fixed projected frame:

$$
A(t)=\operatorname{area}\left(\operatorname{hull}
\{\mathbf{X}_i(t)\}\right).
$$

Numerically collinear configurations have `A(t)=0`. Fewer than three active
members have no two-dimensional hull and receive `NaN`.

The normalized footprint is

$$
A_\mathrm{ratio}(t)=\frac{A(t)}{A(t_0)},
$$

where `t₀` is the first admitted timestamp at which that cluster has a finite
hull area. The summary records `convex_hull_reference_time` and
`convex_hull_reference_area_m2`. The time-series column is
`convex_hull_area_ratio`.

The ratio equals one at `t₀`, values above one indicate expansion relative to the
reference footprint, and values below one indicate contraction. If the reference
area is zero or unavailable, the ratio is `NaN` for the entire cluster because
division by that reference is undefined. With dynamic membership, both `A(t)` and
its ratio can change when the active-member set changes, so they must be read
alongside `active_member_count`.

## 8. Pair separation and vector relative dispersion

Pairs are unique, unordered, and restricted to one stored array/cluster. For
pair `(i,j)`,

$$
\mathbf{R}_{ij}(t)=\mathbf{X}_j(t)-\mathbf{X}_i(t), \qquad
r_{ij}(t)=|\mathbf{R}_{ij}(t)|.
$$

The time series contains the mean and configured percentiles of `r_ij`, plus

$$
\left\langle r_{ij}^2(t)\right\rangle.
$$

Each pair receives its own reference time `t_ij0`, the first admitted timestamp
when both members are simultaneously valid. Its vector-change quantity is

$$
q_{ij}(t)=|\mathbf{R}_{ij}(t)-\mathbf{R}_{ij}(t_{ij0})|^2.
$$

The implemented relative dispersion is

$$
D^2(t)=\operatorname{mean}_{ij}q_{ij}(t).
$$

This is not the scalar radial-separation change
`[r_ij(t)-r_ij(t_ij0)]²`; the latter discards rotations of the relative-position
vector and is not labelled `D²`.

The time series reports `relative_dispersion_D2_m2` and the configured
percentiles of `q_ij` under the `relative_displacement_q_*_m2` columns.

## 9. Percentiles and summaries

Percentiles accept integer or fractional values. Column names are canonical and
unambiguous, for example:

- `0 -> p000`;
- `25 -> p025`;
- `100 -> p100`;
- `12.5 -> p012p5`.

Configurations whose canonical names collide are rejected.

`cluster_timeseries.csv` contains instantaneous means, percentiles, counts, and
geometry. `cluster_summary.csv` contains one row per cluster with:

- membership policy and admitted-interval audit fields;
- assigned and contributing platform counts;
- total valid velocity-sample count;
- speed and internal-speed means and percentiles;
- maximum centroid displacement and maximum valid-pair count;
- means and percentiles of radius of gyration, major scale, minor scale,
  aspect ratio, convex-hull area, and convex-hull area ratio;
- the hull-area reference timestamp and value.

Velocity summaries use every finite admitted platform-level sample retained in
memory. Geometry and shape summaries use every finite admitted time-series value.
Neither calculation combines clusters.

## 10. Figures

Each array receives six figures:

1. `velocity_distributions.png`: absolute and internal platform-speed
   distributions with common bins across clusters in the array; probability-
   density y-axes are independently autoscaled by default, or may share one
   array-wide limit through `share_probability_density_y_axis`;
2. `absolute_displacement.png`: member displacement envelopes and means plus
   centroid displacement;
3. `pair_separation.png`: pair-separation envelopes and means;
4. `relative_dispersion.png`: mean squared separation, vector `D²`, and
   `q_ij` percentiles on the configured linear or logarithmic scale;
5. `cluster_extent_and_shape.png`: radius of gyration, major/minor scales,
   aspect and eigenvalue ratios, convex-hull area, `A(t)/A(t₀)`, and active
   membership;
6. `cluster_orientation.png`: axial major-axis orientation as points, avoiding
   misleading lines across the 0/180-degree wrap.

When percentiles `0, 25, 50, 75, 100` are present, figures show a median line,
25–75% envelope, and min–max envelope. These are descriptive distributions, not
confidence intervals. Minima and maxima are especially sensitive to position
outliers.

## 11. Interpretation limits

These products are exploratory descriptive statistics. They support comparisons
among clusters within the same array but do not establish statistical
equivalence or causality. They currently contain no hypothesis tests, similarity
score, automatic cluster regrouping, separation-binned structure functions,
FSLE, or front/eddy/convergence detection. Those analyses require separately
declared sampling and event definitions.
