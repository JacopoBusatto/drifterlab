# Original-cluster endpoint affine deformation

`drifterlab-analyze` can independently enable `analyses.affine_deformation`.
It reads one reviewed linear-position product in the grouped v2 store and adds
`06_affine_deformation/array_NNN`. Existing covariance, epsilon, FSLE, grouping,
and velocity calculations are unchanged. No velocity derivative enters this
estimator. The first implementation supports daily UTC maps, optionally with
prepared daily or hourly Celsius SST. Raw satellite ingestion remains outside
its scope, as do final-configuration axes, separate LE/FTLE scatter maps, front
detection, and quantitative thermal-alignment statistics.

## Scientific definition

For the same original members at times `s` and `s+T_prime`, subtract the
equal-member-weight centroid at each endpoint:

\[
\xi_k=x_k(s)-\bar x_0,\qquad
\eta_k=x_k(s+T')-\bar x_1.
\]

Fit the dimensionless matrix

\[
F=\arg\min_A\sum_k\|\eta_k-A\xi_k\|^2.
\]

With row-wise `X` and `Y` of shape `(N,2)`, the implementation solves
`X @ F.T ≈ Y` with SVD-based `numpy.linalg.lstsq`. The full-rank derivation is
`F = (sum eta xi.T) (sum xi xi.T)^(-1)`, but no explicit inverse is computed.
Rank-deficient solutions are rejected; a least-squares pseudoinverse does not
make an underdetermined configuration scientifically valid.

`F` maps relative endpoint configurations and contains rotation, shear,
stretching and contraction. Centroid displacement measures translation.
The fitted member displacement is

\[
\Delta x_k\simeq(\bar x_1-\bar x_0)+(F-I)\xi_k.
\]

Thus a direct fit of relative displacements would estimate `F-I`, not `F`.
Intermediate observations do not enter the regression. Continuous interior
coverage is not required: this describes a net endpoint transformation and
cannot recover intervening deformation or rotation history.

The **right Cauchy–Green deformation tensor** is

\[
C=F^\mathsf{T}F,\quad Ce_i=\lambda_i e_i,\quad
\lambda_1\ge\lambda_2\ge0,\quad
a_i=\sqrt{\lambda_i},\quad \sigma_i=\log(a_i)/T'.
\]

The symmetric eigensolver returns initial-configuration principal axes, whose
sign is arbitrary. `a_i` are stretch factors; rates are saved in `h^-1`.
Positive rates mean extension and negative rates contraction. `e_1` is the
largest-stretch axis, but both rates may be negative (or positive).
For a rigid rotation `F=R`, `C=I`, both stretches are one and both rates zero.
More generally, multiplying `F` on the left by a rigid rotation leaves `C`
unchanged. Undeformed means `C=I`, not `C=0`; the Green–Lagrange strain would
be `(C-I)/2`. Shear can change lengths and is retained.

Position covariance describes shape at one instant, in square metres. This
dimensionless `C` describes a transformation between two instants. Output names
always use `cauchy_green_*` to avoid confusing the tensors. Along-axis
contraction does not imply horizontal convergence: the orthogonal axis may
extend. For an orientation-preserving affine map, `det(F)` is the fitted area
factor and `sigma_1+sigma_2=log(det(F))/T_prime`; this is a finite-interval
area-change rate, not an instantaneous velocity-divergence measurement.

## Coordinates and geographic display

The existing WGS84 azimuthal-equidistant (AEQD) transform supplies one fixed
metric frame per array, reused for both endpoints. Its origin is the equal-weight,
wrap-safe spherical mean of each platform's first finite selected-product
position available before the analysis cutoff. Cluster selection does not move
the origin. Platforms without any position before that cutoff cannot supply an
origin contribution. The exact CRS and origin method are saved.

Longitude/latitude degrees never enter least squares. AEQD preserves distances
from its origin; distances and angles elsewhere acquire distortion as the array
travels or spreads. Axis directions are in this fixed projected basis and are
not assumed to be local geographic east/north everywhere. Large spatial extent
requires checking projection distortion; no unsupported scientific maximum
extent is imposed.

Maps inverse-transform both ends of each short metric axis segment to
longitude/latitude. They use a local equirectangular display with fixed
reference-latitude aspect correction; projected components are never added
directly to degrees. There are no arrowheads. With `plotting.fixed_extent: true`
(the default), map bounds stay fixed across dates for an array. With `false`,
each map fits that day's finite, numerically valid initial cluster centroids,
including quality-flagged fits, with a minimum 1 km span in each projected
direction and a 6% margin. Empty frames fall back to the array-wide view. These
are display choices, not claims about spatial resolution. The Array 3 configs
select `false` to make each day's data easier to see. Both panels share the same
bounds, which also determine the SST subset and are saved in each window's
metadata for replotting. Changing the view does not change affine diagnostics.

With `fixed_extent: true`, optionally set `longitude_bounds: [125, 135]` and
`latitude_bounds: [15, 30]` under `plotting` to choose the exact displayed
geographic rectangle. Each pair is `[minimum, maximum]` in degrees; either
may be omitted or `null` to retain the automatic array-wide bounds on that
axis. Longitude must increase within [-180, 180] and latitude within [-90, 90];
crossing the antimeridian is not supported by these explicit bounds. Non-null
bounds with `fixed_extent: false` are rejected to avoid silently ignoring them.
The bounds apply to both panels and SST selection, with one grid-cell halo
for rendering. Fits outside the view remain in the diagnostics. Exact geographic
bounds are saved in each window as `geographic_extent_deg` in
`[west, east, south, north]` order and reused when replotting. The projected
`extent_m` encloses that rectangle for metric segment sizing.

When `axis_length_km: null`, segment length is 2.5% of the current map span;
an explicit length in km preserves the physical symbol length across changing
views. Padding accommodates explicitly configured segments. Symbol length is
a display convention and does not encode stretch magnitude. Rate normalization
stays fixed across dates in both extent modes. A single divergent
`coolwarm` scale, centred at zero, serves both panels; saturated-axis counts
and colourbar extensions expose out-of-range values. Optional SST has its own
fixed grayscale colourbar. Rendering needs no network or coastline downloads.

## Membership, lifecycle, and endpoint interpolation

Original platform IDs are saved in `original_membership.csv` and every interval
row. Stored `cluster_size` must agree with the complete assigned identity set;
inconsistent metadata fails rather than silently repairing membership. Clusters
with fewer than three members receive explicit exclusion reasons. Never reduce
a cluster to surviving members, merge it, or replace members.

Each cluster activates at the first product-grid timestamp at which all its
original members can be reconstructed under the endpoint rules. It ends at
the earliest last usable position among those members. An endpoint exactly at
that boundary is allowed. Temporary holes do not end the cluster; later complete
intervals remain eligible. This intentionally differs from the existing cluster
statistics `stop_on_member_loss` behavior. A trailing missing stretch is treated
as the end of available usable data; no inference about hardware failure is made.

The existing `resample_linear_track` reconstructs requested endpoints without
extrapolation, extra smoothing, or a requirement for exact grid alignment.
Longitude interpolation is wrap-safe. Every original member needs a valid
position at both endpoints. The effective gap limit inherits the selected
product's `maximum_source_gap_minutes`; an explicit affine setting may tighten
it but cannot loosen that upstream policy.

Product samples must themselves have finite source-gap provenance within the
limit. The new endpoint's support gap is the maximum of the resampling bracket
duration and inherited source gaps at both bracket samples. This prevents an
already interpolated product point from being treated as a new exact observation.
No product samples after `analysis_end` support endpoint reconstruction. In
particular, an off-grid endpoint exactly at the cutoff may be unavailable when
its next bracketing observation lies after the cutoff. Previously reconstructed
product values retain their upstream source support; this module cannot undo
the builder's reconstruction or recover source-fix timestamps absent from the store.

Interior product-grid support is not checked for fit validity. An unobserved
excursion, collapse/re-expansion, or rotation between good endpoints cannot be
detected by this estimator. SST availability never affects trajectory eligibility.

## Shared time references

One `analysis_start` per array is the configured UTC `reference_time_utc`, or
the earliest complete cluster with at least three original members. Automatic
reference derivation uses all original clusters in the selected array before
narrowing cluster selection. Each cluster retains its own activation time;
the schedule does not wait for all clusters or restart separately for them.

`map_origin` is UTC midnight containing `analysis_start`. Maps cover consecutive
half-open days `[midnight, next_midnight)`. `map_window_hours` must equal 24 in
this version. Positive `deformation_window_hours` must divide 24 exactly; 6, 12,
and 24 hours are supported. Duration validation uses integer nanoseconds.
Diagnostic intervals are midnight-phased, nonoverlapping `[s,s+T_prime]` fits.
Their start belongs to one map; their final endpoint can lie at the next midnight.

If supplied, positive `maximum_elapsed_hours` gives
`analysis_end=analysis_start+Tmax`; null uses the latest usable lifetime end
among selected eligible clusters, avoiding maps of trailing product padding.
Only complete diagnostic intervals inside the analysis bounds are scheduled.
Cluster activation/lifetime and endpoint checks then determine their validity.
There is no shortening at any boundary. For example, a 02:00 start and 6-hour
fits first admit 06:00–12:00. The first/last map may contain fewer fits.
Requested SST coverage remains a full UTC day even for a partial diagnostic day.
The manifest separately records analysis overlap, scheduled coverage and the
span of valid diagnostics; that span does not imply continuous valid coverage.

## Validity, quality, and display

`numerically_valid`, `quality_flags`, and `axis_displayable` are separate.
The initial-centred matrix rank threshold is configurable (default relative
SVD cutoff `1e-12`). The saved condition number is `s_max(X)/s_min(X)`, not the
squared condition number of `X.T @ X`. Covariance uses population normalization
`X.T @ X/N`; its major/minor RMS scales are `s_i(X)/sqrt(N)`. Aspect ratio is
`sqrt(lambda_cov_minor/lambda_cov_major)=s_min(X)/s_max(X)`.

The provisional `minimum_aspect_ratio: 0.1` flags poor geometry and hides axes
by default. `show_poor_geometry: true` overrides only that display flag, not
rank rejection or other suppression rules. Physical scale and normalized-residual
thresholds default to null; no GPS uncertainty or scientifically established
minimum scale/residual is invented. Raw metrics remain available.

Residual RSS is `sum_k ||eta_k-F xi_k||^2`; per-member Euclidean RMSE is
`sqrt(RSS/N)` in metres. The dimensionless residual is RMSE divided by final
centred RMS radius, undefined when that radius is zero. A three-member fit can
be exact by construction and always receives a warning flag about its lack of
residual degrees of freedom. A zero residual is not independent evidence of
reliability.

Near-isotropy uses `(lambda1-lambda2)/lambda1 <= 0.001`, configurable as
`near_isotropy_relative_tolerance`. It suppresses unstable directions while
retaining stretches/rates. Otherwise acceptable isotropic fits use a neutral
marker in both panels. This provisional tolerance is not an uncertainty estimate.

`det(F)<0` flags orientation reversal; zero determinant or a stretch at/below
`near_collapse_stretch` (default `1e-12`, dimensionless) flags collapse or near
collapse. These fits retain diagnostics but have no physical axes displayed.
The determinant is never replaced by its absolute value. Zero stretch receives
`-inf` logarithmic rate without a log warning; tiny positive stretches retain
their actual logarithmic rates and the collapse flag. Roundoff-scale negative
eigenvalues within `64*eps*max(abs(eigenvalues))` are clipped and recorded;
larger violations are numerical failures.

## Optional prepared SST input

SST preparation is a separate workflow. Select one prepared NetCDF with
`analyses.affine_deformation.sst.file`; no source-product selector is needed.
Both `L3S_daily_20250201_20250331.nc` and
`L3U_hourly_20250201_20250331.nc` use the supported common interface:

- `sea_surface_temperature(time, lat, lon)`, already masked and in Celsius;
- one-dimensional rectilinear `lat` and `lon` coordinates in geographic degrees;
- CF-decoded `time` and its referenced bounds array with shape `(time, 2)`;
- ordered, nonoverlapping bins with a common positive duration. Missing bins
  are allowed, and the nominal time must lie inside its bin.

MUR L4 can use the same interface after the separate
[MUR preparation step](mur_sst_preparation.md). Its daily analysis is assigned
to the nominal date for plotting, while the original nominal time and source
coverage are preserved. This is an explicitly labelled mapping convention,
not a claim that MUR is a UTC daily mean. L4 availability is not observation
coverage.

The reader applies no NOAA quality levels, flag masks, bias corrections or
source-specific variable mappings. Raw-product decoding, quality selection,
Kelvin conversion and any source-specific compositing belong in preparation.
Extra variables such as L3S quality flags, SST gradients or source counts are
ignored by the SST estimator. Global upstream metadata and the prepared-file
path, size, modification time and SHA-256 are retained as provenance. In
particular, the prepared L3S file's recorded mask of 58630 is respected through
its already masked values; the analysis does not reapply the former 49414 mask.

For each UTC daily map, `time_bounds` determine selection, rather than a
filename or the nominal reference time. A supplied daily bin enters unchanged,
including a daily composite whose nominal reference is noon. Hourly input bins
are combined using an equal-hour NaN-ignoring mean. Neither the number of
observations nor the number of overlapping upstream granules weights that mean.
This temporal reduction uses the common bin interface and contains no L3S/L3U
branching. A bin crossing a map boundary is rejected rather than split or reused
as apparent finer-time observations. No interpolation or gap filling occurs.

Pixels with no finite contributors remain NaN. Saved subsets contain
`contributing_bin_count` and per-pixel `temporal_coverage_fraction`, whose
denominator is the number of bins expected in the full requested window:
24 for hourly input and one for daily input. Missing hours/days remain in that
denominator. Window metadata separately records available bin counts, spatial
coverage (pixels with any finite SST divided by all regional subset pixels),
and mean temporal bin coverage. This counts available prepared bins, not
continuous acquisition-time coverage. If an input has `observation_count`, its values are
summed over bins with finite SST and retain their declared meaning; they are
never averaging weights. Counts are not fabricated for other products. Other
ancillary time-offset variables are not interpreted as acquisition times by
this generic reader.

There is no minimum contributing-hour threshold: one finite hourly bin is
enough to give a pixel a daily value, with temporal bin coverage `1/24`.
All-missing pixels remain blank. A missing SST pixel or entire background does
not invalidate a trajectory fit. Prepared-file quality flags are not rechecked;
upstream masking determines which SST values are available to the mean.

Disabled/absent SST is valid; missing temporal or spatial coverage produces a
neutral background with `sst_unavailable`, without removing deformation rows.
An explicitly supplied missing/corrupt file, unsupported units/dimensions or
invalid time bounds raises an actionable error. Longitude convention,
descending latitude and regional subsetting are handled before loading SST
arrays. Subsets are cached across panels/arrays where their footprint matches.

Example configuration:

```yaml
sst:
  enabled: true
  file: C:/Users/Jacopo/Documents/DATI/VIIRS_NOAA/L3S_daily_20250201_20250331.nc
  # Or: C:/Users/Jacopo/Documents/DATI/VIIRS_NOAA/L3U_hourly_20250201_20250331.nc
  minimum_celsius: 24
  maximum_celsius: 26.5
```

Use different output roots when comparing different SST inputs. A daily
composite and an average of observed hourly bins have different temporal
support and preparation histories; both provide context, not an instantaneous
field simultaneous with every deformation estimate.

## Outputs and reproducibility

Each array has `data/affine_intervals.parquet` and `.csv`, one row per selected
cluster and attempted complete diagnostic interval, including invalid attempts.
`data/original_membership.csv` also includes never-eligible clusters and their
reasons. Intervals carry array/cluster/member identities, activation/lifetime,
map/fit times, endpoint counts and support gaps, centroids, displacement, `f_ij`,
`cauchy_green_ij`, sorted eigenpairs, stretches/rates, geometry, residuals,
determinant, anisotropy, status/reasons, quality flags and SST status.

`metadata.json` records the algorithm version, resolved settings, input-product
attributes, effective interpolation policy, CRS, fixed map bounds, counts,
exclusion/flag summaries, UTC windows and SST provenance/coverage. Map-window subsets
are in `sst/YYYYMMDD.nc`; paired figures are in `figures/YYYYMMDD.png`. The unified
root manifest records exact store/product/cadence, selected arrays, source
provenance, software version, configuration-derived run ID and hashes of outputs.
Use separate output roots for alternative windows or thresholds. Existing output
is refused unless explicit `--overwrite` requests atomic replacement.

Runnable configuration templates are
[`06_affine_deformation.yml`](../configs/examples/v2/06_affine_deformation.yml)
and [`06_affine_deformation_l3s.yml`](../configs/examples/v2/06_affine_deformation_l3s.yml).
They require the reviewed store produced by example 01. For NOAA NetCDF4 input,
install an HDF5 reader with `pip install -e ".[sst]"` if none is available.

```powershell
drifterlab-analyze configs/examples/v2/06_affine_deformation.yml
drifterlab-analyze configs/examples/v2/06_affine_deformation_l3s.yml
python -m drifterlab.cli.affine_replot data/examples/v2/06_affine_deformation/no_sst_6h/06_affine_deformation/array_001
python -m pytest tests/test_affine_deformation.py -q
```

The replot command reads saved intervals, settings, map bounds and SST subsets;
it neither reloads trajectories nor recomputes fits. It restores the recorded
rendering settings and writes `replots/`, preserving the original figures and
their published hashes. These later previews are outside the original manifest.
Rerendering with changed scientific thresholds requires
re-evaluating display eligibility from the preserved raw diagnostics.

## Interpretation limits

This is a finite-cluster affine estimate, not automatically the derivative of
a pointwise flow map. Finite cluster size, position error, poor geometry and
non-affine motion limit its meaning. Endpoint-only estimates can conceal complex
intervening histories. Projection distortion grows away from the local origin.
Daily SST can blur moving fronts and its resolved features need not match the
cluster's spatial or temporal scale. SST alone establishes neither density
structure nor a material barrier. Visual correspondence is exploratory evidence,
not a quantitative alignment statistic or proof of causality.

## Implementation validation (2026-10-22)

The affected affine, unified-analysis, cluster-statistics and reconstruction
test set passed 46 cases. The affine suite subsequently passed all 12 cases
after the final null-Tmax lifetime correction. Synthetic checks recover identity,
translation, rotation, anisotropic stretch and shear; verify frame changes,
endpoint interpolation/source-gap provenance, lifecycle and scheduling; exercise
geometry, residual, reflection/collapse and isotropy flags; and test SST masks,
units, empty coverage, replotting and identical affine results with/without SST.

A real-data 30-hour pilot used the finalized `qc_5min` linear-position product
in `array_03_30D.zarr` (65 platforms, campaign Array 3 / local array 1), its
30-minute gap limit, all 16 original clusters, 6-hour fits and the supplied
February 6–7, 2025 L3S Daily files. It produced 64 attempted rows, 39 numerically
valid fits and two inspected maps. Thirteen attempts lacked allowed endpoint
support and twelve had fewer than three original members. Among the numerical
fits, ten were flagged for orientation reversal and seven for poor geometry
(flags may overlap); these are diagnostics, not accepted physical-axis counts.
Regional SST coverage was about 82.0% and 97.6%. SST provenance includes source
file hashes. Fixed regional SST colour limits of 24–26.5 °C made thermal context
visible; the generic examples leave their colour limits configurable.

The pilot is an execution and rendering check, not full-lifetime scientific
validation. Other arrays, the full available campaign lifetime, sensitivity to
provisional thresholds and independent position-error/flow validation were not
evaluated. No quantitative thermal-alignment result is claimed.

On 2026-10-23 the reader was changed to the shared prepared-file interface.
The affected test set passed 53 cases, including equal-hour weighting, absent
bins, per-pixel coverage, invalid time bounds, and source-independent daily SST.
Separate 30-hour pilots with the prepared daily and hourly files each produced
64 attempted rows, 39 numerically valid fits and two inspected maps. Their
affine tables were identical apart from background status. In the February 6
map region, the prepared daily product covered about 82.0% of pixels and the
hourly product about 24.0%; missing hourly coverage was preserved.
