# Poje Figures 4--7 methodology

## Source and clock

The workflow reads `platform_diagnostics.parquet` from a completed epsilon run
configured with `pair_scope: all_array`, a common array anchor, a 15-minute
sampling cadence, and a 12-day window. Arrays remain separate. The source
manifest and source tables are hash-checked before calculation.

Only finite projected positions are required for dispersion. A missing position
creates a missing platform or pair contribution at that time. Velocity validity
does not determine position availability, because Figure 4 is a trajectory-only
calculation. Positions are never extrapolated by this workflow.

## Absolute dispersion

Following Poje et al. Eq. (2), each platform is referenced to its own position at
the common array anchor:

$$
A^2(t)=\frac{1}{N_t}\sum_{i\in V_t}
\left|\mathbf x_i(t)-\mathbf x_i(0)\right|^2.
$$

The plotted value is the RMS displacement $\sqrt{A^2(t)}$ in kilometres over
the first eight days. This includes common translation of the deployment and is
not dispersion about the instantaneous centroid.

The reported early ballistic coefficient is the median of

$$
\frac{A^2(t)}{t^2}
$$

over $0<t\leq12$ h. It has units $\mathrm{m^2\,s^{-2}}$ and is a descriptive
counterpart to Poje's $v_0^2$.

## Adjusted radial relative dispersion

The pair cohort is fixed at the common anchor and contains every unordered pair
within an array satisfying the strict Poje criterion

$$
r_{ij}(0)<300\ \mathrm{m}.
$$

The calculated statistic is Poje et al. Eq. (3):

$$
D_R^2(t)=\left\langle
\left[r_{ij}(t)-r_{ij}(0)\right]^2
\right\rangle.
$$

This uses the change in scalar separation. It is deliberately named *adjusted
radial relative dispersion* in the tables because it differs from the vector
relative-displacement statistic
$\langle|\mathbf R(t)-\mathbf R(0)|^2\rangle$.

Pairs remain members of the cohort when an observation is missing. The mean at
each time uses the cohort members with finite simultaneous positions, and the
table reports their count and coverage fraction.

## Richardson comparison

The compensated time series is

$$
C_R(t)=\frac{D_R^2(t)}{t^3}.
$$

The descriptive Richardson coefficient is the median $C_R(t)$ over the
configured 2--8 day interval. A separate unconstrained log--log regression over
the same interval records the observed exponent; it does not force an exponent
of three.

## Bootstrap

The primary configuration uses 10,000 pair-bootstrap replicates and percentile
95% intervals. A replicate resamples the fixed initial pair identities with
replacement once and applies those weights synchronously to every time. This
preserves a pair's complete temporal history. Poje et al. reported standard
bootstrap intervals from 10,000 subsamples but did not specify every dependence
choice, so the synchronized implementation is an explicit ARCTERX convention.

Bootstrap intervals describe uncertainty conditional on the observed pair
cohort. Pairs share platforms and therefore are not fully independent; the
intervals should not be interpreted as accounting for all deployment-level
dependence.

## Separation memory (Figure 5a)

This calculation uses the complete unordered pair matrix inside each array,
not the Figure 4 cohort restricted by $r(0)<300$ m. For each target time $t$
and backward lag $-t\leq\tau\leq0$, the workflow evaluates

$$
R(t,\tau)=\left\langle r_{ij}(t)r_{ij}(t+\tau)\right\rangle
$$

and saves the normalized correlation

$$
\frac{R(t,\tau)}{R(t,0)}.
$$

Each target/lag value uses only pairs with finite positions at both involved
times. The numerator and denominator therefore use the same support, and the
zero-lag value is exactly one whenever support exists. The normalized lag is
computed from elapsed times as $\tau/t$, rather than inferred from row numbers.

The primary target-time groups are $[1,2)$, $[5,8)$, and $[9,12)$ days. Values
are assigned to 100 uniform $\tau/t$ bins over $[-1,0]$ and averaged with equal
weight per target-time/lag estimate. The readable table reports the number of
estimates, distinct target times, and mean contributing-pair count in every
bin. The unaggregated estimates remain in Parquet. This explicit binning is an
ARCTERX implementation choice because every aggregation detail was not stated
in the paper.

## Lagrangian velocity structure (Figure 5b)

This panel uses every platform with a valid velocity both at the common anchor
and at lag $\tau$:

$$
S_L^2(\tau,0)=\left\langle
\left|\mathbf u_i(\tau)-\mathbf u_i(0)\right|^2
\right\rangle_i.
$$

The plotted dimensional estimate follows Poje's stated constant $C_0=6.5$:

$$
\epsilon_L(\tau)=\frac{S_L^2(\tau,0)}{C_0\tau}.
$$

It is a lag-dependent Lagrangian estimate, distinct from the signed Eulerian
third-order estimate used later in Figure 7. Missing or invalid velocities are
excluded at their time; they are not silently replaced. The table reports the
contributing platform count at every lag.

## Longitudinal-increment distributions (Figure 6)

Figure 6 returns to the complete pair-time observation matrix. At every valid
time, the longitudinal velocity increment is

$$
\Delta u_l(\mathbf r,t)=
\left[\mathbf u(\mathbf x+\mathbf r,t)-\mathbf u(\mathbf x,t)\right]
\mathbin{\cdot}\frac{\mathbf r}{\lVert\mathbf r\rVert}.
$$

Poje reports bins centered at 0.25, 0.5, 1, 5, 7, and 10 km but does not give
their exact edges. The workflow resolves each reported center to the nearest
bin in the immutable source epsilon configuration. With the primary geometric
source bins, the resolved centers are approximately 0.238, 0.476, 0.951,
5.382, 7.611, and 10.763 km. A configured 10% maximum relative discrepancy
prevents a materially different source grid from being accepted silently.

For each scale, all valid pair-time increments in the 12-day window receive
equal weight. Following the paper's wording, raw increments are divided by
their population standard deviation,

$$
z=\frac{\Delta u_l}{\sigma_l},\qquad
\sigma_l^2=\left\langle
(\Delta u_l-\langle\Delta u_l\rangle)^2
\right\rangle.
$$

The mean is not subtracted from the plotted increment. The histogram uses 100
uniform bins on $-5\leq z\leq5$. Probability outside this range is retained as
underflow and overflow counts rather than renormalizing the visible density to
one.

Two Gaussian comparisons are retained. The unit-variance reference is

$$
g_1(z)=\frac{1}{\sqrt{2\pi}}
\exp\left[-\frac{1}{2}(z-\langle z\rangle)^2\right].
$$

The Poje-style central-core curve is fitted as

$$
g_f(z)=A\exp\left[-\frac{1}{2}
\left(\frac{z-\mu_f}{\sigma_f}\right)^2\right]
$$

by unweighted nonlinear least squares to the empirical histogram densities in
the configured interval $|z|\leq1.5$. Amplitude $A$, location $\mu_f$, and
width $\sigma_f$ are all free. The fit interval, bin count, parameters,
integrated Gaussian area $A\sqrt{2\pi}\sigma_f$, RMSE, and status are saved in
`increment_statistics.csv`. This is an explicit ARCTERX convention because
Poje describes a best-fit Gaussian but does not document the objective or fit
interval.

Raw second through fourth moments and their raw normalized ratios are retained
separately from central skewness and flatness. A Gaussian has central skewness
zero and flatness three. The raw third moment, not central skewness, enters the
signed Figure 7 epsilon calculation; the two can have different signs when the
increment mean is nonzero. Because observations from the same pair and shared
platforms are temporally and structurally dependent, the large raw observation
count is not an independent sample size.

Poje panels 6b and 6d use synthetic trajectories driven by AVISO geostrophic
velocities. No ancillary velocity field is required by the present ARCTERX
comparison, so the workflow intentionally implements only observational
analogues of panels 6a and 6c and labels this departure explicitly.

## Structure functions and cascade direction (Figure 7)

Figure 7 uses every source separation bin and the same equal-weight pair-time
population as the 12-day Stage 2 window. It does not reconstruct central
moments from Figure 6 histograms.

### Second-order longitudinal structure

The observational statistic is the raw second-order structure function

$$
S_{ll}^{2}(r)=\left\langle(\Delta u_l)^2\right\rangle.
$$

A free log--log ordinary least-squares fit over supported bins between 100 m
and 10 km estimates

$$
S_{ll}^{2}(r)=C r^p.
$$

The exponent $p$, its regression standard error, coefficient, and fit-bin count
are reported. A separate $p=2/3$ comparison is normalized by the geometric-mean
log residual over exactly the same bins. These fits are descriptive scaling
comparisons; shared platforms and temporal dependence mean the regression
standard error is not a complete uncertainty estimate.

Poje used a fixed Gulf of Mexico value $f=7\times10^{-5}\ \mathrm{s^{-1}}$.
The ARCTERX figure instead adapts rotation to the observations. In every
separation bin it evaluates

$$
f_b=2\Omega\sin(\overline\phi_b),
$$

where $\overline\phi_b$ is the mean pair-midpoint latitude in that bin. The
reference curves and reported Rossby number then use

$$
S_{ll}^{2}(r)=(Ro\,f\,r)^2
$$

for $Ro=1$ and $Ro=0.1$. This is the same latitude-dependent Rossby number
calculated by Stage 2. The output recomputes it and records the identity error.
Because latitude changes slightly among separation bins, the reference is a
bin-conditioned curve rather than a single exact $r^2$ line with constant $f$.

### Signed third-order estimates

Filled points use the exact saved four-fifths estimate

$$
\widehat\epsilon_{4/5}(r)
=-\frac{5}{4}\frac{\langle(\Delta u_l)^3\rangle}{\langle r\rangle}.
$$

The workflow recomputes this value and saves the absolute identity error against
Stage 2. Open points use Poje's alternate mixed relation,

$$
\left\langle(\Delta u_l)^3\right\rangle
+\left\langle\Delta u_l(\Delta u_t)^2\right\rangle
=-2\epsilon r,
$$

so that

$$
\widehat\epsilon_{\mathrm{mixed}}(r)
=-\frac{
\left\langle(\Delta u_l)^3\right\rangle
+\left\langle\Delta u_l(\Delta u_t)^2\right\rangle
}{2\langle r\rangle}.
$$

Both raw moments use the same valid observations in each bin. Positive epsilon
is consistent with forward/downscale transfer under these conventions;
negative epsilon is consistent with inverse/upscale transfer, subject to the
sampling and theoretical limitations already documented for Stage 2.

Poje's values were positive and could be displayed on a logarithmic axis.
ARCTERX changes sign, so the reproduction uses a signed symmetric-logarithmic
axis with an explicit zero line. This is a deliberate departure that prevents
negative estimates from being omitted. Source bootstrap intervals are retained
in `structure_functions.csv`, although the Poje-style panel displays points
without intervals.

## Scientific status

The current source trajectory product remains
`candidate_pending_gap_review`. The results reproduce Poje's calculation method
on ARCTERX; they are not expected to reproduce GLAD numerical values.
