# Poje reproduction figure guide

## Figure 04: dispersion

`figures/figure_04_dispersion.png` follows the layout and logarithmic axes of
Poje et al. (2017) Figure 4.

### Panel A: RMS absolute dispersion

The black curve is $\sqrt{A^2(t)}$ in kilometres, where every platform is
referenced to its own common-anchor position. The red dashed segment is the
configured early ballistic comparison. It includes translation of the complete
array and should not be interpreted as internal spreading alone.

### Panel B: adjusted radial relative dispersion

The black curve is $\langle[r(t)-r(0)]^2\rangle$ for the fixed cohort with
$r(0)<300$ m. The grey band and selected error bars show the same 95% pair-
bootstrap interval. The red line is $C_Rt^3$, using the median compensated value
between two and eight days.

The inset shows $D_R^2/t^3$ in units of
$10^{-9}\ \mathrm{m^2\,s^{-3}}$. A plateau is consistent with Richardson-like
cubic growth over that interval, but it does not establish a cascade direction.
The third-order velocity analysis is required for that inference.

The title reports the common anchor, platform population, and initial pair
count. Differences between Array 3 and Array 4 reflect different deployments
and must not be pooled silently.

## Figure 05: self-similarity diagnostics

`figures/figure_05_self_similarity.png` implements the two trajectory-only
panels in Poje et al. Figure 5.

### Panel A: scale-independent separation memory

Each colored curve is $R(t,\tau)/R(t,0)$ plotted against the normalized
backward lag $\tau/t$ for one target-time group. A value of one at zero lag is
required by the normalization. A slower decrease toward older lags means that
pair separations retain memory for a larger fraction of their current age. The
dashed reference reaches zero at $\tau/t=-1/3$ and only visualizes the
$t/3$ persistence scale discussed by Poje; it is not a fit.

The curve averages target-time/lag estimates in normalized-lag bins. Inspect
`separation_memory.csv` before comparing shapes: bins can contain unequal
numbers of target times and pair contributions.

### Panel B: Lagrangian velocity structure

The black curve is $S_L^2(\tau,0)/(6.5\tau)$ using each platform's velocity
change from the common anchor. If the inertial-range relation
$S_L^2=C_0\epsilon\tau$ applies, a plateau is an estimate of $\epsilon$.
This quantity is nonnegative by construction and does not diagnose cascade
direction. It must not be confused with the signed third-order Figure 7
estimate whose small-scale sign motivated this comparison.

The readable `lagrangian_structure.csv` table reports both $S_L^2$, the plotted
estimate, and the number of valid platforms at every lag.

## Figure 06: longitudinal-increment distributions

`figures/figure_06_increment_distributions.png` contains the ARCTERX analogues
of Poje panels 6a and 6c. The synthetic AVISO panels are intentionally omitted.

### Panel A: small separation bins

The three solid curves show normalized longitudinal increments for source bins
nearest 0.25, 0.5, and 1 km. Dashed curves are Gaussian fits to the central
$|\Delta u_l/\sigma|\leq1.5$ core. Dotted curves are the retained shifted,
unit-variance Gaussian references. Heavy tails appear as probabilities above
the Gaussian comparisons at large $|\Delta u_l/\sigma|$. A longer negative tail
and negative reported skewness indicate more intense convergent than divergent
longitudinal events under the declared pair orientation convention.

### Panel B: large separation bins

The same construction is shown for source bins nearest 5, 7, and 10 km. This
tests whether asymmetry and non-Gaussian tails persist as pair separation
increases. Curves at different scales are normalized independently, so their
horizontal widths cannot be used to compare dimensional velocity variance;
use `increment_statistics.csv` for the dimensional standard deviations.

Each legend reports the resolved source-bin center, pair-time observation
count, explicitly labelled central skewness, and fitted core width. The observation count is
support, not an effective independent sample size. Exact scale edges,
unique-pair/platform/time support, raw moment ratios, central flatness, and
histogram tail counts are in `increment_statistics.csv`. Central skewness must
not be substituted for the raw third moment used by the signed epsilon law.

## Figure 07: structure functions and cascade direction

`figures/figure_07_structure_functions.png` is the final trajectory-only Poje
comparison. The AVISO series from Poje panel 7a is intentionally absent.

### Panel A: second-order longitudinal structure

Black points show $S_{ll}^{2}(r)$. The red curve is a fixed $r^{2/3}$ K41
comparison and the blue dashed curve is the free power-law fit over 100 m--10
km. Its legend reports the fitted exponent and regression standard error. Grey
reference curves show where $Ro=1$ and $Ro=0.1$ would fall using
$f=2\Omega\sin\overline\phi_b$ from each bin's actual mean pair-midpoint
latitude. Grey crosses, if present, are bins that fail the declared Stage 2
support requirements and are excluded from scaling fits.

Agreement with an exponent near $2/3$ is consistent with energy-cascade
scaling but does not determine cascade direction.

### Panel B: signed third-order estimates

Filled black points are the four-fifths estimate. Open blue points include the
mixed longitudinal--transverse moment. Values above zero indicate a forward
sign and values below zero an inverse sign under the declared conventions.

Unlike Poje's positive-only logarithmic plot, this panel uses a signed symlog
axis. The narrow linear region around zero avoids a logarithmic singularity;
outside it, equal vertical distances represent equal orders of magnitude. Sign
changes are therefore visible rather than discarded.

The two estimates need not coincide because the surface velocity field is not
guaranteed to satisfy the isotropic assumptions behind both theoretical forms.
Their agreement or disagreement is itself a useful diagnostic. Exact values,
support, source confidence intervals, bin-mean Coriolis parameters,
latitude-dependent Rossby values, and identity checks are in
`structure_functions.csv`.
