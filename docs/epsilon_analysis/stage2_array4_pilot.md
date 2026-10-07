# Stage 2 Array 4 linear pilot

## Run identity and scope

This report records the numerical Stage 2 pilot for configurable Array 4. The
run used the linear position product, a 30-minute centered two-endpoint velocity,
the seven-position validity stencil, a 30-minute maximum source-support gap, no
speed mask, `minimum_active_members: 2`, and within-cluster pairs. Its
deterministic identity is `epsilon-v1-8f9dcecd8c0233af`. The local run
configuration selected Arrays 3 and 4; this report continues to describe the
Array 4 subset. Output schema 2.0 consolidates each array into four inspection
tables and two normalized Parquet datasets.

This recorded pilot predates the explicit `require_full_stencil_support` option;
its behavior is equivalent to setting that option to `true` for both reference
activation and analysis.

The source product status is `candidate_pending_gap_review`; the configuration
therefore opts into candidate input explicitly. The 62.5 m minimum bin edge,
null minimum reliable separation, support thresholds, and choice of linear
positions are provisional scientific choices rather than accepted instrument
limits or a final primary reconstruction.

No Stage 3 uncertainty interval, plot, leave-one-platform-out analysis, or
sensitivity comparison was run.

## Validation

- The full repository suite passes: 331 tests.
- Focused epsilon and workflow tests pass: 10 tests.
- Tests cover the explicit two-endpoint formula, all-seven-position validity,
  source-support and speed masks, longitudinal/transverse invariance, bin edges,
  raw versus central moments, exact epsilon identity, coverage denominators,
  singleton handling, and fixed reference activation under an invalidating
  analysis sensitivity.
- The largest temporal-bin discrepancy between the direct epsilon estimator and
  the separation-weighted mean of individual $q$ is
  $2.71\times10^{-20}\ \mathrm{m^2\,s^{-3}}$. The snapshot maximum is
  $2.65\times10^{-23}\ \mathrm{m^2\,s^{-3}}$.

## Activation and snapshot coverage

The inventory contains 14 clusters and 64 platforms. Thirteen clusters
activate. `array_004__cluster_004` is an algorithmic singleton and cannot create
an internal pair.

| Cluster | Assigned members | Reference activation (UTC) | Active members | Snapshot pairs |
|---|---:|---|---:|---:|
| 001 | 4 | 2025-02-11 11:35 | 3 | 3 |
| 002 | 4 | 2025-02-11 11:55 | 2 | 1 |
| 003 | 5 | 2025-02-11 12:25 | 3 | 3 |
| 004 | 1 | unavailable | 0 | 0 |
| 005 | 5 | 2025-02-11 12:40 | 2 | 1 |
| 006 | 5 | 2025-02-11 13:00 | 5 | 10 |
| 007 | 5 | 2025-02-11 13:15 | 2 | 1 |
| 008 | 5 | 2025-02-11 13:35 | 2 | 1 |
| 009 | 5 | 2025-02-11 13:55 | 2 | 1 |
| 010 | 5 | 2025-02-11 14:15 | 3 | 3 |
| 011 | 5 | 2025-02-11 14:35 | 2 | 1 |
| 012 | 5 | 2025-02-11 15:00 | 2 | 1 |
| 013 | 5 | 2025-02-11 15:25 | 5 | 10 |
| 014 | 5 | 2025-02-11 16:05 | 5 | 10 |

There are 46 valid snapshot pair observations. They occupy 32 separation bins:
22 contain one pair, seven fail the pair or platform support requirement, and
three have adequate descriptive pair/platform support but no temporal basis for
an uncertainty estimate. All are retained as descriptive estimates and none is
marked suitable for temporal confidence intervals.

## Cluster-window coverage

Coverage uses every assigned internal pair at every scheduled cluster-relative
UTC as its denominator, including time before a late member becomes usable.

| Window | Valid pair-times | Expected pair-times | Aggregate coverage | Cluster range |
|---|---:|---:|---:|---:|
| $[0,6)$ h | 8,492 | 8,784 | 0.966758 | 0.887500–1.000000 |
| $[0,12)$ h | 16,997 | 17,568 | 0.967498 | 0.861111–0.995139 |

The six-hour exclusions are 121 nonfinite-position stencils and 171 stencils
whose source gap exceeds 30 minutes. The 12-hour exclusions are 121 and 450,
respectively. No speed mask was applied.

Of the valid observations, 8,108 of 8,492 six-hour samples and 16,193 of 16,997
12-hour samples fall inside the provisional 62.5 m–128 km bins. The 384 and 804
remaining samples are below 62.5 m; none is above 128 km or has invalid
separation.

## Separation-bin temporal support

Support is assessed separately within each cluster, window, and separation bin.
The criteria are at least 12 observations, three unique pairs, three platforms,
and 12 distinct UTC timestamps.

| Window | Nonempty bins | Supported | Low support | Observations in supported bins | Supported pair range | Supported UTC range |
|---|---:|---:|---:|---:|---:|---:|
| $[0,6)$ h | 111 | 80 | 31 | 7,624 | 3–10 | 15–72 |
| $[0,12)$ h | 135 | 106 | 29 | 15,524 | 3–10 | 18–141 |

Supported mean separations span 75.55–2,154.80 m in the six-hour products and
72.38–3,203.21 m in the 12-hour products. Empty and low-support rows remain in
the tables with explicit flags.

## Signed third order and epsilon

Across the 186 supported temporal bins, 60 epsilon estimates are positive and
126 are negative. The complete supported range is

$$
-1.700834\times10^{-4}
\le \widehat\varepsilon_{\mathrm{eff}}
\le 8.060010\times10^{-5}
\quad \mathrm{m^2\,s^{-3}}.
$$

The 32 descriptive snapshot-bin estimates include nine positive and 23 negative
values, ranging from $-2.406958\times10^{-6}$ to
$2.314702\times10^{-7}\ \mathrm{m^2\,s^{-3}}$. Their lack of temporal support
precludes treating them as robust cascade estimates.

The changing sign across clusters and scales does not establish a common cascade
direction. The signed $S_{LLL}$, epsilon, raw and central longitudinal moments,
and all corresponding transverse quantities are preserved in the numerical
products.

## Influential observations and numerical cautions

Third moments are highly sensitive to rare increments. In 89 of 186 supported
bins, one observation contributes more than 25% of the total absolute cubic
contribution; in 27 bins it contributes more than 50%.

The strongest cases are:

- Cluster 006, $[0,6)$ h, 125–176.8 m: 27 observations, three pairs and 23 UTCs;
  $\widehat\varepsilon=-1.700834\times10^{-4}$, but one observation contributes
  99.93% of absolute cubic magnitude. Removing it gives
  $+1.0183\times10^{-7}\ \mathrm{m^2\,s^{-3}}$ and reverses the sign.
- Cluster 008, $[0,6)$ h, 88.4–125 m: 19 observations, three pairs and 19 UTCs;
  $\widehat\varepsilon=+7.26931\times10^{-5}$. The leading observation
  contributes 98.84%; removing it gives
  $-7.83512\times10^{-7}\ \mathrm{m^2\,s^{-3}}$.
- Cluster 006, $[0,12)$ h, 125–176.8 m: 67 observations, four pairs and 51 UTCs;
  $\widehat\varepsilon=-6.03420\times10^{-5}$. The leading observation
  contributes 94.75%; removing it gives
  $+3.45820\times10^{-6}\ \mathrm{m^2\,s^{-3}}$.

Valid temporal rows span 1.31–4,141.62 m in separation,
$-0.908$ to $1.043\ \mathrm{m\,s^{-1}}$ in longitudinal increment,
$-1.090$ to $0.982\ \mathrm{m\,s^{-1}}$ in transverse increment, and
$-0.00435$ to $0.00489\ \mathrm{m^2\,s^{-3}}$ in individual $q$. Individual
platform speeds reach $1.297\ \mathrm{m\,s^{-1}}$, so the proposed
$1\ \mathrm{m\,s^{-1}}$ speed-mask sensitivity is potentially consequential.
These observations are retained and flagged; the baseline does not trim them.

## Specification status

There are no known departures from the agreed Stage 2 numerical specification.
The fixed-anchor sensitivity behavior is implemented and tested, but the
reconstruction, source-gap, speed-mask, and changed-activation experiments were
not executed in this linear-only pilot. The next scientific decisions remain
the input gap review, a defensible minimum reliable separation, primary
linear-versus-spline choice, and Stage 3 bootstrap and influence settings.
