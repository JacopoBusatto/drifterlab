# Codex prompt: staged ARCTERX epsilon analysis

Work in the existing `drifterlab` repository. Implement the first ARCTERX analysis of signed third-order longitudinal velocity statistics versus instantaneous pair separation. Follow the stages below so I can inspect intermediate results and understand the implementation.

Read the applicable `AGENTS.md` instructions first. Inspect the actual code, configuration, documentation, and available data; resolve variable names and entry points from the repository rather than assuming a schema. Reuse suitable components and keep the changes focused on this analysis.

**Start with Stage 1 only.** Work one stage per turn and wait for my instruction to proceed to the next stage. Within an authorized stage, resolve routine implementation choices yourself and explain them. Keep a short progress record identifying completed work, validation, open scientific choices, and the next stage.

## Scientific objective and initial scope

For each experimental array, assess the sign, magnitude, scale dependence, sampling support, and variability of a signed energy-transfer diagnostic. Compare individual deployment clusters where their separation-scale coverage overlaps, and calculate an array-wide result including pairs connecting different clusters.

The first production windows are the first **6, 12, 18, and 24 elapsed hours of each pair's common record**. Support arbitrary elapsed intervals in configuration so later analyses can use 6–12 h, 12–18 h, and other windows. Run the four cumulative windows first. Treat 36/48 h, moving windows, absolute UTC windows, and spatial regime classification as later extensions.

Keep experimental arrays separate. Never form pairs between arrays or pool their statistics. Use every eligible unordered pair inside an array, including pairs that approach one another. Do not restrict the input to encounter-selected pairs, initially close pairs, or intervals of increasing separation.

Use existing accepted array and cluster assignments. Rebuilding deployment clusters is outside this task. An algorithmic singleton contributes to array-wide and between-cluster populations but has no internal pair statistic.

## Definitions and exact estimator

Use positions and velocities in a consistent metric coordinate representation. At the same UTC time for platforms i and j:

$$
\mathbf r_{ij}=\mathbf x_j-\mathbf x_i,
\qquad \delta_{ij}=|\mathbf r_{ij}|,
\qquad \Delta\mathbf u_{ij}=\mathbf u_j-\mathbf u_i,
$$

$$
\Delta u_{L,ij}=\Delta\mathbf u_{ij}\cdot\frac{\mathbf r_{ij}}{\delta_{ij}}.
$$

Retain the signed cube `(delta_u_L)**3`; do not use absolute increments or replace the third moment with a median.

Define the individual contribution:

$$
q_{ij}(t)=-\frac54\frac{[\Delta u_{L,ij}(t)]^3}{\delta_{ij}(t)}.
$$

For separation bin b, selected population P, and window W, use the primary estimator:

$$
S_{LLL}(b,P,W)=\left\langle(\Delta u_L)^3\right\rangle_{b,P,W},
$$

$$
\widehat\varepsilon_{\mathrm{eff}}(b,P,W)
=-\frac54\frac{S_{LLL}(b,P,W)}{\langle\delta\rangle_{b,P,W}}.
$$

The primary averages give equal weight to each valid pair-time observation on the declared common cadence. Record that weighting explicitly. Use the observed mean separation as the horizontal coordinate for this estimator; also retain the bin edges and nominal bin center.

For finite-width bins this is not generally the arithmetic mean of q. It is exactly the separation-weighted mean `sum(delta*q)/sum(delta)` over the same samples. Preserve this distinction in the code, saved tables, tests, plot labels, and documentation. Save the arithmetic mean of q separately if useful for describing its distribution.

Use metres for separation, m/s for velocities, and m²/s³ for q and epsilon. Label the coefficient as the **3D four-fifths-law convention** and save it in provenance. Positive epsilon_eff is a diagnostic consistent with downscale transfer and negative epsilon_eff with upscale transfer, subject to the theoretical and sampling assumptions. Physical viscous dissipation is nonnegative; these signed estimates are not direct local dissipation measurements.

For a single pair, `delta_u_L = d(delta)/dt`: approaching pairs have positive q and separating pairs negative q under this convention. Do not classify one pair's instantaneous sign as an established turbulent cascade regime. Do not claim that a plateau, sign change, or multimodal distribution alone proves a cascade.

## Pair timing, validity, and binning

- Use synchronized observations from the existing common time grid.
- Define each pair's t0 as its first joint valid position-and-velocity time after baseline validity processing. Retain that UTC timestamp and elapsed age `tau = time - pair_t0`.
- Select `[0, T)` in elapsed wall-clock hours for each pair. Do not restart the clock after a gap or stitch separated intervals into artificial continuous elapsed time.
- Retain only samples with valid simultaneous positions and velocities. Avoid derivatives through missing positions, unsupported gaps, or invalid reconstruction portions.
- Preserve the baseline pair clock for optional additional speed-mask sensitivities; masking extra samples must not reset t0 or shift the analysis window.
- Pair-relative windows do not share the same UTC bounds. Report the actual UTC coverage and the distribution of pair start times; do not describe these results as simultaneous array snapshots.
- Use available valid samples inside the requested window in the primary exploratory analysis. Report coverage relative to the requested full window, early record endings, and incomplete windows explicitly. Never relabel a truncated record as a completed T-hour observation.
- Make any minimum-coverage exclusion explicit and configurable. Avoid silently requiring every trajectory to survive for 24 hours.
- Assign samples to bins by **instantaneous separation**, not initial separation or a whole-window mean. A pair may contribute to many bins or revisit a bin.
- Reject zero/nonfinite separation and expose a configurable minimum reliable separation. Set the latter using positional accuracy and reconstruction limitations when those are known; do not invent a justified physical cutoff.
- Freeze one set of separation-bin edges for each array across all its clusters, cumulative windows, and reconstruction comparisons. Allow explicit configured edges. If pilot data are used to select logarithmic edges, save and reuse those edges rather than recalculating them separately for every plot.
- Report when scales and pair ages are strongly confounded. Missing scale coverage remains missing; do not interpolate the epsilon curve across unsupported bins.

## Reconstruction, velocity estimation, and QC

Select reconstruction methods through configuration using the actual store schema. Support the available linear and spline variants without changing the master trajectory product. Record the exact selected position variables, spline variant, cadence, coordinate representation, and derivative method.

Inspect existing velocity utilities first. Differentiate valid contiguous trajectory portions, and document endpoint treatment. Distinguish a derivative computed from sampled spline positions from an analytic spline derivative; use the latter only if the required spline representation is actually available.

Splines may attenuate some position noise but can preserve bad fixes or introduce overshoot. Verify the resulting velocity increments rather than treating the spline method as automatic QC.

I have proposed investigating a speed threshold of **1 m/s**. Treat this as an **optional sensitivity**, not a new default upstream QC rule or an established physical limit. The baseline uses accepted upstream QC. If enabled, mask affected observations consistently and rerun in a separate output identity. Report the removed fractions by platform, cluster, window, and separation bin. Do not modify source trajectories, trim large q values merely because they are large, or winsorize the third moment.

## Pair populations

Within each array produce:

1. Each deployment cluster separately: both members belong to that cluster.
2. All array pairs: every eligible pair, including between-cluster pairs.
3. Between-cluster pairs separately: members have different cluster IDs.

Retain each pair's two cluster IDs so contributions by cluster combination can be inspected later. Preserve both platform IDs and use stable unordered pair IDs.

Explain that large clusters and cluster combinations supply more pairs under the primary weighting. Compare cluster results only where scales overlap, and show sampling differences. Do not infer disagreement merely because clusters occupy different scales.

## Required plots

For each array, method, population, and cumulative window create one figure with three main panels and a small sampling-support panel:

1. **Scatter:** instantaneous q versus instantaneous separation. Use transparency. If necessary, use reproducible subsampling for display only; retain full data for statistics and histograms. Record the displayed and available counts.
2. **Mean estimate with error bars:** epsilon_eff from the estimator above, with dependence-aware 95% confidence intervals where support permits. Include a clear zero line. The intervals describe uncertainty in the estimate, not the spread of q.
3. **Conditional heatmap:** separation bins horizontally and signed q bins vertically. Normalize each separation column by its number of valid observations. Label the color as conditional probability mass per q bin, not observation count or a density per unit q unless bin-width normalization is actually performed.
4. **Sampling support:** unique pairs and pair-time observations per separation bin. Put platform counts and coverage information in the table and, where readable, the figure annotations.

Use logarithmic separation and a signed linear or symmetric-log vertical scale. Never take log(q), log(epsilon), or their absolute values in a way that removes the sign. Keep the three panels' separation limits aligned.

Use consistent q-bin edges, axis settings, and heatmap color normalization across comparable figures. Define histogram edges in physical q units. If q bins have unequal widths, make the probability-mass interpretation explicit. Represent missing columns distinctly from measured zero. Avoid undocumented clipping; if a display range excludes tails, report their probability mass and preserve the full histogram/data.

Also produce an epsilon comparison figure for each array and method showing cluster curves and the all-pairs curve across the four T values, using a readable arrangement. Keep between-cluster-only results accessible. Choose clear titles and legends rather than overcrowding one panel with every curve.

## Uncertainty and robustness

Adjacent observations and pairs sharing platforms are dependent. Do not compute confidence intervals by independently resampling every pair-time row or treating all rows as independent.

For the first implementation, use a documented synchronized UTC time-block bootstrap: resample blocks while preserving all selected pair observations together within each block. Apply it after the original pair-age/window selection; do not recompute ages on a synthetic reordered timeline. State that these intervals are conditional on the observed drifter population and do not capture every sampling bias.

Make block duration, bootstrap replicate count, and random seed configurable. Justify an initial block duration from the pilot data or label it provisional. Examine at least two plausible block durations. Report how many blocks contribute to each bin and suppress or flag intervals when temporal or platform support is inadequate. Do not manufacture precise error bars for a poorly sampled cluster.

Add leave-one-platform-out checks to identify estimates dominated by one drifter, and compare the all-pairs result after leaving out individual clusters where useful. Keep these checks distinct from confidence intervals. Reconstruction and optional speed-mask comparisons are sensitivities, not independent replications.

## Saved data and output organization

Use the configured output root and a structure equivalent to:

```text
outputs/epsilon_analysis/
  <position_method>/
    <velocity_method_and_cadence>/
      <run_id>/
        config_resolved.yaml
        manifest.json
        progress.md
        array_<id>/
          pair_observations/
          summaries/
          figures/
            all_pairs/
            between_clusters/
            cluster_<id>/
          diagnostics/
```

Choose a deterministic run identity that distinguishes materially different configurations, including speed masks, separation bins, derivative settings, and source versions. Validate provenance before reusing cached outputs. Avoid silent overwrites and support rerendering figures from compatible saved results.

Prefer existing repository formats. A partitioned Parquet pair-observation table is suitable if supported. Keep storage manageable by computing only the required pair-age span and processing time/pair chunks rather than allocating an unnecessary full platform-by-platform-by-time tensor.

Retain, directly or through a documented lookup: array ID, platform IDs, cluster IDs, pair t0, UTC, elapsed pair age, separation, delta_u_L, its signed cube, q, endpoint locations, and pair midpoint. These permit later tracing of contributors without treating a long pair's midpoint as a local dissipation measurement.

Save summary tables with: population and window, bin edges, mean separation, S_LLL, epsilon_eff, mean q, confidence bounds/status, unique pairs, platforms, unique UTC samples, time blocks, coverage, and exclusions. Save numeric histogram outputs and edges as well as figures. Report the range of pair ages contributing to each separation bin.

## Staged workflow and acceptance checks

### Stage 1 — Inspect and propose the concrete integration

Inspect the repository and input schema. Locate array/cluster labels, reconstructions, velocity utilities, configs, loaders, and plotting entry points. Check that loading analysis data does not inadvertently apply trajectory-plotting time windows or color settings. Review relevant scientific-plan passages and identify conflicts with this prompt without rewriting unrelated material.

Report the actual files/functions to reuse or change, input availability, proposed config/CLI entry point, proposed binning and pilot array/method, and any material missing information. Resolve routine choices with stated defaults. Do not implement or run the full analysis in this stage. End with the concrete Stage 2 work ready to begin.

### Stage 2 — Implement calculations and one-array pilot

After I request this stage, implement validity handling, velocities, pair clocks, pair observations, populations, windows, bin summaries, and provenance. Run a limited pilot using one actual array and one available reconstruction. Inspect scale coverage, velocity increments, record gaps, sample counts, and dominant contributions before selecting final exploratory bins.

Add focused scientific correctness tests: uniform translation gives zero increments; swapping pair labels leaves delta_u_L and q unchanged; approaching/separating examples have the expected signs; late starts and gaps preserve the correct pair clock; instantaneous binning is correct; epsilon matches the separation-weighted q identity; and pair-population counts are correct. Test units and invalid/zero-separation handling. Use repository-standard checks.

Report the pilot's numeric results and sampling limitations without asserting physical regimes prematurely. End with Stage 3 ready to begin.

### Stage 3 — Implement plots, uncertainty, and reviewable outputs

After I request this stage, implement the three-panel figures, support panel, comparison plots, saved histograms, synchronized block bootstrap, and influence checks. Generate and visually inspect actual pilot figures. Verify that signs, units, zero lines, normalization, legends, missing bins, and confidence-interval labels are correct. Provide exact output paths and explain how to read the pilot figures.

### Stage 4 — Run the first production comparison

After I request this stage, run the configured arrays, populations, methods, and four cumulative windows using frozen bin definitions. Run explicitly enabled sensitivity cases separately. Complete required checks and document supported scales, provisional signs, influential platforms/clusters, and limitations. Provide commands/configurations to reproduce or rerender the outputs and a concise progress summary.

The deliverable is an inspectable first analysis of epsilon_eff and its pair contributions. Spatial turbulent-regime classification and additional turbulence/dispersion diagnostics remain follow-up work.
