# Poje trajectory-method reproduction

`drifterlab-poje` derives trajectory-only products from a completed immutable
all-array epsilon run. It does not edit or rerun the source product. The current
implementation reproduces the trajectory calculations behind Poje et al. (2017)
Figures 4--7. Figures 6--7 contain the ARCTERX observational analogues only;
ancillary AVISO products are intentionally outside scope.

Run the local ARCTERX configuration with:

```powershell
drifterlab-poje configs/arcterx/10_microSVP_poje.local.yml
```

The output is content-addressed and organized as:

```text
poje-v1-<source-run-id>/
  poje-v1-<analysis-id>/
    config_resolved.yaml
    manifest.json
    summary.csv
    array_003/
      dispersion.csv
      pairs.csv
      separation_memory.csv
      lagrangian_structure.csv
      increment_distributions.csv
      increment_statistics.csv
      structure_functions.csv
      data/
        pair_dispersion.parquet
        separation_memory_samples.parquet
        figure_06_pair_observations.parquet
      figures/
        figure_04_dispersion.png
        figure_05_self_similarity.png
        figure_06_increment_distributions.png
        figure_07_structure_functions.png
```

`summary.csv` contains one row per array and the fitted descriptive constants.
`dispersion.csv` contains the complete time series and support. `pairs.csv`
lists the fixed initial pair cohort. Complete pair-time observations remain in
Parquet. `separation_memory.csv` contains the scale-independent Figure 5a
curves, while `lagrangian_structure.csv` contains Figure 5b and its support.
The individual target-time/lag correlations underlying Figure 5a remain in
`separation_memory_samples.parquet`. `increment_distributions.csv` contains the
Figure 6 histogram curves and fitted Gaussian densities;
`increment_statistics.csv` contains scale selection, support, standard
deviation, raw and central moments, Gaussian-fit parameters, and tail counts. The exact selected pair-time
samples remain in `figure_06_pair_observations.parquet`.
`structure_functions.csv` contains the complete Figure 7 second-order,
four-fifths, mixed third-order, Rossby, support, and confidence-interval values.

See [methodology](methodology.md) for formulas and [figure guide](figure_guide.md)
for panel interpretation.
