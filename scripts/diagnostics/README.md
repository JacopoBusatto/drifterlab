# Optional ARCTERX diagnostics

These repository scripts preserve exploratory timing, sustained-start, and residual
jump checks for troubleshooting. They are outside the installed `drifterlab`
package and are not invoked by the preprocessing CLI or package API.

Downstream broad timing populations use `first_valid_position_time_qc` from the
production master/inventory. All 150 sustained-start estimates equalled that
quantity, so no additional inferred-start field or processing step is needed.
Position, drogue and analysis validity, gap information, and residual-jump flags
remain part of the unchanged production QC output.

## Raw drogue-signal characterization

`drogue_signals.py` reads only raw MicroSVP `ObsTimestamp`, `GpsTTFF`, `Drogue`,
and `HullTemperature`. Requested per-drifter figures run the configured standalone
detector and overlay its TTFF, strain, and final automatic dates. The script does
not read a QC Zarr, review table, or reference loss dates.

Generate one figure plus a one-row summary:

```powershell
python scripts/diagnostics/drogue_signals.py `
  configs/arcterx/drogue_detection.local.yml `
  --platform 300534061906090
```

Generate population summaries and the compact population figure, without 150
individual figures:

```powershell
python scripts/diagnostics/drogue_signals.py `
  configs/arcterx/drogue_detection.local.yml
```

Explicitly generate figures for every drifter:

```powershell
python scripts/diagnostics/drogue_signals.py `
  configs/arcterx/drogue_detection.local.yml `
  --all
```

Outputs are written under `data/diagnostics/drogue_signals/`: population Parquet
and CSV tables, high-level and raw-field metadata JSON, an aggregate distribution
figure, and predictable `figures/drogue_signals_<platform>.png` files when figures
are requested. The detector parameters come from the YAML `detection` section;
rolling excursion, strain-delta, temperature, minimum-count, threshold, and
output settings remain CLI options. Use `--help` for the complete list.

## Run other diagnostics when needed

Activate the repository environment and run from the repository root. Install the
optional plotting dependency with `python -m pip install -e ".[diagnostics]"`,
or pass `--no-figures` to write only tables and reports.

```powershell
python scripts/diagnostics/timing_diagnostic.py `
  --config configs/arcterx/microsvp_preprocessing.local.yml `
  --raw-directory "C:/path/to/ARCTERX/Data/Raw/MicroSVP" `
  --output-directory data/diagnostics

# Requires the existing master and raw/QC timing table:
python scripts/diagnostics/sustained_start.py

# Diagnose existing >3 m/s native QC flags; the master is read-only:
python scripts/diagnostics/residual_jumps.py
# Add --qc-directory "C:/path/to/ARCTERX/Data/Quality Controlled/MicroSVP"
# to verify flagged endpoints against the original QC MATLAB files.

# Classify local geometry around the existing flags:
python scripts/diagnostics/local_jump_classification.py

# Compare native flags with the supplied 30m/60m reconstructed tracks:
python scripts/diagnostics/interpolation_comparison.py
```

`python -m scripts.diagnostics.timing_diagnostic`,
`python -m scripts.diagnostics.sustained_start`, and
`python -m scripts.diagnostics.residual_jumps` are equivalent module commands;
the local classifier is also available as
`python -m scripts.diagnostics.local_jump_classification`, and the interpolation
comparison as `python -m scripts.diagnostics.interpolation_comparison`. These are module commands
from the repository root. All scripts accept `--help`.

Generated artifacts already live in the Git-ignored `data/diagnostics/` directory.
Rerunning a script replaces its own diagnostic outputs; it does not rewrite the
master Zarr, inventory, source data, or QC definitions. Existing reports and hashes
are retained as records of their original runs.

Residual-jump outputs live in `data/diagnostics/residual_jumps/`.
Local classification outputs are nested under `residual_jumps/local_classification/`.
Interpolation-comparison outputs are nested under
`residual_jumps/interpolation_comparison/`. Use repeatable `--platform-code` options
for a partial troubleshooting run; omit them for the complete campaign diagnostic.
See [timing diagnostic methods and findings](arcterx_timing_diagnostic.md),
[sustained-start methods and findings](arcterx_sustained_start.md), and
[residual-jump methods and findings](arcterx_residual_jumps.md). The generated local
classification report documents its rules and findings in full.
Their existing tests remain in the full `python -m pytest` suite.
