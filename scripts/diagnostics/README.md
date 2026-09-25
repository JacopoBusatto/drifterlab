# Optional ARCTERX diagnostics

These repository scripts preserve timing, sustained-start, and residual-jump
checks for troubleshooting unrelated ARCTERX processing. They are outside the
installed `drifterlab` package and are not invoked by the preprocessing or
drogue CLIs.

Activate the repository environment and run from the repository root:

```powershell
python scripts/diagnostics/timing_diagnostic.py `
  --config configs/arcterx/microsvp_preprocessing.local.yml `
  --raw-directory "C:/path/to/ARCTERX/Data/Raw/MicroSVP" `
  --output-directory data/diagnostics

python scripts/diagnostics/sustained_start.py

python scripts/diagnostics/residual_jumps.py
# Add --qc-directory "C:/path/to/ARCTERX/Data/Quality Controlled/MicroSVP"
# to verify flagged endpoints against the original QC MATLAB files.

python scripts/diagnostics/local_jump_classification.py
python scripts/diagnostics/interpolation_comparison.py
```

The corresponding `python -m scripts.diagnostics.<name>` commands are also
available. Generated artifacts live in the Git-ignored `data/diagnostics/`
directory and never rewrite source data, the master Zarr, or QC definitions.

See [timing diagnostic methods and findings](arcterx_timing_diagnostic.md),
[sustained-start methods and findings](arcterx_sustained_start.md), and
[residual-jump methods and findings](arcterx_residual_jumps.md).
