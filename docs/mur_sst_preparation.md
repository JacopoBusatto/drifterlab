# Preparing MUR L4 for affine-deformation maps

MUR support is a standalone conversion step. The affine analysis continues to
read the same prepared Celsius SST interface used by the VIIRS products, with
no MUR-specific variable names or masks in the analysis reader.

The [NASA/JPL product description](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1)
identifies MUR v4.1 as a 0.01-degree gridded L4 foundation-SST analysis using
optimal interpolation. It is an analysed field rather than a direct image of
all observations on that day. Filled coverage does not establish locally
observed coverage, and nominal grid spacing does not guarantee that every
feature is resolved at that scale. Foundation SST also differs from the
subskin SST represented by the VIIRS products; comparisons require that context.

## Prepare the downloaded subsets

Install the optional NetCDF4 support if needed: `pip install -e ".[sst]"`.
From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.prepare_mur_sst --input "C:\Users\Jacopo\Documents\DATI\MUR_L4" --output "C:\Users\Jacopo\Documents\DATI\MUR_L4\MUR_daily_20250201_20250331.nc" --start 2025-02-01 --end 2025-03-31
```

The installed command is `drifterlab-prepare-mur-sst` with the same arguments.
Dates are inclusive and refer to the nominal UTC analysis date. Omitting dates
uses the earliest/latest available nominal dates. The converter reads only
local `*-L4_GHRSST-SSTfnd-MUR-*.nc` files; it never downloads data. Existing
prepared output is refused unless `--overwrite` is explicitly supplied.
Replacement occurs only after the new file is complete. Raw subsets are never
modified. Missing requested dates remain explicit NaN records; rerun preparation
after more downloads arrive if the original requested period was incomplete.

## Conversion and quality policy

The converter:

- validates a consistent MUR v4.1 identity, one analysis per nominal date and
  identical rectilinear coordinates across files;
- decodes packed fill values, valid ranges, scale and offset before converting
  `analysed_sst` from Kelvin to Celsius;
- writes `sea_surface_temperature(time, lat, lon)` without regridding, extra
  interpolation, temporal averaging or bias correction;
- requires the open-sea mask bit 1 and rejects land, lake and ice-flagged bits
  2, 4, 8 and 16. This is the MUR mask schema, not a VIIRS flag mask;
- retains `mask`, `sea_ice_fraction`, `analysis_error`, `dt_1km_data` and a
  prepared `valid_mask` for inspection;
- records per-source SHA-256 hashes, file identities, nominal times and
  original coverage in `source_provenance_json`.

`analysis_error` is an uncertainty, so its decoded Kelvin values are retained
without subtracting 273.15. No uncertainty ceiling is imposed by default.
`--maximum-analysis-error-k 0.5`, for example, excludes cells whose uncertainty
is missing or above 0.5 K; the chosen threshold is recorded. That is an optional
user policy, not a recommended scientific cutoff.

The original `dt_1km_data` values and missing-value mask are retained. Missing
nearby 1 km input does not automatically invalidate an interpolated L4 value.
No observation counts are invented, and the converter applies no threshold to
data age. The global MUR `file_quality_level` is not treated as a VIIRS
per-pixel `quality_level`.

## Time assignment is explicit

The inspected inputs have nominal analyses at 09:00 UTC. For example, the
February 2 analysis declares source coverage from February 1 at 21:00 through
February 2 at 21:00. Those source bounds cross the affine maps' UTC midnight
boundaries.

The prepared product therefore distinguishes:

| Field | Meaning |
| --- | --- |
| `time` | Original nominal analysis time, e.g. February 2 at 09:00 UTC |
| `time_bounds` | Calendar-day **map-assignment** bin, e.g. February 2 00:00–February 3 00:00 |
| `source_time_bounds` | Original declared source coverage, e.g. February 1 21:00–February 2 21:00 |

This deliberately places a daily analysis on the map labelled by its nominal
date. It does not claim that MUR is a UTC daily mean or relabel the source
coverage as calendar-day observations. The policy is recorded in variable and
global metadata. The generic reader selects this one prepared daily record
unchanged. No splitting, daily averaging or duplication into hourly fields
occurs. Captions retain the prepared product title so the background can be
identified when comparing different SST sources.

## Run the Array 3 analysis

The separate config
`configs/arcterx/v2/microSVP/array_03/06a_microsvp_affine_qc_5min_mur.local.yml`
uses the prepared MUR file and its own output root:

```powershell
.\.venv\Scripts\python.exe -m drifterlab.cli.analyze configs\arcterx\v2\microSVP\array_03\06a_microsvp_affine_qc_5min_mur.local.yml
```

For other analysis configs, set only the prepared SST path and use a separate
output root to preserve comparisons:

```yaml
sst:
  enabled: true
  file: C:/Users/Jacopo/Documents/DATI/MUR_L4/MUR_daily_20250201_20250331.nc
  minimum_celsius: 24
  maximum_celsius: 26.5
```

Spatial/temporal coverage counters in the affine outputs describe availability
of the prepared analysis bins. For L4 they do not measure satellite sampling
coverage. MUR uncertainty and data-age fields remain available in the prepared
file for a separate uncertainty assessment; they do not enter the affine fit.

Tests: `python -m pytest tests/test_prepare_mur_sst.py tests/test_prepared_sst.py -q`.

Validation on 2026-10-23: 57 affected tests passed. All 59 downloaded subsets
were converted for February 1–March 31 on their original 1501×1001 grid.
A 30-hour Array 3 pilot produced two inspected MUR-backed maps and the same
affine diagnostics as the VIIRS pilot (64 attempted intervals, 39 numerically
valid fits). All nine published pilot-output hashes were checked. This was a
pipeline/rendering check, not a full-campaign validation of MUR fronts or
uncertainty estimates.
