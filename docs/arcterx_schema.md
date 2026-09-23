# Inspected ARCTERX QC MicroSVP schema

The first implementation was grounded in all 150 local QC MicroSVP files and the
supplied **Arcterx Data Guide.docx**. The workflow prompt's `QualityControlled`
directory spelling differs from the actual **Quality Controlled** directory.
The package uses configured paths and makes no assumption about its location.

All 150 files are MATLAB Level 5 files, containing scalar structures `drifter`
and `drifter_interp`. SciPy's simplified reader removes singleton MATLAB
dimensions; the shapes below describe the resulting scalars and vectors.
The two field-name sets are consistent across all files.

## Field mappings

| Source field | Shape | Master variable |
|---|---|---|
| `drifter.PlatformId` | scalar integer | `platform_code` (exact string) |
| `drifter.ID` | scalar integer | `source_id` (exact string) |
| `drifter.type` | scalar string | `instrument_type` |
| `drifter.time` | native vector | `time_qc`, `time_matlab_qc` |
| `drifter.longitude`, `latitude` | native vectors | `lon_qc`, `lat_qc` |
| `drifter.SST`, `SLP` | native vectors | `sst_qc`, `slp_qc` |
| `drifter.battery` | native vector, except six mismatches | `battery_qc`; mismatches preserved in auxiliary group |
| `drifter.drogue` | native vector | `drogue_strain_qc` |
| `drifter.speed` | native vector | `speed_qc_source` |
| `drifter.drogue_off` | scalar datenum or `-999` | `drogue_off_time`, `drogue_off_matlab` |
| `drifter_interp.time` | reconstructed vector | `time_interp`, `time_matlab_interp` |
| `drifter_interp.longitude_30min`, `latitude_30min` | reconstructed vectors | `lon_interp_30m`, `lat_interp_30m` |
| `drifter_interp.longitude_60min`, `latitude_60min` | reconstructed vectors | `lon_interp_60m`, `lat_interp_60m` |
| `drifter_interp.SST` | reconstructed vector | `sst_interp` |
| `drifter_interp.speed_30min`, `speed_60min` | reconstructed vectors | `speed_interp_30m_source`, `speed_interp_60m_source` |

`ID` and `PlatformId` agree in the inspected dataset. Source observations include
both floating-point and integer arrays; converting sentinels therefore requires
promotion before assigning NaN. SLP is present but entirely missing.

## Time and representation findings

Every native and reconstructed time vector is in descending source order. There
are no duplicate timestamps in the inspected inputs. Native and reconstructed
time axes differ in every file. Some mathematically near-identical clock times
also differ slightly in their floating-point MATLAB encoding; ingestion does not
round or join them.

Both reconstructed position methods use the **same supplied** `drifter_interp.time`
vector, at nominal five-minute resolution. The reconstruction procedures derive
from 30-minute and 60-minute subsampling, spline fitting and averaging over start
phases. These descriptions come from the campaign guide, not from inference based
on apparent smoothness.

UTC is a campaign convention explicitly confirmed by the user; MATLAB datenums
do not encode timezone information. The adapter applies this interpretation
without shifting source clock values. Generic MATLAB conversion has no timezone
assumption. No velocity-derivation algorithm or SST interpolation method is
inferred from the supplied arrays, and undocumented sensor units are not assigned.

## Observed exceptions and audit baseline

| Quantity | Inspected value |
|---|---:|
| Files / unique platforms | 150 / 150 |
| Native rows | 3,755,180 |
| Reconstructed rows (shared by both methods) | 3,803,923 |
| Native lengths | 850–36,477 |
| Reconstructed lengths | 860–36,829 |
| Missing native positions | 1,808 |
| Available 30-minute / 60-minute products | 150 / 150 |
| Source loss estimates / unknown | 144 / 6 |
| Mismatched battery vectors | 6 |
| Recomputed consecutive native speeds > 3 m/s | 1,307 |
| Out-of-range reconstructed 30-minute positions | 1,936 |
| Out-of-range reconstructed 60-minute positions | 3,627 |

The completed local conversion reproduced these counts. Independent readback
compared all 2,100 observational source arrays across all 150 files after sorting
and sentinel normalization, plus time vectors, original row indices, input hashes,
validity masks, and loss cutoffs. All comparisons passed. The local
`data/verification.json` records the validation counts and dependency versions;
this report is ignored by Git along with the data products.

The 32 synthetic tests also passed with warnings treated as errors in an isolated
environment without `kinematicParcels`. A standalone wheel was built and checked
for all package modules and the CLI entry point.

Supplied native speeds remain below 3 m/s, while the independent consecutive-fix
audit finds exceedances in 140 drifters. This does not establish that the source
speed algorithm and the audit algorithm are equivalent. Both are preserved with
separate names; no extra destructive QC is applied.

Two trajectories contain extreme reconstructed coordinates. These values remain
in the master with false representation-specific position masks. In the inspected
data, those out-of-range reconstructed samples occur after the buffered loss cutoff.
The code does not depend on that coincidence.

Battery mismatches have different excess lengths and no supplied battery-only time
axis. They are not truncated, reordered, or aligned to raw data. The original
normalized vectors are retained in `source_unaligned/battery`; source counts and
warnings identify affected trajectories. The inventory is the authoritative
per-file detail, including hashes and original field shapes/types after singleton
dimensions have been removed.

## Policy and compatibility

The 24-hour exclusion before loss follows the preprocessing requirements. The
guide describes source drogue-loss timing as approximate, roughly within 1–2 days.
Unknown loss dates remain unknown and are excluded from drogue-required analyses
by default; reviewed retained/manual-loss decisions can be supplied in the CSV.

No `lon`/`lat` alias chooses a preferred trajectory. The output deliberately does
not claim to be a drop-in Parcels dataset or assert unverified CF compliance.
Future grouping and analysis code must explicitly select its representation and
use the associated time axis and masks.
