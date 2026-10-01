# Running the complete workflow

This is the operational runbook for processing one ARCTERX drifter dataset from
raw MATLAB files through reconstructed trajectories and optional candidate pairs.

Run commands from the repository root. Raw inputs are read-only. Local YAML
files, QC products, figures, Parquets, and Zarr stores are intentionally ignored
by Git.

## 1. Install and select a dataset

Create or update the project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,plotting]"
```

Activate it if command-line entry points are not already available:

```powershell
.\.venv\Scripts\Activate.ps1
```

The current local configurations are named consistently. Select either dataset in
PowerShell:

```powershell
$dataset = "microSVP"  # or "SVP"
$configRoot = "configs/arcterx"

$drogueConfig = "$configRoot/01_${dataset}_drogue_detection.local.yml"
$positionConfig = "$configRoot/02_${dataset}_position_qc.local.yml"
$reconstructionConfig = "$configRoot/03_${dataset}_trajectory_reconstruction.local.yml"
$pairConfig = "$configRoot/04_${dataset}_pairs.local.yml"
```

Before running, check every input and output path. Each dataset must have separate
output directories and review CSVs. Relative paths are resolved relative to the
YAML file, not the current shell directory.

### Quick-reference reviewed run

After defining the variables above, the normal review-oriented sequence is:

```powershell
# 1. Detect drogue loss and resolve problematic cases.
drifterlab-drogue $drogueConfig --automatic
drifterlab-drogue $drogueConfig --semiautomatic

# 2. Build conservative position QC and finish its review queue.
drifterlab-position-qc $positionConfig --semiautomatic

# 3. Reconstruct all accepted trajectories.
drifterlab-reconstruct-trajectories $reconstructionConfig

# 4. Build candidate pairs when needed.
drifterlab-pairs $pairConfig
```

GUI review commands return only after the review window closes.

## 2. Drogue detection and review

First create the automatic evidence table:

```powershell
drifterlab-drogue $drogueConfig --automatic
```

Then resolve only problematic or stale cases in the GUI:

```powershell
drifterlab-drogue $drogueConfig --semiautomatic
```

For an exhaustive human pass over platforms that do not yet have a current
review row, use:

```powershell
drifterlab-drogue $drogueConfig --manual
```

The modes share one automatic Parquet and one review CSV:

- `--automatic` runs without a GUI. It refuses to replace an existing automatic
  table unless `--overwrite` is supplied.
- `--semiautomatic` reuses the automatic table and opens unresolved, conflicting,
  or stale cases only.
- `--manual` reuses the automatic table and opens every undecided or stale
  platform.
- `--mode automatic|semiautomatic|manual` is an alternative drogue-only spelling.
  Do not combine it with the corresponding flag.
- `--overwrite` recomputes the automatic table. It does not delete human review
  rows, although changed evidence can correctly make them stale.

Main YAML controls include the raw reader and file pattern, cutoff margin,
TTFF binning/persistence rules, strain aggregation and change-point thresholds,
and cross-signal agreement tolerance. See
[drogue-loss detection and review](drogue_loss_review.md) for the scientific
definitions.

Do not proceed until every platform has a resolved drogue status. Position QC
preserves uncertainty rather than silently guessing a cutoff.

## 3. Native-position QC

Choose one position-QC policy; these are alternatives, not three consecutive
processing stages.

For aggressive deterministic processing without a GUI:

```powershell
drifterlab-position-qc $positionConfig --automatic
```

For conservative automatic decisions followed by a queue of unresolved events:

```powershell
drifterlab-position-qc $positionConfig --semiautomatic
```

For the conservative policy with full review navigation:

```powershell
drifterlab-position-qc $positionConfig --manual
```

Mode behavior:

- `--automatic` uses the aggressive deterministic resolution policy and never
  opens a GUI.
- `--semiautomatic` uses the conservative policy and opens pending temporal,
  timing, geometry, and decision-conflict events.
- `--manual` uses the same conservative policy and exposes completed reviewable
  history as well as the pending queue.
- `--overwrite` forces per-platform recomputation but preserves the position
  review CSV. Without it, compatible per-platform products are reused.
- Changing between automatic and conservative modes changes the recorded
  resolution policy, so affected products are rebuilt even without `--overwrite`.

Important YAML controls are:

- `deployment.metadata`: optional exact deployment times or uncertainty windows;
- `temporal_segments`: gap, dominant-segment, and detached-fragment rules;
- `position_qc.minimum_local_dt_seconds` and `max_local_gap_seconds`: the timing
  range eligible for local geometry decisions;
- `position_qc.speed_threshold_m_s`: physical speed threshold;
- `position_qc.max_automatic_removal_points`: bounded repair size;
- local baseline, endpoint, bridge-warning, and ambiguity thresholds;
- reviewer context and event-merging settings.

The result is one schema-4 Parquet per platform plus a run summary and a separate
atomic review CSV. The Parquets keep raw coordinates and QC decisions for every
observation between the first and last QC-valid timestamps, including rejected
points within that span. A platform with no valid observations has an empty file
with full QC counts and completeness in its footer. The summary reports full-record
decisions and exported/trimmed counts separately. See [native-position QC](native_position_qc.md).

Reconstruction accepts only complete position-QC products. If it reports an
unresolved platform, return to `--semiautomatic` or `--manual`, finish the review,
and rerun position QC so the affected platform Parquet is republished.

## 4. Common-grid trajectory reconstruction

Build the linear and spline coordinate representations:

```powershell
drifterlab-reconstruct-trajectories $reconstructionConfig
```

The command publishes an atomic bundle containing `trajectories.zarr` and
`build_report.csv`, followed by configured figures. A normal rerun validates and
reuses an unchanged scientific bundle. Use this only when intentionally replacing
it:

```powershell
drifterlab-reconstruct-trajectories $reconstructionConfig --overwrite
```

Key YAML options are:

- `grid.start_time`: UTC validation boundary and phase anchor;
- `grid.end_time`: `null` for the latest eligible input, or an explicit on-grid
  UTC bound that cannot omit eligible data;
- `grid.dt_minutes`: common output cadence;
- `spline.period_minutes`: spline phase-ensemble representations to publish;
- `spline.long_gap_threshold_minutes`: per-period fallback threshold;
- `arrays.maximum_adjacent_start_gap_hours`: a strictly larger chronological gap
  between exact first retained-QC timestamps starts the next numeric array;
- `initial_clustering.distance_reference`: `observed_starts` for deployment-group
  inference, or `first_common_grid` for time-aligned reconstructed positions;
- `initial_clustering.coordinate_method`: reconstructed coordinates used for the
  first-common-grid assignment mode and diagnostic (`native`, `linear`, or a
  configured spline);
- `initial_clustering.defaults.maximum_start_time_difference_seconds`: maximum
  exact observed-start difference for a candidate link and complete cluster;
- `initial_clustering.defaults.maximum_pair_distance_m`: candidate-link distance
  under the selected distance reference;
- `initial_clustering.defaults.maximum_cluster_diameter_m`: maximum distance
  across every pair in a proposed cluster, preventing nearest-neighbor chaining;
- `initial_clustering.defaults.maximum_members`: inferred-cluster cap, or `null`
  for no cap;
- `initial_clustering.array_overrides.array_NNN`: replacements for any defaults,
  or `assignment: single_cluster` for an intentionally whole-array group;
- output chunk sizes;
- optional plotting method, time window, platform subset, per-platform checks,
  map projection/extent, and compatible additional stores.

Every platform receives one positive, chronological `array_id`, a deterministic
`cluster_id`, and a cluster-local `member_id` before publication. Initial cluster
assignment does not trim trajectories. No method extrapolates before the first or
after the last retained QC fix. Review
`build_report.csv` and the figures before treating the candidate Zarr as a frozen
scientific input. Full details are in
[trajectory reconstruction](trajectory_reconstruction.md).

Each `starting_positions__array_XX.png` map uses color for elapsed time from the
array's first observed start and marker shape for candidate cluster number.

## 5. Candidate pairs

Pair construction does not depend on deployment-array assignments. A platform may
belong to several pairs.

Set the following scientific choices in the pair YAML:

```yaml
coordinates:
  selection_method: spline_30
selection:
  maximum_distance_m: 200
  maximum_seconds_from_each_observed_start: 21600
```

Then run:

```powershell
drifterlab-pairs $pairConfig
```

Options and behavior:

- `selection_method` selects pair membership and encounter time. All complete
  linear/spline coordinate representations are still retained in the output.
- `maximum_distance_m` is required, finite, and positive. An encounter is the
  first common stored timestamp at or below this WGS84 distance.
- A finite `maximum_seconds_from_each_observed_start` must hold independently
  from both starts.
- Setting only the time option to `null` enables full-overlap chance encounters;
  it does not disable the distance threshold.
- `--overwrite` atomically replaces an existing pair bundle.

The output contains authoritative grouped trajectories in `pairs.zarr` and a
compact `pair_catalog.csv`. See [candidate encounter pairs](candidate_pairs.md).

## Command-line option summary

| Command | Mode/option | Effect |
|---|---|---|
| `drifterlab-drogue` | exactly one of `--automatic`, `--semiautomatic`, `--manual`, or `--mode ...` | Select detection/review mode |
| `drifterlab-drogue` | `--overwrite` | Rebuild automatic evidence; retain review CSV |
| `drifterlab-position-qc` | exactly one of `--automatic`, `--semiautomatic`, `--manual` | Select aggressive or conservative QC policy |
| `drifterlab-position-qc` | `--overwrite` | Force per-platform recomputation; retain review CSV |
| `drifterlab-reconstruct-trajectories` | `--overwrite` | Validate a replacement before atomically replacing the bundle |
| `drifterlab-pairs` | `--overwrite` | Atomically replace the pair bundle |

Every command supports `--help`, for example:

```powershell
drifterlab-position-qc --help
```

## Safe reruns and troubleshooting

- Do not use `--overwrite` routinely. Read the reported incompatibility or stale
  input first, then rebuild deliberately.
- Review CSVs are durable decisions and are separate from regenerated automatic
  products.
- Reconstruction and pair publication is atomic: a completed existing
  bundle is not replaced until the new bundle validates.
- If an executable is missing after adding a new CLI, rerun the editable install
  command from section 1.
- If plotting imports fail, install the `plotting` extra.
- If a review GUI cannot open, verify that the session has a graphical display;
  use automatic modes only when their different decision policy is intended.
- The SVP raw inventory mixes approximately five-minute and hourly native
  cadence. Its local QC currently mirrors the MicroSVP thresholds; inspect the
  position summary and large-gap diagnostics before accepting that policy for
  final production.
