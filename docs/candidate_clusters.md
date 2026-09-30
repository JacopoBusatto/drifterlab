# Candidate deployment clusters from observed starts

## Scope

This stage infers reviewable candidate groups from the reconstructed Zarr's exact
`start_time`, `start_lon`, and `start_lat` platform variables. They are the first
positions retained by position QC and are deliberately called **observed starts**.
They are not verified deployment positions, and QC may have removed earlier fixes.

The input Zarr is opened read-only. No trajectory position, timestamp, or valid
span is copied, shortened, or changed. Candidate grouping does not require common
lifetime overlap and does not compare post-start motion. Later pair analyses are
responsible for selecting each pair's actual overlapping lifetime.

## Inspect and review

Copy the tracked configuration, set the input and output paths, and run:

```powershell
Copy-Item configs/arcterx/clusters.yml configs/arcterx/clusters.local.yml
drifterlab-clusters configs/arcterx/clusters.local.yml --inspection
```

Inspection sorts exact observed starts and proposes a new broad cohort whenever
the adjacent start gap is greater than
`cohorts.maximum_adjacent_start_gap_hours`. These are proposed observed-start
cohorts, not inferred deployment events.

The atomic `inspection/` bundle contains only:

- `cohort_review.csv`, one row per platform with its exact start, adjacent time
  gaps, proposed/reviewed cohort, inclusion field, map marker, and note;
- `neighbor_diagnostics.csv`, spatial neighbor ranks with WGS84 geodesic distance
  and the associated absolute observed-start gap;
- one quick-look PNG and one multipage diagnostic PDF per cohort;
- `manifest.json` with configuration and input-start provenance.

Each PDF contains the separately scaled cohort map, its full platform/time key,
neighbor-rank distance distributions, and distance-versus-time diagnostics. The
rank plots are empirical diagnostics only; they do not label measurements as
within-cluster or between-cluster.

Review `reviewed_cohort_id`, `include`, and `review_note`. Do not change the
platform identity, observed values, or `observed_start_sha256`. Point
`cohorts.reviewed_assignments` at that reviewed file. A changed reconstructed
start makes the review stale and stops grouping.

## Build candidate groups

Set finite positive values for:

```yaml
grouping:
  maximum_cluster_diameter_m: 1000
  maximum_members_per_cluster: 5
  maximum_observed_start_spread_minutes: 120
  cohort_overrides: {}
```

The numerical values above illustrate syntax only; choose them from the inspection
outputs. Overrides may replace any value for one reviewed cohort.

Within each cohort, deterministic complete linkage uses the larger of the
normalized spatial diameter and normalized observed-start spread. A merge is
allowed only if both remain at or below their configured maximum and membership
does not exceed its maximum. This prevents spatial or temporal chaining. The
method never forces a target number of clusters or a target group size;
singletons remain visible candidates.

Run grouping with:

```powershell
drifterlab-clusters configs/arcterx/clusters.local.yml
```

The atomic `candidate/` bundle contains membership, cluster summaries, the merge
audit, provenance, and one PNG/PDF map pair per reviewed cohort. An existing mode
bundle is preserved unless `--overwrite` is explicit.
