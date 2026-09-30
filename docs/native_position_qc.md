# Native-position QC v2

Native-position QC reads raw fixes, resolves their drogue/deployment/temporal
eligibility, diagnoses exact repeats and local geometry, and writes one logical QC
row for every immutable source observation. It does not reconstruct a trajectory,
alter source coordinates, or use supplied 30/60-minute tracks as evidence.

## Commands and configuration

Run drogue detection first, then position QC. All three position modes share the
same per-trajectory Parquet directory and review CSV:

```powershell
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --automatic
drifterlab-drogue configs/arcterx/drogue_detection.local.yml --semiautomatic

drifterlab-position-qc configs/arcterx/position_qc.local.yml --automatic
drifterlab-position-qc configs/arcterx/position_qc.local.yml --semiautomatic
drifterlab-position-qc configs/arcterx/position_qc.local.yml --manual
```

`--overwrite` forces deterministic position recomputation without deleting the
review CSV. Compatible trajectory files are otherwise reused independently. Relative paths are resolved
against the configuration file. The effective configuration is:

```yaml
input:
  reader: microsvp_mat
  directory: ...
  pattern: "*.mat"
  options:
    missing_value: -999

drogue:
  automatic: .../auto_drogue_loss.parquet
  review: .../drogue_review.csv

deployment:
  metadata: null

output:
  directory: .../position
  review: .../position_review.csv

temporal_segments:
  segment_gap_hours: 6
  fragment_max_observation_fraction: 0.10
  fragment_max_duration_fraction: 0.10
  minimum_main_duration_hours: 48
  minimum_main_observations: 48

position_qc:
  nominal_interval_seconds: 300
  nominal_interval_tolerance_seconds: 1
  minimum_local_dt_seconds: 150
  max_local_gap_seconds: 1800
  gap_tolerance_seconds: 1
  speed_threshold_m_s: 3.0
  max_automatic_removal_points: 5
  local_speed_window_points: 15
  endpoint_speed_min_samples: 8
  boundary_endpoint_speed_multiplier: 10.0
  bridge_speed_warning_z: 3.0
  local_speed_scale_floor_m_s: 0.05
  endpoint_speed_score_margin_z: 2.0

review:
  context_points: 12
  merge_gap_edges: 2
```

Recovery is bounded by both `max_automatic_removal_points` and elapsed time. The
interpolation residual remains evidence rather than a decision gate.
`bridge_speed_warning_z` controls the atypical-bridge warning threshold;
`endpoint_speed_score_margin_z` controls how clearly one conservative endpoint
cure must outperform another. `boundary_endpoint_speed_multiplier` reserves
automatic retained-segment boundary removal for gross jumps: its threshold is
the ordinary speed threshold multiplied by this value.

The optional deployment CSV must have these columns, in order:

```text
platform_code,deployment_time,deployment_window_start,deployment_window_end,provenance,note
```

Platforms are unique. Timestamps require explicit UTC. Each row contains either
one exact deployment time or both ends of one increasing uncertainty window.

## Eligibility and temporal segmentation

The workflow refuses a missing drogue platform or a raw-source hash mismatch.
For a resolved loss, only `time < analysis_cutoff_time` is eligible. A `not_lost`
decision admits the full valid-time record. An `uncertain` decision preserves all
rows, blocks the platform, and skips temporal/local automatic decisions.

An exact deployment time excludes earlier rows. For a deployment window, rows
before its start are outside scope, `[start, end)` remains unresolved, and rows at
or after the end are eligible. Invalid time or coordinates resolve as
`source_invalid` and never establish continuity.

Eligible observations are stably sorted by timestamp for calculations. Distinct
timestamps split only where a gap is strictly greater than `segment_gap_hours`;
duplicates share the segment but do not establish continuity. Segment duration is
last minus first time, cadence uses only positive internal intervals, and duration
fractions divide by the sum of segment durations (or remain null when it is zero).

The dominant segment is the unique maximum by observation count, with duration as
the tie-breaker. An exact tie leaves no dominant segment. A candidate fragment is
automatically excluded only when the dominant segment meets both configured
minimums and both candidate/main ratios are within their maxima. Tiny segments are
classified before, after, or between all substantial segments. If multiple
substantial segments remain, every substantial segment requires a retain/exclude/
uncertain review; local diagnostics may run there, but automatic point rejection
is deferred. Multiple retained segments remain independent. Excluding all segments
is complete QC with no reconstruction-available trajectory.

Time gaps cannot identify continuously sampled pre-deployment observations. Those
rows stay eligible unless deployment metadata or another explicit boundary exists.

## Repeats and local recovery

After temporal resolution, exact finite `(longitude, latitude)` groups are formed
across all eligible retained segments. Any group with at least two observations
automatically rejects every occurrence, including its first and nonconsecutive
occurrences. Rows outside the drogue/deployment/retained scope neither trigger nor
receive this decision. Exact-repeat rejection overrides old human keep, reject, or
uncertain point rows; those CSV rows are preserved for provenance but are no longer
effective. Repeat events and points are excluded from manual/semiautomatic event
and point navigation. They remain only as faint raw context if another event's map
window includes them, and never enter surviving map or velocity adjacency.

After repeat removal, each eligible retained segment is thinned chronologically
against the last surviving observation. A later observation is automatically
rejected when its elapsed time is strictly positive and less than
`minimum_local_dt_seconds`; the same predecessor remains the anchor for the next
comparison. Thus, at a 120-second minimum, observations at 00:00, 00:01, and
00:02 retain the first and third observations. Equality with the threshold is
valid. Segment and eligibility boundaries, invalid data, duplicate/nonpositive
timestamps, and unresolved human decisions break this scan. Exact repeats retain
precedence when both rules apply. Short-interval rejection overrides an existing
point keep/reject/uncertain row while preserving that row as provenance, and uses
`position_decision_source=automatic_short_interval`.

Automatic short-interval events and points, like exact repeats, are excluded from
manual/semiautomatic queues and point navigation. They remain faint raw context
and keep their fixed timing marker when another event includes them, but never
enter live surviving map or velocity adjacency. Surviving adjacency is
recalculated after both automatic removal classes and before local recovery.

Every surviving edge stores haversine distance and actual elapsed time. A trusted
plausible edge requires:

```text
minimum_local_dt_seconds <= dt <= max_local_gap_seconds + gap_tolerance_seconds
speed <= speed_threshold_m_s
```

Nominal, short, large-gap, nonpositive, and high-speed flags remain in the output.
A strictly positive short interval creates a resolved, hidden automatic removal;
an excessive gap remains diagnostic-only. High-speed geometry with unusable
timing is `boundary_or_insufficient_context`; duplicate/nonpositive time is never
handled by short-interval thinning. Conservative modes mark duplicate time for
review, while aggressive automatic mode chooses one deterministic representative.

A directional pass becomes trusted only after two consecutive plausible retained
edges. From a trusted anchor followed by a locally usable high-speed edge, it
expands the skipped block until the first bridge satisfying the same time and
speed limits. Search stops when anchor-to-reconnection elapsed time exceeds
`max_local_gap_seconds + gap_tolerance_seconds`; the permitted distance therefore
grows naturally with elapsed time. Forward and backward passes are independent.
`max_automatic_removal_points` also stops expansion. A candidate block is applied
only after the complete block cures its triggering edge, so fixed-point processing
cannot bypass the cap through a sequence of partial deletions.

`high_confidence_reject` requires the exact same skipped source observations and
compatible endpoints in both directions, a plausible bridge, retained continuation
on both sides, fixed-point confirmation of any recursive continuation blocks, no
overlap/conflict with another proposal, and no human keep/uncertain decision. All
nonoverlapping confirmed blocks are applied together, adjacency is recalculated,
and processing repeats to a fixed point.

An isolated high-speed edge tests removal of either endpoint. Each trial uses the
complete bridge elapsed time and succeeds when its bridge is usable, at or below
the physical speed threshold, and introduces no new anomalous adjacent edge. A
local speed baseline uses the median and MAD of up to 15 clean edges on each side,
with at least eight samples and a 0.05 m/s scale floor. If only one endpoint cures
the edge, it is selected. If both cure it, one is selected only when its robust
smoothness score is at least two units better; otherwise the edge remains
ambiguous. The spherical interpolation residual and direction remain supporting
evidence, not hard gates. A bridge more than three robust deviations from local
speed is explicitly warned about but remains physically valid.

Conservative bidirectional excursion recovery chooses the smallest curing block within the
elapsed-time limit. Both directions must identify identical immutable source
observations with plausible continuation. Single edges without a clear winner,
directional disagreement, insufficient local context, non-returning excursions,
hard boundaries, timing-blocked geometry, duplicate time, and multiple substantial
segments remain review-required. Connected flagged edges are grouped with up to
`merge_gap_edges` plausible intervening edges.

In aggressive automatic mode, every remaining usable high-speed edge tests
left- and right-expanding contiguous blocks. Equal-size cures are ranked by the
maximum vector acceleration at their bridge boundaries, then local speed
smoothness, with an immutable chronological tie-breaker. Vector acceleration uses
the change between incoming and outgoing east/north velocity divided by the
separation of the edge midpoints. It therefore recognizes both abrupt speed
changes and reversals, but never makes an over-threshold bridge acceptable. All
nonoverlapping winners are applied together and adjacency is rebuilt to a fixed
point. If no cure exists within either configured limit, the observations are
retained as a resolved `automatic_keep_removal_limit_reached` anomaly rather than
being deleted indefinitely.

The ordinary bounded solver cannot test the first or last point of a retained
temporal segment because no bridge anchor exists outside the segment. Aggressive
mode therefore applies one separate boundary-endpoint test. It removes a boundary
point only when the adjacent edge exceeds
`speed_threshold_m_s * boundary_endpoint_speed_multiplier`, the inward side has
at least `endpoint_speed_min_samples` consecutive plausible edges, no competing
interior removal within the ordinary point/time limits preserves the endpoint,
and the endpoint has no human keep/uncertain decision. Human uncertainty cannot
create an artificial eligible boundary. The decision and its support edges are
stored as `automatic_boundary_endpoint_speed_cure` evidence.

Automatic mode also resolves duplicate timestamps by retaining the representative
with the best surrounding speed/acceleration score and rejecting the others.
Conservative modes continue to review duplicate timestamps.

## Modes, decisions, and final resolution

- `--automatic` has no GUI and uses the aggressive deterministic policy. It leaves
  no unresolved eligible local-geometry event; capped non-cures remain explicitly
  flagged but valid. Upstream drogue, deployment, temporal, or human uncertainty is
  reported separately and is never disguised as solved geometry.
- `--semiautomatic` queues pending temporal segments, unresolved local events,
  timing-blocked anomalies, persistent/ambiguous geometry, and human/automatic
  conflicts. Exact-repeat and automatic short-interval events are always hidden;
  resolved fragment and speed-cure geometry events stay hidden unless
  overlapped.
- `--manual` exposes pending reviewable events first, with completed reviewable
  history accessible explicitly through P. Unique endpoint and bidirectional block
  repairs are queued for confirmation. The exact proposed block is highlighted,
  but R/K continue to decide one selected observation at a time. Recalculation
  retains, shrinks, replaces, or closes the event after those decisions.

The reviewer uses raw native geometry only. Startup reads compact event summaries
from Parquet footers; it loads the QC frame and MAT trajectory only for the visible
platform and keeps at most two platforms cached. Point and event navigation do not
rerun QC. Decisions are staged until N or Q, then the review CSV and only the
affected platform files are updated atomically.

## Reviewer layout and controls

Manual and semiautomatic review provide three synchronized scientific
panels:

1. the complete native trajectory, with the current temporal segment or local
   evidence highlighted;
2. a local event map containing every displayed observation, raw geometry,
   surviving adjacency, trigger edges, bridge, anchors, decisions, and selection;
3. a full-width surviving-velocity view. A single steel-blue line
   shows the current staged surviving adjacency and updates immediately after
   R/K. Each edge speed belongs to its later retained observation, so rejecting
   one point removes its incoming sample and places the recalculated bridge speed
   at the next retained point. Frozen trigger evidence remains as small crimson
   dots; it does not move or indicate a new rejection. Short intervals, large
   gaps, and duplicate/nonpositive intervals use separate bottom-axis symbols.
   Segment and unusable-timing boundaries break the blue line, the configured
   threshold remains visible, and the velocity range expands for new extrema.
   The horizontal range always comes from every raw timestamp in the displayed
   context, so an animated live line cannot be compressed by one isolated evidence
   marker. A genuinely empty series says `No surviving velocity in this window`.
   The local map retains its staged adjacency preview and rejected-point X.

A compact readout in the velocity panel gives the selected immutable source index,
UTC, current status/manual decision, and raw, surviving, and provisional
incoming/outgoing speeds. This replaces the former UTC/source-index panel while
retaining the evidence needed to choose a point.

The header names the platform and event and reports only the number of pending
events remaining; the old mutable `event X/Y` ordinal is not shown. History is
labelled `COMPLETED HISTORY — item X/Y`, and an all-complete session opens a
dedicated completion screen. The
explanation translates event status, reason, implicated observations, bridge
evidence, and directional agreement into prose; internal JSON is never displayed.
The `Action target` line always gives the exact selected source observation or the
immutable segment fingerprint.

Plot colors and symbols have fixed meanings:

- neutral gray: raw context;
- blue: surviving adjacency;
- translucent dashed red underlay: triggering high-speed map edges, drawn beneath
  the narrower blue surviving line so coincident evidence does not hide it;
- small solid crimson dot: frozen triggering velocity sample at the edge's later endpoint;
- green dashed: proposed bridge;
- crimson: exact proposed geometry block;
- navy square: trusted anchor;
- large gold target and boxed `SELECTED <index>` callout: selected observation;
- green ring, red X, and orange triangle: manual keep, reject, and uncertain;
- purple diamond: unresolved;
- low-opacity gray: outside drogue or operational scope.

Only selected, implicated, proposed, and anchor source indices are annotated. The
figures are presentation-only: they use the QC haversine/time utility for display
edge metrics but never reclassify an observation.

Keyboard and button controls are:

| Control | Effect |
| --- | --- |
| Left / Right or point buttons | previous / next eligible observation on the current platform, stopping at each end |
| click overview or local map | select the nearest plotted immutable source observation; an overview click outside the local window recenters presentation only |
| P | previous pending event; from the first pending event, enter completed history without saving staged edits |
| N | save/recalculate and advance to the next pending event; from history, return to pending review or the completion screen |
| K / R | keep/restore a rejected point or reject the selected eligible point; on temporal events, retain / exclude the segment |
| Q | save all staged edits and affected trajectory files, then close |

The reviewer disconnects Matplotlib's overlapping default shortcuts so `K`, `R`,
`P`, and `Q` cannot also change axis scale, reset the view, enable pan mode, or
bypass the reviewer's clean-close path.

Left/Right navigation is chronological across all eligible observations on the
current platform and skips exact repeats, automatic short-interval removals,
invalid, outside-scope, pending-segment, and otherwise ineligible observations.
It does not wrap, and it is disabled for temporal events. Selecting a point
outside the current local window recenters the local
evidence panels. In manual mode, the event queue remains limited to platforms
with detected review events. N never enters completed history while pending work
exists. P walks backward through pending work and then through completed history;
N exits history back to active review. A point decision cannot
override a drogue, deployment, or segment eligibility boundary.

Point stepping updates persistent overview, local-map, selected-time, velocity,
and readout artists without rebuilding the event, writing a session, or rerunning
QC. Supported Matplotlib backends use blitting, with a normal redraw fallback.
R/K update provisional reject/keep markers, map adjacency, and surviving speeds immediately.
Exact repeated-position groups and automatic short-interval removals require no
reviewer action and are not selectable.
P retains provisional edits in memory; N commits the event being left; Q
commits every remaining event. There is no note editor or separate group-action
control in the simplified reviewer.

## Resumable reviewer session

Manual and semiautomatic runs automatically use the file derived from the review
path, normally `position_review.session.json`. No resume flag or iteration archive
is needed. The review CSV remains authoritative and the session contains no
decisions. Schema version 2 is unchanged. Point movement and P do not write the
session; N and Q save the current cursor atomically:

```json
{
  "schema_version": 2,
  "saved_at_utc": "2026-09-25T12:00:00+00:00",
  "mode": "manual",
  "platform_code": "300534061905670",
  "event_id": "single_edge_ambiguous:a89fb07ab7f76799",
  "event_queue_index": 6,
  "selected_source_obs_index": 16253,
  "selected_time_utc": "2025-03-01T04:25:00+00:00",
  "config_sha256": "...",
  "source_sha256": "...",
  "review_sha256": "...",
  "draft_note": ""
}
```

At startup the workflow independently rebuilds only trajectory files whose review,
configuration, drogue, deployment, or source provenance is stale. It restores the
same event and immutable source observation only while that event remains pending.
If the saved event is complete and pending work remains, it opens the next
chronological pending event; completed history remains available through P. If no
pending work remains, the completion screen opens. Changed grouping first
reconciles surviving immutable event identities and then uses platform, earliest
UTC time, and earliest source index as a deterministic chronological anchor.
Config/review changes reconcile against the rebuilt queue. A source-hash mismatch
rejects the saved cursor. The status line reports whether the session was resumed,
reconciled, redirected
from completed history, or ignored safely.

Q and a normal window close commit all staged decisions, recompute each affected
platform once, atomically replace its trajectory file, save the cursor, and close.
A process crash can lose staged edits, but each already committed CSV decision
survives. A failed trajectory publication leaves its older valid file; the
platform-specific review hash forces only that file to rebuild next time.

Final precedence is: source invalid; outside drogue; outside deployment/segment;
deployment/temporal uncertainty; exact-repeat reject; automatic short-interval
reject; human reject; human keep; human uncertain; geometry reject;
unresolved-event implication; ordinary valid. A local
keep cannot cross a drogue, deployment, or segment boundary. `final_position_valid`
is true only for `valid`. The `resolved_position_trajectory` helper validates the
full in-memory QC result against the raw trajectory and rejects unresolved/uncertain
state unless the caller explicitly opts in. Reconstruction reads the published
Parquets directly and validates their footer provenance and completeness.

## Product schemas and persistence

The configured position directory contains `<platform_code>.parquet`, one file per
native trajectory. Schema `4.0` contains one row per source observation within the
inclusive time span from the first to the last `final_position_valid` observation.
It includes the original coordinates needed for before/after comparison and
reconstruction. Its columns are
grouped as follows:

- identity/validity: `platform_code`, `source_obs_index`, `source_path`,
  `source_sha256`, `time`, `source_lon`, `source_lat`, `valid_timestamp`, `source_position_valid`,
  `source_invalid_reason`;
- drogue/deployment: `final_drogue_status`, `final_drogue_loss_time`,
  `analysis_cutoff_time`, `drogue_decision_source`, `drogue_eligible`,
  `deployment_status`, `deployment_eligible`, `deployment_uncertain`,
  `deployment_provenance`;
- human snapshot: `human_position_decision`, `human_decision_source`,
  `human_review_event_id`, `human_review_config_sha256`,
  `human_segment_decision`, `human_segment_review_config_sha256`;
- segment/event: `segment_id`, `segment_fingerprint`, `segment_start_time`,
  `segment_end_time`, `segment_n_observations`, `segment_duration_hours`,
  `segment_median_dt_seconds`, both segment fractions, both adjacent gaps,
  `segment_status`, `temporal_event_id`, `temporal_event_type`,
  `temporal_auto_status`, `temporal_auto_decision`, `temporal_auto_reason`;
- repeat/edge: `exact_repeat_flag`, `repeated_coordinate_count`,
  `repeated_coordinate_group_id`, `repeat_duration_seconds`, surviving predecessor,
  dt/distance/speed, and the five timing/speed flags;
- local evidence: `local_event_id`, `local_event_type`, implicated source indices,
  point automatic status/decision/reason, iteration, review-required status/reason,
  bridge endpoints/speed, and forward/backward JSON;
- resolution: `final_position_status`, `final_position_valid`,
  `position_decision_source`, `platform_qc_complete`, `reconstruction_available`.

`source_lon` and `source_lat` are copied without normalization or wrapping and are
never modified by review. Invalid values may remain on rejected audit rows; every
final-valid row must have a finite in-range coordinate. Export trimming uses UTC
time, regardless of source order, and preserves original `source_obs_index` values.
All observations at the endpoint timestamps are included, including rejected ones.
Rows with no usable timestamp cannot be placed within the span and are omitted.
If there are no valid observations, a schema-correct empty Parquet is written.

For the native before-QC track, use the published `time`, `source_lon`, and
`source_lat` rows. For the after-QC track, filter those same rows by
`final_position_valid`. Sort chronologically for plotting and preserve segment
and timing breaks. Rejected observations inside the span remain available.

QC still runs on the full raw record. Review history and pending events outside
the exported span stay in the footer event catalog; the reviewer lazily recalculates
the full active platform from the raw source when loading a trimmed product.
This read does not publish files. Committed changes can expand or shrink the
exported span on the next publication.

Compact event details exist
only on implicated rows. Every footer stores schema,
algorithm and product-layout versions, effective configuration, platform-specific
review/drogue/deployment/source hashes, a compact event catalog, and observation/
unresolved counts. It also stores the resolution policy and per-case decision
counts. Event summaries include event type, earliest UTC time, and
earliest source index for deterministic navigation. The reviewer constructs its queue from these footers without
materializing all trajectory rows.

Footer `observation_count` is the number of exported rows, while
`source_observation_count` is the full input count. `trimmed_observation_count`
records their difference. `valid_time_start_utc` and `valid_time_end_utc` record the
inclusive span (both null for an empty product). Completeness, unresolved counts,
and decision summaries describe the full QC result, even for an empty export, so
trimming cannot hide an unresolved platform. `reconstruction_available` records
whether a complete usable trajectory exists; reconstruction still requires at
least two accepted fixes.

`position_review.csv` contains only explicit decisions:

```text
platform_code,source_sha256,target_type,target_id,source_obs_index,event_id,
event_type,decision,decision_source,auto_status_snapshot,auto_decision_snapshot,
auto_reason_snapshot,qc_config_sha256,note,review_timestamp
```

Point decisions target one observation identity. Segment identity is a hash of its
immutable source observations. Existing accepted-automatic and note records remain
readable, although the simplified reviewer creates only manual K/R decisions and
preserves prior notes when changing them.

The CSV is atomically replaced by N or Q. Each affected trajectory Parquet is then
recomputed and atomically replaced independently. Unaffected files remain byte-for-
byte unchanged.

Algorithm version `native-position-qc-v2.6` adds the guarded gross-speed boundary
endpoint cure to the aggressive bounded automatic solver. Switching between
automatic and reviewer modes causes the per-platform products to rebuild under
the appropriate policy; later runs with the same policy reuse them normally.

The product schema is `4.0` and layout is `per-trajectory-v3`. The compatibility
check requires both coordinate columns, so older products rebuild independently.
Human decisions are retained because the authoritative review CSV is separate and
is reapplied during that rebuild.

After every successful command, `position_qc_run_summary.csv` is atomically written
in the trajectory output directory. It has one row per platform plus an
`ALL_PLATFORMS` row and counts repeat, short-interval, duplicate-time, one-point and
multi-point speed-cure removals, block sizes, human rejections, capped retained
anomalies, local unresolved observations, upstream uncertainty, and final valid/
rejected observations. The CLI prints the aggregate counts. The summary is derived
from Parquet footers and does not load or concatenate all trajectory rows. Decision
counts and summary `observation_count` cover the full source record, including QC
rejections removed by export trimming. `exported_observation_count` and
`trimmed_observation_count` separately report publication sizes.

## Raw-data smoke check

The following September 25, 2026 smoke run used raw MicroSVP positions and the
generic drogue resolver. Event counts are QC diagnostics, not comparisons with the
supplied position QC. Segment counts cover the eligible pre-cutoff interval;
`n_flagged_edges` counts surviving high-speed edges.

| platform | source obs | segments | dominant obs | auto fragment obs | flagged edges | events | high-confidence events | review-required events | auto-rejected points |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 300534061905670 | 31,246 | 1 | 16,960 | 0 | 7 | 12 | 7 | 5 | 9 |
| 300534061906090 | 19,072 | 1 | 14,721 | 0 | 44 | 35 | 5 | 27 | 21 |
| 300534063015610 | 23,273 | 2 | 17,694 | 1 | 8 | 12 | 1 | 8 | 10 |

Run all synthetic tests with `python -m pytest`. The v2-focused files are
`tests/test_native_position_qc_v2.py`, `tests/test_position_workflow.py`, and
`tests/test_position_reviewer_gui.py`.
