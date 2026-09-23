# Manual ARCTERX MicroSVP position review

Run these commands from the repository root with the environment activated.
The existing `diagnostics` extra supplies Matplotlib. If needed, install it once:

```powershell
python -m pip install -e ".[diagnostics]"
```

Launch or resume the reviewer:

```powershell
python -m drifterlab.cli.review_arcterx_positions review
```

Generate a fresh candidate queue **with recursive recovery suggestions**, without
exporting Parquet or changing any decisions:

```powershell
python -m drifterlab.cli.review_arcterx_positions suggest
```

Run this once to upgrade an older saved queue, then launch `review`. New queues
built by `review` or `apply` also include suggestions automatically.

Close the reviewer, then apply decisions and generate the next candidate queue:

```powershell
python -m drifterlab.cli.review_arcterx_positions apply
```

Launch `review` again to inspect that queue. The installed entry point
`drifterlab-review-arcterx-positions` accepts the same arguments after reinstalling
the package. The module command also works in the existing editable environment.

## Controls and display

- Click a native point, or use Left/Right to select adjacent displayed points.
- K: keep; R: reject; U: uncertain. Each decision immediately saves to CSV.
- A / **Accept high**: explicitly accept eligible high-confidence suggestion blocks
  in the current region. Existing keep, uncertain and reject decisions are preserved.
- N/P: next/previous candidate region. Navigation does not record a decision.
- S: save decisions and the current position in the queue.
- Q: save and quit. Closing the window also saves.
- Edit Reason and Note before recording a decision. Press Enter to finish text
  editing and return to the keyboard shortcuts. Buttons also record decisions.

The map labels chronological native row indices; the selected point's original
`source_obs_index`, exact UTC timestamp and decision appear below the plots.
Green circles, red crosses and orange diamonds show keep, reject and uncertain.
Rejected points remain visible and selectable, so K or U can revise a rejection.
Sparse labels keep longer regions legible; clicking and arrow navigation still
reach every displayed point. The map uses local east/north kilometres for display;
speed calculations use spherical haversine distance and actual elapsed time.

Red edges and red speed markers show the **frozen queue**. Blue lines and speeds
preview adjacency among currently surviving valid points, including new bridges.
The speed of an edge is assigned to its later endpoint. A rejected selected point
has no incoming surviving speed; its displayed outgoing value belongs to the
next surviving point's edge. Supplied 30m and 60m tracks are dashed references
only and never drive a decision or mask. Invalid reconstructed positions break
their lines. No reconstructions are regenerated.

The queue stays fixed while reviewing, so rejecting a point does not unexpectedly
skip or reorder regions. Stop and resume freely; untouched points are not added
to the decision table. Explicit keeps remain in subsequent queues if their edges
are still above threshold. A candidate flag is not proof that either endpoint
is wrong. An unresolved or uncertain point remains valid by default.

## Recursive jump recovery

Recovery produces proposals in a separate file. It never modifies coordinates,
validity, the manual decision table or the reviewed product by itself. Neither
`suggest` nor `apply` accepts a suggestion. All regions remain in the queue,
including regions with high-confidence proposals, until manual decisions remove
their jumps during an apply/regeneration step.

Both directional passes start with **two consecutive plausible edges** within
the region's saved context (three valid points). The first valid point alone is
not trusted. In the forward pass, the third point becomes the anchor; in the
backward pass the equivalent rule is applied in reverse chronological order.
Untrusted early points remain manual. Context defaults to 12 valid fixes on each
side; a very small `--context-points` can leave insufficient evidence for proposals.

From a trusted anchor, a plausible next point is retained. At a >3 m/s edge,
the pass tests skipping 1, then 2, then 3 consecutive points in the current
adjacency sequence, choosing the first plausible bridge. It retains that
reconnection as the new anchor and continues the same process. If recovery fails,
trust must be established again from a new two-edge coherent run. Existing manual
keeps/uncertainties cannot be skipped; source-invalid points break native-mode
recovery. Surviving mode excludes source-invalid and manually rejected points
before counting skips, consistent with the iterative adjacency calculation.

A plausible edge has finite speed **<=3 m/s**, positive elapsed time, and elapsed
time no longer than the existing 30-minute-plus-one-second local context limit.
Duplicate/nonpositive timestamps inside a bridge cannot be hidden by a skip.
This context guard changes neither the speed threshold nor the candidate queue:
long-gap jumps still require manual review. No bearing, acceleration or supplied
reconstruction is used to establish confidence.

Continuation is the **next retained edge after reconnection**, not necessarily
the immediately next source fix. Thus a coherent prefix followed by
`GOOD -> BAD -> GOOD -> BAD -> GOOD` and a coherent suffix produces two separate
one-point blocks. The first continuation can bypass the second bad fix. There is
no requirement that the immediate edge out of the first reconnection be clean.

- **High**: both directions skip the same exact block, agree on the bridge
  endpoints, and have plausible retained continuations on both sides. Any other
  skipped block used by those continuations must also pass this agreement test.
- **Medium**: a forward bridge is plausible, but backward evidence or continuation
  is unavailable/incomplete. It is a hypothesis for individual manual review.
- **Unresolved**: directions disagree, a recursive continuation lacks confirmation,
  no trusted forward recovery exists, or no clean bridge is found within three
  skipped points. No bulk rejection is offered.

Purple hollow squares mark frozen high-confidence suggestions; orange hollow
triangles mark medium hypotheses. The recovery panel reports counts and, for the
selected point, confidence, skip count, bridge/continuation speeds, directional
agreement and both directional explanations. NaN means evidence is unavailable.
The separate directional skipped-row lists are in the proposal CSV. Multiple
spikes can share a review region while retaining separate proposal block IDs.

**Accept high [A]** rechecks the frozen high-confidence blocks against current
decisions before saving explicit rejects. It accepts only whole blocks that are
still high-confidence, have the same bridge/block identity, and contain no
existing manual decisions. Changes that invalidate a proposal, including an
individual keep/uncertain override, prevent its bulk acceptance. After accepting,
K or U can override any individual rejection. The manual reason records
`user_accepted_high_confidence_recovery` and the note records the proposal block
and speeds. A frozen high marker can therefore remain visible over an explicitly
kept point until regeneration; the manual decision always wins.

## Candidate generation and iteration

The first `review` on a new review directory with no decisions uses immediate
native adjacency, matching the original residual-jump diagnostic. Runs share its
local skip-one classifier and strict **>3 m/s** speed criterion. Nearby runs merge
into one region when separated by at most two edges with usable local context.
The local context limit is the diagnostic's 30 minutes plus one second; this
limits classification/merging, never suppresses a flagged edge. Long-gap and
boundary events remain in the queue with insufficient-context labels.

Priority is persistent runs and grouped excursions, then isolated-spike
candidates, then ambiguous events in descending maximum-speed order. This
orders very high-speed ambiguous edges first without introducing another speed
threshold. Every region includes 12 valid points before and after where available,
plus intervening source rows. Use `--context-points 24` or `--merge-gap-points 0`
when **building** a queue to change these display/grouping choices. Options do
not change an existing frozen queue.

Every `apply` joins the latest decisions to immutable source rows, exports all
native observations, and recalculates consecutive edges among surviving valid
positions with valid timestamps. This bridges both manual rejects and originally
invalid fixes. Consequently even an empty review table can expose edges absent
from the original no-bridging diagnostic. Nonpositive time intervals never receive
a finite speed. New flags create review candidates, never automatic rejections.
The `suggest` command creates a new iteration without exporting the product; on
an existing session it likewise recalculates surviving adjacency before recovery.

The exported validity is exactly:

```text
reviewed_position_valid = position_valid AND NOT manual_reject
```

This position mask does not additionally impose time or drogue validity. Edge
calculations exclude invalid times; other analyses must choose their own masks.
The source `position_valid_qc`, its `position_valid` alias, original coordinates,
original audit flags/speeds and all other native-axis variables are retained.
Additional columns include `manual_review_decision`, reason, note, iteration,
review timestamp, `manual_reject`, `manual_uncertain`, `reviewed_position_valid`,
`reviewed_audit_speed_m_s`, and `reviewed_residual_jump_flag`.

## Files and reproducibility

Defaults, relative to the working directory:

| File | Purpose |
| --- | --- |
| `data/ARCTERX_MicroSVP_QC.zarr` | Read-only supplied master |
| `data/review/microsvp_manual_position_review.csv` | Latest explicit point decisions; no duplicate source-row identities |
| `data/processed/ARCTERX_MicroSVP_reviewed_QC.parquet` | Latest applied reviewed product; atomically replaced by each successful apply |
| `data/review/microsvp_position_review/iteration_NNNN/candidates.csv` | Prioritized region queue for that iteration |
| `data/review/microsvp_position_review/iteration_NNNN/flagged_edges.csv` | Exact native/source endpoint indices, UTC times, speeds and bridged-row counts |
| `data/review/microsvp_position_review/iteration_NNNN/recovery_suggestions.csv` | Separate per-point proposals, unresolved endpoints, confidence and directional evidence |
| `data/review/microsvp_position_review/iteration_NNNN/applied_decisions.csv` | Decision snapshot used to generate the queue/product |
| `data/review/microsvp_position_review/iteration_NNNN/manifest.json` | Input identity, configuration, code/version hashes, counts, queue/product hashes |
| `data/review/microsvp_position_review/latest.json` | Pointer and manifest for the latest completed iteration |
| `data/review/microsvp_position_review/session.json` | Saved candidate and selected point for resuming |

The first queue is iteration 0001. Applying its decisions writes iteration 0002,
and so on; old iteration folders are preserved. The latest Parquet is generated
only by `apply`, not by saving in the GUI. Failed exports leave the previous
Parquet intact; interrupted runs may leave an unused iteration directory, which
future runs skip. A crash between publishing the Parquet and the latest manifest
can be identified by their hashes; rerun `apply` to publish a consistent pair.

`recovery_suggestions.csv` includes `auto_suggested_reject` (true only for high
confidence), `auto_confidence`, `auto_skip_count`, `auto_bridge_speed_m_s`,
`auto_continuation_speed_m_s`, `auto_forward_backward_agree`, and `auto_reason`.
It also records platform, exact UTC time, source observation index/hash, iteration,
candidate ID, proposal block ID, native bridge/continuation row indices, both
directional reasons/skipped-row lists, and backward bridge/continuation speeds.
Only implicated rows are stored; good context points need no proposal row.
Medium hypotheses and unresolved rows have `auto_suggested_reject=false`.

The candidate table adds `auto_region_status`, `auto_high_confidence_points`,
`auto_medium_confidence_points` and `auto_unresolved_points`. A region containing
several confidence levels has status `mixed`. Manifests record the recovery code
hash, proposal-file hash, explicit recovery configuration and high-confidence
point count. These proposal fields are deliberately separate from both the
manual decision CSV and reviewed Parquet. Earlier proposal files remain frozen;
the next iteration reflects accepted rejects and new neighboring jumps.

The review CSV keys each point by string platform ID and original source index,
and verifies exact nanosecond UTC time plus source-file SHA-256 before applying.
Revising a decision replaces that point's latest entry, including timestamp and
review iteration. Applied iteration snapshots preserve earlier applied decisions;
there is no separate history of every unsaved/intermediate key press. CSV saves
use atomic replacement and detect stale edits from another session. Use one
reviewer/application process at a time. Externally edited duplicate or mismatched
identities are rejected rather than silently ignored. If editing the CSV to revisit
a resolved point after its region leaves the queue, preserve its identity fields,
update decision/reason/timestamp as appropriate, then run `apply`.

All paths can be overridden with `--master`, `--review-file`, `--queue-directory`
and `--output`. Use the same overrides for both commands. Review output paths
must be separate and outside the master. No master arrays, source files, global
QC settings, supplied reconstructed tracks, deployment groups or pairs are changed.

Focused synthetic validation (no manual session or campaign-wide run):

```powershell
python -m pytest tests/test_jump_recovery.py tests/test_position_review.py -q
```
