"""Bidirectional, bounded jump recovery proposals; never changes a validity mask."""

from functools import lru_cache
import json

import numpy as np
import pandas as pd

from drifterlab.qc import flags
from drifterlab.qc.review import MAX_LOCAL_GAP_SECONDS, GAP_TOLERANCE_SECONDS

RECOVERY_CONFIG = {"version": 1, "threshold_m_s": 3.0, "max_skip_points": 3,
                   "trusted_run_edges": 2,
                   "max_context_gap_seconds": MAX_LOCAL_GAP_SECONDS + GAP_TOLERANCE_SECONDS,
                   "continuation": "next retained edge; recursive skips must also agree bidirectionally",
                   "manual_decisions": "never skip explicit keep or uncertain decisions"}
SUGGESTION_COLUMNS = [
    "candidate_id", "platform_code", "review_iteration", "row_index", "source_obs_index", "time", "source_sha256",
    "auto_suggested_reject", "auto_confidence", "auto_skip_count", "auto_bridge_speed_m_s",
    "auto_continuation_speed_m_s", "auto_forward_backward_agree", "auto_reason", "auto_block_id",
    "auto_anchor_row", "auto_reconnection_row", "auto_continuation_row",
    "auto_forward_reason", "auto_backward_reason", "auto_forward_skipped_rows", "auto_backward_skipped_rows",
    "auto_backward_bridge_speed_m_s", "auto_backward_continuation_speed_m_s",
]
SUMMARY_DEFAULTS = {"auto_region_status": "unresolved", "auto_high_confidence_points": 0,
                    "auto_medium_confidence_points": 0, "auto_unresolved_points": 0}


def _walk(order, plausible, speed, eligible, protected):
    """Greedy shortest-skip traversal, reseeding only from two coherent edges."""
    blocks, links, retained = {}, {}, set()
    trace = {j: "no_trusted_coherent_run_in_this_direction" for j in order}
    i, trusted = 0, False
    while i < len(order):
        if not trusted:
            if i + 2 >= len(order):
                break
            a, b, c = order[i:i + 3]
            if plausible(a, b) and plausible(b, c):
                for j in (a, b, c):
                    retained.add(j)
                    trace[j] = "trusted_run_of_two_plausible_edges"
                links[a], links[b] = (b, None), (c, None)
                i += 2
                trusted = True
            else:
                i += 1
            continue
        if i + 1 >= len(order):
            break
        a, b = order[i:i + 2]
        if plausible(a, b):
            links[a] = (b, None)
            retained.add(b)
            trace[b] = "accepted_plausible_edge_from_trusted_anchor"
            i += 1
            continue
        # An invalid/nonpositive/distant edge cannot establish a jump-recovery hypothesis.
        recovered = False
        blocked_by_manual = False
        if np.isfinite(speed(a, b)) and speed(a, b) > RECOVERY_CONFIG["threshold_m_s"]:
            for count in range(1, RECOVERY_CONFIG["max_skip_points"] + 1):
                if i + count + 1 >= len(order):
                    break
                skipped = tuple(sorted(order[i + 1:i + count + 1]))
                k = order[i + count + 1]
                if any(protected[j] for j in skipped):
                    blocked_by_manual = True
                    break
                if not all(eligible[j] for j in skipped):
                    break
                # Duplicate/nonpositive times and excessive gaps cannot be hidden by a skip.
                if plausible(a, k):
                    key = (min(a, k), skipped, max(a, k))
                    blocks[key] = {"anchor": a, "reconnect": k, "skipped": skipped,
                                   "bridge": speed(a, k), "continuation": np.nan,
                                   "continuation_row": None, "dependency": None}
                    links[a] = (k, key)
                    for j in skipped:
                        trace[j] = f"skip_{count}_bridge_{a}_to_{k}_speed_{speed(a, k):.6g}_m_s"
                    trace[k] = "reconnected_from_trusted_anchor"
                    retained.add(k)
                    i += count + 1
                    recovered = True
                    break
        if not recovered:
            trace[a] += ("; manual_decision_blocks_recovery" if blocked_by_manual
                         else "; no_clean_bridge_within_three_points_or_unusable_edge")
            trusted = False
            i += 1
    for block in blocks.values():
        if block["reconnect"] in links:
            end, dependency = links[block["reconnect"]]
            block.update(continuation=speed(block["reconnect"], end), continuation_row=end, dependency=dependency)
    return blocks, retained, trace


def recover_region(frame: pd.DataFrame, candidate: pd.Series, edges: pd.DataFrame) -> pd.DataFrame:
    """Use frozen region context, with current manual decisions authoritative.

    High confidence requires the same exact block AND bridge endpoints in both
    directions, a retained continuation on both sides, and confirmation of any
    recursive blocks used by those continuations. Separate spikes stay separate.
    """
    lo, hi = int(candidate.context_start), int(candidate.context_end)
    eligible = (frame.reviewed_position_valid & frame.time.notna()).to_numpy()
    protected = frame.manual_review_decision.isin(["keep", "uncertain"]).to_numpy()
    indices = np.arange(lo, hi + 1)
    if candidate.adjacency == "surviving":
        indices = indices[eligible[indices]]
    time = frame.time_qc.to_numpy(dtype="datetime64[ns]")
    lon, lat = frame.lon_qc.to_numpy(), frame.lat_qc.to_numpy()

    @lru_cache(None)
    def speed(a, b):
        a, b = sorted((a, b))
        if not eligible[a] or not eligible[b]:
            return np.nan
        return float(flags.consecutive_speed(time[[a, b]], lon[[a, b]], lat[[a, b]])[1])

    @lru_cache(None)
    def plausible(a, b):
        a, b = sorted((a, b))
        elapsed = flags.interval_seconds(time[[a, b]])[1]
        return bool(np.isfinite(speed(a, b)) and speed(a, b) <= RECOVERY_CONFIG["threshold_m_s"]
                    and 0 < elapsed <= RECOVERY_CONFIG["max_context_gap_seconds"]
                    and np.all(np.diff(time[a:b + 1][~np.isnat(time[a:b + 1])]) > np.timedelta64(0, "ns")))

    order = list(map(int, indices))
    forward, _, ftrace = _walk(order, plausible, speed, eligible, protected)
    backward, bretained, btrace = _walk(order[::-1], plausible, speed, eligible, protected)
    matching = set(forward) & set(backward)
    high = {key for key in matching if all(np.isfinite(block["continuation"])
                                           for block in (forward[key], backward[key]))}
    # A continuation that itself skips points must not rely on an unconfirmed block.
    while True:
        confirmed = {key for key in high if all(block["dependency"] is None or block["dependency"] in high
                                                for block in (forward[key], backward[key]))}
        if confirmed == high:
            break
        high = confirmed
    bskipped = {j for block in backward.values() for j in block["skipped"]}
    confidence, reasons = {}, {}
    for key, block in forward.items():
        if key in high:
            confidence[key], reasons[key] = "high", "same_block_and_bridge_in_both_directions_with_confirmed_continuations"
        elif key in matching:
            incomplete = any(not np.isfinite(p["continuation"]) for p in (block, backward[key]))
            confidence[key] = "medium" if incomplete else "unresolved"
            reasons[key] = "continuation_unavailable" if incomplete else "recursive_continuation_not_bidirectionally_confirmed"
        elif set(block["skipped"]) & (bretained | bskipped):
            confidence[key], reasons[key] = "unresolved", "forward_backward_disagree_on_skipped_block_or_bridge"
        else:
            confidence[key], reasons[key] = "medium", "forward_bridge_plausible_but_backward_confirmation_unavailable"

    # Retain explicit unresolved edge endpoints even where neither traversal finds a bridge.
    region_edges = edges[edges.candidate_id == candidate.candidate_id]
    fpoints = {j: key for key, block in forward.items() for j in block["skipped"]}
    bpoints = {j: key for key, block in backward.items() for j in block["skipped"]}
    start, end = int(candidate.row_index_start), int(candidate.row_index_end)
    relevant_keys = {key for key in set(forward) | set(backward) if any(start <= j <= end for j in key[1])}
    points = {j for key in relevant_keys for j in key[1]}
    unresolved_endpoints = set()
    for edge in region_edges.itertuples():
        a, b = int(edge.row_index_1), int(edge.row_index_2)
        if a not in points and b not in points:
            unresolved_endpoints.update((a, b))
    rows = []
    for j in sorted(points | unresolved_endpoints):
        fkey, bkey = fpoints.get(j), bpoints.get(j)
        f, b = forward.get(fkey), backward.get(bkey)
        chosen = f or b
        level = confidence[fkey] if f is not None else "unresolved"
        reason = reasons[fkey] if f is not None else (
            "no_trusted_forward_recovery; backward_only_hypothesis" if b is not None
            else "no_confirmed_recovery; inspect_directional_reasoning")
        if protected[j]:
            level, reason = "unresolved", "existing_manual_decision_is_authoritative"
        stamp = pd.Timestamp(frame.time.iloc[j])
        row = {"candidate_id": candidate.candidate_id, "platform_code": str(candidate.platform_code),
               "review_iteration": int(candidate.review_iteration), "row_index": j,
               "source_obs_index": int(frame.source_obs_index.iloc[j]), "time": stamp.isoformat(),
               "source_sha256": str(frame.source_sha256.iloc[j]), "auto_suggested_reject": level == "high",
               "auto_confidence": level, "auto_skip_count": len(chosen["skipped"]) if chosen else 0,
               "auto_bridge_speed_m_s": chosen["bridge"] if chosen else np.nan,
               "auto_continuation_speed_m_s": chosen["continuation"] if chosen else np.nan,
               "auto_forward_backward_agree": fkey is not None and fkey == bkey,
               "auto_reason": reason, "auto_block_id": json.dumps(fkey or bkey) if chosen else "",
               "auto_anchor_row": chosen["anchor"] if chosen else np.nan,
               "auto_reconnection_row": chosen["reconnect"] if chosen else np.nan,
               "auto_continuation_row": chosen["continuation_row"] if chosen else np.nan,
               "auto_forward_reason": ftrace.get(j, "invalid_or_already_rejected"),
               "auto_backward_reason": btrace.get(j, "invalid_or_already_rejected"),
               "auto_forward_skipped_rows": json.dumps(f["skipped"] if f else []),
               "auto_backward_skipped_rows": json.dumps(b["skipped"] if b else []),
               "auto_backward_bridge_speed_m_s": b["bridge"] if b else np.nan,
               "auto_backward_continuation_speed_m_s": b["continuation"] if b else np.nan}
        rows.append(row)
    return pd.DataFrame(rows, columns=SUGGESTION_COLUMNS)


def suggest_regions(frame: pd.DataFrame, queue: pd.DataFrame, edges: pd.DataFrame):
    result = queue.copy()
    for column, default in SUMMARY_DEFAULTS.items():
        result[column] = default
    tables = []
    for index, candidate in queue.iterrows():
        proposals = recover_region(frame, candidate, edges)
        counts = proposals.auto_confidence.value_counts()
        for level in ("high", "medium", "unresolved"):
            column = f"auto_{level}_confidence_points" if level != "unresolved" else "auto_unresolved_points"
            result.at[index, column] = int(counts.get(level, 0))
        levels = set(proposals.auto_confidence)
        result.at[index, "auto_region_status"] = next(iter(levels)) if len(levels) == 1 else "mixed" if levels else "unresolved"
        if len(proposals):
            tables.append(proposals)
    return result, pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=SUGGESTION_COLUMNS)
