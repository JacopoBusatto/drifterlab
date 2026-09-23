"""Local run labels shared by diagnostics and manual position review."""

import numpy as np

CLASSES = ["isolated_spike_candidate", "single_flagged_edge", "persistent_high_speed_run",
           "boundary_or_insufficient_context"]
MAX_LOCAL_GAP_SECONDS = 1800.
GAP_TOLERANCE_SECONDS = 1.


def run_class(n_edges: int, triplets: list[dict], before: dict, after: dict,
              edge_dt: np.ndarray) -> tuple[str, str]:
    if (edge_dt > MAX_LOCAL_GAP_SECONDS + GAP_TOLERANCE_SECONDS).any():
        return CLASSES[3], "flagged_edge_exceeds_local_gap_limit"
    if n_edges == 1:
        if before["usable_context"] and after["usable_context"]:
            return CLASSES[1], "one_existing_flagged_edge_with_nearby_valid_context"
        return CLASSES[3], "missing_nonpositive_or_distant_outer_context"
    if not all(t["triplet_context_usable"] for t in triplets):
        return CLASSES[3], "skip_one_context_missing_or_exceeds_gap_limit"
    if n_edges == 2 and triplets[0]["passes_local_skip_one_test"]:
        return CLASSES[0], "two_flagged_legs_replaced_by_one_reasonable_bridge"
    return CLASSES[2], ("more_than_two_flagged_edges_cannot_be_resolved_by_one_point"
                        if n_edges > 2 else "skip_one_bridge_remains_above_threshold")
