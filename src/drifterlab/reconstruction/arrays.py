"""Assign deployment arrays from exact first retained-QC positions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ArrayAssignment:
    """One platform's deterministic deployment-array assignment."""

    platform_id: str
    array_id: int
    gap_from_previous_start_hours: float | None
    gap_to_next_start_hours: float | None


def assign_start_arrays(
    platform_ids: Sequence[str],
    start_times: Sequence[np.datetime64],
    *,
    maximum_adjacent_start_gap_hours: float,
) -> tuple[ArrayAssignment, ...]:
    """Assign 1-based arrays after sorting starts by time and platform ID.

    A new array begins only when the gap from the previous exact first
    retained-QC timestamp is strictly greater than the configured threshold.
    Results are returned in the same platform order supplied by the caller.
    """
    threshold = float(maximum_adjacent_start_gap_hours)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("maximum_adjacent_start_gap_hours must be finite and positive")
    platforms = tuple(str(value) for value in platform_ids)
    times = np.asarray(start_times, dtype="datetime64[ns]")
    if times.ndim != 1 or len(platforms) != len(times) or not platforms:
        raise ValueError("Array assignment needs equally sized, nonempty platform starts")
    if any(not platform.strip() for platform in platforms):
        raise ValueError("Array assignment contains an empty platform identifier")
    if len(platforms) != len(set(platforms)):
        raise ValueError("Array assignment contains duplicate platform identifiers")
    if np.isnat(times).any():
        raise ValueError("Array assignment contains an invalid start timestamp")

    order = sorted(range(len(platforms)), key=lambda index: (times[index], platforms[index]))
    sorted_times = times[order]
    adjacent = np.diff(sorted_times.astype(np.int64)) / 3_600_000_000_000
    array_ids = np.ones(len(platforms), dtype=np.int32)
    if len(platforms) > 1:
        array_ids[1:] = 1 + np.cumsum(adjacent > threshold, dtype=np.int32)

    by_platform: dict[str, ArrayAssignment] = {}
    for position, original_index in enumerate(order):
        previous = None if position == 0 else float(adjacent[position - 1])
        following = None if position == len(platforms) - 1 else float(adjacent[position])
        platform = platforms[original_index]
        by_platform[platform] = ArrayAssignment(
            platform, int(array_ids[position]), previous, following,
        )
    return tuple(by_platform[platform] for platform in platforms)
