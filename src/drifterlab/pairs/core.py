"""Numerical operations for candidate encounter pairs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np
from pyproj import Geod


WGS84 = Geod(ellps="WGS84")


@dataclass(frozen=True)
class PairCandidate:
    """One selected pair and the grid indices needed to publish it."""

    platform_index_1: int
    platform_index_2: int
    platform_id_1: str
    platform_id_2: str
    overlap_start_index: int
    overlap_end_index: int
    encounter_index: int
    encounter_distance_m: float
    encounter_delay_seconds_1: float
    encounter_delay_seconds_2: float

    @property
    def post_encounter_observations(self) -> int:
        return self.overlap_end_index - self.encounter_index + 1


@dataclass(frozen=True)
class PairSearchResult:
    """Candidate pairs plus counts describing the search universe."""

    candidates: tuple[PairCandidate, ...]
    possible_pair_count: int
    overlapping_pair_count: int


def circular_mean_longitude(longitude_1, longitude_2):
    """Return the two-member circular mean in the interval [-180, 180)."""
    first = np.asarray(longitude_1, dtype=float)
    second = np.asarray(longitude_2, dtype=float)
    radians_1 = np.deg2rad(first)
    radians_2 = np.deg2rad(second)
    sine = np.sin(radians_1) + np.sin(radians_2)
    cosine = np.cos(radians_1) + np.cos(radians_2)
    result = np.rad2deg(np.arctan2(sine, cosine))
    result = (result + 180.0) % 360.0 - 180.0
    ambiguous = np.isclose(sine, 0.0, atol=1e-15) & np.isclose(cosine, 0.0, atol=1e-15)
    return np.where(ambiguous, np.nan, result)


def find_candidate_pairs(
    platform_ids: Sequence[str],
    time: np.ndarray,
    observed_start_time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    *,
    maximum_distance_m: float,
    maximum_seconds_from_each_observed_start: float | None,
    progress: Callable[[str], None] | None = None,
) -> PairSearchResult:
    """Find the first distance-threshold crossing for every eligible platform pair.

    A finite deployment-time limit is enforced independently for both members.
    Passing ``None`` searches each pair's entire common valid lifetime.
    """
    identifiers = np.asarray([str(value) for value in platform_ids], dtype=object)
    grid = np.asarray(time, dtype="datetime64[ns]")
    starts = np.asarray(observed_start_time, dtype="datetime64[ns]")
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    _validate_inputs(
        identifiers, grid, starts, lon, lat, maximum_distance_m,
        maximum_seconds_from_each_observed_start,
    )

    finite = np.isfinite(lon) & np.isfinite(lat)
    order = sorted(range(len(identifiers)), key=lambda index: identifiers[index])
    possible = len(order) * (len(order) - 1) // 2
    overlapping = 0
    candidates: list[PairCandidate] = []
    checked = 0
    time_ns = grid.astype(np.int64)
    starts_ns = starts.astype(np.int64)
    if maximum_seconds_from_each_observed_start is None:
        buffer_ns = None
    else:
        buffer_ns = int(round(maximum_seconds_from_each_observed_start * 1_000_000_000))

    for left_position, first_index in enumerate(order[:-1]):
        for second_index in order[left_position + 1:]:
            checked += 1
            common = np.flatnonzero(finite[first_index] & finite[second_index])
            if common.size == 0:
                continue
            overlapping += 1
            eligible = common
            if buffer_ns is not None:
                # Both deadlines must hold; the earlier deadline is therefore decisive.
                deadline_ns = min(
                    starts_ns[first_index] + buffer_ns,
                    starts_ns[second_index] + buffer_ns,
                )
                eligible = common[time_ns[common] <= deadline_ns]
            if eligible.size == 0:
                continue
            _azimuth_1, _azimuth_2, distances = WGS84.inv(
                lon[first_index, eligible], lat[first_index, eligible],
                lon[second_index, eligible], lat[second_index, eligible],
            )
            distances = np.asarray(distances, dtype=float)
            hits = np.flatnonzero(distances <= maximum_distance_m)
            if hits.size == 0:
                continue
            hit = int(hits[0])
            encounter_index = int(eligible[hit])
            candidates.append(PairCandidate(
                platform_index_1=first_index,
                platform_index_2=second_index,
                platform_id_1=str(identifiers[first_index]),
                platform_id_2=str(identifiers[second_index]),
                overlap_start_index=int(common[0]),
                overlap_end_index=int(common[-1]),
                encounter_index=encounter_index,
                encounter_distance_m=float(distances[hit]),
                encounter_delay_seconds_1=(
                    time_ns[encounter_index] - starts_ns[first_index]
                ) / 1_000_000_000,
                encounter_delay_seconds_2=(
                    time_ns[encounter_index] - starts_ns[second_index]
                ) / 1_000_000_000,
            ))
        if progress is not None and possible >= 1000 and checked % 1000 < len(order):
            progress(f"Pair search: evaluated {checked:,}/{possible:,} platform pairs")

    return PairSearchResult(tuple(candidates), possible, overlapping)


def _validate_inputs(
    identifiers: np.ndarray, grid: np.ndarray, starts: np.ndarray,
    longitude: np.ndarray, latitude: np.ndarray, maximum_distance_m: float,
    maximum_seconds_from_each_observed_start: float | None,
) -> None:
    if len(identifiers) < 2:
        raise ValueError("Pair discovery requires at least two platforms")
    if len(set(identifiers.tolist())) != len(identifiers):
        raise ValueError("Platform identifiers must be unique")
    if any(not value.strip() for value in identifiers):
        raise ValueError("Platform identifiers must be nonempty")
    if grid.ndim != 1 or starts.shape != (len(identifiers),):
        raise ValueError("Time must be one-dimensional and starts must match platforms")
    if np.isnat(grid).any() or np.isnat(starts).any():
        raise ValueError("Time and observed starts cannot contain NaT")
    if len(grid) == 0 or np.any(np.diff(grid.astype(np.int64)) <= 0):
        raise ValueError("Time must be nonempty, strictly increasing, and unique")
    expected = (len(identifiers), len(grid))
    if longitude.shape != expected or latitude.shape != expected:
        raise ValueError(f"Coordinates must have shape {expected}")
    valid = np.isfinite(longitude) & np.isfinite(latitude)
    if not valid.any(axis=1).all():
        raise ValueError("Every platform must have at least one finite coordinate pair")
    if np.any((longitude[valid] < -180) | (longitude[valid] >= 180)):
        raise ValueError("Longitudes must lie in [-180, 180)")
    if np.any((latitude[valid] < -90) | (latitude[valid] > 90)):
        raise ValueError("Latitudes must lie in [-90, 90]")
    if isinstance(maximum_distance_m, bool) or not math.isfinite(maximum_distance_m):
        raise ValueError("maximum_distance_m must be finite and positive")
    if maximum_distance_m <= 0:
        raise ValueError("maximum_distance_m must be finite and positive")
    if maximum_seconds_from_each_observed_start is not None:
        value = maximum_seconds_from_each_observed_start
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(
                "maximum_seconds_from_each_observed_start must be finite and nonnegative or null"
            )
