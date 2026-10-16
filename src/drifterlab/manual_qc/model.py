"""Pure state and calculations for independent manual trajectory review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from pyproj import CRS, Geod, Transformer

from drifterlab.io.position import NativeTrajectory
from drifterlab.reconstruction.core import resample_linear_track
from drifterlab.trajectories.kinematics import spherical_mean

NS_PER_SECOND = 1_000_000_000
SECONDS_PER_DAY = 86_400
GEOD = Geod(ellps="WGS84")


def _valid_coordinates(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    return (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180.0)
        & (longitude <= 180.0)
        & (latitude >= -90.0)
        & (latitude <= 90.0)
    )


def _utc_text(value: np.datetime64) -> str:
    if np.isnat(value):
        return "invalid timestamp"
    return str(np.datetime_as_string(np.datetime64(value, "ns"), unit="s", timezone="UTC"))


@dataclass(frozen=True)
class ReviewChunk:
    """One nonempty, immutable review interval."""

    key: str
    ordinal: int
    time_bin: int | None
    start_time: np.datetime64
    end_time: np.datetime64
    source_indices: tuple[int, ...]
    kind: str = "time"

    @property
    def label(self) -> str:
        if self.kind == "invalid_time":
            return f"Unplaced timestamps ({len(self.source_indices)} points)"
        return f"{_utc_text(self.start_time)} to {_utc_text(self.end_time)}"


@dataclass(frozen=True)
class KinematicSeries:
    """Adjacent-point geodesic kinematics assigned to source observations."""

    source_indices: np.ndarray
    time: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    edge_dt_seconds: np.ndarray
    edge_distance_m: np.ndarray
    speed_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    east_velocity_m_s: np.ndarray
    north_velocity_m_s: np.ndarray


@dataclass(frozen=True)
class ResampledTrajectory:
    """A regular-grid display derived from the currently accepted raw fixes."""

    time: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    source_gap_minutes: np.ndarray
    kinematics: KinematicSeries
    sst: np.ndarray
    sst_source_gap_minutes: np.ndarray


@dataclass(frozen=True)
class ProjectedPositions:
    """Raw positions in a fixed per-trajectory local metric projection."""

    x_m: np.ndarray
    y_m: np.ndarray
    forward: Transformer
    origin_longitude: float
    origin_latitude: float


def adjacent_kinematics(
    time: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
    source_indices: np.ndarray | None = None,
) -> KinematicSeries:
    """Calculate speed and vector acceleration from adjacent values only.

    Speed is assigned to the later endpoint. Acceleration is assigned to the
    shared raw point between adjacent velocity vectors and uses the elapsed
    time between the two edge midpoints. No cadence or anomaly threshold is
    assumed.
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    longitude = np.asarray(longitude, dtype=float)
    latitude = np.asarray(latitude, dtype=float)
    if source_indices is None:
        source_indices = np.arange(len(time), dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
    if not (
        time.ndim == longitude.ndim == latitude.ndim == source_indices.ndim == 1
        and len({len(time), len(longitude), len(latitude), len(source_indices)}) == 1
    ):
        raise ValueError("Kinematic inputs must be equally sized one-dimensional arrays")

    count = len(time)
    dt = np.full(count, np.nan, dtype=float)
    distance = np.full(count, np.nan, dtype=float)
    speed = np.full(count, np.nan, dtype=float)
    east = np.full(count, np.nan, dtype=float)
    north = np.full(count, np.nan, dtype=float)
    acceleration = np.full(count, np.nan, dtype=float)
    if count < 2:
        return KinematicSeries(
            source_indices, time, longitude, latitude, dt, distance, speed,
            acceleration, east, north,
        )

    time_ns = time.astype(np.int64)
    finite_time = ~np.isnat(time)
    coordinates = _valid_coordinates(longitude, latitude)
    edge_valid = finite_time[:-1] & finite_time[1:] & coordinates[:-1] & coordinates[1:]
    edge_dt = (time_ns[1:].astype(object) - time_ns[:-1].astype(object))
    edge_dt = np.asarray(edge_dt, dtype=float) / NS_PER_SECOND
    edge_valid &= edge_dt > 0
    valid_edges = np.flatnonzero(edge_valid)
    if len(valid_edges):
        azimuth, _back, lengths = GEOD.inv(
            longitude[valid_edges], latitude[valid_edges],
            longitude[valid_edges + 1], latitude[valid_edges + 1],
        )
        values = np.asarray(lengths, dtype=float) / edge_dt[valid_edges]
        radians = np.deg2rad(np.asarray(azimuth, dtype=float))
        destinations = valid_edges + 1
        dt[destinations] = edge_dt[valid_edges]
        distance[destinations] = np.asarray(lengths, dtype=float)
        speed[destinations] = values
        east[destinations] = values * np.sin(radians)
        north[destinations] = values * np.cos(radians)

    # Edge values ending at i and i+1 meet at raw point i. Their midpoint
    # separation is (left_dt + right_dt) / 2, including irregular sampling.
    if count > 2:
        midpoint_dt = (dt[1:-1] + dt[2:]) / 2
        delta_east = east[2:] - east[1:-1]
        delta_north = north[2:] - north[1:-1]
        valid_acceleration = (
            np.isfinite(midpoint_dt) & (midpoint_dt > 0)
            & np.isfinite(delta_east) & np.isfinite(delta_north)
        )
        middle_values = np.full(count - 2, np.nan, dtype=float)
        middle_values[valid_acceleration] = np.hypot(
            delta_east[valid_acceleration], delta_north[valid_acceleration],
        ) / midpoint_dt[valid_acceleration]
        acceleration[1:-1] = middle_values

    return KinematicSeries(
        source_indices, time, longitude, latitude, dt, distance, speed,
        acceleration, east, north,
    )


def speed_warning_masks(
    speed_m_s: np.ndarray,
    *,
    high_speed_m_s: float | None,
    zero_speed_tolerance_m_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return advisory high- and zero-speed masks for finite speeds."""

    speed = np.asarray(speed_m_s, dtype=float)
    if speed.ndim != 1:
        raise ValueError("speed_m_s must be one-dimensional")
    finite = np.isfinite(speed)

    high = np.zeros(len(speed), dtype=bool)
    if high_speed_m_s is not None:
        threshold = float(high_speed_m_s)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("high_speed_m_s must be finite and positive")
        high = finite & (speed > threshold)

    zero = np.zeros(len(speed), dtype=bool)
    if zero_speed_tolerance_m_s is not None:
        tolerance = float(zero_speed_tolerance_m_s)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "zero_speed_tolerance_m_s must be finite and nonnegative"
            )
        zero = finite & (speed <= tolerance)

    return high, zero


class ManualTrajectoryModel:
    """Mutable reject/restore state for one lazily loaded raw trajectory."""

    def __init__(
        self,
        trajectory: NativeTrajectory,
        *,
        chunk_seconds: float = 3600.0,
        stop_time: np.datetime64 | None = None,
        hard_stop_reason: str = "record_end",
        maximum_accepted_span_days: float | None = None,
        rejected_sources: set[int] | None = None,
        rejected_sst_sources: set[int] | None = None,
    ):
        if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be finite and positive")
        self.trajectory = trajectory
        self.platform_code = str(trajectory.platform_code)
        self.source_sha256 = str(trajectory.source_sha256)
        self.chunk_seconds = float(chunk_seconds)
        chunk_ns_float = self.chunk_seconds * NS_PER_SECOND
        if not chunk_ns_float.is_integer():
            raise ValueError("chunk_seconds must be exactly representable in nanoseconds")
        self.chunk_ns = int(chunk_ns_float)
        self.stop_time = np.datetime64("NaT" if stop_time is None else stop_time, "ns")
        configured_hard_stop_reason = str(hard_stop_reason)
        if maximum_accepted_span_days is None:
            self.maximum_accepted_span_days = None
            self.maximum_accepted_span_ns = None
        else:
            span = float(maximum_accepted_span_days)
            if not np.isfinite(span) or span <= 0:
                raise ValueError("maximum_accepted_span_days must be finite and positive")
            span_ns = span * SECONDS_PER_DAY * NS_PER_SECOND
            if not span_ns.is_integer():
                raise ValueError(
                    "maximum_accepted_span_days must be exactly representable in nanoseconds"
                )
            self.maximum_accepted_span_days = span
            self.maximum_accepted_span_ns = int(span_ns)
        self.time = np.asarray(trajectory.time, dtype="datetime64[ns]")
        self.longitude = np.asarray(trajectory.lon, dtype=float)
        self.latitude = np.asarray(trajectory.lat, dtype=float)
        self.sst = (
            np.full(len(self.time), np.nan, dtype=float)
            if trajectory.sst is None else np.asarray(trajectory.sst, dtype=float)
        )
        self.gps_pos_qual = (
            np.full(len(self.time), np.nan, dtype=float)
            if trajectory.gps_pos_qual is None
            else np.asarray(trajectory.gps_pos_qual, dtype=float)
        )
        self.gps_num_sat = (
            np.full(len(self.time), np.nan, dtype=float)
            if trajectory.gps_num_sat is None
            else np.asarray(trajectory.gps_num_sat, dtype=float)
        )
        self.sources = np.asarray(trajectory.source_obs_index, dtype=np.int64)
        self.source_to_position = {
            int(source): position for position, source in enumerate(self.sources)
        }
        if len(self.source_to_position) != len(self.sources):
            raise ValueError("Manual QC requires unique source observation indices")
        known = set(self.source_to_position)
        rejected = set(map(int, rejected_sources or set()))
        unknown = rejected - known
        if unknown:
            raise ValueError(f"Saved manual decisions contain unknown observations: {sorted(unknown)}")
        self.rejected_sources = rejected
        rejected_sst = set(map(int, rejected_sst_sources or set()))
        unknown_sst = rejected_sst - known
        if unknown_sst:
            raise ValueError(
                "Saved manual SST decisions contain unknown observations: "
                f"{sorted(unknown_sst)}"
        )
        self.rejected_sst_sources = rejected_sst

        valid_time = ~np.isnat(self.time)
        valid_positions = np.flatnonzero(valid_time)
        self.record_end_time = np.datetime64("NaT", "ns")
        if len(valid_positions):
            self.record_end_time = self.time[valid_positions].max()
        self.hard_stop_reason = configured_hard_stop_reason
        if (
            np.isnat(self.stop_time)
            or (
                not np.isnat(self.record_end_time)
                and self.stop_time > self.record_end_time
            )
        ):
            self.hard_stop_reason = "record_end"
        before_stop = valid_time.copy()
        if not np.isnat(self.stop_time):
            before_stop &= self.time <= self.stop_time
        self.hard_time_scope_mask = before_stop
        self.after_stop_mask = valid_time & ~before_stop
        self.invalid_time_mask = ~valid_time
        self.raw_first_observation_time = np.datetime64("NaT", "ns")
        self.hard_end_time = np.datetime64("NaT", "ns")
        hard_positions = np.flatnonzero(self.hard_time_scope_mask)
        if len(hard_positions):
            hard_times = self.time[hard_positions]
            self.raw_first_observation_time = hard_times.min()
            self.hard_end_time = hard_times.max()
        self.deployment_start_time = np.datetime64("NaT", "ns")
        self.review_end_time = np.datetime64("NaT", "ns")
        self.review_end_reason = self.hard_stop_reason
        self.accepted_review_span_days = 0.0
        self.in_time_scope_mask = self.hard_time_scope_mask.copy()
        self.after_review_mask = np.zeros(len(self.time), dtype=bool)
        self.chunks: tuple[ReviewChunk, ...] = ()
        self.source_to_chunk: dict[int, int] = {}
        self.review_order: list[int] = []
        self.review_position: dict[int, int] = {}
        self.current_chunk_index = 0
        self.selected_source: int | None = None
        self._projection: ProjectedPositions | None = None
        self.refresh_review_scope()

    def refresh_review_scope(self) -> None:
        """Recalculate the deployment start and accepted-duration review horizon."""
        previous_source = self.selected_source
        accepted = self.hard_time_scope_mask & _valid_coordinates(
            self.longitude, self.latitude,
        )
        if self.rejected_sources:
            accepted &= ~np.isin(self.sources, np.fromiter(
                self.rejected_sources, dtype=np.int64,
            ))
        accepted_positions = np.flatnonzero(accepted)
        if len(accepted_positions):
            ordered = accepted_positions[np.lexsort((
                self.sources[accepted_positions],
                self.time[accepted_positions].astype(np.int64),
            ))]
            self.deployment_start_time = self.time[ordered[0]]
            if self.maximum_accepted_span_ns is None:
                self.review_end_time = self.hard_end_time
                self.review_end_reason = self.hard_stop_reason
            else:
                target_ns = (
                    int(self.deployment_start_time.astype(np.int64))
                    + self.maximum_accepted_span_ns
                )
                qualifying = ordered[
                    self.time[ordered].astype(np.int64) >= target_ns
                ]
                if len(qualifying):
                    self.review_end_time = self.time[qualifying[0]]
                    self.review_end_reason = "maximum_span"
                else:
                    self.review_end_time = self.hard_end_time
                    self.review_end_reason = self.hard_stop_reason
        else:
            self.deployment_start_time = np.datetime64("NaT", "ns")
            self.review_end_time = self.hard_end_time
            self.review_end_reason = self.hard_stop_reason

        if np.isnat(self.review_end_time):
            self.in_time_scope_mask = np.zeros(len(self.time), dtype=bool)
        else:
            self.in_time_scope_mask = (
                self.hard_time_scope_mask & (self.time <= self.review_end_time)
            )
        self.after_review_mask = self.hard_time_scope_mask & ~self.in_time_scope_mask

        active_accepted = accepted & self.in_time_scope_mask
        active_positions = np.flatnonzero(active_accepted)
        if len(active_positions) and not np.isnat(self.deployment_start_time):
            last_time = self.time[active_positions].max()
            self.accepted_review_span_days = float(
                (last_time - self.deployment_start_time) / np.timedelta64(1, "D")
            )
        else:
            self.accepted_review_span_days = 0.0

        self.chunks = self._build_chunks()
        self.source_to_chunk = {}
        self.review_order = []
        for index, chunk in enumerate(self.chunks):
            for source in chunk.source_indices:
                self.source_to_chunk[source] = index
                self.review_order.append(source)
        self.review_position = {
            source: position for position, source in enumerate(self.review_order)
        }
        if previous_source in self.review_position:
            self.selected_source = previous_source
        elif self.review_order:
            self.selected_source = self.review_order[-1] if previous_source is not None else self.review_order[0]
        else:
            self.selected_source = None
        self.current_chunk_index = (
            self.source_to_chunk[self.selected_source]
            if self.selected_source is not None else 0
        )
        self._projection = None

    def _chunk_key(self, kind: str, time_bin: int | None, sources: tuple[int, ...]) -> str:
        payload = json.dumps(
            [self.platform_code, self.source_sha256, kind, time_bin, self.chunk_ns, sources],
            separators=(",", ":"),
        )
        return sha256(payload.encode()).hexdigest()[:24]

    def _build_chunks(self) -> tuple[ReviewChunk, ...]:
        positions = np.flatnonzero(self.in_time_scope_mask)
        chunks: list[ReviewChunk] = []
        if len(positions):
            order = positions[np.lexsort((self.sources[positions], self.time[positions].astype(np.int64)))]
            anchor = int(self.time[order[0]].astype(np.int64))
            bins = (self.time[order].astype(np.int64) - anchor) // self.chunk_ns
            for time_bin in np.unique(bins):
                members = order[bins == time_bin]
                sources = tuple(map(int, self.sources[members]))
                start_ns = anchor + int(time_bin) * self.chunk_ns
                end_ns = start_ns + self.chunk_ns
                chunks.append(ReviewChunk(
                    key=self._chunk_key("time", int(time_bin), sources),
                    ordinal=len(chunks), time_bin=int(time_bin),
                    start_time=np.datetime64(start_ns, "ns"),
                    end_time=np.datetime64(end_ns, "ns"),
                    source_indices=sources,
                ))
        invalid_positions = np.flatnonzero(self.invalid_time_mask)
        if len(invalid_positions):
            invalid_positions = invalid_positions[np.argsort(
                self.sources[invalid_positions], kind="stable",
            )]
            sources = tuple(map(int, self.sources[invalid_positions]))
            chunks.append(ReviewChunk(
                key=self._chunk_key("invalid_time", None, sources),
                ordinal=len(chunks), time_bin=None,
                start_time=np.datetime64("NaT", "ns"),
                end_time=np.datetime64("NaT", "ns"),
                source_indices=sources, kind="invalid_time",
            ))
        return tuple(chunks)

    @property
    def current_chunk(self) -> ReviewChunk | None:
        return self.chunks[self.current_chunk_index] if self.chunks else None

    def set_selected_source(self, source_index: int) -> None:
        source = int(source_index)
        if source not in self.review_position:
            raise ValueError(f"Observation {source} is outside the manual review scope")
        self.selected_source = source
        self.current_chunk_index = self.source_to_chunk[source]

    def move_cursor(self, step: int) -> int | None:
        if not self.review_order:
            return None
        position = self.review_position.get(int(self.selected_source), 0)
        position = max(0, min(len(self.review_order) - 1, position + int(step)))
        self.set_selected_source(self.review_order[position])
        return self.selected_source

    def move_chunk(self, step: int) -> ReviewChunk | None:
        if not self.chunks:
            return None
        self.current_chunk_index = max(
            0, min(len(self.chunks) - 1, self.current_chunk_index + int(step)),
        )
        chunk = self.current_chunk
        assert chunk is not None
        self.selected_source = (
            chunk.source_indices[-1] if step < 0 else chunk.source_indices[0]
        )
        return chunk

    def set_chunk(self, index: int) -> ReviewChunk:
        if not 0 <= int(index) < len(self.chunks):
            raise IndexError(index)
        self.current_chunk_index = int(index)
        chunk = self.chunks[self.current_chunk_index]
        if self.selected_source not in chunk.source_indices:
            self.selected_source = chunk.source_indices[0]
        return chunk

    def set_rejected(self, source_indices: list[int] | tuple[int, ...] | set[int], rejected: bool) -> set[int]:
        changed: set[int] = set()
        for value in source_indices:
            source = int(value)
            if source not in self.review_position:
                raise ValueError(f"Observation {source} is outside the manual review scope")
            if rejected and source not in self.rejected_sources:
                self.rejected_sources.add(source)
                changed.add(source)
            elif not rejected and source in self.rejected_sources:
                self.rejected_sources.remove(source)
                changed.add(source)
        if changed:
            self.refresh_review_scope()
        return changed

    def set_sst_rejected(
        self,
        source_indices: list[int] | tuple[int, ...] | set[int],
        rejected: bool,
    ) -> set[int]:
        changed: set[int] = set()
        for value in source_indices:
            source = int(value)
            if source not in self.review_position:
                raise ValueError(f"Observation {source} is outside the manual review scope")
            position = self.source_to_position[source]
            if not np.isfinite(self.sst[position]):
                raise ValueError(f"Observation {source} has no finite SST value")
            if rejected and source not in self.rejected_sst_sources:
                self.rejected_sst_sources.add(source)
                changed.add(source)
            elif not rejected and source in self.rejected_sst_sources:
                self.rejected_sst_sources.remove(source)
                changed.add(source)
        return changed

    def affected_chunk_keys(self, source_indices: set[int] | list[int]) -> set[str]:
        """Return edited chunks and neighbours whose bridged kinematics may change."""
        affected: set[str] = set()
        for source in source_indices:
            chunk_index = self.source_to_chunk.get(int(source))
            if chunk_index is None:
                continue
            for index in range(max(0, chunk_index - 1), min(len(self.chunks), chunk_index + 2)):
                affected.add(self.chunks[index].key)
        return affected

    def affected_sst_chunk_keys(self, source_indices: set[int] | list[int]) -> set[str]:
        """Return chunks spanned by the SST interpolation changed by an edit."""
        accepted = [
            source for source in self.review_order
            if source not in self.rejected_sst_sources
            and np.isfinite(self.sst[self.source_to_position[source]])
            and not np.isnat(self.time[self.source_to_position[source]])
        ]
        accepted_review_positions = np.asarray(
            [self.review_position[source] for source in accepted], dtype=np.int64,
        )
        affected: set[str] = set()
        for value in source_indices:
            source = int(value)
            chunk_index = self.source_to_chunk.get(source)
            if chunk_index is None:
                continue
            review_position = self.review_position[source]
            candidates = accepted_review_positions[
                accepted_review_positions != review_position
            ]
            lower = candidates[candidates < review_position]
            upper = candidates[candidates > review_position]
            first = chunk_index
            last = chunk_index
            if len(lower):
                left_source = self.review_order[int(lower[-1])]
                first = min(first, self.source_to_chunk[left_source])
            if len(upper):
                right_source = self.review_order[int(upper[0])]
                last = max(last, self.source_to_chunk[right_source])
            for index in range(first, last + 1):
                affected.add(self.chunks[index].key)
        return affected

    def _ordered_positions(self, *, accepted_only: bool) -> np.ndarray:
        positions = np.flatnonzero(self.in_time_scope_mask)
        if accepted_only and len(positions):
            keep = np.asarray([
                int(self.sources[position]) not in self.rejected_sources
                for position in positions
            ], dtype=bool)
            positions = positions[keep]
        if not len(positions):
            return positions
        return positions[np.lexsort((self.sources[positions], self.time[positions].astype(np.int64)))]

    def raw_kinematics(self, *, accepted_only: bool = True) -> KinematicSeries:
        positions = self._ordered_positions(accepted_only=accepted_only)
        return adjacent_kinematics(
            self.time[positions], self.longitude[positions], self.latitude[positions],
            self.sources[positions],
        )

    def _ordered_sst_positions(self, *, accepted_only: bool) -> np.ndarray:
        positions = np.asarray([
            self.source_to_position[source] for source in self.review_order
            if np.isfinite(self.sst[self.source_to_position[source]])
            and not np.isnat(self.time[self.source_to_position[source]])
            and (
                not accepted_only or source not in self.rejected_sst_sources
            )
        ], dtype=np.int64)
        if not len(positions):
            return positions
        return positions[np.lexsort((
            self.sources[positions], self.time[positions].astype(np.int64),
        ))]

    def move_sst_cursor(self, step: int) -> int | None:
        sources = [
            int(self.sources[position])
            for position in self._ordered_sst_positions(accepted_only=False)
        ]
        if not sources:
            return None
        current = int(self.selected_source) if self.selected_source is not None else sources[0]
        if current in sources:
            index = sources.index(current)
        else:
            review_position = self.review_position.get(current, 0)
            ordered_positions = [self.review_position[source] for source in sources]
            index = int(np.searchsorted(ordered_positions, review_position))
            if step < 0:
                index -= 1
        index = max(0, min(len(sources) - 1, index + (step if current in sources else 0)))
        self.set_selected_source(sources[index])
        return self.selected_source

    def _resample_sst(
        self, grid: np.ndarray, maximum_gap_minutes: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = self._ordered_sst_positions(accepted_only=True)
        values = np.full(len(grid), np.nan, dtype=float)
        gaps = np.full(len(grid), np.nan, dtype=float)
        if len(positions) < 2:
            return values, gaps
        time = self.time[positions]
        time_ns = time.astype(np.int64)
        if np.any(np.diff(time_ns) <= 0):
            duplicated = self.sources[
                positions[np.r_[False, np.diff(time_ns) <= 0]]
            ].astype(int)
            raise ValueError(
                "Accepted SST timestamps must be strictly increasing before resampling; "
                f"review SST observations {duplicated[:12].tolist()}"
            )
        grid_ns = np.asarray(grid, dtype="datetime64[ns]").astype(np.int64)
        inside = (grid_ns >= time_ns[0]) & (grid_ns <= time_ns[-1])
        inside_indices = np.flatnonzero(inside)
        relative_time = (time_ns - time_ns[0]) / NS_PER_SECOND
        relative_grid = (grid_ns[inside] - time_ns[0]) / NS_PER_SECOND
        values[inside] = np.interp(relative_grid, relative_time, self.sst[positions])
        right = np.searchsorted(time_ns, grid_ns[inside], side="left")
        clipped = np.minimum(right, len(time_ns) - 1)
        exact = (right < len(time_ns)) & (time_ns[clipped] == grid_ns[inside])
        gaps[inside_indices[exact]] = 0.0
        between = ~exact
        if between.any():
            gap_minutes = (
                time_ns[right[between]] - time_ns[right[between] - 1]
            ) / (60 * NS_PER_SECOND)
            target = inside_indices[between]
            gaps[target] = gap_minutes
            if maximum_gap_minutes is not None:
                values[target[gap_minutes > maximum_gap_minutes]] = np.nan
        return values, gaps

    def resample(
        self,
        frequency_seconds: float,
        *,
        sst_maximum_gap_minutes: float | None = None,
    ) -> ResampledTrajectory:
        if not np.isfinite(frequency_seconds) or frequency_seconds <= 0:
            raise ValueError("Resampling frequency must be finite and positive")
        step_float = float(frequency_seconds) * NS_PER_SECOND
        if not step_float.is_integer():
            raise ValueError("Resampling frequency must be exactly representable in nanoseconds")
        positions = self._ordered_positions(accepted_only=True)
        if len(positions) < 2:
            raise ValueError("At least two accepted raw observations are required to resample")
        coordinates = _valid_coordinates(self.longitude[positions], self.latitude[positions])
        if not coordinates.all():
            sources = self.sources[positions[~coordinates]].astype(int).tolist()
            raise ValueError(
                "Reject or repair accepted observations with invalid coordinates before resampling: "
                f"{sources[:12]}"
            )
        times = self.time[positions]
        time_ns = times.astype(np.int64)
        if np.any(np.diff(time_ns) <= 0):
            duplicated = self.sources[positions[np.r_[False, np.diff(time_ns) <= 0]]].astype(int)
            raise ValueError(
                "Accepted timestamps must be strictly increasing before resampling; "
                f"review observations {duplicated[:12].tolist()}"
            )
        step_ns = int(step_float)
        count = int((time_ns[-1] - time_ns[0]) // step_ns) + 1
        grid_ns = time_ns[0] + np.arange(count, dtype=np.int64) * step_ns
        grid = grid_ns.astype("datetime64[ns]")
        result = resample_linear_track(
            times, self.longitude[positions], self.latitude[positions], grid,
            coverage_start=grid[0], coverage_end=grid[-1],
        )
        kinematics = adjacent_kinematics(
            grid, result.longitude, result.latitude,
            np.arange(len(grid), dtype=np.int64),
        )
        sst, sst_gap = self._resample_sst(grid, sst_maximum_gap_minutes)
        return ResampledTrajectory(
            grid, result.longitude, result.latitude, result.source_gap_minutes,
            kinematics, sst, sst_gap,
        )

    def projected_positions(self) -> ProjectedPositions:
        if self._projection is not None:
            return self._projection
        candidates = self.in_time_scope_mask & _valid_coordinates(self.longitude, self.latitude)
        if not candidates.any():
            raise ValueError("Trajectory has no projectable in-scope positions")
        origin_lon, origin_lat = spherical_mean(
            self.longitude[candidates], self.latitude[candidates],
        )
        definition = (
            f"+proj=aeqd +lat_0={origin_lat:.15g} +lon_0={origin_lon:.15g} "
            "+datum=WGS84 +units=m +no_defs +type=crs"
        )
        forward = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_proj4(definition), always_xy=True,
        )
        x = np.full(len(self.sources), np.nan, dtype=float)
        y = np.full(len(self.sources), np.nan, dtype=float)
        valid = _valid_coordinates(self.longitude, self.latitude)
        x[valid], y[valid] = forward.transform(self.longitude[valid], self.latitude[valid])
        self._projection = ProjectedPositions(
            x, y, forward, origin_lon, origin_lat,
        )
        return self._projection

    def completion_issues(self, reviewed_chunk_keys: set[str]) -> list[str]:
        issues: list[str] = []
        pending = [chunk for chunk in self.chunks if chunk.key not in reviewed_chunk_keys]
        if pending:
            issues.append(f"{len(pending)} review chunks are not complete")
        accepted_positions = [
            self.source_to_position[source] for source in self.review_order
            if source not in self.rejected_sources
        ]
        if len(accepted_positions) < 2:
            issues.append("fewer than two raw observations remain accepted")
            return issues
        accepted_positions = np.asarray(accepted_positions, dtype=np.int64)
        invalid_time = np.isnat(self.time[accepted_positions])
        if invalid_time.any():
            issues.append(f"{int(invalid_time.sum())} accepted observations have invalid timestamps")
        invalid_position = ~_valid_coordinates(
            self.longitude[accepted_positions], self.latitude[accepted_positions],
        )
        if invalid_position.any():
            issues.append(f"{int(invalid_position.sum())} accepted observations have invalid coordinates")
        valid = ~(invalid_time | invalid_position)
        ordered_times = np.sort(self.time[accepted_positions[valid]].astype(np.int64))
        if len(ordered_times) > 1 and np.any(np.diff(ordered_times) <= 0):
            issues.append("accepted timestamps are not unique")
        accepted_sst_positions = np.asarray([
            self.source_to_position[source] for source in self.review_order
            if source not in self.rejected_sst_sources
            and np.isfinite(self.sst[self.source_to_position[source]])
        ], dtype=np.int64)
        if len(accepted_sst_positions):
            invalid_sst_time = np.isnat(self.time[accepted_sst_positions])
            valid_sst_times = np.sort(
                self.time[accepted_sst_positions[~invalid_sst_time]].astype(np.int64)
            )
            if len(valid_sst_times) > 1 and np.any(np.diff(valid_sst_times) <= 0):
                issues.append("accepted SST timestamps are not unique")
        return issues
