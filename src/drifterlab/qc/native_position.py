"""Generic native-position QC with temporal and local geometric evidence."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import struct
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from drifterlab.io.position import NativeTrajectory
from drifterlab.qc.drogue import ResolvedDrogueDecision
from drifterlab.qc.flags import EARTH_RADIUS_M
from drifterlab.qc.position import position_valid


@dataclass(frozen=True)
class TemporalSegmentConfig:
    segment_gap_hours: float = 6.0
    fragment_max_observation_fraction: float = .10
    fragment_max_duration_fraction: float = .10
    minimum_main_duration_hours: float = 48.0
    minimum_main_observations: int = 48


@dataclass(frozen=True)
class NativePositionConfig:
    nominal_interval_seconds: float = 300.0
    nominal_interval_tolerance_seconds: float = 1.0
    minimum_local_dt_seconds: float = 150.0
    max_local_gap_seconds: float = 1800.0
    gap_tolerance_seconds: float = 1.0
    speed_threshold_m_s: float = 3.0
    max_automatic_removal_points: int = 5
    local_speed_window_points: int = 15
    endpoint_speed_min_samples: int = 8
    bridge_speed_warning_z: float = 3.0
    local_speed_scale_floor_m_s: float = .05
    endpoint_speed_score_margin_z: float = 2.0


@dataclass(frozen=True)
class PositionReviewConfig:
    context_points: int = 12
    merge_gap_edges: int = 2


@dataclass(frozen=True)
class DeploymentBoundary:
    deployment_time: np.datetime64 = np.datetime64("NaT", "ns")
    window_start: np.datetime64 = np.datetime64("NaT", "ns")
    window_end: np.datetime64 = np.datetime64("NaT", "ns")
    provenance: str = ""
    note: str = ""


@dataclass(frozen=True)
class PositionResolution:
    final_position_status: np.ndarray
    final_position_valid: np.ndarray
    decision_source: np.ndarray
    platform_qc_complete: bool
    reconstruction_available: bool


POINT_REVIEW_DECISIONS = {"keep", "reject", "uncertain"}
SEGMENT_REVIEW_DECISIONS = {"retain", "exclude", "uncertain"}


def _event_id(platform: str, event_type: str, values: Iterable[Any]) -> str:
    text = json.dumps([platform, event_type, *list(values)], separators=(",", ":"), default=str)
    return f"{event_type}:{sha256(text.encode()).hexdigest()[:16]}"


def _segment_fingerprint(source_sha256: str, indices: np.ndarray) -> str:
    payload = source_sha256.encode() + np.asarray(indices, dtype=">i8").tobytes()
    return sha256(payload).hexdigest()


def _repeat_id(platform: str, lon: float, lat: float) -> str:
    payload = platform.encode() + struct.pack(">dd", float(lon), float(lat))
    return f"repeated_position:{sha256(payload).hexdigest()[:16]}"


def _seconds(left: np.datetime64, right: np.datetime64) -> float:
    left = np.datetime64(left, "ns")
    right = np.datetime64(right, "ns")
    if np.isnat(left) or np.isnat(right):
        return np.nan
    # Convert each epoch value to a Python integer before subtraction so dates
    # near the opposite datetime64 limits cannot overflow int64.
    return (int(right.astype(np.int64)) - int(left.astype(np.int64))) / 1e9


def _distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not np.isfinite([lon1, lat1, lon2, lat2]).all():
        return np.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlon = math.radians(lon2 - lon1)
    hav = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, hav))))


def position_edge_metrics(
    time: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    predecessor: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate display/audit edge metrics with the exact native-QC formulas.

    This helper performs no classification.  It exists so presentation code can
    show raw edge sequences without copying the scientific distance/time logic.
    """
    time = np.asarray(time, dtype="datetime64[ns]")
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    predecessor = np.asarray(predecessor, dtype=np.int64)
    current = np.asarray(current, dtype=np.int64)
    if predecessor.shape != current.shape:
        raise ValueError("Edge endpoint arrays must have matching shapes")
    dt = np.full(predecessor.shape, np.nan, dtype=float)
    distance = np.full(predecessor.shape, np.nan, dtype=float)
    for output_index, (a, b) in enumerate(zip(predecessor.flat, current.flat)):
        left, right = time[int(a)], time[int(b)]
        if not np.isnat(left) and not np.isnat(right):
            dt.flat[output_index] = (
                int(right.astype(np.int64)) - int(left.astype(np.int64))
            ) / 1e9
        distance.flat[output_index] = _distance(
            lon[int(a)], lat[int(a)], lon[int(b)], lat[int(b)],
        )
    speed = np.divide(
        distance, dt, out=np.full(predecessor.shape, np.nan, dtype=float),
        where=np.isfinite(distance) & (dt > 0),
    )
    return dt, distance, speed


def _edge_arrays(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, a: int, b: int,
) -> tuple[float, float, float]:
    left, right = time[a], time[b]
    dt = np.nan if np.isnat(left) or np.isnat(right) else (
        int(right.astype(np.int64)) - int(left.astype(np.int64))
    ) / 1e9
    distance = _distance(lon[a], lat[a], lon[b], lat[b])
    speed = distance / dt if np.isfinite(distance) and dt > 0 else np.nan
    return dt, distance, speed


def _edge_velocity_vector(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, a: int, b: int,
) -> tuple[float, float, float]:
    """Return elapsed seconds and local east/north velocity for one edge."""
    dt, _distance_m, speed = _edge_arrays(time, lon, lat, a, b)
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(speed):
        return dt, np.nan, np.nan
    lon1, lon2 = math.radians(float(lon[a])), math.radians(float(lon[b]))
    lat1, lat2 = math.radians(float(lat[a])), math.radians(float(lat[b]))
    delta_lon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    bearing = math.atan2(
        math.sin(delta_lon) * math.cos(lat2),
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon),
    )
    return dt, speed * math.sin(bearing), speed * math.cos(bearing)


def _vertex_acceleration(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, a: int, b: int, c: int,
) -> float:
    """Vector acceleration at B using the separation of the edge midpoints."""
    left_dt, left_east, left_north = _edge_velocity_vector(time, lon, lat, a, b)
    right_dt, right_east, right_north = _edge_velocity_vector(time, lon, lat, b, c)
    midpoint_dt = (left_dt + right_dt) / 2
    if (not np.isfinite(midpoint_dt) or midpoint_dt <= 0
            or not np.isfinite([left_east, left_north, right_east, right_north]).all()):
        return np.nan
    return float(math.hypot(right_east - left_east, right_north - left_north) / midpoint_dt)


def _edge(frame: pd.DataFrame, a: int, b: int) -> tuple[float, float, float]:
    return _edge_arrays(
        frame.time_value.to_numpy(dtype="datetime64[ns]"),
        frame.source_lon.to_numpy(dtype=float), frame.source_lat.to_numpy(dtype=float),
        a, b,
    )


def _review_maps(reviews: pd.DataFrame | None, platform: str) -> tuple[dict[int, dict], dict[str, dict]]:
    points: dict[int, dict] = {}
    segments: dict[str, dict] = {}
    if reviews is None or reviews.empty:
        return points, segments
    selected = reviews[reviews.platform_code.astype(str) == str(platform)]
    for row in selected.to_dict("records"):
        if row["target_type"] == "observation":
            points[int(row["source_obs_index"])] = row
        elif row["target_type"] == "segment":
            segments[str(row["target_id"])] = row
    return points, segments


def _base_frame(trajectory: NativeTrajectory, drogue: ResolvedDrogueDecision,
                deployment: DeploymentBoundary, point_reviews: Mapping[int, dict]) -> pd.DataFrame:
    n = len(trajectory.time)
    time = np.asarray(trajectory.time, dtype="datetime64[ns]")
    lon = np.asarray(trajectory.lon, dtype=float)
    lat = np.asarray(trajectory.lat, dtype=float)
    valid_time = ~np.isnat(time)
    valid_position = position_valid(lon, lat)
    frame = pd.DataFrame({
        "platform_code": np.repeat(trajectory.platform_code, n),
        "source_obs_index": np.asarray(trajectory.source_obs_index, dtype=np.int64),
        "source_path": np.repeat(str(trajectory.source_path), n),
        "source_sha256": np.repeat(trajectory.source_sha256, n),
        "time": pd.to_datetime(time, utc=True),
        "time_value": time,
        # Coordinates are immutable source values and are part of the published
        # audit record.  Only ``time_value`` is a private working column.
        "source_lon": lon,
        "source_lat": lat,
        "valid_timestamp": valid_time,
        "source_position_valid": valid_position,
    })
    frame["source_invalid_reason"] = np.where(
        ~valid_time, "invalid_timestamp", np.where(~valid_position, "invalid_position", "")
    )
    frame["final_drogue_status"] = drogue.final_drogue_status
    frame["final_drogue_loss_time"] = pd.to_datetime(drogue.final_drogue_loss_time, utc=True)
    frame["analysis_cutoff_time"] = pd.to_datetime(drogue.analysis_cutoff_time, utc=True)
    frame["drogue_decision_source"] = drogue.decision_source
    if drogue.final_drogue_status == "lost":
        eligible = valid_time & (time < drogue.analysis_cutoff_time)
    elif drogue.final_drogue_status == "not_lost":
        eligible = valid_time.copy()
    elif drogue.final_drogue_status == "uncertain":
        eligible = np.zeros(n, dtype=bool)
    else:
        raise ValueError(f"Unsupported resolved drogue status: {drogue.final_drogue_status}")
    frame["drogue_eligible"] = eligible
    frame["deployment_status"] = "not_configured"
    deployment_eligible = np.ones(n, dtype=bool)
    deployment_uncertain = np.zeros(n, dtype=bool)
    if not np.isnat(deployment.deployment_time):
        before = valid_time & (time < deployment.deployment_time)
        deployment_eligible[before] = False
        frame.loc[before, "deployment_status"] = "predeployment"
        frame.loc[valid_time & ~before, "deployment_status"] = "eligible"
    elif not np.isnat(deployment.window_start):
        before = valid_time & (time < deployment.window_start)
        inside = valid_time & (time >= deployment.window_start) & (time < deployment.window_end)
        deployment_eligible[before | inside] = False
        deployment_uncertain[inside] = True
        frame.loc[before, "deployment_status"] = "predeployment"
        frame.loc[inside, "deployment_status"] = "deployment_window_uncertain"
        frame.loc[valid_time & (time >= deployment.window_end), "deployment_status"] = "eligible"
    frame["deployment_eligible"] = deployment_eligible
    frame["deployment_uncertain"] = deployment_uncertain
    frame["deployment_provenance"] = deployment.provenance
    frame["human_position_decision"] = ""
    frame["human_decision_source"] = ""
    frame["human_review_event_id"] = ""
    frame["human_review_event_type"] = ""
    frame["human_auto_status_snapshot"] = ""
    frame["human_auto_decision_snapshot"] = ""
    frame["human_auto_reason_snapshot"] = ""
    frame["human_review_config_sha256"] = ""
    for j, source_index in enumerate(frame.source_obs_index):
        row = point_reviews.get(int(source_index))
        if row:
            frame.at[j, "human_position_decision"] = str(row["decision"])
            frame.at[j, "human_decision_source"] = str(row["decision_source"])
            frame.at[j, "human_review_event_id"] = str(row.get("event_id", ""))
            frame.at[j, "human_review_event_type"] = str(row.get("event_type", ""))
            frame.at[j, "human_auto_status_snapshot"] = str(row.get("auto_status_snapshot", ""))
            frame.at[j, "human_auto_decision_snapshot"] = str(row.get("auto_decision_snapshot", ""))
            frame.at[j, "human_auto_reason_snapshot"] = str(row.get("auto_reason_snapshot", ""))
            frame.at[j, "human_review_config_sha256"] = str(row.get("qc_config_sha256", ""))
    return frame


def _segments(frame: pd.DataFrame, source_sha256: str, config: TemporalSegmentConfig,
              segment_reviews: Mapping[str, dict]) -> None:
    n = len(frame)
    defaults: dict[str, Any] = {
        "segment_id": "", "segment_fingerprint": "", "segment_start_time": pd.NaT,
        "segment_end_time": pd.NaT, "segment_n_observations": 0,
        "segment_duration_hours": np.nan, "segment_median_dt_seconds": np.nan,
        "segment_fraction_total_observations": np.nan,
        "segment_fraction_total_duration": np.nan, "segment_gap_before_hours": np.nan,
        "segment_gap_after_hours": np.nan, "segment_status": "not_applicable",
        "temporal_event_id": "", "temporal_event_type": "",
        "temporal_auto_status": "", "temporal_auto_decision": "",
        "temporal_auto_reason": "", "human_segment_decision": "",
        "human_segment_review_config_sha256": "",
    }
    for name, value in defaults.items():
        if name in {"segment_start_time", "segment_end_time"}:
            frame[name] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        else:
            frame[name] = value
    eligible = (
        frame.valid_timestamp.to_numpy() & frame.source_position_valid.to_numpy()
        & frame.drogue_eligible.to_numpy() & frame.deployment_eligible.to_numpy()
    )
    ordered = np.flatnonzero(eligible)
    if not len(ordered):
        return
    times = frame.time_value.to_numpy(dtype="datetime64[ns]")
    ordered = ordered[np.argsort(times[ordered], kind="stable")]
    gap_seconds = config.segment_gap_hours * 3600
    labels = np.zeros(len(ordered), dtype=np.int64)
    current = 0
    previous_distinct = times[ordered[0]]
    for k in range(1, len(ordered)):
        value = times[ordered[k]]
        if value != previous_distinct:
            if _seconds(previous_distinct, value) > gap_seconds:
                current += 1
            previous_distinct = value
        labels[k] = current
    infos: list[dict[str, Any]] = []
    for number in range(current + 1):
        members = ordered[labels == number]
        unique_times = np.unique(times[members])
        positive = np.diff(unique_times).astype("timedelta64[ns]") / np.timedelta64(1, "s")
        duration = _seconds(unique_times[0], unique_times[-1]) / 3600 if len(unique_times) else np.nan
        source_indices = frame.source_obs_index.to_numpy()[members]
        fingerprint = _segment_fingerprint(source_sha256, np.sort(source_indices))
        infos.append({
            "number": number, "members": members, "start": unique_times[0], "end": unique_times[-1],
            "n": len(members), "duration": float(duration),
            "median": float(np.median(positive)) if len(positive) else np.nan,
            "fingerprint": fingerprint,
        })
    total_duration = sum(item["duration"] for item in infos)
    for i, item in enumerate(infos):
        item["gap_before"] = np.nan if i == 0 else _seconds(infos[i - 1]["end"], item["start"]) / 3600
        item["gap_after"] = np.nan if i + 1 == len(infos) else _seconds(item["end"], infos[i + 1]["start"]) / 3600
        item["obs_fraction"] = item["n"] / len(ordered)
        item["duration_fraction"] = item["duration"] / total_duration if total_duration > 0 else np.nan

    best = max((item["n"], item["duration"]) for item in infos)
    leaders = [item for item in infos if (item["n"], item["duration"]) == best]
    dominant = leaders[0] if len(leaders) == 1 else None
    tiny: set[int] = set()
    if dominant and dominant["n"] >= config.minimum_main_observations \
            and dominant["duration"] >= config.minimum_main_duration_hours:
        for item in infos:
            if item is dominant:
                continue
            if (item["n"] / dominant["n"] <= config.fragment_max_observation_fraction
                    and item["duration"] / dominant["duration"] <= config.fragment_max_duration_fraction):
                tiny.add(item["number"])
    substantial = [item for item in infos if item["number"] not in tiny]
    earliest = min((item["number"] for item in substantial), default=0)
    latest = max((item["number"] for item in substantial), default=0)

    for item in infos:
        members = item["members"]
        review = segment_reviews.get(item["fingerprint"])
        human = str(review["decision"]) if review else ""
        if item["number"] in tiny:
            status, auto_decision, auto_status, reason = (
                "excluded", "exclude", "high_confidence_detached_fragment",
                "small_observation_and_duration_fraction_relative_to_sustained_dominant_segment",
            )
        elif len(substantial) == 1:
            status, auto_decision, auto_status, reason = "retained", "retain", "automatic_retain", "only_substantial_segment"
        else:
            status, auto_decision, auto_status, reason = (
                "pending", "", "multiple_substantial_segments", "all_substantial_segments_require_review",
            )
        if human == "retain":
            status = "retained"
        elif human == "exclude":
            status = "excluded"
        elif human == "uncertain":
            status = "uncertain"
        if item["number"] in tiny:
            event_type = ("pre_main_fragment" if item["number"] < earliest else
                          "post_main_fragment" if item["number"] > latest else "detached_middle_fragment")
        elif auto_status == "multiple_substantial_segments":
            event_type = "multiple_substantial_segments"
        else:
            event_type = ""
        event_id = _event_id(str(frame.platform_code.iloc[0]), "temporal_segment", [item["fingerprint"]]) if event_type else ""
        values = {
            "segment_id": f"s{item['number'] + 1:04d}", "segment_fingerprint": item["fingerprint"],
            "segment_start_time": pd.Timestamp(item["start"], tz="UTC"),
            "segment_end_time": pd.Timestamp(item["end"], tz="UTC"),
            "segment_n_observations": item["n"], "segment_duration_hours": item["duration"],
            "segment_median_dt_seconds": item["median"],
            "segment_fraction_total_observations": item["obs_fraction"],
            "segment_fraction_total_duration": item["duration_fraction"],
            "segment_gap_before_hours": item["gap_before"], "segment_gap_after_hours": item["gap_after"],
            "segment_status": status, "temporal_event_id": event_id,
            "temporal_event_type": event_type, "temporal_auto_status": auto_status,
            "temporal_auto_decision": auto_decision, "temporal_auto_reason": reason,
            "human_segment_decision": human,
            "human_segment_review_config_sha256": (
                str(review.get("qc_config_sha256", "")) if review else ""
            ),
        }
        for name, value in values.items():
            frame.loc[members, name] = value


def _walk(
    order: list[int], plausible, speed, anomalous, elapsed,
    protected: np.ndarray, maximum_elapsed_seconds: float,
    maximum_removal_points: int,
):
    blocks: dict[tuple[int, tuple[int, ...], int], dict[str, Any]] = {}
    retained: set[int] = set()
    links: dict[int, tuple[int, Any]] = {}
    i, trusted = 0, False
    while i < len(order):
        if not trusted:
            if i + 2 >= len(order):
                break
            a, b, c = order[i:i + 3]
            if plausible(a, b) and plausible(b, c):
                retained.update((a, b, c)); links[a] = (b, None); links[b] = (c, None)
                i += 2; trusted = True
            else:
                i += 1
            continue
        if i + 1 >= len(order):
            break
        a, b = order[i:i + 2]
        if plausible(a, b):
            retained.add(b); links[a] = (b, None); i += 1
            continue
        recovered = False
        if anomalous(a, b):
            attempts: list[dict[str, Any]] = []
            for count in range(1, min(maximum_removal_points, len(order) - i - 2) + 1):
                if i + count + 1 >= len(order):
                    break
                skipped_order = order[i + 1:i + count + 1]
                skipped = tuple(sorted(skipped_order))
                reconnect = order[i + count + 1]
                bridge_dt = elapsed(a, reconnect)
                if (not np.isfinite(bridge_dt) or bridge_dt <= 0
                        or bridge_dt > maximum_elapsed_seconds):
                    break
                if any(protected[j] for j in skipped_order):
                    break
                bridge_speed = speed(a, reconnect)
                cures = plausible(a, reconnect)
                attempts.append({
                    "skipped": list(skipped),
                    "reconnect": reconnect,
                    "bridge_dt_seconds": bridge_dt,
                    "bridge_speed": bridge_speed,
                    "cures": cures,
                })
                if cures:
                    key = (min(a, reconnect), skipped, max(a, reconnect))
                    blocks[key] = {
                        "anchor": a, "reconnect": reconnect, "skipped": skipped,
                        "bridge_speed": bridge_speed, "continuation_speed": np.nan,
                        "continuation_row": None, "dependency": None,
                        "tested_reconnections": attempts,
                    }
                    links[a] = (reconnect, key); retained.add(reconnect)
                    i += count + 1; recovered = True
                    break
        if not recovered:
            trusted = False; i += 1
    for block in blocks.values():
        if block["reconnect"] in links:
            end, dependency = links[block["reconnect"]]
            block.update(continuation_speed=speed(block["reconnect"], end),
                         continuation_row=end, dependency=dependency)
    return blocks, retained


def _recover(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, order: list[int],
    config: NativePositionConfig, protected: np.ndarray,
) -> tuple[set[tuple], dict, dict]:
    @lru_cache(None)
    def ordered_values(a: int, b: int) -> tuple[float, float, float]:
        return _edge_arrays(time, lon, lat, a, b)

    def values(a: int, b: int) -> tuple[float, float, float]:
        if time[a] > time[b]:
            a, b = b, a
        return ordered_values(a, b)

    def speed(a: int, b: int) -> float:
        return values(a, b)[2]

    def elapsed(a: int, b: int) -> float:
        return values(a, b)[0]

    def plausible(a: int, b: int) -> bool:
        dt, _, value = values(a, b)
        return bool(config.minimum_local_dt_seconds <= dt <= config.max_local_gap_seconds + config.gap_tolerance_seconds
                    and np.isfinite(value) and value <= config.speed_threshold_m_s)

    def anomalous(a: int, b: int) -> bool:
        dt, _, value = values(a, b)
        return bool(config.minimum_local_dt_seconds <= dt <= config.max_local_gap_seconds + config.gap_tolerance_seconds
                    and np.isfinite(value) and value > config.speed_threshold_m_s)

    maximum_elapsed = config.max_local_gap_seconds + config.gap_tolerance_seconds
    forward, fretained = _walk(
        order, plausible, speed, anomalous, elapsed, protected, maximum_elapsed,
        config.max_automatic_removal_points,
    )
    backward, bretained = _walk(
        order[::-1], plausible, speed, anomalous, elapsed, protected, maximum_elapsed,
        config.max_automatic_removal_points,
    )

    order_position = {row: position for position, row in enumerate(order)}

    def enrich(blocks: dict, direction: str) -> None:
        for block in blocks.values():
            anchor, reconnect = int(block["anchor"]), int(block["reconnect"])
            bridge_dt, bridge_distance, bridge_speed = values(anchor, reconnect)
            left = min(order_position[anchor], order_position[reconnect])
            right = max(order_position[anchor], order_position[reconnect])
            lower = max(0, left - config.local_speed_window_points)
            upper = min(len(order) - 1, right + config.local_speed_window_points)
            skipped = set(block["skipped"])
            samples = [
                speed(order[position], order[position + 1])
                for position in range(lower, upper)
                if not ({order[position], order[position + 1]} & skipped)
                and plausible(order[position], order[position + 1])
            ]
            finite = np.asarray([value for value in samples if np.isfinite(value)], dtype=float)
            median = float(np.median(finite)) if len(finite) else np.nan
            mad = float(np.median(np.abs(finite - median))) if len(finite) else np.nan
            scale = (
                max(1.4826 * mad, config.local_speed_scale_floor_m_s)
                if np.isfinite(mad) else np.nan
            )
            score = (
                abs(bridge_speed - median) / scale
                if np.isfinite([bridge_speed, median, scale]).all() and scale > 0 else np.nan
            )
            block.update(
                method="time_bounded_excursion_cure",
                direction=direction,
                bridge_dt_seconds=bridge_dt,
                bridge_distance_m=bridge_distance,
                bridge_speed=bridge_speed,
                skipped_count=len(block["skipped"]),
                maximum_removal_points=config.max_automatic_removal_points,
                search_limit_seconds=maximum_elapsed,
                baseline={
                    "sample_count": int(len(finite)),
                    "speed_median_m_s": median,
                    "speed_mad_m_s": mad,
                    "speed_scale_m_s": scale,
                },
                smoothness_score=score,
                bridge_atypical_warning=bool(
                    np.isfinite(score) and score > config.bridge_speed_warning_z
                ),
            )

    enrich(forward, "forward")
    enrich(backward, "backward")
    high = set(forward) & set(backward)
    high = {key for key in high if np.isfinite(forward[key]["continuation_speed"])
            and np.isfinite(backward[key]["continuation_speed"])}
    while True:
        confirmed = {key for key in high if all(
            block["dependency"] is None or block["dependency"] in high
            for block in (forward[key], backward[key])
        )}
        if confirmed == high:
            break
        high = confirmed
    # Overlapping proposals are order-sensitive and remain unresolved.
    overlaps: set[tuple] = set()
    keys = list(high)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if set(left[1]) & set(right[1]):
                overlaps.update((left, right))
    return high - overlaps, forward, backward


def _interpolation_residual_m(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, a: int, b: int, c: int,
) -> float:
    """Spherical distance from B to the time-interpolated A-to-C position."""
    ta, tb, tc = time[a], time[b], time[c]
    if (np.isnat(ta) or np.isnat(tb) or np.isnat(tc)
            or not ta < tb < tc or not np.isfinite([lon[a], lat[a], lon[b], lat[b], lon[c], lat[c]]).all()):
        return np.nan
    left = int(ta.astype(np.int64))
    fraction = (int(tb.astype(np.int64)) - left) / (int(tc.astype(np.int64)) - left)

    def unit(j: int) -> np.ndarray:
        longitude, latitude = math.radians(lon[j]), math.radians(lat[j])
        return np.asarray([
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ])

    actual = unit(b)
    expected = (1.0 - fraction) * unit(a) + fraction * unit(c)
    norm = float(np.linalg.norm(expected))
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.nan
    expected /= norm
    return float(EARTH_RADIUS_M * math.acos(float(np.clip(np.dot(actual, expected), -1.0, 1.0))))


def _endpoint_speed_cure_proposals(
    time: np.ndarray, lon: np.ndarray, lat: np.ndarray, order: list[int],
    config: NativePositionConfig, protected: np.ndarray,
) -> list[tuple[tuple[int, tuple[int, ...], int], dict[str, Any], dict[str, Any]]]:
    """Return uniquely curing endpoint proposals with reproducible speed evidence."""
    if len(order) < 7:
        return []

    @lru_cache(None)
    def values(a: int, b: int) -> tuple[float, float, float]:
        if time[a] > time[b]:
            a, b = b, a
        return _edge_arrays(time, lon, lat, a, b)

    def plausible(a: int, b: int) -> bool:
        dt, _distance_m, speed_m_s = values(a, b)
        return bool(
            config.minimum_local_dt_seconds <= dt
            <= config.max_local_gap_seconds + config.gap_tolerance_seconds
            and np.isfinite(speed_m_s) and speed_m_s <= config.speed_threshold_m_s
        )

    def anomalous(a: int, b: int) -> bool:
        dt, _distance_m, speed_m_s = values(a, b)
        return bool(
            config.minimum_local_dt_seconds <= dt
            <= config.max_local_gap_seconds + config.gap_tolerance_seconds
            and np.isfinite(speed_m_s) and speed_m_s > config.speed_threshold_m_s
        )

    proposals: list[tuple[tuple[int, tuple[int, ...], int], dict[str, Any], dict[str, Any]]] = []
    window = config.local_speed_window_points
    for edge_position in range(2, len(order) - 3):
        u, v = order[edge_position:edge_position + 2]
        if not anomalous(u, v):
            continue
        p, a = order[edge_position - 2:edge_position]
        d, e = order[edge_position + 2:edge_position + 4]
        if not all((plausible(p, a), plausible(a, u), plausible(v, d), plausible(d, e))):
            continue

        local_speed_samples: list[float] = []
        lower = max(0, edge_position - window)
        upper = min(len(order) - 1, edge_position + 1 + window)
        for edge in range(lower, upper):
            left, right = order[edge:edge + 2]
            if {left, right} & {u, v}:
                continue
            if not plausible(left, right):
                continue
            local_speed = values(left, right)[2]
            if np.isfinite(local_speed):
                local_speed_samples.append(float(local_speed))
        if len(local_speed_samples) < config.endpoint_speed_min_samples:
            continue

        speed_values = np.asarray(local_speed_samples, dtype=float)
        speed_median = float(np.median(speed_values))
        speed_mad = float(np.median(np.abs(speed_values - speed_median)))
        speed_scale = max(
            1.4826 * speed_mad,
            config.local_speed_scale_floor_m_s,
        )
        baseline = {
            "sample_count": len(local_speed_samples),
            "speed_median_m_s": speed_median,
            "speed_mad_m_s": speed_mad,
            "speed_scale_m_s": speed_scale,
        }

        def hypothesis(candidate: int, anchor: int, reconnect: int,
                       continuation: int) -> dict[str, Any]:
            residual = _interpolation_residual_m(
                time, lon, lat, anchor, candidate, reconnect,
            )
            bridge_dt, bridge_distance, bridge_speed = values(anchor, reconnect)
            smoothness_score = (
                abs(bridge_speed - speed_median) / speed_scale
                if np.isfinite(bridge_speed) else np.nan
            )
            cures = bool(
                not protected[candidate]
                and plausible(anchor, reconnect)
                and np.isfinite(smoothness_score)
            )
            return {
                "method": "endpoint_speed_cure",
                "candidate": candidate,
                "anchor": anchor,
                "reconnect": reconnect,
                "skipped": [candidate],
                "bridge_dt_seconds": bridge_dt,
                "bridge_distance_m": bridge_distance,
                "bridge_speed": bridge_speed,
                "continuation_row": continuation,
                "continuation_speed": values(reconnect, continuation)[2],
                "residual_m": residual,
                "smoothness_score": smoothness_score,
                "bridge_atypical_warning": bool(
                    np.isfinite(smoothness_score)
                    and smoothness_score > config.bridge_speed_warning_z
                ),
                "cures": cures,
                "selected": False,
                "baseline": baseline,
            }

        left = hypothesis(u, a, v, d)
        right = hypothesis(v, u, d, e)
        curing = [item for item in (left, right) if bool(item["cures"])]
        if not curing:
            continue
        if len(curing) == 1:
            chosen = curing[0]
        else:
            ordered_hypotheses = sorted(curing, key=lambda item: item["smoothness_score"])
            if (ordered_hypotheses[1]["smoothness_score"]
                    - ordered_hypotheses[0]["smoothness_score"]
                    < config.endpoint_speed_score_margin_z):
                continue
            chosen = ordered_hypotheses[0]
        competitor = right if chosen is left else left
        chosen["selected"] = True
        candidate = int(chosen["candidate"])
        key = (
            min(int(chosen["anchor"]), int(chosen["reconnect"])),
            (candidate,),
            max(int(chosen["anchor"]), int(chosen["reconnect"])),
        )
        proposals.append((key, chosen, competitor))

    candidate_counts: dict[int, int] = {}
    for key, _chosen, _competitor in proposals:
        candidate = int(key[1][0])
        candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
    return [
        proposal for proposal in proposals
        if candidate_counts[int(proposal[0][1][0])] == 1
    ]


def _candidate_runs(
    frame: pd.DataFrame,
    rejected: np.ndarray,
    *,
    statuses: tuple[str, ...] = ("retained",),
) -> list[list[int]]:
    runs: list[list[int]] = []
    time = frame.time_value.to_numpy(dtype="datetime64[ns]")
    human = frame.human_position_decision.to_numpy(dtype=object)
    selected = frame[frame.segment_status.isin(statuses)]
    for _, segment in selected.groupby("segment_id", sort=False):
        order = segment.index.to_numpy()
        order = order[np.argsort(time[order], kind="stable")]
        # Duplicate timestamps and human uncertainties are hard context barriers.
        surviving = np.asarray([
            not rejected[j] and human[j] != "reject" for j in order
        ], dtype=bool)
        duplicate = np.zeros(len(order), dtype=bool)
        duplicate[surviving] = pd.Series(
            time[order[surviving]],
        ).duplicated(keep=False).to_numpy()
        current: list[int] = []
        for position, j in enumerate(order):
            # Automatic repeat rejection is unconditional, including when an
            # older review row kept or marked the observation uncertain.
            if rejected[j] or human[j] == "reject":
                continue
            hard = duplicate[position] or human[j] == "uncertain"
            if hard:
                if current:
                    runs.append(current); current = []
                continue
            current.append(int(j))
        if current:
            runs.append(current)
    return runs


def _automatic_duplicate_rejections(
    frame: pd.DataFrame, rejected: np.ndarray, config: NativePositionConfig,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Keep the duplicate-time representative with the smoothest local motion."""
    result = np.zeros(len(frame), dtype=bool)
    evidence: dict[int, dict[str, Any]] = {}
    time = frame.time_value.to_numpy(dtype="datetime64[ns]")
    lon = frame.source_lon.to_numpy(dtype=float)
    lat = frame.source_lat.to_numpy(dtype=float)
    source = frame.source_obs_index.to_numpy(dtype=np.int64)
    human = frame.human_position_decision.fillna("").astype(str).to_numpy()
    eligible = (
        frame.segment_status.eq("retained").to_numpy()
        & frame.valid_timestamp.to_numpy(dtype=bool)
        & frame.source_position_valid.to_numpy(dtype=bool)
        & frame.drogue_eligible.to_numpy(dtype=bool)
        & frame.deployment_eligible.to_numpy(dtype=bool)
        & ~frame.deployment_uncertain.to_numpy(dtype=bool)
        & ~rejected
        & (human != "reject")
    )
    maximum_elapsed = config.max_local_gap_seconds + config.gap_tolerance_seconds

    for segment_id, segment in frame.loc[eligible].groupby("segment_id", sort=False):
        segment_order = segment.index.to_numpy(dtype=np.int64)
        segment_order = segment_order[np.argsort(time[segment_order], kind="stable")]
        if len(segment_order) < 2:
            continue
        values = pd.Series(time[segment_order])
        duplicate_positions = values.duplicated(keep=False).to_numpy()
        for stamp in pd.unique(values[duplicate_positions]):
            members = segment_order[values.eq(stamp).to_numpy()]
            if len(members) < 2:
                continue
            first_position = int(np.flatnonzero(np.isin(segment_order, members))[0])
            last_position = int(np.flatnonzero(np.isin(segment_order, members))[-1])
            previous = int(segment_order[first_position - 1]) if first_position else None
            following = (
                int(segment_order[last_position + 1])
                if last_position + 1 < len(segment_order) else None
            )

            def representative_score(candidate: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
                edge_values: list[dict[str, Any]] = []
                high_count = 0
                total_excess = 0.0
                for left, right in ((previous, candidate), (candidate, following)):
                    if left is None or right is None:
                        continue
                    dt, distance, speed = _edge_arrays(time, lon, lat, left, right)
                    usable = bool(
                        config.minimum_local_dt_seconds <= dt <= maximum_elapsed
                        and np.isfinite(speed)
                    )
                    excess = (
                        max(0.0, float(speed) - config.speed_threshold_m_s)
                        if usable else math.inf
                    )
                    high_count += int(not usable or excess > 0)
                    total_excess += excess
                    edge_values.append({
                        "left_source_obs_index": int(source[left]),
                        "right_source_obs_index": int(source[right]),
                        "dt_seconds": dt, "distance_m": distance,
                        "speed_m_s": speed, "usable": usable,
                    })
                acceleration = (
                    _vertex_acceleration(time, lon, lat, previous, candidate, following)
                    if previous is not None and following is not None else np.nan
                )
                acceleration_score = acceleration if np.isfinite(acceleration) else math.inf
                # An explicit keep is the preferred representative; immutable
                # source identity makes the remaining tie deterministic.
                score = (
                    0 if human[candidate] == "keep" else 1,
                    high_count, total_excess, acceleration_score, int(source[candidate]),
                )
                return score, {
                    "candidate_source_obs_index": int(source[candidate]),
                    "high_speed_edge_count": high_count,
                    "total_speed_excess_m_s": total_excess,
                    "vector_acceleration_m_s2": acceleration,
                    "edges": edge_values,
                }

            scored = [(representative_score(int(j)), int(j)) for j in members]
            scored.sort(key=lambda item: item[0][0])
            representative = scored[0][1]
            rejected_members = [int(j) for j in members if int(j) != representative]
            if not rejected_members:
                continue
            event_id = _event_id(
                str(frame.platform_code.iloc[0]), "duplicate_timestamp",
                sorted(int(source[j]) for j in members),
            )
            detail = {
                "method": "automatic_duplicate_time_resolution",
                "segment_id": str(segment_id),
                "timestamp": str(pd.Timestamp(stamp)),
                "representative_source_obs_index": int(source[representative]),
                "rejected_source_obs_indices": [int(source[j]) for j in rejected_members],
                "hypotheses": [item[0][1] for item in scored],
            }
            for j in rejected_members:
                result[j] = True
                evidence[j] = {"event_id": event_id, "detail": detail}
    return result, evidence


def _aggressive_speed_cures(
    frame: pd.DataFrame, base_rejected: np.ndarray, config: NativePositionConfig,
    protected: np.ndarray, *, first_iteration: int,
) -> tuple[np.ndarray, dict[int, dict[str, Any]], int]:
    """Apply deterministic bounded speed cures without changing review semantics."""
    n = len(frame)
    result = np.zeros(n, dtype=bool)
    evidence: dict[int, dict[str, Any]] = {}
    time = frame.time_value.to_numpy(dtype="datetime64[ns]")
    lon = frame.source_lon.to_numpy(dtype=float)
    lat = frame.source_lat.to_numpy(dtype=float)
    source = frame.source_obs_index.to_numpy(dtype=np.int64)
    maximum_elapsed = config.max_local_gap_seconds + config.gap_tolerance_seconds
    iteration = first_iteration

    @lru_cache(None)
    def values(a: int, b: int) -> tuple[float, float, float]:
        if time[a] > time[b]:
            a, b = b, a
        return _edge_arrays(time, lon, lat, a, b)

    def usable_high(a: int, b: int) -> bool:
        dt, _distance, speed = values(a, b)
        return bool(
            config.minimum_local_dt_seconds <= dt <= maximum_elapsed
            and np.isfinite(speed) and speed > config.speed_threshold_m_s
        )

    while True:
        rejected = base_rejected | result
        proposals: dict[tuple[int, ...], dict[str, Any]] = {}
        for order in _candidate_runs(frame, rejected):
            if len(order) < 3:
                continue
            local_speeds = np.asarray([
                values(order[k], order[k + 1])[2]
                for k in range(len(order) - 1)
                if (config.minimum_local_dt_seconds
                    <= values(order[k], order[k + 1])[0] <= maximum_elapsed)
                and np.isfinite(values(order[k], order[k + 1])[2])
                and values(order[k], order[k + 1])[2] <= config.speed_threshold_m_s
            ], dtype=float)
            local_median = float(np.median(local_speeds)) if len(local_speeds) else np.nan
            local_mad = (
                float(np.median(np.abs(local_speeds - local_median)))
                if len(local_speeds) else np.nan
            )
            local_scale = (
                max(1.4826 * local_mad, config.local_speed_scale_floor_m_s)
                if np.isfinite(local_mad) else np.nan
            )
            for edge_position in range(len(order) - 1):
                u, v = order[edge_position:edge_position + 2]
                if not usable_high(u, v):
                    continue
                attempts: list[dict[str, Any]] = []
                curing: list[dict[str, Any]] = []
                for direction in ("left", "right"):
                    for count in range(1, config.max_automatic_removal_points + 1):
                        if direction == "right":
                            block_start, block_end = edge_position + 1, edge_position + 1 + count
                            anchor_position, reconnect_position = edge_position, block_end
                        else:
                            block_start, block_end = edge_position - count + 1, edge_position + 1
                            anchor_position, reconnect_position = edge_position - count, edge_position + 1
                        if (block_start < 0 or block_end > len(order)
                                or anchor_position < 0 or reconnect_position >= len(order)):
                            break
                        block = tuple(int(j) for j in order[block_start:block_end])
                        if any(protected[j] for j in block):
                            break
                        anchor = int(order[anchor_position])
                        reconnect = int(order[reconnect_position])
                        bridge_dt, bridge_distance, bridge_speed = values(anchor, reconnect)
                        within_time = bool(
                            np.isfinite(bridge_dt) and bridge_dt > 0
                            and bridge_dt <= maximum_elapsed
                        )
                        if not within_time:
                            break
                        cures = bool(
                            bridge_dt >= config.minimum_local_dt_seconds
                            and np.isfinite(bridge_speed)
                            and bridge_speed <= config.speed_threshold_m_s
                        )
                        survivors = order[:block_start] + order[block_end:]
                        bridge_position = survivors.index(anchor)
                        boundary_accelerations: list[float] = []
                        if bridge_position > 0:
                            boundary_accelerations.append(_vertex_acceleration(
                                time, lon, lat, survivors[bridge_position - 1], anchor, reconnect,
                            ))
                        if bridge_position + 2 < len(survivors):
                            boundary_accelerations.append(_vertex_acceleration(
                                time, lon, lat, anchor, reconnect, survivors[bridge_position + 2],
                            ))
                        finite_accelerations = [
                            float(value) for value in boundary_accelerations if np.isfinite(value)
                        ]
                        maximum_acceleration = (
                            max(finite_accelerations) if finite_accelerations else math.inf
                        )
                        smoothness = (
                            abs(float(bridge_speed) - local_median) / local_scale
                            if np.isfinite([bridge_speed, local_median, local_scale]).all()
                            and local_scale > 0 else math.inf
                        )
                        attempt = {
                            "method": "bounded_iterative_speed_cure",
                            "direction": direction,
                            "anchor": anchor, "reconnect": reconnect,
                            "anchor_source_obs_index": int(source[anchor]),
                            "reconnect_source_obs_index": int(source[reconnect]),
                            "skipped": list(block),
                            "skipped_source_obs_indices": [int(source[j]) for j in block],
                            "skipped_count": len(block),
                            "bridge_dt_seconds": bridge_dt,
                            "bridge_distance_m": bridge_distance,
                            "bridge_speed": bridge_speed,
                            "speed_threshold_m_s": config.speed_threshold_m_s,
                            "maximum_boundary_acceleration_m_s2": (
                                maximum_acceleration if np.isfinite(maximum_acceleration) else np.nan
                            ),
                            "boundary_accelerations_m_s2": boundary_accelerations,
                            "local_speed_median_m_s": local_median,
                            "local_speed_mad_m_s": local_mad,
                            "local_speed_scale_m_s": local_scale,
                            "smoothness_score": smoothness if np.isfinite(smoothness) else np.nan,
                            "maximum_removal_points": config.max_automatic_removal_points,
                            "search_limit_seconds": maximum_elapsed,
                            "cures": cures,
                        }
                        attempts.append(attempt)
                        if cures:
                            curing.append(attempt)
                            # This direction contributes only its smallest cure.
                            break
                if not curing:
                    continue
                for item in curing:
                    first_time = int(time[item["skipped"][0]].astype(np.int64))
                    item["rank"] = (
                        int(item["skipped_count"]),
                        float(item["maximum_boundary_acceleration_m_s2"])
                        if np.isfinite(item["maximum_boundary_acceleration_m_s2"]) else math.inf,
                        float(item["smoothness_score"])
                        if np.isfinite(item["smoothness_score"]) else math.inf,
                        -first_time,
                        tuple(item["skipped_source_obs_indices"]),
                    )
                item = min(curing, key=lambda value: value["rank"])
                item["tested_hypotheses"] = [
                    {key: value for key, value in attempt.items() if key != "rank"}
                    for attempt in attempts
                ]
                block_key = tuple(sorted(int(j) for j in item["skipped"]))
                if block_key not in proposals or item["rank"] < proposals[block_key]["rank"]:
                    proposals[block_key] = item

        if not proposals:
            break
        accepted: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        occupied: set[int] = set()
        for block, proposal in sorted(
            proposals.items(), key=lambda item: (item[1]["rank"], item[0]),
        ):
            if occupied.intersection(block):
                continue
            accepted.append((block, proposal))
            occupied.update(block)
        if not accepted:
            break
        iteration += 1
        for block, proposal in accepted:
            event_id = _event_id(
                str(frame.platform_code.iloc[0]), "automatic_speed_cure",
                [int(source[j]) for j in block],
            )
            reason = (
                "automatic_iterative_endpoint_speed_cure" if len(block) == 1
                else "automatic_iterative_excursion_speed_cure"
            )
            competitor = {
                "method": "competing_speed_cure_hypotheses",
                "tested_hypotheses": proposal.get("tested_hypotheses", []),
                "selected_skipped_source_obs_indices": proposal["skipped_source_obs_indices"],
            }
            for j in block:
                result[j] = True
                evidence[j] = {
                    "event_id": event_id, "iteration": iteration,
                    "implicated": [int(source[value]) for value in block],
                    "forward": proposal, "backward": competitor, "reason": reason,
                }
    return result, evidence, iteration


def _repeat_rejections(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    frame["exact_repeat_flag"] = False
    frame["repeated_coordinate_count"] = 0
    frame["repeated_coordinate_group_id"] = ""
    frame["repeat_duration_seconds"] = np.nan
    eligible = (
        frame.segment_status.eq("retained") & frame.source_position_valid
        & frame.valid_timestamp & frame.drogue_eligible & frame.deployment_eligible
    )
    subset = frame.loc[eligible, ["source_lon", "source_lat"]]
    repeated = subset[subset.duplicated(["source_lon", "source_lat"], keep=False)]
    for (lon, lat), group in repeated.groupby(["source_lon", "source_lat"], sort=False):
        indices = group.index.to_numpy()
        group_id = _repeat_id(str(frame.platform_code.iloc[0]), lon, lat)
        times = frame.loc[indices, "time_value"].to_numpy(dtype="datetime64[ns]")
        duration = _seconds(times.min(), times.max()) if len(times) else np.nan
        frame.loc[indices, "exact_repeat_flag"] = True
        frame.loc[indices, "repeated_coordinate_count"] = len(indices)
        frame.loc[indices, "repeated_coordinate_group_id"] = group_id
        frame.loc[indices, "repeat_duration_seconds"] = duration
        result[indices] = True
    return result


def _short_interval_rejections(
    frame: pd.DataFrame, repeat_reject: np.ndarray, config: NativePositionConfig,
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:
    """Iteratively thin positive intervals below the configured local minimum."""
    result = np.zeros(len(frame), dtype=bool)
    evidence: dict[int, dict[str, Any]] = {}
    time = frame.time_value.to_numpy(dtype="datetime64[ns]")
    source = frame.source_obs_index.to_numpy(dtype=np.int64)
    order = np.argsort(time, kind="stable")
    eligible = (
        frame.segment_status.eq("retained").to_numpy()
        & frame.valid_timestamp.to_numpy(dtype=bool)
        & frame.source_position_valid.to_numpy(dtype=bool)
        & frame.drogue_eligible.to_numpy(dtype=bool)
        & frame.deployment_eligible.to_numpy(dtype=bool)
        & ~frame.deployment_uncertain.to_numpy(dtype=bool)
    )
    duplicate = np.zeros(len(frame), dtype=bool)
    eligible_indices = np.flatnonzero(eligible)
    duplicate[eligible_indices] = pd.Series(
        time[eligible_indices],
    ).duplicated(keep=False).to_numpy()
    segment = frame.segment_id.fillna("").astype(str).to_numpy()
    human = frame.human_position_decision.fillna("").astype(str).to_numpy()
    anchor: int | None = None
    anchor_segment = ""
    iteration = 0
    for raw_index in order:
        j = int(raw_index)
        if not eligible[j] or duplicate[j]:
            anchor, anchor_segment = None, ""
            continue
        current_segment = segment[j]
        if not current_segment:
            anchor, anchor_segment = None, ""
            continue
        if repeat_reject[j]:
            continue
        if anchor is None or current_segment != anchor_segment:
            if human[j] not in {"reject", "uncertain"}:
                anchor, anchor_segment = j, current_segment
            else:
                anchor, anchor_segment = None, ""
            continue
        dt = _seconds(time[anchor], time[j])
        iteration += 1
        if 0 < dt < config.minimum_local_dt_seconds:
            result[j] = True
            event_id = _event_id(
                str(frame.platform_code.iloc[0]), "short_interval", [int(source[j])],
            )
            item = {
                "method": "iterative_short_interval_thinning",
                "predecessor_row": int(anchor),
                "candidate_row": j,
                "predecessor_source_obs_index": int(source[anchor]),
                "candidate_source_obs_index": int(source[j]),
                "dt_seconds": float(dt),
                "minimum_local_dt_seconds": float(config.minimum_local_dt_seconds),
                "iteration": iteration,
            }
            evidence[j] = {"event_id": event_id, "detail": item}
            continue
        if human[j] == "reject":
            continue
        if human[j] == "uncertain":
            anchor, anchor_segment = None, ""
            continue
        anchor, anchor_segment = j, current_segment
    return result, evidence


def _local_qc(frame: pd.DataFrame, position: NativePositionConfig,
              review: PositionReviewConfig, *, resolution_policy: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    repeat_reject = _repeat_rejections(frame)
    short_reject, short_evidence = _short_interval_rejections(
        frame, repeat_reject, position,
    )
    duplicate_reject = np.zeros(n, dtype=bool)
    duplicate_evidence: dict[int, dict[str, Any]] = {}
    if resolution_policy == "aggressive":
        duplicate_reject, duplicate_evidence = _automatic_duplicate_rejections(
            frame, repeat_reject | short_reject, position,
        )
    geometry_reject = np.zeros(n, dtype=bool)
    evidence: dict[int, dict[str, Any]] = {}
    protected = frame.human_position_decision.isin(["keep", "uncertain"]).to_numpy()
    time = frame.time_value.to_numpy(dtype="datetime64[ns]")
    lon = frame.source_lon.to_numpy(dtype=float)
    lat = frame.source_lat.to_numpy(dtype=float)
    iteration = 0
    while True:
        iteration += 1
        rejected = repeat_reject | short_reject | duplicate_reject | geometry_reject
        runs = _candidate_runs(frame, rejected)
        proposed: list[tuple[tuple, dict, dict, str]] = []
        for order in runs:
            if len(order) < 4:
                continue
            high, forward, backward = _recover(time, lon, lat, order, position, protected)
            proposed.extend((
                key, forward[key], backward[key],
                "bidirectionally_confirmed_time_bounded_excursion_cure",
            ) for key in high)
        new_points = {
            j for key, _forward, _backward, _reason in proposed
            for j in key[1] if not geometry_reject[j]
        }
        if not new_points:
            for order in runs:
                proposed.extend((
                    key, chosen, competitor,
                    "unique_endpoint_speed_cure",
                ) for key, chosen, competitor in _endpoint_speed_cure_proposals(
                    time, lon, lat, order, position, protected,
                ))
            new_points = {
                j for key, _chosen, _competitor, _reason in proposed
                for j in key[1] if not geometry_reject[j]
            }
        if not new_points:
            break
        for key, forward, backward, reason in proposed:
            if not any(j in new_points for j in key[1]):
                continue
            event = _event_id(str(frame.platform_code.iloc[0]), "local_spike_or_short_block",
                              frame.source_obs_index.to_numpy()[list(key[1])].tolist())
            for j in key[1]:
                evidence[j] = {
                    "event_id": event, "iteration": iteration,
                    "implicated": frame.source_obs_index.to_numpy()[list(key[1])].tolist(),
                    "forward": forward, "backward": backward, "reason": reason,
                }
        geometry_reject[list(new_points)] = True

    if resolution_policy == "aggressive":
        aggressive_reject, aggressive_evidence, iteration = _aggressive_speed_cures(
            frame,
            repeat_reject | short_reject | duplicate_reject | geometry_reject,
            position, protected, first_iteration=iteration,
        )
        geometry_reject |= aggressive_reject
        evidence.update(aggressive_evidence)

    # Final surviving edge evidence.
    columns: dict[str, Any] = {
        "surviving_predecessor_source_obs_index": np.nan,
        "surviving_dt_seconds": np.nan, "surviving_distance_m": np.nan,
        "surviving_speed_m_s": np.nan, "nonpositive_dt_flag": False,
        "short_dt_flag": False, "nominal_dt_flag": False, "large_gap_flag": False,
        "high_speed_flag": False, "local_event_id": "", "local_event_type": "",
        "local_event_implicated_source_obs_indices": "", "point_auto_status": "",
        "point_auto_decision": "", "point_auto_reason": "", "auto_iteration": 0,
        "local_review_required": False, "local_review_reason": "",
        "candidate_bridge_anchor_source_obs_index": np.nan,
        "candidate_bridge_reconnection_source_obs_index": np.nan,
        "candidate_bridge_speed_m_s": np.nan, "forward_result_json": "",
        "backward_result_json": "",
    }
    for name, default in columns.items():
        frame[name] = default

    for group_id, group in frame[frame.exact_repeat_flag].groupby("repeated_coordinate_group_id"):
        indices = group.index.to_numpy()
        implicated = frame.loc[indices, "source_obs_index"].astype(int).tolist()
        frame.loc[indices, "local_event_id"] = group_id
        frame.loc[indices, "local_event_type"] = "repeated_position"
        frame.loc[indices, "local_event_implicated_source_obs_indices"] = json.dumps(implicated)
        frame.loc[indices, "point_auto_status"] = "repeated_position"
        frame.loc[indices, "point_auto_decision"] = "reject"
        frame.loc[indices, "point_auto_reason"] = "exact_repeated_coordinates"

    for j, item in short_evidence.items():
        detail = item["detail"]
        frame.at[j, "local_event_id"] = item["event_id"]
        frame.at[j, "local_event_type"] = "short_interval"
        frame.at[j, "local_event_implicated_source_obs_indices"] = json.dumps([
            detail["predecessor_source_obs_index"], detail["candidate_source_obs_index"],
        ])
        frame.at[j, "point_auto_status"] = "short_interval_reject"
        frame.at[j, "point_auto_decision"] = "reject"
        frame.at[j, "point_auto_reason"] = "positive_dt_below_minimum_local_interval"
        frame.at[j, "auto_iteration"] = detail["iteration"]
        frame.at[j, "candidate_bridge_anchor_source_obs_index"] = detail[
            "predecessor_source_obs_index"
        ]
        frame.at[j, "forward_result_json"] = json.dumps(detail, sort_keys=True)

    for j, item in duplicate_evidence.items():
        detail = item["detail"]
        frame.at[j, "local_event_id"] = item["event_id"]
        frame.at[j, "local_event_type"] = "duplicate_timestamp"
        frame.at[j, "local_event_implicated_source_obs_indices"] = json.dumps(
            detail["rejected_source_obs_indices"]
        )
        frame.at[j, "point_auto_status"] = "duplicate_timestamp_reject"
        frame.at[j, "point_auto_decision"] = "reject"
        frame.at[j, "point_auto_reason"] = "automatic_duplicate_time_resolution"
        frame.at[j, "forward_result_json"] = json.dumps(detail, default=float, sort_keys=True)

    for j, item in evidence.items():
        forward, backward = item["forward"], item["backward"]
        frame.at[j, "local_event_id"] = item["event_id"]
        frame.at[j, "local_event_type"] = "local_spike_or_short_block"
        frame.at[j, "local_event_implicated_source_obs_indices"] = json.dumps(item["implicated"])
        frame.at[j, "point_auto_status"] = "high_confidence_reject"
        frame.at[j, "point_auto_decision"] = "reject"
        frame.at[j, "point_auto_reason"] = item["reason"]
        frame.at[j, "auto_iteration"] = item["iteration"]
        frame.at[j, "candidate_bridge_anchor_source_obs_index"] = int(frame.at[forward["anchor"], "source_obs_index"])
        frame.at[j, "candidate_bridge_reconnection_source_obs_index"] = int(frame.at[forward["reconnect"], "source_obs_index"])
        frame.at[j, "candidate_bridge_speed_m_s"] = forward["bridge_speed"]
        frame.at[j, "forward_result_json"] = json.dumps(forward, default=float, sort_keys=True)
        frame.at[j, "backward_result_json"] = json.dumps(backward, default=float, sort_keys=True)

    rejected = repeat_reject | short_reject | duplicate_reject | geometry_reject
    high_edges: list[dict[str, Any]] = []
    predecessor = np.full(n, np.nan)
    edge_dt = np.full(n, np.nan)
    edge_distance = np.full(n, np.nan)
    edge_speed = np.full(n, np.nan)
    nonpositive = np.zeros(n, dtype=bool)
    short = np.zeros(n, dtype=bool)
    nominal = np.zeros(n, dtype=bool)
    large = np.zeros(n, dtype=bool)
    fast = np.zeros(n, dtype=bool)
    source_index = frame.source_obs_index.to_numpy(dtype=np.int64)
    for run_number, order in enumerate(
        _candidate_runs(frame, rejected, statuses=("retained", "pending"))
    ):
        if len(order) < 2:
            continue
        a = np.asarray(order[:-1], dtype=np.int64)
        b = np.asarray(order[1:], dtype=np.int64)
        dt = (time[b].astype(np.int64) - time[a].astype(np.int64)) / 1e9
        phi1, phi2 = np.radians(lat[a]), np.radians(lat[b])
        dphi = phi2 - phi1
        dlon = np.radians(lon[b] - lon[a])
        hav = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
        distance = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        speed = np.divide(distance, dt, out=np.full(len(dt), np.nan), where=dt > 0)
        predecessor[b] = source_index[a]
        edge_dt[b], edge_distance[b], edge_speed[b] = dt, distance, speed
        nonpositive[b] = np.isfinite(dt) & (dt <= 0)
        short[b] = np.isfinite(dt) & (dt > 0) & (dt < position.minimum_local_dt_seconds)
        nominal[b] = np.isfinite(dt) & (
            np.abs(dt - position.nominal_interval_seconds) <= position.nominal_interval_tolerance_seconds
        )
        large[b] = np.isfinite(dt) & (
            dt > position.max_local_gap_seconds + position.gap_tolerance_seconds
        )
        fast[b] = np.isfinite(speed) & (speed > position.speed_threshold_m_s)
        for edge_position in np.flatnonzero(fast[b]):
            aa, bb = int(a[edge_position]), int(b[edge_position])
            usable = (
                position.minimum_local_dt_seconds <= dt[edge_position]
                <= position.max_local_gap_seconds + position.gap_tolerance_seconds
            )
            high_edges.append({
                "run": run_number, "position": int(edge_position) + 1,
                "a": aa, "b": bb, "dt": float(dt[edge_position]),
                "distance": float(distance[edge_position]),
                "speed": float(speed[edge_position]), "usable": bool(usable),
            })
    frame["surviving_predecessor_source_obs_index"] = predecessor
    frame["surviving_dt_seconds"] = edge_dt
    frame["surviving_distance_m"] = edge_distance
    frame["surviving_speed_m_s"] = edge_speed
    frame["nonpositive_dt_flag"] = nonpositive
    frame["short_dt_flag"] = short | short_reject
    frame["nominal_dt_flag"] = nominal
    frame["large_gap_flag"] = large
    frame["high_speed_flag"] = fast

    unresolved = np.zeros(n, dtype=bool)
    # Conservative modes review duplicate timestamps. Aggressive automatic mode
    # has already selected one deterministic representative above.
    eligible = (
        frame.segment_status.isin(["retained", "pending"])
        & frame.valid_timestamp & frame.source_position_valid
    )
    duplicate_rows = frame.loc[eligible]
    duplicate_rows = duplicate_rows[duplicate_rows.time_value.duplicated(keep=False)]
    for stamp, group in duplicate_rows.groupby("time_value", sort=False):
        indices = group.index.to_numpy()
        frame.loc[indices, "nonpositive_dt_flag"] = True
        if resolution_policy == "aggressive":
            representative = indices[~duplicate_reject[indices]]
            free_representative = representative[
                frame.loc[representative, "point_auto_status"].eq("").to_numpy()
            ]
            frame.loc[free_representative, "point_auto_status"] = "duplicate_timestamp_representative"
            frame.loc[free_representative, "point_auto_decision"] = "keep"
            frame.loc[free_representative, "point_auto_reason"] = "automatic_duplicate_time_resolution"
            continue
        implicated = frame.loc[indices, "source_obs_index"].astype(int).tolist()
        event = _event_id(str(frame.platform_code.iloc[0]), "boundary_or_insufficient_context", implicated)
        free = indices[frame.loc[indices, "local_event_id"].eq("").to_numpy()]
        frame.loc[free, "local_event_id"] = event
        frame.loc[free, "local_event_type"] = "boundary_or_insufficient_context"
        frame.loc[free, "local_event_implicated_source_obs_indices"] = json.dumps(implicated)
        frame.loc[free, "point_auto_status"] = "review_required"
        frame.loc[free, "point_auto_reason"] = "duplicate_or_nonpositive_timestamp"
        frame.loc[indices, "local_review_required"] = True
        frame.loc[indices, "local_review_reason"] = "duplicate_or_nonpositive_timestamp"
        undecided = ~frame.loc[indices, "human_position_decision"].isin(["keep", "reject"]).to_numpy()
        unresolved[indices[undecided]] = True

    # Group remaining anomalous edges, including nearby runs separated by a few plausible edges.
    high_edges.sort(key=lambda item: (item["run"], item["position"]))
    groups: list[list[dict[str, Any]]] = []
    for item in high_edges:
        if (groups and groups[-1][-1]["run"] == item["run"]
                and item["position"] - groups[-1][-1]["position"] <= review.merge_gap_edges + 1):
            groups[-1].append(item)
        else:
            groups.append([item])
    for group in groups:
        implicated_rows = sorted({item[key] for item in group for key in ("a", "b")})
        implicated = frame.loc[implicated_rows, "source_obs_index"].astype(int).tolist()
        implicated_frame = frame.loc[implicated_rows]
        accepted_endpoint_cure = bool((
            implicated_frame.human_position_decision.eq("keep")
            & implicated_frame.human_auto_reason_snapshot.isin([
                "unique_endpoint_speed_cure",
            ])
        ).any())
        if accepted_endpoint_cure:
            # K answers the explicit endpoint hypothesis: the proposed point is
            # real motion.  Retain the raw high-speed diagnostic, but do not turn
            # the same edge back into a new ambiguous event on its other endpoint.
            continue
        if resolution_policy == "aggressive":
            event_type = "automatic_retained_anomaly"
            reason = "automatic_keep_removal_limit_reached"
        elif any(not item["usable"] for item in group):
            event_type = "boundary_or_insufficient_context"
            reason = "high_speed_edge_has_untrusted_timing"
        elif len(group) == 1:
            event_type = "single_edge_ambiguous"
            reason = "single_high_speed_edge_has_no_unique_bad_endpoint"
        else:
            event_type = "persistent_excursion"
            reason = "no_bidirectionally_confirmed_cure_within_timing_limit"
        event = _event_id(str(frame.platform_code.iloc[0]), event_type, implicated)
        edge_json = json.dumps([{k: value for k, value in item.items() if k != "run"} for item in group])
        for j in implicated_rows:
            if resolution_policy == "aggressive":
                if not frame.at[j, "local_event_id"]:
                    frame.at[j, "local_event_id"] = event
                    frame.at[j, "local_event_type"] = event_type
                    frame.at[j, "local_event_implicated_source_obs_indices"] = json.dumps(implicated)
                    frame.at[j, "point_auto_status"] = "automatic_retained_anomaly"
                    frame.at[j, "point_auto_decision"] = "keep"
                    frame.at[j, "point_auto_reason"] = reason
                    frame.at[j, "forward_result_json"] = edge_json
                continue
            # Human decisions resolve their own point, but undecided implicated points remain blocked.
            if not frame.at[j, "human_position_decision"]:
                unresolved[j] = True
            frame.at[j, "local_review_required"] = True
            frame.at[j, "local_review_reason"] = reason
            if not frame.at[j, "local_event_id"]:
                frame.at[j, "local_event_id"] = event
                frame.at[j, "local_event_type"] = event_type
                frame.at[j, "local_event_implicated_source_obs_indices"] = json.dumps(implicated)
                frame.at[j, "point_auto_status"] = "review_required"
                frame.at[j, "point_auto_reason"] = reason
                frame.at[j, "forward_result_json"] = edge_json
    return repeat_reject | short_reject | duplicate_reject | geometry_reject, unresolved


def _resolve(frame: pd.DataFrame, drogue: ResolvedDrogueDecision,
             auto_reject: np.ndarray, unresolved_event: np.ndarray) -> PositionResolution:
    n = len(frame)
    status = np.full(n, "valid", dtype=object)
    source = np.full(n, "ordinary", dtype=object)
    invalid = ~frame.valid_timestamp.to_numpy() | ~frame.source_position_valid.to_numpy()
    status[invalid], source[invalid] = "source_invalid", "source"
    valid_source = ~invalid
    if drogue.final_drogue_status == "uncertain":
        target = valid_source
        status[target], source[target] = "unresolved", "drogue"
    else:
        outside_drogue = valid_source & ~frame.drogue_eligible.to_numpy()
        status[outside_drogue], source[outside_drogue] = "outside_drogue_window", "drogue"
        active = valid_source & frame.drogue_eligible.to_numpy()
        outside_operational = active & (
            ~frame.deployment_eligible.to_numpy() & ~frame.deployment_uncertain.to_numpy()
        )
        outside_operational |= active & frame.segment_status.eq("excluded").to_numpy()
        status[outside_operational], source[outside_operational] = "outside_operational_segment", "temporal"
        temporal_uncertain = active & frame.deployment_uncertain.to_numpy()
        temporal_uncertain |= active & frame.segment_status.eq("pending").to_numpy()
        status[temporal_uncertain], source[temporal_uncertain] = "unresolved", "temporal"
        human_segment_uncertain = active & frame.segment_status.eq("uncertain").to_numpy()
        status[human_segment_uncertain], source[human_segment_uncertain] = "uncertain", "human"
        eligible = active & frame.deployment_eligible.to_numpy() & frame.segment_status.eq("retained").to_numpy()
        decisions = frame.human_position_decision.to_numpy()
        repeat = eligible & frame.exact_repeat_flag.to_numpy() & auto_reject
        status[repeat], source[repeat] = "rejected", "automatic_repeat"
        short_interval = (
            eligible & ~repeat & auto_reject
            & frame.point_auto_status.eq("short_interval_reject").to_numpy()
        )
        status[short_interval], source[short_interval] = "rejected", "automatic_short_interval"
        duplicate_timestamp = (
            eligible & ~repeat & ~short_interval & auto_reject
            & frame.point_auto_status.eq("duplicate_timestamp_reject").to_numpy()
        )
        status[duplicate_timestamp], source[duplicate_timestamp] = (
            "rejected", "automatic_duplicate_time"
        )
        for decision, value in (("reject", "rejected"), ("keep", "valid"), ("uncertain", "uncertain")):
            mask = eligible & ~repeat & ~short_interval & ~duplicate_timestamp & (decisions == decision)
            status[mask], source[mask] = value, "human"
        undecided = eligible & ~repeat & ~short_interval & ~duplicate_timestamp & (decisions == "")
        geometry = undecided & ~repeat & auto_reject
        status[geometry], source[geometry] = "rejected", "automatic_geometry"
        unresolved = undecided & ~auto_reject & unresolved_event
        status[unresolved], source[unresolved] = "unresolved", "automatic_review_required"
    valid = status == "valid"
    complete = not np.isin(status, ["uncertain", "unresolved"]).any()
    return PositionResolution(status, valid, source, bool(complete), bool(complete and valid.any()))


def run_native_position_qc(
    trajectory: NativeTrajectory,
    drogue: ResolvedDrogueDecision,
    *,
    deployment: DeploymentBoundary | None = None,
    reviews: pd.DataFrame | None = None,
    temporal_config: TemporalSegmentConfig | None = None,
    position_config: NativePositionConfig | None = None,
    review_config: PositionReviewConfig | None = None,
    resolution_policy: str = "conservative",
) -> pd.DataFrame:
    """Return one QC row per immutable source observation."""
    deployment = deployment or DeploymentBoundary()
    temporal_config = temporal_config or TemporalSegmentConfig()
    position_config = position_config or NativePositionConfig()
    review_config = review_config or PositionReviewConfig()
    if resolution_policy not in {"conservative", "aggressive"}:
        raise ValueError(f"Unsupported position resolution policy: {resolution_policy!r}")
    point_reviews, segment_reviews = _review_maps(reviews, trajectory.platform_code)
    frame = _base_frame(trajectory, drogue, deployment, point_reviews)
    _segments(frame, trajectory.source_sha256, temporal_config, segment_reviews)
    if drogue.final_drogue_status == "uncertain":
        # Populate the stable local schema without making scientific decisions.
        for name, default in {
            "exact_repeat_flag": False, "repeated_coordinate_count": 0,
            "repeated_coordinate_group_id": "", "repeat_duration_seconds": np.nan,
            "surviving_predecessor_source_obs_index": np.nan, "surviving_dt_seconds": np.nan,
            "surviving_distance_m": np.nan, "surviving_speed_m_s": np.nan,
            "nonpositive_dt_flag": False, "short_dt_flag": False, "nominal_dt_flag": False,
            "large_gap_flag": False, "high_speed_flag": False, "local_event_id": "",
            "local_event_type": "", "local_event_implicated_source_obs_indices": "",
            "point_auto_status": "blocked_by_drogue", "point_auto_decision": "",
            "point_auto_reason": "resolved_drogue_status_uncertain", "auto_iteration": 0,
            "local_review_required": False, "local_review_reason": "",
            "candidate_bridge_anchor_source_obs_index": np.nan,
            "candidate_bridge_reconnection_source_obs_index": np.nan,
            "candidate_bridge_speed_m_s": np.nan, "forward_result_json": "",
            "backward_result_json": "",
        }.items():
            frame[name] = default
        auto_reject = np.zeros(len(frame), dtype=bool)
        unresolved = np.zeros(len(frame), dtype=bool)
    else:
        auto_reject, unresolved = _local_qc(
            frame, position_config, review_config, resolution_policy=resolution_policy,
        )
    resolution = _resolve(frame, drogue, auto_reject, unresolved)
    frame["final_position_status"] = resolution.final_position_status
    frame["final_position_valid"] = resolution.final_position_valid
    frame["position_decision_source"] = resolution.decision_source
    frame["platform_qc_complete"] = resolution.platform_qc_complete
    frame["reconstruction_available"] = resolution.reconstruction_available
    return frame.drop(columns=["time_value"])


def resolved_position_trajectory(
    trajectory: NativeTrajectory, qc: pd.DataFrame, *, allow_unresolved: bool = False,
) -> tuple[NativeTrajectory, PositionResolution]:
    """Validate a QC join and expose the immutable source trajectory plus masks."""
    rows = qc[qc.platform_code.astype(str) == trajectory.platform_code]
    expected = np.asarray(trajectory.source_obs_index, dtype=np.int64)
    if (len(rows) != len(expected) or rows.source_obs_index.duplicated().any()
            or set(rows.source_obs_index.astype(int)) != set(map(int, expected))
            or set(rows.source_sha256.astype(str)) != {trajectory.source_sha256}):
        raise ValueError("Position QC does not match the native source identity")
    rows = rows.set_index("source_obs_index", drop=False).loc[expected]
    status = rows.final_position_status.astype(str).to_numpy()
    if not allow_unresolved and np.isin(status, ["uncertain", "unresolved"]).any():
        raise ValueError("Native trajectory contains unresolved position QC state")
    resolution = PositionResolution(
        status, rows.final_position_valid.to_numpy(dtype=bool),
        rows.position_decision_source.astype(str).to_numpy(),
        bool(rows.platform_qc_complete.all()), bool(rows.reconstruction_available.all()),
    )
    return trajectory, resolution
