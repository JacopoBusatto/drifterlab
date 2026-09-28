"""Persistence and interactive review for generic native-position QC."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from drifterlab.qc.native_position import NativePositionConfig, position_edge_metrics


REVIEW_COLUMNS = [
    "platform_code", "source_sha256", "target_type", "target_id",
    "source_obs_index", "event_id", "event_type", "decision",
    "decision_source", "auto_status_snapshot", "auto_decision_snapshot",
    "auto_reason_snapshot", "qc_config_sha256", "note", "review_timestamp",
]


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        return ""
    digest = sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PositionReviews:
    """One latest explicit decision per immutable observation or segment."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        if not self.path.exists():
            return
        table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if list(table.columns) != REVIEW_COLUMNS:
            raise ValueError(f"Position review CSV must have columns {REVIEW_COLUMNS}")
        for row in table.to_dict("records"):
            self._validate(row)
            key = row["platform_code"], row["target_type"], row["target_id"]
            if key in self.rows:
                raise ValueError(f"Duplicate position review identity: {key}")
            self.rows[key] = row

    @staticmethod
    def _validate(row: dict[str, Any]) -> None:
        target = str(row.get("target_type", ""))
        decision = str(row.get("decision", ""))
        if not row.get("platform_code") or not row.get("source_sha256") or not row.get("target_id"):
            raise ValueError("Position review requires platform, source hash, and target identity")
        if target == "observation":
            try:
                source_index = int(row.get("source_obs_index", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("Observation review requires source_obs_index") from exc
            if source_index < 0 or decision not in {"keep", "reject", "uncertain"}:
                raise ValueError("Invalid observation review decision")
        elif target == "segment":
            if row.get("source_obs_index", "") not in {"", None}:
                raise ValueError("Segment reviews must not contain source_obs_index")
            if decision not in {"retain", "exclude", "uncertain"}:
                raise ValueError("Invalid segment review decision")
        else:
            raise ValueError("target_type must be observation or segment")
        if row.get("decision_source") not in {"manual", "accepted_auto"}:
            raise ValueError("decision_source must be manual or accepted_auto")
        stamp = pd.Timestamp(row.get("review_timestamp"))
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("review_timestamp must be explicit UTC")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([self.rows[key] for key in sorted(self.rows)], columns=REVIEW_COLUMNS)

    def save(self) -> None:
        if file_sha256(self.path) != self.sha256:
            raise ValueError("Position review CSV changed in another session")
        _atomic(self.path, lambda path: self.table().to_csv(path, index=False))
        self.sha256 = file_sha256(self.path)

    def _set(self, row: dict[str, Any]) -> None:
        key = str(row.get("platform_code", "")), str(row.get("target_type", "")), str(row.get("target_id", ""))
        previous = self.rows.get(key, {})
        normalized = {
            column: str(previous.get(column, "") if row.get(column) is None else row.get(column, ""))
            for column in REVIEW_COLUMNS
        }
        normalized["review_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._validate(normalized)
        key = normalized["platform_code"], normalized["target_type"], normalized["target_id"]
        self.rows[key] = normalized

    def set_observations(
        self, rows: pd.DataFrame, source_indices: list[int], decision: str, *,
        decision_source: str, config_sha256: str, note: str | None = None,
    ) -> None:
        if decision not in {"keep", "reject", "uncertain"}:
            raise ValueError("Observation decision must be keep, reject, or uncertain")
        lookup = rows.set_index("source_obs_index", drop=False)
        for source_index in source_indices:
            if source_index not in lookup.index:
                raise ValueError(f"Unknown source observation: {source_index}")
            point = lookup.loc[source_index]
            event_id = point.local_event_id or point.human_review_event_id
            event_type = point.local_event_type or point.human_review_event_type
            auto_status = point.point_auto_status or point.human_auto_status_snapshot
            auto_decision = point.point_auto_decision or point.human_auto_decision_snapshot
            auto_reason = point.point_auto_reason or point.human_auto_reason_snapshot
            self._set({
                "platform_code": point.platform_code, "source_sha256": point.source_sha256,
                "target_type": "observation", "target_id": f"observation:{source_index}",
                "source_obs_index": source_index, "event_id": event_id,
                "event_type": event_type, "decision": decision,
                "decision_source": decision_source,
                "auto_status_snapshot": auto_status,
                "auto_decision_snapshot": auto_decision,
                "auto_reason_snapshot": auto_reason,
                "qc_config_sha256": config_sha256, "note": note,
            })

    def set_segment(
        self, row: pd.Series, decision: str, *, decision_source: str,
        config_sha256: str, note: str | None = None,
    ) -> None:
        if decision not in {"retain", "exclude", "uncertain"}:
            raise ValueError("Segment decision must be retain, exclude, or uncertain")
        self._set({
            "platform_code": row.platform_code, "source_sha256": row.source_sha256,
            "target_type": "segment", "target_id": row.segment_fingerprint,
            "source_obs_index": "", "event_id": row.temporal_event_id,
            "event_type": "temporal_segment", "decision": decision,
            "decision_source": decision_source,
            "auto_status_snapshot": row.temporal_auto_status,
            "auto_decision_snapshot": row.temporal_auto_decision,
            "auto_reason_snapshot": row.temporal_auto_reason,
            "qc_config_sha256": config_sha256, "note": note,
        })

    def validate_sources(self, source_hashes: dict[str, str], source_indices: dict[str, set[int]]) -> None:
        unknown = {key[0] for key in self.rows} - set(source_hashes)
        if unknown:
            raise ValueError(f"Position review contains unknown platforms: {sorted(unknown)}")
        for (platform, target_type, _), row in self.rows.items():
            if row["source_sha256"] != source_hashes[platform]:
                raise ValueError(f"Position review source hash changed for {platform}")
            if target_type == "observation" and int(row["source_obs_index"]) not in source_indices[platform]:
                raise ValueError(f"Position review contains unknown source observation for {platform}")


@dataclass(frozen=True)
class EventPresentation:
    """Display-only interpretation of evidence already emitted by native QC."""

    platform_code: str
    event_id: str
    kind: str
    event_type: str
    title: str
    explanation: str
    implicated_sources: tuple[int, ...]
    proposed_sources: tuple[int, ...]
    anchor_sources: tuple[int, ...]
    flagged_edges: tuple[tuple[int, int], ...]
    bridge: tuple[int, int, float] | None
    display_sources: tuple[int, ...]


_EVENT_TITLES = {
    "single_edge_ambiguous": "Ambiguous high-speed edge",
    "local_spike_or_short_block": "Confirmed speed-cure repair",
    "persistent_excursion": "Persistent excursion",
    "boundary_or_insufficient_context": "Timing or boundary-limited anomaly",
    "repeated_position": "Exact repeated position",
    "short_interval": "Automatically removed short interval",
    "pre_main_fragment": "Early detached fragment",
    "post_main_fragment": "Late detached fragment",
    "detached_middle_fragment": "Detached middle fragment",
    "multiple_substantial_segments": "Multiple substantial segments",
    "temporal_segment": "Temporal segment",
}

_REASON_TEXT = {
    "single_high_speed_edge_has_no_unique_bad_endpoint":
        "One high-speed edge is present, but the evidence does not identify a unique bad endpoint.",
    "no_bidirectionally_confirmed_block_within_skip_limit":
        "The excursion is not a uniquely confirmed short block in both scan directions.",
    "no_bidirectionally_confirmed_cure_within_timing_limit":
        "The excursion does not return to a uniquely confirmed bridge within the configured timing limit.",
    "high_speed_edge_has_untrusted_timing":
        "High-speed geometry is present, but its timing is outside the trusted local interval.",
    "duplicate_or_nonpositive_timestamp":
        "Duplicate or nonpositive timing prevents a trusted local adjacency decision.",
    "exact_repeated_coordinates":
        "The exact finite longitude/latitude pair occurs more than once in eligible retained data.",
    "positive_dt_below_minimum_local_interval":
        "This observation arrived too soon after the last retained observation and was removed automatically.",
    "bidirectional_exact_block_with_confirmed_continuations":
        "Forward and backward recovery agree on the same skipped observations and plausible bridge.",
    "unique_one_sided_local_residual_with_ordinary_bridge":
        "One endpoint is a strong local interpolation-residual outlier, and only removing that endpoint restores locally ordinary motion.",
    "bidirectionally_confirmed_time_bounded_excursion_cure":
        "Forward and backward searches found the same smallest time-bounded block whose removal cures the speed anomaly.",
    "unique_endpoint_speed_cure":
        "Only one endpoint removal cures the high-speed edge; local speed statistics rank the repaired bridge without vetoing it.",
    "small_observation_and_duration_fraction_relative_to_sustained_dominant_segment":
        "This segment is small in both observation count and duration relative to the sustained dominant segment.",
    "all_substantial_segments_require_review":
        "More than one substantial segment exists, so each segment needs an explicit decision.",
    "only_substantial_segment": "This is the only substantial segment.",
}

_MANUAL_GEOMETRY_CONFIRMATION_REASONS = {
    "unique_one_sided_local_residual_with_ordinary_bridge",
    "unique_endpoint_speed_cure",
    "bidirectionally_confirmed_time_bounded_excursion_cure",
}


def _safe_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _finite_text(value: Any, suffix: str = "", digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not available"
    return f"{number:.{digits}g}{suffix}" if np.isfinite(number) else "not available"


def _json_value(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


EVENT_COLUMNS = [
    "event_id", "kind", "platform_code", "pending", "stale",
    "manual_confirmation_required", "event_type", "event_start_time_utc",
    "event_start_source_obs_index",
]


def position_event_catalog(frame: pd.DataFrame, config_sha256: str) -> pd.DataFrame:
    """Build the compact, mode-independent event index stored in each footer."""
    if frame.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    platform = str(frame.platform_code.iloc[0])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for kind, column in (
        ("temporal", "temporal_event_id"),
        ("local", "local_event_id"),
        ("reviewed_local", "human_review_event_id"),
    ):
        selected = frame.loc[frame[column].fillna("").astype(str).ne("")]
        for event_id, group in selected.groupby(column, sort=False):
            identity = str(event_id)
            if identity in seen:
                continue
            seen.add(identity)
            first = group.iloc[0]
            auto_decision = (
                _safe_text(first.temporal_auto_decision) if kind == "temporal" else
                _safe_text(first.point_auto_decision) if kind == "local" else
                _safe_text(first.human_auto_decision_snapshot)
            )
            auto_reason = (
                _safe_text(first.temporal_auto_reason) if kind == "temporal" else
                _safe_text(first.point_auto_reason) if kind == "local" else
                _safe_text(first.human_auto_reason_snapshot)
            )
            auto_resolved = auto_decision in {"retain", "exclude", "reject"}
            temporal_conflict = bool(
                kind == "temporal" and auto_resolved
                and ((first.human_segment_decision == "retain" and auto_decision == "exclude")
                     or (first.human_segment_decision == "exclude" and auto_decision == "retain"))
            )
            if kind == "temporal":
                reviewed = group.human_segment_decision.fillna("").astype(str).ne("")
                stale = bool(
                    reviewed.any()
                    and group.loc[reviewed, "human_segment_review_config_sha256"]
                    .fillna("").astype(str).ne(config_sha256).any()
                )
            else:
                reviewed = group.human_position_decision.fillna("").astype(str).ne("")
                stale = bool(
                    reviewed.any()
                    and group.loc[reviewed, "human_review_config_sha256"]
                    .fillna("").astype(str).ne(config_sha256).any()
                )
            geometry_confirmation = auto_reason in _MANUAL_GEOMETRY_CONFIRMATION_REASONS
            reviewed_geometry = bool(
                reviewed.any()
                and (
                    geometry_confirmation
                    or group.human_auto_reason_snapshot.fillna("").astype(str)
                    .isin(_MANUAL_GEOMETRY_CONFIRMATION_REASONS).any()
                )
            )
            geometry_confirmation_required = bool(
                geometry_confirmation and not reviewed.all()
            )
            pending = bool(
                group.final_position_status.isin(["unresolved", "uncertain"]).any()
                or group.local_review_required.astype(bool).any()
                or temporal_conflict or stale
                or (group.human_position_decision.fillna("").astype(str).eq("keep").any()
                    and auto_resolved and not reviewed_geometry)
            )
            # R confirms a proposed rejection and K accepts an observed point as
            # real motion. Recalculation may retain, shrink, or replace a block;
            # reviewed historical evidence itself no longer remains pending.
            if reviewed_geometry:
                pending = False
            event_type = (
                _safe_text(first.temporal_event_type) if kind == "temporal" else
                _safe_text(first.human_review_event_type) if kind == "reviewed_local" else
                _safe_text(first.local_event_type)
            )
            if event_type in {"repeated_position", "short_interval"}:
                pending = False
            ordered = group.sort_values(["time", "source_obs_index"], kind="stable")
            start = ordered.iloc[0]
            start_time = "" if pd.isna(start.time) else pd.Timestamp(start.time).isoformat()
            rows.append({
                "event_id": identity, "kind": kind, "platform_code": platform,
                "pending": pending, "stale": stale,
                "manual_confirmation_required": geometry_confirmation_required,
                "event_type": event_type,
                "event_start_time_utc": start_time,
                "event_start_source_obs_index": int(start.source_obs_index),
            })
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


class PositionReviewer:
    """Lazy three-panel native-position reviewer with staged decisions."""

    SESSION_SCHEMA_VERSION = 2

    def __init__(
        self, table: pd.DataFrame | Mapping[str, pd.DataFrame] | None, reviews: PositionReviews, *,
        mode: str, config_sha256: str, trajectory_loader: Callable[[str], Any],
        recompute_platform: Callable[[str], pd.DataFrame],
        checkpoint: Callable[[], Any] | None = None, context_points: int = 12,
        session_path: str | Path | None = None,
        position_config: NativePositionConfig | None = None,
        platform_frames: Mapping[str, pd.DataFrame] | None = None,
        platform_loader: Callable[[str], pd.DataFrame] | None = None,
        event_catalog: pd.DataFrame | None = None,
        cache_size: int = 2,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.widgets import Button

        if mode not in {"manual", "semiautomatic"}:
            raise ValueError("PositionReviewer mode must be manual or semiautomatic")
        self.plt, self.Button, self.Line2D = plt, Button, Line2D
        self.reviews, self.mode = reviews, mode
        self.config_sha256 = config_sha256
        self.context_points = int(context_points)
        self.position_config = position_config or NativePositionConfig()
        self.trajectory_loader = trajectory_loader
        self.recompute_platform = recompute_platform
        self.checkpoint = checkpoint
        self.platform_loader = platform_loader
        self.cache_size = max(1, int(cache_size))
        self.session_path = Path(session_path) if session_path else reviews.path.with_suffix(".session.json")
        supplied = platform_frames if platform_frames is not None else table
        if isinstance(supplied, pd.DataFrame):
            self.frames: OrderedDict[str, pd.DataFrame] = OrderedDict({
                str(code): group.copy()
                for code, group in supplied.groupby(supplied.platform_code.astype(str), sort=False)
            })
        elif supplied is None:
            self.frames = OrderedDict()
        else:
            self.frames = OrderedDict((str(code), frame) for code, frame in supplied.items())
        # Kept only as a compatibility handle; all reviewer lookups use frames.
        self.table = table
        self._trajectory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._platform_events: dict[str, pd.DataFrame] = {}
        if event_catalog is not None:
            event_catalog = event_catalog.copy()
            catalog_defaults = {
                "manual_confirmation_required": False,
                "event_type": "",
                "event_start_time_utc": "",
                "event_start_source_obs_index": np.nan,
            }
            for column, default in catalog_defaults.items():
                if column not in event_catalog:
                    event_catalog[column] = default
            missing_type = event_catalog.event_type.fillna("").astype(str).eq("")
            event_catalog.loc[
                missing_type & event_catalog.event_id.astype(str).str.startswith("repeated_position:"),
                "event_type",
            ] = "repeated_position"
            event_catalog.loc[
                missing_type & event_catalog.event_id.astype(str).str.startswith("short_interval:"),
                "event_type",
            ] = "short_interval"
            for code, group in event_catalog.groupby(event_catalog.platform_code.astype(str), sort=False):
                self._platform_events[str(code)] = group[EVENT_COLUMNS].reset_index(drop=True).copy()
        self._known_platforms = set(self.frames) | set(self._platform_events)
        self._presentation_cache: dict[tuple[str, str, int | None], EventPresentation] = {}
        self._source_row_cache: dict[str, dict[int, int]] = {}
        self._surviving_edges_cache: dict[str, list[tuple[int, int]]] = {}
        self._scope_status_cache: dict[str, dict[int, tuple[str, bool]]] = {}
        self._velocity_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._provisional_speed_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._pick_sources: dict[Any, np.ndarray] = {}
        self._restored_draft = ""
        self._resume_message = ""
        self._status_message = ""
        self._pending_refresh_platforms: set[str] = set()
        self._staged: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._close_handled = False
        self._button_closing = False
        self.selected_source_obs_index: int | None = None
        self.display_source_indices: list[int] = []
        self.selectable_source_indices: list[int] = []
        self.presentation: EventPresentation | None = None
        self._selection_artists: list[Any] = []
        self._overview_platform: str | None = None
        self._overview_pick_artist: Any | None = None
        self._overview_event_artists: list[Any] = []
        self._blit_background: Any | None = None
        self._capturing_background = False
        self._raw_incoming: dict[int, tuple[float, float, int]] = {}
        self._raw_outgoing: dict[int, tuple[float, float, int]] = {}
        self._surviving_incoming: dict[int, tuple[float, float, int]] = {}
        self._surviving_outgoing: dict[int, tuple[float, float, int]] = {}
        self._provisional_incoming: dict[int, tuple[float, float, int]] = {}
        self._provisional_outgoing: dict[int, tuple[float, float, int]] = {}
        self._completion_screen = False
        self.events = self._events()
        self.index = 0
        self._restore_session()

        self.figure = plt.figure(figsize=(15.5, 9.2))
        # K/R/P/Q overlap Matplotlib's built-in scale/home/pan/quit shortcuts.
        # Leaving the default handler connected can silently switch toolbar mode
        # (notably P -> pan), after which map clicks no longer select points.
        default_key_handler = getattr(
            self.figure.canvas.manager, "key_press_handler_id", None,
        )
        if default_key_handler is not None:
            self.figure.canvas.mpl_disconnect(default_key_handler)
        grid = self.figure.add_gridspec(
            2, 2, left=.055, right=.985, bottom=.25, top=.80,
            wspace=.24, hspace=.34, height_ratios=[1, 1.12],
        )
        self.overview_ax = self.figure.add_subplot(grid[0, 0])
        self.local_ax = self.figure.add_subplot(grid[0, 1])
        self.velocity_ax = self.figure.add_subplot(grid[1, :])
        # Compatibility names used by existing callers.
        self.map_ax, self.speed_ax = self.local_ax, self.velocity_ax
        self.header_text = self.figure.text(.055, .955, "", va="top", fontsize=12, weight="bold")
        self.explanation_text = self.figure.text(.055, .905, "", va="top", fontsize=9)
        self.target_text = self.figure.text(.055, .835, "", va="top", fontsize=9, weight="bold")
        self.status_text = self.figure.text(.055, .215, "", va="top", fontsize=8, color=".25")
        self.buttons: dict[str, Any] = {}
        self._make_buttons()
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_press_event", self.on_map_click)
        self.figure.canvas.mpl_connect("draw_event", self.on_canvas_draw)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.draw()

    # ---- event catalog and immutable source lookup ---------------------

    def _frame(self, platform: str) -> pd.DataFrame:
        platform = str(platform)
        if platform not in self.frames:
            if self.platform_loader is None:
                raise KeyError(platform)
            self.frames[platform] = self.platform_loader(platform)
        self.frames.move_to_end(platform)
        while len(self.frames) > self.cache_size:
            evicted, _ = self.frames.popitem(last=False)
            self._source_row_cache.pop(evicted, None)
            self._surviving_edges_cache.pop(evicted, None)
            self._scope_status_cache.pop(evicted, None)
            self._presentation_cache = {
                key: value for key, value in self._presentation_cache.items() if key[0] != evicted
            }
            self._velocity_cache = {
                key: value for key, value in self._velocity_cache.items() if key[0] != evicted
            }
            self._provisional_speed_cache = {
                key: value for key, value in self._provisional_speed_cache.items() if key[0] != evicted
            }
        return self.frames[platform]

    def _source_rows(self, platform: str) -> dict[int, int]:
        platform = str(platform)
        if platform not in self._source_row_cache:
            source = self._frame(platform).source_obs_index.to_numpy(dtype=np.int64)
            self._source_row_cache[platform] = {
                int(value): position for position, value in enumerate(source)
            }
        return self._source_row_cache[platform]

    def _native(self, platform: str) -> dict[str, Any]:
        platform = str(platform)
        if platform not in self._trajectory_cache:
            trajectory = self.trajectory_loader(platform)
            time = np.asarray(trajectory.time, dtype="datetime64[ns]")
            order = np.argsort(time, kind="stable")
            source = np.asarray(trajectory.source_obs_index, dtype=np.int64)
            self._trajectory_cache[platform] = {
                "trajectory": trajectory, "time": time,
                "times": pd.to_datetime(time, utc=True), "order": order,
                "source": source,
                "source_to_native": {int(value): i for i, value in enumerate(source)},
            }
        self._trajectory_cache.move_to_end(platform)
        while len(self._trajectory_cache) > self.cache_size:
            self._trajectory_cache.popitem(last=False)
        return self._trajectory_cache[platform]

    def _events_for_platform(self, platform: str) -> pd.DataFrame:
        if platform in self._platform_events:
            return self._platform_events[platform]
        result = position_event_catalog(self._frame(platform), self.config_sha256)
        self._platform_events[platform] = result
        return result

    def _events(self) -> pd.DataFrame:
        parts = [self._events_for_platform(platform) for platform in sorted(self._known_platforms)]
        result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if result.empty:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        if "manual_confirmation_required" not in result:
            result["manual_confirmation_required"] = False
        for column, default in (
            ("event_type", ""), ("event_start_time_utc", ""),
            ("event_start_source_obs_index", np.nan),
        ):
            if column not in result:
                result[column] = default
        result = result.loc[
            ~result.event_type.fillna("").astype(str).isin(["repeated_position", "short_interval"])
            & ~result.event_id.astype(str).str.startswith("repeated_position:")
            & ~result.event_id.astype(str).str.startswith("short_interval:")
        ].copy()
        if result.empty:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        if self.mode == "manual":
            result = result.copy()
            result["pending"] = (
                result.pending.astype(bool)
                | result.manual_confirmation_required.astype(bool)
            )
        if self.mode == "semiautomatic":
            result = result.loc[result.pending.astype(bool)].copy()
        result["_event_time"] = pd.to_datetime(
            result.event_start_time_utc, errors="coerce", utc=True,
        )
        result["_event_source"] = pd.to_numeric(
            result.event_start_source_obs_index, errors="coerce",
        ).fillna(np.inf)
        return result.sort_values(
            ["pending", "stale", "platform_code", "_event_time", "_event_source", "kind", "event_id"],
            ascending=[False, False, True, True, True, True, True],
            kind="stable", na_position="last",
        ).drop(columns=["_event_time", "_event_source"]).reset_index(drop=True)

    def _group_for(self, event: pd.Series) -> pd.DataFrame:
        column = (
            "temporal_event_id" if event.kind == "temporal" else
            "human_review_event_id" if event.kind == "reviewed_local" else
            "local_event_id"
        )
        frame = self._frame(str(event.platform_code))
        return frame[frame[column].astype(str) == str(event.event_id)]

    def _group(self) -> tuple[pd.Series, pd.DataFrame]:
        event = self.events.iloc[self.index]
        return event, self._group_for(event)

    @staticmethod
    def _implicated(group: pd.DataFrame) -> list[int]:
        values: list[int] = []
        if "local_event_implicated_source_obs_indices" in group:
            for raw in group.local_event_implicated_source_obs_indices.astype(str).unique():
                decoded = _json_value(raw, [])
                if isinstance(decoded, list):
                    values.extend(int(value) for value in decoded)
        values.extend(group.source_obs_index.astype(int).tolist())
        return list(dict.fromkeys(values))

    def _event_contains(self, position: int, source_index: int) -> bool:
        return int(source_index) in self._implicated(self._group_for(self.events.iloc[position]))

    # ---- presentation-only evidence translation -----------------------

    def _edge_pairs(self, group: pd.DataFrame, platform: str) -> list[tuple[int, int]]:
        frame = self._frame(platform).reset_index(drop=True)
        pairs: list[tuple[int, int]] = []
        for raw in group.forward_result_json.astype(str).unique():
            decoded = _json_value(raw, None)
            candidates = decoded if isinstance(decoded, list) else []
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    a, b = int(item["a"]), int(item["b"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= a < len(frame) and 0 <= b < len(frame):
                    pairs.append((int(frame.iloc[a].source_obs_index), int(frame.iloc[b].source_obs_index)))
            if isinstance(decoded, dict):
                try:
                    row_positions = {
                        int(decoded["anchor"]), int(decoded["reconnect"]),
                        *(int(value) for value in decoded.get("skipped", [])),
                    }
                    ordered = sorted(row_positions, key=lambda value: frame.iloc[value].time)
                except (KeyError, TypeError, ValueError, IndexError):
                    ordered = []
                for a, b in zip(ordered[:-1], ordered[1:]):
                    if 0 <= a < len(frame) and 0 <= b < len(frame):
                        pairs.append((int(frame.iloc[a].source_obs_index), int(frame.iloc[b].source_obs_index)))
        if not pairs:
            lookup = frame.set_index("source_obs_index", drop=False)
            for row in group.itertuples():
                if bool(row.high_speed_flag) and not pd.isna(row.surviving_predecessor_source_obs_index):
                    pairs.append((int(row.surviving_predecessor_source_obs_index), int(row.source_obs_index)))
            # Historical rows retain the original event but not necessarily the current edge flag.
            if not pairs and len(group) > 1:
                ordered = group.sort_values("time", kind="stable").source_obs_index.astype(int).tolist()
                pairs.extend(zip(ordered[:-1], ordered[1:]))
        return list(dict.fromkeys(pairs))

    def _presentation(self, event: pd.Series, group: pd.DataFrame, *,
                      center_source: int | None = None) -> EventPresentation:
        platform = str(event.platform_code)
        cache_key = (platform, str(event.event_id), center_source)
        if cache_key in self._presentation_cache:
            return self._presentation_cache[cache_key]
        native = self._native(platform)
        frame = self._frame(platform)
        first = group.iloc[0]
        if event.kind == "temporal":
            event_type = _safe_text(first.temporal_event_type) or "temporal_segment"
            reason = _safe_text(first.temporal_auto_reason)
        elif event.kind == "reviewed_local":
            event_type = _safe_text(first.human_review_event_type) or "reviewed_local_event"
            reason = _safe_text(first.human_auto_reason_snapshot)
        else:
            event_type = _safe_text(first.local_event_type) or "local_event"
            reason = _safe_text(first.point_auto_reason) or _safe_text(first.local_review_reason)
        implicated = self._implicated(group)
        exact_group = event_type in {"repeated_position", "local_spike_or_short_block"}
        proposed = (
            group.sort_values(["time", "source_obs_index"], kind="stable")
            .source_obs_index.astype(int).drop_duplicates().tolist()
            if exact_group else []
        )
        anchors: list[int] = []
        bridge: tuple[int, int, float] | None = None
        bridge_rows = group[group.candidate_bridge_anchor_source_obs_index.notna()]
        if len(bridge_rows):
            bridge_row = bridge_rows.iloc[0]
            a = int(bridge_row.candidate_bridge_anchor_source_obs_index)
            b = int(bridge_row.candidate_bridge_reconnection_source_obs_index)
            speed = float(bridge_row.candidate_bridge_speed_m_s)
            anchors = [a]
            bridge = (a, b, speed)
        order_sources = native["source"][native["order"]].astype(int).tolist()
        source_positions = {value: i for i, value in enumerate(order_sources)}
        focus = [value for value in implicated if value in source_positions]
        if center_source is not None and center_source in source_positions:
            lower = max(0, source_positions[center_source] - self.context_points)
            upper = min(len(order_sources), source_positions[center_source] + self.context_points + 1)
        elif exact_group and focus:
            # Repeated-coordinate groups can contain thousands of observations.
            # Keep the local panes centred on one member; the overview continues
            # to represent the complete group.  Spanning first-to-last here used
            # to create thousands of labels and edge artists and freeze the GUI.
            center = source_positions[focus[0]]
            lower = max(0, center - self.context_points)
            upper = min(len(order_sources), center + self.context_points + 1)
        elif focus:
            positions = [source_positions[value] for value in focus]
            lower = max(0, min(positions) - self.context_points)
            upper = min(len(order_sources), max(positions) + self.context_points + 1)
        else:
            lower, upper = 0, min(len(order_sources), 2 * self.context_points + 1)
        display = order_sources[lower:upper]
        title = _EVENT_TITLES.get(event_type, event_type.replace("_", " ").title())
        if reason in _MANUAL_GEOMETRY_CONFIRMATION_REASONS:
            title = (
                "Speed-cure repair — confirmation required"
                if bool(getattr(event, "manual_confirmation_required", False))
                else "Reviewed speed-cure repair"
            )
        explanation = _REASON_TEXT.get(reason, reason.replace("_", " ").strip().capitalize())
        if event.kind == "temporal":
            explanation += (
                f" Segment {first.segment_id} contains {int(first.segment_n_observations)} observations "
                f"over {_finite_text(first.segment_duration_hours, ' h')}; observation/duration fractions "
                f"are {_finite_text(first.segment_fraction_total_observations)} and "
                f"{_finite_text(first.segment_fraction_total_duration)}."
            )
        elif event_type == "repeated_position":
            explanation += (
                f" The group contains {int(first.repeated_coordinate_count)} occurrences over "
                f"{_finite_text(first.repeat_duration_seconds, ' s')}."
            )
        elif reason == "unique_one_sided_local_residual_with_ordinary_bridge":
            chosen = _json_value(str(first.forward_result_json), {})
            competitor = _json_value(str(first.backward_result_json), {})
            baseline = chosen.get("baseline", {}) if isinstance(chosen, dict) else {}
            explanation += (
                f" Candidate residual {_finite_text(chosen.get('residual_m'), ' m')} "
                f"(robust z {_finite_text(chosen.get('residual_robust_z'))}) from "
                f"{int(baseline.get('sample_count', 0))} clean local triplets; bridge "
                f"{_finite_text(chosen.get('bridge_speed'), ' m/s')} with robust deviation "
                f"{_finite_text(chosen.get('bridge_speed_robust_z'))}. The competing endpoint "
                f"has residual z {_finite_text(competitor.get('residual_robust_z'))} and bridge "
                f"deviation {_finite_text(competitor.get('bridge_speed_robust_z'))}."
            )
        elif reason == "unique_endpoint_speed_cure":
            chosen = _json_value(str(first.forward_result_json), {})
            competitor = _json_value(str(first.backward_result_json), {})
            baseline = chosen.get("baseline", {}) if isinstance(chosen, dict) else {}
            warning = (
                " The repaired bridge is statistically unusual but remains physically valid."
                if bool(chosen.get("bridge_atypical_warning")) else ""
            )
            explanation += (
                f" Removing observation {chosen.get('candidate')} gives a "
                f"{_finite_text(chosen.get('bridge_speed'), ' m/s')} bridge over "
                f"{_finite_text(chosen.get('bridge_dt_seconds'), ' s')} "
                f"(smoothness score {_finite_text(chosen.get('smoothness_score'))}) against "
                f"{int(baseline.get('sample_count', 0))} clean local speeds with median "
                f"{_finite_text(baseline.get('speed_median_m_s'), ' m/s')}. The competing "
                f"endpoint bridge is {_finite_text(competitor.get('bridge_speed'), ' m/s')} "
                f"and {'does' if competitor.get('cures') else 'does not'} cure the anomaly."
                f"{warning}"
            )
        elif bridge is not None:
            chosen = _json_value(str(first.forward_result_json), {})
            warning = (
                " The repaired bridge is statistically unusual but remains physically valid."
                if bool(chosen.get("bridge_atypical_warning")) else ""
            )
            explanation += (
                f" The proposed bridge is {bridge[0]} to {bridge[1]} at "
                f"{_finite_text(bridge[2], ' m/s')}; directional recovery agrees on "
                f"{len(proposed)} skipped observation{'s' if len(proposed) != 1 else ''}. "
                f"Its elapsed time is {_finite_text(chosen.get('bridge_dt_seconds'), ' s')} "
                f"and local-speed smoothness score is "
                f"{_finite_text(chosen.get('smoothness_score'))}.{warning}"
            )
        elif event_type in {"single_edge_ambiguous", "persistent_excursion", "boundary_or_insufficient_context"}:
            explanation += f" {len(self._edge_pairs(group, platform))} triggering edge(s) are shown in red."
        edge_pairs = [] if event_type == "repeated_position" else self._edge_pairs(group, platform)
        result = EventPresentation(
            platform, str(event.event_id), str(event.kind), event_type, title,
            explanation.strip(), tuple(implicated), tuple(proposed), tuple(anchors),
            tuple(edge_pairs), bridge, tuple(display),
        )
        self._presentation_cache[cache_key] = result
        return result

    # ---- session persistence and reconciliation -----------------------

    def _source_hash(self, platform: str) -> str:
        values = self._frame(platform).source_sha256.astype(str).unique()
        return str(values[0]) if len(values) == 1 else ""

    def _selected_time(self) -> str:
        if self.selected_source_obs_index is None or self.events.empty:
            return ""
        platform = str(self.events.iloc[self.index].platform_code)
        native = self._native(platform)
        position = native["source_to_native"].get(int(self.selected_source_obs_index))
        if position is None or np.isnat(native["time"][position]):
            return ""
        return pd.Timestamp(native["time"][position], tz="UTC").isoformat()

    def _session_payload(self) -> dict[str, Any]:
        if self.events.empty:
            return {
                "schema_version": self.SESSION_SCHEMA_VERSION,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(), "mode": self.mode,
                "platform_code": "", "event_id": "", "event_queue_index": 0,
                "selected_source_obs_index": None, "selected_time_utc": "",
                "config_sha256": self.config_sha256, "source_sha256": "",
                "review_sha256": self.reviews.sha256,
                "draft_note": "",
            }
        event = self.events.iloc[self.index]
        platform = str(event.platform_code)
        return {
            "schema_version": self.SESSION_SCHEMA_VERSION,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(), "mode": self.mode,
            "platform_code": platform, "event_id": str(event.event_id),
            "event_queue_index": int(self.index),
            "selected_source_obs_index": self.selected_source_obs_index,
            "selected_time_utc": self._selected_time(),
            "config_sha256": self.config_sha256,
            "source_sha256": self._source_hash(platform),
            "review_sha256": self.reviews.sha256,
            "draft_note": "",
        }

    def _save_session(self) -> None:
        payload = json.dumps(self._session_payload(), indent=2, sort_keys=True)
        _atomic(self.session_path, lambda path: path.write_text(payload + "\n", encoding="utf-8"))

    def _fallback_index(self, start: int = 0) -> int:
        if self.events.empty:
            return 0
        try:
            start = int(start)
        except (TypeError, ValueError):
            start = 0
        start = min(max(start, 0), len(self.events) - 1)
        for position in range(start, len(self.events)):
            if bool(self.events.iloc[position].pending):
                return position
        pending = np.flatnonzero(self.events.pending.to_numpy(dtype=bool))
        if len(pending):
            return int(pending[0])
        return 0

    def _first_pending_index(self) -> int | None:
        if self.events.empty:
            return None
        pending = np.flatnonzero(self.events.pending.to_numpy(dtype=bool))
        return int(pending[0]) if len(pending) else None

    def _completed_positions(self) -> list[int]:
        if self.events.empty:
            return []
        return np.flatnonzero(~self.events.pending.to_numpy(dtype=bool)).astype(int).tolist()

    @staticmethod
    def _event_anchor(event: pd.Series) -> tuple[str, int, int, str, str]:
        stamp = pd.to_datetime(
            getattr(event, "event_start_time_utc", ""), errors="coerce", utc=True,
        )
        time_value = int(stamp.value) if not pd.isna(stamp) else np.iinfo(np.int64).max
        try:
            source = int(event.event_start_source_obs_index)
        except (TypeError, ValueError, OverflowError):
            source = np.iinfo(np.int64).max
        return (
            str(event.platform_code), time_value, source,
            str(event.kind), str(event.event_id),
        )

    def _finish_restore(self) -> None:
        self._completion_screen = self._first_pending_index() is None
        if self._completion_screen:
            self.selected_source_obs_index = None

    def _next_pending_event_id(self, position: int) -> str | None:
        """Return the next pending identity, wrapping but excluding position."""
        if self.events.empty or len(self.events) == 1:
            return None
        for offset in range(1, len(self.events)):
            candidate = (int(position) + offset) % len(self.events)
            if bool(self.events.iloc[candidate].pending):
                return str(self.events.iloc[candidate].event_id)
        return None

    def _restore_session(self) -> None:
        if not self.session_path.exists() or self.events.empty:
            self.index = self._fallback_index()
            self._finish_restore()
            return
        try:
            saved = json.loads(self.session_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("session is not an object")
        except (OSError, ValueError, json.JSONDecodeError):
            self.index = self._fallback_index()
            self._resume_message = "Saved reviewer session was unreadable; opened the next pending event."
            self._finish_restore()
            return
        event_id = _safe_text(saved.get("event_id"))
        matches = np.flatnonzero(self.events.event_id.astype(str).eq(event_id))
        if saved.get("schema_version") != self.SESSION_SCHEMA_VERSION:
            if len(matches):
                matched = int(matches[0])
                pending = self._first_pending_index()
                if not bool(self.events.iloc[matched].pending) and pending is not None:
                    self.index = pending
                    self.selected_source_obs_index = None
                    self._resume_message = (
                        "Saved event is complete; opened the first pending event. "
                        "Use P to browse completed history."
                    )
                else:
                    self.index = matched
                    group = self._group_for(self.events.iloc[self.index]).sort_values(
                        "source_obs_index", kind="stable",
                    )
                    try:
                        legacy_position = int(saved.get("point_index", 0)) % max(1, len(group))
                        self.selected_source_obs_index = int(group.iloc[legacy_position].source_obs_index)
                    except (TypeError, ValueError, IndexError):
                        self.selected_source_obs_index = None
                    self._resume_message = "Restored and upgraded the legacy reviewer cursor."
            else:
                self.index = self._fallback_index()
                self._resume_message = "Legacy cursor no longer matched an event; opened the next pending event."
            self._finish_restore()
            return
        platform = _safe_text(saved.get("platform_code"))
        if platform not in self._known_platforms or _safe_text(saved.get("source_sha256")) != self._source_hash(platform):
            self.index = self._fallback_index()
            self._resume_message = "Saved cursor was ignored because the native source hash changed."
            self._finish_restore()
            return
        try:
            selected = int(saved["selected_source_obs_index"]) if saved.get("selected_source_obs_index") is not None else None
        except (TypeError, ValueError):
            selected = None
        if len(matches):
            candidate_index = int(matches[0])
        elif selected is not None:
            containing = [position for position in range(len(self.events))
                          if self._event_contains(position, selected)]
            candidate_index = (
                containing[0] if containing
                else self._fallback_index(saved.get("event_queue_index", 0))
            )
        else:
            candidate_index = self._fallback_index(saved.get("event_queue_index", 0))
        pending_fallback = (
            not bool(self.events.iloc[candidate_index].pending)
            and self._first_pending_index() is not None
        )
        self.index = self._first_pending_index() if pending_fallback else candidate_index
        if not pending_fallback:
            current_platform = str(self.events.iloc[self.index].platform_code)
            if selected is not None and selected in self._native(current_platform)["source_to_native"]:
                self.selected_source_obs_index = selected
        self._restored_draft = ""
        changed = (
            _safe_text(saved.get("config_sha256")) != self.config_sha256
            or _safe_text(saved.get("review_sha256")) != self.reviews.sha256
            or not len(matches)
        )
        if pending_fallback:
            self._resume_message = (
                "Saved event is complete; opened the first pending event. "
                "Use P to browse completed history."
            )
        else:
            self._resume_message = (
                "Reconciled the saved cursor with the current event queue."
                if changed else "Resumed the saved reviewer event and source observation."
            )
        self._finish_restore()

    # ---- figure creation and drawing ----------------------------------

    def _button(self, key: str, label: str, bounds: list[float], callback: Callable[[], None]) -> None:
        button = self.Button(self.figure.add_axes(bounds), label)
        button.on_clicked(lambda _event: callback())
        self.buttons[key] = button

    def _make_buttons(self) -> None:
        y, height = .145, .042
        entries = (
            ("previous_point", "←  Previous point", .055, .12, lambda: self.navigate_point(-1)),
            ("next_point", "→  Next point", .18, .11, lambda: self.navigate_point(1)),
            ("keep_point", "K  Keep point", .305, .10, lambda: self.action("keep_point")),
            ("reject_point", "R  Reject point", .41, .11, lambda: self.action("reject_point")),
            ("previous_event", "P  Previous event", .535, .12, lambda: self.navigate_event(-1)),
            ("next_event", "N  Next event", .66, .105, lambda: self.navigate_event(1)),
            ("quit", "Q  Save and quit", .78, .145, self.clean_quit),
        )
        for key, label, x, width, callback in entries:
            self._button(key, label, [x, y, width, height], callback)

    def _set_status(self, message: str, *, error: bool = False, render: bool = True) -> None:
        self._status_message = message
        if hasattr(self, "status_text"):
            self.status_text.set_text(message)
            self.status_text.set_color("crimson" if error else ".25")
            if render:
                # Status changes can originate inside a Qt key/click callback.
                # Scheduling a regular draw is safer there than performing a
                # nested blit while the GUI backend is dispatching that event.
                self._blit_background = None
                self.figure.canvas.draw_idle()

    def _default_selection(self, presentation: EventPresentation) -> int | None:
        candidates = presentation.proposed_sources or presentation.implicated_sources
        if candidates:
            return int(candidates[0])
        if presentation.display_sources:
            return int(presentation.display_sources[len(presentation.display_sources) // 2])
        return None

    def _current_row(self) -> pd.Series | None:
        if self._completion_screen or self.events.empty or self.selected_source_obs_index is None:
            return None
        frame = self._frame(str(self.events.iloc[self.index].platform_code))
        position = self._source_rows(str(self.events.iloc[self.index].platform_code)).get(
            int(self.selected_source_obs_index),
        )
        return frame.iloc[position] if position is not None else None

    def _hidden_automatic_source(self, platform: str, source_index: int) -> bool:
        position = self._source_rows(platform).get(int(source_index))
        if position is None:
            return False
        row = self._frame(platform).iloc[position]
        return bool(
            row.exact_repeat_flag
            or _safe_text(row.point_auto_status) == "short_interval_reject"
        )

    @staticmethod
    def _eligible_point(row: pd.Series | None) -> bool:
        if row is None:
            return False
        return bool(
            row.valid_timestamp and row.source_position_valid and row.drogue_eligible
            and row.deployment_eligible and not row.deployment_uncertain
            and _safe_text(row.segment_status) == "retained"
            and not bool(row.exact_repeat_flag)
            and _safe_text(row.point_auto_status) != "short_interval_reject"
        )

    def _refresh_selectable_sources(self) -> None:
        self.selectable_source_indices = []
        if self.events.empty or self.events.iloc[self.index].kind == "temporal":
            return
        platform = str(self.events.iloc[self.index].platform_code)
        frame = self._frame(platform)
        eligible = (
            frame.valid_timestamp.to_numpy(dtype=bool)
            & frame.source_position_valid.to_numpy(dtype=bool)
            & frame.drogue_eligible.to_numpy(dtype=bool)
            & frame.deployment_eligible.to_numpy(dtype=bool)
            & ~frame.deployment_uncertain.to_numpy(dtype=bool)
            & (frame.segment_status.astype(str).to_numpy() == "retained")
            & ~frame.exact_repeat_flag.to_numpy(dtype=bool)
            & ~frame.point_auto_status.fillna("").astype(str).eq(
                "short_interval_reject"
            ).to_numpy(dtype=bool)
        )
        selectable = frame.loc[eligible, ["source_obs_index", "time"]].copy()
        selectable = selectable.sort_values("time", kind="stable")
        self.selectable_source_indices = selectable.source_obs_index.astype(int).tolist()

    def _selected_navigation_position(self) -> int | None:
        if self.selected_source_obs_index is None:
            return None
        try:
            return self.selectable_source_indices.index(int(self.selected_source_obs_index))
        except ValueError:
            return None

    def _staged_point_decision(self, platform: str, source: int) -> str:
        item = self._staged.get((str(platform), "observation", f"observation:{int(source)}"))
        return _safe_text(item.get("decision")) if item else ""

    def _repeat_group_sources(self, platform: str, row: pd.Series | None) -> list[int]:
        if row is None:
            return []
        group_id = _safe_text(row.repeated_coordinate_group_id)
        if not group_id or not bool(row.exact_repeat_flag):
            return [int(row.source_obs_index)]
        frame = self._frame(platform)
        members = frame.loc[
            frame.repeated_coordinate_group_id.fillna("").astype(str).eq(group_id)
            & frame.exact_repeat_flag.astype(bool)
        ]
        return members.sort_values("time", kind="stable").source_obs_index.astype(int).tolist()

    @staticmethod
    def _compact_source_target(values: list[int], *, max_spans: int = 8) -> str:
        """Render large immutable targets without creating enormous Matplotlib text."""
        ordered = sorted(dict.fromkeys(map(int, values)))
        if not ordered:
            return "none"
        spans: list[tuple[int, int]] = []
        start = previous = ordered[0]
        for value in ordered[1:]:
            if value == previous + 1:
                previous = value
                continue
            spans.append((start, previous))
            start = previous = value
        spans.append((start, previous))
        rendered = [str(a) if a == b else f"{a}–{b}" for a, b in spans[:max_spans]]
        if len(spans) > max_spans:
            rendered.append("…")
        suffix = f" ({len(ordered)} observations)" if len(ordered) > 1 else ""
        return ", ".join(rendered) + suffix

    @staticmethod
    def _display_sample(values: list[int], *, maximum: int = 750) -> list[int]:
        """Bound presentation cost without changing the staged decision set."""
        if len(values) <= maximum:
            return values
        positions = np.linspace(0, len(values) - 1, maximum, dtype=int)
        return [values[position] for position in positions]

    def _point_is_rejected(self, platform: str, row: pd.Series | None) -> bool:
        if row is None:
            return False
        staged = self._staged_point_decision(platform, int(row.source_obs_index))
        if staged:
            return staged == "reject"
        human = _safe_text(row.human_position_decision)
        if human:
            return human == "reject"
        return bool(
            _safe_text(row.final_position_status) == "rejected"
            or _safe_text(row.point_auto_decision) == "reject"
            or _safe_text(row.human_auto_decision_snapshot) == "reject"
        )

    def _set_button_states(self) -> None:
        if not hasattr(self, "buttons"):
            return
        has_event = not self.events.empty and not self._completion_screen
        temporal = bool(has_event and self.events.iloc[self.index].kind == "temporal")
        eligible = self._eligible_point(self._current_row()) and not temporal
        current_row = self._current_row()
        platform = str(self.events.iloc[self.index].platform_code) if has_event else ""
        rejected = bool(eligible and self._point_is_rejected(platform, current_row))
        point_labels = (
            ("K  Retain segment", "R  Exclude segment")
            if temporal else ("K  Keep point", "R  Reject point")
        )
        for key, label in zip(("keep_point", "reject_point"), point_labels):
            self.buttons[key].label.set_text(label)
        self.buttons["keep_point"].set_active(temporal or rejected)
        self.buttons["reject_point"].set_active(temporal or eligible)
        navigation_position = self._selected_navigation_position()
        self.buttons["previous_point"].set_active(bool(
            not temporal and self.selectable_source_indices
            and (navigation_position is None or navigation_position > 0)
        ))
        self.buttons["next_point"].set_active(bool(
            not temporal and self.selectable_source_indices
            and (navigation_position is None or navigation_position < len(self.selectable_source_indices) - 1)
        ))
        self.buttons["previous_event"].set_active(bool(
            self._completed_positions() or self._first_pending_index() is not None
        ))
        self.buttons["next_event"].set_active(bool(has_event))

    def _action_target(self) -> str:
        if self._completion_screen or self.events.empty or self.presentation is None:
            return "Action target: none"
        event, group = self._group()
        if event.kind == "temporal":
            return f"Action target: segment fingerprint {_safe_text(group.iloc[0].segment_fingerprint)}"
        selected = "none" if self.selected_source_obs_index is None else str(self.selected_source_obs_index)
        row = self._current_row()
        group = self._repeat_group_sources(str(event.platform_code), row)
        if len(group) > 1:
            return "Action target: repeated-position group [" + self._compact_source_target(group) + "]"
        return f"Action target: selected observation [{selected}]"
        group_text = (
            ", ".join(map(str, self.presentation.proposed_sources))
            if self.presentation.proposed_sources else "disabled — select one point"
        )
        return f"Action target: selected observation [{selected}]; exact block/group [{group_text}]"

    def _legend_handles(self) -> list[Any]:
        return [
            self.Line2D([], [], color=".62", marker=".", lw=.8, label="raw context"),
            self.Line2D([], [], color="tab:blue", lw=1.4, label="surviving adjacency"),
            self.Line2D(
                [], [], color="#00796b", ls="--", lw=2.6, alpha=.65,
                label="decision-updated adjacency preview",
            ),
            self.Line2D(
                [], [], color="red", ls=(0, (4, 3)), lw=3, alpha=.55,
                label="triggering high-speed edge",
            ),
            self.Line2D([], [], color="green", ls="--", lw=1.6, label="proposed bridge"),
            self.Line2D([], [], color="crimson", marker="o", lw=1.2, label="proposed block/group"),
            self.Line2D([], [], color="navy", marker="s", ls="", label="trusted anchor"),
            self.Line2D([], [], color="gold", marker="*", markeredgecolor="black", ls="", label="selected"),
            self.Line2D([], [], color=".6", alpha=.22, marker=".", ls="", label="outside scope"),
            self.Line2D([], [], color="green", marker="o", markerfacecolor="none", ls="", label="manual keep"),
            self.Line2D([], [], color="crimson", marker="x", ls="", label="rejected observation"),
            self.Line2D([], [], color="orange", marker="^", ls="", label="manual uncertain"),
            self.Line2D([], [], color="purple", marker="D", ls="", label="unresolved"),
        ]

    def _register_pick(self, artist: Any, sources: list[int] | np.ndarray) -> None:
        artist.set_picker(True)
        self._pick_sources[artist] = np.asarray(sources, dtype=np.int64)

    def _scope_colors(self, platform: str, sources: np.ndarray) -> list[tuple[float, float, float, float]]:
        platform = str(platform)
        outside_values = {"source_invalid", "outside_drogue_window", "outside_operational_segment"}
        if platform not in self._scope_status_cache:
            frame = self._frame(platform)
            self._scope_status_cache[platform] = dict(zip(
                frame.source_obs_index.astype(int), zip(
                    frame.final_position_status.astype(str),
                    (
                        frame.exact_repeat_flag.astype(bool)
                        | frame.point_auto_status.fillna("").astype(str).eq("short_interval_reject")
                    ),
                ),
            ))
        statuses = self._scope_status_cache[platform]
        return [
            (.55, .55, .55, .12)
            if statuses.get(source, ("", False))[1]
            else ((.55, .55, .55, .18)
                  if statuses.get(source, ("", False))[0] in outside_values
                  else (.30, .30, .30, .72))
            for source in map(int, sources)
        ]

    def _plot_manual_markers(self, axis: Any, rows: pd.DataFrame, native: dict[str, Any], *, map_view: bool) -> None:
        rows = rows.loc[
            ~rows.exact_repeat_flag.astype(bool)
            & ~rows.point_auto_status.fillna("").astype(str).eq("short_interval_reject")
        ]
        styles = {
            "keep": ("none", "green", "o", "manual keep"),
            "reject": ("red", "red", "x", "manual reject"),
            "uncertain": ("orange", "orange", "^", "manual uncertain"),
        }
        for decision, (face, edge, marker, label) in styles.items():
            subset = rows[rows.human_position_decision.astype(str) == decision]
            indices = [native["source_to_native"].get(int(value)) for value in subset.source_obs_index]
            indices = np.asarray([value for value in indices if value is not None], dtype=int)
            if not len(indices):
                continue
            x = native["trajectory"].lon[indices] if map_view else native["times"][indices]
            y = native["trajectory"].lat[indices] if map_view else native["source"][indices]
            style = {"color": edge} if marker == "x" else {"facecolors": face, "edgecolors": edge}
            axis.scatter(x, y, marker=marker, s=45, linewidths=1.4,
                         zorder=7, label=label, **style)

    def _plot_map_edges(self, axis: Any, pairs: list[tuple[int, int]], native: dict[str, Any], **style: Any) -> None:
        lookup = native["source_to_native"]
        for a, b in pairs:
            if a not in lookup or b not in lookup:
                continue
            positions = [lookup[a], lookup[b]]
            axis.plot(native["trajectory"].lon[positions], native["trajectory"].lat[positions], **style)

    def _draw_overview(self, event: pd.Series, group: pd.DataFrame, native: dict[str, Any]) -> None:
        axis, trajectory = self.overview_ax, native["trajectory"]
        platform = str(event.platform_code)
        if self._overview_platform != platform:
            axis.clear()
            order = native["order"]
            axis.plot(trajectory.lon[order], trajectory.lat[order], color=".72", lw=.65, zorder=1)
            self._overview_pick_artist = axis.scatter(
                trajectory.lon, trajectory.lat,
                color=self._scope_colors(platform, native["source"]), s=9, zorder=2,
            )
            self._overview_platform = platform
        if self._overview_pick_artist is not None:
            self._register_pick(self._overview_pick_artist, native["source"])
        for artist in self._overview_event_artists:
            artist.remove()
        self._overview_event_artists = []
        highlighted = group.source_obs_index.astype(int).tolist()
        positions = [native["source_to_native"].get(value) for value in highlighted]
        positions = np.asarray([value for value in positions if value is not None], dtype=int)
        if len(positions):
            self._overview_event_artists.append(axis.scatter(
                trajectory.lon[positions], trajectory.lat[positions],
                facecolors="none", edgecolors="crimson", s=34, linewidths=1.2, zorder=5,
            ))
        axis.set(title="Full native trajectory", xlabel="Longitude", ylabel="Latitude")

    def _draw_local(self, event: pd.Series, native: dict[str, Any], display: list[int]) -> None:
        axis, trajectory = self.local_ax, native["trajectory"]
        lookup = native["source_to_native"]
        positions = np.asarray([lookup[value] for value in display if value in lookup], dtype=int)
        if len(positions):
            axis.plot(trajectory.lon[positions], trajectory.lat[positions], color=".65", lw=.75, zorder=1)
            raw = axis.scatter(
                trajectory.lon[positions], trajectory.lat[positions],
                color=self._scope_colors(str(event.platform_code), native["source"][positions]),
                s=24, zorder=2,
            )
            self._register_pick(raw, native["source"][positions])
        rows = self._frame(str(event.platform_code))
        displayed = set(display)
        platform = str(event.platform_code)
        if platform not in self._surviving_edges_cache:
            self._surviving_edges_cache[platform] = [
                (int(row.surviving_predecessor_source_obs_index), int(row.source_obs_index))
                for row in rows.itertuples()
                if not pd.isna(row.surviving_predecessor_source_obs_index)
            ]
        surviving = [pair for pair in self._surviving_edges_cache[platform]
                     if pair[0] in displayed and pair[1] in displayed]
        local_flagged: list[tuple[int, int]] = []
        if self.presentation:
            local_flagged = [
                pair for pair in self.presentation.flagged_edges
                if pair[0] in displayed and pair[1] in displayed
            ]
            # Draw triggering evidence as a broad translucent dashed underlay.
            # The narrower surviving blue edge is added afterward, so coincident
            # evidence remains red-visible without concealing the original line.
            self._plot_map_edges(axis, local_flagged, native,
                                 color="red", ls=(0, (4, 3)), lw=3.0,
                                 alpha=.55, zorder=3)
        self._plot_map_edges(axis, surviving, native, color="tab:blue", lw=1.5, zorder=4)
        if self.presentation:
            if self.presentation.bridge is not None:
                a, b, _ = self.presentation.bridge
                self._plot_map_edges(axis, [(a, b)], native, color="green", ls="--", lw=1.8, zorder=5)
            proposed_positions = [
                lookup[value] for value in self.presentation.proposed_sources
                if value in lookup and value in displayed
            ]
            if proposed_positions:
                axis.plot(trajectory.lon[proposed_positions], trajectory.lat[proposed_positions],
                          color="crimson", marker="o", lw=1.2, ms=5, zorder=6)
            anchor_positions = [lookup[value] for value in self.presentation.anchor_sources if value in lookup]
            if anchor_positions:
                axis.scatter(trajectory.lon[anchor_positions], trajectory.lat[anchor_positions],
                             color="navy", marker="s", s=48, zorder=7)
            annotate = set(self.presentation.implicated_sources) | set(self.presentation.proposed_sources) \
                | set(self.presentation.anchor_sources)
            for source in annotate & displayed:
                position = lookup.get(int(source))
                if position is not None:
                    axis.annotate(str(source), (trajectory.lon[position], trajectory.lat[position]),
                                  xytext=(4, 4), textcoords="offset points", fontsize=7)
        display_rows = rows[rows.source_obs_index.astype(int).isin(display)]
        self._plot_manual_markers(axis, display_rows, native, map_view=True)
        unresolved = display_rows[display_rows.final_position_status.astype(str) == "unresolved"]
        unresolved_positions = [lookup.get(int(value)) for value in unresolved.source_obs_index]
        unresolved_positions = np.asarray([value for value in unresolved_positions if value is not None], dtype=int)
        if len(unresolved_positions):
            axis.scatter(trajectory.lon[unresolved_positions], trajectory.lat[unresolved_positions],
                         facecolors="none", edgecolors="purple", marker="D", s=42, zorder=6)
        axis.set(title="Local event map", xlabel="Longitude", ylabel="Latitude")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=.2)
        axis.legend(handles=self._legend_handles(), loc="best", fontsize=5.5, ncol=3)

    def _velocity_model(self, event: pd.Series, native: dict[str, Any],
                        display: list[int]) -> dict[str, Any]:
        platform = str(event.platform_code)
        velocity_sources = display
        if event.kind == "temporal":
            group = self._group_for(event).sort_values("time", kind="stable")
            velocity_sources = group.source_obs_index.astype(int).tolist()
        key = (platform, str(event.event_id), tuple(velocity_sources))
        if key in self._velocity_cache:
            return self._velocity_cache[key]
        lookup = native["source_to_native"]
        positions = np.asarray([lookup[value] for value in velocity_sources if value in lookup], dtype=int)
        if len(positions):
            chronological_rank = np.empty(len(native["order"]), dtype=int)
            chronological_rank[native["order"]] = np.arange(len(native["order"]))
            positions = positions[np.argsort(chronological_rank[positions], kind="stable")]
        predecessor = positions[:-1]
        current = positions[1:]
        if len(current):
            dt, distance, speed = position_edge_metrics(
                native["time"], native["trajectory"].lon, native["trajectory"].lat,
                predecessor, current,
            )
        else:
            dt = distance = speed = np.asarray([], dtype=float)
        finite = np.isfinite(speed) & ~np.isnat(native["time"][current])
        timed = np.isfinite(dt) & ~np.isnat(native["time"][current])
        nonpositive = timed & (dt <= 0)
        short = timed & (dt > 0) & (dt < self.position_config.minimum_local_dt_seconds)
        large = timed & (
            dt > self.position_config.max_local_gap_seconds
            + self.position_config.gap_tolerance_seconds
        )
        raw_incoming: dict[int, tuple[float, float, int]] = {}
        raw_outgoing: dict[int, tuple[float, float, int]] = {}
        for a, b, value, delta in zip(
            native["source"][predecessor], native["source"][current], speed, dt,
        ):
            raw_incoming[int(b)] = (float(value), float(delta), int(a))
            raw_outgoing[int(a)] = (float(value), float(delta), int(b))
        rows = self._frame(platform)
        local = rows[rows.source_obs_index.astype(int).isin(velocity_sources)].copy()
        finite_surviving = local.surviving_speed_m_s.notna() & local.time.notna()
        surviving_incoming: dict[int, tuple[float, float, int]] = {}
        surviving_outgoing: dict[int, tuple[float, float, int]] = {}
        surviving_predecessor: list[int] = []
        surviving_current: list[int] = []
        surviving_dt: list[float] = []
        surviving_speed: list[float] = []
        displayed = set(map(int, velocity_sources))
        for row in local.itertuples():
            if pd.isna(row.surviving_predecessor_source_obs_index):
                continue
            a, b = int(row.surviving_predecessor_source_obs_index), int(row.source_obs_index)
            item = (float(row.surviving_speed_m_s), float(row.surviving_dt_seconds), a)
            surviving_incoming[b] = item
            surviving_outgoing[a] = (item[0], item[1], b)
            if (a in displayed and b in displayed and a in lookup and b in lookup
                    and np.isfinite(item[0])):
                surviving_predecessor.append(lookup[a])
                surviving_current.append(lookup[b])
                surviving_dt.append(item[1])
                surviving_speed.append(item[0])
        if surviving_current:
            order = np.argsort(
                chronological_rank[np.asarray(surviving_current, dtype=int)], kind="stable",
            )
            saved_predecessor = np.asarray(surviving_predecessor, dtype=int)[order]
            saved_current = np.asarray(surviving_current, dtype=int)[order]
            saved_dt = np.asarray(surviving_dt, dtype=float)[order]
            saved_speed = np.asarray(surviving_speed, dtype=float)[order]
        else:
            saved_predecessor = saved_current = np.asarray([], dtype=int)
            saved_dt = saved_speed = np.asarray([], dtype=float)
        flagged_times: list[Any] = []
        flagged_speeds: list[float] = []
        if self.presentation:
            displayed = set(map(int, display))
            for a, b in self.presentation.flagged_edges:
                if a not in displayed or b not in displayed:
                    continue
                if a not in lookup or b not in lookup:
                    continue
                aa, bb = lookup[a], lookup[b]
                _dt, _distance, raw_speed = position_edge_metrics(
                    native["time"], native["trajectory"].lon, native["trajectory"].lat,
                    np.asarray([aa]), np.asarray([bb]),
                )
                if np.isfinite(raw_speed[0]) and not np.isnat(native["time"][bb]):
                    flagged_times.append(native["times"][bb])
                    flagged_speeds.append(float(raw_speed[0]))
        result = {
            "positions": positions, "predecessor": predecessor, "current": current,
            "dt": dt, "distance": distance, "speed": speed, "finite": finite,
            "nonpositive": nonpositive, "short": short, "large": large,
            "local": local, "finite_surviving": finite_surviving,
            "raw_incoming": raw_incoming, "raw_outgoing": raw_outgoing,
            "surviving_incoming": surviving_incoming,
            "surviving_outgoing": surviving_outgoing,
            "saved_predecessor": saved_predecessor, "saved_current": saved_current,
            "saved_dt": saved_dt, "saved_speed": saved_speed,
            "flagged_times": flagged_times, "flagged_speeds": flagged_speeds,
        }
        self._velocity_cache[key] = result
        return result

    def _draw_velocity(self, event: pd.Series, native: dict[str, Any], display: list[int]) -> None:
        axis = self.velocity_ax
        model = self._velocity_model(event, native, display)
        current = model["current"]
        displayed_times = native["times"][model["positions"]]
        displayed_times = displayed_times[~pd.isna(displayed_times)]
        if len(displayed_times):
            first, last = displayed_times.min(), displayed_times.max()
            if first == last:
                padding = pd.Timedelta(seconds=self.position_config.nominal_interval_seconds)
            else:
                padding = max(
                    (last - first) * .025,
                    pd.Timedelta(seconds=1),
                )
            axis.set_xlim(first - padding, last + padding)
        timing_styles = (
            ("short", "#d08c00", "v", .035, "short interval"),
            ("large", ".45", "^", .070, "large gap"),
            ("nonpositive", "black", "x", .105, "duplicate/nonpositive interval"),
        )
        for key, color, marker, level, label in timing_styles:
            flagged = model[key]
            if flagged.any():
                axis.scatter(
                    native["times"][current[flagged]], np.full(flagged.sum(), level),
                    transform=axis.get_xaxis_transform(), color=color, marker=marker, s=38,
                    label=label, clip_on=False, zorder=8,
                )
        axis.axhline(self.position_config.speed_threshold_m_s, color="crimson", ls=":", lw=1.2,
                     label=f"threshold ({self.position_config.speed_threshold_m_s:g} m/s)")
        if self.presentation:
            if model["flagged_times"]:
                axis.scatter(
                    model["flagged_times"], model["flagged_speeds"],
                    color="crimson", s=28, label="frozen trigger edge", zorder=6,
                )
        self._raw_incoming = model["raw_incoming"]
        self._raw_outgoing = model["raw_outgoing"]
        self._surviving_incoming = model["surviving_incoming"]
        self._surviving_outgoing = model["surviving_outgoing"]
        axis.set(
            title="Surviving velocity",
            xlabel="UTC time (speed assigned to later retained point)",
            ylabel="Edge speed (m/s)",
        )
        axis.tick_params(axis="x", rotation=18)
        axis.grid(alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        handles.insert(0, self.Line2D(
            [], [], color="steelblue", marker=".", ls="-", lw=1.3,
            label="live surviving speed",
        ))
        labels.insert(0, "live surviving speed")
        axis.legend(handles, labels, loc="best", fontsize=6, ncol=2)

    def _create_selection_artists(self) -> None:
        animated = bool(getattr(self.figure.canvas, "supports_blit", False))
        self._selected_overview_halo = self.overview_ax.scatter(
            [], [], facecolors="none", edgecolors="white", marker="o", s=260,
            linewidths=5, zorder=9, animated=animated,
        )
        self._selected_overview_ring = self.overview_ax.scatter(
            [], [], facecolors="none", edgecolors="black", marker="o", s=260,
            linewidths=1.8, zorder=10, animated=animated,
        )
        self._selected_overview = self.overview_ax.scatter(
            [], [], color="#ffd400", edgecolors="black", marker="*", s=150,
            linewidths=1.2, zorder=11, animated=animated,
        )
        self._selected_local_halo = self.local_ax.scatter(
            [], [], facecolors="none", edgecolors="white", marker="o", s=300,
            linewidths=5, zorder=9, animated=animated,
        )
        self._selected_local_ring = self.local_ax.scatter(
            [], [], facecolors="none", edgecolors="black", marker="o", s=300,
            linewidths=2, zorder=10, animated=animated,
        )
        self._selected_local = self.local_ax.scatter(
            [], [], color="#ffd400", edgecolors="black", marker="*", s=175,
            linewidths=1.2, zorder=11, animated=animated,
        )
        self._staged_reject_overview = self.overview_ax.scatter(
            [], [], color="red", marker="x", s=65, linewidths=2, zorder=8, animated=animated,
        )
        self._staged_keep_overview = self.overview_ax.scatter(
            [], [], facecolors="none", edgecolors="green", marker="o", s=75,
            linewidths=2, zorder=8, animated=animated,
        )
        self._staged_reject_local = self.local_ax.scatter(
            [], [], color="red", marker="x", s=75, linewidths=2.2, zorder=8, animated=animated,
        )
        self._staged_keep_local = self.local_ax.scatter(
            [], [], facecolors="none", edgecolors="green", marker="o", s=85,
            linewidths=2.2, zorder=8, animated=animated,
        )
        callout = {
            "xytext": (12, 12), "textcoords": "offset points", "fontsize": 8,
            "weight": "bold", "color": "black", "zorder": 12, "animated": animated,
            "bbox": {"boxstyle": "round,pad=.25", "facecolor": "#fff4a3", "edgecolor": "black"},
            "arrowprops": {"arrowstyle": "->", "color": "black", "lw": 1.1},
        }
        self._selected_overview_label = self.overview_ax.annotate("", (0, 0), **callout)
        self._selected_label = self.local_ax.annotate(
            "", (0, 0), **callout,
        )
        self._selected_time_line = self.velocity_ax.axvline(
            np.nan, color="black", lw=1.2,
            alpha=.65, visible=False, zorder=8, animated=animated,
        )
        self._live_velocity_line, = self.velocity_ax.plot(
            [], [], color="steelblue", marker=".", ms=5, ls="-", lw=1.3,
            zorder=5, animated=animated,
        )
        self._empty_velocity_text = self.velocity_ax.text(
            .5, .5, "No surviving velocity in this window",
            transform=self.velocity_ax.transAxes, ha="center", va="center",
            color=".35", fontsize=9, visible=False, zorder=6, animated=animated,
        )
        self._provisional_local_line, = self.local_ax.plot(
            [], [], color="#00796b", ls="--", lw=2.6, alpha=.65,
            zorder=3, animated=animated,
        )
        # These blue overlays are drawn after the decision-updated previews.
        # Coincident portions therefore remain blue, while newly bridged edges
        # remain visible in teal between rejected observations.
        self._original_local_overlay, = self.local_ax.plot(
            [], [], color="tab:blue", lw=1.5, zorder=4, animated=animated,
        )
        self.point_info_text = self.velocity_ax.text(
            .01, .97, "", transform=self.velocity_ax.transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": ".75", "alpha": .86, "pad": 3},
            zorder=12, animated=animated,
        )
        self.target_text.set_animated(animated)
        self.status_text.set_animated(animated)
        self._selection_artists = [
            self._provisional_local_line, self._live_velocity_line, self._empty_velocity_text,
            self._original_local_overlay,
            self._staged_keep_overview, self._staged_keep_local,
            self._selected_time_line,
            self._selected_overview_halo, self._selected_overview_ring, self._selected_overview,
            self._selected_local_halo, self._selected_local_ring, self._selected_local,
            self._staged_reject_overview, self._staged_reject_local,
            self._selected_overview_label, self._selected_label,
            self.point_info_text, self.target_text, self.status_text,
        ]

    @staticmethod
    def _speed_text(values: dict[int, tuple[float, float, int]], source: int) -> str:
        item = values.get(source)
        return _finite_text(item[0], " m/s") if item is not None else "not available"

    @staticmethod
    def _segmented_velocity_data(
        predecessor: np.ndarray, current: np.ndarray, times: pd.DatetimeIndex,
        speed: np.ndarray, visible: np.ndarray,
    ) -> tuple[pd.DatetimeIndex, np.ndarray]:
        """Build a display line without joining independent adjacency chains."""
        plotted_time: list[Any] = []
        plotted_speed: list[float] = []
        previous_current: int | None = None
        for edge_predecessor, edge_current, stamp, value, show in zip(
            predecessor, current, times, speed, visible,
        ):
            if not bool(show) or not np.isfinite(value) or pd.isna(stamp):
                previous_current = None
                continue
            if previous_current is not None and int(edge_predecessor) != previous_current:
                plotted_time.append(pd.NaT)
                plotted_speed.append(np.nan)
            plotted_time.append(stamp)
            plotted_speed.append(float(value))
            previous_current = int(edge_current)
        return pd.to_datetime(plotted_time, utc=True), np.asarray(plotted_speed, dtype=float)

    def _provisional_speed_model(self, platform: str) -> dict[str, Any]:
        """Display-only adjacency after applying the current staged decisions."""
        platform = str(platform)
        point_items = [
            item for (code, target_type, _target_id), item in self._staged.items()
            if code == platform and target_type == "observation"
        ]
        key = (platform,)
        if key in self._provisional_speed_cache:
            return self._provisional_speed_cache[key]
        empty = {
            "sources": np.asarray([], dtype=np.int64),
            "predecessor_positions": np.asarray([], dtype=int),
            "positions": np.asarray([], dtype=int),
            "time": pd.DatetimeIndex([]), "dt": np.asarray([], dtype=float),
            "speed": np.asarray([], dtype=float),
            "incoming": {}, "outgoing": {}, "rejected": [], "kept": [],
            "effective_rejected": [],
        }

        frame = self._frame(platform)
        if not point_items:
            empty["effective_rejected"] = frame.loc[
                frame.final_position_status.fillna("").astype(str).eq("rejected"),
                "source_obs_index",
            ].astype(int).tolist()
            self._provisional_speed_cache[key] = empty
            return empty
        native = self._native(platform)
        source_rows = self._source_rows(platform)
        eligible = (
            frame.valid_timestamp.to_numpy(dtype=bool)
            & frame.source_position_valid.to_numpy(dtype=bool)
            & frame.drogue_eligible.to_numpy(dtype=bool)
            & frame.deployment_eligible.to_numpy(dtype=bool)
            & ~frame.deployment_uncertain.to_numpy(dtype=bool)
            & frame.segment_status.fillna("").astype(str).eq("retained").to_numpy(dtype=bool)
        )
        status = frame.final_position_status.fillna("").astype(str).to_numpy()
        removed = eligible & (status == "rejected")
        active = eligible & ~np.isin(status, [
            "rejected", "uncertain", "source_invalid", "outside_drogue_window",
            "outside_operational_segment", "outside_deployment_scope",
        ])
        segment = frame.segment_id.fillna("").astype(str).to_numpy()
        rejected: list[int] = []
        kept: list[int] = []
        for (code, target_type, _target), item in self._staged.items():
            if code != platform or target_type != "observation":
                continue
            source = int(item["source_obs_index"])
            row_position = source_rows.get(source)
            if row_position is None or not eligible[row_position]:
                continue
            decision = _safe_text(item.get("decision"))
            active[row_position] = decision == "keep"
            removed[row_position] = decision == "reject"
            (kept if decision == "keep" else rejected).append(source)

        ordered_sources = native["source"][native["order"]].astype(np.int64)
        predecessor_positions: list[int] = []
        current_positions: list[int] = []
        previous_native: int | None = None
        previous_segment = ""
        for source in ordered_sources:
            row_position = source_rows.get(int(source))
            native_position = native["source_to_native"].get(int(source))
            if row_position is None or native_position is None:
                previous_native, previous_segment = None, ""
                continue
            if not eligible[row_position]:
                previous_native, previous_segment = None, ""
                continue
            if not active[row_position]:
                if not removed[row_position]:
                    previous_native, previous_segment = None, ""
                continue
            current_segment = segment[row_position]
            if previous_native is not None and current_segment == previous_segment and current_segment:
                predecessor_positions.append(previous_native)
                current_positions.append(native_position)
            previous_native, previous_segment = native_position, current_segment
        predecessor = np.asarray(predecessor_positions, dtype=int)
        current = np.asarray(current_positions, dtype=int)
        if len(current):
            dt, _distance, speed = position_edge_metrics(
                native["time"], native["trajectory"].lon, native["trajectory"].lat,
                predecessor, current,
            )
        else:
            dt = speed = np.asarray([], dtype=float)
        incoming: dict[int, tuple[float, float, int]] = {}
        outgoing: dict[int, tuple[float, float, int]] = {}
        for a, b, value, delta in zip(
            native["source"][predecessor], native["source"][current], speed, dt,
        ):
            incoming[int(b)] = (float(value), float(delta), int(a))
            outgoing[int(a)] = (float(value), float(delta), int(b))
        result = {
            "sources": native["source"][current].astype(np.int64),
            "predecessor_positions": predecessor, "positions": current,
            "time": native["times"][current], "dt": dt, "speed": speed,
            "incoming": incoming, "outgoing": outgoing,
            "rejected": rejected, "kept": kept,
            "effective_rejected": frame.loc[
                removed, "source_obs_index"
            ].astype(int).tolist(),
        }
        self._provisional_speed_cache[key] = result
        return result

    def _selected_point_text(self, source: int, native: dict[str, Any], position: int) -> str:
        row = self._current_row()
        stamp = "invalid time" if np.isnat(native["time"][position]) else pd.Timestamp(
            native["time"][position], tz="UTC",
        ).isoformat()
        decision = _safe_text(row.human_position_decision) if row is not None else ""
        status = _safe_text(row.final_position_status) if row is not None else "unknown"
        platform = str(self.events.iloc[self.index].platform_code)
        staged = self._staged_point_decision(platform, source)
        return (
            f"SELECTED {source} | {stamp} | status: {status} | "
            f"staged: {staged or 'none'} | saved: {decision or 'none'}\n"
            f"Raw speed in/out: {self._speed_text(self._raw_incoming, source)} / "
            f"{self._speed_text(self._raw_outgoing, source)}; surviving in/out: "
            f"{self._speed_text(self._surviving_incoming, source)} / "
            f"{self._speed_text(self._surviving_outgoing, source)}\n"
            f"Provisional speed in/out: {self._speed_text(self._provisional_incoming, source)} / "
            f"{self._speed_text(self._provisional_outgoing, source)}"
        )

    def _update_selection_artists(self) -> None:
        if not self._selection_artists:
            return
        persistent_text = {self.target_text, self.status_text}
        for artist in self._selection_artists:
            if artist not in persistent_text:
                artist.set_visible(False)
        if self.events.empty:
            return
        platform = str(self.events.iloc[self.index].platform_code)
        native = self._native(platform)
        display = set(map(int, self.display_source_indices))
        display_array = np.asarray(list(display), dtype=np.int64)
        provisional = self._provisional_speed_model(platform)
        for decision, overview_artist, local_artist in (
            ("reject", self._staged_reject_overview, self._staged_reject_local),
            ("keep", self._staged_keep_overview, self._staged_keep_local),
        ):
            sources = [
                int(item["source_obs_index"])
                for (code, target_type, _target), item in self._staged.items()
                if code == platform and target_type == "observation" and item["decision"] == decision
            ]
            overview_sources = self._display_sample(sources)
            positions = [native["source_to_native"][source] for source in overview_sources
                         if source in native["source_to_native"]]
            local_sources = (
                provisional["effective_rejected"] if decision == "reject" else sources
            )
            local_positions = [native["source_to_native"][source] for source in local_sources
                               if source in display and source in native["source_to_native"]]
            for artist, values in ((overview_artist, positions), (local_artist, local_positions)):
                if values:
                    artist.set_offsets(np.column_stack([
                        native["trajectory"].lon[values], native["trajectory"].lat[values],
                    ]))
                    artist.set_visible(True)
        self._provisional_incoming = provisional["incoming"]
        self._provisional_outgoing = provisional["outgoing"]

        predecessor = provisional["predecessor_positions"]
        current = provisional["positions"]
        original_edges = self._surviving_edges_cache.get(platform, [])
        if len(current):
            local_edge = (
                np.isin(native["source"][predecessor], display_array)
                & np.isin(native["source"][current], display_array)
            )
            if local_edge.any():
                lon = np.column_stack([
                    native["trajectory"].lon[predecessor[local_edge]],
                    native["trajectory"].lon[current[local_edge]],
                    np.full(local_edge.sum(), np.nan),
                ]).ravel()
                lat = np.column_stack([
                    native["trajectory"].lat[predecessor[local_edge]],
                    native["trajectory"].lat[current[local_edge]],
                    np.full(local_edge.sum(), np.nan),
                ]).ravel()
                self._provisional_local_line.set_data(lon, lat)
                self._provisional_local_line.set_visible(True)
        original_pairs = [
            pair for pair in original_edges if pair[0] in display and pair[1] in display
        ]
        if original_pairs:
            original_predecessor = np.asarray([
                native["source_to_native"][pair[0]] for pair in original_pairs
            ], dtype=int)
            original_current = np.asarray([
                native["source_to_native"][pair[1]] for pair in original_pairs
            ], dtype=int)
            lon = np.column_stack([
                native["trajectory"].lon[original_predecessor],
                native["trajectory"].lon[original_current],
                np.full(len(original_pairs), np.nan),
            ]).ravel()
            lat = np.column_stack([
                native["trajectory"].lat[original_predecessor],
                native["trajectory"].lat[original_current],
                np.full(len(original_pairs), np.nan),
            ]).ravel()
            self._original_local_overlay.set_data(lon, lat)
            self._original_local_overlay.set_visible(True)

        event = self.events.iloc[self.index]
        velocity = self._velocity_model(event, native, self.display_source_indices)
        point_staging = any(
            code == platform and target_type == "observation"
            for code, target_type, _target in self._staged
        )
        if point_staging:
            live_predecessor = predecessor
            live_current = current
            live_time = provisional["time"]
            live_dt = provisional["dt"]
            live_speed = provisional["speed"]
        else:
            live_predecessor = velocity["saved_predecessor"]
            live_current = velocity["saved_current"]
            live_time = native["times"][live_current]
            live_dt = velocity["saved_dt"]
            live_speed = velocity["saved_speed"]
        if len(live_current):
            usable_timing = (
                np.isfinite(live_dt)
                & (live_dt >= self.position_config.minimum_local_dt_seconds)
                & (live_dt <= self.position_config.max_local_gap_seconds
                   + self.position_config.gap_tolerance_seconds)
            )
            visible = (
                np.isin(native["source"][live_predecessor], display_array)
                & np.isin(native["source"][live_current], display_array)
                & usable_timing
            )
            live_plot_time, live_plot_speed = self._segmented_velocity_data(
                live_predecessor, live_current, live_time, live_speed, visible,
            )
            if np.isfinite(live_plot_speed).any():
                self._live_velocity_line.set_data(live_plot_time, live_plot_speed)
                self._live_velocity_line.set_visible(True)
                values = live_plot_speed[np.isfinite(live_plot_speed)]
            else:
                values = np.asarray([], dtype=float)
        else:
            values = np.asarray([], dtype=float)
        if not len(values):
            self._empty_velocity_text.set_visible(True)
        if len(values):
            lower, upper = self.velocity_ax.get_ylim()
            minimum, maximum = float(np.nanmin(values)), float(np.nanmax(values))
            span = max(upper - lower, maximum - minimum, 1.0)
            new_lower = min(lower, minimum - .05 * span) if minimum < lower else lower
            new_upper = max(upper, maximum + .05 * span) if maximum > upper else upper
            if new_lower != lower or new_upper != upper:
                self.velocity_ax.set_ylim(new_lower, new_upper)
        if self.selected_source_obs_index is None:
            return
        source = int(self.selected_source_obs_index)
        position = native["source_to_native"].get(source)
        if position is None:
            return
        trajectory = native["trajectory"]
        offset = np.asarray([[trajectory.lon[position], trajectory.lat[position]]], dtype=float)
        for artist in (
            self._selected_overview_halo, self._selected_overview_ring, self._selected_overview,
            self._selected_local_halo, self._selected_local_ring, self._selected_local,
        ):
            artist.set_offsets(offset)
            artist.set_visible(True)
        for label in (self._selected_overview_label, self._selected_label):
            label.xy = tuple(offset[0])
            label.set_text(f"SELECTED {source}")
            label.set_visible(True)
        if not np.isnat(native["time"][position]):
            stamp = native["times"][position]
            self._selected_time_line.set_xdata([stamp, stamp])
            self._selected_time_line.set_visible(True)
        self.point_info_text.set_text(self._selected_point_text(source, native, position))
        self.point_info_text.set_visible(True)

    def _blit_selection(self) -> None:
        canvas = self.figure.canvas
        if self._blit_background is None or not getattr(canvas, "supports_blit", False):
            canvas.draw_idle()
            return
        canvas.restore_region(self._blit_background)
        for artist in self._selection_artists:
            if artist.get_visible():
                self.figure.draw_artist(artist)
        canvas.blit(self.figure.bbox)

    def _capture_blit_background(self, *, redraw: bool = True) -> None:
        if not hasattr(self, "figure"):
            return
        canvas = self.figure.canvas
        if getattr(canvas, "supports_blit", False):
            if redraw:
                self._capturing_background = True
                try:
                    canvas.draw()
                finally:
                    self._capturing_background = False
                self._blit_background = canvas.copy_from_bbox(self.figure.bbox)
                self._blit_selection()
            else:
                # Let the GUI event loop render rebuilt axes. A synchronous draw
                # here makes event navigation block on the full figure.
                self._blit_background = None
                canvas.draw_idle()
        else:
            self._blit_background = None
            canvas.draw_idle()

    def on_canvas_draw(self, _event: Any) -> None:
        """Restore animated selection artists after expose, resize, or toolbar redraw."""
        canvas = self.figure.canvas
        if (self._capturing_background or not self._selection_artists
                or not getattr(canvas, "supports_blit", False)):
            return
        self._blit_background = canvas.copy_from_bbox(self.figure.bbox)
        self._blit_selection()

    def _update_text(self) -> None:
        if self._completion_screen:
            history_count = len(self._completed_positions())
            self.header_text.set_text(
                "Position QC  •  REVIEW COMPLETE  •  Pending events remaining: 0"
            )
            self.explanation_text.set_text(
                "No pending position-QC events remain. "
                + ("Press P to browse completed history."
                   if history_count else "There is no completed history to browse.")
            )
            self.target_text.set_text("Action target: none")
            self.status_text.set_text(self._status_message or self._resume_message)
            return
        if self.events.empty or self.presentation is None:
            self.header_text.set_text("Position QC — no visible review events")
            self.explanation_text.set_text("The selected mode has no review events to display.")
            self.target_text.set_text("Action target: none")
            self.status_text.set_text(self._resume_message or self._status_message)
            return
        event = self.events.iloc[self.index]
        pending_count = int(self.events.pending.astype(bool).sum())
        stale = ", stale review" if bool(event.stale) else ""
        if bool(event.pending):
            queue_text = f"Pending events remaining: {pending_count}"
        else:
            completed = self._completed_positions()
            history_position = completed.index(self.index) + 1 if self.index in completed else 1
            queue_text = f"COMPLETED HISTORY — item {history_position}/{len(completed)}"
        self.header_text.set_text(
            f"Position QC  •  {event.platform_code}  •  {self.presentation.title}  •  "
            f"{queue_text}{stale}"
        )
        self.explanation_text.set_text(self.presentation.explanation)
        self.target_text.set_text(self._action_target())
        self.status_text.set_text(self._status_message or self._resume_message)

    def draw(self, *, recenter_source: int | None = None, restore_note: bool = True) -> None:
        self.target_text.set_animated(False)
        previous_platform = self._overview_platform
        current_platform = (
            str(self.events.iloc[self.index].platform_code)
            if not self.events.empty and not self._completion_screen else None
        )
        if previous_platform != current_platform:
            self._overview_platform = None
            self._overview_pick_artist = None
            self._overview_event_artists = []
        for artist in self._selection_artists:
            if getattr(artist, "axes", None) is None:
                continue
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        for axis in (self.local_ax, self.velocity_ax):
            axis.clear()
        self._pick_sources.clear()
        self._selection_artists.clear()
        self._blit_background = None
        if self._completion_screen:
            self.overview_ax.clear()
            self.presentation = None
            self.display_source_indices = []
            self.selectable_source_indices = []
            for axis in (self.overview_ax, self.local_ax, self.velocity_ax):
                axis.set_axis_off()
            self.overview_ax.text(
                .5, .5, "Review complete", transform=self.overview_ax.transAxes,
                ha="center", va="center", fontsize=15, weight="bold",
            )
            self._update_text(); self._set_button_states(); self.figure.canvas.draw_idle()
            return
        if self.events.empty:
            self.presentation = None
            self.display_source_indices = []
            self.selectable_source_indices = []
            self.overview_ax.text(.5, .5, "No review events", ha="center", va="center")
            self._update_text(); self._set_button_states(); self.figure.canvas.draw_idle()
            return
        for axis in (self.overview_ax, self.local_ax, self.velocity_ax):
            axis.set_axis_on()
        self.index %= len(self.events)
        event, group = self._group()
        self.presentation = self._presentation(event, group, center_source=recenter_source)
        self.display_source_indices = list(self.presentation.display_sources)
        self._refresh_selectable_sources()
        platform = str(event.platform_code)
        native = self._native(platform)
        if (self.selected_source_obs_index is None
                or self.selected_source_obs_index not in native["source_to_native"]
                or self._hidden_automatic_source(
                    platform, int(self.selected_source_obs_index),
                )):
            self.selected_source_obs_index = self._default_selection(self.presentation)
        self._draw_overview(event, group, native)
        self._draw_local(event, native, self.display_source_indices)
        self._draw_velocity(event, native, self.display_source_indices)
        self._create_selection_artists()
        self._update_selection_artists()
        self._update_text()
        self._set_button_states()
        self._capture_blit_background(redraw=False)

    # ---- navigation and actions ---------------------------------------

    def select_source(self, source_index: int, *, save: bool = False) -> None:
        if self._completion_screen or self.events.empty:
            return
        platform = str(self.events.iloc[self.index].platform_code)
        if int(source_index) not in self._native(platform)["source_to_native"]:
            return
        if self._hidden_automatic_source(platform, int(source_index)):
            # These unconditional automatic removals are provenance-only context,
            # not interactive review targets.
            return
        self.selected_source_obs_index = int(source_index)
        if source_index not in self.display_source_indices:
            self.draw(recenter_source=int(source_index), restore_note=False)
        else:
            self._update_selection_artists()
            self._update_text()
            self._set_button_states()
            self._blit_selection()

    def navigate_point(self, step: int) -> None:
        if (self._completion_screen or not self.selectable_source_indices
                or self.events.iloc[self.index].kind == "temporal"):
            return
        position = self._selected_navigation_position()
        if position is None:
            target = 0 if step > 0 else len(self.selectable_source_indices) - 1
        else:
            target = min(max(position + step, 0), len(self.selectable_source_indices) - 1)
        if position != target:
            self.select_source(self.selectable_source_indices[target])

    def _change_event(self, position: int, *, save_session: bool = False) -> None:
        if self.events.empty:
            return
        self._completion_screen = False
        self.index = int(position) % len(self.events)
        event, group = self._group()
        provisional = self._presentation(event, group)
        self.selected_source_obs_index = self._default_selection(provisional)
        self.draw()
        if save_session:
            self._save_session()

    def navigate_event(self, step: int) -> None:
        if self.events.empty:
            return
        if step < 0:
            completed = self._completed_positions()
            if self._completion_screen:
                if completed:
                    self._change_event(completed[-1], save_session=False)
                return
            current = self.events.iloc[self.index]
            if bool(current.pending):
                earlier_pending = [
                    position for position in range(self.index - 1, -1, -1)
                    if bool(self.events.iloc[position].pending)
                ]
                if earlier_pending:
                    self._change_event(earlier_pending[0], save_session=False)
                elif completed:
                    self._change_event(completed[-1], save_session=False)
            else:
                older_history = [position for position in completed if position < self.index]
                if older_history:
                    self._change_event(older_history[-1], save_session=False)
            return

        if self._completion_screen:
            return
        current = self.events.iloc[self.index]
        if not bool(current.pending):
            pending = self._first_pending_index()
            if pending is None:
                self._completion_screen = True
                self.selected_source_obs_index = None
                self.draw()
                self._save_session()
            else:
                self._change_event(pending, save_session=True)
            return

        current_id = str(current.event_id)
        anchor = self._event_anchor(current)
        pending_positions = np.flatnonzero(self.events.pending.to_numpy(dtype=bool)).astype(int).tolist()
        after = [position for position in pending_positions if position > self.index]
        before = [position for position in pending_positions if position < self.index]
        preferred_ids = [
            str(self.events.iloc[position].event_id) for position in after + before
        ]
        if not self._commit_events({current_id}):
            return
        pending = np.flatnonzero(self.events.pending.to_numpy(dtype=bool)).astype(int).tolist()
        if not pending:
            self._completion_screen = True
            self.selected_source_obs_index = None
            self.draw()
            self._save_session()
            self._set_status("All pending events are complete. Press P to browse history.")
            return
        by_id = {
            str(self.events.iloc[position].event_id): position for position in pending
        }
        target = next((by_id[event_id] for event_id in preferred_ids if event_id in by_id), None)
        if target is None:
            later = [
                position for position in pending
                if self._event_anchor(self.events.iloc[position]) > anchor
            ]
            target = later[0] if later else pending[0]
        self._change_event(int(target), save_session=True)

    def on_pick(self, event: Any) -> None:
        sources = self._pick_sources.get(event.artist)
        if sources is None or not len(event.ind):
            return
        candidate_positions = np.asarray(event.ind, dtype=int)
        selected_position = int(candidate_positions[0])
        mouse = getattr(event, "mouseevent", None)
        axis = getattr(event.artist, "axes", None)
        if mouse is not None and axis in {self.overview_ax, self.local_ax}:
            platform = str(self.events.iloc[self.index].platform_code)
            native = self._native(platform)
            lookup = native["source_to_native"]
            candidates = [lookup.get(int(sources[value])) for value in candidate_positions]
            valid = [(offset, value) for offset, value in enumerate(candidates) if value is not None]
            if valid:
                pixels = axis.transData.transform(np.column_stack([
                    [native["trajectory"].lon[value] for _offset, value in valid],
                    [native["trajectory"].lat[value] for _offset, value in valid],
                ]))
                nearest = int(np.argmin(np.linalg.norm(pixels - [mouse.x, mouse.y], axis=1)))
                selected_position = int(candidate_positions[valid[nearest][0]])
        source = int(sources[selected_position])
        self.select_source(source)

    def on_map_click(self, event: Any) -> None:
        if (self._completion_screen or self.events.empty
                or event.inaxes not in {self.overview_ax, self.local_ax}):
            return
        if getattr(event, "button", 1) not in {1, None}:
            return
        platform = str(self.events.iloc[self.index].platform_code)
        native = self._native(platform)
        sources = (
            native["source"] if event.inaxes is self.overview_ax
            else np.asarray(self.display_source_indices, dtype=np.int64)
        )
        positions = np.asarray([
            native["source_to_native"][int(source)] for source in sources
            if int(source) in native["source_to_native"]
        ], dtype=int)
        if not len(positions):
            return
        finite = np.isfinite(native["trajectory"].lon[positions]) & np.isfinite(
            native["trajectory"].lat[positions],
        )
        positions, sources = positions[finite], sources[finite]
        if not len(positions):
            return
        pixels = event.inaxes.transData.transform(np.column_stack([
            native["trajectory"].lon[positions], native["trajectory"].lat[positions],
        ]))
        nearest = int(np.argmin(np.linalg.norm(pixels - [event.x, event.y], axis=1)))
        if np.linalg.norm(pixels[nearest] - [event.x, event.y]) <= 15:
            self.select_source(int(sources[nearest]))

    def _refresh_after_action(self, platform: str, event_id: str, selected: int | None,
                              old_queue_position: int) -> None:
        self._stale_platforms.add(platform)
        updated = self.recompute_platform(platform)
        self.frames[platform] = updated
        self._stale_platforms.discard(platform)
        self._platform_events.pop(platform, None)
        self._source_row_cache.pop(platform, None)
        self._surviving_edges_cache.pop(platform, None)
        self._scope_status_cache.pop(platform, None)
        self._presentation_cache = {
            key: value for key, value in self._presentation_cache.items() if key[0] != platform
        }
        self._velocity_cache = {
            key: value for key, value in self._velocity_cache.items() if key[0] != platform
        }
        self._provisional_speed_cache = {
            key: value for key, value in self._provisional_speed_cache.items() if key[0] != platform
        }
        self.events = self._events()
        matches = np.flatnonzero(self.events.event_id.astype(str).eq(event_id)) if len(self.events) else []
        same_or_containing = False
        if len(matches):
            self.index = int(matches[0])
            same_or_containing = True
        elif selected is not None:
            containing = [position for position in range(len(self.events)) if self._event_contains(position, selected)]
            self.index = containing[0] if containing else self._fallback_index(old_queue_position)
            same_or_containing = bool(containing)
        else:
            self.index = self._fallback_index(old_queue_position)
        recenter = None
        self.selected_source_obs_index = None
        if len(self.events) and selected is not None and same_or_containing:
            current_event, current_group = self._group()
            if str(current_event.platform_code) == platform:
                provisional = self._presentation(current_event, current_group, center_source=selected)
                if selected in provisional.display_sources:
                    self.selected_source_obs_index = selected
                    recenter = selected
        self._restored_draft = ""
        self.draw(recenter_source=recenter)

    def _legacy_immediate_action(self, action: str) -> None:
        if self.events.empty:
            return
        try:
            self._flush_session_save()
        except OSError as exc:
            self._set_status(f"Decision stopped because the session could not be saved: {exc}", error=True)
            return
        aliases = {
            "keep": "retain_segment" if self.events.iloc[self.index].kind == "temporal" else "keep_point",
            "reject": "exclude_segment" if self.events.iloc[self.index].kind == "temporal" else "reject_point",
            "uncertain": "uncertain_segment" if self.events.iloc[self.index].kind == "temporal" else "uncertain_point",
            "accept": "accept_auto",
        }
        action = aliases.get(action, action)
        event, group = self._group()
        first = group.iloc[0]
        platform = str(event.platform_code)
        decision_source = "accepted_auto" if action == "accept_auto" else "manual"
        note = None
        original_rows = {key: dict(value) for key, value in self.reviews.rows.items()}
        committed = False
        try:
            if event.kind == "temporal":
                if action == "accept_auto":
                    decision = _safe_text(first.temporal_auto_decision)
                else:
                    decision = {
                        "retain_segment": "retain", "exclude_segment": "exclude",
                        "uncertain_segment": "uncertain",
                        "keep_point": "retain", "reject_point": "exclude",
                        "uncertain_point": "uncertain",
                    }.get(action, "")
                if decision not in {"retain", "exclude", "uncertain"}:
                    self._set_status("This temporal event has no concrete automatic action to accept.", error=True)
                    return
                self.reviews.set_segment(
                    first, decision, decision_source=decision_source,
                    config_sha256=self.config_sha256, note=note,
                )
            else:
                point_actions = {
                    "keep_point": "keep", "reject_point": "reject", "uncertain_point": "uncertain",
                }
                group_actions = {
                    "keep_group": "keep", "reject_group": "reject", "uncertain_group": "uncertain",
                }
                if action in point_actions:
                    row = self._current_row()
                    if not self._eligible_point(row):
                        self._set_status("The selected observation is outside point-decision scope.", error=True)
                        return
                    source_indices = [int(self.selected_source_obs_index)]
                    decision = point_actions[action]
                elif action in group_actions:
                    if not self.presentation or not self.presentation.proposed_sources:
                        self._set_status("This event has no exact block/group target.", error=True)
                        return
                    source_indices = list(self.presentation.proposed_sources)
                    decision = group_actions[action]
                elif action == "accept_auto":
                    suggested = _safe_text(first.point_auto_decision) or _safe_text(first.human_auto_decision_snapshot)
                    if suggested != "reject" or not self.presentation or not self.presentation.proposed_sources:
                        self._set_status("This event has no concrete automatic point decision to accept.", error=True)
                        return
                    source_indices = list(self.presentation.proposed_sources)
                    decision = "reject"
                else:
                    return
                self.reviews.set_observations(
                    self._frame(platform), source_indices, decision,
                    decision_source=decision_source, config_sha256=self.config_sha256, note=note,
                )
            self.reviews.save()
            committed = True
            self._state_generation += 1
            current_id = str(event.event_id)
            selected = self.selected_source_obs_index
            old_position = self.index
            self._refresh_after_action(platform, current_id, selected, old_position)
            self._set_status("Review CSV saved; the affected platform was recalculated in memory.")
            self._save_session()
        except Exception as exc:
            if committed:
                # The CSV contains the action.  Clean close retries its in-memory
                # calculation before publishing the new review hash.
                self._stale_platforms.add(platform)
                message = f"Review was saved but platform refresh failed: {exc}"
            else:
                # set_observations/set_segment mutate memory before the atomic save.
                # Restore it when persistence itself fails so a later checkpoint
                # cannot publish decisions that never reached the authoritative CSV.
                self.reviews.rows = original_rows
                message = f"Review decision was not saved: {exc}"
            self._set_status(message, error=True)
            try:
                self._save_session()
            except OSError:
                pass

    def _install_updated_platform(self, platform: str, updated: pd.DataFrame) -> None:
        platform = str(platform)
        self.frames[platform] = updated
        self.frames.move_to_end(platform)
        self._platform_events[platform] = position_event_catalog(updated, self.config_sha256)
        self._known_platforms.add(platform)
        self._source_row_cache.pop(platform, None)
        self._surviving_edges_cache.pop(platform, None)
        self._scope_status_cache.pop(platform, None)
        self._presentation_cache = {
            key: value for key, value in self._presentation_cache.items() if key[0] != platform
        }
        self._velocity_cache = {
            key: value for key, value in self._velocity_cache.items() if key[0] != platform
        }
        self._provisional_speed_cache = {
            key: value for key, value in self._provisional_speed_cache.items() if key[0] != platform
        }

    def _commit_events(self, event_ids: set[str] | None = None) -> bool:
        staged = [
            (key, item) for key, item in self._staged.items()
            if event_ids is None or _safe_text(item.get("event_id")) in event_ids
        ]
        if not staged and not self._pending_refresh_platforms:
            return True
        if staged:
            original_rows = {key: dict(value) for key, value in self.reviews.rows.items()}
            try:
                observation_batches: dict[tuple[str, str], list[int]] = {}
                segment_items: list[tuple[str, str, dict[str, Any]]] = []
                for (platform, target_type, target_id), item in staged:
                    if target_type == "observation":
                        observation_batches.setdefault(
                            (platform, str(item["decision"])), [],
                        ).append(int(item["source_obs_index"]))
                    else:
                        segment_items.append((platform, target_id, item))
                for (platform, decision), source_indices in observation_batches.items():
                    self.reviews.set_observations(
                        self._frame(platform), source_indices, decision,
                        decision_source="manual", config_sha256=self.config_sha256, note=None,
                    )
                for platform, target_id, item in segment_items:
                    frame = self._frame(platform)
                    segment = frame.loc[frame.segment_fingerprint.astype(str).eq(target_id)]
                    if segment.empty:
                        raise ValueError(f"Segment is no longer available: {target_id}")
                    self.reviews.set_segment(
                            segment.iloc[0], str(item["decision"]), decision_source="manual",
                            config_sha256=self.config_sha256, note=None,
                        )
                self.reviews.save()
            except Exception as exc:
                self.reviews.rows = original_rows
                self._set_status(f"Staged decisions were not saved: {exc}", error=True)
                return False

        for key, _item in staged:
            self._staged.pop(key, None)
        self._pending_refresh_platforms.update(key[0] for key, _item in staged)
        affected = sorted(self._pending_refresh_platforms)
        try:
            for platform in affected:
                self._install_updated_platform(platform, self.recompute_platform(platform))
                self._pending_refresh_platforms.discard(platform)
            self.events = self._events()
        except Exception as exc:
            self._set_status(
                f"Review CSV was saved, but a trajectory file could not be refreshed: {exc}",
                error=True,
            )
            return False
        self._set_status(
            f"Saved {len(staged)} decision{'s' if len(staged) != 1 else ''}; "
            f"updated {len(affected)} trajectory file{'s' if len(affected) != 1 else ''}."
        )
        return True

    def action(self, action: str) -> None:
        """Stage an action transactionally and keep GUI callback errors contained."""
        staged_before = {key: dict(value) for key, value in self._staged.items()}
        platform = (
            str(self.events.iloc[self.index].platform_code)
            if not self.events.empty else ""
        )
        try:
            self._stage_action(action)
        except Exception as exc:
            # A presentation/backend failure must not leave half of a repeated
            # group staged, nor should it escape a Qt callback and close the GUI.
            self._staged = staged_before
            if platform:
                self._provisional_speed_cache = {
                    key: value for key, value in self._provisional_speed_cache.items()
                    if key[0] != platform
                }
            self._set_status(
                f"Action was rolled back safely ({type(exc).__name__}: {exc}).",
                error=True,
            )

    def _stage_action(self, action: str) -> None:
        """Stage one point or segment decision without running QC or touching disk."""
        if (self._completion_screen or self.events.empty
                or action not in {"keep_point", "reject_point", "keep", "reject"}):
            return
        event, group = self._group()
        platform = str(event.platform_code)
        if event.kind == "temporal":
            target_id = _safe_text(group.iloc[0].segment_fingerprint)
            decision = "retain" if action in {"keep_point", "keep"} else "exclude"
            key = (platform, "segment", target_id)
            self._staged[key] = {
                "event_id": str(event.event_id), "decision": decision,
                "segment_fingerprint": target_id,
            }
            self._set_status(
                f"Staged segment {decision}; N saves it and advances.", render=False,
            )
        else:
            row = self._current_row()
            if not self._eligible_point(row) or self.selected_source_obs_index is None:
                self._set_status("The selected observation is outside point-decision scope.", error=True)
                return
            if action in {"keep_point", "keep"} and not self._point_is_rejected(platform, row):
                self._set_status("Keep is available only for a currently rejected point.", error=True)
                return
            decision = "keep" if action in {"keep_point", "keep"} else "reject"
            sources = self._repeat_group_sources(platform, row)
            for source in sources:
                key = (platform, "observation", f"observation:{source}")
                self._staged[key] = {
                    "event_id": str(event.event_id), "decision": decision,
                    "source_obs_index": source,
                }
            if len(sources) > 1:
                self._set_status(
                    f"Staged {decision} for all {len(sources)} repeated-position observations; "
                    "N saves them and advances.", render=False,
                )
            else:
                self._set_status(
                    f"Staged {decision} for observation {sources[0]}; N saves it and advances.",
                    render=False,
                )
            self._provisional_speed_cache = {
                key: value for key, value in self._provisional_speed_cache.items() if key[0] != platform
            }
        self._update_selection_artists()
        self._update_text()
        self._set_button_states()
        # K/R changes several animated artists and button state together.  A
        # coalesced backend draw avoids the nested/double-blit crash seen with
        # Qt, while point navigation continues to use the fast blit path.
        self._blit_background = None
        self.figure.canvas.draw_idle()

    def on_key(self, event: Any) -> None:
        key = _safe_text(event.key).lower()
        if key == "right":
            self.navigate_point(1)
        elif key == "left":
            self.navigate_point(-1)
        elif key == "n":
            self.navigate_event(1)
        elif key == "p":
            self.navigate_event(-1)
        elif key == "k":
            self.action("keep_point")
        elif key == "r":
            self.action("reject_point")
        elif key == "q":
            self.clean_quit()

    # ---- checkpoint and close lifecycle -------------------------------

    def _ensure_current(self) -> None:
        self._commit_events(None)

    def request_checkpoint(self) -> Any:
        return self._commit_events(None)

    def checkpoint_action(self) -> None:
        if self._commit_events(None):
            self._save_session()

    def clean_quit(self) -> None:
        if self._button_closing:
            return
        self._button_closing = True
        if not self._commit_events(None):
            self._button_closing = False
            return
        try:
            self._save_session()
        except Exception as exc:
            self._button_closing = False
            self._set_status(f"Clean quit failed; the session could not be saved: {exc}", error=True)
            return
        self._close_handled = True
        self.plt.close(self.figure)

    def on_close(self, _event: Any) -> None:
        if self._close_handled:
            return
        errors: list[str] = []
        try:
            if not self._commit_events(None):
                errors.append("staged decisions")
            self._save_session()
        except Exception as exc:
            errors.append(f"session: {exc}")
        self._close_handled = True
        if errors:
            self._status_message = "Close persistence failed (" + "; ".join(errors) + ")"

    @property
    def checkpoint_complete(self) -> bool:
        return not self._staged and not self._pending_refresh_platforms

    @property
    def close_handled(self) -> bool:
        return self._close_handled

    def show(self) -> None:
        self.plt.show()
