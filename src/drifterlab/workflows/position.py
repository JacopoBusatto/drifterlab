"""Generic configuration, persistence, and orchestration for native-position QC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import math
from numbers import Real
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from drifterlab import __version__
from drifterlab.io.position import NativeTrajectory, get_position_reader
from drifterlab.qc.drogue import resolve_drogue_decision
from drifterlab.qc.native_position import (
    DeploymentBoundary, NativePositionConfig, PositionReviewConfig,
    TemporalSegmentConfig, run_native_position_qc,
)
from drifterlab.review.drogue import DrogueLossReviews
from drifterlab.review.position import (
    EVENT_COLUMNS, PositionReviews, file_sha256, position_event_catalog,
)
from .drogue import load_drogue_detection


POSITION_QC_SCHEMA_VERSION = "3.0"
POSITION_QC_ALGORITHM_VERSION = "native-position-qc-v2.5"
DEPLOYMENT_COLUMNS = [
    "platform_code", "deployment_time", "deployment_window_start",
    "deployment_window_end", "provenance", "note",
]


@dataclass(frozen=True)
class PositionWorkflowResult:
    """Lightweight summary of a per-trajectory position-QC product."""

    output_directory: Path
    files: tuple[Path, ...]
    observation_count: int
    platform_count: int
    unresolved_count: int
    summary_path: Path
    decision_summary: dict[str, Any]


def _safe_platform_filename(platform: str) -> str:
    value = str(platform).strip()
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if (not value or value in {".", ".."} or Path(value).name != value
            or value[-1] in {".", " "} or value.upper() in reserved
            or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in value)):
        raise ValueError(f"Platform code is not a safe filename: {platform!r}")
    return value + ".parquet"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode()).hexdigest()


def _mapping(value: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


@dataclass(frozen=True)
class PositionWorkflowConfig:
    input_reader: str
    input_directory: Path
    input_pattern: str
    input_options: dict[str, Any]
    drogue_automatic: Path
    drogue_review: Path
    deployment_metadata: Path | None
    output_directory: Path
    review_output: Path
    temporal: TemporalSegmentConfig
    position: NativePositionConfig
    review: PositionReviewConfig

    def effective(self) -> dict[str, Any]:
        result = asdict(self)
        for name in ("input_directory", "drogue_automatic", "drogue_review",
                     "deployment_metadata", "output_directory", "review_output"):
            result[name] = str(result[name]) if result[name] is not None else None
        return result


def load_position_config(path: str | Path) -> PositionWorkflowConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(data, {"input", "drogue", "deployment", "output", "temporal_segments", "position_qc", "review"}, "configuration")
    source = _mapping(data.get("input", {}), {"reader", "directory", "pattern", "options"}, "input")
    drogue = _mapping(data.get("drogue", {}), {"automatic", "review"}, "drogue")
    deployment = _mapping(data.get("deployment", {}), {"metadata"}, "deployment")
    output = _mapping(data.get("output", {}), {"directory", "review"}, "output")

    def resolved(value: Any, name: str, *, optional: bool = False) -> Path | None:
        if optional and value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (source_path.parent / candidate).resolve()

    reader = source.get("reader")
    if not isinstance(reader, str) or not reader.strip():
        raise ValueError("input.reader must be a nonempty string")
    get_position_reader(reader)
    pattern = source.get("pattern", "*.mat")
    if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("input.pattern must be a relative glob without '..'")
    options = source.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("input.options must be a mapping")
    if reader == "microsvp_mat":
        options = _mapping(options, {"missing_value"}, "input.options")
        if "missing_value" in options:
            missing_value = options["missing_value"]
            if (isinstance(missing_value, bool) or not isinstance(missing_value, (int, float))
                    or not math.isfinite(missing_value)):
                raise ValueError("input.options.missing_value must be finite")

    temporal_values = _mapping(data.get("temporal_segments", {}),
                               {field.name for field in fields(TemporalSegmentConfig)}, "temporal_segments")
    position_values = _mapping(data.get("position_qc", {}),
                               {field.name for field in fields(NativePositionConfig)}, "position_qc")
    review_values = _mapping(data.get("review", {}),
                             {field.name for field in fields(PositionReviewConfig)}, "review")
    temporal = TemporalSegmentConfig(**temporal_values)
    position = NativePositionConfig(**position_values)
    review_config = PositionReviewConfig(**review_values)
    for name, value in asdict(temporal).items():
        if name == "minimum_main_observations":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"temporal_segments.{name} must be a positive integer")
        else:
            _positive(value, f"temporal_segments.{name}")
    if not 0 < temporal.fragment_max_observation_fraction <= 1 or not 0 < temporal.fragment_max_duration_fraction <= 1:
        raise ValueError("Fragment fractions must be in (0, 1]")
    position_integer_fields = {
        "max_automatic_removal_points", "local_speed_window_points",
        "endpoint_speed_min_samples",
    }
    for name, value in asdict(position).items():
        if name in position_integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        else:
            _positive(value, f"position_qc.{name}", zero="tolerance" in name)
    if position.minimum_local_dt_seconds > position.max_local_gap_seconds:
        raise ValueError("minimum_local_dt_seconds cannot exceed max_local_gap_seconds")
    if (isinstance(review_config.context_points, bool)
            or not isinstance(review_config.context_points, int)
            or review_config.context_points < 1):
        raise ValueError("review.context_points must be a positive integer")
    if (isinstance(review_config.merge_gap_edges, bool)
            or not isinstance(review_config.merge_gap_edges, int)
            or review_config.merge_gap_edges < 0):
        raise ValueError("review.merge_gap_edges must be a nonnegative integer")
    result = PositionWorkflowConfig(
        reader.strip(), resolved(source.get("directory"), "input.directory"), pattern, dict(options),
        resolved(drogue.get("automatic"), "drogue.automatic"),
        resolved(drogue.get("review"), "drogue.review"),
        resolved(deployment.get("metadata"), "deployment.metadata", optional=True),
        resolved(output.get("directory"), "output.directory"),
        resolved(output.get("review"), "output.review"), temporal, position, review_config,
    )
    if result.output_directory == result.review_output:
        raise ValueError("Automatic and review outputs must be separate")
    dependencies = {
        result.drogue_automatic, result.drogue_review,
        *([] if result.deployment_metadata is None else [result.deployment_metadata]),
    }
    shared = {result.output_directory, result.review_output} & dependencies
    if shared:
        raise ValueError(f"Position outputs must not overwrite input products: {sorted(map(str, shared))}")
    if result.output_directory.suffix or result.review_output.suffix.lower() != ".csv":
        raise ValueError("Position output.directory must be a directory and output.review must be .csv")
    return result


def position_config_sha256(config: PositionWorkflowConfig) -> str:
    text = json.dumps(config.effective(), sort_keys=True, separators=(",", ":"))
    return sha256(text.encode()).hexdigest()


def _utc(value: Any, name: str) -> np.datetime64:
    if value is None or str(value).strip() in {"", "NaT", "nan", "None"}:
        return np.datetime64("NaT", "ns")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must contain explicit UTC")
    return stamp.tz_convert("UTC").tz_localize(None).to_datetime64().astype("datetime64[ns]")


def load_deployments(path: Path | None) -> tuple[dict[str, DeploymentBoundary], str]:
    if path is None:
        return {}, ""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(table.columns) != DEPLOYMENT_COLUMNS or table.platform_code.duplicated().any():
        raise ValueError(f"Deployment CSV must have unique rows and columns {DEPLOYMENT_COLUMNS}")
    result: dict[str, DeploymentBoundary] = {}
    for row in table.to_dict("records"):
        exact = _utc(row["deployment_time"], "deployment_time")
        start = _utc(row["deployment_window_start"], "deployment_window_start")
        end = _utc(row["deployment_window_end"], "deployment_window_end")
        has_exact = not np.isnat(exact)
        has_window = not np.isnat(start) or not np.isnat(end)
        if has_exact == has_window or (has_window and (np.isnat(start) or np.isnat(end) or start >= end)):
            raise ValueError("Each deployment row needs one exact time or a complete increasing window")
        platform = str(row["platform_code"]).strip()
        if not platform:
            raise ValueError("Deployment platform_code must not be empty")
        result[platform] = DeploymentBoundary(exact, start, end, row["provenance"], row["note"])
    return result, file_sha256(path)


def _metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = pq.read_metadata(path).metadata or {}
        value = metadata.get(b"drifterlab_position_qc")
        return json.loads(value.decode()) if value else {}
    except Exception:
        return {}


def _publish(table: pd.DataFrame, path: Path, provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        arrow = pa.Table.from_pandas(table, preserve_index=False)
        metadata = dict(arrow.schema.metadata or {})
        metadata[b"drifterlab_position_qc"] = json.dumps(provenance, sort_keys=True).encode()
        arrow = arrow.replace_schema_metadata(metadata)
        pq.write_table(arrow, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


SUMMARY_NUMERIC_FIELDS = (
    "observation_count", "final_valid_points", "final_rejected_points",
    "exact_repeat_points_removed", "short_interval_points_removed",
    "duplicate_time_points_removed", "single_point_speed_cure_events",
    "single_point_speed_cure_points", "multi_point_speed_cure_events",
    "multi_point_speed_cure_points", "automatic_geometry_points_removed",
    "human_rejected_points", "retained_at_removal_limit_events",
    "retained_at_removal_limit_points", "local_unresolved_points",
    "upstream_uncertain_points",
)


def _decision_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact per-platform counts suitable for Parquet metadata."""
    source = frame.position_decision_source.fillna("").astype(str)
    final = frame.final_position_status.fillna("").astype(str)
    reason = frame.point_auto_reason.fillna("").astype(str)
    event = frame.local_event_id.fillna("").astype(str)
    speed_reasons = {
        "unique_endpoint_speed_cure",
        "bidirectionally_confirmed_time_bounded_excursion_cure",
        "automatic_iterative_endpoint_speed_cure",
        "automatic_iterative_excursion_speed_cure",
    }
    speed_rows = frame.loc[
        source.eq("automatic_geometry") & final.eq("rejected")
        & reason.isin(speed_reasons)
    ]
    histogram: dict[str, dict[str, int]] = {}
    single_events = single_points = multi_events = multi_points = 0
    if not speed_rows.empty:
        for event_id, group in speed_rows.groupby("local_event_id", sort=False):
            if not str(event_id):
                continue
            size = int(len(group))
            bucket = histogram.setdefault(str(size), {"events": 0, "points": 0})
            bucket["events"] += 1
            bucket["points"] += size
            if size == 1:
                single_events += 1
                single_points += size
            else:
                multi_events += 1
                multi_points += size
    limit = reason.eq("automatic_keep_removal_limit_reached")
    limit_events = int(event[limit & event.ne("")].nunique())
    local_unresolved = final.isin(["unresolved", "uncertain"]) & source.eq(
        "automatic_review_required"
    )
    all_unresolved = final.isin(["unresolved", "uncertain"])
    return {
        "observation_count": int(len(frame)),
        "final_valid_points": int(final.eq("valid").sum()),
        "final_rejected_points": int(final.eq("rejected").sum()),
        "exact_repeat_points_removed": int(source.eq("automatic_repeat").sum()),
        "short_interval_points_removed": int(source.eq("automatic_short_interval").sum()),
        "duplicate_time_points_removed": int(source.eq("automatic_duplicate_time").sum()),
        "single_point_speed_cure_events": single_events,
        "single_point_speed_cure_points": single_points,
        "multi_point_speed_cure_events": multi_events,
        "multi_point_speed_cure_points": multi_points,
        "automatic_geometry_points_removed": int(
            (source.eq("automatic_geometry") & final.eq("rejected")).sum()
        ),
        "human_rejected_points": int((source.eq("human") & final.eq("rejected")).sum()),
        "retained_at_removal_limit_events": limit_events,
        "retained_at_removal_limit_points": int(limit.sum()),
        "local_unresolved_points": int(local_unresolved.sum()),
        "upstream_uncertain_points": int((all_unresolved & ~local_unresolved).sum()),
        "speed_cure_block_size_counts": histogram,
    }


def _aggregate_decision_summaries(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        name: sum(int(value.get(name, 0)) for value in values)
        for name in SUMMARY_NUMERIC_FIELDS
    }
    histogram: dict[str, dict[str, int]] = {}
    for value in values:
        for size, counts in value.get("speed_cure_block_size_counts", {}).items():
            bucket = histogram.setdefault(str(size), {"events": 0, "points": 0})
            bucket["events"] += int(counts.get("events", 0))
            bucket["points"] += int(counts.get("points", 0))
    result["speed_cure_block_size_counts"] = dict(
        sorted(histogram.items(), key=lambda item: int(item[0]))
    )
    return result


def _write_run_summary(
    output_directory: Path, platform_metadata: list[dict[str, Any]],
    resolution_policy: str,
) -> tuple[Path, dict[str, Any]]:
    summaries = [dict(value.get("decision_summary", {})) for value in platform_metadata]
    aggregate = _aggregate_decision_summaries(summaries)
    rows: list[dict[str, Any]] = []
    for metadata, summary in zip(platform_metadata, summaries):
        row = {
            "platform_code": str(metadata.get("platform_code", "")),
            "resolution_policy": str(metadata.get("resolution_policy", resolution_policy)),
            **{name: int(summary.get(name, 0)) for name in SUMMARY_NUMERIC_FIELDS},
            "speed_cure_block_size_counts_json": json.dumps(
                summary.get("speed_cure_block_size_counts", {}), sort_keys=True,
                separators=(",", ":"),
            ),
        }
        rows.append(row)
    rows.append({
        "platform_code": "ALL_PLATFORMS", "resolution_policy": resolution_policy,
        **{name: int(aggregate.get(name, 0)) for name in SUMMARY_NUMERIC_FIELDS},
        "speed_cure_block_size_counts_json": json.dumps(
            aggregate.get("speed_cure_block_size_counts", {}), sort_keys=True,
            separators=(",", ":"),
        ),
    })
    path = output_directory / "position_qc_run_summary.csv"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path, aggregate


def run_position_workflow(
    config_path: str | Path, mode: str, *, overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    reviewer_factory: Callable[..., Any] | None = None,
) -> PositionWorkflowResult:
    """Run native-position QC as independently reusable per-platform products."""
    if mode not in {"automatic", "semiautomatic", "manual"}:
        raise ValueError(f"Unsupported position workflow mode: {mode!r}")
    resolution_policy = "aggressive" if mode == "automatic" else "conservative"
    report = progress or (lambda message: None)
    config = load_position_config(config_path)
    if not config.input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {config.input_directory}")
    raw_paths = sorted(path for path in config.input_directory.glob(config.input_pattern) if path.is_file())
    if not raw_paths:
        raise ValueError(f"No files match {config.input_pattern!r} in {config.input_directory}")

    automatic = load_drogue_detection(config.drogue_automatic)
    automatic_rows = {str(row.platform_code): row for row in automatic.itertuples()}
    reader = get_position_reader(config.input_reader)
    trajectory_cache: OrderedDict[str, NativeTrajectory] = OrderedDict()
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}

    # MicroSVP production files are named by platform. Fall back to parsing only
    # exceptional filenames, retaining the generic reader contract.
    for number, path in enumerate(raw_paths, start=1):
        stem = path.stem
        if stem in automatic_rows:
            code = stem
            digest = file_sha256(path)
        else:
            trajectory = reader(path, **config.input_options)
            code, digest = trajectory.platform_code, trajectory.source_sha256
            trajectory_cache[code] = trajectory
        _safe_platform_filename(code)
        if code in source_paths:
            raise ValueError(f"Duplicate position platform: {code}")
        source_paths[code], source_hashes[code] = path.resolve(), digest
        if number % 25 == 0:
            report(f"Validated source hashes {number}/{len(raw_paths)}")

    output_names = [_safe_platform_filename(code).casefold() for code in source_paths]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Platform codes collide as case-insensitive output filenames")

    missing = set(source_paths) - set(automatic_rows)
    if missing:
        raise ValueError(f"Drogue automatic table is missing platforms: {sorted(missing)}")
    for code in source_paths:
        if str(automatic_rows[code].source_sha256) != source_hashes[code]:
            raise ValueError(f"Drogue/position source hash mismatch for {code}")

    drogue_reviews = DrogueLossReviews(config.drogue_review)
    drogue_reviews.validate_against_automatic(automatic)
    deployments, _global_deployment_hash = load_deployments(config.deployment_metadata)
    unknown_deployments = set(deployments) - set(source_paths)
    if unknown_deployments:
        raise ValueError(f"Deployment metadata contains unknown platforms: {sorted(unknown_deployments)}")
    reviews = PositionReviews(config.review_output)
    unknown_reviews = {key[0] for key in reviews.rows} - set(source_paths)
    if unknown_reviews:
        raise ValueError(f"Position review contains unknown platforms: {sorted(unknown_reviews)}")

    def trajectory_for(code: str) -> NativeTrajectory:
        code = str(code)
        if code not in trajectory_cache:
            trajectory = reader(source_paths[code], **config.input_options)
            if trajectory.platform_code != code or trajectory.source_sha256 != source_hashes[code]:
                raise ValueError(f"Native source identity changed for {code}")
            trajectory_cache[code] = trajectory
        trajectory_cache.move_to_end(code)
        while len(trajectory_cache) > 2:
            trajectory_cache.popitem(last=False)
        return trajectory_cache[code]

    reviewed_platforms = {key[0] for key in reviews.rows}
    review_indices = {
        code: set(map(int, trajectory_for(code).source_obs_index)) for code in reviewed_platforms
    }
    reviews.validate_sources(source_hashes, review_indices)
    config_hash = position_config_sha256(config)
    config.output_directory.mkdir(parents=True, exist_ok=True)

    def platform_review_hash(code: str) -> str:
        values = [
            value for key, value in sorted(reviews.rows.items()) if key[0] == code
        ]
        return _canonical_sha256(values)

    def platform_dependencies(code: str) -> dict[str, Any]:
        automatic_row = automatic_rows[code]
        automatic_value = automatic_row._asdict() if hasattr(automatic_row, "_asdict") else dict(automatic_row)
        deployment = deployments.get(code)
        return {
            "schema_version": POSITION_QC_SCHEMA_VERSION,
            "algorithm_version": POSITION_QC_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "product_layout": "per-trajectory-v2",
            "resolution_policy": resolution_policy,
            "platform_code": code,
            "config_sha256": config_hash,
            "source_path": str(source_paths[code]),
            "source_sha256": source_hashes[code],
            "platform_review_sha256": platform_review_hash(code),
            "drogue_input_sha256": _canonical_sha256({
                "automatic": automatic_value,
                "review": drogue_reviews.rows.get(code),
            }),
            "deployment_input_sha256": _canonical_sha256(
                None if deployment is None else asdict(deployment)
            ),
        }

    dependency_keys = {
        "schema_version", "algorithm_version", "product_layout", "platform_code",
        "resolution_policy",
        "config_sha256", "source_path", "source_sha256", "platform_review_sha256",
        "drogue_input_sha256", "deployment_input_sha256",
    }

    def output_path(code: str) -> Path:
        return config.output_directory / _safe_platform_filename(code)

    def compatible(code: str) -> bool:
        path = output_path(code)
        actual = _metadata(path)
        expected = platform_dependencies(code)
        if not actual or not all(actual.get(key) == expected[key] for key in dependency_keys):
            return False
        try:
            columns = set(pq.read_schema(path).names)
        except Exception:
            return False
        return {"source_lon", "source_lat"} <= columns

    def publish_platform(code: str, frame: pd.DataFrame) -> None:
        catalog = position_event_catalog(frame, config_hash)
        metadata = platform_dependencies(code)
        metadata.update({
            "effective_configuration": config.effective(),
            "source_provenance": {
                code: {"path": str(source_paths[code]), "sha256": source_hashes[code]},
            },
            "event_catalog": [
                {
                    "event_id": str(row.event_id), "kind": str(row.kind),
                    "platform_code": str(row.platform_code),
                    "pending": bool(row.pending), "stale": bool(row.stale),
                    "manual_confirmation_required": bool(
                        row.manual_confirmation_required
                    ),
                    "event_type": str(row.event_type),
                    "event_start_time_utc": str(row.event_start_time_utc),
                    "event_start_source_obs_index": int(
                        row.event_start_source_obs_index
                    ),
                }
                for row in catalog.itertuples(index=False)
            ],
            "observation_count": int(len(frame)),
            "unresolved_count": int(
                frame.final_position_status.isin(["uncertain", "unresolved"]).sum()
            ),
            "platform_qc_complete": bool(frame.platform_qc_complete.all()),
            "decision_summary": _decision_summary(frame),
        })
        _publish(frame, output_path(code), metadata)

    frame_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def remember(code: str, frame: pd.DataFrame) -> pd.DataFrame:
        frame_cache[code] = frame
        frame_cache.move_to_end(code)
        while len(frame_cache) > 2:
            frame_cache.popitem(last=False)
        return frame

    def compute(code: str) -> pd.DataFrame:
        auto = automatic_rows[code]
        resolved = resolve_drogue_decision(
            auto, drogue_reviews.rows.get(code),
            default_margin_hours=float(auto.default_analysis_cutoff_margin_hours),
        )
        frame = run_native_position_qc(
            trajectory_for(code), resolved, deployment=deployments.get(code),
            reviews=reviews.table(), temporal_config=config.temporal,
            position_config=config.position, review_config=config.review,
            resolution_policy=resolution_policy,
        )
        publish_platform(code, frame)
        return remember(code, frame)

    for number, code in enumerate(sorted(source_paths), start=1):
        if overwrite or not compatible(code):
            compute(code)
            report(f"Published position QC {number}/{len(source_paths)}: {output_path(code).name}")
        else:
            report(f"Reusing position QC {number}/{len(source_paths)}: {output_path(code).name}")

    def result_summary() -> PositionWorkflowResult:
        paths = tuple(output_path(code) for code in sorted(source_paths))
        metadata = [_metadata(path) for path in paths]
        summary_path, decision_summary = _write_run_summary(
            config.output_directory, metadata, resolution_policy,
        )
        return PositionWorkflowResult(
            config.output_directory, paths,
            sum(int(value.get("observation_count", 0)) for value in metadata),
            len(paths),
            sum(int(value.get("unresolved_count", 0)) for value in metadata),
            summary_path, decision_summary,
        )

    if mode == "automatic":
        return result_summary()

    def load_platform(code: str) -> pd.DataFrame:
        code = str(code)
        if code in frame_cache:
            frame_cache.move_to_end(code)
            return frame_cache[code]
        return remember(code, pd.read_parquet(output_path(code)))

    def recompute_platform(code: str) -> pd.DataFrame:
        return compute(str(code))

    catalogs: list[pd.DataFrame] = []
    for code in sorted(source_paths):
        records = _metadata(output_path(code)).get("event_catalog", [])
        if records:
            catalogs.append(pd.DataFrame(records, columns=EVENT_COLUMNS))
    event_catalog = (
        pd.concat(catalogs, ignore_index=True)
        if catalogs else pd.DataFrame(columns=EVENT_COLUMNS)
    )

    if reviewer_factory is None:
        from drifterlab.review.position import PositionReviewer
        reviewer_factory = PositionReviewer
    reviewer = reviewer_factory(
        None, reviews, mode=mode, config_sha256=config_hash,
        context_points=config.review.context_points,
        position_config=config.position, event_catalog=event_catalog,
        platform_loader=load_platform, trajectory_loader=trajectory_for,
        recompute_platform=recompute_platform,
    )
    reviewer.show()
    return result_summary()
