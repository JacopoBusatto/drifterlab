"""Configuration, lazy loading, drogue scope, and audit publication."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from drifterlab import __version__
from drifterlab.io.position import get_position_reader
from drifterlab.qc.drogue import ResolvedDrogueDecision, resolve_drogue_decision
from drifterlab.reconstruction import (
    PRODUCT_SPECS,
    InitialClusterRule,
    RegularProductSpec,
    validate_cluster_rule,
)
from drifterlab.workflows.drogue import load_drogue_detection

from .model import ManualTrajectoryModel
from .state import ManualQCState

MANUAL_QC_SCHEMA_VERSION = "2.2"
MANUAL_QC_ALGORITHM_VERSION = (
    "manual-position-sst-gps-reference-accepted-span-adjacent-kinematics-v4"
)


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _utc_datetime(value: Any, name: str) -> np.datetime64:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset or Z suffix")
    return timestamp.tz_convert("UTC").tz_localize(None).to_datetime64().astype(
        "datetime64[ns]"
    )


def _cluster_rule(
    values: dict[str, Any],
    *,
    name: str,
    inherited: InitialClusterRule | None = None,
) -> InitialClusterRule:
    values = _mapping(
        values,
        {
            "assignment", "maximum_start_time_difference_seconds",
            "maximum_pair_distance_m", "maximum_cluster_diameter_m",
            "maximum_members",
        },
        name,
    )
    base = inherited or InitialClusterRule(
        "infer", 21600.0, 500.0, 1000.0, 5,
    )
    maximum_members = values.get("maximum_members", base.maximum_members)
    if maximum_members is not None:
        maximum_members = _positive_integer(
            maximum_members, f"{name}.maximum_members",
        )
    rule = InitialClusterRule(
        str(values.get("assignment", base.assignment)),
        _positive(
            values.get(
                "maximum_start_time_difference_seconds",
                base.maximum_start_time_difference_seconds,
            ),
            f"{name}.maximum_start_time_difference_seconds",
        ),
        _positive(
            values.get("maximum_pair_distance_m", base.maximum_pair_distance_m),
            f"{name}.maximum_pair_distance_m",
        ),
        _positive(
            values.get(
                "maximum_cluster_diameter_m",
                base.maximum_cluster_diameter_m,
            ),
            f"{name}.maximum_cluster_diameter_m",
        ),
        maximum_members,
    )
    validate_cluster_rule(rule, name=name)
    return rule


def _array_override_id(value: Any, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _positive_integer(value, name)
    text = str(value)
    if text.startswith("array_") and text[6:].isdigit():
        return _positive_integer(int(text[6:]), name)
    raise ValueError(f"{name} keys must be positive integers or array_NNN")


def _safe_filename(platform: str) -> str:
    value = str(platform).strip()
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not value or value in {".", ".."} or Path(value).name != value
        or value[-1] in {".", " "} or value.upper() in reserved
        or any(ord(character) < 32 or character in '<>:"/\\|?*' for character in value)
    ):
        raise ValueError(f"Platform code is not a safe filename: {platform!r}")
    return value + ".parquet"


def _datetime_text(value: np.datetime64) -> str | None:
    if np.isnat(value):
        return None
    return str(np.datetime_as_string(np.datetime64(value, "ns"), unit="ns", timezone="UTC"))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManualFinalZarrConfig:
    enabled: bool
    build_when_all_platforms_reviewed: bool
    output_directory: Path | None
    anchor_time: np.datetime64 | None
    products: tuple[RegularProductSpec, ...]
    chunk_platform: int
    chunk_time: int
    maximum_adjacent_start_gap_hours: float
    cluster_distance_reference: str
    cluster_default_rule: InitialClusterRule
    cluster_array_overrides: dict[int, InitialClusterRule]

    def effective(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "build_when_all_platforms_reviewed": self.build_when_all_platforms_reviewed,
            "output_directory": (
                None if self.output_directory is None else str(self.output_directory)
            ),
            "anchor_time": (
                None if self.anchor_time is None else _datetime_text(self.anchor_time)
            ),
            "products": [asdict(item) for item in self.products],
            "chunks": {
                "platform": self.chunk_platform,
                "time": self.chunk_time,
            },
            "arrays": {
                "maximum_adjacent_start_gap_hours":
                    self.maximum_adjacent_start_gap_hours,
            },
            "initial_clustering": {
                "distance_reference": self.cluster_distance_reference,
                "defaults": asdict(self.cluster_default_rule),
                "array_overrides": {
                    f"array_{array_id:03d}": asdict(rule)
                    for array_id, rule in sorted(
                        self.cluster_array_overrides.items()
                    )
                },
            },
        }


@dataclass(frozen=True)
class FinalZarrReadiness:
    ready: bool
    complete: bool
    current_platforms: int
    total_platforms: int
    problems: tuple[str, ...]


@dataclass(frozen=True)
class FinalizedQCInput:
    platform_code: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class FinalZarrBuildPlan:
    inputs: tuple[FinalizedQCInput, ...]
    configured_platform_count: int
    product_status: str

    @property
    def complete(self) -> bool:
        return len(self.inputs) == self.configured_platform_count


@dataclass(frozen=True)
class ManualQCWorkflowConfig:
    input_reader: str
    input_directory: Path
    input_pattern: str
    input_options: dict[str, Any]
    selected_platforms: str | tuple[str, ...]
    drogue_automatic: Path
    drogue_review: Path
    stop_at: str
    state_path: Path
    output_directory: Path
    chunk_seconds: float
    maximum_accepted_span_days: float | None
    resample_frequency_seconds: float
    high_speed_warning_m_s: float | None
    zero_speed_tolerance_m_s: float | None
    cache_size: int
    final_zarr: ManualFinalZarrConfig

    def effective(self) -> dict[str, Any]:
        values = asdict(self)
        # Advisory GUI warnings do not change review scope or QC decisions and
        # therefore must not invalidate otherwise compatible finalized files.
        values.pop("high_speed_warning_m_s")
        values.pop("zero_speed_tolerance_m_s")
        for name in (
            "input_directory", "drogue_automatic", "drogue_review",
            "state_path", "output_directory",
        ):
            values[name] = str(values[name])
        values["final_zarr"] = self.final_zarr.effective()
        return values


@dataclass(frozen=True)
class ManualQCPlatform:
    platform_code: str
    source_path: Path
    source_sha256: str
    drogue: ResolvedDrogueDecision
    stop_time: np.datetime64
    drogue_signature: str
    blocked_reason: str


def load_manual_qc_config(path: str | Path) -> ManualQCWorkflowConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _mapping(
        data,
        {
            "schema_version", "input", "selection", "drogue",
            "output", "review", "final_zarr",
        },
        "configuration",
    )
    schema_version = data.get("schema_version", 2)
    if schema_version != 2:
        raise ValueError(
            f"Manual-QC configuration schema {schema_version!r} is unsupported; "
            "expected 2"
        )
    source = _mapping(
        data.get("input", {}), {"reader", "directory", "pattern", "options"}, "input",
    )
    selection = _mapping(
        data.get("selection", {}), {"platforms"}, "selection",
    )
    drogue = _mapping(
        data.get("drogue", {}), {"automatic", "review", "stop_at"}, "drogue",
    )
    output = _mapping(data.get("output", {}), {"state", "directory"}, "output")
    review = _mapping(
        data.get("review", {}),
        {
            "chunk_hours", "maximum_accepted_span_days",
            "resample_frequency_seconds", "warnings", "cache_size",
        },
        "review",
    )
    warnings = _mapping(
        review.get("warnings", {}),
        {"high_speed_m_s", "zero_speed_tolerance_m_s"},
        "review.warnings",
    )
    final = _mapping(
        data.get("final_zarr", {}),
        {
            "enabled", "build_when_all_platforms_reviewed", "output", "grid",
            "products", "arrays", "initial_clustering",
        },
        "final_zarr",
    )
    final_output = _mapping(
        final.get("output", {}), {"directory", "chunks"}, "final_zarr.output",
    )
    final_chunks = _mapping(
        final_output.get("chunks", {}), {"platform", "time"},
        "final_zarr.output.chunks",
    )
    final_grid = _mapping(
        final.get("grid", {}), {"anchor_time"},
        "final_zarr.grid",
    )
    final_arrays = _mapping(
        final.get("arrays", {}), {"maximum_adjacent_start_gap_hours"},
        "final_zarr.arrays",
    )
    final_clustering = _mapping(
        final.get("initial_clustering", {}),
        {"distance_reference", "defaults", "array_overrides"},
        "final_zarr.initial_clustering",
    )

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (source_path.parent / candidate).resolve()

    reader = source.get("reader")
    if not isinstance(reader, str) or not reader.strip():
        raise ValueError("input.reader must be a nonempty string")
    reader = reader.strip()
    get_position_reader(reader)
    pattern = source.get("pattern", "*.mat")
    if (
        not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute()
        or ".." in Path(pattern).parts
    ):
        raise ValueError("input.pattern must be a relative glob without '..'")
    options = source.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("input.options must be a mapping")
    if reader == "microsvp_mat":
        options = _mapping(options, {"missing_value"}, "input.options")
    raw_selected_platforms = selection.get("platforms", "all")
    if raw_selected_platforms == "all":
        selected_platforms: str | tuple[str, ...] = "all"
    elif isinstance(raw_selected_platforms, list) and raw_selected_platforms:
        if any(
            not isinstance(item, str) or not item.strip()
            for item in raw_selected_platforms
        ):
            raise ValueError(
                "selection.platforms must contain nonempty strings"
            )
        selected_platforms = tuple(
            item.strip() for item in raw_selected_platforms
        )
        if len(set(selected_platforms)) != len(selected_platforms):
            raise ValueError("selection.platforms contains duplicates")
    else:
        raise ValueError(
            "selection.platforms must be 'all' or a nonempty list"
        )
    stop_at = str(drogue.get("stop_at", "physical_loss"))
    if stop_at not in {"physical_loss", "analysis_cutoff"}:
        raise ValueError("drogue.stop_at must be physical_loss or analysis_cutoff")
    chunk_seconds = _positive(review.get("chunk_hours", 1), "review.chunk_hours") * 3600
    raw_maximum_span = review.get("maximum_accepted_span_days")
    maximum_accepted_span_days = (
        None if raw_maximum_span is None else
        _positive(raw_maximum_span, "review.maximum_accepted_span_days")
    )
    frequency = _positive(
        review.get("resample_frequency_seconds", 300),
        "review.resample_frequency_seconds",
    )
    raw_high_speed = warnings.get("high_speed_m_s", 2.0)
    high_speed_warning_m_s = (
        None if raw_high_speed is None else
        _positive(raw_high_speed, "review.warnings.high_speed_m_s")
    )
    raw_zero_speed = warnings.get("zero_speed_tolerance_m_s", 0.0)
    zero_speed_tolerance_m_s = (
        None if raw_zero_speed is None else
        _nonnegative(
            raw_zero_speed,
            "review.warnings.zero_speed_tolerance_m_s",
        )
    )
    cache_size = review.get("cache_size", 2)
    if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
        raise ValueError("review.cache_size must be a positive integer")
    final_enabled = _boolean(final.get("enabled", False), "final_zarr.enabled")
    build_automatically = _boolean(
        final.get("build_when_all_platforms_reviewed", True),
        "final_zarr.build_when_all_platforms_reviewed",
    )
    final_directory = (
        resolved(final_output.get("directory"), "final_zarr.output.directory")
        if final_enabled else None
    )
    anchor_time: np.datetime64 | None = None
    products: tuple[RegularProductSpec, ...] = ()
    chunk_platform = _positive_integer(
        final_chunks.get("platform", 1), "final_zarr.output.chunks.platform",
    )
    chunk_time = _positive_integer(
        final_chunks.get("time", 2016), "final_zarr.output.chunks.time",
    )
    maximum_array_gap = _positive(
        final_arrays.get("maximum_adjacent_start_gap_hours", 24),
        "final_zarr.arrays.maximum_adjacent_start_gap_hours",
    )
    distance_reference = str(
        final_clustering.get("distance_reference", "observed_starts")
    )
    if distance_reference not in {"observed_starts", "first_common_grid"}:
        raise ValueError(
            "final_zarr.initial_clustering.distance_reference must be "
            "observed_starts or first_common_grid"
        )
    defaults = _cluster_rule(
        final_clustering.get("defaults", {}),
        name="final_zarr.initial_clustering.defaults",
    )
    raw_overrides_value = final_clustering.get("array_overrides", {})
    if not isinstance(raw_overrides_value, dict):
        raise ValueError(
            "final_zarr.initial_clustering.array_overrides must be a mapping"
        )
    raw_overrides = raw_overrides_value
    overrides: dict[int, InitialClusterRule] = {}
    for raw_array, raw_rule in raw_overrides.items():
        array_id = _array_override_id(
            raw_array, "final_zarr.initial_clustering.array_overrides",
        )
        if array_id in overrides:
            raise ValueError(
                f"Duplicate clustering override for array {array_id}"
            )
        overrides[array_id] = _cluster_rule(
            raw_rule,
            name=(
                "final_zarr.initial_clustering.array_overrides."
                f"array_{array_id:03d}"
            ),
            inherited=defaults,
        )
    if final_enabled:
        if final_grid.get("anchor_time") is None:
            raise ValueError(
                "final_zarr.grid.anchor_time is required when final Zarr is enabled"
            )
        anchor_time = _utc_datetime(
            final_grid["anchor_time"], "final_zarr.grid.anchor_time",
        )
        raw_products = final.get("products")
        if not isinstance(raw_products, list) or not raw_products:
            raise ValueError(
                "final_zarr.products must be a nonempty list when final Zarr is enabled"
            )
        parsed_products: list[RegularProductSpec] = []
        for index, raw_product in enumerate(raw_products):
            name = f"final_zarr.products[{index}]"
            product = _mapping(
                raw_product,
                {"name", "source", "cadence_minutes", "maximum_gap_minutes"},
                name,
            )
            product_name = product.get("name")
            source_kind = product.get("source")
            if not isinstance(product_name, str) or not product_name.strip():
                raise ValueError(f"{name}.name must be a nonempty string")
            if source_kind not in {"raw", "qc"}:
                raise ValueError(f"{name}.source must be raw or qc")
            parsed_products.append(RegularProductSpec(
                product_name.strip(),
                source_kind,
                _positive_integer(
                    product.get("cadence_minutes"), f"{name}.cadence_minutes",
                ),
                _positive(
                    product.get("maximum_gap_minutes"),
                    f"{name}.maximum_gap_minutes",
                ),
            ))
        products = tuple(parsed_products)
        if products != PRODUCT_SPECS:
            expected = [asdict(item) for item in PRODUCT_SPECS]
            raise ValueError(
                "final_zarr.products must match the agreed v2 product contract: "
                f"{expected}"
            )
    final_config = ManualFinalZarrConfig(
        final_enabled,
        build_automatically,
        final_directory,
        anchor_time,
        products,
        chunk_platform,
        chunk_time,
        maximum_array_gap,
        distance_reference,
        defaults,
        overrides,
    )
    result = ManualQCWorkflowConfig(
        reader,
        resolved(source.get("directory"), "input.directory"),
        pattern, dict(options), selected_platforms,
        resolved(drogue.get("automatic"), "drogue.automatic"),
        resolved(drogue.get("review"), "drogue.review"),
        stop_at,
        resolved(output.get("state"), "output.state"),
        resolved(output.get("directory"), "output.directory"),
        chunk_seconds, maximum_accepted_span_days, frequency,
        high_speed_warning_m_s, zero_speed_tolerance_m_s,
        cache_size, final_config,
    )
    if result.state_path == result.output_directory or result.output_directory.suffix:
        raise ValueError("output.state must be a file and output.directory must be a directory")
    if (
        result.final_zarr.enabled
        and result.final_zarr.output_directory is not None
        and (
            result.final_zarr.output_directory == result.output_directory
            or result.output_directory in result.final_zarr.output_directory.parents
        )
    ):
        raise ValueError("final_zarr output must be outside the per-platform QC directory")
    return result


def _review_scope_signature(
    decision: ResolvedDrogueDecision,
    stop_at: str,
    *,
    chunk_seconds: float,
    maximum_accepted_span_days: float | None,
) -> str:
    payload = {
        "status": decision.final_drogue_status,
        "physical_loss": _datetime_text(decision.final_drogue_loss_time),
        "analysis_cutoff": _datetime_text(decision.analysis_cutoff_time),
        "decision_source": decision.decision_source,
        "stop_at": stop_at,
        "chunk_seconds": chunk_seconds,
        "maximum_accepted_span_days": maximum_accepted_span_days,
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ManualQCWorkflow:
    """Lazy independent manual-QC workflow shared by the GUI and tests."""

    def __init__(self, config_path: str | Path):
        # Imported lazily after workflows.drogue to avoid the legacy eager
        # workflows/review package initializers forming a circular import.
        from drifterlab.review.drogue import DrogueLossReviews

        self.config_path = Path(config_path).resolve()
        self.config = load_manual_qc_config(self.config_path)
        if not self.config.input_directory.is_dir():
            raise ValueError(f"Input directory does not exist: {self.config.input_directory}")
        automatic = load_drogue_detection(self.config.drogue_automatic)
        reviews = DrogueLossReviews(self.config.drogue_review)
        reviews.validate_against_automatic(automatic)
        automatic_platform_codes = automatic.platform_code.astype(str)
        if self.config.selected_platforms == "all":
            selected_automatic = automatic
        else:
            available_codes = set(automatic_platform_codes)
            unknown_codes = sorted(
                set(self.config.selected_platforms) - available_codes
            )
            if unknown_codes:
                raise ValueError(
                    "Selected platforms are absent from the drogue product: "
                    f"{unknown_codes}"
                )

            selected_codes = set(self.config.selected_platforms)
            selected_automatic = automatic.loc[
                automatic_platform_codes.isin(selected_codes)
            ]
        files = {
            path.name: path.resolve()
            for path in self.config.input_directory.glob(self.config.input_pattern)
            if path.is_file()
        }
        automatic_filenames = automatic.source_filename.astype(str)
        if automatic_filenames.duplicated().any():
            duplicates = sorted(automatic_filenames[automatic_filenames.duplicated()].unique())
            raise ValueError(f"Drogue product contains duplicate source filenames: {duplicates}")
        extra_sources = set(files) - set(automatic_filenames)
        if extra_sources:
            raise ValueError(
                "Raw trajectories are missing from the drogue product: "
                f"{sorted(extra_sources)}"
            )
        platforms: list[ManualQCPlatform] = []
        for row in selected_automatic.sort_values(
            "platform_code", kind="stable",
        ).itertuples():
            code = str(row.platform_code)
            filename = str(row.source_filename)
            if filename not in files:
                raise ValueError(
                    f"Raw source for platform {code} is absent from the configured input: {filename}"
                )
            decision = resolve_drogue_decision(
                row, reviews.rows.get(code),
                default_margin_hours=float(row.default_analysis_cutoff_margin_hours),
            )
            if decision.final_drogue_status == "lost":
                stop_time = (
                    decision.final_drogue_loss_time
                    if self.config.stop_at == "physical_loss"
                    else decision.analysis_cutoff_time
                )
                blocked = ""
            elif decision.final_drogue_status == "not_lost":
                stop_time = np.datetime64("NaT", "ns")
                blocked = ""
            else:
                stop_time = np.datetime64("NaT", "ns")
                blocked = "Drogue status is uncertain; the manual-QC endpoint is unresolved."
            platforms.append(ManualQCPlatform(
                code, files[filename], str(row.source_sha256), decision,
                np.datetime64(stop_time, "ns"),
                _review_scope_signature(
                    decision, self.config.stop_at,
                    chunk_seconds=self.config.chunk_seconds,
                    maximum_accepted_span_days=self.config.maximum_accepted_span_days,
                ),
                blocked,
            ))
        if not platforms:
            raise ValueError("The drogue product contains no platforms")
        self.platforms = tuple(platforms)
        self.platform_by_code = {item.platform_code: item for item in platforms}
        self.state = ManualQCState(self.config.state_path)
        self.reader = get_position_reader(self.config.input_reader)
        self.models: OrderedDict[str, ManualTrajectoryModel] = OrderedDict()
        self.review_scope_changed: set[str] = set()

    def close(self) -> None:
        self.state.close()

    def load_model(self, platform_code: str) -> ManualTrajectoryModel:
        code = str(platform_code)
        if code in self.models:
            self.models.move_to_end(code)
            return self.models[code]
        try:
            item = self.platform_by_code[code]
        except KeyError as exc:
            raise ValueError(f"Unknown manual-QC platform: {code}") from exc
        trajectory = self.reader(item.source_path, **self.config.input_options)
        if trajectory.platform_code != code:
            raise ValueError(
                f"Raw source identity {trajectory.platform_code} disagrees with platform {code}"
            )
        if trajectory.source_sha256 != item.source_sha256:
            raise ValueError(f"Raw source hash changed for platform {code}")
        changed = self.state.register_source(code, item.source_sha256, item.drogue_signature)
        if changed:
            self.review_scope_changed.add(code)
        model = ManualTrajectoryModel(
            trajectory, chunk_seconds=self.config.chunk_seconds,
            stop_time=item.stop_time,
            hard_stop_reason=(
                self.config.stop_at
                if item.drogue.final_drogue_status == "lost" else "record_end"
            ),
            maximum_accepted_span_days=self.config.maximum_accepted_span_days,
            rejected_sources=self.state.rejected_sources(code, item.source_sha256),
            rejected_sst_sources=self.state.rejected_sst_sources(
                code, item.source_sha256,
            ),
        )
        self.models[code] = model
        while len(self.models) > self.config.cache_size:
            self.models.popitem(last=False)
        return model

    def output_path(self, platform_code: str) -> Path:
        return self.config.output_directory / _safe_filename(platform_code)

    def _current_finalized_inputs(
        self,
    ) -> tuple[tuple[FinalizedQCInput, ...], tuple[str, ...]]:
        """Return the exact current inputs and why other platforms are excluded."""
        inputs: list[FinalizedQCInput] = []
        problems: list[str] = []
        for item in self.platforms:
            if item.blocked_reason:
                problems.append(f"{item.platform_code}: {item.blocked_reason}")
                continue
            try:
                state = self.state.finalization_state(item.platform_code)
            except ValueError:
                problems.append(f"{item.platform_code}: not yet reviewed")
                continue
            if state["source_sha256"] != item.source_sha256:
                problems.append(
                    f"{item.platform_code}: raw source differs from saved review"
                )
                continue
            if state["drogue_signature"] != item.drogue_signature:
                problems.append(
                    f"{item.platform_code}: review endpoint or duration policy changed "
                    "since review"
                )
                continue
            if not bool(state["current"]):
                problems.append(
                    f"{item.platform_code}: finalized product is absent or stale"
                )
                continue
            destination = self.output_path(item.platform_code)
            if not destination.is_file():
                problems.append(
                    f"{item.platform_code}: finalized Parquet is missing"
                )
                continue
            digest = _file_sha256(destination)
            if digest != state["parquet_sha256"]:
                problems.append(
                    f"{item.platform_code}: finalized Parquet hash changed"
                )
                continue
            inputs.append(FinalizedQCInput(
                item.platform_code, destination.resolve(), digest,
            ))
        return tuple(inputs), tuple(problems)

    def final_zarr_readiness(self) -> FinalZarrReadiness:
        """Report the current finalized subset available to the v2 builder."""
        if not self.config.final_zarr.enabled:
            return FinalZarrReadiness(
                False, False, 0, len(self.platforms), ("Final Zarr is disabled",),
            )
        inputs, problems = self._current_finalized_inputs()
        current = len(inputs)
        total = len(self.platforms)
        return FinalZarrReadiness(
            current > 0,
            current == total,
            current,
            total,
            problems,
        )

    def prepare_final_zarr_build(self) -> FinalZarrBuildPlan:
        """Freeze one reproducible build membership on the GUI thread."""
        if not self.config.final_zarr.enabled:
            raise ValueError("Final Zarr is disabled in the manual-QC configuration")
        inputs, problems = self._current_finalized_inputs()
        if not inputs:
            detail = "; ".join(problems[:5]) or "no finalized platforms"
            raise ValueError(f"Zarr has no current finalized inputs: {detail}")
        total = len(self.platforms)
        return FinalZarrBuildPlan(
            inputs,
            total,
            (
                "final_complete_qc"
                if len(inputs) == total
                else "provisional_partial_qc"
            ),
        )

    def build_final_zarr(
        self,
        *,
        plan: FinalZarrBuildPlan | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Any:
        """Atomically build the grouped v2 products from a frozen finalized subset."""
        if not self.config.final_zarr.enabled:
            raise ValueError("Final Zarr is disabled in the manual-QC configuration")
        frozen = plan or self.prepare_final_zarr_build()
        if frozen.configured_platform_count != len(self.platforms):
            raise ValueError("Frozen Zarr build plan has the wrong platform count")
        expected_status = (
            "final_complete_qc"
            if frozen.complete else "provisional_partial_qc"
        )
        if frozen.product_status != expected_status:
            raise ValueError("Frozen Zarr build plan has an inconsistent status")
        final = self.config.final_zarr
        if final.output_directory is None or final.anchor_time is None:
            raise ValueError("Final Zarr output and grid anchor are not configured")

        from drifterlab.workflows.manual_qc_zarr import (
            build_manual_qc_product_store,
        )

        return build_manual_qc_product_store(
            frozen.inputs,
            final.output_directory,
            anchor=final.anchor_time,
            configured_platform_count=frozen.configured_platform_count,
            product_status=frozen.product_status,
            effective_configuration=final.effective(),
            maximum_adjacent_start_gap_hours=(
                final.maximum_adjacent_start_gap_hours
            ),
            cluster_distance_reference=final.cluster_distance_reference,
            cluster_default_rule=final.cluster_default_rule,
            cluster_array_overrides=final.cluster_array_overrides,
            chunk_platform=final.chunk_platform,
            chunk_time=final.chunk_time,
            progress=progress,
        )

    def publish(self, model: ManualTrajectoryModel) -> Path:
        item = self.platform_by_code[model.platform_code]
        if item.blocked_reason:
            raise ValueError(item.blocked_reason)
        reviewed = self.state.reviewed_chunk_keys(model.platform_code, model.source_sha256)
        reviewed &= {chunk.key for chunk in model.chunks}
        issues = model.completion_issues(reviewed)
        if issues:
            raise ValueError("Manual QC is incomplete: " + "; ".join(issues))
        chunk_by_source = {
            source: chunk.key for chunk in model.chunks for source in chunk.source_indices
        }
        review_states = self.state.chunk_states(model.platform_code, model.source_sha256)
        decisions = self.state.rejected_records(model.platform_code, model.source_sha256)
        sst_decisions = self.state.rejected_sst_records(
            model.platform_code, model.source_sha256,
        )
        rows: list[dict[str, Any]] = []
        for position, source_value in enumerate(model.sources):
            source = int(source_value)
            chunk_key = chunk_by_source.get(source, "")
            in_scope = source in model.review_position
            rejected = source in model.rejected_sources
            sst_rejected = source in model.rejected_sst_sources
            if not in_scope:
                status = "outside_review_scope"
            elif rejected:
                status = "rejected"
            else:
                status = "accepted"
            if not in_scope:
                sst_status = "outside_review_scope"
            elif not np.isfinite(model.sst[position]):
                sst_status = "missing"
            elif np.isnat(model.time[position]):
                sst_status = "invalid_time"
            elif sst_rejected:
                sst_status = "rejected"
            else:
                sst_status = "accepted"
            rows.append({
                "platform_code": model.platform_code,
                "source_sha256": model.source_sha256,
                "source_path": str(model.trajectory.source_path),
                "source_obs_index": source,
                "time": pd.Timestamp(model.time[position], tz="UTC") if not np.isnat(model.time[position]) else pd.NaT,
                "source_lon": float(model.longitude[position]),
                "source_lat": float(model.latitude[position]),
                "source_sst": float(model.sst[position]),
                "source_gps_pos_qual": float(model.gps_pos_qual[position]),
                "source_gps_num_sat": float(model.gps_num_sat[position]),
                "in_review_scope": in_scope,
                "manual_qc_status": status,
                "manual_qc_accepted": status == "accepted",
                "manual_position_qc_status": status,
                "manual_position_qc_accepted": status == "accepted",
                "manual_qc_reviewed": review_states.get(chunk_key) == "reviewed",
                "manual_qc_decision_utc": decisions.get(source, ""),
                "manual_position_qc_decision_utc": decisions.get(source, ""),
                "manual_sst_qc_status": sst_status,
                "manual_sst_qc_accepted": sst_status == "accepted",
                "manual_sst_qc_decision_utc": sst_decisions.get(source, ""),
                "review_chunk_key": chunk_key,
                "physical_drogue_loss_time": (
                    pd.Timestamp(item.drogue.final_drogue_loss_time, tz="UTC")
                    if not np.isnat(item.drogue.final_drogue_loss_time) else pd.NaT
                ),
                "analysis_cutoff_time": (
                    pd.Timestamp(item.drogue.analysis_cutoff_time, tz="UTC")
                    if not np.isnat(item.drogue.analysis_cutoff_time) else pd.NaT
                ),
                "review_stop_time": (
                    pd.Timestamp(model.review_end_time, tz="UTC")
                    if not np.isnat(model.review_end_time) else pd.NaT
                ),
                "review_end_time": (
                    pd.Timestamp(model.review_end_time, tz="UTC")
                    if not np.isnat(model.review_end_time) else pd.NaT
                ),
                "configured_drogue_stop_time": (
                    pd.Timestamp(item.stop_time, tz="UTC")
                    if not np.isnat(item.stop_time) else pd.NaT
                ),
                "raw_first_observation_time": (
                    pd.Timestamp(model.raw_first_observation_time, tz="UTC")
                    if not np.isnat(model.raw_first_observation_time) else pd.NaT
                ),
                "deployment_start_time": (
                    pd.Timestamp(model.deployment_start_time, tz="UTC")
                    if not np.isnat(model.deployment_start_time) else pd.NaT
                ),
                "requested_review_span_days": model.maximum_accepted_span_days,
                "accepted_review_span_days": model.accepted_review_span_days,
                "review_end_reason": model.review_end_reason,
            })
        frame = pd.DataFrame(rows)
        decision_revision, _resample_revision = self.state.revisions(model.platform_code)
        effective_configuration = self.config.effective()
        config_sha256 = sha256(json.dumps(
            effective_configuration, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        metadata = {
            "schema_version": MANUAL_QC_SCHEMA_VERSION,
            "algorithm_version": MANUAL_QC_ALGORITHM_VERSION,
            "drifterlab_version": __version__,
            "platform_code": model.platform_code,
            "source_sha256": model.source_sha256,
            "configuration": effective_configuration,
            "config_sha256": config_sha256,
            "review_display_warnings": {
                "high_speed_m_s": self.config.high_speed_warning_m_s,
                "zero_speed_tolerance_m_s": (
                    self.config.zero_speed_tolerance_m_s
                ),
            },
            "product_layout": "per-trajectory-manual-v1",
            "resolution_policy": "manual-point-decisions",
            "drogue": item.drogue.as_dict(),
            "drogue_stop_policy": self.config.stop_at,
            "review_scope": {
                "raw_first_observation_time": _datetime_text(
                    model.raw_first_observation_time,
                ),
                "deployment_start_time": _datetime_text(model.deployment_start_time),
                "review_end_time": _datetime_text(model.review_end_time),
                "review_end_reason": model.review_end_reason,
                "requested_review_span_days": model.maximum_accepted_span_days,
                "accepted_review_span_days": model.accepted_review_span_days,
            },
            "manual_qc_complete": True,
            "decision_revision": decision_revision,
            "reviewed_chunk_count": len(reviewed),
            "rejected_observation_count": len(model.rejected_sources),
            "rejected_sst_observation_count": len(model.rejected_sst_sources),
            "published_utc": datetime.now(timezone.utc).isoformat(),
        }
        table = pa.Table.from_pandas(frame, preserve_index=False)
        schema_metadata = dict(table.schema.metadata or {})
        schema_metadata[b"drifterlab_manual_qc"] = json.dumps(
            metadata, sort_keys=True, default=str,
        ).encode()
        table = table.replace_schema_metadata(schema_metadata)
        destination = self.output_path(model.platform_code)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.state.mark_finalized(model.platform_code, _file_sha256(destination))
        return destination
