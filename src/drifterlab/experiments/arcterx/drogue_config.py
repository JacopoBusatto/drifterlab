"""Configuration for the standalone ARCTERX drogue-loss workflow."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import math
import pandas as pd
import yaml

from drifterlab.qc.drogue import DrogueDetectionConfig


def _keys(mapping: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return mapping


@dataclass(frozen=True)
class DrogueWorkflowConfig:
    input_directory: Path
    pattern: str
    automatic_output: Path
    review_output: Path
    missing_value: float
    analysis_cutoff_margin_hours: float
    review_zoom_window: str
    detection: DrogueDetectionConfig


def load_drogue_config(path: str | Path) -> DrogueWorkflowConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _keys(data, {"experiment", "input", "output", "processing", "detection", "review"},
                 "configuration")
    if data.get("experiment") != "arcterx":
        raise ValueError("experiment must be arcterx")
    source = _keys(data.get("input", {}), {"directory", "pattern"}, "input")
    output = _keys(data.get("output", {}), {"automatic", "review"}, "output")
    processing = _keys(
        data.get("processing", {}),
        {"missing_value", "analysis_cutoff_margin_hours", "analysis_safety_margin_hours"},
        "processing",
    )
    review = _keys(data.get("review", {}), {"zoom_window"}, "review")
    detection_data = _keys(data.get("detection", {}),
                           {field.name for field in fields(DrogueDetectionConfig)}, "detection")

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve()

    pattern = source.get("pattern", "*.mat")
    if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("input.pattern must be a nonempty relative glob without '..'")
    missing = processing.get("missing_value", -999)
    if {"analysis_cutoff_margin_hours", "analysis_safety_margin_hours"} <= set(processing):
        raise ValueError(
            "Use only processing.analysis_cutoff_margin_hours; "
            "analysis_safety_margin_hours is a legacy alias"
        )
    margin = processing.get(
        "analysis_cutoff_margin_hours", processing.get("analysis_safety_margin_hours", 24)
    )
    if (isinstance(missing, bool) or not isinstance(missing, (int, float))
            or not math.isfinite(missing)):
        raise ValueError("processing.missing_value must be finite")
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or margin < 0):
        raise ValueError("processing.analysis_cutoff_margin_hours must be finite and nonnegative")
    zoom = review.get("zoom_window", "3D")
    try:
        if pd.Timedelta(zoom) <= pd.Timedelta(0):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("review.zoom_window must be a positive duration") from exc
    automatic = resolved(output.get("automatic"), "output.automatic")
    manual = resolved(output.get("review"), "output.review")
    if automatic == manual:
        raise ValueError("Automatic and manual-review outputs must be separate")
    return DrogueWorkflowConfig(
        resolved(source.get("directory"), "input.directory"), pattern, automatic, manual,
        float(missing), float(margin), str(zoom), DrogueDetectionConfig.from_dict(detection_data),
    )
