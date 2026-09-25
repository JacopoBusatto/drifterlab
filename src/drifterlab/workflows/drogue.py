"""Generic configuration and orchestration for drogue-loss processing."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import json
import math
import pandas as pd
import yaml

from drifterlab.io.drogue import get_drogue_reader
from drifterlab.qc.drogue import (
    DrogueDetectionConfig,
    analysis_cutoff_time,
    detect_drogue_loss,
)


RESOLVED_AUTOMATIC_STATUSES = {
    "clear_agreement", "clear_ttff_only", "clear_strain_only",
}
AUTOMATIC_REQUIRED_COLUMNS = {
    "platform_code", "source_path", "source_filename", "source_sha256",
    "auto_drogue_loss_time", "auto_status", "auto_source",
    "default_analysis_cutoff_time", "default_analysis_cutoff_margin_hours",
    "ttff_change_time", "ttff_status", "ttff_detail_status",
    "strain_change_time", "strain_status", "strain_level_before",
    "strain_level_after", "strain_absolute_drop", "strain_relative_drop",
    "strain_fit_improvement", "strain_n_blocks_before",
    "strain_n_blocks_after", "strain_n_valid_blocks",
}


def _keys(mapping: Any, allowed: set[str], name: str) -> dict:
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return mapping


@dataclass(frozen=True)
class DrogueWorkflowConfig:
    input_reader: str
    input_directory: Path
    pattern: str
    automatic_output: Path
    review_output: Path
    missing_value: float
    analysis_cutoff_margin_hours: float
    detection: DrogueDetectionConfig


def load_drogue_config(path: str | Path) -> DrogueWorkflowConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    data = _keys(data, {"input", "output", "processing", "detection"}, "configuration")
    source = _keys(data.get("input", {}), {"reader", "directory", "pattern"}, "input")
    output = _keys(data.get("output", {}), {"automatic", "review"}, "output")
    processing = _keys(
        data.get("processing", {}),
        {"missing_value", "analysis_cutoff_margin_hours"},
        "processing",
    )
    detection_data = _keys(
        data.get("detection", {}),
        {field.name for field in fields(DrogueDetectionConfig)},
        "detection",
    )

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve()

    reader = source.get("reader")
    if not isinstance(reader, str) or not reader.strip():
        raise ValueError("input.reader must be a nonempty string")
    reader = reader.strip()
    get_drogue_reader(reader)
    pattern = source.get("pattern", "*.mat")
    if (
        not isinstance(pattern, str)
        or not pattern
        or Path(pattern).is_absolute()
        or ".." in Path(pattern).parts
    ):
        raise ValueError("input.pattern must be a nonempty relative glob without '..'")
    missing = processing.get("missing_value", -999)
    margin = processing.get("analysis_cutoff_margin_hours", 24)
    if (
        isinstance(missing, bool)
        or not isinstance(missing, (int, float))
        or not math.isfinite(missing)
    ):
        raise ValueError("processing.missing_value must be finite")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin < 0
    ):
        raise ValueError(
            "processing.analysis_cutoff_margin_hours must be finite and nonnegative"
        )
    automatic = resolved(output.get("automatic"), "output.automatic")
    review = resolved(output.get("review"), "output.review")
    if automatic == review:
        raise ValueError("Automatic and manual-review outputs must be separate")
    return DrogueWorkflowConfig(
        reader,
        resolved(source.get("directory"), "input.directory"),
        pattern,
        automatic,
        review,
        float(missing),
        float(margin),
        DrogueDetectionConfig.from_dict(detection_data),
    )


def load_drogue_detection(path: str | Path) -> pd.DataFrame:
    """Load and validate a published automatic drogue result table."""
    source = Path(path)
    try:
        table = pd.read_parquet(source)
    except Exception as exc:
        raise ValueError(
            f"Automatic result cannot be read: {source}; use --overwrite to regenerate it"
        ) from exc
    missing = AUTOMATIC_REQUIRED_COLUMNS - set(table)
    if missing:
        raise ValueError(
            "Automatic result uses an incompatible schema; use --overwrite to "
            f"regenerate it. Missing columns: {sorted(missing)}"
        )
    if table.empty or table.platform_code.astype(str).duplicated().any():
        raise ValueError(
            "Automatic result must contain unique platform rows; use --overwrite "
            "to regenerate it"
        )
    table = table.copy()
    table["platform_code"] = table.platform_code.astype(str)
    return table


def run_drogue_detection(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Detect every configured source file and atomically publish one table."""
    config = load_drogue_config(config_path)
    report = progress or (lambda message: None)
    if not config.input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {config.input_directory}")
    files = sorted(
        path for path in config.input_directory.glob(config.pattern) if path.is_file()
    )
    if not files:
        raise ValueError(
            f"No input files match {config.pattern!r} in {config.input_directory}"
        )
    if config.automatic_output.exists() and not overwrite:
        raise FileExistsError(
            f"Automatic output already exists: {config.automatic_output}; use --overwrite"
        )
    reader = get_drogue_reader(config.input_reader)
    rows = []
    platforms: set[str] = set()
    effective = json.dumps(
        config.detection.as_dict(), sort_keys=True, separators=(",", ":")
    )
    for index, path in enumerate(files, start=1):
        signals = reader(path, missing_value=config.missing_value)
        if signals.platform_code in platforms:
            raise ValueError(f"Duplicate raw platform ID: {signals.platform_code}")
        platforms.add(signals.platform_code)
        detected = detect_drogue_loss(
            signals.platform_code,
            signals.time,
            signals.ttff,
            strain=signals.strain,
            config=config.detection,
        )
        rows.append({
            **detected.result.as_dict(),
            "default_analysis_cutoff_margin_hours": config.analysis_cutoff_margin_hours,
            "default_analysis_cutoff_time": analysis_cutoff_time(
                detected.result.auto_drogue_loss_time,
                config.analysis_cutoff_margin_hours,
            ),
            "source_filename": path.name,
            "source_path": str(path),
            "source_sha256": signals.source_sha256,
            "n_observations": len(signals.time),
            "detection_config_json": effective,
        })
        report(f"Detected {index}/{len(files)} raw trajectories")
    table = pd.DataFrame(rows).sort_values(
        "platform_code", kind="stable"
    ).reset_index(drop=True)
    for name in (
        "auto_drogue_loss_time", "ttff_change_time", "strain_change_time",
        "default_analysis_cutoff_time",
    ):
        table[name] = pd.to_datetime(table[name], errors="coerce", utc=True)
    destination = config.automatic_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        table.to_parquet(temporary, engine="pyarrow", index=False)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Automatic output already exists: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return table


def platforms_requiring_review(automatic: pd.DataFrame) -> list[str]:
    """Return unresolved/conflicting platforms in automatic-table order."""
    required = {"platform_code", "auto_status", "auto_drogue_loss_time"}
    missing = required - set(automatic)
    if missing:
        raise ValueError(f"Automatic result is missing columns: {sorted(missing)}")
    attention = (
        ~automatic.auto_status.isin(RESOLVED_AUTOMATIC_STATUSES)
        | automatic.auto_drogue_loss_time.isna()
    )
    return automatic.loc[attention, "platform_code"].astype(str).tolist()


def run_drogue_workflow(
    config_path: str | Path,
    mode: str,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    reviewer_factory: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Run generic automatic detection and optional human review."""
    report = progress or (lambda message: None)
    if mode not in {"automatic", "semiautomatic", "manual"}:
        raise ValueError(f"Unsupported drogue workflow mode: {mode!r}")
    config = load_drogue_config(config_path)
    if mode != "automatic" and config.automatic_output.exists() and not overwrite:
        report(f"Reusing automatic results: {config.automatic_output}")
        automatic = load_drogue_detection(config.automatic_output)
    else:
        if mode == "automatic" and config.automatic_output.exists() and not overwrite:
            raise FileExistsError(
                f"Automatic output already exists: {config.automatic_output}; "
                "use --overwrite"
            )
        action = "Regenerating" if config.automatic_output.exists() else "Creating"
        report(f"{action} automatic results: {config.automatic_output}")
        automatic = run_drogue_detection(
            config_path, overwrite=overwrite, progress=report,
        )
    detected = int(automatic.auto_drogue_loss_time.notna().sum())
    report(f"Automatic drogue-loss decisions: {detected}/{len(automatic)}")
    if mode == "automatic":
        return automatic

    from drifterlab.review.drogue import DrogueLossReviewer, DrogueLossReviews

    if reviewer_factory is None:
        reviewer_factory = DrogueLossReviewer
    reviews = DrogueLossReviews(
        config.review_output,
        cutoff_margin_hours=config.analysis_cutoff_margin_hours,
    )
    stale = reviews.validate_against_automatic(automatic)
    platform_codes: list[str] | None = None
    if mode == "semiautomatic":
        problematic = set(platforms_requiring_review(automatic))
        platform_codes = [
            platform for platform in automatic.platform_code.astype(str)
            if platform in stale or (platform in problematic and platform not in reviews.rows)
        ]
        if not platform_codes:
            report("No unfinished semiautomatic drogue reviews remain.")
            return automatic
        report(f"Opening reviewer for {len(platform_codes)} problematic cases.")
    else:
        pending = [
            platform for platform in automatic.platform_code.astype(str)
            if platform not in reviews.rows or platform in stale
        ]
        if not pending:
            report("No unfinished manual drogue reviews remain.")
            return automatic
        report(
            f"Opening reviewer with {len(pending)} of {len(automatic)} platforms pending."
        )
    reviewer_factory(config_path, platform_codes=platform_codes).show()
    return automatic
