"""Run standalone drogue detection over raw ARCTERX MicroSVP files."""

from pathlib import Path
from typing import Callable
from uuid import uuid4

import json
import pandas as pd

from drifterlab.qc.drogue import analysis_cutoff_time, detect_drogue_loss
from .drogue_config import load_drogue_config
from .raw_drogue import read_raw_drogue_signals


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


def run_drogue_detection(config_path: str | Path, *, overwrite: bool = False,
                         mode: str | None = None,
                         progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Detect every configured raw file and atomically publish one summary table."""
    config = load_drogue_config(config_path)
    if mode is not None:
        config = config.for_mode(mode)
    report = progress or (lambda message: None)
    if not config.input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {config.input_directory}")
    files = sorted(path for path in config.input_directory.glob(config.pattern) if path.is_file())
    if not files:
        raise ValueError(f"No input files match {config.pattern!r} in {config.input_directory}")
    if config.automatic_output.exists() and not overwrite:
        raise FileExistsError(f"Automatic output already exists: {config.automatic_output}; use --overwrite")
    rows = []
    platforms: set[str] = set()
    effective = json.dumps(config.detection.as_dict(), sort_keys=True, separators=(",", ":"))
    for index, path in enumerate(files, start=1):
        signals = read_raw_drogue_signals(
            path, missing_value=config.missing_value,
            include_hull_temperature=False,
        )
        if signals.platform_code in platforms:
            raise ValueError(f"Duplicate raw platform ID: {signals.platform_code}")
        platforms.add(signals.platform_code)
        detected = detect_drogue_loss(
            signals.platform_code, signals.time, signals.ttff,
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
    table = pd.DataFrame(rows).sort_values("platform_code", kind="stable").reset_index(drop=True)
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
