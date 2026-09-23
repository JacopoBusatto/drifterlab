"""Trajectory inventory tables and file-level error reporting."""

from pathlib import Path
from uuid import uuid4

import pandas as pd

from .records import TrajectoryRecord


def inventory_row(record: TrajectoryRecord) -> dict:
    row = {"platform_code": record.platform_code, "status": "ok", "error": "", **record.metadata}
    for axis, series in record.series.items():
        row[f"n_obs_{axis}"] = len(series.time)
    n = row.get("n_obs_qc", 0)
    row["missing_position_fraction_qc"] = row.get("n_missing_positions_qc", 0) / n if n else float("nan")
    return row


def write_inventory(rows: list[dict], path: Path) -> None:
    """Write a complete inventory, including failed input files, without overwrite."""
    if path.exists():
        raise FileExistsError(f"Inventory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False)
        if path.exists():
            raise FileExistsError(f"Inventory already exists: {path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
