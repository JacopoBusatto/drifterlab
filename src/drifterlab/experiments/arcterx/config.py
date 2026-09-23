"""Validated YAML configuration; paths are relative to the configuration file."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import math
import yaml

from drifterlab.qc.flags import AuditConfig


def _keys(mapping: dict, allowed: set, name: str) -> None:
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a mapping")
    if set(mapping) - allowed:
        raise ValueError(f"Unknown {name} keys: {sorted(set(mapping) - allowed)}")


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


@dataclass(frozen=True)
class Config:
    input_directory: Path
    pattern: str
    zarr: Path
    inventory: Path
    review_table: Path | None
    missing_value: float
    drogue_buffer_hours: float
    audit: AuditConfig
    chunk_trajectory: int
    chunk_observation: int

    def effective(self) -> dict:
        result = asdict(self)
        for name in ("input_directory", "zarr", "inventory", "review_table"):
            result[name] = str(result[name]) if result[name] is not None else None
        return result


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle)
    _keys(data, {"experiment", "input", "output", "processing", "validation"}, "configuration")
    if data.get("experiment") != "arcterx":
        raise ValueError("experiment must be arcterx")
    inp, out = data.get("input", {}), data.get("output", {})
    processing, validation = data.get("processing", {}), data.get("validation", {})
    _keys(inp, {"directory", "pattern"}, "input")
    _keys(out, {"zarr", "inventory", "chunks"}, "output")
    _keys(processing, {"missing_value", "drogue_buffer_hours", "drogue_review_table"}, "processing")
    _keys(validation, set(AuditConfig.__dataclass_fields__), "validation")
    chunks = out.get("chunks", {})
    _keys(chunks, {"trajectory", "obs"}, "output.chunks")

    def resolved(value: Any, name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty path string")
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve()

    audit_values = {k: _number(v, k, zero=k == "interval_tolerance_seconds") for k, v in validation.items()}
    chunk_values = []
    for key, default in (("trajectory", 1), ("obs", 4096)):
        value = chunks.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"output.chunks.{key} must be a positive integer")
        chunk_values.append(value)
    missing = processing.get("missing_value", -999)
    if isinstance(missing, bool) or not isinstance(missing, (float, int)) or not math.isfinite(missing):
        raise ValueError("missing_value must be a finite number")
    pattern = inp.get("pattern", "*.mat")
    if not isinstance(pattern, str) or not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("input.pattern must be a nonempty relative glob without '..'")
    review = processing.get("drogue_review_table")
    result = Config(
        resolved(inp.get("directory"), "input.directory"), pattern,
        resolved(out.get("zarr"), "output.zarr"), resolved(out.get("inventory"), "output.inventory"),
        resolved(review, "processing.drogue_review_table") if review is not None else None,
        float(missing), _number(processing.get("drogue_buffer_hours", 24), "drogue_buffer_hours", zero=True),
        AuditConfig(**audit_values), *chunk_values,
    )
    if result.zarr == result.inventory or result.zarr in result.inventory.parents or result.inventory in result.zarr.parents:
        raise ValueError("Inventory and Zarr output paths must be separate")
    return result
