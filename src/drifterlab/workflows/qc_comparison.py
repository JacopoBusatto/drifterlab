"""Compare manual and automatic trajectory products on their native support."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from drifterlab.trajectories.product_store import load_trajectory_product

COMPARISON_SCHEMA_VERSION = 1
COMPARISON_ALGORITHM_VERSION = "native-support-manual-automatic-comparison-v1"


@dataclass(frozen=True)
class ComparisonInput:
    label: str
    trajectory_zarr: Path
    trajectory_product: str
    analysis_root: Path | None


@dataclass(frozen=True)
class QCComparisonConfig:
    source_path: Path
    manual: ComparisonInput
    automatic: ComparisonInput
    output_root: Path


@dataclass(frozen=True)
class QCComparisonResult:
    output_directory: Path
    platform_count: int
    changed_start_count: int
    changed_array_count: int
    changed_cluster_count: int

    def format(self) -> str:
        return "\n".join((
            f"QC comparison complete: {self.platform_count} platform(s)",
            f"Different starts: {self.changed_start_count}",
            f"Different arrays: {self.changed_array_count}",
            f"Different clusters: {self.changed_cluster_count}",
            f"Output: {self.output_directory}",
        ))


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def load_qc_comparison_config(path: str | Path) -> QCComparisonConfig:
    source_path = Path(path).expanduser().resolve()
    with source_path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(
        raw, {"schema_version", "manual", "automatic", "output"},
        "configuration",
    )
    if root.get("schema_version", COMPARISON_SCHEMA_VERSION) != COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"QC comparison schema must be {COMPARISON_SCHEMA_VERSION}"
        )

    def resolved(value: Any, name: str) -> Path:
        candidate = Path(_nonempty(value, name)).expanduser()
        return (source_path.parent / candidate).resolve()

    def comparison_input(key: str) -> ComparisonInput:
        values = _mapping(
            root.get(key, {}),
            {"label", "trajectory_zarr", "trajectory_product", "analysis_root"},
            key,
        )
        raw_analysis = values.get("analysis_root")
        return ComparisonInput(
            label=_nonempty(values.get("label", key), f"{key}.label"),
            trajectory_zarr=resolved(
                values.get("trajectory_zarr"), f"{key}.trajectory_zarr",
            ),
            trajectory_product=_nonempty(
                values.get("trajectory_product"), f"{key}.trajectory_product",
            ),
            analysis_root=(
                None if raw_analysis is None else resolved(
                    raw_analysis, f"{key}.analysis_root",
                )
            ),
        )

    output = _mapping(root.get("output", {}), {"root"}, "output")
    manual = comparison_input("manual")
    automatic = comparison_input("automatic")
    output_root = resolved(output.get("root"), "output.root")
    for item in (manual, automatic):
        if output_root == item.trajectory_zarr or item.trajectory_zarr in output_root.parents:
            raise ValueError("Comparison output must be outside input Zarr stores")
    return QCComparisonConfig(source_path, manual, automatic, output_root)


def _utc_text(value: np.datetime64) -> str | None:
    if np.isnat(value):
        return None
    return np.datetime_as_string(
        np.datetime64(value, "ns"), unit="s", timezone="UTC",
    )


def _finite_summary(dataset: Any, index: int) -> dict[str, Any]:
    longitude = dataset.longitude.values[index].astype(float)
    latitude = dataset.latitude.values[index].astype(float)
    finite = np.isfinite(longitude) & np.isfinite(latitude)
    times = dataset.time.values.astype("datetime64[ns]")
    source_gap = dataset.source_gap_minutes.values[index].astype(float)
    finite_gaps = source_gap[finite & np.isfinite(source_gap)]
    if finite.any():
        indices = np.flatnonzero(finite)
        first = _utc_text(times[indices[0]])
        last = _utc_text(times[indices[-1]])
    else:
        first = last = None
    return {
        "grid_cell_count": len(times),
        "finite_position_count": int(np.count_nonzero(finite)),
        "missing_position_count": int(len(times) - np.count_nonzero(finite)),
        "first_finite_utc": first,
        "last_finite_utc": last,
        "source_gap_p50_minutes": (
            float(np.percentile(finite_gaps, 50)) if len(finite_gaps) else None
        ),
        "source_gap_p95_minutes": (
            float(np.percentile(finite_gaps, 95)) if len(finite_gaps) else None
        ),
        "source_gap_maximum_minutes": (
            float(np.max(finite_gaps)) if len(finite_gaps) else None
        ),
    }


def _platform_table(
    manual: Any,
    automatic: Any,
    manual_label: str,
    automatic_label: str,
) -> pd.DataFrame:
    manual_ids = manual.platform_id.values.astype(str).tolist()
    automatic_ids = automatic.platform_id.values.astype(str).tolist()
    all_ids = sorted(set(manual_ids) | set(automatic_ids))
    manual_index = {value: index for index, value in enumerate(manual_ids)}
    automatic_index = {value: index for index, value in enumerate(automatic_ids)}
    rows: list[dict[str, Any]] = []
    for platform in all_ids:
        row: dict[str, Any] = {
            "platform_id": platform,
            "present_manual": platform in manual_index,
            "present_automatic": platform in automatic_index,
        }
        starts: dict[str, np.datetime64] = {}
        assignments: dict[str, tuple[int, str]] = {}
        for prefix, dataset, lookup in (
            ("manual", manual, manual_index),
            ("automatic", automatic, automatic_index),
        ):
            if platform not in lookup:
                continue
            index = lookup[platform]
            start = dataset.start_time.values[index].astype("datetime64[ns]")
            array_id = int(dataset.array_id.values[index])
            cluster_id = str(dataset.cluster_id.values[index])
            starts[prefix] = start
            assignments[prefix] = (array_id, cluster_id)
            row.update({
                f"{prefix}_label": manual_label if prefix == "manual" else automatic_label,
                f"{prefix}_start_utc": _utc_text(start),
                f"{prefix}_array_id": array_id,
                f"{prefix}_cluster_id": cluster_id,
                f"{prefix}_cluster_size": int(dataset.cluster_size.values[index]),
                **{
                    f"{prefix}_{name}": value
                    for name, value in _finite_summary(dataset, index).items()
                },
            })
        both = set(starts) == {"manual", "automatic"}
        row["start_difference_seconds"] = (
            float(
                (starts["automatic"] - starts["manual"])
                / np.timedelta64(1, "s")
            ) if both else None
        )
        row["same_start"] = bool(
            both and starts["manual"] == starts["automatic"]
        )
        row["same_array"] = bool(
            set(assignments) == {"manual", "automatic"}
            and assignments["manual"][0] == assignments["automatic"][0]
        )
        row["same_cluster_id"] = bool(
            set(assignments) == {"manual", "automatic"}
            and assignments["manual"][1] == assignments["automatic"][1]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _store_hash(path: Path) -> str:
    metadata = path / ".zmetadata"
    if not metadata.is_file():
        raise ValueError(f"Zarr lacks consolidated metadata: {path}")
    return sha256(metadata.read_bytes()).hexdigest()


def _analysis_tables(item: ComparisonInput) -> pd.DataFrame | None:
    if item.analysis_root is None:
        return None
    if not item.analysis_root.is_dir():
        raise ValueError(f"Analysis root does not exist: {item.analysis_root}")
    files = sorted(item.analysis_root.glob(
        "03_epsilon/array_*/tables/epsilon_by_scale.csv"
    ))
    if not files:
        raise ValueError(
            f"No epsilon_by_scale.csv tables found under {item.analysis_root}"
        )
    tables: list[pd.DataFrame] = []
    for path in files:
        table = pd.read_csv(path)
        table.insert(0, "comparison_source", item.label)
        table.insert(1, "comparison_product", item.trajectory_product)
        table.insert(2, "comparison_table", str(path.relative_to(item.analysis_root)))
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)


def _remove_exact_directory(path: Path, *, parent: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != path.name:
        raise RuntimeError(f"Refusing cleanup outside {parent}")
    shutil.rmtree(resolved)


def _publish(temporary: Path, destination: Path, *, overwrite: bool, token: str) -> None:
    if not destination.exists():
        temporary.rename(destination)
        return
    if not overwrite:
        raise FileExistsError(f"Comparison output already exists: {destination}")
    if not destination.is_dir():
        raise ValueError(f"Comparison output is not a directory: {destination}")
    backup = destination.with_name(f".{destination.name}.{token}.backup")
    destination.rename(backup)
    try:
        temporary.rename(destination)
    except Exception:
        if not destination.exists() and backup.exists():
            backup.rename(destination)
        raise
    _remove_exact_directory(backup, parent=destination.parent)


def run_qc_comparison(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> QCComparisonResult:
    """Publish a native-support comparison without intersecting finite masks."""

    config = load_qc_comparison_config(config_path)
    report = progress or (lambda _message: None)
    report("Loading manual and automatic trajectory products")
    manual, manual_info = load_trajectory_product(
        config.manual.trajectory_zarr, config.manual.trajectory_product,
    )
    automatic, automatic_info = load_trajectory_product(
        config.automatic.trajectory_zarr, config.automatic.trajectory_product,
    )
    table = _platform_table(
        manual, automatic, config.manual.label, config.automatic.label,
    )
    changed_start = int((~table.same_start).sum())
    changed_array = int((~table.same_array).sum())
    changed_cluster = int((~table.same_cluster_id).sum())
    summary = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "algorithm_version": COMPARISON_ALGORITHM_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "support_policy": (
            "Each product is summarized on its own finite support; no common-support "
            "mask is constructed."
        ),
        "platform_count_union": len(table),
        "platform_count_manual": int(manual.sizes["platform"]),
        "platform_count_automatic": int(automatic.sizes["platform"]),
        "different_start_count": changed_start,
        "different_array_count": changed_array,
        "different_cluster_id_count": changed_cluster,
        "manual": {
            "label": config.manual.label,
            "trajectory_zarr": str(config.manual.trajectory_zarr),
            "trajectory_product": config.manual.trajectory_product,
            "product_status": manual_info.product_status,
            "metadata_sha256": _store_hash(config.manual.trajectory_zarr),
            "finite_position_count": int(np.count_nonzero(
                np.isfinite(manual.longitude.values)
                & np.isfinite(manual.latitude.values)
            )),
        },
        "automatic": {
            "label": config.automatic.label,
            "trajectory_zarr": str(config.automatic.trajectory_zarr),
            "trajectory_product": config.automatic.trajectory_product,
            "product_status": automatic_info.product_status,
            "metadata_sha256": _store_hash(config.automatic.trajectory_zarr),
            "finite_position_count": int(np.count_nonzero(
                np.isfinite(automatic.longitude.values)
                & np.isfinite(automatic.latitude.values)
            )),
        },
    }
    manual_epsilon = _analysis_tables(config.manual)
    automatic_epsilon = _analysis_tables(config.automatic)

    destination = config.output_root
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Comparison output already exists: {destination}; use --overwrite"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    temporary.mkdir()
    try:
        table.to_csv(temporary / "platform_comparison.csv", index=False)
        if manual_epsilon is not None or automatic_epsilon is not None:
            epsilon = pd.concat(
                [
                    item for item in (manual_epsilon, automatic_epsilon)
                    if item is not None
                ],
                ignore_index=True,
                sort=False,
            )
            epsilon.to_csv(temporary / "epsilon_by_scale_long.csv", index=False)
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish(temporary, destination, overwrite=overwrite, token=token)
    finally:
        if temporary.exists():
            _remove_exact_directory(temporary, parent=destination.parent)
    report(f"Published QC comparison: {destination}")
    return QCComparisonResult(
        destination, len(table), changed_start, changed_array, changed_cluster,
    )

