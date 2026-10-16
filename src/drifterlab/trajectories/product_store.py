"""Validated read-only access to grouped trajectory-product Zarr stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class TrajectoryProductInfo:
    """Identity and grid metadata for one selected trajectory product."""

    name: str
    source_kind: str
    cadence_minutes: int
    maximum_source_gap_minutes: float
    product_status: str
    root_attributes: dict[str, Any]
    product_attributes: dict[str, Any]


def load_trajectory_product(
    path: str | Path,
    product_name: str,
) -> tuple[xr.Dataset, TrajectoryProductInfo]:
    """Load and validate one product together with root platform metadata.

    The returned dataset is detached from the Zarr handles and can therefore be
    used after this function returns.  The source store is opened read-only and
    is never modified.
    """

    store = Path(path).expanduser().resolve()
    if not store.is_dir():
        raise ValueError(f"Trajectory Zarr does not exist: {store}")
    if not isinstance(product_name, str) or not product_name.strip():
        raise ValueError("Trajectory product must be a nonempty name")
    name = product_name.strip()
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError("Trajectory product must be one direct group name")

    try:
        root_context = xr.open_zarr(store, consolidated=True, chunks=None)
        product_context = xr.open_zarr(
            store,
            group=f"products/{name}",
            consolidated=True,
            chunks=None,
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot open trajectory product {name!r} in {store}: {exc}"
        ) from exc

    with root_context as root, product_context as product:
        root_loaded = root.load()
        product_loaded = product.load()

    schema = str(root_loaded.attrs.get("schema_version", ""))
    if schema != SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported trajectory-product schema {schema!r}; expected "
            f"{SUPPORTED_PRODUCT_STORE_SCHEMA_VERSION!r}"
        )

    _require_dimensions(root_loaded, {
        "platform_id": ("platform",),
        "array_id": ("platform",),
        "cluster_id": ("platform",),
        "cluster_size": ("platform",),
        "cluster_assignment_status": ("platform",),
        "deployment_start_time": ("platform",),
    }, scope="trajectory-product root")
    _require_dimensions(product_loaded, {
        "platform_id": ("platform",),
        "time": ("time",),
        "longitude": ("platform", "time"),
        "latitude": ("platform", "time"),
        "source_gap_minutes": ("platform", "time"),
        "eastward_velocity": ("platform", "time"),
        "northward_velocity": ("platform", "time"),
        "speed": ("platform", "time"),
        "velocity_dt_seconds": ("platform", "time"),
    }, scope=f"trajectory product {name!r}")

    root_platforms = root_loaded.platform_id.values.astype(str)
    product_platforms = product_loaded.platform_id.values.astype(str)
    if len(set(root_platforms.tolist())) != len(root_platforms) or any(
        not value for value in root_platforms
    ):
        raise ValueError("Trajectory-product root platform IDs must be unique and nonempty")
    if not np.array_equal(root_platforms, product_platforms):
        raise ValueError(
            f"Trajectory product {name!r} platform IDs are not exactly aligned "
            "with the root metadata"
        )

    times = product_loaded.time.values.astype("datetime64[ns]")
    if len(times) < 2 or np.isnat(times).any():
        raise ValueError(f"Trajectory product {name!r} needs at least two finite times")
    differences = np.diff(times.astype(np.int64))
    if np.any(differences <= 0) or len(np.unique(differences)) != 1:
        raise ValueError(f"Trajectory product {name!r} time must be regular and increasing")
    cadence_minutes = differences[0] / (60 * 1_000_000_000)
    declared_cadence = _positive_integer(
        product_loaded.attrs.get("cadence_minutes"),
        f"products/{name}.cadence_minutes",
    )
    if cadence_minutes != declared_cadence:
        raise ValueError(
            f"Trajectory product {name!r} declares a {declared_cadence}-minute "
            f"cadence but its grid is {cadence_minutes:g} minutes"
        )

    declared_name = str(product_loaded.attrs.get("product_name", ""))
    if declared_name != name:
        raise ValueError(
            f"Trajectory product group {name!r} declares product_name={declared_name!r}"
        )
    source_kind = str(product_loaded.attrs.get("source_kind", ""))
    if source_kind not in {"raw", "qc"}:
        raise ValueError(
            f"Trajectory product {name!r} has invalid source_kind={source_kind!r}"
        )
    maximum_gap = _positive_number(
        product_loaded.attrs.get("maximum_source_gap_minutes"),
        f"products/{name}.maximum_source_gap_minutes",
    )

    longitude = product_loaded.longitude.values.astype(float)
    latitude = product_loaded.latitude.values.astype(float)
    if not np.array_equal(np.isfinite(longitude), np.isfinite(latitude)):
        raise ValueError(
            f"Trajectory product {name!r} longitude and latitude finite masks differ"
        )
    finite = np.isfinite(longitude)
    if np.any(finite & ((longitude < -180) | (longitude >= 180))):
        raise ValueError(f"Trajectory product {name!r} longitude lies outside [-180, 180)")
    if np.any(finite & ((latitude < -90) | (latitude > 90))):
        raise ValueError(f"Trajectory product {name!r} latitude lies outside [-90, 90]")

    # Root metadata owns the platform dimension.  The duplicate product-level
    # platform ID has already been checked and is dropped before merging.
    product_without_platform = product_loaded.drop_vars("platform_id")
    combined = xr.merge(
        [root_loaded, product_without_platform],
        compat="equals",
        join="exact",
    )
    combined["start_time"] = combined.deployment_start_time
    combined.attrs = dict(root_loaded.attrs)
    combined.attrs.update({
        "selected_trajectory_product": name,
        "selected_product_source_kind": source_kind,
        "selected_product_cadence_minutes": declared_cadence,
        "selected_product_maximum_source_gap_minutes": maximum_gap,
    })

    status = str(root_loaded.attrs.get("product_status", ""))
    info = TrajectoryProductInfo(
        name=name,
        source_kind=source_kind,
        cadence_minutes=declared_cadence,
        maximum_source_gap_minutes=maximum_gap,
        product_status=status,
        root_attributes=dict(root_loaded.attrs),
        product_attributes=dict(product_loaded.attrs),
    )
    return combined, info


def _require_dimensions(
    dataset: xr.Dataset,
    required: dict[str, tuple[str, ...]],
    *,
    scope: str,
) -> None:
    for name, dimensions in required.items():
        if name not in dataset.variables or dataset[name].dims != dimensions:
            raise ValueError(
                f"{scope} lacks {name!r} with dimensions {dimensions}"
            )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result
