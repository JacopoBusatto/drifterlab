"""Grouped Zarr storage for regular trajectory products."""

from __future__ import annotations

from pathlib import Path

from numcodecs import Blosc
import numpy as np
import zarr

from .core import RegularTrajectoryProduct
from .products import PRODUCT_SPECS
from dataclasses import dataclass

PRODUCT_STORE_SCHEMA_VERSION = "2.0"

_PRODUCT_VARIABLES = {
    "longitude": {
        "standard_name": "longitude",
        "units": "degrees_east",
    },
    "latitude": {
        "standard_name": "latitude",
        "units": "degrees_north",
    },
    "source_gap_minutes": {
        "long_name": "source-observation interval bracketing each position",
        "units": "minutes",
    },
    "eastward_velocity": {
        "long_name": "forward eastward velocity",
        "units": "m s-1",
    },
    "northward_velocity": {
        "long_name": "forward northward velocity",
        "units": "m s-1",
    },
    "speed": {
        "long_name": "forward horizontal speed",
        "units": "m s-1",
    },
    "velocity_dt_seconds": {
        "long_name": "elapsed time used for forward velocity",
        "units": "seconds",
    },
}

@dataclass(frozen=True)
class PlatformProductMetadata:
    deployment_start_time: np.datetime64
    deployment_start_longitude: float
    deployment_start_latitude: float
    trajectory_end_time: np.datetime64
    review_end_time: np.datetime64
    requested_review_span_days: float
    accepted_review_span_days: float
    review_end_reason: str
    accepted_fix_count: int
    raw_fix_count: int
    source_sha256: str
    qc_input_sha256: str


@dataclass(frozen=True)
class PlatformAssignmentMetadata:
    array_id: int
    gap_from_previous_start_hours: float | None
    gap_to_next_start_hours: float | None
    cluster_id: str
    member_id: int
    cluster_size: int
    cluster_assignment_status: str
    cluster_start_spread_seconds: float
    cluster_diameter_m: float | None
    cluster_maximum_available_pair_distance_m: float | None
    cluster_observed_start_diameter_m: float | None
    cluster_first_common_grid_diameter_m: float | None
    cluster_first_common_grid_unavailable_pair_count: int
    cluster_candidate_link_count: int
    cluster_unavailable_pair_count: int

def create_product_store(
    path: str | Path,
    platforms: list[str],
    grids: dict[int, np.ndarray],
    *,
    chunk_platform: int = 1,
    chunk_time: int = 2016,
) -> zarr.Group:
    """Create an empty grouped trajectory-product store."""

    if not platforms:
        raise ValueError("At least one platform is required")
    if chunk_platform < 1 or chunk_time < 1:
        raise ValueError("Zarr chunk sizes must be positive")

    path = Path(path)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update({
        "schema_version": PRODUCT_STORE_SCHEMA_VERSION,
        "description": "Manual-QC trajectory products for downstream analysis",
        "velocity_method": "forward_difference",
        "velocity_assignment": "earlier_endpoint",
    })

    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE,
    )
    platform_width = max(1, *(len(value) for value in platforms))
    platform_values = np.asarray(
        platforms,
        dtype=f"<U{platform_width}",
    )

    root_platform = root.create_dataset(
        "platform_id",
        data=platform_values,
        chunks=(min(chunk_platform, len(platforms)),),
        compressor=compressor,
    )
    root_platform.attrs.update({
        "_ARRAY_DIMENSIONS": ["platform"],
        "cf_role": "trajectory_id",
    })

    platform_chunks = (
        min(chunk_platform, len(platforms)),
    )

    array_id = root.create_dataset(
        "array_id",
        shape=(len(platforms),),
        chunks=platform_chunks,
        dtype="i4",
        fill_value=-1,
        compressor=compressor,
    )
    array_id.attrs.update({
        "_ARRAY_DIMENSIONS": ["platform"],
        "long_name": "deployment array identifier",
    })

    cluster_id = root.create_dataset(
        "cluster_id",
        shape=(len(platforms),),
        chunks=platform_chunks,
        dtype="<U64",
        fill_value="",
        compressor=compressor,
    )
    cluster_id.attrs.update({
        "_ARRAY_DIMENSIONS": ["platform"],
        "long_name": "initial deployment cluster identifier",
    })

    assignment_status = root.create_dataset(
        "cluster_assignment_status",
        shape=(len(platforms),),
        chunks=platform_chunks,
        dtype="<U32",
        fill_value="",
        compressor=compressor,
    )
    assignment_status.attrs.update({
        "_ARRAY_DIMENSIONS": ["platform"],
        "long_name": "initial cluster assignment status",
        "flag_values": [
            "candidate_cluster",
            "algorithmic_singleton",
            "configured_single_cluster",
        ],
    })

    assignment_counts = {
        "member_id": "one-based member number within the initial cluster",
        "cluster_size": "number of platforms in the initial cluster",
        "cluster_first_common_grid_unavailable_pair_count": (
            "cluster pairs without a first-common-grid distance"
        ),
        "cluster_candidate_link_count": "candidate pair links inside the cluster",
        "cluster_unavailable_pair_count": (
            "cluster pairs without the configured assignment distance"
        ),
    }
    for name, description in assignment_counts.items():
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype="i4",
            fill_value=-1,
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "long_name": description,
        })

    assignment_metrics = {
        "gap_from_previous_start_hours": (
            "gap from the previous platform start in chronological order", "hours",
        ),
        "gap_to_next_start_hours": (
            "gap to the next platform start in chronological order", "hours",
        ),
        "cluster_start_spread_seconds": (
            "start-time spread of the assigned cluster", "seconds",
        ),
        "cluster_diameter_m": (
            "complete-link diameter using the configured distance reference", "m",
        ),
        "cluster_maximum_available_pair_distance_m": (
            "maximum available configured-reference distance in the cluster", "m",
        ),
        "cluster_observed_start_diameter_m": (
            "cluster diameter between exact observed starts", "m",
        ),
        "cluster_first_common_grid_diameter_m": (
            "cluster diameter at first common finite grid positions", "m",
        ),
    }
    for name, (description, units) in assignment_metrics.items():
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype="f8",
            fill_value=np.nan,
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "long_name": description,
            "units": units,
        })

    datetime_variables = {
        "deployment_start_time":
            "first manually accepted valid position",
        "trajectory_end_time":
            "last manually accepted valid position",
        "review_end_time":
            "final temporal boundary inspected during manual QC",
    }
    for name, description in datetime_variables.items():
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype="datetime64[ns]",
            fill_value=None,
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "standard_name": "time",
            "timezone": "UTC",
            "long_name": description,
        })

    numeric_variables = {
        "deployment_start_longitude": (
            "longitude of the first manually accepted valid position",
            "degrees_east",
        ),
        "deployment_start_latitude": (
            "latitude of the first manually accepted valid position",
            "degrees_north",
        ),
        "requested_review_span_days": (
            "configured accepted review duration", "days",
        ),
        "accepted_review_span_days": (
            "achieved accepted review duration", "days",
        ),
    }
    for name, (description, units) in numeric_variables.items():
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype="f8",
            fill_value=np.nan,
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "long_name": description,
            "units": units,
        })

    count_variables = {
        "accepted_fix_count": "accepted manual-QC position count",
        "raw_fix_count": "valid pre-manual-QC position count",
    }
    for name, description in count_variables.items():
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype="i8",
            fill_value=None,
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "long_name": description,
        })

    for name, width, description in (
        ("review_end_reason", 32, "reason the manual review interval ended"),
        ("source_sha256", 64, "SHA-256 of the original raw trajectory"),
        ("qc_input_sha256", 64, "SHA-256 of the finalized manual-QC Parquet"),
    ):
        array = root.create_dataset(
            name,
            shape=(len(platforms),),
            chunks=platform_chunks,
            dtype=f"<U{width}",
            compressor=compressor,
        )
        array.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "long_name": description,
        })


    product_root = root.create_group("products")

    for specification in PRODUCT_SPECS:
        grid = np.asarray(
            grids[specification.cadence_minutes],
            dtype="datetime64[ns]",
        )
        if not len(grid):
            raise ValueError(
                f"Grid for {specification.name} is empty"
            )

        group = product_root.create_group(specification.name)
        group.attrs.update({
            "product_name": specification.name,
            "source_kind": specification.source,
            "cadence_minutes": specification.cadence_minutes,
            "maximum_source_gap_minutes":
                specification.maximum_gap_minutes,
            "interpolation": "linear",
            "endpoint_extrapolation": "none",
            "velocity_method": "forward_difference",
            "velocity_assignment": "earlier_endpoint",
        })

        platform = group.create_dataset(
            "platform_id",
            data=platform_values,
            chunks=(min(chunk_platform, len(platforms)),),
            compressor=compressor,
        )
        platform.attrs.update({
            "_ARRAY_DIMENSIONS": ["platform"],
            "cf_role": "trajectory_id",
        })

        time = group.create_dataset(
            "time",
            data=grid,
            chunks=(min(chunk_time, len(grid)),),
            compressor=compressor,
        )
        time.attrs.update({
            "_ARRAY_DIMENSIONS": ["time"],
            "standard_name": "time",
            "timezone": "UTC",
        })

        chunks = (
            min(chunk_platform, len(platforms)),
            min(chunk_time, len(grid)),
        )
        for name, attributes in _PRODUCT_VARIABLES.items():
            array = group.create_dataset(
                name,
                shape=(len(platforms), len(grid)),
                chunks=chunks,
                dtype="f8",
                fill_value=np.nan,
                compressor=compressor,
            )
            array.attrs.update({
                "_ARRAY_DIMENSIONS": ["platform", "time"],
                "coordinates": "platform_id",
                **attributes,
            })

    return root

def write_platform_products(
    root: zarr.Group,
    platform_index: int,
    products: dict[str, RegularTrajectoryProduct],
) -> None:
    """Write every regular product for one platform."""

    expected = {item.name for item in PRODUCT_SPECS}
    if set(products) != expected:
        raise ValueError(
            "Platform products do not match the configured product set"
        )

    platform_count = len(root["platform_id"])
    if not 0 <= platform_index < platform_count:
        raise IndexError(platform_index)

    for specification in PRODUCT_SPECS:
        product = products[specification.name]
        group = root["products"][specification.name]
        stored_time = np.asarray(group["time"][:]).astype(
            "datetime64[ns]"
        )

        if not np.array_equal(stored_time, product.time):
            raise ValueError(
                f"Time grid differs for {specification.name}"
            )

        values = {
            "longitude": product.position.longitude,
            "latitude": product.position.latitude,
            "source_gap_minutes":
                product.position.source_gap_minutes,
            "eastward_velocity":
                product.velocity.east_m_s,
            "northward_velocity":
                product.velocity.north_m_s,
            "speed": product.velocity.speed_m_s,
            "velocity_dt_seconds":
                product.velocity.dt_seconds,
        }

        for name, value in values.items():
            value = np.asarray(value, dtype=float)
            if value.shape != stored_time.shape:
                raise ValueError(
                    f"{specification.name}.{name} has the wrong shape"
                )
            group[name][platform_index, :] = value

def write_platform_metadata(
    root: zarr.Group,
    platform_index: int,
    metadata: PlatformProductMetadata,
) -> None:
    """Write temporal and provenance metadata for one platform."""

    platform_count = len(root["platform_id"])
    if not 0 <= platform_index < platform_count:
        raise IndexError(platform_index)

    start = np.datetime64(metadata.deployment_start_time, "ns")
    end = np.datetime64(metadata.trajectory_end_time, "ns")
    review_end = np.datetime64(metadata.review_end_time, "ns")

    if np.isnat([start, end, review_end]).any():
        raise ValueError("Platform temporal metadata must be valid")
    if start > end or end > review_end:
        raise ValueError(
            "Platform times must satisfy deployment <= trajectory end <= review end"
        )
    start_lon = float(metadata.deployment_start_longitude)
    start_lat = float(metadata.deployment_start_latitude)
    if (
        not np.isfinite(start_lon)
        or not np.isfinite(start_lat)
        or not -180 <= start_lon <= 180
        or not -90 <= start_lat <= 90
    ):
        raise ValueError("Platform deployment-start coordinates are invalid")

    root["deployment_start_time"][platform_index] = start
    root["deployment_start_longitude"][platform_index] = start_lon
    root["deployment_start_latitude"][platform_index] = start_lat
    root["trajectory_end_time"][platform_index] = end
    root["review_end_time"][platform_index] = review_end
    root["requested_review_span_days"][platform_index] = (
        metadata.requested_review_span_days
    )
    root["accepted_review_span_days"][platform_index] = (
        metadata.accepted_review_span_days
    )
    root["review_end_reason"][platform_index] = metadata.review_end_reason
    root["accepted_fix_count"][platform_index] = metadata.accepted_fix_count
    root["raw_fix_count"][platform_index] = metadata.raw_fix_count
    root["source_sha256"][platform_index] = metadata.source_sha256
    root["qc_input_sha256"][platform_index] = metadata.qc_input_sha256

def write_platform_assignment(
    root: zarr.Group,
    platform_index: int,
    *,
    assignment: PlatformAssignmentMetadata,
) -> None:
    """Write complete array and initial-cluster assignment for one platform."""

    platform_count = len(root["platform_id"])
    if not 0 <= platform_index < platform_count:
        raise IndexError(platform_index)
    if (
        isinstance(assignment.array_id, bool)
        or not isinstance(assignment.array_id, (int, np.integer))
        or assignment.array_id < 1
    ):
        raise ValueError("array_id must be a positive integer")
    if (
        not isinstance(assignment.cluster_id, str)
        or not assignment.cluster_id.strip()
        or len(assignment.cluster_id) > 64
    ):
        raise ValueError("cluster_id must contain between 1 and 64 characters")
    if (
        not isinstance(assignment.cluster_assignment_status, str)
        or not assignment.cluster_assignment_status.strip()
        or len(assignment.cluster_assignment_status) > 32
    ):
        raise ValueError("cluster_assignment_status is invalid")

    root["array_id"][platform_index] = assignment.array_id
    root["cluster_id"][platform_index] = assignment.cluster_id
    root["cluster_assignment_status"][platform_index] = (
        assignment.cluster_assignment_status
    )
    for name in (
        "member_id",
        "cluster_size",
        "cluster_first_common_grid_unavailable_pair_count",
        "cluster_candidate_link_count",
        "cluster_unavailable_pair_count",
    ):
        value = getattr(assignment, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        minimum = 1 if name in {"member_id", "cluster_size"} else 0
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        root[name][platform_index] = value

    optional_metrics = {
        "gap_from_previous_start_hours",
        "gap_to_next_start_hours",
        "cluster_diameter_m",
        "cluster_maximum_available_pair_distance_m",
        "cluster_observed_start_diameter_m",
        "cluster_first_common_grid_diameter_m",
    }
    for name in (
        "gap_from_previous_start_hours",
        "gap_to_next_start_hours",
        "cluster_start_spread_seconds",
        "cluster_diameter_m",
        "cluster_maximum_available_pair_distance_m",
        "cluster_observed_start_diameter_m",
        "cluster_first_common_grid_diameter_m",
    ):
        raw_value = getattr(assignment, name)
        if raw_value is None and name in optional_metrics:
            value = np.nan
        else:
            value = float(raw_value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        root[name][platform_index] = value
