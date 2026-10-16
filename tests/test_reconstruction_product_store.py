import numpy as np
import xarray as xr
import zarr
import pytest

from drifterlab.reconstruction.product_store import (
    PlatformAssignmentMetadata,
    create_product_store,
    write_platform_products,
    write_platform_assignment,
)
from drifterlab.reconstruction.products import (
    TrajectorySource,
    build_platform_products,
    build_regular_grids,
)


def times(minutes):
    return (
        np.datetime64("2025-01-12T00:00:00", "ns")
        + np.asarray(minutes).astype("timedelta64[m]")
    )


def test_grouped_product_store_round_trip(tmp_path):
    grids = build_regular_grids(
        times([0])[0],
        times([0])[0],
        times([30])[0],
    )

    raw = TrajectorySource(
        times([0, 5, 10, 15, 20, 25, 30]),
        np.asarray([0.0, 0.05, 5.0, 0.15, 0.20, 0.25, 0.30]),
        np.zeros(7),
    )
    qc = TrajectorySource(
        times([0, 5, 15, 20, 25, 30]),
        np.asarray([0.0, 0.05, 0.15, 0.20, 0.25, 0.30]),
        np.zeros(6),
    )
    products = build_platform_products(
        raw,
        qc,
        grids,
        coverage_start=times([0])[0],
        coverage_end=times([30])[0],
    )

    path = tmp_path / "products.zarr"
    root = create_product_store(
        path,
        ["p1"],
        grids,
        chunk_platform=1,
        chunk_time=4,
    )

    write_platform_assignment(
        root,
        0,
        assignment=PlatformAssignmentMetadata(
            array_id=2,
            gap_from_previous_start_hours=1.0,
            gap_to_next_start_hours=None,
            cluster_id="array_002__cluster_001",
            member_id=1,
            cluster_size=1,
            cluster_assignment_status="algorithmic_singleton",
            cluster_start_spread_seconds=0.0,
            cluster_diameter_m=0.0,
            cluster_maximum_available_pair_distance_m=0.0,
            cluster_observed_start_diameter_m=0.0,
            cluster_first_common_grid_diameter_m=0.0,
            cluster_first_common_grid_unavailable_pair_count=0,
            cluster_candidate_link_count=0,
            cluster_unavailable_pair_count=0,
        ),
    )

    write_platform_products(root, 0, products)
    zarr.consolidate_metadata(str(path))

    reopened = zarr.open_group(str(path), mode="r")
    assert set(reopened["products"].group_keys()) == {
        "raw_5min",
        "qc_5min",
        "qc_15min",
        "qc_30min",
    }

    with xr.open_zarr(
        path,
        group="products/raw_5min",
        consolidated=True,
        chunks=None,
    ) as raw_dataset:
        assert dict(raw_dataset.sizes) == {
            "platform": 1,
            "time": 7,
        }
        assert raw_dataset.platform_id.values.tolist() == ["p1"]
        assert raw_dataset.longitude.isel(
            platform=0,
            time=2,
        ).item() == 5.0
        assert raw_dataset.speed.isel(
            platform=0,
            time=-1,
        ).isnull().item()

    with xr.open_zarr(
        path,
        group="products/qc_5min",
        consolidated=True,
        chunks=None,
    ) as qc_dataset:
        assert qc_dataset.longitude.isel(
            platform=0,
            time=2,
        ).item() == pytest.approx(0.10)
        assert qc_dataset.attrs["maximum_source_gap_minutes"] == 30
        assert qc_dataset.attrs["velocity_method"] == "forward_difference"

    with xr.open_zarr(
        path,
        consolidated=True,
        chunks=None,
    ) as metadata:
        assert metadata.array_id.item() == 2
        assert metadata.cluster_id.item() == "array_002__cluster_001"
        assert metadata.member_id.item() == 1
        assert metadata.cluster_size.item() == 1
        assert metadata.cluster_assignment_status.item() == "algorithmic_singleton"
