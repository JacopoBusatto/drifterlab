import numpy as np
import pytest

from drifterlab.reconstruction.products import (
    PRODUCT_SPECS,
    TrajectorySource,
    build_platform_products,
    build_regular_grids,
)

def times(minutes):
    return (
        np.datetime64("2025-01-12T00:00:00", "ns")
        + np.asarray(minutes).astype("timedelta64[m]")
    )

def test_product_specifications_freeze_the_agreed_strategy():
    assert [
        (
            item.name,
            item.source,
            item.cadence_minutes,
            item.maximum_gap_minutes,
        )
        for item in PRODUCT_SPECS
    ] == [
        ("raw_5min", "raw", 5, 30),
        ("qc_5min", "qc", 5, 30),
        ("qc_15min", "qc", 15, 60),
        ("qc_30min", "qc", 30, 120),
    ]


def test_regular_grids_share_an_anchor_and_cover_available_time():
    grids = build_regular_grids(
        np.datetime64("2025-01-12T00:00:00"),
        np.datetime64("2025-01-12T00:07:00"),
        np.datetime64("2025-01-12T01:07:00"),
    )

    np.testing.assert_array_equal(
        grids[5][[0, -1]],
        np.asarray(
            ["2025-01-12T00:10", "2025-01-12T01:05"],
            dtype="datetime64[m]",
        ),
    )
    np.testing.assert_array_equal(
        grids[15][[0, -1]],
        np.asarray(
            ["2025-01-12T00:15", "2025-01-12T01:00"],
            dtype="datetime64[m]",
        ),
    )
    np.testing.assert_array_equal(
        grids[30][[0, -1]],
        np.asarray(
            ["2025-01-12T00:30", "2025-01-12T01:00"],
            dtype="datetime64[m]",
        ),
    )

def test_platform_products_separate_raw_and_qc_effects():
    grids = build_regular_grids(
        times([0])[0],
        times([0])[0],
        times([30])[0],
    )

    raw = TrajectorySource(
        time=times([0, 5, 10, 16, 20, 25, 30]),
        longitude=np.asarray([0.0, 0.05, 5.0, 0.15, 0.20, 0.25, 0.30]),
        latitude=np.zeros(7),
    )
    qc = TrajectorySource(
        # The raw outlier at minute 10 has been rejected.
        time=times([0, 5, 15, 20, 25, 30]),
        longitude=np.asarray([0.0, 0.05, 0.15, 0.20, 0.25, 0.30]),
        latitude=np.zeros(6),
    )

    products = build_platform_products(
        raw,
        qc,
        grids,
        coverage_start=times([0])[0],
        coverage_end=times([30])[0],
    )

    assert set(products) == {
        "raw_5min",
        "qc_5min",
        "qc_15min",
        "qc_30min",
    }

    # The raw product preserves the source outlier.
    assert products["raw_5min"].position.longitude[2] == 5.0

    # QC interpolates through the removed observation.
    assert products["qc_5min"].position.longitude[2] == pytest.approx(0.10)

    # Both products use the identical five-minute time coordinate.
    np.testing.assert_array_equal(
        products["raw_5min"].time,
        products["qc_5min"].time,
    )

    # The outlier produces a different raw velocity.
    assert (
        products["raw_5min"].velocity.speed_m_s[1]
        != pytest.approx(products["qc_5min"].velocity.speed_m_s[1])
    )

    assert len(products["qc_5min"].time) == 7
    assert len(products["qc_15min"].time) == 3
    assert len(products["qc_30min"].time) == 2