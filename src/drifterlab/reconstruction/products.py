"""Definitions and common time grids for analysis trajectory products."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from .core import RegularTrajectoryProduct, build_regular_product

import numpy as np


NS_PER_MINUTE = 60_000_000_000

@dataclass(frozen=True)
class TrajectorySource:
    """Ordered source positions used to construct regular products."""

    time: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray

@dataclass(frozen=True)
class RegularProductSpec:
    """Configuration for one analysis-ready trajectory product."""

    name: str
    source: Literal["raw", "qc"]
    cadence_minutes: int
    maximum_gap_minutes: float


PRODUCT_SPECS = (
    RegularProductSpec("raw_5min", "raw", 5, 30),
    RegularProductSpec("qc_5min", "qc", 5, 30),
    RegularProductSpec("qc_15min", "qc", 15, 60),
    RegularProductSpec("qc_30min", "qc", 30, 120),
)


def build_regular_grids(
    anchor: np.datetime64,
    available_start: np.datetime64,
    available_end: np.datetime64,
    cadences_minutes: tuple[int, ...] = (5, 15, 30),
) -> dict[int, np.ndarray]:
    """Build phase-aligned grids covering the available common interval."""

    timestamps = np.asarray(
        [anchor, available_start, available_end],
        dtype="datetime64[ns]",
    )
    if np.isnat(timestamps).any():
        raise ValueError("Grid timestamps must be valid")

    anchor_ns, start_ns, end_ns = map(int, timestamps.astype(np.int64))
    if start_ns > end_ns:
        raise ValueError("available_start must not be after available_end")

    grids: dict[int, np.ndarray] = {}
    for cadence in cadences_minutes:
        if (
            isinstance(cadence, bool)
            or not isinstance(cadence, int)
            or cadence <= 0
        ):
            raise ValueError("Grid cadences must be positive integers")

        step_ns = cadence * NS_PER_MINUTE
        start_delta = start_ns - anchor_ns
        end_delta = end_ns - anchor_ns

        # Ceiling division finds the first aligned cell on or after the start.
        first_index = -(-start_delta // step_ns)
        last_index = end_delta // step_ns

        if last_index < first_index:
            raise ValueError(
                f"No {cadence}-minute grid point falls inside the available interval"
            )

        indices = np.arange(
            first_index,
            last_index + 1,
            dtype=np.int64,
        )
        grids[cadence] = (
            anchor_ns + indices * step_ns
        ).astype("datetime64[ns]")

    return grids

def build_platform_products(
    raw: TrajectorySource,
    qc: TrajectorySource,
    grids: dict[int, np.ndarray],
    *,
    coverage_start: np.datetime64,
    coverage_end: np.datetime64,
) -> dict[str, RegularTrajectoryProduct]:
    """Build every configured analysis product for one platform."""

    sources = {
        "raw": raw,
        "qc": qc,
    }
    products: dict[str, RegularTrajectoryProduct] = {}

    for specification in PRODUCT_SPECS:
        if specification.cadence_minutes not in grids:
            raise ValueError(
                f"Missing {specification.cadence_minutes}-minute grid "
                f"for {specification.name}"
            )

        source = sources[specification.source]
        products[specification.name] = build_regular_product(
            source.time,
            source.longitude,
            source.latitude,
            grids[specification.cadence_minutes],
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            maximum_gap_minutes=specification.maximum_gap_minutes,
        )

    return products