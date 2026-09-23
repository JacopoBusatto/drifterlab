"""Campaign-independent trajectory records; each series owns its time axis."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drifterlab.io.matlab import matlab_datenum_to_datetime64


@dataclass
class ObservationSeries:
    time: np.ndarray
    time_matlab: np.ndarray
    source_index: np.ndarray
    variables: dict[str, np.ndarray]
    source_order: str
    attributes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_matlab(cls, times: np.ndarray, variables: dict[str, np.ndarray],
                    attributes: dict[str, dict[str, Any]] | None = None) -> "ObservationSeries":
        times = np.asarray(times, dtype=np.float64).reshape(-1)
        converted = matlab_datenum_to_datetime64(times)
        for name, values in variables.items():
            if np.asarray(values).shape != times.shape:
                raise ValueError(f"{name}: shape {np.asarray(values).shape} does not match time {times.shape}")
        finite = times[np.isfinite(times)]
        diffs = np.diff(finite)
        if len(finite) != len(times):
            source_order = "invalid_times"
        elif len(times) < 2 or np.all(diffs >= 0):
            source_order = "ascending"
        elif np.all(diffs <= 0):
            source_order = "descending"
        else:
            source_order = "mixed"
        order = np.argsort(converted, kind="stable")  # NaT sorts last.
        return cls(converted[order], times[order], order.astype(np.int64),
                   {k: np.asarray(v)[order] for k, v in variables.items()},
                   source_order, attributes or {})


@dataclass
class TrajectoryRecord:
    platform_code: str
    metadata: dict[str, Any]
    series: dict[str, ObservationSeries]
    position_axes: dict[str, str]
    unaligned: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata_attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    unaligned_attributes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        names = set(self.metadata) | {"platform_code"}
        for axis, series in self.series.items():
            n = len(series.time)
            generated = {f"time_{axis}", f"time_matlab_{axis}",
                         f"source_obs_index_{axis}", f"observation_present_{axis}", f"n_obs_{axis}"}
            all_names = generated | set(series.variables)
            if generated & set(series.variables) or names & all_names:
                raise ValueError(f"Variable name collision on axis {axis}")
            names |= all_names
            if len(series.time_matlab) != n or len(series.source_index) != n:
                raise ValueError(f"Mismatched provenance arrays on axis {axis}")
            for name, value in series.variables.items():
                if np.asarray(value).shape != (n,):
                    raise ValueError(f"{name} must have exactly {n} observations")
        if not set(self.position_axes.values()) <= set(self.series):
            raise ValueError("A position representation refers to a missing time axis")
