"""Generic native-position records and explicit source-format readers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .matlab import read_matlab


@dataclass(frozen=True)
class NativeTrajectory:
    """One immutable source trajectory in its original observation order."""

    platform_code: str
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    source_obs_index: np.ndarray
    source_path: Path
    source_sha256: str

    def __post_init__(self) -> None:
        lengths = {
            len(np.asarray(self.time)), len(np.asarray(self.lon)),
            len(np.asarray(self.lat)), len(np.asarray(self.source_obs_index)),
        }
        if len(lengths) != 1:
            raise ValueError("Native trajectory arrays must have equal lengths")
        indices = np.asarray(self.source_obs_index)
        if indices.ndim != 1 or len(np.unique(indices)) != len(indices) or (indices < 0).any():
            raise ValueError("source_obs_index must contain unique nonnegative values")


def _identifier(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    item = array.item()
    if isinstance(item, (int, np.integer)):
        return str(int(item))
    if isinstance(item, (float, np.floating)) and np.isfinite(item) and item.is_integer():
        return str(int(item))
    text = str(item).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def read_microsvp_position_mat(
    path: str | Path, *, missing_value: float = -999,
) -> NativeTrajectory:
    """Read raw MicroSVP positions without sorting or changing coordinates.

    ``missing_value`` is accepted as a reader option for symmetry with the raw
    signal reader.  It is deliberately not substituted in longitude/latitude:
    source validity is a QC result, not an I/O mutation.
    """
    del missing_value
    path = Path(path).resolve()
    loaded = read_matlab(path)
    dataset = loaded.values.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Raw file must contain a scalar dataset structure")
    names = [
        name for name, value in dataset.items()
        if name.startswith("drifter_") and isinstance(value, dict)
    ]
    if len(names) != 1:
        raise ValueError(
            f"Expected one dataset.drifter_<ID> structure, found {len(names)}"
        )
    name = names[0]
    track = dataset[name]
    required = {"PlatformId", "ObsTimestamp", "GpsLongitude", "GpsLatitude"}
    missing = required - set(track)
    if missing:
        raise ValueError(f"Raw drifter structure is missing fields: {sorted(missing)}")
    platform = _identifier(track["PlatformId"], f"{name}.PlatformId")
    if name.removeprefix("drifter_") != platform:
        raise ValueError("Raw structure name disagrees with its embedded PlatformId")
    raw_time = np.asarray(track["ObsTimestamp"]).reshape(-1)
    time = pd.to_datetime(raw_time, errors="coerce", utc=True).tz_convert(None).to_numpy(
        dtype="datetime64[ns]"
    )
    lon = np.asarray(track["GpsLongitude"]).reshape(-1).astype(float, copy=True)
    lat = np.asarray(track["GpsLatitude"]).reshape(-1).astype(float, copy=True)
    if len(lon) != len(time) or len(lat) != len(time):
        raise ValueError("Raw position lengths do not match ObsTimestamp")
    return NativeTrajectory(
        platform, time, lon, lat, np.arange(len(time), dtype=np.int64),
        path, loaded.sha256,
    )


PositionReader = Callable[..., NativeTrajectory]
POSITION_READERS: dict[str, PositionReader] = {
    "microsvp_mat": read_microsvp_position_mat,
}


def get_position_reader(name: str) -> PositionReader:
    try:
        return POSITION_READERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(POSITION_READERS))
        raise ValueError(
            f"Unsupported position input reader {name!r}; supported readers: {supported}"
        ) from exc

