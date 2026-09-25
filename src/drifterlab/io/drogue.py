"""Generic raw drogue signals and explicit source-format readers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .matlab import normalize_missing, read_matlab


@dataclass(frozen=True)
class RawDrogueSignals:
    platform_code: str
    time: np.ndarray
    ttff: np.ndarray
    strain: np.ndarray | None
    source_path: Path
    source_sha256: str


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


def _optional_vector(
    track: dict, field: str, length: int, missing_value: float,
) -> np.ndarray | None:
    if field not in track:
        return None
    values = normalize_missing(track[field], missing_value).reshape(-1)
    if len(values) != length:
        raise ValueError(
            f"{field} length {len(values)} does not match ObsTimestamp length {length}"
        )
    return values


def read_microsvp_mat(
    path: str | Path, *, missing_value: float = -999,
) -> RawDrogueSignals:
    """Map a MicroSVP MATLAB signal export to the generic drogue schema."""
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
    platform = _identifier(track.get("PlatformId"), f"{name}.PlatformId")
    if name.removeprefix("drifter_") != platform:
        raise ValueError("Raw structure name disagrees with its embedded PlatformId")
    if "ObsTimestamp" not in track or "GpsTTFF" not in track:
        raise ValueError("Raw drifter structure requires ObsTimestamp and GpsTTFF")
    source_time = np.asarray(track["ObsTimestamp"]).reshape(-1)
    time = pd.to_datetime(source_time, errors="coerce", utc=True).tz_convert(None).to_numpy(
        dtype="datetime64[ns]"
    )
    ttff = normalize_missing(track["GpsTTFF"], missing_value).reshape(-1)
    if len(ttff) != len(time):
        raise ValueError(
            f"GpsTTFF length {len(ttff)} does not match ObsTimestamp length {len(time)}"
        )
    strain = _optional_vector(track, "Drogue", len(time), missing_value)
    order = np.argsort(time, kind="stable")
    return RawDrogueSignals(
        platform,
        time[order],
        ttff[order],
        None if strain is None else strain[order],
        path,
        loaded.sha256,
    )


DrogueReader = Callable[..., RawDrogueSignals]
READERS: dict[str, DrogueReader] = {"microsvp_mat": read_microsvp_mat}


def get_drogue_reader(name: str) -> DrogueReader:
    """Return a configured source adapter from the small explicit registry."""
    try:
        return READERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(READERS))
        raise ValueError(
            f"Unsupported drogue input reader {name!r}; supported readers: {supported}"
        ) from exc
