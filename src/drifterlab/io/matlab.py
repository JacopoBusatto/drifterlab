"""MATLAB utilities without campaign or timezone assumptions."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

DAY_NS = 86_400_000_000_000
UNIX_EPOCH_DATENUM = 719529


@dataclass
class MatlabFile:
    values: dict[str, Any]
    sha256: str
    header: str


def read_matlab(path: str | Path) -> MatlabFile:
    """Read the supported MAT formats; never reinterpret an unsupported file."""
    content = Path(path).read_bytes()
    try:
        loaded = loadmat(BytesIO(content), simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3/HDF5 is not supported by this reader") from exc
    header = loaded.get("__header__", b"")
    if isinstance(header, bytes):
        header = header.decode("utf-8", errors="replace")
    return MatlabFile(
        {k: v for k, v in loaded.items() if not k.startswith("__")},
        sha256(content).hexdigest(), str(header),
    )


def normalize_missing(values: Any, missing_value: float = -999) -> np.ndarray:
    """Copy numeric data to float64 before replacing integer or float sentinels."""
    result = np.array(values, dtype=np.float64, copy=True)
    result[result == missing_value] = np.nan
    return result


def matlab_datenum_to_datetime64(values: Any, missing_value: float = -999) -> np.ndarray:
    """Convert serial days to nanoseconds, preserving subsecond source precision.

    The result has no timezone semantics. Nonfinite, sentinel, and out-of-range
    values become NaT. Callers retain the original numbers for auditing.
    Integer days and fractional days are converted separately to avoid losing
    precision through a floating-point nanosecond timestamp.
    """
    source = normalize_missing(values, missing_value)
    flat = source.reshape(-1)
    output = np.full(flat.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = np.isfinite(flat) & (flat >= UNIX_EPOCH_DATENUM - 106752) & (
        flat < UNIX_EPOCH_DATENUM + 106752
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return output.reshape(source.shape)
    whole = np.floor(flat[indices]).astype(np.int64)
    days = whole - UNIX_EPOCH_DATENUM
    fraction_ns = np.rint((flat[indices] - whole) * DAY_NS).astype(np.int64)
    interior = (days > -106751) & (days < 106751)
    output.view("i8")[indices[interior]] = days[interior] * DAY_NS + fraction_ns[interior]
    # Only the two range boundaries require Python integers to avoid overflow.
    for index, day, fraction in zip(indices[~interior], days[~interior], fraction_ns[~interior]):
        ns = int(day) * DAY_NS + int(fraction)
        if np.iinfo(np.int64).min < ns <= np.iinfo(np.int64).max:
            output.view("i8")[index] = ns
    return output.reshape(source.shape)
