"""Write trajectory records with independent observation axes to Zarr."""

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import xarray as xr
import zarr

from .records import TrajectoryRecord


@dataclass
class VariableSpec:
    axis: str | None
    dtype: np.dtype
    attributes: dict = field(default_factory=dict)


def record_arrays(record: TrajectoryRecord) -> dict[str, tuple[str | None, np.ndarray, dict]]:
    arrays = {name: (None, np.asarray(value), record.metadata_attributes.get(name, {}))
              for name, value in {**record.metadata, "platform_code": record.platform_code}.items()}
    arrays["platform_code"][2].update(cf_role="trajectory_id")
    for axis, series in record.series.items():
        time_attrs = {"long_name": f"Source observation time on {axis} axis"}
        if "time_reference" in record.metadata:
            time_attrs["time_reference"] = record.metadata["time_reference"]
        arrays[f"time_{axis}"] = (axis, series.time, time_attrs)
        arrays[f"time_matlab_{axis}"] = (axis, series.time_matlab, {"long_name": "Original MATLAB datenum, with missing sentinel normalized"})
        arrays[f"source_obs_index_{axis}"] = (axis, series.source_index, {"long_name": "Zero-based row index in source MATLAB vector; -1 marks padding"})
        arrays[f"observation_present_{axis}"] = (axis, np.ones(len(series.time), dtype=bool), {"long_name": "True for a source row, including rows with invalid timestamps"})
        arrays[f"n_obs_{axis}"] = (None, np.asarray(len(series.time)), {})
        for name, value in series.variables.items():
            arrays[name] = (axis, np.asarray(value), series.attributes.get(name, {}))
    return arrays


def _dtype(array: np.ndarray) -> np.dtype:
    kind = array.dtype.kind
    if kind == "M":
        return np.dtype("datetime64[ns]")
    if kind == "b":
        return np.dtype(bool)
    if kind in "iu":
        return np.dtype("int64")
    if kind == "f":
        return np.dtype("float64")
    if kind in "US":
        width = max([1, *(len(str(v)) for v in array.reshape(-1))])
        return np.dtype(f"U{width}")
    raise ValueError(f"Unsupported output dtype {array.dtype}")


@dataclass
class DatasetLayout:
    axis_sizes: dict[str, int] = field(default_factory=dict)
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    unaligned_sizes: dict[str, int] = field(default_factory=dict)
    platforms: list[str] = field(default_factory=list)

    def observe(self, record: TrajectoryRecord) -> None:
        record.validate()
        if record.platform_code in self.platforms:
            raise ValueError(f"Duplicate platform ID: {record.platform_code}")
        self.platforms.append(record.platform_code)
        for axis, series in record.series.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", axis):
                raise ValueError(f"Invalid axis name {axis!r}")
            self.axis_sizes[axis] = max(self.axis_sizes.get(axis, 0), len(series.time))
        for name, (axis, values, attrs) in record_arrays(record).items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid variable name {name!r}")
            dtype = _dtype(values)
            if name in self.variables:
                previous = self.variables[name]
                if previous.axis != axis or previous.dtype.kind != dtype.kind:
                    raise ValueError(f"Incompatible output schemas for {name}")
                if dtype.kind == "U" and dtype.itemsize > previous.dtype.itemsize:
                    previous.dtype = dtype
            else:
                self.variables[name] = VariableSpec(axis, dtype, attrs.copy())
        for name, values in record.unaligned.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid unaligned field name {name!r}")
            self.unaligned_sizes[name] = max(self.unaligned_sizes.get(name, 0), len(values))


def _fill(dtype: np.dtype, *, count: bool = False) -> Any:
    return {"M": np.datetime64("NaT", "ns"), "f": np.nan, "b": False,
            "i": 0 if count else -1, "U": ""}[dtype.kind]


def _row_dataset(record: TrajectoryRecord, layout: DatasetLayout, index: int) -> xr.Dataset:
    arrays = record_arrays(record)
    variables = {}
    for name, spec in layout.variables.items():
        length = 1 if spec.axis is None else layout.axis_sizes[spec.axis]
        shape = (1,) if spec.axis is None else (1, length)
        output = np.full(shape, _fill(spec.dtype, count=name.startswith("n_obs_")), dtype=spec.dtype)
        if name in arrays:
            _, values, _ = arrays[name]
            if spec.axis is None:
                output[0] = values.item()
            else:
                output[0, :len(values)] = values
        dims = ("trajectory",) if spec.axis is None else ("trajectory", f"obs_{spec.axis}")
        variables[name] = xr.Variable(dims, output, attrs=spec.attributes.copy())
    coords = {"trajectory": np.array([index], dtype=np.int64)}
    coords.update({f"obs_{axis}": np.arange(n, dtype=np.int64) for axis, n in layout.axis_sizes.items()})
    return xr.Dataset(variables, coords=coords)


def _encoding(dataset: xr.Dataset, trajectory_chunk: int, observation_chunk: int) -> dict:
    encoding = {}
    for name, variable in dataset.variables.items():
        chunks = tuple(trajectory_chunk if dim == "trajectory" else min(observation_chunk, max(1, dataset.sizes[dim]))
                       for dim in variable.dims)
        spec: dict[str, Any] = {"chunks": chunks}
        if variable.dtype.kind in "biu":
            spec["_FillValue"] = None  # False and integer zero are real data.
        elif variable.dtype.kind == "M":
            spec.update(dtype="int64", units="nanoseconds since 1970-01-01")
        encoding[name] = spec
    return encoding


def _verify(store: Path, layout: DatasetLayout, platforms: list[str]) -> None:
    with xr.open_zarr(store, consolidated=True, chunks=None) as ds:
        expected = {"trajectory": len(platforms), **{f"obs_{k}": v for k, v in layout.axis_sizes.items()}}
        if dict(ds.sizes) != expected or ds.platform_code.values.tolist() != platforms:
            raise ValueError("Zarr readback failed: dimensions or platform identifiers differ")
        for name, spec in layout.variables.items():
            if ds[name].dtype.kind != spec.dtype.kind:
                raise ValueError(f"Zarr readback changed dtype of {name}")
        for axis, size in layout.axis_sizes.items():
            counts = ds[f"n_obs_{axis}"].values
            if np.any((counts < 0) | (counts > size)):
                raise ValueError(f"Invalid observation counts on {axis}")
            actual_counts = ds[f"observation_present_{axis}"].values.sum(axis=1)
            if not np.array_equal(counts, actual_counts):
                raise ValueError(f"Padding/observation count mismatch on {axis}")


def write_master(records: Iterable[TrajectoryRecord], layout: DatasetLayout, path: Path,
                 *, attributes: dict, trajectory_chunk: int = 1, observation_chunk: int = 4096) -> None:
    """Stream one padded row at a time, verify, then publish a new local store."""
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Zarr already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    temporary.mkdir()
    platforms: list[str] = []
    auxiliary_written: set[str] = set()
    try:
        for index, record in enumerate(records):
            row = _row_dataset(record, layout, index)
            row.attrs = attributes
            if index == 0:
                row.to_zarr(temporary, mode="w", zarr_format=2, consolidated=False,
                            encoding=_encoding(row, trajectory_chunk, observation_chunk))
            else:
                row.to_zarr(temporary, mode="a", append_dim="trajectory", zarr_format=2, consolidated=False)
            platforms.append(record.platform_code)
            for name, values in record.unaligned.items():
                length = layout.unaligned_sizes[name]
                output = np.full((1, length), np.nan)
                output[0, :len(values)] = values
                aux = xr.Dataset({name: (("trajectory", "source_obs"), output),
                                  "platform_code": ("trajectory", np.asarray([record.platform_code], dtype=layout.variables["platform_code"].dtype)),
                                  "n_source_values": ("trajectory", np.asarray([len(values)], dtype=np.int64))},
                                 coords={"trajectory": [index], "source_obs": np.arange(length, dtype=np.int64)},
                                 attrs={"description": "Unaligned source values in original source order, with missing sentinel normalized. No timestamps inferred.",
                                        "parent_trajectory_coordinate": "Matches the root trajectory coordinate"})
                aux[name].attrs.update(record.unaligned_attributes.get(name, {}))
                kwargs = {"encoding": _encoding(aux, trajectory_chunk, observation_chunk)} if name not in auxiliary_written else {"append_dim": "trajectory"}
                aux.to_zarr(temporary, group=f"source_unaligned/{name}", mode="a", zarr_format=2, consolidated=False, **kwargs)
                auxiliary_written.add(name)
        if len(platforms) != len(layout.platforms) or platforms != sorted(layout.platforms):
            raise ValueError("Written platforms do not match the inventory's unique sorted platform IDs")
        zarr.consolidate_metadata(str(temporary))
        _verify(temporary, layout, platforms)
        if path.exists():
            raise FileExistsError(f"Zarr already exists: {path}")
        temporary.rename(path)
    finally:
        if temporary.exists():
            # Only remove the exact temporary directory created by this call.
            resolved = temporary.resolve()
            if resolved.parent != path.parent or resolved.name != f".{path.name}.{token}.tmp":
                raise RuntimeError("Refusing cleanup outside the expected output directory")
            shutil.rmtree(resolved)
