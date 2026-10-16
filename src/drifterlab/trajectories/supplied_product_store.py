"""Write configurable grouped trajectory products without manual-QC semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import xarray as xr
import zarr
from numcodecs import Blosc


def _encoding(dataset: xr.Dataset, *, platform: int, time: int) -> dict[str, dict]:
    """Return conservative Zarr-v2 compression and chunking for one dataset."""

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    result: dict[str, dict] = {}
    for name, variable in dataset.variables.items():
        chunks: list[int] = []
        for dimension, size in zip(variable.dims, variable.shape, strict=True):
            if dimension == "platform":
                chunks.append(min(platform, size))
            elif dimension == "time":
                chunks.append(min(time, size))
            else:
                chunks.append(size)
        values: dict[str, object] = {"compressor": compressor}
        if chunks:
            values["chunks"] = tuple(chunks)
        result[name] = values
    return result


def write_supplied_product_store(
    path: str | Path,
    root: xr.Dataset,
    products: Mapping[str, xr.Dataset],
    *,
    manifest: Mapping[str, object],
    chunk_platform: int = 1,
    chunk_time: int = 2016,
) -> Path:
    """Write one new schema-compatible store and consolidate its metadata."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"Product store already exists: {destination}")
    if not products:
        raise ValueError("At least one supplied-QC product is required")
    if chunk_platform < 1 or chunk_time < 1:
        raise ValueError("Zarr chunk sizes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)

    root.to_zarr(
        destination,
        mode="w",
        consolidated=False,
        zarr_format=2,
        encoding=_encoding(root, platform=chunk_platform, time=chunk_time),
    )
    for name, dataset in products.items():
        dataset.to_zarr(
            destination,
            group=f"products/{name}",
            mode="a",
            consolidated=False,
            zarr_format=2,
            encoding=_encoding(
                dataset,
                platform=chunk_platform,
                time=chunk_time,
            ),
        )

    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    zarr.consolidate_metadata(str(destination))
    return destination

