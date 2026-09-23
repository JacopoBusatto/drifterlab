"""Lossless normalization of the inspected ARCTERX QC MicroSVP schema."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from drifterlab.io.matlab import matlab_datenum_to_datetime64, normalize_missing, read_matlab
from drifterlab.trajectories.records import ObservationSeries, TrajectoryRecord
from .schema import INTERP_FIELDS, NATIVE_FIELDS, NATIVE_METADATA_FIELDS, field_attributes


def _vector(value: Any, name: str, missing_value: float) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name}: expected a real numeric vector, got {array.shape}/{array.dtype}")
    return normalize_missing(np.atleast_1d(array), missing_value)


def _identifier(value: Any, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    scalar = array.item()
    if isinstance(scalar, str):
        result = scalar.strip()
    elif isinstance(scalar, (int, np.integer)):
        result = str(scalar)
    elif isinstance(scalar, (float, np.floating)) and np.isfinite(scalar) and scalar.is_integer():
        if abs(scalar) > 2**53:
            raise ValueError(f"{name}: float identifier exceeds exact integer precision")
        result = str(int(scalar))
    else:
        raise ValueError(f"{name}: invalid identifier {scalar!r}")
    if not result:
        raise ValueError(f"{name} is empty")
    return result


def _schema(value: Any) -> dict:
    if isinstance(value, dict):
        return {k: _schema(v) for k, v in sorted(value.items())}
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _extra_fields(structure: str, obj: dict, known: set[str], n: int,
                  variables: dict, metadata: dict, attributes: dict, warnings: list,
                  missing_value: float) -> None:
    for name in sorted(set(obj) - known):
        value = obj[name]
        array = np.asarray(value)
        target = f"source_{structure}_{name}"
        if isinstance(value, dict) or array.dtype.kind not in "biufUS":
            raise ValueError(f"Unsupported source field {structure}.{name}: {array.shape}/{array.dtype}")
        if array.ndim == 0:
            metadata[target] = (normalize_missing(array, missing_value).item()
                                if array.dtype.kind in "biuf" else str(array.item()))
        elif array.shape == (n,):
            variables[target] = (normalize_missing(array, missing_value)
                                 if array.dtype.kind in "biuf" else array.astype(str))
            attributes[target] = {"source_field": f"{structure}.{name}"}
        else:
            raise ValueError(f"Cannot align unknown field {structure}.{name}: {array.shape}, time length {n}")
        warnings.append(f"preserved_unmapped_field:{structure}.{name}")


def read_microsvp(path: str | Path, *, missing_value: float = -999) -> TrajectoryRecord:
    """Read and sort each source time axis, without applying an analysis policy.

    Returned datetimes represent UTC by the ARCTERX campaign convention.
    Audit and drogue decisions are added by ``preprocess``.
    """
    path = Path(path)
    loaded = read_matlab(path)
    if set(loaded.values) - {"drifter", "drifter_interp"}:
        raise ValueError(f"Unsupported top-level MATLAB fields: {sorted(set(loaded.values) - {'drifter', 'drifter_interp'})}")
    native = loaded.values.get("drifter")
    interp = loaded.values.get("drifter_interp", {})
    if not isinstance(native, dict) or not isinstance(interp, dict):
        raise ValueError("Expected scalar drifter and drifter_interp structures")
    required = {"PlatformId", "type", "time", "longitude", "latitude"}
    if required - set(native):
        raise ValueError(f"Missing native fields: {sorted(required - set(native))}")
    platform = _identifier(native["PlatformId"], "drifter.PlatformId")
    instrument = str(native["type"])
    if instrument != "MicroSVP":
        raise ValueError(f"Expected MicroSVP, found {instrument!r}")
    time_qc = _vector(native["time"], "drifter.time", missing_value)
    time_interp = _vector(interp.get("time", []), "drifter_interp.time", missing_value)
    if interp and "time" not in interp:
        raise ValueError("drifter_interp has variables but no time axis")
    n, m = len(time_qc), len(time_interp)
    loss = normalize_missing(native.get("drogue_off", np.nan), missing_value)
    if loss.size != 1:
        raise ValueError("drifter.drogue_off must be scalar")
    loss = float(loss.item())
    loss_time = matlab_datenum_to_datetime64(loss)[()]
    warnings: list[str] = []
    metadata = {
        "source_filename": path.name, "source_sha256": loaded.sha256,
        "source_matlab_header": loaded.header, "instrument_type": instrument, "time_reference": "UTC",
        "source_id": _identifier(native["ID"], "drifter.ID") if "ID" in native else "",
        "drogue_off_matlab": loss, "drogue_off_time": loss_time,
        "source_schema_json": json.dumps(_schema(loaded.values), sort_keys=True),
        "z": 0.0, "nominal_drogue_depth_m": 2.5,
    }
    if metadata["source_id"] and metadata["source_id"] != platform:
        warnings.append("ID_differs_from_PlatformId")
    if platform not in path.stem:
        warnings.append("filename_does_not_contain_PlatformId")
    if np.isnat(loss_time):
        warnings.append("unknown_source_drogue_loss")
        if np.isfinite(loss):
            warnings.append("unrepresentable_source_drogue_loss")
    variables: dict[str, np.ndarray] = {}
    attributes: dict[str, dict] = {}
    unaligned = {}
    for source, target in NATIVE_FIELDS.items():
        attributes[target] = field_attributes("drifter", source, target)
        if source not in native:
            variables[target] = np.full(n, np.nan)
            warnings.append(f"missing_field:drifter.{source}")
            continue
        values = _vector(native[source], f"drifter.{source}", missing_value)
        if len(values) != n:
            if source != "battery":
                raise ValueError(f"drifter.{source}: length {len(values)} does not match time length {n}")
            unaligned["battery"] = values
            variables[target] = np.full(n, np.nan)
            warnings.append(f"unaligned_battery:values={len(values)},timestamps={n}")
        else:
            variables[target] = values
    _extra_fields("drifter", native, set(NATIVE_FIELDS) | NATIVE_METADATA_FIELDS,
                  n, variables, metadata, attributes, warnings, missing_value)
    qc = ObservationSeries.from_matlab(time_qc, variables, attributes)
    variables, attributes = {}, {}
    for suffix in ("30min", "60min"):
        pair = {f"latitude_{suffix}", f"longitude_{suffix}"}
        if len(pair & set(interp)) == 1:
            raise ValueError(f"Incomplete reconstructed position pair: {suffix}")
        metadata[f"has_interp_{suffix.replace('min', 'm')}"] = pair <= set(interp)
    for source, target in INTERP_FIELDS.items():
        attributes[target] = field_attributes("drifter_interp", source, target)
        values = (_vector(interp[source], f"drifter_interp.{source}", missing_value)
                  if source in interp else np.full(m, np.nan))
        if len(values) != m:
            raise ValueError(f"drifter_interp.{source}: length {len(values)} does not match time length {m}")
        variables[target] = values
        if source not in interp:
            warnings.append(f"missing_field:drifter_interp.{source}")
    _extra_fields("drifter_interp", interp, set(INTERP_FIELDS) | {"time"},
                  m, variables, metadata, attributes, warnings, missing_value)
    reconstructed = ObservationSeries.from_matlab(time_interp, variables, attributes)
    metadata["battery_unaligned_count"] = len(unaligned.get("battery", []))
    record = TrajectoryRecord(platform, metadata, {"qc": qc, "interp": reconstructed},
                              {"qc": "qc", "interp_30m": "interp", "interp_60m": "interp"},
                              unaligned, warnings)
    record.metadata_attributes = {
        "z": {"units": "m", "positive": "down", "long_name": "Surface position depth (fixed approximation)"},
        "nominal_drogue_depth_m": {"units": "m", "positive": "down", "long_name": "Nominal drogue depth from campaign guide"},
        "drogue_off_time": {"time_reference": "UTC", "source_field": "drifter.drogue_off", "long_name": "Source estimated loss date; uncertainty approximately 1-2 days"},
    }
    record.unaligned_attributes = {"battery": {"source_field": "drifter.battery"}}
    record.validate()
    return record
