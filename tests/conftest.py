from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
import xarray as xr

from drifterlab.experiments.arcterx import read_microsvp
from drifterlab.qc.flags import consecutive_speed
from drifterlab.qc.position import position_valid


@pytest.fixture
def mat_factory(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()

    def make(platform="1001", *, native_changes=None, interp_changes=None, no_interp=False):
        native = {
            "PlatformId": np.int64(platform), "ID": np.int64(platform), "type": "MicroSVP",
            "time": np.array([719532., 719531., 719530.]),
            "longitude": np.array([130.02, 130.01, 130.]), "latitude": np.array([20.02, 20.01, 20.]),
            "SST": np.array([24., 23., 22.]), "SLP": np.array([-999, -999, -999], dtype=np.int16),
            "battery": np.array([6., 7., 8.]), "drogue": np.array([12, 13, -999], dtype=np.int16),
            "speed": np.array([np.nan, .2, .1]), "drogue_off": 719533.,
        }
        interp = {
            "time": np.array([719532., 719531.5, 719531., 719530.5, 719530.]),
            "longitude_30min": np.array([130.04, 130.03, 130.02, 130.01, 130.]),
            "latitude_30min": np.array([20.04, 20.03, 20.02, 20.01, 20.]),
            "longitude_60min": np.array([130.08, 130.06, 130.04, 130.02, 130.]),
            "latitude_60min": np.array([20.08, 20.06, 20.04, 20.02, 20.]),
            "SST": np.array([24., 23.5, 23., 22.5, 22.]),
            "speed_30min": np.array([np.nan, .3, .2, .1, np.nan]),
            "speed_60min": np.array([np.nan, .6, .4, .2, np.nan]),
        }
        for target, changes in ((native, native_changes), (interp, interp_changes)):
            for key, value in (changes or {}).items():
                if value is None:
                    target.pop(key, None)
                else:
                    target[key] = value
        payload = {"drifter": native}
        if not no_interp:
            payload["drifter_interp"] = interp
        path = directory / f"ARCTERYX_microsvp_{platform}.mat"
        savemat(path, payload)
        return path, payload

    return make


@pytest.fixture
def legacy_master_factory(tmp_path):
    """Build the old diagnostic schema without retaining its production writer."""
    def make() -> Path:
        records = [read_microsvp(path) for path in sorted((tmp_path / "input").glob("*.mat"))]
        if not records:
            raise ValueError("No synthetic supplied-QC MAT files exist")
        n_trajectory = len(records)
        n_qc = max(len(record.series["qc"].time) for record in records)
        n_interp = max(len(record.series["interp"].time) for record in records)

        def floats(length):
            return np.full((n_trajectory, length), np.nan, dtype=float)

        def times(length):
            return np.full((n_trajectory, length), np.datetime64("NaT", "ns"))

        def integers(length):
            return np.full((n_trajectory, length), -1, dtype=np.int64)

        variables = {
            "time_qc": (("trajectory", "obs_qc"), times(n_qc)),
            "source_obs_index_qc": (("trajectory", "obs_qc"), integers(n_qc)),
            "lon_qc": (("trajectory", "obs_qc"), floats(n_qc)),
            "lat_qc": (("trajectory", "obs_qc"), floats(n_qc)),
            "speed_qc_source": (("trajectory", "obs_qc"), floats(n_qc)),
            "position_valid_qc": (("trajectory", "obs_qc"), np.zeros((n_trajectory, n_qc), bool)),
            "audit_speed_qc": (("trajectory", "obs_qc"), floats(n_qc)),
            "residual_jump_flag_qc": (("trajectory", "obs_qc"), np.zeros((n_trajectory, n_qc), bool)),
            "time_interp": (("trajectory", "obs_interp"), times(n_interp)),
            "source_obs_index_interp": (("trajectory", "obs_interp"), integers(n_interp)),
        }
        for name in (
            "lon_interp_30m", "lat_interp_30m", "lon_interp_60m", "lat_interp_60m",
        ):
            variables[name] = (("trajectory", "obs_interp"), floats(n_interp))
            variables[f"position_valid_{name.removeprefix('lon_').removeprefix('lat_')}"] = (
                ("trajectory", "obs_interp"), np.zeros((n_trajectory, n_interp), bool),
            )

        platforms, filenames, hashes = [], [], []
        n_obs_qc, n_obs_interp, n_jumps, has30, has60 = [], [], [], [], []
        inventory = []
        for index, record in enumerate(records):
            qc, interp = record.series["qc"], record.series["interp"]
            qn, rn = len(qc.time), len(interp.time)
            valid = position_valid(qc.variables["lon_qc"], qc.variables["lat_qc"])
            speed = consecutive_speed(qc.time, qc.variables["lon_qc"], qc.variables["lat_qc"])
            flags = speed > 3.0
            for name, values in {
                "time_qc": qc.time, "source_obs_index_qc": qc.source_index,
                "lon_qc": qc.variables["lon_qc"], "lat_qc": qc.variables["lat_qc"],
                "speed_qc_source": qc.variables["speed_qc_source"],
                "position_valid_qc": valid, "audit_speed_qc": speed,
                "residual_jump_flag_qc": flags,
            }.items():
                variables[name][1][index, :qn] = values
            for name, values in {
                "time_interp": interp.time, "source_obs_index_interp": interp.source_index,
                "lon_interp_30m": interp.variables["lon_interp_30m"],
                "lat_interp_30m": interp.variables["lat_interp_30m"],
                "lon_interp_60m": interp.variables["lon_interp_60m"],
                "lat_interp_60m": interp.variables["lat_interp_60m"],
            }.items():
                variables[name][1][index, :rn] = values
            for suffix in ("30m", "60m"):
                variables[f"position_valid_interp_{suffix}"][1][index, :rn] = position_valid(
                    interp.variables[f"lon_interp_{suffix}"], interp.variables[f"lat_interp_{suffix}"],
                )
            platforms.append(record.platform_code)
            filenames.append(record.metadata["source_filename"])
            hashes.append(record.metadata["source_sha256"])
            n_obs_qc.append(qn)
            n_obs_interp.append(rn)
            n_jumps.append(int(flags.sum()))
            has30.append(bool(record.metadata["has_interp_30m"]))
            has60.append(bool(record.metadata["has_interp_60m"]))
            finite_time = qc.time[~np.isnat(qc.time)]
            valid_time = qc.time[valid & ~np.isnat(qc.time)]
            inventory.append({
                "platform_code": record.platform_code,
                "source_sha256": record.metadata["source_sha256"],
                "first_time_qc": finite_time.min() if len(finite_time) else np.datetime64("NaT"),
                "first_valid_position_time_qc": valid_time.min() if len(valid_time) else np.datetime64("NaT"),
            })

        dataset = xr.Dataset(
            variables | {
                "platform_code": ("trajectory", np.asarray(platforms)),
                "source_filename": ("trajectory", np.asarray(filenames)),
                "source_sha256": ("trajectory", np.asarray(hashes)),
                "n_obs_qc": ("trajectory", np.asarray(n_obs_qc, dtype=np.int64)),
                "n_obs_interp": ("trajectory", np.asarray(n_obs_interp, dtype=np.int64)),
                "n_residual_jumps_qc": ("trajectory", np.asarray(n_jumps, dtype=np.int64)),
                "has_interp_30m": ("trajectory", np.asarray(has30, dtype=bool)),
                "has_interp_60m": ("trajectory", np.asarray(has60, dtype=bool)),
            },
            coords={"trajectory": np.arange(n_trajectory), "obs_qc": np.arange(n_qc),
                    "obs_interp": np.arange(n_interp)},
            attrs={"residual_jump_speed_m_s": 3.0},
        )
        dataset["residual_jump_flag_qc"].attrs["threshold_m_s"] = 3.0
        destination = tmp_path / "out/master.zarr"
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_zarr(destination, mode="w", consolidated=True, zarr_format=2)
        pd.DataFrame(inventory).to_parquet(tmp_path / "out/inventory.parquet", index=False)
        return destination

    return make
