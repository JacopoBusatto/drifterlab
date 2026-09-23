from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
import yaml


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
def config_factory(tmp_path):
    def make(*, processing=None, validation=None, suffix=""):
        config = {
            "experiment": "arcterx", "input": {"directory": "input", "pattern": "*.mat"},
            "output": {"zarr": f"out/master{suffix}.zarr", "inventory": f"out/inventory{suffix}.parquet", "chunks": {"trajectory": 1, "obs": 2}},
            "processing": processing or {}, "validation": validation or {},
        }
        path = tmp_path / f"config{suffix}.yml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path
    return make
