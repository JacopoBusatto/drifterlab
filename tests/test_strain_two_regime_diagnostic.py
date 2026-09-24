from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
import yaml

from scripts.diagnostics.strain_two_regime import run


def test_selected_platform_diagnostic_writes_only_experimental_outputs(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    length, event = 480, 240
    timestamps = pd.date_range("2025-01-01", periods=length, freq="h")
    track = {
        "PlatformId": 1001,
        "ObsTimestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsTTFF": np.full(length, 10.0),
        "Drogue": np.r_[np.full(event, 20.0), np.full(length - event, 10.0)],
    }
    savemat(raw / "source.mat", {"dataset": {"drifter_1001": track}})
    input_config = tmp_path / "drogue.yml"
    input_config.write_text(yaml.safe_dump({
        "input": {"directory": str(raw), "pattern": "*.mat"},
        "processing": {"missing_value": -999},
    }), encoding="utf-8")
    output = tmp_path / "diagnostic"
    primary, sensitivity = run(
        input_config,
        Path("configs/arcterx/strain_two_regime.yml"),
        output,
        platforms=["1001"],
    )
    assert primary.status.tolist() == ["clear"]
    assert primary.aggregation_hours.tolist() == [6]
    assert sensitivity.aggregation_hours.tolist() == [3, 6, 12]
    assert set(sensitivity.status) == {"clear"}
    assert (output / "strain_two_regime_results.csv").exists()
    assert (output / "strain_two_regime_sensitivity.csv").exists()
    assert (output / "strain_two_regime_settings.json").exists()
    assert (output / "figures/strain_two_regime_1001.png").exists()
    assert not (output / "auto_drogue_loss.parquet").exists()
