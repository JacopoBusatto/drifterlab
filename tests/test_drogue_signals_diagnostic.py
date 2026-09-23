from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
import yaml

from drifterlab.experiments.arcterx.raw_drogue import RawDrogueSignals
from scripts.diagnostics.drogue_signals import (
    DiagnosticSettings, main, plot_drifter, rolling_diagnostics, summarize_drifter,
)
from drifterlab.qc.drogue import DrogueDetectionConfig, detect_drogue_loss


def signals(ttff, *, time=None, strain=None, temperature=None):
    if time is None:
        time = np.datetime64("2025-01-01") + np.arange(len(ttff)) * np.timedelta64(1, "h")
    return RawDrogueSignals(
        "1001", np.asarray(time, dtype="datetime64[ns]"), np.asarray(ttff, dtype=float),
        None if strain is None else np.asarray(strain, dtype=float),
        None if temperature is None else np.asarray(temperature, dtype=float),
        Path("raw.mat"), "abc",
    )


def settings(**changes):
    values = {
        "rolling_window": "2h", "strain_pre_window": "2h", "strain_post_window": "2h",
        "temperature_background_window": "2h", "temperature_variability_window": "2h",
        "minimum_observations": 1, "ttff_thresholds": (20., 40., 60., 120.),
    }
    values.update(changes)
    return DiagnosticSettings(**values)


def test_rolling_excursion_counts_fractions_and_exact_4095():
    data = rolling_diagnostics(signals([10, 30, 4095]), settings())
    assert data.rolling_count_ttff_gt_20.tolist() == [0, 1, 2]
    assert data.rolling_fraction_ttff_gt_20.tolist() == [0, .5, 1]
    assert data.rolling_count_ttff_4095.tolist() == [0, 0, 1]
    assert data.rolling_fraction_ttff_4095.tolist() == [0, 0, .5]
    summary = summarize_drifter(signals([10, 30, 4095]), data, settings())
    assert summary["count_ttff_4095"] == 1
    assert summary["fraction_ttff_4095"] == 1 / 3


def test_irregular_timestamps_use_elapsed_time_not_observation_count():
    time = np.array(["2025-01-01T00", "2025-01-01T01", "2025-01-01T05"],
                    dtype="datetime64[h]")
    data = rolling_diagnostics(signals([10, 30, 4095], time=time), settings())
    # At 05:00 the two earlier observations are outside the two-hour window.
    assert data.rolling_count_ttff_4095.iloc[-1] == 1
    assert data.rolling_fraction_ttff_4095.iloc[-1] == 1
    assert data.rolling_median_ttff.iloc[-1] == 4095


def test_missing_strain_and_temperature_are_explicit():
    source = signals([10, 20, 30])
    data = rolling_diagnostics(source, settings())
    assert data.rolling_median_strain.isna().all()
    assert data.strain_delta.isna().all()
    assert data.rolling_temperature_background.isna().all()
    assert data.rolling_temperature_mad.isna().all()
    summary = summarize_drifter(source, data, settings())
    assert summary["strain_missing"]
    assert summary["hull_temperature_missing"]


def test_drifter_figure_marks_all_detected_dates_on_all_panels(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    n, ttff_event, strain_event = 700, 240, 300
    before = np.resize(np.array([10., 30., 80., 150., 400.]), ttff_event)
    after = np.resize(np.array([2., 4., 6., 8.]), n - ttff_event)
    source = signals(
        np.r_[before, after],
        strain=np.r_[
            np.resize(np.array([18., 20., 22., 24.]), strain_event),
            np.resize(np.array([8., 10., 12., 14.]), n - strain_event),
        ],
        temperature=20 + np.sin(np.arange(n) / 12),
    )
    diagnostic_settings = settings(rolling_window="12h", minimum_observations=6)
    data = rolling_diagnostics(source, diagnostic_settings)
    detection = detect_drogue_loss(
        source.platform_code, source.time, source.ttff, strain=source.strain,
        hull_temperature=source.hull_temperature,
        config=DrogueDetectionConfig(),
    )
    labels = []
    original = Axes.axvline

    def record_line(axis, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "axvline", record_line)
    output = tmp_path / "drifter.png"
    plot_drifter(source, data, diagnostic_settings, output, detection)
    assert output.exists()
    assert labels.count("TTFF change") == 7
    assert labels.count("Strain change / Automatic drogue loss") == 7


def test_single_platform_cli_smoke(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)

    raw = tmp_path / "raw"
    raw.mkdir()
    n = 12
    timestamps = pd.date_range("2025-01-01", periods=n, freq="h")
    track = {
        "PlatformId": 1001,
        "ObsTimestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsTTFF": np.array([10, 20, 30, 4095, 40, 60, 120, 300, 10, 20, 30, 40]),
        "Drogue": np.r_[np.full(6, 12), np.full(6, 4)],
        "HullTemperature": 20 + np.sin(np.arange(n)),
    }
    savemat(raw / "not_the_platform_id.mat", {"dataset": {"drifter_1001": track}})
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({
        "input": {"directory": str(raw), "pattern": "*.mat"},
        "processing": {"missing_value": -999},
    }), encoding="utf-8")
    output = tmp_path / "diagnostic"
    assert main([
        str(config), "--output", str(output), "--platform", "1001",
        "--window", "3h", "--strain-pre-window", "3h", "--strain-post-window", "3h",
        "--temperature-background-window", "3h", "--temperature-variability-window", "3h",
        "--minimum-observations", "1", "--no-population-figures",
    ]) == 0
    table = pd.read_parquet(output / "ttff_population_summary.parquet")
    assert table.platform_code.astype(str).tolist() == ["1001"]
    assert table.count_ttff_4095.tolist() == [1]
    assert (output / "figures/drogue_signals_1001.png").exists()
    assert (output / "population_summary.json").exists()
    assert (output / "raw_variable_metadata.json").exists()
