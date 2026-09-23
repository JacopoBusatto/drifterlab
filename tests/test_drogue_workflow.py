import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
import yaml

from drifterlab.cli.detect_arcterx_drogue import main as detect_main
from drifterlab.experiments.arcterx.drogue_loss_review import DrogueLossReviews
from drifterlab.experiments.arcterx.raw_drogue import read_raw_drogue_signals


def write_raw(path: Path, platform="1001"):
    n = 600
    time = pd.date_range("2025-01-01", periods=n, freq="h")
    before = np.resize(np.array([22., 28., 35., 80., 31., 100.]), 240)
    after = np.resize(np.array([5., 6., 7.]), n - len(before))
    track = {
        "PlatformId": int(platform),
        "ObsTimestamp": time.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsTTFF": np.r_[before, after],
        "Drogue": np.r_[np.full(300, 12.), np.full(n - 300, 4.)],
        "HullTemperature": 20 + np.sin(np.arange(n) / 12),
    }
    savemat(path, {"dataset": {f"drifter_{platform}": track}})


def test_raw_adapter_and_automatic_cli_smoke(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    raw_path = source / "unrelated_name.mat"
    write_raw(raw_path)
    automatic = tmp_path / "data/qc/drogue/auto.parquet"
    review = tmp_path / "data/review/drogue/review.csv"
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "experiment": "arcterx",
        "input": {"directory": str(source), "pattern": "*.mat"},
        "output": {"automatic": str(automatic), "review": str(review)},
        "processing": {"missing_value": -999, "analysis_cutoff_margin_hours": 24},
        "detection": {},
        "review": {"zoom_window": "3D"},
    }), encoding="utf-8")
    signals = read_raw_drogue_signals(raw_path)
    assert signals.platform_code == "1001"
    assert len(signals.time) == 600
    assert detect_main([str(config)]) == 0
    table = pd.read_parquet(automatic)
    assert table.platform_code.astype(str).tolist() == ["1001"]
    assert table.auto_status.tolist() == ["detected_strain_primary_corroborated"]
    assert pd.notna(table.auto_drogue_loss_time.iloc[0])
    effective = json.loads(table.detection_config_json.iloc[0])
    assert effective["ttff"]["binning"]["scale"] == "log"
    assert effective["ttff"]["comparison"]["pre_activity_window_hours"] == 168
    assert "tail" not in effective["ttff"]
    assert "wasserstein_threshold" not in effective["ttff"]["comparison"]
    assert effective["strain"]["window"] == "12h"
    assert effective["strain"]["minimum_drop_absolute"] == 2.0
    assert "binning" not in effective["strain"]
    assert table.ttff_agreeing_bin_count.iloc[0] >= 2
    assert "ttff_w1" not in table.columns
    assert not review.exists()

    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer
    reviewer = DrogueLossReviewer(config)
    reviewer.action("accept")
    reviewer.action("save")
    reviewer.plt.close(reviewer.figure)
    saved = pd.read_csv(review, dtype=str, keep_default_na=False)
    assert saved.review_status.tolist() == ["accepted_auto"]


def test_manual_review_persistence_loading_and_cutoff(tmp_path):
    path = tmp_path / "review/drogue_review.csv"
    event = pd.Timestamp("2025-02-10T12:00:00Z")
    reviews = DrogueLossReviews(path, cutoff_margin_hours=24)
    reviews.set(
        platform_code="1001", auto_drogue_loss_time=event,
        review_status="accepted_auto", reviewed_drogue_loss_time=None,
        review_reason="ttff_strain_agree", source_sha256="abc123",
    )
    reviews.set(
        platform_code="1002", auto_drogue_loss_time=pd.NaT,
        review_status="no_detectable_loss", reviewed_drogue_loss_time=None,
        review_reason="ambiguous", source_sha256="def456",
    )
    reviews.save()
    loaded = DrogueLossReviews(path, cutoff_margin_hours=24)
    table = loaded.table().set_index("platform_code")
    assert table.loc["1001", "reviewed_drogue_loss_time"] == event.isoformat()
    assert table.loc["1001", "analysis_cutoff_time"] == (event - pd.Timedelta(hours=24)).isoformat()
    assert table.loc["1002", "reviewed_drogue_loss_time"] == ""
    automatic = pd.DataFrame({
        "platform_code": ["1001", "1002"],
        "auto_drogue_loss_time": [event, pd.NaT],
        "source_sha256": ["abc123", "def456"],
    })
    loaded.validate_against_automatic(automatic)
    stale = automatic.copy()
    stale.loc[0, "source_sha256"] = "changed"
    with pytest.raises(ValueError, match="Raw source changed"):
        loaded.validate_against_automatic(stale)
