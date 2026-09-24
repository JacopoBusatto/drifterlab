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
        "Drogue": np.r_[np.full(264, 12.), np.full(n - 264, 4.)],
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
    }), encoding="utf-8")
    signals = read_raw_drogue_signals(raw_path)
    assert signals.platform_code == "1001"
    assert len(signals.time) == 600
    assert detect_main([str(config)]) == 0
    table = pd.read_parquet(automatic)
    assert table.platform_code.astype(str).tolist() == ["1001"]
    assert table.auto_status.tolist() == ["clear_agreement"]
    assert table.auto_source.tolist() == ["ttff+strain"]
    assert pd.notna(table.auto_drogue_loss_time.iloc[0])
    effective = json.loads(table.detection_config_json.iloc[0])
    assert effective["ttff"]["binning"]["scale"] == "log"
    assert effective["ttff"]["comparison"]["pre_activity_window_hours"] == 168
    assert "method" not in effective["strain"]
    assert effective["strain"]["aggregation_hours"] == 6
    assert effective["strain"]["minimum_drop_absolute"] == 2.0
    assert effective["combination"]["agreement_tolerance_hours"] == 48
    assert "window" not in effective["strain"]
    assert table.ttff_agreeing_bin_count.iloc[0] >= 2
    assert "ttff_w1" not in table.columns
    assert table.default_analysis_cutoff_margin_hours.tolist() == [24]
    assert (
        table.default_analysis_cutoff_time.iloc[0]
        == table.auto_drogue_loss_time.iloc[0] - pd.Timedelta(hours=24)
    )
    assert not review.exists()

    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer
    reviewer = DrogueLossReviewer(config)
    from matplotlib.collections import QuadMesh

    assert reviewer.ttff_ax.get_yscale() == "log"
    assert not any(isinstance(item, QuadMesh) for item in reviewer.ttff_ax.collections)
    assert not hasattr(reviewer, "temperature_ax")
    assert not hasattr(reviewer, "ttff_count_ax")
    assert reviewer.display_cache["1001"][0].hull_temperature is None
    cached_signals = reviewer.display_cache["1001"][0]
    reviewer.loaded_platform = None
    reviewer.draw()
    assert reviewer.display_cache["1001"][0] is cached_signals
    base_ttff_line = reviewer.ttff_ax.lines[0]
    reviewer.on_margin_submit("18")
    assert reviewer.ttff_ax.lines[0] is base_ttff_line
    reviewer.action("accept")
    assert reviewer.closed
    saved = pd.read_csv(review, dtype=str, keep_default_na=False)
    assert saved.review_status.tolist() == ["accepted_auto"]
    assert saved.auto_status.tolist() == ["clear_agreement"]
    assert saved.auto_source.tolist() == ["ttff+strain"]
    assert saved.analysis_cutoff_margin_hours.tolist() == ["18"]


def test_manual_review_persistence_loading_and_cutoff(tmp_path):
    path = tmp_path / "review/drogue_review.csv"
    event = pd.Timestamp("2025-02-10T12:00:00Z")
    reviews = DrogueLossReviews(path, cutoff_margin_hours=24)
    reviews.set(
        platform_code="1001", auto_drogue_loss_time=event,
        review_status="accepted_auto", reviewed_drogue_loss_time=None,
        review_reason="automatic_agreement", source_sha256="abc123",
        ttff_change_time=event, strain_change_time=event,
        auto_status="clear_agreement", auto_source="ttff+strain",
    )
    reviews.set(
        platform_code="1002", auto_drogue_loss_time=pd.NaT,
        review_status="not_lost", reviewed_drogue_loss_time=None,
        review_reason="not_lost", source_sha256="def456",
        auto_status="unresolved", auto_source="none",
    )
    reviews.save()
    loaded = DrogueLossReviews(path, cutoff_margin_hours=24)
    table = loaded.table().set_index("platform_code")
    assert table.loc["1001", "reviewed_drogue_loss_time"] == event.isoformat()
    assert table.loc["1001", "analysis_cutoff_time"] == (event - pd.Timedelta(hours=24)).isoformat()
    assert table.loc["1002", "reviewed_drogue_loss_time"] == ""
    automatic = pd.DataFrame({
        "platform_code": ["1001", "1002"],
        "ttff_change_time": [event, pd.NaT],
        "strain_change_time": [event, pd.NaT],
        "auto_drogue_loss_time": [event, pd.NaT],
        "auto_status": ["clear_agreement", "unresolved"],
        "auto_source": ["ttff+strain", "none"],
        "source_sha256": ["abc123", "def456"],
    })
    loaded.validate_against_automatic(automatic)
    stale = automatic.copy()
    stale.loc[0, "source_sha256"] = "changed"
    with pytest.raises(ValueError, match="Raw source changed"):
        loaded.validate_against_automatic(stale)


def test_custom_margin_changes_only_cutoff_and_resumes(tmp_path):
    path = tmp_path / "review.csv"
    event = pd.Timestamp("2025-02-10T12:00:00Z")
    reviews = DrogueLossReviews(path)
    reviews.set(
        platform_code="1001", auto_drogue_loss_time=pd.NaT,
        review_status="manual_date", reviewed_drogue_loss_time=event,
        review_reason="manual_adjustment", source_sha256="abc",
        auto_status="unresolved", auto_source="none",
        analysis_cutoff_margin_hours=36,
    )
    original_reviewed = reviews.rows["1001"]["reviewed_drogue_loss_time"]
    original_auto = reviews.rows["1001"]["auto_drogue_loss_time"]
    reviews.set_margin("1001", 12)
    assert reviews.rows["1001"]["reviewed_drogue_loss_time"] == original_reviewed
    assert reviews.rows["1001"]["auto_drogue_loss_time"] == original_auto
    assert reviews.rows["1001"]["analysis_cutoff_time"] == (
        event - pd.Timedelta(hours=12)
    ).isoformat()
    reviews.save()
    resumed = DrogueLossReviews(path)
    assert resumed.rows["1001"]["analysis_cutoff_margin_hours"] == "12"


@pytest.mark.parametrize("status,reason", [("not_lost", "not_lost"), ("uncertain", "uncertain")])
def test_not_lost_and_uncertain_are_distinct_and_have_no_cutoff(tmp_path, status, reason):
    reviews = DrogueLossReviews(tmp_path / f"{status}.csv")
    reviews.set(
        platform_code="1001", auto_drogue_loss_time=pd.NaT,
        review_status=status, reviewed_drogue_loss_time=None,
        review_reason=reason, source_sha256="abc",
        auto_status="unresolved", auto_source="none",
    )
    row = reviews.rows["1001"]
    assert row["review_status"] == status
    assert row["reviewed_drogue_loss_time"] == ""
    assert row["analysis_cutoff_time"] == ""


def test_detector_rerun_marks_review_stale_without_overwriting_it(tmp_path):
    event = pd.Timestamp("2025-02-10T12:00:00Z")
    reviews = DrogueLossReviews(tmp_path / "review.csv")
    reviews.set(
        platform_code="1001", auto_drogue_loss_time=event,
        review_status="accepted_auto", reviewed_drogue_loss_time=None,
        review_reason="automatic_agreement", source_sha256="abc",
        ttff_change_time=event, strain_change_time=event,
        auto_status="clear_agreement", auto_source="ttff+strain",
    )
    before = dict(reviews.rows["1001"])
    changed = pd.DataFrame({
        "platform_code": ["1001"], "source_sha256": ["abc"],
        "ttff_change_time": [event + pd.Timedelta(hours=6)],
        "strain_change_time": [event],
        "auto_drogue_loss_time": [event + pd.Timedelta(hours=6)],
        "auto_status": ["clear_agreement"], "auto_source": ["ttff+strain"],
    })
    assert reviews.validate_against_automatic(changed) == {"1001"}
    assert reviews.rows["1001"] == before


def test_reviewer_can_set_manual_date_when_automatic_date_is_missing(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    source = tmp_path / "raw"
    source.mkdir()
    n = 600
    time = pd.date_range("2025-01-01", periods=n, freq="h")
    track = {
        "PlatformId": 1001,
        "ObsTimestamp": time.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsTTFF": np.full(n, 5.0),
        "Drogue": np.full(n, 12.0),
    }
    savemat(source / "stable.mat", {"dataset": {"drifter_1001": track}})
    automatic = tmp_path / "auto.parquet"
    review = tmp_path / "review.csv"
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "experiment": "arcterx",
        "input": {"directory": str(source), "pattern": "*.mat"},
        "output": {"automatic": str(automatic), "review": str(review)},
        "processing": {"missing_value": -999, "analysis_cutoff_margin_hours": 24},
        "detection": {},
    }), encoding="utf-8")
    assert detect_main([str(config)]) == 0
    automatic_row = pd.read_parquet(automatic).iloc[0]
    assert pd.isna(automatic_row.auto_drogue_loss_time)

    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer

    reviewer = DrogueLossReviewer(config)
    assert not reviewer.buttons[0].active
    selected = reviewer.selected_time
    reviewer.on_margin_submit("30")
    reviewer.action("manual")
    decision = reviewer.reviews.rows["1001"]
    assert decision["review_status"] == "manual_date"
    assert pd.Timestamp(decision["reviewed_drogue_loss_time"]) == selected
    assert pd.Timestamp(decision["analysis_cutoff_time"]) == selected - pd.Timedelta(hours=30)
    assert reviewer.closed
    assert review.exists()


def test_reviewer_skips_forward_then_wraps_to_remaining_pending_platform(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    source = tmp_path / "raw"
    source.mkdir()
    for platform in (1001, 1002, 1003):
        write_raw(source / f"{platform}.mat", str(platform))
    automatic = tmp_path / "auto.parquet"
    review = tmp_path / "review.csv"
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "experiment": "arcterx",
        "input": {"directory": str(source), "pattern": "*.mat"},
        "output": {"automatic": str(automatic), "review": str(review)},
        "processing": {"missing_value": -999, "analysis_cutoff_margin_hours": 24},
        "detection": {},
    }), encoding="utf-8")
    assert detect_main([str(config)]) == 0

    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer

    reviewer = DrogueLossReviewer(config)
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1001"
    reviewer.action("next")
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1002"
    reviewer.action("uncertain")
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1003"
    reviewer.action("previous")
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1002"
    reviewer.on_margin_submit("18")
    persisted = pd.read_csv(review, dtype=str).set_index("platform_code")
    assert persisted.loc["1002", "analysis_cutoff_margin_hours"] == "18"
    reviewer.action("next")
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1003"
    reviewer.action("uncertain")
    assert not reviewer.closed
    assert str(reviewer.automatic.iloc[reviewer.cursor].platform_code) == "1001"
    reviewer.action("uncertain")
    assert reviewer.closed
    saved = pd.read_csv(review)
    assert set(saved.platform_code.astype(str)) == {"1001", "1002", "1003"}


def test_nudge_is_a_draft_and_save_failure_does_not_advance(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    source = tmp_path / "raw"
    source.mkdir()
    write_raw(source / "1001.mat", "1001")
    automatic = tmp_path / "auto.parquet"
    review = tmp_path / "review.csv"
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "experiment": "arcterx",
        "input": {"directory": str(source), "pattern": "*.mat"},
        "output": {"automatic": str(automatic), "review": str(review)},
        "processing": {"missing_value": -999, "analysis_cutoff_margin_hours": 24},
        "detection": {},
    }), encoding="utf-8")
    assert detect_main([str(config)]) == 0

    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer

    reviewer = DrogueLossReviewer(config)
    original = reviewer.selected_time
    reviewer.action("+1")
    assert reviewer.selected_time == original + pd.Timedelta(hours=1)
    assert reviewer.reviews.rows == {}
    assert not review.exists()

    monkeypatch.setattr(
        reviewer.reviews, "save",
        lambda: (_ for _ in ()).throw(ValueError("concurrent edit")),
    )
    reviewer.action("manual")
    assert reviewer.cursor == 0
    assert not reviewer.closed
    assert reviewer.reviews.rows == {}
    assert "Could not save decision" in reviewer.status.get_text()
    reviewer.closed = True
    reviewer.plt.close(reviewer.figure)


def test_legacy_review_schema_migrates_and_infers_its_margin(tmp_path):
    path = tmp_path / "legacy.csv"
    event = pd.Timestamp("2025-02-10T12:00:00Z")
    legacy = pd.DataFrame([{
        "platform_code": "1001",
        "auto_drogue_loss_time": event.isoformat(),
        "review_status": "manual_time",
        "reviewed_drogue_loss_time": event.isoformat(),
        "analysis_cutoff_time": (event - pd.Timedelta(hours=30)).isoformat(),
        "review_reason": "manual_adjustment",
        "review_timestamp": pd.Timestamp("2025-02-11T00:00:00Z").isoformat(),
        "source_sha256": "abc",
    }])
    legacy.to_csv(path, index=False)
    loaded = DrogueLossReviews(path, cutoff_margin_hours=24)
    row = loaded.rows["1001"]
    assert row["review_status"] == "manual_date"
    assert row["analysis_cutoff_margin_hours"] == "30"
