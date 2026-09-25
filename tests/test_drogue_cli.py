from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
import yaml

from drifterlab.cli.drogue import main
from drifterlab.io.drogue import READERS, RawDrogueSignals
from drifterlab.review.drogue import DrogueLossReviews
from drifterlab.workflows.drogue import (
    load_drogue_config,
    run_drogue_detection,
    run_drogue_workflow,
)


def _signals(platform: str, path: Path, *, event: bool) -> RawDrogueSignals:
    length = 600
    time = pd.date_range("2025-01-01", periods=length, freq="h").to_numpy()
    if event:
        ttff = np.r_[
            np.resize(np.array([22., 28., 35., 80., 100.]), 240),
            np.resize(np.array([5., 6., 7.]), length - 240),
        ]
        strain = np.r_[np.full(264, 12.), np.full(length - 264, 4.)]
    else:
        ttff = np.full(length, 5.)
        strain = np.full(length, 12.)
    content = path.read_bytes() if path.exists() else platform.encode()
    return RawDrogueSignals(
        platform, time, ttff, strain, path.resolve(), sha256(content).hexdigest(),
    )


def _write_track(path: Path, platform: int, *, event: bool) -> None:
    signals = _signals(str(platform), path, event=event)
    track = {
        "PlatformId": platform,
        "ObsTimestamp": pd.DatetimeIndex(signals.time).strftime(
            "%Y-%m-%d %H:%M:%S"
        ).to_numpy(),
        "GpsTTFF": signals.ttff,
        "Drogue": signals.strain,
    }
    savemat(path, {"dataset": {f"drifter_{platform}": track}})


def _config(tmp_path: Path, *, reader: str = "microsvp_mat") -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    if reader == "microsvp_mat":
        _write_track(raw / "resolved.mat", 1001, event=True)
        _write_track(raw / "unresolved.mat", 1002, event=False)
        pattern = "*.mat"
    else:
        (raw / "1001.raw").write_text("1001", encoding="utf-8")
        pattern = "*.raw"
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "input": {"reader": reader, "directory": str(raw), "pattern": pattern},
        "output": {
            "automatic": str(tmp_path / "auto.parquet"),
            "review": str(tmp_path / "review.csv"),
        },
        "processing": {
            "missing_value": -999,
            "analysis_cutoff_margin_hours": 24,
        },
        "detection": {},
    }), encoding="utf-8")
    return config


class FakeReviewer:
    calls: list[dict] = []

    def __init__(self, config_path, *, platform_codes=None):
        self.calls.append({
            "config": str(config_path),
            "platform_codes": None if platform_codes is None else list(platform_codes),
        })

    def show(self):
        self.calls[-1]["shown"] = True


def _save_uncertain_reviews(config: Path, automatic: pd.DataFrame) -> None:
    workflow_config = load_drogue_config(config)
    reviews = DrogueLossReviews(workflow_config.review_output)
    for row in automatic.itertuples():
        reviews.set(
            platform_code=str(row.platform_code),
            ttff_change_time=row.ttff_change_time,
            strain_change_time=row.strain_change_time,
            auto_drogue_loss_time=row.auto_drogue_loss_time,
            auto_status=row.auto_status,
            auto_source=row.auto_source,
            review_status="uncertain",
            reviewed_drogue_loss_time=None,
            review_reason="uncertain",
            source_sha256=row.source_sha256,
        )
    reviews.save()


def test_generic_cli_automatic_uses_configured_shared_output(tmp_path):
    config = _config(tmp_path)
    assert main([str(config), "--automatic"]) == 0
    table = pd.read_parquet(tmp_path / "auto.parquet")
    assert set(table.platform_code.astype(str)) == {"1001", "1002"}
    assert not (tmp_path / "review.csv").exists()
    assert not list(tmp_path.glob("auto_*.parquet"))


def test_semiautomatic_reuses_shared_output_and_only_reviews_problems(tmp_path):
    config = _config(tmp_path)
    FakeReviewer.calls.clear()
    first = run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == [{
        "config": str(config), "platform_codes": ["1002"], "shown": True,
    }]
    unresolved = first[first.platform_code.astype(str) == "1002"]
    _save_uncertain_reviews(config, unresolved)

    FakeReviewer.calls.clear()
    messages: list[str] = []
    second = run_drogue_workflow(
        config, "semiautomatic", progress=messages.append,
        reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == []
    assert any("Reusing automatic results" in message for message in messages)
    assert list(second.columns) == list(first.columns)


def test_manual_and_semiautomatic_share_automatic_and_review_files(tmp_path):
    config = _config(tmp_path)
    automatic = run_drogue_workflow(
        config, "manual", reviewer_factory=FakeReviewer,
    )
    _save_uncertain_reviews(config, automatic)
    FakeReviewer.calls.clear()
    messages: list[str] = []
    run_drogue_workflow(
        config, "manual", progress=messages.append, reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == []
    assert any("No unfinished manual" in message for message in messages)
    assert (tmp_path / "auto.parquet").exists()
    assert (tmp_path / "review.csv").exists()
    assert not list(tmp_path.glob("auto_*.parquet"))
    assert not list(tmp_path.glob("review_*.csv"))


def test_automatic_requires_overwrite(tmp_path):
    config = _config(tmp_path)
    first = run_drogue_workflow(config, "automatic")
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_drogue_workflow(config, "automatic")
    second = run_drogue_workflow(config, "automatic", overwrite=True)
    assert list(first.columns) == list(second.columns)


def test_reused_incompatible_result_requires_overwrite(tmp_path):
    config = _config(tmp_path)
    pd.DataFrame({"platform_code": ["1001"]}).to_parquet(
        tmp_path / "auto.parquet", index=False,
    )
    with pytest.raises(ValueError, match="use --overwrite"):
        run_drogue_workflow(config, "manual", reviewer_factory=FakeReviewer)


def test_stale_semiautomatic_review_returns_to_queue(tmp_path):
    config = _config(tmp_path)
    automatic = run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    _save_uncertain_reviews(config, automatic)
    changed = automatic.copy()
    changed.loc[changed.platform_code.astype(str) == "1001", "ttff_change_time"] += (
        pd.Timedelta(hours=6)
    )
    changed.to_parquet(tmp_path / "auto.parquet", index=False)

    FakeReviewer.calls.clear()
    run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == [{
        "config": str(config), "platform_codes": ["1001"], "shown": True,
    }]


def test_fake_reader_drives_generic_pipeline_and_reviewer(tmp_path, monkeypatch):
    config = _config(tmp_path, reader="fake")
    calls: list[Path] = []

    def fake_reader(path, *, missing_value=-999):
        path = Path(path)
        calls.append(path)
        return _signals("1001", path, event=True)

    monkeypatch.setitem(READERS, "fake", fake_reader)
    table = run_drogue_detection(config)
    assert table.platform_code.tolist() == ["1001"]
    assert calls == [tmp_path / "raw" / "1001.raw"]

    import matplotlib
    matplotlib.use("Agg", force=True)
    from drifterlab.review.drogue import DrogueLossReviewer

    reviewer = DrogueLossReviewer(config)
    assert len(calls) == 2
    reviewer.action("uncertain")
    assert reviewer.closed


def test_mode_choice_and_unsupported_reader_errors(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(SystemExit) as missing:
        main([str(config)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as multiple:
        main([str(config), "--automatic", "--mode", "manual"])
    assert multiple.value.code == 2

    unsupported = tmp_path / "unsupported.yml"
    unsupported.write_text(yaml.safe_dump({
        "input": {
            "reader": "unknown", "directory": str(tmp_path), "pattern": "*.mat",
        },
        "output": {
            "automatic": str(tmp_path / "other.parquet"),
            "review": str(tmp_path / "other.csv"),
        },
        "processing": {},
        "detection": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported drogue input reader 'unknown'"):
        load_drogue_config(unsupported)
