from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
import yaml

from drifterlab.cli.drogue import main, run_drogue_workflow
from drifterlab.experiments.arcterx.drogue_config import load_drogue_config
from drifterlab.experiments.arcterx.drogue_loss_review import DrogueLossReviews


def _write_track(path: Path, platform: int, *, event: bool) -> None:
    length = 600
    time = pd.date_range("2025-01-01", periods=length, freq="h")
    if event:
        ttff = np.r_[
            np.resize(np.array([22., 28., 35., 80., 100.]), 240),
            np.resize(np.array([5., 6., 7.]), length - 240),
        ]
        strain = np.r_[np.full(264, 12.), np.full(length - 264, 4.)]
    else:
        ttff = np.full(length, 5.)
        strain = np.full(length, 12.)
    track = {
        "PlatformId": platform,
        "ObsTimestamp": time.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsTTFF": ttff,
        "Drogue": strain,
        "HullTemperature": np.full(length, 20.),
    }
    savemat(path, {"dataset": {f"drifter_{platform}": track}})


def _config(tmp_path: Path, *, experiment: str = "arcterx") -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_track(raw / "resolved.mat", 1001, event=True)
    _write_track(raw / "unresolved.mat", 1002, event=False)
    config = tmp_path / "drogue.yml"
    config.write_text(yaml.safe_dump({
        "experiment": experiment,
        "input": {"directory": str(raw), "pattern": "*.mat"},
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

    def __init__(self, config_path, *, mode=None, platform_codes=None):
        self.calls.append({
            "config": str(config_path),
            "mode": mode,
            "platform_codes": None if platform_codes is None else list(platform_codes),
        })

    def show(self):
        self.calls[-1]["shown"] = True


def test_generic_cli_automatic_dispatches_arcterx_without_opening_reviewer(tmp_path):
    config = _config(tmp_path)
    FakeReviewer.calls.clear()
    assert main([str(config), "--automatic"]) == 0
    table = pd.read_parquet(tmp_path / "auto_auto.parquet")
    assert set(table.platform_code.astype(str)) == {"1001", "1002"}
    assert not (tmp_path / "review_auto.csv").exists()
    assert not (tmp_path / "auto.parquet").exists()
    assert FakeReviewer.calls == []


def test_semiautomatic_reviews_only_unresolved_and_creates_no_fake_rows(tmp_path):
    config = _config(tmp_path)
    FakeReviewer.calls.clear()
    first = run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == [{
        "config": str(config), "mode": "semiautomatic",
        "platform_codes": ["1002"], "shown": True,
    }]
    assert not (tmp_path / "review_semi.csv").exists()
    columns = list(pd.read_parquet(tmp_path / "auto_semi.parquet").columns)
    unresolved = first[first.platform_code.astype(str) == "1002"].iloc[0]
    workflow_config = load_drogue_config(config).for_mode("semiautomatic")
    reviews = DrogueLossReviews(workflow_config.review_output)
    reviews.set(
        platform_code="1002",
        ttff_change_time=unresolved.ttff_change_time,
        strain_change_time=unresolved.strain_change_time,
        auto_drogue_loss_time=unresolved.auto_drogue_loss_time,
        auto_status=unresolved.auto_status,
        auto_source=unresolved.auto_source,
        review_status="uncertain",
        reviewed_drogue_loss_time=None,
        review_reason="uncertain",
        source_sha256=unresolved.source_sha256,
    )
    reviews.save()
    FakeReviewer.calls.clear()
    second = run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == []
    saved = pd.read_csv(tmp_path / "review_semi.csv")
    assert saved.platform_code.astype(str).tolist() == ["1002"]
    assert list(second.columns) == columns == list(first.columns)


def test_manual_mode_opens_every_platform_and_uses_same_automatic_product(tmp_path):
    config = _config(tmp_path)
    FakeReviewer.calls.clear()
    table = run_drogue_workflow(config, "manual", reviewer_factory=FakeReviewer)
    assert len(table) == 2
    assert FakeReviewer.calls == [{
        "config": str(config), "mode": "manual",
        "platform_codes": None, "shown": True,
    }]
    assert (tmp_path / "auto_manual.parquet").exists()


def test_modes_have_independent_automatic_outputs_and_reuse_review_results(tmp_path):
    config = _config(tmp_path)
    pd.DataFrame({"legacy": [1]}).to_parquet(tmp_path / "auto.parquet", index=False)
    messages: list[str] = []
    FakeReviewer.calls.clear()
    run_drogue_workflow(
        config, "manual", progress=messages.append, reviewer_factory=FakeReviewer,
    )
    assert (tmp_path / "auto_manual.parquet").exists()
    assert not (tmp_path / "auto_auto.parquet").exists()
    assert not (tmp_path / "auto_semi.parquet").exists()
    assert pd.read_parquet(tmp_path / "auto.parquet").columns.tolist() == ["legacy"]

    import matplotlib

    matplotlib.use("Agg", force=True)
    from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer

    reviewer = DrogueLossReviewer(config, mode="manual")
    reviewer.action("uncertain")
    reviewer.action("uncertain")
    assert reviewer.closed
    assert (tmp_path / "review_manual.csv").exists()
    assert not (tmp_path / "review.csv").exists()

    messages.clear()
    FakeReviewer.calls.clear()
    run_drogue_workflow(
        config, "manual", progress=messages.append, reviewer_factory=FakeReviewer,
    )
    assert any("Reusing manual-mode automatic results" in message for message in messages)
    assert FakeReviewer.calls == []

    run_drogue_workflow(config, "automatic")
    run_drogue_workflow(config, "semiautomatic", reviewer_factory=FakeReviewer)
    assert (tmp_path / "auto_auto.parquet").exists()
    assert (tmp_path / "auto_semi.parquet").exists()


def test_automatic_requires_overwrite_and_only_replaces_its_own_output(tmp_path):
    config = _config(tmp_path)
    first = run_drogue_workflow(config, "automatic")
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_drogue_workflow(config, "automatic")
    second = run_drogue_workflow(config, "automatic", overwrite=True)
    assert list(first.columns) == list(second.columns)
    assert not (tmp_path / "auto_manual.parquet").exists()


def test_reused_incompatible_result_requires_overwrite(tmp_path):
    config = _config(tmp_path)
    pd.DataFrame({"platform_code": ["1001"]}).to_parquet(
        tmp_path / "auto_manual.parquet", index=False,
    )
    with pytest.raises(ValueError, match="use --overwrite"):
        run_drogue_workflow(config, "manual", reviewer_factory=FakeReviewer)


def test_completed_manual_workflow_does_not_reopen_reviewer(tmp_path):
    config = _config(tmp_path)
    automatic = run_drogue_workflow(
        config, "manual", reviewer_factory=FakeReviewer,
    )
    workflow_config = load_drogue_config(config).for_mode("manual")
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

    FakeReviewer.calls.clear()
    messages: list[str] = []
    run_drogue_workflow(
        config, "manual", progress=messages.append, reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == []
    assert any("No unfinished manual" in message for message in messages)


def test_stale_semiautomatic_review_returns_to_queue(tmp_path):
    config = _config(tmp_path)
    automatic = run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    workflow_config = load_drogue_config(config).for_mode("semiautomatic")
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
    changed = automatic.copy()
    changed.loc[changed.platform_code.astype(str) == "1001", "ttff_change_time"] += (
        pd.Timedelta(hours=6)
    )
    changed.to_parquet(workflow_config.automatic_output, index=False)

    FakeReviewer.calls.clear()
    run_drogue_workflow(
        config, "semiautomatic", reviewer_factory=FakeReviewer,
    )
    assert FakeReviewer.calls == [{
        "config": str(config), "mode": "semiautomatic",
        "platform_codes": ["1001"], "shown": True,
    }]


def test_mode_choice_errors_cleanly(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(SystemExit) as missing:
        main([str(config)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as multiple:
        main([str(config), "--automatic", "--mode", "manual"])
    assert multiple.value.code == 2


def test_unsupported_experiment_fails_before_dispatch(tmp_path):
    config = _config(tmp_path, experiment="unknown_campaign")
    with pytest.raises(ValueError, match="Unsupported drogue experiment"):
        run_drogue_workflow(config, "automatic")
