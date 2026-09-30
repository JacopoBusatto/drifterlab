"""Per-trajectory position-QC workflow, provenance, and lazy-loading tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from scipy.io import savemat
import yaml

import drifterlab.workflows.position as position_workflow_module
from drifterlab.cli.position_qc import main as position_main
from drifterlab.io.position import read_microsvp_position_mat
from drifterlab.review.position import PositionReviews, REVIEW_COLUMNS, file_sha256
from drifterlab.workflows.drogue import AUTOMATIC_REQUIRED_COLUMNS
from drifterlab.workflows.position import (
    DEPLOYMENT_COLUMNS, POSITION_QC_ALGORITHM_VERSION, PositionWorkflowResult, load_deployments,
    load_position_config, run_position_workflow,
)


def write_raw(path: Path, platform: str) -> None:
    time = pd.date_range("2025-01-01", periods=10, freq="5min")[::-1]
    lon = np.array([0.009, 0.008, 0.007, 0.006, 0.2, 0.004, 0.2, 0.002, 0.001, 0.0])
    savemat(path, {"dataset": {f"drifter_{platform}": {
        "PlatformId": int(platform), "ObsTimestamp": time.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
        "GpsLongitude": lon, "GpsLatitude": np.zeros(len(time)),
    }}})


def write_drogue(path: Path, raw_paths: list[Path], *, stale_platform: str | None = None) -> None:
    rows = []
    for raw_path in raw_paths:
        trajectory = read_microsvp_position_mat(raw_path)
        row = {column: np.nan for column in AUTOMATIC_REQUIRED_COLUMNS}
        loss = pd.Timestamp("2025-02-01T00:00:00Z")
        row.update({
            "platform_code": trajectory.platform_code, "source_path": str(raw_path),
            "source_filename": raw_path.name,
            "source_sha256": "stale" if trajectory.platform_code == stale_platform else trajectory.source_sha256,
            "auto_drogue_loss_time": loss, "auto_status": "clear_agreement",
            "auto_source": "ttff+strain", "default_analysis_cutoff_time": loss - pd.Timedelta(hours=24),
            "default_analysis_cutoff_margin_hours": 24.0, "ttff_change_time": loss,
            "ttff_status": "clear", "ttff_detail_status": "clear",
            "strain_change_time": loss, "strain_status": "clear",
        })
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_config(tmp_path: Path, raw_directory: Path, drogue: Path) -> Path:
    config = tmp_path / "position.yml"
    config.write_text(yaml.safe_dump({
        "input": {"reader": "microsvp_mat", "directory": str(raw_directory),
                  "pattern": "*.mat", "options": {"missing_value": -999}},
        "drogue": {"automatic": str(drogue), "review": str(tmp_path / "drogue_review.csv")},
        "deployment": {"metadata": None},
        "output": {"directory": str(tmp_path / "position_qc"),
                   "review": str(tmp_path / "position_review.csv")},
        "temporal_segments": {}, "position_qc": {}, "review": {},
    }), encoding="utf-8")
    return config


def make_inputs(tmp_path: Path, platforms=("1001",)):
    source = tmp_path / "raw"
    source.mkdir()
    raw_paths = []
    for platform in platforms:
        raw = source / f"{platform}.mat"
        write_raw(raw, platform)
        raw_paths.append(raw)
    drogue = tmp_path / "auto_drogue.parquet"
    write_drogue(drogue, raw_paths)
    return raw_paths, drogue, write_config(tmp_path, source, drogue)


@pytest.fixture
def position_inputs(tmp_path):
    raws, drogue, config = make_inputs(tmp_path)
    return raws[0], drogue, config


def metadata(path: Path) -> dict:
    return json.loads(pq.read_metadata(path).metadata[b"drifterlab_position_qc"])


def test_microsvp_reader_preserves_reverse_source_order_and_coordinates(position_inputs):
    raw, _, _ = position_inputs
    trajectory = read_microsvp_position_mat(raw)
    assert trajectory.source_obs_index.tolist() == list(range(10))
    assert np.all(np.diff(trajectory.time).astype("timedelta64[s]").astype(int) < 0)
    assert np.array_equal(trajectory.lon, np.array([.009, .008, .007, .006, .2, .004, .2, .002, .001, 0]))
    assert trajectory.source_sha256 == file_sha256(raw)


def test_automatic_writes_one_platform_file_with_complete_metadata_and_reuses(position_inputs):
    _, _, config_path = position_inputs
    messages = []
    result = run_position_workflow(config_path, "automatic", progress=messages.append)
    assert isinstance(result, PositionWorkflowResult)
    assert result.platform_count == 1 and result.observation_count == 10
    output = result.output_directory / "1001.parquet"
    assert result.files == (output,) and output.exists()
    frame = pd.read_parquet(output)
    assert len(frame) == 10 and frame.source_obs_index.tolist() == list(range(10))
    assert {"source_lon", "source_lat"} <= set(frame)
    assert not {"lon", "lat"} & set(frame)
    np.testing.assert_array_equal(frame.source_lon, [.009, .008, .007, .006, .2, .004, .2, .002, .001, 0])
    np.testing.assert_array_equal(frame.source_lat, np.zeros(10))
    provenance = metadata(output)
    assert provenance["schema_version"] == "3.0"
    assert provenance["product_layout"] == "per-trajectory-v2"
    assert provenance["resolution_policy"] == "aggressive"
    assert provenance["platform_code"] == "1001"
    assert provenance["observation_count"] == 10
    assert isinstance(provenance["event_catalog"], list)
    assert provenance["algorithm_version"] == POSITION_QC_ALGORITHM_VERSION
    assert isinstance(provenance["decision_summary"], dict)
    assert all(
        "manual_confirmation_required" in event
        for event in provenance["event_catalog"]
    )
    assert all(
        {"event_type", "event_start_time_utc", "event_start_source_obs_index"} <= set(event)
        for event in provenance["event_catalog"]
    )
    effective = provenance["effective_configuration"]["position"]
    assert effective["local_speed_window_points"] == 15
    assert effective["endpoint_speed_min_samples"] == 8
    assert effective["boundary_endpoint_speed_multiplier"] == 10.0
    assert effective["bridge_speed_warning_z"] == 3.0
    assert effective["local_speed_scale_floor_m_s"] == .05
    assert effective["endpoint_speed_score_margin_z"] == 2.0
    assert effective["max_automatic_removal_points"] == 5
    assert result.summary_path.exists()
    summary = pd.read_csv(result.summary_path)
    assert summary.platform_code.astype(str).tolist() == ["1001", "ALL_PLATFORMS"]
    assert result.decision_summary["observation_count"] == 10
    before = file_sha256(output)
    messages.clear()
    reused = run_position_workflow(config_path, "automatic", progress=messages.append)
    assert reused == result and file_sha256(output) == before
    assert any("Reusing position QC" in message for message in messages)


def test_multiple_platforms_are_independent_and_unrelated_output_is_untouched(tmp_path):
    _, _, config_path = make_inputs(tmp_path, ("1001", "1002"))
    config = load_position_config(config_path)
    config.output_directory.mkdir(parents=True)
    unrelated = config.output_directory / "position_qc.parquet"
    unrelated.write_bytes(b"unrelated output")
    before = unrelated.read_bytes()
    result = run_position_workflow(config_path, "automatic")
    assert [path.name for path in result.files] == ["1001.parquet", "1002.parquet"]
    assert all(path.exists() for path in result.files)
    assert unrelated.read_bytes() == before


def test_review_change_rebuilds_only_affected_platform(tmp_path):
    _, _, config_path = make_inputs(tmp_path, ("1001", "1002"))
    result = run_position_workflow(config_path, "automatic")
    first, second = result.files
    frame = pd.read_parquet(first)
    target = frame.loc[frame.exact_repeat_flag, "source_obs_index"].astype(int).tolist()
    reviews = PositionReviews(load_position_config(config_path).review_output)
    reviews.set_observations(frame, target, "keep", decision_source="manual", config_sha256="test")
    reviews.save()
    before_second = file_sha256(second)
    messages = []
    run_position_workflow(config_path, "automatic", progress=messages.append)
    assert file_sha256(second) == before_second
    assert any("Published" in message and "1001.parquet" in message for message in messages)
    assert any("Reusing" in message and "1002.parquet" in message for message in messages)
    kept = pd.read_parquet(first)
    repeated = kept.loc[kept.source_obs_index.isin(target)]
    assert repeated.final_position_status.eq("rejected").all()
    assert repeated.position_decision_source.eq("automatic_repeat").all()
    assert repeated.human_position_decision.eq("keep").all()


def test_overwrite_preserves_reviews_and_existing_notes(position_inputs):
    _, _, config_path = position_inputs
    result = run_position_workflow(config_path, "automatic")
    frame = pd.read_parquet(result.files[0])
    source = int(frame.loc[frame.exact_repeat_flag, "source_obs_index"].iloc[0])
    reviews = PositionReviews(load_position_config(config_path).review_output)
    reviews.set_observations(frame, [source], "reject", decision_source="manual",
                             config_sha256="old", note="field observation")
    reviews.save()
    reviews.set_observations(frame, [source], "keep", decision_source="manual",
                             config_sha256="new", note=None)
    reviews.save()
    assert reviews.table().iloc[0].note == "field observation"
    before = file_sha256(reviews.path)
    run_position_workflow(config_path, "automatic", overwrite=True)
    assert file_sha256(reviews.path) == before


def test_schema_two_product_rebuilds_with_coordinates_and_preserves_review(position_inputs):
    _, _, config_path = position_inputs
    result = run_position_workflow(config_path, "automatic")
    output = result.files[0]
    frame = pd.read_parquet(output)
    source = int(frame.source_obs_index.iloc[0])
    reviews = PositionReviews(load_position_config(config_path).review_output)
    reviews.set_observations(
        frame, [source], "reject", decision_source="manual",
        config_sha256="review-before-schema-migration", note="retain this decision",
    )
    reviews.save()
    review_hash = file_sha256(reviews.path)
    run_position_workflow(config_path, "automatic")

    arrow = pq.read_table(output).drop(["source_lon", "source_lat"])
    footer = metadata(output)
    footer.update(schema_version="2.0", product_layout="per-trajectory-v1")
    arrow_metadata = dict(arrow.schema.metadata or {})
    arrow_metadata[b"drifterlab_position_qc"] = json.dumps(footer, sort_keys=True).encode()
    pq.write_table(arrow.replace_schema_metadata(arrow_metadata), output)

    messages = []
    run_position_workflow(config_path, "automatic", progress=messages.append)
    rebuilt = pd.read_parquet(output)
    reviewed = rebuilt.loc[rebuilt.source_obs_index == source].iloc[0]
    assert {"source_lon", "source_lat"} <= set(rebuilt)
    assert reviewed.human_position_decision == "reject"
    assert file_sha256(reviews.path) == review_hash
    assert any("Published position QC" in message for message in messages)


def test_hash_mismatch_unsafe_platform_and_shared_paths_fail_fast(position_inputs, tmp_path):
    raw, drogue, config_path = position_inputs
    write_drogue(drogue, [raw], stale_platform="1001")
    with pytest.raises(ValueError, match="source hash mismatch"):
        run_position_workflow(config_path, "automatic", overwrite=True)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["output"]["directory"] = data["drogue"]["automatic"]
    shared = tmp_path / "shared.yml"
    shared.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        load_position_config(shared)


def test_stale_position_review_source_is_rejected(position_inputs):
    _, _, config_path = position_inputs
    result = run_position_workflow(config_path, "automatic")
    frame = pd.read_parquet(result.files[0])
    source = int(frame.source_obs_index.iloc[0])
    reviews = PositionReviews(load_position_config(config_path).review_output)
    reviews.set_observations(frame, [source], "reject", decision_source="manual", config_sha256="old")
    reviews.save()
    table = pd.read_csv(reviews.path, dtype=str, keep_default_na=False)
    table["source_sha256"] = "stale"
    table.to_csv(reviews.path, index=False)
    with pytest.raises(ValueError, match="source hash changed"):
        run_position_workflow(config_path, "automatic")


def test_deployment_csv_requires_unique_explicit_utc_and_one_boundary(tmp_path):
    valid = tmp_path / "deployment.csv"
    pd.DataFrame([["1001", "2025-01-01T00:00:00Z", "", "", "log", ""]],
                 columns=DEPLOYMENT_COLUMNS).to_csv(valid, index=False)
    loaded, digest = load_deployments(valid)
    assert set(loaded) == {"1001"} and digest == file_sha256(valid)
    pd.DataFrame([["1001", "2025-01-01 00:00:00", "", "", "", ""]],
                 columns=DEPLOYMENT_COLUMNS).to_csv(valid, index=False)
    with pytest.raises(ValueError, match="explicit UTC"):
        load_deployments(valid)


def test_review_schema_remains_explicit(position_inputs):
    _, _, config_path = position_inputs
    result = run_position_workflow(config_path, "automatic")
    frame = pd.read_parquet(result.files[0])
    reviews = PositionReviews(load_position_config(config_path).review_output)
    source = int(frame.source_obs_index.iloc[0])
    reviews.set_observations(frame, [source], "reject", decision_source="manual", config_sha256="abc")
    reviews.save()
    saved = pd.read_csv(reviews.path, dtype=str, keep_default_na=False)
    assert saved.columns.tolist() == REVIEW_COLUMNS
    assert saved.target_id.tolist() == [f"observation:{source}"]


def test_manual_workflow_passes_lazy_callbacks_and_does_not_materialize_all_frames(tmp_path):
    _, _, config_path = make_inputs(tmp_path, ("1001", "1002"))
    run_position_workflow(config_path, "automatic")
    loaded = []

    class InspectLazyReviewer:
        close_handled = True
        def __init__(self, table, reviews, *, event_catalog, platform_loader,
                     trajectory_loader, recompute_platform, **_):
            assert table is None
            assert set(event_catalog.platform_code.astype(str)) == {"1001", "1002"}
            self.platform_loader = platform_loader
            self.trajectory_loader = trajectory_loader
            self.recompute_platform = recompute_platform
        def show(self):
            frame = self.platform_loader("1001")
            trajectory = self.trajectory_loader("1001")
            loaded.append((str(frame.platform_code.iloc[0]), trajectory.platform_code))

    result = run_position_workflow(config_path, "manual", reviewer_factory=InspectLazyReviewer)
    assert loaded == [("1001", "1001")]
    assert result.platform_count == 2


def test_manual_commit_rewrites_only_changed_trajectory(tmp_path):
    _, _, config_path = make_inputs(tmp_path, ("1001", "1002"))
    class NoopReviewer:
        close_handled = True
        def __init__(self, *_args, **_kwargs):
            pass
        def show(self):
            pass

    initial = run_position_workflow(config_path, "manual", reviewer_factory=NoopReviewer)
    unchanged_before = file_sha256(initial.output_directory / "1002.parquet")

    class CommitOneReviewer:
        close_handled = True
        def __init__(self, table, reviews, *, platform_loader, recompute_platform, **_):
            frame = platform_loader("1001")
            source = int(frame.source_obs_index.iloc[0])
            reviews.set_observations(frame, [source], "reject", decision_source="manual", config_sha256="test")
            reviews.save()
            self.recompute_platform = recompute_platform
        def show(self):
            self.recompute_platform("1001")

    run_position_workflow(config_path, "manual", reviewer_factory=CommitOneReviewer)
    assert file_sha256(initial.output_directory / "1002.parquet") == unchanged_before


def test_resolution_policy_change_rebuilds_products_before_review(tmp_path):
    _, _, config_path = make_inputs(tmp_path, ("1001", "1002"))
    automatic = run_position_workflow(config_path, "automatic")
    automatic_hashes = {path.name: file_sha256(path) for path in automatic.files}

    class NoopReviewer:
        close_handled = True
        def __init__(self, *_args, **_kwargs):
            pass
        def show(self):
            pass

    messages = []
    conservative = run_position_workflow(
        config_path, "semiautomatic", progress=messages.append,
        reviewer_factory=NoopReviewer,
    )
    assert all(metadata(path)["resolution_policy"] == "conservative" for path in conservative.files)
    assert all(file_sha256(path) != automatic_hashes[path.name] for path in conservative.files)
    assert sum("Published position QC" in message for message in messages) == 2


def test_position_cli_reports_directory_summary(position_inputs, capsys):
    _, _, config_path = position_inputs
    assert position_main([str(config_path), "--automatic"]) == 0
    assert "10 observations across 1 platforms" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        position_main([str(config_path), "--automatic", "--manual"])
    assert exc.value.code == 2
