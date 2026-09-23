import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from drifterlab.experiments.arcterx import read_microsvp
from drifterlab.experiments.arcterx.config import load_config
from drifterlab.experiments.arcterx.drogue_review import read_review_table, ReviewTable
from drifterlab.experiments.arcterx.qc import prepare_record


def test_reader_preserves_all_fields_on_independent_axes(mat_factory):
    path, payload = mat_factory()
    record = read_microsvp(path)
    assert record.platform_code == "1001"
    assert len(record.series["qc"].time) == 3
    assert len(record.series["interp"].time) == 5
    for source, target in [("longitude", "lon_qc"), ("SST", "sst_qc"), ("speed", "speed_qc_source")]:
        np.testing.assert_array_equal(record.series["qc"].variables[target], payload["drifter"][source][::-1])
    for source, target in [("longitude_30min", "lon_interp_30m"), ("longitude_60min", "lon_interp_60m"), ("SST", "sst_interp")]:
        np.testing.assert_array_equal(record.series["interp"].variables[target], payload["drifter_interp"][source][::-1])
    assert np.isnan(record.series["qc"].variables["slp_qc"]).all()
    assert np.isnan(record.series["qc"].variables["drogue_strain_qc"][0])


def test_battery_shape_preserved_without_invented_alignment(mat_factory):
    path, _ = mat_factory(native_changes={"battery": np.array([5., -999., 6., 7.])})
    record = read_microsvp(path)
    assert np.isnan(record.series["qc"].variables["battery_qc"]).all()
    np.testing.assert_array_equal(record.unaligned["battery"], [5., np.nan, 6., 7.])
    assert record.metadata["battery_unaligned_count"] == 4


def test_unknown_loss_and_bad_reconstructions(mat_factory, config_factory):
    path, _ = mat_factory(native_changes={"drogue_off": -999}, interp_changes={"latitude_30min": np.array([3000., 20., 20., 20., 20.])})
    record = prepare_record(read_microsvp(path), load_config(config_factory()), ReviewTable({}))
    assert record.metadata["drogue_decision"] == "unknown"
    assert np.isnat(record.metadata["drogue_off_time"])
    assert not record.series["qc"].variables["analysis_valid_qc"].any()
    assert not record.series["interp"].variables["analysis_valid_interp_30m"].any()
    assert record.series["interp"].variables["lat_interp_30m"][-1] == 3000
    assert not record.series["interp"].variables["position_valid_interp_30m"][-1]
    assert record.series["interp"].variables["position_valid_interp_60m"][-1]


def test_residual_jump_is_advisory(mat_factory, config_factory):
    path, _ = mat_factory(native_changes={"time": np.array([719530. + 600/86400, 719530. + 300/86400, 719530.]), "longitude": np.array([132., 131., 130.])})
    record = prepare_record(read_microsvp(path), load_config(config_factory()), ReviewTable({}))
    qc = record.series["qc"].variables
    assert qc["residual_jump_flag_qc"].sum() == 2
    assert qc["analysis_valid_qc"].all()
    assert not np.array_equal(qc["audit_speed_qc"], qc["speed_qc_source"])


def test_missing_reconstruction_and_unfamiliar_fields(mat_factory):
    path, _ = mat_factory(native_changes={"new_sensor": np.array([3., -999., 1.]), "new_metadata": "recorded"}, interp_changes={"longitude_60min": None, "latitude_60min": None})
    record = read_microsvp(path)
    assert not record.metadata["has_interp_60m"]
    assert np.isnan(record.series["interp"].variables["lon_interp_60m"]).all()
    np.testing.assert_array_equal(record.series["qc"].variables["source_drifter_new_sensor"], [1, np.nan, 3])
    assert record.metadata["source_drifter_new_metadata"] == "recorded"


@pytest.mark.parametrize("changes", [{"longitude": np.array([1., 2.])}, {"unknown": np.ones((2, 2))}, {"unknown": {"nested": 1}}])
def test_unsupported_shapes_are_errors(mat_factory, changes):
    path, _ = mat_factory(native_changes=changes)
    with pytest.raises(ValueError):
        read_microsvp(path)


def write_reviews(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["platform_code", "decision", "drogue_off_time_utc", "note"])
        writer.writerows(rows)


@pytest.mark.parametrize("decision,date", [("retained", ""), ("lost", "1970-01-03T00:00:00Z")])
def test_review_changes_effective_policy_not_source(mat_factory, config_factory, tmp_path, decision, date):
    path, _ = mat_factory()
    review_path = tmp_path / "review.csv"
    write_reviews(review_path, [["1001", decision, date, "reviewed"]])
    reviews = read_review_table(review_path)
    reviews.validate_platforms({"1001"})
    record = prepare_record(read_microsvp(path), load_config(config_factory()), reviews)
    assert record.metadata["drogue_off_time"] == np.datetime64("1970-01-05")
    assert record.metadata["drogue_decision_provenance"] == "review_table"
    assert record.metadata["drogue_review_note"] == "reviewed"
    if decision == "retained":
        assert record.series["qc"].variables["analysis_valid_qc"].all()
        assert np.isnat(record.metadata["effective_drogue_off_time"])
    else:
        assert record.metadata["effective_drogue_off_time"] == np.datetime64("1970-01-03")
        assert not record.series["qc"].variables["analysis_valid_qc"].any()


@pytest.mark.parametrize("rows", [
    [["1001", "retained", "2025-01-01T00:00:00Z", ""]],
    [["1001", "lost", "", ""]], [["1001", "lost", "2025-01-01", ""]],
    [["1001", "lost", "2025-01-01T00:00:00+01:00", ""]],
    [["1001", "lost", "NaT", ""]], [["1001", "unknown", "", ""]],
    [["1001", "retained", "", ""], ["1001", "retained", "", ""]],
])
def test_invalid_reviews_fail(tmp_path, rows):
    path = tmp_path / "review.csv"
    write_reviews(path, rows)
    with pytest.raises(ValueError):
        read_review_table(path)


def test_unknown_review_platform_fails(tmp_path):
    path = tmp_path / "review.csv"
    write_reviews(path, [["999", "retained", "", ""]])
    with pytest.raises(ValueError, match="unknown platform"):
        read_review_table(path).validate_platforms({"1001"})
