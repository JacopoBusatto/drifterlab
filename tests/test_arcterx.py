import numpy as np
import pytest

from drifterlab.experiments.arcterx import read_microsvp


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
