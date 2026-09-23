"""Apply ARCTERX QC policy and explicit decisions to normalized records."""

import json

import numpy as np

from drifterlab.qc.drogue import drogue_cutoff, drogue_valid
from drifterlab.qc.flags import audit_time, consecutive_speed
from drifterlab.qc.position import analysis_valid, position_valid
from drifterlab.trajectories.records import TrajectoryRecord
from .config import Config
from .drogue_review import ReviewTable


def _bounds(time: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.datetime64, np.datetime64]:
    valid = ~np.isnat(time)
    if mask is not None:
        valid &= mask
    selected = time[valid]
    if not len(selected):
        return np.datetime64("NaT", "ns"), np.datetime64("NaT", "ns")
    return selected.min(), selected.max()


def prepare_record(record: TrajectoryRecord, config: Config, reviews: ReviewTable) -> TrajectoryRecord:
    meta = record.metadata
    source_loss = meta["drogue_off_time"]
    decision = "unknown" if np.isnat(source_loss) else "lost"
    effective_loss = source_loss
    provenance = "unknown" if decision == "unknown" else "source_estimate"
    note = ""
    if record.platform_code in reviews.decisions:
        review = reviews.decisions[record.platform_code]
        decision, effective_loss, note = review.decision, review.loss_time, review.note
        provenance = "review_table"
    cutoff = drogue_cutoff(effective_loss, config.drogue_buffer_hours)
    meta.update(
        time_reference="UTC", source_drogue_status="unknown" if np.isnat(source_loss) else "loss_estimated",
        drogue_decision=decision, drogue_decision_provenance=provenance,
        drogue_review_note=note, effective_drogue_off_time=effective_loss,
        analysis_drogue_end_time=cutoff,
    )
    for axis, series in record.series.items():
        temporal = audit_time(series.time, config.audit)
        for name, values in temporal.items():
            series.variables[f"{name}_{axis}"] = values
            if name.endswith("flag"):
                count = int(values.sum())
                meta[f"n_{name}_{axis}"] = count
                if count:
                    record.warnings.append(f"{name}_{axis}:{count}")
        series.attributes[f"interval_seconds_{axis}"] = {"units": "s", "long_name": "Elapsed time since immediately preceding sorted source row"}
        meta[f"source_time_order_{axis}"] = series.source_order
        if series.source_order != "ascending":
            record.warnings.append(f"source_time_order_{axis}:{series.source_order}")
        meta[f"first_time_{axis}"], meta[f"last_time_{axis}"] = _bounds(series.time)
        positive = temporal["interval_seconds"][temporal["interval_seconds"] > 0]
        meta[f"median_interval_seconds_{axis}"] = float(np.median(positive)) if len(positive) else np.nan
        series.variables[f"drogue_valid_{axis}"] = drogue_valid(series.time, decision, cutoff)
        series.attributes[f"drogue_valid_{axis}"] = {"long_name": "Valid time before strict buffered loss cutoff, or explicitly reviewed retained; unknown is false"}
    for representation, axis in record.position_axes.items():
        series = record.series[axis]
        lon, lat = series.variables[f"lon_{representation}"], series.variables[f"lat_{representation}"]
        good = position_valid(lon, lat)
        missing = ~np.isfinite(lon) | ~np.isfinite(lat)
        outside = ~missing & ~good
        series.variables[f"position_valid_{representation}"] = good
        series.variables[f"missing_position_flag_{representation}"] = missing
        series.variables[f"invalid_position_flag_{representation}"] = outside
        series.variables[f"analysis_valid_{representation}"] = analysis_valid(good, series.variables[f"drogue_valid_{axis}"], series.time)
        series.attributes[f"position_valid_{representation}"] = {"long_name": "Finite position inside latitude [-90,90] and longitude [-180,180]; advisory speed flags do not change this mask"}
        meta[f"n_missing_positions_{representation}"] = int(missing.sum())
        meta[f"n_invalid_positions_{representation}"] = int(outside.sum())
        meta[f"n_analysis_valid_{representation}"] = int(series.variables[f"analysis_valid_{representation}"].sum())
        if outside.any():
            record.warnings.append(f"invalid_positions_{representation}:{outside.sum()}")
    qc = record.series["qc"]
    speed = consecutive_speed(qc.time, qc.variables["lon_qc"], qc.variables["lat_qc"])
    qc.variables["audit_speed_qc"] = speed
    qc.variables["residual_jump_flag_qc"] = speed > config.audit.residual_jump_speed_m_s
    qc.attributes["audit_speed_qc"] = {"units": "m s-1", "long_name": "Haversine speed from preceding adjacent valid native row; assigned to later row"}
    qc.attributes["residual_jump_flag_qc"] = {"long_name": "Advisory consecutive speed exceedance; does not change analysis validity", "threshold_m_s": config.audit.residual_jump_speed_m_s}
    meta["n_residual_jumps_qc"] = int(qc.variables["residual_jump_flag_qc"].sum())
    if meta["n_residual_jumps_qc"]:
        record.warnings.append(f"residual_jumps_qc:{meta['n_residual_jumps_qc']}")
    meta["first_valid_position_time_qc"], meta["last_valid_position_time_qc"] = _bounds(qc.time, qc.variables["position_valid_qc"])
    meta["warnings_json"] = json.dumps(record.warnings)
    record.validate()
    return record
