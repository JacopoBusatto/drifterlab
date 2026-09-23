"""Two-pass ARCTERX ingestion: complete inventory, then bounded-memory output."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from drifterlab import __version__
from drifterlab.trajectories.inventory import inventory_row, write_inventory
from drifterlab.trajectories.zarr import DatasetLayout, write_master
from .config import load_config
from .microsvp import read_microsvp
from .drogue_review import read_review_table
from .qc import prepare_record
from .schema import REPRESENTATION_PROVENANCE


@dataclass(frozen=True)
class PreprocessingSummary:
    input_files: int
    trajectories: int
    native_observations: int
    reconstructed_observations: int
    missing_qc_positions: int
    residual_jump_warnings: int
    source_loss_estimates: int
    source_loss_unknown: int
    effective_drogue_unknown: int
    reviewed_drifters: int
    reconstruction_30m_available: int
    reconstruction_60m_available: int
    invalid_positions_30m: int
    invalid_positions_60m: int
    unaligned_battery_trajectories: int
    zarr_path: str
    inventory_path: str

    def as_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        return "\n".join([
            f"Input files / trajectories: {self.input_files} / {self.trajectories}",
            f"Native observations: {self.native_observations:,}",
            f"Reconstructed observations: {self.reconstructed_observations:,}",
            f"Missing QC positions: {self.missing_qc_positions:,}",
            f"Residual speed warnings: {self.residual_jump_warnings:,}",
            f"Source drogue loss known / unknown: {self.source_loss_estimates} / {self.source_loss_unknown}",
            f"Reviewed drifters / effective unknown: {self.reviewed_drifters} / {self.effective_drogue_unknown}",
            f"30m / 60m reconstruction availability: {self.reconstruction_30m_available} / {self.reconstruction_60m_available}",
            f"Out-of-range 30m / 60m positions: {self.invalid_positions_30m:,} / {self.invalid_positions_60m:,}",
            f"Unaligned battery arrays preserved: {self.unaligned_battery_trajectories}",
            f"Zarr: {self.zarr_path}", f"Inventory: {self.inventory_path}",
        ])


def preprocess(config_path: str | Path, *, progress: Callable[[str], None] | None = None) -> PreprocessingSummary:
    """Build inventory and a new master store; never overwrite existing products."""
    config = load_config(config_path)
    report = progress or (lambda message: None)
    for destination in (config.zarr, config.inventory):
        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}; choose new output paths")
    if not config.input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {config.input_directory}")
    files = sorted(p for p in config.input_directory.glob(config.pattern) if p.is_file())
    if not files:
        raise ValueError(f"No input files match {config.pattern!r} in {config.input_directory}")
    reviews = read_review_table(config.review_table)
    layout = DatasetLayout()
    rows = []
    sources = []
    errors = []
    for index, source in enumerate(files, start=1):
        try:
            record = prepare_record(read_microsvp(source, missing_value=config.missing_value), config, reviews)
            layout.observe(record)
            row = inventory_row(record)
            row["source_path"] = str(source)
            rows.append(row)
            sources.append((record.platform_code, source, record.metadata["source_sha256"]))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            errors.append(f"{source.name}: {exc}")
            rows.append({"source_filename": source.name, "source_path": str(source), "status": "error", "error": str(exc)})
        if index % 25 == 0 or index == len(files):
            report(f"Inventoried {index}/{len(files)} files")
    write_inventory(rows, config.inventory)
    if errors:
        raise ValueError(f"Conversion stopped after writing inventory: {len(errors)} invalid input file(s). " + " | ".join(errors[:5]))
    reviews.validate_platforms(set(layout.platforms))
    counts = Counter()
    for row in rows:
        for field in ("n_obs_qc", "n_obs_interp", "n_missing_positions_qc", "n_residual_jumps_qc",
                      "has_interp_30m", "has_interp_60m", "n_invalid_positions_interp_30m", "n_invalid_positions_interp_60m"):
            counts[field] += int(row[field])
        counts["source_loss_unknown"] += row["source_drogue_status"] == "unknown"
        counts["effective_unknown"] += row["drogue_decision"] == "unknown"
        counts["unaligned_battery"] += row["battery_unaligned_count"] > 0
    summary = PreprocessingSummary(
        len(files), len(rows), counts["n_obs_qc"], counts["n_obs_interp"],
        counts["n_missing_positions_qc"], counts["n_residual_jumps_qc"],
        len(rows) - counts["source_loss_unknown"], counts["source_loss_unknown"],
        counts["effective_unknown"], len(reviews.decisions), counts["has_interp_30m"], counts["has_interp_60m"],
        counts["n_invalid_positions_interp_30m"], counts["n_invalid_positions_interp_60m"],
        counts["unaligned_battery"], str(config.zarr), str(config.inventory),
    )
    attributes = {
        "title": "ARCTERX QC MicroSVP master trajectories", "experiment": "ARCTERX",
        "source": "Supplied Quality Controlled MicroSVP MATLAB files",
        "drifterlab_version": __version__, "schema_version": "1.0",
        "processing_date_utc": datetime.now(timezone.utc).isoformat(),
        "time_reference": "UTC (ARCTERX campaign convention)",
        "missing_value_normalization": f"Numeric {config.missing_value:g} normalized to NaN/NaT; padding distinguished by observation_present and source indices",
        "drogue_validity_convention": "Strict time < effective loss time minus configured buffer; unknown false; reviewed retained true for valid times",
        "drogue_buffer_hours": config.drogue_buffer_hours,
        "residual_jump_speed_m_s": config.audit.residual_jump_speed_m_s,
        "audit_speed_method": "Haversine, sphere radius 6371008.8 m, adjacent chronological native rows, no bridging invalid fixes, later-row assignment",
        "advisory_flags_change_validity": False,
        "representation_provenance": REPRESENTATION_PROVENANCE,
        "position_axes": {"qc": "obs_qc", "interp_30m": "obs_interp", "interp_60m": "obs_interp"},
        "effective_configuration": config.effective(),
        "review_table_sha256": reviews.sha256,
        "run_summary": summary.as_dict(),
        "inventory_path": str(config.inventory),
        "interpretation_notes": "Reconstructed SST is one supplied product; interpolation method and undocumented sensor units are not inferred. No trajectory representation is selected as preferred.",
    }

    def records():
        for index, (_, source, expected_hash) in enumerate(sorted(sources), start=1):
            record = read_microsvp(source, missing_value=config.missing_value)
            if record.metadata["source_sha256"] != expected_hash:
                raise ValueError(f"Input changed after inventory: {source}")
            yield prepare_record(record, config, reviews)
            if index % 25 == 0 or index == len(sources):
                report(f"Wrote {index}/{len(sources)} trajectories")

    write_master(records(), layout, config.zarr, attributes=attributes,
                 trajectory_chunk=config.chunk_trajectory, observation_chunk=config.chunk_observation)
    report("Verified and published master Zarr")
    return summary
