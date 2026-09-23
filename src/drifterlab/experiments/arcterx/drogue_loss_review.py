"""Persistent manual decisions for standalone ARCTERX drogue-loss review."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from drifterlab.qc.drogue import analysis_cutoff_time


REVIEW_COLUMNS = [
    "platform_code", "auto_drogue_loss_time", "review_status",
    "reviewed_drogue_loss_time", "analysis_cutoff_time", "review_reason",
    "review_timestamp", "source_sha256",
]
REVIEW_STATUSES = {"accepted_auto", "manual_time", "no_detectable_loss", "uncertain"}
REVIEW_REASONS = {
    "ttff_clear", "strain_clear", "ttff_strain_agree", "ttff_temperature_agree", "all_agree",
    "manual_adjustment", "ambiguous",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _parse_optional(value: object, name: str) -> pd.Timestamp | None:
    if value is None or str(value).strip() in {"", "NaT", "nan"}:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO UTC timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must include UTC as Z or +00:00")
    return timestamp.tz_convert("UTC").as_unit("ns")


def _format(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.as_unit("ns").isoformat()


def _same_time(left: object, right: object) -> bool:
    a = _parse_optional(_format(left), "time")
    b = _parse_optional(_format(right), "time")
    return (a is None and b is None) or (a is not None and b is not None and a == b)


class DrogueLossReviews:
    """One latest decision per platform, with stale-save protection."""

    def __init__(self, path: str | Path, *, cutoff_margin_hours: float = 24):
        self.path = Path(path).resolve()
        self.cutoff_margin_hours = float(cutoff_margin_hours)
        if not np.isfinite(self.cutoff_margin_hours) or self.cutoff_margin_hours < 0:
            raise ValueError("cutoff_margin_hours must be finite and nonnegative")
        self.sha256 = _digest(self.path)
        self.rows: dict[str, dict[str, str]] = {}
        if not self.path.exists():
            return
        table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if list(table.columns) != REVIEW_COLUMNS:
            raise ValueError(f"Drogue review CSV must have columns {REVIEW_COLUMNS}")
        for row in table.to_dict("records"):
            self._validate_row(row)
            platform = row["platform_code"]
            if platform in self.rows:
                raise ValueError(f"Duplicate drogue review platform: {platform}")
            self.rows[platform] = row

    def _validate_row(self, row: dict[str, str]) -> None:
        platform = row.get("platform_code", "").strip()
        status = row.get("review_status", "").strip()
        reason = row.get("review_reason", "").strip()
        if not platform or not row.get("source_sha256", "").strip():
            raise ValueError("Drogue review requires platform_code and source_sha256")
        if status not in REVIEW_STATUSES or reason not in REVIEW_REASONS:
            raise ValueError(f"Invalid drogue review status/reason for {platform}")
        auto = _parse_optional(row.get("auto_drogue_loss_time"), "auto_drogue_loss_time")
        reviewed = _parse_optional(row.get("reviewed_drogue_loss_time"), "reviewed_drogue_loss_time")
        cutoff = _parse_optional(row.get("analysis_cutoff_time"), "analysis_cutoff_time")
        _parse_optional(row.get("review_timestamp"), "review_timestamp")
        if status == "accepted_auto" and (auto is None or reviewed != auto):
            raise ValueError(f"accepted_auto must preserve the automatic time for {platform}")
        if status == "manual_time" and reviewed is None:
            raise ValueError(f"manual_time requires a reviewed time for {platform}")
        if status in {"no_detectable_loss", "uncertain"} and reviewed is not None:
            raise ValueError(f"{status} must not store a reviewed event time for {platform}")
        if reviewed is None:
            if cutoff is not None:
                raise ValueError(f"A missing reviewed time requires a missing cutoff for {platform}")
        else:
            expected = pd.Timestamp(analysis_cutoff_time(
                reviewed.tz_localize(None).to_datetime64(), self.cutoff_margin_hours
            ), tz="UTC")
            if cutoff != expected:
                raise ValueError(f"Analysis cutoff does not match the configured margin for {platform}")

    def set(self, *, platform_code: str, auto_drogue_loss_time: object,
            review_status: str, reviewed_drogue_loss_time: object | None,
            review_reason: str, source_sha256: str) -> None:
        auto = _parse_optional(_format(auto_drogue_loss_time), "auto_drogue_loss_time")
        reviewed = _parse_optional(_format(reviewed_drogue_loss_time), "reviewed_drogue_loss_time")
        if review_status == "accepted_auto":
            reviewed = auto
        elif review_status in {"no_detectable_loss", "uncertain"}:
            reviewed = None
        cutoff = None if reviewed is None else pd.Timestamp(analysis_cutoff_time(
            reviewed.tz_localize(None).to_datetime64(), self.cutoff_margin_hours
        ), tz="UTC")
        row = dict(zip(REVIEW_COLUMNS, [
            str(platform_code), _format(auto), review_status, _format(reviewed), _format(cutoff),
            review_reason, datetime.now(timezone.utc).isoformat(), str(source_sha256),
        ]))
        self._validate_row(row)
        self.rows[str(platform_code)] = row

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([self.rows[key] for key in sorted(self.rows)], columns=REVIEW_COLUMNS)

    def save(self) -> None:
        if _digest(self.path) != self.sha256:
            raise ValueError("Drogue review CSV changed in another session; reopen before saving")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            self.table().to_csv(temporary, index=False)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.sha256 = _digest(self.path)

    def validate_against_automatic(self, automatic: pd.DataFrame) -> None:
        required = {"platform_code", "auto_drogue_loss_time", "source_sha256"}
        if not required <= set(automatic):
            raise ValueError(f"Automatic result is missing columns: {sorted(required - set(automatic))}")
        rows = {str(row.platform_code): row for row in automatic.itertuples()}
        unknown = set(self.rows) - set(rows)
        if unknown:
            raise ValueError(f"Drogue review contains unknown platforms: {sorted(unknown)}")
        for platform, decision in self.rows.items():
            source = rows[platform]
            if decision["source_sha256"] != str(source.source_sha256):
                raise ValueError(f"Raw source changed for reviewed platform {platform}")
            if not _same_time(decision["auto_drogue_loss_time"], source.auto_drogue_loss_time):
                raise ValueError(f"Automatic candidate changed for reviewed platform {platform}")
