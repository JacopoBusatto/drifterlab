"""Persistent human decisions for standalone ARCTERX drogue-loss review."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from drifterlab.qc.drogue import analysis_cutoff_time


REVIEW_COLUMNS = [
    "platform_code", "ttff_change_time", "strain_change_time",
    "auto_drogue_loss_time", "auto_status", "auto_source", "review_status",
    "reviewed_drogue_loss_time", "analysis_cutoff_margin_hours",
    "analysis_cutoff_time", "review_reason", "review_timestamp", "source_sha256",
]
LEGACY_REVIEW_COLUMNS = [
    "platform_code", "auto_drogue_loss_time", "review_status",
    "reviewed_drogue_loss_time", "analysis_cutoff_time", "review_reason",
    "review_timestamp", "source_sha256",
]
REVIEW_STATUSES = {"accepted_auto", "manual_date", "not_lost", "uncertain"}
STATUS_ALIASES = {"manual_time": "manual_date", "no_detectable_loss": "not_lost"}
REVIEW_REASONS = {
    "automatic_agreement", "ttff_only", "strain_only", "signal_conflict",
    "manual_adjustment", "not_lost", "uncertain", "ambiguous",
    # Accepted legacy reason values are retained during schema migration.
    "ttff_clear", "strain_clear", "ttff_strain_agree", "ttff_temperature_agree",
    "all_agree",
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _parse_optional(value: object, name: str) -> pd.Timestamp | None:
    if value is None or str(value).strip() in {"", "NaT", "nan", "None"}:
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


def _margin(value: object, name: str = "analysis_cutoff_margin_hours") -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


class DrogueLossReviews:
    """One latest decision per platform, with migration and stale-save protection."""

    def __init__(self, path: str | Path, *, cutoff_margin_hours: float = 24):
        self.path = Path(path).resolve()
        self.cutoff_margin_hours = _margin(cutoff_margin_hours, "cutoff_margin_hours")
        self.sha256 = _digest(self.path)
        self.rows: dict[str, dict[str, str]] = {}
        self.stale_automatic: set[str] = set()
        if not self.path.exists():
            return
        table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if list(table.columns) == LEGACY_REVIEW_COLUMNS:
            records = [self._migrate_legacy(row) for row in table.to_dict("records")]
        elif list(table.columns) == REVIEW_COLUMNS:
            records = table.to_dict("records")
        else:
            raise ValueError(
                "Drogue review CSV has an unsupported schema; expected current or "
                "legacy review columns"
            )
        for row in records:
            self._validate_row(row)
            platform = row["platform_code"]
            if platform in self.rows:
                raise ValueError(f"Duplicate drogue review platform: {platform}")
            self.rows[platform] = row

    def _migrate_legacy(self, row: dict[str, str]) -> dict[str, str]:
        status = STATUS_ALIASES.get(row["review_status"], row["review_status"])
        reviewed = _parse_optional(
            row["reviewed_drogue_loss_time"], "reviewed_drogue_loss_time"
        )
        cutoff = _parse_optional(row["analysis_cutoff_time"], "analysis_cutoff_time")
        margin = self.cutoff_margin_hours
        if reviewed is not None and cutoff is not None:
            margin = float((reviewed - cutoff) / pd.Timedelta(hours=1))
            margin = _margin(margin)
        return dict(zip(REVIEW_COLUMNS, [
            row["platform_code"], "", "", row["auto_drogue_loss_time"], "", "",
            status, row["reviewed_drogue_loss_time"], f"{margin:g}",
            row["analysis_cutoff_time"], row["review_reason"], row["review_timestamp"],
            row["source_sha256"],
        ]))

    def _validate_row(self, row: dict[str, str]) -> None:
        platform = row.get("platform_code", "").strip()
        status = STATUS_ALIASES.get(
            row.get("review_status", "").strip(), row.get("review_status", "").strip()
        )
        row["review_status"] = status
        reason = row.get("review_reason", "").strip()
        if not platform or not row.get("source_sha256", "").strip():
            raise ValueError("Drogue review requires platform_code and source_sha256")
        if status not in REVIEW_STATUSES or reason not in REVIEW_REASONS:
            raise ValueError(f"Invalid drogue review status/reason for {platform}")
        _parse_optional(row.get("ttff_change_time"), "ttff_change_time")
        _parse_optional(row.get("strain_change_time"), "strain_change_time")
        auto = _parse_optional(row.get("auto_drogue_loss_time"), "auto_drogue_loss_time")
        reviewed = _parse_optional(row.get("reviewed_drogue_loss_time"), "reviewed_drogue_loss_time")
        cutoff = _parse_optional(row.get("analysis_cutoff_time"), "analysis_cutoff_time")
        _parse_optional(row.get("review_timestamp"), "review_timestamp")
        margin = _margin(row.get("analysis_cutoff_margin_hours"))
        if status == "accepted_auto" and (auto is None or reviewed != auto):
            raise ValueError(f"accepted_auto must preserve the automatic time for {platform}")
        if status == "manual_date" and reviewed is None:
            raise ValueError(f"manual_date requires a reviewed time for {platform}")
        if status in {"not_lost", "uncertain"} and reviewed is not None:
            raise ValueError(f"{status} must not store a reviewed event time for {platform}")
        if reviewed is None:
            if cutoff is not None:
                raise ValueError(f"A missing reviewed time requires a missing cutoff for {platform}")
        else:
            expected = pd.Timestamp(analysis_cutoff_time(
                reviewed.tz_localize(None).to_datetime64(), margin,
            ), tz="UTC")
            if cutoff != expected:
                raise ValueError(f"Analysis cutoff does not match the stored margin for {platform}")

    def set(
        self,
        *,
        platform_code: str,
        auto_drogue_loss_time: object,
        review_status: str,
        reviewed_drogue_loss_time: object | None,
        review_reason: str,
        source_sha256: str,
        ttff_change_time: object = None,
        strain_change_time: object = None,
        auto_status: str = "",
        auto_source: str = "",
        analysis_cutoff_margin_hours: float | None = None,
    ) -> None:
        status = STATUS_ALIASES.get(review_status, review_status)
        auto = _parse_optional(_format(auto_drogue_loss_time), "auto_drogue_loss_time")
        reviewed = _parse_optional(_format(reviewed_drogue_loss_time), "reviewed_drogue_loss_time")
        margin = _margin(
            self.cutoff_margin_hours
            if analysis_cutoff_margin_hours is None else analysis_cutoff_margin_hours
        )
        if status == "accepted_auto":
            reviewed = auto
        elif status in {"not_lost", "uncertain"}:
            reviewed = None
        cutoff = None if reviewed is None else pd.Timestamp(analysis_cutoff_time(
            reviewed.tz_localize(None).to_datetime64(), margin,
        ), tz="UTC")
        row = dict(zip(REVIEW_COLUMNS, [
            str(platform_code), _format(ttff_change_time), _format(strain_change_time),
            _format(auto), str(auto_status), str(auto_source), status, _format(reviewed),
            f"{margin:g}", _format(cutoff), review_reason,
            datetime.now(timezone.utc).isoformat(), str(source_sha256),
        ]))
        self._validate_row(row)
        self.rows[str(platform_code)] = row
        self.stale_automatic.discard(str(platform_code))

    def set_margin(self, platform_code: str, margin_hours: float) -> None:
        """Change only the analysis margin, preserving the human decision snapshot."""
        platform = str(platform_code)
        if platform not in self.rows:
            raise ValueError(f"No drogue review exists for platform {platform}")
        margin = _margin(margin_hours)
        row = dict(self.rows[platform])
        reviewed = _parse_optional(
            row["reviewed_drogue_loss_time"], "reviewed_drogue_loss_time"
        )
        cutoff = None if reviewed is None else pd.Timestamp(analysis_cutoff_time(
            reviewed.tz_localize(None).to_datetime64(), margin,
        ), tz="UTC")
        row["analysis_cutoff_margin_hours"] = f"{margin:g}"
        row["analysis_cutoff_time"] = _format(cutoff)
        row["review_timestamp"] = datetime.now(timezone.utc).isoformat()
        self._validate_row(row)
        self.rows[platform] = row

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [self.rows[key] for key in sorted(self.rows)], columns=REVIEW_COLUMNS,
        )

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

    def validate_against_automatic(self, automatic: pd.DataFrame) -> set[str]:
        required = {
            "platform_code", "source_sha256", "ttff_change_time", "strain_change_time",
            "auto_drogue_loss_time", "auto_status", "auto_source",
        }
        if not required <= set(automatic):
            raise ValueError(
                f"Automatic result is missing columns: {sorted(required - set(automatic))}"
            )
        rows = {str(row.platform_code): row for row in automatic.itertuples()}
        unknown = set(self.rows) - set(rows)
        if unknown:
            raise ValueError(f"Drogue review contains unknown platforms: {sorted(unknown)}")
        self.stale_automatic.clear()
        for platform, decision in self.rows.items():
            source = rows[platform]
            if decision["source_sha256"] != str(source.source_sha256):
                raise ValueError(f"Raw source changed for reviewed platform {platform}")
            time_fields = (
                "ttff_change_time", "strain_change_time", "auto_drogue_loss_time",
            )
            changed = any(
                decision[name] and not _same_time(decision[name], getattr(source, name))
                for name in time_fields
            )
            changed |= bool(
                decision["auto_status"]
                and decision["auto_status"] != str(source.auto_status)
            )
            changed |= bool(
                decision["auto_source"]
                and decision["auto_source"] != str(source.auto_source)
            )
            if changed:
                self.stale_automatic.add(platform)
        return set(self.stale_automatic)
