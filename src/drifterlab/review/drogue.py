"""Generic persistence and interactive review for drogue-loss decisions."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import numpy as np
import pandas as pd

from drifterlab.qc.drogue import analysis_cutoff_time
from drifterlab.qc.strain_two_regime import aggregate_strain_blocks
from drifterlab.io.drogue import RawDrogueSignals, get_drogue_reader
from drifterlab.workflows.drogue import load_drogue_config, load_drogue_detection


REVIEW_COLUMNS = [
    "platform_code", "ttff_change_time", "strain_change_time",
    "auto_drogue_loss_time", "auto_status", "auto_source", "review_status",
    "reviewed_drogue_loss_time", "analysis_cutoff_margin_hours",
    "analysis_cutoff_time", "review_reason", "review_timestamp", "source_sha256",
]
REVIEW_STATUSES = {"accepted_auto", "manual_date", "not_lost", "uncertain"}
REVIEW_REASONS = {
    "automatic_agreement", "ttff_only", "strain_only", "signal_conflict",
    "manual_adjustment", "not_lost", "uncertain",
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
        if list(table.columns) != REVIEW_COLUMNS:
            raise ValueError("Drogue review CSV has an unsupported schema")
        records = table.to_dict("records")
        for row in records:
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
        status = review_status
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

def _utc(value: object) -> pd.Timestamp | None:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


class DrogueLossReviewer:
    """Review a selected automatic-table subset without rerunning either detector."""

    def __init__(self, config_path: str | Path, *,
                 platform_codes: Iterable[str] | None = None):
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.mdates, self.plt = mdates, plt
        self.config = load_drogue_config(config_path)
        self.reader = get_drogue_reader(self.config.input_reader)
        if not self.config.automatic_output.exists():
            raise ValueError(f"Automatic result does not exist: {self.config.automatic_output}")
        all_automatic = load_drogue_detection(self.config.automatic_output)
        self.reviews = DrogueLossReviews(
            self.config.review_output,
            cutoff_margin_hours=self.config.analysis_cutoff_margin_hours,
        )
        self.reviews.validate_against_automatic(all_automatic)

        if platform_codes is None:
            self.automatic = all_automatic.reset_index(drop=True)
        else:
            requested = list(dict.fromkeys(map(str, platform_codes)))
            known = set(all_automatic.platform_code)
            unknown = set(requested) - known
            if unknown:
                raise ValueError(f"Requested review platforms are unknown: {sorted(unknown)}")
            indexed = all_automatic.set_index("platform_code", drop=False)
            self.automatic = indexed.loc[requested].reset_index(drop=True)
        if self.automatic.empty:
            raise ValueError("No platforms require drogue review")

        pending = np.flatnonzero(
            ~self.automatic.platform_code.isin(self.reviews.rows)
            | self.automatic.platform_code.isin(self.reviews.stale_automatic)
        )
        self.cursor = int(pending[0]) if len(pending) else 0
        self.selected_time: pd.Timestamp | None = None
        self.review_margin_hours = self.config.analysis_cutoff_margin_hours
        self.selection_dirty = False
        self.loaded_platform: str | None = None
        self.display_cache: dict[str, tuple[RawDrogueSignals, pd.DataFrame]] = {}
        self.overlay_artists: list[object] = []

        self.figure, (self.ttff_ax, self.strain_ax) = plt.subplots(
            2, 1, figsize=(15, 8), sharex=True,
        )
        self.figure.subplots_adjust(left=.07, right=.96, top=.88, bottom=.24, hspace=.24)
        handler = getattr(self.figure.canvas.manager, "key_press_handler_id", None)
        if handler is not None:
            self.figure.canvas.mpl_disconnect(handler)
        buttons = [
            ("Accept auto", "accept"), ("Set manual", "manual"),
            ("Not lost", "none"), ("Uncertain", "uncertain"),
            ("-6 h", "-6"), ("-1 h", "-1"), ("+1 h", "+1"), ("+6 h", "+6"),
            ("Previous", "previous"), ("Next", "next"),
            ("Quit", "quit"),
        ]
        self.buttons = []
        for index, (label, action) in enumerate(buttons):
            button_row, column = divmod(index, 6)
            button = Button(
                self.figure.add_axes((.055 + column * .155, .135 - button_row * .058, .14, .042)),
                label,
            )
            button.on_clicked(lambda event, value=action: self.action(value))
            self.buttons.append(button)
        self.margin_box = TextBox(
            self.figure.add_axes((.81, .014, .09, .032)), "Cutoff margin (h) ",
            initial=f"{self.review_margin_hours:g}",
        )
        self.margin_box.on_submit(self.on_margin_submit)
        self.status = self.figure.text(.07, .018, "", fontsize=9)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.closed = False
        self.draw()

    def _automatic_time(self, row: pd.Series) -> pd.Timestamp | None:
        return _utc(row.auto_drogue_loss_time)

    def _load(self) -> pd.Series:
        row = self.automatic.iloc[self.cursor]
        platform = str(row.platform_code)
        if platform == self.loaded_platform:
            return row
        if platform not in self.display_cache:
            path = Path(str(row.source_path))
            if not path.exists():
                path = self.config.input_directory / str(row.source_filename)
            signals = self.reader(
                path, missing_value=self.config.missing_value,
            )
            if signals.source_sha256 != str(row.source_sha256):
                raise ValueError(f"Raw source changed for platform {platform}")
            strain = (
                np.full(len(signals.time), np.nan)
                if signals.strain is None else signals.strain
            )
            blocks = aggregate_strain_blocks(
                signals.time, strain, self.config.detection.strain,
                missing_values=(self.config.missing_value,),
            )
            self.display_cache[platform] = signals, blocks
        self.signals, self.strain_blocks = self.display_cache[platform]

        decision = self.reviews.rows.get(platform)
        reviewed = None if decision is None else _utc(decision["reviewed_drogue_loss_time"])
        automatic = self._automatic_time(row)
        self.selected_time = reviewed if reviewed is not None else automatic
        if self.selected_time is None:
            valid_time = pd.DatetimeIndex(self.signals.time).dropna()
            self.selected_time = _utc(valid_time[len(valid_time) // 2])
        self.review_margin_hours = float(
            decision["analysis_cutoff_margin_hours"]
            if decision is not None else row.default_analysis_cutoff_margin_hours
        )
        self.margin_box.eventson = False
        self.margin_box.set_val(f"{self.review_margin_hours:g}")
        self.margin_box.eventson = True
        self.selection_dirty = False
        self.loaded_platform = platform
        return row

    def _reason(self, status: str) -> str:
        auto_status = str(self.automatic.iloc[self.cursor].auto_status)
        if status == "manual_date":
            return "manual_adjustment"
        if status == "not_lost":
            return "not_lost"
        if status == "uncertain":
            return "signal_conflict" if auto_status == "signal_conflict" else "uncertain"
        return {
            "clear_agreement": "automatic_agreement",
            "clear_ttff_only": "ttff_only",
            "clear_strain_only": "strain_only",
        }.get(auto_status, "uncertain")

    def _set(self, status: str) -> bool:
        row = self.automatic.iloc[self.cursor]
        reviewed = self.selected_time if status == "manual_date" else None
        platform = str(row.platform_code)
        previous = self.reviews.rows.get(platform)
        previous = None if previous is None else previous.copy()
        was_stale = platform in self.reviews.stale_automatic
        try:
            self.reviews.set(
                platform_code=platform,
                ttff_change_time=row.ttff_change_time,
                strain_change_time=row.strain_change_time,
                auto_drogue_loss_time=row.auto_drogue_loss_time,
                auto_status=str(row.auto_status),
                auto_source=str(row.auto_source),
                review_status=status,
                reviewed_drogue_loss_time=reviewed,
                review_reason=self._reason(status),
                source_sha256=str(row.source_sha256),
                analysis_cutoff_margin_hours=self.review_margin_hours,
            )
            self.reviews.save()
        except (OSError, ValueError) as exc:
            if previous is None:
                self.reviews.rows.pop(platform, None)
            else:
                self.reviews.rows[platform] = previous
            if was_stale:
                self.reviews.stale_automatic.add(platform)
            self.status.set_text(f"Could not save decision: {exc}")
            self.figure.canvas.draw_idle()
            return False
        self.selection_dirty = False
        self.status.set_text(f"Saved {status} to {self.config.review_output}")
        return True

    def _pending_indices(self) -> list[int]:
        return [
            index for index, platform in enumerate(self.automatic.platform_code.astype(str))
            if platform not in self.reviews.rows
            or platform in self.reviews.stale_automatic
        ]

    def _next_pending(self, *, exclude_current: bool = False) -> int | None:
        pending = self._pending_indices()
        if exclude_current:
            pending = [index for index in pending if index != self.cursor]
        if not pending:
            return None
        later = [index for index in pending if index > self.cursor]
        return later[0] if later else pending[0]

    def _advance_after_save(self) -> None:
        following = self._next_pending()
        if following is None:
            self.closed = True
            self.plt.close(self.figure)
            return
        self.cursor = following
        self.loaded_platform = None
        self.draw()

    def _record(self, status: str) -> None:
        if self._set(status):
            self._advance_after_save()

    def _displayed_loss_time(self, row: pd.Series) -> pd.Timestamp | None:
        if self.selection_dirty:
            return self.selected_time
        decision = self.reviews.rows.get(str(row.platform_code))
        if decision is None:
            return self._automatic_time(row)
        return _utc(decision["reviewed_drogue_loss_time"])

    def _displayed_cutoff_time(self, row: pd.Series) -> pd.Timestamp | None:
        physical = self._displayed_loss_time(row)
        if physical is None:
            return None
        return _utc(analysis_cutoff_time(
            physical.tz_localize(None).to_datetime64(), self.review_margin_hours,
        ))

    def _render_platform(self, row: pd.Series) -> None:
        for axis in (self.ttff_ax, self.strain_ax):
            axis.clear()
        time = pd.DatetimeIndex(self.signals.time)
        ttff = np.asarray(self.signals.ttff, dtype=float)
        positive = np.isfinite(ttff) & (ttff > 0)
        self.ttff_ax.plot(
            time[positive], ttff[positive], ".", color=".4", ms=2,
            label="Positive raw TTFF",
        )
        self.ttff_ax.set_yscale("log")
        self.ttff_ax.set_ylabel("TTFF (log)")
        self.ttff_ax.set_title(
            f"TTFF: {row.ttff_status} ({row.ttff_detail_status})"
        )
        self.ttff_ax.grid(alpha=.2)

        if self.signals.strain is not None:
            self.strain_ax.plot(
                time, self.signals.strain, ".", color=".75", ms=2,
                label="Raw strain",
            )
        blocks = self.strain_blocks
        if not blocks.empty and blocks.strain_median.notna().any():
            self.strain_ax.plot(
                blocks.time_center, blocks.strain_median, "o-",
                color="tab:green", ms=2.5, lw=.8, label="Strain block median",
            )
            change = _utc(row.strain_change_time)
            if change is not None:
                block_time = pd.DatetimeIndex(blocks.time_center)
                fitted = np.where(
                    blocks.strain_median.notna(),
                    np.where(block_time < change, row.strain_level_before,
                             row.strain_level_after),
                    np.nan,
                )
                self.strain_ax.plot(
                    block_time, fitted, color="tab:blue", lw=1.8,
                    label="Fitted pre/post levels",
                )
        self.strain_ax.set_ylabel("Strain")
        self.strain_ax.set_xlabel("UTC time")
        self.strain_ax.set_title(
            f"Robust strain: {row.strain_status}; "
            f"levels={row.strain_level_before:.3g}->{row.strain_level_after:.3g}; "
            f"drop={row.strain_absolute_drop:.3g} ({row.strain_relative_drop:.1%}); "
            f"fit={row.strain_fit_improvement:.1%}"
        )
        self.strain_ax.grid(alpha=.2)
        self.ttff_ax.legend(loc="upper right", fontsize=8)
        self.strain_ax.legend(loc="upper right", fontsize=8)

    def _update_overlays(self, row: pd.Series) -> None:
        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists.clear()
        automatic = self._automatic_time(row)
        self.buttons[0].set_active(automatic is not None)
        physical = self._displayed_loss_time(row)
        cutoff = self._displayed_cutoff_time(row)
        events: list[tuple[pd.Timestamp, str, str, str, float]] = []
        for value, label, color, style, width in (
            (row.ttff_change_time, "TTFF change", "tab:purple", "--", 1.3),
            (row.strain_change_time, "Strain change", "tab:green", "-.", 1.3),
        ):
            timestamp = _utc(value)
            if timestamp is not None:
                events.append((timestamp, label, color, style, width))
        if automatic is not None:
            events.append((automatic, "Automatic decision", "tab:red", "-", 1.7))
        if physical is not None:
            events.append((physical, "Selected physical loss", "tab:blue", "--", 1.4))
        if cutoff is not None:
            events.append((cutoff, "Analysis cutoff", "black", ":", 1.4))
        for axis in (self.ttff_ax, self.strain_ax):
            drawn: set[int] = set()
            for event_time, label, color, style, width in events:
                key = int(event_time.value)
                if key in drawn:
                    continue
                drawn.add(key)
                same = [item[1] for item in events if int(item[0].value) == key]
                self.overlay_artists.append(axis.axvline(
                    event_time, color=color, lw=width, ls=style,
                    label=" / ".join(same),
                ))
            axis.legend(loc="upper right", fontsize=8)
        platform = str(row.platform_code)
        decision = self.reviews.rows.get(platform)
        decision_text = "unreviewed" if decision is None else decision["review_status"]
        stale = " | STALE DETECTOR SNAPSHOT" if platform in self.reviews.stale_automatic else ""
        self.figure.suptitle(
            f"{platform}  {self.cursor + 1}/{len(self.automatic)}  "
            f"auto={row.auto_status} ({row.auto_source})  review={decision_text}{stale}\n"
            f"TTFF={row.ttff_status}, strain={row.strain_status}, "
            f"margin={self.review_margin_hours:g} h"
        )
        self.figure.canvas.draw_idle()

    def draw(self) -> None:
        previous = self.loaded_platform
        row = self._load()
        if previous != self.loaded_platform:
            self.overlay_artists.clear()
            self._render_platform(row)
        self._update_overlays(row)

    def action(self, action: str) -> None:
        if action == "accept":
            if self._automatic_time(self.automatic.iloc[self.cursor]) is None:
                self.status.set_text("No automatic decision is available to accept.")
                self.figure.canvas.draw_idle()
            else:
                self._record("accepted_auto")
        elif action == "manual":
            self._record("manual_date")
        elif action == "none":
            self._record("not_lost")
        elif action == "uncertain":
            self._record("uncertain")
        elif action in {"-6", "-1", "+1", "+6"}:
            self.selected_time += pd.Timedelta(hours=int(action))
            self.selection_dirty = True
            self.status.set_text("Selected time changed; press Set manual to save it.")
            self._update_overlays(self.automatic.iloc[self.cursor])
        elif action == "previous":
            self.cursor = (self.cursor - 1) % len(self.automatic)
            self.loaded_platform = None
            self.status.set_text("")
            self.draw()
        elif action == "next":
            following = self._next_pending(exclude_current=True)
            if following is None:
                self.status.set_text("This is the only unfinished platform.")
                self.figure.canvas.draw_idle()
                return
            self.cursor = following
            self.loaded_platform = None
            self.status.set_text("")
            self.draw()
        elif action == "quit":
            self.closed = True
            self.plt.close(self.figure)

    def on_margin_submit(self, text: str) -> None:
        try:
            margin = float(text)
            if not np.isfinite(margin) or margin < 0:
                raise ValueError
        except ValueError:
            self.status.set_text("Cutoff margin must be finite and nonnegative.")
            self.margin_box.eventson = False
            self.margin_box.set_val(f"{self.review_margin_hours:g}")
            self.margin_box.eventson = True
            self.figure.canvas.draw_idle()
            return
        self.review_margin_hours = margin
        platform = str(self.automatic.iloc[self.cursor].platform_code)
        if platform in self.reviews.rows:
            previous = self.reviews.rows[platform].copy()
            try:
                self.reviews.set_margin(platform, margin)
                self.reviews.save()
                message = f"Margin saved to {self.config.review_output}."
            except (OSError, ValueError) as exc:
                self.reviews.rows[platform] = previous
                self.review_margin_hours = float(
                    previous["analysis_cutoff_margin_hours"]
                )
                self.margin_box.eventson = False
                self.margin_box.set_val(f"{self.review_margin_hours:g}")
                self.margin_box.eventson = True
                message = f"Could not save margin: {exc}"
        else:
            message = "Margin selected; record a review decision to store it."
        self.status.set_text(message)
        self._update_overlays(self.automatic.iloc[self.cursor])

    def on_click(self, event) -> None:
        if event.inaxes not in {self.ttff_ax, self.strain_ax} or event.xdata is None:
            return
        self.selected_time = pd.Timestamp(self.mdates.num2date(event.xdata)).tz_convert("UTC")
        self.selection_dirty = True
        self.status.set_text("Selected physical loss time changed; press Set manual to record it.")
        self._update_overlays(self.automatic.iloc[self.cursor])

    def on_key(self, event) -> None:
        mapping = {
            "a": "accept", "m": "manual", "l": "none", "u": "uncertain",
            "n": "next", "p": "previous", "q": "quit",
        }
        if event.key in mapping:
            self.action(mapping[event.key])

    def on_close(self, event) -> None:
        if not self.closed:
            self.closed = True

    def show(self) -> None:
        self.plt.show()
