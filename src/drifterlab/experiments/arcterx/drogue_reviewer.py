"""Matplotlib reviewer for standalone ARCTERX drogue-loss evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from drifterlab.qc.drogue import detect_drogue_loss
from .drogue_config import load_drogue_config
from .drogue_loss_review import DrogueLossReviews
from .raw_drogue import read_raw_drogue_signals


class DrogueLossReviewer:
    def __init__(self, config_path: str | Path):
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.mdates, self.plt = mdates, plt
        self.config = load_drogue_config(config_path)
        if not self.config.automatic_output.exists():
            raise ValueError(f"Automatic result does not exist: {self.config.automatic_output}")
        self.automatic = pd.read_parquet(self.config.automatic_output)
        required = {
            "platform_code", "source_path", "source_filename", "source_sha256",
            "auto_drogue_loss_time", "auto_status", "auto_confidence",
            "ttff_change_time", "ttff_change_status", "ttff_eligible_bin_count",
            "ttff_stable_drop_bin_count", "ttff_agreeing_bin_count",
            "ttff_consensus_span_hours", "ttff_agreeing_pre_events",
            "ttff_agreeing_post_events", "ttff_min_persistence_coverage_fraction",
            "strain_change_time", "strain_change_status", "strain_change_strength",
            "strain_drop_absolute", "strain_drop_relative", "strain_normalized_drop",
            "strain_pre_coverage_fraction", "ttff_strain_offset_hours",
        }
        missing = required - set(self.automatic)
        if missing:
            raise ValueError(
                "Automatic result uses an incompatible schema; rerun detection with "
                f"--overwrite. Missing columns: {sorted(missing)}"
            )
        if self.automatic.empty or self.automatic.platform_code.astype(str).duplicated().any():
            raise ValueError("Automatic result must contain unique platform rows")
        self.automatic["platform_code"] = self.automatic.platform_code.astype(str)
        self.reviews = DrogueLossReviews(
            self.config.review_output,
            cutoff_margin_hours=self.config.analysis_cutoff_margin_hours,
        )
        self.reviews.validate_against_automatic(self.automatic)
        undecided = np.flatnonzero(~self.automatic.platform_code.isin(self.reviews.rows))
        self.cursor = int(undecided[0]) if len(undecided) else 0
        self.selected_time: pd.Timestamp | None = None
        self.loaded_platform: str | None = None

        self.figure = plt.figure(figsize=(16, 10))
        grid = self.figure.add_gridspec(3, 2, left=.07, right=.94, top=.91, bottom=.22,
                                       hspace=.34, wspace=.25)
        self.full_ax = self.figure.add_subplot(grid[0, :])
        self.ttff_ax = self.figure.add_subplot(grid[1, 0])
        self.ttff_count_ax = self.ttff_ax.twinx()
        self.strain_ax = self.figure.add_subplot(grid[1, 1], sharex=self.ttff_ax)
        self.strain_metric_ax = self.strain_ax.twinx()
        self.temperature_ax = self.figure.add_subplot(grid[2, :], sharex=self.ttff_ax)
        self.temperature_variability_ax = self.temperature_ax.twinx()
        handler = getattr(self.figure.canvas.manager, "key_press_handler_id", None)
        if handler is not None:
            self.figure.canvas.mpl_disconnect(handler)
        buttons = [
            ("Accept auto", "accept"), ("Set manual", "manual"), ("No loss", "none"),
            ("Uncertain", "uncertain"), ("-6 h", "-6"), ("-1 h", "-1"),
            ("+1 h", "+1"), ("+6 h", "+6"), ("Previous", "previous"),
            ("Next", "next"), ("Save", "save"), ("Quit", "quit"),
        ]
        self.buttons = []
        for index, (label, action) in enumerate(buttons):
            row, column = divmod(index, 6)
            button = Button(self.figure.add_axes((.055 + column * .155, .125 - row * .057, .14, .042)), label)
            button.on_clicked(lambda event, value=action: self.action(value))
            self.buttons.append(button)
        self.status = self.figure.text(.07, .015, "", fontsize=9)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("close_event", self.on_close)
        self.closed = False
        self.draw()

    def _automatic_time(self, row: pd.Series) -> pd.Timestamp | None:
        value = pd.Timestamp(row.auto_drogue_loss_time)
        if pd.isna(value):
            return None
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")

    def _load(self) -> pd.Series:
        row = self.automatic.iloc[self.cursor]
        platform = str(row.platform_code)
        if platform != self.loaded_platform:
            path = Path(str(row.source_path))
            if not path.exists():
                path = self.config.input_directory / str(row.source_filename)
            self.signals = read_raw_drogue_signals(path, missing_value=self.config.missing_value)
            if self.signals.source_sha256 != str(row.source_sha256):
                raise ValueError(f"Raw source changed for platform {platform}")
            self.detection = detect_drogue_loss(
                platform, self.signals.time, self.signals.ttff,
                strain=self.signals.strain, hull_temperature=self.signals.hull_temperature,
                config=self.config.detection,
            )
            recomputed = pd.Timestamp(self.detection.result.auto_drogue_loss_time)
            stored = self._automatic_time(row)
            if (stored is None) != pd.isna(recomputed):
                raise ValueError(f"Automatic result cannot be reproduced for platform {platform}")
            if stored is not None and recomputed.tz_localize("UTC") != stored:
                raise ValueError(f"Automatic candidate cannot be reproduced for platform {platform}")
            decision = self.reviews.rows.get(platform)
            reviewed = None if decision is None else decision["reviewed_drogue_loss_time"]
            self.selected_time = (pd.Timestamp(reviewed) if reviewed else stored)
            if self.selected_time is None:
                times = pd.DatetimeIndex(self.detection.diagnostics.time)
                self.selected_time = times[len(times) // 2]
            if self.selected_time.tzinfo is None:
                self.selected_time = self.selected_time.tz_localize("UTC")
            self.loaded_platform = platform
        return row

    def _reason(self) -> str:
        result = self.detection.result
        if result.ttff_change_status == "clear" and result.strain_change_status == "clear":
            return "ttff_strain_agree"
        if result.strain_change_status == "clear":
            return "strain_clear"
        if result.ttff_change_status == "clear":
            return "ttff_clear"
        return "ambiguous"

    def _set(self, status: str, reason: str) -> None:
        row = self.automatic.iloc[self.cursor]
        reviewed = self.selected_time if status == "manual_time" else None
        self.reviews.set(
            platform_code=str(row.platform_code),
            auto_drogue_loss_time=row.auto_drogue_loss_time,
            review_status=status,
            reviewed_drogue_loss_time=reviewed,
            review_reason=reason,
            source_sha256=str(row.source_sha256),
        )
        self.status.set_text(f"Recorded {status} ({reason}); press Save to persist.")
        self.draw()

    def draw(self) -> None:
        row = self._load()
        result = self.detection.result
        data = self.detection.diagnostics
        for axis in (self.full_ax, self.ttff_ax, self.ttff_count_ax, self.strain_ax,
                     self.strain_metric_ax,
                     self.temperature_ax, self.temperature_variability_ax):
            axis.clear()
        time = pd.DatetimeIndex(data.time)
        automatic = self._automatic_time(row)
        self.full_ax.plot(time, data.ttff, ".", color=".45", ms=2, label="Raw TTFF")
        self.full_ax.set_ylabel("TTFF")
        self.full_ax.set_title("Full record")
        count_table = self.detection.ttff_counts
        raw_counts = np.stack(count_table.bin_counts.to_numpy()).T
        x_edges = [*count_table.time_start, count_table.time_end.iloc[-1]]
        self.ttff_ax.pcolormesh(
            x_edges, np.arange(raw_counts.shape[0] + 1), raw_counts,
            shading="flat", cmap="viridis", vmin=0,
        )
        self.ttff_ax.set_yticks(np.arange(len(self.detection.ttff_bins.labels)) + .5)
        self.ttff_ax.set_yticklabels(self.detection.ttff_bins.labels, fontsize=6)
        marker_label = True
        for component in self.detection.ttff_bin_results.itertuples():
            if pd.notna(component.drop_time):
                self.ttff_ax.plot(
                    component.drop_time, component.bin_index + .5,
                    marker="v", ms=5, color="white", mec="black",
                    label=("Per-bin stable drop" if marker_label else None),
                )
                marker_label = False
        forward_counts = np.stack(count_table.forward_counts.to_numpy()).T
        colors = self.plt.cm.viridis(
            np.linspace(.05, .95, len(self.detection.ttff_bins.labels))
        )
        for bin_index, color in enumerate(colors):
            self.ttff_count_ax.step(
                count_table.time_start, forward_counts[bin_index], where="post",
                color=color, lw=.65, alpha=.65,
                label=("Per-bin forward 48 h counts" if bin_index == 0 else None),
            )
        self.ttff_count_ax.plot(
            count_table.time_center, count_table.n_valid, color="tab:orange", lw=.9,
            ls="--",
            label="N_valid",
        )
        for availability in count_table[
            ~count_table.forward_coverage_adequate
        ].itertuples():
            self.ttff_ax.axvspan(
                availability.time_start, availability.time_end,
                color="tab:red", alpha=.10,
            )
        self.ttff_count_ax.set_ylabel("Forward events / N_valid", color="tab:orange")
        self.ttff_ax.set_ylabel("TTFF value bins")
        status_counts = self.detection.ttff_bin_results.status.value_counts().to_dict()
        self.ttff_ax.set_title(
            f"TTFF bin cessation: {result.ttff_change_status} "
            f"(eligible={result.ttff_eligible_bin_count}, "
            f"stable={result.ttff_stable_drop_bin_count}, "
            f"agreeing={result.ttff_agreeing_bin_count}; {status_counts})"
        )
        self.strain_ax.plot(time, data.strain, ".", color=".6", ms=2, label="Raw strain")
        self.strain_ax.plot(time, data.rolling_median_strain, color="tab:green",
                            label="Rolling median")
        self.strain_ax.fill_between(
            time, data.rolling_q25_strain, data.rolling_q75_strain,
            color="tab:green", alpha=.18, label="Rolling q25-q75",
        )
        strain_metrics = self.detection.strain_candidates
        if not strain_metrics.empty:
            self.strain_metric_ax.plot(
                strain_metrics.time, strain_metrics.drop_absolute,
                color="tab:purple", lw=1.0, label="Absolute drop",
            )
            self.strain_metric_ax.plot(
                strain_metrics.time, strain_metrics.normalized_drop,
                color="tab:orange", lw=.8, label="Normalized drop",
            )
        self.strain_metric_ax.set_ylabel("Step diagnostics", color="tab:purple")
        self.strain_ax.set_ylabel("Strain")
        self.strain_ax.set_title(
            f"Persistent strain step: {result.strain_change_status} "
            f"(drop={result.strain_drop_absolute:.3g}, "
            f"relative={result.strain_drop_relative:.1%}, "
            f"normalized={result.strain_normalized_drop:.3g})"
        )
        self.temperature_ax.plot(time, data.hull_temperature, color="tab:red", lw=.7,
                                 label="Hull temperature")
        self.temperature_ax.plot(time, data.temperature_background, color="tab:orange", lw=1,
                                 label="Rolling median background")
        self.temperature_variability_ax.plot(time, data.temperature_rolling_mad,
                                              color="tab:purple", lw=.9,
                                              label="Local anomaly MAD")
        self.temperature_ax.set_ylabel("Hull temperature")
        self.temperature_variability_ax.set_ylabel("Local MAD", color="tab:purple")
        self.temperature_ax.set_title(
            f"Hull temperature context only: {result.temperature_context_status}"
        )
        events = []
        if not np.isnat(result.ttff_change_time):
            events.append((pd.Timestamp(result.ttff_change_time), "TTFF change", "tab:purple", "--"))
        if not np.isnat(result.strain_change_time):
            events.append((pd.Timestamp(result.strain_change_time), "Strain change", "tab:green", "-."))
        if automatic is not None:
            events.append((automatic, "Automatic candidate", "tab:red", "-"))
        for axis in (self.full_ax, self.ttff_ax, self.strain_ax, self.temperature_ax):
            drawn: dict[int, list[str]] = {}
            for event_time, label, color, style in events:
                key = int(event_time.value)
                if key in drawn:
                    drawn[key].append(label)
                    continue
                drawn[key] = [label]
                same = [item[1] for item in events if int(item[0].value) == key]
                axis.axvline(event_time, color=color, lw=1.6, ls=style,
                             label=" / ".join(same))
            if self.selected_time is not None:
                axis.axvline(self.selected_time, color="tab:blue", lw=1.2, ls="--",
                             label="Selected review time")
            axis.grid(alpha=.2)
        center = automatic if automatic is not None else self.selected_time
        if center is not None:
            half = pd.Timedelta(self.config.review_zoom_window)
            for axis in (self.ttff_ax, self.strain_ax, self.temperature_ax):
                axis.set_xlim(center - half, center + half)
        self.full_ax.legend(loc="upper right", fontsize=8)
        handles, labels = self.ttff_ax.get_legend_handles_labels()
        extra_handles, extra_labels = self.ttff_count_ax.get_legend_handles_labels()
        self.ttff_ax.legend(handles + extra_handles, labels + extra_labels,
                            loc="upper right", fontsize=8)
        strain_handles, strain_labels = self.strain_ax.get_legend_handles_labels()
        strain_extra_handles, strain_extra_labels = self.strain_metric_ax.get_legend_handles_labels()
        self.strain_ax.legend(strain_handles + strain_extra_handles,
                              strain_labels + strain_extra_labels,
                              loc="upper right", fontsize=8)
        temp_handles, temp_labels = self.temperature_ax.get_legend_handles_labels()
        temp_extra_handles, temp_extra_labels = self.temperature_variability_ax.get_legend_handles_labels()
        self.temperature_ax.legend(temp_handles + temp_extra_handles,
                                   temp_labels + temp_extra_labels,
                                   loc="upper right", fontsize=8)
        decision = self.reviews.rows.get(str(row.platform_code))
        decision_text = "unreviewed" if decision is None else decision["review_status"]
        self.figure.suptitle(
            f"{row.platform_code}  {self.cursor + 1}/{len(self.automatic)}  "
            f"status={row.auto_status} confidence={row.auto_confidence} review={decision_text}\n"
            f"TTFF eligible/stable/agreeing="
            f"{row.ttff_eligible_bin_count}/{row.ttff_stable_drop_bin_count}/"
            f"{row.ttff_agreeing_bin_count}, span={row.ttff_consensus_span_hours:.2f} h; "
            f"strain drop abs/relative/normalized="
            f"{row.strain_drop_absolute:.3g}/{row.strain_drop_relative:.1%}/"
            f"{row.strain_normalized_drop:.3g}; "
            f"component offset={row.ttff_strain_offset_hours:.2f} h"
        )
        self.figure.canvas.draw_idle()

    def action(self, action: str) -> None:
        if action == "accept":
            if self._automatic_time(self.automatic.iloc[self.cursor]) is None:
                self.status.set_text("No automatic candidate is available to accept.")
            else:
                self._set("accepted_auto", self._reason())
        elif action == "manual":
            self._set("manual_time", "manual_adjustment")
        elif action == "none":
            self._set("no_detectable_loss", "ambiguous")
        elif action == "uncertain":
            self._set("uncertain", "ambiguous")
        elif action in {"-6", "-1", "+1", "+6"}:
            self.selected_time += pd.Timedelta(hours=int(action))
            self._set("manual_time", "manual_adjustment")
        elif action in {"previous", "next"}:
            delta = -1 if action == "previous" else 1
            self.cursor = (self.cursor + delta) % len(self.automatic)
            self.loaded_platform = None
            self.status.set_text("")
            self.draw()
        elif action == "save":
            self.reviews.save()
            self.status.set_text(f"Saved {self.config.review_output}")
            self.figure.canvas.draw_idle()
        elif action == "quit":
            self.reviews.save()
            self.closed = True
            self.plt.close(self.figure)

    def on_click(self, event) -> None:
        if event.inaxes not in {self.full_ax, self.ttff_ax, self.ttff_count_ax, self.strain_ax,
                                self.strain_metric_ax,
                                self.temperature_ax, self.temperature_variability_ax}:
            return
        if event.xdata is None:
            return
        self.selected_time = pd.Timestamp(self.mdates.num2date(event.xdata)).tz_convert("UTC")
        self.status.set_text("Selected time changed; press Set manual to record it.")
        self.draw()

    def on_key(self, event) -> None:
        mapping = {"a": "accept", "m": "manual", "u": "uncertain", "n": "next",
                   "p": "previous", "s": "save", "q": "quit"}
        if event.key in mapping:
            self.action(mapping[event.key])

    def on_close(self, event) -> None:
        if not self.closed:
            self.reviews.save()
            self.closed = True

    def show(self) -> None:
        self.plt.show()
