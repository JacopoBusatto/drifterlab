"""Characterize raw ARCTERX drogue-related signals and show automatic change dates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from drifterlab.experiments.arcterx.raw_drogue import RawDrogueSignals, read_raw_drogue_signals
from drifterlab.experiments.arcterx.drogue_loss_review import DrogueLossReviews
from drifterlab.qc.drogue import (
    DrogueDetection, DrogueDetectionConfig, analysis_cutoff_time, detect_drogue_loss,
)


SPECIAL_TTFF_VALUE = 4095.0
DEFAULT_THRESHOLDS = (20.0, 40.0, 60.0, 120.0, 300.0, 1000.0)
RAW_VARIABLES = {
    "time": {"source_field": "dataset.drifter_<ID>.ObsTimestamp", "units": "UTC campaign time"},
    "ttff": {"source_field": "dataset.drifter_<ID>.GpsTTFF", "units": None,
             "note": "Raw units and the meaning of special values are not documented in this repository."},
    "strain": {"source_field": "dataset.drifter_<ID>.Drogue", "units": None,
               "note": "Treated as an uncalibrated strain-like value for this diagnostic."},
    "hull_temperature": {"source_field": "dataset.drifter_<ID>.HullTemperature", "units": None,
                         "note": "Raw units are not asserted by this diagnostic."},
}


@dataclass(frozen=True)
class DiagnosticSettings:
    rolling_window: str = "24h"
    strain_pre_window: str = "24h"
    strain_post_window: str = "24h"
    temperature_background_window: str = "24h"
    temperature_variability_window: str = "24h"
    minimum_observations: int = 12
    ttff_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS

    def __post_init__(self) -> None:
        for name in ("rolling_window", "strain_pre_window", "strain_post_window",
                     "temperature_background_window", "temperature_variability_window"):
            try:
                duration = pd.Timedelta(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a valid duration") from exc
            if duration <= pd.Timedelta(0):
                raise ValueError(f"{name} must be positive")
        if (isinstance(self.minimum_observations, bool)
                or not isinstance(self.minimum_observations, (int, np.integer))
                or self.minimum_observations < 1):
            raise ValueError("minimum_observations must be a positive integer")
        if not self.ttff_thresholds:
            raise ValueError("At least one TTFF diagnostic threshold is required")
        if any(not np.isfinite(value) for value in self.ttff_thresholds):
            raise ValueError("TTFF diagnostic thresholds must be finite")
        if len(set(self.ttff_thresholds)) != len(self.ttff_thresholds):
            raise ValueError("TTFF diagnostic thresholds must be unique")

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["ttff_thresholds"] = list(self.ttff_thresholds)
        return result


def _label(value: float) -> str:
    return f"{value:g}".replace("-", "minus").replace(".", "p")


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def _forward_median(series: pd.Series, window: str, minimum: int) -> pd.Series:
    time_ns = series.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    mirrored = pd.Timestamp("2000-01-01") + pd.to_timedelta(time_ns[-1] - time_ns[::-1], unit="ns")
    reversed_series = pd.Series(series.to_numpy()[::-1], index=mirrored)
    result = reversed_series.rolling(pd.Timedelta(window), min_periods=minimum).median()
    return result.iloc[::-1].set_axis(series.index)


def _plot_with_gaps(axis, time: Any, values: Any, max_gap: pd.Timedelta, **kwargs) -> None:
    """Plot observed segments without drawing lines across telemetry outages."""
    index = pd.DatetimeIndex(time)
    array = np.asarray(values, dtype=float)
    if not len(index):
        return
    boundaries = [0, *(np.flatnonzero(np.diff(index.asi8) > max_gap.value) + 1), len(index)]
    label = kwargs.pop("label", None)
    for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        axis.plot(
            index[start:stop], array[start:stop],
            label=label if segment == 0 else None, **kwargs,
        )


def rolling_diagnostics(signals: RawDrogueSignals,
                        settings: DiagnosticSettings = DiagnosticSettings()) -> pd.DataFrame:
    """Return independent, time-based diagnostics without classifying an event."""
    time = pd.to_datetime(signals.time, errors="coerce", utc=True)
    frame = pd.DataFrame({
        "time": time,
        "ttff": np.asarray(signals.ttff, dtype=float),
        "strain": (np.nan if signals.strain is None
                   else np.asarray(signals.strain, dtype=float)),
        "hull_temperature": (np.nan if signals.hull_temperature is None
                             else np.asarray(signals.hull_temperature, dtype=float)),
    })
    frame = frame[frame.time.notna()].sort_values("time", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{signals.platform_code}: no valid timestamps")
    index = pd.DatetimeIndex(frame.time)
    minimum = settings.minimum_observations
    window = pd.Timedelta(settings.rolling_window)
    ttff = pd.Series(frame.ttff.to_numpy(), index=index)
    ttff_rolling = ttff.rolling(window, min_periods=minimum)
    frame["rolling_median_ttff"] = ttff_rolling.median().to_numpy()
    frame["rolling_q90_ttff"] = ttff_rolling.quantile(.90).to_numpy()
    frame["rolling_q95_ttff"] = ttff_rolling.quantile(.95).to_numpy()
    frame["rolling_max_ttff"] = ttff_rolling.max().to_numpy()
    valid_count = ttff_rolling.count()
    for threshold in settings.ttff_thresholds:
        label = _label(threshold)
        indicator = pd.Series(np.where(ttff.notna(), (ttff > threshold).astype(float), np.nan), index=index)
        count = indicator.rolling(window, min_periods=minimum).sum()
        frame[f"rolling_count_ttff_gt_{label}"] = count.to_numpy()
        frame[f"rolling_fraction_ttff_gt_{label}"] = (count / valid_count).to_numpy()
    exact = pd.Series(np.where(ttff.notna(), (ttff == SPECIAL_TTFF_VALUE).astype(float), np.nan),
                      index=index)
    exact_count = exact.rolling(window, min_periods=minimum).sum()
    frame["rolling_count_ttff_4095"] = exact_count.to_numpy()
    frame["rolling_fraction_ttff_4095"] = (exact_count / valid_count).to_numpy()

    strain = pd.Series(frame.strain.to_numpy(), index=index)
    strain_rolling = strain.rolling(window, min_periods=minimum)
    frame["rolling_median_strain"] = strain_rolling.median().to_numpy()
    frame["rolling_q25_strain"] = strain_rolling.quantile(.25).to_numpy()
    frame["rolling_q75_strain"] = strain_rolling.quantile(.75).to_numpy()
    frame["rolling_mad_strain"] = strain_rolling.apply(_mad, raw=True).to_numpy()
    before = strain.rolling(pd.Timedelta(settings.strain_pre_window),
                            min_periods=minimum, closed="left").median()
    after = _forward_median(strain, settings.strain_post_window, minimum)
    frame["strain_delta"] = (after - before).to_numpy()

    temperature = pd.Series(frame.hull_temperature.to_numpy(), index=index)
    background = temperature.rolling(
        pd.Timedelta(settings.temperature_background_window),
        min_periods=minimum, center=True,
    ).median()
    anomaly = temperature - background
    temperature_mad = anomaly.rolling(
        pd.Timedelta(settings.temperature_variability_window),
        min_periods=minimum, center=True,
    ).apply(_mad, raw=True)
    frame["rolling_temperature_background"] = background.to_numpy()
    frame["temperature_anomaly"] = anomaly.to_numpy()
    frame["rolling_temperature_mad"] = temperature_mad.to_numpy()
    return frame


def _quantile(values: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if len(values) else np.nan


def summarize_drifter(signals: RawDrogueSignals, diagnostics: pd.DataFrame,
                      settings: DiagnosticSettings) -> dict[str, Any]:
    ttff = np.asarray(signals.ttff, dtype=float)
    valid_ttff = ttff[np.isfinite(ttff)]
    strain = np.array([], dtype=float) if signals.strain is None else np.asarray(signals.strain, dtype=float)
    valid_strain = strain[np.isfinite(strain)]
    temperature = (np.array([], dtype=float) if signals.hull_temperature is None
                   else np.asarray(signals.hull_temperature, dtype=float))
    valid_temperature = temperature[np.isfinite(temperature)]
    row: dict[str, Any] = {
        "platform_code": signals.platform_code,
        "source_filename": signals.source_path.name,
        "source_path": str(signals.source_path),
        "source_sha256": signals.source_sha256,
        "n_observations": len(signals.time),
        "n_valid_ttff": len(valid_ttff),
        "ttff_min": _quantile(valid_ttff, 0),
        "ttff_median": _quantile(valid_ttff, .5),
        "ttff_q75": _quantile(valid_ttff, .75),
        "ttff_q90": _quantile(valid_ttff, .90),
        "ttff_q95": _quantile(valid_ttff, .95),
        "ttff_q99": _quantile(valid_ttff, .99),
        "ttff_max": _quantile(valid_ttff, 1),
        "count_ttff_4095": int((valid_ttff == SPECIAL_TTFF_VALUE).sum()),
        "fraction_ttff_4095": (float((valid_ttff == SPECIAL_TTFF_VALUE).mean())
                               if len(valid_ttff) else np.nan),
        "strain_missing": not len(valid_strain),
        "n_valid_strain": len(valid_strain),
        "strain_min": _quantile(valid_strain, 0),
        "strain_median": _quantile(valid_strain, .5),
        "strain_q05": _quantile(valid_strain, .05),
        "strain_q25": _quantile(valid_strain, .25),
        "strain_q75": _quantile(valid_strain, .75),
        "strain_q95": _quantile(valid_strain, .95),
        "strain_max": _quantile(valid_strain, 1),
        "strain_delta_min": _quantile(diagnostics.strain_delta.to_numpy(), 0),
        "strain_delta_q05": _quantile(diagnostics.strain_delta.to_numpy(), .05),
        "hull_temperature_missing": not len(valid_temperature),
        "n_valid_hull_temperature": len(valid_temperature),
    }
    for threshold in settings.ttff_thresholds:
        label = _label(threshold)
        count = int((valid_ttff > threshold).sum())
        row[f"count_ttff_gt_{label}"] = count
        row[f"fraction_ttff_gt_{label}"] = count / len(valid_ttff) if len(valid_ttff) else np.nan
    return row


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {
        "count": len(values), "min": float(values.min()), "q25": float(values.quantile(.25)),
        "median": float(values.median()), "q75": float(values.quantile(.75)),
        "max": float(values.max()),
    }


def population_report(table: pd.DataFrame, exact_values: Counter,
                      settings: DiagnosticSettings) -> dict[str, Any]:
    valid_total = int(table.n_valid_ttff.sum())
    columns = ["ttff_min", "ttff_median", "ttff_q75", "ttff_q90", "ttff_q95",
               "ttff_q99", "ttff_max", "fraction_ttff_4095", "strain_min",
               "strain_median", "strain_q05", "strain_q95", "strain_max",
               "strain_delta_min"]
    columns += [f"fraction_ttff_gt_{_label(value)}" for value in settings.ttff_thresholds]
    frequent = []
    for value, count in exact_values.most_common(20):
        display = int(value) if float(value).is_integer() else float(value)
        frequent.append({"value": display, "count": count,
                         "fraction_of_valid_ttff": count / valid_total if valid_total else None})
    return {
        "raw_variable_metadata": RAW_VARIABLES,
        "settings": settings.serializable(),
        "number_of_drifters": len(table),
        "total_observations": int(table.n_observations.sum()),
        "total_valid_ttff": valid_total,
        "count_ttff_4095": int(table.count_ttff_4095.sum()),
        "fraction_ttff_4095": (float(table.count_ttff_4095.sum() / valid_total)
                               if valid_total else None),
        "missing_variable_counts": {
            "strain": int(table.strain_missing.sum()),
            "hull_temperature": int(table.hull_temperature_missing.sum()),
        },
        "across_drifter_distributions": {name: _distribution(table[name]) for name in columns},
        "most_frequent_exact_ttff_values": frequent,
    }


def plot_drifter(signals: RawDrogueSignals, data: pd.DataFrame, settings: DiagnosticSettings,
                 output: Path, detection: DrogueDetection | None = None, *,
                 review_row: dict[str, str] | None = None,
                 cutoff_margin_hours: float = 24) -> None:
    import matplotlib.pyplot as plt

    if detection is None:
        detection = detect_drogue_loss(
            signals.platform_code, signals.time, signals.ttff,
            strain=signals.strain,
        )
    return _plot_drifter_evidence(
        signals, data, output, detection, plt,
        review_row=review_row, cutoff_margin_hours=cutoff_margin_hours,
        temperature_gap=pd.Timedelta(settings.temperature_background_window),
    )


def _plot_drifter_evidence(signals: RawDrogueSignals, data: pd.DataFrame,
                           output: Path, detection: DrogueDetection, plt, *,
                           review_row: dict[str, str] | None = None,
                           cutoff_margin_hours: float = 24,
                           temperature_gap: pd.Timedelta = pd.Timedelta("24h")) -> None:
    result = detection.result
    figure, axes = plt.subplots(7, 1, figsize=(15, 18), sharex=True,
                               constrained_layout=True)
    time = data.time
    axes[0].plot(time, data.ttff, ".", color=".35", ms=2, alpha=.65)
    axes[0].set_yscale("symlog", linthresh=20)
    axes[0].set_ylabel("Raw GpsTTFF\n(unit unknown)")
    axes[0].set_title(f"Raw drogue-related signals - platform {signals.platform_code}")

    count_table = detection.ttff_counts
    raw_counts = np.stack(count_table.bin_counts.to_numpy()).T
    raw_counts = np.ma.masked_where(
        np.broadcast_to(count_table.n_valid.to_numpy() == 0, raw_counts.shape),
        raw_counts,
    )
    x_edges = [*count_table.time_start, count_table.time_end.iloc[-1]]
    axes[1].pcolormesh(x_edges, np.arange(raw_counts.shape[0] + 1), raw_counts,
                       shading="flat", cmap="viridis", vmin=0)
    axes[1].set_yticks(np.arange(len(detection.ttff_bins.labels)) + .5)
    axes[1].set_yticklabels(detection.ttff_bins.labels, fontsize=6)
    axes[1].set_ylabel("TTFF value bins")
    lower_policy = (
        "lower-than-first bin included"
        if detection.ttff_bins.include_lower_than_first_bin
        else "values below first edge excluded"
    )
    axes[1].set_title(f"Raw TTFF event counts; {lower_policy}; open upper bin included")
    for row in detection.ttff_bin_results.itertuples():
        if pd.notna(row.drop_time):
            axes[1].plot(row.drop_time, row.bin_index + .5, marker="v", ms=5,
                         color="white", mec="black")

    forward = np.stack(count_table.forward_counts.to_numpy()).T.astype(float)
    forward[:, ~count_table.forward_coverage_adequate.to_numpy()] = np.nan
    colors = plt.cm.viridis(np.linspace(.05, .95, len(detection.ttff_bins.labels)))
    for bin_index, (label, color) in enumerate(zip(detection.ttff_bins.labels, colors)):
        axes[2].step(count_table.time_start, forward[bin_index], where="post",
                     color=color, lw=.9, label=label)
    for row in detection.ttff_bin_results.itertuples():
        if pd.notna(row.drop_time):
            axes[2].axvline(row.drop_time, color=colors[row.bin_index], lw=.8, alpha=.7)
    availability_axis = axes[2].twinx()
    availability_axis.plot(count_table.time_center, count_table.n_valid,
                           color=".35", lw=.7, alpha=.7, label="N_valid")
    availability_axis.set_ylabel("All-valid TTFF count", color=".35")
    for row in count_table[~count_table.forward_coverage_adequate].itertuples():
        axes[2].axvspan(row.time_start, row.time_end, color="tab:red", alpha=.08)
    handles, labels = axes[2].get_legend_handles_labels()
    extra_handles, extra_labels = availability_axis.get_legend_handles_labels()
    axes[2].legend(handles + extra_handles, labels + extra_labels,
                   loc="upper right", fontsize=6, ncol=2)
    status_counts = detection.ttff_bin_results.status.value_counts().to_dict()
    axes[2].set_ylabel("Forward rolling\nevent count")
    axes[2].set_title(
        f"TTFF last-stable-drop: {result.ttff_status} ({result.ttff_detail_status}); "
        f"eligible={result.ttff_eligible_bin_count}, "
        f"agreeing={result.ttff_agreeing_bin_count}; bins={status_counts}"
    )

    axes[3].plot(time, data.strain, ".", color=".6", ms=2, alpha=.6,
                 label="Raw strain")
    blocks = detection.strain_blocks
    valid_blocks = blocks[blocks.strain_median.notna()]
    if not valid_blocks.empty:
        axes[3].plot(
            blocks.time_center, blocks.strain_median, "o-",
            color="tab:green", ms=2.5, lw=.8, label="Strain block median",
        )
    candidate = (
        None if np.isnat(detection.strain_candidate_time)
        else pd.Timestamp(detection.strain_candidate_time, tz="UTC")
    )
    if candidate is not None and not valid_blocks.empty:
        block_time = pd.DatetimeIndex(blocks.time_center)
        fitted = np.where(
            blocks.strain_median.notna(),
            np.where(block_time < candidate, result.strain_level_before,
                     result.strain_level_after),
            np.nan,
        )
        axes[3].plot(
            block_time, fitted, color="tab:blue", lw=1.8,
            label="Fitted pre/post levels",
        )
        axes[3].axvline(
            candidate, color=("tab:green" if result.strain_status == "clear" else ".35"),
            ls=":", lw=1.1, label="Best strain split",
        )
    axes[3].set_ylabel("Strain\n(unit unknown)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_title(
        f"Robust two-regime strain: {result.strain_status}; "
        f"levels={result.strain_level_before:.3g}->{result.strain_level_after:.3g}; "
        f"drop={result.strain_absolute_drop:.3g} ({result.strain_relative_drop:.1%})"
    )

    strain_metrics = detection.strain_objective
    if not strain_metrics.empty:
        objective_gap = (
            2 * (blocks.time_end.iloc[0] - blocks.time_start.iloc[0])
            if not blocks.empty else pd.Timedelta("12h")
        )
        _plot_with_gaps(
            axes[4], strain_metrics.time, strain_metrics.j1,
            objective_gap,
            color="tab:purple", label="Two-regime L1 cost",
        )
        if np.isfinite(detection.strain_null_objective):
            axes[4].axhline(
                detection.strain_null_objective,
                color="tab:orange", ls="--", lw=.8,
                label="One-level L1 cost",
            )
        axes[4].legend(loc="upper right", fontsize=8)
    axes[4].set_ylabel("L1 objective")
    axes[4].set_title(
        f"Strain fit improvement={result.strain_fit_improvement:.1%}; "
        f"blocks before/after/valid={result.strain_n_blocks_before}/"
        f"{result.strain_n_blocks_after}/{result.strain_n_valid_blocks}"
    )

    axes[5].plot(time, data.hull_temperature, ".", color=".55", ms=1.5,
                 label="Raw hull temperature")
    _plot_with_gaps(
        axes[5], time, data.rolling_temperature_background,
        temperature_gap,
        color="tab:orange", label="Rolling median background",
    )
    axes[5].set_ylabel("Hull temperature\n(unit unknown)")
    axes[5].legend(loc="upper right", fontsize=8)
    _plot_with_gaps(
        axes[6], time, data.rolling_temperature_mad,
        temperature_gap,
        color="tab:purple", lw=.8,
    )
    axes[6].set_ylabel("Temperature\nanomaly MAD")
    axes[6].set_xlabel("UTC time")

    change_lines = [
        (result.ttff_change_time, "TTFF change", "tab:purple", "--", 1.4),
        (result.strain_change_time, "Strain change", "tab:green", "-.", 1.4),
        (result.auto_drogue_loss_time, "Automatic decision", "tab:red", "-", 1.8),
    ]
    if review_row is None:
        reviewed = result.auto_drogue_loss_time
        cutoff = analysis_cutoff_time(reviewed, cutoff_margin_hours)
        reviewed_label = "Default physical loss"
    else:
        reviewed_value = review_row.get("reviewed_drogue_loss_time", "")
        cutoff_value = review_row.get("analysis_cutoff_time", "")
        reviewed = (
            np.datetime64("NaT", "ns") if not reviewed_value
            else pd.Timestamp(reviewed_value).to_datetime64()
        )
        cutoff = (
            np.datetime64("NaT", "ns") if not cutoff_value
            else pd.Timestamp(cutoff_value).to_datetime64()
        )
        reviewed_label = "Reviewed physical loss"
    change_lines.extend([
        (reviewed, reviewed_label, "tab:blue", "--", 1.4),
        (cutoff, "Analysis cutoff", "black", ":", 1.4),
    ])
    for axis in axes:
        drawn = set()
        for value, label, color, style, width in change_lines:
            if np.isnat(value):
                continue
            timestamp = pd.Timestamp(value)
            key = int(timestamp.value)
            if key in drawn:
                continue
            drawn.add(key)
            same = [item[1] for item in change_lines
                    if not np.isnat(item[0]) and int(pd.Timestamp(item[0]).value) == key]
            axis.axvline(timestamp, label=" / ".join(same), color=color,
                         ls=style, lw=width)
        axis.grid(alpha=.2)
    date_handles, date_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(date_handles, date_labels, loc="upper right", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_population(table: pd.DataFrame, ttff_values: np.ndarray,
                    settings: DiagnosticSettings, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    finite = ttff_values[np.isfinite(ttff_values)]
    if len(finite):
        upper = np.quantile(finite, .99)
        axes[0, 0].hist(finite[finite <= upper], bins=80, color="steelblue")
        axes[0, 0].set_title(f"Raw TTFF through population q99 ({upper:g})")
        axes[0, 0].set_yscale("log")
    axes[0, 1].hist(table.fraction_ttff_4095.dropna(), bins=30, color=".4")
    axes[0, 1].set_title("Across-drifter fraction TTFF == 4095")
    threshold = 60.0 if 60.0 in settings.ttff_thresholds else settings.ttff_thresholds[0]
    axes[1, 0].hist(table[f"fraction_ttff_gt_{_label(threshold)}"].dropna(),
                    bins=30, color="tab:orange")
    axes[1, 0].set_title(f"Across-drifter fraction TTFF > {threshold:g}")
    axes[1, 1].hist(table.strain_delta_min.dropna(), bins=40, color="tab:red")
    axes[1, 1].set_title("Minimum local strain median change per drifter")
    for axis in axes.reshape(-1):
        axis.grid(alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_config(
    path: Path,
) -> tuple[Path, str, float, DrogueDetectionConfig, Path | None, float]:
    path = path.resolve()
    with path.open(encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    try:
        directory_value = config["input"]["directory"]
    except (TypeError, KeyError) as exc:
        raise ValueError("Configuration requires input.directory") from exc
    directory = Path(directory_value).expanduser()
    if not directory.is_absolute():
        directory = (path.parent / directory).resolve()
    pattern = config.get("input", {}).get("pattern", "*.mat")
    missing = config.get("processing", {}).get("missing_value", -999)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("input.pattern must be a nonempty string")
    if isinstance(missing, bool) or not isinstance(missing, (int, float)) or not np.isfinite(missing):
        raise ValueError("processing.missing_value must be finite")
    detection = config.get("detection", {})
    if not isinstance(detection, dict):
        raise ValueError("detection must be a mapping")
    try:
        detection_config = DrogueDetectionConfig.from_dict(detection)
    except TypeError as exc:
        raise ValueError(f"Invalid detection configuration: {exc}") from exc
    review_value = config.get("output", {}).get("review")
    review_path = None
    if isinstance(review_value, str) and review_value.strip():
        review_path = Path(review_value).expanduser()
        if not review_path.is_absolute():
            review_path = (path.parent / review_path).resolve()
    margin = config.get("processing", {}).get("analysis_cutoff_margin_hours", 24)
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not np.isfinite(margin) or margin < 0):
        raise ValueError(
            "processing.analysis_cutoff_margin_hours must be finite and nonnegative"
        )
    return directory, pattern, float(missing), detection_config, review_path, float(margin)


def _select_files(files: list[Path], platforms: list[str], missing_value: float) -> list[Path]:
    if not platforms:
        return files
    requested = set(map(str, platforms))
    selected = {path.stem: path for path in files if path.stem in requested}
    unresolved = requested - set(selected)
    if unresolved:
        for path in files:
            if path in selected.values():
                continue
            signals = read_raw_drogue_signals(path, missing_value=missing_value)
            if signals.platform_code in unresolved:
                selected[signals.platform_code] = path
                unresolved.remove(signals.platform_code)
                if not unresolved:
                    break
    if unresolved:
        raise ValueError(f"Requested platforms not found: {sorted(unresolved)}")
    return [selected[platform] for platform in platforms]


def run(config_path: Path, output: Path, *, platforms: list[str] | None = None,
        all_figures: bool = False, population_figures: bool = True,
        settings: DiagnosticSettings = DiagnosticSettings()) -> pd.DataFrame:
    (input_directory, pattern, missing_value, detection_config,
     review_path, cutoff_margin_hours) = _input_config(config_path)
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    output = output.resolve()
    if output == input_directory or input_directory in output.parents:
        raise ValueError("Diagnostic output must be outside the raw input directory")
    files = sorted(path for path in input_directory.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No input files match {pattern!r} in {input_directory}")
    platforms = list(dict.fromkeys(platforms or []))
    if platforms and all_figures:
        raise ValueError("Use either --platform or --all, not both")
    files = _select_files(files, platforms, missing_value)
    reviews = (
        DrogueLossReviews(review_path, cutoff_margin_hours=cutoff_margin_hours)
        if review_path is not None and review_path.exists() else None
    )
    make_drifter_figures = bool(platforms) or all_figures
    print(json.dumps(RAW_VARIABLES, indent=2))
    rows, exact_values, ttff_chunks = [], Counter(), []
    seen: set[str] = set()
    for index, path in enumerate(files, start=1):
        signals = read_raw_drogue_signals(path, missing_value=missing_value)
        if signals.platform_code in seen:
            raise ValueError(f"Duplicate raw platform ID: {signals.platform_code}")
        seen.add(signals.platform_code)
        diagnostics = rolling_diagnostics(signals, settings)
        rows.append(summarize_drifter(signals, diagnostics, settings))
        valid_ttff = np.asarray(signals.ttff, dtype=float)
        valid_ttff = valid_ttff[np.isfinite(valid_ttff)]
        exact_values.update(map(float, valid_ttff))
        ttff_chunks.append(valid_ttff)
        if make_drifter_figures:
            detection = detect_drogue_loss(
                signals.platform_code, signals.time, signals.ttff,
                strain=signals.strain,
                config=detection_config,
            )
            review_row = None if reviews is None else reviews.rows.get(signals.platform_code)
            if (review_row is not None
                    and review_row["source_sha256"] != signals.source_sha256):
                raise ValueError(
                    f"Raw source changed for reviewed platform {signals.platform_code}"
                )
            plot_drifter(
                signals, diagnostics, settings,
                output / "figures" / f"drogue_signals_{signals.platform_code}.png",
                detection, review_row=review_row,
                cutoff_margin_hours=cutoff_margin_hours,
            )
        print(f"Diagnosed {index}/{len(files)}: {signals.platform_code}", flush=True)
    table = pd.DataFrame(rows).sort_values("platform_code", kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output / "ttff_population_summary.parquet", engine="pyarrow", index=False)
    table.to_csv(output / "ttff_population_summary.csv", index=False)
    report = population_report(table, exact_values, settings)
    _atomic_text(output / "population_summary.json", json.dumps(report, indent=2) + "\n")
    _atomic_text(output / "raw_variable_metadata.json",
                 json.dumps(RAW_VARIABLES, indent=2) + "\n")
    if population_figures and len(table) > 1:
        values = np.concatenate(ttff_chunks) if ttff_chunks else np.array([])
        plot_population(table, values, settings, output / "population_distributions.png")
    return table


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("config", type=Path, help="ARCTERX drogue YAML; only raw input and missing value are read")
    result.add_argument("--output", type=Path, default=Path("data/diagnostics/drogue_signals"))
    result.add_argument("--platform", action="append", default=[], help="platform ID to diagnose and plot; repeatable")
    result.add_argument("--all", action="store_true", help="explicitly generate a figure for every input drifter")
    result.add_argument("--window", default="24h", help="time window for rolling TTFF/strain diagnostics")
    result.add_argument("--strain-pre-window", default="24h")
    result.add_argument("--strain-post-window", default="24h")
    result.add_argument("--temperature-background-window", default="24h")
    result.add_argument("--temperature-variability-window", default="24h")
    result.add_argument("--minimum-observations", type=int, default=12)
    result.add_argument("--ttff-thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    result.add_argument("--no-population-figures", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = DiagnosticSettings(
            rolling_window=args.window,
            strain_pre_window=args.strain_pre_window,
            strain_post_window=args.strain_post_window,
            temperature_background_window=args.temperature_background_window,
            temperature_variability_window=args.temperature_variability_window,
            minimum_observations=args.minimum_observations,
            ttff_thresholds=tuple(args.ttff_thresholds),
        )
        table = run(args.config, args.output, platforms=args.platform, all_figures=args.all,
                    population_figures=not args.no_population_figures, settings=settings)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser().exit(1, f"Drogue signal diagnostic: {exc}\n")
    print(f"Summary rows: {len(table)}")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
