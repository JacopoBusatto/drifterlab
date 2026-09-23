"""TTFF per-bin cessation and strain-step drogue detection plus analysis masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValueBinningConfig:
    """Value bins and temporal cadence for one signal."""

    scale: str
    first_edge: float
    factor: float | None
    width: float | None
    last_edge: float
    time_bin_hours: float
    include_lower_than_first_bin: bool = True

    def __post_init__(self) -> None:
        if self.scale not in {"log", "linear"}:
            raise ValueError("binning.scale must be 'log' or 'linear'")
        if not isinstance(self.include_lower_than_first_bin, (bool, np.bool_)):
            raise ValueError("binning.include_lower_than_first_bin must be boolean")
        for name in ("first_edge", "last_edge", "time_bin_hours"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"binning.{name} must be finite")
        if self.last_edge <= self.first_edge:
            raise ValueError("binning.last_edge must exceed first_edge")
        if self.time_bin_hours <= 0:
            raise ValueError("binning.time_bin_hours must be positive")
        if self.scale == "log":
            if self.first_edge <= 0:
                raise ValueError("log binning.first_edge must be strictly positive")
            if self.factor is None or isinstance(self.factor, bool) or not np.isfinite(self.factor):
                raise ValueError("log binning.factor must be finite")
            if self.factor <= 1:
                raise ValueError("log binning.factor must exceed one")
            if self.width is not None:
                raise ValueError("log binning.width must be null")
        else:
            if self.width is None or isinstance(self.width, bool) or not np.isfinite(self.width):
                raise ValueError("linear binning.width must be finite")
            if self.width <= 0:
                raise ValueError("linear binning.width must be positive")
            if self.factor is not None:
                raise ValueError("linear binning.factor must be null")


@dataclass(frozen=True)
class TTFFCessationComparisonConfig:
    """Activity, quiet-state, persistence, and cross-bin agreement settings."""

    rolling_count_window_hours: float = 48.0
    pre_activity_window_hours: float = 168.0
    min_pre_events: int = 12
    min_pre_occupied_time_bins: int = 3
    min_coverage_fraction: float = .75
    quiet_fraction_of_pre_rate: float = .10
    max_events_in_quiet_window: int = 1
    persistence_hours: float = 120.0
    reactivation_window_hours: float = 48.0
    agreement_tolerance_hours: float = 48.0
    min_agreeing_bins: int = 2

    def __post_init__(self) -> None:
        durations = (
            "rolling_count_window_hours", "pre_activity_window_hours",
            "persistence_hours", "reactivation_window_hours",
            "agreement_tolerance_hours",
        )
        for name in durations:
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"comparison.{name} must be finite and positive")
        for name in (
            "min_pre_events", "min_pre_occupied_time_bins",
            "max_events_in_quiet_window", "min_agreeing_bins",
        ):
            value = getattr(self, name)
            lower = 0 if name == "max_events_in_quiet_window" else 1
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < lower):
                raise ValueError(f"comparison.{name} must be an integer >= {lower}")
        for name in ("min_coverage_fraction", "quiet_fraction_of_pre_rate"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"comparison.{name} must be in (0, 1]")
        if self.pre_activity_window_hours < self.rolling_count_window_hours:
            raise ValueError(
                "comparison.pre_activity_window_hours cannot be shorter than "
                "rolling_count_window_hours"
            )
        if self.persistence_hours < self.rolling_count_window_hours:
            raise ValueError(
                "comparison.persistence_hours cannot be shorter than "
                "rolling_count_window_hours"
            )


@dataclass(frozen=True)
class TTFFCessationConfig:
    binning: ValueBinningConfig
    comparison: TTFFCessationComparisonConfig

    def __post_init__(self) -> None:
        cadence = self.binning.time_bin_hours
        aligned = (
            "rolling_count_window_hours", "pre_activity_window_hours",
            "persistence_hours", "reactivation_window_hours",
        )
        for name in aligned:
            value = getattr(self.comparison, name)
            if not np.isclose(value / cadence, round(value / cadence)):
                raise ValueError(
                    f"comparison.{name} must be a multiple of binning.time_bin_hours"
                )


@dataclass(frozen=True)
class StrainStepConfig:
    """Time windows and thresholds for persistent strain level drops."""

    window: str = "12h"
    pre_window: str = "48h"
    post_window: str = "48h"
    persistence_window: str = "48h"
    minimum_drop_absolute: float = 2.0
    minimum_drop_relative: float = .10
    minimum_normalized_drop: float = 1.5
    weak_drop_absolute: float = 1.0
    weak_drop_relative: float = .05
    weak_normalized_drop: float = .75
    relative_floor: float = 1.0
    variability_floor: float = 1.0
    persistence_tolerance: float = 1.0

    def __post_init__(self) -> None:
        for name in ("window", "pre_window", "post_window", "persistence_window"):
            try:
                duration = pd.Timedelta(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"strain.{name} must be a valid duration") from exc
            if duration <= pd.Timedelta(0):
                raise ValueError(f"strain.{name} must be positive")
        nonnegative = (
            "minimum_drop_absolute", "minimum_drop_relative",
            "minimum_normalized_drop", "weak_drop_absolute",
            "weak_drop_relative", "weak_normalized_drop", "persistence_tolerance",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"strain.{name} must be finite and nonnegative")
        for name in ("relative_floor", "variability_floor"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"strain.{name} must be finite and positive")
        pairs = (
            ("minimum_drop_absolute", "weak_drop_absolute"),
            ("minimum_drop_relative", "weak_drop_relative"),
            ("minimum_normalized_drop", "weak_normalized_drop"),
        )
        for minimum, weak in pairs:
            if getattr(self, minimum) < getattr(self, weak):
                raise ValueError(f"strain.{minimum} cannot be less than strain.{weak}")


def _default_ttff() -> TTFFCessationConfig:
    return TTFFCessationConfig(
        ValueBinningConfig("log", 10.0, 1.5, None, 1000.0, 12.0),
        TTFFCessationComparisonConfig(),
    )


@dataclass(frozen=True)
class DrogueDetectionConfig:
    """Settings for TTFF-bin cessation, strain steps, and contextual temperature."""

    ttff: TTFFCessationConfig = field(default_factory=_default_ttff)
    strain: StrainStepConfig = field(default_factory=StrainStepConfig)
    temperature_background_window: str = "24h"
    temperature_variability_window: str = "6h"

    def __post_init__(self) -> None:
        for name in ("temperature_background_window", "temperature_variability_window"):
            try:
                duration = pd.Timedelta(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a valid duration") from exc
            if duration <= pd.Timedelta(0):
                raise ValueError(f"{name} must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrogueDetectionConfig:
        if not isinstance(data, dict):
            raise ValueError("detection must be a mapping")
        allowed = {"ttff", "strain", "temperature_background_window",
                   "temperature_variability_window"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown detection keys: {sorted(unknown)}")

        raw_ttff = data.get("ttff")
        if raw_ttff is None:
            ttff = _default_ttff()
        else:
            if not isinstance(raw_ttff, dict):
                raise ValueError("detection.ttff must be a mapping")
            unknown_ttff = set(raw_ttff) - {"binning", "comparison"}
            if unknown_ttff:
                raise ValueError(
                    f"Unknown detection.ttff keys: {sorted(unknown_ttff)}"
                )

            def ttff_section(section: str, default_object: Any,
                             object_type: type) -> Any:
                values = raw_ttff.get(section)
                if values is None:
                    return default_object
                if not isinstance(values, dict):
                    raise ValueError(f"detection.ttff.{section} must be a mapping")
                valid = set(default_object.__dataclass_fields__)
                unknown_nested = set(values) - valid
                if unknown_nested:
                    raise ValueError(
                        f"Unknown detection.ttff.{section} keys: {sorted(unknown_nested)}"
                    )
                return object_type(**{**asdict(default_object), **values})

            default_ttff = _default_ttff()
            ttff = TTFFCessationConfig(
                ttff_section("binning", default_ttff.binning, ValueBinningConfig),
                ttff_section(
                    "comparison", default_ttff.comparison,
                    TTFFCessationComparisonConfig,
                ),
            )

        raw_strain = data.get("strain")
        if raw_strain is None:
            strain = StrainStepConfig()
        else:
            if not isinstance(raw_strain, dict):
                raise ValueError("detection.strain must be a mapping")
            valid_strain = set(StrainStepConfig.__dataclass_fields__)
            unknown_strain = set(raw_strain) - valid_strain
            if unknown_strain:
                raise ValueError(
                    f"Unknown detection.strain keys: {sorted(unknown_strain)}"
                )
            strain = StrainStepConfig(**{
                **asdict(StrainStepConfig()), **raw_strain,
            })

        return cls(
            ttff=ttff,
            strain=strain,
            temperature_background_window=data.get(
                "temperature_background_window", "24h"
            ),
            temperature_variability_window=data.get(
                "temperature_variability_window", "6h"
            ),
        )


@dataclass(frozen=True)
class ValueBins:
    physical_edges: np.ndarray
    histogram_edges: np.ndarray
    labels: tuple[str, ...]
    scale: str
    include_lower_than_first_bin: bool


@dataclass
class TTFFCessationDetection:
    status: str
    selected: pd.Series | None
    bin_results: pd.DataFrame
    temporal_counts: pd.DataFrame
    bins: ValueBins


@dataclass
class StrainStepDetection:
    status: str
    selected: pd.Series | None
    candidates: pd.DataFrame


@dataclass(frozen=True)
class DrogueDetectionResult:
    platform_code: str
    auto_drogue_loss_time: np.datetime64
    auto_status: str
    auto_confidence: str
    ttff_strain_relation: str
    ttff_change_time: np.datetime64
    ttff_change_status: str
    ttff_eligible_bin_count: int
    ttff_stable_drop_bin_count: int
    ttff_agreeing_bin_count: int
    ttff_consensus_span_hours: float
    ttff_agreeing_pre_events: int
    ttff_agreeing_post_events: int
    ttff_min_persistence_coverage_fraction: float
    strain_change_time: np.datetime64
    strain_change_strength: float
    strain_change_status: str
    strain_drop_absolute: float
    strain_drop_relative: float
    strain_normalized_drop: float
    strain_median_before: float
    strain_median_after: float
    strain_mad_before: float
    strain_mad_after: float
    strain_q25_before: float
    strain_q75_before: float
    strain_q25_after: float
    strain_q75_after: float
    strain_pre_coverage_fraction: float
    strain_post_coverage_fraction: float
    strain_followup_coverage_fraction: float
    ttff_strain_offset_hours: float
    temperature_context_status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DrogueDetection:
    result: DrogueDetectionResult
    diagnostics: pd.DataFrame
    ttff_bin_results: pd.DataFrame
    strain_candidates: pd.DataFrame
    ttff_counts: pd.DataFrame
    ttff_bins: ValueBins


def make_value_bins(config: ValueBinningConfig) -> ValueBins:
    """Create finite comparison edges and configured open boundary bins."""
    if config.scale == "log":
        generated = [float(config.first_edge)]
        while generated[-1] * float(config.factor) < config.last_edge * (1 - 1e-12):
            generated.append(generated[-1] * float(config.factor))
        if not np.isclose(generated[-1], config.last_edge):
            generated.append(float(config.last_edge))
        else:
            generated[-1] = float(config.last_edge)
        if config.include_lower_than_first_bin:
            physical = np.array([0.0, *generated], dtype=float)
        else:
            physical = np.array(generated, dtype=float)
    else:
        count = int(np.floor((config.last_edge - config.first_edge) / float(config.width)))
        generated = config.first_edge + np.arange(count + 1) * float(config.width)
        generated = generated[generated < config.last_edge * (1 - 1e-12)]
        physical = np.r_[generated, float(config.last_edge)].astype(float)
    if len(physical) < 2 or np.any(np.diff(physical) <= 0):
        raise ValueError("Configured binning did not produce increasing finite edges")
    if config.include_lower_than_first_bin:
        histogram = np.r_[-np.inf, physical, np.inf]
        labels = [f"<{physical[0]:g}"]
    else:
        histogram = np.r_[physical, np.inf]
        labels = []
    labels.extend(
        f"[{left:g}, {right:g})" for left, right in zip(physical[:-1], physical[1:])
    )
    labels.append(f">={physical[-1]:g}")
    return ValueBins(
        physical, histogram, tuple(labels), config.scale,
        config.include_lower_than_first_bin,
    )


def _valid_values(values: Any, missing_values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    valid = np.isfinite(array)
    for sentinel in missing_values:
        valid &= array != float(sentinel)
    return valid


def histogram_counts(values: Any, bins: ValueBins, *,
                     missing_values: Iterable[float] = ()) -> tuple[np.ndarray, int]:
    """Count eligible finite, non-sentinel values, including upper overflow."""
    array = np.asarray(values, dtype=float).reshape(-1)
    valid = _valid_values(array, missing_values)
    if not bins.include_lower_than_first_bin:
        valid &= array >= bins.physical_edges[0]
    counts = np.histogram(array[valid], bins=bins.histogram_edges)[0].astype(np.int64)
    return counts, int(valid.sum())


def _nominal_interval_ns(time_ns: np.ndarray, cadence_ns: int) -> int:
    unique = np.unique(time_ns)
    differences = np.diff(unique)
    differences = differences[differences > 0]
    if not len(differences):
        return 0
    return int(min(np.median(differences), cadence_ns))


def _coverage_hours(time_ns: np.ndarray, start_ns: int, end_ns: int,
                    nominal_ns: int) -> float:
    if not len(time_ns) or nominal_ns <= 0:
        return 0.0
    half = nominal_ns // 2
    intervals = sorted(
        (max(start_ns, int(value) - half), min(end_ns, int(value) + nominal_ns - half))
        for value in np.unique(time_ns)
    )
    total = 0
    current_start, current_end = intervals[0]
    for left, right in intervals[1:]:
        if left <= current_end:
            current_end = max(current_end, right)
        else:
            total += max(0, current_end - current_start)
            current_start, current_end = left, right
    total += max(0, current_end - current_start)
    return total / 3_600_000_000_000


def _prepared(time: Any, values: Any, missing_values: Iterable[float]
              ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    raw_time = np.asarray(time).reshape(-1)
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) != len(raw_time):
        raise ValueError("values length does not match time length")
    parsed = pd.to_datetime(raw_time, errors="coerce", utc=True)
    valid_time = ~pd.isna(parsed)
    unsorted = pd.DatetimeIndex(parsed[valid_time])
    original_positions = np.flatnonzero(valid_time)
    order = np.argsort(
        unsorted.to_numpy(dtype="datetime64[ns]").astype(np.int64), kind="stable"
    )
    parsed = unsorted[order]
    array = array[original_positions[order]]
    valid_value = _valid_values(array, missing_values)
    return parsed, array, valid_value


def temporal_event_counts(time: Any, values: Any, config: TTFFCessationConfig, *,
                          missing_values: Iterable[float] = ()) -> tuple[pd.DataFrame, ValueBins]:
    """Return raw per-bin TTFF counts and all-valid observation availability."""
    index, array, valid_all = _prepared(time, values, missing_values)
    if not len(index):
        raise ValueError("time contains no valid timestamps")
    bins = make_value_bins(config.binning)
    cadence_ns = int(round(config.binning.time_bin_hours * 3_600_000_000_000))
    rolling_ns = int(round(
        config.comparison.rolling_count_window_hours * 3_600_000_000_000
    ))
    time_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    start_ns = int(time_ns.min() // cadence_ns * cadence_ns)
    stop_ns = int(time_ns.max() // cadence_ns * cadence_ns + cadence_ns)
    starts = np.arange(start_ns, stop_ns, cadence_ns, dtype=np.int64)
    nominal_ns = _nominal_interval_ns(time_ns[valid_all], cadence_ns)
    rows: list[dict[str, Any]] = []
    for left in starts:
        right = int(left + cadence_ns)
        selected = (time_ns >= left) & (time_ns < right) & valid_all
        counts, _ = histogram_counts(array[selected], bins)
        coverage = _coverage_hours(time_ns[selected], int(left), right, nominal_ns)
        rows.append({
            "time_start": pd.Timestamp(left, unit="ns", tz="UTC"),
            "time_end": pd.Timestamp(right, unit="ns", tz="UTC"),
            "time_center": pd.Timestamp(left + cadence_ns // 2, unit="ns", tz="UTC"),
            "bin_counts": counts,
            "n_valid": int(selected.sum()),
            "coverage_hours": coverage,
            "coverage_fraction": min(1.0, coverage / config.binning.time_bin_hours),
            "coverage_adequate": (
                coverage / config.binning.time_bin_hours
                >= config.comparison.min_coverage_fraction
            ),
        })
    table = pd.DataFrame(rows)
    record_end_ns = int(time_ns.max() + nominal_ns)
    forward_counts, forward_coverage, forward_complete = [], [], []
    for left in starts:
        right = int(left + rolling_ns)
        selected_rows = (starts >= left) & (starts < right)
        counts = np.stack(table.loc[selected_rows, "bin_counts"].to_numpy()).sum(axis=0)
        coverage = float(table.loc[selected_rows, "coverage_hours"].sum())
        forward_counts.append(counts.astype(np.int64))
        forward_coverage.append(min(
            1.0, coverage / config.comparison.rolling_count_window_hours
        ))
        forward_complete.append(record_end_ns >= right)
    table["forward_counts"] = forward_counts
    table["forward_coverage_fraction"] = forward_coverage
    table["forward_coverage_adequate"] = (
        table.forward_coverage_fraction >= config.comparison.min_coverage_fraction
    )
    table["forward_complete"] = forward_complete
    return table, bins


def _count_window(table: pd.DataFrame, bin_index: int, start: pd.Timestamp,
                  end: pd.Timestamp) -> dict[str, float | int]:
    selected = (table.time_start >= start) & (table.time_start < end)
    rows = table.loc[selected]
    counts = np.array([
        int(np.asarray(value)[bin_index]) for value in rows.bin_counts
    ], dtype=np.int64)
    duration_hours = (end - start) / pd.Timedelta(hours=1)
    coverage_hours = float(rows.coverage_hours.sum())
    return {
        "events": int(counts.sum()),
        "occupied_time_bins": int(np.count_nonzero(counts)),
        "coverage_hours": coverage_hours,
        "coverage_fraction": min(1.0, coverage_hours / float(duration_hours)),
    }


def _quiet_window(window: dict[str, float | int], pre_rate: float,
                  config: TTFFCessationComparisonConfig) -> bool:
    events = int(window["events"])
    coverage_hours = float(window["coverage_hours"])
    rate = events / coverage_hours if coverage_hours > 0 else np.inf
    return (
        events <= config.max_events_in_quiet_window
        or rate <= config.quiet_fraction_of_pre_rate * pre_rate
    )


def _bin_boundaries(bins: ValueBins, index: int) -> tuple[float, float]:
    return float(bins.histogram_edges[index]), float(bins.histogram_edges[index + 1])


def _per_bin_last_stable_drop(table: pd.DataFrame, bins: ValueBins, bin_index: int,
                              config: TTFFCessationConfig,
                              record_end: pd.Timestamp) -> dict[str, Any]:
    comparison = config.comparison
    cadence = pd.Timedelta(hours=config.binning.time_bin_hours)
    rolling = pd.Timedelta(hours=comparison.rolling_count_window_hours)
    pre_duration = pd.Timedelta(hours=comparison.pre_activity_window_hours)
    persistence = pd.Timedelta(hours=comparison.persistence_hours)
    reactivation = pd.Timedelta(hours=comparison.reactivation_window_hours)
    starts = pd.DatetimeIndex(table.time_start)
    ever_eligible = False
    saw_incomplete = False
    saw_reactivation = False
    fallback: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None

    for candidate in reversed(starts[1:]):
        pre = _count_window(table, bin_index, candidate - pre_duration, candidate)
        eligible = (
            pre["events"] >= comparison.min_pre_events
            and pre["occupied_time_bins"] >= comparison.min_pre_occupied_time_bins
            and pre["coverage_fraction"] >= comparison.min_coverage_fraction
        )
        if not eligible:
            continue
        ever_eligible = True
        pre_rate = float(pre["events"]) / float(pre["coverage_hours"])
        previous = _count_window(
            table, bin_index, candidate - cadence, candidate - cadence + rolling
        )
        post = _count_window(table, bin_index, candidate, candidate + rolling)
        if (previous["coverage_fraction"] < comparison.min_coverage_fraction
                or _quiet_window(previous, pre_rate, comparison)):
            continue
        if post["coverage_fraction"] < comparison.min_coverage_fraction:
            if post["events"] <= comparison.max_events_in_quiet_window:
                saw_incomplete = True
                fallback = fallback or {"time": candidate, "pre": pre, "post": post}
            continue
        if not _quiet_window(post, pre_rate, comparison):
            continue

        evidence = {"time": candidate, "pre": pre, "post": post}
        persistence_end = candidate + persistence
        if record_end < persistence_end:
            saw_incomplete = True
            fallback = fallback or evidence
            continue

        persistence_coverages: list[float] = []
        persistence_quiet = True
        window_start = candidate
        while window_start + rolling <= persistence_end:
            window = _count_window(
                table, bin_index, window_start, window_start + rolling
            )
            persistence_coverages.append(float(window["coverage_fraction"]))
            if (window["coverage_fraction"] < comparison.min_coverage_fraction
                    or not _quiet_window(window, pre_rate, comparison)):
                persistence_quiet = False
                break
            window_start += cadence
        if not persistence_quiet:
            if (persistence_coverages
                    and min(persistence_coverages) < comparison.min_coverage_fraction):
                saw_incomplete = True
                fallback = fallback or evidence
            continue

        reactivated = False
        confirmed_until = persistence_end
        window_start = persistence_end
        while window_start + reactivation <= record_end:
            window = _count_window(
                table, bin_index, window_start, window_start + reactivation
            )
            if window["coverage_fraction"] >= comparison.min_coverage_fraction:
                if not _quiet_window(window, pre_rate, comparison):
                    reactivated = True
                    break
                confirmed_until = window_start + reactivation
            window_start += cadence
        if reactivated:
            saw_reactivation = True
            fallback = fallback or evidence
            continue

        selected = {
            **evidence,
            "persistence_min_coverage_fraction": min(persistence_coverages),
            "confirmed_until": confirmed_until,
        }
        break

    lower, upper = _bin_boundaries(bins, bin_index)
    base = {
        "bin_index": bin_index,
        "bin_label": bins.labels[bin_index],
        "lower_edge": lower,
        "upper_edge": upper,
        "eligible": ever_eligible,
        "status": "insufficient_pre_activity",
        "drop_time": pd.NaT,
        "pre_events": np.nan,
        "pre_occupied_time_bins": np.nan,
        "post_events": np.nan,
        "pre_rate": np.nan,
        "pre_coverage_fraction": np.nan,
        "post_coverage_fraction": np.nan,
        "persistence_min_coverage_fraction": np.nan,
        "late_reactivation": saw_reactivation,
        "in_consensus": False,
        "confirmed_until": pd.NaT,
    }
    evidence = selected or fallback
    if evidence is not None:
        pre, post = evidence["pre"], evidence["post"]
        base.update({
            "pre_events": int(pre["events"]),
            "pre_occupied_time_bins": int(pre["occupied_time_bins"]),
            "post_events": int(post["events"]),
            "pre_rate": float(pre["events"]) / float(pre["coverage_hours"]),
            "pre_coverage_fraction": float(pre["coverage_fraction"]),
            "post_coverage_fraction": float(post["coverage_fraction"]),
        })
    if selected is not None:
        base.update({
            "status": "stable_drop",
            "drop_time": selected["time"],
            "persistence_min_coverage_fraction": selected[
                "persistence_min_coverage_fraction"
            ],
            "confirmed_until": selected["confirmed_until"],
        })
    elif saw_incomplete:
        base["status"] = "insufficient_followup"
    elif saw_reactivation:
        base["status"] = "reactivated"
    elif ever_eligible:
        base["status"] = "no_stable_drop"
    return base


def _maximal_date_groups(rows: pd.DataFrame, tolerance: pd.Timedelta) -> list[list[int]]:
    ordered = rows.sort_values("drop_time")
    indices = ordered.index.to_list()
    times = pd.DatetimeIndex(ordered.drop_time)
    groups: set[tuple[int, ...]] = set()
    for left in range(len(times)):
        right = left
        while right + 1 < len(times) and times[right + 1] - times[left] <= tolerance:
            right += 1
        groups.add(tuple(indices[left:right + 1]))
    maximal = [group for group in groups if not any(
        set(group) < set(other) for other in groups
    )]
    return [list(group) for group in maximal]


def detect_ttff_cessation(time: Any, values: Any, config: TTFFCessationConfig, *,
                          missing_values: Iterable[float] = ()) -> TTFFCessationDetection:
    """Detect the last stable loss of raw event activity in configurable TTFF bins."""
    index, _, valid_all = _prepared(time, values, missing_values)
    counts, bins = temporal_event_counts(
        time, values, config, missing_values=missing_values
    )
    if not valid_all.any():
        columns = (
            "bin_index", "bin_label", "lower_edge", "upper_edge", "eligible",
            "status", "drop_time", "pre_events", "pre_occupied_time_bins",
            "post_events", "pre_rate", "pre_coverage_fraction",
            "post_coverage_fraction", "persistence_min_coverage_fraction",
            "late_reactivation", "in_consensus", "confirmed_until",
        )
        return TTFFCessationDetection(
            "unavailable", None, pd.DataFrame(columns=columns), counts, bins
        )
    time_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cadence_ns = int(round(config.binning.time_bin_hours * 3_600_000_000_000))
    nominal_ns = _nominal_interval_ns(time_ns[valid_all], cadence_ns)
    record_end = pd.Timestamp(int(time_ns.max() + nominal_ns), unit="ns", tz="UTC")
    rows = [
        _per_bin_last_stable_drop(counts, bins, bin_index, config, record_end)
        for bin_index in range(len(bins.labels))
    ]
    results = pd.DataFrame(rows)
    stable = results[results.status == "stable_drop"]
    eligible_count = int(results.eligible.sum())
    if stable.empty:
        if (results.status == "insufficient_followup").any():
            status = "insufficient_followup"
        elif eligible_count == 0:
            status = "insufficient_data"
        else:
            status = "none"
        return TTFFCessationDetection(status, None, results, counts, bins)

    tolerance = pd.Timedelta(hours=config.comparison.agreement_tolerance_hours)
    groups = _maximal_date_groups(stable, tolerance)

    def group_key(group: list[int]) -> tuple[int, int]:
        times = pd.DatetimeIndex(results.loc[group, "drop_time"]).asi8
        return len(group), int(np.median(times))

    groups.sort(key=group_key, reverse=True)
    winner = groups[0]
    minimum = config.comparison.min_agreeing_bins
    disjoint_qualifier = any(
        len(group) >= minimum and set(group).isdisjoint(winner)
        for group in groups[1:]
    )
    if len(winner) >= minimum and not disjoint_qualifier:
        status = "clear"
    elif len(stable) == 1:
        status = "weak"
    else:
        status = "ambiguous"
    results.loc[winner, "in_consensus"] = True
    winner_rows = results.loc[winner]
    winner_times = pd.DatetimeIndex(winner_rows.drop_time).asi8
    consensus_ns = int(np.median(winner_times))
    consensus_time = pd.Timestamp(consensus_ns, unit="ns", tz="UTC")
    selected = pd.Series({
        "time": consensus_time,
        "confirmed_until": winner_rows.confirmed_until.min(),
        "agreeing_bin_count": len(winner),
        "consensus_span_hours": float(
            (winner_rows.drop_time.max() - winner_rows.drop_time.min())
            / pd.Timedelta(hours=1)
        ),
        "agreeing_pre_events": int(winner_rows.pre_events.sum()),
        "agreeing_post_events": int(winner_rows.post_events.sum()),
        "min_persistence_coverage_fraction": float(
            winner_rows.persistence_min_coverage_fraction.min()
        ),
    })
    return TTFFCessationDetection(status, selected, results, counts, bins)


def _mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def _duration_ns(value: str) -> int:
    return int(pd.Timedelta(value).value)


def _strain_window_summary(time_ns: np.ndarray, raw: np.ndarray, smoothed: np.ndarray,
                           start_ns: int, end_ns: int, nominal_ns: int) -> dict[str, float]:
    left = int(np.searchsorted(time_ns, start_ns, side="left"))
    right = int(np.searchsorted(time_ns, end_ns, side="left"))
    raw_values = raw[left:right]
    raw_valid = np.isfinite(raw_values)
    finite_raw = raw_values[raw_valid]
    smooth_values = smoothed[left:right]
    smooth_values = smooth_values[np.isfinite(smooth_values)]
    duration_hours = (end_ns - start_ns) / 3_600_000_000_000
    observed_times = time_ns[left:right][raw_valid]
    coverage = _coverage_hours(observed_times, start_ns, end_ns, nominal_ns)
    return {
        "median": float(np.median(smooth_values)) if len(smooth_values) else np.nan,
        "mad": _mad(finite_raw),
        "q25": float(np.quantile(finite_raw, .25)) if len(finite_raw) else np.nan,
        "q75": float(np.quantile(finite_raw, .75)) if len(finite_raw) else np.nan,
        "n_valid": int(len(finite_raw)),
        "coverage_fraction": min(1.0, coverage / duration_hours),
    }


def _strain_thresholds(drop: float, relative: float, normalized: float,
                       config: StrainStepConfig, *, weak: bool) -> bool:
    prefix = "weak" if weak else "minimum"
    return (
        drop >= getattr(config, f"{prefix}_drop_absolute")
        and relative >= getattr(config, f"{prefix}_drop_relative")
        and normalized >= getattr(config, f"{prefix}_normalized_drop")
    )


def _strain_episode_peaks(rows: pd.DataFrame, window: pd.Timedelta) -> list[pd.Series]:
    if rows.empty:
        return []
    working = rows.sort_values("time").copy()
    working["_episode"] = working.time.diff().gt(window).cumsum()
    peaks: list[pd.Series] = []
    for _, episode in working.groupby("_episode", sort=False):
        maximum = episode.normalized_drop.max()
        maxima = episode[np.isclose(episode.normalized_drop, maximum)]
        peaks.append(maxima.iloc[len(maxima) // 2])
    peaks.sort(
        key=lambda row: (float(row.normalized_drop), float(row.drop_absolute)),
        reverse=True,
    )
    return peaks


def detect_strain_step(time: Any, values: Any, config: StrainStepConfig, *,
                       missing_values: Iterable[float] = ()) -> StrainStepDetection:
    """Detect a persistent downward step in time-smoothed raw strain."""
    index, raw, valid = _prepared(time, values, missing_values)
    raw = raw.copy()
    raw[~valid] = np.nan
    if not valid.any():
        return StrainStepDetection("unavailable", None, pd.DataFrame())

    series = pd.Series(raw, index=index)
    smoothed = series.rolling(
        pd.Timedelta(config.window), min_periods=1, center=True,
    ).median().to_numpy()
    time_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    valid_time_ns = time_ns[valid]
    nominal_ns = _nominal_interval_ns(valid_time_ns, _duration_ns(config.window))
    pre_ns = _duration_ns(config.pre_window)
    post_ns = _duration_ns(config.post_window)
    persistence_ns = _duration_ns(config.persistence_window)
    record_end_ns = int(time_ns[-1] + nominal_ns)

    rows: list[dict[str, Any]] = []
    for candidate_ns in np.unique(time_ns)[1:]:
        candidate_ns = int(candidate_ns)
        before = _strain_window_summary(
            time_ns, raw, smoothed, candidate_ns - pre_ns, candidate_ns, nominal_ns,
        )
        after = _strain_window_summary(
            time_ns, raw, smoothed, candidate_ns, candidate_ns + post_ns, nominal_ns,
        )
        persistence_end = candidate_ns + post_ns + persistence_ns
        followup = _strain_window_summary(
            time_ns, raw, smoothed, candidate_ns + post_ns,
            persistence_end, nominal_ns,
        )
        if not np.isfinite(before["median"]) or not np.isfinite(after["median"]):
            continue

        drop = before["median"] - after["median"]
        relative = drop / max(abs(before["median"]), config.relative_floor)
        variability = max(
            before["mad"] if np.isfinite(before["mad"]) else 0.0,
            after["mad"] if np.isfinite(after["mad"]) else 0.0,
            config.variability_floor,
        )
        normalized = drop / variability
        clear_initial = _strain_thresholds(
            drop, relative, normalized, config, weak=False,
        )
        weak_initial = _strain_thresholds(
            drop, relative, normalized, config, weak=True,
        )
        full_followup = record_end_ns >= persistence_end
        followup_drop = before["median"] - followup["median"]
        followup_relative = (
            followup_drop / max(abs(before["median"]), config.relative_floor)
            if np.isfinite(followup_drop) else np.nan
        )
        followup_variability = max(
            before["mad"] if np.isfinite(before["mad"]) else 0.0,
            followup["mad"] if np.isfinite(followup["mad"]) else 0.0,
            config.variability_floor,
        )
        followup_normalized = followup_drop / followup_variability
        level_persists = (
            np.isfinite(followup["median"])
            and followup["median"] <= after["median"] + config.persistence_tolerance
        )
        clear_persistence = (
            level_persists
            and _strain_thresholds(
                followup_drop, followup_relative, followup_normalized,
                config, weak=False,
            )
        )
        weak_persistence = (
            level_persists
            and _strain_thresholds(
                followup_drop, followup_relative, followup_normalized,
                config, weak=True,
            )
        )
        if clear_initial and full_followup and clear_persistence:
            level = "clear"
        elif weak_initial and full_followup and weak_persistence:
            level = "weak"
        elif weak_initial and not full_followup:
            level = "insufficient_followup"
        elif weak_initial:
            level = "transient"
        else:
            level = "none"
        rows.append({
            "time": pd.Timestamp(candidate_ns, unit="ns", tz="UTC"),
            "candidate_level": level,
            "drop_absolute": float(drop),
            "drop_relative": float(relative),
            "normalized_drop": float(normalized),
            "median_before": before["median"],
            "median_after": after["median"],
            "mad_before": before["mad"],
            "mad_after": after["mad"],
            "q25_before": before["q25"],
            "q75_before": before["q75"],
            "q25_after": after["q25"],
            "q75_after": after["q75"],
            "persistence_median": followup["median"],
            "persistence_rise": (
                followup["median"] - after["median"]
                if np.isfinite(followup["median"]) else np.nan
            ),
            "pre_n_valid": before["n_valid"],
            "post_n_valid": after["n_valid"],
            "followup_n_valid": followup["n_valid"],
            "pre_coverage_fraction": before["coverage_fraction"],
            "post_coverage_fraction": after["coverage_fraction"],
            "followup_coverage_fraction": followup["coverage_fraction"],
            "confirmed_until": pd.Timestamp(persistence_end, unit="ns", tz="UTC"),
        })

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return StrainStepDetection("none", None, candidates)
    window = pd.Timedelta(config.window)
    clear_peaks = _strain_episode_peaks(
        candidates[candidates.candidate_level == "clear"], window,
    )
    if clear_peaks:
        status = "ambiguous" if len(clear_peaks) > 1 else "clear"
        selected = clear_peaks[0]
    else:
        weak_peaks = _strain_episode_peaks(
            candidates[candidates.candidate_level == "weak"], window,
        )
        insufficient_peaks = _strain_episode_peaks(
            candidates[candidates.candidate_level == "insufficient_followup"], window,
        )
        if insufficient_peaks:
            status, selected = "insufficient_followup", insufficient_peaks[0]
        elif weak_peaks:
            status, selected = "weak", weak_peaks[0]
        else:
            status, selected = "none", None
    return StrainStepDetection(status, selected, candidates)


def _raw_frame(time: Any, ttff: Any, strain: Any | None,
               hull_temperature: Any | None) -> pd.DataFrame:
    raw_time = np.asarray(time).reshape(-1)

    def values(raw: Any | None, name: str) -> np.ndarray:
        if raw is None:
            return np.full(len(raw_time), np.nan)
        array = np.asarray(raw, dtype=float).reshape(-1)
        if len(array) != len(raw_time):
            raise ValueError(f"{name} length does not match time length")
        array = array.copy()
        array[~np.isfinite(array)] = np.nan
        array[array == -999] = np.nan
        return array

    frame = pd.DataFrame({
        "time": pd.to_datetime(raw_time, errors="coerce", utc=True),
        "ttff": values(ttff, "ttff"),
        "strain": values(strain, "strain"),
        "hull_temperature": values(hull_temperature, "hull_temperature"),
    })
    frame = frame[frame.time.notna()].sort_values("time", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("time contains no valid timestamps")
    return frame


def _context_diagnostics(frame: pd.DataFrame, config: DrogueDetectionConfig) -> pd.DataFrame:
    index = pd.DatetimeIndex(frame.time)
    strain = pd.Series(frame.strain.to_numpy(), index=index)
    strain_window = pd.Timedelta(config.strain.window)
    rolling_strain = strain.rolling(strain_window, min_periods=1, center=True)
    minimum = 12
    temperature = pd.Series(frame.hull_temperature.to_numpy(), index=index)
    background = temperature.rolling(
        pd.Timedelta(config.temperature_background_window), min_periods=minimum, center=True
    ).median()
    anomaly = temperature - background
    variability = anomaly.rolling(
        pd.Timedelta(config.temperature_variability_window), min_periods=minimum, center=True
    ).apply(_mad, raw=True)
    return pd.DataFrame({
        "rolling_median_strain": rolling_strain.median().to_numpy(),
        "rolling_q25_strain": rolling_strain.quantile(.25).to_numpy(),
        "rolling_q75_strain": rolling_strain.quantile(.75).to_numpy(),
        "rolling_mad_strain": rolling_strain.apply(_mad, raw=True).to_numpy(),
        "temperature_background": background.to_numpy(),
        "temperature_anomaly": anomaly.to_numpy(),
        "temperature_rolling_mad": variability.to_numpy(),
    })


def _time(selected: pd.Series | None) -> np.datetime64:
    return np.datetime64("NaT", "ns") if selected is None else pd.Timestamp(selected.time).to_datetime64()


def _value(selected: pd.Series | None, name: str) -> float:
    return np.nan if selected is None else float(selected[name])


def _intervals_corroborate(ttff: pd.Series, strain: pd.Series,
                           config: DrogueDetectionConfig) -> bool:
    ttff_time, strain_time = pd.Timestamp(ttff.time), pd.Timestamp(strain.time)
    ttff_until = pd.Timestamp(ttff.confirmed_until)
    strain_until = pd.Timestamp(strain.confirmed_until)
    if ttff_time <= strain_time:
        return ttff_until >= strain_time + pd.Timedelta(hours=config.ttff.binning.time_bin_hours)
    return strain_until >= ttff_time + pd.Timedelta(config.strain.window)


def _combine(ttff: TTFFCessationDetection, strain: StrainStepDetection,
             config: DrogueDetectionConfig) -> tuple[np.datetime64, str, str, str, float]:
    ttff_time, strain_time = _time(ttff.selected), _time(strain.selected)
    offset = np.nan
    if not np.isnat(ttff_time) and not np.isnat(strain_time):
        offset = float((strain_time - ttff_time) / np.timedelta64(1, "h"))
    if "ambiguous" in {ttff.status, strain.status}:
        return (np.datetime64("NaT", "ns"), "ambiguous_component_changes", "low",
                "ambiguous", offset)
    if strain.status == "clear":
        if ttff.status == "clear":
            if _intervals_corroborate(ttff.selected, strain.selected, config):
                return strain_time, "detected_strain_primary_corroborated", "high", "corroborating", offset
            return (np.datetime64("NaT", "ns"), "ambiguous_signal_conflict", "low",
                    "contradictory", offset)
        relation = "unavailable" if ttff.status == "unavailable" else ttff.status
        return strain_time, "detected_strain_primary", "medium", relation, offset
    if ttff.status == "clear":
        relation = "strain_unavailable" if strain.status == "unavailable" else "strain_unclear"
        return ttff_time, "detected_ttff_provisional", "medium", relation, offset
    if "insufficient_followup" in {ttff.status, strain.status}:
        return (np.datetime64("NaT", "ns"), "insufficient_followup", "uncertain",
                "insufficient_followup", offset)
    return (np.datetime64("NaT", "ns"), "no_clear_component_change", "uncertain",
            "none", offset)


def detect_drogue_loss(platform_code: str, time: Any, ttff: Any, *, strain: Any | None = None,
                       hull_temperature: Any | None = None,
                       config: DrogueDetectionConfig = DrogueDetectionConfig()) -> DrogueDetection:
    """Detect TTFF-bin cessation and an independent persistent strain step."""
    frame = _raw_frame(time, ttff, strain, hull_temperature)
    ttff_detection = detect_ttff_cessation(
        frame.time, frame.ttff, config.ttff, missing_values=(-999,)
    )
    strain_detection = detect_strain_step(
        frame.time, frame.strain, config.strain, missing_values=(-999,)
    )
    diagnostics = pd.concat(
        [frame.reset_index(drop=True), _context_diagnostics(frame, config)], axis=1
    )
    automatic, status, confidence, relation, offset = _combine(
        ttff_detection, strain_detection, config
    )
    ttff_selected, strain_selected = ttff_detection.selected, strain_detection.selected
    result = DrogueDetectionResult(
        platform_code=str(platform_code),
        auto_drogue_loss_time=automatic,
        auto_status=status,
        auto_confidence=confidence,
        ttff_strain_relation=relation,
        ttff_change_time=_time(ttff_selected),
        ttff_change_status=ttff_detection.status,
        ttff_eligible_bin_count=int(ttff_detection.bin_results.eligible.sum()),
        ttff_stable_drop_bin_count=int(
            (ttff_detection.bin_results.status == "stable_drop").sum()
        ),
        ttff_agreeing_bin_count=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_bin_count)
        ),
        ttff_consensus_span_hours=_value(ttff_selected, "consensus_span_hours"),
        ttff_agreeing_pre_events=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_pre_events)
        ),
        ttff_agreeing_post_events=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_post_events)
        ),
        ttff_min_persistence_coverage_fraction=_value(
            ttff_selected, "min_persistence_coverage_fraction"
        ),
        strain_change_time=_time(strain_selected),
        strain_change_strength=_value(strain_selected, "normalized_drop"),
        strain_change_status=strain_detection.status,
        strain_drop_absolute=_value(strain_selected, "drop_absolute"),
        strain_drop_relative=_value(strain_selected, "drop_relative"),
        strain_normalized_drop=_value(strain_selected, "normalized_drop"),
        strain_median_before=_value(strain_selected, "median_before"),
        strain_median_after=_value(strain_selected, "median_after"),
        strain_mad_before=_value(strain_selected, "mad_before"),
        strain_mad_after=_value(strain_selected, "mad_after"),
        strain_q25_before=_value(strain_selected, "q25_before"),
        strain_q75_before=_value(strain_selected, "q75_before"),
        strain_q25_after=_value(strain_selected, "q25_after"),
        strain_q75_after=_value(strain_selected, "q75_after"),
        strain_pre_coverage_fraction=_value(strain_selected, "pre_coverage_fraction"),
        strain_post_coverage_fraction=_value(strain_selected, "post_coverage_fraction"),
        strain_followup_coverage_fraction=_value(strain_selected, "followup_coverage_fraction"),
        ttff_strain_offset_hours=offset,
        temperature_context_status=(
            "available" if frame.hull_temperature.notna().any() else "unavailable"
        ),
    )
    return DrogueDetection(
        result, diagnostics, ttff_detection.bin_results, strain_detection.candidates,
        ttff_detection.temporal_counts, ttff_detection.bins,
    )


def drogue_cutoff(loss_time: np.datetime64, buffer_hours: float = 24) -> np.datetime64:
    if not np.isfinite(buffer_hours) or buffer_hours < 0:
        raise ValueError("Drogue buffer must be finite and nonnegative")
    if np.isnat(loss_time):
        return np.datetime64("NaT", "ns")
    offset = int(round(buffer_hours * 3_600_000_000_000))
    result = int(loss_time.astype("datetime64[ns]").astype(np.int64)) - offset
    if not np.iinfo(np.int64).min < result <= np.iinfo(np.int64).max:
        raise ValueError("Drogue cutoff is outside the datetime64[ns] range")
    return np.datetime64(result, "ns")


def analysis_cutoff_time(reviewed_drogue_loss_time: np.datetime64,
                         cutoff_margin_hours: float = 24) -> np.datetime64:
    """Keep the reviewed physical event separate from its analysis margin."""
    return drogue_cutoff(reviewed_drogue_loss_time, cutoff_margin_hours)


def drogue_valid(time: np.ndarray, decision: str, cutoff: np.datetime64) -> np.ndarray:
    valid_time = ~np.isnat(time)
    if decision == "retained":
        return valid_time
    if decision == "unknown":
        return np.zeros(time.shape, dtype=bool)
    if decision == "lost":
        if np.isnat(cutoff):
            raise ValueError("A lost drogue requires a valid cutoff")
        return valid_time & (time < cutoff)
    raise ValueError(f"Unsupported drogue decision: {decision}")
