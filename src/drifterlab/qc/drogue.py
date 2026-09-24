"""Independent TTFF/strain drogue detection and explicit decision combination."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .strain_two_regime import (
    StrainTwoRegimeConfig,
    StrainTwoRegimeDetection,
    robust_two_regime_strain_change,
)


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
class DrogueCombinationConfig:
    agreement_tolerance_hours: float = 48.0

    def __post_init__(self) -> None:
        value = self.agreement_tolerance_hours
        if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                or value < 0):
            raise ValueError(
                "combination.agreement_tolerance_hours must be finite and nonnegative"
            )


def _default_ttff() -> TTFFCessationConfig:
    return TTFFCessationConfig(
        ValueBinningConfig("log", 10.0, 1.5, None, 1000.0, 12.0),
        TTFFCessationComparisonConfig(),
    )


@dataclass(frozen=True)
class DrogueDetectionConfig:
    """Settings for independent detectors and their experiment-level combination."""

    ttff: TTFFCessationConfig = field(default_factory=_default_ttff)
    strain: StrainTwoRegimeConfig = field(default_factory=StrainTwoRegimeConfig)
    combination: DrogueCombinationConfig = field(default_factory=DrogueCombinationConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrogueDetectionConfig:
        if not isinstance(data, dict):
            raise ValueError("detection must be a mapping")
        allowed = {"ttff", "strain", "combination"}
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
            strain = StrainTwoRegimeConfig()
        else:
            if not isinstance(raw_strain, dict):
                raise ValueError("detection.strain must be a mapping")
            try:
                strain = StrainTwoRegimeConfig.from_dict(raw_strain)
            except ValueError as exc:
                raise ValueError(f"Invalid detection.strain: {exc}") from exc

        raw_combination = data.get("combination", {})
        if not isinstance(raw_combination, dict):
            raise ValueError("detection.combination must be a mapping")
        valid_combination = set(DrogueCombinationConfig.__dataclass_fields__)
        unknown_combination = set(raw_combination) - valid_combination
        if unknown_combination:
            raise ValueError(
                "Unknown detection.combination keys: "
                f"{sorted(unknown_combination)}"
            )
        combination = DrogueCombinationConfig(**raw_combination)

        return cls(
            ttff=ttff,
            strain=strain,
            combination=combination,
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


@dataclass(frozen=True)
class ComponentDetectionResult:
    """Common interface consumed by the experiment-level decision layer."""

    change_time: np.datetime64
    status: str


@dataclass(frozen=True)
class CombinedDrogueDecision:
    change_time: np.datetime64
    status: str
    source: str


@dataclass(frozen=True)
class ResolvedDrogueDecision:
    """Effective decision computed from automatic evidence and an optional review."""

    final_drogue_status: str
    final_drogue_loss_time: np.datetime64
    analysis_cutoff_margin_hours: float
    analysis_cutoff_time: np.datetime64
    decision_source: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DrogueDetectionResult:
    platform_code: str
    auto_drogue_loss_time: np.datetime64
    auto_status: str
    auto_source: str
    ttff_change_time: np.datetime64
    ttff_status: str
    ttff_detail_status: str
    ttff_eligible_bin_count: int
    ttff_stable_drop_bin_count: int
    ttff_agreeing_bin_count: int
    ttff_consensus_span_hours: float
    ttff_agreeing_pre_events: int
    ttff_agreeing_post_events: int
    ttff_min_persistence_coverage_fraction: float
    strain_change_time: np.datetime64
    strain_status: str
    strain_level_before: float
    strain_level_after: float
    strain_absolute_drop: float
    strain_relative_drop: float
    strain_fit_improvement: float
    strain_n_blocks_before: int
    strain_n_blocks_after: int
    strain_n_valid_blocks: int
    ttff_strain_offset_hours: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DrogueDetection:
    result: DrogueDetectionResult
    ttff_bin_results: pd.DataFrame
    strain_blocks: pd.DataFrame
    strain_objective: pd.DataFrame
    strain_null_objective: float
    strain_candidate_time: np.datetime64
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


def _raw_frame(time: Any, ttff: Any, strain: Any | None) -> pd.DataFrame:
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
    })
    frame = frame[frame.time.notna()].sort_values("time", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("time contains no valid timestamps")
    return frame


def _nat() -> np.datetime64:
    return np.datetime64("NaT", "ns")


def _selected_value(selected: pd.Series | None, name: str) -> float:
    return np.nan if selected is None else float(selected[name])


def _ttff_component(detection: TTFFCessationDetection) -> ComponentDetectionResult:
    status_map = {
        "clear": "clear",
        "weak": "weak",
        "ambiguous": "ambiguous",
        "none": "no_change",
        "insufficient_followup": "insufficient_data",
        "insufficient_data": "insufficient_data",
        "unavailable": "insufficient_data",
    }
    try:
        status = status_map[detection.status]
    except KeyError as exc:
        raise ValueError(f"Unsupported TTFF detector status: {detection.status}") from exc
    time = (
        pd.Timestamp(detection.selected.time).to_datetime64()
        if detection.selected is not None and status in {"clear", "weak"}
        else _nat()
    )
    return ComponentDetectionResult(time, status)


def _strain_component(detection: StrainTwoRegimeDetection) -> ComponentDetectionResult:
    result = detection.result
    return ComponentDetectionResult(result.change_time, result.status)


def combine_drogue_detections(
    ttff: ComponentDetectionResult,
    strain: ComponentDetectionResult,
    config: DrogueCombinationConfig = DrogueCombinationConfig(),
) -> CombinedDrogueDecision:
    """Combine independent component decisions without changing either date."""
    valid_statuses = {"clear", "weak", "ambiguous", "no_change", "insufficient_data"}
    for name, component in (("TTFF", ttff), ("strain", strain)):
        if component.status not in valid_statuses:
            raise ValueError(f"Unsupported {name} component status: {component.status}")
        if component.status == "clear" and np.isnat(component.change_time):
            raise ValueError(f"A clear {name} component requires a change time")

    if ttff.status == strain.status == "clear":
        difference = abs(float(
            (strain.change_time - ttff.change_time) / np.timedelta64(1, "h")
        ))
        if difference <= config.agreement_tolerance_hours:
            return CombinedDrogueDecision(
                min(ttff.change_time, strain.change_time),
                "clear_agreement",
                "ttff+strain",
            )
        return CombinedDrogueDecision(_nat(), "signal_conflict", "none")

    benign = {"no_change", "insufficient_data"}
    if ttff.status == "clear" and strain.status in benign:
        return CombinedDrogueDecision(
            ttff.change_time, "clear_ttff_only", "ttff",
        )
    if strain.status == "clear" and ttff.status in benign:
        return CombinedDrogueDecision(
            strain.change_time, "clear_strain_only", "strain",
        )
    return CombinedDrogueDecision(_nat(), "unresolved", "none")


def detect_drogue_loss(platform_code: str, time: Any, ttff: Any, *, strain: Any | None = None,
                       config: DrogueDetectionConfig = DrogueDetectionConfig()) -> DrogueDetection:
    """Run independent production detectors, then combine their decisions."""
    frame = _raw_frame(time, ttff, strain)
    ttff_detection = detect_ttff_cessation(
        frame.time, frame.ttff, config.ttff, missing_values=(-999,)
    )
    strain_detection = robust_two_regime_strain_change(
        frame.time, frame.strain, config.strain, missing_values=(-999,)
    )
    ttff_component = _ttff_component(ttff_detection)
    strain_component = _strain_component(strain_detection)
    automatic = combine_drogue_detections(
        ttff_component, strain_component, config.combination,
    )
    ttff_selected = ttff_detection.selected
    offset = np.nan
    if (not np.isnat(ttff_component.change_time)
            and not np.isnat(strain_component.change_time)):
        offset = float(
            (strain_component.change_time - ttff_component.change_time)
            / np.timedelta64(1, "h")
        )
    strain_result = strain_detection.result
    result = DrogueDetectionResult(
        platform_code=str(platform_code),
        auto_drogue_loss_time=automatic.change_time,
        auto_status=automatic.status,
        auto_source=automatic.source,
        ttff_change_time=ttff_component.change_time,
        ttff_status=ttff_component.status,
        ttff_detail_status=ttff_detection.status,
        ttff_eligible_bin_count=int(ttff_detection.bin_results.eligible.sum()),
        ttff_stable_drop_bin_count=int(
            (ttff_detection.bin_results.status == "stable_drop").sum()
        ),
        ttff_agreeing_bin_count=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_bin_count)
        ),
        ttff_consensus_span_hours=_selected_value(
            ttff_selected, "consensus_span_hours"
        ),
        ttff_agreeing_pre_events=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_pre_events)
        ),
        ttff_agreeing_post_events=(
            0 if ttff_selected is None else int(ttff_selected.agreeing_post_events)
        ),
        ttff_min_persistence_coverage_fraction=_selected_value(
            ttff_selected, "min_persistence_coverage_fraction"
        ),
        strain_change_time=strain_component.change_time,
        strain_status=strain_component.status,
        strain_level_before=strain_result.level_before,
        strain_level_after=strain_result.level_after,
        strain_absolute_drop=strain_result.absolute_drop,
        strain_relative_drop=strain_result.relative_drop,
        strain_fit_improvement=strain_result.fit_improvement,
        strain_n_blocks_before=strain_result.n_blocks_before,
        strain_n_blocks_after=strain_result.n_blocks_after,
        strain_n_valid_blocks=strain_result.n_valid_blocks,
        ttff_strain_offset_hours=offset,
    )
    return DrogueDetection(
        result, ttff_detection.bin_results, strain_detection.blocks,
        strain_detection.candidates, strain_result.j0, strain_result.candidate_time,
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


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(name, default)
    if hasattr(row, "get"):
        return row.get(name, default)
    return getattr(row, name, default)


def _optional_datetime64(value: Any) -> np.datetime64:
    if value is None or str(value).strip() in {"", "NaT", "nan", "None"}:
        return _nat()
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return _nat()
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_datetime64().astype("datetime64[ns]")


def resolve_drogue_decision(
    automatic_row: Any,
    review_row: Any | None = None,
    *,
    default_margin_hours: float = 24,
) -> ResolvedDrogueDecision:
    """Resolve an effective drogue state without creating another product table."""
    margin = float(default_margin_hours)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("default_margin_hours must be finite and nonnegative")

    if review_row is not None:
        status = str(_row_value(review_row, "review_status", "")).strip()
        reviewed_margin = _row_value(
            review_row, "analysis_cutoff_margin_hours", margin,
        )
        margin = float(reviewed_margin)
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Reviewed analysis cutoff margin must be finite and nonnegative")
        if status in {"accepted_auto", "manual_date"}:
            loss_time = _optional_datetime64(
                _row_value(review_row, "reviewed_drogue_loss_time")
            )
            if np.isnat(loss_time):
                raise ValueError(f"{status} review requires a physical loss time")
            return ResolvedDrogueDecision(
                "lost", loss_time, margin,
                analysis_cutoff_time(loss_time, margin), status,
            )
        if status in {"not_lost", "uncertain"}:
            return ResolvedDrogueDecision(
                status, _nat(), margin, _nat(), status,
            )
        raise ValueError(f"Unsupported drogue review status: {status!r}")

    automatic_time = _optional_datetime64(
        _row_value(automatic_row, "auto_drogue_loss_time")
    )
    if not np.isnat(automatic_time):
        return ResolvedDrogueDecision(
            "lost", automatic_time, margin,
            analysis_cutoff_time(automatic_time, margin), "automatic",
        )
    return ResolvedDrogueDecision(
        "uncertain", _nat(), margin, _nat(), "automatic",
    )


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
