"""TTFF-distribution and strain-step drogue detection plus analysis masks."""

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
class DistributionComparisonConfig:
    """Coverage, direction, persistence, and ambiguity settings."""

    pre_window_hours: float
    post_window_hours: float
    persistence_hours: float
    min_valid_samples: int
    min_coverage_fraction: float
    min_down_distance: float
    min_downward_fraction: float
    max_reactivation_fraction: float
    peak_separation_hours: float
    comparable_strength_fraction: float

    def __post_init__(self) -> None:
        positive = (
            "pre_window_hours", "post_window_hours", "persistence_hours",
            "min_down_distance", "peak_separation_hours",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"comparison.{name} must be finite and positive")
        fractions = (
            "min_coverage_fraction", "min_downward_fraction",
            "max_reactivation_fraction", "comparable_strength_fraction",
        )
        for name in fractions:
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"comparison.{name} must be in [0, 1]")
        if self.min_coverage_fraction == 0:
            raise ValueError("comparison.min_coverage_fraction must exceed zero")
        if self.min_downward_fraction <= .5:
            raise ValueError("comparison.min_downward_fraction must exceed 0.5")
        if self.comparable_strength_fraction == 0:
            raise ValueError("comparison.comparable_strength_fraction must exceed zero")
        if (isinstance(self.min_valid_samples, bool)
                or not isinstance(self.min_valid_samples, (int, np.integer))
                or self.min_valid_samples < 2):
            raise ValueError("comparison.min_valid_samples must be an integer of at least two")


@dataclass(frozen=True)
class TailSupportConfig:
    lower_edge: float
    upper_edge: float

    def __post_init__(self) -> None:
        if (not np.isfinite(self.lower_edge) or not np.isfinite(self.upper_edge)
                or self.lower_edge < 0 or self.upper_edge <= self.lower_edge):
            raise ValueError("tail edges must be finite, nonnegative, and increasing")


@dataclass(frozen=True)
class SignalDistributionConfig:
    binning: ValueBinningConfig
    comparison: DistributionComparisonConfig
    tail: TailSupportConfig | None = None

    def __post_init__(self) -> None:
        if self.tail is not None and self.tail.upper_edge > self.binning.last_edge:
            raise ValueError("tail.upper_edge cannot exceed binning.last_edge")


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


def _default_ttff() -> SignalDistributionConfig:
    return SignalDistributionConfig(
        ValueBinningConfig("log", 10.0, 1.5, None, 1000.0, 12.0),
        DistributionComparisonConfig(
            48.0, 48.0, 72.0, 24, .75, .35, .75, .20, 48.0, .80,
        ),
        TailSupportConfig(50.0, 1000.0),
    )


@dataclass(frozen=True)
class DrogueDetectionConfig:
    """Settings for TTFF distributions, strain steps, and contextual temperature."""

    ttff: SignalDistributionConfig = field(default_factory=_default_ttff)
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

        def signal(name: str, default: SignalDistributionConfig) -> SignalDistributionConfig:
            raw = data.get(name)
            if raw is None:
                return default
            if not isinstance(raw, dict):
                raise ValueError(f"detection.{name} must be a mapping")
            allowed_signal = {"binning", "comparison", "tail"}
            unknown_signal = set(raw) - allowed_signal
            if unknown_signal:
                raise ValueError(
                    f"Unknown detection.{name} keys: {sorted(unknown_signal)}"
                )

            def nested(section: str, default_object: Any, object_type: type) -> Any:
                values = raw.get(section)
                if values is None:
                    return default_object
                if not isinstance(values, dict):
                    raise ValueError(f"detection.{name}.{section} must be a mapping")
                valid = set(default_object.__dataclass_fields__)
                unknown_nested = set(values) - valid
                if unknown_nested:
                    raise ValueError(
                        f"Unknown detection.{name}.{section} keys: {sorted(unknown_nested)}"
                    )
                merged = {**asdict(default_object), **values}
                return object_type(**merged)

            binning = nested("binning", default.binning, ValueBinningConfig)
            comparison = nested(
                "comparison", default.comparison, DistributionComparisonConfig
            )
            if "tail" in raw:
                if raw["tail"] is None:
                    tail = None
                elif default.tail is None:
                    values = raw["tail"]
                    if not isinstance(values, dict):
                        raise ValueError(f"detection.{name}.tail must be a mapping")
                    unknown_tail = set(values) - {"lower_edge", "upper_edge"}
                    if unknown_tail:
                        raise ValueError(
                            f"Unknown detection.{name}.tail keys: {sorted(unknown_tail)}"
                        )
                    tail = TailSupportConfig(**values)
                else:
                    tail = nested("tail", default.tail, TailSupportConfig)
            else:
                tail = default.tail
            return SignalDistributionConfig(binning, comparison, tail)

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
            ttff=signal("ttff", _default_ttff()),
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
    coordinate_edges: np.ndarray
    histogram_edges: np.ndarray
    labels: tuple[str, ...]
    scale: str
    include_lower_than_first_bin: bool


@dataclass(frozen=True)
class DistributionShift:
    d_down: float
    d_up: float
    w1: float
    downward_fraction: float


@dataclass
class SignalDistributionDetection:
    status: str
    selected: pd.Series | None
    candidates: pd.DataFrame
    temporal_histograms: pd.DataFrame
    bins: ValueBins
    detail: dict[str, np.ndarray]


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
    ttff_change_strength: float
    ttff_change_status: str
    ttff_d_down: float
    ttff_d_up: float
    ttff_w1: float
    ttff_downward_fraction: float
    ttff_tail_area_before: float
    ttff_tail_area_after: float
    ttff_tail_area_drop: float
    ttff_tail_area_relative_drop: float
    ttff_pre_coverage_fraction: float
    ttff_post_coverage_fraction: float
    ttff_followup_coverage_fraction: float
    ttff_underflow_fraction_before: float
    ttff_overflow_fraction_before: float
    ttff_underflow_fraction_after: float
    ttff_overflow_fraction_after: float
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
    ttff_candidates: pd.DataFrame
    strain_candidates: pd.DataFrame
    ttff_histograms: pd.DataFrame
    ttff_bins: ValueBins
    ttff_detail: dict[str, np.ndarray]


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
    coordinate = np.log1p(physical) if config.scale == "log" else physical.copy()
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
        physical, coordinate, histogram, tuple(labels), config.scale,
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


def _positive_linear_integral(left: float, right: float, width: float) -> float:
    if left >= 0 and right >= 0:
        return .5 * (left + right) * width
    if left <= 0 and right <= 0:
        return 0.0
    if np.isclose(left, right):
        return max(left, 0.0) * width
    crossing = -left / (right - left)
    if left > 0:
        return .5 * left * width * crossing
    return .5 * right * width * (1 - crossing)


def directional_wasserstein(before_counts: Any, after_counts: Any,
                            bins: ValueBins) -> DistributionShift:
    """Integrate signed CDF separation with uniform mass inside finite bins.

    Underflow and overflow mass are clipped to the first and last finite edges.
    Thus open bins affect occupancy and direction but receive no arbitrary
    midpoint or unbounded leverage.
    """
    before = np.asarray(before_counts, dtype=float)
    after = np.asarray(after_counts, dtype=float)
    offset = int(bins.include_lower_than_first_bin)
    expected = len(bins.coordinate_edges) + offset
    if before.shape != (expected,) or after.shape != (expected,):
        raise ValueError(f"Histogram counts must each have length {expected}")
    if before.sum() <= 0 or after.sum() <= 0:
        return DistributionShift(np.nan, np.nan, np.nan, np.nan)
    before /= before.sum()
    after /= after.sum()
    delta_left = after[0] - before[0] if offset else 0.0
    down = up = 0.0
    for index, width in enumerate(np.diff(bins.coordinate_edges)):
        count_index = index + offset
        delta_right = delta_left + after[count_index] - before[count_index]
        down += _positive_linear_integral(delta_left, delta_right, float(width))
        up += _positive_linear_integral(-delta_left, -delta_right, float(width))
        delta_left = delta_right
    total = down + up
    fraction = down / total if total > np.finfo(float).eps else np.nan
    return DistributionShift(float(down), float(up), float(total), float(fraction))


def survival_curve(counts: Any, bins: ValueBins,
                   physical_points: Any) -> np.ndarray:
    """Histogram survival curve using uniform mass within finite bins."""
    counts = np.asarray(counts, dtype=float)
    if counts.sum() <= 0:
        return np.full(np.asarray(physical_points).shape, np.nan, dtype=float)
    probabilities = counts / counts.sum()
    points = np.asarray(physical_points, dtype=float)
    coordinates = np.log1p(np.clip(points, 0, None)) if bins.scale == "log" else points
    output = np.empty(points.shape, dtype=float)
    edges = bins.coordinate_edges
    offset = int(bins.include_lower_than_first_bin)
    for position, coordinate in np.ndenumerate(coordinates):
        if coordinate < edges[0]:
            cdf = 0.0
        elif coordinate >= edges[-1]:
            cdf = 1.0
        else:
            interval = int(np.searchsorted(edges, coordinate, side="right") - 1)
            cdf = probabilities[0] if offset else 0.0
            cdf += probabilities[offset:offset + interval].sum()
            proportion = (coordinate - edges[interval]) / (edges[interval + 1] - edges[interval])
            cdf += probabilities[offset + interval] * proportion
        output[position] = max(0.0, min(1.0, 1.0 - cdf))
    return output


def bounded_tail_area(counts: Any, bins: ValueBins, tail: TailSupportConfig) -> float:
    """Integrate survival in log1p space over configured physical bounds."""
    if bins.scale != "log":
        raise ValueError("Bounded tail area requires log-scale bins")
    internal = bins.physical_edges[
        (bins.physical_edges > tail.lower_edge) & (bins.physical_edges < tail.upper_edge)
    ]
    points = np.r_[tail.lower_edge, internal, tail.upper_edge]
    curve = survival_curve(counts, bins, points)
    if np.isnan(curve).all():
        return np.nan
    coordinate = np.log1p(points)
    return float(np.sum(.5 * (curve[:-1] + curve[1:]) * np.diff(coordinate)))


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


def temporal_histograms(time: Any, values: Any, config: ValueBinningConfig, *,
                        missing_values: Iterable[float] = ()) -> tuple[pd.DataFrame, ValueBins]:
    """Normalized time-bin histograms with counts, samples, and coverage."""
    index, array, valid_value = _prepared(time, values, missing_values)
    if not len(index):
        raise ValueError("time contains no valid timestamps")
    bins = make_value_bins(config)
    cadence_source = valid_value.copy()
    if not bins.include_lower_than_first_bin:
        valid_value &= array >= bins.physical_edges[0]
    cadence_ns = int(round(config.time_bin_hours * 3_600_000_000_000))
    time_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    start_ns = int(time_ns.min() // cadence_ns * cadence_ns)
    stop_ns = int(time_ns.max() // cadence_ns * cadence_ns + cadence_ns)
    starts = np.arange(start_ns, stop_ns, cadence_ns, dtype=np.int64)
    nominal_ns = _nominal_interval_ns(time_ns[cadence_source], cadence_ns)
    rows = []
    for left in starts:
        right = int(left + cadence_ns)
        selected = (time_ns >= left) & (time_ns < right) & valid_value
        counts, count = histogram_counts(array[selected], bins)
        fractions = counts / count if count else np.full(len(counts), np.nan)
        coverage = _coverage_hours(time_ns[selected], int(left), right, nominal_ns)
        rows.append({
            "time_start": pd.Timestamp(left, unit="ns", tz="UTC"),
            "time_end": pd.Timestamp(right, unit="ns", tz="UTC"),
            "time_center": pd.Timestamp(left + cadence_ns // 2, unit="ns", tz="UTC"),
            "bin_counts": counts,
            "fractions": fractions,
            "n_valid": count,
            "coverage_hours": coverage,
            "coverage_fraction": min(1.0, coverage / config.time_bin_hours),
            "underflow_fraction": (
                fractions[0] if count and bins.include_lower_than_first_bin else np.nan
            ),
            "overflow_fraction": fractions[-1] if count else np.nan,
        })
    return pd.DataFrame(rows), bins


def _window(time_ns: np.ndarray, values: np.ndarray, valid_value: np.ndarray,
            start_ns: int, end_ns: int, bins: ValueBins,
            nominal_ns: int) -> dict[str, Any]:
    selected = (time_ns >= start_ns) & (time_ns < end_ns) & valid_value
    subset = values[selected]
    counts, count = histogram_counts(subset, bins)
    duration_hours = (end_ns - start_ns) / 3_600_000_000_000
    coverage = _coverage_hours(time_ns[selected], start_ns, end_ns, nominal_ns)
    return {
        "counts": counts,
        "n_valid": count,
        "coverage_hours": coverage,
        "coverage_fraction": min(1.0, coverage / duration_hours),
        "median": float(np.median(subset)) if count else np.nan,
        "q25": float(np.quantile(subset, .25)) if count else np.nan,
        "q75": float(np.quantile(subset, .75)) if count else np.nan,
        "underflow_fraction": (
            counts[0] / count
            if count and bins.include_lower_than_first_bin else np.nan
        ),
        "overflow_fraction": counts[-1] / count if count else np.nan,
    }


def _adequate(window: dict[str, Any], config: DistributionComparisonConfig) -> bool:
    return (window["n_valid"] >= config.min_valid_samples
            and window["coverage_fraction"] >= config.min_coverage_fraction)


def _peak_rows(rows: pd.DataFrame, config: SignalDistributionConfig) -> list[pd.Series]:
    if rows.empty:
        return []
    cadence = pd.Timedelta(hours=config.binning.time_bin_hours)
    working = rows.sort_values("time").copy()
    working["_episode"] = working.time.diff().gt(cadence * 1.5).cumsum()
    peaks: list[pd.Series] = []
    for _, episode in working.groupby("_episode", sort=False):
        maximum = episode.d_down.max()
        maxima = episode[np.isclose(episode.d_down, maximum)]
        peaks.append(maxima.iloc[len(maxima) // 2])
    peaks.sort(key=lambda row: float(row.d_down), reverse=True)
    separated: list[pd.Series] = []
    separation = pd.Timedelta(hours=config.comparison.peak_separation_hours)
    for peak in peaks:
        if all(abs(peak.time - retained.time) >= separation for retained in separated):
            separated.append(peak)
    return separated


def _candidate_detail(selected: pd.Series | None, bins: ValueBins,
                      tail: TailSupportConfig | None) -> dict[str, np.ndarray]:
    if selected is None:
        return {}
    detail = {
        "physical_edges": bins.physical_edges.copy(),
        "before_counts": np.asarray(selected.before_counts).copy(),
        "after_counts": np.asarray(selected.after_counts).copy(),
        "followup_counts": np.asarray(selected.followup_counts).copy(),
    }
    if tail is not None:
        internal = bins.physical_edges[
            (bins.physical_edges > tail.lower_edge) & (bins.physical_edges < tail.upper_edge)
        ]
        points = np.r_[tail.lower_edge, internal, tail.upper_edge]
        detail.update({
            "survival_x": points,
            "survival_before": survival_curve(selected.before_counts, bins, points),
            "survival_after": survival_curve(selected.after_counts, bins, points),
        })
    return detail


def detect_distribution_change(time: Any, values: Any, config: SignalDistributionConfig, *,
                               missing_values: Iterable[float] = ()) -> SignalDistributionDetection:
    """Apply the shared pre/post histogram-CDF detector to one signal."""
    index, array, valid_value = _prepared(time, values, missing_values)
    histograms, bins = temporal_histograms(
        index, array, config.binning, missing_values=missing_values
    )
    cadence_source = valid_value.copy()
    if not bins.include_lower_than_first_bin:
        valid_value &= array >= bins.physical_edges[0]
    if not valid_value.any():
        return SignalDistributionDetection(
            "unavailable", None, pd.DataFrame(), histograms, bins, {}
        )
    comparison = config.comparison
    time_ns = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    cadence_ns = int(round(config.binning.time_bin_hours * 3_600_000_000_000))
    nominal_ns = _nominal_interval_ns(time_ns[cadence_source], cadence_ns)
    pre_ns = int(round(comparison.pre_window_hours * 3_600_000_000_000))
    post_ns = int(round(comparison.post_window_hours * 3_600_000_000_000))
    persistence_ns = int(round(comparison.persistence_hours * 3_600_000_000_000))
    rows = []
    for candidate in histograms.time_start.iloc[1:]:
        candidate_ns = int(candidate.value)
        before = _window(
            time_ns, array, valid_value, candidate_ns - pre_ns, candidate_ns,
            bins, nominal_ns,
        )
        after = _window(
            time_ns, array, valid_value, candidate_ns, candidate_ns + post_ns,
            bins, nominal_ns,
        )
        followup = _window(
            time_ns, array, valid_value, candidate_ns + post_ns,
            candidate_ns + post_ns + persistence_ns, bins, nominal_ns,
        )
        terminal_start = candidate_ns + post_ns + persistence_ns - cadence_ns
        terminal = _window(
            time_ns, array, valid_value, terminal_start,
            candidate_ns + post_ns + persistence_ns, bins, nominal_ns,
        )
        shift = directional_wasserstein(before["counts"], after["counts"], bins)
        follow_shift = directional_wasserstein(before["counts"], followup["counts"], bins)
        reactivation = directional_wasserstein(after["counts"], followup["counts"], bins)
        terminal_shift = directional_wasserstein(before["counts"], terminal["counts"], bins)
        terminal_reactivation = directional_wasserstein(
            after["counts"], terminal["counts"], bins
        )
        pre_post_adequate = _adequate(before, comparison) and _adequate(after, comparison)
        substantial = (
            pre_post_adequate and shift.d_down >= comparison.min_down_distance
            and shift.downward_fraction >= comparison.min_downward_fraction
        )
        terminal_minimum = max(
            2, int(np.ceil(
                comparison.min_valid_samples
                * config.binning.time_bin_hours / comparison.post_window_hours
            )),
        )
        terminal_adequate = (
            terminal["n_valid"] >= terminal_minimum
            and terminal["coverage_fraction"] >= comparison.min_coverage_fraction
        )
        followup_adequate = _adequate(followup, comparison) and terminal_adequate
        reactivation_up = max(reactivation.d_up, terminal_reactivation.d_up)
        persistent = (
            substantial and followup_adequate
            and follow_shift.d_down >= comparison.min_down_distance
            and follow_shift.downward_fraction >= comparison.min_downward_fraction
            and terminal_shift.d_down >= comparison.min_down_distance
            and terminal_shift.downward_fraction >= comparison.min_downward_fraction
            and reactivation_up
            <= comparison.max_reactivation_fraction * max(shift.d_down, np.finfo(float).eps)
        )
        if persistent:
            level = "clear"
        elif substantial and not followup_adequate:
            level = "insufficient_followup"
        elif substantial:
            level = "transient"
        elif (pre_post_adequate and shift.d_down > 0
              and shift.downward_fraction >= comparison.min_downward_fraction):
            level = "weak"
        else:
            level = "none"
        tail_before = tail_after = np.nan
        if config.tail is not None and before["n_valid"] and after["n_valid"]:
            tail_before = bounded_tail_area(before["counts"], bins, config.tail)
            tail_after = bounded_tail_area(after["counts"], bins, config.tail)
        tail_drop = tail_before - tail_after
        rows.append({
            "time": candidate,
            "candidate_level": level,
            "d_down": shift.d_down,
            "d_up": shift.d_up,
            "w1": shift.w1,
            "downward_fraction": shift.downward_fraction,
            "followup_d_down": follow_shift.d_down,
            "followup_downward_fraction": follow_shift.downward_fraction,
            "reactivation_d_up": reactivation_up,
            "tail_area_before": tail_before,
            "tail_area_after": tail_after,
            "tail_area_drop": tail_drop,
            "tail_area_relative_drop": (
                tail_drop / tail_before if np.isfinite(tail_before) and tail_before > 0 else np.nan
            ),
            "pre_n_valid": before["n_valid"],
            "post_n_valid": after["n_valid"],
            "followup_n_valid": followup["n_valid"],
            "pre_coverage_fraction": before["coverage_fraction"],
            "post_coverage_fraction": after["coverage_fraction"],
            "followup_coverage_fraction": followup["coverage_fraction"],
            "underflow_fraction_before": before["underflow_fraction"],
            "overflow_fraction_before": before["overflow_fraction"],
            "underflow_fraction_after": after["underflow_fraction"],
            "overflow_fraction_after": after["overflow_fraction"],
            "median_before": before["median"],
            "median_after": after["median"],
            "q25_before": before["q25"],
            "q75_before": before["q75"],
            "q25_after": after["q25"],
            "q75_after": after["q75"],
            "confirmed_until": candidate + pd.Timedelta(
                hours=comparison.post_window_hours + comparison.persistence_hours
            ),
            "before_counts": before["counts"],
            "after_counts": after["counts"],
            "followup_counts": followup["counts"],
        })
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return SignalDistributionDetection(
            "none", None, candidates, histograms, bins, {}
        )
    substantial_rows = candidates[candidates.candidate_level.isin(
        ["clear", "transient", "insufficient_followup"]
    )]
    episode_peaks = _peak_rows(substantial_rows, config)
    clear_peaks = [peak for peak in episode_peaks if peak.candidate_level == "clear"]
    if clear_peaks:
        selected = clear_peaks[0]
        if (len(clear_peaks) > 1 and clear_peaks[1].d_down
                >= clear_peaks[0].d_down * comparison.comparable_strength_fraction):
            status = "ambiguous"
        else:
            status = "clear"
    else:
        insufficient_peaks = [
            peak for peak in episode_peaks
            if peak.candidate_level == "insufficient_followup"
        ]
        weak = candidates[candidates.candidate_level == "weak"]
        if insufficient_peaks:
            selected = insufficient_peaks[0]
            status = "insufficient_followup"
        elif not weak.empty:
            selected = weak.loc[weak.d_down.idxmax()]
            status = "weak"
        else:
            selected = None
            status = "none"
    return SignalDistributionDetection(
        status, selected, candidates, histograms, bins,
        _candidate_detail(selected, bins, config.tail),
    )


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


def _combine(ttff: SignalDistributionDetection, strain: StrainStepDetection,
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
    """Detect TTFF redistribution and an independent persistent strain step."""
    frame = _raw_frame(time, ttff, strain, hull_temperature)
    ttff_detection = detect_distribution_change(
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
        ttff_change_strength=_value(ttff_selected, "d_down"),
        ttff_change_status=ttff_detection.status,
        ttff_d_down=_value(ttff_selected, "d_down"),
        ttff_d_up=_value(ttff_selected, "d_up"),
        ttff_w1=_value(ttff_selected, "w1"),
        ttff_downward_fraction=_value(ttff_selected, "downward_fraction"),
        ttff_tail_area_before=_value(ttff_selected, "tail_area_before"),
        ttff_tail_area_after=_value(ttff_selected, "tail_area_after"),
        ttff_tail_area_drop=_value(ttff_selected, "tail_area_drop"),
        ttff_tail_area_relative_drop=_value(ttff_selected, "tail_area_relative_drop"),
        ttff_pre_coverage_fraction=_value(ttff_selected, "pre_coverage_fraction"),
        ttff_post_coverage_fraction=_value(ttff_selected, "post_coverage_fraction"),
        ttff_followup_coverage_fraction=_value(ttff_selected, "followup_coverage_fraction"),
        ttff_underflow_fraction_before=_value(ttff_selected, "underflow_fraction_before"),
        ttff_overflow_fraction_before=_value(ttff_selected, "overflow_fraction_before"),
        ttff_underflow_fraction_after=_value(ttff_selected, "underflow_fraction_after"),
        ttff_overflow_fraction_after=_value(ttff_selected, "overflow_fraction_after"),
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
        result, diagnostics, ttff_detection.candidates, strain_detection.candidates,
        ttff_detection.temporal_histograms, ttff_detection.bins,
        ttff_detection.detail,
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
