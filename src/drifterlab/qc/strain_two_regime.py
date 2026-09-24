"""Production robust one-change-point model for raw drogue strain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrainTwoRegimeAmbiguityConfig:
    """Settings for recognizing a second, comparably good split."""

    comparable_cost_fraction: float = .02
    minimum_separation_hours: float = 48.0

    def __post_init__(self) -> None:
        if (isinstance(self.comparable_cost_fraction, (bool, np.bool_))
                or not np.isfinite(self.comparable_cost_fraction)
                or self.comparable_cost_fraction < 0):
            raise ValueError(
                "ambiguity.comparable_cost_fraction must be finite and nonnegative"
            )
        if (isinstance(self.minimum_separation_hours, (bool, np.bool_))
                or not np.isfinite(self.minimum_separation_hours)
                or self.minimum_separation_hours <= 0):
            raise ValueError(
                "ambiguity.minimum_separation_hours must be finite and positive"
            )


@dataclass(frozen=True)
class StrainTwoRegimeConfig:
    """Aggregation, acceptance, and ambiguity settings."""

    aggregation_hours: float = 6.0
    min_samples_per_block: int = 3
    minimum_side_duration_hours: float = 72.0
    minimum_drop_absolute: float = 2.0
    minimum_drop_relative: float = .08
    minimum_fit_improvement: float = .10
    relative_floor: float = 1.0
    ambiguity: StrainTwoRegimeAmbiguityConfig = field(
        default_factory=StrainTwoRegimeAmbiguityConfig
    )

    def __post_init__(self) -> None:
        positive = (
            "aggregation_hours", "minimum_side_duration_hours", "relative_floor",
        )
        for name in positive:
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (isinstance(self.min_samples_per_block, (bool, np.bool_))
                or not isinstance(self.min_samples_per_block, (int, np.integer))
                or self.min_samples_per_block < 1):
            raise ValueError("min_samples_per_block must be a positive integer")
        for name in (
            "minimum_drop_absolute", "minimum_drop_relative",
            "minimum_fit_improvement",
        ):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value)
                    or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.minimum_fit_improvement > 1:
            raise ValueError("minimum_fit_improvement cannot exceed one")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrainTwoRegimeConfig:
        if not isinstance(data, dict):
            raise ValueError("strain_two_regime must be a mapping")
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown strain_two_regime keys: {sorted(unknown)}")
        values = dict(data)
        ambiguity = values.get("ambiguity")
        if ambiguity is not None:
            if not isinstance(ambiguity, dict):
                raise ValueError("strain_two_regime.ambiguity must be a mapping")
            valid_ambiguity = set(StrainTwoRegimeAmbiguityConfig.__dataclass_fields__)
            unknown_ambiguity = set(ambiguity) - valid_ambiguity
            if unknown_ambiguity:
                raise ValueError(
                    "Unknown strain_two_regime.ambiguity keys: "
                    f"{sorted(unknown_ambiguity)}"
                )
            values["ambiguity"] = StrainTwoRegimeAmbiguityConfig(**ambiguity)
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrainTwoRegimeResult:
    status: str
    change_time: np.datetime64
    candidate_time: np.datetime64
    level_before: float
    level_after: float
    absolute_drop: float
    relative_drop: float
    j0: float
    j1_best: float
    fit_improvement: float
    n_blocks_before: int
    n_blocks_after: int
    n_valid_blocks: int
    aggregation_hours: float
    n_comparable_alternatives: int
    comparable_alternative_time: np.datetime64

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrainTwoRegimeDetection:
    result: StrainTwoRegimeResult
    blocks: pd.DataFrame
    candidates: pd.DataFrame


def _nat() -> np.datetime64:
    return np.datetime64("NaT", "ns")


def _empty_result(status: str, config: StrainTwoRegimeConfig, *,
                  n_valid_blocks: int = 0, j0: float = np.nan) -> StrainTwoRegimeResult:
    return StrainTwoRegimeResult(
        status=status,
        change_time=_nat(),
        candidate_time=_nat(),
        level_before=np.nan,
        level_after=np.nan,
        absolute_drop=np.nan,
        relative_drop=np.nan,
        j0=float(j0),
        j1_best=np.nan,
        fit_improvement=np.nan,
        n_blocks_before=0,
        n_blocks_after=0,
        n_valid_blocks=int(n_valid_blocks),
        aggregation_hours=float(config.aggregation_hours),
        n_comparable_alternatives=0,
        comparable_alternative_time=_nat(),
    )


def aggregate_strain_blocks(
    time: Any,
    strain: Any,
    config: StrainTwoRegimeConfig = StrainTwoRegimeConfig(),
    *,
    missing_values: Iterable[float] = (),
) -> pd.DataFrame:
    """Aggregate valid strain observations into UTC-aligned block medians.

    Block timestamps are centers of non-overlapping intervals aligned to the
    Unix epoch. Blocks below ``min_samples_per_block`` retain their sample count
    but have a missing median. The table is restricted to intervals between the
    first and last valid strain observations.
    """
    raw_time = np.asarray(time).reshape(-1)
    values = np.asarray(strain, dtype=float).reshape(-1)
    if len(raw_time) != len(values):
        raise ValueError("strain length does not match time length")
    parsed = pd.to_datetime(raw_time, errors="coerce", utc=True)
    valid = ~pd.isna(parsed) & np.isfinite(values)
    for sentinel in missing_values:
        valid &= values != float(sentinel)
    columns = (
        "time_start", "time_end", "time_center", "n_valid", "strain_median",
    )
    if not valid.any():
        return pd.DataFrame(columns=columns)

    time_ns = (
        pd.DatetimeIndex(parsed[valid])
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    valid_values = values[valid]
    order = np.argsort(time_ns, kind="stable")
    time_ns = time_ns[order]
    valid_values = valid_values[order]
    block_ns = int(round(config.aggregation_hours * 3_600_000_000_000))
    first_start = int(time_ns[0] // block_ns * block_ns)
    last_start = int(time_ns[-1] // block_ns * block_ns)
    starts = np.arange(first_start, last_start + block_ns, block_ns, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for start in starts:
        end = int(start + block_ns)
        selected = (time_ns >= start) & (time_ns < end)
        count = int(selected.sum())
        median = (
            float(np.median(valid_values[selected]))
            if count >= config.min_samples_per_block else np.nan
        )
        rows.append({
            "time_start": pd.Timestamp(start, unit="ns", tz="UTC"),
            "time_end": pd.Timestamp(end, unit="ns", tz="UTC"),
            "time_center": pd.Timestamp(start + block_ns // 2, unit="ns", tz="UTC"),
            "n_valid": count,
            "strain_median": median,
        })
    return pd.DataFrame(rows, columns=columns)


def robust_two_regime_strain_change(
    time: Any,
    strain: Any,
    config: StrainTwoRegimeConfig = StrainTwoRegimeConfig(),
    *,
    missing_values: Iterable[float] = (),
) -> StrainTwoRegimeDetection:
    """Fit one robust downward change point to aggregated strain blocks.

    A candidate time is the left boundary of the first valid block in the
    post-change regime. Missing blocks are never filled, smoothed, or included
    in either L1 cost.
    """
    blocks = aggregate_strain_blocks(
        time, strain, config, missing_values=missing_values,
    )
    candidate_columns = (
        "split_index", "time", "level_before", "level_after", "absolute_drop",
        "relative_drop", "j1", "n_blocks_before", "n_blocks_after", "downward",
    )
    valid_blocks = blocks[blocks.strain_median.notna()].reset_index(drop=True)
    n_valid = len(valid_blocks)
    minimum_side_blocks = int(np.ceil(
        config.minimum_side_duration_hours / config.aggregation_hours
    ))
    if n_valid < 2 * minimum_side_blocks:
        return StrainTwoRegimeDetection(
            _empty_result("insufficient_data", config, n_valid_blocks=n_valid),
            blocks,
            pd.DataFrame(columns=candidate_columns),
        )

    values = valid_blocks.strain_median.to_numpy(dtype=float)
    null_level = float(np.median(values))
    j0 = float(np.abs(values - null_level).sum())
    rows: list[dict[str, Any]] = []
    for split in range(minimum_side_blocks, n_valid - minimum_side_blocks + 1):
        before = values[:split]
        after = values[split:]
        level_before = float(np.median(before))
        level_after = float(np.median(after))
        drop = level_before - level_after
        relative = drop / max(abs(level_before), config.relative_floor)
        j1 = float(
            np.abs(before - level_before).sum()
            + np.abs(after - level_after).sum()
        )
        rows.append({
            "split_index": split,
            "time": valid_blocks.iloc[split].time_start,
            "level_before": level_before,
            "level_after": level_after,
            "absolute_drop": float(drop),
            "relative_drop": float(relative),
            "j1": j1,
            "n_blocks_before": split,
            "n_blocks_after": n_valid - split,
            "downward": bool(drop > 0),
        })
    candidates = pd.DataFrame(rows, columns=candidate_columns)
    downward = candidates[candidates.downward]
    if downward.empty:
        return StrainTwoRegimeDetection(
            _empty_result(
                "no_change", config, n_valid_blocks=n_valid, j0=j0,
            ),
            blocks,
            candidates,
        )

    best = downward.loc[downward.j1.idxmin()]
    if j0 > np.finfo(float).eps:
        fit_improvement = float((j0 - float(best.j1)) / j0)
    else:
        fit_improvement = 0.0
    best_time = pd.Timestamp(best.time)
    separated = (
        (downward.time - best_time).abs()
        >= pd.Timedelta(hours=config.ambiguity.minimum_separation_hours)
    )
    tolerance = max(
        np.finfo(float).eps,
        float(best.j1) * config.ambiguity.comparable_cost_fraction,
    )
    alternatives = downward[
        separated & (downward.j1 <= float(best.j1) + tolerance)
    ].sort_values(["j1", "time"], kind="stable")

    absolute_passes = float(best.absolute_drop) >= config.minimum_drop_absolute
    relative_passes = float(best.relative_drop) >= config.minimum_drop_relative
    improvement_passes = fit_improvement >= config.minimum_fit_improvement
    useful = improvement_passes and (absolute_passes or relative_passes)
    if useful and not alternatives.empty:
        status = "ambiguous"
    elif improvement_passes and absolute_passes and relative_passes:
        status = "clear"
    elif useful:
        status = "weak"
    else:
        status = "no_change"

    accepted_time = (
        best_time.to_datetime64() if status in {"clear", "weak"} else _nat()
    )
    alternative_time = (
        _nat() if alternatives.empty
        else pd.Timestamp(alternatives.iloc[0].time).to_datetime64()
    )
    result = StrainTwoRegimeResult(
        status=status,
        change_time=accepted_time,
        candidate_time=best_time.to_datetime64(),
        level_before=float(best.level_before),
        level_after=float(best.level_after),
        absolute_drop=float(best.absolute_drop),
        relative_drop=float(best.relative_drop),
        j0=j0,
        j1_best=float(best.j1),
        fit_improvement=fit_improvement,
        n_blocks_before=int(best.n_blocks_before),
        n_blocks_after=int(best.n_blocks_after),
        n_valid_blocks=n_valid,
        aggregation_hours=float(config.aggregation_hours),
        n_comparable_alternatives=len(alternatives),
        comparable_alternative_time=alternative_time,
    )
    return StrainTwoRegimeDetection(result, blocks, candidates)
