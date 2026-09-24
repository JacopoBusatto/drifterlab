"""Run production strain detection and aggregation sensitivity on selected platforms."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from drifterlab.experiments.arcterx.raw_drogue import (
    RawDrogueSignals,
    read_raw_drogue_signals,
)
from drifterlab.qc.strain_two_regime import (
    StrainTwoRegimeConfig,
    StrainTwoRegimeDetection,
    robust_two_regime_strain_change,
)


def load_experiment_settings(path: str | Path) -> tuple[StrainTwoRegimeConfig, tuple[float, ...]]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Experimental settings must be a mapping")
    unknown = set(data) - {"strain_two_regime", "sensitivity_aggregation_hours"}
    if unknown:
        raise ValueError(f"Unknown experimental setting keys: {sorted(unknown)}")
    detector = StrainTwoRegimeConfig.from_dict(data.get("strain_two_regime", {}))
    raw_hours = data.get("sensitivity_aggregation_hours", [3, 6, 12])
    if not isinstance(raw_hours, list) or not raw_hours:
        raise ValueError("sensitivity_aggregation_hours must be a nonempty list")
    hours: list[float] = []
    for value in raw_hours:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0):
            raise ValueError(
                "sensitivity_aggregation_hours must contain finite positive numbers"
            )
        numeric = float(value)
        if numeric not in hours:
            hours.append(numeric)
    return detector, tuple(hours)


def _input_settings(path: str | Path) -> tuple[Path, str, float]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    try:
        directory_value = data["input"]["directory"]
    except (TypeError, KeyError) as exc:
        raise ValueError("Configuration requires input.directory") from exc
    directory = Path(directory_value).expanduser()
    if not directory.is_absolute():
        directory = (path.parent / directory).resolve()
    pattern = data.get("input", {}).get("pattern", "*.mat")
    missing = data.get("processing", {}).get("missing_value", -999)
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("input.pattern must be a nonempty string")
    if (isinstance(missing, bool) or not isinstance(missing, (int, float))
            or not np.isfinite(missing)):
        raise ValueError("processing.missing_value must be finite")
    return directory, pattern, float(missing)


def _select_files(files: list[Path], platforms: list[str],
                  missing_value: float) -> list[Path]:
    requested = list(dict.fromkeys(map(str, platforms)))
    selected = {path.stem: path for path in files if path.stem in requested}
    unresolved = set(requested) - set(selected)
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
    return [selected[platform] for platform in requested]


def _strain_values(signals: RawDrogueSignals) -> np.ndarray:
    if signals.strain is None:
        return np.full(len(signals.time), np.nan)
    return np.asarray(signals.strain, dtype=float)


def sensitivity_detections(
    signals: RawDrogueSignals,
    config: StrainTwoRegimeConfig,
    aggregation_hours: tuple[float, ...],
    *,
    missing_value: float,
) -> dict[float, StrainTwoRegimeDetection]:
    hours = list(dict.fromkeys([config.aggregation_hours, *aggregation_hours]))
    strain = _strain_values(signals)
    return {
        float(value): robust_two_regime_strain_change(
            signals.time,
            strain,
            replace(config, aggregation_hours=float(value)),
            missing_values=(missing_value,),
        )
        for value in hours
    }


def _finite_text(value: float, specifier: str = ".3g") -> str:
    return format(value, specifier) if np.isfinite(value) else "n/a"


def plot_strain_experiment(
    signals: RawDrogueSignals,
    primary: StrainTwoRegimeDetection,
    sensitivity: dict[float, StrainTwoRegimeDetection],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    strain = _strain_values(signals)
    time = pd.to_datetime(signals.time, errors="coerce", utc=True)
    valid_raw = ~pd.isna(time) & np.isfinite(strain)
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].plot(time[valid_raw], strain[valid_raw], ".", ms=2, alpha=.22,
                 color=".35", label="Raw strain")
    blocks = primary.blocks
    valid_blocks = blocks[blocks.strain_median.notna()]
    axes[0].plot(
        valid_blocks.time_center, valid_blocks.strain_median,
        "o-", ms=3, lw=.8, color="tab:blue", label="Block medians",
    )

    result = primary.result
    candidate = pd.Timestamp(result.candidate_time)
    if pd.notna(candidate) and valid_raw.any():
        if candidate.tzinfo is None:
            candidate = candidate.tz_localize("UTC")
        first_valid = pd.DatetimeIndex(time[valid_raw]).min()
        last_valid = pd.DatetimeIndex(time[valid_raw]).max()
        axes[0].hlines(
            result.level_before, first_valid, candidate,
            color="tab:green", lw=2, label="Fitted pre-change level",
        )
        axes[0].hlines(
            result.level_after, candidate, last_valid,
            color="tab:orange", lw=2, label="Fitted post-change level",
        )
        accepted = not np.isnat(result.change_time)
        axes[0].axvline(
            candidate, color="tab:red" if accepted else ".4", ls="--", lw=1.6,
            label="Proposed change" if accepted else "Best rejected split",
        )

    sensitivity_colors = {3.0: "tab:cyan", 6.0: "tab:purple", 12.0: "tab:brown"}
    for hours, detection in sensitivity.items():
        if np.isclose(hours, result.aggregation_hours) or np.isnat(detection.result.change_time):
            continue
        marker_time = pd.Timestamp(detection.result.change_time)
        if marker_time.tzinfo is None:
            marker_time = marker_time.tz_localize("UTC")
        axes[0].axvline(
            marker_time, color=sensitivity_colors.get(hours, ".55"),
            ls=":", lw=1, alpha=.8, label=f"Sensitivity {hours:g} h",
        )

    annotation = (
        f"status={result.status}\n"
        f"levels={_finite_text(result.level_before)} -> "
        f"{_finite_text(result.level_after)}\n"
        f"drop={_finite_text(result.absolute_drop)} "
        f"({_finite_text(result.relative_drop, '.1%')})\n"
        f"fit improvement={_finite_text(result.fit_improvement, '.1%')}\n"
        f"valid blocks={result.n_blocks_before}/{result.n_blocks_after} "
        f"(before/after)"
    )
    axes[0].text(
        .01, .98, annotation, transform=axes[0].transAxes, va="top", ha="left",
        fontsize=9, bbox={"facecolor": "white", "alpha": .85, "edgecolor": ".7"},
    )
    axes[0].set_ylabel("Raw strain / block median")
    axes[0].set_title(
        f"Experimental robust two-regime strain fit - platform {signals.platform_code}"
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    candidates = primary.candidates
    if not candidates.empty:
        axes[1].plot(candidates.time, candidates.j1, color=".65", lw=.9,
                     label="All admissible split costs")
        downward = candidates[candidates.downward]
        axes[1].plot(downward.time, downward.j1, color="tab:purple", lw=1.2,
                     label="Downward split costs")
        if pd.notna(candidate):
            axes[1].axvline(candidate, color="tab:red", ls="--", lw=1.2,
                            label="Minimum downward cost")
        axes[1].axhline(result.j0, color=".25", ls=":", lw=1, label="One-level J0")
    else:
        axes[1].text(.5, .5, "No admissible split candidates", ha="center", va="center",
                     transform=axes[1].transAxes)
    axes[1].set_ylabel("L1 objective J1")
    axes[1].set_xlabel("UTC time")
    axes[1].set_title("Candidate objective curve")
    axes[1].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    if valid_raw.any():
        valid_times = pd.DatetimeIndex(time[valid_raw])
        axes[1].set_xlim(valid_times.min(), valid_times.max())
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _result_row(platform: str, source: Path, detection: StrainTwoRegimeDetection,
                *, primary: bool) -> dict[str, Any]:
    return {
        "platform": str(platform),
        "source_filename": source.name,
        "primary": primary,
        **detection.result.as_dict(),
    }


def run(
    config_path: str | Path,
    settings_path: str | Path,
    output: str | Path,
    *,
    platforms: list[str],
    sensitivity_hours: tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not platforms:
        raise ValueError("Select at least one platform with --platform")
    input_directory, pattern, missing_value = _input_settings(config_path)
    if not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist: {input_directory}")
    detector_config, configured_sensitivity = load_experiment_settings(settings_path)
    sensitivity_hours = configured_sensitivity if sensitivity_hours is None else sensitivity_hours
    output = Path(output).resolve()
    if output == input_directory or input_directory in output.parents:
        raise ValueError("Diagnostic output must be outside the raw input directory")
    files = sorted(path for path in input_directory.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No input files match {pattern!r} in {input_directory}")
    selected = _select_files(files, platforms, missing_value)

    primary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for index, path in enumerate(selected, start=1):
        signals = read_raw_drogue_signals(path, missing_value=missing_value)
        detections = sensitivity_detections(
            signals, detector_config, sensitivity_hours,
            missing_value=missing_value,
        )
        primary = detections[float(detector_config.aggregation_hours)]
        primary_rows.append(_result_row(
            signals.platform_code, path, primary, primary=True,
        ))
        for hours, detection in sorted(detections.items()):
            sensitivity_rows.append(_result_row(
                signals.platform_code, path, detection,
                primary=np.isclose(hours, detector_config.aggregation_hours),
            ))
        plot_strain_experiment(
            signals,
            primary,
            detections,
            output / "figures" / f"strain_two_regime_{signals.platform_code}.png",
        )
        result = primary.result
        print(
            f"Diagnosed {index}/{len(selected)}: {signals.platform_code} "
            f"status={result.status} time={result.change_time}",
            flush=True,
        )

    primary_table = pd.DataFrame(primary_rows).sort_values("platform", kind="stable")
    sensitivity_table = pd.DataFrame(sensitivity_rows).sort_values(
        ["platform", "aggregation_hours"], kind="stable"
    )
    output.mkdir(parents=True, exist_ok=True)
    primary_table.to_csv(output / "strain_two_regime_results.csv", index=False)
    sensitivity_table.to_csv(output / "strain_two_regime_sensitivity.csv", index=False)
    (output / "strain_two_regime_settings.json").write_text(
        json.dumps({
            "strain_two_regime": detector_config.as_dict(),
            "sensitivity_aggregation_hours": list(sensitivity_hours),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return primary_table.reset_index(drop=True), sensitivity_table.reset_index(drop=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "config", type=Path,
        help="ARCTERX drogue YAML; only raw input and missing-value settings are read",
    )
    result.add_argument(
        "--settings", type=Path,
        default=Path("configs/arcterx/strain_two_regime.yml"),
        help="experimental strain detector YAML",
    )
    result.add_argument(
        "--output", type=Path,
        default=Path("data/diagnostics/strain_two_regime"),
    )
    result.add_argument(
        "--platform", action="append", required=True,
        help="platform ID to diagnose; repeat for a small comparison set",
    )
    result.add_argument(
        "--sensitivity-hours", type=float, nargs="+",
        help="override diagnostic aggregation sensitivities (for example 3 6 12)",
    )
    result.add_argument(
        "--no-sensitivity", action="store_true",
        help="run only the configured primary aggregation",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.no_sensitivity and args.sensitivity_hours:
            raise ValueError("Use either --no-sensitivity or --sensitivity-hours, not both")
        sensitivity = (
            () if args.no_sensitivity else
            (None if args.sensitivity_hours is None else tuple(args.sensitivity_hours))
        )
        primary, _ = run(
            args.config,
            args.settings,
            args.output,
            platforms=args.platform,
            sensitivity_hours=sensitivity,
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser().exit(1, f"Strain two-regime diagnostic: {exc}\n")
    columns = (
        "platform", "status", "change_time", "level_before", "level_after",
        "absolute_drop", "relative_drop", "fit_improvement", "aggregation_hours",
    )
    print(primary.loc[:, columns].to_string(index=False))
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
