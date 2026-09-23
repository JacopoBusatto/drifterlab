"""Infer the first sustained native-QC segment without changing source products.

Run ``python scripts/diagnostics/sustained_start.py --help``.
All clock values represent UTC under the ARCTERX convention.
"""

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab import __version__
from drifterlab.qc.position import position_valid
if __package__:
    from .timing_diagnostic import assign_cohorts, plot_timelines
else:
    from timing_diagnostic import assign_cohorts, plot_timelines


@dataclass(frozen=True)
class Settings:
    continuity_gap_threshold_minutes: float = 30
    minimum_segment_duration_minutes: float = 60
    minimum_segment_n_obs: int = 13
    comparison_tolerance_seconds: float = 1
    nearly_same_tolerance_minutes: float = 5

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.continuity_gap_threshold_minutes <= 0 or self.minimum_segment_duration_minutes <= 0:
            raise ValueError("Continuity and sustained-duration thresholds must be positive")
        if self.minimum_segment_n_obs < 2 or int(self.minimum_segment_n_obs) != self.minimum_segment_n_obs:
            raise ValueError("minimum_segment_n_obs must be an integer of at least two")


@dataclass
class ValidTrack:
    platform_code: str
    time: np.ndarray
    source_index: np.ndarray


SEGMENT_COLUMNS = [
    "platform_code", "segment_index", "segment_start", "segment_end",
    "segment_duration_hours", "n_valid_positions", "n_distinct_times",
    "median_dt_seconds", "max_internal_gap_seconds", "start_source_obs_index",
    "end_source_obs_index",
]


def _utc(value):
    return pd.NaT if pd.isna(value) else pd.Timestamp(value).tz_localize("UTC")


def read_native_tracks(master: Path, timing: pd.DataFrame) -> list[ValidTrack]:
    """Read native arrays only; cross-check the prior diagnostic and validity mask."""
    if timing.platform_code.isna().any() or timing.platform_code.duplicated().any():
        raise ValueError("Timing table must have unique, nonmissing platform IDs")
    previous = timing.set_index("platform_code")
    tracks = []
    with xr.open_zarr(master, chunks=None) as ds:
        platforms = ds.platform_code.values.astype(str)
        if len(set(platforms)) != len(platforms) or set(platforms) != set(previous.index):
            raise ValueError("Master and timing table platform IDs must match one-to-one")
        counts = ds.n_obs_qc.values
        hashes = ds.source_sha256.values.astype(str)
        native = ds[["time_qc", "lon_qc", "lat_qc", "position_valid_qc", "source_obs_index_qc"]]
        for i, platform in enumerate(platforms):
            n = int(counts[i])
            if n < 0 or n > ds.sizes["obs_qc"]:
                raise ValueError(f"Invalid native observation count for {platform}")
            row = native.isel(trajectory=i, obs_qc=slice(0, n))
            time = row.time_qc.values.astype("datetime64[ns]")
            geometric = position_valid(row.lon_qc.values, row.lat_qc.values)
            if not np.array_equal(geometric, row.position_valid_qc.values):
                raise ValueError(f"Native position validity disagrees with coordinates for {platform}")
            valid = geometric & ~np.isnat(time)
            order = np.argsort(time[valid], kind="stable")
            track = ValidTrack(platform, time[valid][order], row.source_obs_index_qc.values[valid][order])
            finite = time[~np.isnat(time)]
            for column, values in [("qc_first_time", finite), ("qc_first_valid_position_time", track.time)]:
                current = _utc(values.min()) if len(values) else pd.NaT
                before = previous.loc[platform, column]
                if not ((pd.isna(current) and pd.isna(before)) or current == before):
                    raise ValueError(f"Master disagrees with previous {column} for {platform}")
            if hashes[i] != previous.loc[platform, "qc_source_sha256"]:
                raise ValueError(f"QC source hash mismatch for {platform}")
            tracks.append(track)
    return tracks


def gap_distribution(tracks: list[ValidTrack], tolerance_seconds: float = 1) -> dict:
    gaps = np.concatenate([np.diff(t.time) / np.timedelta64(1, "s") for t in tracks]) if tracks else np.array([])
    levels = {"median": .5, "p90": .9, "p95": .95, "p99": .99, "p99_9": .999, "max": 1}
    return {
        "n_trajectories": len(tracks), "n_valid_positions_with_time": sum(len(t.time) for t in tracks),
        "n_gaps": len(gaps), "n_zero_gaps": int((gaps == 0).sum()),
        "quantiles_seconds": {k: float(np.quantile(gaps, q)) if len(gaps) else None for k, q in levels.items()},
        "comparison_tolerance_seconds": tolerance_seconds,
        "exceedances": [{"threshold_minutes": m, "strict_count": int((gaps > m * 60).sum()),
                         "tolerance_adjusted_count": int((gaps > m * 60 + tolerance_seconds).sum())}
                        for m in (10, 15, 20, 30, 60, 360, 1440)],
    }


def segment_track(track: ValidTrack, settings: Settings) -> pd.DataFrame:
    """Stable-sort valid timed fixes and split strictly above gap + tolerance.

    Missing-position rows are absent here: continuity is measured between valid
    fixes, so short missing runs may be bridged. Duplicate times remain counted
    and are also reported as distinct-time counts.
    """
    if len(track.time) != len(track.source_index) or np.isnat(track.time).any():
        raise ValueError("ValidTrack must contain aligned, valid timestamps and source indices")
    order = np.argsort(track.time, kind="stable")
    time, source = track.time[order], track.source_index[order]
    rows = []
    if len(time):
        gaps = np.diff(time) / np.timedelta64(1, "s")
        boundaries = np.r_[0, np.flatnonzero(gaps > settings.continuity_gap_threshold_minutes * 60
                                           + settings.comparison_tolerance_seconds) + 1, len(time)]
        for k, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            internal = gaps[lo:hi - 1]
            rows.append({
                "platform_code": track.platform_code, "segment_index": k,
                "segment_start": _utc(time[lo]), "segment_end": _utc(time[hi - 1]),
                "segment_duration_hours": float((time[hi - 1] - time[lo]) / np.timedelta64(1, "h")),
                "n_valid_positions": int(hi - lo), "n_distinct_times": int(1 + (internal > 0).sum()),
                "median_dt_seconds": float(np.median(internal)) if len(internal) else np.nan,
                "max_internal_gap_seconds": float(internal.max()) if len(internal) else np.nan,
                "start_source_obs_index": int(source[lo]), "end_source_obs_index": int(source[hi - 1]),
            })
    result = pd.DataFrame(rows, columns=SEGMENT_COLUMNS)
    for col in ("segment_start", "segment_end"):
        result[col] = pd.to_datetime(result[col], utc=True).astype("datetime64[ns, UTC]")
    return result


def select_start(platform: str, segments: pd.DataFrame, settings: Settings) -> dict:
    qualifying = segments[
        (segments.n_valid_positions >= settings.minimum_segment_n_obs)
        & (segments.n_distinct_times >= settings.minimum_segment_n_obs)
        & (segments.segment_duration_hours * 3600 >= settings.minimum_segment_duration_minutes * 60
           - settings.comparison_tolerance_seconds)
    ]
    result = {
        "platform_code": platform, "inferred_start_time": pd.NaT,
        "delta_first_valid_to_inferred_hours": np.nan, "n_valid_segments": len(segments),
        "n_segments_before_inferred_start": None, "n_valid_positions_before_inferred_start": None,
        "inferred_start_segment_duration_hours": np.nan, "inferred_start_segment_n_obs": None,
        "inferred_start_segment_n_distinct_times": None, "inferred_start_source_obs_index": None,
        "start_method": "first_sustained_segment", "start_status": "no_sustained_segment_found",
        "start_nearly_same_as_first_valid": False,
        **asdict(settings),
    }
    if qualifying.empty:
        return result
    chosen = qualifying.iloc[0]
    k = int(chosen.segment_index)
    delta = (chosen.segment_start - segments.iloc[0].segment_start) / pd.Timedelta(hours=1)
    near = delta * 3600 <= settings.nearly_same_tolerance_minutes * 60 + settings.comparison_tolerance_seconds
    status = "multiple_early_segments" if k > 1 else "earlier_isolated_fragment_removed" if k else "same_or_nearly_same"
    result.update({
        "inferred_start_time": chosen.segment_start, "delta_first_valid_to_inferred_hours": delta,
        "n_segments_before_inferred_start": k,
        "n_valid_positions_before_inferred_start": int(segments.iloc[:k].n_valid_positions.sum()),
        "inferred_start_segment_duration_hours": chosen.segment_duration_hours,
        "inferred_start_segment_n_obs": int(chosen.n_valid_positions),
        "inferred_start_segment_n_distinct_times": int(chosen.n_distinct_times),
        "inferred_start_source_obs_index": int(chosen.start_source_obs_index),
        "start_status": status, "start_nearly_same_as_first_valid": bool(near),
    })
    return result


def start_table(tracks: list[ValidTrack], segments: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    grouped = dict(tuple(segments.groupby("platform_code", sort=False)))
    frame = pd.DataFrame([select_start(t.platform_code, grouped.get(t.platform_code, segments.iloc[:0]), settings) for t in tracks])
    frame["inferred_start_time"] = pd.to_datetime(frame.inferred_start_time, utc=True).astype("datetime64[ns, UTC]")
    for col in ("n_segments_before_inferred_start", "n_valid_positions_before_inferred_start",
                "inferred_start_segment_n_obs", "inferred_start_segment_n_distinct_times", "inferred_start_source_obs_index"):
        frame[col] = frame[col].astype("Int64")
    return frame


def populations(times: pd.Series) -> list[dict]:
    frame = pd.DataFrame({"qc_first_valid_position_time": times})
    labels = assign_cohorts(frame)
    return [{"population": label, "n": int((labels == label).sum()),
             "first_time_utc": times[labels == label].min().isoformat(),
             "last_time_utc": times[labels == label].max().isoformat()}
            for label in sorted(labels[times.notna()].unique())]


def compare_starts(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    left = candidate.set_index("platform_code").inferred_start_time
    right = baseline.set_index("platform_code").inferred_start_time.reindex(left.index)
    changed = ~((left == right) | (left.isna() & right.isna()))
    changes = [{"platform_code": p,
                "baseline_start_utc": None if pd.isna(right[p]) else right[p].isoformat(),
                "candidate_start_utc": None if pd.isna(left[p]) else left[p].isoformat(),
                "delta_hours": None if pd.isna(left[p]) or pd.isna(right[p]) else float((left[p] - right[p]) / pd.Timedelta(hours=1))}
               for p in left.index[changed]]
    groups = populations(candidate.inferred_start_time)
    baseline_groups = populations(baseline.inferred_start_time)
    return {"n_changed": len(changes), "changes": changes, "n_without_sustained_start": int(left.isna().sum()),
            "populations": groups, "population_counts_unchanged": [(g["population"], g["n"]) for g in groups]
            == [(g["population"], g["n"]) for g in baseline_groups]}


def _write_table(frame: pd.DataFrame, path: Path, *, parquet: bool = True) -> None:
    if parquet:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    csv = frame.copy()
    for col in csv:
        if isinstance(csv[col].dtype, pd.DatetimeTZDtype):
            csv[col] = csv[col].map(lambda t: "" if pd.isna(t) else t.isoformat())
    csv.to_csv(path.with_suffix(".csv"), index=False)


def write_report(frame: pd.DataFrame, summary: dict, output: Path) -> None:
    settings = summary["settings"]
    gap = summary["gap_distribution"]
    if summary["counts"]["exactly_equal_first_valid"] == len(frame):
        interpretation = (
            "The earlier-date raw observations or QC timestamps in the cross-check below do not enter the valid-QC sequence: "
            "their timestamps were omitted by supplied QC or their positions are invalid. The sustained rule confirms the later starts; "
            "it does not newly remove those points from first-valid QC times."
        )
    else:
        interpretation = (
            f"The sustained rule skips early valid-QC segments for {summary['counts']['one_or_more_early_fragments_skipped']} trajectories; "
            f"{summary['counts']['no_sustained_segment_found']} have no qualifying segment. "
            "The segment table distinguishes removed fragments from earlier timestamps that already lacked valid QC positions."
        )
    lines = ["# ARCTERX MicroSVP sustained-start diagnostic", "",
             "Native QC only. UTC throughout. Inferred starts are derived diagnostics, not verified release times.", "",
             "## Decision and observed gap distribution", "",
             f"Use a **{settings['continuity_gap_threshold_minutes']:g}-minute** continuity gap and at least "
             f"**{settings['minimum_segment_duration_minutes']:g} minutes / {settings['minimum_segment_n_obs']} valid observations** "
             "at distinct times. Both duration and count must pass. A segment starts at its first valid fix; it is not shifted forward by the duration requirement.", "",
             f"The {gap['n_gaps']:,} consecutive valid-QC gaps have the following quantiles:", "",
             "| Quantile | Seconds |", "|---|---:|"]
    lines += [f"| {k} | {v:.6f} |" for k, v in gap["quantiles_seconds"].items() if v is not None]
    lines += ["", "| Gap threshold (minutes) | Strictly exceeding | Exceeding with comparison tolerance |", "|---:|---:|---:|"]
    lines += [f"| {r['threshold_minutes']} | {r['strict_count']:,} | {r['tolerance_adjusted_count']:,} |" for r in gap["exceedances"]]
    lines += ["", "The default 30-minute continuity choice was calibrated on the delivered campaign's gaps (p99 about 6 minutes and p99.9 about 10 minutes). "
              "It allows occasional missing fixes while separating long interruptions. The effective setting is stated above. "
              "This is a documented analysis choice, not a uniquely determined physical boundary. "
              "The continuity sensitivity check below makes its effect on inferred starts explicit.", "",
              f"Comparisons allow {settings['comparison_tolerance_seconds']:g} second(s) for MATLAB clock precision: split when gap > threshold + tolerance; "
              "accept duration >= minimum - tolerance. Saved timestamps and gap measurements retain full nanosecond precision. "
              "Strict and tolerance-adjusted gap counts are both reported because tiny encoding errors can straddle exact 10/15/30-minute boundaries.", "",
              "Missing or out-of-range positions and invalid times do not enter segmentation. Short runs of missing positions may therefore be bridged up to the continuity threshold. "
              "Duplicate timed fixes remain in segment counts, but cannot inflate the required distinct-time support. No drogue, interpolation, geographic-box, or residual-speed filtering is applied.", "",
              "## Selected starts", ""]
    lines += [f"- {name}: {count}" for name, count in summary["counts"].items()]
    lines += ["", "| Inferred-start timing population | Drifters | First UTC | Last UTC |", "|---|---:|---|---|"]
    lines += [f"| {p['population']} | {p['n']} | {p['first_time_utc']} | {p['last_time_utc']} |" for p in summary["populations"]]
    lines += ["", "Populations combine consecutive occupied UTC calendar dates, using the same descriptive rule as the previous diagnostic. "
              "No expected counts, release windows, or deployment labels are used to assign them.", "",
              "## Do the isolated early timestamps disappear from the inferred start?", "",
              f"Under the selected rule, **{summary['counts']['exactly_equal_first_valid']} / {len(frame)}** inferred starts equal first valid QC times exactly. "
              + interpretation + " All original timing columns remain preserved.", "",
              "Earlier-date raw/QC records provide a direct cross-check:", "",
              "| Platform | Raw first UTC | QC first UTC | First valid QC UTC | Inferred UTC |", "|---|---|---|---|---|"]
    for row in frame.itertuples():
        valid_day = row.qc_first_valid_position_time.normalize() if pd.notna(row.qc_first_valid_position_time) else pd.NaT
        if row.raw_first_time < valid_day or row.qc_first_time < valid_day:
            values = [getattr(row, c) for c in ("raw_first_time", "qc_first_time", "qc_first_valid_position_time", "inferred_start_time")]
            labels = ["missing" if pd.isna(t) else t.round("s").strftime("%Y-%m-%d %H:%M:%S") for t in values]
            lines.append(f"| {row.platform_code} | " + " | ".join(labels) + " |")
    lines += ["", "Report times in this cross-check are rounded to seconds for display only.", "", "## Sensitivity", "",
              "| Minimum duration (minutes) | Required observations | Changed starts | Population counts stable |", "|---:|---:|---:|---|"]
    duration_details = []
    for item in summary["duration_sensitivity"]:
        lines.append(f"| {item['minimum_segment_duration_minutes']:g} | {item['minimum_segment_n_obs']} | {item['n_changed']} | {item['population_counts_unchanged']} |")
        for change in item["changes"]:
            duration_details.append(f"At {item['minimum_segment_duration_minutes']:g} minutes, platform **{change['platform_code']}**: {change['baseline_start_utc']} -> {change['candidate_start_utc']}; delta {change['delta_hours']} hours.")
    lines += [""] + duration_details
    lines += ["", "Continuity sensitivity at the selected sustained criterion:", "",
              "| Continuity threshold (minutes) | Changed starts | Population counts stable |", "|---:|---:|---|"]
    for item in summary["continuity_sensitivity"]:
        lines.append(f"| {item['continuity_gap_threshold_minutes']:g} | {item['n_changed']} | {item['population_counts_unchanged']} |")
    for item in summary["continuity_sensitivity"]:
        for change in item["changes"]:
            lines += ["", f"At {item['continuity_gap_threshold_minutes']:g} minutes, {change['platform_code']} shifts by {change['delta_hours']} hours."]
    lines += ["", "## Manual review", ""]
    review = frame[frame.manual_review_recommended]
    if review.empty:
        lines.append("No unresolved starts or duration-sensitive platforms were found.")
    else:
        for row in review.itertuples():
            support = (f"Selected first segment: {row.inferred_start_segment_duration_hours:g} hours and {row.inferred_start_segment_n_obs} observations."
                       if pd.notna(row.inferred_start_time) else "No segment satisfies the selected criterion.")
            lines.append(f"- **{row.platform_code}**: {row.manual_review_reason}. {support}")
    lines += ["", "A duration-sensitive flag records uncertainty in this analysis choice; it does not invalidate the default start. "
              "Late same-day starts remain visible; temporal continuity does not establish their actual release time.", "",
              "The main CSV/Parquet preserves every prior diagnostic column and adds settings, inferred times, segment support, descriptive status, and review flags. "
              "The segments CSV/Parquet exposes every segment, including rejected fragments. The sensitivity CSV contains per-platform starts for each duration criterion. "
              "The master Zarr, prior timing outputs, inventory, and source data are read-only inputs."]
    (output / "microsvp_sustained_start_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(master: Path, timing_table: Path, output: Path, *, settings: Settings = Settings(), figures: bool = True) -> dict:
    master, timing_table, output = master.resolve(), timing_table.resolve(), output.resolve()
    if output == master or master in output.parents:
        raise ValueError("Diagnostic output must be outside the master Zarr")
    if timing_table == output / "microsvp_sustained_start_diagnostic.parquet":
        raise ValueError("Output would overwrite the input timing table")
    timing = pd.read_parquet(timing_table)
    if timing.empty or timing.platform_code.isna().any():
        raise ValueError("Timing table must have at least one trajectory and nonmissing IDs")
    if "inferred_start_time" in timing:
        raise ValueError("Use the original timing diagnostic as input, not a derived sustained-start table")
    timing["platform_code"] = timing.platform_code.astype(str)
    tracks = read_native_tracks(master, timing)
    gap = gap_distribution(tracks, settings.comparison_tolerance_seconds)
    print("Valid native QC gap distribution:\n" + json.dumps(gap, indent=2), flush=True)
    segments = pd.concat([segment_track(t, settings) for t in tracks], ignore_index=True)
    baseline = start_table(tracks, segments, settings)
    frame = timing.merge(baseline, on="platform_code", validate="one_to_one", sort=False)
    frame["inferred_start_timing_population"] = assign_cohorts(pd.DataFrame({"qc_first_valid_position_time": frame.inferred_start_time}))
    frame["manual_review_recommended"] = frame.inferred_start_time.isna()
    frame["manual_review_reason"] = np.where(frame.inferred_start_time.isna(), "no_sustained_segment_found", "")
    duration_checks, sensitivity_frames = [], []
    for duration in sorted({30, 60, 120, settings.minimum_segment_duration_minutes}):
        required = settings.minimum_segment_n_obs if duration == settings.minimum_segment_duration_minutes else math.ceil(duration / 5) + 1
        choice = Settings(**{**asdict(settings), "minimum_segment_duration_minutes": duration, "minimum_segment_n_obs": required})
        candidate = start_table(tracks, segments, choice)
        sensitivity_frames.append(candidate)
        comparison = compare_starts(candidate, baseline)
        duration_checks.append({"minimum_segment_duration_minutes": duration, "minimum_segment_n_obs": required, **comparison})
        for change in comparison["changes"]:
            mask = frame.platform_code == change["platform_code"]
            frame.loc[mask, "manual_review_recommended"] = True
            frame.loc[mask, "manual_review_reason"] = frame.loc[mask, "manual_review_reason"].map(lambda text: "; ".join(filter(None, [text, f"start changes under {duration:g}-minute criterion"])))
    continuity_checks = []
    for minutes in sorted({10, 15, 20, 30, 60, settings.continuity_gap_threshold_minutes}):
        choice = Settings(**{**asdict(settings), "continuity_gap_threshold_minutes": minutes})
        alternate_segments = segments if minutes == settings.continuity_gap_threshold_minutes else pd.concat([segment_track(t, choice) for t in tracks], ignore_index=True)
        continuity_checks.append({"continuity_gap_threshold_minutes": minutes, **compare_starts(start_table(tracks, alternate_segments, choice), baseline)})
    counts = {
        "same_or_nearly_same": int((frame.delta_first_valid_to_inferred_hours * 60 <= settings.nearly_same_tolerance_minutes + settings.comparison_tolerance_seconds / 60).sum()),
        "exactly_equal_first_valid": int((frame.inferred_start_time == frame.qc_first_valid_position_time).sum()),
        "one_or_more_early_fragments_skipped": int((frame.n_segments_before_inferred_start > 0).sum()),
        "multiple_early_segments_skipped": int((frame.n_segments_before_inferred_start > 1).sum()),
        "no_sustained_segment_found": int(frame.inferred_start_time.isna().sum()),
    }
    summary = {
        "processed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "software_version": __version__,
        "time_reference": "UTC", "start_method": "first_sustained_segment", "settings": asdict(settings),
        "inputs": {"master_zarr": str(master), "timing_table": str(timing_table),
                   "timing_table_sha256": sha256(timing_table.read_bytes()).hexdigest(),
                   "master_consolidated_metadata_sha256": sha256((master / ".zmetadata").read_bytes()).hexdigest()},
        "diagnostic_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "gap_distribution": gap, "n_segments": len(segments), "counts": counts,
        "status_counts": {k: int(v) for k, v in frame.start_status.value_counts().items()},
        "populations": populations(frame.inferred_start_time),
        "previous_first_valid_qc_populations": populations(frame.qc_first_valid_position_time),
        "duration_sensitivity": duration_checks, "continuity_sensitivity": continuity_checks,
        "manual_review_platforms": frame.loc[frame.manual_review_recommended, "platform_code"].tolist(),
        "conventions": {"validity": "Finite in-range native QC positions and valid time; no drogue or audit-speed condition",
                        "continuity": "Consecutive valid timed fixes; gap > threshold + tolerance splits",
                        "sustained": "Both duration >= minimum - tolerance and n distinct observation times >= minimum count",
                        "no_fallback": True, "population_rule": "Consecutive occupied UTC calendar dates; descriptive only"},
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_table(frame, output / "microsvp_sustained_start_diagnostic")
    _write_table(segments, output / "microsvp_valid_qc_segments")
    _write_table(pd.concat(sensitivity_frames, ignore_index=True), output / "microsvp_sustained_start_sensitivity", parquet=False)
    (output / "microsvp_valid_qc_gap_distribution.json").write_text(json.dumps(gap, indent=2) + "\n", encoding="utf-8")
    (output / "microsvp_sustained_start_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(frame, summary, output)
    if figures:
        plot_timelines(frame, output)
    print(json.dumps({k: summary[k] for k in ("settings", "counts", "populations", "duration_sensitivity", "manual_review_platforms")}, indent=2))
    print(f"Sustained-start diagnostic: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("data/ARCTERX_MicroSVP_QC.zarr"))
    parser.add_argument("--timing-table", type=Path, default=Path("data/diagnostics/microsvp_timing_diagnostic.parquet"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/diagnostics"))
    parser.add_argument("--continuity-gap-minutes", type=float, default=30)
    parser.add_argument("--minimum-duration-minutes", type=float, default=60)
    parser.add_argument("--minimum-observations", type=int, help="Default: ceil(duration / 5 minutes) + 1")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    required = args.minimum_observations if args.minimum_observations is not None else math.ceil(args.minimum_duration_minutes / 5) + 1
    settings = Settings(args.continuity_gap_minutes, args.minimum_duration_minutes, required)
    run(args.master, args.timing_table, args.output_directory, settings=settings, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
