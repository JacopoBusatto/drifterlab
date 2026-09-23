"""Explain stored native-QC jump flags without changing preprocessing or data.

Run ``python scripts/diagnostics/residual_jumps.py --help`` from the repository.
"""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab import __version__
from drifterlab.io.matlab import matlab_datenum_to_datetime64, normalize_missing, read_matlab
from drifterlab.qc import flags
from drifterlab.qc.position import position_valid


THRESHOLD_M_S = 3.0
CADENCE_TOLERANCE_SECONDS = 1.0  # Summary bins only; never changes speeds or flags.
FIELDS = ["time_qc", "lon_qc", "lat_qc", "position_valid_qc", "source_obs_index_qc",
          "audit_speed_qc", "residual_jump_flag_qc", "speed_qc_source"]
JUMP_COLUMNS = [
    "platform_code", "source_filename", "source_sha256", "time_1", "time_2",
    "dt_minutes", "distance_m", "speed_m_s", "row_index_1", "row_index_2",
    "n_rows_between", "n_missing_positions_between", "n_invalid_positions_between",
    "position_1_valid", "position_2_valid", "source_obs_index_1", "source_obs_index_2",
    "n_source_rows_between", "lon_1", "lat_1", "lon_2", "lat_2",
    "invalid_position_immediately_before_pair", "invalid_position_immediately_after_pair",
    "supplied_speed_qc_at_time_1_m_s_definition_unverified",
    "supplied_speed_qc_at_time_2_m_s_definition_unverified",
]


@dataclass
class Track:
    platform: str
    source_filename: str
    source_sha256: str
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    valid: np.ndarray
    source_index: np.ndarray
    speed: np.ndarray
    flags: np.ndarray
    supplied_speed: np.ndarray


def load_track(ds: xr.Dataset, index: int) -> Track:
    n = int(ds.n_obs_qc.values[index])
    if n < 0 or n > ds.sizes["obs_qc"]:
        raise ValueError("Invalid native observation count")
    row = ds[FIELDS].isel(trajectory=index, obs_qc=slice(0, n))
    return Track(str(ds.platform_code.values[index]), str(ds.source_filename.values[index]),
                 str(ds.source_sha256.values[index]), row.time_qc.values.astype("datetime64[ns]"),
                 row.lon_qc.values, row.lat_qc.values, row.position_valid_qc.values,
                 row.source_obs_index_qc.values, row.audit_speed_qc.values,
                 row.residual_jump_flag_qc.values, row.speed_qc_source.values)


def flagged_pairs(track: Track) -> tuple[pd.DataFrame, dict]:
    """Reproduce every stored speed/flag before tracing its immediate predecessor."""
    arrays = (track.lon, track.lat, track.valid, track.source_index, track.speed, track.flags, track.supplied_speed)
    if any(a.shape != track.time.shape for a in arrays) or track.time.ndim != 1:
        raise ValueError(f"Misaligned native arrays for {track.platform}")
    if not np.array_equal(position_valid(track.lon, track.lat), track.valid):
        raise ValueError(f"Position mask mismatch for {track.platform}")
    if not np.array_equal(np.argsort(track.time, kind="stable"), np.arange(len(track.time))):
        raise ValueError(f"Native rows are not in stored chronological order for {track.platform}")
    speed = flags.consecutive_speed(track.time, track.lon, track.lat)
    if not np.array_equal(speed, track.speed, equal_nan=True):
        raise ValueError(f"Stored audit speeds do not reproduce exactly for {track.platform}")
    if not np.array_equal(speed > THRESHOLD_M_S, track.flags):
        raise ValueError(f"Stored residual flags do not reproduce exactly for {track.platform}")
    dt = flags.interval_seconds(track.time)
    missing = ~np.isfinite(track.lon) | ~np.isfinite(track.lat)
    rows = []
    for right in np.flatnonzero(track.flags):
        left = int(right - 1)
        rows.append({
            "platform_code": track.platform, "source_filename": track.source_filename,
            "source_sha256": track.source_sha256,
            "time_1": pd.Timestamp(track.time[left]).tz_localize("UTC"),
            "time_2": pd.Timestamp(track.time[right]).tz_localize("UTC"),
            "dt_minutes": float(dt[right] / 60), "distance_m": float(speed[right] * dt[right]),
            "speed_m_s": float(speed[right]), "row_index_1": left, "row_index_2": int(right),
            "n_rows_between": int(right - left - 1),
            "n_missing_positions_between": int(missing[left + 1:right].sum()),
            "n_invalid_positions_between": int((~track.valid[left + 1:right]).sum()),
            "position_1_valid": bool(track.valid[left]), "position_2_valid": bool(track.valid[right]),
            "source_obs_index_1": int(track.source_index[left]), "source_obs_index_2": int(track.source_index[right]),
            "n_source_rows_between": int(abs(track.source_index[right] - track.source_index[left]) - 1),
            "lon_1": track.lon[left], "lat_1": track.lat[left], "lon_2": track.lon[right], "lat_2": track.lat[right],
            "invalid_position_immediately_before_pair": bool(left > 0 and not track.valid[left - 1]),
            "invalid_position_immediately_after_pair": bool(right + 1 < len(track.time) and not track.valid[right + 1]),
            "supplied_speed_qc_at_time_1_m_s_definition_unverified": track.supplied_speed[left],
            "supplied_speed_qc_at_time_2_m_s_definition_unverified": track.supplied_speed[right],
        })
    frame = pd.DataFrame(rows, columns=JUMP_COLUMNS)
    for col in ("time_1", "time_2"):
        frame[col] = pd.to_datetime(frame[col], utc=True).astype("datetime64[ns, UTC]")
    eligible = np.isfinite(speed)
    pair_good = track.valid & ~np.isnat(track.time)
    blocked = ~(pair_good[:-1] & pair_good[1:])
    stats = {
        "platform_code": track.platform, "n_native_observations": len(track.time),
        "n_eligible_adjacent_pairs": int(eligible.sum()), "n_flagged_jumps": int(track.flags.sum()),
        "flags_per_10000_eligible_pairs": float(10000 * track.flags.sum() / eligible.sum()) if eligible.any() else 0.,
        "n_pairs_touching_invalid_position_or_time": int(blocked.sum()),
        "n_finite_speeds_on_blocked_pairs": int(np.isfinite(speed[1:][blocked]).sum()),
        "max_flagged_speed_m_s": float(speed[track.flags].max()) if track.flags.any() else np.nan,
        "all_speeds_reproduced_exactly": True, "all_flags_reproduced_exactly": True,
    }
    return frame, stats


def verify_source(track: Track, directory: Path) -> int:
    """Check flagged endpoints against source MATLAB indices, values and clock."""
    path = directory / track.source_filename
    if path.resolve().parent != directory.resolve():
        raise ValueError("Source filename must resolve within the QC directory")
    loaded = read_matlab(path)
    if loaded.sha256 != track.source_sha256:
        raise ValueError(f"Source hash mismatch: {path}")
    native = loaded.values["drifter"]
    right = np.flatnonzero(track.flags)
    endpoints = np.unique(np.r_[right - 1, right])
    indices = track.source_index[endpoints]
    for field, values in [("longitude", track.lon), ("latitude", track.lat), ("speed", track.supplied_speed)]:
        source = np.atleast_1d(normalize_missing(native[field]))
        if not np.array_equal(source[indices], values[endpoints], equal_nan=True):
            raise ValueError(f"Source {field} does not match flagged endpoints for {track.platform}")
    source_time = np.atleast_1d(matlab_datenum_to_datetime64(native["time"]))
    if not np.array_equal(source_time[indices], track.time[endpoints]):
        raise ValueError(f"Source times do not match flagged endpoints for {track.platform}")
    return len(endpoints)


def quantiles(values: pd.Series) -> dict:
    return {name: float(values.quantile(q)) if len(values) else None
            for name, q in [("min", 0), ("median", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1)]}


def duration_counts(minutes: pd.Series, tolerance_seconds: float) -> dict:
    return {"dt_le_6_min": int((minutes <= 6 + tolerance_seconds / 60).sum()),
            "dt_le_10_min": int((minutes <= 10 + tolerance_seconds / 60).sum()),
            "dt_gt_10_min": int((minutes > 10 + tolerance_seconds / 60).sum()),
            "dt_gt_30_min": int((minutes > 30 + tolerance_seconds / 60).sum())}


def select_cases(frame: pd.DataFrame, platforms: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    ranked = frame.sort_values(["speed_m_s", "platform_code", "row_index_2"], ascending=[False, True, True])
    top_platform = platforms.iloc[0].platform_code
    nominal = ranked[(ranked.dt_minutes * 60 - 300).abs() <= CADENCE_TOLERANCE_SECONDS]
    choices = [("Largest speed", ranked.iloc[0]),
               ("Largest displacement", frame.loc[frame.distance_m.idxmax()])]
    if not nominal.empty:
        choices.append(("Largest speed at nominal 5-minute cadence", nominal.iloc[0]))
    choices.append(("Highest-count platform: its largest speed", ranked[ranked.platform_code == top_platform].iloc[0]))
    cases, seen = [], set()
    for reason, row in choices:
        key = (row.platform_code, int(row.row_index_2))
        if key in seen:
            continue
        seen.add(key)
        cases.append({"case_id": chr(65 + len(cases)), "reason": reason, "platform_code": key[0],
                      "row_index_2": key[1], "time_2_utc": row.time_2.isoformat(),
                      "speed_m_s": row.speed_m_s, "distance_m": row.distance_m,
                      "dt_minutes": row.dt_minutes})
    return cases


def case_context(ds: xr.Dataset, cases: list[dict]) -> pd.DataFrame:
    rows = []
    platforms = list(ds.platform_code.values.astype(str))
    for case in cases:
        track = load_track(ds, platforms.index(case["platform_code"]))
        center = track.time[case["row_index_2"]]
        selected = np.flatnonzero((track.time >= center - np.timedelta64(45, "m")) &
                                  (track.time <= center + np.timedelta64(45, "m")))
        first = min(int(selected[0]), case["row_index_2"] - 1)
        for i in range(first, int(selected[-1]) + 1):
            rows.append({"case_id": case["case_id"], "platform_code": track.platform,
                         "row_index": i, "source_obs_index": int(track.source_index[i]),
                         "time": pd.Timestamp(track.time[i]).tz_localize("UTC"),
                         "lon_qc": track.lon[i], "lat_qc": track.lat[i],
                         "position_valid_qc": bool(track.valid[i]), "audit_speed_qc_m_s": track.speed[i],
                         "residual_jump_flag_qc": bool(track.flags[i]),
                         "supplied_speed_qc_m_s_definition_unverified": track.supplied_speed[i]})
    return pd.DataFrame(rows, columns=["case_id", "platform_code", "row_index", "source_obs_index", "time",
                                       "lon_qc", "lat_qc", "position_valid_qc", "audit_speed_qc_m_s",
                                       "residual_jump_flag_qc", "supplied_speed_qc_m_s_definition_unverified"])


def write_table(frame: pd.DataFrame, path: Path, *, parquet: bool = False) -> None:
    if parquet:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
    csv = frame.copy()
    for col in csv:
        if isinstance(csv[col].dtype, pd.DatetimeTZDtype):
            csv[col] = csv[col].map(lambda t: "" if pd.isna(t) else t.isoformat())
    csv.to_csv(path.with_suffix(".csv"), index=False)


def summarize(frame: pd.DataFrame, platforms: pd.DataFrame) -> dict:
    n = len(frame)
    near = frame.invalid_position_immediately_before_pair | frame.invalid_position_immediately_after_pair
    return {
        "total_flagged_jumps": n, "affected_drifters": int((platforms.n_flagged_jumps > 0).sum()),
        "strict_duration_counts": duration_counts(frame.dt_minutes, 0),
        "duration_counts_with_1s_comparison_tolerance": duration_counts(frame.dt_minutes, CADENCE_TOLERANCE_SECONDS),
        "flags_spanning_missing_positions": int((frame.n_missing_positions_between > 0).sum()),
        "flags_spanning_invalid_positions": int((frame.n_invalid_positions_between > 0).sum()),
        "flags_between_directly_adjacent_valid_qc_rows": int(((frame.n_rows_between == 0) & frame.position_1_valid & frame.position_2_valid).sum()),
        "flags_with_nonadjacent_original_qc_source_rows": int((frame.n_source_rows_between != 0).sum()),
        "flags_with_invalid_position_immediately_outside_pair": int(near.sum()),
        "flags_at_nominal_5_minutes_within_1s": int(((frame.dt_minutes * 60 - 300).abs() <= 1).sum()),
        "flagged_dt_rounded_to_minutes_for_display": {str(int(k)): int(v) for k, v in frame.dt_minutes.round().value_counts().sort_index().items()},
        "speed_quantiles_m_s": quantiles(frame.speed_m_s), "distance_quantiles_m": quantiles(frame.distance_m),
        "duration_quantiles_minutes": quantiles(frame.dt_minutes),
        "top_1_flag_share_percent": float(100 * platforms.iloc[0].n_flagged_jumps / n) if n else 0.,
        "top_10_flag_share_percent": float(100 * platforms.head(10).n_flagged_jumps.sum() / n) if n else 0.,
        "native_observations_checked": int(platforms.n_native_observations.sum()),
        "eligible_adjacent_pairs_checked": int(platforms.n_eligible_adjacent_pairs.sum()),
        "blocked_pairs_checked": int(platforms.n_pairs_touching_invalid_position_or_time.sum()),
        "finite_speeds_on_blocked_pairs": int(platforms.n_finite_speeds_on_blocked_pairs.sum()),
        "all_speeds_and_flags_reproduced_exactly": True,
        "calculation_bridges_missing_observations": False,
    }


def write_report(summary: dict, platforms: pd.DataFrame, worst: pd.DataFrame, output: Path) -> None:
    counts = summary["duration_counts_with_1s_comparison_tolerance"]
    lines = ["# ARCTERX residual native-QC jump flags", "", "All times UTC. Diagnostic only; no QC or validity changes.", "",
             f"**{summary['total_flagged_jumps']:,} flags across {summary['affected_drifters']} drifters** reproduce the stored audit exactly.", "",
             "## What the calculation does", "",
             "`drifterlab.qc.flags.consecutive_speed` uses immediately adjacent chronological native rows, finite in-range positions, valid times, "
             "and positive actual elapsed time. It divides spherical haversine distance (radius 6,371,008.8 m) by that elapsed time, and assigns the speed "
             "and strict `> 3 m/s` flag to the later row. It neither compresses the series to valid fixes nor substitutes a nominal five-minute denominator. "
             "It has no maximum-gap exclusion, so a long interval could in principle be flagged; the observed intervals are reported below.", "",
             f"All {summary['native_observations_checked']:,} native rows were checked. Recomputed speeds (including NaNs) and flags match every stored value exactly. "
             f"Of {summary['blocked_pairs_checked']:,} adjacent pairs touching an invalid position or time, {summary['finite_speeds_on_blocked_pairs']} have a finite audit speed. "
             "Thus the implementation does **not bridge missing/QC-invalid observations**.", "",
             "## Adjacency and elapsed time", "",
             "| Requested count | Strict timestamp comparison | With 1-second boundary tolerance |", "|---|---:|---:|"]
    for name, count in counts.items():
        lines.append(f"| {name} | {summary['strict_duration_counts'][name]} | {count} |")
    lines += ["", "Tolerance is used only to summarize cadence boundaries. MATLAB fractional-day encoding contributes a few microseconds around whole minutes; "
              "all saved times, durations, speeds, and the 3 m/s threshold remain unchanged. The strict >10-minute cases must be read together with the maximum interval below.", "",
              f"- Directly adjacent valid QC pairs: **{summary['flags_between_directly_adjacent_valid_qc_rows']}**.",
              f"- Pairs spanning missing positions: **{summary['flags_spanning_missing_positions']}**; spanning any invalid positions: **{summary['flags_spanning_invalid_positions']}**.",
              f"- Pairs nonadjacent in the original delivered QC vector: **{summary['flags_with_nonadjacent_original_qc_source_rows']}**.",
              f"- Pairs with an invalid position immediately *outside* the pair: **{summary['flags_with_invalid_position_immediately_outside_pair']}**. Nearby QC gaps are distinct from bridging a gap.",
              f"- Nominal five-minute pairs (within one second): **{summary['flags_at_nominal_5_minutes_within_1s']}**.", "",
              "| Rounded interval for display (min) | Flagged pairs |", "|---:|---:|"]
    lines += [f"| {minutes} | {count} |" for minutes, count in summary["flagged_dt_rounded_to_minutes_for_display"].items()]
    lines += ["", "Shorter actual intervals increase speed for the same displacement; this is a consequence of the supplied timestamps. "
              "The flags should not all be described as five-minute jumps. No alternative denominator or replacement QC flag is introduced.", "",
              "## Magnitudes", "", "| Quantile | Speed (m/s) | Distance (m) | Interval (min) |", "|---|---:|---:|---:|"]
    for key in summary["speed_quantiles_m_s"]:
        values = [summary[name][key] for name in ("speed_quantiles_m_s", "distance_quantiles_m", "duration_quantiles_minutes")]
        lines.append(f"| {key} | " + " | ".join("missing" if v is None else f"{v:.9f}" for v in values) + " |")
    lines += ["", "## Concentration", "",
              f"The highest-count drifter contributes {summary['top_1_flag_share_percent']:.2f}% of flags; the top ten contribute {summary['top_10_flag_share_percent']:.2f}%. "
              "Flags are distributed across the affected drifters rather than attributable solely to the most extreme examples. Rates below account for unequal trajectory lengths.", "",
              "| Platform | Flags | Per 10,000 eligible adjacent pairs | Maximum speed (m/s) |", "|---|---:|---:|---:|"]
    for row in platforms.head(10).itertuples():
        lines.append(f"| {row.platform_code} | {row.n_flagged_jumps} | {row.flags_per_10000_eligible_pairs:.2f} | {row.max_flagged_speed_m_s:.3f} |")
    lines += ["", "## Supplied speed and source verification", "",
              "`drifter.speed` is preserved as `speed_qc_source`, in m/s. Its differentiation stencil, time denominator, filtering, "
              "and endpoint assignment are not established by the available metadata. The supplied data guide describes speed QC and the processing PDF discusses smoothing, "
              "but neither establishes an equivalent adjacent-fix definition. Both endpoint values are included with `definition_unverified` in their names as source context only. "
              "They are not used as an alternative flag, reference truth, error score, or direct speed comparison.", "",
              f"Flagged-endpoint verification against original QC MATLAB files: **{summary['source_verification']['status']}**; "
              f"{summary['source_verification']['files_checked']} files and {summary['source_verification']['unique_endpoints_checked']} unique endpoints checked. "
              "Where enabled, source hashes, source row indices, positions, timestamps and supplied speed values are matched exactly.", "",
              "## Worst 20 individual jumps (descending audit speed)", "",
              "| Platform | Later UTC (display seconds) | Native rows | Interval (min) | Distance (km) | Speed (m/s) |", "|---|---|---|---:|---:|---:|"]
    for row in worst.itertuples():
        lines.append(f"| {row.platform_code} | {row.time_2.round('s').strftime('%Y-%m-%d %H:%M:%S')} | {row.row_index_1} -> {row.row_index_2} | {row.dt_minutes:.3f} | {row.distance_m / 1000:.3f} | {row.speed_m_s:.3f} |")
    lines += ["", "## Cases for manual inspection", ""]
    for case in summary["selected_cases"]:
        lines.append(f"- **{case['platform_code']}**, {case['time_2_utc']}: {case['reason'].lower()}, "
                     f"{case['distance_m'] / 1000:.3f} km in {case['dt_minutes']:.3f} minutes ({case['speed_m_s']:.3f} m/s).")
    lines += ["", "The worst-20 table is an additional inspection list; multiple flagged legs can belong to one excursion and are not independent incidents. "
              "Inspect coordinates and neighboring masked points together. These diagnostics alone do not establish which endpoint is wrong or the true physical velocity.", "",
              "The case figure shows native positions and audit speed in a 90-minute window. Lines break at invalid fixes. "
              "Gray marks on the speed panels locate invalid positions; red segments/points are existing flags. "
              "The local east/north projection is only for display, not the speed calculation.", "",
              "## Conclusion", "",
              f"The {summary['total_flagged_jumps']:,} flags are adjacent-position differences in the delivered QC data, not an artifact of bridging missing observations. "
              "Their speed magnitudes reflect actual elapsed times, including short irregular intervals. "
              "They warrant inspection as residual coordinate/time anomalies; they are not proof of real drifter motion at those speeds.", "",
              "Outputs: `flagged_jumps.csv` / `.parquet`, `flags_by_platform.csv`, `worst_20_jumps.csv`, "
              "`selected_case_observations.csv`, `summary.json`, and two figures (PNG/PDF). "
              "Row indices are zero-based chronological native QC rows including invalid observations, not indices in a filtered-valid subset. "
              "Source indices refer to the original delivered MATLAB QC vectors; adjacency cannot establish whether upstream QC deleted rows from raw data."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diagnostics(frame: pd.DataFrame, context: pd.DataFrame, cases: list[dict], output: Path) -> None:
    if frame.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rc = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
          "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(frame.dt_minutes, frame.speed_m_s, s=18, alpha=.4, color="#0072B2", edgecolors="none")
        ax.axhline(THRESHOLD_M_S, color="#D55E00", ls="--", lw=1, label="Existing strict >3 m/s threshold")
        ax.axvline(5, color="#888888", ls=":", lw=1, label="Nominal 5-minute cadence")
        for case in cases:
            ax.annotate(case["case_id"] + " · …" + case["platform_code"][-6:],
                        (case["dt_minutes"], case["speed_m_s"]), xytext=(7, 7), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Actual interval between flagged QC fixes (minutes)", ylabel="Audit speed (m/s; log scale)", yscale="log",
               title=f"ARCTERX MicroSVP: {len(frame):,} existing residual jump flags")
        ax.margins(x=.1, y=.2)
        ax.grid(alpha=.2)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.text(.12, .025, "Every point represents two adjacent stored QC rows. Cases A–D are shown in the trajectory/time-series figure.", fontsize=8)
        fig.subplots_adjust(left=.12, right=.97, top=.89, bottom=.17)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"01_speed_vs_interval.{extension}", dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(len(cases), 2, figsize=(12, 3.5 * len(cases)), squeeze=False)
        for row_axes, case in zip(axes, cases):
            data = context[context.case_id == case["case_id"]].copy()
            good = data.position_valid_qc.to_numpy()
            origin = data[good].iloc[0]
            east = flags.EARTH_RADIUS_M * np.cos(np.radians(origin.lat_qc)) * np.radians(data.lon_qc.to_numpy() - origin.lon_qc) / 1000
            north = flags.EARTH_RADIUS_M * np.radians(data.lat_qc.to_numpy() - origin.lat_qc) / 1000
            east[~good], north[~good] = np.nan, np.nan
            ax, speed_ax = row_axes
            ax.plot(east, north, "o-", color="#555555", ms=3, lw=.8)
            for k in np.flatnonzero(data.residual_jump_flag_qc.to_numpy()):
                if k:
                    ax.plot(east[k - 1:k + 1], north[k - 1:k + 1], "o-", color="#D55E00", ms=4, lw=1.4)
            target = np.flatnonzero(data.row_index.to_numpy() == case["row_index_2"])[0]
            for k in (target - 1, target):
                ax.annotate(str(data.iloc[k].row_index), (east[k], north[k]), xytext=(5, 5), textcoords="offset points", fontsize=8)
            ax.set(xlabel="Local east (km)", ylabel="Local north (km)", aspect="equal")
            # Square limits keep nearly meridional tracks legible at equal scale.
            span = 1.25 * max(np.ptp(east[good]), np.ptp(north[good]), .1)
            cx = (np.nanmin(east) + np.nanmax(east)) / 2
            cy = (np.nanmin(north) + np.nanmax(north)) / 2
            ax.set_xlim(cx - span / 2, cx + span / 2)
            ax.set_ylim(cy - span / 2, cy + span / 2)
            ax.set_title(f"{case['case_id']} · {case['platform_code']}\n{case['reason']}", loc="left", fontsize=10)
            speed_ax.plot(data.time, data.audit_speed_qc_m_s, "o-", color="#555555", ms=3, lw=.8)
            flagged = data[data.residual_jump_flag_qc]
            speed_ax.scatter(flagged.time, flagged.audit_speed_qc_m_s, color="#D55E00", s=25, zorder=4)
            speed_ax.axhline(THRESHOLD_M_S, color="#D55E00", ls="--", lw=.8)
            for t in data.loc[~data.position_valid_qc, "time"]:
                speed_ax.axvline(t, color="#aaaaaa", lw=1.5, alpha=.7)
            speed_ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
            speed_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            speed_ax.set(xlabel="UTC time on " + data.time.iloc[0].strftime("%Y-%m-%d"), ylabel="Adjacent-fix audit speed (m/s)")
            speed_ax.set_title(f"Selected pair: {case['distance_m'] / 1000:.2f} km / {case['dt_minutes']:.2f} min = {case['speed_m_s']:.2f} m/s", loc="left", fontsize=10)
            for panel in row_axes:
                panel.grid(alpha=.2)
        fig.suptitle("Residual-jump inspection: native QC positions and actual adjacent-fix speeds", x=.08, ha="left", fontsize=14, y=.99)
        handles = [Line2D([], [], color="#555555", marker="o", ms=3, label="Native QC; lines break at invalid fixes"),
                   Line2D([], [], color="#D55E00", marker="o", ms=3, label="Existing >3 m/s pair"),
                   Line2D([], [], color="#aaaaaa", lw=2, label="Invalid position (time panel)")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.075, .967), ncol=3, frameon=False, fontsize=9)
        fig.text(.08, .012, "Map annotations are zero-based native row indices. Nearby masked fixes are retained in the context and are not bridged.\nSupplied speed is not plotted: its differentiation/filtering definition is unverified. Local projection is for display only.", fontsize=8)
        fig.subplots_adjust(left=.08, right=.975, bottom=.075, top=.92, hspace=.62, wspace=.3)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"02_selected_cases.{extension}", dpi=300)
        plt.close(fig)


def run(master: Path, output: Path, *, qc_directory: Path | None = None, figures: bool = True) -> dict:
    master, output = master.resolve(), output.resolve()
    protected = [master] + ([qc_directory.resolve()] if qc_directory else [])
    if any(output == path or path in output.parents for path in protected):
        raise ValueError("Diagnostic output must be outside the master and QC source directories")
    tables, stats = [], []
    source_check = {"status": "not requested", "files_checked": 0, "unique_endpoints_checked": 0,
                    "qc_directory": str(qc_directory.resolve()) if qc_directory else None}
    with xr.open_zarr(master, chunks=None) as ds:
        if ds.attrs.get("residual_jump_speed_m_s") != THRESHOLD_M_S or ds.residual_jump_flag_qc.attrs.get("threshold_m_s") != THRESHOLD_M_S:
            raise ValueError("Expected the existing strict 3 m/s audit threshold in master metadata")
        platforms = ds.platform_code.values.astype(str)
        if not len(platforms) or len(set(platforms)) != len(platforms):
            raise ValueError("Master must contain unique platform IDs")
        for i, platform in enumerate(platforms):
            track = load_track(ds, i)
            frame, detail = flagged_pairs(track)
            if int(ds.n_residual_jumps_qc.values[i]) != len(frame):
                raise ValueError(f"Stored per-platform flag count mismatch for {platform}")
            if qc_directory and len(frame):
                source_check["unique_endpoints_checked"] += verify_source(track, qc_directory)
                source_check["files_checked"] += 1
            tables.append(frame)
            stats.append(detail)
        frame = pd.concat(tables, ignore_index=True)
        by_platform = pd.DataFrame(stats).sort_values(["n_flagged_jumps", "platform_code"], ascending=[False, True]).reset_index(drop=True)
        cases = select_cases(frame, by_platform)
        context = case_context(ds, cases)
    if qc_directory:
        source_check["status"] = "passed"
    summary = summarize(frame, by_platform)
    summary.update({"threshold_m_s": THRESHOLD_M_S, "time_reference": "UTC", "software_version": __version__,
                    "processed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "master_zarr": str(master),
                    "master_consolidated_metadata_sha256": sha256((master / ".zmetadata").read_bytes()).hexdigest(),
                    "audit_implementation_sha256": sha256(Path(flags.__file__).read_bytes()).hexdigest(),
                    "diagnostic_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                    "source_verification": source_check, "selected_cases": cases,
                    "supplied_speed_comparison": "Definition unverified; endpoint values preserved only as context, no numerical equivalence assumed"})
    worst = frame.sort_values(["speed_m_s", "platform_code", "row_index_2"], ascending=[False, True, True]).head(20)
    output.mkdir(parents=True, exist_ok=True)
    write_table(frame, output / "flagged_jumps", parquet=True)
    write_table(by_platform, output / "flags_by_platform")
    write_table(worst, output / "worst_20_jumps")
    write_table(context, output / "selected_case_observations")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, by_platform, worst, output)
    if figures:
        plot_diagnostics(frame, context, cases, output)
    print(json.dumps(summary, indent=2))
    print(f"Residual-jump diagnostic: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("data/ARCTERX_MicroSVP_QC.zarr"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/diagnostics/residual_jumps"))
    parser.add_argument("--qc-directory", type=Path, help="Optional original QC MATLAB directory for exact flagged-endpoint verification")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    run(args.master, args.output_directory, qc_directory=args.qc_directory, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
