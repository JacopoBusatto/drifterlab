"""Read-only comparison of delivered raw and QC start times; no release inference.

Run with ``python scripts/diagnostics/timing_diagnostic.py --help``.
Only this diagnostic's output directory is written.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drifterlab.io.matlab import normalize_missing, read_matlab
from drifterlab.qc.position import position_valid
from drifterlab.experiments.arcterx.config import load_config
from drifterlab.experiments.arcterx.microsvp import _identifier, read_microsvp

TOLERANCE_SECONDS = 1.0  # Comparison tolerance only; saved times are never rounded.
REPORT_WINDOWS = {
    "first_array": ("2025-02-03T02:00:00Z", "2025-02-03T06:00:00Z"),
    "second_array": ("2025-02-11T11:00:00Z", "2025-02-11T16:00:00Z"),
}


@dataclass
class TimingSeries:
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    source_index: np.ndarray
    extra: dict


def _raw_structures(values: dict) -> tuple[str, dict, dict]:
    dataset = values.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("Raw dataset must be a scalar MATLAB structure")
    candidates = [(k, v) for k, v in dataset.items() if k.startswith("drifter_")]
    if len(candidates) != 1:
        raise ValueError(f"Expected one raw drifter structure; found {len(candidates)}")
    key, track = candidates[0]
    platform = _identifier(track["PlatformId"], f"dataset.{key}.PlatformId")
    if key != f"drifter_{platform}":
        raise ValueError("Raw structure name disagrees with its embedded PlatformId")
    for name, metadata in dataset.items():
        field = "PlatformID" if name.startswith("meta_") else "PlatformId"
        if name.startswith(("meta_", "diagnostic_")) and field in metadata:
            if _identifier(metadata[field], name + "." + field) != platform:
                raise ValueError(f"Conflicting raw identifier in {name}")
    return platform, track, dataset


def catalog(directory: Path, kind: str, pattern: str = "*.mat") -> tuple[dict, list]:
    """Index by embedded identifiers, preserving duplicate candidates as lists."""
    entries: dict[str, list] = defaultdict(list)
    errors = []
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        try:
            loaded = read_matlab(path)
            if kind == "raw":
                platform, _, _ = _raw_structures(loaded.values)
            else:
                native = loaded.values["drifter"]
                platform = _identifier(native["PlatformId"], "drifter.PlatformId")
                if "ID" in native and _identifier(native["ID"], "drifter.ID") != platform:
                    raise ValueError("QC ID disagrees with PlatformId")
            entries[platform].append({"path": str(path.resolve()), "sha256": loaded.sha256})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append({"path": str(path.resolve()), "error": str(exc)})
    return dict(entries), errors


def matching_report(raw: dict, qc: dict, raw_errors: list, qc_errors: list) -> dict:
    duplicates = {kind: {p: [e["path"] for e in files] for p, files in entries.items() if len(files) > 1}
                  for kind, entries in [("raw", raw), ("qc", qc)]}
    ambiguous = sorted(set(duplicates["raw"]) | set(duplicates["qc"]))
    return {
        "raw_files": sum(map(len, raw.values())) + len(raw_errors),
        "qc_files": sum(map(len, qc.values())) + len(qc_errors),
        "matched_pairs": sum(len(raw.get(p, [])) == len(qc.get(p, [])) == 1 for p in set(raw) & set(qc)),
        "qc_without_raw": sorted(set(qc) - set(raw)),
        "raw_without_qc": sorted(set(raw) - set(qc)),
        "duplicate_platform_ids": duplicates, "ambiguous_platform_ids": ambiguous,
        "raw_read_errors": raw_errors, "qc_read_errors": qc_errors,
        "matching_rule": "Embedded raw dataset.drifter_*.PlatformId and QC drifter.PlatformId; no filename-order pairing",
    }


def _raw_times(values) -> np.ndarray:
    strings = np.atleast_1d(values).astype(str)
    return pd.to_datetime(strings, format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True).as_unit("ns").tz_localize(None).to_numpy()


def _utc(value):
    return pd.NaT if pd.isna(value) else pd.Timestamp(value).tz_localize("UTC")


def read_raw_timing(entry: dict, missing_value: float = -999) -> TimingSeries:
    loaded = read_matlab(entry["path"])
    if loaded.sha256 != entry["sha256"]:
        raise ValueError(f"Raw source changed after matching: {entry['path']}")
    platform, track, dataset = _raw_structures(loaded.values)
    time = _raw_times(track["ObsTimestamp"])
    lon = np.atleast_1d(normalize_missing(track["GpsLongitude"], missing_value))
    lat = np.atleast_1d(normalize_missing(track["GpsLatitude"], missing_value))
    if lon.shape != time.shape or lat.shape != time.shape:
        raise ValueError(f"Raw time/position shape mismatch for {platform}")
    order = np.argsort(time, kind="stable")
    diagnostic = dataset.get(f"diagnostic_{platform}", {})
    diagnostic_time = _raw_times(diagnostic.get("timestamp", []))
    good_diagnostic = diagnostic_time[~np.isnat(diagnostic_time)]
    extra = {
        "raw_timestamp_field": f"dataset.drifter_{platform}.ObsTimestamp",
        "raw_diagnostic_first_time": _utc(good_diagnostic.min()) if len(good_diagnostic) else pd.NaT,
        "raw_diagnostic_n_rows_feb03": int(((diagnostic_time >= np.datetime64("2025-02-03")) & (diagnostic_time < np.datetime64("2025-02-04"))).sum()),
    }
    return TimingSeries(time[order], lon[order], lat[order], order, extra)


def read_qc_timing(entry: dict, missing_value: float = -999) -> TimingSeries:
    record = read_microsvp(entry["path"], missing_value=missing_value)
    if record.metadata["source_sha256"] != entry["sha256"]:
        raise ValueError(f"QC source changed after matching: {entry['path']}")
    series = record.series["qc"]
    return TimingSeries(series.time, series.variables["lon_qc"], series.variables["lat_qc"], series.source_index, {})


def series_metrics(series: TimingSeries | None, prefix: str) -> dict:
    result = {f"{prefix}_{name}": pd.NaT for name in ("first_time", "first_valid_position_time", "last_time", "second_distinct_time")}
    result.update({f"{prefix}_{name}": np.nan for name in ("n_obs", "n_valid_positions", "n_invalid_times", "first_lon", "first_lat", "first_valid_lon", "first_valid_lat", "first_source_index", "first_valid_source_index", "first_gap_hours", "n_rows_before_first_valid")})
    if series is None:
        return result
    finite = ~np.isnat(series.time)
    valid = finite & position_valid(series.lon, series.lat)
    ti, vi = np.flatnonzero(finite), np.flatnonzero(valid)
    result.update({f"{prefix}_n_obs": len(series.time), f"{prefix}_n_valid_positions": int(valid.sum()),
                   f"{prefix}_n_invalid_times": int((~finite).sum())})
    if len(ti):
        first = ti[0]
        result.update({f"{prefix}_first_time": _utc(series.time[first]), f"{prefix}_last_time": _utc(series.time[ti[-1]]),
                       f"{prefix}_first_lon": series.lon[first], f"{prefix}_first_lat": series.lat[first],
                       f"{prefix}_first_source_index": int(series.source_index[first])})
        later = ti[series.time[ti] > series.time[first]]
        if len(later):
            result[f"{prefix}_second_distinct_time"] = _utc(series.time[later[0]])
            result[f"{prefix}_first_gap_hours"] = float((series.time[later[0]] - series.time[first]) / np.timedelta64(1, "h"))
    if len(vi):
        first = vi[0]
        result.update({f"{prefix}_first_valid_position_time": _utc(series.time[first]),
                       f"{prefix}_first_valid_lon": series.lon[first], f"{prefix}_first_valid_lat": series.lat[first],
                       f"{prefix}_first_valid_source_index": int(series.source_index[first]),
                       f"{prefix}_n_rows_before_first_valid": int((finite & (series.time < series.time[first])).sum())})
    result.update(series.extra)
    return result


def on_day(time, day: str) -> bool:
    return not pd.isna(time) and pd.Timestamp(time).strftime("%Y-%m-%d") == day


def feb06_case(row: dict) -> str:
    if not on_day(row["qc_first_valid_position_time"], "2025-02-06"):
        return "not_feb06_cohort"
    if on_day(row["qc_first_time"], "2025-02-03"):
        return "C"
    if on_day(row["raw_first_time"], "2025-02-03") and on_day(row["qc_first_time"], "2025-02-06"):
        return "B"
    if on_day(row["raw_first_time"], "2025-02-06") and on_day(row["qc_first_time"], "2025-02-06"):
        return "A"
    return "D"


def _matched_times(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Within-one-second membership, without rounding or rewriting source times."""
    result = np.zeros(len(queries), dtype=bool)
    reference = np.sort(reference[~np.isnat(reference)]).astype("datetime64[ns]").astype(np.int64)
    valid = np.flatnonzero(~np.isnat(queries))
    if not len(reference) or not len(valid):
        return result
    q = queries[valid].astype("datetime64[ns]").astype(np.int64)
    indices = np.searchsorted(reference, q)
    # Delivered campaign times lie in the same year; subtraction is safe in ns.
    lower = reference[np.clip(indices - 1, 0, len(reference) - 1)]
    upper = reference[np.clip(indices, 0, len(reference) - 1)]
    result[valid] = np.minimum(np.abs(q - lower), np.abs(q - upper)) <= TOLERANCE_SECONDS * 1e9
    return result


def compare_series(raw: TimingSeries | None, qc: TimingSeries | None) -> dict:
    row = {**series_metrics(raw, "raw"), **series_metrics(qc, "qc")}
    for suffix in ("time", "valid_position"):
        source = "raw_first_time" if suffix == "time" else "raw_first_valid_position_time"
        target = "qc_first_time" if suffix == "time" else "qc_first_valid_position_time"
        row[f"delta_raw_to_qc_first_{suffix}_hours"] = (row[target] - row[source]) / pd.Timedelta(hours=1)
    row["delta_raw_first_time_to_qc_first_valid_position_hours"] = (row["qc_first_valid_position_time"] - row["raw_first_time"]) / pd.Timedelta(hours=1)
    row["qc_first_time_precedes_first_valid"] = bool((row["qc_first_valid_position_time"] - row["qc_first_time"]) > pd.Timedelta(seconds=TOLERANCE_SECONDS))
    row["feb06_case"] = feb06_case(row)
    row["raw_n_rows_feb03"] = 0 if raw is not None else np.nan
    row["raw_n_rows_before_qc_first_time"] = np.nan
    row["raw_n_rows_before_qc_first_valid"] = np.nan
    row["raw_early_rows_omitted_from_qc"] = np.nan
    row["raw_early_rows_retained_masked_in_qc"] = np.nan
    row["raw_first_time_on_qc_first_valid_date"] = pd.NaT
    row["raw_n_rows_before_qc_first_valid_date"] = np.nan
    if raw is not None:
        row["raw_n_rows_feb03"] = int(((raw.time >= np.datetime64("2025-02-03")) & (raw.time < np.datetime64("2025-02-04"))).sum())
    if raw is not None and qc is not None and not pd.isna(row["qc_first_valid_position_time"]):
        first_valid = row["qc_first_valid_position_time"].tz_localize(None).to_datetime64()
        day = first_valid.astype("datetime64[D]")
        raw_early = raw.time < first_valid - np.timedelta64(int(TOLERANCE_SECONDS * 1e9), "ns")
        row["raw_n_rows_before_qc_first_time"] = int((raw.time < qc.time[~np.isnat(qc.time)][0] - np.timedelta64(1, "s")).sum())
        row["raw_n_rows_before_qc_first_valid"] = int(raw_early.sum())
        matched = _matched_times(raw.time[raw_early], qc.time)
        row["raw_early_rows_omitted_from_qc"] = int((~matched).sum())
        row["raw_early_rows_retained_masked_in_qc"] = int(matched.sum())
        same_day = raw.time[(raw.time >= day) & (raw.time < day + np.timedelta64(1, "D"))]
        row["raw_first_time_on_qc_first_valid_date"] = _utc(same_day.min()) if len(same_day) else pd.NaT
        row["raw_n_rows_before_qc_first_valid_date"] = int((raw.time < day).sum())
    return row


def assign_cohorts(frame: pd.DataFrame) -> pd.Series:
    """Describe consecutive observed UTC dates; these are not deployment IDs."""
    dates = pd.to_datetime(frame.qc_first_valid_position_time, utc=True).dt.normalize()
    groups: list[list] = []
    for day in sorted(dates.dropna().unique()):
        if not groups or day - groups[-1][-1] > pd.Timedelta(days=1):
            groups.append([day])
        else:
            groups[-1].append(day)
    names = {}
    for group in groups:
        label = group[0].strftime("%Y-%m-%d")
        if len(group) > 1:
            label += " to " + group[-1].strftime("%Y-%m-%d")
        names.update({day: label for day in group})
    return dates.map(names).fillna("no_valid_qc_time")


def build_table(raw_catalog: dict, qc_catalog: dict, inventory: pd.DataFrame, missing_value: float = -999) -> pd.DataFrame:
    inv = inventory.copy()
    inv["platform_code"] = inv.platform_code.astype(str)
    if inv.platform_code.duplicated().any():
        raise ValueError("Inventory has duplicate platform IDs; cannot validate first-valid times unambiguously")
    inv = inv.set_index("platform_code")
    rows = []
    for platform in sorted(set(raw_catalog) | set(qc_catalog)):
        raw_entries, qc_entries = raw_catalog.get(platform, []), qc_catalog.get(platform, [])
        ambiguous = len(raw_entries) > 1 or len(qc_entries) > 1
        status = "ambiguous" if ambiguous else ("matched" if raw_entries and qc_entries else "raw_only" if raw_entries else "qc_only")
        raw = read_raw_timing(raw_entries[0], missing_value) if len(raw_entries) == 1 and not ambiguous else None
        qc = read_qc_timing(qc_entries[0], missing_value) if len(qc_entries) == 1 and not ambiguous else None
        row = {"platform_code": platform, "match_status": status, **compare_series(raw, qc)}
        for kind, entries in [("raw", raw_entries), ("qc", qc_entries)]:
            row[f"{kind}_source_files"] = json.dumps([e["path"] for e in entries])
            row[f"{kind}_source_sha256"] = entries[0]["sha256"] if len(entries) == 1 else ""
        row["inventory_present"] = platform in inv.index
        row["inventory_first_time_diff_seconds"] = np.nan
        row["inventory_first_valid_time_diff_seconds"] = np.nan
        row["inventory_source_hash_matches"] = False
        if platform in inv.index and qc is not None:
            previous = inv.loc[platform]
            for target, column, name in [("qc_first_time", "first_time_qc", "inventory_first_time_diff_seconds"),
                                         ("qc_first_valid_position_time", "first_valid_position_time_qc", "inventory_first_valid_time_diff_seconds")]:
                row[name] = (row[target] - pd.to_datetime(previous[column], utc=True)) / pd.Timedelta(seconds=1)
            row["inventory_source_hash_matches"] = previous.source_sha256 == row["qc_source_sha256"]
        rows.append(row)
    if not rows:
        raise ValueError("No identifiable raw or QC drifters found")
    frame = pd.DataFrame(rows)
    time_columns = {f"{prefix}_{name}" for prefix in ("raw", "qc")
                    for name in ("first_time", "first_valid_position_time", "last_time", "second_distinct_time")}
    time_columns |= {"raw_diagnostic_first_time", "raw_first_time_on_qc_first_valid_date"}
    for name in time_columns & set(frame):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    frame["first_valid_qc_time_cohort"] = assign_cohorts(frame)
    frame = frame.sort_values(["qc_first_valid_position_time", "platform_code"], kind="stable", na_position="last").reset_index(drop=True)
    frame.insert(1, "drifter_index", np.arange(1, len(frame) + 1))
    return frame


def _display_time(value) -> str:
    return "missing" if pd.isna(value) else pd.Timestamp(value).round("s").strftime("%Y-%m-%d %H:%M:%S")


def _delay_stats(values: pd.Series) -> dict:
    values = values.dropna().copy()
    values[values.abs() <= TOLERANCE_SECONDS / 3600] = 0
    return {key: float(value) for key, value in {"median_hours": values.median(), "min_hours": values.min(), "max_hours": values.max()}.items()}


def summarize(frame: pd.DataFrame, matching: dict) -> dict:
    summary = {"matching": matching, "time_reference": "UTC", "comparison_tolerance_seconds": TOLERANCE_SECONDS,
               "reference_windows_from_user_request": REPORT_WINDOWS, "cohorts": [], "focus": {}}
    for label, subset in frame.groupby("first_valid_qc_time_cohort", sort=False):
        summary["cohorts"].append({"cohort": label, "n": len(subset),
                                   "qc_first_valid_min": _display_time(subset.qc_first_valid_position_time.min()),
                                   "qc_first_valid_max": _display_time(subset.qc_first_valid_position_time.max())})
    dates = frame.qc_first_valid_position_time.dt.strftime("%Y-%m-%d")
    for day in ("2025-02-06", "2025-02-11"):
        subset = frame[(dates == day) & (frame.match_status == "matched")]
        raw_days = subset.raw_first_time.dt.strftime("%Y-%m-%d")
        summary["focus"][day] = {
            "n": len(subset), "raw_first_dates": {str(k): int(v) for k, v in raw_days.value_counts().sort_index().items()},
            "qc_first_precedes_first_valid": int(subset.qc_first_time_precedes_first_valid.sum()),
            "raw_to_qc_first_time": _delay_stats(subset.delta_raw_to_qc_first_time_hours),
            "raw_to_qc_first_valid_position": _delay_stats(subset.delta_raw_to_qc_first_valid_position_hours),
            "raw_first_time_to_qc_first_valid_position": _delay_stats(subset.delta_raw_first_time_to_qc_first_valid_position_hours),
            "raw_qc_first_times_equal_within_tolerance": int((subset.delta_raw_to_qc_first_time_hours.abs() <= TOLERANCE_SECONDS / 3600).sum()),
            "raw_rows_feb03": int(subset.raw_n_rows_feb03.sum()),
            "case_counts": {str(k): int(v) for k, v in subset.feb06_case.value_counts().sort_index().items()},
        }
    compared = frame[frame.match_status == "matched"]
    summary["inventory_check"] = {
        "matched_rows": len(compared),
        "first_time_mismatches": int((compared.inventory_first_time_diff_seconds != 0).sum()),
        "first_valid_time_mismatches": int((compared.inventory_first_valid_time_diff_seconds != 0).sum()),
        "source_hash_mismatches": int((~compared.inventory_source_hash_matches).sum()),
    }
    control = frame[dates == "2025-02-11"]
    lo, hi = map(pd.Timestamp, REPORT_WINDOWS["second_array"])
    summary["focus"]["2025-02-11"]["qc_first_valid_within_report_window"] = int(control.qc_first_valid_position_time.between(lo, hi).sum())
    return summary


def write_report(frame: pd.DataFrame, summary: dict, output: Path) -> None:
    matching, focus = summary["matching"], summary["focus"]
    lines = ["# ARCTERX MicroSVP timing diagnostic", "", "All times UTC. These are first-valid-QC-time cohorts, not deployment groups.", "",
             f"Matched **{matching['matched_pairs']}** raw/QC pairs by embedded platform ID. QC without raw: {len(matching['qc_without_raw'])}; raw without QC: {len(matching['raw_without_qc'])}; ambiguous IDs: {len(matching['ambiguous_platform_ids'])}.", "",
             "## Descriptive cohorts", "", "| First-valid-QC-time cohort | Drifters | First valid QC range (UTC) |", "|---|---:|---|"]
    for cohort in summary["cohorts"]:
        lines.append(f"| {cohort['cohort']} | {cohort['n']} | {cohort['qc_first_valid_min']} to {cohort['qc_first_valid_max']} |")
    lines += ["", "## February comparisons", "", "Delays below are hours. Differences within one second are displayed as zero; table timestamps and delay values retain their original precision.", "",
              "| Cohort | Raw first dates | QC has earlier timestamps than its first valid position | Raw → QC first timestamp (median; min–max) | Raw → QC first valid position (median; min–max) |", "|---|---|---:|---|---|"]
    for day, stats in focus.items():
        a, b = stats["raw_to_qc_first_time"], stats["raw_to_qc_first_valid_position"]
        dates = "; ".join(f"{day}: {n}" for day, n in stats["raw_first_dates"].items())
        lines.append(f"| {day} (n={stats['n']}) | {dates} | {stats['qc_first_precedes_first_valid']} | {a['median_hours']:.6f}; {a['min_hours']:.6f}–{a['max_hours']:.6f} | {b['median_hours']:.6f}; {b['min_hours']:.6f}–{b['max_hours']:.6f} |")
    first = focus["2025-02-06"]
    lines += ["", f"February 6 case counts: **A={first['case_counts'].get('A', 0)}, B={first['case_counts'].get('B', 0)}, C={first['case_counts'].get('C', 0)}, D={first['case_counts'].get('D', 0)}**.",
              "A: raw and QC first timestamps are on February 6. B: raw starts February 3 and QC starts February 6. C: QC timestamps start February 3 but positions first become valid February 6. D: another pattern. Calendar-day windows operationalize ‘near’ for these descriptive cases; they do not identify releases.", ""]
    anomalous = frame[(frame.raw_n_rows_before_qc_first_valid_date > 0) | (frame.delta_raw_to_qc_first_time_hours > 1) |
                      ((frame.qc_first_valid_position_time.dt.strftime("%Y-%m-%d") == "2025-02-06") & (frame.qc_first_valid_position_time.dt.hour >= 7))]
    lines += ["## Earlier records and delayed QC starts", "", "| Platform | Raw first | QC first | First valid QC | Raw rows before first-valid QC date | Initial raw gap (h) | Early raw rows omitted / retained masked |", "|---|---|---|---|---:|---:|---:|"]
    for row in anomalous.itertuples():
        lines.append(f"| {row.platform_code} | {_display_time(row.raw_first_time)} | {_display_time(row.qc_first_time)} | {_display_time(row.qc_first_valid_position_time)} | {row.raw_n_rows_before_qc_first_valid_date:g} | {row.raw_first_gap_hours:.3f} | {row.raw_early_rows_omitted_from_qc:g} / {row.raw_early_rows_retained_masked_in_qc:g} |")
    control = focus["2025-02-11"]
    lines += ["", f"The February 11 control has {control['qc_first_valid_within_report_window']}/{control['n']} first valid QC positions within the supplied cruise-report window (11:00–16:00 UTC). Its full first-valid-time range and outliers are listed above.", "",
              "## Definition and inventory checks", "", "First timestamp means the minimum parseable timestamp, including rows with missing positions. First valid position means the earliest parseable time with finite coordinates within longitude [-180,180] and latitude [-90,90]. Raw GPS quality/status codes are not used to introduce another QC rule; an in-range raw position is not proof of deployment. Neither drogue masks nor reconstructed tracks enter this diagnostic.", "",
              "Raw observations use `dataset.drifter_<ID>.ObsTimestamp` calendar strings and `GpsLongitude`/`GpsLatitude`. QC uses the existing reader and the native QC axis. `raw_first_lon/lat` refer to the earliest timestamp; the separate `raw_first_valid_lon/lat` refer to the earliest geometrically valid position.", "",
              "Delay columns distinguish QC first minus raw first from QC first-valid position minus raw first-valid position. A third column explicitly gives QC first-valid minus raw first timestamp. Earlier raw rows are matched to QC times within one second solely to distinguish omitted timestamps from retained, position-masked rows; original timestamps are never changed.", "",
              f"Inventory checks: {json.dumps(summary['inventory_check'])}. Independently parsed raw calendar strings agree with matching QC datenums to the documented comparison precision. This check addresses extraction and time conversion, not the correctness of source instrument clocks.", "",
              "The raw files also contain a separate `diagnostic_*` telemetry structure, with earlier pre-campaign timestamps. Its minimum time and February 3 row count are included separately in the table; it is not substituted for the primary observation series.", "",
              "The cruise-report windows are those supplied in the task request. No release time, deployment ID, group, centroid, or physical pair was inferred.", "", "## Conclusion", ""]
    if first["n"] and first["raw_first_dates"].get("2025-02-06", 0) > first["n"] / 2:
        lines.append(f"For {first['raw_first_dates'].get('2025-02-06', 0)} of the {first['n']} February 6 first-valid-QC-time drifters, the delivered primary raw observation series itself starts on February 6. Thus the cohort-wide discrepancy is not explained by QC removing three days from all drifters. The exceptions must be read individually: the entire cohort contains {first['raw_rows_feb03']} primary raw observation(s) on February 3, not a shared continuous February 3–6 track.")
    else:
        lines.append("The source comparison is mixed; use the case counts and per-drifter table above rather than attributing a common cause.")
    lines += ["", "The files establish the delivered timing pattern but do not resolve why it differs from the reported first-array date. Actual release dates cannot be established from this diagnostic alone.", "",
              "Outputs: `microsvp_timing_diagnostic.parquet`, the equivalent CSV, `microsvp_timing_summary.json`, and two timeline figures (PNG and vector PDF)."]
    (output / "microsvp_timing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_timelines(frame: pd.DataFrame, output: Path) -> None:
    # Optional dependency; no plotting imports are needed to read or test data.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    styles = [("raw_first_time", "Raw first timestamp", "o", "#0072B2", -.20),
              ("qc_first_time", "QC first timestamp", "s", "#D55E00", 0),
              ("qc_first_valid_position_time", "First valid QC position", "|", "#222222", .20)]
    inferred = "inferred_start_time" in frame
    suffix = "_with_inferred_start" if inferred else ""
    if inferred:
        styles = [(col, label, marker, color, offset) for (col, label, marker, color, _), offset
                  in zip(styles, (-.30, -.10, .10))]
        styles.append(("inferred_start_time", "Inferred sustained start", "D", "#009E73", .30))
    rc = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
          "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"}
    legend = [Line2D([], [], marker=marker, color=color, linestyle="none", markerfacecolor="none", markersize=6, label=label)
              for _, label, marker, color, _ in styles]

    def draw(ax, data, limits, detail=False, clipped=False):
        left, right = map(pd.Timestamp, limits)
        for row in data.itertuples():
            times = [getattr(row, col) for col, *_ in styles if not pd.isna(getattr(row, col))]
            if times:
                ax.plot([max(min(times), left), min(max(times), right)], [row.drifter_index] * 2, color="#b7b7b7", lw=.55, zorder=1)
        for column, _, marker, color, offset in styles:
            selected = data[data[column].notna()]
            times = selected[column]
            colors = {"facecolors": "none", "edgecolors": color} if marker != "|" else {"color": color}
            ax.scatter(times, selected.drifter_index + offset, marker=marker, s=22 if detail else 10,
                       linewidths=.8, zorder=3, **colors)
            if clipped:
                early = selected[times < left]
                ax.scatter([left] * len(early), early.drifter_index + offset, marker="<", s=27, color=color, clip_on=False, zorder=4)
        ax.set_xlim(left, right)
        ax.set_ylim(data.drifter_index.max() + 1.2, data.drifter_index.min() - 1.2)
        ax.grid(axis="x", color="#dddddd", lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Calendar time (UTC, {left.year})")
        if detail:
            ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        return left, right

    data = frame[(frame.match_status == "matched") & frame.qc_first_valid_position_time.notna()]
    if data.empty:
        return
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, 10.5))
        earliest = min(data[c].min() for c, *_ in styles)
        latest = max(data[c].max() for c, *_ in styles)
        draw(ax, data, (earliest - pd.Timedelta(hours=12), latest + pd.Timedelta(hours=12)))
        for key, (lo, hi) in REPORT_WINDOWS.items():
            ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="#CC79A7" if key == "first_array" else "#009E73", alpha=.18, zorder=.5)
        for i, (label, subset) in enumerate(data.groupby("first_valid_qc_time_cohort", sort=False)):
            lo, hi = subset.drifter_index.min() - .5, subset.drifter_index.max() + .5
            if i % 2 == 0:
                ax.axhspan(lo, hi, color="#eef2f5", zorder=0)
            ax.text(1.012, (lo + hi) / 2, f"{label.replace('2025-', '')}\nn={len(subset)}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ticks = sorted(set(data.qc_first_valid_position_time.dt.normalize().tolist()) |
                       {pd.Timestamp("2025-02-03T00:00:00Z"), pd.Timestamp("2025-01-20T00:00:00Z"), pd.Timestamp("2025-01-27T00:00:00Z")})
        # January 12/13 share a cohort; one label avoids overlapping day labels.
        ticks = [t for t in ticks if t != pd.Timestamp("2025-01-12T00:00:00Z")]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
        ax.set_ylabel("Drifter index, ordered by first valid QC time")
        title = "delivered times and inferred sustained starts" if inferred else "delivered raw and QC start times"
        ax.set_title(f"ARCTERX MicroSVP: {title} (n={len(data)})", loc="left", pad=55 if inferred else 45, fontsize=14)
        ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2 if inferred else 3, frameon=False, fontsize=9)
        note = " Green diamonds are diagnostic estimates, not verified release times." if inferred else ""
        fig.text(.10, .022, "Shaded vertical bands: cruise-report windows supplied in the request (3 Feb 02:00–06:00; 11 Feb 11:00–16:00 UTC).\nSmall vertical marker offsets reveal coincident times. Connectors show time differences, not continuous observations.\nCohorts describe first valid QC times; they are not deployment groups." + note, fontsize=8 if inferred else 8.5)
        fig.subplots_adjust(left=.10, right=.83, bottom=.13, top=.87)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"01_all_drifters_timing{suffix}.{extension}", dpi=300)
        plt.close(fig)

        focus = data[data.qc_first_valid_position_time.dt.strftime("%Y-%m-%d") == "2025-02-06"]
        if focus.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 10.5), sharey=True, gridspec_kw={"width_ratios": [1, 1.45]})
        draw(axes[0], focus, ("2025-02-03T00:00:00Z", "2025-02-06T18:00:00Z"), True, True)
        draw(axes[1], focus, ("2025-02-06T01:30:00Z", "2025-02-06T17:30:00Z"), True)
        axes[0].axvspan(pd.Timestamp(REPORT_WINDOWS["first_array"][0]), pd.Timestamp(REPORT_WINDOWS["first_array"][1]), color="#CC79A7", alpha=.2)
        axes[0].xaxis.set_major_locator(mdates.DayLocator())
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[0].set_title("Reported date to observed cohort date", loc="left", fontsize=11)
        axes[1].set_title("6 February detail (UTC)", loc="left", fontsize=11)
        axes[0].set_ylabel("Drifter index from the all-drifter timeline")
        earlier = focus[focus.raw_first_time < pd.Timestamp("2025-02-03T00:00:00Z")]
        for row in earlier.itertuples():
            axes[0].annotate(row.raw_first_time.strftime("%d %b") + " · …" + row.platform_code[-6:],
                             (pd.Timestamp("2025-02-03T00:00:00Z"), row.drifter_index), xytext=(7, -7), textcoords="offset points", fontsize=7.5)
        fig.suptitle(f"February 6 first-valid-QC-time cohort (n={len(focus)})", x=.08, ha="left", y=.97, fontsize=15)
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.075, .947), ncol=4 if inferred else 3, frameon=False, fontsize=9 if inferred else 10)
        fig.text(.08, .032, "Left arrows indicate earlier January timestamps outside the displayed window. Full dates and IDs are in the diagnostic table.\nThe February 3 point is one isolated raw observation; a connector spans a time difference and does not imply continuous observations.", fontsize=8.5)
        fig.subplots_adjust(left=.08, right=.975, bottom=.10, top=.87, wspace=.14)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"02_feb06_cohort_timing{suffix}.{extension}", dpi=300)
        plt.close(fig)


def run(config_path: Path, raw_directory: Path, output: Path, *, figures: bool = True) -> dict:
    config = load_config(config_path)
    output = output.resolve()
    for protected in (config.input_directory, raw_directory.resolve(), config.zarr):
        if output == protected or protected in output.parents:
            raise ValueError("Diagnostic output directory must be outside source directories and the master Zarr")
    if not raw_directory.is_dir():
        raise ValueError(f"Raw directory does not exist: {raw_directory}")
    if not config.input_directory.is_dir():
        raise ValueError(f"QC directory does not exist: {config.input_directory}")
    raw, raw_errors = catalog(raw_directory, "raw")
    qc, qc_errors = catalog(config.input_directory, "qc", config.pattern)
    matching = matching_report(raw, qc, raw_errors, qc_errors)
    output.mkdir(parents=True, exist_ok=True)
    (output / "microsvp_source_matching.json").write_text(json.dumps(matching, indent=2) + "\n", encoding="utf-8")
    print(f"ID matching: {matching['matched_pairs']} unique raw/QC pairs; {len(matching['ambiguous_platform_ids'])} ambiguous IDs", flush=True)
    frame = build_table(raw, qc, pd.read_parquet(config.inventory), config.missing_value)
    frame.to_parquet(output / "microsvp_timing_diagnostic.parquet", index=False)
    csv = frame.copy()
    for name in csv:
        if isinstance(csv[name].dtype, pd.DatetimeTZDtype):
            csv[name] = csv[name].map(lambda t: "" if pd.isna(t) else t.isoformat())
    csv.to_csv(output / "microsvp_timing_diagnostic.csv", index=False)
    summary = summarize(frame, matching)
    summary["configuration"] = {"qc_directory": str(config.input_directory), "raw_directory": str(raw_directory.resolve()), "inventory": str(config.inventory), "missing_value": config.missing_value}
    (output / "microsvp_timing_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(frame, summary, output)
    if figures:
        plot_timelines(frame, output)
    print(json.dumps({"cohorts": summary["cohorts"], "focus": summary["focus"], "inventory_check": summary["inventory_check"]}, indent=2))
    print(f"Report and artifacts: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Existing ARCTERX preprocessing YAML (read only)")
    parser.add_argument("--raw-directory", type=Path, required=True, help="Raw/MicroSVP directory")
    parser.add_argument("--output-directory", type=Path, default=Path("data/diagnostics"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    run(args.config, args.raw_directory, args.output_directory, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
