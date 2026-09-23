"""Classify local geometry of existing jump flags; never changes production QC."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab import __version__
from drifterlab.qc import flags
from drifterlab.qc.review import CLASSES, MAX_LOCAL_GAP_SECONDS, GAP_TOLERANCE_SECONDS, run_class
if __package__:
    from .residual_jumps import Track, THRESHOLD_M_S, flagged_pairs, load_track, write_table
else:
    from residual_jumps import Track, THRESHOLD_M_S, flagged_pairs, load_track, write_table


def utc(time, index):
    return pd.NaT if index is None else pd.Timestamp(time[index]).tz_localize("UTC")


def neighbors(valid_indices: np.ndarray, index: int) -> tuple[int | None, int | None]:
    lo = np.searchsorted(valid_indices, index, side="left")
    hi = np.searchsorted(valid_indices, index, side="right")
    return (int(valid_indices[lo - 1]) if lo else None,
            int(valid_indices[hi]) if hi < len(valid_indices) else None)


def pair(track: Track, left: int | None, right: int | None) -> dict:
    result = {"speed_m_s": np.nan, "distance_m": np.nan, "dt_seconds": np.nan,
              "n_rows_between": None, "n_invalid_positions_between": None, "usable_context": False}
    if left is None or right is None:
        return result
    indices = [left, right]
    dt = float(flags.interval_seconds(track.time[indices])[1])
    speed = float(flags.consecutive_speed(track.time[indices], track.lon[indices], track.lat[indices])[1])
    result.update(speed_m_s=speed, distance_m=speed * dt, dt_seconds=dt,
                  n_rows_between=right - left - 1,
                  n_invalid_positions_between=int((~track.valid[left + 1:right]).sum()),
                  usable_context=bool(np.isfinite(speed) and 0 < dt <= MAX_LOCAL_GAP_SECONDS + GAP_TOLERANCE_SECONDS))
    return result


def turning_angle(track: Track, a: int | None, b: int, c: int | None) -> float:
    """Spherical tangent heading change at B: 0 degrees straight, 180 reversal."""
    if a is None or c is None:
        return np.nan
    lon, lat = np.radians(track.lon[[a, b, c]]), np.radians(track.lat[[a, b, c]])
    vectors = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    va, vb, vc = vectors
    incoming = vb * np.dot(va, vb) - va
    outgoing = vc - vb * np.dot(vc, vb)
    scale = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
    return float(np.degrees(np.arccos(np.clip(np.dot(incoming, outgoing) / scale, -1, 1)))) if scale > 1e-24 else np.nan


def geometry(track: Track, b: int, valid_indices: np.ndarray) -> dict:
    a, c = neighbors(valid_indices, b)
    p = neighbors(valid_indices, a)[0] if a is not None else None
    d = neighbors(valid_indices, c)[1] if c is not None else None
    row = {}
    for label, index in [("P", p), ("A", a), ("B", b), ("C", c), ("D", d)]:
        row[f"row_index_{label}"] = index
        row[f"source_obs_index_{label}"] = None if index is None else int(track.source_index[index])
        row[f"time_{label}"] = utc(track.time, index)
    for label, left, right in [("PA", p, a), ("AB", a, b), ("BC", b, c), ("CD", c, d), ("AC", a, c)]:
        values = pair(track, left, right)
        row.update({f"{key}_{label}": value for key, value in values.items()})
    # A/C must be immediate native neighbors: never invent flags across masked rows.
    actual_flagged_legs = a == b - 1 and c == b + 1 and track.flags[b] and track.flags[b + 1]
    row["two_existing_flagged_legs"] = bool(actual_flagged_legs)
    row["triplet_context_usable"] = bool(all(row[f"usable_context_{label}"] for label in ("AB", "BC", "AC")))
    row["passes_local_skip_one_test"] = bool(actual_flagged_legs and row["triplet_context_usable"]
                                               and row["speed_m_s_AC"] <= THRESHOLD_M_S)
    row["turning_angle_at_B_degrees"] = turning_angle(track, a, b, c)
    row["outer_context_usable"] = bool(row["usable_context_PA"] and row["usable_context_CD"])
    row["outer_context_above_threshold"] = bool(any(row[f"usable_context_{k}"] and row[f"speed_m_s_{k}"] > THRESHOLD_M_S for k in ("PA", "CD")))
    return row


def classify_track(track: Track, *, validate_stored: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base, _ = flagged_pairs(track) if validate_stored else (None, None)
    right = np.flatnonzero(track.flags)
    runs, points, edges = [], [], []
    valid = np.flatnonzero(track.valid & ~np.isnat(track.time))
    cached_geometry = {}
    dt = flags.interval_seconds(track.time)
    if len(right):
        parts = np.split(right, np.flatnonzero(np.diff(right) > 1) + 1)
        for number, part in enumerate(parts, start=1):
            start, end = int(part[0] - 1), int(part[-1])
            run_id = f"{track.platform}:r{number:04d}"
            before_index = neighbors(valid, start)[0]
            after_index = neighbors(valid, end)[1]
            before, after = pair(track, before_index, start), pair(track, end, after_index)
            triplets = []
            for b in range(start + 1, end):
                cached_geometry[b] = geometry(track, b, valid)
                triplets.append(cached_geometry[b])
            category, reason = run_class(len(part), triplets, before, after, dt[part])
            passing = [g["row_index_B"] for g in triplets if g["passes_local_skip_one_test"]]
            suspect = passing[0] if category == CLASSES[0] else None
            outer_bridge = pair(track, start, end)
            run = {"platform_code": track.platform, "run_id": run_id, "classification": category, "classification_reason": reason,
                   "n_flagged_edges": len(part), "run_start_time": utc(track.time, start), "run_end_time": utc(track.time, end),
                   "run_duration_seconds": outer_bridge["dt_seconds"], "row_index_start": start, "row_index_end": end,
                   "point_indices_json": json.dumps(list(range(start, end + 1))),
                   "source_point_indices_json": json.dumps(track.source_index[start:end + 1].astype(int).tolist()),
                   "max_speed_m_s": float(track.speed[part].max()), "total_flagged_distance_m": float((track.speed[part] * dt[part]).sum()),
                   "run_endpoint_speed_m_s": outer_bridge["speed_m_s"], "n_local_skip_one_candidates": len(passing),
                   "local_skip_one_candidate_indices_json": json.dumps(passing),
                   "suspected_point_index": suspect, "suspected_source_obs_index": None if suspect is None else int(track.source_index[suspect]),
                   "context_before_index": before_index, "context_after_index": after_index,
                   "outer_context_usable": bool(before["usable_context"] and after["usable_context"]),
                   "context_before_speed_m_s": before["speed_m_s"], "context_after_speed_m_s": after["speed_m_s"],
                   "context_before_dt_seconds": before["dt_seconds"], "context_after_dt_seconds": after["dt_seconds"],
                   "n_rows_crossed_before_context": before["n_rows_between"], "n_rows_crossed_after_context": after["n_rows_between"],
                   "outer_context_above_threshold": bool(any(v["usable_context"] and v["speed_m_s"] > THRESHOLD_M_S for v in (before, after))),
                   "manual_inspection_required": True}
            runs.append(run)
            for g in triplets:
                points.append({"platform_code": track.platform, "run_id": run_id, "run_classification": category,
                               "isolated_run_candidate": category == CLASSES[0] and g["passes_local_skip_one_test"],
                               "candidate_scope": "isolated_two_edge_run" if category == CLASSES[0] else "overlapping_or_unresolved_run",
                               **g})
            for j in part:
                j = int(j)
                if j not in cached_geometry:
                    cached_geometry[j] = geometry(track, j, valid)
                support = [b for b in passing if j in (b, b + 1)]
                edges.append({"platform_code": track.platform, "row_index_2": j, "run_id": run_id,
                              "classification": category, "run_n_flagged_edges": len(part),
                              "suspected_point_index": suspect,
                              "explained_by_isolated_run_candidate": category == CLASSES[0],
                              "has_local_skip_one_support": bool(support), "n_local_candidates_supporting_edge": len(support),
                              "local_candidate_indices_json": json.dumps(support), **cached_geometry[j]})
    run_columns = ["platform_code", "run_id", "classification", "n_flagged_edges", "max_speed_m_s", "run_start_time", "run_end_time",
                   "n_local_skip_one_candidates", "suspected_point_index", "row_index_start", "row_index_end"]
    point_columns = ["platform_code", "run_id", "run_classification", "isolated_run_candidate", "candidate_scope",
                     "row_index_B", "passes_local_skip_one_test", "speed_m_s_AC", "outer_context_usable", "outer_context_above_threshold"]
    edge_columns = ["platform_code", "row_index_2", "run_id", "classification", "explained_by_isolated_run_candidate", "has_local_skip_one_support"]
    result = [pd.DataFrame(edges) if edges else pd.DataFrame(columns=edge_columns),
              pd.DataFrame(runs) if runs else pd.DataFrame(columns=run_columns),
              pd.DataFrame(points) if points else pd.DataFrame(columns=point_columns)]
    for table in result:
        for column in [c for c in table if c.startswith("time_") or c in ("run_start_time", "run_end_time")]:
            table[column] = pd.to_datetime(table[column], utc=True).astype("datetime64[ns, UTC]")
        for column in [c for c in table if c.startswith(("row_index_", "source_obs_index_")) or c in ("suspected_point_index", "suspected_source_obs_index", "context_before_index", "context_after_index")]:
            table[column] = table[column].astype("Int64")
    if validate_stored:
        result[0] = base.merge(result[0], on=["platform_code", "row_index_2"], how="left", validate="one_to_one")
    return tuple(result)


def precision_sensitivity(track: Track, baseline: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[dict]:
    finite_time = ~np.isnat(track.time)
    rounded = pd.DatetimeIndex(track.time).round("s").to_numpy()
    jitter = np.where(np.arange(len(track.time)) % 2, -10000, 10000).astype("timedelta64[ns]")
    plus, minus = track.time.copy(), track.time.copy()
    plus[finite_time] += jitter[finite_time]
    minus[finite_time] -= jitter[finite_time]
    base_edges, _, base_points = baseline
    base_labels = dict(zip(base_edges.row_index_2, base_edges.classification))
    base_candidates = set(base_points.loc[base_points.passes_local_skip_one_test.astype(bool), "row_index_B"])
    results = []
    for name, times in [("nearest_second_in_memory", rounded), ("alternating_plus_minus_10_microseconds", plus),
                        ("alternating_minus_plus_10_microseconds", minus)]:
        speed = flags.consecutive_speed(times, track.lon, track.lat)
        variant = replace(track, time=times, speed=speed, flags=speed > THRESHOLD_M_S)
        ed, _, pts = classify_track(variant, validate_stored=False)
        labels = dict(zip(ed.row_index_2, ed.classification))
        candidates = set(pts.loc[pts.passes_local_skip_one_test.astype(bool), "row_index_B"])
        common = np.isfinite(speed) & np.isfinite(track.speed)
        changes = sum(base_labels.get(k) != labels.get(k) for k in set(base_labels) | set(labels))
        results.append({"scenario": name, "platform_code": track.platform,
                        "changed_flags": int((track.flags != variant.flags).sum()), "changed_edge_classes": changes,
                        "changed_candidate_points": len(base_candidates ^ candidates),
                        "max_speed_difference_m_s": float(np.max(np.abs(speed[common] - track.speed[common]))) if common.any() else 0.,
                        "max_clock_shift_microseconds": float(np.max(np.abs((times[finite_time] - track.time[finite_time]) / np.timedelta64(1, "ns")))) / 1000 if finite_time.any() else 0.})
    return results


def summarize(edges: pd.DataFrame, runs: pd.DataFrame, points: pd.DataFrame, sensitivity: pd.DataFrame) -> dict:
    passing = points[points.passes_local_skip_one_test.astype(bool)]
    isolated = points[points.isolated_run_candidate.astype(bool)]
    counts = []
    for category in CLASSES:
        selected = runs[runs.classification == category]
        counts.append({"classification": category, "n_runs": len(selected), "n_edges": int(selected.n_flagged_edges.sum()),
                       "n_drifters": selected.platform_code.nunique()})
    precision = []
    for scenario, group in sensitivity.groupby("scenario", sort=False):
        changed = group[(group.changed_flags > 0) | (group.changed_edge_classes > 0) | (group.changed_candidate_points > 0)]
        precision.append({"scenario": scenario, **{col: int(group[col].sum()) for col in ("changed_flags", "changed_edge_classes", "changed_candidate_points")},
                          "affected_platforms": changed.platform_code.tolist(), "max_speed_difference_m_s": float(group.max_speed_difference_m_s.max()),
                          "max_clock_shift_microseconds": float(group.max_clock_shift_microseconds.max())})
    return {"total_flagged_edges": len(edges), "total_flagged_runs": len(runs), "affected_drifters": edges.platform_code.nunique(),
            "class_counts": counts, "isolated_spike_candidate_points": len(isolated),
            "edges_explained_by_isolated_run_candidates": int(edges.explained_by_isolated_run_candidate.sum()),
            "isolated_run_edge_fraction": float(edges.explained_by_isolated_run_candidate.sum() / len(edges)) if len(edges) else 0.,
            "all_local_skip_one_candidate_points": len(passing), "edges_with_any_local_skip_one_support": int(edges.has_local_skip_one_support.sum()),
            "overlapping_candidates_in_persistent_runs": int((passing.run_classification == CLASSES[2]).sum()),
            "isolated_candidates_with_outer_context_above_threshold": int(isolated.outer_context_above_threshold.sum()),
            "isolated_candidates_without_usable_outer_context": int((~isolated.outer_context_usable.astype(bool)).sum()),
            "run_length_distribution": {str(k): int(v) for k, v in runs.n_flagged_edges.value_counts().sort_index().items()},
            "precision_sensitivity": precision,
            "minimum_skip_one_speed_margin_to_3_m_s": float((passing.speed_m_s_AC - 3).abs().min()) if len(passing) else None}


def platform_counts(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for platform, group in runs.groupby("platform_code", sort=True):
        row = {"platform_code": platform, "n_flagged_runs": len(group), "n_flagged_edges": int(group.n_flagged_edges.sum())}
        row.update({f"n_{category}": int((group.classification == category).sum()) for category in CLASSES})
        rows.append(row)
    return pd.DataFrame(rows, columns=["platform_code", "n_flagged_runs", "n_flagged_edges", *[f"n_{c}" for c in CLASSES]])


def choose_examples(runs: pd.DataFrame, points: pd.DataFrame) -> dict[str, list[str]]:
    isolated = runs[runs.classification == CLASSES[0]].sort_values(["max_speed_m_s", "run_id"], ascending=[False, True])
    persistent = runs[runs.classification == CLASSES[2]]
    # Include both complete surrounding context and a severe but ambiguous case.
    clean = isolated[isolated.outer_context_usable & ~isolated.outer_context_above_threshold] if len(isolated) else isolated
    chosen_isolated = []
    for table in (clean.head(1), isolated.head(1), clean.tail(1), isolated):
        for run_id in table.run_id:
            if run_id not in chosen_isolated and len(chosen_isolated) < 3:
                chosen_isolated.append(run_id)
    chosen_persistent = []
    for table in (persistent.sort_values(["n_flagged_edges", "max_speed_m_s"], ascending=False).head(1),
                  persistent.sort_values("max_speed_m_s", ascending=False)):
        for run_id in table.run_id:
            if run_id not in chosen_persistent and len(chosen_persistent) < 3:
                chosen_persistent.append(run_id)
    return {"isolated_spike_candidate": chosen_isolated, "persistent_high_speed_run": chosen_persistent}


def write_report(summary: dict, runs: pd.DataFrame, points: pd.DataFrame, platforms: pd.DataFrame, output: Path) -> None:
    n = summary["total_flagged_edges"]
    isolated_edges = summary["edges_explained_by_isolated_run_candidates"]
    supported = summary["edges_with_any_local_skip_one_support"]
    lines = ["# Local geometry of ARCTERX residual jump flags", "",
             f"**{n:,} flagged edges in {summary['total_flagged_runs']:,} runs across {summary['affected_drifters']} drifters.** "
             "All original speeds and flags were reproduced exactly. This is a diagnostic; all production data, flags and validity masks remain unchanged.", "",
             "## Classification rules", "",
             "Runs are maximal sequences of consecutive flagged native edge indices. An unflagged edge, invalid fix or invalid time separates runs. "
             "No missing rows are removed and runs are never joined across them. Speeds and distances use the existing production haversine/actual-elapsed-time functions.", "",
             "For every internal point B of a run, evaluate its immediate native neighbors A and C. A local skip-one candidate requires both existing AB and BC "
             "flags and a direct AC speed <=3 m/s. Each leg and the total AC interval must be positive and <=30 minutes plus the existing one-second gap tolerance. "
             "No speed tolerance is applied and no nominal five-minute denominator is used. Turning angle is descriptive only (0 degrees straight; 180 reversal).", "",
             "The four run/edge labels are exclusive:", "",
             "- `isolated_spike_candidate`: exactly two flagged edges; skipping their one middle point gives a reasonable AC bridge. B is recorded as a candidate, not removed.",
             "- `single_flagged_edge`: one flagged edge, with a nearest valid point on either side within the local gap limit. Neither endpoint is guessed to be wrong.",
             "- `persistent_high_speed_run`: at least two flagged edges not fully resolved by one skip. Runs longer than two edges remain in this class even when individual overlapping triples pass the bridge test: one skip necessarily leaves flagged edges.",
             "- `boundary_or_insufficient_context`: required context is absent, has nonpositive elapsed time, or exceeds the local gap limit.", "",
             "Outside a run, P/D are the nearest valid timed QC fixes. Context-only PA/CD comparisons can cross masked rows; those crossings and indices are recorded explicitly "
             "and never create or connect production flags. A valid A/B/C triple is sufficient to identify a geometric candidate; P/D are additional evidence and can expose ambiguity.", "",
             "## Counts", "", "| Class | Runs/cases | Flagged edges | Affected drifters |", "|---|---:|---:|---:|"]
    for item in summary["class_counts"]:
        lines.append(f"| {item['classification']} | {item['n_runs']} | {item['n_edges']} | {item['n_drifters']} |")
    lines += ["", f"Isolated candidate points in two-edge runs: **{summary['isolated_spike_candidate_points']}**, covering "
              f"**{isolated_edges}/{n} edges ({100 * isolated_edges / n:.2f}%)**." if n else "", "",
              f"The literal local AB/BC/AC criterion is satisfied by **{summary['all_local_skip_one_candidate_points']} points**, including "
              f"**{summary['overlapping_candidates_in_persistent_runs']} overlapping candidates within longer runs**. "
              f"Their union covers **{supported} distinct edges**; edges shared by adjacent candidate triples are counted once. "
              "This broader local support must not be presented as resolving all those runs through independent single-point removals.", "",
              f"Among the isolated two-edge candidates, **{summary['isolated_candidates_with_outer_context_above_threshold']}** have an additional >3 m/s nearest-valid "
              f"outer-context comparison, and **{summary['isolated_candidates_without_usable_outer_context']}** lack usable context on at least one outer side. "
              "These are warnings for interpretation, not new QC flags. A suspect B can itself be on the surrounding trajectory when A/C are displaced; the geometric test alone cannot identify ground truth.", "",
              "| Run length (flagged edges) | Runs |", "|---:|---:|"]
    lines += [f"| {length} | {count} |" for length, count in summary["run_length_distribution"].items()]
    for category, title in [(CLASSES[0], "Top drifters by isolated candidate count"), (CLASSES[2], "Top drifters by persistent-run count")]:
        col = f"n_{category}"
        top = platforms[platforms[col] > 0].sort_values([col, "platform_code"], ascending=[False, True]).head(10)
        lines += ["", "## " + title, "", "| Platform | Count |", "|---|---:|"]
        lines += [f"| {row.platform_code} | {getattr(row, col)} |" for row in top.itertuples()]
    persistent = runs[runs.classification == CLASSES[2]].sort_values(["max_speed_m_s", "run_id"], ascending=[False, True])
    lines += ["", "## Worst persistent runs", "",
              "Ranked by maximum existing audit speed. A run is a sequence of flagged coordinate differences, not a claim of sustained physical drifter speed.", "",
              "| Run | Start UTC (display seconds) | Edges | Native point indices | Maximum m/s | Local skip-one candidates |", "|---|---|---:|---|---:|---:|"]
    for row in persistent.head(20).itertuples():
        lines.append(f"| {row.run_id} | {row.run_start_time.round('s').strftime('%Y-%m-%d %H:%M:%S')} | {row.n_flagged_edges} | {row.row_index_start}–{row.row_index_end} | {row.max_speed_m_s:.3f} | {row.n_local_skip_one_candidates} |")
    boundary = runs[runs.classification == CLASSES[3]]
    lines += ["", "## Insufficient-context cases", "",
              "| Run | Native points | Context interval before (min) | Context interval after (min) |", "|---|---|---:|---:|"]
    for row in boundary.itertuples():
        lines.append(f"| {row.run_id} | {row.row_index_start}–{row.row_index_end} | {row.context_before_dt_seconds / 60:.3f} | {row.context_after_dt_seconds / 60:.3f} |")
    lines += ["", "## Timestamp precision test", "",
              "Three in-memory experiments recompute flag eligibility and classification: nearest-second clocks and both phases of alternating +/-10-microsecond perturbations. "
              "All pair and bridge speeds use the perturbed actual elapsed times. Original stored clocks and output geometry retain full precision.", "",
              "| Experiment | Changed flags | Changed edge classes | Changed candidate points | Maximum clock shift (microseconds) |", "|---|---:|---:|---:|---:|"]
    for row in summary["precision_sensitivity"]:
        lines.append(f"| {row['scenario']} | {row['changed_flags']} | {row['changed_edge_classes']} | {row['changed_candidate_points']} | {row['max_clock_shift_microseconds']:.6f} |")
    lines += ["", f"Minimum absolute candidate-bridge speed margin to 3 m/s: {summary['minimum_skip_one_speed_margin_to_3_m_s']} m/s. "
              "The detailed sensitivity table records changed platform IDs and maximum numerical speed differences.", "",
              "## Files and interpretation", "",
              "`classified_flagged_edges.csv` / `.parquet` preserve the previous flagged-edge fields and add run class, suspect/support indices, and local P/A/B/C/D geometry. "
              "In this table B is the later endpoint of the edge; `suspected_point_index` explicitly identifies the run's candidate and may differ from B. "
              "`classified_runs.csv` / `.parquet` record all affected points, source indices and context. "
              "`candidate_points.csv` / `.parquet` record every middle point that passes the local skip-one test, including overlapping candidates in persistent runs. "
              "`counts_by_platform.csv`, `worst_persistent_runs.csv`, and `precision_sensitivity.csv` support the summary. All indices are zero-based native/source indices as named.", "",
              "Two figures show selected isolated and persistent cases. Green dashed AC bridges are hypothetical diagnostic connections, not corrected or interpolated trajectories. "
              "Gray marks identify existing invalid fixes. No reconstruction, drogue mask, new position validity or point-removal rule is used.", "",
              "## Conclusion", ""]
    remaining = summary["total_flagged_runs"] - summary["isolated_spike_candidate_points"]
    lines.append(f"{isolated_edges}/{n} flagged edges ({100 * isolated_edges / n:.2f}%) fit isolated two-edge/single-point geometry; "
                 f"{remaining} runs remain single-edge ambiguous, persistent, or context-limited. "
                 "Passing a local skip-one test identifies a review candidate, not a justified automatic deletion." if n else "No residual flagged edges were found.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_examples(master: Path, runs: pd.DataFrame, points: pd.DataFrame, selected: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rc = {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    with xr.open_zarr(master, chunks=None) as ds, plt.rc_context(rc):
        platforms = list(ds.platform_code.values.astype(str))
        for category, ids in selected.items():
            if not ids:
                continue
            fig, axes = plt.subplots(len(ids), 2, figsize=(13.5, 3.7 * len(ids)), squeeze=False)
            for (ax, speed_ax), run_id in zip(axes, ids):
                row = runs[runs.run_id == run_id].iloc[0]
                track = load_track(ds, platforms.index(row.platform_code))
                start, end = int(row.row_index_start), int(row.row_index_end)
                selected_time = np.flatnonzero((track.time >= track.time[start] - np.timedelta64(20, "m")) &
                                               (track.time <= track.time[end] + np.timedelta64(20, "m")))
                lo, hi = int(selected_time[0]), int(selected_time[-1]) + 1
                lon, lat, valid = track.lon[lo:hi], track.lat[lo:hi], track.valid[lo:hi]
                origin = np.flatnonzero(valid)[0]
                east = flags.EARTH_RADIUS_M * np.cos(np.radians(lat[origin])) * np.radians(lon - lon[origin]) / 1000
                north = flags.EARTH_RADIUS_M * np.radians(lat - lat[origin]) / 1000
                east[~valid], north[~valid] = np.nan, np.nan
                ax.plot(east, north, "o-", color="#777777", ms=3, lw=.8)
                for j in range(start + 1, end + 1):
                    k = j - lo
                    ax.plot(east[k - 1:k + 1], north[k - 1:k + 1], "o-", color="#D55E00", ms=4, lw=1.3)
                local = points[(points.run_id == run_id) & points.passes_local_skip_one_test]
                for p in local.itertuples():
                    a, b, c = int(p.row_index_A) - lo, int(p.row_index_B) - lo, int(p.row_index_C) - lo
                    ax.plot(east[[a, c]], north[[a, c]], "--", color="#009E73", lw=1.3)
                    speed_ax.plot([track.time[a + lo], track.time[c + lo]], [p.speed_m_s_AC] * 2, "--", color="#009E73", lw=1.2)
                    if category == CLASSES[0]:
                        ax.scatter([east[b]], [north[b]], marker="*", s=100, color="#CC79A7", zorder=5)
                for k in range(start - lo, end - lo + 1):
                    label = str(k + lo)
                    if category == CLASSES[0]:
                        label = "ABC"[k - (start - lo)] + " · " + label
                    ax.annotate(label, (east[k], north[k]), xytext=(5, 5), textcoords="offset points", fontsize=8)
                span = 1.3 * max(np.ptp(east[valid]), np.ptp(north[valid]), .1)
                cx, cy = (np.nanmin(east) + np.nanmax(east)) / 2, (np.nanmin(north) + np.nanmax(north)) / 2
                ax.set(xlabel="Local east (km)", ylabel="Local north (km)", aspect="equal",
                       xlim=(cx - span / 2, cx + span / 2), ylim=(cy - span / 2, cy + span / 2))
                ax.set_title(f"{row.platform_code}\n{row.run_start_time.round('s').strftime('%Y-%m-%d %H:%M UTC')} · {row.n_flagged_edges} edges · max {row.max_speed_m_s:.2f} m/s", loc="left", fontsize=9)
                times = pd.DatetimeIndex(track.time[lo:hi]).tz_localize("UTC")
                speed_ax.plot(times, track.speed[lo:hi], "o-", color="#777777", ms=3, lw=.8)
                flag = track.flags[lo:hi]
                speed_ax.scatter(times[flag], track.speed[lo:hi][flag], s=25, color="#D55E00", zorder=4)
                speed_ax.axhline(THRESHOLD_M_S, color="#D55E00", ls=":", lw=1)
                for t in times[~valid]:
                    speed_ax.axvline(t, color="#bbbbbb", alpha=.8, lw=1.5)
                speed_ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
                speed_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                speed_ax.set(xlabel="UTC time", ylabel="Speed (m/s)")
                if category == CLASSES[0]:
                    p = local.iloc[0]
                    title = f"AB {p.speed_m_s_AB:.2f}; BC {p.speed_m_s_BC:.2f}; AC {p.speed_m_s_AC:.2f} m/s\nActual AC interval {p.dt_seconds_AC / 60:.2f} min"
                    if row.outer_context_above_threshold or not row.outer_context_usable:
                        title += "; outer context needs review"
                else:
                    title = f"{len(local)} overlapping skip-one candidates\nOne skipped point leaves {max(0, row.n_flagged_edges - 2)} original flagged edges"
                speed_ax.set_title(title, loc="left", fontsize=9)
                for panel in (ax, speed_ax):
                    panel.grid(alpha=.2)
            title = "Isolated two-edge spike geometry: candidate B and hypothetical A→C bridge" if category == CLASSES[0] else "Persistent / consecutive flagged runs: local bridges do not resolve a whole run"
            fig.suptitle(title, x=.08, ha="left", y=.99, fontsize=13)
            handles = [Line2D([], [], color="#777777", marker="o", ms=3, label="Native QC; gaps retained"),
                       Line2D([], [], color="#D55E00", marker="o", ms=3, label="Existing flagged edge"),
                       Line2D([], [], color="#009E73", ls="--", label="Hypothetical skip-one bridge"),
                       Line2D([], [], color="#bbbbbb", lw=2, label="Invalid position (time panel)")]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.075, .969), ncol=2, frameon=False, fontsize=8)
            fig.text(.08, .012, "Native positions only. Indices refer to chronological master rows. Green bridges are diagnostic comparisons; no positions are removed.\nA passing bridge does not prove the middle coordinate is wrong. Map projection is for display only; speeds use the production spherical distance.", fontsize=8)
            fig.subplots_adjust(left=.07, right=.98, bottom=.08, top=.87, hspace=.65, wspace=.42)
            stem = "01_isolated_spike_examples" if category == CLASSES[0] else "02_persistent_run_examples"
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{stem}.{extension}", dpi=300)
            plt.close(fig)


def run(master: Path, output: Path, *, flag_table: Path | None = None, figures: bool = True) -> dict:
    master, output = master.resolve(), output.resolve()
    if output == master or master in output.parents:
        raise ValueError("Output must be outside the master Zarr")
    tables = [[], [], []]
    sensitivity = []
    with xr.open_zarr(master, chunks=None) as ds:
        if ds.attrs.get("residual_jump_speed_m_s") != THRESHOLD_M_S or ds.residual_jump_flag_qc.attrs.get("threshold_m_s") != THRESHOLD_M_S:
            raise ValueError("Expected the existing strict 3 m/s flag threshold")
        platforms = ds.platform_code.values.astype(str)
        if not len(platforms) or len(set(platforms)) != len(platforms):
            raise ValueError("Expected nonempty unique trajectory IDs")
        for i in range(len(platforms)):
            track = load_track(ds, i)
            result = classify_track(track)
            if len(result[0]) != int(ds.n_residual_jumps_qc.values[i]):
                raise ValueError("Stored per-platform jump count mismatch")
            for group, table in zip(tables, result):
                if len(table):
                    group.append(table)
            sensitivity.extend(precision_sensitivity(track, result))
        # Preserve schemas even when the complete synthetic input has no flags.
        edges, runs, points = [pd.concat(group, ignore_index=True) if group else empty.iloc[:0].copy()
                               for group, empty in zip(tables, result)]
    if flag_table:
        previous = pd.read_parquet(flag_table)
        keys = ["platform_code", "row_index_2"]
        pd.testing.assert_frame_equal(edges[previous.columns].sort_values(keys).reset_index(drop=True),
                                      previous.sort_values(keys).reset_index(drop=True), check_dtype=False)
    sensitivity = pd.DataFrame(sensitivity)
    summary = summarize(edges, runs, points, sensitivity)
    counts = platform_counts(runs)
    selected = choose_examples(runs, points)
    summary.update({"settings": {"threshold_m_s": THRESHOLD_M_S, "max_local_gap_seconds": MAX_LOCAL_GAP_SECONDS,
                                  "gap_comparison_tolerance_seconds": GAP_TOLERANCE_SECONDS,
                                  "isolated_run_requires_exactly_two_flagged_edges": True,
                                  "outer_context_rule": "nearest valid timed QC point, within 30 minutes + 1 second",
                                  "run_grouping": "consecutive existing flagged native edge indices only"},
                    "processed_utc": pd.Timestamp.now(tz="UTC").isoformat(), "time_reference": "UTC", "software_version": __version__,
                    "master_zarr": str(master), "previous_flag_table": str(flag_table.resolve()) if flag_table else None,
                    "previous_flag_table_sha256": sha256(flag_table.read_bytes()).hexdigest() if flag_table else None,
                    "master_metadata_sha256": sha256((master / ".zmetadata").read_bytes()).hexdigest(),
                    "diagnostic_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                    "audit_code_sha256": sha256(Path(flags.__file__).read_bytes()).hexdigest(),
                    "selected_examples": selected})
    output.mkdir(parents=True, exist_ok=True)
    for table, name in [(edges, "classified_flagged_edges"), (runs, "classified_runs"), (points, "candidate_points")]:
        write_table(table, output / name, parquet=True)
    write_table(counts, output / "counts_by_platform")
    write_table(sensitivity, output / "precision_sensitivity")
    persistent = runs[runs.classification == CLASSES[2]].sort_values(["max_speed_m_s", "run_id"], ascending=[False, True])
    write_table(persistent.head(20), output / "worst_persistent_runs")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, runs, points, counts, output)
    if figures and len(edges):
        plot_examples(master, runs, points, selected, output)
    print(json.dumps(summary, indent=2))
    print(f"Local jump classification: {output}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("data/ARCTERX_MicroSVP_QC.zarr"))
    parser.add_argument("--flag-table", type=Path, default=Path("data/diagnostics/residual_jumps/flagged_jumps.parquet"))
    parser.add_argument("--output-directory", type=Path, default=Path("data/diagnostics/residual_jumps/local_classification"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    run(args.master, args.output_directory, flag_table=args.flag_table, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
