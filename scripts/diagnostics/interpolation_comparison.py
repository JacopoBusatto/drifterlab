"""Compare native residual jumps with supplied 30/60-minute reconstructions.

This is a read-only diagnostic. It uses stored samples only and never creates a
new interpolation. Run from the repository root; see ``--help``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from drifterlab import __version__
from drifterlab.qc import flags
from drifterlab.qc.position import position_valid

if __package__:
    from .residual_jumps import THRESHOLD_M_S, flagged_pairs, load_track, quantiles, write_table
else:
    from residual_jumps import THRESHOLD_M_S, flagged_pairs, load_track, quantiles, write_table


TIME_MATCH_TOLERANCE_SECONDS = 1.0
OVERLAP_TOLERANCE_SECONDS = 1.0
REPRESENTATIONS = ("interp30", "interp60")
BEHAVIORS = (
    "resolved_in_both",
    "resolved_in_30_only",
    "resolved_in_60_only",
    "remains_flagged_in_both",
    "missing_reconstructed_context",
)
LOCAL_CLASSES = (
    "isolated_spike_candidate",
    "single_flagged_edge",
    "persistent_high_speed_run",
    "boundary_or_insufficient_context",
)


@dataclass(frozen=True)
class ReconstructionTrack:
    platform: str
    time: np.ndarray
    source_index: np.ndarray
    lon30: np.ndarray
    lat30: np.ndarray
    valid30: np.ndarray
    lon60: np.ndarray
    lat60: np.ndarray
    valid60: np.ndarray
    has30: bool = True
    has60: bool = True


def _utc(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def _datetime64(value) -> np.datetime64:
    stamp = _utc(value).tz_localize(None)
    return stamp.to_datetime64().astype("datetime64[ns]")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_reconstruction(ds: xr.Dataset, index: int) -> ReconstructionTrack:
    n = int(ds.n_obs_interp.values[index])
    if n < 0 or n > ds.sizes["obs_interp"]:
        raise ValueError("Invalid reconstructed observation count")
    selection = dict(trajectory=index, obs_interp=slice(0, n))
    values = {name: ds[name].isel(**selection).values for name in (
        "time_interp", "source_obs_index_interp", "lon_interp_30m", "lat_interp_30m",
        "position_valid_interp_30m", "lon_interp_60m", "lat_interp_60m",
        "position_valid_interp_60m")}
    track = ReconstructionTrack(
        platform=str(ds.platform_code.values[index]),
        time=values["time_interp"].astype("datetime64[ns]"),
        source_index=values["source_obs_index_interp"],
        lon30=values["lon_interp_30m"], lat30=values["lat_interp_30m"],
        valid30=values["position_valid_interp_30m"].astype(bool),
        lon60=values["lon_interp_60m"], lat60=values["lat_interp_60m"],
        valid60=values["position_valid_interp_60m"].astype(bool),
        has30=bool(ds.has_interp_30m.values[index]),
        has60=bool(ds.has_interp_60m.values[index]),
    )
    arrays = (track.source_index, track.lon30, track.lat30, track.valid30,
              track.lon60, track.lat60, track.valid60)
    if track.time.ndim != 1 or any(array.shape != track.time.shape for array in arrays):
        raise ValueError(f"Misaligned reconstructed arrays for {track.platform}")
    if not np.array_equal(position_valid(track.lon30, track.lat30), track.valid30):
        raise ValueError(f"30-minute position mask mismatch for {track.platform}")
    if not np.array_equal(position_valid(track.lon60, track.lat60), track.valid60):
        raise ValueError(f"60-minute position mask mismatch for {track.platform}")
    finite_time = track.time[~np.isnat(track.time)]
    if len(finite_time) and np.any(finite_time[1:] < finite_time[:-1]):
        raise ValueError(f"Reconstructed time is not chronological for {track.platform}")
    return track


def match_stored_time(time: np.ndarray, target, tolerance_seconds: float = TIME_MATCH_TOLERANCE_SECONDS) -> dict:
    """Find one stored timestamp near target; ties/duplicates remain unavailable."""
    wanted = _datetime64(target)
    valid = np.flatnonzero(~np.isnat(time))
    result = {"index": None, "matched_time": pd.NaT, "offset_microseconds": np.nan,
              "exact": False, "status": "no_timestamp_within_tolerance"}
    if not len(valid):
        return result
    epochs = time[valid].astype("datetime64[ns]").astype(np.int64)
    target_ns = wanted.astype(np.int64)
    insertion = int(np.searchsorted(epochs, target_ns))
    positions = sorted(set(i for i in (insertion - 1, insertion) if 0 <= i < len(valid)))
    if not positions:
        return result
    deltas = np.array([abs(int(epochs[i]) - int(target_ns)) for i in positions], dtype=np.int64)
    minimum = int(deltas.min())
    if minimum > tolerance_seconds * 1e9:
        return result
    winners = [positions[i] for i in np.flatnonzero(deltas == minimum)]
    # Any duplicate stored occurrence of the winning clock is ambiguous, even if
    # searchsorted exposed only one side of the duplicate block.
    winning_epoch = epochs[winners[0]]
    duplicate_positions = np.flatnonzero(epochs == winning_epoch)
    if len(winners) != 1 or len(duplicate_positions) != 1:
        result["status"] = "ambiguous_duplicate_or_equidistant_timestamp"
        return result
    index = int(valid[winners[0]])
    offset_ns = int(time[index].astype(np.int64)) - int(target_ns)
    result.update(index=index, matched_time=_utc(time[index]),
                  offset_microseconds=offset_ns / 1000, exact=offset_ns == 0, status="matched")
    return result


def _representation_arrays(track: ReconstructionTrack, representation: str):
    if representation == "interp30":
        return track.lon30, track.lat30, track.valid30, track.has30
    if representation == "interp60":
        return track.lon60, track.lat60, track.valid60, track.has60
    raise ValueError(f"Unknown representation: {representation}")


def reconstructed_interval(track: ReconstructionTrack, representation: str, time_1, time_2) -> dict:
    """Compare two stored reconstruction samples nearest to native endpoints."""
    prefix = representation
    lon, lat, valid, present = _representation_arrays(track, representation)
    first = match_stored_time(track.time, time_1)
    second = match_stored_time(track.time, time_2)
    result = {
        f"{prefix}_available": False,
        f"{prefix}_unavailable_reason": "",
        f"{prefix}_time_1": first["matched_time"], f"{prefix}_time_2": second["matched_time"],
        f"{prefix}_time_1_offset_microseconds": first["offset_microseconds"],
        f"{prefix}_time_2_offset_microseconds": second["offset_microseconds"],
        f"{prefix}_time_1_exact": first["exact"], f"{prefix}_time_2_exact": second["exact"],
        f"{prefix}_row_index_1": first["index"], f"{prefix}_row_index_2": second["index"],
        f"{prefix}_n_rows_between": None,
        f"{prefix}_source_obs_index_1": None, f"{prefix}_source_obs_index_2": None,
        f"{prefix}_position_1_valid": False, f"{prefix}_position_2_valid": False,
        f"{prefix}_dt_minutes": np.nan, f"{prefix}_distance_m": np.nan,
        f"{prefix}_speed_m_s": np.nan, f"{prefix}_flagged_gt_3_m_s": False,
    }
    if not present:
        result[f"{prefix}_unavailable_reason"] = "product_not_supplied"
        return result
    if first["status"] != "matched":
        result[f"{prefix}_unavailable_reason"] = f"time_1_{first['status']}"
        return result
    if second["status"] != "matched":
        result[f"{prefix}_unavailable_reason"] = f"time_2_{second['status']}"
        return result
    left, right = int(first["index"]), int(second["index"])
    result[f"{prefix}_n_rows_between"] = right - left - 1
    result[f"{prefix}_source_obs_index_1"] = int(track.source_index[left])
    result[f"{prefix}_source_obs_index_2"] = int(track.source_index[right])
    result[f"{prefix}_position_1_valid"] = bool(valid[left])
    result[f"{prefix}_position_2_valid"] = bool(valid[right])
    if left == right:
        result[f"{prefix}_unavailable_reason"] = "native_endpoints_match_same_stored_row"
        return result
    if not valid[left] or not valid[right]:
        result[f"{prefix}_unavailable_reason"] = "one_or_both_endpoint_positions_invalid"
        return result
    times = track.time[[left, right]]
    speed = float(flags.consecutive_speed(times, lon[[left, right]], lat[[left, right]])[1])
    dt = float(flags.interval_seconds(times)[1])
    if not np.isfinite(speed) or not np.isfinite(dt) or dt <= 0:
        result[f"{prefix}_unavailable_reason"] = "nonpositive_time_or_invalid_pair"
        return result
    result.update({
        f"{prefix}_available": True, f"{prefix}_unavailable_reason": "",
        f"{prefix}_dt_minutes": dt / 60, f"{prefix}_distance_m": speed * dt,
        f"{prefix}_speed_m_s": speed, f"{prefix}_flagged_gt_3_m_s": bool(speed > THRESHOLD_M_S),
    })
    return result


def classify_behavior(row: dict) -> str:
    if not row["interp30_available"] or not row["interp60_available"]:
        return "missing_reconstructed_context"
    flag30, flag60 = row["interp30_flagged_gt_3_m_s"], row["interp60_flagged_gt_3_m_s"]
    if not flag30 and not flag60:
        return "resolved_in_both"
    if not flag30 and flag60:
        return "resolved_in_30_only"
    if flag30 and not flag60:
        return "resolved_in_60_only"
    return "remains_flagged_in_both"


def compare_native_edges(native_edges: pd.DataFrame, reconstruction: ReconstructionTrack) -> pd.DataFrame:
    rows = []
    for edge in native_edges.itertuples(index=False):
        row = {
            "platform_code": str(edge.platform_code), "time_1": _utc(edge.time_1),
            "time_2": _utc(edge.time_2), "dt_minutes": float(edge.dt_minutes),
            "qc_distance_m": float(edge.distance_m), "qc_speed_m_s": float(edge.speed_m_s),
            "row_index_1": int(edge.row_index_1), "row_index_2": int(edge.row_index_2),
            "run_id": getattr(edge, "run_id", ""),
            "local_jump_class": getattr(edge, "classification", "unclassified"),
            "explained_by_isolated_run_candidate": bool(
                getattr(edge, "explained_by_isolated_run_candidate", False)),
        }
        row.update(reconstructed_interval(reconstruction, "interp30", edge.time_1, edge.time_2))
        row.update(reconstructed_interval(reconstruction, "interp60", edge.time_1, edge.time_2))
        row["reconstruction_behavior"] = classify_behavior(row)
        rows.append(row)
    if rows:
        return pd.DataFrame(rows)
    columns = ["platform_code", "time_1", "time_2", "dt_minutes", "qc_distance_m",
               "qc_speed_m_s", "row_index_1", "row_index_2", "run_id",
               "local_jump_class", "explained_by_isolated_run_candidate"]
    for prefix in REPRESENTATIONS:
        columns += [
            f"{prefix}_available", f"{prefix}_unavailable_reason",
            f"{prefix}_time_1", f"{prefix}_time_2",
            f"{prefix}_time_1_offset_microseconds", f"{prefix}_time_2_offset_microseconds",
            f"{prefix}_time_1_exact", f"{prefix}_time_2_exact",
            f"{prefix}_row_index_1", f"{prefix}_row_index_2", f"{prefix}_n_rows_between",
            f"{prefix}_source_obs_index_1", f"{prefix}_source_obs_index_2",
            f"{prefix}_position_1_valid", f"{prefix}_position_2_valid",
            f"{prefix}_dt_minutes", f"{prefix}_distance_m", f"{prefix}_speed_m_s",
            f"{prefix}_flagged_gt_3_m_s",
        ]
    return pd.DataFrame(columns=columns + ["reconstruction_behavior"])


def _association(time_1: np.datetime64, time_2: np.datetime64, native_edges: pd.DataFrame) -> dict:
    if native_edges.empty:
        return {"exact_native_pair": False, "native_pair_within_1s": False,
                "max_native_flag_overlap_seconds": 0.0, "associated_with_native_flag": False}
    start, end = time_1.astype(np.int64), time_2.astype(np.int64)
    native_start = pd.to_datetime(native_edges.time_1, utc=True).array.asi8
    native_end = pd.to_datetime(native_edges.time_2, utc=True).array.asi8
    exact = (native_start == start) & (native_end == end)
    near = (np.abs(native_start - start) <= TIME_MATCH_TOLERANCE_SECONDS * 1e9) & (
        np.abs(native_end - end) <= TIME_MATCH_TOLERANCE_SECONDS * 1e9)
    overlap_ns = np.maximum(0, np.minimum(native_end, end) - np.maximum(native_start, start))
    maximum = float(overlap_ns.max() / 1e9) if len(overlap_ns) else 0.0
    associated = bool(exact.any() or near.any() or maximum > OVERLAP_TOLERANCE_SECONDS)
    return {"exact_native_pair": bool(exact.any()), "native_pair_within_1s": bool(near.any()),
            "max_native_flag_overlap_seconds": maximum, "associated_with_native_flag": associated}


def scan_reconstructed_flags(track: ReconstructionTrack, representation: str,
                             native_edges: pd.DataFrame) -> pd.DataFrame:
    """Scan adjacent stored reconstruction rows; never bridge an invalid row."""
    lon, lat, valid, present = _representation_arrays(track, representation)
    columns = ["platform_code", "representation", "time_1", "time_2", "dt_minutes",
               "distance_m", "speed_m_s", "row_index_1", "row_index_2",
               "source_obs_index_1", "source_obs_index_2", "position_1_valid",
               "position_2_valid", "exact_native_pair", "native_pair_within_1s",
               "max_native_flag_overlap_seconds", "associated_with_native_flag",
               "reconstruction_only_event"]
    if not present:
        return pd.DataFrame(columns=columns)
    speed = flags.consecutive_speed(track.time, lon, lat)
    dt = flags.interval_seconds(track.time)
    rows = []
    for right in np.flatnonzero(speed > THRESHOLD_M_S):
        left = int(right - 1)
        association = _association(track.time[left], track.time[right], native_edges)
        rows.append({
            "platform_code": track.platform, "representation": representation,
            "time_1": _utc(track.time[left]), "time_2": _utc(track.time[right]),
            "dt_minutes": float(dt[right] / 60), "distance_m": float(speed[right] * dt[right]),
            "speed_m_s": float(speed[right]), "row_index_1": left, "row_index_2": int(right),
            "source_obs_index_1": int(track.source_index[left]),
            "source_obs_index_2": int(track.source_index[right]),
            "position_1_valid": bool(valid[left]), "position_2_valid": bool(valid[right]),
            **association, "reconstruction_only_event": not association["associated_with_native_flag"],
        })
    return pd.DataFrame(rows, columns=columns)


def _speed_stats(values: pd.Series) -> dict:
    clean = values.dropna()
    if clean.empty:
        return {"median": None, "p95": None, "max": None}
    return {"median": float(clean.median()), "p95": float(clean.quantile(.95)), "max": float(clean.max())}


def build_summary(comparison: pd.DataFrame, reconstructed_flags: pd.DataFrame,
                  *, selected_platforms: list[str], total_master_platforms: int) -> tuple[dict, pd.DataFrame]:
    counts = {name: int((comparison.reconstruction_behavior == name).sum()) for name in BEHAVIORS}
    complete = comparison[comparison.interp30_available & comparison.interp60_available]
    by_class_rows = []
    for local_class, group in comparison.groupby("local_jump_class", dropna=False, sort=True):
        row = {"local_jump_class": local_class, "native_flagged_edges": len(group),
               "affected_drifters": group.platform_code.nunique()}
        row.update({name: int((group.reconstruction_behavior == name).sum()) for name in BEHAVIORS})
        by_class_rows.append(row)
    by_class = pd.DataFrame(by_class_rows)
    scans = {}
    for representation in REPRESENTATIONS:
        group = reconstructed_flags[reconstructed_flags.representation == representation]
        scans[representation] = {
            "total_gt_3_m_s_edges": len(group),
            "associated_with_native_flag_intervals": int(group.associated_with_native_flag.sum()) if len(group) else 0,
            "reconstruction_only_events": int(group.reconstruction_only_event.sum()) if len(group) else 0,
            "affected_drifters": int(group.platform_code.nunique()) if len(group) else 0,
            "worst_reconstruction_only_speed_m_s": (
                float(group.loc[group.reconstruction_only_event, "speed_m_s"].max())
                if group.reconstruction_only_event.any() else None),
        }
    resolved30 = int((complete.interp30_speed_m_s <= THRESHOLD_M_S).sum())
    resolved60 = int((complete.interp60_speed_m_s <= THRESHOLD_M_S).sum())
    comparison_text = "tie"
    if resolved60 > resolved30:
        comparison_text = "60m_suppresses_more"
    elif resolved30 > resolved60:
        comparison_text = "30m_suppresses_more"
    summary = {
        "scope": {
            "selected_platform_count": len(selected_platforms),
            "total_master_platform_count": total_master_platforms,
            "complete_master_scope": len(selected_platforms) == total_master_platforms,
            "selected_platforms": selected_platforms,
        },
        "native_flagged_edges": len(comparison), "behavior_counts": counts,
        "complete_reconstruction_context_edges": len(complete),
        "complete_case_speed_m_s": {
            "qc": _speed_stats(complete.qc_speed_m_s),
            "interp30": _speed_stats(complete.interp30_speed_m_s),
            "interp60": _speed_stats(complete.interp60_speed_m_s),
        },
        "resolved_on_common_available_intervals": {
            "interp30": resolved30, "interp60": resolved60,
            "difference_60m_minus_30m": resolved60 - resolved30,
            "comparison": comparison_text,
        },
        "reconstructed_track_scans": scans,
        "threshold_m_s_strictly_above": THRESHOLD_M_S,
        "timestamp_match_tolerance_seconds": TIME_MATCH_TOLERANCE_SECONDS,
        "meaningful_temporal_overlap_tolerance_seconds": OVERLAP_TOLERANCE_SECONDS,
    }
    return summary, by_class


def make_scatter(comparison: pd.DataFrame, output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for axis, representation, label in zip(axes, REPRESENTATIONS, ("30-minute", "60-minute")):
        subset = comparison[comparison[f"{representation}_available"]]
        axis.scatter(subset.qc_speed_m_s, subset[f"{representation}_speed_m_s"], s=15, alpha=.55)
        maximum = max(THRESHOLD_M_S * 1.2, float(np.nanmax(np.r_[subset.qc_speed_m_s, subset[f"{representation}_speed_m_s"]])) if len(subset) else 4)
        axis.plot([0, maximum], [0, maximum], color=".5", linestyle="--", linewidth=1, label="equal speed")
        axis.axhline(THRESHOLD_M_S, color="#d95f02", linestyle=":", linewidth=1.2)
        axis.axvline(THRESHOLD_M_S, color="#d95f02", linestyle=":", linewidth=1.2)
        axis.set(xlabel="Native-QC speed (m/s)", ylabel=f"{label} speed (m/s)", title=f"Stored {label} reconstruction")
        axis.grid(alpha=.2)
    fig.suptitle("Native flagged intervals: QC versus supplied reconstruction")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"01_qc_vs_reconstructed_speed.{suffix}"
        fig.savefig(path, dpi=200 if suffix == "png" else None)
        paths.append(path.name)
    plt.close(fig)
    return paths


def _select_plot_cases(comparison: pd.DataFrame, reconstruction_only: pd.DataFrame) -> list[dict]:
    cases, seen = [], set()
    for behavior in ("resolved_in_both", "remains_flagged_in_both"):
        rows = comparison[comparison.reconstruction_behavior == behavior]
        if len(rows):
            row = rows.loc[rows.qc_speed_m_s.idxmax()]
            key = (str(row.platform_code), int(row.time_1.value), int(row.time_2.value))
            if key not in seen:
                seen.add(key); cases.append({"kind": behavior, **row.to_dict()})
    for representation in REPRESENTATIONS:
        rows = reconstruction_only[reconstruction_only.representation == representation]
        if len(rows):
            row = rows.loc[rows.speed_m_s.idxmax()]
            key = (str(row.platform_code), int(row.time_1.value), int(row.time_2.value))
            if key not in seen:
                seen.add(key); cases.append({"kind": f"{representation}_only", **row.to_dict()})
    return cases[:4]


def make_case_plot(ds: xr.Dataset, comparison: pd.DataFrame, reconstruction_only: pd.DataFrame,
                   output: Path) -> list[str]:
    import matplotlib.pyplot as plt

    cases = _select_plot_cases(comparison, reconstruction_only)
    if not cases:
        return []
    platforms = list(ds.platform_code.values.astype(str))
    fig, axes = plt.subplots(1, len(cases), figsize=(max(6, 4.8 * len(cases)), 4.7),
                             squeeze=False, constrained_layout=True)
    for axis, case in zip(axes[0], cases):
        index = platforms.index(str(case["platform_code"]))
        native = load_track(ds, index)
        recon = load_reconstruction(ds, index)
        center = _datetime64(case["time_1"]) + (_datetime64(case["time_2"]) - _datetime64(case["time_1"])) // 2
        lower, upper = center - np.timedelta64(45, "m"), center + np.timedelta64(45, "m")
        native_mask = (native.time >= lower) & (native.time <= upper) & native.valid
        axis.plot(native.lon[native_mask], native.lat[native_mask], ".-", color=".45", label="native QC")
        for rep, color, label in (("interp30", "#1b9e77", "interp 30m"), ("interp60", "#7570b3", "interp 60m")):
            lon, lat, valid, present = _representation_arrays(recon, rep)
            mask = (recon.time >= lower) & (recon.time <= upper) & valid
            if present:
                axis.plot(lon[mask], lat[mask], ".-", color=color, label=label)
        if case["kind"].endswith("_only"):
            rep = case["kind"].removesuffix("_only")
            lon, lat, _, _ = _representation_arrays(recon, rep)
            endpoints = [int(case["row_index_1"]), int(case["row_index_2"])]
        else:
            lon, lat = native.lon, native.lat
            endpoints = [int(case["row_index_1"]), int(case["row_index_2"])]
        axis.plot(lon[endpoints], lat[endpoints], "*-", color="#d95f02", markersize=10,
                  linewidth=2, label="examined edge")
        axis.set(title=f"{case['platform_code']}\n{case['kind']}", xlabel="Longitude", ylabel="Latitude")
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.suptitle("Representative stored trajectory context\n(no diagnostic interpolation)", fontsize=13)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"02_representative_events.{suffix}"
        fig.savefig(path, dpi=200 if suffix == "png" else None)
        paths.append(path.name)
    plt.close(fig)
    return paths


def render_report(summary: dict, by_class: pd.DataFrame, reconstruction_only: pd.DataFrame) -> str:
    scope = summary["scope"]
    title = "# Native-QC jumps versus supplied reconstructed tracks"
    lines = [title, "", (f"**COMPLETE {scope['total_master_platform_count']}-platform diagnostic.**" if scope["complete_master_scope"] else
                          f"**PARTIAL diagnostic: {scope['selected_platform_count']} of {scope['total_master_platform_count']} platforms.**"),
             "", "This is read-only diagnostic output. No source coordinate, mask, flag, or Zarr value was changed.",
             "", "## Native flagged-edge behavior", "",
             "| Classification | Edges |", "|---|---:|"]
    for behavior in BEHAVIORS:
        lines.append(f"| `{behavior}` | {summary['behavior_counts'][behavior]} |")
    stats = summary["complete_case_speed_m_s"]
    lines += ["", f"Complete reconstruction context exists for **{summary['complete_reconstruction_context_edges']}** native flagged edges.",
              "", "| Product | Median m/s | p95 m/s | Maximum m/s |", "|---|---:|---:|---:|"]
    for product in ("qc", "interp30", "interp60"):
        values = stats[product]
        fmt = lambda value: "unavailable" if value is None else f"{value:.6f}"
        lines.append(f"| {product} | {fmt(values['median'])} | {fmt(values['p95'])} | {fmt(values['max'])} |")
    lines += ["", "## Results by previous local-jump class", ""]
    if by_class.empty:
        lines.append("No native flagged edges were selected.")
    else:
        headings = ["Local class", "Edges", *BEHAVIORS]
        lines += ["| " + " | ".join(headings) + " |", "|" + "---|" * len(headings)]
        for row in by_class.itertuples(index=False):
            lines.append("| " + " | ".join([str(row.local_jump_class), str(row.native_flagged_edges),
                         *[str(getattr(row, name)) for name in BEHAVIORS]]) + " |")
    lines += ["", "## Independent reconstructed-track scan", "",
              "Adjacent stored rows are scanned without bridging invalid positions. An event is associated with a native flag when endpoints match within one second or temporal overlap exceeds one second.", "",
              "| Product | >3 m/s edges | Associated | Reconstruction-only | Affected drifters |", "|---|---:|---:|---:|---:|"]
    for representation in REPRESENTATIONS:
        values = summary["reconstructed_track_scans"][representation]
        lines.append(f"| {representation} | {values['total_gt_3_m_s_edges']} | {values['associated_with_native_flag_intervals']} | {values['reconstruction_only_events']} | {values['affected_drifters']} |")
    common = summary["resolved_on_common_available_intervals"]
    lines += ["", f"On intervals available in both products, 30m resolves **{common['interp30']}** and 60m resolves **{common['interp60']}** native flags. Result: `{common['comparison']}`.",
              "", "### Worst reconstruction-only events", ""]
    if reconstruction_only.empty:
        lines.append("None in the selected scope.")
    else:
        lines += ["| Product | Platform | Start UTC | dt min | Speed m/s | Distance m |", "|---|---|---|---:|---:|---:|"]
        worst = reconstruction_only.sort_values("speed_m_s", ascending=False).head(20)
        for row in worst.itertuples(index=False):
            lines.append(f"| {row.representation} | {row.platform_code} | {row.time_1.isoformat()} | {row.dt_minutes:.6f} | {row.speed_m_s:.6f} | {row.distance_m:.3f} |")
    lines += ["", "## Method", "",
              "Native flags are reproduced with the production haversine and actual-time calculation. Native endpoints are matched only to existing reconstruction samples within one second; matched timestamps and offsets are retained. Missing timestamps, duplicate matches, absent products, and invalid coordinates remain unavailable. Reconstruction speeds use the matched stored timestamps. No spline, nearest-coordinate estimate, or other trajectory interpolation is created.",
              "", "The reconstructed-track scan uses consecutive stored rows with valid in-range coordinates and positive elapsed time. Speeds strictly above 3 m/s are flagged."]
    if not scope["complete_master_scope"]:
        lines += ["", "Run without `--platform-code` to produce the complete campaign result."]
    return "\n".join(lines) + "\n"


def _load_classifications(path: Path, selected: set[str]) -> pd.DataFrame:
    frame = (pd.read_parquet(path) if path.suffix.lower() == ".parquet"
             else pd.read_csv(path, dtype={"platform_code": "string"}))
    required = {"platform_code", "row_index_1", "row_index_2", "classification"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Classification table lacks columns: {sorted(missing)}")
    frame["platform_code"] = frame.platform_code.astype(str)
    frame = frame[frame.platform_code.isin(selected)].copy()
    if frame.duplicated(["platform_code", "row_index_1", "row_index_2"]).any():
        raise ValueError("Classification table has duplicate native edges")
    return frame


def run(master: Path, classification_table: Path, output: Path, *,
        platform_codes: list[str] | None = None, figures: bool = True) -> dict:
    master, classification_table, output = map(Path, (master, classification_table, output))
    master_resolved, output_resolved = master.resolve(), output.resolve()
    if output_resolved == master_resolved or master_resolved in output_resolved.parents:
        raise ValueError("Diagnostic output must be outside the master Zarr")
    if not master.exists() or not classification_table.exists():
        raise FileNotFoundError("Master Zarr and prior classification table are required")
    output.mkdir(parents=True, exist_ok=True)
    comparison_parts, scan_parts = [], []
    with xr.open_zarr(master, consolidated=True, chunks=None) as ds:
        platforms = list(ds.platform_code.values.astype(str))
        requested = platforms if platform_codes is None else platform_codes
        if len(set(requested)) != len(requested):
            raise ValueError("Duplicate --platform-code values")
        unknown = sorted(set(requested) - set(platforms))
        if unknown:
            raise ValueError(f"Unknown platform codes: {unknown}")
        selected = set(requested)
        classifications = _load_classifications(classification_table, selected)
        for platform in requested:
            index = platforms.index(platform)
            native, _ = flagged_pairs(load_track(ds, index))
            local = classifications[classifications.platform_code == platform].copy()
            keys = ["platform_code", "row_index_1", "row_index_2"]
            local_columns = keys + ["run_id", "classification", "explained_by_isolated_run_candidate"]
            for column, default in (("run_id", ""), ("explained_by_isolated_run_candidate", False)):
                if column not in local:
                    local[column] = default
            merged = native.merge(local[local_columns], on=keys, how="left", validate="one_to_one")
            if len(merged) != len(native) or merged.classification.isna().any() or len(local) != len(native):
                raise ValueError(f"Prior classifications do not exactly cover native flags for {platform}")
            reconstruction = load_reconstruction(ds, index)
            comparison_parts.append(compare_native_edges(merged, reconstruction))
            for representation in REPRESENTATIONS:
                scan_parts.append(scan_reconstructed_flags(reconstruction, representation, native))
        comparison = pd.concat(comparison_parts, ignore_index=True) if comparison_parts else pd.DataFrame()
        reconstructed = pd.concat(scan_parts, ignore_index=True) if scan_parts else pd.DataFrame()
        reconstruction_only = reconstructed[reconstructed.reconstruction_only_event].copy() if len(reconstructed) else reconstructed.copy()
        summary, by_class = build_summary(comparison, reconstructed, selected_platforms=requested,
                                          total_master_platforms=len(platforms))
        generated_figures = []
        if figures:
            generated_figures += make_scatter(comparison, output)
            generated_figures += make_case_plot(ds, comparison, reconstruction_only, output)
    for frame, name in ((comparison, "jump_interpolation_comparison"),
                        (reconstructed, "all_reconstructed_flags"),
                        (reconstruction_only, "reconstruction_only_flags")):
        write_table(frame, output / name, parquet=True)
    write_table(by_class, output / "summary_by_local_class")
    summary.update({
        "master_zarr": str(master.resolve()), "classification_table": str(classification_table.resolve()),
        "classification_table_sha256": file_sha256(classification_table),
        "master_metadata_sha256": file_sha256(master / ".zmetadata"),
        "diagnostic_code_sha256": file_sha256(Path(__file__)),
        "software_version": __version__, "generated_figures": generated_figures,
        "output_directory": str(output.resolve()),
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(render_report(summary, by_class, reconstruction_only), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    result = argparse.ArgumentParser(description="Compare native jump flags with stored 30m/60m reconstructions.")
    result.add_argument("--master", type=Path, default=root / "data/ARCTERX_MicroSVP_QC.zarr")
    result.add_argument("--classification-table", type=Path, default=root / "data/diagnostics/residual_jumps/local_classification/classified_flagged_edges.parquet")
    result.add_argument("--output-directory", type=Path, default=root / "data/diagnostics/residual_jumps/interpolation_comparison")
    result.add_argument("--platform-code", action="append", help="Restrict to one platform; repeat as needed. Omit for all platforms.")
    result.add_argument("--no-figures", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    summary = run(args.master, args.classification_table, args.output_directory,
                  platform_codes=args.platform_code, figures=not args.no_figures)
    print(json.dumps(summary, indent=2))
    print(f"Interpolation comparison: {args.output_directory.resolve()}")


if __name__ == "__main__":
    main()
