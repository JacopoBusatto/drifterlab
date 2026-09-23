"""Non-destructive, iterative native-position review and bounded-memory export."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr

from drifterlab import __version__
from drifterlab.qc import flags
from drifterlab.qc import review as review_rules
from drifterlab.qc.review import run_class, MAX_LOCAL_GAP_SECONDS, GAP_TOLERANCE_SECONDS
from . import jump_recovery
from .jump_recovery import RECOVERY_CONFIG, SUGGESTION_COLUMNS, SUMMARY_DEFAULTS, suggest_regions

THRESHOLD = 3.0
REVIEW_COLUMNS = ["platform_code", "time", "source_obs_index", "decision", "reason",
                  "review_iteration", "note", "reviewed_at", "source_sha256", "candidate_id"]
QUEUE_COLUMNS = ["candidate_id", "platform_code", "review_iteration", "classification", "reason",
                 "priority", "row_index_start", "row_index_end", "context_start", "context_end",
                 "selected_row", "n_flagged_edges", "max_speed_m_s", "start_time", "end_time",
                 "adjacency", "source_sha256"]
EDGE_COLUMNS = ["candidate_id", "platform_code", "row_index_1", "row_index_2",
                "source_obs_index_1", "source_obs_index_2", "time_1", "time_2",
                "speed_m_s", "dt_seconds", "n_rows_between"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    if not path.exists():
        return ""
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_write(path: Path, write) -> None:
    """Replace one derived file only after its complete temporary file is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8"))


def utc_string(value) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return "NaT"
    return (timestamp.tz_localize("UTC") if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")).isoformat()


def parse_utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("Review times require an explicit UTC offset (Z or +00:00)")
    return timestamp.as_unit("ns")


class Decisions:
    """One latest decision per immutable source row; stale concurrent saves fail."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.sha256 = digest(self.path)
        self.rows: dict[tuple[str, int], dict] = {}
        if not self.path.exists():
            return
        table = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if list(table.columns) != REVIEW_COLUMNS:
            raise ValueError(f"Review CSV must have columns {REVIEW_COLUMNS}")
        for row in table.to_dict("records"):
            row["source_obs_index"] = int(row["source_obs_index"])
            row["review_iteration"] = int(row["review_iteration"])
            if (not row["platform_code"] or row["source_obs_index"] < 0 or row["review_iteration"] < 1
                    or row["decision"] not in {"keep", "reject", "uncertain"}
                    or not row["source_sha256"] or not row["reason"]):
                raise ValueError("Invalid manual position review row")
            row["time"] = parse_utc(row["time"]).isoformat()
            parse_utc(row["reviewed_at"])
            key = row["platform_code"], row["source_obs_index"]
            if key in self.rows:
                raise ValueError(f"Duplicate manual review identity: {key}")
            self.rows[key] = row

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([self.rows[k] for k in sorted(self.rows)], columns=REVIEW_COLUMNS)

    def save(self) -> None:
        if digest(self.path) != self.sha256:
            raise ValueError("Review CSV changed in another session; reopen before saving")
        atomic_write(self.path, lambda p: self.table().to_csv(p, index=False))
        self.sha256 = digest(self.path)

    def set(self, point: pd.Series, decision: str, *, iteration: int, reason: str,
            candidate_id: str, note: str = "") -> bool:
        if decision not in {"keep", "reject", "uncertain"}:
            raise ValueError("Decision must be keep, reject, or uncertain")
        timestamp = parse_utc(utc_string(point.time)).isoformat()
        key = str(point.platform_code), int(point.source_obs_index)
        previous = self.rows.get(key)
        if previous and all(previous[k] == v for k, v in
                            (("decision", decision), ("reason", reason), ("note", note))):
            return False
        self.rows[key] = dict(zip(REVIEW_COLUMNS, [key[0], timestamp, key[1], decision, reason,
                              iteration, note, now(), str(point.source_sha256), candidate_id]))
        return True


def native_frame(ds: xr.Dataset, index: int) -> pd.DataFrame:
    n = int(ds.n_obs_qc.values[index])
    if not 0 <= n <= ds.sizes["obs_qc"]:
        raise ValueError("Invalid native observation count")
    names = [name for name, var in ds.data_vars.items() if var.dims == ("trajectory", "obs_qc")]
    row = ds[names].isel(trajectory=index, obs_qc=slice(0, n)).load()
    frame = pd.DataFrame({name: row[name].values for name in names})
    frame["platform_code"] = str(ds.platform_code.values[index])
    frame["source_sha256"] = str(ds.source_sha256.values[index])
    frame["source_filename"] = str(ds.source_filename.values[index])
    frame["row_index"] = np.arange(n)
    frame["time"] = pd.to_datetime(frame.time_qc, utc=True).astype("datetime64[ns, UTC]")
    frame["source_obs_index"] = frame.source_obs_index_qc.astype("int64")
    frame["position_valid"] = frame.position_valid_qc.astype(bool)
    if frame.source_obs_index.duplicated().any() or (frame.source_obs_index < 0).any():
        raise ValueError("Native source indices must be unique and nonnegative")
    times = frame.time_qc.to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(np.argsort(times, kind="stable"), np.arange(n)):
        raise ValueError("Native observations must be in chronological order")
    return frame


def apply_decisions(frame: pd.DataFrame, decisions: Decisions) -> pd.DataFrame:
    result = frame.copy()
    result["manual_review_decision"] = ""
    result["manual_review_reason"] = ""
    result["manual_review_note"] = ""
    result["manual_review_iteration"] = np.zeros(len(frame), dtype=np.int64)
    result["manual_reviewed_at"] = ""
    if len(frame):
        platform = str(frame.platform_code.iloc[0])
        lookup = dict(zip(frame.source_obs_index, frame.index))
        for (code, source_index), decision in decisions.rows.items():
            if code != platform:
                continue
            if source_index not in lookup:
                raise ValueError(f"Unknown source observation in review: {code}/{source_index}")
            j = lookup[source_index]
            if (decision["time"] != utc_string(frame.at[j, "time"])
                    or decision["source_sha256"] != frame.at[j, "source_sha256"]):
                raise ValueError(f"Review source identity/time mismatch: {code}/{source_index}")
            for target, source in [("manual_review_decision", "decision"), ("manual_review_reason", "reason"),
                                   ("manual_review_note", "note"), ("manual_review_iteration", "review_iteration"),
                                   ("manual_reviewed_at", "reviewed_at")]:
                result.at[j, target] = decision[source]
    result["manual_reject"] = result.manual_review_decision.eq("reject")
    result["manual_uncertain"] = result.manual_review_decision.eq("uncertain")
    result["reviewed_position_valid"] = result.position_valid & ~result.manual_reject
    return result


def edge_arrays(frame: pd.DataFrame, adjacency: str):
    good = frame.reviewed_position_valid.to_numpy() & frame.time.notna().to_numpy()
    indices = np.flatnonzero(good) if adjacency == "surviving" else np.arange(len(frame))
    if adjacency not in {"native", "surviving"}:
        raise ValueError("Unknown adjacency mode")
    time = frame.time_qc.to_numpy(dtype="datetime64[ns]")[indices]
    lon, lat = frame.lon_qc.to_numpy()[indices].copy(), frame.lat_qc.to_numpy()[indices].copy()
    lon[~good[indices]], lat[~good[indices]] = np.nan, np.nan
    return indices, flags.consecutive_speed(time, lon, lat), flags.interval_seconds(time)


def candidates(frame: pd.DataFrame, iteration: int, *, adjacency: str,
               context_points: int = 12, merge_gap_points: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group neighboring runs; local skip-one geometry is advisory only."""
    indices, speeds, dt = edge_arrays(frame, adjacency)
    right = np.flatnonzero(speeds > THRESHOLD)
    if not len(right):
        return pd.DataFrame(columns=QUEUE_COLUMNS), pd.DataFrame(columns=EDGE_COLUMNS)
    platform = str(frame.platform_code.iloc[0])
    time = frame.time_qc.to_numpy(dtype="datetime64[ns]")[indices]
    lon, lat = frame.lon_qc.to_numpy()[indices], frame.lat_qc.to_numpy()[indices]
    good = frame.reviewed_position_valid.to_numpy()[indices] & ~np.isnat(time)
    valid = np.flatnonzero(good)
    gap_limit = MAX_LOCAL_GAP_SECONDS + GAP_TOLERANCE_SECONDS

    def pair(a, b):
        if a is None or b is None or not (good[a] and good[b]):
            return {"usable_context": False, "speed": np.nan}
        elapsed = flags.interval_seconds(time[[a, b]])[1]
        speed = flags.consecutive_speed(time[[a, b]], lon[[a, b]], lat[[a, b]])[1]
        return {"usable_context": bool(np.isfinite(speed) and 0 < elapsed <= gap_limit), "speed": speed}

    runs = []
    for part in np.split(right, np.flatnonzero(np.diff(right) > 1) + 1):
        start, end = int(part[0] - 1), int(part[-1])
        before = valid[valid < start]
        after = valid[valid > end]
        triplets = []
        for b in range(start + 1, end):
            legs = [pair(b - 1, b), pair(b, b + 1), pair(b - 1, b + 1)]
            usable = all(p["usable_context"] for p in legs)
            triplets.append({"triplet_context_usable": usable,
                             "passes_local_skip_one_test": usable and legs[-1]["speed"] <= THRESHOLD})
        category, reason = run_class(len(part), triplets,
                                     pair(int(before[-1]) if len(before) else None, start),
                                     pair(end, int(after[0]) if len(after) else None), dt[part])
        runs.append({"part": part, "start": start, "end": end, "category": category, "reason": reason})
    groups = []
    for run in runs:
        previous = groups[-1][-1] if groups else None
        # Only merge locally neighboring runs with continuous usable intervening time.
        if (previous is not None and run["start"] - previous["end"] <= merge_gap_points
                and all(pair(j - 1, j)["usable_context"]
                        for j in range(previous["end"] + 1, run["start"] + 1))):
            groups[-1].append(run)
        else:
            groups.append([run])
    queue, edges = [], []
    full_valid = np.flatnonzero(frame.reviewed_position_valid & frame.time.notna())
    for number, group in enumerate(groups, 1):
        flagged = np.concatenate([r["part"] for r in group])
        start, end = int(indices[group[0]["start"]]), int(indices[group[-1]["end"]])
        category = group[0]["category"] if len(group) == 1 else "neighboring_excursion_region"
        categories = {r["category"] for r in group}
        priority = (1 if "persistent_high_speed_run" in categories or len(group) > 1 else
                    2 if "isolated_spike_candidate" in categories else 3)
        # All ambiguous edges sort by actual speed; no second flagging threshold.
        before, after = full_valid[full_valid < start], full_valid[full_valid > end]
        lo = int(before[max(0, len(before) - context_points)]) if len(before) else start
        hi = int(after[min(len(after), context_points) - 1]) if len(after) else end
        candidate_id = f"i{iteration:04d}:{platform}:c{number:04d}"
        selected = int(indices[group[0]["start"] + 1]) if category == "isolated_spike_candidate" else start
        queue.append(dict(zip(QUEUE_COLUMNS, [candidate_id, platform, iteration, category,
                          "; ".join(dict.fromkeys(r["reason"] for r in group)), priority, start, end, lo, hi,
                          selected, len(flagged), float(speeds[flagged].max()), utc_string(frame.time.iloc[start]),
                          utc_string(frame.time.iloc[end]), adjacency, frame.source_sha256.iloc[0]])))
        for j in flagged:
            a, b = int(indices[j - 1]), int(indices[j])
            edges.append(dict(zip(EDGE_COLUMNS, [candidate_id, platform, a, b,
                              int(frame.source_obs_index.iloc[a]), int(frame.source_obs_index.iloc[b]),
                              utc_string(frame.time.iloc[a]), utc_string(frame.time.iloc[b]),
                              float(speeds[j]), float(dt[j]), b - a - 1])))
    return pd.DataFrame(queue, columns=QUEUE_COLUMNS), pd.DataFrame(edges, columns=EDGE_COLUMNS)


def master_identity(ds: xr.Dataset, master: Path) -> dict:
    if ds.attrs.get("residual_jump_speed_m_s") != THRESHOLD:
        raise ValueError("Master must use the existing strict >3 m/s threshold")
    platforms = [str(v) for v in ds.platform_code.values]
    if len(set(platforms)) != len(platforms):
        raise ValueError("Duplicate master platform identifiers")
    return {"path": str(master.resolve()), "metadata_sha256": digest(master / ".zmetadata"),
            "source_sha256": dict(zip(platforms, map(str, ds.source_sha256.values))),
            "native_counts": list(map(int, ds.n_obs_qc.values))}


def validate_paths(master: Path, review: Path, directory: Path, output: Path) -> None:
    master, review, directory, output = [p.resolve() for p in (master, review, directory, output)]
    for path in (review, directory, output):
        if path == master or master in path.parents or path in master.parents:
            raise ValueError("Review outputs must be outside the master Zarr")
    if (review == output or directory == review or directory in review.parents
            or directory == output or directory in output.parents
            or review in directory.parents or output in directory.parents):
        raise ValueError("Review CSV, iteration directory and reviewed output must be separate paths")
    if review.suffix.lower() != ".csv" or output.suffix.lower() != ".parquet":
        raise ValueError("Review and reviewed output must be .csv and .parquet files")


def build_iteration(master: Path, review: Path, directory: Path, output: Path, *,
                    export: bool, context_points: int = 12, merge_gap_points: int = 2) -> dict:
    """Initial queue uses native adjacency; apply/export recomputes surviving adjacency."""
    master, review, directory, output = [Path(p).resolve() for p in (master, review, directory, output)]
    validate_paths(master, review, directory, output)
    if context_points < 1 or merge_gap_points < 0:
        raise ValueError("Context must be positive and merge gap nonnegative")
    decisions = Decisions(review)
    state_path = directory / "latest.json"
    old = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    iteration = int(old["review_iteration"]) + 1 if old else 1
    # Skip abandoned/incomplete iteration directories; never overwrite their artifacts.
    while (directory / f"iteration_{iteration:04d}").exists():
        iteration += 1
    iteration_dir = directory / f"iteration_{iteration:04d}"
    adjacency = "surviving" if export or decisions.rows or old else "native"
    queues, edge_tables, proposal_tables = [], [], []
    total = rejected = uncertain = 0
    output.parent.mkdir(parents=True, exist_ok=True) if export else None
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    writer = None
    try:
        with xr.open_zarr(master, consolidated=True, chunks=None) as ds:
            identity = master_identity(ds, master)
            if old and old["master"] != identity:
                raise ValueError("Master differs from saved review session; use a separate review directory")
            if old and old["review_file"] != str(review):
                raise ValueError("Review CSV differs from saved review session")
            unknown = {k[0] for k in decisions.rows} - set(identity["source_sha256"])
            if unknown:
                raise ValueError(f"Review contains unknown platforms: {sorted(unknown)}")
            for index in range(ds.sizes["trajectory"]):
                frame = apply_decisions(native_frame(ds, index), decisions)
                queue, edges = candidates(frame, iteration, adjacency=adjacency,
                                           context_points=context_points, merge_gap_points=merge_gap_points)
                queue, proposals = suggest_regions(frame, queue, edges)
                if len(queue):
                    queues.append(queue)
                    edge_tables.append(edges)
                    proposal_tables.append(proposals)
                total += len(frame)
                rejected += int(frame.manual_reject.sum())
                uncertain += int(frame.manual_uncertain.sum())
                if export:
                    indices, speeds, _ = edge_arrays(frame, "surviving")
                    frame["reviewed_audit_speed_m_s"] = np.nan
                    frame.loc[indices, "reviewed_audit_speed_m_s"] = speeds
                    frame["reviewed_residual_jump_flag"] = frame.reviewed_audit_speed_m_s > THRESHOLD
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                    writer.write_table(table)
            if writer is not None:
                writer.close()
                writer = None
            elif export:
                raise ValueError("Master has no native trajectories to export")
        queue = pd.concat(queues, ignore_index=True) if queues else pd.DataFrame(columns=[*QUEUE_COLUMNS, *SUMMARY_DEFAULTS])
        edges = pd.concat(edge_tables, ignore_index=True) if edge_tables else pd.DataFrame(columns=EDGE_COLUMNS)
        proposals = pd.concat(proposal_tables, ignore_index=True) if proposal_tables else pd.DataFrame(columns=SUGGESTION_COLUMNS)
        queue = queue.sort_values(["priority", "max_speed_m_s", "platform_code", "row_index_start"],
                                  ascending=[True, False, True, True], kind="stable")
        if digest(review) != decisions.sha256:
            raise ValueError("Review CSV changed during candidate generation; retry")
        iteration_dir.mkdir(parents=True)
        queue.to_csv(iteration_dir / "candidates.csv", index=False)
        edges.to_csv(iteration_dir / "flagged_edges.csv", index=False)
        proposals.to_csv(iteration_dir / "recovery_suggestions.csv", index=False)
        decisions.table().to_csv(iteration_dir / "applied_decisions.csv", index=False)
        state = {"review_iteration": iteration, "created_at": now(), "master": identity,
                 "review_file": str(review), "review_sha256": decisions.sha256,
                 "iteration_directory": str(iteration_dir), "adjacency": adjacency,
                 "threshold_m_s": THRESHOLD, "context_points": context_points, "merge_gap_points": merge_gap_points,
                 "drifterlab_version": __version__, "review_code_sha256": digest(Path(__file__)),
                 "classification_code_sha256": digest(Path(review_rules.__file__)),
                 "recovery_configuration": RECOVERY_CONFIG,
                 "recovery_code_sha256": digest(Path(jump_recovery.__file__)),
                 "suggestions_sha256": digest(iteration_dir / "recovery_suggestions.csv"),
                 "high_confidence_suggestion_points": int(proposals.auto_confidence.eq("high").sum()),
                 "audit_code_sha256": digest(Path(flags.__file__)), "native_observations": total,
                 "manual_rejects": rejected, "manual_uncertain": uncertain,
                 "candidate_regions": len(queue), "flagged_edges": len(edges),
                 "reviewed_output": str(output) if export else None,
                 "reviewed_output_sha256": digest(temporary) if export else None,
                 "candidates_sha256": digest(iteration_dir / "candidates.csv"),
                 "edges_sha256": digest(iteration_dir / "flagged_edges.csv")}
        write_json(iteration_dir / "manifest.json", state)
        if export:
            temporary.replace(output)
        write_json(state_path, state)
        return state
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def load_queue(directory: Path):
    state = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
    folder = Path(state["iteration_directory"])
    if (digest(folder / "candidates.csv") != state["candidates_sha256"]
            or digest(folder / "flagged_edges.csv") != state["edges_sha256"]):
        raise ValueError("Candidate queue changed; regenerate it with apply")
    queue = pd.read_csv(folder / "candidates.csv", dtype={"platform_code": str}, keep_default_na=False)
    edges = pd.read_csv(folder / "flagged_edges.csv", dtype={"platform_code": str}, keep_default_na=False)
    return state, queue, edges


def load_suggestions(state: dict) -> pd.DataFrame:
    """Legacy sessions remain manually reviewable until the next suggest/apply."""
    if "suggestions_sha256" not in state:
        return pd.DataFrame(columns=SUGGESTION_COLUMNS)
    path = Path(state["iteration_directory"]) / "recovery_suggestions.csv"
    if digest(path) != state["suggestions_sha256"]:
        raise ValueError("Recovery suggestions changed; regenerate them with suggest")
    return pd.read_csv(path, dtype={"platform_code": str, "auto_block_id": str,
                                    "auto_suggested_reject": bool, "auto_forward_backward_agree": bool},
                       keep_default_na=False, na_values={column: [""] for column in (
                           "auto_bridge_speed_m_s", "auto_continuation_speed_m_s", "auto_anchor_row",
                           "auto_reconnection_row", "auto_continuation_row", "auto_backward_bridge_speed_m_s",
                           "auto_backward_continuation_speed_m_s")})
