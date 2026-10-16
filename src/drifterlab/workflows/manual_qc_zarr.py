"""Build and atomically publish grouped products from finalized manual QC."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
import zarr

from drifterlab import __version__
from drifterlab.reconstruction import (
    PRODUCT_SPECS,
    ArrayAssignment,
    InitialClusterResult,
    InitialClusterRule,
    assign_initial_clusters,
    assign_start_arrays,
    build_regular_grids,
)
from drifterlab.reconstruction.product_store import (
    PRODUCT_STORE_SCHEMA_VERSION,
    PlatformAssignmentMetadata,
    PlatformProductMetadata,
    create_product_store,
    write_platform_assignment,
    write_platform_metadata,
    write_platform_products,
)
from drifterlab.workflows.reconstruction import (
    AcceptedTrack,
    _build_track_products,
    _load_track,
    _utc,
)


BUILD_MANIFEST_SCHEMA_VERSION = "1.0"
BUILD_ALGORITHM_VERSION = "manual-qc-gap-limited-products-arrays-clusters-v1"
VALID_PRODUCT_STATUSES = {"provisional_partial_qc", "final_complete_qc"}
DEFAULT_CLUSTER_RULE = InitialClusterRule(
    "infer", 21600.0, 500.0, 1000.0, 5,
)


class FinalizedQCInputLike(Protocol):
    """Frozen finalized input supplied by the GUI-thread build plan."""

    platform_code: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class ManualQCProductBuildResult:
    zarr_path: Path
    platform_count: int
    configured_platform_count: int
    product_status: str
    array_count: int
    cluster_count: int
    action: str


def _utc_text(value: np.datetime64) -> str:
    return np.datetime_as_string(
        np.datetime64(value, "ns"),
        unit="s",
        timezone="UTC",
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()

def write_manual_qc_product_store(
    tracks: Sequence[AcceptedTrack],
    path: str | Path,
    *,
    anchor: np.datetime64,
    configured_platform_count: int | None = None,
    product_status: str = "final_complete_qc",
    effective_configuration: Mapping[str, Any] | None = None,
    maximum_adjacent_start_gap_hours: float = 24.0,
    cluster_distance_reference: str = "observed_starts",
    cluster_default_rule: InitialClusterRule = DEFAULT_CLUSTER_RULE,
    cluster_array_overrides: Mapping[int, InitialClusterRule] | None = None,
    chunk_platform: int = 1,
    chunk_time: int = 2016,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Calculate one complete, auditable grouped store at a new path."""

    tracks = tuple(sorted(
        tracks,
        key=lambda item: item.inventory.platform_id,
    ))
    if not tracks:
        raise ValueError("At least one finalized track is required")
    if product_status not in VALID_PRODUCT_STATUSES:
        raise ValueError(f"Unknown product status: {product_status}")

    platforms = [
        item.inventory.platform_id
        for item in tracks
    ]
    if len(set(platforms)) != len(platforms):
        raise ValueError("Finalized platform identifiers must be unique")
    configured_count = (
        len(platforms)
        if configured_platform_count is None
        else int(configured_platform_count)
    )
    if configured_count < len(platforms):
        raise ValueError("Configured platform count is smaller than the included subset")
    if product_status == "final_complete_qc" and configured_count != len(platforms):
        raise ValueError("A final-complete store must include every configured platform")
    if product_status == "provisional_partial_qc" and configured_count <= len(platforms):
        raise ValueError("A provisional store must omit at least one configured platform")

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Product store already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    available_start = min(item.time[0] for item in tracks)
    available_end = max(item.time[-1] for item in tracks)
    grids = build_regular_grids(
        anchor,
        available_start,
        available_end,
    )

    root = create_product_store(
        destination,
        platforms,
        grids,
        chunk_platform=chunk_platform,
        chunk_time=chunk_time,
    )
    report = progress or (lambda _message: None)

    for platform_index, track in enumerate(tracks):
        products = _build_track_products(track, grids)
        write_platform_metadata(
            root,
            platform_index,
            _product_metadata(track),
        )
        write_platform_products(
            root,
            platform_index,
            products,
        )
        report(
            f"Wrote resampled products {platform_index + 1}/{len(tracks)}: "
            f"{track.inventory.platform_id}"
        )

    array_assignments = assign_start_arrays(
        platforms,
        [item.time[0] for item in tracks],
        maximum_adjacent_start_gap_hours=maximum_adjacent_start_gap_hours,
    )
    qc_specs = [item for item in PRODUCT_SPECS if item.source == "qc"]
    if not qc_specs:
        raise ValueError("Cluster assignment requires at least one QC product")
    coordinate_spec = min(qc_specs, key=lambda item: item.cadence_minutes)
    coordinate_group = root["products"][coordinate_spec.name]
    coordinate_grid = np.asarray(coordinate_group["time"][:]).astype(
        "datetime64[ns]"
    )
    active_array_ids = {item.array_id for item in array_assignments}
    configured_overrides = dict(cluster_array_overrides or {})
    active_overrides = {
        array_id: rule
        for array_id, rule in configured_overrides.items()
        if array_id in active_array_ids
    }
    cluster_result = assign_initial_clusters(
        platforms,
        [item.array_id for item in array_assignments],
        [item.time[0] for item in tracks],
        coordinate_grid,
        np.asarray(coordinate_group["longitude"][:], dtype=float),
        np.asarray(coordinate_group["latitude"][:], dtype=float),
        cluster_default_rule,
        active_overrides,
        distance_reference=cluster_distance_reference,
        start_longitude=[item.inventory.first_longitude for item in tracks],
        start_latitude=[item.inventory.first_latitude for item in tracks],
    )
    if len(cluster_result.assignments) != len(tracks):
        raise ValueError("Cluster assignment did not return every included platform")
    for index, (array, cluster) in enumerate(zip(
        array_assignments, cluster_result.assignments, strict=True,
    )):
        write_platform_assignment(
            root,
            index,
            assignment=_assignment_metadata(array, cluster),
        )
    array_count = len(active_array_ids)
    cluster_count = len({item.cluster_id for item in cluster_result.assignments})
    report(
        f"Assigned {len(tracks)} platforms to {array_count} arrays and "
        f"{cluster_count} initial clusters"
    )

    array_metadata = _array_metadata(
        tracks, array_assignments, maximum_adjacent_start_gap_hours,
    )
    cluster_metadata = _cluster_metadata(
        cluster_result,
        distance_reference=cluster_distance_reference,
        coordinate_product=coordinate_spec.name,
        default_rule=cluster_default_rule,
        array_overrides=configured_overrides,
    )
    input_hashes = {
        item.inventory.platform_id: item.inventory.sha256 for item in tracks
    }
    root.attrs.update({
        "algorithm_version": BUILD_ALGORITHM_VERSION,
        "drifterlab_version": __version__,
        "processing_time_utc": datetime.now(timezone.utc).isoformat(),
        "product_status": product_status,
        "assignment_scope_status": (
            "final" if product_status == "final_complete_qc" else "provisional"
        ),
        "configured_grid_anchor": _utc_text(anchor),
        "available_start": _utc_text(available_start),
        "available_end": _utc_text(available_end),
        "included_platform_count": len(platforms),
        "configured_platform_count": configured_count,
        "included_platform_ids": platforms,
        "input_qc_sha256": input_hashes,
        "effective_configuration": dict(effective_configuration or {}),
        "product_names": [item.name for item in PRODUCT_SPECS],
        **array_metadata,
        **cluster_metadata,
    })
    audit_hashes = _write_audit_files(
        destination,
        tracks,
        array_assignments,
        cluster_result,
        product_status=product_status,
    )
    root.attrs.update(audit_hashes)
    manifest = {
        "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
        "product_store_schema_version": PRODUCT_STORE_SCHEMA_VERSION,
        "algorithm_version": BUILD_ALGORITHM_VERSION,
        "product_status": product_status,
        "configured_platform_count": configured_count,
        "included_platform_count": len(platforms),
        "included_platform_ids": platforms,
        "input_qc_sha256": input_hashes,
        "effective_configuration": dict(effective_configuration or {}),
        "array_assignment_sha256": array_metadata["array_assignment_sha256"],
        "initial_cluster_assignment_sha256": cluster_metadata[
            "initial_cluster_assignment_sha256"
        ],
        **audit_hashes,
    }
    root.attrs["build_manifest"] = manifest
    root.attrs["build_manifest_sha256"] = _canonical_sha256(manifest)
    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    zarr.consolidate_metadata(str(destination))
    return destination

def _product_metadata(track: AcceptedTrack) -> PlatformProductMetadata:
    scope = track.inventory.metadata.get("review_scope")
    if not isinstance(scope, dict):
        raise ValueError(
            f"Manual-QC review scope is missing for "
            f"{track.inventory.platform_id}"
        )

    requested = scope.get("requested_review_span_days")
    return PlatformProductMetadata(
        deployment_start_time=_utc(
            scope.get("deployment_start_time"),
            "review_scope.deployment_start_time",
        ),
        deployment_start_longitude=track.inventory.first_longitude,
        deployment_start_latitude=track.inventory.first_latitude,
        trajectory_end_time=track.time[-1],
        review_end_time=_utc(
            scope.get("review_end_time"),
            "review_scope.review_end_time",
        ),
        requested_review_span_days=(
            np.nan if requested is None else float(requested)
        ),
        accepted_review_span_days=float(
            scope["accepted_review_span_days"]
        ),
        review_end_reason=str(scope["review_end_reason"]),
        accepted_fix_count=track.inventory.accepted_count,
        raw_fix_count=track.inventory.native_count,
        source_sha256=track.inventory.source_sha256,
        qc_input_sha256=track.inventory.sha256,
    )


def _array_metadata(
    tracks: Sequence[AcceptedTrack],
    assignments: Sequence[ArrayAssignment],
    maximum_adjacent_start_gap_hours: float,
) -> dict[str, Any]:
    records = [
        {
            "platform_id": track.inventory.platform_id,
            "array_id": assignment.array_id,
            "start_time": _utc_text(track.time[0]),
        }
        for track, assignment in zip(tracks, assignments, strict=True)
    ]
    summaries: list[dict[str, Any]] = []
    for array_id in sorted({item.array_id for item in assignments}):
        members = [
            track
            for track, assignment in zip(tracks, assignments, strict=True)
            if assignment.array_id == array_id
        ]
        summaries.append({
            "array_id": array_id,
            "platform_count": len(members),
            "first_start_time": _utc_text(min(item.time[0] for item in members)),
            "last_start_time": _utc_text(max(item.time[0] for item in members)),
        })
    return {
        "array_assignment_policy": {
            "identifier": "1-based contiguous integer in chronological array order",
            "source": "exact first manually accepted position per platform",
            "ordering": "start timestamp ascending, then platform_id ascending for ties",
            "new_array_rule": "adjacent start-time gap strictly greater than threshold",
            "maximum_adjacent_start_gap_hours": (
                maximum_adjacent_start_gap_hours
            ),
        },
        "array_assignment_sha256": _canonical_sha256(records),
        "array_summary": summaries,
    }


def _cluster_metadata(
    result: InitialClusterResult,
    *,
    distance_reference: str,
    coordinate_product: str,
    default_rule: InitialClusterRule,
    array_overrides: Mapping[int, InitialClusterRule],
) -> dict[str, Any]:
    records = [
        {
            "platform_id": item.platform_id,
            "array_id": item.array_id,
            "cluster_id": item.cluster_id,
            "member_id": item.member_id,
            "cluster_size": item.cluster_size,
            "cluster_assignment_status": item.cluster_assignment_status,
        }
        for item in result.assignments
    ]
    return {
        "initial_cluster_assignment_policy": {
            "distance_reference": distance_reference,
            "coordinate_product": coordinate_product,
            "distance_ellipsoid": "WGS84",
            "observed_start_distance": (
                "WGS84 distance between exact first manually accepted coordinates"
            ),
            "first_common_grid_distance": (
                "WGS84 distance at the first common finite coordinate in the "
                "selected QC product"
            ),
            "trajectory_coverage": "unchanged by cluster assignment",
            "effective_configuration": {
                "distance_reference": distance_reference,
                "defaults": asdict(default_rule),
                "array_overrides": {
                    f"array_{array_id:03d}": asdict(rule)
                    for array_id, rule in sorted(array_overrides.items())
                },
            },
        },
        "initial_cluster_assignment_sha256": _canonical_sha256(records),
        "initial_cluster_summary": list(result.array_summary),
        "initial_cluster_pair_diagnostic_summary": {
            "same_array_pair_count": len(result.pair_diagnostics),
            "candidate_link_count": sum(
                item.candidate_link for item in result.pair_diagnostics
            ),
            "unavailable_assignment_distance_count": sum(
                item.distance_m is None for item in result.pair_diagnostics
            ),
            "unavailable_first_common_grid_distance_count": sum(
                item.first_common_grid_distance_m is None
                for item in result.pair_diagnostics
            ),
        },
    }


def _assignment_metadata(
    array: ArrayAssignment,
    cluster: Any,
) -> PlatformAssignmentMetadata:
    if array.platform_id != cluster.platform_id or array.array_id != cluster.array_id:
        raise ValueError("Array and cluster assignments are not aligned")
    return PlatformAssignmentMetadata(
        array.array_id,
        array.gap_from_previous_start_hours,
        array.gap_to_next_start_hours,
        cluster.cluster_id,
        cluster.member_id,
        cluster.cluster_size,
        cluster.cluster_assignment_status,
        cluster.cluster_start_spread_seconds,
        cluster.cluster_diameter_m,
        cluster.cluster_maximum_available_pair_distance_m,
        cluster.cluster_observed_start_diameter_m,
        cluster.cluster_first_common_grid_diameter_m,
        cluster.cluster_first_common_grid_unavailable_pair_count,
        cluster.cluster_candidate_link_count,
        cluster.cluster_unavailable_pair_count,
    )


def _write_audit_files(
    destination: Path,
    tracks: Sequence[AcceptedTrack],
    arrays: Sequence[ArrayAssignment],
    clusters: InitialClusterResult,
    *,
    product_status: str,
) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    for track, array, cluster in zip(
        tracks, arrays, clusters.assignments, strict=True,
    ):
        cluster_values = asdict(cluster)
        cluster_values.pop("platform_id")
        cluster_values.pop("array_id")
        rows.append({
            "platform_id": track.inventory.platform_id,
            "product_status": product_status,
            "qc_filename": track.inventory.path.name,
            "qc_input_sha256": track.inventory.sha256,
            "source_sha256": track.inventory.source_sha256,
            "deployment_start_utc": _utc_text(track.time[0]),
            "trajectory_end_utc": _utc_text(track.time[-1]),
            "deployment_start_longitude": track.inventory.first_longitude,
            "deployment_start_latitude": track.inventory.first_latitude,
            "accepted_fix_count": track.inventory.accepted_count,
            "raw_fix_count": track.inventory.native_count,
            "array_id": array.array_id,
            "gap_from_previous_start_hours": array.gap_from_previous_start_hours,
            "gap_to_next_start_hours": array.gap_to_next_start_hours,
            **cluster_values,
        })
    report_path = destination / "build_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)

    pair_columns = tuple(asdict(clusters.pair_diagnostics[0])) if (
        clusters.pair_diagnostics
    ) else (
        "platform_a", "platform_b", "array_id",
        "start_time_difference_seconds", "first_common_time",
        "observed_start_distance_m", "first_common_grid_distance_m",
        "distance_reference", "distance_m", "candidate_link",
    )
    pair_rows = [asdict(item) for item in clusters.pair_diagnostics]
    pair_path = destination / "cluster_pair_diagnostics.csv"
    pd.DataFrame(pair_rows, columns=pair_columns).to_csv(pair_path, index=False)
    return {
        "build_report_sha256": _file_sha256(report_path),
        "cluster_pair_diagnostics_sha256": _file_sha256(pair_path),
    }


def _verify_product_store(
    path: Path,
    *,
    platforms: Sequence[str],
    input_hashes: Mapping[str, str],
    configured_platform_count: int,
    product_status: str,
    effective_configuration: Mapping[str, Any],
) -> tuple[int, int]:
    root = zarr.open_group(str(path), mode="r")
    attributes = dict(root.attrs)
    expected_platforms = list(platforms)
    if attributes.get("schema_version") != PRODUCT_STORE_SCHEMA_VERSION:
        raise ValueError("Zarr readback found the wrong product-store schema")
    for name, expected in (
        ("product_status", product_status),
        ("configured_platform_count", configured_platform_count),
        ("included_platform_count", len(expected_platforms)),
        ("included_platform_ids", expected_platforms),
        ("input_qc_sha256", dict(input_hashes)),
        ("effective_configuration", dict(effective_configuration)),
    ):
        if attributes.get(name) != expected:
            raise ValueError(f"Zarr readback changed {name}")
    stored_platforms = np.asarray(root["platform_id"][:]).astype(str).tolist()
    if stored_platforms != expected_platforms:
        raise ValueError("Zarr readback changed platform identifiers")

    array_ids = np.asarray(root["array_id"][:], dtype=np.int32)
    cluster_ids = np.asarray(root["cluster_id"][:]).astype(str)
    member_ids = np.asarray(root["member_id"][:], dtype=np.int32)
    cluster_sizes = np.asarray(root["cluster_size"][:], dtype=np.int32)
    statuses = np.asarray(root["cluster_assignment_status"][:]).astype(str)
    if (
        np.any(array_ids < 1)
        or np.any(member_ids < 1)
        or np.any(cluster_sizes < 1)
        or any(not value for value in cluster_ids)
        or any(not value for value in statuses)
    ):
        raise ValueError("Zarr contains an incomplete array or cluster assignment")

    product_root = root["products"]
    expected_products = {item.name for item in PRODUCT_SPECS}
    if set(product_root.group_keys()) != expected_products:
        raise ValueError("Zarr readback changed the configured product groups")
    expected_variables = {
        "platform_id", "time", "longitude", "latitude", "source_gap_minutes",
        "eastward_velocity", "northward_velocity", "speed", "velocity_dt_seconds",
    }
    for specification in PRODUCT_SPECS:
        group = product_root[specification.name]
        if set(group.array_keys()) != expected_variables:
            raise ValueError(
                f"Zarr readback changed variables for {specification.name}"
            )
        if np.asarray(group["platform_id"][:]).astype(str).tolist() != expected_platforms:
            raise ValueError(
                f"Zarr readback changed platforms for {specification.name}"
            )
        times = np.asarray(group["time"][:]).astype("datetime64[ns]")
        if not len(times) or np.isnat(times).any():
            raise ValueError(f"Zarr has an invalid grid for {specification.name}")
        if len(times) > 1 and np.any(np.diff(times.astype(np.int64)) <= 0):
            raise ValueError(
                f"Zarr grid is not strictly increasing for {specification.name}"
            )
        expected_shape = (len(expected_platforms), len(times))
        for variable in expected_variables - {"platform_id", "time"}:
            if group[variable].shape != expected_shape:
                raise ValueError(
                    f"Zarr has the wrong shape for {specification.name}.{variable}"
                )
    manifest = attributes.get("build_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Zarr is missing its build manifest")
    if attributes.get("build_manifest_sha256") != _canonical_sha256(manifest):
        raise ValueError("Zarr build-manifest hash is invalid")
    manifest_path = path / "build_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Zarr is missing build_manifest.json")
    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Zarr build_manifest.json differs from root metadata")
    for filename, attribute in (
        ("build_report.csv", "build_report_sha256"),
        ("cluster_pair_diagnostics.csv", "cluster_pair_diagnostics_sha256"),
    ):
        audit_path = path / filename
        if (
            not audit_path.is_file()
            or _file_sha256(audit_path) != attributes.get(attribute)
        ):
            raise ValueError(f"Zarr audit file is missing or changed: {filename}")
    return len(set(array_ids.tolist())), len(set(cluster_ids.tolist()))


def _remove_exact_directory(path: Path, *, parent: Path, expected_name: str) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != expected_name:
        raise RuntimeError(f"Refusing cleanup outside {parent}")
    shutil.rmtree(resolved)


def _publish_store(temporary: Path, destination: Path, *, token: str) -> None:
    if not destination.exists():
        temporary.rename(destination)
        return
    if not destination.is_dir():
        raise ValueError(f"Zarr output exists and is not a directory: {destination}")
    backup = destination.with_name(f".{destination.name}.{token}.backup")
    if backup.exists():
        raise FileExistsError(f"Zarr backup path already exists: {backup}")
    destination.rename(backup)
    try:
        temporary.rename(destination)
    except Exception:
        if not destination.exists() and backup.exists():
            backup.rename(destination)
        raise
    _remove_exact_directory(
        backup, parent=destination.parent, expected_name=backup.name,
    )


def _existing_store_matches(
    destination: Path,
    *,
    platforms: Sequence[str],
    input_hashes: Mapping[str, str],
    configured_platform_count: int,
    product_status: str,
    effective_configuration: Mapping[str, Any],
) -> tuple[int, int] | None:
    if not destination.is_dir():
        return None
    try:
        return _verify_product_store(
            destination,
            platforms=platforms,
            input_hashes=input_hashes,
            configured_platform_count=configured_platform_count,
            product_status=product_status,
            effective_configuration=effective_configuration,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None


def build_manual_qc_product_store(
    frozen_inputs: Sequence[FinalizedQCInputLike],
    path: str | Path,
    *,
    anchor: np.datetime64,
    configured_platform_count: int,
    product_status: str,
    effective_configuration: Mapping[str, Any],
    maximum_adjacent_start_gap_hours: float,
    cluster_distance_reference: str,
    cluster_default_rule: InitialClusterRule,
    cluster_array_overrides: Mapping[int, InitialClusterRule],
    chunk_platform: int = 1,
    chunk_time: int = 2016,
    progress: Callable[[str], None] | None = None,
) -> ManualQCProductBuildResult:
    """Build, verify, and failure-safely replace the configured Zarr store."""

    report = progress or (lambda _message: None)
    inputs = tuple(sorted(
        frozen_inputs, key=lambda item: str(item.platform_code),
    ))
    if not inputs:
        raise ValueError("At least one frozen finalized input is required")
    platforms = [str(item.platform_code) for item in inputs]
    if len(set(platforms)) != len(platforms):
        raise ValueError("Frozen build inputs contain duplicate platform identifiers")
    input_hashes = {str(item.platform_code): str(item.sha256) for item in inputs}
    destination = Path(path).resolve()

    existing_counts = _existing_store_matches(
        destination,
        platforms=platforms,
        input_hashes=input_hashes,
        configured_platform_count=configured_platform_count,
        product_status=product_status,
        effective_configuration=effective_configuration,
    )
    if existing_counts is not None:
        report("Validated and reused the existing grouped Zarr")
        return ManualQCProductBuildResult(
            destination,
            len(platforms),
            configured_platform_count,
            product_status,
            existing_counts[0],
            existing_counts[1],
            "reused",
        )

    tracks: list[AcceptedTrack] = []
    for index, frozen in enumerate(inputs, start=1):
        source = Path(frozen.path).resolve()
        if not source.is_file():
            raise ValueError(f"Frozen manual-QC input is missing: {source}")
        if _file_sha256(source) != str(frozen.sha256):
            raise ValueError(
                f"Frozen manual-QC input changed before build: {frozen.platform_code}"
            )
        track = _load_track(source)
        if (
            track.inventory.platform_id != str(frozen.platform_code)
            or track.inventory.sha256 != str(frozen.sha256)
        ):
            raise ValueError(
                f"Frozen manual-QC identity changed: {frozen.platform_code}"
            )
        tracks.append(track)
        report(
            f"Validated finalized manual QC {index}/{len(inputs)}: "
            f"{frozen.platform_code}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = destination.with_name(f".{destination.name}.{token}.tmp")
    try:
        write_manual_qc_product_store(
            tracks,
            temporary,
            anchor=anchor,
            configured_platform_count=configured_platform_count,
            product_status=product_status,
            effective_configuration=effective_configuration,
            maximum_adjacent_start_gap_hours=maximum_adjacent_start_gap_hours,
            cluster_distance_reference=cluster_distance_reference,
            cluster_default_rule=cluster_default_rule,
            cluster_array_overrides=cluster_array_overrides,
            chunk_platform=chunk_platform,
            chunk_time=chunk_time,
            progress=report,
        )
        counts = _verify_product_store(
            temporary,
            platforms=platforms,
            input_hashes=input_hashes,
            configured_platform_count=configured_platform_count,
            product_status=product_status,
            effective_configuration=effective_configuration,
        )
        changed = [
            str(item.platform_code)
            for item in inputs
            if not Path(item.path).is_file()
            or _file_sha256(Path(item.path)) != str(item.sha256)
        ]
        if changed:
            raise ValueError(
                f"Finalized manual-QC inputs changed during build: {changed}"
            )
        _publish_store(temporary, destination, token=token)
        report("Verified and published the grouped Zarr")
    finally:
        if temporary.exists():
            _remove_exact_directory(
                temporary,
                parent=destination.parent,
                expected_name=f".{destination.name}.{token}.tmp",
            )

    return ManualQCProductBuildResult(
        destination,
        len(platforms),
        configured_platform_count,
        product_status,
        counts[0],
        counts[1],
        "built",
    )
