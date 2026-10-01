"""Orchestrate enabled postprocessing blocks without modifying trajectory inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4

from drifterlab import __version__

from .config import load_postprocessing_config
from .trajectories import load_array_trajectories


POSTPROCESSING_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "postprocessing_manifest.json"


@dataclass(frozen=True)
class PostprocessingResult:
    output_directory: Path
    manifest_path: Path | None
    array_count: int
    figure_paths: tuple[Path, ...]
    movie_paths: tuple[Path, ...]
    status: str

    def format(self) -> str:
        if self.status == "disabled":
            return "Postprocessing: no enabled analysis blocks"
        return "\n".join([
            f"Postprocessing complete: {self.array_count} arrays",
            f"Trajectory figures: {len(self.figure_paths)}",
            f"Trajectory movies: {len(self.movie_paths)}",
            f"Output: {self.output_directory}",
            f"Manifest: {self.manifest_path}",
        ])


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def run_postprocessing(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> PostprocessingResult:
    """Run all enabled postprocessing blocks in their declared workflow order."""
    config = load_postprocessing_config(config_path)
    report = progress or (lambda message: None)
    if not config.trajectory_plotting.enabled:
        return PostprocessingResult(
            config.output_directory, None, 0, (), (), "disabled",
        )

    report(f"Opening trajectory product: {config.trajectory_path}")
    arrays, input_metadata = load_array_trajectories(config)
    report(f"Validated {len(arrays)} independent experimental arrays")

    figure_paths = tuple(
        config.output_directory / f"array_{item.array_id:03d}" /
        "trajectory_plotting" / "trajectories.png"
        for item in arrays
    )
    movie_paths = (
        tuple(
            config.output_directory / f"array_{item.array_id:03d}" /
            "trajectory_plotting" / f"trajectories.{config.trajectory_plotting.movie.format}"
            for item in arrays
        )
        if config.trajectory_plotting.movie.enabled else ()
    )
    manifest_path = config.output_directory / MANIFEST_NAME
    planned = (*figure_paths, *movie_paths, manifest_path)
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        preview = ", ".join(str(path) for path in existing[:3])
        more = " ..." if len(existing) > 3 else ""
        raise ValueError(
            f"Postprocessing outputs already exist: {preview}{more}; "
            "rerun with --overwrite to replace these products"
        )

    try:
        from drifterlab.plotting.array_trajectories import (
            render_array_trajectory_figure,
            render_array_trajectory_movie,
        )
    except ImportError as exc:
        raise ImportError(
            "Trajectory postprocessing requires the optional plotting dependencies; "
            "install drifterlab[plotting]"
        ) from exc

    array_manifest: list[dict] = []
    for index, item in enumerate(arrays):
        figure_path = figure_paths[index]
        report(f"Creating Array {item.array_id} trajectory figure")
        render_array_trajectory_figure(item, config.trajectory_plotting, figure_path)
        if config.trajectory_plotting.movie.enabled:
            movie_path = movie_paths[index]
            report(f"Creating Array {item.array_id} trajectory movie")
            render_array_trajectory_movie(
                item, config.trajectory_plotting, movie_path, progress=report,
            )
        array_manifest.append({
            "array_id": item.array_id,
            "platform_count": len(item.platform_ids),
            "time_start": _format_utc(item.times[0]),
            "time_end": _format_utc(item.times[-1]),
            "time_reference": _format_utc(item.reference_time),
            "time_reference_kind": item.reference_kind,
            "figure": str(figure_path),
            "movie": str(movie_paths[index]) if config.trajectory_plotting.movie.enabled else None,
        })

    manifest = {
        "schema_version": POSTPROCESSING_SCHEMA_VERSION,
        "drifterlab_version": __version__,
        "processing_time_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_path": str(config.source_path),
        "effective_configuration": config.effective(),
        "input": {
            "trajectory_path": str(config.trajectory_path),
            "consolidated_metadata_sha256": _file_sha256(
                config.trajectory_path / ".zmetadata",
            ),
            **asdict(input_metadata),
        },
        "analyses": {"trajectory_plotting": {"enabled": True, "arrays": array_manifest}},
    }
    _write_json_atomic(manifest_path, manifest)
    report(f"Created postprocessing manifest: {manifest_path}")
    return PostprocessingResult(
        config.output_directory, manifest_path, len(arrays),
        figure_paths, movie_paths, "complete",
    )


def _format_utc(value) -> str:
    import pandas as pd

    return pd.Timestamp(value, tz="UTC").isoformat().replace("+00:00", "Z")
