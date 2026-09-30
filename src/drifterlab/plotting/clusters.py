"""Compact PNG/PDF diagnostics for observed starts and candidate clusters."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from drifterlab.plotting.projections import get_projection


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "cohort"


def _map_axes(config: Any, group: pd.DataFrame):
    figure = plt.figure(figsize=config.figsize)
    axes = figure.add_subplot(
        1, 1, 1,
        projection=get_projection(config.projection, central_longitude=config.central_longitude),
    )
    if config.land:
        axes.add_feature(cfeature.LAND.with_scale("110m"), facecolor="0.92", zorder=0)
    if config.coastlines:
        axes.coastlines(resolution="110m", linewidth=.7, color="0.3")
    if config.gridlines:
        lines = axes.gridlines(draw_labels=True, linestyle="--", alpha=.35, linewidth=.6)
        lines.top_labels = False
        lines.right_labels = False
    if config.extent is not None:
        extent = config.extent
    else:
        longitude = group.observed_start_lon.to_numpy(dtype=float)
        latitude = group.observed_start_lat.to_numpy(dtype=float)
        lon_span = max(float(np.ptp(longitude)), .02)
        lat_span = max(float(np.ptp(latitude)), .02)
        lon_pad = max(.15 * lon_span, .01)
        lat_pad = max(.15 * lat_span, .01)
        extent = (
            max(-180, float(np.min(longitude)) - lon_pad),
            min(180, float(np.max(longitude)) + lon_pad),
            max(-90, float(np.min(latitude)) - lat_pad),
            min(90, float(np.max(latitude)) + lat_pad),
        )
    axes.set_extent(extent, crs=ccrs.PlateCarree())
    return figure, axes


def _inspection_map(group: pd.DataFrame, cohort: str, config: Any):
    figure, axes = _map_axes(config, group)
    transform = ccrs.PlateCarree()
    times = pd.to_datetime(group.observed_start_time, utc=True)
    elapsed = (times - times.min()).dt.total_seconds().to_numpy(dtype=float) / 3600
    plotted = axes.scatter(
        group.observed_start_lon, group.observed_start_lat, c=elapsed,
        transform=transform, cmap="viridis", s=56, edgecolor="black", linewidth=.55,
        zorder=4,
    )
    for row in group.itertuples(index=False):
        axes.annotate(
            str(int(row.map_marker)), (row.observed_start_lon, row.observed_start_lat),
            xytext=(4, 4), textcoords="offset points", fontsize=6.5,
            transform=transform, zorder=5,
        )
    colorbar = figure.colorbar(plotted, ax=axes, pad=.03, shrink=.8)
    colorbar.set_label("Hours after first observed start in cohort")
    axes.set_title(
        f"{cohort}: exact first retained QC positions (n={len(group)})\n"
        "Numbers key to the PDF table; positions are not verified deployments"
    )
    return figure


def _table_pages(pdf: PdfPages, group: pd.DataFrame, cohort: str, config: Any) -> None:
    ordered = group.sort_values("map_marker", kind="stable")
    rows_per_page = 28
    for offset in range(0, len(ordered), rows_per_page):
        page = ordered.iloc[offset:offset + rows_per_page]
        figure, axes = plt.subplots(figsize=config.figsize)
        axes.axis("off")
        values = [[
            int(row.map_marker), str(row.platform_id),
            pd.Timestamp(row.observed_start_time).isoformat().replace("+00:00", "Z"),
            f"{float(row.observed_start_lon):.6f}", f"{float(row.observed_start_lat):.6f}",
        ] for row in page.itertuples(index=False)]
        table = axes.table(
            cellText=values,
            colLabels=["Map", "Platform ID", "Observed start UTC", "Longitude", "Latitude"],
            colLoc="left", cellLoc="left", loc="upper center",
            colWidths=[.07, .25, .32, .16, .16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        axes.set_title(
            f"{cohort}: observed starts {offset + 1}–{offset + len(page)} of {len(ordered)}",
            loc="left", pad=16,
        )
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def _diagnostic_page(
    pdf: PdfPages, neighbors: pd.DataFrame, cohort: str, config: Any,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=config.figsize)
    if neighbors.empty:
        for axis in axes:
            axis.axis("off")
        axes[0].text(.5, .5, "No neighbors: this cohort contains one platform",
                     ha="center", va="center")
    else:
        ranks = sorted(neighbors.neighbor_rank.astype(int).unique())
        distributions = [
            neighbors.loc[neighbors.neighbor_rank.astype(int).eq(rank), "distance_m"].to_numpy(float)
            for rank in ranks
        ]
        axes[0].boxplot(distributions, tick_labels=[str(rank) for rank in ranks], showfliers=True)
        axes[0].set_xlabel("Spatial nearest-neighbor rank")
        axes[0].set_ylabel("WGS84 geodesic distance (m)")
        axes[0].grid(axis="y", alpha=.25)
        scatter = axes[1].scatter(
            neighbors.distance_m, neighbors.start_time_gap_minutes,
            c=neighbors.neighbor_rank, cmap="viridis", s=20, alpha=.75,
        )
        axes[1].set_xlabel("WGS84 geodesic distance (m)")
        axes[1].set_ylabel("Absolute observed-start gap (minutes)")
        axes[1].grid(alpha=.25)
        colorbar = figure.colorbar(scatter, ax=axes[1], pad=.02)
        colorbar.set_label("Neighbor rank")
        summaries = []
        for rank in ranks:
            rows = neighbors.loc[neighbors.neighbor_rank.astype(int).eq(rank)]
            summaries.append(
                f"r{rank}: n={len(rows)}, distance p25/median/p75/max="
                f"{rows.distance_m.quantile(.25):.0f}/{rows.distance_m.median():.0f}/"
                f"{rows.distance_m.quantile(.75):.0f}/{rows.distance_m.max():.0f} m; "
                f"time-gap p25/median/p75/max="
                f"{rows.start_time_gap_minutes.quantile(.25):.1f}/"
                f"{rows.start_time_gap_minutes.median():.1f}/"
                f"{rows.start_time_gap_minutes.quantile(.75):.1f}/"
                f"{rows.start_time_gap_minutes.max():.1f} min"
            )
        figure.text(.01, .01, "\n".join(summaries), ha="left", va="bottom", fontsize=7.5)
    figure.suptitle(
        f"{cohort}: observed-start neighbor diagnostics\n"
        "Empirical ranks only; no within/between-cluster labels are assumed",
        y=.995,
    )
    figure.tight_layout(rect=(0, .12, 1, .96))
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def generate_inspection_figures(
    starts: pd.DataFrame, neighbors: pd.DataFrame, config: Any, output: Path,
) -> tuple[Path, ...]:
    """Create one quick-look PNG and one multipage diagnostic PDF per cohort."""
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for cohort, raw_group in starts.groupby("cohort_id", sort=True):
        cohort = str(cohort)
        group = raw_group.sort_values("map_marker", kind="stable")
        stem = _safe_filename(cohort)
        png = output / f"{stem}__observed_starts.png"
        pdf_path = output / f"{stem}__inspection.pdf"
        figure = _inspection_map(group, cohort, config)
        figure.tight_layout()
        figure.savefig(png, dpi=config.dpi, bbox_inches="tight")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            _table_pages(pdf, group, cohort, config)
            cohort_neighbors = neighbors.loc[neighbors.cohort_id.astype(str).eq(cohort)]
            _diagnostic_page(pdf, cohort_neighbors, cohort, config)
        paths.extend([png, pdf_path])
    return tuple(paths)


def _candidate_map(group: pd.DataFrame, cohort: str, config: Any):
    figure, axes = _map_axes(config, group)
    transform = ccrs.PlateCarree()
    cluster_ids = sorted(group.candidate_cluster_id.astype(str).unique())
    palette = plt.get_cmap("tab20")
    marker = 1
    for number, cluster_id in enumerate(cluster_ids, start=1):
        subset = group.loc[group.candidate_cluster_id.astype(str).eq(cluster_id)]
        color = palette((number - 1) % 20)
        if len(subset) > 1:
            center_lon = float(subset.observed_start_lon.mean())
            center_lat = float(subset.observed_start_lat.mean())
            for row in subset.itertuples(index=False):
                axes.plot(
                    [center_lon, row.observed_start_lon], [center_lat, row.observed_start_lat],
                    transform=transform, color=color, linewidth=.8, alpha=.65, zorder=2,
                )
        axes.scatter(
            subset.observed_start_lon, subset.observed_start_lat, transform=transform,
            color=color, s=58, edgecolor="black", linewidth=.55, zorder=4,
            label=f"C{number:02d} (n={len(subset)})",
        )
        for row in subset.sort_values("platform_id", kind="stable").itertuples(index=False):
            axes.annotate(
                str(marker), (row.observed_start_lon, row.observed_start_lat),
                xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                transform=transform, zorder=5,
            )
            marker += 1
    axes.legend(loc="best", fontsize=7, ncols=2, frameon=True)
    axes.set_title(
        f"{cohort}: candidate deployment clusters from observed starts\n"
        "Space and observed start time constrain membership; trajectory lifetimes are unchanged"
    )
    return figure


def _candidate_tables(
    pdf: PdfPages, members: pd.DataFrame, summary: pd.DataFrame, cohort: str, config: Any,
) -> None:
    ordered = members.sort_values(
        ["candidate_cluster_id", "platform_id"], kind="stable",
    ).reset_index(drop=True)
    ordered["map_marker"] = np.arange(1, len(ordered) + 1)
    for offset in range(0, len(ordered), 28):
        page = ordered.iloc[offset:offset + 28]
        figure, axes = plt.subplots(figsize=config.figsize)
        axes.axis("off")
        values = [[
            int(row.map_marker), str(row.candidate_cluster_id).rsplit("__", 1)[-1],
            str(row.platform_id), str(row.observed_start_time_utc),
        ] for row in page.itertuples(index=False)]
        table = axes.table(
            cellText=values, colLabels=["Map", "Candidate", "Platform ID", "Observed start UTC"],
            colLoc="left", cellLoc="left", loc="upper center",
            colWidths=[.08, .22, .28, .38],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)
        axes.set_title(f"{cohort}: candidate members", loc="left", pad=16)
        figure.tight_layout()
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    figure, axes = plt.subplots(figsize=config.figsize)
    axes.axis("off")
    values = [[
        str(row.candidate_cluster_id).rsplit("__", 1)[-1], int(row.member_count),
        f"{float(row.diameter_m):.1f}",
        f"{float(row.observed_start_spread_minutes):.1f}", str(row.candidate_status),
    ] for row in summary.itertuples(index=False)]
    table = axes.table(
        cellText=values,
        colLabels=["Candidate", "Members", "Diameter (m)", "Start spread (min)", "Status"],
        colLoc="left", cellLoc="left", loc="upper center",
        colWidths=[.24, .13, .18, .22, .18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    axes.set_title(f"{cohort}: candidate-cluster summary", loc="left", pad=16)
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def generate_candidate_figures(
    members: pd.DataFrame, summary: pd.DataFrame, config: Any, output: Path,
) -> tuple[Path, ...]:
    """Create one candidate map PNG and one review PDF per cohort."""
    output.mkdir(parents=True, exist_ok=True)
    included = members.loc[~members.candidate_status.astype(str).eq("excluded_by_review")]
    paths: list[Path] = []
    for cohort, group in included.groupby("cohort_id", sort=True):
        cohort = str(cohort)
        cohort_summary = summary.loc[summary.cohort_id.astype(str).eq(cohort)]
        stem = _safe_filename(cohort)
        png = output / f"{stem}__candidate_clusters.png"
        pdf_path = output / f"{stem}__candidate_clusters.pdf"
        figure = _candidate_map(group, cohort, config)
        figure.tight_layout()
        figure.savefig(png, dpi=config.dpi, bbox_inches="tight")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            _candidate_tables(pdf, group, cohort_summary, cohort, config)
        paths.extend([png, pdf_path])
    return tuple(paths)
