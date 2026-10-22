"""Geographic daily maps of supported spatial epsilon cells."""

from __future__ import annotations

import math
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib
from cartopy.mpl.ticker import LatitudeLocator, LongitudeLocator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import SymLogNorm

from drifterlab.analysis.spatial_grid import SpatialBand, cell_polygons
from drifterlab.epsilon import BinSupportRequirements
from drifterlab.epsilon_figures import EPSILON_LABEL
from drifterlab.trajectories.kinematics import ArrayProjection


def plot_daily_epsilon(
    cells: pd.DataFrame,
    *,
    observations: pd.DataFrame,
    bands: list[SpatialBand],
    projection: ArrayProjection,
    extent: tuple[float, float, float, float],
    title: str,
    start_utc: str,
    end_utc: str,
    output_path: Path,
    color: dict[str, float],
    dpi: int,
    support: BinSupportRequirements | None = None,
    enforce_support: bool = True,
) -> None:
    """Draw available cell estimates without interpolation or empty-cell filling."""
    columns = min(2, len(bands))
    rows = math.ceil(len(bands) / columns)
    map_crs = ccrs.PlateCarree(central_longitude=projection.origin_longitude)
    figure, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 5.5 * rows + 1), squeeze=False,
        subplot_kw={"projection": map_crs}, layout="constrained",
    )
    maximum = color["maximum_absolute_m2_s3"]
    norm = SymLogNorm(
        linthresh=color["linear_threshold_m2_s3"], vmin=-maximum, vmax=maximum, base=10,
    )
    try:
        for axis, band in zip(axes.flat, bands):
            selected = cells[cells.separation_bin_index == band.index]
            supported = selected[selected.support_ok.astype(bool)]
            displayed = selected[
                selected.display_ok.astype(bool) if "display_ok" in selected else selected.support_ok.astype(bool)
            ]
            axis.set_extent([
                extent[0] - projection.origin_longitude, extent[1] - projection.origin_longitude,
                extent[2], extent[3],
            ], crs=map_crs)
            if len(displayed):
                collection = PolyCollection(
                    cell_polygons(displayed, projection), array=displayed.epsilon_eff_m2_s3.to_numpy(dtype=float),
                    cmap="coolwarm", norm=norm, edgecolors="none", antialiaseds=False,
                    transform=ccrs.PlateCarree(), zorder=2,
                )
                axis.add_collection(collection)
            axis.coastlines(resolution="110m", linewidth=0.6, zorder=3)
            gridlines = axis.gridlines(
                draw_labels=True, xlocs=LongitudeLocator(nbins=4), ylocs=LatitudeLocator(nbins=5),
                linewidth=0.4, alpha=0.3, zorder=4,
            )
            gridlines.top_labels = False
            gridlines.right_labels = False
            gridlines.xlabel_style = {"size": 8}
            gridlines.ylabel_style = {"size": 8}
            size = "cell size unavailable"
            if band.cell_size_m is not None:
                size = (
                    f"{band.cell_size_m / 1000:.4g} km cells" if band.cell_size_m >= 1000
                    else f"{band.cell_size_m:.4g} m cells"
                )
            in_band = observations[observations.separation_m >= band.lower_m]
            if band.upper_m is not None:
                in_band = in_band[in_band.separation_m < band.upper_m]
            axis.set_title(
                f"{band.label} | {size}\n"
                f"{len(in_band):,} observations | {in_band.pair_id.nunique():,} pairs | "
                f"{len(displayed):,}/{len(selected):,} occupied cells shown"
                + ("" if enforce_support else f" | {len(supported):,} meet support thresholds"),
                fontsize=9,
            )
            if displayed.empty:
                message = (
                    "No observations" if in_band.empty else
                    "Insufficient support in every occupied cell" if enforce_support else "No finite cell estimates"
                )
                axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", fontsize=9)
        for axis in list(axes.flat)[len(bands):]:
            axis.set_visible(False)
        figure.colorbar(
            ScalarMappable(norm=norm, cmap="coolwarm"), ax=list(axes.flat)[:len(bands)],
            orientation="horizontal", shrink=0.65, pad=0.03, extend="both", label=EPSILON_LABEL,
        )
        figure.suptitle(f"{title}\n[{start_utc}, {end_utc})", fontsize=12)
        requirements = (
            f"Minimum per cell: {support.observations} observations, {support.unique_pairs} pairs, "
            f"{support.platforms} drifters, {support.unique_utc} timestamps\n"
            if support is not None else ""
        )
        if not enforce_support:
            requirements = "Minimum support screening disabled; counts retained\n"
        blank_label = "empty or insufficient support" if enforce_support else "empty or invalid estimate"
        figure.supxlabel(
            requirements + f"Blank: {blank_label} | Fixed region and grids | No interpolation\n"
            "Signed effective transfer from sampled pair increments",
            fontsize=9,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi)
    finally:
        plt.close(figure)
