r"""Load Array 3 / Cluster 1 from the trajectory Zarr into pandas.

Run from the repository root with:

    .\.venv\Scripts\python.exe -i .\explore_array3_cluster1.py

The ``-i`` keeps the Python session open after the example plots are closed, so
the variables ``dataset``, ``cluster_dataset``, and ``df`` remain available.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drifterlab.trajectories.product_store import load_trajectory_product


# These settings match configs/arcterx/v2/02_microsvp_analysis.local.yml.
ZARR_PATH = Path(
    r"C:\Users\Jacopo\Documents\DATI\ARCTERX\ARCTERX\Data"
    r"\manual_QC\MicroSVP\02_trajectory_products.zarr"
)
TRAJECTORY_PRODUCT = "qc_30min"  # For example: qc_5min, qc_15min, qc_30min
ARRAY_ID = 3
CLUSTER_ID = "array_003__cluster_001"

# Change this to True if you also want a simple CSV copy beside this script.
EXPORT_CSV = False
CSV_PATH = Path(f"array_003_cluster_001_{TRAJECTORY_PRODUCT}.csv")


def load_cluster_dataframe() -> tuple:
    """Return the complete dataset, selected cluster dataset, metadata, and DataFrame."""
    dataset, product_info = load_trajectory_product(
        ZARR_PATH,
        TRAJECTORY_PRODUCT,
    )

    platform_mask = (
        (dataset["array_id"].values.astype(int) == ARRAY_ID)
        & (dataset["cluster_id"].values.astype(str) == CLUSTER_ID)
    )
    platform_indices = np.flatnonzero(platform_mask)
    if not len(platform_indices):
        raise ValueError(
            f"No platforms found for Array {ARRAY_ID}, cluster {CLUSTER_ID!r}"
        )

    cluster_dataset = dataset.isel(platform=platform_indices)

    # Platform-level columns are automatically repeated for each time row.
    columns = [
        "platform_id",
        "member_id",
        "array_id",
        "cluster_id",
        "deployment_start_time",
        "longitude",
        "latitude",
        "source_gap_minutes",
        "eastward_velocity",
        "northward_velocity",
        "speed",
        "velocity_dt_seconds",
    ]
    frame = cluster_dataset[columns].to_dataframe().reset_index()
    frame = frame.drop(columns="platform")  # Internal xarray row number.
    frame = frame.rename(columns={"time": "time_utc"})
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame["deployment_start_time"] = pd.to_datetime(
        frame["deployment_start_time"], utc=True
    )
    frame["platform_id"] = frame["platform_id"].astype(str)
    frame["array_id"] = frame["array_id"].astype(int)

    # Remove the common Zarr grid rows where this cluster has no position.
    # Delete this line if you specifically want to inspect missing grid rows.
    frame = frame.dropna(subset=["longitude", "latitude"]).copy()
    frame = frame.sort_values(["platform_id", "time_utc"]).reset_index(drop=True)
    return dataset, cluster_dataset, product_info, frame


dataset, cluster_dataset, product_info, df = load_cluster_dataframe()


if __name__ == "__main__":
    print(f"Product: {product_info.name} ({product_info.cadence_minutes}-minute grid)")
    print(f"Platforms: {df['platform_id'].nunique()}")
    print(f"Rows with positions: {len(df):,}")
    print(f"Time range: {df['time_utc'].min()} to {df['time_utc'].max()}")
    print("\nFirst rows:")
    print(df.head().to_string(index=False))
    print(
        "\nNote: eastward_velocity/northward_velocity/speed are the stored "
        "forward velocities in the Zarr. The configured epsilon analysis can "
        "recalculate a different centered velocity."
    )

    if EXPORT_CSV:
        df.to_csv(CSV_PATH, index=False)
        print(f"\nCSV written to: {CSV_PATH.resolve()}")

    # Two deliberately simple examples; edit these interactively as needed.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for platform_id, platform in df.groupby("platform_id", sort=True):
        axes[0].plot(
            platform["longitude"],
            platform["latitude"],
            marker=".",
            markersize=2,
            label=platform_id,
        )
        axes[1].plot(platform["time_utc"], platform["speed"], label=platform_id)

    axes[0].set(title="Trajectories", xlabel="Longitude", ylabel="Latitude")
    axes[0].axis("equal")
    axes[1].set(title="Stored forward speed", xlabel="UTC time", ylabel="Speed [m/s]")
    axes[0].legend(fontsize=7)
    figure.autofmt_xdate()
    figure.tight_layout()
    plt.show()
