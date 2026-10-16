"""Import campaign-supplied automatic-QC MATLAB trajectories to grouped Zarr."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.supplied_qc_zarr import run_supplied_qc_import


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="build and validate a replacement before replacing the existing Zarr",
    )
    args = parser.parse_args(argv)
    try:
        result = run_supplied_qc_import(
            args.config,
            overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (TypeError, ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Supplied-QC import: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

