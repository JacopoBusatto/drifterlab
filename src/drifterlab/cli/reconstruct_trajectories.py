"""Reconstruct common-grid trajectories from finalized position-QC Parquets."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.reconstruction import run_reconstruction_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="build and validate a replacement before replacing an existing bundle",
    )
    args = parser.parse_args(argv)
    try:
        result = run_reconstruction_workflow(
            args.config, overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Trajectory reconstruction: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
