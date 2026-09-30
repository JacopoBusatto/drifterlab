"""Inspect observed starts or build candidate deployment clusters."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.clusters import run_cluster_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--inspection", action="store_true",
        help="write cohort/start diagnostics without assigning candidate clusters",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="atomically replace the selected inspection or candidate output",
    )
    args = parser.parse_args(argv)
    try:
        result = run_cluster_workflow(
            args.config, inspection=args.inspection, overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Candidate clusters: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
