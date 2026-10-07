"""Run trajectory-only Poje et al. method-reproduction products."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.poje import run_poje_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="atomically replace the same deterministic Poje product",
    )
    args = parser.parse_args(argv)
    try:
        result = run_poje_workflow(
            args.config,
            overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Poje analysis: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
