"""Compare manual and automatic grouped trajectory products on native support."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.qc_comparison import run_qc_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace the configured comparison output",
    )
    args = parser.parse_args(argv)
    try:
        result = run_qc_comparison(
            args.config,
            overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (TypeError, ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"QC comparison: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

