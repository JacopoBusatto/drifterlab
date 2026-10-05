"""Calculate overshoot-aware FSLE spectra from candidate pair trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.fsle import run_fsle_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="atomically replace an existing FSLE output bundle",
    )
    args = parser.parse_args(argv)
    try:
        result = run_fsle_workflow(
            args.config,
            overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"FSLE: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
