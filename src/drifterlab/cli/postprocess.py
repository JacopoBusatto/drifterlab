"""Run configured array-level postprocessing for reconstructed trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.postprocessing import run_postprocessing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace existing products declared by this configuration",
    )
    args = parser.parse_args(argv)
    try:
        result = run_postprocessing(
            args.config,
            overwrite=args.overwrite,
            progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Postprocessing: {exc}\n")
    print(result.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
