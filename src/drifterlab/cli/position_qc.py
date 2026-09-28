"""Run generic native-position QC and optional interactive review."""

from __future__ import annotations

import argparse
from pathlib import Path

from drifterlab.workflows.position import run_position_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--automatic", action="store_true")
    modes.add_argument("--semiautomatic", action="store_true")
    modes.add_argument("--manual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    mode = "automatic" if args.automatic else "semiautomatic" if args.semiautomatic else "manual"
    try:
        result = run_position_workflow(
            args.config, mode, overwrite=args.overwrite, progress=lambda message: print(message, flush=True),
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Position QC: {exc}\n")
    print(
        f"Position QC: {result.observation_count} observations across "
        f"{result.platform_count} platforms; {result.unresolved_count} "
        f"unresolved/uncertain observations in {result.output_directory}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
