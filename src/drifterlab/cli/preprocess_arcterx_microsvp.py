"""Command-line ARCTERX QC MicroSVP conversion."""

import argparse
import sys

from drifterlab.experiments.arcterx import preprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preserve and audit ARCTERX QC MicroSVP MATLAB trajectories in a master Zarr")
    parser.add_argument("config", help="YAML configuration path; relative data paths resolve beside this file")
    args = parser.parse_args(argv)
    try:
        summary = preprocess(args.config, progress=lambda message: print(message, flush=True))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
