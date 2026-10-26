"""Prepare local MUR L4 subsets for the common SST reader; no downloads."""

import argparse
import json
from pathlib import Path

from drifterlab.workflows.prepare_mur_sst import prepare_mur_sst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Root containing downloaded MUR v4.1 .nc subsets"
    )
    parser.add_argument("--output", type=Path, required=True, help="New prepared NetCDF path")
    parser.add_argument(
        "--start", help="First nominal UTC date, inclusive; default earliest available"
    )
    parser.add_argument("--end", help="Last nominal UTC date, inclusive; default latest available")
    parser.add_argument(
        "--maximum-analysis-error-k",
        type=float,
        default=None,
        help="Optional uncertainty ceiling; unset by default",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the prepared output after successful conversion",
    )
    args = parser.parse_args(argv)
    try:
        result = prepare_mur_sst(
            args.input,
            args.output,
            start=args.start,
            end=args.end,
            maximum_analysis_error_k=args.maximum_analysis_error_k,
            overwrite=args.overwrite,
            progress=lambda value: print(value, flush=True),
        )
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"prepare-mur-sst: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
