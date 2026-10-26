"""Replot a published affine array without recalculating trajectory fits."""

import argparse
from pathlib import Path

from drifterlab.plotting.affine_deformation import replot_array


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("array_directory", type=Path)
    args = parser.parse_args(argv)
    print(f"Rendered {replot_array(args.array_directory)} daily affine maps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
