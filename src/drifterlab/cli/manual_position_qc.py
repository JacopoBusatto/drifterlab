"""Launch independent threshold-free manual trajectory QC."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    try:
        from drifterlab.manual_qc.gui import run_manual_qc_gui

        return run_manual_qc_gui(str(args.config))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Manual position QC: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
