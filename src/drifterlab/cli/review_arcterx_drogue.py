"""Deprecated ARCTERX-only reviewer; use drifterlab-drogue --manual."""

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the standalone drogue-detection YAML")
    args = parser.parse_args(argv)
    try:
        from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer
        DrogueLossReviewer(args.config).show()
    except (ValueError, OSError, ImportError, TypeError, KeyError) as exc:
        parser.exit(1, f"Drogue review: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
