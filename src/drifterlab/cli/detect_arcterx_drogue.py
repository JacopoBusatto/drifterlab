"""Run standalone drogue-loss detection on raw ARCTERX MicroSVP files."""

import argparse

from drifterlab.experiments.arcterx.drogue_pipeline import run_drogue_detection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to the standalone drogue-detection YAML")
    parser.add_argument("--overwrite", action="store_true",
                        help="atomically replace an existing automatic result table")
    args = parser.parse_args(argv)
    try:
        table = run_drogue_detection(args.config, overwrite=args.overwrite, progress=print)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(1, f"Drogue detection: {exc}\n")
    detected = int(table.auto_drogue_loss_time.notna().sum())
    print(f"Automatic drogue-loss candidates: {detected}/{len(table)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
