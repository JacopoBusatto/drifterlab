"""Run generic drogue detection and optional review."""

import argparse

from drifterlab.workflows.drogue import run_drogue_workflow


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("config", help="Path to a drogue workflow YAML")
    explicit = result.add_mutually_exclusive_group()
    explicit.add_argument("--automatic", action="store_true", help="run detection without GUI")
    explicit.add_argument(
        "--semiautomatic", action="store_true",
        help="review only unresolved/problematic cases",
    )
    explicit.add_argument("--manual", action="store_true", help="review every platform")
    result.add_argument("--mode", choices=("automatic", "semiautomatic", "manual"))
    result.add_argument(
        "--overwrite", action="store_true",
        help="replace the existing automatic result table",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    aliases = [
        name for name in ("automatic", "semiautomatic", "manual")
        if getattr(args, name)
    ]
    selected = [*aliases, *([args.mode] if args.mode else [])]
    if len(selected) != 1:
        argument_parser.error(
            "choose exactly one of --automatic, --semiautomatic, --manual, or --mode"
        )
    try:
        run_drogue_workflow(
            args.config, selected[0], overwrite=args.overwrite, progress=print,
        )
    except (ValueError, OSError, ImportError, TypeError, KeyError) as exc:
        argument_parser.exit(1, f"Drogue workflow: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
