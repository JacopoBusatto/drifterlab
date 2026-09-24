"""Run generic drogue detection and optional review from an experiment config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml


RESOLVED_AUTOMATIC_STATUSES = {
    "clear_agreement", "clear_ttff_only", "clear_strain_only",
}


def _experiment(config_path: str | Path) -> str:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Drogue configuration must be a mapping")
    experiment = data.get("experiment")
    if not isinstance(experiment, str) or not experiment.strip():
        raise ValueError("Drogue configuration requires a nonempty experiment")
    return experiment.strip().lower()


def platforms_requiring_review(automatic: pd.DataFrame) -> list[str]:
    """Return unresolved/conflicting platforms in automatic-table order."""
    required = {"platform_code", "auto_status", "auto_drogue_loss_time"}
    missing = required - set(automatic)
    if missing:
        raise ValueError(f"Automatic result is missing columns: {sorted(missing)}")
    attention = (
        ~automatic.auto_status.isin(RESOLVED_AUTOMATIC_STATUSES)
        | automatic.auto_drogue_loss_time.isna()
    )
    return automatic.loc[attention, "platform_code"].astype(str).tolist()


def run_drogue_workflow(
    config_path: str | Path,
    mode: str,
    *,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    reviewer_factory: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Run one detection pass and optionally review the appropriate platforms."""
    report = progress or (lambda message: None)
    if mode not in {"automatic", "semiautomatic", "manual"}:
        raise ValueError(f"Unsupported drogue workflow mode: {mode!r}")
    experiment = _experiment(config_path)
    if experiment != "arcterx":
        raise ValueError(
            f"Unsupported drogue experiment {experiment!r}; currently implemented: arcterx"
        )

    from drifterlab.experiments.arcterx.drogue_config import (
        DROGUE_MODE_LABELS,
        load_drogue_config,
    )
    from drifterlab.experiments.arcterx.drogue_pipeline import (
        load_drogue_detection,
        run_drogue_detection,
    )

    config = load_drogue_config(config_path).for_mode(mode)
    label = DROGUE_MODE_LABELS[mode]
    if mode != "automatic" and config.automatic_output.exists() and not overwrite:
        report(f"Reusing {label}-mode automatic results: {config.automatic_output}")
        automatic = load_drogue_detection(config.automatic_output)
    else:
        if mode == "automatic" and config.automatic_output.exists() and not overwrite:
            raise FileExistsError(
                f"Automatic output already exists: {config.automatic_output}; "
                "use --overwrite"
            )
        action = "Regenerating" if config.automatic_output.exists() else "Creating"
        report(f"{action} {label}-mode automatic results: {config.automatic_output}")
        automatic = run_drogue_detection(
            config_path, overwrite=overwrite, mode=mode, progress=report,
        )
    detected = int(automatic.auto_drogue_loss_time.notna().sum())
    report(f"Automatic drogue-loss decisions: {detected}/{len(automatic)}")
    if mode == "automatic":
        return automatic

    if reviewer_factory is None:
        from drifterlab.experiments.arcterx.drogue_reviewer import DrogueLossReviewer
        reviewer_factory = DrogueLossReviewer

    platform_codes: list[str] | None = None
    if mode == "semiautomatic":
        from drifterlab.experiments.arcterx.drogue_loss_review import DrogueLossReviews

        reviews = DrogueLossReviews(
            config.review_output,
            cutoff_margin_hours=config.analysis_cutoff_margin_hours,
        )
        stale = reviews.validate_against_automatic(automatic)
        problematic = set(platforms_requiring_review(automatic))
        platform_codes = [
            platform for platform in automatic.platform_code.astype(str)
            if platform in stale or (platform in problematic and platform not in reviews.rows)
        ]
        if not platform_codes:
            report(f"No unfinished {mode} drogue reviews remain.")
            return automatic
        report(f"Opening reviewer for {len(platform_codes)} problematic cases.")
    else:
        from drifterlab.experiments.arcterx.drogue_loss_review import DrogueLossReviews

        reviews = DrogueLossReviews(
            config.review_output,
            cutoff_margin_hours=config.analysis_cutoff_margin_hours,
        )
        stale = reviews.validate_against_automatic(automatic)
        pending = [
            platform for platform in automatic.platform_code.astype(str)
            if platform not in reviews.rows or platform in stale
        ]
        if not pending:
            report(f"No unfinished {mode} drogue reviews remain.")
            return automatic
        report(f"Opening reviewer with {len(pending)} of {len(automatic)} platforms pending.")

    reviewer_factory(config_path, mode=mode, platform_codes=platform_codes).show()
    return automatic


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
        help="replace this mode's existing automatic result table",
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
