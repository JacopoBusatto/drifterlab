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
    summary = result.decision_summary
    print("Position-QC decision summary")
    print(f"  Observations saved in valid span:{summary['exported_observation_count']:>10,}")
    print(f"  Observations trimmed from export:{summary['trimmed_observation_count']:>10,}")
    print(f"  Exact repeats removed:           {summary['exact_repeat_points_removed']:,}")
    print(f"  Short-interval points removed:   {summary['short_interval_points_removed']:,}")
    print(f"  Duplicate-time points removed:   {summary['duplicate_time_points_removed']:,}")
    print(
        "  One-point speed cures:          "
        f"{summary['single_point_speed_cure_events']:,} events / "
        f"{summary['single_point_speed_cure_points']:,} points"
    )
    print("  Multi-point speed cures:")
    histogram = summary.get("speed_cure_block_size_counts", {})
    multi = [(size, counts) for size, counts in histogram.items() if int(size) > 1]
    if multi:
        for size, counts in multi:
            print(
                f"    {int(size)} points: {int(counts['events']):,} events / "
                f"{int(counts['points']):,} points"
            )
    else:
        print("    none")
    print(
        "  Retained at removal limit:      "
        f"{summary['retained_at_removal_limit_events']:,} events"
    )
    print(f"  Human-rejected observations:    {summary['human_rejected_points']:,}")
    print(f"  Local unresolved observations:  {summary['local_unresolved_points']:,}")
    print(f"  Upstream uncertain observations:{summary['upstream_uncertain_points']:>8,}")
    print(f"  Total rejected observations:    {summary['final_rejected_points']:,}")
    print(f"  Detailed summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
