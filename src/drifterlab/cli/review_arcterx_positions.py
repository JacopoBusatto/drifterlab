"""Launch manual point review or apply decisions and regenerate the candidate queue."""

import argparse
from pathlib import Path

from drifterlab.experiments.arcterx.position_review import build_iteration, load_queue, validate_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["review", "suggest", "apply"],
                        help="review opens the GUI; suggest regenerates candidates/proposals; apply also exports reviewed QC")
    parser.add_argument("--master", type=Path, default=Path("data/ARCTERX_MicroSVP_QC.zarr"))
    parser.add_argument("--review-file", type=Path, default=Path("data/review/microsvp_manual_position_review.csv"))
    parser.add_argument("--queue-directory", type=Path, default=Path("data/review/microsvp_position_review"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ARCTERX_MicroSVP_reviewed_QC.parquet"))
    parser.add_argument("--context-points", type=int, default=12, help="valid points before/after each region when building a queue")
    parser.add_argument("--merge-gap-points", type=int, default=2, help="maximum intervening edges joining nearby runs (default 2)")
    args = parser.parse_args(argv)
    try:
        validate_paths(args.master, args.review_file, args.queue_directory, args.output)
        if args.mode in {"apply", "suggest"} or not (args.queue_directory / "latest.json").exists():
            print("Building candidate queue" + (" and reviewed Parquet..." if args.mode == "apply" else "..."), flush=True)
            state = build_iteration(args.master, args.review_file, args.queue_directory, args.output,
                                     export=args.mode == "apply", context_points=args.context_points,
                                     merge_gap_points=args.merge_gap_points)
            print(f"Iteration {state['review_iteration']}: {state['candidate_regions']} regions, "
                  f"{state['flagged_edges']} flagged edges; {state['manual_rejects']} explicit rejects.")
            print(f"Queue: {state['iteration_directory']}")
            print(f"High-confidence suggested points: {state['high_confidence_suggestion_points']} (not accepted).")
            if args.mode == "apply":
                print(f"Reviewed QC: {state['reviewed_output']}")
                return 0
            if args.mode == "suggest":
                return 0
        state, queue, edges = load_queue(args.queue_directory)
        if "suggestions_sha256" not in state:
            print("This saved queue predates recovery suggestions. Run 'suggest' to generate them; manual review still works.")
        if queue.empty:
            print("No candidate regions remain in this queue. Apply again after changing decisions to regenerate it.")
            return 0
        from drifterlab.experiments.arcterx.position_reviewer import PositionReviewer
        reviewer = PositionReviewer(args.master, args.review_file, args.queue_directory, state, queue, edges)
        reviewer.show()
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Position review: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
