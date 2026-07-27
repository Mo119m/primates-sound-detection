"""
Evaluate the automatic cleanup against the manual review labels.

The manual review is done on the raw detections, before cleanup, so the
reviewer's verdicts are ground truth: this script cross-tabulates them with the
cleanup's clean/suspicious split and prints the numbers the manuscript reports
(false positives removed, genuine calls retained, precision before vs. after,
and the share of detections no longer needing review).

Usage:
    python scripts/evaluate_cleanup.py \\
        --review reviews/ \\
        --cleanup data/outputs/auto_cleanup

    # if a blank MANUAL ID means "not reviewed yet" rather than a confirmed call
    python scripts/evaluate_cleanup.py --review reviews/ --cleanup ... --blank-unreviewed
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cleanup_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", required=True,
                    help="Folder (or glob) of the per-site manual-review CSVs.")
    ap.add_argument("--cleanup", required=True,
                    help="Folder holding clean_detections.csv and "
                         "suspicious_detections.csv from run_auto_cleanup().")
    ap.add_argument("--blank-unreviewed", action="store_true",
                    help="Treat a blank MANUAL ID as 'not reviewed' instead of "
                         "a confirmed call.")
    ap.add_argument("--start-tolerance", type=int, default=1,
                    help="Seconds of slack when matching a reviewed clip to a "
                         "detection row (default: 1).")
    ap.add_argument("--out", default=None,
                    help="Directory for the output CSVs (default: --cleanup).")
    args = ap.parse_args()

    matched, ev = cleanup_eval.run(
        args.review, args.cleanup,
        blank_is_confirmed=not args.blank_unreviewed,
        start_tolerance=args.start_tolerance)

    print(cleanup_eval.report_text(matched))

    if len(ev["confusion"]):
        print("\nManual verdict x cleanup verdict:")
        print(ev["confusion"].to_string())
    if len(ev["per_species"]):
        print("\nPer species:")
        print(ev["per_species"].to_string())

    out_dir = args.out or args.cleanup
    os.makedirs(out_dir, exist_ok=True)
    matched.to_csv(os.path.join(out_dir, "cleanup_vs_review.csv"), index=False)
    if len(ev["per_species"]):
        ev["per_species"].to_csv(os.path.join(out_dir, "cleanup_eval_per_species.csv"))
    print(f"\nWrote cleanup_vs_review.csv to {out_dir}/")

    if ev["unmatched"]:
        print(f"\nNOTE: {ev['unmatched']} reviewed detections had no matching "
              f"cleanup row and were excluded. Check that --review and --cleanup "
              f"cover the same stations and the same detection run.")


if __name__ == "__main__":
    main()
