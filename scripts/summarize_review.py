"""
Summarize the manual-review CSVs into the paper's field-deployment numbers.

Reviewers produce one Kaleidoscope Pro CSV per site (with a ``MANUAL ID`` column
marking false positives as ``Noise``). Point this script at the folder holding
those CSVs and it prints the aggregate per-site / per-species tallies and writes
them to CSV.

Usage:
    python scripts/summarize_review.py --dir path/to/review_csvs
    python scripts/summarize_review.py --glob "reviews/*IPA*.csv"
    python scripts/summarize_review.py --dir reviews --blank-unreviewed

By default a blank ``MANUAL ID`` counts as a CONFIRMED call (the reviewer only
tags the false positives). Pass --blank-unreviewed if a blank instead means the
clip has not been reviewed yet.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import review_import  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="Folder of review CSVs (searched recursively).")
    src.add_argument("--glob", help="Glob pattern for review CSVs.")
    ap.add_argument("--blank-unreviewed", action="store_true",
                    help="Treat a blank MANUAL ID as 'not yet reviewed' instead "
                         "of a confirmed call.")
    ap.add_argument("--out", default=None,
                    help="Directory to write summary CSVs (default: alongside "
                         "the input).")
    args = ap.parse_args()

    blank_is_confirmed = not args.blank_unreviewed
    source = args.dir or args.glob
    df = review_import.load_review_dir(source, blank_is_confirmed=blank_is_confirmed)

    print(review_import.report_text(df))

    s = review_import.summarize(df)
    print("\nDetections per site x species:")
    print(s["per_site"].to_string())
    print("\nConfirmed calls per site x species:")
    print(s["confirmed_by_site"].to_string())

    out_dir = args.out or (args.dir if args.dir else os.path.dirname(args.glob) or ".")
    os.makedirs(out_dir, exist_ok=True)
    s["per_species"].to_csv(os.path.join(out_dir, "review_per_species.csv"))
    s["per_site"].to_csv(os.path.join(out_dir, "review_per_site.csv"))
    s["confirmed_by_site"].to_csv(os.path.join(out_dir, "review_confirmed_by_site.csv"))
    df.to_csv(os.path.join(out_dir, "review_all_detections.csv"), index=False)
    print(f"\nWrote summary CSVs to {out_dir}/")
    if s["totals"]["unreviewed"]:
        print(f"\nNOTE: {s['totals']['unreviewed']} clips are still unreviewed "
              f"(blank MANUAL ID with --blank-unreviewed). Numbers are not final.")


if __name__ == "__main__":
    main()
