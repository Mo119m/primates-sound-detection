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


def _copy_disagreement_clips(dis_df, clips_dir, out_dir):
    """Copy each disagreed-on clip into out_dir/<disagreement>/<species>/."""
    import glob
    import shutil

    clips_dir = os.path.expanduser(str(clips_dir))
    index = {}
    for path in glob.glob(os.path.join(clips_dir, "**", "*.wav"), recursive=True):
        index.setdefault(os.path.basename(path), path)

    copied = 0
    for _, r in dis_df.iterrows():
        src = index.get(str(r.get("file", "")))
        if not src:
            continue
        dest_dir = os.path.join(out_dir, str(r["disagreement"]),
                                str(r.get("species", "")))
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
        copied += 1
    return copied


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
    ap.add_argument("--clips-dir", default=None,
                    help="Folder of exported detection clips. When given, the "
                         "clips the cleanup and the reviewer disagree on are "
                         "copied into wrongly_flagged/ and missed/ so they can "
                         "be listened to in bulk.")
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

    pf = cleanup_eval.per_filter_analysis(matched)
    if len(pf):
        print("\nPer filter (lift = P(flag|false positive) / P(flag|call);")
        print("            <= 1 means the filter discards real calls at least")
        print("            as fast as noise and lowers precision):")
        print(pf.to_string())

    out_dir = args.out or args.cleanup
    os.makedirs(out_dir, exist_ok=True)
    matched.to_csv(os.path.join(out_dir, "cleanup_vs_review.csv"), index=False)
    if len(ev["per_species"]):
        ev["per_species"].to_csv(os.path.join(out_dir, "cleanup_eval_per_species.csv"))
    if len(pf):
        pf.to_csv(os.path.join(out_dir, "cleanup_eval_per_filter.csv"))
    print(f"\nWrote cleanup_vs_review.csv to {out_dir}/")

    fc = cleanup_eval.filter_combination_analysis(matched)
    if len(fc):
        print("\nEvery filter combination, from the same run's flags")
        print("(no re-run needed; best precision first):")
        print(fc.to_string())
        fc.to_csv(os.path.join(out_dir, "cleanup_eval_combinations.csv"))

    for col, when in (("confidence", "low"), ("n_neighbours", "low")):
        hs = cleanup_eval.station_holdout_sweep(matched, col, flag_when=when)
        if len(hs):
            print(f"\nHeld-out check for {col}: cutoff chosen on half the")
            print("stations, scored on the other half (an honest estimate,")
            print("unlike a cutoff tuned and reported on the same data):")
            print(hs.to_string())
            hs.to_csv(os.path.join(out_dir, f"cleanup_eval_holdout_{col}.csv"))

    opt = cleanup_eval.optimize_thresholds(matched)
    if len(opt):
        print("\nBest cutoffs keeping >= 95% of genuine calls (the current")
        print("settings are one point on this grid, not necessarily the best):")
        print(opt.to_string())
        opt.to_csv(os.path.join(out_dir, "cleanup_eval_tuned.csv"))

    for col, when in (("mahalanobis_d2", "high"), ("n_neighbours", "low"),
                      ("confidence", "low"), ("softmax_margin", "low"),
                      ("recurrence_knn_dist", "low")):
        sw = cleanup_eval.signal_sweep(matched, col, flag_when=when)
        if len(sw):
            print(f"\nSweeping {col} alone (top 5):")
            print(sw.head(5).to_string())
            sw.to_csv(os.path.join(out_dir, f"cleanup_eval_sweep_{col}.csv"))

    cb = cleanup_eval.confidence_baseline(matched)
    if len(cb):
        print("\nDoes the cleanup beat just dropping the least confident")
        print("detections? (same number discarded, so directly comparable):")
        print(cb.to_string())
        cb.to_csv(os.path.join(out_dir, "cleanup_eval_vs_confidence.csv"))

    va = cleanup_eval.vote_analysis(matched)
    if len(va):
        print("\nRequiring filters to agree before discarding:")
        print(va.to_string())
        va.to_csv(os.path.join(out_dir, "cleanup_eval_votes.csv"))

    ys = cleanup_eval.yamnet_score_sweep(matched)
    if len(ys):
        print("\nYAMNet with a minimum confidence (it always returns a top")
        print("class, so a low score means it did not recognise the sound):")
        print(ys.to_string())
        ys.to_csv(os.path.join(out_dir, "cleanup_eval_yamnet_scores.csv"))

    yc = cleanup_eval.yamnet_class_analysis(matched)
    if len(yc):
        print("\nYAMNet by assigned AudioSet class (highest share of genuine")
        print("calls first -- these are the classes mislabelling the target):")
        print(yc.head(15).to_string())
        yc.to_csv(os.path.join(out_dir, "cleanup_eval_yamnet_classes.csv"))

    dis = cleanup_eval.disagreements(matched)
    dis.to_csv(os.path.join(out_dir, "disagreements.csv"), index=False)
    n_wrong = int((dis["disagreement"] == cleanup_eval.WRONGLY_FLAGGED).sum())
    n_missed = int((dis["disagreement"] == cleanup_eval.MISSED).sum())
    print(f"Wrote disagreements.csv  ({n_wrong} genuine calls the cleanup would "
          f"discard, {n_missed} false positives it let through)")

    if args.clips_dir:
        n_copied = _copy_disagreement_clips(dis, args.clips_dir, out_dir)
        print(f"Copied {n_copied} clips into {out_dir}/wrongly_flagged/ and "
              f"{out_dir}/missed/ for listening.")
        print("Listen to these to understand *why* the cleanup fails. Do not "
              "revise the labels from this subset alone -- correcting only "
              "where the cleanup disagreed biases the ground truth in its "
              "favour.")

    if ev["unmatched"]:
        print(f"\nNOTE: {ev['unmatched']} reviewed detections had no matching "
              f"cleanup row and were excluded. Check that --review and --cleanup "
              f"cover the same stations and the same detection run.")


if __name__ == "__main__":
    main()
