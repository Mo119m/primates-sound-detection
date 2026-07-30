"""
Measure field recall: plan a listening sample, then score it.

Precision is known from the review of every detection. Recall is not, and cannot
be, because a call the detector missed produced no clip to review. The only way
to get it is to listen to some continuous audio and write down every call in it.

This script does the two halves of that.

    # 1. Decide what to listen to. Writes the plan and a blank annotation sheet.
    python scripts/recall_sample.py plan \\
        --recordings /path/to/recordings \\
        --segments 40 --segment-s 300 --species Cernic \\
        --out data/outputs/recall

    # How much listening is worth committing? (uses the existing review to get
    # the call density, so the answer is grounded rather than a rule of thumb)
    python scripts/recall_sample.py budget \\
        --review reviews/ --recordings /path/to/recordings

    # 2. After annotating recall_annotations.csv, score it.
    python scripts/recall_sample.py score \\
        --annotations data/outputs/recall/recall_annotations.csv \\
        --detections data/outputs/auto_cleanup/clean_detections.csv \\
        --species Cernic

Filling in the sheet: one row per call heard, ``call_start_s`` and
``call_end_s`` measured from the start of the *recording*. Leave a segment's row
blank if it contains no calls -- do not delete it. That row is the evidence the
segment was listened to, and removing empty segments biases recall upward.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import recall_eval  # noqa: E402


def _station_of(path):
    """Infer the station from a recording path, e.g. .../IPA4ST/xyz.wav."""
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts[:-1]):
        if p.upper().startswith("IPA"):
            return p
    return parts[-2] if len(parts) > 1 else ""


def scan_recordings(root):
    """Durations of every WAV under ``root``, read from the headers."""
    import soundfile as sf

    root = os.path.expanduser(str(root))
    paths = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
                   + glob.glob(os.path.join(root, "**", "*.WAV"), recursive=True))
    rows = []
    for p in paths:
        try:
            info = sf.info(p)
            dur = float(info.frames) / float(info.samplerate)
        except Exception as exc:                       # unreadable / truncated
            print(f"  [skip] {os.path.basename(p)} ({type(exc).__name__})")
            continue
        rows.append({"site": _station_of(p),
                     "recording": os.path.splitext(os.path.basename(p))[0],
                     "duration_s": dur, "path": p})
    return pd.DataFrame(rows)


def cmd_plan(args):
    recs = scan_recordings(args.recordings)
    if not len(recs):
        sys.exit(f"No readable WAV files under {args.recordings}")
    total_h = recs["duration_s"].sum() / 3600.0
    print(f"Found {len(recs)} recordings across {recs['site'].nunique()} "
          f"stations, {total_h:.1f} h of audio.")

    plan = recall_eval.plan_segments(
        recs, n_segments=args.segments, segment_s=args.segment_s,
        seed=args.seed)
    if not len(plan):
        sys.exit(f"No recording is at least {args.segment_s:.0f} s long; "
                 f"lower --segment-s.")

    os.makedirs(args.out, exist_ok=True)
    plan.to_csv(os.path.join(args.out, "recall_plan.csv"), index=False)
    tmpl = recall_eval.annotation_template(plan, species=args.species)
    tmpl.to_csv(os.path.join(args.out, "recall_annotations.csv"), index=False)

    hours = plan["segment_s"].sum() / 3600.0
    print(f"\nPlanned {len(plan)} segments of {args.segment_s:.0f} s across "
          f"{plan['site'].nunique()} stations -- {hours:.2f} h to listen to "
          f"(seed {args.seed}).")
    print(plan.groupby("site").size().to_string())
    print(f"\nWrote recall_plan.csv and recall_annotations.csv to {args.out}/")
    print("\nAnnotate recall_annotations.csv: one row per call heard, times in "
          "seconds from the START OF THE RECORDING. Keep the row for a segment "
          "with no calls -- deleting it biases recall upward.")


def cmd_budget(args):
    """How much listening buys how tight an interval."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import cleanup_eval

    calls_per_hour = args.calls_per_hour
    if calls_per_hour is None:
        if not (args.review and args.recordings):
            sys.exit("give --calls-per-hour, or both --review and --recordings "
                     "so it can be estimated")
        rev = cleanup_eval.review_import.load_review_dir(args.review)
        n_calls = int((rev["verdict"] == "call").sum()) if "verdict" in rev else 0
        recs = scan_recordings(args.recordings)
        hours = recs["duration_s"].sum() / 3600.0
        if not hours or not n_calls:
            sys.exit("could not estimate a call density from those inputs")
        # Confirmed detections per hour of audio. This is a *lower* bound on the
        # true call density, since it counts only calls the detector found -- so
        # the listening estimate below is conservative in the useful direction.
        calls_per_hour = n_calls / hours
        print(f"From the review: {n_calls} confirmed detections over "
              f"{hours:.1f} h = {calls_per_hour:.1f} per hour.")
        print("That counts only calls the detector found, so the true density "
              "is higher and the estimates below are conservative.\n")

    rows = []
    for hw in (0.10, 0.05, 0.03):
        need = recall_eval.calls_needed_for_ci(
            half_width=hw, expected_recall=args.expected_recall)
        got = recall_eval.segments_needed(need, calls_per_hour,
                                          segment_s=args.segment_s)
        rows.append({"interval": f"+/-{hw:.0%}", **got})
    print(f"At an expected recall of {args.expected_recall:.0%}, "
          f"{calls_per_hour:.1f} calls/h:")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nPick a row you can afford. The widest interval is usually enough "
          "to say whether recall is near 0.9 or near 0.6, which is the "
          "question a reader actually has.")


def cmd_score(args):
    ann = pd.read_csv(args.annotations)
    det = pd.concat([pd.read_csv(p) for p in _expand(args.detections)],
                    ignore_index=True)
    print(f"{len(ann)} annotation rows, {len(det)} detections.")

    r = recall_eval.score_recall(ann, det, min_overlap_s=args.min_overlap_s,
                                 window_s=args.window_s, species=args.species)
    if r["recall"] is None:
        sys.exit("No annotated calls found. Fill in call_start_s in the sheet.")

    print(f"\nExhaustively annotated : {r['segments']} segments, "
          f"{r['audio_s'] / 3600.0:.2f} h")
    print(f"Calls heard            : {r['calls']}")
    print(f"Calls the detector found: {r['detected']}")
    print(f"\nFIELD RECALL: {r['recall']:.1%}  "
          f"(95% CI {r['ci_low']:.1%} to {r['ci_high']:.1%})")
    if len(r["per_station"]):
        print("\nPer station (intervals are wide on small samples -- read them):")
        print(r["per_station"].to_string(index=False))

    out_dir = args.out or os.path.dirname(os.path.abspath(args.annotations))
    os.makedirs(out_dir, exist_ok=True)
    r["annotated_calls"].to_csv(
        os.path.join(out_dir, "recall_scored_calls.csv"), index=False)
    if len(r["per_station"]):
        r["per_station"].to_csv(
            os.path.join(out_dir, "recall_per_station.csv"), index=False)
    print(f"\nWrote recall_scored_calls.csv to {out_dir}/  (the 'detected' "
          f"column marks which calls were missed -- listen to those to see "
          f"what the detector is systematically losing)")


def _expand(pattern):
    paths = sorted(glob.glob(os.path.expanduser(pattern)))
    if not paths and os.path.isfile(os.path.expanduser(pattern)):
        paths = [os.path.expanduser(pattern)]
    if not paths:
        sys.exit(f"no detection CSV matched {pattern}")
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="choose segments to annotate exhaustively")
    p.add_argument("--recordings", required=True, help="Folder of WAV recordings.")
    p.add_argument("--segments", type=int, default=40)
    p.add_argument("--segment-s", type=float, default=300.0)
    p.add_argument("--species", default="")
    p.add_argument("--seed", type=int, default=0,
                   help="Fixed so the same sample can be regenerated.")
    p.add_argument("--out", default="data/outputs/recall")
    p.set_defaults(func=cmd_plan)

    b = sub.add_parser("budget", help="how much listening buys how tight an interval")
    b.add_argument("--review", default=None)
    b.add_argument("--recordings", default=None)
    b.add_argument("--calls-per-hour", type=float, default=None)
    b.add_argument("--expected-recall", type=float, default=0.9)
    b.add_argument("--segment-s", type=float, default=300.0)
    b.set_defaults(func=cmd_budget)

    s = sub.add_parser("score", help="score a completed annotation sheet")
    s.add_argument("--annotations", required=True)
    s.add_argument("--detections", required=True,
                   help="Detection CSV, or a glob of several.")
    s.add_argument("--species", default=None)
    s.add_argument("--min-overlap-s", type=float,
                   default=recall_eval.DEFAULT_MIN_OVERLAP_S)
    s.add_argument("--window-s", type=float, default=2.0)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
