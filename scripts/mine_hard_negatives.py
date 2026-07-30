"""
Build a retraining set from the manual review, balanced across stations.

The pipeline's own hard-negative mining uses the detections its filters flagged.
That is circular -- the model scores its own errors -- and it reaches only the
errors the filters catch. Once a review exists, every confirmed false positive is
a hard negative, established independently of the model, and there are far more
of them.

The catch is that they are not evenly spread. One station supplied 2 370 of the
3 654 confirmed false positives here, because an untrained species was calling
there, so mining in proportion produces a model that has learned that station's
intruder and nothing transferable. This script balances the draw across stations,
spreads it across listening episodes within each station, and -- the part that
makes the result checkable -- lets you hold whole stations out so the retrained
model can be tested on noise it has provably never seen.

    # look before copying: what would be taken from where
    python scripts/mine_hard_negatives.py \\
        --matched data/outputs/auto_cleanup/cleanup_vs_review.csv \\
        --total 2000 --holdout IPA11ST,IPA19ST --dry-run

    # then copy the clips into the training folders
    python scripts/mine_hard_negatives.py \\
        --matched data/outputs/auto_cleanup/cleanup_vs_review.csv \\
        --clips-dir data/outputs/detection_clips \\
        --out data/training/mined_v13 \\
        --total 2000 --holdout IPA11ST,IPA19ST --recover-calls

Choosing the holdout: pick stations with enough false positives to measure a
change but which are not the dominant one -- their whole point is to be an
honest test, and holding out the station with most of the errors would both
weaken training and leave the test unrepresentative. After retraining, pass the
same names to scripts/gate_retrain.py as --mined-from's complement; the gate
reports the gain at unmined stations separately, and that is what decides
whether the full detection run is worth it.
"""
import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import hard_negatives  # noqa: E402


def _clip_index(clips_dir):
    """basename -> path for every exported clip, so a row can be located."""
    clips_dir = os.path.expanduser(str(clips_dir))
    index = {}
    for path in glob.glob(os.path.join(clips_dir, "**", "*.wav"), recursive=True):
        index.setdefault(os.path.basename(path), path)
    return index


def _candidate_names(row):
    """The filenames extract_clips() may have written for this detection."""
    rec = str(row.get("recording", ""))
    start = row.get("start_s")
    conf = row.get("confidence")
    names = []
    if isinstance(row.get("file"), str) and row["file"]:
        names.append(row["file"])
    try:
        s = int(float(start))
        names.append(f"{rec}__t{s:05d}s__conf{float(conf):.2f}.wav")
        # Tolerate a one-second difference in how the start was rounded.
        for d in (-1, 1):
            names.append(f"{rec}__t{s + d:05d}s__conf{float(conf):.2f}.wav")
    except (TypeError, ValueError):
        pass
    return names


def copy_clips(rows, index, dest):
    """Copy each selected row's clip into ``dest``. Returns (copied, missing)."""
    os.makedirs(dest, exist_ok=True)
    copied, missing = 0, []
    for _, r in rows.iterrows():
        src = next((index[n] for n in _candidate_names(r) if n in index), None)
        if src is None:
            missing.append(f"{r.get('recording')}@{r.get('start_s')}")
            continue
        shutil.copy2(src, os.path.join(dest, os.path.basename(src)))
        copied += 1
    return copied, missing


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched", required=True,
                    help="cleanup_vs_review.csv -- reviewed detections with "
                         "verdicts.")
    ap.add_argument("--clips-dir", default=None,
                    help="Folder of exported detection clips. Omit with "
                         "--dry-run.")
    ap.add_argument("--out", default=None,
                    help="Destination root. Negatives go to <out>/hard_negatives/, "
                         "recovered calls to <out>/recovered_calls/.")
    ap.add_argument("--total", type=int, default=2000,
                    help="How many hard negatives to select in all.")
    ap.add_argument("--holdout", default="",
                    help="Comma-separated stations that contribute NOTHING, "
                         "reserved as the transfer test after retraining.")
    ap.add_argument("--per-station-cap", type=int, default=None)
    ap.add_argument("--per-episode-cap", type=int, default=None,
                    help="At most this many clips from any one listening "
                         "episode, so a quota is not spent on one bout.")
    ap.add_argument("--recover-calls", action="store_true",
                    help="Also select confirmed calls to fold back into the "
                         "positive set.")
    ap.add_argument("--recover-below", type=float, default=None,
                    help="With --recover-calls, take only calls the model scored "
                         "below this confidence -- the ones iterative mining "
                         "would otherwise keep losing.")
    ap.add_argument("--recover-per-station", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and copy nothing.")
    args = ap.parse_args()

    matched = pd.read_csv(args.matched)
    if "verdict" not in matched.columns:
        sys.exit("--matched needs a 'verdict' column; pass cleanup_vs_review.csv")
    holdout = [s.strip() for s in args.holdout.split(",") if s.strip()]

    p = hard_negatives.plan(matched, total=args.total, holdout=holdout,
                            per_station_cap=args.per_station_cap,
                            per_episode_cap=args.per_episode_cap, seed=args.seed)
    print(hard_negatives.summarise_text(p, holdout=holdout))
    print()
    print(p.to_string(index=False))

    sel = hard_negatives.select(matched, total=args.total, holdout=holdout,
                                per_station_cap=args.per_station_cap,
                                per_episode_cap=args.per_episode_cap,
                                seed=args.seed)
    recovered = pd.DataFrame()
    if args.recover_calls:
        recovered = hard_negatives.recover_calls(
            matched, max_per_station=args.recover_per_station,
            only_low_confidence=args.recover_below, seed=args.seed)
        print(f"\nConfirmed calls to recover into the positive set: "
              f"{len(recovered)}"
              + (f" (scored below {args.recover_below})" if args.recover_below
                 else ""))

    if args.dry_run:
        print("\n--dry-run: nothing copied. Check the share_taken column -- a "
              "station near 1.0 is exhausted, one far below is being held back "
              "so it cannot dominate.")
        return
    if not (args.clips_dir and args.out):
        sys.exit("--clips-dir and --out are required unless --dry-run")

    index = _clip_index(args.clips_dir)
    if not index:
        sys.exit(f"No .wav clips found under {args.clips_dir}")
    print(f"\nIndexed {len(index)} exported clips.")

    neg_dir = os.path.join(args.out, "hard_negatives")
    n_copied, missing = copy_clips(sel, index, neg_dir)
    print(f"Copied {n_copied}/{len(sel)} hard negatives to {neg_dir}/")

    if len(recovered):
        pos_dir = os.path.join(args.out, "recovered_calls")
        n_pos, missing_pos = copy_clips(recovered, index, pos_dir)
        print(f"Copied {n_pos}/{len(recovered)} recovered calls to {pos_dir}/")
        missing += missing_pos

    os.makedirs(args.out, exist_ok=True)
    sel.to_csv(os.path.join(args.out, "mined_manifest.csv"), index=False)
    p.to_csv(os.path.join(args.out, "mined_plan.csv"), index=False)
    if len(recovered):
        recovered.to_csv(os.path.join(args.out, "recovered_manifest.csv"),
                         index=False)
    if missing:
        print(f"\nWARNING: {len(missing)} selected detections had no matching "
              f"clip file and were skipped. First few: {missing[:5]}")
        print("If that is most of them, --clips-dir is pointing at the wrong "
              "folder, or the clips were exported by a different run.")

    print(f"\nWrote mined_manifest.csv and mined_plan.csv to {args.out}/")
    print("\nNext:")
    print("  1. Retrain with these folded into Background (and the recovered "
          "calls into the positive class).")
    print("  2. Gate the result WITHOUT a full detection run:")
    mined = [s for s in p[p.columns[0]].astype(str)
             if s not in ("TOTAL",) and s not in holdout]
    print(f"       python scripts/gate_retrain.py \\")
    print(f"           --reviewed {args.matched} \\")
    print(f"           --clips-dir {args.clips_dir} --model <new model> \\")
    print(f"           --mined-from {','.join(mined)}")
    if holdout:
        print(f"     The gain at {', '.join(holdout)} is the number that "
              f"decides it -- those stations are in no training set.")


if __name__ == "__main__":
    main()
