"""
Re-derive the auto-cleanup rule against the 6 189 human verdicts, leave-one-
station-out.

``auto_cleanup`` invented its thresholds and trusted them. The review table has
been sitting next to it the whole time with a human verdict on every row, and
nothing was ever fitted against it. Cross-checking the signals it already
computes against those verdicts says two things:

**The strongest signal in the file is being thrown away.** ``flag_isolated`` is
exactly ``n_neighbours == 0`` -- verified, 530 True / 5 659 False, zero
disagreement -- so a count that ranges 0..58 is collapsed to one bit. Graded,
and applied after the time gate, that count is monotone:

    n_neighbours 0     n=384    precision 0.104
    n_neighbours 1     n=437    precision 0.249
    n_neighbours 2-3   n=366    precision 0.577
    n_neighbours 4-7   n=594    precision 0.806
    n_neighbours 8+    n=1790   precision 0.930

AUC 0.880 inside the time window, and the direction holds at 16 of 16 stations.
For comparison, the model's own confidence reaches 0.771 there and
``mahalanobis_d2`` 0.706.

**The order matters.** ``n_neighbours >= 2`` alone is worth almost nothing
(precision 0.465): a nocturnal insect chorus is not temporally isolated, it is
the densest thing in the recording. The time gate has to remove it first, and
then isolation separates what is left.

    no gate                       6 189 detections, 2 535 calls, precision 0.410
    time gate only                3 571 detections, 2 503 calls, precision 0.701
    time gate + n_neighbours>=2   2 750 detections, 2 354 calls, precision 0.856

``n_neighbours`` is computable at detection time -- ``filter_temporal_isolation``
counts same-species detections within +/- 30 s of the same recording, using only
the detector's own output -- so this is a deployable rule and not a retrospective
description.

WHY LEAVE-ONE-STATION-OUT: picking the threshold on all 16 stations and then
reporting what it scores on those same stations is the error this whole branch
exists to fix. Every number below is measured on a station whose data played no
part in choosing the threshold applied to it.

Usage:
    python scripts/calibrate_cleanup.py
    python scripts/calibrate_cleanup.py --min-recall 0.95
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "src"))
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")


def load(review_path):
    d = pd.read_csv(review_path)
    rec = pd.to_datetime(d["timestamp"], format="%Y%m%dT%H%M%S")
    det = rec + pd.to_timedelta(d["start_s"], unit="s")
    d["hour"] = det.dt.hour + det.dt.minute / 60.0
    d["is_call"] = d["verdict"].eq("call")
    return d


def gate_bounds():
    import config
    def _h(s):
        hh, mm = s.split(":")
        return int(hh) + int(mm) / 60.0
    if not (config.TIME_FILTER_START and config.TIME_FILTER_END):
        return None
    return _h(config.TIME_FILTER_START), _h(config.TIME_FILTER_END)


def fit_threshold(train, min_recall):
    """Lowest n_neighbours cut that maximises precision at >= min_recall."""
    best, best_p = 0, -1.0
    n_call = int(train["is_call"].sum())
    if not n_call:
        return 0
    for t in range(0, int(train["n_neighbours"].max()) + 1):
        k = train[train["n_neighbours"] >= t]
        if not len(k):
            break
        if k["is_call"].sum() / n_call < min_recall:
            break
        p = k["is_call"].mean()
        if p > best_p:
            best, best_p = t, p
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", default=REVIEW)
    ap.add_argument("--min-recall", type=float, default=0.90,
                    help="Recall floor the fitted threshold must respect, "
                         "measured on the training stations.")
    args = ap.parse_args()

    if not os.path.exists(args.review):
        sys.exit(f"missing {args.review}")
    d = load(args.review)
    gate = gate_bounds()
    if gate is None:
        sys.exit("TIME_FILTER is disabled in config; the isolation rule is "
                 "worth almost nothing without it (precision 0.465). Set the "
                 "gate first -- see scripts/calibrate_time_gate.py.")
    lo, hi = gate
    d["inside"] = (d["hour"] >= lo) & (d["hour"] < hi)

    C = int(d["is_call"].sum())
    F = len(d) - C
    print(f"{len(d)} reviewed detections, {C} calls, precision {C/len(d):.4f}")
    print(f"time gate {lo:05.2f}-{hi:05.2f} keeps {int(d['inside'].sum())}\n")

    print("flag_isolated is exactly n_neighbours == 0: "
          f"{bool((d['flag_isolated'] == (d['n_neighbours'] == 0)).all())}\n")

    rows = []
    for station, held in d.groupby("site"):
        train = d[(d["site"] != station) & d["inside"]]
        t = fit_threshold(train, args.min_recall)
        ev = held[held["inside"]]
        if not len(ev) or not held["is_call"].sum():
            continue
        kept = ev[ev["n_neighbours"] >= t]
        n_call_all = int(held["is_call"].sum())
        n_fp_all = len(held) - n_call_all
        rows.append({
            "station": station,
            "threshold_fitted_elsewhere": t,
            "detections": len(held),
            "precision_ungated": round(held["is_call"].mean(), 4),
            "precision_gated": round(ev["is_call"].mean(), 4) if len(ev) else None,
            "precision_gated_iso": round(kept["is_call"].mean(), 4) if len(kept) else None,
            "recall": round(int(kept["is_call"].sum()) / n_call_all, 4),
            "fps_removed": (round(1 - (len(kept) - int(kept["is_call"].sum())) / n_fp_all, 4)
                            if n_fp_all else None),
        })

    r = pd.DataFrame(rows)
    print("Held-out per station (threshold chosen on the other 15):")
    print(r.to_string(index=False))

    tot_kept = sum(len(d[(d["site"] == x.station) & d["inside"]
                         & (d["n_neighbours"] >= x.threshold_fitted_elsewhere)])
                   for x in r.itertuples())
    tot_kept_call = sum(int(d[(d["site"] == x.station) & d["inside"]
                             & (d["n_neighbours"] >= x.threshold_fitted_elsewhere)]
                            ["is_call"].sum()) for x in r.itertuples())
    print(f"\nPooled held-out: {tot_kept} kept, {tot_kept_call} calls, "
          f"precision {tot_kept_call/max(tot_kept,1):.4f}, "
          f"recall {tot_kept_call/C:.4f}, "
          f"fps removed {1-(tot_kept-tot_kept_call)/F:.4f}")
    print(f"Macro-average over stations: "
          f"precision {r['precision_gated_iso'].astype(float).mean():.4f}, "
          f"recall {r['recall'].astype(float).mean():.4f}")

    print("\nBaselines on the same population:")
    for name, m in [("no gate", pd.Series(True, index=d.index)),
                    ("time gate only", d["inside"]),
                    ("flag_isolated dropped (n>=1), gated",
                     d["inside"] & (d["n_neighbours"] >= 1))]:
        k = d[m]
        print(f"  {name:38s} n={len(k):5d} precision {k['is_call'].mean():.4f} "
              f"recall {k['is_call'].sum()/C:.4f}")

    print("\nThe existing filters, for comparison:")
    for c in ["flag_mahal", "flag_isolated"]:
        if c not in d:
            continue
        f = d[d[c] == True]  # noqa: E712 -- column is object dtype in some exports
        if not len(f):
            print(f"  {c:16s} flags nothing")
            continue
        lost = int(f["is_call"].sum())
        print(f"  {c:16s} flags {len(f):5d}, of which {lost} "
              f"({f['is_call'].mean():.1%}) are real calls -- "
              f"removes {(len(f)-lost)/F:.1%} of FPs "
              f"at {lost/C:.1%} of calls lost")


if __name__ == "__main__":
    main()
