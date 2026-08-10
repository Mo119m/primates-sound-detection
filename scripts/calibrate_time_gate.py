"""
Re-derive TIME_FILTER_START / TIME_FILTER_END against the reviewed detections.

The committed window was 05:30-10:30. Nothing in the repo records where it came
from, and it was never checked against the review: it keeps 1 264 of the 6 189
reviewed detections and only **40.2 % of the confirmed calls**. A gate whose job
is to remove false positives cheaply cannot be allowed to discard three fifths
of the recall on the way.

Sweeping every (start, end) hour pair against
``data/outputs/auto_cleanup/cleanup_vs_review.csv`` and keeping only windows that
retain >= 95 % of the confirmed calls, the optimum is **05:00-19:00**:

    no gate       6 189 detections, 2 535 calls, precision 0.4096
    05:00-19:00   3 571 detections, 2 503 calls, precision 0.7009
    dropped       2 618 detections,    32 calls, precision 0.0122

70.8 % of the false positives for 1.3 % of the calls. The removed mass is a
nocturnal insect chorus rather than anything the classifier was designed against
-- hour 03 alone contributes 1 641 detections and not one call, and IPA4ST, the
station with the worst precision in the deployment (0.0405), is not a bad
station at all: gated to daytime it scores 0.6947, in line with the median.

As a discriminator the single binary "is it daytime" reaches AUC 0.847, above
the model's own confidence (0.730) and well above raw hour (0.687).

TWO CAVEATS, both of which belong in the manuscript:

1. **32 confirmed calls occur between 19:00 and 05:00.** A hard gate deletes
   them. Any claim about diel calling patterns has to be measured with the gate
   OFF, otherwise it recovers the window that was imposed.
2. The review spans **four consecutive days** (2021-02-22 .. 2021-02-25). This
   window is calibrated on late February only; nothing here supports a seasonal
   claim, and the insect chorus that dominates the night is exactly the sort of
   thing that varies across the year.

Usage:
    python scripts/calibrate_time_gate.py
    python scripts/calibrate_time_gate.py --min-recall 0.99
"""
import argparse
import os
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")


def detection_hour(df):
    """Wall-clock hour of each detection: recording start + offset within it."""
    rec = pd.to_datetime(df["timestamp"], format="%Y%m%dT%H%M%S")
    det = rec + pd.to_timedelta(df["start_s"], unit="s")
    return det.dt.hour + det.dt.minute / 60.0


def auc(score, positive):
    """Rank-based AUC; no sklearn dependency for a two-line computation."""
    r = pd.Series(score).rank().to_numpy()
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", default=REVIEW)
    ap.add_argument("--min-recall", type=float, default=0.95,
                    help="Windows retaining less than this share of confirmed "
                         "calls are not considered.")
    args = ap.parse_args()

    if not os.path.exists(args.review):
        sys.exit(f"missing {args.review}")
    d = pd.read_csv(args.review)
    d["hour"] = detection_hour(d)
    d["is_call"] = d["verdict"].eq("call")
    n_call = int(d["is_call"].sum())
    n_fp = len(d) - n_call

    print(f"{len(d)} reviewed detections: {n_call} calls, {n_fp} false positives")
    print(f"ungated precision {d['is_call'].mean():.4f}")
    days = pd.to_datetime(d["timestamp"], format="%Y%m%dT%H%M%S").dt.date
    print(f"review spans {days.min()} .. {days.max()} "
          f"({days.nunique()} days) -- no seasonal claim is supportable\n")

    print("precision by hour")
    for h in range(24):
        m = d["hour"].astype(int) == h
        if m.sum():
            bar = "#" * int(40 * d.loc[m, "is_call"].mean())
            print(f"  {h:02d}  n={m.sum():5d}  calls={d.loc[m,'is_call'].sum():5d}"
                  f"  {d.loc[m,'is_call'].mean():.4f}  {bar}")

    rows = []
    for start in range(0, 13):
        for end in range(13, 25):
            m = (d["hour"] >= start) & (d["hour"] < end)
            if not m.sum():
                continue
            recall = d.loc[m, "is_call"].sum() / n_call
            if recall < args.min_recall:
                continue
            rows.append({
                "start": start, "end": end, "kept": int(m.sum()),
                "recall": recall, "precision": d.loc[m, "is_call"].mean(),
                "fps_removed": (n_fp - int((~d.loc[m, "is_call"]).sum())) / n_fp,
            })
    best = pd.DataFrame(rows).sort_values("precision", ascending=False)
    print(f"\nwindows retaining >= {args.min_recall:.0%} of calls, best first")
    print(best.head(10).to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"))

    top = best.iloc[0]
    print(f"\nchosen: {int(top['start']):02d}:00-{int(top['end']):02d}:00  "
          f"precision {top['precision']:.4f}  recall {top['recall']:.4f}  "
          f"fps removed {top['fps_removed']:.4f}")

    lost = d[~((d["hour"] >= top["start"]) & (d["hour"] < top["end"]))]
    print(f"  calls lost to the gate: {int(lost['is_call'].sum())} "
          f"-- measure diel patterns with the gate OFF")

    y = d["is_call"].to_numpy()
    day = ((d["hour"] >= top["start"]) & (d["hour"] < top["end"])).astype(float)
    print(f"\nAUC  is-daytime {auc(day, y):.4f}   "
          f"model confidence {auc(d['confidence'], y):.4f}   "
          f"raw hour {auc(d['hour'], y):.4f}")

    print("\nper station: ungated -> gated precision")
    for site, g in d.groupby("site"):
        gd = g[(g["hour"] >= top["start"]) & (g["hour"] < top["end"])]
        if not len(gd):
            continue
        print(f"  {site:>8}  n={len(g):5d}  {g['is_call'].mean():.4f} -> "
              f"{gd['is_call'].mean():.4f}   "
              f"calls lost {int(g['is_call'].sum() - gd['is_call'].sum())}")


if __name__ == "__main__":
    main()
