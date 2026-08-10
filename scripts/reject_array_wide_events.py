"""
Reject detections that fire across the whole recorder array at once.

The 253 Colobus detections the deployment produced were all listened to and not
one is a genuine roar. What they are is visible without listening to anything:

    2021-02-24 15:00   71 detections across  9 stations
    2021-02-24 19:30   66 detections across 11 stations

Those two half-hours are 54.2 % of every Colobus detection ever made, and 191 of
the 253 fall on that single day. The stations span 0.484-0.500 N by
12.717-12.760 E -- a 2.3 x 6.2 km grid, kilometres apart. **A troop does not
roar at nine stations simultaneously. A storm front does.** Thunder is the one
sound in this corpus that is guaranteed to arrive everywhere at once, and
low-frequency energy cannot separate it from a roar (``lf_ratio`` AUC 0.840 is
the best of the acoustic candidates and it still overlaps).

So this rejects on geometry and time instead of on sound, which costs nothing:
no listening, no retraining, no audio decoded. A detection is dropped when at
least ``--min-stations`` distinct stations fire within the same
``--window-minutes`` slot.

CALIBRATE IT BEFORE TRUSTING IT. ``--validate`` scores the rule against both
ground-truth sets already on disk:

  * the 253 Colobus detections, every one a confirmed non-roar -- the rule
    should remove as many as possible;
  * the 6 189 reviewed Cernic detections, 2 535 of them confirmed real -- the
    rule must NOT remove those. Cernic choruses genuinely propagate between
    neighbouring stations, and the same test on Cernic is far weaker (75.0 % of
    false positives versus 58.5 % of real calls sit in >= 3-station slots), so
    an aggressive setting will delete real data.

That asymmetry is the whole reason this is a Colobus tool and not a general one.

THE TRAP THIS RULE WALKS INTO, AND THE FIX
------------------------------------------
"A troop does not call at nine stations at once" is **false for a chorusing
species**, and the Cernic column proves it: even at ``--min-stations 6``, 46.3 %
of the 2 535 confirmed real Cernic calls sit in array-wide slots. Choruses
propagate; that is what a chorus is.

*C. guereza* roars are a dawn chorus -- troops answer each other across the
forest -- so a genuine roar event is likely to look **exactly like** what this
rule rejects. Applied naively it would delete the signal it was built to find.

What rescues it is that the two storm slots are at **15:00 and 19:30**, not at
dawn. So the rule is time-conditional, and ``--chorus-window`` encodes that:

    outside the chorus window   array-wide  ->  weather, reject
    inside  the chorus window   array-wide  ->  chorus, keep

Sweeping ``--min-stations`` shows 4, 5 and 6 all reject the same 137 detections
(the two fronts are the entire signal), so take the highest -- it costs nothing
and cuts collateral damage. **Never run this over a dawn window with the chorus
exemption disabled.**

Usage:
    python scripts/reject_array_wide_events.py --validate
    python scripts/reject_array_wide_events.py \
        --detections data/outputs/detections_v13 --species Colobus_guereza
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DETECTIONS = os.path.join(REPO, "data/outputs/detections")
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")

# Two filename schemes coexist in this corpus. Matching only one silently drops
# the other -- that is how "90.7 % of Colobus detections are nocturnal" was
# computed from 97 of 253 files when the real figure over all 253 is 48.6 %.
# Both are matched here and anything unparsed is reported, never discarded.
START_RE = re.compile(r"S?(\d{8}T\d{6})")


def load_detections(root, species=None):
    """Every per-recording detection CSV under `root`, with wall-clock times."""
    rows = []
    unparsed = 0
    for path in glob.glob(os.path.join(root, "**", "*_detections.csv"),
                          recursive=True):
        name = os.path.basename(path)
        m = START_RE.search(name)
        if not m:
            unparsed += 1
            continue
        try:
            start = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S")
        except ValueError:
            unparsed += 1
            continue
        station = None
        for part in os.path.normpath(path).split(os.sep):
            if part.startswith("IPA"):
                station = part
                break
        try:
            df = pd.read_csv(path)
        except Exception:
            unparsed += 1
            continue
        if not len(df):
            continue
        df["station"] = station or "UNKNOWN"
        df["recording"] = name
        df["when"] = start + pd.to_timedelta(df["start_time"], unit="s")
        rows.append(df)
    if unparsed:
        print(f"  {unparsed} files could not be parsed and were NOT dropped "
              f"silently -- inspect them before trusting any total")
    if not rows:
        return pd.DataFrame(columns=["station", "recording", "when",
                                     "species", "confidence"])
    out = pd.concat(rows, ignore_index=True)
    if species:
        out = out[out["species"] == species]
    return out.reset_index(drop=True)


def flag_array_wide(df, window_minutes=30, min_stations=3, chorus_window=None):
    """
    True where a detection shares its slot with >= min_stations stations.

    ``chorus_window`` is an (start_hour, end_hour) pair that is EXEMPTED. Inside
    it, array-wide co-occurrence is read as a chorus rather than as weather, and
    nothing is flagged. Without this the rule deletes exactly the dawn events a
    guereza search is looking for -- see the module docstring.
    """
    if not len(df):
        return pd.Series(dtype=bool)
    slot = df["when"].dt.floor(f"{window_minutes}min")
    flagged = df.groupby(slot)["station"].transform("nunique") >= min_stations
    if chorus_window is not None:
        lo, hi = chorus_window
        hour = df["when"].dt.hour + df["when"].dt.minute / 60.0
        flagged = flagged & ~((hour >= lo) & (hour < hi))
    return flagged


def parse_window(s):
    if not s:
        return None
    a, b = s.split("-")
    def _h(t):
        hh, mm = (t.split(":") + ["0"])[:2]
        return int(hh) + int(mm) / 60.0
    return _h(a), _h(b)


def validate(window_minutes, min_stations, chorus_window=None):
    """Score the rule against the two ground-truth sets already on disk."""
    print("=" * 68)
    print(f"VALIDATION  window={window_minutes} min  min_stations={min_stations}")
    print("=" * 68)

    col = load_detections(DETECTIONS, species="Colobus_guereza")
    if len(col):
        f = flag_array_wide(col, window_minutes, min_stations, chorus_window)
        print(f"\nColobus ({len(col)} detections, ALL confirmed non-roars by "
              f"listening)")
        print(f"  rejected {int(f.sum())} / {len(f)} = {f.mean():.1%}"
              f"   <- higher is better, every one is known false")
        top = (col[f].groupby(col.loc[f, "when"].dt.floor(f"{window_minutes}min"))
               .agg(n=("station", "size"), stations=("station", "nunique")))
        for when, r in top.sort_values("n", ascending=False).head(4).iterrows():
            print(f"    {when}  {int(r.n):3d} detections across "
                  f"{int(r.stations):2d} stations")
    else:
        print(f"\nNo Colobus detections under {DETECTIONS}")

    if not os.path.exists(REVIEW):
        print(f"\nNo review table at {REVIEW} -- cannot check the recall cost")
        return
    rev = pd.read_csv(REVIEW)
    rec = pd.to_datetime(rev["timestamp"], format="%Y%m%dT%H%M%S")
    rev["when"] = rec + pd.to_timedelta(rev["start_s"], unit="s")
    rev["station"] = rev["site"]
    f = flag_array_wide(rev, window_minutes, min_stations, chorus_window)
    is_call = rev["verdict"].eq("call")
    print(f"\nCernic ({len(rev)} reviewed, {int(is_call.sum())} confirmed real)")
    print(f"  would reject {int((f & is_call).sum())} REAL calls "
          f"({(f & is_call).sum() / is_call.sum():.1%} of them)"
          f"   <- this is the cost, and it must stay small")
    print(f"  would reject {int((f & ~is_call).sum())} false positives "
          f"({(f & ~is_call).sum() / (~is_call).sum():.1%})")
    if (f & is_call).mean() > 0.05:
        print("\n  WARNING: this setting deletes real Cernic calls. Cernic "
              "choruses\n  genuinely propagate between stations -- apply the "
              "rule to Colobus only,\n  or raise --min-stations.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", default=DETECTIONS)
    ap.add_argument("--species", default="Colobus_guereza")
    ap.add_argument("--window-minutes", type=int, default=30)
    ap.add_argument("--min-stations", type=int, default=3)
    ap.add_argument("--chorus-window", default=None,
                    help="HH:MM-HH:MM exempted from rejection, because inside "
                         "it array-wide firing means a chorus, not weather. "
                         "Use 05:00-08:00 for a guereza dawn search.")
    ap.add_argument("--validate", action="store_true",
                    help="Score the rule against the 253 known-false Colobus "
                         "detections and the 6 189 reviewed Cernic ones, "
                         "instead of filtering.")
    ap.add_argument("--out", default=None,
                    help="Write the surviving detections here.")
    args = ap.parse_args()

    if args.validate:
        validate(args.window_minutes, args.min_stations,
                 parse_window(args.chorus_window))
        return

    df = load_detections(args.detections, species=args.species)
    if not len(df):
        sys.exit(f"no {args.species} detections under {args.detections}")
    f = flag_array_wide(df, args.window_minutes, args.min_stations,
                        parse_window(args.chorus_window))
    keep = df[~f].sort_values("confidence", ascending=False)
    print(f"{len(df)} detections -> rejected {int(f.sum())} as array-wide, "
          f"{len(keep)} survive")
    if args.out:
        keep.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
        print("These are the ones worth listening to. Rank is by confidence, "
              "which\nis only AUC 0.73 against the review, so do not stop at "
              "the top of the list.")


if __name__ == "__main__":
    main()
