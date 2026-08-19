"""
Score a fresh detection run against the manual review, without new listening.

This is what makes a staged rollout possible. Re-detecting all 3 014 recordings
is 5.4 M windows -- 156 h on a CPU, hours on a GPU -- and there is no reason to
pay it before knowing whether the new model is better. One station is 345 600
windows: ten hours on a CPU, about twenty minutes on a working GPU. And because
16 stations were already reviewed detection by detection, a single station's
re-run can be scored against that review immediately.

Each new detection falls into one of three cases against the reviewed windows of
the same recording:

- it overlaps a window the reviewer confirmed as a **call** -> a genuine call the
  new model still finds;
- it overlaps a window the reviewer marked **Noise** -> a false positive the new
  model still makes;
- it overlaps nothing that was reviewed -> **new**. Unknown, and the interesting
  case: V12 never fired there, so nobody has ever listened. These are exported as
  a listening batch.

And each reviewed window falls into two:

- confirmed call with no new detection -> a call the new model **lost**;
- confirmed false positive with no new detection -> a false positive **removed**.

So precision and the retained share of V12's calls are both measurable at once,
with no listening at all. What still cannot be measured this way is absolute
recall: a call that neither model ever fired on is invisible to both. Only the
"new" column can grow the count of known calls, and only after someone listens
to it.

Usage:
    python scripts/compare_detection_to_review.py --station IPA11ST \\
        --detections data/outputs/detections_v13/IPA11ST

    # and build a listening page for whatever came back new
    python scripts/compare_detection_to_review.py --station IPA11ST \\
        --detections <dir> --export-new data/outputs/v13_new_IPA11ST
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REVIEW_TABLE = os.path.join(REPO,
                            "data/outputs/auto_cleanup/cleanup_vs_review.csv")
REVIEW_CLIP_RE = re.compile(r"^Cernic__(.+?)__(\d+)s__conf([\d.]+)\.wav$")


def _recording_stem(path):
    """
    Recording name from a path, stripping only a real extension.

    ``os.path.splitext`` cannot be used here. AudioMoth writes the deployment
    coordinates into the name -- ``S20210224T175957188+0100_E..._<lat><lon>``
    -- and splitext takes ``.7522`` for an extension and removes it. The review
    side keeps the full name, so the two never match and every detection reads
    as new ground: a silent, total failure that looks like a result.
    """
    return re.sub(r"\.(wav|WAV|csv|CSV)$", "", os.path.basename(str(path)))


def load_review(station):
    """Reviewed windows for one station: recording, start second, verdict."""
    review = pd.read_csv(REVIEW_TABLE)
    review = review[review["site"] == station]
    rows = []
    for _, r in review.iterrows():
        m = REVIEW_CLIP_RE.match(str(r["file"]))
        if m:
            rows.append({"recording": m.group(1), "start_s": int(m.group(2)),
                         "verdict": r["verdict"], "species": r["species"]})
    return pd.DataFrame(rows)


def load_detections(det_dir, species="Cernic"):
    """
    Detections from a run, as (recording, start_s, confidence).

    The per-file CSVs carry only start_time, end_time, species, confidence and
    low_freq_ratio -- the recording is not a column, it is the CSV's own name
    (``<recording>_detections.csv``). Reading it from the path is not a
    convenience: without it every detection in the station collapses to one
    nameless stream and matching against the review is meaningless.
    """
    frames = []
    for path in glob.glob(os.path.join(det_dir, "**", "*_detections.csv"),
                          recursive=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not len(df):
            continue
        base = os.path.basename(path)
        df = df.copy()
        df["_recording"] = re.sub(r"_detections\.csv$", "", base)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["recording", "start_s", "confidence"])
    det = pd.concat(frames, ignore_index=True)
    if "species" in det.columns:
        det = det[det["species"] == species]

    # A column wins over the filename when the run recorded one.
    rec_col = next((c for c in ("recording", "source_file", "file")
                    if c in det.columns), "_recording")
    start_col = next((c for c in ("start_time", "start_s", "start")
                      if c in det.columns), None)
    if start_col is None:
        sys.exit(f"detection CSVs lack a start column: {list(det.columns)}")
    out = pd.DataFrame({
        "recording": det[rec_col].astype(str).map(_recording_stem),
        "start_s": pd.to_numeric(det[start_col], errors="coerce"),
        "confidence": pd.to_numeric(det.get("confidence", np.nan),
                                    errors="coerce"),
    }).dropna(subset=["start_s"])
    return out.reset_index(drop=True)


def match(det, rev, window=2.0, tolerance=1.0):
    """
    Pair new detections with reviewed windows of the same recording.

    Two windows are the same event when they overlap. Detection advances in 1 s
    steps on a 2 s window, so a call the review recorded at t and the new run
    reports at t+1 is one call seen twice, not two events -- ``tolerance``
    absorbs that. Anything stricter would count a one-second shift as both a
    lost call and a new discovery.
    """
    rev_index = {}
    for i, r in rev.iterrows():
        rev_index.setdefault(r["recording"], []).append((r["start_s"], i))
    for k in rev_index:
        rev_index[k].sort()

    det_to_rev = np.full(len(det), -1, dtype=int)
    rev_hit = np.zeros(len(rev), dtype=bool)
    span = window + tolerance
    for j, d in det.iterrows():
        best, best_gap = -1, None
        for start, i in rev_index.get(d["recording"], ()):
            gap = abs(start - d["start_s"])
            if gap <= span and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best >= 0:
            det_to_rev[j] = best
            rev_hit[best] = True
    return det_to_rev, rev_hit


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--detections", required=True,
                    help="Folder of *_detections.csv from the new run.")
    ap.add_argument("--species", default="Cernic")
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--export-new", default=None,
                    help="Write the new detections' clips here as a listening "
                         "batch.")
    args = ap.parse_args()

    rev = load_review(args.station)
    rev = rev[rev["species"] == args.species].reset_index(drop=True)
    det = load_detections(args.detections, args.species)
    if not len(rev):
        sys.exit(f"no reviewed {args.species} detections for {args.station}")
    print(f"{args.station}: {len(rev)} reviewed windows, "
          f"{len(det)} new detections\n")

    det_to_rev, rev_hit = match(det, rev, tolerance=args.tolerance)
    is_call = (rev["verdict"] == "call").to_numpy()

    matched = det_to_rev >= 0
    on_call = matched & np.isin(det_to_rev, np.where(is_call)[0])
    on_fp = matched & ~on_call
    new = ~matched

    n_call, n_fp = int(is_call.sum()), int((~is_call).sum())
    kept_call = int(rev_hit[is_call].sum())
    kept_fp = int(rev_hit[~is_call].sum())

    print("=" * 66)
    print("AGAINST THE REVIEW (no listening needed)")
    print("=" * 66)
    print(f"  confirmed calls, still detected  : {kept_call}/{n_call} "
          f"({100 * kept_call / n_call:.1f}%)")
    print(f"  confirmed calls, now lost        : {n_call - kept_call}")
    print(f"  false positives, still made      : {kept_fp}/{n_fp} "
          f"({100 * kept_fp / n_fp:.1f}%)")
    print(f"  false positives, removed         : {n_fp - kept_fp} "
          f"({100 * (n_fp - kept_fp) / n_fp:.1f}%)")
    print()
    print(f"  V12 precision at this station    : "
          f"{n_call / (n_call + n_fp):.3f}  ({n_call}/{n_call + n_fp})")
    known = int(on_call.sum() + on_fp.sum())
    if known:
        print(f"  new run, on reviewed ground      : "
              f"{int(on_call.sum()) / known:.3f}  "
              f"({int(on_call.sum())}/{known})")
    print()
    print("=" * 66)
    print("NEW GROUND (needs listening)")
    print("=" * 66)
    print(f"  detections where V12 never fired  : {int(new.sum())}")
    print("  Nobody has heard these. They are the only place absolute recall")
    print("  can improve -- and the only place a new kind of false positive")
    print("  can hide. The precision above says nothing about them.")

    if int(new.sum()):
        nd = det[new].copy()
        print(f"\n  confidence: median {nd['confidence'].median():.2f}, "
              f"{(nd['confidence'] >= 0.9).sum()} at 0.90+")
        out = os.path.join(REPO, f"data/outputs/new_detections_{args.station}.csv")
        nd.to_csv(out, index=False)
        print(f"  wrote {out}")

    lost = rev[is_call & ~rev_hit]
    if len(lost):
        out = os.path.join(REPO, f"data/outputs/lost_calls_{args.station}.csv")
        lost.to_csv(out, index=False)
        print(f"\n  {len(lost)} confirmed calls the new model no longer finds "
              f"-> {out}")
        print("  Check these before celebrating the precision: a model that "
              "stops\n  firing improves precision and loses calls in the same "
              "motion.")


if __name__ == "__main__":
    main()
