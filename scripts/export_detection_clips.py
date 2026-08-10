"""
Cut listenable clips from a detection run, so a person can check the model.

A detection CSV is not evidence. The only way to know whether 1 154 Cernic
detections at one station are calls or noise is to listen to them, and the only
way to make that tractable is to sample deliberately rather than from the top.

Sampling from the top confidence alone is the mistake this avoids. A model that
is confidently wrong looks identical to one that is confidently right if you
only ever hear its best guesses. This takes a stratified sample across the
confidence range plus every detection of any rare class, and it says which
stratum each clip came from in the filename, so a listener can tell afterwards
where the model was sure and where it was not.

    python scripts/export_detection_clips.py --station IPA4ST
    python scripts/export_detection_clips.py --station IPA4ST --per-stratum 15
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAD_S = 0.5          # context either side, matching the review exports
STRATA = [(0.4, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.0001)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True)
    ap.add_argument("--detections", default="data/outputs/detections",
                    help="Directory holding <station>/<station>_summary.csv. "
                         "Not hardcoded, because a detection run that writes to "
                         "the default overwrites the previous one: the first "
                         "run of this pipeline did exactly that to the deployed "
                         "model's own output.")
    ap.add_argument("--label", default=None,
                    help="Name the review folder after the run rather than the "
                         "station, so two models' clips can sit side by side.")
    ap.add_argument("--per-stratum", type=int, default=12,
                    help="Clips per confidence band per species.")
    ap.add_argument("--all-below", type=int, default=60,
                    help="A species with fewer detections than this is exported "
                         "in full: a rare class is the whole point of looking.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reliability", action="store_true",
                    help="Sample for a precision-versus-confidence curve rather "
                         "than in bands: 100 detections spread evenly over the "
                         "confidence range plus 20 more from the top 5 %%, which "
                         "is the design Sagar et al. use to fit a logistic "
                         "regression for detection reliability. Band sampling "
                         "answers 'where does it stop being right'; this answers "
                         "'what precision should I expect at threshold t', which "
                         "is what a reader needs to pick one.")
    ap.add_argument("--n-uniform", type=int, default=100)
    ap.add_argument("--n-top", type=int, default=20)
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import config

    summary = os.path.join(REPO, args.detections, args.station,
                           f"{args.station}_summary.csv")
    if not os.path.exists(summary):
        sys.exit(f"no detection summary at {summary}")
    d = pd.read_csv(summary)
    out_root = os.path.join(REPO, "data/outputs/detection_review",
                            args.label or args.station)
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f"{len(d)} detections at {args.station}")
    picks = []

    if args.reliability:
        for sp, g in d.groupby("species"):
            if len(g) <= args.all_below:
                picks.append(g.assign(stratum="all"))
                print(f"  {sp:16s} {len(g):5d} -> all of them (rare class)")
                continue
            # Even coverage of the confidence axis, not of the detections: most
            # detections cluster near the top, and sampling them in proportion
            # would leave the low end, where the curve actually bends, with
            # almost no points.
            g = g.sort_values("confidence").reset_index(drop=True)
            lo, hi = g.confidence.min(), g.confidence.max()
            edges = np.linspace(lo, hi, args.n_uniform + 1)
            taken = []
            for k in range(args.n_uniform):
                band = g[(g.confidence >= edges[k]) & (g.confidence <= edges[k + 1])]
                if len(band):
                    taken.append(band.iloc[rng.choice(len(band))])
            top = g[g.confidence >= g.confidence.quantile(0.95)]
            if len(top):
                n = min(args.n_top, len(top))
                taken += [top.iloc[i] for i in rng.choice(len(top), n, replace=False)]
            sel = pd.DataFrame(taken).drop_duplicates()
            picks.append(sel.assign(stratum="reliability"))
            print(f"  {sp:16s} {len(g):5d} -> {len(sel)} spread over "
                  f"{lo:.3f}-{hi:.3f} plus the top 5 %")
    else:
      for sp, g in d.groupby("species"):
        if len(g) <= args.all_below:
            picks.append(g.assign(stratum="all"))
            print(f"  {sp:16s} {len(g):5d} -> all of them (rare class)")
            continue
        taken = 0
        for lo, hi in STRATA:
            band = g[(g.confidence >= lo) & (g.confidence < hi)]
            if not len(band):
                continue
            n = min(args.per_stratum, len(band))
            sel = band.iloc[rng.choice(len(band), n, replace=False)]
            picks.append(sel.assign(stratum=f"{lo:.2f}-{hi:.2f}"))
            taken += n
        print(f"  {sp:16s} {len(g):5d} -> {taken} across "
              f"{len(STRATA)} confidence bands")

    sel = pd.concat(picks, ignore_index=True)

    # The CSV stores a bare filename, not a path, so build a basename -> path
    # index over the station's recordings. Without this every lookup fails and
    # the script reports "0 clips written" while looking like it ran.
    station_dir = os.path.join(config.IPA_ROOT, args.station)
    if not os.path.isdir(station_dir):
        sys.exit(f"station audio not found at {station_dir}\n"
                 f"Set PRIMATE_IPA_ROOT to the drive holding the IPA* folders.")
    index = {}
    for root, _dirs, files in os.walk(station_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                index.setdefault(f, os.path.join(root, f))
    print(f"\n{len(index)} recordings under {station_dir}")

    pad = int(PAD_S * config.SAMPLE_RATE)
    cache, written, missing = {}, 0, 0
    for r in sel.itertuples():
        src = index.get(os.path.basename(str(r.source_file)))
        if src is None:
            missing += 1
            continue
        if src not in cache:
            cache.clear()          # one long recording in memory at a time
            cache[src] = librosa.load(src, sr=config.SAMPLE_RATE, mono=True)[0]
        y = cache[src]
        a = max(0, int(r.start_time * config.SAMPLE_RATE) - pad)
        b = min(y.size, int(r.end_time * config.SAMPLE_RATE) + pad)
        if b <= a:
            continue
        sub = os.path.join(out_root, r.species, str(r.stratum))
        os.makedirs(sub, exist_ok=True)
        name = (f"conf{r.confidence:.3f}__"
                f"{os.path.splitext(os.path.basename(src))[0]}__"
                f"t{int(r.start_time):05d}s.wav")
        sf.write(os.path.join(sub, name), y[a:b], config.SAMPLE_RATE)
        written += 1

    print(f"\nwrote {written} clips to {os.path.relpath(out_root, REPO)}")
    if missing:
        print(f"  {missing} skipped: source recording not found")
    print("Folders are species/confidence-band. Listen to a whole band before "
          "moving on;\nthe question is not whether the top clip is right but "
          "where the model stops\nbeing right.")


if __name__ == "__main__":
    main()
