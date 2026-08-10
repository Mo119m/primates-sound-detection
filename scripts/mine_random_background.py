"""
Sample the forest at random, so Background stops meaning "what V12 got wrong".

93 % of the current Background class is the deployed model's own false positives
or reviewed detections. The remaining 7 % is curated reference material, and even
the 17 101 BirdNET clips inside it were chosen by a detector rather than drawn
from the recordings. The class therefore describes one narrow slice of the
forest: the slice a detector already reacted to.

That is enough to re-rank a detector's candidates, which is what the
leave-one-station-out numbers measure and why they are good. It is not enough to
scan raw audio, which is why a held-out model firing on IPA4ST produced 1 154
detections of which the first six checked by ear were all wrong, at confidences
up to 0.988. Most of what a recording contains belongs to neither class the model
knows, so it picks one confidently.

This draws windows uniformly from the deployment itself. Three things have to be
right or it makes matters worse:

1. **Station attribution.** A mined clip carries the station it came from, so
   leave-one-station-out still holds that station out. Without it, twenty
   thousand clips from every station would enter every fold's training set and
   every held-out number in the paper would be quietly inflated.

2. **Not silently mining positives.** A random window can contain a genuine call.
   Labelling it Background teaches the model that a call is not a call. Every
   window is screened by the current model, and anything it finds interesting is
   written to a separate folder for a person to check rather than used.

3. **Not re-mining the review.** Windows overlapping a reviewed detection are
   skipped: those already have human labels and belong in their own classes.

    python scripts/mine_random_background.py --n 20000 --dry-run
    python scripts/mine_random_background.py --n 20000
"""
import argparse
import os
import random
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")
GUARD_S = 4.0          # keep this far away from any reviewed detection


def reviewed_windows():
    """(recording stem, start second) of every window a human has already judged."""
    if not os.path.exists(REVIEW):
        print("  ! no review table; cannot avoid re-mining judged windows")
        return {}
    d = pd.read_csv(REVIEW)
    out = {}
    for r in d.itertuples():
        stem = str(r.timestamp)
        out.setdefault(stem, []).append(float(r.start_s))
    return {k: np.sort(np.array(v)) for k, v in out.items()}


def too_close(judged, stem, t):
    arr = judged.get(stem)
    if arr is None or not arr.size:
        return False
    i = int(np.searchsorted(arr, t))
    for j in (i - 1, i):
        if 0 <= j < arr.size and abs(arr[j] - t) < GUARD_S:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000,
                    help="Windows to draw, spread evenly over the stations.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--screen-threshold", type=float, default=0.5,
                    help="A window the current model scores above this on any "
                         "target group is set aside for review, not used.")
    ap.add_argument("--model", default=os.path.join(
        REPO, "data/outputs/models/best_model_v12.h5"))
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/background/random_forest"))
    ap.add_argument("--suspect-out", default=os.path.join(
        REPO, "data/outputs/random_mine_suspect"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import config

    root = config.IPA_ROOT
    stations = sorted(s for s in os.listdir(root)
                      if s.upper().startswith("IPA")
                      and os.path.isdir(os.path.join(root, s)))
    if not stations:
        sys.exit(f"no IPA* stations under {root}\n"
                 f"Set PRIMATE_IPA_ROOT to the drive holding them.")
    print(f"{len(stations)} stations under {root}")

    rnd = random.Random(args.seed)
    per_station = max(1, args.n // len(stations))
    judged = reviewed_windows()
    win = float(config.CLIP_DURATION)

    plan = []
    for st in stations:
        files = []
        for dirpath, _d, names in os.walk(os.path.join(root, st)):
            files += [os.path.join(dirpath, n) for n in names
                      if n.lower().endswith(".wav")]
        if not files:
            continue
        for _ in range(per_station):
            p = rnd.choice(files)
            try:
                info = sf.info(p)
            except Exception:
                continue
            if info.duration <= win + 1:
                continue
            t = rnd.uniform(0.0, info.duration - win)
            m = re.search(r"(\d{8}T\d{6})", os.path.basename(p))
            stem = m.group(1) if m else ""
            if too_close(judged, stem, t):
                continue
            plan.append((st, p, t))
    print(f"planned {len(plan)} windows "
          f"({per_station} per station before exclusions)")
    if args.dry_run:
        by = pd.Series([s for s, _p, _t in plan]).value_counts().sort_index()
        print(by.to_string())
        return

    import model as model_module
    import preprocessing
    mdl = model_module.load_trained_model(args.model)

    # See the note in rescreen_random_background.py: a model whose output width
    # disagrees with config.CLASS_NAMES cannot be read positionally, and doing
    # it anyway is silent rather than loud. The first run of this script screened
    # against a four-output model while config listed five classes, so index 3
    # meant Background to the model and C_pogonias to the config, and 76 % of
    # random forest audio was flagged as a possible call.
    n_out = int(mdl.output_shape[-1])
    if n_out != len(config.CLASS_NAMES):
        sys.exit(
            f"{os.path.basename(args.model)} has {n_out} outputs but "
            f"config.CLASS_NAMES lists {len(config.CLASS_NAMES)}.\n"
            f"Pass --model a model built for this config, e.g. one from "
            f"scripts/assemble_fold_model.py.")

    names = list(config.CLASS_NAMES)
    targets = [i for i, n in enumerate(names)
               if config.DETECTION_GROUPS.get(n, n) not in ("Background",)]
    print(f"screening against {[names[i] for i in targets]}")

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.suspect_out, exist_ok=True)
    kept = suspect = failed = 0
    batch, meta = [], []

    def flush():
        nonlocal kept, suspect
        if not batch:
            return
        probs = mdl.predict(np.stack(batch), batch_size=64, verbose=0)
        for (st, y, name), pr in zip(meta, probs):
            hot = float(pr[targets].max())
            dest = args.out if hot < args.screen_threshold else args.suspect_out
            # The station goes in the filename: possible_stations() reads
            # "ipaNst_" from the containing directory, so mined clips live in a
            # per-station subfolder rather than one flat pile.
            sub = os.path.join(dest, f"{st.lower()}_random")
            os.makedirs(sub, exist_ok=True)
            sf.write(os.path.join(sub, name), y, config.SAMPLE_RATE)
            if hot < args.screen_threshold:
                kept += 1
            else:
                suspect += 1
        batch.clear()
        meta.clear()

    for k, (st, p, t) in enumerate(plan):
        try:
            y, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True,
                                offset=t, duration=win)
        except Exception:
            failed += 1
            continue
        need = int(win * config.SAMPLE_RATE)
        if y.size < need:
            y = np.pad(y, (0, need - y.size))
        img = preprocessing.preprocess_for_model(
            preprocessing.preprocess_audio(y))
        batch.append(img)
        meta.append((st, y,
                     f"{os.path.splitext(os.path.basename(p))[0]}__t{int(t):05d}s.wav"))
        if len(batch) >= 256:
            flush()
            print(f"\r  {k + 1}/{len(plan)}  kept {kept}  suspect {suspect}",
                  end="", flush=True)
    flush()

    print(f"\n\nkept as Background : {kept}")
    print(f"set aside for review: {suspect}  -> "
          f"{os.path.relpath(args.suspect_out, REPO)}")
    print(f"failed to read     : {failed}")
    print("\nThe suspect pile is not a mistake to be deleted. It is the random "
          "sample's\nestimate of how often a call occurs in ordinary audio, "
          "which is the recall\ninformation this project has never had. Listen "
          "to it before deciding what it is.")
    print(f"\nAdd '{os.path.relpath(args.out, os.path.join(REPO, 'data'))}' to "
          f"config.BACKGROUND_FOLDERS, then rebuild the manifest.")


if __name__ == "__main__":
    main()
