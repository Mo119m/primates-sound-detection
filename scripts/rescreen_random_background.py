"""
Re-sort the mined windows using the rule the deployment actually applies.

The first screen asked whether any target class scored above 0.5, and set aside
15 053 of 19 746 windows on that basis. That rule is not the one the pipeline
uses. ``src/detection.py`` collapses the softmax onto ``config.DETECTION_GROUPS``
and takes the argmax, so a window scoring Cernic 0.5 against Background 0.5
produces no detection at all. Screening on the raw class score therefore flags
three quarters of ordinary forest audio and says nothing.

This re-sorts the clips already on disk with the deployed rule: the grouped
argmax must land on a target group AND clear the detection threshold. Nothing is
re-mined; the audio is the same audio.

    python scripts/rescreen_random_background.py --dry-run
    python scripts/rescreen_random_background.py
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", default=os.path.join(
        REPO, "data/background/random_forest"))
    ap.add_argument("--suspect", default=os.path.join(
        REPO, "data/outputs/random_mine_suspect"))
    ap.add_argument("--model", default=os.path.join(
        REPO, "data/outputs/models/best_model_v12.h5"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import librosa
    import config
    import preprocessing
    import model as model_module
    import detection

    files = (sorted(glob.glob(os.path.join(args.background, "*", "*.wav")))
             + sorted(glob.glob(os.path.join(args.suspect, "*", "*.wav"))))
    if not files:
        sys.exit("no mined clips found; run mine_random_background.py first")
    print(f"{len(files)} mined windows on disk")

    mdl = model_module.load_trained_model(args.model)

    # A model with fewer outputs than config.CLASS_NAMES is a different model
    # from the one config describes, and reading it positionally silently maps
    # Background's probability onto whichever class config thinks sits at that
    # index. That is exactly how the first screen flagged 76 % of ordinary
    # forest audio: best_model_v12.h5 has four outputs, config now lists five,
    # and index 3 is Background in one and C_pogonias in the other.
    n_out = int(mdl.output_shape[-1])
    if n_out != len(config.CLASS_NAMES):
        sys.exit(
            f"{os.path.basename(args.model)} has {n_out} outputs but "
            f"config.CLASS_NAMES lists {len(config.CLASS_NAMES)}: "
            f"{list(config.CLASS_NAMES)}.\n"
            f"Reading it against this config would map the wrong class to every "
            f"index.\nUse a model built for this config, e.g.\n"
            f"  python scripts/assemble_fold_model.py --station IPA20ST\n"
            f"  ... --model data/outputs/models/fold_IPA20ST.h5")

    labels, indices = detection.get_detection_groups()
    thr = config.DETECTION_CONFIDENCE_THRESHOLD
    print(f"groups {labels}, threshold {thr}")

    moves, kept, flagged = [], 0, 0
    B = 256
    for lo in range(0, len(files), B):
        chunk = files[lo:lo + B]
        X = []
        for p in chunk:
            y, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
            X.append(preprocessing.preprocess_for_model(
                preprocessing.preprocess_audio(y)))
        probs = mdl.predict(np.stack(X), batch_size=64, verbose=0)
        for p, pr in zip(chunk, probs):
            g = detection.group_probabilities(pr, labels, indices)
            top = labels[int(np.argmax(g))]
            fires = top != "Background" and float(g.max()) >= thr
            station = os.path.basename(os.path.dirname(p))
            want_dir = os.path.join(
                args.suspect if fires else args.background, station)
            if fires:
                flagged += 1
            else:
                kept += 1
            if os.path.dirname(p) != want_dir:
                moves.append((p, os.path.join(want_dir, os.path.basename(p))))
        print(f"\r  {min(lo + B, len(files))}/{len(files)}  "
              f"background {kept}  flagged {flagged}", end="", flush=True)

    print(f"\n\nunder the deployed rule:")
    print(f"  usable as Background : {kept}  ({100 * kept / len(files):.1f}%)")
    print(f"  fires, needs a human : {flagged}  "
          f"({100 * flagged / len(files):.1f}%)")
    print(f"  files to move        : {len(moves)}")
    if args.dry_run:
        print("\ndry run, nothing moved")
        return
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    print(f"moved {len(moves)} files")
    print("\nThe flagged fraction is the deployed model's false-positive rate on "
          "ordinary\naudio, near enough: these windows were drawn at random and "
          "kept away from\nanything already reviewed, so whatever a person finds "
          "in them is the rate at\nwhich this detector would interrupt someone "
          "scanning the whole deployment.")


if __name__ == "__main__":
    main()
