"""Score the fine-tuned arms threshold-free, from the weights already in Drive.

Why this rather than a second draw of block34.

The sweep reports block34 at +0.0109 precision over frozen, t = +2.45, and that
is the paper's only nominally significant backbone effect. The obvious way to
test it is to train it again. But the head ablation was tested the same way on
2026-08-30 and the answer was not what the retrain would have told us: the band
split's +0.0096 at fitted thresholds fell to +0.0021 measured threshold-free,
because the two arms' LOSO-fitted thresholds land in different places (0.287
against 0.544) and their per-station precision and recall differences
anti-correlate at r = -0.59. Most of the apparent architectural gain was the
operating point sliding along one curve.

block34 is exposed to exactly the same confound and has never been checked for
it. Average precision integrates over every threshold, so it cannot be moved by
where a fitted cut happened to land -- and unlike a retrain it needs no
training at all, because the fine-tuned models are already on Drive: the
unfreeze runner saved each fold's whole model, trunk included, not just the head
(train_v13_loso.py builds `head` from the image input when --unfreeze > 0).

So this is inference over 16 folds instead of 16 fine-tuning runs: roughly half
an hour against six and a half, and it answers the question a retrain would
leave open.

The frozen side is not recomputed here. It was measured locally on 2026-08-30
(data/outputs/v13_runs/full_2026-08-19/head_ablation_prauc.csv, arm "freqpos",
macro AP 0.9825) over the same evaluation rows, and both sides produce a
probability for the same target class on the same rows, so the per-station
values are comparable. Bring that file's numbers to the comparison rather than
retraining a frozen arm here to reproduce them.

    REP_ARM_DIR=/content/drive/MyDrive/primates-sound-detection/unfreeze_2026-08-21 \
      python colab/prauc_arms.py
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.environ.get("REPO", "/content/repo")
DATA = os.environ.get("DATAC", "/content/dataF")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import config  # noqa: E402
import model as model_module  # noqa: E402
import train_v13_loso as T  # noqa: E402
import tensorflow as tf  # noqa: E402

STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST", "IPA8ST",
            "IPA10ST", "IPA11ST", "IPA13ST", "IPA14ST", "IPA15ST", "IPA16ST",
            "IPA17ST", "IPA18ST", "IPA19ST", "IPA20ST"]
TAP = getattr(T, "TAP_LAYER", "block4_conv4")


def _hour(s):
    hh, mm = s.split(":")
    return int(hh) + int(mm) / 60.0


def build_full_model(n_classes, pooling="temporal_freqpos"):
    """The architecture --unfreeze builds: images in, VGG19 trunk, then the head.

    Rebuilt rather than deserialised because the saved artefact is a weights
    file, not a model file. The layer graph has to match the one that produced
    it or load_weights raises, which is the check that this is the right shape.
    """
    inp = tf.keras.Input(shape=(224, 224, 3))
    from tensorflow.keras.applications import VGG19
    base = VGG19(weights="imagenet", include_top=False, input_tensor=inp)
    base.trainable = False
    tapped = base.get_layer(TAP).output
    out = model_module.build_dense_tail(
        model_module.build_temporal_pool(tapped, pooling), num_classes=n_classes)
    return tf.keras.Model(inp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-dir", default=os.environ.get(
        "REP_ARM_DIR",
        "/content/drive/MyDrive/primates-sound-detection/unfreeze_2026-08-21"))
    ap.add_argument("--arms", default="block34,block4")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.arm_dir, "arms_prauc.csv")

    index = T.load_index(os.path.join(DATA, "v13_index.csv"),
                         os.path.join(DATA, "manifest.csv"))
    index["hour"] = T.detection_hours(index)
    pack_row = index["row"].to_numpy()
    class_names = sorted(index["label"].unique())
    ti = class_names.index("Cernic")
    images = np.load(os.path.join(DATA, "v13_images.npy"), mmap_mode="r")
    gate = (_hour(config.TIME_FILTER_START), _hour(config.TIME_FILTER_END))

    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        hd = os.path.join(args.arm_dir, arm)
        if not os.path.isdir(hd):
            print(f"  skip {arm}: {hd} not found")
            continue
        for st in STATIONS:
            wf = os.path.join(hd, f"head_{st}.weights.h5")
            if not os.path.exists(wf):
                print(f"  skip {arm}/{st}: no weights")
                continue
            t0 = time.time()
            _, ev_mask, _ = T.fold_masks(index, st, keep_all_background=True)
            hours = index.loc[ev_mask, "hour"].to_numpy()
            keep = (hours >= gate[0]) & (hours < gate[1])
            y = (index.loc[ev_mask, "label"].to_numpy() == "Cernic").astype(int)[keep]
            if y.sum() in (0, len(y)):
                print(f"  {arm}/{st}: single-class pool, skipped")
                continue

            mdl = build_full_model(len(class_names))
            mdl.load_weights(wf)
            X = images[pack_row[ev_mask][keep]].astype("float32") / 255.0
            p = mdl.predict(X, batch_size=64, verbose=0)[:, ti]
            tf.keras.backend.clear_session()

            rows.append({"arm": arm, "station": st, "n": int(len(y)),
                         "n_calls": int(y.sum()),
                         "ap": round(float(average_precision_score(y, p)), 4),
                         "minutes": round((time.time() - t0) / 60, 2)})
            print(f"  {arm:8s} {st:8s} n={len(y):5d} AP={rows[-1]['ap']:.4f} "
                  f"({rows[-1]['minutes']} min)", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\n  wrote {out_path}")
    d = pd.DataFrame(rows)
    if len(d):
        for a, g in d.groupby("arm"):
            print(f"  {a:8s} macro AP {g.ap.mean():.4f} over {len(g)} folds")
    print("\n  Pair these against the frozen arm's per-station AP in\n"
          "  data/outputs/v13_runs/full_2026-08-19/head_ablation_prauc.csv\n"
          "  (arm 'freqpos', macro 0.9825). If block34's +0.0109 at fitted\n"
          "  thresholds was the operating point rather than the backbone, it\n"
          "  shrinks here the way the band split's +0.0096 shrank to +0.0021.")


if __name__ == "__main__":
    main()
