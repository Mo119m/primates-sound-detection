"""Compare the ablation heads threshold-free, because the paired comparison is not.

The sixteen-fold head ablation scores each arm at its own LOSO-fitted threshold.
That is the right thing for reporting a deployment, and the wrong thing for
asking whether one head is better than another, because the fitted thresholds do
not land in the same place: averaged over the folds the simplest head is scored
at 0.287 and the band-split head at 0.544. The two arms are being compared at
materially different operating points on curves nobody drew.

The symptom is visible in the results. Across the sixteen stations the precision
and recall differences between those two arms are anti-correlated at r = -0.59:
the stations that gain recall lose precision and vice versa, and no station gains
both. That is the signature of an operating point sliding along one curve, not of
one curve sitting above another. A reader can slide their own threshold for free;
what they cannot get for free is a better curve.

So this scores both heads over the same evaluation rows and reports average
precision, which integrates over every threshold and cannot be moved by where a
fitted cut happened to land. If the band split lifts the curve, AP rises. If it
only moved the operating point, AP does not.

The heads already exist -- this trains nothing. It reuses the scoring path
validated on 2026-08-30 by reproducing loso16_freqpos_evalfix.csv at 16 of 16
stations across 12 columns from its own saved weights.

    python scripts/prauc_head_ablation.py
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import config  # noqa: E402
import model as model_module  # noqa: E402
import train as train_module  # noqa: E402
import train_v13_loso as T  # noqa: E402
import tensorflow as tf  # noqa: E402

BASE = os.path.join(REPO, "data/outputs/v13_runs/full_2026-08-19")
ARMS = {
    "temporal": ("heads_temporal_evalfix", "temporal"),
    "freq": ("heads_freq_evalfix", "temporal_freq"),
    "freqpos": ("heads_freqpos_evalfix", "temporal_freqpos"),
}
STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST", "IPA8ST",
            "IPA10ST", "IPA11ST", "IPA13ST", "IPA14ST", "IPA15ST", "IPA16ST",
            "IPA17ST", "IPA18ST", "IPA19ST", "IPA20ST"]


def _hour(s):
    hh, mm = s.split(":")
    return int(hh) + int(mm) / 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/outputs/v13_runs/full_2026-08-19/head_ablation_prauc.csv"))
    args = ap.parse_args()

    index = T.load_index(f"{BASE}/v13_index.csv", f"{BASE}/manifest.csv")
    index["hour"] = T.detection_hours(index)
    pack_row = index["row"].to_numpy()
    class_names = sorted(index["label"].unique())
    feats = np.load(f"{BASE}/v13_features.npy", mmap_mode="r")
    gate = (_hour(config.TIME_FILTER_START), _hour(config.TIME_FILTER_END))

    # The reported detection target. Scored exactly as the sweep scores it:
    # C. nictitans against everything else, inside the deployment time window.
    target = "Cernic"
    ti = class_names.index(target)

    rows = []
    for arm, (head_dir, pooling) in ARMS.items():
        hd = os.path.join(BASE, head_dir)
        if not os.path.isdir(hd):
            print(f"  skip {arm}: {head_dir} missing")
            continue
        for st in STATIONS:
            t0 = time.time()
            _, ev_mask, _ = T.fold_masks(index, st, keep_all_background=True)
            hours = index.loc[ev_mask, "hour"].to_numpy()
            keep = (hours >= gate[0]) & (hours < gate[1])
            if keep.sum() < 5:
                continue
            y = (index.loc[ev_mask, "label"].to_numpy() == target).astype(int)[keep]
            if y.sum() == 0 or y.sum() == len(y):
                print(f"  {arm}/{st}: single-class pool, skipped")
                continue

            inp = tf.keras.Input(shape=feats.shape[1:])
            out = model_module.build_dense_tail(
                model_module.build_temporal_pool(inp, pooling),
                num_classes=len(class_names))
            head = tf.keras.Model(inp, out)
            head.load_weights(os.path.join(hd, f"head_{st}.weights.h5"))

            X = feats[pack_row[ev_mask][keep]]
            p = head.predict(X, batch_size=256, verbose=0)[:, ti]
            tf.keras.backend.clear_session()

            rows.append({
                "arm": arm, "station": st, "n": int(len(y)),
                "n_calls": int(y.sum()),
                "base_rate": round(float(y.mean()), 4),
                "ap": round(float(average_precision_score(y, p)), 4),
                "auc": round(float(roc_auc_score(y, p)), 4),
                "minutes": round((time.time() - t0) / 60, 2),
            })
            print(f"  {arm:9s} {st:8s} n={len(y):5d} calls={int(y.sum()):4d} "
                  f"AP={rows[-1]['ap']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n  wrote {os.path.relpath(args.out, REPO)}")

    piv = out.pivot(index="station", columns="arm", values="ap")
    print("\n  macro average precision:")
    for a in piv.columns:
        print(f"    {a:9s} {piv[a].mean():.4f}")
    print("\n  paired, threshold-free:")
    for a, b in (("freq", "temporal"), ("freqpos", "temporal"),
                 ("freqpos", "freq")):
        if a not in piv.columns or b not in piv.columns:
            continue
        d = (piv[a] - piv[b]).dropna().to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"    {a} - {b}: {d.mean():+.4f}  t {d.mean()/se:+.2f}  "
              f"better at {int((d > 0).sum())}/{len(d)}")
    print("\n  If the band split lifts the curve, freq - temporal is positive"
          "\n  here too. If it only moved the operating point, it is not.")


if __name__ == "__main__":
    main()
