"""Measure the noise floor in AVERAGE PRECISION, the metric the paper argues in.

The manuscript has a measured floor and uses it everywhere: changing nothing at
all moves paired sixteen-station macro precision by up to 0.0035. That floor was
measured on precision at each fold's own fitted threshold.

Three of the paper's conclusions are not in that metric. The band split is
+0.0021 of average precision, the frequency-position encoding +0.0017, the
unfreeze contrast +0.0073 -- all threshold-free, all read against a floor
measured on a different quantity. Nobody has measured what changing nothing does
to average precision, so nobody knows whether those three numbers are effects.

The three draws needed are already on disk. The frozen temporal_freqpos
specification was trained three times -- same index, same folds, same schedule,
same seed, different unseeded weight initialisation and batch order -- and all
three head sets survive:

    full_2026-08-19/heads_freqpos
    full_2026-08-19/heads_freqpos_evalfix
    replicates_2026-08-30/frozen_rep3/heads

Verified distinct by sha256 before use; identical weights would make this
circular. Nothing is trained here. Each draw is scored through the same path
prauc_head_ablation.py uses -- same evaluation rows, same 05:00-19:00 gate, same
Cernic-against-everything target, same average_precision_score -- so the spread
between them is attributable to the draw and nothing else.

    python scripts/prauc_noise_floor.py
"""
import argparse
import hashlib
import itertools
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
import train_v13_loso as T  # noqa: E402
import tensorflow as tf  # noqa: E402

BASE = os.path.join(REPO, "data/outputs/v13_runs/full_2026-08-19")

# Three draws of ONE specification. The pooling is identical in all three; that
# is the point. Paths are absolute because the third lives in another run dir.
DRAWS = {
    "d1_freqpos": os.path.join(BASE, "heads_freqpos"),
    "d2_evalfix": os.path.join(BASE, "heads_freqpos_evalfix"),
    "d3_rep3": os.path.join(REPO,
                            "data/outputs/v13_runs/replicates_2026-08-30",
                            "frozen_rep3/heads"),
}
POOLING = "temporal_freqpos"
STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST", "IPA8ST",
            "IPA10ST", "IPA11ST", "IPA13ST", "IPA14ST", "IPA15ST", "IPA16ST",
            "IPA17ST", "IPA18ST", "IPA19ST", "IPA20ST"]


def _hour(s):
    hh, mm = s.split(":")
    return int(hh) + int(mm) / 60.0


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(
        BASE, "prauc_noise_floor.csv"))
    args = ap.parse_args()

    # ---- the draws must be distinct, or the floor is zero by construction ----
    print("  weight sha256 (first 16), three stations:")
    identical = []
    for st in ("IPA1ST", "IPA10ST", "IPA20ST"):
        hs = {}
        for nm, d in DRAWS.items():
            p = os.path.join(d, f"head_{st}.weights.h5")
            hs[nm] = _sha(p) if os.path.exists(p) else "MISSING"
        print(f"    {st:8s} " + "  ".join(f"{k}={v}" for k, v in hs.items()))
        vals = [v for v in hs.values() if v != "MISSING"]
        if len(set(vals)) != len(vals):
            identical.append(st)
    if identical:
        print(f"  ! draws share weights at {identical}; the floor would be "
              f"circular. Stopping.")
        return 1
    print("  all three draws are distinct.\n")

    index = T.load_index(f"{BASE}/v13_index.csv", f"{BASE}/manifest.csv")
    index["hour"] = T.detection_hours(index)
    pack_row = index["row"].to_numpy()
    class_names = sorted(index["label"].unique())
    feats = np.load(f"{BASE}/v13_features.npy", mmap_mode="r")
    gate = (_hour(config.TIME_FILTER_START), _hour(config.TIME_FILTER_END))
    ti = class_names.index("Cernic")

    rows = []
    for arm, hd in DRAWS.items():
        if not os.path.isdir(hd):
            print(f"  skip {arm}: {hd} missing")
            continue
        for st in STATIONS:
            t0 = time.time()
            _, ev_mask, _ = T.fold_masks(index, st, keep_all_background=True)
            hours = index.loc[ev_mask, "hour"].to_numpy()
            keep = (hours >= gate[0]) & (hours < gate[1])
            if keep.sum() < 5:
                continue
            y = (index.loc[ev_mask, "label"].to_numpy() == "Cernic"
                 ).astype(int)[keep]
            if y.sum() == 0 or y.sum() == len(y):
                continue

            inp = tf.keras.Input(shape=feats.shape[1:])
            out = model_module.build_dense_tail(
                model_module.build_temporal_pool(inp, POOLING),
                num_classes=len(class_names))
            head = tf.keras.Model(inp, out)
            head.load_weights(os.path.join(hd, f"head_{st}.weights.h5"))
            X = feats[pack_row[ev_mask][keep]]
            p = head.predict(X, batch_size=256, verbose=0)[:, ti]
            tf.keras.backend.clear_session()

            rows.append({"arm": arm, "station": st, "n": int(len(y)),
                         "n_calls": int(y.sum()),
                         "base_rate": round(float(y.mean()), 4),
                         "ap": round(float(average_precision_score(y, p)), 6),
                         "auc": round(float(roc_auc_score(y, p)), 6),
                         "minutes": round((time.time() - t0) / 60, 2)})
            print(f"  {arm:11s} {st:8s} n={len(y):5d} calls={int(y.sum()):4d} "
                  f"AP={rows[-1]['ap']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n  wrote {os.path.relpath(args.out, REPO)}")

    piv = out.pivot(index="station", columns="arm", values="ap")
    print("\n  macro average precision, three draws of ONE specification:")
    for a in piv.columns:
        print(f"    {a:11s} {piv[a].mean():.5f}")

    print("\n  pairwise -- this is the floor. Changing nothing at all:")
    sds, mxs, means, ts = [], [], [], []
    for a, b in itertools.combinations(list(piv.columns), 2):
        d = (piv[a] - piv[b]).dropna().to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se else float("nan")
        sds.append(d.std(ddof=1)); mxs.append(abs(d).max())
        means.append(abs(d.mean())); ts.append(abs(t))
        print(f"    {a} - {b}: mean {d.mean():+.4f}  t {t:+.2f}  "
              f"SD {d.std(ddof=1):.4f}  max|station| {abs(d).max():.4f}  "
              f"better at {int((d > 0).sum())}/{len(d)}")

    print(f"\n  THE FLOOR IN AVERAGE PRECISION")
    print(f"    largest paired mean       {max(means):.4f}")
    print(f"    largest |t|               {max(ts):.2f}")
    print(f"    largest per-station SD    {max(sds):.4f}")
    print(f"    largest single station    {max(mxs):.4f}")
    print("\n  Read against it, the effects the manuscript reports "
          "threshold-free:")
    for nm, eff in (("band split (freq - temporal)", 0.0021),
                    ("freqpos - freq", 0.0017),
                    ("unfreeze block34", 0.0073)):
        verdict = "ABOVE the floor" if eff > max(means) else "inside the floor"
        print(f"    {nm:30s} {eff:+.4f}   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
