"""Score frozen-trunk arms threshold-free, against the incumbent and the floor.

The backbone comparison is the same question the head ablation asked, so it is
answered the same way and by the same code path: average precision over the
identical gated evaluation rows, no fitted threshold anywhere. A fitted-threshold
comparison of two heads once read +0.0096 and became +0.0021 threshold-free,
because the fitted cuts landed at 0.287 and 0.544; there is no reason a backbone
swap would move the operating point any less.

What differs from prauc_head_ablation.py is only that each arm brings its own
feature cache, since the trunks tap different channel counts (512, 320, 256,
192, 128). The evaluation rows, the 05:00-19:00 gate, the Cernic-against-
everything target and the metric are identical across arms and identical to the
incumbent's, because they come from fold_masks and the index, which no arm
touches.

Read the result against data/outputs/v13_runs/full_2026-08-19/prauc_noise_floor.csv
-- three draws of ONE specification, differing by up to 0.0026 in the paired
mean, |t| to 1.89, and one pair favouring a side at twelve of sixteen stations.
Training here is unseeded exactly as it is there, so an arm is a draw, not a
point, and a difference under that floor is not a difference.

    python scripts/prauc_trunk_arms.py --arms effnetv2s_d1,convnext_tiny_d1
"""
import argparse
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

SRC = os.path.join(REPO, "data/outputs/v13_runs/full_2026-08-19")
TRUNKDIR = os.path.join(REPO, "data/outputs/v13_runs/trunks_2026-09-04")
FLOOR = os.path.join(SRC, "prauc_noise_floor.csv")
POOLING = "temporal_freqpos"
STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST", "IPA8ST",
            "IPA10ST", "IPA11ST", "IPA13ST", "IPA14ST", "IPA15ST", "IPA16ST",
            "IPA17ST", "IPA18ST", "IPA19ST", "IPA20ST"]

# The incumbent, scored from the same weights the paper reports.
INCUMBENT = ("vgg19_reported",
             os.path.join(SRC, "v13_features.npy"),
             os.path.join(SRC, "heads_freqpos_evalfix"))


def _hour(s):
    hh, mm = s.split(":")
    return int(hh) + int(mm) / 60.0


def arm_paths(name):
    """'effnetv2s_d1' -> its cache and head directory."""
    trunk = name.rsplit("_", 1)[0]
    return (os.path.join(TRUNKDIR, f"feats_{trunk}.npy"),
            os.path.join(TRUNKDIR, f"heads_{name}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names, e.g. effnetv2s_d1")
    ap.add_argument("--out", default=os.path.join(TRUNKDIR, "trunk_prauc.csv"))
    args = ap.parse_args()

    arms = [INCUMBENT]
    for nm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        cache, heads = arm_paths(nm)
        arms.append((nm, cache, heads))

    index = T.load_index(f"{SRC}/v13_index.csv", f"{SRC}/manifest.csv")
    index["hour"] = T.detection_hours(index)
    pack_row = index["row"].to_numpy()
    class_names = sorted(index["label"].unique())
    gate = (_hour(config.TIME_FILTER_START), _hour(config.TIME_FILTER_END))
    ti = class_names.index("Cernic")

    rows = []
    for arm, cache_path, head_dir in arms:
        if not os.path.exists(cache_path) or not os.path.isdir(head_dir):
            print(f"  skip {arm}: cache or heads missing")
            continue
        feats = np.load(cache_path, mmap_mode="r")
        print(f"  {arm}: cache {feats.shape}")
        for st in STATIONS:
            hp = os.path.join(head_dir, f"head_{st}.weights.h5")
            if not os.path.exists(hp):
                print(f"    {arm}/{st}: head not written yet, skipping")
                continue
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
            head.load_weights(hp)
            X = feats[pack_row[ev_mask][keep]]
            p = head.predict(X, batch_size=256, verbose=0)[:, ti]
            tf.keras.backend.clear_session()

            rows.append({"arm": arm, "station": st, "n": int(len(y)),
                         "n_calls": int(y.sum()),
                         "base_rate": round(float(y.mean()), 4),
                         "ap": round(float(average_precision_score(y, p)), 6),
                         "auc": round(float(roc_auc_score(y, p)), 6),
                         "minutes": round((time.time() - t0) / 60, 2)})
            print(f"    {arm:18s} {st:8s} n={len(y):5d} calls={int(y.sum()):4d} "
                  f"AP={rows[-1]['ap']:.4f}", flush=True)

    if not rows:
        print("  nothing scored")
        return 1
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n  wrote {os.path.relpath(args.out, REPO)}")

    piv = out.pivot(index="station", columns="arm", values="ap")
    print("\n  macro average precision:")
    for a in piv.columns:
        print(f"    {a:20s} {piv[a].mean():.5f}  ({piv[a].notna().sum()} folds)")

    # ---- the floor these differences have to clear ----
    fl_mean = fl_t = fl_win = None
    if os.path.exists(FLOOR):
        f = pd.read_csv(FLOOR).pivot(index="station", columns="arm",
                                     values="ap")
        ms, ts, ws = [], [], []
        for a, b in itertools.combinations(sorted(f.columns), 2):
            d = (f[a] - f[b]).dropna().to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d))
            ms.append(abs(d.mean())); ts.append(abs(d.mean() / se))
            ws.append(max(int((d > 0).sum()), len(d) - int((d > 0).sum())))
        fl_mean, fl_t, fl_win = max(ms), max(ts), max(ws)
        print(f"\n  same-specification floor: mean {fl_mean:.4f}, "
              f"|t| {fl_t:.2f}, {fl_win}/16 stations")

    base = "vgg19_reported"
    if base in piv.columns:
        print("\n  paired against the incumbent, threshold-free:")
        for a in piv.columns:
            if a == base:
                continue
            d = (piv[a] - piv[base]).dropna().to_numpy()
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            t = d.mean() / se if se else float("nan")
            wins = int((d > 0).sum())
            verdict = "under the floor -- not resolvable"
            if fl_mean is not None and abs(d.mean()) > fl_mean:
                verdict = ("CLEARS the floor" if d.mean() > 0
                           else "CLEARS the floor, downward")
            print(f"    {a:20s} {d.mean():+.4f}  t {t:+.2f}  "
                  f"{wins}/{len(d)} stations   {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
