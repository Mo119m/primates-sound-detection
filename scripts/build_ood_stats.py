"""Fit the class statistics the out-of-distribution distance needs.

Detection has to answer a question softmax cannot: does this window resemble
anything the model was trained on? Mahalanobis distance to a class cluster
answers it, and needs three things fitted once -- a mean feature vector per
class, a shared inverse covariance, and a per-class cutoff. This computes them
from a packed training run and writes them where ``src/detection.py`` looks.

Why a separate step rather than fitting inside detection: the statistics depend
on the training set, not on the recording being scanned, and refitting them per
recording would make two runs of the same command disagree. Writing them once,
with the run they came from recorded in the file, also means a detection can be
traced back to the training set that defined "in distribution" for it.

    python scripts/build_ood_stats.py \
        --run data/outputs/v13_runs/20260809T1730Z_colobus_mixed \
        --model data/outputs/models/fold_IPA4ST.h5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="Run directory holding v13_images.npy and v13_index.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-class", type=int, default=800,
                    help="Images sampled per class. The covariance is shared, so "
                         "this bounds the fit rather than the accuracy of any "
                         "one class mean.")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="Added to the covariance diagonal before inversion. The "
                         "feature dimension is 256 and classes are correlated, "
                         "so the raw covariance is often near-singular.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import config
    import model as model_module
    import auto_cleanup

    # One file per head, named after the model, because each LOSO fold has its
    # own feature space and a shared path meant the second station silently
    # loaded the first station's statistics.
    stem = os.path.splitext(os.path.basename(args.model))[0]
    out = os.path.abspath(args.out or os.path.join(
        REPO, "data", getattr(config, "OOD_STATS_DIR", "outputs/ood_stats"),
        f"{stem}.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    images = np.load(os.path.join(args.run, "v13_images.npy"), mmap_mode="r")
    index = pd.read_csv(os.path.join(args.run, "v13_index.csv"))
    if len(images) != len(index):
        sys.exit(f"image pack has {len(images)} rows, index {len(index)}")

    mdl = model_module.load_trained_model(args.model)
    names = list(config.CLASS_NAMES)
    if int(mdl.output_shape[-1]) != len(names):
        sys.exit(f"model has {mdl.output_shape[-1]} outputs, config lists "
                 f"{len(names)}")
    fe = auto_cleanup.build_feature_extractor(mdl, config.OOD_FEATURE_LAYER)

    rng = np.random.RandomState(args.seed)
    feats_by_class, centered = {}, []
    for ci, lab in enumerate(names):
        rows = index[index.label == lab].row.to_numpy()
        if not len(rows):
            sys.exit(f"class {lab} has no rows in {args.run}")
        if len(rows) > args.per_class:
            rows = rows[rng.choice(len(rows), args.per_class, replace=False)]
        rows = np.sort(rows)
        x = np.asarray(images[rows], dtype=np.float32) / 255.0
        f = fe.predict(x, batch_size=64, verbose=0)
        feats_by_class[ci] = f
        centered.append(f - f.mean(axis=0))
        print(f"  {lab:18s} {len(f):5d} images")

    dim = centered[0].shape[1]
    means = np.stack([feats_by_class[c].mean(axis=0) for c in range(len(names))])
    cov = np.cov(np.concatenate(centered, axis=0), rowvar=False)
    inv_cov = np.linalg.inv(cov + args.ridge * np.eye(dim)).astype(np.float32)

    # Per-class cutoffs as percentiles of that class's own distances, which is
    # what makes "far" mean the same thing for a class with a tight cluster and
    # one with a loose one.
    cuts = {}
    for ci, lab in enumerate(names):
        d = feats_by_class[ci] - means[ci]
        d2 = np.einsum("ij,jk,ik->i", d, inv_cov, d)
        cuts[lab] = {int(q): float(np.percentile(d2, q))
                     for q in (50, 75, 90, 95, 96, 97, 98, 99)}
        print(f"  {lab:18s} in-sample distance  median {np.median(d2):8.1f}   "
              f"p90 {cuts[lab][90]:8.1f}   p99 {cuts[lab][99]:8.1f}")

    # A fingerprint of the head these statistics describe. Each LOSO fold trains
    # its own head, so their 256-d feature spaces are unrelated: applying one
    # fold's class means to another fold's features produces distances in the
    # tens of thousands against cutoffs in the hundreds, and nothing raises,
    # because the arithmetic is valid. Detection compares this fingerprint to
    # the model it was handed and refuses to annotate on a mismatch.
    import hashlib
    h = hashlib.sha256()
    for w in fe._head_tap.get_weights() if hasattr(fe, "_head_tap") else mdl.get_weights():
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    fingerprint = h.hexdigest()
    print(f"\nhead fingerprint {fingerprint[:16]}")

    np.savez(out, class_means=means, inv_cov=inv_cov,
             class_names=np.array(names),
             percentiles=np.array(sorted(cuts[names[0]].keys())),
             cutoffs=np.array([[cuts[n][q] for q in sorted(cuts[n])]
                               for n in names]),
             head_fingerprint=np.array(fingerprint),
             source_run=np.array(os.path.abspath(args.run)),
             source_model=np.array(os.path.abspath(args.model)))
    print(f"\nwrote {out}")
    print("detection.py loads this if present; without it the ood_distance "
          "column is\nleft empty rather than guessed.")


if __name__ == "__main__":
    main()
