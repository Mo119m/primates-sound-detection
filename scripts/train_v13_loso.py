"""
Train V13 and score it leave-one-station-out against the human review.

What this measures, and what it cannot
--------------------------------------
The 6 189 reviewed detections are the only ground truth this project has that
was paid for in listening hours. Each is a 2 s window the deployed V12 fired on,
labelled by a person as a genuine *C. nictitans* call (2 535) or not (3 654).
Re-classifying those exact windows with a new model answers one question
exactly: **of the false positives a reviewer had to sit through, how many would
the new model no longer have produced, and how many real calls would it have
lost?** That is precision, and it is the number the field cares about.

It is not recall. A model that finds calls V12 missed cannot show that here,
because a window V12 never fired on was never exported and never reviewed. Recall
needs a re-detection pass over the recordings; those are 444 GB on an external
drive and about 170 h of CPU, so it is a separate job (see
``scripts/recall_sample.py`` for the sampling approach that bounds recall from a
few hours of exhaustively annotated audio).

Why leave-one-station-out
-------------------------
The obvious experiment -- fold the 3 654 reviewed false positives into training,
then check how many of them the model now rejects -- answers nothing, because it
tests on its own training data. Worse, the same trap is already in the *existing*
numbers: V12's training set contains auto-flagged negatives mined from the same
2021-02 recordings at IPA2, IPA10, IPA11, IPA13, IPA14 and IPA16, so its reported
41.0 % field precision is partly in-sample too.

So each fold holds out one station completely. A clip is withheld whenever the
held-out station appears in its ``possible_stations``, which includes the 1 348
clips whose filenames narrow the source to a group of stations without naming
one -- the five that recorded with GPS off write identical filenames, and
guessing between them would leak quietly and inflate every fold.

Usage:
    # one fold, to check the pipeline end to end
    python scripts/train_v13_loso.py --folds IPA20ST --epochs 8

    # the full sweep (a GPU job; ~113 h on this CPU)
    python scripts/train_v13_loso.py --folds all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAP_LAYER = "block4_conv4"


def load_index(index_path, manifest_path=None):
    """
    The rows to train on, each still pointing at its row in the image pack.

    ``row`` is the position in ``v13_images.npy`` and stays fixed: the pack is
    expensive to rebuild, and the manifest can shrink under it when clips are
    found to be mislabelled (the mahal/yamnet dumps were dropped that way).
    Filtering here and indexing by ``row`` keeps the pack, the feature cache and
    the manifest in step without re-rendering 4.8 GB of spectrograms.
    """
    idx = pd.read_csv(index_path)
    idx["possible_stations"] = idx["possible_stations"].fillna("")
    idx["station"] = idx["station"].fillna("")
    if "ok" in idx:
        idx = idx[idx["ok"]]
    if manifest_path and os.path.exists(manifest_path):
        keep = set(pd.read_csv(manifest_path)["path"])
        before = len(idx)
        idx = idx[idx["path"].isin(keep)]
        if before != len(idx):
            print(f"  manifest excludes {before - len(idx)} packed clips")
    return idx.reset_index(drop=True)


def feature_cache(images_path, cache_path, batch=64):
    """
    Run the frozen VGG19 once and keep the tap activations.

    The base is frozen for stage 1, so its output for a given image never
    changes between epochs or between folds. Recomputing it is the whole cost of
    training: on this CPU the base runs at 8.9 images/s and the head at 19, so a
    cached run is not an approximation of stage 1 -- it is stage 1, with the
    constant part evaluated once instead of sixteen times over.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import VGG19

    images = np.load(images_path, mmap_mode="r")
    n = len(images)
    if os.path.exists(cache_path):
        feats = np.load(cache_path, mmap_mode="r")
        if len(feats) == n:
            print(f"  reusing {cache_path} ({feats.shape})")
            return feats
        print(f"  {cache_path} has {len(feats)} rows, need {n} -- rebuilding")

    base = VGG19(weights="imagenet", include_top=False,
                 input_shape=(224, 224, 3))
    extractor = tf.keras.Model(base.input, base.get_layer(TAP_LAYER).output)
    shape = tuple(extractor.output.shape[1:])
    print(f"  extracting {n} x {shape} -> {cache_path} "
          f"({n * np.prod(shape) * 2 / 1e9:.1f} GB float16)")

    out = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                    shape=(n,) + shape)
    t0 = time.time()
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        x = images[lo:hi].astype("float32") / 255.0
        out[lo:hi] = extractor.predict(x, verbose=0).astype("float16")
        if lo % (batch * 20) == 0:
            rate = hi / max(time.time() - t0, 1e-9)
            print(f"\r  {hi}/{n}  {rate:.0f} img/s  "
                  f"eta {(n - hi) / rate / 60:.1f} min", end="", flush=True)
    out.flush()
    print()
    return np.load(cache_path, mmap_mode="r")


def fold_masks(index, station):
    """Training rows, and the reviewed rows this fold is scored on."""
    possible = index["possible_stations"].str.split(";")
    withheld = possible.apply(lambda c: station in c)
    train = ~withheld
    evaluate = (index["station"] == station) & \
               index["source"].str.startswith("review")
    return train.to_numpy(), evaluate.to_numpy()


class MemmapBatches:
    """
    Keras Sequence over a feature memmap, reading one batch at a time.

    The cache is 25 GB and this machine has 16 GB, so ``feats[rows]`` -- the
    obvious way to gather a fold's training set -- would materialise 22 GB and
    die. Reading batches keeps the working set at a few tens of megabytes and
    costs nothing but sequential disk reads, which is what a memmap is for.
    Rows are sorted so each epoch walks the file forwards rather than seeking.
    """

    def __init__(self, feats, rows, labels, batch_size=32, shuffle=False):
        self.feats = feats
        self.rows = np.sort(np.asarray(rows))
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._order = np.arange(len(self.rows))

    def __len__(self):
        return int(np.ceil(len(self.rows) / self.batch_size))

    def __getitem__(self, i):
        sel = self._order[i * self.batch_size:(i + 1) * self.batch_size]
        sel = np.sort(sel)
        r = self.rows[sel]
        return (np.asarray(self.feats[r], dtype="float32"), self.labels[sel])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._order)

    def as_dataset(self, feature_shape):
        import tensorflow as tf
        ds = tf.data.Dataset.from_generator(
            lambda: (self[i] for i in range(len(self))),
            output_signature=(
                tf.TensorSpec(shape=(None,) + tuple(feature_shape), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64)))
        return ds.prefetch(tf.data.AUTOTUNE)


def train_head(feats, rows, labels, groups, class_names, epochs, seed,
               verbose=0):
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    import model as model_module
    import train as train_module

    local = np.arange(len(rows))
    tr, va, y_tr, y_va = train_module.grouped_split(
        local.reshape(-1, 1), labels, groups, test_size=0.2, seed=seed)
    tr, va = np.sort(tr.ravel()), np.sort(va.ravel())

    shape = feats.shape[1:]
    inp = tf.keras.Input(shape=shape)
    out = model_module.build_dense_tail(
        model_module.build_temporal_pool(inp, "temporal_freqpos"),
        num_classes=len(class_names))
    head = tf.keras.Model(inp, out)
    head.compile(tf.keras.optimizers.Adam(1e-4),
                 "sparse_categorical_crossentropy", metrics=["accuracy"])

    present = np.unique(labels[tr])
    weights = compute_class_weight("balanced", classes=present, y=labels[tr])
    class_weight = {int(c): float(w) for c, w in zip(present, weights)}

    train_seq = MemmapBatches(feats, rows[tr], labels[tr], shuffle=True)
    val_seq = MemmapBatches(feats, rows[va], labels[va])
    head.fit(train_seq.as_dataset(shape),
             validation_data=val_seq.as_dataset(shape),
             epochs=epochs, class_weight=class_weight, verbose=verbose)
    val_acc = head.evaluate(val_seq.as_dataset(shape), verbose=0)[1]
    return head, float(val_acc)


def score_fold(head, feats, rows, truth, class_names, threshold):
    """
    Re-classify one station's reviewed windows and compare with the review.

    A window "still fires" when the model's top class is Cernic at or above the
    deployment confidence threshold -- the same rule that produced the detection
    in the first place, so the comparison is like for like.
    """
    order = np.argsort(rows)
    truth = np.asarray(truth)[order]
    seq = MemmapBatches(feats, rows, np.zeros(len(rows)), batch_size=64)
    probs = np.concatenate([head.predict(seq[i][0], verbose=0)
                            for i in range(len(seq))])
    cernic = class_names.index("Cernic")
    fires = (probs.argmax(axis=1) == cernic) & (probs[:, cernic] >= threshold)

    is_call = truth == "call"
    n_call, n_fp = int(is_call.sum()), int((~is_call).sum())
    kept_call = int((fires & is_call).sum())
    kept_fp = int((fires & ~is_call).sum())
    kept = kept_call + kept_fp
    return {
        "detections": n_call + n_fp,
        "calls": n_call,
        "false_positives": n_fp,
        "v12_precision": round(n_call / max(n_call + n_fp, 1), 4),
        "kept_calls": kept_call,
        "kept_false_positives": kept_fp,
        "calls_retained": round(kept_call / n_call, 4) if n_call else None,
        "fps_removed": round(1 - kept_fp / n_fp, 4) if n_fp else None,
        "v13_precision": round(kept_call / kept, 4) if kept else None,
        "review_reduction": round(1 - kept / max(n_call + n_fp, 1), 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(REPO, "data/outputs/v13_images.npy"))
    ap.add_argument("--index", default=os.path.join(REPO, "data/outputs/v13_index.csv"))
    ap.add_argument("--cache", default=os.path.join(REPO, "data/outputs/v13_features.npy"))
    ap.add_argument("--manifest",
                    default=os.path.join(REPO, "data/outputs/v13_manifest.csv"),
                    help="Restricts the packed clips to those still in the "
                         "manifest, without re-rendering the image pack.")
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/v13_loso.csv"))
    ap.add_argument("--folds", default="IPA20ST",
                    help="'all', or a comma-separated list of stations.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None,
                    help="Cap the training pool per fold. For checking that the "
                         "pipeline runs on a CPU; a capped run's numbers are "
                         "not the experiment.")
    args = ap.parse_args()

    import config
    import train as train_module

    index = load_index(args.index, args.manifest)
    print(f"Manifest: {len(index)} clips")
    pack_row = index["row"].to_numpy()

    class_names = sorted(index["label"].unique())
    label_id = {c: i for i, c in enumerate(class_names)}
    labels = index["label"].map(label_id).to_numpy()
    groups = np.array([f"{r.label}/{train_module.source_group(r.path)}"
                       for r in index.itertuples()])
    print(f"Classes: {class_names}")
    print(f"Source groups: {len(set(groups))}\n")

    print("Feature cache")
    feats = feature_cache(args.images, args.cache)

    stations = sorted(s for s in index["station"].unique() if s)
    if args.folds != "all":
        stations = [s.strip() for s in args.folds.split(",") if s.strip()]

    results = []
    for station in stations:
        tr_mask, ev_mask = fold_masks(index, station)
        n_eval = int(ev_mask.sum())
        if not n_eval:
            print(f"{station}: no reviewed detections -- skipped")
            continue
        print(f"\n=== hold out {station} ===")
        print(f"  train {int(tr_mask.sum())} clips, "
              f"withheld {int((~tr_mask).sum())}, score on {n_eval} reviewed")

        # Positions in `index` -> rows in the image pack and feature cache.
        tr_rows = pack_row[tr_mask]
        groups_tr = groups[tr_mask]
        labels_tr = labels[tr_mask]
        if args.max_train and len(tr_rows) > args.max_train:
            # Sample by group, not by row, so a capped run still cannot put two
            # windows of one recording on opposite sides of the inner split.
            rng = np.random.default_rng(args.seed)
            uniq = sorted(set(groups_tr))
            keep = set(rng.choice(uniq, size=min(len(uniq), args.max_train // 4),
                                  replace=False))
            sel = np.array([g in keep for g in groups_tr])
            tr_rows, groups_tr, labels_tr = tr_rows[sel], groups_tr[sel], labels_tr[sel]
            print(f"  --max-train: capped to {len(tr_rows)} clips "
                  f"({len(keep)} groups) -- SMOKE TEST, not the experiment")
        t0 = time.time()
        head, val_acc = train_head(feats, tr_rows, labels_tr, groups_tr,
                                   class_names, args.epochs, args.seed,
                                   args.verbose)
        row = score_fold(head, feats, pack_row[ev_mask],
                         index.loc[ev_mask, "label"].map(
                             lambda l: "call" if l == "Cernic" else "fp"),
                         class_names, config.DETECTION_CONFIDENCE_THRESHOLD)
        row.update(station=station, grouped_val_accuracy=round(val_acc, 4),
                   minutes=round((time.time() - t0) / 60, 1))
        results.append(row)
        print(f"  grouped val acc {val_acc:.4f}   "
              f"precision {row['v12_precision']:.3f} -> "
              f"{row['v13_precision']}   "
              f"calls kept {row['calls_retained']}   "
              f"FPs removed {row['fps_removed']}")

    if not results:
        sys.exit("no fold produced a score")

    df = pd.DataFrame(results)[
        ["station", "detections", "calls", "false_positives", "v12_precision",
         "kept_calls", "kept_false_positives", "calls_retained", "fps_removed",
         "v13_precision", "review_reduction", "grouped_val_accuracy", "minutes"]]
    df.to_csv(args.out, index=False)
    print(f"\n{df.to_string(index=False)}")

    tot_call = df["calls"].sum()
    tot_fp = df["false_positives"].sum()
    kept_call = df["kept_calls"].sum()
    kept_fp = df["kept_false_positives"].sum()
    print(f"\nPooled over {len(df)} held-out stations:")
    print(f"  V12 precision {tot_call / (tot_call + tot_fp):.3f}  "
          f"({tot_call} calls / {tot_call + tot_fp} detections)")
    if kept_call + kept_fp:
        print(f"  V13 precision {kept_call / (kept_call + kept_fp):.3f}  "
              f"({kept_call} calls / {kept_call + kept_fp} detections)")
    print(f"  calls retained {kept_call / tot_call:.3f}, "
          f"false positives removed {1 - kept_fp / tot_fp:.3f}")
    print(f"\nWrote {args.out}")
    print("\nThis is precision only. Calls V12 never fired on are not in this "
          "set\nand cannot be scored here -- recall needs a re-detection pass.")


if __name__ == "__main__":
    main()
