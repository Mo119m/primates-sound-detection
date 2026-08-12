"""
Rebuild a full detector from one leave-one-station-out fold.

The sweep trains only the dense tail, on cached VGG19 features, and saves that
tail. Detection needs a whole model: waveform in, class probabilities out. This
welds the two halves back together and writes a single ``.h5`` that
``scripts/run_detection_ipa.py`` can load unchanged.

Which fold to use is not a detail. Each fold's tail was trained without one
station, so running fold ``IPA20ST``'s model over IPA20ST's recordings is
detection by a model that has never seen a single window from that site. Any
other pairing is not. The script therefore refuses to assemble a model without
being told which station it is for, and stamps that station into the filename.

    python scripts/assemble_fold_model.py --station IPA20ST
    python scripts/run_detection_ipa.py --station IPA20ST \
        --model data/outputs/models/fold_IPA20ST.h5 --time-window 05:00-19:00
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TAP_LAYER = "block4_conv4"


def permute_output_units(model, perm):
    """Reorder a model's output classes by permuting its final Dense weights.

    Done in the weights rather than with a Lambda layer on top. A Lambda that
    closes over the permutation cannot be serialised -- Keras deepcopies layer
    configs and the closure drags a module reference in, which fails with
    "cannot pickle 'module' object" only at save time, after everything else has
    apparently worked. Permuting the kernel columns and the bias is exactly
    equivalent, leaves no custom layer for a loader to reconstruct, and cannot
    fail later.
    """
    last = None
    for layer in model.layers:
        if layer.weights:
            last = layer
    if last is None or len(last.get_weights()) != 2:
        raise RuntimeError(
            f"expected the final layer to be a Dense with kernel and bias, "
            f"got {last.name if last else None}")
    kernel, bias = last.get_weights()
    if kernel.shape[1] != len(perm):
        raise RuntimeError(
            f"final layer has {kernel.shape[1]} units, permutation has "
            f"{len(perm)}")
    last.set_weights([kernel[:, perm], bias[perm]])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", required=True,
                    help="The held-out station this fold's tail was trained "
                         "without. The assembled model is only a held-out "
                         "detector for this station.")
    ap.add_argument("--heads", default=os.path.join(REPO, "data/outputs/v13_heads"))
    ap.add_argument("--index", default=os.path.join(REPO, "data/outputs/v13_index.csv"))
    # The seam check addresses the image pack and the feature cache by the row
    # numbers in --index. Those three files are one artifact set, so pointing
    # --index at a run directory while these still resolved to the default pack
    # compared unrelated rows and reported a seam that was never tested. They
    # default beside --index for exactly that reason.
    ap.add_argument("--images", default=None,
                    help="Image pack matching --index. Default: v13_images.npy "
                         "in the same directory as --index.")
    ap.add_argument("--cache", default=None,
                    help="Feature cache matching --index. Default: "
                         "v13_features.npy beside --index.")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data/outputs/models"))
    args = ap.parse_args()

    import h5py
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.applications import VGG19
    import config
    import model as model_module

    station = args.station.upper()
    weights = os.path.join(args.heads, f"head_{station}.weights.h5")
    if not os.path.exists(weights):
        sys.exit(f"no head for {station} at {weights}\n"
                 f"available: "
                 f"{sorted(os.listdir(args.heads)) if os.path.isdir(args.heads) else '(no head dir)'}")

    # The tail's output width is the number of labels the sweep saw, in the
    # order it saw them: sorted unique labels of the index. Reading it from the
    # index rather than from config keeps a model assembled today loadable even
    # if config.CLASS_NAMES is edited tomorrow.
    idx = pd.read_csv(args.index)
    class_names = sorted(idx["label"].unique())
    print(f"fold {station}: {len(class_names)} classes {class_names}")
    if list(config.CLASS_NAMES) != class_names:
        print(f"  note: config.CLASS_NAMES is {list(config.CLASS_NAMES)}, which "
              f"differs in order or content.\n"
              f"  The assembled model follows the index, which is what the tail "
              f"was trained against.")

    # A fine-tuned fold saves the whole model, not a tail. train_v13_loso.py
    # --unfreeze builds VGG19 and the tail as one graph and checkpoints that, so
    # the file carries the convolutional weights the fine-tuning changed --
    # 132 MB against 24 MB for a frozen fold. Welding a fresh ImageNet base to
    # it would discard exactly the thing under test and produce a model that
    # runs, reports plausible numbers, and is not the model that was trained.
    # Detected from the file rather than from a flag, because a flag can be
    # forgotten and the failure is silent.
    # Discriminated by layer count, not by looking for convolutions. The
    # tail contains one Conv2D of its own (the frequency-coordinate fusion),
    # which Keras checkpoints under the generic name "conv2d", so "does this
    # file mention conv2d" is true of both kinds and mis-routes the frozen path.
    # Counting is exact: a tail-only checkpoint has as many layers as the tail,
    # a fine-tuned one has as many as the whole graph, and the two differ by the
    # fifteen VGG layers upstream of the tap.
    with h5py.File(weights, "r") as h:
        n_saved = len(h["layers"].keys()) if "layers" in h else 0

    def _tail_only(n_classes):
        b = VGG19(weights="imagenet", include_top=False,
                  input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH,
                               config.IMG_CHANNELS))
        t = b.get_layer(TAP_LAYER).output
        i = tf.keras.Input(shape=tuple(t.shape[1:]))
        return b, t, tf.keras.Model(i, model_module.build_dense_tail(
            model_module.build_temporal_pool(i, "temporal_freqpos"),
            num_classes=n_classes))

    def _whole(n_classes):
        i = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH,
                                  config.IMG_CHANNELS))
        b = VGG19(weights="imagenet", include_top=False, input_tensor=i)
        return b, tf.keras.Model(i, model_module.build_dense_tail(
            model_module.build_temporal_pool(
                b.get_layer(TAP_LAYER).output, "temporal_freqpos"),
            num_classes=n_classes))

    _, _, probe_tail = _tail_only(len(class_names))
    n_tail = len(probe_tail.layers)
    full_model_weights = n_saved != n_tail
    print(f"  checkpoint has {n_saved} layers; a tail alone has {n_tail} -> "
          f"{'fine-tuned whole model' if full_model_weights else 'tail only'}")

    if full_model_weights:
        print(f"  {os.path.basename(weights)} carries convolutional weights "
              f"({os.path.getsize(weights)/1e6:.0f} MB): this is a fine-tuned "
              f"fold, loading it whole rather than welding a frozen base")
        base, full = _whole(len(class_names))
        base.trainable = False
        tail = full
        full.load_weights(weights)
    else:
        base, tap, tail = _tail_only(len(class_names))
        base.trainable = False
        tail.load_weights(weights)

    # Reorder the outputs into config.CLASS_NAMES order before saving.
    #
    # This is not cosmetic. src/detection.py reads a softmax vector positionally
    # against config.CLASS_NAMES, and the sweep trains against sorted(labels),
    # which is a different order: position 0 is 'Cernic' to detection.py and
    # 'Background' to the head. An assembled model saved in the head's order
    # would run through the whole detection pipeline without error and label
    # every detection as the wrong species. Permuting here means every consumer
    # downstream keeps working unchanged.
    perm = [class_names.index(n) for n in config.CLASS_NAMES]
    if perm != list(range(len(perm))):
        print(f"  reordering outputs {class_names}\n"
              f"                  -> {list(config.CLASS_NAMES)}")
        permute_output_units(tail, perm)
    if not full_model_weights:
        full = tf.keras.Model(base.input, tail(tap))
    full.compile(tf.keras.optimizers.Adam(1e-4),
                 "sparse_categorical_crossentropy", metrics=["accuracy"])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"fold_{station}.h5")
    full.save(out)
    print(f"wrote {out}")

    # A model that silently mispredicts because the halves were glued in the
    # wrong order is worse than one that fails to build, so check the seam.
    art = os.path.dirname(os.path.abspath(args.index))
    images_path = args.images or os.path.join(art, "v13_images.npy")
    cache_path = args.cache or os.path.join(art, "v13_features.npy")
    images = np.load(images_path, mmap_mode="r")
    feats = np.load(cache_path, mmap_mode="r")
    if not (len(images) == len(feats) == len(idx)):
        sys.exit(f"artifact set disagrees: index {len(idx)} rows, "
                 f"{images_path} {len(images)}, {cache_path} {len(feats)}. "
                 f"The seam check addresses all three by the same row number, "
                 f"so it cannot run on a mismatched set.")
    print(f"  checking against {os.path.relpath(images_path, REPO)}")
    if full_model_weights:
        # The seam check asks whether a base and a tail were glued together
        # correctly, by running the cached features through the tail and the
        # images through the whole model and requiring the same answer. A
        # fine-tuned fold has no seam -- it is loaded as one graph -- and the
        # question is not merely unnecessary but unanswerable: the cache is the
        # output of the *frozen* base, and fine-tuning moved the convolutions
        # away from it. A disagreement here would be the treatment working.
        #
        # Feeding (28, 28, 512) features to a model whose input is
        # (224, 224, 3) is what the first version of this branch did, and it
        # failed with a graph execution error rather than a useful message.
        print("  seam check skipped: no seam. The cache is the frozen base's "
              "output and this model's base is not frozen, so the two paths "
              "are not supposed to agree.")
    else:
        rows = np.linspace(0, len(images) - 1, 24).astype(int)
        x = np.asarray(images[rows], dtype=np.float32) / 255.0
        direct = full.predict(x, batch_size=8, verbose=0)
        viacache = tail.predict(np.asarray(feats[rows], dtype=np.float32),
                                batch_size=8, verbose=0)
        # The tail was permuted in place, so both sides are already in
        # CLASS_NAMES order and compare directly.
        gap = float(np.abs(direct - viacache).max())
        print(f"seam check on 24 rows: max |full - cached-tail| = {gap:.2e}")
        if gap > 1e-3:
            sys.exit("ABORT: the assembled model disagrees with the "
                     "cached-feature path. The preprocessing or the tap layer "
                     "does not match.")

    # And check the permutation actually landed, by asking the model about rows
    # whose label is known. A saved model in the wrong class order is invisible
    # downstream, so this is the one check that must not be skipped.
    lab = idx["label"].to_numpy()
    wrong = []
    for name in config.CLASS_NAMES[:-1]:
        want = np.where(lab == name)[0][:8]
        if not want.size:
            continue
        p = full.predict(np.asarray(images[want], dtype=np.float32) / 255.0,
                         batch_size=8, verbose=0)
        hit = (p.argmax(axis=1) == list(config.CLASS_NAMES).index(name)).mean()
        print(f"  {name:18s} argmax lands on its own index for "
              f"{hit:.0%} of 8 training rows")
        if hit < 0.4:
            wrong.append(name)
    if wrong:
        sys.exit(f"ABORT: outputs look permuted for {wrong}. A model saved in "
                 f"the wrong class order mislabels every detection silently.")
    print("assembled model reproduces the sweep's scores, in CLASS_NAMES order")


if __name__ == "__main__":
    main()
