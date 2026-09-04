"""Prove --trunk did not move the incumbent, and that each trunk is fed right.

Two things this file exists to catch, both silent.

THE INCUMBENT. Parameterising the frozen trunk touched the function that built
every feature cache in this project. If that edit changed VGG19's output by so
much as a rounding step, every reported number would drift and nothing would
error. So the first check re-extracts a sample of rows through the NEW code path
with --trunk vgg19 and requires them bit-identical to the cache on disk, which
was written by the old one.

THE INPUT CONVENTION. ConvNeXt and EfficientNetV2 carry a Normalization layer
inside the graph that expects [0, 255]. Handing them x/255 -- which is what this
pipeline feeds VGG19, and what a careless swap would keep feeding -- shrinks
every input by 255x. The model still runs, still converges, and still reports a
loss; only the features are wrong. The second check feeds one batch both ways
and requires the outputs to differ, so a trunk silently getting the wrong
convention shows up as a failure here rather than as a bad number in a table.

    python scripts/check_trunk.py [--rows 64]
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

import train_v13_loso as T  # noqa: E402

BASE = os.path.join(REPO, "data/outputs/v13_runs/full_2026-08-19")
IMAGES = os.path.join(BASE, "v13_images.npy")
CACHE = os.path.join(BASE, "v13_features.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras import applications as apps

    bad = 0

    # ---------------------------------------------- 1. the incumbent is intact
    if not (os.path.exists(IMAGES) and os.path.exists(CACHE)):
        print(f"  SKIP incumbent check: {IMAGES} or {CACHE} missing")
    else:
        images = np.load(IMAGES, mmap_mode="r")
        cached = np.load(CACHE, mmap_mode="r")
        n = min(args.rows, len(images))
        spec = T.trunk_spec("vgg19")
        base = getattr(apps, spec["app"])(weights="imagenet",
                                          include_top=False,
                                          input_shape=T.IMAGE_SHAPE)
        ex = tf.keras.Model(base.input, base.get_layer(spec["tap"]).output)
        x = T._scale_batch(images[:n].astype("float32"), spec["scale"])
        got = ex.predict(x, verbose=0).astype("float16")
        want = np.asarray(cached[:n])
        same = np.array_equal(got, want)
        nd = int((got != want).sum())
        print(f"  vgg19 through the new path, {n} rows: "
              f"{'BIT-IDENTICAL to the cache' if same else f'{nd} values differ'}")
        if not same:
            print(f"    max abs difference {np.abs(got.astype('float32') - want.astype('float32')).max():.6g}")
            print("    the incumbent moved. Every reported number is at risk.")
            bad += 1
        tf.keras.backend.clear_session()

    # ------------------------------------- 2. each trunk's convention matters
    print("\n  input convention -- feeding [0,255] vs [0,1] must NOT agree:")
    rng = np.random.default_rng(0)
    probe = (rng.random((8,) + T.IMAGE_SHAPE) * 255).astype("float32")
    for name, spec in sorted(T.TRUNKS.items()):
        try:
            base = getattr(apps, spec["app"])(weights="imagenet",
                                              include_top=False,
                                              input_shape=T.IMAGE_SHAPE)
            ex = tf.keras.Model(base.input, base.get_layer(spec["tap"]).output)
        except Exception as e:                                # noqa: BLE001
            print(f"    {name:14s} could not build: {type(e).__name__}")
            bad += 1
            continue
        shape = tuple(ex.output.shape[1:])
        ok_shape = shape == tuple(spec["shape"])
        a = ex.predict(T._scale_batch(probe.copy(), spec["scale"]), verbose=0)
        b = ex.predict(probe.copy() / 255.0, verbose=0)
        differs = not np.allclose(a, b, atol=1e-6)
        # For vgg19 the declared convention IS /255, so they must agree.
        want_differs = spec["scale"] != "div255"
        ok_conv = differs == want_differs
        print(f"    {name:14s} tap {str(shape):16s} "
              f"{'shape OK' if ok_shape else 'SHAPE WRONG'}   "
              f"scale={spec['scale']:7s} "
              f"{'distinguishes its convention' if ok_conv else 'CONVENTION NOT DISTINGUISHED'}")
        if not (ok_shape and ok_conv):
            bad += 1
        tf.keras.backend.clear_session()

    print()
    if bad:
        print(f"  {bad} problem(s).")
        return 1
    print("  the incumbent is unchanged and every trunk is fed its own way.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
