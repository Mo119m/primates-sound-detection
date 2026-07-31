"""
Check that TensorFlow can actually train on this machine's GPU.

Run this before anything else on a new machine. ``tf.config.list_physical_devices('GPU')``
returning a device is not the same as that device working: a card newer than the
wheel it is running under is visible, is selected, and then either falls back to
PTX JIT (slow, sometimes silently) or fails at the first kernel launch. This
script does a real forward and backward pass through the model's own base and
reports throughput, so "the GPU works" is measured rather than assumed.

Two known traps on the machine this was written for:

- **Blackwell (RTX 50-series, compute capability 12.0).** TensorFlow's released
  wheels are built for older architectures. Reaching sm_120 depends on PTX
  forward-compatibility and needs a recent CUDA runtime. If throughput here is
  not far above the ~3 images/s a laptop CPU manages, the GPU is not really
  being used.
- **Windows.** TensorFlow dropped native Windows GPU support after 2.10. On
  Windows this has to run inside WSL2; a Windows-native install will report no
  GPU no matter what is in the machine.

Usage:
    python scripts/check_gpu.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402


def main():
    print("=" * 68)
    print("ENVIRONMENT")
    print("=" * 68)
    print(f"  platform      : {sys.platform}")
    print(f"  python        : {sys.version.split()[0]}")

    try:
        import tensorflow as tf
    except Exception as exc:
        sys.exit(f"  TensorFlow did not import: {exc}")
    print(f"  tensorflow    : {tf.__version__}")

    build = tf.sysconfig.get_build_info()
    print(f"  built with CUDA : {build.get('cuda_version', '?')}, "
          f"cuDNN {build.get('cudnn_version', '?')}")

    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPUs visible  : {len(gpus)}")
    for g in gpus:
        try:
            d = tf.config.experimental.get_device_details(g)
            cc = d.get("compute_capability")
            print(f"    {g.name}  {d.get('device_name', '?')}"
                  + (f"  compute capability {cc[0]}.{cc[1]}" if cc else ""))
            if cc and cc[0] >= 12:
                print("    ^ Blackwell. TensorFlow's wheels predate this "
                      "architecture; if the\n      throughput below is poor, it "
                      "is running through PTX JIT or not at all.")
        except Exception:
            print(f"    {g.name}")

    if not gpus:
        print("\n  NO GPU VISIBLE.")
        if sys.platform.startswith("win"):
            print("  On Windows this is expected: TensorFlow dropped native "
                  "Windows GPU\n  support after 2.10. Run inside WSL2.")
        print("  Training the full sweep on CPU is ~113 h. Use "
              "colab/v13_train.ipynb instead.")

    print()
    print("=" * 68)
    print("REAL THROUGHPUT (VGG19 to block4_conv4, the frozen base)")
    print("=" * 68)

    from tensorflow.keras.applications import VGG19
    try:
        base = VGG19(weights=None, include_top=False,
                     input_shape=(224, 224, 3))
        extractor = tf.keras.Model(base.input,
                                   base.get_layer("block4_conv4").output)
    except Exception as exc:
        sys.exit(f"  could not build the model: {exc}")

    x = np.random.rand(16, 224, 224, 3).astype("float32")
    try:
        extractor.predict(x, verbose=0)
        t0 = time.time()
        for _ in range(3):
            extractor.predict(x, verbose=0)
        fwd = (time.time() - t0) / 3 / 16
    except Exception as exc:
        sys.exit(f"\n  A kernel failed to launch: {exc}\n"
                 f"  The device is visible but unusable. Fall back to Colab.")
    print(f"  forward  : {1 / fwd:7.1f} images/s  ({fwd * 1000:.0f} ms each)")

    head = tf.keras.Sequential([
        extractor,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    head.compile("adam", "sparse_categorical_crossentropy")
    y = np.random.randint(0, 4, 16)
    head.train_on_batch(x, y)
    t0 = time.time()
    for _ in range(3):
        head.train_on_batch(x, y)
    bwd = (time.time() - t0) / 3 / 16
    print(f"  backward : {1 / bwd:7.1f} images/s  ({bwd * 1000:.0f} ms each)")

    print()
    print("=" * 68)
    print("WHAT THAT MEANS FOR THIS PROJECT")
    print("=" * 68)
    n_clips, n_windows = 31021, 5_400_000
    print(f"  feature cache over {n_clips} clips : "
          f"{n_clips / (1 / fwd) / 60:6.1f} min")
    print(f"  re-detection over {n_windows / 1e6:.1f}M windows : "
          f"{n_windows / (1 / fwd) / 3600:6.1f} h")

    # A laptop CPU manages ~9 forward and ~3 backward. Anything in that
    # neighbourhood means the GPU is not doing the work.
    if 1 / fwd < 40:
        print("\n  This is CPU-class throughput. Whatever the device list says,")
        print("  the GPU is not doing the work -- check the CUDA runtime, or")
        print("  use colab/v13_train.ipynb and keep this machine for the")
        print("  re-detection pass, which needs the 444 GB drive, not speed.")
    else:
        print("\n  GPU is doing the work. Proceed with HANDOFF.md section 3b.")


if __name__ == "__main__":
    main()
