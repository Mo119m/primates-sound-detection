"""
Turn every clip in the V13 manifest into the 224x224x3 image the model eats.

This exists because of where the work has to happen. Training the head on this
machine runs at 19 clips/s and a sixteen-fold leave-one-station-out sweep would
take 113 hours, so the training belongs on a GPU. The field recordings are 444 GB
and cannot go anywhere; the *clips* are far smaller, and once turned into
spectrogram images they are 4.8 GB of uint8 -- one file that uploads and needs no
audio decoding, no librosa, and no external drive at the other end.

Packing here rather than there also removes a class of silent error. Every clip
must be reduced to the same 2 s analysis window the model was trained on, and
each source stores that window differently:

- ``review`` / ``colobus_field_fp``: clips exported for manual review, cut from
  ``max(0, start - 0.5)``, so the window begins ``min(start, 0.5)`` s in. This is
  the convention ``auto_cleanup.load_clips_from_dir`` implements and
  ``test_clip_source.py`` pins; getting it wrong shifts every evaluation clip by
  half a second and would quietly cost accuracy that no test would catch.
- ``auto_flagged_fp``: already exactly 2 s, no padding.
- ``reference`` / ``birdnet``: 3-6 s clips with no fixed convention, so they go
  through ``data_loader.load_audio_file`` with the loudest-window crop that
  training itself uses.

Output is a uint8 memmap plus an index CSV in the same row order. Rows that fail
to load are marked and kept in place, so image row *i* is always manifest row
*i* and no fold can be silently shifted.

Usage:
    python scripts/pack_v13_images.py
    python scripts/pack_v13_images.py --limit 200        # smoke test
    python scripts/pack_v13_images.py --workers 4
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PADDED_SOURCES = ("review", "colobus_field_fp")
EXPORT_PADDING = 0.5  # seconds, matches utils.extract_all_detected_clips


def _start_second(path):
    """The detection's offset into its recording, from the exported filename."""
    import re
    m = re.search(r"__t?(\d+)s__conf", os.path.basename(path))
    return float(m.group(1)) if m else 0.0


def load_window(path, source):
    """
    The 2 s analysis window for one clip, as a waveform.

    Raises on failure rather than returning silence: a silent clip trains and
    evaluates as a confident Background, which is exactly the kind of error that
    improves the numbers while breaking the model.
    """
    import librosa
    import config
    import data_loader

    group = source.split(":")[0]
    clip_len = int(round(config.WINDOW_SIZE * config.SAMPLE_RATE))

    if group in PADDED_SOURCES:
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
        offset = int(round(min(_start_second(path), EXPORT_PADDING)
                           * config.SAMPLE_RATE))
        clip = y[offset:offset + clip_len]
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)))
        return clip

    if group == "auto_flagged_fp":
        y, _ = librosa.load(path, sr=config.SAMPLE_RATE, mono=True)
        if len(y) >= clip_len:
            return y[:clip_len]
        return np.pad(y, (0, clip_len - len(y)))

    # Reference and BirdNET clips have no window convention of their own.
    return data_loader.load_audio_file(path, crop="loudest")


def _render(args):
    """Worker: one clip -> one uint8 image, or None."""
    path, source = args
    try:
        import preprocessing
        y = load_window(os.path.join(REPO, path), source)
        if y is None or not len(y) or not np.isfinite(y).all():
            return None
        img = preprocessing.preprocess_audio(y)
        return np.ascontiguousarray(img, dtype=np.uint8)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest",
                    default=os.path.join(REPO, "data/outputs/v13_manifest.csv"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data/outputs"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    n = len(manifest)

    import config
    h, w, c = config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS
    img_path = os.path.join(args.out_dir, "v13_images.npy")
    idx_path = os.path.join(args.out_dir, "v13_index.csv")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Packing {n} clips -> {img_path}")
    print(f"  {n} x {h}x{w}x{c} uint8 = {n * h * w * c / 1e9:.2f} GB")
    print(f"  {args.workers} workers\n")

    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                       shape=(n, h, w, c))
    ok = np.zeros(n, dtype=bool)

    from concurrent.futures import ProcessPoolExecutor
    jobs = list(zip(manifest["path"], manifest["source"]))
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for lo in range(0, n, args.chunk):
            hi = min(n, lo + args.chunk)
            for j, img in enumerate(pool.map(_render, jobs[lo:hi])):
                if img is not None:
                    images[lo + j] = img
                    ok[lo + j] = True
            done = hi
            rate = done / (time.time() - t0)
            eta = (n - done) / rate / 60 if rate else 0
            print(f"\r  {done}/{n}  {rate:.0f} clips/s  "
                  f"eta {eta:.1f} min  failed {done - int(ok[:done].sum())}",
                  end="", flush=True)
    images.flush()

    manifest = manifest.reset_index(drop=True)
    manifest["row"] = range(n)
    manifest["ok"] = ok
    manifest.to_csv(idx_path, index=False)

    print(f"\n\nWrote {img_path}")
    print(f"Wrote {idx_path}")
    bad = int((~ok).sum())
    if bad:
        print(f"\n{bad} clips failed to render and are marked ok=False. They "
              f"keep their row so the index stays aligned; drop them by "
              f"filtering on `ok` -- never by position.")
        for _, r in manifest[~manifest["ok"]].head(5).iterrows():
            print(f"   {r['source']:28s} {r['path']}")
    print(f"\nLabel counts (renderable only):")
    print(manifest[manifest["ok"]]["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
