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


_BG_POOL = None


def _seed_for(path):
    """A stable seed per clip, so re-packing reproduces the same beds.

    The bed is drawn at random and this pack is written once and trained on for
    many epochs, so an unseeded draw would make the dataset differ between two
    runs of the same command. Hashing the path keeps it reproducible while still
    giving every clip a different bed.
    """
    import hashlib
    return int(hashlib.md5(path.encode("utf-8")).hexdigest()[:8], 16)


def _background_pool(limit_per_folder=250):
    """Real ambient waveforms, loaded once per worker process."""
    global _BG_POOL
    if _BG_POOL is not None:
        return _BG_POOL
    import librosa
    import config
    import data_loader
    data_dir = os.path.join(REPO, "data")
    pool = []
    for folder in config.BACKGROUND_FOLDERS:
        for q in data_loader.scan_audio_files(data_dir, folder)[:limit_per_folder]:
            try:
                y, _ = librosa.load(q, sr=config.SAMPLE_RATE, mono=True)
            except Exception:
                continue
            if y.size >= int(config.CLIP_DURATION * config.SAMPLE_RATE):
                pool.append(y)
    if not pool:
        raise RuntimeError(
            "no background audio available to embed short clips in. Packing "
            "would zero-pad them, which makes any all-short class separable by "
            "the silence alone. Fix the background folders before packing.")
    _BG_POOL = pool
    return _BG_POOL


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
    # A clip shorter than the window has to be filled, and WHAT it is filled
    # with decides whether the class is learnable or trivially separable.
    # Zero-padding leaves a block of digital silence that occurs in no field
    # recording, so a class made entirely of short clips becomes "the one with
    # the silence in it". This is not hypothetical: packing the 612 guereza
    # pulses without a bed gave them a median flat-time-column fraction of 0.799
    # against 0.000 for every other class, and the 326 Cernic syllables sit at
    # 0.946. data_loader.embed_in_background exists precisely to prevent this and
    # was simply never reached from here, because load_audio_file only embeds
    # when it is handed a pool.
    # Seeded here rather than through load_audio_file, which has no seed
    # parameter and is shared with the older training path. A worker renders one
    # clip at a time and the only draw inside this call is the choice of bed and
    # SNR, so setting the global seed immediately before it is both sufficient
    # and contained.
    np.random.seed(_seed_for(path))
    return data_loader.load_audio_file(
        path, crop="loudest", background_pool=_background_pool())


_BG_SPECS = None


def _background_specs(limit=200):
    """Mel-spectrograms of real ambient, for the noise-mixing augmentation."""
    global _BG_SPECS
    if _BG_SPECS is not None:
        return _BG_SPECS
    import preprocessing
    pool = _background_pool()
    idx = np.linspace(0, len(pool) - 1, min(limit, len(pool))).astype(int)
    _BG_SPECS = [preprocessing.audio_to_melspectrogram(pool[i]) for i in idx]
    return _BG_SPECS


def _render(args):
    """Worker: one clip -> one uint8 image, or None.

    ``aug`` selects a variant. 0 is the clip itself; anything higher applies one
    of the transformations from src/augmentation.py to the mel-spectrogram
    before it is normalised and resized, which is where those functions expect
    to operate. The variant index seeds the draw, so the same row always yields
    the same image and a rebuilt pack is identical to the one it replaces.
    """
    path, source, aug = args
    try:
        import preprocessing
        y = load_window(os.path.join(REPO, path), source)
        if y is None or not len(y) or not np.isfinite(y).all():
            return None
        if not aug:
            img = preprocessing.preprocess_audio(y)
            return np.ascontiguousarray(img, dtype=np.uint8)

        import random as _random
        import augmentation as A
        spec = preprocessing.audio_to_melspectrogram(y)
        shape = spec.shape
        seed = (_seed_for(path) + int(aug) * 7919) % (2 ** 31)
        _random.seed(seed)
        np.random.seed(seed)
        # Cycle the transformations so a class needing twenty variants gets a
        # spread rather than twenty draws of the same one.
        kind = int(aug) % 4
        if kind == 0:
            bg = _background_specs()
            spec = A.add_background_noise(spec, _random.choice(bg))
        elif kind == 1:
            spec = A.resize_to_original_shape(A.time_chop(spec.copy()), shape)
        elif kind == 2:
            spec = A.resize_to_original_shape(A.freq_chop(spec.copy()), shape)
        else:
            spec = A.translate(spec.copy())
        img = preprocessing.spectrogram_to_rgb(
            preprocessing.resize_spectrogram(
                preprocessing.normalize_spectrogram(spec)))
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

    # Build both outputs beside the real ones and swap them in together at the
    # end. The pair has to move as a unit: the index carries the row number that
    # addresses the image array, so a run killed part-way used to leave a new,
    # half-written image file next to the *previous* index, and every row the
    # trainer read after that pointed at the wrong clip. open_memmap also
    # preallocates, so the half-written file is full-size and full of zeros --
    # nothing downstream could tell it was incomplete.
    img_tmp, idx_tmp = img_path + ".partial", idx_path + ".partial"
    images = np.lib.format.open_memmap(img_tmp, mode="w+", dtype=np.uint8,
                                       shape=(n, h, w, c))
    ok = np.zeros(n, dtype=bool)

    from concurrent.futures import ProcessPoolExecutor
    aug = (manifest["aug"] if "aug" in manifest.columns
           else pd.Series(0, index=manifest.index))
    jobs = list(zip(manifest["path"], manifest["source"], aug))
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
    del images

    manifest = manifest.reset_index(drop=True)
    manifest["row"] = range(n)
    manifest["ok"] = ok
    manifest.to_csv(idx_tmp, index=False)

    os.replace(img_tmp, img_path)
    os.replace(idx_tmp, idx_path)

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
