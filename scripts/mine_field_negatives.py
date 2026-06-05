"""
Harvest confirmed field false positives into the Background class.

After per-station auto_cleanup with ``save_all_clips=True`` every detection is
saved as a 2s WAV under::

    {cleanup_root}/{station}/clean_clips/{species}/*.wav      (0 flags)
    {cleanup_root}/{station}/suspicious_clips/{species}/*.wav (>=1 flag)

These are false positives recorded under the exact conditions the detector runs
in -- the same AudioMoth, the same forest soundscape (birds, insects, sawing,
wind, human speech). Folding them into the Background class is high-value,
distribution-matched hard-negative mining that attacks the field false-alarm
rate far more directly than random ImageNet-style negatives.

LABEL SAFETY (the whole point of this script)
---------------------------------------------
The cardinal rule is: never put a real target call into Background. The two
target species need very different handling:

* **Colobus guereza** -- the reviewer has found *no* real Colobus at any dev
  station, so every saved Colobus clip is a confirmed FP. Harvest them all.

* **Cernic** (Cercopithecus nictitans) -- real putty-nose calls ARE present at
  many stations, so harvesting every Cernic clip by station would poison the
  Background with real calls. Instead each Cernic clip is gated by YAMNet:
  it is kept only if YAMNet's top AudioSet class is a clearly non-primate sound
  (bird / insect / wind / speech / mechanical -- the set in auto_cleanup's
  DEFAULT_SUSPICIOUS_YAMNET). A genuine Cernic call tags as "Animal" /
  "Wild animals" (not in that set) and is therefore left out. We would rather
  miss a negative than mislabel a positive, so this errs on the safe side and
  needs no per-station Cernic ground truth at all.

IMPORTANT: only run this on DEV stations (IPA1-18). The held-out test stations
IPA19/20 must never contribute to training -- the script hard-refuses any
station whose name contains 19 or 20.

Example
-------
    python scripts/mine_field_negatives.py \
        --cleanup-root "/content/drive/MyDrive/primates-sound-detection/outputs/auto_cleanup" \
        --output       "/content/drive/MyDrive/primates-sound-detection/background/field_fp_negatives"

By default it auto-discovers every (non-test) station folder under
--cleanup-root. Pass --stations to restrict the set, or --cernic-all-stations
to harvest *all* Cernic clips (ungated) from stations you have personally
certified contain no real Cernic.
"""

import argparse
import os
import sys

import librosa
import numpy as np
import soundfile as sf

# Allow importing the project config whether run from repo root or scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src import config
    from src.auto_cleanup import DEFAULT_SUSPICIOUS_YAMNET
    SAMPLE_RATE = config.SAMPLE_RATE
    CLIP_DURATION = config.CLIP_DURATION
except Exception:  # pragma: no cover - fall back to the known defaults
    from auto_cleanup import DEFAULT_SUSPICIOUS_YAMNET  # type: ignore
    SAMPLE_RATE = 44100
    CLIP_DURATION = 2.0

CLIP_SUBDIRS = ("clean_clips", "suspicious_clips")
CERNIC_NAME = "Cernic"
COLOBUS_NAME = "Colobus_guereza"

# Any station whose name contains one of these tokens is a held-out test
# station and must never be mined.
TEST_TOKENS = ("19", "20")


def _is_test_station(station: str) -> bool:
    return any(tok in station for tok in TEST_TOKENS)


def _load_clean_window(path: str):
    """Load a saved detection clip and return an exactly-2s, finite, non-silent
    waveform, or None if it should be skipped.

    The clips were written by the detector so they are normally already at the
    right sample rate and length, but we re-validate cheaply: an empty, silent
    or non-finite clip blows up the mel-spectrogram downstream (we hit exactly
    this with the Colobus re-slice), so guard against it here.
    """
    try:
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 - librosa backends raise many types
        print(f"     skip (decode error: {exc}): {os.path.basename(path)}")
        return None

    if audio.size == 0:
        return None
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    target = int(round(CLIP_DURATION * SAMPLE_RATE))
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    elif len(audio) > target:
        audio = audio[:target]

    if float(np.max(np.abs(audio))) < 1e-4:
        return None  # effectively silent
    return audio.astype(np.float32)


class YamnetGate:
    """Keep a clip only if YAMNet's top class is a clearly non-primate sound.

    Loaded lazily so a Colobus-only run never pays the model-load cost.
    """

    def __init__(self):
        import tensorflow_hub as hub  # local import: heavy, optional dependency
        import pandas as pd
        print("Loading YAMNet for Cernic label-safety gating...")
        self._yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        class_map = self._yamnet.class_map_path().numpy().decode("utf-8")
        self._classes = pd.read_csv(class_map)["display_name"].tolist()

    def top_class(self, clip: np.ndarray) -> str:
        import tensorflow as tf
        clip16 = librosa.resample(clip.astype(np.float32),
                                  orig_sr=SAMPLE_RATE, target_sr=16000)
        scores, _, _ = self._yamnet(clip16)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        return self._classes[int(np.argmax(mean_scores))]

    def is_nonprimate(self, clip: np.ndarray) -> bool:
        return self.top_class(clip) in DEFAULT_SUSPICIOUS_YAMNET


def _iter_clip_paths(cleanup_root, station, species):
    for sub in CLIP_SUBDIRS:
        src = os.path.join(cleanup_root, station, sub, species)
        if not os.path.isdir(src):
            continue
        for fname in sorted(os.listdir(src)):
            if fname.lower().endswith(".wav"):
                yield sub, os.path.join(src, fname), fname


def _write(out_dir, station, species, sub, fname, audio):
    # Prefix with station + source bucket so filenames never collide and the
    # negative stays traceable back to where it came from.
    out_name = f"{station}__{species}__{sub}__{fname}"
    sf.write(os.path.join(out_dir, out_name), audio, SAMPLE_RATE)


def harvest_colobus(cleanup_root, station, out_dir, colobus_name) -> int:
    """Harvest every Colobus clip (all confirmed FP -- no real Colobus anywhere)."""
    written = 0
    for sub, path, fname in _iter_clip_paths(cleanup_root, station, colobus_name):
        audio = _load_clean_window(path)
        if audio is None:
            continue
        _write(out_dir, station, colobus_name, sub, fname, audio)
        written += 1
    return written


def harvest_cernic_gated(cleanup_root, station, out_dir, gate, cernic_name) -> tuple:
    """Harvest only the Cernic clips YAMNet calls non-primate. Returns (kept, dropped)."""
    kept = dropped = 0
    for sub, path, fname in _iter_clip_paths(cleanup_root, station, cernic_name):
        audio = _load_clean_window(path)
        if audio is None:
            continue
        if gate.is_nonprimate(audio):
            _write(out_dir, station, cernic_name, sub, fname, audio)
            kept += 1
        else:
            dropped += 1  # tagged as Animal/Wild animals -> possibly real, keep out
    return kept, dropped


def harvest_cernic_all(cleanup_root, station, out_dir, cernic_name) -> int:
    """Harvest every Cernic clip ungated -- ONLY for stations certified Cernic-free."""
    written = 0
    for sub, path, fname in _iter_clip_paths(cleanup_root, station, cernic_name):
        audio = _load_clean_window(path)
        if audio is None:
            continue
        _write(out_dir, station, cernic_name, sub, fname, audio)
        written += 1
    return written


def discover_stations(cleanup_root):
    out = []
    for name in sorted(os.listdir(cleanup_root)):
        if name == "auto_flagged_fp":
            continue
        if not os.path.isdir(os.path.join(cleanup_root, name)):
            continue
        out.append(name)
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cleanup-root", required=True,
                        help="outputs/auto_cleanup root holding per-station "
                             "subfolders with clean_clips/ and suspicious_clips/")
    parser.add_argument("--output", required=True,
                        help="destination Background folder for the mined FP "
                             "clips (e.g. .../background/field_fp_negatives)")
    parser.add_argument("--stations", nargs="*", default=None,
                        help="stations to mine (default: auto-discover all "
                             "non-test station folders under --cleanup-root)")
    parser.add_argument("--cernic-all-stations", nargs="*", default=[],
                        help="stations you have certified contain NO real Cernic: "
                             "their Cernic clips are harvested ungated")
    parser.add_argument("--cernic-name", default=CERNIC_NAME)
    parser.add_argument("--colobus-name", default=COLOBUS_NAME)
    parser.add_argument("--no-cernic", action="store_true",
                        help="skip Cernic entirely; harvest only Colobus")
    args = parser.parse_args()

    stations = args.stations if args.stations is not None else discover_stations(args.cleanup_root)

    # Safety rail: never mine a held-out test station, however it got into the list.
    bad = [s for s in (stations + args.cernic_all_stations) if _is_test_station(s)]
    if bad:
        parser.error(
            f"refusing to mine held-out test station(s): {', '.join(bad)}. "
            "IPA19/20 must stay untouched for the final evaluation.")
    if not stations:
        parser.error(f"no station folders found under {args.cleanup_root}")

    certified = set(args.cernic_all_stations)
    os.makedirs(args.output, exist_ok=True)

    # Only load YAMNet if we actually need to gate Cernic clips.
    gate = None
    if not args.no_cernic and any(s not in certified for s in stations):
        gate = YamnetGate()

    total_col = total_cer = total_cer_drop = 0
    print(f"\nMining {len(stations)} stations -> {args.output}")
    for st in stations:
        col = harvest_colobus(args.cleanup_root, st, args.output, args.colobus_name)
        if args.no_cernic:
            cer, drop = 0, 0
        elif st in certified:
            cer = harvest_cernic_all(args.cleanup_root, st, args.output, args.cernic_name)
            drop = 0
        else:
            cer, drop = harvest_cernic_gated(args.cleanup_root, st, args.output,
                                             gate, args.cernic_name)
        total_col += col
        total_cer += cer
        total_cer_drop += drop
        gate_note = "ungated" if st in certified else "yamnet-gated"
        print(f"   {st:10s} Colobus={col:4d}   Cernic={cer:4d} ({gate_note}, "
              f"{drop} kept out as possible-real)")

    print(f"\nDone -> {args.output}")
    print(f"   Colobus negatives: {total_col}")
    print(f"   Cernic negatives:  {total_cer}  "
          f"({total_cer_drop} Cernic clips withheld as possibly-real)")
    print(f"   TOTAL:             {total_col + total_cer}")
    print("\nConfig already lists 'background/field_fp_negatives' in "
          "BACKGROUND_FOLDERS, so just retrain to fold these in.")


if __name__ == "__main__":
    main()
