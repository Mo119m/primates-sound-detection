"""
Harvest confirmed field false positives into the Background class.

After per-station auto_cleanup with ``save_all_clips=True`` every detection is
saved as a 2s WAV under::

    {cleanup_root}/{station}/clean_clips/{species}/*.wav      (0 flags)
    {cleanup_root}/{station}/suspicious_clips/{species}/*.wav (>=1 flag)

At a dev station where the field reviewer has confirmed a species has **no real
calls**, every one of these clips is a false positive recorded under the exact
conditions the detector runs in -- the same AudioMoth, the same forest
soundscape (birds, insects, sawing, wind). Folding them into the Background
class is high-value, distribution-matched hard-negative mining that attacks the
field false-alarm rate far more directly than random ImageNet-style negatives.

The ``clean_clips`` are the most valuable: they fooled the classifier *and* all
three auto-cleanup filters, yet a human confirmed they are FP. None of them are
in the training set yet (only the >=2-flag "strong FP" clips were auto-saved to
``auto_flagged_fp``). ``suspicious_clips`` overlap partially with those already
-saved strong FPs; the duplication is harmless (at worst it mildly upweights a
few hundred clips), so both folders are harvested by default.

IMPORTANT
---------
Only run this on DEV stations (IPA1-18). Never harvest the held-out test
stations (IPA19/20) or the final evaluation is contaminated -- the script hard
-refuses any station whose name contains 19 or 20 as a safety rail.

Two station groups handle the label-safety nuance::

    --stations-all           BOTH target species confirmed FP -> harvest all
    --stations-colobus-only  real Cernic present but Colobus FP -> Colobus only

Example
-------
    python scripts/mine_field_negatives.py \
        --cleanup-root "/content/drive/MyDrive/primates-sound-detection/outputs/auto_cleanup" \
        --output       "/content/drive/MyDrive/primates-sound-detection/background/field_fp_negatives" \
        --stations-all IPA1ST IPA2ST IPA4ST IPA6ST IPA7ST IPA8ST IPA10ST IPA11ST IPA13ST IPA14ST IPA16ST \
        --stations-colobus-only IPA15ST IPA17ST IPA18ST
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
    SAMPLE_RATE = config.SAMPLE_RATE
    CLIP_DURATION = config.CLIP_DURATION
except Exception:  # pragma: no cover - fall back to the known defaults
    SAMPLE_RATE = 44100
    CLIP_DURATION = 2.0

CLIP_SUBDIRS = ("clean_clips", "suspicious_clips")
DEFAULT_SPECIES = ("Cernic", "Colobus_guereza")
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


def harvest_station(cleanup_root: str, station: str, species_list, out_dir: str) -> int:
    """Copy every saved FP clip for ``species_list`` at ``station`` into out_dir.

    Returns the number of clips written. Missing folders (a station that was not
    run with save_all_clips, or a species with no detections) are skipped
    silently so the caller can pass a superset of station names.
    """
    written = 0
    found_any = False
    for species in species_list:
        for sub in CLIP_SUBDIRS:
            src = os.path.join(cleanup_root, station, sub, species)
            if not os.path.isdir(src):
                continue
            found_any = True
            for fname in sorted(os.listdir(src)):
                if not fname.lower().endswith(".wav"):
                    continue
                audio = _load_clean_window(os.path.join(src, fname))
                if audio is None:
                    continue
                # Prefix with station + source bucket so filenames never collide
                # and the negative stays traceable back to where it came from.
                out_name = f"{station}__{species}__{sub}__{fname}"
                sf.write(os.path.join(out_dir, out_name), audio, SAMPLE_RATE)
                written += 1
    if not found_any:
        print(f"   {station}: no clip folders found (skipped)")
    else:
        print(f"   {station}: harvested {written} clips ({', '.join(species_list)})")
    return written


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
    parser.add_argument("--stations-all", nargs="*", default=[],
                        help="stations where BOTH species are confirmed FP "
                             "(harvest every species' clips)")
    parser.add_argument("--stations-colobus-only", nargs="*", default=[],
                        help="stations with real Cernic but FP Colobus "
                             "(harvest Colobus clips only)")
    parser.add_argument("--species-all", nargs="*", default=list(DEFAULT_SPECIES),
                        help="species subfolders to take for --stations-all")
    parser.add_argument("--colobus-name", default=COLOBUS_NAME,
                        help="species subfolder name for the Colobus-only group")
    args = parser.parse_args()

    # Safety rail: never mine a held-out test station.
    bad = [s for s in (args.stations_all + args.stations_colobus_only)
           if _is_test_station(s)]
    if bad:
        parser.error(
            f"refusing to mine held-out test station(s): {', '.join(bad)}. "
            "IPA19/20 must stay untouched for the final evaluation.")

    if not args.stations_all and not args.stations_colobus_only:
        parser.error("nothing to do: pass --stations-all and/or "
                     "--stations-colobus-only")

    os.makedirs(args.output, exist_ok=True)

    total = 0
    if args.stations_all:
        print(f"\nHarvesting ALL species from {len(args.stations_all)} stations:")
        for st in args.stations_all:
            total += harvest_station(args.cleanup_root, st, args.species_all,
                                     args.output)
    if args.stations_colobus_only:
        print(f"\nHarvesting Colobus only from "
              f"{len(args.stations_colobus_only)} stations:")
        for st in args.stations_colobus_only:
            total += harvest_station(args.cleanup_root, st, [args.colobus_name],
                                     args.output)

    print(f"\nDone: {total} field-FP negatives written to {args.output}")
    print("Add 'background/field_fp_negatives' to config.BACKGROUND_FOLDERS "
          "(already done if you pulled the latest branch), then retrain.")


if __name__ == "__main__":
    main()
