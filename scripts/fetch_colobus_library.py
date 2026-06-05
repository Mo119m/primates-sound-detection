"""
Fetch and slice Colobus guereza recordings from the Macaulay Library.

The expert-labelled Colobus guereza set (172 clips) is too small and gives the
classifier a loose, poorly-defined decision boundary (the auto-cleanup
Mahalanobis threshold for Colobus runs ~2x higher than Cernic). This script
augments the positive set with public Macaulay Library recordings.

Workflow
--------
1. Sign in to https://search.macaulaylibrary.org (free Cornell Lab account).
2. Open the species audio catalog, e.g.
   https://search.macaulaylibrary.org/catalog?taxonCode=t-10558596&includeChildTaxa=true&mediaType=audio
3. Click "Export" to download a CSV of the results, OR copy the ML catalog
   numbers (the digits after "ML", e.g. ML657223371 -> 657223371) into a plain
   text file, one per line.
4. Run this script pointing at that CSV / text file.

Each recording is downloaded from the Macaulay CDN, then sliced into
overlapping ``CLIP_DURATION`` second windows (``WINDOW_STRIDE`` second hop, the
same geometry the detector uses). A Colobus roar is a long, rhythmic bout, so
slicing the whole bout into overlapping windows captures the different phases of
the roar rather than just the single loudest snort -- which is exactly the
temporal structure a single ``loudest`` crop throws away. Near-silent windows
are skipped so we do not feed empty clips into the positive class.

Please record the ML asset IDs and recordists you use; Macaulay media must be
cited (recordist + asset ID) in any publication.
"""

import argparse
import csv
import os
import re
import sys
import time

import librosa
import numpy as np
import requests
import soundfile as sf

# Allow importing the project config whether run from repo root or scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src import config
    SAMPLE_RATE = config.SAMPLE_RATE
    CLIP_DURATION = config.CLIP_DURATION
    WINDOW_STRIDE = config.WINDOW_STRIDE
except Exception:  # pragma: no cover - fall back to the known defaults
    SAMPLE_RATE = 44100
    CLIP_DURATION = 2.0
    WINDOW_STRIDE = 1.0

# Macaulay serves the original media for an asset from this CDN path. The asset
# ID is the numeric part of the ML catalog number (ML657223371 -> 657223371).
CDN_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{asset_id}/audio"

HEADERS = {
    # A browser-like UA avoids the API's bot filtering on plain requests.
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def parse_ids(path: str) -> list:
    """Read ML catalog numbers from a Macaulay CSV export or a plain text file.

    Accepts any file containing the numbers: a CSV with a column whose header
    mentions "catalog"/"ML"/"asset", or a text file with one ID per line.
    Strips an optional leading "ML".
    """
    ids = []
    with open(path, newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        if "," in sample and ("\n" in sample):
            reader = csv.DictReader(f)
            if reader.fieldnames:
                # Find the column most likely to hold the catalog number.
                col = None
                for name in reader.fieldnames:
                    low = name.lower()
                    if "catalog" in low or low in ("ml", "asset id", "assetid", "asset_id"):
                        col = name
                        break
                if col is not None:
                    for row in reader:
                        raw = (row.get(col) or "").strip()
                        m = re.search(r"(\d{4,})", raw)
                        if m:
                            ids.append(m.group(1))
                    return ids
        # Fall back: treat every numeric token in the file as an ID.
        f.seek(0)
        for line in f:
            m = re.search(r"(\d{4,})", line)
            if m:
                ids.append(m.group(1))
    return ids


def download_asset(asset_id: str, tmp_dir: str) -> str:
    """Download one Macaulay asset to tmp_dir. Returns the local path or None."""
    url = CDN_URL.format(asset_id=asset_id)
    out_path = os.path.join(tmp_dir, f"ML{asset_id}.audio")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        if resp.status_code != 200:
            print(f"   ML{asset_id}: HTTP {resp.status_code}, skipped")
            return None
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return out_path
    except requests.RequestException as exc:
        print(f"   ML{asset_id}: download error ({exc}), skipped")
        return None


def slice_recording(audio_path: str, label: str, out_dir: str,
                    silence_rel_db: float = 25.0, prefix: str = "ML") -> int:
    """Slice one recording into overlapping windows and save the loud ones.

    A window is kept only if its RMS is within ``silence_rel_db`` dB of the
    recording's loudest window, which drops the quiet gaps between roar bouts.
    Returns the number of clips written.

    ``label`` + ``prefix`` form the output filename stem (e.g. ML657223371 for a
    downloaded asset, or the source filename for a local re-slice).
    """
    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        print(f"   {prefix}{label}: could not decode ({exc}), skipped")
        return 0

    # Guard against empty/corrupt decodes and any non-finite samples up front so
    # we never emit a zero-length or NaN clip (which blows up mel-spectrogram).
    if audio.size == 0:
        print(f"   {prefix}{label}: empty audio, skipped")
        return 0
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    win = int(CLIP_DURATION * SAMPLE_RATE)
    hop = int(WINDOW_STRIDE * SAMPLE_RATE)
    if len(audio) < win:
        # Pad short recordings up to one window with a tiny tail of themselves.
        audio = np.pad(audio, (0, win - len(audio)))

    starts = list(range(0, len(audio) - win + 1, hop))
    if not starts:
        starts = [0]

    # Per-window RMS, then a relative-energy gate against the loudest window.
    rms = np.array([
        float(np.sqrt(np.mean(audio[s:s + win].astype(np.float64) ** 2)) + 1e-12)
        for s in starts
    ])
    peak = rms.max()
    if peak <= 1e-9:
        print(f"   {prefix}{label}: silent recording, skipped")
        return 0
    floor = peak * (10 ** (-silence_rel_db / 20))

    written = 0
    for s, r in zip(starts, rms):
        if r < floor:
            continue
        clip = audio[s:s + win]
        clip_peak = float(np.max(np.abs(clip)))
        # Absolute silence guard: never write an effectively-empty window even
        # if it passed the recording-relative gate (a uniformly quiet bout).
        if clip.size != win or clip_peak < 1e-4:
            continue
        # Peak-normalise so library-recording levels match our pipeline.
        clip = clip / (clip_peak + 1e-9) * 0.95
        t = s / SAMPLE_RATE
        fname = f"{prefix}{label}__t{t:05.1f}s.wav"
        sf.write(os.path.join(out_dir, fname), clip.astype(np.float32), SAMPLE_RATE)
        written += 1
    return written


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff")


def run_local_mode(input_dir: str, output_dir: str, silence_rel_db: float):
    """Re-slice an existing local folder of clips into overlapping 2s windows.

    Used to expand the expert-labelled Colobus 5s clips: instead of the pipeline
    taking only the single loudest 2s window from each clip, this emits every
    loud 2s window so the model sees the full roar bout (different phases of the
    repeated roar pulses), multiplying the positive set several-fold.
    """
    files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(AUDIO_EXTS)
    )
    if not files:
        print(f"No audio files found in {input_dir}")
        return
    print(f"Re-slicing {len(files)} local clips from {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    total_clips, used = 0, 0
    for n, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        # Sanitise the stem for use as a filename prefix.
        stem = re.sub(r"[^\w.-]+", "_", stem)
        clips = slice_recording(path, stem, output_dir,
                                silence_rel_db=silence_rel_db, prefix="")
        if clips:
            total_clips += clips
            used += 1
        if n % 25 == 0:
            print(f"   [{n}/{len(files)}] {total_clips} windows so far")

    print(f"\nDone: {total_clips} windows from {used}/{len(files)} clips "
          f"-> {output_dir}")


def run_download_mode(ids_path: str, output_dir: str, silence_rel_db: float,
                      delay: float):
    ids = parse_ids(ids_path)
    if not ids:
        print(f"No ML catalog numbers found in {ids_path}")
        return
    # De-duplicate while preserving order.
    seen = set()
    ids = [i for i in ids if not (i in seen or seen.add(i))]
    print(f"Found {len(ids)} unique recordings to fetch")

    os.makedirs(output_dir, exist_ok=True)
    tmp_dir = os.path.join(output_dir, "_downloads")
    os.makedirs(tmp_dir, exist_ok=True)

    total_clips = 0
    used = []
    for n, asset_id in enumerate(ids, 1):
        print(f"[{n}/{len(ids)}] ML{asset_id}")
        path = download_asset(asset_id, tmp_dir)
        if path is None:
            continue
        clips = slice_recording(path, asset_id, output_dir,
                                silence_rel_db=silence_rel_db, prefix="ML")
        if clips:
            print(f"   -> {clips} clips")
            total_clips += clips
            used.append(asset_id)
        os.remove(path)
        time.sleep(delay)

    # Drop a citation manifest next to the clips for the methods write-up.
    manifest = os.path.join(output_dir, "macaulay_assets_used.txt")
    with open(manifest, "w") as f:
        f.write("# Macaulay Library asset IDs used (cite recordist + asset ID)\n")
        for asset_id in used:
            f.write(f"ML{asset_id}\thttps://macaulaylibrary.org/asset/{asset_id}\n")

    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"\nDone: {total_clips} clips from {len(used)} recordings -> {output_dir}")
    print(f"Asset manifest: {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids",
                     help="CSV export from Macaulay, or text file of ML IDs "
                          "(one per line) -> download then slice")
    src.add_argument("--local",
                     help="Folder of existing audio clips to re-slice into "
                          "overlapping 2s windows (no download)")
    parser.add_argument("--output", required=True,
                        help="Output folder for the sliced 2s clips "
                             "(e.g. .../species/Colobus guereza 2s windows)")
    parser.add_argument("--silence-rel-db", type=float, default=25.0,
                        help="Drop windows quieter than this many dB below the "
                             "recording's loudest window (default 25)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between downloads (be polite)")
    args = parser.parse_args()

    if args.local:
        run_local_mode(args.local, args.output, args.silence_rel_db)
    else:
        run_download_mode(args.ids, args.output, args.silence_rel_db, args.delay)


if __name__ == "__main__":
    main()
