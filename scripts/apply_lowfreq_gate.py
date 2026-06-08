"""
Apply the low-frequency spectral-energy gate to saved Colobus detection clips.

The detection workflow exports one WAV clip per detection under
    <OUTPUT_ROOT>/detection_clips_<version>/<STATION>/<species>/
This script scores every Colobus clip by its low-frequency energy fraction
(see config.LOWFREQ_GATE_CUTOFF / LOWFREQ_GATE_THRESHOLD) and reports which
detections the gate keeps versus rejects. Real C. guereza roars are
low-frequency and pass; high-frequency out-of-distribution false positives
(insects, cicadas) are rejected. The gate runs on saved clips, so it needs no
model and no retraining.

Usage (from repo root or Colab):
    python scripts/apply_lowfreq_gate.py \
        --clip-root /content/drive/MyDrive/primates-data/outputs/detection_clips_model_v10 \
        --station IPA1ST

    # all stations under the clip root:
    python scripts/apply_lowfreq_gate.py --clip-root .../detection_clips_model_v10

Outputs, per station, a <STATION>_colobus_gated.csv with columns
    clip, lowfreq_ratio, keep
and (with --move-rejected) moves rejected clips into a _rejected_by_gate/
subfolder so the kept clips are ready for review.
"""

import argparse
import glob
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import librosa
import pandas as pd

import config
import detection


def gate_station(clip_root, station, species, cutoff, threshold,
                 move_rejected=False):
    """Score and optionally prune one station's Colobus clips."""
    clip_dir = os.path.join(clip_root, station, species)
    clips = sorted(glob.glob(os.path.join(clip_dir, '*.wav')))
    if not clips:
        print(f"  {station}: no {species} clips")
        return pd.DataFrame(columns=['clip', 'lowfreq_ratio', 'keep'])

    rows = []
    for path in clips:
        audio, _ = librosa.load(path, sr=config.SAMPLE_RATE)
        ratio = detection.lowfreq_energy_ratio(audio, cutoff=cutoff)
        keep = True if ratio is None else ratio >= threshold
        rows.append({'clip': os.path.basename(path),
                     'lowfreq_ratio': None if ratio is None else round(ratio, 3),
                     'keep': keep,
                     '_path': path})
    df = pd.DataFrame(rows)

    kept = int(df['keep'].sum())
    print(f"  {station}: {len(df)} {species} -> keep {kept}, reject {len(df) - kept}")

    out_csv = os.path.join(clip_root, station, f"{station}_colobus_gated.csv")
    df.drop(columns='_path').to_csv(out_csv, index=False)

    if move_rejected:
        rej_dir = os.path.join(clip_dir, '_rejected_by_gate')
        os.makedirs(rej_dir, exist_ok=True)
        for _, row in df[~df['keep']].iterrows():
            shutil.move(row['_path'], os.path.join(rej_dir, row['clip']))

    return df.drop(columns='_path')


def main():
    parser = argparse.ArgumentParser(
        description="Low-frequency spectral-energy gate for Colobus detections")
    parser.add_argument('--clip-root', type=str, required=True,
                        help='Root holding <STATION>/<species>/ clip folders')
    parser.add_argument('--station', type=str, default=None,
                        help='Single station (default: every station under the root)')
    parser.add_argument('--species', type=str, default='Colobus_guereza',
                        help='Species folder to gate')
    parser.add_argument('--cutoff', type=float, default=config.LOWFREQ_GATE_CUTOFF,
                        help='Low-frequency cutoff in Hz')
    parser.add_argument('--threshold', type=float, default=config.LOWFREQ_GATE_THRESHOLD,
                        help='Minimum low-frequency energy fraction to keep')
    parser.add_argument('--move-rejected', action='store_true',
                        help='Move rejected clips into a _rejected_by_gate/ subfolder')
    args = parser.parse_args()

    if args.station:
        stations = [args.station]
    else:
        stations = sorted(s for s in os.listdir(args.clip_root)
                          if os.path.isdir(os.path.join(args.clip_root, s)))

    print(f"Low-frequency gate: cutoff={args.cutoff} Hz, threshold={args.threshold}\n")
    all_gated = []
    for station in stations:
        df = gate_station(args.clip_root, station, args.species,
                          args.cutoff, args.threshold, args.move_rejected)
        if len(df):
            all_gated.append(df.assign(station=station))

    if all_gated:
        combined = pd.concat(all_gated, ignore_index=True)
        kept = int(combined['keep'].sum())
        print(f"\nTotal: {len(combined)} {args.species} detections "
              f"-> keep {kept}, reject {len(combined) - kept}")


if __name__ == '__main__':
    main()
