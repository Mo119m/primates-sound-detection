"""
Run detection on IPA field recording stations.

Usage (from repo root or Colab):
    python scripts/run_detection_ipa.py --station IPA1ST --model path/to/best_model.h5

Processes all WAV files for the given station that fall within the configured
time window (default 05:30–10:30), runs sliding-window detection, and saves
per-file CSV results plus an aggregated summary.
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
import data_loader
import detection


def main():
    parser = argparse.ArgumentParser(description="Run detection on an IPA station")
    parser.add_argument('--station', type=str, default='IPA1ST',
                        help='Station folder name (e.g. IPA1ST)')
    parser.add_argument('--model', type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5'),
                        help='Path to trained .h5 model')
    parser.add_argument('--threshold', type=float,
                        default=config.DETECTION_CONFIDENCE_THRESHOLD,
                        help='Confidence threshold')
    parser.add_argument('--no-time-filter', action='store_true',
                        help='Disable time filtering (process all 24h)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for CSVs (default: detections/<station>)')
    args = parser.parse_args()

    # --- Load model ---
    from tensorflow import keras
    print(f"Loading model from {args.model}")
    model = keras.models.load_model(args.model)
    print(f"Model loaded — {config.N_CLASSES} classes: {config.CLASS_NAMES}")

    # --- Gather files ---
    use_filter = not args.no_time_filter
    files = data_loader.get_ipa_station_files(args.station, time_filter=use_filter)
    if not files:
        print("No files found. Check IPA_ROOT and station name.")
        return

    # --- Output dir ---
    out_dir = args.output or os.path.join(config.DETECTION_OUTPUT_DIR, args.station)
    os.makedirs(out_dir, exist_ok=True)

    # --- Run detection ---
    all_dfs = []
    t0 = time.time()
    for i, fpath in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}] {os.path.basename(fpath)}")
        print(f"{'='*60}")

        df = detection.detect_in_long_audio(model, fpath, args.threshold)
        if len(df) > 0:
            df['source_file'] = os.path.basename(fpath)
        detection.save_detections(df, os.path.basename(fpath), output_dir=out_dir)
        all_dfs.append(df)

    elapsed = time.time() - t0

    # --- Aggregate summary ---
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    summary_path = os.path.join(out_dir, f'{args.station}_summary.csv')
    if len(combined) > 0:
        combined.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — {args.station}")
    print(f"  Files processed: {len(files)}")
    print(f"  Total detections: {len(combined)}")
    print(f"  Elapsed: {elapsed/60:.1f} min")
    if len(combined) > 0:
        print(f"\n  Per-species breakdown:")
        for sp in sorted(combined['species'].unique()):
            n = len(combined[combined['species'] == sp])
            avg = combined[combined['species'] == sp]['confidence'].mean()
            print(f"    {sp:20s}: {n:4d}  (avg conf {avg:.3f})")
    print(f"\n  Summary CSV: {summary_path}")
    print(f"  Per-file CSVs: {out_dir}/")


if __name__ == '__main__':
    main()
