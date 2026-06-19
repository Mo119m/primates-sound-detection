"""
Run the three-filter automatic false-positive cleanup over detection CSVs.

Usage (from repo root or Colab):
    python scripts/run_auto_cleanup.py \
        --detection-dir /content/drive/MyDrive/primates-data/outputs/detections/IPA1ST

Produces, under <output>/:
    clean_detections.csv       - passed all three filters (trust without listening)
    suspicious_detections.csv  - flagged, with a flag_reason column
    auto_flagged_fp/<reason>/   - >=2-flag clips to fold into Background

Requires tensorflow-hub for the YAMNet filter:
    pip install tensorflow-hub
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
import auto_cleanup


def main():
    parser = argparse.ArgumentParser(description="Auto-cleanup detection false positives")
    parser.add_argument('--model', type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, 'best_model.h5'),
                        help='Path to trained .h5 model')
    parser.add_argument('--detection-dir', type=str, default=None,
                        help='Dir with *_detections.csv (default: config.DETECTION_OUTPUT_DIR)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output dir (default: OUTPUT_ROOT/auto_cleanup)')
    parser.add_argument('--percentile', type=int, default=95,
                        help='In-distribution percentile for the Mahalanobis cutoff')
    parser.add_argument('--isolation-window', type=float, default=30.0,
                        help='Temporal-isolation neighbour window in seconds')
    parser.add_argument('--no-save-clips', action='store_true',
                        help='Do not write hard-negative WAV clips')
    parser.add_argument('--no-cache', action='store_true',
                        help='Recompute Mahalanobis stats instead of using the cache')
    parser.add_argument('--mahal-calibration', type=str, default='detections',
                        choices=['detections', 'training'],
                        help='Calibrate Mahalanobis on detection distances '
                             '(robust to domain shift) or training distances')
    args = parser.parse_args()

    auto_cleanup.run_auto_cleanup(
        model_path=args.model,
        detection_dir=args.detection_dir,
        output_dir=args.output,
        percentile=args.percentile,
        isolation_window_s=args.isolation_window,
        save_clips=not args.no_save_clips,
        use_cached_stats=not args.no_cache,
        mahal_calibration=args.mahal_calibration,
    )


if __name__ == '__main__':
    main()
