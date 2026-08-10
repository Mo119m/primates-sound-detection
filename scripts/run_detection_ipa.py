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
    # The deployment that produced the 6189 reviewed detections ran over the
    # whole day: 81.5% of those detections fall outside 05:30-10:30, so a run
    # with the filter on reproduces less than a fifth of them and cannot be
    # compared with the review. The paper also settled on not using the filter
    # at all. Off is therefore the default, and turning it on is the choice that
    # has to be made explicitly.
    parser.add_argument('--time-filter', action='store_true',
                        help=f'Only process files starting between '
                             f'{config.TIME_FILTER_START} and '
                             f'{config.TIME_FILTER_END}. Off by default: the '
                             f'field results were produced over all 24h.')
    parser.add_argument('--no-time-filter', action='store_true',
                        help=argparse.SUPPRESS)   # accepted, now the default
    parser.add_argument('--time-window', type=str, default=None,
                        help="HH:MM-HH:MM overriding config.TIME_FILTER_* when "
                             "--time-filter is set. Use 05:00-08:00 for a "
                             "guereza dawn search; the shipped 05:00-19:00 is "
                             "the whole day-active period.")
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for CSVs (default: detections/<station>)')
    args = parser.parse_args()

    # --- Load model ---
    # Via model.load_trained_model, not keras.models.load_model: the V11/V12
    # 'temporal_freqpos' head contains a custom FrequencyCoord layer, and a bare
    # load raises on it.
    import model as model_module
    model = model_module.load_trained_model(args.model)
    print(f"Model loaded — {config.N_CLASSES} classes: {config.CLASS_NAMES}")

    # --- Gather files ---
    use_filter = args.time_filter
    window = tuple(args.time_window.split('-')) if args.time_window else None
    if window and len(window) != 2:
        sys.exit(f"--time-window must be HH:MM-HH:MM, got {args.time_window!r}")
    # Report the window that will actually be used. This line used to describe
    # only --time-filter and said "OFF (all 24h)" even when --time-window had
    # restricted the run to a few hours, which is the kind of log that gets
    # believed later.
    if window:
        print(f"Time filter: ON ({window[0]}-{window[1]}, from --time-window)")
    elif use_filter:
        print(f"Time filter: ON ({config.TIME_FILTER_START}-"
              f"{config.TIME_FILTER_END}, from config)")
    else:
        print("Time filter: OFF (all 24h, as deployed)")
    files = data_loader.get_ipa_station_files(args.station, time_filter=use_filter,
                                              window=window)
    if not files:
        # Exit non-zero. This used to `return`, i.e. exit 0, so an overnight
        # sweep or any wrapper read "no files" as success and produced nothing.
        # IPA_ROOT defaults to data/field_recordings/, which is empty in a fresh
        # checkout, so this is the DEFAULT outcome, not an edge case.
        sys.exit(f"No files found for {args.station} under {config.IPA_ROOT}\n"
                 f"Set PRIMATE_IPA_ROOT to the drive holding the IPA* folders, "
                 f"e.g.\n  PRIMATE_IPA_ROOT='D:/Gabon raw acoustic data National Park'")

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
