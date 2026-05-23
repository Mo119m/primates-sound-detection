"""
Filter IPA field recordings by time of day.

Copies only recordings within the target time window (default 05:30–10:30)
to a new directory, preserving the station/date folder structure.
Run this LOCALLY before uploading to Google Drive.

Usage:
    python scripts/filter_recordings_by_time.py \
        --input  "/path/to/Gabon raw acoustic data National Park" \
        --output "/path/to/filtered_field_recordings" \
        --start 05:30 --end 10:30 \
        --station IPA1ST          # optional: only one station
"""

import argparse
import os
import re
import shutil


def parse_time(filename):
    """Extract (hour, minute) from YYYYMMDDTHHMMSS+... filename."""
    m = re.match(r'\d{8}T(\d{2})(\d{2})\d{2}', os.path.basename(filename))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def in_time_window(hour, minute, start_min, end_min):
    return start_min <= hour * 60 + minute <= end_min


def main():
    parser = argparse.ArgumentParser(
        description="Copy only WAV files within a time window")
    parser.add_argument('--input', required=True,
                        help='Root dir with IPA station folders')
    parser.add_argument('--output', required=True,
                        help='Destination dir for filtered files')
    parser.add_argument('--start', default='05:30',
                        help='Start time HH:MM (default 05:30)')
    parser.add_argument('--end', default='10:30',
                        help='End time HH:MM (default 10:30)')
    parser.add_argument('--station', default=None,
                        help='Process only this station (e.g. IPA1ST)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be copied without copying')
    args = parser.parse_args()

    sh, sm = map(int, args.start.split(':'))
    eh, em = map(int, args.end.split(':'))
    start_min = sh * 60 + sm
    end_min = eh * 60 + em

    # Collect stations
    if args.station:
        stations = [args.station]
    else:
        stations = sorted([
            d for d in os.listdir(args.input)
            if os.path.isdir(os.path.join(args.input, d)) and d.startswith('IPA')
        ])

    total_copied = 0
    total_skipped = 0
    total_bytes = 0

    for station in stations:
        station_src = os.path.join(args.input, station)
        if not os.path.isdir(station_src):
            print(f"  Skipping {station} (not found)")
            continue

        for dirpath, _, filenames in os.walk(station_src):
            for fname in sorted(filenames):
                if not fname.lower().endswith('.wav'):
                    continue

                t = parse_time(fname)
                if t is None:
                    total_skipped += 1
                    continue

                if not in_time_window(t[0], t[1], start_min, end_min):
                    total_skipped += 1
                    continue

                # Preserve relative structure: station/date/file
                rel = os.path.relpath(os.path.join(dirpath, fname), args.input)
                dst = os.path.join(args.output, rel)

                if args.dry_run:
                    print(f"  [dry-run] {rel}")
                else:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(os.path.join(dirpath, fname), dst)

                total_copied += 1
                total_bytes += os.path.getsize(os.path.join(dirpath, fname))

    gb = total_bytes / (1024 ** 3)
    print(f"\nDone!")
    print(f"  Time window: {args.start} – {args.end}")
    print(f"  Copied:  {total_copied} files ({gb:.1f} GB)")
    print(f"  Skipped: {total_skipped} files (outside window)")
    if args.dry_run:
        print("  (dry-run mode — nothing was actually copied)")


if __name__ == '__main__':
    main()
