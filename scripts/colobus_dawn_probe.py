"""
Ask whether the detector ever comes close to firing on Colobus at dawn.

`run_detection_ipa.py` records only windows above a threshold, so a dawn sweep
that finds nothing returns a zero that cannot be interpreted. Two very different
worlds produce the same zero:

  (a) no *C. guereza* vocalises at these stations;
  (b) *C. guereza* is there and the model cannot recognise it in field
      conditions -- which is entirely plausible, because every one of the 789
      positives it was trained on is Macaulay Library media (owner-confirmed),
      recorded elsewhere, at a different distance, through a different channel.

The pilot makes this concrete: 18 dawn recordings at IPA11ST, threshold lowered
to 0.30, produced **10 detections, all Cernic, zero Colobus**. Useless as
evidence either way.

So this records the *distribution* of the Colobus score over every window
instead of thresholding it. The compute is identical -- the model already runs
on every window -- but the outcome is interpretable whichever way it falls:

  max score stays near zero    the model never comes close. Library-trained
                               weights do not transfer to this channel, and
                               "we did not detect it" is a statement about the
                               detector, not about the forest. That is a
                               reportable negative and it is honest.

  a tail at 0.25-0.45          real candidates, ranked, in listenable numbers.

Cernic is scored alongside as a positive control: it is known present at 56.43
ind/km2 and the model is known to work on it, so if the Cernic distribution
looks normal and the Colobus one is flat at zero, the difference is about
Colobus and not about the audio or the pipeline.

Output is one row per window above --keep-above (default 0.10, far below any
usable threshold) plus a per-file summary, written incrementally so an
interrupted run keeps what it has.

Usage:
    python scripts/colobus_dawn_probe.py --stations IPA11ST
    python scripts/colobus_dawn_probe.py --stations all --window 05:00-08:00
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config          # noqa: E402
import data_loader     # noqa: E402
import preprocessing   # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def probe_file(model, path, cls_idx, keep_above, batch=64):
    """Per-window class scores for one recording. Returns (rows, summary)."""
    audio, sr = data_loader.load_long_audio(path)
    if audio is None:
        return [], None
    windows, times = preprocessing.extract_sliding_windows(audio, sr)
    if not len(windows):
        return [], None

    X = np.stack([preprocessing.preprocess_for_model(
        preprocessing.preprocess_audio(w, sr)) for w in windows])
    probs = model.predict(X, batch_size=batch, verbose=0)

    name = os.path.basename(path)
    rows = []
    for cls, i in cls_idx.items():
        s = probs[:, i]
        hits = np.where(s >= keep_above)[0]
        for h in hits:
            rows.append({"file": name, "species": cls,
                         "start_s": float(times[h][0]),
                         "score": float(s[h])})
    summary = {"file": name, "windows": len(windows)}
    for cls, i in cls_idx.items():
        s = probs[:, i]
        summary[f"{cls}_max"] = round(float(s.max()), 4)
        summary[f"{cls}_p99"] = round(float(np.percentile(s, 99)), 4)
        summary[f"{cls}_mean"] = round(float(s.mean()), 4)
        summary[f"{cls}_n_over_0.3"] = int((s >= 0.3).sum())
    return rows, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="all",
                    help="'all' or a comma-separated list.")
    ap.add_argument("--window", default="05:00-08:00",
                    help="HH:MM-HH:MM. The guereza dawn chorus.")
    ap.add_argument("--model",
                    default=os.path.join(REPO, "data/outputs/models/best_model_v12.h5"))
    ap.add_argument("--keep-above", type=float, default=0.10,
                    help="Record every window scoring at least this. Deliberately "
                         "far below any usable threshold -- the point is the tail.")
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/colobus_dawn_probe"))
    args = ap.parse_args()

    if not os.path.isdir(config.IPA_ROOT):
        sys.exit(f"IPA_ROOT does not exist: {config.IPA_ROOT}\n"
                 f"Set PRIMATE_IPA_ROOT to the drive holding the IPA* folders.")

    import model as model_module
    model = model_module.load_trained_model(args.model)
    cls_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)
               if c in ("Cernic", "Colobus_guereza")}
    if "Colobus_guereza" not in cls_idx:
        sys.exit(f"no Colobus_guereza in CLASS_NAMES: {config.CLASS_NAMES}")
    print(f"scoring classes {cls_idx}")

    stations = ([d for d in sorted(os.listdir(config.IPA_ROOT))
                 if d.startswith("IPA")]
                if args.stations == "all"
                else [s.strip() for s in args.stations.split(",")])
    win = tuple(args.window.split("-"))
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    n_done = 0
    for station in stations:
        files = data_loader.get_ipa_station_files(station, time_filter=True,
                                                  window=win)
        if not files:
            print(f"  {station}: no files in {args.window} -- skipped")
            continue
        rows_path = os.path.join(args.out, f"{station}_windows.csv")
        sums_path = os.path.join(args.out, f"{station}_summary.csv")
        if os.path.exists(sums_path):
            print(f"  {station}: already done, skipping "
                  f"(delete {sums_path} to redo)")
            continue
        rows, sums = [], []
        for i, f in enumerate(files, 1):
            r, s = probe_file(model, f, cls_idx, args.keep_above)
            rows += r
            if s:
                sums.append(s)
            n_done += 1
            rate = (time.time() - t0) / max(n_done, 1)
            print(f"  {station} [{i}/{len(files)}] {os.path.basename(f)[:44]} "
                  f"col_max={s['Colobus_guereza_max'] if s else 'NA'} "
                  f"({rate/60:.1f} min/file)", flush=True)
            # Incremental: an 8-hour job must not lose everything to one crash.
            pd.DataFrame(rows).to_csv(rows_path, index=False)
            pd.DataFrame(sums).to_csv(sums_path, index=False)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min over {n_done} recordings")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
