"""
Scan raw recordings for the roaring pulse train, without the classifier.

The published description of a guereza roar is a ~1.4 Hz pulse train sustained
for ~10 s: a 'roaring phrase' of roughly 15 pulses of ~0.7 s each, repeated into
a sequence. That is a temporal signature, and it is the one thing thunder cannot
imitate -- thunder is a single broadband transient with an exponential decay and
no periodic structure. Low-frequency energy, which is all the deployed gate
measures, cannot separate them because both are low-frequency.

WHY THIS DOES NOT USE THE MODEL. Two independent reasons, and either alone would
justify it:

  * The classifier's analysis window is 2 s. At 1.4 Hz that holds 2.8 cycles,
    which cannot establish periodicity. The window is shorter than the feature.
  * Every one of the 789 Colobus training clips is archival library media, so a
    model that fails to transfer to this recording channel and a species that is
    absent produce identical output. A signal-processing test has no such
    dependency: it asks a question about the audio, not about the model.

It is also ~30x cheaper, which is what makes scanning 634 h of night audio
feasible at all where a model pass would take about a day.

CALIBRATION. `scripts/test_roar_pulse_rate.py` scored 172 reference roars
against the 253 field detections that were listened to and confirmed NOT to be
roars, both at 5 s:

    pulse_share_1_2hz   AUC 0.831     (the wide 1-8 Hz band used previously: 0.771)
    low_freq_ratio      AUC 0.883
    the two summed      AUC 0.944     (correlation +0.26, so they are near-independent)

The default threshold below keeps roughly 70 % of reference roars while
admitting 18 % of confirmed non-roars.

WHAT A HIT DOES AND DOES NOT MEAN. This finds audio with the temporal structure
of a roar. It does not identify a species: the calibration positives are library
recordings and the negatives are field audio from one site, so the separation is
confounded with recording channel and no field-verified positive exists anywhere
in this project to break that confound. Every hit needs a human ear. That is the
point -- the scan exists to make the listening pile small enough to be possible.

Usage:
    PRIMATE_IPA_ROOT=... python scripts/scan_roar_pulse.py --window 19:00-05:00
    PRIMATE_IPA_ROOT=... python scripts/scan_roar_pulse.py --window all --stations IPA11ST
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

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SR = 8000            # the roar is below 1.5 kHz
SEG_S = 10.0         # one roaring phrase
HOP_S = 5.0
NFFT, HOP = 1024, 64


def scan_recording(path, seg_s=SEG_S, hop_s=HOP_S):
    """Rolling pulse-train score over one recording. Returns a DataFrame."""
    import librosa
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    if y.size < int(SR * seg_s):
        return None

    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=NFFT)
    band = (f >= config.FMIN) & (f < min(config.FMAX, sr / 2 - 1))
    low = (f >= 100) & (f < 1500)
    env_full = S[low].sum(axis=0)
    band_full = S[band].sum(axis=0)
    fs = sr / HOP                      # envelope sample rate

    n = int(seg_s * fs)
    step = int(hop_s * fs)
    if env_full.size < n:
        return None
    win = np.hanning(n)
    rows = []
    for i in range(0, env_full.size - n + 1, step):
        e = env_full[i:i + n]
        if e.max() <= 0:
            continue
        lf = float(e.sum() / (band_full[i:i + n].sum() + 1e-12))
        en = e / e.max()
        sp = np.abs(np.fft.rfft((en - en.mean()) * win))
        fr = np.fft.rfftfreq(n, d=1.0 / fs)
        tot = sp[(fr > 0.3) & (fr < 20)].sum() + 1e-12
        m = (fr >= 1.0) & (fr < 2.0)
        share = float(sp[m].max() / tot) if m.any() else 0.0
        rows.append({"start_s": round(i / fs, 1),
                     "pulse_share": round(share, 4),
                     "pulse_hz": round(float(fr[m][np.argmax(sp[m])]), 2) if m.any() else 0.0,
                     "low_freq_ratio": round(lf, 4)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="all")
    ap.add_argument("--window", default="19:00-05:00",
                    help="HH:MM-HH:MM, or 'all'. Wraps past midnight.")
    ap.add_argument("--min-pulse", type=float, default=0.0501,
                    help="Keeps ~70%% of reference roars, ~18%% of confirmed "
                         "non-roars (test_roar_pulse_rate.py).")
    ap.add_argument("--min-lf", type=float, default=config.LOWFREQ_GATE_THRESHOLD)
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/roar_pulse_scan"))
    args = ap.parse_args()

    if not os.path.isdir(config.IPA_ROOT):
        sys.exit(f"IPA_ROOT does not exist: {config.IPA_ROOT}\n"
                 f"Set PRIMATE_IPA_ROOT to the drive holding the IPA* folders.")
    os.makedirs(args.out, exist_ok=True)

    stations = ([d for d in sorted(os.listdir(config.IPA_ROOT)) if d.startswith("IPA")]
                if args.stations == "all"
                else [s.strip() for s in args.stations.split(",")])

    def files_for(station):
        if args.window == "all":
            return data_loader.get_ipa_station_files(station, time_filter=False)
        a, b = args.window.split("-")
        ah = int(a.split(":")[0]) + int(a.split(":")[1]) / 60
        bh = int(b.split(":")[0]) + int(b.split(":")[1]) / 60
        if ah <= bh:
            return data_loader.get_ipa_station_files(station, True, window=(a, b))
        # wraps midnight: two ranges, deduplicated
        f1 = data_loader.get_ipa_station_files(station, True, window=(a, "23:59"))
        f2 = data_loader.get_ipa_station_files(station, True, window=("00:00", b))
        return sorted(set(f1) | set(f2))

    t0 = time.time()
    n_done = 0
    for station in stations:
        out_csv = os.path.join(args.out, f"{station}_hits.csv")
        if os.path.exists(out_csv):
            print(f"  {station}: already done, skipping")
            continue
        files = files_for(station)
        if not files:
            print(f"  {station}: no files in {args.window}")
            continue
        hits, summ = [], []
        for i, p in enumerate(files, 1):
            df = scan_recording(p)
            n_done += 1
            if df is None or not len(df):
                continue
            # Record the DISTRIBUTION, not only what clears a threshold. The
            # dawn probe taught this the expensive way: a threshold calibrated
            # on archival recordings did not transfer to field audio, and a
            # thresholded scan would have returned zero hits that read as
            # "no roars here" when they only meant "no roars above a number
            # chosen somewhere else". With the per-recording maximum kept, the
            # operating point can be chosen tomorrow against what is actually
            # in this corpus.
            best = df.loc[df.pulse_share.idxmax()]
            lowf = df[df.low_freq_ratio >= args.min_lf]
            best_lf = lowf.loc[lowf.pulse_share.idxmax()] if len(lowf) else None
            summ.append({
                "file": os.path.basename(p), "station": station,
                "windows": len(df),
                "pulse_max": best.pulse_share,
                "pulse_max_hz": best.pulse_hz,
                "pulse_max_lf": best.low_freq_ratio,
                "pulse_max_t": best.start_s,
                # the best window that is ALSO low-frequency, i.e. the best
                # candidate that could physically be a roar at all
                "lowf_pulse_max": best_lf.pulse_share if best_lf is not None else 0.0,
                "lowf_pulse_t": best_lf.start_s if best_lf is not None else -1,
                "n_lowf_windows": int(len(lowf)),
            })
            keep = df[(df.pulse_share >= args.min_pulse)
                      & (df.low_freq_ratio >= args.min_lf)]
            if len(keep):
                keep = keep.assign(file=os.path.basename(p), station=station)
                hits.append(keep)
            if i % 25 == 0 or i == len(files):
                rate = (time.time() - t0) / max(n_done, 1)
                nh = sum(len(h) for h in hits)
                print(f"  {station} [{i}/{len(files)}] hits={nh} "
                      f"({rate:.1f} s/recording)", flush=True)
        out = pd.concat(hits, ignore_index=True) if hits else pd.DataFrame(
            columns=["start_s", "pulse_share", "pulse_hz", "low_freq_ratio",
                     "file", "station"])
        out.to_csv(out_csv, index=False)
        pd.DataFrame(summ).to_csv(
            os.path.join(args.out, f"{station}_summary.csv"), index=False)

    print(f"\nscanned {n_done} recordings in {(time.time()-t0)/60:.1f} min")
    print(f"wrote {args.out}")
    print("\nEvery hit needs a human ear. The scan finds roar-shaped audio; it "
          "does not\nname a species, and its calibration positives are library "
          "recordings.")


if __name__ == "__main__":
    main()
