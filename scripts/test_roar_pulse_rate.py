"""
Can a pulse-train test separate a guereza roar from thunder?

The published description of the call is unusually specific. A roaring sequence
is built from a 'roaring phrase', and a phrase is roughly 15 pulses of about
0.7 s each -- a **~1.4 Hz pulse train sustained for ~10 s**. Thunder cannot
imitate that: it is a single broadband transient with an exponential decay, and
it has no periodic structure at all. Low-frequency energy, the only criterion the
gate uses, cannot tell the two apart because both are low-frequency. Pulse rate
should.

WHY THIS WAS MISSED BEFORE. HANDOFF records that "1-8 Hz envelope modulation"
was tried and reached AUC 0.715 -- promising but not enough to act on. Two things
were wrong with that test, and both suppress exactly this signal:

  1. The band. 1-8 Hz is wide; a 1.4 Hz peak is at the very bottom of it and gets
     averaged away by everything above 3 Hz.
  2. The window. It was measured on 2 s clips. At 1.4 Hz a 2 s window holds 2.8
     cycles, which is not enough to establish periodicity -- the modulation
     spectrum has no resolution there.

A THIRD PROBLEM, FOUND WHILE SETTING THIS UP. The reference material is itself
too short: the 617 windows come from 172 source clips that reconstruct to a
median of 5 s and a maximum of 6 s. **No reference clip contains a complete
roaring phrase.** The model has never been shown one. That is a data problem
this script cannot fix, and it bounds what the test below can prove: at 5 s a
1.4 Hz train gives ~7 pulses, enough to detect periodicity but not enough to
measure a phrase.

DESIGN. Equal durations on both sides or the comparison is about clip length:
  positives  172 reference roars, the uniform 5.0 s clips
  negatives  253 field detections, every one confirmed by ear NOT to be a roar,
             re-cut to 5.0 s from the source recordings on the drive
Both are scored on the amplitude envelope of the low band only, since that is
where the roar lives and where thunder competes.

Usage:
    PRIMATE_IPA_ROOT=... python scripts/test_roar_pulse_rate.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import config  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SR = 8000          # the roar is below 1.5 kHz; 8 kHz is ample and 5x faster
DUR = 5.0


def pulse_features(y, sr):
    """Periodicity of the low-band amplitude envelope."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=64)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low = (f >= 100) & (f < 1500)
    env = S[low].sum(axis=0)
    if env.size < 32 or env.max() <= 0:
        return None
    env = env / env.max()
    e = env - env.mean()
    # Envelope sampling rate = sr / hop
    fs = sr / 64.0
    win = np.hanning(e.size)
    sp = np.abs(np.fft.rfft(e * win))
    fr = np.fft.rfftfreq(e.size, d=1.0 / fs)

    def band_peak(lo, hi):
        m = (fr >= lo) & (fr < hi)
        if not m.any():
            return 0.0, 0.0
        i = np.argmax(sp[m])
        return float(fr[m][i]), float(sp[m][i])

    tot = sp[(fr > 0.3) & (fr < 20)].sum() + 1e-12
    f_roar, p_roar = band_peak(1.0, 2.0)     # the published ~1.4 Hz
    f_wide, p_wide = band_peak(0.5, 8.0)
    return {
        "pulse_hz": f_roar,
        "pulse_share_1_2hz": p_roar / tot,   # how much of the modulation is at 1-2 Hz
        "pulse_share_wide": p_wide / tot,
        "peak_hz_wide": f_wide,
        "env_cv": float(env.std() / (env.mean() + 1e-12)),
    }


def auc(score, positive):
    r = pd.Series(score).rank().to_numpy()
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", default=config.IPA_ROOT)
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/roar_pulse_test.csv"))
    args = ap.parse_args()

    import librosa
    rows = []

    pos = sorted(glob.glob(os.path.join(REPO, "data/species/Colobus guereza Clips 5s/*.wav")))
    print(f"reference roars: {len(pos)}")
    for p in pos:
        y, sr = librosa.load(p, sr=SR, mono=True, duration=DUR)
        d = pulse_features(y, sr)
        if d:
            rows.append({**d, "cls": "reference_roar", "src": os.path.basename(p)})

    # Field detections, re-cut to the same length from the source recordings.
    det = []
    for f in glob.glob(os.path.join(REPO, "data/outputs/detections/**/*_detections.csv"),
                       recursive=True):
        try:
            t = pd.read_csv(f)
        except Exception:
            continue
        if len(t) and "species" in t:
            t = t[t["species"] == "Colobus_guereza"]
            if len(t):
                t = t.assign(rec=os.path.basename(f).replace("_detections.csv", ".wav"))
                det.append(t)
    det = pd.concat(det, ignore_index=True) if det else pd.DataFrame()
    print(f"field detections (all confirmed NOT roars): {len(det)}")

    index = {}
    for dirpath, _, files in os.walk(args.audio_root):
        for fn in files:
            if fn.lower().endswith(".wav"):
                index[fn] = os.path.join(dirpath, fn)

    missing = 0
    for r in det.itertuples():
        src = index.get(r.rec)
        if src is None:
            missing += 1
            continue
        start = max(0.0, float(r.start_time) - (DUR - 2.0) / 2)
        try:
            y, sr = librosa.load(src, sr=SR, mono=True, offset=start, duration=DUR)
        except Exception:
            missing += 1
            continue
        if y.size < int(SR * DUR * 0.9):
            missing += 1
            continue
        d = pulse_features(y, sr)
        if d:
            rows.append({**d, "cls": "field_not_roar", "src": r.rec})
    if missing:
        print(f"  {missing} field detections could not be re-cut and are excluded "
              f"(reported, not silently dropped)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    n_pos = int((df.cls == "reference_roar").sum())
    n_neg = int((df.cls == "field_not_roar").sum())
    print(f"\nscored {n_pos} reference roars vs {n_neg} confirmed non-roars, "
          f"both at {DUR:.0f} s\n")
    if not n_pos or not n_neg:
        sys.exit("one side is empty -- cannot compare")

    y = (df.cls == "reference_roar").to_numpy()
    print(f"{'feature':>20} {'roar median':>12} {'nonroar median':>15} {'AUC':>7}")
    for c in ["pulse_share_1_2hz", "pulse_share_wide", "pulse_hz", "peak_hz_wide", "env_cv"]:
        a = auc(df[c], y)
        print(f"{c:>20} {df.loc[y, c].median():12.4f} {df.loc[~y, c].median():15.4f} "
              f"{a:7.4f}")

    best = "pulse_share_1_2hz"
    print(f"\noperating points on {best} (keep >= t):")
    for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
        t = df.loc[y, best].quantile(1 - q)
        kept_pos = (df.loc[y, best] >= t).mean()
        kept_neg = (df.loc[~y, best] >= t).mean()
        print(f"  t={t:.4f}  roars kept {kept_pos:.1%}  non-roars kept {kept_neg:.1%}")

    print("\nCAVEAT. The positives are archival library recordings and the "
          "negatives are\nfield audio from one site, so any separation here is "
          "confounded with recording\nchannel. Treat a strong result as a reason "
          "to test on field positives, of which\nthis project currently has none "
          "-- not as a validated detector.")


if __name__ == "__main__":
    main()
