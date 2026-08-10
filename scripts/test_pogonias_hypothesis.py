"""
Are the daytime false positives *Cercopithecus pogonias*?

The confirmed false positives split into two populations with very different
character. About 70 % are a nocturnal insect chorus: 2-4 kHz, roughly 9 dB
louder than a real call, 98.8 % at night, and a 97.2 %-pure cluster. Those are
removed for free by the 05:00-19:00 time gate. The remaining ~26 % are the hard
ones. They occur in daylight, they sit in the same 4-8 kHz band as genuine
\\textit{C. nictitans} calls, and they survive the time gate, the isolation
filter and the V13 retraining alike. Every cheap acoustic feature collapses on
them: restricted to daytime, the best of them reaches only 0.703.

The species expert on this project has now identified them by ear as
*Cercopithecus pogonias*, the crowned monkey. That is a congener of the target,
which would explain the failure exactly: this is a closed-set problem, not a
threshold problem. A classifier with no class for a congeneric species has
nowhere to put its calls except the class they most resemble, and no amount of
retuning fixes that.

This script tests the claim against the audio rather than accepting it. Two
reference recordings of *C. pogonias* are on the drive already
(`Primates training data/cercopithecus pogonias/`, xeno-canto XC1033595 and
XC962629), unused by any manifest. It asks whether the daytime false positives
resemble those recordings more than the genuine calls do.

READ THE RESULT CAREFULLY. Two reference recordings is a very thin basis, and
they are archival while the field clips are not, so channel differences are
confounded with species differences here exactly as they were for the Colobus
comparison. A positive result is a reason to obtain more *C. pogonias* material
and train a fifth class, not a confirmation on its own.

Usage:
    python scripts/test_pogonias_hypothesis.py
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
REVIEW = os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv")
CLIPS = os.path.join(REPO, "data/outputs/detected_clips/Cernic")
SR = 22050


def describe(y, sr):
    import librosa
    if y.size < 2048:
        return None
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = S.sum() + 1e-12

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(S[m].sum() / tot)

    mag = np.sqrt(S)
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    env = S.sum(axis=0)
    env = env / (env.max() + 1e-12)
    return dict(
        e_0_1k=band(0, 1000), e_1k_2k=band(1000, 2000), e_2k_4k=band(2000, 4000),
        e_4k_8k=band(4000, 8000), e_8k=band(8000, sr // 2),
        centroid=float(librosa.feature.spectral_centroid(S=mag, sr=sr).mean()),
        bandwidth=float(librosa.feature.spectral_bandwidth(S=mag, sr=sr).mean()),
        flatness=float(librosa.feature.spectral_flatness(S=mag).mean()),
        env_cv=float(env.std() / (env.mean() + 1e-12)),
        **{f"mfcc{i}": float(v) for i, v in enumerate(mf.mean(axis=1))},
    )


def load_windows(path, win_s=2.0, hop_s=1.0, limit=60):
    """A long reference recording contributes many windows, not one."""
    import librosa
    out = []
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return out
    n, step = int(win_s * sr), int(hop_s * sr)
    for i in range(0, max(1, y.size - n + 1), step):
        d = describe(y[i:i + n], sr)
        if d:
            out.append(d)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-group", type=int, default=250)
    ap.add_argument("--pogonias", default=os.path.join(
        os.path.dirname(config.IPA_ROOT.rstrip("/\\")),
        "Primates training data", "cercopithecus pogonias"))
    args = ap.parse_args()

    refs = sorted(glob.glob(os.path.join(args.pogonias, "*")))
    refs = [r for r in refs if r.lower().endswith((".wav", ".mp3", ".flac"))]
    if not refs:
        sys.exit(f"no C. pogonias reference audio under {args.pogonias}")
    print(f"reference recordings: {len(refs)}")
    for r in refs:
        print("   ", os.path.basename(r))

    pog = []
    for r in refs:
        pog += load_windows(r)
    if not pog:
        sys.exit("could not window the reference audio")
    print(f"  -> {len(pog)} windows\n")

    d = pd.read_csv(REVIEW)
    rec = pd.to_datetime(d["timestamp"], format="%Y%m%dT%H%M%S")
    det = rec + pd.to_timedelta(d["start_s"], unit="s")
    d["hour"] = det.dt.hour + det.dt.minute / 60.0
    d["is_call"] = d["verdict"].eq("call")
    index = {}
    for dirpath, _, files in os.walk(CLIPS):
        for f in files:
            if f.lower().endswith(".wav"):
                index[f] = os.path.join(dirpath, f)
    d = d[d["file"].isin(index)]

    day = (d.hour >= 5) & (d.hour < 19)
    groups = {
        "genuine call (daytime)": d[d.is_call & day],
        "FALSE POSITIVE (daytime)": d[~d.is_call & day],
        "false positive (night)": d[~d.is_call & ~day],
    }

    import librosa
    feats = {"C. pogonias reference": pd.DataFrame(pog)}
    for name, sub in groups.items():
        sub = sub.sample(min(args.per_group, len(sub)), random_state=0)
        rows = []
        for r in sub.itertuples():
            try:
                y, sr = librosa.load(index[r.file], sr=SR, mono=True)
            except Exception:
                continue
            v = describe(y, sr)
            if v:
                rows.append(v)
        feats[name] = pd.DataFrame(rows)
        print(f"  {name}: {len(rows)} clips")

    cols = [c for c in feats["C. pogonias reference"].columns]
    allf = pd.concat(feats.values())
    mu, sd = allf[cols].mean(), allf[cols].std().replace(0, 1)
    Z = {k: ((v[cols] - mu) / sd).to_numpy() for k, v in feats.items()}

    ref = Z["C. pogonias reference"]
    print("\ndistance to the C. pogonias reference "
          "(median over clips of the nearest reference window):")
    for k in ["genuine call (daytime)", "FALSE POSITIVE (daytime)",
              "false positive (night)"]:
        if not len(Z[k]):
            continue
        dists = [np.linalg.norm(ref - z, axis=1).min() for z in Z[k]]
        print(f"  {k:28s} {np.median(dists):7.3f}")

    print("\nIf the daytime false positives are C. pogonias, their distance "
          "should be\nclearly the smallest of the three. If genuine calls are "
          "just as close, the\ncomparison is measuring recording channel and "
          "cannot support the claim.")


if __name__ == "__main__":
    main()
