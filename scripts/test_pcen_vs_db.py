"""
Would PCEN separate these calls from these false positives better than dB does?

The pipeline renders every window with ``librosa.power_to_db(mel, ref=np.max)``
followed by a per-clip min-max rescale. Per-channel energy normalization is the
standard alternative in bioacoustics, and the reason to try it here is specific
rather than general: PCEN applies adaptive gain control that suppresses
*stationary* noise while preserving transients, and the dominant false positive
in this deployment is a nocturnal insect chorus -- 70 % of confirmed false
positives, a 97.2 %-pure cluster, continuous stridulation at 2-4 kHz running
about 9 dB louder than a real call. Stationary noise is exactly what PCEN is
built to remove. Lostanlen et al. report false-alarm reductions of 50x near
field and 5x far field on avian and marine data.

Re-rendering the 4.8 GB image pack and re-running the 16-fold sweep to find out
costs about a day. This asks the question for a few minutes instead, by fitting
the same simple classifier on the same clips under both representations,
leave-one-station-out so the comparison cannot be won by fitting the test
stations.

WHAT THIS CAN AND CANNOT SETTLE. A win here means the information PCEN preserves
is linearly available where dB's is not; it does not prove VGG19 would exploit
it, because a deep network can undo some normalisations that a linear probe
cannot. A loss is the stronger signal: if PCEN does not help a probe that sees
the representation directly, it is unlikely to help through a frozen ImageNet
trunk. Treat this as a gate on whether to spend the day, not as the result.

Usage:
    python scripts/test_pcen_vs_db.py
    python scripts/test_pcen_vs_db.py --per-class 500
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


def mel_power(y, sr):
    import librosa
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS, fmin=config.FMIN,
        fmax=min(config.FMAX, sr // 2))


def repr_db(S, sr):
    """What the pipeline does today: dB relative to the clip max, then min-max."""
    import librosa
    D = librosa.power_to_db(S, ref=np.max)
    lo, hi = D.min(), D.max()
    return (D - lo) / (hi - lo + 1e-12)


def repr_pcen(S, sr, time_constant=0.400):
    """PCEN. librosa's defaults are tuned for speech; the time constant is the
    parameter that matters for a 1-2 s animal call and is swept below."""
    import librosa
    P = librosa.pcen(S * (2 ** 31), sr=sr, hop_length=config.HOP_LENGTH,
                     time_constant=time_constant)
    return P / (P.max() + 1e-12)


def repr_db_absolute(S, sr):
    """
    dB against a FIXED reference, so absolute level survives.

    The shipped pipeline destroys it twice -- ``power_to_db(ref=np.max)``
    subtracts the clip's own maximum, then a per-clip min-max rescale removes
    whatever scale is left. That matters here specifically: measured over 1 400
    reviewed clips, the false positives are **9.26 dB LOUDER** than the genuine
    calls, and absolute RMS separates them at ROC area 0.830 -- better than the
    detector's own confidence (0.726). The pipeline cannot see any of it.
    """
    import librosa
    return librosa.power_to_db(S, ref=1e-10, top_db=None) / 100.0


def summarise(M):
    """Per-band mean and std, plus a transient/stationary contrast."""
    mu = M.mean(axis=1)
    sd = M.std(axis=1)
    # How much each band fluctuates relative to its own level: a stationary
    # chorus scores low, a call scores high. This is the quantity PCEN is
    # supposed to sharpen.
    contrast = sd / (mu + 1e-6)
    return np.concatenate([mu, sd, contrast])


def loso_auc(X, y, groups):
    """Mean held-out AUC of a logistic probe, one fold per station."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    aucs = []
    for st in sorted(set(groups)):
        tr, te = groups != st, groups == st
        if len(set(y[te])) < 2 or te.sum() < 20:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.1))
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), len(aucs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=700)
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/pcen_vs_db.csv"))
    args = ap.parse_args()

    d = pd.read_csv(REVIEW)
    d["is_call"] = d["verdict"].eq("call")
    index = {}
    for dirpath, _, files in os.walk(CLIPS):
        for f in files:
            if f.lower().endswith(".wav"):
                index[f] = os.path.join(dirpath, f)
    d = d[d["file"].isin(index)]
    sel = pd.concat([
        d[d.is_call].sample(min(args.per_class, int(d.is_call.sum())), random_state=0),
        d[~d.is_call].sample(min(args.per_class, int((~d.is_call).sum())), random_state=0),
    ])
    print(f"{len(sel)} clips ({int(sel.is_call.sum())} call / "
          f"{int((~sel.is_call).sum())} false positive) "
          f"across {sel.site.nunique()} stations")

    import librosa
    rows_db, rows_pc, rows_abs, keep = [], [], [], []
    for i, r in enumerate(sel.itertuples()):
        if i % 200 == 0:
            print(f"  {i}/{len(sel)}", flush=True)
        try:
            y, sr = librosa.load(index[r.file], sr=config.SAMPLE_RATE, mono=True)
        except Exception:
            continue
        if y.size < config.N_FFT:
            continue
        S = mel_power(y, sr)
        rows_db.append(summarise(repr_db(S, sr)))
        rows_pc.append(summarise(repr_pcen(S, sr)))
        rows_abs.append(summarise(repr_db_absolute(S, sr)))
        keep.append((r.is_call, r.site))

    if not keep:
        sys.exit("no clips loaded")
    y = np.array([k[0] for k in keep])
    g = np.array([k[1] for k in keep])
    Xd, Xp, Xa = np.vstack(rows_db), np.vstack(rows_pc), np.vstack(rows_abs)

    print(f"\nfitted on {len(y)} clips, {Xd.shape[1]} features each\n")
    a_db, n1 = loso_auc(Xd, y, g)
    a_pc, n2 = loso_auc(Xp, y, g)
    a_both, n3 = loso_auc(np.hstack([Xd, Xp]), y, g)
    a_abs, n4 = loso_auc(Xa, y, g)
    a_mix, n5 = loso_auc(np.hstack([Xd, Xa]), y, g)
    print(f"{'representation':>34} {'held-out AUC':>14} {'folds':>7}")
    print(f"{'dB + per-clip min-max (current)':>34} {a_db:>14.4f} {n1:>7d}")
    print(f"{'PCEN':>34} {a_pc:>14.4f} {n2:>7d}")
    print(f"{'dB + PCEN':>34} {a_both:>14.4f} {n3:>7d}")
    print(f"{'dB, ABSOLUTE level kept':>34} {a_abs:>14.4f} {n4:>7d}")
    print(f"{'current + absolute':>34} {a_mix:>14.4f} {n5:>7d}")
    print(f"\nPCEN     - current : {a_pc - a_db:+.4f}")
    print(f"absolute - current : {a_abs - a_db:+.4f}")
    print(f"both     - current : {a_mix - a_db:+.4f}")

    pd.DataFrame([{"representation": "db", "loso_auc": a_db},
                  {"representation": "pcen", "loso_auc": a_pc},
                  {"representation": "both", "loso_auc": a_both},
                  {"representation": "db_absolute", "loso_auc": a_abs},
                  {"representation": "current+absolute", "loso_auc": a_mix}]).to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    if a_pc - a_db > 0.02:
        print("\nWorth the day: PCEN carries information dB does not. Re-render "
              "the pack\nwith it and re-run the sweep.")
    elif a_both - max(a_db, a_pc) > 0.02:
        print("\nNeither alone, but they are complementary -- consider feeding "
              "both as channels\nrather than replacing one with the other.")
    else:
        print("\nNot worth the day on this evidence. The representations carry "
              "the same\nlinearly-available information about these clips.")


if __name__ == "__main__":
    main()
