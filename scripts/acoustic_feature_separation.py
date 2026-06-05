"""
Measure whether interpretable acoustic features separate real Cernic calls
from field false positives -- WITHOUT touching the held-out test stations.

Motivation
----------
Visual inspection of detections shows the model's confidence is essentially
uninformative (real putty-nose calls score 0.61-0.90 while pure-noise / insect
clips score up to 0.99). The human ear/eye separates them on cues the VGG19 +
global-pooling pipeline averages away:

  * real Cernic energy lives LOW (~300-1500 Hz) as frequency-modulated
    down-sweeping arcs;
  * the common FPs are insect/cicada pulse trains at ~4 kHz (rhythmic but at a
    fixed frequency) or smeared mid-high noise bands -- weak, unstructured low
    band.

This script computes a few cheap, interpretable features on the saved 2s
detection clips and reports how well each one separates real from FP (per-
feature AUC + distribution plot). If they separate, the same features can be
used as a post-hoc filter on the ALREADY-SAVED detection clips -- no re-running
detection over the long audio, no retraining.

Label safety / test-set hygiene
--------------------------------
The operating point must be designed on DEV stations only. The held-out test
stations IPA19/20 are HARD-REFUSED as design inputs (any name containing 19/20).
You may later --apply the chosen features to IPA19/20 for REPORTING, but never
to choose a threshold.

Positives  : clean Cernic at --real-cernic-stations (e.g. IPA15/17/18).
             NOTE the clean bucket is known to be a mix -- pass --real-clips-dir
             to use a hand-curated folder of confirmed-real clips instead.
Negatives  : all Cernic at --fp-cernic-stations (dev stations with no real
             Cernic, e.g. IPA13/14/16).

Example
-------
    python scripts/acoustic_feature_separation.py \
        --cleanup-root "/content/drive/MyDrive/primates-sound-detection/outputs/auto_cleanup" \
        --real-cernic-stations IPA15ST IPA17ST IPA18ST \
        --fp-cernic-stations   IPA13ST IPA14ST IPA16ST \
        --save-dir "/content/drive/MyDrive/primates-sound-detection/outputs/visualizations"
"""

import argparse
import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

SR = config.SAMPLE_RATE
N_FFT = config.N_FFT
HOP = config.HOP_LENGTH
N_MELS = config.N_MELS
FMIN = config.FMIN
FMAX = config.FMAX
CLIP_DURATION = config.CLIP_DURATION
EPS = 1e-9

TEST_TOKENS = ("19", "20")
CERNIC = "Cernic"
CLIP_SUBDIRS = ("clean_clips", "suspicious_clips")

# Frequency bands (Hz) that matter for the real-vs-FP distinction.
LOW_BAND = (300, 1500)     # real Cernic structure lives here
HIGH_BAND = (3000, 5500)   # insect pulse trains / smeared noise bands


def _is_test_station(station: str) -> bool:
    return any(tok in station for tok in TEST_TOKENS)


def _load_clip(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    target = int(round(CLIP_DURATION * SR))
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    elif len(audio) > target:
        audio = audio[:target]
    return audio.astype(np.float32)


def clip_features(audio) -> dict:
    """Compute interpretable acoustic features from a 2s waveform."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX)                       # (n_mels, n_frames), power
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN, fmax=FMAX)

    low_m = (freqs >= LOW_BAND[0]) & (freqs <= LOW_BAND[1])
    high_m = (freqs >= HIGH_BAND[0]) & (freqs <= HIGH_BAND[1])

    e_low = float(S[low_m].sum())
    e_high = float(S[high_m].sum())
    e_tot = float(S.sum()) + EPS

    # 1) Log low/high energy ratio: real calls put energy low, FPs high.
    low_high_logratio = np.log((e_low + EPS) / (e_high + EPS))

    # 2) Fraction of total energy in the low band.
    low_fraction = e_low / e_tot

    # 3) Low-band frequency modulation: within the low band, track the peak
    #    frequency per frame; FM down-sweeps make it wander, flat tones don't.
    low_S = S[low_m]
    low_freqs = freqs[low_m]
    frame_energy = low_S.sum(axis=0)
    # Only trust frames that actually carry low-band energy (above the clip's
    # own median) so silent frames don't inject spurious peak picks.
    active = frame_energy > np.median(frame_energy)
    if active.sum() >= 3:
        peak_freq = low_freqs[low_S[:, active].argmax(axis=0)]
        fm_spread = float(peak_freq.std())
    else:
        fm_spread = 0.0

    # 4) Low-band temporal modulation (coefficient of variation): discrete
    #    bursts -> high CV; continuous drone -> low CV.
    low_env = low_S.sum(axis=0)
    low_mod = float(low_env.std() / (low_env.mean() + EPS))

    # 5) Spectral centroid (overall brightness): FPs sit higher.
    centroid = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())

    return {
        "low_high_logratio": low_high_logratio,
        "low_fraction": low_fraction,
        "fm_spread_hz": fm_spread,
        "low_mod_cv": low_mod,
        "centroid_hz": centroid,
    }


def collect(cleanup_root, stations, bucket="all"):
    subdirs = {"clean": ["clean_clips"], "suspicious": ["suspicious_clips"],
               "all": list(CLIP_SUBDIRS)}[bucket]
    feats = []
    for st in stations:
        for sub in subdirs:
            d = os.path.join(cleanup_root, st, sub, CERNIC)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.lower().endswith(".wav"):
                    continue
                try:
                    feats.append(clip_features(_load_clip(os.path.join(d, f))))
                except Exception as exc:  # noqa: BLE001
                    print(f"   skip {f}: {exc}")
    return feats


def collect_dir(clips_dir):
    feats = []
    for f in sorted(os.listdir(clips_dir)):
        if f.lower().endswith(".wav"):
            try:
                feats.append(clip_features(_load_clip(os.path.join(clips_dir, f))))
            except Exception as exc:  # noqa: BLE001
                print(f"   skip {f}: {exc}")
    return feats


def auc(pos, neg):
    """Mann-Whitney AUC: P(random positive scores above random negative)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = ranks[:len(pos)].sum() + len(pos)  # 1-based ranks
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(pos, neg, higher_is_real=True):
    """Pick the single threshold maximizing balanced accuracy on the dev set."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    cand = np.unique(np.concatenate([pos, neg]))
    best = (None, -1.0, 0.0, 0.0)
    for t in cand:
        if higher_is_real:
            tpr = (pos >= t).mean()
            tnr = (neg < t).mean()
        else:
            tpr = (pos <= t).mean()
            tnr = (neg > t).mean()
        bal = 0.5 * (tpr + tnr)
        if bal > best[1]:
            best = (float(t), bal, float(tpr), float(tnr))
    return best  # (threshold, balanced_acc, recall, fp_cut)


def report(real_feats, fp_feats, save_dir=None):
    keys = ["low_high_logratio", "low_fraction", "fm_spread_hz",
            "low_mod_cv", "centroid_hz"]
    # For these features, higher == more real, except centroid (higher == FP).
    higher_is_real = {"low_high_logratio": True, "low_fraction": True,
                      "fm_spread_hz": True, "low_mod_cv": True,
                      "centroid_hz": False}

    print(f"\n  positives (real): {len(real_feats)}   "
          f"negatives (FP): {len(fp_feats)}")
    print(f"\n  {'feature':>18} {'AUC':>7} {'real_med':>10} {'fp_med':>10} "
          f"{'thr':>9} {'recall':>8} {'fp_cut':>8}")
    rows = {}
    for k in keys:
        pos = [f[k] for f in real_feats]
        neg = [f[k] for f in fp_feats]
        hi = higher_is_real[k]
        # AUC convention: orient so "real scores higher".
        a = auc(pos, neg) if hi else auc(neg, pos)
        thr, bal, rec, cut = best_threshold(pos, neg, higher_is_real=hi)
        rows[k] = dict(pos=pos, neg=neg, auc=a, thr=thr, recall=rec, cut=cut, hi=hi)
        print(f"  {k:>18} {a:7.3f} {np.median(pos):10.3f} {np.median(neg):10.3f} "
              f"{thr:9.3f} {rec:8.2%} {cut:8.2%}")

    print("\n  AUC reading: 0.5 = useless (real & FP indistinguishable on this "
          "feature), 1.0 = perfect separation.")
    print("  recall = fraction of real calls kept at that single-feature "
          "threshold; fp_cut = fraction of FPs removed.")

    # Distribution plot.
    try:
        import matplotlib.pyplot as plt
        n = len(keys)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
        for ax, k in zip(axes, keys):
            r = rows[k]
            lo = min(min(r["pos"]), min(r["neg"]))
            hi_ = max(max(r["pos"]), max(r["neg"]))
            bins = np.linspace(lo, hi_, 25)
            ax.hist(r["pos"], bins=bins, alpha=0.6, label="real", color="b",
                    density=True)
            ax.hist(r["neg"], bins=bins, alpha=0.6, label="FP", color="r",
                    density=True)
            ax.axvline(r["thr"], color="k", ls="--", lw=1)
            ax.set_title(f"{k}\nAUC={r['auc']:.2f}", fontsize=10)
            ax.legend(fontsize=8)
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "acoustic_feature_separation.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"\n  Saved distribution plot: {path}")
        plt.show()
    except Exception as exc:  # noqa: BLE001
        print(f"  (plot skipped: {exc})")

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cleanup-root", required=True)
    p.add_argument("--real-cernic-stations", nargs="*", default=[],
                   help="dev stations whose CLEAN Cernic is confirmed real")
    p.add_argument("--fp-cernic-stations", nargs="*", default=[],
                   help="dev stations with no real Cernic (all Cernic = FP)")
    p.add_argument("--real-clips-dir", default=None,
                   help="optional folder of hand-curated confirmed-real clips "
                        "(used instead of the clean bucket for positives)")
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    design_stations = list(args.real_cernic_stations) + list(args.fp_cernic_stations)
    bad = [s for s in design_stations if _is_test_station(s)]
    if bad:
        p.error(f"refusing to DESIGN on held-out test station(s): {', '.join(bad)}. "
                "IPA19/20 may only be used for final reporting, never to choose "
                "features/thresholds.")
    if not args.fp_cernic_stations:
        p.error("need --fp-cernic-stations (dev FP stations) for negatives")
    if not args.real_clips_dir and not args.real_cernic_stations:
        p.error("need --real-cernic-stations or --real-clips-dir for positives")

    print("Computing acoustic features on dev detections...")
    if args.real_clips_dir:
        print(f"   positives: curated dir {args.real_clips_dir}")
        real_feats = collect_dir(args.real_clips_dir)
    else:
        print(f"   positives: clean Cernic at {args.real_cernic_stations} "
              "(NOTE: clean bucket is a mix -- pass --real-clips-dir for a "
              "curated positive set)")
        real_feats = collect(args.cleanup_root, args.real_cernic_stations,
                             bucket="clean")
    print(f"   negatives: all Cernic at {args.fp_cernic_stations}")
    fp_feats = collect(args.cleanup_root, args.fp_cernic_stations, bucket="all")

    if not real_feats or not fp_feats:
        p.error("no clips collected -- check the paths / station names")

    report(real_feats, fp_feats, save_dir=args.save_dir)
    print("\nNext: if some feature separates well (AUC>~0.8), it can post-filter "
          "the saved detection clips. APPLY the chosen threshold to IPA19/20 "
          "only to REPORT final numbers -- never to tune.")


if __name__ == "__main__":
    main()
