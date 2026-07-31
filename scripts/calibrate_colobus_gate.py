"""
Recalibrate the Colobus low-frequency gate against the negatives it actually meets.

``config.LOWFREQ_GATE_THRESHOLD = 0.20`` carries its own calibration note: "FP
max=0.092, Colobus p05=0.261". Those false positives were the ``Colobus_confuser``
clips -- the pulsed forest sound the model kept mistaking for a roar -- and
against that population the gate is nearly perfect, because a confuser clip has
almost no energy below 1500 Hz (median 0.022).

The field disagrees. Of the 253 *C. guereza* detections the deployment produced,
the reviewer found no genuine roar at all: they are thunder and other
low-frequency noise. Thunder is not a high-frequency sound that a low-frequency
test can reject -- it *is* low-frequency, median ratio 0.396 and upper quartile
0.826, overlapping the reference roars the gate exists to protect. A threshold
calibrated on a negative population that never contained thunder cannot exclude
thunder at any setting that keeps roars.

So this script does two things:

1. Re-derives the threshold against the field detections as the negative class,
   and prints the whole trade-off rather than one number, because the choice
   depends on whether recall or precision matters more at this site.
2. Searches for a second, orthogonal criterion. A roar is a pulse train -- a
   sequence of discrete snorts a few per second -- while thunder is a single
   slow decay and rain is stationary noise. Features that measure that
   difference can separate the two populations *within* the low band, where the
   energy ratio cannot.

A caveat the numbers cannot remove: the positive class here is the 617 reference
windows, cut from 172 expert-labelled source clips, and **no field detection has
ever been confirmed as a genuine roar**. Every "roars kept" figure below is
therefore measured on reference material, and transfers to the field only as far
as that material is representative. A feature that merely separates *reference
clips* from *field clips* -- by noise floor, by level, by recorder -- would score
perfectly here and fail completely in deployment. The confuser column is printed
for exactly that reason: it is field audio too, so a feature that scores high
against the field detections but cannot tell reference roars from field confuser
clips is reading provenance, not sound.

Usage:
    python scripts/calibrate_colobus_gate.py
    python scripts/calibrate_colobus_gate.py --out data/outputs/colobus_gate.csv
"""
import argparse
import glob
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def candidate_features(y, sr, cutoff=1500.0):
    """
    Measurements of the low band that a roar and a thunderclap differ on.

    All are computed inside the band below ``cutoff``, because above it the two
    populations are not in dispute. All are scale-free -- ratios, flatness,
    normalised modulation -- so a quiet distant roar and a loud near one score
    alike, and none can be satisfied by loudness alone.
    """
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = S[freqs < cutoff]
    if band.size == 0 or band.sum() <= 0:
        return None

    frame_rate = sr / 256.0
    env = band.sum(axis=0)
    if env.std() < 1e-12 or len(env) < 16:
        return None

    # Spectral flatness of the band, averaged over time. Thunder and rain are
    # noise: broad and flat. A roar has a fundamental and harmonics: peaky.
    mean_spec = band.mean(axis=1) + 1e-12
    flatness = float(np.exp(np.log(mean_spec).mean()) / mean_spec.mean())

    # Modulation spectrum of the band envelope. A roar bout repeats at a few
    # hertz; thunder has no repeat rate and rain has none either.
    v = env - env.mean()
    win = np.hanning(len(v))
    mag = np.abs(np.fft.rfft(v * win))
    mod_f = np.fft.rfftfreq(len(v), d=1.0 / frame_rate)
    pulse_band = (mod_f >= 1.0) & (mod_f <= 8.0)
    ref_band = (mod_f > 0.2) & (mod_f <= 30.0)
    pulse = float(mag[pulse_band].max() / (mag[ref_band].mean() + 1e-12))
    pulse_hz = float(mod_f[pulse_band][np.argmax(mag[pulse_band])])

    # Envelope shape. A pulse train swings; a decaying boom or steady rain does
    # not swing as much relative to its mean.
    crest = float(env.max() / (env.mean() + 1e-12))
    env_cv = float(env.std() / (env.mean() + 1e-12))

    # Onset density: discrete snorts produce onsets, a single boom produces one.
    flux = np.diff(band, axis=1).clip(min=0).sum(axis=0)
    onset_rate = float((flux > flux.mean() + flux.std()).sum()
                       / (len(flux) / frame_rate))

    return dict(flatness=flatness, pulse=pulse, pulse_hz=pulse_hz,
                crest=crest, env_cv=env_cv, onset_rate=onset_rate)


def scan(paths, group, limit=None):
    import librosa
    import config
    import detection

    rows = []
    for p in (paths[:limit] if limit else paths):
        try:
            y, sr = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
        except Exception:
            continue
        ratio = detection.lowfreq_energy_ratio(y)
        if ratio is None:
            continue
        feats = candidate_features(y, sr, config.LOWFREQ_GATE_CUTOFF)
        if feats is None:
            continue
        rows.append(dict(group=group, lf_ratio=float(ratio), path=p, **feats))
    return rows


def auc(pos, neg):
    """Probability a random positive scores above a random negative."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average ranks over ties so a coarse integer feature is not flattered.
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/outputs/colobus_gate_calibration.csv"))
    args = ap.parse_args()

    import config

    groups = {
        "reference_roar": sorted(glob.glob(os.path.join(
            REPO, "data/species/Colobus guereza 2s windows/*.wav"))),
        "field_detection": sorted(glob.glob(os.path.join(
            REPO, "data/outputs/detected_clips/Colobus_guereza/*/*/*.wav"))),
        "confuser": sorted(glob.glob(os.path.join(
            REPO, "data/species/Colobus_confuser/*.wav"))),
    }
    rows = []
    for name, paths in groups.items():
        print(f"scanning {name}: {len(paths)} clips")
        rows += scan(paths, name)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    feats = ["lf_ratio", "flatness", "pulse", "crest", "env_cv", "onset_rate"]
    roar = df[df.group == "reference_roar"]
    field = df[df.group == "field_detection"]
    conf = df[df.group == "confuser"]

    print(f"\n{'':14s} {'roar':>18s} {'field detection':>18s} {'confuser':>18s}")
    print(f"{'feature':14s} {'median (IQR)':>18s} {'median (IQR)':>18s} "
          f"{'median (IQR)':>18s}")
    print("-" * 72)
    for f in feats:
        cells = []
        for g in (roar, field, conf):
            q = g[f].quantile([0.25, 0.5, 0.75])
            cells.append(f"{q[0.5]:.2f} ({q[0.25]:.2f}-{q[0.75]:.2f})")
        print(f"{f:14s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")

    print(f"\nSeparation (AUC, roar as the positive class)")
    print(f"{'feature':14s} {'vs field detection':>20s} {'vs confuser':>14s}"
          f"   {'reading':s}")
    print("-" * 72)
    for f in feats:
        a_field = auc(roar[f], field[f])
        a_conf = auc(roar[f], conf[f])
        # A feature that only works against the field detections, while failing
        # against confuser clips recorded by the same hardware at the same site,
        # is separating provenance rather than sound.
        note = ""
        if a_field > 0.75 and a_conf < 0.6:
            note = "SUSPECT -- may be reading provenance"
        elif a_field > 0.75 and a_conf > 0.75:
            note = "separates both populations"
        print(f"{f:14s} {a_field:20.3f} {a_conf:14.3f}   {note}")

    print(f"\nThreshold on lf_ratio alone "
          f"(current setting {config.LOWFREQ_GATE_THRESHOLD})")
    print(f"{'threshold':>10s} {'reference roars kept':>21s} "
          f"{'field detections kept':>22s}")
    for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        mark = "  <- current" if abs(t - config.LOWFREQ_GATE_THRESHOLD) < 1e-9 else ""
        print(f"{t:10.2f} {100 * (roar.lf_ratio >= t).mean():20.1f}% "
              f"{100 * (field.lf_ratio >= t).mean():21.1f}%{mark}")

    best = max(("pulse", "flatness", "crest", "env_cv", "onset_rate"),
               key=lambda f: auc(roar[f], field[f]))
    lo_is_better = auc(roar[best], field[best]) < 0.5
    print(f"\nBest second criterion by AUC: {best}")
    print(f"{'lf_ratio':>10s} {best:>12s} {'roars kept':>12s} "
          f"{'field kept':>12s}")
    for lt in [0.30, 0.40, 0.50, 0.60]:
        for q in (0.10, 0.20, 0.30):
            cut = roar[best].quantile(q if not lo_is_better else 1 - q)
            keep_r = ((roar.lf_ratio >= lt) &
                      ((roar[best] <= cut) if lo_is_better
                       else (roar[best] >= cut))).mean()
            keep_f = ((field.lf_ratio >= lt) &
                      ((field[best] <= cut) if lo_is_better
                       else (field[best] >= cut))).mean()
            print(f"{lt:10.2f} {cut:12.2f} {100 * keep_r:11.1f}% "
                  f"{100 * keep_f:11.1f}%")

    print(f"\nWrote {args.out}")
    print("\nRemember: 'roars kept' is measured on reference windows. No field "
          "\ndetection has been confirmed as a genuine roar, so the gate cannot "
          "\nbe scored on field recall at all -- only on how much of a known-bad "
          "\npopulation it removes.")


if __name__ == "__main__":
    main()
