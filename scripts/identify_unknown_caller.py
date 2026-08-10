"""
Narrow down what the surviving dawn candidates actually are.

The dawn probe searched 194 hours of dawn audio across all 16 stations, scored
698,007 windows, and produced 124 windows above 0.10 on the Colobus channel. Of
those, 117 are physically incapable of being a roar -- more than 95 % of their
energy sits above 1.5 kHz. The 7 that survive the low-frequency screen all come
from one recording, IPA17ST on 2021-02-25 at 05:30, and form a bout spread over
twelve minutes with a peak model score of 1.0000.

The project owner listened to all seven and reported: **low guttural roars, but
not Colobus.** So they are a real animal, low-frequency, calling at dawn, in
bouts, at one station while nine others stay silent -- and the classifier has no
class for them, so its Colobus unit absorbs them. That is a closed-set failure,
not a threshold problem, and no gate can fix it: the low-frequency gate passes
them for exactly the reason it passes a genuine roar.

This script does not claim to name the species. It ranks the reference material
already on disk by acoustic distance to those seven windows, so that a human who
knows the site has two or three specific things to check instead of an open
question. Everything it compares against is material this project already owns:

  data/species/Colobus*            the archival guereza roars the model learned
  data/background/Cercocebus*      collared mangabey, a loud low-frequency caller
  data/background/Pan troglodytes  chimpanzee
  data/background/*                site background and confirmed field negatives
  <drive>/Primates training data/  field-mined clips, incl. 78 Cercocebus
                                   windows that appear in no manifest

Features are deliberately interpretable rather than learned: band energies,
spectral shape, MFCC means, and the modulation rate that separates a rhythmic
bout from a single transient. A learned embedding from the four-class model
would be circular here -- it is the thing that got this wrong.

Read the output as a shortlist, not an answer. Distances between a 2 s field
window and a curated reference clip are dominated by channel as much as by
species.

Usage:
    python scripts/identify_unknown_caller.py
    python scripts/identify_unknown_caller.py --top 5
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
QUEUE = os.path.join(REPO, "data/outputs/colobus_dawn_queue")
SR = 22050          # enough for a low-frequency comparison, and 2x faster to load
MAX_PER_CLASS = 60  # cap so one large folder cannot dominate the ranking


def describe(y, sr):
    """Interpretable descriptors. Nothing learned, nothing model-derived."""
    import librosa
    if y.size < 1024:
        return None
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = S.sum() + 1e-12

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(S[m].sum() / tot)

    mag = np.sqrt(S)
    cent = float(librosa.feature.spectral_centroid(S=mag, sr=sr).mean())
    bw = float(librosa.feature.spectral_bandwidth(S=mag, sr=sr).mean())
    roll = float(librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(S=mag).mean())

    # Amplitude modulation: a rhythmic bout and a single transient differ here
    # far more than they differ in average spectrum.
    env = S.sum(axis=0)
    env = env / (env.max() + 1e-12)
    if env.size > 8:
        e = env - env.mean()
        sp = np.abs(np.fft.rfft(e))
        fr = np.fft.rfftfreq(e.size, d=512 / sr)
        keep = (fr > 0.5) & (fr < 30)
        mod_peak = float(fr[keep][np.argmax(sp[keep])]) if keep.any() else 0.0
        mod_depth = float(sp[keep].max() / (sp[keep].sum() + 1e-12)) if keep.any() else 0.0
    else:
        mod_peak = mod_depth = 0.0

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    return dict(
        e_0_250=band(0, 250), e_250_500=band(250, 500), e_500_1k=band(500, 1000),
        e_1k_2k=band(1000, 2000), e_2k_4k=band(2000, 4000), e_4k_8k=band(4000, 8000),
        centroid=cent, bandwidth=bw, rolloff=roll, flatness=flat,
        mod_peak_hz=mod_peak, mod_depth=mod_depth,
        **{f"mfcc{i}": float(v) for i, v in enumerate(mf)},
    )


def load_folder(paths, label, limit=MAX_PER_CLASS):
    import librosa
    rows = []
    for p in paths[:limit]:
        try:
            y, sr = librosa.load(p, sr=SR, mono=True, duration=5.0)
        except Exception:
            continue
        d = describe(y, sr)
        if d:
            d["label"] = label
            d["path"] = os.path.basename(p)
            rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", default=QUEUE)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/unknown_caller_ranking.csv"))
    args = ap.parse_args()

    targets = sorted(glob.glob(os.path.join(args.queue, "*.wav")))
    if not targets:
        sys.exit(f"no candidate wavs in {args.queue} -- run screen_dawn_candidates.py --export")
    print(f"{len(targets)} candidate windows to identify\n")

    # Reference material, all of it already in this project.
    refs = {}
    for name, pattern in [
        ("Colobus_guereza (archival)", "data/species/Colobus guereza*/**/*.wav"),
        ("Colobus_confuser (field)",   "data/species/Colobus_confuser/**/*.wav"),
        ("Cernic (site)",              "data/species/CERNIC*/**/*.wav"),
        ("Cercocebus torquatus",       "data/background/Cercocebus torquatus*/**/*.wav"),
        ("Pan troglodytes",            "data/background/Pan troglodytes*/**/*.wav"),
        ("background noise",           "data/background/background noise*/**/*.wav"),
        ("field FP negatives",         "data/background/field_fp_negatives/**/*.wav"),
        ("wrong classified",           "data/background/wrong classified/**/*.wav"),
    ]:
        hits = glob.glob(os.path.join(REPO, pattern), recursive=True)
        if hits:
            refs[name] = hits
    drive_ref = os.path.join(os.path.dirname(config.IPA_ROOT.rstrip("/\\")),
                             "Primates training data")
    if os.path.isdir(drive_ref):
        for d in sorted(os.listdir(drive_ref)):
            hits = glob.glob(os.path.join(drive_ref, d, "**", "*.wav"), recursive=True)
            if hits:
                refs[f"drive: {d[:38]}"] = hits

    rows = []
    for name, paths in refs.items():
        r = load_folder(paths, name)
        print(f"  {name:44s} {len(r):3d} clips")
        rows += r
    if not rows:
        sys.exit("no reference audio found")

    ref = pd.DataFrame(rows)

    import librosa
    trows = []
    for p in targets:
        y, sr = librosa.load(p, sr=SR, mono=True)
        d = describe(y, sr)
        if d:
            d["path"] = os.path.basename(p)
            trows.append(d)
    tgt = pd.DataFrame(trows)

    feats = [c for c in ref.columns if c not in ("label", "path")]
    mu, sd = ref[feats].mean(), ref[feats].std().replace(0, 1)
    R = ((ref[feats] - mu) / sd).to_numpy()
    T = ((tgt[feats] - mu) / sd).to_numpy()

    print(f"\n{len(feats)} descriptors, {len(ref)} reference clips, {len(tgt)} targets\n")
    out = []
    for i, t in enumerate(T):
        d = np.linalg.norm(R - t, axis=1)
        ref_d = ref.assign(dist=d)
        per = (ref_d.groupby("label")["dist"]
               .agg(nearest="min", median="median", n="size")
               .sort_values("nearest"))
        print(f"--- {tgt.path.iloc[i][:64]}")
        print(per.head(args.top).to_string())
        print()
        for lab, r in per.iterrows():
            out.append({"target": tgt.path.iloc[i], "reference": lab,
                        "nearest": round(r.nearest, 3), "median": round(r["median"], 3),
                        "n": int(r.n)})

    o = pd.DataFrame(out)
    o.to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    print("\nAggregate over all candidate windows (lower = closer):")
    print(o.groupby("reference")["nearest"].mean().sort_values().head(args.top).to_string())
    print("\nA shortlist, not an identification. Channel differences between a "
          "field window\nand a curated clip can outweigh species differences.")


if __name__ == "__main__":
    main()
