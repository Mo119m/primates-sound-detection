"""
Turn the raw dawn-probe output into a listening queue worth someone's time.

`colobus_dawn_probe.py` deliberately records every window scoring above 0.10 and
applies no filtering, because the point was to see the score distribution rather
than to trust a threshold. That raw list is not listenable as-is. The
highest-scoring candidate it found -- 0.8624, higher than anything else in 31
hours of dawn audio -- has a low-frequency ratio of **0.0056**: 99.4 % of its
energy sits above 1.5 kHz. A *C. guereza* roar is a low-frequency
vocalisation, around 512 Hz. That window cannot be one, and the deployment's
gate had already dropped it (its per-recording CSV is empty), correctly.

That single case is the whole argument for this script. It also says something
about the model worth keeping in view: `FrequencyCoord` exists specifically so
the network can learn "this texture *at low frequency* is Colobus"
(`src/model.py:39-46`), and it assigned 0.86 to a window with essentially no
low-frequency energy at all. The architectural mechanism did not do its job; a
hand-written gate caught it afterwards.

So every candidate is screened on the same physical criterion the gate uses,
and the survivors are exported as audio ready to play. Two numbers travel with
each: the model score, and the low-frequency ratio that decides whether it could
be a roar at all.

WHAT A NULL RESULT HERE MEANS, AND WHAT IT DOES NOT
---------------------------------------------------
If nothing survives screening, that is evidence the detector never produced a
plausible roar candidate at dawn across the array. It is **not** evidence that
*C. guereza* is absent: every one of the 789 training positives is Macaulay
Library media recorded elsewhere, so a model that cannot recognise this site's
channel would look identical. Cernic is scored alongside in the probe as a
positive control precisely so that distinction stays visible -- if Cernic's
distribution is bimodal and healthy on the same audio, the pipeline works and
the Colobus silence is about Colobus.

Usage:
    python scripts/screen_dawn_candidates.py
    python scripts/screen_dawn_candidates.py --min-ratio 0.4 --export
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
PROBE = os.path.join(REPO, "data/outputs/colobus_dawn_probe")


def low_freq_ratio(y, sr):
    """Share of in-band energy below LOWFREQ_GATE_CUTOFF, as the gate computes it."""
    import librosa
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (f >= config.FMIN) & (f <= config.FMAX)
    low = (f >= config.FMIN) & (f < config.LOWFREQ_GATE_CUTOFF)
    return float(S[low].sum() / (S[band].sum() + 1e-12))


def find_source(name, root):
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--audio-root", default=config.IPA_ROOT)
    ap.add_argument("--min-score", type=float, default=0.10)
    # Defaults to 0.0, i.e. no filtering. It used to default to
    # config.LOWFREQ_GATE_THRESHOLD (0.40) on the reasoning that "a roar cannot
    # be high-frequency". Measured on the nine expert-confirmed field roars in
    # data/species/Colobus guereza field/, that cutoff rejects three of them:
    # their ratios are 0.0200, 0.0712, 0.2665, 0.4271, 0.4719, 0.5588, 0.6172,
    # 0.6924 and 0.7234. The ratio is whole-clip energy, so a genuine roar
    # arriving faintly inside a loud insect chorus scores low -- the threshold
    # was calibrated on close-range archival clips where the roar dominates the
    # spectrum, and does not transfer. config.py:403 already disables the same
    # gate in the detection pipeline for the same reason; this script and
    # scan_roar_pulse.py were the two places still applying it.
    #
    # It matters here more than in most places because this script decides what
    # a person listens to. Of 124 candidates, 117 were dropped below 0.40 and
    # never heard, 56 of them at ratios at or above the lowest confirmed roar.
    ap.add_argument("--min-ratio", type=float, default=0.0,
                    help="Reject candidates whose low-frequency ratio is below "
                         "this. Off by default: the 0.40 cutoff this used to "
                         "carry rejects 3 of the 9 confirmed field roars.")
    ap.add_argument("--export", action="store_true",
                    help="Write the surviving windows as wav files to listen to.")
    ap.add_argument("--out", default=os.path.join(REPO, "data/outputs/colobus_dawn_queue"))
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.probe, "*_windows.csv"))
    if not files:
        sys.exit(f"no probe output under {args.probe}")
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    c = w[(w["species"] == "Colobus_guereza") & (w["score"] >= args.min_score)]
    c = c.sort_values("score", ascending=False).reset_index(drop=True)
    print(f"{len(c)} raw candidates at score >= {args.min_score}")
    if not len(c):
        return

    import librosa
    import soundfile as sf
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for r in c.itertuples():
        src = find_source(r.file, args.audio_root)
        if src is None:
            print(f"  MISSING SOURCE: {r.file} -- not screened, not dropped")
            rows.append({"file": r.file, "start_s": r.start_s, "score": r.score,
                         "low_freq_ratio": None, "verdict": "source-missing"})
            continue
        y, sr = librosa.load(src, sr=config.SAMPLE_RATE, offset=float(r.start_s),
                             duration=config.CLIP_DURATION, mono=True)
        ratio = low_freq_ratio(y, sr)
        keep = ratio >= args.min_ratio
        rows.append({"file": r.file, "start_s": r.start_s, "score": round(r.score, 4),
                     "low_freq_ratio": round(ratio, 4),
                     "verdict": "listen" if keep else "high-frequency, not a roar"})
        if keep and args.export:
            pad = 1.0
            y2, _ = librosa.load(src, sr=config.SAMPLE_RATE,
                                 offset=max(0.0, float(r.start_s) - pad),
                                 duration=config.CLIP_DURATION + 2 * pad, mono=True)
            stem = f"{r.score:.3f}_lf{ratio:.3f}_{os.path.splitext(r.file)[0][:48]}_t{int(r.start_s)}s.wav"
            sf.write(os.path.join(args.out, stem), y2, sr)

    out = pd.DataFrame(rows)
    path = os.path.join(args.out, "candidates.csv")
    out.to_csv(path, index=False)

    keep = out[out["verdict"] == "listen"]
    print(f"\n{len(keep)} survive the low-frequency screen (ratio >= {args.min_ratio})")
    print(f"{int((out['verdict'] == 'high-frequency, not a roar').sum())} rejected as "
          f"high-frequency -- a roar cannot look like that")
    if len(out[out["verdict"] == "source-missing"]):
        print(f"{len(out[out['verdict'] == 'source-missing'])} could not be screened "
              f"(source audio not found) -- these are NOT silently dropped")
    if len(keep):
        print()
        print(keep.to_string(index=False))
    print(f"\nwrote {path}")
    if args.export and len(keep):
        print(f"audio in {args.out}/  (1 s of context padded either side)")


if __name__ == "__main__":
    main()
