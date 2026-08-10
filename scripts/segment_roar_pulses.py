"""
Cut guereza roars into pulses, the way the C. nictitans class is cut into
hacks, keks and pyows.

Why this might matter. The putty-nosed class works because its reference
material is already decomposed into discrete syllables of a few tenths of a
second, each embedded in ambient to fill the 2 s window the model eats. The
guereza class is not decomposed at all: a roaring phrase is about fifteen pulses
of roughly 0.7 s spread over ten seconds, and every training clip is a 2 s or 5 s
crop taken at the loudest point, so where the crop falls relative to the pulses
is arbitrary. Two clips of the same roar can be cut half a pulse apart and the
model has to learn both as the same thing.

Cutting on the pulses instead gives aligned units, and units the size of what the
detector actually sees: a 2 s window at deployment can hold two or three pulses
and never a whole phrase, so a phrase is not what the model should be trained on.

Alignment is the whole of the argument, and it is worth being clear that the
obvious second argument does not survive contact with the data. Segmenting all 79
library recordings yields 383 pulses, against the 617 windows the class already
has: a 2 s window sliding over the same material produces more units than the
pulses in it do. This gives better-aligned examples, not more of them.

Two other things it does not fix. Source diversity is unchanged, since
``train.source_group`` groups every pulse from one recording together, as it
should. And the archival-to-field gap that this class most likely suffers from is
untouched, because the material is the same material.

    python scripts/segment_roar_pulses.py --report
    python scripts/segment_roar_pulses.py --export-listening 12
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SR = 22050
LOW, HIGH = 100, 1500       # the band a guereza roar occupies
MIN_PULSE, MAX_PULSE = 0.15, 1.60
MIN_GAP = 0.10              # two runs closer than this are one pulse


def envelope(y, sr):
    """Low-band energy envelope and its frame rate."""
    import librosa
    hop = 128
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop)) ** 2
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    env = S[(f >= LOW) & (f < HIGH)].sum(axis=0)
    return env, sr / hop


def quietest_stretch(y, sr, seconds):
    """The lowest-energy window of the recording: its own background."""
    n = int(seconds * sr)
    if y.size <= n:
        return y
    # Energy in a sliding window, cheaply, via the cumulative sum of squares.
    cs = np.concatenate(([0.0], np.cumsum(y.astype(np.float64) ** 2)))
    energy = cs[n:] - cs[:-n]
    i = int(np.argmin(energy))
    return y[i:i + n]


def denoise_recording(y, sr, seconds):
    """Spectral-gate a whole recording against its own quietest stretch.

    Done on the recording rather than on each pulse, which is what Sagar et al.
    describe and what the alternative fails at: a roar is a pulse train, so the
    audio beside one pulse is another pulse.
    """
    import noisereduce
    noise = quietest_stretch(y, sr, seconds)
    if noise.size < int(0.2 * sr):
        return y
    return noisereduce.reduce_noise(y=y, sr=sr, y_noise=noise, stationary=True)


def find_pulses(y, sr):
    """Start and end time of each pulse, in seconds."""
    env, fps = envelope(y, sr)
    if env.size < 8 or env.max() <= 0:
        return []
    e = env / env.max()
    # A pulse is a run above a floor set relative to this recording's own peak,
    # since library material varies in level by tens of dB. The floor is low on
    # purpose. A percentile floor was tried first and failed for a reason worth
    # recording: in a recording that is mostly roar, the 60th percentile sits
    # *inside* the roar, so only the few loudest peaks cleared it and the
    # segmenter returned one pulse per file against a published fifteen.
    thr = 0.05
    above = e > thr
    runs, start = [], None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(above)))

    out = []
    for s, t in runs:
        t0, t1 = s / fps, t / fps
        if t1 - t0 < MIN_PULSE:
            continue
        if out and t0 - out[-1][1] < MIN_GAP:      # merge near-touching runs
            out[-1] = (out[-1][0], t1)
            continue
        out.append((t0, min(t1, t0 + MAX_PULSE)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="D:/Primates training data")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--export-listening", type=int, default=0,
                    help="Write this many recordings' pulses to a folder so a "
                         "person can check the cuts by ear before anything is "
                         "trained on them.")
    ap.add_argument("--export-class", metavar="DIR", default=None,
                    help="Write every kept pulse as a RAW short clip to DIR, to "
                         "be listed in config.SPECIES_FOLDERS. Raw is correct: "
                         "hacks, keks and pyows are stored at 0.05-0.15 s and "
                         "embed_in_background fills them to the window at load "
                         "time, drawing a fresh bed and SNR on every epoch. "
                         "Pre-embedding here would freeze one bed per pulse and "
                         "throw that variation away.")
    ap.add_argument("--export-bouts", metavar="DIR", default=None,
                    help="Write full-length windows carrying a run of pulses "
                         "instead of one pulse each, so the class encodes the "
                         "roar's rhythm rather than a single thump.")
    ap.add_argument("--bout-seconds", type=float, default=2.0,
                    help="Window length; match config.WINDOW_SIZE.")
    ap.add_argument("--bout-min-pulses", type=int, default=2,
                    help="Discard a window carrying fewer pulses than this.")
    ap.add_argument("--bout-lead", type=float, default=0.15,
                    help="Seconds of run-in before the anchoring pulse onset.")
    ap.add_argument("--export-chunks", metavar="DIR", default=None,
                    help="Write contiguous windows over the roaring region with "
                         "no pulse-count requirement, so the class carries the "
                         "continuous form as well as the discrete one.")
    ap.add_argument("--chunk-seconds", type=float, default=1.8,
                    help="Window length. Keep below config.WINDOW_SIZE or the "
                         "clip is cropped instead of embedded and arrives at "
                         "library SNR.")
    ap.add_argument("--chunk-hop", type=float, default=0.9)
    ap.add_argument("--chunk-min-voiced", type=float, default=0.25,
                    help="Discard a window whose low-band envelope is above "
                         "floor for less than this fraction of its length.")
    ap.add_argument("--min-duration", type=float, default=0.25,
                    help="Drop pulses shorter than this. The expert called a "
                         "0.27 s pulse obvious and a 0.22 s one 'hard, I am not "
                         "100%% sure', so the marginal ones carry label noise "
                         "rather than signal.")
    ap.add_argument("--denoise", action="store_true",
                    help="Spectral-gating noise reduction on each pulse before "
                         "writing it, isolating the call from the library "
                         "recording's own background. Without this the training "
                         "clip ends up carrying two backgrounds: the archive's "
                         "and the deployment ambient it is later embedded in. "
                         "Sagar et al. denoise then overlay PAM background for "
                         "exactly this reason.")
    ap.add_argument("--noise-seconds", type=float, default=1.5,
                    help="Length of the quietest stretch of each recording, "
                         "used as its noise profile. Taking the audio directly "
                         "around a pulse instead does not work here: the "
                         "inter-pulse gap is about 0.57 s, so a one-second "
                         "context is mostly the neighbouring pulses, and "
                         "spectral gating then learns the roar as the thing to "
                         "remove. Measured that way it cut RMS to 14 % of the "
                         "original and made the roar band a SMALLER share of "
                         "the clip, which is the opposite of the intent.")
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/outputs/roar_pulses"))
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    # Every audio file under --source, recursively, with no name matching. An
    # earlier version globbed for directories matching "*olobus*", which looked
    # convenient and was not: pointed at data/species it also matched
    # Colobus_confuser, so 907 confirmed FALSE POSITIVES were segmented into the
    # positive class, and it matched this script's own output directory as well.
    # The caller now says exactly which folder to read, and a wrong folder is
    # visible in the count rather than silent.
    files = [p for p in sorted(glob.glob(
        os.path.join(args.source, "**", "*"), recursive=True))
        if p.lower().endswith((".wav", ".mp3", ".flac")) and os.path.isfile(p)]
    if not files:
        sys.exit(f"no audio under {args.source}")
    out_dir = os.path.abspath(args.export_class) if args.export_class else None
    if out_dir:
        files = [p for p in files
                 if os.path.dirname(os.path.abspath(p)) != out_dir]
        if not files:
            sys.exit(f"--source {args.source} contains only the output folder")
    print(f"{len(files)} library recordings\n")

    per_file, ipis, widths = [], [], []
    detail = []
    for p in files:
        try:
            y, sr = librosa.load(p, sr=SR, mono=True)
        except Exception:
            continue
        pulses = find_pulses(y, sr)
        per_file.append(len(pulses))
        widths += [b - a for a, b in pulses]
        starts = [a for a, _ in pulses]
        ipis += list(np.diff(starts)) if len(starts) > 1 else []
        detail.append((p, y, sr, pulses))

    n = np.array(per_file)
    w = np.array(widths)
    g = np.array(ipis)
    print(f"pulses per recording : median {np.median(n):.0f}  "
          f"mean {n.mean():.1f}  max {n.max()}  zero-pulse files {int((n == 0).sum())}")
    if w.size:
        print(f"pulse width (s)      : median {np.median(w):.2f}  "
              f"IQR {np.percentile(w, 25):.2f}-{np.percentile(w, 75):.2f}")
    if g.size:
        g = g[(g > 0.2) & (g < 3.0)]
        print(f"inter-pulse gap (s)  : median {np.median(g):.2f}  "
              f"IQR {np.percentile(g, 25):.2f}-{np.percentile(g, 75):.2f}  "
              f"=> {1 / np.median(g):.2f} Hz")
    print(f"total pulses         : {int(n.sum())}  "
          f"(the class currently has 617 windows from {len(files)} recordings)")
    print("\npublished description: ~15 pulses of ~0.7 s, a ~1.4 Hz train over "
          "~10 s.\nCompare the two lines above before trusting any of this.")

    if args.export_class:
        os.makedirs(args.export_class, exist_ok=True)
        kept = 0
        for p, y, sr, pulses in detail:
            stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            # Pulses are located on the ORIGINAL envelope, which is what the
            # thresholds were calibrated against, and then cut from the
            # denoised copy. Segmenting the denoised signal instead would move
            # the boundaries for reasons that have nothing to do with the call.
            clean = (denoise_recording(y, sr, args.noise_seconds)
                     if args.denoise else None)
            for i, (a, b) in enumerate(pulses, 1):
                if b - a < args.min_duration:
                    continue
                lo_i = int(a * sr)
                hi_i = int(min(b, a + MAX_PULSE) * sr)
                seg = y[lo_i:hi_i]
                if seg.size < int(0.05 * sr):
                    continue
                if args.denoise:
                    seg = clean[lo_i:hi_i]
                # The source recording is kept in the name so
                # train.source_group keeps every pulse of one roar on the same
                # side of a split. Without it each pulse would look like an
                # independent recording and validation accuracy would be a
                # memorisation score.
                sf.write(os.path.join(
                    args.export_class,
                    f"{stem}__pulse{i:02d}.wav"), seg, SR)
                kept += 1
        print(f"\nwrote {kept} raw pulses (>= {args.min_duration}s) to\n  "
              f"{args.export_class}")
        print("These are shorter than the analysis window on purpose. "
              "load_audio_file\nembeds anything short in a real background bed, "
              "which is what makes the\nputty-nose syllables work and what "
              "zero-padding would break.")

    if args.export_bouts:
        # One pulse per training example throws away the thing that makes a roar
        # a roar. A putty-nosed hack is a single event of about 0.08 s and a 2 s
        # window holding one of them is a faithful example; a guereza roar is a
        # train of ~0.37 s pulses at roughly 1.4 Hz, and a window holding one
        # pulse is a low-frequency thump, which is a category many non-calls
        # belong to. Trained on those, the model has no way to learn the rhythm,
        # and the symptom is visible in deployment: it returns 0.9999 on clips
        # whose energy is 99 % above 1.5 kHz -- nothing a roar could produce --
        # because a lone thump was all it was ever asked to recognise.
        #
        # These windows are the length of the analysis window and carry two or
        # more pulses each. Alignment, which was the original argument for
        # cutting pulses at all, is kept: every window is anchored a fixed lead
        # before a pulse onset, so the train always enters at the same phase.
        os.makedirs(args.export_bouts, exist_ok=True)
        win = args.bout_seconds
        lead = args.bout_lead
        kept, per_rec, counts = 0, 0, []
        for p, y, sr, pulses in detail:
            if len(pulses) < args.bout_min_pulses:
                continue
            stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            clean = (denoise_recording(y, sr, args.noise_seconds)
                     if args.denoise else None)
            src = clean if args.denoise else y
            starts = [a for a, _ in pulses]
            wrote_here = 0
            for i, a in enumerate(starts):
                t0 = max(0.0, a - lead)
                t1 = t0 + win
                if t1 > len(y) / sr:
                    t0 = max(0.0, len(y) / sr - win)
                    t1 = t0 + win
                inside = sum(1 for s in starts if t0 <= s < t1)
                if inside < args.bout_min_pulses:
                    continue
                seg = src[int(t0 * sr):int(t1 * sr)]
                if seg.size < int(0.9 * win * sr):
                    continue
                sf.write(os.path.join(
                    args.export_bouts,
                    f"{stem}__bout{i:02d}_{inside}p.wav"), seg, SR)
                kept += 1
                wrote_here += 1
                counts.append(inside)
            if wrote_here:
                per_rec += 1
        c = np.array(counts) if counts else np.array([0])
        print(f"\nwrote {kept} {win:g}s bout windows from {per_rec} recordings "
              f"to\n  {args.export_bouts}")
        print(f"pulses per window: median {np.median(c):.0f}  "
              f"max {c.max()}  (minimum required {args.bout_min_pulses})")
        print("Full-length windows, so no background bed is added: these are "
              "already\nthe size the model eats, and the ambient between pulses "
              "is the recording's\nown rather than one borrowed from a "
              "deployment site.")

    if args.export_chunks:
        # Contiguous windows over the roaring region, with no pulse alignment and
        # no minimum pulse count. This exists because of how the bout experiment
        # failed: bouts required a run of two or more pulses, which made "a
        # 1.7 Hz train" part of the definition of the class, and the model
        # learned to demand it. Field recordings do not deliver it -- distance
        # reverberation fills the gaps between pulses, so a roar arrives as one
        # smeared event (Richards & Wiley 1980) -- and field sensitivity fell
        # from 2/9 to 1/9.
        #
        # Sliding without a structural requirement produces windows holding one
        # pulse, several, or a pulse and its tail, in whatever proportion the
        # recording contains. Combined with the single pulses this class is
        # otherwise built from, the model sees both the discrete and the
        # continuous form and is not forced to treat either as necessary.
        #
        # 1.8 s rather than the 2 s window, and for a reason that has bitten
        # this class once already: anything at or over the window length skips
        # embed_in_background and is cropped instead, which is how the 172 5 s
        # clips arrived at +12.2 dB against the +/-1.0 dB of the field-verified
        # material -- the easiest examples in a class with no hard ones.
        os.makedirs(args.export_chunks, exist_ok=True)
        win, hop = args.chunk_seconds, args.chunk_hop
        kept, per_rec = 0, 0
        for p, y, sr, pulses in detail:
            if not pulses:
                continue
            stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
            clean = (denoise_recording(y, sr, args.noise_seconds)
                     if args.denoise else None)
            src = clean if args.denoise else y
            lo = max(0.0, pulses[0][0] - 0.2)
            hi = min(len(y) / sr, pulses[-1][1] + 0.2)
            t, wrote_here = lo, 0
            while t + win <= hi + 1e-6:
                seg = src[int(t * sr):int((t + win) * sr)]
                if seg.size >= int(0.95 * win * sr):
                    # Keep only windows that actually carry roar energy; a
                    # window landing in a gap is ambient labelled as a call.
                    env, fps = envelope(seg, sr)
                    if env.size and env.max() > 0:
                        voiced = float((env / env.max() > 0.05).mean())
                        if voiced >= args.chunk_min_voiced:
                            sf.write(os.path.join(
                                args.export_chunks,
                                f"{stem}__chunk{int(t * 10):04d}.wav"), seg, SR)
                            kept += 1
                            wrote_here += 1
                t += hop
            if wrote_here:
                per_rec += 1
        print(f"\nwrote {kept} {win:g}s contiguous chunks from {per_rec} "
              f"recordings to\n  {args.export_chunks}")
        print("No pulse-count requirement: these carry whatever structure the "
              "recording\nhas at that moment, which is the point.")

    if not args.export_listening:
        return

    # The background bed the C. nictitans syllables are embedded in. Reusing the
    # same pool matters: the point of this exercise is to build the guereza class
    # the way the working class is built, not a near-miss of it.
    import config
    import data_loader
    pool = []
    for folder in config.BACKGROUND_FOLDERS:
        for q in data_loader.scan_audio_files(
                os.path.join(REPO, "data"), folder)[:400]:
            try:
                b, _ = librosa.load(q, sr=SR, mono=True)
            except Exception:
                continue
            if b.size >= SR:
                pool.append(b)
    if not pool:
        sys.exit("no background audio found; embedding needs a real bed")
    print(f"\nbackground bed: {len(pool)} real ambient clips")

    os.makedirs(args.out, exist_ok=True)
    detail.sort(key=lambda d: -len(d[3]))
    target = int(config.CLIP_DURATION * SR)
    written = 0
    for p, y, sr, pulses in detail[: args.export_listening]:
        stem = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        sf.write(os.path.join(args.out, f"{stem}__ORIGINAL.wav"), y, sr)
        for i, (a, b) in enumerate(pulses, 1):
            # One pulse, alone. Not a 2 s window centred on it: that would carry
            # the neighbouring pulses along and teach the model to recognise a
            # stretch of roar rather than a single call, which is not what a
            # distant or partly masked roar looks like in a 2 s window.
            lo, hi = int(a * sr), int(min(b, a + MAX_PULSE) * sr)
            pulse = y[lo:hi]
            if pulse.size < int(0.05 * sr):
                continue
            clip = data_loader.embed_in_background(pulse, target, pool)
            sf.write(os.path.join(
                args.out, f"{stem}__pulse{i:02d}_{a:.2f}s_{(b - a):.2f}s.wav"),
                clip, SR)
            written += 1
    print(f"\nwrote {written} single-pulse clips and {args.export_listening} "
          f"originals to\n  {args.out}")
    print("Each pulse clip is one pulse alone in real forest ambient at a random "
          "position\nand a random 3-15 dB SNR, which is exactly how a hack, kek "
          "or pyow is prepared.\nListen to an original and then its pulses: if a "
          "single pulse is still recognisable\non its own, the model has a "
          "chance; if it only means something as part of the\ntrain, a 2 s window "
          "cannot carry it and the decomposition has to happen at the\nevent "
          "level instead.")


if __name__ == "__main__":
    main()
