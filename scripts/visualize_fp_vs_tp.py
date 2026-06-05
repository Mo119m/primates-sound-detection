"""
Visualize mel spectrograms of true positives vs false positives.

Plots the raw dB mel spectrogram (what the audio actually looks like) and the
224×224 normalized image (what VGG19 actually sees) side-by-side for real
detections vs false positives, so you can visually compare WHY the model
confuses certain sound types.

Reads the saved 2s WAV clips from auto_cleanup's clean_clips / suspicious_clips
directories. No model weights are needed -- this is pure audio inspection.

Example (Colab)
---------------
    !cd /content/primates-sound-detection && python scripts/visualize_fp_vs_tp.py \
        --cleanup-root "/content/drive/MyDrive/primates-sound-detection/outputs/auto_cleanup" \
        --real-cernic-stations IPA15ST IPA17ST IPA18ST \
        --fp-cernic-stations   IPA13ST IPA14ST IPA16ST IPA19ST \
        --n 5

    # Or just Cernic from specific stations:
    !cd /content/primates-sound-detection && python scripts/visualize_fp_vs_tp.py \
        --cleanup-root "/content/drive/MyDrive/primates-sound-detection/outputs/auto_cleanup" \
        --real-cernic-stations IPA15ST IPA17ST IPA18ST \
        --fp-cernic-stations   IPA19ST \
        --species Cernic \
        --n 6
"""

import argparse
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing import (
    audio_to_melspectrogram,
    normalize_spectrogram,
    resize_spectrogram,
)

SAMPLE_RATE = config.SAMPLE_RATE
CLIP_DURATION = config.CLIP_DURATION
CLIP_SUBDIRS = ("clean_clips", "suspicious_clips")


def _load_clip(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    target = int(round(CLIP_DURATION * SAMPLE_RATE))
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    elif len(audio) > target:
        audio = audio[:target]
    return audio


def _collect_clips(cleanup_root, stations, species, bucket="all", max_clips=None):
    """Collect clip paths from the given stations.

    bucket: "clean" = clean_clips only, "suspicious" = suspicious_clips only,
            "all" = both.
    """
    paths = []
    subdirs = {"clean": ["clean_clips"], "suspicious": ["suspicious_clips"],
               "all": list(CLIP_SUBDIRS)}[bucket]
    for st in stations:
        for sub in subdirs:
            d = os.path.join(cleanup_root, st, sub, species)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".wav"):
                    paths.append((st, sub, os.path.join(d, f)))
    if max_clips and len(paths) > max_clips:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(paths), max_clips, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def plot_comparison(real_clips, fp_clips, species, save_path=None):
    """Plot side-by-side raw mel spectrograms + 224×224 model-input images."""
    n_real = len(real_clips)
    n_fp = len(fp_clips)
    n_rows = max(n_real, n_fp)
    if n_rows == 0:
        print(f"No clips to plot for {species}.")
        return

    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"{species}: Real (left 2 cols) vs False Positive (right 2 cols)\n"
        f"Col 1,3 = raw mel-spectrogram (dB);  Col 2,4 = 224×224 model input",
        fontsize=13, y=1.01,
    )

    col_headers = [
        f"Real – raw mel-spec",
        f"Real – model sees",
        f"FP – raw mel-spec",
        f"FP – model sees",
    ]
    for j, h in enumerate(col_headers):
        axes[0, j].set_title(h, fontsize=11, fontweight="bold")

    for i in range(n_rows):
        # Real clip (left two columns)
        if i < n_real:
            st, sub, path = real_clips[i]
            audio = _load_clip(path)
            mel_db = audio_to_melspectrogram(audio)
            mel_norm = normalize_spectrogram(mel_db)
            mel_224 = resize_spectrogram(mel_norm)

            ax = axes[i, 0]
            librosa.display.specshow(
                mel_db, sr=SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                x_axis="time", y_axis="mel", fmin=config.FMIN, fmax=config.FMAX,
                ax=ax, cmap="magma",
            )
            label = f"{st}/{sub.replace('_clips','')}"
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("")

            ax2 = axes[i, 1]
            ax2.imshow(mel_224, aspect="auto", origin="lower", cmap="gray",
                       vmin=0, vmax=255)
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            axes[i, 0].axis("off")
            axes[i, 1].axis("off")

        # FP clip (right two columns)
        if i < n_fp:
            st, sub, path = fp_clips[i]
            audio = _load_clip(path)
            mel_db = audio_to_melspectrogram(audio)
            mel_norm = normalize_spectrogram(mel_db)
            mel_224 = resize_spectrogram(mel_norm)

            ax = axes[i, 2]
            librosa.display.specshow(
                mel_db, sr=SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                x_axis="time", y_axis="mel", fmin=config.FMIN, fmax=config.FMAX,
                ax=ax, cmap="magma",
            )
            label = f"{st}/{sub.replace('_clips','')}"
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlabel("")

            ax2 = axes[i, 3]
            ax2.imshow(mel_224, aspect="auto", origin="lower", cmap="gray",
                       vmin=0, vmax=255)
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            axes[i, 2].axis("off")
            axes[i, 3].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_distribution_overlay(real_clips, fp_clips, species, save_path=None):
    """Plot average spectral energy profiles for real vs FP clips."""
    def _avg_profile(clips):
        profiles = []
        for _, _, path in clips:
            audio = _load_clip(path)
            mel_db = audio_to_melspectrogram(audio)
            profiles.append(mel_db.mean(axis=1))
        return np.array(profiles) if profiles else np.array([])

    real_profiles = _avg_profile(real_clips)
    fp_profiles = _avg_profile(fp_clips)

    mel_freqs = librosa.mel_frequencies(n_mels=config.N_MELS,
                                         fmin=config.FMIN, fmax=config.FMAX)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: mean spectral profile
    if len(real_profiles):
        mean_r = real_profiles.mean(axis=0)
        std_r = real_profiles.std(axis=0)
        ax1.plot(mel_freqs, mean_r, "b-", label=f"Real (n={len(real_profiles)})")
        ax1.fill_between(mel_freqs, mean_r - std_r, mean_r + std_r, alpha=0.2, color="b")
    if len(fp_profiles):
        mean_f = fp_profiles.mean(axis=0)
        std_f = fp_profiles.std(axis=0)
        ax1.plot(mel_freqs, mean_f, "r-", label=f"FP (n={len(fp_profiles)})")
        ax1.fill_between(mel_freqs, mean_f - std_f, mean_f + std_f, alpha=0.2, color="r")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Mean power (dB)")
    ax1.set_title(f"{species}: average spectral profile")
    ax1.legend()
    ax1.set_xscale("log")

    # Right: mean temporal energy
    def _avg_temporal(clips):
        temporals = []
        for _, _, path in clips:
            audio = _load_clip(path)
            mel_db = audio_to_melspectrogram(audio)
            mel_norm = normalize_spectrogram(mel_db)
            mel_224 = resize_spectrogram(mel_norm)
            temporals.append(mel_224.mean(axis=0).astype(float))
        return np.array(temporals) if temporals else np.array([])

    real_t = _avg_temporal(real_clips)
    fp_t = _avg_temporal(fp_clips)
    x = np.linspace(0, CLIP_DURATION, 224)
    if len(real_t):
        ax2.plot(x, real_t.mean(axis=0), "b-", label="Real")
        ax2.fill_between(x, real_t.mean(0) - real_t.std(0),
                         real_t.mean(0) + real_t.std(0), alpha=0.2, color="b")
    if len(fp_t):
        ax2.plot(x, fp_t.mean(axis=0), "r-", label="FP")
        ax2.fill_between(x, fp_t.mean(0) - fp_t.std(0),
                         fp_t.mean(0) + fp_t.std(0), alpha=0.2, color="r")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Mean pixel intensity (0-255)")
    ax2.set_title(f"{species}: average temporal energy (model input)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cleanup-root", required=True)
    p.add_argument("--real-cernic-stations", nargs="*", default=[],
                   help="stations with confirmed real Cernic (positives)")
    p.add_argument("--fp-cernic-stations", nargs="*", default=[],
                   help="stations where all Cernic detections are FP")
    p.add_argument("--species", nargs="*", default=["Cernic", "Colobus_guereza"],
                   help="species to visualize (default: both)")
    p.add_argument("--n", type=int, default=5,
                   help="max clips per category to plot (default 5)")
    p.add_argument("--save-dir", default=None,
                   help="directory to save figures (default: just display)")
    args = p.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for species in args.species:
        print(f"\n{'='*60}")
        print(f"  {species}")
        print(f"{'='*60}")

        if species == "Cernic":
            if not args.real_cernic_stations:
                print("  No --real-cernic-stations given, skipping Cernic real.")
                real = []
            else:
                real = _collect_clips(
                    args.cleanup_root, args.real_cernic_stations, species,
                    bucket="clean", max_clips=args.n)
                print(f"  Real Cernic clips (clean bucket): {len(real)} "
                      f"from {args.real_cernic_stations}")

            if not args.fp_cernic_stations:
                print("  No --fp-cernic-stations given, skipping Cernic FP.")
                fp = []
            else:
                fp = _collect_clips(
                    args.cleanup_root, args.fp_cernic_stations, species,
                    bucket="all", max_clips=args.n)
                print(f"  FP Cernic clips: {len(fp)} from {args.fp_cernic_stations}")

        elif species == "Colobus_guereza":
            real = []
            print("  (No real Colobus at any dev station — only FP available)")
            all_stations = sorted(set(
                (args.real_cernic_stations or []) +
                (args.fp_cernic_stations or [])))
            fp = _collect_clips(
                args.cleanup_root, all_stations, species,
                bucket="all", max_clips=args.n)
            print(f"  FP Colobus clips: {len(fp)} from {all_stations}")
        else:
            print(f"  Unknown species {species}, skipping.")
            continue

        if not real and not fp:
            print("  No clips found — check paths.")
            continue

        save_comp = (os.path.join(args.save_dir, f"{species}_comparison.png")
                     if args.save_dir else None)
        plot_comparison(real, fp, species, save_path=save_comp)

        save_dist = (os.path.join(args.save_dir, f"{species}_profiles.png")
                     if args.save_dir else None)
        plot_distribution_overlay(real, fp, species, save_path=save_dist)


if __name__ == "__main__":
    main()
