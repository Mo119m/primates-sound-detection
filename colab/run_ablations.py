"""The ablations that decide whether this head is a contribution or a detail.

The paper argues for two components the head adds over a conventional CRNN --
a four-band frequency split and a frequency-position encoding -- and until now
argued for them mechanistically, on six folds, without a number at sixteen. A
methods paper whose claimed contribution is the model cannot leave that
unmeasured, so this measures it.

Four arms, sixteen folds each, everything else identical:

    1. temporal_freqpos      the head as described. Run first so that if the
                             session dies early the baseline still exists.
    2. temporal              frequency pooled away entirely, the V7 path. The
                             largest architectural difference, and therefore
                             the comparison that carries the claim.
    3. temporal_freq         the band split without the position encoding, so
                             arms 2 and 3 separate the two components rather
                             than testing them as a bundle.
    4. temporal_freqpos      minus the 253 reviewed detections added to
       --drop-extra-confuser Colobus_confuser, which is a data ablation rather
                             than an architectural one and is last for that
                             reason.

The existing sixteen-fold result is not reused as arm 1. It was trained before
nine labels were corrected on 2026-08-18, and nine rows in 21,120 is small
enough to be invisible and large enough to be the difference between two arms.
An ablation whose arms saw different data measures the data.

Each arm syncs to Drive the moment it finishes. Sixteen frozen folds is about
ninety minutes on a T4 and four arms is longer than a free session can be
relied on, so an arm that completes must survive a disconnect during the next.
Arms already in Drive are skipped, which makes rerunning the notebook a resume.
"""
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("REPO", "/content/repo")
DATA = os.environ.get("DATAC", "/content/dataC")
OUT = os.environ.get("ABL_OUT",
                     "/content/drive/MyDrive/primates-sound-detection/ablations_2026-08-19")
FOLDS = ("IPA1ST,IPA2ST,IPA4ST,IPA6ST,IPA7ST,IPA8ST,IPA10ST,IPA11ST,"
         "IPA13ST,IPA14ST,IPA15ST,IPA16ST,IPA17ST,IPA18ST,IPA19ST,IPA20ST")

ARMS = [
    ("freqpos", ["--pooling", "temporal_freqpos"]),
    ("temporal", ["--pooling", "temporal"]),
    ("freq", ["--pooling", "temporal_freq"]),
    ("freqpos_noconfuser", ["--pooling", "temporal_freqpos",
                            "--drop-extra-confuser"]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(REPO)
    common = [
        "--folds", FOLDS, "--epochs", "15", "--patience", "3", "--overwrite",
        "--keep-all-background",
        "--manifest", os.path.join(DATA, "manifest.csv"),
        "--index", os.path.join(DATA, "v13_index.csv"),
        "--images", os.path.join(DATA, "v13_images.npy"),
        "--cache", os.path.join(DATA, "v13_features.npy"),
    ]

    done, failed = [], []
    for name, extra in ARMS:
        csv = f"/content/abl_{name}.csv"
        meta = f"/content/abl_{name}.run.json"
        landed = os.path.join(OUT, os.path.basename(csv))
        if os.path.exists(landed):
            print(f"\n=== {name}: already in Drive, skipping")
            done.append(name)
            continue

        heads = f"/content/heads_{name}"
        os.makedirs(heads, exist_ok=True)
        cmd = ([sys.executable, "scripts/train_v13_loso.py"] + common + extra
               + ["--out", csv, "--head-dir", heads, "--run-metadata", meta])
        print(f"\n{'=' * 62}")
        print(f"=== arm {name}   {' '.join(extra)}")
        print(f"{'=' * 62}", flush=True)

        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! arm {name} failed, moving on to the next")
            failed.append(name)
            continue

        shutil.copy(csv, landed)
        if os.path.exists(meta):
            shutil.copy(meta, os.path.join(OUT, os.path.basename(meta)))
        print(f"++ arm {name} synced to Drive")
        done.append(name)

    print(f"\n{len(done)} arms done, {len(failed)} failed: {failed or 'none'}")
    summarise(done)


def summarise(done):
    """Print the comparison the arms exist to make, once two of them exist."""
    import pandas as pd
    have = {}
    for name, _ in ARMS:
        p = os.path.join(OUT, f"abl_{name}.csv")
        if os.path.exists(p):
            have[name] = pd.read_csv(p)
    if len(have) < 2:
        print("\nfewer than two arms finished; nothing to compare yet")
        return

    print(f"\n{'arm':22s} {'folds':>6s} {'precision':>10s} {'recall':>8s}")
    for name, t in have.items():
        print(f"{name:22s} {len(t):6d} "
              f"{t['gated_loso_precision'].mean():10.4f} "
              f"{t['gated_loso_calls_retained'].mean():8.4f}")

    if "freqpos" not in have:
        return
    base = have["freqpos"]
    print("\nagainst freqpos, paired by station:")
    for name, t in have.items():
        if name == "freqpos":
            continue
        j = base[["station", "gated_loso_precision"]].merge(
            t[["station", "gated_loso_precision"]], on="station",
            suffixes=("_base", "_arm"))
        d = j["gated_loso_precision_base"] - j["gated_loso_precision_arm"]
        se = d.std(ddof=1) / (len(d) ** 0.5)
        print(f"  freqpos - {name:20s} mean {d.mean():+.4f}  "
              f"t {d.mean() / se if se else float('nan'):+.2f}  "
              f"freqpos better at {int((d > 0).sum())}/{len(d)}")
    print("\nA t below about 2 is noise. Report it as a null if it is one:")
    print("a head that does not pay for itself is worth knowing.")


if __name__ == "__main__":
    main()
