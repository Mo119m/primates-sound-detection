"""Three degrees of fine-tuning, at sixteen folds, on the current dataset.

The existing measurement of this says frozen 0.6992, block4 0.9416, block3+4
0.9790, and it should not be read. Those are three folds, and per station they
are 0.959 / 0.965 / 0.987 at IPA13ST and 0.969 / 0.971 / 0.970 at IPA20ST --
that is, no difference at all. The entire spread comes from IPA4ST, where the
frozen arm scores 0.1693.

IPA4ST is the station with 100 calls in 2,470 detections, a 4.0 % base rate, and
at that base rate precision is a knife edge: keeping 45 extra false positives
puts it at 0.69 and keeping 1,059 puts it at 0.085. The frozen arm's 0.1693 is
not a model that failed, it is a fitted threshold that landed on the wrong side
of that edge. The three numbers measure threshold instability under an extreme
base rate, which is the same thing the 2026-08-19 comparison found when nine
changed labels moved IPA17ST's fitted threshold from 0.063 to 0.928 while its
precision did not move at all.

So this is built to be unfoolable by that:

  sixteen folds     one station cannot carry a paired test over sixteen
  paired t per arm  reported instead of a macro mean, which is what let one
                    station carry the earlier result
  the scan decides  each arm's heads are assembled and run over raw audio, and
                    the detection output is what gets compared. A scan does not
                    depend on where a threshold falls within a fixed 4 %-base-
                    rate set, and this project's own standard is that the scan
                    is the only test it trusts

Arms are ordered frozen, block4, block3+4, so a session that dies partway still
holds the comparison that carries the claim.
"""
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("REPO", "/content/repo")
DATA = os.environ.get("DATAC", "/content/dataF")
OUT = os.environ.get("UNF_OUT",
                     "/content/drive/MyDrive/primates-sound-detection/unfreeze_2026-08-21")
FOLDS = ("IPA1ST,IPA2ST,IPA4ST,IPA6ST,IPA7ST,IPA8ST,IPA10ST,IPA11ST,"
         "IPA13ST,IPA14ST,IPA15ST,IPA16ST,IPA17ST,IPA18ST,IPA19ST,IPA20ST")

ARMS = [
    ("frozen", []),
    ("block4", ["--unfreeze", "1", "--finetune-epochs", "5",
                "--finetune-lr", "1e-5"]),
    ("block34", ["--unfreeze", "2", "--finetune-epochs", "5",
                 "--finetune-lr", "1e-5"]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(REPO)
    common = [
        "--folds", FOLDS, "--epochs", "15", "--patience", "3", "--overwrite",
        "--keep-all-background", "--pooling", "temporal_freqpos",
        "--manifest", os.path.join(DATA, "manifest.csv"),
        "--index", os.path.join(DATA, "v13_index.csv"),
        "--images", os.path.join(DATA, "v13_images.npy"),
        "--cache", os.path.join(DATA, "v13_features.npy"),
    ]

    done, failed = [], []
    for name, extra in ARMS:
        csv = f"/content/unf_{name}.csv"
        landed = os.path.join(OUT, os.path.basename(csv))
        if os.path.exists(landed):
            print(f"\n=== {name}: already in Drive, skipping")
            done.append(name)
            continue
        heads = f"/content/heads_unf_{name}"
        os.makedirs(heads, exist_ok=True)
        meta = f"/content/unf_{name}.run.json"
        cmd = ([sys.executable, "scripts/train_v13_loso.py"] + common + extra
               + ["--out", csv, "--head-dir", heads, "--run-metadata", meta])
        print(f"\n{'=' * 62}\n=== arm {name}   {' '.join(extra) or 'base frozen'}"
              f"\n{'=' * 62}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            print(f"!! arm {name} failed, continuing")
            failed.append(name)
            continue
        shutil.copy(csv, landed)
        if os.path.exists(meta):
            shutil.copy(meta, os.path.join(OUT, os.path.basename(meta)))
        d = os.path.join(OUT, f"heads_unf_{name}")
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(heads):
            shutil.copy(os.path.join(heads, f), os.path.join(d, f))
        print(f"++ {name} synced, {len(os.listdir(d))} head files")
        done.append(name)

    print(f"\n{len(done)} arms done, {len(failed)} failed: {failed or 'none'}")
    summarise()


def summarise():
    """Paired, per station, and with the station that fooled the last one shown."""
    import numpy as np
    import pandas as pd
    have = {}
    for name, _ in ARMS:
        p = os.path.join(OUT, f"unf_{name}.csv")
        if os.path.exists(p):
            have[name] = pd.read_csv(p).set_index("station")
    if len(have) < 2:
        print("\nfewer than two arms; nothing to compare")
        return
    sts = sorted(next(iter(have.values())).index)
    same = len({tuple(int(t.loc[s, "gated_detections"]) for s in sts)
                for t in have.values()}) == 1
    print(f"\nevaluation sets identical across arms: {same}")
    print(f"\n{'arm':10s} {'precision':>10s} {'recall':>9s}  "
          f"{'IPA4ST alone':>13s}")
    for k, t in have.items():
        i4 = (t.loc["IPA4ST", "gated_loso_precision"]
              if "IPA4ST" in t.index else float("nan"))
        print(f"{k:10s} {t['gated_loso_precision'].mean():10.4f} "
              f"{t['gated_loso_calls_retained'].mean():9.4f} {i4:13.4f}")

    if "frozen" not in have:
        return
    base = have["frozen"]
    print("\nagainst frozen, paired by station:")
    for k, t in have.items():
        if k == "frozen":
            continue
        d = np.array([t.loc[s, "gated_loso_precision"]
                      - base.loc[s, "gated_loso_precision"] for s in sts])
        se = d.std(ddof=1) / np.sqrt(len(d))
        d_no4 = np.array([x for s, x in zip(sts, d) if s != "IPA4ST"])
        se4 = d_no4.std(ddof=1) / np.sqrt(len(d_no4))
        print(f"  {k:9s} mean {d.mean():+.4f}  t {d.mean()/se:+.2f}  "
              f"better at {int((d > 0).sum())}/{len(d)}")
        print(f"  {'':9s} without IPA4ST: {d_no4.mean():+.4f}  "
              f"t {d_no4.mean()/se4:+.2f}")
    print("\nThe second line of each pair is the one to read first. The earlier"
          "\nthree-fold version of this experiment was carried entirely by"
          "\nIPA4ST, whose 4.0 % base rate makes precision a knife edge.")


if __name__ == "__main__":
    main()
