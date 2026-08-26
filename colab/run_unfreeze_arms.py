"""The two fine-tuned arms, one fold at a time, syncing after each.

Two changes from the first version, both because it lost thirteen folds.

It ran three arms and synced after each *arm*. A session died on the thirteenth
fold of the first arm and every one of those folds went with it, although each
had been trained and its head written to local disk. Sync is now per fold: a
death costs the fold in flight and nothing before it, and a rerun picks up where
it stopped.

And the frozen arm is not here. It trains on cached features, which is a hundred
minutes on this project's CPU -- loso16_freqpos.csv on the current dataset
already exists and was produced that way. Spending a GPU session on it and then
losing it was the actual cost of the first attempt. The GPU is for the two arms
that cannot run locally: unfreezing reads the image pack instead of the cache
and is roughly eight times slower per epoch.

The comparison the arms exist for is still against frozen, and summarise() reads
that from the local run rather than retraining it, so nothing is lost by
skipping it here.

Why this experiment is built the way it is: the existing three-fold answer says
frozen 0.6992, block4 0.9416, block34 0.9790, and per station it says 0.959 /
0.965 / 0.987 at IPA13ST and 0.969 / 0.971 / 0.970 at IPA20ST -- no difference.
The entire spread is IPA4ST, where the frozen arm scores 0.1693. That station
has 100 calls in 2,470 detections, a 4.0 % base rate at which precision is a
knife edge, so 0.1693 is a fitted threshold landing on the wrong side of it
rather than a model that failed. Hence sixteen folds, paired differences per
station rather than a macro mean, and every comparison printed again with IPA4ST
removed.
"""
import json
import os
import shutil
import subprocess
import sys

REPO = os.environ.get("REPO", "/content/repo")
DATA = os.environ.get("DATAC", "/content/dataF")
OUT = os.environ.get("UNF_OUT",
                     "/content/drive/MyDrive/primates-sound-detection/unfreeze_2026-08-21")
STATIONS = ["IPA1ST", "IPA2ST", "IPA4ST", "IPA6ST", "IPA7ST", "IPA8ST",
            "IPA10ST", "IPA11ST", "IPA13ST", "IPA14ST", "IPA15ST", "IPA16ST",
            "IPA17ST", "IPA18ST", "IPA19ST", "IPA20ST"]

ARMS = [
    ("block4", ["--unfreeze", "1", "--finetune-epochs", "5",
                "--finetune-lr", "1e-5"]),
    ("block34", ["--unfreeze", "2", "--finetune-epochs", "5",
                 "--finetune-lr", "1e-5"]),
    # Not a fine-tuning arm. The paper reports a null on dropping the
    # C. pogonias class, but that null was last measured on 8 August against a
    # dataset with no field pogonias in it, so it cannot speak to the 27 field
    # clips the current dataset carries. This remeasures it where the claim
    # lives. Frozen trunk, cached features: minutes per fold, not twenty-five.
    ("nopogonias", ["--drop-pogonias"]),
]


def _atomic_copy(src, dst):
    """Copy so the destination is never half-written.

    Drive is a network filesystem behind a FUSE mount and a copy into it is not
    instantaneous. A session killed mid-copy -- a closed lid, a lost tab, an
    exhausted quota -- leaves a partial file whose presence the resume rule
    would read as a finished fold. Writing beside it and renaming makes the
    destination appear whole or not at all.
    """
    tmp = dst + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


def _sweep_partials(arm_dir):
    """Remove .part files a previous session left behind."""
    n = 0
    for f in os.listdir(arm_dir):
        if f.endswith(".part"):
            os.remove(os.path.join(arm_dir, f))
            n += 1
    if n:
        print(f"  cleared {n} partial file(s) from an interrupted session")


GATED = ["gated_loso_threshold", "gated_loso_calls_retained",
         "gated_loso_fps_removed", "gated_loso_precision"]


def unusable(csv):
    """Why this synced fold cannot be used, or None if it can.

    Existing-on-disk was the only test until 2026-08-25, and it passed three
    folds that were missing every column the comparison reads. They had been
    trained on a runtime with no copy of the review table, so the time gate had
    no clock; train_v13_loso.py printed one line about it and wrote the CSV
    anyway. Twenty-five GPU-minutes each, and the run looked like progress.

    A fold that exists but cannot be compared is worse than a fold that does
    not exist, because the skip rule protects it. So the skip rule checks.
    """
    import pandas as pd
    if not os.path.exists(csv):
        return "missing"
    try:
        d = pd.read_csv(csv)
    except Exception as e:
        return f"unreadable ({e})"
    gone = [c for c in GATED if c not in d.columns]
    if gone:
        return f"missing {len(gone)} gated columns"
    if d[GATED].isna().all().any():
        return "gated columns present but empty"
    # A copy cut inside the final field leaves a value that still parses.
    # Precisions and rates are proportions; a threshold is a probability.
    # Anything outside [0, 1] is a truncation artefact, not a result.
    for c in GATED:
        v = d[c].dropna()
        if len(v) and ((v < 0).any() or (v > 1).any()):
            return f"{c} outside [0,1] -- the file looks truncated"
    # One human verdict, one evaluation row. Before 2026-08-25 the evaluation
    # mask did not restrict to aug == 0, so the sixteen variant rows of every
    # reviewed clip relabelled C_pogonias were counted as detections: IPA2ST
    # scored 144 where the review holds 80 originals, 44 percent of its pool
    # being copies of four sounds. A fold trained in the window between the
    # gate-table fix and the evaluation fix has all its columns and is still
    # wrong, so the count is checked against the index this runner trains from.
    exp = _expected_counts()
    if not exp:
        # No index means no way to tell a clean fold from a contaminated one.
        # Returning None here would silently disable screening and protect
        # every bad fold already in Drive, which is the failure this function
        # exists to prevent. Refuse instead: the caller redoes the fold, which
        # costs GPU minutes, and a wasted fold is cheaper than a kept lie.
        return "cannot verify: no index to compare the evaluation pool against"
    # Every row, not row zero. A per-fold CSV normally holds one row, but the
    # local sweep writes all completed folds into one file and a resumed run
    # can append -- checking only the first row would pass a file whose later
    # rows are contaminated.
    for _, r in d.iterrows():
        try:
            st = str(r["station"])
            got = int(r["detections"])
        except Exception as e:
            return f"unreadable station/detections ({e})"
        want = exp.get(st)
        if want is not None and got != want:
            return (f"{st}: evaluation pool is {got} rows where the review "
                    f"holds {want} originals -- trained before the aug==0 fix")
    return None


_EXPECTED = None


def _expected_counts():
    """Reviewed originals per station, from the index this runner trains from."""
    global _EXPECTED
    if _EXPECTED is None:
        import pandas as pd
        idx_path = os.path.join(DATA, "v13_index.csv")
        if not os.path.exists(idx_path):
            _EXPECTED = {}
        else:
            idx = pd.read_csv(idx_path)
            m = idx["source"].astype(str).str.startswith("review") & (idx["aug"] == 0)
            _EXPECTED = idx[m].groupby("station").size().to_dict()
    return _EXPECTED


def main():
    os.makedirs(OUT, exist_ok=True)
    os.chdir(REPO)
    common = [
        "--epochs", "15", "--patience", "3", "--overwrite",
        "--keep-all-background", "--pooling", "temporal_freqpos",
        "--manifest", os.path.join(DATA, "manifest.csv"),
        "--index", os.path.join(DATA, "v13_index.csv"),
        "--images", os.path.join(DATA, "v13_images.npy"),
        "--cache", os.path.join(DATA, "v13_features.npy"),
    ]

    for name, extra in ARMS:
        arm_dir = os.path.join(OUT, name)
        os.makedirs(arm_dir, exist_ok=True)
        heads = f"/content/heads_unf_{name}"
        os.makedirs(heads, exist_ok=True)
        _sweep_partials(arm_dir)
        done = sorted(f[:-4] for f in os.listdir(arm_dir) if f.endswith(".csv"))
        # Say what survives from previous sessions, per fold, before doing
        # anything. Interruption is the normal case here -- a closed lid, a
        # lost tab, an exhausted quota -- so the run should open by showing
        # exactly what it resumes from rather than leaving it to be inferred
        # from a fold count.
        usable, redo = [], []
        for st in STATIONS:
            why = unusable(os.path.join(arm_dir, f"{st}.csv"))
            (redo if why else usable).append((st, why))
        print(f"\n{'=' * 62}\n=== arm {name}   {' '.join(extra)}"
              f"\n=== {len(usable)} of {len(STATIONS)} folds usable in Drive"
              f"\n{'=' * 62}", flush=True)
        if usable:
            print("  keeping:  " + " ".join(st for st, _ in usable))
        stale = [(st, why) for st, why in redo if why != "missing"]
        if stale:
            print(f"  redoing {len(stale)} fold(s) already in Drive:")
            for st, why in stale:
                print(f"    {st}: {why}")
        todo = [st for st, _ in redo]
        eta = len(todo) * 25
        print(f"  to run:   {len(todo)} fold(s), about {eta // 60}h {eta % 60}m"
              f" on a T4")
        print("  safe to interrupt: each fold syncs to Drive as it finishes,"
              " and re-running\n  this cell picks up from whatever is already"
              " there.", flush=True)

        for st in STATIONS:
            landed = os.path.join(arm_dir, f"{st}.csv")
            why = unusable(landed)
            if os.path.exists(landed) and why is None:
                print(f"  skip {st}, already synced")
                continue
            if os.path.exists(landed):
                print(f"  redoing {st}: the synced fold is {why}")
            csv = f"/content/{name}_{st}.csv"
            meta = f"/content/{name}_{st}.run.json"
            cmd = ([sys.executable, "scripts/train_v13_loso.py",
                    "--folds", st] + common + extra
                   + ["--out", csv, "--head-dir", heads,
                      "--run-metadata", meta])
            print(f"\n--- {name} / {st}", flush=True)
            if subprocess.run(cmd).returncode != 0:
                print(f"!! {name}/{st} failed, moving on")
                continue
            # Sync immediately, and sync atomically. One fold in flight is the
            # most a dead session can cost, which is the whole point of doing
            # it here -- but a runtime that dies *during* the copy leaves a
            # truncated file in Drive, and a truncated CSV that still parses
            # would be skipped forever by the resume rule. Copy to a temp name
            # beside the target and rename, so the target either does not
            # exist or is complete.
            #
            # The heavy file goes first and the CSV last, because the CSV is
            # what resume keys on: if the session dies between them the fold
            # is simply redone, which costs 25 minutes and nothing else.
            hf = os.path.join(heads, f"head_{st}.weights.h5")
            if os.path.exists(hf):
                _atomic_copy(hf, os.path.join(arm_dir,
                                              f"head_{st}.weights.h5"))
            if os.path.exists(meta):
                _atomic_copy(meta, os.path.join(arm_dir, f"{st}.run.json"))
            _atomic_copy(csv, landed)
            print(f"++ {name}/{st} synced")

    summarise()


def _read_arm(arm_dir):
    """One arm's sixteen per-fold CSVs, concatenated."""
    import pandas as pd
    if not os.path.isdir(arm_dir):
        return None
    good, bad = [], []
    for f in sorted(os.listdir(arm_dir)):
        if not f.endswith(".csv"):
            continue
        full = os.path.join(arm_dir, f)
        why = unusable(full)
        (bad if why else good).append((f, why))
    if bad:
        print(f"  {arm_dir}: ignoring {len(bad)} unusable folds")
        for f, why in bad:
            print(f"    {f}: {why}")
    parts = [pd.read_csv(os.path.join(arm_dir, f)) for f, _ in good]
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).set_index("station")


def summarise(frozen_csv=None):
    """Compare the fine-tuned arms against the frozen one.

    The frozen arm is read from wherever it was produced -- normally the local
    sixteen-fold run -- rather than retrained here.
    """
    import numpy as np
    import pandas as pd

    have = {}
    for name, _ in ARMS:
        t = _read_arm(os.path.join(OUT, name))
        if t is not None:
            have[name] = t
    for cand in ([frozen_csv] if frozen_csv else []) + [
            os.path.join(OUT, "frozen.csv"),
            os.path.join(REPO, "data/outputs/v13_runs/full_2026-08-19/loso16_freqpos.csv")]:
        if cand and os.path.exists(cand):
            have["frozen"] = pd.read_csv(cand).set_index("station")
            break

    if not have:
        print("\nnothing to compare yet")
        return
    print(f"\n{'arm':10s} {'folds':>6s} {'precision':>10s} {'recall':>9s} "
          f"{'IPA4ST':>9s}")
    for k, t in have.items():
        i4 = t.loc["IPA4ST", "gated_loso_precision"] if "IPA4ST" in t.index else float("nan")
        print(f"{k:10s} {len(t):6d} {t['gated_loso_precision'].mean():10.4f} "
              f"{t['gated_loso_calls_retained'].mean():9.4f} {i4:9.4f}")

    if "frozen" not in have:
        print("\nno frozen arm to compare against yet")
        return
    base = have["frozen"]
    print("\nagainst frozen, paired by station:")
    for k, t in have.items():
        if k == "frozen":
            continue
        sts = [s for s in t.index if s in base.index]
        if len(sts) < 3:
            print(f"  {k}: only {len(sts)} shared folds, too few to pair")
            continue
        d = np.array([t.loc[s, "gated_loso_precision"]
                      - base.loc[s, "gated_loso_precision"] for s in sts])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {k:9s} n={len(d):2d}  mean {d.mean():+.4f}  "
              f"t {d.mean()/se if se else 0:+.2f}  better at {int((d>0).sum())}/{len(d)}")
        no4 = np.array([x for s, x in zip(sts, d) if s != "IPA4ST"])
        if len(no4) > 2:
            se4 = no4.std(ddof=1) / np.sqrt(len(no4))
            print(f"  {'':9s} without IPA4ST: {no4.mean():+.4f}  "
                  f"t {no4.mean()/se4 if se4 else 0:+.2f}")
    print("\nThe second line of each pair is the one to read first: the earlier"
          "\nthree-fold version of this experiment was carried entirely by"
          "\nIPA4ST, whose 4.0 % base rate makes precision a knife edge.")


if __name__ == "__main__":
    main()
