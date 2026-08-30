"""Second draws of the arms the paper had to call "measured once".

The sweep's conclusion currently rests on a sentence we could not close:

    Blocks 3+4's +0.0109 (t = +2.45) and the pogonias-free variant's +0.0102
    (t = +1.92), each measured once, cannot be distinguished from the luck of
    which run happened to be analysed.

That sentence exists because on 2026-08-29 the run-to-run floor stopped being a
guess. Two independent unseeded draws of the frozen specification -- same index
sha, same folds, same seed, a code diff between them touching only the
evaluation mask -- differ by +0.0035 in the paired sixteen-station mean at
t = +1.34, with one station moving 0.0256. A comparison of one specification
against ITSELF produced most of a conventional significance threshold. Against
that, a single run of block34 at t = +2.45 is suggestive and nothing more.

The only thing that settles it is a second draw of each arm, which is what this
runs. Three arms, cheapest first, because a session that dies after two hours
should leave completed work rather than three partial sweeps:

    nopogonias_rep2   frozen trunk, cached features, minutes per fold
    frozen_rep3       a third draw of the floor itself, cached features
    block34_rep2      the decisive one, and the only one that needs the GPU

frozen_rep3 is worth its slot for a reason that is not obvious. The floor is
currently one realisation: two draws give one paired difference, and a single
paired difference has no spread of its own. A third draw turns +0.0035 from a
point into a range, and it does so on a different machine from the first two,
which is the variation a reader who reruns this on their own hardware will
actually meet. Report it as such -- platform is confounded with draw here, and
that makes it an upper bound on pure run-to-run noise, not a purer estimate.

Everything about interruption, atomic sync, resume, and the fold-level
usability screen is inherited from run_unfreeze_arms rather than reimplemented.
Each of those rules is there because it was learned the expensive way -- the
first version of that runner lost thirteen trained folds to a dead session, and
a later one kept three folds that had trained fine and come out missing every
column the comparison reads.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_unfreeze_arms as R  # noqa: E402

OUT = os.environ.get(
    "REP_OUT",
    "/content/drive/MyDrive/primates-sound-detection/replicates_2026-08-29")

ARMS = [
    # Cheap, and it settles the arm whose recall cost is the only effect in the
    # whole sweep that clears the floor. If -0.0462 calls kept reproduces, that
    # finding is safe; if it does not, the sentence recommending the class be
    # kept has to be rewritten.
    ("nopogonias_rep2", ["--drop-pogonias"]),
    # A third draw of the specification the floor is measured on.
    ("frozen_rep3", []),
    # The decisive one. block34 is the paper's only nominally significant
    # backbone effect and its whole claim to being more than noise rests on one
    # unseeded run. ~25 min/fold on a T4; safe to interrupt.
    ("block34_rep2", ["--unfreeze", "2", "--finetune-epochs", "5",
                      "--finetune-lr", "1e-5"]),
]


def summarise():
    """Each replicate against frozen, beside the original draw of the same arm.

    The number to read is not whether a replicate is significant. It is whether
    the two draws of one arm agree, because that is what decides whether the
    original was measuring the model or measuring the draw.
    """
    import numpy as np
    import pandas as pd

    local = os.environ.get("LOCAL_RUNS", "/content/repo/data/outputs/v13_runs")
    frozen_csv = os.path.join(
        local, "full_2026-08-19/loso16_freqpos_evalfix.csv")
    if not os.path.exists(frozen_csv):
        print("no local frozen arm to pair against; nothing to summarise")
        return
    frozen = pd.read_csv(frozen_csv).set_index("station")

    originals = {
        "nopogonias_rep2": os.path.join(
            local, "nopogonias_fixed_2026-08-29/loso16_nopogonias.csv"),
        "block34_rep2": os.path.join(
            local, "unfreeze_2026-08-21_drive/block34_loso16.csv"),
        "frozen_rep3": os.path.join(
            local, "full_2026-08-19/loso16_freqpos_replicate.csv"),
    }

    def load(arm_dir):
        rows = []
        for f in sorted(os.listdir(arm_dir)):
            if f.endswith(".csv") and R.unusable(os.path.join(arm_dir, f)) is None:
                rows.append(pd.read_csv(os.path.join(arm_dir, f)))
        return pd.concat(rows).set_index("station") if rows else None

    def paired(t, col="gated_loso_precision"):
        sts = [s for s in t.index if s in frozen.index]
        d = np.array([t.loc[s, col] - frozen.loc[s, col] for s in sts])
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
        return len(d), d.mean(), (d.mean() / se if se else 0.0)

    print(f"\n{'arm':18s} {'n':>3s} {'d_prec':>8s} {'t':>7s}   "
          f"{'d_kept':>8s} {'t':>7s}")
    for name, _ in ARMS:
        arm_dir = os.path.join(OUT, name)
        if not os.path.isdir(arm_dir):
            continue
        t = load(arm_dir)
        if t is None:
            print(f"{name:18s}  no usable folds yet")
            continue
        n, mp, tp = paired(t)
        _, mk, tk = paired(t, "gated_loso_calls_retained")
        print(f"{name:18s} {n:3d} {mp:+8.4f} {tp:+7.2f}   {mk:+8.4f} {tk:+7.2f}")
        orig = originals.get(name)
        if orig and os.path.exists(orig):
            o = pd.read_csv(orig).set_index("station")
            n2, mp2, tp2 = paired(o)
            _, mk2, tk2 = paired(o, "gated_loso_calls_retained")
            print(f"{'  (first draw)':18s} {n2:3d} {mp2:+8.4f} {tp2:+7.2f}   "
                  f"{mk2:+8.4f} {tk2:+7.2f}")
            if n == len(R.STATIONS):
                print(f"{'  draw-to-draw':18s}     {mp - mp2:+8.4f} in the "
                      f"paired mean; the measured floor is 0.0035")
    print("\nRead the draw-to-draw line, not the significance. An arm whose two"
          "\ndraws differ by as much as the arm differs from frozen was never"
          "\nmeasuring the model.")


if __name__ == "__main__":
    R.OUT = OUT
    R.ARMS = ARMS
    # main() ends by calling summarise() through the module global, and the
    # inherited one looks for an arm literally named "frozen" in OUT. There
    # isn't one here, so it would print "no frozen arm to compare against yet"
    # after a six-hour sweep. Point it at the version that pairs each replicate
    # against its own first draw, which is the comparison this run exists for.
    R.summarise = summarise
    R.main()
