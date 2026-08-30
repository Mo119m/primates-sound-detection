"""Score the nine field-verified C. guereza roars against the shipped OOD gate.

The manuscript tells a reproducer to copy one fitted parameter: the per-class
percentile override that lets \\textit{Colobus\\_guereza} through the
out-of-distribution gate at the 97th percentile instead of the global 90th
(``config.OOD_GATE_PERCENTILE_BY_CLASS``). The reason given was that all nine
field-verified roars -- the entire field record for this species at these sites
-- pass at the 97th and do not all pass at the 90th.

That reason was true of the statistics it was fitted on and is not true of the
statistics the repository ships. ``data/outputs/ood_stats/fold_IPA4ST.npz`` was
rewritten under its own filename on 2026-08-20 when the class statistics were
refitted on the 2026-08-19 build, and the file it replaced -- Colobus p90 202.9,
p97 328.4, nine of nine roars admitted -- is gone. The shipped file gives p90
283.7 and p97 377.8, and admits one of the nine. Nineteen days passed with the
manuscript asserting the old file's behaviour in the present tense, because no
check in this repository ever scored the controls again.

This is that check. It writes one small CSV so the claim becomes a number the
verifier can recompute, rather than a sentence someone has to remember to
re-examine. Run it whenever ood_stats/ or the heads change:

    python scripts/score_colobus_controls.py

Two window conventions are reported because the answer depends on which one the
deployment uses, and being explicit is cheaper than being asked. ``loudest`` is
the crop ``build_v13_dataset`` applies when packing a reference clip; ``best``
is the most generous reading available, the minimum distance over every 2 s
window in the clip, which is what a sliding-window scanner effectively gets. If
the gate cannot admit the controls even under ``best``, no framing rescues it.
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(REPO, "src"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import auto_cleanup  # noqa: E402
import config  # noqa: E402
import model as model_module  # noqa: E402
import preprocessing  # noqa: E402

CLIP_DIR = os.path.join(REPO, "data/species/Colobus guereza field")
STATS_DIR = os.path.join(REPO, "data/outputs/ood_stats")
DEFAULT_OUT = os.path.join(REPO, "data/outputs/colobus_ood_controls.csv")

# Which head each shipped statistics file was fitted on. Kept explicit rather
# than derived: fold_IPA20ST.npz points at an older run directory than the rest,
# and a mapping that silently guessed would hide that.
HEADS = {
    "fold_IPA1ST": "models_full_2026-08-19/fold_IPA1ST.h5",
    "fold_IPA2ST": "models_full_2026-08-19/fold_IPA2ST.h5",
    "fold_IPA4ST": "models_full_2026-08-19/fold_IPA4ST.h5",
    "fold_IPA19ST": "models_full_2026-08-19/fold_IPA19ST.h5",
    "fold_IPA20ST": "models/fold_IPA20ST.h5",
}

WIN = int(round(config.WINDOW_SIZE * config.SAMPLE_RATE))
HOP = 441  # 10 ms, dense enough that the best window is not missed by luck


def loudest_window(y):
    """The crop build_v13_dataset applies: highest short-time energy, 2 s."""
    if len(y) <= WIN:
        return np.pad(y, (0, WIN - len(y)))
    e = np.cumsum(np.concatenate([[0.0], y.astype(np.float64) ** 2]))
    best = max(range(0, len(y) - WIN + 1, HOP), key=lambda a: e[a + WIN] - e[a])
    return y[best:best + WIN]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    clips = sorted(glob.glob(os.path.join(CLIP_DIR, "*.wav")))
    if not clips:
        sys.exit(f"no control clips under {CLIP_DIR}")
    print(f"  {len(clips)} field-verified control clips")

    prepared = []
    for p in clips:
        y, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
        windows = [y[a:a + WIN] for a in range(0, len(y) - WIN + 1, HOP)]
        prepared.append((loudest_window(y), windows or [loudest_window(y)]))

    rows = []
    for name, rel in HEADS.items():
        npz = os.path.join(STATS_DIR, name + ".npz")
        head = os.path.join(REPO, "data/outputs", rel)
        if not (os.path.exists(npz) and os.path.exists(head)):
            print(f"  skip {name}: statistics or head missing")
            continue
        z = np.load(npz, allow_pickle=False)
        names = [str(s) for s in z["class_names"]]
        pcts = [int(x) for x in z["percentiles"]]
        ci = names.index("Colobus_guereza")
        cut = {q: float(z["cutoffs"][ci][pcts.index(q)]) for q in (90, 97, 99)}

        mdl = model_module.load_trained_model(head)
        fe = auto_cleanup.build_feature_extractor(mdl, config.OOD_FEATURE_LAYER)
        mean, inv = z["class_means"][ci], z["inv_cov"]

        loud_d, best_d = [], []
        for loud, windows in prepared:
            X = np.stack([
                preprocessing.preprocess_for_model(preprocessing.preprocess_audio(s))
                for s in [loud] + windows])
            f = fe.predict(X, batch_size=16, verbose=0)
            d = f - mean
            dist = np.einsum("ij,jk,ik->i", d, inv, d)
            loud_d.append(float(dist[0]))
            best_d.append(float(dist[1:].min()))
        loud_d, best_d = np.array(loud_d), np.array(best_d)

        for conv, dist in (("loudest", loud_d), ("best", best_d)):
            rows.append({
                "stats": name,
                "convention": conv,
                "n_controls": len(clips),
                "p90": round(cut[90], 1),
                "p97": round(cut[97], 1),
                "p99": round(cut[99], 1),
                "pass_p90": int((dist <= cut[90]).sum()),
                "pass_p97": int((dist <= cut[97]).sum()),
                "pass_p99": int((dist <= cut[99]).sum()),
                "min_dist": round(float(dist.min()), 1),
                "max_dist": round(float(dist.max()), 1),
            })
        print(f"  {name:14s} p90 {cut[90]:6.1f} p97 {cut[97]:6.1f} | "
              f"best-window pass p90 {(best_d <= cut[90]).sum()}/{len(clips)} "
              f"p97 {(best_d <= cut[97]).sum()}/{len(clips)}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n  wrote {os.path.relpath(args.out, REPO)}")
    best = out[out.convention == "best"]
    print(f"  best any head at p97: {best.pass_p97.max()}/{len(clips)}; "
          f"on the fitted head (fold_IPA4ST): "
          f"{int(best[best.stats == 'fold_IPA4ST'].pass_p97.iloc[0])}/{len(clips)}")


if __name__ == "__main__":
    main()
