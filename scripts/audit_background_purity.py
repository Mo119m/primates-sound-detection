"""
Find target-species calls hiding in the Background class.

A call filed as a negative is the one training error that cannot be undone by
adding data: the model is told, with the full weight of a label, that the thing
it exists to find is not there. Every other kind of noise in the dataset costs
accuracy; this one costs the thing itself.

The Background class here is 26 323 clips from ten sources, and 65 % of them
were chosen by a bird detector rather than by a person. Nobody can listen to
26 323 clips. What can be done is ask the current detector which of them it
believes are calls, under the same grouped-argmax rule deployment uses, and then
put a human ear on only those.

The two failure modes are asymmetric and the handling reflects it. A genuine
call left in Background is unrecoverable. A hard negative removed from
Background costs a fraction of one class. So anything the model flags comes out,
and stays out until somebody listens to it; nothing is silently reassigned to a
positive class, because an unlistened clip moved the other way is the same
mistake with the sign flipped.

    python scripts/audit_background_purity.py --model data/outputs/models/fold_IPA4ST.h5
    python scripts/audit_background_purity.py --resume   # reuse existing scores
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SCORES = os.path.join(REPO, "data/outputs/background_scores.csv")
CLIPS = os.path.join(REPO, "data/outputs/background_suspects")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(
        REPO, "data/outputs/models/fold_IPA4ST.h5"))
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "data/outputs/v13_manifest.csv"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--export", type=int, default=150,
                    help="How many of the flagged clips to write out for a "
                         "person, highest score first.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse an existing score file and only score rows "
                         "missing from it.")
    args = ap.parse_args()

    import librosa
    import soundfile as sf
    import config
    import preprocessing
    import model as model_module
    import detection

    man = pd.read_csv(args.manifest)
    bg = man[man.label == "Background"].copy()
    print(f"{len(bg)} Background clips, {bg.source.nunique()} sources")

    done = {}
    if args.resume:
        for f in (SCORES, os.path.join(REPO, "data/outputs/birdnet_scores.csv")):
            if os.path.exists(f):
                d = pd.read_csv(f)
                done.update(dict(zip(d.path, d.target_score)))
        print(f"  reusing {len(done)} scores already on disk")

    todo = bg[~bg.path.isin(done)]
    print(f"  scoring {len(todo)}")

    if len(todo):
        mdl = model_module.load_trained_model(args.model)
        n_out = int(mdl.output_shape[-1])
        if n_out != len(config.CLASS_NAMES):
            sys.exit(f"model has {n_out} outputs, config lists "
                     f"{len(config.CLASS_NAMES)}; refusing to read it "
                     f"positionally")
        labels, indices = detection.get_detection_groups()
        tgt = [i for i, n in enumerate(labels) if n != "Background"]
        print(f"  target groups: {[labels[i] for i in tgt]}")

        paths = todo.path.tolist()
        B = 128
        for lo in range(0, len(paths), B):
            chunk = paths[lo:lo + B]
            X, keep = [], []
            for p in chunk:
                try:
                    y, _ = librosa.load(p, sr=config.SAMPLE_RATE, mono=True)
                except Exception:
                    done[p] = np.nan
                    continue
                n = int(config.WINDOW_SIZE * config.SAMPLE_RATE)
                y = y[:n] if y.size >= n else np.pad(y, (0, n - y.size))
                X.append(preprocessing.preprocess_for_model(
                    preprocessing.preprocess_audio(y)))
                keep.append(p)
            if X:
                pr = mdl.predict(np.stack(X), batch_size=64, verbose=0)
                for p, row in zip(keep, pr):
                    g = detection.group_probabilities(row, labels, indices)
                    done[p] = float(g[tgt].max())
            print(f"\r  {min(lo + B, len(paths))}/{len(paths)}",
                  end="", flush=True)
        print()

    bg["target_score"] = bg.path.map(done)
    bg[["path", "source", "target_score"]].to_csv(SCORES, index=False)
    ok = bg[bg.target_score.notna()]
    print(f"\nscored {len(ok)} of {len(bg)}")

    print("\nscore distribution, whole class:")
    for q in (50, 90, 99, 99.9, 100):
        print(f"  p{q:<5} {np.percentile(ok.target_score, q):.4f}")

    flagged = ok[ok.target_score >= args.threshold]
    print(f"\nflagged at >= {args.threshold}: {len(flagged)} "
          f"({100 * len(flagged) / len(ok):.2f}%)")
    print("\nby source:")
    tab = (ok.assign(flag=ok.target_score >= args.threshold)
             .groupby(ok.source.str.replace(r"birdnet:.*", "birdnet(all)",
                                            regex=True))
             .agg(n=("flag", "size"), flagged=("flag", "sum")))
    tab["rate"] = (100 * tab.flagged / tab.n).round(2)
    print(tab.sort_values("flagged", ascending=False).to_string())

    if not len(flagged):
        print("\nNothing to review.")
        return

    shutil.rmtree(CLIPS, ignore_errors=True)
    os.makedirs(CLIPS, exist_ok=True)
    for r in flagged.nlargest(args.export, "target_score").itertuples():
        try:
            y, _ = librosa.load(r.path, sr=config.SAMPLE_RATE, mono=True)
        except Exception:
            continue
        src = re.sub(r"[^A-Za-z0-9_.-]", "_", str(r.source))[:28]
        stem = os.path.splitext(os.path.basename(r.path))[0][:52]
        sf.write(os.path.join(
            CLIPS, f"score{r.target_score:.3f}__{src}__{stem}.wav"),
            y, config.SAMPLE_RATE)
    print(f"\nwrote {len(os.listdir(CLIPS))} clips to "
          f"{os.path.relpath(CLIPS, REPO)}")
    print(f"Rebuild with: python scripts/build_v13_dataset.py "
          f"--drop-call-like {args.threshold}")


if __name__ == "__main__":
    main()
