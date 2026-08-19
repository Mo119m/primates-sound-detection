"""Fail if anything a person called a target species is training as a negative.

This is the one labelling error more data cannot undo. A target call filed as
Background tells the model, with the full weight of a label, that the thing it
exists to find is not there, and every later epoch reinforces it. It has
happened twice here. Three IPA4ST calls sat in Background under a blanket rule
that sent every false positive at that station to the negative class, and two of
those turned out to be C. pogonias rather than the C. nictitans their filename
prefix implied. Both were found by an expert listening, not by any check.

So the check reads every place a human verdict is recorded, rather than the one
file nearest to hand:

    data/labels/relabel_2026-08-18.csv                the re-identifications
    data/labels/auto_flagged_fp_review_2026-08-18.csv the 3,143-clip audit
    data/outputs/auto_cleanup/cleanup_vs_review.csv   verdict == "call"
    <expert pogonias folder>                          optional, --pogonias-dir

Reading several matters more than it looks. Each was believed complete at some
point and each was wrong: the review table did not know about the three IPA4ST
calls, the relabel list first had two of them as the wrong species, and the
pogonias folder named clips no other file mentioned. A source that cannot be
found is reported as missing rather than passed over, because a check that
quietly loses a source is a check that quietly loses coverage.

Exit status is 1 if any target call is a trainable negative, so this can gate a
training run.
"""
import argparse
import os
import re
import sys

import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Classes a target call must never be sitting in. Colobus_confuser counts: it is
# a negative with a nicer name, and a genuine roar in it is the same mistake.
NEGATIVE = {"Background", "Colobus_confuser"}
# The expert re-exports clips with a counter suffix, sometimes twice over.
RE_EXPORT = re.compile(r"(_\d{5}_\d{3})+\.wav$", re.I)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=os.path.join(
        REPO, "data/outputs/v13_runs/clean_2026-08-17/v13_index.csv"),
        help="Packed index to check")
    ap.add_argument("--pogonias-dir", default=None,
                    help="Optional folder of clips an expert returned as "
                         "C. pogonias; every .wav in it counts as a verdict")
    args = ap.parse_args()

    import build_v13_dataset as B

    claims, sources, missing = {}, [], []

    def add(name, where, what):
        key = B._canon(RE_EXPORT.sub(".wav", str(name)))
        claims.setdefault(key, []).append((where, what))

    def read(path, label):
        if not os.path.exists(path):
            missing.append(f"{label} ({os.path.relpath(path, REPO)})")
            return None
        sources.append(label)
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    d = read(os.path.join(REPO, "data/labels/relabel_2026-08-18.csv"),
             "relabel list")
    if d is not None:
        for _, r in d.iterrows():
            add(r["file"], "relabel list", r["new_label"])

    d = read(os.path.join(
        REPO, "data/labels/auto_flagged_fp_review_2026-08-18.csv"),
        "auto_flagged_fp audit")
    if d is not None:
        for _, r in d[d["manual_id"] != "Noise"].iterrows():
            add(r["file"], "auto_flagged_fp audit", r["manual_id"])

    d = read(os.path.join(REPO, "data/outputs/auto_cleanup/cleanup_vs_review.csv"),
             "original review")
    if d is not None:
        for _, r in d[d["verdict"] == "call"].iterrows():
            add(r["file"], "original review", "C. nictitans call")

    if args.pogonias_dir and os.path.isdir(args.pogonias_dir):
        sources.append("expert pogonias folder")
        for f in os.listdir(args.pogonias_dir):
            if f.lower().endswith(".wav"):
                add(f, "expert pogonias folder", "C_pogonias")
    elif args.pogonias_dir:
        missing.append(f"pogonias folder ({args.pogonias_dir})")

    print(f"{len(claims)} clips carry a human species verdict, "
          f"from {len(sources)} sources: {', '.join(sources)}")
    for m in missing:
        print(f"  ! source not read: {m}")

    idx = pd.read_csv(args.index, keep_default_na=False)
    ok = (idx["ok"].astype(str).str.lower().isin(("true", "1"))
          if "ok" in idx.columns else pd.Series(True, index=idx.index))
    base = idx["path"].map(lambda p: B._canon(os.path.basename(str(p))))

    seen = sum(1 for b in base if b in claims)
    print(f"of those, {seen} are in {os.path.relpath(args.index, REPO)}")

    bad = [(base[i], idx.at[i, "label"], idx.at[i, "source"], claims[base[i]])
           for i in idx.index
           if base[i] in claims and idx.at[i, "label"] in NEGATIVE and ok[i]]

    if bad:
        print(f"\nFAIL: {len(bad)} target calls are trainable negatives")
        for name, lab, src, c in bad[:25]:
            print(f"  {lab:18s} {src:30s} {name[:48]}")
            for where, what in c:
                print(f"      {where} says: {what}")
        return 1

    print("\nno clip a person called a target species is a trainable negative")
    trained = idx[ok]
    print(f"\ntrainable rows {len(trained)} of {len(idx)}")
    for k, v in sorted(trained["label"].value_counts().items()):
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
